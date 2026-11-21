"""mechfight bot rebuilt from scratch (GUIDELINES.md §3). Every decision runs the same layers in order:

    perception & memory -> assessment -> decision -> execution -> action

This file is the frame (the per-decision clock, the log line) and the wiring. The layers live in their own modules:
world.py and memory.py (perception), nav.py and gunnery.py (execution). M2: the decision is a stand-in, the nearest
mech in sight is fought with everything.
"""
import math
import sys
import time
import traceback

import mechfight as mf
from gunnery import Gunner, Target
from memory import Memory, wrap, yaw_to
from nav import Legs
from world import World
from zone import Zone

NAME = "scratch"
TICK_CAP = 0.025   # s we allow ourselves per decision (the server's budget is 50 ms)
PATH_CAP = 0.010   # s of A* per decision
TURN_TO_UNSEEN_S = 0.6  # s we face the bearing of a hit from someone we can't see


class Clock:
    """Wall time left in this decision."""

    def __init__(self, cap: float):
        self.t0, self.cap = time.perf_counter(), cap

    def left(self) -> float:
        return self.cap - (time.perf_counter() - self.t0)

    def deadline(self, most: float) -> float:
        return min(time.perf_counter() + most, self.t0 + self.cap - 0.004)

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000


def log(line: str) -> None:
    sys.stderr.write(line + "\n")


class Bot:
    name = NAME

    def __init__(self, hello):
        t0 = time.perf_counter()
        self.cfg = cfg = hello["config"]
        try:
            self.world = World(hello["map"]["dir"], cfg["mech"])
        except Exception:  # no map: play on without walls (straight walks, every line of fire taken as clear)
            log(traceback.format_exc())
            self.world = None
        self.mem = Memory(cfg, self.world)
        self.zone = Zone()
        self.legs = Legs(self.world, hello.get("seed", 0))
        self.gun = Gunner(cfg, self.world)
        self.reach = cfg["loot"]["pickup_radius"]
        log(f"hello: world {'ok' if self.world else 'MISSING'} in {time.perf_counter() - t0:.2f} s")

    # ---------- per decision ----------

    def act(self, obs):
        clock = Clock(TICK_CAP)
        t, me = obs["t"], obs["self"]
        self.mem.update(obs)
        self.zone.update(obs["zone"], t)
        tgt = self.target(obs, me, t)
        aim, fire, shots = self.gun.plan(me, t, tgt, t < self.mem.short_until)
        act = {"aim": aim or self.look(me, t), "fire": fire}
        act["move"], act["jump"], act["boost"], mode = self.legs_for(obs, me, t, tgt, clock)
        self.weapons(obs, me, act)
        if obs.get("offer"):
            act["pick"] = 0
        gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
        if tgt is None and gun and 0 < gun.get("ammo", 0) < gun["stats"]["mag"] and gun.get("reload", 0) == 0:
            act["reload"] = True
        mf.draw.label(mode)
        x, y = me["pos"][0], me["pos"][1]
        log(f"t={t:.1f} m={mode} hp={me['hp']:.0f} p={x:.0f},{y:.0f} tg={tgt.track.id if tgt and tgt.track else '-'} "
            f"f={shots} gh={len(self.mem.ghosts)} ms={clock.ms():.1f}")
        return act

    # ---------- decision (M2 stand-in) ----------

    def target(self, obs, me, t):
        best, bd = None, 1e9
        mz = (me["pos"][0], me["pos"][1], me["pos"][2] + me["stats"].get("muzzle_height", 7.0))
        for a in obs["mechs"]:
            if a["kind"] != "mech":
                continue
            tr = self.mem.tracks[a["id"]]
            d = math.dist(mz, tr.pos)
            if d < bd and (self.world is None or self.world.clear(mz, (tr.pos[0], tr.pos[1], tr.pos[2] + 4.5 * tr.size))):
                best, bd = tr, d
        if best is not None:
            return Target(best.pos, 2.5 * best.size, 9.0 * best.size, ("laser", "gun", "lightning"), best)
        return None

    def look(self, me, t):
        hit = self.mem.unseen_hit
        if hit and t - hit[0] < TURN_TO_UNSEEN_S:
            return [hit[1], 0.0]
        c = self.zone.next_circle()
        return [yaw_to(me["pos"], c), 0.0]

    def safe(self, me, t):
        """Is a step to p safe: open ground in the map, not off a ledge, inside the zone a few seconds from now?"""
        floor = me["pos"][2]

        def ok(p):
            if self.world is not None:
                if not self.world.passable(p[0], p[1]) or self.world.floor_at(p[0], p[1]) < floor - 2.0:
                    return False
            return self.zone.inside(p, t + 3.0, 3.0)
        return ok

    def legs_for(self, obs, me, t, tgt, clock):
        safe = self.safe(me, t)
        x, y = me["pos"][0], me["pos"][1]
        cx, cy, r = self.zone.next_circle()
        goal, climb, mode = None, False, "hold"
        if not self.zone.inside((x, y), t + 5.0, 5.0) or math.hypot(x - cx, y - cy) > r:
            goal, climb = self.zone_spot(me, t, (cx, cy, r))
            mode = "climb" if climb else "zone"
        move = (0.0, 0.0)
        if goal is not None:
            move = self.legs.toward(me, goal, t, clock.deadline(PATH_CAP), climb=climb)
        if tgt is not None and tgt.track is not None and tgt.track.kind == "mech":
            s = self.legs.strafe(me, yaw_to(me["pos"], tgt.pos), t, safe)
            move = (move[0] + s[0], move[1] + s[1])
            mode = "fight" if mode == "hold" else mode
        d = self.legs.dodge(me, obs["projectiles"], self.cfg["weapons"]["mortar"]["gravity"], safe)
        if d is not None:
            move = d
        n = math.hypot(*move)
        if n > 1:
            move = (move[0] / n, move[1] / n)
        move, jump, boost = self.legs.unstick(me, move, t)
        if mode == "zone" and not me["in_zone"] and me["energy"] > 30 and me["grounded"]:
            boost = True
        return [round(move[0], 3), round(move[1], 3)], jump, boost, mode

    def zone_spot(self, me, t, circle):
        """(goal, climb): the nearest point inside the circle we can walk to, or its centre to climb to (cached 1 s)."""
        cx, cy, r = circle
        key = (round(cx), round(cy), round(r), int(t))
        if getattr(self, "_spot_key", None) != key:
            self._spot_key = key
            inner = max(r - min(15.0, r / 2), 0.7 * r)
            x, y = me["pos"][0], me["pos"][1]
            spot = self.world.spot_in((x, y), cx, cy, inner, self.world.area(x, y)) if self.world else (cx, cy)
            self._spot = (spot, False) if spot else ((cx, cy), True)
        return self._spot

    def weapons(self, obs, me, act):
        """Drop a mortar at once (a laser cooks it off); take a laser, or a lightning while the slot is empty."""
        ws = me["weapons"]
        for i, w in enumerate(ws):
            if w and w["kind"] == "mortar":
                act["drop"] = i
                return
        kinds = {w["kind"]: i for i, w in enumerate(ws) if w}
        x, y, z = me["pos"]
        for p in obs["pickups"]:
            if math.hypot(p["pos"][0] - x, p["pos"][1] - y) > self.reach - 0.3 or abs(p["pos"][2] - z) > 3.8:
                continue
            if p["kind"] not in ("laser", "lightning") or p["kind"] in kinds:
                continue
            if None in ws:
                act["take"] = p["id"]
                return
            if p["kind"] == "laser" and "lightning" in kinds:
                act["take"], act["drop"] = p["id"], kinds["lightning"]
                return


if __name__ == "__main__":
    mf.run(Bot)
