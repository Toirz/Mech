"""mechfight bot rebuilt from scratch (GUIDELINES.md §3). Every decision runs the same layers in order:

    perception & memory -> assessment -> decision -> execution -> action
    world, memory          assess          decide      nav, gunnery

This file is the frame (the per-decision clock, one log line per decision to stderr) and the wiring.
"""
import math
import sys
import time
import traceback

import mechfight as mf
from assess import Assess
from decide import Brain
from gunnery import Gunner
from memory import Memory
from nav import Legs, unit
from world import World
from zone import Zone

NAME = "scratch"
TICK_CAP = 0.025   # s we allow ourselves per decision (the server's budget is 50 ms)
PATH_CAP = 0.010   # s of A* per decision


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
        self.assess = Assess(cfg, self.world, self.mem)
        self.brain = Brain(cfg, self.world, self.mem, self.zone, self.assess)
        self.legs = Legs(self.world, hello.get("seed", 0))
        self.gun = Gunner(cfg, self.world)
        self.reach = cfg["loot"]["pickup_radius"]
        self.calm_since = 0.0
        log(f"hello: world {'ok' if self.world else 'MISSING'} in {time.perf_counter() - t0:.2f} s")

    def act(self, obs):
        clock = Clock(TICK_CAP)
        t, me = obs["t"], obs["self"]
        self.mem.update(obs)
        self.zone.update(obs["zone"], t)
        plan = self.brain.decide(obs, me, t, clock)
        aim, fire, shots = self.gun.plan(me, t, plan.tgt, t < self.mem.short_until)
        act = {"aim": aim or plan.face or me["aim"], "fire": fire}
        act["move"], act["jump"], act["boost"] = self.legs_for(obs, me, t, plan, clock)
        self.weapons(obs, me, act)
        if obs.get("offer"):
            act["pick"] = Brain.pick_upgrade(obs["offer"], me)
        if plan.tgt is not None:
            self.calm_since = t
        gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
        if t - self.calm_since > 2.0 and gun and 0 < gun.get("ammo", 0) < gun["stats"]["mag"] and gun.get("reload", 0) == 0:
            act["reload"] = True
        mf.draw.label(plan.mode)
        x, y = me["pos"][0], me["pos"][1]
        tg = plan.tgt.track.id if plan.tgt is not None and plan.tgt.track is not None else ("crate" if plan.tgt else "-")
        g = f"{plan.goal[0]:.0f},{plan.goal[1]:.0f}{'^' if plan.climb else ''}" if plan.goal is not None else "-"
        zx, zy, zr = self.zone.circle(t)
        log(f"t={t:.1f} m={plan.mode} hp={me['hp']:.0f} p={x:.0f},{y:.0f} tg={tg} {plan.note} f={shots} "
            f"g={g} z={zx:.0f},{zy:.0f},{zr:.0f} cu={me['cubes']} gh={len(self.mem.ghosts)} ms={clock.ms():.1f}")
        return act

    def safe(self, me, t):
        """Is a step to p safe: open ground in the map, not off a ledge, inside the zone a few seconds from now?"""
        floor = me["pos"][2]

        def ok(p):
            if self.world is not None:
                if not self.world.passable(p[0], p[1]) or self.world.floor_at(p[0], p[1]) < floor - 2.0:
                    return False
            return self.zone.inside(p, t + 3.0, 3.0)
        return ok

    def legs_for(self, obs, me, t, plan, clock):
        safe = self.safe(me, t)
        x, y = me["pos"][0], me["pos"][1]
        move = (0.0, 0.0)
        if plan.goal is not None:
            zx, zy, zr = self.zone.circle(t + 5.0)  # paths go round buildings on the inside of the zone
            move = self.legs.toward(me, plan.goal, t, clock.deadline(PATH_CAP), plan.arrive, plan.climb, (zx, zy, zr - 3.0))
        else:
            self.legs.climb_to = None
        if plan.wiggle is not None:  # in a drop's ring: circle slowly round its centre, never stand still
            cx, cy = plan.wiggle
            tx, ty = unit(-(y - cy), x - cx)
            px, py = unit(cx - x, cy - y) if math.hypot(cx - x, cy - y) > 3.5 else (0.0, 0.0)
            move = (0.5 * tx + px, 0.5 * ty + py)
        if plan.strafe is not None:
            s = self.legs.strafe(me, plan.strafe, t, safe)
            move = (move[0] + s[0], move[1] + s[1])
        d = self.legs.dodge(me, obs["projectiles"], self.cfg["weapons"]["mortar"]["gravity"], safe)
        if d is not None:
            move = d
        n = math.hypot(*move)
        if n > 1:
            move = (move[0] / n, move[1] / n)
        move, jump, boost = self.legs.unstick(me, move, t)
        if plan.boost and me["grounded"] and me["energy"] > 10 and math.hypot(*move) > 0.5:
            boost = True
        return [round(move[0], 3), round(move[1], 3)], jump, boost

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
