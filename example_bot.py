"""Rebuild on the Robocode melee frame: every decision runs the same independent modules in order.

    sense    -> Memory: what we see, saw lately, heard, and who hit us
    target   -> one target: whoever shoots us, then the cheapest kill (near, low HP, busy, in the open)
    gun      -> per-weapon aim point from the motion predictor (predict.py) and slew-aware fire gates
    radar    -> where the torso looks when no weapon needs it
    goal     -> one strategic goal by priority: zone > early laser > supply drop > loot > zone edge
    movement -> Minimum Risk Movement: score candidate points around us, walk to the lowest risk

Minimum Risk Movement is the strongest melee movement on RoboWiki: risk from every known enemy (strength / d^2,
more when moving straight at or away from it, so it strafes on its own), from our recent spots (keeps moving),
and from the goal and the zone. Out of a fight it just walks the A* path. The map, A* and line of fire are world.py.
"""
import math
import random

import mechfight as mf
from predict import Predictor
from world import World

# strategy (the values the old bot's sim and server runs settled on)
EDGE_MARGIN = 12.0     # m inside the zone edge
LOOKAHEAD_S = 4.0      # aim for where the edge will be
LOOT_RANGE = 90.0      # m: farthest walk for a weapon or cube
EARLY_S, EARLY_RANGE = 60.0, 250.0  # first minute: walk this far for a laser (else a lightning)
CUBE_KEEP = 15
MAP_MARGIN = 15.0
WEAPON_VALUE = {"laser": 1.0, "lightning": 0.7}  # never a mortar: a laser cooks it off
GUN_OPEN = 100.0       # m: the gun opens a fight only this close (joins one at any range)
GUN_TOL = 1.0          # m off the lead point at the target
GUN_MIN_P = 0.15
LASER_MARGIN = 0.8
LATENCY = 0.03
PATH_BUDGET = 0.01
MECH_SOUNDS = {"gun", "mortar", "laser", "laser_charge", "lightning", "step", "boost", "reload", "jet"}
UPGRADES = ["plating", "laser_damage", "nanites", "plating", "laser_damage", "laser_charge", "plating", "armor",
            "gun_damage", "gun_rate", "sensors", "armor", "gyros", "nanites", "servos", "gun_mag", "gun_velocity",
            "lightning_efficiency", "lightning_range", "lightning_chain", "capacitor", "reactor", "jets", "laser_pierce"]

# minimum risk movement
MRM_N = 16             # candidate points per decision
MRM_R = (8.0, 22.0)    # m: candidate distance band
MRM_EVERY = 0.4        # s between re-picks (sooner once there)
ENEMY_W = 900.0        # risk of an equal enemy at 1 m; /d^2
MEMORY_S = 6.0         # s we keep an enemy we stopped seeing


def ang_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(dx, dy):
    d = math.hypot(dx, dy)
    return [dx / d, dy / d] if d > 1e-6 else [0.0, 0.0]


def aim_at(src, p):
    return [math.atan2(p[1] - src[1], p[0] - src[0]), math.atan2(p[2] - src[2], math.dist(p[:2], src[:2]))]


def aim_error(a, b):
    return math.hypot(ang_diff(a[0], b[0]) * math.cos(b[1]), a[1] - b[1])


def zone_at(z, t):
    span = z["shrink_end"] - z["shrink_start"]
    f = min(max((t - z["shrink_start"]) / span, 0.0), 1.0) if span > 0 else float(t >= z["shrink_end"])
    return ([z["center"][i] + (z["next_center"][i] - z["center"][i]) * f for i in (0, 1)],
            z["radius"] + (z["next_radius"] - z["radius"]) * f)


def risk(p, me, enemies, recent, goal, zone_c, zone_r):
    """Minimum Risk Movement score of standing at p (lower is better)."""
    r = 0.0
    for e in enemies:
        d = max(d2(p, e["pos"]), 3.0)
        # moving along the line to it (straight at or away) is easy to hit: Robocode's perpendicularity factor
        along = abs(math.cos(math.atan2(p[1] - me[1], p[0] - me[0]) - math.atan2(e["pos"][1] - me[1], e["pos"][0] - me[0])))
        r += e["w"] * ENEMY_W / (d * d) * (1 + along)
    for q in recent:
        r += 40.0 / max(d2(p, q), 2.0) ** 2
    if goal:
        r += d2(p, goal) / 15.0
    out = d2(p, zone_c) - max(zone_r - EDGE_MARGIN, 0.0)
    if out > 0:
        r += 20 + out
    return r


class Memory:
    """Sense: everything the modules read about the world beyond this observation."""

    def __init__(self, short_s):
        self.enemies = {}       # id -> last seen mech dict + "t"
        self.hit_by, self.hit_on = {}, {}
        self.threat = self.heard = None   # (bearing, t)
        self.rear = None                  # (bearing, t): hit from outside our front arc
        self.pickups, self.cubes, self.bad = {}, {}, {}
        self.short_until, self.short_s = 0.0, short_s

    def update(self, obs, t, pos):
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                self.enemies[m["id"]] = dict(m, t=t)
        self.enemies = {k: v for k, v in self.enemies.items() if t - v["t"] < MEMORY_S}
        for p in obs["pickups"]:
            self.pickups[p["id"]] = (p["kind"], p["pos"], t)
        for c in obs["cubes"]:
            self.cubes[c["id"]] = (c["pos"], t)
        seen_p, seen_c = {p["id"] for p in obs["pickups"]}, {c["id"] for c in obs["cubes"]}
        self.pickups = {k: v for k, v in self.pickups.items() if (k in seen_p or d2(v[1], pos) > 25) and t - v[2] < 60}
        self.cubes = {k: v for k, v in self.cubes.items() if (k in seen_c or d2(v[0], pos) > 25) and t - v[1] < 60}
        for e in obs["events"]:
            ty = e["type"]
            if ty == "damaged" and e.get("bearing") is not None:
                self.threat = (e["bearing"], t)
                if e.get("rear"):
                    self.rear = (e["bearing"], t)
                if e.get("by") is not None:
                    self.hit_by[e["by"]] = t
            elif ty == "hit":
                self.hit_on[e["target"]] = t
            elif ty == "take_failed":
                self.bad[e["id"]] = t + 5
                if e["reason"] == "gone":
                    self.pickups.pop(e["id"], None)
            elif ty == "took":
                self.pickups.pop(e["id"], None)
            elif ty == "effect" and e["effect"] == "short":
                self.short_until = t + self.short_s
        heard = [s for s in obs["sounds"] if s["kind"] in MECH_SOUNDS]
        if heard:
            s = min(heard, key=lambda s: (["near", "mid", "far"].index(s["dist"]), s["age"]))
            self.heard = (s["bearing"], t - s["age"])

    def recently(self, rec, t, s):
        return rec is not None and t - rec[1] < s


class Bot:
    name = "mrm-melee"

    def __init__(self, hello: mf.Hello):
        cfg = self.cfg = hello["config"]
        self.mech = cfg["mech"]
        self.rng = random.Random(hello["seed"])
        try:
            self.world = World(hello["map"]["dir"], self.mech)
        except Exception as e:  # still play without the map: straight lines, no cover checks
            mf.log(f"map not loaded: {e!r}")
            self.world = None
        self.mem = Memory(cfg["weapons"]["lightning"].get("short_s", 2))
        self.pred = Predictor(self.mech.get("gravity", 22), self.world.ground_at if self.world else None)
        self.path, self.path_goal, self.path_t = [], None, -1e9
        self.dest, self.dest_t = None, -1e9
        self.recent = []                 # our own spots, one per MRM pick
        self.hist = []                   # (t, x, y, wanted to move) for stuck detection
        self.stuck_since, self.unstick_until, self.unstick_dir = None, -1.0, [0.0, 0.0]
        self.burn = True
        self.t = 0.0

    # ---------- helpers ----------
    def los(self, a, b):
        return self.world.clear(a, b) if self.world else True

    def size(self, m):
        return 1 + self.cfg["cubes"].get("size_per_cube", 0.05) * m.get("cubes", 0)

    def wants(self, kind, me):
        """(take it?, slot to swap out). Loadout: gun + laser, a lightning until a laser turns up."""
        ws = me["weapons"]
        if kind not in WEAPON_VALUE or any(w and w["kind"] == kind for w in ws):
            return False, None
        if None in ws:
            return True, None
        swap = [i for i, w in enumerate(ws) if w["kind"] != "gun" and not w.get("charge")]
        if not swap:
            return False, None
        i = min(swap, key=lambda i: WEAPON_VALUE.get(ws[i]["kind"], 0))
        return WEAPON_VALUE[kind] > WEAPON_VALUE.get(ws[i]["kind"], 0) + 0.2, i

    # ---------- target ----------
    def target(self, obs, me, muzzle, hpf):
        t, mem = self.t, self.mem
        mechs, others = [], []
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                s = self.size(m)
                mechs.append(dict(id=m["id"], kind="mech", pos=m["pos"], vel=m["vel"], m=m, r=self.mech["radius"] * s,
                                  c=[m["pos"][0], m["pos"][1], m["pos"][2] + self.mech["height"] * s / 2]))
            else:
                n = self.cfg["npc"]
                others.append(dict(id=m["id"], kind="npc", pos=m["pos"], vel=m["vel"], m=m, r=n["radius"],
                                   c=[m["pos"][0], m["pos"][1], m["pos"][2] + n["height"] / 2]))
        cr = self.cfg["crates"]
        for k in obs["crates"]:
            others.append(dict(id=k["id"], kind="crate", pos=k["pos"], vel=[0, 0, 0], m=k, r=cr["radius"],
                               c=[k["pos"][0], k["pos"][1], k["pos"][2] + cr["height"] / 2]))
        for x in mechs + others:
            x["d"] = math.dist(muzzle, x["c"])
        mechs = sorted((x for x in mechs if x["d"] < 300), key=lambda x: x["d"])[:6]
        for x in mechs:
            x["los"] = self.los(muzzle, x["c"])
        shooting = lambda x: t - mem.hit_by.get(x["id"], -1e9) < 3  # noqa: E731
        npc = next((x for x in others if x["kind"] == "npc" and shooting(x)), None)
        if npc and not any(shooting(x) for x in mechs):
            npc["los"] = self.los(muzzle, npc["c"])
            return npc
        if mechs:
            busy = {b["by"] for b in obs["beams"]}

            def cost(x):  # melee targeting: shooter first, then the cheapest kill
                m = x["m"]
                away = abs(ang_diff(m["aim"][0], math.atan2(muzzle[1] - x["pos"][1], muzzle[0] - x["pos"][0]))) > math.pi / 2
                hp = m.get("hp", 1) / max(m.get("max_hp", 1), 1)
                return x["d"] / 100 + 1.5 * hp + 1.5 * (not x["los"]) - 0.8 * away - 0.4 * (x["id"] in busy) - 1.0 * shooting(x)
            return min(mechs, key=cost)
        if hpf < 0.6:
            return None
        for x in sorted(others, key=lambda x: (x["kind"] != "npc", x["d"])):
            if x["d"] < (150 if x["kind"] == "npc" else 120) and (x["kind"] == "npc" or me["cubes"] < CUBE_KEEP or None in me["weapons"]):
                x["los"] = self.los(muzzle, x["c"])
                if x["los"]:
                    return x
        return None

    # ---------- gun ----------
    def point_at(self, T, tau, hgt):
        p, _ = self.pred.predict(T["id"], T["pos"], T["vel"], tau)
        return [p[0], p[1], p[2] + hgt]

    def gun_point(self, muzzle, T, w):
        """(lead point with a clear corridor, hit chance) or (None, 0)."""
        st = w["stats"]
        lat = LATENCY + max(w["cooldown"], 0.0)
        mid = T["c"][2] - T["pos"][2]
        hgt = 2 * mid
        tau, p, info = lat + max(T["d"] - 3, 0) / st["speed"], T["pos"], (None, None)
        for _ in range(3):
            p, info = self.pred.predict(T["id"], T["pos"], T["vel"], tau)
            tau = lat + max(math.dist(muzzle, [p[0], p[1], p[2] + mid]) - 3, 0) / st["speed"]
        d = math.dist(muzzle, [p[0], p[1], p[2] + mid])
        sig = max(0.4, d * math.radians(st.get("spread_deg", 0.6)))
        p_hit = (info[1] if info[1] is not None else 0.7) * math.erf((T["r"] + 0.3) / (sig * math.sqrt(2)))
        if not self.world:
            return [p[0], p[1], p[2] + mid], p_hit
        side = unit(-(p[1] - muzzle[1]), p[0] - muzzle[0])
        for z in (mid, min(mid + 1.5, hgt - 1.5), max(mid - 1.5, 1.5)):
            pt = [p[0], p[1], p[2] + z]
            if not self.los(muzzle, pt):
                continue
            shifts = ([0, 0, -sig], [0, 0, sig], [side[0] * sig, side[1] * sig, 0], [-side[0] * sig, -side[1] * sig, 0])
            clear = sum(self.los(muzzle, [pt[j] + sh[j] for j in range(3)]) for sh in shifts)
            if clear >= 3:
                return pt, p_hit * (0.75 + 0.0625 * clear)
        return None, 0.0

    def beam_point(self, muzzle, T):
        mid = T["c"][2] - T["pos"][2]
        for z in (mid, 2 * mid - 1.2, 2 * mid - 2.5, 1.5):
            pt = self.point_at(T, LATENCY, z)
            if self.los(muzzle, pt):
                return pt
        return None

    def gun(self, me, muzzle, T):
        """(aim or None, fire[2], reload): each weapon aims for itself, the torso serves the one about to fire."""
        t, ws, now = self.t, me["weapons"], me["aim"]
        aims, ok, dist = [None, None], [False, False], [1.0, 1.0]
        for i, w in enumerate(ws):
            if not w or not T:
                continue
            k, st = w["kind"], w["stats"]
            if k == "gun":
                pt, p_hit = self.gun_point(muzzle, T, w)
                joined = T["kind"] != "mech" or t - max(self.mem.hit_by.get(T["id"], -1e9), self.mem.hit_on.get(T["id"], -1e9)) < 5
                ok[i] = (pt is not None and T["los"] and T["d"] <= (st["range"] if joined else GUN_OPEN)
                         and w.get("ammo", 1) > 0 and not w.get("reload") and p_hit >= GUN_MIN_P)
            elif k == "laser":
                if T["kind"] == "crate":
                    continue
                pt = self.beam_point(muzzle, T)
                ok[i] = pt is not None and T["los"] and T["d"] <= st["range"] and t >= self.mem.short_until
                pt = pt or T["c"]
            elif k == "lightning":
                pt = self.point_at(T, LATENCY, T["c"][2] - T["pos"][2])
                ok[i] = T["los"] and math.dist(muzzle, pt) <= st["range"] * 0.95 and me["energy"] >= st["energy"] + 5
            else:
                continue
            if pt:
                aims[i], dist[i] = aim_at(muzzle, pt), math.dist(muzzle, pt)
        slot = {w["kind"]: i for i, w in enumerate(ws) if w}
        ready = lambda i: i is not None and ok[i] and ws[i]["cooldown"] <= 0.3  # noqa: E731
        g, la, li = slot.get("gun"), slot.get("laser"), slot.get("lightning")
        if la is not None and ws[la].get("charge", 0) > 0:  # charging: follow the gun, swing to the beam point at the end
            full = ws[la]["charge"] >= ws[la]["stats"]["charge"] - 0.15
            p = la if (full or not ready(g)) and aims[la] else g
        else:
            p = next((i for i in (g, la, li) if ready(i)), None)
        if p is None and T:
            p = next((i for i in (g, la, li) if i is not None and aims[i]), None)
        aim = aims[p] if p is not None else None

        dt = 1.0 / self.cfg["match"].get("tick_hz", 20)
        tr, pr = me["stats"].get("turn_rate", 4) * dt, me["stats"].get("pitch_rate", 3) * dt
        then = now if aim is None else [now[0] + max(-tr, min(tr, ang_diff(aim[0], now[0]))), now[1] + max(-pr, min(pr, aim[1] - now[1]))]
        fire = [False, False]
        for i, w in enumerate(ws):
            if not w:
                continue
            if aims[i] is None:
                fire[i] = w["kind"] == "laser" and w.get("charge", 0) > 0  # hold a charge with nothing to hit
                continue
            k, err = w["kind"], aim_error(now, aims[i])
            if k == "laser":
                if w.get("charge", 0) > 0:
                    full = w["charge"] >= w["stats"]["charge"] - 0.05
                    on = aim_error(then, aims[i]) * dist[i] < T["r"] - LASER_MARGIN and err * dist[i] < T["r"] + 1.0
                    fire[i] = not (full and ok[i] and on)
                else:
                    worth = T["kind"] == "mech" or t - self.mem.hit_by.get(T["id"], -1e9) < 3  # no laser on calm drones
                    fire[i] = (ok[i] and worth and w["cooldown"] <= 0 and err < 0.35
                               and self.los(muzzle, self.point_at(T, w["stats"]["charge"], T["c"][2] - T["pos"][2])))
            elif k == "lightning":
                fire[i] = ok[i] and w["cooldown"] <= 0.1 and abs(ang_diff(now[0], aims[i][0])) < math.radians(w["stats"]["cone_deg"]) * 0.4
            elif k == "gun":
                fire[i] = ok[i] and w["cooldown"] <= 0.1 and aim_error(then, aims[i]) * dist[i] < GUN_TOL and err * dist[i] < T["r"] + GUN_TOL
        gw = ws[g] if g is not None else None
        calm = T is None or T["kind"] != "mech"
        reload = bool(gw) and not gw.get("reload") and (gw.get("ammo", 1) == 0 or (calm and gw.get("ammo", 0) < gw["stats"]["mag"] / 2))
        return aim, fire, reload

    # ---------- radar ----------
    def radar(self, me, pos, move, far, zone_c):
        t, mem = self.t, self.mem
        if t < 1.6:
            return [me["aim"][0] + 1.0, 0.0]  # full turn at spawn: see the loot behind us
        if mem.recently(mem.threat, t, 3):
            return [mem.threat[0], 0.0]
        if mem.recently(mem.heard, t, 2):
            return [mem.heard[0], 0.0]
        if far and math.hypot(*move) > 0.5:
            return [math.atan2(move[1], move[0]), 0.0]
        return [math.atan2(zone_c[1] - pos[1], zone_c[0] - pos[0]), 0.0]  # back to the edge: they come from inside

    # ---------- goal ----------
    def edge_spot(self, pos, c, r):
        rr = max(r - EDGE_MARGIN, 0.0)
        base = math.atan2(pos[1] - c[1], pos[0] - c[0]) if d2(pos, c) > 1e-6 else 0.0
        for k in range(40):  # slide along the circle to a walkable spot inside the map
            for s in ((1,) if k == 0 else (1, -1)):
                a = base + s * k * min(0.05, 4 / max(rr, 1))
                p = [c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr]
                if not self.world:
                    return p
                p = self.world.clamp(p, MAP_MARGIN)
                if self.world.free(*p):
                    return p
        return self.world.clamp([c[0] + math.cos(base) * rr, c[1] + math.sin(base) * rr], MAP_MARGIN)

    def goal(self, obs, me, pos, c, r, enemies, hpf):
        """(goal point, name, arrive radius): the one strategic thing to walk to."""
        t, mem = self.t, self.mem
        if not me["in_zone"]:
            return self.edge_spot(pos, c, r), "zone", 3.0
        usable = lambda pid, k, p: mem.bad.get(pid, -1) < t and self.wants(k, me)[0] and d2(p, c) < r - 6  # noqa: E731
        if t < EARLY_S and not mem.recently(mem.threat, t, 3):
            for kind in ("laser", "lightning"):
                best = min(((d2(p, pos), p) for pid, (k, p, _) in mem.pickups.items()
                            if k == kind and usable(pid, k, p) and d2(p, pos) < EARLY_RANGE and abs(p[2] - pos[2]) < 4
                            and not any(d2(e["pos"], p) < d2(p, pos) for e in enemies)), default=None)
                if best:
                    return best[1], "laser", 1.0
        walk = me["stats"].get("walk_speed", 11.7)
        for dr in obs.get("drops", []):
            dd = d2(dr["pos"], pos)
            if (dr["holder"] in (None, me["id"]) and dd < 250 and d2(dr["pos"], c) < r - 3 and dr["lands_in"] <= dd / walk + 10
                    and not any(d2(e["pos"], dr["pos"]) < 60 for e in enemies)):
                return dr["pos"], "drop", max(dr["radius"] - 3, 1.0)
        near = min((d2(e["pos"], pos) for e in enemies), default=1e9)
        if near > 70 and hpf >= 0.35:
            best, bd = None, LOOT_RANGE
            for pid, (k, p, _) in mem.pickups.items():
                if usable(pid, k, p) and d2(p, pos) < bd:
                    best, bd = p, d2(p, pos)
            if me["cubes"] < CUBE_KEEP:
                for p, _ in mem.cubes.values():
                    if d2(p, c) < r - 6 and d2(p, pos) < bd:
                        best, bd = p, d2(p, pos)
            if best:
                return best, "loot", 1.0
        return self.edge_spot(pos, c, r), "edge", 3.0

    # ---------- movement ----------
    def steer(self, pos, goal, force=False):
        """Unit move toward goal along an A* path (corners cut where the straight line is walkable)."""
        if not self.world:
            return unit(goal[0] - pos[0], goal[1] - pos[1])
        t = self.t
        if force or not self.path or self.path_goal is None or d2(goal, self.path_goal) > 6 or t - self.path_t > 2.5:
            self.path, self.path_goal, self.path_t = self.world.path(pos, goal, PATH_BUDGET), list(goal), t
        while len(self.path) > 1 and d2(pos, self.path[0]) < 4:
            self.path.pop(0)
        if not self.path:
            return unit(goal[0] - pos[0], goal[1] - pos[1])
        tgt = self.path[0]
        for wp in self.path[1:6]:
            if not self.world.line_free(pos, wp):
                break
            tgt = wp
        return unit(tgt[0] - pos[0], tgt[1] - pos[1])

    def movement(self, me, pos, goal, arrive, threats, c, r):
        """Minimum risk move. With nobody to fear, walk the path to the goal (stand still once there)."""
        t = self.t
        gap = d2(goal, pos)
        if not threats:
            self.dest = None
            return self.steer(pos, goal, force=self.stuck_since is not None and t - self.stuck_since > 1) if gap > arrive else [0.0, 0.0]
        # the goal pulls through its next path waypoint, so MRM never walks into a dead end
        way = goal
        if gap > MRM_R[1]:
            v = self.steer(pos, goal)
            way = [pos[0] + v[0] * MRM_R[1], pos[1] + v[1] * MRM_R[1]]
        if self.dest is None or t - self.dest_t > MRM_EVERY or d2(pos, self.dest) < 2.5:
            best = None
            for k in range(MRM_N):
                a = 2 * math.pi * (k + self.rng.random()) / MRM_N
                dd = self.rng.uniform(*MRM_R)
                p = [pos[0] + math.cos(a) * dd, pos[1] + math.sin(a) * dd]
                if self.world and not (self.world.free(*p) and self.world.line_free(pos, p)):
                    continue
                s = risk(p, pos, threats, self.recent, way, c, r)
                if best is None or s < best[0]:
                    best = (s, p)
            if best:
                self.dest, self.dest_t = best[1], t
                self.recent = (self.recent + [list(pos[:2])])[-6:]
        if self.dest is None:
            return self.steer(pos, goal)
        return unit(self.dest[0] - pos[0], self.dest[1] - pos[1])

    def unstick(self, me, pos, move, fighting):
        """(move, jump): commanded to move but going nowhere -> hop; still stuck -> sidestep a second."""
        t, (x, y) = self.t, pos[:2]
        wants = math.hypot(*move) > 0.5
        self.hist = [h for h in self.hist if t - h[0] < 1.2] + [(t, x, y, wants)]
        h0 = self.hist[0]
        stuck = len(self.hist) >= 10 and all(h[3] for h in self.hist) and math.hypot(x - h0[1], y - h0[2]) < 1.5
        self.stuck_since = (t if self.stuck_since is None else self.stuck_since) if stuck else None
        if t < self.unstick_until:
            return self.unstick_dir, False, False
        if stuck:
            if t - self.stuck_since > 2.5:
                a = math.atan2(move[1], move[0]) + self.rng.choice((-1, 1)) * math.pi / 2
                self.unstick_dir, self.unstick_until = [math.cos(a), math.sin(a)], t + 1.0
                self.stuck_since, self.path, self.dest = None, [], None
            return move, me["grounded"] and not fighting, True
        return move, False, False

    # ---------- the frame ----------
    def act(self, obs: mf.Observation) -> mf.Action:
        me, t = obs["self"], obs["t"]
        self.t = t
        pos = me["pos"]
        st = me["stats"]
        hpf = me["hp"] / max(me["max_hp"], 1)
        muzzle = [pos[0], pos[1], pos[2] + st.get("muzzle_height", 7)]
        act: mf.Action = {"tick": obs["tick"]}

        # sense
        self.mem.update(obs, t, pos)
        for m in obs["mechs"]:
            rad = self.mech["radius"] * self.size(m) if m["kind"] == "mech" else self.cfg["npc"]["radius"]
            self.pred.update(t, m["id"], m["pos"], m["vel"], rad + 0.3, pos)
        if obs["offer"]:
            act["pick"] = self.pick_upgrade([ch["id"] for ch in obs["offer"]["choices"]], me)
        self.loot_hands(obs, me, pos, act)

        # target + gun
        T = self.target(obs, me, muzzle, hpf)
        aim, fire, reload = self.gun(me, muzzle, T)
        mem = self.mem
        if mem.recently(mem.rear, t, 1.0) and not (T and T["kind"] == "mech" and T["los"] and t - mem.hit_by.get(T["id"], -1e9) < 1.0):
            aim = [mem.rear[0], 0.0]  # hit in the back (x1.5) by someone not in front: turn round
            fire = [bool(w and w["kind"] == "laser" and w.get("charge", 0) > 0) for w in me["weapons"]]

        # goal + movement
        c, r = zone_at(obs["zone"], t + LOOKAHEAD_S)
        enemies = list(mem.enemies.values())
        goal, mode, arrive = self.goal(obs, me, pos, c, r, enemies, hpf)
        my_hp = max(me["hp"], 1)
        threats = [dict(pos=e["pos"], w=min(max(e.get("hp", my_hp) / my_hp, 0.3), 3.0) * (1.5 if "laser" in e.get("weapons", []) else 1.0))
                   for e in enemies if d2(e["pos"], pos) < 200] if mode not in ("zone",) else []
        move = self.dodge(obs, pos)
        if move is None:
            move = self.movement(me, pos, goal, arrive, threats, c, r)
        fighting = bool(T and T["kind"] == "mech" and T["los"] and me["in_zone"])
        move, jump, stuck = self.unstick(me, pos, move, fighting)

        gap = d2(goal, pos)
        boost = mode == "zone" or (stuck and not fighting) or (gap > 40 and me["energy"] > 60 and not T)
        if fighting:  # burn through the fight, walk just long enough to recharge
            self.burn = me["energy"] > 70 if not self.burn else me["energy"] > 25
            boost = boost or (self.burn and me["grounded"])
        if aim is None:
            aim = self.radar(me, pos, move, gap > 30, c)

        act.update(move=move, boost=boost, jump=jump, aim=aim, fire=fire, reload=reload)
        mf.draw.label(f"{mode} · hp {me['hp']:.0f} · {T['kind'] + ' ' + str(T['id']) if T else '-'}")
        if self.dest:
            mf.draw.point([self.dest[0], self.dest[1], pos[2]], "#ffd060", 6)
        return act

    def dodge(self, obs, pos):
        """Move away from where an incoming shell lands (beams are hitscan: no dodging those)."""
        mo = self.cfg["weapons"]["mortar"]
        for p in obs["projectiles"]:
            if p["kind"] != "mortar":
                continue
            v, z = p["vel"], p["pos"][2] - (pos[2] + 1)
            disc = v[2] ** 2 + 2 * mo["gravity"] * z
            if disc < 0:
                continue
            tl = (v[2] + math.sqrt(disc)) / mo["gravity"]
            land = [p["pos"][0] + v[0] * tl, p["pos"][1] + v[1] * tl]
            if tl < 3 and d2(land, pos) < mo["splash_radius"] * 1.3 + 3:
                u = unit(pos[0] - land[0], pos[1] - land[1])
                return u if u != [0.0, 0.0] else unit(-v[1], v[0])
        return None

    def loot_hands(self, obs, me, pos, act):
        """Take a wanted weapon within reach; drop a mortar (it only gets us cooked off)."""
        reach = self.cfg["loot"]["pickup_radius"] - 0.3
        for p in obs["pickups"]:
            if d2(p["pos"], pos) < reach and abs(p["pos"][2] - pos[2]) < 3.8 and self.mem.bad.get(p["id"], -1) < self.t:
                want, slot = self.wants(p["kind"], me)
                if want:
                    act["take"] = p["id"]
                    if slot is not None:
                        act["drop"] = slot
                    return
        junk = next((i for i, w in enumerate(me["weapons"]) if w and w["kind"] == "mortar"), None)
        if junk is not None:
            act["drop"] = junk

    def pick_upgrade(self, ids, me):
        taken = me.get("upgrades") or []

        def rank(u):
            if u.startswith("mortar"):
                return 1e9
            spots = [i for i, v in enumerate(UPGRADES) if v == u]
            n = taken.count(u)
            return spots[n] if n < len(spots) else len(UPGRADES) + 1
        return min(range(len(ids)), key=lambda i: rank(ids[i]))


if __name__ == "__main__":
    mf.run(Bot)
