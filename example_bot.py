"""Bot: hold the inside edge of the zone, fight with the right weapon for the range, dodge, loot, use the map to path and aim.

Priorities each decision (movement): dodge a laser/shell > get back in the zone > flee when weak
> capture a supply drop > grab nearby loot > hold the zone edge (strafing while fighting).
"""
import math
import random
import time

import mechfight as mf
from world import World

EDGE_MARGIN = 12.0   # m to stay inside the zone edge
LOOKAHEAD_S = 4.0    # aim for where the edge will be, so a shrinking zone doesn't catch us
ARRIVED = 3.0        # m from a goal: stand still (silent, and HP regens)
LOOT_RANGE = 90.0    # m: farthest we walk for a weapon or cube
CUBE_KEEP = 15       # past this, cubes only make us bigger (stats cap at 10; 15 = one more upgrade)
MORTAR_MIN = 20.0    # m: closer than this our own splash and knockback hurt us
PATH_BUDGET = 0.01   # s of A* per decision (the whole decision must fit in 50 ms)
MAP_MARGIN = 15.0    # m: goals stay this far inside the map edge (a wall); the first zone is bigger than the map
MECH_SOUNDS = {"gun", "mortar", "laser", "laser_charge", "lightning", "step", "boost", "reload", "jet"}
WEAPON_VALUE = {"laser": 1.0, "lightning": 0.7, "gun": 0.5}  # never a mortar: a laser hit on its carrier cooks off (+60%)
EARLY_S, EARLY_RANGE = 60.0, 250.0  # the first minute: walk this far for a laser (else a lightning)
GUN_OPEN = 100.0     # m: the gun doesn't start a fight with a mech farther than this (it joins one at any range)
SCAN_EVERY, SCAN_S = 8.0, 1.6  # s: standing at the edge, a full look around this often (turn_rate 4 rad/s: 1.6 s)
QUIET_MOVE, QUIET_NEAR = 0.42, 60.0  # under 5 m/s (|move| 0.42 of 11.7) makes no step sound; walk so near enemies
GUN_HIT = [(40, 1.0), (80, 0.56), (120, 0.40), (180, 0.14), (230, 0.05)]  # our gun vs a strafer, by range (aim range)
THEIR_GUN_HIT = [(50, 0.9), (100, 0.75), (150, 0.6), (230, 0.3)]          # the built-ins' gun (recorded match: ~60% at 150 m)
WIN_MARGIN = 0.75    # we win if our time to kill < this x theirs (room for a third mech)
STICKY = 0.7         # a new target must score this much better (lower) than the current one
COVER_NEAR = 30.0    # m: losing a fight, fall back to cover only this close (farther, stand and fight)
PEEK_S = 1.5         # s out of cover for one shot before stepping back
AMBUSH_S = 1.0       # s: a foe about to come round the corner this soon finds our laser charged and waiting
# Upgrade order, one entry per stack; with lightning in hand its two upgrades come right after laser_charge.
UPGRADE_PLAN = ["plating", "laser_damage", "nanites", "plating", "laser_damage", "laser_charge", "plating", "armor",
                "gun_damage", "gun_rate", "sensors", "armor", "gyros"]
UPGRADE_REST = ["armor", "nanites", "laser_charge", "laser_damage", "gun_damage", "gun_rate", "servos", "gyros", "sensors",
                "gun_mag", "gun_velocity", "lightning_efficiency", "lightning_range", "lightning_chain", "capacitor",
                "reactor", "jets", "laser_pierce"]


# ---------- geometry (pure, tested in test_example_bot.py) ----------

def zone_at(zone, t):
    """Zone circle at match time t: holds, then moves linearly to next_* between shrink_start and shrink_end."""
    span = zone["shrink_end"] - zone["shrink_start"]
    f = min(max((t - zone["shrink_start"]) / span, 0.0), 1.0) if span > 0 else float(t >= zone["shrink_end"])
    (cx, cy), (nx, ny) = zone["center"], zone["next_center"]
    return [cx + (nx - cx) * f, cy + (ny - cy) * f], zone["radius"] + (zone["next_radius"] - zone["radius"]) * f


def edge_spot(pos, center, radius, margin):
    """Closest point to pos that sits `margin` inside the circle's edge."""
    dx, dy = pos[0] - center[0], pos[1] - center[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        dx, dy, d = 1.0, 0.0, 1.0
    r = max(radius - margin, 0.0)
    return [center[0] + dx / d * r, center[1] + dy / d * r]


def lead(src, tpos, tvel, speed):
    """Point to aim at so a straight shot at `speed` meets a target moving at constant tvel."""
    p = tpos
    for _ in range(4):  # fixed-point iteration; converges fast since bullets are much faster than mechs
        t = math.dist(src, p) / speed
        p = [tpos[i] + tvel[i] * t for i in range(3)]
    return p


def aim_at(src, p):
    return [math.atan2(p[1] - src[1], p[0] - src[0]), math.atan2(p[2] - src[2], math.dist(p[:2], src[:2]))]


def ang_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def aim_error(actual, want):
    return math.hypot(ang_diff(actual[0], want[0]) * math.cos(want[1]), actual[1] - want[1])


def mortar_aim(src, tgt, vel, v, g, high=False, max_pitch=1.3):
    """([yaw, pitch], flight time) for a shell from src to land on a target moving at vel, or None if out of reach.
    `high` picks the lob over obstacles instead of the flat arc."""
    p, pitch, t = tgt, 0.0, 0.0
    for _ in range(4):
        dx, dz = math.dist(src[:2], p[:2]), p[2] - src[2]
        disc = v ** 4 - g * (g * dx * dx + 2 * dz * v * v)
        if dx < 1 or disc < 0:
            return None
        r = math.sqrt(disc)
        pitch = math.atan((v * v + r) / (g * dx)) if high else math.atan((v * v - r) / (g * dx))
        t = dx / (v * math.cos(pitch))
        p = [tgt[0] + vel[0] * t, tgt[1] + vel[1] * t, tgt[2]]
    if pitch > max_pitch:
        return None
    return [math.atan2(p[1] - src[1], p[0] - src[0]), pitch], t


def shell_landing(pos, vel, g, floor_z):
    """(x, y, seconds) where a falling shell crosses height floor_z, or None if it never does."""
    disc = vel[2] ** 2 + 2 * g * (pos[2] - floor_z)
    if disc < 0:
        return None
    t = (vel[2] + math.sqrt(disc)) / g
    return pos[0] + vel[0] * t, pos[1] + vel[1] * t, t


def interp(table, x):
    """Piecewise-linear lookup in [(x, y), ...] sorted by x, flat beyond both ends."""
    if x <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def unit(dx, dy):
    d = math.hypot(dx, dy)
    return [dx / d, dy / d] if d > 1e-6 else [0.0, 0.0]


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------- the bot ----------

class Bot:
    name = "edge-hugger"

    def __init__(self, hello: mf.Hello):
        cfg = self.cfg = hello["config"]
        self.mech = cfg["mech"]
        self.rng = random.Random(hello["seed"])
        self.world = None
        try:
            self.world = World(hello["map"]["dir"], self.mech)
        except Exception as e:  # without the map we still play: straight lines, no cover checks
            mf.log(f"map not loaded, playing without it: {e!r}")
        self.path, self.path_goal, self.path_t = [], None, -1e9
        self.hist = []                   # (t, x, y, wanted to move)
        self.stuck_since = None
        self.unstick_until, self.unstick_dir = -1.0, [0.0, 0.0]
        self.threat = None               # (bearing, t) of the last hit we took
        self.turn_to = None              # (bearing, t): hit in the back by someone we can't see: face that way a moment
        self.heard = None                # (bearing, t, dist) of the last mech sound
        self.strafe_sign, self.strafe_until = 1, 0.0
        self.burn = True  # boosting in a fight: on until energy drops to 25, back on at 70
        self.vhist = {}  # target id -> [(t, its velocity)] over the last 2 s
        self.short_until = 0.0
        self.pickups = {}                # id -> (kind, pos, t seen)
        self.cubes = {}                  # id -> (pos, t seen)
        self.bad_pickups = {}            # id -> t until we stop trying it
        self.hit_by, self.hit_on = {}, {}  # id -> last time it damaged us / we damaged it
        self.foes = {}                   # mech id -> (pos, t) last seen, kept 30 s
        self.busy, self.hp_seen = {}, {}  # mech id -> last time it fought someone else / (hp, t) last seen
        self.target_id = None
        self.hide = None                 # (spot [x, y], t chosen): out of sight of whoever we're losing to
        self.peek_until, self.next_peek, self.peek_from, self.peek_charged = -1.0, 0.0, None, False
        self.scan_t, self.scan_until = 0.0, -1.0

    # ----- memory -----
    def remember(self, obs, t, pos):
        for p in obs["pickups"]:
            self.pickups[p["id"]] = (p["kind"], p["pos"], t)
        for c in obs["cubes"]:
            self.cubes[c["id"]] = (c["pos"], t)
        seen_p = {p["id"] for p in obs["pickups"]}
        seen_c = {c["id"] for c in obs["cubes"]}
        # Close by and not listed: someone took it. Old memories expire.
        self.pickups = {k: v for k, v in self.pickups.items()
                        if (k in seen_p or d2(v[1], pos) > 25) and t - v[2] < 60}
        self.cubes = {k: v for k, v in self.cubes.items() if (k in seen_c or d2(v[0], pos) > 25) and t - v[1] < 60}
        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("bearing") is not None:
                self.threat = (e["bearing"], t)
                if e.get("rear"):
                    self.turn_to = (e["bearing"], t)
                if e.get("by") is not None:
                    self.hit_by[e["by"]] = t
            elif e["type"] == "hit":
                self.hit_on[e["target"]] = t
            elif e["type"] == "take_failed":
                self.bad_pickups[e["id"]] = t + 5
                if e["reason"] == "gone":
                    self.pickups.pop(e["id"], None)
            elif e["type"] == "took":
                self.pickups.pop(e["id"], None)
            elif e["type"] == "effect" and e["effect"] == "short":
                self.short_until = t + self.cfg["weapons"]["lightning"].get("short_s", 2)
        for e in obs["events"]:
            if e["type"] == "kill":
                self.foes.pop(e["target"], None)
        for b in obs["beams"]:  # a beam that doesn't end near us: its shooter is busy with someone else
            if d2(b["path"][-1], pos) > 10:
                self.busy[b["by"]] = t - b["age"]
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                self.foes[m["id"]] = (m["pos"], t)
                h = self.hp_seen.get(m["id"])
                if h and m["hp"] < h[0] - 1 and t - self.hit_on.get(m["id"], -1e9) > 1.0:  # losing HP, not to us
                    self.busy[m["id"]] = t
                self.hp_seen[m["id"]] = (m["hp"], t)
        self.foes = {k: v for k, v in self.foes.items() if t - v[1] < 30}
        sounds = [s for s in obs["sounds"] if s["kind"] in MECH_SOUNDS]
        if sounds:
            s = min(sounds, key=lambda s: (["near", "mid", "far"].index(s["dist"]), s["age"]))
            self.heard = (s["bearing"], t - s["age"], s["dist"])

    def los(self, a, b):
        return self.world.clear(a, b) if self.world else True

    # ----- weapons -----
    def wants(self, kind, me):
        """Is a pickup of this kind worth taking, and into which slot (drop) if both are full? Aim: gun + laser."""
        ws = me["weapons"]
        if kind not in WEAPON_VALUE or kind == "gun" or any(w and w["kind"] == kind for w in ws):
            return False, None
        if None in ws:
            return True, None
        swap = [i for i, w in enumerate(ws) if w["kind"] != "gun" and not w.get("charge")]
        if not swap:
            return False, None
        i = min(swap, key=lambda i: WEAPON_VALUE.get(ws[i]["kind"], 0))
        return WEAPON_VALUE[kind] > WEAPON_VALUE.get(ws[i]["kind"], 0) + 0.2, i  # a laser replaces a lightning

    def classify(self, m, pos):
        """1 attacking us (hit us in 3 s, or charging a laser at us), 2 fighting someone else or under 40% HP,
        3 not aiming at us (more than 0.3 rad off), 4 the rest (watching us)."""
        off = abs(ang_diff(m["aim"][0], math.atan2(pos[1] - m["pos"][1], pos[0] - m["pos"][0])))
        if self.t - self.hit_by.get(m["id"], -1e9) < 3 or (m.get("charging") and off < 0.12):
            return 1
        if self.t - self.busy.get(m["id"], -1e9) < 3 or m.get("hp", 1) < 0.4 * m.get("max_hp", 1):
            return 2
        return 3 if off > 0.3 else 4

    def dps(self, kinds, cubes, d, table, stats=None):
        """Damage per second of these weapons at range d (stats: ours after upgrades, else the config's)."""
        W, tot = self.cfg["weapons"], 0.0
        for k in kinds:
            st = (stats or {}).get(k) or W.get(k)
            if not st or (k != "mortar" and d > st["range"]):
                continue
            if k == "laser":
                tot += st["damage"] / (st["charge"] + st["cooldown"])
            elif k == "gun":
                tot += st["damage"] * st["mag"] / (st["mag"] * st["cooldown"] + st["reload"]) * interp(table, d)
            elif k == "lightning":
                tot += st["damage"] / st["cooldown"]
            elif k == "mortar":
                tot += 40.0
        return tot * (1 + self.cfg["cubes"].get("damage_per_cube", 0.04) * min(cubes, self.cfg["cubes"].get("stat_cap", 10)))

    def wins(self, me, T, mechs):
        """Our time to kill T < WIN_MARGIN x the time T plus everyone shooting us take to kill us. Our lightning on T
        (within its range) shorts T's laser."""
        stats = {w["kind"]: w["stats"] for w in me["weapons"] if w}
        if not T["los"]:
            return False
        ours = self.dps(list(stats), me.get("cubes", 0), T["d"], GUN_HIT, stats)
        light = "lightning" in stats and T["d"] <= stats["lightning"]["range"]
        theirs = 0.0
        for x in mechs:
            if x is T or x.get("cls") == 1:
                kinds = [k for k in x["m"].get("weapons", []) if not (light and x is T and k == "laser")]
                theirs += self.dps(kinds, x["m"].get("cubes", 0), x["d"], THEIR_GUN_HIT)
        return ours > 0 and T["m"]["hp"] / ours < WIN_MARGIN * me["hp"] / max(theirs, 1e-9)

    def targets(self, obs, me, muzzle, hpf):
        """The best thing to shoot: a mech, else (when healthy and calm) an NPC drone or a crate."""
        spc = self.cfg["cubes"].get("size_per_cube", 0.05)
        mechs, others = [], []
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                s = 1 + spc * m.get("cubes", 0)
                c = [m["pos"][0], m["pos"][1], m["pos"][2] + self.mech["height"] * s / 2]
                mechs.append(dict(id=m["id"], kind="mech", pos=m["pos"], c=c, vel=m["vel"], r=self.mech["radius"] * s, m=m))
            else:
                n = self.cfg["npc"]
                c = [m["pos"][0], m["pos"][1], m["pos"][2] + n["height"] / 2]
                others.append(dict(id=m["id"], kind="npc", pos=m["pos"], c=c, vel=m["vel"], r=n["radius"], m=m))
        cr = self.cfg["crates"]
        for k in obs["crates"]:
            c = [k["pos"][0], k["pos"][1], k["pos"][2] + cr["height"] / 2]
            others.append(dict(id=k["id"], kind="crate", pos=k["pos"], c=c, vel=[0, 0, 0], r=cr["radius"], m=k))
        for x in mechs + others:
            x["d"] = math.dist(muzzle, x["c"])
        def score(x):  # lower is better; attackers first, then the busy and the weak, then the unaware
            m = x["m"]
            away = abs(ang_diff(m["aim"][0], math.atan2(muzzle[1] - x["pos"][1], muzzle[0] - x["pos"][0]))) > math.pi / 2
            hp = m.get("hp", 1) / max(m.get("max_hp", 1), 1)
            return (1 + x["d"] / 100 + hp * 1.5 + (0 if x["los"] else 1.5) - 0.8 * away) * {1: 0.4, 2: 0.7, 3: 1.0, 4: 1.2}[x["cls"]]

        mechs = sorted((x for x in mechs if x["d"] < 300), key=lambda x: x["d"])[:6]
        for x in mechs:
            x["los"] = self.los(muzzle, x["c"])
            x["cls"] = self.classify(x["m"], muzzle)
        npc = next((x for x in others if x["kind"] == "npc" and self.t - self.hit_by.get(x["id"], -1e9) < 3), None)
        if npc and not any(self.t - self.hit_by.get(x["id"], -1e9) < 3 for x in mechs):  # a drone is shooting us: answer it
            npc["los"] = self.los(muzzle, npc["c"])
            return npc
        if mechs:
            best = min(mechs, key=score)
            old = next((x for x in mechs if x["id"] == self.target_id), None)
            if old is not None and score(best) > STICKY * score(old):  # stick with the current one unless clearly better
                best = old
            self.target_id = best["id"]
            joined = self.t - max(self.hit_by.get(best["id"], -1e9), self.hit_on.get(best["id"], -1e9)) < 5
            # Fire: back at an attacker or in a fight already on; third-party the busy or weak; first strike with the
            # laser only on one that isn't looking at us and that we beat; never on one just watching us.
            best["open"] = {"gun", "laser", "lightning", "mortar"} if joined or best["cls"] == 1 else (
                {"gun", "laser", "lightning"} if best["cls"] == 2 else
                {"laser"} if best["cls"] == 3 and self.wins(me, best, mechs) else set())
            best["losing"] = (joined or best["cls"] == 1) and not self.wins(me, best, mechs)  # in a fight we don't clearly win
            best["foes"] = [x for x in mechs if x is best or x["cls"] == 1]  # everyone shooting at us
            return best
        if hpf < 0.6:
            return None
        for x in sorted(others, key=lambda x: (x["kind"] != "npc", x["d"])):
            if x["d"] < (150 if x["kind"] == "npc" else 120) and (x["kind"] == "npc" or me["cubes"] < CUBE_KEEP or None in me["weapons"]):
                x["los"] = self.los(muzzle, x["c"])
                if x["los"]:
                    return x
        return None

    def weapons(self, me, muzzle, tube, T, t, precharge=False):
        """(aim or None, fire[2], reload). precharge: start the laser behind cover, with no line to T yet."""
        ws, now = me["weapons"], me["aim"]
        aims, ok, tol = [None, None], [False, False], [0.0, 0.0]
        if T:
            h = self.vhist.setdefault(T["id"], [])
            h[:] = [e for e in h if t - e[0] < 2.0] + [(t, T["vel"])]

            def vavg(s):  # its mean velocity over the last s seconds
                hs = [v for ts, v in h if t - ts <= s + 1e-6]
                return [sum(v[j] for v in hs) / len(hs) for j in range(3)]
            for i, w in enumerate(ws):
                if not w:
                    continue
                k, st = w["kind"], w["stats"]
                size = math.atan2(T["r"], T["d"])
                if k == "gun":
                    p = lead(muzzle, T["c"], T["vel"], st["speed"])
                    aims[i] = aim_at(muzzle, [T["c"][j] + (p[j] - T["c"][j]) * 0.8 for j in range(3)])  # strafers reverse: 80% of the lead hits most (duel-tested)
                    joined = T["kind"] != "mech" or t - max(self.hit_by.get(T["id"], -1e9), self.hit_on.get(T["id"], -1e9)) < 5 or T["cls"] <= 2
                    ok[i] = (T["los"] and T["d"] <= (st["range"] if joined else GUN_OPEN) and w.get("ammo", 1) > 0
                             and not w.get("reload"))  # the gun misses a lot far off and its noise carries 320 m
                    tol[i] = size + 0.006
                elif k == "laser":
                    aims[i] = aim_at(muzzle, T["c"])
                    ok[i] = T["los"] and T["d"] <= st["range"] and t >= self.short_until
                    tol[i] = size + 0.004
                elif k == "lightning":
                    aims[i] = aim_at(muzzle, T["c"])
                    ok[i] = T["los"] and T["d"] <= st["range"] * 0.95 and me["energy"] >= st["energy"] + 5
                    tol[i] = math.radians(st["cone_deg"]) / 2 * 0.7
                elif k == "mortar":
                    # lead by its mean velocity over 2 s: a strafer's shell lands on the middle of its dance (aim-range tested)
                    sol = mortar_aim(tube, [T["pos"][0], T["pos"][1], T["pos"][2] + 0.5], vavg(2.0), st["speed"],
                                     st["gravity"], high=not T["los"], max_pitch=self.mech["max_pitch"])
                    if sol:
                        aims[i] = sol[0]
                    ok[i] = sol is not None and d2(T["pos"], muzzle) >= MORTAR_MIN
                    tol[i] = math.atan2(3, T["d"]) + 0.01

        if T and T["kind"] == "mech":  # only the weapons targets() opened on it (none: just watch it)
            ok = [ok[i] and bool(ws[i]) and ws[i]["kind"] in T["open"] for i in range(2)]

        def ready(i):
            return ok[i] and ws[i]["cooldown"] <= 0.3

        def pick(kinds):
            for k in kinds:
                for i, w in enumerate(ws):
                    if w and w["kind"] == k and ready(i):
                        return i
            return None

        charging = next((i for i, w in enumerate(ws) if w and w["kind"] == "laser" and w.get("charge", 0) > 0), None)
        if charging is not None and aims[charging]:
            p = charging
        elif T and not T["los"]:
            p = pick(["mortar"])
        else:
            p = pick(["laser", "lightning", "gun", "mortar"])
        if p is None and T:
            p = next((i for i, w in enumerate(ws) if w and w["kind"] == "gun" and aims[i]), None)
            p = p if p is not None else next((i for i in (0, 1) if aims[i]), None)
        aim = aims[p] if p is not None else None

        fire = [False, False]
        for i, w in enumerate(ws):
            if not w or aims[i] is None:
                continue
            k, err = w["kind"], aim_error(now, aims[i])
            if k == "laser":
                if w.get("charge", 0) > 0:  # hold until full and on target, then let go (it fires on release)
                    full = w["charge"] >= w["stats"]["charge"] - 0.05
                    fire[i] = not (full and err < tol[i] and ok[i])
                else:
                    fire[i] = (ok[i] and err < 0.35 or precharge and t >= self.short_until) and w["cooldown"] <= 0
            elif k == "lightning":
                fire[i] = ok[i] and w["cooldown"] <= 0.1 and abs(ang_diff(now[0], aims[i][0])) < tol[i]
            else:
                fire[i] = ok[i] and w["cooldown"] <= 0.1 and err < tol[i]
        gun = next((w for w in ws if w and w["kind"] == "gun"), None)
        calm = T is None or T["kind"] != "mech"
        reload = bool(gun) and not gun.get("reload") and (gun.get("ammo", 1) == 0 or (calm and gun.get("ammo", 0) < gun["stats"]["mag"] / 2))
        return aim, fire, reload

    def pick_upgrade(self, ids, me):
        """Index of the offer to take: the earliest next stack in UPGRADE_PLAN, then UPGRADE_REST; never a mortar one."""
        plan = list(UPGRADE_PLAN)
        if any(w and w["kind"] == "lightning" for w in me["weapons"]):
            k = plan.index("laser_charge") + 1
            plan[k:k] = ["lightning_efficiency", "lightning_range"]
        taken = me.get("upgrades") or []

        def rank(u):
            if u.startswith("mortar"):
                return 1e9
            spots = [i for i, v in enumerate(plan) if v == u]
            n = taken.count(u)
            return spots[n] if n < len(spots) else len(plan) + (UPGRADE_REST.index(u) if u in UPGRADE_REST else 99)
        return min(range(len(ids)), key=lambda i: rank(ids[i]))

    def early_laser(self, me, pos, enemies):
        """In the first minute: the nearest laser seen within EARLY_RANGE (else a lightning) we want and can reach first."""
        for kind in ("laser", "lightning"):
            best = min(((d2(p, pos), p) for pid, (k, p, _) in self.pickups.items()
                        if k == kind and self.bad_pickups.get(pid, -1) < self.t and self.wants(k, me)[0]
                        and d2(p, pos) < EARLY_RANGE and abs(p[2] - pos[2]) < 4  # not on a roof
                        and not any(d2(m["pos"], p) < d2(p, pos) for m in enemies)), default=None)  # nobody closer to it
            if best:
                return best[1]
        return None

    # ----- movement -----
    def dodge(self, obs, me, pos):
        """A move vector away from an incoming shell, or None. (No laser dodge: a hitscan beam can't be sidestepped.)"""
        mo = self.cfg["weapons"]["mortar"]
        for p in obs["projectiles"]:
            if p["kind"] == "mortar":
                land = shell_landing(p["pos"], p["vel"], mo["gravity"], pos[2] + 1)
                if land and land[2] < 3 and d2(land, pos) < mo["splash_radius"] * 1.3 + 3:
                    v = unit(pos[0] - land[0], pos[1] - land[1])
                    return (v if v != [0.0, 0.0] else [-p["vel"][1], p["vel"][0]]), "dodge shell"
        return None

    def free_edge_spot(self, pos, c, r):
        """edge_spot, kept inside the map (the first zone is bigger than it), slid along the circle to a walkable cell."""
        spot = edge_spot(pos, c, r, EDGE_MARGIN)
        if not self.world:
            return spot
        spot = self.world.clamp(spot, MAP_MARGIN)
        if self.world.free(*spot):
            return spot
        base = math.atan2(spot[1] - c[1], spot[0] - c[0])
        rr = max(r - EDGE_MARGIN, 0.0)
        for k in range(1, 40):
            for s in (1, -1):
                a = base + s * k * min(0.05, 4 / max(rr, 1))
                p = self.world.clamp([c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr], MAP_MARGIN)
                if self.world.free(*p):
                    return p
        return spot

    def steer(self, pos, goal, t, force=False):
        """Unit move vector toward goal along an A* path."""
        if not self.world:
            return unit(goal[0] - pos[0], goal[1] - pos[1])
        if force or not self.path or self.path_goal is None or d2(goal, self.path_goal) > 6 or t - self.path_t > 2.5:
            self.path, self.path_goal, self.path_t = self.world.path(pos, goal, PATH_BUDGET), list(goal), t
        while len(self.path) > 1 and d2(pos, self.path[0]) < 4:
            self.path.pop(0)
        if not self.path:
            return unit(goal[0] - pos[0], goal[1] - pos[1])
        tgt = self.path[0]
        for wp in self.path[1:6]:  # cut corners where the straight line is walkable
            if not self.world.line_free(pos, wp):
                break
            tgt = wp
        return unit(tgt[0] - pos[0], tgt[1] - pos[1])

    def loot_goal(self, me, pos, c, r):
        best, bd = None, LOOT_RANGE
        for pid, (kind, p, _) in self.pickups.items():
            if self.bad_pickups.get(pid, -1) < self.t and self.wants(kind, me)[0] and d2(p, c) < r - 6 and d2(p, pos) < bd:
                best, bd = p, d2(p, pos)
        if me["cubes"] < CUBE_KEEP:
            for p, _ in self.cubes.values():
                if d2(p, c) < r - 6 and d2(p, pos) < bd:
                    best, bd = p, d2(p, pos)
        return best

    def foe_near(self, obs, pos, t):
        """A mech within QUIET_NEAR m: listed, seen in the last 10 s, or its steps heard close by."""
        if any(d2(p, pos) < QUIET_NEAR for p, ts in self.foes.values() if t - ts < 10):
            return True
        return any(s["kind"] in ("step", "boost", "jet") and s["dist"] == "near" for s in obs["sounds"])

    def hidden(self, p, foes):
        """Is a mech standing at p out of sight of all these foes (no line from their eyes to its middle or top)?"""
        z = self.world.ground_at(p[0], p[1]) if self.world else 0.0
        mid, top = [p[0], p[1], z + self.mech["height"] / 2], [p[0], p[1], z + self.mech["height"]]
        return not any(self.los(e, mid) or self.los(e, top) for e in foes)

    def hide_spot(self, pos, foes):
        """Nearest walkable spot within COVER_NEAR m, walkable to in a straight line, out of sight of all foes."""
        if not self.world:
            return None
        deadline = time.perf_counter() + 0.008  # line-of-sight checks are slow in Python: stop after 8 ms
        for rad in (6.0, 12.0, 20.0, COVER_NEAR):
            best = None
            for k in range(16):
                if time.perf_counter() > deadline:
                    return best[1] if best else None
                a = k * math.pi / 8
                p = [pos[0] + math.cos(a) * rad, pos[1] + math.sin(a) * rad]
                if self.world.free(*p) and self.world.line_free(pos, p) and self.hidden(p, foes):
                    far = min(d2(p, e) for e in foes)  # of the hidden ones on this ring, the one farthest from them
                    if best is None or far > best[0]:
                        best = (far, p)
            if best:
                return best[1]
        return None

    def fall_back(self, me, pos, T, t):
        """Losing a fight with cover near: the spot to hide at (kept while it hides us), else None."""
        if not (T and T["kind"] == "mech" and T.get("losing") and me["in_zone"]):
            self.hide = None
            return None
        foes = [[x["pos"][0], x["pos"][1], x["pos"][2] + self.mech["eye_height"]] for x in T["foes"]]
        if self.hide and (t - self.hide[1] < 1.0 or self.hidden(self.hide[0], foes)):
            return self.hide[0]
        spot = self.hide_spot(pos, foes)
        self.hide = (spot, t) if spot else None
        return spot

    # ----- decision -----
    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        pos = me["pos"]
        x, y, z = pos
        t = self.t = obs["t"]
        self.remember(obs, t, pos)
        act: mf.Action = {"tick": obs["tick"]}
        hpf = me["hp"] / max(me["max_hp"], 1)
        st = me["stats"]
        muzzle = [x, y, z + st.get("muzzle_height", 7)]
        tube = [x, y, z + st.get("height", self.mech["height"]) + 0.5]

        if obs["offer"]:
            act["pick"] = self.pick_upgrade([ch["id"] for ch in obs["offer"]["choices"]], me)

        # Take a wanted weapon within reach (swap out the weakest non-gun if both slots are full).
        reach = self.cfg["loot"]["pickup_radius"] - 0.3
        for p in obs["pickups"]:
            if d2(p["pos"], pos) < reach and abs(p["pos"][2] - z) < 3.8 and self.bad_pickups.get(p["id"], -1) < t:
                want, slot = self.wants(p["kind"], me)
                if want:
                    act["take"] = p["id"]
                    if slot is not None:
                        act["drop"] = slot
                    break
        if "drop" not in act:  # a mortar in hand only gets us cooked off: leave it
            junk = next((i for i, w in enumerate(me["weapons"]) if w and w["kind"] == "mortar"), None)
            if junk is not None:
                act["drop"] = junk

        # Who is dangerous, and should we run?
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        near = min(enemies, key=lambda m: d2(m["pos"], pos), default=None)
        T = self.targets(obs, me, muzzle, hpf)  # no running off in the open: a laser hits a runner just the same
        hide = self.fall_back(me, pos, T, t)  # losing with cover near: shoot from behind it
        if T and T["kind"] == "mech" and T["los"]:
            self.peek_from = list(pos[:2])  # a spot we see it from
        aim, fire, reload = self.weapons(me, muzzle, tube, T, t, precharge=hide is not None and not T["los"])
        if self.turn_to and t - self.turn_to[1] < 1.0 and not (T and T["kind"] == "mech" and T["los"] and T["id"] in self.hit_by
                                                              and t - self.hit_by[T["id"]] < 1.0):
            aim = [self.turn_to[0], 0.0]  # hit in the back (x1.5) by someone we don't have in front of us: turn round now

        # Movement goal.
        c, r = zone_at(obs["zone"], t + LOOKAHEAD_S)
        spot = self.free_edge_spot(pos, c, r)
        safe_r = max(r - EDGE_MARGIN, 0.0)
        arrive, mode, goal, move = ARRIVED, "edge", spot, None
        dodge = self.dodge(obs, me, pos)
        if dodge:
            move, mode = dodge
        elif not me["in_zone"]:
            mode = "zone"
        elif hide:
            mode, goal, arrive = "cover", hide, 1.5
            laser = next((w for w in me["weapons"] if w and w["kind"] == "laser"), None)
            charged = laser is not None and laser.get("charge", 0) >= laser["stats"]["charge"] - 0.05
            if t < self.peek_until:  # stepped out for a shot: back in once the laser has gone off
                if laser is not None and self.peek_charged and not laser.get("charge", 0):
                    self.peek_until = t
                elif self.peek_from:
                    goal, mode = self.peek_from, "peek"
            elif d2(pos, hide) < 2.5 and not T["los"] and (charged or (laser is None and t >= self.next_peek)):
                tp = [T["pos"][i] + T["vel"][i] * AMBUSH_S for i in range(3)]
                coming = self.los([tp[0], tp[1], tp[2] + self.mech["eye_height"]], [x, y, z + self.mech["height"] / 2])
                if not coming and self.peek_from:  # it isn't coming round the corner: step out to where we saw it from
                    self.peek_until, self.next_peek, self.peek_charged = t + PEEK_S, t + PEEK_S + 2.5, charged
                    goal, mode = self.peek_from, "peek"
        else:
            drop = None
            for dr in obs.get("drops", []):
                dd = d2(dr["pos"], pos)
                if (dr["holder"] in (None, me["id"]) and dd < 250 and d2(dr["pos"], c) < r - 3
                        and dr["lands_in"] <= dd / st.get("walk_speed", 11.7) + 10
                        and not any(d2(m["pos"], dr["pos"]) < 60 for m in enemies)):
                    drop = dr
                    break
            loot = None if (near and d2(near["pos"], pos) < 70) or hpf < 0.35 else self.loot_goal(me, pos, c, r)
            shot_at = self.threat is not None and t - self.threat[1] < 3
            early = self.early_laser(me, pos, enemies) if t < EARLY_S and not shot_at else None
            if early:
                goal, mode, arrive = early, "laser", 1.0
            elif drop:
                goal, mode, arrive = drop["pos"], "drop", max(drop["radius"] - 3, 1.0)
            elif loot:
                goal, mode, arrive = loot, "loot", 1.0

        gap = d2(goal, pos)
        engaged = bool(T and T["kind"] == "mech" and T["los"] and T["open"])  # in a fight with it (not just watching it)
        if move is None:
            move = self.steer(pos, goal, t, force=self.stuck_since is not None and t - self.stuck_since > 1) if gap > arrive else [0.0, 0.0]
            if mode == "edge" and engaged:  # strafe across their line of fire
                if t >= self.strafe_until:
                    self.strafe_sign, self.strafe_until = -self.strafe_sign, t + self.rng.uniform(0.35, 0.9)  # reverse before their led shot lands
                yaw = math.atan2(T["pos"][1] - y, T["pos"][0] - x) + self.strafe_sign * math.pi / 2
                pull = min(gap / 6, 1.0)
                to_spot = unit(goal[0] - x, goal[1] - y)
                move = unit(math.cos(yaw) + to_spot[0] * pull, math.sin(yaw) + to_spot[1] * pull)

        # Stuck: commanded to move but not getting anywhere -> jump and jet; if that fails too, sidestep.
        # Not while fighting inside the zone (hopping about loses the fight); outside it the zone kills faster than they do.
        fighting = engaged and me["in_zone"]
        wants_move = math.hypot(*move) > 0.5
        self.hist = [h for h in self.hist if t - h[0] < 1.2] + [(t, x, y, wants_move)]
        h0 = self.hist[0]
        stuck = len(self.hist) >= 10 and all(h[3] for h in self.hist) and math.hypot(x - h0[1], y - h0[2]) < 1.5
        self.stuck_since = (t if self.stuck_since is None else self.stuck_since) if stuck else None
        jump = False
        if t < self.unstick_until:
            move = self.unstick_dir
        elif stuck:
            jump = me["grounded"] and not fighting
            if t - self.stuck_since > 2.5:
                a = math.atan2(move[1], move[0]) + self.rng.choice((-1, 1)) * math.pi / 2
                self.unstick_dir, self.unstick_until = [math.cos(a), math.sin(a)], t + 1.0
                self.stuck_since, self.path = None, []
        boost = mode in ("dodge shell", "zone") or (stuck and not fighting) or (
            gap > 40 and me["energy"] > 60 and not T)
        if fighting:  # boost through the whole fight, walking just long enough to recharge (duel-tested; jets lost)
            self.burn = me["energy"] > 70 if not self.burn else me["energy"] > 25
            boost = boost or (self.burn and me["grounded"])
        elif mode not in ("zone", "dodge shell") and not stuck and self.foe_near(obs, pos, t):  # sneak: no step sounds
            n = math.hypot(*move)
            if n > QUIET_MOVE:
                move = [move[0] / n * QUIET_MOVE, move[1] / n * QUIET_MOVE]
            boost = False

        # Where to look when there's nothing to shoot.
        if aim is None:
            inward = math.atan2(c[1] - y, c[0] - x)
            if t < 1.6:  # a full turn at spawn (turn_rate 4 rad/s): see the loot behind us too
                aim = [me["aim"][0] + 1.0, 0.0]
            elif self.threat and t - self.threat[1] < 3:  # turn to face whoever hit us
                aim = [self.threat[0], 0.0]
            elif self.heard and t - self.heard[1] < 2:
                aim = [self.heard[0], 0.0]
            elif gap > 30 and wants_move:
                aim = [math.atan2(move[1], move[0]), 0.0]
            else:  # at the edge: watch inward (enemies come from inside), look all round now and then
                if t >= self.scan_t:
                    self.scan_t, self.scan_until = t + SCAN_EVERY, t + SCAN_S
                aim = [me["aim"][0] + 1.0, 0.0] if t < self.scan_until else [inward, 0.0]

        act.update(move=move, boost=boost, jump=jump, aim=aim, fire=fire, reload=reload)
        self.draw(pos, muzzle, goal, T, mode, me)
        return act

    def draw(self, pos, muzzle, goal, T, mode, me):
        x, y, z = pos
        prev = [x, y, z + 1]
        for wp in self.path[:8]:
            q = [wp[0], wp[1], z + 1]
            mf.draw.line(prev, q, "#40c0ff")
            prev = q
        mf.draw.point([goal[0], goal[1], z], "#40c0ff", 6)
        if T:
            mf.draw.line(muzzle, T["c"], "#ff5040" if T["los"] else "#806060")
        mf.draw.label(f"{mode} · hp {me['hp']:.0f} · {T['kind'] + ' ' + str(T['id']) if T else '-'}")


if __name__ == "__main__":
    mf.run(Bot)
