"""Bot: roam the safe circle hunting lasers, cubes and supply drops; hide to heal when hurt; fight with the right weapon.

Priorities each decision (movement): dodge a shell > get back in the zone > capture a supply drop
> hide out of sight while HP < HIDE_HP (until HEALED_HP) > a laser in the first minute > loot on the ground
> loot on a roof (only when calm: from within FLY_RANGE, jet straight at it and drop onto it) > stand and strafe against a mech in a fight
> farm a drone or crate we saw (most cubes per metre; shot on the way, its cubes picked up as loot)
> roam to the least recently visited cell (scanning).
"""
import math
import random
import time

import mechfight as mf
from predict import Predictor
from threat import Threats, gun_hit_p
from world import World

EDGE_MARGIN = 12.0   # m to stay inside the zone edge
LOOKAHEAD_S = 4.0    # aim for where the edge will be, so a shrinking zone doesn't catch us
ARRIVED = 3.0        # m from a goal: stand still (silent, and HP regens)
LOOT_RANGE = 200.0   # m: farthest we walk for a weapon or cube (roaming: we go looking)
CUBE_KEEP = 15       # past this, cubes only make us bigger (stats cap at 10; 15 = one more upgrade)
FARM_GUARD = 100.0   # m: a drone or crate this close to a mech we see isn't worth the walk
FARM_S = 120.0       # s we remember a drone or crate we saw (crates stay put, drones keep near their home)
MORTAR_MIN = 20.0    # m: closer than this our own splash and knockback hurt us
PATH_BUDGET = 0.01   # s of A* per decision (the whole decision must fit in 50 ms)
MAP_MARGIN = 15.0    # m: goals stay this far inside the map edge (a wall); the first zone is bigger than the map
MECH_SOUNDS = {"gun", "mortar", "laser", "laser_charge", "lightning", "step", "boost", "reload", "jet"}
WEAPON_VALUE = {"laser": 1.0, "lightning": 0.7, "gun": 0.5}  # never a mortar: a laser hit on its carrier cooks off (+60%)
EARLY_S, EARLY_RANGE = 60.0, 250.0  # the first minute: walk this far for a laser (else a lightning)
GUN_OPEN = 100.0     # m: the gun doesn't start a fight with a mech farther than this (it joins one at any range)
LATENCY = 0.03       # s from an observation to the shot leaving (the action applies on the next ticks)
GUN_TOL = 1.0        # m: the gun fires only when its aim is this close to the lead point, at the target
GUN_MIN_P = 0.15     # the gun holds fire below this estimated hit chance
LASER_MARGIN = 0.8   # m: the beam must point this far inside the target's edge when we let go
KILL_MARGIN = 0.75   # share of a beam's damage sure to land: 3 armor upgrades take 24%
DROP_RANGE = 250.0   # m: farthest we go for a supply drop (3 cubes, full HP and energy)
ROAM_CELL = 40.0     # m: cells of the visited map; roaming heads for the one we left longest ago
ROAM_REPICK = 25.0   # s: give up on a roam waypoint after this
HIDE_HP, HEALED_HP = 0.25, 0.9  # HP fraction: hide below the first, go back out at the second
FIGHT_RANGE = 150.0  # m: a seen mech this close (or one shooting us) stops the roam: stand and strafe
ROOF_Z = 3.0         # m above the terrain: an item this high is on a roof (a jump clears 2.75 m)
ROOF_MAX = 45.0      # m: highest roof we jet up to for loot (100 energy climbs about 60 m)
ROOF_CALM = 120.0    # m: climb only with no known mech this close (in the air we can't dodge)
ROOF_TIMEOUT = 30.0  # s: give up a roof trip after this (walk up to 200 m, then fly)
FLY_RANGE = 40.0     # m: walk this close to roof loot, then jet straight at it and drop onto it
RETREAT_RISK = 1.15  # engagement risk (damage we'd take / our HP) over which we break off the fight
RETREAT_HURT = 0.85  # ... or over this when below HURT_HP
HURT_HP = 0.45
RETREAT_OFF = 0.6    # back to fighting under this (and not before RETREAT_MIN_S)
RETREAT_MIN_S = 3.0
FINISH_TTK = 1.2     # s: a target we kill this fast is finished first, retreat or not
COVER_REACH = 2.5    # s of walking: cover farther than this is not worth crossing open ground under fire
SAFE_LOOT = 70.0     # m: loot or a drop with a foe remembered this close is not safe to fetch
SAFE_DROP = 130.0    # m: no foe seen or heard this close to a drop we stand in
ENERGY_RESERVE = 35  # energy kept for a dodge or an escape while foes are around
BUDGET_S = 0.030     # s of a decision after which the optional searches (cover, escape spots) wait
DEBUG = bool(__import__("os").environ.get("BOT_DEBUG"))
# Upgrade order, one entry per stack; with lightning in hand its two upgrades come right after laser_charge.
# gun_mag before the other gun ones: +8 rounds and -20% reload is +31% sustained gun dps (gun_damage +15%, gun_rate +8%).
UPGRADE_PLAN = ["plating", "laser_damage", "gun_mag", "nanites", "plating", "laser_damage", "laser_charge", "plating", "armor",
                "gun_mag", "gun_damage", "gun_rate", "sensors", "armor", "gyros"]
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
        self.strafe_off, self.strafe_stop = 0.0, False
        self.burn = True  # boosting in a fight: on until energy drops to 25, back on at 70
        self.vhist = {}  # target id -> [(t, its velocity)] over the last 2 s
        self.short_until = 0.0
        self.pickups = {}                # id -> (kind, pos, t seen)
        self.cubes = {}                  # id -> (pos, t seen)
        self.crates, self.npcs = {}, {}  # id -> (pos, t seen): what we can farm for cubes
        self.bad_pickups = {}            # id -> t until we stop trying it
        self.hit_by, self.hit_on = {}, {}  # id -> last time it damaged us / we damaged it
        self.tid = None                  # id of the target we shot at last decision (kept unless another is clearly better)
        self.dbg = {}
        self.visited, self.roam, self.roam_t = {}, None, -1e9  # cell -> last t there; current waypoint
        self.hiding, self.cover, self.cover_t = False, None, -1e9
        self.climb = None                # roof loot: {p: item, t0}
        self.pred = Predictor(self.mech.get("gravity", 22), self.world.ground_at if self.world else None)
        self.threats = Threats(cfg)
        self.risk, self.risk_info, self.ttk = 0.0, [], 0.0
        self.retreat, self.retreat_t = False, -1e9   # breaking off a fight we lose; since when
        self.escape, self.escape_t = None, -1e9      # where to (a spot out of their lines of fire), picked when
        self.drop_id = None                          # the drop we set off for
        self.errors = {}                             # part -> failures (logged the first few times)
        self.t0 = 0.0                                # perf_counter at the start of this decision
        self.tele = {"mode": {}, "taken": {}, "dealt": 0.0, "kills": 0, "retreats": 0, "late": 0, "slow": 0}
        self.mode_prev = None

    # ----- memory -----
    def safe(self, part, fn, *args, default=None):
        """fn(*args), or `default` if it raises: one broken part must not cost the whole decision."""
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            n = self.errors[part] = self.errors.get(part, 0) + 1
            if n <= 3:
                mf.log(f"{part} failed at t={self.t:.1f}: {e!r}")
            return default

    def busy(self):
        """True when this decision has used its budget: the optional searches keep their last answer."""
        return time.perf_counter() - self.t0 > BUDGET_S

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
        for k in obs["crates"]:
            self.crates[k["id"]] = (k["pos"], t)
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        for m in npcs:
            self.npcs[m["id"]] = (m["pos"], t)
        # Close by and not listed: destroyed (a drone 40 m off would be in sensor range unless a building is between).
        seen_k, seen_n = {k["id"] for k in obs["crates"]}, {m["id"] for m in npcs}
        self.crates = {k: v for k, v in self.crates.items() if (k in seen_k or d2(v[0], pos) > 25) and t - v[1] < FARM_S}
        self.npcs = {k: v for k, v in self.npcs.items() if (k in seen_n or d2(v[0], pos) > 40) and t - v[1] < FARM_S}
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
        self.safe("threats", self.threats.update, t, obs, pos)
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
        busy = {b["by"] for b in obs["beams"]}

        def can_hit(x):  # some part of it in the open, within the range one of our weapons fires at (as weapons() gates them)
            joined = self.t - max(self.hit_by.get(x["id"], -1e9), self.hit_on.get(x["id"], -1e9)) < 5
            return x["los"] and any(x["d"] <= (GUN_OPEN if w["kind"] == "gun" and not joined else w["stats"].get("range", 0))
                                    for w in me["weapons"] if w and w["kind"] != "mortar")

        def score(x):  # lower is better: near, few HP to burn through (shots to kill), shooting us, the one we're on
            m = x["m"]
            away = abs(ang_diff(m["aim"][0], math.atan2(muzzle[1] - x["pos"][1], muzzle[0] - x["pos"][0]))) > math.pi / 2
            return (x["d"] / 100 + m.get("hp", 1400) / 1000 - 0.8 * away - 0.4 * (x["id"] in busy)
                    - 1.0 * (self.t - self.hit_by.get(x["id"], -1e9) < 3) - 0.5 * (x["id"] == self.tid))

        mechs = sorted((x for x in mechs if x["d"] < 300), key=lambda x: x["d"])[:6]
        for x in mechs:  # "los": some part of it (middle, head and shoulders, legs) is in the open
            x["los"] = self.beam_point(muzzle, x, 0.0) is not None
        npc = next((x for x in others if x["kind"] == "npc" and self.t - self.hit_by.get(x["id"], -1e9) < 3), None)
        if npc and not any(self.t - self.hit_by.get(x["id"], -1e9) < 3 for x in mechs):  # a drone is shooting us: answer it
            npc["los"] = self.beam_point(muzzle, npc, 0.0) is not None
            return npc
        if mechs:  # one we can hit now beats the rest; else the best of them (the aim waits where it will show)
            return min([x for x in mechs if can_hit(x)] or mechs, key=score)
        if hpf < 0.6:
            return None
        for x in sorted(others, key=lambda x: (x["kind"] != "npc", x["d"])):
            if x["d"] < (150 if x["kind"] == "npc" else 120) and (x["kind"] == "npc" or me["cubes"] < CUBE_KEEP or None in me["weapons"]):
                x["los"] = self.beam_point(muzzle, x, 0.0) is not None
                if x["los"]:
                    return x
        return None

    def weapons(self, me, muzzle, tube, T, t):
        """(aim or None, fire[2], reload).

        Each weapon gets its own aim point from the motion predictor: the gun leads by its flight time,
        the laser and lightning by the decision latency. One torso serves all of them, so it follows the
        gun's lead while a laser charges and swings to the laser's point just before letting go (at
        4 rad/s the swing takes a tick). Shots are gated in meters at the target, not in angle, and the
        gun only fires down a corridor that is clear around its actual path."""
        ws, now = me["weapons"], me["aim"]
        aims, ok, dists = [None, None], [False, False], [1.0, 1.0]
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
                if k == "gun":
                    pt, p_hit = self._gun_dbg = self.gun_point(muzzle, T, w)
                    joined = T["kind"] != "mech" or t - max(self.hit_by.get(T["id"], -1e9), self.hit_on.get(T["id"], -1e9)) < 5
                    if pt:
                        aims[i], dists[i] = aim_at(muzzle, pt), math.dist(muzzle, pt)
                    ok[i] = (pt is not None and T["los"] and T["d"] <= (st["range"] if joined else GUN_OPEN) and w.get("ammo", 1) > 0
                             and not w.get("reload") and p_hit >= GUN_MIN_P)  # the gun misses a lot far off and its noise carries 320 m
                elif k == "laser":
                    pt = self.beam_point(muzzle, T)
                    if pt:
                        aims[i], dists[i] = aim_at(muzzle, pt), math.dist(muzzle, pt)
                    else:
                        aims[i] = aim_at(muzzle, T["c"])
                    ok[i] = pt is not None and T["los"] and T["d"] <= st["range"] and t >= self.short_until
                elif k == "lightning":
                    pt = self.point_at(T, LATENCY, T["c"][2] - T["pos"][2])
                    aims[i], dists[i] = aim_at(muzzle, pt), math.dist(muzzle, pt)
                    ok[i] = T["los"] and dists[i] <= st["range"] * 0.95 and me["energy"] >= st["energy"] + 5
                elif k == "mortar":
                    # lead by its mean velocity over 2 s: a strafer's shell lands on the middle of its dance (aim-range tested)
                    sol = mortar_aim(tube, [T["pos"][0], T["pos"][1], T["pos"][2] + 0.5], vavg(2.0), st["speed"],
                                     st["gravity"], high=not T["los"], max_pitch=self.mech["max_pitch"])
                    if sol:
                        aims[i] = sol[0]
                    ok[i] = sol is not None and d2(T["pos"], muzzle) >= MORTAR_MIN
                    dists[i] = T["d"]

        def ready(i):
            return ok[i] and ws[i]["cooldown"] <= 0.3

        def slot(kind):
            return next((i for i, w in enumerate(ws) if w and w["kind"] == kind), None)
        g, la, li, mo = slot("gun"), slot("laser"), slot("lightning"), slot("mortar")
        need = self.laser_need(ws[la], T, me) if la is not None else 0.0  # charge to let go at
        charging = la is not None and ws[la].get("charge", 0) > 0
        if charging:  # follow the gun's lead while it charges; on its point for the last 0.15 s and after
            full = ws[la]["charge"] >= need - 0.15
            p = la if (full or g is None or not ready(g)) and aims[la] else g if g is not None and ready(g) else la
        elif T and not T["los"]:
            p = mo if mo is not None and ready(mo) else None
        else:
            p = next((i for i in (g, la, li, mo) if i is not None and ready(i)), None)
        if p is None and T:
            p = g if g is not None and aims[g] else next((i for i in (0, 1) if aims[i]), None)
        aim = aims[p] if p is not None else None

        # Where the torso will point when the shots leave: it slews toward `aim` first (turn_rate per tick).
        dt = 1.0 / self.cfg["match"].get("tick_hz", 20)
        st_ = me["stats"]
        then = now
        if aim is not None:
            then = [now[0] + max(-st_.get("turn_rate", 4) * dt, min(st_.get("turn_rate", 4) * dt, ang_diff(aim[0], now[0]))),
                    now[1] + max(-st_.get("pitch_rate", 3) * dt, min(st_.get("pitch_rate", 3) * dt, aim[1] - now[1]))]
        fire = [False, False]
        for i, w in enumerate(ws):
            if w and aims[i] is None and w["kind"] == "laser" and w.get("charge", 0) > 0:
                fire[i] = True  # no target in sight: keep the charge (letting go would fire it into nothing)
            if not w or aims[i] is None:
                continue
            k, err = w["kind"], aim_error(now, aims[i])
            off = err * dists[i]  # meters off the aim point, at the target
            if k == "laser":
                if w.get("charge", 0) > 0:  # hold until charged enough (full, or enough to kill) and on target, then let go
                    full = w["charge"] >= need - 0.05
                    on = aim_error(then, aims[i]) * dists[i] < T["r"] - LASER_MARGIN and off < T["r"] + 1.0
                    fire[i] = not (full and ok[i] and on)
                else:  # start only if some of it will still be in the open when it is charged
                    fire[i] = ok[i] and w["cooldown"] <= 0 and err < 0.35 and self.beam_point(muzzle, T, need) is not None
            elif k == "lightning":
                fire[i] = ok[i] and w["cooldown"] <= 0.1 and abs(ang_diff(now[0], aims[i][0])) < math.radians(w["stats"]["cone_deg"]) / 2 * 0.8
            elif k == "gun":  # on the lead point once the torso has slewed, and not wildly off now
                fire[i] = ok[i] and w["cooldown"] <= 0.1 and aim_error(then, aims[i]) * dists[i] < GUN_TOL and off < T["r"] + GUN_TOL
                if DEBUG and T and T["kind"] == "mech" and T["los"]:
                    why = ("fire" if fire[i] else "reload" if w.get("reload") or not w.get("ammo", 1) else "cooldown" if w["cooldown"] > 0.1
                           else "nopoint" if self._gun_dbg[0] is None else "p" if self._gun_dbg[1] < GUN_MIN_P
                           else "range" if not ok[i] else "tol")
                    self.dbg[why] = self.dbg.get(why, 0) + 1
            else:
                fire[i] = ok[i] and w["cooldown"] <= 0.1 and err < math.atan2(3, T["d"]) + 0.01
        gun = ws[g] if g is not None else None
        calm = T is None or T["kind"] != "mech"
        reload = bool(gun) and not gun.get("reload") and (gun.get("ammo", 1) == 0 or (calm and gun.get("ammo", 0) < gun["stats"]["mag"] / 2))
        return aim, fire, reload

    def laser_need(self, w, T, me):
        """Seconds of laser charge whose beam kills T even through full armor (a full charge if it can't)."""
        st, cu = w["stats"], self.cfg["cubes"]
        hp = T["m"].get("hp") if T else None
        if hp is None:
            return st["charge"]
        dmg = st["damage"] * (1 + cu.get("damage_per_cube", 0.04) * min(me.get("cubes", 0), cu.get("stat_cap", 10))) * KILL_MARGIN
        lo = st.get("min_frac", 0.35)  # damage share at no charge, rising linearly to 1 at full
        return min(max((hp / dmg - lo) / (1 - lo) * st["charge"], st.get("min_charge", 0.15)), st["charge"])

    def point_at(self, T, tau, hgt):
        """T's predicted position after tau s, hgt m above its feet."""
        p, _ = self.pred.predict(T["id"], T["pos"], T["vel"], tau)
        return [p[0], p[1], p[2] + hgt]

    def gun_point(self, muzzle, T, w):
        """(point to aim the gun at, estimated hit chance), or (None, 0) if no line of fire is clear.
        Leads by the predicted flight time; tries the body's middle, then higher and lower, and takes the
        first height whose corridor (the path, and the path shifted by one spread sigma each way) is clear."""
        st = w["stats"]
        lat = LATENCY + max(w["cooldown"], 0.0)
        spc = self.cfg["cubes"].get("size_per_cube", 0.05)
        hgt = self.mech["height"] * (1 + spc * T["m"].get("cubes", 0)) if T["kind"] == "mech" else T["c"][2] - T["pos"][2]
        mid = T["c"][2] - T["pos"][2]
        tau, p, info = lat + max(T["d"] - 3, 0) / st["speed"], T["pos"], None
        for _ in range(3):
            p, info = self.pred.predict(T["id"], T["pos"], T["vel"], tau)
            tau = lat + max(math.dist(muzzle, [p[0], p[1], p[2] + mid]) - 3, 0) / st["speed"]
        d = math.dist(muzzle, [p[0], p[1], p[2] + mid])
        sig = max(0.4, d * math.radians(st.get("spread_deg", 0.6)))
        r = T["r"] + self.cfg["projectile"].get("gun_radius", 0.3)
        p_pred = info[1] if info[1] is not None else 0.7
        p_spread = math.erf(r / (sig * math.sqrt(2)))
        if not self.world:
            return [p[0], p[1], p[2] + mid], p_pred * p_spread
        side = unit(-(p[1] - muzzle[1]), p[0] - muzzle[0])
        for z in (mid, min(mid + 1.5, hgt - 1.5), max(mid - 1.5, 1.5)):
            pt = [p[0], p[1], p[2] + z]
            if not self.los(muzzle, pt):
                continue
            shifts = ([0, 0, -sig], [0, 0, sig], [side[0] * sig, side[1] * sig, 0], [-side[0] * sig, -side[1] * sig, 0])
            clear = sum(self.los(muzzle, [pt[j] + sh[j] for j in range(3)]) for sh in shifts)
            if clear >= 3:
                return pt, p_pred * p_spread * (0.75 + 0.0625 * clear)
        return None, 0.0

    def beam_point(self, muzzle, T, tau=LATENCY):
        """Where to point the laser: T's body predicted tau s ahead (where it is now for 0), at the middle, else
        the highest or lowest part of it a beam can reach (a mech half behind a wall or a crest). None: all hidden."""
        mid = T["c"][2] - T["pos"][2]
        top = 2 * mid
        p = self.pred.predict(T["id"], T["pos"], T["vel"], tau)[0] if tau else T["pos"]
        for z in (mid, top - 1.2, top - 2.5, 1.5):
            pt = [p[0], p[1], p[2] + z]
            if self.los(muzzle, pt):
                return pt
        return None

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

    def early_laser(self, me, pos, enemies, known=()):
        """In the first minute: the nearest laser seen within EARLY_RANGE (else a lightning) we want and can reach first."""
        for kind in ("laser", "lightning"):
            best = min(((d2(p, pos), p) for pid, (k, p, _) in self.pickups.items()
                        if k == kind and self.bad_pickups.get(pid, -1) < self.t and self.wants(k, me)[0]
                        and d2(p, pos) < EARLY_RANGE and abs(p[2] - pos[2]) < 4  # not on a roof
                        and not any(d2(m["pos"], p) < d2(p, pos) for m in enemies)  # nobody closer to it
                        and not any(d2(self.threats.where(f), p) < min(d2(p, pos), SAFE_LOOT) for f in known)), default=None)
            if best:
                return best[1]
        return None

    # ----- the trade -----
    def our_dps(self, me, f, d):
        """Our damage per second on foe f, d m off, with what we hold now (upgraded stats, cubes)."""
        th, out = self.threats, 0.0
        for w in me["weapons"]:
            if not w:
                continue
            k, st = w["kind"], w["stats"]
            base = self.cfg["weapons"][k].get("damage", 1) or 1
            if k == "laser" and self.t < self.short_until:
                continue
            if k == "lightning" and me["energy"] < st.get("energy", 30):
                continue
            if k == "mortar":
                continue
            if d > st.get("range", 230) + 5:
                continue
            out += th.weapon_dps(k, d, me.get("cubes", 0)) * st.get("damage", base) / base
        return out

    def fire_line(self, f, fp, at=None):
        """Has foe f (standing at fp) a line of fire on us (or on `at`, a spot on the ground)?"""
        if at is None:
            q = self.me_mid
        else:
            gz = self.world.ground_at(at[0], at[1], self.me_mid[2] - 4.5) if self.world else self.me_mid[2] - 4.5
            q = [at[0], at[1], gz + self.mech["height"] / 2]
        return self.los([fp[0], fp[1], fp[2] + self.mech.get("muzzle_height", 7)], q)

    def assess(self, me, pos, T):
        """combat_risk_score: damage we expect to take fighting everyone who fights us, over our HP (1 = we die)."""
        yaw = math.atan2(T["pos"][1] - pos[1], T["pos"][0] - pos[0]) if T else me["aim"][0]
        cache = {}

        def los(f, fp):
            if f.id not in cache:
                cache[f.id] = self.fire_line(f, fp)
            return cache[f.id]
        risk, info, tt = self.threats.assess(me, pos, yaw, lambda f, d: self.our_dps(me, f, d), los)
        self.risk, self.risk_info, self.ttk = risk, info, tt
        return risk

    def ttk_on(self, me, T):
        """Seconds we need to kill T with what we hold (inf if we can't hurt it)."""
        if not T or T["kind"] != "mech":
            return 1e9
        dps = self.our_dps(me, None, T["d"])
        return T["m"].get("hp", 1400) / dps if dps > 0 else 1e9

    def escape_spot(self, pos, c, r, T=None, keep_sight=False):
        """Minimum-risk spot within ~45 m: walkable, inside the safe circle, the least exposed to the foes we know
        (their damage rates, more when close, much less where they have no line of fire), not far to walk.
        keep_sight: it must keep a line of fire on T (fighting from it, hidden from the others)."""
        t = self.t
        if self.escape and t - self.escape_t < 0.6 and d2(self.escape, pos) > 2.0:
            return self.escape
        if self.busy() and self.escape:
            self.tele["slow"] += 1
            return self.escape
        rr = max(r - EDGE_MARGIN, 0.0)
        th = self.threats
        foes = sorted(th.known(), key=lambda f: d2(th.where(f), pos))[:5]
        walk = self.mech.get("walk_speed", 11.7)
        muz = self.mech.get("muzzle_height", 7)
        cands = [list(pos[:2])]
        for rad in (14.0, 28.0, 45.0):
            for k in range(8):
                a = k * math.pi / 4 + (rad / 14.0) * 0.2
                cands.append([pos[0] + rad * math.cos(a), pos[1] + rad * math.sin(a)])
        if self.escape:
            cands.append(self.escape)
        best, bs, prev_s = None, None, None
        for p in cands:
            if self.world:
                if p is not cands[0] and (not self.world.free(*p) or self.world.clamp(p, MAP_MARGIN) != p):
                    continue
            if d2(p, c) > rr and d2(p, c) > d2(pos, c):
                continue  # out of the safe circle and no nearer it
            sc = 0.0
            for f in foes:
                fp = th.where(f)
                dd = max(d2(fp, p) - th.spread(f) * 0.3, 5.0)
                if dd > 300:
                    continue
                w = th.foe_dps(f, dd) * (0.6 if f.ghost else 1.0) * max(0.3, th.awareness(f, pos, True))
                if not f.ghost and t - f.t_seen < 4 and self.world:
                    w *= 1.0 if self.fire_line(f, fp, p) else 0.15
                sc += w * (1.0 + max(0.0, 120.0 - dd) / 60.0)
            far = d2(p, pos)
            sc += far / walk * 12.0  # every second of walking in the open costs
            if self.world and far > 3 and not self.world.line_free(pos, p):
                sc += 40.0  # a detour round a wall
            if d2(p, c) > rr:
                sc += 200.0
            if keep_sight and T:
                q = [p[0], p[1], (self.world.ground_at(p[0], p[1], pos[2]) if self.world else pos[2]) + muz]
                if not self.los(q, T["c"]):
                    sc += 150.0
            if p is self.escape:
                prev_s = sc
            if bs is None or sc < bs:
                best, bs = p, sc
        if prev_s is not None and bs is not None and prev_s <= bs * 1.15 + 5:
            best = self.escape  # keep the plan unless another spot is clearly better
        self.escape, self.escape_t = best, t
        return best

    # ----- movement -----
    def strafe(self, pos, T, goal, gap, anchor_pull, safe_r, c, t):
        """A strafe step across the line of fire of the foe that hurts us most (the target unless another shooter
        out-damages it). Not a metronome: each leg lasts 0.2-1.0 s, a reversal is skipped now and then, the
        heading wanders +-30 deg off the perpendicular, and a rare short stop breaks the rhythm. A leg that would
        run into a wall or out of the zone turns round at once."""
        x, y = pos[0], pos[1]
        src = T["pos"]
        best = max((e for e in self.risk_info if e[4] >= 0.8 and not e[0].ghost), key=lambda e: e[2], default=None)
        if best and best[2] > 1.3 * next((e[2] for e in self.risk_info if e[0].id == T["id"]), 0.0):
            src = self.threats.where(best[0])
        if t >= self.strafe_until:
            r = self.rng.random()
            if r < 0.1:
                self.strafe_stop, dur = True, self.rng.uniform(0.15, 0.3)
            else:
                self.strafe_stop = False
                if r < 0.85:
                    self.strafe_sign = -self.strafe_sign
                dur = self.rng.uniform(0.2, 0.45) if self.rng.random() < 0.35 else self.rng.uniform(0.45, 1.0)
            self.strafe_off, self.strafe_until = self.rng.uniform(-0.5, 0.5), t + dur
        yaw = math.atan2(src[1] - y, src[0] - x) + self.strafe_sign * (math.pi / 2 + self.strafe_off)
        u = [math.cos(yaw), math.sin(yaw)]
        probe = [x + u[0] * 7, y + u[1] * 7]
        if (self.world and not self.world.line_free(pos, probe)) or d2(probe, c) > safe_r:
            self.strafe_sign, self.strafe_off = -self.strafe_sign, 0.0
            self.strafe_until, self.strafe_stop = t + self.rng.uniform(0.3, 0.7), False
            yaw = math.atan2(src[1] - y, src[0] - x) + self.strafe_sign * math.pi / 2
            u = [math.cos(yaw), math.sin(yaw)]
        pull = min(gap / 6, 1.0) * anchor_pull
        to_spot = unit(goal[0] - x, goal[1] - y)
        if self.strafe_stop and pull < 0.5:
            return [to_spot[0] * pull, to_spot[1] * pull]
        return unit(u[0] + to_spot[0] * pull, u[1] + to_spot[1] * pull)

    def dodge(self, obs, me, pos):
        """A move vector out of every incoming shell's splash, or None. Tries 16 directions and standing still:
        where we would be when each shell lands (walking or boosting, after the turn to go), kept on walkable
        ground and in the zone; takes the one farthest out of the nearest splash. (No laser dodge: a hitscan beam
        can't be sidestepped.)"""
        mo = self.cfg["weapons"]["mortar"]
        R = mo.get("splash_radius", 8) * 1.3 + 3
        lands = []
        for p in obs["projectiles"]:
            if p["kind"] == "mortar":
                land = shell_landing(p["pos"], p["vel"], mo["gravity"], pos[2] + 1)
                if land and land[2] < 3 and d2(land, pos) < R + 12:
                    lands.append(land)
        if not lands or not any(d2(l, pos) < R + l[2] * 6 for l in lands):
            return None
        st = me["stats"]
        v = st.get("boost_speed", self.mech.get("boost_speed", 16)) if me["energy"] > 15 else st.get("walk_speed", 11.7)
        c, r = zone_at(self.zone, self.t + 1)
        best, bs = None, None
        for k in range(17):
            u = [0.0, 0.0] if k == 16 else [math.cos(k * math.pi / 8), math.sin(k * math.pi / 8)]
            reach = 0.0
            if k < 16:
                reach = v * max(min(l[2] for l in lands) - 0.12, 0.0)
                if self.world:
                    probe = [pos[0] + u[0] * min(reach, 8) + u[0] * 2, pos[1] + u[1] * min(reach, 8) + u[1] * 2]
                    if not self.world.free(*probe):
                        continue
            clear = min(d2([pos[0] + u[0] * v * max(l[2] - 0.12, 0), pos[1] + u[1] * v * max(l[2] - 0.12, 0)], l) for l in lands) \
                if k < 16 else min(d2(pos, l) for l in lands)
            sc = min(clear, R + 2)  # beyond the splash every direction is as good: prefer ...
            end = [pos[0] + u[0] * reach, pos[1] + u[1] * reach]
            sc -= 0.5 * max(d2(end, c) - r, 0)  # ... staying in the zone
            sc -= 0.02 * self.threats.danger_at(end) if k < 16 else 0.0  # ... and not stepping toward the others
            if bs is None or sc > bs:
                best, bs = u, sc
        if best is None or best == [0.0, 0.0]:
            return None
        return best, "dodge shell"

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

    def roam_goal(self, pos, c, r, t):
        """A walkable waypoint inside the safe circle, in the cell we have been away from longest
        (unvisited first, then nearest), so we sweep the map for loot instead of holding the edge."""
        self.visited[(int(pos[0] // ROAM_CELL), int(pos[1] // ROAM_CELL))] = t
        rr = max(r - EDGE_MARGIN - 5, 0.0)
        g = self.roam
        if g and d2(g, pos) > 8 and t - self.roam_t < ROAM_REPICK and d2(g, c) < rr:
            return g
        best, bs = None, None
        for _ in range(32):
            a, k = self.rng.uniform(0, 2 * math.pi), math.sqrt(self.rng.random())
            p = [c[0] + math.cos(a) * rr * k, c[1] + math.sin(a) * rr * k]
            if self.world:
                p = self.world.clamp(p, MAP_MARGIN)
                if not self.world.free(*p):
                    continue
            if d2(p, c) >= rr or d2(p, pos) < 30:
                continue
            s = self.visited.get((int(p[0] // ROAM_CELL), int(p[1] // ROAM_CELL)), -300.0) + d2(p, pos) / 10  # 10 m ~ 1 s
            if bs is None or s < bs:
                best, bs = p, s
        self.roam, self.roam_t = best, t
        return best

    def cover_spot(self, pos, c, r, enemies, t):
        """A walkable spot within 40 m, inside the safe circle, out of sight of every known enemy (and of a
        shooter we only felt, assumed 80 m out along the hit's bearing). Our own spot if nobody is known."""
        eye = self.mech.get("eye_height", 8.0)
        ws = sorted(enemies, key=lambda m: d2(m["pos"], pos))[:4]
        watchers = [[m["pos"][0], m["pos"][1], m["pos"][2] + eye] for m in ws]
        if self.threat and t - self.threat[1] < 5:
            b = self.threat[0]
            watchers.append([pos[0] + 80 * math.cos(b), pos[1] + 80 * math.sin(b), pos[2] + eye])
        if not watchers or not self.world:
            return list(pos[:2])
        mid = self.mech["height"] / 2

        def hidden(p):
            q = [p[0], p[1], self.world.ground_at(p[0], p[1]) + mid]
            return not any(self.los(w, q) for w in watchers)
        if t - self.cover_t < 1.0:  # the search costs line-of-sight rays: redo it once a second (None included)
            return self.cover
        if self.cover and hidden(self.cover):
            self.cover_t = t
            return self.cover
        rr = max(r - EDGE_MARGIN, 0.0)
        for dist in (0, 8, 16, 28, 40):
            for i in range(12 if dist else 1):
                a = i * math.pi / 6
                p = [pos[0] + dist * math.cos(a), pos[1] + dist * math.sin(a)]
                if (d2(p, c) < rr and self.world.free(*p) and hidden(p)
                        and all(d2(m["pos"], p) > d2(m["pos"], pos) - 5 for m in ws)):  # not toward them
                    self.cover, self.cover_t = p, t
                    return p
        self.cover, self.cover_t = None, t
        return None

    def danger_map(self, pos):
        """Extra A* cost around the foes we remember within 250 m (weighted by their damage rate), or None."""
        th = self.threats
        spots = []
        for f in th.known(12.0):
            fp = th.where(f)
            if d2(fp, pos) < 250:
                spots.append((fp[0], fp[1], min(th.foe_dps(f, 60) / 150.0, 2.0) * (0.6 if f.ghost else 1.0)))
        if not spots:
            return None
        return self.world.danger_cells(sorted(spots, key=lambda e: d2(e, pos))[:4])

    def steer(self, pos, goal, t, force=False, careful=False):
        """Unit move vector toward goal along an A* path (careful: the path keeps away from the foes we know)."""
        if not self.world:
            return unit(goal[0] - pos[0], goal[1] - pos[1])
        if force or not self.path or self.path_goal is None or d2(goal, self.path_goal) > 6 or t - self.path_t > 2.5:
            extra = self.safe("danger", self.danger_map, pos) if careful else None
            self.path, self.path_goal, self.path_t = self.world.path(pos, goal, PATH_BUDGET, extra), list(goal), t
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

    def loot_items(self, me):
        """Remembered loot we want: weapon pickups we'd take, and cubes while we still gain from them."""
        out = [p for pid, (kind, p, _) in self.pickups.items() if self.bad_pickups.get(pid, -1) < self.t and self.wants(kind, me)[0]]
        return out + ([p for p, _ in self.cubes.values()] if me["cubes"] < CUBE_KEEP else [])

    def on_roof(self, p):
        """Up on a building: above the walkable surface (ground or bridge deck; under a deck a jet only hits it)."""
        w = self.world
        if not w or w.free(p[0], p[1]):
            return False
        i = w._idx(p[0], p[1])
        return i >= 0 and p[2] - w.surf[i] > ROOF_Z

    def loot_goal(self, me, pos, c, r, safe=None):
        """The cheapest wanted loot on the ground (roof loot is roof_target's): distance, longer when a wall is in
        the way, and only where `safe(p)` says no foe we remember is near it."""
        best, bs = None, None
        for p in self.loot_items(me):
            dd = d2(p, pos)
            if d2(p, c) >= r - 6 or dd >= LOOT_RANGE or self.on_roof(p):
                continue
            cost = dd * (1.0 if not self.world or dd < 8 or self.world.line_free(pos, p) else 1.5)
            if bs is not None and cost >= bs:
                continue
            if safe is not None and not safe(p):
                continue
            best, bs = p, cost
        return best

    def farm_goal(self, me, pos, c, r, enemies):
        """The remembered drone or crate on the ground worth the most cubes per metre within LOOT_RANGE, inside the circle,
        with no mech we see near it; targets() shoots it on the way and loot_goal picks up the cubes it drops."""
        if me["cubes"] >= CUBE_KEEP:
            return None
        n, k = self.cfg["npc"].get("cubes", 2), self.cfg["crates"].get("cubes", 1)
        spots = [(p, n) for p, _ in self.npcs.values()] + [(p, k) for p, _ in self.crates.values()]
        return min(((d2(p, pos) / v, p) for p, v in spots if d2(p, c) < r - 6 and d2(p, pos) < LOOT_RANGE and not self.on_roof(p)
                    and not any(d2(m["pos"], p) < FARM_GUARD for m in enemies)), key=lambda e: e[0], default=(0, None))[1]

    def roof_target(self, me, pos, c, r):
        """The nearest wanted loot up on a roof we have the energy to jet onto, or None."""
        def ok(p):
            return self.on_roof(p) and p[2] - pos[2] <= ROOF_MAX and d2(p, c) < r - 6 and d2(p, pos) < LOOT_RANGE
        return min((p for p in self.loot_items(me) if ok(p)), key=lambda p: d2(p, pos), default=None)

    def launch_spot(self, p):
        """The walkable cell nearest to p (the way down off a roof), or None."""
        for dist in range(2, 32, 2):
            for i in range(16):
                q = [p[0] + dist * math.cos(i * math.pi / 8), p[1] + dist * math.sin(i * math.pi / 8)]
                if self.world.free(*q):
                    return q
        return None

    # ----- decision -----
    def account(self, obs, mode, t):
        """Decision telemetry: time and damage taken per mode, a line every 60 s (stderr, seen after the match)."""
        tl = self.tele
        tl["mode"][mode] = tl["mode"].get(mode, 0) + 1
        for e in obs["events"]:
            if e["type"] == "damaged":
                k = mode if e.get("weapon") != "zone" else "zone"
                tl["taken"][k] = tl["taken"].get(k, 0.0) + e["amount"]
            elif e["type"] == "hit":
                tl["dealt"] += e["amount"]
            elif e["type"] == "kill":
                tl["kills"] += 1
        if mode != self.mode_prev and mode in ("retreat", "hide") and DEBUG:
            mf.log(f"t={t:.1f} {mode} risk {self.risk:.2f} hp {obs['self']['hp']:.0f} foes {len(self.risk_info)}")
        self.mode_prev = mode
        if int(t) % 60 == 0 and int(t) != int(t - 0.1):
            self.report(t)

    def report(self, t=None):
        tl = self.tele
        modes = " ".join(f"{k}:{v / 10:.0f}s" for k, v in sorted(tl["mode"].items(), key=lambda e: -e[1]))
        taken = " ".join(f"{k}:{v:.0f}" for k, v in sorted(tl["taken"].items(), key=lambda e: -e[1]))
        mf.log(f"[t={self.t if t is None else t:.0f}] modes {modes} | taken {taken} | dealt {tl['dealt']:.0f} kills {tl['kills']} "
               f"retreats {tl['retreats']} slow {tl['slow']} errors {sum(self.errors.values())}")

    def act(self, obs: mf.Observation) -> mf.Action:
        self.t0 = time.perf_counter()
        me = obs["self"]
        pos = me["pos"]
        x, y, z = pos
        t = self.t = obs["t"]
        self.zone = obs["zone"]
        self.remember(obs, t, pos)
        spc = self.cfg["cubes"].get("size_per_cube", 0.05)
        for m in obs["mechs"]:  # feed the motion predictor (and score it in hindsight)
            r = self.mech["radius"] * (1 + spc * m.get("cubes", 0)) if m["kind"] == "mech" else self.cfg["npc"]["radius"]
            self.pred.update(t, m["id"], m["pos"], m["vel"], r + 0.3, pos)
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

        # Who is dangerous, and do we win the trade?
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        th = self.threats
        known = th.known(12.0)
        self.me_mid = [x, y, z + self.mech["height"] / 2]
        T = self.targets(obs, me, muzzle, hpf)
        self.tid = T["id"] if T else None
        risk = self.safe("assess", self.assess, me, pos, T, default=self.risk)
        aim, fire, reload = self.weapons(me, muzzle, tube, T, t)
        if self.turn_to and t - self.turn_to[1] < 1.0 and not (T and T["kind"] == "mech" and T["los"] and T["id"] in self.hit_by
                                                              and t - self.hit_by[T["id"]] < 1.0):
            aim = [self.turn_to[0], 0.0]  # hit in the back (x1.5) by someone we don't have in front of us: turn round now
            fire = [bool(w and w["kind"] == "laser" and w.get("charge", 0) > 0) for w in me["weapons"]]  # nothing leaves off-aim; a charge is held

        # Disengage: break off a trade we lose (risk over 1: we die before they do), unless the target is all but dead.
        ttk = self.ttk_on(me, T) if T and T["kind"] == "mech" and T["los"] else 1e9
        losing = risk > RETREAT_RISK or (risk > RETREAT_HURT and hpf < HURT_HP)
        if losing and ttk > FINISH_TTK and not self.retreat:
            self.retreat, self.retreat_t = True, t
            self.tele["retreats"] += 1
        elif self.retreat and (risk < RETREAT_OFF or ttk < FINISH_TTK * 0.6) and t - self.retreat_t > RETREAT_MIN_S:
            self.retreat = False

        if hpf < HIDE_HP:
            self.hiding = True
        elif hpf >= HEALED_HP:
            self.hiding = False
        if self.hiding and not (T and T["kind"] == "mech" and (t - self.hit_by.get(T["id"], -1e9) < 5 or T["d"] < 40)):
            # regen needs 5 s without dealing damage too: shoot only whoever is on us; a charge is held
            fire = [bool(w and w["kind"] == "laser" and w.get("charge", 0) > 0) for w in me["weapons"]]

        # Movement goal. Priorities: dodge a shell > back into the zone > break off a lost fight > fight > hide to heal
        # > a supply drop > a laser in the first minute > loot > roof loot > farm > roam. Every trip is safe-checked
        # against the foes we remember (seen in the last 12 s, or heard), not only the ones in view.
        c, r = zone_at(obs["zone"], t + LOOKAHEAD_S)
        spot = self.free_edge_spot(pos, c, r)
        safe_r = max(r - EDGE_MARGIN, 0.0)
        arrive, mode, goal, move = ARRIVED, "edge", spot, None
        dodge = self.safe("dodge", self.dodge, obs, me, pos)
        anchor_pull = 1.0

        def safe_trip(p, rad):  # no remembered foe near the spot (nor visible one in the way)
            return not any(d2(th.where(f), p) < rad + th.spread(f) * 0.5 for f in known)
        if dodge:
            move, mode = dodge
        elif not me["in_zone"]:
            mode = "zone"
        else:
            shot_at = self.threat is not None and t - self.threat[1] < 3
            engaged = bool(T and T["kind"] == "mech" and T["los"]
                           and (T["d"] < FIGHT_RANGE or t - self.hit_by.get(T["id"], -1e9) < 3))
            drop = None
            if hpf >= 0.6 and risk < 0.3 and not engaged and not shot_at and not self.hiding:
                for dr in obs.get("drops", []):
                    dd = d2(dr["pos"], pos)
                    if (dr["holder"] in (None, me["id"]) and dd < DROP_RANGE and d2(dr["pos"], c) < r - 3
                            and dr["lands_in"] <= dd / st.get("walk_speed", 11.7) + 10
                            and not any(d2(m["pos"], dr["pos"]) < 60 for m in enemies)
                            and (safe_trip(dr["pos"], SAFE_DROP) or dr["id"] == self.drop_id)):
                        drop = dr
                        break
            self.drop_id = drop["id"] if drop else None
            loot = None
            if hpf >= 0.35 and (not engaged or risk < 0.5):
                loot = self.safe("loot", self.loot_goal, me, pos, c, r, lambda p: safe_trip(p, SAFE_LOOT))
                if loot and engaged and d2(loot, pos) > 20:
                    loot = None  # in a fight only what is a few steps away: strafe onto it
            early = self.early_laser(me, pos, enemies, known) if t < EARLY_S and not shot_at else None
            farm = None
            if hpf >= 0.6 and not self.hiding and risk < 0.2 and not engaged:
                farm = self.safe("farm", self.farm_goal, me, pos, c, r, enemies + [{"pos": th.where(f)} for f in known])
            calm = (not self.hiding and hpf >= 0.6 and not shot_at and not engaged and risk < 0.1
                    and not any(d2(th.where(f), pos) < ROOF_CALM for f in known))
            if self.climb and (t - self.climb["t0"] > ROOF_TIMEOUT or (me["grounded"] and not calm and not self.on_roof(pos))):
                self.climb = None  # trouble while still on the ground: forget the roof
            if not self.climb and calm and not (drop or early or loot):
                rp = self.roof_target(me, pos, c, r)
                if rp:
                    self.climb = dict(p=rp, t0=t)
            if self.retreat:
                esc = self.safe("escape", self.escape_spot, pos, c, r, T, default=None)
                if esc is not None and d2(esc, pos) <= COVER_REACH * st.get("walk_speed", 11.7) * 2:
                    goal, mode, arrive = esc, "retreat", 1.5
                    anchor_pull = 1.8  # mostly toward the spot, still crossing their line of fire
                else:
                    goal, mode = list(pos[:2]), "fight"
            elif engaged:
                goal, mode = list(pos[:2]), "fight"  # stand and trade, strafing
                if early or loot:
                    goal, arrive = (early or loot), 1.0  # a weapon or cube a few steps off: strafe onto it
                elif risk > 0.5 and len(self.risk_info) > 1:  # others around: fight from where only the target sees us
                    esc = self.safe("escape", self.escape_spot, pos, c, r, T, True, default=None)
                    if esc is not None and d2(esc, pos) < 30:
                        goal = esc
            elif self.hiding:
                hide = self.safe("cover", self.cover_spot, pos, c, r, enemies, t)
                if hide is None:  # no cover within reach: the least exposed spot we can get to, away from them
                    hide = self.safe("escape", self.escape_spot, pos, c, r, None, default=None)
                if hide is not None:
                    goal, mode, arrive = hide, "hide", 1.5
            elif drop:
                goal, mode, arrive = drop["pos"], "drop", max(drop["radius"] - 3, 1.0)
            elif early:
                goal, mode, arrive = early, "laser", 1.0
            elif loot:
                goal, mode, arrive = loot, "loot", 1.0
            elif self.climb:
                goal, mode, arrive = self.climb["p"], "roof", 1.0
            elif farm:
                goal, mode, arrive = farm, "farm", 6.0  # cubes for the upgrades: shoot it, then pick them up as loot
            else:
                roam = self.roam_goal(pos, c, r, t)
                if roam:
                    goal, mode, arrive = roam, "roam", 6.0

        gap = d2(goal, pos)
        climbing = False
        if move is None and mode == "roof":
            top, ms = goal[2], self.mech
            # jet time to get over it: rising to its height or crossing the gap, whichever is longer, plus a margin
            need = ms.get("jet_drain", 15) * (max((top - z) / ms.get("max_rise_speed", 10), gap / ms.get("walk_speed", 11.7)) + 1.5)
            if me["grounded"] and z < top - 1 and (gap > FLY_RANGE or me["energy"] < need):
                move = self.steer(pos, goal, t) if gap > FLY_RANGE else [0.0, 0.0]  # closer first / let energy refill
            else:
                move = unit(goal[0] - x, goal[1] - y)  # fly straight at it with the jet held...
                climbing = gap > 3 or z < top - 1      # ...until over it, then let go and drop onto it
                if gap < 1.5 and z > top - 1:
                    self.climb = None                  # there: the take / cube pickup happens on its own
        if move is None and mode != "roof" and self.on_roof(pos) and gap > arrive:
            down = self.launch_spot(pos)  # A* sees roofs as walls: walk off the nearest edge first
            if down:
                move = unit(down[0] - x, down[1] - y)
        if move is None:
            careful = mode in ("retreat", "hide", "loot", "laser", "drop", "farm") or hpf < 0.6
            move = self.steer(pos, goal, t, force=self.stuck_since is not None and t - self.stuck_since > 1,
                              careful=careful) if gap > arrive else [0.0, 0.0]
            if mode in ("edge", "fight", "retreat") and T and T["kind"] == "mech" and T["los"]:  # strafe across their line of fire
                move = self.strafe(pos, T, goal, gap, anchor_pull, safe_r, c, t)

        # Stuck: commanded to move but not getting anywhere -> jump and jet; if that fails too, sidestep.
        # Not while fighting inside the zone (hopping about loses the fight); outside it the zone kills faster than they do.
        fighting = bool(T and T["kind"] == "mech" and T["los"] and me["in_zone"])
        wants_move = math.hypot(*move) > 0.5
        self.hist = [h for h in self.hist if t - h[0] < 1.2] + [(t, x, y, wants_move)]
        h0 = self.hist[0]
        stuck = (len(self.hist) >= 10 and all(h[3] for h in self.hist) and math.hypot(x - h0[1], y - h0[2]) < 1.5
                 and not climbing)  # pressed to the wall while jetting up is the plan, not stuck
        self.stuck_since = (t if self.stuck_since is None else self.stuck_since) if stuck else None
        jump = climbing and me["grounded"]
        if t < self.unstick_until:
            move = self.unstick_dir
        elif stuck:
            jump = me["grounded"] and not fighting
            if t - self.stuck_since > 2.5:
                a = math.atan2(move[1], move[0]) + self.rng.choice((-1, 1)) * math.pi / 2
                self.unstick_dir, self.unstick_until = [math.cos(a), math.sin(a)], t + 1.0
                self.stuck_since, self.path = None, []
        # Energy: keep a reserve for a dodge or an escape while foes are around (more of them, more reserve).
        crowd = sum(1 for f in known if d2(th.where(f), pos) < 200)
        reserve = 0 if not crowd else ENERGY_RESERVE if crowd == 1 else ENERGY_RESERVE + 10
        boost = climbing or mode in ("dodge shell", "zone") or (stuck and not fighting) or (
            gap > 40 and me["energy"] > 60 + reserve * 0.5 and not T and mode != "roof")  # save the energy for the jet
        if mode in ("retreat", "hide") and gap > arrive and me["energy"] > 12:
            boost = boost or me["grounded"]  # getting out of their sight is what the energy is for
        if fighting:  # boost through the whole fight, walking just long enough to recharge (duel-tested; jets lost)
            lo = max(25, reserve)
            self.burn = me["energy"] > lo + 45 if not self.burn else me["energy"] > lo
            boost = boost or (self.burn and me["grounded"])

        # Where to look when there's nothing to shoot.
        if aim is None:
            inward = math.atan2(c[1] - y, c[0] - x)
            if t < 1.6:  # a full turn at spawn (turn_rate 4 rad/s): see the loot behind us too
                aim = [me["aim"][0] + 1.0, 0.0]
            elif self.threat and t - self.threat[1] < 3:  # turn to face whoever hit us
                aim = [self.threat[0], 0.0]
            elif self.heard and t - self.heard[1] < 2:
                aim = [self.heard[0], 0.0]
            elif mode in ("roam", "farm") and wants_move:  # scan: sweep +-1 rad across the way we walk
                aim = [math.atan2(move[1], move[0]) + math.sin(t * 2.0), 0.0]
            elif gap > 30 and wants_move:
                aim = [math.atan2(move[1], move[0]), 0.0]
            else:
                aim = [inward, 0.0]  # back to the zone edge: enemies come from inside

        act.update(move=move, boost=boost, jump=jump, aim=aim, fire=fire, reload=reload)
        self.account(obs, mode, t)
        if DEBUG and obs["tick"] % 200 == 0 and self.dbg:
            mf.log("gun", sorted(self.dbg.items()))
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


_BOTS = []


def make_bot(hello):
    b = Bot(hello)
    _BOTS.append(b)
    return b


make_bot.name = Bot.name

if __name__ == "__main__":
    mf.run(make_bot)
    for b in _BOTS:  # the match is over: the decision summary for the team's log
        b.report()
