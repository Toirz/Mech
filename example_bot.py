"""Bot: roam the safe circle hunting lasers, cubes and supply drops; hide to heal when hurt; fight with the right weapon.

Priorities each decision (movement): dodge a shell > get back in the zone > capture a supply drop
> hide out of sight while HP < HIDE_HP (until HEALED_HP) > a laser in the first minute > loot on the ground
> loot on a roof (only when calm: from within FLY_RANGE, jet straight at it and drop onto it) > stand and strafe against a mech in a fight
> farm a drone or crate we saw (most cubes per metre; shot on the way, its cubes picked up as loot)
> roam to the least recently visited cell (scanning).
"""
import math
import random

import mechfight as mf
from predict import Predictor
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

    def loot_goal(self, me, pos, c, r):
        """The nearest wanted loot on the ground (roof loot is roof_target's)."""
        return min((p for p in self.loot_items(me) if d2(p, c) < r - 6 and d2(p, pos) < LOOT_RANGE and not self.on_roof(p)),
                   key=lambda p: d2(p, pos), default=None)

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
    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        pos = me["pos"]
        x, y, z = pos
        t = self.t = obs["t"]
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

        # Who is dangerous, and should we run?
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        near = min(enemies, key=lambda m: d2(m["pos"], pos), default=None)
        T = self.targets(obs, me, muzzle, hpf)  # no running off in the open: a laser hits a runner just the same
        self.tid = T["id"] if T else None
        aim, fire, reload = self.weapons(me, muzzle, tube, T, t)
        if self.turn_to and t - self.turn_to[1] < 1.0 and not (T and T["kind"] == "mech" and T["los"] and T["id"] in self.hit_by
                                                              and t - self.hit_by[T["id"]] < 1.0):
            aim = [self.turn_to[0], 0.0]  # hit in the back (x1.5) by someone we don't have in front of us: turn round now
            fire = [bool(w and w["kind"] == "laser" and w.get("charge", 0) > 0) for w in me["weapons"]]  # nothing leaves off-aim; a charge is held

        if hpf < HIDE_HP:
            self.hiding = True
        elif hpf >= HEALED_HP:
            self.hiding = False
        if self.hiding and not (T and T["kind"] == "mech" and (t - self.hit_by.get(T["id"], -1e9) < 5 or T["d"] < 40)):
            # regen needs 5 s without dealing damage too: shoot only whoever is on us; a charge is held
            fire = [bool(w and w["kind"] == "laser" and w.get("charge", 0) > 0) for w in me["weapons"]]

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
        else:
            drop = None
            for dr in obs.get("drops", []):
                dd = d2(dr["pos"], pos)
                if (dr["holder"] in (None, me["id"]) and dd < DROP_RANGE and d2(dr["pos"], c) < r - 3
                        and dr["lands_in"] <= dd / st.get("walk_speed", 11.7) + 10
                        and not any(d2(m["pos"], dr["pos"]) < 60 for m in enemies)):
                    drop = dr
                    break
            loot = None if (near and d2(near["pos"], pos) < 70) or hpf < 0.35 else self.loot_goal(me, pos, c, r)
            shot_at = self.threat is not None and t - self.threat[1] < 3
            early = self.early_laser(me, pos, enemies) if t < EARLY_S and not shot_at else None
            engaged = bool(T and T["kind"] == "mech" and T["los"]
                           and (T["d"] < FIGHT_RANGE or t - self.hit_by.get(T["id"], -1e9) < 3))
            # shot by a mech we can see: stand and trade (running shows it our back, x1.5); hide once it lets up
            under_fire = bool(T and T["kind"] == "mech" and T["los"] and t - self.hit_by.get(T["id"], -1e9) < 3)
            hide = self.cover_spot(pos, c, r, enemies, t) if self.hiding and not drop and not under_fire else None
            farm = self.farm_goal(me, pos, c, r, enemies) if hpf >= 0.6 and not self.hiding else None  # targets() shoots them from 0.6
            calm = (not self.hiding and hpf >= 0.6 and not shot_at and not engaged
                    and not any(d2(m["pos"], pos) < ROOF_CALM for m in enemies))
            if self.climb and (t - self.climb["t0"] > ROOF_TIMEOUT or (me["grounded"] and not calm and not self.on_roof(pos))):
                self.climb = None  # trouble while still on the ground: forget the roof
            if not self.climb and calm and not (drop or hide or early or loot):
                rp = self.roof_target(me, pos, c, r)
                if rp:
                    self.climb = dict(p=rp, t0=t)
            losing = (under_fire and hpf < 0.5 and T["m"].get("hp", 1400) > me["hp"] * 1.2
                      and not self.on_roof(pos))  # its HP well over ours: we lose this trade
            run_to = self.cover_spot(pos, c, r, enemies, t) if losing else None
            if run_to is not None and d2(run_to, pos) > 2:
                goal, mode, arrive = run_to, "hide", 1.5  # zigzag into cover facing it, still shooting (strafe below)
            elif drop:
                goal, mode, arrive = drop["pos"], "drop", max(drop["radius"] - 3, 1.0)
            elif hide:
                goal, mode, arrive = hide, "hide", 1.5
            elif self.hiding and not engaged:
                pass  # no cover found: hold the zone edge (back to the gas, HP regens) instead of roaming into mechs
            elif early:
                goal, mode, arrive = early, "laser", 1.0
            elif loot:
                goal, mode, arrive = loot, "loot", 1.0
            elif self.climb:
                goal, mode, arrive = self.climb["p"], "roof", 1.0
            elif engaged:
                goal, mode = list(pos[:2]), "fight"  # stand and trade, strafing (running off loses fights)
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
            move = self.steer(pos, goal, t, force=self.stuck_since is not None and t - self.stuck_since > 1) if gap > arrive else [0.0, 0.0]
            shot = T and T["kind"] == "mech" and T["los"] and (T["d"] < FIGHT_RANGE or t - self.hit_by.get(T["id"], -1e9) < 3)
            if T and T["kind"] == "mech" and T["los"] and (mode in ("edge", "fight") or (
                    shot and mode in ("loot", "laser", "drop", "farm", "roam", "hide"))):  # strafe across their line of fire
                # (walking to loot, a drop or cover in a straight line under fire is a free target: zigzag there)
                if t >= self.strafe_until:
                    self.strafe_sign, self.strafe_until = -self.strafe_sign, t + self.rng.uniform(0.35, 0.9)  # reverse before their led shot lands
                yaw = math.atan2(T["pos"][1] - y, T["pos"][0] - x) + self.strafe_sign * math.pi / 2
                pull = min(gap / 6, 1.0)
                to_spot = unit(goal[0] - x, goal[1] - y)
                move = unit(math.cos(yaw) + to_spot[0] * pull, math.sin(yaw) + to_spot[1] * pull)

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
        boost = climbing or mode in ("dodge shell", "zone") or (stuck and not fighting) or (
            gap > 40 and me["energy"] > 60 and not T and mode != "roof")  # save the energy for the jet
        if fighting:  # boost through the whole fight, walking just long enough to recharge (duel-tested; jets lost)
            self.burn = me["energy"] > 70 if not self.burn else me["energy"] > 25
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


if __name__ == "__main__":
    mf.run(Bot)
