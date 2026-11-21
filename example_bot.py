"""Bot: hold the inside edge of the zone, fight with the right weapon for the range, dodge, loot, use the map to path and aim.

Priorities each decision (movement): dodge a laser/shell > get back in the zone > flee when weak
> capture a supply drop > grab nearby loot > hold the zone edge (strafing while fighting).
"""
import math
import random

import mechfight as mf
from predict import Predictor
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
LATENCY = 0.03       # s from an observation to the shot leaving (the action applies on the next ticks)
GUN_TOL = 1.0        # m: the gun fires only when its aim is this close to the lead point, at the target
GUN_MIN_P = 0.15     # the gun holds fire below this estimated hit chance
LASER_MARGIN = 0.8   # m: the beam must point this far inside the target's edge when we let go
GHOST_S = 8.0       # s we keep guessing where an unseen mech is
GHOST_DIST = {"near": 40.0, "mid": 110.0, "far": 220.0}  # m: a guess inside each sound distance band
GHOST_W = {"laser": 1.5, "laser_charge": 1.5, "gun": 1.0, "lightning": 1.0, "mortar": 1.0}  # steps, boosts, jets: 0.5
DEBUG = bool(__import__("os").environ.get("BOT_DEBUG"))
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


def bullet_miss(p, v, c):
    """(seconds until a straight bullet at p moving at v is closest to point c, vector from that closest point to c)."""
    vv = sum(x * x for x in v)
    if vv < 1e-9:
        return -1.0, [0.0, 0.0, 0.0]
    tca = sum((c[i] - p[i]) * v[i] for i in range(3)) / vv
    return tca, [c[i] - (p[i] + v[i] * tca) for i in range(3)]


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
        self.foes = {}                   # mech id -> (pos, t) last listed in obs.mechs
        self.ghosts = []                 # [x, y, t, w]: where an unseen mech probably is (sound, hit from out of view, left view)
        self.spot, self.spot_t = None, -1e9  # the edge spot we picked, and when
        self.open_cache = {}             # 4 m cell -> share of 8 directions with a clear 60 m line of fire
        self.dbg = {}
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
        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("bearing") is not None:
                self.threat = (e["bearing"], t)
                if e.get("rear") or (e.get("by") is not None and e["by"] not in {m["id"] for m in obs["mechs"]}):
                    self.turn_to = (e["bearing"], t)  # behind us, or out of view (60-90 deg off our aim)
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
        self.track_ghosts(obs, t, pos)

    def track_ghosts(self, obs, t, pos):
        """Guess where the mechs we can't see are: from sounds, from hits by someone out of view, and from where a
        tracked mech dropped out of the list. A guess we can now see and find empty is dropped."""
        seen = {m["id"]: m["pos"] for m in obs["mechs"] if m["kind"] == "mech"}
        new = [(p[0], p[1], 1.0) for k, (p, _) in self.foes.items() if k not in seen]  # left view (and the server's 5 s memory)
        self.foes = {**{k: v for k, v in self.foes.items() if k in seen}, **{k: (p, t) for k, p in seen.items()}}

        def at(bearing, d):
            return pos[0] + math.cos(bearing) * d, pos[1] + math.sin(bearing) * d
        for s in obs["sounds"]:
            if s["kind"] not in MECH_SOUNDS:
                continue
            d = min(GHOST_DIST[s["dist"]], self.cfg["sound"]["range"].get(s["kind"], 400) * 0.8)
            if any(abs(ang_diff(math.atan2(p[1] - pos[1], p[0] - pos[0]), s["bearing"])) < 0.3 for p in seen.values()):
                continue  # most likely a mech we already see
            new.append((*at(s["bearing"], d), GHOST_W.get(s["kind"], 0.5)))
        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("by") is not None and e["by"] not in seen and e.get("bearing") is not None:
                d = 60.0 if e["weapon"] == "lightning" else 110.0  # the kills came from 45-160 m
                new.append((*at(e["bearing"], d), 2.0))
        for gx, gy, w in new:
            g = next((g for g in self.ghosts if math.hypot(g[0] - gx, g[1] - gy) < 35), None)
            if g:
                g[:] = [gx, gy, t, max(g[3], w)]
            else:
                self.ghosts.append([gx, gy, t, w])
        vis = self.cfg["vision"]
        eye = [pos[0], pos[1], pos[2] + self.mech["eye_height"]]
        yaw = obs["self"]["aim"][0]
        keep = []
        for g in self.ghosts:
            if t - g[2] > GHOST_S:
                continue
            d = d2(g, pos)
            in_view = d < vis["near_range"] or (d < vis["range"] and abs(ang_diff(math.atan2(g[1] - pos[1], g[0] - pos[0]), yaw))
                                                < math.radians(vis["fov_deg"]) / 2 - 0.1)
            gz = self.world.ground_at(g[0], g[1], pos[2]) if self.world else pos[2]
            if (t - g[2] > 0.5 and in_view and not any(d2(p, g) < 35 for p in seen.values())
                    and self.los(eye, [g[0], g[1], gz + 4.5])):
                continue  # we look right at it and nobody is there
            keep.append(g)
        self.ghosts = sorted(keep, key=lambda g: -self.ghost_danger(g, pos))[:8]

    def ghost_danger(self, g, pos):
        return g[3] * (1 - (self.t - g[2]) / GHOST_S) / max(d2(g, pos) / 100, 0.5)

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

        def score(x):
            m = x["m"]
            away = abs(ang_diff(m["aim"][0], math.atan2(muzzle[1] - x["pos"][1], muzzle[0] - x["pos"][0]))) > math.pi / 2
            hp = m.get("hp", 1) / max(m.get("max_hp", 1), 1)
            return (x["d"] / 100 + hp * 1.5 + (0 if x["los"] else 1.5) - 0.8 * away - 0.8 * (x["id"] in busy)
                    - 0.4 * (away and m.get("charging", False))  # its laser is on someone else: it can't answer us in time
                    - 0.5 * (self.t - self.hit_on.get(x["id"], -1e9) < 2)  # keep on the one we are hurting
                    - 1.0 * (self.t - self.hit_by.get(x["id"], -1e9) < 3))  # whoever is shooting us first

        mechs = sorted((x for x in mechs if x["d"] < 300), key=lambda x: x["d"])[:6]
        for x in mechs:
            x["los"] = self.los(muzzle, x["c"])
        npc = next((x for x in others if x["kind"] == "npc" and self.t - self.hit_by.get(x["id"], -1e9) < 3), None)
        if npc and not any(self.t - self.hit_by.get(x["id"], -1e9) < 3 for x in mechs):  # a drone is shooting us: answer it
            npc["los"] = self.los(muzzle, npc["c"])
            return npc
        if mechs:
            return min(mechs, key=score)
        if hpf < 0.6 or any(self.t - g[2] < 3 and d2(g, me["pos"]) < 160 for g in self.ghosts):
            return None  # a fresh guess nearby: look that way instead of farming (the gun's noise carries 320 m)
        for x in sorted(others, key=lambda x: (x["kind"] != "npc", x["d"])):
            if x["d"] < (150 if x["kind"] == "npc" else 120) and (x["kind"] == "npc" or me["cubes"] < CUBE_KEEP or None in me["weapons"]):
                x["los"] = self.los(muzzle, x["c"])
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
        charging = la is not None and ws[la].get("charge", 0) > 0
        if charging:  # follow the gun's lead while it charges; on its point for the last 0.15 s and after
            full = ws[la]["charge"] >= ws[la]["stats"]["charge"] - 0.15
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
                if w.get("charge", 0) > 0:  # hold until full and on target, then let go (it fires on release)
                    full = w["charge"] >= w["stats"]["charge"] - 0.05
                    on = aim_error(then, aims[i]) * dists[i] < T["r"] - LASER_MARGIN and off < T["r"] + 1.0
                    fire[i] = not (full and ok[i] and on)
                else:  # start only if it will still be in the open when the charge is full
                    fire[i] = (ok[i] and w["cooldown"] <= 0 and err < 0.35
                               and self.los(muzzle, self.point_at(T, w["stats"]["charge"], T["c"][2] - T["pos"][2])))
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

    def beam_point(self, muzzle, T):
        """Where to point the laser: T's predicted body after the decision latency, at the middle, else
        the highest or lowest part of it a beam can reach (a mech half behind a wall or a crest)."""
        mid = T["c"][2] - T["pos"][2]
        top = 2 * mid
        for z in (mid, top - 1.2, top - 2.5, 1.5):
            pt = self.point_at(T, LATENCY, z)
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
        c = [pos[0], pos[1], pos[2] + self.mech["height"] / 2]
        for p in obs["projectiles"]:  # a bullet headed for our body: step out of its line (at 100 m it flies 0.4 s)
            if p["kind"] != "mortar":
                tca, miss = bullet_miss(p["pos"], p["vel"], c)
                if 0 < tca < 0.6 and math.hypot(*miss[:2]) < self.mech["radius"] + 1.0:
                    side = unit(miss[0], miss[1]) if math.hypot(*miss[:2]) > 0.3 else [v * self.strafe_sign for v in
                                                                                     unit(-p["vel"][1], p["vel"][0])]
                    return side, "dodge bullet"
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

    def exposure(self, p):
        """Share of 8 directions from which a mech 60 m away could shoot a mech standing at p (0 = walled in)."""
        key = (round(p[0] / 4), round(p[1] / 4))
        if key not in self.open_cache:
            z = self.world.ground_at(p[0], p[1], 0.0) + self.mech["eye_height"]
            a = [p[0], p[1], z]
            self.open_cache[key] = sum(self.world.clear(a, [p[0] + 60 * math.cos(k * math.pi / 4), p[1] + 60 * math.sin(k * math.pi / 4), z])
                                       for k in range(8)) / 8
        return self.open_cache[key]

    def edge_goal(self, pos, c, r, enemies, hold):
        """The edge spot with the least risk: the walk there, known and guessed mechs near it, and open lines of fire.
        Re-picked once a second (never mid-fight) while it stays inside the circle."""
        base = self.free_edge_spot(pos, c, r)
        if not self.world:
            return base
        rr = max(r - EDGE_MARGIN, 1.0)
        if self.spot and d2(self.spot, c) < rr + EDGE_MARGIN / 2 and (hold or self.t - self.spot_t < 1.0):
            return self.spot
        a0 = math.atan2(base[1] - c[1], base[0] - c[0])
        threats = [(m["pos"], 1.5 if "laser" in m.get("weapons", []) else 1.0) for m in enemies] + [(g, g[3]) for g in self.ghosts]
        best, bs = base, 1e9
        for k in range(-4, 5):
            a = a0 + k * min(15 / rr, 0.3)
            p = base if k == 0 else self.world.clamp([c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr], MAP_MARGIN)
            if k and not self.world.free(*p):
                continue
            s = (d2(p, pos) / 40 + sum(w / max(d2(p, q) / 60, 0.5) for q, w in threats) + 1.5 * self.exposure(p)
                 - (0.3 if self.spot and d2(p, self.spot) < 8 else 0))  # a little stickiness: no flip-flopping
            if s < bs:
                best, bs = p, s
        self.spot, self.spot_t = best, self.t
        return best

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
        aim, fire, reload = self.weapons(me, muzzle, tube, T, t)
        if self.turn_to and t - self.turn_to[1] < 1.0 and not (T and T["kind"] == "mech" and T["los"] and T["id"] in self.hit_by
                                                              and t - self.hit_by[T["id"]] < 1.0):
            aim = [self.turn_to[0], 0.0]  # hit in the back (x1.5) by someone we don't have in front of us: turn round now
            fire = [bool(w and w["kind"] == "laser" and w.get("charge", 0) > 0) for w in me["weapons"]]  # nothing leaves off-aim; a charge is held

        # Movement goal.
        c, r = zone_at(obs["zone"], t + LOOKAHEAD_S)
        spot = self.edge_goal(pos, c, r, enemies, hold=bool(T and T["kind"] == "mech" and T["los"]))
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
                if (dr["holder"] in (None, me["id"]) and dd < 250 and d2(dr["pos"], c) < r - 3
                        and dr["lands_in"] <= dd / st.get("walk_speed", 11.7) + 10
                        and not any(d2(m["pos"], dr["pos"]) < 60 for m in enemies)
                        and not any(d2(g, dr["pos"]) < 60 for g in self.ghosts)):
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
        if move is None:
            move = self.steer(pos, goal, t, force=self.stuck_since is not None and t - self.stuck_since > 1) if gap > arrive else [0.0, 0.0]
            if mode == "edge" and T and T["kind"] == "mech" and T["los"]:  # strafe across their line of fire
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
        boost = mode in ("dodge shell", "dodge bullet", "zone") or (stuck and not fighting) or (
            gap > 40 and me["energy"] > 60 and not T)
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
            elif self.ghosts and t - self.ghosts[0][2] < 5:  # the likeliest unseen mech: look until it shows or the spot is empty
                g = self.ghosts[0]
                aim = [math.atan2(g[1] - y, g[0] - x), 0.0]
            elif self.heard and t - self.heard[1] < 2:
                aim = [self.heard[0], 0.0]
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
