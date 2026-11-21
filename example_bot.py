"""Bot: hold the inside edge of the zone, fight with the right weapon for the range, dodge, loot, use the map to path and aim.

Priorities each decision (movement): dodge a laser/shell > get back in the zone > flee when weak
> capture a supply drop > grab nearby loot > hold the zone edge (strafing while fighting).
"""
import math
import random

import mechfight as mf
from world import World

EDGE_MARGIN = 12.0   # m to stay inside the zone edge
LOOKAHEAD_S = 4.0    # aim for where the edge will be, so a shrinking zone doesn't catch us
ARRIVED = 3.0        # m from a goal: stand still (silent, and HP regens)
FLEE_HP = 0.4        # flee from a visible mech below this HP fraction
LOOT_RANGE = 90.0    # m: farthest we walk for a weapon or cube
CUBE_KEEP = 15       # past this, cubes only make us bigger (stats cap at 10; 15 = one more upgrade)
MORTAR_MIN = 20.0    # m: closer than this our own splash and knockback hurt us
PATH_BUDGET = 0.01   # s of A* per decision (the whole decision must fit in 50 ms)
MECH_SOUNDS = {"gun", "mortar", "laser", "laser_charge", "lightning", "step", "boost", "reload", "jet"}
SOUND_DIST = {"near": 45.0, "mid": 110.0}
BEATS = {"laser": "mortar", "mortar": "lightning", "lightning": "laser"}
WEAPON_VALUE = {"laser": 1.0, "mortar": 0.8, "lightning": 0.6, "gun": 0.5}
UPGRADE_RANK = ["armor", "plating", "nanites", "laser_damage", "gun_damage", "gun_rate", "laser_charge", "sensors",
                "servos", "gyros", "gun_mag", "gun_velocity", "lightning_efficiency", "mortar_reload", "lightning_range",
                "capacitor", "reactor", "jets", "mortar_velocity", "mortar_splash", "lightning_chain", "laser_pierce"]


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
        self.heard = None                # (bearing, t, dist) of the last mech sound
        self.strafe_sign, self.strafe_until = 1, 0.0
        self.burn = True  # boosting in a fight: on until energy drops to 25, back on at 70
        self.vhist = {}  # target id -> [(t, its velocity)] over the last 2 s
        self.short_until = 0.0
        self.seen_weapons = {}           # mech id -> weapons it was seen holding
        self.pickups = {}                # id -> (kind, pos, t seen)
        self.cubes = {}                  # id -> (pos, t seen)
        self.bad_pickups = {}            # id -> t until we stop trying it

    # ----- memory -----
    def remember(self, obs, t, pos):
        for m in obs["mechs"]:
            if m["kind"] == "mech" and m.get("weapons"):
                self.seen_weapons[m["id"]] = m["weapons"]
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
    def weapon_value(self, kind):
        v = WEAPON_VALUE.get(kind, 0)
        for ws in self.seen_weapons.values():
            for w in ws:
                v += 0.15 * ((BEATS.get(kind) == w) - (BEATS.get(w) == kind))
        return v

    def wants(self, kind, me):
        """Is a pickup of this kind worth taking, and into which slot (drop) if both are full?"""
        ws = me["weapons"]
        if kind == "gun" or any(w and w["kind"] == kind for w in ws):
            return False, None
        if None in ws:
            return True, None
        swap = [i for i, w in enumerate(ws) if w["kind"] != "gun" and not w.get("charge")]
        if not swap:
            return False, None
        i = min(swap, key=lambda i: self.weapon_value(ws[i]["kind"]))
        return self.weapon_value(kind) > self.weapon_value(ws[i]["kind"]) + 0.3, i

    def targets(self, obs, me, muzzle, hpf, fleeing):
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
            return x["d"] / 100 + hp * 1.5 + (0 if x["los"] else 1.5) - 0.8 * away - 0.4 * (x["id"] in busy)

        mechs = sorted((x for x in mechs if x["d"] < 300), key=lambda x: x["d"])[:6]
        for x in mechs:
            x["los"] = self.los(muzzle, x["c"])
        if mechs:
            return min(mechs, key=score)
        if fleeing or hpf < 0.6:
            return None
        for x in sorted(others, key=lambda x: (x["kind"] != "npc", x["d"])):
            if x["d"] < (150 if x["kind"] == "npc" else 120) and (x["kind"] == "npc" or me["cubes"] < CUBE_KEEP or None in me["weapons"]):
                x["los"] = self.los(muzzle, x["c"])
                if x["los"]:
                    return x
        return None

    def weapons(self, me, muzzle, tube, T, fleeing, t):
        """(aim or None, fire[2], reload)."""
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
                    ok[i] = T["los"] and T["d"] <= st["range"] and w.get("ammo", 1) > 0 and not w.get("reload")
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
                else:  # start charging only when not running away: charging halves our speed
                    fire[i] = ok[i] and w["cooldown"] <= 0 and not fleeing and err < 0.35
            elif k == "lightning":
                fire[i] = ok[i] and w["cooldown"] <= 0.1 and abs(ang_diff(now[0], aims[i][0])) < tol[i]
            else:
                fire[i] = ok[i] and w["cooldown"] <= 0.1 and err < tol[i]
        gun = next((w for w in ws if w and w["kind"] == "gun"), None)
        calm = T is None or T["kind"] != "mech"
        reload = bool(gun) and not gun.get("reload") and (gun.get("ammo", 1) == 0 or (calm and gun.get("ammo", 0) < gun["stats"]["mag"] / 2))
        return aim, fire, reload

    # ----- movement -----
    def dodge(self, obs, me, pos):
        """A move vector away from an aimed laser or an incoming shell, or None."""
        laser_range = self.cfg["weapons"]["laser"]["range"]
        for m in obs["mechs"]:
            if m["kind"] == "mech" and m.get("charging"):
                d = d2(m["pos"], pos)
                off = ang_diff(math.atan2(pos[1] - m["pos"][1], pos[0] - m["pos"][0]), m["aim"][0])
                if d < laser_range and abs(off) < math.atan2(8, d) + 0.05:
                    yaw = m["aim"][0] + (math.pi / 2 if off >= 0 else -math.pi / 2)  # step further off their line
                    return [math.cos(yaw), math.sin(yaw)], "dodge laser"
        mo = self.cfg["weapons"]["mortar"]
        for p in obs["projectiles"]:
            if p["kind"] == "mortar":
                land = shell_landing(p["pos"], p["vel"], mo["gravity"], pos[2] + 1)
                if land and land[2] < 3 and d2(land, pos) < mo["splash_radius"] * 1.3 + 3:
                    v = unit(pos[0] - land[0], pos[1] - land[1])
                    return (v if v != [0.0, 0.0] else [-p["vel"][1], p["vel"][0]]), "dodge shell"
        return None

    def free_edge_spot(self, pos, c, r):
        """edge_spot, slid along the circle to the nearest walkable cell."""
        spot = edge_spot(pos, c, r, EDGE_MARGIN)
        if not self.world or self.world.free(*spot):
            return spot
        base = math.atan2(spot[1] - c[1], spot[0] - c[0])
        rr = max(r - EDGE_MARGIN, 0.0)
        for k in range(1, 40):
            for s in (1, -1):
                a = base + s * k * min(0.05, 4 / max(rr, 1))
                p = [c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr]
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
            ids = [ch["id"] for ch in obs["offer"]["choices"]]
            act["pick"] = min(range(len(ids)), key=lambda i: UPGRADE_RANK.index(ids[i]) if ids[i] in UPGRADE_RANK else 99)

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

        # Who is dangerous, and should we run?
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        near = min(enemies, key=lambda m: d2(m["pos"], pos), default=None)
        stronger = near is not None and near.get("hp", 0) > 1.5 * me["hp"] and near.get("cubes", 0) >= me["cubes"] + 3
        # Low on HP: run, unless the one in front of us has even less (running would throw away a fight we're winning).
        fleeing = near is not None and d2(near["pos"], pos) < 150 and ((hpf < FLEE_HP and near.get("hp", 0) >= me["hp"]) or stronger)
        T = self.targets(obs, me, muzzle, hpf, fleeing)
        aim, fire, reload = self.weapons(me, muzzle, tube, T, fleeing, t)  # we shoot while running too: holding fire lost duels

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
        elif fleeing:
            away = unit(x - near["pos"][0], y - near["pos"][1])
            g = [x + away[0] * 40, y + away[1] * 40]
            if d2(g, c) > safe_r:  # stay inside: slide along the edge instead
                g = edge_spot(g, c, r, EDGE_MARGIN)
            goal, mode = g, "flee"
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
            if drop:
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
        boost = mode in ("dodge laser", "dodge shell", "zone", "flee") or (stuck and not fighting) or (
            gap > 40 and me["energy"] > 60 and not T)
        if fighting:  # boost through the whole fight, walking just long enough to recharge (duel-tested; jets lost)
            self.burn = me["energy"] > 70 if not self.burn else me["energy"] > 25
            boost = boost or (self.burn and me["grounded"])

        # Where to look when there's nothing to shoot.
        if aim is None:
            inward = math.atan2(c[1] - y, c[0] - x)
            if self.threat and t - self.threat[1] < 3:  # turn to face whoever hit us
                aim = [self.threat[0], 0.0]
            elif self.heard and t - self.heard[1] < 2:
                aim = [self.heard[0], 0.0]
                aim, fire = self.blind_mortar(me, tube, hpf, fleeing, aim, fire)
            elif gap > 30 and wants_move:
                aim = [math.atan2(move[1], move[0]), 0.0]
            else:
                aim = [inward, 0.0]  # back to the zone edge: enemies come from inside

        act.update(move=move, boost=boost, jump=jump, aim=aim, fire=fire, reload=reload)
        self.draw(pos, muzzle, goal, T, mode, me)
        return act

    def blind_mortar(self, me, tube, hpf, fleeing, aim, fire):
        """Lob a shell at a heard mech we can't see."""
        b, _, dist = self.heard
        for i, w in enumerate(me["weapons"]):
            if w and w["kind"] == "mortar" and dist in SOUND_DIST and hpf > 0.5 and not fleeing:
                rng = SOUND_DIST[dist]
                p = [tube[0] + math.cos(b) * rng, tube[1] + math.sin(b) * rng, 0.0]
                p[2] = self.world.ground_at(p[0], p[1], me["pos"][2]) if self.world else me["pos"][2]
                sol = mortar_aim(tube, p, [0, 0, 0], w["stats"]["speed"], w["stats"]["gravity"], high=True,
                                 max_pitch=self.mech["max_pitch"])
                if sol:
                    aim = sol[0]
                    fire = list(fire)
                    fire[i] = w["cooldown"] <= 0 and aim_error(me["aim"], aim) < 0.03
        return aim, fire

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
