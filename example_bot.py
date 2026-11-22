"""Bot: roam the safe circle hunting lasers, cubes and supply drops; hide to heal when hurt; fight with the right weapon.

Priorities each decision (movement): dodge a shell > get back in the zone > capture a supply drop
> hide out of sight (below HIDE_HP, until HEALED_HP: silently, along safe ground, into a safe wall corner) > get away from strong mechs (and, hurt, from any mech near us)
> run down a weak mech (healthy) > a laser in the first minute > loot on the ground > loot on a roof (only when calm:
from within FLY_RANGE, jet straight at it and drop onto it) > stand and strafe against a mech in a fight
> go for a lone mech (healthy) > roam to the least recently visited cell.
Everywhere, the safety map (how many of 24 directions each 4 m cell can be seen from) keeps us on safe ground: walks
take paths that charge for exposed cells, roam waypoints and edge spots are never red (seen from RED+ directions)
when there is another choice, and loot in the open counts as farther.

Strength = HP x cube damage bonus x firepower (best weapon + a third of the second), ours against each mech we see:
above STRONG x ours it is strong: we keep AVOID_R away, skip loot near it and never shoot first; below WEAK x ours
(few cubes, little HP, only a gun) it is weak: healthy, we hunt it down while it is alone.

The enemy map: every mech we see, and every mech sound we hear (bearing and near/mid/far give a rough spot), goes on
it; a mech that leaves our view stays where we lost it (a ghost) until we look there and it is gone.
- HP >= HEALED_HP (healthy): loot all the time, taking on one mech for an item (zigzagging to it while we trade
  shots), never two; with nothing to loot, go and take on a lone mech from the map.
- HIDE_HP..HEALED_HP (hurt): start no fights; loot only what no mech on the map is near, and walk away from any mech
  we see or hear close by. Shot by one we can see: stand and trade.
- below HIDE_HP: hide out of sight until back at HEALED_HP.
Where we look: at what we can shoot; else always at the closest enemy we know of (one behind cover, one we lost or
heard in the last FACE_S, the sound we just heard; whoever just hit us first); only knowing of none do we sweep left
and right and glance behind every few seconds (looking first at any older ghost near where we are going).
"""
import math
import os
import random
import time

import mechfight as mf
from predict import Predictor
from world import CELL, CORNER, NAV, World

EDGE_MARGIN = 12.0   # m to stay inside the zone edge
LOOKAHEAD_S = 4.0    # aim for where the edge will be, so a shrinking zone doesn't catch us
ARRIVED = 3.0        # m from a goal: stand still (silent, and HP regens)
LOOT_RANGE = 200.0   # m: farthest we walk for a weapon or cube (roaming: we go looking)
CUBE_KEEP = 15       # past this, cubes only make us bigger (stats cap at 10; 15 = one more upgrade)
MORTAR_MIN = 20.0    # m: closer than this our own splash and knockback hurt us
SLIDE_AHEAD = 1.2    # m: look this far along the move for walls (one decision at walk speed)
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
DROP_RANGE = 250.0   # m: farthest we go for a supply drop (3 cubes, full HP and energy)
ROAM_CELL = 40.0     # m: cells of the visited map; roaming heads for the one we left longest ago
ROAM_REPICK = 25.0   # s: give up on a roam waypoint after this
HIDE_HP, HEALED_HP = 0.25, 0.7  # HP fraction: below the first hide at once, until the second; below the second, start no fights
GHOST_S = 30.0       # s we remember where a mech was when it left our view (the server tracks it for 5 s of that)
HEARD_S = 10.0       # s we keep a spot we only heard (a moving mech makes a new sound every 0.6 s)
FACE_S = 10.0        # s: an enemy seen or heard this recently is where we look while there's nothing to shoot
CROWD_R = 80.0       # m: mechs this close to an item guard it; healthy we take on one for it, never two
EVADE_R = 100.0      # m: hurt, we walk away from any mech on the map this close
HUNT_RANGE = 200.0   # m: healthy with nothing to loot, we go for a lone mech on the map this close
STRONG, WEAK = 1.4, 0.7  # a mech this many times our strength is strong (avoid it) / weak (hunt it down)
AVOID_R = 150.0      # m we keep away from a strong mech (it reaches us with a laser well past that, but rarely sees us)
SWEEP, SWEEP_W = 1.2, 2.5  # rad either side of the way we face, and the sweep's rate (peak slew 3 rad/s; the torso does 4)
GLANCE_EVERY = (5.0, 8.0)  # s between two looks behind
GLANCE_S = 1.0       # s a look behind lasts (turning round takes 0.75 s at 4 rad/s)
SILENT = 0.4         # move length that keeps us under step_speed (5 m/s): no step sounds
RED = 19             # a cell seen from this many of 24 directions or more is red: no roaming to it, no edge spot on it
MOVE_SAFE_W = 3.0    # extra A* cost per 4 m step, walking anywhere, through a cell seen from all 24 directions
                     # (paths 60-220 m: 14% -> 9% of the way on red cells, no longer; weighted A* needs > 2.5 to care)
ROAM_EXPO = 4.0      # s of roam score per direction a roam waypoint can be seen from
LOOT_EXPO = 4.0      # m added to an item's distance per direction its cell can be seen from
EDGE_SLIDE = 60.0    # m along the zone edge we look for the safest edge spot
HIDE_R = 60.0        # m: hiding spots are looked for this far around us (4 m cells)
HIDE_EVERY = 2.0     # s between two searches for a hiding spot
HIDE_CHECK = (24, 0.006)  # at most this many spots, or s, of line-of-sight checks per search
HIDE_NEAR = 30.0     # m: no hiding spot this close to a mech on our map
DANGER_R = 70.0      # m: a mech on our map makes hiding spots and the way there worse this close to it
SAFE_W = 1.5         # extra A* cost per 4 m step, hiding, through a cell seen from all 24 directions
FIGHT_RANGE = 150.0  # m: a seen mech this close (or one shooting us) stops the roam: stand and strafe
ROOF_Z = 3.0         # m above the terrain: an item this high is on a roof (a jump clears 2.75 m)
ROOF_MAX = 45.0      # m: highest roof we jet up to for loot (100 energy climbs about 60 m)
ROOF_CALM = 120.0    # m: climb only with no known mech this close (in the air we can't dodge)
ROOF_TIMEOUT = 30.0  # s: give up a roof trip after this (walk up to 200 m, then fly)
FLY_RANGE = 40.0     # m: walk this close to roof loot, then jet straight at it and drop onto it
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
        self.world, self.move_costs = None, None
        try:
            self.world = World(hello["map"]["dir"], self.mech)
            if self.world.load_safety(os.path.join(os.path.dirname(os.path.abspath(__file__)), "safety.json")) is None:
                mf.log("no safety map for this map: hiding by sight lines and walls only")
            self.move_costs = self.world.safe_costs(MOVE_SAFE_W)  # every walk keeps to safe ground where it can
        except Exception as e:  # without the map we still play: straight lines, no cover checks
            mf.log(f"map not loaded, playing without it: {e!r}")
        self.path, self.path_goal, self.path_t, self.path_kind = [], None, -1e9, None
        self.hide_dbg = []               # the hiding spots the last search weighed: [spot, score, out of sight / None]
        self.hist = []                   # (t, x, y, wanted to move)
        self.stuck_since = None
        self.unstick_until, self.unstick_dir = -1.0, [0.0, 0.0]
        self.threat = None               # (bearing, t) of the last hit we took
        self.turn_to = None              # (bearing, t): hit in the back by someone we can't see: face that way a moment
        self.heard = None                # (bearing, t, guessed m) of the closest mech sound we last heard
        self.strafe_sign, self.strafe_until = 1, 0.0
        self.burn = True  # boosting in a fight: on until energy drops to 25, back on at 70
        self.vhist = {}  # target id -> [(t, its velocity)] over the last 2 s
        self.short_until = 0.0
        self.pickups = {}                # id -> (kind, pos, t seen)
        self.cubes = {}                  # id -> (pos, t seen)
        self.bad_pickups = {}            # id -> t until we stop trying it
        self.hit_by, self.hit_on = {}, {}  # id -> last time it damaged us / we damaged it
        self.dbg = {}
        self.visited, self.roam, self.roam_t = {}, None, -1e9  # cell -> last t there; current waypoint
        self.hiding, self.cover, self.cover_t = False, None, -1e9
        self.ghosts = {}                 # the enemy map: mech id (seen) or ("h", n) (heard) -> {pos, t, r: how far off it may be}
        self.glance_next, self.glance_until, self.glance_side = 3.0, -1.0, 1
        self.heard_n = 0                 # keys of heard spots on the map: ("h", n)
        self.evade, self.evade_t = None, -1e9
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
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                self.ghosts[m["id"]] = {"pos": list(m["pos"]), "t": t, "r": 0.0,
                                        "s": self.strength(m.get("hp", self.mech["hp"]), m.get("cubes", 0), m.get("weapons") or ["gun"])}
        self.ghosts = {k: g for k, g in self.ghosts.items() if t - g["t"] < (GHOST_S if g["r"] == 0 else HEARD_S)}
        for e in obs["events"]:
            if e["type"] == "kill":
                self.ghosts.pop(e["target"], None)
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
        heard = [(self.hear(s, pos, t), s) for s in obs["sounds"] if s["kind"] in MECH_SOUNDS]
        if heard:
            d, s = min(heard, key=lambda e: (e[0], e[1]["age"]))
            self.heard = (s["bearing"], t - s["age"], d)

    def los(self, a, b):
        return self.world.clear(a, b) if self.world else True

    def sees(self, a, b):
        """Sight, not a shot: trees count (trunks, and thick crown past the near range)."""
        return self.world.sees(a, b) if self.world else True

    def strength(self, hp, cubes, kinds):
        """How much fight a mech has, to compare with ours: HP x its cube damage bonus x firepower (best weapon
        plus a third of the second: a gun 0.5, a laser 1.0)."""
        C = self.cfg["cubes"]
        v = sorted((WEAPON_VALUE.get(k, 0.3) for k in kinds), reverse=True) + [0.0, 0.0]
        return hp * (1 + C["damage_per_cube"] * min(cubes, C["stat_cap"])) * (v[0] + v[1] / 3)

    def strong_spots(self, t, mine):
        """Where the mechs we judged strong (over STRONG x our strength) were in the last FACE_S."""
        return [g["pos"] for g in self.ghosts.values() if g.get("s") and g["s"] > STRONG * mine and t - g["t"] <= FACE_S]

    def chase_goal(self, pos, c, r, t, mine, strong):
        """The nearest mech we judged weak (under WEAK x our strength), seen in the last FACE_S within HUNT_RANGE and
        inside the safe circle, alone (no other mech within CROWD_R of it, strong or not)."""
        rr = max(r - EDGE_MARGIN, 0.0)
        return min((g["pos"] for g in self.ghosts.values() if g.get("s") is not None and g["s"] < WEAK * mine
                    and t - g["t"] <= FACE_S and d2(g["pos"], pos) < HUNT_RANGE and d2(g["pos"], c) < rr
                    and self.crowd(g["pos"], t) <= 1 and not any(d2(q, g["pos"]) < CROWD_R for q in strong)),
                   key=lambda q: d2(q, pos), default=None)

    def hear(self, s, pos, t):
        """Put a mech sound on the enemy map: along its bearing, at the middle of its distance class (capped by
        how far that kind carries), off by half the class plus the bearing noise. Near a spot already on the map
        it moves that spot (a listed mech made it: nothing to learn). Returns the guessed distance."""
        S = self.cfg["sound"]
        reach = S["range"].get(s["kind"], S["mid"])
        lo, hi = {"near": (0.0, S["near"]), "mid": (S["near"], S["mid"])}.get(s["dist"], (S["mid"], max(reach, S["mid"])))
        hi = min(hi, reach)
        lo = min(lo, hi)
        d, ts = (lo + hi) / 2, t - s["age"]
        q = [pos[0] + d * math.cos(s["bearing"]), pos[1] + d * math.sin(s["bearing"]), 0.0]
        q[2] = self.world.ground_at(q[0], q[1], pos[2]) if self.world else pos[2]
        u = (hi - lo) / 2 + d * math.radians(S.get("bearing_noise_deg", 7)) * 2
        near = min(self.ghosts.values(), key=lambda g: d2(g["pos"], q), default=None)
        if near and d2(near["pos"], q) < u + near["r"] + 20:
            if near["t"] < t and ts > near["t"]:
                if u < near["r"] + 6.0 * (ts - near["t"]):  # the sound pins it down better than the old spot
                    near["pos"], near["r"] = q, u
                near["t"] = ts
            return d
        self.heard_n += 1
        self.ghosts[("h", self.heard_n)] = {"pos": q, "t": ts, "r": u, "s": None}  # heard only: strength unknown
        return d

    def lost(self, pos, t, n=6):
        """The n nearest spots on the map of mechs not listed now (lost from view, or only heard)."""
        return sorted((g["pos"] for g in self.ghosts.values() if g["t"] < t), key=lambda q: d2(q, pos))[:n]

    def clear_ghosts(self, aim_yaw, pos, t):
        """Forget a spot on the map once it is in our view and nothing is listed there (in view it would be);
        a spot only heard far off is too rough for that: it just expires."""
        V, mid = self.cfg["vision"], self.mech["height"] / 2
        eye = [pos[0], pos[1], pos[2] + self.mech.get("eye_height", 8)]
        for k, g in list(self.ghosts.items()):
            q = g["pos"]
            d = d2(q, pos)
            if g["t"] >= t or d > V["range"] - 5 or g["r"] > 40:
                continue
            if d > V["near_range"] and abs(ang_diff(math.atan2(q[1] - pos[1], q[0] - pos[0]), aim_yaw)) > math.radians(V["fov_deg"]) / 2 - 0.1:
                continue
            if self.sees(eye, [q[0], q[1], q[2] + mid]):
                del self.ghosts[k]

    def crowd(self, p, t, r=CROWD_R):
        """How many mechs on the map may be within r m of p: listed ones where they are, the others where we lost
        or heard them, widened by how rough the spot is (up to 40 m) and how far they could have walked since
        (6 m/s, up to 60 m)."""
        return sum(d2(g["pos"], p) < r + min(g["r"], 40.0) + min(6.0 * (t - g["t"]), 60.0) for g in self.ghosts.values())

    def evade_spot(self, pos, c, r, threats, t):
        """Somewhere 25-60 m farther from the mechs near us (seen or heard), inside the safe circle, out of their
        sight if we can find such a spot; redone once a second."""
        if self.evade and t - self.evade_t < 1.0 and d2(self.evade, pos) > 4:
            return self.evade
        ax = sum((pos[0] - q[0]) / max(d2(q, pos), 1.0) ** 2 for q in threats)
        ay = sum((pos[1] - q[1]) / max(d2(q, pos), 1.0) ** 2 for q in threats)
        a0 = math.atan2(ay, ax) if abs(ax) + abs(ay) > 1e-12 else math.atan2(c[1] - pos[1], c[0] - pos[0])
        rr = max(r - EDGE_MARGIN, 0.0)
        eye, mid = self.mech.get("eye_height", 8.0), self.mech["height"] / 2
        watchers = [[q[0], q[1], q[2] + eye] for q in sorted(threats, key=lambda q: d2(q, pos))[:3]]
        first = None
        for dist in (60, 40, 25):
            for da in (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5):
                p = [pos[0] + dist * math.cos(a0 + da), pos[1] + dist * math.sin(a0 + da)]
                if self.world:
                    p = self.world.clamp(p, MAP_MARGIN)
                    if not self.world.free(*p):
                        continue
                if d2(p, c) >= rr or not all(d2(q, p) > d2(q, pos) + 10 for q in threats):
                    continue
                first = first or p
                z = self.world.ground_at(p[0], p[1]) if self.world else pos[2]
                if not any(self.sees(w, [p[0], p[1], z + mid]) for w in watchers):
                    self.evade, self.evade_t = p, t
                    return p
        self.evade, self.evade_t = first, t
        return first

    def hunt_goal(self, pos, c, r, t, mine=None):
        """The nearest lone mech on the map (no other within CROWD_R of it) within HUNT_RANGE, inside the safe circle,
        not one we judged strong."""
        rr = max(r - EDGE_MARGIN, 0.0)
        spots = [g["pos"] for g in self.ghosts.values() if g["r"] <= 80 and d2(g["pos"], pos) < HUNT_RANGE and d2(g["pos"], c) < rr
                 and not (mine and g.get("s") and g["s"] > STRONG * mine)]
        return min((q for q in spots if self.crowd(q, t) <= 1), key=lambda q: d2(q, pos), default=None)

    def check_spot(self, pos, goal, t):
        """The last spot of a lost mech near where we are going (or near us) that we could see if we looked, or None."""
        eye = [pos[0], pos[1], pos[2] + self.mech.get("eye_height", 8)]
        for q in self.lost(pos, t, 3):
            if d2(q, pos) < self.cfg["vision"]["range"] - 5 and (d2(q, goal) < CROWD_R or d2(q, pos) < 120) \
                    and self.sees(eye, [q[0], q[1], q[2] + self.mech["height"] / 2]):
                return q
        return None

    def foe_spot(self, pos, t):
        """Where the closest enemy we know of is: a mech listed now (in view, or tracked behind cover), one we lost or
        heard in the last FACE_S, or the sound we just heard; None if we know of none."""
        spots = [g["pos"] for g in self.ghosts.values() if t - g["t"] <= FACE_S]
        if self.heard and t - self.heard[1] < 2:  # the sound itself too: on the map it may have joined another spot
            b, d = self.heard[0], self.heard[2]
            spots.append([pos[0] + d * math.cos(b), pos[1] + d * math.sin(b), pos[2]])
        return min(spots, key=lambda q: d2(q, pos), default=None)

    def look(self, base, t):
        """Scan with nothing to shoot: sweep SWEEP either side of base, and every few seconds a look behind."""
        if t >= self.glance_next:
            self.glance_next = t + self.rng.uniform(*GLANCE_EVERY)
            self.glance_until, self.glance_side = t + GLANCE_S, 1 if math.sin(t * SWEEP_W) >= 0 else -1  # round the near way
        if t < self.glance_until:
            return base + self.glance_side * (math.pi - 0.3)
        return base + SWEEP * math.sin(t * SWEEP_W)

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
            return (x["d"] / 100 + hp * 1.5 + (0 if x["los"] else 1.5) - 0.8 * away - 0.4 * (x["id"] in busy)
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
        if hpf < 0.6:
            return None
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

    def early_laser(self, me, pos, enemies, most=1, avoid=()):
        """In the first minute: the nearest laser seen within EARLY_RANGE (else a lightning) we want and can reach first."""
        for kind in ("laser", "lightning"):
            best = min(((d2(p, pos), p) for pid, (k, p, _) in self.pickups.items()
                        if k == kind and self.bad_pickups.get(pid, -1) < self.t and self.wants(k, me)[0]
                        and d2(p, pos) < EARLY_RANGE and abs(p[2] - pos[2]) < 4  # not on a roof
                        and not any(d2(m["pos"], p) < d2(p, pos) for m in enemies)  # nobody closer to it
                        and self.crowd(p, self.t) <= most and not any(d2(q, p) < CROWD_R for q in avoid)), default=None)
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

    def expo_at(self, p):
        """How many of 24 directions the cell at p can be seen from (the safety map), or None."""
        w = self.world
        if not w or not w.expo:
            return None
        e = w.expo[w.nav_cell(p[0], p[1])]
        return e if e <= 24 else None

    def free_edge_spot(self, pos, c, r):
        """edge_spot, kept inside the map (the first zone is bigger than it), slid along the circle to a walkable cell:
        with the safety map, the safest one within EDGE_SLIDE m along the edge (a red one only if there's nothing else)."""
        spot = edge_spot(pos, c, r, EDGE_MARGIN)
        if not self.world:
            return spot
        spot = self.world.clamp(spot, MAP_MARGIN)
        base = math.atan2(spot[1] - c[1], spot[0] - c[0])
        rr = max(r - EDGE_MARGIN, 0.0)
        da = min(0.05, 4 / max(rr, 1))  # about a 4 m cell along the circle
        best, bs = None, None
        for k in range(0, 40):
            for s in ((1, -1) if k else (1,)):
                p = spot if not k else self.world.clamp([c[0] + math.cos(base + s * k * da) * rr, c[1] + math.sin(base + s * k * da) * rr], MAP_MARGIN)
                if not self.world.free(*p):
                    continue
                e = self.expo_at(p)
                if e is None:
                    return p  # no safety map: the nearest walkable one, as always
                sc = (e if e < RED else 100 + e) + k * da * rr / 10
                if bs is None or sc < bs:
                    best, bs = p, sc
            if best is not None and k * da * rr > EDGE_SLIDE:
                break
        return best or spot

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
            e = self.expo_at(p) or 0
            if d2(p, c) >= rr or d2(p, pos) < 30 or e >= RED:  # never roam to a red cell
                continue
            s = (self.visited.get((int(p[0] // ROAM_CELL), int(p[1] // ROAM_CELL)), -300.0) + d2(p, pos) / 10  # 10 m ~ 1 s
                 + ROAM_EXPO * e)  # a safe waypoint over an exposed one
            if bs is None or s < bs:
                best, bs = p, s
        self.roam, self.roam_t = best, t
        return best

    def hide_spot(self, pos, c, r, t):
        """Where to hide: a walkable spot within HIDE_R, inside the safe circle, out of sight of the mechs we know of
        (the 4 nearest on our map, seen, lost or heard, and a shooter we only felt, assumed 80 m out along the hit's
        bearing), the best for: seen from few directions (the safety map), walls on two sides (a corner), far from
        the mechs on our map, near us; if none of the best few is out of sight, the best of all anyway (less exposed,
        farther off). Redone every HIDE_EVERY s; the old spot stays while it is still out of sight and not clearly
        worse. None if there is no walkable spot."""
        w = self.world
        rr = max(r - EDGE_MARGIN, 0.0)
        if not w:
            return list(pos[:2]) if d2(pos, c) < rr else None
        if t - self.cover_t < HIDE_EVERY:
            return self.cover
        eye, mid, S = self.mech.get("eye_height", 8.0), self.mech["height"] / 2, CELL * NAV
        spots = [g["pos"] for g in self.ghosts.values() if t - g["t"] <= GHOST_S]
        watchers = [[q[0], q[1], q[2] + eye] for q in sorted(spots, key=lambda q: d2(q, pos))[:4]]
        if self.threat and t - self.threat[1] < 5:
            b = self.threat[0]
            watchers.append([pos[0] + 80 * math.cos(b), pos[1] + 80 * math.sin(b), pos[2] + eye])

        def hidden(p):
            q = [p[0], p[1], w.ground_at(p[0], p[1]) + mid]
            return not any(self.sees(o, q) for o in watchers)

        def score(p):
            i = w.nav_cell(p[0], p[1])
            e = w.expo[i] if w.expo and w.expo[i] <= 24 else 12
            danger = sum(max(0.0, 1 - d2(q, p) / DANGER_R) for q in spots)
            return d2(p, pos) / 10 + e + (0 if CORNER[w.walls[i]] == 2 else 6) + 30 * danger
        cands = []
        for J in range(max(int((pos[1] - HIDE_R - w.y0) // S), 0), min(int((pos[1] + HIDE_R - w.y0) // S), w.mny - 1) + 1):
            for I in range(max(int((pos[0] - HIDE_R - w.x0) // S), 0), min(int((pos[0] + HIDE_R - w.x0) // S), w.mnx - 1) + 1):
                if w.cost[J * w.mnx + I] > 1.0:
                    continue
                p = [w.x0 + (I + 0.5) * S, w.y0 + (J + 0.5) * S]
                if d2(p, pos) <= HIDE_R and d2(p, c) < rr and not any(d2(q, p) < HIDE_NEAR for q in spots):
                    cands.append((score(p), p))
        cands.sort(key=lambda e: e[0])
        self.hide_dbg = [[p, sc, None] for sc, p in cands[:HIDE_CHECK[0]]]  # the best few, and which we found out of sight
        best, deadline = None, time.perf_counter() + HIDE_CHECK[1]
        for k, (sc, p) in enumerate(cands[:HIDE_CHECK[0]]):  # the line-of-sight rays cost: only the best few
            self.hide_dbg[k][2] = ok = hidden(p)
            if ok:
                best = (sc, p)
                break
            if time.perf_counter() > deadline:
                break
        if best is None and cands and not (self.cover and hidden(self.cover)):
            best = cands[0]  # nowhere out of sight close by: the least exposed, safest spot anyway
        cur = self.cover
        if cur and d2(cur, c) < rr and not any(d2(q, cur) < HIDE_NEAR for q in spots) and hidden(cur) \
                and (best is None or score(cur) <= best[0] + 3):
            best = (None, cur)  # still good: no wandering between near-equal spots
        self.cover, self.cover_t = (best[1] if best else None), t
        return self.cover

    def safety_cost(self, t):
        """Extra A* cost per step while hiding: cells seen from many directions (the safety map), and cells near the
        mechs on our map."""
        w = self.world
        expo, mnx, x0, y0, S = w.expo, w.mnx, w.x0, w.y0, CELL * NAV
        spots = [g["pos"] for g in self.ghosts.values() if t - g["t"] <= GHOST_S]

        def extra(i):
            e = expo[i] if expo and expo[i] <= 24 else 12
            x, y = x0 + (i % mnx + 0.5) * S, y0 + (i // mnx + 0.5) * S
            return SAFE_W * e / 24 + sum(4.0 * max(0.0, 1 - math.hypot(x - q[0], y - q[1]) / DANGER_R) for q in spots)
        return extra

    def steer(self, pos, goal, t, force=False, extra=None, safe=True):
        """Unit move vector toward goal along an A* path: on safe ground where it can (the safety map's move costs)
        unless not `safe`, with `extra` step costs if given (see World.path)."""
        if not self.world:
            return unit(goal[0] - pos[0], goal[1] - pos[1])
        kind = "extra" if extra else "safe" if safe and self.move_costs else "plain"
        if (force or not self.path or self.path_goal is None or d2(goal, self.path_goal) > 6 or t - self.path_t > 2.5
                or kind != self.path_kind):
            self.path = self.world.path(pos, goal, PATH_BUDGET, extra, self.move_costs if kind == "safe" else None)
            self.path_goal, self.path_t, self.path_kind = list(goal), t, kind
        while len(self.path) > 1 and d2(pos, self.path[0]) < 4:
            self.path.pop(0)
        if not self.path:
            return unit(goal[0] - pos[0], goal[1] - pos[1])
        tgt = self.path[0]
        for wp in self.path[1:6]:  # cut corners where the straight line is walkable
            if not self.world.line_free(pos, wp):
                break
            tgt = wp
        return self.slide(pos, unit(tgt[0] - pos[0], tgt[1] - pos[1]), self.path[1] if len(self.path) > 1 else goal)

    def slide(self, pos, d, nxt):
        """Hug the wall: if the body would hit something SLIDE_AHEAD m along d, turn toward open ground in steps,
        on the side of the waypoint after this one; head-on into a wall or a corner we'd just stand there."""
        w, (x, y, z) = self.world, pos
        if w.body_free(x + d[0] * SLIDE_AHEAD, y + d[1] * SLIDE_AHEAD, z):
            return d
        a0 = math.atan2(d[1], d[0])
        side = 1 if ang_diff(math.atan2(nxt[1] - y, nxt[0] - x), a0) >= 0 else -1
        for k in range(1, 9):  # 20 deg steps, up to 160 deg: the preferred side first, then the other
            for s in (side, -side):
                a = a0 + s * k * math.radians(20)
                if w.body_free(x + math.cos(a) * SLIDE_AHEAD, y + math.sin(a) * SLIDE_AHEAD, z):
                    return [math.cos(a), math.sin(a)]
        return d

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

    def loot_goal(self, me, pos, c, r, most=1, avoid=()):
        """The nearest wanted loot on the ground (roof loot is roof_target's) that at most `most` mechs on the map guard
        and no strong mech (`avoid`) is near; an item in the open counts LOOT_EXPO m farther per direction it can be
        seen from."""
        return min((p for p in self.loot_items(me) if d2(p, c) < r - 6 and d2(p, pos) < LOOT_RANGE and not self.on_roof(p)
                    and self.crowd(p, self.t) <= most and not any(d2(q, p) < CROWD_R for q in avoid)),
                   key=lambda p: d2(p, pos) + LOOT_EXPO * (self.expo_at(p) or 0), default=None)

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
        self.clear_ghosts(me["aim"][0], pos, t)
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
        if hpf < HIDE_HP:
            self.hiding = True
        elif hpf >= HEALED_HP:
            self.hiding = False
        healthy = not self.hiding and hpf >= HEALED_HP
        mine = max(self.strength(me["hp"], me.get("cubes", 0), [w["kind"] for w in me["weapons"] if w]), 1.0)
        T = self.targets(obs, me, muzzle, hpf)  # no running off in the open: a laser hits a runner just the same
        if not healthy and T and T["kind"] != "mech" and t - self.hit_by.get(T["id"], -1e9) >= 3:
            T = None  # hurt: no farming crates or drones (regen needs 5 s without dealing damage)
        aim, fire, reload = self.weapons(me, muzzle, tube, T, t)
        if self.turn_to and t - self.turn_to[1] < 1.0 and not (T and T["kind"] == "mech" and T["los"] and T["id"] in self.hit_by
                                                              and t - self.hit_by[T["id"]] < 1.0):
            aim = [self.turn_to[0], 0.0]  # hit in the back (x1.5) by someone we don't have in front of us: turn round now
            fire = [bool(w and w["kind"] == "laser" and w.get("charge", 0) > 0) for w in me["weapons"]]  # nothing leaves off-aim; a charge is held
            why = "turn"
        elif aim is not None:
            why = "target"
            if T["kind"] == "mech" and not T["los"] and t - self.hit_by.get(T["id"], -1e9) >= 3:
                aim = None  # behind cover there's no shot: look at the closest enemy we know of below (it may be this one)

        strong_T = bool(T and T["kind"] == "mech" and ((self.ghosts.get(T["id"]) or {}).get("s") or 0) > STRONG * mine)
        if (not healthy or strong_T) and not (T and T["kind"] == "mech" and (t - self.hit_by.get(T["id"], -1e9) < 5 or T["d"] < 40)):
            # hurt (regen needs 5 s without dealing damage), or it's strong: we start no fight; shoot only whoever is on us; a charge is held
            fire = [bool(w and w["kind"] == "laser" and w.get("charge", 0) > 0) for w in me["weapons"]]

        # Movement goal.
        c, r = zone_at(obs["zone"], t + LOOKAHEAD_S)
        spot = self.free_edge_spot(pos, c, r)
        safe_r = max(r - EDGE_MARGIN, 0.0)
        arrive, mode, goal, move = ARRIVED, "edge", spot, None
        engaged = False
        dodge = self.dodge(obs, me, pos)
        if dodge:
            move, mode = dodge
        elif not me["in_zone"]:
            mode = "zone"
        else:
            strong = self.strong_spots(t, mine)  # where the strong mechs we know of are: we keep away from them
            drop = None
            for dr in obs.get("drops", []):
                dd = d2(dr["pos"], pos)
                if (dr["holder"] in (None, me["id"]) and dd < DROP_RANGE and d2(dr["pos"], c) < r - 3
                        and dr["lands_in"] <= dd / st.get("walk_speed", 11.7) + 10
                        and self.crowd(dr["pos"], t, 60) <= (1 if healthy else 0)  # hurt: only an unguarded one
                        and not any(d2(q, dr["pos"]) < CROWD_R for q in strong)):
                    drop = dr
                    break
            most = 1 if healthy else 0  # mechs we take on for an item
            shot_at = self.threat is not None and t - self.threat[1] < 3
            engaged = bool(T and T["kind"] == "mech" and T["los"]
                           and (T["d"] < FIGHT_RANGE or t - self.hit_by.get(T["id"], -1e9) < 3))
            # shot by a mech we can see: stand and trade (running shows it our back, x1.5); hide once it lets up
            under_fire = bool(T and T["kind"] == "mech" and T["los"] and t - self.hit_by.get(T["id"], -1e9) < 3)
            near = [g["pos"] for g in self.ghosts.values() if d2(g["pos"], pos) < EVADE_R]  # seen or heard close by
            out = not self.hiding and (healthy or not near)  # hurt, we go for things only while the map is clear around us
            loot = self.loot_goal(me, pos, c, r, most, strong) if out else None
            early = self.early_laser(me, pos, enemies, most, strong) if t < EARLY_S and not shot_at and out else None
            hide = self.hide_spot(pos, c, r, t) if self.hiding and not drop and not under_fire else None
            # keep away from strong mechs near us; hurt, from any mech near us
            threats = [q for q in strong if d2(q, pos) < AVOID_R] + ([] if healthy else near)
            evade = (self.evade_spot(pos, c, r, threats, t) if threats and not self.hiding and not under_fire
                     and not drop else None)
            chase = self.chase_goal(pos, c, r, t, mine, strong) if healthy and not evade else None
            calm = healthy and not shot_at and not engaged and self.crowd(pos, t, ROOF_CALM) == 0
            hunt = self.hunt_goal(pos, c, r, t, mine) if healthy and not (drop or early or loot or chase) else None
            if self.climb and (t - self.climb["t0"] > ROOF_TIMEOUT or (me["grounded"] and not calm and not self.on_roof(pos))):
                self.climb = None  # trouble while still on the ground: forget the roof
            if not self.climb and calm and not (drop or hide or early or loot):
                rp = self.roof_target(me, pos, c, r)
                if rp:
                    self.climb = dict(p=rp, t0=t)
            if drop:
                goal, mode, arrive = drop["pos"], "drop", max(drop["radius"] - 3, 1.0)
            elif hide:
                goal, mode, arrive = hide, "hide", 1.5
            elif evade:
                goal, mode, arrive = evade, "evade" if not healthy else "avoid", 3.0
            elif chase:
                goal, mode, arrive = chase, "chase", 5.0  # a weak mech on its own: run it down
            elif early:
                goal, mode, arrive = early, "laser", 1.0
            elif loot:
                goal, mode, arrive = loot, "loot", 1.0
            elif self.climb:
                goal, mode, arrive = self.climb["p"], "roof", 1.0
            elif engaged and (healthy or under_fire):
                goal, mode = list(pos[:2]), "fight"  # stand and trade, strafing (running off loses fights)
            elif healthy and hunt:
                goal, mode, arrive = hunt, "hunt", 30.0  # a lone mech: go and take it on
            elif self.hiding or (near and not healthy):
                pass  # no cover / no way away to be had: wait at the edge spot (enemies come from inside)
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
            move = self.steer(pos, goal, t, force=self.stuck_since is not None and t - self.stuck_since > 1,
                              extra=self.safety_cost(t) if mode == "hide" and self.world else None,
                              safe=mode not in ("zone", "dodge shell")) if gap > arrive else [0.0, 0.0]  # the zone won't wait
            duel = engaged and mode in ("loot", "laser", "chase") and gap > 8  # trading shots on the way: zigzag there
            if (mode in ("edge", "fight") or duel) and T and T["kind"] == "mech" and T["los"]:  # strafe across their line of fire
                if t >= self.strafe_until:
                    self.strafe_sign, self.strafe_until = -self.strafe_sign, t + self.rng.uniform(0.35, 0.9)  # reverse before their led shot lands
                yaw = math.atan2(T["pos"][1] - y, T["pos"][0] - x) + self.strafe_sign * math.pi / 2
                pull = min(gap / 6, 1.0)
                to_spot = move if duel else unit(goal[0] - x, goal[1] - y)  # the item along the path
                move = unit(math.cos(yaw) + to_spot[0] * pull, math.sin(yaw) + to_spot[1] * pull)

        # Stuck: commanded to move but not getting anywhere -> jump and jet; if that fails too, sidestep.
        # Not while fighting inside the zone (hopping about loses the fight); outside it the zone kills faster than they do.
        fighting = bool(T and T["kind"] == "mech" and T["los"] and me["in_zone"])
        wants_move = math.hypot(*move) > 0.3
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
            gap > 40 and me["energy"] > 60 and not T and mode != "roof" and healthy)  # save the energy for the jet; hurt: quiet
        if self.hiding and not stuck and t >= self.unstick_until and mode in ("hide", "edge"):
            move = [move[0] * SILENT, move[1] * SILENT]  # hiding: creep under step_speed, no step sounds
        if fighting:  # boost through the whole fight, walking just long enough to recharge (duel-tested; jets lost)
            self.burn = me["energy"] > 70 if not self.burn else me["energy"] > 25
            boost = boost or (self.burn and me["grounded"])

        # Where to look when there's nothing to shoot: at the enemy whenever we know of one; else look round.
        if aim is None:
            foe = self.foe_spot(pos, t)
            chk = None if foe else self.check_spot(pos, goal, t)
            if self.threat and t - self.threat[1] < 3:  # turn to face whoever hit us
                aim, why = [self.threat[0], 0.0], "hit"
            elif foe:  # the closest mech we see behind cover, lost or heard
                aim, why = [math.atan2(foe[1] - y, foe[0] - x), 0.0], "enemy"
            elif t < 1.6:  # a full turn at spawn (turn_rate 4 rad/s): see the loot behind us too
                aim, why = [me["aim"][0] + 1.0, 0.0], "spawn"
            elif chk:  # an older ghost near where we're going: is it still there?
                aim, why = [math.atan2(chk[1] - y, chk[0] - x), 0.0], "ghost"
            else:
                lost = self.lost(pos, t, 1)
                base = (math.atan2(move[1], move[0]) if math.hypot(*move) > 0.1
                        else math.atan2(lost[0][1] - y, lost[0][0] - x) if self.hiding and lost
                        else math.atan2(c[1] - y, c[0] - x))  # walking: the way we go; standing: the danger, else inward
                aim = [self.look(base, t), 0.0]
                why = "glance" if t < self.glance_until else "scan"

        act.update(move=move, boost=boost, jump=jump, aim=aim, fire=fire, reload=reload)
        if DEBUG and obs["tick"] % 200 == 0 and self.dbg:
            mf.log("gun", sorted(self.dbg.items()))
        self.draw(pos, muzzle, goal, T, mode, me, why)
        return act

    def draw(self, pos, muzzle, goal, T, mode, me, why):
        x, y, z = pos
        prev = [x, y, z + 1]
        for wp in self.path[:8]:
            q = [wp[0], wp[1], z + 1]
            mf.draw.line(prev, q, "#40c0ff")
            prev = q
        mf.draw.point([goal[0], goal[1], z], "#40c0ff", 6)
        if T:
            mf.draw.line(muzzle, T["c"], "#ff5040" if T["los"] else "#806060")
        mf.draw.label(f"{mode} · hp {me['hp']:.0f} · {T['kind'] + ' ' + str(T['id']) if T else '-'} · eye {why}")


if __name__ == "__main__":
    mf.run(Bot)
