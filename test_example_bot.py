"""Self-check for the bot's math and map model: python3 test_example_bot.py"""
import math
import os
import random

from example_bot import Bot, ang_diff, edge_spot, lead, mortar_aim, shell_landing, zone_at

zone = {"center": [0, 0], "radius": 500, "next_center": [100, 0], "next_radius": 300, "shrink_start": 60, "shrink_end": 110}
assert zone_at(zone, 0) == ([0, 0], 500)
assert zone_at(zone, 85) == ([50, 0], 400)
assert zone_at(zone, 200) == ([100, 0], 300)

assert edge_spot([10, 0], [0, 0], 100, 10) == [90, 0]       # inside: walk outward to the edge
assert edge_spot([0, -300], [0, 0], 100, 10) == [0, -90]    # outside: walk in to the edge
assert edge_spot([0, 0], [0, 0], 100, 10) == [90, 0]        # dead center: any direction works

# A bullet aimed at the lead point must meet the target at the same moment.
src, tpos, tvel, speed = [0, 0, 7], [200, 0, 4.5], [0, 15, 0], 252
p = lead(src, tpos, tvel, speed)
t = math.dist(src, p) / speed
assert math.dist(p, [tpos[i] + tvel[i] * t for i in range(3)]) < 0.01

# A shell fired with mortar_aim must land on the (moving) target, on both arcs.
v, g = 72, 22
for high in (False, True):
    tube, feet, vel = [0, 0, 9.5], [120, 30, 2], [5, -3, 0]
    (yaw, pitch), ft = mortar_aim(tube, feet, vel, v, g, high=high, max_pitch=1.5)
    sx = tube[0] + v * math.cos(pitch) * math.cos(yaw) * ft
    sy = tube[1] + v * math.cos(pitch) * math.sin(yaw) * ft
    sz = tube[2] + v * math.sin(pitch) * ft - g * ft * ft / 2
    assert math.dist([sx, sy, sz], [feet[0] + vel[0] * ft, feet[1] + vel[1] * ft, feet[2]]) < 0.5, high
assert mortar_aim([0, 0, 0], [1000, 0, 0], [0, 0, 0], v, g) is None  # out of range

# shell_landing inverts a ballistic flight.
x, y, lt = shell_landing([0, 0, 50], [10, 0, 5], g, 0)
assert abs(50 + 5 * lt - g * lt * lt / 2) < 1e-9 and abs(x - 10 * lt) < 1e-9

# Map model, when the kit's arena folder is next to this one.
arena = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "arena")
if os.path.isdir(arena):
    from world import World
    w = World(arena, {"height": 9, "step_up": 1.2, "max_slope_deg": 50})
    top = max(range(len(w.hi)), key=lambda i: w.hi[i])  # a cell inside the tallest building
    cx, cy = w.x0 + (top % w.nx + 0.5) * 2, w.y0 + (top // w.nx + 0.5) * 2
    zmid = (w.lo[top] + w.hi[top]) / 2
    assert not w.clear([cx - 40, cy, zmid], [cx + 40, cy, zmid])            # through the building: blocked
    assert w.clear([cx - 40, cy, w.hi[top] + 20], [cx + 40, cy, w.hi[top] + 20])  # well over it: clear
    assert not w.free(cx, cy)
    rng = random.Random(1)
    free = [q for q in ([rng.uniform(-300, 300), rng.uniform(-200, 200)] for _ in range(400)) if w.free(*q)]
    reached = 0
    for a, b in zip(free[::2], free[1::2]):
        if math.dist(a, b) < 150:
            path = w.path(a, b, 0.2)
            assert path and all(math.dist(p, q) < 9 for p, q in zip([a] + path, path)), "path steps are neighbor cells"
            reached += math.dist(path[-1], b) < 1
    assert reached > 0
    # Roof loot: an item on a 5-45 m roof counts as on a roof, and its launch spot is walkable ground near it.
    rb = Bot.__new__(Bot)
    rb.world = w
    roofs = [i for i in range(len(w.hi)) if 5 < w.hi[i] - w.ground[i] < 45 and not w.free(w.x0 + (i % w.nx + 0.5) * 2, w.y0 + (i // w.nx + 0.5) * 2)]
    i = roofs[len(roofs) // 2]
    item = [w.x0 + (i % w.nx + 0.5) * 2, w.y0 + (i // w.nx + 0.5) * 2, w.hi[i]]
    assert rb.on_roof(item) and not rb.on_roof([item[0], item[1], w.ground[i]])
    ls = rb.launch_spot(item)
    assert ls and w.free(*ls) and math.dist(ls, item[:2]) < 32, ls
    # Hiding: the shipped safety map loads for this map only; corners are where walls meet. With a mech 60 m off,
    # the bot picks a spot within 60 m with a wall between them, and the way there costs more near that mech.
    import json as _json
    import tempfile as _tmp
    from world import CORNER
    here = os.path.dirname(os.path.abspath(__file__))
    assert w.load_safety(os.path.join(here, "safety.json")) is not None and w.expo
    other = _tmp.NamedTemporaryFile("w", suffix=".json", delete=False)
    _json.dump({"map": [0, 0, 0, 0, 0], "cell": 4.0, "data": "A" * (w.mnx * w.mny)}, other)
    other.close()
    assert w.load_safety(other.name) is None and w.expo  # made for another map: ignored, the good one stays
    os.unlink(other.name)
    assert sum(CORNER[m] == 2 for m in w.walls) > 100
    hb = Bot.__new__(Bot)
    hb.world, hb.mech, hb.threat, hb.cover, hb.cover_t, hb.hide_dbg = w, {"height": 9, "eye_height": 8}, None, None, -1e9, []
    hb.marks = {}
    rng = random.Random(3)
    cells = [i for i in range(len(w.expo)) if 8 <= w.expo[i] <= 20 and w.cost[i] <= 1.0]

    def seen_by(e, q):
        eye, g = [e[0], e[1], w.ground_at(*e) + 8], w.ground_at(*q)
        return w.clear(eye, [q[0], q[1], g + 4.5]) or w.clear(eye, [q[0], q[1], g + 9])  # a line of fire: trees don't stop it
    while True:  # a spot half in the open, a mech 60 m off, and somewhere within 60 m out of its sight
        p = w.nav_center(rng.choice(cells))
        a = rng.uniform(0, 2 * math.pi)
        e = [p[0] + 60 * math.cos(a), p[1] + 60 * math.sin(a)]
        if w.free(*e) and any(not seen_by(e, q) for q in map(w.nav_center, range(len(w.cost)))
                              if math.dist(q, p) <= 60 and math.dist(q, e) >= 30 and w.free(*q)):
            break
    hb.ghosts = {1: {"pos": [e[0], e[1], w.ground_at(*e)], "t": 0.0, "r": 0.0}}
    spot = hb.hide_spot([p[0], p[1], w.ground_at(*p)], [0, 0], 5000, 0.0)
    assert spot and w.free(*spot) and math.dist(spot, e) >= 30 and not seen_by(e, spot), spot
    cost = hb.safety_cost(0.0)
    assert cost(w.nav_cell(*e)) > cost(w.nav_cell(*spot)) and w.path(p, spot, 0.05, cost)
    # Fight or run by the ground: on ground open to fire from fewer directions we take on stronger mechs; an item
    # in the open next to a mech as strong as us is guarded, the same item on safe ground isn't.
    from example_bot import FIGHT_OPEN, FIGHT_SAFE
    hb.cfg = {"cubes": {"damage_per_cube": 0.04, "stat_cap": 10}}
    walk = [i for i in range(len(w.cost)) if w.cost[i] <= 1.0]
    safe = w.nav_center(min(walk, key=lambda i: w.expo[i]))
    openg = w.nav_center(next(i for i in walk if w.expo[i] == 24))
    assert FIGHT_OPEN <= hb.fight_limit(openg) < 1.0 < hb.fight_limit(safe) <= FIGHT_SAFE
    me_s = hb.strength(1400, 0, ["gun"])
    for spot, guarded in ((openg, True), (safe, False)):
        hb.ghosts = {7: {"pos": [spot[0] + 20, spot[1], 0], "t": 0.0, "r": 0.0, "s": me_s}}  # our equal, 20 m off
        assert hb.guarded(spot, 0.0, me_s) is guarded, spot
# A close fight: seen within 60 m locks us in; it holds while that mech is listed within 90 m (even behind cover) and
# ends 2 s after it isn't.
f = Bot.__new__(Bot)
f.close = None
foe = {"id": 4, "kind": "mech", "los": True, "d": 50.0}
assert not f.close_fight(dict(foe, d=80.0), [], [0, 0, 0], 10.0)        # 80 m: far, we may still choose
assert f.close_fight(foe, [], [0, 0, 0], 10.0)                         # 50 m in sight: locked
assert f.close_fight(None, [{"id": 4, "pos": [85, 0, 0]}], [0, 0, 0], 11.0)  # out of sight, still listed at 85 m
assert f.close_fight(None, [], [0, 0, 0], 12.5)                        # gone 1.5 s: still on
assert not f.close_fight(None, [], [0, 0, 0], 13.5)                    # gone 2.5 s: over
# Roaming heads for cells we haven't been to, inside the safe circle; hiding without a map stays put.
b = Bot.__new__(Bot)
b.rng, b.world, b.visited, b.roam, b.roam_t = random.Random(2), None, {}, None, -1e9
b.threat, b.cover, b.cover_t, b.mech = None, None, -1e9, {"height": 9}
b.ghosts = {}
b.marks = {}
for x in range(-200, 1, 40):  # the west half is freshly visited
    for y in range(-200, 201, 40):
        b.visited[(x // 40, y // 40)] = 100.0
g = b.roam_goal([0, 0, 0], [0, 0], 250, 100.0)
assert g and math.hypot(*g) < 233 and g[0] > 0, g
assert b.roam_goal([1, 1, 0], [0, 0], 250, 101.0) == g  # keeps its waypoint until there
assert b.hide_spot([5, 6, 0], [0, 0], 250, 0.0) == [5, 6]  # no map: hide where we stand
# Ghosts: a mech that left our view stays where we lost it; it counts near items, and looking there with
# nothing listed forgets it. With nothing to shoot we sweep either side of the way we face, and glance behind.
g = Bot.__new__(Bot)
g.cfg, g.mech, g.world, g.rng = {"vision": {"range": 200, "near_range": 45, "fov_deg": 120}}, {"height": 9, "eye_height": 8}, None, random.Random(0)
g.marks = {}
g.ghosts = {1: {"pos": [100, 0, 0], "t": 5.0, "r": 0.0}, 2: {"pos": [-100, 0, 0], "t": 5.0, "r": 0.0}, 3: {"pos": [0, 30, 0], "t": 10.0, "r": 0.0}}
assert g.crowd([90, 0], 10.0) == 1 and g.crowd([0, 0], 10.0, 20) == 0 and g.crowd([0, 0], 10.0, 31) == 1
g.clear_ghosts(0.0, [0, 0, 0], 10.0)
assert sorted(g.ghosts) == [2, 3]  # the one in front is gone; the one behind and the one listed now stay
assert g.check_spot([0, 0, 0], [-90, 0], 10.0) == [-100, 0, 0]  # going its way: look there first
g.glance_next, g.glance_until, g.glance_side = 3.0, -1.0, 1
assert abs(math.pi - abs(ang_diff(g.look(0.0, 3.0), 0.0))) < 0.31  # a look behind...
assert abs(g.look(0.0, 3.0 + 1.1)) <= 1.2 + 1e-9                    # ...then the sweep again
# Hearing: a mech sound goes on the map along its bearing, at the middle of its distance class (a step carries 40 m);
# the next step of the same mech moves that spot instead of adding one; a far gunshot is a rough spot far off.
g.cfg["sound"] = {"near": 60, "mid": 160, "bearing_noise_deg": 7, "range": {"step": 40, "gun": 320}}
g.ghosts, g.heard_n = {}, 0
g.hear({"kind": "step", "bearing": 0.0, "dist": "near", "age": 0.0}, [0, 0, 0], 20.0)
(h,) = g.ghosts.values()
assert abs(h["pos"][0] - 20) < 1e-9 and abs(h["pos"][1]) < 1e-9 and 20 < h["r"] < 30, h
g.hear({"kind": "step", "bearing": 0.1, "dist": "near", "age": 0.0}, [0, 0, 0], 20.6)
assert len(g.ghosts) == 1 and h["t"] == 20.6
g.hear({"kind": "gun", "bearing": math.pi / 2, "dist": "far", "age": 0.0}, [0, 0, 0], 21.0)
far = max(g.ghosts.values(), key=lambda e: e["pos"][1])
assert len(g.ghosts) == 2 and abs(far["pos"][1] - 240) < 1e-6 and far["r"] > 80
# Hurt, we walk away from the mech we heard (out of the circle is no way out); healthy we go for a lone mech, not a pair.
g.evade, g.evade_t = None, -1e9
e = g.evade_spot([0, 0, 0], [0, 0], 500, [h["pos"]], 21.0)
assert e[0] < 0 and math.dist(e, h["pos"][:2]) > 50, e
assert g.hunt_goal([0, 0, 0], [0, 0], 500, 21.0) == h["pos"]  # the far gunshot is too rough to go after
g.ghosts[("h", 99)] = {"pos": [30, 10, 0], "t": 21.0, "r": 0.0}
assert g.hunt_goal([0, 0, 0], [0, 0], 500, 21.0) is None  # two together: leave them
# Looking: with nothing to shoot we face the closest enemy we know of (fresh on the map, or the sound just heard);
# knowing of none (all older than FACE_S), we scan.
g.heard = (math.pi, 21.0, 15.0)  # a step 15 m behind us
assert g.foe_spot([0, 0, 0], 21.5) == [-15.0, math.sin(math.pi) * 15.0, 0]
g.heard = None
assert g.foe_spot([0, 0, 0], 21.5) == h["pos"]  # the heard step at 20 m beats the pair at 30 m and the far gunshot
assert g.foe_spot([0, 0, 0], 40.0) is None
# Strength: HP x cube damage bonus x firepower. Next to us at full HP with only the gun, a mech with 10 cubes and a
# laser is strong (keep away), one at 30% HP with the gun is weak (run it down, while it is alone).
g.cfg["cubes"] = {"damage_per_cube": 0.04, "stat_cap": 10}
mine = g.strength(1400, 0, ["gun"])
big, small = g.strength(1960, 10, ["gun", "laser"]), g.strength(420, 0, ["gun"])
assert big > 1.4 * mine and small < 0.7 * mine and 0.7 * mine < g.strength(1400, 0, ["gun"]) < 1.4 * mine
g.ghosts = {7: {"pos": [120, 0, 0], "t": 50.0, "r": 0.0, "s": big}, 8: {"pos": [-100, 0, 0], "t": 50.0, "r": 0.0, "s": small}}
# Marks: the strong one turns the ground round it dangerous and the weak one's attractive, for minutes after; a flee
# spot lies away from the strong one; a lone mech on dangerous ground isn't hunted.
g.marks = {}
g.mark([{"id": 7, "pos": [120, 0, 0]}, {"id": 8, "pos": [-100, 0, 0]}], 50.0, mine, 1.4)
assert {k: m["kind"] for k, m in g.marks.items()} == {7: "strong", 8: "weak"}
assert g.field([120, 0], 50.0) == 1.0 and g.field([300, 0], 50.0) == 0.0 and abs(g.field([120, 0], 140.0) - 0.5) < 1e-9
assert g.field([-100, 0], 50.0, "weak") == 1.0
g.mark([], 260.0, mine, 1.4)
assert g.marks == {}  # 3 minutes on, unseen: forgotten
g.mark([{"id": 7, "pos": [120, 0, 0]}], 50.0, mine, 1.4)
fs = g.flee_spot([0, 0, 0], [0, 0], 1000, [0, 0], 1000, 50.0)
assert fs and fs[0] < 0 and math.hypot(*fs) >= 180, fs
g.ghosts[8]["pos"] = [100, 30, 0]  # the weak one next to the strong one: not worth the walk
assert g.hunt_goal([0, 0, 0], [0, 0], 500, 50.0, mine) is None
print("ok")
