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
# Roaming heads for cells we haven't been to, inside the safe circle; hiding without a map stays put.
b = Bot.__new__(Bot)
b.rng, b.world, b.visited, b.roam, b.roam_t = random.Random(2), None, {}, None, -1e9
b.threat, b.cover, b.cover_t, b.mech = None, None, -1e9, {"height": 9}
b.ghosts = {}
for x in range(-200, 1, 40):  # the west half is freshly visited
    for y in range(-200, 201, 40):
        b.visited[(x // 40, y // 40)] = 100.0
g = b.roam_goal([0, 0, 0], [0, 0], 250, 100.0)
assert g and math.hypot(*g) < 233 and g[0] > 0, g
assert b.roam_goal([1, 1, 0], [0, 0], 250, 101.0) == g  # keeps its waypoint until there
assert b.cover_spot([5, 6, 0], [0, 0], 250, [{"pos": [50, 0, 0]}], 0.0) == [5, 6]
# Ghosts: a mech that left our view stays where we lost it; it counts near items, and looking there with
# nothing listed forgets it. With nothing to shoot we sweep either side of the way we face, and glance behind.
g = Bot.__new__(Bot)
g.cfg, g.mech, g.world, g.rng = {"vision": {"range": 200, "near_range": 45, "fov_deg": 120}}, {"height": 9, "eye_height": 8}, None, random.Random(0)
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
print("ok")
