"""Self-check for the bot's math and map model: python3 test_example_bot.py"""
import math
import os
import random

from example_bot import Bot, edge_spot, lead, mortar_aim, shell_landing, zone_at

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
    # Bird's flight: from that launch spot toward the roof item the building is in the way and needs jet energy;
    # standing at its foot with no energy we wait; across open ground nothing needs it.
    rb.mech = {}
    me = {"vel": [0, 0, 0], "grounded": True, "energy": 100}
    start = [ls[0], ls[1], w.ground_at(*ls)]
    mv, jet, reserve = rb.travel(me, start, item)
    assert reserve > 0 and abs(math.hypot(*mv) - 1) < 1e-9, (mv, reserve)
    foot = next(p for p in ([start[0] + (item[0] - start[0]) * k / 40, start[1] + (item[1] - start[1]) * k / 40] for k in range(40))
                if w.top_under(*p) > start[2] + 3)  # first spot where the body touches the building
    back = [foot[0] - (item[0] - start[0]) / math.dist(start[:2], item[:2]) * 1.5, foot[1] - (item[1] - start[1]) / math.dist(start[:2], item[:2]) * 1.5]
    assert rb.travel(dict(me, energy=0), [back[0], back[1], start[2]], item)[:2] == ([0.0, 0.0], False)
    assert rb.travel(dict(me, energy=100), [back[0], back[1], start[2]], item)[1] is True
    def flat(a, b):  # nothing along a -> b stands more than half a metre above a's ground
        return all(w.top_under(a[0] + (b[0] - a[0]) * k / 30, a[1] + (b[1] - a[1]) * k / 30) < w.ground_at(*a) + 0.5 for k in range(31))
    open_a, near_b = next((q, [q[0] + 15, q[1]]) for q in free if flat(q, [q[0] + 15, q[1]]))
    assert rb.travel(me, [open_a[0], open_a[1], w.ground_at(*open_a)], near_b)[1:] == (False, 0.0)
# Roaming heads for cells we haven't been to, inside the safe circle; hiding without a map stays put.
b = Bot.__new__(Bot)
b.rng, b.world, b.visited, b.roam, b.roam_t = random.Random(2), None, {}, None, -1e9
b.threat, b.cover, b.cover_t, b.mech = None, None, -1e9, {"height": 9}
for x in range(-200, 1, 40):  # the west half is freshly visited
    for y in range(-200, 201, 40):
        b.visited[(x // 40, y // 40)] = 100.0
g = b.roam_goal([0, 0, 0], [0, 0], 250, 100.0)
assert g and math.hypot(*g) < 233 and g[0] > 0, g
assert b.roam_goal([1, 1, 0], [0, 0], 250, 101.0) == g  # keeps its waypoint until there
assert b.cover_spot([5, 6, 0], [0, 0], 250, [{"pos": [50, 0, 0]}], 0.0) == [5, 6]
print("ok")
