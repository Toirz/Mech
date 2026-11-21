"""Self-check for the bot's math and map model: python3 test_example_bot.py"""
import math
import os
import random

from example_bot import ang_diff, bullet_miss, edge_spot, lead, mortar_aim, shell_landing, zone_at

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

# bullet_miss: time and offset of the closest approach.
tca, miss = bullet_miss([0, 0, 5], [100, 0, 0], [50, 2, 5])
assert abs(tca - 0.5) < 1e-9 and math.dist(miss, [0, 2, 0]) < 1e-9
assert bullet_miss([0, 0, 5], [-100, 0, 0], [50, 2, 5])[0] < 0  # flying away
assert bullet_miss([0, 0, 0], [0, 0, 0], [1, 1, 1])[0] < 0      # not moving

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
print("ok")

# Ghosts: a sound from out of view leaves a guess; we turn to it, and once we see the spot empty it is dropped.
stream = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "check", "stream.jsonl.gz")
if os.path.exists(stream):
    import copy, gzip, json
    from example_bot import Bot
    msgs = [json.loads(l) for _, l in zip(range(3), gzip.open(stream, "rt"))]
    hello = msgs[0]
    hello["map"]["dir"] = "/nonexistent"  # no map: every line is clear, so the test is about the logic only
    bot = Bot(hello)
    base = next(m for m in msgs if m.get("type") == "obs")
    for k in ("mechs", "projectiles", "crates", "cubes", "pickups", "sounds", "beams", "events", "drops"):
        base[k] = []
    base["offer"] = None
    me = base["self"]
    me["aim"] = [math.pi, 0.0]  # looking west
    o = copy.deepcopy(base)
    o["t"], o["tick"] = 10.0, 200
    o["sounds"] = [{"kind": "laser", "bearing": 0.0, "dist": "mid", "age": 0.0}]  # east, 60-160 m
    a = bot.act(o)
    assert len(bot.ghosts) == 1 and abs(bot.ghosts[0][0] - (me["pos"][0] + 110)) < 1e-6
    assert abs(ang_diff(a["aim"][0], 0.0)) < 1e-6, a["aim"]  # turn to it
    o = copy.deepcopy(base)
    o["t"], o["tick"] = 11.0, 220
    o["self"]["aim"] = [0.0, 0.0]  # now facing it, nobody there
    bot.act(o)
    assert bot.ghosts == [], bot.ghosts
    print("ghosts ok")
