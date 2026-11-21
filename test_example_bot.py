"""Self-check for the bot's math and map model: python3 test_example_bot.py"""
import math
import os
import random

from example_bot import edge_spot, lead, mortar_aim, shell_landing, zone_at

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

# Fight-or-flight estimate, with the real rules from the kit's recorded match when it's next to this folder.
stream = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "check", "stream.jsonl.gz")
if os.path.isfile(stream):
    import gzip
    import json
    from example_bot import Bot
    with gzip.open(stream, "rt") as f:
        hello = json.loads(f.readline())
    hello["map"]["dir"] = arena
    bot, W = Bot(hello), hello["config"]["weapons"]
    me = {"hp": 1400, "max_hp": 1400, "cubes": 0, "stats": {"armor": 0}, "weapons": [{"kind": "gun", "stats": W["gun"]}, None]}
    def foe(hp, at): return {"id": 5, "pos": at, "hp": hp, "max_hp": 1400, "cubes": 0, "weapons": ["gun"]}
    def wins(*foes): return bot.can_win(me, bot.threats(me, [0, 0, 0], list(foes)), 0.2)
    assert not wins(foe(1400, [100, 0, 0]))                       # an even duel: don't
    assert wins(foe(350, [100, 0, 0]))                            # it's nearly dead: do
    assert not wins(foe(500, [100, 0, 0]), foe(500, [0, 100, 0]))  # two half-dead ones at once: don't
    assert wins(foe(1400, [400, 0, 0]))                           # out of its reach: nothing to lose
    me["weapons"][1] = {"kind": "lightning", "stats": W["lightning"]}
    assert wins(foe(1400, [35, 0, 0]))                            # in arc range against a gun: do

    # Modes, on the recorded first observation with the room emptied (no map: every line is clear).
    with gzip.open(stream, "rt") as f:
        f.readline()
        o = json.loads(f.readline())
    o.update(mechs=[], events=[], sounds=[], beams=[], projectiles=[], pickups=[], cubes=[], crates=[], drops=[])
    bot = Bot(dict(hello, map={"dir": "/nonexistent", "files": []}))
    def tick(**kw):
        o.update(kw, t=o["t"] + 0.1, tick=o["tick"] + 2)
        bot.act(o)
        return bot.mode
    px, py, pz = o["self"]["pos"]
    def mech(i, dx, hp=1400, aim=math.pi): return {"id": i, "kind": "mech", "pos": [px + dx, py, pz], "vel": [0, 0, 0],
                                                   "aim": [aim, 0], "hp": hp, "max_hp": 1400, "cubes": 0, "weapons": ["gun"]}
    assert tick() == "hide"                                        # we always start hidden
    assert tick(mechs=[mech(7, 150, aim=0.0)]) == "hide"           # 150 m off, looking away: unseen, stay hidden
    assert tick(mechs=[mech(7, 150)]) == "counter"                 # it turned our way: it sees us
    assert all(tick(mechs=[]) == "counter" for _ in range(15))     # nobody on us yet for 2 s: keep running
    assert [tick() for _ in range(6)][-1] == "hide"                # 2 s with nobody on us: hide again
    assert tick(mechs=[mech(8, 100, hp=1400, aim=0.0), mech(9, 130, hp=100, aim=0.0)]) == "hide"  # nobody fighting
    bot.hurt_t[9] = o["t"]                                         # one of them is being hit: a fight
    assert tick(mechs=[mech(8, 100, hp=1400, aim=0.0), mech(9, 130, hp=100, aim=0.0)]) == "hide"  # the healthy one: too risky
    assert tick(mechs=[mech(9, 130, hp=100, aim=0.0)]) == "combat" and bot.locked == {9}         # alone and nearly dead: go
    assert tick(mechs=[], events=[{"type": "kill", "target": 9}]) == "hide"                       # cleared: hide again
    assert tick(events=[], alive=2) == "combat" and bot.locked is None                             # two of us left
print("ok")
