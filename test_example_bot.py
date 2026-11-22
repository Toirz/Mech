"""Self-check for the bot's math and map model: python3 test_example_bot.py"""
import math
import os
import random

from example_bot import Bot, aim_at, edge_spot, lead, mortar_aim, shell_landing, zone_at

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
for x in range(-200, 1, 40):  # the west half is freshly visited
    for y in range(-200, 201, 40):
        b.visited[(x // 40, y // 40)] = 100.0
g = b.roam_goal([0, 0, 0], [0, 0], 250, 100.0)
assert g and math.hypot(*g) < 233 and g[0] > 0, g
assert b.roam_goal([1, 1, 0], [0, 0], 250, 101.0) == g  # keeps its waypoint until there
assert b.cover_spot([5, 6, 0], [0, 0], 250, [{"pos": [50, 0, 0]}], 0.0) == [5, 6]
# Farming: drones and crates we saw are remembered until we're close and they're gone; the trip worth most cubes per metre wins.
b.cfg = {"npc": {"cubes": 2}, "crates": {"cubes": 1}}
b.pickups, b.cubes, b.crates, b.npcs = {}, {}, {}, {}
seen = {"pickups": [], "cubes": [], "events": [], "sounds": [], "crates": [{"id": 1000, "pos": [60, 0, 0], "hp": 300}],
        "mechs": [{"id": 100, "kind": "npc", "pos": [-100, 0, 0]}]}
b.remember(seen, 10.0, [0, 0, 0])
assert b.farm_goal({"cubes": 0}, [0, 0, 0], [0, 0], 250, []) == [-100, 0, 0]  # 2 cubes 100 m off beat 1 cube 60 m off
assert b.farm_goal({"cubes": 0}, [0, 0, 0], [0, 0], 250, [{"pos": [-90, 0, 0]}]) == [60, 0, 0]  # a mech by the drone
assert b.farm_goal({"cubes": 0}, [0, 0, 0], [0, 0], 80, []) == [60, 0, 0]  # the drone is outside the circle
assert b.farm_goal({"cubes": 15}, [0, 0, 0], [0, 0], 250, []) is None  # enough cubes
b.remember(dict(seen, crates=[], mechs=[]), 11.0, [55, 0, 0])  # at the crate and it isn't listed: gone; the drone is far
assert list(b.crates) == [] and list(b.npcs) == [100]
b.remember(dict(seen, crates=[], mechs=[]), 200.0, [55, 0, 0])  # not seen for FARM_S: forgotten
assert b.npcs == {}
# Upgrades: the drum magazine comes before the other gun upgrades (and armor), plating and laser_damage before it.
gun = {"weapons": [{"kind": "gun"}, None], "upgrades": []}
assert b.pick_upgrade(["gun_rate", "armor", "gun_mag"], gun) == 2
assert b.pick_upgrade(["gun_mag", "plating", "servos"], gun) == 1
assert b.pick_upgrade(["gun_damage", "gun_mag", "sensors"], dict(gun, upgrades=["gun_mag", "gun_mag"])) == 0  # gun_mag maxed
# Predictor: a strafer's remembered moves are worked out once per sample, the same as a fresh recall, and dropped with the sample.
from predict import Predictor  # noqa: E402
pr, tt, px, s = Predictor(22.0), 0.0, 0.0, 1
for k in range(120):
    tt, s = tt + 0.1, -s if k % 7 == 0 else s
    px += s * 1.17
    pr.update(tt, 3, [px, 50, 0], [s * 11.7, 0, 0], 2.8, [0, 0, 0])
hs = pr.hist[3]
f = pr.recall(3, hs[-1][0], hs[-1], 2.8, hs[-1][0] + 1e-6)
assert f and f(0.4) == pr._recall(3, hs[-1][0], hs[-1], 2.8, hs[-1][0] + 1e-6)(0.4)
assert min(pr.recalled[3]) >= hs[0][0]


# Fights, on a made-up map: a 5 m wall along x = 50 and a wall too tall to see over along y = 10, flat ground.
class Walls:
    def clear(self, a, b):
        for k, at, top in ((0, 50.0, 5.0), (1, 10.0, 1e9)):
            if (a[k] - at) * (b[k] - at) < 0 and a[2] + (at - a[k]) / (b[k] - a[k]) * (b[2] - a[2]) < top:
                return False
        return True

    def ground_at(self, x, y):
        return 0.0


def fight_bot():
    fb = Bot.__new__(Bot)
    fb.cfg = {"cubes": {"size_per_cube": 0.05, "damage_per_cube": 0.04, "stat_cap": 10}, "match": {"tick_hz": 20},
              "projectile": {"gun_radius": 0.3}, "npc": {"radius": 1.8, "height": 3.5}, "crates": {"radius": 1.4, "height": 2.4}}
    fb.mech = {"radius": 2.5, "height": 9, "eye_height": 8, "max_pitch": 1.3}
    fb.world, fb.pred, fb.t = Walls(), Predictor(22.0), 10.0
    fb.hit_by, fb.hit_on, fb.tid, fb.vhist, fb.short_until, fb.dbg = {}, {}, None, {}, 0.0, {}
    return fb


def mech(i, p, hp=1400, aim=0.0):
    return {"id": i, "kind": "mech", "pos": p, "vel": [0, 0, 0], "aim": [aim, 0], "hp": hp, "max_hp": 1400, "cubes": 0}


GUN = {"kind": "gun", "cooldown": 0, "ammo": 12, "reload": 0,
       "stats": {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "spread_deg": 0.6, "mag": 12}}
LASER = {"kind": "laser", "cooldown": 0, "charge": 0.75,
         "stats": {"damage": 315, "charge": 0.75, "min_charge": 0.15, "min_frac": 0.35, "range": 280}}
muz = [0, 0, 7]
me = {"weapons": [GUN, dict(LASER)], "cubes": 0, "energy": 100, "aim": [0, 0], "stats": {"turn_rate": 4, "pitch_rate": 3}}
fb = fight_bot()
# Only its head and shoulders show over the 5 m wall: it is a target in the open, and a full charge on them is let go.
T = fb.targets({"mechs": [mech(5, [60, 0, 0])], "crates": [], "beams": []}, me, muz, 1.0)
assert T["los"] and not fb.los(muz, T["c"])
me["aim"] = list(aim_at(muz, fb.beam_point(muz, T)))
assert fb.weapons(me, muz, muz, T, fb.t)[1] == [False, False]  # laser let go (the gun waits: its point is 1.8 m lower)
# A 100 HP mech behind the tall wall 20 m off (its back to us) loses to the one in the open 100 m off that just shot us.
fb.hit_by[7] = fb.t - 1
obs = {"mechs": [mech(6, [0, 20, 0], hp=100, aim=math.pi / 2), mech(7, [-100, 0, 0])], "crates": [], "beams": []}
assert fb.targets(obs, me, muz, 1.0)["id"] == 7
# Hidden ones only: the best of them is still the target (the aim waits where it will show), not in the open.
T = fb.targets(dict(obs, mechs=obs["mechs"][:1]), me, muz, 1.0)
assert T["id"] == 6 and not T["los"]
# Gun only: a mech in the open 180 m off, past where the gun opens a fight, loses to a hidden one 50 m off, until it joins in.
gun_only = dict(me, weapons=[GUN, None])
obs = {"mechs": [mech(6, [0, 50, 0]), mech(7, [-180, 0, 0])], "crates": [], "beams": []}
fb.hit_by = {}
assert fb.targets(obs, gun_only, muz, 1.0)["id"] == 6
fb.hit_by[7] = fb.t - 1
assert fb.targets(obs, gun_only, muz, 1.0)["id"] == 7
# The one we are on is kept against a slightly better one; a clearly better one takes over.
fb.tid, fb.hit_by = 7, {}
obs = {"mechs": [mech(7, [-100, 0, 0]), mech(8, [-90, 5, 0])], "crates": [], "beams": []}
assert fb.targets(obs, me, muz, 1.0)["id"] == 7
obs["mechs"][1]["hp"] = 400
assert fb.targets(obs, me, muz, 1.0)["id"] == 8
# The laser lets go once its charge kills through full armor (100 HP: at the 0.15 s minimum); 1400 HP waits for full.
T = fb.targets({"mechs": [mech(9, [-60, 0, 0], hp=100)], "crates": [], "beams": []}, me, muz, 1.0)
me["aim"], me["weapons"][1] = list(aim_at(muz, fb.beam_point(muz, T))), dict(LASER, charge=0.15)
assert fb.weapons(me, muz, muz, T, fb.t)[1][1] is False
T["m"]["hp"] = 1400
assert fb.weapons(me, muz, muz, T, fb.t)[1][1] is True
assert abs(fb.laser_need(LASER, dict(T, m={"hp": 200}), me) - 0.573) < 0.001 and fb.laser_need(LASER, None, me) == 0.75
print("ok")
