import math

import pytest
from conftest import box, make_map, real_config

import gunnery as G
import nav as N
import world as W
from memory import Track
from zone import Zone

CFG = real_config()


def zobs(c=(0, 0), r=100, nc=(50, 0), nr=40, s=10, e=20):
    return {"center": list(c), "radius": r, "next_center": list(nc), "next_radius": nr, "shrink_start": s, "shrink_end": e, "dps": 20}


def test_zone_moves_linearly_and_says_when_its_edge_arrives():
    z = Zone()
    z.update(zobs(), 0.0)
    assert z.circle(5) == (0, 0, 100)
    assert z.circle(15) == pytest.approx((25, 0, 70))
    assert z.circle(30) == pytest.approx((50, 0, 40))
    assert z.time_left((60, 0), 0.0) == 1e9          # inside the next circle: never caught this phase
    left = z.time_left((-80, 0), 0.0)                # caught while it shrinks
    cx, cy, r = z.circle(left)
    assert 10 < left < 20 and math.hypot(-80 - cx, -cy) == pytest.approx(r, abs=0.1)
    assert z.time_left((-200, 0), 0.0) == 0.0        # already outside


def test_zone_uses_a_moving_circle_as_reported():
    z = Zone()
    z.update(zobs(), 9.0)
    z.update(zobs(r=97), 11.0)  # the server reports the moving radius itself
    assert z.live and z.circle(11.0)[2] == 97


def me(pos=(0, 0, 0), aim=(0.0, 0.0), weapons=(), vel=(0, 0, 0), energy=100, grounded=True):
    return {"pos": list(pos), "aim": list(aim), "vel": list(vel), "energy": energy, "grounded": grounded,
            "stats": dict(CFG["mech"]), "weapons": list(weapons) + [None] * (2 - len(weapons))}


def laser(charge=0.0, cooldown=0.0):
    return {"kind": "laser", "cooldown": cooldown, "charge": charge, "stats": dict(CFG["weapons"]["laser"])}


def gun(cooldown=0.0, ammo=12):
    return {"kind": "gun", "cooldown": cooldown, "ammo": ammo, "reload": 0, "stats": dict(CFG["weapons"]["gun"])}


def track(x, y, vx=0.0, vy=0.0, n=20):
    tr = Track({"id": 4, "kind": "mech", "pos": [x - vx * n * 0.1, y - vy * n * 0.1, 0], "vel": [vx, vy, 0], "aim": [0, 0]}, 0.0)
    for k in range(1, n + 1):
        tr.update({"id": 4, "kind": "mech", "pos": [x - vx * (n - k) * 0.1, y - vy * (n - k) * 0.1, 0], "vel": [vx, vy, 0], "aim": [0, 0]}, k * 0.1)
    return tr


def test_miss_measures_how_far_a_shot_passes():
    src = (0, 0, 7)
    assert G.miss(src, [0.0, 0.0], (100, 0, 7)) == pytest.approx((0, 0), abs=1e-9)
    side, up = G.miss(src, [math.atan2(2, 100), 0.0], (100, 0, 7))
    assert side == pytest.approx(2.0, rel=0.02) and up == pytest.approx(0, abs=1e-6)
    assert G.miss(src, [math.pi, 0.0], (100, 0, 7))[0] == 1e9  # behind us


def test_slew_is_limited_by_the_turn_rate():
    assert G.slew([0, 0], [1.0, 0], 4.0, 3.0, 1) == pytest.approx([0.2, 0])
    assert G.slew([3.0, 0], [-3.0, 0], 4.0, 3.0, 1)[0] == pytest.approx(3.2)  # the short way round, through +-pi


def test_predictor_learns_a_steady_runner():
    p = G.Predictor()
    tr = track(100, 0, 0, 10, n=10)
    for k in range(11, 31):
        tr.update({"id": 4, "kind": "mech", "pos": [100, k * 1.0 - 10, 0], "vel": [0, 10, 0], "aim": [0, 0]}, k * 0.1)
        p.learn(tr, (0, 0, 0), 2.5)
    x, y, _ = p.predict(tr, 3.0, 0.5)
    assert y == pytest.approx(20 + 5.0, abs=1.0)  # a steady runner is led by its full speed


def test_gun_leads_a_runner_and_fires_only_when_the_aim_is_on():
    g = G.Gunner(CFG, None)
    tr = track(100, 0, 0, 10)
    tgt = G.Target(tr.pos, 2.5, 9.0, ("gun",), tr)
    aim, fire, _ = g.plan(me(weapons=[gun()]), 2.0, tgt)
    assert aim[0] > 0.02  # ahead of it
    _, fire_on, _ = g.plan(me(aim=aim, weapons=[gun()]), 2.0, tgt)
    _, fire_off, _ = g.plan(me(aim=[aim[0] - 1.0, aim[1]], weapons=[gun()]), 2.0, tgt)
    assert fire_on[0] and not fire_off[0]


def test_laser_charges_then_lets_go_only_when_full_and_on_target():
    g = G.Gunner(CFG, None)
    tr = track(120, 0)
    tgt = G.Target(tr.pos, 2.5, 9.0, ("laser",), tr)
    aim, fire, _ = g.plan(me(aim=(0.8, 0.0), weapons=[laser()]), 2.0, tgt)
    assert fire[0]                                   # 0.8 rad off: start charging, the torso turns meanwhile
    _, fire, _ = g.plan(me(aim=aim, weapons=[laser(charge=0.4)]), 2.0, tgt)
    assert fire[0]                                   # not full yet: hold
    _, fire, _ = g.plan(me(aim=aim, weapons=[laser(charge=0.7)]), 2.0, tgt)
    assert not fire[0]                               # full on the tick it fires, on target: let go
    _, fire, _ = g.plan(me(aim=[aim[0] + 0.6, aim[1]], weapons=[laser(charge=0.75)]), 2.0, tgt)
    assert fire[0]                                   # off target: keep holding
    _, fire, _ = g.plan(me(aim=aim, weapons=[laser(charge=0.75)]), 2.0, None)
    assert fire[0]                                   # target lost: hold the charge, don't fire into the air


def test_laser_does_not_fire_through_a_wall(tmp_path):
    w = W.World(make_map(str(tmp_path), 300, [("building", 0.0, 30.0, box(40, -20, 50, 20))]), CFG["mech"])
    g = G.Gunner(CFG, w)
    tr = track(100, 0)
    tgt = G.Target(tr.pos, 2.5, 9.0, ("laser",), tr)
    _, fire, _ = g.plan(me(weapons=[laser()]), 2.0, tgt)
    assert not fire[0]


def test_strafe_reverses_and_never_steps_somewhere_unsafe():
    legs = N.Legs(None, 1)
    m = me()
    sides = {round(legs.strafe(m, 0.0, t * 0.1, lambda p: True)[1]) for t in range(40)}
    assert sides == {-1, 1}
    for t in range(40):  # only north is safe
        d = legs.strafe(m, 0.0, 10 + t * 0.1, lambda p: p[1] > 0)
        assert d[1] > 0.99
    assert legs.strafe(m, 0.0, 20.0, lambda p: False) == (0.0, 0.0)


def test_dodge_steps_out_of_a_bullets_line():
    legs = N.Legs(None, 1)
    bullet = {"kind": "gun", "pos": [60.0, 0.5, 4.5], "vel": [-252.0, 0.0, 0.0]}
    d = legs.dodge(me(), [bullet], 22.0, lambda p: True)
    assert d is not None and abs(d[0]) < 1e-6 and d[1] < 0  # sideways, away from the side it passes on
    away = {"kind": "gun", "pos": [60.0, 30.0, 4.5], "vel": [-252.0, 0.0, 0.0]}
    assert legs.dodge(me(), [away], 22.0, lambda p: True) is None


def test_stuck_means_slow_for_a_while_then_hop_and_a_short_jet():
    legs = N.Legs(None, 1)
    m = me(vel=(0.5, 0, 0))
    assert legs.unstick(m, (1, 0), 0.0)[1:] == (False, False)
    assert legs.unstick(m, (1, 0), 0.6)[1] is True             # 0.6 s slow: hop
    air = me(vel=(0.5, 0, 5), grounded=False)
    assert legs.unstick(air, (1, 0), 0.9)[2] is True           # jet just after the hop
    assert legs.unstick(air, (1, 0), 1.3)[2] is False          # but only for 0.6 s
    fast = me(vel=(8, 0, 0))
    legs2 = N.Legs(None, 1)
    assert legs2.unstick(fast, (1, 0), 0.0)[1:] == (False, False) and legs2.unstick(fast, (1, 0), 5.0)[1:] == (False, False)
