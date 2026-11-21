import math

import pytest
from conftest import box, make_map, real_config

import memory as M
import world as W

CFG = real_config()


def obs(t, pos=(0.0, 0.0, 0.0), yaw=0.0, mechs=(), events=(), sounds=(), pickups=()):
    st = dict(CFG["mech"])
    return {"t": t, "self": {"pos": list(pos), "aim": [yaw, 0.0], "stats": st},
            "mechs": list(mechs), "events": list(events), "sounds": list(sounds), "beams": [],
            "pickups": list(pickups), "cubes": [], "crates": []}


def mech(i, x, y):
    return {"id": i, "kind": "mech", "pos": [x, y, 0.0], "vel": [0, 0, 0], "aim": [0, 0], "hp": 1400, "max_hp": 1400}


@pytest.fixture
def world(tmp_path):
    return W.World(make_map(str(tmp_path), 400, [("building", 0.0, 20.0, box(40, -30, 50, 30))]), CFG["mech"])


def test_a_mech_that_drops_out_of_the_list_becomes_a_ghost_that_spreads(world):
    m = M.Memory(CFG, world)
    m.update(obs(0.0, mechs=[mech(5, 100, 100)]))
    m.update(obs(1.0, yaw=math.pi))  # looking away: it is gone from the list
    assert len(m.ghosts) == 1
    g = m.ghosts[0]
    assert (g.x, g.y, g.mid) == (100, 100, 5)
    assert g.radius(11.0, m.walk) == pytest.approx(g.radius(1.0, m.walk) + 10 * CFG["mech"]["walk_speed"])


def test_a_ghost_is_cleared_when_we_look_at_its_empty_spot(world):
    m = M.Memory(CFG, world)
    m.update(obs(0.0, mechs=[mech(5, -100, 20)]))
    m.update(obs(0.5, yaw=0.0))  # facing east, the ghost is west, behind us
    assert len(m.ghosts) == 1
    m.update(obs(1.0, yaw=math.pi))  # turn round: open ground, nobody there
    assert m.ghosts == []


def test_a_ghost_behind_a_building_is_kept(world):
    m = M.Memory(CFG, world)
    m.update(obs(0.0, mechs=[mech(5, 100, 0)]))
    m.update(obs(0.5, yaw=0.0))  # it went behind the building at x 40..50 while we look straight at it
    assert len(m.ghosts) == 1


def test_a_ghost_is_kept_while_a_mech_stands_near_it(world):
    m = M.Memory(CFG, world)
    m.update(obs(0.0, mechs=[mech(5, -100, 20)]))
    m.update(obs(0.5, yaw=0.0))
    m.update(obs(1.0, yaw=math.pi, mechs=[mech(7, -95, 25)]))
    assert len(m.ghosts) == 1


def test_hit_from_someone_unseen_puts_a_ghost_on_the_bearing(world):
    m = M.Memory(CFG, world)
    m.update(obs(0.0, events=[{"type": "damaged", "amount": 315, "weapon": "laser", "bearing": math.pi / 2, "by": 9}]))
    g = m.ghosts[0]
    assert g.mid == 9 and abs(g.x) < 1 and g.y == pytest.approx(M.SHOT_DIST["laser"])
    assert m.unseen_hit == (0.0, math.pi / 2)


def test_sound_from_a_seen_mech_adds_no_ghost(world):
    m = M.Memory(CFG, world)
    snd = {"kind": "gun", "bearing": math.atan2(30, -100), "dist": "mid", "age": 0.05}
    m.update(obs(0.0, mechs=[mech(3, -100, 30)], sounds=[snd]))
    assert all(g.src != "sound" for g in m.ghosts)
    m.update(obs(0.1, sounds=[dict(snd, bearing=-1.0)]))
    assert any(g.src == "sound" for g in m.ghosts)


def test_loot_is_forgotten_only_when_its_spot_is_in_view(world):
    m = M.Memory(CFG, world)
    m.update(obs(0.0, pickups=[{"id": 8001, "kind": "laser", "pos": [80, 0, 0]}, {"id": 8002, "kind": "laser", "pos": [-80, 0, 0]}]))
    m.update(obs(1.0, yaw=0.0))  # 8001 is behind the building, 8002 behind our back
    assert set(m.pickups) == {8001, 8002}
    m.update(obs(2.0, yaw=math.pi))  # now 8002's spot is in view and it is not there
    assert set(m.pickups) == {8001}
