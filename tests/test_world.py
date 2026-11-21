import math
import random
import time

import pytest
from conftest import KIT, box, make_map, real_config

import world as W

MECH = {"height": 9.0, "step_up": 1.2}


def solve(w, a, b):
    job = W.PathJob(w, a, b)
    while not job.run(time.perf_counter() + 1.0):
        pass
    return job.path


@pytest.mark.parametrize("h,ok", [(3.0, False), (2.0, True)])
def test_wall_taller_than_a_jump_is_never_crossed(tmp_path, h, ok):
    w = W.World(make_map(str(tmp_path), 200, [("building", 0.0, h, box(-100, -2, 100, 2))]), MECH)
    path = solve(w, (0, -50), (0, 50))
    assert (path is not None) == ok
    if path:
        cells = [w.cell(*p) for p in [(0, -50)] + path]
        assert all(w.floor[b] - w.floor[a] <= W.CLIMB for a, b in zip(cells, cells[1:]))


def test_real_map_paths_never_climb_more_than_a_jump():
    w = W.World(f"{KIT}/arena", real_config()["mech"])
    rng = random.Random(3)
    found = 0
    for _ in range(25):
        a = w.center(w.snap(rng.uniform(-450, 500), rng.uniform(-300, 350), 30, comp=w.main))
        b = w.center(w.snap(rng.uniform(-450, 500), rng.uniform(-300, 350), 30, comp=w.main))
        path = solve(w, a, b)
        if path:
            found += 1
            cells = [w.cell(*a)] + [w.cell(*p) for p in path]
            assert all(w.floor[d] - w.floor[c] <= W.CLIMB for c, d in zip(cells, cells[1:]))
            assert all(not w.blocked[c] for c in cells[1:])
    assert found >= 20  # points of the main area reach each other


def test_goal_on_a_roof_ends_next_to_the_building_or_fails(tmp_path):
    w = W.World(make_map(str(tmp_path), 200, [("building", 0.0, 20.0, box(20, -20, 60, 20))]), MECH)
    assert w.floor_at(40, 0) == pytest.approx(20.0)
    path = solve(w, (-50, 0), (40, 0))  # roof centre: 20 m from the nearest street cell, beyond the snap reach
    assert path is None or w.floor_at(*path[-1]) < 1.0
    path = solve(w, (-50, 0), (24, 0))  # next to the wall: ends at its foot
    assert path and w.floor_at(*path[-1]) < 1.0
    path = solve(w, (40, 0), (-50, 0))  # from the roof down: dropping off is allowed
    assert path and w.floor_at(*path[-1]) < 1.0


def test_paths_go_round_a_building_on_the_side_inside_the_zone(tmp_path):
    w = W.World(make_map(str(tmp_path), 200, [("building", 0.0, 20.0, box(-10, -30, 10, 30))]), MECH)
    free = solve(w, (-40, 5), (40, 5))
    job = W.PathJob(w, (-40, 5), (40, 5), keep=(0, -60, 80))  # the zone lies to the south
    while not job.run(time.perf_counter() + 1.0):
        pass
    assert job.path and min(p[1] for p in job.path) < -30          # round the south end
    assert all(math.hypot(p[0], p[1] + 60) <= 80 for p in job.path)  # never out of the zone
    assert free  # and without a zone it still finds its way


def test_standing_points_are_inside_the_map(tmp_path):
    w = W.World(make_map(str(tmp_path), 200), MECH)
    for p in [(-500, 0), (0, 500), (99, 99), (-1000, -1000)]:
        x, y = w.clamp(*p)
        assert w.in_map(x, y)
        c = w.snap(*p)
        assert c >= 0 and not w.blocked[c] and w.in_map(*w.center(c), W.EDGE_PAD - 1.5)


def test_buildings_block_sight_and_high_bridges_only_at_their_deck(tmp_path):
    w = W.World(make_map(str(tmp_path), 200, [("building", 0.0, 15.0, box(-5, -5, 5, 5)),
                                               ("bridge", 12.0, 14.0, box(-60, 30, 60, 40))]), MECH)
    assert not w.clear((-50, 0, 8), (50, 0, 8))      # through the building
    assert w.clear((-50, 0, 20), (50, 0, 20))        # over it
    assert w.clear((-50, 35, 6), (50, 35, 6))        # under the bridge
    assert not w.clear((0, 0 + 20, 13), (0, 60, 13))  # through its deck
    assert w.passable(0, 35)                          # walk under it
    assert w.floor_at(0, 35) == pytest.approx(0.0)


def test_low_bridge_over_a_ravine_is_walked_on(tmp_path):
    ravine = lambda x, y: -5.0 if abs(x) < 10 else 0.0
    w = W.World(make_map(str(tmp_path), 200, [("bridge", -1.7, 0.3, box(-12, -4, 12, 4))], ravine), MECH)
    assert w.floor_at(0, 0) == pytest.approx(0.3)
    path = solve(w, (-50, 0), (50, 0))
    assert path and all(w.floor_at(*p) > -1 for p in path)  # over the deck, not down the ravine


def test_cover_counts_blocked_directions(tmp_path):
    w = W.World(make_map(str(tmp_path), 600, [("building", 0.0, 30.0, box(-60, 4, 60, 10))]), MECH)
    assert w.cover(0, -150) == 0.0
    assert 0.3 < w.cover(0, 0) < 0.7  # the wall hides the whole north half
