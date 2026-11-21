from example_bot import risk

ME, C, R = [0.0, 0.0], [0.0, 0.0], 500.0
E = [dict(pos=[50.0, 0.0], w=1.0)]


def test_strafes_instead_of_charging_or_running():
    side, toward, away = risk([0, 15], ME, E, [], None, C, R), risk([15, 0], ME, E, [], None, C, R), risk([-15, 0], ME, E, [], None, C, R)
    assert side < away < toward


def test_zone_and_goal():
    assert risk([0, 10], ME, [], [], None, [0, 0], 15) > risk([0, 1], ME, [], [], None, [0, 0], 15) + 10  # outside the edge
    assert risk([10, 0], ME, [], [], [40, 0], C, R) < risk([-10, 0], ME, [], [], [40, 0], C, R)


def test_keeps_moving():
    assert risk([1, 0], ME, [], [[0, 0]], None, C, R) > risk([15, 0], ME, [], [[0, 0]], None, C, R)
