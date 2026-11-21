"""Self-check for the duel simulator's geometry and zone: python3 duel/test_sim.py"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import Match, seg_cylinder  # noqa: E402

# A bullet through a mech's middle hits at its near side; one passing over its head or beside it misses.
f = seg_cylinder([0, 0, 5], [20, 0, 5], 10, 0, 0, 9, 2.5)
assert f is not None and abs(f * 20 - 7.5) < 1e-9
assert seg_cylinder([0, 0, 12], [20, 0, 12], 10, 0, 0, 9, 2.5) is None
assert seg_cylinder([0, 3, 5], [20, 3, 5], 10, 0, 0, 9, 2.5) is None
assert seg_cylinder([10, 0, 20], [10, 0, 0], 10, 0, 0, 9, 2.5) is not None  # straight down onto its head
assert abs(seg_cylinder([10, 0, 20], [10, 0, 0], 10, 0, 0, 9, 2.5) * 20 - 11) < 1e-9

# The zone holds, then moves linearly to the next circle.
m = Match.__new__(Match)
m.zone_def = {"center": [0, 0], "radius": 100, "phases": [{"hold_s": 10, "shrink_s": 20, "radius": 50, "dps": 5, "center": [20, 0]}]}
m.t = 5
assert m.zone_now() == ([0, 0], 100, 5)
m.t = 20
c, r, _ = m.zone_now()
assert math.dist(c, [10, 0]) < 1e-9 and abs(r - 75) < 1e-9
m.t = 99
c, r, _ = m.zone_now()
assert math.dist(c, [20, 0]) < 1e-9 and r == 50
print("ok")
