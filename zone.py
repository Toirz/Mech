"""The zone circle at any time (protocol.md §9), from `obs.zone`.

`center`/`radius` is the circle the current shrink starts from and `next_*` where it ends; between `shrink_start`
and `shrink_end` the circle moves linearly. If the server turns out to report the moving circle itself (its radius
changes during a shrink), we use what it reports instead of interpolating again.
"""
import math


class Zone:
    def __init__(self):
        self.z, self.live, self._seen = None, False, None

    def update(self, z, t):
        self.z = z
        key = (z["shrink_start"], z["shrink_end"])
        if self._seen is None or self._seen[0] != key:
            self._seen, self.live = (key, z["radius"]), False
        elif t > z["shrink_start"] and abs(z["radius"] - self._seen[1]) > 0.01:
            self.live = True

    def circle(self, t):
        """(cx, cy, r) at match time t (assumes t is within the current phase)."""
        z = self.z
        (cx, cy), r = z["center"], z["radius"]
        if self.live:
            return cx, cy, r
        s, e = z["shrink_start"], z["shrink_end"]
        f = 0.0 if t <= s else 1.0 if t >= e or e <= s else (t - s) / (e - s)
        nx, ny = z["next_center"]
        return cx + (nx - cx) * f, cy + (ny - cy) * f, r + (z["next_radius"] - r) * f

    def inside(self, p, t, margin=0.0):
        cx, cy, r = self.circle(t)
        return math.hypot(p[0] - cx, p[1] - cy) <= r - margin

    def next_circle(self):
        z = self.z
        return z["next_center"][0], z["next_center"][1], z["next_radius"]

    def time_left(self, p, now, margin=0.0):
        """Seconds until the edge (less `margin`) reaches p during this phase; 0 if it has, 1e9 if it never will."""
        if not self.inside(p, now, margin):
            return 0.0
        nx, ny, nr = self.next_circle()
        if math.hypot(p[0] - nx, p[1] - ny) <= nr - margin:
            return 1e9
        lo, hi = max(now, self.z["shrink_start"]), max(now, self.z["shrink_end"])
        if self.inside(p, hi, margin):
            return 1e9
        for _ in range(30):  # inside at lo, outside at hi: bisect
            mid = (lo + hi) / 2
            if self.inside(p, mid, margin):
                lo = mid
            else:
                hi = mid
        return lo - now
