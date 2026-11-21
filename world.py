"""Static map model from the arena files (protocol.md §11): ground height, solids, line of sight, A* paths.

Two grids over the terrain extent:
- fine (CELL = 2 m): ground height, the vertical span of solids (for shots), and walls (for walking);
- nav (NAV x NAV fine cells = 4 m): walking cost for A*.
"""
import array
import heapq
import json
import math
import os
import sys
import time

CELL = 2.0
NAV = 2
BLOCKED = 25.0  # nav cost of a wall/steep cell: crossable (jet over, walk off a roof) but rarely worth it
PATH_WEIGHT = 2.5  # weighted A*: somewhat longer paths, far fewer expansions in Python
SQ2 = math.sqrt(2)
NEIGH = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]


class World:
    def __init__(self, map_dir: str, mech: dict):
        base = os.path.join(map_dir, "v2")
        with open(os.path.join(base, "world.json")) as f:
            w = json.load(f)
        t = w["terrain"]
        raw = array.array("h")
        with open(os.path.join(base, t.get("file", "terrain.i16")), "rb") as f:
            raw.frombytes(f.read())
        if sys.byteorder == "big":
            raw.byteswap()
        tnx, tny, tstep, scale = t["nx"], t["ny"], t["step"], t.get("scale", 0.01)
        self.x0, self.y0 = t["minX"], t["minY"]
        nx = self.nx = int((tnx - 1) * tstep // CELL)
        ny = self.ny = int((tny - 1) * tstep // CELL)
        k = CELL / tstep
        cols = [min(int((i + 0.5) * k + 0.5), tnx - 1) for i in range(nx)]
        ground = self.ground = array.array("f")
        for j in range(ny):
            row = min(int((j + 0.5) * k + 0.5), tny - 1) * tnx
            ground.extend([raw[row + c] * scale for c in cols])

        # Solids: lo/hi = vertical span that stops shots; wall = blocks walking at ground level.
        lo = self.lo = array.array("f", [1e9]) * (nx * ny)
        hi = self.hi = array.array("f", [-1e9]) * (nx * ny)
        wall = bytearray(nx * ny)
        surf = array.array("f", ground)  # what you walk on: ground, or a low bridge deck
        height, step = mech.get("height", 9), mech.get("step_up", 1.2)
        for p in w["prisms"]:
            pb, pt, bridge = p["base"], p["top"], p["kind"] == "bridge"
            edges = []
            for poly in [p["ring"]] + list(p.get("holes") or []):
                n = len(poly) // 2
                for e in range(n):
                    x1, y1 = poly[2 * e], poly[2 * e + 1]
                    x2, y2 = poly[(2 * e + 2) % (2 * n)], poly[(2 * e + 3) % (2 * n)]
                    if y1 != y2:
                        edges.append((x1, y1, x2, y2))
            ys = p["ring"][1::2]
            j0 = max(int((min(ys) - self.y0) // CELL), 0)
            j1 = min(int((max(ys) - self.y0) // CELL), ny - 1)
            for j in range(j0, j1 + 1):  # scanline fill, even-odd rule handles holes
                cy = self.y0 + (j + 0.5) * CELL
                xs = sorted(x1 + (cy - y1) * (x2 - x1) / (y2 - y1) for x1, y1, x2, y2 in edges if (y1 <= cy) != (y2 <= cy))
                for a, b in zip(xs[0::2], xs[1::2]):
                    i0 = max(int((a - self.x0) // CELL), 0)
                    i1 = min(int((b - self.x0) // CELL), nx - 1)
                    for idx in range(j * nx + i0, j * nx + i1 + 1):
                        if pb < lo[idx]:
                            lo[idx] = pb
                        if pt > hi[idx]:
                            hi[idx] = pt
                        g = ground[idx]
                        if pb < g + height:
                            if bridge:
                                surf[idx] = max(surf[idx], pt)
                            elif pt > g + step:
                                wall[idx] = 1

        # Too steep to walk up (either way, since paths are reused both ways).
        rise = math.tan(math.radians(mech.get("max_slope_deg", 50))) * CELL
        steep = bytearray(nx * ny)
        for j in range(ny - 1):
            for idx in range(j * nx, j * nx + nx - 1):
                s = surf[idx]
                if abs(surf[idx + 1] - s) > rise:
                    steep[idx] = steep[idx + 1] = 1
                if abs(surf[idx + nx] - s) > rise:
                    steep[idx] = steep[idx + nx] = 1

        mnx, mny = self.mnx, self.mny = nx // NAV, ny // NAV
        cost = self.cost = array.array("f", [1.0]) * (mnx * mny)
        for J in range(mny):
            for I in range(mnx):
                for dj in range(NAV):
                    row = (J * NAV + dj) * nx + I * NAV
                    if any(wall[row:row + NAV]) or any(steep[row:row + NAV]):
                        cost[J * mnx + I] = BLOCKED
                        break

    # --- fine grid ---
    def _idx(self, x, y):
        i, j = int((x - self.x0) // CELL), int((y - self.y0) // CELL)
        return j * self.nx + i if 0 <= i < self.nx and 0 <= j < self.ny else -1

    def ground_at(self, x, y, default=0.0):
        idx = self._idx(x, y)
        return self.ground[idx] if idx >= 0 else default

    def clear(self, a, b, step=1.5, skip=2.5):
        """True if a straight shot from a to b is not stopped by terrain or a solid (trees never stop shots).
        Ignores the first and last `skip` m, where the coarse grid would catch the shooter's or target's own wall."""
        d = math.dist(a, b)
        n = int(d / step)
        if n < 2:
            return True
        x0, y0, nx, ny, ground, lo, hi = self.x0, self.y0, self.nx, self.ny, self.ground, self.lo, self.hi
        dx, dy, dz = (b[0] - a[0]) / n, (b[1] - a[1]) / n, (b[2] - a[2]) / n
        k0 = int(skip / (d / n)) + 1
        for k in range(k0, n - k0 + 1):
            x, y, z = a[0] + dx * k, a[1] + dy * k, a[2] + dz * k
            i, j = int((x - x0) // CELL), int((y - y0) // CELL)
            if not (0 <= i < nx and 0 <= j < ny):
                return False
            idx = j * nx + i
            if z < ground[idx] - 0.5 or lo[idx] <= z <= hi[idx]:
                return False
        return True

    # --- nav grid ---
    def nav_cell(self, x, y):
        I = min(max(int((x - self.x0) // (CELL * NAV)), 0), self.mnx - 1)
        J = min(max(int((y - self.y0) // (CELL * NAV)), 0), self.mny - 1)
        return J * self.mnx + I

    def nav_center(self, c):
        s = CELL * NAV
        return [self.x0 + (c % self.mnx + 0.5) * s, self.y0 + (c // self.mnx + 0.5) * s]

    def free(self, x, y):
        I, J = int((x - self.x0) // (CELL * NAV)), int((y - self.y0) // (CELL * NAV))
        return 0 <= I < self.mnx and 0 <= J < self.mny and self.cost[J * self.mnx + I] <= 1.0  # off the map is not walkable

    def clamp(self, p, m):
        """p moved at least m inside the map edge (the edge is a wall)."""
        return [min(max(p[0], self.x0 + m), self.x0 + self.nx * CELL - m), min(max(p[1], self.y0 + m), self.y0 + self.ny * CELL - m)]

    def line_free(self, a, b):
        n = int(math.dist(a[:2], b[:2]) / CELL) + 1
        return all(self.free(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n + 1))

    def path(self, a, b, budget_s=0.01):
        """Waypoints [[x, y], ...] from a toward b. If the time budget runs out, the path to the
        explored cell closest to b (replanning later continues from there). [] = no progress."""
        mnx, mny, cost = self.mnx, self.mny, self.cost
        s, g = self.nav_cell(a[0], a[1]), self.nav_cell(b[0], b[1])
        if s == g:
            return [[b[0], b[1]]]
        gx, gy = g % mnx, g // mnx

        def h(c):
            dx, dy = abs(c % mnx - gx), abs(c // mnx - gy)
            return (max(dx, dy) + (SQ2 - 1) * min(dx, dy)) * PATH_WEIGHT

        best, bh = s, h(s)
        came, gc = {s: s}, {s: 0.0}
        heap = [(bh, 0.0, s)]
        deadline = time.perf_counter() + budget_s
        n = 0
        while heap:
            f, gcur, cur = heapq.heappop(heap)
            if cur == g:
                best = g
                break
            if gcur > gc[cur]:
                continue
            if f - gcur < bh:
                best, bh = cur, f - gcur
            n += 1
            if n & 63 == 0 and time.perf_counter() > deadline:
                break
            cx, cy = cur % mnx, cur // mnx
            for dx, dy, d in NEIGH:
                x, y = cx + dx, cy + dy
                if 0 <= x < mnx and 0 <= y < mny:
                    nb = y * mnx + x
                    c = cost[nb]
                    if dx and dy:  # no cutting corners past a wall
                        c = max(c, cost[cy * mnx + x], cost[y * mnx + cx])
                    ng = gcur + d * c
                    if ng < gc.get(nb, 1e18):
                        gc[nb] = ng
                        came[nb] = cur
                        heapq.heappush(heap, (ng + h(nb), ng, nb))
        out = []
        c = best
        while c != s:
            out.append(self.nav_center(c))
            c = came[c]
        out.reverse()
        if best == g and out:
            out[-1] = [b[0], b[1]]
        return out
