"""Static map model from the arena files (protocol.md §11): ground height, solids, line of sight, A* paths.

Three grids over the terrain extent:
- shot (FINE = 1 m): the vertical span of solids, for line-of-fire checks (with bilinear terrain);
- fine (CELL = 2 m): ground height, the vertical span of solids, and walls (for walking);
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
FINE = 1.0  # m: the shot grid
NAV = 2
BLOCKED = 25.0  # nav cost of a wall/steep cell we can jump: crossable, but rarely worth it
TALL = 1e4      # nav cost of a wall or cliff taller than a jump: go round (no climbing routine yet)
CLIMB = 2.5     # m: a jump clears 11^2 / (2 * 22) = 2.75 m
PATH_WEIGHT = 2.5  # weighted A*: somewhat longer paths, far fewer expansions in Python
NEED = 3.5      # m of clearance (cell centre to wall-cell centre) a 2.5 m-radius body needs on the 2 m grid
ROOMY = 6.0     # m: below this a cell is walkable but tight, and costs a little more (paths keep off walls)
TIGHT = 1.4     # nav cost of a tight cell
SQ2 = math.sqrt(2)
NEIGH = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]


def _clearance(blk, nx, ny, level):
    """Chamfer distance (m) from each cell centre to the nearest cell with blk >= level; off the map counts as blocked."""
    INF, a, b = 1e9, CELL, CELL * SQ2
    d = array.array("f", [0.0 if v >= level else INF for v in blk])
    for j in range(ny):  # forward pass: neighbours left, below-left, below, below-right
        row = j * nx
        for i in range(nx):
            k = row + i
            v = d[k]
            if v == 0.0:
                continue
            v = min(v, (d[k - 1] + a) if i else a, (d[k - nx] + a) if j else a,
                    (d[k - nx - 1] + b) if i and j else b, (d[k - nx + 1] + b) if j and i < nx - 1 else b)
            d[k] = v
    for j in range(ny - 1, -1, -1):  # backward pass: right, above-right, above, above-left
        row = j * nx
        for i in range(nx - 1, -1, -1):
            k = row + i
            v = d[k]
            if v == 0.0:
                continue
            up = j < ny - 1
            v = min(v, (d[k + 1] + a) if i < nx - 1 else a, (d[k + nx] + a) if up else a,
                    (d[k + nx + 1] + b) if up and i < nx - 1 else b, (d[k + nx - 1] + b) if up and i else b)
            d[k] = v
    return d


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
        self.traw, self.tnx, self.tny, self.tstep, self.tscale = raw, tnx, tny, tstep, scale
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
        surf = self.surf = array.array("f", ground)  # what you walk on: ground, or a low bridge deck
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
                                wall[idx] = max(wall[idx], 2 if pt > g + CLIMB else 1)

        # Shots: a finer solid grid (thin walls and building corners fall between the 2 m cell centers).
        fnx, fny = self.fnx, self.fny = int((tnx - 1) * tstep // FINE), int((tny - 1) * tstep // FINE)
        flo = self.flo = array.array("f", [1e9]) * (fnx * fny)
        fhi = self.fhi = array.array("f", [-1e9]) * (fnx * fny)
        for p in w["prisms"]:
            pb, pt = p["base"], p["top"]
            edges = []
            for poly in [p["ring"]] + list(p.get("holes") or []):
                n = len(poly) // 2
                for e in range(n):
                    x1, y1 = poly[2 * e], poly[2 * e + 1]
                    x2, y2 = poly[(2 * e + 2) % (2 * n)], poly[(2 * e + 3) % (2 * n)]
                    if y1 != y2:
                        edges.append((x1, y1, x2, y2))
            ys = p["ring"][1::2]
            j0 = max(int((min(ys) - self.y0) // FINE), 0)
            j1 = min(int((max(ys) - self.y0) // FINE), fny - 1)
            for j in range(j0, j1 + 1):
                cy = self.y0 + (j + 0.5) * FINE
                xs = sorted(x1 + (cy - y1) * (x2 - x1) / (y2 - y1) for x1, y1, x2, y2 in edges if (y1 <= cy) != (y2 <= cy))
                for a, b in zip(xs[0::2], xs[1::2]):
                    i0 = max(int((a - self.x0) // FINE), 0)
                    i1 = min(int((b - self.x0) // FINE), fnx - 1)
                    for idx in range(j * fnx + i0, j * fnx + i1 + 1):
                        if pb < flo[idx]:
                            flo[idx] = pb
                        if pt > fhi[idx]:
                            fhi[idx] = pt

        # Too steep to walk up (either way, since paths are reused both ways).
        rise = math.tan(math.radians(mech.get("max_slope_deg", 50))) * CELL
        steep = bytearray(nx * ny)
        for j in range(ny - 1):
            for idx in range(j * nx, j * nx + nx - 1):
                s = surf[idx]
                for nb in (idx + 1, idx + nx):
                    dz = abs(surf[nb] - s)
                    if dz > rise:  # 2: too tall to jump (the terrain also carries some roofs)
                        v = 2 if dz > CLIMB else 1
                        steep[idx], steep[nb] = max(steep[idx], v), max(steep[nb], v)
        # The same on the 1 m terrain the server samples: a short steep bank between two 2 m cell centres
        # (stadium tiers, embankments) is invisible above.
        rise1, k = rise / CELL * tstep, int(CELL / tstep)
        for j in range(tny - 1):
            row = j * tnx
            for i in range(tnx - 1):
                h0 = raw[row + i]
                for nb in (row + i + 1, row + i + tnx):
                    dz = abs(raw[nb] - h0) * scale
                    if dz > rise1:
                        for v in (row + i, nb):  # the 2 m cells holding both ends
                            ci, cj = (v % tnx) // k, (v // tnx) // k
                            c = cj * nx + ci
                            if ci < nx and cj < ny and surf[c] == ground[c]:  # not under a bridge deck
                                steep[c] = max(steep[c], 2 if dz > CLIMB else 1)

        # Clearance: metres from each fine cell's centre to the nearest wall / steep / map-edge cell. The body
        # (radius 2.5 m) fits where it is >= NEED; paths keep to cells that fit, so they no longer scrape corners.
        blk = bytes(max(a, b) for a, b in zip(wall, steep))
        self.clr = clr = _clearance(blk, nx, ny, 1)        # from anything we can't walk through
        tall = _clearance(blk, nx, ny, 2)                  # from what a jump can't clear either
        mnx, mny = self.mnx, self.mny = nx // NAV, ny // NAV
        cost = self.cost = array.array("f", [1.0]) * (mnx * mny)
        self.rep = []  # per nav cell: its fine cell with the most room (waypoints go there, not to the cell centre)
        for J in range(mny):
            for I in range(mnx):
                best, bi = -1.0, 0
                for dj in range(NAV):
                    for di in range(NAV):
                        idx = (J * NAV + dj) * nx + I * NAV + di
                        if clr[idx] > best:
                            best, bi = clr[idx], idx
                self.rep.append([self.x0 + (bi % nx + 0.5) * CELL, self.y0 + (bi // nx + 0.5) * CELL])
                if best < NEED:
                    t = max(tall[(J * NAV + dj) * nx + I * NAV + di] for dj in range(NAV) for di in range(NAV))
                    cost[J * mnx + I] = BLOCKED if t >= NEED else TALL
                elif best < ROOMY:
                    cost[J * mnx + I] = TIGHT

    # --- fine grid ---
    def _idx(self, x, y):
        i, j = int((x - self.x0) // CELL), int((y - self.y0) // CELL)
        return j * self.nx + i if 0 <= i < self.nx and 0 <= j < self.ny else -1

    def ground_at(self, x, y, default=0.0):
        idx = self._idx(x, y)
        return self.ground[idx] if idx >= 0 else default

    def ground_fine(self, x, y):
        """Terrain height, bilinear on the 1 m source grid (what the server samples)."""
        fx, fy = (x - self.x0) / self.tstep, (y - self.y0) / self.tstep
        i, j = min(max(int(fx), 0), self.tnx - 2), min(max(int(fy), 0), self.tny - 2)
        ax, ay = min(max(fx - i, 0.0), 1.0), min(max(fy - j, 0.0), 1.0)
        n, h = self.tnx, self.traw
        a, b, c, d = h[j * n + i], h[j * n + i + 1], h[(j + 1) * n + i], h[(j + 1) * n + i + 1]
        return (a + (b - a) * ax + (c - a) * ay + (a - b - c + d) * ax * ay) * self.tscale

    def clear(self, a, b, step=1.0, skip=3.0, skip_end=1.0):
        """True if a straight shot from a to b is not stopped by terrain or a solid (trees never stop shots),
        on the 1 m shot grid with bilinear terrain. Ignores the first `skip` m (a shot leaves 3 m out from
        the muzzle) and the last `skip_end` m (the target's own body)."""
        d = math.dist(a, b)
        n = int(d / step)
        if n < 2:
            return True
        x0, y0, fnx, fny, lo, hi, gf = self.x0, self.y0, self.fnx, self.fny, self.flo, self.fhi, self.ground_fine
        dx, dy, dz = (b[0] - a[0]) / n, (b[1] - a[1]) / n, (b[2] - a[2]) / n
        sl = d / n
        for k in range(int(skip / sl) + 1, n - int(skip_end / sl)):
            x, y, z = a[0] + dx * k, a[1] + dy * k, a[2] + dz * k
            i, j = int((x - x0) // FINE), int((y - y0) // FINE)
            if not (0 <= i < fnx and 0 <= j < fny):
                return False
            idx = j * fnx + i
            if lo[idx] <= z <= hi[idx] or z < gf(x, y):
                return False
        return True

    # --- nav grid ---
    def nav_cell(self, x, y):
        I = min(max(int((x - self.x0) // (CELL * NAV)), 0), self.mnx - 1)
        J = min(max(int((y - self.y0) // (CELL * NAV)), 0), self.mny - 1)
        return J * self.mnx + I

    def nav_center(self, c):
        return list(self.rep[c])

    def free(self, x, y):
        I, J = int((x - self.x0) // (CELL * NAV)), int((y - self.y0) // (CELL * NAV))
        return 0 <= I < self.mnx and 0 <= J < self.mny and self.cost[J * self.mnx + I] < BLOCKED  # off the map is not walkable

    def clamp(self, p, m):
        """p moved at least m inside the map edge (the edge is a wall)."""
        return [min(max(p[0], self.x0 + m), self.x0 + self.nx * CELL - m), min(max(p[1], self.y0 + m), self.y0 + self.ny * CELL - m)]

    def line_free(self, a, b):
        """The body fits all along the straight walk from a to b (checked every metre)."""
        n = int(math.dist(a[:2], b[:2])) + 1
        clr = self.clr
        for k in range(n + 1):
            idx = self._idx(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            if idx < 0 or clr[idx] < NEED:
                return False
        return True

    def path(self, a, b, budget_s=0.01):
        """Waypoints [[x, y], ...] from a toward b. If the time budget runs out, the path to the
        explored cell closest to b (replanning later continues from there). [] = no progress."""
        st = self.search(a, b)
        self.advance(st, budget_s)
        return self.route(st)

    def search(self, a, b):
        """A weighted A* search from a to b that advance() can run a slice at a time, over several decisions."""
        s, g = self.nav_cell(a[0], a[1]), self.nav_cell(b[0], b[1])
        hs = self._h(s, g)
        return {"s": s, "g": g, "b": [b[0], b[1]], "best": s, "bh": hs, "came": {s: s}, "gc": {s: 0.0},
                "heap": [(hs, 0.0, s)], "done": s == g}

    def _h(self, c, g):
        mnx = self.mnx
        dx, dy = abs(c % mnx - g % mnx), abs(c // mnx - g // mnx)
        return (max(dx, dy) + (SQ2 - 1) * min(dx, dy)) * PATH_WEIGHT

    def advance(self, st, budget_s):
        """Expand st for up to budget_s seconds. True once it is finished (goal reached or nothing left)."""
        if st["done"]:
            return True
        mnx, mny, cost, h = self.mnx, self.mny, self.cost, self._h
        g, came, gc, heap = st["g"], st["came"], st["gc"], st["heap"]
        best, bh = st["best"], st["bh"]
        deadline = time.perf_counter() + budget_s
        n = 0
        while heap:
            n += 1
            if n & 63 == 0 and time.perf_counter() > deadline:
                break
            f, gcur, cur = heapq.heappop(heap)
            if cur == g:
                best, st["done"] = g, True
                break
            if gcur > gc[cur]:
                continue
            if f - gcur < bh:
                best, bh = cur, f - gcur
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
                        heapq.heappush(heap, (ng + h(nb, g), ng, nb))
        else:
            st["done"] = True
        st["best"], st["bh"] = best, bh
        return st["done"]

    def route(self, st):
        """Waypoints to the goal if the search reached it, else to the explored cell closest to it."""
        s, best, came = st["s"], st["best"], st["came"]
        if s == st["g"]:
            return [list(st["b"])]
        out = []
        c = best
        while c != s:
            out.append(self.nav_center(c))
            c = came[c]
        out.reverse()
        if best == st["g"] and out:
            out[-1] = list(st["b"])
        return out
