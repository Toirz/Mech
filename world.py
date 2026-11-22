"""Static map model from the arena files (protocol.md §11): ground height, solids, line of sight, A* paths.

Three grids over the terrain extent:
- shot (FINE = 1 m): the vertical span of solids, for line-of-fire checks (with bilinear terrain);
- fine (CELL = 2 m): ground height, the vertical span of solids, and walls (for walking);
- nav (NAV x NAV fine cells = 4 m): walking cost for A*, the walls around each cell (corners), and the
  exposure shipped with the bot (load_safety): how many of 24 directions a mech there can be seen from.
Plus the trees, for sight (sees): trunks stop it, and so does too much crown past the all-round near range.
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
SQ2 = math.sqrt(2)
TREE_CELL = 20.0  # m: buckets of trees for sight checks
WALL_REACH = (3.5, 5.0)  # m out from a nav cell's centre where a wall counts as beside it
# 8-bit masks of walled directions (45 deg apart, bit k = k x 45 deg): 2 = walls on two perpendicular sides (a corner)
CORNER = bytes(2 if any(m >> k & 1 and m >> (k + 2) % 8 & 1 for k in range(8)) else 1 if m else 0 for m in range(256))
RIM = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (0.7, -0.7), (-0.7, 0.7), (-0.7, -0.7)]
NEIGH = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]


def _seg_cyl(p, q, x, y, r, z0, z1):
    """Fraction along p -> q where it first enters the upright cylinder at (x, y), radius r, from z0 to z1, or None."""
    dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    fx, fy = p[0] - x, p[1] - y
    a, b, cc = dx * dx + dy * dy, 2 * (fx * dx + fy * dy), fx * fx + fy * fy - r * r
    if a < 1e-12:
        if cc > 0:
            return None
        tin, tout = 0.0, 1.0
    else:
        disc = b * b - 4 * a * cc
        if disc < 0:
            return None
        sq = math.sqrt(disc)
        tin, tout = max((-b - sq) / (2 * a), 0.0), min((-b + sq) / (2 * a), 1.0)
    if abs(dz) < 1e-9:
        if not z0 <= p[2] <= z1:
            return None
    else:
        za, zb = (z0 - p[2]) / dz, (z1 - p[2]) / dz
        tin, tout = max(tin, min(za, zb)), min(tout, max(za, zb))
    return tin if tin <= tout else None


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
        self.n_prisms, self.expo = len(w["prisms"]), None
        rules, trees = w.get("rules") or {}, w.get("trees") or {}
        self.canopy_density, self.canopy_block = rules.get("canopyDensity", 0.35), rules.get("canopyBlockT", 0.25)
        self.trees = {}  # 20 m cell -> (x, y, base, top, trunk r, crown x, crown y, crown base, crown r) of trees touching it
        f, data = trees.get("fields", []), trees.get("data", [])
        for k in range(0, len(data), len(f) or 1):
            r = dict(zip(f, data[k:k + len(f)]))
            z, cx, cy = r["z"], r.get("crownX", r["x"]), r.get("crownY", r["y"])  # heights are above the tree's base
            tree = (r["x"], r["y"], z, z + r["height"], r["trunkR"], cx, cy, z + r["crownBase"], r["crownR"])
            reach = max(r["crownR"] + abs(cx - r["x"]) + abs(cy - r["y"]), r["trunkR"])
            for i in range(int((r["x"] - reach) // TREE_CELL), int((r["x"] + reach) // TREE_CELL) + 1):
                for j in range(int((r["y"] - reach) // TREE_CELL), int((r["y"] + reach) // TREE_CELL) + 1):
                    self.trees.setdefault((i, j), []).append(tree)
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
        self.body = (mech.get("radius", 2.5), height, step)
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

        mnx, mny = self.mnx, self.mny = nx // NAV, ny // NAV
        cost = self.cost = array.array("f", [1.0]) * (mnx * mny)
        for J in range(mny):
            for I in range(mnx):
                w = 0
                for dj in range(NAV):
                    row = (J * NAV + dj) * nx + I * NAV
                    w = max(w, max(wall[row:row + NAV]), max(steep[row:row + NAV]))
                if w:
                    cost[J * mnx + I] = TALL if w == 2 else BLOCKED

        # Walls beside each walkable nav cell: bit k set if 3.5-5 m out at k x 45 deg a body would hit a solid or a
        # step up (or the map edge).
        self.walls = bytearray(mnx * mny)
        dirs = [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)]
        S, x0, y0 = CELL * NAV, self.x0, self.y0
        for J in range(mny):
            for I in range(mnx):
                c = J * mnx + I
                if cost[c] > 1.0:
                    continue
                x, y = x0 + (I + 0.5) * S, y0 + (J + 0.5) * S
                z, m = ground[self._idx(x, y)], 0
                for k, (dx, dy) in enumerate(dirs):
                    for r in WALL_REACH:
                        i, j = int((x + dx * r - x0) // CELL), int((y + dy * r - y0) // CELL)
                        idx = j * nx + i
                        if not (0 <= i < nx and 0 <= j < ny) or ground[idx] > z + step or (lo[idx] < z + height and hi[idx] > z + step):
                            m |= 1 << k
                            break
                self.walls[c] = m

    def load_safety(self, path):
        """The exposure map made offline for this map (the kit's duel/exposure.py --export): per nav cell, how many
        of 24 directions a mech standing there can be seen from, 0-24 (255: nobody stands there). Stays None if the
        file is missing or was made for another map."""
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return None
        if d.get("map") != [self.x0, self.y0, self.nx, self.ny, self.n_prisms] or d.get("cell") != CELL * NAV \
                or len(d.get("data", "")) != self.mnx * self.mny:
            return None
        self.expo = bytes(255 if ch == "." else ord(ch) - 65 for ch in d["data"])
        return self.expo

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

    def sees(self, a, b, near=45.0):
        """Can an eye at a see the point b (the referee's sight): no solid or terrain between, no tree trunk, and
        past `near` m (the all-round proximity range) not more crown in the way than the canopy lets through."""
        if not self.clear(a, b, skip=1.5, skip_end=1.5):
            return False
        trunk, leaves = self.foliage(a, b)
        return not trunk and (math.dist(a[:2], b[:2]) <= near or math.exp(-self.canopy_density * leaves) >= self.canopy_block)

    def foliage(self, a, b):
        """(a trunk is in the way, m of crown the line a -> b goes through); crowns are upright ellipsoids."""
        cells, seen, leaves, d = set(), set(), 0.0, math.dist(a, b)
        n = max(int(math.dist(a[:2], b[:2]) / 10), 1)
        for s in range(n + 1):
            x, y = a[0] + (b[0] - a[0]) * s / n, a[1] + (b[1] - a[1]) * s / n
            i, j = int(x // TREE_CELL), int(y // TREE_CELL)
            cells.update((i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1))
        vx, vy, vz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        for c in cells:
            for tr in self.trees.get(c, ()):
                if tr in seen:
                    continue
                seen.add(tr)
                x, y, z0, z1, rt, cx, cy, cb, cr = tr
                f = _seg_cyl(a, b, x, y, rt, z0, min(z1, cb))
                if f is not None and 1.0 < f * d < d - 1.0:
                    return True, leaves
                hz = max((z1 - cb) / 2, 0.1)
                k = cr / hz  # squash the ellipsoid into a sphere of radius cr
                px, py, pz = a[0] - cx, a[1] - cy, (a[2] - (cb + z1) / 2) * k
                wz = vz * k
                qa = vx * vx + vy * vy + wz * wz
                qb = 2 * (px * vx + py * vy + pz * wz)
                qc = px * px + py * py + pz * pz - cr * cr
                disc = qb * qb - 4 * qa * qc
                if qa > 1e-12 and disc > 0:
                    r = math.sqrt(disc)
                    leaves += max(min((-qb + r) / (2 * qa), 1.0) - max((-qb - r) / (2 * qa), 0.0), 0.0) * d
        return False, leaves

    # --- nav grid ---
    def nav_cell(self, x, y):
        I = min(max(int((x - self.x0) // (CELL * NAV)), 0), self.mnx - 1)
        J = min(max(int((y - self.y0) // (CELL * NAV)), 0), self.mny - 1)
        return J * self.mnx + I

    def nav_center(self, c):
        s = CELL * NAV
        return [self.x0 + (c % self.mnx + 0.5) * s, self.y0 + (c // self.mnx + 0.5) * s]

    def body_free(self, x, y, z):
        """Could a body with feet at z stand here: no ground more than a step up, no solid in it, not off the map
        (its centre and 8 points on its rim, on the 2 m grid)."""
        r, height, step = self.body
        lo, hi, ground, top = self.lo, self.hi, self.ground, z + step
        for dx, dy in RIM:
            idx = self._idx(x + dx * r, y + dy * r)
            if idx < 0 or ground[idx] > top or (lo[idx] < z + height and hi[idx] > top):
                return False
        return True

    def free(self, x, y):
        I, J = int((x - self.x0) // (CELL * NAV)), int((y - self.y0) // (CELL * NAV))
        return 0 <= I < self.mnx and 0 <= J < self.mny and self.cost[J * self.mnx + I] <= 1.0  # off the map is not walkable

    def clamp(self, p, m):
        """p moved at least m inside the map edge (the edge is a wall)."""
        return [min(max(p[0], self.x0 + m), self.x0 + self.nx * CELL - m), min(max(p[1], self.y0 + m), self.y0 + self.ny * CELL - m)]

    def line_free(self, a, b):
        n = int(math.dist(a[:2], b[:2]) / CELL) + 1
        return all(self.free(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n + 1))

    def path(self, a, b, budget_s=0.01, extra=None):
        """Waypoints [[x, y], ...] from a toward b. If the time budget runs out, the path to the
        explored cell closest to b (replanning later continues from there). [] = no progress.
        `extra(cell)`, if given, is added to the cost of every step into that nav cell."""
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
                    ng = gcur + d * (c + extra(nb) if extra else c)
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
