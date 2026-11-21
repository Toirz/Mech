"""Static map from the arena files (protocol.md §11) and the walking rules of §7. Built once, at hello.

Fine grid = the terrain's own vertices (1 m). `top[k]` is the height of the solid that reaches up from the ground at
vertex k: terrain, buildings, and any overhang too low for a mech to walk under (a wall at ground level, a floor on
top). Overhangs that clear a mech's head (high bridges) are kept apart as spans: they stop sight and shots, not legs.
Trees are left out: they never stop shots, and only thin sight (shortcut: we see through every crown and trunk).

Nav grid = 2 x 2 fine vertices (2 m cells). A cell's floor is the highest of its 4 vertices. A cell is blocked when a
wall taller than CLIMB stands in it or within ~1.5 m of its center, or near the map edge. Going up more than CLIMB
from one cell to the next is impossible (a jump clears 11^2 / (2 * 22) = 2.75 m); dropping is always allowed.
"""
import array
import heapq
import json
import math
import os
import sys
import time

NAV = 2           # fine vertices per nav cell side
CLIMB = 2.5       # m: the tallest step a jump gets over
EDGE_PAD = 4.0    # m kept from the map edge (a wall)
JUMP_COST = 6.0   # extra nav cost of a step that needs a jump
DROP_COST = 3.0   # extra nav cost of dropping more than DROP_H
DROP_H = 4.0
NEAR_MULT = 1.6   # nav cost multiplier within ~3.5 m of a tall wall: paths keep off walls when they can
SQ2 = math.sqrt(2.0)


def _row_spans(polys, x0, y0, inv, nx, ny):
    """(j, i0, i1): runs of grid vertices inside the polygons (even-odd over ring and holes)."""
    edges = []
    for poly in polys:
        n = len(poly) // 2
        for k in range(n):
            ax, ay = poly[2 * k], poly[2 * k + 1]
            bx, by = poly[(2 * k + 2) % (2 * n)], poly[(2 * k + 3) % (2 * n)]
            if ay != by:
                edges.append((ax, ay, bx, by) if ay < by else (bx, by, ax, ay))
    if not edges:
        return
    edges.sort(key=lambda e: e[1])
    j0 = max(0, math.ceil((edges[0][1] - y0) * inv))
    j1 = min(ny - 1, math.floor((max(e[3] for e in edges) - y0) * inv))
    active, nxt = [], 0
    for j in range(j0, j1 + 1):
        y = y0 + j / inv
        while nxt < len(edges) and edges[nxt][1] <= y:
            active.append(edges[nxt])
            nxt += 1
        active = [e for e in active if e[3] > y]
        xs = sorted(ax + (y - ay) * (bx - ax) / (by - ay) for ax, ay, bx, by in active)
        for a, b in zip(xs[::2], xs[1::2]):
            i0, i1 = max(0, math.ceil((a - x0) * inv)), min(nx - 1, math.floor((b - x0) * inv))
            if i0 <= i1:
                yield j, i0, i1


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
        nx, ny, scale = t["nx"], t["ny"], t.get("scale", 0.01)
        self.nx, self.ny, self.x0, self.y0 = nx, ny, t["minX"], t["minY"]
        self.step = t.get("step", 1.0)
        self.inv = 1.0 / self.step
        self.terrain = array.array("f", [v * scale for v in raw])
        top = array.array("f", self.terrain)
        head = mech.get("height", 9.0)
        spans = {}
        for p in sorted(w["prisms"], key=lambda p: p["base"]):  # low solids first, so stacked ones rest on them
            pb, pt = p["base"], p["top"]
            for j, i0, i1 in _row_spans([p["ring"]] + list(p.get("holes") or []), self.x0, self.y0, self.inv, nx, ny):
                for k in range(j * nx + i0, j * nx + i1 + 1):
                    if pb - top[k] < head:  # sits on what is below, or leaves no head room: solid to the floor
                        if pt > top[k]:
                            top[k] = pt
                    else:
                        spans.setdefault(k, []).append((pb, pt))
        self.top, self.spans = top, spans
        self.has_span = bytearray(nx * ny)
        for k in spans:
            self.has_span[k] = 1
        b = w["bounds"]
        self.bounds = (b["minX"], b["maxX"], b["minY"], b["maxY"])
        self.step_up = mech.get("step_up", 1.2)
        self._build_nav()
        self._cover = {}

    # ---------- heights and sight ----------

    def _k(self, x, y):
        i, j = int((x - self.x0) * self.inv + 0.5), int((y - self.y0) * self.inv + 0.5)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return j * self.nx + i
        return -1

    def floor_at(self, x, y, default=0.0):
        """Height of the solid under (x, y): where a mech standing there has its feet."""
        k = self._k(x, y)
        return self.top[k] if k >= 0 else default

    def in_map(self, x, y, pad=EDGE_PAD):
        x0, x1, y0, y1 = self.bounds
        return x0 + pad <= x <= x1 - pad and y0 + pad <= y <= y1 - pad

    def clamp(self, x, y, pad=EDGE_PAD + 4.0):
        x0, x1, y0, y1 = self.bounds
        return min(max(x, x0 + pad), x1 - pad), min(max(y, y0 + pad), y1 - pad)

    def clear(self, a, b, skip=3.0, skip_end=0.5):
        """True if nothing solid is on the segment a -> b (3D points), leaving out `skip` m at its start (a shot
        leaves 3 m out) and `skip_end` m at its end. Sampled every metre at the nearest vertex."""
        ax, ay, az = a[0], a[1], a[2]
        dx, dy, dz = b[0] - ax, b[1] - ay, b[2] - az
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-6:
            return True
        n = int(length) + 1
        top, has_span, spans, nx, ny = self.top, self.has_span, self.spans, self.nx, self.ny
        x0, y0, inv = self.x0, self.y0, self.inv
        k0, k1 = int(skip * n / length) + 1, n - int(skip_end * n / length)
        for s in range(k0, k1):
            f = s / n
            z = az + dz * f
            i, j = int((ax + dx * f - x0) * inv + 0.5), int((ay + dy * f - y0) * inv + 0.5)
            if not (0 <= i < nx and 0 <= j < ny):
                return False
            k = j * nx + i
            if z < top[k]:
                return False
            if has_span[k]:
                for lo, hi in spans[k]:
                    if lo <= z <= hi:
                        return False
        return True

    def cover(self, x, y, dirs=16, reach=120.0):
        """Share of `dirs` directions in which a body standing at (x, y) is hidden from a mech `reach` m away
        (eye at floor + 8 there, body at floor + 4.5 here). Cached per 4 m."""
        key = (int(x // 4), int(y // 4))
        c = self._cover.get(key)
        if c is None:
            f = self.floor_at(x, y)
            me = (x, y, f + 4.5)
            hid = 0
            for d in range(dirs):
                a = 2 * math.pi * d / dirs
                ox, oy = x + reach * math.cos(a), y + reach * math.sin(a)
                if not self.clear((ox, oy, self.floor_at(ox, oy, f) + 8.0), me, skip=0.0):
                    hid += 1
            c = self._cover[key] = hid / dirs
        return c

    # ---------- nav grid ----------

    def _build_nav(self):
        nx, ny, top = self.nx, self.ny, self.top
        NX, NY = (nx - 1) // NAV, (ny - 1) // NAV
        self.NX, self.NY = NX, NY
        rmax2, rmin2, rmax4 = [], [], []
        for j in range(ny):
            r = top[j * nx:(j + 1) * nx]
            rp = [r[0]] + list(r) + [r[-1], r[-1]]  # rp[i + 1] = r[i]
            a, b, c, d = rp[0:2 * NX:2], rp[1:2 * NX + 1:2], rp[2:2 * NX + 2:2], rp[3:2 * NX + 3:2]
            rmax2.append(list(map(max, b, c)))
            rmin2.append(list(map(min, b, c)))
            rmax4.append(list(map(max, a, b, c, d)))
        floor, low, near = [], [], []
        for J in range(NY):
            j = NAV * J
            floor.extend(map(max, rmax2[j], rmax2[j + 1]))
            low.extend(map(min, rmin2[j], rmin2[j + 1]))
            near.extend(map(max, rmax4[max(j - 1, 0)], rmax4[j], rmax4[j + 1], rmax4[min(j + 2, ny - 1)]))
        x0b, x1b, y0b, y1b = self.bounds
        blocked = bytearray(NX * NY)
        for C in range(NX * NY):
            f = floor[C]
            if near[C] > f + CLIMB or f - low[C] > CLIMB:
                blocked[C] = 1
        for J in range(NY):  # the map edge is a wall
            cy = self.y0 + (NAV * J + 0.5) * self.step
            row_out = not (y0b + EDGE_PAD <= cy <= y1b - EDGE_PAD)
            for I in range(NX):
                cx = self.x0 + (NAV * I + 0.5) * self.step
                if row_out or not (x0b + EDGE_PAD <= cx <= x1b - EDGE_PAD):
                    blocked[J * NX + I] = 1
        # within ~3.5 m of a tall wall: dearer to walk (a 3 x 3 max of `near`, row then column)
        tall = [1 if near[C] > floor[C] + CLIMB else 0 for C in range(NX * NY)]
        rows = []
        for J in range(NY):
            r = tall[J * NX:(J + 1) * NX]
            rp = [0] + r + [0]
            rows.append(list(map(max, rp[0:NX], rp[1:NX + 1], rp[2:NX + 2])))
        nearwall = bytearray(NX * NY)
        for J in range(NY):
            m = map(max, rows[max(J - 1, 0)], rows[J], rows[min(J + 1, NY - 1)])
            nearwall[J * NX:(J + 1) * NX] = bytes(m)
        self.floor = array.array("f", floor)
        self.blocked, self.nearwall = blocked, nearwall
        terr = self.terrain
        self.roof = bytearray(1 if floor[C] - terr[min(NAV * (C // NX), ny - 1) * nx + min(NAV * (C % NX), nx - 1)] > CLIMB
                              and not blocked[C] else 0 for C in range(NX * NY))
        self._components()

    def _components(self):
        """Connected walkable areas over two-way edges (steps of at most CLIMB both ways)."""
        NX, NY, floor, blocked = self.NX, self.NY, self.floor, self.blocked
        comp = array.array("i", [-1]) * (NX * NY)
        sizes = []
        for s in range(NX * NY):
            if blocked[s] or comp[s] >= 0:
                continue
            label, stack, n = len(sizes), [s], 0
            comp[s] = label
            while stack:
                C = stack.pop()
                n += 1
                I, fc = C % NX, floor[C]
                for D in (C - 1 if I > 0 else -1, C + 1 if I < NX - 1 else -1, C - NX, C + NX):
                    if 0 <= D < NX * NY and comp[D] < 0 and not blocked[D] and abs(floor[D] - fc) <= CLIMB:
                        comp[D] = label
                        stack.append(D)
            sizes.append(n)
        self.comp = comp
        self.main = max(range(len(sizes)), key=sizes.__getitem__) if sizes else -1

    def cell(self, x, y):
        I = int((x - self.x0) * self.inv / NAV)
        J = int((y - self.y0) * self.inv / NAV)
        if 0 <= I < self.NX and 0 <= J < self.NY:
            return J * self.NX + I
        return -1

    def center(self, C):
        return (self.x0 + (NAV * (C % self.NX) + 0.5) * self.step, self.y0 + (NAV * (C // self.NX) + 0.5) * self.step)

    def passable(self, x, y):
        C = self.cell(x, y)
        return C >= 0 and not self.blocked[C]

    def snap(self, x, y, reach=12, comp=None):
        """Nearest open cell to (x, y) within `reach` cells (of component `comp` if given), or -1."""
        C = self.cell(x, y)
        if C < 0:
            x, y = self.clamp(x, y)
            C = self.cell(x, y)
        I0, J0 = C % self.NX, C // self.NX
        NX, NY, blocked, cp = self.NX, self.NY, self.blocked, self.comp
        for d in range(reach + 1):
            best, bd = -1, 1e9
            for J in range(J0 - d, J0 + d + 1):
                if not 0 <= J < NY:
                    continue
                step = 1 if J in (J0 - d, J0 + d) else 2 * d or 1
                for I in range(I0 - d, I0 + d + 1, step):
                    if 0 <= I < NX:
                        D = J * NX + I
                        if not blocked[D] and (comp is None or cp[D] == comp):
                            e = (I - I0) ** 2 + (J - J0) ** 2
                            if e < bd:
                                best, bd = D, e
            if best >= 0:
                return best
        return -1

    def reachable(self, a, b):
        """Can a mech at a walk to b? Same two-way component, or b in the main area (we drop off our roof to it)."""
        A, B = self.snap(a[0], a[1], 4), self.snap(b[0], b[1], 4)
        if A < 0 or B < 0:
            return False
        return self.comp[A] == self.comp[B] or self.comp[B] == self.main

    def spot_in(self, me, cx, cy, r, comp=None):
        """The open cell nearest `me` inside the circle (cx, cy, r) that a mech in area `comp` can walk to (its own
        area, or the main one it can drop into), or None when the whole circle is out of reach on foot."""
        d = math.hypot(me[0] - cx, me[1] - cy)
        base = math.atan2(me[1] - cy, me[0] - cx) if d > 1e-6 else 0.0
        first = min(d, r)
        best, bd = None, 1e18
        for rho in (first, 0.66 * r, 0.33 * r, 0.0):
            if rho > first:
                continue
            for da in (0.0, 0.35, -0.35, 0.7, -0.7, 1.1, -1.1, 1.6, -1.6, 2.3, -2.3, math.pi):
                px, py = cx + rho * math.cos(base + da), cy + rho * math.sin(base + da)
                C = self.snap(px, py, 4, comp=comp)
                if C < 0 and comp is not None and comp != self.main:
                    C = self.snap(px, py, 4, comp=self.main)
                if C < 0:
                    continue
                qx, qy = self.center(C)
                e = math.hypot(qx - me[0], qy - me[1])
                if math.hypot(qx - cx, qy - cy) <= r and e < bd:
                    best, bd = (qx, qy), e
            if best is not None:
                break
        return best

    def area(self, x, y):
        C = self.snap(x, y, 4)
        return self.comp[C] if C >= 0 else -1

    def walkable_line(self, a, b):
        """A straight walk from a to b crosses only open cells and steps a mech walks up without a jump."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = int(math.hypot(dx, dy)) + 1
        prev = self.cell(a[0], a[1])
        if prev < 0:
            return False
        floor, blocked, up = self.floor, self.blocked, self.step_up
        for s in range(1, n + 1):
            C = self.cell(a[0] + dx * s / n, a[1] + dy * s / n)
            if C != prev:
                if C < 0 or blocked[C] or floor[C] - floor[prev] > up:
                    return False
                prev = C
        return True


OUT_COST = 40.0    # nav cost per cell outside `keep` (the zone): a path leaves it only when there is no other way


class PathJob:
    """Weighted A* on the nav grid that can be run a little each decision (run() with a deadline). With
    keep = (cx, cy, r), cells outside that circle cost OUT_COST more: the path stays inside the zone if it can."""

    def __init__(self, world: World, start, goal, weight=1.5, max_expand=60000, keep=None):
        self.w, self.weight, self.left, self.keep = world, weight, max_expand, keep
        self.goal_xy = (goal[0], goal[1])
        self.path, self.done = None, False
        s = world.snap(start[0], start[1], 6)
        g = world.snap(goal[0], goal[1], 12, comp=world.comp[s]) if s >= 0 else -1
        if s >= 0 and g < 0 and world.comp[s] != world.main:  # on a roof or in the stadium: drop off its edge
            g = world.snap(goal[0], goal[1], 12, comp=world.main)
        if s < 0 or g < 0:
            self.done = True
            return
        self.s, self.g = s, g
        self.gx, self.gy = g % world.NX, g // world.NX
        self.best = {s: 0.0}
        self.parent = {s: -1}
        self.open = [(self._h(s), 0.0, s)]

    def _h(self, C):
        dx, dy = abs(C % self.w.NX - self.gx), abs(C // self.w.NX - self.gy)
        return self.weight * (max(dx, dy) + (SQ2 - 1) * min(dx, dy))

    def run(self, deadline):
        """Expand until done or perf_counter() passes `deadline`. True once finished (path set, or None if none)."""
        if self.done:
            return True
        w = self.w
        NX, NY, floor, blocked, nearwall = w.NX, w.NY, w.floor, w.blocked, w.nearwall
        best, parent, open_, g_cell, up = self.best, self.parent, self.open, self.g, w.step_up
        keep = self.keep
        if keep is not None:  # the circle in cell units: cell (I, J) has its centre at x0 + (2 I + 0.5) step
            kx = ((keep[0] - w.x0) * w.inv - 0.5) / NAV
            ky = ((keep[1] - w.y0) * w.inv - 0.5) / NAV
            kr2 = (max(keep[2], 0.0) * w.inv / NAV) ** 2
        n = 0
        while open_:
            n += 1
            if n % 64 == 0 and time.perf_counter() > deadline:
                return False
            _, gc, C = heapq.heappop(open_)
            if gc > best.get(C, 1e18):
                continue
            if C == g_cell:
                self._finish(C)
                return True
            self.left -= 1
            if self.left <= 0:
                break
            I, J, fc = C % NX, C // NX, floor[C]
            for di, dj, dist in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                                 (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)):
                I2, J2 = I + di, J + dj
                if not (0 <= I2 < NX and 0 <= J2 < NY):
                    continue
                D = J2 * NX + I2
                if blocked[D]:
                    continue
                rise = floor[D] - fc
                if rise > CLIMB:
                    continue
                if di and dj and (blocked[J * NX + I2] or blocked[J2 * NX + I]):
                    continue
                cost = dist * (NEAR_MULT if nearwall[D] else 1.0)
                if rise > up:
                    cost += JUMP_COST
                elif rise < -DROP_H:
                    cost += DROP_COST
                if keep is not None and (I2 - kx) ** 2 + (J2 - ky) ** 2 > kr2:
                    cost += OUT_COST
                g2 = gc + cost
                if g2 < best.get(D, 1e18):
                    best[D] = g2
                    parent[D] = C
                    heapq.heappush(open_, (g2 + self._h(D), g2, D))
            # Drop links: off a roof or the stands, the cells along the edge are blocked (the edge runs through
            # them, or the wall below is too close). Jump that band of 1-2 cells to open ground at least CLIMB lower.
            for di, dj, dist in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                                 (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)):
                for k in (2, 3):
                    I2, J2 = I + di * k, J + dj * k
                    I1, J1 = I + di * (k - 1), J + dj * (k - 1)
                    if not (0 <= I2 < NX and 0 <= J2 < NY):
                        break
                    B = J1 * NX + I1
                    if not blocked[B] or floor[B] > fc + up:
                        break
                    D = J2 * NX + I2
                    if blocked[D]:
                        continue
                    if floor[D] < fc - CLIMB:
                        g2 = gc + k * dist + DROP_COST
                        if keep is not None and (I2 - kx) ** 2 + (J2 - ky) ** 2 > kr2:
                            g2 += OUT_COST
                        if g2 < best.get(D, 1e18):
                            best[D] = g2
                            parent[D] = C
                            heapq.heappush(open_, (g2 + self._h(D), g2, D))
                    break
        self.done = True  # no way there
        return True

    def _finish(self, C):
        cells = []
        while C >= 0:
            cells.append(C)
            C = self.parent[C]
        cells.reverse()
        self.path = [self.w.center(c) for c in cells[1:]] or [self.w.center(cells[0])]
        self.done = True
