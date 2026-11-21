"""A 1v1 mechfight/1 match simulator, written from the kit's protocol.md.

It is NOT the real server: it models what matters for a duel (movement with walls and ledges, aim
slewing, gun / laser / lightning / mortar exactly as protocol.md §7 describes them, rear armor,
vision with buildings and canopy, sound, a shrinking zone, HP and energy regen) and leaves out
crates, NPCs, cubes, upgrades and supply drops. Bots are real subprocesses speaking the protocol in
lockstep: obs at tick N, the action holds for ticks N and N+1.
"""
import array
import json
import math
import os
import queue
import random
import subprocess
import threading
import time

DT = 0.05
RANGE_BUCKETS = (50, 100, 150, 200, 1e9)


def bucket(d):
    for b in RANGE_BUCKETS:
        if d < b:
            return b
    return RANGE_BUCKETS[-1]


def ang_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def direction(yaw, pitch):
    c = math.cos(pitch)
    return [c * math.cos(yaw), c * math.sin(yaw), math.sin(pitch)]


# ------------------------------------------------------------------ the map
class Arena:
    """Terrain (bilinear, 1 m), solids rasterized on a CELL grid (vertical span per cell), trees for sight."""

    def __init__(self, arena_dir, cell=1.0, flat=False):
        base = os.path.join(arena_dir, "v2")
        with open(os.path.join(base, "world.json")) as f:
            w = json.load(f)
        t = w["terrain"]
        self.tx0, self.ty0, self.tstep = t["minX"], t["minY"], t["step"]
        self.tnx, self.tny = t["nx"], t["ny"]
        if flat:
            self.h = array.array("f", [0.0]) * (self.tnx * self.tny)
        else:
            raw = array.array("h")
            with open(os.path.join(base, t.get("file", "terrain.i16")), "rb") as f:
                raw.frombytes(f.read())
            sc = t.get("scale", 0.01)
            self.h = array.array("f", (v * sc for v in raw))
        self.minx, self.miny = self.tx0, self.ty0
        self.maxx = self.tx0 + (self.tnx - 1) * self.tstep
        self.maxy = self.ty0 + (self.tny - 1) * self.tstep
        self.cell = cell
        nx = self.nx = int((self.maxx - self.minx) / cell)
        ny = self.ny = int((self.maxy - self.miny) / cell)
        self.lo = array.array("f", [1e9]) * (nx * ny)
        self.hi = array.array("f", [-1e9]) * (nx * ny)
        prisms = [] if flat else w["prisms"]
        for p in prisms:
            self._fill(p)
        r = w.get("rules", {})
        self.density, self.block_t = r.get("canopyDensity", 0.35), r.get("canopyBlockT", 0.25)
        self.trees, self.tb = [], {}
        if not flat:
            fs, d = w["trees"]["fields"], w["trees"]["data"]
            n = len(fs)
            for k in range(len(d) // n):
                tr = dict(zip(fs, d[k * n:(k + 1) * n]))
                cz0, cz1 = tr["z"] + tr["crownBase"], tr["z"] + tr["height"]
                tree = (tr["x"], tr["y"], tr["z"], cz0, tr["trunkR"], tr.get("crownX", tr["x"]), tr.get("crownY", tr["y"]),
                        (cz0 + cz1) / 2, max(tr["crownR"], 0.1), max((cz1 - cz0) / 2, 0.1))
                self.trees.append(tree)
                key = (int(tree[5] // 20), int(tree[6] // 20))
                self.tb.setdefault(key, []).append(tree)

    def _fill(self, p):
        c, x0, y0, nx, ny = self.cell, self.minx, self.miny, self.nx, self.ny
        edges = []
        for poly in [p["ring"]] + list(p.get("holes") or []):
            n = len(poly) // 2
            for e in range(n):
                x1, y1 = poly[2 * e], poly[2 * e + 1]
                x2, y2 = poly[(2 * e + 2) % (2 * n)], poly[(2 * e + 3) % (2 * n)]
                if y1 != y2:
                    edges.append((x1, y1, x2, y2))
        ys = p["ring"][1::2]
        j0, j1 = max(int((min(ys) - y0) // c), 0), min(int((max(ys) - y0) // c), ny - 1)
        lo, hi, pb, pt = self.lo, self.hi, p["base"], p["top"]
        for j in range(j0, j1 + 1):
            cy = y0 + (j + 0.5) * c
            xs = sorted(x1 + (cy - y1) * (x2 - x1) / (y2 - y1) for x1, y1, x2, y2 in edges if (y1 <= cy) != (y2 <= cy))
            for a, b in zip(xs[0::2], xs[1::2]):
                i0, i1 = max(int((a - x0) // c), 0), min(int((b - x0) // c), nx - 1)
                for idx in range(j * nx + i0, j * nx + i1 + 1):
                    if pb < lo[idx]:
                        lo[idx] = pb
                    if pt > hi[idx]:
                        hi[idx] = pt

    def ground(self, x, y):
        fx, fy = (x - self.tx0) / self.tstep, (y - self.ty0) / self.tstep
        i, j = min(max(int(fx), 0), self.tnx - 2), min(max(int(fy), 0), self.tny - 2)
        ax, ay = min(max(fx - i, 0.0), 1.0), min(max(fy - j, 0.0), 1.0)
        n, h = self.tnx, self.h
        a, b, c, d = h[j * n + i], h[j * n + i + 1], h[(j + 1) * n + i], h[(j + 1) * n + i + 1]
        return a + (b - a) * ax + (c - a) * ay + (a - b - c + d) * ax * ay

    def idx(self, x, y):
        i, j = int((x - self.minx) // self.cell), int((y - self.miny) // self.cell)
        return j * self.nx + i if 0 <= i < self.nx and 0 <= j < self.ny else -1

    def inside(self, x, y):
        return self.minx <= x <= self.maxx and self.miny <= y <= self.maxy

    def solid(self, x, y, z):
        k = self.idx(x, y)
        if k < 0:
            return True
        return z < self.ground(x, y) or self.lo[k] <= z <= self.hi[k]

    def seg_block(self, a, b, step=0.5):
        """Fraction along a->b of the first terrain/solid sample, or None if the segment is clear."""
        d = math.dist(a, b)
        n = max(int(d / step), 1)
        for k in range(1, n + 1):
            f = k / n
            if self.solid(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f):
                return (k - 0.5) / n
        return None

    def trees_near(self, a, b, pad=6.0):
        x0, x1 = min(a[0], b[0]) - pad, max(a[0], b[0]) + pad
        y0, y1 = min(a[1], b[1]) - pad, max(a[1], b[1]) + pad
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy or 1e-9
        out = []
        for bx in range(int(x0 // 20), int(x1 // 20) + 1):
            for by in range(int(y0 // 20), int(y1 // 20) + 1):
                for tr in self.tb.get((bx, by), ()):
                    f = min(max(((tr[5] - a[0]) * dx + (tr[6] - a[1]) * dy) / L2, 0.0), 1.0)
                    if math.hypot(a[0] + dx * f - tr[5], a[1] + dy * f - tr[6]) < tr[8] + pad - 4:
                        out.append(tr)
        return out

    def trees_block(self, a, b, canopy=True):
        """Trunks stop sight; crowns fade it (exp(-density * path length) under block_t blocks)."""
        if not self.trees:
            return False
        L = 0.0
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        seg = math.sqrt(dx * dx + dy * dy + dz * dz) or 1e-9
        h2 = dx * dx + dy * dy or 1e-9
        for (tx, ty, tz, cz0, trR, cx, cy, cz, cr, chz) in self.trees_near(a, b):
            f = min(max(((tx - a[0]) * dx + (ty - a[1]) * dy) / h2, 0.0), 1.0)  # trunk: vertical cylinder tz..cz0
            if math.hypot(a[0] + dx * f - tx, a[1] + dy * f - ty) < trR and tz <= a[2] + dz * f <= cz0:
                return True
            if canopy:  # crown: ellipsoid, chord length
                px, py, pz = (a[0] - cx) / cr, (a[1] - cy) / cr, (a[2] - cz) / chz
                qx, qy, qz = dx / cr, dy / cr, dz / chz
                A = qx * qx + qy * qy + qz * qz
                B = 2 * (px * qx + py * qy + pz * qz)
                C = px * px + py * py + pz * pz - 1
                disc = B * B - 4 * A * C
                if disc > 0:
                    s = math.sqrt(disc)
                    t0, t1 = max((-B - s) / (2 * A), 0.0), min((-B + s) / (2 * A), 1.0)
                    if t1 > t0:
                        L += (t1 - t0) * seg
                        if math.exp(-self.density * L) < self.block_t:
                            return True
        return False


def seg_cylinder(p, q, cx, cy, z0, z1, R):
    """Fraction along p->q where it first enters the vertical cylinder, or None."""
    dx, dy, dz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    fx, fy = p[0] - cx, p[1] - cy
    a = dx * dx + dy * dy
    c = fx * fx + fy * fy - R * R
    if a < 1e-12:
        if c > 0:
            return None
        lo, hi = 0.0, 1.0
    else:
        b = 2 * (fx * dx + fy * dy)
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        s = math.sqrt(disc)
        lo, hi = max((-b - s) / (2 * a), 0.0), min((-b + s) / (2 * a), 1.0)
        if lo > hi:
            return None
    za = p[2] + dz * lo
    if z0 <= za <= z1:
        return lo
    if abs(dz) < 1e-12:
        return None
    for zc in (z1, z0):  # enters through the top or bottom cap
        t = (zc - p[2]) / dz
        if lo <= t <= hi:
            return t
    return None


# ------------------------------------------------------------------ bots
class BotProc:
    def __init__(self, cmd, cwd, log_path=None):
        self.err = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, bufsize=0)
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in self.proc.stdout:
            self.q.put(raw)
        self.q.put(None)

    def send(self, msg):
        try:
            self.proc.stdin.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def wait_for(self, want, deadline):
        """The bot's line carrying tick `want` (or a ready when want is None), else None at the deadline."""
        while True:
            try:
                raw = self.q.get(timeout=max(0.0, deadline - time.perf_counter()))
            except queue.Empty:
                return None
            if raw is None:
                return None
            try:
                m = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(m, dict):
                continue
            if want is None and m.get("type") == "ready":
                return m
            if want is not None and m.get("tick") == want:
                return m

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self.err is not subprocess.DEVNULL:
            self.err.close()


# ------------------------------------------------------------------ match
class Mech:
    def __init__(self, mid, pos, yaw, cfg, loadout):
        mc = cfg["mech"]
        self.id, self.cfg, self.mc = mid, cfg, mc
        self.pos, self.vel = list(pos), [0.0, 0.0, 0.0]
        self.aim, self.cmd_aim = [yaw, 0.0], [yaw, 0.0]
        self.hp = self.max_hp = mc["hp"]
        self.energy = self.max_energy = mc["energy"]
        self.grounded, self.alive, self.died = True, True, None
        self.weapons = [self.weapon(k) for k in loadout]
        self.move, self.boost, self.fire, self.jump = [0.0, 0.0], False, [False, False], False
        self.last_spend, self.last_fight = -1e9, -1e9
        self.short_until, self.step_t, self.jet_t = 0.0, 0.0, 0.0
        self.events, self.sounds, self.beams, self.seen = [], [], [], {}
        self.last_ok, self.last_answer_t = {}, 0.0

    def weapon(self, kind):
        if kind is None:
            return None
        st = dict(self.cfg["weapons"][kind])
        w = {"kind": kind, "cooldown": 0.0, "equip": 0.0, "stats": st}
        if kind == "gun":
            w.update(ammo=int(st["mag"]), reload=0.0)
        if kind == "laser":
            w.update(charge=0.0)
        return w

    def holds(self, kind):
        return any(w and w["kind"] == kind for w in self.weapons)

    def charging(self):
        return any(w and w["kind"] == "laser" and w["charge"] > 0 for w in self.weapons)


class Match:
    def __init__(self, arena, map_dir, hello_cfg, bots, spawns, loadouts, zone, seed, max_t=180.0, timeout=0.2, log_dir=None):
        """bots: [(cmd, cwd)] for mech ids 0 and 1; spawns: [[x, y, z, yaw]]; loadouts: [[kind, kind|None]]."""
        self.A, self.map_dir, self.cfg = arena, map_dir, json.loads(json.dumps(hello_cfg))
        self.rng = random.Random(seed)
        self.zone_def, self.max_t, self.timeout = zone, max_t, timeout
        self.t, self.tick = 0.0, 0
        self.mechs = [Mech(i, s[:3], s[3], self.cfg, loadouts[i]) for i, s in enumerate(spawns)]
        self.projectiles, self.pickups = [], []
        self.next_id = {"proj": 1, "pick": 8000}
        self.stats = [{"shots": {}, "hits": {}, "dmg": {}, "late": 0} for _ in self.mechs]
        self.procs = [BotProc(cmd, cwd, os.path.join(log_dir, f"bot{i}.log") if log_dir else None) for i, (cmd, cwd) in enumerate(bots)]
        self.names = ["?"] * len(self.mechs)

    # ----- zone -----
    def zone(self):
        """Current circle + the next one + shrink window, like obs.zone."""
        c, r = list(self.zone_def["center"]), self.zone_def["radius"]
        t0 = 0.0
        for ph in self.zone_def["phases"]:
            s0, s1 = t0 + ph["hold_s"], t0 + ph["hold_s"] + ph["shrink_s"]
            if self.t < s1:
                return {"center": c, "radius": r, "next_center": ph["center"], "next_radius": ph["radius"],
                        "shrink_start": s0, "shrink_end": s1, "dps": ph["dps"]}
            c, r, t0 = list(ph["center"]), ph["radius"], s1
        last = self.zone_def["phases"][-1]
        return {"center": c, "radius": r, "next_center": c, "next_radius": r, "shrink_start": t0, "shrink_end": t0, "dps": last["dps"]}

    def zone_now(self):
        z = self.zone()
        span = z["shrink_end"] - z["shrink_start"]
        f = min(max((self.t - z["shrink_start"]) / span, 0.0), 1.0) if span > 0 else 0.0
        c = [z["center"][i] + (z["next_center"][i] - z["center"][i]) * f for i in (0, 1)]
        return c, z["radius"] + (z["next_radius"] - z["radius"]) * f, z["dps"]

    # ----- sound / damage -----
    def emit(self, kind, pos, src):
        rng_ = self.cfg["sound"]["range"].get(kind, 0)
        for m in self.mechs:
            if m is src or not m.alive:
                continue
            d = math.dist(m.pos, pos)
            if d <= rng_:
                sd = self.cfg["sound"]
                b = math.atan2(pos[1] - m.pos[1], pos[0] - m.pos[0]) + self.rng.gauss(0, math.radians(sd["bearing_noise_deg"]) * m.mc["hearing"])
                m.sounds.append((kind, (b + math.pi) % (2 * math.pi) - math.pi,
                                 "near" if d < sd["near"] else "mid" if d < sd["mid"] else "far", self.t))

    def damage(self, tgt, amount, weapon, src_pos, by, effect=None, record=True):
        if not tgt.alive or amount <= 0:
            return 0.0
        yaw_src = math.atan2(src_pos[1] - tgt.pos[1], src_pos[0] - tgt.pos[0])
        rear = abs(ang_diff(yaw_src, tgt.aim[0])) > math.radians(tgt.mc["front_arc_deg"]) / 2
        amt = amount * (tgt.mc["rear_mult"] if rear else 1.0) * (1 - tgt.mc["armor"])
        tgt.hp -= amt
        tgt.last_fight = self.t
        ev = {"type": "damaged", "amount": round(amt, 1), "weapon": weapon, "bearing": round(yaw_src, 4), "by": by.id}
        if rear:
            ev["rear"] = True
        tgt.events.append(ev)
        if effect:
            tgt.events.append({"type": "effect", "effect": effect, "by": by.id})
        tk = self.stats[tgt.id].setdefault("taken", {})
        tk[weapon] = tk.get(weapon, 0.0) + amt
        if by is not tgt:
            by.last_fight = self.t
            hit = {"type": "hit", "target": tgt.id, "amount": round(amt, 1), "weapon": weapon}
            if effect:
                hit["effect"] = effect
            if rear:
                hit["rear"] = True
            by.events.append(hit)
            if record:
                s = self.stats[by.id]
                s["dmg"][weapon] = s["dmg"].get(weapon, 0.0) + amt
        if tgt.hp <= 0:
            tgt.hp, tgt.alive, tgt.died = 0.0, False, self.tick
            if by is not tgt:
                by.events.append({"type": "kill", "target": tgt.id})
            self.emit("death", tgt.pos, tgt)
        return amt

    def nearest_foe(self, m):
        foes = [x for x in self.mechs if x is not m and x.alive] or [x for x in self.mechs if x is not m]
        return min(foes, key=lambda x: math.dist(m.pos, x.pos))

    def count_shot(self, m, kind, hit):
        foe = self.nearest_foe(m)
        b = bucket(math.dist(m.pos, foe.pos))
        s = self.stats[m.id]
        s["shots"].setdefault(kind, {}).setdefault(b, 0)
        s["hits"].setdefault(kind, {}).setdefault(b, 0)
        s["shots"][kind][b] += 1
        if hit:
            s["hits"][kind][b] += 1
        return b

    # ----- shots -----
    def shot_origin(self, m, height, dirv):
        mz = [m.pos[0], m.pos[1], m.pos[2] + height]
        out = [mz[i] + dirv[i] * (m.mc["radius"] + 0.5) for i in range(3)]
        f = self.A.seg_block(mz, out, 0.25)
        if f is not None:
            out = [mz[i] + (out[i] - mz[i]) * f for i in range(3)]
        return out

    def hitscan(self, m, dirv, rng_, origin_h):
        """(point, mech hit or None) of a beam along dirv."""
        o = self.shot_origin(m, origin_h, dirv)
        end = [o[i] + dirv[i] * rng_ for i in range(3)]
        best, hitm = 1.0, None
        for x in self.mechs:
            if x is m or not x.alive:
                continue
            f = seg_cylinder(o, end, x.pos[0], x.pos[1], x.pos[2], x.pos[2] + x.mc["height"], x.mc["radius"])
            if f is not None and f < best:
                best, hitm = f, x
        wall = self.A.seg_block(o, [o[i] + (end[i] - o[i]) * best for i in range(3)], 0.5)
        if wall is not None:
            best, hitm = best * wall, None
        return o, [o[i] + (end[i] - o[i]) * best for i in range(3)], hitm

    def fire_weapons(self, m):
        mh = m.mc["muzzle_height"]
        for i, w in enumerate(m.weapons):
            if not w:
                continue
            st, held = w["stats"], m.fire[i]
            if w["kind"] != "laser" or w["charge"] == 0:  # held down, a weapon fires at exactly its rate (the overshoot carries)
                w["cooldown"] = max(w["cooldown"] - DT, -DT)
            w["equip"] = max(w["equip"] - DT, 0.0)
            k = w["kind"]
            if k == "gun":
                if w["reload"] > 0:
                    w["reload"] -= DT
                    if w["reload"] <= 0:
                        w["reload"], w["ammo"] = 0.0, int(st["mag"])
                    continue
                if w["ammo"] <= 0:
                    w["reload"] = st["reload"]
                    self.emit("reload", m.pos, m)
                    continue
                if held and w["cooldown"] <= 0 and w["equip"] <= 0:
                    sp = math.radians(st["spread_deg"])
                    d = direction(m.aim[0] + self.rng.gauss(0, sp), m.aim[1] + self.rng.gauss(0, sp))
                    o = self.shot_origin(m, mh, d)
                    foe = self.nearest_foe(m)
                    self.projectiles.append({"id": self.next_id["proj"], "kind": "gun", "by": m, "pos": o,
                                             "vel": [d[j] * st["speed"] for j in range(3)], "left": st["range"], "dmg": st["damage"],
                                             "bucket": bucket(math.dist(m.pos, foe.pos)), "hit": False})
                    self.next_id["proj"] += 1
                    w["ammo"] -= 1
                    w["cooldown"] += st["cooldown"]
                    self.emit("gun", m.pos, m)
            elif k == "mortar":
                if held and w["cooldown"] <= 0 and w["equip"] <= 0:
                    d = direction(*m.aim)
                    o = self.shot_origin(m, m.mc["height"] + 0.5, d)
                    foe = self.nearest_foe(m)
                    self.projectiles.append({"id": self.next_id["proj"], "kind": "mortar", "by": m, "pos": o,
                                             "vel": [d[j] * st["speed"] for j in range(3)], "ttl": st["ttl"], "st": st,
                                             "bucket": bucket(math.dist(m.pos, foe.pos)), "hit": False})
                    self.next_id["proj"] += 1
                    w["cooldown"] += st["cooldown"]
                    self.emit("mortar", m.pos, m)
            elif k == "lightning":
                if held and w["cooldown"] <= 0 and w["equip"] <= 0 and m.energy >= st["energy"]:
                    m.energy -= st["energy"]
                    m.last_spend = self.t
                    w["cooldown"] += st["cooldown"]
                    mz = [m.pos[0], m.pos[1], m.pos[2] + mh]
                    best = None
                    for x in self.mechs:
                        if x is m or not x.alive:
                            continue
                        c = [x.pos[0], x.pos[1], x.pos[2] + x.mc["height"] / 2]
                        d = math.dist(mz, c)
                        yaw = math.atan2(c[1] - mz[1], c[0] - mz[0])
                        if (d <= st["range"] and abs(ang_diff(yaw, m.aim[0])) <= math.radians(st["cone_deg"]) / 2
                                and self.A.seg_block(mz, c, 0.5) is None and (best is None or d < best[0])):
                            best = (d, x, c)
                    if best:
                        x = best[1]
                        eff = None
                        if x.holds("laser"):
                            eff = "short"
                            x.short_until = self.t + st["short_s"]
                            for lw in x.weapons:
                                if lw and lw["kind"] == "laser":
                                    lw["charge"] = 0.0
                        self.damage(x, st["damage"], "lightning", mz, m, eff)
                        path = [mz, best[2]]
                        hit_ids, cur, dmg = {x.id}, best[2], st["damage"]
                        for _ in range(int(st.get("chain", 0))):  # jumps to the nearest other target in chain_range
                            dmg *= st.get("chain_falloff", 0.7)
                            nxt = min(((math.dist(cur, [y.pos[0], y.pos[1], y.pos[2] + y.mc["height"] / 2]), y) for y in self.mechs
                                       if y.alive and y is not m and y.id not in hit_ids), default=None, key=lambda e: e[0])
                            if nxt is None or nxt[0] > st.get("chain_range", 14):
                                break
                            y = nxt[1]
                            yc = [y.pos[0], y.pos[1], y.pos[2] + y.mc["height"] / 2]
                            if self.A.seg_block(cur, yc, 0.5) is not None:
                                break
                            self.damage(y, dmg, "lightning", cur, m)
                            hit_ids.add(y.id)
                            path.append(yc)
                            cur = yc
                    else:
                        d = direction(m.aim[0], m.aim[1])
                        o, end, _ = self.hitscan(m, d, st["range"], mh)
                        path = [mz, end]
                    self.count_shot(m, "lightning", best is not None)
                    for x in self.mechs:
                        if x is not m:
                            x.beams.append({"by": m.id, "kind": "lightning", "path": [[round(v, 2) for v in p] for p in path], "t": self.t})
                    self.emit("lightning", m.pos, m)
            elif k == "laser":
                full = st["charge"]
                if w["charge"] == 0:
                    if held and w["cooldown"] <= 0 and w["equip"] <= 0 and self.t >= m.short_until:
                        w["charge"] = DT
                        self.emit("laser_charge", m.pos, m)
                    continue
                if self.t < m.short_until:
                    w["charge"] = 0.0
                    continue
                go = False
                if held:
                    w["charge"] += DT
                    go = w["charge"] >= full + st["hold_s"]
                elif w["charge"] >= st["min_charge"]:
                    go = True
                else:
                    w["charge"] += DT
                    go = w["charge"] >= st["min_charge"]
                if go:
                    frac = min(w["charge"] / full, 1.0)
                    dmg = st["damage"] * (st["min_frac"] + (1 - st["min_frac"]) * frac)
                    d = direction(*m.aim)
                    o, end, hitm = self.hitscan(m, d, st["range"], mh)
                    if hitm:
                        cook = hitm.holds("mortar")
                        self.damage(hitm, dmg * (1 + st["cookoff"] if cook else 1.0), "laser", o, m, "cookoff" if cook else None)
                    self.count_shot(m, "laser", hitm is not None)
                    for x in self.mechs:
                        if x is not m:
                            x.beams.append({"by": m.id, "kind": "laser", "path": [[round(v, 2) for v in o], [round(v, 2) for v in end]], "t": self.t})
                    w["charge"], w["cooldown"] = 0.0, st["cooldown"]
                    self.emit("laser", m.pos, m)

    def explode(self, p, st, by, direct):
        R, edge = st["splash_radius"], st["splash_edge"]
        hit_foe = False
        for x in self.mechs:
            if not x.alive:
                continue
            z0, z1 = x.pos[2], x.pos[2] + x.mc["height"]
            cz = min(max(p[2], z0), z1)
            dh = math.hypot(p[0] - x.pos[0], p[1] - x.pos[1])
            d = max(dh - x.mc["radius"], 0.0)
            d = math.hypot(d, p[2] - cz)
            dmg = st["damage"] if x is direct else 0.0
            f = 0.0
            if d <= R and (x is direct or self.A.seg_block(p, [x.pos[0], x.pos[1], cz], 0.5) is None):
                f = 1 - (1 - edge) * d / R
                dmg += st["splash"] * f
                ang = math.atan2(x.pos[1] - p[1], x.pos[0] - p[0])
                x.vel[0] += math.cos(ang) * st["knock"] * f
                x.vel[1] += math.sin(ang) * st["knock"] * f
                x.vel[2] += st["knock_up"] * f
                x.grounded = False
                x.energy = max(x.energy - st["drain"] * f, 0.0)
            if dmg > 0:
                if x is by:
                    dmg *= x.mc["self_damage_mult"]
                self.damage(x, dmg, "mortar", p, by, "concuss" if f > 0 else None)
                if x is not by:
                    hit_foe = True
        self.emit("explosion", p, None)
        return hit_foe

    def step_projectiles(self):
        keep = []
        g = self.cfg["weapons"]["mortar"]["gravity"]
        pr = self.cfg["projectile"].get("gun_radius", 0.3)
        for p in self.projectiles:
            if p["kind"] == "mortar":
                p["vel"][2] -= g * DT
            q = [p["pos"][i] + p["vel"][i] * DT for i in range(3)]
            if p["kind"] == "gun":
                step = math.dist(p["pos"], q)
                if step > p["left"]:
                    q = [p["pos"][i] + (q[i] - p["pos"][i]) * p["left"] / step for i in range(3)]
                p["left"] -= step
            best, hitm = 1.0 + 1e-9, None
            for x in self.mechs:
                if x is p["by"] or not x.alive:
                    continue
                f = seg_cylinder(p["pos"], q, x.pos[0], x.pos[1], x.pos[2], x.pos[2] + x.mc["height"], x.mc["radius"] + (pr if p["kind"] == "gun" else 0))
                if f is not None and f < best:
                    best, hitm = f, x
            end = [p["pos"][i] + (q[i] - p["pos"][i]) * min(best, 1.0) for i in range(3)]
            wall = self.A.seg_block(p["pos"], end, 0.5)
            if wall is not None:
                hitm, end = None, [p["pos"][i] + (end[i] - p["pos"][i]) * wall for i in range(3)]
            done = wall is not None or hitm is not None
            if p["kind"] == "gun":
                if hitm:
                    self.damage(hitm, p["dmg"], "gun", p["pos"], p["by"])
                    p["hit"] = True
                if done or p["left"] <= 0:
                    self.record_proj(p)
                    continue
            else:
                p["ttl"] -= DT
                if done or p["ttl"] <= 0:
                    p["hit"] = self.explode(end, p["st"], p["by"], hitm) if done else False
                    self.record_proj(p)
                    continue
            p["pos"] = q
            keep.append(p)
        self.projectiles = keep

    def record_proj(self, p):
        s = self.stats[p["by"].id]
        b = p["bucket"]
        s["shots"].setdefault(p["kind"], {}).setdefault(b, 0)
        s["hits"].setdefault(p["kind"], {}).setdefault(b, 0)
        s["shots"][p["kind"]][b] += 1
        s["hits"][p["kind"]][b] += 1 if p["hit"] else 0

    # ----- movement -----
    def support(self, x, y, z, step):
        g = self.A.ground(x, y)
        k = self.A.idx(x, y)
        if k >= 0:
            h = self.A.hi[k]
            if g < h <= z + step + 0.05:
                return h
        return g

    def blocked(self, m, x, y):
        r, z, mc, A = m.mc["radius"], m.pos[2], m.mc, self.A
        if not (A.minx + r <= x <= A.maxx - r and A.miny + r <= y <= A.maxy - r):
            return True
        if self.support(x, y, z, mc["step_up"]) > z + mc["step_up"]:
            return True
        top, bot = z + mc["height"], z + mc["step_up"]
        for k in range(9):
            if k == 0:
                px, py = x, y
            else:
                a = k * math.pi / 4
                px, py = x + math.cos(a) * (r - 0.3), y + math.sin(a) * (r - 0.3)
            i = A.idx(px, py)
            if i < 0 or (A.lo[i] < top and A.hi[i] > bot):
                return True
        return False

    def move_mech(self, m):
        mc = m.mc
        mv = list(m.move)
        L = math.hypot(*mv)
        if L > 1:
            mv = [mv[0] / L, mv[1] / L]
            L = 1.0
        boosting = m.boost and m.energy > 0
        speed = mc["boost_speed"] if (boosting and m.grounded) else mc["walk_speed"]
        if m.charging():
            speed *= mc["charge_move_mult"]
        want = [mv[0] * speed, mv[1] * speed]
        acc = (mc["ground_accel"] if m.grounded else mc["air_accel"]) * DT
        dvx, dvy = want[0] - m.vel[0], want[1] - m.vel[1]
        dv = math.hypot(dvx, dvy)
        if dv > acc:
            dvx, dvy = dvx / dv * acc, dvy / dv * acc
        m.vel[0] += dvx
        m.vel[1] += dvy
        if boosting and m.grounded and L > 0.1:
            m.energy = max(m.energy - mc["boost_drain"] * DT, 0.0)
            m.last_spend = self.t
        if m.jump and m.grounded:
            m.vel[2] = mc["jump_speed"]
            m.grounded = False
        m.jump = False
        if not m.grounded:
            if boosting:
                m.vel[2] = min(m.vel[2] + mc["jet_accel"] * DT, mc["max_rise_speed"])
                m.energy = max(m.energy - mc["jet_drain"] * DT, 0.0)
                m.last_spend = self.t
                m.jet_t += DT
                if m.jet_t >= 0.5:
                    m.jet_t = 0.0
                    self.emit("jet", m.pos, m)
            m.vel[2] -= mc["gravity"] * DT
        nx = m.pos[0] + m.vel[0] * DT
        if self.blocked(m, nx, m.pos[1]):
            m.vel[0] = 0.0
        else:
            m.pos[0] = nx
        ny = m.pos[1] + m.vel[1] * DT
        if self.blocked(m, m.pos[0], ny):
            m.vel[1] = 0.0
        else:
            m.pos[1] = ny
        s = self.support(m.pos[0], m.pos[1], m.pos[2], mc["step_up"])
        z = m.pos[2] + m.vel[2] * DT
        if z <= s:
            m.pos[2], m.grounded = s, True
            m.vel[2] = 0.0
        elif m.grounded and m.vel[2] <= 0 and z - s < 1.0:
            m.pos[2] = s
            m.vel[2] = 0.0
        else:
            m.pos[2], m.grounded = z, False
        hs = math.hypot(m.vel[0], m.vel[1])
        if m.grounded and hs > self.cfg["sound"]["step_speed"]:
            m.step_t += DT
            if m.step_t >= self.cfg["sound"]["step_every"]:
                m.step_t = 0.0
                self.emit("boost" if boosting else "step", m.pos, m)

    def slew(self, m):
        mc = m.mc
        dy = ang_diff(m.cmd_aim[0], m.aim[0])
        tr = mc["turn_rate"] * DT
        m.aim[0] = (m.aim[0] + max(-tr, min(tr, dy)) + math.pi) % (2 * math.pi) - math.pi
        want_p = min(max(m.cmd_aim[1], mc["min_pitch"]), mc["max_pitch"])
        pr = mc["pitch_rate"] * DT
        m.aim[1] += max(-pr, min(pr, want_p - m.aim[1]))

    def push_apart(self):
        live = [m for m in self.mechs if m.alive]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                self._push(live[i], live[j])

    def _push(self, a, b):
        dx, dy = b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]
        d = math.hypot(dx, dy)
        rr = a.mc["radius"] + b.mc["radius"]
        if d < rr and abs(a.pos[2] - b.pos[2]) < a.mc["height"]:
            if d < 1e-6:
                dx, dy, d = 1.0, 0.0, 1.0
            push = (rr - d) / 2
            for m, s in ((a, -1), (b, 1)):
                nx, ny = m.pos[0] + s * dx / d * push, m.pos[1] + s * dy / d * push
                if not self.blocked(m, nx, ny):
                    m.pos[0], m.pos[1] = nx, ny

    def upkeep(self, m):
        mc = m.mc
        c, r, dps = self.zone_now()
        m.in_zone = math.hypot(m.pos[0] - c[0], m.pos[1] - c[1]) <= r
        if not m.in_zone:
            m.hp -= dps * DT
            m.last_fight = self.t
            tk = self.stats[m.id].setdefault("taken", {})
            tk["zone"] = tk.get("zone", 0.0) + dps * DT
            if not m.events or m.events[-1].get("weapon") != "zone":
                m.events.append({"type": "damaged", "amount": round(dps * DT, 1), "weapon": "zone", "bearing": None, "by": None})
            if m.hp <= 0:
                m.hp, m.alive, m.died = 0.0, False, self.tick
                self.emit("death", m.pos, m)
                return
        elif self.t - m.last_fight >= mc["regen_delay"]:
            m.hp = min(m.hp + mc["regen_per_s"] * DT, m.max_hp)
        if self.t - m.last_spend >= mc["energy_regen_delay"]:
            m.energy = min(m.energy + mc["energy_regen"] * DT, m.max_energy)

    # ----- observations -----
    def sees(self, o, x):
        if not x.alive:
            return False
        eye = [o.pos[0], o.pos[1], o.pos[2] + o.mc["eye_height"]]
        d = math.hypot(x.pos[0] - o.pos[0], x.pos[1] - o.pos[1])
        vc = self.cfg["vision"]
        near = d <= vc["near_range"] + o.mc["near_range_bonus"]
        wedge = d <= vc["range"] + o.mc["vision_bonus"] and abs(ang_diff(math.atan2(x.pos[1] - o.pos[1], x.pos[0] - o.pos[0]), o.aim[0])) <= math.radians(vc["fov_deg"]) / 2
        if not (near or wedge):
            return False
        for hgt in (x.mc["height"] / 2, x.mc["height"]):
            p = [x.pos[0], x.pos[1], x.pos[2] + hgt]
            if self.A.seg_block(eye, p, 1.0) is None and not self.A.trees_block(eye, p, canopy=not near and vc.get("canopy_blocks", True)):
                return True
        return False

    def sees_point(self, o, p):
        eye = [o.pos[0], o.pos[1], o.pos[2] + o.mc["eye_height"]]
        d = math.hypot(p[0] - o.pos[0], p[1] - o.pos[1])
        vc = self.cfg["vision"]
        if not (d <= vc["near_range"] or (d <= vc["range"] and abs(ang_diff(math.atan2(p[1] - o.pos[1], p[0] - o.pos[0]), o.aim[0])) <= math.radians(vc["fov_deg"]) / 2)):
            return False
        return self.A.seg_block(eye, p, 1.5) is None

    def weapon_obs(self, w):
        o = {"kind": w["kind"], "cooldown": round(max(w["cooldown"], w["equip"], 0.0), 3), "stats": w["stats"]}
        if w["kind"] == "gun":
            o.update(ammo=w["ammo"], reload=round(max(w["reload"], 0.0), 3))
        if w["kind"] == "laser":
            o["charge"] = round(w["charge"], 3)
        return o

    def observe(self, o):
        r2 = lambda v: [round(c, 3) for c in v]
        mechs = []
        for x in self.mechs:
            if x is o or not x.alive:
                continue
            if self.sees(o, x):
                o.seen[x.id] = self.t
            if self.t - o.seen.get(x.id, -1e9) <= self.cfg["vision"].get("memory_s", 5):
                mechs.append({"id": x.id, "kind": "mech", "pos": r2(x.pos), "vel": r2(x.vel), "aim": r2(x.aim),
                              "hp": round(x.hp, 1), "max_hp": x.max_hp, "cubes": 0,
                              "weapons": [w["kind"] for w in x.weapons if w], "charging": x.charging()})
        projs = [{"id": p["id"], "kind": p["kind"], "pos": r2(p["pos"]), "vel": r2(p["vel"])}
                 for p in self.projectiles if p["by"] is not o and self.sees_point(o, p["pos"])]
        beams = [{"by": b["by"], "kind": b["kind"], "path": b["path"], "age": round(self.t - b["t"], 2)}
                 for b in o.beams if any(x.id == b["by"] for x in self.mechs if self.t - o.seen.get(x.id, -1e9) < 0.11)
                 or self.sees_point(o, b["path"][-1])]
        sounds = [{"kind": k, "bearing": round(bb, 4), "dist": dd, "age": round(self.t - ts, 2)} for k, bb, dd, ts in o.sounds]
        pick = [{"id": p["id"], "kind": p["kind"], "pos": r2(p["pos"])} for p in self.pickups if self.sees_point(o, [p["pos"][0], p["pos"][1], p["pos"][2] + 0.5])]
        obs = {"type": "obs", "tick": self.tick, "t": round(self.t, 2), "alive": sum(x.alive for x in self.mechs),
               "self": {"id": o.id, "pos": r2(o.pos), "vel": r2(o.vel), "aim": r2(o.aim), "hp": round(o.hp, 1), "max_hp": o.max_hp,
                        "energy": round(o.energy, 1), "max_energy": o.max_energy, "grounded": o.grounded, "cubes": 0, "upgrades": [],
                        "weapons": [self.weapon_obs(w) if w else None for w in o.weapons], "in_zone": getattr(o, "in_zone", True),
                        "stats": dict(o.mc)},
               "mechs": mechs, "projectiles": projs, "crates": [], "cubes": [], "pickups": pick, "sounds": sounds,
               "beams": beams, "events": o.events, "zone": self.zone(), "offer": None, "drops": []}
        o.events, o.sounds, o.beams = [], [], []
        return obs

    def apply(self, m, a):
        if a is None:  # no answer: coast on move/aim/boost, then idle after coast_s
            m.fire, m.jump = [False, False], False
            if self.t - m.last_answer_t > self.cfg["bot"]["coast_s"]:
                m.move, m.boost = [0.0, 0.0], False
            return
        m.last_answer_t = self.t
        mv = a.get("move", [0, 0])
        if isinstance(mv, list) and len(mv) == 2 and all(isinstance(v, (int, float)) for v in mv):
            m.move = [float(mv[0]), float(mv[1])]
        else:
            m.move = [0.0, 0.0]
        m.boost = bool(a.get("boost", False))
        m.jump = bool(a.get("jump", False))
        aim = a.get("aim")
        if isinstance(aim, list) and len(aim) == 2 and all(isinstance(v, (int, float)) for v in aim):
            m.cmd_aim = [float(aim[0]), float(aim[1])]
        f = a.get("fire", [False, False])
        m.fire = [bool(f[i]) if isinstance(f, list) and i < len(f) else False for i in (0, 1)]
        g = next((w for w in m.weapons if w and w["kind"] == "gun"), None)
        if a.get("reload") and g and g["reload"] <= 0 and g["ammo"] < g["stats"]["mag"]:
            g["reload"] = g["stats"]["reload"]
            self.emit("reload", m.pos, m)
        self.loot(m, a.get("take"), a.get("drop"))

    def loot(self, m, take, drop):
        if drop not in (0, 1):
            drop = None
        if drop is not None and m.weapons[drop] and m.weapons[drop]["kind"] == "laser" and m.weapons[drop]["charge"] > 0:
            if take is not None:
                m.events.append({"type": "take_failed", "id": take, "reason": "busy"})
            return
        if take is not None:
            p = next((p for p in self.pickups if p["id"] == take), None)
            if p is None:
                m.events.append({"type": "take_failed", "id": take, "reason": "gone"})
                return
            if math.hypot(p["pos"][0] - m.pos[0], p["pos"][1] - m.pos[1]) > self.cfg["loot"]["pickup_radius"] or abs(p["pos"][2] - m.pos[2]) > 4:
                m.events.append({"type": "take_failed", "id": take, "reason": "range"})
                return
            if m.holds(p["kind"]):
                m.events.append({"type": "take_failed", "id": take, "reason": "have"})
                return
            slot = drop if drop is not None else (m.weapons.index(None) if None in m.weapons else None)
            if slot is None:
                m.events.append({"type": "take_failed", "id": take, "reason": "full"})
                return
            self.drop_slot(m, slot)
            self.pickups.remove(p)
            w = p["w"]
            w["equip"] = max(self.cfg["loot"]["equip_s"], w["stats"]["cooldown"])
            m.weapons[slot] = w
            m.events.append({"type": "took", "id": p["id"], "kind": w["kind"], "slot": slot})
        elif drop is not None:
            self.drop_slot(m, drop)

    def drop_slot(self, m, slot):
        w = m.weapons[slot]
        if not w:
            return
        if w["kind"] == "laser":
            w["charge"] = 0.0
        pid = self.next_id["pick"]
        self.next_id["pick"] += 1
        self.pickups.append({"id": pid, "kind": w["kind"], "pos": list(m.pos), "w": w})
        m.weapons[slot] = None
        m.events.append({"type": "dropped", "id": pid, "kind": w["kind"], "slot": slot})

    def places(self):
        """Placement per mech (protocol.md §4): death order, same-tick deaths share; at the clock, survivors by HP."""
        alive = sorted((m for m in self.mechs if m.alive), key=lambda m: -m.hp / m.max_hp)
        out = {}
        for k, m in enumerate(alive):
            out[m.id] = 1 + sum(1 for x in alive[:k] if x.hp / x.max_hp > m.hp / m.max_hp + 1e-9)
        for m in self.mechs:
            if not m.alive:
                out[m.id] = 1 + len(alive) + sum(1 for x in self.mechs if not x.alive and x.died > m.died)
        return [out[m.id] for m in self.mechs]

    # ----- run -----
    def run(self):
        hz = self.cfg["match"]["tick_hz"]
        every = self.cfg["match"]["decide_every"]
        try:
            for i, p in enumerate(self.procs):
                p.send({"type": "hello", "protocol": "mechfight/1", "slot": i, "seed": 1000 + i,
                        "map": {"dir": self.map_dir, "files": ["v2/world.json", "v2/terrain.i16"]}, "config": self.cfg})
            dl = time.perf_counter() + self.cfg["bot"]["hello_budget_ms"] / 1000
            for i, p in enumerate(self.procs):
                r = p.wait_for(None, dl)
                self.names[i] = (r or {}).get("name", "?")
            while True:
                alive = [m for m in self.mechs if m.alive]
                if len(alive) < 2 or self.t >= self.max_t:
                    break
                if self.tick % every == 0:
                    for m in alive:
                        self.procs[m.id].send(self.observe(m))
                    dl = time.perf_counter() + self.timeout
                    for m in alive:
                        a = self.procs[m.id].wait_for(self.tick, dl)
                        if a is None:
                            self.stats[m.id]["late"] += 1
                        self.apply(m, a)
                for m in alive:
                    self.slew(m)
                for m in alive:
                    if m.alive:
                        self.fire_weapons(m)
                for m in alive:
                    if m.alive:
                        self.move_mech(m)
                self.step_projectiles()
                self.push_apart()
                for m in alive:
                    if m.alive:
                        self.upkeep(m)
                self.tick += 1
                self.t = self.tick / hz
        finally:
            places = self.places()
            for i, p in enumerate(self.procs):
                p.send({"type": "end", "place": places[i], "kills": 0})
                p.close()
        if len(self.mechs) > 2:
            return {"places": places, "t": round(self.t, 1), "hp": [round(m.hp) for m in self.mechs], "stats": self.stats,
                    "names": self.names, "timeout": self.t >= self.max_t}
        a, b = self.mechs
        if a.alive and not b.alive:
            win = 0
        elif b.alive and not a.alive:
            win = 1
        elif a.alive and b.alive:
            fa, fb = a.hp / a.max_hp, b.hp / b.max_hp
            win = 0 if fa > fb + 1e-6 else 1 if fb > fa + 1e-6 else None
        else:
            win = None
        return {"winner": win, "t": round(self.t, 1), "hp": [round(m.hp) for m in self.mechs], "stats": self.stats, "names": self.names,
                "timeout": self.t >= self.max_t}


# ------------------------------------------------------------------ duel setup
def duel_setup(arena, rng, dmin=90.0, dmax=170.0, max_t=180.0):
    """Random spawn pair on open ground and a zone around them that forces the fight."""
    A = arena
    for _ in range(5000):
        D = rng.uniform(dmin, dmax)
        R0 = D / 2 + 50
        cx = rng.uniform(A.minx + 60, A.maxx - 60)
        cy = rng.uniform(A.miny + 60, A.maxy - 60)
        ang = rng.uniform(-math.pi, math.pi)
        pts = []
        for s in (1, -1):
            x, y = cx + s * math.cos(ang) * D / 2, cy + s * math.sin(ang) * D / 2
            if not (A.minx + 20 < x < A.maxx - 20 and A.miny + 20 < y < A.maxy - 20):
                break
            g = A.ground(x, y)
            ok = True
            for k in range(17):  # a 6 m disc of open ground, no solid above the feet
                a = k * math.pi / 8
                rr = 0 if k == 0 else (3 if k < 9 else 6)
                px, py = x + math.cos(a) * rr, y + math.sin(a) * rr
                i = A.idx(px, py)
                if i < 0 or A.lo[i] < g + 12 or abs(A.ground(px, py) - g) > 1.0:
                    ok = False
                    break
            if not ok:
                break
            pts.append([x, y, g, rng.uniform(-math.pi, math.pi)])
        if len(pts) < 2:
            continue
        mid = [(pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2]
        if A.seg_block([pts[0][0], pts[0][1], pts[0][2] + 4.5], [pts[1][0], pts[1][1], pts[1][2] + 4.5]) is None and rng.random() < 0.5:
            continue  # half the duels start without a clear line between the spawns (more campus, less open field)

        def inner(c, r, rn):
            a, d = rng.uniform(-math.pi, math.pi), rng.uniform(0, max(r - rn, 0) * 0.6)
            return [c[0] + math.cos(a) * d, c[1] + math.sin(a) * d]
        c1 = inner(mid, R0, R0 * 0.55)
        c2 = inner(c1, R0 * 0.55, R0 * 0.2)
        zone = {"center": mid, "radius": R0, "phases": [
            {"hold_s": 15, "shrink_s": 30, "radius": R0 * 0.55, "dps": 20, "center": c1},
            {"hold_s": 15, "shrink_s": 30, "radius": R0 * 0.2, "dps": 60, "center": c2},
            {"hold_s": 10, "shrink_s": 25, "radius": 0, "dps": 160, "center": c2}]}
        return pts, zone
    raise RuntimeError("no duel spawn found")


def ffa_setup(arena, rng, n, sep=90.0):
    """n spawns on open ground at least `sep` m apart inside a circle, and a zone that closes on them over ~3 min."""
    A = arena
    for _ in range(200):
        R0 = 60 + 40 * math.sqrt(n)
        cx = rng.uniform(A.minx + R0 * 0.6, A.maxx - R0 * 0.6)
        cy = rng.uniform(A.miny + R0 * 0.6, A.maxy - R0 * 0.6)
        pts = []
        for _ in range(4000):
            if len(pts) == n:
                break
            a, d = rng.uniform(-math.pi, math.pi), R0 * 0.85 * math.sqrt(rng.random())
            x, y = cx + math.cos(a) * d, cy + math.sin(a) * d
            if not (A.minx + 20 < x < A.maxx - 20 and A.miny + 20 < y < A.maxy - 20):
                continue
            if any(math.hypot(x - p[0], y - p[1]) < sep for p in pts):
                continue
            g = A.ground(x, y)
            ok = True
            for k in range(17):
                aa = k * math.pi / 8
                rr = 0 if k == 0 else (3 if k < 9 else 6)
                px, py = x + math.cos(aa) * rr, y + math.sin(aa) * rr
                i = A.idx(px, py)
                if i < 0 or A.lo[i] < g + 12 or abs(A.ground(px, py) - g) > 1.0:
                    ok = False
                    break
            if ok:
                pts.append([x, y, g, rng.uniform(-math.pi, math.pi)])
        if len(pts) < n:
            continue

        def inner(c, r, rn):
            a, d = rng.uniform(-math.pi, math.pi), rng.uniform(0, max(r - rn, 0) * 0.6)
            return [c[0] + math.cos(a) * d, c[1] + math.sin(a) * d]
        c0 = [cx, cy]
        c1 = inner(c0, R0, R0 * 0.6)
        c2 = inner(c1, R0 * 0.6, R0 * 0.3)
        c3 = inner(c2, R0 * 0.3, R0 * 0.1)
        zone = {"center": c0, "radius": R0, "phases": [
            {"hold_s": 30, "shrink_s": 35, "radius": R0 * 0.6, "dps": 20, "center": c1},
            {"hold_s": 25, "shrink_s": 30, "radius": R0 * 0.3, "dps": 35, "center": c2},
            {"hold_s": 20, "shrink_s": 25, "radius": R0 * 0.1, "dps": 60, "center": c3},
            {"hold_s": 15, "shrink_s": 20, "radius": 0, "dps": 160, "center": c3}]}
        return pts, zone
    raise RuntimeError("no ffa spawns found")
