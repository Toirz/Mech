"""Where will a mech be in tau seconds? An online-scored family of motion predictors, per target.

Every observation adds the target's (t, pos, vel) to its history. In hindsight, for each horizon in
HORIZONS, we take the sample from that long ago, ask every predictor where it would have put the target
now, and score it: did it land within the target's radius (a hit), and how far off was it. The predictor
with the best recent hit rate at the horizon nearest the one asked for is used. A target's own scores
take over from the match-wide ones once it has enough samples, so a bot that strafes on a rhythm, runs
straight, or stands still each get the predictor that fits them.

Predictors (horizontal displacement over tau):
- vel k:      k * velocity * tau (k = 0 is "it stays put")
- mean n:     mean velocity of the last n samples * tau (a strafer's drift)
- ou n tc:    mean velocity * tau plus the current deviation from it, decaying with time constant tc
              (a strafer keeps its current direction for about tc, then its average takes over)
- center n w: the middle of its last n positions carried on at their mean velocity (where a strafer's
              dance is centred), pulled a fraction w toward the half-velocity lead
- strafe:     the target's own rhythm, from memory. Its reversals (its run turning the other way) are
              detected, and every sample is filed by its age (seconds since the last reversal) with how
              far it then moved along / across its run over each horizon. Asked again at a similar age,
              we aim where the most of those remembered moves end within a body radius (not at their
              mean, which between "turns now" and "keeps running" hits neither).
A hit is scored on the error across our line of sight: along it, a bullet still goes through the body.
Vertical: airborne targets fall at gravity; grounded ones follow the ground under the predicted spot.
"""
import math
from bisect import bisect_left, bisect_right

HORIZONS = (0.1, 0.3, 0.5, 0.7, 0.9)
KEEP_S = 3.5          # s of history kept per target
ALPHA = 0.08          # EMA rate of the scores
OWN_N = 12            # samples of a target's own before its scores outweigh the match-wide ones
PREDICTORS = ([("vel", k) for k in (0.0, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0)]
              + [("mean", 10), ("mean", 20)]
              + [("ou", 10, 0.3), ("ou", 20, 0.15), ("ou", 20, 0.3)]
              + [("center", 14, 0.3), ("center", 14, 0.5), ("center", 20, 0.5)]
              + [("strafe",)])
FAST = 4.0            # m/s: running, for spotting reversals
MAX_FLIPS = 16        # reversals remembered per target
MEM = 160             # remembered moves per target and horizon
AGE_WIN = (0.06, 0.12, 0.2)  # s: how close in age a remembered move must be (widened until there are enough)
PRIOR = ("vel", 0.5)  # best in duels against the old bots, used until there is data


def _mean_v(h, i, n):
    s = h[max(0, i - n + 1): i + 1]
    return (sum(e[2][0] for e in s) / len(s), sum(e[2][1] for e in s) / len(s))


def displacement(h, i, p, tau, memory=None):
    """Horizontal displacement over tau predicted by predictor p from history h up to index i."""
    v = h[i][2]
    if p[0] == "strafe":
        r = memory(h[i][0]) if memory else None
        if r is None:
            return v[0] * tau * 0.5, v[1] * tau * 0.5
        return r(tau)
    if p[0] == "vel":
        return v[0] * tau * p[1], v[1] * tau * p[1]
    vm = _mean_v(h, i, p[1])
    if p[0] == "mean":
        return vm[0] * tau, vm[1] * tau
    if p[0] == "center":
        s = h[max(0, i - p[1] + 1): i + 1]
        cx, cy = sum(e[1][0] for e in s) / len(s), sum(e[1][1] for e in s) / len(s)
        ct = sum(e[0] for e in s) / len(s)
        dt = h[i][0] + tau - ct
        bx, by = cx + vm[0] * dt - h[i][1][0], cy + vm[1] * dt - h[i][1][1]  # displacement to the dance's centre
        w = p[2]
        return bx + (v[0] * tau * 0.5 - bx) * w, by + (v[1] * tau * 0.5 - by) * w
    tc = p[2]
    f = tc * (1 - math.exp(-tau / tc))
    return vm[0] * tau + (v[0] - vm[0]) * f, vm[1] * tau + (v[1] - vm[1]) * f


class Score:
    __slots__ = ("hit", "sq", "n")

    def __init__(self):
        self.hit, self.sq, self.n = 0.5, 4.0, 0

    def add(self, hit, err2):
        a = max(ALPHA, 1.0 / (self.n + 1))
        self.hit += a * (hit - self.hit)
        self.sq += a * (err2 - self.sq)
        self.n += 1


class Predictor:
    def __init__(self, gravity=22.0, ground=None):
        self.g = gravity
        self.flips, self.run = {}, {}  # id -> reversal times; id -> (unit run direction, last time it ran)
        self.radius, self.mem, self.recalled = {}, {}, {}  # recalled: id -> {sample time: recall()}
        self.ground = ground          # ground(x, y) -> z, or None
        self.hist = {}                # id -> [(t, pos, vel)]
        self.own = {}                 # (id, p, horizon) -> Score
        self.all = {(p, hz): Score() for p in PREDICTORS for hz in HORIZONS}

    def update(self, t, mid, pos, vel, radius, observer=None):
        """Add an observation of target mid and score the predictors on it (across the line from
        `observer` when given, else on the whole horizontal error)."""
        h = self.hist.setdefault(mid, [])
        if h and t <= h[-1][0] + 1e-6:
            return
        if h and t - h[-1][0] > 0.6:  # out of sight a while: start over
            h.clear()
        h.append((t, list(pos), list(vel)))
        while h and t - h[0][0] > KEEP_S:
            h.pop(0)
        c = self.recalled.get(mid)
        for k in [k for k in c if k < h[0][0]] if c else ():  # samples no longer in the history
            del c[k]
        sp = math.hypot(vel[0], vel[1])
        if sp >= FAST:  # a reversal: running the other way from its last run
            u = (vel[0] / sp, vel[1] / sp)
            last = self.run.get(mid)
            if last and u[0] * last[0][0] + u[1] * last[0][1] < -0.5 and t - last[1] < 1.5:
                fl = self.flips.setdefault(mid, [])
                fl.append((t + last[1]) / 2)
                del fl[:-MAX_FLIPS]
            self.run[mid] = (u, t)
        self.radius[mid] = radius
        r2 = radius * radius
        side = None
        if observer is not None:
            a = math.atan2(pos[1] - observer[1], pos[0] - observer[0])
            side = (-math.sin(a), math.cos(a))
        for hz in HORIZONS:
            i = self._index_at(h, t - hz)
            if i is None:
                continue
            tau = t - h[i][0]
            mem_at = lambda t0, hi=h[i]: self.recall(mid, t0, hi, radius, t0 + 1e-6)
            self.remember(mid, h[i], hz, pos, t)
            for p in PREDICTORS:
                dx, dy = displacement(h, i, p, tau, mem_at)
                ex, ey = h[i][1][0] + dx - pos[0], h[i][1][1] + dy - pos[1]
                e2 = (ex * side[0] + ey * side[1]) ** 2 if side else ex * ex + ey * ey
                hit = 1.0 if e2 < r2 else 0.0
                self.all[(p, hz)].add(hit, e2)
                s = self.own.get((mid, p, hz))
                if s is None:
                    s = self.own[(mid, p, hz)] = Score()
                s.add(hit, e2)

    def age(self, mid, t):
        """Seconds since mid's last reversal at or before t, or None."""
        fl = self.flips.get(mid)
        if not fl:
            return None
        for f in reversed(fl):
            if f <= t:
                return t - f if t - f < 3.0 else None
        return None

    def remember(self, mid, sample, hz, pos_now, t_now):
        """File how far mid moved over hz from `sample`, along / across its run then, by its age then."""
        t0, p0, v0 = sample
        sp = math.hypot(v0[0], v0[1])
        a = self.age(mid, t0)
        if sp < FAST or a is None:
            return
        u = (v0[0] / sp, v0[1] / sp)
        dx, dy = pos_now[0] - p0[0], pos_now[1] - p0[1]
        m = self.mem.setdefault((mid, hz), [])
        m.append((a, dx * u[0] + dy * u[1], -dx * u[1] + dy * u[0], t_now))
        del m[:-MEM]

    def recall(self, mid, t0, sample, radius, known_by):
        """tau -> displacement from `sample` (at time t0) by remembered moves, or None without enough of them.
        Only moves filed by `known_by` count (honest hindsight scoring). Worked out once per sample: the
        prediction at t0 and the hindsight scores at each horizon after it all ask for the same one."""
        c = self.recalled.setdefault(mid, {})
        if t0 not in c:
            c[t0] = self._recall(mid, t0, sample, radius, known_by)
        return c[t0]

    def _recall(self, mid, t0, sample, radius, known_by):
        _, _, v0 = sample
        sp = math.hypot(v0[0], v0[1])
        a = self.age(mid, t0)
        if sp < FAST or a is None:
            return None
        u = (v0[0] / sp, v0[1] / sp)
        modes = {}
        for hz in HORIZONS:
            ms = [e for e in self.mem.get((mid, hz), ()) if e[3] <= known_by]
            near = []
            for win in AGE_WIN:
                near = [e for e in ms if abs(e[0] - a) < win]
                if len(near) >= 6:
                    break
            if len(near) < 6:
                continue
            along = sorted(f[1] for f in near)  # count neighbours by bisection: O(n log n), not O(n^2)
            best, bn = None, -1
            for e in near:
                n = bisect_left(along, e[1] + radius) - bisect_right(along, e[1] - radius)
                if n > bn:
                    best, bn = e, n
            xs = [f[1] for f in near if abs(f[1] - best[1]) < radius]
            ys = [f[2] for f in near if abs(f[1] - best[1]) < radius]
            modes[hz] = (sum(xs) / len(xs), sum(ys) / len(ys))
        if len(modes) < 2:
            return None
        hzs = sorted(modes)

        def at(tau):
            if tau <= hzs[0]:
                x, y = (c * tau / hzs[0] for c in modes[hzs[0]])
            elif tau >= hzs[-1]:
                x, y = modes[hzs[-1]]
            else:
                k = next(j for j in range(len(hzs) - 1) if hzs[j] <= tau <= hzs[j + 1])
                f = (tau - hzs[k]) / (hzs[k + 1] - hzs[k])
                x, y = (modes[hzs[k]][j] + (modes[hzs[k + 1]][j] - modes[hzs[k]][j]) * f for j in (0, 1))
            return u[0] * x - u[1] * y, u[1] * x + u[0] * y
        return at

    @staticmethod
    def _index_at(h, t):
        for i in range(len(h) - 1, -1, -1):
            if abs(h[i][0] - t) < 0.051:
                return i
            if h[i][0] < t - 0.06:
                return None
        return None

    def _score(self, mid, p, hz):
        g = self.all[(p, hz)]
        s = self.own.get((mid, p, hz))
        if s is None or s.n == 0:
            return g.hit, g.sq, g.n
        w = s.n / (s.n + OWN_N)
        return w * s.hit + (1 - w) * g.hit, w * s.sq + (1 - w) * g.sq, s.n + g.n

    def best(self, mid, tau):
        """(predictor, expected hit rate, rms error m) for target mid at horizon tau."""
        hz = min(HORIZONS, key=lambda x: abs(x - tau))
        best, bs = PRIOR, None
        for p in PREDICTORS:
            hit, sq, n = self._score(mid, p, hz)
            if n < 5:
                continue
            if bs is None or hit > bs[0] + 1e-9 or (abs(hit - bs[0]) < 1e-9 and sq < bs[1]):
                best, bs = p, (hit, sq)
        if bs is None:
            return PRIOR, None, None
        return best, bs[0], math.sqrt(bs[1])

    def predict(self, mid, pos, vel, tau, grounded=None):
        """Predicted feet position of target mid after tau s, plus (predictor, hit rate, rms error)."""
        h = self.hist.get(mid)
        p, hit, rms = self.best(mid, tau)
        if h and abs(h[-1][1][0] - pos[0]) < 1e-6 and abs(h[-1][1][1] - pos[1]) < 1e-6:
            dx, dy = displacement(h, len(h) - 1, p, tau, lambda t0: self.recall(mid, t0, h[-1], self.radius.get(mid, 2.8), t0 + 1e-6))
        else:
            dx, dy = vel[0] * tau * 0.5, vel[1] * tau * 0.5
        x, y = pos[0] + dx, pos[1] + dy
        z = pos[2]
        air = abs(vel[2]) > 0.3 if grounded is None else not grounded
        g0 = self.ground(pos[0], pos[1]) if self.ground else None
        g1 = self.ground(x, y) if self.ground else None
        if air:
            z = pos[2] + vel[2] * tau - self.g * tau * tau / 2
            if g1 is not None and abs(pos[2] - g0) < 30:
                z = max(z, g1)
        elif g0 is not None and g1 is not None and abs(pos[2] - g0) < 1.5:  # on the ground (not a roof): follow it
            z = pos[2] + (g1 - g0)
        return [x, y, z], (p, hit, rms)
