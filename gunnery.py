"""Aim and triggers (GUIDELINES §3.4).

Predictor ("virtual guns"): per target, a family of motion models scored online. Every decision we ask each model
where, from the sample h seconds ago, it would have put the target now, and score whether that lands inside the
body across our line of sight. Asked for tau seconds ahead, the best model at the nearest horizon is used (the
target's own scores once it has enough, the match-wide ones before).

Gates in metres: a weapon fires only if the shot, along the aim the torso will really have on the tick it leaves
(the torso slews toward our command at turn_rate), passes within GATE of the body's radius of where the target
will be, and nothing solid is in the way.
"""
import math

DT = 0.05            # s per sim tick: a shot leaves on the first or second tick after our answer
HORIZONS = (0.1, 0.3, 0.5, 0.7, 0.9)
MODELS = (("vel", 0.0), ("vel", 0.5), ("vel", 0.8), ("vel", 1.0), ("mean", 10), ("mean", 20), ("center", 10), ("center", 20))
ALPHA = 0.08         # score EMA rate
OWN_N = 10           # samples before a target's own scores are used
PRIOR = 2            # ("vel", 0.8): used until the scores say otherwise
GATE = 0.8           # share of the body radius (and half height) a shot may miss by
LASER_START = 1.2    # rad: start charging once the torso is this close (it turns 3 rad in a 0.75 s charge)
SHOT_OUT = 3.0       # m: shots leave this far out along the aim


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def disp(h, i, model, tau):
    """Horizontal move that `model` predicts from history sample i (t, x, y, z, vx, vy, vz) over tau s."""
    kind, k = model
    if kind == "vel":
        return k * h[i][4] * tau, k * h[i][5] * tau
    j = max(0, i - k + 1)
    n = i - j + 1
    mvx = sum(s[4] for s in h[j:i + 1]) / n
    mvy = sum(s[5] for s in h[j:i + 1]) / n
    if kind == "mean":
        return mvx * tau, mvy * tau
    cx = sum(s[1] for s in h[j:i + 1]) / n  # "center": the middle of its recent dance, carried along
    cy = sum(s[2] for s in h[j:i + 1]) / n
    lag = (h[i][0] - h[j][0]) / 2
    return cx + mvx * (tau + lag) - h[i][1], cy + mvy * (tau + lag) - h[i][2]


class Predictor:
    def __init__(self):
        self.own, self.n, self.done = {}, {}, {}
        self.glob = [[0.5 + (0.05 if m == PRIOR else 0.0) for m in range(len(MODELS))] for _ in HORIZONS]

    @staticmethod
    def _at(h, t):
        k = len(h) - 1 - int(round((h[-1][0] - t) / 0.1))
        for j in (k, k - 1, k + 1):
            if 0 <= j < len(h) and abs(h[j][0] - t) < 0.04:
                return j
        return None

    def learn(self, tr, me_xy, radius):
        h = tr.hist
        if len(h) < 3 or self.done.get(tr.id) == h[-1][0]:
            return
        self.done[tr.id] = h[-1][0]
        own = self.own.setdefault(tr.id, [[0.5] * len(MODELS) for _ in HORIZONS])
        tn, xn, yn = h[-1][0], h[-1][1], h[-1][2]
        dx, dy = xn - me_xy[0], yn - me_xy[1]
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        scored = False
        for hi, H in enumerate(HORIZONS):
            j = self._at(h, tn - H)
            if j is None or j == len(h) - 1:
                continue
            tau = tn - h[j][0]
            for mi, m in enumerate(MODELS):
                px, py = disp(h, j, m, tau)
                ex, ey = xn - h[j][1] - px, yn - h[j][2] - py
                hit = max(0.0, 1.0 - ((ex * uy - ey * ux) / radius) ** 2)  # 1 dead centre, 0 at the body's edge
                own[hi][mi] += ALPHA * (hit - own[hi][mi])
                self.glob[hi][mi] += 0.25 * ALPHA * (hit - self.glob[hi][mi])
            scored = True
        if scored:
            self.n[tr.id] = self.n.get(tr.id, 0) + 1

    def predict(self, tr, t, tau, floor_at=None, gravity=22.0):
        """(x, y, z) of the target's feet tau s after now."""
        h = tr.hist
        if not h:
            return tuple(tr.pos)
        last = h[-1]
        tau = tau + max(0.0, t - last[0])
        hi = min(range(len(HORIZONS)), key=lambda i: abs(HORIZONS[i] - tau))
        row = (self.own[tr.id] if self.n.get(tr.id, 0) >= OWN_N else self.glob)[hi]
        mi = max(range(len(MODELS)), key=row.__getitem__)
        px, py = disp(h, len(h) - 1, MODELS[mi], tau)
        x, y, z, vz = last[1] + px, last[2] + py, last[3], last[6]
        if abs(vz) > 0.5:  # in the air: it falls
            z = z + vz * tau - 0.5 * gravity * tau * tau
            if floor_at is not None:
                z = max(z, floor_at(x, y, z))
        return x, y, z


def aim_at(src, p):
    dx, dy, dz = p[0] - src[0], p[1] - src[1], p[2] - src[2]
    return [math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))]


def slew(aim, want, turn, tilt, ticks):
    """Where the torso points after `ticks` sim ticks of turning toward `want`."""
    dy = max(-turn * DT * ticks, min(turn * DT * ticks, wrap(want[0] - aim[0])))
    dp = max(-tilt * DT * ticks, min(tilt * DT * ticks, want[1] - aim[1]))
    return [aim[0] + dy, aim[1] + dp]


def miss(src, aim, p):
    """(sideways, vertical) metres by which a shot leaving src along aim passes point p (1e9 if p is behind)."""
    yaw, pitch = aim
    d = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
    o = (src[0] + SHOT_OUT * d[0], src[1] + SHOT_OUT * d[1], src[2] + SHOT_OUT * d[2])
    v = (p[0] - o[0], p[1] - o[1], p[2] - o[2])
    along = v[0] * d[0] + v[1] * d[1] + v[2] * d[2]
    if along < 0:
        return 1e9, 1e9
    c = (o[0] + along * d[0], o[1] + along * d[1], o[2] + along * d[2])
    return math.hypot(p[0] - c[0], p[1] - c[1]), abs(p[2] - c[2])


class Target:
    """What to shoot: a Track (mech or drone) or a fixed point (crate), its body, and the weapons allowed on it."""

    def __init__(self, pos, radius, height, allow, track=None):
        self.pos, self.radius, self.height, self.allow, self.track = pos, radius, height, set(allow), track


class Gunner:
    def __init__(self, cfg, world):
        self.w = world
        self.pred = Predictor()
        self.gravity = cfg["mech"].get("gravity", 22.0)

    def _clear(self, a, b):
        return self.w is None or self.w.clear(a, b)

    def point(self, tgt, t, tau):
        """Centre of the target's body tau s from now."""
        if tgt.track is None:
            x, y, z = tgt.pos
        else:
            x, y, z = self.pred.predict(tgt.track, t, tau, self.w.floor_at if self.w else None, self.gravity)
        return x, y, z + tgt.height / 2

    def plan(self, me, t, tgt, shorted=False):
        """(aim, fire, info) for one decision against tgt (None: no target, nothing fires)."""
        st = me["stats"]
        turn, tilt = st.get("turn_rate", 4.0), st.get("pitch_rate", 3.0)
        mz = (me["pos"][0], me["pos"][1], me["pos"][2] + st.get("muzzle_height", 7.0))
        weapons = me["weapons"]
        fire = [False] * len(weapons)
        if tgt is None:
            for i, w in enumerate(weapons):  # a charge in progress is held (it fires by itself after hold_s)
                if w and w["kind"] == "laser" and w.get("charge", 0) > 0:
                    fire[i] = True
            return None, fire, ""
        if tgt.track is not None:
            self.pred.learn(tgt.track, me["pos"], tgt.radius)
        slots = {w["kind"]: i for i, w in enumerate(weapons) if w}
        laser = weapons[slots["laser"]] if "laser" in slots else None
        laser_on = laser is not None and "laser" in tgt.allow and not shorted and (laser.get("charge", 0) > 0 or laser["cooldown"] <= 0.3)
        # the torso follows the laser while it is (nearly) ready, else the gun's lead, else the body itself
        if laser_on:
            p = self.point(tgt, t, DT)
        elif "gun" in slots and "gun" in tgt.allow:
            p = self.lead(tgt, t, mz, weapons[slots["gun"]], DT)
        else:
            p = self.point(tgt, t, DT)
        aim = aim_at(mz, p)
        rx, rz = GATE * tgt.radius, GATE * tgt.height / 2
        info = []
        for kind, i in slots.items():
            w = weapons[i]
            if kind == "laser":
                if shorted or "laser" not in tgt.allow:
                    fire[i] = w.get("charge", 0) > 0
                    continue
                pl = self.point(tgt, t, DT)
                if w.get("charge", 0) > 0:
                    # the tick we let go still charges, so one tick short of full is full when it fires
                    full = w.get("charge", 0) >= w["stats"]["charge"] - DT - 1e-6
                    sx, sz = miss(mz, slew(me["aim"], aim, turn, tilt, 1), pl)
                    on = sx < rx and sz < rz and math.dist(mz, pl) <= w["stats"]["range"] and self._clear(mz, pl)
                    fire[i] = not (full and on)  # let go once full and on target; else keep holding
                    info.append("L!" if not fire[i] else "Lc")
                elif w["cooldown"] <= 2 * DT and math.dist(mz, pl) <= w["stats"]["range"]:  # held, it starts the tick it can
                    err = abs(wrap(aim[0] - me["aim"][0])) + abs(aim[1] - me["aim"][1])
                    fire[i] = err < LASER_START and self._clear(mz, pl)
            elif kind == "gun":
                if "gun" not in tgt.allow or w.get("ammo", 0) <= 0 or w.get("reload", 0) > 0:
                    continue
                cd = w["cooldown"]
                ticks = 1 if cd <= DT else 2 if cd <= 2 * DT else 0
                if not ticks:
                    continue
                pg = self.lead(tgt, t, mz, w, ticks * DT)
                if math.dist(mz, pg) > w["stats"]["range"]:
                    continue
                sx, sz = miss(mz, slew(me["aim"], aim, turn, tilt, ticks), pg)
                fire[i] = sx < rx and sz < rz and self._clear(mz, pg)
                if fire[i]:
                    info.append("G")
            elif kind == "lightning":
                s = w["stats"]
                pz = self.point(tgt, t, DT)
                a1 = slew(me["aim"], aim, turn, tilt, 1)
                yaw_err = abs(wrap(math.atan2(pz[1] - mz[1], pz[0] - mz[0]) - a1[0]))
                fire[i] = ("lightning" in tgt.allow and w["cooldown"] <= DT and me["energy"] >= s["energy"] + 5
                           and math.dist(mz, pz) <= 0.95 * s["range"] and yaw_err < 0.7 * math.radians(s["cone_deg"]) / 2
                           and self._clear(mz, pz))
        return aim, fire, "".join(info)

    def lead(self, tgt, t, mz, gun, latency):
        """Where to point the gun: the target's body when a bullet fired after `latency` s reaches it."""
        speed = gun["stats"]["speed"]
        tau = latency
        p = self.point(tgt, t, tau)
        for _ in range(3):
            tau = latency + max(0.0, math.dist(mz, p) - SHOT_OUT) / speed
            p = self.point(tgt, t, tau)
        return p
