"""Enemy memory and the engagement model: who can hurt us, how fast, and do we win the trade.

Foes: every mech we saw (kept FOE_S s after it left view, its spot carried on by its last velocity for a
moment, then held with a growing uncertainty) plus ghosts: mechs we only heard, or that hit us from out
of view, placed along the bearing at the distance the sound class says.

Damage rates per weapon come from the config: gun = its sustained rate (mag, cooldown, reload) times a hit
chance that falls with distance; laser = a full beam per charge + cooldown; lightning = its rate, capped by
the energy it costs; mortar = a rough share of its shells landing. Cubes add damage_per_cube each.

engagement(): we kill the foes that fight us one by one, in the order that minimises the damage we take
(Smith's rule: highest dps / time-to-kill first), while every one still standing keeps shooting. The damage
taken over our effective HP is the risk: under 1 we expect to win the trade, over 1 to lose it.
"""
import math

FOE_S = 25.0        # s we remember a mech after it left view
GHOST_S = 10.0      # s a heard-only mech counts
CARRY_S = 1.5       # s we carry a lost mech along its last velocity
SIDE_S = 5.0        # s: the most a foe not in the fight yet (or one we cannot hit back) is counted shooting us
SOUND_DIST = {"near": 35.0, "mid": 100.0, "far": 210.0}
MECH_SOUNDS = {"gun", "mortar", "laser", "laser_charge", "lightning", "step", "boost", "reload", "jet"}
LOUD = {"gun", "laser", "lightning", "mortar", "laser_charge"}  # a mech that fires is in a fight


def gun_hit_p(d):
    """Share of gun rounds that land at distance d (duel stats: ~80% close, ~60% at 100 m, ~35% past 200 m)."""
    return max(0.25, min(0.85, 0.95 - d * 0.0032))


def bearing(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def ang_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class Foe:
    __slots__ = ("id", "pos", "vel", "hp", "max_hp", "cubes", "weapons", "aim", "t_seen", "t_hit_us", "t_we_hit",
                 "t_loud", "charging", "ghost", "t_heard", "busy_t")

    def __init__(self, fid):
        self.id = fid
        self.pos, self.vel, self.aim = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0]
        self.hp, self.max_hp, self.cubes, self.weapons = 1400.0, 1400.0, 0, ["gun"]
        self.t_seen = self.t_hit_us = self.t_we_hit = self.t_loud = self.t_heard = self.busy_t = -1e9
        self.charging, self.ghost = False, True


class Threats:
    def __init__(self, cfg):
        self.cfg = cfg
        self.foes = {}           # id -> Foe (ids < 0: ghosts we only heard)
        self.next_ghost = -1
        self.t = 0.0
        w = cfg["weapons"]
        g = w["gun"]
        mag, cd, rl = g.get("mag", 12), g.get("cooldown", 0.24), g.get("reload", 2.4)
        self.gun_dps = g.get("damage", 45) * mag / (mag * cd + rl)
        la = w["laser"]
        self.laser_dps = la.get("damage", 315) / (la.get("charge", 0.75) + la.get("cooldown", 1.65))
        li = w["lightning"]
        ms = cfg["mech"]
        regen = ms.get("energy_regen", 20)
        per_s = min(1.0 / max(li.get("cooldown", 1.2), 0.1), regen / max(li.get("energy", 30), 1) * 1.5)
        self.lightning_dps = li.get("damage", 240) * per_s
        mo = w["mortar"]
        self.mortar_dps = 0.35 * (mo.get("damage", 260) + mo.get("splash", 160)) / 2 / max(mo.get("cooldown", 2.5), 0.5)
        self.range = {k: w[k].get("range", 230) for k in ("gun", "laser", "lightning")}
        self.range["mortar"] = mo.get("speed", 72) ** 2 / max(mo.get("gravity", 22), 1) * 0.95
        cu = cfg.get("cubes", {})
        self.dpc, self.cap = cu.get("damage_per_cube", 0.04), cu.get("stat_cap", 10)
        self.rear_mult = ms.get("rear_mult", 1.5)
        self.vision = cfg.get("vision", {}).get("range", 195)

    # ----- memory -----
    def update(self, t, obs, pos):
        self.t = t
        for m in obs["mechs"]:
            if m["kind"] != "mech":
                continue
            f = self.foes.get(m["id"])
            if f is None:
                f = self.foes[m["id"]] = Foe(m["id"])
                self._absorb_ghost(m["pos"])
            f.pos, f.vel, f.aim = list(m["pos"]), list(m["vel"]), list(m.get("aim", f.aim))
            f.hp, f.max_hp = m.get("hp", f.hp), m.get("max_hp", f.max_hp)
            f.cubes, f.weapons = m.get("cubes", 0), m.get("weapons") or f.weapons
            f.charging, f.ghost, f.t_seen = bool(m.get("charging")), False, t
        for b in obs.get("beams", []):
            f = self.foes.get(b["by"])
            if f:
                f.t_loud = t
                # a beam that ends far from us went at someone else: it is busy
                end = b["path"][-1] if b.get("path") else None
                if end and math.hypot(end[0] - pos[0], end[1] - pos[1]) > 12:
                    f.busy_t = t
        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("by") is not None:
                f = self.foes.get(e["by"])
                if f is None or (f.ghost and t - f.t_seen > 1):
                    if f is None:
                        f = self.foes[e["by"]] = Foe(e["by"])
                        f.t_heard = t
                    if e.get("bearing") is not None:  # out of view: put it along the bearing, at its weapon's usual range
                        d = {"laser": 150.0, "gun": 120.0, "lightning": 35.0, "mortar": 150.0}.get(e.get("weapon"), 120.0)
                        f.pos = [pos[0] + d * math.cos(e["bearing"]), pos[1] + d * math.sin(e["bearing"]), pos[2]]
                        if e.get("weapon") in ("gun", "laser", "lightning", "mortar") and e["weapon"] not in f.weapons:
                            f.weapons = list(f.weapons) + [e["weapon"]]
                f.t_hit_us = t
            elif e["type"] == "hit":
                f = self.foes.get(e["target"])
                if f:
                    f.t_we_hit = t
            elif e["type"] == "kill":
                self.foes.pop(e["target"], None)
        for s in obs.get("sounds", []):
            if s["kind"] not in MECH_SOUNDS:
                continue
            ts = t - s.get("age", 0)
            d = SOUND_DIST.get(s["dist"], 120.0)
            p = [pos[0] + d * math.cos(s["bearing"]), pos[1] + d * math.sin(s["bearing"]), pos[2]]
            f = self._match(p, s["bearing"], pos, d)
            if f is None:
                f = self.foes[self.next_ghost] = Foe(self.next_ghost)
                self.next_ghost -= 1
                f.pos = p
            elif f.ghost or t - f.t_seen > 2:
                f.pos = p if f.ghost else f.pos
            f.t_heard = max(f.t_heard, ts)
            if s["kind"] in LOUD:
                f.t_loud = max(f.t_loud, ts)
            if s["kind"] in ("laser", "laser_charge") and "laser" not in f.weapons and f.ghost:
                f.weapons = list(f.weapons) + ["laser"]
        self.foes = {k: f for k, f in self.foes.items()
                     if (not f.ghost and t - f.t_seen < FOE_S) or (f.ghost and t - max(f.t_heard, f.t_hit_us) < GHOST_S)}

    def _match(self, p, b, pos, d):
        """A remembered foe the sound likely came from (within 25 deg of its bearing, distance class fitting)."""
        best, bs = None, None
        for f in self.foes.values():
            if not f.ghost and self.t - f.t_seen < 0.2:
                continue  # in view now: we would not need the sound
            fp = self.where(f)
            fd = math.hypot(fp[0] - pos[0], fp[1] - pos[1])
            da = abs(ang_diff(bearing(pos, fp), b))
            if da < 0.45 and 0.4 * d < fd + 20 and fd < 2.5 * d + 30:
                s = da + abs(fd - d) / 200
                if bs is None or s < bs:
                    best, bs = f, s
        return best

    def _absorb_ghost(self, p):
        """A mech came into view: drop the ghost that was standing in for it."""
        for k, f in list(self.foes.items()):
            if k < 0 and math.hypot(f.pos[0] - p[0], f.pos[1] - p[1]) < 60:
                del self.foes[k]
                return

    def where(self, f):
        """Best guess of f's position now: seen now -> there; lost -> carried along its velocity a moment."""
        age = self.t - f.t_seen
        if f.ghost or age <= 0.05:
            return f.pos
        c = min(age, CARRY_S)
        return [f.pos[0] + f.vel[0] * c, f.pos[1] + f.vel[1] * c, f.pos[2]]

    def spread(self, f):
        """Uncertainty radius of where(f), in m."""
        if f.ghost:
            return 40.0
        return min(5.0 + 6.0 * max(self.t - f.t_seen - CARRY_S, 0.0), 80.0)

    def known(self, max_age=FOE_S):
        return [f for f in self.foes.values() if (self.t - (f.t_seen if not f.ghost else max(f.t_heard, f.t_hit_us))) < max_age]

    # ----- damage rates -----
    def weapon_dps(self, kind, d, cubes=0, hit_p=None):
        mult = 1 + self.dpc * min(cubes or 0, self.cap)
        if kind == "gun":
            return self.gun_dps * (hit_p if hit_p is not None else gun_hit_p(d)) * mult if d <= self.range["gun"] + 10 else 0.0
        if kind == "laser":
            return self.laser_dps * 0.92 * mult if d <= self.range["laser"] + 5 else 0.0
        if kind == "lightning":
            return self.lightning_dps * mult if d <= self.range["lightning"] + 3 else 0.0
        if kind == "mortar":
            return self.mortar_dps * mult if 20 <= d <= self.range["mortar"] else 0.0
        return 0.0

    def foe_dps(self, f, d):
        """What f deals to a target d m off with every weapon it holds (a mech with a laser and a gun uses both)."""
        return sum(self.weapon_dps(k, d, f.cubes) for k in set(f.weapons or ["gun"]))

    def awareness(self, f, pos, los):
        """0..1: how much of its fire we should expect. Shooting us -> 1; aimed our way and in sight -> 0.85 (1 charging);
        fighting someone else -> 0.2; unaware but in sight -> 0.35 (our shots are heard 300 m off); out of sight -> 0.15."""
        t = self.t
        if t - f.t_hit_us < 4:
            return 1.0
        if not f.ghost and t - f.t_seen < 1.5:
            fp = f.pos
            to_us = bearing(fp, pos)
            aimed = abs(ang_diff(f.aim[0], to_us)) < 0.45
            d = math.hypot(pos[0] - fp[0], pos[1] - fp[1])
            if aimed and los and d < self.vision + 20:
                return 0.85 if not f.charging else 1.0
            if t - f.busy_t < 3:
                return 0.2
            return 0.35 if los else 0.15
        return 0.2 if t - f.t_loud < 4 else 0.15

    # ----- the trade -----
    @staticmethod
    def engagement(our_hp, foes):
        """foes: [(dps on us, our dps on it, its hp, must_kill)]. Returns (damage we take, seconds) fighting it out:
        the must_kill ones we can hurt are killed one by one (Smith's rule order: highest dps x our dps / hp first)
        while every one still standing shoots; the rest (not in the fight yet, or out of our reach) shoot for at
        most SIDE_S s (they join late, or we step out of their line)."""
        fight = [f for f in foes if f[3] and f[1] > 1e-6]
        side = sum(f[0] for f in foes if not (f[3] and f[1] > 1e-6))
        order = sorted(fight, key=lambda f: -f[0] * f[1] / max(f[2], 1.0))
        live = sum(f[0] for f in order)
        taken, tt = 0.0, 0.0
        for dps, ours, hp, _ in order:
            k = hp / ours
            taken += live * k
            tt += k
            live -= dps
        return taken + side * min(max(tt, 2.0), SIDE_S), tt

    def assess(self, me, pos, aim_yaw, our_weapons_dps, los_fn, extra=None):
        """Risk of fighting it out where we stand: (risk = damage taken / our effective HP, details per foe).
        our_weapons_dps(foe, d) -> our dps on it; los_fn(foe_pos) -> True if it has a line of fire on us."""
        hp_eff = me["hp"] / max(1.0 - me.get("stats", {}).get("armor", 0.0), 0.2)
        rows, info = [], []
        for f in self.known():
            fp = self.where(f)
            d = math.hypot(fp[0] - pos[0], fp[1] - pos[1])
            if d > 320:
                continue
            los = los_fn(f, fp) if (not f.ghost and self.t - f.t_seen < 1.0) else False
            aw = self.awareness(f, pos, los)
            dps = self.foe_dps(f, d) * aw
            if aw >= 0.8 and abs(ang_diff(bearing(pos, fp), aim_yaw)) > math.pi / 2:
                dps *= self.rear_mult  # it would hit our back while we face the one we fight
            if not los and not f.ghost and self.t - f.t_hit_us > 2:
                dps *= 0.5  # behind cover now; it has to come round
            ours = our_weapons_dps(f, d) if los else 0.0
            must = aw >= 0.8 and d < 260
            rows.append((dps, ours, max(f.hp, 1.0), must))
            info.append((f, d, dps, ours, aw, los))
        if extra:
            rows.extend(extra)
        if not rows or not any(r[3] for r in rows):
            return 0.0, info, 0.0  # nobody fights us: no trade to lose (the ones around only matter once it starts)
        taken, tt = self.engagement(hp_eff, rows)
        return taken / max(hp_eff, 1.0), info, tt

    def danger_at(self, p, los_fn=None, skip=None):
        """Threat score of standing at p: sum over remembered foes of their dps on p, more when close."""
        s = 0.0
        for f in self.known():
            if skip is not None and f.id == skip:
                continue
            fp = self.where(f)
            d = math.hypot(fp[0] - p[0], fp[1] - p[1]) - self.spread(f) * 0.3
            d = max(d, 5.0)
            if d > 300:
                continue
            w = self.foe_dps(f, d) * (1.0 if not f.ghost else 0.6)
            age = self.t - (f.t_seen if not f.ghost else max(f.t_heard, f.t_hit_us))
            w *= max(0.25, 1.0 - age / FOE_S)
            if los_fn is not None and not f.ghost and age < 3:
                w *= 1.0 if los_fn(f, fp, p) else 0.2
            s += w * (1.0 + max(0.0, 120.0 - d) / 60.0)
        return s
