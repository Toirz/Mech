"""Decision (GUIDELINES §3.3 and §5). Each decision:

1. the target, by the rules of §5.5: whoever is shooting us; then whoever is busy with someone else or low; then
   a first laser shot at one not looking at us if we would win; anyone else is only faced. Drones shooting us get
   the gun back, whatever our HP;
2. a few modes scored by utility, with hysteresis (fight, zone, arm, farm, drop, hold): each one says where the legs go;
3. the plan the execution layer carries out (legs goal, strafe, torso target or look direction, boost).
"""
import collections
import math

from gunnery import Target
from memory import wrap, yaw_to

# §5.2: each entry is one more stack of that upgrade; then the rest; never a mortar upgrade
UPGRADE_ORDER = ["plating", "laser_damage", "laser_damage", "plating", "laser_charge", "laser_damage", "plating", "nanites"]
UPGRADE_REST = ["armor", "laser_charge", "nanites", "gun_damage", "gun_rate", "sensors", "gyros", "servos", "armor",
                "gun_mag", "gun_velocity", "capacitor", "reactor", "jets", "armor", "lightning_efficiency",
                "lightning_range", "lightning_chain", "laser_pierce"]

HYST = 0.1             # a new mode must beat the current one by this much
HYST_ZONE = 0.3        # more for leaving the zone mode: its score falls as we walk in, so it would flap
STICK = 0.7            # a new target in the same class must score under this share of the current one's
SPIN_S = 1.6           # s of the 360-degree look at spawn (turn_rate 4 rad/s)
EARLY_S = 60.0         # first minute: stay around the spawn, fetch a laser from far, farm
HP_S, LATE_S = 195.0, 325.0
EARLY_LASER_M, LASER_M, LIGHTNING_M = 250.0, 120.0, 100.0
GUN_OPEN_M = 100.0     # the gun opens a fight only this close
FIGHTING_S = 5.0       # s since we hurt it or it hurt us: already in a fight, the gun fires at any range
FARM_SHOOT_M = 60.0    # the gun on crates and calm drones only this close (gunfire carries 320 m)
CUBE_M = 80.0
FARM_CLEAR_M = 70.0    # no mech or ghost this close while farming
CUBE_CAP = 10
ROOF_M = 4.0           # loot this far above our feet is on a roof: left alone
DROP_HP, DROP_CALM_S, DROP_CLEAR_M, DROP_FAR_M = 0.7, 5.0, 150.0, 250.0
EDGE_MARGIN = 15.0     # m inside the next circle's edge
SPOT_EVERY = 4.0       # s between two choices of the standing spot
SPOT_KEEP = 2.0        # a new spot must score this much better to replace the old one
UNSEEN_S = 1.0         # s we face the bearing of a hit from someone we can't see
SWEEP, SWEEP_S = 0.9, 6.0  # rad either side of inward while holding, and the period
KNOWN_S = 10.0


class Plan:
    __slots__ = ("mode", "goal", "climb", "arrive", "tgt", "rule", "face", "strafe", "boost", "wiggle", "note")

    def __init__(self):
        self.mode, self.goal, self.climb, self.arrive, self.tgt, self.rule = "hold", None, False, 3.0, None, None
        self.face, self.strafe, self.boost, self.wiggle, self.note = None, None, False, None, ""


class Brain:
    def __init__(self, cfg, world, mem, zone, assess):
        self.cfg, self.w, self.mem, self.zone, self.a = cfg, world, mem, zone, assess
        self.walk = cfg["mech"]["walk_speed"]
        self.npc_body = (cfg["npc"].get("radius", 1.8), cfg["npc"].get("height", 3.5))
        self.crate_body = (cfg["crates"].get("radius", 1.4), cfg["crates"].get("height", 2.4))
        self.mode, self.tgt_id = "hold", None
        self.spawn = None
        self.spot, self.spot_climb, self.spot_score, self.spot_t, self.spot_key = None, False, 1e9, -1e9, None
        self._zkey, self._zspot = None, None

    # ---------- the decision ----------

    def decide(self, obs, me, t, clock=None):
        x, y, z = me["pos"]
        if self.spawn is None:
            self.spawn = (x, y)
        info = self.a.update(me, t)
        p = Plan()
        p.tgt, p.rule = self.target(me, t, info)
        hold, hclimb = self.hold_spot(me, t, clock)
        zgoal, zclimb, uz = self.zone_need(me, t, hold, hclimb)
        arm = self.arm_target(me, t)
        drop = self.drop_target(obs, me, t)
        farm = self.farm_target(me, t, shoot=p.tgt is None)
        u = {"hold": 0.3, "zone": 1.2 if uz >= 0.95 else uz, "fight": {1: 0.9, 2: 0.7, 3: 0.7, 5: 0.6}.get(p.rule, 0.0),
             "arm": arm[1] if arm else 0.0, "drop": 0.65 if drop else 0.0, "farm": 0.5 if farm else 0.0}
        # (an edge about to catch us beats even a fight: we keep shooting and strafing on the way in)
        u[self.mode] += HYST_ZONE if self.mode == "zone" else HYST
        mode = max(u, key=u.get)
        self.mode = p.mode = mode
        fighting = p.tgt is not None and bool(p.tgt.allow) or p.rule == 1
        if mode == "zone":
            p.goal, p.climb = zgoal, zclimb
            # boost back in from the second phase on (35 HP/s and up), or when the edge would catch us walking
            p.boost = (not me["in_zone"] and obs["zone"]["dps"] >= 35) or (me["in_zone"] and uz >= 0.95 and not fighting)
        elif mode == "arm":
            p.goal, p.arrive = arm[0], 1.0
        elif mode == "drop":
            c = drop["pos"]
            p.goal, p.arrive = (c[0], c[1]), 2.0
            if math.hypot(c[0] - x, c[1] - y) < drop["radius"] - 1.5:
                p.wiggle = (c[0], c[1])  # in the ring: keep moving, small
        elif mode == "farm":
            kind, pos, ref = farm
            d = math.hypot(pos[0] - x, pos[1] - y)
            if kind == "cube":
                p.goal, p.arrive = (pos[0], pos[1]), 0.8
            else:
                if d > FARM_SHOOT_M - 10:
                    p.goal, p.arrive = (pos[0], pos[1]), FARM_SHOOT_M - 15
                if d <= FARM_SHOOT_M and p.tgt is None:
                    r, h = self.crate_body if kind == "crate" else self.npc_body
                    p.tgt = Target(pos, r, h, ("gun",), ref if kind == "npc" else None)
        elif mode == "hold":
            p.goal, p.climb = hold, hclimb
        # fight: legs stay where they are; in every mode we strafe while fighting (§3.4)
        if fighting and p.tgt is not None:
            p.strafe = yaw_to(me["pos"], p.tgt.pos)
        if p.tgt is None:
            p.face = self.look(me, t, p)
        p.note = f"r{p.rule or '-'} u={'/'.join(f'{k[:2]}{v:.2f}' for k, v in u.items() if v > 0)}"
        return p

    # ---------- §5.5 targets ----------

    def target(self, me, t, info):
        late = t >= LATE_S
        best, key = None, None
        for mid, i in info.items():
            if i.d > 300:
                continue
            exchange = t - i.tr.shot_me < FIGHTING_S and t - i.tr.hit_by_me < FIGHTING_S  # a fight still going on
            if i.shooting or exchange:
                rule = 1
            elif i.low or (i.busy and not late):
                rule = 2
            elif not late and not i.looking and self.a.win(me, i):
                rule = 3
            else:
                rule = 4
            s = i.d / 100 + 1.5 * i.tr.hp / max(i.tr.max_hp, 1) + (0.0 if i.los else 1.0) - (0.8 if i.back else 0.0)
            if mid == self.tgt_id:
                s = s * STICK - 0.1
            k = (rule, s)
            if key is None or k < key:
                best, key = i, k
        if best is None or key[0] == 4:  # rule 5 beats a mech we would only face: a drone shooting us gets the gun
            shot_by = self.mem.shot_by(t, 3.0)
            for mid in self.mem.listed:
                tr = self.mem.tracks[mid]
                if tr.kind == "npc" and mid in shot_by and tr.hp > 0:
                    self.tgt_id = None
                    return Target(tr.pos, self.npc_body[0], self.npc_body[1], ("gun",), tr), 5
        if best is not None:
            self.tgt_id = best.tr.id
            rule, tr = key[0], best.tr
            fighting = t - tr.hit_by_me < FIGHTING_S or t - tr.shot_me < FIGHTING_S
            if rule == 1:
                allow = ("laser", "gun", "lightning")
            elif rule in (2, 3):
                allow = ("laser", "lightning") + (("gun",) if best.d <= GUN_OPEN_M or fighting else ())
            else:
                allow = ()
            return Target(tr.pos, 2.5 * tr.size, 9.0 * tr.size, allow, tr), rule
        self.tgt_id = None
        return None, None

    # ---------- where to stand (§5.4) ----------

    def hold_spot(self, me, t, clock=None):
        """First minute: the spawn, if it is inside the next circle. Otherwise just inside the next circle's edge,
        the spot nearest us that is not too exposed (walk time plus danger), kept unless a new one is clearly better."""
        cx, cy, r = self.zone.next_circle()
        if t < EARLY_S and math.hypot(self.spawn[0] - cx, self.spawn[1] - cy) <= r - min(EDGE_MARGIN, r / 2):
            return self.spawn, False
        key = (round(cx), round(cy), round(r))
        if self.spot is not None and (key == self.spot_key and t - self.spot_t < SPOT_EVERY
                                      or clock is not None and clock.left() < 0.015):  # no time now: next decision
            return self.spot, self.spot_climb
        x, y = me["pos"][0], me["pos"][1]
        rr = max(r - min(EDGE_MARGIN, r / 2), 0.0)
        if self.w is None or rr < 8:  # a small circle (or no map): the nearest point we can reach inside it
            d = max(math.hypot(x - cx, y - cy), 1e-6)
            spot, climb = (cx + (x - cx) * min(rr, d) / d, cy + (y - cy) * min(rr, d) / d), False
            if self.w is not None:
                found = self.w.spot_in((x, y), cx, cy, max(rr, 0.7 * r), self.w.area(x, y))
                spot, climb = (found, False) if found else ((cx, cy), True)  # all of it on a roof: climb
            self.spot, self.spot_climb, self.spot_score, self.spot_t, self.spot_key = spot, climb, 0.0, t, key
            return spot, climb
        base = math.atan2(y - cy, x - cx)
        step = 12.0 / rr
        area = self.w.area(x, y)
        best, bs = None, 1e18
        for k in (0, 1, -1, 2, -2, 3, -3):
            if best is not None and clock is not None and clock.left() < 0.008:
                break  # each candidate costs sight lines and a cover look: keep the best so far
            a = base + k * step
            C = self.w.snap(cx + rr * math.cos(a), cy + rr * math.sin(a), 4, comp=area)
            if C < 0 and area != self.w.main:
                C = self.w.snap(cx + rr * math.cos(a), cy + rr * math.sin(a), 4, comp=self.w.main)
            if C < 0:
                continue
            q = self.w.center(C)
            if math.hypot(q[0] - cx, q[1] - cy) > r - min(EDGE_MARGIN, r / 2) / 2:
                continue
            s = math.hypot(q[0] - x, q[1] - y) / self.walk + 6.0 * self.a.danger(q, t)
            if s < bs:
                best, bs = q, s
        climb = False
        if best is None:
            found = self.w.spot_in((x, y), cx, cy, rr, area)
            best, climb, bs = (found, False, 0.0) if found else ((cx, cy), True, 0.0)
        old_ok = (self.spot is not None and key == self.spot_key and math.hypot(self.spot[0] - cx, self.spot[1] - cy) <= r)
        if old_ok:
            old = math.hypot(self.spot[0] - x, self.spot[1] - y) / self.walk + 6.0 * self.a.danger(self.spot, t)
            if old <= bs + SPOT_KEEP:
                best, climb, bs = self.spot, self.spot_climb, old
        self.spot, self.spot_climb, self.spot_score, self.spot_t, self.spot_key = best, climb, bs, t, key
        return best, climb

    def zone_need(self, me, t, hold, hold_climb):
        """(goal, climb, utility): where to go when the zone presses, and how hard it presses (A*-free: straight
        distance x 1.4 at walk speed plus 10 s against the time the moving edge needs to reach us)."""
        x, y = me["pos"][0], me["pos"][1]
        cx, cy, r = self.zone.next_circle()
        if hold is not None and math.hypot(hold[0] - cx, hold[1] - cy) <= r - 2:
            goal, climb = hold, hold_climb
        else:
            goal, climb = self.zone_spot(me, t, (cx, cy, r))
        if not me["in_zone"]:
            return goal, climb, 1.0
        left = self.zone.time_left((x, y), t, 2.0)
        if left >= 1e8:
            return goal, climb, 0.0
        need = 1.4 * math.hypot(goal[0] - x, goal[1] - y) / self.walk + 10.0
        return goal, climb, 0.95 if left < need else max(0.0, 0.9 - (left - need) / 40.0)

    def zone_spot(self, me, t, circle):
        cx, cy, r = circle
        key = (round(cx), round(cy), round(r), int(t))
        if self._zkey != key:
            self._zkey = key
            inner = max(r - min(EDGE_MARGIN, r / 2), 0.7 * r)
            x, y = me["pos"][0], me["pos"][1]
            spot = self.w.spot_in((x, y), cx, cy, inner, self.w.area(x, y)) if self.w else (cx, cy)
            self._zspot = (spot, False) if spot else ((cx, cy), True)
        return self._zspot

    # ---------- loot (§5.1, §5.2, §5.6) ----------

    def known_mechs(self, t):
        return [tr for tr in self.mem.tracks.values() if tr.kind == "mech" and tr.hp > 0 and t - tr.t_seen < KNOWN_S]

    def ok_loot(self, me, pos, t):
        if pos[2] > me["pos"][2] + ROOF_M:
            return False
        if self.w is not None and not self.w.reachable(me["pos"], pos):
            return False
        d = math.hypot(pos[0] - me["pos"][0], pos[1] - me["pos"][1])
        # there by then, and still 25 s before the edge gets there: no running in and out for it
        return self.zone.inside(pos, t + d / self.walk + 5.0, 5.0) and self.zone.time_left(pos, t, 2.0) > d / self.walk + 25.0

    def arm_target(self, me, t):
        """(pos, utility, id) of the weapon to fetch: a laser while we have none, a lightning while we have nothing."""
        kinds = {w["kind"] for w in me["weapons"] if w}
        if "laser" in kinds:
            return None
        want = ("laser", "lightning") if "lightning" not in kinds and None in me["weapons"] else ("laser",)
        x, y = me["pos"][0], me["pos"][1]
        mechs = self.known_mechs(t)
        best, bk = None, None
        for pid, (kind, pos, _, _) in self.mem.pickups.items():
            if kind not in want:
                continue
            d = math.hypot(pos[0] - x, pos[1] - y)
            lim = (EARLY_LASER_M if t < EARLY_S else LASER_M) if kind == "laser" else LIGHTNING_M
            if d > lim or not self.ok_loot(me, pos, t):
                continue
            if any(math.hypot(tr.pos[0] - pos[0], tr.pos[1] - pos[1]) < d for tr in mechs):
                continue  # someone else is nearer to it
            k = (kind != "laser", d)
            if bk is None or k < bk:
                best, bk = ((pos[0], pos[1]), 0.75 if kind == "laser" else 0.55, pid), k
        return best

    def calm(self, me, t, p, clear_m):
        """No known mech or guess within clear_m of p."""
        return (not any(math.hypot(tr.pos[0] - p[0], tr.pos[1] - p[1]) < clear_m for tr in self.known_mechs(t))
                and not self.mem.ghosts_near(p, clear_m, t))

    def farm_target(self, me, t, shoot):
        """(kind, pos, ref) to farm: a cube to walk over, or (with `shoot`) a crate or a calm drone for the gun."""
        if me["cubes"] >= CUBE_CAP or t >= LATE_S or (t >= HP_S and me["hp"] < 0.7 * me["max_hp"]):
            return None
        x, y, z = me["pos"]
        if not self.calm(me, t, (x, y), FARM_CLEAR_M):
            return None
        best, bd = None, 1e18
        for cid, (pos, _, _) in self.mem.cubes.items():
            d = math.hypot(pos[0] - x, pos[1] - y)
            if d < CUBE_M and d < bd and self.ok_loot(me, pos, t):
                best, bd = ("cube", pos, cid), d
        if shoot and any(w and w["kind"] == "gun" for w in me["weapons"]):
            for cid, (pos, hp, _, _) in self.mem.crates.items():
                d = math.hypot(pos[0] - x, pos[1] - y) + 20
                if d < FARM_SHOOT_M + 50 and d < bd and self.ok_loot(me, pos, t):
                    best, bd = ("crate", pos, cid), d
            if me["hp"] >= 0.5 * me["max_hp"]:
                for mid in self.mem.listed:
                    tr = self.mem.tracks[mid]
                    if tr.kind != "npc" or tr.hp <= 0:
                        continue
                    d = math.hypot(tr.pos[0] - x, tr.pos[1] - y) + 30
                    if d < FARM_SHOOT_M + 60 and d < bd and self.ok_loot(me, tr.pos, t):
                        best, bd = ("npc", tr.pos, tr), d
        return best

    def drop_target(self, obs, me, t):
        """A supply drop worth walking to (§5.6): we are healthy and calm, nobody known near it."""
        if me["hp"] < DROP_HP * me["max_hp"]:
            return None
        if any(d[1] is not None and t - d[0] < DROP_CALM_S for d in self.mem.damage):
            return None
        x, y = me["pos"][0], me["pos"][1]
        for d in obs.get("drops") or []:
            if d.get("holder") not in (None, me["id"]):
                continue
            pos = d["pos"]
            dist = math.hypot(pos[0] - x, pos[1] - y)
            if dist > DROP_FAR_M or d["lands_in"] > dist / self.walk + 15.0:
                continue
            if not self.ok_loot(me, pos, t) or not self.zone.inside(pos, t + dist / self.walk + d["capture_s"] + 5.0, d["radius"]):
                continue
            if self.calm(me, t, pos, DROP_CLEAR_M):
                return d
        return None

    # ---------- torso without a target ----------

    def look(self, me, t, p):
        if t < SPIN_S:
            return [me["aim"][0] + 1.5, 0.0]
        hit = self.mem.unseen_hit
        if hit and t - hit[0] < UNSEEN_S:
            return [hit[1], 0.0]
        x, y = me["pos"][0], me["pos"][1]
        for d in reversed(self.mem.damage):  # someone hurt us lately: face where it was
            if t - d[0] > 3.0:
                break
            tr = self.mem.tracks.get(d[1]) if d[1] is not None else None
            if tr is not None:
                return [yaw_to((x, y), tr.pos), 0.0]
            if d[2] is not None:
                return [d[2], 0.0]
        near = [g for g in self.mem.ghosts if t - g.t < 6.0 and math.hypot(g.x - x, g.y - y) < 150]
        if near:
            g = max(near, key=lambda g: g.t)
            return [yaw_to((x, y), (g.x, g.y)), 0.0]
        if p.goal is not None and math.hypot(p.goal[0] - x, p.goal[1] - y) > 20:
            return [yaw_to((x, y), p.goal), 0.0]
        cx, cy, _ = self.zone.next_circle()
        inward = yaw_to((x, y), (cx, cy)) if math.hypot(cx - x, cy - y) > 5 else me["aim"][0]
        return [inward + SWEEP * math.sin(2 * math.pi * t / SWEEP_S), 0.0]

    # ---------- upgrades (§5.2) ----------

    @staticmethod
    def pick_upgrade(offer, me):
        have = collections.Counter(me.get("upgrades") or [])

        def rank(cid):
            if cid.startswith("mortar"):
                return 10000
            seen = collections.Counter()
            for k, u in enumerate(UPGRADE_ORDER):
                seen[u] += 1
                if u == cid and seen[u] > have[cid]:
                    return k
            return 100 + UPGRADE_REST.index(cid) if cid in UPGRADE_REST else 1000

        ch = offer["choices"]
        return min(range(len(ch)), key=lambda i: rank(ch[i]["id"]))
