"""Assessment (GUIDELINES §3.2): who shoots whom, a duel estimate per enemy, and how dangerous a spot is.

Duel estimate: the time we need to kill it against the time everyone shooting us needs to kill us, from HP, each
weapon's damage per second at that range (table in §5.1), cubes, and backs (x1.5). We call it a win when ours is
under WIN_MARGIN of theirs: the slack is for a third mech joining in.
"""
import math

from memory import wrap, yaw_to

WIN_MARGIN = 0.75
MY_GUN_HIT = ((0, 1.0), (40, 1.0), (80, 0.56), (120, 0.40), (180, 0.14), (230, 0.0))  # ours on a strafer
THEIR_GUN_HIT = ((0, 1.0), (150, 0.6), (230, 0.3))  # the built-in bots: ~60% at 150 m in the recorded match
SHOT_RECENT = 3.0      # s: "is shooting us" if it hurt us this recently
BUSY_S = 2.0           # s: "busy with someone else" if it fired at someone else, or lost HP to someone else, this recently
LOOKING = 0.3          # rad: its aim within this of us = it is looking at us
KNOWN_S = 10.0         # s a track still counts as a known position


def interp(table, d):
    if d <= table[0][0]:
        return table[0][1]
    for (d0, v0), (d1, v1) in zip(table, table[1:]):
        if d <= d1:
            return v0 + (v1 - v0) * (d - d0) / (d1 - d0)
    return table[-1][1]


class Info:
    """What one listed mech is doing, as seen from us this decision."""
    __slots__ = ("tr", "d", "los", "shooting", "busy", "low", "looking", "back")


class Assess:
    def __init__(self, cfg, world, mem):
        self.cfg, self.w, self.mem = cfg, world, mem
        self.cube_dmg = cfg["cubes"]["damage_per_cube"]
        self.cap = cfg["cubes"]["stat_cap"]
        self.rear = cfg["mech"].get("rear_mult", 1.5)
        self.info = {}

    def update(self, me, t):
        x, y, z = me["pos"]
        mz = (x, y, z + me["stats"].get("muzzle_height", 7.0))
        shot_by = self.mem.shot_by(t, SHOT_RECENT)
        self.info = {}
        for mid in self.mem.listed:
            tr = self.mem.tracks[mid]
            if tr.kind != "mech" or tr.hp <= 0:
                continue
            i = Info()
            i.tr = tr
            i.d = math.dist(mz, tr.pos)
            i.los = self.w is None or self.w.clear(mz, (tr.pos[0], tr.pos[1], tr.pos[2] + 4.5 * tr.size))
            to_me = yaw_to(tr.pos, (x, y))
            off = abs(wrap(tr.aim[0] - to_me))
            i.looking = off < LOOKING
            aiming = off < math.atan2(4.0, max(i.d, 1.0)) + 0.05
            i.shooting = mid in shot_by or (tr.charging and aiming)
            beam_elsewhere = t - tr.beam_t < BUSY_S and tr.beam_end is not None and math.hypot(tr.beam_end[0] - x, tr.beam_end[1] - y) > 15
            i.busy = beam_elsewhere or t - tr.hp_drop_t < BUSY_S
            i.low = tr.hp < 0.4 * tr.max_hp
            i.back = off > math.pi / 2  # we are outside its front arc: our hits land x1.5
            self.info[mid] = i
        return self.info

    # ---------- damage per second ----------

    def cube_mult(self, cubes):
        return 1.0 + self.cube_dmg * min(cubes, self.cap)

    def my_dps(self, me, d, holds_mortar=False, back=False):
        m = self.cube_mult(me["cubes"]) * (self.rear if back else 1.0)
        total = 0.0
        for w in me["weapons"]:
            if not w:
                continue
            s = w["stats"]
            if w["kind"] == "laser" and d <= s["range"]:
                total += s["damage"] * (1 + s.get("cookoff", 0.6) * holds_mortar) / (s["charge"] + s["cooldown"])
            elif w["kind"] == "gun" and d <= s["range"]:
                total += s["damage"] * s["mag"] / (s["mag"] * s["cooldown"] + s["reload"]) * interp(MY_GUN_HIT, d)
            elif w["kind"] == "lightning" and d <= s["range"]:
                total += s["damage"] / s["cooldown"]
        return total * m

    def their_dps(self, tr, d, me):
        W = self.cfg["weapons"]
        m = self.cube_mult(tr.cubes) * (1.0 - me["stats"].get("armor", 0.0))
        mortar = any(w and w["kind"] == "mortar" for w in me["weapons"])
        total = 0.0
        for k in tr.weapons:
            s = W.get(k)
            if s is None:  # a weapon kind this config doesn't describe
                continue
            if k == "laser" and d <= s["range"]:
                total += s["damage"] * (1 + s.get("cookoff", 0.6) * mortar) / (s["charge"] + s["cooldown"])
            elif k == "gun" and d <= s["range"]:
                total += s["damage"] * s["mag"] / (s["mag"] * s["cooldown"] + s["reload"]) * interp(THEIR_GUN_HIT, d)
            elif k == "lightning" and d <= s["range"]:
                total += s["damage"] / s["cooldown"]
            elif k == "mortar" and d <= 235:
                total += 40.0
        return total * m

    def win(self, me, i):
        """Would we win a fight with this mech, counting everyone already shooting us? If it is not looking at us
        and our laser is ready, our first beam lands before it can answer (§5.5 rule 3)."""
        mine = self.my_dps(me, i.d, "mortar" in i.tr.weapons, i.back)
        if mine <= 0:
            return False
        theirs = self.their_dps(i.tr, i.d, me) + sum(self.their_dps(o.tr, o.d, me) for o in self.info.values()
                                                     if o is not i and o.shooting)
        if theirs <= 0:
            return True
        hp = i.tr.hp
        laser = next((w for w in me["weapons"] if w and w["kind"] == "laser"), None)
        if laser is not None and not i.looking and laser["cooldown"] <= 0.3 and i.d <= laser["stats"]["range"]:
            hp -= laser["stats"]["damage"] * self.cube_mult(me["cubes"]) * (self.rear if i.back else 1.0)
        return max(hp, 0.0) / mine < WIN_MARGIN * me["hp"] / theirs

    # ---------- spots ----------

    def danger(self, p, t):
        """How exposed a standing spot is: known mechs and guesses that would see it, and how open it is."""
        fz = self.w.floor_at(p[0], p[1]) if self.w else 0.0
        body = (p[0], p[1], fz + 4.5)
        risk = 0.0
        for tr in self.mem.tracks.values():
            if tr.kind != "mech" or tr.hp <= 0 or t - tr.t_seen > KNOWN_S:
                continue
            d = math.hypot(tr.pos[0] - p[0], tr.pos[1] - p[1])
            if d < 280 and (self.w is None or self.w.clear((tr.pos[0], tr.pos[1], tr.pos[2] + 8.0), body, skip=0.0)):
                risk += 1.0
        for g in self.mem.ghosts:
            if math.hypot(g.x - p[0], g.y - p[1]) < 200:
                risk += 0.5
        if self.w is not None:
            risk += 1.0 - self.w.cover(p[0], p[1])
        return risk
