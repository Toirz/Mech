"""What we know beyond this observation (GUIDELINES §3.1).

- Tracks: every mech and drone we have seen, with its last state, a short motion history (for aiming), who it was
  shooting and when it last hurt us. The server itself keeps a mech listed for `memory_s` after it leaves view.
- Ghosts: guesses of where unseen mechs are. Made when a listed mech drops out of the list, when something we
  cannot see hurts us (`damaged.by`, `bearing`), and when we hear a mech we cannot account for. A ghost spreads at
  walk speed and is removed once we look at its spot and nobody is there.
- Loot: weapons, cubes and crates we have seen. One is forgotten only when its spot is in view and it isn't there.
"""
import math

GHOST_TTL = 20.0      # s a ghost lives without news
TRACK_KEEP = 60.0     # s a mech's record outlives its last listing
HIST_S = 3.5          # s of motion history per track
MERGE_M = 30.0        # m: a new ghost this close to an old one refreshes it
LOOK_EVERY = 0.5      # s between two looks at the same remembered item
SOUND_DIST = {"near": (35.0, 25.0), "mid": (110.0, 50.0), "far": (230.0, 90.0)}  # band: (guess, spread) m
BAND = {"near": (0.0, 70.0), "mid": (50.0, 175.0), "far": (150.0, 1e9)}
MECH_SOUNDS = {"gun", "mortar", "laser", "laser_charge", "lightning", "jet", "step", "boost", "reload"}
SHOT_DIST = {"laser": 180.0, "gun": 140.0, "mortar": 160.0, "lightning": 35.0}


def yaw_to(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Track:
    def __init__(self, a, t):
        self.id, self.kind, self.t_first = a["id"], a["kind"], t
        self.hist = []
        self.shot_me = -1e9        # last time it damaged us
        self.beam_t, self.beam_end = -1e9, None   # its last beam or arc we saw
        self.hp_drop_t = -1e9      # last time its HP fell without us hitting it: someone else is
        self.hit_by_me = -1e9      # last time we damaged it
        self.ghosted = False
        self.hp, self.max_hp = a.get("hp", 0.0), a.get("max_hp", 1400.0)
        self.update(a, t)

    def update(self, a, t):
        self.pos, self.vel, self.aim = a["pos"], a["vel"], a["aim"]
        self.max_hp = a.get("max_hp", self.max_hp)
        self.hp_prev, self.hp = self.hp, a.get("hp", self.hp)
        self.cubes = a.get("cubes", 0)
        self.weapons = a.get("weapons", [])
        self.charging = a.get("charging", False)
        self.t_seen, self.ghosted = t, False
        p, v = self.pos, self.vel
        self.hist.append((t, p[0], p[1], p[2], v[0], v[1], v[2]))
        while self.hist and self.hist[0][0] < t - HIST_S:
            self.hist.pop(0)

    @property
    def size(self):
        return 1.0 + 0.05 * self.cubes


class Ghost:
    __slots__ = ("x", "y", "r0", "t", "mid", "src")

    def __init__(self, x, y, r0, t, mid, src):
        self.x, self.y, self.r0, self.t, self.mid, self.src = x, y, r0, t, mid, src

    def radius(self, t, speed):
        return self.r0 + speed * (t - self.t)


class Memory:
    def __init__(self, cfg, world):
        self.world = world
        self.walk = cfg["mech"]["walk_speed"]
        self.vision = cfg["vision"]
        self.short_s = cfg["weapons"]["lightning"].get("short_s", 2.0)
        self.tracks = {}
        self.ghosts = []
        self.pickups = {}   # id -> [kind, pos, t_seen, t_looked]
        self.cubes = {}     # id -> [pos, t_seen, t_looked]
        self.crates = {}    # id -> [pos, hp, t_seen, t_looked]
        self.damage = []    # (t, by, bearing, weapon, amount, rear): the last 10 s
        self.hits = []      # (t, target, amount, weapon): the last 10 s
        self.kills = []     # (t, target id, pos, cubes)
        self.short_until = -1e9
        self.unseen_hit = None   # (t, bearing) of the last hit from someone not listed
        self.listed = set()

    # ---------- what we can see ----------

    def in_view(self, me, p, z_off=0.5):
        """Would a thing at p be in our sight now (range, wedge or proximity, and no building in the way)?"""
        st = me["stats"]
        eye = (me["pos"][0], me["pos"][1], me["pos"][2] + st.get("eye_height", 8.0))
        d = math.hypot(p[0] - eye[0], p[1] - eye[1])
        near = self.vision["near_range"] + st.get("near_range_bonus", 0.0)
        far = self.vision["range"] + st.get("vision_bonus", 0.0)
        if d > far:
            return False
        if d > near and abs(wrap(yaw_to(eye, p) - me["aim"][0])) > math.radians(self.vision["fov_deg"]) / 2:
            return False
        return self.world is None or self.world.clear(eye, (p[0], p[1], p[2] + z_off), skip=0.0, skip_end=0.3)

    # ---------- per observation ----------

    def update(self, obs):
        t, me = obs["t"], obs["self"]
        my = me["pos"]
        listed = set()
        hit_now = {e["target"] for e in obs["events"] if e["type"] == "hit"}
        for a in obs["mechs"]:
            tr = self.tracks.get(a["id"])
            if tr is None:
                tr = self.tracks[a["id"]] = Track(a, t)
            else:
                tr.update(a, t)
                if tr.hp < tr.hp_prev - 1 and a["id"] not in hit_now:
                    tr.hp_drop_t = t
            listed.add(a["id"])
        self.listed = listed
        for mid, tr in list(self.tracks.items()):
            if mid in listed:
                continue
            if tr.kind == "mech" and not tr.ghosted:  # out of the server's own memory: from now on a guess
                self._ghost(tr.pos[0], tr.pos[1], 5.0, tr.t_seen, mid, "lost")
                tr.ghosted = True
            if t - tr.t_seen > TRACK_KEEP:
                del self.tracks[mid]

        for e in obs["events"]:
            kind = e["type"]
            if kind == "damaged":
                by = e.get("by")
                self.damage.append((t, by, e.get("bearing"), e.get("weapon"), e.get("amount", 0.0), e.get("rear", False)))
                if by is None or e.get("bearing") is None:
                    continue
                tr = self.tracks.get(by)
                if tr is not None:
                    tr.shot_me = t
                if by not in listed and by < 100:
                    self.unseen_hit = (t, e["bearing"])
                    d = SHOT_DIST.get(e.get("weapon"), 140.0)
                    if tr is not None and t - tr.t_seen < 10:
                        d = max(20.0, math.dist(tr.pos[:2], my[:2]))
                    b = e["bearing"]
                    self._ghost(my[0] + d * math.cos(b), my[1] + d * math.sin(b), 40.0, t, by, "shot")
            elif kind == "hit":
                self.hits.append((t, e["target"], e.get("amount", 0.0), e.get("weapon")))
                tr = self.tracks.get(e["target"])
                if tr is not None:
                    tr.hit_by_me = t
            elif kind == "effect" and e.get("effect") == "short":
                self.short_until = t + self.short_s
            elif kind in ("took",) or (kind == "take_failed" and e.get("reason") == "gone"):
                self.pickups.pop(e["id"], None)
            elif kind == "dropped":
                self.pickups[e["id"]] = [e["kind"], list(my), t, t]
            elif kind == "kill":
                tr = self.tracks.get(e["target"])
                if tr is not None:
                    self.kills.append((t, tr.id, list(tr.pos), tr.cubes))
                    tr.hp = 0.0
        self.damage = [d for d in self.damage if t - d[0] < 10]
        self.hits = [h for h in self.hits if t - h[0] < 10]

        for b in obs["beams"]:
            tr = self.tracks.get(b["by"])
            if tr is not None and b.get("path"):
                tr.beam_t, tr.beam_end = t - b.get("age", 0.0), b["path"][-1]

        for s in obs["sounds"]:
            if s["kind"] in MECH_SOUNDS:
                self._sound(s, me, t)

        self._loot(obs, me, t)
        self._clear_ghosts(me, t)

    def _ghost(self, x, y, r0, t, mid, src):
        if self.world is not None:
            x, y = self.world.clamp(x, y, 2.0)
        for g in self.ghosts:
            if (mid is not None and g.mid == mid) or math.hypot(g.x - x, g.y - y) < max(MERGE_M, 0.5 * g.radius(t, self.walk)):
                if r0 <= g.radius(t, self.walk) or src == "shot":
                    g.x, g.y, g.r0, g.t, g.src = x, y, r0, t, src
                if mid is not None:
                    g.mid = mid
                return
        self.ghosts.append(Ghost(x, y, r0, t, mid, src))

    def _sound(self, s, me, t):
        my, b = me["pos"], s["bearing"]
        lo, hi = BAND[s["dist"]]
        for mid in self.listed:
            tr = self.tracks[mid]
            if tr.kind != "mech":
                continue
            d = math.hypot(tr.pos[0] - my[0], tr.pos[1] - my[1])
            if lo <= d <= hi and abs(wrap(yaw_to(my, tr.pos) - b)) < 0.3:
                return  # someone we see made it
        guess, spread = SOUND_DIST[s["dist"]]
        sigma = math.radians(7.0) * me["stats"].get("hearing", 1.0)
        self._ghost(my[0] + guess * math.cos(b), my[1] + guess * math.sin(b), spread + 2 * sigma * guess, t - s.get("age", 0.0), None, "sound")

    def _clear_ghosts(self, me, t):
        keep = []
        for g in self.ghosts:
            r = g.radius(t, self.walk)
            if t - g.t > GHOST_TTL or r > 400:
                continue
            p = (g.x, g.y, self.world.floor_at(g.x, g.y) + 4.5 if self.world else 4.5)
            if t - g.t >= 1.0 and self.in_view(me, p, 0.0):  # a fresh guess gets a second before a look can clear it
                near = any(self.tracks[m].kind == "mech" and math.hypot(self.tracks[m].pos[0] - g.x, self.tracks[m].pos[1] - g.y) < max(r, 40.0)
                           for m in self.listed)
                if not near:  # shortcut: seeing the centre clears the whole guess, however wide
                    continue
            keep.append(g)
        self.ghosts = keep

    def _loot(self, obs, me, t):
        seen = set()
        for p in obs["pickups"]:
            self.pickups[p["id"]] = [p["kind"], p["pos"], t, t]
            seen.add(p["id"])
        for c in obs["cubes"]:
            self.cubes[c["id"]] = [c["pos"], t, t]
            seen.add(c["id"])
        for c in obs["crates"]:
            self.crates[c["id"]] = [c["pos"], c.get("hp", 300.0), t, t]
            seen.add(c["id"])
        for store in (self.pickups, self.cubes, self.crates):
            for iid, rec in list(store.items()):
                if iid in seen or t - rec[-1] < LOOK_EVERY:
                    continue
                pos = rec[0] if store is not self.pickups else rec[1]
                rec[-1] = t
                if self.in_view(me, pos, 0.8):
                    del store[iid]

    # ---------- questions the other layers ask ----------

    def shot_by(self, t, within=3.0):
        """Ids that damaged us in the last `within` s."""
        return {d[1] for d in self.damage if d[1] is not None and t - d[0] <= within}

    def taken_damage(self, t, within):
        return sum(d[4] for d in self.damage if t - d[0] <= within)

    def ghosts_near(self, p, r, t):
        return [g for g in self.ghosts if math.hypot(g.x - p[0], g.y - p[1]) < r + 0.5 * g.radius(t, self.walk)]
