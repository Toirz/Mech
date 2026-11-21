"""Legs (GUIDELINES §3.4): follow an A* path, get unstuck, strafe and dodge, never stepping somewhere unsafe."""
import math
import random

from world import PathJob

REPLAN_MOVE = 6.0     # m the goal may move before we plan again
REPLAN_S = 4.0        # s a finished path is trusted
WAYPOINT_R = 2.0      # m: a waypoint this close is passed
LOOKAHEAD = 8         # waypoints tried for a straight walk (corner cutting)
STUCK_SPEED, STUCK_S = 1.5, 0.6   # stuck: slower than this for this long while trying to walk
JET_S = 0.6           # s of jet after the hop
SIDESTEP_S = 1.0      # s of walking sideways after three hops that did not help
STRAFE_S = (0.35, 0.9)  # s between strafe reversals


def unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)


class Legs:
    def __init__(self, world, seed):
        self.w, self.rng = world, random.Random(seed)
        self.job, self.path, self.goal, self.t_plan = None, [], None, -1e9
        self.slow_since, self.jet_until, self.hops, self.side_until, self.side = None, -1e9, [], -1e9, (0.0, 0.0)
        self.sdir, self.s_until = 1, 0.0
        self.climb_to, self.climbing = None, False

    # ---------- walking ----------

    def toward(self, me, goal, t, deadline, arrive=3.0, climb=False):
        """Move vector toward goal (x, y) along the nav path; (0, 0) within `arrive` m. With `climb` (no walkable
        way, e.g. the zone closes on a roof) it heads straight there and unstick() jets up the wall in the way."""
        x, y = me["pos"][0], me["pos"][1]
        self.climb_to = self.w.floor_at(goal[0], goal[1]) if climb and self.w is not None else None
        if math.hypot(goal[0] - x, goal[1] - y) < arrive:
            self.path, self.job = [], None
            return 0.0, 0.0
        if self.w is None or climb:
            return unit(goal[0] - x, goal[1] - y)
        moved = self.goal is None or math.hypot(goal[0] - self.goal[0], goal[1] - self.goal[1]) > REPLAN_MOVE
        if moved or (self.job is None and (not self.path or t - self.t_plan > REPLAN_S)):
            self.job, self.goal, self.t_plan = PathJob(self.w, (x, y), goal), (goal[0], goal[1]), t
        if self.job is not None and self.job.run(deadline):
            self.path, self.job, self.t_plan = list(self.job.path or []), None, t
        while self.path and math.hypot(self.path[0][0] - x, self.path[0][1] - y) < WAYPOINT_R:
            self.path.pop(0)
        if not self.path:
            return unit(goal[0] - x, goal[1] - y)  # no route (yet): straight, the stuck routine copes
        tx, ty = self.path[0]
        for k in range(min(LOOKAHEAD, len(self.path)) - 1, 0, -1):
            if self.w.walkable_line((x, y), self.path[k]):
                tx, ty = self.path[k]
                break
        return unit(tx - x, ty - y)

    def drop_path(self):
        self.path, self.job, self.goal = [], None, None

    def unstick(self, me, move, t):
        """(move, jump, boost): slower than STUCK_SPEED for STUCK_S while walking = stuck: hop and jet for JET_S;
        after three hops in 6 s, walk sideways for SIDESTEP_S and plan again."""
        speed = math.hypot(me["vel"][0], me["vel"][1])
        trying = math.hypot(move[0], move[1]) > 0.5
        z = me["pos"][2]
        if self.climb_to is not None and not (me["grounded"] and z >= self.climb_to - 1.0):
            # climbing on purpose: hop at the wall, then jet while the wall still holds us or we are below the
            # goal's floor; the wall may stand taller than the floor behind it (100 energy is ~60 m of climb)
            if trying and speed < STUCK_SPEED and me["grounded"]:
                self.climbing = True
            if not self.climbing:
                return move, False, False
            if me["grounded"]:
                return move, True, False
            return move, False, me["energy"] > 2 and (speed < 3.0 or z < self.climb_to + 1.0)
        self.climbing = False
        if trying and speed < STUCK_SPEED and me["grounded"]:
            if self.slow_since is None:
                self.slow_since = t
        else:
            self.slow_since = None
        jump = boost = False
        if self.slow_since is not None and t - self.slow_since >= STUCK_S:
            jump, self.jet_until, self.slow_since = True, t + JET_S, None
            self.hops = [h for h in self.hops if t - h < 6.0] + [t]
            if len(self.hops) >= 3:
                s = self.rng.choice((-1, 1))
                self.side, self.side_until, self.hops = (-move[1] * s, move[0] * s), t + SIDESTEP_S, []
                self.drop_path()
        if t < self.side_until:
            move = self.side
        if t < self.jet_until and not me["grounded"]:
            boost = True
        return move, jump, boost

    # ---------- fighting footwork ----------

    def strafe(self, me, yaw_to_threat, t, safe):
        """Sideways across the line to the threat, reversing every 0.35-0.9 s, only toward safe ground."""
        if t >= self.s_until:
            self.sdir, self.s_until = -self.sdir, t + self.rng.uniform(*STRAFE_S)
        x, y = me["pos"][0], me["pos"][1]
        for s in (self.sdir, -self.sdir):
            a = yaw_to_threat + s * math.pi / 2
            d = (math.cos(a), math.sin(a))
            if safe((x + 4 * d[0], y + 4 * d[1])):
                self.sdir = s
                return d
        return 0.0, 0.0

    def dodge(self, me, projectiles, gravity, safe):
        """A step out of the way of a visible bullet headed for our body, or away from where a shell lands; else None."""
        x, y, z = me["pos"]
        st = me["stats"]
        r = st.get("radius", 2.5) + 1.0
        cz = z + st.get("height", 9.0) / 2
        best, soonest = None, 1e9
        for p in projectiles:
            (px, py, pz), (vx, vy, vz) = p["pos"], p["vel"]
            if p["kind"] in ("gun", "npc"):
                v2 = vx * vx + vy * vy + vz * vz
                if v2 < 1:
                    continue
                rx, ry, rz = x - px, y - py, cz - pz
                tc = (rx * vx + ry * vy + rz * vz) / v2
                if not 0 < tc < 0.6:
                    continue
                mx, my = rx - vx * tc, ry - vy * tc
                if math.hypot(mx, my) > r or tc > soonest:
                    continue
                side = unit(-vy, vx)
                if side[0] * mx + side[1] * my < 0:  # keep going the way we are already off its line
                    side = (-side[0], -side[1])
                for d in (side, (-side[0], -side[1])):
                    if safe((x + 4 * d[0], y + 4 * d[1])):
                        best, soonest = d, tc
                        break
            elif p["kind"] == "mortar":
                a, b, c = -0.5 * gravity, vz, pz - z  # height above our feet: c + b t + a t^2 = 0
                disc = b * b - 4 * a * c
                if disc < 0:
                    continue
                tl = (-b - math.sqrt(disc)) / (2 * a)
                if not 0 < tl < 3.0:
                    continue
                lx, ly = px + vx * tl, py + vy * tl
                if math.hypot(lx - x, ly - y) > 9.0 or tl > soonest:
                    continue
                d = unit(x - lx, y - ly)
                if d == (0.0, 0.0):
                    d = unit(-vy, vx)
                if safe((x + 4 * d[0], y + 4 * d[1])):
                    best, soonest = d, tl
        return best
