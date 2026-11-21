"""Bot: hug the inside edge of the zone, grab a weapon, shoot the nearest mech with lead, draw what it's thinking."""
import math
import mechfight as mf

EDGE_MARGIN = 12.0  # m to stay inside the zone edge
LOOKAHEAD_S = 4.0   # aim for where the edge will be, so a shrinking zone doesn't catch us
ARRIVED = 3.0       # m from the edge spot: stand still (silent, and HP regens)


def zone_at(zone, t):
    """Zone circle at match time t: holds, then moves linearly to next_* between shrink_start and shrink_end."""
    span = zone["shrink_end"] - zone["shrink_start"]
    f = min(max((t - zone["shrink_start"]) / span, 0.0), 1.0) if span > 0 else float(t >= zone["shrink_end"])
    (cx, cy), (nx, ny) = zone["center"], zone["next_center"]
    return [cx + (nx - cx) * f, cy + (ny - cy) * f], zone["radius"] + (zone["next_radius"] - zone["radius"]) * f


def edge_spot(pos, center, radius, margin):
    """Closest point to pos that sits `margin` inside the circle's edge."""
    dx, dy = pos[0] - center[0], pos[1] - center[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        dx, dy, d = 1.0, 0.0, 1.0
    r = max(radius - margin, 0.0)
    return [center[0] + dx / d * r, center[1] + dy / d * r]


def lead(src, tpos, tvel, speed):
    """Point to aim at so a straight shot at `speed` meets a target moving at constant tvel."""
    p = tpos
    for _ in range(4):  # fixed-point iteration; converges fast since bullets are much faster than mechs
        t = math.dist(src, p) / speed
        p = [tpos[i] + tvel[i] * t for i in range(3)]
    return p


def aim_at(src, p):
    return [math.atan2(p[1] - src[1], p[0] - src[0]), math.atan2(p[2] - src[2], math.dist(p[:2], src[:2]))]


class Bot:
    name = "example-py"

    def __init__(self, hello: mf.Hello):
        self.reach = hello["config"]["loot"]["pickup_radius"]

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        act: mf.Action = {"tick": obs["tick"]}
        if obs["offer"]:
            act["pick"] = 0
        near = [p for p in obs["pickups"] if math.dist(p["pos"][:2], (x, y)) < self.reach]
        if me["weapons"][1] is None and near and near[0]["kind"] != "gun":
            act["take"] = near[0]["id"]

        # Movement: go to the zone edge as it will be in LOOKAHEAD_S, then hold there.
        center, radius = zone_at(obs["zone"], obs["t"] + LOOKAHEAD_S)
        spot = edge_spot(me["pos"], center, radius, EDGE_MARGIN)
        dx, dy = spot[0] - x, spot[1] - y
        gap = math.hypot(dx, dy)
        move = [dx / gap, dy / gap] if gap > ARRIVED else [0.0, 0.0]
        # Back to the edge, face inward: enemies come from the zone side, and it protects our rear armor.
        inward = math.atan2(center[1] - y, center[0] - x)
        act.update(move=move, aim=[inward, 0.0], boost=not me["in_zone"] or gap > 30,
                   jump=gap > ARRIVED and obs["tick"] % 40 == 0)

        # Shooting: lead the nearest mech using our gun's bullet speed.
        enemies = sorted((m for m in obs["mechs"] if m["kind"] == "mech"), key=lambda m: math.dist(m["pos"], me["pos"]))
        if enemies:
            e = enemies[0]
            muzzle = [x, y, z + me["stats"]["muzzle_height"]]
            target = [e["pos"][0], e["pos"][1], e["pos"][2] + 4.5]  # center of mass
            gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
            aim_point = lead(muzzle, target, e["vel"], gun["stats"]["speed"]) if gun else target
            act.update(aim=aim_at(muzzle, aim_point), fire=[True, True])
            mf.draw.line(muzzle, aim_point, "#ff5040")
            mf.draw.point(aim_point, "#ffd060", 5)
        mf.draw.point([spot[0], spot[1], z], "#40c0ff", 6)
        mf.draw.label(f"hp {me['hp']:.0f} · edge {gap:.0f} m · {len(enemies)} in sight")
        return act


if __name__ == "__main__":
    mf.run(Bot)
