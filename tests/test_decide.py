import math

import pytest
from conftest import real_config

import assess as A
import decide as D
from memory import Memory
from zone import Zone

CFG = real_config()


def me(pos=(0.0, 0.0, 0.0), aim=(0.0, 0.0), hp=1400, cubes=0, weapons=None, t_upgrades=()):
    w = weapons if weapons is not None else [
        {"kind": "gun", "cooldown": 0, "ammo": 12, "reload": 0, "stats": dict(CFG["weapons"]["gun"])},
        {"kind": "laser", "cooldown": 0, "charge": 0, "stats": dict(CFG["weapons"]["laser"])}]
    return {"id": 3, "pos": list(pos), "aim": list(aim), "vel": [0, 0, 0], "hp": hp, "max_hp": 1400, "energy": 100,
            "grounded": True, "cubes": cubes, "upgrades": list(t_upgrades), "weapons": w, "in_zone": True, "stats": dict(CFG["mech"])}


def mech(i, x, y, aim_yaw=0.0, hp=1400, weapons=("gun", "laser"), charging=False):
    return {"id": i, "kind": "mech", "pos": [x, y, 0.0], "vel": [0, 0, 0], "aim": [aim_yaw, 0.0], "hp": hp, "max_hp": 1400,
            "cubes": 0, "weapons": list(weapons), "charging": charging}


def zone_obs():
    return {"center": [0, 0], "radius": 560, "next_center": [0, 0], "next_radius": 380, "shrink_start": 60, "shrink_end": 110, "dps": 20}


def obs(t, mechs=(), events=(), drops=()):
    return {"t": t, "tick": int(t * 20), "self": me(), "mechs": list(mechs), "events": list(events), "sounds": [], "beams": [],
            "pickups": [], "cubes": [], "crates": [], "projectiles": [], "zone": zone_obs(), "drops": list(drops), "offer": None}


def brain():
    mem = Memory(CFG, None)
    z = Zone()
    a = A.Assess(CFG, None, mem)
    return D.Brain(CFG, None, mem, z, a), mem, z


def step(b, mem, z, o, m=None):
    mem.update(o)
    z.update(o["zone"], o["t"])
    return b.decide(o, m or o["self"], o["t"])


def test_whoever_shoots_us_comes_first_and_gets_every_gun():
    b, mem, z = brain()
    shooter, busy = mech(7, 0, 150, aim_yaw=-math.pi / 2), mech(8, 60, 0, aim_yaw=math.pi / 2)
    hit = {"type": "damaged", "amount": 45, "weapon": "gun", "bearing": math.pi / 2, "by": 7}
    p = step(b, mem, z, obs(10.0, [shooter, busy], [hit]))
    assert p.rule == 1 and p.tgt.track.id == 7 and p.tgt.allow == {"laser", "gun", "lightning"}
    assert p.mode == "fight" and p.strafe is not None and p.goal is None  # stand and fight, strafing: never run


def test_a_far_mech_that_is_not_fighting_us_is_only_faced():
    b, mem, z = brain()
    p = step(b, mem, z, obs(10.0, [mech(7, 0, 200, aim_yaw=-math.pi / 2)]))  # looking at us, full HP, far
    assert p.rule == 4 and p.tgt.allow == set() and p.mode != "fight"


def test_a_busy_or_low_mech_gets_the_laser_and_the_gun_only_within_100_m():
    b, mem, z = brain()
    p = step(b, mem, z, obs(10.0, [mech(7, 0, 150, aim_yaw=0.0, hp=400)]))
    assert p.rule == 2 and "laser" in p.tgt.allow and "gun" not in p.tgt.allow
    b, mem, z = brain()
    p = step(b, mem, z, obs(10.0, [mech(7, 0, 80, aim_yaw=0.0, hp=400)]))
    assert p.rule == 2 and {"laser", "gun"} <= p.tgt.allow


def test_late_in_the_match_only_shooters_and_low_mechs_are_fought():
    b, mem, z = brain()
    away = mech(7, 0, 80, aim_yaw=math.pi / 2)  # its back to us: a first shot we win, rule 3 before 325 s
    p = step(b, mem, z, obs(330.0, [away]))
    assert p.rule == 4
    b, mem, z = brain()
    p = step(b, mem, z, obs(100.0, [away]))
    assert p.rule == 3 and "laser" in p.tgt.allow
    b, mem, z = brain()
    p = step(b, mem, z, obs(100.0, [mech(7, 0, 80, aim_yaw=0.0)]))  # side-on, equal: no clear win, only faced
    assert p.rule == 4


def test_a_drone_shooting_us_always_gets_the_gun_back():
    b, mem, z = brain()
    npc = {"id": 104, "kind": "npc", "pos": [30, 0, 0], "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 350, "max_hp": 350}
    hit = {"type": "damaged", "amount": 25, "weapon": "npc", "bearing": 0.0, "by": 104}
    m = me(hp=200)
    o = obs(20.0, [npc], [hit])
    o["self"] = m
    p = step(b, mem, z, o, m)
    assert p.rule == 5 and p.tgt.allow == {"gun"}
    b, mem, z = brain()
    o = obs(20.0, [npc, mech(7, 0, 200, aim_yaw=-math.pi / 2)], [hit])  # a mech we would only face is in sight too
    p = step(b, mem, z, o)
    assert p.rule == 5 and p.tgt.track.id == 104


def test_the_target_sticks_unless_a_new_one_is_clearly_better():
    b, mem, z = brain()
    hits = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 1.0, "by": i} for i in (7, 8)]
    p = step(b, mem, z, obs(10.0, [mech(7, 100, 0, math.pi), mech(8, 0, 105, -math.pi / 2)], hits))
    first = p.tgt.track.id
    p = step(b, mem, z, obs(10.1, [mech(7, 102, 0, math.pi), mech(8, 0, 98, -math.pi / 2)], hits))
    assert p.tgt.track.id == first


def test_duel_estimate_counts_everyone_shooting_us():
    b, mem, z = brain()
    away = mech(7, 0, 120, aim_yaw=math.pi / 2)
    mem.update(obs(10.0, [away]))
    info = b.a.update(me(), 10.0)
    assert b.a.win(me(), info[7])  # its back to us and our laser ready: a clear win alone
    hits = [{"type": "damaged", "amount": 300, "weapon": "laser", "bearing": 1.0, "by": i} for i in (8, 9)]
    mem.update(obs(10.1, [away, mech(8, 100, 0, math.pi), mech(9, -100, 0, 0.0)], hits))
    info = b.a.update(me(hp=500), 10.1)
    assert not b.a.win(me(hp=500), info[7])  # two more shooting us at 500 HP: no


def test_a_fight_in_progress_stays_rule_1_after_3_s_without_a_hit():
    b, mem, z = brain()
    enemy = mech(7, 0, 100, aim_yaw=-math.pi / 2)
    hit = {"type": "damaged", "amount": 315, "weapon": "laser", "bearing": math.pi / 2, "by": 7}
    step(b, mem, z, obs(10.0, [enemy], [hit]))
    step(b, mem, z, obs(12.0, [enemy], [{"type": "hit", "target": 7, "amount": 315, "weapon": "laser"}]))
    p = step(b, mem, z, obs(14.0, [enemy]))  # its last hit was 4 s ago, ours 2 s ago: still an exchange
    assert p.rule == 1 and "laser" in p.tgt.allow
    p = step(b, mem, z, obs(17.5, [enemy]))  # 5 s quiet on both sides: over, it is only faced
    assert p.rule == 4


def test_an_edge_about_to_catch_us_beats_a_fight():
    b, mem, z = brain()
    o = obs(100.0, [mech(7, 0, 100, aim_yaw=-math.pi / 2)],
            [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": math.pi / 2, "by": 7}])
    o["zone"] = {"center": [300, 0], "radius": 320, "next_center": [300, 0], "next_radius": 100,
                 "shrink_start": 90, "shrink_end": 110, "dps": 35}  # its edge passes us in ~1 s
    p = step(b, mem, z, o)
    assert p.mode == "zone" and p.goal is not None and p.strafe is not None and "gun" in p.tgt.allow


def test_upgrade_order_follows_the_guidelines():
    pick = D.Brain.pick_upgrade
    offer = lambda *ids: {"choices": [{"id": i} for i in ids]}
    assert pick(offer("armor", "plating", "sensors"), me()) == 1
    assert pick(offer("armor", "laser_damage", "plating"), me(t_upgrades=["plating"])) == 1
    assert pick(offer("plating", "laser_charge", "armor"), me(t_upgrades=["plating", "laser_damage", "laser_damage"])) == 0
    assert pick(offer("mortar_splash", "jets", "mortar_reload"), me()) == 1


def test_drop_only_when_healthy_calm_and_alone():
    b, mem, z = brain()
    drop = {"id": 20000, "pos": [60, 0, 0], "radius": 8, "lands_in": 0, "capture_s": 4, "progress": 0, "holder": None}
    p = step(b, mem, z, obs(100.0, drops=[drop]))
    assert p.mode == "drop"
    b, mem, z = brain()
    p = step(b, mem, z, obs(100.0, [mech(7, 140, 0, aim_yaw=0.0)], drops=[drop]))  # a mech near the drop
    assert p.mode != "drop"
    b, mem, z = brain()
    o = obs(100.0, drops=[drop])
    o["self"] = me(hp=800)
    assert step(b, mem, z, o, o["self"]).mode != "drop"  # under 70% HP
