"""mechfight bot rebuilt from scratch (GUIDELINES.md §3). Every decision runs the same layers in order:

    perception & memory -> assessment -> decision -> execution -> action

This file is the frame: the per-decision clock, the log line, and the wiring. The layers live in their own modules.
"""
import math
import sys
import time
import traceback

import mechfight as mf
from memory import Memory
from world import World

NAME = "scratch"
TICK_CAP = 0.025  # s we allow ourselves per decision (the server's budget is 50 ms)


class Clock:
    """Wall time left in this decision."""

    def __init__(self, cap: float):
        self.t0, self.cap = time.perf_counter(), cap

    def left(self) -> float:
        return self.cap - (time.perf_counter() - self.t0)

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000


def log(line: str) -> None:
    sys.stderr.write(line + "\n")


class Bot:
    name = NAME

    def __init__(self, hello):
        t0 = time.perf_counter()
        self.cfg = hello["config"]
        try:
            self.world = World(hello["map"]["dir"], self.cfg["mech"])
        except Exception:  # no map: play on without walls (straight walks, every line of fire taken as clear)
            log(traceback.format_exc())
            self.world = None
        self.mem = Memory(self.cfg, self.world)
        log(f"hello: world {'ok' if self.world else 'MISSING'} in {time.perf_counter() - t0:.2f} s")

    def act(self, obs):
        clock = Clock(TICK_CAP)
        self.mem.update(obs)
        me = obs["self"]
        x, y = me["pos"][0], me["pos"][1]
        c = obs["zone"]["next_center"]
        yaw = math.atan2(c[1] - y, c[0] - x)
        act = {"aim": [yaw, 0.0]}
        if obs.get("offer"):
            act["pick"] = 0
        mode = "frame"
        mf.draw.label(mode)
        log(f"t={obs['t']:.1f} m={mode} hp={me['hp']:.0f} p={x:.0f},{y:.0f} ms={clock.ms():.1f}")
        return act


if __name__ == "__main__":
    mf.run(Bot)
