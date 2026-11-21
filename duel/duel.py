#!/usr/bin/env python3
"""Duel a candidate bot against older versions and say whether to keep it.

    python3 duel/duel.py --kit ~/mechfight-kit WORK 12e0e5f 1eb8f4b      # working tree vs two commits
    python3 duel/duel.py --kit ~/mechfight-kit --rounds 30 --loadout gun+laser WORK HEAD

The first bot is the candidate: WORK (the working tree), a git revision, or a folder. Every other bot
is an opponent; the first opponent is the baseline the verdict is about. Each round is a pair of games
on the same spawns and zone with the sides swapped, so neither bot gets the better spawn. Games run in
duel/sim.py (an approximation of the server written from protocol.md, see its docstring).

Output: wins / losses / draws per opponent, the win rate with a sign-test p-value, per-weapon hit
rates by range for both sides, and a verdict against the baseline:
KEEP (better, p < --alpha), REJECT (worse, p < --alpha), or UNSURE (run more rounds).
Exit code: 0 KEEP, 1 REJECT, 2 UNSURE.

A duel is not the 16-mech free-for-all the scorer plays: use it to compare fighting skills (aim,
dodging, weapon use), and confirm a KEEP with a practice submission.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
import gzip

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sim  # noqa: E402

BOT_FILES = ["example_bot.py", "world.py", "mechfight.py", "predict.py"]
CACHE = os.path.join(HERE, ".bots")
LOADOUTS = {"gun": ["gun", None], "gun+laser": ["gun", "laser"], "gun+lightning": ["gun", "lightning"], "gun+mortar": ["gun", "mortar"]}


def find_kit(arg):
    for c in [arg, os.environ.get("MECH_KIT"), os.path.join(REPO, "mechfight-kit"), os.path.join(REPO, "..", "mechfight-kit"),
              os.path.join(REPO, "kit")]:
        if c and os.path.isfile(os.path.join(c, "check", "stream.jsonl.gz")) and os.path.isdir(os.path.join(c, "arena")):
            return os.path.abspath(c)
    sys.exit("mechfight kit not found: pass --kit <folder with arena/ and check/>, or set MECH_KIT")


def prepare_bot(spec):
    """A folder holding the bot files for WORK, a git revision, or an existing folder. Returns (label, folder)."""
    if os.path.isdir(spec):
        return spec, os.path.abspath(spec)
    if spec == "WORK":
        blobs = {f: open(os.path.join(REPO, f), "rb").read() for f in BOT_FILES if os.path.isfile(os.path.join(REPO, f))}
        label = "WORK"
    else:
        rev = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", spec], capture_output=True, text=True)
        if rev.returncode:
            sys.exit(f"not a folder, WORK, or a git revision: {spec}")
        label = rev.stdout.strip()
        blobs = {}
        for f in BOT_FILES:
            r = subprocess.run(["git", "-C", REPO, "show", f"{label}:{f}"], capture_output=True)
            if r.returncode == 0:
                blobs[f] = r.stdout
    if "example_bot.py" not in blobs:
        sys.exit(f"{spec}: no example_bot.py")
    h = hashlib.sha1(b"".join(blobs[f] for f in sorted(blobs))).hexdigest()[:10]
    d = os.path.join(CACHE, f"{label}-{h}")
    if not os.path.isdir(d):
        os.makedirs(d + ".tmp", exist_ok=True)
        for f, b in blobs.items():
            with open(os.path.join(d + ".tmp", f), "wb") as fh:
                fh.write(b)
        os.replace(d + ".tmp", d)
    return label, d


def flat_map_dir(kit):
    """A copy of the arena with no buildings, no trees and flat ground (what bots load for --map open)."""
    d = os.path.join(CACHE, "open-arena")
    if not os.path.isfile(os.path.join(d, "v2", "terrain.i16")):
        os.makedirs(os.path.join(d, "v2"), exist_ok=True)
        with open(os.path.join(kit, "arena", "v2", "world.json")) as f:
            w = json.load(f)
        w["prisms"], w["trees"]["data"] = [], []
        with open(os.path.join(d, "v2", "world.json"), "w") as f:
            json.dump(w, f)
        t = w["terrain"]
        with open(os.path.join(d, "v2", "terrain.i16"), "wb") as f:
            f.write(bytes(2 * t["nx"] * t["ny"]))
    return d


# ------------------------------------------------------------------ worker
G = {}


def init_worker(arena, map_dir, cfg, args):
    G.update(arena=arena, map_dir=map_dir, cfg=cfg, args=args)


def play(job):
    """One game: (opponent index, round, swapped) -> result dict, from the candidate's point of view."""
    oi, rnd, swapped, cand_dir, opp_dir = job
    a = G["args"]
    rng = random.Random(a.seed * 1000003 + rnd)
    spawns, zone = sim.duel_setup(G["arena"], rng, a.dmin, a.dmax)
    if a.loadout == "mixed":
        lo = [LOADOUTS[rng.choice(sorted(LOADOUTS))], LOADOUTS[rng.choice(sorted(LOADOUTS))]]
    else:
        lo = [LOADOUTS[a.loadout], LOADOUTS[a.loadout]]
    if swapped:
        spawns = spawns[::-1]
        lo = lo[::-1]
    bots = [(["python3", "-u", "example_bot.py"], cand_dir), (["python3", "-u", "example_bot.py"], opp_dir)]  # BOT_DEBUG=1 in the env: debug counters in --logs
    log_dir = None
    if a.logs:
        log_dir = os.path.join(a.logs, f"o{oi}-r{rnd}-{'b' if swapped else 'a'}")
        os.makedirs(log_dir, exist_ok=True)
    m = sim.Match(G["arena"], G["map_dir"], G["cfg"], bots, spawns, lo, zone, seed=rng.randrange(1 << 30),
                  max_t=a.max_t, timeout=a.timeout, log_dir=log_dir)
    res = m.run()
    res.update(opp=oi, round=rnd, swapped=swapped, loadout=[[k for k in l if k] for l in lo])
    return res


# ------------------------------------------------------------------ report
def sign_p(w, l):
    """Two-sided exact sign test p-value for w wins vs l losses (draws dropped)."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def hit_table(results, side):
    shots, hits = {}, {}
    for r in results:
        s = r["stats"][side]
        for kind, by in s["shots"].items():
            for b, n in by.items():
                shots[(kind, b)] = shots.get((kind, b), 0) + n
                hits[(kind, b)] = hits.get((kind, b), 0) + s["hits"][kind].get(b, 0)
    out = {}
    for kind in sorted({k for k, _ in shots}):
        row, tot_s, tot_h = [], 0, 0
        for b in sim.RANGE_BUCKETS:
            n, h = shots.get((kind, b), 0), hits.get((kind, b), 0)
            tot_s, tot_h = tot_s + n, tot_h + h
            row.append(f"{h}/{n} {100 * h / n:4.0f}%" if n else "-")
        out[kind] = (row, f"{tot_h}/{tot_s} {100 * tot_h / tot_s:4.0f}%" if tot_s else "-")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("bots", nargs="+", help="candidate, then opponents (WORK, a git revision, or a folder)")
    ap.add_argument("--kit", help="the mechfight kit folder (arena/, check/); or set MECH_KIT")
    ap.add_argument("--rounds", type=int, default=50, help="paired rounds per opponent (2 games each)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--loadout", default="gun+laser", choices=sorted(LOADOUTS) + ["mixed"])
    ap.add_argument("--map", default="campus", choices=["campus", "open"])
    ap.add_argument("--seed", type=int, default=1, help="same seed = same spawns, zones and loadouts")
    ap.add_argument("--dmin", type=float, default=90.0, help="m between the spawns, at least")
    ap.add_argument("--dmax", type=float, default=170.0)
    ap.add_argument("--max-t", type=float, default=180.0, help="s; at the end survivors are ranked by HP")
    ap.add_argument("--timeout", type=float, default=0.2, help="s per decision (the server's is 0.05; more leaves room for parallel games)")
    ap.add_argument("--alpha", type=float, default=0.1, help="sign-test p-value for KEEP / REJECT")
    ap.add_argument("--logs", help="folder for each game's bot stderr")
    ap.add_argument("--json", help="write every game's result here")
    a = ap.parse_args()
    if len(a.bots) < 2:
        ap.error("give a candidate and at least one opponent")

    kit = find_kit(a.kit)
    with gzip.open(os.path.join(kit, "check", "stream.jsonl.gz"), "rt") as f:
        cfg = json.loads(f.readline())["config"]
    cand_label, cand_dir = prepare_bot(a.bots[0])
    opps = [prepare_bot(b) for b in a.bots[1:]]
    map_dir = flat_map_dir(kit) if a.map == "open" else os.path.join(kit, "arena")
    t0 = time.time()
    arena = sim.Arena(map_dir, flat=a.map == "open")
    print(f"map {a.map} loaded in {time.time() - t0:.1f} s; candidate {cand_label}; loadout {a.loadout}; "
          f"{a.rounds} rounds x 2 games per opponent; {a.jobs} parallel games", flush=True)

    jobs = [(oi, r, sw, cand_dir, od) for oi, (_, od) in enumerate(opps) for r in range(a.rounds) for sw in (False, True)]
    results = []
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(a.jobs, initializer=init_worker, initargs=(arena, map_dir, cfg, a)) as pool:
        for i, res in enumerate(pool.imap_unordered(play, jobs), 1):
            results.append(res)
            w = {0: "W", 1: "L", None: "D"}[res["winner"]]
            print(f"\r{i}/{len(jobs)} games ({time.time() - t0:.0f} s) last: vs {opps[res['opp']][0]} {w} at {res['t']:.0f}s "
                  f"hp {res['hp'][0]}-{res['hp'][1]}{' (time)' if res['timeout'] else ''}   ", end="", flush=True)
    print()

    verdict = None
    for oi, (label, _) in enumerate(opps):
        rs = [r for r in results if r["opp"] == oi]
        w = sum(r["winner"] == 0 for r in rs)
        l = sum(r["winner"] == 1 for r in rs)
        d = len(rs) - w - l
        p = sign_p(w, l)
        rate = (w + 0.5 * d) / len(rs) if rs else 0
        rounds_won = sum(1 for r in range(a.rounds)
                         if sum((x["winner"] == 0) - (x["winner"] == 1) for x in rs if x["round"] == r) > 0)
        rounds_lost = sum(1 for r in range(a.rounds)
                          if sum((x["winner"] == 0) - (x["winner"] == 1) for x in rs if x["round"] == r) < 0)
        late = sum(r["stats"][0]["late"] for r in rs), sum(r["stats"][1]["late"] for r in rs)
        tos = sum(r["timeout"] for r in rs)
        print(f"\n=== {cand_label} vs {label}: {w} W  {l} L  {d} D  -> {100 * rate:.0f}% (sign test p = {p:.3f}); "
              f"rounds {rounds_won}-{rounds_lost}; {tos} went to time; late answers {late[0]} / {late[1]}")
        for side, name in ((0, cand_label), (1, label)):
            print(f"  hit rate {name:>10}   " + "  ".join(f"{('<' + str(b)) if b < 1e9 else '200+':>12}" for b in sim.RANGE_BUCKETS) + "         all")
            for kind, (row, tot) in hit_table(rs, side).items():
                print(f"  {kind:>19}   " + "  ".join(f"{c:>12}" for c in row) + f"  {tot:>14}")
        if oi == 0:
            verdict = "KEEP" if w > l and p < a.alpha else "REJECT" if l > w and p < a.alpha else "UNSURE"
            vp = (w, l, d, p, label)
    if verdict is None:
        return 2
    w, l, d, p, label = vp
    why = {"KEEP": f"better than {label} (p = {p:.3f} < {a.alpha})",
           "REJECT": f"worse than {label} (p = {p:.3f} < {a.alpha})",
           "UNSURE": f"no clear difference from {label} yet (p = {p:.3f}): run more rounds"}[verdict]
    print(f"\nverdict: {verdict}: {cand_label} is {why}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f)
    return {"KEEP": 0, "REJECT": 1, "UNSURE": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
