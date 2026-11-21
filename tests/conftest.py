import array
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOT = os.path.dirname(HERE)
sys.path.insert(0, BOT)
KIT = os.environ.get("MECHFIGHT_KIT", os.path.join(os.path.dirname(BOT), "mechfight-kit"))


def real_config():
    with gzip.open(os.path.join(KIT, "check", "stream.jsonl.gz"), "rt") as f:
        return json.loads(f.readline())["config"]


def make_map(d, size=200, prisms=(), height=lambda x, y: 0.0):
    """A flat (or `height`-shaped) square map of `size` m centred on 0 with the given prisms, in folder d."""
    h = size // 2
    os.makedirs(os.path.join(d, "v2"), exist_ok=True)
    n = size + 1
    raw = array.array("h", [int(round(height(-h + i, -h + j) * 100)) for j in range(n) for i in range(n)])
    with open(os.path.join(d, "v2", "terrain.i16"), "wb") as f:
        f.write(raw.tobytes())
    world = {"bounds": {"minX": -h, "maxX": h, "minY": -h, "maxY": h},
             "terrain": {"file": "terrain.i16", "minX": -h, "minY": -h, "step": 1.0, "nx": n, "ny": n, "scale": 0.01},
             "prisms": [{"kind": k, "base": b, "top": t, "ring": r, "holes": []} for k, b, t, r in prisms],
             "trees": {"fields": [], "data": []}}
    with open(os.path.join(d, "v2", "world.json"), "w") as f:
        json.dump(world, f)
    return d


def box(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]
