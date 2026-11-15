"""Erzeugt `data/lgc_frozen.json`: die schweren Messreihen, die die App nicht live rechnet (Laufzeit einige Minuten).

Aufruf (Wurzelordner der Demo): python tools/make_frozen.py
Inhalt:
  fine_grid   feines Gitter (4000 Punkte, 6000 Einschwing- und 6000 Messschritte) in den Chaos-Bereichen von L = 3, 6, 12
  gegenprobe  Gegenprobe für L = 3, alpha in [1,677; 2,397]: 1000 zufällige alpha (SplitMix64, nicht das Gitter), je 6 zufällige Anfangszustände,
              doppelte Messdauer (12000 + 12000 Schritte)
  pow2        Zweierpotenz-Falle, L = 3, D = 10, Schrittgrenze 3·10⁶: einfache Startwerte und 100 gestreute Startwerte für alpha = 2, 2 + 1e-9·√2 und 1,1
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lgc_constants as C  # noqa: E402
import lgc_dynamics as K  # noqa: E402
from lgc_evaluation import EINFACHE_STARTS  # noqa: E402
from lgc_rng import SplitMix64, uniform_array  # noqa: E402

RANGES = ((3, 1.677, 2.397), (6, 1.520, 2.381), (12, 1.632, 2.340))
CAP = C.FROZEN_CAP


def fine_grid() -> dict:
    out = {}
    for L, lo, hi in RANGES:
        al = K.alpha_grid(lo, hi, 4000)
        res = K.lyapunov_grid(L, al)
        ch = res.lam > K.LAMBDA_SCHWELLE
        out[str(L)] = {"lo": lo, "hi": hi, "n": 4000, "n_chaotic": int(ch.sum()), "n_window": int((res.lam < -K.LAMBDA_SCHWELLE).sum()),
                       "runs_non_chaotic": K.count_runs(~ch), "lam_min": float(res.lam.min()), "gap": float((hi - lo) / 4000)}
        print("fine_grid", L, out[str(L)], flush=True)
    return out


def gegenprobe() -> dict:
    lo, hi, n_alpha, n_start = 1.677, 2.397, 1000, 6
    a = lo + (hi - lo) * uniform_array(20261114, n_alpha)
    i0 = (-50.0 + 100.0 * uniform_array(77, n_alpha * n_start)).reshape(n_alpha, n_start)
    al = np.repeat(a, n_start)
    res = K.lyapunov_grid(3, al, burn=12000, rec=12000, i0=i0.reshape(-1))
    ch = res.lam > K.LAMBDA_SCHWELLE
    out = {"alpha_lo": lo, "alpha_hi": hi, "n_alpha": n_alpha, "n_start": n_start, "runs": int(al.size), "n_chaotic": int(ch.sum()),
           "lam_min": float(res.lam.min()), "burn": 12000, "rec": 12000, "start_range": [-50.0, 50.0]}
    print("gegenprobe", out, flush=True)
    return out


def _search(alpha: float, starts) -> dict:
    step = K.make_step(3, alpha, 10.0)
    res = [K.brent_cycle(step, K.start_state(3, s0), CAP) for s0 in starts]
    cyc = [r for r in res if r is not None]
    return {"n": len(starts), "n_cycle": len(cyc), "periods": sorted({r[0] for r in cyc})[:20], "results": [None if r is None else list(r) for r in res][:20]}


def pow2() -> dict:
    alpha2 = 2.0 + 1e-9 * math.sqrt(2)
    rng = SplitMix64(C.DEFAULT_SEED)
    scattered = [rng.uniform(-50.0, 50.0) for _ in range(100)]
    out = {"cap": CAP, "L": 3, "D": 10.0, "alpha2": alpha2, "scattered_seed": C.DEFAULT_SEED, "scattered_range": [-50.0, 50.0]}
    for name, al, st in (("einfach_alpha2", 2.0, EINFACHE_STARTS), ("einfach_alpha2b", alpha2, EINFACHE_STARTS), ("einfach_alpha11", 1.1, EINFACHE_STARTS),
                         ("gestreut_alpha2", 2.0, scattered), ("gestreut_alpha2b", alpha2, scattered), ("gestreut_alpha11", 1.1, scattered)):
        out[name] = _search(al, st)
        print("pow2", name, {k: v for k, v in out[name].items() if k != "results"}, flush=True)
    return out


if __name__ == "__main__":
    data = {"meta": {"erzeugt_von": "tools/make_frozen.py", "schwelle": K.LAMBDA_SCHWELLE, "burn": K.BURN, "rec": K.REC}, "fine_grid": fine_grid(), "gegenprobe": gegenprobe(), "pow2": pow2()}
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "lgc_frozen.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("geschrieben: data/lgc_frozen.json")
