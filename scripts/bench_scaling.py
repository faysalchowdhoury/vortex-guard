"""Online-update latency and memory versus history length L.

For each detector and L, a fresh process (so peak RSS is clean) consumes L
events of history taken from ADFA-LD test traces, then times ``REPS`` further
micro-batch updates of 16 events. TransformerIDS re-encodes a window of L
events per update (its only way to see L events of context); VortexGuard and
LSTMIDS carry recurrent state.

    python3 bench_scaling.py              # driver: writes results/scaling.json
    python3 bench_scaling.py --cell M L   # worker
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384]
MODELS = ["VortexGuard", "LSTMIDS", "TransformerIDS"]
MB = 16
REPS = 200
MEM_BUDGET_MB = 1536          # per-tenant detector memory budget (RLIMIT_DATA)


def token_stream(n):
    from vortexguard.data import load_split
    from vortexguard.tokenizer import encode_adfa
    sp = load_split()
    toks = np.concatenate([encode_adfa(t.seq).tok for t in sp.test if t.label == 0])
    reps = int(np.ceil(n / len(toks)))
    return np.tile(toks, reps)[:n]


def cell(model, L):
    lim = MEM_BUDGET_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_DATA, (lim, lim))
    import torch
    torch.set_num_threads(1)
    from vortexguard.scorers import make_scorer
    stream = token_stream(L + MB * (REPS + 10))
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sd = torch.load(os.path.join(ROOT, "results", "checkpoints", f"{model}_seed0.pt"))
    sc = make_scorer(model, sd, window=L)
    sc.reset()
    hist = stream[:L]
    t0 = time.perf_counter()
    for i in range(0, L, 256):                      # consume history
        sc.feed(hist[i:i + 256])
    t_hist = time.perf_counter() - t0
    lat = []
    pos = L
    for _ in range(10):                             # warm-up updates
        sc.feed(stream[pos:pos + MB]); pos += MB
    for _ in range(REPS):
        t = time.perf_counter()
        sc.feed(stream[pos:pos + MB])
        lat.append((time.perf_counter() - t) * 1e3)
        pos += MB
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"model": model, "L": L, "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99)),
            "mean_ms": float(np.mean(lat)), "per_event_us_p50": float(np.percentile(lat, 50)) * 1e3 / MB,
            "history_ingest_s": t_hist, "peak_rss_mb": rss1 / 1024, "delta_rss_mb": (rss1 - rss0) / 1024,
            "state_bytes": sc.state_bytes()}


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--cell":
        print(json.dumps(cell(sys.argv[2], int(sys.argv[3]))))
        return
    out = []
    for m in MODELS:
        for L in LENGTHS:
            p = subprocess.run([sys.executable, __file__, "--cell", m, str(L)], capture_output=True, text=True)
            if p.returncode != 0:
                oom = "MemoryError" in p.stderr or "alloc" in p.stderr.lower() or p.returncode < 0
                out.append({"model": m, "L": L, "error": p.stderr[-400:], "over_budget": bool(oom),
                            "budget_mb": MEM_BUDGET_MB})
                print(m, L, "ERROR", p.stderr[-200:], flush=True)
                continue
            r = json.loads(p.stdout.strip().splitlines()[-1])
            out.append(r)
            print(m, L, f"p50={r['p50_ms']:.2f}ms p99={r['p99_ms']:.2f}ms rss={r['peak_rss_mb']:.0f}MB", flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "scaling.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
