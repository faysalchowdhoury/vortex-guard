"""Live containment experiments (run as root on a BTF-enabled Linux host).

Phases
------
pilot      a few benign and exploit runs without containment (score ranges, throughput)
calibrate  benign-only runs; thresholds are set from these runs alone (no attack data)
trials     exploit trials with containment plus held-out benign runs

Outputs JSON to results/containment_<phase>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vortexguard.daemon import Daemon, sink_servers  # noqa: E402
from vortexguard.overlay_sandbox import OverlaySandbox  # noqa: E402
from vortexguard.scorers import make_scorer  # noqa: E402

PROBE = os.path.join(ROOT, "kernel", "build", "ebpf_syscall_probe")
EXEC = os.path.join(ROOT, "kernel", "build", "vg_sandbox_exec")
SANDBOX_DIR = "/var/lib/vortexguard/sandbox"
BASE_DIR = "/var/lib/vortexguard/base"
DECOYS = {
    ".aws/credentials": "[default]\naws_access_key_id = AKIADECOYDECOYDECOY0\naws_secret_access_key = decoy/" + "x" * 32 + "\n",
    ".ssh/id_rsa": "-----BEGIN OPENSSH PRIVATE KEY-----\n" + "DECOY" * 60 + "\n-----END OPENSSH PRIVATE KEY-----\n",
    ".git-credentials": "https://agent:decoy-token-0000@git.example.invalid\n",
    ".env": "DATABASE_URL=postgres://agent:decoy@db.invalid/app\nAPI_KEY=decoy-0000\n",
    "README.md": "Agent workspace.\n",
}


def build_base():
    shutil.rmtree(BASE_DIR, ignore_errors=True)
    os.makedirs(BASE_DIR)
    for rel, txt in DECOYS.items():
        p = os.path.join(BASE_DIR, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(txt)
    shutil.copy(os.path.join(ROOT, "scripts", "workloads", "agent_task.py"), BASE_DIR)


def new_sandbox():
    if os.path.ismount(os.path.join(SANDBOX_DIR, "merged")):
        os.system(f"umount -l {SANDBOX_DIR}/merged")
    shutil.rmtree(SANDBOX_DIR, ignore_errors=True)
    return OverlaySandbox(SANDBOX_DIR, owner=(1000, 1000)).create(BASE_DIR)


def load_scorer():
    sd = torch.load(os.path.join(ROOT, "results", "checkpoints", "VortexGuard_seed0.pt"))
    return make_scorer("VortexGuard", sd)


def summarize(r, received, keep_trace):
    d = {k: v for k, v in r.__dict__.items() if k not in ("trace", "tokens")}
    d["exfil_bytes_received"] = received
    if keep_trace:
        d["trace"] = r.trace
    return d


def run_many(daemon, sb, jobs, received, keep_first=1):
    out = []
    for i, (mode, seed, mimicry) in enumerate(jobs):
        received["bytes"] = 0
        cmd = ["python3", "agent_task.py", mode, str(seed)] + (["--mimicry"] if mimicry else [])
        r = daemon.run(cmd)
        time.sleep(0.05)
        rec = summarize(r, received["bytes"], keep_trace=i < keep_first)
        rec.update({"mode": mode, "seed": seed, "mimicry": mimicry})
        out.append(rec)
        # reset the workspace for the next run (outside any measured path)
        sb.unmount()
        shutil.rmtree(sb.upper); os.makedirs(sb.upper)
        shutil.rmtree(sb.work); os.makedirs(sb.work)
        sb._own(sb.upper)
        sb.mount()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(jobs)} {mode}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "calibrate", "trials"], required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-benign", type=int, default=200)
    ap.add_argument("--mimicry-frac", type=float, default=0.3)
    args = ap.parse_args()
    torch.set_num_threads(1)
    build_base()
    sb = new_sandbox()
    stop = threading.Event()
    received = sink_servers(stop)
    time.sleep(0.3)
    scorer = load_scorer()
    res_dir = os.path.join(ROOT, "results")

    if args.phase in ("pilot", "calibrate"):
        d = Daemon(scorer, float("inf"), float("inf"), sb, PROBE, EXEC, contain=False)
        if args.phase == "pilot":
            jobs = [("benign", s, False) for s in range(5)] + [("exploit", s, False) for s in range(3)] + \
                   [("exploit", s, True) for s in range(3, 5)]
        else:
            jobs = [("benign", 1000 + s, False) for s in range(args.n)]
        out = run_many(d, sb, jobs, received, keep_first=len(jobs) if args.phase == "pilot" else 0)
        for r in out:
            print(r["mode"], r["mimicry"], "events", r["n_events"], "max", round(r["max_score"], 3),
                  "us/event", round(1e6 * r["scoring_s"] / max(1, r["n_events"]), 1))
        if args.phase == "calibrate":
            mx = np.array([r["max_score"] for r in out])
            cal = {"n_benign": len(out), "max_benign_score": float(mx.max()),
                   "p99_benign_score": float(np.percentile(mx, 99)),
                   "tau_soft": float(mx.max()), "tau_hard": float(mx.max() + 1.0),
                   "rule": "tau_soft = max benign calibration score; tau_hard = tau_soft + 1 logit"}
            json.dump({"calibration": cal, "runs": out}, open(os.path.join(res_dir, "containment_calibrate.json"), "w"))
            print(cal)
        else:
            json.dump(out, open(os.path.join(res_dir, "containment_pilot.json"), "w"))
    else:
        cal = json.load(open(os.path.join(res_dir, "containment_calibrate.json")))["calibration"]
        d = Daemon(scorer, cal["tau_soft"], cal["tau_hard"], sb, PROBE, EXEC, contain=True)
        rng = np.random.RandomState(7)
        n_mim = int(round(args.mimicry_frac * args.n))
        mim = np.zeros(args.n, dtype=bool); mim[rng.choice(args.n, n_mim, replace=False)] = True
        jobs = [("exploit", 10000 + i, bool(mim[i])) for i in range(args.n)]
        t0 = time.time()
        attacks = run_many(d, sb, jobs, received, keep_first=3)
        benign = run_many(d, sb, [("benign", 5000 + s, False) for s in range(args.n_benign)], received, keep_first=0)
        json.dump({"calibration": cal, "attacks": attacks, "benign": benign, "wall_s": time.time() - t0},
                  open(os.path.join(res_dir, "containment_trials.json"), "w"))
    stop.set()
    sb.unmount()


if __name__ == "__main__":
    main()
