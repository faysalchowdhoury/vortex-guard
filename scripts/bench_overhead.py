"""Runtime overhead of each enforcement / detection configuration (run as root).

Configurations
--------------
native            no enforcement
static_seccomp    seccomp-bpf allowlist profiled from benign runs of the workload
ebpf_probe        CO-RE ring-buffer probe only (events consumed and discarded)
seccomp_notify    dynamic-tightening supervisor in its permissive phase
VortexGuard / LSTMIDS / TransformerIDS
                  full pipeline: OverlayFS sandbox + probe + notify supervisor +
                  online scorer (no containment action)

Workload time is measured between the workload's own STAGE start/end markers.
For detector pipelines we also report the *verdict lag*: time from workload end
until every event it produced has been scored.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vortexguard.daemon import Daemon, sink_servers, NOTIFY_SET  # noqa: E402
from vortexguard.overlay_sandbox import OverlaySandbox  # noqa: E402
from vortexguard.scorers import make_scorer  # noqa: E402

PROBE = os.path.join(ROOT, "kernel", "build", "ebpf_syscall_probe")
EXEC = os.path.join(ROOT, "kernel", "build", "vg_sandbox_exec")
WL = os.path.join(ROOT, "scripts", "workloads")
BENCH_DIR = "/var/lib/vortexguard/bench"
SB_DIR = "/var/lib/vortexguard/bench_sandbox"
WORKLOADS = ["agent", "build", "fileio"]
REPS = int(os.environ.get("VG_REPS", "7"))


def markers(text):
    st = {}
    for line in text.splitlines():
        if line.startswith("STAGE "):
            _, n, t = line.split()
            st[n] = int(t)
    return (st["end"] - st["start"]) / 1e6


def run_plain(prefix, wl):
    env = dict(os.environ, VG_BENCH_DIR=BENCH_DIR)
    p = subprocess.run(prefix + ["python3", os.path.join(WL, "bench_workload.py"), wl],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env)
    return markers(p.stderr)


def profile_allowlist():
    """Static profile: every syscall the benign workloads issue (as a tracing-based profile generator would)."""
    from vortexguard.tokenizer import X86_64_BY_NR
    import struct
    names = set()
    for wl in WORKLOADS:
        out = tempfile.mktemp()
        subprocess.run([PROBE, "--out", out, "--", EXEC, "--uid", "1000", "--gid", "1000", "--none", "--",
                        "python3", os.path.join(WL, "bench_workload.py"), wl],
                       stderr=subprocess.DEVNULL, env=dict(os.environ, VG_BENCH_DIR=BENCH_DIR))
        b = open(out, "rb").read()
        for i in range(0, len(b) - 31, 32):
            nr = struct.unpack_from("<i", b, i + 24)[0]
            if nr in X86_64_BY_NR:
                names.add(X86_64_BY_NR[nr])
        os.unlink(out)
    path = "/var/lib/vortexguard/static_profile.txt"
    open(path, "w").write("\n".join(sorted(names)) + "\n")
    return path, len(names)


def main():
    torch.set_num_threads(1)
    os.makedirs(BENCH_DIR, exist_ok=True)
    os.chmod(BENCH_DIR, 0o777)
    prof, nprof = profile_allowlist()
    drop = ["--uid", "1000", "--gid", "1000"]
    plain = {
        "native": [EXEC] + drop + ["--none", "--"],
        "static_seccomp": [EXEC] + drop + ["--allowlist", prof, "--action", "errno", "--"],
        "ebpf_probe": [PROBE, "--count-only", "--", EXEC] + drop + ["--none", "--"],
        "seccomp_notify": [EXEC] + drop + ["--notify", NOTIFY_SET, "--"],
    }
    res = {"reps": REPS, "static_profile_size": nprof, "configs": {}}
    stop = threading.Event()
    sink_servers(stop)
    time.sleep(0.3)
    # warm-up (page cache, Python bytecode)
    for wl in WORKLOADS:
        run_plain(plain["native"], wl)

    for name, prefix in plain.items():
        res["configs"][name] = {wl: [run_plain(prefix, wl) for _ in range(REPS)] for wl in WORKLOADS}
        print(name, {wl: round(float(np.median(v)), 1) for wl, v in res["configs"][name].items()}, flush=True)

    base = "/var/lib/vortexguard/bench_base"
    shutil.rmtree(base, ignore_errors=True)
    shutil.copytree(WL, base)
    for det in ("VortexGuard", "LSTMIDS", "TransformerIDS"):
        sd = torch.load(os.path.join(ROOT, "results", "checkpoints", f"{det}_seed0.pt"))
        scorer = make_scorer(det, sd)
        if os.path.ismount(os.path.join(SB_DIR, "merged")):
            os.system(f"umount -l {SB_DIR}/merged")
        shutil.rmtree(SB_DIR, ignore_errors=True)
        sb = OverlaySandbox(SB_DIR, owner=(1000, 1000)).create(base)
        d = Daemon(scorer, float("inf"), float("inf"), sb, PROBE, EXEC, contain=False)
        per = {}
        for wl in WORKLOADS:
            vals = []
            for _ in range(REPS):
                r = d.run(["env", "VG_BENCH_DIR=.", "python3", "bench_workload.py", wl], timeout_s=600)
                t_ret = time.monotonic_ns()
                vals.append({"workload_ms": (r.stages["end"] - r.stages["start"]) / 1e6,
                             "verdict_lag_ms": (t_ret - r.stages["end"]) / 1e6,
                             "events": r.n_events, "scoring_s": r.scoring_s})
            per[wl] = vals
        res["configs"][det] = per
        sb.unmount()
        print(det, {wl: round(float(np.median([v["workload_ms"] for v in vals])), 1) for wl, vals in per.items()},
              {wl: round(float(np.median([v["verdict_lag_ms"] for v in vals])), 1) for wl, vals in per.items()}, flush=True)
    stop.set()
    json.dump(res, open(os.path.join(ROOT, "results", "overhead.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
