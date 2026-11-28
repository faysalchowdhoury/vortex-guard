"""Benchmark workloads for runtime-overhead measurements.

    python3 bench_workload.py agent|build|fileio

Prints ``STAGE start`` / ``STAGE end`` with CLOCK_MONOTONIC timestamps to stderr,
so start-up of the measurement harness is excluded from the workload time.
"""
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def stage(n):
    sys.stderr.write(f"STAGE {n} {time.monotonic_ns()}\n")
    sys.stderr.flush()


def agent():
    import agent_task
    rng = random.Random(0)
    for _ in range(6):
        agent_task.benign_work(rng, 3)
        agent_task.fetch_index(rng)


def build():
    src = "bench.c"
    with open(src, "w") as fh:
        fh.write("#include <stdio.h>\n" + "".join(f"int f{i}(int x){{return x*{i}+1;}}\n" for i in range(200))
                 + "int main(void){int s=0;" + "".join(f"s+=f{i}(s);" for i in range(200)) + "printf(\"%d\\n\",s);return 0;}\n")
    for i in range(8):
        subprocess.run(["gcc", "-O1", "-o", f"bench{i}", src], check=True)
        subprocess.run([f"./bench{i}"], stdout=subprocess.DEVNULL, check=True)


def fileio():
    d = "tree"
    for i in range(40):
        os.makedirs(f"{d}/{i}", exist_ok=True)
        for j in range(25):
            with open(f"{d}/{i}/f{j}.txt", "w") as fh:
                fh.write(("line %d %d\n" % (i, j)) * 200)
    subprocess.run(["tar", "-czf", "tree.tgz", d], check=True)
    os.makedirs("out", exist_ok=True)
    subprocess.run(["tar", "-xzf", "tree.tgz", "-C", "out"], check=True)
    subprocess.run(["grep", "-r", "-c", "line 7", "out"], stdout=subprocess.DEVNULL)


def main():
    work = tempfile.mkdtemp(prefix="vgbench_", dir=os.environ.get("VG_BENCH_DIR", "."))
    os.chdir(work)
    stage("start")
    {"agent": agent, "build": build, "fileio": fileio}[sys.argv[1]]()
    stage("end")
    os.chdir("/")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
