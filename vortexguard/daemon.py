"""VortexGuard containment daemon.

Runs an agent command inside an OverlayFS sandbox under the eBPF probe and the
seccomp notification supervisor, scores the live syscall stream, and escalates:

* score >= tau_soft: tighten seccomp (SIGUSR1 to the supervisor). Network
  egress and privilege-changing syscalls now fail with EPERM.
* score >= tau_hard: freeze the process group (SIGSTOP), roll the workspace
  back to the last checkpoint, then kill the frozen tree.

Every kernel event carries a CLOCK_MONOTONIC timestamp (bpf_ktime_get_ns), so
latencies are measured on one clock from the syscall that crossed a threshold.
"""
from __future__ import annotations

import json
import os
import select
import signal
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

from .overlay_sandbox import OverlaySandbox
from .tokenizer import x86_64_to_token

REC = struct.Struct("<QIIIIiI")            # struct vg_event (32 bytes)
NOTIFY_SET = "connect,sendto,sendmsg,setuid,setgid,setreuid,setresuid,setresgid,ptrace,bpf,mount,init_module,finit_module"


@dataclass
class RunResult:
    n_events: int = 0
    max_score: float = float("-inf")
    t_start_ns: int = 0
    stages: dict = field(default_factory=dict)
    soft_event_ns: Optional[int] = None      # kernel ts of the event that crossed tau_soft
    soft_decision_ns: Optional[int] = None
    hard_event_ns: Optional[int] = None
    hard_decision_ns: Optional[int] = None
    contained_ns: Optional[int] = None
    freeze_ms: Optional[float] = None
    unmount_ms: Optional[float] = None
    swap_ms: Optional[float] = None
    remount_ms: Optional[float] = None
    rollback_ms: Optional[float] = None
    quarantined_files: int = 0
    exfil_done: bool = False
    exfil_blocked: bool = False
    denied_syscalls: List[str] = field(default_factory=list)
    trace: List[list] = field(default_factory=list)   # [kernel_ts_ns, score, n_events]
    tokens: List[int] = field(default_factory=list)
    leftover_payload: bool = False
    scoring_s: float = 0.0


class Daemon:
    def __init__(self, scorer, tau_soft: float, tau_hard: float, sandbox: OverlaySandbox,
                 probe_bin: str, exec_bin: str, uid: int = 1000, gid: int = 1000,
                 micro_batch: int = 16, keep_tokens: bool = False, contain: bool = True):
        self.scorer, self.tau_soft, self.tau_hard = scorer, tau_soft, tau_hard
        self.sb, self.probe_bin, self.exec_bin = sandbox, probe_bin, exec_bin
        self.uid, self.gid, self.mb = uid, gid, micro_batch
        self.keep_tokens, self.do_contain = keep_tokens, contain

    def run(self, cmd: List[str], timeout_s: float = 60.0) -> RunResult:
        res = RunResult()
        self.scorer.reset()
        err = tempfile.NamedTemporaryFile("w+", delete=False, suffix=".log")
        argv = [self.probe_bin, "--", self.exec_bin, "--uid", str(self.uid), "--gid", str(self.gid),
                "--notify", NOTIFY_SET, "--", *cmd]
        res.t_start_ns = time.monotonic_ns()
        proc = subprocess.Popen(argv, cwd=self.sb.merged, stdout=subprocess.PIPE, stderr=err, bufsize=0)
        fd = proc.stdout.fileno()
        root_pid = None
        buf = b""
        pending_tok, pending_ts = [], 0
        soft_sent = hard_done = False
        deadline = time.monotonic() + timeout_s
        while True:
            if root_pid is None:
                root_pid = _probe_root(err.name)
            r, _, _ = select.select([fd], [], [], 0.05)
            chunk = os.read(fd, 1 << 16) if r else b""
            if r and not chunk:
                break                                  # probe exited, stream closed
            buf += chunk
            n = len(buf) // REC.size
            for i in range(n):
                ts, tgid, pid, ppid, uid, nr, depth = REC.unpack_from(buf, i * REC.size)
                if depth < 2:                          # supervisor itself (depth 1)
                    continue
                tok = x86_64_to_token(nr)
                pending_tok.append(tok)
                pending_ts = ts
                if self.keep_tokens:
                    res.tokens.append(tok)
                if len(pending_tok) >= self.mb:
                    self._score(res, pending_tok, pending_ts)
                    pending_tok = []
            buf = buf[n * REC.size:]
            if pending_tok and not r:                  # idle: flush the partial micro-batch
                self._score(res, pending_tok, pending_ts)
                pending_tok = []
            score = res.trace[-1][1] if res.trace else float("-inf")
            if not soft_sent and score >= self.tau_soft and root_pid:
                os.kill(root_pid, signal.SIGUSR1)
                res.soft_event_ns, res.soft_decision_ns = res.trace[-1][0], time.monotonic_ns()
                soft_sent = True
            if self.do_contain and not hard_done and score >= self.tau_hard and root_pid:
                res.hard_event_ns, res.hard_decision_ns = res.trace[-1][0], time.monotonic_ns()
                rep = self.sb.contain(root_pid)        # process group id == root pid
                res.contained_ns = time.monotonic_ns()
                res.freeze_ms, res.unmount_ms, res.swap_ms, res.remount_ms = (
                    rep.t_freeze_ms, rep.t_unmount_ms, rep.t_swap_ms, rep.t_remount_ms)
                res.rollback_ms = rep.t_unmount_ms + rep.t_swap_ms + rep.t_remount_ms
                res.quarantined_files = rep.quarantined_files
                os.killpg(root_pid, signal.SIGKILL)
                hard_done = True
            if time.monotonic() > deadline:
                if root_pid:
                    os.killpg(root_pid, signal.SIGKILL)
                break
        proc.wait()
        err.seek(0)
        for line in err.read().splitlines():
            if line.startswith("STAGE "):
                _, name, t = line.split()
                res.stages[name] = int(t)
            elif line.startswith("EXFIL_DONE"):
                res.exfil_done = True
            elif line.startswith("EXFIL_BLOCKED"):
                res.exfil_blocked = True
            elif line.startswith("{") and '"deny"' in line:
                res.denied_syscalls.append(json.loads(line)["syscall"])
        err.close()
        os.unlink(err.name)
        res.leftover_payload = os.path.exists(os.path.join(self.sb.merged, "payload.sh"))
        return res

    def _score(self, res: RunResult, toks, ts):
        t0 = time.perf_counter()
        s = self.scorer.feed(toks)
        res.scoring_s += time.perf_counter() - t0
        res.n_events += len(toks)
        res.max_score = max(res.max_score, s)
        res.trace.append([ts, s, res.n_events])


def _probe_root(path: str) -> Optional[int]:
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith('{"probe":"ready"'):
                    return json.loads(line)["pid"]
    except OSError:
        pass
    return None


def sink_servers(stop_evt):
    """Local 'package mirror' (8080, serves bytes) and 'attacker sink' (9999, counts bytes)."""
    import socket
    import threading
    received = {"bytes": 0}

    def serve(port, handler):
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(64)
        srv.settimeout(0.2)
        while not stop_evt.is_set():
            try:
                c, _ = srv.accept()
            except OSError:
                continue
            threading.Thread(target=handler, args=(c,), daemon=True).start()
        srv.close()

    def mirror(c):
        try:
            c.recv(1024)
            c.sendall(b"HTTP/1.0 200 OK\r\n\r\n" + b"<a href='pkg'>pkg</a>\n" * 200)
        finally:
            c.close()

    def sink(c):
        try:
            c.settimeout(5)
            while True:
                d = c.recv(4096)
                if not d:
                    break
                received["bytes"] += len(d)
        except OSError:
            pass
        finally:
            c.close()

    for port, h in ((8080, mirror), (9999, sink)):
        threading.Thread(target=serve, args=(port, h), daemon=True).start()
    return received
