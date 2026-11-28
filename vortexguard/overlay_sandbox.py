"""OverlayFS copy-on-write workspace with freeze and atomic rollback.

Layout under ``root``::

    lower/     read-only base image of the agent workspace
    upper/     copy-on-write layer receiving every write of the agent
    snap/      last known-good copy of upper (checkpoint)
    work/      OverlayFS work directory
    merged/    the mount the agent sees
    quarantine/<n>/   evicted upper layers kept for forensics

Containment (:meth:`OverlaySandbox.contain`):

1. ``SIGSTOP`` to the agent's process group: every task stops before its next
   instruction in user space.
2. Lazy-detach ``merged`` (``MNT_DETACH``), so stopped tasks keep their old,
   now unreachable, view while the namespace is repaired.
3. ``renameat2(RENAME_EXCHANGE)`` swaps ``upper`` and ``snap`` in a single
   atomic step: the tainted layer becomes ``snap`` and is moved to quarantine.
4. Remount ``merged`` over ``lower`` plus the restored layer, and refresh
   ``snap`` from it.

Requires CAP_SYS_ADMIN (root in the sandbox host).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import signal
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]
_libc.umount2.argtypes = [ctypes.c_char_p, ctypes.c_int]
_libc.renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]

AT_FDCWD = -100
RENAME_EXCHANGE = 1 << 1
MNT_DETACH = 2


def _check(rc: int, what: str):
    if rc != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"{what}: {os.strerror(e)}")


def _process_state(pid: int) -> Optional[str]:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


@dataclass
class ContainmentReport:
    t_freeze_ms: float
    t_unmount_ms: float
    t_swap_ms: float
    t_remount_ms: float
    t_total_ms: float
    stopped_tasks: int
    quarantined_files: int
    quarantine_dir: str
    extra: Dict[str, float] = field(default_factory=dict)


class OverlaySandbox:
    def __init__(self, root: str, owner: Optional[tuple] = None):
        self.root = os.path.abspath(root)
        self.owner = owner                      # (uid, gid) of the agent, if unprivileged
        self.lower, self.upper, self.snap, self.work, self.merged, self.quar = (
            os.path.join(self.root, d) for d in ("lower", "upper", "snap", "work", "merged", "quarantine"))
        self.n_rollbacks = 0

    # ---- lifecycle -------------------------------------------------------
    def create(self, base_image: Optional[str] = None):
        for d in (self.upper, self.snap, self.work, self.merged, self.quar):
            os.makedirs(d, exist_ok=True)
        if base_image and not os.path.exists(self.lower):
            shutil.copytree(base_image, self.lower, symlinks=True)
        os.makedirs(self.lower, exist_ok=True)
        self._own(self.lower, recursive=True)
        self._own(self.upper)
        self._own(self.snap)
        self.mount()
        return self

    def _own(self, path: str, recursive: bool = False):
        if not self.owner:
            return
        os.chown(path, *self.owner)
        if recursive:
            for d, dirs, files in os.walk(path):
                for n in dirs + files:
                    os.lchown(os.path.join(d, n), *self.owner)

    def mount(self):
        opts = f"lowerdir={self.lower},upperdir={self.upper},workdir={self.work}".encode()
        _check(_libc.mount(b"overlay", self.merged.encode(), b"overlay", 0, opts), "mount overlay")

    def unmount(self, lazy: bool = True):
        _check(_libc.umount2(self.merged.encode(), MNT_DETACH if lazy else 0), "umount")

    def destroy(self):
        try:
            self.unmount()
        except OSError:
            pass
        shutil.rmtree(self.root, ignore_errors=True)

    def checkpoint(self):
        """Record the current upper layer as known-good (agent quiescent)."""
        shutil.rmtree(self.snap, ignore_errors=True)
        shutil.copytree(self.upper, self.snap, symlinks=True)
        self._own(self.snap, recursive=True)

    # ---- containment -----------------------------------------------------
    @staticmethod
    def freeze(pgid: int, timeout_s: float = 0.5) -> int:
        """SIGSTOP the process group and wait until every member is stopped."""
        os.killpg(pgid, signal.SIGSTOP)
        deadline = time.perf_counter() + timeout_s
        members = []
        while time.perf_counter() < deadline:
            members = [int(p) for p in os.listdir("/proc") if p.isdigit() and _pgid_of(int(p)) == pgid]
            if all(_process_state(p) in ("T", "t", "Z", "X", None) for p in members):
                break
        return len(members)

    def rollback(self) -> Dict[str, float]:
        t0 = time.perf_counter()
        self.unmount(lazy=True)
        t1 = time.perf_counter()
        _check(_libc.renameat2(AT_FDCWD, self.upper.encode(), AT_FDCWD, self.snap.encode(), RENAME_EXCHANGE),
               "renameat2(RENAME_EXCHANGE)")
        t2 = time.perf_counter()
        # overlayfs requires an empty, unused workdir after a layer change
        fresh_work = self.work + ".new"
        os.makedirs(fresh_work, exist_ok=True)
        old_work = self.work + f".old{self.n_rollbacks}"
        os.rename(self.work, old_work)
        os.rename(fresh_work, self.work)
        self.mount()
        t3 = time.perf_counter()
        return {"unmount": (t1 - t0) * 1e3, "swap": (t2 - t1) * 1e3, "remount": (t3 - t2) * 1e3,
                "_old_work": old_work}

    def contain(self, pgid: int) -> ContainmentReport:
        t0 = time.perf_counter()
        n = self.freeze(pgid)
        t1 = time.perf_counter()
        r = self.rollback()
        t2 = time.perf_counter()
        # off the critical path: quarantine the tainted layer, refresh snapshot
        qdir = os.path.join(self.quar, str(self.n_rollbacks))
        os.rename(self.snap, qdir)
        shutil.copytree(self.upper, self.snap, symlinks=True)
        self._own(self.snap, recursive=True)
        shutil.rmtree(r.pop("_old_work"), ignore_errors=True)
        nfiles = sum(len(f) for _, _, f in os.walk(qdir))
        self.n_rollbacks += 1
        return ContainmentReport(t_freeze_ms=(t1 - t0) * 1e3, t_unmount_ms=r["unmount"], t_swap_ms=r["swap"],
                                 t_remount_ms=r["remount"], t_total_ms=(t2 - t0) * 1e3, stopped_tasks=n,
                                 quarantined_files=nfiles, quarantine_dir=qdir)


def _pgid_of(pid: int) -> Optional[int]:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None
