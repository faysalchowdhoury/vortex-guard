"""Multi-channel syscall event tokenizer.

Every syscall event is mapped to three aligned channels:

* ``tok``: a vocabulary id in the i386 numbering used by ADFA-LD. Live x86_64
  events are translated by syscall *name*, so a detector trained on ADFA-LD can
  score events captured by the eBPF probe on a 64-bit kernel.
* ``cat``: a coarse semantic category (process, file, network, memory, ipc,
  privilege, other).
* ``edge``: the edge of the resource-flux graph the event contributes to
  (see :mod:`vortexguard.models`), or ``-1`` when it carries no flux.

The resource-flux graph has five nodes: P (the sandboxed process tree),
F (filesystem), N (network), C (spawned children / exec images) and
K (kernel privilege state). Directed edges carry event rates.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "syscall_tables.json")) as fh:
    _TABLES = json.load(fh)

I386 = _TABLES["i386"]                     # name -> i386 nr
X86_64 = _TABLES["x86_64"]                 # name -> x86_64 nr
I386_BY_NR = {v: k for k, v in I386.items()}
X86_64_BY_NR = {v: k for k, v in X86_64.items()}

PAD = 0
UNK = 1
OFFSET = 2                                  # token = i386 nr + OFFSET
VOCAB_SIZE = max(I386.values()) + OFFSET + 1

# 64-bit names that have a differently named i386 equivalent.
_ALIASES = {
    "newfstatat": "fstatat64", "fstat": "fstat64", "stat": "stat64", "lstat": "lstat64",
    "mmap": "mmap2", "fadvise64": "fadvise64_64", "sendfile": "sendfile64",
    "getuid": "getuid32", "geteuid": "geteuid32", "getgid": "getgid32", "getegid": "getegid32",
    "setuid": "setuid32", "setgid": "setgid32", "setresuid": "setresuid32",
    "setresgid": "setresgid32", "setreuid": "setreuid32", "setregid": "setregid32",
    "chown": "chown32", "fchown": "fchown32", "lchown": "lchown32",
    "getresuid": "getresuid32", "getresgid": "getresgid32", "getgroups": "getgroups32",
    "setgroups": "setgroups32", "fcntl": "fcntl64", "truncate": "truncate64",
    "ftruncate": "ftruncate64", "statfs": "statfs64", "fstatfs": "fstatfs64",
    "getdents": "getdents64",
}

# ---- semantic categories -------------------------------------------------
CATEGORIES = ["other", "process", "file", "network", "memory", "ipc", "privilege"]
_CAT_MEMBERS = {
    "process": "clone fork vfork execve execveat exit exit_group wait4 waitpid waitid kill tgkill tkill "
               "getpid getppid gettid set_tid_address sched_yield nanosleep clock_nanosleep",
    "file": "open openat openat2 creat close read write readv writev pread64 pwrite64 lseek _llseek "
            "stat64 lstat64 fstat64 fstatat64 statx access faccessat faccessat2 unlink unlinkat rename "
            "renameat renameat2 mkdir mkdirat rmdir link linkat symlink symlinkat readlink readlinkat "
            "getdents getdents64 truncate64 ftruncate64 fsync fdatasync dup dup2 dup3 fcntl64 chdir fchdir getcwd "
            "sendfile64 splice",
    "network": "socketcall socket connect accept accept4 bind listen send sendto sendmsg sendmmsg recv recvfrom "
               "recvmsg recvmmsg shutdown getsockname getpeername socketpair setsockopt getsockopt",
    "memory": "brk mmap2 mmap munmap mprotect mremap madvise mlock munlock",
    "ipc": "pipe pipe2 ipc futex select _newselect poll ppoll epoll_wait epoll_ctl epoll_create "
           "eventfd2 rt_sigaction rt_sigprocmask sigreturn rt_sigreturn",
    "privilege": "setuid32 setgid32 setreuid32 setregid32 setresuid32 setresgid32 setuid setgid setreuid "
                 "setresuid setgroups32 capset capget chmod fchmod fchmodat chown32 fchown32 lchown32 "
                 "chown lchown ptrace mount umount umount2 init_module finit_module delete_module "
                 "prctl personality chroot pivot_root unshare setns bpf keyctl add_key request_key",
}
_CAT_OF = {}
for _c, _names in _CAT_MEMBERS.items():
    for _n in _names.split():
        _CAT_OF[_n] = CATEGORIES.index(_c)

# ---- resource-flux graph -------------------------------------------------
NODES = ["P", "F", "N", "C", "K"]
EDGES = [("F", "P"), ("P", "F"), ("N", "P"), ("P", "N"), ("P", "C"), ("P", "K")]
_EDGE_MEMBERS = {
    0: "read readv pread64 open openat openat2 getdents getdents64 readlink readlinkat",
    1: "write writev pwrite64 creat unlink unlinkat rename renameat renameat2 mkdir mkdirat rmdir "
       "link linkat symlink symlinkat truncate64 ftruncate64",
    2: "recv recvfrom recvmsg recvmmsg accept accept4",
    3: "send sendto sendmsg sendmmsg connect socketcall socket sendfile64 bind listen",
    4: "clone fork vfork execve execveat",
    5: "setuid32 setgid32 setreuid32 setregid32 setresuid32 setresgid32 setuid setgid setreuid setresuid "
       "setgroups32 capset chmod fchmod fchmodat chown32 fchown32 lchown32 chown lchown ptrace mount "
       "umount2 init_module finit_module delete_module prctl personality chroot pivot_root unshare setns "
       "bpf keyctl add_key",
}
_EDGE_OF = {}
for _e, _names in _EDGE_MEMBERS.items():
    for _n in _names.split():
        _EDGE_OF[_n] = _e


def incidence_matrix() -> np.ndarray:
    """Node-by-edge incidence B with B[u,e]=+1 if e leaves u, -1 if it enters u.

    The graph divergence of an edge flux vector phi is ``B @ phi``: the net
    outflow at every node.
    """
    B = np.zeros((len(NODES), len(EDGES)), dtype=np.float32)
    for e, (u, v) in enumerate(EDGES):
        B[NODES.index(u), e] += 1.0
        B[NODES.index(v), e] -= 1.0
    return B


# ---- lookup tables indexed by token id ------------------------------------
_TOK_CAT = np.zeros(VOCAB_SIZE, dtype=np.int64)
_TOK_EDGE = -np.ones(VOCAB_SIZE, dtype=np.int64)
for _name, _nr in I386.items():
    _TOK_CAT[_nr + OFFSET] = _CAT_OF.get(_name, 0)
    _TOK_EDGE[_nr + OFFSET] = _EDGE_OF.get(_name, -1)


def i386_name(nr: int) -> str:
    return I386_BY_NR.get(nr, f"sys_{nr}")


def x86_64_to_token(nr: int) -> int:
    """Translate a live x86_64 syscall number to the ADFA-LD (i386) vocabulary."""
    name = X86_64_BY_NR.get(nr)
    if name is None:
        return UNK
    name = _ALIASES.get(name, name)
    i = I386.get(name)
    return UNK if i is None else i + OFFSET


def i386_to_token(nr: int) -> int:
    return nr + OFFSET if 0 <= nr < VOCAB_SIZE - OFFSET else UNK


@dataclass
class Encoded:
    tok: np.ndarray   # (L,) int64
    cat: np.ndarray   # (L,) int64
    edge: np.ndarray  # (L,) int64, -1 when no flux


def encode_tokens(tok: Sequence[int]) -> Encoded:
    t = np.asarray(tok, dtype=np.int64)
    return Encoded(tok=t, cat=_TOK_CAT[t], edge=_TOK_EDGE[t])


def encode_adfa(nrs: Iterable[int]) -> Encoded:
    return encode_tokens([i386_to_token(n) for n in nrs])


def encode_live(nrs: Iterable[int]) -> Encoded:
    return encode_tokens([x86_64_to_token(n) for n in nrs])


def edge_lookup() -> np.ndarray:
    return _TOK_EDGE.copy()


def native_allowlist_from_i386(names: Iterable[str]) -> List[str]:
    """Map i386 syscall names (e.g. from a profile learned on ADFA-LD) to x86_64 names."""
    inv = {v: k for k, v in _ALIASES.items()}
    out = set()
    for n in names:
        n64 = inv.get(n, n)
        if n64 in X86_64:
            out.add(n64)
    return sorted(out)
