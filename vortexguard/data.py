"""ADFA-LD loading and a leakage-free split.

Protocol
--------
* Normal traces: ``Training_Data_Master`` goes to train. ``Validation_Data_Master``
  is shuffled with a fixed seed and split 30/70 into validation and test.
* Attack traces: ADFA-LD stores every attack family as 10 independent attack
  *instances* (folders ``<Family>_1`` .. ``<Family>_10``). Splitting is done at
  the instance level (folders 1-5 train, 6-7 validation, 8-10 test), so no
  trace of one attack execution appears in two splits.
* Exact-duplicate traces are removed: a validation or test trace whose syscall
  sequence is byte-identical to a training trace is dropped, and test traces
  identical to validation traces are dropped.
* All fitting (model weights, static allowlists, thresholds) uses train or
  validation only. Test traces are scored once.
"""
from __future__ import annotations

import glob
import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List

FAMILIES = ["Adduser", "Hydra_FTP", "Hydra_SSH", "Java_Meterpreter", "Meterpreter", "Web_Shell"]
ROOT_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "adfa", "ADFA-LD")


@dataclass
class Trace:
    seq: List[int]
    label: int          # 0 normal, 1 attack
    family: str         # "Normal" or attack family
    instance: str       # attack folder or "normal"
    path: str


@dataclass
class Split:
    train: List[Trace] = field(default_factory=list)
    val: List[Trace] = field(default_factory=list)
    test: List[Trace] = field(default_factory=list)
    dropped_duplicates: Dict[str, int] = field(default_factory=dict)


def _read(path: str) -> List[int]:
    with open(path) as fh:
        return [int(x) for x in fh.read().split()]


def _digest(seq: List[int]) -> str:
    return hashlib.sha1(" ".join(map(str, seq)).encode()).hexdigest()


def load_split(root: str = ROOT_DEFAULT, seed: int = 1337) -> Split:
    sp = Split()
    for p in sorted(glob.glob(os.path.join(root, "Training_Data_Master", "*.txt"))):
        sp.train.append(Trace(_read(p), 0, "Normal", "normal", p))
    val_norm = sorted(glob.glob(os.path.join(root, "Validation_Data_Master", "*.txt")))
    random.Random(seed).shuffle(val_norm)
    cut = int(0.3 * len(val_norm))
    for i, p in enumerate(val_norm):
        (sp.val if i < cut else sp.test).append(Trace(_read(p), 0, "Normal", "normal", p))
    for fam in FAMILIES:
        for k in range(1, 11):
            folder = os.path.join(root, "Attack_Data_Master", f"{fam}_{k}")
            bucket = sp.train if k <= 5 else (sp.val if k <= 7 else sp.test)
            for p in sorted(glob.glob(os.path.join(folder, "*.txt"))):
                bucket.append(Trace(_read(p), 1, fam, f"{fam}_{k}", p))

    seen = {_digest(t.seq) for t in sp.train}
    before = len(sp.val)
    sp.val = [t for t in sp.val if _digest(t.seq) not in seen]
    sp.dropped_duplicates["val_vs_train"] = before - len(sp.val)
    seen |= {_digest(t.seq) for t in sp.val}
    before = len(sp.test)
    sp.test = [t for t in sp.test if _digest(t.seq) not in seen]
    sp.dropped_duplicates["test_vs_trainval"] = before - len(sp.test)
    return sp


def describe(sp: Split) -> Dict[str, Dict[str, int]]:
    out = {}
    for name in ("train", "val", "test"):
        part = getattr(sp, name)
        d = {"normal": sum(t.label == 0 for t in part), "attack": sum(t.label == 1 for t in part)}
        for fam in FAMILIES:
            d[fam] = sum(t.family == fam for t in part)
        d["instances"] = len({t.instance for t in part if t.label == 1})
        out[name] = d
    return out
