"""Train and evaluate VortexGuard, its ablations and the baselines on ADFA-LD.

Writes one JSON per (model, seed) to results/runs/ with test scores and labels,
and saves the VortexGuard checkpoint for seed 0 to results/checkpoints/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vortexguard.data import load_split, describe  # noqa: E402
from vortexguard.models import (VortexGuard, TransformerIDS, LSTMIDS, StaticSeccompRuleBaseline,  # noqa: E402
                                flux_features, count_params, FLUX_DIM)
from vortexguard.tokenizer import encode_adfa  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
CKPT = os.path.join(ROOT, "results", "checkpoints")
LMAX = 512


def featurize(traces):
    out = []
    for t in traces:
        tok = encode_adfa(t.seq[:LMAX]).tok
        out.append((tok, flux_features(tok), t.label))
    return out


def batches(items, bs, shuffle, rng, token_budget=4096):
    """Length-sorted batches of at most ``bs`` traces and ``token_budget`` padded tokens."""
    idx = np.argsort([len(x[0]) for x in items], kind="stable")
    chunks, cur = [], []
    for i in idx:
        L = len(items[i][0])
        if cur and (len(cur) + 1 > bs or (len(cur) + 1) * L > token_budget):
            chunks.append(np.array(cur))
            cur = []
        cur.append(i)
    if cur:
        chunks.append(np.array(cur))
    if shuffle:
        rng.shuffle(chunks)
    for ch in chunks:
        L = max(len(items[i][0]) for i in ch)
        tok = torch.zeros(len(ch), L, dtype=torch.long)
        flux = torch.zeros(len(ch), L, FLUX_DIM)
        mask = torch.zeros(len(ch), L)
        y = torch.zeros(len(ch))
        for j, i in enumerate(ch):
            n = len(items[i][0])
            tok[j, :n] = torch.from_numpy(items[i][0])
            flux[j, :n] = torch.from_numpy(items[i][1])
            mask[j, :n] = 1
            y[j] = items[i][2]
        yield ch, tok, flux, mask, y


@torch.no_grad()
def predict(model, items):
    model.eval()
    scores = np.zeros(len(items))
    extra = {"s_ssm": np.zeros(len(items)), "s_flux": np.zeros(len(items)), "gate": np.zeros(len(items))}
    for ch, tok, flux, mask, _ in batches(items, 64, False, None):
        out = model(tok, flux, mask)
        scores[ch] = out["logit"].numpy()
        for k in extra:
            if k in out:
                extra[k][ch] = out[k].numpy()
    return scores, extra


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def build(name):
    if name == "VortexGuard":
        return VortexGuard(mode="full")
    if name.startswith("VG-"):
        return VortexGuard(mode={"VG-SSMOnly": "ssm_only", "VG-FluxOnly": "flux_only", "VG-NoGate": "no_gate"}[name])
    if name == "TransformerIDS":
        return TransformerIDS()
    if name == "LSTMIDS":
        return LSTMIDS()
    raise ValueError(name)


def train_neural(name, seed, tr, va, te, epochs, patience=3):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = build(name)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    ys = np.array([x[2] for x in tr])
    pos_w = torch.tensor((ys == 0).sum() / max(1, (ys == 1).sum()))
    best, best_state, bad, log = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _, tok, flux, mask, y in batches(tr, 32, True, rng):
            out = model(tok, flux, mask)
            loss = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos_w)
            if isinstance(model, VortexGuard) and model.mode in ("full", "no_gate"):
                loss = loss + 0.5 * (F.binary_cross_entropy_with_logits(out["s_ssm"], y, pos_weight=pos_w)
                                     + F.binary_cross_entropy_with_logits(out["s_flux"], y, pos_weight=pos_w))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(y)
        sv, _ = predict(model, va)
        ap = auprc([x[2] for x in va], sv)
        log.append({"epoch": ep, "loss": tot / len(tr), "val_auprc": ap})
        print(f"  {name} s{seed} ep{ep} loss={tot/len(tr):.4f} val_auprc={ap:.4f} t={time.time()-t0:.0f}s", flush=True)
        if ap > best + 1e-4:
            best, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    train_s = time.time() - t0
    model.load_state_dict(best_state)
    sv, _ = predict(model, va)
    t1 = time.time()
    st, extra = predict(model, te)
    infer_ms = 1000 * (time.time() - t1) / len(te)
    return model, sv, st, extra, {"train_s": train_s, "infer_ms_per_trace": infer_ms,
                                  "params": count_params(model), "epochs_run": len(log),
                                  "best_val_auprc": best, "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="StaticSeccompRuleBaseline,LSTMIDS,TransformerIDS,VortexGuard,VG-SSMOnly,VG-FluxOnly,VG-NoGate")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)

    sp = load_split()
    with open(os.path.join(ROOT, "results", "split.json"), "w") as fh:
        json.dump({"counts": describe(sp), "dropped_duplicates": sp.dropped_duplicates, "lmax": LMAX,
                   "protocol": "attack instances 1-5/6-7/8-10 per family; normal validation 30/70; exact-duplicate removal"},
                  fh, indent=1)
    print("featurizing...", flush=True)
    tr, va, te = featurize(sp.train), featurize(sp.val), featurize(sp.test)
    meta = {"y_val": [t.label for t in sp.val], "y_test": [t.label for t in sp.test],
            "family_test": [t.family for t in sp.test], "family_val": [t.family for t in sp.val],
            "len_test": [len(t.seq) for t in sp.test]}

    for name in args.models.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            path = os.path.join(RUNS, f"{name}_seed{seed}.json")
            if os.path.exists(path):
                print("skip", path)
                continue
            print(f"== {name} seed {seed}", flush=True)
            if name == "StaticSeccompRuleBaseline":
                if seed != 0:
                    continue                      # deterministic
                t0 = time.time()
                b = StaticSeccompRuleBaseline().fit([t.seq for t in sp.train], [t.label for t in sp.train])
                sv = np.array([b.score(t.seq) for t in sp.val])
                t1 = time.time()
                st = np.array([b.score(t.seq) for t in sp.test])
                info = {"train_s": t1 - t0, "infer_ms_per_trace": 1000 * (time.time() - t1) / len(st),
                        "params": 0, "allowlist_size": len(b.allow), "allowlist": sorted(b.allow)}
                extra = {}
            else:
                model, sv, st, extra, info = train_neural(name, seed, tr, va, te, args.epochs)
                if seed == 0:
                    torch.save(model.state_dict(), os.path.join(CKPT, f"{name}_seed0.pt"))
                extra = {k: v.tolist() for k, v in extra.items()}
            rec = {"model": name, "seed": seed, "score_val": list(map(float, sv)),
                   "score_test": list(map(float, st)), "extra_test": extra, "info": info, **meta}
            with open(path, "w") as fh:
                json.dump(rec, fh)
            print(f"   done {name} seed {seed}: {info.get('train_s', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()
