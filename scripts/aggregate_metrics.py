"""Aggregate every experiment into results/metrics.json (single source of truth for the paper).

Thresholds for F1/precision/recall are fixed on the validation split (max-F1),
except for the static seccomp baseline, whose decision rule is fixed by
construction (any syscall outside the profile is a violation).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FAMILIES = ["Adduser", "Hydra_FTP", "Hydra_SSH", "Java_Meterpreter", "Meterpreter", "Web_Shell"]
MAIN = ["StaticSeccompRuleBaseline", "LSTMIDS", "TransformerIDS", "VortexGuard"]
ABL = ["VG-SSMOnly", "VG-FluxOnly", "VG-NoGate", "VortexGuard"]
RNG = np.random.RandomState(0)


def val_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[np.argmax(f[:-1])])


def fpr_at_tpr(y, s, tpr=0.99):
    y, s = np.asarray(y), np.asarray(s)
    thr = np.quantile(s[y == 1], 1 - tpr, method="lower")
    return float(np.mean(s[y == 0] >= thr))


def metrics_for(run):
    y, s = np.array(run["y_test"]), np.array(run["score_test"])
    if run["model"] == "StaticSeccompRuleBaseline":
        thr = 0.5
    else:
        thr = val_threshold(np.array(run["y_val"]), np.array(run["score_val"]))
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fam = np.array(run["family_test"])
    per_fam = {}
    for f in FAMILIES:
        m = fam == f
        sub_y = np.concatenate([y[fam == "Normal"], np.ones(m.sum(), int)])
        sub_s = np.concatenate([s[fam == "Normal"], s[m]])
        per_fam[f] = {"recall": float(pred[m].mean()), "fpr_at_99tpr": fpr_at_tpr(sub_y, sub_s),
                      "n": int(m.sum())}
    return {"macro_f1": float(f1_score(y, pred, average="macro")), "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)), "fpr_at_99tpr": fpr_at_tpr(y, s),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "fpr": fp / max(1, fp + tn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": thr, "per_family": per_fam,
            "params": run["info"].get("params", 0), "train_s": run["info"].get("train_s", 0.0),
            "infer_ms_per_trace": run["info"].get("infer_ms_per_trace", 0.0)}


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": len(a)}


def summarize(model, runs):
    ms = [metrics_for(r) for r in runs]
    keys = ["macro_f1", "auroc", "auprc", "fpr_at_99tpr", "precision", "recall", "fpr", "params", "train_s",
            "infer_ms_per_trace", "tp", "fp", "fn", "tn"]
    out = {k: mean_std([m[k] for m in ms]) for k in keys}
    out["per_family"] = {f: {k: mean_std([m["per_family"][f][k] for m in ms]) for k in ("recall", "fpr_at_99tpr")}
                         for f in FAMILIES}
    out["seeds"] = [r["seed"] for r in runs]
    return out


def paired_bootstrap(runs_a, runs_b, metric, B=2000):
    """95% CI of metric(A) - metric(B) on the test set, resampling traces; seeds paired by index."""
    fn = {"auprc": average_precision_score, "auroc": roc_auc_score}[metric]
    diffs = []
    pairs = list(zip(runs_a, runs_b)) if len(runs_b) > 1 else [(a, runs_b[0]) for a in runs_a]
    y = np.array(runs_a[0]["y_test"])
    n = len(y)
    for _ in range(B):
        idx = RNG.randint(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        d = [fn(y[idx], np.array(a["score_test"])[idx]) - fn(y[idx], np.array(b["score_test"])[idx]) for a, b in pairs]
        diffs.append(np.mean(d))
    diffs = np.array(diffs)
    point = np.mean([fn(y, np.array(a["score_test"])) - fn(y, np.array(b["score_test"])) for a, b in pairs])
    return {"diff": float(point), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_le_0": float(np.mean(diffs <= 0))}


def load_runs():
    runs = {}
    for p in sorted(glob.glob(os.path.join(RES, "runs", "*.json"))):
        r = json.load(open(p))
        runs.setdefault(r["model"], []).append(r)
    for v in runs.values():
        v.sort(key=lambda r: r["seed"])
    return runs


def containment_summary():
    path = os.path.join(RES, "containment_trials.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    A, Bn = d["attacks"], d["benign"]

    def pct(v, q):
        return float(np.percentile(v, q)) if len(v) else None

    def stats(v):
        v = [x for x in v if x is not None]
        return {"p50": pct(v, 50), "p90": pct(v, 90), "p99": pct(v, 99), "max": float(max(v)) if v else None,
                "mean": float(np.mean(v)) if v else None, "n": len(v)}

    out = {"calibration": d["calibration"], "n_attack_trials": len(A), "n_benign_runs": len(Bn)}
    for tag, sub in (("all", A), ("standard", [a for a in A if not a["mimicry"]]),
                     ("mimicry", [a for a in A if a["mimicry"]])):
        cont = [a for a in sub if a["contained_ns"]]
        e2e = [(a["contained_ns"] - a["hard_event_ns"]) / 1e6 for a in cont]
        dec = [(a["hard_decision_ns"] - a["hard_event_ns"]) / 1e6 for a in cont]
        # detection delay relative to the first malicious stage (recon)
        delay = [(a["hard_event_ns"] - a["stages"]["recon"]) / 1e6 for a in cont if "recon" in a["stages"]]
        stage_at = {}
        for a in cont:
            st = sorted((t, n) for n, t in a["stages"].items())
            reached = [n for t, n in st if t <= a["hard_event_ns"]]
            stage_at[reached[-1] if reached else "none"] = stage_at.get(reached[-1] if reached else "none", 0) + 1
        out[tag] = {
            "n": len(sub), "contained": len(cont), "containment_rate": len(cont) / max(1, len(sub)),
            "soft_tightened": sum(1 for a in sub if a["soft_event_ns"]),
            "exfil_completed": sum(1 for a in sub if a["exfil_done"]),
            "exfil_blocked_by_seccomp": sum(1 for a in sub if a["exfil_blocked"]),
            "bytes_exfiltrated_total": int(sum(a["exfil_bytes_received"] for a in sub)),
            "trials_with_any_bytes_exfiltrated": sum(1 for a in sub if a["exfil_bytes_received"] > 0),
            "payload_survived_rollback": sum(1 for a in cont if a["leftover_payload"]),
            "e2e_containment_ms": stats(e2e), "event_to_decision_ms": stats(dec),
            "freeze_ms": stats([a["freeze_ms"] for a in cont]), "rollback_ms": stats([a["rollback_ms"] for a in cont]),
            "swap_ms": stats([a["swap_ms"] for a in cont]), "remount_ms": stats([a["remount_ms"] for a in cont]),
            "unmount_ms": stats([a["unmount_ms"] for a in cont]),
            "detection_delay_from_recon_ms": stats(delay), "stage_at_containment": stage_at,
            "events_scored": stats([a["n_events"] for a in sub]),
            "scoring_us_per_event": stats([1e6 * a["scoring_s"] / max(1, a["n_events"]) for a in sub]),
        }
    out["benign"] = {"n": len(Bn), "false_containments": sum(1 for b in Bn if b["contained_ns"]),
                     "false_tightenings": sum(1 for b in Bn if b["soft_event_ns"]),
                     "max_score": stats([b["max_score"] for b in Bn])}
    return out


def overhead_summary():
    path = os.path.join(RES, "overhead.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    cfg = d["configs"]

    def med(v):
        return float(np.median([x["workload_ms"] if isinstance(x, dict) else x for x in v]))
    nat = {wl: med(v) for wl, v in cfg["native"].items()}
    out = {"reps": d["reps"], "static_profile_size": d["static_profile_size"], "native_ms": nat, "configs": {}}
    for name, per in cfg.items():
        o = {wl: 100.0 * (med(v) - nat[wl]) / nat[wl] for wl, v in per.items()}
        rec = {"overhead_pct": o, "overhead_pct_mean": float(np.mean(list(o.values()))),
               "workload_ms": {wl: med(v) for wl, v in per.items()}}
        if isinstance(next(iter(per.values()))[0], dict):
            rec["verdict_lag_ms"] = {wl: float(np.median([x["verdict_lag_ms"] for x in v])) for wl, v in per.items()}
            rec["events"] = {wl: float(np.median([x["events"] for x in v])) for wl, v in per.items()}
            rec["scoring_us_per_event"] = {wl: float(np.median([1e6 * x["scoring_s"] / max(1, x["events"]) for x in v]))
                                           for wl, v in per.items()}
            # time-to-verdict overhead: workload end-to-end including scoring backlog
            ttv = {wl: 100.0 * (med(v) + rec["verdict_lag_ms"][wl] - nat[wl]) / nat[wl] for wl, v in per.items()}
            rec["time_to_verdict_overhead_pct"] = ttv
            rec["time_to_verdict_overhead_pct_mean"] = float(np.mean(list(ttv.values())))
        out["configs"][name] = rec
    return out


def main():
    runs = load_runs()
    split = json.load(open(os.path.join(RES, "split.json")))
    det = {m: summarize(m, runs[m]) for m in MAIN if m in runs}
    abl = {m: summarize(m, runs[m]) for m in ABL if m in runs}
    sig = {}
    if "VortexGuard" in runs:
        for b in ("LSTMIDS", "TransformerIDS", "StaticSeccompRuleBaseline"):
            if b in runs:
                sig[b] = {k: paired_bootstrap(runs["VortexGuard"], runs[b], k) for k in ("auprc", "auroc")}
    vg_gate = None
    if "VortexGuard" in runs:
        r = runs["VortexGuard"][0]
        g = np.array(r["extra_test"].get("gate", []))
        fam = np.array(r["family_test"])
        if len(g):
            vg_gate = {f: float(g[fam == f].mean()) for f in ["Normal"] + FAMILIES}
    scaling = json.load(open(os.path.join(RES, "scaling.json"))) if os.path.exists(os.path.join(RES, "scaling.json")) else None
    metrics = {"split": split, "detection": det, "ablation": abl, "significance": sig, "gate_by_family": vg_gate,
               "scaling": scaling, "overhead": overhead_summary(), "containment": containment_summary()}
    # headline numbers
    if "VortexGuard" in det:
        v = det["VortexGuard"]
        best_base = max((m for m in det if m != "VortexGuard"), key=lambda m: det[m]["macro_f1"]["mean"])
        metrics["headline"] = {
            "vg_macro_f1": v["macro_f1"]["mean"], "vg_auroc": v["auroc"]["mean"], "vg_auprc": v["auprc"]["mean"],
            "vg_fpr_at_99tpr": v["fpr_at_99tpr"]["mean"], "best_baseline": best_base,
            "best_baseline_macro_f1": det[best_base]["macro_f1"]["mean"],
            "macro_f1_gain_pts": 100 * (v["macro_f1"]["mean"] - det[best_base]["macro_f1"]["mean"]),
            "static_macro_f1": det.get("StaticSeccompRuleBaseline", {}).get("macro_f1", {}).get("mean"),
        }
    json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=1)
    print(json.dumps(metrics.get("headline", {}), indent=1))


if __name__ == "__main__":
    main()
