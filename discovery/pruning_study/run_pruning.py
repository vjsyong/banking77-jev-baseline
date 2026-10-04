#!/usr/bin/env python3
"""Pruning study §3: original vs corrected pruning screens on frozen mixed-strict banks.

Conditions per seed (budget: 150 candidate-bank evaluations per seed, all
conditions counted; identical evaluations reused across conditions):
  (a) unpruned reference             — quality (CV) + latency-proxy reference
  (b) original pruning, clean replay — original rule (single-fold rolling screen,
      tolerance 0.01, whole-question removals with correct column mapping)
  (c) corrected screen, tol 0.005    — fixed reference: cv5(candidate) >= cv5(unpruned) - 0.005
  (d) corrected screen, tol 0.010    — same with 0.010
Plus a small column-level arm (<=3 evals/seed), reported separately.

Learner is frozen: StandardScaler -> LogisticRegression(C=1.0) (the deployed LR,
selected best-of-panel-CV on the complete unpruned bank; all seeds = lr).
Preprocessing refit within each training fold (pipeline does this).

Writes: snapshots/../paths/{bank_id}_{cond}.jsonl (full removal paths),
        pruning_results.json (summary), frozen_points.json (deployment points §4).
"""
import json
import sys
import time
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
STUDY = HERE / "discovery" / "pruning_study"
SNAP = STUDY / "snapshots"
PATHS = STUDY / "paths"
PATHS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE / "runs" / "banking77" / "discovery" / "pilot"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

BUDGET = 150
ORIG_TOL = 0.010   # original rule's rolling tolerance
FLOOR_ORIG = 8     # original rule's stop floor


def lr_pipeline():
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))])


def lr_cv5(X, y, folds):
    vals = []
    for tr, va in folds:
        m = lr_pipeline()
        m.fit(X[tr], y[tr])
        vals.append(f1_score(y[va], m.predict(X[va]), average="macro", zero_division=0))
    return float(np.mean(vals))


def lr_screen(X, y, fold):
    tr, va = fold
    m = lr_pipeline()
    m.fit(X[tr], y[tr])
    return float(f1_score(y[va], m.predict(X[va]), average="macro", zero_division=0))


def main():
    idx = pd.read_csv(HERE / "runs/banking77/discovery/smallregime_index.csv")["csv_index"].to_numpy()
    y = pd.read_csv(HERE / "runs/banking77/jev_predictions.csv").iloc[idx]["label"].to_numpy()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
                 .split(np.zeros(len(y)), y))
    fold0 = folds[0]

    results = {}
    frozen_points = {}
    for bank_id in ("mixed-strict-s0.3", "mixed-strict-s0.7", "mixed-strict-s1.0"):
        snap = json.loads((SNAP / f"{bank_id}.json").read_text())
        X = np.load(SNAP / f"{bank_id}.npz")["X"]
        probes = snap["probes"]
        cols = [tuple(c) for c in snap["column_order"]]
        groups = []  # column indices per probe
        j = 0
        for p in probes:
            k = len(p["options"]) if p.get("format") == "choice" else 1
            groups.append(list(range(j, j + k)))
            j += k
        print(f"\n== {bank_id}: {len(probes)} probes, {len(cols)} columns ==", flush=True)
        budget = {"used": 0, "reused": 0, "cap": BUDGET}
        cache = {}

        def eval_cand(metric, kept_probe_idx, tag):
            key = (metric, tuple(kept_probe_idx))
            if key in cache:
                budget["reused"] += 1
                return cache[key], "reused"
            if budget["used"] >= budget["cap"]:
                return None, "budget"
            keep_cols = [c for gi in kept_probe_idx for c in groups[gi]]
            Xc = X[:, keep_cols]
            val = lr_cv5(Xc, y, folds) if metric == "cv5" else lr_screen(Xc, y, fold0)
            cache[key] = val
            budget["used"] += 1
            return val, "fresh"

        ref_cv5 = lr_cv5(X, y, folds)
        ref_screen = lr_screen(X, y, fold0)
        print(f"   reference: cv5 {ref_cv5:.4f} | screen {ref_screen:.4f} | budget {BUDGET}", flush=True)

        def run_screen(kind, tol):
            """kind: 'orig' (rolling, floor) or 'fixed' (vs unpruned reference)."""
            kept = list(range(len(probes)))
            steps, cur_baseline = [], ref_screen
            removed_order = []
            stop_reason = "no permitted removal remains"
            passes = 0
            while len(kept) > (FLOOR_ORIG if kind == "orig" else 0):
                passes += 1
                accepted_this_pass = False
                i = 0
                while i < len(kept):
                    cand = kept[:i] + kept[i + 1:]
                    metric = "screen" if kind == "orig" else "cv5"
                    val, src = eval_cand(metric, cand, f"{kind}-tol{tol}")
                    if val is None:
                        stop_reason = "budget exhausted"
                        break
                    if kind == "orig":
                        ok = val >= cur_baseline - tol
                        threshold_txt = f"rolling>={cur_baseline - tol:.4f}"
                    else:
                        ok = val >= ref_cv5 - tol
                        threshold_txt = f"fixed>={ref_cv5 - tol:.4f}"
                    steps.append({"step": len(steps) + 1, "pass": passes,
                                  "candidate_removes": probes[kept[i]]["id"],
                                  "remaining_after": len(kept) - 1,
                                  "columns_after": sum(len(groups[g]) for g in cand),
                                  "metric": metric, "value": round(val, 4),
                                  "threshold": threshold_txt, "accept": bool(ok), "src": src})
                    if ok:
                        removed_order.append(probes[kept[i]]["id"])
                        kept = cand
                        accepted_this_pass = True
                        if kind == "orig":
                            cur_baseline = val
                        continue  # do not advance i after a removal
                    i += 1
                if stop_reason == "budget exhausted":
                    break
                if not accepted_this_pass:
                    break
            return kept, steps, removed_order, stop_reason

        cond_results = {}
        for kind, tol, name in (("orig", ORIG_TOL, "b_original_clean"),
                                ("fixed", 0.005, "c_corrected_tol005"),
                                ("fixed", 0.010, "d_corrected_tol010")):
            t0 = time.perf_counter()
            kept, steps, removed, stop_reason = run_screen(kind, tol)
            final_cv5 = lr_cv5(X[:, [c for g in kept for c in groups[g]]], y, folds)
            with (PATHS / f"{bank_id}_{name}.jsonl").open("w") as f:
                for st in steps:
                    f.write(json.dumps(st) + "\n")
            cond_results[name] = {
                "removed": removed, "n_removed": len(removed),
                "kept_probes": [probes[g]["id"] for g in kept],
                "kept_n": len(kept),
                "kept_columns": sum(len(groups[g]) for g in kept),
                "final_cv5": round(final_cv5, 4),
                "delta_vs_unpruned_pp": round(100 * (final_cv5 - ref_cv5), 2),
                "n_evaluated": len(steps), "stop": stop_reason,
                "wall_s": round(time.perf_counter() - t0, 1),
            }
            print(f"   {name}: removed {len(removed)}, kept {len(kept)} probes "
                  f"({cond_results[name]['kept_columns']} cols), cv5 {final_cv5:.4f} "
                  f"(Δ {100*(final_cv5-ref_cv5):+.2f}pp) [{len(steps)} evals, {stop_reason}]", flush=True)

        # column-level arm (separate; <=3 evals/seed, counted)
        col_arm = []
        for gi, p in enumerate(probes):
            if p.get("format") != "choice" or len(p["options"]) < 3 or len(col_arm) >= 3:
                continue
            # drop the lowest-variance option column of this choice probe
            opt_vars = [float(X[:, c].var()) for c in groups[gi]]
            drop_local = int(np.argmin(opt_vars))
            keep_cols = [c for c in range(X.shape[1]) if c != groups[gi][drop_local]]
            if budget["used"] >= budget["cap"]:
                col_arm.append({"probe": p["id"], "dropped_option": p["options"][drop_local],
                                "status": "budget-skipped"})
                continue
            val = lr_cv5(X[:, keep_cols], y, folds)
            budget["used"] += 1
            col_arm.append({"probe": p["id"], "dropped_option": p["options"][drop_local],
                            "cv5_after": round(val, 4),
                            "delta_pp": round(100 * (val - ref_cv5), 2),
                            "status": "evaluated"})
            print(f"   col-arm: drop '{p['options'][drop_local]}' from {p['id']} -> cv5 {val:.4f} "
                  f"(Δ {100*(val-ref_cv5):+.2f}pp)", flush=True)

        results[bank_id] = {"reference": {"cv5": round(ref_cv5, 4), "screen_fold0": round(ref_screen, 4),
                                          "n_probes": len(probes), "n_columns": len(cols)},
                            "conditions": cond_results, "column_arm": col_arm,
                            "budget_used": budget["used"], "budget_reused": budget["reused"],
                            "wording_verified": True, "snapshot_cv": snap["snapshot_cv_best"]}
        fz = frozen_points.setdefault(bank_id, {"reference": {"cv5": round(ref_cv5, 4)}})
        for name in ("c_corrected_tol005", "d_corrected_tol010"):
            kept_ids = cond_results[name]["kept_probes"]
            fz[name] = {"kept_probes": [next(p for p in probes if p["id"] == pid) for pid in kept_ids],
                        "kept_n": cond_results[name]["kept_n"],
                        "kept_columns": cond_results[name]["kept_columns"],
                        "cv5": cond_results[name]["final_cv5"]}

    (STUDY / "pruning_results.json").write_text(json.dumps(results, indent=2))
    (STUDY / "frozen_points.json").write_text(json.dumps(frozen_points, indent=2))
    print("\nwritten pruning_results.json + frozen_points.json")

    print("\n=== SUMMARY ===")
    for bank_id, r in results.items():
        ref = r["reference"]
        print(f"{bank_id}: ref cv5 {ref['cv5']:.4f} ({ref['n_probes']}p/{ref['n_columns']}c)")
        for name, c in r["conditions"].items():
            print(f"   {name:20s}: {c['kept_n']:2d}p/{c['kept_columns']:2d}c | cv5 {c['final_cv5']:.4f} "
                  f"| Δ {c['delta_vs_unpruned_pp']:+.2f}pp | evals {c['n_evaluated']}")
        print(f"   budget used {r['budget_used']}/{BUDGET} (reused {r['budget_reused']})")


if __name__ == "__main__":
    main()
