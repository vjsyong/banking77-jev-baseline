#!/usr/bin/env python3
"""Final test-split evaluation for the factorial + paired effect analysis.

Extracts every final bank's probes on the untouched test split (3,080), scores
each run (panel fit on the 1k; best-CV model reports), and computes the paired
per-seed effects the review asked for:
  representation: mixed - noul   (within each selection)
  selection     : pareto - strict (within each representation)
Registered with taildash before running.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))

for _p in ("/home/xrim/taildash/client", "/home/xrim/progtrack/client"):
    if Path(_p).is_dir():
        sys.path.insert(0, _p)
        break
try:
    from taildash import TaskMonitor as _TaskMonitor
except Exception:  # noqa: BLE001
    _TaskMonitor = None

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from extract_probes import ProbeExtractor, ProbeStore  # noqa: E402
from run_pilot import bank_cv, make_panel  # noqa: E402

RUNS = HERE / "runs" / "banking77"
FACT = RUNS / "discovery" / "factorial"

_ACTIVE_MONITOR = None


class _NullMonitor:
    def log(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def complete(self, *a, **k):
        pass

    def fail(self, *a, **k):
        pass


def main():
    global _ACTIVE_MONITOR
    summary = json.loads((FACT / "summary_full.json").read_text())
    runs = summary["runs"]
    mon = (_TaskMonitor(server="http://localhost:8080",
                        title=f"banking77 factorial eval ({len(runs)} banks on test)",
                        total=len(runs) + 1, agent_name="banking77-factorial-eval")
           if _TaskMonitor else _NullMonitor())
    _ACTIVE_MONITOR = mon
    mon.log(f"eval started: {len(runs)} runs, test split 3,080 rows")

    idx = pd.read_csv(RUNS / "discovery" / "smallregime_index.csv")["csv_index"].to_numpy()
    pred = pd.read_csv(RUNS / "jev_predictions.csv")
    train = pred.iloc[idx].reset_index(drop=True)
    test = pred[pred["split"] == "test"].reset_index(drop=True)
    texts_1k, y = train["text"].tolist(), train["label"].to_numpy()
    texts_te, yte = test["text"].tolist(), test["label"].to_numpy()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
                 .split(np.zeros(len(y)), y))

    store = ProbeStore(RUNS / "discovery" / "pilot" / "probe_scores.sqlite")
    ex = ProbeExtractor(store)

    unique = {}
    for run in runs:
        for p in run["final_bank"]:
            unique[p["id"]] = p
    mon.log(f"unique probes across final banks: {len(unique)}; extracting on test")
    print(f"unique probes: {len(unique)}", flush=True)
    t0 = time.perf_counter()
    for i, (pid, p) in enumerate(unique.items(), 1):
        dt = ex.extract_choice(pid, p["question"], p["options"], texts_te) \
            if p.get("format") == "choice" else ex.extract_probe(pid, p["question"], texts_te)
        if i % 10 == 0 or i == len(unique):
            print(f"  extracted {i}/{len(unique)} probes ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"test extraction done in {time.perf_counter()-t0:.0f}s", flush=True)

    results = []
    for i, run in enumerate(runs, 1):
        bank = run["final_bank"]
        X1k, _ = ex.matrix_mixed(bank, texts_1k)
        Xte, _ = ex.matrix_mixed(bank, texts_te)
        cv_scores, cv_best = bank_cv(X1k, y, folds)
        best_name = max(cv_scores, key=cv_scores.get)
        per_model = {}
        for name, m in make_panel().items():
            m.fit(X1k, y)
            p = m.predict(Xte)
            per_model[name] = round(float(f1_score(yte, p, average="macro", zero_division=0)), 4)
        model = make_panel()[best_name]
        model.fit(X1k, y)
        preds = model.predict(Xte)
        r = {"cell": run["cell"], "repr": run["repr"], "sel": run["sel"], "seed": run["seed"],
             "bank_size": run["bank_size"], "n_choices": run["n_choices"],
             "cv_best_1k": round(float(cv_best), 4),
             "best_model": best_name,
             "test_macro_f1": round(float(f1_score(yte, preds, average="macro", zero_division=0)), 4),
             "test_accuracy": round(float(accuracy_score(yte, preds)), 4),
             "test_per_model": per_model}
        results.append(r)
        print(f"{r['cell']:13s} s{r['seed']}: bank {r['bank_size']:2d}, test {r['test_macro_f1']:.4f} "
              f"(cv {r['cv_best_1k']:.4f})", flush=True)
        mon.update(i, message=f"{r['cell']} s{r['seed']}: test {r['test_macro_f1']:.4f}")

    # paired effects
    def get(repr_, sel, seed):
        return next(r for r in results if r["repr"] == repr_ and r["sel"] == sel and r["seed"] == seed)

    seeds = sorted({r["seed"] for r in results})
    effects = {"repr_mixed_minus_noul": {}, "sel_pareto_minus_strict": {}}
    for sel in ("strict", "pareto"):
        diffs = [get("mixed", sel, s)["test_macro_f1"] - get("noul", sel, s)["test_macro_f1"]
                 for s in seeds]
        effects["repr_mixed_minus_noul"][sel] = [round(d * 100, 2) for d in diffs]
        print(f"repr effect (mixed-noul, {sel}): {[round(d*100,2) for d in diffs]} pp")
    for repr_ in ("noul", "mixed"):
        diffs = [get(repr_, "pareto", s)["test_macro_f1"] - get(repr_, "strict", s)["test_macro_f1"]
                 for s in seeds]
        effects["sel_pareto_minus_strict"][repr_] = [round(d * 100, 2) for d in diffs]
        print(f"sel effect (pareto-strict, {repr_}): {[round(d*100,2) for d in diffs]} pp")

    cell_means = {}
    for cell in sorted({r["cell"] for r in results}):
        sub = [r for r in results if r["cell"] == cell]
        cell_means[cell] = {
            "mean_test_macro_f1": round(float(np.mean([r["test_macro_f1"] for r in sub])), 4),
            "mean_bank_size": round(float(np.mean([r["bank_size"] for r in sub])), 2),
            "mean_choices": round(float(np.mean([r["n_choices"] for r in sub])), 2),
        }
    out = {"runs": results, "paired_effects_pp": effects, "cell_means": cell_means,
           "refs": json.loads((RUNS / "discovery" / "smallregime_refs.json").read_text())}
    (FACT / "test_eval_factorial.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(cell_means, indent=2))
    print("written", FACT / "test_eval_factorial.json")
    mon.complete(f"eval done: {len(results)} banks scored")
    store.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        if _ACTIVE_MONITOR is not None:
            _ACTIVE_MONITOR.fail(f"{type(exc).__name__}: {exc}")
        raise
