#!/usr/bin/env python3
"""Run-scoped re-verification of the mixed-pareto final banks.

Their wording-scoped CV deltas (vs recorded) were large (-6..-11pp), unlike every
other cell (<=0.6pp), indicating cross-run store value drift for their probes
during the search. This re-extracts each mixed-pareto bank's probes under fully
run-scoped slots (rv::<cell>::<seed>::<probe>) on the 1k and test splits and
reports definitive CV + test numbers for those three runs.
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

from extract_probes import ProbeExtractor, ProbeStore, text_key  # noqa: E402
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
    runs = [r for r in summary["runs"] if r["cell"] == "mixed-pareto"]
    mon = (_TaskMonitor(server="http://localhost:8080",
                        title="mixed-pareto run-scoped re-verification", total=len(runs),
                        agent_name="banking77-reverify")
           if _TaskMonitor else _NullMonitor())
    _ACTIVE_MONITOR = mon
    mon.log(f"re-verifying {len(runs)} mixed-pareto banks (run-scoped slots)")

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

    results = []
    for i, run in enumerate(runs, 1):
        pairs = []
        t0 = time.perf_counter()
        for p in run["final_bank"]:
            slot = f"rv::{run['cell']}::{run['seed']}::{p['id']}"
            if p.get("format") == "choice":
                ex.extract_choice(slot, p["question"], p["options"], texts_1k)
                ex.extract_choice(slot, p["question"], p["options"], texts_te)
            else:
                ex.extract_probe(slot, p["question"], texts_1k)
                ex.extract_probe(slot, p["question"], texts_te)
            pairs.append((slot, p))
        dt = time.perf_counter() - t0
        data1 = []
        datat = []
        for slot, p in pairs:
            if p.get("format") == "choice":
                for o in p.get("options") or []:
                    data1.append([store.choice_score_of(slot, text_key(t), o) for t in texts_1k])
                    datat.append([store.choice_score_of(slot, text_key(t), o) for t in texts_te])
            else:
                data1.append([store.score_of(slot, text_key(t)) for t in texts_1k])
                datat.append([store.score_of(slot, text_key(t)) for t in texts_te])
        X1k = np.array(data1, dtype=np.float64).T
        Xte = np.array(datat, dtype=np.float64).T
        cv_scores, cv_best = bank_cv(X1k, y, folds)
        best_name = max(cv_scores, key=cv_scores.get)
        model = make_panel()[best_name]
        model.fit(X1k, y)
        preds = model.predict(Xte)
        r = {"cell": run["cell"], "seed": run["seed"], "bank_size": run["bank_size"],
             "cv_recorded": run["final_cv_best"],
             "cv_run_scoped": round(float(cv_best), 4),
             "test_macro_f1_run_scoped": round(float(f1_score(yte, preds, average="macro",
                                                              zero_division=0)), 4),
             "test_accuracy": round(float(accuracy_score(yte, preds)), 4),
             "best_model": best_name, "extract_s": round(dt, 1)}
        results.append(r)
        print(f"mixed-pareto s{r['seed']}: bank {r['bank_size']} | cv run-scoped {r['cv_run_scoped']:.4f} "
              f"(recorded {r['cv_recorded']:.4f}) | test {r['test_macro_f1_run_scoped']:.4f} "
              f"| {dt:.0f}s", flush=True)
        mon.update(i, message=f"s{r['seed']}: test {r['test_macro_f1_run_scoped']:.4f}")

    (FACT / "mixed_pareto_reverify.json").write_text(json.dumps(results, indent=2))
    print("written", FACT / "mixed_pareto_reverify.json")
    mon.complete("mixed-pareto re-verification done")
    store.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        if _ACTIVE_MONITOR is not None:
            _ACTIVE_MONITOR.fail(f"{type(exc).__name__}: {exc}")
        raise
