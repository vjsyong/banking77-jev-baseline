#!/usr/bin/env python3
"""Final test-split evaluation of pilot final banks (plus cost accounting).

For each (policy, seed) run in summary_full.json: extract its bank's probes on the
3,080 test texts (cached), fit the learner panel on the 1k earlier-set, report
test macro-F1/accuracy for all panel models and the best-CV pick; account test
extraction cost per probe.
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
sys.path.insert(0, str(HERE / "discovery" / "pilot"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from extract_probes import ProbeExtractor, ProbeStore  # noqa: E402
from run_pilot import make_panel, bank_cv  # noqa: E402

RUNS = HERE / "runs" / "banking77"
PILOT = RUNS / "discovery" / "pilot"


def main():
    summary = json.loads((PILOT / "summary_full.json").read_text())
    idx = pd.read_csv(RUNS / "discovery" / "smallregime_index.csv")["csv_index"].to_numpy()
    pred = pd.read_csv(RUNS / "jev_predictions.csv")
    train = pred.iloc[idx].reset_index(drop=True)
    test = pred[pred["split"] == "test"].reset_index(drop=True)
    texts_1k, y = train["text"].tolist(), train["label"].to_numpy()
    texts_te, yte = test["text"].tolist(), test["label"].to_numpy()
    print(f"train {len(texts_1k)} / test {len(texts_te)}", flush=True)

    store = ProbeStore(PILOT / "probe_scores.sqlite")
    ex = ProbeExtractor(store)

    # unique probes across all final banks
    unique = {}
    for run in summary["runs"]:
        for p in run["final_bank"]:
            unique[p["id"]] = p["question"]
    print(f"unique probes across final banks: {len(unique)}", flush=True)
    extract_ms = {}
    for pid, q in unique.items():
        t0 = time.perf_counter()
        dt = ex.extract_probe(pid, q, texts_te)
        extract_ms[pid] = round(1000 * (time.perf_counter() - t0) / len(texts_te), 3)
        print(f"  {pid}: {extract_ms[pid]} ms/text", flush=True)

    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
                 .split(np.zeros(len(y)), y))
    results = []
    for run in summary["runs"]:
        bank = run["final_bank"]
        ids = [p["id"] for p in bank]
        X1k = ex.matrix(ids, texts_1k)
        Xte = ex.matrix(ids, texts_te)
        cv_scores, cv_best = bank_cv(X1k, y, folds)
        best_name = max(cv_scores, key=cv_scores.get)
        model = make_panel()[best_name]
        model.fit(X1k, y)
        preds = model.predict(Xte)
        per_model = {}
        for name, m in make_panel().items():
            m.fit(X1k, y)
            p = m.predict(Xte)
            per_model[name] = {"macro_f1": round(float(f1_score(yte, p, average="macro",
                                                                zero_division=0)), 4),
                               "accuracy": round(float(accuracy_score(yte, p)), 4)}
        bank_ms = sum(extract_ms[p["id"]] for p in bank)
        new_ids = [p["id"] for p in bank if p["id"] not in {q["id"] for q in json.loads((HERE / 'probes.json').read_text())}]
        results.append({
            "policy": run["policy"], "seed": run["seed"],
            "bank_size": len(bank), "new_probes": len(new_ids),
            "cv_best_1k": round(float(cv_best), 4), "cv_scores_1k": {k: round(float(v), 4) for k, v in cv_scores.items()},
            "best_model": best_name, "best_cv_pick": best_name,
            "test_best": {"macro_f1": round(float(f1_score(yte, preds, average="macro", zero_division=0)), 4),
                          "accuracy": round(float(accuracy_score(yte, preds)), 4)},
            "test_per_model": per_model,
            "extract_ms_per_text": round(bank_ms, 3),
            "final_bank_ids": ids,
        })
        print(f"{run['policy']} s{run['seed']}: bank {len(bank)} | test best {results[-1]['test_best']} "
              f"| {bank_ms:.2f} ms/text extraction", flush=True)

    # aggregates per policy
    agg = {}
    for policy in summary["policies"]:
        sub = [r for r in results if r["policy"] == policy]
        agg[policy] = {
            "n_seeds": len(sub),
            "mean_test_macro_f1": round(float(np.mean([r["test_best"]["macro_f1"] for r in sub])), 4),
            "mean_bank_size": round(float(np.mean([r["bank_size"] for r in sub])), 2),
            "mean_extract_ms": round(float(np.mean([r["extract_ms_per_text"] for r in sub])), 3),
            "mean_new_probes": round(float(np.mean([r["new_probes"] for r in sub])), 2),
        }

    out = {"runs": results, "per_policy": agg, "probe_extract_ms": extract_ms,
           "refs": json.loads((RUNS / "discovery" / "smallregime_refs.json").read_text())}
    (PILOT / "test_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(agg, indent=2))
    print("written", PILOT / "test_eval.json")
    store.close()


if __name__ == "__main__":
    main()
