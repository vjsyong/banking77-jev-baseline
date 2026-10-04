#!/usr/bin/env python3
"""Confidence threshold study from training-fold (OOF) predictions.

5-fold stratified OOF on the TRAIN split with the frozen readout config; reports
accuracy-vs-coverage for accept/defer at candidate thresholds; then applies the
selected thresholds to the TEST split using the frozen bundle (reporting only).

Writes runs/banking77/analysis/confidence_coverage.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from readout_pipeline import Banking77Readout  # noqa: E402

RUNS = REPO / "runs" / "banking77"
OUT = RUNS / "analysis" / "confidence_coverage.json"
CLIP = (1e-6, 1.0)


def main():
    pred = pd.read_csv(RUNS / "jev_predictions.csv")
    probs = [json.loads(s) for s in pred["probabilities_json"]]
    labels = sorted(probs[0].keys())
    P = np.array([[p[l] for l in labels] for p in probs], dtype=float)
    X = np.log(np.clip(P, CLIP[0], CLIP[1]))
    y = pred["label"].to_numpy()
    te = (pred["split"] == "test").to_numpy()

    Xtr, ytr = X[~te], y[~te]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    oof_conf = np.zeros(len(ytr))
    oof_correct = np.zeros(len(ytr), dtype=bool)
    for fold, (tr, va) in enumerate(skf.split(Xtr, ytr), 1):
        model = Pipeline([("sc", StandardScaler()),
                          ("m", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs"))])
        model.fit(Xtr[tr], ytr[tr])
        proba = model.predict_proba(Xtr[va])
        preds = model.classes_[proba.argmax(1)]
        oof_conf[va] = proba.max(1)
        oof_correct[va] = preds == ytr[va]
        print(f"fold {fold}: acc {oof_correct[va].mean():.4f}", flush=True)

    curve = []
    for tau in np.round(np.arange(0.0, 1.0001, 0.005), 3):
        m = oof_conf >= tau
        curve.append({"tau": float(tau), "coverage": round(float(m.mean()), 4),
                      "accuracy": round(float(oof_correct[m].mean()), 4) if m.sum() else None})

    targets = {}
    for target in (0.90, 0.95, 0.98):
        cand = [c for c in curve if c["accuracy"] is not None
                and c["accuracy"] >= target and c["coverage"] >= 0.2]
        targets[str(target)] = cand[0] if cand else None

    # test reporting with the frozen bundle
    ro = Banking77Readout.load(HERE / "banking77-readout-v1")
    out = ro.predict(P[te])
    test_conf = np.array([o["confidence"] for o in out])
    test_correct = np.array([o["label"] for o in out]) == y[te]
    test_rows = []
    for target in (0.90, 0.95, 0.98):
        t = targets[str(target)]
        if t:
            m = test_conf >= t["tau"]
            test_rows.append({"target": target, "tau": t["tau"],
                              "test_coverage": round(float(m.mean()), 4),
                              "test_accuracy_on_accepted":
                                  round(float(test_correct[m].mean()), 4) if m.sum() else None})

    result = {"oof_rows": int((~te).sum()), "test_rows": int(te.sum()),
              "curve": curve, "targets_oof": targets, "test_reporting": test_rows,
              "notes": "thresholds selected on train OOF (5-fold, seed 77); test reported only"}
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"targets_oof": targets, "test_reporting": test_rows}, indent=2))
    print("written", OUT)


if __name__ == "__main__":
    main()
