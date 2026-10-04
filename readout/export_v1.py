#!/usr/bin/env python3
"""Train + freeze the v1 readout bundle (deterministic reproduction of the round-1 experiment).

Config frozen exactly as validated:
  features  = log(clip(p, 1e-6, 1.0)) over the 77 intents in sorted order
  model     = StandardScaler -> LogisticRegression(C=1.0, multinomial, lbfgs, max_iter=3000)
  C         = 1.0 chosen by 3-fold stratified CV (macro-F1) on the training split
  training  = official train rows from runs/banking77/jev_predictions.csv

Writes readout/banking77-readout-v1/{readout_lr.joblib,manifest.json,labels.json} and
verifies the frozen artifact reproduces the round-1 test metrics (0.9008 / 0.9006 / 0.9766).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sklearn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = REPO / "runs" / "banking77"
BUNDLE = HERE / "banking77-readout-v1"
CLIP = (1e-6, 1.0)
EXPECTED = {"macro_f1": 0.9008, "accuracy": 0.9006, "top3": 0.9766}


def versions() -> dict:
    out = {}
    try:
        r = subprocess.run([str(REPO / "venv-serve/bin/python"), "-c",
                            "import torch,transformers,tinyjev;"
                            "print(torch.__version__,transformers.__version__,tinyjev.__version__)"],
                           capture_output=True, text=True, timeout=120)
        t, tr, tj = r.stdout.split()
        out = {"torch": t, "transformers": tr, "tinyjev": tj}
    except Exception as exc:  # noqa: BLE001
        out = {"error": repr(exc)}
    return out


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    pred = pd.read_csv(RUNS / "jev_predictions.csv")
    probs = [json.loads(s) for s in pred["probabilities_json"]]
    labels = sorted(probs[0].keys())
    P = np.array([[p[l] for l in labels] for p in probs], dtype=np.float64)
    X = np.log(np.clip(P, CLIP[0], CLIP[1]))
    y = pred["label"].to_numpy()
    te = (pred["split"] == "test").to_numpy()

    model = Pipeline([("sc", StandardScaler()),
                      ("m", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs"))])
    model.fit(X[~te], y[~te])

    preds = model.predict(X[te])
    proba = model.predict_proba(X[te])
    metrics = {
        "macro_f1": float(f1_score(y[te], preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y[te], preds)),
        "top3": float(top_k_accuracy_score(y[te], proba, k=3, labels=model.classes_)),
    }
    print("reproduced metrics:", json.dumps(metrics))
    for k, v in EXPECTED.items():
        assert abs(metrics[k] - v) < 5e-4, f"{k}: {metrics[k]} != {v}"

    BUNDLE.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BUNDLE / "readout_lr.joblib")
    (BUNDLE / "labels.json").write_text(json.dumps(labels, indent=1))
    manifest = {
        "name": "banking77-readout-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "script": "readout/export_v1.py",
        "upstream": {
            "model": "TinyJev-0.6B", "hf_repo": "AnkitAI/TinyJev-0.6B",
            "hf_snapshot": "c559c2f7ea95069f92af8b999f512345bbb87d22",
            "extraction": "batched width-grouped in-process (tools/batched_extract.py, chunk=64)",
            "stack": versions(),
        },
        "input": {
            "type": "choice_probabilities",
            "intent_count": len(labels),
            "clip": list(CLIP),
            "transform": "log(clip(p, lo, hi)) in labels.json order",
        },
        "model": {
            "sklearn": sklearn.__version__,
            "pipeline": "StandardScaler -> LogisticRegression(solver=lbfgs, max_iter=3000)",
            "C": 1.0,
            "C_selected_by": "3-fold stratified CV macro-F1 on train, grid [0.1, 1, 10], seed 77",
            "cv_macro_f1": 0.8917,
        },
        "training": {"train_rows": int((~te).sum()), "test_rows": int(te.sum()),
                     "source": "runs/banking77/jev_predictions.csv (train split)"},
        "evaluation": metrics,
        "artifacts": {"readout_lr.joblib": "fitted sklearn pipeline",
                      "labels.json": "77 intents, exact feature order"},
    }
    (BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("bundle written:", BUNDLE)


if __name__ == "__main__":
    main()
