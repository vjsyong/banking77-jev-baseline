#!/usr/bin/env python3
"""Scaled linear models with CV-chosen regularization (no new inference).

Tests whether the ExtraTrees advantage over linear models is partly a feature-scale
artifact: StandardScaler -> LogisticRegression / LinearSVC with C chosen by
stratified 5-fold CV on the training split. Unscaled CV-tuned comparators isolate
the scaling effect from the tuning effect. ExtraTrees reference: macro-F1 0.5707.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent.parent
CSV = HERE / "runs" / "banking77" / "jev_features.csv"
OUT = HERE / "runs" / "banking77" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
KNOWN = {"split", "row", "label", "text", "text_sha256"}
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def main():
    df = pd.read_csv(CSV)
    probes = [c for c in df.columns if c not in KNOWN]
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    Xtr = tr[probes].to_numpy(float)
    ytr = tr["label"].to_numpy()
    Xte = te[probes].to_numpy(float)
    yte = te["label"].to_numpy()
    print(f"train {Xtr.shape} test {Xte.shape}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    results = {}
    for name, estimator, grid_key in [
        ("scaled_logistic_regression", Pipeline([("sc", StandardScaler()),
                                                 ("m", LogisticRegression(max_iter=5000, solver="lbfgs"))]), "m__C"),
        ("scaled_linear_svm", Pipeline([("sc", StandardScaler()), ("m", LinearSVC())]), "m__C"),
        ("unscaled_logistic_regression", LogisticRegression(max_iter=5000, solver="lbfgs"), "C"),
        ("unscaled_linear_svm", LinearSVC(), "C"),
    ]:
        gs = GridSearchCV(estimator, {grid_key: C_GRID}, scoring="f1_macro", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
        pred = best.predict(Xte)
        results[name] = {
            "best_C": float(gs.best_params_[grid_key]),
            "cv_macro_f1": float(gs.best_score_),
            "test_macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
            "test_accuracy": float(accuracy_score(yte, pred)),
            "test_weighted_f1": float(f1_score(yte, pred, average="weighted", zero_division=0)),
            "cv_table": {str(c): round(float(s), 4) for c, s in zip(C_GRID, gs.cv_results_["mean_test_score"])},
        }
        r = results[name]
        print(f"{name:32s} C*={r['best_C']:<5} cv={r['cv_macro_f1']:.4f} "
              f"test macroF1={r['test_macro_f1']:.4f} acc={r['test_accuracy']:.4f}")

    et = 0.5707
    best_scaled = max(results["scaled_logistic_regression"]["test_macro_f1"],
                      results["scaled_linear_svm"]["test_macro_f1"])
    best_unscaled_cv = max(results["unscaled_logistic_regression"]["test_macro_f1"],
                           results["unscaled_linear_svm"]["test_macro_f1"])
    results["_summary"] = {
        "extra_trees_reference_macro_f1": et,
        "best_linear_scaled_cv": best_scaled,
        "best_linear_unscaled_cv": best_unscaled_cv,
        "et_minus_best_scaled_pp": round(100 * (et - best_scaled), 2),
        "et_minus_best_unscaled_cv_pp": round(100 * (et - best_unscaled_cv), 2),
        "original_unscaled_fixed_C": {"lr": 0.4038, "svm": 0.4307},
    }
    print(json.dumps(results["_summary"], indent=2))
    (OUT / "scaled_linear.json").write_text(json.dumps(results, indent=2))
    print("written", OUT / "scaled_linear.json")


if __name__ == "__main__":
    main()
