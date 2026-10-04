#!/usr/bin/env python3
"""Learned readout over the 77-way Choice probability vectors (no new inference).

Trains supervised readouts on Jev's own intent probabilities (jev_predictions.csv):
  - LR on log-probabilities (77 dims), C by stratified CV on train
  - LR on log-probs + the 16 probe values (93 dims)
  - ExtraTrees on log-probs (nonlinear comparator)
Compares against the direct argmax baseline (reproduced from the same file).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs" / "banking77"
OUT = RUNS / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    pred = pd.read_csv(RUNS / "jev_predictions.csv")
    feats = pd.read_csv(RUNS / "jev_features.csv")
    known = {"split", "row", "label", "text", "text_sha256"}
    probes = [c for c in feats.columns if c not in known]

    probs = [json.loads(s) for s in pred["probabilities_json"]]
    labels = sorted(probs[0].keys())
    assert all(sorted(p.keys()) == labels for p in probs[::500]), "inconsistent label order"
    P = np.array([[p[l] for l in labels] for p in probs], dtype=np.float64)
    X77 = np.log(np.clip(P, 1e-6, 1.0))
    y = pred["label"].to_numpy()
    is_train = (pred["split"] == "train").to_numpy()

    # join probes
    fkey = feats.set_index(["split", "row"])
    probe_mat = np.array([fkey.loc[(s, r), probes].to_numpy(float)
                          for s, r in zip(pred["split"], pred["row"])])

    # argmax baseline (on the test split, for comparability with the bundle metrics)
    te_mask = ~is_train
    argmax_idx = P.argmax(1)
    argmax_pred = np.array([labels[i] for i in argmax_idx])
    base = {
        "accuracy": float(accuracy_score(y[te_mask], argmax_pred[te_mask])),
        "macro_f1": float(f1_score(y[te_mask], argmax_pred[te_mask], average="macro", zero_division=0)),
        "top3": float(top_k_accuracy_score(y[te_mask], P[te_mask], k=3, labels=labels)),
    }
    print("argmax baseline (test):", json.dumps(base))

    X77tr, X77te = X77[is_train], X77[~is_train]
    X93tr = np.hstack([X77tr, probe_mat[is_train]])
    X93te = np.hstack([X77te, probe_mat[~is_train]])
    ytr, yte = y[is_train], y[~is_train]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=77)
    c_grid = [0.1, 1.0, 10.0]
    results = {"argmax_baseline": base, "test_rows": int((~is_train).sum())}

    def run_lr(name, Xtr, Xte):
        pipe = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=3000, solver="lbfgs"))])
        gs = GridSearchCV(pipe, {"m__C": c_grid}, scoring="f1_macro", cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        model = gs.best_estimator_
        preds = model.predict(Xte)
        proba = model.predict_proba(Xte)
        res = {
            "best_C": float(gs.best_params_["m__C"]),
            "cv_macro_f1": float(gs.best_score_),
            "test_macro_f1": float(f1_score(yte, preds, average="macro", zero_division=0)),
            "test_accuracy": float(accuracy_score(yte, preds)),
            "test_weighted_f1": float(f1_score(yte, preds, average="weighted", zero_division=0)),
            "test_top3": float(top_k_accuracy_score(yte, proba, k=3, labels=model.classes_)),
        }
        results[name] = res
        print(f"{name}: C*={res['best_C']} cv={res['cv_macro_f1']:.4f} "
              f"test macroF1={res['test_macro_f1']:.4f} acc={res['test_accuracy']:.4f} "
              f"top3={res['test_top3']:.4f}")

    run_lr("readout_lr_logprobs", X77tr, X77te)
    run_lr("readout_lr_logprobs_plus_probes", X93tr, X93te)

    et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                              n_jobs=-1, random_state=77)
    et.fit(X77tr, ytr)
    proba = et.predict_proba(X77te)
    preds = et.classes_[proba.argmax(1)]
    results["readout_extra_trees_logprobs"] = {
        "test_macro_f1": float(f1_score(yte, preds, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(yte, preds)),
        "test_weighted_f1": float(f1_score(yte, preds, average="weighted", zero_division=0)),
        "test_top3": float(top_k_accuracy_score(yte, proba, k=3, labels=et.classes_)),
    }
    print("readout_extra_trees_logprobs:", json.dumps(results["readout_extra_trees_logprobs"]))

    (OUT / "choice_readout.json").write_text(json.dumps(results, indent=2))
    print("written", OUT / "choice_readout.json")


if __name__ == "__main__":
    main()
