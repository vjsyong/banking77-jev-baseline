#!/usr/bin/env python3
"""Representation control: LR on one-hot direct predictions vs LR on 77 log-probs.

Identical folds and tuning budget (StandardScaler -> LogisticRegression,
3-fold stratified CV, seed 77, C in [0.1, 1, 10], scoring macro-F1).

Isolates two effects:
  supervision gain   = one-hot LR  - argmax
  representation gain = logprob LR - one-hot LR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs" / "banking77"
OUT = RUNS / "discovery"
OUT.mkdir(parents=True, exist_ok=True)
CLIP = (1e-6, 1.0)
C_GRID = [0.1, 1.0, 10.0]
SEED = 77


def fit_eval(Xtr, ytr, Xte, yte):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    gs = GridSearchCV(Pipeline([("sc", StandardScaler()),
                                ("m", LogisticRegression(max_iter=3000, solver="lbfgs"))]),
                      {"m__C": C_GRID}, scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)
    model = gs.best_estimator_
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)
    return model, {
        "best_C": float(gs.best_params_["m__C"]),
        "cv_macro_f1": float(gs.best_score_),
        "test_macro_f1": float(f1_score(yte, preds, average="macro", zero_division=0)),
        "test_accuracy": float(accuracy_score(yte, preds)),
        "test_top3": float(top_k_accuracy_score(yte, proba, k=3, labels=model.classes_)),
    }


def main():
    pred = pd.read_csv(RUNS / "jev_predictions.csv")
    probs = [json.loads(s) for s in pred["probabilities_json"]]
    labels = sorted(probs[0].keys())
    P = np.array([[p[l] for l in labels] for p in probs], dtype=float)
    y = pred["label"].to_numpy()
    te = (pred["split"] == "test").to_numpy()

    # one-hot of the argmax prediction
    onehot = np.zeros_like(P)
    onehot[np.arange(len(P)), P.argmax(1)] = 1.0
    X_logp = np.log(np.clip(P, *CLIP))

    # argmax baseline (test)
    argmax_pred = np.array(labels)[P.argmax(1)]
    base = {
        "macro_f1": float(f1_score(y[te], argmax_pred[te], average="macro")),
        "accuracy": float(accuracy_score(y[te], argmax_pred[te])),
        "top3": float(top_k_accuracy_score(y[te], P[te], k=3, labels=labels)),
    }
    print("argmax baseline:", json.dumps(base))

    _, onehot_res = fit_eval(onehot[~te], y[~te], onehot[te], y[te])
    print("one-hot LR      :", json.dumps(onehot_res))
    _, logp_res = fit_eval(X_logp[~te], y[~te], X_logp[te], y[te])
    print("logprob LR      :", json.dumps(logp_res))

    result = {
        "protocol": "StandardScaler->LR, 3-fold stratified CV seed 77, C grid [0.1,1,10]",
        "argmax_baseline": base,
        "onehot_lr": onehot_res,
        "logprob_lr": logp_res,
        "decomposition_pp": {
            "supervision_gain_macro_f1": round(100 * (onehot_res["test_macro_f1"] - base["macro_f1"]), 2),
            "representation_gain_macro_f1": round(100 * (logp_res["test_macro_f1"] - onehot_res["test_macro_f1"]), 2),
            "total_gain_macro_f1": round(100 * (logp_res["test_macro_f1"] - base["macro_f1"]), 2),
        },
    }
    (OUT / "control_onehot.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["decomposition_pp"], indent=2))
    print("written", OUT / "control_onehot.json")


if __name__ == "__main__":
    main()
