#!/usr/bin/env python3
"""Small-label regime references: TF-IDF, Choice readout, manual probes on 1,000 labels.

Stratified 1,000-row training sample (seed 77); all references rebuilt on those
labels only and evaluated on the full official test split (3,080 rows).
Saves the sampled row index for reuse by the discovery pilot.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs" / "banking77"
OUT = RUNS / "discovery"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 77
N_SMALL = 1000
CLIP = (1e-6, 1.0)


def metrics(y, preds, proba=None, labels=None):
    m = {"macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
         "accuracy": float(accuracy_score(y, preds))}
    if proba is not None and labels is not None:
        m["top3"] = float(top_k_accuracy_score(y, proba, k=3, labels=labels))
    return m


def fit_lr_cv(X, y, Xte, yte):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    gs = GridSearchCV(Pipeline([("sc", StandardScaler()),
                                ("m", LogisticRegression(max_iter=3000, solver="lbfgs"))]),
                      {"m__C": [0.1, 1.0, 10.0]}, scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(X, y)
    model = gs.best_estimator_
    return metrics(yte, model.predict(Xte), model.predict_proba(Xte), model.classes_), float(gs.best_score_), float(gs.best_params_["m__C"])


def main():
    pred = pd.read_csv(RUNS / "jev_predictions.csv")
    feats = pd.read_csv(RUNS / "jev_features.csv")
    known = {"split", "row", "label", "text", "text_sha256"}
    probes = [c for c in feats.columns if c not in known]

    probs = [json.loads(s) for s in pred["probabilities_json"]]
    labels = sorted(probs[0].keys())
    P = np.array([[p[l] for l in labels] for p in probs], dtype=float)
    y = pred["label"].to_numpy()
    te = (pred["split"] == "test").to_numpy()
    tr_idx = np.where(~te)[0]

    small_idx, _ = train_test_split(tr_idx, train_size=N_SMALL, stratify=y[tr_idx],
                                    random_state=SEED)
    small_idx = np.sort(small_idx)
    print(f"small sample: {len(small_idx)} train rows")

    # save index for the pilot
    pd.DataFrame({"csv_index": small_idx,
                  "row": pred.iloc[small_idx]["row"].to_numpy()}).to_csv(
        OUT / "smallregime_index.csv", index=False)

    Xlogp = np.log(np.clip(P, *CLIP))
    Xsmall = Xlogp[small_idx]
    ysmall = y[small_idx]
    print(f"Xsmall {Xsmall.shape}")

    # 1. TF-IDF + LR (bundle config), trained on the small texts
    texts_small = pred.iloc[small_idx]["text"].tolist()
    texts_test = pred[te]["text"].tolist()
    vect = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 strip_accents="unicode", max_features=300_000)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, max_features=300_000)),
    ], n_jobs=1)
    Xtr_t = vect.fit_transform(texts_small)
    Xte_t = vect.transform(texts_test)
    clf = LogisticRegression(C=4.0, max_iter=1000, solver="lbfgs")
    clf.fit(Xtr_t, ysmall)
    tfidf_m = metrics(y[te], clf.predict(Xte_t))
    print("tfidf_1k:", tfidf_m)

    # 2. Choice readout (frozen protocol), trained on the small labels
    readout_m, cv_r, c_r = fit_lr_cv(Xsmall, ysmall, Xlogp[te], y[te])
    print(f"readout_1k: {readout_m} (C*={c_r}, cv={cv_r:.4f})")

    # 3. Manual 16 probes -> panel on the small labels
    fkey = feats.set_index(["split", "row"])
    probe_all = np.array([[fkey.loc[(s, r), c] for c in probes]
                          for s, r in zip(pred["split"], pred["row"])], dtype=float)
    lr_m, cv_p, c_p = fit_lr_cv(probe_all[small_idx], ysmall, probe_all[te], y[te])
    svm = LinearSVC(C=10.0)
    svm.fit(StandardScaler().fit_transform(probe_all[small_idx]), ysmall)
    svm_m = metrics(y[te], svm.predict(StandardScaler().fit_transform(probe_all[te])))
    et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                              n_jobs=-1, random_state=SEED)
    et.fit(probe_all[small_idx], ysmall)
    et_m = metrics(y[te], et.predict(probe_all[te]), et.predict_proba(probe_all[te]), et.classes_)
    print(f"probes16_1k: LR {lr_m} | SVM {svm_m['macro_f1']:.4f} | ET {et_m}")

    # 4. Direct argmax (no training)
    argmax_pred = np.array(labels)[P[te].argmax(1)]
    direct_m = metrics(y[te], argmax_pred, P[te], labels)
    print("direct_argmax:", direct_m)

    result = {
        "n_small": int(N_SMALL), "seed": SEED,
        "tfidf_lr_1k": tfidf_m,
        "readout_lr_1k": {**readout_m, "C": c_r, "cv_macro_f1": cv_r},
        "probes16_1k": {"lr": {**lr_m, "C": c_p, "cv_macro_f1": cv_p},
                        "svm": svm_m, "extra_trees": et_m},
        "direct_argmax": direct_m,
        "note": "trained on 1,000 stratified train rows; evaluated on the full 3,080-row test split",
    }
    (OUT / "smallregime_refs.json").write_text(json.dumps(result, indent=2))
    print("written", OUT / "smallregime_refs.json")


if __name__ == "__main__":
    main()
