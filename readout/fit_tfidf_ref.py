#!/usr/bin/env python3
"""Refit the bundle's classical TF-IDF + LogisticRegression (exact config) and save it.

Config copied from banking77_baseline.py: word(1,2)+char(2,5) TF-IDF union,
min_df=2, sublinear_tf, max_features=300k each, strip_accents unicode; LR C=4.0,
max_iter=1000, lbfgs. Saves runs/banking77/tfidf_lr_refit.joblib for benchmarking
and verifies test metrics against the recorded values (0.9094 / 0.9091).
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from banking77_baseline import load_dataset_rows  # noqa: E402

OUT = HERE / "runs" / "banking77" / "tfidf_lr_refit.joblib"
EXPECTED = {"macro_f1": 0.9094, "accuracy": 0.9091}


def main():
    rows, _, _ = load_dataset_rows()
    tr, te = rows["train"], rows["test"]
    vect = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 strip_accents="unicode", max_features=300_000)),
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, max_features=300_000)),
    ], n_jobs=1)
    Xtr = vect.fit_transform([r["text"] for r in tr])
    Xte = vect.transform([r["text"] for r in te])
    ytr = np.asarray([r["label"] for r in tr])
    yte = np.asarray([r["label"] for r in te])
    clf = LogisticRegression(C=4.0, max_iter=1000, solver="lbfgs")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    m = {"macro_f1": float(f1_score(yte, pred, average="macro")),
         "accuracy": float(accuracy_score(yte, pred))}
    print("refit metrics:", m)
    for k, v in EXPECTED.items():
        assert abs(m[k] - v) < 5e-4, f"{k}: {m[k]} != {v}"
    joblib.dump({"vect": vect, "clf": clf}, OUT, compress=3)
    print("saved", OUT)


if __name__ == "__main__":
    main()
