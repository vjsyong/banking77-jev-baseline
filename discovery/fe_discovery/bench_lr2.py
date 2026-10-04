#!/usr/bin/env python3
"""Solver benchmark v2 at CLINC150 scale (150 classes, 2250 discovery rows).

Compares multinomial lbfgs settings vs OneVsRest liblinear for fit time and
macro-F1 on fold0. Determines the frozen solver config for all runs.
"""
import sys
import time
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery" / "fe_discovery"))

import numpy as np  # noqa: E402
from learner import FoldTFIDF, load_seed  # noqa: E402

seed = load_seed(11)
texts, y, folds = seed["texts_d"], seed["y_d"], seed["folds"]
tf = FoldTFIDF(texts, folds)
mat = tf.mats[0][1]
tr = np.where(folds != 0)[0]
va = np.where(folds == 0)[0]
Xtr, Xva = mat[tr], mat[va]
ytr, yva = y[tr], y[va]
print(f"CLINC fold0: X {Xtr.shape}, {len(np.unique(y))} classes", flush=True)

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

def run(name, est):
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    dt = time.perf_counter() - t0
    f1 = f1_score(yva, est.predict(Xva), average="macro", zero_division=0)
    print(f"{name:42s}: fit {dt:7.1f}s | f1 {f1:.4f}", flush=True)
    return dt

run("lbfgs C=1 max_iter=1000 tol=1e-4 (frozen)",
    LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000))
run("lbfgs C=4 max_iter=1000 tol=1e-4",
    LogisticRegression(C=4.0, solver="lbfgs", max_iter=1000))
run("lbfgs C=4 max_iter=300 tol=1e-3",
    LogisticRegression(C=4.0, solver="lbfgs", max_iter=300, tol=1e-3))
run("lbfgs C=8 max_iter=300 tol=1e-3",
    LogisticRegression(C=8.0, solver="lbfgs", max_iter=300, tol=1e-3))
run("ovr-liblinear C=4 n_jobs=10",
    OneVsRestClassifier(LogisticRegression(C=4.0, solver="liblinear", max_iter=400), n_jobs=10))
run("ovr-liblinear C=16 n_jobs=10",
    OneVsRestClassifier(LogisticRegression(C=16.0, solver="liblinear", max_iter=400), n_jobs=10))
