#!/usr/bin/env python3
"""Timing bench: TF-IDF cache build + LR fits for CLINC150 (150 classes).

Determines the frozen solver config and informs the candidate-eval cap.
"""
import sys
import time
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery" / "fe_discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))

import numpy as np  # noqa: E402

from learner import FoldTFIDF, eval_bank, load_seed, _hstack  # noqa: E402

seed = load_seed(11)
texts, y, folds = seed["texts_d"], seed["y_d"], seed["folds"]
print(f"seed 11: {len(texts)} discovery rows, {len(np.unique(y))} classes", flush=True)

t0 = time.perf_counter()
tf = FoldTFIDF(texts, folds)
print(f"FoldTFIDF build (5 folds, fit+transform): {time.perf_counter()-t0:.1f}s", flush=True)
vec, mat = tf.mats[0]
print(f"  fold0 matrix: {mat.shape}, nnz {mat.nnz/1e6:.2f}M", flush=True)

# --- solver comparison on fold 0 ---
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

tr = np.where(folds != 0)[0]
va = np.where(folds == 0)[0]
Xtr, Xva = mat[tr], mat[va]
ytr, yva = y[tr], y[va]

import scipy.sparse as sp  # noqa: E402
rng = np.random.default_rng(0)
sem = rng.random((len(y), 8))

for name, kwargs in [
    ("lbfgs n1600", dict(solver="lbfgs", max_iter=1600)),
    ("lbfgs n400", dict(solver="lbfgs", max_iter=400)),
    ("liblinear ovo n400", dict(solver="liblinear", max_iter=400)),
    ("saga n300", dict(solver="saga", max_iter=300)),
]:
    t0 = time.perf_counter()
    clf = LogisticRegression(C=1.0, **kwargs)
    clf.fit(Xtr, ytr)
    tfit = time.perf_counter() - t0
    tp = time.perf_counter()
    pred = clf.predict(Xva)
    tpred = time.perf_counter() - tp
    f1 = f1_score(yva, pred, average="macro", zero_division=0)
    print(f"{name:20s}: fit {tfit:6.1f}s | predict {tpred:5.2f}s | fold0 macro-F1 {f1:.4f}",
          flush=True)
    if tfit > 900:
        print("   (too slow; stopping comparisons)", flush=True)
        break

# also: matrix with 8 dense cols attached (joint shape)
Xtrj = _hstack(Xtr, sem[tr])
Xvaj = _hstack(Xva, sem[va])
t0 = time.perf_counter()
clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=400)
clf.fit(Xtrj, ytr)
print(f"joint(+8 dense) lbfgs n400: fit {time.perf_counter()-t0:.1f}s", flush=True)
