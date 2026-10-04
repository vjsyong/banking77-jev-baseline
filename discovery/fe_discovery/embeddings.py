#!/usr/bin/env python3
"""Frozen sentence-embeddings reference (brief §7): bge-small-en-v1.5 via fastembed.

CPU ONNX inference; full inference cost measured and counted. Cache: npz keyed by
(model revision, sha256 of ordered texts).
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

HERE = Path("/home/xrim/banking77-jev-baseline")
CACHE = HERE / "runs" / "clinc150" / "fe_discovery" / "emb_cache"
CACHE.mkdir(parents=True, exist_ok=True)
MODEL = "BAAI/bge-small-en-v1.5"


def encode(texts, model=MODEL):
    """(embeddings, seconds) with on-disk cache; returns n x 384 float32."""
    key = hashlib.sha256((model + "||" + "||".join(texts)).encode()).hexdigest()[:32]
    f = CACHE / f"{key}.npz"
    if f.exists():
        return np.load(f)["emb"], 0.0
    from fastembed import TextEmbedding
    t0 = time.perf_counter()
    m = TextEmbedding(model)
    emb = np.array(list(m.embed(texts)), dtype=np.float32)
    dt = time.perf_counter() - t0
    np.savez_compressed(f, emb=emb)
    return emb, dt


def cv_eval(emb, y, folds, C=2.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    scores = []
    for f in range(5):
        tr = np.where(folds != f)[0]
        va = np.where(folds == f)[0]
        sc = StandardScaler().fit(emb[tr])
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(sc.transform(emb[tr]), y[tr])
        scores.append(float(f1_score(y[va], clf.predict(sc.transform(emb[va])),
                                     average="macro", zero_division=0)))
    return {"cv_macro_f1": float(np.mean(scores)), "fold_scores": [round(s, 4) for s in scores]}


def fit_full(emb, y, C=2.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(emb)
    clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(sc.transform(emb), y)
    return sc, clf
