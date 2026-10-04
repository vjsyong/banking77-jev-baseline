#!/usr/bin/env python3
"""Downstream pipeline for the FE-discovery experiment.

Frozen design (brief §7):
  TF-IDF(text) + scaled Choice log-probability blocks -> LogisticRegression.
  - word (1,2) & char_wb (2,5) TF-IDF; sparse joint pipeline;
  - semantic block log(clip(p,1e-6,1.0)); scaling fit within training fold only;
  - LR with a small predefined C grid selected once on the initial (empty-bank)
    baseline per seed, then fixed for all arms/candidates;
  - vectorizer / IDF / scaling / classifier fit strictly within training folds.
"""
import json
import os
from pathlib import Path

import numpy as np

HERE = Path("/home/xrim/banking77-jev-baseline")
DATA = HERE / "data" / "clinc150" / "samples"

# frozen feature constants (declared, not tuned on outcomes)
TFIDF_WORD = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                  strip_accents="unicode", max_features=100_000)
TFIDF_CHAR = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                  max_features=100_000)
C_GRID = [4.0, 16.0, 64.0]
CLIP = (1e-6, 1.0)
LR_OVR_JOBS = int(os.environ.get("FE_LR_JOBS", "8"))


def make_lr(C):
    """Frozen learner factory: OvR liblinear (fast at 150 classes; bench-frozen)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    return OneVsRestClassifier(
        LogisticRegression(C=C, solver="liblinear", max_iter=400), n_jobs=LR_OVR_JOBS)


def load_seed(seed: int):
    flat = json.loads((DATA / "flat_rows.json").read_text())
    rec = json.loads((DATA / f"seed_{seed}.json").read_text())
    row = {r["row_id"]: r for r in flat}
    disc = [row[r] for r in rec["discovery_row_ids"]]
    conf = [row[r] for r in rec["confirmation_row_ids"]]
    texts_d = [r["text"] for r in disc]
    labels_d = np.array([r["label"] for r in disc])
    folds = np.zeros(len(disc), dtype=int)
    pos = {rid: j for j, rid in enumerate(rec["discovery_row_ids"])}
    for rid, f in rec["folds"].items():
        folds[pos[rid]] = f
    texts_c = [r["text"] for r in conf]
    labels_c = np.array([r["label"] for r in conf])
    return {"seed": seed, "texts_d": texts_d, "y_d": labels_d, "folds": folds,
            "texts_c": texts_c, "y_c": labels_c}


class FoldTFIDF:
    """Per-fold TF-IDF caches for one seed: vectorizers fit on fold-train only.

    mats[f] = (vectorizer, transformed_all_rows_csr)  (all rows, transform only)
    Disk-cached per (seed texts, folds, params) so parallel arm workers load in
    seconds instead of refitting.
    """

    def __init__(self, texts, folds, n_folds=5, cache_path=None):
        import hashlib
        import pickle
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
        self.n_folds = int(n_folds)
        self.folds = folds
        self.n = len(texts)
        self.mats = {}
        if cache_path is not None and Path(cache_path).exists():
            try:
                with open(cache_path, "rb") as fh:
                    blob = pickle.load(fh)
                if blob.get("key") == self._key(texts, folds, n_folds):
                    self.mats = blob["mats"]
                    return
            except Exception:
                pass  # rebuild on any cache trouble
        for f in range(self.n_folds):
            tr = np.where(folds != f)[0]
            vec = FeatureUnion([("word", TfidfVectorizer(**TFIDF_WORD)),
                                ("char", TfidfVectorizer(**TFIDF_CHAR))])
            vec.fit([texts[i] for i in tr])
            self.mats[f] = (vec, vec.transform(texts).tocsr())
        if cache_path is not None:
            try:
                with open(cache_path, "wb") as fh:
                    pickle.dump({"key": self._key(texts, folds, n_folds), "mats": self.mats}, fh)
            except Exception:
                pass

    @staticmethod
    def _key(texts, folds, n_folds):
        import hashlib
        h = hashlib.sha256()
        h.update(str(n_folds).encode())
        h.update(np.asarray(folds).tobytes())
        h.update(str(len(texts)).encode())
        for t in texts:
            h.update(t.encode("utf-8", "ignore"))
        return h.hexdigest()[:24]


def semantic_block(store, defs, row_keys, clip=CLIP):
    """Dense log(clip(p)) block over the given definitions' columns (row order kept)."""
    from extract_probes import text_key  # noqa: F401 (signature hint)
    cols = []
    for d in defs:
        for o in d["options"]:
            col = np.empty(len(row_keys), dtype=np.float64)
            for i, k in enumerate(row_keys):
                v = store.choice_score_of(d["slot"], k, o["id"])
                col[i] = np.log(min(max(v, clip[0]), clip[1]))
            cols.append(col)
    return np.array(cols, dtype=np.float64).T


def _hstack(tfidf_csr, sem_dense):
    import scipy.sparse as sp
    if sem_dense is None or sem_dense.shape[1] == 0:
        return tfidf_csr
    return sp.hstack([tfidf_csr, sp.csr_matrix(sem_dense)], format="csr")


def eval_bank(tfidf: FoldTFIDF, sem, y, C: float, want_oof=False, n_jobs=None):
    """5-fold CV macro-F1 for a bank (folds fixed by tfidf.folds).

    sem: dense semantic block aligned to row order (n x w) or None.
    Preprocessing (scaler) fit strictly within each training fold.
    Uses the frozen learner factory make_lr(C).
    """
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    classes = np.unique(y)
    oof = np.full(n, None, dtype=object)
    oof_proba = np.zeros((n, len(classes)), dtype=np.float32) if want_oof else None
    scores = []
    for f in range(tfidf.n_folds):
        tr = np.where(tfidf.folds != f)[0]
        va = np.where(tfidf.folds == f)[0]
        Xtr_full = tfidf.mats[f][1]
        if sem is not None and sem.shape[1]:
            scaler = StandardScaler().fit(sem[tr])
            Xtr = _hstack(Xtr_full[tr], scaler.transform(sem[tr]))
            Xva = _hstack(Xtr_full[va], scaler.transform(sem[va]))
        else:
            Xtr, Xva = Xtr_full[tr], Xtr_full[va]
        clf = make_lr(C)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xva)
        scores.append(float(f1_score(y[va], pred, average="macro", zero_division=0)))
        if want_oof:
            oof[va] = pred
            proba = clf.predict_proba(Xva)
            for j, c in enumerate(clf.classes_):
                oof_proba[va, np.where(classes == c)[0][0]] = proba[:, j]
    out = {"cv_macro_f1": float(np.mean(scores)),
           "fold_scores": [round(float(s), 4) for s in scores]}
    if want_oof:
        out["oof_pred"] = oof
        out["oof_proba"] = oof_proba
        out["classes"] = classes
    return out


def select_C(tfidf: FoldTFIDF, y, c_grid=None):
    """One-time C selection on the initial (empty-bank) baseline. Returns (C, table)."""
    c_grid = c_grid or C_GRID
    table = {}
    for C in c_grid:
        r = eval_bank(tfidf, None, y, C)
        table[C] = round(r["cv_macro_f1"], 4)
    best = max(table, key=table.get)
    return best, table
