#!/usr/bin/env python3
"""Deployable readout: TinyJev Choice probabilities -> 77-way intent prediction.

Frozen bundle `banking77-readout-v1/` contains:
  readout_lr.joblib  sklearn Pipeline(StandardScaler -> LogisticRegression)
  manifest.json      intent order, clip constants, upstream model revisions, metrics
  labels.json        the 77 intents in exact feature order

Usage:
  import sys; sys.path.insert(0, "<repo>/readout")
  from readout_pipeline import Banking77Readout
  ro = Banking77Readout.load("<repo>/readout/banking77-readout-v1")

  # input: {intent: probability} (any key order), or ndarray (77,) / (n, 77)
  out = ro.predict({"card_arrival": 0.9, ...all 77...})
  # -> {"label", "confidence", "probabilities": {intent: p}, "top3": [...]}
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np


class Banking77Readout:
    def __init__(self, model, labels: list[str], clip_lo: float, clip_hi: float, manifest: dict):
        if sorted(labels) != sorted(model.classes_.tolist()):
            raise ValueError("bundle labels do not match model classes")
        self.model = model
        self.labels = [str(x) for x in labels]
        self.clip_lo = float(clip_lo)
        self.clip_hi = float(clip_hi)
        self.manifest = manifest

    @classmethod
    def load(cls, bundle_dir) -> "Banking77Readout":
        bundle = Path(bundle_dir)
        manifest = json.loads((bundle / "manifest.json").read_text())
        labels = json.loads((bundle / "labels.json").read_text())
        model = joblib.load(bundle / "readout_lr.joblib")
        clip = manifest["input"]["clip"]
        return cls(model, labels, clip[0], clip[1], manifest)

    # ---- input handling ----
    def _to_matrix(self, probs) -> tuple[np.ndarray, bool]:
        """Returns (raw probability matrix (n, 77), was_single)."""
        single = False
        if isinstance(probs, dict):
            probs, single = [probs], True
        elif isinstance(probs, np.ndarray) and probs.ndim == 1:
            probs, single = probs[None, :], True
        elif isinstance(probs, np.ndarray) and probs.ndim == 2:
            pass
        elif isinstance(probs, (list, tuple)):
            if probs and isinstance(probs[0], dict):
                single = False
            else:
                arr = np.asarray(probs, dtype=float)
                if arr.ndim == 1:
                    arr, single = arr[None, :], True
                probs = arr
        else:
            raise TypeError(f"unsupported input type {type(probs)!r}")

        if isinstance(probs, np.ndarray):
            raw = np.asarray(probs, dtype=float)
            if raw.ndim == 1:
                raw = raw[None, :]
            if raw.shape[1] != len(self.labels):
                raise ValueError(f"expected {len(self.labels)} columns, got {raw.shape[1]}")
        else:
            rows = []
            for d in probs:
                missing = [l for l in self.labels if l not in d]
                if missing:
                    raise ValueError(f"missing intents: {missing[:5]}"
                                     + (" ..." if len(missing) > 5 else ""))
                rows.append([float(d[l]) for l in self.labels])
            raw = np.array(rows, dtype=float)
        return raw, single

    def features(self, probs) -> np.ndarray:
        """log(clip(p, lo, hi)) in the frozen intent order."""
        raw, _ = self._to_matrix(probs)
        return np.log(np.clip(raw, self.clip_lo, self.clip_hi))

    # ---- prediction ----
    def predict(self, probs):
        raw, single = self._to_matrix(probs)
        X = np.log(np.clip(raw, self.clip_lo, self.clip_hi))
        labels = self.model.predict(X)
        proba = self.model.predict_proba(X)
        classes = [str(c) for c in self.model.classes_]
        out = []
        for i in range(X.shape[0]):
            row = proba[i]
            k = min(3, len(classes))
            order = np.argsort(row)[::-1][:k]
            out.append({
                "label": str(labels[i]),
                "confidence": float(row.max()),
                "probabilities": {classes[j]: float(row[j]) for j in range(len(classes))},
                "top3": [{"label": classes[j], "p": float(row[j])} for j in order],
            })
        return out[0] if single else out


if __name__ == "__main__":  # tiny self-check
    import sys
    ro = Banking77Readout.load(Path(__file__).resolve().parent / "banking77-readout-v1")
    rng = np.random.default_rng(0)
    v = rng.dirichlet(np.ones(77))
    a = ro.predict(v)
    b = ro.predict({l: p for l, p in zip(reversed(ro.labels), reversed(v))})
    assert a["label"] == b["label"], "key order must not matter"
    print("self-check ok:", a["label"], round(a["confidence"], 4))
