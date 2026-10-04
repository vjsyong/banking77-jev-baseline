#!/usr/bin/env python3
"""Serving-latency protocol + frozen cost model (brief §10).

Primary workload: batched inference, batch size 64, one frozen hardware config.
End-to-end text->prediction: encoding, upstream forward passes, probability
processing, TF-IDF, downstream prediction. NO cached semantic outputs in the
benchmark (fresh extraction every measurement).

The linear cost model is fitted during development (BANKING77) and frozen; it
screens candidate banks for feasibility. Retained archive points get full
measurements via measure_batched().
"""
import time
from dataclasses import dataclass

import numpy as np
import torch

from tinyjev.families import softmax

BATCH_SIZE = 64
WARMUP_CHUNKS = 1


def raw_block(extractor, defs, texts, chunk: int = 64, collect_times: bool = False):
    """Dense log(clip) block for defs over texts, extracted fresh (no store).

    Returns (block, per_chunk_seconds) — block columns follow defs/option order.
    """
    fam, model, dev = extractor.fam, extractor.model, extractor.dev
    n = len(texts)
    width = sum(len(d["options"]) for d in defs)
    X = np.empty((n, width), dtype=np.float64)
    times = []
    col = 0
    for d in defs:
        criteria = {o["id"]: o["definition"] for o in d["options"]}
        q = {"id": "q", "type": "choice", "instructions": d["question"], "criteria": criteria}
        for s in range(0, n, chunk):
            part = texts[s:s + chunk]
            t0 = time.perf_counter()
            encs = [fam.encode({"id": f"x{i}", "state": t, "questions": [dict(q)]})
                    for i, t in enumerate(part)]
            rows = [e.prefix + e.rows[0] for e in encs]
            w = max(len(r) for r in rows)
            ids = torch.full((len(rows), w), fam.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(rows), w), dtype=torch.long)
            for i, r in enumerate(rows):
                ids[i, :len(r)] = torch.tensor(r)
                att[i, :len(r)] = 1
            with torch.inference_mode():
                hs = model(input_ids=ids.to(dev), attention_mask=att.to(dev),
                           use_cache=False).last_hidden_state.float().cpu().numpy()
            for j, e in enumerate(encs):
                probs = softmax(fam.logits([hs[j, :len(rows[j])]], e)["q"])
                keys = e.questions[0]["keys"]
                for oi, oid in enumerate(criteria):
                    X[s + j, col + oi] = np.log(min(max(float(probs[keys.index(oid)]), 1e-6), 1.0))
            if collect_times:
                times.append(time.perf_counter() - t0)
        col += len(d["options"])
    return X, times


def build_pipeline(texts_disc, y_disc, block_disc, c_value=None):
    """Deployment-shaped pipeline fit on all discovery rows (for latency runs)."""
    from learner import TFIDF_CHAR, TFIDF_WORD, _hstack, make_lr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import StandardScaler
    vec = FeatureUnion([("word", TfidfVectorizer(**TFIDF_WORD)),
                        ("char", TfidfVectorizer(**TFIDF_CHAR))]).fit(texts_disc)
    scaler = StandardScaler().fit(block_disc) if block_disc.shape[1] else None
    X = _hstack(vec.transform(texts_disc).tocsr(),
                scaler.transform(block_disc) if scaler is not None else None)
    C = c_value or 1.0
    clf = make_lr(C).fit(X, y_disc)
    return {"vec": vec, "scaler": scaler, "clf": clf, "C": C}


def measure_batched(extractor, pipe, defs, texts, batch=BATCH_SIZE):
    """End-to-end batched latency (fresh extraction): ms/text + throughput."""
    import scipy.sparse as sp
    # warmup (not timed)
    for _ in range(WARMUP_CHUNKS):
        block, _ = raw_block(extractor, defs, texts[:batch])
        _predict(pipe, texts[:batch], block, sp)
    t0 = time.perf_counter()
    for s in range(0, len(texts), batch):
        part = texts[s:s + batch]
        block, _ = raw_block(extractor, defs, part)
        _predict(pipe, part, block, sp)
    dt = time.perf_counter() - t0
    return {"ms_per_text": round(1000 * dt / len(texts), 3),
            "throughput": round(len(texts) / dt, 2), "total_s": round(dt, 2),
            "n_texts": len(texts)}


def measure_single(extractor, pipe, defs, texts, n=30, warmup=3):
    """Single-request p50/p95 (milliseconds), sequential, no caching."""
    import scipy.sparse as sp
    for t in texts[:warmup]:
        block, _ = raw_block(extractor, defs, [t])
        _predict(pipe, [t], block, sp)
    times = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        block, _ = raw_block(extractor, defs, [t])
        _predict(pipe, [t], block, sp)
        times.append(1000 * (time.perf_counter() - t0))
    times = np.sort(np.array(times))
    return {"p50_ms": round(float(np.percentile(times, 50)), 1),
            "p95_ms": round(float(np.percentile(times, 95)), 1),
            "n": n}


def _predict(pipe, texts, block, sp):
    X_t = pipe["vec"].transform(texts).tocsr()
    if pipe["scaler"] is not None and block.shape[1]:
        X = sp.hstack([X_t, sp.csr_matrix(pipe["scaler"].transform(block))], format="csr")
    else:
        X = X_t
    return pipe["clf"].predict(X)


@dataclass
class CostModel:
    """ms_per_text ~ b0 + b1 * n_questions + b2 * n_options_total."""
    b0: float
    b1: float
    b2: float
    model_id: str = "linear-v1"

    def predict_ms(self, n_questions: int, n_options_total: int) -> float:
        return self.b0 + self.b1 * n_questions + self.b2 * n_options_total

    def predict_bank_ms(self, defs) -> float:
        return self.predict_ms(len(defs), sum(len(d["options"]) for d in defs))

    @staticmethod
    def fit(points):
        """points: [(n_questions, n_options_total, ms_per_text)] -> CostModel"""
        import numpy as np
        A = np.array([[1.0, q, o] for q, o, _ in points])
        b = np.array([m for _, _, m in points])
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return CostModel(b0=float(coef[0]), b1=float(coef[1]), b2=float(coef[2]))

    def to_json(self):
        return {"b0": self.b0, "b1": self.b1, "b2": self.b2, "model_id": self.model_id}
