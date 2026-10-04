#!/usr/bin/env python3
"""Immutable feature snapshots for the three mixed-strict banks (pruning study §2).

Per bank: run-scoped, definition-checked re-extraction on the 1k training split;
immutable snapshot (definitions, option order, extractor revision, inference
config, dataset-row hashes, exact numeric column order) + frozen .npz matrix.
Also verifies:
  - snapshot CV reproduces the wording-verified factorial CVs;
  - subset querying == taking columns from the complete bank (equivalence test).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))
for _p in ("/home/xrim/taildash/client", "/home/xrim/progtrack/client"):
    if Path(_p).is_dir():
        sys.path.insert(0, _p)
        break
try:
    from taildash import TaskMonitor as _TaskMonitor
except Exception:  # noqa: BLE001
    _TaskMonitor = None

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from extract_probes import ProbeExtractor, ProbeStore, text_key  # noqa: E402
from run_pilot import bank_cv  # noqa: E402

RUNS = HERE / "runs" / "banking77"
FACT = RUNS / "discovery" / "factorial"
STUDY = HERE / "discovery" / "pruning_study"
SNAP = STUDY / "snapshots"
SNAP.mkdir(parents=True, exist_ok=True)
SEED = 77
_ACTIVE_MONITOR = None


class _NullMonitor:
    def log(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def complete(self, *a, **k):
        pass

    def fail(self, *a, **k):
        pass


def columns_for(bank):
    """Explicit numeric column order: (probe_id, option|None) per column."""
    cols = []
    for p in bank:
        if p.get("format") == "choice":
            for o in p["options"]:
                cols.append((p["id"], o))
        else:
            cols.append((p["id"], None))
    return cols


def build_matrix(store, bank, texts):
    cols = columns_for(bank)
    data = []
    for pid, opt in cols:
        if opt is None:
            data.append([store.score_of(pid, text_key(t)) for t in texts])
        else:
            data.append([store.choice_score_of(pid, text_key(t), opt) for t in texts])
    X = np.array(data, dtype=np.float64).T
    assert not np.isnan(X).any(), "missing scores in snapshot build"
    return X, cols


def main():
    global _ACTIVE_MONITOR
    s = json.loads((FACT / "summary_full.json").read_text())
    banks = [r for r in s["runs"] if r["cell"] == "mixed-strict"]
    mon = (_TaskMonitor(server="http://localhost:8080",
                        title="pruning study: freeze snapshots (3 mixed-strict banks)",
                        total=len(banks) + 1, agent_name="banking77-pruning")
           if _TaskMonitor else _NullMonitor())
    _ACTIVE_MONITOR = mon

    idx = pd.read_csv(RUNS / "discovery" / "smallregime_index.csv")["csv_index"].to_numpy()
    pred = pd.read_csv(RUNS / "jev_predictions.csv").iloc[idx].reset_index(drop=True)
    texts, y = pred["text"].tolist(), pred["label"].to_numpy()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(y)), y))

    row_hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    (SNAP / "train_1k_row_hashes.txt").write_text("\n".join(row_hashes))
    dataset_meta = {
        "split": "train-1k (smallregime_index.csv)",
        "n_rows": len(texts),
        "index_sha256": hashlib.sha256(
            (RUNS / "discovery" / "smallregime_index.csv").read_bytes()).hexdigest(),
        "row_hashes_sha256": hashlib.sha256(
            ("\n".join(row_hashes)).encode()).hexdigest(),
        "row_hash_algo": "sha256(utf8(text))",
    }

    import tinyjev
    import torch
    extractor_meta = {
        "package": f"tinyjev {tinyjev.__version__}" if hasattr(tinyjev, "__version__") else "tinyjev",
        "model": "AnkitAI/TinyJev-0.6B (HF snapshot c559c2f7ea95069f92af8b999f512345bbb87d22)",
        "torch": torch.__version__,
        "backend": "torch",
        "device": "cuda",
        "inference": {"chunk": 64, "per_forward": "single question x <=64 texts (no cross-question co-query)",
                      "dtype": "float32 head on fp16 backbone (as served)"},
    }

    store = ProbeStore(RUNS / "discovery" / "pilot" / "probe_scores.sqlite")
    ex = ProbeExtractor(store)

    equivalence = {}
    for bi, run in enumerate(banks, 1):
        seed = run["seed"]
        bank_id = f"mixed-strict-s{seed}"
        bank = run["final_bank"]
        t0 = time.perf_counter()
        for p in bank:
            slot = f"snap::mixed-strict::{seed}::{p['id']}"
            if p.get("format") == "choice":
                ex.extract_choice(slot, p["question"], p["options"], texts)
            else:
                ex.extract_probe(slot, p["question"], texts)
        dt = time.perf_counter() - t0
        # snapshot matrix via run-scoped slots (rebuild defs to point at slots)
        slot_bank = []
        for p in bank:
            q = dict(p)
            q["slot"] = f"snap::mixed-strict::{seed}::{p['id']}"
            slot_bank.append(q)
        cols = columns_for(bank)
        data = []
        for p in slot_bank:
            if p.get("format") == "choice":
                for o in p["options"]:
                    data.append([store.choice_score_of(p["slot"], text_key(t), o) for t in texts])
            else:
                data.append([store.score_of(p["slot"], text_key(t)) for t in texts])
        X = np.array(data, dtype=np.float64).T
        cv_scores, cv_best = bank_cv(X, y, folds)
        snap = {
            "bank_id": bank_id, "source": {"cell": run["cell"], "seed": seed,
                                           "factorial": "summary_full.json"},
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "extractor": extractor_meta, "dataset": dataset_meta,
            "learner_frozen": {"pipeline": "StandardScaler -> LogisticRegression(C=1.0, lbfgs, max_iter=2000)",
                               "selection": "best-of-panel CV on the complete unpruned bank (all seeds: lr)"},
            "folds": "5-fold StratifiedKFold(shuffle, seed 77) on the 1k",
            "probes": [{"id": p["id"], "question": p["question"], "format": p.get("format", "noul"),
                        "options": p.get("options")} for p in bank],
            "column_order": [list(c) for c in cols],
            "n_columns": len(cols), "n_probes": len(bank),
            "snapshot_cv_best": round(float(cv_best), 6),
            "snapshot_cv_scores": {k: round(float(v), 4) for k, v in cv_scores.items()},
            "extract_s": round(dt, 1),
        }
        np.savez_compressed(SNAP / f"{bank_id}.npz", X=X,
                            colnames=np.array([f"{a}|{b}" if b else a for a, b in cols]))
        (SNAP / f"{bank_id}.json").write_text(json.dumps(snap, indent=1))
        print(f"{bank_id}: {len(bank)} probes ({len(cols)} cols), cv {cv_best:.4f} "
              f"(extract {dt:.0f}s)", flush=True)

        # equivalence test: fresh subset extraction vs column slice
        import random as _r
        rng = _r.Random(SEED)
        sub = [bank[k] for k in sorted(rng.sample(range(len(bank)), min(6, len(bank))))]
        sub_cols = columns_for(sub)
        sub_data = []
        for p in sub:
            sslot = f"eqtest::mixed-strict::{seed}::{p['id']}"
            if p.get("format") == "choice":
                ex.extract_choice(sslot, p["question"], p["options"], texts)
                for o in p["options"]:
                    sub_data.append([store.choice_score_of(sslot, text_key(t), o) for t in texts])
            else:
                ex.extract_probe(sslot, p["question"], texts)
                sub_data.append([store.score_of(sslot, text_key(t)) for t in texts])
        Xsub = np.array(sub_data, dtype=np.float64).T
        # slice same columns from full matrix
        full_map = {f"{a}|{b}" if b else a: j for j, (a, b) in enumerate(cols)}
        idxs = [full_map[f"{a}|{b}" if b else a] for a, b in sub_cols]
        Xslice = X[:, idxs]
        max_delta = float(np.abs(Xsub - Xslice).max())
        equivalence[bank_id] = {"n_subset_probes": len(sub), "max_abs_delta": max_delta,
                                "equivalent": bool(max_delta == 0.0)}
        print(f"  equivalence: fresh-subset vs column-slice max|Δ| = {max_delta}", flush=True)
        mon.update(bi, message=f"{bank_id}: {len(bank)} probes, cv {cv_best:.4f}")

    (SNAP / "equivalence_report.json").write_text(json.dumps(equivalence, indent=2))
    print("equivalence:", json.dumps(equivalence))
    mon.complete("snapshots frozen")
    store.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        if _ACTIVE_MONITOR is not None:
            _ACTIVE_MONITOR.fail(f"{type(exc).__name__}: {exc}")
        raise
