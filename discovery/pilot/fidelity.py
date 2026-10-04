#!/usr/bin/env python3
"""Concept-fidelity mini-audit for discovered probes.

Config: JSON list of
  {"probe_id": ..., "question": ..., "pos_intents": [...], "neg_intents": [...],
   "n_per_side": 18}
Samples messages from the full train split (search purity preserved: this audit is
validation only), extracts the probe (cached store), and scores vs the
intent-derived labels. Writes fidelity_report.json + a reviewable sheet.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from extract_probes import ProbeExtractor, ProbeStore, text_key  # noqa: E402

PILOT = HERE / "runs" / "banking77" / "discovery" / "pilot"
SEED = 77


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    d = load_dataset("PolyAI/banking77")
    names = d["train"].features["label"].names
    by_intent = {}
    for i, item in enumerate(d["train"]):
        by_intent.setdefault(names[int(item["label"])], []).append(item["text"])

    rng = np.random.default_rng(SEED)
    store = ProbeStore(PILOT / "probe_scores.sqlite")
    ex = ProbeExtractor(store)
    report, sheet = {}, []
    for c in cfg:
        pid, n = c["probe_id"], c.get("n_per_side", 18)
        pos_pool = [t for nme in c["pos_intents"] for t in by_intent.get(nme, [])]
        neg_pool = [t for nme in c["neg_intents"] for t in by_intent.get(nme, [])]
        pos = [pos_pool[j] for j in rng.choice(len(pos_pool), size=min(n, len(pos_pool)), replace=False)]
        neg = [neg_pool[j] for j in rng.choice(len(neg_pool), size=min(n, len(neg_pool)), replace=False)]
        texts = pos + neg
        y = np.array([1] * len(pos) + [0] * len(neg))
        ex.extract_probe(pid, c["question"], texts)
        scores = np.array([store.score_of(pid, text_key(t)) for t in texts])
        acc = float(np.mean((scores >= 0.5).astype(int) == y))
        auc = float(roc_auc_score(y, scores))
        sep = float(scores[y == 1].mean() - scores[y == 0].mean())
        report[pid] = {"question": c["question"], "n": len(texts),
                       "accuracy_at_0.5": round(acc, 3), "auc": round(auc, 3),
                       "separation": round(sep, 3),
                       "mean_pos": round(float(scores[y == 1].mean()), 3),
                       "mean_neg": round(float(scores[y == 0].mean()), 3)}
        print(f"{pid}: acc={acc:.3f} auc={auc:.3f} sep={sep:+.3f} "
              f"(pos {scores[y == 1].mean():.3f} / neg {scores[y == 0].mean():.3f})")
        for t, yy, s in zip(texts, y, scores):
            sheet.append({"probe_id": pid, "label": int(yy), "score": round(float(s), 3),
                          "text": t})
    (PILOT / "fidelity_report.json").write_text(json.dumps(report, indent=2))
    with (PILOT / "fidelity_sheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["probe_id", "label", "score", "text"])
        w.writeheader()
        w.writerows(sheet)
    print("written fidelity_report.json + fidelity_sheet.csv")
    store.close()


if __name__ == "__main__":
    main()
