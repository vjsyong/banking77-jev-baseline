#!/usr/bin/env python3
"""CLINC150 data prep for the example-guided discovery experiment.

In-scope classification only (150 intents; OOS excluded). For each of five
preregistered seeds: sample 20 examples per intent from the official training
partition -> 15 discovery (2,250 rows) + 5 confirmation (750 rows). The official
test partition (30/intent, 4,500 rows) is the registered final evaluation set.

Outputs (data/clinc150/samples/):
  flat_rows.json         - canonical flat index of all in-scope rows (hash basis)
  seed_{s}.json          - discovery/confirmation row ids + 5-fold assignment
  PREP_MANIFEST.json     - hashes, counts, constants
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path("/home/xrim/banking77-jev-baseline/data/clinc150")
OUT = HERE / "samples"
OUT.mkdir(parents=True, exist_ok=True)

# Preregistered constants (declared before any confirmation/test access).
SEEDS = [11, 23, 37, 53, 71]
N_DISCOVERY = 15
N_CONFIRMATION = 5
N_FOLDS = 5


def main():
    raw = json.loads((HERE / "data_full.json").read_text())
    data_hash = hashlib.sha256((HERE / "data_full.json").read_bytes()).hexdigest()

    train = raw["train"]
    test = raw["test"]
    intents = sorted({label for _, label in train if label != "oos"})
    assert len(intents) == 150, f"expected 150 intents, got {len(intents)}"
    assert all(label != "oos" for _, label in test), "test split unexpectedly contains oos"

    # canonical flat index: train rows then test rows; (id, text, label, split)
    flat = []
    by_intent_train = {}
    for i, (text, label) in enumerate(train):
        if label == "oos":
            continue
        rid = f"train:{i}"
        flat.append({"row_id": rid, "text": text, "label": label, "split": "train"})
        by_intent_train.setdefault(label, []).append(rid)
    for i, (text, label) in enumerate(test):
        rid = f"test:{i}"
        flat.append({"row_id": rid, "text": text, "label": label, "split": "test"})
    counts = {}
    for r in flat:
        counts[(r["split"], r["label"])] = counts.get((r["split"], r["label"]), 0) + 1
    assert all(counts[("train", L)] == 100 for L in intents), "train not 100/intent in-scope"
    assert all(counts[("test", L)] == 30 for L in intents), "test not 30/intent"
    row_index = {r["row_id"]: j for j, r in enumerate(flat)}
    label_of = {r["row_id"]: r["label"] for r in flat}
    (OUT / "flat_rows.json").write_text(json.dumps(flat))

    manifest = {
        "source": "clinc/oos-eval data_full.json",
        "data_full_sha256": data_hash,
        "intents": intents,
        "n_intents": len(intents),
        "seeds": SEEDS,
        "n_discovery_per_intent": N_DISCOVERY,
        "n_confirmation_per_intent": N_CONFIRMATION,
        "n_folds": N_FOLDS,
        "flat_rows_sha256": hashlib.sha256((OUT / "flat_rows.json").read_bytes()).hexdigest(),
    }

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        disc, conf = [], []
        for L in intents:
            rows = np.array(by_intent_train[L])
            pick = rng.choice(len(rows), size=N_DISCOVERY + N_CONFIRMATION, replace=False)
            d = rows[np.sort(pick[:N_DISCOVERY])]
            c = rows[np.sort(pick[N_DISCOVERY:])]
            disc.extend(d.tolist())
            conf.extend(c.tolist())
        assert len(disc) == 2250 and len(conf) == 750
        # stratifed fold assignment on discovery rows
        y = np.array([label_of[rid] for rid in disc])
        fold_of = {}
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for f, (_, va) in enumerate(skf.split(np.zeros(len(disc)), y)):
            for j in va:
                fold_of[disc[j]] = int(f)
        assert set(fold_of.values()) == set(range(N_FOLDS))
        rec = {"seed": seed, "discovery_row_ids": disc, "confirmation_row_ids": conf,
               "folds": {rid: fold_of[rid] for rid in disc}}
        (OUT / f"seed_{seed}.json").write_text(json.dumps(rec))
        manifest[f"seed_{seed}_sha256"] = hashlib.sha256(
            (OUT / f"seed_{seed}.json").read_bytes()).hexdigest()
        print(f"seed {seed}: discovery {len(disc)}, confirmation {len(conf)}, "
              f"fold sizes {np.bincount([fold_of[r] for r in disc]).tolist()}")

    (OUT / "PREP_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("manifest:", json.dumps({k: v for k, v in manifest.items()
                                   if k not in ("intents",)}, indent=1)[:600])


if __name__ == "__main__":
    main()
