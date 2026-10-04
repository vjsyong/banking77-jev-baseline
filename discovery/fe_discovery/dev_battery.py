#!/usr/bin/env python3
"""Stage A development battery (BANKING77): bounded end-to-end checks + calibration.

Runs the exact same machinery as confirmation on a B77 subset: packet types,
Choice schema, common selector, cost accounting, cache identity, boundaries.
Also measures: extraction unit cost, cost-model points, per-round timings.

CLI: python dev_battery.py --arm E --rounds 2 --slots 4 --n-rows 1200
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))
sys.path.insert(0, str(HERE / "discovery" / "fe_discovery"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from extract_probes import ProbeStore, text_key  # noqa: E402
from fe_extract import FEExtractor  # noqa: E402
from learner import FoldTFIDF, select_C  # noqa: E402
from costs import CostModel  # noqa: E402
from selector import Selector  # noqa: E402
from run_arm import Runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="E")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--n-rows", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()

    import run_arm as RA
    RA.SLOTS_PER_ROUND = args.slots
    RA.MAX_ROUNDS = args.rounds

    proc_t0 = time.perf_counter()
    pred = pd.read_csv(HERE / "runs/banking77/jev_predictions.csv")
    train = pred[pred["split"] == "train"].reset_index(drop=True)
    if args.n_rows and args.n_rows < len(train):
        rng = np.random.RandomState(7)
        idx = np.sort(rng.choice(len(train), size=args.n_rows, replace=False))
        train = train.iloc[idx].reset_index(drop=True)
    texts = train["text"].tolist()
    y = train["label"].to_numpy()
    from sklearn.model_selection import StratifiedKFold
    folds = np.zeros(len(texts), dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for f, (_, va) in enumerate(skf.split(np.zeros(len(texts)), y)):
        folds[va] = f
    intent_labels = sorted(set(y))

    print(f"dev data: {len(texts)} rows, {len(intent_labels)} classes", flush=True)
    t0 = time.perf_counter()
    tfidf = FoldTFIDF(texts, folds)
    print(f"tfidf cache: {time.perf_counter()-t0:.1f}s", flush=True)
    C, ctable = select_C(tfidf, y)
    print(f"C selected: {C} ({ctable})", flush=True)

    store = ProbeStore(str(HERE / "runs/banking77/fe_discovery/probe_scores.sqlite"))
    ex = FEExtractor(store)
    # use calibrated cost model / unit cost when available (falls back to placeholders)
    import json as _json
    calib_p = HERE / "runs/banking77/fe_discovery/cost_calibration.json"
    if calib_p.exists():
        _c = _json.loads(calib_p.read_text())
        unit = float(_c["unit_ms_per_text_median"])
        cm = CostModel(**_c["cost_model"])
        print(f"calibrated: unit {unit} ms/text, cost model {_c['cost_model']}", flush=True)
    else:
        unit = 3.0
        cm = CostModel(b0=0.5, b1=0.9, b2=0.35, model_id="dev-placeholder")
    limits = {"low": {"ms_per_text": 8.0}, "primary": {"ms_per_text": 18.0},
              "high": {"ms_per_text": 36.0}}
    sel = Selector(tfidf, y, C, cm, limits, store, per_round_cap=18, global_cap=60,
                   max_questions=12, limit_split=(5, 8, 5))
    sel.attach_texts(texts)

    out_dir = (HERE / "runs" / "banking77" / "fe_discovery" / args.tag /
               f"{args.arm}_s{args.seed}")
    runner = Runner(arm=args.arm, seed=args.seed,
                    dataset={"texts_d": texts, "y_d": y, "folds": folds},
                    extractor=ex, store=store, tfidf=tfidf, selector=sel,
                    out_dir=out_dir, task_name="banking77-dev",
                    intent_labels=intent_labels,
                    ceilings={"teacher_tokens": 120_000, "evals": 60,
                              "extract_logical_s": 900.0},
                    unit_ms_per_text=unit,
                    label_of_row={t: l for t, l in zip(texts, y)},
                    texts_cal={"texts": texts[:64]})
    final = runner.run()
    print(json.dumps({k: final[k] for k in ("stop_reason", "rounds", "teacher_calls",
                                            "teacher_tokens", "ledger", "unique_evals",
                                            "wall_s")}, indent=1))
    print("current:", json.dumps({ln: {"names": final["current"][ln]["names"],
                                       "cv": final["current"][ln]["cv"],
                                       "cost": final["current"][ln]["cost"]}
                                  for ln in ("low", "primary", "high")}, indent=1))
    print(f"total wall {time.perf_counter()-proc_t0:.0f}s")
    store.close()


if __name__ == "__main__":
    main()
