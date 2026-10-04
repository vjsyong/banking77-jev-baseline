#!/usr/bin/env python3
"""One Stage C arm run in an isolated process (pool worker).

Usage: stage_c_arm.py --seed 11 --arm U
Reuses the frozen protocol + seed setup written by stage_c.py. The Runner writes
final.json/rounds.jsonl into the same per-run dir the serial campaign used.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))
sys.path.insert(0, str(HERE / "discovery" / "fe_discovery"))

import numpy as np  # noqa: E402

from extract_probes import ProbeStore  # noqa: E402
from fe_extract import FEExtractor  # noqa: E402
from learner import FoldTFIDF  # noqa: E402
from costs import CostModel  # noqa: E402
from selector import Selector  # noqa: E402

import run_arm as RA  # noqa: E402
import stage_c as SC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", required=True)
    args = ap.parse_args()
    seed, arm = args.seed, args.arm

    proto = SC.load_protocol()
    ceilings = proto["ceilings"]
    cm = CostModel(**proto["cost_model"])
    unit_ms = proto["unit_ms_per_text"]
    RA.SLOTS_PER_ROUND = proto["slots_per_round"]
    RA.MAX_ROUNDS = proto["max_rounds"]
    RA.DEFS_PER_RUN = proto["defs_per_run"]

    rec = json.loads((SC.DATA / f"seed_{seed}.json").read_text())
    flat = json.loads((SC.DATA / "flat_rows.json").read_text())
    row = {r["row_id"]: r for r in flat}
    disc = [row[r] for r in rec["discovery_row_ids"]]
    texts = [r["text"] for r in disc]
    y = np.array([r["label"] for r in disc])
    folds = np.zeros(len(disc), dtype=int)
    pos = {rid: j for j, rid in enumerate(rec["discovery_row_ids"])}
    for rid, f in rec["folds"].items():
        folds[pos[rid]] = f

    sdir = SC.STAGE / f"seed_{seed}"
    setup = json.loads((sdir / "seed_setup.json").read_text())
    C = setup["C"]
    limits = setup["limits"]
    intents, _ = SC.load_intent_descriptions()

    store = ProbeStore(str(HERE / "runs" / "clinc150" / "fe_discovery" / "probe_scores.sqlite"))
    ex = FEExtractor(store)
    tfidf = FoldTFIDF(texts, folds, cache_path=str(sdir / "tfidf_cache.pkl"))

    outdir = sdir / arm
    sel = Selector(tfidf, y, C, cm, limits, store,
                   per_round_cap=proto["selector"]["per_round_cap"],
                   global_cap=proto["selector"]["global_cap"],
                   max_questions=proto["max_questions"],
                   limit_split=tuple(proto["selector"]["limit_split"]))
    sel.attach_texts(texts)
    runner = RA.Runner(arm=arm, seed=seed,
                       dataset={"texts_d": texts, "y_d": y, "folds": folds},
                       extractor=ex, store=store, tfidf=tfidf, selector=sel,
                       out_dir=outdir, task_name="clinc150",
                       intent_labels=[n for n, _ in intents], ceilings=ceilings,
                       unit_ms_per_text=unit_ms,
                       label_of_row={},
                       texts_cal={"texts": texts[:128]})
    fin = runner.run()
    print(f"[{arm} s{seed}] done: {fin['stop_reason']} | rounds {fin['rounds']} | "
          f"evals {fin['unique_evals']} | tok {fin['teacher_tokens']}", flush=True)
    store.close()


if __name__ == "__main__":
    main()
