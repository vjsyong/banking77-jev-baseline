#!/usr/bin/env python3
"""Stage C: CLINC150 confirmation campaign — 4 arms x 5 seeds + bank freezing.

Reads ONLY the frozen protocol (plus dev-calibrated, frozen constants). Resume-
safe: runs whose final.json exists are skipped. Per seed: TF-IDF cache, one-time
C selection on the empty baseline, supervised Choice-readout reference latency
(defines L_low/L_primary/L_high), then arms U -> R -> E -> F in sequence.

Taildash: one campaign card + one card per run (Runner registers its own).
"""
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
from learner import FoldTFIDF, select_C, C_GRID  # noqa: E402
from costs import CostModel, build_pipeline, measure_batched  # noqa: E402
from selector import Selector  # noqa: E402

import run_arm as RA  # noqa: E402

SEEDS = [11, 23, 37, 53, 71]
ARMS = ["U", "R", "E", "F"]
STAGE = HERE / "runs" / "clinc150" / "fe_discovery" / "stage_c"
STAGE.mkdir(parents=True, exist_ok=True)
DATA = HERE / "data" / "clinc150" / "samples"


def load_intent_descriptions():
    f = HERE / "data" / "clinc150" / "intent_descriptions_frozen.json"
    rec = json.loads(f.read_text())
    return rec["intents"], rec


def load_protocol():
    f = HERE / "discovery" / "fe_discovery" / "frozen_protocol.json"
    proto = json.loads(f.read_text())
    # tolerate both flat and nested layouts
    d = proto.get("declared", {})
    c = proto.get("calibrated", {})
    proto.setdefault("ceilings", d.get("ceilings"))
    proto.setdefault("cost_model", c.get("cost_model"))
    proto.setdefault("unit_ms_per_text", c.get("unit_ms_per_text"))
    proto.setdefault("max_questions", proto.get("selector", {}).get("max_questions", 12))
    proto.setdefault("slots_per_round", d.get("ceilings", {}).get("slots_per_round"))
    proto.setdefault("defs_per_run", d.get("ceilings", {}).get("defs_per_run"))
    proto.setdefault("max_rounds", d.get("ceilings", {}).get("max_rounds"))
    return proto


def main():
    proto = load_protocol()
    ceilings = proto["ceilings"]
    cm = CostModel(**proto["cost_model"])
    unit_ms = proto["unit_ms_per_text"]
    RA.SLOTS_PER_ROUND = proto["slots_per_round"]
    RA.MAX_ROUNDS = proto["max_rounds"]
    RA.DEFS_PER_RUN = proto["defs_per_run"]
    percaps = proto["selector"]["per_round_cap"]
    glocap = proto["selector"]["global_cap"]

    from sklearn.model_selection import StratifiedKFold
    flat = json.loads((DATA / "flat_rows.json").read_text())
    row = {r["row_id"]: r for r in flat}
    intents, idrec = load_intent_descriptions()
    print(f"protocol loaded; {len(intents)} intents described (sha {idrec['sha256'][:12]})",
          flush=True)

    store = ProbeStore(str(HERE / "runs" / "clinc150" / "fe_discovery" / "probe_scores.sqlite"))
    ex = FEExtractor(store)

    summarize = []
    for seed in SEEDS:
        rec = json.loads((DATA / f"seed_{seed}.json").read_text())
        disc = [row[r] for r in rec["discovery_row_ids"]]
        texts = [r["text"] for r in disc]
        y = np.array([r["label"] for r in disc])
        folds = np.zeros(len(disc), dtype=int)
        pos = {rid: j for j, rid in enumerate(rec["discovery_row_ids"])}
        for rid, f in rec["folds"].items():
            folds[pos[rid]] = f

        sdir = STAGE / f"seed_{seed}"
        sdir.mkdir(exist_ok=True)
        seed_meta = sdir / "seed_setup.json"
        if seed_meta.exists():
            setup = json.loads(seed_meta.read_text())
            C = setup["C"]
            limits = setup["limits"]
            tfidf = FoldTFIDF(texts, folds)
            print(f"seed {seed}: resume (C={C}, limits={limits})", flush=True)
        else:
            t0 = time.perf_counter()
            tfidf = FoldTFIDF(texts, folds)
            C, ctable = select_C(tfidf, y)
            # reference latency: supervised Choice readout pipeline (150 options + LR)
            task_opts = [{"id": f"i{k:03d}", "definition": f"{name}: {desc}"}
                         for k, (name, desc) in enumerate(intents)]
            task_def = {"name": "task_intent", "question": "Which intent does this message express?",
                        "options": task_opts}
            from schema import canonical_hash
            task_def["slot"] = f"fe::task::{canonical_hash(task_def)[:12]}"
            # extraction for readout features on discovery rows (cached for readout eval too)
            wall, _ = ex.extract_choice_fe(task_def["slot"], task_def["question"],
                                           task_def["options"], texts)
            from fe_extract import bank_matrix
            rk = [text_key(t) for t in texts]
            block = bank_matrix(store, [task_def], rk)
            pipe = build_pipeline(texts, y, block, c_value=C)
            cal = texts[:128]
            ref = measure_batched(ex, pipe, [task_def], cal)
            L = ref["ms_per_text"]
            limits = {"low": {"ms_per_text": round(0.5 * L, 3), "L_reference": L},
                      "primary": {"ms_per_text": round(1.0 * L, 3), "L_reference": L},
                      "high": {"ms_per_text": round(2.0 * L, 3), "L_reference": L}}
            setup = {"seed": seed, "C": C, "C_table": ctable, "limits": limits,
                     "reference_ms_per_text": L, "readout_task_slot": task_def["slot"],
                     "tfidf_build_s": round(time.perf_counter() - t0, 1),
                     "task_def_sha": canonical_hash(task_def)}
            seed_meta.write_text(json.dumps(setup, indent=1))
            print(f"seed {seed}: C={C} | L_ref={L} ms/text | limits "
                  f"{limits['low']['ms_per_text']}/{limits['primary']['ms_per_text']}/"
                  f"{limits['high']['ms_per_text']}", flush=True)

        for arm in ARMS:
            outdir = sdir / f"{arm}"
            if (outdir / "final.json").exists():
                fin = json.loads((outdir / "final.json").read_text())
                print(f"  {arm} s{seed}: resume-skip (stop {fin['stop_reason']})", flush=True)
            else:
                sel = Selector(tfidf, y, C, cm, limits, store,
                               per_round_cap=percaps, global_cap=glocap,
                               max_questions=proto["max_questions"],
                               limit_split=tuple(proto["selector"]["limit_split"]))
                sel.attach_texts(texts)
                runner = RA.Runner(arm=arm, seed=seed,
                                   dataset={"texts_d": texts, "y_d": y, "folds": folds},
                                   extractor=ex, store=store, tfidf=tfidf, selector=sel,
                                   out_dir=outdir, task_name="clinc150",
                                   intent_labels=intents, ceilings=ceilings,
                                   unit_ms_per_text=unit_ms,
                                   label_of_row={},
                                   texts_cal={"texts": texts[:128]})
                fin = runner.run()
                print(f"  {arm} s{seed} done: {fin['stop_reason']} | rounds {fin['rounds']} | "
                      f"evals {fin['unique_evals']} | tok {fin['teacher_tokens']}", flush=True)
            fin = json.loads((outdir / "final.json").read_text())
            cur = fin["current"]["primary"]
            summarize.append({"arm": arm, "seed": seed, "stop": fin["stop_reason"],
                              "rounds": fin["rounds"], "evals": fin["unique_evals"],
                              "tok_in": fin["teacher_tokens"]["in"], "tok_out": fin["teacher_tokens"]["out"],
                              "primary_cv": cur["cv"] if cur else None,
                              "primary_nq": cur["nq"] if cur else None,
                              "primary_cost": cur["cost"] if cur else None})
            (STAGE / "campaign_summary.json").write_text(json.dumps(summarize, indent=1))

    print("\n=== campaign summary ===")
    for s in summarize:
        print(f"{s['arm']} s{s['seed']}: cv {s['primary_cv']} nq {s['primary_nq']} "
              f"cost {s['primary_cost']} | evals {s['evals']} tok {s['tok_in']}+{s['tok_out']}")
    store.close()


if __name__ == "__main__":
    main()
