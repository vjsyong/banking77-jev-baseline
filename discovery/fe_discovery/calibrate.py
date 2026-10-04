#!/usr/bin/env python3
"""Stage A calibration: cost-model points, extraction unit cost, cap proposals.

Measures real banks (0/4/8/12 questions) end-to-end on a B77 development subset:
fit model on (n_questions, n_options_total) -> ms/text. Derives:
  - unit_ms_per_text (extraction ledger charge)
  - CostModel coefficients (frozen)
  - proposed caps: per-round evals, global evals, extraction logical ceiling,
    teacher token ceiling (based on observed per-round consumption)
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
from schema import validate_proposal, canonical_hash  # noqa: E402
from costs import build_pipeline, measure_batched, measure_single, CostModel  # noqa: E402

OUT = HERE / "runs" / "banking77" / "fe_discovery" / "cost_calibration.json"

# 12 real definitions (7 audit concepts + 5 variants) for measurement banks
BASE = [
    ("transaction_state", "What state of the transaction does the message describe?",
     [("pending", "Awaiting completion or settlement."), ("completed", "Explicitly completed or posted."),
      ("declined", "Explicitly refused, rejected or failed."), ("reversed", "Explicitly reversed or refunded."),
      ("unclear", "No definite state is stated.")]),
    ("transfer_path", "Which direction does the money transfer move?",
     [("incoming", "Money arriving to the user."), ("outgoing", "Money leaving to another party."),
      ("between_own", "Between the user's own accounts."), ("not_transfer", "Not a transfer."),
      ("unclear", "Unclear direction.")]),
    ("card_trouble", "What kind of card problem is described?",
     [("lost_stolen", "Lost or stolen."), ("damaged", "Damaged or not working."),
      ("not_received", "Not received."), ("other", "Other issue."), ("unclear", "Unclear.")]),
    ("fee_context", "Does the message concern a fee the user questions?",
     [("unexpected_fee", "A fee charged and questioned."), ("fee_question", "General fee question."),
      ("no_fee", "No fee."), ("unclear", "Unclear.")]),
    ("cash_context", "Does the message describe a cash or ATM problem?",
     [("atm_problem", "ATM failed or declined."), ("wrong_amount", "Incorrect amount."),
      ("cash_ok", "Cash without a problem."), ("no_cash", "No cash topic."), ("unclear", "Unclear.")]),
    ("pin_context", "What does the message say about a PIN or passcode?",
     [("forgot", "Forgot it."), ("blocked", "Blocked or locked."), ("change", "Wants to change it."),
      ("no_pin", "No PIN topic."), ("unclear", "Unclear.")]),
    ("topup_context", "What top-up or deposit method is described?",
     [("card", "Via card."), ("transfer", "Via bank transfer."), ("cash", "Via cash or cheque."),
      ("no_topup", "No top-up."), ("unclear", "Unclear.")]),
    ("urgency_tone", "How urgent is the user's situation?",
     [("urgent_now", "Blocked right now."), ("soon", "Needs it soon."), ("routine", "Routine."),
      ("unclear", "Unclear.")]),
    ("account_scope", "Whose account does the issue concern?",
     [("own", "The user's own."), ("other_person", "Another person's."), ("unclear", "Unclear.")]),
    ("payment_channel", "What payment channel is involved?",
     [("online", "Online or app."), ("in_store", "In store or terminal."), ("atm", "ATM."),
      ("transfer_channel", "Bank transfer."), ("none", "No channel mentioned."), ("unclear", "Unclear.")]),
    ("refund_state", "What is the state of any refund mentioned?",
     [("requested", "Requested."), ("processing", "In processing."), ("received", "Received."),
      ("refused", "Refused."), ("no_refund", "No refund mentioned."), ("unclear", "Unclear.")]),
    ("card_use_intent", "What does the user want to do with their card?",
     [("activate", "Activate it."), ("freeze", "Freeze or block it."), ("replace", "Replace it."),
      ("limit_change", "Change limits."), ("nothing", "No card action."), ("unclear", "Unclear.")]),
]


def main():
    import sys as _sys
    _sys.path.insert(0, str(HERE / "discovery" / "fe_discovery"))
    sys.path.insert(0, "/home/xrim/taildash/client")
    try:
        from taildash import TaskMonitor
        mon = TaskMonitor(server="http://localhost:8080",
                          title="FE: cost calibration v2 (B77 dev)", total=6,
                          agent_name="fe-discovery")
    except Exception:  # noqa: BLE001
        class _N:
            def log(self, *a, **k): pass
            def update(self, *a, **k): pass
            def complete(self, *a, **k): pass
            def fail(self, *a, **k): pass
        mon = _N()
    try:
        _calib_main(mon)
        mon.complete("calibration done")
    except BaseException as exc:  # noqa: BLE001
        mon.fail(f"{type(exc).__name__}: {exc}")
        raise


def _calib_main(mon):
    pred = pd.read_csv(HERE / "runs/banking77/jev_predictions.csv")
    train = pred[pred["split"] == "train"].reset_index(drop=True)
    rng = np.random.RandomState(7)
    idx = np.sort(rng.choice(len(train), size=1200, replace=False))
    train = train.iloc[idx].reset_index(drop=True)
    texts = train["text"].tolist()
    y = train["label"].to_numpy()
    cal_texts = texts[:128]

    defs = []
    for name, q, opts in BASE:
        d, _ = validate_proposal({"name": name, "question": q,
                                  "options": [{"id": i, "definition": dd} for i, dd in opts]})
        d["slot"] = f"fe::cal::{canonical_hash(d)[:12]}"
        defs.append(d)
    print(f"{len(defs)} defs for calibration", flush=True)

    store = ProbeStore(str(HERE / "runs/banking77/fe_discovery" / "probe_scores.sqlite"))
    ex = FEExtractor(store)
    mon.log(f"calibration on {len(texts)} rows, {len(defs)} defs")
    points, per_def_ms = [], {}
    for k in (0, 4, 8, 12):
        bank = defs[:k]
        if k:
            t0 = time.perf_counter()
            for d in bank:
                wall, _ = ex.extract_choice_fe(d["slot"], d["question"], d["options"], texts)
                per_def_ms[d["name"]] = round(1000 * wall / max(len(texts), 1), 2)
            print(f"  extraction {k} defs: {time.perf_counter()-t0:.0f}s total", flush=True)
        block = None
        from fe_extract import bank_matrix
        rk = [text_key(t) for t in texts]
        block = bank_matrix(store, bank, rk) if bank else np.zeros((len(texts), 0))
        pipe = build_pipeline(texts, y, block, c_value=8.0)
        out = measure_batched(ex, pipe, bank, cal_texts)
        n_opts = sum(len(d["options"]) for d in bank)
        points.append((len(bank), n_opts, out["ms_per_text"]))
        mon.update(len(points) + 1, message=f"bank {k}q: {out['ms_per_text']} ms/text")
        print(f"  bank {k:2d}q/{n_opts:2d}opt: {out['ms_per_text']} ms/text ({out['throughput']} /s)", flush=True)

    model = CostModel.fit(points)
    unit = float(np.median([v for v in per_def_ms.values()])) if per_def_ms else 4.5
    res = {
        "points": points,
        "cost_model": model.to_json(),
        "unit_ms_per_text_median": unit,
        "per_def_ms_per_text": per_def_ms,
        "proposed_caps": {
            "per_round_evals": 30, "global_evals": 160,
            "extract_logical_s": 900.0, "teacher_tokens": 100000,
            "slots_per_round": 12, "defs_per_run": 60, "max_rounds": 5,
        },
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({"cost_model": res["cost_model"], "unit_ms": unit,
                      "proposed_caps": res["proposed_caps"]}, indent=1))
    print("written", OUT)
    store.close()


if __name__ == "__main__":
    main()
