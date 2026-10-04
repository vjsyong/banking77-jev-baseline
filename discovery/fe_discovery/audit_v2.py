#!/usr/bin/env python3
"""Audit conventions v2: frame-first option definitions with explicit routing rules.

Settled via the development instrument audit (brief §6 allows this). Changes vs v1:
  - every concept gets a crisply-defined negative option that explicitly covers
    "mentions for other topics belong here";
  - 'unclear' is reserved for "the concept is referenced but cannot be categorized";
  - option counts simplified where overlap was high.
Re-extracts v2 definitions on the same audit pairs and reports:
  - agreement vs annotator labels (loose reference) for v1 vs v2;
  - per-concept breakdown; disagreement samples for adjudication.
"""
import json
import sys
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))
sys.path.insert(0, str(HERE / "discovery" / "fe_discovery"))

import numpy as np  # noqa: E402

CONCEPTS_V2 = [
    ("transaction_status", "What state of the transaction does the message describe?",
     [("pending", "The message says the transaction is still awaiting completion or settlement."),
      ("completed", "The message says the transaction was completed, posted or succeeded."),
      ("declined", "The message says the transaction was refused, rejected or failed."),
      ("reversed", "The message says the transaction was reversed, refunded or returned."),
      ("unclear", "A transaction is referenced but its state cannot be determined from the message."),
      ("no_transaction", "The message is not about a specific transaction's state at all.")]),
    ("transfer_path", "Which direction does a money transfer move, according to the message?",
     [("incoming", "Money arrives to the user (salary, refunds, transfers received)."),
      ("outgoing", "Money leaves the user to another party (sending money, paying someone)."),
      ("between_own", "Money moves between the user's own accounts."),
      ("unclear", "A transfer is referenced but its direction cannot be determined."),
      ("no_transfer", "The message is not about a transfer's direction (questions about limits, "
                      "fees or top-ups are NOT directions).")]),
    ("card_problem_type", "What kind of card problem does the message describe?",
     [("lost_or_stolen", "The message reports the card being lost or stolen."),
      ("damaged", "The message reports the card being physically broken or no longer working."),
      ("not_received", "The message reports an ordered card not having arrived."),
      ("unclear", "The message reports a card problem but its kind cannot be determined."),
      ("no_card_problem", "No card problem is reported. Messages that merely mention a card for "
                          "other topics (fees, top-ups, transfers, refunds) belong here.")]),
    ("fee_or_charge", "Does the message concern a fee or charge?",
     [("unexpected_fee", "The message reports a fee that was actually charged and questions it."),
      ("fee_question_general", "The message asks about fees in general, without a specific charge."),
      ("unclear", "A fee is referenced but its state cannot be determined."),
      ("no_fee", "No fee or charge is involved. Messages about amounts, limits or balances "
                 "belong here unless a fee is explicitly named.")]),
    ("cash_access", "Does the message describe a cash withdrawal or ATM problem?",
     [("atm_problem", "An ATM or withdrawal attempt failed, declined, or dispensed nothing."),
      ("wrong_amount", "A withdrawn cash amount was incorrect."),
      ("unclear", "Cash or an ATM is referenced but no problem state can be determined."),
      ("no_cash", "No cash withdrawal or ATM topic is present. Messages about deposits, cheques "
                  "or card payments belong here.")]),
    ("pin_or_passcode", "What does the message say about a PIN or passcode?",
     [("forgot", "The user says they forgot their PIN or passcode."),
      ("blocked_or_locked", "The PIN/passcode is blocked, or the user is locked out."),
      ("change_request", "The user wants to change or reset the PIN/passcode."),
      ("unclear", "A PIN/passcode is referenced but its state cannot be determined."),
      ("no_pin_topic", "No PIN or passcode topic is present. Messages about card replacement, "
                       "stolen cards or money transfers belong here.")]),
    ("topup_mechanism", "What top-up or deposit method does the message describe?",
     [("card_topup", "Funds are added to the account by a card payment."),
      ("transfer_topup", "Funds are added to the account by a bank transfer."),
      ("cash_or_cheque", "Funds are added by cash or cheque."),
      ("unclear", "A top-up or deposit is referenced but its method cannot be determined."),
      ("no_topup", "No top-up or deposit is described. Messages about requesting a card, "
                   "payments, or refunds belong here.")]),
]


def main():
    from schema import validate_proposal, canonical_hash
    from fe_extract import FEExtractor
    from extract_probes import ProbeStore, text_key
    from audit_instrument import CONCEPTS

    pairs = json.loads((HERE / "runs/banking77/fe_discovery/audit/audit_pairs.json").read_text())
    store = ProbeStore(str(HERE / "runs/banking77/fe_discovery/probe_scores.sqlite"))
    ex = FEExtractor(store)

    defs = []
    for name, q, opts in CONCEPTS_V2:
        d, notes = validate_proposal({"name": name, "question": q,
                                      "options": [{"id": i, "definition": dd} for i, dd in opts]})
        assert not notes, f"{name}: {notes}"  # v2 must be complete, no auto-append
        d["slot"] = f"fe::audit2::{canonical_hash(d)[:12]}"
        defs.append(d)
    texts_all = [p["text"] for p in pairs]
    for d in defs:
        wall, _ = ex.extract_choice_fe(d["slot"], d["question"], d["options"], texts_all)
        print(f"  extract v2 {d['name']}: {wall:.1f}s", flush=True)

    by_concept = {d["name"]: d for d in defs}
    ALIAS = {"transfer_direction": "transfer_path"}
    v1 = {p["pair_id"]: p["model_label"] for p in pairs}
    v1_ann = {p["pair_id"]: p.get("annotator_label") for p in pairs}
    rows = []
    for p in pairs:
        cname_v2 = ALIAS.get(p["concept"], p["concept"])
        d = by_concept[cname_v2]
        opts = [o["id"] for o in d["options"]]
        probs = {o: float(store.choice_score_of(d["slot"], text_key(p["text"]), o)) for o in opts}
        best = max(probs, key=probs.get)
        rows.append({"pair_id": p["pair_id"], "concept": p["concept"],
                     "concept_v2": cname_v2, "text": p["text"],
                     "model_v2": best, "probs_v2": {k: round(v, 3) for k, v in probs.items()},
                     "model_v1": v1[p["pair_id"]], "annotator": v1_ann[p["pair_id"]],
                     "sample": p["sample"]})
    store.close()

    def agree(rec, key_model):
        return sum(1 for r in rec if norm_v2(r["concept"], r[key_model]) == r["annotator"]) / len(rec)

    # v2 label -> v1-compatible label where the sets were renamed/simplified
    COMPAT = {
        "transfer_path": {"between_own": "between_own_accounts", "no_transfer": "not_a_transfer"},
        "transaction_status": {"no_transaction": "unclear"},
    }

    def norm_v2(cname, lab):
        return COMPAT.get(cname, {}).get(lab, lab)

    print(f"\noverall agreement with annotator (v1-vocab comparable): "
          f"v1 {agree(rows,'model_v1'):.3f} -> v2 {agree(rows,'model_v2'):.3f}")
    for cname_v2 in [c for c, _, _ in CONCEPTS_V2]:
        cr = [r for r in rows if r["concept_v2"] == cname_v2]
        print(f"  {cname_v2:18s}: v1 {agree(cr,'model_v1'):.2f} -> v2 {agree(cr,'model_v2'):.2f} "
              f"(n={len(cr)})")
    # v2 vs annotator disagreements (for adjudication)
    dis2 = [r for r in rows if r["model_v2"] != r["annotator"] and r["annotator"] is not None]
    print(f"\nv2 disagreements: {len(dis2)}/{len(rows)}")
    (HERE / "runs/banking77/fe_discovery/audit/audit_v2_rows.json").write_text(
        json.dumps(rows, indent=1))
    print("written audit_v2_rows.json")


if __name__ == "__main__":
    main()
