#!/usr/bin/env python3
"""Audit conventions v3: merge the (explicit-negative vs unclear) boundary.

Audit finding (v1/v2): the extractor reliably separates concept-present subtypes
but does not implement the fine distinction between an explicitly-denied negative
option and 'unclear' (dominant 'unclear' attractor for absent topics). Convention:
ONE merged option = 'not present or cannot be determined'. Re-extracts and scores
agreement against annotator labels mapped to the same merged convention.
"""
import json
import sys
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))
sys.path.insert(0, str(HERE / "discovery" / "fe_discovery"))

CONCEPTS_V3 = [
    ("transaction_status", "What state of the transaction does the message describe?",
     [("pending", "The transaction is awaiting completion or settlement."),
      ("completed", "The transaction was completed, posted or succeeded."),
      ("declined", "The transaction was refused, rejected or failed."),
      ("reversed", "The transaction was reversed, refunded or returned."),
      ("unclear", "No definite transaction state is described (or none can be determined).")]),
    ("transfer_path", "Which direction does a money transfer move, according to the message?",
     [("incoming", "Money arrives to the user (salary, refunds, transfers received)."),
      ("outgoing", "Money leaves the user to another party."),
      ("between_own", "Money moves between the user's own accounts."),
      ("unclear", "No definite transfer direction is described (or none can be determined).")]),
    ("card_problem_type", "What kind of card problem does the message describe?",
     [("lost_or_stolen", "The card was lost or stolen."),
      ("damaged", "The card is broken or no longer works."),
      ("not_received", "An ordered card has not arrived."),
      ("other_issue", "Another card issue: declined, blocked, expiring, not accepted."),
      ("unclear", "No definite card problem is described (or none can be determined).")]),
    ("fee_or_charge", "Does the message concern a fee or charge?",
     [("unexpected_fee", "A fee was actually charged and the message questions it."),
      ("fee_question_general", "The message asks about fees in general."),
      ("unclear", "No definite fee topic is described (or none can be determined).")]),
    ("cash_access", "Does the message describe a cash withdrawal or ATM problem?",
     [("atm_problem", "An ATM or withdrawal attempt failed, declined, or dispensed nothing."),
      ("wrong_amount", "A withdrawn cash amount was incorrect."),
      ("cash_general", "Cash or ATM is discussed without a problem (locations, limits, timing)."),
      ("unclear", "No definite cash or ATM topic is described (or none can be determined).")]),
    ("pin_or_passcode", "What does the message say about a PIN or passcode?",
     [("forgot", "The user forgot their PIN or passcode."),
      ("blocked_or_locked", "The PIN/passcode is blocked, or the user is locked out."),
      ("change_request", "The user wants to change or reset the PIN/passcode."),
      ("unclear", "No definite PIN or passcode topic is described (or none can be determined).")]),
    ("topup_mechanism", "What top-up or deposit method does the message describe?",
     [("card_topup", "Funds are added to the account by a card payment."),
      ("transfer_topup", "Funds are added to the account by a bank transfer."),
      ("cash_or_cheque", "Funds are added by cash or cheque."),
      ("unclear", "No definite top-up or deposit is described (or none can be determined).")]),
]

# annotator (v1-vocab) -> v3 label per concept (merged negative/unclear)
ANN_MAP = {
    "transaction_status": {"no_transaction": "unclear"},
    "transfer_direction": {"not_a_transfer": "unclear"},
    "transfer_path": {"not_a_transfer": "unclear", "no_transfer": "unclear"},
    "card_problem_type": {"no_card_problem": "unclear"},
    "fee_or_charge": {"no_fee": "unclear"},
    "cash_access": {"no_cash": "unclear"},
    "pin_or_passcode": {"no_pin_topic": "unclear"},
    "topup_mechanism": {"no_topup": "unclear"},
}


def main():
    from schema import validate_proposal, canonical_hash
    from fe_extract import FEExtractor
    from extract_probes import ProbeStore, text_key

    pairs = json.loads((HERE / "runs/banking77/fe_discovery/audit/audit_pairs.json").read_text())
    store = ProbeStore(str(HERE / "runs/banking77/fe_discovery/probe_scores.sqlite"))
    ex = FEExtractor(store)
    defs = {}
    for name, q, opts in CONCEPTS_V3:
        d, notes = validate_proposal({"name": name, "question": q,
                                      "options": [{"id": i, "definition": dd} for i, dd in opts]})
        assert not notes
        d["slot"] = f"fe::audit3::{canonical_hash(d)[:12]}"
        defs[name] = d
        wall, _ = ex.extract_choice_fe(d["slot"], d["question"], d["options"],
                                       [p["text"] for p in pairs])
        print(f"  extract v3 {name}: {wall:.1f}s", flush=True)

    ALIAS = {"transfer_direction": "transfer_path"}
    rows = []
    for p in pairs:
        c2 = ALIAS.get(p["concept"], p["concept"])
        d = defs[c2]
        opts = [o["id"] for o in d["options"]]
        probs = {o: float(store.choice_score_of(d["slot"], text_key(p["text"]), o)) for o in opts}
        rows.append({"pair_id": p["pair_id"], "concept_v3": c2, "text": p["text"], "sample": p["sample"],
                     "model_v3": max(probs, key=probs.get),
                     "probs_v3": {k: round(v, 3) for k, v in probs.items()},
                     "annotator_raw": p.get("annotator_label"),
                     "annotator_v3": ANN_MAP.get(p["concept"], {}).get(p.get("annotator_label"),
                                                                       p.get("annotator_label"))})
    store.close()
    ok = sum(1 for r in rows if r["model_v3"] == r["annotator_v3"])
    print(f"\noverall agreement v3-vocab: {ok}/{len(rows)} = {ok/len(rows):.3f}")
    for c2 in [c for c, _, _ in CONCEPTS_V3]:
        cr = [r for r in rows if r["concept_v3"] == c2]
        a = sum(1 for r in cr if r["model_v3"] == r["annotator_v3"])
        print(f"  {c2:18s}: {a}/{len(cr)} = {a/len(cr):.2f}")
    (HERE / "runs/banking77/fe_discovery/audit/audit_v3_rows.json").write_text(json.dumps(rows, indent=1))
    print("written audit_v3_rows.json")


if __name__ == "__main__":
    main()
