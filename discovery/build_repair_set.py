#!/usr/bin/env python3
"""Build the channel-repair judgment set: (concept, message, label) triples.

Three concepts, ~60 train messages each: clear positives, clear negatives, and
forced known-failure cases from the audit. Labels are intent-derived, reviewed
and frozen here for auditing (see repair_set.csv; the harness re-scores if the
labels change).

Writes runs/banking77/discovery/repair_set.csv and a review printout.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "runs" / "banking77" / "discovery"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 77

POS_INTENTS = {
    "card": ["card_arrival", "activate_my_card", "get_physical_card", "card_not_working",
             "card_swallowed", "lost_or_stolen_card", "compromised_card",
             "declined_card_payment", "card_linking", "visa_or_mastercard"],
    "unauth": ["card_payment_not_recognised", "cash_withdrawal_not_recognised",
               "direct_debit_payment_not_recognised"],
    "pin": ["pin_blocked", "change_pin", "passcode_forgotten"],
}
NEG_INTENTS = {
    "card": ["exchange_rate", "transfer_timing", "top_up_failed", "Refund_not_showing_up",
             "receiving_money", "balance_not_updated_after_bank_transfer",
             "verify_my_identity", "terminate_account", "transfer_fee_charged",
             "exchange_via_app"],
    "unauth": ["exchange_rate", "card_arrival", "transfer_timing", "top_up_failed",
               "Refund_not_showing_up", "receiving_money", "change_pin", "age_limit",
               "balance_not_updated_after_bank_transfer", "transfer_fee_charged"],
    "pin": ["card_arrival", "exchange_rate", "transfer_timing", "top_up_failed",
            "Refund_not_showing_up", "receiving_money", "age_limit", "cash_withdrawal_charge",
            "balance_not_updated_after_bank_transfer", "transfer_fee_charged"],
}
# forced known-failure / audit cases: (concept, intent, label, text prefix, note)
FORCED = [
    ("card", "card_not_working", 1, "Nothing goes through on my card",
     "audit flagship: scored 0.17 on existing card probe (test split)"),
    ("card", "card_arrival", 1, "I am still waiting on my card",
     "audit: card probe ok here (0.68)"),
    ("card", "age_limit", 0, "Can my 19 year old daughter open a savings account",
     "audit: scored 0.87 on existing card probe (false high, test split)"),
    ("unauth", "change_pin", 0, "I want to change my pin number from an ATM",
     "audit: scored 0.85 on existing unauthorised probe (false high)"),
    ("unauth", "wrong_exchange_rate_for_cash_withdrawal", 0,
     "Do any of your machines provide cash from my home country",
     "clear negative for unauthorised"),
    ("pin", "change_pin", 1, "I want to change my pin number from an ATM",
     "clear positive for pin"),
    ("pin", "cash_withdrawal_not_recognised", 0,
     "According to the app, I got cash from an ATM but I haven't made any",
     "regression guard: unauthorised text, not a pin issue"),
]

# review-pass exclusions of borderline/ambiguous sampled rows
EXCLUDE = [
    ("unauth", "what is the word?"),
    ("unauth", "i didnt put that money in my account"),
    ("pin", "How do I unblock my card?"),
    ("card", "How do I get a PIN?"),
    ("card", "I havn't received my PIN yet."),
]

PER_CONCEPT_POS = 21
PER_CONCEPT_NEG = 21


def main():
    d = load_dataset("PolyAI/banking77")
    names = d["train"].features["label"].names
    name_to_idx = {n: i for i, n in enumerate(names)}
    for lst in list(POS_INTENTS.values()) + list(NEG_INTENTS.values()):
        unknown = [n for n in lst if n not in name_to_idx]
        if unknown:
            raise SystemExit(f"unknown intent names: {unknown}")

    rng = np.random.default_rng(SEED)
    rows = []
    by_intent = {}
    by_intent_test = {}
    for i, item in enumerate(d["train"]):
        by_intent.setdefault(names[int(item["label"])], []).append((i, item["text"]))
    for i, item in enumerate(d["test"]):
        by_intent_test.setdefault(names[int(item["label"])], []).append((i, item["text"]))

    for concept in ("card", "unauth", "pin"):
        pos_pool = [(t, i) for n in POS_INTENTS[concept] for i, t in by_intent[n]]
        neg_pool = [(t, i) for n in NEG_INTENTS[concept] for i, t in by_intent[n]]
        pos_idx = rng.choice(len(pos_pool), size=PER_CONCEPT_POS, replace=False)
        neg_idx = rng.choice(len(neg_pool), size=PER_CONCEPT_NEG, replace=False)
        for j in pos_idx:
            t, i = pos_pool[j]
            rows.append({"concept": concept, "label": 1, "bucket": "clear_pos",
                         "split_row": i, "source_split": "train", "text": t})
        for j in neg_idx:
            t, i = neg_pool[j]
            rows.append({"concept": concept, "label": 0, "bucket": "clear_neg",
                         "split_row": i, "source_split": "train", "text": t})
        for c, intent, lab, prefix, note in FORCED:
            if c != concept:
                continue
            hit, src = None, None
            for t in by_intent.get(intent, []):
                if t[1].startswith(prefix):
                    hit, src = t[1], "train"
                    break
            if hit is None:
                for t in by_intent_test.get(intent, []):
                    if t[1].startswith(prefix):
                        hit, src = t[1], "test"
                        break
            if hit is None:
                print(f"WARN forced case not found: {prefix[:50]!r}")
                continue
            rows.append({"concept": concept, "label": lab, "bucket": "known_failure",
                         "split_row": -1, "source_split": src, "text": hit})

    # review-pass exclusions
    for concept, prefix in EXCLUDE:
        before = len(rows)
        rows = [r for r in rows if not (r["concept"] == concept and r["text"].startswith(prefix))]
        if len(rows) == before:
            print(f"WARN exclusion not applied: {concept}:{prefix[:40]!r}")

    # dedupe per concept on text
    seen = set()
    deduped = []
    for r in rows:
        k = (r["concept"], r["text"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)

    with (OUT / "repair_set.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "label", "bucket", "split_row",
                                          "source_split", "text"])
        w.writeheader()
        w.writerows(deduped)

    print(f"total rows: {len(deduped)}")
    for concept in ("card", "unauth", "pin"):
        sub = [r for r in deduped if r["concept"] == concept]
        npos = sum(r["label"] for r in sub)
        print(f"\n===== {concept}: {len(sub)} rows ({npos} pos / {len(sub) - npos} neg) =====")
        for r in sub:
            mark = "P" if r["label"] else "n"
            print(f"[{mark}] {r['bucket'][:9]:9s} {r['text'][:88]}")


if __name__ == "__main__":
    main()
