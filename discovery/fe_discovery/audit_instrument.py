#!/usr/bin/env python3
"""Stage A development instrument audit (brief §6).

~200 message-question pairs over 7 predefined development concepts (BANKING77).
Instrument = the frozen Choice extractor (TinyJev). Annotation = an independent
annotator model + researcher adjudication on all disagreements and a 20%
subsample (replace with a human panel for strict confirmation; recorded as a
deviation in the freeze checklist).

Outputs: audit_pairs.json (pairs + model labels/probs + annotator labels),
         adjudication_sheet.jsonl (to be filled),
         audit_report.json (metrics after --score).

Usage:
  python audit_instrument.py --build          # build pairs + model labels + annotate
  python audit_instrument.py --score FILE     # score with filled adjudications
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

AUDIT = HERE / "runs" / "banking77" / "fe_discovery" / "audit"
AUDIT.mkdir(parents=True, exist_ok=True)
SEED = 4242

CONCEPTS = [
    ("transaction_status", "What state of the transaction does the message describe?",
     [("pending", "Awaiting completion or settlement."),
      ("completed", "Explicitly completed or posted."),
      ("declined", "Explicitly refused, rejected or failed."),
      ("reversed", "Explicitly reversed, refunded or returned."),
      ("unclear", "No definite state is stated.")]),
    ("transfer_direction", "Which direction does the money transfer move?",
     [("incoming", "Money arriving to the user."),
      ("outgoing", "Money leaving from the user to another party."),
      ("between_own_accounts", "Between the user's own accounts."),
      ("not_a_transfer", "The message is not about a transfer."),
      ("unclear", "Transfer is mentioned but direction is unclear.")]),
    ("card_problem_type", "What kind of card problem does the message describe?",
     [("lost_or_stolen", "Card lost or stolen."),
      ("damaged", "Card physically damaged or not working."),
      ("not_received", "Card ordered but not received."),
      ("other_issue", "Another card issue (declined, blocked, etc.)."),
      ("no_card_problem", "No card problem is described."),
      ("unclear", "Card is mentioned but the problem is unclear.")]),
    ("fee_or_charge", "Does the message concern a fee or charge the user questions?",
     [("unexpected_fee", "A fee was charged and the user questions it."),
      ("fee_question_general", "The user asks about fees in general."),
      ("no_fee", "No fee or charge is mentioned."),
      ("unclear", "It is unclear whether a fee is involved.")]),
    ("cash_access", "Does the message describe a cash withdrawal or ATM problem?",
     [("atm_problem", "ATM did not dispense or declined the withdrawal."),
      ("wrong_amount", "A cash amount was incorrect."),
      ("cash_general", "Cash is mentioned without a problem."),
      ("no_cash", "No cash or ATM topic is present."),
      ("unclear", "Cash is mentioned but the issue is unclear.")]),
    ("pin_or_passcode", "What does the message say about a PIN or passcode?",
     [("forgot", "User forgot the PIN or passcode."),
      ("blocked_or_locked", "PIN blocked, locked out."),
      ("change_request", "User wants to change or reset it."),
      ("no_pin_topic", "No PIN or passcode topic is present."),
      ("unclear", "PIN is mentioned but the state is unclear.")]),
    ("topup_mechanism", "What is the top-up or deposit method described?",
     [("card_topup", "Top-up via card."),
      ("transfer_topup", "Top-up via bank transfer."),
      ("cash_or_cheque", "Top-up via cash or cheque."),
      ("no_topup", "No top-up or deposit is described."),
      ("unclear", "Top-up mentioned but method unclear.")]),
]


def build():
    from schema import validate_proposal, canonical_hash
    from fe_extract import FEExtractor
    from extract_probes import ProbeStore, text_key
    from teacher_client import teacher_chat_full

    pred = pd.read_csv(HERE / "runs/banking77/jev_predictions.csv")
    train = pred[pred["split"] == "train"].reset_index(drop=True)
    rng = np.random.RandomState(SEED)
    n = len(train)
    short_idx = [i for i in range(n) if len(train.iloc[i]["text"]) <= 28]

    defs, pairs = [], []
    for cname, question, opts in CONCEPTS:
        defn, notes = validate_proposal({"name": cname, "question": question,
                                         "options": [{"id": i, "definition": d} for i, d in opts]})
        defn["slot"] = f"fe::audit::{canonical_hash(defn)[:12]}"
        h = canonical_hash(defn)
        defs.append(dict(defn, idh=h))
        # targeted classes: share a token with the concept name
        toks = set(cname.split("_"))
        tgt = [i for i in range(n)
               if (set(str(train.iloc[i]["label"]).split("_")) & toks)]
        tgt = list(rng.choice(tgt, size=min(12, len(tgt)), replace=False)) if tgt else []
        oth = list(rng.choice([i for i in range(n) if i not in set(tgt)], size=8, replace=False))
        amb = list(rng.choice(short_idx, size=8, replace=False))
        for i in tgt + oth + amb:
            pairs.append({"pair_id": len(pairs), "concept": cname,
                          "text": train.iloc[i]["text"], "label_true": train.iloc[i]["label"],
                          "sample": "targeted" if i in tgt else ("other" if i in oth else "short")})
    print(f"concepts {len(defs)} | pairs {len(pairs)}", flush=True)

    store = ProbeStore(str(HERE / "runs" / "banking77" / "fe_discovery" / "probe_scores.sqlite"))
    ex = FEExtractor(store)
    texts_all = [p["text"] for p in pairs]
    for d in defs:
        t0 = time.perf_counter()
        wall, _ = ex.extract_choice_fe(d["slot"], d["question"], d["options"], texts_all)
        print(f"  extract {d['name']}: {wall:.1f}s", flush=True)
    # model labels per pair (column of its concept)
    for p in pairs:
        d = next(d for d in defs if d["name"] == p["concept"])
        opts = [o["id"] for o in d["options"]]
        row = texts_all.index(p["text"])
        probs = []
        for o in opts:
            probs.append(store.choice_score_of(d["slot"], text_key(p["text"]), o))
        best = int(np.argmax(probs))
        p["model_label"] = opts[best]
        p["model_probs"] = {o: round(float(v), 4) for o, v in zip(opts, probs)}
    store.close()

    # annotator model pass (batches of 25 pairs)
    defn_by_concept = {d["name"]: d for d in defs}
    for b in range(0, len(pairs), 25):
        batch = pairs[b:b + 25]
        lines = []
        for p in batch:
            d = defn_by_concept[p["concept"]]
            opts = "; ".join(f"{o['id']}: {o['definition']}" for o in d["options"])
            lines.append(f"[{p['pair_id']}] concept={p['concept']} | question: {d['question']} | "
                         f"options: {opts} | message: \"{p['text']}\"")
        prompt = ("You are auditing a text-understanding instrument. For each item, choose the "
                  "single best option id for the message under the given question. Answer ONLY "
                  "with a JSON array of {\"pair_id\": int, \"option\": string, \"confidence\": 0..1}. "
                  "Use 'unclear' when the message does not support a definite category.\n\n"
                  + "\n".join(lines))
        resp = teacher_chat_full([{"role": "user", "content": prompt}],
                                 model="gpt-6-astra", provider="lmuai-pro",
                                 temperature=0.2, seed=SEED + b, max_tokens=2000)
        try:
            objs = json.loads(resp["text"][resp["text"].index("["):resp["text"].rindex("]") + 1])
        except Exception as e:  # noqa: BLE001
            print("annotator parse fail batch", b, e, flush=True)
            objs = []
        by_id = {o.get("pair_id"): o for o in objs if isinstance(o, dict)}
        for p in batch:
            o = by_id.get(p["pair_id"])
            p["annotator_label"] = (o or {}).get("option", None)
            p["annotator_conf"] = (o or {}).get("confidence", None)
        print(f"  annotated batch {b//25+1}: {sum(1 for p in batch if p.get('annotator_label'))}/25",
              flush=True)
        time.sleep(1)

    (AUDIT / "audit_pairs.json").write_text(json.dumps(pairs, indent=1))
    # adjudication sheet: disagreements + 20% subsample
    rng2 = np.random.RandomState(SEED)
    dis = [p for p in pairs if p.get("annotator_label") != p["model_label"]]
    rest = [p for p in pairs if p not in dis]
    sub = [rest[i] for i in rng2.choice(len(rest), size=max(1, int(0.2 * len(pairs))), replace=False)]
    sheet = dis + sub
    with (AUDIT / "adjudication_sheet.jsonl").open("w") as f:
        for p in sheet:
            f.write(json.dumps({"pair_id": p["pair_id"], "concept": p["concept"],
                                "text": p["text"],
                                "model_label": p["model_label"],
                                "annotator_label": p.get("annotator_label"),
                                "adj_label": None}) + "\n")
    print(f"sheet: {len(sheet)} rows to adjudicate ({len(dis)} disagreements + {len(sub)} subsample)")
    print("done --build")


def score(adj_file):
    pairs = json.loads((AUDIT / "audit_pairs.json").read_text())
    adj = {}
    for line in Path(adj_file).read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("adj_label"):
                adj[rec["pair_id"]] = rec["adj_label"]
    from sklearn.metrics import confusion_matrix, f1_score
    report = {"n_pairs": len(pairs), "n_adjudicated": len(adj), "concepts": {}, "overall": {}}
    agree_m, agree_a, disc_ok, disc_bad = [], [], [], []
    for cname, _, opts in [(c, q, o) for c, q, o in CONCEPTS]:
        cp = [p for p in pairs if p["concept"] == cname]
        oids = [i for i, _ in opts]
        truth = [adj.get(p["pair_id"], p["annotator_label"]) for p in cp]
        pred = [p["model_label"] for p in cp]
        valid = [(t, pr) for t, pr in zip(truth, pred) if t in oids]
        if not valid:
            continue
        t, pr = zip(*valid)
        f1 = f1_score(t, pr, average="macro", labels=oids, zero_division=0)
        cm = confusion_matrix(t, pr, labels=oids).tolist()
        agree_m.append(np.mean([a == b for a, b in zip(t, pr)]))
        for p in cp:
            tlab = adj.get(p["pair_id"], p.get("annotator_label"))
            if tlab not in oids:
                continue
            ptop = max(p["model_probs"].values())
            (disc_ok if p["model_label"] == tlab else disc_bad).append(ptop)
        report["concepts"][cname] = {"n": len(cp), "macro_f1": round(float(f1), 3),
                                     "agreement_vs_adjudicated": round(float(np.mean([a == b for a, b in zip(t, pr)])), 3),
                                     "confusion (rows=true, cols=model)": cm,
                                     "options": oids}
    report["overall"] = {
        "mean_concept_macro_f1": round(float(np.mean(list(
            c["macro_f1"] for c in report["concepts"].values()))), 3),
        "mean_agreement": round(float(np.mean(agree_m)), 3),
        "mean_topprob_correct": round(float(np.mean(disc_ok)), 3) if disc_ok else None,
        "mean_topprob_incorrect": round(float(np.mean(disc_bad)), 3) if disc_bad else None,
        "disagreement_rate_model_vs_annotator": round(float(np.mean([
            1 if p.get("annotator_label") != p["model_label"] else 0 for p in pairs])), 3),
    }
    (AUDIT / "audit_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["overall"], indent=1))
    print("written audit_report.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", metavar="ADJ_FILE")
    args = ap.parse_args()
    if args.build:
        build()
    elif args.score:
        score(args.score)
