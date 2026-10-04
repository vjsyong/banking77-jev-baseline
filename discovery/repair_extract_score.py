#!/usr/bin/env python3
"""Channel-repair scoring: three question variants per concept vs human-audit labels.

v1_existing : the current probe verbatim (sanity-checked against the main cache)
v2_short    : shorter single-concept Noul question
v3_choice   : two-option Choice question (score = P(yes option))

Reads runs/banking77/discovery/repair_set.csv; writes repair_scores.json and
repair_audit_sheet.csv; prints accuracy@0.5, AUC, separation, and the forced
known-failure case scores per variant.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(5)

from sklearn.metrics import roc_auc_score  # noqa: E402

from banking77_baseline import ResultCache, cache_key, question_set  # noqa: E402
from tinyjev import load as load_agent  # noqa: E402
from tinyjev.families import softmax  # noqa: E402

DISCOVERY = HERE / "runs" / "banking77" / "discovery"
MAIN_CACHE = HERE / "runs" / "banking77" / "jev_cache.sqlite"

QUESTIONS = {
    "card": {
        "v1_existing": {"type": "noul", "instructions":
                        "Is a physical or virtual payment card central to the customer's issue or request?"},
        "v2_short": {"type": "noul", "instructions": "Is this message about a payment card?"},
        "v3_choice": {"type": "choice", "instructions": "Which is true of this message?",
                      "criteria": {"yes": "the message is about a payment card",
                                   "no": "the message is not about a payment card"}},
    },
    "unauth": {
        "v1_existing": {"type": "noul", "instructions":
                        "Does the customer say they do not recognize or did not authorize an account event, card payment, or withdrawal?"},
        "v2_short": {"type": "noul", "instructions": "Is this message about a payment or withdrawal the customer did not make?"},
        "v3_choice": {"type": "choice", "instructions": "Which is true of this message?",
                      "criteria": {"yes": "the customer reports a payment or withdrawal they did not make",
                                   "no": "nothing unauthorized is reported"}},
    },
    "pin": {
        "v1_existing": {"type": "noul", "instructions":
                        "Is the issue about signing in, a passcode, identity verification, account access, a lost device, or account security?"},
        "v2_short": {"type": "noul", "instructions": "Is this message about a PIN, passcode, or sign-in problem?"},
        "v3_choice": {"type": "choice", "instructions": "Which is true of this message?",
                      "criteria": {"yes": "the message is about a PIN, passcode, or sign-in problem",
                                   "no": "it is not about a PIN, passcode, or sign-in"}},
    },
}
CACHE_PROBE = {"card": "physical_or_virtual_card", "unauth": "unauthorised_activity",
               "pin": "account_access_or_security"}


def score_of(qtype, keys, probs):
    if qtype == "boolean":
        return float(probs[1])
    return float(probs[keys.index("yes")])


def main():
    rows = list(csv.DictReader((DISCOVERY / "repair_set.csv").open()))
    print(f"repair set: {len(rows)} rows")

    agent = load_agent("TinyJev-0.6B", backend="torch", device="cuda")
    fam, model, dev = agent.family, agent.backbone.model, agent.backbone.device

    def extract(pairs, question):
        """pairs: list of texts; returns list of scores (one question each)."""
        q = dict(question)
        q["id"] = "q"
        if q["type"] == "noul":
            q["type"] = "boolean"
        qtype = q["type"]
        out = []
        for s in range(0, len(pairs), 64):
            chunk = pairs[s:s + 64]
            encs = [fam.encode({"id": f"x{i}", "state": t, "questions": [dict(q)]})
                    for i, t in enumerate(chunk)]
            rows_tok = [e.prefix + e.rows[0] for e in encs]
            width = max(len(r) for r in rows_tok)
            ids = torch.full((len(rows_tok), width), fam.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(rows_tok), width), dtype=torch.long)
            for i, r in enumerate(rows_tok):
                ids[i, :len(r)] = torch.tensor(r)
                att[i, :len(r)] = 1
            with torch.inference_mode():
                hs = model(input_ids=ids.to(dev), attention_mask=att.to(dev),
                           use_cache=False).last_hidden_state.float().cpu().numpy()
            for j, e in enumerate(encs):
                h = hs[j, :len(rows_tok[j])]
                z = fam.logits([h], e)
                probs = softmax(z["q"])
                out.append(score_of(qtype, e.questions[0]["keys"], probs))
        return out

    # warmup
    extract(["This is a warmup message about a card."], QUESTIONS["card"]["v1_existing"])

    results = {}
    sheet = []
    for concept in ("card", "unauth", "pin"):
        sub = [r for r in rows if r["concept"] == concept]
        texts = [r["text"] for r in sub]
        y = np.array([int(r["label"]) for r in sub])
        concept_res = {}
        variant_scores = {}
        for vname, q in QUESTIONS[concept].items():
            t0 = time.perf_counter()
            scores = np.array(extract(texts, q))
            dt = time.perf_counter() - t0
            variant_scores[vname] = scores
            acc = float(np.mean((scores >= 0.5).astype(int) == y))
            auc = float(roc_auc_score(y, scores))
            sep = float(scores[y == 1].mean() - scores[y == 0].mean())
            concept_res[vname] = {"accuracy_at_0.5": round(acc, 4), "auc": round(auc, 4),
                                  "separation": round(sep, 4),
                                  "extract_s": round(dt, 2),
                                  "mean_pos": round(float(scores[y == 1].mean()), 4),
                                  "mean_neg": round(float(scores[y == 0].mean()), 4)}
            print(f"{concept:7s} {vname:12s} acc={acc:.3f} auc={auc:.3f} sep={sep:+.3f} "
                  f"(pos {scores[y == 1].mean():.3f} / neg {scores[y == 0].mean():.3f})", flush=True)
        results[concept] = concept_res
        for i, r in enumerate(sub):
            sheet.append({"concept": concept, "bucket": r["bucket"], "label": r["label"],
                          "v1_existing": round(float(variant_scores["v1_existing"][i]), 4),
                          "v2_short": round(float(variant_scores["v2_short"][i]), 4),
                          "v3_choice": round(float(variant_scores["v3_choice"][i]), 4),
                          "text": r["text"]})
        # sanity: v1 vs main cache
        try:
            labels = None
            from datasets import load_dataset
            labels = load_dataset("PolyAI/banking77")["train"].features["label"].names
            qmap = question_set(labels, "both")
            cache = ResultCache(MAIN_CACHE)
            diffs = []
            for i, r in enumerate(sub):
                if int(r["split_row"]) < 0:
                    continue
                k = cache_key(r["text"], "TinyJev-0.6B", qmap)
                hit = cache.get(k)
                if hit is None:
                    continue
                cached = float(hit[0]["answers"]["probe__" + CACHE_PROBE[concept]]["noul"])
                diffs.append(abs(cached - variant_scores["v1_existing"][i]))
            cache.close()
            results[concept]["v1_cache_check"] = {
                "n_compared": len(diffs),
                "max_abs_diff": round(float(max(diffs)), 5) if diffs else None}
            print(f"{concept:7s} v1 vs cache: n={len(diffs)} max|diff|={max(diffs) if diffs else None:.5f}")
        except Exception as exc:  # noqa: BLE001
            print("cache check skipped:", exc)

    (DISCOVERY / "repair_scores.json").write_text(json.dumps(results, indent=2))
    with (DISCOVERY / "repair_audit_sheet.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "bucket", "label", "v1_existing",
                                          "v2_short", "v3_choice", "text"])
        w.writeheader()
        w.writerows(sheet)
    print("\n=== known-failure case scores ===")
    for row in sheet:
        if row["bucket"] == "known_failure":
            print(f"[{row['concept']}] label={row['label']} v1={row['v1_existing']:+.3f} "
                  f"v2={row['v2_short']:+.3f} v3={row['v3_choice']:+.3f} | {row['text'][:70]}")
    print("\nwritten:", DISCOVERY / "repair_scores.json", "and repair_audit_sheet.csv")


if __name__ == "__main__":
    main()
