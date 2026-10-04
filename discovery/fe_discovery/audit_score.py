#!/usr/bin/env python3
"""Score audit v3 vs adjudicated gold -> audit_report.json (freeze gate input).

Gold labels: researcher adjudications where present, else annotator_v3.
Reports per-concept: confusion matrix, macro-F1, agreement; overall: means,
annotator-reference agreement, and probability discrimination (top-1 probability
on correct vs incorrect predictions)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE / "discovery" / "fe_discovery"))
AUDIT = HERE / "runs" / "banking77" / "fe_discovery" / "audit"


def main():
    rows = json.loads((AUDIT / "audit_v3_rows.json").read_text())
    adj = json.loads((AUDIT / "adjudications.json").read_text())["labels"]
    from sklearn.metrics import confusion_matrix, f1_score
    from audit_v3 import CONCEPTS_V3

    opts_by = {c: [i for i, _ in opts] for c, q, opts in CONCEPTS_V3}
    per_concept = {}
    disc_ok, disc_bad = [], []
    agree_m, ann_agree = [], []
    for cname in opts_by:
        cr = [r for r in rows if r["concept_v3"] == cname]
        gold = [adj.get(str(r["pair_id"]), r["annotator_v3"]) for r in cr]
        pred = [r["model_v3"] for r in cr]
        oids = opts_by[cname]
        f1 = f1_score(gold, pred, average="macro", labels=oids, zero_division=0)
        cm = confusion_matrix(gold, pred, labels=oids).tolist()
        ag = float(np.mean([g == p for g, p in zip(gold, pred)]))
        agree_m.append(ag)
        ann_agree.append(float(np.mean([r["annotator_v3"] == r["model_v3"] for r in cr])))
        for r, g in zip(cr, gold):
            top = max(r["probs_v3"].values())
            (disc_ok if r["model_v3"] == g else disc_bad).append(top)
        per_concept[cname] = {"n": len(cr), "macro_f1": round(float(f1), 3),
                              "agreement_vs_gold": round(ag, 3),
                              "confusion (rows=gold, cols=model)": cm, "options": oids}
    report = {
        "vocabulary": "v3-merged",
        "n_pairs": len(rows), "n_adjudicated": len(adj),
        "concepts": per_concept,
        "overall": {
            "mean_concept_macro_f1": round(float(np.mean(
                [c["macro_f1"] for c in per_concept.values()])), 3),
            "min_concept_macro_f1": round(float(np.min(
                [c["macro_f1"] for c in per_concept.values()])), 3),
            "mean_agreement": round(float(np.mean(agree_m)), 3),
            "annotator_reference_mean_agreement": round(float(np.mean(ann_agree)), 3),
            "mean_topprob_correct": round(float(np.mean(disc_ok)), 3) if disc_ok else None,
            "mean_topprob_incorrect": round(float(np.mean(disc_bad)), 3) if disc_bad else None,
        },
        "method": "extractor v3 (merged negative/unclear) vs gold = researcher adjudication "
                  "(all disagreements + 20% subsample) blended with annotator labels elsewhere; "
                  "annotator-only agreement reported as reference",
    }
    (AUDIT / "audit_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["overall"], indent=1))
    for c, v in per_concept.items():
        print(f"  {c:18s} n={v['n']:2d} f1 {v['macro_f1']:.2f} agree {v['agreement_vs_gold']:.2f}")
    print("written audit_report.json")


if __name__ == "__main__":
    main()
