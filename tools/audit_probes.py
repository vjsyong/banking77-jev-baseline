#!/usr/bin/env python3
"""Probe-value audit for the BANKING77 Jev feature extraction (no new inference).

Per probe column: distribution stats, saturation fractions, correlation structure,
intent-conditional separability (eta^2 on per-intent means), and high/low example
texts for manual inspection.

Reads runs/banking77/jev_features.csv; writes runs/banking77/analysis/probe_audit.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
CSV = HERE / "runs" / "banking77" / "jev_features.csv"
OUT = HERE / "runs" / "banking77" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

KNOWN = {"split", "row", "label", "text", "text_sha256"}


def main():
    df = pd.read_csv(CSV)
    probes = [c for c in df.columns if c not in KNOWN]
    print(f"rows: {len(df)} | probes: {len(probes)}")
    result = {"n_rows": len(df), "probes": {}, "analysis": {}}

    stats = {}
    for c in probes:
        x = df[c].astype(float)
        stats[c] = {
            "mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4),
            "min": round(float(x.min()), 4), "q05": round(float(x.quantile(0.05)), 4),
            "q25": round(float(x.quantile(0.25)), 4), "q50": round(float(x.quantile(0.50)), 4),
            "q75": round(float(x.quantile(0.75)), 4), "q95": round(float(x.quantile(0.95)), 4),
            "max": round(float(x.max()), 4),
            "frac_lt_05": round(float((x < 0.05).mean()), 4),
            "frac_gt_95": round(float((x > 0.95).mean()), 4),
            "frac_451_549": round(float(((x > 0.451) & (x < 0.549)).mean()), 4),
        }
    result["probes"] = stats

    # correlation structure
    corr = df[probes].astype(float).corr()
    pairs = []
    for i, a in enumerate(probes):
        for b in probes[i + 1:]:
            pairs.append((a, b, float(corr.loc[a, b])))
    pairs.sort(key=lambda t: -abs(t[2]))
    result["analysis"]["top_corr_pairs"] = [[a, b, round(r, 3)] for a, b, r in pairs[:15]]
    result["analysis"]["max_abs_corr_per_probe"] = {
        a: [b, round(r, 3)] for a, b, r in pairs if abs(r) >= 0.5
    }

    # intent-conditional separability: eta^2 = var(per-intent means, weighted) / total var
    eta = {}
    by = df.groupby("label")[probes].mean()
    counts = df.groupby("label").size()
    for c in probes:
        x = df[c].astype(float)
        grand = x.mean()
        between = float((((by[c] - grand) ** 2) * counts).sum() / counts.sum())
        total = float(x.var(ddof=0))
        eta[c] = round(between / total, 4) if total > 0 else 0.0
    result["analysis"]["eta2"] = dict(sorted(eta.items(), key=lambda kv: -kv[1]))

    # per-intent extremes for a showcase set
    showcase = ["physical_or_virtual_card", "cash_or_atm", "bank_transfer",
                "unauthorised_activity", "transaction_is_pending", "cancel_or_reverse",
                "currency_conversion", "account_access_or_security"]
    extremes = {}
    for c in showcase:
        means = by[c].sort_values()
        n = counts
        top = [(str(k), round(float(v), 3), int(n[k])) for k, v in means.tail(4).items()]
        bot = [(str(k), round(float(v), 3), int(n[k])) for k, v in means.head(4).items()]
        extremes[c] = {"top_intents": top, "bottom_intents": bot}
    result["analysis"]["intent_extremes"] = extremes

    # examples for manual inspection
    examples = {}
    for c in ["physical_or_virtual_card", "unauthorised_activity", "transaction_is_pending",
              "reports_a_problem", "requests_an_action"]:
        top = df.nlargest(3, c)[["label", c, "text"]]
        bot = df.nsmallest(3, c)[["label", c, "text"]]
        examples[c] = {
            "high": [{"label": r["label"], "p": round(float(r[c]), 3), "text": str(r["text"])[:90]}
                     for _, r in top.iterrows()],
            "low": [{"label": r["label"], "p": round(float(r[c]), 3), "text": str(r["text"])[:90]}
                    for _, r in bot.iterrows()],
        }
    result["analysis"]["examples"] = examples

    (OUT / "probe_audit.json").write_text(json.dumps(result, indent=2))

    # console summary
    print("\n== probe stats (mean | std | <0.05 | >0.95 | near-0.5) ==")
    for c in probes:
        s = stats[c]
        print(f"{c:32s} {s['mean']:.3f} | {s['std']:.3f} | {s['frac_lt_05']:.3f} | "
              f"{s['frac_gt_95']:.3f} | {s['frac_451_549']:.3f}")
    print("\n== top |r| pairs ==")
    for a, b, r in pairs[:10]:
        print(f"  {a} ~ {b}: {r:+.3f}")
    print("\n== eta^2 by probe (separability, high = more intent-dependent) ==")
    for k, v in list(result["analysis"]["eta2"].items())[:8]:
        print(f"  {k}: {v}")
    print("  ...")
    for k, v in list(result["analysis"]["eta2"].items())[-4:]:
        print(f"  {k}: {v}")
    print("\nwritten", OUT / "probe_audit.json")


if __name__ == "__main__":
    main()
