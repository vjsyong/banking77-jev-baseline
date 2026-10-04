#!/usr/bin/env python3
"""Bounded sensitivity check for pilot probe-id wording collisions.

Pilot-era cache semantics served one stored extraction per id even when runs
used slightly different wording under the same slug. This re-extracts every
distinct variant of the colliding ids (fresh namespaced ids) on the 1k training
texts and reports per-id delta distributions + run exposure, bounding the
substitution effect on the pilot's numbers.
"""
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))
for _p in ("/home/xrim/taildash/client", "/home/xrim/progtrack/client"):
    if Path(_p).is_dir():
        sys.path.insert(0, _p)
        break
try:
    from taildash import TaskMonitor as _TaskMonitor
except Exception:  # noqa: BLE001
    _TaskMonitor = None

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from extract_probes import ProbeExtractor, ProbeStore, text_key  # noqa: E402

PILOT = HERE / "runs" / "banking77" / "discovery" / "pilot"
_ACTIVE_MONITOR = None


class _NullMonitor:
    def log(self, *a, **k):
        pass

    def update(self, *a, **k):
        pass

    def complete(self, *a, **k):
        pass

    def fail(self, *a, **k):
        pass


def main():
    global _ACTIVE_MONITOR
    s = json.loads((PILOT / "summary_full.json").read_text())
    variants = collections.defaultdict(set)
    runs_with = collections.defaultdict(set)
    for r in s["runs"]:
        for p in r["final_bank"]:
            variants[p["id"]].add(p["question"])
            runs_with[p["id"]].add(f"{r['policy']}/s{r['seed']}")
    coll = {k: sorted(v) for k, v in variants.items() if len(v) > 1}

    mon = (_TaskMonitor(server="http://localhost:8080",
                        title="pilot collision sensitivity check",
                        total=max(1, len(coll)), agent_name="banking77-sensitivity")
           if _TaskMonitor else _NullMonitor())
    _ACTIVE_MONITOR = mon
    mon.log(f"colliding ids: {len(coll)}; variants {sum(len(v) for v in coll.values())}")

    idx = pd.read_csv(HERE / "runs/banking77/discovery/smallregime_index.csv")["csv_index"].to_numpy()
    texts = pd.read_csv(HERE / "runs/banking77/jev_predictions.csv").iloc[idx]["text"].tolist()

    store = ProbeStore(HERE / "runs/banking77/discovery/pilot/probe_scores.sqlite")
    ex = ProbeExtractor(store)

    t0 = time.perf_counter()
    report = {}
    for pid, qs in sorted(coll.items()):
        scores = {}
        for k, q in enumerate(qs):
            sid = f"sens_{pid}__v{k}"
            ex.extract_probe(sid, q, texts)
            scores[k] = np.array([store.score_of(sid, text_key(t)) for t in texts])
        pairs = []
        for a in range(len(qs)):
            for b in range(a + 1, len(qs)):
                d = np.abs(scores[a] - scores[b])
                pairs.append({"a": a, "b": b,
                              "mean_abs_delta": round(float(d.mean()), 4),
                              "p95_abs_delta": round(float(np.percentile(d, 95)), 4),
                              "max_abs_delta": round(float(d.max()), 4),
                              "pearson": round(float(np.corrcoef(scores[a], scores[b])[0, 1]), 4)})
        report[pid] = {"n_variants": len(qs), "variants": qs,
                       "exposed_runs": sorted(runs_with[pid]), "pairs": pairs}
        print(f"{pid}: {len(qs)} variants | " + " | ".join(
            f"meanΔ {p['mean_abs_delta']:.3f} p95Δ {p['p95_abs_delta']:.3f} r {p['pearson']:.3f}"
            for p in pairs), flush=True)
        mon.update(len(report), message=f"{len(report)}/{len(coll)} ids done")

    all_means = [p["mean_abs_delta"] for r in report.values() for p in r["pairs"]]
    all_p95 = [p["p95_abs_delta"] for r in report.values() for p in r["pairs"]]
    all_max = [p["max_abs_delta"] for r in report.values() for p in r["pairs"]]
    all_r = [p["pearson"] for r in report.values() for p in r["pairs"]]
    agg = {
        "n_ids": len(report), "n_pairs": len(all_means),
        "mean_of_mean_abs_delta": round(float(np.mean(all_means)), 4),
        "median_of_mean_abs_delta": round(float(np.median(all_means)), 4),
        "p95_of_mean_abs_delta": round(float(np.percentile(all_means, 95)), 4),
        "worst_pair_max_abs_delta": round(float(np.max(all_max)), 4),
        "mean_p95_abs_delta": round(float(np.mean(all_p95)), 4),
        "min_pearson": round(float(np.min(all_r)), 4),
        "mean_pearson": round(float(np.mean(all_r)), 4),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    out = {"aggregate": agg, "per_id": report}
    (PILOT / "collision_sensitivity.json").write_text(json.dumps(out, indent=2))
    print("\nAGGREGATE:", json.dumps(agg, indent=2))
    mon.complete(f"sensitivity done: {agg['n_pairs']} pairs, mean|Δ| {agg['mean_of_mean_abs_delta']}")
    store.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        if _ACTIVE_MONITOR is not None:
            _ACTIVE_MONITOR.fail(f"{type(exc).__name__}: {exc}")
        raise
