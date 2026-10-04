#!/usr/bin/env python3
"""Compare a batched-run cache against the single-path backup (equivalence record).

Usage: compare_batched_run.py <batched_cache.sqlite> <single_backup.sqlite> [out.json]

Compares every cache key present in both: choice strings (flips), noul values,
option probabilities. Prints stats and optionally writes JSON.
"""
import json
import sqlite3
import sys


def load(path):
    db = sqlite3.connect(path)
    rows = db.execute("SELECT cache_key, payload FROM jev_cache").fetchall()
    db.close()
    return {k: json.loads(v) for k, v in rows}


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    out_json = sys.argv[3] if len(sys.argv) > 3 else None
    A, B = load(a_path), load(b_path)  # A = batched, B = single-path reference
    common = [k for k in A if k in B]

    choice_n = flips = 0
    flip_list = []
    noul = []
    prob_max = 0.0
    prob_where = None
    for k in common:
        aa, bb = A[k].get("answers", {}), B[k].get("answers", {})
        for qid, xa in aa.items():
            xb = bb.get(qid)
            if xb is None:
                continue
            if "choice" in xa and "choice" in xb:
                choice_n += 1
                if xa["choice"] != xb["choice"]:
                    flips += 1
                    if len(flip_list) < 20:
                        flip_list.append({"key": k[:16], "qid": qid,
                                          "batched": xa["choice"], "single": xb["choice"]})
            if "noul" in xa and "noul" in xb:
                noul.append(abs(float(xa["noul"]) - float(xb["noul"])))
            pa, pb = xa.get("probabilities") or {}, xb.get("probabilities") or {}
            if pa and pb:
                for kk in set(pa) & set(pb):
                    d = abs(float(pa[kk]) - float(pb[kk]))
                    if d > prob_max:
                        prob_max = d
                        prob_where = {"key": k[:16], "qid": qid, "option": kk}

    noul_sorted = sorted(noul, reverse=True)
    stats = {
        "cache_a": a_path, "cache_b": b_path,
        "entries_a": len(A), "entries_b": len(B), "common_keys": len(common),
        "choice_comparisons": choice_n, "choice_flips": flips, "flips_sample": flip_list,
        "noul_comparisons": len(noul),
        "noul_max": noul_sorted[0] if noul_sorted else 0.0,
        "noul_p95": noul_sorted[int(0.05 * len(noul_sorted))] if noul_sorted else 0.0,
        "noul_mean": sum(noul) / len(noul) if noul else 0.0,
        "noul_over_1e-3": sum(1 for d in noul if d > 1e-3),
        "noul_over_5e-3": sum(1 for d in noul if d > 5e-3),
        "noul_over_1e-2": sum(1 for d in noul if d > 1e-2),
        "prob_max": prob_max, "prob_max_where": prob_where,
    }
    print(json.dumps(stats, indent=2))
    if out_json:
        with open(out_json, "w") as f:
            json.dump(stats, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
