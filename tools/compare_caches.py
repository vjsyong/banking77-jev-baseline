#!/usr/bin/env python3
"""Compare two jev caches: verify in-process shard output matches HTTP output.

For every cache key present in the second cache, compares the stored payloads
(noul values, choice, probabilities) and reports max absolute differences.
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
    a, b = load(a_path), load(b_path)
    common = [k for k in b if k in a]
    print(f"cache A {a_path}: {len(a)} entries; cache B {b_path}: {len(b)} entries; "
          f"common keys: {len(common)}")
    if not common:
        print("NO COMMON KEYS — cannot compare")
        return 1
    max_noul_diff = 0.0
    max_prob_diff = 0.0
    choice_mismatch = 0
    for k in common:
        an = a[k].get("answers", {})
        bn = b[k].get("answers", {})
        for qid, av in an.items():
            bv = bn.get(qid)
            if bv is None:
                print(f"  key {k[:12]}: missing {qid} in B")
                continue
            if "noul" in av:
                max_noul_diff = max(max_noul_diff, abs(float(av["noul"]) - float(bv["noul"])))
            if "choice" in av:
                if av["choice"] != bv["choice"]:
                    choice_mismatch += 1
                    print(f"  key {k[:12]}: choice A={av['choice']} B={bv['choice']}")
                pa, pb = av.get("probabilities") or {}, bv.get("probabilities") or {}
                for opt in set(pa) | set(pb):
                    max_prob_diff = max(max_prob_diff, abs(float(pa.get(opt, 0)) - float(pb.get(opt, 0))))
    print(f"max noul abs diff:  {max_noul_diff:.6g}")
    print(f"max prob abs diff:  {max_prob_diff:.6g}")
    print(f"choice mismatches:  {choice_mismatch}")
    ok = max_noul_diff < 1e-6 and max_prob_diff < 1e-4 and choice_mismatch == 0
    print("EQUIVALENT" if ok else "DIFFERS — inspect")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
