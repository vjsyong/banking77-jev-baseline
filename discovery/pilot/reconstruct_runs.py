#!/usr/bin/env python3
"""One-off: rebuild run summaries for completed factorial runs from the archive.

Replays accepted rounds' applied actions + pruned removals onto B0, validates
the reconstructed bank against the run log (size, choice count, CV recomputed
deterministically on the same folds), and writes partial_full.json for --resume.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "discovery"))
sys.path.insert(0, str(HERE / "discovery" / "pilot"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from extract_probes import ProbeStore, text_key  # noqa: E402
from run_pilot import bank_cv  # noqa: E402

OUT = HERE / "runs/banking77/discovery/factorial"
LOG = HERE / "runs/factorial_full.log"

records = [json.loads(l) for l in (OUT / "archive_full.jsonl").read_text().splitlines() if l.strip()]
runs = {}
for r in records:
    runs.setdefault((r["cell"], r["seed"]), []).append(r)
print(f"archive records: {len(records)} | runs seen: {len(runs)}")

logtxt = LOG.read_text()
val = {}
for m in re.finditer(r"== (\S+) seed ([\d.]+): bank (\d+) \(choices (\d+)\), cv ([\d.]+)", logtxt):
    val[(m.group(1), float(m.group(2)))] = (int(m.group(3)), int(m.group(4)), float(m.group(5)))

idx = pd.read_csv(HERE / "runs/banking77/discovery/smallregime_index.csv")["csv_index"].to_numpy()
pred = pd.read_csv(HERE / "runs/banking77/jev_predictions.csv").iloc[idx].reset_index(drop=True)
texts, y = pred["text"].tolist(), pred["label"].to_numpy()
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
             .split(np.zeros(len(y)), y))
B0 = [{"id": p["id"], "question": p["question"], "format": "noul"}
      for p in json.loads((HERE / "probes.json").read_text())]
store = ProbeStore(HERE / "runs/banking77/discovery/pilot/probe_scores.sqlite")


def matrix(bank):
    data = []
    for p in bank:
        if p.get("format") == "choice":
            for o in p.get("options") or []:
                data.append([store.choice_score_of(p["id"], text_key(t), o) for t in texts])
        else:
            data.append([store.score_of(p["id"], text_key(t)) for t in texts])
    return np.array(data, dtype=np.float64).T


out = []
for (cell, seed), recs in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    if len(recs) != 3:
        print(f"{cell} s{seed}: incomplete ({len(recs)} rounds) - skipped")
        continue
    bank = [dict(p) for p in B0]
    for r in recs:
        if r.get("accepted"):
            for a in r["applied"]:
                if a["action"] == "add":
                    rec = {"id": a["id"], "question": a["question"], "format": a["format"]}
                    if a["format"] == "choice":
                        rec["options"] = a["options"]
                    bank.append(rec)
                elif a["action"] == "revise":
                    bank = [p for p in bank if p["id"] != a.get("replaced", "")]
                    rec = {"id": a["id"], "question": a["question"], "format": a["format"]}
                    if a["format"] == "choice":
                        rec["options"] = a["options"]
                    bank.append(rec)
                else:
                    bank = [p for p in bank if p["id"] != a["id"]]
        for pid in r["pruned"]:
            bank = [p for p in bank if p["id"] != pid]
    X = matrix(bank)
    _, cv_best = bank_cv(X, y, folds)
    n_ch = sum(1 for p in bank if p.get("format") == "choice")
    v = val.get((cell, seed))
    ok = (v is not None and v[0] == len(bank) and v[1] == n_ch
          and abs(v[2] - cv_best) < 5e-4)
    print(f"{cell} s{seed}: bank {len(bank)} (choices {n_ch}), cv {cv_best:.4f} | "
          f"log {v} | {'PASS' if ok else 'MISMATCH'}")
    if ok:
        out.append({
            "cell": cell, "repr": recs[0]["repr"], "sel": recs[0]["sel"], "seed": seed,
            "final_bank": bank, "final_cv_best": round(float(cv_best), 4),
            "bank_size": len(bank), "n_choices": n_ch,
            "t_teacher": round(sum(r["teacher_wall_s"] for r in recs), 1),
            "t_extract_1k": round(sum(r["extract_added_s"] for r in recs), 1),
            "teacher_calls": len(recs), "archive": [], "trace": recs,
        })

(OUT / "partial_full.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote partial_full.json with {len(out)}/{len(runs)} runs")
