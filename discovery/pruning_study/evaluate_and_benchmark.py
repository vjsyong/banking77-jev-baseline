#!/usr/bin/env python3
"""Pruning study §4: frozen deployment points — test evaluation + latency benchmark.

Reads ONLY pruning_study/frozen_points.json (points selected & frozen on training
data by run_pruning.py) and its hash is recorded for provenance.

Test evaluation: frozen learner (StandardScaler -> LR C=1) fit on the 1k, applied
to the 3,080-row test split for each config {unpruned, c_tol005, d_tol010} x 3
seed banks; paired vs-unpruned comparisons (per-message diffs, exact McNemar,
bootstrap 95% CIs over messages).

Latency: per config, single-request p50/p95 (50 texts) and batched ms/text +
throughput (1,024 texts, chunk 64), including the downstream classifier, measured
in the deployed pipeline shape (per-question forwards, single question per
forward), same hardware (RTX 3090), warmup 3 requests, fixed text sample.

The test split has informed earlier redesigns: this evaluation is EXPLORATORY.
"""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
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
import torch  # noqa: E402
torch.set_num_threads(5)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from extract_probes import ProbeExtractor, ProbeStore, text_key  # noqa: E402
from tinyjev import load as load_agent  # noqa: E402
from tinyjev.families import softmax  # noqa: E402

RUNS = HERE / "runs" / "banking77"
STUDY = HERE / "discovery" / "pruning_study"
SNAP = STUDY / "snapshots"
BANKS = ("mixed-strict-s0.3", "mixed-strict-s0.7", "mixed-strict-s1.0")
CONFIGS = ("unpruned", "c_corrected_tol005", "d_corrected_tol010")
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


def lr_pipeline():
    return Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))])


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value given discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def main():
    global _ACTIVE_MONITOR
    frozen_raw = (STUDY / "frozen_points.json").read_bytes()
    frozen_sha = hashlib.sha256(frozen_raw).hexdigest()
    frozen = json.loads(frozen_raw)

    mon = (_TaskMonitor(server="http://localhost:8080",
                        title="pruning study: freeze-eval + latency benchmark (9 configs)",
                        total=len(BANKS) + len(BANKS) * len(CONFIGS), agent_name="banking77-pruning")
           if _TaskMonitor else _NullMonitor())
    _ACTIVE_MONITOR = mon
    mon.log(f"frozen_points sha256 {frozen_sha[:16]}…; evaluating {len(BANKS)}x3 configs")

    idx = pd.read_csv(RUNS / "discovery" / "smallregime_index.csv")["csv_index"].to_numpy()
    pred = pd.read_csv(RUNS / "jev_predictions.csv")
    train = pred.iloc[idx].reset_index(drop=True)
    test = pred[pred["split"] == "test"].reset_index(drop=True)
    y1k = train["label"].to_numpy()
    yte = test["label"].to_numpy()
    texts_1k = train["text"].tolist()
    texts_te = test["text"].tolist()

    store = ProbeStore(RUNS / "discovery" / "pilot" / "probe_scores.sqlite")
    ex = ProbeExtractor(store)

    def probe_matrix(probe_defs, texts, slot_prefix):
        """Extract (fresh, run-scoped) and build matrix + colnames for defs."""
        cols, data = [], []
        for p in probe_defs:
            slot = f"{slot_prefix}::{p['id']}"
            if p.get("format") == "choice":
                ex.extract_choice(slot, p["question"], p["options"], texts)
                for o in p["options"]:
                    cols.append(f"{p['id']}|{o}")
                    data.append([store.choice_score_of(slot, text_key(t), o) for t in texts])
            else:
                ex.extract_probe(slot, p["question"], texts)
                cols.append(p["id"])
                data.append([store.score_of(slot, text_key(t)) for t in texts])
        return np.array(data, dtype=np.float64).T, cols

    results, latency = {}, {}
    t0_all = time.perf_counter()
    for bi, bank_id in enumerate(BANKS, 1):
        snap = json.loads((SNAP / f"{bank_id}.json").read_text())
        X1k_full = np.load(SNAP / f"{bank_id}.npz")["X"]
        all_probes = snap["probes"]
        full_cols = [tuple(c) for c in snap["column_order"]]
        print(f"\n== {bank_id}: {len(all_probes)} probes ==", flush=True)

        # test features for the full (unpruned) bank, once
        t0 = time.perf_counter()
        Xte_full, te_cols = probe_matrix(all_probes, texts_te, f"dep::{bank_id}")
        print(f"   test extraction: {time.perf_counter()-t0:.0f}s ({len(te_cols)} cols)", flush=True)

        col_idx = {name: j for j, name in enumerate(full_cols)}
        bank_res = {}
        for cfg in CONFIGS:
            if cfg == "unpruned":
                probes_cfg = all_probes
            else:
                probes_cfg = frozen[bank_id][cfg]["kept_probes"]
            keep = [col_idx[(p["id"], o) if o is not None else (p["id"], None)]
                    for p in probes_cfg
                    for o in (p["options"] if p.get("format") == "choice" else [None])]
            X1k = X1k_full[:, keep]
            Xte = Xte_full[:, keep]
            model = lr_pipeline()
            model.fit(X1k, y1k)
            pr = model.predict(Xte)
            f1 = float(f1_score(yte, pr, average="macro", zero_division=0))
            acc = float(accuracy_score(yte, pr))
            bank_res[cfg] = {"macro_f1": round(f1, 4), "accuracy": round(acc, 4),
                             "n_probes": len(probes_cfg), "n_columns": len(keep),
                             "n_choice_options": sum(len(p["options"]) for p in probes_cfg
                                                     if p.get("format") == "choice"),
                             "preds": pr}
            print(f"   {cfg:18s}: {len(probes_cfg):2d}p/{len(keep):2d}c | f1 {f1:.4f} acc {acc:.4f}",
                  flush=True)

        # paired stats vs unpruned
        ref_ok = bank_res["unpruned"]["preds"] == yte
        paired = {}
        rng = np.random.default_rng(77)
        for cfg in ("c_corrected_tol005", "d_corrected_tol010"):
            ok = bank_res[cfg]["preds"] == yte
            b = int(np.sum(ref_ok & ~ok))
            c = int(np.sum(~ref_ok & ok))
            # bootstrap CI on metric differences
            diffs_f1, diffs_acc = [], []
            n = len(yte)
            idx_all = np.arange(n)
            for _ in range(1000):
                rs = rng.choice(idx_all, size=n, replace=True)
                f1r = f1_score(yte[rs], bank_res["unpruned"]["preds"][rs], average="macro", zero_division=0)
                f1p = f1_score(yte[rs], bank_res[cfg]["preds"][rs], average="macro", zero_division=0)
                diffs_f1.append(f1p - f1r)
                diffs_acc.append(ok[rs].mean() - ref_ok[rs].mean())
            paired[cfg] = {
                "macro_f1_delta_pp": round(100 * (bank_res[cfg]["macro_f1"] - bank_res["unpruned"]["macro_f1"]), 2),
                "accuracy_delta_pp": round(100 * (bank_res[cfg]["accuracy"] - bank_res["unpruned"]["accuracy"]), 2),
                "b_unpruned_right_pruned_wrong": b,
                "c_unpruned_wrong_pruned_right": c,
                "mcnemar_p": round(mcnemar_exact(b, c), 4),
                "bootstrap95_f1_delta_pp": [round(100 * float(np.percentile(diffs_f1, 2.5)), 2),
                                            round(100 * float(np.percentile(diffs_f1, 97.5)), 2)],
                "bootstrap95_acc_delta_pp": [round(100 * float(np.percentile(diffs_acc, 2.5)), 2),
                                             round(100 * float(np.percentile(diffs_acc, 97.5)), 2)],
            }
            print(f"   paired {cfg}: Δf1 {paired[cfg]['macro_f1_delta_pp']:+.2f}pp "
                  f"(95% CI {paired[cfg]['bootstrap95_f1_delta_pp']}), McNemar p={paired[cfg]['mcnemar_p']}",
                  flush=True)

        results[bank_id] = {cfg: {k: v for k, v in bank_res[cfg].items() if k != "preds"}
                            for cfg in CONFIGS}
        results[bank_id]["paired_vs_unpruned"] = paired
        mon.update(bi, message=f"{bank_id}: evaluated")

    # ---------------- latency benchmark ----------------
    print("\n== latency benchmark ==", flush=True)
    fam = ex.fam
    model = ex.model
    dev = ex.dev
    rng = np.random.default_rng(77)
    single_idx = sorted(rng.choice(len(texts_te), size=50, replace=False).tolist())
    batch_texts = texts_te[:1024]

    def run_bank_timing(probes_cfg):
        """Deployed pipeline shape: per-question forwards; returns single-request ms list."""
        qs = []
        for p in probes_cfg:
            if p.get("format") == "choice":
                qs.append({"id": p["id"], "type": "choice", "instructions": p["question"],
                           "criteria": {o: None for o in p["options"]}})
            else:
                qs.append({"id": p["id"], "type": "boolean", "instructions": p["question"]})

        def features_for(text_list):
            n = len(text_list)
            X = np.empty((n, sum(len(p["options"]) if p.get("format") == "choice" else 1
                                 for p in probes_cfg)))
            col = 0
            for q in qs:
                encs = [fam.encode({"id": f"x{i}", "state": t, "questions": [dict(q)]})
                        for i, t in enumerate(text_list)]
                rows = [e.prefix + e.rows[0] for e in encs]
                width = max(len(r) for r in rows)
                ids = torch.full((n, width), fam.pad_token_id, dtype=torch.long)
                att = torch.zeros((n, width), dtype=torch.long)
                for i, r in enumerate(rows):
                    ids[i, :len(r)] = torch.tensor(r)
                    att[i, :len(r)] = 1
                with torch.inference_mode():
                    hs = model(input_ids=ids.to(dev), attention_mask=att.to(dev),
                               use_cache=False).last_hidden_state.float().cpu().numpy()
                for j, e in enumerate(encs):
                    z = fam.logits([hs[j, :len(rows[j])]], e)
                    probs = softmax(z[q["id"]])
                    if q["type"] == "choice":
                        keys = e.questions[0]["keys"]
                        for o in q["criteria"]:
                            X[j, col] = probs[keys.index(o)]
                            col += 1
                    else:
                        X[j, col] = probs[1]
                        col += 1
            return X

        # warmup 3
        for t in texts_te[:3]:
            _ = features_for([t])
        times = []
        for i in single_idx:
            t0 = time.perf_counter()
            Xf = features_for([texts_te[i]])
            _ = clf.predict(Xf)
            times.append(1000 * (time.perf_counter() - t0))
        times = np.sort(np.array(times))
        t0 = time.perf_counter()
        for s in range(0, len(batch_texts), 64):
            Xf = features_for(batch_texts[s:s + 64])
            _ = clf.predict(Xf)
        dt = time.perf_counter() - t0
        return {"single_p50_ms": round(float(np.percentile(times, 50)), 1),
                "single_p95_ms": round(float(np.percentile(times, 95)), 1),
                "batched_ms_per_text": round(1000 * dt / len(batch_texts), 2),
                "batched_throughput": round(len(batch_texts) / dt, 1)}

    for bank_id in BANKS:
        snap = json.loads((SNAP / f"{bank_id}.json").read_text())
        all_probes = snap["probes"]
        X1k_full = np.load(SNAP / f"{bank_id}.npz")["X"]
        lat = {}
        for cfg in CONFIGS:
            probes_cfg = all_probes if cfg == "unpruned" else frozen[bank_id][cfg]["kept_probes"]
            keep = []
            colmap = {tuple(c): j for j, c in enumerate(snap["column_order"])}
            for p in probes_cfg:
                if p.get("format") == "choice":
                    for o in p["options"]:
                        keep.append(colmap[(p["id"], o)])
                else:
                    keep.append(colmap[(p["id"], None)])
            clf = lr_pipeline()
            clf.fit(X1k_full[:, keep], y1k)
            lat[cfg] = run_bank_timing(probes_cfg)
            lat[cfg]["n_probes"] = len(probes_cfg)
            lat[cfg]["n_columns"] = len(keep)
            print(f"   {bank_id} {cfg}: p50 {lat[cfg]['single_p50_ms']}ms p95 {lat[cfg]['single_p95_ms']}ms "
                  f"| batched {lat[cfg]['batched_ms_per_text']}ms/text @ {lat[cfg]['batched_throughput']}/s",
                  flush=True)
        red_c = 100 * (1 - lat["c_corrected_tol005"]["batched_ms_per_text"]
                       / lat["unpruned"]["batched_ms_per_text"])
        red_d = 100 * (1 - lat["d_corrected_tol010"]["batched_ms_per_text"]
                       / lat["unpruned"]["batched_ms_per_text"])
        lat["latency_reduction_pct"] = {"c_corrected_tol005": round(red_c, 1),
                                        "d_corrected_tol010": round(red_d, 1)}
        latency[bank_id] = lat

    out = {"provenance": {"frozen_points_sha256": frozen_sha,
                          "note": "test split has informed earlier redesigns: this evaluation is exploratory"},
           "test_eval": results, "latency": latency,
           "targets": {"latency_reduction_required_pct": 25.0, "quality_loss_allowed_pp": 1.0}}
    (STUDY / "deployment_eval.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwritten deployment_eval.json")
    mon.complete("deployment eval + benchmark done")
    store.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        if _ACTIVE_MONITOR is not None:
            _ACTIVE_MONITOR.fail(f"{type(exc).__name__}: {exc}")
        raise
