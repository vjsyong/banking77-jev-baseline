#!/usr/bin/env python3
"""Triple-diff: golden cache vs fresh single-record vs fresh batched (batched path validation).

Separates (a) run-to-run noise of the CURRENT path from (b) genuine drift introduced
by the batched (width-grouped, cross-record) forward.

Pairs:
  S-G  fresh single  vs golden cache   -> current-path run-to-run noise
  B-G  fresh batched vs golden cache
  B-S  fresh batched vs fresh single   -> batched-vs-current drift (same process, same session)

Fields compared per question: choice string (flips), p_true (noul), option
probabilities (only when both sides carry them; golden noul answers don't).

Usage: ./venv-serve/bin/python tools/slice_diffdetail.py --n 256 [--chunk 64]
"""
import importlib.util
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

spec = importlib.util.spec_from_file_location("bst", HERE / "tools" / "batched_slice_test.py")
bst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bst)

from datasets import load_dataset  # noqa: E402
from banking77_baseline import ResultCache, cache_key, response_answers  # noqa: E402
from tinyjev import load as load_agent  # noqa: E402

RUNS = HERE / "runs" / "banking77"
MODEL = "TinyJev-0.6B"


def p_of(ans):
    """noul value from any answer shape (p_true preferred, else noul)."""
    if "p_true" in ans:
        return float(ans["p_true"])
    return float(ans["noul"])


def compare(name, texts, ref, other):
    """ref/other: dict qid -> answer. Returns metrics dict."""
    choice_total = flips = 0
    noul_max = 0.0
    noul_deltas = []
    prob_max = 0.0
    prob_max_where = None
    flip_details = []
    for i, qid_answers in ref.items():
        for qid, ga in qid_answers.items():
            ma = other.get(i, {}).get(qid)
            if ma is None:
                continue
            if "choice" in ga and "choice" in ma:
                choice_total += 1
                if ga["choice"] != ma["choice"]:
                    flips += 1
                    flip_details.append({"row": i, "qid": qid,
                                         "a": ga["choice"], "b": ma["choice"]})
            if "noul" in ga or "p_true" in ga:
                d = abs(p_of(ga) - p_of(ma))
                noul_deltas.append(d)
                noul_max = max(noul_max, d)
            pa, pb = ga.get("probabilities") or {}, ma.get("probabilities") or {}
            if pa and pb:
                for kk in set(pa) & set(pb):
                    d = abs(float(pa[kk]) - float(pb[kk]))
                    if d > prob_max:
                        prob_max = d
                        prob_max_where = {"row": i, "qid": qid, "key": kk,
                                          "a": float(pa[kk]), "b": float(pb[kk])}
    deltas = sorted(noul_deltas, reverse=True)
    p95 = deltas[int(0.05 * len(deltas))] if deltas else 0.0
    return {
        "pair": name, "choice_total": choice_total, "choice_flips": flips,
        "flips": flip_details[:10],
        "noul_n": len(deltas), "noul_max": noul_max, "noul_p95": p95,
        "noul_mean": float(np.mean(deltas)) if deltas else 0.0,
        "prob_max": prob_max, "prob_max_where": prob_max_where,
        "top_noul": deltas[:5],
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    out_path = RUNS / "slice_diffdetail.json"

    d = load_dataset("PolyAI/banking77")
    labels = d["train"].features["label"].names
    texts = [d["train"][i]["text"] for i in range(n)]
    recs, qmap = bst.make_records(texts, labels)

    # golden
    cache = ResultCache(RUNS / "jev_cache.sqlite")
    golden = {}
    for i, t in enumerate(texts):
        hit = cache.get(cache_key(t, MODEL, qmap))
        if hit is not None:
            golden[i] = response_answers(hit[0])
    cache.close()
    print(f"golden: {len(golden)}/{n}")

    print("loading model ...", flush=True)
    agent = load_agent(MODEL, backend="torch", device="cuda")
    fam, model, device = agent.family, agent.backbone.model, agent.backbone.device

    # fresh single-record (current path, in-process)
    print("fresh single-record pass ...", flush=True)
    t0 = time.perf_counter()
    single = {}
    for i, t in enumerate(texts):
        out = agent.predict({"state": t, "questions": qmap})
        single[i] = out["states"][0]["answers"]
    single_s = time.perf_counter() - t0
    print(f"single: {single_s / n * 1000:.1f} ms/text")

    # fresh batched
    encs = [fam.encode(r) for r in recs]
    batched = {}
    t0 = time.perf_counter()
    for s in range(0, n, chunk):
        idxs = list(range(s, min(s + chunk, n)))
        hidden = bst.forward_groups(model, device, fam, encs, idxs)
        for ri in idxs:
            enc = encs[ri]
            hidden_rows = [None] * len(enc.rows)
            for q in enc.questions:
                hidden_rows[q["row"]] = hidden[(ri, q["id"])]
            z = fam.logits(hidden_rows, enc)
            batched[ri] = {q["id"]: fam.answer(q, bst.softmax(z[q["id"]])) for q in enc.questions}
    batched_s = time.perf_counter() - t0
    print(f"batched: {batched_s / n * 1000:.1f} ms/text")

    results = [
        compare("single_vs_golden", texts, golden, single),
        compare("batched_vs_golden", texts, golden, batched),
        compare("batched_vs_single", texts, single, batched),
    ]

    print("\n=== triple diff (n=%d) ===" % n)
    for r in results:
        print(f"\n[{r['pair']}]")
        print(f"  choice flips: {r['choice_flips']}/{r['choice_total']}")
        print(f"  noul |delta|: max {r['noul_max']:.2e} | p95 {r['noul_p95']:.2e} | mean {r['noul_mean']:.2e}")
        print(f"  prob |delta|: max {r['prob_max']:.2e}" + (f" at {r['prob_max_where']}" if r['prob_max_where'] else ""))
        if r["flips"]:
            print(f"  flips: {r['flips'][:4]}")

    print(f"\nsingle (in-process) : {single_s / n * 1000:.1f} ms/text")
    print(f"batched (in-process): {batched_s / n * 1000:.1f} ms/text")
    print(f"speedup             : {single_s / batched_s:.2f}x")

    out = {"n": n, "chunk": chunk, "pairs": results,
           "single_ms_per_text": single_s / n * 1000, "batched_ms_per_text": batched_s / n * 1000,
           "speedup": single_s / batched_s}
    out_path.write_text(json.dumps(out, indent=2))
    print("written", out_path)


if __name__ == "__main__":
    main()
