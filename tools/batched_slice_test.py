#!/usr/bin/env python3
"""Equivalence + speed slice test for the batched (width-grouped) Jev path.

Prototype: instead of one padded call per record (17 rows x 836 width), collect
rows from many records, split them by width class (the single 77-option choice row
vs the 16 compact noul rows) and run one padded forward per class per chunk of
texts. Reads-only against the golden cache produced by the current pipeline.

Usage:
  ./venv-serve/bin/python tools/batched_slice_test.py --n 256 [--chunk 64]
"""
import argparse
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

from datasets import load_dataset  # noqa: E402
from banking77_baseline import ResultCache, cache_key, question_set, response_answers  # noqa: E402
from tinyjev import load as load_agent  # noqa: E402
from tinyjev.families import softmax  # noqa: E402

RUNS = HERE / "runs" / "banking77"
MODEL = "TinyJev-0.6B"


def make_records(texts, labels):
    qmap = question_set(labels, "both")
    questions = [dict(id=k, **v) for k, v in qmap.items()]
    for q in questions:
        if q["type"] == "noul":
            q["type"] = "boolean"
    return [{"id": f"t{i}", "state": t, "questions": questions} for i, t in enumerate(texts)], qmap


def forward_groups(model, device, fam, encs, idxs):
    """One forward per width class (choice group, noul group) over the given records.

    Returns {(record idx, qid): hidden ndarray (len = prefix+row)}.
    """
    choice, noul = [], []
    for ri in idxs:
        enc = encs[ri]
        for q in enc.questions:
            row = enc.prefix + enc.rows[q["row"]]
            (choice if q["type"] == "choice" else noul).append((ri, q["id"], row))
    hidden = {}
    for group in (choice, noul):
        if not group:
            continue
        width = max(len(r[2]) for r in group)
        ids = torch.full((len(group), width), fam.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(group), width), dtype=torch.long)
        for i, (_, _, toks) in enumerate(group):
            ids[i, :len(toks)] = torch.tensor(toks)
            att[i, :len(toks)] = 1
        with torch.inference_mode():
            h = model(input_ids=ids.to(device), attention_mask=att.to(device),
                      use_cache=False).last_hidden_state.float().cpu().numpy()
        for i, (ri, qid, toks) in enumerate(group):
            hidden[(ri, qid)] = h[i, :len(toks)]
    return hidden


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--out", default=str(RUNS / "batched_slice_report.json"))
    args = ap.parse_args()

    d = load_dataset("PolyAI/banking77")
    labels = d["train"].features["label"].names
    texts = [d["train"][i]["text"] for i in range(args.n)]
    recs, qmap = make_records(texts, labels)

    # golden entries + reference latency
    cache = ResultCache(RUNS / "jev_cache.sqlite")
    golden, ref_elapsed, missing = {}, [], []
    for i, t in enumerate(texts):
        hit = cache.get(cache_key(t, MODEL, qmap))
        if hit is None:
            missing.append(i)
        else:
            golden[i] = response_answers(hit[0])
            ref_elapsed.append(hit[1])
    cache.close()
    print(f"cache coverage: {len(golden)}/{args.n} golden; missing: {len(missing)}")
    if len(missing) > args.n // 10:
        raise SystemExit("coverage too low; is the extraction past these rows? aborting")

    print("loading model ...", flush=True)
    t0 = time.time()
    agent = load_agent(MODEL, backend="torch", device="cuda")
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    fam, model, device = agent.family, agent.backbone.model, agent.backbone.device

    # encode
    t0 = time.perf_counter()
    encs = [fam.encode(r) for r in recs]
    encode_s = time.perf_counter() - t0

    # warmup on two unseen texts (not part of the slice)
    warm_texts = [d["train"][i]["text"] for i in (6060, 6061)]
    we, _ = make_records(warm_texts, labels)
    wencs = [fam.encode(r) for r in we]
    forward_groups(model, device, fam, wencs, [0, 1])

    # timed batched loop
    results = [None] * len(recs)
    chunk_times = []
    grid_batched = 0
    t_all = time.perf_counter()
    t_head_total = 0.0
    for s in range(0, len(recs), args.chunk):
        idxs = list(range(s, min(s + args.chunk, len(recs))))
        t0 = time.perf_counter()
        hidden = forward_groups(model, device, fam, encs, idxs)
        chunk_times.append(time.perf_counter() - t0)
        # grid accounting for this chunk
        ch_w = max(max(len(e.prefix) + len(e.rows[q["row"]]) for q in e.questions if q["type"] == "choice") for e in [encs[i] for i in idxs])
        nu_w = max(len(e.prefix) + len(e.rows[q["row"]]) for e in [encs[i] for i in idxs] for q in e.questions if q["type"] != "choice")
        grid_batched += len(idxs) * ch_w + len(idxs) * 16 * nu_w
        # per-record readout
        t0 = time.perf_counter()
        for ri in idxs:
            enc = encs[ri]
            hidden_rows = [None] * len(enc.rows)
            for q in enc.questions:
                hidden_rows[q["row"]] = hidden[(ri, q["id"])]
            z = fam.logits(hidden_rows, enc)
            ans = {}
            for q in enc.questions:
                ans[q["id"]] = fam.answer(q, softmax(z[q["id"]]))
            results[ri] = ans
        t_head_total += time.perf_counter() - t0
    total_s = time.perf_counter() - t_all

    # grid of the current path over the slice
    grid_cur = sum(len(e.rows) * (len(e.prefix) + max(len(r) for r in e.rows)) for e in encs)

    # comparison
    choice_total = choice_flips = 0
    flips, max_noul, max_prob = [], 0.0, 0.0
    noul_total = 0
    for ri in sorted(golden):
        mine = results[ri]
        for qid, ga in golden[ri].items():
            ma = mine[qid]
            if "choice" in ga:
                choice_total += 1
                if ga["choice"] != ma.get("choice"):
                    choice_flips += 1
                    flips.append({"row": ri, "qid": qid, "golden": ga["choice"], "batched": ma.get("choice")})
            if "noul" in ga:
                noul_total += 1
                max_noul = max(max_noul, abs(float(ga["noul"]) - float(ma["p_true"])))
            pa, pb = ga.get("probabilities") or {}, ma.get("probabilities") or {}
            if pa and pb:
                for kk in set(pa) & set(pb):
                    max_prob = max(max_prob, abs(float(pa.get(kk, 0)) - float(pb.get(kk, 0))))

    ref_mean = float(np.mean(ref_elapsed))
    ref_med = float(np.median(ref_elapsed))
    ref_steady = [e for e in ref_elapsed if e < 1.0]
    ref_med_steady = float(np.median(ref_steady)) if len(ref_steady) >= 50 else None
    fwd_per_text = sum(chunk_times) / len(encs)
    tot_per_text = total_s / len(encs)
    projected_min = tot_per_text * 13083 / 60

    verdict = "PASS" if (choice_flips == 0 and max_noul < 2e-3 and max_prob < 2e-3) else "REVIEW"

    print("\n=== section timings (n=%d texts, chunk=%d) ===" % (len(encs), args.chunk))
    print(f"encode total:      {encode_s:7.2f}s  ({1000 * encode_s / len(encs):5.2f} ms/text)")
    print(f"forward chunks:    {sum(chunk_times):7.2f}s  ({1000 * fwd_per_text:5.2f} ms/text)  per-chunk: "
          + ", ".join(f"{t:.2f}" for t in chunk_times))
    print(f"head/readout:      {t_head_total:7.2f}s  ({1000 * t_head_total / len(encs):5.2f} ms/text)")
    print(f"TOTAL (no load):   {total_s:7.2f}s  ({1000 * tot_per_text:5.2f} ms/text)")

    print("\n=== equivalence vs golden cache ===")
    print(f"choice comparisons: {choice_total}, flips: {choice_flips}")
    print(f"noul  comparisons: {noul_total}, max |delta|: {max_noul:.2e}")
    print(f"max probability |delta|: {max_prob:.2e}")
    if flips:
        print("flips:", json.dumps(flips[:10]))

    print("\n=== grids (positions) ===")
    print(f"current path: {grid_cur:,}  |  batched: {grid_batched:,}  |  ratio {grid_cur / grid_batched:.2f}x")
    print(f"\nreference latency (cache): mean {ref_mean:.3f}s, median {ref_med:.3f}s"
          + (f", steady (<1s subset) median {ref_med_steady:.3f}s" if ref_med_steady else ""))
    base = ref_med_steady or ref_med
    print(f"batched speedup vs steady reference: {base / tot_per_text:.2f}x total "
          f"({base / fwd_per_text:.2f}x forward-only)")
    print(f"projected full 13,083-text extraction at this rate: {projected_min:.0f} min")
    print(f"\nVERDICT: {verdict}")

    report = {
        "n": len(encs), "chunk": args.chunk,
        "choice_comparisons": choice_total, "choice_flips": choice_flips, "flips": flips,
        "noul_comparisons": noul_total, "max_noul_delta": max_noul, "max_prob_delta": max_prob,
        "encode_s": encode_s, "forward_s": sum(chunk_times), "head_s": t_head_total, "total_s": total_s,
        "ms_per_text_total": 1000 * tot_per_text, "ms_per_text_forward": 1000 * fwd_per_text,
        "grid_current": grid_cur, "grid_batched": grid_batched,
        "ref_latency_mean_s": ref_mean, "ref_latency_median_s": ref_med,
        "ref_latency_median_steady_s": ref_med_steady,
        "speedup_vs_steady": base / tot_per_text,
        "projected_full_run_min": projected_min, "verdict": verdict,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("report written to", args.out)


if __name__ == "__main__":
    main()
