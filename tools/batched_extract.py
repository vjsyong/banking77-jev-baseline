#!/usr/bin/env python3
"""Batched extraction for the BANKING77 Jev baseline (production path).

In-process, width-grouped, cross-record batched forward: rows from many texts are
collected per chunk and split by width class (the single 77-option choice row vs
the 16 compact noul rows), then run as two padded forwards per chunk on the GPU.

Writes the bundle's `jev_cache.sqlite` with byte-compatible cache keys and
System-One-shaped payloads (identical conversion to `Agent.systemone()`, which the
HTTP endpoint used), so `extract-jev` reconciliation and `evaluate-jev` run
unchanged.

Validated vs the single-record path: 256-text slice (0/256 choice flips, noul
|delta| <= 3.1e-3, fp16 kernel-shape noise) and a full-run cross-check over all
commonly-cached rows (`tools/compare_batched_run.py`).

Usage:
  ./venv-serve/bin/python tools/batched_extract.py --out runs/banking77 --overwrite
  # resume without re-doing done rows: drop --overwrite
"""
from __future__ import annotations

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
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "5")))

from banking77_baseline import (  # noqa: E402
    ResultCache, cache_key, load_dataset_rows, question_set, response_answers,
)
from tinyjev import load as load_agent  # noqa: E402
from tinyjev.families import softmax  # noqa: E402


def to_systemone(q: dict, a: dict) -> dict:
    """Exact replica of Agent.systemone()'s per-answer conversion."""
    if a["type"] in ("boolean", "noul"):
        return {"type": "noul", "noul": round(float(a["p_true"]), 4)}
    if a["type"] == "choice":
        return {"type": "choice", "choice": a["choice"], "confidence": a.get("confidence"),
                "probabilities": a["probabilities"]}
    return {"type": "score", "score": round(float(a["score"]), 4), "legend": a.get("legend"),
            "probabilities": a["probabilities"], "confidence": a.get("confidence")}


def forward_groups(model, device, fam, encs, idxs):
    """Two forwards per chunk: choice-row group, then noul-row group."""
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="TinyJev-0.6B")
    ap.add_argument("--out", default="runs/banking77")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-extract even cached rows (single-method pass)")
    ap.add_argument("--limit", type=int, default=None, help="per-split row cap (debug)")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out

    rows_by_split, labels, official = load_dataset_rows(args.limit)
    qmap = question_set(labels, "both")
    questions = [dict(id=k, **v) for k, v in qmap.items()]
    for q in questions:
        if q["type"] == "noul":
            q["type"] = "boolean"

    seen: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        for r in rows:
            k = cache_key(r["text"], args.model, qmap)
            if k not in seen:
                seen[k] = r["text"]

    cache = ResultCache(out / "jev_cache.sqlite")
    todo = [(k, t) for k, t in seen.items() if args.overwrite or cache.get(k) is None]
    print(f"unique texts: {len(seen):,} | to extract: {len(todo):,} (overwrite={args.overwrite})", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    print("loading model ...", flush=True)
    t0 = time.time()
    agent = load_agent(args.model, backend="torch", device=args.device)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    fam, model, device = agent.family, agent.backbone.model, agent.backbone.device

    t0 = time.perf_counter()
    encs, keys, skipped = [], [], 0
    for k, t in todo:
        try:
            encs.append(fam.encode({"id": k[:12], "state": t, "questions": questions}))
            keys.append(k)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"encode skip ({type(exc).__name__}: {exc})", flush=True)
    print(f"encoded {len(encs):,} in {time.perf_counter() - t0:.1f}s ({skipped} skipped)", flush=True)

    # warmup on the last two encoded records (discarded; they run again in the loop)
    wencs = encs[-2:] if len(encs) >= 2 else encs
    forward_groups(model, device, fam, wencs, list(range(len(wencs))))

    done = 0
    chunk_ms = []
    t_all = time.perf_counter()
    n = len(encs)
    for s in range(0, n, args.chunk):
        idxs = list(range(s, min(s + args.chunk, n)))
        t0 = time.perf_counter()
        hidden = forward_groups(model, device, fam, encs, idxs)
        fwd_share = (time.perf_counter() - t0) / len(idxs)  # forward share per record
        for ri in idxs:
            enc = encs[ri]
            t1 = time.perf_counter()
            hidden_rows = [None] * len(enc.rows)
            for q in enc.questions:
                hidden_rows[q["row"]] = hidden[(ri, q["id"])]
            z = fam.logits(hidden_rows, enc)
            answers = {q["id"]: to_systemone(q, fam.answer(q, softmax(z[q["id"]]))) for q in enc.questions}
            record_s = fwd_share + (time.perf_counter() - t1)  # forward share + this record's readout
            payload = {"model": args.model, "answers": answers,
                       "latency_ms": round(record_s * 1000, 2)}
            got = response_answers(payload)
            assert all(q["id"] in got for q in enc.questions)
            cache.put(keys[ri], payload, record_s)
            done += 1
        chunk_ms.append(1000 * (time.perf_counter() - t0) / len(idxs))
        if (s // args.chunk) % 25 == 24 or s + args.chunk >= n:
            rate = done / (time.perf_counter() - t_all)
            print(f"chunk {min(s + args.chunk, n):,}/{n:,} done | {done / (time.perf_counter() - t_all):.1f} texts/s | "
                  f"chunk {chunk_ms[-1]:.0f} ms/text", flush=True)
    cache.close()
    wall = time.perf_counter() - t_all
    mean_ms = 1000 * wall / done
    print(f"DONE {done:,} texts in {wall:.0f}s = {mean_ms:.1f} ms/text "
          f"({1000 / mean_ms:.1f} texts/s); errors={skipped}", flush=True)

    manifest = {
        "method": "batched width-grouped in-process forward (choice group + noul group per chunk)",
        "model": args.model, "device": args.device, "chunk": args.chunk,
        "unique_texts": len(seen), "extracted": done, "encode_skips": skipped,
        "elapsed_s": wall, "ms_per_text": mean_ms,
        "latency_attribution": "payload latency_ms = (chunk forward + record readout) attributed equally per record",
        "rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "official_rows": official,
    }
    (out / "batched_extract_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
