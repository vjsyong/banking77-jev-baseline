#!/usr/bin/env python3
"""Sharded in-process extraction for the BANKING77 Jev baseline.

Processes shard `i` of `N` of the bundle's extraction workload using the
tinyjev agent directly on torch/CUDA (no HTTP hop), but writes into the
bundle's own `jev_cache.sqlite` with identical cache keys and payload
semantics: the value stored per key is the output of the exact same
`agent.systemone(payload)` call that the /v1/systemone server executes per
request, and the questions/labels come from the bundle's own functions.

After all shards finish, run the bundle's `extract-jev` once (a no-op fetch,
writes manifest + CSVs) and `evaluate-jev`, exactly as in the README.

This is a throughput accelerator, not a replacement for the HTTP path; the
HTTP path was smoke-validated separately (see runs/).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "5")
os.environ.setdefault("MKL_NUM_THREADS", "5")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from banking77_baseline import (  # noqa: E402
    ResultCache, cache_key, load_dataset_rows, question_set, response_answers,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=4)
    ap.add_argument("--model", default="TinyJev-0.6B")
    ap.add_argument("--out", default="runs/banking77")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="per-split row cap (debug only)")
    args = ap.parse_args()

    if not (0 <= args.shard < args.of):
        raise SystemExit("--shard must be in [0, --of)")

    import torch  # noqa: PLC0415
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "5")))
    import tinyjev  # noqa: PLC0415

    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out

    rows_by_split, labels, _ = load_dataset_rows(args.limit)
    questions = question_set(labels, "both")
    cache = ResultCache(out / "jev_cache.sqlite")
    print(f"[shard {args.shard}/{args.of}] loading {args.model} on {args.device} ...", flush=True)
    t0 = time.time()
    agent = tinyjev.load(args.model, backend="torch", device=args.device)
    print(f"[shard {args.shard}/{args.of}] model loaded in {time.time() - t0:.1f}s", flush=True)

    done = skipped = errors = 0
    latencies = []
    for split, rows in rows_by_split.items():
        for i, row in enumerate(rows):
            if i % args.of != args.shard:
                continue
            key = cache_key(row["text"], args.model, questions)
            if cache.get(key) is not None:
                skipped += 1
                continue
            payload = {"model": args.model, "state": row["text"], "questions": questions}
            t0 = time.perf_counter()
            try:
                resp = agent.systemone(payload)
                elapsed = time.perf_counter() - t0
                answers = response_answers(resp)
                missing = [k for k in questions if k not in answers]
                if missing:
                    raise ValueError("missing answers: " + ", ".join(missing[:4]))
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"[shard {args.shard}] ERROR {split}:{i}: {type(exc).__name__}: {exc}", flush=True)
                continue
            cache.put(key, resp, elapsed)
            latencies.append(elapsed)
            done += 1
            if done % 100 == 0:
                print(f"[shard {args.shard}] {done} done, {skipped} pre-cached skipped, "
                      f"{errors} errors, last {elapsed * 1000:.0f}ms", flush=True)
    cache.close()
    mean_ms = 1000 * sum(latencies) / len(latencies) if latencies else 0.0
    print(f"[shard {args.shard}] DONE done={done} skipped={skipped} errors={errors} "
          f"mean_ms={mean_ms:.0f}", flush=True)


if __name__ == "__main__":
    main()
