#!/usr/bin/env python3
"""Measure whether batching multiple texts into one tinyjev call beats per-text calls.

Times: 8 sequential single-text predict() calls vs one predict() call carrying
{4,8,16} states. Reports total + per-text wall and the model's internal model_ms.
Run on a quiet GPU (pause concurrent extraction first).
"""
import os
import sys
import time

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "5")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
torch.set_num_threads(5)
import tinyjev  # noqa: E402
from datasets import load_dataset  # noqa: E402
from banking77_baseline import question_set  # noqa: E402


def main():
    d = load_dataset("PolyAI/banking77")
    labels = d["train"].features["label"].names
    q = question_set(labels, "both")
    texts = [d["train"][i]["text"] for i in range(20)]

    agent = tinyjev.load("TinyJev-0.6B", backend="torch", device="cuda")

    for t in texts[:2]:  # warmup
        agent.predict({"state": t, "questions": q})

    t0 = time.perf_counter()
    for t in texts[:8]:
        agent.predict({"state": t, "questions": q})
    sep = time.perf_counter() - t0
    print(f"separate 8 calls : total {sep:6.3f}s | per-text {sep / 8:.3f}s", flush=True)

    for B in (4, 8, 16):
        recs = [{"id": f"r{i}", "state": t, "questions": q} for i, t in enumerate(texts[:B])]
        t0 = time.perf_counter()
        out = agent.predict({"states": recs})
        dt = time.perf_counter() - t0
        assert len(out["states"]) == B, "missing states in batch response"
        print(f"one call, {B:2d} states: total {dt:6.3f}s | per-text {dt / B:.3f}s | "
              f"model_ms {out['execution']['model_ms']:.0f}", flush=True)

    t0 = time.perf_counter()
    for t in texts[:8]:
        agent.predict({"state": t, "questions": q})
    sep2 = time.perf_counter() - t0
    print(f"separate 8 calls (again): total {sep2:6.3f}s | per-text {sep2 / 8:.3f}s", flush=True)


if __name__ == "__main__":
    main()
