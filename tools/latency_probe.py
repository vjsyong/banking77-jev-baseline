#!/usr/bin/env python3
"""Probe sustained System One latency with the full extraction payload (17 questions).

Issues N single-text requests (same schema as the bundle's extract-jev) against
the local endpoint and reports wall + server model_ms distributions.
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
sys.path.insert(0, "/home/xrim/banking77-jev-baseline")

from datasets import load_dataset  # noqa: E402
from banking77_baseline import question_set, json_request  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8077/v1/systemone"
    d = load_dataset("PolyAI/banking77")
    labels = d["train"].features["label"].names
    q = question_set(labels, "both")
    texts = [d["train"][i]["text"] for i in range(n)]
    wall, model = [], []
    for i, t in enumerate(texts, 1):
        t0 = time.perf_counter()
        raw, elapsed = json_request(endpoint, "TinyJev-0.6B", t, q, None, 180, 2)
        w = time.perf_counter() - t0
        wall.append(w)
        model.append(raw.get("latency_ms") or 0.0)
        print(f"  {i:3d}  wall {w:.3f}s  model {model[-1]:.0f}ms", flush=True)
    def stats(xs, fmt="%.3f"):
        return "mean " + fmt + " | p50 " + fmt + " | min " + fmt + " | max " + fmt % (
            statistics.mean(xs), statistics.median(xs), min(xs), max(xs))
    print("\nrequests:", len(wall))
    print("wall s:", stats(wall))
    print("model ms:", stats(model, "%.0f"))


if __name__ == "__main__":
    main()
