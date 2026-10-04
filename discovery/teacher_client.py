#!/usr/bin/env python3
"""Teacher model client for the discovery pilot.

Talks to the operator-configured strong model (custom_providers entry
"Api.lmuai.com", default model gpt-5.6-sol) through its OpenAI-compatible
endpoint. The API key is read from ~/.hermes/config.yaml at runtime and is
never logged or written anywhere.

Library:
    from teacher_client import teacher_chat
    text = teacher_chat([{"role": "user", "content": "..."}], temperature=0.4, seed=1)

CLI:
    python teacher_client.py "prompt" [--temperature 0.4] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

CONFIG = Path.home() / ".hermes" / "config.yaml"
PROVIDER_NAME = "Api.lmuai.com"
DEFAULT_MODEL = "gpt-5.6-sol"


def _provider() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text())
    for p in cfg.get("custom_providers", []):
        if p.get("name") == PROVIDER_NAME:
            return p
    raise RuntimeError(f"provider {PROVIDER_NAME!r} not found in {CONFIG}")


def teacher_chat(messages: list[dict], model: str = DEFAULT_MODEL, temperature: float = 0.4,
                 seed: int | None = None, max_tokens: int = 1600, timeout: int = 150,
                 retries: int = 2) -> str:
    p = _provider()
    body: dict = {"model": model, "messages": messages, "temperature": temperature,
                  "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        p["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + p["api_key"]},
        method="POST")
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode("utf-8"))
            dt = time.perf_counter() - t0
            print(f"[teacher] {model} ok in {dt:.1f}s", file=sys.stderr, flush=True)
            return out["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"teacher call failed after retries: {last!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    out = teacher_chat([{"role": "user", "content": args.prompt}],
                       temperature=args.temperature, seed=args.seed, model=args.model)
    print(out)


if __name__ == "__main__":
    main()
