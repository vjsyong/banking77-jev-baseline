#!/usr/bin/env python3
"""Teacher model client for the discovery experiments.

Resolves two provider sources at runtime (API keys are read from config on
disk and are never logged or printed):

- "lmuai-pro" (default) -> ~/.config/opencode/opencode.json, provider.lmuai-pro,
  base https://api.lmuai.ai/v1 (serves gpt-6.1-sol)
- "Api.lmuai.com" -> ~/.hermes/config.yaml custom_providers entry (legacy)

Library:
    from teacher_client import teacher_chat
    text = teacher_chat([{"role": "user", "content": "..."}], temperature=0.4, seed=1)

CLI:
    python teacher_client.py "prompt" [--provider lmuai-pro] [--model gpt-6.1-sol]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import yaml

HERMES_CONFIG = Path.home() / ".hermes" / "config.yaml"
OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"
HERMES_PROVIDER = "Api.lmuai.com"
DEFAULT_PROVIDER = "lmuai-pro"
DEFAULT_MODEL = "gpt-6.1-sol"


def _env_secret(name: str) -> str | None:
    """Resolve an env-style secret: process env first, then ~/.bashrc export."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        for line in (Path.home() / ".bashrc").read_text().splitlines():
            m = re.match(rf"\s*(?:export\s+)?{re.escape(name)}=(.*)$", line)
            if m:
                v = m.group(1).strip().strip("'\"")
                if v:
                    return v
    except Exception:  # noqa: BLE001
        pass
    return None


def _resolve_provider(name: str) -> dict:
    """Return {"base_url", "api_key"} for a provider name. Keys stay in-process."""
    if name == HERMES_PROVIDER:
        cfg = yaml.safe_load(HERMES_CONFIG.read_text())
        for p in cfg.get("custom_providers", []):
            if p.get("name") == name:
                return {"base_url": p["base_url"], "api_key": p["api_key"]}
        raise RuntimeError(f"provider {name!r} not found in {HERMES_CONFIG}")
    oc = json.loads(OPENCODE_CONFIG.read_text())
    prov = (oc.get("provider") or {}).get(name)
    if not prov:
        raise RuntimeError(f"provider {name!r} not found in {OPENCODE_CONFIG}")
    opts = prov.get("options") or {}
    if not opts.get("baseURL"):
        raise RuntimeError(f"provider {name!r} missing baseURL")
    key = str(opts.get("apiKey") or "")
    m = re.fullmatch(r"\{env:([A-Za-z0-9_]+)\}", key.strip())
    if m:
        key = _env_secret(m.group(1)) or ""
    if not key:
        raise RuntimeError(f"provider {name!r}: apiKey could not be resolved")
    return {"base_url": opts["baseURL"], "api_key": key}


def teacher_chat(messages: list[dict], model: str | None = None, provider: str | None = None,
                 temperature: float = 0.4, seed: int | None = None, max_tokens: int = 1600,
                 timeout: int = 240, retries: int = 2) -> str:
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL
    p = _resolve_provider(provider)
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
            print(f"[teacher] {provider}/{model} ok in {dt:.1f}s", file=sys.stderr, flush=True)
            return out["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"teacher call failed after retries: {last!r}")


def teacher_chat_full(messages: list[dict], model: str | None = None, provider: str | None = None,
                      temperature: float = 0.4, seed: int | None = None, max_tokens: int = 1600,
                      timeout: int = 240, retries: int = 2) -> dict:
    """Like teacher_chat but returns {"text", "usage", "elapsed_s"}.

    Same endpoint/behavior; additive for experiments that record token usage.
    """
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODEL
    p = _resolve_provider(provider)
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
            print(f"[teacher] {provider}/{model} ok in {dt:.1f}s "
                  f"(usage {out.get('usage')})", file=sys.stderr, flush=True)
            return {"text": out["choices"][0]["message"]["content"],
                    "usage": out.get("usage") or {}, "elapsed_s": dt}
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
    ap.add_argument("--provider", default=DEFAULT_PROVIDER)
    args = ap.parse_args()
    out = teacher_chat([{"role": "user", "content": args.prompt}],
                       temperature=args.temperature, seed=args.seed,
                       model=args.model, provider=args.provider)
    print(out)


if __name__ == "__main__":
    main()
