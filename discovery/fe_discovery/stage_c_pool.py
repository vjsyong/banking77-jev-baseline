#!/usr/bin/env python3
"""Stage C parallel pool: N arm-run subprocesses + serial seed setups.

Ops deviation (documented in docs/fe-ops-notes.md): the registered campaign runs
each arm to completion in sequence; this coordinator runs up to FE_POOL_WORKERS
(default 2) independent arm-runs concurrently. Per-run protocol is unchanged
(each arm is an independent unit with its own teacher conversation, budget
accounting, and artifacts). Seed setups remain strictly serial.

Resume-safe: skips runs with final.json; deletes partial run dirs before relaunch.
"""
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path("/home/xrim/banking77-jev-baseline")
STAGE = HERE / "runs" / "clinc150" / "fe_discovery" / "stage_c"
LOGS = HERE / "runs"
PY = str(HERE / "venv-serve" / "bin" / "python")
AGENT = str(HERE / "discovery" / "fe_discovery" / "stage_c_arm.py")
SETUP = str(HERE / "discovery" / "fe_discovery" / "stage_c.py")

SEEDS = [11, 23, 37, 53, 71]
ARMS = ["U", "R", "E", "F"]
N = int(os.environ.get("FE_POOL_WORKERS", "2"))


def env_for_children():
    env = dict(os.environ)
    env["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    env["FE_LR_JOBS"] = env.get("FE_LR_JOBS", "6")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "4")
    return env


def ensure_setup(seed):
    meta = STAGE / f"seed_{seed}" / "seed_setup.json"
    if meta.exists():
        return True
    print(f"[pool] running setup for seed {seed} (serial, may take ~10 min)...", flush=True)
    t0 = time.time()
    log = open(LOGS / f"fe_pool_setup_s{seed}.log", "w")
    rc = subprocess.run([PY, SETUP, "--setup-only", str(seed)],
                        stdout=log, stderr=subprocess.STDOUT,
                        env=env_for_children(), cwd=str(HERE)).returncode
    log.close()
    ok = rc == 0 and meta.exists()
    print(f"[pool] setup seed {seed}: {'ok' if ok else 'FAILED rc=' + str(rc)} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return ok


def launch(seed, arm):
    outdir = STAGE / f"seed_{seed}" / arm
    if (outdir / "final.json").exists():
        return None
    if outdir.exists():
        subprocess.run(["rm", "-rf", str(outdir)])
    log = open(LOGS / f"fe_pool_s{seed}_{arm}.log", "w")
    p = subprocess.Popen([PY, AGENT, "--seed", str(seed), "--arm", arm],
                         stdout=log, stderr=subprocess.STDOUT,
                         env=env_for_children(), cwd=str(HERE))
    return p


def main():
    queue = deque((s, a) for s in SEEDS for a in ARMS
                  if not (STAGE / f"seed_{s}" / a / "final.json").exists())
    print(f"[pool] {len(queue)} runs pending, {N} workers", flush=True)
    active = {}
    failed = []
    while queue or active:
        while queue and len(active) < N:
            seed, arm = queue.popleft()
            if not ensure_setup(seed):
                failed.append({"seed": seed, "arm": arm, "why": "setup-failed"})
                continue
            p = launch(seed, arm)
            if p is None:
                continue
            active[p] = (seed, arm)
            print(f"[pool] launch {arm} s{seed} (pid {p.pid})", flush=True)
        time.sleep(15)
        for p in list(active):
            if p.poll() is not None:
                seed, arm = active.pop(p)
                ok = p.returncode == 0 and (STAGE / f"seed_{seed}" / arm / "final.json").exists()
                print(f"[pool] {'done' if ok else 'FAILED'} {arm} s{seed} rc={p.returncode}", flush=True)
                if not ok:
                    failed.append({"seed": seed, "arm": arm, "rc": p.returncode})
        (STAGE / "pool_status.json").write_text(json.dumps(
            {"active": [{"seed": s, "arm": a, "pid": p.pid} for p, (s, a) in active.items()],
             "queued": [{"seed": s, "arm": a} for s, a in queue],
             "failed": failed, "updated": time.strftime("%H:%M:%S")}, indent=1))
    print(f"[pool] ALL DONE. failures: {failed if failed else 'none'}", flush=True)


if __name__ == "__main__":
    main()
