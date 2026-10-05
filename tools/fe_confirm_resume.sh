#!/usr/bin/env bash
# Resume the CLINC150 FE-discovery confirmation run after a host reboot.
# Installed as systemd user unit fe-confirm.service (linger enabled:
# starts at boot; MemoryMax=24G carried by the unit).
set -u
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
REPO=/home/xrim/banking77-jev-baseline
cd "$REPO" || exit 1
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

echo "[resume] $(date -u +"%Y-%m-%dT%H:%M:%SZ") fe_confirm_resume.sh starting"

# 1) wait for at least one working GPU (up to 10 min), then a short settle.
# NOTE: `nvidia-smi -L` exits non-zero while the faulty GPU 0 is present even
# when GPU 1 works, so readiness is "any GPU line in the output".
gpu_ready() { nvidia-smi -L 2>/dev/null | grep -q "^GPU "; }
for _ in $(seq 1 60); do
  gpu_ready && break
  sleep 10
done
if ! gpu_ready; then
  echo "[resume] no working GPU detected after 10 min; giving up (start fe-confirm.service manually)"
  exit 1
fi
sleep 20

# 2) guards: already running / already complete
if pgrep -f "venv-serve/bin/python.*confirm\.py full" >/dev/null 2>&1; then
  echo "[resume] confirm.py full already running; nothing to do"
  exit 0
fi
if [ -f runs/clinc150/fe_discovery/confirm_report.json ]; then
  echo "[resume] confirm_report.json already exists; run complete; nothing to do"
  exit 0
fi

# 3) launch (unit provides MemoryMax and log append)
echo "[resume] launching confirm.py full"
exec ./venv-serve/bin/python discovery/fe_discovery/confirm.py full
