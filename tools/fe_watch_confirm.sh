#!/usr/bin/env bash
# Watch the FE discovery confirm run: wait until the python process appears,
# then wait until it exits; print a summary tail on exit. Run with notify.
# NOTE: pattern is deliberately specific (venv-serve python + script path) to
# avoid matching unrelated shells/monitors (pgrep -f self-match footgun).
set -u
cd /home/xrim/banking77-jev-baseline || exit 1
PAT="venv-serve/bin/python.*confirm\.py full"
found=0
for _ in $(seq 1 40); do
  if pgrep -f "$PAT" >/dev/null; then found=1; break; fi
  sleep 30
done
if [ "$found" -eq 0 ]; then
  echo "confirm run never appeared within 20 min"
  exit 1
fi
while pgrep -f "$PAT" >/dev/null; do sleep 120; done
echo "confirm run finished $(date -u +%H:%M UTC)"
tail -10 runs/fe_confirm.log
