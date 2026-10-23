#!/bin/bash
# nsys.sh CARD TAG: card.sh under Nsight Systems, a $DURATION s capture (40) starting $DELAY s in
# (240, past the load), child processes (vLLM's engine core) included; then the SQLite export and
# KohakuvLLM's bench/profile/nsys_summary.py ($NSYS_SUMMARY) over it.
set -u
CARD=$1; TAG=$2
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT=${OUT_ROOT:-$HERE/../../../runs/bench/machine}
NSYS=${NSYS:-nsys}
mkdir -p "$OUT_ROOT"
$NSYS profile -o "$OUT_ROOT/$TAG" -f true --trace=cuda,osrt --cuda-graph-trace=node \
  --trace-fork-before-exec=true --delay "${DELAY:-240}" --duration "${DURATION:-40}" \
  bash "$HERE/card.sh" "$CARD" "$TAG"
$NSYS export --type sqlite --force-overwrite true -o "$OUT_ROOT/$TAG.sqlite" "$OUT_ROOT/$TAG.nsys-rep"
${PY:-python} "$NSYS_SUMMARY" "$OUT_ROOT/$TAG.sqlite" --trim 0.02 --out "$OUT_ROOT/$TAG.summary.json"
echo "NSYS_DONE $TAG"
