#!/bin/bash
# card.sh CARD TAG: one GBC machine on one card (configs/machine/qwen35_card.py with the mid + sam3
# controls) over $IMAGES; writes $OUT_ROOT/TAG/ (scripts/bench/machine/run.py) and $OUT_ROOT/TAG.log.
# The machine's knobs pass through the environment (GBC_VLM_*, GBC_DETECT_*, KOHAKU_*).
# SERVE_PORT set: the VLM runs behind `vllm serve` on that port of the same card (serve.py, log in
# $OUT_ROOT/TAG.server.log), the machine calls it over HTTP; unset: in-process.
# MEMORY_S set: memory.py samples the server's and the machine's processes every MEMORY_S seconds
# into $OUT_ROOT/TAG/ and reports them (memory.json). SERVE_ENV: extra VAR=VALUE for the server.
#   PY=/path/to/venv/bin/python IMAGES=/path/to/images OUT_ROOT=runs/bench/machine \
#   GBC_VLM_UTIL=0.75 bash scripts/bench/machine/card.sh 0 u75
set -u
CARD=$1; TAG=$2
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-python}
OUT_ROOT=${OUT_ROOT:-$ROOT/runs/bench/machine}
CONFIG=configs/machine/qwen35_card.py
mkdir -p "$OUT_ROOT"
cd "$ROOT"
PORTS=${PORTS:-none}
SERVER=
if [ -n "${SERVE_PORT:-}" ]; then
  env CUDA_VISIBLE_DEVICES=$CARD ${SERVE_ENV:-} $PY scripts/bench/machine/serve.py $CONFIG \
    "$SERVE_PORT" > "$OUT_ROOT/$TAG.server.log" 2>&1 &
  SERVER=$!
  until curl -sf "http://127.0.0.1:$SERVE_PORT/health" > /dev/null; do
    kill -0 $SERVER 2> /dev/null || { echo "FAIL $TAG: server exited"; exit 1; }
    sleep 5
  done
  export GBC_VLM_URL=http://127.0.0.1:$SERVE_PORT/v1
  PORTS=$SERVE_PORT
fi
GBC_CARD=$CARD GBC_OUT=$OUT_ROOT/$TAG $PY scripts/bench/machine/run.py "$OUT_ROOT/$TAG" "$IMAGES" \
  "$PORTS" $CONFIG configs/controls/mid.py configs/controls/sam3.py > "$OUT_ROOT/$TAG.log" 2>&1 &
MACHINE=$!

# The machine runs in the background so the sampler can follow its pid (and the server's)
SAMPLER=
if [ -n "${MEMORY_S:-}" ]; then
  $PY scripts/bench/machine/memory.py sample "$OUT_ROOT/$TAG" "$MEMORY_S" "$PORTS" $MACHINE \
    $SERVER &
  SAMPLER=$!
fi
wait $MACHINE

[ -n "$SAMPLER" ] && kill -TERM $SAMPLER && wait $SAMPLER
[ -n "$SAMPLER" ] && $PY scripts/bench/machine/memory.py report "$OUT_ROOT/$TAG" > /dev/null
[ -n "$SERVER" ] && kill -INT $SERVER && wait $SERVER
echo "DONE $TAG"
