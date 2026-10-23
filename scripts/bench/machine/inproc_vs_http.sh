#!/bin/bash
# inproc_vs_http.sh PREFIX C0 C1 C2 C3: the same machine (card.sh) on four cards at once, the VLM
# in-process (C0, C1) and behind `vllm serve` on its card (C2, C3), each with vLLM's own scheduler
# (C0, C2) and KohakuvLLM's PhasedScheduler (C1, C3; KOHAKU_PREFILL_EVERY, 16 by default). Tags:
# PREFIX{inproc,http}_{stock,kohaku}. Every other knob passes through the environment (card.sh).
set -u
PREFIX=$1
HERE=$(cd "$(dirname "$0")" && pwd)
PHASED=kohakuvllm.sched.phased.PhasedScheduler
export KOHAKU_PREFILL_EVERY=${KOHAKU_PREFILL_EVERY:-16}
bash "$HERE/card.sh" "$2" "${PREFIX}inproc_stock" &
GBC_VLM_SCHEDULER=$PHASED bash "$HERE/card.sh" "$3" "${PREFIX}inproc_kohaku" &
SERVE_PORT=$((8100 + $4)) bash "$HERE/card.sh" "$4" "${PREFIX}http_stock" &
SERVE_PORT=$((8100 + $5)) GBC_VLM_SCHEDULER=$PHASED bash "$HERE/card.sh" "$5" "${PREFIX}http_kohaku" &
wait
echo "ALL DONE $PREFIX"
