#!/usr/bin/env bash
# Score all pairs with Qwen3.8-27B: start the repo's server on the GPUs that
# gpu-lock gave us (CUDA_VISIBLE_DEVICES; one tensor-parallel-2 replica per
# pair), wait for it, run score_server.py, exit with its code. Run from the
# repo root through gpu-lock, which kills the server (same process group)
# and verifies the GPU memory afterwards:
#   .venv/bin/python .claude/skills/gpu-lock/gpu_lock.py run --exp 20q-qwen38-scoring \
#       --log experiments/2026-10-17-20q-clue-ablation/logs/score_qwen38.log --gpus 4,5 -- \
#       bash experiments/2026-10-17-20q-clue-ablation/run_qwen38_scoring.sh
set -euo pipefail
E=experiments/2026-10-17-20q-clue-ablation
PORT=8011
GPUS="${CUDA_VISIBLE_DEVICES:?run through gpu-lock}"
n=$(echo "$GPUS" | tr , '\n' | wc -l)
(( n % 2 == 0 )) || { echo "need an even number of GPUs (tensor parallel 2), got $GPUS"; exit 2; }
dp=$(( n / 2 ))
echo "[run] serving Qwen3.8-27B on GPUs $GPUS ($dp replica(s) x TP2), port $PORT"
.venv-serve/bin/python serving/serve_qwen38.py --gpus "$GPUS" --data_parallel_size "$dp" --port "$PORT" \
    --hf_hub_offline true --log_dir "$E/logs" &
for _ in $(seq 180); do
    curl -sf -m 5 "localhost:$PORT/v1/models" >/dev/null && break
    sleep 10
done
curl -sf -m 5 "localhost:$PORT/v1/models" >/dev/null || { echo "[run] server did not come up in 30 min"; exit 3; }
echo "[run] server up; scoring"
.venv/bin/python "$E/score_server.py" --base-url "http://localhost:$PORT/v1" --model Qwen/Qwen3.8-27B \
    --out qwen38_scores.jsonl --concurrency $(( 256 * dp ))
echo "[run] scoring finished"
