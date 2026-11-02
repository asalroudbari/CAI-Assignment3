#!/bin/bash
# Run the TA-bot against a real vLLM server on PACE-ICE and save everything to pace/results/.
# The project folder can be in home or scratch; the vLLM environment (many GB) and all caches
# go to your scratch folder, because home has a 30 GB quota.
#
#   1. One-time setup, on a login node (no GPU needed, ~10 min):
#        bash pace/run_pace.sh --setup
#   2. The run, as a batch job (~15 min once it starts):
#        sbatch pace/run_pace.sh               # progress: tail -f pace-<jobid>.out
#      or inside an interactive GPU session (salloc --gres=gpu:1 ...):
#        bash pace/run_pace.sh
#
# Overrides (environment variables): MODEL (e.g. Llama-3.2-3B-Instruct), PORT, A3_HF_HOME,
# A3_SCRATCH (your scratch folder, if it is not found automatically; see pace-quota).

#SBATCH -J a3-pace-check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o pace-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
ROOT=$(pwd -P)
SCRATCH=${A3_SCRATCH:-$(ls -d /storage/ice1/*/*/"$USER" 2>/dev/null | head -1 || true)}
[ -d "$SCRATCH" ] || { echo "ERROR: scratch folder not found. Run pace-quota and set A3_SCRATCH=<the Scratch path>."; exit 1; }
WORK=$SCRATCH/a3-pace
echo "== project: $ROOT | environment and caches: $WORK"

ENV_DIR=$WORK/venv
export TMPDIR=$WORK/tmp XDG_CACHE_HOME=$WORK/cache PIP_NO_CACHE_DIR=1
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export HF_HOME=${A3_HF_HOME:-/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" pace/results

# --- environment (built once) ------------------------------------------------
if [ ! -x "$ENV_DIR/bin/vllm" ]; then
  echo "== building $ENV_DIR"
  module load anaconda3 2>/dev/null || true   # system python3 is 3.9, too old for vLLM
  python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
    || { echo "ERROR: need Python >= 3.10. In your shell run 'module load anaconda3' (or a python/3.1x"\
              "module from 'module avail python'), then re-run this script."; exit 1; }
  rm -rf "$ENV_DIR"
  python3 -m venv "$ENV_DIR"
  "$ENV_DIR/bin/pip" install --upgrade pip
  "$ENV_DIR/bin/pip" install vllm requests
fi
source "$ENV_DIR/bin/activate"
python -c 'import sys, vllm; print("== python", sys.version.split()[0], "| vllm", vllm.__version__)'
if [ "$1" = "--setup" ]; then
  echo "== setup done. Next: sbatch pace/run_pace.sh"
  exit 0
fi

# --- GPU and model -----------------------------------------------------------
command -v nvidia-smi >/dev/null || { echo "ERROR: no GPU here. Use sbatch, or salloc --gres=gpu:1 first."; exit 1; }
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv | tee pace/results/gpu.txt
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
dtype=auto
[ "${cc%%.*}" -lt 8 ] && dtype=half   # V100 and older have no bfloat16

if [ -z "$MODEL" ]; then
  MODEL=Llama-3.1-8B-Instruct
  if [ "$mem" -lt 22000 ]; then
    MODEL=Llama-3.2-3B-Instruct
    echo "WARNING: this GPU has $mem MiB, too little for the 8B model; using $MODEL."
  fi
fi
dir=$(ls "$HF_HOME/hub" | grep -i -- "--$MODEL\$" | head -1 || true)
[ -n "$dir" ] || { echo "ERROR: $MODEL not found. Available in $HF_HOME/hub:"; ls "$HF_HOME/hub"; exit 1; }
repo=${dir#models--}
repo=${repo/--//}   # models--meta-llama--Llama-3.1-8B-Instruct -> meta-llama/Llama-3.1-8B-Instruct

# --- vLLM server -------------------------------------------------------------
PORT=${PORT:-$((20000 + RANDOM % 10000))}   # random port: other users' jobs may share the node
LOG=pace/results/vllm-$MODEL.log
echo "== starting vLLM: $repo as '$MODEL', dtype=$dtype, port $PORT (log: $LOG)"
vllm serve "$repo" --served-model-name "$MODEL" --dtype "$dtype" --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --enforce-eager --port "$PORT" > "$LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for _ in $(seq 120); do   # up to 20 min
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "ERROR: vLLM exited. Last lines of $LOG:"; tail -40 "$LOG"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "ERROR: vLLM not ready after 20 min"; tail -40 "$LOG"; exit 1; }
echo "== vLLM ready"

# --- checks ------------------------------------------------------------------
URL=http://localhost:$PORT/v1
python eval_dev.py --endpoint "$URL" --model "$MODEL" | tee "pace/results/eval_dev-$MODEL.txt" \
  || echo "(eval_dev.py exited non-zero; see the output above)"
python pace/pace_check.py --endpoint "$URL" --model "$MODEL"
echo "== done. Results are in $ROOT/pace/results/"
