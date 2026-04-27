#!/bin/bash
# Auto-restart wrapper: runs training, on crash kills residue, picks latest
# step-N ckpt (falls back to 49f), and retries. Stops on success or MAX_RESTARTS.
set -u

MAX_RESTARTS=30
CKPT_DIR="./models/train/TextVACE_14B_sft_121f"
FALLBACK="./models/train/TextVACE_14B_sft_49f/epoch-4.safetensors"
LOGDIR="./logs"
mkdir -p "$LOGDIR"

pick_ckpt() {
  # latest valid step-N.safetensors by step number (not mtime, in case of skew)
  ls -1 "$CKPT_DIR"/step-*.safetensors 2>/dev/null \
    | awk -F'step-|\\.safetensors' '{print $2" "$0}' \
    | sort -k1 -n \
    | tail -1 \
    | awk '{print $2}'
}

cleanup() {
  pkill -9 -f "examples/wanvideo/model_training/train.py" 2>/dev/null || true
  pkill -9 -f "accelerate launch" 2>/dev/null || true
  sleep 5
}

for attempt in $(seq 1 $MAX_RESTARTS); do
  ckpt="$(pick_ckpt)"
  if [ -z "$ckpt" ]; then
    ckpt="$FALLBACK"
    step=0
  else
    # extract N from .../step-N.safetensors
    step=$(basename "$ckpt" | sed -E 's/^step-([0-9]+)\.safetensors$/\1/')
    [[ "$step" =~ ^[0-9]+$ ]] || step=0
  fi

  ts="$(date +%Y%m%d_%H%M%S)"
  runlog="$LOGDIR/train_14b_121f_run${attempt}_${ts}.log"

  echo "========================================" | tee -a "$LOGDIR/train_14b_121f.log"
  echo "[$(date)] attempt $attempt / $MAX_RESTARTS" | tee -a "$LOGDIR/train_14b_121f.log"
  echo "[$(date)] resume from: $ckpt (step=$step)" | tee -a "$LOGDIR/train_14b_121f.log"
  echo "[$(date)] per-run log: $runlog" | tee -a "$LOGDIR/train_14b_121f.log"
  echo "========================================" | tee -a "$LOGDIR/train_14b_121f.log"

  # create symlink BEFORE running so `tail -F` can follow real-time
  ln -sfn "$(basename "$runlog")" "$LOGDIR/train_14b_121f.log.latest"

  CKPT_OVERRIDE="$ckpt" RESUME_STEP="$step" bash scripts/train_textvace_14b_121f_debug.sh > "$runlog" 2>&1
  rc=$?

  tail -200 "$runlog" >> "$LOGDIR/train_14b_121f.log"

  if [ $rc -eq 0 ]; then
    echo "[$(date)] training complete (rc=0)" | tee -a "$LOGDIR/train_14b_121f.log"
    exit 0
  fi

  echo "[$(date)] crash rc=$rc, cleaning up" | tee -a "$LOGDIR/train_14b_121f.log"
  cleanup

  # check GPU clear before retry
  for j in $(seq 1 12); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -rn | head -1)
    [ -z "$used" ] && used=99999
    if [ "$used" -lt 500 ]; then break; fi
    sleep 5
  done
done

echo "[$(date)] giving up after $MAX_RESTARTS restarts" | tee -a "$LOGDIR/train_14b_121f.log"
exit 1
