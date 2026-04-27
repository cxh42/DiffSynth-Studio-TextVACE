#!/bin/bash
# Wait for epoch-2 training (autoloop) to complete, then run:
#   1. Full inference on 235 samples with step-576 (8 GPUs)
#   2. Overfitting check with 4 ckpts on 30-sample subset (8 GPUs)
# Refuses to launch if training didn't fully succeed.
set -u

CKPT_FINAL="models/train/TextVACE_14B_sft_121f/step-576.safetensors"
LOGFILE="logs/post_training_inference.log"
mkdir -p "$(dirname "$LOGFILE")"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGFILE"; }

log "Watching for epoch-2 training completion..."

# 1. Wait for autoloop process to be gone
while pgrep -f "train_textvace_14b_121f_autoloop" > /dev/null; do
  sleep 60
done
log "autoloop process exited."

# 2. Wait briefly for ckpt write to settle (in case file just appeared)
sleep 30

# 3. Verify final ckpt exists
if [ ! -f "$CKPT_FINAL" ]; then
  log "ERROR: $CKPT_FINAL not found. Training may have failed all retries. Aborting inference."
  exit 1
fi
log "Found $CKPT_FINAL ($(stat -c %s "$CKPT_FINAL") bytes)."

# 4. Confirm no leftover GPU users (residual ranks holding context)
for j in $(seq 1 12); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -rn | head -1)
  [ -z "$used" ] && used=99999
  if [ "$used" -lt 500 ]; then break; fi
  log "GPU still has $used MiB residual; waiting..."
  sleep 10
done

# 5. Full inference
log "===== Step 1/2: full inference (235 samples, step-576) ====="
bash scripts/inference_full_8gpu.sh 2>&1 | tee -a "$LOGFILE"

# 6. Overfitting eval
log "===== Step 2/2: overfitting eval (30 samples × 4 ckpts) ====="
bash scripts/inference_overfitting_8gpu.sh 2>&1 | tee -a "$LOGFILE"

log "All inference complete."
log "  Full results:        outputs/inference_full_step576/"
log "  Overfitting results: outputs/inference_overfitting/step-{100,288,432,576}/"
