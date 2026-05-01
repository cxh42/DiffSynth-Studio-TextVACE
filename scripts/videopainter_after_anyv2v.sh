#!/bin/bash
# Wait for AnyV2V 160 done, then launch VideoPainter 7-GPU inference.
set -u
ANYV2V_OUT="outputs/baseline_anyv2v_inference_160"
LOG="logs/videopainter_auto_start.log"
mkdir -p logs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "Waiting for AnyV2V 160/160..."
while [ "$(ls "$ANYV2V_OUT"/*.mp4 2>/dev/null | wc -l)" -lt 160 ]; do sleep 60; done
log "AnyV2V complete."

# wait for run_group_*.py processes to fully exit
for j in $(seq 1 12); do
  alive=$(ps -eo cmd --no-headers | awk '/run_group_(ddim_inversion|pnp_edit)\.py/' | wc -l)
  if [ "$alive" -eq 0 ]; then break; fi
  log "  $alive AnyV2V workers still alive..."; sleep 30
done

# wait for GPU 1-7 memory to clear
for j in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2,3,4,5,6,7 | tr -d ' ' | sort -rn | head -1)
  [ -z "$used" ] && used=99999
  if [ "$used" -lt 500 ]; then break; fi
  log "  GPU residual $used MiB..."; sleep 10
done

log "Launching VideoPainter on 7 GPUs..."
nohup bash scripts/inference_videopainter_7gpu.sh > logs/videopainter_main.out 2>&1 &
disown
log "VideoPainter launcher PID=$!"
