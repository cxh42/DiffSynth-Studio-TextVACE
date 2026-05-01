#!/bin/bash
# Wait until VACE baseline (160 mp4) is fully done, then:
#   1. Update inference_160/metadata.csv prompt field to instruction template (per user)
#   2. Launch AnyV2V on GPU 1-7
set -u

VACE_OUT="outputs/baseline_vace14b_zeroshot_inference_160"
META="data/inference_new/inference_160/metadata.csv"
LOG="logs/anyv2v_auto_start.log"
mkdir -p logs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "Waiting for VACE baseline 160/160..."
while [ "$(ls "$VACE_OUT"/*.mp4 2>/dev/null | wc -l)" -lt 160 ]; do
  sleep 60
done
log "VACE baseline complete: 160 mp4 present"

# Confirm no active VACE workers (so GPU 1-7 free)
for j in $(seq 1 12); do
  alive=$(ps -eo cmd --no-headers | awk '/inference_textvace_14b\.py.*inference_160/' | wc -l)
  if [ "$alive" -eq 0 ]; then break; fi
  log "Waiting for $alive VACE workers to exit..."
  sleep 30
done

# Wait for GPU memory to clear (max 5 min)
for j in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2,3,4,5,6,7 | tr -d ' ' | sort -rn | head -1)
  [ -z "$used" ] && used=99999
  if [ "$used" -lt 500 ]; then break; fi
  log "GPU residual $used MiB; waiting..."
  sleep 10
done

# Update metadata.csv: prompt field stays target_text (used by GlyphVACE),
# but per user, also update the 'instruction' column to match Family-B template.
log "Updating $META instruction column..."
/home/xinghao/miniconda3/envs/DiffSynth-Studio/bin/python <<'PY'
import csv
from pathlib import Path
p = Path("data/inference_new/inference_160/metadata.csv")
rows = list(csv.DictReader(open(p)))
fieldnames = list(rows[0].keys())
if "instruction" not in fieldnames:
    fieldnames.append("instruction")
changed = 0
for r in rows:
    new = f"Change {r['source_text']} to {r['target_text']}; preserve everything else."
    if r.get("instruction") != new:
        r["instruction"] = new; changed += 1
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
print(f"updated {changed}/{len(rows)} rows")
PY

log "Launching AnyV2V (7 GPU)..."
nohup bash scripts/inference_anyv2v_7gpu.sh > logs/anyv2v_main.out 2>&1 &
disown
log "AnyV2V launcher PID=$!"
log "Per-shard logs in logs/anyv2v_inference_160/"
