#!/bin/bash
# Debug wrapper for train_textvace_14b_121f.sh
# Enables NCCL flight recorder + WARN logs so a collective timeout dumps per-rank last op + stack.
set -e

# NCCL: keep logs WARN-level (avoid INFO firehose), surface desync traces
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=./logs/nccl_trace/nccl_trace

# Workaround for NCCL 2.26.2 UB / symmetric-memory allgather hang with ZeRO-3
# (torch 2.7.0 bundles this NCCL; known regression — pytorch/pytorch#150381, NVIDIA/nccl#1702)
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

# Where the flight recorder dumps go
mkdir -p ./logs/nccl_trace

exec bash "$(dirname "$0")/train_textvace_14b_121f.sh"
