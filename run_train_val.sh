#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/cm-dgseg/cm_dgseg_b0_cityscapes.py}
GPUS=${2:-1}
WORK_DIR=${3:-work_dirs/$(basename "${CONFIG}" .py)}
CHECKPOINT=${4:-latest.pth}

mkdir -p "${WORK_DIR}"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"
SMI_LOG="${LOG_DIR}/nvidia_smi.csv"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "timestamp,index,name,utilization.gpu [%],utilization.memory [%],memory.total [MiB],memory.used [MiB],memory.free [MiB]" > "${SMI_LOG}"
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free \
    --format=csv,noheader -l 60 >> "${SMI_LOG}" &
  SMI_PID=$!
else
  echo "nvidia-smi not found; skipping GPU telemetry." >&2
  SMI_PID=0
fi

cleanup() {
  if [[ ${SMI_PID} -ne 0 ]]; then
    kill ${SMI_PID} >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

TRAIN_CMD=(python tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" --auto-resume)
if [[ ${GPUS} -gt 1 ]]; then
  TRAIN_CMD+=(--launcher pytorch --devices "${GPUS}")
fi
"${TRAIN_CMD[@]}"

TEST_CMD=(python tools/test.py "${CONFIG}" "${WORK_DIR}/${CHECKPOINT}" --eval mIoU)
"${TEST_CMD[@]}"
