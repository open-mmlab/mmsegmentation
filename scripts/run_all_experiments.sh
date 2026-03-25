#!/bin/bash
# ============================================================================
# FloodNet Paper Experiments - Complete Run Script
# Multi-Modal Flood Segmentation with Swin-Base + MoE
# ============================================================================
#
# NOTE: Full Model (Swin-B + MoE, E=8 K=3) is already trained and tested.
#       This script only runs ablation/hyperparameter/single-modal experiments.
#
# Usage:
#   bash scripts/run_all_experiments.sh table2      # Component Ablation
#   bash scripts/run_all_experiments.sh table3      # MoE Hyperparameter Study
#   bash scripts/run_all_experiments.sh table4      # Single-Modal vs Multi-Modal
#   bash scripts/run_all_experiments.sh all         # Run all tables
#
# Environment Variables:
#   GPU_IDS   GPU device IDs (default: 0)
#
# Example:
#   GPU_IDS=0 bash scripts/run_all_experiments.sh table2
#
# Each experiment uses --seed 42 for reproducibility.
# After training, each experiment runs per-modality testing automatically.
# ============================================================================

set -e

SEED=42
GPU_IDS=${GPU_IDS:-0}
CONFIG_DIR="configs/floodnet"
ABLATION_DIR="${CONFIG_DIR}/ablations"
WORK_ROOT="work_dirs/paper_experiments"
RESULTS_LOG="${WORK_ROOT}/results_summary.txt"

GROUP=${1:-"all"}

mkdir -p "${WORK_ROOT}"

# ============================================================================
# Helper Functions
# ============================================================================

run_train() {
    local config=$1
    local work_dir=$2
    local desc=$3

    echo "============================================================"
    echo "[TRAIN] ${desc}"
    echo "[CFG]   ${config}"
    echo "[DIR]   ${work_dir}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/train.py \
        ${config} \
        --work-dir ${work_dir} \
        --cfg-options randomness.seed=${SEED}

    echo "[TRAIN DONE] ${desc}"
    echo ""
}

find_best_ckpt() {
    local work_dir=$1
    local best_ckpt=$(ls ${work_dir}/best_mIoU_*.pth 2>/dev/null | head -1)
    if [ -z "$best_ckpt" ]; then
        best_ckpt=$(ls ${work_dir}/epoch_*.pth 2>/dev/null | sort -V | tail -1)
    fi
    echo "$best_ckpt"
}

run_test_modal() {
    local config=$1
    local checkpoint=$2
    local work_dir=$3
    local modal=$4
    local desc=$5

    local test_work_dir="${work_dir}/test_${modal}"
    mkdir -p "${test_work_dir}"

    echo "------------------------------------------------------------"
    echo "[TEST] ${desc} | Modality: ${modal}"
    echo "[CKPT] ${checkpoint}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/test.py \
        ${config} \
        ${checkpoint} \
        --work-dir ${test_work_dir} \
        --cfg-options \
            test_dataloader.dataset.filter_modality="${modal}" \
        2>&1 | tee "${test_work_dir}/test_log.txt"

    echo "[${desc}] modal=${modal} -> see ${test_work_dir}/test_log.txt" >> "${RESULTS_LOG}"
    echo ""
}

run_test_all_modals() {
    local config=$1
    local checkpoint=$2
    local work_dir=$3
    local desc=$4

    echo "============================================================"
    echo "[TEST ALL MODALS] ${desc}"
    echo "============================================================"

    run_test_modal "${config}" "${checkpoint}" "${work_dir}" "sar" "${desc}"
    run_test_modal "${config}" "${checkpoint}" "${work_dir}" "rgb" "${desc}"
    run_test_modal "${config}" "${checkpoint}" "${work_dir}" "GF"  "${desc}"
}

train_and_test_all_modals() {
    local config=$1
    local work_dir=$2
    local desc=$3

    run_train "${config}" "${work_dir}" "${desc}"

    local ckpt=$(find_best_ckpt "${work_dir}")
    if [ -z "$ckpt" ]; then
        echo "[ERROR] No checkpoint found in ${work_dir} after training"
        return 1
    fi

    run_test_all_modals "${config}" "${ckpt}" "${work_dir}" "${desc}"
}

train_and_test_single_modal() {
    local config=$1
    local work_dir=$2
    local modal=$3
    local desc=$4

    run_train "${config}" "${work_dir}" "${desc}"

    local ckpt=$(find_best_ckpt "${work_dir}")
    if [ -z "$ckpt" ]; then
        echo "[ERROR] No checkpoint found in ${work_dir} after training"
        return 1
    fi

    run_test_modal "${config}" "${ckpt}" "${work_dir}" "${modal}" "${desc}"
}

# ============================================================================
# TABLE 2: Component Ablation Study
# ============================================================================
# Full Model (a) is already trained and tested — NOT included here.
# Only the ablation variants (b)-(f) are trained and tested.
#
# (b) w/o MoE              — train + test SAR/RGB/GF
# (c) w/o ModalSpecificStem— train + test SAR/RGB/GF
# (d) w/o Modal Bias       — train + test SAR/RGB/GF
# (e) w/o Shared Experts   — train + test SAR/RGB/GF
# (f) w/o Separate Decoder — train + test SAR/RGB/GF
# ============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "table2" ]]; then
    echo ""
    echo "################################################################"
    echo "#  TABLE 2: Component Ablation Study                          #"
    echo "#  (Full Model result is already available)                    #"
    echo "################################################################"
    echo ""

    # (b) w/o MoE
    train_and_test_all_modals \
        "${ABLATION_DIR}/ablation_no_moe.py" \
        "${WORK_ROOT}/table2/no_moe" \
        "Table2(b) w/o MoE"

    # (c) w/o ModalSpecificStem
    train_and_test_all_modals \
        "${ABLATION_DIR}/ablation_no_modal_specific_stem.py" \
        "${WORK_ROOT}/table2/no_modal_specific_stem" \
        "Table2(c) w/o ModalSpecificStem"

    # (d) w/o Modal Bias
    train_and_test_all_modals \
        "${ABLATION_DIR}/ablation_no_modal_bias.py" \
        "${WORK_ROOT}/table2/no_modal_bias" \
        "Table2(d) w/o Modal Bias"

    # (e) w/o Shared Experts
    train_and_test_all_modals \
        "${ABLATION_DIR}/ablation_no_shared_experts.py" \
        "${WORK_ROOT}/table2/no_shared_experts" \
        "Table2(e) w/o Shared Experts"

    # (f) w/o Separate Decoder (use shared decoder)
    train_and_test_all_modals \
        "${ABLATION_DIR}/ablation_shared_decoder.py" \
        "${WORK_ROOT}/table2/shared_decoder" \
        "Table2(f) w/o Separate Decoder"

    echo ""
    echo "[TABLE 2 COMPLETE] Results in ${WORK_ROOT}/table2/"
    echo ""
fi

# ============================================================================
# TABLE 3: MoE Hyperparameter Study
# ============================================================================
# Grid: num_experts={6, 8} x top_k={1, 2, 3}
# (8, 3) = Full Model — already done, NOT included here.
#
# Each variant tested on SAR / RGB / GF separately.
# ============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "table3" ]]; then
    echo ""
    echo "################################################################"
    echo "#  TABLE 3: MoE Hyperparameter Study                          #"
    echo "#  (E=8 K=3 Full Model result is already available)           #"
    echo "################################################################"
    echo ""

    # E=6, K=1
    train_and_test_all_modals \
        "${ABLATION_DIR}/ablation_e6_k1.py" \
        "${WORK_ROOT}/table3/e6_k1" \
        "Table3 E=6 K=1"

    # E=6, K=2
    train_and_test_all_modals \
        "${ABLATION_DIR}/ablation_e6_k2.py" \
        "${WORK_ROOT}/table3/e6_k2" \
        "Table3 E=6 K=2"

    # E=6, K=3
    train_and_test_all_modals \
        "${ABLATION_DIR}/ablation_e6_k3.py" \
        "${WORK_ROOT}/table3/e6_k3" \
        "Table3 E=6 K=3"

    # E=8, K=1
    train_and_test_all_modals \
        "${ABLATION_DIR}/ablation_e8_k1.py" \
        "${WORK_ROOT}/table3/e8_k1" \
        "Table3 E=8 K=1"

    # E=8, K=2
    train_and_test_all_modals \
        "${ABLATION_DIR}/ablation_e8_k2.py" \
        "${WORK_ROOT}/table3/e8_k2" \
        "Table3 E=8 K=2"

    # E=8, K=3 = Full Model — SKIP (already trained and tested)
    echo "[SKIP] E=8 K=3 = Full Model (already trained and tested)"

    echo ""
    echo "[TABLE 3 COMPLETE] Results in ${WORK_ROOT}/table3/"
    echo ""
fi

# ============================================================================
# TABLE 4: Single-Modal vs Multi-Modal Training
# ============================================================================
# SAR-only  → train + test SAR
# RGB-only  → train + test RGB
# GF-only   → train + test GF
# Multi-modal (Full Model) → SKIP (already trained and tested)
# ============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "table4" ]]; then
    echo ""
    echo "################################################################"
    echo "#  TABLE 4: Single-Modal vs Multi-Modal                       #"
    echo "#  (Multi-modal Full Model result is already available)        #"
    echo "################################################################"
    echo ""

    # SAR-only → test SAR
    train_and_test_single_modal \
        "${CONFIG_DIR}/multimodal_floodnet_sar_only_swinbase_moe_config.py" \
        "${WORK_ROOT}/table4/sar_only" \
        "sar" \
        "Table4 SAR-only"

    # RGB-only → test RGB
    train_and_test_single_modal \
        "${ABLATION_DIR}/ablation_rgb_only.py" \
        "${WORK_ROOT}/table4/rgb_only" \
        "rgb" \
        "Table4 RGB-only"

    # GF-only → test GF
    train_and_test_single_modal \
        "${ABLATION_DIR}/ablation_gf_only.py" \
        "${WORK_ROOT}/table4/gf_only" \
        "GF" \
        "Table4 GF-only"

    # Multi-modal = Full Model — SKIP
    echo "[SKIP] Multi-modal = Full Model (already trained and tested)"

    echo ""
    echo "[TABLE 4 COMPLETE] Results in ${WORK_ROOT}/table4/"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "EXPERIMENTS COMPLETED"
echo "Results log: ${RESULTS_LOG}"
echo "============================================================"
echo ""
echo "Result directories:"
if [[ "$GROUP" == "all" || "$GROUP" == "table2" ]]; then
    echo "  Table 2 (Ablation):     ${WORK_ROOT}/table2/"
fi
if [[ "$GROUP" == "all" || "$GROUP" == "table3" ]]; then
    echo "  Table 3 (MoE Hyper):    ${WORK_ROOT}/table3/"
fi
if [[ "$GROUP" == "all" || "$GROUP" == "table4" ]]; then
    echo "  Table 4 (Single-Modal): ${WORK_ROOT}/table4/"
fi
echo ""
echo "Per-modality test logs: <work_dir>/test_{sar,rgb,GF}/test_log.txt"
