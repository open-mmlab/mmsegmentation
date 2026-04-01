#!/bin/bash
# ============================================================================
# Re-test All Table 2/3/4 Experiments with Expanded Metrics
# Metrics: mIoU, mDice, mFscore (Precision, Recall, F1), aAcc (OA)
#
# Usage:
#   bash scripts/test_all_metrics.sh table2
#   bash scripts/test_all_metrics.sh table3
#   bash scripts/test_all_metrics.sh table4
#   bash scripts/test_all_metrics.sh full       # Full Model only
#   bash scripts/test_all_metrics.sh all        # All tables + Full Model
#
# Environment Variables:
#   GPU_IDS   GPU device IDs (default: 0)
#
# Example:
#   GPU_IDS=0 bash scripts/test_all_metrics.sh table2
# ============================================================================

set -e

GPU_IDS=${GPU_IDS:-0}
CONFIG_DIR="configs/floodnet"
ABLATION_DIR="${CONFIG_DIR}/ablations"
WORK_ROOT="work_dirs/paper_experiments"
METRICS_LOG="${WORK_ROOT}/metrics_summary.txt"

GROUP=${1:-"all"}

mkdir -p "${WORK_ROOT}"
echo "========== Metrics Test Run: $(date) ==========" >> "${METRICS_LOG}"

# ============================================================================
# Helper Functions
# ============================================================================

find_best_ckpt() {
    local work_dir=$1
    local best_ckpt=$(ls ${work_dir}/best_mIoU_*.pth 2>/dev/null | head -1)
    if [ -z "$best_ckpt" ]; then
        best_ckpt=$(ls ${work_dir}/epoch_*.pth 2>/dev/null | sort -V | tail -1)
    fi
    echo "$best_ckpt"
}

run_test_metrics() {
    local config=$1
    local checkpoint=$2
    local work_dir=$3
    local modal=$4
    local desc=$5

    local test_work_dir="${work_dir}/metrics_${modal}"
    mkdir -p "${test_work_dir}"

    echo "------------------------------------------------------------"
    echo "[TEST] ${desc} | Modality: ${modal}"
    echo "[CKPT] ${checkpoint}"
    echo "[OUT]  ${test_work_dir}"
    echo "------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/test.py \
        ${config} \
        ${checkpoint} \
        --work-dir ${test_work_dir} \
        --cfg-options \
            test_dataloader.dataset.filter_modality="${modal}" \
            "test_evaluator.iou_metrics=['mIoU','mDice','mFscore']" \
        2>&1 | tee "${test_work_dir}/metrics_log.txt"

    echo "[${desc}] modal=${modal} -> ${test_work_dir}/metrics_log.txt" >> "${METRICS_LOG}"
    echo ""
}

run_test_all_modals_metrics() {
    local config=$1
    local checkpoint=$2
    local work_dir=$3
    local desc=$4

    echo "============================================================"
    echo "[METRICS TEST] ${desc}"
    echo "============================================================"

    run_test_metrics "${config}" "${checkpoint}" "${work_dir}" "sar" "${desc}"
    run_test_metrics "${config}" "${checkpoint}" "${work_dir}" "rgb" "${desc}"
    run_test_metrics "${config}" "${checkpoint}" "${work_dir}" "GF"  "${desc}"
}

# ============================================================================
# Full Model
# ============================================================================

run_full_model() {
    echo ""
    echo "############################################################"
    echo "# Full Model (Swin-B + MoE, E=8 K=3)"
    echo "############################################################"

    local config="${CONFIG_DIR}/multimodal_floodnet_sar_boost_swinbase_moe_config.py"
    local work_dir="work_dirs/floodnet/SwinmoeB/655"
    local ckpt="${work_dir}/best_mIoU_epoch_100.pth"

    if [ ! -f "$ckpt" ]; then
        echo "[WARN] Full Model checkpoint not found: ${ckpt}"
        echo "[WARN] Trying to find best checkpoint in ${work_dir}..."
        ckpt=$(find_best_ckpt "${work_dir}")
    fi

    if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
        echo "[ERROR] No checkpoint found for Full Model. Skipping."
        return 0
    fi

    run_test_all_modals_metrics "${config}" "${ckpt}" "${work_dir}" "Full Model"
}

# ============================================================================
# Table 2: Component Ablation
# ============================================================================

run_table2() {
    echo ""
    echo "############################################################"
    echo "# Table 2: Component Ablation Study"
    echo "############################################################"

    local base_config="${CONFIG_DIR}/multimodal_floodnet_sar_boost_swinbase_moe_config.py"

    declare -A T2_CONFIGS
    T2_CONFIGS=(
        ["no_moe"]="${ABLATION_DIR}/ablation_no_moe.py"
        ["no_modal_specific_stem"]="${ABLATION_DIR}/ablation_no_modal_specific_stem.py"
        ["no_modal_bias"]="${ABLATION_DIR}/ablation_no_modal_bias.py"
        ["no_shared_experts"]="${ABLATION_DIR}/ablation_no_shared_experts.py"
        ["shared_decoder"]="${ABLATION_DIR}/ablation_shared_decoder.py"
    )

    declare -A T2_DESCS
    T2_DESCS=(
        ["no_moe"]="Table2: w/o MoE"
        ["no_modal_specific_stem"]="Table2: w/o ModalSpecificStem"
        ["no_modal_bias"]="Table2: w/o Modal Bias"
        ["no_shared_experts"]="Table2: w/o Shared Experts"
        ["shared_decoder"]="Table2: w/o Separate Decoder"
    )

    for key in no_moe no_modal_specific_stem no_modal_bias no_shared_experts shared_decoder; do
        local config="${T2_CONFIGS[$key]}"
        local work_dir="${WORK_ROOT}/table2/${key}"
        local desc="${T2_DESCS[$key]}"

        local ckpt=$(find_best_ckpt "${work_dir}")
        if [ -z "$ckpt" ]; then
            echo "[ERROR] No checkpoint found for ${desc} in ${work_dir}. Skipping."
            continue
        fi

        run_test_all_modals_metrics "${config}" "${ckpt}" "${work_dir}" "${desc}"
    done
}

# ============================================================================
# Table 3: MoE Hyperparameter Study
# ============================================================================

run_table3() {
    echo ""
    echo "############################################################"
    echo "# Table 3: MoE Hyperparameter Study"
    echo "############################################################"

    declare -A T3_CONFIGS
    T3_CONFIGS=(
        ["e6_k1"]="${ABLATION_DIR}/ablation_e6_k1.py"
        ["e6_k2"]="${ABLATION_DIR}/ablation_e6_k2.py"
        ["e6_k3"]="${ABLATION_DIR}/ablation_e6_k3.py"
        ["e8_k1"]="${ABLATION_DIR}/ablation_e8_k1.py"
        ["e8_k2"]="${ABLATION_DIR}/ablation_e8_k2.py"
    )

    declare -A T3_DESCS
    T3_DESCS=(
        ["e6_k1"]="Table3: E=6 K=1"
        ["e6_k2"]="Table3: E=6 K=2"
        ["e6_k3"]="Table3: E=6 K=3"
        ["e8_k1"]="Table3: E=8 K=1"
        ["e8_k2"]="Table3: E=8 K=2"
    )

    for key in e6_k1 e6_k2 e6_k3 e8_k1 e8_k2; do
        local config="${T3_CONFIGS[$key]}"
        local work_dir="${WORK_ROOT}/table3/${key}"
        local desc="${T3_DESCS[$key]}"

        local ckpt=$(find_best_ckpt "${work_dir}")
        if [ -z "$ckpt" ]; then
            echo "[ERROR] No checkpoint found for ${desc} in ${work_dir}. Skipping."
            continue
        fi

        run_test_all_modals_metrics "${config}" "${ckpt}" "${work_dir}" "${desc}"
    done
}

# ============================================================================
# Table 4: Single-Modal vs Multi-Modal
# ============================================================================

run_table4() {
    echo ""
    echo "############################################################"
    echo "# Table 4: Single-Modal vs Multi-Modal"
    echo "############################################################"

    # SAR-only: test on SAR
    local sar_config="${CONFIG_DIR}/multimodal_floodnet_sar_only_swinbase_moe_config.py"
    local sar_dir="${WORK_ROOT}/table4/sar_only"
    local sar_ckpt=$(find_best_ckpt "${sar_dir}")
    if [ -n "$sar_ckpt" ]; then
        local test_dir="${sar_dir}/metrics_sar"
        mkdir -p "${test_dir}"
        echo "------------------------------------------------------------"
        echo "[TEST] Table4: SAR-Only | Modality: sar"
        echo "[CKPT] ${sar_ckpt}"
        echo "------------------------------------------------------------"
        CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/test.py \
            ${sar_config} ${sar_ckpt} \
            --work-dir ${test_dir} \
            --cfg-options \
                "test_evaluator.iou_metrics=['mIoU','mDice','mFscore']" \
            2>&1 | tee "${test_dir}/metrics_log.txt"
        echo "[Table4: SAR-Only] modal=sar -> ${test_dir}/metrics_log.txt" >> "${METRICS_LOG}"
    else
        echo "[ERROR] No checkpoint found for SAR-Only in ${sar_dir}. Skipping."
    fi

    # RGB-only: test on RGB
    local rgb_config="${ABLATION_DIR}/ablation_rgb_only.py"
    local rgb_dir="${WORK_ROOT}/table4/rgb_only"
    local rgb_ckpt=$(find_best_ckpt "${rgb_dir}")
    if [ -n "$rgb_ckpt" ]; then
        local test_dir="${rgb_dir}/metrics_rgb"
        mkdir -p "${test_dir}"
        echo "------------------------------------------------------------"
        echo "[TEST] Table4: RGB-Only | Modality: rgb"
        echo "[CKPT] ${rgb_ckpt}"
        echo "------------------------------------------------------------"
        CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/test.py \
            ${rgb_config} ${rgb_ckpt} \
            --work-dir ${test_dir} \
            --cfg-options \
                "test_evaluator.iou_metrics=['mIoU','mDice','mFscore']" \
            2>&1 | tee "${test_dir}/metrics_log.txt"
        echo "[Table4: RGB-Only] modal=rgb -> ${test_dir}/metrics_log.txt" >> "${METRICS_LOG}"
    else
        echo "[ERROR] No checkpoint found for RGB-Only in ${rgb_dir}. Skipping."
    fi

    # GF-only: test on GF
    local gf_config="${ABLATION_DIR}/ablation_gf_only.py"
    local gf_dir="${WORK_ROOT}/table4/gf_only"
    local gf_ckpt=$(find_best_ckpt "${gf_dir}")
    if [ -n "$gf_ckpt" ]; then
        local test_dir="${gf_dir}/metrics_GF"
        mkdir -p "${test_dir}"
        echo "------------------------------------------------------------"
        echo "[TEST] Table4: GF-Only | Modality: GF"
        echo "[CKPT] ${gf_ckpt}"
        echo "------------------------------------------------------------"
        CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/test.py \
            ${gf_config} ${gf_ckpt} \
            --work-dir ${test_dir} \
            --cfg-options \
                "test_evaluator.iou_metrics=['mIoU','mDice','mFscore']" \
            2>&1 | tee "${test_dir}/metrics_log.txt"
        echo "[Table4: GF-Only] modal=GF -> ${test_dir}/metrics_log.txt" >> "${METRICS_LOG}"
    else
        echo "[ERROR] No checkpoint found for GF-Only in ${gf_dir}. Skipping."
    fi
}

# ============================================================================
# Main
# ============================================================================

echo "============================================================"
echo " Expanded Metrics Testing"
echo " Metrics: mIoU, mDice, mFscore (Precision/Recall/F1), aAcc"
echo " Group: ${GROUP}"
echo " GPU: ${GPU_IDS}"
echo "============================================================"

case "${GROUP}" in
    full)
        run_full_model
        ;;
    table2)
        run_table2
        ;;
    table3)
        run_table3
        ;;
    table4)
        run_table4
        ;;
    all)
        run_full_model
        run_table2
        run_table3
        run_table4
        ;;
    *)
        echo "Unknown group: ${GROUP}"
        echo "Usage: bash scripts/test_all_metrics.sh {full|table2|table3|table4|all}"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " All metrics tests completed!"
echo " Results log: ${METRICS_LOG}"
echo "============================================================"
