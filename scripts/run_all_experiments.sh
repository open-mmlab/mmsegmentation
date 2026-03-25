#!/bin/bash
# ============================================================================
# FloodNet Paper Experiments - Complete Run Script
# Multi-Modal Flood Segmentation with Swin-Base + MoE
# ============================================================================
#
# Usage:
#   bash scripts/run_all_experiments.sh          # Run all experiments
#   bash scripts/run_all_experiments.sh --group ablation   # Run only ablation group
#   bash scripts/run_all_experiments.sh --group sota       # Run only SOTA comparison
#   bash scripts/run_all_experiments.sh --group moe_hyper  # Run MoE hyperparameter sweep
#   bash scripts/run_all_experiments.sh --group single     # Run single-modal experiments
#
# Each experiment uses --seed 42 for reproducibility.
# ============================================================================

set -e

SEED=42
GPU_IDS=${GPU_IDS:-0}
CONFIG_DIR="configs/floodnet"
ABLATION_DIR="${CONFIG_DIR}/ablations"
WORK_ROOT="work_dirs/paper_experiments"

GROUP=${2:-"all"}

run_train() {
    local config=$1
    local work_dir=$2
    local desc=$3

    echo "============================================================"
    echo "[EXP] ${desc}"
    echo "[CFG] ${config}"
    echo "[DIR] ${work_dir}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/train.py \
        ${config} \
        --work-dir ${work_dir} \
        --seed ${SEED}

    echo "[DONE] ${desc}"
    echo ""
}

run_test() {
    local config=$1
    local checkpoint=$2
    local work_dir=$3
    local desc=$4

    echo "[TEST] ${desc}"
    CUDA_VISIBLE_DEVICES=${GPU_IDS} python tools/test.py \
        ${config} \
        ${checkpoint} \
        --work-dir ${work_dir}
}

# ============================================================================
# GROUP 1: Main Model (Full / Final)
# ============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "main" ]]; then
    echo "========== GROUP: Main Model =========="

    # Exp 0: Full Model (Swin-Base + MoE, SAR Boost) -- THE FINAL MODEL
    run_train \
        "${CONFIG_DIR}/multimodal_floodnet_sar_boost_swinbase_moe_config.py" \
        "${WORK_ROOT}/full_model_swinB_moe" \
        "Full Model: Swin-B + MoE + All Components"
fi

# ============================================================================
# GROUP 2: SOTA Comparison Baselines
# ============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "sota" ]]; then
    echo "========== GROUP: SOTA Baselines =========="

    # Exp 1a: Swin-Tiny + MoE (smaller backbone)
    run_train \
        "${CONFIG_DIR}/multimodal_floodnet_sar_boost_swin_moe_config.py" \
        "${WORK_ROOT}/sota/swinT_moe" \
        "Backbone Scale: Swin-Tiny + MoE"

    # Exp 1b: Swin-Base WITHOUT MoE (standard FFN baseline)
    run_train \
        "${ABLATION_DIR}/ablation_no_moe.py" \
        "${WORK_ROOT}/sota/swinB_no_moe" \
        "Baseline: Swin-Base + UPerNet (No MoE)"
fi

# ============================================================================
# GROUP 3: Component Ablation Study (Table 2)
# ============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "ablation" ]]; then
    echo "========== GROUP: Ablation Study =========="

    # Exp 2a: w/o MoE
    run_train \
        "${ABLATION_DIR}/ablation_no_moe.py" \
        "${WORK_ROOT}/ablation/no_moe" \
        "Ablation: w/o MoE"

    # Exp 2b: w/o Modal Bias
    run_train \
        "${ABLATION_DIR}/ablation_no_modal_bias.py" \
        "${WORK_ROOT}/ablation/no_modal_bias" \
        "Ablation: w/o Modal Bias"

    # Exp 2c: w/o Shared Experts
    run_train \
        "${ABLATION_DIR}/ablation_no_shared_experts.py" \
        "${WORK_ROOT}/ablation/no_shared_experts" \
        "Ablation: w/o Shared Experts"

    # Exp 2d: w/o Expert Diversity Loss
    run_train \
        "${ABLATION_DIR}/ablation_no_diversity_loss.py" \
        "${WORK_ROOT}/ablation/no_diversity_loss" \
        "Ablation: w/o Expert Diversity Loss"

    # Exp 2e: w/o SAR Boost (Uniform Sampling)
    run_train \
        "${ABLATION_DIR}/ablation_uniform_sampling.py" \
        "${WORK_ROOT}/ablation/uniform_sampling" \
        "Ablation: w/o SAR Boost (Uniform Sampling)"

    # Exp 2f: Shared Decoder
    run_train \
        "${ABLATION_DIR}/ablation_shared_decoder.py" \
        "${WORK_ROOT}/ablation/shared_decoder" \
        "Ablation: Shared Decoder (vs Separate)"
fi

# ============================================================================
# GROUP 4: MoE Hyperparameter Study (Table 3)
# ============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "moe_hyper" ]]; then
    echo "========== GROUP: MoE Hyperparameter Study =========="

    # Exp 3a: 4 experts, top_k=2
    run_train \
        "${ABLATION_DIR}/ablation_experts_4.py" \
        "${WORK_ROOT}/moe_hyper/experts_4_topk_2" \
        "MoE Hyper: 4 experts, top_k=2"

    # Exp 3b: 8 experts, top_k=3 (default - reuse full model result)
    echo "[SKIP] 8 experts, top_k=3 = Full Model (already run)"

    # Exp 3c: 16 experts, top_k=4
    run_train \
        "${ABLATION_DIR}/ablation_experts_16.py" \
        "${WORK_ROOT}/moe_hyper/experts_16_topk_4" \
        "MoE Hyper: 16 experts, top_k=4"

    # Exp 3d: 8 experts, top_k=1
    run_train \
        "${ABLATION_DIR}/ablation_topk_1.py" \
        "${WORK_ROOT}/moe_hyper/experts_8_topk_1" \
        "MoE Hyper: 8 experts, top_k=1"
fi

# ============================================================================
# GROUP 5: Single-Modal vs Multi-Modal (Table 4)
# ============================================================================
if [[ "$GROUP" == "all" || "$GROUP" == "single" ]]; then
    echo "========== GROUP: Single-Modal vs Multi-Modal =========="

    # Exp 4a: SAR-only (already exists)
    run_train \
        "${CONFIG_DIR}/multimodal_floodnet_sar_only_swinbase_moe_config.py" \
        "${WORK_ROOT}/single_modal/sar_only" \
        "Single-Modal: SAR-only"

    # Exp 4b: RGB-only
    run_train \
        "${ABLATION_DIR}/ablation_rgb_only.py" \
        "${WORK_ROOT}/single_modal/rgb_only" \
        "Single-Modal: RGB-only"

    # Exp 4c: GF-only
    run_train \
        "${ABLATION_DIR}/ablation_gf_only.py" \
        "${WORK_ROOT}/single_modal/gf_only" \
        "Single-Modal: GF-only"

    # Exp 4d: Multi-modal (reuse full model)
    echo "[SKIP] Multi-Modal = Full Model (already run)"
fi

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "Results saved in: ${WORK_ROOT}/"
echo "============================================================"
