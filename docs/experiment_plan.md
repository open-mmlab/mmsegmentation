# FloodNet Paper Experiment Plan
## Multi-Modal Flood Segmentation via Mixture-of-Experts with Modal-Aware Routing

---

## 1. Model Overview

**Final Model**: `Swin-Base + MoE + UPerNet` (multimodal_floodnet_sar_boost_swinbase_moe_config.py)

| Component | Configuration |
|-----------|--------------|
| Backbone | Swin-Base (embed_dims=128, depths=[2,2,18,2]) |
| MoE | 8 experts, top_k=3, noisy gating |
| Shared Experts | Stage 2: 2, Stage 3: 1 |
| Modal Bias | Learnable per-modal routing bias in gating |
| Diversity Loss | Expert diversity loss (weight=0.1) + Balance loss (weight=1.0) |
| Decoder | Separate UPerHead per modality |
| Sampling | SAR Boost (6:5:5 ratio) |
| Modalities | SAR (8ch), RGB (3ch), GaoFen (5ch) |
| Params | ~456M |

---

## 2. Experiment Design

### Table 1: Comparison with State-of-the-Art Methods

**Purpose**: Demonstrate the overall advantage of our approach over existing methods.

**Evaluation**: Report mIoU on test set, split by modality (SAR / RGB / GF) and overall average.

| Method | Backbone | Params | SAR mIoU | RGB mIoU | GF mIoU | Avg mIoU |
|--------|----------|--------|----------|----------|---------|----------|
| U-Net | ResNet-50 | ~25M | - | - | - | - |
| DeepLabV3+ | ResNet-101 | ~63M | - | - | - | - |
| SegFormer | MiT-B2 | ~25M | - | - | - | - |
| Swin-B + UPerNet (no MoE) | Swin-Base | ~109M | - | - | - | - |
| Swin-T + MoE | Swin-Tiny | ~278M | - | - | - | - |
| **Ours (Swin-B + MoE)** | **Swin-Base** | **~456M** | **-** | **-** | **-** | **-** |

**Configs**:
- Swin-B no MoE → `ablations/ablation_no_moe.py`
- Swin-T + MoE → `multimodal_floodnet_sar_boost_swin_moe_config.py`
- Ours → `multimodal_floodnet_sar_boost_swinbase_moe_config.py`
- U-Net, DeepLabV3+, SegFormer → use standard mmseg configs with multi-modal wrapper

**Discussion Points**:
- MoE brings significant gains by dedicating expert capacity to each modality
- Swin-B + MoE > Swin-T + MoE: larger backbone provides richer base features for expert specialization
- Swin-B + MoE > Swin-B alone: MoE adds modality-adaptive capacity beyond parameter scaling

---

### Table 2: Component Ablation Study

**Purpose**: Isolate the contribution of each proposed component.

**Evaluation**: All variants use Swin-Base backbone, test on SAR / RGB / GF separately.

| Row | Configuration | Change | SAR | RGB | GF | Avg |
|-----|--------------|--------|-----|-----|-----|-----|
| (a) | **Full Model** | — | - | - | - | - |
| (b) | w/o MoE | Standard FFN, no experts | - | - | - | - |
| (c) | w/o Modal Bias | MoE gating without modal-specific bias | - | - | - | - |
| (d) | w/o Shared Experts | All experts are routed, no shared | - | - | - | - |
| (e) | w/o Diversity Loss | No expert specialization regularization | - | - | - | - |
| (f) | w/o SAR Boost | Uniform random sampling | - | - | - | - |
| (g) | Shared Decoder | One UPerHead for all modalities | - | - | - | - |

**Configs**:
- (a) `multimodal_floodnet_sar_boost_swinbase_moe_config.py`
- (b) `ablations/ablation_no_moe.py`
- (c) `ablations/ablation_no_modal_bias.py`
- (d) `ablations/ablation_no_shared_experts.py`
- (e) `ablations/ablation_no_diversity_loss.py`
- (f) `ablations/ablation_uniform_sampling.py`
- (g) `ablations/ablation_shared_decoder.py`

**Expected Findings & Discussion**:

**(b) w/o MoE**: Largest drop expected. Standard FFN cannot adapt to modality-specific feature distributions. All tokens share the same FFN weights regardless of whether they come from SAR backscatter or RGB reflectance.

**(c) w/o Modal Bias**: Moderate drop, especially on SAR. Without modal bias, the gating network treats all modalities identically during routing. The modal bias allows the model to learn that certain experts specialize in SAR-specific patterns (e.g., speckle noise, backscatter intensity) vs RGB-specific patterns (e.g., color, texture).

**(d) w/o Shared Experts**: Shared experts capture modality-invariant features (e.g., spatial structure of water bodies). Removing them forces each routed expert to redundantly learn common features, reducing capacity for specialization.

**(e) w/o Diversity Loss**: Without diversity regularization, experts may collapse to similar representations (expert collapse problem). The diversity loss encourages each expert to learn distinct feature transformations, verified by higher pairwise cosine distance between expert outputs.

**(f) w/o SAR Boost**: SAR is the minority modality with fewer training samples. Without oversampling, the model underperforms on SAR due to insufficient exposure. This demonstrates the importance of modality-aware sampling in imbalanced multi-modal settings.

**(g) Shared Decoder**: A single decoder must compromise between modality-specific feature distributions. Separate decoders allow each modality to have tailored upsampling and classification boundaries.

---

### Table 3: MoE Hyperparameter Study

**Purpose**: Study the effect of expert count and top-k routing.

| num_experts | top_k | Active Ratio | Params | SAR | RGB | GF | Avg |
|-------------|-------|-------------|--------|-----|-----|-----|-----|
| 4 | 2 | 50% | ~280M | - | - | - | - |
| **8** | **3** | **37.5%** | **~456M** | **-** | **-** | **-** | **-** |
| 16 | 4 | 25% | ~850M | - | - | - | - |
| 8 | 1 | 12.5% | ~456M | - | - | - | - |

**Configs**:
- `ablations/ablation_experts_4.py`
- Full model (8 experts, k=3)
- `ablations/ablation_experts_16.py`
- `ablations/ablation_topk_1.py`

**Discussion Points**:
- **4 experts**: Insufficient capacity for 3 modalities — experts must share across modalities
- **8 experts (ours)**: Sweet spot — roughly 2-3 experts per modality with cross-modal sharing
- **16 experts**: Marginal gains or slight degradation due to sparse gradient issues and insufficient data to train all experts
- **top_k=1**: Too sparse — each token only uses 1 expert, limiting multi-expert ensemble benefit
- **top_k=3**: Allows tokens to benefit from multiple specialized experts simultaneously

---

### Table 4: Single-Modal vs Multi-Modal Training

**Purpose**: Demonstrate that multi-modal co-training with MoE improves each individual modality through cross-modal knowledge transfer.

| Training Data | Test Modality | mIoU |
|--------------|---------------|------|
| SAR-only | SAR | - |
| RGB-only | RGB | - |
| GF-only | GF | - |
| Multi-modal (Ours) | SAR | - |
| Multi-modal (Ours) | RGB | - |
| Multi-modal (Ours) | GF | - |

**Configs**:
- SAR-only → `multimodal_floodnet_sar_only_swinbase_moe_config.py`
- RGB-only → `ablations/ablation_rgb_only.py`
- GF-only → `ablations/ablation_gf_only.py`
- Multi-modal → Full model, tested per-modality

**Discussion Points**:
- Multi-modal training improves **every** modality compared to single-modal training
- SAR benefits most: limited SAR data is augmented by knowledge transfer from RGB/GF
- The MoE architecture prevents negative transfer: modality-specific experts avoid interference between different sensor types
- Shared experts capture universal flood patterns (water body geometry, boundary structures) that transfer across all modalities

---

## 3. Additional Analysis (Figures)

### Figure A: Expert Routing Visualization
- Visualize the gating weights for each modality
- Show that different experts are preferentially activated for SAR vs RGB vs GF
- Plot: Heatmap of expert activation probability per modality (averaged over test set)

### Figure B: Training Convergence Curves
- Plot mIoU vs epoch for Full Model vs key ablations
- Show that MoE converges faster and to a higher plateau
- Compare SAR-specific convergence with/without SAR Boost

### Figure C: Qualitative Segmentation Results
- Side-by-side comparison on challenging flood scenes:
  - SAR images with complex backscatter patterns
  - RGB images with cloud shadows / urban areas
  - GaoFen images with mixed land cover
- Show predictions from: Full Model vs No MoE vs Single-Modal

### Figure D: Expert Diversity Analysis
- t-SNE visualization of expert output features
- With diversity loss: experts form distinct clusters
- Without diversity loss: experts collapse to similar representations

---

## 4. Paper Section Outline

### Results Section (Section 4)

**4.1 Experimental Setup**
- Dataset: FloodNet multi-modal (SAR/RGB/GaoFen), binary flood segmentation
- Metrics: mIoU, per-class IoU (flood/non-flood)
- Implementation: MMSegmentation, AdamW, lr=6e-5, 100 epochs, batch_size=16

**4.2 Comparison with State-of-the-Art** (Table 1)
- Our method achieves best results across all modalities
- Key insight: MoE scales model capacity without proportional compute increase at inference

**4.3 Ablation Studies** (Table 2)
- Each component contributes meaningfully
- MoE and Modal Bias are the two most critical components
- SAR Boost is essential for handling modality imbalance

**4.4 MoE Configuration Analysis** (Table 3)
- 8 experts with top_k=3 provides optimal balance
- Justification for design choices

**4.5 Cross-Modal Transfer Learning** (Table 4)
- Multi-modal MoE enables positive cross-modal transfer
- No negative transfer thanks to expert routing

### Discussion Section (Section 5)

**5.1 Why MoE for Multi-Modal Segmentation?**
- Different modalities have fundamentally different feature distributions
- MoE naturally partitions the feature space via expert routing
- Modal bias makes this partitioning explicit and learnable

**5.2 The Role of Shared Experts**
- Bridge between modality-specific and universal features
- Capture flood-invariant patterns (shape, boundary, spatial context)
- Prevent complete expert isolation across modalities

**5.3 Handling Modality Imbalance**
- SAR data is typically scarcer than optical data
- Fixed-ratio sampling (SAR Boost) ensures sufficient exposure
- Combined with MoE, prevents dominant modality from monopolizing experts

**5.4 Scalability and Efficiency**
- ~456M params but only top_k=3 experts active per token
- Actual FLOPs comparable to ~200M dense model
- Linear scaling of experts without quadratic compute growth

**5.5 Limitations and Future Work**
- Current approach trains modalities in mixed batches (not fused at pixel level)
- Future: explore pixel-level multi-modal fusion with MoE
- Extend to more modalities (thermal, LiDAR)
- Investigate dynamic expert allocation based on image difficulty

---

## 5. Running the Experiments

```bash
# Run all experiments sequentially
bash scripts/run_all_experiments.sh

# Run specific groups
bash scripts/run_all_experiments.sh --group main       # Full model only
bash scripts/run_all_experiments.sh --group sota       # SOTA baselines
bash scripts/run_all_experiments.sh --group ablation   # Component ablation
bash scripts/run_all_experiments.sh --group moe_hyper  # MoE hyperparameters
bash scripts/run_all_experiments.sh --group single     # Single-modal baselines

# Multi-GPU (set GPU_IDS)
GPU_IDS=0,1,2,3 bash scripts/run_all_experiments.sh
```

**Total Experiments**: 14 training runs
- 1 full model
- 2 SOTA baselines (Swin-T MoE, Swin-B no MoE)
- 6 component ablations
- 3 MoE hyperparameter variants
- 3 single-modal baselines
(Note: some experiments serve double duty across tables)
