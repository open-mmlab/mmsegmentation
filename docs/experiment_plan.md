# FloodNet Paper Experiment Plan
## Multi-Modal Flood Segmentation via Mixture-of-Experts with Modal-Aware Routing

---

## 1. Model Overview (Final Model)

**Config**: `multimodal_floodnet_sar_boost_swinbase_moe_config.py`

| Component | Configuration |
|-----------|--------------|
| Backbone | Swin-Base (embed_dims=128, depths=[2,2,18,2]) |
| Patch Embed | ModalSpecificStem (独立卷积 per modality) |
| MoE | 8 experts, top_k=3, noisy gating |
| Shared Experts | Stage 2: 2, Stage 3: 1 |
| Modal Bias | Learnable per-modal routing bias |
| Diversity Loss | weight=0.1; Balance Loss weight=1.0 |
| Decoder | Separate UPerHead per modality |
| Sampling | SAR Boost (6:5:5) |
| Modalities | SAR (8ch), RGB (3ch), GaoFen (5ch) |

---

## 2. Experiment Design

### Table 2: Component Ablation Study (组件消融)

**目的**: 隔离每个组件的贡献，验证设计合理性。

**评估**: 每个变体训练后，分别在 SAR / RGB / GF 三个模态上独立测试 mIoU。

| Row | 配置 | 变更 | SAR | RGB | GF | Avg |
|-----|------|------|-----|-----|-----|-----|
| (a) | **Full Model** | — | - | - | - | - |
| (b) | w/o MoE | 标准 FFN, 无专家路由 | - | - | - | - |
| (c) | w/o ModalSpecificStem | 统一卷积+零填充 替代 模态独立卷积 | - | - | - | - |
| (d) | w/o Modal Bias | MoE gating 无模态偏置 | - | - | - | - |
| (e) | w/o Shared Experts | 所有专家均为路由专家，无共享 | - | - | - | - |
| (f) | w/o Separate Decoder | 一个 UPerHead 共享所有模态 | - | - | - | - |

**Config Files**:
- (a) `multimodal_floodnet_sar_boost_swinbase_moe_config.py`
- (b) `ablations/ablation_no_moe.py`
- (c) `ablations/ablation_no_modal_specific_stem.py`
- (d) `ablations/ablation_no_modal_bias.py`
- (e) `ablations/ablation_no_shared_experts.py`
- (f) `ablations/ablation_shared_decoder.py`

**运行**:
```bash
bash scripts/run_all_experiments.sh table2
```

**讨论要点**:

**(b) w/o MoE**: 预期最大下降。标准 FFN 无法适配不同模态的特征分布差异。所有 token 共享相同权重，无论来自 SAR 后向散射还是 RGB 反射率。

**(c) w/o ModalSpecificStem**: 统一 patch embedding 将所有模态零填充到 max_channels(8) 再卷积，丢失了模态特异性的早期特征提取。SAR 的 8 通道数据被原样保留，但 RGB(3ch) 和 GF(5ch) 被大量零填充，引入噪声。

**(d) w/o Modal Bias**: 去除模态路由偏置后，gating 网络对所有模态一视同仁。模态偏置允许模型学习到某些专家专门处理 SAR 的散斑噪声和后向散射强度 vs RGB 的颜色纹理。

**(e) w/o Shared Experts**: 共享专家捕获模态不变特征（如水体空间结构、边界形态）。移除后，路由专家需冗余学习通用特征，降低专门化能力。

**(f) w/o Separate Decoder**: 单一解码器须在不同模态特征分布间妥协。独立解码器允许每个模态有定制化的上采样和分类边界。

---

### Table 3: MoE Hyperparameter Study (MoE超参数研究)

**目的**: 研究专家数量和 top-k 路由对性能的影响。

**评估**: 每个变体训练后，分别在 SAR / RGB / GF 三个模态上独立测试 mIoU。

| num_experts | top_k | SAR | RGB | GF | Avg |
|-------------|-------|-----|-----|-----|-----|
| 6 | 1 | - | - | - | - |
| 6 | 2 | - | - | - | - |
| 6 | 3 | - | - | - | - |
| 8 | 1 | - | - | - | - |
| 8 | 2 | - | - | - | - |
| **8** | **3** | **-** | **-** | **-** | **-** |

**Config Files**:
- `ablations/ablation_e6_k1.py`
- `ablations/ablation_e6_k2.py`
- `ablations/ablation_e6_k3.py`
- `ablations/ablation_e8_k1.py`
- `ablations/ablation_e8_k2.py`
- Full Model (E=8, K=3)

**运行**:
```bash
bash scripts/run_all_experiments.sh table3
```

**讨论要点**:
- **top_k=1**: 过于稀疏，每个 token 仅使用 1 个专家，限制了多专家集成效果
- **top_k=2**: 中等稀疏度，基本满足 3 模态路由需求
- **top_k=3 (ours)**: 允许 token 同时受益于多个专门化专家
- **6 experts**: 对 3 种模态来说容量偏小，平均每模态仅 2 个专家
- **8 experts (ours)**: 最佳平衡点，每模态约 2-3 个专家并允许跨模态共享

---

### Table 4: Single-Modal vs Multi-Modal Training (单模态 vs 多模态)

**目的**: 证明多模态联合训练通过跨模态知识迁移提升了每个模态的性能。

**评估**: 单模态训练只测试本模态；多模态训练测试所有三个模态。

| 训练数据 | 测试模态 | mIoU |
|----------|----------|------|
| SAR-only | SAR | - |
| RGB-only | RGB | - |
| GF-only | GF | - |
| Multi-modal (Ours) | SAR | - |
| Multi-modal (Ours) | RGB | - |
| Multi-modal (Ours) | GF | - |

**Config Files**:
- SAR-only → `multimodal_floodnet_sar_only_swinbase_moe_config.py`
- RGB-only → `ablations/ablation_rgb_only.py`
- GF-only → `ablations/ablation_gf_only.py`
- Multi-modal → Full Model

**运行**:
```bash
bash scripts/run_all_experiments.sh table4
```

**讨论要点**:
- 多模态训练预期在**每个**模态上都优于单模态训练
- SAR 受益最大：有限的 SAR 数据通过 RGB/GF 知识迁移得到增强
- MoE 架构防止负迁移：模态专用专家避免了不同传感器类型间的干扰
- 共享专家捕获通用洪水模式（水体几何、边界结构），在所有模态间迁移

---

## 3. Running Experiments

```bash
# 分表运行（推荐）
bash scripts/run_all_experiments.sh table2      # 组件消融 (6 实验 × 3 模态测试)
bash scripts/run_all_experiments.sh table3      # MoE 超参数 (6 实验 × 3 模态测试)
bash scripts/run_all_experiments.sh table4      # 单/多模态 (4 实验)

# 全部运行
bash scripts/run_all_experiments.sh all

# 指定 GPU
GPU_IDS=0,1 bash scripts/run_all_experiments.sh table2
```

**结果目录结构**:
```
work_dirs/paper_experiments/
├── results_summary.txt          # 汇总日志
├── table2/
│   ├── full_model/
│   │   ├── best_mIoU_*.pth
│   │   ├── test_sar/test_log.txt
│   │   ├── test_rgb/test_log.txt
│   │   └── test_GF/test_log.txt
│   ├── no_moe/
│   ├── no_modal_specific_stem/
│   ├── no_modal_bias/
│   ├── no_shared_experts/
│   └── shared_decoder/
├── table3/
│   ├── e6_k1/
│   ├── e6_k2/
│   ├── e6_k3/
│   ├── e8_k1/
│   └── e8_k2/
└── table4/
    ├── sar_only/
    │   └── test_sar/test_log.txt
    ├── rgb_only/
    │   └── test_rgb/test_log.txt
    ├── gf_only/
    │   └── test_GF/test_log.txt
    └── (multi_modal reuses table2/full_model)
```

**实验总量**: 15 个训练 + 48 次测试
- Table 2: 6 训练 × 3 模态测试 = 6 + 18
- Table 3: 5 训练 × 3 模态测试 + 1 复用 = 5 + 18
- Table 4: 3 训练 × 1 模态测试 + 1 复用 × 3 测试 = 3 + 6
