# 论文 Method 部分 — 技术内容详细提纲与草稿素材

> 以下内容基于代码实现逐一提取，所有数字、公式、维度均与代码一致。
> 供写作时参考，可直接改写为论文正文。

---

## 3.1 Overall Framework（整体框架）

### 需要写的内容

一段话概述整体流水线，配合 Fig.1（架构图）。

### 技术事实

整体流水线为：

```
多模态输入（SAR/RGB/GF，通道数各异）
  → Modal-Specific Stem（模态独立 Patch Embedding）
  → 4-Stage Swin Transformer（选定 block 中 FFN 替换为 Sparse MoE）
  → Modality-Separate UPerNet Decoder（每种模态独立解码头）
  → 二分类分割输出（flood / non-flood）
```

模型类：`MultiModalEncoderDecoderV2`。

**输入处理**：`MultiModalDataPreProcessor` 不对多模态输入做通道堆叠（因为各模态通道数不同），而是保持为 `List[Tensor]` 传入骨干网络。每个 tensor 仅做空间维度 padding 到 crop_size (256×256)。

**训练采样**：采用 `FixedRatioModalSampler` 按固定比例（SAR:RGB:GF = 6:5:5）组成每个 batch，以 GF 为参考模态确定 epoch 长度。这确保 SAR 数据（样本量最少）在每个 batch 中占 37.5%（6/16），高于其在数据集中的自然比例，缓解模态不均衡问题。

---

## 3.2 Modal-Specific Stem（模态特定嵌入层）

### 需要写的内容

解释为什么不能直接用标准 Patch Embedding，以及你的解决方案。

### 技术事实

**问题**：三种模态通道数不同（SAR=8, RGB=3, GF=5），标准 Swin Transformer 的 Patch Embedding 是一个固定输入通道的 Conv2d，无法处理可变通道输入。

**朴素方案（UnifiedPatchEmbed，用于消融对比）**：将所有模态零填充到最大通道数（8），然后共享一个 `Conv2d(8, embed_dims, kernel_size=4, stride=4)`。缺点是 RGB（3ch→8ch）和 GF（5ch→8ch）引入大量零值，稀释有效信息。

**本文方案（ModalSpecificPatchEmbed）**：为每种模态配置独立的投影卷积：

```
对于模态 m（通道数为 C_m）:
  Conv2d(C_m, embed_dims, kernel_size=patch_size, stride=patch_size)  +  LayerNorm(embed_dims)
```

具体实例化：
- SAR: `Conv2d(8, 128, kernel_size=4, stride=4)` + `LayerNorm(128)`
- RGB: `Conv2d(3, 128, kernel_size=4, stride=4)` + `LayerNorm(128)`
- GF:  `Conv2d(5, 128, kernel_size=4, stride=4)` + `LayerNorm(128)`

输入图像经 patch embedding 后，空间分辨率缩小为 H/4 × W/4，特征维度统一为 embed_dims=128。之后所有模态在同一特征空间中处理，共享 Transformer 的注意力层。

### 公式

$$
\mathbf{z}_0^{(m)} = \text{LN}\left(\text{Conv}_{m}(\mathbf{x}^{(m)})\right), \quad \mathbf{z}_0^{(m)} \in \mathbb{R}^{\frac{HW}{P^2} \times D}
$$

其中 $\text{Conv}_{m}: \mathbb{R}^{C_m \times H \times W} \to \mathbb{R}^{D \times \frac{H}{P} \times \frac{W}{P}}$ 为模态 $m$ 专属的卷积投影，$P=4$ 为 patch 大小，$D=128$ 为嵌入维度。

---

## 3.3 Swin Transformer Backbone with Sparse MoE（骨干网络）

### 需要写的内容

先简要介绍 Swin Transformer 基本结构（审稿人可能不熟悉），然后重点描述你在哪些位置插入了 MoE，以及为什么是稀疏放置。

### 技术事实

**Swin-Base 配置**：

| 参数 | 值 |
|------|-----|
| embed_dims | 128 |
| depths | [2, 2, 18, 2] |
| num_heads | [4, 8, 16, 32] |
| window_size | 7 |
| mlp_ratio | 4.0 |
| drop_path_rate | 0.3 |
| 各 stage 输出通道 | [128, 256, 512, 1024] |

**每个 Swin Block 的结构**（标准部分，简要提及即可）：

```
x = x + DropPath(W-MSA(LN(x)))    // 窗口多头自注意力
x = x + DropPath(FFN(LN(x)))      // 前馈网络（在选定 block 中替换为 MoE）
```

偶数 block 用 W-MSA，奇数 block 用 SW-MSA（shifted window）。

**MoE 替换位置（Sparse Placement）**：

并非所有 block 的 FFN 都替换为 MoE，而是选择性稀疏放置：

| Stage | 总 block 数 | MoE block 索引 | MoE 数量 | 说明 |
|-------|------------|---------------|---------|------|
| 0 | 2 | [] | 0 | 浅层特征通用性强，无需 MoE |
| 1 | 2 | [1] | 1 | 仅最后一个 block |
| 2 | 18 | [1,3,5,7,9,11,13,15,17] | 9 | 每隔一个 block 放置（交替） |
| 3 | 2 | [0, 1] | 2 | 全部 block |

**共享专家配置**：

| Stage | 共享专家数 |
|-------|-----------|
| 0 | 0 |
| 1 | 0 |
| 2 | 2 |
| 3 | 1 |

Stage 2 是主力计算阶段（18 个 block），交替放置 MoE 允许普通 FFN block 做特征整合，MoE block 做模态专门化。

**Stage 间下采样**：使用 PatchMerging，将空间分辨率减半、通道数加倍。

---

## 3.4 Sparse Mixture-of-Experts Module（稀疏 MoE 模块）

### 需要写的内容

这是方法的核心。需要分三个子部分详细描述：(1) 余弦相似度门控 (2) 模态偏置 (3) 稀疏路由与专家计算。

### 3.4.1 Cosine Similarity Gating（余弦相似度门控）

**技术事实**：

门控网络 `CosineTopKGate` 的完整计算流程：

1. **特征池化**：将输入 $\mathbf{X} \in \mathbb{R}^{B \times N \times C}$ 沿 token 维度均值池化得到 $\bar{\mathbf{x}} \in \mathbb{R}^{B \times C}$

2. **余弦相似度计算**：
   $$\mathbf{l} = \frac{\mathbf{W}_p \bar{\mathbf{x}}}{\|\mathbf{W}_p \bar{\mathbf{x}}\|_2} \cdot \frac{\mathbf{S}}{\|\mathbf{S}\|_2}$$
   其中 $\mathbf{W}_p \in \mathbb{R}^{C \times d}$ 为投影矩阵（`cosine_projector`），$d = \min(C/2, 256)$；$\mathbf{S} \in \mathbb{R}^{d \times E}$ 为相似度矩阵（`sim_matrix`），$E$ 为专家数。

3. **温度缩放**：
   $$\mathbf{l} = \mathbf{l} \cdot \exp\left(\text{clamp}(\tau, \max=\ln 100)\right)$$
   其中 $\tau$ 为可学习温度参数，初始化为 $\ln(1/0.5) \approx 0.693$。

4. **模态偏置注入**（见 3.4.2）

5. **噪声注入**（仅训练阶段）：
   $$\tilde{\mathbf{l}} = \mathbf{l} + \epsilon \cdot \text{Softplus}(\bar{\mathbf{x}} \mathbf{W}_{\text{noise}}), \quad \epsilon \sim \mathcal{N}(0, 1)$$
   其中 $\mathbf{W}_{\text{noise}} \in \mathbb{R}^{C \times E}$。噪声鼓励探索，防止路由僵化。

6. **Top-K 选择与归一化**：
   $$\text{TopK}(\tilde{\mathbf{l}}, k) \to (\mathbf{l}_{\text{top}}, \mathbf{I}_{\text{top}})$$
   $$\mathbf{g}_{\text{top}} = \text{Softmax}(\mathbf{l}_{\text{top}})$$
   将 $\mathbf{g}_{\text{top}}$ scatter 回完整的 $E$ 维向量，未被选中的专家权重为 0。

本文配置：$E=8$, $k=3$。

### 公式（汇总版，适合论文）

$$
\mathbf{g} = \text{TopK-Softmax}\Big(\underbrace{\text{CosSim}(\mathbf{W}_p \bar{\mathbf{x}},\ \mathbf{S})}_{\text{content-based routing}} \cdot e^{\tau} + \underbrace{\mathbf{b}_{m}}_{\text{modal bias}},\ k\Big)
$$

### 3.4.2 Learnable Modal Bias（可学习模态偏置）

**技术事实**：

参数：$\mathbf{B}_{\text{modal}} \in \mathbb{R}^{M \times E}$，其中 $M=3$（三种模态），$E=8$（专家数）。初始化为零矩阵。

应用方式：对于模态 $m$ 的输入样本，在门控 logits 上叠加偏置：
$$\mathbf{l}_i = \mathbf{l}_i + \mathbf{B}_{\text{modal}}[m, :]$$

每个 MoE 层有独立的 modal_bias 参数（不共享）。

**作用**：即使两个不同模态的 token 在特征空间中相近（余弦相似度门控给出相似路由），modal bias 也能将它们导向不同的专家。这为路由网络提供了显式的模态先验。

**学习率**：modal_bias 使用 3× 的学习率倍率（`lr_mult=3.0`），加速模态偏好的学习。

### 3.4.3 Expert Computation & Shared Experts（专家计算与共享专家）

**路由专家（Routed Experts）**：

每个 MoE 层包含 $E=8$ 个结构相同但参数独立的 FFN：
$$\text{FFN}_i(\mathbf{x}) = \mathbf{W}_2^{(i)} \cdot \text{GELU}(\mathbf{W}_1^{(i)} \mathbf{x}) + \mathbf{bias}_2^{(i)}$$
其中 $\mathbf{W}_1^{(i)} \in \mathbb{R}^{C \times 4C}$, $\mathbf{W}_2^{(i)} \in \mathbb{R}^{4C \times C}$（mlp_ratio=4）。

**稀疏派发（Sparse Dispatch）**：

通过 `SparseDispatcher` 实现，仅将每个样本发送到被选中的 top-k 专家：
1. 根据 gate 矩阵 $\mathbf{G} \in \mathbb{R}^{B \times E}$ 的非零位置确定路由
2. 按专家分组派发输入
3. 各专家独立计算
4. 按 gate 权重加权合并：$\mathbf{y}_{\text{routed}} = \sum_{i \in \text{TopK}} g_i \cdot \text{FFN}_i(\mathbf{x})$

**共享专家（Shared Experts）**：

共享专家不经过门控路由，对所有输入无条件执行：
$$\mathbf{y}_{\text{shared}} = \text{FFN}_{\text{shared}}(\mathbf{x})$$

当 Stage 2 配置 2 个共享专家时，其 FFN hidden_dim 为 $2 \times 4C$（等效于将两个专家的 FFN 拼接为一个更宽的 FFN）。

**最终输出**：
$$\mathbf{y} = \mathbf{y}_{\text{routed}} + \mathbf{y}_{\text{shared}}$$

---

## 3.5 Modality-Separate Decoder（模态独立解码头）

### 需要写的内容

解释为什么不用共享解码头，以及独立解码头的实现方式。

### 技术事实

**结构**：`decoder_mode='separate'` 时，为每种模态（sar/rgb/GF）各创建一个独立的 UPerHead + FCNHead（辅助头）。

**主解码头（UPerHead）配置**：

| 参数 | 值 |
|------|-----|
| in_channels | [128, 256, 512, 1024]（对应 4 个 stage 输出） |
| pool_scales | (1, 2, 3, 6)（PPM 模块） |
| channels | 512 |
| dropout_ratio | 0.1 |
| num_classes | 2（flood / non-flood） |
| loss | CrossEntropyLoss (weight=1.0) |

**辅助解码头（FCNHead）配置**：

| 参数 | 值 |
|------|-----|
| in_channels | 512（取 Stage 2 输出） |
| channels | 256 |
| num_convs | 1 |
| loss | CrossEntropyLoss (weight=0.4) |

**路由逻辑**：训练时，根据每个样本 metainfo 中的 `dataset_name` 字段，将同一 batch 内的样本按模态分组，分别送入对应的解码头计算 loss。推理时，根据输入样本的模态类型选择对应的解码头。

**与共享解码头对比**（消融实验 Table 2(f)）：共享模式下所有模态使用同一组 UPerHead 参数。

---

## 3.6 Training Objectives（训练目标）

### 需要写的内容

详细描述总损失函数的组成。

### 技术事实

总损失由三部分组成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{\text{div}} \mathcal{L}_{\text{diversity}}$$

**1. 分割损失 $\mathcal{L}_{\text{seg}}$**：

各模态独立计算 CrossEntropy loss，主头权重 1.0 + 辅助头权重 0.4：
$$\mathcal{L}_{\text{seg}} = \sum_{m \in \{sar, rgb, GF\}} \left(\mathcal{L}_{\text{CE}}^{m} + 0.4 \cdot \mathcal{L}_{\text{CE,aux}}^{m}\right)$$

**2. 负载均衡损失 $\mathcal{L}_{\text{balance}}$**（$\lambda_{\text{bal}}=1.0$）：

鼓励专家负载均匀，防止少数专家被过度使用：
$$\mathcal{L}_{\text{balance}} = \text{CV}^2(\text{importance}) + \text{CV}^2(\text{load})$$

其中：
- $\text{importance}_i = \sum_{b=1}^{B} g_{b,i}$（专家 $i$ 的总 gate 权重）
- $\text{load}_i = \sum_{b=1}^{B} \mathbb{1}[g_{b,i} > 0]$（专家 $i$ 被选中的次数）
- $\text{CV}^2(\mathbf{x}) = \frac{\text{Var}(\mathbf{x})}{(\text{Mean}(\mathbf{x}))^2 + \epsilon}$（变异系数的平方）

该损失在所有 MoE 层上求平均。

**3. 专家多样性损失 $\mathcal{L}_{\text{diversity}}$**（$\lambda_{\text{div}}=0.1$）：

防止不同专家学到相似的表征（专家坍缩）：

$$\mathcal{L}_{\text{diversity}} = \text{ReLU}\left(\frac{1}{E(E-1)/2} \sum_{i<j} \text{CosSim}(\mathbf{v}_i, \mathbf{v}_j)\right)$$

其中 $\mathbf{v}_i = \frac{1}{B}\sum_{b=1}^{B} \text{AvgPool}(\mathbf{h}_i^{(b)})$ 为专家 $i$ 在当前 batch 上的平均输出特征（L2 归一化后）。

ReLU 保证当专家表征已经足够不同（负余弦相似度）时，loss 为零。

---

## 3.7 Implementation Details（实现细节）

### 需要写的内容

训练和推理的所有超参数，确保可复现。

### 技术事实

**训练配置**：

| 项目 | 值 |
|------|-----|
| 框架 | MMSegmentation 1.2.2 + PyTorch |
| 优化器 | AdamW, lr=6×10⁻⁵, β=(0.9, 0.999), weight_decay=0.01 |
| 学习率调度 | 5 epoch linear warmup (start_factor=10⁻⁶) → Poly decay (power=1.0) |
| 训练轮次 | 100 epochs |
| Batch size | 16 |
| Crop size | 256×256 |
| 随机种子 | 42 |

**分组学习率**：

| 参数组 | lr 倍率 | weight_decay 倍率 |
|--------|---------|------------------|
| patch_embed, modal_patch_embeds | 2× | 1× |
| gating | 2× | 1× |
| experts | 1.5× | 1× |
| shared_experts | 2× | 1× |
| modal_bias | 3× | 1× |
| decode head | 10× | 1× |
| relative_position_bias_table, cls_token | 1× | 0× |
| norm layers | 1× | 0× |

**数据增强（训练）**：

```
RandomResize(scale=(2048, 512), ratio_range=(0.5, 2.0), keep_ratio=True)
RandomCrop(256×256, cat_max_ratio=0.75)
RandomFlip(prob=0.5)
MultiModalNormalize
```

**推理配置**：

| 项目 | 值 |
|------|-----|
| 模式 | Slide window |
| Crop size | 256×256 |
| Stride | 170×170（overlap ~33%） |
| Test image resize | 1024×1024 (keep_ratio=True) |

**SAR Boost 采样**：

| 模态 | 每 batch 样本数 | 比例 |
|------|---------------|------|
| SAR | 6 | 37.5% |
| RGB | 5 | 31.25% |
| GF | 5 | 31.25% |
| 合计 | 16 | 100% |

参考模态为 GF，即 epoch 长度由 GF 数据量决定。

---

## 数据集部分需要你自己补充的内容

以下信息代码中无法获取，需要你自行补充：

1. **每个数据集的详细描述**：
   - FloodNet：来源、地理覆盖、影像分辨率、采集传感器、洪涝事件名称
   - GF-FloodNet：5 通道的具体含义（R/G/B/NIR/？）、GaoFen 卫星型号
   - UrbanSARflood：8 通道的具体波段/极化方式（如 VV/VH/...）、SAR 传感器型号

2. **数据量统计表**：
   - 各数据集 train/val/test 的样本数量
   - 各数据集的 flood/non-flood 像素比例（类别不均衡程度）
   - 各数据集的空间分辨率

3. **数据集划分策略**：
   - 按地理区域划分还是随机划分？
   - 是否存在空间自相关泄露的风险？

4. **混合数据集的构建方式**：
   - `data_root = '../floodnet/data/mixed_dataset/'` 中三个数据集是如何合并的
   - 文件命名规则如何标识模态类型（代码中通过文件名 pattern 匹配：`sar`/`rgb`/`GF`）
