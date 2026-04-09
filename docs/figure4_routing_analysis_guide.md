# Figure 4: Expert Routing Analysis — 技术文档

本文档详细说明 Figure 4 四张子图的数据来源、制作方法、数学原理和读图方式，供撰写论文时参考。

---

## 总体背景

本模型在 Swin Transformer 的 FFN 位置替换为 Mixture-of-Experts (MoE) 层。每个 MoE 层包含 E=8 个并行的 expert FFN，每次推理仅激活 top_k=3 个。路由（routing）决策由 **CosineTopKGate** 完成，其输出决定每个输入样本由哪些 expert 处理。

### 模型中 MoE 的分布

| Stage | 维度 | 总 block 数 | MoE block 数 | 共享 expert 数 |
|-------|------|-------------|-------------|---------------|
| 1     | 256  | 2           | 1           | 0             |
| 2     | 512  | 18          | 9           | 2             |
| 3     | 1024 | 2           | 2           | 1             |

Stage 0 无 MoE block。

### Gating 机制数学表达

对于输入特征 x ∈ R^(B×N×C)（B=batch, N=token数, C=通道数），gating 过程如下：

1. **全局池化**: x̄ = mean(x, dim=1)，得到 x̄ ∈ R^(B×C)
2. **余弦相似度投影**:
   ```
   logits = cosine_sim(W_proj · x̄, S) × τ
   ```
   其中 W_proj ∈ R^(C×d) 是投影矩阵，S ∈ R^(d×E) 是 expert 相似度矩阵，τ = exp(temperature) 是可学习温度参数。
3. **加入模态偏置**:
   ```
   logits = logits + modal_bias[modal_idx]
   ```
   modal_bias ∈ R^(M×E)，M=3 个模态，E=8 个 expert。这是一个可学习参数，为每个模态提供对 expert 的先验偏好。
4. **Top-K 选择**:
   ```
   top_indices = topk(logits, k=3)
   gates = softmax(logits[top_indices])
   ```
   只有被选中的 top-3 expert 获得非零 gate 权重，其余为 0。

### 数据来源

**Fig 4b** 直接读取模型权重，不需要输入数据。

**Fig 4a/4c/4d** 使用**测试集的真实图像**。具体流程：
- 从 `test/images/` 目录按文件名匹配三个模态（sar/rgb/gf）
- 每个模态取 50 张图像（可通过 `--num-samples` 调整）
- 通过 pipeline 加载：LoadMultiModalImageFromFile → Resize(256×256) → MultiModalNormalize → Pack
- 逐张送入模型做 forward pass
- 通过 forward hook 在每个 MoE 层的 gating 模块上截获路由输出
- Hook 中重新计算 clean gate（eval 模式，无噪声），记录每个样本在每个 MoE 层上的 gate 向量

---

## Fig 4a: Expert Activation Probability Heatmap

### 含义
展示每个模态在每个 stage 中激活各 expert 的**概率**。

### 制作方法
1. 对每个模态，收集该模态 50 张测试图在所有 MoE 层上的 gate 向量（shape: [N, 8]，N=50×该stage的MoE block数）
2. 按 stage 分组聚合
3. 计算激活概率：`P(expert_j | modality_m, stage_s) = mean(gate_j > 0)`
   即 expert j 被选中（gate 权重 > 0）的样本占比
4. 得到矩阵 [3 modals × 8 experts]，每个 stage 一张热力图

### 图的结构
- **3 张子图**（Stage 1, Stage 2, Stage 3），每张是一个 3×8 的热力图
- **纵轴**：三个数据集/模态（仅最左图显示标签）
- **横轴**：Expert E0–E7
- **颜色**：YlOrRd 色系，0（从不激活）到 1.0（始终激活）
- **标注值**：每个格子内的数字是激活概率

### 如何读图
- **值接近 1.0（深红）**：该 expert 几乎总是被该模态选中 → 该 expert 是该模态的"专属 expert"
- **值接近 0.0（浅黄/白）**：该 expert 几乎不被该模态使用
- **三个模态同一 expert 都高**：该 expert 是"共享 expert"，处理跨模态通用特征
- **某模态独占某 expert**：模型学到了模态特异性的特征提取路径

### 论文中的解读角度
- 观察是否存在 expert 专业化（某些 expert 只服务特定模态）
- 对比不同 stage：浅层 stage 是否路由更均匀，深层是否更专业化
- 与 Table 2 的 "w/o MoE" 消融对照：MoE 通过这种专业化分工带来的性能提升

---

## Fig 4b: Learned Modal Bias Matrix

### 含义
直接展示模型训练后学到的 **modal_bias** 参数值。这是 gating 函数中加在 logits 上的模态先验偏好。

### 制作方法
1. 遍历模型 backbone 所有 stage 的所有 block
2. 找到使用 MoE 的 block，读取 `block.mlp.gating.modal_bias` 参数（nn.Parameter）
3. 按 stage 分组，同一 stage 内多个 MoE block 的 bias 取**平均**
4. 得到 [3 modals × 8 experts] 矩阵，每个 stage 一张

### 图的结构
- **3 张子图**（Stage 1, Stage 2, Stage 3）
- **纵轴**：三个模态（仅最左图显示标签）
- **横轴**：Expert E0–E7
- **颜色**：RdBu_r 发散色系，以 0 为中心（蓝=负偏置，白=零，红=正偏置）
- **标注值**：每个格子内的实际 bias 值

### 如何读图
- **正值（红色）**：该模态被"推向"使用该 expert（先验偏好）
- **负值（蓝色）**：该模态被"推离"该 expert（先验排斥）
- **接近 0（白色）**：无先验偏好，路由完全由输入特征决定
- 对比同一行（同一模态跨不同 expert）：可以看出模型为该模态学到的 expert 偏好谱
- 对比同一列（同一 expert 跨不同模态）：可以看出该 expert 对不同模态的吸引/排斥

### 与 Fig 4a 的关系
- Fig 4b 展示的是**静态的权重**（不依赖输入）
- Fig 4a 展示的是**动态的激活结果**（modal_bias + 输入特征共同决定）
- 如果 4a 的模式和 4b 高度一致 → modal_bias 主导路由决策
- 如果 4a 和 4b 有差异 → 输入特征对路由也有显著影响
- 论文应讨论二者的一致性和差异

### 与 Table 2 消融实验的关系
- Table 2 的 "w/o Modal Bias" 实验将 `use_modal_bias=False`，即 gating 中不加这个偏置
- 性能下降量反映了 modal_bias 对路由质量的贡献

---

## Fig 4c: Expert Selection Frequency (Grouped Bar Chart)

### 含义
展示三个模态**在所有 MoE 层上聚合**后的 expert 选择频率对比。是 Fig 4a 的简化汇总视图。

### 制作方法
1. 对每个模态，将所有 stage、所有 block 的 gate 向量拼接成一个大矩阵 [N_total, 8]
2. 计算 `freq[modal][expert_j] = mean(gate_j > 0)`
3. 三个模态并排画分组柱状图

### 图的结构
- **单张图**，横轴为 Expert E0–E7
- **三组柱子**：蓝色 = UrbanSARFlood, 绿色 = FloodNet, 橙色 = GF-FloodNet
- **纵轴**：Selection Frequency (0–1.0)

### 如何读图
- **三个模态柱子高度相近的 expert**：跨模态共享 expert
- **某个模态明显突出的 expert**：该模态的专属 expert
- **top_k=3 意味着每个样本一定选 3 个 expert**，所以所有 expert 的频率之和 ≈ 3/8×8 = 3.0（即平均每个 expert 被选中 37.5% 的时间）。如果某 expert 远超 37.5%，说明它是高频 expert
- 可以直接在论文中引用此图说明："Expert E2 is predominantly selected by SAR data (frequency=0.85), while Expert E5 serves as a shared expert across all modalities"

### 与 Table 2 的关系
- "w/o Shared Experts" 消融移除了共享 expert，此图可以定性说明哪些 expert 原本承担了共享角色

---

## Fig 4d: Gate Weight Distribution (Box Plot)

### 含义
展示每个模态的 gate 权重**分布**——不仅看是否被选中，还看被选中时分配了多大的权重。

### 制作方法
1. 对每个模态，拼接所有 gate 向量
2. 对每个 expert，提取**非零**的 gate 权重值
3. 画箱线图（box plot），展示中位数、四分位距和分布范围

### 图的结构
- **3 张子图**，分别对应三个模态
- 每张图横轴为 Expert E0–E7
- 纵轴为 Gate Weight (non-zero)
- 箱线图颜色对应模态颜色
- 不显示离群点（`showfliers=False`）

### 如何读图
- **箱体位置高（中位数高）**：该 expert 不仅被选中，而且获得了较大的权重 → 模型高度信任该 expert 处理该模态
- **箱体窄**：权重分配稳定一致
- **箱体宽 / 跨度大**：权重分配有较大变异，说明不同样本对该 expert 的依赖程度不同
- **箱体位置低但存在**：该 expert 偶尔被选中，但权重较小

### 与 Fig 4c 的互补关系
- Fig 4c 只看"是否被选中"（二值化：>0 or not）
- Fig 4d 看"被选中后分配了多少权重"（连续值）
- 可能出现的情况：某 expert 被频繁选中（4c 高），但每次分到的权重很小（4d 中位数低）→ 该 expert 被当作"补充"角色而非主力

---

## 四图综合叙述建议

论文中建议按以下逻辑串联四张图：

1. **Fig 4b** → 首先展示模型学到了什么样的模态偏好先验（静态权重分析）
2. **Fig 4a** → 然后展示在真实测试数据上的激活模式（动态行为分析），与 4b 对比讨论
3. **Fig 4c** → 汇总全局视角：哪些 expert 被哪些模态偏好，是否存在专业化
4. **Fig 4d** → 深入分析：被选中的 expert 实际获得了多大权重，路由是否 confident

### 关键论述点
- **Expert 专业化**：某些 expert 专注于特定模态（专属 expert），某些跨模态共享（shared expert）
- **跨 stage 变化**：浅层 stage 路由可能更均匀，深层 stage 专业化更强（语义级别的特征分化）
- **Modal bias 的作用**：对比 4a 和 4b 说明先验引导 + 数据驱动的协同效应
- **与消融实验呼应**：4c 可以定性解释为什么移除 shared expert (Table 2) 或 modal bias (Table 2) 会导致性能下降

---

## 生成命令

```bash
python tools/analysis_tools/visualize_expert_routing.py \
    configs/floodnet/multimodal_floodnet_sar_boost_swinbase_moe_config.py \
    work_dirs/floodnet/SwinmoeB/655/best_mIoU_epoch_100.pth \
    --data-root ../floodnet/data/mixed_dataset/ \
    --output-dir work_dirs/figures/expert_routing \
    --num-samples 50
```

### 输出文件
- `fig4a_expert_activation_heatmap.pdf/png`
- `fig4b_modal_bias_matrix.pdf/png`
- `fig4c_expert_selection_frequency.pdf/png`
- `fig4d_gate_weight_distribution.pdf/png`

### 图片规格
- 300 DPI
- Times New Roman 字体
- PDF + PNG 双格式
- `savefig.bbox='tight'` 自动裁剪白边
