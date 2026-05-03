# CS336 Lecture 11 · Scaling Laws (二)：案例研究与细节

> 本笔记对应 Stanford CS336 (Spring 2025) Lecture 11 "Scaling — Case Study and Details"。
> 目标：对照 PDF 讲义阅读本笔记，即可无损替代视频学习。

---

## 0. 本节课的定位与动机

上一节课（Lecture 10）从理论角度讲了 scaling laws 的来龙去脉：Kaplan 2020、Chinchilla 2022，以及 µP 的初步介绍。但学完之后，一个合理的怀疑是：

> "在对数–对数坐标系里拟合一条直线，真的就那么可靠吗？"

所以本节课的主线有两条：

1. **案例研究（Case Study）**：看一看真正在训练大模型的团队（Cerebras、MiniCPM、DeepSeek、Llama 3、Hunyuan、MiniMax-01）如何把 scaling laws 用进自己的训练流程。
2. **µP 深入剖析（Maximum Update Parametrization）**：推导 µP 的数学原理，看它能不能让"最优超参跨规模保持不变"，并用最新的独立验证论文做实证检验。

贯穿全课的核心关切只有一个：

> **最佳实践：如何以可预测、低成本的方式把模型规模放大。**

更具体地，落到工程上就是三件事：

- 设置模型结构超参（宽度、深度、纵横比 …）
- 设置优化器超参（学习率、batch size …）
- 用尽量少的算力把上面这些东西"钉死"

---

## 1. 时代背景：ChatGPT 之后，scaling 变成了黑箱

- Chinchilla（2022, DeepMind）是最后一篇真正把 scaling 细节讲透的论文。
- ChatGPT 出来之后，前沿实验室基本停止公开 scaling 细节。
- 所以我们只能去看那些**做得不错且愿意分享**的"半开源"模型。
- 去年（2024）老师挑了三篇：**Cerebras-GPT、MiniCPM、DeepSeek LLM**。这是三个黄金样本。
- 今年（2025）新增了几篇较简略的：**Llama 3、Hunyuan Large、MiniMax-01**。

> **记住这句话**：Cerebras-GPT / MiniCPM / DeepSeek 依然是目前最详尽的公开 scaling study。

---

## 2. Cerebras-GPT：把 µP 跑通的第一个公开验证

### 2.1 基本信息
- 模型规模：0.1B — 13B，一个家族的模型。
- 训练配方：**Chinchilla 风格**（token 数与参数量大致按 20:1）。
- 核心贡献：**公开验证了 µP (Maximum Update Parametrization) 让 scaling 更稳定**。

### 2.2 punch line：µP vs. 标准参数化

他们在 The Pile 上画 test loss 随模型规模的 scaling 曲线：

- 蓝色：标准参数化（Standard Parametrization, SP）的 Cerebras-GPT
- 橙色：µP 参数化的 Cerebras-GPT

结果：

- 标准参数化的点围着拟合直线**上下剧烈震荡**——因为每个规模都必须重新调 LR，一旦调得不完美，损失就偏离预测。
- µP 版本的点**紧贴**拟合直线——说明同一个 LR 可以跨规模用。

这张图是 µP 在大模型上的**第一批公开验证**之一。顺带一提，据说 Llama 4 也在用一个叫 "MetaP" 的 µP 变体。

### 2.3 µP 的"一句话总结"（实现层面）

在 Cerebras-GPT 论文附录里，他们给了一张对照表，告诉你 SP 和 µP 的每一层分别怎么初始化、怎么设学习率。一句话版：

> **非 embedding 的每一层，初始化方差按 `1/width` 缩放，per-layer 学习率也按 `1/width` 缩放。**

跟标准参数化相比，**初始化差别不大**（Kaiming 初始化本身就是 `1/sqrt(fan_in)`），真正的新东西是 **per-layer 学习率**——这是多数人以前没接触过的。

### 2.4 Cerebras 的超参搜索策略

- 把代理模型一路缩到 **40M 参数**。
- 在这个小模型上做大规模超参网格搜索（每个点一个 run）。
- 选出最小损失的超参。
- 然后**用 µP 原封不动地把它们搬到大模型**。

这是非常干净的做法，但能不能真的推广到超大规模还存疑。后面你会看到 MiniCPM 和 DeepSeek 都走类似思路：**小模型过度搜索 + 稳定外推**。

---

## 3. MiniCPM：把数据规模推到极致的小模型

### 3.1 基本信息
- 清华/面壁智能团队 2024 年发布。
- 目标：**小模型 + 大算力 = 极高性能**。
- 1.2B — 2.4B / 4B 参数，但当时性能可对标很多 7B 模型。
- 也用 µP 稳定 scaling，同时首次把 **WSD 学习率调度** 在 LM 训练里推火。

### 3.2 超参缩放配方

参考他们的 µP 配置：

- `scale_emb = 12`
- `scale_depth = 1.4`
- `init_std = 0.1`
- `lr = 0.01`

对比 Cerebras-GPT：`scale_emb = 10, lr = 6e-3, init_base = 0.08` —— **两者几乎落在同一个量级**。

具体规则（µP 的经验形式）：

- Embedding：乘一个常数即可。
- Residual 连接处（比如 MLP 输出）：乘 $1/\sqrt{\text{num\_layers}}$。
- 矩阵初始化：按 fan-in 的 $1/\sqrt{\text{width}}$。
- 学习率：按 $1/\text{width}$ 缩放。

### 3.3 整体 scaling 流程

1. 固定**纵横比**（aspect ratio = $d_\text{ff}/d_\text{model}$ 等），因为它对损失很敏感但并不难定。
2. 依赖 µP，假设**最优 LR 跨规模不变**。
3. 在 9M / 30M / 170M 上扫 batch、LR、token-to-param ratio。
4. 直接外推到 0.5B—1B 量级，大约 5× 的外推倍率。

### 3.4 最优 batch size 的 scaling

这是 Chinchilla / Kaplan 的经典分析的复刻。

- **临界 batch size (critical batch size)** 的直觉：模型 loss 越低，能"并行吞"的 batch 就越大（收益递减点越高）。
- 模型越大 → loss 越低 → critical batch 越大。
- 经验规律（log-log 线性）：

$$
B_{\mathrm{crit}} \propto L^{-\alpha}
$$

即 **loss 越低，batch size 多项式级变大**。

具体做法：

1. 固定模型尺寸、固定 batch，把 training curve 画出来（x = batch，y = data）。
2. 对每个目标 data size，纵列拟合一条抛物线，取最低点，这就是该 (模型大小, 数据量) 下的"最优 batch"。
3. 得到 (terminal loss, optimal batch) 后，在 log-log 图上画一条线——这就是 Kaplan 式的 critical batch scaling。
4. 用 scaling law 预测大模型的目标 loss → 反推大模型的 batch size。

### 3.5 最优学习率真的稳定吗？

他们画的图很有意思：横轴 LR，纵轴 loss，不同颜色是不同模型大小（淡 → 小，深 → 大）。大模型因为算力贵只跑了很短。结果：

> **所有模型的 loss 最低点都落在大约 $10^{-2}$ 这个 LR 附近，跨越数个数量级。**

这是 µP 工作的**强力证据**——说明"只要初始化和 per-layer LR 按 µP 规则来，全局 LR 就可以不随模型放大而改"。

### 3.6 核心创新：WSD 学习率

**问题背景**：Chinchilla 式的数据–模型 trade-off 分析需要 $n \times n$ 的 runs——每个 (参数量, token 数) 组合都要**从头训练**。
为什么不能"训一次大 run 然后取早期 checkpoint"？因为 **cosine LR 对应的不同终止点 → 不同的学习率曲线**，早期 checkpoint 的 LR 和"目标小数据量的训练 run"的 LR 根本不一样。这是**非常常见的陷阱**。

**解决方案：WSD (Warmup - Stable - Decay)**，又名 trapezoid LR。

- **Warmup**：和 cosine 的 warmup 一样。
- **Stable**：一个又长又平的 LR 不变阶段（cosine 没有）。
- **Decay**：最后一段快速冷却（通常 ~10% 总 step），把 LR 退火到 0 或最小值。

为什么好用？

> **Stable 阶段是平的，可以反复复用。**
> 训练到 stable 阶段的不同位置后，分别"分叉"做一次 decay，就等价于得到了不同 token 数量下的完整训练结果。
> 这样 **Chinchilla 分析的代价从 $O(n^2)$ 降到约 $O(n)$**（只需一次大 run + 若干次短 decay）。

训练曲线外观：

- Cosine：loss 平滑缓慢下降到终点。
- WSD：stable 阶段平平下降，**进入 decay 后 loss 陡崖式骤降**——看起来吓人，但完全正常。

经验规律：WSD 和 cosine 的 final loss 基本持平，有时前者更好有时后者，但 WSD 在"可复用"这一点上全面胜出。

### 3.7 补充：另一种 Chinchilla 估计法（Gadre et al.）

另一条思路：不做 WSD，而是对"overtraining 惩罚"建模。把不同的 tokens/param 比（20, 320, 640 …）对应多条近平行的 scaling 线，然后**在小规模下外推"超训惩罚"**。据我所知尚未被大规模使用，但思想很漂亮。

### 3.8 Chinchilla 分析结果

MiniCPM 最终给出：

- **Method 1**（lower envelope）：把所有训练曲线的下包络当作 power law 拟合。
- **Method 3**（joint fit）：直接拟合两变量 scaling law：

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

他们主要用 Method 3，结论：**最优 token-to-param ratio ≈ 192**。

这个数远高于 Chinchilla 的 20，略显离群。但和 Llama 3（~39）、Hunyuan（~96）的方向一致——

> **Chinchilla 的 "20×模型大小" 只是经验起点，并不是硬约束。随着架构与数据质量改进，这个比值会持续变大。**

---

## 4. DeepSeek LLM（v1, 2024）：不信 µP，直接硬拟合

### 4.1 基本信息
- 7B / 67B 两个版本。
- 当时性能对标 Llama 2 / Mistral。
- 这是 DeepSeek V3 爆红之前的第一篇"正经 scaling"论文。
- **风格**：老老实实做实验，每个超参都仔细跑。

### 4.2 与 µP 的分岔口

DeepSeek **不用 µP**，而是**直接对最优 batch 与最优 LR 分别拟合 scaling law**。

- 前提假设：batch 和 LR 的**最优区是很宽的 basin**，所以不需要精确，只要大致"数量级对"即可。
- 在几个不同 flop scale 下，扫 (batch × LR) 网格。
- 用 "within 0.25% of min" 的点当作"近最优"集合。
- 把最优 batch 和最优 LR **分别**作为 compute 的函数，在 log-log 图上拟合直线。

老师自嘲：LR 那条拟合"**画一条水平线也大致合理**"，但他们就这么用了。这说明**LR 的 scaling law 拟合其实相当不稳**。

### 4.3 WSD 变体

DeepSeek 用的是 **Warmup + Stable + 两段各 10% 的 Decay**（共 20% 算力）。效果和 cosine 差不多，但保留了 WSD 的可复用优势。

### 4.4 Chinchilla Method 2（IsoFLOP）

他们用的是 Chinchilla 原文的 **Method 2（IsoFLOP）**：

- 固定 FLOPs 预算，扫不同模型大小。
- 每个预算下拟合一条抛物线，取最低点。
- 把 (compute, optimal N) 和 (compute, optimal D) 在 log-log 画成直线。

这是所有 scaling 分析里**最稳最干净**的一环。老师说：

> **超参的 scaling 图总是抖的，但 IsoFLOP 图在所有论文里都非常漂亮。**

### 4.5 最终验证

DeepSeek 从 $10^{20}$ FLOPs 外推到 $10^{24}$ FLOPs 做 7B / 67B 的预测，**预测 loss 与实际 loss 几乎重合**。这是 scaling law 可预测性的教科书式展示。

---

## 5. 近期（2024+）其他模型的 scaling 片段

### 5.1 Llama 3
- IsoFLOP-style 分析，得到最优 **tokens/param ≈ 39**。
- 额外工作：把 NLL 映射到下游任务准确率——拟合 **sigmoid 曲线**：
  - 小模型 + Llama 2 → 拟 sigmoid
  - 用它预测 Llama 3-405B 的下游表现
- 用途：指导数据选择。但细节讲得很少。

### 5.2 Hunyuan-1 Large
- 还是 IsoFLOP。
- 但做的是 **MoE**，所以最优比是 **"data : active param ≈ 96"**。
- 再次说明 20:1 不是硬规律。

### 5.3 MiniMax-01
- 线性/混合 attention（Lightning Attention）。
- 用 **Chinchilla Method 1（lower envelope）做架构选择**：
  - Softmax attention vs. Linear attention vs. Hybrid 的 scaling 曲线对照
  - 发现 hybrid 和全 softmax 在 compute 层面几乎重合 → 证明 hybrid 不损性能
- 这种 "把 scaling law 当架构决策工具" 的用法在 Mamba / Mamba 2 / Delta 等论文里也常见，但 MiniMax 是首次做到接近生产规模。

### 5.4 汇总对比表

| 模型 | µP? | Chinchilla 方法 | LR / batch 选法 | 特色 |
| --- | --- | --- | --- | --- |
| **Cerebras-GPT** | ✅ | 直接套公式 | 小模型搜 → µP 外推 | µP 的第一个公开大规模验证 |
| **MiniCPM** | ✅ | Method 1 + 3 | µP 保持不变 | 推广 WSD 学习率 |
| **DeepSeek v1** | ❌ | Method 2 (IsoFLOP) | 直接拟 scaling law | WSD 变体（两段 decay） |
| **Llama 3** | ❌ | IsoFLOP | 未披露 | NLL → 下游 accuracy 的 sigmoid 映射 |
| **Hunyuan** | ❌ | IsoFLOP（MoE） | 未披露 | MoE 上的 96:1 |
| **MiniMax-01** | — | Method 1 | — | 用 scaling 做架构决策 |

**共同主题**：

- 大家都做 Chinchilla 复制——而且发现 **最优 tokens/param 普遍大于 20**。
- 大家都很怕 LR 和 batch size 选错——要么用 µP 把它们钉死，要么用 scaling law 外推。
- **固定 aspect ratio + 只放大总规模** 是大家事实上的标准动作。

---

## 6. µP 的深度推导

µP 的卖点：**让最优超参对模型宽度不变（scale-invariant）**。如果真的做到，小规模网格搜索 → 大规模零成本复用。这节我们一步步推导它到底在控制什么。

### 6.1 µP 的两个核心断言（assertions）

设模型宽度为 $n_l$（第 $l$ 层的输出维度），我们希望随着 $n_l \to \infty$：

- **A1**：初始化时，每个激活坐标应是 $\Theta(1)$。
- **A2**：做完**一步**梯度更新后，每个激活坐标的**变化量**也应是 $\Theta(1)$。

"每个坐标都是 $\Theta(1)$" 等价于"整个激活向量的 L2 范数是 $\Theta(\sqrt{n_l})$"（因为不同坐标近似独立，范数按维度根号增长）。

**直觉**：如果 A1 不成立，放大模型时激活就会爆炸或消失；如果 A2 不成立，一次梯度更新后激活又会爆炸或消失。这两条是最朴素的"放大稳定性"需求。

### 6.2 从 A1 推导初始化规则

**设定**：考虑一个**深度线性网络**（纯矩阵乘法，没有非线性）：

$$
h_l = W_l h_{l-1}
$$

初始化用各向同性高斯：

$$
W_l \sim \mathcal{N}\!\left(0, \sigma_l^2 \, I_{n_l \times n_{l-1}}\right)
$$

这里 $\sigma_l$ 是待定的噪声尺度。

**工具**：随机矩阵理论告诉我们，高斯矩阵的**算子范数**会集中到：

$$
\|W_l\|_* \;\to\; \sigma_l \left(\sqrt{n_l} + \sqrt{n_{l-1}}\right)
$$

（这里 $\|\cdot\|_*$ 表示算子 / 谱范数。）

又因为 $W_l$ 与 $h_{l-1}$ 在初始化时独立，所以：

$$
\|h_l\|_2 \;\approx\; \|W_l\|_* \cdot \|h_{l-1}\|_2
$$

**关键选择**：取

$$
\sigma_l = \sqrt{\frac{n_l}{n_{l-1}}}\cdot \left(\sqrt{n_l} + \sqrt{n_{l-1}}\right)^{-1} = \Theta\!\left(\sqrt{\frac{1}{n_{l-1}}} \cdot \min\!\left(1, \sqrt{\frac{n_l}{n_{l-1}}}\right)\right)
$$

直观上就是 "$1/\sqrt{\text{fan\_in}}$" 再乘一个在 fan-in 远大于 fan-out 时起作用的修正。

**归纳法证明 A1**：

- **归纳假设**：$\|h_{l-1}\|_2 = \Theta(\sqrt{n_{l-1}})$。
- **归纳推进**：

$$
\|W_l\|_* \;\to\; \sigma_l\left(\sqrt{n_l}+\sqrt{n_{l-1}}\right) = \sqrt{\frac{n_l}{n_{l-1}}}
$$

代入：

$$
\|h_l\|_2 \;\approx\; \sqrt{\frac{n_l}{n_{l-1}}}\cdot\sqrt{n_{l-1}} \;=\; \sqrt{n_l} + o(\sqrt{n_l})
$$

正是我们想要的 $\Theta(\sqrt{n_l})$。

> **一句话结论**：µP 的初始化就是 "$1/\sqrt{\text{fan\_in}}$"（Kaiming 那一套）再加一个 fan-in/fan-out 比值修正项。

### 6.3 从 A2 推导学习率规则

**设定**：考虑 SGD 的一步更新，在线性层上：

$$
\Delta W_l = -\eta_l \, \nabla_{h_l} \ell \cdot h_{l-1}^\top
$$

这是一个**秩一 (rank-one) 外积**。秩一矩阵的算子范数等于它两个向量的范数之积，所以：

$$
\|\Delta W_l \, h_{l-1}\|_2 = \|\Delta W_l\|_* \cdot \|h_{l-1}\|_2
$$

**展开一步后的激活变化**。把 "$h_{l-1} \to h_{l-1} + \Delta h_{l-1}$" 和 "$W_l \to W_l + \Delta W_l$" 代进去，写出：

$$
\Delta h_l = W_l \Delta h_{l-1} + \Delta W_l \left(h_{l-1} + \Delta h_{l-1}\right)
$$

我们希望 $\|\Delta h_l\|_2 = \Theta(\sqrt{n_l})$。分析右边三项的量级（假设领项不抵消）：

1. **$W_l \Delta h_{l-1}$**：由归纳假设（A2 对 $l-1$ 层）+ A1（$W_l$ 的算子范数），这一项是 $\Theta(\sqrt{n_l})$。
2. **$\Delta W_l \cdot h_{l-1}$**：等于 $\|\Delta W_l\|_* \cdot \sqrt{n_{l-1}}$。要它是 $\Theta(\sqrt{n_l})$，必须：

$$
\|\Delta W_l\|_* = \Theta\!\left(\sqrt{\frac{n_l}{n_{l-1}}}\right)
$$

3. **$\Delta W_l \Delta h_{l-1}$**：是 $O(\|\Delta W_l\|_* \sqrt{n_{l-1}})$ = 和第 2 项同阶，OK。

所以**整件事归结为**：如何选 $\eta_l$ 使 $\|\Delta W_l\|_* = \Theta(\sqrt{n_l / n_{l-1}})$ ？

**第三条辅助假设（A3）**：一步更新对 loss 的改变量 $\Delta \ell$ 也应是 $\Theta(1)$（不随宽度爆炸或消失）。用一阶泰勒：

$$
\Delta \ell = \Theta\!\left(\langle \Delta W_l,\; \nabla_{W_l} \ell\rangle\right) = \Theta\!\left(\|\Delta W_l\|_F \cdot \|\nabla_{W_l}\ell\|_F\right)
$$

在 rank-one 的情况下，Frobenius 范数和算子范数同阶，所以：

$$
\Delta \ell = \Theta\!\left(\|\Delta W_l\|_* \cdot \|\nabla_{W_l}\ell\|_*\right)
$$

代入 $\Delta \ell = \Theta(1)$ 和 $\|\Delta W_l\|_* = \Theta(\sqrt{n_l/n_{l-1}})$，解出：

$$
\|\nabla_{W_l}\ell\|_* = \Theta\!\left(\sqrt{\frac{n_{l-1}}{n_l}}\right)
$$

**再结合 SGD 更新式** $\Delta W_l = -\eta_l \nabla_{h_l}\ell \cdot h_{l-1}^\top$，其算子范数大概是 $\eta_l \cdot \|\nabla_{h_l}\ell\|_2 \cdot \|h_{l-1}\|_2$。经一番代入与约去（详见 PDF 第 40–41 页），得到**SGD 下的 µP 学习率**：

$$
\eta_l^{\mathrm{SGD}} = \Theta\!\left(\frac{n_l}{n_{l-1}}\right) = \Theta\!\left(\frac{\text{fan\_out}}{\text{fan\_in}}\right)
$$

**Adam 的情形略有不同**：因为 Adam 对梯度做了按元素归一化，等价于把梯度的 "单位" 换成 "每个坐标都是 $\Theta(1)$" 的形式，因此可证 $\|\Delta W_l\|_* \cdot \sqrt{n_{l-1}} = \Theta(1)$ 的条件直接转化为：

$$
\eta_l^{\mathrm{Adam}} = \Theta\!\left(\frac{1}{n_{l-1}}\right) = \Theta\!\left(\frac{1}{\text{fan\_in}}\right)
$$

### 6.4 µP 总结对照表

**µP（本课推导结果）**

$$
\sigma_l = \Theta\!\left(\sqrt{\frac{1}{n_{l-1}}}\cdot\min\!\left(1,\sqrt{\frac{n_l}{n_{l-1}}}\right)\right), \quad
\eta_l = \begin{cases}
\Theta\!\left(\dfrac{n_l}{n_{l-1}}\right) & \text{SGD} \\[2mm]
\Theta\!\left(\dfrac{1}{n_{l-1}}\right) & \text{Adam}
\end{cases}
$$

**标准参数化（SP）**

$$
\sigma_l = \Theta\!\left(\sqrt{\frac{1}{n_{l-1}}}\right), \quad \eta_l = \Theta(1) \ \text{（全局常数）}
$$

**关键对比**：

- **初始化**：SP 跟 µP **几乎一样**（都在用 Kaiming 式 $1/\sqrt{\text{fan\_in}}$）。只有在 fan-out < fan-in 时，µP 多出一个修正项。
- **学习率**：这是**真正的差别**。SP 用全局常数；µP 为**每一层**单独设置一个随宽度缩放的 LR。
- **Adam 下**：差别最大——SP 是常数，µP 按 $1/\text{fan\_in}$ 缩小。这正是大多数工程实现里 "看起来的 µP" 的来源。

### 6.5 回到 Cerebras-GPT 的实现表

有了推导，我们现在能完全看懂 Cerebras-GPT 附录的 µP 表：

- Embedding 层**特殊处理**：输入是 one-hot，范数不随词表规模变，所以不做宽度缩放。
- 所有其他线性层：初始化缩放 $\sim 1/\sqrt{\text{width}}$（加修正项），LR 缩放 $\sim 1/\text{width}$（因为用的是 AdamW）。

这就解释了为什么"µP 实现起来就是两行代码"：一行改初始化，一行改 per-layer LR。

> **一句概念性总结**：µP 做的事情本质上是"深度学习版的**重整化 (renormalization)**"——像物理里取极限时保持可观测量稳定一样，控制 $W$ 和 $\Delta W$ 使激活的均值、方差在宽度 $\to \infty$ 下都保持 $\Theta(1)$。

### 6.6 推导的边界条件（老师现场答疑）

- **假设了深度线性网络**。加入非线性、attention、GLU 都需要额外分析；有些要更复杂的改造（比如 gated linear units）。
- **$n_l$ / $n_{l-1}$ 就是该层的 fan-out / fan-in**。对 MLP，一般 $d_\text{ff} = 4 d_\text{model}$，所以比值是常数 4——那"µP vs. SP" 的差别几乎只体现在 LR 上，而不是初始化上。
- **DeepSeek 用全局 LR 不做 µP 是否违反 A2？** 是的，从渐近视角看 DeepSeek 的激活更新会随宽度"越来越大"。所以他们才**主动把全局 LR 随规模往下调**——相当于手动补偿 µP 该做的事。换句话说：**µP 不是唯一出路，但不用 µP 就得手动调 LR**。

---

## 7. µP 的实证验证：Everett 2024 preprint

老师后半段讲了一篇 COLM 2024 的论文，做了大量 ablation，系统回答：µP 到底**在什么条件下**工作？

### 7.1 实验协议

- 只缩放**宽度**（保持深度固定）——这是可控实验，和实际放大时"深度 + 宽度齐缩"略有不同。
- 用 µP 原始形式：方差按 $1/m$、per-layer LR 按 $1/m$ 缩放（$m$ 是宽度倍率）。
- 特别注意：所有 µP 实现**把 attention 内积的缩放改成 $1/d$**（不是标准的 $1/\sqrt{d}$）——这是为了控制激活和更新大小的稳定性。

### 7.2 核心问题：LR 是否跨宽度不变？

- 宽度：128 / 512 / 2048
- 扫多个 LR
- 结果：**最优 LR 在不同宽度下稳定落在同一个点**——µP 做到了它承诺的事。

### 7.3 µP 对哪些现代组件鲁棒？

✅ **鲁棒的**：

| 变动 | 结论 |
| --- | --- |
| Nonlinearity（SwiGLU、Squared ReLU vs. ReLU） | 最优 LR 不变；SwiGLU / Squared ReLU 还稍好 |
| Batch size 上下波动 4× | 最优 LR 不变 |
| Zero Query 初始化 | 最优 LR 不变 |
| Unembedding 用 SP 还是 µP（$1/m$ vs. $1/m^2$） | 最优 LR 不变 |

❌ **不鲁棒的（µP 会失败）**：

| 变动 | 结论 |
| --- | --- |
| RMSNorm **可学习 gain** | 破坏 µP。好消息：去掉 gain 对性能影响很小，所以可以直接去掉 |
| 非标准优化器（如 **Lion**，基于 gradient sign） | 完全失效。预期之中，因为 µP 是为 AdamW 推的，换优化器就得重推 |
| 强 weight decay（≥ 0.1） | **唯一真正值得注意的失败**——因为 weight decay 在实际训练中很常用 |

### 7.4 最终大规模验证

他们把小规模下选出的最优 LR（$2^{-6}$ 附近）**直接外推到 10B 参数模型**，loss 依然在最优点附近——这是目前 µP 最干净的大规模证据之一。

### 7.5 结论

- µP **确实让 LR 跨宽度更稳定**；SP 在同条件下会灾难性崩盘。
- µP 在 SwiGLU / 多种初始化 / batch size 变化下都能撑住。
- 但 RMSNorm gain、非主流优化器、强 weight decay 会打破它——用的时候要注意。
- **业界现状**：Meta 在 Llama 4 里用的是 MetaP（µP 变体），但"大家都用 µP" 远远谈不上共识。

---

## 8. 全局回顾：scaling in the wild

### 8.1 三大实际挑战

1. **模型结构超参**（宽度、深度、aspect ratio）
2. **优化器超参**（LR、batch size）
3. **Chinchilla 全扫的算力代价**

### 8.2 三类解决方案

1. **假设稳定性**——直接声称某些超参跨规模不变（aspect ratio 常用），或**用 µP 把 LR / 初始化钉死**。
2. **小规模搜 + 外推**——DeepSeek 式 scaling law 拟合，或 Cerebras 式"狠狠缩小代理 + µP 搬运"。
3. **替代 LR 调度**——WSD 把 Chinchilla 代价从 $O(n^2)$ 降到 $O(n)$。

### 8.3 记住这几条

- **Chinchilla 的 20:1 不是硬规律**——随着算法、数据、架构进步，最优 tokens/param 会越来越大（Llama 3 ~39，Hunyuan ~96，MiniCPM ~192）。
- **IsoFLOP 分析在所有团队手里都很稳**，而 LR/batch 的 scaling law 往往噪声大，需要提防。
- **WSD 是过去两年最重要的工程技巧之一**，几乎已经取代 cosine 成为 scaling 研究的事实默认。
- **µP 的价值不是"必须"，而是"稳"**——没它也能训好，但有它能让 LR 搜索不再随模型规模爆炸。

---

## 附录：关键公式速查

**µP 初始化（Gaussian $\sigma_l$）：**

$$
\sigma_l = \Theta\!\left(\sqrt{\frac{1}{n_{l-1}}} \cdot \min\!\left(1, \sqrt{\frac{n_l}{n_{l-1}}}\right)\right)
$$

**µP 学习率：**

$$
\eta_l^{\mathrm{SGD}} = \Theta\!\left(\frac{n_l}{n_{l-1}}\right), \qquad \eta_l^{\mathrm{Adam}} = \Theta\!\left(\frac{1}{n_{l-1}}\right)
$$

**Chinchilla 两变量 scaling law（Method 3 拟合的对象）：**

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

**Kaplan 式 critical batch size 与 loss 的关系：**

$$
B_{\mathrm{crit}} \propto L^{-\alpha}
$$

**一步激活更新的分解（µP 推导核心）：**

$$
\Delta h_l = W_l \, \Delta h_{l-1} + \Delta W_l \, \left(h_{l-1} + \Delta h_{l-1}\right)
$$

三项都需要是 $\Theta(\sqrt{n_l})$ —— 由此逐步推出 µP 的初始化与 LR 规则。
