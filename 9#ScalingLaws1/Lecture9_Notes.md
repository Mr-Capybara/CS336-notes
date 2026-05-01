# CS336 Lecture 9: Scaling Laws — Basics 详细笔记

> 本讲主题：**为什么可以通过训一堆小模型，外推出大模型的最佳配置**。
>
> 全篇配合讲义 `2025 Lecture 9 - Scaling laws basics.pdf`（共 53 页）阅读，每节都标注对应 slide 页码。主线如下：
>
> 1. **背景 & 历史**（Slide 1–11）：把"scaling laws"放回统计学习与 1993 年 Bell Labs 的起点。
> 2. **数据 Scaling Law**（Slide 12–25）：从均值估计、非参数回归两个 toy example，理解"为什么误差对数-对数线性下降"。
> 3. **模型工程 Scaling Law**（Slide 26–39）：架构、优化器、深宽比、Batch size、Learning rate (µP)。
> 4. **联合 Data–Model Scaling + Chinchilla**（Slide 40–50）：给定 FLOPs 预算如何分配参数量与 token 数。
> 5. **其他例子 & 总结**（Slide 51–53）：Diffusion 模型的复现；核心 takeaways。

### Slide 对照表

| PDF Slide | 主题 | 对应章节 |
|---|---|---|
| 1–4 | Motivation & Outline | §0 |
| 5–11 | Scaling laws 的历史与统计理论视角 | §1 |
| 12–15 | Data scaling law 的经验观察（power law） | §2.1 |
| 16–20 | 为何是 power law：均值估计 & 非参数回归两个 toy example | §2.2 |
| 21–25 | 数据组成、重复数据、数据选择的 scaling | §2.3 |
| 26–33 | 模型工程：架构 / 优化器 / 深宽比 | §3.1 |
| 34–36 | Critical Batch Size | §3.2 |
| 37 | Learning Rate & muP（µP） | §3.3 |
| 38–39 | 下游任务预测性的警告 + scaling law 设计流程 | §3.4 |
| 40–44 | 联合 Data–Model scaling 与 Chinchilla 背景 | §4.1 |
| 45–49 | Chinchilla 三种拟合方法 + Method 3 的复现事件 | §4.2 |
| 50 | Train-optimal 与 inference-optimal 的张力 | §4.3 |
| 51–53 | 扩散模型复现 + 全讲总结 | §5 |

---

## 0. 本讲目标（Slide 1–4）

**核心 motivation**：你的富朋友给你 10 万张 H100 租一个月，让你训一个最强开源 LM。你已经有了：
- 基建团队 + 分布式训练框架（Lec 7–8 讲过）
- 预训练数据（A4 作业会做）

剩下一个大问题：**选什么模型？** 宽还是深？多少头？什么非线性？跟风 Llama 是最简单但最无聊的答案——这样永远打不到 frontier。

**Scaling laws 的承诺**：

> **旧世界**（unpleasant）：直接在大模型上调超参数——贵到做不起。
> **新世界**（optimism）：在一堆小模型上调，拟合出简单的预测规律，**再外推到大模型**。一次打满。

本讲要做的就是把"scaling laws"从"log 图上拟一条线、然后喊 AGI"拉回到严肃的工程工具。

---

## 1. Scaling Law 的历史与统计理论视角

> 对应 Slide 5–11。

### 1.1 统计学习视角：Sample Complexity 就是 Scaling Law 的理论雏形（Slide 6）

如果你熟悉 **VC 维、Rademacher 复杂度**，其实经典统计学习理论里已经有了"scaling law"的形式：

- **有限假设类（k 个假设）的泛化上界**：超额风险（excess risk）以 $O(1/\sqrt{m})$ 速率衰减，$m$ 是样本数。
- **光滑密度估计**（非参数生成建模）的 L2 误差上界：

$$
\| \hat{p} - p \|_2^2 \;\lesssim\; n^{-\frac{\beta}{2\beta+1}}
$$

这就是理论家所说的**非参数率**（nonparametric rates），β 刻画了被估函数的光滑度。

> **关键区别**：这些是**上界（upper bounds）**，不是**实际误差值**。Scaling laws 做的事情正是——从"理论上界"跨到"经验上真实拟合到的误差值"。

### 1.2 最早的 Scaling Law 论文：1993 Bell Labs（Slide 7）

Vapnik / Cortes 等人 1993 NeurIPS 的文章，思想几乎跟现代 scaling law 一模一样：

- 动机："训分类器太贵，我们需要在训练前就预测它训完多好"；
- 方法：**训一堆小模型 → 拟合曲线 → 外推**；
- 函数形式：$\text{test error} = \text{irreducible error} + \text{polynomial decay}$。

现代 scaling law 的所有核心元素都在里面了。

### 1.3 其他早期里程碑

| 论文 | Slide | 贡献 |
|---|---|---|
| **Banko & Brill '01** | 8 | 在 NLP 系统上展示 log-linear data scaling；提出"与其花钱搞算法不如花钱搞数据"。 |
| **Kolachina et al. 2012** | 9 | 明确讨论 power law（power-3, power-4）**作为函数形式**是否真的准。 |
| **Hestness et al. 2017** | 10–11 | 第一篇大规模神经 scaling law，覆盖机器翻译、语音、LM 等。提出"三段论"。 |

#### Hestness 的**三段式误差曲线**（非常重要，Slide 10 原图核心）

模型随数据量增长的行为分为三段：

```
  log(error)
      │
  ════│═══════════════════ best-guess region（随机猜）
      │ \
      │  \           ← power law region（稳定的幂律下降）
      │   \______
      │          \____   ← irreducible error（不可约误差区）
      │───────────────── log(n)
```

1. **Best-guess region**：数据太少，模型相当于乱猜；
2. **Power law region**：**我们最感兴趣的区域**，log-log 是直线；
3. **Irreducible error region**：触底，再加数据也没用（被噪声、标签错误、模型容量等限死）。

> Hestness 2017 **非常超前**：已经提到了后来才被大讲特讲的"emergence"（离开 best-guess 区时能力突然出现）、"scaling by compute"、甚至"量化（quantization）换算力"的想法——**早在 2017 年就都被看见了**。

### 1.4 关于"不会 scaling"的情况（字幕Q&A）

> Q：哪些情况下看不到 scaling？

**答**：
- 在**训练损失及其 held-out 版本**上，scaling 是很自然的（经典统计保证了收敛）。
- 但存在 **Inverse Scaling Prize** 类现象——模型越大表现越差，例如"让模型抑制复制行为"任务。
- 本质：**一旦你跑出数据分布足够远**，行为就不再由数据约束，什么都可能发生（no scaling, inverse scaling, …），这是深度学习鲁棒性问题的延伸。

---

## 2. 数据 Scaling Law

> 对应 Slide 12–25。这一块是全讲最"理论干净"的部分：**为什么 error vs data 在 log-log 上是直线**？

### 2.1 经验观察：Data scaling 是 power law（Slide 13–15）

**经验事实**（Kaplan 2020 等）：画图规则固定——

- x 轴：$\log(\text{数据量 } n)$
- y 轴：$\log(\text{test loss})$

得到的就是一条**近乎完美的直线**。等价地：

$$
\text{Error}(n) \;\approx\; C \cdot n^{-\alpha}
$$

称为"**scale-free**"或"**power law**"。

**数据 scaling law 的正式定义**：一个简单公式，把数据量 $n$ 映射到 excess error（超过 irreducible 的那部分）。

**重要细节（Slide 15 + 字幕）**：当我们说"变数据量看误差"时，**模型要足够大**，大到不被数据量饱和——否则你进入了 irreducible error 区，误差被模型容量限死，看不到 power law。字幕里同学的提问点就是这个：

> **Q**："data scaling 图里的模型是固定一个大模型？"
> **A**：是，**固定一个大到能容纳所有数据量的模型**，否则数据多了就卡在 irreducible error 区看不到线。

### 2.2 为什么是 power law？——两个 toy example

#### 2.2.1 Toy 1：均值估计（Slide 16–18）

- 输入：$x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$
- 任务：估计均值 $\hat{\mu} = \frac{1}{n} \sum_i x_i$
- 误差：由标准论证，$\mathbb{E}\,[(\hat{\mu} - \mu)^2] = \frac{\sigma^2}{n}$

取 log：

$$
\log(\text{Error}) = -\log n + 2\log \sigma
$$

这就是一条 **slope = −1 的直线**。**均值估计本身就是一个 scaling law**。

推广：经典参数估计（含线性回归）通常给出 $\frac{1}{n}$ 或 $\frac{1}{\sqrt{n}}$ 的率——对应 log-log 直线斜率 **−1 或 −0.5**。

**但经验数据给出的斜率完全不同**（Slide 18）：

| 任务 | 经验斜率 $\alpha$ |
|---|---|
| Machine Translation | **0.13** |
| Speech | **0.3** |
| Language Modeling | **0.095** |

——**都远小于 1 或 0.5**。这意味着经典"参数估计率"不足以解释 LLM 的 scaling 行为。那是什么模型能解释？

#### 2.2.2 Toy 2：非参数回归（Slide 19）

- 输入：$x_1, \ldots, x_n$ 均匀分布在 $d$ 维单位盒子里；$y_i = f(x_i) + \mathcal{N}(0, 1)$。
- 任务：估计任意光滑函数 $f$。
- 方法（最简单的非参数估计）：把 $d$ 维空间切成**小盒子**，每个盒子里取 $y$ 均值当估计。

在 $d = 2$ 维，设边长为 $n^{-1/4}$，则：

- 总盒子数 $\sim \sqrt{n}$
- 每个盒子里有 $\sim \sqrt{n}$ 个样本
- 每盒均值估计误差 $\sim \frac{1}{\sqrt{n}}$（Toy 1）

但我们同时要估**很多盒子**，经过细致分析，整体误差 $\sim n^{-1/2}$ 维。推广到 $d$ 维：

$$
\boxed{\text{Error} \approx n^{-1/d}}
$$

于是 log-log 斜率为 $-1/d$——**维度越高，scaling 越慢**。

> **Takeaway**：灵活的非参数学习（神经网络近似任意函数就是这一类）的 scaling 率**天然依赖"有效维度"**。

#### 2.2.3 "Intrinsic Dimensionality" 解释（Slide 20, Bahri 2021）

这套说法把 Toy 2 推到 LLM：

1. Scaling laws 来自 polynomial rate $n^{-\alpha}$；
2. 斜率 $\alpha$ 大致等于 **数据的本征维度的倒数** $1/d_{\text{intrinsic}}$——任务越复杂（本征维度越高），scaling 越慢。

> **注**：本征维度的估计本身非常难（"and as difficult as modeling the data overall"），所以这个解释是**直观但非严格**的。但它给了我们一个统一的故事：**为什么 LM 的 α 特别小**？因为语言数据的本征维度非常高。

### 2.3 数据 Scaling Law 的进阶应用（Slide 21–24）

确立了"data 与 error 的 log-log 直线"之后，可以扩展成很多工程工具：

#### 2.3.1 数据组成（Data Composition）（Slide 22）

**Kaplan 2021 的关键发现**：换训练数据分布，**只改 offset，不改 slope**。

$$
\log \text{Error} = -\alpha \log n + \underbrace{C_{\text{dataset}}}_{\text{只有这里变}}
$$

**工程含义**：**在小模型上做数据筛选实验就够用了**！不需要每次筛完再跑一个 GPT-3 规模的 run。后续这也催生了大量用 scaling law 做 **optimal data mixing** 的工作。

#### 2.3.2 数据重复（Multi-Epoch Scaling）（Slide 23）

**问题**：互联网可能没那么多数据了，能不能重复训？

**发现**：存在"**有效数据量**"的概念——

- 在前 $\sim 4$ 个 epoch 内，重复数据基本等价于新数据；
- 超过 4 个 epoch，收益迅速衰减。

记号（Slide 23）：
- $U_D$：unique tokens
- $R_D$：repetition count
- $R_D^*$：常数，约为 4
- $D'$：effective data（代入 scaling law 用的"有效数据量"）

通过修改标准 scaling law 代入 $D'$ 代替 $D$，可以预测重复数据下的训练曲线。

#### 2.3.3 "重复高质量" vs "引入低质量新数据"（Slide 24）

当你有 trillions of tokens 预算，有两条轴可选：

- **纵轴**：要不要把 Wikipedia / 盗版书重复 10 遍？
- **横轴**：要不要引入新的低质量数据？

CMU 一篇工作（Slide 24）做了 **联合 scaling law**，可以在这两条轴上找最优。

### 2.4 小结（Slide 25）

- Log-data 与 log-error 的线性关系**非常稳健**，跨领域、跨模型都成立；
- 理论理解：类似经典泛化上界（Toy 1）；非参数率给出维度依赖（Toy 2）；
- 应用：数据收集、数据清洗、数据 mixing、判断是否该重复数据。

---

## 3. 模型工程的 Scaling Laws

> 对应 Slide 26–39。现在从"数据 vs 误差"切到**模型超参数 vs 误差**——这是更工程的问题。

### 3.0 要回答的工程问题（Slide 26–27）

训练一个大 LM 时要做的决策：

| 类别 | 选择 |
|---|---|
| **架构** | Transformer vs LSTM vs State-Space？ |
| **优化器** | Adam vs SGD？ |
| **深宽比** | 宽但浅 vs 窄但深？多少层？ |
| **Batch size** | 多大合适？怎么随规模变？ |
| **资源分配** | 训久一点 vs 训大一点？更多数据 vs 更多 GPU？ |

Scaling law 给一个**统一的程序**来回答这些问题。下面按 Kaplan 2020 的顺序走。

### 3.1 架构 / 优化器 / 深宽比（Slide 28–33）

#### 3.1.1 Transformer vs LSTM（Slide 28）

- **Brute force 做法**：把 LSTM 放大到 GPT-3 规模看看——花几千万美元。
- **Scaling law 做法**：训很多不同 compute 预算下的 LSTM 和 Transformer，画 log-log 曲线。

**Kaplan 2020 的结论**：

- 两条曲线**不相交、几乎平行**；
- Transformer 恒定地比 LSTM 高效一个常数倍（字幕粗估"≈15× compute"）；
- **结论**：无论你将来 scale 到多大，LSTM 都会落后一个常数因子——**不值得继续押注**。

#### 3.1.2 跨架构横扫（Slide 29, Tay et al.）

Google 团队同类型实验，把一堆候选架构与 Transformer baseline 对比：

- x 轴：compute
- 红线：候选架构
- 绿线：Transformer baseline

**结果**：真正在 scaling 上**持续优于** Transformer 的只有两个——

1. **Gated Linear Units (GLU)**
2. **Mixture of Experts (MoE)**

这恰好就是今天 SOTA 模型都在用的东西。**Performer 等架构则输给了 baseline**。Scaling law 给了"为什么现在大家都跑 SwiGLU + MoE"的硬证据。

#### 3.1.3 Adam vs SGD（Slide 30，Hestness 2017）

同样看到：

- Adam 和 SGD 的 scaling 曲线**斜率几乎相同**；
- Adam 有一个**常数 compute 因子优势**。

（这里 RHN = Recurrent Highway Networks，2017 的老架构，只是陪衬。）

#### 3.1.4 深度 vs 宽度（Slide 31）

Kaplan 实验：固定参数量，扫不同层数。

- **1 层 vs 2 层**：差距巨大（1 层根本不 work）；
- **2 层以上**：差距非常小，**基本是一个平的盆**——很多深宽比都接近最优。

对应 Lec 3（架构）讲过的 rule of thumb：**aspect ratio $h/L \in [10, 100]$ 都 OK**，Kaplan 这里再次印证。

#### 3.1.5 "不是所有参数都平等" —— Embedding 参数的异常（Slide 32）

**踩坑点**：

- 若把 **embedding 参数也算进总参数量**，scaling 曲线会出现奇怪的**弯折**；
- 只算 **non-embedding parameters** 时才是干净的 log-log 直线。

**原因**：embedding 参数只在查表时用，在前向计算里不像普通 matmul 参数那样贡献"学习容量"。

**延伸（字幕）**：类似问题在 **MoE scaling laws** 里更严重——sparsely activated 参数如何折算成"等效 dense 参数"？近年有专门论文处理。

#### 3.1.6 深宽比在不同尺度上是否稳定？（Slide 33）

Kaplan 对比了 50M、270M、1.5B 模型的 aspect ratio vs loss 曲线：

**发现**：曲线**形状基本不变**，最低点位置也几乎不变。这意味着：

> **可以在 50M 上扫 aspect ratio，把最优值直接搬到 1.5B**。

**思维范式切换**：
- 传统深度学习：在实际部署规模上调超参；
- Scaling law 思维：**scale-aware hyperparameter tuning**——在小规模扫，但要相信规律可外推。

同样的图对 **FFN 维度比例、attention head 维度** 都成立。

### 3.2 Batch Size 与 Critical Batch Size（Slide 34–36）

#### 3.2.1 Critical Batch Size 定义（Slide 34）

Batch size 有**两个区域**：

- **Batch 很小时**（小于 noise scale）：加倍 batch ≈ 多走 1 步梯度——几乎**免费加速**。
- **Batch 过大后**（大于 noise scale）：加倍 batch 几乎**不减少所需 step 数**——只是浪费算力。

**Critical batch size $B_{\text{crit}}$**：这两个 regime 的交界点。

$$
B_{\text{crit}} \approx \text{达到目标 loss 所需的最少样本数} \;/\; \text{最少梯度步数}
$$

(这个表达式可以理解为：把"总样本量"除以"最少步数"，得到一张卡能压到的并行 batch 上限。)

#### 3.2.2 Critical Batch Size 随 loss 降低而**上升**（Slide 35）

**重要且反直觉**：

- **Loss 目标越低（模型训得越好），$B_{\text{crit}}$ 越大**。

> **字幕 Q&A 的直觉解释**：loss 越低，更新越"精细"，**需要更低的学习率 + 更低的梯度噪声**。而降低噪声的一种办法就是用更大的 batch 去平均。所以 loss 降到后期，$B_{\text{crit}}$ 自然变大。

**工程后果**：Llama 3 训练里显式**在训练中途逐步增大 batch size**——就是这个规律的直接应用。

#### 3.2.3 Batch Size 随 compute 如何变（Slide 36）

Kaplan 的 scaling 分析：

- **x 轴**：compute 预算
- **y 轴**：optimal batch size

**发现**：在相当大的 compute 区间里，**optimal batch size 可以不断增大**，而**总 step 数近似不变**——对数据并行（DP）是极大利好：batch 越大越方便 parallelize。

### 3.3 Learning Rate & µP（Slide 37）

#### 3.3.1 Naive scaling 下的学习率问题（左图）

"标准做法"的图象：

- 横轴：模型宽度 $n$（例如 MLP 宽度）
- 纵轴：loss
- 不同曲线对应不同 learning rate

**现象**：

- 模型越宽 → 最优 learning rate 越小（rule of thumb: $\text{lr}^* \propto 1/\text{width}$）；
- **最优学习率随规模漂移**——每变一个规模都要重扫。

进阶做法：对这些最优点再拟一个 learning-rate scaling law，预测大模型的最优 lr。但拟合误差会累积，不稳定。

#### 3.3.2 µP（"maximal update parameterization"）的思路（右图）

**替代方案**：**重新参数化模型**，让最优学习率**与规模无关**。

具体做法（概念层，下一讲会详细展开）：
- 根据宽度**对不同层的 learning rate 做缩放**；
- 根据宽度**缩放初始化方差**；
- 在 forward pass 中对某些层乘一个宽度相关的因子。

**效果**：在**最小的模型上调一次 lr**，**直接转移到最大的模型**——无需重调。

**工业界动向**：
- 原始论文（Yang et al. 2022）叫 µP
- Yao et al. 2024 有改进
- **Meta 在 Llama 4 声称自研了 "metaP"**——具体内容目前未公开
- 多个 frontier lab 都在做类似的 "scale-aware parameterization"

> 为什么这么火？如果 lr 随 scale 是不稳定的，scaling law 外推就有风险；但如果模型被重参数化到 lr-invariant，**整个外推问题就消失了**。

### 3.4 一个重要警告：下游任务的预测性（Slide 38）

**事实**：
- 在**训练 loss / perplexity** 上，scaling law 极其稳定（左图：参数数 vs 负对数困惑度，几乎是完美的线）；
- 在**下游 benchmark**（如 SuperGLUE 准确率）上，**就乱了**——不同架构、不同超参的模型表现散得很开。

**含义**：

- **不要把 perplexity 的 scaling 等同于下游能力的 scaling**；
- 这在**状态空间模型（SSM）** 的文献里尤其突出：perplexity 很好，但 in-context learning、QA 等能力明显弱于 Transformer。

### 3.5 基于 Scaling Law 的设计流程（Slide 39）

总结为**三步**：

> **Step 1**：训一堆小模型，跨越几个数量级的 compute。
> **Step 2**：在这些小模型上**确立一条 scaling law**（验证 log-log 线性）。
> **Step 3**：基于这条 scaling law 预测最优超参，**直接用在大模型上**。

**特例**：当两条候选的 scaling law 斜率相同、只有 offset 不同时（例如 Adam vs SGD），**在小模型上的胜者就是大模型上的胜者**——连 scaling 都不用做，结论直接 transfer。

**但 learning rate 是显著例外**——因为它本身跟 scale 强相关，要么显式拟 lr 的 scaling，要么上 µP。

---

## 4. 联合 Data-Model Scaling 与 Chinchilla

> 对应 Slide 40–50。这是 scaling law 对工业界影响最大的一段——直接决定了"训多大的模型、用多少 token"。

### 4.1 核心问题：给定 FLOPs 预算，怎么分（Slide 40–42）

**背景设定（2021–2023 的真实场景）**：
- 数据远比 compute 充足（互联网几乎无限）；
- **唯一稀缺资源 = total FLOPs**。

**问题**：同样的 FLOPs，训一个巨大的少走几步的模型，还是一个小模型训很久？两个极端显然都浪费——需要最优点。

#### 4.1.1 联合 scaling law 的函数形式

两种经典写法（Slide 40）：

- **Rosenfeld 2020 形式**：

$$
\text{Error}(n, m) \;=\; n^{-\alpha} + m^{-\beta} + C
$$

- **Kaplan 2020 形式**：

$$
\text{Error}(n, m) \;=\; m^{-\alpha} + n^{-1} \beta
$$

符号：$n$ = 参数数量，$m$ = token 数量，$C$ = irreducible error。

> 两种写法**一阶等价**，形式虽"经验凑出来的"，但实际**拟合极精准**。

#### 4.1.2 3D 曲面的拟合（Slide 41）

Rosenfeld 展示：

- 3 个轴：数据量、参数量、loss；
- 曲面：拟合的函数形式；
- 散点：真实 training run。

结果：**曲面几乎完美贴合所有散点**。更关键的是：

> **只用 $n, m$ 较小的一半数据做拟合，可以精准外推到更大的 $n, m$**。

这为"小规模外推"提供了定量证据。

#### 4.1.3 固定 FLOPs 的权衡图（Slide 42）

Kaplan 的经典图：

- x 轴：参数数 $n$
- 每条颜色线：固定的 compute 预算
- 沿一条等 compute 线变动 $n$，$m$ 自动随之变（因为 $\text{FLOPs} \approx 6\,n\,m$）
- 每条线都有一个**明确的极小点**——那就是这个 compute 下的"最优模型大小"。

### 4.2 Chinchilla 的三种方法（Slide 43–49）

**Chinchilla（Hoffmann et al. 2022）** 对 Kaplan 的估计提出重大修正：原 Kaplan 的 ratio 差得很远。

#### 4.2.1 Kaplan 错在哪：Learning-Rate Schedule 的处理（Slide 44）

现代 Transformer 都用 **cosine learning rate schedule**：预热 → 余弦下降到最小 lr。

**关键陷阱**：
- Cosine schedule **必须走完全程才能得到"好的模型"**——下降段最后的 lr cool-down 是拿 loss 的关键一段。
- **不能在中途截断**当作一个短训练——因为那样相当于用"错误的 lr schedule"在训。

> Kaplan 的做法是在训练中的不同时间点取 checkpoint 作为"不同 token 数下的模型"，这违反了 cosine schedule 的一致性，**导致预测的 token/param 比率偏低**。

Chinchilla 的每个小模型都**独立训练到完成**，每个都有自己正确的 cosine schedule——结果显著不同。

#### 4.2.2 Chinchilla 三种方法概览（Slide 45）

| 方法 | 核心 | 估得的 $\alpha$（关于 $n$） | 估得的 $\beta$（关于 $m$） |
|---|---|---|---|
| **方法 1：Min over curves** | 训很多不同 size 模型的完整曲线，对所有曲线取下包络 | ≈ 0.50 | ≈ 0.50 |
| **方法 2：IsoFLOPs** | 固定 FLOPs 扫 $n$，拟二次找极小点 | ≈ 0.49 | ≈ 0.51 |
| **方法 3：Joint fit** | 对 Rosenfeld 函数形式直接最小二乘拟 | 差 0.03 左右 | 同上 |

> **惊人一致**：方法 1、2 都给出 **$\alpha \approx \beta \approx 0.5$**——这意味着参数量和 token 数应该 **等幂律** 增加。

#### 4.2.3 方法 1：Minimum over Training Curves（Slide 46）

**流程**：
1. 训**很多个不同 size** 的模型，每个训到完整的训练曲线；
2. 把所有曲线画在一张图上（x = FLOPs, y = train loss）；
3. 取**下包络**——即任意 FLOPs 预算下，**所有模型取到的最小 loss**；
4. 下包络会近似一条**幂律直线**；
5. 读出每个 FLOPs 预算下最优的模型 size 和 token 数——这两者各自都满足 power law。

> 这是对 Hestness / Kaplan "最小包络是 power law"这一观察的形式化。

#### 4.2.4 方法 2：IsoFLOPs Analysis（Slide 47）—— **最经典、最直观**

**流程**：
1. 挑几个 FLOPs 预算（例如 $10^{18}, 10^{19}, 10^{20}, \ldots$）；
2. **固定 FLOPs**，扫一个范围的 $n$（参数量），对应 $m$ 自动变化；
3. 每条"等 FLOPs 曲线"画 loss vs 参数量——呈**凸形**（过小或过大都差）；
4. **取每条曲线的最低点**（或拟合二次函数取极小）；
5. 把"最低点的参数数"和"最低点的 token 数"分别对 FLOPs 画图——得到两条 power law。

**结果举例**：在 Chinchilla 的目标 compute 预算下，最优参数量 ≈ **63–67 B**（方法 1、2 给出的值极一致）。

> **方法 2 是行业里最常用的 scaling law 流程**。

#### 4.2.5 方法 3：Joint Fit（Slide 48）

直接对 Rosenfeld 的联合函数形式做最小二乘——在 size-data 网格里扫一堆模型，整体拟合 3D 曲面，再反推最优。

**问题**：拟合 3D 曲面本身误差大，方法 3 原始论文的估计跟方法 1、2 差了 0.03——**长期是一个悬案**。

#### 4.2.6 悬案反转：方法 3 的复现事件（Slide 49，Besiroglu et al. 2024）

**故事**：Epoch AI 的几个研究员去年（2024）去复现 Chinchilla method 3：

1. **原始数据拿不到**——于是他们用**图像取点**工具从原论文的图里还原了数据点；
2. 用重建的数据，他们发现**原论文的拟合有系统性残差非零**（回归的基本健全性检查没过）；
3. 修正拟合后，方法 3 的估计**与方法 1、2 完全一致**。

> **结论**：原论文的数据是对的，但**曲线拟合做错了**——修正后 Chinchilla 的三个方法完全统一。这是一个"replication 证实而非推翻"的有趣案例。

### 4.3 "Train-Optimal" ≠ "Deploy-Optimal"（Slide 50）

Chinchilla 回答的是：**固定 training compute，训出最强模型**。但这忽略了一个现实：

> **真实部署里，大部分 compute 花在 inference 上，不是 training 上**。

**含义**：哪怕 training 时"小模型 + 多 token"和"大模型 + 少 token"训出的模型同样强，**小模型的推理成本低得多**——值得"**over-train**"小模型。

**业界趋势（tokens per parameter 持续膨胀）**：

| 模型 | Tokens / Param |
|---|---|
| **GPT-3** | 2 |
| **Chinchilla** | 20 |
| **LLaMA 65B** | 22 |
| **Llama 2 70B** | 29 |
| **Mistral 7B** | 110 |
| **Llama 3 70B** | 215 |
| **Qwen 最新版**（字幕提及） | 已达 30T tokens 级 |

**关键洞察**：**预计使用量越大，越值得为更小、更过度训练的模型买单**。这解释了为什么今天大家疯狂堆 token，而不是一味堆参数。

---

## 5. 其他模型家族 & 总结

> 对应 Slide 51–53。

### 5.1 Scaling Law 不限于 Transformer-LM（Slide 51）

讲者组（Gulrajani+ 2023）曾想推**文本扩散模型**，完全不知道它能不能 scale、tokens/param 比率是多少。**做法**：

- 用 **IsoFLOPs 分析**（Chinchilla 方法 2）**原样套在 diffusion LM 上**；
- 对 autoregressive LM 也做一次作为对照。

**结果**：
- 两者都给出干净的 IsoFLOPs 凸曲线和幂律最小包络；
- 最小包络之间只差一个**常数 offset**——意味着可以**直接对比不同模型家族的 compute-效率**。

> **Takeaway**：scaling law 不是"精心挑选"出来的现象——**不同 generative model 上都自然成立**，可以把它当作**比较新架构的通用工具**。

### 5.2 整讲总结（Slide 52–53）

**Log-linearity 三件套**：
1. **Data scaling**：数据量 vs 误差——最干净，有 Toy 1/2 理论支撑；
2. **Model scaling**：参数量、compute vs 误差——让我们选架构、选优化器、选深宽比；
3. **Joint scaling**：data + model 共同决定 error，给出资源分配的最优解（Chinchilla）。

**一句话总结**：

> **Scaling laws 让我们"不用跑大模型就能预测大模型"——工程上的意义怎么强调都不过分。**

具体能干什么：

- 在**小模型**上做全部架构 / 优化器 / 深宽比 / 数据混合的选型实验；
- 用 **IsoFLOPs 分析**决定真正训练时的参数量和 token 数配比；
- 用 **µP** 让学习率不随规模漂移，降低外推风险；
- **始终警惕** 下游任务的预测性可能失效；
- **始终记住** train-optimal ≠ deploy-optimal——推理量大就要 over-train。

这些都是你下一讲（Scaling Laws 2）会看到的**更实际的 case studies** 会反复用到的底层工具。

---

## 附：一份"考前速查"

| 记忆点 | 一句话 |
|---|---|
| Scaling law 的"第一性原理" | Error = irreducible + polynomial decay（Bell Labs 1993 就有） |
| 为什么 log-log 是直线 | 参数估计率 $1/n$；非参数估计率 $n^{-1/d}$；LLM 的 $\alpha$ 小是因为 intrinsic dimension 高 |
| Hestness 三段论 | best-guess → power law → irreducible |
| 数据组成的效应 | 只移 offset，不动 slope（所以小模型筛数据够用） |
| 数据重复上限 | 约 4 epoch，之后 effective data 迅速衰减 |
| Transformer 的 scaling 地位 | 比 LSTM 常数倍更高效；GLU + MoE 能进一步超越 |
| Aspect ratio | 10–100 是宽盆，各 scale 几乎不变 |
| Embedding 参数 | **不要** 把它算进"参数量"做 scaling 分析 |
| Critical batch size | loss 目标越低，$B_{\text{crit}}$ 越大（Llama 3 中途涨 batch 的依据） |
| muP | 通过重参数化让最优 lr 不随 scale 变 |
| 下游任务 | Perplexity scaling ≠ 能力 scaling，要小心 |
| Chinchilla 核心比例 | **≈ 20 tokens / parameter**（方法 1、2、修正后的 3 都一致） |
| Kaplan vs Chinchilla 的差别来源 | Learning rate schedule 的处理（cosine 必须跑完） |
| Train-optimal vs Deploy-optimal | 推理占大头 → over-train 小模型才划算 |

