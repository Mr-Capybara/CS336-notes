
# CS336 Lecture 7: Parallelism (Basics) — 详细笔记

> 本讲主题：**从"单 GPU 极致优化"跨越到"多机多卡并行训练"**。目标是把大模型（参数量远超单卡显存）在整个数据中心上跑起来，并且**算力随 GPU 数量线性扩展**、**显存随 GPU 数量线性扩展**。
>
> 本笔记配合 PDF 讲义 `2025 Lecture 7 - Parallelism basics.pdf`（共 60 页）阅读。全篇沿着讲义顺序推进，每节都给出对应 slide 页码，主线如下：
>
> 1. 网络硬件 & 集合通信原语（Slide 4–10）
> 2. **Data Parallelism**（Slide 12–27）：Naive DP → ZeRO Stage 1/2/3 (FSDP)
> 3. **Model Parallelism**（Slide 28–39）：Pipeline Parallel、Tensor Parallel
> 4. **Activation Parallelism**（Slide 40–45）：Sequence Parallel、Activation Recomputation
> 5. 其它并行（Slide 46）：Context/Ring Attention、Expert Parallel
> 6. 组合策略与 rule of thumb（Slide 47–49）
> 7. 真实大模型训练的案例（Slide 50–60）

### Slide 对照表（按讲义官方分节）

| PDF Slide 范围 | 主题 | 本笔记对应章节 |
|---|---|---|
| 1–3 | Outline / Goals | §0 |
| 4–10 | Part 1: Basics of networking | §1 |
| 11 | Parallelism 分类总览（DP / MP / Activation Parallelism）| §2–4 章首 |
| 12–14 | Naive DP + 16× memory 账 | §2.1–2.2 |
| 15–25 | ZeRO stage 1/2/3 (FSDP) | §2.3–2.6 |
| 26–27 | DP 的剩余问题（batch size 瓶颈、激活显存） | §2.7 |
| 28–35 | Pipeline Parallel（含 zero-bubble） | §3.1 |
| 36–39 | Tensor Parallel | §3.2 |
| 40–45 | Activation memory + Sequence Parallel | §4 |
| 46 | Context / Expert Parallel | §5 |
| 47–49 | Recap table + 3D Parallelism rule of thumb | §6 |
| 50–53 | Megatron / Narayanan 2021 | §7.1 |
| 54–60 | 近期大模型案例 + Llama 3 故障统计 + 总结 | §7.2–7.4、§8 |

---

## 0. 本讲目标

> 对应 Slide 2–5。


单 GPU 有两个硬约束：

- **算力不够**：即使 H100 的 FLOPs 曲线再陡峭，训练一个 SOTA LLM 仍需要数千卡同时工作。全球最快的超级计算机是 EFLOPs 级别（exaflops），这才是训大模型的底。
- **显存不够**：一个几十 B 的模型及其优化器状态根本装不进一张卡。

所以必须**多机并行**。目标是：

$$
\text{GPU 数}\uparrow \;\Rightarrow\; \text{可训模型最大尺寸}\uparrow \text{（线性）}\;\wedge\;\text{有效算力}\uparrow\text{（线性）}
$$

且分析并行算法时，我们只数**集合通信原语**的次数与字节数，不深入底层实现。

---

## 1. 基础设施：硬件拓扑 & 集合通信

> 对应 Slide 6–10。

### 1.1 GPU 集群的分层拓扑（Slide 6）

一个典型集群（以 GPT-NeoX 的 8×A100 节点为例，H100 节点同构）：

- **节点内（intra-node）**：单机 8 卡 GPU，通过 **NVSwitch / NVLink** 互联，带宽极高。
- **节点间（inter-node）**：多台机器之间通过 **InfiniBand (HDR)** 的交换机互联，带宽约为 NVLink 的 **1/8**。
- **集群级（multi-rack）**：超过约 **256 GPU** 时，进一步要经过 **leaf / spine switch**，带宽再降一档。

**关键结论**：

| 层级 | 带宽 | 延迟 | 典型规模 |
|---|---|---|---|
| 节点内 (NVLink) | 极高 | 极低 | ≤ 8 GPU |
| 节点间 (InfiniBand, all-to-all) | 高 | 中等 | ≤ 256 GPU |
| 跨 rack (leaf/spine) | 低 | 高 | > 256 GPU |

**这个层级结构直接决定了每种并行策略应该放在哪一层**——带宽要求高的放节点内，带宽要求低的放更远的层级。

### 1.2 GPU vs TPU 拓扑（Slide 9）

- **GPU**：节点内 8 卡 all-to-all 快通信；跨节点 all-to-all 直到 ~256；再往上变慢。
- **TPU**：采用 **toroidal mesh（环面网格）**，每个 TPU 芯片只跟**邻居**快速通信，但可以无限扩展。

**差异的意义**：环面网格对 all-to-all 通信没有天然支持，但对 **ring-based 的集合通信（如 all-reduce、reduce-scatter）效率一样高**。这让 TPU 更适合纯集合通信的工作负载，而不喜欢某些异构通信模式（如 pipeline parallel 的点对点）。

### 1.3 集合通信原语（Slide 7）

这是分析并行算法的"汇编语言"。记住它们的通信代价（以参数总量 $P$ 计）：

| 原语 | 语义 | 通信代价 |
|---|---|---|
| **Broadcast** | rank $k$ 的一份数据复制到所有 rank | $\sim 1 \cdot P$ |
| **Reduce** | 所有 rank 的数据求和后发给 rank $k$ | $\sim 1 \cdot P$ |
| **All-Reduce** | 所有 rank 的数据求和后**每个 rank 都得到完整和** | $\sim 2 \cdot P$ |
| **All-Gather** | 每个 rank 贡献一段，最终每个 rank 拿到拼接后的全量 | $\sim 1 \cdot P$ |
| **Reduce-Scatter** | All-Reduce 的"部分版本"：每个 rank 只拿到求和结果的一段 | $\sim 1 \cdot P$ |

### 1.4 关键恒等式：All-Reduce = Reduce-Scatter + All-Gather（Slide 8）

**这是本讲最重要的一个身份**，后续 ZeRO 的证明完全依赖它：

$$
\underbrace{\text{All-Reduce}(x)}_{\text{代价 }2P} \;\equiv\; \underbrace{\text{Reduce-Scatter}(x)}_{\text{代价 }P} \;+\; \underbrace{\text{All-Gather}(\cdot)}_{\text{代价 }P}
$$

- Reduce-Scatter 让每个 rank 拿到求和结果的**一段**；
- 紧接着 All-Gather 把这些段拼回去。
- **总带宽代价完全相同**，都是 $2P$。

> 这个等价意味着：凡是做 All-Reduce 的地方，都可以拆成"先 Reduce-Scatter，再 All-Gather"，**中间可以插入额外计算而不增加带宽**——这是 ZeRO 的核心套路。

---

## 2. Data Parallelism（数据并行）

> 对应 Slide 12–27。PDF Slide 11 将整个 Part 2 的并行家族先列出来：
>
> - **Data parallelism**：Naïve DP、ZeRO levels 1–3
> - **Model parallelism**：Pipeline parallel、Tensor parallel
> - **Activation parallelism**：Sequence parallel
>
> 本章覆盖第一类。

### 2.1 Naive DP（Distributed Data Parallel）（Slide 12）

**设定**：batch size $B$ 切成 $M$ 份发给 $M$ 个 GPU，每张卡有**完整的**模型参数副本。SGD 的更新为

$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell(x_i;\theta_t)
$$

每步做一次 **All-Reduce** 同步所有梯度，然后各自更新。

**评分**：

- **算力扩展**：✅ 好。每卡处理 $B/M$ 样本；只要 $B/M$ 足够大，GPU 就能跑满。
- **通信开销**：每 step 一次 All-Reduce，代价 $2P$。批越大越容易摊销。
- **显存扩展**：❌ 根本没扩展。每张卡都存一份完整模型 + 优化器状态。

### 2.2 显存开销到底有多惨？—— "16× 参数量"（Slide 13–14）

假设混合精度 + Adam，**每个参数需要约 16 字节**（PDF Slide 14 原话："We need 5 copies of weights and 16 bytes per param!"）：

| 类别 | 字节/参数 |
|---|---|
| BF16 权重（用于前向/反向） | 2 |
| BF16 梯度 | 2 |
| FP32 master weights（SGD 的累积目标） | 4 |
| Adam 一阶矩 $m$（FP32） | 4 |
| Adam 二阶矩 $v$（FP32） | 4 |
| **合计** | **16** |

> 即使模型只有 2 字节参数本身，加上梯度 & 优化器状态就是 8 倍；再考虑 master weights 就是 16 倍。**内存主要被 Adam 的 $m, v$ 和 FP32 master weights 吃掉**（统称 "Optimizer state"）。

**讲解示例（字幕）**：一个 7.5B 模型在 64 卡上每张卡全复制 → 每卡约 **120 GB**，其中大部分是优化器状态。

分片后的示意数据（字幕口述，用作理解 ZeRO 各阶段相对效果）：

| 阶段 | 被分片的内容 | 每卡显存（7.5B, 64 卡；口述近似值） |
|---|---|---|
| baseline (naive DP) | 什么都不分 | ~120 GB |
| **ZeRO Stage 1** | 仅优化器状态 | ~31.4 GB |
| **ZeRO Stage 2** | + 梯度 | ~16.6 GB |
| **ZeRO Stage 3 (= FSDP)** | + 参数 | ~1.9 GB |

### 2.3 ZeRO 的核心 idea：分片无用的副本（Slide 15）

观察：参数、梯度、优化器状态**真的需要在每张卡上都有一份**吗？不需要。只要每张卡**在需要的时候能拿到**就行。ZeRO（Zero Redundancy Optimizer）逐阶段把这三类数据切开。

PDF Slide 15 原文：
> **Core idea: split up the expensive parts (state) and use the reduce-scatter equivalence.**

也就是说 ZeRO 的全部魔法 = **"把贵的东西分片" + §1.4 的恒等式**。下面逐阶段展开。

### 2.4 ZeRO Stage 1：切优化器状态（Slide 16–18）

**想法**：$m, v, \text{master weights}$ 这些东西只在"更新参数"那一刻用到，跟前向/反向无关，完全可以分片。

**流程**（把参数按 rank 划段，rank $k$ 负责参数段 $k$）：

1. 每卡拿到一份 mini-batch，前向 + 反向算出**完整梯度**（暂时还没分）。
2. **Reduce-Scatter** 梯度：rank $k$ 只保留段 $k$ 的**求和后梯度**。
3. rank $k$ 用自己保存的段 $k$ 优化器状态，**只更新自己那段参数**。
4. **All-Gather** 更新后的参数，把整模型重新拼回每张卡（下一个 step 需要完整参数做前向）。

**通信代价**：Reduce-Scatter ($P$) + All-Gather ($P$) = $2P$，**和 Naive DP 的 All-Reduce 完全一样**。

> **Stage 1 是免费的**——在带宽受限条件下，完全不多花通信就白赚了一大块显存。所以"有条件就一定要上"。

PDF Slide 18 的对照表（原表重画）：

| | Naïve DDP | ZeRO Stage 1 |
|---|---|---|
| 通信原语 | 一次 All-Reduce（gradients） | 一次 Reduce-Scatter（send gradients） + 一次 All-Gather（collect params） |
| 通信代价 | $2P$ | $2P$ |
| 每卡显存 | $(4+K)\cdot P$ | $(4 + K / N_\text{gpu})\cdot P$ |

其中 $K$ 是"除参数之外每参数的字节数之和"（优化器状态 + master weights），$N_\text{gpu}$ 是 DP 组 GPU 数。


**QA 澄清（为什么能分优化器状态）**：
- 每张卡有完整的参数和梯度，这些都是计算梯度所必需的；
- 但**"更新参数"这个动作**只需要「参数那一段 + 梯度那一段 + 优化器状态那一段」。只要每张卡只负责更新"自己那一段"，优化器状态就可以只存一段。
- 更新完再 All-Gather 把各段参数拼回。

### 2.5 ZeRO Stage 2：再切梯度（Slide 19–20）

**困难**：梯度要通过反向传播来算。如果先算完**完整**的梯度向量再 Reduce-Scatter，那一瞬间的峰值显存仍然是完整梯度——没省到。

**解法**：**反向一边走一边分发**。每算完一层的梯度就立刻对这一层做 Reduce-Scatter，然后**立即释放**这一层在本卡上的梯度（本卡只保留自己那段）。

- 通信次数变多（按层触发），但**总字节数不变**，仍然 $2P$。
- 峰值显存从"完整梯度"降到"一层梯度"。

> Stage 2 相比 Stage 1 多了一点"调度复杂度"（按层 fire-and-forget），但本质仍然是免费午餐。

### 2.6 ZeRO Stage 3 = FSDP：连参数也切（Slide 21–24）

这是**PyTorch 的 `FullyShardedDataParallel` 背后的算法**。最激进：连参数本身都不完整保存了，只在**需要用到某一层时**临时 gather 一下。

**流程概览**：

**前向 pass**：
1. rank $k$ 本地只保留自己那段参数。
2. 要算第 $\ell$ 层前向时：All-Gather 第 $\ell$ 层的完整权重。
3. 做 `x = layer_l(x)`，然后**立刻释放** gather 过来的权重。
4. 循环直到最后一层。**激活**必须保留（反向要用）——这是显存瓶颈。

**反向 pass**（按层反向）：
1. 要算第 $\ell$ 层反向时：再 All-Gather 第 $\ell$ 层权重。
2. 算这一层梯度。
3. 对这一层梯度做 Reduce-Scatter（每个 rank 只拿走自己那段）。
4. 释放该层权重和不属于自己的梯度。

**通信代价**：
- 前向：1 次 All-Gather ($P$)
- 反向：1 次 All-Gather ($P$) + 1 次 Reduce-Scatter ($P$)
- **合计 $3P$**（不是 $2P$，比 Stage 1/2 多了 50%）

#### 2.6.1 FSDP 为什么没慢爆？—— Overlap Communication & Computation（Slide 23）

核心技巧：**预取（prefetch）**。在 GPU 忙着做第 $\ell$ 层的 matmul 时，后台已经在偷偷 All-Gather 第 $\ell+1$ 层的权重。等前者算完，后者的权重刚好就绪——**计算和通信几乎完全并行**。

PDF Slide 23 用的计算图例子是 $y = (W_1\,W_0 + W_2\,W_0)\,x$（权重在一次前向里**重复使用** $W_0$，用来演示 FSDP 如何识别"这层权重刚加载过，不必重新 gather"）。改写成链式看，时间线大致如下：

```
Comm:  [AG_0][AG_1       ][AG_2  ][free_1][...  反向也是同样套路  ...]
Comp:         [fwd_0][fwd_1]       [fwd_2][free_2][bwd …]
              ↑ 开始算时 W_0 已就绪   ↑ 这里 W_1 也已经 prefetch 完
```

- 前几个 bubble 不可避免（第一次启动时 W_0 还得现 gather）；
- 一旦预取流水起来，**通信被计算完全掩盖**。
- 反向 pass 因为要做 Reduce-Scatter + All-Gather，bubble 略大，但仍可通过预取大幅缩小。

**QA 澄清：prefetch 到哪？** —— 到 GPU 显存里专门开的 **buffer**。这部分显存没计入上表那个 1.9 GB，实际会有额外开销；更大的"房间里的大象"是**激活**，FSDP 并不能省激活显存。

#### 2.6.2 对比表（Slide 24：数据并行三阶段通信代价）

| 策略 | 参数 | 梯度 | 优化器状态 | 通信代价 | 实现复杂度 |
|---|---|---|---|---|---|
| Naive DDP | 复制 | 复制 | 复制 | $2P$ All-Reduce | ⭐ |
| ZeRO-1 | 复制 | 复制 | **分片** | $2P$ (RS+AG) —— **免费** | ⭐⭐ |
| ZeRO-2 | 复制 | **分片** | 分片 | $2P$（按层 RS，几乎免费） | ⭐⭐⭐ |
| ZeRO-3 / FSDP | **分片** | 分片 | 分片 | $3P$（1.5× 通信量，但 not bad） | ⭐⭐⭐⭐ |

#### 2.6.3 ZeRO in practice：会不会装得下？（Slide 25）

**这是 PDF 给出的最权威的实操数据表**。假设**纯 BF16 训练 + Kahan summation**（BF16 做 optimizer 里的一切，只保留 FP32 master weights），即**每参数 12 字节**：

| 阶段 | 每参数字节公式 | 单节点（8×A100 80G）能塞下最大模型 |
|---|---|---|
| Baseline (naive DP) | $12$ | **~6.66 B** |
| **ZeRO Stage 1** | $5$ （2 参 + 2 梯度 + 1 optimizer state/8） | **~16 B** |
| **ZeRO Stage 2** | $2 + 10/8$ （参数整份 + grad 和 state 都除 8） | **~24.62 B** |
| **ZeRO Stage 3** | $12/8$ （什么都除以 8） | **~53.33 B** |

> **记忆口诀**：从 baseline 到 ZeRO-3，能塞下的模型尺寸大约 **放大 8 倍**（6.66 B → 53.33 B）——这正是"分 8 卡"的极限收益。

**提示**：FSDP 的一个巨大优势是**架构无关**——它是一个 block wrapper，对几乎任何网络都能工作，不需要侵入模型代码。

### 2.7 Data Parallelism 的致命资源：Batch Size（Slide 26–27）

数据并行的并行度 ≤ batch size。你不可能给一张卡分 0.5 个样本。

**更要命的是：Batch size 本身有天花板**。OpenAI 的 *critical batch size* 论文指出，**超过某临界点后 batch size 对优化基本无边际收益**。直觉：

- 小 batch 时，梯度噪声是主要瓶颈，更大 batch 能有效减噪；
- 大 batch 超过临界点后，**瓶颈变成"梯度步数本身"**，再加样本也没用。

**这意味着**：

- batch size 是一种**有限、可分配的资源**，既可用于 DP 并行度，也可用于 pipeline parallel 减 bubble，或填充更多的 micro-batch；
- **光靠 DP 无法无限扩展**——必须引入模型并行。

**ZeRO 的另一个限制**：它只解决参数 / 梯度 / 优化器状态，**不解决激活显存**。如果你的模型激活爆显存，FSDP 也救不了你。

---

## 3. Model Parallelism（模型并行）

> 对应 Slide 28–39。

动机（Slide 28）：在**不消耗 batch size 资源**的前提下扩展显存，并且把激活也切开。

**关键区别**（与 FSDP 对比）：
- FSDP 把**参数**切开后在运行时临时 gather → 通信的是参数；
- 模型并行把参数**永久**留在固定的卡上 → 通信的是**激活**，参数从不跨卡搬。

两种切法：

- **Pipeline Parallel (PP)**：沿**深度方向**切，按层分卡。
- **Tensor Parallel (TP)**：沿**宽度方向**切，把单个大 matmul 拆成子 matmul。

### 3.1 Pipeline Parallel（PP）（Slide 29–35）

#### 3.1.1 Naive 按层切分 —— 灾难性的 bubble（Slide 29–30）

把第 0~L/4 层放 GPU0、L/4~L/2 放 GPU1，以此类推。前向时激活从 GPU0 流到 GPU3，反向时梯度反向流。

问题：**时间轴上几乎所有卡都在摸鱼**。单个样本过流水线时，每一时刻只有一张卡在算。加了 4 张卡却只有 1 张卡的吞吐——**最坏的并行**。

#### 3.1.2 Micro-batch 流水 —— 1F1B（Slide 31）

把 batch 切成 $m$ 个 micro-batch，像 CPU 流水线一样灌入：

```
GPU0: |F1|F2|F3|F4|                |B4|B3|B2|B1|
GPU1:    |F1|F2|F3|F4|          |B4|B3|B2|B1|
GPU2:       |F1|F2|F3|F4|    |B4|B3|B2|B1|
GPU3:          |F1|F2|F3|F4|B4|B3|B2|B1|
```

流水线两端仍然有 bubble。bubble 占比大致是：

$$
\text{bubble ratio} = \frac{S-1}{m}\qquad (S=\text{pipeline stages},\; m=\text{micro-batches})
$$

- $m$ 越大 bubble 越小 → **需要大 batch size** 来喂饱流水线。
- 这再次说明 batch size 是稀缺资源。

**NVIDIA 的实验数据（1F1B，Slide 33）**：

| Batch Size | PP=2 | PP=4 | PP=8 |
|---|---|---|---|
| 8 | 相对好 | 吞吐掉一半 | 灾难 |
| 128 | 几乎无损 | 稍降 | 可接受 |

**结论**：PP 对 batch size 极度敏感；batch 够大才值得用。

> Slide 32 另外补充了 PP 的两个**好处**：1) 相较于 DDP 节省显存；2) 通信属性好——只依赖激活（大小为 $b \times s \times h$）、且是**点对点**通信。这让 PP 特别适合**较慢的网络链路**，典型做法：PP 跨 node、TP 在 node 内。

#### 3.1.3 Zero-Bubble / DualPipe（高级调度）（Slide 34–35）

> PDF Slide 34 标题原话："Trading communication bandwidth for utilization" —— 即更激进的 pipeline 调度靠牺牲一部分带宽来填满气泡。

**关键洞察**：反向 pass 可以拆成两个**独立**的子计算：

- **B（activation gradient）**：$\partial \mathcal{L}/\partial x$，即对**输入**求导。这一步**有**时序依赖——下一层的 B 要依赖这一层的 B。
- **W（weight gradient）**：$\partial \mathcal{L}/\partial W$，即对**本层权重**求导。这一步**没有下游依赖**——只要本层的 $\partial\mathcal{L}/\partial y$ 和本层的输入激活 $x$ 都在，就能随时算。

```
      前向:   x ─► (mul A) ─► (nonlinearity) ─► y
     反向:   dy ─► (·)        ─► dx    ←── "B"，有依赖链
               └─► (mul x)    ─► dA    ←── "W"，无下游依赖，可任意重排
```

**调度策略**：
- 把 B 留在关键路径上维持依赖顺序；
- 把 W **挪到原本是 bubble 的空档**里填满 —— **bubble 被彻底填没**。

这就是 DeepSeek 称作 **DualPipe** 的核心技巧。代价：**实现极其复杂**，要深度侵入 autograd，需要自定义的任务队列。

> 课堂轶事：某 frontier lab 的整个 pipeline parallel 基础设施只有 2 人能看懂；其中 1 人离职后，PP 变成了"单点故障"。

#### 3.1.4 PP 的优缺点（Slide 32 提炼）

**优点**：
- 节省显存：参数 **和** 激活都按层分布。
- 通信只有**点对点**传激活，且可以是跨慢速链路的（inter-node / 跨 rack）。
- 因此 PP 最适合用在**最慢的网络层级**——"能用 PP 就不用 TP"。

**缺点**：
- 有 bubble；严重依赖大 batch。
- 实现复杂。

**一个真实观察**：Google 的 TPU 因为环面网格带宽大，基本不用 PP；GPU 集群过了 256 卡的 all-to-all 边界后则不得不上 PP。

### 3.2 Tensor Parallel（TP）（Slide 36–39）

**核心观察（Slide 36）**：Transformer 的绝大多数参数和 FLOPs 都是 matmul。那就**拆 matmul**——把一次大 matmul 分解成若干子 matmul，最后合并 partial sums。

#### 3.2.1 拆 matmul 的两种方式

对 $Y = X A$（$A \in \mathbb{R}^{H\times H'}$）：

**按列切 $A$**：$A = [A_1 \mid A_2]$
$$
Y = X\,[A_1\mid A_2] = [XA_1 \mid XA_2]
$$
每张卡拿 $A_i$，接收完整 $X$，算出一片 $Y$，**不需要通信就能得到部分输出**。

**按行切 $A$**（配合按列切 $X$）：$A = \begin{bmatrix}A_1\\ A_2\end{bmatrix}$
$$
Y = [X_1\mid X_2]\begin{bmatrix}A_1\\ A_2\end{bmatrix} = X_1 A_1 + X_2 A_2
$$
每张卡拿 $A_i$ 和 $X_i$，算出 $X_i A_i$，**最后做 All-Reduce 求和**。

#### 3.2.2 Megatron-LM 风格 MLP 的 TP 实现（Slide 37）

MLP 块：$Z = \text{Dropout}\big(\text{GeLU}(XA)\,B\big)$。切法：

- $A$ 按**列**切为 $[A_1 \mid A_2]$（第一次 matmul 不用通信）。
- $B$ 按**行**切为 $[B_1;B_2]$（第二次 matmul 算完后 All-Reduce 求和）。

**前向计算图**：

```
         ┌── X A_1 ──► GeLU ──► Y_1 ──► Y_1 B_1 ──┐
   X ──f─┤                                        ├──g─► Z
         └── X A_2 ──► GeLU ──► Y_2 ──► Y_2 B_2 ──┘
         (copy/identity)                    (All-Reduce)
```

- 前向：`f` 是 identity（复制 X 到两卡），`g` 是 **All-Reduce**（求和 $Y_1B_1 + Y_2B_2$）。
- 反向：对偶翻转。`f` 的反向是 All-Reduce（两路梯度相加），`g` 的反向是 identity（复制梯度到两路）。

**每层每次前/反向都要一次 All-Reduce 激活**，通信量 = 激活大小 × 2次 × 层数。**极度带宽饥饿**。

#### 3.2.3 TP 的硬规则：**不超过 8 卡**（Slide 38）

因为 TP 要求极高的带宽，**只应在节点内**（NVLink 连接的 8 卡）使用。HuggingFace 的实测数据：

| TP size | 吞吐下降 |
|---|---|
| 2–8 | 10–12%（可接受，节点内 NVLink 撑得住） |
| 16 | **~42%** |
| 32 | **~65%** |

**TP=8 是黄金配置**——装满一个节点，再多就跨节点吃 IB，性能断崖。

#### 3.2.4 TP vs PP 精确对照（Slide 39）

| 维度 | PP | TP |
|---|---|---|
| 切法 | 按深度（按层） | 按宽度（拆 matmul） |
| 主要通信 | 点对点传激活 | **每层 All-Reduce 激活** |
| 是否有 bubble | 有（吃 batch size） | **无** |
| 对 batch size 的消耗 | 有 | **无** |
| 实现复杂度 | 高 | 中 |
| 推荐部署层级 | 跨节点/跨 rack | **节点内 ≤ 8 卡** |

**PDF Slide 39 给出的精确通信量公式**（便于量化对比）：

$$
\text{PP 通信}_{/\text{micro-batch}} \sim b \cdot s \cdot h \qquad \text{（点对点）}
$$

$$
\text{TP 通信}_{/\text{layer}} \sim 8 \cdot b \cdot s \cdot h \cdot \frac{n_\text{devices}-1}{n_\text{devices}} \qquad \text{（All-Reduce）}
$$

其中 $b$ = batch size，$s$ = sequence length，$h$ = hidden dim。TP 的系数 8 来自 "每层 2 次 All-Reduce（前+反）× 一次 All-Reduce = 2P 通信" 的综合常数。

**结论**：

- **TP 通信量远大于 PP**（系数差一个数量级）；
- **TP 只能在低延迟、高带宽的链路上用**（节点内 NVLink）；
- **PP 能容忍慢链路**，适合放在集群的"慢通信"层级。

**答 Q**：TP 和 PP 可以也经常同时使用 —— 节点内 TP、跨节点 PP + DP。几乎唯一只用 PP 不用 TP 的大模型例子是 **DeepSeek V3**。

---

## 4. Activation Parallelism

> 对应 Slide 40–45。Percy/Tatsu 把 Sequence Parallel 归到 "Activation Parallelism" 这一章，因为它真正解决的是**激活显存**问题。

### 4.1 激活显存：大模型的真正瓶颈（Slide 40–41）

PyTorch 官方 profiler 的典型显存曲线（一个 step 内）：

- **参数**：全程常驻，不变。
- **优化器状态**：第一次 `.step()` 之后才出现，此后常驻。
- **激活**：前向过程中**单调上升**（每层都要缓存以备反向）。
- **梯度**：反向过程中逐层累积。
- **峰值**：出现在**反向中段**——激活还没释放完、梯度又在攒。

在**超大模型**上，即使 TP/PP 把参数/优化器状态都分开了，**激活显存仍随模型规模不可抑制地增长**。这是因为：**激活里有一部分既不走 matmul 也不能被 TP 切开**（见下）。

### 4.2 每层激活显存公式（Slide 42，来源 Korthikanti 2022）

NVIDIA 的经典公式（单层 Transformer，无任何并行）：

$$
M_\text{act}^\text{per layer} = s\,b\,h \cdot 34 \;+\; 5 \, a \, s^2 \, b
$$

> PDF Slide 42 里第二项的原始写法是 $5 \cdot a \cdot s / h$ 形式的紧凑表示，展开后等于 $5as^2b$（按照 attention head 数 $a$、序列长 $s$、batch $b$ 重新写清楚）。

符号：$s$ = 序列长度，$b$ = batch size，$h$ = hidden dim，$a$ = attention heads。

- **左项** $34\,sbh$：来自 MLP 和各种 pointwise op，随 hidden dim 线性涨。
- **右项** $5\,as^2b$：attention 中 softmax / QK 乘积等**序列长度平方项**。

> **Flash Attention 的意义**：通过 online tiling + recomputation，**直接把右项干掉**。

### 4.3 做了 TP 之后还剩什么？—— 未被并行的"边角料"（Slide 43）

对 matmul 用 TP（MLP + attention 里的 QKV）之后，**每层**激活显存变为：

$$
M_\text{act}^\text{TP} = \frac{sbh \cdot 24 + 5\,as^2b}{t} \;+\; sbh\cdot 10
$$

最后那个 $10\,sbh$ **没被 $t$ 除**，PDF Slide 43 明确拆解了它的来源：

| 项 | 贡献 |
|---|---|
| LayerNorm | $4\,sbh$ |
| Dropout | $2\,sbh$ |
| attention 和 MLP 块的输入（残差输入） | $4\,sbh$ |
| **合计** | $\mathbf{10\,sbh}$ |

这些都不是 matmul，TP 拆不开。所以**单纯 TP 时，这部分激活会一直随模型变大而增长**。

### 4.4 Sequence Parallel：把边角料沿序列维度切开（Slide 44）

**观察**：LayerNorm、Dropout 这些 pointwise op 在**序列维度上完全独立**——第 $t$ 个 token 的 LayerNorm 不依赖其它 token。

**做法**：把序列长度 $s$ 切成 $t$ 段，分给 $t$ 张 TP 卡。每张卡做自己那段的 LayerNorm / Dropout。

**需要的同步**（在 TP 块和 SP 块之间插入转换）：

- 前向：
  - `g`（从 SP 进入 TP）：**All-Gather**（把序列拼回来以便做 TP 下的 matmul）
  - `g̅`（从 TP 回到 SP）：**Reduce-Scatter**（把 TP 块要 All-Reduce 的结果直接分发）
- 反向：前向里的 AG 和 RS **互换**（duality）。

**奇妙之处**：原本 TP 要做的 All-Reduce 可以**拆成 Reduce-Scatter（紧接 SP 段）+ All-Gather（进入下一个 TP 段）**，**总字节数不变**！（还记得 §1.4 的恒等式吗？就是它在起作用。）

### 4.5 激活显存的最终公式（Slide 45）

把所有招式叠满：

$$
M_\text{act}^\text{TP+SP+Flash} = \frac{sbh \cdot 34}{t}
$$

- TP 把 matmul 部分除以 $t$
- SP 把边角料 pointwise 部分也除以 $t$
- Flash Attention (recomputation) 把 $s^2$ 项干掉

这就是你在各种 transformer 显存估算公式里常见的 "$\sim 34\,sbh/t$ per layer" 的来源。

### 4.6 Activation Recomputation（重算）（Slide 45/53）

反向时**重新跑一遍前向**来重建激活，从而不存激活。经典的"空间换时间"：

- 多付 $\sim$33% 的 FLOPs；
- 但能让更大 batch / 更大 micro-batch 装下，从而反过来**减小 PP bubble**、或喂饱 DP。
- **经常净赚**——Flash Attention 已经证明这套路行得通。

---

## 5. 其他并行策略（简介）

> 对应 Slide 46（单张 slide 把两者一起带过）。

### 5.1 Context Parallel / Ring Attention

用于**长上下文**训练。每张卡持有**一段 query**，keys & values 在卡之间像环一样传递；配合 Flash Attention 的 online tiling 思想逐块算 attention。

本质上是把 $s^2$ 项的 attention 也切开——在长上下文（几十 K 甚至上百 K token）时**不可或缺**。

### 5.2 Expert Parallelism（MoE）

**想法**：把一个大 MLP 换成一堆稀疏激活的 Expert MLP，每张卡放不同的 expert。

与 TP 的相似：都是切 MLP 到不同卡。
与 TP 的关键不同：**稀疏**激活 + **路由**。通信模式不再是规则的 all-to-all，可能出现 expert 负载不均、路由碰撞等问题。

---

## 6. 组合策略：3D / 4D Parallelism

> 对应 Slide 47–49。

### 6.1 四个"有限资源"

训练大模型时要同时平衡：

| 资源 | 说明 | 谁消耗它 |
|---|---|---|
| 显存 | 每卡 HBM | 参数 + 梯度 + 优化器状态 + 激活 |
| 带宽 | 节点内 / 节点间 | 所有集合通信 |
| 算力 | FLOPs | 所有计算 |
| **Batch size** | 有上限（critical batch size） | DP 数、PP 的 micro-batch 数 |

### 6.2 TPU Book 里的相图（Slide 48：Model vs Tensor parallel）

把"batch size / GPU 数"作为横轴，纵轴是效率，大致分三个区：

- **太小**：通信占主导，怎么摆都 communication-bound。
- **中间**：**FSDP + TP 混合**能刚好打满算力 → compute-bound。
- **足够大**：纯 FSDP（pure DP）就能做到 compute-bound。

**结论**：batch size 越大，策略越可以简化；batch size 紧张时要靠组合拳。

### 6.3 总览表：各策略的权衡（Slide 47 原表复刻）

PDF Slide 47 的官方 recap 表：

| 策略 | Sync overhead | Memory | Bandwidth | Batch size 影响 | 易用性 |
|---|---|---|---|---|---|
| **DDP / ZeRO-1** | Per-batch | 不扩展 | $2P$ | 线性（占 batch size） | 非常易 |
| **FSDP (ZeRO-3)** | 3× Per-FSDP-block | 线性扩展 | $3P$ | 线性（占 batch size） | 非常易 |
| **Pipeline** | Per-pipeline | 线性扩展 | **Activations**（点对点） | 线性（占 batch size） | **不易** |
| **Tensor + Sequence** | **2× transformer block** | 线性扩展 | **$8 \times$ activations per-layer All-Reduce** | **不占 batch size** | 不易 |

核心观察：
- **没有完美策略**，每种都在 memory / bandwidth / batch-size / ease-of-use 上有不同权衡；
- **TP+SP 的独特价值**是"不吃 batch size"；
- **FSDP 的独特价值**是"易用、无需改架构"；
- **PP 的独特价值**是"带宽需求最低，可以跨慢链路"。

### 6.4 经典 Rule of Thumb（Slide 49，**面试/实战都要记**）

PDF Slide 49 原文直译：

**Step 1**：直到模型（+ 激活）能塞进显存为止：
- **Tensor parallel** 到 "GPUs per machine"（= 节点内卡数，通常 8）；
- **Pipeline parallel** 跨机器（*或使用 ZeRO-3，视带宽而定*）。

**Step 2**：直到 GPU 用完为止：
- 剩下的 GPU 全部用 **Data Parallel**（FSDP 友好，对带宽不敏感）。

**Step 3**：如果 batch size 太小：
- 用 **Gradient Accumulation**——本机多累几步再同步，用 "更大等效 batch" 换 "更少 DP 通信频次"。

| 步骤 | 动作 | 原因 |
|---|---|---|
| ① 装下模型 | 节点内 **TP ≤ 8** | TP 最吃带宽，只能放最快链路 |
| ② 装下模型 | 节点间 **PP 或 ZeRO-3** | PP 和 ZeRO-3 能跨慢链路 |
| ③ 扩 GPU 规模 | 剩余 GPU 全做 **DP** | DP 最抗高延迟 |
| ④ 通信效率优化 | **Gradient Accumulation** | batch size 紧时，降同步频率 |

---

## 7. 真实大模型案例

> 对应 Slide 50–60。

### 7.1 Megatron-LM 的经典表（Narayanan 2021，Slide 50–53）

从 **1.7B 到 1T** 参数，使用 3D 并行（TP + PP + DP）训练，**MFU 在 40–52%**。

**PDF Slide 50 的关键数据（DP size 随规模递减）**：

从小模型到 1T 模型，DP 规模的演化（共 10 档模型）：

$$
\text{DP size}:\; 32,\; 32,\; 32,\; 32,\; 32,\; 32,\; 24,\; 15,\; 9,\; 6
$$

- **TP** 从 1 起，**上到 8 就封顶**（节点内上限；Slide 52 确认 TP=8 "几乎总是最优"）；
- **PP** 一开始是 1，**模型变大后逐渐增加**，用来塞下模型；
- **DP** 一开始尽可能大（32），**模型变大后被 PP 挤压**（因为 batch size 要分给 PP 做 micro-batch），最后降到 6。

**Slide 51–53 的三个重要结论**：

1. **近线性扩展**：careful 3D parallelism 给出几乎**平直的 per-GPU utilization 曲线**——GPU 加越多，总吞吐线性涨（Slide 51）。
2. **TP=8 是黄金**：64 机时最优配置是 **TP=8 × PP=8**（Slide 52）。即便 batch 小，TP=8 依然最优。
3. **Activation recomputation 自己挣回成本**：多付的 FLOPs 通过允许更大 batch、缩小 PP bubble 被赚回来。Slide 53 给出 $t=8, p=16$ 的实验例，throughput 反而更高。

### 7.2 近期模型的并行配方（Slide 54–59）

| 模型 | 来源 slide | 并行策略 |
|---|---|---|
| **OLMo / Dolma 7B** | 54 | 纯 **FSDP**（7B 模型大概率可以节点内塞下）|
| **DeepSeek v1** | 55 | **ZeRO-1 + TP + SP + PP** ← 教科书级 3D+ |
| **DeepSeek V3** | 55 | **PP=16 + Expert Parallel=64（8 节点）+ ZeRO-1**（罕见：**不用 TP**）|
| **Yi** | 56 | ZeRO-1 + TP + PP |
| **Yi-Lightning (2025)** | 56 | ZeRO-1 + **Expert Parallel 替换 TP** + PP |
| **Llama 3 405B** | 57 | 分阶段：Stage 1 小 batch 训练、Stage 2 主预训练、Stage 3 长上下文（引入 CP） |
| **Gemma 2 (2 / 9 / 27B)** | 59 | **ZeRO-3 + MP (= TP+SP) + DP**（TPU 网格让 MP 拉得更长） |

### 7.3 Llama 3 的带宽优先级排序（可直接当作部署手册）

按"对带宽的饥饿程度"排序，越饥饿越要部署在越快的链路上：

$$
\underbrace{\text{TP}}_{\text{节点内 NVLink}} \;>\; \underbrace{\text{CP}}_{\text{次高带宽}}\;>\; \underbrace{\text{PP}}_{\text{可容忍}}\;>\; \underbrace{\text{DP}}_{\text{最能忍高延迟}}
$$

DP 之所以放最后，是因为 FSDP 的 async prefetch 能掩盖网络延迟。

### 7.4 超大规模训练的现实：**故障**（Slide 58）

Llama 3 训练中的事故统计：

- **148 次** 由 **GPU 硬件故障** 导致的中断（占总中断 ~30%）
- **32 次** 非计划停机维护
- 除了"硬崩"，更可怕的是 **silent data corruption**——GPU 偷偷出错却不报告，直接毁掉一次 run

**训几万卡级的模型，算法之外还必须要有 fault-tolerant 的训练架构**。

---

## 8. 一句话总结（Slide 60）

PDF 收尾原话：

> - Scaling beyond a certain point requires multi-GPU, multi-node parallelism.
> - No single solution to the parallelism problem (probably want all 3 approaches).
> - Simple, interpretable rules of thumb for combining different forms of parallelism.

翻译成一句可以写在笔记本首页的话：

> **单机够用时用 FSDP；不够就节点内 TP=8，节点间 PP 或 ZeRO-3 用于塞下模型，剩下所有 GPU 做 DP 扩吞吐；Batch size 是核心稀缺资源，要会"花在哪儿"。**

这正是从 Llama 3 到 DeepSeek、从 Megatron 到 Gemma 全都在遵循的 rule of thumb。

