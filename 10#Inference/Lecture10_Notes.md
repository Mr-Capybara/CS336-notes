# CS336 Lecture 10：Inference（推理）完整笔记

> 课程背景：从本讲开始暂别 scaling laws，聚焦「推理」这一主题。**推理**即：给定一个**已训练好的固定模型**，针对用户输入（prompt）生成响应（response）。推理本身是一个非常深的话题，内容足以撑起多节课，本讲做了高度浓缩。

---

## 0. 为什么推理很重要

推理的场景远不止"部署一个聊天机器人"这么简单，它渗透进语言模型的方方面面：

- **实际应用**：聊天、代码补全（Cursor）、批处理等，用户直接消费生成的 token。
- **模型评测**：比如在指令跟随（instruction following）任务上评分，本质也是生成 token 再打分。
- **Test-time compute（测试时计算）**：近年大火的 "thinking before answering"（比如 o1 系列），让模型在输出最终答案前先"思考"一段，思考过程就是生成大量 token。
- **强化学习训练**：RLHF / RL 流程需要先采样响应、再用奖励模型打分，采样这一步就是推理。

一句话：推理是很多核心功能的底层。训练是一次性成本，而推理被反复执行，因此**效率**尤为关键。

**业界数据感受下量级**：OpenAI 一天生成约 1000 亿个 token，Cursor 一天被接受的代码约 10 亿行。推理成本占比相对训练在持续上升。

### 三个核心指标

| 指标 | 含义 | 适用场景 |
|---|---|---|
| **TTFT**（Time-To-First-Token） | 用户提交后，等多久才看到**第一个** token | 交互式应用（聊天） |
| **Latency**（延迟，秒/token） | 后续 token 的生成速度 | 交互式应用 |
| **Throughput**（吞吐量，token/秒） | 整体系统**每秒**能吐出多少 token（跨所有用户） | 批处理 |

注意：**高吞吐 ≠ 低延迟**。批处理可以靠堆并发拉高吞吐，但单个请求的延迟可能更糟。

### 训练 vs 推理的关键差异

- **训练（监督）**：所有 token 一次性可见，可以在序列维度完美并行（Transformer 的本质就是大矩阵乘）。
- **推理**：必须**自回归逐 token 生成**，后一个 token 依赖前面所有 token，天然难以并行 → 很难"吃满"算力 → 接下来我们会严格证明：**推理是 memory-limited（访存受限）**。

### 相关生态

- **闭源服务商**：OpenAI、Anthropic、Google 等。
- **开源权重服务商**：Together、Fireworks、DeepInfra 等。
- **开源推理框架**：vLLM（Berkeley）、TensorRT-LLM（NVIDIA）、TGI（HuggingFace）。

---

## 1. 理解推理的工作负载

### 1.1 Transformer 符号约定（沿用 Scaling Book）

| 符号 | 含义 |
|---|---|
| `B` | batch size（并发请求数） |
| `S` | prompt 中已有的（conditioning）token 数 |
| `T` | 当前 query / 要生成的 token 数 |
| `L` | 层数 |
| `D` | 模型隐层维度（hidden size） |
| `F` | MLP 上投影维度，通常 `F = 4D` |
| `N` | query head 数；`D = N*H` |
| `K` | key/value head 数（GQA 中 K ≤ N） |
| `G` | 组数，`N = K*G` |
| `H` | attention head 维度 |
| `V` | 词表大小 |

**前向计算 FLOPs 速算**：`6 * (B*T) * (num_params) + O(T^2)` 量级。

### 1.2 Arithmetic Intensity（算术强度）复习

**算术强度** = 每传输 1 字节做多少次浮点运算 = `FLOPs / Bytes`。这是判断一个算子是 compute-bound 还是 memory-bound 的关键指标。

**以一次 matmul 为例**：`X(B×D) @ W(D×F) = Y(B×F)`，全部用 bf16（2 字节）。

| 操作 | FLOPs | Bytes |
|---|---|---|
| 从 HBM 读 X | - | 2·B·D |
| 从 HBM 读 W | - | 2·D·F |
| 计算 X@W | 2·B·D·F | - |
| 写回 Y | - | 2·B·F |

$$
\text{intensity} = \frac{2BDF}{2BD + 2DF + 2BF}
$$

若 `B << D, F`（令 `D = cB, F = cB, c → ∞`），化简得：

$$
\text{intensity} \approx B
$$

### 1.3 判定阈值：Accelerator Intensity

以 **H100** 为例：

- 算力：989 TFLOPs/s（bf16）
- 显存带宽：3.35 TB/s
- **Accelerator Intensity** = 989e12 / 3.35e12 ≈ **295**

判定规则：

- `算法 intensity > 295` → **compute-limited**（好：算力被用满）
- `算法 intensity < 295` → **memory-limited**（坏：GPU 在等数据）

所以这次 matmul 要想 compute-limited，需要 `B > 295`。

### 1.4 极端情况：矩阵-向量乘（B=1）

当 `B=1` 时，intensity ≈ 1 ⇒ 严重 memory-bound：读了一整个 `D×F` 权重矩阵，却只做了 `2DF` 次 FLOPs，比值刚好是 1。这基本就是**逐 token 生成**时发生的事情——这是推理慢的根源。

---

## 2. 推理的算术强度

### 2.1 朴素推理 vs KV Cache

**朴素做法**：每生成一个 token，把"prompt + 已生成部分"整个再丢进 Transformer 走一遍 → 生成 T 个 token 总共 `O(T^3)` FLOPs。

**观察**：自回归、因果的 Transformer 里，前缀的计算是**完全重复**的，完全可以复用。

**方案：KV Cache**——把每层、每个 head、每个 token 的 K、V 向量存在 HBM 里。

- **KV Cache 大小**：`B × S × L × K × H × 2(K+V) × 2(bf16)` 字节
- 两阶段推理：
  1. **Prefill**：把 prompt 一次性编码，计算 KV cache。与训练一样可并行，**compute-limited**。
  2. **Generation**：逐 token 生成。每一步只算新 token 的 K、V 并追加到 cache，**memory-limited**。

### 2.2 MLP 层的算术强度

对一层 Gated MLP（有 `W_up, W_gate, W_down` 三个权重），设输入 `X: B×T×D`：

| 步骤 | FLOPs | Bytes |
|---|---|---|
| 读 X | - | 2·B·T·D |
| 读三个权重 | - | 3·2·D·F |
| 算 U = X @ W_up | 2·B·T·D·F | - |
| 写 U | - | 2·B·T·F |
| 算 G = X @ W_gate | 2·B·T·D·F | - |
| 写 G | - | 2·B·T·F |
| 算 Y = (GeLU(G)*U) @ W_down | 2·B·T·D·F | - |
| 写 Y | - | 2·B·T·D |

合计 `FLOPs = 6·B·T·D·F`，`Bytes = 4·B·T·D + 4·B·T·F + 6·D·F`。

设 `B·T << D, F`，化简得：

$$
\text{intensity}_{\text{MLP}} \approx B \cdot T
$$

**两阶段表现**：
- **Prefill**：`T=S` 通常较大，`B·T` 容易 > 295，compute-limited ✓
- **Generation**：`T=1`，要让 `B > 295` 就必须有足够多的**并发请求**——能否做到完全取决于线上流量。

### 2.3 Attention 层的算术强度（使用 FlashAttention）

设 Q: `B×T×D`，K、V: `B×S×D`：

| 步骤 | FLOPs | Bytes |
|---|---|---|
| 读 Q, K, V | - | 2·B·T·D + 2·B·S·D + 2·B·S·D |
| 算 A = Q @ K^T | 2·B·S·T·D | - |
| 算 Y = softmax(A) @ V | 2·B·S·T·D | - |
| 写 Y | - | 2·B·T·D |

合计 `FLOPs = 4·B·S·T·D`，`Bytes = 4·B·S·D + 4·B·T·D`，化简：

$$
\text{intensity}_{\text{attn}} = \frac{S \cdot T}{S + T}
$$

- **Prefill**（T=S）：`intensity = S/2`，长序列下很好 ✓
- **Generation**（T=1）：`intensity = S/(S+1) < 1`，**无论怎么 batch 都救不回来** ✗

#### 为什么 attention 对 B 无感？

**关键直觉**：MLP 里所有序列**共享同一组权重** `W_up, W_gate, W_down`（不依赖 B），所以加大 B 能均摊权重的读取开销；而 attention 里，**每个序列都有自己独一份的 Q、K、V（KV cache）**，B 越大读的 KV 也越多，分子分母里的 B 直接抵消。数学上：分子 `4BSTD` 里有 B，分母 `4BSD + 4BTD` 里也有 B，一除就没了。

### 2.4 小结

| 阶段 | 瓶颈 | MLP intensity | Attn intensity |
|---|---|---|---|
| Prefill | compute | B·S (大) | S/2 (大) |
| Generation | **memory** | B (依赖并发) | **≈ 1（无救）** |

---

## 3. Throughput 与 Latency 理论估算

**假设**（教学用的 napkin math）：
- 通信和计算完美重叠
- 忽略各种 overhead
- 推理 memory-bound，只看内存 IO

### 3.1 公式推导

对一个 Transformer：

- **参数数**：`2·V·D + 3·L·D·F + L·(2·D·N·H + 2·D·K·H)`
- **参数占用**：`num_params × 2`（bf16）
- **每序列 KV cache**：`S × K × H × L × 2(K+V) × 2(bf16)`
- **总内存**：`Memory = B × kv_cache_size + parameter_size`
- **延迟**：`Latency = Memory / memory_bandwidth`
- **吞吐**：`Throughput = B / Latency`（每步生成 B 个 token）

### 3.2 Llama 2 13B on H100 实例

配置：`S=1024, D=5120, F=13824, N=40, K=40, H=128, L=40, V=32000, bw=3.35 TB/s`。

| Batch | Memory | Latency | Throughput |
|---|---|---|---|
| B=1 | ~26 GB | ~8 ms/token | ~124 tok/s |
| B=64 | ↑（加 64 份 KV cache） | ↑（读取更多） | ↑↑ |
| B=256 | **~240 GB（爆显存）** | ↑↑ | 边际收益递减 |

### 3.3 关键 Tradeoff 与结论

1. **Batch 越小** → latency 越低，throughput 越差
2. **Batch 越大** → throughput 越高（但受显存和边际收益限制），latency 越差
3. **最简单的并行**：开 M 份模型副本 → latency 不变，throughput × M（推理没有参数更新，无需通信，非常爽）
4. **更复杂的并行**：分片模型、分片 KV cache（参考 Scaling Book）
5. **TTFT 实质上就是 prefill 的耗时**，compute-bound；想降 TTFT 就调小 prefill 的 batch，而 generation 阶段用大 batch 提升吞吐

---

## 4. 有损捷径（Lossy Shortcuts）

核心思路：既然**瓶颈是 KV cache**，那就想办法**压缩 KV cache**，同时**尽量不掉精度**。

### 4.1 减小 KV Cache 的 4 种变体

#### (a) GQA — Grouped-Query Attention

- **MHA**（vanilla）：K = N，query/key/value 头数一样
- **MQA**：K = 1，只保留 1 个 KV 头（过于激进，表达力不足）
- **GQA**：K 在中间，N 个 query 头被分成 K 组，每组共享一个 KV 头

**KV cache 缩小倍数：N/K**。

以 Llama 2 13B 为例：
- 原始：`K=40, B=64`
- GQA (1:5)：`K=8, B=64` → 显存↓、吞吐↑↑
- 进一步：`K=8, B=256` → 之前爆显存的配置现在装得下了，吞吐再涨一截（虽然 latency 变差）

**精度**：基本和 MHA 持平。Llama 3 全面采用 GQA。

#### (b) MLA — Multi-head Latent Attention（DeepSeek-v2）

- **思路**：不减头数，而是把 K、V 向量**投影到低维潜空间** C
- DeepSeek-v2 从 `N×H = 16384` 压到 `C = 512`（非常激进）
- **小 bug**：MLA 与 RoPE 不兼容，所以额外补 64 维给 RoPE → 总共 `512 + 64 = 576` 维
- **精度**：MHA > GQA（MHA 更贵），而 **MLA 比 MHA 略好且便宜得多**

#### (c) CLA — Cross-Layer Attention

- **思路**：GQA 是**跨头**共享 KV，CLA 是**跨层**共享 KV
- 相邻几层复用同一组 K、V 投影（**权重也共享**，保证一致性）
- 在"KV cache size vs 困惑度"的 Pareto 前沿上有改进

#### (d) Local Attention（Sliding Window）

- **思路**：每个 token 只看过去 W 个 token（如 Longformer、Sparse Transformer、Mistral 7B）
- **关键好处**：**KV cache 不再随序列长度增长，大小常数！**
- 多层堆叠后，有效感受野随层数线性扩大
- **风险**：损失远程依赖
- **实践方案**：**全局层 + 局部层交错**
  - Character.AI：每 6 层放 1 层 global + 5 层 local，再叠加 CLA

### 4.2 超越 Transformer：更激进的架构

Transformer 当初是为**训练效率**而设计的，并未考虑推理。自回归 + full attention 是根本瓶颈。

#### (a) State-Space Models (SSM)

- 灵感来自信号处理/控制论的线性动态系统
- 目标：亚二次复杂度建模长序列；对推理的好处是附带的
- **S4**：长上下文合成任务很强，但**在关联召回（associative recall）任务上弱**——给一串 key-value 对，问某个 key 对应的 value，Transformer 轻松解决，S4 不行
- **Mamba**：让 SSM 参数**依赖输入**（input-dependent），1B 规模匹配 Transformer
- **Jamba (AI21)**：1:7 的 Transformer-Mamba 交错，52B MoE（仍需少量 full attention）
- **BASED**：linear attention + local attention
- **MiniMax-01**：linear + full attention，456B MoE，达到 SOTA

**Linear Attention 简述**：把 `softmax(QK^T)` 用 Taylor 展开近似为 `φ(Q)·φ(K)^T`，让它变成**可线性聚合的类 RNN** 结构，KV cache 从 `O(T)` 变 `O(1)`（常数状态）。

**要点**：Linear + Local Attention **正在构成 SOTA 级架构**；多数层可以去掉 full attention，只保留少量层即可。推理时把 `O(T)` 的 KV cache 换成 `O(1)` 的固定状态，效率巨大提升。

#### (b) Diffusion Models（文本）

- 图像生成里很成熟，文本很难做好，但近期有进展
- **思路**：不再自回归，**所有 token 并行生成**，通过多步迭代去噪来 refine
- 从随机噪声开始，迭代细化到最终输出
- **Inception Labs** 的模型在代码 benchmark 上 tokens/s 远超所有 Transformer（甚至 Jamba）
- 好处：完全绕开自回归瓶颈，容易吃满 GPU

**总结**：架构层面的激进变革，才是推理加速的最大红利所在。

### 4.3 Quantization（量化）

**核心思想**：减少数值精度 → 减少显存占用 → 降低访存 → 提升延迟/吞吐。代价是精度损失。

| 格式 | 字节 | 备注 |
|---|---|---|
| fp32 | 4 | 训练参数/优化器状态 |
| bf16 | 2 | 推理默认 |
| fp8 (e4m3) | 1 | H100 支持，训练也能用 |
| int8 | 1 | 范围 [-128, 127]，比 fp8 便宜，仅推理 |
| int4 | 0.5 | 范围 [-8, 7]，更便宜，精度损失更大 |

**两种做法**：
- **QAT (Quantization-Aware Training)**：训练时就带量化，但不好 scale
- **PTQ (Post-Training Quantization)**：训完再量化，跑校准数据估计每层/每 tensor 的 scale 和 zero-point

#### LLM.int8()

- **标准量化**：按绝对值最大值 scale 到 int8，dequantize 再乘回
- **问题**：大模型存在**outlier（离群激活）**，一个巨大的值会把整列的 scale 拉得很离谱
- **解法**：识别 outlier 列，它们**保留 fp16** 单独算；其余走 int8
- 精度好，但比纯 fp16 **慢 15–23%**（混合精度的额外开销）
- 本来的动机不是加速，而是让大模型**装进显存**

#### Activation-Aware Quantization (AWQ)

- **思路**：根据激活幅度，挑出 **0.1–1% 最重要的权重**保留高精度，其余激进量化（到 int3）
- fp16 → int3 能做到 **4× 显存下降、3.2× 加速**

### 4.4 Model Pruning（剪枝）

**核心思想**：把"胖模型"里不重要的部分直接砍掉，然后修补。

**NVIDIA 的算法**：
1. 用小校准集（1024 样本）识别重要的 {层, head, hidden 维度}
2. 删掉不重要的部件，得到小模型
3. **蒸馏**：把原模型当 teacher，小模型当 student 修补精度

**结果**：15B → 8B 几乎无损 MMLU；15B → 4B 有一定下降但可接受。

### 4.5 总结两种加速配方

| 配方 | 步骤 |
|---|---|
| **From scratch** | 1. 定义更快的架构 2. 从头训练 |
| **Distillation** | 1. 定义更快的架构 2. 用原模型权重初始化（即使架构不同） 3. 用蒸馏修复 |

---

## 5. 无损捷径：Speculative Sampling（投机采样）

### 5.1 核心洞察

- **Prefill**（一次给你一堆 token 的概率）= compute-bound = 快
- **Generation**（一次只生成一个）= memory-bound = 慢
- ⇒ **"检查（verify）比生成（generate）快"**

### 5.2 算法

- 用一个**便宜的 draft model p** 自回归生成 K 个候选 token（小模型快）
- 用**目标 model q** 一次性并行验证这 K 个 token（prefill 模式）
- 按某种概率决定接受/拒绝

对每个候选 token `x`：
- 以 `min(1, q(x)/p(x))` 的概率**接受**
- 若拒绝，则从**残差分布** `max(q-p, 0)`（归一化后）中采样一个替代 token
- 细节补丁：为了不陷入死循环，**至少保证生成一个 token**

### 5.3 为什么这是精确采样？（Vocab={A, B} 举例）

设 `p(A) > q(A)`（draft 过采样 A），则 `p(B) < q(B)`。残差分布 `max(q-p, 0) = [0, 1]`（全权重给 B）：

- `P[最终采 A] = p(A) · (q(A)/p(A)) + p(B) · 1 · 0 = q(A)` ✓
- `P[最终采 B] = p(B) · 1 + p(A) · (1 - q(A)/p(A)) · 1 = p(B) + p(A) - q(A) = 1 - q(A) = q(B)` ✓

最终分布**严格等于 q**——与 Metropolis-Hastings / rejection sampling 同源。

### 5.4 实践与扩展

- **常见搭配**：70B target + 8B draft；或 8B target + 1B draft
- draft 越接近 target，接受率越高（可用蒸馏拉近）
- 典型加速：**~2×**，且**精度完全无损**
- **Medusa**：draft 一次并行生成多个 token
- **EAGLE**：draft 吃 target 的高层特征，不再独立，质量更高

### 5.5 总结

- **数学上保证**是从 target 的严格精确采样
- 利用了"**检查 vs 生成**"的不对称性
- draft model 设计空间极大（架构、量化、蒸馏都能叠加）

---

## 6. 动态负载处理

真实线上流量与训练不同：
1. 请求**到达时间不同**（等 batch 会让早到请求 latency 变差）
2. 请求**前缀可能共享**（system prompt、同一 prompt 多样采样）
3. 请求**长度不一**（padding 浪费算力）

### 6.1 Continuous Batching（Orca, 2022）

#### 问题 1：Static batching 浪费
 
经典静态 batching：等一批请求齐了一起处理，先完成的要空等到最慢的完成。

#### 解法：Iteration-level Scheduling

- 每生成一个 token 就**把控制权还给 scheduler**
- **新请求随时加入**当前 batch，不必等"下一班车"
- 已完成的请求立即出队，腾位置

#### 问题 2：不同长度无法直接 batch

#### 解法：Selective Batching

- **Attention 计算**：每个序列**单独**处理（长度不同，形状对不上）
- **非 attention（MLP、LayerNorm 等）**：把各序列的 token 沿 batch 维**flatten 拼接**成 `[sum(T_i), H]` 的大 tensor，因为这些操作**不跨 token 交互**，可以完全并行

### 6.2 PagedAttention（vLLM, 2023）

#### 问题：KV Cache 内存碎片化

传统做法：为每个请求预分配一块**连续**的 KV cache 空间（按 max length）。

- **内部碎片**：实际生成 token 数 < 预分配量
- **外部碎片**：请求之间的空白

#### 解法：借鉴操作系统的**分页**

- 把 KV cache 切成**非连续的固定大小 block**
- 每个序列维护一个**block 表**，按需从 block 池中取空闲块
- 这样物理上不连续，逻辑上连续，碎片消失

#### 进阶：Copy-on-Write 共享

- 多个请求共享系统 prompt → 共享底层 block（引用计数）
- 一旦某个请求写入新内容 → 该 block 复制一份，引用计数减一 → 各自发展

#### 其他 vLLM 优化

- 融合 block 读取 + attention 的 kernel，减少 kernel launch 开销
- 使用最新 kernel（FlashAttention、FlashDecoding）
- 用 CUDA Graph 进一步削减 kernel launch

**一句话**：**操作系统学过的技巧（paging、COW）可以原样搬进推理系统**。

---

## 7. 全局总结

- **推理极其重要**：实际使用、评测、test-time compute、RL 采样都离不开
- **与训练特征不同**：**memory-limited**、**dynamic**
- **加速技术全景**：
  - 架构层：GQA / MLA / CLA / Local Attn / SSM(Mamba) / Diffusion
  - 数值层：Quantization（LLM.int8、AWQ）
  - 模型层：Pruning + Distillation
  - 采样层：Speculative Sampling（精确无损）
  - 系统层：Continuous Batching + PagedAttention
- **最大红利在于"新架构"**：narrow 地看"我要跑这个模型多快"意义有限，核心问题是"**给定算力预算，我能交付多高精度**"——绕开 full attention 自回归这堵墙本身就是最大优化

---

## 附：关键公式速查

| 场景 | Intensity | 是否 memory-bound |
|---|---|---|
| 一次 matmul | B | 若 B < 295（H100）则 memory-bound |
| MLP Prefill | B·S | 否 |
| MLP Generation | B | 若并发 B 不足则 memory-bound |
| Attn Prefill | S/2 | 否 |
| Attn Generation | S/(S+1) ≈ 1 | **永远是** |

> H100 accelerator intensity ≈ 989 TFLOPs ÷ 3.35 TB/s ≈ **295**
