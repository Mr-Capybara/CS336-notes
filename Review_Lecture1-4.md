# CS336 阶段性复习笔记：Lecture 1–4

> Stanford CS336 - Language Modeling from Scratch (Spring 2025)
>
> 覆盖：Overview + Tokenization (L1)、PyTorch + Resource Accounting (L2)、Architecture + Hyperparameters (L3)、Mixture of Experts (L4)
>
> 这份笔记面向**复习**：只保留真正值得反复回看的核心知识、公式、表格、设计决策背后的理由。不重复细节例子，需要细看时回到各讲笔记。

---

## 第 0 章 全局主线

整门课（至少前 4 讲）围绕一个中心问题：

> **给定有限的 compute 和 data，怎样训出最好的模型？**
>
> `Accuracy = Efficiency × Resources`

Bitter Lesson 的正确解读不是"算力至上"，而是 **"Algorithms at scale is what matters"**。

贯穿四讲的三条元经验：

1. **Residual identity 要干净** —— pre-norm、不在残差主路径上放非线性
2. **归一化控 activation scale** —— LayerNorm/RMSNorm 是稳定性神器，哪里容易炸就往哪里塞
3. **FLOPs ≠ runtime** —— memory movement（HBM 访问、all-to-all 通信）往往才是真瓶颈

---

## 第 1 章 Tokenization（L1 后半）

### 1.1 Tokenizer 的本质

- 输入：Unicode 字符串；输出：整数序列
- 要求**可逆**（encode ↔ decode）
- `vocab_size` = 整数的取值范围

### 1.2 四种方案的取舍

| 方案 | 优点 | 致命缺点 |
| --- | --- | --- |
| Character-based | 直观 | 词表巨大（Unicode 10 万+），稀有 emoji 浪费词表 |
| Byte-based | 词表只有 256，优雅 | **压缩比 = 1**，序列太长 → attention $O(n^2)$ 爆炸 |
| Word-based | 符合人类直觉 | 词表无界，OOV 要 UNK，粒度不自适应 |
| **BPE** | 根据语料**自适应**分配词表 | 实现细节复杂、数字切分反常等小毛病 |

### 1.3 BPE 算法核心

1. 先把文本转成字节序列
2. 反复执行：**统计最频繁的相邻 token 对 → 合并为新 token**，记录到 merges 表
3. Encode 时对新输入**按训练时记录的 merge 顺序 replay**

**关键工程细节**：

- 先用正则 pre-tokenize（切成单词块），BPE 只在块内跑，防止跨词合并
- 空格通常放在 token **前面**（`" hello"`，而不是 `"hello "`）
- **`"hello"` 和 `" hello"` 是完全不同的 token**，会导致各种奇怪问题
- 数字按从左到右切分，不按千分位语义

### 1.4 为什么 BPE 还没被淘汰

理想情况是 byte-level 直接建模（优雅、无 OOV、多语友好），但 attention $O(n^2)$ 和训练效率还撑不起 Frontier 规模。BPE 这个 1994 年发明的数据压缩算法，到今天依然是 LM 的标配。

---

## 第 2 章 Resource Accounting（L2 核心）

这一章是全课**最重要的"数感基础"**，后面所有架构/并行/推理讨论都建立在这上面。

### 2.1 浮点精度选择

| 类型 | 位数 (S/E/M) | 动态范围 | 典型用途 |
| --- | --- | --- | --- |
| FP32 | 1/8/23 | 极大 | **参数、optimizer state、梯度累积** |
| FP16 | 1/5/10 | 小（易 underflow） | 不推荐（BF16 全面更优） |
| **BF16** | 1/**8**/7 | ≈ FP32 | **前向/反向 MatMul** |
| FP8 (E4M3/E5M2) | 1/4/3 或 1/5/2 | 非常有限 | H100+ 推理、激进训练 |

**核心原则**：

> **需要长期积累的 → FP32；一次性计算的 → BF16。**
>
> BF16 的优势不是精度而是**和 FP32 一样的动态范围**——不会 underflow。

### 2.2 内存预算

**每个参数的内存 = 参数本身 + 梯度 + optimizer state**：

| 优化器 | 每参数字节数（纯 FP32） |
| --- | --- |
| SGD | 4 + 4 = 8 |
| Adagrad / RMSProp | 4 + 4 + 4 = 12 |
| **AdamW** | 4 + 4 + 4 + 4 = **16** |

**模型总内存**：

$$
\text{Total} = \text{Params} + \text{Gradients} + \text{Optimizer State} + \text{Activations}
$$

Activations 取决于 batch size × seq length × num_layers，通常不能忽略。

**BF16 混合精度不省内存的陷阱**：BF16 参数（2）+ BF16 梯度（2）+ FP32 master 参数副本（4）+ FP32 $m,v$（8）= 仍然 16 bytes。BF16 省的是**算力**，不是显存。

### 2.3 FLOPs 计算：三条核心公式

**单次矩阵乘法**（$M\times K$ 乘 $K\times N$）：

$$
\text{MatMul FLOPs} = 2 \cdot M \cdot N \cdot K
$$

**Transformer 训练的 6N 法则**：

$$
\text{Forward FLOPs} \approx 2 \cdot (\text{tokens}) \cdot (\text{params})
$$

$$
\text{Backward FLOPs} \approx 4 \cdot (\text{tokens}) \cdot (\text{params})
$$

$$
\boxed{\text{Total Training FLOPs} \approx 6 \cdot (\text{tokens}) \cdot (\text{params})}
$$

**Backward 为什么是 Forward 的 2 倍**：每一层 backward 要算**两个**梯度：

- **参数梯度** $\partial L / \partial W$（给 optimizer 用）—— 1 个 MatMul
- **输入梯度** $\partial L / \partial X$（继续向上反传）—— 1 个 MatMul

所以每层 backward = 2 × forward。

### 2.4 MFU：硬件利用率

$$
\text{MFU} = \frac{\text{model FLOPs} / \text{actual time}}{\text{peak hardware FLOPs/s}}
$$

**经验基准**：

- MFU > 0.5：不错
- MFU ≈ 0.8+：优秀（纯 MatMul 场景）
- MFU ≈ 0.05：硬件严重浪费

**实战陷阱**：

- Spec sheet 上带星号的 FLOPs 是 2:4 稀疏数，**实际 dense 取一半**
- H100 的 FP32 只有 67.5 TFLOPS，BF16 能到 ~990 TFLOPS → 差 **15 倍**，训练必须 BF16
- 测时间必须 `torch.cuda.synchronize()`，否则测到的是 CPU 发指令时间

### 2.5 PyTorch 的三个关键机制

**① View 是零拷贝**：

- `x[0]`、`x[:,1]`、`.view()`、`.transpose()` 都只改 stride/offset，**共享 storage**
- 修改一个会影响另一个（踩过一次就记住了）
- `.transpose()` 后变 non-contiguous，要 `.contiguous()`（**会拷贝**）才能再 view

**② Einops + Jaxtyping**：

- 维度命名 > 用 `-1, -2` 位置索引
- `einsum("batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")`：**出现在输出里就保留、不出现就 contract（求和）**
- `rearrange` 的 `(heads hidden1)` 语法是 multi-head attention 的标配

**③ Optimizer 的 state 字典**：

- `self.state[p]` 跨 step 持久化每个参数的状态
- `optimizer.zero_grad(set_to_none=True)` 比设 0 更省内存

### 2.6 两个 Napkin Math 案例（必须会反算）

**70B × 15T tokens × 1024 H100 要多久？**

$$
\text{Total FLOPs} = 6 \times 70 \times 10^9 \times 15 \times 10^{12} = 6.3 \times 10^{24}
$$

$$
\text{cluster per-day FLOPs} = 1024 \times \underbrace{990 \times 10^{12}}_{\text{H100 BF16 dense}} \times \underbrace{0.5}_{\text{MFU}} \times 86400 \approx 4.4 \times 10^{22}
$$

$$
\text{days} = 6.3 \times 10^{24} / 4.4 \times 10^{22} \approx 144
$$

**8 张 H100 能训多大模型？**

$$
N = \frac{8 \times 80 \times 10^9 \text{ bytes}}{16 \text{ bytes/param}} = 40 \text{B}
$$

---

## 第 3 章 现代 Transformer 架构（L3 核心）

### 3.1 LLaMA-like 四大现代化改动

相对 2017 年原版：

1. **Pre-norm**（替换 post-norm）
2. **RoPE**（替换 sin/cos PE）
3. **SwiGLU**（替换 ReLU）
4. **去掉 Linear 和 LN 的 bias 项**

### 3.2 Pre-norm vs Post-norm（**共识度最高**）

Post-norm：

$$
x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))
$$

Pre-norm：

$$
x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))
$$

**核心差别**：pre-norm 把 LayerNorm 从残差主路径移开。residual 通路保持 identity，梯度传播非常干净。

**优势**：

- 梯度范数跨层稳定
- loss spike 明显更少，可以用更大 LR
- **去掉了 warmup 的必要性**

**Double norm**（新玩法）：block 的 sub-layer 前后都加 norm（但仍在 residual 外），Grok / Gemma 2 / OLMo 2 在用。

### 3.3 LayerNorm vs RMSNorm

**LayerNorm**：

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot \gamma + \beta
$$

**RMSNorm**：

$$
y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}} \cdot \gamma
$$

**RMSNorm 比 LN 少了两件事**：不减均值、不加 bias $\beta$。

**为什么更快（关键 insight）**：

- MatMul 占了 Transformer **99.8% 的 FLOPs**
- 但 Norm 操作占了 **25% 的 runtime** —— 因为是 memory-bound 的
- RMSNorm 少搬一次 activation，wallclock 真实变快

→ **不能只看 FLOPs 评估架构改动**。

### 3.4 激活函数：SwiGLU 家族

| 函数 | 形式 |
| --- | --- |
| ReLU | $\max(0, x W_1) W_2$ |
| GeLU | $x \cdot \Phi(x)$ 代替 ReLU |
| **SwiGLU** | $\bigl(\text{Swish}(x W_1) \otimes x V\bigr) W_2$ |

SwiGLU 多了一个 gating 参数 $V$，实际中会把 $d_{ff}$ 从 $4 d_{model}$ 缩小到 $\frac{8}{3} d_{model}$ 保持总参数量。

**一句话记忆**：*GLU 反复被证明**小幅**但**稳定**更好，现代模型基本都是 SwiGLU / GeGLU。

### 3.5 位置编码：RoPE

**理想的相对位置编码应满足**：

$$
\langle f(x, i), f(y, j)\rangle = g(x, y, i - j)
$$

即 attention 只依赖 $i - j$。sine / learned / relative 三种历史方案都不完全满足。

**RoPE 的关键几何事实**：**内积对旋转不变**。

把 $d$ 维向量切成 $d/2$ 对 2D 坐标，第 $k$ 对用固定频率 $\theta_k$、位置 $m$ 做 2D 旋转：

$$
R_{m,k} = \begin{pmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{pmatrix}
$$

**实现要点**：

- RoPE 作用在 **Q 和 K 上**，不作用在 embedding 上
- $\theta_k$ **不可学**，是固定 schedule
- 每层 attention 都重新旋转（反复施加位置不变性）
- base frequency 是长上下文外推的关键接口

### 3.6 关键超参数共识

| 超参数 | 共识值 | 备注 |
| --- | --- | --- |
| $d_{ff} / d_{model}$ | 4（非 gated） / $\frac{8}{3}$（gated） | Kaplan 2020 在 [1, 10] 区间是平坦谷底 |
| $n_{\text{heads}} \times \text{head\_dim}$ | $= d_{model}$ | 总维度守恒，切得越细越多 head |
| Aspect ratio $d_{model} / n_{layer}$ | **100–200** | Kaplan 表明最优点**跨规模稳定** |
| Vocab size | 单语 30–50k，多语 100–250k | Cohere、多语模型必须大 |
| Dropout | 基本 0 | pre-training 一遍过，没有 overfit |
| Weight decay | 仍然用 | **不是为了正则化**，是和 LR schedule 的优化动力学耦合 |

**T5 11B 是出名的反例**：$d_{ff}/d_{model} = 64$（理由是大矩阵乘在 TPU 上更快），T5 v1.1 自己就改回 2.5 了。

### 3.7 稳定性三件套（**softmax 是万恶之源**）

Transformer 里有两个 softmax，对应两种稳定性 intervention：

**① Z-loss**（管输出层 softmax 的分母）：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \cdot \bigl(\log Z(x)\bigr)^2
$$

PaLM 用 $\alpha = 10^{-4}$。让归一化常数 $Z$ 保持在 1 附近，softmax 数值极稳。

**② QK-Norm**（管 attention softmax 的输入幅度）：

$$
\text{Attn} = \text{softmax}\left(\frac{\text{Norm}(Q) \text{Norm}(K)^\top}{\sqrt{d}}\right) V
$$

直接对 Q、K 做 RMSNorm 再算内积，从根源上防止 logit 爆炸。推理时**必须保留**，因为有可学参数 $\gamma$。

**③ Logit soft-cap**：`softcap · tanh(logits / softcap)`。Gemma 2 在用，但 Nvidia ablation 显示 PPL 反而变差，**不是主流**。

### 3.8 Attention 变体：KV cache 救星

**为什么要变**：训练是 compute-bound，**推理（decode 阶段）是 memory-bound** —— 每生成一 token 都要把整个 KV cache 从 HBM 里搬出来。

**算术强度对比**：

- 训练：$\left(\frac{1}{h} + \frac{1}{bn}\right)^{-1}$，$h, b, n$ 都大 → 打满 GPU
- 推理（带 KV cache）：$\left(\frac{n}{d} + \frac{1}{b}\right)^{-1}$，$n/d$ 常压制利用率

**三种主流 attention**：

| 方案 | Q heads | KV heads | KV cache 开销 |
| --- | --- | --- | --- |
| MHA | $h$ | $h$ | 最大 |
| **GQA**（主流） | $h$ | $g$ 组（$1 < g < h$） | 可调折中 |
| MQA | $h$ | 1 | 最小，但 PPL 略升 |

LLaMA-2/3、大部分现代模型都用 GQA。

**长上下文方案 —— Interleaved Full + Local**（Cohere Command A / LLaMA 4 / Gemma）：

```
4k+0:  Full attention, NoPE
4k+1,2,3: Sliding window, RoPE
```

- 4 层里只有 1 层全局，**compute 大降**
- 全局层 NoPE（避免 RoPE 外推精度问题）
- 局部层 RoPE 保留精细位置

---

## 第 4 章 Mixture of Experts（L4 核心）

### 4.1 MoE 是什么

> 把 transformer block 里那一个大 FFN 换成：**N 个（更小或等大的）FFN 副本 + 一个 router**，每次只激活 top-k 个。

**核心价值**：**相同 FLOPs，更多参数**。因为没激活的 expert 完全不参与 forward。

**命名误导**：不是"写代码专家 / 翻译专家"，只是**稀疏激活**的架构。专业化是 router 学出来的，**不是人工设计的**。

### 4.2 为什么 2025 年所有人都在做

- 同 FLOPs 下 MoE 训练 loss 更低（Fedus 2022、OLMoE 验证）
- 同 compute 下训练速度 ~7× 加速
- Active params 对推理极其友好（DeepSeek V2 的招牌曲线）
- **Expert parallelism** 给了多一个并行维度

### 4.3 Top-k Routing 公式（必须记住）

设 token 输入 $u_t$，expert $i$ 的 gate 向量 $e_i$：

$$
s_{i,t} = \text{Softmax}_i(u_t^\top e_i)
$$

$$
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{TopK}(\{s_{j,t}\}, K) \\ 0, & \text{otherwise} \end{cases}
$$

$$
h_t = u_t + \sum_{i=1}^{N} g_{i,t} \cdot \text{FFN}_i(u_t)
$$

**关键设计点**：

- Router **就是一个向量内积**，没必要搞复杂 —— router 学不好的根本原因是**梯度信号弱**（没选中的 expert 没有梯度）
- Softmax 在 top-k **之前**（DeepSeek V1/V2、Qwen）还是**之后**（Mixtral、V3）都可以，差别不大
- **绝不能去掉 top-k** —— 否则训练时要 forward 所有 expert，稀疏性的意义全没了
- K 的选择：Switch Transformer 用 1，多数模型用 2，DeepSeek V3 用 8

### 4.4 Fine-grained + Shared Experts（DeepSeek 的创新）

**Fine-grained experts**：把 FFN 切小（1/4 ~ 1/14 标准大小），数量成倍增加 → 专业化更好，FLOPs 不变。**基本是无脑采用的改动**。

**Shared experts**：0~2 个每次必激活的 expert，处理公共模式。OLMoE ablation 显示收益不明显，可选。

### 4.5 训练 MoE：不可微路由的三条路

| 路线 | 实际采用 |
| --- | --- |
| RL 学路由 | 正确但贵，基本没人用 |
| 随机扰动做 exploration | 早期 Google 用过，现已放弃 |
| **启发式 balance loss** | **实际方案** |

**Switch Transformer 的 balance loss**：

$$
\mathcal{L}_{\text{balance}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$

- $f_i$：这 batch 中被路由到 expert $i$ 的 token 比例（实际）
- $P_i$：router softmax 给 expert $i$ 的平均概率（期望）

**梯度直觉**：$\partial \mathcal{L} / \partial P_i = \alpha N \cdot f_i$，分到 token 越多的 expert 梯度越被往下压 → 自动均衡。

**如果不加 balance loss**：几乎必然退化到 1~2 个 expert 吃所有 token，其它 expert 死掉。

**DeepSeek 在 Switch 基础上的扩展**：

- V1：加了 **device-level balance**
- V3：提出 **aux-loss-free balancing**，用可学 bias $b_i$ + online learning 替代 aux loss。但又补了一个 **per-sequence aux loss** 防止推理时 OOD 序列的不均衡

### 4.6 系统与稳定性

**Expert parallelism**：每个 expert 独占一台设备，MoE 层做两次 **all-to-all 通信**（dispatch + combine）。

**MegaBlocks**：一台设备上多个 expert 时，用 block-sparse MatMul 一次算完，避免小 matmul 低效。

**Token dropping**：每个 expert 有 load factor 上限，超了就丢。**batch 里别人的 query 可以让你的 token 被 drop** —— GPT-4 temperature=0 仍不稳定的一个可能解释。

**router 稳定性**：

- router softmax 用 **FP32** 算
- 加 **z-loss**（和 3.7 一样的那个），防止 loss spike

**Fine-tuning 过拟合**：MoE 参数量巨大，SFT 容易过拟合。两个解法：Zoph 方案（交替 dense/MoE 层，只 finetune dense）、DeepSeek 方案（140 万条 SFT 堆数据）。

### 4.7 DeepSeek V1 → V2 → V3 演进（必掌握）

| 版本 | 规模 | 新增 |
| --- | --- | --- |
| **V1** (DeepSeekMoE) | 16B / 2.8B active | 64 fine-grained + 2 shared，标准 top-k，expert + device balance loss。**架构就已基本最终形态** |
| **V2** | 236B / 21B active | **Top-M device routing**（先选 top-M 设备再在设备内选 top-k expert，控通信）+ communication balance loss |
| **V3** | 671B / 37B active | **Aux-loss-free balance**（bias $b_i$ + online learning）+ sequence-wise aux loss + sigmoid/softmax 调整 |

### 4.8 DeepSeek V3 的非 MoE 创新（彩蛋）

**MLA（Multi-head Latent Attention）**：

$$
c_t^{KV} = W_{DKV} h_t, \quad K = W_{UK} c_t^{KV}, \quad V = W_{UV} c_t^{KV}
$$

KV cache 只存**低维** $c_t^{KV}$，推理时再上投影。看似多一个 $W_{UK}$，但可以 **merge 进 $W_Q$**（关联性 $Q^\top K = \langle h W_Q W_{UK}, c_t^{KV}\rangle$），**运行时零额外 MatMul**。

RoPE 冲突问题（$R_q R_k$ 夹在中间导致无法 merge）的解法：**保留少量非 latent 的 key 维度**承载 RoPE，其余走 latent 压缩。

**MTP（Multi-Token Prediction）**：训练时用轻量级单层 transformer 预测再下一个 token（T+2），V3 实际上只展开了 1 步。

---

## 第 5 章 跨讲知识关联图

这些概念在不同讲中反复出现，**它们之间的关联**是考试、面试、项目里真正有用的东西。

### 5.1 Softmax 是所有稳定性问题的源头

| 位置 | 病症 | 药方 | 出现在 |
| --- | --- | --- | --- |
| 输出层 softmax | 分母爆炸 / 消失 | **Z-loss** | L3 |
| Attention softmax | logit 过大 | **QK-Norm** + FP32 计算 | L3 |
| Attention softmax | logit 过大 | **logit soft-cap**（tanh 限幅） | L3，争议 |
| MoE router softmax | spike | **FP32 计算 + z-loss** | L4 |

### 5.2 "FLOPs ≠ Runtime" 在哪些地方显现

| 场景 | 真正的瓶颈 |
| --- | --- |
| Norm 操作 | memory-bound，占 25% runtime 但只 0.17% FLOPs |
| 推理 decode | memory-bound（KV cache 搬运） |
| MoE | all-to-all 通信 |
| 去 bias 项 | 主要为了减少 memory movement |

**推论**：架构改动必须同时看 FLOPs 和 memory access pattern。

### 5.3 稀疏性贯穿四讲

| 类型 | 含义 |
| --- | --- |
| **Token 稀疏** (L1) | 词表大，每个 token 的 embedding 只被少数序列激活 |
| **Activation 稀疏** (L3) | SwiGLU 的 gating 本质上是软稀疏 |
| **Attention 稀疏** (L3) | sliding window、interleaved full + local |
| **Expert 稀疏** (L4) | MoE 的 top-k 硬稀疏 |

趋势：**越往后越稀疏**，每一种稀疏都在尝试"以更少计算获得同等能力"。

### 5.4 为什么 Pre-norm 和 MoE 都用 residual identity

| 机制 | 作用 |
| --- | --- |
| Pre-norm | LN 不在 residual 主路径，梯度传播干净 |
| MoE 的残差 $h_t = u_t + \sum g_{i,t} \text{FFN}_i(u_t)$ | 即使所有 expert 被 gate 掉，残差也能把信息传到上层 |
| MoE token dropping | 被 drop 的 token 走残差连接，保证 forward 不崩 |

**共同原则**：**能走 identity 就走 identity，把非线性作为可选的加法项**。

### 5.5 Attention 和 MoE 的 router 的相似性

都是**用内积计算相关性 → softmax → 选出 top 几个 → 加权组合**：

- Attention：query 找 keys，加权求和 values
- MoE router：token 找 experts，加权求和 FFN 输出

本质都是"可学的 soft 路由"。这解释了：

- 为什么 MoE router 用向量内积就够，不需要 MLP
- 为什么 attention 也能稀疏化（sliding window = hard top-k by position）

---

## 第 6 章 公式速查表

| 公式 | 含义 | 出处 |
| --- | --- | --- |
| `mem = numel × dtype_bytes` | Tensor 内存 | L2 |
| `MatMul FLOPs = 2MNK` | 矩阵乘法计算量 | L2 |
| `forward ≈ 2 × tokens × params` | 前向 FLOPs | L2 |
| `backward ≈ 4 × tokens × params` | 反向 FLOPs（每层 2 个 MatMul） | L2 |
| `total ≈ 6 × tokens × params` | 训练总 FLOPs（6N 法则） | L2 |
| `MFU = model_flops/s ÷ peak_flops/s` | 硬件利用率 | L2 |
| `AdamW: 16 bytes/param` | 每参数显存 | L2 |
| $y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2+\varepsilon}}\gamma$ | RMSNorm | L3 |
| $\text{FF}_{\text{SwiGLU}} = (\text{Swish}(xW_1) \otimes xV) W_2$ | SwiGLU | L3 |
| $d_{ff} = \frac{8}{3} d_{model}$ | gated FFN 默认 | L3 |
| $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha (\log Z)^2$ | z-loss | L3 |
| $h_t = u_t + \sum g_{i,t} \text{FFN}_i(u_t)$ | MoE 层 | L4 |
| $\mathcal{L}_{\text{balance}} = \alpha N \sum f_i P_i$ | Switch balance loss | L4 |

---

## 第 7 章 常考/常问问题（自测）

这些问题如果能**不看资料**流畅回答，说明这四讲掌握了：

1. 为什么训练 FLOPs 是 `6 × tokens × params`？backward 的 4 从哪来？
2. BF16 和 FP16 都是 16 位，为什么选 BF16 不选 FP16？
3. AdamW 一个参数占多少字节？如何从 H100 80GB 推出最大模型规模？
4. Pre-norm 相对 post-norm 的本质区别是什么？为什么能去掉 warmup？
5. RMSNorm 的 FLOPs 节省几乎为零，为什么实际能加速？
6. RoPE 为什么能"只依赖相对位置"？用一句话解释几何直觉。
7. `d_ff / d_model` 为什么 gated 版本是 8/3 而不是 4？
8. Weight decay 现在真正的作用是什么（提示：不是正则化）？
9. Z-loss 和 QK-Norm 分别解决哪个 softmax 的稳定性问题？
10. 为什么 attention 在推理时是 memory-bound，GQA 如何缓解？
11. MoE 为什么"同 FLOPs 更多参数"？top-k 为什么不能去掉？
12. 不加 balance loss 的 MoE 会发生什么？DeepSeek V3 的 aux-loss-free 机制是什么？
13. MLA 如何在压缩 KV cache 的同时避免增加运行时 MatMul？RoPE 为什么会和 MLA 冲突？
14. Fine-grained experts 和 shared experts 各自解决什么问题？

---

## 第 8 章 下一阶段预告（L5+）

- **Systems**：GPU kernel（Triton）、Flash Attention、并行策略（DP / TP / PP / FSDP / EP）、推理优化（speculative decoding）
- **Scaling Laws**：Chinchilla optimal、小规模→大规模外推
- **Data**：Common Crawl 处理、filter、deduplication
- **Alignment**：SFT、DPO、GRPO

前 4 讲搭好的是**"单节点训练一个 dense / MoE transformer"的全部基础**。后面都是在这基础上：

- 横向扩展到多机多卡
- 决定参数量、数据量、超参
- 决定训什么数据
- 把 base model 对齐成有用的工具
