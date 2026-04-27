# CS336 Lecture 3: LLM 架构与超参数

> *Everything you didn't want to know about LM architecture and training*
>
> 本节课主题：第一是 hands-on，第二是 learn from others' experience。我们不可能亲手训练所有 LLM，所以通过分析近两年 19+ 个新 dense 模型的「趋同演化」来理解哪些选择是关键的。

---

## 0. 路线图

本课分三块，外加一个扩展话题：

1. **架构变体**：activations、FFN、norm、serial vs parallel、position embeddings
2. **超参数**：$d_{ff}/d_{model}$、head dim、aspect ratio、vocab、regularization
3. **稳定性技巧**：z-loss、QK-norm、logit soft-capping
4. **（扩展）注意力变体**：MQA / GQA、sparse / sliding window、interleaved attention

三条贯穿全课的可泛化经验：

- **Residual 要干净**：identity 通路尽量不被打断（pre-norm 的本质）
- **控制 activation scale**：LayerNorm / RMSNorm 之类的归一化对稳定性极其有效
- **不要只盯着 FLOPs**：memory movement（HBM 访问）才是 runtime 的真凶

---

## 1. Transformer 回顾：从原始版到现代版

### 1.1 原始 Transformer（2017）

- Position embedding：正余弦（sin/cos）
- FFN 激活：ReLU
- Norm：**post-norm + LayerNorm**（norm 在 residual add 之后）

### 1.2 现代「LLaMA-like」变体（作业要你实现的）

相对原始版的四个修改：

1. LayerNorm 挪到 block 前面（**pre-norm**）
2. 位置编码用 **RoPE**（Rotary Position Embedding）
3. FFN 用 **SwiGLU** 代替 ReLU
4. Linear 层和 LayerNorm **去掉 bias**

2023 年后的新模型几乎都收敛到这套组合；课程里 19 个近期模型的架构表里，position embedding 那列几乎清一色是 RoPE。

---

## 2. 架构变体

### 2.1 Pre-norm vs Post-norm（**所有人都同意的一件事**）

post-norm 原版：

$$
x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))
$$

pre-norm 现代版：

$$
x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))
$$

**核心差别**：pre-norm 让 LayerNorm **不在 residual 主路径上**。残差通路保持为纯 identity，从顶到底无任何非线性，这让梯度传播非常干净。

为什么 pre-norm 更好：

- **梯度衰减**：post-norm 里梯度在层间会被放大/衰减，pre-norm 下梯度范数基本恒定（Xiong 2020）
- **Loss spike 更少**：post-norm 要配 warmup 才稳，pre-norm 更稳，能用更大 LR（Salazar & Nguyen 2019）
- **去掉 warmup**：pre-norm 原本的宣传点就是「我不需要学习率 warmup」

现代视角：pre-norm 的价值是**训练稳定性**，尤其是大模型 + 大 LR 时。

**例外**：OPT-350M 是 post-norm（大概率是训歪了），BERT 是 post-norm（老模型）。

### 2.2 Double Norm（近期新玩法）

既然 residual 里放 norm 不好，那能不能在 sub-layer **之后**再加一个 norm（但仍然在 residual 之外）？

```
x ---+---> LayerNorm ---> SubLayer ---> LayerNorm ---> + ---> x'
     |                                                  |
     +--------------------------------------------------+
```

- **Grok、Gemma 2**：前后都加 norm
- **OLMo 2**：只在 sub-layer 之后加 norm（即非残差的 post-norm）

据报告更稳，但不算 post-norm（不破坏 residual identity）。

### 2.3 LayerNorm vs RMSNorm（几乎全员切到 RMSNorm）

**LayerNorm**：

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot \gamma + \beta
$$

**RMSNorm**：

$$
y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}} \cdot \gamma
$$

RMSNorm 相对 LayerNorm 的变化：

- 不减均值
- 不加 bias $\beta$
- 效果基本一致，但更快、参数更少

**"RMSNorm 更快"这件事要掰开讲**：

矩阵乘占了 Transformer **99.8% 的 FLOPs**（Ivanov 2023 的 profiling 表），所以 LayerNorm / softmax / 激活这些"非矩阵乘"操作加起来只占 0.17% FLOPs。如果只看 FLOPs，RMSNorm 省不了多少。

**但** normalization 操作占了 **25% 的 runtime**，因为它们 memory-bound：搬运 activation 的开销远大于算一次 mean。RMSNorm 少一次均值统计、少一个 bias 参数，**数据搬运少**，所以真实 wallclock 更快。

> 关键 insight：架构设计不能只看 FLOPs，必须考虑 memory movement。

Narang et al. 2020 的 ablation：vanilla Transformer 3.5 step/s，RMSNorm 版 3.68 step/s，且 loss 还更低——免费的午餐。

用 RMSNorm 的模型：LLaMA 家族、PaLM、Chinchilla、T5。
仍用 LayerNorm：GPT-1/2/3、OPT、GPT-J、BLOOM、Cohere Command / R+（不太清楚为什么）。

### 2.4 去掉 bias 项

原版 FFN：

$$
\text{FFN}(x) = \sigma(xW_1 + b_1) W_2 + b_2
$$

现代实现（非 gated）：

$$
\text{FFN}(x) = \sigma(xW_1) W_2
$$

理由和 RMSNorm 一样：

- **省 memory movement**
- **优化更稳**（empirical 发现，机制不完全清楚）

### 2.5 激活函数：ReLU / GeLU / *GLU 大动物园

**ReLU**：

$$
\text{FF}(x) = \max(0, xW_1) W_2
$$

用户：原始 Transformer、T5、Gopher、Chinchilla、OPT。

**GeLU**（Gaussian Error Linear Unit）：

$$
\text{GELU}(x) := x \cdot \Phi(x)
$$

其中 $\Phi$ 是标准正态 CDF。相比 ReLU 在 0 附近更平滑。

用户：GPT-1/2/3、GPT-J、GPT-NeoX、BLOOM。

**Gated Linear Units（*GLU 家族）**：

把 FFN 的第一步做一个 **entry-wise 的 gating**：

$$
\max(0, xW_1) \;\;\longrightarrow\;\; \max(0, xW_1) \otimes (xV)
$$

多了一个参数 $V$。完整 ReGLU：

$$
\text{FF}_{\text{ReGLU}}(x) = \big(\max(0, xW_1) \otimes xV\big) W_2
$$

换掉 ReLU 就是 GeGLU / SwiGLU：

- **SwiGLU**：用 $\text{Swish}(x) = x \cdot \sigma(x)$ 作为非线性
- **GeGLU**：用 GeLU 作为非线性

SwiGLU FFN：

$$
\text{FF}_{\text{SwiGLU}}(x) = \big(\text{Swish}(xW_1) \otimes xV\big) W_2
$$

> **关于 Swish 非单调**：学生提问 Swish / GeLU 在负区间不单调递增，会不会让梯度方向反了？答：实际训练用大 LR + momentum，activation 不会收敛在这一小段反常区间；实践中不是问题。

**关键实现约定**：gated 变体因为多了 $V$，为了参数数量和普通 FFN 对齐，把 $d_{ff}$ 缩小 $2/3$。于是：

$$
\text{default } d_{ff} = 4 \cdot d_{model} \;\;\longrightarrow\;\; \text{gated version: } d_{ff} = \frac{8}{3} \cdot d_{model}
$$

用 *GLU 的模型（基本是 2023 年后所有模型）：LLaMA 1/2/3、PaLM、Mistral、OLMo、T5 v1.1、mT5、Phi-3、Gemma 2/3。

**反例**：GPT-3（GeLU）、Nemotron 340B（Squared ReLU）、Falcon 2 11B（ReLU）——说明 *GLU 不是必需，只是显著更好。

Shazeer 2020 和 Narang 2020 的 ablation 都一致表明 *GLU > 非 gated 版本，差距虽小但统计显著。

### 2.6 Serial vs Parallel Layers

**Serial**（绝大多数模型）：

$$
y = x + \text{MLP}\big(x + \text{Attn}(x)\big)
$$

**Parallel**（GPT-J 首创，PaLM、GPT-NeoX 跟进）：

$$
y = x + \text{Attn}(x) + \text{MLP}(x)
$$

两个 sub-layer 同时算。

- 优点：LayerNorm 可共享；Attn 和 MLP 的 input projection 可以 **fuse 成一个大矩阵乘**，在大规模并行时效率更高
- 缺点：理论表达力略差（复合 → 相加）
- 近况：没有特别扎实的 ablation，但最近少见，只剩 Cohere Command A / R+、Falcon 2 11B 在用

### 2.7 架构部分小结

| 维度 | 共识程度 | 结论 |
| --- | --- | --- |
| pre-norm vs post-norm | **完全共识** | pre-norm（+ 可选 double norm） |
| LayerNorm vs RMSNorm | 大多数 | RMSNorm |
| Bias 项 | 大多数 | 去掉 |
| 激活函数 | 大多数 | SwiGLU / GeGLU |
| Serial vs Parallel | 大多数 | Serial |

---

## 3. 位置编码

### 3.1 几种历史方案

| 方案 | 公式要点 | 代表模型 |
| --- | --- | --- |
| Sinusoidal | $\text{Embed}(x,i) = v_x + PE_i$，PE 是 sin/cos | 原始 Transformer |
| Absolute learned | $\text{Embed}(x,i) = v_x + u_i$，$u_i$ 可学 | GPT-1/2/3、OPT |
| Relative | 加在 attention 矩阵里 | T5、Gopher、Chinchilla |
| **RoPE** | 旋转 query/key | GPT-J、PaLM、LLaMA、2024+ 几乎所有模型 |

### 3.2 RoPE 的动机

相对位置编码的**理想形式**：存在函数 $g$ 使得

$$
\langle f(x, i),\; f(y, j)\rangle = g(x, y, i - j)
$$

也就是 attention 只依赖于相对位置 $i - j$。

检查历史方案：

- **Sine**：展开内积后有 $\langle v_x, PE_j\rangle$ 之类的交叉项，**泄漏绝对位置信息**
- **Absolute learned**：顾名思义，绝对
- **Relative**：确实相对，但**不是内积形式**，不满足上面的结构约束

### 3.3 RoPE 的核心想法

利用一个几何事实：**内积对旋转不变**。

- 单词 $x$ 的"静止"嵌入是 $v_x$
- 位置 $i$ 上的嵌入 = 把 $v_x$ 旋转 $i$ 个单位

两个单词在位置 $(i, j)$ 的内积只依赖 $i - j$（因为整体同旋转被抵消）。

例：序列 `we know that` 和 `of course we know`，两句话里 `we` 和 `know` 的相对位置都是 1，所以它们的 attention 内积完全一样。

### 3.4 高维怎么"旋转"？

2D 旋转很自然（只有一个角度），高维要选一种旋转方式。RoPE 的选择（Su et al. 2021）：

**把 $d$ 维向量切成 $d/2$ 对 2D 坐标，每对用一个频率 $\theta_k$ 做 2D 旋转**。

第 $k$ 对坐标、位置 $m$ 的旋转矩阵：

$$
R_{m,k} = \begin{pmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{pmatrix}
$$

和 sin/cos PE 一样，$\theta_k$ 取一组从高到低的频率，覆盖从局部到长程的位置信息。

$\theta_k$ **不是可学参数**，是固定的 schedule（不同模型略有不同，但都是固定的）。不同模型之间 base frequency 会变，这也是后来做长上下文外推的关键接口。

### 3.5 RoPE 实现关键

**RoPE 作用在 query/key 上，不作用在 input embedding 上**。

```
q = X @ Wq                      # 常规 projection
k = X @ Wk
cos, sin = rope_angles(seq_len) # 预计算 cos/sin table
q = rotate(q, cos, sin)         # 按位置旋转 q
k = rotate(k, cos, sin)         # 按位置旋转 k
attn = softmax(q @ k.T / sqrt(d)) @ v
```

每一层 attention 都会重新做旋转——保证位置不变性的结构被反复施加。

实现上 rotation 就是一个**固定矩阵乘法**，没有额外可训练参数，也不涉及对 trig 求梯度。

---

## 4. 超参数

### 4.1 共识 #1：$d_{ff} / d_{model}$

**默认规则**：

$$
d_{ff} = 4 \cdot d_{model}
$$

GLU 变体因为 $2/3$ 缩放：

$$
d_{ff} = \frac{8}{3} \cdot d_{model} \approx 2.66 \cdot d_{model}
$$

现实中的比例（大多落在 2.5–4）：

| 模型 | $d_{ff} / d_{model}$ |
| --- | --- |
| PaLM | 4 |
| Mistral 7B | 3.5 |
| LLaMA-2 70B | 3.5 |
| LLaMA 70B | 2.68 |
| Qwen 14B | 2.67 |
| DeepSeek 67B | 2.68 |
| Yi 34B | 2.85 |
| T5 v1.1 | 2.5 |

**大离谱例外：T5 11B**：

$$
d_{model} = 1024, \quad d_{ff} = 65{,}536 \;\;\Rightarrow\;\; \text{ratio} = 64
$$

T5 作者的理由是「更宽的矩阵乘在 TPU 上更高效」——确实真实，但**信息瓶颈**也更大，表达方式偏 parallel 而非 serial，表达效率更差。后续 T5 v1.1 就改回 2.5 了——侧面说明 64× 是次优选择。

Gemma 2（8×）、SmolLM / Gemma 3（4× GLU）是近期的相对大 multiplier。

**为什么这个范围"合理"**：Kaplan+ 2020 的 scaling law 论文里做了 ablation，发现 loss 在 $d_{ff}/d_{model} \in [1, 10]$ 存在一个很宽的 basin，4 刚好落在谷底附近。

### 4.2 共识 #2：$\text{num\_heads} \times \text{head\_dim} = d_{model}$

标准做法：给定 $d_{model}$，加 head 就是把维度切得更细，总维度守恒。

| 模型 | num_heads | head_dim | $d_{model}$ | ratio |
| --- | --- | --- | --- | --- |
| GPT-3 | 96 | 128 | 12288 | 1 |
| T5 | 128 | 128 | 1024 | **16** |
| T5 v1.1 | 64 | 64 | 4096 | 1 |
| LaMDA | 128 | 128 | 8192 | 2 |
| PaLM | 48 | 256 | 18432 | 1.48 |
| LLaMA-2 | 64 | 128 | 8192 | 1 |

Bhojanapalli 2020 论证过多 head 会导致 head 内部出现 low-rank bottleneck，但实际模型没怎么暴露出这个问题。

### 4.3 共识 #3：Aspect Ratio（$d_{model} / n_{layer}$）

宽一点还是深一点？

| 模型 | $d_{model} / n_{layer}$ |
| --- | --- |
| BLOOM | 205 |
| T5 v1.1 | 171 |
| PaLM 540B | 156 |
| GPT-3 / OPT / Mistral / Qwen | ~128 |
| LLaMA / LLaMA-2 / Chinchilla | ~102 |
| T5 11B | 43（很深） |
| GPT-2 | 33 |

**sweet spot 大致在 100–200**。

Kaplan 2020 的 ablation 表明，在 50M / 274M / 1.5B 三种规模下，aspect ratio 的最优点位置**几乎不随规模变化**——跨数量级的可迁移经验。

**Tay 2021 的补充**：从 **pre-training loss** 看，深度意义不大，只要参数够就行；但从**下游 fine-tuning 精度**（如 SuperGLUE）看，在同 FLOPs 下更深的模型可能更好。

**系统层面考量**（后续 parallelism 课会详讲）：

- **Pipeline parallel**：切层，需要层够深才能切得均匀，对网络带宽要求相对低
- **Tensor parallel**：切矩阵，需要矩阵够宽，对网络带宽要求高

所以深 / 宽的选择经常是被 interconnect / topology 倒逼的。

### 4.4 Vocabulary Size

- **单语模型**：30k–50k（原始 Transformer 37k、GPT-2/3 50257、LLaMA 32000、T5 32128）
- **多语言 / 生产级**：100k–250k（mT5 250k、PaLM 256k、GPT-4 ~100k、Command A 255k、Qwen 15B 152k）

多语言模型词表必须大，否则低资源语言被切碎成一堆 token，推理成本高、表达差。Cohere 的卖点之一就是大词表 + 多语友好。

对于高资源语言（比如英语），大词表收益有限；但考虑到模型越大越能用好大词表，近期趋势是词表持续变大。

### 4.5 Dropout 与 Weight Decay

**直觉**：pre-training 一般只走 1 epoch，数据量远大于参数，按理不会过拟合，应该**不需要正则化**。

**实际做法**：

| 模型 | Dropout | Weight Decay |
| --- | --- | --- |
| 原始 Transformer | 0.1 | 0 |
| GPT-2 | 0.1 | 0.1 |
| T5 | 0.1 | 0 |
| GPT-3 | 0.1 | 0.1 |
| T5 v1.1 | 0 | 0 |
| PaLM | 0 | variable |
| OPT | 0.1 | 0.1 |
| LLaMA | 0 | 0.1 |
| Qwen 14B | 0.1 | 0.1 |

趋势：

- **Dropout** 基本被放弃（原因没有特别清晰的 paper，但逻辑是：一遍过的 pre-training 根本没有 overfit 问题）
- **Weight Decay** 仍被广泛使用，但**不是为了防过拟合**

**为什么继续用 weight decay？Andriushchenko 2023 发现**：

- 不同 weight decay 对 train/val loss **gap** 没有影响——不是在防 overfit
- Weight decay 真正的作用是**和 LR schedule（尤其 cosine）发生微妙耦合**

具体表现：

- 常数 LR：高 weight decay 会让训练变慢，但一旦 LR 突然下降，loss 会快速下跌
- Cosine schedule：高 weight decay 模型前期训得慢，但在 LR 冷却阶段会**加速优化**，最终到达更低 **training** loss

所以：**为了更好的 training loss 而用 weight decay**，是一种和优化器动力学相关的 implicit acceleration，不是经典意义的正则化。

### 4.6 超参数小结

- `d_ff = 4 * d_model`（非 gated）或 `8/3 * d_model`（gated）：几乎是 no-brainer
- `n_heads * head_dim = d_model`：标准但弱验证
- aspect ratio 100–200：经验上稳，具体值常被 systems 倒逼
- weight decay 要上，dropout 可省，但注意这两者都是优化动力学问题，不是正则化

---

## 5. 稳定性技巧

### 5.1 问题：训练崩了

大模型训练常见现象：loss 曲线看似还行，但 **gradient norm** 开始周期性 spike，直到某次 spike 彻底爆炸、训练报废。OLMo 2 / PaLM 等论文里都有血淋淋的对比图。

大部分稳定性问题的**共同根源**：**Softmax**。

- 取 $\exp$ → 数值越界
- 分母正则化 → 分母过小甚至为 0

Transformer 里有两个 softmax，分别对应两类 intervention：

1. **输出层 softmax** → z-loss
2. **Attention 内部 softmax** → QK-norm（和 logit soft-capping）

### 5.2 Z-loss（搞定输出 softmax）

输出 softmax：

$$
P(x) = \frac{\exp(\text{logits}(x))}{Z(x)}, \qquad Z(x) = \sum_{x'} \exp(\text{logits}(x'))
$$

想让 $Z(x) \approx 1$，即 $\log Z(x) \approx 0$——这样 softmax 数值就非常稳。

加辅助损失（Devlin 2014 原本是 MT 里的做法）：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \cdot \big(\log Z(x)\big)^2
$$

PaLM 用 $\alpha = 10^{-4}$，并给这个叫法起了名字 "z-loss"。后续 Baichuan 2、DCLM、OLMo 2 都采用。

本质：如果 $\log Z \to 0$，那 $\exp$ 和 $\log$ 抵消，softmax 输出就只剩 logits 本身，数值行为极其稳定。

### 5.3 QK-Norm（搞定 Attention softmax）

原始 attention：

$$
\text{Attn} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

**QK-Norm**：对 $Q, K$ 做 LayerNorm / RMSNorm **再进 softmax**：

$$
\text{Attn} = \text{softmax}\!\left(\frac{\text{Norm}(Q)\,\text{Norm}(K)^\top}{\sqrt{d}}\right) V
$$

z-loss 是控制 softmax 的**分母**，QK-Norm 是控制 softmax 的**输入幅度**。QK 被 norm 后，内积不会无限增大，从根源上避免爆炸。

起源：vision / multimodal（Dehghani 2023 的大 ViT，Idefics，Chameleon），后来 Gemma 2、DCLM、OLMo 2 都抄过来。

Nvidia 的 ablation：baseline PPL 11.19，QK-Norm **更低**（不仅更稳，还能用更激进的 LR）；soft-cap 反而略差。

**老师的 running joke**：稳定性上 **LayerNorm 几乎是万能药**——pre-norm 放一个，block 后面再放一个，Q/K 上再放一个，层层都塞 LayerNorm。

**推理时 QK-Norm 必须保留**：它有可学参数 ($\gamma$)，模型已经学会在 "已 norm" 的空间里工作，去掉会崩。

### 5.4 Logit Soft-Capping

对 logit 做 tanh 限幅：

$$
\text{logits}_{\text{capped}} = \text{softcap} \cdot \tanh\!\left(\frac{\text{logits}}{\text{softcap}}\right)
$$

logit 大时 tanh 饱和到 $\pm 1$，绝对值被限制在 `softcap` 内。

Gemma 2（以及 Gemma 3）用了。但 Nvidia 的对比里 soft-cap 反而让 PPL 变差，所以**不是主流选择**，比 QK-Norm 要"重手"。

### 5.5 稳定性小结

| Trick | 位置 | 作用 | 代表模型 |
| --- | --- | --- | --- |
| Z-loss | 输出 softmax | 控制分母 $Z$ 接近 1 | PaLM、Baichuan 2、DCLM、OLMo 2 |
| QK-Norm | attention 输入 | 控制 logit 幅度 | Gemma 2、DCLM、OLMo 2 |
| Logit soft-cap | attention logits | 限幅 | Gemma 2（争议） |

---

## 6. 注意力变体

架构核心变化不多，主要围绕**推理效率**和**长上下文**。

### 6.1 为什么需要 MQA / GQA——算术强度视角

**训练时**的 attention 算一次（一个 batch）：

$$
\text{FLOPs} \sim b\,n\,d^2, \qquad \text{Memory} \sim b\,n\,d + b\,h\,n^2 + d^2
$$

算术强度 $=\text{FLOPs} / \text{Memory}$：

$$
\text{Arithmetic Intensity} \sim \left(\frac{1}{h} + \frac{1}{b\,n}\right)^{-1}
$$

$h$（heads）、$b$（batch）、$n$（seqlen）都大 → 算术强度高 → GPU 能打满。

**推理时**是 autoregressive：

- 一次生成一个 token，无法并行
- 需要 **KV-Cache**：每生成一步，把新算的 $K, V$ 存下来；$Q$ 只算当前这一个 token

KV-Cache 下的 attention：

$$
\text{FLOPs} \sim b\,n\,d^2, \qquad \text{Memory} \sim b\,n^2 d + n\,d^2
$$

算术强度：

$$
\text{Arithmetic Intensity} \sim \left(\frac{n}{d} + \frac{1}{b}\right)^{-1}
$$

想让强度高 → 要 $n/d$ 小（短序列或大模型）且 $b$ 大。

**现实中**：部署往往序列长、模型没那么大、batch 也受限——$n/d$ 这一项死死压制 GPU 利用率，推理变成 **memory-bound**。

### 6.2 MQA（Multi-Query Attention）

**核心想法**：**多个 query head，但只保留 1 个 KV head**。

```
Q: [h heads]           —— 每个 head 独立
K: [1 head]            —— 所有 query head 共享
V: [1 head]            —— 所有 query head 共享
```

KV-Cache 大小缩小 $h$ 倍——memory movement 大幅下降。

算术强度改进：

$$
\text{Arithmetic Intensity} \sim \left(\frac{1}{d} + \frac{n}{d\,h} + \frac{1}{b}\right)^{-1}
$$

$n/d$ 项被额外除以 $h$，压制变得宽容很多；$1/d$ 项也只是 KV 的 projection。

**代价**：表达力下降，PPL 略升（Shazeer 2019 有小的 hit）。

### 6.3 GQA（Grouped-Query Attention）

**折中方案**：$h$ 个 query head 分成 $g$ 组，每组共享 1 个 KV head（$g$ 是可调旋钮，$g=1$ 退化为 MQA，$g=h$ 退化为 MHA）。

- 可控的 expressiveness / inference 效率 trade-off
- Ainslie 2023：GQA 性能损失几乎为零
- 现代 LLaMA-2 / 3 等都用 GQA

### 6.4 Sparse / Sliding Window Attention

全 attention 是 $O(n^2)$，想做 10M token 上下文就必须砍掉密集连接。

**Sparse Attention**（Child 2019，GPT-3 用过）：构造结构化稀疏 pattern，比如 local window + 若干 strided diagonal，兼顾局部和跨距信息传递。

**Sliding Window Attention**（Mistral 等）：每层只看当前位置周围一段局部窗口。深度 × 窗口 = 有效感受野，靠层数堆叠把信息传得更远。

### 6.5 当前主流：Interleaved Full + Local Attention

Cohere Command A、LLaMA 4、Gemma 等的最新做法：

```
Block 4k+0:  Full attention,   NoPE (no position embedding)
Block 4k+1:  Sliding window,   RoPE
Block 4k+2:  Sliding window,   RoPE
Block 4k+3:  Sliding window,   RoPE
```

- **4 层里只有 1 层做全 attention**——compute 压力大幅下降
- **全 attention 层不加位置编码（NoPE）**——长距离信息不受 RoPE 外推精度限制，外推能力强
- **sliding window 层用 RoPE**——局部精细位置信息保留

这是 10M-token 级别长上下文模型的关键配方之一。

---

## 7. 全课总结

### 架构共识（按"共识强度"排序）

1. **Pre-norm**：几乎无例外
2. **RoPE**：2024+ 所有新模型
3. **RMSNorm**：主流
4. **SwiGLU / GeGLU**：主流
5. **去 bias**：主流
6. **Serial layers**：主流（Parallel 有少量拥趸）

### 超参数共识

- $d_{ff} = 4 d_{model}$（非 gated） / $\frac{8}{3} d_{model}$（gated）
- $\text{head\_dim} \times n_{heads} = d_{model}$
- aspect ratio $d_{model}/n_{layer} \in [100, 200]$
- vocab：单语 30–50k，多语 / 生产 100–250k
- weight decay 有，dropout 基本无

### 稳定性

- Z-loss 控输出 softmax
- QK-Norm 控 attention softmax
- （LayerNorm 见缝插针）

### 注意力

- 训练：MHA + RoPE + Flash Attention
- 推理：MQA / GQA 降 KV cache 成本
- 长上下文：interleaved full + sliding window（+ NoPE on full）

### 可泛化的元经验

1. Residual identity 路径要干净
2. 归一化控制 activation scale 是稳定性神器
3. FLOPs ≠ runtime，memory movement 常是真瓶颈
4. Gating 反复被证明有效（*GLU 是一例，GQA 的 KV 共享也是一种 gating 思路）
5. 大部分"更好"的差别都很小，**保守抄袭主流配方通常不错**，但 T5 11B 说明激进配置也能 work

### 主要仍在变化的东西

- 位置编码（RoPE 的各种变种、NoPE 组合）
- 注意力 pattern（MHA / MQA / GQA / SWA / full-interleaved）
- Tokenization / vocab size
- 稳定性 trick（还在出新）
