# CS336 Lecture 4：Mixture of Experts (MoE) 笔记

> Stanford CS336 - Language Modeling from Scratch (Spring 2025)
> 授课：Tatsu Hashimoto
>
> 配套讲义：`2025 Lecture 4 - MoEs.pdf`（47 页）

---

## 一、为什么要学 MoE？

2024 之前 MoE 还只是一节"彩蛋课"，2025 已成为必修：

- **GPT-4**（泄漏传闻是 8×220B MoE）、**Grok**、**Mixtral**、**DeepSeek V1–V3**、**Qwen MoE**、**Llama 4** 全面采用 MoE
- 在**所有 compute scale** 下，MoE 训得好就比同 FLOPs 的 dense 模型效果更好
- 2025 的共识：**要想在给定 FLOPs 预算下训出最好的模型，就用 MoE**

### MoE 这个名字很有误导性

听到 "Mixture of Experts"，第一反应是有个"写代码专家"、"英语专家"、"翻译专家"。**完全不是这样**。

MoE 只是一种**稀疏激活的架构**：

- 架构里有若干子模块（experts），每次前向只激活其中一小部分
- action 全发生在 **MLP / FFN** 层；其他部分（attention、LN、embedding）和 dense 完全相同

### MoE 层长什么样

把 transformer block 里那一个大 FFN 换成：

1. 多个（更小或等大的）FFN 副本 —— 称为 experts
2. 一个 **router**（门控 / 选择器），决定 token 走哪几个 expert

```
                   ┌── Expert 1 ──┐
 residual ── router┼── Expert 2 ──┼── 加权求和 ── residual
                   ├── Expert 3 ──┤
                   └── Expert N ──┘
```

如果每次只激活 1 个 expert，且该 expert 大小等于原 dense FFN，那么 MoE 和 dense 的 **FLOPs 相同**，但**参数量** 为 N 倍。

---

## 二、MoE 好在哪

### 1. 同 FLOPs，更多参数 = 更好效果

Fedus et al. 2022（Switch Transformer）等大量工作验证：**固定训练 FLOPs，expert 数越多 training loss 越低**。

代价：experts 的内存必须存下；跨设备路由有系统复杂度。但纯看 FLOPs 视角，这是白捡的性能。

### 2. 训练更快

OLMoE 做了严格 ablation：同 compute 下 MoE 的 loss 下降速度比 dense 明显更快，Fedus 报出过 7× 加速。

### 3. Active parameters 曲线非常好看

DeepSeek V2 论文图：x 轴只是 **activated params**，y 轴 MMLU。因为 deactivated experts 不参与 forward 计算，所以推理和训练 FLOPs 都只和 active params 相关。

### 4. 天然支持 Expert Parallelism

每个 expert 是独立的 FFN，可以各自放到一台不同设备上。token 到达时路由到对应设备，在本地算完再返回。这是**一种新的并行维度**，和 data / tensor / pipeline parallel 可以组合使用。

### 5. 开源生态

西方闭源实验室很早就用；开源方面 **中国团队（Qwen、DeepSeek）** 先做出来，Mixtral、DBRX、Llama 4 随后跟进。DeepSeek V3 是目前最有代表性的 SOTA 开源 MoE。

> 有趣事实：DeepSeek V3 的架构和 V1（DeepSeekMoE，2024 年初）几乎**一模一样**，他们很早就把架构钉死了，后续主要是工程上的打磨。

---

## 三、为什么 MoE 没更早流行？

两个痛点：

1. **基础设施复杂**。优势主要出现在**多节点训练**时（反正模型都要切，不如按 expert 切）；单节点 MoE 收益有限
2. **路由是离散决策，不可微**。训练目标要么是启发式、要么不稳定

这两件事都贯穿本讲剩余内容。

---

## 四、MoE 的典型结构

**最常见**：把每个 transformer block 里的 MLP 换成 MoE 层。

**少见**：对 attention head 也做 MoE（ModuleFormer、JetMoE）。网络上普遍反馈这样更不稳定，主流大模型基本都不这么做。

MoE 的三个核心设计维度：

1. **Routing function**：token 如何匹配 expert
2. **Expert sizes**：多少个 expert，每个多大
3. **Training objective**：怎么训这个不可微的路由

---

## 五、Routing Function

### 5.1 三大类路由

| 类别 | 做法 | 特点 |
| --- | --- | --- |
| **Token choice top-k** | 每个 token 选 top-k 个 expert | 默认选择，几乎所有主流 MoE 都用 |
| **Expert choice** | 每个 expert 挑 top-k 个 token | 天然 balance，但语义选择性变差 |
| **Global assignment** | 解一个全局最优匹配问题 | 优雅但计算贵，实际很少用 |

**直观对比**：把 score 想成一个 token × expert 的矩阵。

- token choice = 沿 **列** 方向取 top-k（每个 token 取几个 expert）
- expert choice = 沿 **行** 方向取 top-k（每个 expert 取几个 token）

Expert choice 的好处：每个 expert 吃到的 token 数完全相等 → 设备负载天然均衡。
Token choice 的好处：语义上"为每个 token 挑最佳 expert"更自然。

OLMoE 的 ablation 给了决定性结论：**token choice** 的 validation loss 下降又快又稳。→ 主流都选它。

### 5.2 Top-k 路由的数学形式

这是本讲最重要的一组公式，来自 DeepSeek V1–V2（Qwen、Grok 也几乎相同）。

设 token 的 residual stream 输入为 $u_t \in \mathbb{R}^d$，第 $i$ 个 expert 的门控向量为 $e_i \in \mathbb{R}^d$。

Step 1：对每个 token 计算和每个 expert 的亲和度，过 softmax

$$
s_{i,t} = \text{Softmax}_i(u_t^\top e_i)
$$

Step 2：只保留 top-k 项作为 gate，其余置 0

$$
g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \text{TopK}(\{s_{j,t}\}_{j=1}^N, K) \\ 0, & \text{otherwise} \end{cases}
$$

Step 3：加权求和所有被激活 expert 的输出，再加残差

$$
h_t = u_t + \sum_{i=1}^{N} g_{i,t} \cdot \text{FFN}_i(u_t)
$$

### 5.3 设计决策的关键细节

**Q：Router 为什么这么简单（只是一个向量内积）？**

A：router 本身要吃 FLOPs，做复杂了反而亏。更关键的是**路由的梯度信号很弱**：只有激活的 top-k 个 expert 会回传梯度，router 没法看到"没选的那些 expert 会是什么结果"，所以即使做复杂也学不好。

**Q：Softmax 在哪？在 top-k 之前还是之后？**

- **Softmax 在 top-k 之前**（DeepSeek V1–V2、Qwen、Grok）：gate 之和不为 1，但无所谓（后面有 LayerNorm）
- **Softmax 在 top-k 之后**（Mixtral、DBRX、DeepSeek V3）：只对被选中的 k 个重新归一化，和为 1

两者差别不大，更多是审美选择。

**Q：为什么不去掉 top-k，只用 softmax？**

如果不做 top-k，就得 forward 所有 N 个 expert，**训练时**就失去了稀疏性 → MoE 的意义全没了。top-k 的存在就是为了保证 **train 和 inference 都稀疏**。

**Q：为什么不直接 RL 采样 / 按 softmax 采样？**

可以做，Google 有论文让最大那项必取，第二项按剩余 softmax 抽样（带 exploration）。但 test 时不采样会带来 train/test mismatch，普遍没采用。

**Q：K 怎么选？**

早期论文主张 K ≥ 2，理由是给一点 exploration：如果 K=1，就一直"利用最好的臂"，第二臂永远学不到信息。Switch Transformer 用 K=1，多数模型用 K=2，DeepSeek V3 用到 K=8。K 翻倍 FLOPs 也翻倍，大家通常说的"active params" 已经把 K 算进去了。

### 5.4 其它路由方法（不常用但需要知道）

| 方法 | 备注 |
| --- | --- |
| **Hashing** | 用随机哈希函数路由。没有任何语义，居然也能比 dense 好 —— 说明 MoE 的大部分收益其实来自"分组 + 专业化"本身 |
| **RL learned routing** | 原理正确（离散决策天生该用 RL），但梯度方差大、成本高，早期尝试后基本没人用 |
| **Linear assignment / Optimal transport** | Clark 2022 等工作。优雅，但计算代价超过收益 |

为什么 hashing 也能 work？因为同样的 token（比如单词 "the"）每次都会哈希到同一个 expert → **非语义但稳定** 的专业化依然发生。

---

## 六、Expert 数量与粒度

### 6.1 Shared + Fine-grained experts（DeepSeek 的关键创新）

**Fine-grained experts**：

- 以前：一个 expert 就是一个标准大小的 FFN（hidden × 4×hidden）
- 现在：把 FFN 切小（比如 hidden × 2×hidden 或更小），数量成倍增加
- FLOPs 不变，参数更细粒度，专业化更好

**Shared experts**：

- 每次都激活、不参与路由的"常驻 expert"，数量 0~2 个
- 思路：总有一些计算每个 token 都要做，分摊到多个普通 expert 就浪费了，单独开一个 shared expert 处理这些公共模式

### 6.2 Ablation 结论有点矛盾

- **DeepSeek 原论文**：fine-grained 和 shared 都有显著收益
- **OLMoE**：fine-grained 随 8→32→64 单调提升；但 shared experts 基本看不到收益

所以 fine-grained 是无脑采用；shared 用不用看情况。

### 6.3 主流 MoE 的配置表

| Model | Routed | Active | Shared | Fine-grained ratio |
| --- | --- | --- | --- | --- |
| GShard | 2048 | 2 | 0 | - |
| Switch Transformer | 64 | 1 | 0 | - |
| ST-MoE | 64 | 2 | 0 | - |
| Mixtral | 8 | 2 | 0 | - |
| DBRX | 16 | 4 | 0 | - |
| Grok | 8 | 2 | 0 | - |
| **DeepSeek V1** | 64 | 6 | 2 | 1/4 |
| Qwen 1.5 | 60 | 4 | 4 | 1/8 |
| **DeepSeek V3** | 256 | 8 | 1 | 1/14 |
| OLMoE | 64 | 8 | 0 | 1/8 |
| MiniMax | 32 | 2 | 0 | ~1/4 |
| Llama 4 Maverick | 128 | 1 | 1 | 1/2 |

**ratio** 的含义：每个 expert 的 FFN 中间维度是 dense 标准大小的 $\frac{1}{\text{ratio}}$。例如 DeepSeek V1 的 1/4 表示每个 expert 是标准 FFN 的四分之一大。

趋势：模型越大，routed experts 越多；fine-grained 比例越极致（V3 到了 1/14）。

---

## 七、如何训练 MoE（最核心的部分）

### 7.1 根本矛盾

训练时必须稀疏（否则 FLOPs 爆炸），但 **top-k 是离散决策，不可微**。

三条路线：

1. **RL 优化路由策略** —— 正确但贵
2. **随机扰动做 exploration** —— 折中
3. **启发式 balancing loss** —— 实际大家都用这个

### 7.2 路线 1：RL（基本被放弃）

REINFORCE 可以 work（Clark et al. 2020 的 baseline），但：

- 梯度方差大
- 训练不稳定，本来 MoE 就已经够难训了
- 相比 hashing 等 baseline 也没优势

### 7.3 路线 2：随机扰动（历史用过，现已少见）

**Shazeer 2017**：给 gate 的 logit 加高斯噪声

$$
H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_\text{noise})_i)
$$

噪声幅度 $W_\text{noise}$ 是学的。这本质是 **ε-greedy 风格的 exploration**：让 expert 偶尔收到意外的 token，变得更鲁棒。

**Fedus 2022**：用乘性均匀扰动（jitter），目的类似。Zoph 2022 后期把它去掉了 —— 发现 balancing loss 就够了。

### 7.4 路线 3：Balancing Loss（实际方案）

**为什么必须 balance？** 如果不加约束，训练很容易收敛到"所有 token 都路由到一两个 expert，其他 expert 是死的" 的退化解。这不仅浪费参数，还让 MoE 在 FLOPs 对等的情况下实际上变成 dense 模型。

#### 7.4.1 Switch Transformer 的经典 balance loss

对每一批数据，定义两个向量：

- $f_i$ = 这一 batch 中被路由到 expert $i$ 的 token 比例（实际路由）
- $P_i$ = router softmax 给 expert $i$ 的平均概率（期望路由）

Balance loss：

$$
\mathcal{L}_\text{balance} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$

理想 balance 时 $f_i = P_i = 1/N$，损失为 $\alpha / N$。

**梯度直觉**：对 $P_i$ 求导

$$
\frac{\partial \mathcal{L}_\text{balance}}{\partial P_i} = \alpha N \cdot f_i
$$

分配到的 token 越多，梯度反推得越强 → **自动把过载的 expert 往下压**。

为什么用 $f_i \cdot P_i$ 而不是 $P_i^2$ 之类？因为 $f_i$（指示函数的统计）本身不可微，把它当常数系数，梯度只流经可微的 $P_i$ 路径。

#### 7.4.2 DeepSeek V1–V2 的做法

在 Switch 的基础上多加了一项 **device-level balance**：

- **Expert balance**：按 expert 聚合（和 Switch 一样）
- **Device balance**：按 device 组聚合，鼓励每台 GPU / TPU 分到的 token 数相等

两者一起优化，让路由同时满足"专家均衡"和"设备均衡"。

#### 7.4.3 DeepSeek V3 的 "Auxiliary-loss-free balancing"

V3 去掉了 per-expert 的 aux loss，引入一个**可学的每-expert bias** $b_i$：

$$
s'_{i,t} = s_{i,t} + b_i, \quad g_{i,t} = \text{TopK based on } s'_{i,t}
$$

注意 $b_i$ **只用于选 top-k**，**不**参与加权求和（加权仍用原始 $s_{i,t}$）。

$b_i$ 通过极简的 online gradient 更新：

- 如果 expert $i$ 在本 batch 吃到的 token 比平均少 → $b_i \leftarrow b_i + \gamma$（让它下次更容易被选）
- 如果吃到的比平均多 → $b_i \leftarrow b_i - \gamma$
- $\gamma$ 是超参

**但 V3 也不是完全 aux-loss-free**：论文后面补了一个 **per-sequence aux loss**，防止单个 sequence 级别的严重不均衡。这一点对**推理时收到 OOD 序列**特别重要 —— 推理时你没法控制用户输入，希望即使单条序列也能均匀用到 experts。

#### 7.4.4 去掉 balance loss 会怎样？

OLMoE 的 ablation 图：不做 load balance 时，pink + yellow 两个 expert 吃掉约 50% token，其它 6 个 expert 完全沉默。MoE 退化成 2-expert 模型，validation loss 明显变差。

---

## 八、系统实现

### 8.1 Expert Parallelism

每个（或少量几个）expert 放到一张独立设备上。一个 token 经过 MoE 层时：

1. 先过 router 决定去哪几个 expert
2. **All-to-all dispatch**：把 token 发到对应设备
3. 各设备本地 forward 自己的 FFN
4. **All-to-all combine**：把结果收回原 token 所在设备

只要 FFN 计算量够大，就能吃得下 all-to-all 的通信开销。

Expert parallel 是 **独立于 data / tensor / pipeline parallel 的第四个维度**，可以组合。

### 8.2 MegaBlocks：一台设备上多个 expert 怎么高效？

如果一台设备上有多个 expert，朴素做法是对每个 expert 单独做 matmul → 小 matmul 很低效。

MegaBlocks 的做法：把所有 expert 的权重拼成大矩阵，用 **block-sparse matmul** 一次算完。现代 GPU 的 sparse matmul kernel 能高效支持这类"按行分块稀疏"的模式，MoE 用起来没有明显 kernel 损失。

### 8.3 一个有趣现象：MoE 的随机性

GPT-4 刚出 API 时，即使 temperature=0 响应也不稳定。一个可能原因：**token dropping**。

训练和推理都可能设置 **load factor**（每个 expert 每 batch 最多能吃多少 token）。batch 里其他人的 query 如果集中涌向 expert 3，把它撑爆，**你的 token 会被 drop** —— drop 的 token 直接走残差连接，相当于那层 FFN 没算。

所以 **batch 里"邻居"的改变** 会通过 token dropping 改变你的输出。这是推理时很少被讨论的 cross-batch 效应。

---

## 九、MoE 的稳定性问题

### 9.1 Router 不稳定 → Float32 + Z-loss

和上讲一样，softmax 永远是不稳定性的源头。MoE 把 router 的 softmax 计算用 **float32** 做；有时再加一个 **z-loss**：

$$
\mathcal{L}_z = \frac{1}{B} \sum_{i=1}^{B} \left( \log \sum_{j=1}^{N} e^{x_i^{(j)}} \right)^2
$$

惩罚 logit 的 logsumexp 偏离 0，让 softmax 的归一化常数保持在 1 附近。

Zoph 2022 的 ablation：去掉 z-loss 后 validation loss 出现巨大 **spike**，最终也比带 z-loss 的差。

### 9.2 Fine-tuning 过拟合

MoE 参数量大，在小 SFT 数据集上很容易过拟合（BERT / T5 时代观察到 train/val gap 远大于 dense）。两种解法：

- **Zoph 方案**：交替放 dense 层和 MoE 层，只 fine-tune dense 部分
- **DeepSeek 方案**：堆数据，直接上 140 万条 SFT，过拟合自然不是问题

---

## 十、Upcycling：用 dense 模型初始化 MoE

不是从头训 MoE，而是：

1. 训好（或拿一个现成的）dense 模型
2. 把它的 FFN **复制** 多份作为初始 experts
3. 加小扰动打破对称性
4. 随机初始化一个 router
5. 当作 MoE 继续训

优点：非常 cost-effective，省去大量 from scratch 的训练量。

成功案例：

- **MiniCPM MoE**：topk=2，8 experts，~4B active，continue train 约 520B tokens 就从 dense baseline 涨了一大截
- **Qwen MoE**：从 Qwen 1.8B dense upcycle 出来，60 experts + 4 shared，top-k=4。2.7B active params 就达到 dense 7B 水平，是首个确认成功的开源 upcycling 案例

---

## 十一、案例串讲：DeepSeek MoE V1 → V2 → V3

本节把前面所有模块串起来看 DeepSeek 的演进。

### 11.1 DeepSeek V1（DeepSeekMoE，16B / 2.8B active）

- Fine-grained experts：64 routed（每个 1/4 标准大小）+ 2 shared
- 每 token 激活 ~6 routed + 2 shared
- 标准 top-k routing（softmax 在 top-k 之前）
- 标准 aux loss：expert 级 + device 级

**这一版就已经把架构钉死了**，后续几乎没变。

### 11.2 DeepSeek V2（236B / 21B active）

参数量翻了 14 倍，架构只加了两个系统侧的小优化：

**Top-M device routing**：

- 问题：expert 太细碎时，一个 token 的 top-k 可能散布到很多台 device，通信爆炸
- 解法：先按 device 算分，**选 top-M 台设备**，再在这 M 台里选 top-k experts
- 严格限制每 token 的目标设备数，控制通信

**Communication balance loss**：

- 一个 expert 有"进"和"出"两个方向的通信
- 除了 balance 进来的 token 数，再加一项 balance 出去的 token 数

### 11.3 DeepSeek V3（671B / 37B active）

架构图几乎和 V1 一样。变化：

- **Sigmoid + softmax top-k + top-M**：gate 分数用 sigmoid（不是 exp），top-k 之后再 softmax 归一化到 1
- **Aux-loss-free balancing**：前文 7.4.3 的 per-expert bias $b_i$ + online learning
- **Seq-wise aux loss**：补充项，处理推理时 OOD sequence
- 保留 V2 的 top-M device routing，**丢掉** 了 communication balance loss

---

## 十二、彩蛋：DeepSeek V3 的非 MoE 部分

V3 除了 MoE 外还有两项重要创新。

### 12.1 MLA（Multi-head Latent Attention）

目标和 GQA / MQA 一样 —— **压缩 KV cache**。思路不同：

- GQA：减 head 数
- **MLA**：把 K、V 投影到一个**低维 latent 向量** $c_t^{KV}$ 再存，用时再上投影回来

流程：

$$
c_t^{KV} = W_{DKV} h_t
$$

$$
K = W_{UK} c_t^{KV}, \quad V = W_{UV} c_t^{KV}
$$

KV cache 里只存 $c_t^{KV}$，维度小很多。

看起来多算了一个 $W_{UK}$？不用怕 —— attention 计算时要算 $Q^\top K$，而 $Q = h W_Q$：

$$
Q^\top K = (h W_Q)^\top (W_{UK} c_t^{KV}) = \langle h \cdot W_Q W_{UK}, c_t^{KV} \rangle
$$

$W_Q W_{UK}$ 可以**预先 merge** 成一个矩阵。所以 **运行时没有多出任何 matmul**。

**Query 端** 也做了同样的压缩（主要省训练时的显存，对 KV cache 无影响）。

**RoPE 的冲突**：RoPE 在 $Q, K$ 之间插入旋转矩阵 $R_q, R_k$：

$$
Q^\top K = h W_Q R_q R_k W_{UK} c_t^{KV}
$$

由于 $R_q R_k$ 依赖位置，**没法把它 merge 进** $W_Q W_{UK}$。

解法：保留少量"非 latent 的 key 维度"专门承载 RoPE，这几维正常 rotate；剩下大部分维度走 latent 压缩路径。

### 12.2 MTP（Multi-Token Prediction）

训练时让模型不光预测下一个 token，也预测再下一个：

- 主 transformer 照常预测 $t+1$
- 在 hidden state 后面接一个**轻量级单层 transformer**，预测 $t+2$

图上画得像可以预测多步，实际 V3 **只做了 1 个 token ahead**（即预测下下个 token），没展开更多步。

---

## 十三、一页速览（Takeaway）

- **MoE = 稀疏激活**：不是每个 token 都要过完整模型
- **离散路由很难**，但 top-k + balance loss 这套启发式就是能 work
- **大量经验证据表明** MoE 在 FLOPs 受限的场景下一定是正确选择
- **DeepSeek V1 就已基本是最终形态**，V3 的大部分创新在工程与规模上
- 训练 MoE 的关键清单：
  1. Fine-grained experts（无脑用）
  2. Top-k token choice routing（K=1~8）
  3. Balance loss（expert 级 + device 级；或 V3 的 bias 方案）
  4. Float32 router + Z-loss（防 spike）
  5. MegaBlocks / Expert parallel（系统侧）
  6. 实在预算紧张 → upcycling from dense

