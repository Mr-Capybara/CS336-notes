# CS336 Lecture 9：Scaling Laws（上）

配合讲义 `2025 Lecture 9 - Scaling laws basics.pdf`（共 53 页）阅读。每节我都会标出对应的 slide 编号，方便你对着原图看。

## 这节课到底想回答什么（Slide 1–4）

想象你有一个很有钱的朋友，给了你 10 万张 H100、一个月时间，让你训出最好的开源大模型。你已经搭好了分布式训练框架（A2），也准备好了预训练数据（A4），现在只剩一个问题：**这个大模型到底要怎么设计？**

宽一点还是深一点？用几个 attention head？激活函数选哪个？最省事的答案是抄 Llama，但如果你在 frontier lab，目标是把模型推到前沿，那抄作业是没有出路的——你必须自己做决定。

问题是，在 10 万卡规模上做超参数搜索，每一次 ablation 都要烧掉几百万美元。这就是 scaling laws 想解决的事：

> 在一堆小模型上做完所有实验、拟合出几条规律，然后**一次性外推到大模型**，把最后那把大钱花在已经被小实验选好的配置上。

整节课分成两大块：**数据 scaling law** 和**模型 scaling law**。前者比较干净，有很清楚的理论直觉；后者是真正指导工程的部分（选架构、选优化器、分配 FLOPs 预算）。

---

## 一、Scaling Laws 的历史与直觉（Slide 5–11）

讲 scaling law 的时候，外行人很容易把它讲成 AGI 布道——"log-log 图上是一条直线，所以只要我们一直 scale 就能达到超级智能"。课程想把这个话题拉回地面：scaling law 不是什么玄学，它在统计学习里其实有很长的历史。

### 从经典泛化上界说起（Slide 6）

如果你在机器学习 101 里学过 VC 维和 Rademacher 复杂度，那你其实已经见过 scaling law 的"理论版本"了。在有限假设类上，excess risk 的上界是 $O(1/\sqrt{m})$；在非参数密度估计里，L2 误差的上界是 $n^{-\beta/(2\beta+1)}$，β 刻画被估函数的光滑度。这些公式的形式——**误差随样本数按幂律衰减**——跟今天我们画的 log-log 直线在本质上是一回事。

两者的差别只在：那些是**上界**（upper bounds），真实误差可能小得多；而 scaling law 是直接**经验拟合出来的真实误差曲线**。从"证一个松的上界"跨到"直接拟真实值"，就是理论 → 经验的那一步跳跃。

### 第一篇 scaling law 论文其实是 1993 年的（Slide 7）

一个冷知识：最早的 scaling law 论文来自 1993 年 Bell Labs 的一篇 NeurIPS，作者里有 Vapnik 和 Corinna Cortes 这些统计学习的奠基人。那篇论文的动机跟今天几乎一模一样——"训分类器太贵，我们要在训之前就能预测它好不好"，方法也几乎一模一样——训一堆小模型，拟合出 "irreducible error + 多项式衰减项" 的形式，然后外推。现代 scaling law 的所有核心要素在那里都已经有了。

### 2001、2012、2017 三个里程碑（Slide 8–11）

- **Banko & Brill 2001**：在一个 NLP 任务上画出 log(data) vs performance 的直线，第一次明确提出"与其花钱搞算法不如花钱搞数据"。这个论点现在听上去像废话，但 2001 年是很超前的。
- **Kolachina et al. 2012**：开始认真讨论函数形式——到底是 power-3 还是 power-4 能更好地拟合？换句话说，大家开始意识到这东西不是随便一条线，背后可能有特定的代数形式。
- **Hestness et al. 2017**（百度的工作）：第一篇大规模的神经网络 scaling law，覆盖机器翻译、语音识别、图像等多种任务。这篇论文的贡献不只是"又拟了一条线"，而是提出了一张非常重要的图——**误差-数据量曲线分成三段**：

  1. **最左边**是 best-guess 区。数据太少，模型基本在乱猜，loss 卡在随机基线附近；
  2. **中段**才是我们熟悉的 power law 区，log-log 上是漂亮的直线；
  3. **最右边**是 irreducible error 区，数据再加也没用，被任务本身的噪声或模型容量限死了。

  Hestness 这篇论文其实特别超前：它已经讨论了今天大家爱聊的 emergence（离开 best-guess 区时能力突然出现）、compute 而不仅是参数数量才是真正该 scale 的东西、甚至用量化换精度的主意——这些在 2017 年就全都看见了。

### 有没有不满足 scaling 的情况？（字幕 Q&A）

有同学问："是不是什么任务都能 scale？"讲者的回答是：**在训练 loss 或它的 held-out 版本上**，scaling 几乎总是自然发生的，因为经典统计理论本来就保证了收敛。但有一个现象叫 **Inverse Scaling Prize**——专门找那些"模型越大越差"的任务，比如让模型去压制自己"复读"的倾向（强模型更会复读，所以更难压）。更一般地，**一旦你跑出训练分布足够远**，一切就都可能发生：no scaling、inverse scaling、怪异跳变都见过。这是深度学习鲁棒性问题的一个延伸。

---

## 二、数据 Scaling Law（Slide 12–25）

先讲数据那一侧，因为它理论上最干净。

### 经验事实：log-log 上就是一条直线（Slide 13–15）

Kaplan 2020 的图你一定看过：x 轴是 log(数据量 $n$)，y 轴是 log(test loss)，出来的就是一条**近乎完美的直线**。等价地：

$$
\mathrm{Error}(n) \;\approx\; C \cdot n^{-\alpha}
$$

这种形式叫 **scale-free** 或 **power law**。它还在很多"非标准"场景下成立——比如 Kaplan 2020 发现即使 train 分布和 test 分布不一样，scaling 依然是幂律，只是换了个截距。

有一个容易忽略的细节：当我们说"改变数据量看 loss"的时候，**模型必须足够大**，大到它不会被数据量饱和。否则你会掉进 irreducible error 区，看到的是模型容量的天花板而不是 scaling law。这就是为什么 Kaplan 画这类图时总是选一个非常大的模型作为 baseline。

### 为什么是幂律？两个 toy example（Slide 16–19）

这是这一节最值得花时间的地方。log-log 是直线这件事在神经网络里看上去很神奇，但其实在非常简单的统计问题里就已经出现了。

**Toy 1：估计高斯均值（Slide 17）**

设 $x_1, \dots, x_n \sim \mathcal{N}(\mu, \sigma^2)$，我们用样本均值 $\hat\mu = \frac{1}{n}\sum_i x_i$ 去估计 $\mu$。标准结果：

$$
\mathbb{E}\,[(\hat\mu - \mu)^2] \;=\; \frac{\sigma^2}{n}
$$

两边取 log：

$$
\log \mathrm{Error} \;=\; -\log n + 2\log\sigma
$$

这就是一条 **斜率为 −1 的 log-log 直线**。换句话说，"估计均值"这个最简单的统计任务本身就是一条 scaling law。一般来说，所有经典的参数估计问题（包括线性回归）给出的都是 $1/n$ 或 $1/\sqrt{n}$ 的率，对应 log-log 斜率 −1 或 −0.5。

但是——**真实 LLM 的斜率根本不是这个数量级**（Slide 18）：机器翻译 α ≈ 0.13，语音 α ≈ 0.3，语言模型 α ≈ 0.095。这些斜率比 0.5 甚至 1 小得多。这就告诉我们：经典参数估计不足以解释 LLM 的 scaling，一定有别的东西在起作用。

**Toy 2：非参数回归（Slide 19）**

神经网络的灵活性意味着它更像一个非参数估计器——能拟合任意光滑函数。这种估计器的 scaling 率会完全不同。

举例：$x_i$ 在 $d$ 维单位方盒里均匀分布，$y_i = f(x_i) + \mathcal{N}(0,1)$，要估计光滑的 $f$。最简单的非参方法是把空间切成小盒子，每个盒子里取 $y$ 的均值当作 $f$ 的估计。

假设盒子边长选 $n^{-1/4}$（$d=2$ 的情形）：总共 $\sqrt{n}$ 个盒子，每个盒子里有 $\sqrt{n}$ 个样本，每个盒子内均值估计的误差是 $1/\sqrt{n}$（这就是 Toy 1 的结论）。综合起来，整体误差大约是 $n^{-1/2}$。

推广到 $d$ 维：

$$
\mathrm{Error} \;\approx\; n^{-1/d}
$$

log-log 斜率就是 $-1/d$。**维度越高，scaling 越慢**——这就是为什么神经 scaling 的斜率会这么小。

**Bahri 2021 的"本征维度"猜想（Slide 20）**

把 Toy 2 的思路推到 LLM：神经网络学到的其实是数据流形上的结构，所以真正决定 scaling 斜率的不是数据空间的名义维度，而是**数据的本征维度**。语言数据的本征维度很高 → α 很小。这个解释直观但并不严格，因为本征维度本身就非常难估。但作为一个 mental model，它把"为什么 LLM 的 α 这么小"解释得还挺自洽。

### 数据 scaling law 的几个延伸（Slide 21–24）

一旦你相信"log(data) vs log(error) 是直线"这件事，就能衍生出一大堆有用的工程工具。

**数据组成只影响截距，不影响斜率（Slide 22）**

Kaplan 2021 的发现很惊人：换训练数据的分布（比如加多少代码、加多少 web data）——**只会上下平移那条直线，不会改变斜率**。写成公式：

$$
\log \mathrm{Error} \;=\; -\alpha \log n + C_{\text{dataset}}
$$

只有 $C_{\text{dataset}}$ 会变。工程上这极其有用：**在小模型上做数据筛选实验，结论就能迁移到大模型**，不用每换一次 mix 就训一次 GPT-3。后来大量的 optimal data mixing 工作都是基于这个观察。

**重复数据的代价（Slide 23）**

互联网的高质量 token 没那么多，能不能把数据重复训？经验规律是：

- 大约前 **4 个 epoch** 以内，重复数据几乎等价于新数据；
- 超过 4 个 epoch，每一遍的边际收益迅速衰减。

这篇文章的参数化方式是定义一个"**有效数据量** $D'$"：你实际训了 $D$ 个 token（包含重复），但在 scaling law 里要代入的是 $D' < D$。$D'$ 随重复次数增加而饱和，$R_D^* \approx 4$ 是那个饱和点附近的常数。

**数据选择也要随规模调整（Slide 24）**

既然重复数据变得不值钱，那"是继续重复高质量数据，还是去收集低质量的新数据"就是一个权衡。CMU 一篇工作把两条轴放在一起拟 scaling law，就能算出在给定 token 预算下，最优的"高质量重复率 × 新数据引入量"配比。

### 这一节的直觉总结

- log-data vs log-error 的线性关系**跨任务、跨模型都很稳**；
- 理论上跟经典泛化上界是同一类东西，Toy 1 和 Toy 2 给出两种斜率的来源；
- 实用上，它让数据收集、筛选、mixing 这些决策都可以"在小模型上解决"。

---

## 三、模型工程的 Scaling Laws（Slide 26–39）

从数据那侧切到模型这侧。模型 scaling law 的目的很实际：**我要在大模型上做这些决策，但我只能在小模型上做实验，怎么办？** 本节按 Kaplan 2020 的顺序走：架构 → 优化器 → 深宽比 → batch size → learning rate。

### 架构：Transformer vs LSTM（Slide 28）

如果只有一个 LSTM 和一个 Transformer 要比，暴力做法是把它们都 scale 到 GPT-3 规模跑一遍——做不起。Scaling law 的做法是：在一大串不同规模的 compute 预算下分别训 LSTM 和 Transformer，画 log-log 曲线。

Kaplan 2020 看到的是：两条曲线**斜率几乎一样，但 Transformer 整体比 LSTM 低了大概一个常数**（课堂里粗估大约 15× compute 差距）。也就是说，**无论你未来 scale 到多大，LSTM 都会稳定地落后一个常数倍**——这基本就判了 LSTM 的死刑：不值得为它继续投入。

这个论证模式很典型：**当两条 scaling law 曲线平行时，小模型上的胜者就是大模型上的胜者**，你甚至不需要真的 scale 上去验证。

### 一次性扫很多架构（Slide 29，Tay et al.）

Google 做过一次大规模 ablation：把一堆候选架构和 Transformer baseline 画在同一张 log-log 图上。结论是：在 scaling 上真正能**持续**战胜 Transformer 的只有两类——

- **Gated Linear Unit（GLU）类**，比如 SwiGLU；
- **Mixture of Experts（MoE）**。

这两个恰好就是今天 frontier 模型都在用的东西。Performer 这类"看上去很酷"的架构实际上输给了 baseline。Scaling law 在这里提供了一个非常干脆的证据链。

### Adam vs SGD（Slide 30）

Hestness 2017 对优化器做过类似的 scaling 对比（那是 2017 年，所以 baseline 里还有 Recurrent Highway Networks 这种老架构）。看到的模式跟 Transformer vs LSTM 一样：**Adam 和 SGD 的斜率一样，Adam 常数倍占优**。结论同样是：Adam 的优势是"结构性"的，不会被 scale 吃掉。

### 深度还是宽度？（Slide 31–33）

Kaplan 固定参数量扫层数的结果：**1 层和 2 层差别巨大**（1 层根本学不到东西），**2 层以上差别就非常小**——曲线是一个相当平的盆地。这对上了 Lec 3 里给你的那条 rule of thumb：aspect ratio（hidden size / 层数）在大约 10–100 之间都差不多，没必要锱铢必较。

这里有一个极其重要的坑（Slide 32）：**如果你把 embedding 参数算进"参数量"里**，scaling 曲线会出现一个奇怪的弯折；只有用 **non-embedding 参数**去画，才是干净的幂律直线。原因也好理解——embedding 参数只负责查表，不像 FFN/attention 那样真正参与每一步计算。MoE 时代这个问题更严重：稀疏激活的参数到底该折算成多少"等效 dense 参数"？最近几年有专门的 scaling law 论文在处理这个。

然后 Kaplan 在几个不同规模（50M、270M、1.5B）上都扫了 aspect ratio vs loss（Slide 33）。发现这几条曲线**形状几乎一样，最低点也几乎对齐**。意义是：**你在 50M 上扫出来的最优 aspect ratio，可以直接搬到 1.5B 用**。这才是 scaling law 思维真正的威力——你不再是"在部署规模上调超参"，而是"在小规模上调，相信规律会外推"。

### Batch Size 和 Critical Batch Size（Slide 34–36）

batch size 有两个区域：

- **batch 很小**（小于某个"噪声尺度"）时，把 batch 翻倍 ≈ 多走一步梯度——免费加速；
- **batch 变得很大**之后，翻倍 batch 几乎不减少达到目标 loss 所需的步数——纯浪费 compute。

这两个区域的交界点叫 **critical batch size**：

$$
B_{\text{crit}} \;\approx\; \frac{\text{达到目标 loss 所需的最少样本数}}{\text{达到目标 loss 所需的最少梯度步数}}
$$

一个反直觉的现象（Slide 35）：**loss 目标越低，$B_{\text{crit}}$ 越大**。课堂给出的直觉是：loss 越低意味着更新要越精细，需要更低的学习率和更干净的梯度，而降低梯度噪声的一种方式就是用更大的 batch 去平均。所以训练后期天然需要更大的 batch——这就是 Llama 3 训练里"**中途逐步把 batch size 调大**"的依据。

Slide 36 进一步研究的是：当 compute 预算变大时，最优 batch 怎么变？Kaplan 的结论是**最优 batch size 会随 compute 单调增大**，而**达到最优 loss 所需的步数近似不变**。这对数据并行是极大利好——大 batch 更好切，并行效率更高。

### Learning Rate 与 muP（Slide 37）

如果你什么都不做地 scale 模型（比如只是把 MLP 的宽度从 1024 调到 4096），你会发现**最优学习率随宽度变小**（rule of thumb 是 $\text{lr}^* \propto 1/\text{width}$）。这带来一个麻烦：每换一个 scale 都要重扫 lr，甚至要为最优 lr 本身再拟一条 scaling law——误差叠误差，非常脆。

**muP（maximal update parameterization，Yang et al. 2022）** 换了个思路：既然 lr 随 scale 飘是参数化选得不好造成的，那就**重新设计参数化方式**，让最优 lr 跟 scale 无关。具体做法是按宽度对不同层的学习率和初始化做特定的缩放（下一讲会详细推），效果是——**在最小的模型上调一次 lr，就能直接迁移到最大的模型**。

今天几个 frontier lab 都在这条路上走：Yao et al. 2024 对 muP 做了改进；Meta 最近在 Llama 4 的宣传里提到一个自研的 **metaP**（细节没公开）。这些东西之所以火，是因为**它们把"lr 要随 scale 外推"这个风险点直接消除了**。

### 一个重要警告：下游任务没这么听话（Slide 38）

到目前为止我们一直在看 train loss 或 perplexity 对 compute 的 scaling——这上面 scaling law 极其稳。但**一旦换到下游 benchmark（SuperGLUE、MMLU 这些）上，曲线就乱得多**：不同架构、不同超参的模型在 perplexity 上差不多，但在下游上可能差一截。

这在状态空间模型（SSM）那边尤其明显：SSM 在 language modeling perplexity 上可以跟 Transformer 打平，但在 in-context learning、长依赖 QA 这些下游任务上常常明显吃亏。**不要把 perplexity scaling 直接当成能力 scaling**。

### 基于 scaling law 的设计流程（Slide 39）

把前面的所有内容归结起来，整个流程其实就三步：

1. 训一组小模型，覆盖几个数量级的 compute；
2. 在这些小模型上确立一条 scaling law（验证 log-log 线性，拟出斜率和截距）；
3. 用这条规律**预测大模型的最优超参**，一把梭。

特别地，当两条候选的 scaling law 曲线平行（比如 Adam vs SGD），连外推都不需要——**小模型上赢的就是大模型上赢的**。唯一显著的例外是 learning rate，因为它本身对 scale 就敏感，要么显式去拟 lr scaling，要么上 muP 把这个问题消掉。

---

## 四、联合 Data-Model Scaling 与 Chinchilla（Slide 40–50）

现在到了 scaling law 对工业界影响最大的一段——它直接决定了今天所有大模型"训多大、训多少 token"的答案。

### 问题设定（Slide 40–42）

2021–2023 的现实是：互联网数据近乎无限，真正稀缺的资源是 **total FLOPs**。问题变成：给定一个 FLOPs 预算，训一个巨大但没训透的模型，还是一个小模型但训得很久？两个极端显然都浪费，中间必然有一个最优点。

联合 scaling law 给出了这个最优点的函数形式。两种等价写法：

- **Rosenfeld 2020**：$\mathrm{Error}(n, m) = n^{-\alpha} + m^{-\beta} + C$
- **Kaplan 2020**：$\mathrm{Error}(n, m) = m^{-\alpha} + n^{-1}\,\beta$

（$n$ = 参数量，$m$ = token 数量，$C$ = irreducible error。两种写法一阶等价，形式有点"经验凑"，但拟合 3D 曲面时非常准。）

Rosenfeld 的演示很漂亮（Slide 41）：三维坐标是 $(n, m, \text{loss})$，散点是真实 run，曲面是拟合。结果是曲面几乎完美贴合所有散点——更关键的是，**只用小 $n, m$ 的那一半散点拟曲面，就能精确外推到大 $n, m$**。这给了"小规模外推"第一次定量层面的证据。

然后就是 Kaplan 的经典图（Slide 42）：固定 FLOPs 预算，扫参数量 $n$，用 FLOPs ≈ $6\,n\,m$ 自动确定 token 数 $m$——每条等 compute 曲线都是一个 U 形，最低点就是这个 compute 下"最省 loss 的模型大小"。

### Chinchilla 对 Kaplan 的修正（Slide 43–44）

**Chinchilla（Hoffmann et al. 2022）**说：Kaplan 的估计严重偏小——Kaplan 得出的 "tokens/param 最优 ≈ 2" 根本不对，真实值应该更接近 20。差别从哪来？

差在 **learning rate schedule 的处理**。现代 Transformer 训练都用 cosine schedule：warmup → 余弦退火到一个很小的 lr。这个 schedule 有个硬约束——**必须走完整个周期才能得到"好模型"**，因为最后那段 lr cool-down 对 loss 的影响很大。

Kaplan 的问题是：他把一次长训练中间的不同 checkpoint 当作"不同 token 数下的模型"来画曲线。但这些 checkpoint 对应的其实是"用错 schedule 训出来的短模型"——在那个点上 lr 还没降下来，loss 自然不是该 token 数下真正能达到的最优值。这个系统偏差把 Kaplan 的"最优 tokens/param"算低了。

Chinchilla 的做法是**每个小模型都独立训练到完整的 schedule 结束**，每个 run 有自己正确的 cosine——于是得出了完全不同的结论。

### Chinchilla 的三种拟合方法（Slide 45–48）

Chinchilla 论文里给了三种方法，前两种结论一致，第三种（论文版）偏了一点，后来被改回来——这部分后面讲。

**方法 1：Minimum over training curves（Slide 46）**

训很多不同 size 的模型，每个都有一条完整的"FLOPs vs loss"曲线。把所有曲线画在一张图上，**取下包络**——也就是对任意 FLOPs 预算，所有模型里能达到的最小 loss。这个下包络本身是一条幂律直线。从包络上读出"每个 FLOPs 预算下哪个模型 size 取到了最低"，这就给出了最优 $n$ 关于 FLOPs 的 scaling。

**方法 2：IsoFLOPs（Slide 47）——这个最经典**

选几个固定的 FLOPs 预算（比如 $10^{18}, 10^{19}, 10^{20}, \dots$），对每个预算：

- 扫一个范围的模型参数量 $n$；
- 每个 $n$ 下 token 数 $m$ 由 $6nm = \text{FLOPs}$ 自动定下来；
- 画出 loss vs $n$，你会看到一条漂亮的 **U 形曲线**；
- 取 U 形的最低点（或者用二次函数拟合取极小）。

然后把"每个 FLOPs 预算下的最优 $n$"和"最优 $m$"分别画成 FLOPs 的函数——两条幂律直线。从这两条线就能读出 **tokens/param 的最优比例**。Chinchilla 在它关心的 compute 量级上得到的答案是："最优参数量 63–67B，tokens/param ≈ 20"。

方法 2 几乎是今天工业界做 scaling law 的**标准动作**。

**方法 3：Joint fit（Slide 48）**

直接对 Rosenfeld 的联合函数形式做最小二乘——在 $n, m$ 网格上扫一批模型，整体拟 3D 曲面，然后在曲面上解最优。原理上最严格，但因为要拟的参数多、曲面拟合误差大，在 Chinchilla 原论文里跟方法 1、2 差了大约 0.03 的斜率，长期是一桩悬案。

**方法 3 的反转（Slide 49，Besiroglu et al. 2024）**

去年（2024）Epoch AI 的几个研究员去复现方法 3：原始数据拿不到，他们就用从原论文的图里**用取点工具反推数据点**，然后重跑方法 3 的拟合。发现——原论文的拟合残差是系统性非零的（回归的基本健全性检查都过不了），修正之后，方法 3 的结果和方法 1、2 完全一致。

所以这个故事的结局是：Chinchilla 论文的数据本身是对的，只是那次曲线拟合做错了；修正之后三个方法完美统一。这算是 replication 证实而非推翻的一个好例子。

### Train-optimal ≠ Deploy-optimal（Slide 50）

Chinchilla 回答的问题是："**给定训练 compute**，怎么训出最强的模型？" 但真实部署里大头不是训练，而是**推理**——模型训好之后要被调用很多次。

这意味着：即使大模型和小模型在训练时能打平，小模型的推理成本要低得多——**值得故意 over-train 小模型**，也就是用远超 Chinchilla 最优的 token 数去喂它。于是这些年 tokens/param 比例持续膨胀：

- GPT-3：2 tokens/param
- Chinchilla：20
- LLaMA 65B：22
- Llama 2 70B：29
- Mistral 7B：110
- Llama 3 70B：215

趋势很清楚：**预计使用量越大，越划算把训练预算往"小模型多 token"那一侧倾**。这解释了为什么今天大家疯狂堆 token 而不是堆参数。

---

## 五、Scaling Law 不止适用于 LM（Slide 51）

讲者自己组（Gulrajani et al. 2023）曾经想知道：文本扩散模型到底能不能 scale、tokens/param 比是多少？他们的办法就是**把 Chinchilla 的 IsoFLOPs 分析原样套在 diffusion LM 上**，同时对 autoregressive LM 也跑一遍作对照。

结果是：两边都有干净的 U 形 IsoFLOPs 曲线和整齐的幂律最小包络，只差一个常数截距。这件事的含义比单次 benchmark 比较重要得多——**IsoFLOPs 分析可以作为一个通用工具，用来公平地比较不同模型家族的 compute-效率**。

---

## 六、把整节课串起来（Slide 52–53）

这节课真正想让你带走的东西就三件：

**log-linearity 无处不在，但要正确理解它。** 数据量、模型参数量、compute，这三者对 loss 的关系都近似是 log-log 直线。理论上有 Toy 1/Toy 2 两套直觉（参数估计率、非参数估计率 + 本征维度）支撑，实证上跨任务、跨模型都成立。

**scaling law 让你能在小模型上做大模型的决策。** 架构选 Transformer + GLU + MoE、优化器选 Adam、深宽比随便选、batch 越训越大、lr 用 muP 处理——这些结论不是靠 GPT-3 级别的 ablation 得出来的，而是在小模型上拟 scaling law、外推过去的。

**给定预算时有清楚的资源分配方式。** 想知道训多大模型、用多少 token？跑 IsoFLOPs，取 U 形最低点，读出最优配比。只是注意两件事：第一，这是"train-optimal"，推理量大时要故意 over-train；第二，perplexity scaling 不等于下游能力 scaling，下游任务要单独评估。

下一节课（Scaling Laws 2）会把这些工具用到更多真实的 case study 上。

---

## 附：几个容易混淆的点

- **数据 scaling 画图时**，模型要足够大不会被数据量饱和，否则看不到幂律。
- **参数量**不要把 embedding 算进去，否则曲线有弯折；MoE 要算"等效 dense 参数"。
- **两条平行的 scaling 曲线**意味着小模型上的赢家就是大模型上的赢家（架构、优化器常常是这样）。
- **lr 是个例外**，它会随 scale 漂移，要么显式拟、要么用 muP。
- **Critical batch size 会随训练进程变大**，所以 Llama 3 中途涨 batch 不是调参魔法，是 scaling law 的直接后果。
- **Chinchilla 的 20 tokens/param 是 train-optimal**，不是 deploy-optimal。真实部署下比例应该远比 20 大。
- **Kaplan 比 Chinchilla 小一个数量级**的根源是 learning rate schedule 的处理方式——Kaplan 在一次 run 里取中间 checkpoint，那些点对应的是"用错 schedule 的训练"，系统性低估了最优 token 数。

