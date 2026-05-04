
# CS336 Lecture 16 · 可验证奖励的强化学习（RLVR）助学笔记

> 配套资料：`2025 Lecture 16 - RLVR.pdf` + 课堂字幕
>
> 本笔记按讲义顺序组织，力求用最少的背景假设讲清每一页内容。阅读时建议左边放 PDF，右边看本笔记，一边对照一边理解。

---

## 目录

1. [前情回顾：RLHF 与 DPO 的世界](#一前情回顾rlhf-与-dpo-的世界)
2. [RLHF 的两大隐患：过优化与模式崩溃](#二rlhf-的两大隐患过优化与模式崩溃)
3. [为什么要转向可验证奖励（RLVR）](#三为什么要转向可验证奖励rlvr)
4. [PPO：从策略梯度一路演化而来](#四ppo从策略梯度一路演化而来)
5. [GRPO：为语言模型量身定做的简化版](#五grpo为语言模型量身定做的简化版)
6. [GRPO 的缺陷与 Dr.GRPO 的修正](#六grpo-的缺陷与-drgrpo-的修正)
7. [案例一：DeepSeek R1](#七案例一deepseek-r1)
8. [案例二：Kimi K1.5](#八案例二kimi-k15)
9. [案例三：Qwen 3](#九案例三qwen-3)
10. [全课总结](#十全课总结)

---

## 一、前情回顾：RLHF 与 DPO 的世界

> 对应 PDF Page 3 – 10

### 1.1 我们要解决什么问题

RLHF（基于人类反馈的强化学习）的设定是：

- 我们只能收集到**成对偏好数据**：给定 prompt $x$ 与两个候选回复 $y_w, y_l$，人类告诉我们哪个更好。
- 我们希望训练出一个语言模型策略 $\pi_\theta$，让它最大化某个**隐含的奖励**，且这个奖励能解释上面这些偏好。

原始 RLHF 目标写成带 KL 正则的期望奖励最大化：

$$
\max_{\pi_\theta}\; \mathbb{E}_{x\sim D,\, y\sim \pi_\theta(\cdot\mid x)}\big[r(x,y)\big] - \beta\, \mathrm{KL}\big(\pi_\theta\,\|\,\pi_{\text{ref}}\big)
$$

难点在于：**策略 $\pi_\theta$ 出现在期望的下标里**（也就是我们要从它里面采样），这不是一个普通的最大似然问题，所以需要 PPO 这类 RL 算法。

### 1.2 DPO 为什么省事

DPO 的核心思想是：把 RL 问题**等价地**转换成一个监督学习问题，从而绕过采样与在线 rollout。推导三步走：

1. **非参数假设**：假设策略类可以是所有函数，那上面那个带 KL 的最优化问题存在闭式最优解：

   $$
   \pi^\star(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\left(\tfrac{1}{\beta}\,r(x,y)\right)
   $$

2. **反解奖励**：把上式反解，把 $r$ 写成策略的比值：

   $$
   r(x,y) = \beta\,\log\frac{\pi^\star(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\,\log Z(x)
   $$

3. **代入 Bradley–Terry 偏好模型**（人类选 $y_w$ 优于 $y_l$ 的概率由奖励差的 sigmoid 给出），配分函数 $Z(x)$ 会在相减时抵消，最终得到 DPO 损失：

   $$
   \mathcal{L}_{\mathrm{DPO}} = -\,\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\!\left(\beta\,\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\,\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
   $$

直觉理解：**这就是在成对偏好数据上做极大似然，只不过我们对奖励做了一个用策略参数化的"再参数化"**。训练就是普通的监督学习——不用 rollout、不用 reward model、不用复杂 RL。

### 1.3 DPO 的梯度：好的上调，坏的下调

对 DPO 损失求梯度，可整理为一个非常直观的形式：

$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}} \propto -\,\beta\,\sigma(\hat r_l - \hat r_w)\cdot\Big[\,\underbrace{\nabla_\theta \log \pi_\theta(y_w\mid x)}_{\text{推高好的}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l\mid x)}_{\text{压低坏的}}\Big]
$$

其中 $\hat r_w, \hat r_l$ 是"隐含奖励"的估计。这里有两个关键点：

- 前面的权重 $\sigma(\hat r_l - \hat r_w)$ 相当于**预测误差**：当模型已经把好的给高分、坏的给低分时，$\sigma$ 趋近 0，这个样本就不更新；当模型判断错了，权重接近 1，大力更新。
- 实际更新就是**把好例子的对数似然往上推、把坏例子往下压**。

> 🎯 **口诀**：RL 算法几乎都可归结为"好事上调、坏事下调"，差别只在于**如何定义"好事"以及"上调多少"**。

### 1.4 DPO 的常见变体

DPO 推出后掀起了"一大堆 *PO"的浪潮，其中 Tulu 3 论文里用得比较多的两个：

- **SimPO**：做两处简单修改。
  1. 用**回复长度**对更新进行归一化（length-normalize），避免模型偏好长回复。
  2. **丢掉参考策略 $\pi_{\text{ref}}$**。代价是失去了"策略比值"的数学故事，只剩下朴素的"抬好压坏"。
- **Length-normalized DPO**：只做长度归一化，保留参考策略。

> 💡 **经验教训**：Tulu 3 里发现，**只要 SFT 做得够好，PPO/DPO 的相对优劣可能被"吃光"**。RL 的结论**严重依赖具体实验设定**（base model、数据、训练细节），不要把单篇论文的结论当真理。

---

## 二、RLHF 的两大隐患：过优化与模式崩溃

> 对应 PDF Page 11 – 13

### 2.1 过优化（Over-optimization）

**现象**：随着你越来越卖力地优化策略，**代理奖励**（你的 reward model 打的分）一直涨，但**真实奖励**（真实人类偏好）先涨后跌。

这本质上就是**过拟合**，只不过换了个更花哨的名字。原因：你的 reward model 是在有限偏好数据上训出来的，它并不等于真实人类偏好。继续优化就在 reward model 的**漏洞**上猛钻。

讲义给出三张对比图：

| 场景 | 是否出现过优化 |
|---|---|
| 人类偏好 | ✅ 出现 |
| 噪声 LM 偏好 | ✅ 出现 |
| 无噪声 LM 偏好 | ❌ 几乎没有 |

→ **结论**：过优化主要由**偏好数据的噪声与复杂性**驱动。你应当预期训练曲线像左图——proxy reward 一路向上，真实胜率却中途掉头。

### 2.2 模式崩溃 / 校准丧失

在预训练/SFT 阶段，我们做的是**分布匹配**（distribution matching），模型是一个概率模型，概率有物理意义。

但 RLHF 训出来的**只是一个策略，不一定再是概率分布**。多篇论文（Anthropic、GPT-4 技术报告等）都发现：

- RLHF 之后模型在 temperature=1 下往往**过度自信**，校准（calibration）变差。
- 这并不算 bug——只要 reward 里不包含 calibration，就没有理由期待它被保留。

→ **别把 RLHF 后的模型当成校准良好的概率模型来用**，尤其是做不确定性估计时要小心。

---

## 三、为什么要转向可验证奖励（RLVR）

> 对应 PDF Page 14 – 16

RLHF 有几个天花板：

- 人类反馈**难采集、难扩规模、容易被攻击（reward hacking）**。
- **过优化几乎无法避免**。

那么能不能换个打法？看看 RL 历史上真正大获成功的地方：AlphaGo、AlphaFold。这些领域的共同点是：

- **奖励是客观可验证的**（围棋胜负、蛋白折叠能量）。
- **可以在大规模、低成本下自动评估**。

如果我们把 LLM 训练也放到这种领域——比如数学题（答案对错可判）、代码（跑测试）——就能复用过去几十年 RL 的成功经验。这就是 **RLVR（RL from Verifiable Rewards）** 的核心动机。

本节课的两大块：

1. **算法**：PPO → GRPO → GRPO 的变体。
2. **案例研究**：DeepSeek R1、Kimi K1.5、Qwen 3 这三个公开的中文社区开源推理模型。

---

## 四、PPO：从策略梯度一路演化而来

> 对应 PDF Page 17 – 28

要理解 GRPO，先得理解 PPO 为什么长成那样。PPO 是一步步在修补策略梯度的缺点中长出来的。

### 4.1 起点：策略梯度（Policy Gradient / REINFORCE）

目标：

$$
\max_\theta\; J(\theta) = \mathbb{E}_{z\sim \pi_\theta}[R(z)]
$$

用对数求导技巧得到梯度（这一步是关键恒等式，一定要记住）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{z\sim \pi_\theta}\big[R(z)\,\nabla_\theta \log \pi_\theta(z)\big]
$$

直觉：**按奖励的符号和大小，相应地提升或压低样本 $z$ 的对数概率**。

**问题**：

1. **方差极大**。一次只用一个样本估计期望，梯度非常嘈杂。
2. **纯 on-policy**。每次更新都必须用**当前策略**采样的数据。换句话说，**每走一小步都要做一次昂贵的 rollout（让模型生成）**，效率低下。

> 📌 在你的作业 5 里会亲身体会到这一点：**RL 里最慢的一步是 rollout**（不是反传），因为要调用自回归生成。

### 4.2 第二步：TRPO——让旧策略的数据也能用

想法：**让我能用"旧策略 $\pi_{\theta_{\text{old}}}$"采的数据去更新"新策略 $\pi_\theta$"**，这样一次 rollout 可以做多步梯度更新。

工具：**重要性采样（importance sampling）校正**。

$$
\nabla_\theta J \approx \mathbb{E}_{z\sim \pi_{\theta_{\text{old}}}}\!\left[\frac{\pi_\theta(z)}{\pi_{\theta_{\text{old}}}(z)}\,A(z)\,\nabla_\theta \log \pi_\theta(z)\right]
$$

但如果 $\pi_\theta$ 和 $\pi_{\theta_{\text{old}}}$ 差得太远，比值会爆炸。于是 TRPO 再加一个**信赖域约束**：KL 散度不能超过 $\delta$。

这里 $A(z)$ 是**优势函数**（比纯 $R(z)$ 方差更小），细节先不展开。

### 4.3 第三步：PPO——用 clip 代替 KL

TRPO 要解约束优化，工程上很麻烦。PPO 的关键 idea：**直接把比值剪裁（clip）住**，用无约束优化得到近似的信赖域效果。

PPO-Clip 目标（按 token）：

$$
\mathcal{L}^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\big(r_t(\theta)\,A_t,\; \mathrm{clip}(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon)\,A_t\big)\right]
$$

其中 $r_t(\theta) = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$，典型 $\varepsilon = 0.2$。

**为什么这样做能起到"保持靠近旧策略"的效果？**
一旦 $r_t$ 超过 $1+\varepsilon$（或低于 $1-\varepsilon$），$A_t$ 为正/为负时奖励都会被"封顶"。**新策略离得再远也拿不到更多奖励**，于是策略没有动力走太远。这是一种软化的信赖域。

### 4.4 PPO 的"理论 vs 实战"落差

> PDF Page 20 — 那个著名的 "37 Implementation Details of PPO" 博客不是在开玩笑。

一个标准 PPO-for-RLHF 实现里要同时维护的部件：

- 🎯 **策略网络**（language model 本体）
- 💰 **奖励模型**（reward model）
- 📊 **价值网络 $V_\phi$**（和策略一样大的另一个网络，用于估计优势）
- 🔬 **广义优势估计 GAE**：
  $$
  A_t = \sum_{k\ge 0}(\gamma\lambda)^k\,\delta_{t+k},\quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
  $$
- 🧩 **reward shaping**：per-token KL 惩罚 + last-token 真实奖励。
- 📐 **重要性比值裁剪**、梯度裁剪、KL 数值稳定 hack 等等。

**为什么语言模型里 PPO 其实接近于 contextual bandit？**
语言建模里，给 prompt → 生成完整回复 → 拿到奖励，**中间没有环境状态转移**。严格来讲就是"上下文赌博机"。有趣的是，如果你令 $\gamma=\lambda=1$，GAE 就退化成"回报 − 价值基线"，整个 PPO 就简化成带 baseline 的策略梯度——这个简化版**效果也很好**。

### 4.5 AlpacaFarm 代码走读（PDF Page 22 – 27）

老师带大家过了一遍 AlpacaFarm 的 PPO 实现，需要重点记住的几点：

- **外循环**：采一批 rollout → 在这批数据上做几步梯度 → 继续。和普通训练循环长得很像。
- **loss 计算**：同时更新**价值函数**（回归到真实回报）和**策略**（PPO-Clip 目标）。`cliprange=0.2` 是典型值。
- **Rollouts**：调用 inference，得到样本序列；如果 reward/value/policy 用了不同 tokenizer，还得做 retokenization。
- **Reward shaping**：per-token KL 惩罚加到 token reward 上，真正的任务 reward 只在最后一个 token 给。
- **KL 数值稳定 trick**：当 $\log \pi_\theta - \log \pi_{\text{ref}} < 0$ 时，真 KL 的数值估计容易出问题，代码里会 clamp 到 0，所以严格说"那不是 KL 了，只是 KL 的一个近似"。这是实战中常见的一种"原理上不对但能跑"的 hack。
- **"更新就是普通的反向传播"**：你不需要显式实现策略梯度的那种公式。你只要写下加权损失 $-R\cdot \log \pi_\theta$，再对 $\theta$ 求导（PyTorch autograd 就行），得到的梯度就是策略梯度。记住：**对 $R$ 里关于 $\theta$ 的部分要 stop-gradient**。

### 4.6 PPO 训练中你会看到什么

- 总奖励（含 reward model 项）**稳步上升**。
- KL 相关的"负奖励"**缓慢下降**。
- 因为是 bandit 任务，曲线比较稳定，**不会像真正的 RL 任务那样剧烈抖动**。

→ PPO 能工作，但**又重又难**。这就给 GRPO 留足了动机：**能不能把价值网络干掉？**

---

## 五、GRPO：为语言模型量身定做的简化版

> 对应 PDF Page 29 – 33

### 5.1 为什么需要第三种 RL 算法

- **PPO**：实现复杂；价值网络和策略一样大（显存翻倍）；需要额外调参。
- **DPO**：要求数据天然成对；本质是离线算法（虽能迭代 online 化，但不是它的主场）。在"做一批数学题判对错"这种场景下不适用。

→ 我们要一个既简单又适合**逐题多回复、可验证奖励**场景的算法。

### 5.2 GRPO 的核心思想：组内 z-score 作为优势

GRPO 的出发点：**直接沿用 PPO 的 clip 框架**，但把那个又贵又烦的**优势估计**换掉。

**关键：一次对同一个 prompt 采样一"组" G 个回复**。用组内 z-score 当优势：

$$
A_i = \frac{r_i - \operatorname{mean}(\{r_1,\dots,r_G\})}{\operatorname{std}(\{r_1,\dots,r_G\})}
$$

**为什么"组"是自然的单位？**
数学题有难有易。同一道题的回复互相比较才公平——某回复比**同组平均**好多少，就是它真正该得到的"额外信用"。这其实就是 policy gradient 中的 **leave-one-out 基线**（减去同组其他样本的均值）。

GRPO 的完整目标（与 PPO 一致的 clip 结构 + 新的优势 + per-token KL 正则）：

$$
\mathcal{L}_{\mathrm{GRPO}}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\!\big(r_{i,t}A_i,\;\mathrm{clip}(r_{i,t},1-\varepsilon,1+\varepsilon)A_i\big) - \beta\,D_{\mathrm{KL}}[\pi_\theta\|\pi_{\text{ref}}]\right]
$$

$r_{i,t} = \dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}$ 是 token 级重要性比值。

> ⚠️ **注意（后面会批判）**：上式对每条回复都按 $1/|o_i|$ 做了**长度归一化**；优势也除以了 std。这两处是 Dr.GRPO 讨论的"bug"所在。

### 5.3 一个巧妙的 KL 估计

GRPO 里使用的 KL 估计不是普通的 $\mathbb{E}[\log(\pi_\theta/\pi_{\text{ref}})]$，而是：

$$
\widehat{D}_{\mathrm{KL}}[\pi_\theta\|\pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(o\mid q)}{\pi_\theta(o\mid q)} - \log\frac{\pi_{\text{ref}}(o\mid q)}{\pi_\theta(o\mid q)} - 1
$$

这是一个**方差更低的无偏估计**（控制变量技巧：加了一项期望为 0 的修正，使估计更稳定），而且保证非负。如果你以后需要从样本估 KL，可以直接用这个。

### 5.4 纯 online 的极简特例

如果每次采完一组 rollout 只走**一步**梯度（$\pi_\theta = \pi_{\theta_{\text{old}}}$，比值永远是 1），clip 完全不起作用。算法就退化成：

$$
\nabla_\theta \mathcal{L} \propto \sum_{i,t} A_i \cdot \nabla_\theta \log \pi_\theta(o_{i,t}\mid \cdot)
$$

——**就是一个加了组内归一化的朴素策略梯度**。简单到令人发指。

### 5.5 实现要点（Page 31 – 32）

老师推荐了一个玩具级实现：`McGill-NLP/nano-aha-moment`。外循环长这样：

1. 对每个 prompt 采样 G 个 rollout。
2. 每个 rollout 打分得到 $r_i$。
3. 对每组做 mean/var 归一化得到 $A_i$。
4. 算 KL 项。
5. 把损失丢进 autograd，走一步梯度。

**工程细节**：计算 std 时要加一个小常数避免数值爆炸，比如 `std + 1e-4`——你在作业里也必须这么做。

### 5.6 DeepSeek-Math 里的效果

在 DeepSeek-Math 原论文里对比了几种方法：

| 方法 | 一句话概括 |
|---|---|
| RFT / Online RFT | 只拿答对的回复来做 SFT（拒绝采样微调） |
| **GRPO + Outcome 奖励** | 只看最终答案对错 |
| GRPO + Process 奖励 | 每一步推理都打分（需要 PRM） |

三者都在涨，其中 process-level 奖励（蓝线）那时被认为最强。**但我们后面会看到，R1 阶段 DeepSeek 自己放弃了 PRM 路线**——这是一处关键转折。

---

## 六、GRPO 的缺陷与 Dr.GRPO 的修正

> 对应 PDF Page 34 – 36

Liu et al. 2025（Dr.GRPO 论文）指出：GRPO 的优势估计**在理论上违反了策略梯度定理**，而且这两个违反都对应着实际的副作用。

### 6.1 回顾：合法的 baseline 必须是什么

策略梯度恒等式告诉我们：可以从奖励里**减去任何不依赖于当前采样轨迹 $z$ 的量**（比如状态相关的 baseline $b(s)$），因为：

$$
\mathbb{E}_{z\sim \pi_\theta}[b(s)\,\nabla_\theta \log \pi_\theta(z)] = b(s)\,\nabla_\theta \underbrace{\textstyle\int \pi_\theta(z)\,dz}_{=1} = 0
$$

——加减 baseline 只减方差，不引偏差。

### 6.2 问题 1：除以 std 不是合法 baseline

$A_i = (r_i - \bar r)/\operatorname{std}(r)$ 里的 $\operatorname{std}(r)$ **是依赖于这批样本的随机量**，严格来说它不是一个合法的"乘性常数"。会发生什么？

**方向性分析**：当 std 很小，$A_i$ 被放大，整组样本被"加大力度"更新。**什么时候 std 会小？**

- 题目**太简单**：所有回复都对 → reward 全 1，std ≈ 0。
- 题目**太难**：所有回复都错 → reward 全 0，std ≈ 0。

→ **GRPO 系统性地放大了"太简单"和"太难"题目的更新强度**，反而冷落了处于"能做对一部分"的中等难度题。而直觉和课程学习告诉我们：**模型最该学的恰恰是中等难度的题**，这类题目才有稳定的学习信号。

### 6.3 问题 2：除以长度 $|o_i|$ 带来畸形激励

目标里 $\dfrac{1}{|o_i|}\sum_t$ 这一项对长度做了归一化。结果是：

- 当 $A_i < 0$（答错），模型的最优策略是**把回复写得越长越好**——这样每个 token 分摊到的负奖励更小。
- 当 $A_i > 0$（答对），模型的最优策略是**把回复写得越短越好**——放大正奖励。

**综合效果**：**模型倾向于在不会做的题上使劲瞎说，尽量拖长**。这可能正是你在某些 RL 后的推理模型里看到"特别长的胡言乱语"的原因！

### 6.4 Dr.GRPO 的修正

Dr.GRPO 提出的补丁很简单：

- **去掉除 std**：优势就是 $A_i = r_i - \bar r$（纯 leave-one-out baseline）。
- **去掉除 $|o_i|$**：不要按长度归一化。

实验结果：

- 任务奖励**不降反升**。
- 回复长度**稳定下来**，不再一路膨胀。

这说明"推理模型的回复越来越长"**未必是解题需要**，很可能只是**优化目标畸形造成的副作用**。当然这条结论还在讨论中，但值得警觉。

### 6.5 对"aha moment"的再评价

回过头看 R1-zero 论文里著名的两件"奇迹"：

1. **CoT 越来越长** → Dr.GRPO 指出：这可能只是目标函数里长度偏置的产物。
2. **模型学会了"aha"、回溯等反思行为** → Dr.GRPO 指出：把 DeepSeek-V3 base 直接跑在数学题上，**本来就会时不时说"aha, I can try..."**，并不是 RL 无中生有。

→ 不必过度神话 RL；R1-zero 做出来的是一个**好数学模型**，但"涌现"的说法有水分。

---

## 七、案例一：DeepSeek R1

> 对应 PDF Page 37 – 50

### 7.1 R1 的三大科学贡献

1. **正面结论**：**outcome-based 奖励 + GRPO 就能做出 o1 级别推理模型**（性能达标）。
2. **两条负面结论**（反而对社区影响巨大）：
   - **MCTS 路线并不必要**。
   - **PRM（过程奖励模型）这条路走不通**（至少在 R1 的设定下）。
3. **SFT 的重要性与数据收集方法的洞察**。

整体流水线（图示）：

```
DeepSeek-V3 (Base)
        │
        ├──► R1-zero：直接 GRPO on 数学题       （研究对照组）
        │
        └──► 长 CoT SFT  →  推理 RL (GRPO)  →  通用 SFT  →  通用 RLHF (GRPO)
                                    (R1 正式版)
```

### 7.2 R1-zero：受控的纯 RL 实验

**设置**：

- **Base**：DeepSeek-V3（预训练 + mid-training 之后、instruction-tuning 之前的原始模型）。
- **算法**：GRPO。
- **数据**：大批数学类题目（未公开）。
- **奖励**：
  - **准确率奖励**：答案对 → 1，错 → 0。二元判别。
  - **格式奖励**：强制输出落在 `<think>...</think>` 这样的思考标签内。

→ **不做任何 SFT，不做任何指令微调，直接在 base 上 RL**。结果已经逼近 o1。

> 💡 **意外重要细节**："格式奖励"看起来像是附送的小功能，但很多人反映它**对整个 RL 推理流程能否跑通至关重要**。

### 7.3 两个"奇迹"现象（及其反驳）

- **CoT 越训越长**：R1 论文暗示这是"模型学会用更多思考解更难的题"。Dr.GRPO 反驳：这可能只是长度归一化偏置。
- **Aha moment**：模型突然会说"Wait, let me reconsider..."。反驳：base model 本来就会偶尔说这种话。

→ **别神化这些现象**，但 R1-zero 作为一个数学模型确实很强。

### 7.4 R1 正式版：从研究到产品

光靠 R1-zero 做产品不行——它是个"数学天才白痴"，别的任务可能很差。R1 正式版加了若干步：

1. **长 CoT SFT 冷启动**：先在一批"带长推理过程的数据"上做 SFT，给模型一个基础"思考姿势"。数据来源论文写得很模糊。
   - 好处：**可读性 / 可解释性**。模型不会从一上来就输出天书。
2. **推理 RL**（GRPO）**加一个语言一致性奖励**。因为纯 RL 会出现思考链**中英文乱切**（你可能在 Grok 3 这类模型里也见过类似现象）。加一个奖励鼓励 CoT 全程用单一语言。
   - **代价**：消融显示加了语言一致性奖励后**性能略微下降**——为了可读性做了权衡。
3. **通用 SFT**（2 epochs）：
   - **推理型不可验证任务**（比如"证明 X 命题"）：无法自动判分，用 V3 自己做裁判。600k 样本。
   - **非推理型数据**：沿用 V3 的 SFT 数据集，200k 样本。
4. **通用 RLHF**：
   - 推理部分：复用 R1-zero 风格 RL。
   - 非推理部分：走 V3 的 RLHF 管线，**RL 算法仍然是 GRPO**（简洁划一）。

### 7.5 SFT 数据里的"少即是多"

老师提到自己学生与 Percy 合作的工作：从 Gemini 2.0 Flash Thinking 收集 **1,000 条长 CoT**，用来微调 Qwen 2.5，**数学榜单就大幅上涨**。

→ **base model 其实"本就会推理"**，SFT 只是在**激活并对齐推理姿势**。这也和 R1-zero 的成功相互印证。

### 7.6 蒸馏：把 R1 的推理能力灌给小模型

R1 生成 **80 万条 CoT**，用来 distill Qwen 2.5。32B 模型在 AIME 上从 ~50% 直接涨到 70%+。说明：**纯 SFT 蒸馏就能把大模型的推理能力有效传递给小模型**，哪怕不再做 RL。

### 7.7 两条负面结果：PRM 与 MCTS

- **PRM（过程奖励模型）**：给推理链的**每一步**打分，理论上是更丰富的反馈。但难点在**怎么训出一个靠谱的 PRM**——DeepSeek-Math 阶段曾寄予厚望，R1 阶段**全面放弃**，因为 outcome-based 简单 reward 已经足够好。
- **MCTS（蒙特卡洛树搜索）**：搜索派路线。**也没跑通**。

→ 目前整个社区的共识：**outcome-based reward + GRPO 是最强 baseline**，更花哨的方法还没给出明确增量。

---

## 八、案例二：Kimi K1.5

> 对应 PDF Page 51 – 60

### 8.1 为什么值得单独讲

- 和 R1 **同一时期发布**，是"对照样本"。
- **也用 outcome-based RL 达到 o1 水准**，但**算法细节不同**。看看"殊途同归"的另一个实现。

### 8.2 数据构造（比 R1 讲得细）

1. **话题均衡**：用 LM 对题目自动打标签，跨学科/领域尽量均衡。
2. **排除选择题与判断题**：这类题能蒙对，奖励容易被 hack。保留**短答案可 regex/LM 检查**的题。
3. **难度筛选（关键）**：让未经 RL 的 SFT 模型对每题生成 10 个回复，**只保留 best-of-8 都失败的题**。也就是说——**模型能偶尔做对的题都剔除了，保留当前"做不到但有机会学会"的题**。

   → 这正好和 GRPO 的设计初衷相配：**std 大 ⇔ 有学习信号**的题才留下来。Qwen 3 后来也沿用了这套做法。
4. **SFT 数据**：论文只说"做了 prompt engineering"，大概率是蒸馏自更强的模型，但未说来源。

### 8.3 Kimi 的 RL 算法：DPO-风格的推导，GRPO-风格的结果

**一个很优雅的推导故事，演化路径和 DPO 非常像**：

1. **同样的目标**：带 KL 正则的最大化期望奖励。
2. **非参数假设** → 写出最优策略的闭式解。
3. **反解**：

   $$
   r(x,y) = \beta\,\log\frac{\pi^\star(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\,\log Z(x)
   $$

4. **不代入 Bradley–Terry**（因为没有成对偏好）。而是：**让等式两边直接相等 → 加上平方损失**，强迫它们靠拢：

   $$
   \mathcal{L}_{\mathrm{Kimi}} = \mathbb{E}\!\left[\Big(r(x,y) - \beta\,\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} - \beta\,\log Z(x)\Big)^{\!2}\right]
   $$

5. **对 $\theta$ 求梯度**：最终整理后会发现**形式和 GRPO 非常像**：
   - 一个"带 baseline 的策略梯度"项（baseline 用批内平均）。
   - 一个和"$\log$-ratio 二次惩罚"等价的正则项（代替 PPO 里的 clip）。

→ **两种不同思路，殊途同归**：只要有 **"策略梯度 + 合适的 baseline + 保持靠近参考策略的正则"**，就能做出可用的 RL 算法。

### 8.4 长度控制：Kimi 比 R1 更未雨绸缪

R1 一度把"CoT 越来越长"当优点宣传。Kimi 则意识到：**CoT 长 = 推理成本高 = 真金白银的开销**。所以 Kimi 主动设计了一个**长度奖励**来压缩 CoT：

对一组 rollout，定义（L 为序列长度）：

$$
\lambda_i = \frac{L_{\max} - L_i}{L_{\max} - L_{\min}} - 0.5 \in [-0.5,\,+0.5]
$$

即**组内最短的 $\lambda \approx 0.5$，最长的 $\lambda \approx -0.5$**。长度奖励定义为：

- 答案正确：reward 就是 $\lambda$ → **答对越短越赚**。
- 答案错误：reward 取 $\min(0,\lambda)$ → **只惩罚偏长的错答，不奖励瞎短**。

直觉效果：

- **正确答案**：强烈激励**更短**。
- **错误答案**：激励落在**中等长度附近**（不鼓励用瞎拉长来逃避惩罚，也不鼓励瞎短蒙）。

**关键工程细节**：**训练初期不能开这个长度奖励**。否则模型没学会解题前就被按进"最短局部极小"（反正都错，索性全写最短）爬不出来。**在 RL 走过一段无约束训练之后再加入**。

### 8.5 其他细节

**课程学习**：

- 数据集事先标好难度。
- 先易后难。
- 动态采样：采样概率正比于 $(1 - \text{success\_rate})$——做得很好的题不再抽。

**奖励设计**：

- **代码题**：题目自带 ground truth 解 → 用它生成**新测试用例** → 跑代码看通过率。
- **数学题**：训练一个 **800k 样本规模的 CoT 奖励模型**，专门做"两个答案是否等价"的判定。比 regex 或 SymPy 的符号判等更鲁棒。
  - 有趣的是：虽然名义上叫"RL from verifiable rewards"，但判等这一步其实用了模型——因为本质上只是**高级字符串匹配**，模型够准确就行。

### 8.6 RL Infrastructure：Kimi 是少数讲系统的

RL 训 LLM 为什么比预训练更难利用 GPU？

1. **必须做 rollout**：推理很慢。
2. **训练↔推理切换**：两边常用不同框架，**权重必须在两端来回同步**。
3. **长 CoT → batch 不均**：一组里有的回复短，有的超长，padding 浪费严重。

Kimi 的做法（标准但完善）：

- **训练 worker** 和 **推理 worker** 分开。
- 推理侧用 **vLLM**。
- 每轮训练结束后，把新权重从训练 worker 同步到推理 worker。**这一步在现在依然是痛点**：
  - 实验性的 NCCL collective API 可以直接把权重灌进 vLLM，但还不够成熟（未文档化参数多）。
  - **工程临时解法**：vLLM 以 dummy weights 启动，然后通过各种 hack 把真权重塞进内存；每一迭代甚至要 kill 掉 vLLM 再拉起来，才能彻底释放 GPU 显存。

→ **LLM RL 的基础设施还年轻，一两年内应会成熟很多**。

### 8.7 多种奖励如何组合？

**经验答案**：**加权求和**。权重是**靠在下游任务上调出来的**，没什么魔法。

- 像**格式奖励**这种，可以看作**代理奖励 / shaping reward**：你本身不真的在乎格式，只是借它引导模型产出结构良好的 CoT，从而更容易拿到**真正的**正确性奖励。

### 8.8 Kimi 的规模化曲线

- 随着 RL 迭代，性能**单调上升**且**长度曲线趋于稳定**，不像 vanilla GRPO 那样一路增长——这再次印证 Dr.GRPO 的观点：**长度无限增长是目标函数偏置造成的**。

---

## 九、案例三：Qwen 3

> 对应 PDF Page 61 – 66

### 9.1 为什么讲 Qwen 3

- **最新的开源推理模型**（超过 o1 与 R1）。
- 有两个值得注意的新点：
  1. **低样本 RL**：仅 **3,995** 条数据就做出了好 RL 效果。
  2. **Thinking Mode Fusion**：让同一个模型既能"开思考模式"也能"不思考直接答"。

### 9.2 整体流水线

```
Base
  │
  │ 长 CoT SFT（冷启动）
  │
  │ 推理 RL（GRPO，仅 3995 例）
  │
  │ Thinking Mode Fusion（SFT，把 think/no-think 两种模式融合进同一参数）
  │
  └─ 通用 RLHF
```

熟悉的配方：大致就是 R1 的路子——**推理 RL 先做，RLHF 后做，蒸馏放最后**。

### 9.3 数据筛选（Kimi 风格进化版）

- **Best-of-N 难度筛选**：模型不借助 CoT 就能答对 → 踢掉（太简单）。
- **去污染**：和验证集过于相似的题 → 踢掉。
- **人工过 CoT 质量**：特别是判断"模型是真懂还是蒙对的"。

### 9.4 低样本 RL 的启示

仅 **3,995 例**就跑出显著增益。这呼应了以前的发现：

- 指令微调用几千条就能对齐。
- 长 CoT SFT 1k 条就显著涨点。
- RL 似乎也能"以小搏大"。

**为什么可行**：很可能是因为 **base model 已经具备大部分推理能力**，RL/SFT 只是在**激活并聚焦**这些能力，而不是从零教学。

### 9.5 Thinking Mode Fusion：单模型双模式

**动机**：推理模型开思考链推理能力强但贵。**能不能一个模型同时支持两种模式，让用户/系统自己决定要不要思考**？

**做法**：在完成推理 RL 之后，再做一轮 SFT，数据里混两种格式：

- **Think 模式**：
  ```
  <think> [长 CoT] </think>
  [最终回答]
  ```
- **No-think 模式**：
  ```
  <think></think>
  [直接回答]
  ```

训完之后得到一个**可以通过标签切换模式**的单一模型。

**更酷的副作用：思考预算（thinking budget）控制**
你可以在推理到一半时强行注入一段提示：

```
Considering the limited time of the user, I have to give a solution by thinking directly now.
</think>
```

模型就会立刻结束思考并给答案。这个机制让你拿到一根**连续的"思考长度 → 性能"**曲线：思考预算越大，性能越高；预算为 0，性能略降但不崩。**从单个模型里拿出了平滑的 test-time scaling**。

### 9.6 各阶段消融：RLHF 会损害数学能力

Qwen 3 给出了 reasoning RL / thinking mode fusion / general RLHF 各阶段独立消融。一个有意思的现象：

| 能力 | reasoning RL | thinking mode fusion | general RLHF |
|---|---|---|---|
| 通用任务、指令遵循 | ⬆️ | ⬆️ | ⬆️ |
| Thinking 模式下的数学/STEM | ⬆️ | ⬆️ | **⬇️ 略降** |
| No-thinking 模式下的数学/STEM | ⬆️ | ⬆️ | ⬆️ |

→ **通用 RLHF 会稍稍伤害"开思考模式下的数学能力"**。这是一种明显的**能力权衡**：你不能同时拿到最强通用对齐和最强数学推理。未来的模型该如何绕过这个 trade-off，是开放问题。

---

## 十、全课总结

> 对应 PDF Page 67

一句话版：**RLHF 有天花板，RLVR 是破局之道；GRPO 又简单又好用（但有小坑），各家配方大同小异**。

可带走的要点：

1. **过优化 / reward hacking 是 RLHF 的天然毛病**。解决办法之一：**缩小到客观可验证的领域做 RL**（数学、代码、任务完成度）。这就是 RLVR。
2. **PPO** 理论清晰但工程笨重（价值网络、GAE、KL shaping、clip...），因此社区在找替代。
3. **GRPO**：把 PPO 的复杂优势估计换成**组内 z-score**。只需要"策略梯度 + 好 baseline + 某种形式的正则"就能跑，实现极简。
4. **GRPO 的两个小毛病**（Dr.GRPO）：
   - 除以 std 会**系统性放大过易/过难题**。
   - 除以长度会**激励错答拉长瞎说**。
   - 去掉它们后长度稳、性能还涨。
5. **R1 的启示**：
   - outcome-based 奖励 + GRPO 就能复刻 o1 级别推理能力。
   - PRM、MCTS 在目前阶段没跑通。
   - base model 已经会推理，SFT 只是唤醒。
6. **Kimi K1.5 的启示**：
   - 另一种推导（DPO 风格 + 平方损失）也能等价达到 GRPO。
   - **长度奖励**的巧妙设计：让正确答案变短，错误答案不乱长。
   - **系统工程**是 RL-for-LLM 的关键瓶颈。
7. **Qwen 3 的启示**：
   - 几千条数据就能做好 RL。
   - Thinking Mode Fusion 实现单模型"可选思考 + 可控预算"。
   - 通用 RLHF 会轻微损害思考模式下的数学能力。

---

## 附录：几个你可能会弄混的小概念

**Q1：PPO 里的"价值函数 $V_\phi$"到底在干嘛？**
策略梯度里真正推动更新的是"这个动作比平均水平好多少"（优势 $A$），而不是"这个动作得了多少分"（奖励 $R$）。**价值函数就是用来估"平均水平"的**——把它从奖励里减掉，方差显著降低。但它本身是和策略一样大的网络，贵。GRPO 的突破就是：**用同一个 prompt 下组内平均直接当 baseline，连价值网络都不要了**。

**Q2：GRPO 的 KL 项是按 token 算的，这正常吗？**
对。GRPO（和 PPO）里的 KL 正则是**每个 token 都加一份**的，属于 reward shaping 的一部分；真正的"任务奖励"（答对没）只在最后一个 token 给。这样可以让策略沿着整个序列都受到约束，不至于突然变成 $\pi_{\text{ref}}$ 上概率极低的行为。

**Q3：纯 on-policy 还是 off-policy？**
课堂里反复强调：**rollout 很贵**。所以大家都想"一次采样、多步更新"。一旦多步更新，新策略就偏离了采样策略 $\pi_{\theta_{\text{old}}}$，就必须**重要性采样校正**或**clip**。PPO 和 GRPO 都是用 clip 来控制这种偏离。纯 online（每次只一步）的 GRPO 最简单，clip 根本不起作用，退化成朴素策略梯度。

**Q4："contextual bandit" 和 "full RL" 的区别？**
- **Contextual bandit**：给上下文 → 出一个动作（可视为整段回复）→ 立刻得到奖励。**没有状态转移**。
- **Full RL**：有连续的状态、动作、状态转移，奖励是延迟的。
- **LLM RL 本质上是 contextual bandit**：prompt 进来，一口气生成回复，最后一个 token 给奖励。所以 PPO 里设 $\gamma=\lambda=1$ 就已经够用。这也解释了为什么我们不需要特别花哨的时序差分工具。

**Q5：为什么 KL 项有时候会 clamp 到 0？**
因为 `log(pi_theta / pi_ref)` 在采样时取自有限样本，**样本估计的 KL 可能变成负数**（真 KL 是非负的，但有限样本估计不保证）。直接当 0 处理是一种工程折衷——这就不再是严格的 KL，但大致方向对，训练稳定得多。类似地，GRPO 用的那个 $k - \log k - 1$ 估计，天然非负，是更优雅的选择。

---

**写在最后**：RL 在 LLM 上的实践"结论高度依赖设定"。不同 base model、不同数据、不同训练顺序，都可能得到相反的结论。读论文时请保留一份怀疑——**单篇实验不代表普适规律**。动手时请多看代码——**理论里看不到的 37 个实现细节，才是模型真正的样子**。

