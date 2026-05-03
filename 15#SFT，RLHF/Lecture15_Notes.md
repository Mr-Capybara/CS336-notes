# CS336 Lecture 15：SFT 与 RLHF 助学笔记

> 本笔记配套讲义：`2025 Lecture 15 - RLHF Alignment.pdf`
> 目标：对照讲义阅读本笔记，即可在不看视频的情况下彻底掌握本节课内容。

---

## 〇、开篇：这节课要解决什么问题？

在之前的课程里，我们重点讲了**预训练**（pre-training）——如何把一个 Transformer 在海量互联网文本上训练成 GPT-3 那样的"大语言模型"。

但 GPT-3 并不是一个"好用"的产品。它：

- 不会老老实实跟着你的指令走；
- 不会像 ChatGPT 那样礼貌、详尽地回答你；
- 也不会拒绝恶意请求。

从 **GPT-3 → ChatGPT** 这一步，靠的就是**后训练（post-training）**，具体包含两大技术：

1. **SFT（Supervised Fine-Tuning，监督微调）**：用"专家示范"教模型怎么说话。
2. **RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）**：用"人的偏好"把模型进一步对齐。

这节课的地图（来自 InstructGPT 原论文的三步流程图）：

```
[预训练大模型 GPT-3]
      │
      ▼
Step 1: SFT（用示范数据监督微调）         ←— 本节课 Part 1
      │
      ▼
Step 2: 训练奖励模型（Reward Model）       ←— 本节课 Part 2
      │
      ▼
Step 3: 用 PPO/DPO 优化策略                ←— 本节课 Part 2
      │
      ▼
  [ChatGPT / InstructGPT]
```

本节课的核心问题可以拆成三个：

1. 我们想要的"好行为"数据长什么样？
2. 如何用好这些数据（算法层面）？
3. 这个过程怎么 scale up？

---

## 一、SFT：监督微调

### 1.1 为什么需要 SFT？

预训练模型的参数里**已经装载了大量能力**（推理、问答、代码等），但它默认不会把这些能力"显露"出来。SFT 的思路非常直观：

> 我们搜集**人类专家**（或 GPT-4 等强模型）对各种指令的**示范回答**，然后让模型去**模仿（imitate）**这些回答。

形式上，就是给定 `(instruction, response)` 这样一对数据，直接用标准的语言建模损失（next-token prediction）做微调。这就是为什么它叫"监督"微调——它本质上就是在有标签的数据上做梯度下降。

### 1.2 SFT 数据长什么样？三种典型范式

讲师挑了三个具有代表性的数据集，正好对应三条不同的构建思路。

#### 范式 A：FLAN — 聚合已有 NLP 任务

- **思路**：把 NLP 社区积累了多年的大量任务数据集（问答、分类、摘要、翻译……）统一改写成"指令-回答"的格式，然后全部拼起来，形成一个庞大的 meta 数据集。
- **代表任务**：Natural Instructions v2、T0 SF、Adversarial QA、话题分类……
- **例子**：
  - "Write a subject line for this email: ..." → "Ronald Chisholm LOI"（本质是 Enron 邮件数据集改写）
  - "What is this text about? OPTIONS: - World - Sports - Business - Science/Tech" → "Business"（本质是 AG News 话题分类）
- **优点**：规模庞大、几乎免费（现成的数据集）。
- **缺点**：**不自然**——回答常常只有一个词或一个短语，和真实用户在 ChatGPT 里的交互感觉完全不同；能很明显看到"把原 NLP 数据集粗暴改装成指令形式"的痕迹。

#### 范式 B：OpenAssistant — 人工众包高质量数据

- **思路**：ChatGPT 发布后，一群网络志愿者自发组织起来，**亲手写**指令和高质量回答。
- **例子**：一个关于 monopsony（买方垄断）经济学概念的问题，回答长达一整段，末尾还**附带学术引用**：
  > "References: Bivens, J., & Mishel, L. (2013)..."
- **优点**：自然、详细、覆盖复杂问题。
- **缺点**：人写长回答**极其费力**、费钱；且高知识密度可能反而有害（后面会讲）。

#### 范式 C：Alpaca — 用 LLM 合成数据

- **思路**：Stanford Alpaca 是最早用 LLM 生成 SFT 数据的尝试之一：
  1. 准备一小批人工种子指令；
  2. 让 LLM（如 text-davinci-003）生成更多**新指令**；
  3. 再让 LLM **生成对应的回答**。
- **例子**：
  - "Give three tips for staying healthy." → "1. Eat a balanced diet... 2. Exercise regularly... 3. Get enough sleep..."
- **优点**：便宜、规模可无限扩展、风格像 ChatGPT。
- **缺点**：指令本身不够多样、可能继承教师模型的偏见和错误。

### 1.3 课堂互动实验 1：让你亲手写一条指令数据

讲师现场让全班对"What is CS336? Should I take it?"写一条 SFT 回答。观察到的现象：

- 大多数人写得很**短**、写得**敷衍**（因为 5 分钟时间紧）；
- 少数特别长、特别精致、带 emoji 的——基本都是偷偷用了 ChatGPT；
- 甚至有人直接乱答（所谓的 troll / NSFW），必须过滤。

**教学意义**：即便是"受过良好训练的 Stanford 学生"，在时间压力下也很难产出高质量长回答。这就是为什么真实的人工标注又贵又难——同时也解释了**为什么"让 LLM 生成 SFT 数据"这条路会火起来**。

### 1.4 数据的关键维度：长度、风格、知识密度

#### 长度的"房间里的大象"

- 不同数据集的 **prompt 长度** 和 **response 长度** 差异巨大（Wang+ 2023 的综述）；
- **严重 bias**：无论是人类评审员，还是用 GPT-4 当评审员，都**强烈偏好更长的回答**（大约 60%–70% 的偏好率投给更长的那个）——而且也都偏好列表式（bullet points）的风格。
- **结果**：如果你只看"chatbot 擂台"式的偏好评测，很容易被长度 bias 误导，以为模型变强了，其实只是变长了。
- **对策**：评测时必须同时看**基准测试（benchmark）**——因为 MMLU 这类基准对长度、风格不敏感。不同评估策略要交叉使用，避免单一评测被 gaming。

#### 风格 vs 基准性能

一个有趣的实证现象：即便不同 SFT 数据集训出的模型在**长度、风格上差异巨大**，它们在 **MMLU** 等基准上的得分**其实差不多**。说明：

> SFT 主要改变的是**输出的风格/形式（type signature）**，而不是底层的知识储备。

### 1.5 一个关键"反直觉"陷阱：高质量 ≠ 好 SFT 数据

先看 OpenAssistant 里的经济学问答例子：用户问 monopsony，回答里写了一整段知识并**附上参考文献**。

你可能觉得："这质量真高，用它训练肯定好！"——但 **John Schulman（OpenAI）** 在 Berkeley 的一个讲座里指出了一个问题：

> 模型其实同时学到了两件事：
>
> 1. **具体知识**：monopsony 对应 Bivens & Mishel (2013) 这条引用 —— ✅ 好
> 2. **泛化规律**：只要问题看起来复杂，我就应该在回答末尾**编一条参考文献** —— ❌ 这就是在**教模型幻觉（hallucinate）**！

**根本原因**（从 token 级 loss 的角度看）：

- 模型的预训练中可能**根本没见过 Bivens & Mishel 这本书**；
- 但 SFT 的 next-token loss 要求它必须预测出这些 token；
- 对模型来说，相比"完全不输出引用"，"瞎编一条引用"的 loss 反而更低——因为至少结构（"Reference: 某作者, 某年份..."）是对的；
- 于是模型**学会了一个捷径：补齐结构、内容瞎编**。

**推论（Schulman 的主张）**：

1. 不要用模型**不会的知识**去微调它，否则会促进幻觉；
2. 如果模型不知道某个事实，SFT 数据里最好写 **"我不知道"**，而不是强塞一个答案；
3. 这正是**强化学习（RL）风格**训练很重要的原因——RL 是 **on-policy** 的，它能知道"模型现在到底会什么"，只奖励模型真正能做到的行为。

> **课堂 Q&A 补充**：有同学问"会不会是教模型在应该加引用的地方**调用工具**去检索？"——这个想法好，但需要在预训练阶段就支持工具使用，涉及 RL 层面的改造；单纯在 SFT 阶段无法优雅解决。

另一个相关的实证结果（Gekhman 2023）：模型学"预训练已见过的事实"很快，但学**未见过的事实**时收敛极慢——再次印证"SFT 主要是在激活已有知识，而非注入新知识"。

### 1.6 SFT 与安全性（Safety Tuning）

部署到终端用户的 LLM，必须能：

- **拒绝恶意请求**（诈骗、生成仇恨言论、造谣……）
- 但**不要过度拒绝**（"How do I **kill** a Python process?" 这种合理问题也得好好回答）

核心 tradeoff：

$$
\text{安全} \longleftrightarrow \text{不过度拒绝（over-refusal）}
$$

有趣的实证（讲义 P25）：

- **仅仅 ~500 条安全示例**混进 SFT 数据，就能显著提升模型遵守安全准则的比例；
- 这和 SFT 整体"少量高质量数据也能改变行为"的规律一致。

### 1.7 SFT 部分的三点总结

1. **SFT 非常强大**：用一个体面的开源 SFT 数据集（Open Hermes、OpenAssistant 等）加合理超参数训练 base model，就能得到接近 Llama/ChatGPT 行为的模型。
2. **"高质量数据"的定义很微妙**：不是越长越详细越好——要考虑模型能否消化。
3. **少量数据杠杆巨大**：SFT 阶段对模型行为有极强的改造力。

### 1.8 怎么做 SFT？—— 一句话：梯度下降

学术界：就是在 `(instruction, response)` pair 上做标准的 next-token-prediction 梯度下降。

但**工业前沿实验室**玩法更深：

#### 进阶：Midtraining（中训练）/ 两阶段训练

- **动机**：既然 SFT 数据也是一串 token，为什么非要和预训练分家？
- **做法**（miniCPM、jetMoE 等公开披露，很多闭源大厂也在用）：
  1. **Stage 1 — 纯预训练**：大杂烩（Common Crawl、code、pile、dolma 等）+ WSD 学习率调度中的 **stable** 阶段；
  2. **Stage 2 — decay 阶段（"mid-training"）**：学习率开始衰减，此时混入**高质量**与**指令风格**数据（Wikipedia、Code SFT、中文书、UltraChat、StackExchange、EvolInstruct、OSS-Instruct……）；
  3. **Stage 3 — 正式 SFT**：一轮短时间、小规模的指令微调收尾。

- **为什么有效**：
  - 避免了"先大规模预训练、再硬拉到 SFT 领域"带来的**灾难性遗忘**；
  - 把指令数据更深地融入模型的表征，杠杆更大；
  - 借助学习率衰减期的 loss 大幅下降，**把模型 anneal（退火）到目标模式**。

- **副作用**：今天的 "base model" 标签越来越**模糊**——Qwen、DeepSeek 等模型号称 base，实际上可能已经在 mid-training 阶段见过大量指令数据。

> **课堂 Q&A**：
> - Q：这个办法能解决引用幻觉吗？ A：**不能**。它没有"检测模型是否已知某个事实"的机制，只是让 SFT 数据和预训练融合得更顺滑。
> - Q：SFT 到底能不能教模型新知识？ A：经典小规模 SFT 很难；但当 SFT 被 scale 到 mid-training 级别时，确实能教新知识——mid-training 正在模糊"预训练 vs SFT"的边界。

---

## 二、RLHF：从模仿到优化

### 2.1 视角切换：生成式建模 vs 策略优化

这是全课最重要的概念跃迁，必须理解透：

| 视角 | SFT（模仿） | RLHF（优化） |
|---|---|---|
| 目标 | 拟合 $\hat{p}(y\|x) \approx p^*(y\|x)$ | 找一个策略 $\hat{p}(y\|x)$ 使 $\max_p \mathbb{E}_p[R(y,x)]$ |
| 本质 | **纯生成式建模** | **奖励最大化**（LM 被视为策略） |
| 需要什么数据 | 参考分布的**完整样本**（完整回答） | 对样本打**奖励分**（标量反馈即可） |

换句话说：**SFT 关心"模型分布像不像专家分布"；RLHF 根本不关心分布，它只关心"拿到的奖励够不够高"**。在 RLHF 视角下，LM 不再是"世界的概率模型"，而是一个**决策策略（policy）**。

### 2.2 为什么要做 RLHF？两大动机

#### 动机 1：成本（Cost）

SFT 要求**完整的专家示范**；而 RLHF 只要**标量反馈/偏好**（A 比 B 好？）。

讲义 P32 给出了一个小型 7B 模型的成本图：

- Base model 训练：~$300K
- **SFT 数据**：~$25K（最贵的人工部分）
- **Pairwise feedback 数据**：~$4K（显著便宜）
- RL 训练：~$100
- Evaluation：~$100

> 前沿实验室在 post-training 数据上花的钱**动辄百万美元**——所以找更便宜的数据采集方式非常重要。

#### 动机 2：Generator-Validator Gap（生成-判别差距）

一个有意思的发现（Zhang et al. 2023，新闻摘要基准）：

> 让一个专业撰稿人**写**一篇摘要，然后再让他/她**比较**自己写的 vs LM 写的——**他们有时会更喜欢 LM 写的版本**。

当被采访时，这位撰稿人说："我自己写的时候觉得必须用华丽的辞藻，但看 AI 写的，读起来就是更顺。"

**启示**：

- **判断（validate）比生成（generate）容易**，而且有时**判断还更准确**；
- 这意味着用"偏好比较"收集数据，不仅更便宜，质量可能**反而更高**。

### 2.3 RLHF 的整体流程

```
1. 用 SFT 模型对同一个 prompt 采样多个候选回答 y1, y2, ...
2. 人（或 LM）做成对比较：yA 比 yB 好？
3. 用这些成对偏好训练一个奖励模型 R(y, x) → 标量
4. 用强化学习（PPO / DPO 等）让 SFT 模型的输出 R 值尽量高
```

### 2.4 RLHF 的数据：怎么收集成对反馈？

#### 标注员界面

一般就是个 Web App：同时显示两条 AI 回答，让标注员点选哪个更好（通常是 4 档："A 明显更好 / A 稍好 / B 稍好 / B 明显更好"）。

#### InstructGPT 标注指南（罕见公开）

要求标注员评估三大维度：

- **Helpful（有用）**：用清晰语言、真正回答用户想问的、对国际化语境敏感（比如 "football" 不默认是美式橄榄球）、看不懂就要求澄清；
- **Truthful（真实）**：不要幻觉；
- **Harmless（无害）**：不毒、不脏、不 NSFW。

规模：**40 人**（通过 Scale + Upwork 招募）——以今天的标准看小得惊人，但 ChatGPT 就是从这里起步的。

#### Google Bard 标注指南（据传泄漏版）

类似结构：helpfulness / style / 评分量表；每题**给 1 分钟**完成（引发争议，标注员投诉时间不够）。

### 2.5 课堂互动实验 2：让你亲手做成对比较

题目包括一些**事实错误被刻意塞进长回答**、还有**数学推导走向错误结论**的例子。课堂结果：

- 绝大多数人**没有核查事实**；
- 几乎没人**核查数学推导**；
- 大家**更偏好更长的回答**——哪怕它有明显幻觉；
- 这和新闻里 Google Bard 标注员"1 分钟根本不够"的投诉完全一致。

**教训**：

1. 高质量、能真正核查的标注员难找；
2. 在时间压力下，人基本不会核查正确性；
3. 长度 bias 顽固；
4. **要警惕标注员偷偷用 GPT-4 作弊**——曾有研究发现标注员和 GPT-4 的一致性高达 95%+，高得可疑。

#### 标注员的偏见与伦理问题

- **demographic bias**：Santurkar+ 2023 发现 InstructGPT 在主观问题上明显偏向东南亚宗教观点——查 InstructGPT 附录发现标注员主要是菲律宾和孟加拉籍，只有 17% 美国人。
- **标注员类型影响**：Hosking et al. 2024 发现众包标注员**几乎不关心事实准确性**，更关心**格式**；而专家作者更关心有没有引用、事实对不对。
- **伦理**：大规模外包到低收入国家涉及工资、心理健康（审核暴力内容）等问题，需要警惕。

### 2.6 AI Feedback 的兴起

由于人工标注又贵、又慢、又 bias、又容易作弊，业界越来越依赖 **LLM-as-a-judge**：

- **GPT-4 和人类的评分一致性**和**人类内部一致性**几乎相当；
- **成本**却低得多；
- 典型数据集：**UltraFeedback**、**Tulu 3** 等都大量采用 AI feedback；
- HuggingFace 的 Zephyr 7B 团队最初也坚信人工标注更好，但最后发现 GPT-4 反馈在实操中**更优**。

**重要警告**：LLM 评分有**严重的自我偏好（self-preference bias）**——一个模型倾向于给自己的输出打高分。用同一模型既生成又评价时要特别小心。

#### 追溯：Constitutional AI（Anthropic）

"AI 反馈"路线的里程碑——Anthropic 的 **Constitutional AI**：提前写一份"宪法"（原则集），让 AI 自己按宪法去批评和修改自己的输出，不再依赖人类标注。

### 2.7 长度效应：RLHF 的常客

- RLHF 之后的模型几乎**总是变长**（Chen et al. 2024；Singhal et al. 2024）；
- "人偏好长回答 → 奖励模型偏好长回答 → RL 推着模型写更长"——形成正反馈。
- 需要用**长度归一化**或惩罚项来缓解。

### 2.8 课堂 Q&A：on-policy vs off-policy 数据

- **Off-policy 数据**：成对反馈**不是**从你正在训练的模型采样的（比如用 GPT-4、Claude 生成候选答案）——告诉你"世界的其他部分长什么样"。
- **On-policy 数据**：从你自己当前模型采样 → 评分 → 训练——告诉你"你自己怎么改进自己"。
- Tulu 3 同时用了两种：off-policy 提供多样性，on-policy 提供精细调整。

---

## 三、RLHF 的算法：PPO 与 DPO

这一节进入**数学密集区**。

### 3.1 RLHF 的优化目标（来自 InstructGPT, Eq. 2）

$$
\mathcal{L}_{\text{RLHF}}(\theta) = \mathbb{E}_{(x,y)\sim \pi_\theta}\big[R_\phi(x,y)\big] \;-\; \beta\, \mathbb{E}_{x}\Big[\operatorname{KL}\big(\pi_\theta(\cdot|x)\;\|\;\pi_{\text{SFT}}(\cdot|x)\big)\Big] \;+\; \gamma\, \mathbb{E}_{x\sim D_{\text{pretrain}}}\big[\log \pi_\theta(x)\big]
$$

三项含义：

1. **第一项（奖励项）**：我们希望策略 $\pi_\theta$ 产生的回答在奖励模型 $R_\phi$ 下得分尽量高。
2. **第二项（KL 项）**：约束 $\pi_\theta$ 不要偏离 SFT 初始策略 $\pi_{\text{SFT}}$ 太远——如果没有这项，RL 会很快让模型**模式崩溃**（mode collapse，输出千篇一律的高分短语）。$\beta$ 越大，约束越强。
3. **第三项（预训练正则项）**：持续在预训练数据上做语言建模，防止灾难性遗忘。$\gamma$ 一般较小；**很多团队直接省略这一项**。

> 第二项是 RLHF 的灵魂。即便算法从 PPO 换成 DPO 换成别的，几乎**都保留 KL 约束**。

### 3.2 奖励模型：Bradley–Terry 概率模型

关键假设：

> 世界上每一条可能的回答 $y$，都有一个**隐藏**的标量奖励 $r^*(x,y)$。
> 当一个人对 A、B 做成对比较时，他选 A 的概率服从：

$$
P(y_A \succ y_B \mid x) \;=\; \sigma\big(r^*(x,y_A) - r^*(x,y_B)\big) \;=\; \frac{1}{1+\exp\big(-(r^*(x,y_A)-r^*(x,y_B))\big)}
$$

即 **Bradley–Terry 模型**：奖励差 → 逻辑斯蒂概率 → 抛硬币得到人类偏好。

训练奖励模型 $r_\phi$ 时的损失（Stiennon et al. 2020）：

$$
\mathcal{L}_{\text{RM}}(\phi) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)\sim D}\Big[\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]
$$

其中 $y_w$ 是被选中的（winning）回答，$y_l$ 是落选的（losing）回答。本质上是对**偏好对**做最大似然估计。

### 3.3 PPO（Proximal Policy Optimization）

本节课只做"高层介绍"，下节课会详细讲。三步演进：

#### Step 1：原始的 policy gradient（方差太高）

$$
\nabla_\theta \mathbb{E}_{z\sim \pi_\theta}\big[R(z)\big] \;=\; \mathbb{E}_{z\sim \pi_\theta}\big[R(z)\,\nabla_\theta \log \pi_\theta(z)\big]
$$

直觉：如果一条轨迹 $z$ 奖励 $R(z)$ 高，就**上调**它的 log-probability；低则下调。这就是 REINFORCE 算法。

**问题**：$R(z)$ 方差巨大，梯度估计噪声极大，训练极不稳定。

#### Step 2：引入 advantage + TRPO

**Advantage（优势函数）**：从 $R$ 里减掉一个 baseline $b(x)$（可以是 value function 的估计）：

$$
A(x, z) \;=\; R(x, z) - b(x)
$$

数学上，由于 $\mathbb{E}[\nabla_\theta \log \pi_\theta] = 0$，减去任何只依赖状态的量**不会改变梯度期望**，但能**大幅降低方差**。

**TRPO（Trust Region Policy Optimization）**：允许从一次采样里**多走几步**梯度——但为了补偿"采样分布和当前策略不同"，要加**重要性采样权重**：

$$
\rho(\theta) \;=\; \frac{\pi_\theta(z)}{\pi_{\theta_{\text{old}}}(z)}
$$

并强制 KL 约束把 $\pi_\theta$ 限制在 $\pi_{\theta_{\text{old}}}$ 的"信赖域"内。

#### Step 3：PPO（把 KL 约束改成 clipping）

TRPO 的硬约束实现起来复杂。PPO 用一个**简单技巧**替代：直接把重要性权重 $\rho$ **裁剪**在 $[1-\epsilon, 1+\epsilon]$ 区间内：

$$
\mathcal{L}_{\text{PPO}}(\theta) \;=\; \mathbb{E}\Big[\min\big(\rho(\theta)\,A,\; \operatorname{clip}(\rho(\theta), 1-\epsilon, 1+\epsilon)\,A\big)\Big]
$$

这样既享受了"多步更新"的样本效率，又保证策略不会跑太远。

**但 PPO 很烦人**——实现细节多（value head、GAE、advantage 标准化、clipping、KL 惩罚……），工程陷阱密布。这就是学术界一直在问"能不能绕开 PPO?"的原因。

### 3.4 绕过 PPO 的失败尝试

研究者们试过几种更简单的方案（讲义 P54），但效果都不如 RLHF：

1. **Control token**：SFT 时在 "chosen" 前加 `[GOOD]`、"rejected" 前加 `[BAD]`；推理时强制 condition 在 `[GOOD]` 上。✗ 效果差。
2. **只在 chosen 上做 SFT**：✗ 效果一般。
3. **用奖励模型筛选 + SFT**：训一个 RM，把模型采样的输出里分最高的拿来做 SFT。尚可但不够。
4. **Best-of-N**：推理时采 1024 条，用 RM 挑最好的。✗ 推理成本爆炸。

真正取得突破的是 **DPO**。

### 3.5 DPO（Direct Preference Optimization）：优雅的替代

**核心思想**：跳过"训奖励模型 + 再用 RL 优化"这两步，**把整个 RLHF 问题转化为一个监督学习问题**。

#### DPO 推导（本节最关键的数学）

**第一步**：写出原始 RLHF 目标（简化版，忽略预训练正则项）：

$$
\max_{\pi_\theta}\; \mathbb{E}_{x\sim D,\, y\sim \pi_\theta(\cdot|x)}\big[r(x,y)\big] \;-\; \beta\,\operatorname{KL}\big(\pi_\theta(\cdot|x)\;\|\;\pi_{\text{ref}}(\cdot|x)\big)
$$

**第二步**：做一个**非参数化（nonparametric）假设** —— 先假装 $\pi_\theta$ 可以是**任意函数**（不受神经网络结构限制）。这时候目标可以**解析求解**。

展开 KL，目标等价于：

$$
\max_{\pi} \sum_y \pi(y|x)\Big[r(x,y) - \beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\Big]
$$

对 $\pi(y|x)$ 求极值（配合归一化约束 $\sum_y \pi(y|x)=1$），用拉格朗日乘子法或变分法可得**最优策略**的闭式解：

$$
\pi^*(y|x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\Big(\frac{1}{\beta}r(x,y)\Big)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$ 是配分函数。

**第三步**：反解出 $r(x,y)$（**这是 DPO 的精华一步**）。

对上式两边取对数：

$$
\log \pi^*(y|x) \;=\; \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta}r(x,y) - \log Z(x)
$$

移项得到**隐式奖励（implied reward）**：

$$
r(x,y) \;=\; \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

**关键观察**：在非参数假设下，**任何策略 $\pi$ 都可以"反推出"一个对应的奖励函数 $r$**，二者一一对应。所以"优化策略"等价于"优化它对应的奖励"。

**第四步**：把这个 $r$ 代回 Bradley–Terry 偏好模型。注意 $\log Z(x)$ **在差 $r(x,y_w) - r(x,y_l)$ 里会抵消**，所以：

$$
P(y_w \succ y_l \mid x) \;=\; \sigma\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

**第五步**：把这个概率作为模型对偏好对的似然，最大化它——等价于最小化负对数似然：

$$
\mathcal{L}_{\text{DPO}}(\theta) \;=\; -\,\mathbb{E}_{(x,y_w,y_l)\sim D}\!\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

**这就是 DPO 的损失函数。** 它可以直接用标准的反向传播优化，不需要任何 RL！

#### 三步走总结

> 1. 做非参数假设 → 策略和奖励一一对应；
> 2. 用策略 **参数化** 奖励（把 $r$ 替换成 $\pi_\theta$ 的比值表达）；
> 3. 用**监督式损失**（偏好对的 MLE）优化策略——顺带就优化了奖励。
>
> 概念上：**DPO = 在非参数假设下，对成对奖励做 MLE**。

#### DPO 梯度的直觉

对 DPO loss 求梯度，可以写成：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} \;\propto\; -\,\underbrace{\sigma\!\big(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w)\big)}_{\text{隐式奖励的预测误差}}\; \cdot\; \big[\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\big]
$$

含义非常直观：

- **上调** chosen $y_w$ 的对数概率；
- **下调** rejected $y_l$ 的对数概率；
- 两者权重由"当前模型对这对偏好的判断错得有多离谱"决定（错得越离谱，梯度越大）。

一句话：**正样本做梯度上升，负样本做梯度下降，按预测误差加权**。

### 3.6 DPO 为什么流行？

- **实现极简**：一个 loss 函数，不需要 reward model、不需要 PPO 的 value head、不需要 rollout 循环；
- **效果不差**：在 AlpacaFarm 等基准上和 PPO 打平手；
- **开源主流**：大多数 "top open-source RLHF models"（Chris Manning 语）用的都是 DPO。

### 3.7 DPO 的变种

DPO 变种非常多，讲义提了 Tulu 3 关注的两个：

- **SimPO**：去掉对参考模型 $\pi_{\text{ref}}$ 的依赖（用**长度归一化**的 log-prob 替代）；
- **Length-normalized DPO**：对 log-prob 按 token 数归一，抑制长度 bias。

### 3.8 DPO vs PPO：谁更好？

- 论文结果常常**相互矛盾**——每篇都说自己赢；
- RL 实证研究**高度依赖超参数、实现细节**；
- 保守结论：两者各有场景，工业界大规模训练仍偏好 PPO（或其 GRPO 等变种）。

---

## 四、RLHF 的坑（必须警惕的副作用）

### 4.1 奖励过优化（Reward Over-optimization / Overfitting）

现象：

- 随着对奖励的优化加深，**真实**偏好（比如金标准人类评分）会**先上升、后下降**——典型的过拟合曲线；
- 模型学会 **game** 奖励模型的漏洞（生成长、带 emoji、多列表……以骗取高分），但实际质量在下降。

Gao et al. 的结论（讲义 P64）：

- 对**人类偏好**和**有噪声的 LM 偏好**，过优化现象都存在；
- 但如果奖励是**完全无噪声**的（ground truth），就不会出现过优化——**验证了问题源于奖励模型的误差**。

**对策**：

- 用 **KL 惩罚** 限制策略远离 $\pi_{\text{ref}}$（这是 KL 项的核心作用之一）；
- 早停；
- 周期性更新奖励模型。

### 4.2 模式崩溃（Mode Collapse） / 熵损失

现象：

- RLHF 后的模型不再是一个良好的"概率模型"——分布变得**尖锐、确定**；
- 同一个 prompt 采样多次，回答几乎**一模一样**；
- 模型**校准（calibration）**崩溃：它对自己回答的"置信度"和实际正确率脱钩。

这是做 **不确定性量化** 或需要**多样性生成**（采样、搜索）时的大麻烦。

---

## 五、本讲小结

### SFT 部分

1. SFT 的本质是**激活**预训练中已有的能力，不是注入新知识；
2. 数据质量 ≠ 数据长度/详细度，**不要用模型不会的知识去训它**；
3. 少量数据（比如 500 条安全示例）就能显著改变模型行为；
4. 现代训练中，**SFT 与预训练的边界越来越模糊**（mid-training 范式）。

### RLHF 部分

1. **RLHF 数据收集同样困难**：偏好有长度 bias、标注员 bias、伦理问题、GPT-4 作弊问题……
2. **PPO 是 OG 算法**，但实现复杂；**DPO 是优雅替代**，通过非参数化假设 + 偏好 MLE，把 RL 问题转化为监督学习；
3. 必须警惕 **奖励过优化** 和 **模式崩溃**；KL 约束是关键护栏；
4. AI Feedback（用 GPT-4 等当评审员）在成本/质量曲线上非常有吸引力，但要注意**自我偏好偏差**。

### 下一讲预告

> Thursday：**RL from Verifiable Rewards**（可验证奖励的 RL）——针对数学、代码这类**有明确正确答案**的任务，直接用正确性作为奖励（而不再需要偏好数据），训练推理能力（reasoning training）。PPO 的详细版本也将在那里展开。

---

## 附录：重要术语快速对照

| 术语 | 英文 | 一句话解释 |
|---|---|---|
| 监督微调 | SFT | 用 `(指令, 示范回答)` 对做 next-token prediction |
| 人类反馈强化学习 | RLHF | 用人类对输出的偏好训练 LM 最大化奖励 |
| 奖励模型 | Reward Model, RM | 把 $(x,y)$ 映射到标量的模型，由偏好对训练 |
| 策略 | Policy $\pi_\theta$ | RL 视角下的 LM，输入 prompt 输出回答 |
| 参考策略 | $\pi_{\text{ref}}$ | 通常是 SFT 后的模型；RL 中用 KL 约束拉住 $\pi_\theta$ |
| KL 散度 | KL divergence | 衡量两个分布距离，RLHF 中作为正则项 |
| Bradley–Terry 模型 | BT model | 把"奖励差"映射到"被偏好概率" |
| 优势函数 | Advantage | 奖励减去 baseline，降低策略梯度方差 |
| PPO | Proximal Policy Optimization | 用 ratio clipping 替代硬 KL 约束的策略梯度算法 |
| DPO | Direct Preference Optimization | 跳过奖励模型，直接对偏好做 MLE 的监督式 RLHF |
| On-policy / Off-policy | — | 数据是否来自当前模型的采样 |
| Mid-training | 两阶段训练 | 在预训练末期混入指令数据 |
| 模式崩溃 | Mode collapse | RLHF 后模型输出过于确定、多样性丧失 |
| 奖励过优化 | Reward over-optimization | RL 钻奖励模型漏洞，真实质量反而下降 |
| G-V Gap | Generator-Validator gap | 判断一个答案比生成一个答案更容易 |

