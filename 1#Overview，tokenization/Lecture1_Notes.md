# CS336 Lecture 1：Overview & Tokenization 笔记

> Stanford CS336 - Language Modeling from Scratch (Spring 2025)
> 授课：Percy Liang & Tatsunori Hashimoto

---

## 一、课程动机：为什么要"从零构建"语言模型？

### 1.1 研究者与底层技术的脱节危机

当前 AI 研究正面临一个核心问题：**研究者正在与底层技术脱节**。

- **8 年前**：研究者自己实现并训练模型
- **6 年前**：至少还会下载 BERT 等模型并做 fine-tune
- **现在**：很多人只需 prompt 一个闭源模型就能发论文

抽象层虽然带来便利，但**这些抽象是"有漏洞"的（leaky abstractions）**。与操作系统/编程语言不同，你并不真正理解这个抽象的内部——它只是一个 string-in / string-out 的黑盒。

> **课程哲学**：To understand it, you have to build it.

### 1.2 前沿模型的"工业化"鸿沟

- GPT-4：传闻 1.8T 参数，训练成本 1 亿美元
- xAI：建设 200,000 张 H100 的集群
- 未来 4 年总投资预计超过 5000 亿美元
- 大厂因竞争/安全原因**不公开技术细节**

=> 个人/学术界无法训练 Frontier Model，但可以训练小模型。

### 1.3 小模型的代表性问题

小规模实验的结论**可能无法外推到大规模**：

1. **Attention vs MLP 的 FLOPs 占比**：小模型中两者相当；175B 时 MLP 占绝对主导。如果你在小规模上优化 attention，可能优化了错误的目标。
2. **涌现能力（Emergent Behavior）**：某些能力（如 in-context learning）只在达到某个计算量阈值后才出现。小规模看不到这些现象。

### 1.4 本课程能教给你什么

| 知识类型                | 能否教授   | 说明                                       |
| ----------------------- | ---------- | ------------------------------------------ |
| **Mechanics（机制）**   | ✅ 可以     | Transformer 实现、模型并行、GPU 利用等     |
| **Mindset（思维方式）** | ✅ 可以     | 把 scaling 当回事、榨干硬件                |
| **Intuitions（直觉）**  | ⚠️ 部分可以 | 数据与架构选择，小规模与大规模结论未必一致 |

一个值得记住的幽默注脚：SwiGLU 论文作者在结论中坦白"We offer no explanation except for this is divine benevolence."——**很多 Transformer 设计选择就是实验出来的，没有理论**。

---

## 二、核心思想：The Bitter Lesson 的正确解读

对 Bitter Lesson 的常见误解："只要堆算力就行，算法不重要"。

**正确解读**：**Algorithms at scale is what matters.**

`Accuracy = Efficiency × Resources`

### 为什么 Efficiency 在大规模下更重要

- 花费几亿美元时，浪费不起
- OpenAI 论文（2020）显示：2012-2019 期间，训练 ImageNet 到某一精度的算法效率提升了 **44 倍**，超过摩尔定律

### 课程核心问题

> **Given a certain compute and data budget, what is the best model you can build?**

无论什么规模，这个"每单位资源的精度"问题都成立。研究者的目标就是**提升算法效率**。

---

## 三、语言模型发展简史

### 早期（Shannon 时代至 2000s）
- Shannon 用语言模型估计英语熵
- 2007 年 Google 就已经训练 5-gram 模型，在 **2 万亿 tokens** 上（比 GPT-3 的数据还多），但 n-gram 不具备今天 LM 的能力

### 2010s：深度学习时代的各个零件逐步就位
- 2003：Bengio 的首个神经语言模型
- seq2seq（Ilya 等）
- Adam 优化器（至今主流）
- Attention 机制（起源于机器翻译）
- 2017：Transformer（Attention is All You Need）
- Mixture of Experts
- 模型并行（已能训练 100B 参数，但训练时间短）

### 基础模型时代
- ELMo、BERT、T5：pre-train + fine-tune 范式兴起

### OpenAI 与 Scaling
- 把所有零件组合 + 严谨工程 + **拥抱 scaling laws**
- 诞生 GPT-2、GPT-3

### 开放程度的分层
1. **Closed**：仅 API（如 GPT-4）
2. **Open-weight**：放出权重 + 架构细节，但数据集保密
3. **Open-source**：权重 + 数据 + 尽可能完整的论文

### 当前格局
OpenAI、Anthropic、xAI、Google、Meta、DeepSeek、Alibaba、Tencent 等主导 Frontier。

---

## 四、课程结构与要求

### 基本信息
- 5 学分课程，作业量极大（第一个作业 ≈ CS224N 全部作业 + 期末项目的工作量）
- 第二次开课，规模扩大 50%
- 所有 Lecture 将发布到 YouTube
- 集群：Together AI 提供的 H100s

### 不适合你的情况
- 本学期想做研究产出
- 只想学最新最热的 trick
- 想真正构建一个 LM 产品（prompting / fine-tuning 更实用）

### 作业通用策略
- 不提供脚手架代码，只给 blank 文件 + 单元测试 + adapter 接口
- **先在本地笔记本实现并验证正确性**，再上集群做 benchmark
- 不要用 1B 参数模型在集群上 debug
- AI 工具（Copilot / Cursor）可用但自担风险——会影响你的学习

---

## 五、课程五大模块总览

**核心原则贯穿始终：Efficiency given hardware and data.**

### 模块 1：Basics（基础）
目标：实现完整 pipeline
- Tokenizer（BPE）
- Transformer 架构
- 训练循环

### 模块 2：Systems（系统）
- Kernels（GPU kernel，用 Triton）
- Parallelism（数据并行、张量并行等）
- Inference（推理）

### 模块 3：Scaling Laws（缩放律）
小规模实验 → 预测大规模超参数

### 模块 4：Data（数据）
- 评估
- 数据采集与清洗

### 模块 5：Alignment（对齐）
- SFT
- RLHF / DPO / GRPO

---

## 六、模块 1：Basics 细节

### 6.1 Tokenizer
- 本课程用 **BPE (Byte Pair Encoding)**
- Tokenizer-free（直接处理 bytes）虽有前景，但尚未在 Frontier 规模被验证

### 6.2 Model Architecture
以原始 Transformer 为起点，但 2017 后有许多微改进（累积起来效果显著）：

| 组件          | 原始                 | 现代选择                                |
| ------------- | -------------------- | --------------------------------------- |
| 激活函数      | ReLU/GELU            | **SwiGLU**                              |
| 位置编码      | Sinusoidal / Learned | **RoPE（旋转位置编码）**                |
| Normalization | LayerNorm            | **RMSNorm**                             |
| Norm 位置     | Post-Norm            | **Pre-Norm**                            |
| MLP           | Dense                | **MoE (Mixture of Experts)**            |
| Attention     | Full                 | Sliding Window / Linear / **GQA / MLA** |
| 替代架构      | ——                   | State-Space Models（如 Hyena）、Hybrid  |

### 6.3 Training
- 优化器：**AdamW** 为主流；新秀有 Muon、Soap
- 细节很关键：学习率调度、batch size、正则化、超参数——调优好坏能相差一个数量级

### 作业 1
- 实现 BPE tokenizer（**据学生反馈意外地耗时**）
- 实现 Transformer、cross-entropy loss、AdamW、训练循环
- 允许使用 PyTorch，但不能用现成的 Transformer 实现
- 数据集：TinyStories、OpenWebText
- Leaderboard：90 分钟 / 1 张 H100，最小化 OpenWebText perplexity

---

## 七、模块 2：Systems 细节

### 7.1 GPU 基础
- GPU = 大量浮点运算单元的阵列
- 片外（HBM）+ 片上（L1/L2 cache）两级存储
- **瓶颈常在数据移动**，而非计算本身

类比：
- Memory = 仓库
- Compute = 工厂
- 运输成本（data movement）才是关键

### 7.2 Kernels
- 使用 **Triton**（OpenAI 出品）
- 关键技术：**Fusion（融合）** 与 **Tiling（分块）**

### 7.3 Parallelism
多 GPU 场景（甚至 8 卡就已有趣）：
- GPU 间通过 NVSwitch / NVLink 连接
- 数据并行（Data Parallel）
- 张量并行（Tensor Parallel）
- FSDP 等

### 7.4 Inference
全球推理成本正在超过训练成本（训练是一次性的，推理随使用量线性增长）。

**两阶段**：
1. **Prefill**：处理 prompt，可并行，**compute-bound**（与训练相似）
2. **Decode**：autoregressive 一个 token 一个 token，**memory-bound**，难以打满 GPU

**加速技巧**：
- 用更便宜的模型
- **Speculative Decoding**：小模型预测多个 token，大模型并行验证接受
- 各种系统级优化

推理用途不止聊天：RL、test-time compute、模型评估都需要。

### 作业 2
- 实现一个 kernel
- 实现数据并行
- 实现"baby 版" FSDP（完整版太复杂）
- **养成 benchmark / profile 的习惯**——否则就是 flying blind

---

## 八、模块 3：Scaling Laws

### 核心问题
给定 FLOPs 预算，**参数量 N 与数据量 D 如何分配最优**？

- 大模型 + 少数据 vs 小模型 + 多数据

### Chinchilla Optimal
对每个 compute 预算，通过变化参数量测精度，找出最优点。把这些最优点连线，**惊人地线性**。

**经验法则**：`D ≈ 20 × N`
- 例：1.4B 参数模型 → 应训练 28B tokens

注意：这个规则**不考虑推理成本**，仅追求最好训练 loss。

### 作业 3
- 提供一个"training API"：你指定超参数与架构，返回 loss
- 你有一个 FLOPs 预算
- 要求：自主选择实验 → 拟合 scaling law → 预测大规模最佳超参
- **一旦 FLOPs 耗尽就结束**——模拟真实研究中的实验优先级决策

---

## 九、模块 4：Data

> "训练在互联网上"是不准确的说法——**数据必须主动获取与加工**。

### 9.1 Evaluation 先行
- Perplexity
- 标准化测试（MMLU 等）
- Instruction following 评估
- Ensemble / Chain-of-Thought 的评估
- 系统级（Agent）评估

### 9.2 Data Curation
- 来源：Common Crawl、书籍、arXiv、论文、GitHub 等
- 法律问题：哪些数据能用
- Frontier 模型越来越多地**购买数据**
- 原始数据不是文本：HTML、PDF、代码目录 → 需要 **HTML-to-text** 等有损转换

### 9.3 实际看一眼 Common Crawl
真相：**大部分是垃圾/spam**，远比你想象的糟。

### 9.4 关键流程
- Filter（同时解决质量与有害内容）——通常训 classifier
- Deduplication（去重）

### 作业 4
- 给你原始 Common Crawl dump
- 训练分类器、去重
- Leaderboard：在 token 预算下最小化 perplexity

---

## 十、模块 5：Alignment

训练完的 base model 只会补全下一个 token，需要**对齐**成有用工具。

### 10.1 Alignment 的三个目标
1. **Instruction following**（指令遵循）
2. **Style control**（长短、bullet、语气等）
3. **Safety**（拒绝有害请求）

### 10.2 SFT（Supervised Fine-Tuning）
- 收集 (user, assistant) 对话对
- 直接做监督学习（本质和 pre-train 类似：最大化文本概率）
- **LIMA 论文**：仅 1000 条高质量例子就能让好的 base 获得指令遵循能力

### 10.3 Learning from Feedback
SFT 数据昂贵（需人工标注完整回答），feedback 形式更轻：

**Preference 数据**：模型生成 A/B 两答案，人标出哪个更好。

**Verifiers**：
- Formal verifiers（数学、代码）
- Learned verifiers（训一个 LM 当评委）

### 10.4 算法
- **PPO**（Proximal Policy Optimization）——最早用于 instruction tuning
- **DPO**（Direct Preference Optimization）——纯偏好数据下更简单有效
- **GRPO**（Group Relative Preference Optimization，DeepSeek 提出）——去掉 value function，更高效，适用于 verifier 数据

### 作业 5
- 实现 SFT / DPO / GRPO
- 评估

---

## 十一、以 Efficiency 为透镜重读所有设计

当前课程环境处于 **compute-constrained**（GPU poor）状态：数据多、算力少。

| 设计决策       | 效率动机                       |
| -------------- | ------------------------------ |
| 激进的数据过滤 | 不浪费算力在垃圾数据上         |
| Tokenization   | byte-level 优雅但 compute 低效 |
| 架构选择       | 大多出于效率考虑               |
| 单 epoch 训练  | 赶时间，宁愿看更多新数据       |
| Scaling laws   | 小算力确定大模型超参           |
| Alignment      | 投入对齐可减少 base 模型规模   |

### 未来趋势
Frontier 实验室正转向 **data-constrained**，设计会变：
- 一个 epoch 不够，需要多 epoch 或更智能的策略
- Transformer 原本是为 compute 效率设计的，可能会被替代

---

## 十二、Tokenization 深入讲解

### 12.1 基本定义
- 输入：Unicode 字符串
- 输出：整数序列
- 要求：可逆（encode ↔ decode）
- **Vocabulary size** = 整数可取值的范围

### 12.2 Tokenizer 在线工具观察
用 GPT-4o tokenizer 观察：
- **空格被包含在 token 中**（与传统 NLP 不同，不丢弃信息）
- 约定俗成：空格放在 token **前面**（`" hello"` 而非 `"hello "`）
- `"hello"` 与 `" hello"` 是**完全不同**的 token → 这会导致一些诡异问题
- 数字被**从左到右**切分，不按千分位或语义

### 12.3 Tokenizer 方案演进

#### 方案 A：Character-based
- 每个 Unicode 字符 → 1 个 code point
- 问题：
  - 稀有字符浪费词表空间（例：emoji 码点 127757）
  - 词表巨大（Unicode 有 100k+ 字符）
  - 压缩比约 1.5 bytes/token

#### 方案 B：Byte-based
- 用 UTF-8 编码成字节序列
- 词表只有 256
- 问题：**压缩比 = 1**，序列过长 → attention 是 O(n²) → 灾难
- 理想很优雅，实际不行

#### 方案 C：Word-based
- 用正则表达式切词（GPT-2 也用作 pre-tokenize 步骤）
- 问题：
  - 词表无界
  - 稀有词 / 新词要用 UNK token，影响 perplexity 计算
  - 粒度不自适应

#### 方案 D：BPE（Byte Pair Encoding）—— 现代选择

### 12.4 BPE 算法

**历史**：
- 1994 年 Philip Gage 为**数据压缩**发明
- 首先被引入 NMT（Neural Machine Translation）
- GPT-2 采用后成为 LM 标配

**核心思想**：
> 不预设切分规则，而是**在语料上训练** tokenizer；常见多字符序列合并为单 token，罕见序列留作多 token。

**流程**：
1. 把文本转为字节序列
2. 反复执行：**统计最频繁的相邻 token 对 → 合并为新 token**
3. 记录 merges 映射

**实现细节**：
- 为了效率，先用正则 pre-tokenize 切成大块，再在每块内跑 BPE

### 12.5 BPE 算法走读

以 `"the cat in the hat"` 为例：

**初始化**：
- 转字节序列：`[116, 104, 101, 32, 99, 97, 116, ...]`
- `merges = {}`, `vocab = {0..255 → bytes}`

**迭代 3 次**：

| 步  | 操作                                          | 新 token |
| --- | --------------------------------------------- | -------- |
| 1   | 统计对子，`(116, 104)` = "th" 出现最多 → 合并 | 256      |
| 2   | `(256, 101)` = "the" → 合并                   | 257      |
| 3   | `(257, 32)` = "the " → 合并                   | 258      |

每次合并后替换序列中的对应对，**序列长度减少**，压缩比提升。

### 12.6 Encode 过程
对新字符串：
1. 先转字节序列
2. **按训练时记录的 merge 顺序依次 replay**
3. 得到整数序列

### 12.7 BPE 的工程改进点（作业方向）
朴素实现的问题：
- encode 遍历所有 merges，效率低 → 应只遍历当前序列相关的 merges
- 需支持 **special tokens**
- 需加入 **pre-tokenization**
- 可以并行化

### 12.8 总结
- Tokenizer = string ↔ int sequence 的映射
- character / byte / word based 都有明显缺陷
- **BPE**（1994 年的老算法）利用语料统计**自适应分配词表**，至今有效
- 理想是未来架构能直接处理 bytes，淘汰 tokenization

---

## 十三、下一讲预告

PyTorch 细节与 **Resource Accounting**：精确追踪 FLOPs 去哪了。虽然大家都写过 PyTorch，但要真正看清每一步算力的流向。

