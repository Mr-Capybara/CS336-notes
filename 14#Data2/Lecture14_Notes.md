# CS336 Lecture 14 助学笔记：数据处理之二（过滤与去重）

> 本笔记对应 Stanford CS336 Spring 2025 的第 14 讲。上一讲纵览了各类训练语料（从 BERT 到现代 LLM 所用的数据集），这一讲深入**两个具体工艺**：数据过滤（Filtering）和数据去重（Deduplication）。
>
> 这一讲和前面的 Transformer、优化器等话题风格迥异，更像一节"面向大数据处理的算法课"——会用到一点组合数学、哈希、概率。别担心，我会把每一步拆开讲。

---

## 0. 先建立全局图景

训练一个 LLM，数据不是凭空来的。真实流程大致是：

```
活服务（GitHub、论坛、新闻站） 
    ↓ 爬取/打包
原始 dump（GH Archive、CommonCrawl）
    ↓ 处理
可训练语料（The Stack、C4、Dolma……）
```

"处理"这一步要做的事情包括：

- **HTML → 纯文本**
- **语言识别**：只留下你关心的语言
- **质量过滤**：把"真·废话网页"扔掉
- **毒性过滤**：去掉攻击性内容
- **去重**：同一份文本不要训 6 万遍

这一讲就是把"过滤"和"去重"这两个环节里的**具体算法**讲清楚。讲义 `lecture_14.py` 中的执行顺序就是课堂顺序，对照着看就好。

本讲分三大块：

1. **过滤算法**（Filtering Algorithms）——怎么做？给你三种工具：KenLM（n-gram）、fastText（分类器）、DSIR（重要性重采样）。
2. **过滤的应用**（Applications）——拿这些工具干嘛？语言识别、质量过滤、毒性过滤。
3. **去重**（Deduplication）——哈希函数、精确去重、Bloom filter、MinHash、LSH。

---

## 1. 过滤算法：核心抽象

### 1.1 任务的统一形式

所有数据过滤问题都可以写成同一句话：

> 给定一个**小而精的 target 数据集** $T$（你心中"想要"的样子）和一个**又大又杂的 raw 数据集** $R$（比如整个 CommonCrawl），找出 $R$ 的子集 $T'$，使得 $T'$ 在分布上接近 $T$。

对这个算法有两个硬性要求（desiderata）：

1. **要能泛化**。如果 $T' = T$，那没意义——你只是把已有的数据复制了一份。我们希望在 $R$ 里找到"新的但像 $T$ 的"数据。
2. **要非常快**。$R$ 可能是整个互联网。如果你用 GPT-4 去逐条打分，那还不如直接拿这些 FLOPs 去训模型。

下面三种方法都在这套框架下工作，只是"像不像 $T$"的打分函数不一样。

| 方法 | 打分函数 $\text{score}(x)$ | 一句话直觉 |
|---|---|---|
| KenLM (n-gram) | $p_T(x)$ | $x$ 在 target 分布下有多大概率 |
| fastText (分类器) | $p(T \mid x)$ | $x$ 有多像是来自 target 而非 raw |
| DSIR (重要性重采样) | $p_T(x) / p_R(x)$ | $x$ 在 target 下相比 raw 下被"过度表示"的倍数 |

---

### 1.2 方法一：KenLM（n-gram 模型）

#### 1.2.1 n-gram 是什么

最朴素的语言模型。要估计 `p(in | the cat)`（"the cat" 后面接 "in" 的概率），最大似然估计就是数数：

$$
p(\text{in} \mid \text{the cat}) = \frac{\text{count}(\text{the cat in})}{\text{count}(\text{the cat})}
$$

对一个三元组（trigram, n=3）模型，扫一遍语料统计所有长度为 3 的片段出现次数就行。这就是 n-gram 模型的全部精华——**数数，然后归一化**。

#### 1.2.2 稀疏问题与 Kneser-Ney 平滑

数数的毛病叫**稀疏计数**（sparse counts）：很多合理的 n-gram 在训练语料里压根没出现过，count = 0，导致整个句子概率直接归零。n 越大问题越严重，这就是著名的"维度灾难"。

**Kneser-Ney 平滑**的核心思想：

> 如果 3-gram 数据不够，就退一步用 2-gram 估计，甚至更低。把高阶和低阶的估计按一定权重插值或回退（back-off）组合。

具体公式可以去维基百科看，课堂里老师没展开。你只要知道："Kneser-Ney = 一种让 n-gram 模型不会因为数据稀疏而崩掉的经典平滑方法"。

#### 1.2.3 KenLM 和它的用法

**KenLM** 是一个开源工具，原本给机器翻译用的，实现了带 Kneser-Ney 平滑的 n-gram 模型，特点：**极快**。

用它做数据过滤的套路非常简单：

1. 在 target 语料（比如 Wikipedia）上训一个 n-gram 模型。
2. 对 raw 语料的每条文本（通常是"段落"为单位）算 **困惑度（perplexity）**：

$$
\text{PPL}(x) = \exp\!\left(-\frac{\log p(x)}{|x|}\right)
$$

   这里除以 token 数是为了不让"短文本因为概率乘得少所以看起来更像"——归一化后对比才公平。

3. 困惑度**越低** → 越"像"target → 越要保留。

讲义代码里做了 4 个例子：

```python
compute("Stanford University was founded in 1885 ...")   # 来自维基，PPL 低
compute("If you believe that the course staff ...")       # 学校官网文本，PPL 偏高
compute("asdf asdf asdf asdf asdf")                       # 乱码，PPL 高
compute("the the the the the ...")                        # 奇怪的是这句 PPL 不高
```

老师现场吐槽：最后那句 `the the the ...` 的 PPL 居然不算高——说明 n-gram 模型**很蠢**。但做数据过滤不需要 SOTA 模型，只需要"能把垃圾滤掉就好"。

#### 1.2.4 CCNet 实战

CCNet（Facebook 的经典工作，后来 LLaMA 用了它的产物）的做法：

- 把 CommonCrawl 切成**段落**
- 用 Wikipedia 训 KenLM，给每段算 PPL
- 按 PPL 升序排，**只留前 1/3**

简单粗暴，但 work。

#### 1.2.5 课堂 Q&A：n-gram 只看局部怎么评价全局质量？

> 学生问：如果一个文档里把句子顺序打乱，n-gram 模型打分不变，那它怎么能评估"整体质量"？
>
> 老师答：n-gram 确实能被对抗样本骗。但**平均意义上**它 work。你把它理解成"过滤掉真·胡言乱语的网页"——只要不是完全乱码、保持基本的英语语法，n-gram 就会打高分。至于更高层的连贯性，n-gram 就管不了了（你也不需要它管）。

---

### 1.3 方法二：fastText（线性分类器）

#### 1.3.1 背景

fastText 是 Facebook 2016 年的论文。当时神经网络分类器正热，他们证明了**一个几乎线性的分类器**在文本分类上**和神经网络一样好，但快得多**。

#### 1.3.2 朴素版：词袋分类

假设输入句子长 $L$，词表大小 $V$，分 $K$ 类。朴素词袋做法：

```python
W = nn.Embedding(V, K)       # V × K 参数
x = [token ids ...]           # 长度 L
y = softmax(W(x).mean(dim=0)) # 平均 embedding，再 softmax
```

参数量 $V \cdot K$，$V$ 动辄几万甚至百万，爆炸。

#### 1.3.3 fastText 的改进：加一个瓶颈层

```python
H = 16                           # 瓶颈维度
W = nn.Embedding(V, H)           # V × H
U = nn.Linear(H, K)              # H × K
y = softmax(U(W(x).mean(dim=0)))
```

参数量从 $V \cdot K$ 降到 $H \cdot (V + K)$。注意 forward pass 里**没有非线性激活**——就是矩阵分解，本质还是线性分类器。

实现上：**异步并行 SGD，学习率从某个初值线性衰减到 0**。

#### 1.3.4 Bag of n-grams + 哈希技巧

只看单词太弱，想考虑 bigram、trigram。问题是 bigram 数量无上界——语料里能冒出多少不同 bigram 你事先不知道。

**解法：哈希把任意 bigram 压到固定 bin 数**

```python
x = ["the cat", "cat in", "in the", "the hat"]
num_bins = 8                 # 实际用 1000 万
hashed_x = [mmh3.hash(bigram) % num_bins for bigram in x]
```

两个完全不同的 bigram 可能撞到同一个 bin——没关系，训练过程中这个 bin 对应的权重会被调整为两者的"平均效应"，损失函数会自然吸收这种碰撞。

#### 1.3.5 质量过滤的特殊情形

做"好/坏"二分类时 $K=2$，如果再让 $H=2$，fastText 就彻底退化为一个**纯线性分类器**——快到飞起。

> 当然，有钱有算力的话你可以把 fastText 换成 BERT 或 LLaMA 去当分类器。**但**：
> - 分类器跑一遍 $R$ 的 FLOPs = 预训练一次前向的 FLOPs × $|R|/$batch
> - 如果你把 $R$ 过滤到 1%，那分类器的 FLOPs 必须小于预训练 FLOPs 的 1%，否则不如直接拿这些算力去训模型。

---

### 1.4 方法三：DSIR（重要性重采样）

DSIR = **D**ata **S**election via **I**mportance **R**esampling。论文：<https://arxiv.org/abs/2302.03169>。

#### 1.4.1 前置：重要性重采样是什么

这是一个经典的 Monte Carlo 技巧，常出现在粒子滤波等场景。

**问题**：你想从目标分布 $p$ 采样，但只能从提议分布 $q$ 采样。

**做法**：

1. 从 $q$ 里采 $n$ 个样本 $x_1, \dots, x_n$。
2. 给每个样本算**重要性权重**：

$$
w_i \propto \frac{p(x_i)}{q(x_i)}
$$

   然后归一化：$w_i \leftarrow w_i / \sum_j w_j$。

3. 以权重 $w$ 为概率，从这 $n$ 个样本里**有放回地**再采 $n$ 个——这就是"resample"。

**直觉**：$q$ 把你采偏了，$p/q$ 是"纠正倍率"——$p$ 偏爱的地方被采少了就加权，$q$ 偏爱但 $p$ 不爱的就降权。

讲义里的具体代码：

```python
vocabulary = [0, 1, 2, 3]
p = [0.1, 0.2, 0.3, 0.4]      # target，偏爱大的
q = [0.4, 0.3, 0.2, 0.1]      # proposal，偏爱小的

samples = np.random.choice(vocabulary, p=q, size=n)    # 从 q 采，0 多 3 少
w = [p[x] / q[x] for x in samples]                      # 算权重
w = [wi / sum(w) for wi in w]                           # 归一化
resampled = np.random.choice(samples, p=w, size=n)      # 重采样 → 3 变多
```

#### 1.4.2 从"分布"到"数据集"

回到数据过滤。我们手上不是两个分布，而是两个**数据集**：

- $D_p$：target 数据（小而精）
- $D_q$：raw 数据（大而杂）

**Take 1（朴素）**：在 $D_p$ 上拟合一个分布 $p$，在 $D_q$ 上拟合 $q$，然后对 $D_q$ 的样本做重要性重采样。

**问题**：$D_p$ 太小，拟合不出好模型（这正是我们想过滤更多数据的原因啊）。

**Take 2（DSIR 实际做法）**：和 fastText 一样，**用哈希 n-gram 作为特征**，把分布拟合成一个低维的"哈希桶上的 unigram 模型"。这样参数量极少，小数据也能训出来。

讲义代码：

```python
num_bins = 4
def get_hashed_ngrams(text):
    ngrams = text.split()                          # 这里用 unigram 演示
    return [mmh3.hash(ng) % num_bins for ng in ngrams]

training_hashed = get_hashed_ngrams("the cat in the hat")
# 比如 → [2, 1, 3, 2, 2]，注意 "the" 和 "hat" 都撞到 2 了（不管）

# 用桶内频率估计概率
probs = [count(training_hashed, x) / len(training_hashed) for x in range(num_bins)]

# 对新文本打分：哈希后连乘
new = get_hashed_ngrams("the text")
prob = np.prod([probs[x] for x in new])
```

（注意：Python 原生 `hash` 是非确定性的，不同进程结果不同——`mmh3` 才是确定性的。）

#### 1.4.3 和 fastText 的对比

| 维度 | fastText | DSIR |
|---|---|---|
| 打分函数 | $p(T \mid x)$（判别式） | $p_T(x) / p_R(x)$（重要性比） |
| 理论动机 | 二分类 | 从 $p_R$ 重构 $p_T$ 的样本 |
| 多样性 | 没保证 | **理论上**能保留 $p_T$ 的多样性 |
| 速度 | 快 | 同量级 |

在 GLUE benchmark 上 DSIR 略胜 fastText，但提升不大——两者都还有巨大的改进空间（用更大的 n、更好的模型）。

---

### 1.5 过滤算法小结

三种方法的统一框架：

1. 基于 $R$ 和 $T$ 估一个**打分函数** $\text{score}(x)$。
2. 根据 $\text{score}(x)$ 决定保不保留 $x$（可以阈值化，也可以随机化）。

三种实例化：

- **生成式（KenLM）**：$\text{score}(x) = p_T(x)$，阈值保留。
- **判别式（fastText）**：$\text{score}(x) = p(T \mid x)$，阈值保留。
- **重要性重采样（DSIR）**：$\text{score}(x) = p_T(x)/p_R(x)$，按此概率重采样。

综述论文推荐：<https://arxiv.org/abs/2402.16827>。

---

## 2. 过滤的应用

同一套机器可以干不同的活。

### 2.1 语言识别（Language Identification）

#### 为什么要识别语言？

为什么不直接训多语模型？

- **数据侧**：给定语言高质量数据的收集/处理本身就难。
- **算力侧**：在算力受限时，每种语言能分到的 token 更少，你最关心的那门语言可能被拖累。

历史案例：

- **BLOOM**（2022 年）英语只占 30%，结果英语性能被拖。
- 现代前沿模型（GPT-4、Claude、Gemini、LLaMA、Qwen）训练规模大到可以"多语全通吃"，甚至有**正迁移**。

#### fastText 语言识别模型

Facebook 开源了一个现成的 fastText 语言识别模型（`lid.176.bin`），支持 176 种语言，训练数据来自 Wikipedia、Tatoeba（翻译站）、SETimes（东南欧新闻）。

**Dolma 的用法**：保留 $p(\text{English}) \geq 0.5$ 的页面。

讲义里的有趣测试：

```python
"The quick brown fox ..."                   → English, 概率 0.71
"OMG that movie was 🔥🔥! So dope 😎🤘!"     → 能识别，但偏弱
"Auf dem Wasser zu singen"                   → German（对）
"The quadratic formula is $x = ...$"         → 识别较弱（LaTeX 混杂）
"for (int i = 0; i < 10; i++)"               → 居然判成俄语
"Hello!"                                     → 判成意大利语（太短）
"Feliz Navidad / ... / Merry Christmas"      → 判成西班牙语（代码混用）
```

**经验教训**：

- 短句不可靠（信息量不够）
- 低资源语言困难
- 方言容易被误判为"非英语"
- 相似语言难分（马来 vs 印尼）
- 代码混用（code-switching）本身就没标准答案

#### 案例：OpenWebMath

把"数学"当成一种"语言"来过滤：

1. 规则过滤（如必须含 LaTeX 命令）
2. 在数学证明语料 **ProofPile** 上训 KenLM，保留 PPL < 15000 的文本
3. 训一个 fastText 分类器判定"是否是数学写作"：
   - 如果规则判定是数学 → 阈值低一点（0.17）
   - 如果规则判定不是数学 → 阈值高一点（0.8）

产物：14.7B 数学 tokens，训出的 1.4B 模型表现**超过用 20× 一般数据训的同规模模型**。

> 这展示了数据过滤的巨大价值：**定向收集**几乎永远比"用更多杂数据"划算。

---

### 2.2 质量过滤（Quality Filtering）

"质量"本身是个 catch-all 概念，各家定义不同。

**态度分两派**：

- 不用模型过滤（觉得规则够了）：C4、Gopher、RefinedWeb、FineWeb、Dolma
- 用模型过滤：GPT-3、LLaMA、DCLM ← 正在成为主流

#### GPT-3 的做法

- **正例**：{Wikipedia, WebText2, Books1, Books2} 的样本
- **负例**：CommonCrawl 的样本
- 训一个**基于词特征的线性分类器**
- **随机保留**，用 Pareto 分布做"软阈值"：

```python
def keep_document(score):
    return np.random.pareto(9) > 1 - score
```

Pareto 分布长尾很重，这个公式的效果是：score 越高，保留概率越大，但分数低的也有一小撮可能被保留——**增加多样性**。

#### LLaMA / RedPajama 的做法

- **正例**：被 Wikipedia **引用**过的页面（不是 Wikipedia 本身）
- **负例**：CommonCrawl
- 分类为正的就保留（没有随机化）

#### phi-1 的做法（非常有意思）

phi-1 哲学："**用极高质量的数据（像教科书）训小模型（1.5B）**"。

```
R = Python 子集（来自 The Stack）
prompt = "判断这段代码对想学基础编程的学生的教育价值"
T = 用 GPT-4 按上面 prompt 给 R 里 100K 条打标 → 正例
```

然后：

- 拿 T 训一个**随机森林分类器**（特征用预训练 codegen 模型的 embedding）
- 用这个分类器去筛选 $R$

**结果**（HumanEval 基准）：

| 方案 | 模型 | 步数 | 性能 |
|---|---|---|---|
| 直接在 The Stack Python 子集训 | 1.3B | 96K | 12.19% |
| 在新过滤子集训 | 1.3B | 36K | 17.68% |

更少步数，更好性能。

**学生问**：为啥不直接用 GPT-4 当分类器？
**答**：因为 GPT-4 只跑了 100K 条（生成 T），然后**蒸馏**到一个便宜快速的随机森林。100K ≪ $R$ 的规模（千万/亿级）。

> **大趋势**：以前我们说"Wikipedia 是好数据、Books 是好数据"，这是"选源头"。现在，有了好模型，你可以说"我要更多化学数据/证明题数据"，直接让强模型帮你**定义** $T$。拿到 $T$ 后走之前的套路。

---

### 2.3 毒性过滤（Toxicity Filtering）

⚠ 本节涉及冒犯性内容示例。

以 Dolma 为例：

**数据来源**：**Jigsaw Toxic Comments**（2018 Kaggle 竞赛）。Wikipedia 讨论页评论被人工标注为 {toxic, severe_toxic, obscene, threat, insult, identity_hate}。

Dolma 训了**两个** fastText 分类器：

- **hate**：positive = {unlabeled, obscene}，negative = 其他
- **NSFW**：positive = {obscene}，negative = 其他

（这俩定义看着奇怪，但就是 Dolma 的实际做法。）

讲义里的两个样例：

```python
# 一句辩论性质的话（label=0，safe）
"Are you threatening me for disputing neutrality? ..."
# → 模型判定 safe-for-work

# 一句明显骂人的（label=1，obscene）
"Stupid peace of shit stop deleting my stuff ..."
# → 模型判定 NSFW
```

---

## 3. 去重（Deduplication）

### 3.1 为什么要去重

**两种重复**：

1. **精确重复**：镜像站（Project Gutenberg 有无数镜像）、GitHub 分叉——CommonCrawl 没法知道这是同一份内容。
2. **近似重复**：同一段话差几个 token。例子：
   - 服务条款、MIT 许可证（到处复制）
   - 模板化文本（把国家名从 "Canada" 换成 "USA" 的广告页）
   - 复制粘贴时漏个逗号

**极端案例**：C4 里有一段广告文案**原封不动重复了 61,036 次**：

> "by combining fantastic ideas, interesting arrangements, and follow the current trends..."

这段话本身不是"低质量"，但训练 6 万遍显然是浪费。

**所以去重 vs 质量过滤**：

- 质量过滤：这条数据**永远**别用
- 去重：这条数据**可以**用，但我**只想要一份**

**去重的好处**：

- **训练更高效**：token 少了
- **减少记忆**：模型不会原样吐出训练集 → 缓解版权、隐私风险

### 3.2 去重的设计空间

三个维度：

1. **item 是什么？**——句子 / 段落 / 文档？
2. **怎么算"重复"？**——精确相等 / 有共享子项 / 共享子项比例高？
3. **怎么处理？**——全删 / 留一份？

**核心难点**：去重天然是**成对**问题——要比较 item 两两之间。朴素 $O(n^2)$ 根本不可行（$n$ 是百亿级）。**我们需要线性时间算法**。

---

### 3.3 哈希函数速览

哈希函数 $h$：把任意 item（字符串、文档）映到一个整数/字符串，**值远小于 item**。

**碰撞**（collision）：$h(x) = h(y)$ 但 $x \neq y$。

两大类：

| 类型 | 代表 | 速度 | 抗碰撞 | 用途 |
|---|---|---|---|---|
| 密码学哈希 | SHA-256 | 慢 | 强 | 比特币、签名 |
| 快速哈希 | DJB2、MurmurHash、CityHash | 快 | 弱 | 哈希表 |

我们做数据处理**不需要**抗碰撞强度，要的是**速度**。讲义统一用 **MurmurHash**（Python 包 `mmh3`）。

一个有趣的转变：**平常我们视碰撞为敌人，但后面 MinHash 会把碰撞变成朋友**——让"更像的 item 碰撞概率更高"。

---

### 3.4 精确去重

最简单的版本：

```python
items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]

# 按哈希值分组
grouped = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)
# 每组留一个
deduped = [next(group) for h, group in grouped]
```

- **优点**：简单、语义清楚、精度高（不会误删）
- **缺点**：抓不到近似重复
- **额外好处**：这是 MapReduce 友好的写法，天然可分布式。

**C4 的做法**：以**3 句连续片段**为 item，精确匹配，留一份。

> 老师吐槽：如果一个文档中间那三句被去掉了，剩下的上下文可能就不连贯了。但"Who cares, 大家都这么干"。

---

### 3.5 Bloom Filter（布隆过滤器）

#### 3.5.1 核心特性

一个**近似集合成员查询**的数据结构：

- **省内存**
- **能插入，不能删除**
- **返回"否" → 绝对不在集合**
- **返回"是" → 很可能在，但有小概率假阳性**
- **假阳率可以通过参数指数级压降**

#### 3.5.2 单哈希版本

```python
m = 8                                 # bit 数组长度
table = bitarray(m)                   # [0,0,0,0,0,0,0,0]
items = ["the", "cat", "in", "the", "hat"]

for item in items:
    h = mmh3.hash(item) % m           # 比如 the→2, cat→7, ...
    table[h] = 1                      # 把对应位置 1
```

查询：

```python
def query(item):
    return table[mmh3.hash(item) % m]
```

讲义里测了 `["what", "who", "why", "when", "where", "which", "how"]`，居然有 **4 个误报**！假阳率 = $4/(5+4) \approx 0.44$，惨不忍睹。

**假阳率**（$m=$ bin 数）：

$$
f \approx \frac{1}{m}
$$

要降到 $10^{-10}$ 就得 $m = 10^{10}$——太浪费内存。

#### 3.5.3 多哈希版本（真正的 Bloom Filter）

**核心 trick**：用 $k$ 个不同的哈希函数（用不同的 seed 实现），每个 item 把 $k$ 个位置都置 1；查询时必须**所有** $k$ 个位置都是 1 才返回 yes。

```python
def build_table_k(items, m, k):
    table = bitarray(m)
    for item in items:
        for seed in range(k):
            h = mmh3.hash(item, seed) % m
            table[h] = 1
    return table

def query_table_k(table, item, m, k):
    return all(table[mmh3.hash(item, seed) % m] for seed in range(k))
```

**为啥有效**：假阳需要 $k$ 个位置"同时"被其他 item 占亮——比单次巧合概率小得多。

#### 3.5.4 假阳率公式推导（重点，慢慢讲）

**设定**：

- $m$ = bin 数
- $k$ = 哈希函数数
- $n$ = 插入的 item 数
- 考察一个"不在集合里的测试 item"，它的某个哈希会落到某个 bin $i$。问：插入 $n$ 个 item 后 $B[i] = 1$ 的概率 $f$ 是多少？（如果 $B[i]=1$，才可能造成假阳。）

**Step 1：$n=1, k=1$（插 1 个，用 1 个哈希）**

插入一个 item 时，它被某个哈希函数打到 bin $i$ 的概率是 $1/m$。所以

$$
\Pr[B[i]=1] = \frac{1}{m}
$$

**Step 2：$n=1, k$（插 1 个，用 $k$ 个哈希）**

这个 item 会置亮 $k$ 个位置。$B[i]$ 被置亮，等价于"$k$ 次哈希里至少一次打到 $i$"。

反过来看：$k$ 次哈希**都没**打到 $i$ 的概率是 $(1 - 1/m)^k$。所以

$$
\Pr[B[i]=1] = 1 - \left(1 - \frac{1}{m}\right)^k
$$

**Step 3：$n$ 个，$k$ 个哈希**

每个 item 要插入 $k$ 次。$n$ 个 item 总共有 $k n$ 次哈希操作，每次都要"没打到 $i$"才能让 $B[i]$ 保持 0。假设独立：

$$
\Pr[B[i]=0] = \left(1 - \frac{1}{m}\right)^{k n}
$$

$$
\Pr[B[i]=1] = 1 - \left(1 - \frac{1}{m}\right)^{k n}
$$

**Step 4：真正的假阳率**

我们关心的"测试 item 被判为 yes"的概率——它得**所有 $k$ 个哈希位置**都是 1：

$$
f = \left[1 - \left(1 - \frac{1}{m}\right)^{k n}\right]^{k}
$$

代入讲义例子 $m=1000, k=10, n=100$：

- $(1 - 1/1000)^{1000} \approx 1/e$，所以 $(1 - 1/m)^{kn} = (1-1/1000)^{1000} \approx 0.368$
- $1 - 0.368 = 0.632$
- $0.632^{10} \approx 0.01$

用 Python 对应：

```python
f = 1 / m                             # Step 1: 0.001
f = 1 - (1 - 1/m)**k                  # Step 2: ~0.01
f = 1 - (1 - 1/m)**(k*n)              # Step 3: ~0.63
f = f**k                              # Step 4: ~0.008
```

#### 3.5.5 最优 $k$

固定 $m/n$ 比率下，对 $k$ 求导最小化 $f$，得到（渐近结果）：

$$
k^* = \ln 2 \cdot \frac{m}{n} \approx 0.693 \cdot \frac{m}{n}
$$

代入最优 $k$ 后，恰好有 $(1 - 1/m)^{k n} \approx 0.5$，所以

$$
f_{\min} = 0.5^{k^*} = \left(\frac{1}{2}\right)^{0.693 m/n}
$$

**直觉**：$k$ 太小 → 每个 item 占的位置少，单哈希易碰撞；$k$ 太大 → 位数组被塞满，大家都是 1，查啥都 yes。中间存在最优。

#### 3.5.6 Dolma 的设置

Dolma 用 Bloom Filter 做**段落级**去重，设定假阳率 $10^{-15}$。

---

### 3.6 近似去重：Jaccard 相似度 + MinHash

精确去重抓不到"差一个逗号"的近似重复。要做近似去重，先要定义"相似"。

#### 3.6.1 Jaccard 相似度

两个集合 $A, B$ 的 **Jaccard 相似度**：

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

例：$A = \{1,2,3,4\}, B = \{1,2,3,5\}$，交集 $\{1,2,3\}$ 大小 3，并集大小 5，$J = 3/5 = 0.6$。

**定义近似重复**：$J(A, B) \geq $ 某阈值（如 0.9）。

（注意：Jaccard 没有语义，只看表面符号重合。漏掉一个 "not" 会改变意思，但 Jaccard 看不出来——这是已知局限。）

**算法挑战**：在**线性时间**里找出所有近似重复对。

#### 3.6.2 MinHash 的神奇性质

我们有个成对函数 $J(A,B)$，需要把它**转成只需算单个 item 的函数**才能线性化。这就是 MinHash 的作用：

> 定义一个随机哈希函数 $h_{\text{min}}(S)$，使得 $\Pr[h_{\text{min}}(A) = h_{\text{min}}(B)] = J(A, B)$。

也就是**有意让"相似 item 的哈希碰撞概率高"**——和平常哈希反着来。

**定义**（给定种子 $s$）：

```python
def minhash(S, seed):
    return min(mmh3.hash(x, seed) for x in S)
```

对集合 $S$，用 $h$ 把每个元素映到整数，取**最小值**作为 $S$ 的"签名"。

#### 3.6.3 为什么 $\Pr[\text{MinHash}(A) = \text{MinHash}(B)] = J(A,B)$？

关键证明。用"特征矩阵"视角：

| item | $\in A$? | $\in B$? |
|---|---|---|
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 3 | 1 | 1 |
| 4 | 1 | 0 |
| 5 | 0 | 1 |

- $A = \{1,2,3,4\}, B = \{1,2,3,5\}$, $A \cup B = \{1,2,3,4,5\}$, $A \cap B = \{1,2,3\}$

随机哈希函数 $h$ **等价于**对 $A \cup B$ 做一个均匀随机排列。具体地，哈希值的相对大小关系就是一个随机排列。

现在考虑：对这个随机排列，**谁排在最前面**（即哈希值最小）？由对称性，$A \cup B$ 里每个 item 当第一的概率都是 $\frac{1}{|A \cup B|}$。

关键观察：

- 如果"第一名"是 item **1、2 或 3**（都在 $A \cap B$ 里）：它既在 $A$ 也在 $B$，所以 $\min_{x \in A} h(x) = \min_{x \in B} h(x) = h(\text{item 1/2/3})$。**碰撞**。
- 如果"第一名"是 item **4**（只在 $A$）：$A$ 的 min 是 $h(4)$，但 $B$ 里没有 4，$B$ 的 min 是别的什么——**不碰撞**。
- 如果"第一名"是 item **5**（只在 $B$）：同理**不碰撞**。

所以

$$
\Pr[\text{MinHash}(A) = \text{MinHash}(B)] = \frac{|A \cap B|}{|A \cup B|} = J(A, B)
$$

✅ 证毕。

**经验验证**：讲义用 100 个不同 seed 的 MinHash，统计碰撞比例 ≈ 0.6（等于真实 Jaccard）。

```python
n = 100
matches = [minhash(A, s) == minhash(B, s) for s in range(n)]
estimated = sum(matches) / n           # ≈ 0.6
```

---

### 3.7 局部敏感哈希（LSH）

#### 3.7.1 动机

用 1 个 MinHash：$\Pr[\text{碰撞}] = J(A,B)$，相似度 0.9 的有 90% 机会碰撞，相似度 0.1 的也有 10% 机会——**还是太随机**。

**目标**：碰撞概率随相似度产生**阶跃**——超过阈值几乎必然碰撞，低于阈值几乎不碰撞。

#### 3.7.2 Band 构造（核心 trick）

用 $n$ 个 MinHash 函数，把它们分成 $b$ 个 **band**，每个 band 有 $r$ 个哈希函数（$n = b \cdot r$）。

例子：$n=12, b=3, r=4$

```
band 1     |  band 2      |  band 3
h1 h2 h3 h4 | h5 h6 h7 h8  | h9 h10 h11 h12
```

**碰撞规则**：$A$ 和 $B$ 碰撞 ↔ **存在至少一个 band**，使得该 band 内 $r$ 个哈希值**全部相等**。

注意这是 "OR over bands, AND within band" 的嵌套结构——关键就是这个 AND-OR 构造能**锐化**概率曲线。

#### 3.7.3 碰撞概率推导

设 $s = J(A, B)$。

**Step 1**：某一个哈希函数位置上 $A$ 和 $B$ 相等的概率：$s$（MinHash 性质）。

**Step 2**：某一个 band（$r$ 个哈希）**全部相等**的概率：

$$
p_{\text{band}} = s^r
$$

（每个 MinHash 的 seed 不同，独立。）

**Step 3**：某一个 band**不**全等的概率：$1 - s^r$。

**Step 4**：**所有 $b$ 个 band 都不全等**的概率（即 $A,B$ 完全不碰撞）：$(1 - s^r)^b$。

**Step 5**：**至少一个 band 全等**（即 $A,B$ 碰撞）的概率：

$$
\boxed{\ p_{\text{collide}}(s) = 1 - (1 - s^r)^b \ }
$$

Python：

```python
def get_prob_collision(sim, b, r):
    prob_match = sim ** r
    prob_collision = 1 - (1 - prob_match) ** b
    return prob_collision
```

#### 3.7.4 $r$ 和 $b$ 分别在调什么

画 $p_{\text{collide}}(s)$ 的曲线：

- **增大 $r$**：曲线**向右平移**并且**更陡**（每个 band 更难全等，但你 AND 完一旦全等相似度就很高）。
- **增大 $b$**：曲线**向左平移**（给了更多"抽奖机会"，总会有一个 band 成功）。

所以 $b, r$ 协同调节：$r$ 控制"对相似度的苛刻度"，$b$ 控制"给多少次机会"。最终能得到一条**接近阶跃的 sigmoid**——这就是 LSH 的精髓。

#### 3.7.5 理论阈值

近似地，曲线的"跃迁点"（斜率最大处）满足：

$$
s^* \approx \left(\frac{1}{b}\right)^{1/r}
$$

在 $s = s^*$ 处，$p_{\text{band}} = 1/b$，此时碰撞概率：

$$
p_{\text{collide}}(s^*) = 1 - \left(1 - \frac{1}{b}\right)^b \xrightarrow{b \to \infty} 1 - \frac{1}{e} \approx 0.632
$$

——阈值点附近概率是 $1 - 1/e$，上下两边迅速拉开。

#### 3.7.6 实战数值（论文 arXiv:2107.06499）

$n = 9000, b = 20, r = 450$

阈值：

$$
s^* = (1/20)^{1/450} \approx 0.993
$$

意思是：**每 100 个 word 只允许差不到 1 个**——极其严格的近似去重。

- 一个 band 全匹配的概率：$1/b = 0.05$
- 阈值处总碰撞概率：$\approx 0.632$

---

### 3.8 课堂 Q&A 节选

**Q：合成数据的"改写式"重复怎么去？**

A：可以用文档的 **embedding** 代替字面哈希，在 embedding 空间做近邻搜索——这仍然能套进 LSH 框架（LSH 本来就是给 ANN 用的）。代价：跑 embedding 贵。警告：太宽松的相似阈值会误删大量数据。

**Q：有没有时候**反而**想保留重复？**

A：有。**Mid-training** 阶段，人们会故意让高质量数据**过多个 epoch**。去重主要是 pre-training 阶段针对"6 万份同样垃圾"。更聪明的做法：根据 count 取 log 或 sqrt 来决定采样权重——既承认它"重要"，又不让它霸榜。

---

## 4. 全局总结

### 算法工具箱

- **过滤**：KenLM（n-gram）、fastText（线性分类器）、DSIR（重要性重采样）
- **过滤应用**：语言识别、质量过滤、毒性过滤
- **去重**：
  - 精确去重 → Bloom Filter（多哈希 + bit 数组，AND 结构）
  - 近似去重 → MinHash + LSH（MinHash 把成对 Jaccard 变成单点哈希；LSH 用 band 的 AND-OR 构造锐化阈值）

### 一个更抽象的认识

这一讲本质是两个主题：

1. **"从大海捞针"的统一框架**：不管是过滤质量、识别语言还是找教科书级代码——都是在学 $T$ 和 $R$ 的差异，用它打分/重采样。
2. **哈希是大数据处理的原子武器**：通过"多个哈希 + AND/OR 构造"，能把 $O(n^2)$ 的成对比较压到 $O(n)$，同时精细调控误差。

### 最后的忠告

> "Now you have the tools (mechanics), just have to spend time with data (intuitions)."

算法讲完只是开始。真正理解"什么叫好数据"要靠和数据泡在一起——看样本、调阈值、训模型、再回来看样本。Assignment 4 会给你这个机会。
