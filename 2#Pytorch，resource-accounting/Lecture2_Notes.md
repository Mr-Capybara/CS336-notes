# CS336 Lecture 2：PyTorch & Resource Accounting 笔记

> Stanford CS336 - Language Modeling from Scratch (Spring 2025)
> 授课：Percy Liang

---

## 一、开场：Napkin Math 的重要性

本讲从两个"餐巾纸计算"问题切入，展示 resource accounting 的核心思维。

### 问题 1：训练 70B 模型需要多久？

> 70B 参数 dense transformer，15T tokens，1024 张 H100。

推导过程：

1. **总 FLOPs** = 6 × 参数量 × token 数 = 6 × 70B × 15T
2. **单卡 FLOPs/s**：H100 的标称值（BF16 约 990 TFLOP/s）
3. **MFU**（Model FLOPs Utilization）设为 0.5
4. **集群每日 FLOPs** = 1024 × FLOPs/s × 86400 × MFU
5. 总天数 = 总 FLOPs / 每日 FLOPs ≈ **144 天**

> 其中 `6 × N × D` 的来源将在本讲后半部分推导。

### 问题 2：8 张 H100 最大能训多大的模型？

- H100 HBM = 80 GB
- 使用 AdamW 时，每个参数需 **16 bytes**（参数 + 梯度 + 两个 optimizer state，各 4 bytes）
- 最大参数量 ≈ (8 × 80 GB) / 16 bytes ≈ **40B**

> 这是粗略估计，未计入 activations（取决于 batch size 和 sequence length）。

### 核心观点

> 效率是核心。训练成本直接等于美元。你必须知道每一个 FLOP 花在了哪里。

---

## 二、本讲知识类型

| 类型       | 内容                                      |
| ---------- | ----------------------------------------- |
| Mechanics  | PyTorch 底层原语：tensor、view、einops 等 |
| Mindset    | Resource accounting：内存 & 算力精确计量  |
| Intuitions | 本讲较少涉及模型设计直觉                  |

> 本讲**不讲 Transformer 架构**（下一讲由 Tatu 讲），而是用简单线性模型讲清原语和资源计量。

---

## 三、Memory Accounting

### 3.1 Tensor 基础

Tensor 是深学习中存储一切的基本单元：参数、梯度、optimizer state、数据、activations。

**内存公式**：

```
内存 = 元素数量 × 每个元素的字节数
```

例：`torch.zeros(4, 8)` → float32 → 32 个元素 × 4 bytes = **128 bytes**

> 直觉参考：GPT-3 中 FFN 层的一个权重矩阵 ≈ **2.3 GB**。

### 3.2 浮点数类型对比

| 类型     | 位数 | 指数位 | 尾数位 | 特点                                        |
| -------- | ---- | ------ | ------ | ------------------------------------------- |
| FP32     | 32   | 8      | 23     | 默认精度，"full precision"（对 ML 而言）    |
| FP16     | 16   | 5      | 10     | 半精度，动态范围差，容易 underflow/overflow |
| **BF16** | 16   | 8      | 7      | 与 FP32 相同动态范围，分辨率略差            |
| FP8      | 8    | 4/5    | 3/2    | H100 支持，非常粗糙，两种变体               |

#### FP16 的问题

```python
torch.tensor(1e-8, dtype=torch.float16)  # → 0.0（underflow）
```

大模型训练中容易出现数值不稳定。

#### BF16 的优势

- 2018 年 Google Brain 提出，专为深度学习设计
- **动态范围 = FP32**，分辨率比 FP16 差，但对 DL 影响不大

```python
torch.tensor(1e-8, dtype=torch.bfloat16)  # → 非零值 ✓
```

#### 实践建议

| 用途                  | 推荐精度     |
| --------------------- | ------------ |
| 参数存储 & optimizer  | FP32         |
| 前向/反向计算         | BF16         |
| 梯度累积              | FP32         |
| Attention（保守做法） | FP32         |
| 推理 / 量化           | FP8 甚至更低 |

> **不推荐使用 FP16**——BF16 在所有场景下都是更好的选择。

---

## 四、Compute Accounting

### 4.1 GPU 与数据位置

PyTorch 默认将 tensor 放在 CPU 上，必须显式移到 GPU：

```python
x = torch.zeros(32, 32)           # CPU
x = x.to("cuda")                  # → GPU
# 或直接创建在 GPU 上
x = torch.zeros(32, 32, device="cuda")
```

> **始终清楚每个 tensor 在哪**。代码中看不出来，必要时用 assert 检查。

### 4.2 Tensor 的内部表示：Storage + Stride

Tensor 在底层是一段连续内存 + 元数据（stride）。

对于 4×4 矩阵：
- 底层存储：长度 16 的一维数组
- `stride = (4, 1)`：行方向跳 4，列方向跳 1
- 访问 `[i, j]` → 偏移 = `i × stride[0] + j × stride[1]`

### 4.3 View 操作：零拷贝

多个 tensor 可以共享同一块 storage，只是 stride/offset 不同：

| 操作            | 是否拷贝 | 说明                                |
| --------------- | -------- | ----------------------------------- |
| `x[0]`          | ❌        | 取行，view                          |
| `x[:, 1]`       | ❌        | 取列，view                          |
| `.view(3, 2)`   | ❌        | reshape，view                       |
| `.transpose()`  | ❌        | 转置，view（但变为 non-contiguous） |
| `.contiguous()` | ✅        | 非连续 → 连续，**会拷贝**           |
| `.reshape()`    | 可能     | = `.contiguous().view()`            |

**关键陷阱**：view 共享 storage，修改一个会影响另一个：

```python
x = torch.tensor([[1,2,3],[4,5,6]])
y = x[0]
x[0, 0] = 99
# y[0] 也变成 99
```

> View 是免费的，放心用来提升代码可读性。但注意 `contiguous()` / `reshape()` 可能触发拷贝。

### 4.4 创建新 Tensor 的操作

- 所有 element-wise 操作（加减乘除、`triu` 等）都会分配新内存
- `torch.triu` 常用于构造 causal attention mask

---

## 五、矩阵乘法：深度学习的核心运算

### 5.1 基本 MatMul

```
(16×32) @ (32×2) → (16×2)
```

### 5.2 Batched MatMul

实际场景中 tensor 通常是 `(batch, seq, hidden)` 形状：

```
(B, S, D) @ (D, K) → (B, S, K)
```

PyTorch 自动对前面的维度做 broadcast。

### 5.3 Einops & Einstein Summation

传统写法可读性差：

```python
x @ y.transpose(-2, -1)  # -2 是啥？-1 是啥？
```

**Einsum 写法**——给每个维度命名：

```python
torch.einsum("bsh, bth -> bst", x, y)
```

规则：
- 输出中**出现**的维度 → 保留（遍历）
- 输出中**不出现**的维度 → 求和（contracted）

#### Jaxtyping 标注

```python
x: Float[Tensor, "batch seq hidden"]
```

> 不强制检查，但比注释更自然的文档方式。

#### Reduce 操作

```python
reduce(x, "batch seq hidden -> batch seq", "sum")
# hidden 维度消失 → 对 hidden 求和
```

#### Rearrange 操作

用于拆分/合并维度（如 multi-head attention 的 head 拆分）：

```python
rearrange(x, "b s (heads h) -> b heads s h", heads=8)
# 反向：
rearrange(x, "b heads s h -> b s (heads h)")
```

> 推荐投入时间学习 einops，assignment 1 中会用到。`torch.compile` 会将 einsum 编译为高效实现。

---

## 六、FLOPs 计量

### 6.1 定义澄清

| 写法            | 含义                                         |
| --------------- | -------------------------------------------- |
| flops（小写 s） | floating point operations（计算量）          |
| FLOP/s          | floating point operations per second（速度） |

> 本课程不用大写 FLOPS，因为歧义太大。

### 6.2 规模直觉

- GPT-3 训练：~3×10²³ flops
- GPT-4 训练：~2×10²⁵ flops（推测）
- 美国行政令（已撤销）：>10²⁶ flops 的模型需报告
- EU AI Act（仍有效）：>10²⁵ flops

### 6.3 硬件 FLOP/s

| GPU  | FP32 FLOP/s | BF16 FLOP/s（无稀疏） | BF16（含稀疏） |
| ---- | ----------- | --------------------- | -------------- |
| A100 | 19.5 T      | 312 T                 | —              |
| H100 | 67 T        | ~990 T                | ~1979 T        |

> **稀疏性**：H100 spec 上标的数字带星号，指 2:4 structured sparsity。实际 dense 矩阵取一半。"Marketing department uses it."

### 6.4 MatMul 的 FLOPs 公式

**核心公式**：

```
MatMul FLOPs = 2 × M × N × K
```

其中 `(M×N) @ (N×K) → (M×K)`。每个输出元素需要 N 次乘法 + N 次加法 = 2N 次操作。

> **记住这个**：矩阵乘法的 flops = 2 × 三个维度的乘积。

其他操作（element-wise 等）的 flops 通常是线性的，对于足够大的矩阵，**MatMul 绝对主导**。

> 这也是为什么 napkin math 很简单——只需要数 MatMul。

### 6.5 线性模型的 FLOPs

对于 `Y = X @ W`，其中 X 是 `(B, D)`，W 是 `(D, K)`：

```
Forward FLOPs = 2 × B × D × K = 2 × tokens × params
```

> 这个关系对 Transformer 也近似成立（当 sequence length 不太大时）。

---

## 七、MFU（Model FLOPs Utilization）

### 定义

```
MFU = (模型理论 FLOPs / 实际耗时) / 硬件标称 FLOP/s
```

即：你实际榨出了硬件多少比例的算力。

### 实测示例

FP32 MatMul on H100：
- 实测 5.4×10¹³ FLOP/s
- 标称 6.7×10¹³ FLOP/s
- MFU ≈ **0.8**

BF16 MatMul on H100：
- 速度快约 5 倍（0.03s vs 0.16s）
- 但标称值也高得多 → MFU 反而更低

### 经验法则

| MFU   | 评价                      |
| ----- | ------------------------- |
| > 0.5 | 不错                      |
| ~0.05 | 很差，硬件严重浪费        |
| ~0.8+ | 优秀（通常只有纯 MatMul） |

> MFU 衡量的是**模型有效计算**，不是硬件实际执行的 flops。如果你用了 activation checkpointing 等技巧重复计算，MFU 不会因此降低——因为它只看"模型需要多少 flops"。

> **Always benchmark your code.** 不要假设能达到标称性能。

---

## 八、Backward Pass 的 FLOPs

### 8.1 两层线性网络示例

模型：`X → H1 = X @ W1 → H2 = H1 @ W2 → Loss`

**Forward FLOPs**：
- `X @ W1`：2 × B × D × D
- `H1 @ W2`：2 × B × D × K
- 合计 = **2 × B × (总参数量)**

### 8.2 Backward Pass 分析

对于 W2 层，需要计算两个梯度：

1. **∂L/∂W2** = H1ᵀ @ ∂L/∂H2 → FLOPs = 2 × B × D × K
2. **∂L/∂H1** = ∂L/∂H2 @ W2ᵀ → FLOPs = 2 × B × D × K

每层的 backward = **2 × forward**（因为要算两个梯度：参数梯度 + 输入梯度用于继续反传）。

W1 层同理。

### 8.3 总结：6N 法则

| 阶段     | FLOPs              |
| -------- | ------------------ |
| Forward  | 2 × B × params     |
| Backward | 4 × B × params     |
| **总计** | **6 × B × params** |

> 这就是开头 `6 × N × D` 的来源。对大多数模型（参数不共享的情况下）近似成立。

---

## 九、模型构建

### 9.1 参数初始化

朴素初始化（标准正态）的问题：

```python
w = nn.Parameter(torch.randn(input_dim, hidden_dim))
output = x @ w  # 输出值随 hidden_dim 增大而爆炸
```

输出的方差 ∝ `hidden_dim`，大模型会 blow up。

**Xavier 初始化**：

```python
w = nn.Parameter(torch.randn(input_dim, hidden_dim) / math.sqrt(input_dim))
```

输出稳定在 ~N(0, 1)。常用变体：**truncated normal**（截断到 ±3σ），避免极端值。

### 9.2 简单模型示例

```python
class Cruncher(nn.Module):
    def __init__(self, dim, num_layers):
        self.layers = nn.ModuleList([Linear(dim, dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
```

参数量 = `num_layers × d² + d`

### 9.3 随机性管理

随机性来源：初始化、dropout、数据顺序。

最佳实践：
- **固定 random seed** 以保证可复现
- 不同随机源用不同 seed（可以固定初始化但变化数据顺序）
- **Determinism is your friend when debugging.**

---

## 十、数据加载

### Memory-Mapped Files

语言模型数据 = 整数序列（tokenizer 输出），序列化为 numpy 数组。

问题：数据可能巨大（如 Llama 数据集 2.8 TB），不能全部加载到内存。

解决方案：`np.memmap`——将文件映射到内存地址空间，按需加载：

```python
data = np.memmap("tokens.npy", dtype=np.uint16, mode="r")
# 访问 data[i:j] 时才真正从磁盘读取
```

---

## 十一、Optimizer

### 11.1 优化器演进

| 优化器   | 年份 | 核心思想                                 |
| -------- | ---- | ---------------------------------------- |
| SGD      | —    | 沿梯度方向更新                           |
| Momentum | —    | 梯度的指数移动平均                       |
| Adagrad  | —    | 按梯度平方和缩放学习率                   |
| RMSProp  | —    | Adagrad + 指数平均（替代累积平均）       |
| **Adam** | 2014 | RMSProp + Momentum，维护一阶和二阶矩估计 |

> Assignment 1 要求实现 Adam。

### 11.2 PyTorch 中实现 Optimizer

继承 `torch.optim.Optimizer`，核心在 `step()` 方法：

```python
def step(self):
    for group in self.param_groups:
        for p in group["params"]:
            grad = p.grad
            state = self.state[p]
            # 更新 state（如 Adagrad 的梯度平方和）
            state["grad_sq"] += grad ** 2
            # 更新参数
            p.data -= lr * grad / (state["grad_sq"].sqrt() + eps)
```

`self.state` 是跨 step 持久化的字典，存储 optimizer state。

---

## 十二、完整 Memory Budget

以两层线性模型为例，总内存 = 4 bytes × (参数 + activations + 梯度 + optimizer state)：

| 组成部分        | 大小                              | 说明                         |
| --------------- | --------------------------------- | ---------------------------- |
| 参数            | d² × num_layers + d               | 模型权重                     |
| Activations     | B × d × num_layers                | 前向传播中间结果（反向需要） |
| 梯度            | = 参数量                          | 每个参数一个梯度             |
| Optimizer state | 取决于优化器（Adagrad: 1×参数量） | Adam: 2×参数量               |

> 对于 Adam/AdamW：每参数 = 4（param）+ 4（grad）+ 4（m）+ 4（v）= **16 bytes**（FP32）。

### 为什么要存 Activations？

Backward pass 中，第 i 层的梯度依赖第 i 层的 activation。

> 优化技巧：**Activation Checkpointing**——不存所有 activations，需要时重新计算。后续课程会讲。

---

## 十三、训练循环 & Checkpointing

### 标准训练循环

```python
model = Cruncher(dim, num_layers).cuda()
optimizer = Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    x, y = batch
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Checkpointing（训练检查点）

训练时间长 + 必然会崩溃 → 定期保存：

需要保存的内容：
- **模型参数**
- **Optimizer state**
- **当前迭代数**

```python
torch.save({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step}, "checkpoint.pt")
```

---

## 十四、Mixed Precision Training

### 核心思想

不同阶段使用不同精度，在速度和稳定性之间取平衡。

| 阶段              | 精度 |
| ----------------- | ---- |
| 参数存储          | FP32 |
| Optimizer state   | FP32 |
| Forward pass 计算 | BF16 |
| MatMul            | BF16 |
| 梯度累积          | FP32 |

> 参数以 FP32 存储，前向时 cast 到 BF16 计算，梯度累积回 FP32。"BF16 是 transitory 的——跑一趟就扔；长期积累的东西要用高精度。"

### PyTorch 支持

PyTorch 提供 `torch.cuda.amp` 自动混合精度工具，避免手动指定每层精度。

### 前沿探索

- 有论文证明全程 FP8 训练可行，但需要额外的数值稳定技巧
- **训练**对精度敏感，**推理**可以激进量化（INT4 等）
- 模型架构设计正在与硬件特性协同演化

---

## 十五、总结

本讲从底层到顶层构建了完整的 resource accounting 框架：

```
Tensor（存储单元）
  → 浮点类型（FP32 / BF16 / FP8）→ 内存计量
  → 操作（view / elementwise / matmul）→ 计算计量
    → Forward: 2 × tokens × params
    → Backward: 4 × tokens × params
    → Total: 6 × tokens × params
  → 模型 = 参数 + activations + 梯度 + optimizer state → 总内存
  → MFU = 实际有效 FLOP/s / 标称 FLOP/s → 效率度量
```

### 关键公式速查

| 公式                                 | 含义                     |
| ------------------------------------ | ------------------------ |
| `mem = numel × dtype_bytes`          | Tensor 内存              |
| `matmul_flops = 2 × M × N × K`       | 矩阵乘法计算量           |
| `forward_flops ≈ 2 × B × params`     | 前向 FLOPs               |
| `backward_flops ≈ 4 × B × params`    | 反向 FLOPs               |
| `total_flops ≈ 6 × B × params`       | 训练总 FLOPs             |
| `MFU = model_flops/s / peak_flops/s` | 硬件利用率               |
| `Adam: 16 bytes/param`               | AdamW 每参数内存（FP32） |

---

## 十六、下一讲预告

Tatu 将讲解 **Transformer 架构**的概念总览。Assignment 1 的 handout 中也有详细的数学描述和图示。
