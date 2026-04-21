# CS336 Lecture 2：PyTorch & Resource Accounting 笔记

> Stanford CS336 - Language Modeling from Scratch (Spring 2025)
> 授课：Percy Liang
>
> 配套讲义：`lecture_02.py`（可执行的 live coding 讲义，`main()` 定义了整堂课的线性讲解流程）

---

## 一、开场：Napkin Math 的重要性

本讲从两个"餐巾纸计算"问题切入，展示 resource accounting 的核心思维。

Percy 强调：你可能不习惯做这种计算——以前只是实现模型、训练、看结果。但效率是这门课的核心，当数字变大时，FLOPs 直接等于美元。

### 问题 1：训练 70B 模型需要多久？

> 70B 参数 dense transformer，15T tokens，1024 张 H100。

推导过程（对应 `motivating_questions()`）：

```python
total_flops = 6 * 70e9 * 15e12                          # = 6.3e24
# H100 标称 1979 TFLOP/s（含稀疏），dense 取一半
assert h100_flop_per_sec == 1979e12 / 2                  # ≈ 990 TFLOP/s
mfu = 0.5
flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24
days = total_flops / flops_per_day                       # ≈ 144 天
```

步骤拆解：

1. **总 FLOPs** = 6 × 参数量 × token 数。其中 `6` 的来源 = forward(2) + backward(4)，本讲后半部分推导
2. **单卡有效 FLOP/s** = 标称 FLOP/s × MFU。标称值从 [Nvidia spec sheet](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet) 查到，MFU 取 0.5（经验值，后面会讲）
3. **集群每日 FLOPs** = 1024 卡 × 单卡有效 FLOP/s × 86400s
4. 总天数 = 总 FLOPs / 每日 FLOPs ≈ **144 天**

### 问题 2：8 张 H100 最大能训多大的模型？

```python
h100_bytes = 80e9                           # 80 GB HBM per card
bytes_per_parameter = 4 + 4 + (4 + 4)      # = 16 bytes（下面拆解）
num_parameters = (h100_bytes * 8) / bytes_per_parameter  # ≈ 40B
```

16 bytes/param 的拆解（使用 AdamW + FP32）：

| 组成          | 大小         |
| ------------- | ------------ |
| 参数 w        | 4 bytes      |
| 梯度 ∂L/∂w    | 4 bytes      |
| Adam 一阶矩 m | 4 bytes      |
| Adam 二阶矩 v | 4 bytes      |
| **合计**      | **16 bytes** |

**两个 Caveat**（Percy 特别提到）：

1. **Caveat 1**：可以用 BF16 存参数和梯度（2+2 bytes），但需额外保留一份 FP32 参数副本（4 bytes）用于精确更新。总计仍是 16 bytes，不省内存但计算更快。参见 [ZeRO 论文](https://arxiv.org/abs/1910.02054)
2. **Caveat 2**：未计入 **activations**（取决于 batch size 和 sequence length），这在 Assignment 1 中会很重要

---

## 二、本讲知识类型与结构

| 类型       | 内容                                         |
| ---------- | -------------------------------------------- |
| Mechanics  | PyTorch 底层原语：tensor、view、einops 等    |
| Mindset    | Resource accounting：内存 & 算力精确计量     |
| Intuitions | 本讲较少涉及——没有大模型，只有 broad strokes |

> 本讲**不讲 Transformer 架构**（下一讲由 Tatu 讲概念总览），而是用简单线性模型讲清原语和资源计量。

学习 Transformer 的推荐资源：
- [Assignment 1 handout](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf)（做完就一定懂了）
- [Mathematical description](https://johnthickstun.com/docs/transformers.pdf)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

**本讲自底向上的结构**：

```
Tensor → 浮点类型 → 操作（view / elementwise / matmul）
  → einops → FLOPs 计量 → 梯度 FLOPs
    → 模型构建 → 优化器 → 训练循环 → checkpointing → mixed precision
```

---

## 三、Memory Accounting

### 3.1 Tensor 基础

> 对应 `tensors_basics()`

Tensor 是深度学习中存储一切的基本单元：参数、梯度、optimizer state、数据、activations——它们是"原子"。

创建方式：

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # 从已有数据创建
x = torch.zeros(4, 8)                       # 全零
x = torch.ones(4, 8)                        # 全一
x = torch.randn(4, 8)                       # iid N(0,1) 采样
x = torch.empty(4, 8)                       # 分配内存但不初始化
```

`torch.empty` 的用途：你想用自定义逻辑设置值（如截断正态），所以先分配空间，再用 in-place 操作填入：

```python
nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)  # 截断到 [-2, 2]
```

### 3.2 Tensor 内存计算

> 对应 `tensors_memory()`

**内存公式**：

```
内存(bytes) = 元素数量 × 每个元素的字节数
```

```python
x = torch.zeros(4, 8)
assert x.dtype == torch.float32     # PyTorch 默认类型
assert x.numel() == 32              # 4 × 8 = 32 个元素
assert x.element_size() == 4        # float32 = 4 bytes/element
# 总内存 = 32 × 4 = 128 bytes
```

**直觉参考**——GPT-3 中 FFN 层的一个权重矩阵有多大？

```python
# FFN 层：hidden_dim=12288, intermediate=12288*4=49152
assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # = 2.3 GB
```

> 一个矩阵就 2.3 GB。

### 3.3 浮点数类型详解

这是理解 mixed precision training 的基础。Percy 花了大量时间讲解不同格式的取舍。

#### FP32（float32）

![fp32](images/fp32.png)

- 32 位 = 1 符号 + 8 指数 + 23 尾数
- 也叫 single precision，在 ML 中被称为 "full precision"
- 科学计算的人会嘲笑你说 float32 是 "full"——他们用 float64。但对 DL 来说，float32 已经是你需要的最高精度了，因为 **deep learning is kind of sloppy like that**

可以用 `torch.finfo` 查看各类型的详细规格：

```python
torch.finfo(torch.float32)
# finfo(resolution=1e-06, min=-3.40282e+38, max=3.40282e+38, eps=1.19209e-07, ...)
```

#### FP16（float16）

![fp16](images/fp16.png)

- 16 位 = 1 符号 + 5 指数 + 10 尾数
- 内存减半，但**动态范围严重不足**：

```python
x = torch.tensor([1e-8], dtype=torch.float16)
assert x == 0  # Underflow! 直接变成 0
```

- Percy 说："小模型可能没事，但大模型中大量矩阵运算会导致 instability、underflow、overflow，bad things happen。"
- **结论：不推荐使用 FP16，BF16 在所有场景下都是更好的选择。**

#### BF16（bfloat16）

![bf16](images/bf16.png)

- 16 位 = 1 符号 + **8 指数**（与 FP32 相同！）+ 7 尾数
- 2018 年 Google Brain 开发，名字中的 "b" = brain
- **核心设计思想**：对 DL 来说，动态范围比分辨率更重要。BF16 把更多位分给指数，牺牲尾数精度
- 与 FP16 相同内存，但**动态范围 = FP32**：

```python
x = torch.tensor([1e-8], dtype=torch.bfloat16)
assert x != 0  # 不会 underflow ✓
```

三种类型的动态范围对比：

```python
torch.finfo(torch.float32)   # min=-3.40282e+38, max=3.40282e+38
torch.finfo(torch.float16)   # min=-65504,       max=65504          ← 范围小得多
torch.finfo(torch.bfloat16)  # min=-3.38953e+38, max=3.38953e+38   ← 几乎 = FP32
```

Percy 的精辟总结："**BF16 是 transitory 的——你把参数 cast 到 BF16，跑一趟前向就扔掉；但长期积累的东西（参数、optimizer state）要用高精度。**"

#### FP8

- 2022 年标准化（[论文](https://arxiv.org/pdf/2209.05433.pdf)），由 Nvidia 推动，专为 ML 设计
- H100 支持两种变体（[Nvidia FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)）：
  - **E4M3**：4 位指数 + 3 位尾数，范围 [-448, 448]
  - **E5M2**：5 位指数 + 2 位尾数，范围 [-57344, 57344]
- Percy 说："你看 FP8，真的没几个 bit 来存东西，非常粗糙。"
- 仅 H100 及以上支持，前代 GPU 不可用

#### 精度选择实践总结

| 用途                  | 推荐精度      | 原因                     |
| --------------------- | ------------- | ------------------------ |
| 参数存储              | FP32          | 长期积累，需要精度       |
| Optimizer state       | FP32          | 同上                     |
| 前向/反向 MatMul      | BF16          | 速度快，动态范围够       |
| 梯度累积              | FP32          | 多步累积需要精度         |
| Attention（保守做法） | FP32          | 确保 softmax 数值稳定    |
| 推理 / 量化           | FP8 甚至 INT4 | 训练完的模型对量化更鲁棒 |

> 一般原则：**需要长期积累的用 FP32，一次性计算的用 BF16**。

---

## 四、Compute Accounting

### 4.1 GPU 与数据位置

> 对应 `tensors_on_gpus()`

PyTorch 默认将 tensor 放在 CPU 上。**你必须时刻清楚每个 tensor 在哪里——光看代码看不出来。**

```python
x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")     # 默认在 CPU
```

CPU 和 GPU 是两块独立的内存，数据需要显式搬运：

![cpu-gpu](images/cpu-gpu.png)

```python
# 方式 1：先创建在 CPU，再搬到 GPU（有数据传输开销）
y = x.to("cuda:0")
assert y.device == torch.device("cuda", 0)

# 方式 2：直接在 GPU 上创建（推荐，避免传输）
z = torch.zeros(32, 32, device="cuda:0")
```

可以用 `torch.cuda.memory_allocated()` 精确验证 GPU 内存分配：

```python
memory_before = torch.cuda.memory_allocated()
y = x.to("cuda:0")                              # 搬一个 32×32 float32 到 GPU
z = torch.zeros(32, 32, device="cuda:0")         # 在 GPU 上新建一个
memory_after = torch.cuda.memory_allocated()
assert memory_after - memory_before == 2 * (32 * 32 * 4)  # 恰好 2 个矩阵的大小
```

> 建议：在代码关键位置用 `assert` 检查 tensor 的 device，作为文档和安全保障。

### 4.2 Tensor 的内部表示：Storage + Stride

> 对应 `tensor_storage()`

理解 stride 是理解 view 操作（零拷贝）的前提。

**PyTorch 的 tensor 不是一个独立的数据块，而是一个指针 + 元数据**，指向底层的一段连续内存（storage）。元数据中最重要的就是 **stride**——告诉你沿每个维度移动一步需要在 storage 中跳过多少个元素。

![stride](https://martinlwx.github.io/img/2D_tensor_strides.png)

以 4×4 矩阵为例：

```python
x = torch.tensor([
    [0., 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
])
# 底层存储是一维的: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

assert x.stride(0) == 4   # 沿 dim 0（行方向）：跳 4 个元素到下一行
assert x.stride(1) == 1   # 沿 dim 1（列方向）：跳 1 个元素到下一列
```

**寻址公式**：`x[r, c]` 在 storage 中的偏移 = `r × stride(0) + c × stride(1)`

```python
r, c = 1, 2
index = r * x.stride(0) + c * x.stride(1)  # = 1*4 + 2*1 = 6
assert index == 6  # storage[6] = 6 ✓
```

### 4.3 View 操作：零拷贝（重点）

> 对应 `tensor_slicing()`

多个 tensor 可以共享同一块 storage，只是 stride/offset 不同。这就是 **view**——不拷贝数据，只改变"看数据的方式"。

Percy 说："Views are free, so feel free to use them to make your code easier to read — they don't allocate any memory."

可以用 `same_storage(x, y)` 验证两个 tensor 是否共享底层存储：

```python
def same_storage(x, y):
    return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
```

**各种 view 操作演示**：

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])

# 取行 → view（不拷贝）
y = x[0]                           # tensor([1., 2., 3.])
assert same_storage(x, y)          # ✓ 共享 storage

# 取列 → view（不拷贝）
y = x[:, 1]                        # tensor([2., 5.])
assert same_storage(x, y)          # ✓

# reshape → view（不拷贝）
y = x.view(3, 2)                   # tensor([[1,2],[3,4],[5,6]])
assert same_storage(x, y)          # ✓

# 转置 → view（不拷贝，但变为 non-contiguous）
y = x.transpose(1, 0)              # tensor([[1,4],[2,5],[3,6]])
assert same_storage(x, y)          # ✓
```

**关键陷阱：view 共享 storage，修改一个会影响另一个**：

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])
y = x[0]           # y 是 x 第一行的 view
x[0][0] = 100      # 修改 x
assert y[0] == 100  # y 也被修改了！因为指向同一块内存
```

**Contiguous 与 Non-contiguous**：

如果遍历 tensor 时在底层 storage 中是连续滑动的，就是 contiguous。转置后变成 non-contiguous（因为遍历时在 storage 中"跳来跳去"）：

```python
x = torch.tensor([[1., 2, 3], [4, 5, 6]])
y = x.transpose(1, 0)
assert not y.is_contiguous()

# 对 non-contiguous tensor 做 view 会报错：
try:
    y.view(2, 3)
except RuntimeError as e:
    # "view size is not compatible with input tensor's size and stride"
    pass

# 解决：先 .contiguous()（触发拷贝），再 view
y = x.transpose(1, 0).contiguous().view(2, 3)
assert not same_storage(x, y)  # 不再共享 storage，因为 contiguous() 做了拷贝
```

**总结**：

| 操作            | 是否拷贝 | 说明                                |
| --------------- | -------- | ----------------------------------- |
| `x[0]`          | ❌        | 取行，view                          |
| `x[:, 1]`       | ❌        | 取列，view                          |
| `.view(3, 2)`   | ❌        | reshape，view                       |
| `.transpose()`  | ❌        | 转置，view（但变为 non-contiguous） |
| `.contiguous()` | ✅        | 非连续 → 连续，**会拷贝**           |
| `.reshape()`    | 可能     | 等价于 `.contiguous().view()`       |

> **Views are free, copying takes both additional memory and compute.**

### 4.4 Element-wise 操作

> 对应 `tensor_elementwise()`

这些操作对每个元素独立运算，**都会分配新内存**（因为需要存放新值）：

```python
x = torch.tensor([1, 4, 9])
assert torch.equal(x.pow(2),  torch.tensor([1, 16, 81]))
assert torch.equal(x.sqrt(),  torch.tensor([1, 2, 3]))
assert torch.equal(x.rsqrt(), torch.tensor([1, 1/2, 1/3]))  # 1/sqrt(x)

assert torch.equal(x + x, torch.tensor([2, 8, 18]))
assert torch.equal(x * 2, torch.tensor([2, 8, 18]))
```

**`torch.triu`**（上三角）在构造 causal attention mask 时很有用：

```python
x = torch.ones(3, 3).triu()
# tensor([[1, 1, 1],
#         [0, 1, 1],
#         [0, 0, 1]])
```

> `M[i,j] = 1` 表示位置 j 对位置 i 可见。上三角 = causal mask（每个位置只能看到自己和之前的位置）。Assignment 1 中构造 attention mask 会用到。

---

## 五、矩阵乘法：深度学习的核心运算

> 对应 `tensor_matmul()`

### 5.1 基本 MatMul

```python
x = torch.ones(16, 32)
w = torch.ones(32, 2)
y = x @ w
assert y.size() == torch.Size([16, 2])
```

### 5.2 Batched MatMul

实际中所有操作都是 batched 的。对语言模型来说，维度通常是 `(batch, sequence, hidden)`：

![batch-sequence](images/batch-sequence.png)

```python
x = torch.ones(4, 8, 16, 32)  # (batch=4, seq=8, tokens=16, hidden=32)
w = torch.ones(32, 2)
y = x @ w
assert y.size() == torch.Size([4, 8, 16, 2])
# PyTorch 自动对前面的维度做 broadcast：
# 对每个 batch、每个 sequence position 都做 (16,32) @ (32,2) = (16,2)
```

---

## 六、Einops & Einstein Summation

> 对应 `tensor_einops()` → `einops_motivation()` → `jaxtyping_basics()` → `einops_einsum()` → `einops_reduce()` → `einops_rearrange()`

### 6.1 动机：传统写法可读性差

```python
x = torch.ones(2, 2, 3)  # batch, sequence, hidden
y = torch.ones(2, 2, 3)
z = x @ y.transpose(-2, -1)  # -2 是啥？-1 是啥？
```

Percy 说："你看到 -1, -2 就得想半天，如果你很勤快会写注释，但注释会和代码 out of date，然后 debug 就很痛苦。"

### 6.2 Jaxtyping：给维度命名的文档方式

```python
# 旧写法（靠注释）：
x = torch.ones(2, 2, 1, 3)  # batch seq heads hidden

# 新写法（jaxtyping，维度名写在类型标注里）：
x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)
```

> 注意：这只是文档，**默认不做运行时检查**。可以用 checker 强制，但 PyTorch 类型系统本身不强制。

### 6.3 Einsum：带命名维度的广义矩阵乘法

Einsum = "generalized matrix multiplication with good bookkeeping"。

```python
x: Float[Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
y: Float[Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)

# 旧写法：
z = x @ y.transpose(-2, -1)  # (2, 3, 3)

# 新写法（einops）：
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

**规则**：
- 输出中**出现**的维度 → 保留（遍历）
- 输出中**不出现**的维度 → 求和（contracted）
- 上例中 `hidden` 不在输出中 → 对 hidden 维度求和（这就是矩阵乘法的本质）

可以用 `...` 表示任意数量的 broadcast 维度：

```python
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
# 这样无论前面有几个维度（batch, batch1 batch2, ...）都能处理
```

> 学生提问："einsum 能保证编译成高效实现吗？" 回答：**是的**。`torch.compile` 会找到最优的维度 reduce 顺序，只做一次，然后复用，"better than anything designed by hand"。

### 6.4 Reduce：对单个 tensor 聚合

```python
x: Float[Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)

# 旧写法：
y = x.sum(dim=-1)  # (2, 3)

# 新写法（einops）：
y = reduce(x, "... hidden -> ...", "sum")  # (2, 3)
# hidden 消失 → 对 hidden 维度求和
```

### 6.5 Rearrange：拆分/合并维度（Multi-Head Attention 的关键操作）

典型场景：multi-head attention 中需要把 `total_hidden` 拆成 `heads × hidden_per_head`，对每个 head 独立做变换，再合并回去。

**完整 3 步流程**：

```python
# 初始：x 的最后一维 total_hidden=8 实际上是 heads=2 × hidden1=4 的 flatten
x: Float[Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
w: Float[Tensor, "hidden1 hidden2"] = torch.ones(4, 4)

# Step 1: 拆分 total_hidden → heads × hidden1
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
# x.shape = (2, 3, 2, 4)
# 必须指定 heads=2，因为 8 可以拆成 (2,4) 或 (4,2) 等多种方式

# Step 2: 对 hidden1 维度做线性变换
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
# x.shape = (2, 3, 2, 4)  — hidden1 被 contracted，hidden2 出现在输出中

# Step 3: 合并 heads × hidden2 → total_hidden
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
# x.shape = (2, 3, 8)  — 回到原始形状
```

> Percy 推荐："投入时间学习 einops，Assignment 1 中会用到。你不是必须用它（从零构建嘛），但我们给了 guidance，值得投资。"
>
> 教程：[einops basics](https://einops.rocks/1-einops-basics/)

---

## 七、FLOPs 计量

> 对应 `tensor_operations_flops()`

### 7.1 定义澄清（Percy 的 pet peeve）

| 写法            | 含义                                         |
| --------------- | -------------------------------------------- |
| FLOPs（小写 s） | floating point operations（计算量，一个数）  |
| FLOP/s          | floating point operations per second（速度） |

> Percy 说："这是我的 pet peeve。大写 FLOPS 太模糊了——到底是计算量还是速度？本课程统一用 `/s` 表示速度。"

> 加法和乘法被视为等价的 1 个 FLOP。

### 7.2 规模直觉

| 事件                            | FLOPs      |
| ------------------------------- | ---------- |
| GPT-3 训练（2020）              | ~3.14×10²³ |
| GPT-4 训练（2023，推测）        | ~2×10²⁵    |
| 美国行政令阈值（2025 年已撤销） | ≥10²⁶      |
| EU AI Act 阈值（仍有效）        | ≥10²⁵      |

**8 张 H100 跑一周**能做多少计算？

```python
total_flops = 8 * (60 * 60 * 24 * 7) * h100_flop_per_sec  # ≈ 4.7e21 FLOPs
```

### 7.3 硬件 FLOP/s

| GPU  | FP32 FLOP/s | BF16 FLOP/s（dense） | BF16（含 2:4 稀疏） |
| ---- | ----------- | -------------------- | ------------------- |
| A100 | 19.5 T      | 312 T                | —                   |
| H100 | 67.5 T      | ~990 T               | ~1979 T             |

Percy 特别指出几个关键点：

1. **FP32 在 H100 上极慢**：67.5T vs 990T，差 15 倍。如果你用 FP32 跑 MatMul 而不是 BF16，你在浪费硬件
2. **稀疏性是营销手段**：spec sheet 上的数字带星号（*），指 2:4 structured sparsity（每 4 个元素中 2 个为零）。实际 dense 矩阵取一半。Percy 原话："No one uses it. The marketing department uses it."
3. **FLOP/s 强烈依赖数据类型**：同一块 GPU，FP32 和 BF16 的速度差一个数量级

### 7.4 MatMul 的 FLOPs 公式

**核心公式**（必须记住）：

```
MatMul FLOPs = 2 × M × N × K
```

其中 `(M×K) @ (K×N) → (M×N)`。

**推导**：输出矩阵有 M×N 个元素。每个元素 `y[i,j] = Σ_k x[i,k] × w[k,j]`，需要 K 次乘法 + K 次加法 = 2K 次操作。总计 2 × M × N × K。

```python
# 讲义中的具体例子
B, D, K = 16384, 32768, 8192
x = torch.ones(B, D, device="cuda")
w = torch.randn(D, K, device="cuda")
y = x @ w
actual_num_flops = 2 * B * D * K  # = 8.8e12
```

> Percy 说："这个数字你应该记住——矩阵乘法的 FLOPs = 2 × 三个维度的乘积。"

> 学生提问："实际的矩阵乘法算法（如 Strassen）会改变这个数吗？" Percy 回答：底层有大量优化（下下周讲 kernels 时会看到），但 2MNK 作为 crude estimate 是正确的数量级。

### 7.5 为什么只需要数 MatMul

- Element-wise 操作的 FLOPs 是 O(mn)，线性的
- MatMul 的 FLOPs 是 O(mnk)，三次方级别
- **对于足够大的矩阵，MatMul 绝对主导**

> Percy 说："这就是为什么 napkin math 很简单——我们只看模型中的矩阵乘法。当然如果矩阵很小，其他操作的开销会开始显现，但那不是你想要的 regime，因为硬件就是为大矩阵乘法设计的。这有点 circular——但我们就是会处于 MatMul 主导的 regime。"

### 7.6 线性模型的 FLOPs 语义解读

对于 `Y = X @ W`，其中 X 是 `(B, D)`，W 是 `(D, K)`：

```
Forward FLOPs = 2 × B × D × K = 2 × (数据点数) × (参数量)
```

Percy 给出了一个重要的语义解读：
- B = 数据点数（tokens）
- D × K = 参数量
- **Forward FLOPs = 2 × tokens × params**

> "这个关系对 Transformer 也近似成立（to a first-order approximation）。有个星号——sequence length 和其他因素会影响，但大致是对的。"

---

## 八、MFU（Model FLOPs Utilization）

### 8.1 从 FLOPs 到实际时间

FLOPs 是理论计算量，但你真正关心的是**墙钟时间**（wall-clock time）。怎么从一个到另一个？

讲义中的 `time_matmul` 函数展示了正确的 benchmark 方法：

```python
def time_matmul(a, b):
    torch.cuda.synchronize()  # 等待之前的 CUDA 操作完成
    def run():
        a @ b
        torch.cuda.synchronize()  # 等待本次 MatMul 完成
    num_trials = 5
    total_time = timeit.timeit(run, number=num_trials)
    return total_time / num_trials
```

> 关键：必须调用 `torch.cuda.synchronize()`。因为 CUDA 操作是异步的——`a @ b` 返回时 GPU 可能还没算完。不 sync 的话你测到的是 CPU 发指令的时间，不是 GPU 实际计算的时间。

### 8.2 MFU 定义

```
MFU = (actual FLOP/s) / (promised FLOP/s)
    = (model_flops / actual_time) / peak_hardware_flops_per_sec
```

即：你实际榨出了硬件多少比例的算力。

### 8.3 实测示例

**FP32 MatMul on H100**（B=16384, D=32768, K=8192）：

```python
actual_time = 0.16                                       # 秒
actual_flop_per_sec = (2 * B * D * K) / actual_time      # ≈ 5.4e13
promised_flop_per_sec = 67.5e12                          # H100 FP32 标称
mfu = actual_flop_per_sec / promised_flop_per_sec        # ≈ 0.8
```

**BF16 MatMul on H100**（同样的矩阵大小）：

```python
bf16_actual_time = 0.03                                  # 比 FP32 快约 5 倍
bf16_actual_flop_per_sec = (2 * B * D * K) / 0.03       # 更高
bf16_promised_flop_per_sec = 1979e12 / 2                 # 标称值也高得多
bf16_mfu = bf16_actual_flop_per_sec / bf16_promised_flop_per_sec  # 反而更低
```

> Percy 说："BF16 的 MFU 反而更低，可能是因为 promised FLOP/s 有点乐观。所以 **always benchmark your code, don't just assume you're going to get certain levels of performance.**"

### 8.4 经验法则

| MFU   | 评价                      |
| ----- | ------------------------- |
| > 0.5 | 不错                      |
| ~0.05 | 很差，硬件严重浪费        |
| ~0.8+ | 优秀（通常只有纯 MatMul） |

### 8.5 MFU vs HFU（Hardware FLOPs Utilization）

Percy 解释了一个微妙但重要的区别：

- **MFU** 看的是"模型需要多少有效计算"——如果你用了 activation checkpointing 重复计算了一些东西，MFU 不会因此降低，因为模型本身的复杂度没变
- **HFU** 看的是硬件实际执行了多少 FLOPs
- MFU 是更公平的度量：**你不应该因为用了聪明的优化技巧而被惩罚**

> 学生提问："PyTorch 默认会用 tensor cores 吗？" 回答：是的，spec sheet 上的数字都是 tensor core 的。用 `torch.compile` 会生成正确使用 tensor core 的代码。

---

## 九、Backward Pass 的 FLOPs（推导 6N 法则）

### 9.1 简单梯度示例

> 对应 `gradients_basics()`

```python
x = torch.tensor([1., 2, 3])
w = torch.tensor([1., 1, 1], requires_grad=True)  # 只有 requires_grad=True 的才会算梯度
pred_y = x @ w                    # = 1+2+3 = 6
loss = 0.5 * (pred_y - 5).pow(2)  # = 0.5 * 1 = 0.5

loss.backward()

# 只有 w 有梯度（因为只有 w 设了 requires_grad=True）
assert loss.grad is None      # loss 本身没有梯度
assert pred_y.grad is None    # 中间变量默认不保留梯度
assert x.grad is None         # x 没有 requires_grad
assert torch.equal(w.grad, torch.tensor([1., 2., 3.]))  # ∂L/∂w = (pred_y - 5) * x = 1 * [1,2,3]
```

> 学生提问："Assignment 里需要手动算梯度吗？" Percy 回答：不需要，直接用 PyTorch autograd。这里手动推导只是为了数 FLOPs。

### 9.2 两层线性网络

> 对应 `gradients_flops()`

模型：

```
X --[W1]--> H1 --[W2]--> H2 --> Loss = mean(H2²)
(B,D)  (D,D)  (B,D)  (D,K)  (B,K)
```

```python
h1 = x @ w1       # (B,D) @ (D,D) → (B,D)
h2 = h1 @ w2      # (B,D) @ (D,K) → (B,K)
loss = h2.pow(2).mean()
```

**Forward FLOPs**：
- `X @ W1`：2 × B × D × D
- `H1 @ W2`：2 × B × D × K
- 合计 = 2 × B × (D² + D×K) = **2 × B × 总参数量**

### 9.3 Backward Pass 详细推导

为了在 backward 中检查中间变量的梯度，需要调用 `.retain_grad()`：

```python
h1.retain_grad()  # 默认中间变量的梯度会被释放，retain_grad() 让它保留
h2.retain_grad()
loss.backward()
# 现在可以访问 h1.grad, h2.grad, w1.grad, w2.grad
```

**以 W2 层为例**，需要计算**两个**梯度：

**① 参数梯度 ∂L/∂W2**（用于更新 W2）：

```
w2.grad[j,k] = Σ_i h1[i,j] × h2.grad[i,k]
```

维度分析：h1 是 (B,D)，h2.grad 是 (B,K)，对 i（即 B 维度）求和 → 结果是 (D,K) = W2 的形状 ✓

这本质上是 `H1ᵀ @ H2.grad`，FLOPs = 2 × B × D × K

**② 输入梯度 ∂L/∂H1**（用于继续反传到 W1）：

```
h1.grad[i,j] = Σ_k w2[j,k] × h2.grad[i,k]
```

维度分析：w2 是 (D,K)，h2.grad 是 (B,K)，对 k 求和 → 结果是 (B,D) = H1 的形状 ✓

这本质上是 `H2.grad @ W2ᵀ`，FLOPs = 2 × B × D × K

**W2 层 backward 总计** = 2 × (2 × B × D × K) = 4 × B × D × K = **2 × forward of this layer**

W1 层同理：backward = 4 × B × D × D

> 可视化参考：[The FLOPs Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)（有动画展示 forward/backward 中每个 MatMul 的流向）

### 9.4 总结：6N 法则

```python
num_forward_flops  = (2 * B * D * D) + (2 * B * D * K)      # = 2 × B × params
num_backward_flops = (4 * B * D * D) + (4 * B * D * K)      # = 4 × B × params（每层 2 个 MatMul）
total_flops        = num_forward_flops + num_backward_flops  # = 6 × B × params
```

| 阶段     | FLOPs              | 解释                                    |
| -------- | ------------------ | --------------------------------------- |
| Forward  | 2 × B × params     | 每层 1 个 MatMul                        |
| Backward | 4 × B × params     | 每层 2 个 MatMul（参数梯度 + 输入梯度） |
| **总计** | **6 × B × params** | Backward = 2 × Forward                  |

> 这就是开头 `6 × N × D` 的来源。Percy 说："对大多数模型这基本上就是计算的主体——只要每次计算都 touch 新的参数（没有 parameter sharing），这个近似就成立。如果你有 parameter sharing（比如一个参数用了一百万次），那就不适用了，但一般模型不是这样的。"

---

## 十、模型构建

### 10.1 nn.Parameter

> 对应 `module_parameters()`

```python
w = nn.Parameter(torch.randn(input_dim, output_dim))
assert isinstance(w, torch.Tensor)    # 行为像 tensor
assert type(w.data) == torch.Tensor   # 底层是普通 tensor
```

`nn.Parameter` 就是一个 tensor，但它告诉 PyTorch "这是模型参数，需要计算梯度、需要被 optimizer 管理"。

### 10.2 参数初始化（为什么重要）

**朴素初始化的问题**：

```python
input_dim = 16384
output_dim = 32
w = nn.Parameter(torch.randn(input_dim, output_dim))  # 每个元素 ~ N(0,1)
x = nn.Parameter(torch.randn(input_dim))               # 输入也是 N(0,1)
output = x @ w
```

`output` 的每个元素 = Σ x[i] × w[i,j]，是 input_dim 个 N(0,1) 随机变量的乘积之和。根据中心极限定理，其标准差 ∝ √input_dim。

当 input_dim = 16384 时，输出值 ≈ ±128。**大模型会 blow up，梯度也会爆炸，训练不稳定。**

**Xavier 初始化**：除以 √input_dim 使输出方差稳定在 ~1

```python
w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
output = x @ w  # 每个元素 ≈ N(0, 1)，不随 input_dim 变化
```

> 参考：[Xavier initialization 原始论文](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

**Truncated Normal**：更安全，截断到 ±3σ 避免极端值

```python
w = nn.Parameter(nn.init.trunc_normal_(
    torch.empty(input_dim, output_dim),
    std=1/np.sqrt(input_dim), a=-3, b=3
))
```

> Percy 说："你不信任正态分布的无界尾巴，就截断到 ±3，这样不会有大值来搞乱你的训练。"

### 10.3 简单模型示例（Cruncher）

> 对应 `custom_model()` 和 `Linear` / `Cruncher` 类定义

```python
class Linear(nn.Module):
    """只有一个权重矩阵的线性层（无 bias）"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))

    def forward(self, x):
        return x @ self.weight


class Cruncher(nn.Module):
    """深度线性网络：num_layers 个 D→D 线性层 + 一个 D→1 的 head"""
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([Linear(dim, dim) for _ in range(num_layers)])
        self.final = Linear(dim, 1)

    def forward(self, x):
        B, D = x.size()
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)       # (B, 1)
        x = x.squeeze(-1)       # (B,) — 去掉最后一维
        return x
```

验证参数量：

```python
D, num_layers = 64, 2
model = Cruncher(dim=D, num_layers=num_layers)

# 查看每个参数的名字和大小
param_sizes = [(name, param.numel()) for name, param in model.state_dict().items()]
assert param_sizes == [
    ("layers.0.weight", D * D),   # 第一层：64×64 = 4096
    ("layers.1.weight", D * D),   # 第二层：64×64 = 4096
    ("final.weight", D),          # head：64×1 = 64
]
# 总参数量 = D² × num_layers + D
```

> 注意：必须用 `nn.ModuleList` 而不是普通 Python list，否则 PyTorch 不会追踪子模块的参数（`model.parameters()` 会漏掉它们）。

### 10.4 随机性管理

> 对应 `note_about_randomness()`

随机性来源：初始化、dropout、数据顺序。

```python
seed = 0
torch.manual_seed(seed)     # PyTorch
np.random.seed(seed)        # NumPy
random.seed(seed)           # Python 内置
```

Percy 的建议：
- **三个地方都要设**，just to be safe
- 不同随机源用不同 seed（可以固定初始化但变化数据顺序）
- **"Determinism is your friend when debugging."**——如果你在追一个 bug，随机性会让你抓狂

---

## 十一、数据加载

> 对应 `data_loading()` 和 `get_batch()`

### 11.1 Memory-Mapped Files

语言模型数据 = 整数序列（tokenizer 输出），序列化为 numpy 数组。

**问题**：数据可能巨大（如 Llama 数据集 **2.8 TB**），不能全部加载到内存。

**解决方案**：`np.memmap`——将文件映射到虚拟内存地址空间，**按需加载**（访问哪部分才从磁盘读哪部分）：

```python
# 序列化
orig_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
orig_data.tofile("data.npy")

# 加载（不会真正读入全部数据）
data = np.memmap("data.npy", dtype=np.int32)
assert np.array_equal(data, orig_data)  # 用起来和普通数组一样
```

### 11.2 Data Loader：采样 batch

```python
def get_batch(data, batch_size, sequence_length, device):
    # 随机采样 batch_size 个起始位置
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))

    # 从每个起始位置取 sequence_length 长度的序列
    x = torch.tensor([data[start:start + sequence_length] for start in start_indices])
    assert x.size() == torch.Size([batch_size, sequence_length])

    return x.to(device)
```

### 11.3 Pinned Memory（进阶优化）

默认情况下，CPU tensor 在 paged memory 中（可能被操作系统换出到磁盘）。可以显式 pin 到物理内存：

```python
x = x.pin_memory()                    # 固定在物理内存中
x = x.to(device, non_blocking=True)   # 异步传输到 GPU（不阻塞 CPU）
```

这允许**流水线并行**：
- GPU 处理当前 batch
- 同时 CPU 准备下一个 batch 的数据并异步传输到 GPU

> 参考：[How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)

---

## 十二、Optimizer

> 对应 `optimizer()` 和 `SGD` / `AdaGrad` 类定义

### 12.1 优化器演进

Percy 快速梳理了优化器的发展脉络：

| 优化器       | 核心思想                                            |
| ------------ | --------------------------------------------------- |
| **SGD**      | 计算 batch 梯度，沿梯度方向更新，no questions asked |
| **Momentum** | 梯度的指数移动平均（来自经典优化，Nesterov 等）     |
| **Adagrad**  | 按历史梯度平方和缩放学习率（自适应学习率）          |
| **RMSProp**  | Adagrad 的改进：用指数平均替代累积平均              |
| **Adam**     | RMSProp + Momentum，维护一阶矩 m 和二阶矩 v         |

> Adam 出现在 2014 年，本质上就是把 RMSProp 和 Momentum 结合起来。这就是为什么 Adam 要维护两个 state：梯度的 running average（m）和梯度平方的 running average（v）。

> Assignment 1 要求实现 **Adam**（不是 Adagrad）。讲义中用 Adagrad 做演示是因为它更简单。

### 12.2 PyTorch 中实现 Optimizer

讲义中给出了 SGD 和 Adagrad 的完整实现。核心模式是继承 `torch.optim.Optimizer`，实现 `step()` 方法：

**SGD**（最简单的 optimizer）：

```python
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                p.data -= lr * p.grad.data  # 就这么简单
```

**Adagrad**（有 optimizer state 的版本）：

```python
class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]       # 跨 step 持久化的字典
                grad = p.grad.data

                # 获取/初始化 optimizer state
                g2 = state.get("g2", torch.zeros_like(grad))

                # 更新 state：累积梯度平方
                g2 += torch.square(grad)
                state["g2"] = g2

                # 更新参数：学习率按梯度平方和的 sqrt 缩放
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```

关键概念：
- `self.param_groups`：参数按组管理（如不同层可以有不同学习率）
- `self.state[p]`：每个参数的 optimizer state，**跨 step 持久化**——这就是为什么 Adam 需要额外内存
- 梯度 `p.grad` 由 `loss.backward()` 计算，optimizer 只负责用它更新参数

训练后调用 `optimizer.zero_grad(set_to_none=True)` 释放梯度内存。`set_to_none=True` 比默认的设为零更省内存（直接释放 tensor 而不是填零）。

---

## 十三、完整 Memory Budget

> 对应 `optimizer()` 中的 Memory 部分

以两层线性模型（D=4, num_layers=2, B=2, Adagrad optimizer）为例，**逐项计算**：

```python
# 参数
num_parameters = (D * D * num_layers) + D  # = 4*4*2 + 4 = 36

# Activations（前向传播中间结果，反向需要）
num_activations = B * D * num_layers       # = 2*4*2 = 16

# 梯度（每个参数一个）
num_gradients = num_parameters             # = 36

# Optimizer state（Adagrad: 每个参数存一个 g²）
num_optimizer_states = num_parameters      # = 36

# 总内存（FP32 = 4 bytes/element）
total_memory = 4 * (num_parameters + num_activations + num_gradients + num_optimizer_states)
# = 4 * (36 + 16 + 36 + 36) = 4 * 124 = 496 bytes
```

**一般化**：

| 组成部分        | 数量               | Adagrad | Adam |
| --------------- | ------------------ | ------- | ---- |
| 参数            | P                  | P       | P    |
| Activations     | B × D × num_layers | —       | —    |
| 梯度            | P                  | P       | P    |
| Optimizer state | 取决于优化器       | 1×P     | 2×P  |
| **每参数总计**  |                    | 3×P     | 4×P  |

对于 **Adam/AdamW**（FP32）：每参数 = 4(param) + 4(grad) + 4(m) + 4(v) = **16 bytes**

### 为什么要存 Activations？

Percy 回答学生提问："Backward pass 中，第 i 层的梯度依赖第 i 层的 activation。比如 `w2.grad[j,k] = Σ_i h1[i,j] × h2.grad[i,k]`——你需要 h1（第一层的 activation）才能算 w2 的梯度。"

> 优化技巧：**Activation Checkpointing**——不存所有 activations，需要时重新计算（用计算换内存）。后续课程会讲。

### Compute Budget

```python
flops = 6 * B * num_parameters  # 一个 training step 的总 FLOPs
```

> Percy 说："Assignment 1 中你要对 Transformer 做同样的计算——更复杂（有 attention、多种矩阵），但思路完全一样：数参数、数 activations、数梯度、数 optimizer state。"
>
> 参考：
> - [Transformer memory usage](https://erees.dev/transformer-memory/)
> - [Transformer FLOPs](https://www.adamcasson.com/posts/transformer-flops)

---

## 十四、训练循环 & Checkpointing

### 14.1 标准训练循环

> 对应 `train_loop()` 和 `train()`

讲义中用一个简单的线性回归任务演示完整训练循环：

```python
# 生成数据：y = x @ true_w，其中 true_w = [0, 1, 2, ..., D-1]
D = 16
true_w = torch.arange(D, dtype=torch.float32, device=device)

def get_batch(B):
    x = torch.randn(B, D).to(device)
    true_y = x @ true_w
    return x, true_y
```

训练循环：

```python
model = Cruncher(dim=D, num_layers=0).to(device)  # num_layers=0 → 只有 final head
optimizer = SGD(model.parameters(), lr=0.01)

for t in range(num_train_steps):
    x, y = get_batch(B=4)                          # 1. 获取数据
    pred_y = model(x)                               # 2. 前向
    loss = F.mse_loss(pred_y, y)                    # 3. 计算 loss
    loss.backward()                                  # 4. 反向（计算梯度）
    optimizer.step()                                 # 5. 更新参数
    optimizer.zero_grad(set_to_none=True)            # 6. 清空梯度
```

> 讲义中还演示了超参数调优：同样的模型，lr=0.01 vs lr=0.1 的效果差异。

### 14.2 Checkpointing（训练检查点）

> 对应 `checkpointing()`

Percy 说："训练语言模型需要很长时间，而且**一定会崩溃**（certainly will crash）。你不想丢失所有进度。"

需要保存的内容：
- **模型参数**（`model.state_dict()`）
- **Optimizer state**（`optimizer.state_dict()`）——包含 m、v 等，丢了就得重新 warm up
- **当前迭代数**——知道从哪里继续

```python
# 保存
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    # 应该还加上 "step": current_step
}
torch.save(checkpoint, "model_checkpoint.pt")

# 加载
loaded = torch.load("model_checkpoint.pt")
model.load_state_dict(loaded["model"])
optimizer.load_state_dict(loaded["optimizer"])
```

---

## 十五、Mixed Precision Training

> 对应 `mixed_precision_training()`

### 核心问题

高精度（FP32）：准确稳定，但慢且费内存。低精度（BF16/FP8）：快且省内存，但可能不稳定。**怎么两全其美？**

### 解决方案：混合精度

不同阶段使用不同精度：

| 阶段              | 精度 | 原因                         |
| ----------------- | ---- | ---------------------------- |
| 参数存储          | FP32 | 长期积累，需要精度           |
| Optimizer state   | FP32 | 同上                         |
| Forward pass 计算 | BF16 | 一次性计算，速度优先         |
| MatMul            | BF16 | 硬件对低精度 MatMul 优化极大 |
| 梯度累积          | FP32 | 多步累积需要精度             |

> Percy 的核心原则："**BF16 是 transitory 的——跑一趟就扔；长期积累的东西要用高精度。**"

### PyTorch 支持

PyTorch 提供 `torch.cuda.amp`（Automatic Mixed Precision）库，自动决定每个操作用什么精度，避免手动指定（那会很烦，因为精度需要"横切"模型的模块化结构）。

- [PyTorch AMP 文档](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Mixed Precision Training 指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)

### 前沿探索

1. **全程 FP8 训练**：[论文](https://arxiv.org/pdf/2310.18313.pdf) 证明可行，但需要额外的数值稳定技巧（控制训练过程中的数值范围）
2. **训练 vs 推理的精度差异**：训练对精度敏感得多；推理可以激进量化（INT4 等），因为已训练好的模型对量化更鲁棒。"Somehow training is a lot more difficult to do with low precision, but once you have a trained model, it's much easier to make it low precision."
3. **模型架构与硬件协同设计**：Percy 说 "a lot of model design is just governed by hardware"——Transformer 本身就是因为 GPU 才成功的。如果能让模型在 INT4 下训练（虽然很难），速度提升会是巨大的
4. **NVIDIA Transformer Engine**：支持 FP8 线性层，进一步降低精度

> 参考论文：[Mixed Precision Training (2017)](https://arxiv.org/pdf/1710.03740.pdf)

---

## 十六、总结

本讲从底层到顶层构建了完整的 resource accounting 框架：

```
Tensor（存储单元）
  ├─ 浮点类型（FP32 / BF16 / FP8）→ 内存计量
  ├─ 操作
  │   ├─ View：零拷贝，共享 storage（free!）
  │   ├─ Elementwise：分配新内存，FLOPs = O(n)
  │   └─ MatMul：分配新内存，FLOPs = 2MNK（绝对主导）
  ├─ 梯度
  │   ├─ Forward: 2 × tokens × params
  │   ├─ Backward: 4 × tokens × params（每层 2 个 MatMul）
  │   └─ Total: 6 × tokens × params
  └─ 总内存 = 参数 + activations + 梯度 + optimizer state
      └─ MFU = 实际有效 FLOP/s / 标称 FLOP/s → 效率度量
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

## 十七、下一讲预告

Tatu 将讲解 **Transformer 架构**的概念总览。Assignment 1 的 handout 中也有详细的数学描述和图示。Percy 说："一旦你做完 Assignment 1，你一定会知道 Transformer 是什么。"
