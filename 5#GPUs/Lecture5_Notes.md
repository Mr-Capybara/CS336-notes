
# CS336 Lecture 5：GPUs 笔记

> Stanford CS336 - Language Modeling from Scratch (Spring 2025)
> 授课：Tatsu Hashimoto
>
> 配套讲义：`2025 Lecture 5 - GPUs.pdf`（51 页）
>
> **本笔记严格对应 PDF 页码，配合 PDF 阅读可无损替代视频。**

---

## 〇、开场与本讲目标（PDF p.1–p.4）

本讲目标有两个：

1. **让 CUDA 与 GPU 不再神秘**。讲师会带你理解为什么 matmul 性能随矩阵 size 呈现「奇怪的波浪形」——某些倍数特别快、某些特别慢。
2. **让你能写出快算法**，例如 FlashAttention 这样的 CUDA kernel 就是本讲终点的集成案例。

### 讲师致谢的核心参考资料

- Horace He 的博客 `thonking.ai` / `horace.io/brrr_intro.html`（多个关键图来自此处）
- CUDA Mode group
- Google 新出的 TPU 在线书（今年才有）

### 本讲三大部分

| 部分   | 内容                                                          |
| ------ | ------------------------------------------------------------- |
| Part 1 | GPU 内部结构：执行模型、内存层级                              |
| Part 2 | GPU 性能优化：Roofline、精度、Fusion、Recompute、Coalesce、Tiling |
| Part 3 | 综合应用：拆解 Flash Attention                                |

> 本讲只讨论**单卡** GPU，不涉及多卡并行（下讲 Parallelism）。

---

## 一、Part 1：GPU 深度解剖

### 1.1 Compute → Performance（PDF p.5）

Scaling law 的基本事实：**compute 越多 → 模型越好**。而 compute 来自三项：

1. 更快的硬件
2. 更好的利用率 (MFU)
3. 更好的并行

> 深度学习的革命不仅是算法，更是硬件+工程（"faster hardware, better utilization, improved parallelization alone can drive progress"）。

### 1.2 Dennard Scaling 的终结（PDF p.6）

早期 CPU 提速依赖 **Dennard Scaling**：

- 摩尔定律每年晶体管翻倍
- 晶体管越小 → 时钟越快 → 功耗反而越低 → 单线程性能直线上升

但 **1980–2000 年代后期 Dennard Scaling 枯竭**。Hennessy & Patterson 的经典图显示：晶体管数继续增长，但**单线程性能曲线拍平**。

结论：无法再靠「绝对意义上更快算」，只能靠**并行**。

### 1.3 Parallel Scaling（PDF p.7）

Bill Dally (NVIDIA) HotChips keynote 的图：从 K20 到 H100，GPU 并行算力 **10 年增长 >1000×**，是超指数曲线。

> **"There is no LLM scaling without GPU scaling."**

### 1.4 CPU vs GPU（PDF p.8）

设计哲学对立：

| 对比项   | CPU                                                      | GPU                                          |
| -------- | -------------------------------------------------------- | -------------------------------------------- |
| 线程数   | 少 (几~几十)                                             | 极多 (上万)                                  |
| 控制单元 | 大（分支预测、乱序执行、Cache）                          | 极小                                         |
| ALU 数量 | 少                                                       | 极多                                         |
| 优化目标 | **Latency**（每个任务尽快结束）                          | **Throughput**（总吞吐最大）                 |
| 类比     | 有 T1..T4 四个任务，CPU 先快速做完 T1，再做 T2... 串行化 | GPU 让 T1..T4 几乎同时在大量线程上跑到终点 |

关键点：GPU 单个任务的 latency 反而比 CPU 差，但**总吞吐**远胜。

### 1.5 GPU 执行单元解剖（PDF p.9）

两级结构：

- **SM**（Streaming Multiprocessor）= 相对独立的"工人"，一个 GPU 有很多个 SM
  - A100: **108 SMs**（注意：字幕 128 说法为口误，A100 事实是 108，后面 wave quantization 的关键数字就是 108）
- **SP**（Streaming Processor）= SM 内的线程执行单元，一个 SM 含许多 SP
- **Tensor Core** = SM 内的专用 matmul 硬件

心智模型：

- 一个 block 被派发到一个 SM 上
- SM 负责控制逻辑（例如 branch）
- SP 把同一条指令应用到大量不同数据上（SIMT）

### 1.6 GPU 内存层级（PDF p.10）

**物理距离 = 访问速度**，这是理解一切性能的基础：

| 位置          | 类型                        | 访问延迟         | 相对速度           |
| ------------- | --------------------------- | ---------------- | ------------------ |
| SM 内部       | Registers / L1 / Shared Mem (SRAM) | ~20 cycles       | 最快（基准）       |
| On-die        | L2 Cache                    | ~200–300 cycles  | 慢 ~10×            |
| Off-chip DRAM | Global Memory (HBM)         | 更慢             | 比 SRAM 慢 ~8–10×，但容量大 100× |

讲师原话："**SRAM 比 DRAM 贵 100×，但快 ~8×。**"

> 核心教训：如果算法不停访问 global memory，就会让 SM 空转。内存 = 瓶颈。

### 1.7 执行模型：Thread / Warp / Block（PDF p.11）

写 CUDA / Triton 必须理解的三层粒度（**从大到小**）：

1. **Block**：一组 threads，被整体派发到一个 SM。SM 间独立。Block 内共享 shared memory。
2. **Warp**：**32 个连续编号**的线程为一组，它们在**同一 cycle 执行同一条指令**（SIMT）。Warp 是硬件调度的基本单位。
3. **Thread**：最小工作单位，执行指令、持有自己的寄存器。

关系图：

```
GPU
 └─ SM × 108 (A100)
     └─ Block (被调度到某个 SM)
         └─ Warp (32 threads, 同步同指令)
             └─ Thread (最小执行单元)
```

**为什么是 32？** 硬件固定，warp 内线程必须「走同一条指令路径」，这直接导致 1.11 节的 **control divergence** 问题。

### 1.8 内存模型：谁能访问谁（PDF p.12）

| 存储类型       | 谁能读写                   | 速度 |
| -------------- | -------------------------- | ---- |
| Register       | 单 thread 私有             | 最快 |
| Local memory   | 单 thread (溢出到 DRAM)    | 慢   |
| Shared memory  | **同一 block 内所有线程**   | 快   |
| Global memory  | 所有 block                 | 慢   |
| Constant       | 所有线程（只读）           | 中   |

关键约束：**Block 之间只能通过 global memory 通信**。所以一旦你的算法要跨 block 交换数据，就不得不付出 global memory 的代价——**好算法应尽量让一个 block 独立完成尽可能多的工作**。

### 1.9 旁路：TPU（PDF p.13）

GPU 和 TPU 在「加速器」层面是同构的：

| GPU                          | TPU                              |
| ---------------------------- | -------------------------------- |
| SM (Streaming Multiprocessor) | TC (Tensor Core)                 |
| 许多 SM，每个功能较杂          | 较少 TC，但单 TC 算力强          |
| Tensor Core (SM 内部 matmul 单元) | MXU (专门做 matmul)          |
|                              | Scalar Unit (控制 + CPU-like 运算) |
|                              | Vector Unit (向量运算)           |
| Shared Mem / L1              | VMEM / SMEM                      |
| HBM (off-chip)               | HBM (off-chip)                   |

**核心差异**：
- TPU **没有 warp 概念**，只在 block 粒度调度（trade-off：简化、但非 matmul 工作负载不如 GPU 灵活）
- 互联方式不同（下讲 Parallelism 细讲）

核心架构哲学仍完全一致：**轻量控制 + 大块 matmul 硬件 + 快/慢两级内存**。

### 1.10 GPU 模型的优势（PDF p.14）

1. **易扩展**：想要更多算力就加更多 SM，无需提高时钟频率（避免散热灾难）
2. **编程模型简单**（相对）：SIMT —— 同一指令作用到多数据，对矩阵操作天然契合
3. **线程轻量**：可快速启停，某线程等 memory 时可休眠，让其他线程上。这直接带来**高利用率**

### 1.11 GPU 成为快 matmul 器（PDF p.15–p.16）

- 早期 NVIDIA GPU 是为 graphics 设计的，研究者 hack 纹理 buffer 来做 matmul
- **V 系列（Volta）起引入 Tensor Core**：专用 matmul 电路

PDF p.16 的关键图（TFLOPS per generation）：
- **Matmul FLOPS（橙线）在 V100 起大跳跃**，比 non-matmul FLOPS 高 >10×
- **设计任何神经网络架构时：你的工作量必须以 matmul 为主**，否则直接损失一个数量级的算力

### 1.12 Compute 增长 >> Memory 增长（PDF p.17）

三条曲线（log 尺度）：

| 曲线             | 从哪到哪             | 增长倍数    |
| ---------------- | -------------------- | ----------- |
| PCIe/NVLink 互联 | 最缓慢               | 数×量级     |
| 显存带宽 (GDDR → HBM2E) | 中等 (~100×)         | ~100×       |
| Compute (matmul FLOPS) | 极快                 | ~100,000×   |

推论：
- 早期瓶颈是 **FLOPS**（算不过来）
- 现在瓶颈是 **内存带宽**（喂不饱 compute）
- 未来 DRAM 难以 scale，这道鸿沟只会变宽

> **"Thinking about memory is the key to thinking about how GPUs work."**

### 1.13 Part 1 小结（PDF p.18）

三条必须记住的事实：

1. GPU 是海量并行 SIMT 系统（SM 为基本单元）
2. Compute（尤其 matmul）scale 速度 >> memory scale
3. 必须尊重 memory hierarchy 才能跑得快

---

## 二、Part 2：让 ML 工作负载在 GPU 上跑得快（PDF p.19–p.45）

### 2.1 开场悬念：方阵 matmul 的神秘图（PDF p.19）

x 轴：方阵 size。y 轴：实测 TFLOPS（硬件利用率）。

图像特征：
- 随 size 增大，整体利用率上升（overhead 被摊薄）
- 但出现**多条并列的 wavy 曲线**，且曲线内部有**剧烈的锯齿/崖跌**
- 不同 size 的性能差可达数倍

本 Part 的终极目标是**完全解释这张图**。

### 2.2 Roofline Model（PDF p.20）

任何 GPU kernel 的性能都落在两个 regime 之一：

```
  throughput
      ↑
  peak┤         ──────────── (Compute-bound)
      │        ╱
      │       ╱
      │      ╱ (Memory-bound，斜率 = 带宽)
      │     ╱
      └────┴──────────────────→ arithmetic intensity (FLOP / byte)
```

- 左侧（低 intensity）：**memory-bound**，throughput 正比于带宽
- 右侧（高 intensity）：**compute-bound**，throughput 饱和 = peak FLOPS
- 目标：**远离左侧**，让所有 matmul 单元满载

> arithmetic intensity = 每 byte 的内存传输支撑多少 FLOP。越高越好。

### 2.3 六大优化技巧总览（PDF p.21）

| # | 技巧                  | 属于哪类                  |
| - | --------------------- | ------------------------- |
| 1 | Control divergence    | 非内存（唯一例外）        |
| 2 | Low precision         | 减少 byte 数              |
| 3 | Operator fusion       | 减少 read/write           |
| 4 | Recomputation         | 用 compute 换 memory      |
| 5 | Memory coalescing     | 利用 DRAM burst           |
| 6 | Tiling                | 把数据搬进 shared memory  |

### 2.4 Trick 1: Control Divergence（PDF p.22）

SIMT 的约束：**一个 warp 内 32 个线程必须执行同一条指令**。

反例代码：

```cuda
if (threadIdx.x < 4) {
    A();  // 前 4 个线程做 A
} else {
    X();  // 后 4 个线程做 X
}
```

硬件实际执行方式：

1. 先让前 4 个线程跑 `A()`，后 4 个**挂起（idle）**
2. 再让后 4 个线程跑 `X()`，前 4 个挂起
3. 完全**串行化**了 if / else 两个分支

所以：**warp 内部尽量无分支**。若必须有分支，**让分支边界与 32 对齐**（整个 warp 走同一路径就没事）。

### 2.5 Trick 2: Low Precision（PDF p.23–p.25）

#### 直观理由

> "If you have fewer bits, you have fewer bits to move."

Bill Dally 曲线里超指数 compute 增长，**相当大一部分来自精度下降**（FP32 → FP16 → INT8 → FP8 → ...），而非纯晶体管进步。

#### 定量：Element-wise ReLU 的 arithmetic intensity

输入向量长度 $n$，运算 $x \leftarrow \max(0, x)$。

**Float32 情形：**

- Memory：1 read + 1 write，每元素 4 bytes → **8 bytes / element**
- Compute：1 comparison + 1 FLOP → **1 FLOP / element**
- Intensity = **8 bytes / FLOP**

**Float16 情形：**

- Memory：每元素 2 bytes → **4 bytes / element**
- Compute：仍 1 FLOP / element
- Intensity = **4 bytes / FLOP**

**带宽瓶颈下 FP16 直接吞吐翻倍**——而 FLOPS 开销没变。

#### Mixed Precision Matmul（PDF p.25）

经典配方（Tensor Core 原生支持）：

```
FP16 inputs × FP16 inputs → FP32 accumulator → FP32 output (可下 cast 为 FP16)
```

原因：
- **输入用低精度**省带宽
- **累加用高精度**，防止大量小数相加产生数值灾难
- **指数型操作**（e.g. softmax 的 exp）用 BF16 而非 FP16，因为 BF16 动态范围更大，避免爆掉/清零

> 训练时这些精度选择需**精细工程**才能保持稳定。但回报巨大——带宽瓶颈下吞吐翻倍。

### 2.6 Trick 3: Operator Fusion（PDF p.26–p.29）

#### Horace He 的工厂隐喻（PDF p.26）

- Compute unit = 工厂
- Global memory = 仓库
- 两者之间有**传送带**（带宽有限）
- 即便工厂扩建（compute scale），**传送带不变就无意义**

#### Naive vs Fused（PDF p.27）

**Naive**：每步运算后把中间结果送回 memory，再取回来做下一步。

```
memory ─□→ compute ─△→ memory ─△→ compute ─○→ memory ─○→ compute ─▽→ memory
```

**Fused**：

```
memory ─□→ compute (□→△→○→▽) → memory
                     └────────────所有中间计算都在 compute 单元里完成
```

#### 例子：$\sin^2 x + \cos^2 x$（PDF p.28–p.29）

PyTorch 朴素实现会启动 **5 个 CUDA kernel**：
1. $\sin(x)$
2. $\cos(x)$
3. $\sin(x)^2$
4. $\cos(x)^2$
5. 加法

每次 kernel 间都要**写回 global memory + 读回来**，带宽消耗巨大。

融合后只需 **1 个 kernel**，输入 $x$ 进去，出来就是结果。

**实践**：`torch.compile` 会自动做这种简单 fusion。**推荐默认开启**。复杂 fusion 才需要手写 Triton/CUDA。

### 2.7 Trick 4: Recomputation（PDF p.30–p.32）

#### 背景：反向传播的内存代价

前向传播时，每层产生的 activation（黄色节点）必须**存起来**（写 global memory）；反向时要**读回**来以计算 Jacobian。这是巨大的读写负担。

#### 例子：三层 sigmoid 堆叠（PDF p.31）

$f(x) = \sigma(\sigma(\sigma(x)))$

**普通做法**的内存操作总数：

- Forward：read $x$ (1) + write $s_1, s_2, \text{out}$ (3) = **4**
- Backward：读取 $s_1, s_2, \text{d\_out}$ 及写 $dx$（需读 3 个梯度/激活 + 写 1）= **4**
- 合计：**8 次 mem read/write**，且 arithmetic intensity 极低（无 matmul）

#### Recomputation 做法（PDF p.32）

**前向不存 $s_1, s_2$，只存 $x$ 和 out**：
- Forward：read $x$ (1) + write out (1) = **2**

**反向时，在 SM 的 local memory 里重新算 $s_1, s_2$**：
- Backward：read $x$ 和 d_out (2) + write $dx$ (1) = **3**（无需从 global 读 $s_1, s_2$）

总计 **5 次 mem read/write**，相比 naive 的 8 次节省 **3/8**。代价：3 次额外的 sigmoid 计算。

**关键洞察**：在 memory-bound regime 下，**compute 是过剩资源**，拿过剩的 compute 换短缺的 memory 带宽就是净赚。

> 这与 activation checkpointing（省显存）是**同一技术**，但**目的不同**：checkpointing 为了省 HBM 容量，recomputation 这里是为了**省带宽/提速**。

### 2.8 Trick 5: Memory Coalescing（PDF p.33–p.35）

#### DRAM 的 Burst Mode（PDF p.33）

DRAM 的物理特性：读一个地址时，硬件会**免费附送同一 row 里的若干相邻值**。这是因为：

> 从 storage 单元移动数据到 sense amplifier 是昂贵的慢步；但一旦数据上了 amplifier，取多少字节都很便宜。

所以 DRAM 把地址空间切成 **burst section**（如每 4 个连续元素一组），每次访问以 burst 为单位返回。

示意（灰方块为 burst section 边界）：

```
┌── burst 0 ──┬── burst 1 ──┬── burst 2 ──┬ ...
[0  1  2  3] [4  5  6  7]  [8  9  10 11]
```

#### Memory Coalescing（PDF p.34）

**规则**：一个 warp 内的 32 个线程如果恰好都访问**同一个 burst section**，硬件把它们**合并为 1 次内存请求**；否则每个线程都是单独请求。

→ 若 warp 里线程访问模式连续对齐，**带宽可提高到 4× 甚至更高**。

#### 例子：Matmul 的 row-major 布局（PDF p.35）

存储方式：矩阵按 row-major 存放，一个 row 的元素在内存里连续。

考虑 matmul，两种线程布局：

| 方案 | 每个 thread 负责 | 相邻 thread 访问的元素 | 是否 coalesced |
| ---- | ---------------- | ---------------------- | -------------- |
| 左：每个 thread 沿**列**方向 traverse | 沿一列往下遍历 | 相隔一整行 → 跨 burst | **否**，慢 |
| 右：每个 thread 沿**行**方向 traverse | 沿一行往右遍历 | 相邻元素 → 同 burst | **是**，快 |

**直觉性陷阱**：这张图初看容易搞反。正确理解方式：
- 在**同一时刻**（time step 1），warp 里 32 个 thread 同时发起 1 次 load
- 这 32 次 load 的目标地址如果连续对齐，就合并成 1 次 burst 读取
- 第二种布局下，thread 0 读 $M_{0,0}$，thread 1 读 $M_{0,1}$... 全在同一 row（同一 burst）→ 合并
- 第一种布局下，thread 0 读 $M_{0,0}$，thread 1 读 $M_{1,0}$... 在不同 row（不同 burst）→ 不能合并

> 这种低层细节若搞反，matmul 速度差 **4× 起步**。

### 2.9 Trick 6: Tiling（PDF p.36–p.40）

#### 问题（PDF p.36）

Naive matmul：
$$
P_{i,j} = \sum_k M_{i,k} \cdot N_{k,j}
$$

每个 thread 负责 $P_{i,j}$，遍历 k。结果：
- 同一个 $M_{0,0}$ 被**反复**从 global memory 读取（被所有 $P_{0,\cdot}$ 用到）
- $N_{1,0}$ 同理被反复读
- 访问还可能非 coalesced

#### Tiling 思路（PDF p.37）

理想情景：**一次性把一块 tile 从 global 搬进 shared memory → 在 shared memory 上做大量计算 → 彻底用完才换下一块**。

具体算法（方阵，tile size $T$）：

1. 把 $M, N$ 切成 $T \times T$ 的 tile。如 $M$ 被切成 $M_{0,0}, M_{0,1}, M_{1,0}, \dots$
2. 对于输出 tile $P_{0,0}$：
   - Phase 1：把 $M_{0,0}$ 和 $N_{0,0}$ load 进 SHM → 计算 partial sum，累加到 $P_{0,0}$
   - Phase 2：把 $M_{0,1}$ 和 $N_{1,0}$ load 进 SHM → 累加
   - ... 直到遍历完所有 k 方向的 tile 对
3. 对每个输出 tile 重复

**优势**：
- 同一 tile 在 shared memory 内被**反复复用**，这部分读取无 global 代价
- Load tile 时可以用 coalesced 访问

#### Tiling 的数学分析（PDF p.38）

方阵 $N \times N$，tile size $T \times T$。

**Non-tiled**：每个元素被读 $N$ 次（因为参与 $N$ 次内积运算）

**Tiled**：每个元素从 **global memory** 读 $N/T$ 次（只在 tile 被 load 进来时读一次），**在 shared memory 内** 被读 $T$ 次

→ Global memory 访问**减少 $T$ 倍**。

直觉：能 load 进 SHM 的 tile 越大，global 访问越少。但 tile 太大 SHM 放不下，必须权衡。

#### Tiling 复杂度 1：Tile Quantization（PDF p.39）

如果矩阵维度**不是 tile size 的整数倍**：

- 矩阵 $256 \times 256$，tile $128 \times 128$ → 2×2 = 4 tiles，完美
- 矩阵 $257 \times 256$，tile $128 \times 128$ → 需要 3×2 = 6 tiles，多出的两个 tile **几乎空**

因为每个 tile 对应一个 block，派给一个 SM——空 tile 让那个 SM **几乎闲置**，整体利用率掉。

**tile 大小需要考虑**：
- Coalescing 要求（与 burst 对齐）
- Shared memory 容量（不能超）
- 矩阵维度能被整除（或尽量接近）

#### Tiling 复杂度 2：Burst 对齐（PDF p.40）

假设某矩阵内存布局恰好使每行 = 一个整数倍 burst section → load 一行就是几次 burst 读取，完美。

现在给矩阵**加一列**（一个多余元素）：每行的起始地址**错开**了一个单位 →

- 第 1 行还在一个 burst 内（好）
- 第 2 行开始**跨越两个 burst** → **读取次数翻倍**！
- 第 3 行跨越更多...

→ **一个多余元素能使 memory access 次数直接翻倍**。

解决：**padding**，把矩阵维度填充成 burst size 的整数倍。

这也解释了 Andrej Karpathy 的著名 tweet：把 nanoGPT 的 vocab 从 50257 改成 **50304**（64 的倍数）→ **25% 性能提升**。

#### 思考：能否预取 tile？

学生问：GPU 能不能在计算当前 tile 时 prefetch 下一个 tile？

答：**GPU 天然就会尽量 overlap memory 和 compute**——只要 shared memory 有空间就会去填。但当 SM 利用率跑满时，SHM 本来就是瓶颈资源，**没有空间可以 prefetch**。

### 2.10 解开悬念：Matmul 神秘图（PDF p.41–p.44）

用上面的工具重审悬念图：

#### 整体形态：Roofline

- 横轴方阵 size $< 1536$：**memory-bound**，compute 喂不饱（左侧斜坡）
- 横轴方阵 size $> 1536$：**compute-bound**（右侧平台）
- 这解释了整体上升趋势和上界

#### 并列多条曲线：Tile 对齐（PDF p.43）

按矩阵 size 的可除性给点染色：

| 颜色    | size 可被整除的最大 2 的幂 $k$ | 位置     |
| ------- | ------------------------------ | -------- |
| 紫      | 32                             | 最上（快） |
| 蓝/绿   | 16                             | 次上     |
| 绿      | 8                              | 中       |
| 橙      | 2                              | 下       |
| 红      | 1（如质数）                    | 最下（慢） |

可除性越差 → 越难做 burst-aligned tile load → 越慢。**别用质数作为维度**。

#### 单条曲线内的锯齿：Wave Quantization（PDF p.44）

关键案例：从 $1792$ 突变到 $1793$ 时性能**直接崩溃**。

计算：设 tile $256 \times 128$（Tensor Core 原生尺寸）

- $1792 \times 1792$ 矩阵：$\frac{1792}{256} \times \frac{1792}{128} = 7 \times 14 = 98$ tiles
- $1793 \times 1793$ 矩阵：向上取整 → $8 \times 15 = 120$ tiles

A100 有 **108 SMs**：

- 98 tiles：**一波全部 parallel 跑完**（108 > 98），完美
- 120 tiles：先 108 个并行 → 剩下 **12 个** 单独再跑一波，只用 12 个 SM，**剩下 96 个 SM 空转**

→ **利用率断崖下跌**，这就是 "wave quantization"。

**启示**：设计矩阵维度时，要么 tiles 数远大于 SM 数（摊薄效应），要么刚好匹配，**最坏情况是略超 SM 数的一小波**。

### 2.11 Part 2 小结（PDF p.45）

三大类优化手段：

1. **减少 memory access**
   - Coalescing（利用 burst）
   - Fusion（减少中间 read/write）
2. **把 memory 搬到更快的地方**
   - Tiling（把热数据放进 shared memory）
3. **用其他资源换 memory**
   - Quantization（用精度换带宽）
   - Recomputation（用 compute 换带宽）

共同主线：**GPU 性能 = 数据移动最优化**。

---

## 三、Part 3：Flash Attention 全拆解（PDF p.46–p.50）

### 3.1 为什么 Flash Attention 快（PDF p.46）

论文原话：

> "We apply two established techniques —— **tiling and recomputation** —— to overcome the technical challenge of computing exact attention in **sub-quadratic HBM accesses**."

**关键不是 compute 复杂度**（仍 $O(N^2)$），而是 **HBM 访问复杂度** sub-quadratic。

因为 attention 的瓶颈是 memory，不是 FLOPS。

### 3.2 Attention Recap（PDF p.47）

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

三个 matmul + 中间一个 softmax：

```
         QK^T          softmax         ·V
Q, K  ────────→  S  ────────→  P  ────────→  O
```

- 两个 matmul（QK^T 和 PV）完全可以用 **tiling** 搞定
- **难点是 softmax**

### 3.3 QKV Matmul 的 Tiling（PDF p.48）

Flash Attention 论文 Figure 1 本质：

- $Q, K, V$ 切成块 → load 进 SRAM → 计算 inner product → 写回 HBM
- 完全照搬 Part 2 的 tiled matmul

**问题**：softmax 是 **跨整行的 global 操作**（要对整行求和做归一化）。tile 只有部分行信息，怎么办？

### 3.4 Online Softmax（PDF p.49）

Milakov & Gimelshein (2018) 的算法，核心是**增量计算 softmax**。

#### 标准数值稳定 softmax（batch 版）

对输入 $x_1, \dots, x_N$：

$$
m = \max_k x_k, \qquad y_i = \frac{\exp(x_i - m)}{\sum_{k=1}^{N} \exp(x_k - m)}
$$

需要**先扫一遍得到 $m$，再扫一遍求和**，必须一次性见到所有 $x$。

#### Online Softmax：telescoping 技巧

维护两个状态，随 $j$ 从 $1$ 增量更新：

- $m_j = \max(x_1, \dots, x_j)$：当前已见最大值
- $d_j = \sum_{k=1}^{j} \exp(x_k - m_j)$：当前归一化分母（基于当前 max）

**递推公式**（当新的 $x_j$ 来时）：

$$
m_j = \max(m_{j-1},\, x_j)
$$

$$
d_j = d_{j-1} \cdot \exp(m_{j-1} - m_j) + \exp(x_j - m_j)
$$

直觉：

- 如果 max 被更新了（$m_j > m_{j-1}$），就要把之前累积的 $d_{j-1}$ **按新的 max 修正**（乘上 $\exp(m_{j-1} - m_j)$，这个因子 $\le 1$）
- 然后加上新项 $\exp(x_j - m_j)$

最终：

$$
y_i = \frac{\exp(x_i - m_N)}{d_N}
$$

**关键性质**：只需**流式**看到 $x_1, \dots, x_N$ 一次，就能算出整个 softmax。**不需要先见所有 $x$**。

→ 可以 **tile by tile** 处理：每看完一个 tile，更新 $(m, d)$；最后输出时用最终的 $(m_N, d_N)$。**不需要把 $N \times N$ 的 attention 矩阵 $S$ 实体化**。

### 3.5 前向传播全流程（PDF p.50）

Flash Attention forward pass 集成的三大技巧：

| 技巧                       | 作用                                         |
| -------------------------- | -------------------------------------------- |
| **Tiled inner products (S = QK^T)** | 不实体化 $N^2$ attention score 矩阵          |
| **Fused exponential**      | `softmax → matmul V` 融合，避免回写 HBM      |
| **Online softmax**         | softmax 可 tile by tile 增量计算             |

算法骨架（对 output 的每个 tile $O_i$）：

```
初始化 O_i = 0, m_i = -inf, d_i = 0
for j in (K, V 的每个 tile):
    把 Q_i, K_j, V_j load 进 SRAM
    计算 S_ij = Q_i @ K_j^T               (tile-wise matmul)
    m_new = max(m_i, rowmax(S_ij))
    更新 d_i 和 O_i 的 running sum，用 telescoping 修正因子 exp(m_i - m_new)
    O_i += exp(S_ij - m_new) @ V_j        (tile-wise matmul)
    m_i = m_new
最终输出 O_i / d_i
```

**关键**：整个过程只往 HBM 写**最终的 $O_i$**（形状 $N \times d$，而不是 $N \times N$）。HBM 访问从 $O(N^2)$ 降到 $O(N \cdot d \cdot N/T)$。

#### 学生提问：这不是还要扫完所有 tile 才能输出吗？

讲师答：**是的**，但只需扫**一遍**所有 tile。扫完后 $(m_N, d_N)$ 已就绪，最后一个 tile 处理完时直接用它归一化输出，**无需第二遍扫描**。

### 3.6 Backward Pass（讲师略讲）

反向传播用同样的 tile-wise **recomputation**：
- 不存 $N^2$ 的 attention 矩阵
- 反向需要时就从 $Q, K, V$ tile-wise 重算（用的还是已存的 $(m_N, d_N)$）
- 牺牲一些 compute，换 HBM 带宽

这是 Part 2 "recomputation" 技巧的直接应用。

---

## 四、整讲回顾（PDF p.51）

三个必须记住的核心结论：

1. **硬件 = scale 的动力**。低层细节决定谁能 scale、谁不能
2. **当今 GPU 强烈鼓励 "matmul + 数据移动" 思维**——神经网络要以 matmul 为主，算法要精心设计 memory flow
3. **认真考虑 GPU 特性（coalescing / tiling / fusion）是获得好性能的关键**

核心心法（贯穿整讲）：

> **内存移动是瓶颈。不要只盯着 FLOPS。把数据搬进 shared memory，让它在那儿待得尽量久。**

这就是 Flash Attention 能成立的根本理由，也是 Assignment 2 中你要动手实现这一切的原因。

---

## 附录：关键数字速查

| 量                            | 值                              |
| ----------------------------- | ------------------------------- |
| Warp size                     | **32 threads**                  |
| A100 SM 数                    | **108**                         |
| On-SM 内存延迟                | ~20 cycles                      |
| L2 / global 延迟              | ~200–300 cycles                 |
| SRAM vs DRAM                  | 贵 100×，快 ~8×                 |
| Matmul vs non-matmul FLOPS (V100+) | Matmul 快 **>10×**         |
| 10 年 GPU compute 增长        | **>1000×**                      |
| Memory vs compute scale 差距  | compute 快约 **1000×** 于 memory |
| nanoGPT vocab padding 案例    | 50257 → 50304（64 倍数）= **+25%** |
| Wave quantization 临界        | tile 数 > SM 数（A100: 108）    |

