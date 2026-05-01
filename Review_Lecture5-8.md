# CS336 阶段性复习笔记：Lecture 5–8

> Stanford CS336 - Language Modeling from Scratch (Spring 2025)
>
> 覆盖：GPUs (L5)、Kernels & Triton (L6)、Parallelism Basics (L7)、Parallelism Code Walkthrough (L8)
>
> 这四讲构成课程的 **"Systems" 模块**：从单 GPU 硬件 → 单 GPU 内核优化 → 多 GPU 并行理论 → 多 GPU 并行代码，是一条一气呵成的主线。本笔记面向复习：只保留真正值得反复回看的核心知识、公式、表格、设计决策背后的理由。

---

## 第 0 章 全模块主线

整个 Systems 模块围绕**一条贯穿所有层级的中心原则**：

> **算力单元永远离数据很远。所有性能优化的本质都是"组织计算去避开数据传输瓶颈"。**

无论粒度是：
- SM 内部的寄存器 ↔ shared memory
- 单 GPU 的 SRAM ↔ HBM
- 单节点的 GPU ↔ GPU（NVLink）
- 多节点的节点 ↔ 节点（InfiniBand）

这条原则都成立。**FLOPs 通常不是瓶颈，memory movement 才是。**

### 0.1 统一的存储层级（从快到慢，代价从小到大）

| 层级 | 容量 | 带宽（H100 参考） | 谁能访问 |
| --- | --- | --- | --- |
| 寄存器 | 字节级 | 极快 | 单 thread |
| Shared Memory / L1 | 192 KB/SM | 极快（~20 cycles） | 同一 block |
| L2 Cache | 40 MB | ~200–300 cycles | 整卡 |
| HBM（global memory） | 80 GB | 3.9 TB/s | 所有 block |
| NVLink | — | 900 GB/s | 节点内 GPU |
| InfiniBand | — | ~NVLink / 8 | 节点间（≤256 GPU） |
| Leaf / Spine Switch | — | 再降一档 | 跨 rack |

**结论**：**NVLink 比 HBM 只慢 ~4×**（节点内多卡"不算太亏"）；**IB 比 NVLink 慢 ~8×**（节点间要精打细算）；**跨 rack 再降一档**（尽量避免）。

### 0.2 三选一原则（贯穿单卡/多卡所有粒度）

面对"我需要这个张量，但它不在本地"，永远是三选一：

1. **重算 (recompute)**：放弃存它，要用时重新算一遍 —— 用 compute 换 memory
2. **存本地 (store)**：用显存换时间 —— 占容量但最快
3. **存异地 + 通信 (communicate)**：用带宽换显存 —— 要算通信代价

所有并行策略（DP / TP / PP / FSDP）、activation checkpointing、flash attention 都是这三者在不同维度上的组合。

### 0.3 四讲的结构关系

```
L5 GPU 硬件  ──┬──► L6 Kernel（单 GPU 内部的"省搬运"）
               │
               └──► L7 Parallelism 理论（多 GPU 间的"省搬运"）
                      │
                      └──► L8 Parallelism 代码（把 L7 落到 PyTorch）
```

---

## 第 1 章 GPU 硬件模型（L5 Part 1）

### 1.1 为什么是 GPU

**Dennard Scaling 在 2000s 后终结**：晶体管继续变小，但单线程性能曲线拍平（功耗密度挡住了时钟频率）。

**结论**：**无法再靠"绝对更快"，只能靠"大规模并行"。**

GPU 用 10 年把 compute 增长 **>1000×**（K20 → H100），完全是通过并行+降精度实现的。

### 1.2 CPU vs GPU 设计哲学

| 对比项 | CPU | GPU |
| --- | --- | --- |
| 线程数 | 几~几十 | 上万 |
| 控制单元 | 大（分支预测、乱序、缓存） | 极小 |
| ALU 占比 | 低 | 极高 |
| 优化目标 | **Latency**（单任务尽快） | **Throughput**（总吞吐） |

GPU 单任务 latency 反而比 CPU 差，但**总吞吐**远胜。

### 1.3 执行单元分层（必须记）

**硬件两级**：
- **SM**（Streaming Multiprocessor）= 独立的"工人"。A100 有 **108 SMs**，H100 更多
- **SP**（Streaming Processor）= SM 内的线程执行单元
- **Tensor Core** = SM 内的专用 matmul 电路

**软件三级粒度**（从大到小）：

1. **Block**：一组 threads，整体调度到一个 SM。Block 之间**只能通过 global memory 通信**
2. **Warp**：**32 个连续编号 threads**，**同一 cycle 执行同一条指令**（SIMT）。硬件调度基本单位
3. **Thread**：最小单元，执行指令、持有自己的寄存器

```
GPU
 └─ SM × 108 (A100)
     └─ Block (调度到某 SM)
         └─ Warp (32 threads, 同步同指令)
             └─ Thread (最小单元)
```

**为什么 warp = 32**：硬件固定。这直接导致 **Control Divergence** 问题（见 §2.4）。

### 1.4 内存访问权限表（必须记）

| 存储类型 | 谁能读写 | 速度 |
| --- | --- | --- |
| Register | 单 thread 私有 | 最快 |
| Local memory | 单 thread（溢出到 DRAM） | 慢 |
| **Shared memory** | **同一 block 内所有 threads** | 快 |
| Global memory | 所有 block | 慢（HBM） |
| Constant | 所有 threads（只读） | 中 |

**核心约束**：**Block 间只能通过 global memory 通信** → 好算法应尽量让一个 block 独立完成尽可能多的工作。

### 1.5 Compute 增长 >> Memory 增长

三条曲线（log 尺度）：

| 曲线 | 10 年增长 |
| --- | --- |
| PCIe/NVLink 互联 | 数×量级 |
| 显存带宽（GDDR → HBM2E） | ~100× |
| Compute (matmul FLOPS) | **~100,000×** |

**推论**：早期瓶颈是 FLOPS；**现在瓶颈是内存带宽**。这是本模块一切优化的根源。

### 1.6 Matmul FLOPS ≫ 非 matmul FLOPS（V100 起）

Tensor Core 让 matmul FLOPs 比非 matmul 高 **>10×**。**设计架构时必须以 matmul 为主**，否则直接损失一个数量级算力。

---

## 第 2 章 单 GPU 性能优化六大技巧（L5 Part 2）

### 2.1 Roofline Model

任何 kernel 落在两个 regime：

```
throughput
    ↑
peak┤         ──────── (Compute-bound)
    │        ╱
    │       ╱
    │      ╱ (Memory-bound, 斜率=带宽)
    └────┴──────────→ arithmetic intensity (FLOP/byte)
```

**Arithmetic intensity** = FLOPs / bytes moved。越高越好。

**一般规律**：matmul 是 compute-bound，**其他几乎都是 memory-bound**。

### 2.2 六大技巧总览

| # | 技巧 | 类别 |
| --- | --- | --- |
| 1 | Control divergence | 非内存（唯一例外） |
| 2 | Low precision | 减少 byte 数 |
| 3 | Operator fusion | 减少 read/write |
| 4 | Recomputation | compute 换 memory |
| 5 | Memory coalescing | 利用 DRAM burst |
| 6 | Tiling | 搬进 shared memory |

### 2.3 Control Divergence

**SIMT 约束**：warp 内 32 threads 必须执行同一指令。

```cuda
if (threadIdx.x < 4) { A(); } else { X(); }
```

硬件实际：先让前 4 做 A（后 4 挂起），再让后 4 做 X（前 4 挂起）—— **完全串行化**。

**规则**：warp 内尽量无分支；必须有则让分支边界 **32 对齐**。

### 2.4 Low Precision

> "If you have fewer bits, you have fewer bits to move."

Bill Dally 曲线里 compute 增长的**一大半来自精度下降**（FP32→FP16→FP8→...）。

**Element-wise ReLU 的 arithmetic intensity**：

| 精度 | bytes/element | FLOP/element | Intensity (bytes/FLOP) |
| --- | --- | --- | --- |
| FP32 | 8 | 1 | 8 |
| FP16 | 4 | 1 | 4 |

**带宽瓶颈下 FP16 直接吞吐翻倍**。

**Mixed Precision MatMul 配方**：

```
FP16 输入 × FP16 输入 → FP32 累加器 → FP32 / BF16 输出
```

- **输入低精度**省带宽
- **累加高精度**防数值灾难
- **指数型操作（softmax 的 exp）用 BF16**：动态范围更大

### 2.5 Operator Fusion（Horace He 的"工厂/仓库"比喻）

- Compute unit = 工厂
- Global memory = 仓库
- 传送带（带宽）有限

**Naive**：每步 op 后把中间结果送回 memory
```
memory → compute → memory → compute → memory → ...
```

**Fused**：所有中间计算都在 compute 单元里完成
```
memory → compute (op1→op2→op3→...) → memory
```

**实战**：`torch.compile` 自动 fuse 简单操作。默认开启。

### 2.6 Recomputation

**思路**：memory-bound regime 下 compute 是过剩资源。拿过剩的 compute 换短缺的 memory 带宽 → 净赚。

**三层 sigmoid 例**：普通做法 8 次 mem read/write；不存中间 $s_1, s_2$，反向时在 SM local memory 里重新算 → 5 次 mem read/write，代价是 3 次额外 sigmoid。

**与 activation checkpointing 是同一技术，但目的不同**：
- Checkpointing：省 HBM 容量
- Recomputation：省带宽/提速

### 2.7 Memory Coalescing

**DRAM Burst Mode**：读一个地址时硬件免费附送同一 row 的相邻值。DRAM 以 **burst section**（如 4 连续元素一组）为单位访问。

**规则**：warp 内 32 threads 如果恰好访问**同一个 burst section**，硬件合并成 **1 次**内存请求；否则每个 thread 单独请求。

**Matmul row-major 例**：
- 沿**列**遍历（thread 0 读 $M_{0,0}$，thread 1 读 $M_{1,0}$...）：跨 burst，**不合并**，慢
- 沿**行**遍历（thread 0 读 $M_{0,0}$，thread 1 读 $M_{0,1}$...）：同 burst，**合并**，快

→ 搞反布局，matmul 速度差 **4× 起步**。

### 2.8 Tiling（Shared Memory）

**动机**：Naive matmul 里每个 $M_{i,k}$ 被反复从 global memory 读取。

**思路**：一次把一块 tile 搬进 shared memory → 在 SHM 上做大量计算 → 彻底用完再换下一块。

**数学收益**：方阵 $N\times N$、tile size $T$：
- Non-tiled：每元素从 global 读 $N$ 次
- Tiled：每元素从 global 读 $N/T$ 次，在 SHM 内读 $T$ 次

→ **Global memory 访问减少 $T$ 倍。**

**复杂度 1：Tile Quantization**

矩阵维度不是 tile size 整数倍 → 多出的 tile 几乎空 → 对应 SM 闲置 → 利用率掉。

**复杂度 2：Burst 对齐**

每行起始地址要对齐 burst section。**加一列多余元素**就会让每行跨 2 个 burst → **读取次数翻倍**。

→ 经典案例：**nanoGPT 把 vocab 从 50257 改成 50304（64 倍数）→ +25% 性能**（Karpathy 的著名 tweet）。

### 2.9 Matmul "神秘图" 的完整解释（必懂）

横轴：方阵 size；纵轴：实测 TFLOPS。三个现象：

**（a）整体上升趋势**：Roofline。size < 1536 memory-bound；> 1536 compute-bound。

**（b）并列多条曲线（tile 对齐）**：按 size 的最大 2 的幂可除性染色：

| 颜色 | 可除性 $k$ | 位置 |
| --- | --- | --- |
| 紫 | 32 | 最上（快） |
| 蓝/绿 | 16 | 次上 |
| 橙 | 2 | 下 |
| 红 | 1（质数） | 最下（慢） |

**→ 别用质数作为维度。**

**（c）单曲线锯齿（Wave Quantization）**：

从 1792 → 1793 性能断崖。设 tile $256\times 128$：
- $1792$：$7 \times 14 = 98$ tiles，一波跑完（A100 有 108 SMs）
- $1793$：$8 \times 15 = 120$ tiles → 第一波 108 个 + 第二波仅 12 个 → **96 个 SM 空转**

**启示**：要么 tiles 数 ≫ SM 数（摊薄），要么刚好匹配；**最坏的是略超 SM 数的一小波**。

### 2.10 三大类优化手段

1. **减少 memory access**：Coalescing（利用 burst）、Fusion（减少中间 r/w）
2. **把 memory 搬到更快的地方**：Tiling（热数据放 SHM）
3. **用其他资源换 memory**：Quantization（精度换带宽）、Recomputation（compute 换带宽）

共同主线：**GPU 性能 = 数据移动最优化**。

---

## 第 3 章 Flash Attention（L5 Part 3）

### 3.1 为什么 Flash Attention 快

论文原话：

> "We apply two established techniques — **tiling and recomputation** — to overcome the technical challenge of computing exact attention in **sub-quadratic HBM accesses**."

**关键不是 compute 复杂度**（仍 $O(N^2)$），而是 **HBM 访问复杂度** sub-quadratic。

### 3.2 技术难点：Softmax 是跨整行的 global 操作

Attention 三步：$Q K^\top \to \text{softmax} \to \cdot V$。QK 和 PV 两个 matmul 能用 tiling，但**softmax 需要整行 max 和 sum**，tile 只有部分信息。

### 3.3 Online Softmax（Milakov & Gimelshein 2018）

维护两个状态随 $j$ 增量更新：
- $m_j = \max(x_1, \dots, x_j)$：当前已见最大值
- $d_j = \sum_{k=1}^{j} \exp(x_k - m_j)$：当前归一化分母

**递推公式**：

$$
m_j = \max(m_{j-1}, x_j)
$$

$$
d_j = d_{j-1} \cdot \exp(m_{j-1} - m_j) + \exp(x_j - m_j)
$$

**直觉**：max 更新了就把之前累积的 $d_{j-1}$ 按新的 max 修正（乘上 $\exp(m_{j-1}-m_j) \le 1$），再加新项。

**性质**：只需**流式**扫一遍 $x_1, \dots, x_N$ 即可算出整个 softmax → **可以 tile by tile 处理**，**无需实体化 $N \times N$ 的 attention 矩阵**。

### 3.4 Flash Attention 集成三大技巧

| 技巧 | 作用 |
| --- | --- |
| **Tiled inner products** | 不实体化 $N^2$ attention score |
| **Fused exponential** | `softmax → matmul V` 融合，避免回写 HBM |
| **Online softmax** | softmax tile by tile 增量计算 |

**HBM 访问从 $O(N^2)$ 降到 $O(N \cdot d \cdot N/T)$**。

反向同样用 tile-wise **recomputation**：不存 $N^2$ 矩阵，反向时从 $Q, K, V$ 重算。牺牲少量 compute 换 HBM 带宽。

---

## 第 4 章 Kernels & Triton（L6）

### 4.1 五种写法的选择树

| 方式 | 抽象层级 | 何时用 |
| --- | --- | --- |
| 朴素 PyTorch | 最高 | 原型、可读 |
| 官方 fused (`F.gelu`) | 高 | 能用就用 |
| **`torch.compile`** | 中高 | **默认首选加速** |
| Triton | 中 | 需要算法级定制 |
| CUDA | 低 | 极少需要 |

**实操工作流**：
1. 写朴素实现跑通
2. benchmark 测端到端时间
3. profile 找瓶颈 kernel
4. 试 `torch.compile` —— 通常够用
5. 还不够快 → profiler 定位后写 Triton
6. Triton 都不行 → 算法级重写（Flash Attention 那种层次）

**心态**：自动编译器只会越来越好，**手写 CUDA 的 ROI 在下降**。

### 4.2 Benchmark 模板（必背）

```python
def benchmark(run, num_warmups=1, num_trials=3):
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()   # 等 GPU 做完

    times = []
    for _ in range(num_trials):
        start = time.time()
        run()
        torch.cuda.synchronize()   # 每次计时末尾也要同步
        end = time.time()
        times.append((end - start) * 1000)
    return mean(times)
```

**两件必做的事**：
1. **Warmup**：消除 JIT 编译、kernel 加载、cache 未命中等一次性开销
2. **`torch.cuda.synchronize()`**：CPU 和 GPU 异步，不同步就计时会得到"瞬间完成"的荒谬结果

### 4.3 Profiling：`torch.profiler` & Nsight Systems

**torch.profiler** 能打开到 CUDA kernel 级：
- `aten::mm` → dispatch 到 cutlass → 真正的 kernel `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`
- **PyTorch 的 matmul 在底层会根据形状/硬件分派到不同 kernel** → `torch.compile` 的 autotune 能白给 ~10%

**Nsight Systems** 看 CPU/GPU 双泳道时间线，关键观察：

**CPU 一直在"跑超前"**：CPU 把 kernel 不断入队，并不等 GPU。这是为什么 Python 这种慢语言做训练仍能打满 GPU 的根本原因。

**陷阱：`print(loss)` 会破坏异步** —— CPU 要拿到 loss 必须等 GPU 算完，触发隐式 sync，让 GPU 出现空泡。

### 4.4 Kernel Fusion 实测（GeLU, dim=16384）

| 实现 | 耗时 |
| --- | --- |
| `manual_gelu`（朴素 PyTorch） | ~8.1 ms |
| `pytorch_gelu`（官方 fused） | ~1.1 ms |
| `cuda_gelu`（手写 CUDA） | ~1.8 ms |
| `triton_gelu` | ~1.85 ms |
| `compiled_gelu`（`torch.compile`） | **~1.47 ms** |

**观察**：
- 朴素 → fused **约 8× 加速**，纯来自 kernel fusion
- `torch.compile` 反而**比手写 Triton 还快** —— 它生成的 Triton 代码比我们的更优化

### 4.5 CUDA Kernel 模板（逐元素）

```cpp
__global__ void gelu_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;     // 坐标计算
    if (i < n) {                                        // 边界检查
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + tanhf(...));
    }
}

torch::Tensor gelu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(x.is_contiguous());                     // 必须连续
    auto y = torch::empty_like(x);                      // 不要 zeros
    int block_size = 1024;
    int num_blocks = (x.numel() + block_size - 1) / block_size;  // ceil-div
    gelu_kernel<<<num_blocks, block_size>>>(...);
    return y;
}
```

**关键点**：
- `__global__` 标记从 CPU 调、GPU 跑的 kernel
- 坐标 `i = blockIdx.x * blockDim.x + threadIdx.x`
- 边界检查 `if (i < n)`
- `empty_like` 而非 `zeros_like`（反正要全写一遍）
- 调用方要 `.contiguous()`
- Debug 时设 `CUDA_LAUNCH_BLOCKING=1`

### 4.6 Triton：Python 写 GPU Kernel

**定位**：**以 thread block 为单位思考**，编译器自动处理 coalescing、shared memory、SM 内调度。

| 任务 | CUDA | Triton |
| --- | --- | --- |
| Memory coalescing | 手动 | **自动** |
| Shared memory | 手动 | **自动** |
| SM 内调度 | 手动 | **自动** |
| SM 间调度 | 手动 | 手动 |

**关键差别**：
- CUDA kernel 里 `i` 是**标量**（单 thread 位置）
- Triton kernel 里 `offsets` 是**向量**（整个 block 位置），`x` 也是向量
- **`mask` 机制替代 CUDA 的 `if (i < n)`**
- `tl.constexpr` 的参数会参与编译（每个 `BLOCK_SIZE` 生成一个专门的 kernel）

**Triton GeLU kernel**：

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    # ... 计算 ...
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 4.7 Triton Softmax（含 Reduction）

**核心设计**：**一个 block 处理一整行**。
- 行所有元素塞进一个 block
- Block 内部自由做 max / sum 聚合
- Block 之间无依赖，完美并行

```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_stride, y_stride, N, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    x = tl.load(x_ptr + row_idx * x_stride + col_offsets,
                mask=mask, other=float("-inf"))   # padding 填 -inf
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(y_ptr + row_idx * y_stride + col_offsets, num / den, mask=mask)
```

**细节**：
- `other=float("-inf")`：padding 位 `exp(-inf)=0`，自动不参与求和
- 减 max 是数值稳定性标准技巧
- `BLOCK_SIZE` 必须是 2 的幂（Triton 要求）

### 4.8 Tiled Matmul 思想

**Block 遍历顺序**决定 L2 命中率：
- **行主序**：访问 9 + 81 = 90 个 block
- **分组遍历**（grouped）：访问 27 + 27 = 54 个 block

**差别纯粹是遍历顺序**，数据局部性不同，L2 命中率差异巨大。

**为什么还要自己写 matmul？** 唯一理由：**和其他算子融合**（`gelu(A@B)`、`softmax(QK^T)` —— 这就是 Flash Attention 的起点）。

### 4.9 核心原则（一句话）

> **Organize computation to minimize reads/writes. 核心思想：kernel fusion + tiling。**

---

## 第 5 章 分布式基础：网络与集合通信原语（L7 §1, L8 §1-3）

### 5.1 GPU 集群分层拓扑（必记）

| 层级 | 带宽 | 延迟 | 规模 | 含义 |
| --- | --- | --- | --- | --- |
| 节点内（NVLink/NVSwitch） | 极高 | 极低 | ≤ 8 GPU | 放最贵的通信 |
| 节点间（InfiniBand） | 高 | 中 | ≤ 256 GPU | 可容忍中等通信 |
| 跨 rack（Leaf/Spine） | 低 | 高 | > 256 | 尽量点对点小数据 |

**这个层级结构直接决定并行策略放在哪一层**：带宽要求高的放节点内，低的放更远。

### 5.2 GPU vs TPU 拓扑

- **GPU**：节点内 8 卡 all-to-all；节点间 all-to-all 到 ~256
- **TPU**：**toroidal mesh（环面网格）**，每芯片只跟邻居通信，但可无限扩展 → 对 all-to-all 无天然支持，但 **ring-based 集合（all-reduce、reduce-scatter）一样高效**

这解释了为什么 TPU 很少用 PP（点对点），而 GPU 跨 256 卡后必须上 PP。

### 5.3 集合通信原语（world_size = W）

分析并行算法的"汇编语言"。以参数总量 $P$ 计：

| 原语 | 语义 | 通信代价 |
| --- | --- | --- |
| **Broadcast** | rank 0 的数据复制到所有 rank | $\sim P$ |
| **Scatter** | rank 0 切片分发 | $\sim P$ |
| **Gather** | rank $i$ 的 $T_i$ 收集到 rank 0 | $\sim P$ |
| **Reduce** | 求和后发给 rank 0 | $\sim P$ |
| **All-Reduce** | 求和后每 rank 都有完整和 | $\sim 2P$ |
| **All-Gather** | 每 rank 贡献一段，最终每 rank 拿全量 | $\sim P$ |
| **Reduce-Scatter** | 求和结果分散到各 rank | $\sim P$ |

### 5.4 关键恒等式（本模块最重要的一个）

$$
\boxed{\underbrace{\text{All-Reduce}}_{2P} \;\equiv\; \underbrace{\text{Reduce-Scatter}}_{P} \;+\; \underbrace{\text{All-Gather}}_{P}}
$$

**含义**：凡做 All-Reduce 的地方都可以拆成"先 RS，再 AG"，**中间可以插额外计算而不增加带宽** —— 这是 **ZeRO 的核心套路**，也是 **Sequence Parallel 能免费替代 TP All-Reduce** 的原理。

### 5.5 PyTorch 分布式栈（L8 §2）

```
用户代码 (Python / PyTorch)
   │
torch.distributed         ← API 层，all_gather_into_tensor 等
   │
NCCL (NVIDIA 集合通信库)   ← 探测拓扑、选路径、翻译成 CUDA kernel
   │
硬件 (NVLink / NVSwitch / IB)
```

- **NCCL** 是 GPU 后端；**Gloo** 是 CPU 后端（笔记本 debug）
- `nvidia-smi topo -m` 查本机 GPU 互联拓扑

**多进程启动模型**：`spawn(fn, world_size=W)` 同时启动 W 个进程跑 fn，每个进程传入自己的 rank。**一份代码，同时在 W 个进程里运行**。

**Setup 模板**：

```python
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "15623"
backend = "nccl" if torch.cuda.is_available() else "gloo"
dist.init_process_group(backend, rank=rank, world_size=world_size)
```

### 5.6 Benchmark 通信（L8 §3）

**模板**：

```python
# Warmup
dist.all_reduce(tensor); torch.cuda.synchronize(); dist.barrier()

# 计时
start = time.time()
dist.all_reduce(tensor); torch.cuda.synchronize(); dist.barrier()
end = time.time()
```

- `torch.cuda.synchronize()`：等**本 rank** 的 kernel 做完
- `dist.barrier()`：等**所有 rank** 到齐

**两个必记带宽公式**：

$$
\text{All-Reduce sent bytes/rank} = 2(W-1) \cdot \text{tensor\_bytes}
$$

$$
\text{Reduce-Scatter / All-Gather sent bytes/rank} = (W-1) \cdot \text{tensor\_bytes}
$$

实测（4×H100, 100M fp32）：All-Reduce ~277 GB/s；Reduce-Scatter ~70 GB/s。**差距远大于 2×** —— 原因：NCCL 对 All-Reduce 有深度优化（SHARP 网内规约等）。**Benchmark 自己的硬件是必要的。**

---

## 第 6 章 Data Parallelism & ZeRO / FSDP（L7 §2, L8 §4）

### 6.1 Naive DDP

每张卡完整复制模型 + optimizer state；每步末尾一次 All-Reduce 梯度。

**PyTorch 代码仅多一行**：

```python
for p in params:
    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
```

**评分**：
- ✅ 算力扩展好
- ❌ 显存完全不扩展

### 6.2 "16× 参数量"的内存预算（混合精度 + Adam）

| 类别 | 字节/参数 |
| --- | --- |
| BF16 权重 | 2 |
| BF16 梯度 | 2 |
| FP32 master weights | 4 |
| Adam $m$ (FP32) | 4 |
| Adam $v$ (FP32) | 4 |
| **合计** | **16** |

**7.5B 模型在 64 卡上**，每卡的显存（近似）：

| 阶段 | 分片内容 | 每卡显存 |
| --- | --- | --- |
| Baseline | 什么都不分 | ~120 GB |
| ZeRO-1 | 仅优化器状态 | ~31.4 GB |
| ZeRO-2 | + 梯度 | ~16.6 GB |
| ZeRO-3 / FSDP | + 参数 | ~1.9 GB |

### 6.3 ZeRO 核心 idea

> **Split up the expensive parts (state) and use the reduce-scatter equivalence.**

### 6.4 ZeRO Stage 1：切优化器状态

rank $k$ 负责参数段 $k$：
1. 前/反向算完整梯度
2. **Reduce-Scatter** 梯度（rank $k$ 只保留段 $k$ 求和梯度）
3. 各 rank 用自己那段 optimizer state **只更新自己那段参数**
4. **All-Gather** 更新后的参数，下一步前向需要完整参数

**通信代价** = RS ($P$) + AG ($P$) = $2P$ = **和 Naive DDP 完全一样** → **Stage 1 是免费的**，有条件就一定要上。

### 6.5 ZeRO Stage 2：再切梯度

**困难**：一次性算完整梯度再 RS，峰值显存仍是完整梯度。

**解法**：**按层触发 —— 一边反向一边分发**。算完一层梯度立刻 RS，立刻释放本地梯度。总字节不变，峰值降到"一层梯度"。

### 6.6 ZeRO Stage 3 = FSDP：连参数也切

**流程**：

**前向**：rank $k$ 只存自己那段参数 → 要算第 $\ell$ 层时 All-Gather 完整权重 → 算完立刻释放 → 激活必须保留

**反向**：按层反向 → 再 All-Gather 权重 → 算梯度 → Reduce-Scatter 梯度（只留自己那段）→ 释放

**通信代价**：
- 前向：1 次 AG ($P$)
- 反向：1 次 AG ($P$) + 1 次 RS ($P$)
- **合计 $3P$**（比 Stage 1/2 多 50%）

**FSDP 为什么没慢爆？—— Overlap Communication & Computation**

**Prefetch**：GPU 做第 $\ell$ 层 matmul 时后台偷偷 AG 第 $\ell+1$ 层权重。计算与通信几乎完全并行。

```
Comm: [AG_0][AG_1 ][AG_2 ][...]
Comp:       [fwd_0][fwd_1][fwd_2][...]
```

**FSDP 的优势**：架构无关，几乎任何网络都能用，不侵入模型代码。

### 6.7 ZeRO 四档对比表（必背）

| 策略 | 参数 | 梯度 | 优化器状态 | 通信 | 难度 |
| --- | --- | --- | --- | --- | --- |
| Naive DDP | 复制 | 复制 | 复制 | $2P$ | ⭐ |
| ZeRO-1 | 复制 | 复制 | **分片** | $2P$ 免费 | ⭐⭐ |
| ZeRO-2 | 复制 | **分片** | 分片 | $2P$ 几乎免费 | ⭐⭐⭐ |
| ZeRO-3 / FSDP | **分片** | 分片 | 分片 | $3P$（overlap 后还好） | ⭐⭐⭐⭐ |

### 6.8 实操：ZeRO 能塞多大模型（单节点 8×A100 80G）

纯 BF16 训练 + Kahan summation（12 bytes/param）：

| 阶段 | 最大模型 |
| --- | --- |
| Baseline | ~6.66 B |
| ZeRO-1 | ~16 B |
| ZeRO-2 | ~24.62 B |
| **ZeRO-3** | **~53.33 B** |

**记忆口诀**：从 baseline 到 ZeRO-3，能塞下的模型尺寸放大 **~8 倍**（8 卡的极限）。

### 6.9 DP 的致命资源：Batch Size

**DP 并行度 ≤ batch size**。而且 batch size 有天花板（**critical batch size** 论文）：超过某临界点后加 batch 无边际收益。

**意义**：
- Batch size 是**有限、可分配的资源**，要在 DP、PP 的 micro-batch、梯度累积中平衡
- **光靠 DP 无法无限扩展** → 必须引入模型并行
- **ZeRO 不解决激活显存** → 激活爆了 FSDP 也救不了

---

## 第 7 章 Model Parallelism（L7 §3, L8 §5-6）

动机：在**不消耗 batch size 资源**的前提下扩展显存，把激活也切开。

**关键区别**（对比 FSDP）：
- FSDP 切参数，运行时临时 gather → **通信参数**
- MP 参数**永久**留在固定的卡上 → **通信激活**，参数从不跨卡搬

两种切法：**PP 切深度**、**TP 切宽度**。

### 7.1 Pipeline Parallel (PP)

**Naive 按层切 —— 灾难**：单样本走完流水线，加了 4 卡却只有 1 卡吞吐。

**Micro-batch 流水（1F1B）**：把 batch 切 $m$ 个 micro-batch，灌入流水。

```
GPU0: |F1|F2|F3|F4|          |B4|B3|B2|B1|
GPU1:    |F1|F2|F3|F4|    |B4|B3|B2|B1|
GPU2:       |F1|F2|F3|F4||B4|B3|B2|B1|
GPU3:          |F1|F2|F3|F4|B3|B2|B1|
```

**Bubble 占比**：

$$
\text{bubble ratio} = \frac{S-1}{m}
$$

（$S$ = pipeline stages，$m$ = micro-batches）

→ **要大 batch 才能喂饱流水线**。

**NVIDIA 实测**：PP=2/4/8 在小 batch 下吞吐崩；batch=128 时几乎无损。

**PyTorch 代码骨架**（L8）：

```python
if rank - 1 >= 0:
    dist.recv(tensor=x, src=rank-1)
for p in local_params:
    x = F.gelu(x @ p)
if rank + 1 < world_size:
    dist.send(tensor=x, dst=rank+1)
```

**PP 通信特性**：只有**点对点传激活**，且是 $b \cdot s \cdot h$ 量级 → 通信量低，可以用慢链路。**PP 最适合放在集群的慢通信层级（跨节点/跨 rack）。**

### 7.2 Zero-Bubble / DualPipe（DeepSeek V3）

**关键洞察**：反向可拆成两步：
- **B（activation gradient）** $\partial L/\partial x$：有依赖链
- **W（weight gradient）** $\partial L/\partial W$：**无下游依赖**，可任意重排

**策略**：B 留在关键路径维持依赖；W 挪到 bubble 空档 → **气泡被彻底填没**。

**代价**：实现极其复杂，侵入 autograd，自定义任务队列。某 lab 的 PP 基础设施只有 2 人能看懂。

### 7.3 Tensor Parallel (TP)

**核心观察**：Transformer 99% 参数和 FLOPs 在 matmul → 拆 matmul 成子 matmul。

**两种拆法**（对 $Y = XA$）：

**按列切 $A = [A_1 | A_2]$**：
$$Y = X[A_1 | A_2] = [XA_1 | XA_2]$$
每卡拿 $A_i$、接收完整 $X$、算一片 $Y$，**不需要通信**就有部分输出。

**按行切 $A = \begin{bmatrix}A_1 \\ A_2\end{bmatrix}$**（配合按列切 $X$）：
$$Y = X_1 A_1 + X_2 A_2$$
每卡算 $X_i A_i$，**最后 All-Reduce 求和**。

**Megatron MLP 的 TP**：$Z = \text{Dropout}(\text{GeLU}(XA) B)$
- $A$ 按**列**切（第一次 matmul 无通信）
- $B$ 按**行**切（第二次 matmul 后 All-Reduce）
- 前向 `g` = All-Reduce；反向对偶

**每层每次前/反向都要一次 All-Reduce 激活** → **极度带宽饥饿**。

**PyTorch 代码骨架**（L8）：

```python
local_D = D // world_size
params = [get_init_params(D, local_D, rank) for _ in range(num_layers)]

for i in range(num_layers):
    x = F.gelu(x @ params[i])   # [B, D] @ [D, D/W] -> [B, D/W]
    activations = [torch.empty(B, local_D, device=...) for _ in range(W)]
    dist.all_gather(tensor_list=activations, tensor=x)
    x = torch.cat(activations, dim=1)
```

### 7.4 TP 的硬规则：**不超过 8 卡**

HuggingFace 实测：

| TP size | 吞吐下降 |
| --- | --- |
| 2–8 | 10–12%（节点内 NVLink 撑得住） |
| 16 | **~42%** |
| 32 | **~65%** |

→ **TP=8 是黄金配置**，装满一个节点；再多就跨节点吃 IB，断崖。

### 7.5 PP vs TP 精确对比

| 维度 | PP | TP |
| --- | --- | --- |
| 切法 | 深度（层） | 宽度（matmul） |
| 主要通信 | 点对点传激活 | 每层 All-Reduce 激活 |
| Bubble | 有（吃 batch） | **无** |
| 消耗 batch size | 是 | **否** |
| 实现复杂度 | 高 | 中 |
| 推荐部署 | 跨节点/跨 rack | **节点内 ≤ 8** |

$$
\text{PP 通信}_{/\text{micro-batch}} \sim b \cdot s \cdot h \quad\text{（点对点）}
$$

$$
\text{TP 通信}_{/\text{layer}} \sim 8 \cdot b \cdot s \cdot h \cdot \frac{W-1}{W} \quad\text{（All-Reduce）}
$$

**TP 通信量远大于 PP，系数差一个数量级。**

### 7.6 DP / TP / PP 三维对比（L8 总结）

| 维度 | DP | TP | PP |
| --- | --- | --- | --- |
| 切什么 | batch (N) | width (hidden) | depth (layer) |
| 参数复制？ | 是 | 否（切掉） | 否 |
| 通信时机 | 反向结束 AR grads | 每层 AG 激活 | 逐层 send/recv 激活 |
| 对带宽要求 | 中 | **高** | **低** |
| 主要开销 | 参数全复制 → 显存 | 每层通信 → 带宽 | pipeline bubble → 时间 |
| 适合层级 | 节点间 | **节点内** | **跨节点** |

---

## 第 8 章 Activation Parallelism & Sequence Parallel（L7 §4）

### 8.1 激活显存 —— 大模型的真正瓶颈

**PyTorch profiler 典型显存曲线**：
- 参数：全程常驻
- 优化器状态：第一次 .step() 后常驻
- **激活：前向单调上升**（每层要缓存给反向）
- 梯度：反向逐层累积
- **峰值在反向中段**

**即使 TP/PP 把参数优化器状态都分了，激活仍随模型规模增长**。因为**激活里有一部分不走 matmul 也不能被 TP 切开**。

### 8.2 Korthikanti 2022 每层激活公式（必记）

**无任何并行**：

$$
M_\text{act}^\text{per layer} = 34\,sbh + 5\,as^2 b
$$

- **$34\,sbh$**：MLP 和 pointwise op
- **$5\,as^2 b$**：attention 的序列长度平方项

（$s$=序列，$b$=batch，$h$=hidden，$a$=heads）

**Flash Attention 直接干掉第二项。**

**做了 TP 后**：

$$
M_\text{act}^\text{TP} = \frac{24\,sbh + 5\,as^2 b}{t} + 10\,sbh
$$

最后 $10\,sbh$ **没被 $t$ 除**，来自：

| 项 | 贡献 |
| --- | --- |
| LayerNorm | $4\,sbh$ |
| Dropout | $2\,sbh$ |
| Attention 和 MLP 的残差输入 | $4\,sbh$ |

### 8.3 Sequence Parallel：切开边角料

**观察**：LayerNorm / Dropout 等 pointwise op 在**序列维度上完全独立** → 第 $t$ 个 token 的 LN 不依赖其他 token。

**做法**：把序列长度 $s$ 切 $t$ 段给 TP 卡，各自做自己那段的 LN / Dropout。

**同步原语**：
- 前向 SP → TP：**All-Gather**（序列拼回来做 matmul）
- 前向 TP → SP：**Reduce-Scatter**（TP 块要 AR 的结果直接分发）
- 反向：两者互换

**奇妙之处**：原本 TP 要做的 All-Reduce 被**等价拆成 Reduce-Scatter + All-Gather**（§5.4 恒等式），**总字节不变，免费获得 SP 并行**。

### 8.4 激活显存最终公式（TP + SP + Flash）

$$
\boxed{M_\text{act}^\text{TP+SP+Flash} = \frac{34\,sbh}{t}}
$$

- TP 切 matmul 部分
- SP 切 pointwise 边角料
- Flash 干掉 $s^2$ 项

**这是各种 transformer 显存估算里"$\sim 34\,sbh/t$ per layer"的来源。**

### 8.5 Activation Recomputation

反向时重跑前向重建激活。**约 +33% FLOPs**，但能让更大 batch 装下 → 减小 PP bubble / 喂饱 DP → **经常净赚**。

---

## 第 9 章 组合策略与 Rule of Thumb（L7 §6）

### 9.1 四个有限资源（必须同时平衡）

| 资源 | 谁消耗它 |
| --- | --- |
| 显存 | 参数 + 梯度 + 优化器状态 + 激活 |
| 带宽（节点内 / 间 / rack） | 所有集合通信 |
| 算力（FLOPs） | 所有计算 |
| **Batch size**（有上限） | DP 数 + PP micro-batch 数 |

### 9.2 策略权衡总表

| 策略 | Sync 开销 | Memory | Bandwidth | Batch 影响 | 易用性 |
| --- | --- | --- | --- | --- | --- |
| DDP / ZeRO-1 | Per-batch | 不扩展 | $2P$ | 线性占用 | **非常易** |
| FSDP (ZeRO-3) | 3× per-block | **线性扩展** | $3P$ | 线性占用 | **非常易** |
| Pipeline | Per-pipeline | 线性扩展 | 激活（点对点） | 线性占用 | 不易 |
| **Tensor + SP** | 2× transformer block | 线性扩展 | $8\times$ 激活/层 AR | **不占 batch** | 不易 |

**核心观察**：
- 没有完美策略，各有 trade-off
- **TP+SP 独特价值**：不吃 batch size
- **FSDP 独特价值**：易用，无需改架构
- **PP 独特价值**：带宽需求最低，可跨慢链路

### 9.3 经典 Rule of Thumb（面试/实战必背）

**Step 1**：直到模型（+激活）能塞进显存：
- **TP ≤ 8** 在节点内（NVLink）
- **PP 或 ZeRO-3** 跨节点（带宽不足时）

**Step 2**：直到 GPU 用完：
- 剩余 GPU 全做 **Data Parallel（FSDP）**，最抗高延迟

**Step 3**：batch 太小：
- **Gradient Accumulation**，用大等效 batch 换同步频次

| 步骤 | 动作 | 原因 |
| --- | --- | --- |
| ① 装下模型 | 节点内 **TP ≤ 8** | TP 最吃带宽 |
| ② 装下模型 | 节点间 **PP 或 ZeRO-3** | 能跨慢链路 |
| ③ 扩 GPU 规模 | 剩余全做 **DP** | DP 最抗高延迟 |
| ④ 通信优化 | **梯度累积** | batch 紧时降同步频率 |

### 9.4 Llama 3 带宽优先级（部署手册）

按"对带宽的饥饿程度"排序，越饥饿越要部署在越快的链路上：

$$
\underbrace{\text{TP}}_{\text{节点内 NVLink}} > \underbrace{\text{CP}}_{\text{次高}} > \underbrace{\text{PP}}_{\text{可容忍}} > \underbrace{\text{DP}}_{\text{最能忍高延迟}}
$$

---

## 第 10 章 其他并行策略（简介）

### 10.1 Context Parallel / Ring Attention

用于**长上下文**训练。每卡持一段 query，keys/values 在卡间环状传递，配合 Flash Attention 的 online tiling 逐块算 attention。

在几十 K ~ 上百 K token 的上下文上**不可或缺**。

### 10.2 Expert Parallelism (MoE)

每卡放不同 expert。MoE 层两次 all-to-all（dispatch + combine）。

- 与 TP 相似：都是切 MLP
- 与 TP 不同：**稀疏激活 + 路由**，通信模式不规则，有负载不均 / 路由碰撞问题

---

## 第 11 章 真实大模型案例（L7 §7）

### 11.1 Megatron-LM (Narayanan 2021)

1.7B 到 1T 参数，3D 并行（TP + PP + DP），**MFU 40–52%**。

**DP size 演化（10 档模型）**：$32, 32, 32, 32, 32, 32, 24, 15, 9, 6$

- **TP**：从 1 起，**上到 8 就封顶**
- **PP**：模型越大越多，用来塞模型
- **DP**：模型大时被 PP 挤压，从 32 降到 6

**三个结论**：
1. 近线性扩展（perfect 3D parallelism 给出几乎平直 per-GPU 利用率）
2. **TP=8 黄金配置**（64 机最优是 TP=8 × PP=8）
3. Activation recomputation 自己挣回成本

### 11.2 近期模型配方

| 模型 | 并行策略 |
| --- | --- |
| OLMo 7B | 纯 **FSDP** |
| DeepSeek V1 | **ZeRO-1 + TP + SP + PP** 教科书级 |
| **DeepSeek V3** | **PP=16 + EP=64 + ZeRO-1**（罕见：**不用 TP**） |
| Yi-Lightning 2025 | ZeRO-1 + **EP 替换 TP** + PP |
| Llama 3 405B | 分阶段（主预训练 → 长上下文引入 CP） |
| Gemma 2 | ZeRO-3 + MP (TP+SP) + DP（TPU 网格） |

### 11.3 超大规模的现实：故障

Llama 3 训练中：
- **148 次** GPU 硬件故障中断（~30%）
- **32 次** 非计划停机
- 更怕 **silent data corruption**（GPU 偷偷出错不报）

**几万卡级训练必须有 fault-tolerant 架构。**

---

## 第 12 章 跨讲关联 & 核心原则

### 12.1 "数据移动是瓶颈"在四讲中的一致体现

| 层级 | 表现 | 对策 |
| --- | --- | --- |
| SM 内部（L5 L6） | SRAM ↔ 寄存器 | Shared memory 复用 |
| 单 GPU（L5 L6） | HBM ↔ SRAM | Fusion、Tiling、Flash Attention |
| 节点内（L7 L8） | NVLink | TP、节点内 FSDP |
| 节点间（L7 L8） | InfiniBand | PP、跨节点 FSDP |
| 跨 rack（L7） | Leaf/Spine | DP（最抗延迟） |

每一层级都是同样的原则：**组织计算，让热数据留在近处，少搬远处**。

### 12.2 "三选一"在每一层的体现

| 层级 | 重算 | 存本地 | 存异地通信 |
| --- | --- | --- | --- |
| 单 GPU | Recomputation（激活重算） | 存 HBM | — |
| Attention | Flash Attention 重算 softmax | 存 $N^2$ 矩阵 | — |
| 多 GPU | Activation checkpointing | 完整复制（Naive DDP） | ZeRO / TP / PP |

### 12.3 All-Reduce = Reduce-Scatter + All-Gather 的反复应用

| 位置 | 具体用法 |
| --- | --- |
| ZeRO-1/2 | DDP 的 AR 拆成 RS + AG，中间插参数更新 |
| ZeRO-3/FSDP | 前向用 AG，反向用 RS + AG |
| Sequence Parallel | TP 的 AR 拆成 RS（SP 段）+ AG（下一 TP 段） |

**同一条恒等式，在三个地方免费省出功能。**

### 12.4 Batch Size 是稀缺资源

| 要 batch 的地方 | 用处 |
| --- | --- |
| DP / FSDP | 并行度 = batch 切片数 |
| PP | micro-batch 数 $m$，减小 bubble |
| 梯度累积 | 增大等效 batch |

**Critical batch size 是硬上限**，所以 batch size 要在这三者间精心分配。

### 12.5 "FLOPs ≠ Runtime" 的四讲演进

| 讲 | 表现 |
| --- | --- |
| L3 | Norm 占 0.17% FLOPs 但 25% runtime（memory-bound） |
| L5 | Wave quantization、tile quantization 导致 FLOPs 相同但时间差数倍 |
| L6 | Fusion 把 FLOPs 不变的计算加速 8× |
| L7 | TP 的 FLOPs 完美 $1/t$，但通信代价让 TP>8 崩溃 |

**永远不要只盯着 FLOPs。**

---

## 第 13 章 公式速查表

| 公式 | 含义 | 出处 |
| --- | --- | --- |
| `arithmetic_intensity = FLOPs / bytes moved` | 算术强度 | L5 L6 |
| `coalesced load: 同 warp 32 threads → 1 burst` | 合并访存 | L5 |
| `tiled matmul: global 读从 N 降到 N/T` | Tiling 收益 | L5 |
| Online softmax: $m_j = \max(m_{j-1}, x_j)$，$d_j = d_{j-1}\exp(m_{j-1}-m_j) + \exp(x_j - m_j)$ | 增量 softmax | L5 |
| `All-Reduce = Reduce-Scatter + All-Gather` | 恒等式 | L7 L8 |
| `All-Reduce bytes/rank = 2(W-1) × tensor_bytes` | AR 带宽 | L8 |
| `RS / AG bytes/rank = (W-1) × tensor_bytes` | RS/AG 带宽 | L8 |
| `Adam: 16 bytes/param` | DP 显存瓶颈 | L7 |
| PP bubble ratio = $(S-1)/m$ | PP 气泡占比 | L7 |
| $M_\text{act}^\text{per layer} = 34sbh + 5as^2b$ | 激活公式（Korthikanti） | L7 |
| $M_\text{act}^\text{TP+SP+Flash} = 34sbh/t$ | 最终激活公式 | L7 |
| TP 通信/层 $\sim 8bsh(W-1)/W$ | TP 带宽 | L7 |
| PP 通信/micro-batch $\sim bsh$ | PP 带宽 | L7 |

---

## 第 14 章 常考 / 常问问题（自测）

这些问题如果能**不看资料**流畅回答，说明这四讲掌握了：

### GPU & Kernel
1. Warp 为什么是 32？Control divergence 怎么发生？怎么避免？
2. DRAM burst mode 是什么？Row-major matmul 的两种线程布局哪种快？差多少？
3. Tiling 怎么减少 global memory 访问？数学上从 $N$ 降到多少？
4. Wave quantization 是什么？为什么 1792×1792 比 1793×1793 快？（给出数字）
5. Operator fusion 的工厂/仓库比喻是什么？`torch.compile` 能自动做什么？
6. Recomputation 和 Activation Checkpointing 是同一技术吗？目的有何不同？

### Flash Attention
7. Flash Attention 快的本质是什么？Compute 复杂度变了吗？
8. Online softmax 的递推公式是什么？为什么不需要先见到所有 $x$？
9. Flash Attention 集成了哪三大技巧？HBM 访问从 $O(?)$ 降到 $O(?)$？

### Triton
10. Triton vs CUDA 的本质区别是什么？Triton 自动帮你做了哪三件事？
11. Triton softmax 为什么用"一个 block 处理一整行"？`other=float("-inf")` 的用处？
12. `torch.compile` 能搞定什么？搞不定什么？

### 通信原语
13. All-Reduce 的通信代价是 $2P$ 而不是 $P$，为什么？
14. 写出 All-Reduce = Reduce-Scatter + All-Gather，解释它为什么免费。
15. `torch.cuda.synchronize()` 和 `dist.barrier()` 各自的作用？缺一个会怎样？

### Data Parallel / FSDP
16. Adam 为什么每参数 16 bytes？展开每一项。
17. ZeRO Stage 1 为什么"免费"？通信量和 Naive DDP 比是多少？
18. FSDP 通信量 $3P$ 比 DDP 的 $2P$ 多 50%，为什么没慢爆？
19. 单节点 8×A100 80G，纯 BF16 训练能塞多大模型？（ZeRO-3）
20. FSDP 不解决什么？Critical batch size 是什么概念？

### Model Parallel
21. PP 的 bubble ratio 公式？为什么需要大 batch？
22. Zero-bubble / DualPipe 的核心洞察是什么？（B vs W）
23. Megatron MLP 的 TP 怎么切 $A$ 和 $B$？每层要几次 All-Reduce？
24. 为什么 TP 硬规则是 ≤ 8 卡？
25. PP、TP、DP 分别应该部署在集群的哪个层级？为什么？

### Activation Parallel
26. Korthikanti 激活公式的两项分别代表什么？Flash Attention 干掉哪项？
27. TP 后还剩 $10\,sbh$ 没被切，这些是什么？
28. Sequence Parallel 为什么"免费"？（提示：Reduce-Scatter + All-Gather）
29. 最终激活公式 $34\,sbh/t$ 是怎么来的？

### 组合策略
30. 3D Parallelism rule of thumb 的四步是什么？
31. Llama 3 的带宽优先级排序是什么？为什么这样排？
32. DeepSeek V3 为什么不用 TP？（提示：EP + PP）

---

## 第 15 章 下一阶段预告（L9+）

- **L9 Scaling Laws**：Chinchilla optimal（$N \sim 20 \times D$）、小规模→大规模外推、IsoFLOP / 如何指导"训多大模型用多少数据"
- **后续**：推理优化（speculative decoding / continuous batching）、数据（Common Crawl 处理、filter、dedup）、Alignment（SFT、DPO、GRPO）

**L5-8 这个 Systems 模块搭好的是**："给定大模型架构 → 在真实 GPU 集群上训得又快又稳"的完整工具链。后面的课会**用 Scaling Laws 决定训什么规模**、**用数据课决定训什么内容**、**用 Alignment 决定怎么让 base model 变有用**。

---

## 结语：Systems 模块的一句话心法

> **所有性能瓶颈本质都是 memory movement。从 SM 到跨 rack，层级不同但原则一致：组织计算，让热数据留在近处。GPU 给你的唯一快路径是 matmul，给你的唯一免费路径是 residual identity 和 Reduce-Scatter + All-Gather 恒等式。把这三点刻在脑子里，后面看任何大模型训练方案都能秒懂它在省什么。**
