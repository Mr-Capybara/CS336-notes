
# CS336 Lecture 6: Kernels & Triton — 详细笔记

> 本笔记配合 [`lecture_06.py`](./lecture_06.py) 阅读，可无损替代视频。脚本的函数调用顺序即为课堂讲解顺序：`main()` → `review_of_gpus()` → `benchmarking_and_profiling()` → `kernel_fusion_motivation()` → `cuda_kernels()` → `triton_kernels()` → `pytorch_compilation()` → `triton_softmax_main()`。

---

## 0. 本讲目标与总览

本讲的核心主题是：**让 GPU 跑得更快**。具体要解决两个层次的问题：

1. **怎么知道慢在哪**：基准测试（benchmarking）+ 性能剖析（profiling）。
2. **怎么让它更快**：用 CUDA / Triton 写自定义 kernel，或者直接用 `torch.compile` 让编译器帮你做。

课堂贯穿的一条主线：**从最高层的 PyTorch API 一路向下，直到 PTX 汇编**，看清楚每一层究竟在做什么。最终的写作工具箱有 5 种：

- 手写朴素 PyTorch（`manual_gelu`）
- 调 PyTorch 原生 fused 实现（`pytorch_gelu`）
- `torch.compile` 自动 fuse（`compiled_gelu`）
- 手写 CUDA kernel（`create_cuda_gelu`）
- 手写 Triton kernel（`triton_gelu`）

三个代表性算子：
- **GeLU**（逐元素，element-wise）—— 最简单
- **softmax**（逐行 reduction）—— 中等
- **matmul**（复杂聚合）—— 最复杂

---

## 1. GPU 执行模型回顾

对应脚本函数：`review_of_gpus()` / `print_gpu_specs()`。

### 1.1 硬件结构

一块 A100 的典型参数：

| 组件 | 规模 | 特性 |
|---|---|---|
| SM（Streaming Multiprocessor） | 108 个 | 实际算力单元 |
| DRAM（global memory） | 80 GB | 大但慢 |
| L2 cache | 40 MB | 中等 |
| L1 / Shared Memory | 192 KB per SM（其中 shared 可达 164 KB） | 小但快，与 L1 同速 |
| Register File | 每个 thread 独占一部分 | 最快 |

**关键原则**：越靠近 SM 的存储越小越快。高性能 kernel 的核心就是**尽量让数据留在寄存器和 shared memory 里**，少往返 DRAM。

### 1.2 执行模型：Thread / Block / Grid

- **Thread**：最小执行单元，处理 `f(i)` 中的一个 `i`。
- **Thread Block**（CTA）：一组 thread，**调度到同一个 SM 上**。同一 block 内的 thread 可通过 shared memory 通信、可同步。
- **Grid**：所有 block 的集合。**跨 block 不能同步**，也不能共享 shared memory。

**为什么要有 block 这一层？** —— 为了让需要共享数据的 thread 聚在同一个 SM 内，利用快速的 shared memory。

### 1.3 Warp（课堂 Q&A 补充）

Warp 是 **32 个 thread 为一组**的执行单元，由硬件统一调度。存在的意义是**省控制逻辑**：一个 warp scheduler 控制 32 个 thread，不需要给每个 thread 配一个。这是 GPU 相对 CPU 的核心权衡——CPU 大量硅片用于分支预测/控制，GPU 把硅片让给计算单元。

### 1.4 Wave Quantization（波次量化）

block 按批（wave）被调度到 SM 上。如果总 block 数不是 SM 数的整数倍，最后一波会有部分 SM 空闲，拉低 occupancy。

**经验法则**：block 数应 ≥ 4× SM 数，并尽量让 block 数能整除 SM 数。

### 1.5 Arithmetic Intensity（算术强度）

$$
\text{arithmetic intensity} = \frac{\text{\# FLOPs}}{\text{\# bytes moved}}
$$

- 高 → **compute-bound**（好，算力在真干活）
- 低 → **memory-bound**（坏，算力在等内存）

**一般规律**：matmul 是 compute-bound，**其他几乎都是 memory-bound**。因此优化的主战场是减少内存搬运。

---

## 2. Benchmarking：测端到端耗时

对应脚本函数：`benchmarking()` / `benchmark()`。

> 核心信念：**每改一处都要 benchmark**。不要凭感觉猜瓶颈。

### 2.1 Benchmark 的两个必做动作

看 `benchmark()` 函数实现：

```python
def benchmark(description, run, num_warmups=1, num_trials=3):
    for _ in range(num_warmups):   # (1) warmup
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()   # (2) 同步

    times = []
    for trial in range(num_trials):
        start = time.time()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # 每次计时末尾也要同步
        end = time.time()
        times.append((end - start) * 1000)
    return mean(times)
```

**两个必须做的事**：

1. **Warmup**：第一次调用会触发 JIT 编译、kernel 加载、cache 未命中等一次性开销。我们关心的是稳态性能。
2. **`torch.cuda.synchronize()`**：CPU 和 GPU 是**独立的计算设备**。Python 调 `a @ b` 时，CPU 只是把 kernel 入队然后立即返回，**GPU 还在后台跑**。不同步就计时，会得到"大矩阵乘瞬间完成"这种荒谬结果。

### 2.2 MatMul 扩展性实验

`dims = (1024, 2048, 4096, 8192, 16384)`，做 `a @ b`。

**观察**：
- 小尺寸（1024, 2048）时间不怎么涨 —— 被**启动开销/kernel launch 固定开销**主导。
- 尺寸足够大后，时间呈**超线性**增长（理论上 $O(N^3)$），符合预期。

### 2.3 MLP 扩展性实验

基线：`dim=256, num_layers=4, batch_size=256, num_steps=2`。

分别把 `num_steps / num_layers / batch_size / dim` 乘 2、3、4、5 倍：
- **steps / layers / batch_size** 的缩放 → 近似**线性增长**。
- **dim** 的缩放 → 超线性（因为每层是矩阵乘）。

> 备注：作者建议生产代码里可以直接用 `torch.utils.benchmark`，功能更全。本讲为了透明演示，自己写了一个。

---

## 3. Profiling：看时间花在哪

对应脚本函数：`profiling()` / `profile()`。

Benchmark 只给出端到端时间，**不告诉你时间花在哪**。Profiler 能一路打开到 CUDA kernel 级别。

### 3.1 `torch.profiler` 用法

```python
with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=with_stack,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
    run()
    torch.cuda.synchronize()

table = prof.key_averages().table(sort_by="cuda_time_total", ...)
```

### 3.2 逐个算子的拆解观察

#### `add`（`a + b`, dim=2048）

Profiler 会展示：
- `aten::add`：PyTorch C++ 层 wrapper
- `vectorized_elementwise_kernel<..., native::CUDAFunctor_add<...>>`：真正干活的 CUDA kernel
- `cudaLaunchKernel`：CPU 把 kernel 送入 GPU 队列的开销
- `cudaDeviceSynchronize`：等 GPU 跑完

CPU 端开销其实不小（~1.4 ms），GPU 端极短（~17 μs）。

#### `matmul` (dim=2048)

- `aten::mm` → dispatch 到 **cutlass**（NVIDIA 的 CUDA 线代库）
- 真正的 kernel 名长得像：`cutlass_80_simt_sgemm_256x128_8x4_nn_align1`
  - `cutlass`：库名
  - `256x128`：tile size
  - `sgemm`：single precision GEMM
  - `nn`：两个矩阵都不转置

#### `matmul` (dim=128)

**小矩阵走的是完全不同的 kernel**（不经过 cutlass，直接 `XMMA GEMM` 之类）。这说明：

> PyTorch 的 matmul 在底层会**根据形状和硬件分派到不同的 kernel**。这也是为什么 `torch.compile` 的 autotune 能白给 ~10% 提升——它会实测选最优 kernel。

#### `cdist`（pairwise 欧氏距离）

复合算子，会分解成多个 primitive：`aten::mm`（78%）+ `aten::pow`（5%）+ `aten::sum`（3%）+ copy/concat（6%）。这种拆解能直接告诉你该优化 matmul。

#### `gelu`（`torch.nn.functional.gelu(a + b)`）

`gelu` 是 fused kernel，不会展开成 `tanh / pow / mul` 一堆 primitive，而是一个专门写好的 CUDA kernel。softmax 同理。**常用算子几乎都有 fused kernel**。

### 3.3 Nsight Systems：工业级 profiler

对应代码：`run_mlp(..., with_stack=True)` + `prof.export_stacks(...)`。

Nsight 能看到 torch profiler 看不到的东西：

#### CPU / GPU 双泳道时间线

Nsight 把 CPU 线程活动和 CUDA 硬件活动分两行显示。可以用 NVTX 标注代码区段（`range_push` / `range_pop`），对应到时间线上。

#### 关键观察：CPU 一直在"跑超前"

CPU 并不等 GPU。它把 kernel 不断入队：当 CPU 已经跑到 `step=2` 的 forward 时，GPU 可能还在跑 `step=0`。这是正常的、也是**为什么 Python 这种慢语言做训练仍能打满 GPU** 的根本原因——CPU 永远不是瓶颈。

直到 GPU 的命令队列满了，CPU 才会停下来等。

#### 陷阱：`print(loss)` 会破坏异步

在训练 loop 里打印 loss 会强制 CPU 等 GPU：
- CPU 要拿到 loss → 需要 GPU 算完 → 触发 `cudaStreamSynchronize`
- CPU 停下来等 → 失去"跑超前"的优势 → 极端情况下 GPU 出现空泡

从 Nsight 上能看到：原本 CPU 的 step0/step1/step2 是连在一起的，加了 print 后 CPU 的每个 step 都和 GPU 的 step 对齐（被拖住了）。

---

## 4. Kernel Fusion 动机

对应脚本函数：`kernel_fusion_motivation()`。参考 Horace He 的博文（warehouse / factory 比喻）。

### 4.1 仓库/工厂比喻

- **DRAM = 仓库**（大，远）
- **SRAM/寄存器 = 工厂**（小，近，快）

每个算子都要：**从仓库取料 → 工厂加工 → 运回仓库**。

如果你朴素地串 5 个算子，就是 5 次来回搬运：

```
读 → op1 → 写 → 读 → op2 → 写 → 读 → op3 → 写 → ...
```

**Fusion**：把多个算子塞进同一个 kernel，**一次取料 → 连续加工 → 一次回仓**：

```
读 → op1 → op2 → op3 → ... → 写
```

### 4.2 用 GeLU 验证

tanh 近似的 GeLU 公式：

$$
\operatorname{GeLU}(x) \approx 0.5\,x\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\,x^3)\right)\right)
$$

脚本里两种写法：

```python
def pytorch_gelu(x):
    return torch.nn.functional.gelu(x, approximate="tanh")

def manual_gelu(x):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
```

（`0.79788456 ≈ sqrt(2/pi)`。用 `approximate="tanh"` 是为了让两者数值完全对得上。）

### 4.3 实测结果（dim=16384）

| 实现 | 耗时 |
|---|---|
| `manual_gelu` | ~8.1 ms |
| `pytorch_gelu` | ~1.1 ms |

**约 8× 加速，全部来自 kernel fusion**。Profiler 里能看到：
- `manual_gelu`：多次 `vectorized_elementwise_kernel`（乘法至少被拆成 3 次 kernel），还有 `tanh`、`add` 等。
- `pytorch_gelu`：**一个**融合 kernel 吞下全部。

---

## 5. 手写 CUDA Kernel

对应脚本函数：`cuda_kernels()` / `create_cuda_gelu()`。同目录下 [`gelu.cu`](./gelu.cu) 是 CUDA 源码。

### 5.1 CUDA 编程模型

- CUDA 是 C/C++ 的扩展，加了管理 GPU 的 API。
- 简化心智模型：**写 `f(i)`，CUDA 在所有 `i` 上并行跑**。
- 你用 `(blockIdx, blockDim, threadIdx)` 算出当前 thread 该处理哪个 `i`。

调试时务必：

```python
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

否则错误消息是异步抛的，几乎无法定位。

### 5.2 `torch.utils.cpp_extension.load_inline`

`create_cuda_gelu()` 的精彩之处：**直接在 Python 里把 CUDA 源码字符串编译成 Python 可调用模块**，不用单独的 setup.py / Makefile。

```python
module = load_inline(
    cuda_sources=[cuda_gelu_src],
    cpp_sources=["torch::Tensor gelu(torch::Tensor x);"],
    functions=["gelu"],
    extra_cflags=["-O2"],
    name="inline_gelu",
    build_directory="var/cuda_gelu",
)
cuda_gelu = module.gelu
```

### 5.3 典型 CUDA kernel 结构（两段式）

```cpp
// ============ GPU 侧：kernel 本体 ============
__global__ void gelu_kernel(const float* in, float* out, int num_elements) {
    // 当前线程在全局向量中的坐标
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查，末尾 block 可能越界
    if (i < num_elements) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

// ============ CPU 侧：wrapper ============
torch::Tensor gelu(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda());      // 必须在 GPU
    TORCH_CHECK(x.is_contiguous());          // 必须连续

    torch::Tensor y = torch::empty_like(x);  // 用 empty_like，不要 zeros
    int n = x.numel();
    int block_size = 1024;
    int num_blocks = (n + block_size - 1) / block_size;   // 向上取整

    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(),
                                            y.data_ptr<float>(), n);
    return y;
}
```

**关键点**：

1. **`__global__`** 标记一个 CUDA kernel（可从 CPU 调、在 GPU 跑）。
2. **坐标计算**：`i = blockIdx.x * blockDim.x + threadIdx.x`。kernel 自己不知道 `i`，必须自己算出来。
3. **边界检查**：`if (i < num_elements)`。几乎所有 CUDA kernel 都有这个模式。
4. **`empty_like` 而非 `zeros_like`**：反正要全写一遍，没必要先清零。
5. **contiguous 检查**：kernel 按线性内存地址索引，非连续张量（比如 transpose 后）会算错。调用方必要时先 `.contiguous()`。
6. **ceil-div 算 block 数**：`(n + b - 1) / b`，保证末尾元素被覆盖。

### 5.4 实测（dim=16384）

| 实现 | 耗时 |
|---|---|
| `manual_gelu`（朴素 PyTorch） | ~8.1 ms |
| `pytorch_gelu`（官方 fused） | ~1.1 ms |
| `cuda_gelu`（我们手写的） | ~1.8 ms |

手写 CUDA 一下就追到 PyTorch 的 60% 水平，而且 profiler 里只剩**一个 kernel launch** 占掉 100% GPU 时间——fusion 目标达成。

### 5.5 何时值得写 CUDA？

- **Element-wise** 操作好写，直接套模板。
- **需要多值聚合**（matmul、softmax、RMSNorm、flash attention）的 kernel 要操心 shared memory、同步，复杂得多。
- 大多数情况下 **Triton 是更好的起点**。

---

## 6. Triton：用 Python 写 GPU Kernel

对应脚本函数：`triton_kernels()` / `triton_introduction()` / `triton_gelu_main()` / `triton_gelu_kernel`。

### 6.1 Triton 的定位

OpenAI 2021 年发布。**让你以 thread block 为单位思考**（而不是单个 thread），编译器自动处理很多底层苦活。

|  | CUDA | Triton |
|---|---|---|
| Memory coalescing (DRAM 搬运) | 手动 | 自动 |
| Shared memory 管理 | 手动 | 自动 |
| SM 内调度 | 手动 | 自动 |
| SM 间调度 | 手动 | 手动 |

编译器多干活 → 甚至可能超过 PyTorch 手写实现。

### 6.2 Triton GeLU 实现

Wrapper（CPU 侧）：

```python
def triton_gelu(x):
    assert x.is_cuda and x.is_contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(n, block_size)

    triton_gelu_kernel[(num_blocks,)](x, y, n, BLOCK_SIZE=block_size)
    return y
```

Kernel（GPU 侧）：

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                 # 当前 block 的 id
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # 一整个向量！
    mask = offsets < num_elements

    x = tl.load(x_ptr + offsets, mask=mask)     # 向量化 load

    # Triton 里没有 tanh，用 tanh(a) = (e^{2a}-1)/(e^{2a}+1) 展开
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    tl.store(y_ptr + offsets, y, mask=mask)
```

**和 CUDA 对比的关键差别**：

- CUDA kernel 里 `i` 是**标量**（单个 thread 的位置）。
- Triton kernel 里 `offsets` 是**向量**（整个 block 的位置），`x` 也是向量。你在 block 级别思考，向量化操作交给编译器拆。
- `mask` 机制替代 CUDA 的 `if (i < n)`。
- `tl.constexpr` 的参数会参与编译（每个 `BLOCK_SIZE` 生成一个专门的 kernel）。

### 6.3 看生成的 PTX

对应函数：`print_ptx_main()` / `print_ptx()`。

PTX = Parallel Thread eXecution，**GPU 的汇编语言**。Triton 会把 Python kernel 编译成 PTX。

```python
ptx = list(kernel.cache[0].values())[0].asm["ptx"]
```

PTX 片段里能看到的东西：

| 指令/寄存器 | 含义 |
|---|---|
| `ld.global.*` / `st.global.*` | 从 DRAM 读 / 写 |
| `%ctaid.x` | block index（CTA = Cooperative Thread Array） |
| `%tid.x` | thread index |
| `%f*` | 浮点寄存器 |
| `%r*` | 整数寄存器 |
| `%rd*` | 64-bit 地址寄存器 |
| `mul.f32` / `fma.f32` | 浮点运算 |

**一个值得注意的优化**：每个 thread 一次处理 **8 个元素**（ld.global 一次读 4 个，thread coarsening 又放大 2 倍），这是 Triton 编译器自动做的**线程粗化**（thread coarsening）。

> 注意：`TRITON_INTERPRET=1` 时 Triton 以解释模式运行，可逐行 debug 但**不生成 PTX 也不快**。benchmark 前记得关掉。

### 6.4 Triton vs CUDA 性能

| 实现 | 耗时 |
|---|---|
| `manual_gelu` | ~8.1 ms |
| `pytorch_gelu` | ~1.1 ms |
| `cuda_gelu` | ~1.84 ms |
| `triton_gelu` | ~1.85 ms |

Triton 和手写 CUDA **基本持平**，但代码量小得多、可以在 Python 里 debug。对这个简单 case，Triton 的 block 级优化空间没充分发挥。

---

## 7. `torch.compile`：让编译器自己搞定

对应脚本函数：`pytorch_compilation()`。

```python
compiled_gelu = torch.compile(manual_gelu)
```

就这一行，`torch.compile` 会把朴素的 `manual_gelu` 自动 **fuse** 成一个 Triton kernel。

实测（dim=16384）：

| 实现 | 耗时 |
|---|---|
| `manual_gelu` | ~8.1 ms |
| `pytorch_gelu` | ~1.1 ms |
| `cuda_gelu` | ~1.8 ms |
| `triton_gelu` | ~1.85 ms |
| `compiled_gelu` | ~1.47 ms |

**compile 反而比我们手写的 Triton 还快**——它生成的 Triton 代码比我们的更优化。profiler 里也是单个 fused kernel（带 `add_mul_tanh` 之类的合成名）。

### 7.1 什么时候 `torch.compile` 还不够？

- **能搞定的**：算子融合、已知形状下的 matmul 选核。这些场景基本别想超越它。
- **搞不定的**：Flash Attention 2 / 3 这类**算法级重写**（online softmax、跨 SM 的 warp specialization、H100-specific 优化）。这是需要你亲自下场写 Triton 的场合。

**建议心态**：
- 默认：`torch.compile`。
- 遇到 profiler 显示的明确瓶颈且确信能算法级改写时：写 Triton。
- 几乎永远不需要从零写 CUDA。

---

## 8. 案例：Triton Softmax（含 reduction）

对应脚本函数：`triton_softmax_main()` / `manual_softmax()` / `triton_softmax()` / `triton_softmax_kernel`。

前面都是 element-wise，现在升级到**需要跨元素聚合的 reduction**。

### 8.1 朴素实现的内存代价

```python
def manual_softmax(x):
    M, N = x.shape
    x_max = x.max(dim=1)[0]           # MN 读, M 写
    x = x - x_max[:, None]            # MN + M 读, MN 写
    numerator = torch.exp(x)          # MN 读, MN 写
    denominator = numerator.sum(dim=1)  # MN 读, M 写
    y = numerator / denominator[:, None]  # MN 读, MN 写
    return y
```

**合计：$5MN + M$ 读，$3MN + 2M$ 写**。

理想下界：**$MN$ 读 + $MN$ 写**（读一次 x，写一次 y）。朴素实现多出约 **4×** 的内存流量——这就是 fusion 的理论加速上限。

### 8.2 Triton Softmax 设计

**核心决策：一个 block 处理一整行**。

- 每行所有元素都能塞进一个 block（前提：列数 ≤ SM shared mem 容量）。
- 一个 block 内部可以自由做 max / sum 聚合。
- block 之间没有依赖 —— 完美并行。

Wrapper：

```python
def triton_softmax(x):
    y = torch.empty_like(x)
    M, N = x.shape
    block_size = triton.next_power_of_2(N)   # 行填充到 2 的幂
    num_blocks = M                            # 每行一个 block

    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size,
    )
    return y
```

Kernel：

```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride,
                          num_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 读一整行
    x_row_ptr = x_ptr + row_idx * x_row_stride + col_offsets
    x_row = tl.load(x_row_ptr, mask=col_offsets < num_cols, other=float("-inf"))

    # 数值稳定 softmax
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # 写回
    y_row_ptr = y_ptr + row_idx * y_row_stride + col_offsets
    tl.store(y_row_ptr, y_row, mask=col_offsets < num_cols)
```

**几个细节**：

1. **`other=float("-inf")`**：padding 位填 $-\infty$，`exp(-inf)=0`，自动不参与求和。
2. **减 max** 是数值稳定性标准技巧，防止 `exp` 上溢。
3. **`row_stride`**：为了处理非连续张量（stride 不等于列数）。
4. **`BLOCK_SIZE` 必须是 2 的幂**（Triton 要求）。

### 8.3 实测对比（dim=16384）

| 实现 | 耗时 |
|---|---|
| `manual_softmax` | ~3.7 ms |
| `compiled_softmax`（`torch.compile`） | ~1.3 ms |
| `pytorch_softmax` | ~1.5 ms |
| `triton_softmax`（我们的） | ~1.9 ms |

**重要观察**：`torch.compile` 甚至打败了 `torch.nn.functional.softmax`——因为它知道具体形状，能做更 aggressive 的优化。

---

## 9. 扩展：Tiled Matmul 的思想

对应脚本函数：`triton_matmul_main()`（代码没实际 run，但讲了思想）。

### 9.1 问题

$$
C_{ij} = \sum_{k} A_{ik} B_{kj}
$$

朴素实现：$MKN$ 次读，$MN$ 次写。

**观察**：计算 $C_{i,j}$ 和 $C_{i,j+1}$ 都要读 $A$ 的同一行。能不能只读一次？—— 能，用 **shared memory**。

### 9.2 Tiling 策略

小矩阵情形：把整个 A、B 塞进 shared memory，算出 C。开销降到 $MK + KN$ 读 + $MN$ 写。

大矩阵情形：

1. 把 A、B、C 分成小块（tile）。
2. 每个 block 负责 C 的一个 tile。
3. 按 k 维度循环：加载 A 的一个 tile + B 的一个 tile 到 shared memory，做一次 mini-matmul，累加到 C tile。
4. 最后写回 C tile。

### 9.3 利用 L2 Cache：Block 遍历顺序

计算 3×3 的 C 有两种遍历顺序：

- **行主序**（row-major）：访问 9 + 81 = 90 个 block
- **分组遍历**（grouped）：访问 27 + 27 = 54 个 block

**差别纯粹是遍历顺序**——数据局部性不同，L2 命中率差异巨大。Triton 教程里有详细图解。

### 9.4 为什么还是要自己写 matmul？

唯一理由：**和其他算子融合**，比如 `gelu(A @ B)`、`softmax(Q @ K.T)`（就是 flash attention 的起点）。

---

## 10. 总结

### 10.1 本讲的五种写法

| 方式 | 代码在哪 | 抽象层级 | 何时用 |
|---|---|---|---|
| 朴素 PyTorch | Python | 最高 | 原型、可读 |
| 官方 fused (F.gelu) | 已有 C++/CUDA | 高 | 能用就用 |
| `torch.compile` | 自动生 Triton | 中高 | 默认首选加速 |
| Triton | Python + `@triton.jit` | 中 | 需要算法级定制 |
| CUDA | `.cu` 文件 | 低 | 极少需要 |

### 10.2 三种算子复杂度

- **Element-wise**（GeLU）：CUDA/Triton 都容易写
- **Row-wise reduction**（softmax）：Triton 仍然很舒服，一行一 block
- **复杂聚合**（matmul）：tiling + shared memory，要动脑

### 10.3 核心原则

> **Key principle**: organize computation to minimize reads/writes.
>
> **Key ideas**: kernel fusion（仓库/工厂比喻）+ tiling（shared memory）.

### 10.4 实操工作流

1. 写朴素实现跑通。
2. `benchmark()` 测端到端时间。
3. `profile()` 找瓶颈 kernel。
4. 试 `torch.compile`——通常够用。
5. 还不够快 → profiler 定位后写 Triton。
6. Triton 都不行 → 算法级重写（Flash Attention 那种层次）。

### 10.5 心态

- 永远先 benchmark 再优化，别猜。
- 自动编译器只会越来越好，**手写 CUDA 的 ROI 在下降**。
- CPU 和 GPU 是**异步的两台机器**，这决定了 `cuda.synchronize()`、`print(loss)` 陷阱、以及为什么 Python 还能打满 GPU。

---

## 附：延伸阅读

- Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [CUDA MODE](https://www.youtube.com/@CUDAMODE) 系列讲座（Lec 1/2/3/4/8）
- HetSys Course（Lec 1–5）：GPU 软硬件层级
- [Triton 官方 Tutorial](https://triton-lang.org/main/getting-started/tutorials/)
- [PTX ISA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [GPU Puzzles](https://github.com/srush/gpu-puzzles)（强烈推荐用于巩固 Triton 直觉）
- [Triton Paper (MAPL 2019)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

