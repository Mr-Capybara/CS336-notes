# CS336 Lecture 8: Parallelism (Code Walkthrough) — 详细笔记

> 本讲定位：**把 Lecture 7 的并行理论全部"落到 PyTorch 代码上"**。上一讲讲的是"为什么要这样切、怎么切最省通信"；这一讲是"用最朴素的方式，在 MLP 上写出来，让你看清每一行 `all_reduce` / `all_gather` / `send` 在干什么"。
>
> 全篇与 [lecture_08.py](/Users/zehuayin/CS336/CS336-notes/8#Parallelism2/lecture_08.py) 严格对齐。函数调用顺序 = 讲课顺序 = 本笔记顺序：
>
> 1. **回顾与主题**：单 GPU 内部 vs 跨 GPU，统一主题是"靠近数据编排计算"
> 2. **Part 1 通信原语**
>    - `collective_operations()`：概念层（broadcast / scatter / gather / reduce / all-gather / reduce-scatter / all-reduce）
>    - `torch_distributed()`：硬件 (NVLink/NVSwitch) → NCCL → `torch.distributed` 三层抽象，跑一个 4-rank 的 all-reduce / reduce-scatter / all-gather 验证
>    - `benchmarking()`：实测单节点内的 all-reduce、reduce-scatter 带宽
> 3. **Part 2 分布式训练**（都在深层 MLP 上演示，因为 Transformer 的算力瓶颈就是 MLP）
>    - `data_parallelism()`：沿 batch 维切
>    - `tensor_parallelism()`：沿 width 维切
>    - `pipeline_parallelism()`：沿 depth 维切
> 4. **总结**：三种切法是同一套"重算 / 存本地 / 存异地靠通信"三选一 trade-off 在不同维度上的体现

---

## 0. 起点：从"单 GPU 内"到"多 GPU 间"

对应 `main()` 开头。

### 0.1 统一主题

无论在哪一层，**算力单元 (ALU) 总是离数据很远**，优化目标永远是：

> 组织计算，避开数据传输瓶颈，让算术强度 (arithmetic intensity) 足够高以打满 ALU。

- **Lecture 6**：单 GPU 内，用 **fusion / tiling** 减少 HBM ↔ SRAM 往返。
- **Lecture 7**：多 GPU 间，用 **replication / sharding** 减少跨 GPU/节点的通信量。

### 0.2 统一存储层级（从快到慢）

| 层级 | 位置 | 典型带宽（H100） |
|---|---|---|
| L1 / shared memory | 单 SM 内 | 极快，容量 KB 级 |
| HBM | 单 GPU 内 | ~3.9 TB/s |
| NVLink | 同节点跨 GPU | 每卡 900 GB/s（18 × NVLink 4.0） |
| NVSwitch / IB | 跨节点 | 更慢一档 |

**对比直觉**：NVLink 比 HBM 仅慢 ~4×，所以节点内多卡并行 "不算太亏"；跨节点带宽再降一档，会强烈影响并行策略的选择。

---

## 1. 集合通信原语（Collective Operations）

对应 `collective_operations()`。

### 1.1 术语

- **world size**：进程/设备总数（例 4）
- **rank**：某个设备编号（0, 1, 2, 3）——注意这里的 rank 是"设备 id"，和线性代数里的秩无关

### 1.2 七个原语

假设 world_size = 4，每个 rank 持有一个张量 $T_i$。

| 原语 | 输入分布 | 输出分布 | 直觉 |
|---|---|---|---|
| **Broadcast** | $T$ 只在 rank 0 | 所有 rank 都有 $T$ | 单点 → 所有点 |
| **Scatter** | $[T_0, T_1, T_2, T_3]$ 在 rank 0 | rank $i$ 拿到 $T_i$ | 切片分发 |
| **Gather** | rank $i$ 有 $T_i$ | rank 0 收集成 $[T_0, \dots, T_3]$ | Scatter 的逆 |
| **Reduce** | rank $i$ 有 $T_i$ | rank 0 得到 $\sum_i T_i$ | Gather + sum |
| **All-gather** | rank $i$ 有 $T_i$ | 所有 rank 都有 $[T_0, \dots, T_3]$ | Gather 的 "All" 版 |
| **Reduce-scatter** | rank $i$ 有 $[T_{i,0}, T_{i,1}, T_{i,2}, T_{i,3}]$ | rank $j$ 得到 $\sum_i T_{i,j}$ | 逐块 reduce，结果分散 |
| **All-reduce** | rank $i$ 有 $T_i$ | 所有 rank 都有 $\sum_i T_i$ | Reduce + Broadcast |

### 1.3 命名记忆法

- `reduce` = 做某种**结合+交换**的聚合（sum / min / max / avg）
- `broadcast` / `scatter` 是 `gather` 的**逆向**
- `all-` 前缀 = 目标是**所有** rank

### 1.4 关键恒等式

$$
\text{All-reduce} \;=\; \text{Reduce-scatter} \;+\; \text{All-gather}
$$

这个分解是**后面所有带宽分析的基础**：因为 all-reduce 必须送出再收回，通信量是单向原语的 2×。

---

## 2. 硬件与 PyTorch 分布式栈

对应 `torch_distributed()`。

### 2.1 两代硬件拓扑

**经典（家用/传统机房）**：

```
CPU ── PCIe ── GPU
 │              │
 ├── Ethernet ──┤  (跨节点)
```

- 同节点 GPU：**PCIe**（PCIe 7.0 × 16 lanes ≈ 242 GB/s），数据还要经 CPU kernel buffer 转运，开销大。
- 跨节点：**Ethernet**（~200 MB/s），慢且为通用协议设计。

**现代（数据中心）**：

```
GPU ── NVLink ── GPU    (同节点, 绕开 CPU)
     NVSwitch            (跨节点, 绕开 Ethernet)
```

- H100：18 条 NVLink 4.0 → 每卡 **900 GB/s** 聚合带宽。
- 作为对照：HBM 带宽 3.9 TB/s，仅快 ~4×。
- 可以用 `nvidia-smi topo -m` 查看本机的 GPU 互联拓扑（会看到类似 `NV18` 的标记，表示两卡之间有 18 条 NVLink）。

### 2.2 三层软件栈

```
用户代码 (Python / PyTorch)
   │
torch.distributed         ← 干净的 Python API, 提供 all_gather_into_tensor 等
   │
NCCL (NVIDIA Collective Communication Library)
   │   - 探测硬件拓扑
   │   - 选择最优路径
   │   - 把 collective op 翻译成 CUDA kernel (send/recv 包)
   │
硬件 (NVLink / NVSwitch / IB)
```

- **NCCL** 是 GPU 后端；
- **Gloo** 是 CPU 后端（`backend="gloo"`），方便在笔记本上不带 GPU 调试。
- PyTorch 还提供更高层的 `FullyShardedDataParallel` 等，但本课**手写从底层原语起**，不直接用。

### 2.3 多进程启动模型

`spawn(fn, world_size=4)` 是一个薄封装，基于 `torch.multiprocessing`，**启动 `world_size` 个进程同时执行 `fn`**，每个进程传入自己的 `rank`。

> 关键心智模型：你看到的是**一份代码**，但它**同时在 `world_size` 个进程里运行**。代码里的 `if rank == 0: ...` 就是让不同进程走不同分支的唯一方式。

### 2.4 初始化 / 清理模板

```python
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()
```

- `MASTER_ADDR/PORT` 只用于进程**互相发现 + 协调**，实际数据走 NCCL。

### 2.5 三个原语的代码示范

对应 `collective_operations_main()`。每次 op 前加 `dist.barrier()`——**不是逻辑上需要同步，而是为了让前面的 `print` 刷完再打下一批**。

#### (1) All-reduce（原地操作）

```python
tensor = torch.tensor([0.,1,2,3], device=get_device(rank)) + rank  # 每个 rank 不同
dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)      # 就地改写
```

- before：rank $r$ 持有 `[r, r+1, r+2, r+3]`
- after：所有 rank 都是 `[0+1+2+3, 1+2+3+4, ...] = [6, 10, 14, 18]`
- `async_op=True` 返回 handle，用于通信/计算重叠。

#### (2) Reduce-scatter（分开 in/out）

```python
input  = torch.arange(world_size, dtype=torch.float32, device=...) + rank  # shape [W]
output = torch.empty(1, device=...)                                        # shape [1]
dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM)
```

- 输入第 0 维必须是 `world_size`。
- rank $j$ 只收 $\sum_i \texttt{input}[i][j]$（这里 input 是 1D，所以就是第 $j$ 个标量）。

#### (3) All-gather

```python
input  = ...                         # shape [1]
output = torch.empty(world_size, device=...)   # shape [W]
dist.all_gather_into_tensor(output, input)
```

把 reduce-scatter 的 output 喂进去，all-gather 后得到和 all-reduce 一模一样的结果，**实证了 all-reduce = reduce-scatter + all-gather**。

---

## 3. 通信带宽实测 (Benchmarking)

对应 `benchmarking()`、`all_reduce()`、`reduce_scatter()`。

### 3.1 Benchmark 模板（务必记住）

```python
# 1. Warmup: 跑一次让 kernel / 连接就绪
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()   # 等 CUDA kernel 做完
dist.barrier()             # 等所有 rank 到齐

# 2. 计时
start = time.time()
dist.all_reduce(tensor, ...)
torch.cuda.synchronize()
dist.barrier()
end = time.time()
```

- `torch.cuda.synchronize()`：等**本 rank** 的 GPU kernel 跑完（否则 `time.time()` 只量了 kernel launch）。
- `dist.barrier()`：等**所有 rank** 到这一行（否则不同 rank 的计时基准不齐）。

### 3.2 All-reduce 带宽公式

单张量 `size_bytes = numel * element_size`（如 100M × fp32 = 400 MB）。

$$
\text{sent\_bytes} \;=\; \text{size\_bytes} \;\times\; 2 \;\times\; (\text{world\_size}-1)
$$

- **×2** 因为要先把 input 送出去、再把 output 拿回来（等价于 reduce-scatter + all-gather 两步）。
- **×(world_size−1)** 因为一个 rank 要和其余所有 rank 通信。

$$
\text{bandwidth} \;=\; \frac{\text{sent\_bytes}}{\text{world\_size} \times \text{duration}}
$$

这里 `world_size × duration` 当作"总墙钟时间"——把 world_size 个 rank 并行消耗的时间折算成串行总量。

实测：4 × H100，100M fp32 → **约 277 GB/s**（远低于理论峰值 900 GB/s，因实际张量大小、拓扑、NCCL 算法选择都会影响）。

### 3.3 Reduce-scatter 带宽公式

```python
input  = torch.randn(world_size, num_elements, device=...)  # 每 rank 一整块
output = torch.empty(num_elements, device=...)
```

$$
\text{sent\_bytes} \;=\; \text{data\_bytes} \;\times\; (\text{world\_size}-1)
$$

**没有 ×2**——reduce-scatter 只需单向送数据聚合即可。

实测 ≈ **70 GB/s**。比 all-reduce 低的原因不完全明确，和 NCCL 对 all-reduce 的深度优化（如 SHARP in-network reduction）、算法选择等有关。**结论：benchmark 你自己的硬件是必要的。**

### 3.4 小结

- All-reduce 的通信量是 reduce-scatter（或 all-gather）的 **2×**，这与恒等式 `all-reduce = reduce-scatter + all-gather` 完全一致。
- 写 benchmark 一定要 warmup + `cuda.synchronize()` + `barrier()`，缺一不可。

---

## 4. Data Parallelism (DP / DDP)

对应 `data_parallelism()`、`data_parallelism_main()`。

![切法示意](images/data-parallelism.png)

### 4.1 切法

- 沿 **batch 维** 把数据切成 `world_size` 份。
- 模型参数 / 优化器状态在**每个 rank 完整复制一份**。

### 4.2 训练循环（关键只多一行）

```python
# 切 batch
local_bsz = batch_size // world_size
data = data[rank*local_bsz : (rank+1)*local_bsz].to(device)

# 每个 rank 自己一份完整参数 + 自己一份 AdamW state
params = [get_init_params(D, D, rank) for _ in range(num_layers)]
optimizer = torch.optim.AdamW(params, lr=1e-3)

for step in range(num_steps):
    # Forward
    x = data
    for p in params:
        x = F.gelu(x @ p)
    loss = x.square().mean()

    # Backward
    loss.backward()

    # ★ 唯一的"分布式"改动：把梯度平均掉
    for p in params:
        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    optimizer.step()
```

### 4.3 原理与不变量

- **loss 在不同 rank 上不同**（算在各自 batch 切片上）。
- **grad 被 all-reduce 取均值 → 各 rank 梯度相同**。
- 只要参数初始值（相同 seed）+ 优化器状态一致 + 每步梯度一致，**参数在所有 rank 上自始至终保持相同**。
- 可以把 DDP 理解成"每张卡独立跑一次 SGD，但梯度被偷偷换成了平均值"。

### 4.4 为什么不切优化器状态而是全复制？

这里等价于用**重算/复制** trade 掉**通信**：让每张卡自己跑一遍 optimizer step，比把 optimizer state 搬来搬去便宜得多。要进一步省显存就得上 ZeRO / FSDP（本课不展开实现）。

### 4.5 一致性是怎么保证的？

所有 rank 都会在 `all_reduce` 这一行**隐式同步**——该函数本身是同步屏障。

> ⚠️ 由此也有一个坑：如果某个 rank 的控制流漏掉了一次 `all_reduce`，整个集群就会**hang 住**。

---

## 5. Tensor Parallelism (TP)

对应 `tensor_parallelism()`、`tensor_parallelism_main()`。

![切法示意](images/tensor-parallelism.png)

### 5.1 切法

- 沿**宽度维 (hidden dim)** 把每一层的参数切成 `world_size` 片。
- 数据和激活需要**在每层之间通信**（因为算完的激活只有 local 一部分列）。

### 5.2 前向代码

```python
local_D = D // world_size          # 每 rank 负责 D/W 列
params = [get_init_params(D, local_D, rank) for _ in range(num_layers)]

x = data                            # shape [B, D], 每 rank 都有完整 x
for i in range(num_layers):
    # 本 rank 只算自己那几列: [B,D] @ [D,local_D] -> [B, local_D]
    x = F.gelu(x @ params[i])

    # 汇总所有 rank 的列，得到完整激活 [B, D]
    activations = [torch.empty(B, local_D, device=...) for _ in range(world_size)]
    dist.all_gather(tensor_list=activations, tensor=x)
    x = torch.cat(activations, dim=1)   # [B, D]
```

### 5.3 理解要点

- **参数切了**：每 rank 只保存 `D × D/W` 的子矩阵 → 参数显存 / 参数相关的 compute 都降为 1/W。
- **激活全聚**：每层后要 `all_gather`，带宽压力重 → TP **要求高带宽互联**，所以 TP 通常**只放节点内（NVLink）**。
- **反向**：课上没写（作业题）。直觉上反向需要的是 reduce（因为梯度要累加），与前向的 all-gather 对称。

### 5.4 与 DP 的对比

| 维度 | DP | TP |
|---|---|---|
| 切什么 | batch | hidden dim |
| 参数复制？ | 是 | 否（切掉） |
| 激活通信 | 无（只在反向 all-reduce 梯度） | 每层都要 |
| 对带宽要求 | 中 | 高 |

---

## 6. Pipeline Parallelism (PP)

对应 `pipeline_parallelism()`、`pipeline_parallelism_main()`。

![切法示意](images/pipeline-parallelism.png)

### 6.1 切法

- 沿**深度维 (layer)** 切：rank 0 拿前几层，rank 1 拿后几层……
- 数据是完整 batch，按 **micro-batch** 流水。

### 6.2 前向代码（world_size=2, num_layers=4, num_micro_batches=4）

```python
local_L = num_layers // world_size           # 每 rank 2 层
local_params = [get_init_params(D, D, rank) for _ in range(local_L)]
micro_bsz = batch_size // num_micro_batches  # 32

if rank == 0:
    micro_batches = data.chunk(num_micro_batches, dim=0)   # 真实数据
else:
    # 下游 rank 先分配好接收 buffer
    micro_batches = [torch.empty(micro_bsz, D, device=...)
                     for _ in range(num_micro_batches)]

for x in micro_batches:
    # 1) 从上游收
    if rank - 1 >= 0:
        dist.recv(tensor=x, src=rank-1)

    # 2) 本 rank 的层串行算
    for p in local_params:
        x = F.gelu(x @ p)

    # 3) 发给下游
    if rank + 1 < world_size:
        dist.send(tensor=x, dst=rank+1)
```

### 6.3 为什么要切 micro-batch？

不切的话，rank 1 必须等 rank 0 算完**整个** batch 才能开始，形成巨大的 **pipeline bubble**。切成 $M$ 个 micro-batch 后，rank 0 做完第 1 个 micro-batch 就交给 rank 1，自己立刻开始第 2 个，气泡占比下降到 $\mathcal{O}(\frac{W-1}{M+W-1})$ 量级。

### 6.4 本实现里缺失的东西

1. **反向**：写起来繁琐，交作业。一旦加上反向，就要决定 forward / backward 的交错策略（1F1B、interleaved 1F1B、zero-bubble 等，见 Lecture 7）。
2. **通信/计算重叠**：这里 `send` / `recv` 是同步的 → `x = x @ p` 没跑完 `send` 不会发，`recv` 没到下一轮不会算。实际上应改成 `isend` / `irecv`（异步），返回 handle，最后统一 `wait()`。
3. **点对点原语的语义细节**：
   - `send/recv` 的匹配靠 **(src, dst) 对**，不看 tensor 变量名。
   - 同 (src, dst) 之间多次 send 的**顺序被保留**（像一条 FIFO 流）。
   - 收不到 / 没人收 → 进程直接 **hang**（这不是错误，是等）。

### 6.5 与 DP、TP 的对比

| 维度 | DP | TP | PP |
|---|---|---|---|
| 切什么 | batch | width (hidden) | depth (layer) |
| 通信时机 | 反向结束 all-reduce grads | 每层 all-gather 激活 | 逐层 send/recv 激活 |
| 通信量/步 | 与参数规模成正比 | 与激活规模成正比，次数多 | 与激活规模成正比，次数 = W |
| 适合层级 | 节点间 | 节点内（NVLink） | 节点间（通信量低） |
| 主要开销 | 参数全复制 → 显存 | 每层通信 → 带宽 | pipeline bubble → 时间 |

---

## 7. Jax / TPU 的"声明式"风格（对照）

一句话：**Jax + Levanter 把 sharding 变成"标注"，Jax 编译器自动搞定通信**。

```python
# 伪代码：声明"这个参数沿 embed 维切到 mp 轴上"
sharded_params = shard(params, PartitionSpec('mp', None))
```

- 你只需要说：**模型的哪些维度要切** + **硬件上有哪些轴（data/model/pipeline）**。
- 编译器自动：生成 all-gather / reduce-scatter / send-recv 的具体 kernel。
- 对比 PyTorch：你得**亲手**写每一次 collective。

**课程选择 PyTorch 的理由**：能看清"引擎盖下"到底在干什么。真实生产里反而推荐用 FSDP / Megatron-LM / Jax 这些现成轮子。

---

## 8. 总结

### 8.1 四种切法

| 切哪个维度 | 名字 |
|---|---|
| batch (N) | Data parallel |
| width (hidden dim) / experts | Tensor / Expert parallel |
| depth (layers) | Pipeline parallel |
| sequence (context length) | Sequence / Context parallel |

### 8.2 贯穿始终的三选一

对于"我需要这个张量，但它不在本地"这个问题，永远是三选一：

1. **重算 (recompute)**：放弃存它，要用时重新算一遍。
2. **存本地 (store in memory)**：用显存换时间。
3. **存异地 + 通信 (store remotely + communicate)**：用带宽换显存。

所有并行策略、激活检查点都是这三件事在不同粒度上的组合。

### 8.3 为什么这套层级永远都在？

- 硬件确实在进步，H100 之后是 Blackwell，带宽、显存都会翻倍。
- **但**模型永远会**增长到打满当前硬件**——人类对更强模型的胃口没有上限。
- 所以 "L1 / HBM / NVLink / NVSwitch / IB" 这种**层级式 memory hierarchy** 永远存在，**避免数据传输成为瓶颈**永远是优化的核心。

---

## 附录 A：关键 API 速查

| API | 作用 | 典型用途 |
|---|---|---|
| `dist.init_process_group(backend, rank, world_size)` | 初始化进程组 | setup |
| `dist.barrier()` | 全 rank 同步点 | 计时 / 打印对齐 |
| `dist.all_reduce(t, op)` | 原地全规约 | DDP 梯度平均 |
| `dist.reduce_scatter_tensor(out, in, op)` | 规约后分片 | ZeRO, FSDP |
| `dist.all_gather_into_tensor(out, in)` | 分片后聚合 | TP 激活聚合、FSDP 参数聚合 |
| `dist.all_gather(list, t)` | 同上，list 版 | 本课 TP 实现 |
| `dist.send(t, dst)` / `dist.recv(t, src)` | 点对点同步 | PP 激活传递 |
| `dist.isend` / `dist.irecv` | 点对点异步 | 计算/通信 overlap |
| `torch.cuda.synchronize()` | 等本卡 kernel 做完 | benchmark |

## 附录 B：两个必记公式

**All-reduce 通信量**：

$$
\text{bytes sent per rank} \;=\; 2(W-1)\cdot \text{tensor\_bytes}
$$

**Reduce-scatter / All-gather 通信量**：

$$
\text{bytes sent per rank} \;=\; (W-1)\cdot \text{tensor\_bytes}
$$

其中 $W$ = world size。两者的 2× 关系直接来自 `all-reduce = reduce-scatter + all-gather`。
