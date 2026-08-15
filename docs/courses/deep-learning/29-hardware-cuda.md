---
course: deep-learning
edition: industrial
title: AI 训练硬件、推理硬件与 CUDA 生态
description: 从 SIMT/Tensor Core、算力/带宽/容量、HBM/互联进入训练与 prefill/decode 瓶颈，并建立 CUDA runtime、库、通信与 profiler 地图。
outline: [2, 3]
---

<DlLessonHero number="29" module="GPU 与 AI Infra" title="AI 训练硬件、推理硬件与 CUDA 生态" question="峰值 FLOPS、HBM 带宽、显存容量、互联和软件栈怎样共同决定训练吞吐、TTFT 与 decode token/s？" duration="建议 360–470 分钟" :outcomes="['区分 CPU、GPU、TPU/NPU 的执行与存储取舍', '用算术强度和 Roofline 判断计算/带宽瓶颈', '为训练、prefill、decode 分别建立状态与数据流账本', '解释 SIMT、Tensor Core、HBM、PCIe/NVLink 与集合通信接口', '绘制 CUDA driver/runtime/compiler、cuBLAS/cuDNN/NCCL/Nsight 的职责边界']" />

硬件章节不做型号排行榜。我们研究一个 workload 为何快或慢：它执行多少算术、搬多少字节、需多少常驻状态、跨设备交换什么，以及软件能否把硬件利用起来。峰值数字只有在 dtype、稀疏条件和利用率同时满足时才有意义。

## 贯穿案例：7B 模型训练与在线推理共用集群

训练侧需要 BF16 吞吐、优化器/激活容量和高速 all-reduce；在线侧同时有长 prompt prefill 与逐 token decode，受 KV Cache、权重带宽、动态 batch 和尾延迟影响。我们用同一硬件账本解释为什么“训练快”不等于“单请求 decode 快”。

<DlQuiz kind="先判断" question="某 GPU BF16 峰值是另一张卡的 2 倍，端到端 LLM 一定快 2 倍吗？" :options="['一定', '不一定，还受带宽、容量、互联、形状、kernel 和利用率影响', '只要显存相同就一定', '峰值与 dtype 无关']" :answer="1" explanation="峰值只是 Roofline 上界的一部分。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>为 GEMM、LayerNorm、embedding lookup、all-reduce、decode attention 分别猜测计算/带宽/通信瓶颈，再用字节和 FLOPs 证明。</div>

## 1. CPU、GPU、TPU/NPU：控制流与数据流

CPU 有强单线程、复杂控制与大缓存，适合调度、数据处理和串行逻辑；GPU 用大量线程和高带宽隐藏延迟，适合规则张量并行；TPU/NPU 常以 systolic/tensor 阵列、片上 SRAM 和编译图换高能效。实际系统通常异构：CPU 发起、GPU/加速器计算、NIC 交换、存储供数。

SIMT 将线程按 warp/wavefront 执行；分支分歧使同一组线程串行走不同路径。Tensor Core/MMA 对 tile 做矩阵乘累加，要求适合的 dtype、对齐和布局。稀疏峰值只有在规定的结构稀疏和 kernel 支持下成立。

<DlQuiz kind="SIMT" question="同一 warp 内线程走不同 if 分支，典型后果是什么？" :options="['分支路径可能被掩码串行执行，利用率下降', '自动增加 HBM', 'Tensor Core 变成 CPU', '无任何影响']" :answer="0" explanation="SIMT 对规则控制流最友好。" />

## 2. Roofline：算术强度连接算法与硬件

算术强度 $I=\mathrm{FLOPs}/\mathrm{bytes}$；可达性能受 $\min(P_{peak},B_{mem}I)$ 约束。低强度算子即使 FLOPs 少也可能被 HBM 限制，fusion 通过减少中间写回提高强度；GEMM 通过 tile 重用进入计算区。

<HardwareRooflineLab />

```python
def roofline(flops, moved_bytes, peak, bandwidth):
    intensity = flops/moved_bytes
    return min(peak, bandwidth*intensity), intensity

perf, ai = roofline(2e12, 200e9, 1e15, 3e12)
assert ai == 10 and perf == 3e13  # 带宽上界低于峰值
```

<DlQuiz kind="Roofline" question="算子融合为何常能加速 elementwise/normalization 链？" :options="['减少中间张量 HBM 读写，提高有效算术强度', '增加 Python 行数', '降低时钟', '删除正确性检查']" :answer="0" explanation="收益常来自 memory traffic 而非少量 FLOPs。" />

## 3. 存储层级：register、shared/SRAM、cache、HBM 与主存

越靠近计算单元容量越小、带宽越高。register spill 会落到 local/global memory；shared memory 允许 block 内显式重用，但有容量和 bank conflict；L2 跨 SM 缓存；HBM 容纳权重/状态/KV；PCIe/CXL/主存 offload 容量大但延迟和带宽差。优化不是“把所有东西放 shared”，而是匹配生命周期与重用。

## 4. 训练、Prefill 与 Decode 的不同账本

<TrainingInferenceMemoryLab />

训练常驻权重、梯度、master/optimizer state，并保存/重算激活；prefill 对长 prompt 做大矩阵并生成 KV，较计算密集；decode 每步矩阵较瘦、要读权重与历史 KV，常受内存带宽和调度限制。批量可提高算术强度，但增加排队与尾延迟。

```python
def adam_mixed_bytes(params):
    # FP16 weight+grad，FP32 master+m+v 的常用粗估
    return params*(2+2+4+4+4)

assert adam_mixed_bytes(7e9)/2**30 > 100
```

<DlQuiz kind="阶段" question="LLM decode 为什么比 prefill 更容易受内存带宽限制？" :options="['每步 token 的矩阵更瘦，权重/KV 读取相对算术更多', 'decode 不读权重', 'prefill 没有计算', '只有 CPU 才 decode']" :answer="0" explanation="batch 和 speculative decoding 可改善利用率，但改变延迟。" />

## 5. 互联：PCIe、NVLink/NVSwitch 与网络

单机 GPU 通过 PCIe 或高带宽 GPU 互联，跨机经 NIC/InfiniBand/RoCE。带宽、延迟、拓扑和并发集合共同决定 all-reduce/all-to-all。模型能放下不代表通信可扩展；MoE all-to-all、TP all-reduce 与 DP 梯度同步有不同消息形状。

```python
def transfer_seconds(gib, bandwidth_gib_s, latency_us=10):
    return gib/bandwidth_gib_s + latency_us*1e-6

assert transfer_seconds(8, 400) < transfer_seconds(8, 32)
```

<DlQuiz kind="互联" question="同节点 NVLink 很快，是否证明跨节点 TP 也同样快？" :options="['证明', '不证明，跨节点网络带宽/延迟和拓扑不同', 'TP 不通信', '只与显存有关']" :answer="1" explanation="并行映射应尊重物理拓扑。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>画 8 GPU 节点：哪些通信留在 NVLink 域，哪些跨 NIC？TP、PP、DP 各优先怎样摆放？</div>

## 6. CUDA 生态地图

CUDA driver 管设备与上下文，runtime 提供更高层启动/内存/stream/event，NVCC/LLVM/PTX/SASS 构成编译路径；cuBLAS/cuBLASLt 提供 GEMM，cuDNN 提供深度学习 primitives，NCCL 提供 GPU 集合通信，CUDA Graph 减少重复 launch 开销，Nsight Systems/Compute 分析时间线与 kernel。框架还叠加 allocator、autograd、compiler 和第三方 kernel。

Stream 异步意味着 CPU 计时若不 event/synchronize 会虚假偏快；kernel error 也可能延迟暴露。正确 benchmark 要 warm-up、同步、重复、分位数、形状/dtype/版本/功耗，并先与库实现做数值对照。

<DlQuiz kind="CUDA 计时" question="只用 Python `time.time()` 包住异步 CUDA kernel launch 最可能测到什么？" :options="['主要是发射时间而非完成时间', '精确 GPU 墙钟', 'HBM 容量', '编译器正确性']" :answer="0" explanation="需 CUDA event 或同步边界。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>为一个 GEMM benchmark 写清 warm-up、event、同步、100 次分位数、容差、shape、dtype、TF32 和功耗状态。</div>

## 7. 推荐论文阅读

<DlPaperCard title="Roofline: An Insightful Visual Performance Model for Multicore Architectures" meta="Williams, Waterman & Patterson · 2009" priority="必读 · 性能上界" question="算术强度、峰值计算与带宽怎样形成可达性能屋顶？" reading="精读 ridge point 和优化方向；用本章五类算子重画 roofline。" href="https://people.eecs.berkeley.edu/~kubitron/courses/cs252-S10/handouts/papers/RooflineVyNo.pdf" />

<DlPaperCard title="In-Datacenter Performance Analysis of a Tensor Processing Unit" meta="Jouppi et al. · 2017" priority="选读 · 专用加速器" question="systolic matrix unit、片上存储和部署 workload 如何影响性能/功耗结论？" reading="读架构、roofline 与 CPU/GPU 对照；核对生产推理负载。" href="https://arxiv.org/pdf/1704.04760" />

<DlPaperCard title="CUDA C++ Programming Guide" meta="NVIDIA · 持续更新" priority="工具书 · 官方生态" question="thread hierarchy、memory、stream、graph 与 compute capability 的正式语义是什么？" reading="按当前安装版本查 Programming Model、Memory Hierarchy、Asynchronous Execution；结论绑定版本。" href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/" />

## 8. 练习与门禁

算五类 AI；复算训练状态；区分 prefill/decode；画互联拓扑；用 event 基准；检查 dtype/稀疏峰值条件；提交 Nsight 时间线和瓶颈证据。

## 本章小结

- 真实性能由算力、带宽、容量、互联和软件利用率共同决定。
- 训练、prefill、decode 的常驻状态和瓶颈不同，不能用单一峰值排名。
- CUDA 是从执行模型到库、通信和 profiler 的生态；可靠基准必须同步、对照和绑定环境。

<DlLessonProgress lesson-id="29" next-href="/courses/deep-learning/30-cuda-triton-tilelang" next-label="下一章：AI 算子与三种实现层级" />
