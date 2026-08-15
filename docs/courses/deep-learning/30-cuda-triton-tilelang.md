---
course: deep-learning
edition: industrial
title: AI 算子与 CUDA、Triton、TileLang
description: 从 kernel、fusion、coalescing、tiling、online reduction 与 profiling 比较 CUDA、Triton、TileLang，并用四类算子说明选择边界。
outline: [2, 3]
---

<DlLessonHero number="30" module="GPU 与 AI Infra" title="AI 算子与 CUDA、Triton、TileLang" question="同一个数学算子如何映射为线程、program/tile 与 dataflow pipeline；三种工具何时值得用、如何证明正确且更快？" duration="建议 390–520 分钟" :outcomes="['区分算子语义、kernel 实现、fusion 与调度', '解释 coalescing、tiling、shared/register 重用和 occupancy', '比较 CUDA、Triton、TileLang 的抽象层和生态边界', '分析 Vector Add、Fused Softmax、GEMM/Dequant 与 FlashAttention', '建立 correctness-first 的 GPU benchmark 与 profiler 工作流']" />

“用 Triton 重写”不是性能结论，“CUDA 最快”也不是普遍定律。工具选择必须绑定算子、shape、dtype、目标硬件、开发/维护成本和编译器版本；任何自定义 kernel 先通过库/PyTorch 正确性基线。

## 贯穿案例：为推理服务优化四个热点

Profiler 显示 Vector Add launch-bound、Softmax 产生大中间张量、4-bit Linear 反量化后再 GEMM、Attention score 占 HBM。我们分别判断 fusion/tiling/dataflow 机会，而不是用同一模板重写所有算子。

<DlQuiz kind="先判断" question="自定义 kernel 比 PyTorch 单个 op 快，是否证明端到端服务更快？" :options="['证明', '不证明，还需看图融合、编译、调度、形状覆盖和端到端占比', '只要 kernel 名短就证明', '微基准无意义']" :answer="1" explanation="Amdahl 定律和上层系统会限制收益。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>在写 kernel 前记录基线调用图、shape 分布、dtype、设备、端到端占比和数值容差；若热点只占 5%，理论最大加速多少？</div>

## 1. Kernel、launch 与融合

算子是数学/框架语义，kernel 是一次或多次设备程序。小 elementwise 链常受 launch 和 HBM 中间读写限制，fusion 把 producer/consumer 保存在 register/SRAM；但过度融合会增加 register、降低 occupancy、放大编译和形状特化成本。

```python
def vector_add_intensity(elements, bytes_per=2):
    flops = elements
    moved = elements*bytes_per*3  # 读 x/y，写 z
    return flops/moved

assert abs(vector_add_intensity(1_000_000)-1/6) < 1e-12
```

<DlQuiz kind="Vector Add" question="Vector Add 通常更接近哪类瓶颈？" :options="['内存带宽/launch', '矩阵计算峰值', 'NCCL all-to-all', 'tokenizer']" :answer="0" explanation="每搬多个字节只做一次加法，算术强度低。" />

## 2. Coalescing、Tiling、Shared 与 Register

相邻线程访问连续地址可合并 transaction；不规则 stride 会浪费带宽。tile 将工作集装入 shared/SRAM，多个 FMA 重用；register fragment 再靠近 ALU/Tensor Core。tile 太大消耗 shared/register，减少可驻留 blocks；边界 mask 和非整除 shape 也有成本。

<KernelTilingLab />

## 3. 三种实现层级不是互相替代

<KernelLanguageChoiceLab />

CUDA 提供 thread/block、memory、stream、graph、库和最新硬件能力，控制最细、代码/硬件细节多。Triton 用 Python 写 program instance 对 tile 的张量操作，编译器映射线程，适合快速融合和形状特化，但仍需理解 layout、num_warps/stages、occupancy。TileLang 基于 TVM IR，强调 tile/dataflow、shared/fragment、pipeline 与可组合 GEMM primitives，适合表达复杂数据搬运；生态较新，后端与版本需实机验证。

<DlQuiz kind="工具边界" question="使用 Triton 是否意味着无需理解 memory coalescing 和 tile？" :options="['意味着', '不意味着，抽象提高但性能仍由布局、tile、warps 和硬件资源决定', 'Triton 在 CPU 解释执行', '编译器总能全局最优']" :answer="1" explanation="更高抽象不会删除硬件约束。" />

## 4. Fused Softmax 与 online reduction

稳定 softmax 要先求 max，再求 exp-sum；朴素多 kernel 会重复读写。若一行能放入片上资源，可在一个 program/block 做归约；超长行需分块与 online 合并状态 $(m,l)$。

```python
import math

def merge_softmax_state(m1,l1,m2,l2):
    m=max(m1,m2)
    return m, l1*math.exp(m1-m)+l2*math.exp(m2-m)

m,l=merge_softmax_state(2,1.5,4,1.2)
assert m==4 and l>1.2
```

<DlQuiz kind="Softmax" question="online softmax 分块状态为何同时保存 max 与归一化和？" :options="['为跨块稳定重标 exp-sum', '为了增加词表', '只算 argmax', '替代输入']" :answer="0" explanation="全局 max 变化时旧块的和要缩放。" />

## 5. GEMM、Dequant GEMM 与 Tensor Core

GEMM 性能依赖 M/N/K、transpose、dtype、alignment、tile、warps/stages 和 epilogue。通常先用 cuBLASLt；自定义价值来自特殊 shape、融合 bias/activation、量化解包/scale。Dequant 后写 BF16 再 GEMM 会增加 HBM，融合在 tile load/fragment 中可减少流量，但量化格式和 accumulator 精度决定误差。

<DlQuiz kind="Dequant" question="Dequant GEMM 融合的主要机会是什么？" :options="['避免完整反量化权重写回 HBM，在 tile 内解包/缩放后累加', '删除矩阵乘', '把 4-bit 当 FP32 直接加', '不需要 scale']" :answer="0" explanation="融合数据搬运是核心。" />

## 6. FlashAttention/MLA：复杂数据流案例

FlashAttention 在 Q tile 上流式遍历 K/V tile，用 online softmax 更新输出，避免 $T^2$ score 写 HBM。CUDA 可精细控制 MMA/async copy，Triton 适合清晰表达 program tile，TileLang 适合描述 shared/fragment pipeline；哪个更快必须按具体形状/硬件测。MLA decode 还涉及 latent KV、RoPE 分支和小 batch 特殊 shape。

## 7. 正确性与基准协议

对照高精度库结果，测试随机、极值、非整除、stride、空/小维、NaN/Inf（若合同允许）；用绝对/相对容差并检查梯度。基准 warm-up 编译，GPU event，同步，报告中位/P95、吞吐/带宽、shape 分布和端到端收益。Profiler 查 achieved occupancy、dram throughput、Tensor Core、stall 和 launch gap。

```python
import math

times_ms=[.91,.88,.93,.87,1.20,.90,.89]
s=sorted(times_ms)
median=s[len(s)//2]
p95=s[min(len(s)-1,math.ceil(.95*len(s))-1)]
assert median==.90 and p95==1.20
```

<DlQuiz kind="Benchmark" question="为什么首次 kernel 调用通常不纳入稳态延迟？" :options="['可能包含 JIT 编译、module load 与缓存冷启动', '首次一定最快', '没有 kernel', '会改变数学结果']" :answer="0" explanation="应另报冷启动与稳态。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>同一个 kernel 在 128×128、4096×4096、非整除 4100×4090 上性能不同；如何避免只挑有利 shape？</div>

<div class="dl-stop"><strong>停下来推一推 3</strong>列出 Vector Add、Softmax、GEMM/Dequant、FlashAttention 的基线、正确性容差、主瓶颈、候选语言和拒绝自定义的条件。</div>

## 8. 推荐论文阅读

<DlPaperCard title="Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" meta="Tillet, Kung & Cox · 2019" priority="必读 · Program/tile 抽象" question="blocked program、layout 与编译优化怎样替代手写线程映射的一部分？" reading="读语言模型、编译与 GEMM/conv 实验；对照现代 Triton 版本差异。" href="https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf" />

<DlPaperCard title="TileLang: A Composable Tiled Programming Model for AI Systems" meta="TileLang authors · 2025" priority="必读 · Tile/dataflow" question="tile primitives、layout、pipeline 与自动调优怎样表达复杂 AI kernel？" reading="精读编程抽象、GEMM/attention 案例和后端；结论绑定论文版本。" href="https://arxiv.org/pdf/2504.17577" />

<DlPaperCard title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" meta="Dao et al. · 2022" priority="必读 · IO-aware 案例" question="分块与 online softmax 怎样减少 HBM IO 并保持精确 attention？" reading="重读算法与 IO 分析，这次把每步映射到 tile/shared/register。" href="https://arxiv.org/pdf/2205.14135" />

## 9. 练习与门禁

Profile 基线；实现/阅读 Vector Add、Softmax；复算 tile 资源；比较三语言伪代码；对 Dequant fusion 计字节；覆盖 ragged shape；提交冷/热、P50/P95、容差和端到端收益。

## 本章小结

- 算子语义与 kernel 实现分离；fusion、tile 和 dataflow 的价值主要由数据搬运与资源约束决定。
- CUDA、Triton、TileLang 位于不同抽象层，没有脱离算子/硬件/团队的胜负。
- 自定义 kernel 必须先正确、再稳定基准、最后证明端到端收益。

<DlLessonProgress lesson-id="30" next-href="/courses/deep-learning/31-distributed-training" next-label="下一章：分布式训练" />
