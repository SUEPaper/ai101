---
course: deep-learning
edition: industrial
title: 分布式训练、ZeRO、FSDP 与 Megatron
description: 建立 DP/TP/PP/CP/EP 拓扑、集合通信、流水线气泡、激活重算、ZeRO/FSDP 状态分片与 Megatron 组合并行账本。
outline: [2, 3]
---

<DlLessonHero number="31" module="GPU 与 AI Infra" title="分布式训练、ZeRO、FSDP 与 Megatron" question="模型、激活和优化器状态怎样跨设备切分；通信、流水线气泡、拓扑与故障又怎样限制扩展效率？" duration="建议 390–520 分钟" :outcomes="['计算数据、张量、流水线、上下文、专家并行的分组与通信', '解释 column/row parallel 与 transformer block 集合操作', '估算 pipeline bubble 与 micro-batch 调度', '区分 ZeRO stage 与 FSDP sharding strategy/生命周期', '设计 Megatron 组合并行、激活重算、checkpoint 与扩展实验']" />

多卡数量不是线性加速证明。分布式方案要同时满足容量、计算、通信、拓扑、同步语义和故障恢复；如果不先写 world/group/rank 和张量形状，框架名会掩盖错误。

## 贯穿案例：70B 模型在 64 GPU 上训练

单卡放不下模型状态和激活。节点内 8 GPU 有高速互联，节点间网络较慢。我们将 TP 尽量留在节点内，PP 跨连续层，DP/FSDP 跨副本，并用 micro-batch、重算和通信重叠满足容量/吞吐。

<DlQuiz kind="先判断" question="64 GPU 的训练吞吐一定是单卡 64 倍吗？" :options="['一定', '不一定，通信、气泡、负载、同步和输入会降低扩展效率', '只要 batch 相同就一定', 'GPU 之间不通信']" :answer="1" explanation="应报告 strong/weak scaling 与效率。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>把 64=DP×TP×PP 分解三种方案，标出每种主要集合通信是否跨节点，并估计气泡。</div>

## 1. Data Parallel 与集合通信

DP 每副本有完整模型，处理不同 micro-batch，反向后 all-reduce 梯度。Ring all-reduce 通信量每 rank 约 $2(P-1)S/P$；实际受 latency、bucket、拓扑和重叠影响。全局 batch = micro-batch×gradient accumulation×DP，改变 DP 时学习率/调度合同也可能变化。

```python
def ring_allreduce_bytes(message_bytes, ranks):
    return 2*(ranks-1)/ranks*message_bytes

assert ring_allreduce_bytes(1e9,8) == 1.75e9
```

<DlQuiz kind="DP" question="增加 gradient accumulation 是否增加 DP 通信次数/每 optimizer step？" :options="['通常可减少到 accumulation 完成后再同步，但实现需 no_sync/语义核对', '必然每 micro-step 同步', '会删除梯度', '与全局 batch 无关']" :answer="0" explanation="不必要的每步 all-reduce 会浪费通信。" />

## 2. Tensor Parallel 与 Megatron block

Megatron 将 MLP/attention 大线性层按 column/row 切。Column-parallel $Y=X[W_1...W_p]$ 可各 rank 得到输出分片；后续 row-parallel 对输入/权重相应分片并 all-reduce 部分和。attention heads 可按 TP 分，vocab parallel 需分布式 softmax/cross-entropy。

TP 每层通信频繁，优先放高速域。sequence/context parallel 进一步按序列切归一化/attention 状态，长上下文时降低激活但引入 all-gather/reduce-scatter 或 ring attention。

<DlQuiz kind="TP" question="为什么 TP 常优先放节点内高速互联？" :options="['每层可能有频繁集合通信，延迟/带宽敏感', 'TP 不用显存', '节点内没有通信', '只为命名方便']" :answer="0" explanation="跨慢网络会把每层同步暴露在关键路径。" />

## 3. Pipeline Parallel 与 micro-batch 气泡

PP 按连续层分 stage，micro-batches 流水通过。朴素 GPipe 的 fill/drain 气泡比例近似 $(P-1)/(M+P-1)$；1F1B 可降低激活峰值并改善调度，interleaved virtual stages 再缩短 bubble，但增加通信/实现复杂度。

<ParallelismPlannerLab />

```python
def bubble(stages, microbatches):
    return (stages-1)/(microbatches+stages-1)

assert bubble(4,32) < bubble(4,4)
assert bubble(1,8) == 0
```

<DlQuiz kind="PP" question="无限增大 micro-batch 数是否无代价？" :options="['无代价', '会增加调度/激活或 step 延迟，且每 micro-batch 过小降低 kernel 效率', '会删除气泡且无其他变化', '只影响 tokenizer']" :answer="1" explanation="气泡、内存、算子效率和收敛要一起优化。" />

## 4. ZeRO 与 FSDP：分片什么、何时 materialize

ZeRO-1 分 optimizer states，stage 2 再分 gradients，stage 3 再分 parameters。FSDP 以 module/flat parameter 为单位，在 forward 前 all-gather 完整参数、计算后 reshard，并以 reduce-scatter 聚合梯度；`FULL_SHARD` 接近 ZeRO-3，但配置/生命周期术语不完全相同。

<ZeroStateLab />

```python
def mixed_adam_gib(params_b, world, stage):
    total=params_b*1e9*16
    if stage==0: per=total
    elif stage==1: per=params_b*1e9*(8/world+8)
    elif stage==2: per=params_b*1e9*(10/world+6)
    else: per=total/world
    return per/2**30

assert mixed_adam_gib(70,8,3) < mixed_adam_gib(70,8,0)
```

<DlQuiz kind="ZeRO" question="ZeRO-3/FULL_SHARD 降低常驻参数显存的主要代价是什么？" :options="['运行中需 all-gather/reshard 参数，通信和预取调度增加', '模型不再有参数', '不需 checkpoint', '梯度自动正确']" :answer="0" explanation="容量通过通信和生命周期管理换取。" />

## 5. Activation checkpoint、offload 与组合并行

activation checkpoint 只保存边界，反向重算内部 forward，换显存为 FLOPs。CPU/NVMe offload 再换设备容量为 PCIe/存储流量。组合 DP/TP/PP/CP/EP 时，总 world size 与每组必须整除，专家 all-to-all、TP all-reduce 和 DP reduce-scatter 需要映射到物理拓扑并尽量重叠。

<DlQuiz kind="重算" question="activation checkpoint 主要减少哪类内存？" :options="['中间激活，代价是反向重算', '模型词表', '数据集', 'NCCL communicator']" :answer="0" explanation="不会直接消除权重/优化器状态。" />

## 6. Checkpoint、确定性与故障恢复

分布式 checkpoint 要记录模型/optimizer/RNG/dataloader、并行 topology、global step 和数据位置。sharded checkpoint 便于并行 IO，但 world-size 变化需 reshard。故障恢复要验证 loss 连续、样本不重/漏、scheduler 不回退；弹性训练改变 world size 时全局 batch/优化语义也可能变化。

<div class="dl-stop"><strong>停下来推一推 2</strong>中断发生在 optimizer step 与 checkpoint commit 之间，如何用原子 manifest/step ID 防止模型、优化器和 dataloader 不一致？</div>

## 7. 扩展效率与 profiler

Strong scaling 固定全局工作量增设备，理想时间反比；weak scaling 每设备工作量固定。报告吞吐、step time 分解、MFU、通信占比、bubble、straggler、显存和失败率。只报“训练成功”不能选择 topology。

<div class="dl-stop"><strong>停下来推一推 3</strong>从 8→16→32→64 GPU 画 strong-scaling；若 64 卡效率 55%，用 profiler 区分 all-reduce、PP bubble、输入和慢节点。</div>

## 8. 推荐论文阅读

<DlPaperCard title="Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" meta="Shoeybi et al. · 2019" priority="必读 · 张量并行" question="MLP/attention 怎样按 column/row 切分并最小化同步点？" reading="精读模型并行块与扩展实验；画出每层张量和集合通信。" href="https://arxiv.org/pdf/1909.08053" />

<DlPaperCard title="ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" meta="Rajbhandari et al. · 2020" priority="必读 · 状态分片" question="三个阶段分别消除 optimizer、gradient、parameter 的哪种复制？" reading="精读内存公式、通信分析与 ZeRO-R；用 70B/64 卡复算。" href="https://arxiv.org/pdf/1910.02054" />

<DlPaperCard title="PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" meta="Zhao et al. · 2023" priority="选读 · FSDP 工程" question="parameter flattening、prefetch、rate limiter、mixed precision 与 checkpoint 如何影响可用性？" reading="读设计经验、性能和限制；对照当前 PyTorch API 版本。" href="https://arxiv.org/pdf/2304.11277" />

## 9. 练习与门禁

画并行组；实现状态公式；复算 ring 流量/PP bubble；设计节点映射；做 activation 重算消融；恢复 checkpoint；报告 strong/weak scaling、MFU 和通信分解。

## 本章小结

- DP/TP/PP/CP/EP 切分不同张量和状态，组合必须匹配物理拓扑与通信。
- ZeRO/FSDP 用分片与动态 materialize 换容量，激活重算再用 FLOPs 换内存。
- 扩展必须用效率、通信、气泡、显存和恢复正确性证明，而非 GPU 数量。

<DlLessonProgress lesson-id="31" next-href="/courses/deep-learning/32-vllm-sglang" next-label="下一章：vLLM、SGLang 与推理服务" />
