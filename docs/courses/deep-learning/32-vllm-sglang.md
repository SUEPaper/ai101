---
course: deep-learning
edition: industrial
title: vLLM、SGLang 与大模型推理服务
description: 从 prefill/decode、continuous batching、PagedAttention、RadixAttention、prefix cache 进入 vLLM/SGLang 服务接口、结构化生成与公平基准。
outline: [2, 3]
---

<DlLessonHero number="32" module="GPU 与 AI Infra" title="vLLM、SGLang 与大模型推理服务" question="在线请求长度不断变化时，调度器怎样管理 KV、共享前缀和结构化程序；又该怎样公平比较 vLLM 与 SGLang？" duration="建议 370–490 分钟" :outcomes="['区分 prefill/decode、TTFT/TPOT/E2E 和 token/request throughput', '解释 continuous batching 与 chunked prefill 的调度权衡', '说明 PagedAttention 的分页 KV 与 RadixAttention 的前缀复用', '比较 vLLM/SGLang 的共同系统问题和关键设计入口', '设计版本/模型/硬件/请求集固定的质量—延迟—吞吐—显存基准']" />

推理服务不是把 `model.generate()` 包成 HTTP。请求到达随机、prompt/output 长度不同，prefill 与 decode 争用 GPU，KV 生命周期碎片化，adapter/结构化输出/工具调用改变 batch 兼容。框架性能结论必须绑定版本和 workload。

## 贯穿案例：维修助手混合流量

短请求占 70%，长手册 RAG 占 25%，结构化工具调用占 5%；有共享 system prompt 和手册前缀。SLO 为 P95 TTFT<1.5s、P95 TPOT<50ms、失败率<0.1%。我们比较 vLLM/SGLang 时固定 checkpoint、quant、tokenizer、请求到达和输出质量。

<DlQuiz kind="先判断" question="离线 batch token/s 更高是否证明在线 P95 更好？" :options="['证明', '不证明，在线排队、到达分布、长度和调度会改变 TTFT/TPOT 尾延迟', '只要 batch=1 就证明', 'P95 与吞吐相同']" :answer="1" explanation="吞吐与尾延迟是不同目标。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>画一次请求的 queue→tokenize→prefill→first token→decode→finish；在哪些时间段属于 TTFT、TPOT、E2E？</div>

## 1. Prefill、Decode 与服务指标

Prefill 对 prompt 并行计算，常较 compute-bound；decode 每步生成一个 token，需读取权重/KV，常 bandwidth/scheduler-bound。TTFT 包含排队与 prefill，TPOT 是后续 token 间隔，E2E 约 TTFT+(output−1)×TPOT 但流式/队列会偏离。

```python
latencies_ms=[420,510,490,900,470,530,1200,480,505,610]
def percentile(xs,p):
    s=sorted(xs); i=min(len(s)-1, max(0, int((len(s)-1)*p)))
    return s[i]

assert percentile(latencies_ms,.5)==505
assert percentile(latencies_ms,.95)==900
```

<DlQuiz kind="指标" question="请求输出 token 数翻倍且 TPOT 不变，哪个指标通常最直接增长？" :options="['端到端延迟', '模型参数', '词表', 'TTFT 必然翻倍']" :answer="0" explanation="TTFT 主要由排队/prefill 决定。" />

## 2. Continuous batching 与 chunked prefill

静态 batching 等整批完成，短请求被最长请求拖住。Continuous/in-flight batching 让完成请求退出、新请求加入 decode step，提高利用率。长 prefill 若独占 GPU 会阻塞 decode，chunked prefill 将其分块与 decode 交错，改善 inter-token SLO，但可能增加 TTFT/调度开销。

<ServingSchedulerLab />

<DlQuiz kind="调度" question="continuous batching 的主要收益是什么？" :options="['按活跃序列动态补批，减少长度不齐造成的空闲', '删除 KV Cache', '保证所有请求同延迟', '无需 admission control']" :answer="0" explanation="在线动态性提高利用率，也需公平/优先级策略。" />

## 3. PagedAttention 与分页 KV 管理

连续大块 KV 预留会因未知输出长度产生内部/外部碎片。vLLM 的 PagedAttention 将每序列 KV 分为固定 block，以逻辑 block table 映射物理 block，支持非连续分配、copy-on-write 和共享；attention kernel 按表读取。页太小 metadata/寻址开销高，太大尾块浪费。

```python
import math
def kv_blocks(tokens, block_size):
    return math.ceil(tokens/block_size)

assert kv_blocks(33,16)==3
assert kv_blocks(32,16)*16-32==0
assert kv_blocks(33,16)*16-33==15
```

<DlQuiz kind="PagedAttention" question="PagedAttention 主要解决什么？" :options="['KV Cache 的动态分页、碎片与共享管理', '训练反向传播', 'tokenizer 去重', '模型安全']" :answer="0" explanation="它借鉴虚拟内存分页思想管理 KV。" />

## 4. RadixAttention 与前缀复用

SGLang 的 RadixAttention 用 radix tree 组织 prompt/KV 前缀，自动匹配和复用 system prompt、few-shot、RAG/程序分支；缓存节点需引用计数/淘汰。前缀只有 token 完全一致、模型/adapter/位置/量化和权限兼容时才可复用。

<PrefixCacheLab />

```python
prompts=[("system","manual-A","q1"),("system","manual-A","q2"),("system","manual-B","q3")]
shared=sum(1 for p in prompts if p[:2]==("system","manual-A"))
assert shared==2
```

<DlQuiz kind="Prefix cache" question="两个请求文本肉眼相同就一定能命中 KV 前缀吗？" :options="['一定', '不一定，tokenizer/template/model/adapter/position 和精确 token 序列都要兼容', '缓存按字符截图', '只有输出参与']" :answer="1" explanation="KV 是特定模型与 token 位置的状态。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>system prompt 更新一个空格、adapter 改变、租户不同三种场景，哪些可共享 token，哪些绝不能共享 KV？</div>

## 5. vLLM 与 SGLang：共同问题、不同入口

vLLM 以 PagedAttention、调度与广泛模型/服务兼容为核心入口，围绕 OpenAI-compatible server、tensor parallel、quantization、prefix caching 等发展。SGLang 除高性能 runtime 外，强调 structured language model programs、RadixAttention、前缀复用、约束生成与复杂控制流。两者都快速演进，具体支持矩阵需查当前官方版本，不能把某次 benchmark 永久化成胜负。

选择问题包括：目标模型/量化/adapter 是否支持，结构化/工具程序复杂度，请求前缀复用率，TP/多节点需求，可观测性，部署/升级成本，以及真实 SLO 下的 Pareto。也可选择 TensorRT-LLM、TGI 等；课程重点是比较方法，不是框架忠诚。

<DlQuiz kind="框架比较" question="公平比较 vLLM/SGLang 最少要固定什么？" :options="['版本、模型/量化、硬件、请求到达/长度、解码和质量检查', '只固定框架名', '只固定峰值 FLOPS', '只跑一个 prompt']" :answer="0" explanation="系统结论绑定 workload 与版本。" />

## 6. Quantization、adapter 与 structured output

权重量化降低容量/带宽，但 kernel、精度和不同 shape 决定收益；KV quantization 直接影响长上下文。多 LoRA 服务要调度 adapter 权重和 batch compatibility。Grammar/JSON Schema 会动态屏蔽 token，改变每步开销；工具调用/多阶段程序还会打断 GPU 批次并增加 CPU/网络等待。

## 7. Admission、取消、隔离与可观测性

Admission control 依据 token budget/KV/SLO 拒绝或排队，不能让超长请求拖垮全体。客户端取消应尽快回收 KV；OOM 要有降级而非进程崩溃。Prefix cache 与日志按租户隔离，prompt/输出可能含隐私。指标包括队列、active/running/swapped、KV 使用/碎片/命中、prefill/decode token/s、TTFT/TPOT/E2E 分位、失败/取消和 GPU 功耗。

<DlQuiz kind="过载" question="系统接近 KV 容量时继续无限接收请求的典型后果是什么？" :options="['排队/换入换出和尾延迟崩溃，甚至 OOM', '吞吐无限增加', '模型自动缩小', 'tokenizer 变快']" :answer="0" explanation="需要 admission 与 backpressure。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>设计 open-loop Poisson 与 trace replay 两类基准；为什么 closed-loop 客户端会因服务变慢而自动降压、掩盖过载？</div>

## 8. 公平服务基准

固定容器/driver/CUDA/framework commit、模型 hash、dtype/quant、TP、max context、GPU clocks；请求集公开 prompt/output 长度与共享前缀，开放/闭环到达率，warm-up 后跑足够时间。核对输出 token/质量与错误，报告 request/s、input/output token/s、TTFT/TPOT/E2E P50/P95/P99、峰值显存、KV 命中、失败率和成本。

## 9. 推荐论文阅读

<DlPaperCard title="Efficient Memory Management for Large Language Model Serving with PagedAttention" meta="Kwon et al. · 2023" priority="必读 · vLLM" question="KV 碎片从何而来，block table、copy-on-write 与调度怎样改善吞吐？" reading="精读 PagedAttention 设计、Orca 对照、实验 workload 与质量等价。" href="https://arxiv.org/pdf/2309.06180" />

<DlPaperCard title="SGLang: Efficient Execution of Structured Language Model Programs" meta="Zheng et al. · 2024" priority="必读 · SGLang" question="RadixAttention、compressed FSM 与 API speculative execution 分别优化什么程序模式？" reading="精读语言/runtime 接口、缓存树和 benchmark；核对当前版本差异。" href="https://arxiv.org/pdf/2312.07104" />

<DlPaperCard title="Orca: A Distributed Serving System for Transformer-Based Generative Models" meta="Yu et al. · 2022" priority="必读 · 迭代级调度" question="iteration-level scheduling 与 selective batching 怎样处理生成长度不齐？" reading="读调度、执行引擎和 workload；与 continuous batching 对照。" href="https://www.usenix.org/system/files/osdi22-yu.pdf" />

## 10. 练习与发布门禁

复算 TTFT/TPOT；实现 block 账本；测前缀命中；开/闭环压测；注入取消/OOM/超长；固定质量；比较 vLLM/SGLang 的 P50/P95/P99、吞吐、显存、命中与失败。

## 本章小结

- 在线服务要分离 prefill/decode 与 TTFT/TPOT/E2E，吞吐不能代替尾延迟。
- PagedAttention 解决动态 KV 分页，RadixAttention 强化共享前缀组织；缓存必须版本/权限隔离。
- vLLM 与 SGLang 的性能结论只在固定版本、模型、硬件和请求集内成立，公平基准必须同时检查质量和 SLO。

<DlLessonProgress lesson-id="32" />
