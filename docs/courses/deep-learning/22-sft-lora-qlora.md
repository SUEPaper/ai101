---
course: deep-learning
edition: industrial
title: 指令微调、LoRA 与 QLoRA
description: 从对话模板和 SFT loss mask 进入全量微调、低秩适配、4-bit 量化底座、显存账本、合并与遗忘评价。
outline: [2, 3]
---

<DlLessonHero number="22" module="大语言模型" title="指令微调、LoRA 与 QLoRA" question="怎样把预训练模型变成遵循业务指令的助手，同时控制可训练参数、显存、量化误差和能力遗忘？" duration="建议 320–420 分钟" :outcomes="['定义 SFT 样本、chat template、packing 与 assistant-only loss 合同', '比较 full fine-tuning、adapter 与 LoRA 的状态成本', '计算 LoRA rank/target modules 的参数和合并行为', '解释 QLoRA 的 NF4、double quantization 与 compute dtype', '用冻结能力、格式、安全和领域切片评价适配']" />

参数高效不等于实验简单。LoRA/QLoRA 能显著减少可训练状态，但数据模板、loss mask、激活、底座权重与评测仍决定结果。本章先冻结“什么被监督”，再讨论“哪些参数被更新”。

## 贯穿案例：维修助手的 20k 条 SFT

样本含 system 安全约束、user 故障描述、assistant 结构化诊断。我们只对 assistant 内容计 loss，保留多轮边界；比较全量、LoRA、QLoRA 在领域成功率、通用能力、拒答、JSON 合规、峰值显存和 tokens/s 上的差异。

<DlQuiz kind="先判断" question="SFT 时把 system/user prompt 也计入 loss 是否一定错误？" :options="['一定', '不是数学上一定，但会改变训练目标；必须明确是否只监督 assistant，并验证行为', '永远最好', '与模板无关']" :answer="1" explanation="loss mask 是数据合同，不应由框架默认值暗中决定。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>把一轮 system/user/assistant 序列写成 input_ids 与 labels；标出 BOS、角色 token、assistant 内容、EOS、padding 的 loss mask。</div>

## 1. SFT 数据合同与模板

chat template 将角色、轮次和终止符编译为 token。训练与服务必须使用同一模板；重复 BOS、漏 EOS、角色 token 漂移会直接改变生成。packing 把多样本拼进一条序列时要保留边界，并决定是否允许跨样本 attention。

```python
tokens = ["<sys>", "安全", "<user>", "振动", "<assistant>", "停机", "检查", "<eos>"]
start = tokens.index("<assistant>") + 1
labels = [tok if i >= start else -100 for i, tok in enumerate(tokens)]
assert labels[:start] == [-100] * start
assert labels[-1] == "<eos>"
```

<DlQuiz kind="模板" question="训练模板有 `<assistant>`，服务模板漏掉它，最直接的风险是什么？" :options="['条件分布发生漂移，模型可能续写错误角色或格式', '参数量翻倍', 'KV cache 归零', 'LoRA rank 自动变大']" :answer="0" explanation="角色 token 是模型条件的一部分。" />

## 2. 全量微调与参数高效适配

全量微调更新全部权重，容量最大但需要梯度、优化器状态与分布式状态；还可能灾难性遗忘。Adapter 在层内加入瓶颈模块，prompt/prefix tuning 优化连续提示，LoRA 在既有线性层旁加入低秩增量。选择取决于数据量、部署是否允许多 adapter、目标任务和合并需求。

## 3. LoRA：低秩更新不是低秩权重

LoRA 冻结 $W$，训练 $\Delta W=BA$，其中 $A\in\mathbb R^{r\times d_{in}}$、$B\in\mathbb R^{d_{out}\times r}$，常用 $\alpha/r$ 缩放。假设是任务更新存在低内在秩，而不是原权重低秩。

<LoraBudgetLab />

```python
import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(size=(4, 6)); A = rng.normal(size=(2, 6)); B = rng.normal(size=(4, 2))
x = rng.normal(size=6); scale = 8/2
separate = W @ x + scale * B @ (A @ x)
merged = (W + scale * B @ A) @ x
assert np.allclose(separate, merged)
```

<DlQuiz kind="LoRA" question="LoRA rank r 增大通常直接增加什么？" :options="['可训练参数、优化器状态和更新容量', '底座参数数量', 'tokenizer 词表', '上下文上限']" :answer="0" explanation="效果是否提高仍需任务消融。" />

目标模块常选 attention 的 Q/V 或 Q/K/V/O，也可覆盖 MLP。不同架构命名不同，正则表达式误匹配可能导致“训练成功但没训想要的层”。发布 adapter 时必须记录 base checkpoint 精确 hash、target modules、rank、alpha、dropout 和 bias 策略。

## 4. QLoRA：量化底座，高精度更新

QLoRA 将冻结底座以 4-bit NF4 存储，计算时反量化到 BF16/FP16，LoRA 参数保持较高精度；double quantization 继续压缩量化 scale，paged optimizer 缓解显存峰值。它减少底座存储，不等同于 4-bit 训练所有梯度。

<QuantizationErrorLab />

```python
def symmetric_quantize(values, bits=4):
    qmax = 2**(bits-1)-1
    scale = max(abs(v) for v in values)/qmax
    codes = [round(v/scale) for v in values]
    return [c*scale for c in codes]

w = [-1.0, -.3, .2, .9]
q = symmetric_quantize(w)
assert len(q) == len(w) and max(abs(a-b) for a,b in zip(w,q)) < .12
```

<DlQuiz kind="QLoRA" question="4-bit QLoRA 中通常哪部分仍以较高精度训练？" :options="['LoRA adapter 及其梯度/优化器状态', '所有底座权重梯度', 'token ID', '数据文件']" :answer="0" explanation="量化底座冻结，adapter 承担更新。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>为 7B 模型分别列出 FP16 full FT、BF16 LoRA、4-bit QLoRA 的权重、梯度、master weight、Adam 状态和激活；哪些项不会随 LoRA 消失？</div>

## 5. 训练稳定、合并和多 adapter 服务

LoRA 学习率通常可高于 full FT，但仍需 warmup、梯度范数和 overfit-one-batch 检查。合并到高精度底座可移除推理旁路；合并到量化权重可能再次量化并累积误差。多 adapter 服务要处理 base 兼容、批内切换、缓存和访问权限。

<DlQuiz kind="合并" question="为什么不能把任意 LoRA adapter 加到同名但不同版本的底座？" :options="['增量是相对特定 W 学到的，形状相同也不保证语义/权重基准一致', '文件名太短', 'LoRA 没有矩阵', '所有底座完全相同']" :answer="0" explanation="base hash 是 adapter 制品合同。" />

## 6. 评价：领域收益与能力遗忘并列

训练 loss 下降只能说明模仿数据。应冻结领域任务、通用能力、安全拒答、格式、长上下文、多语言和校准切片；多种子报告方差，并对 SFT 训练集与评测做模板/近重复审计。领域提升若伴随危险建议或通用能力大降，不可发布。

<DlQuiz kind="评价" question="LoRA 可训练参数只有 0.1%，是否证明不会灾难性遗忘？" :options="['证明', '不证明，小参数更新仍可显著改变输出分布', '只在 rank=1 证明', '只需看训练 loss']" :answer="1" explanation="遗忘是行为变化，必须实测。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>设计 adapter 发布矩阵：领域成功率、通用回归、安全拒答、格式、P95 延迟、显存和 base/adapter hash 各设什么门槛？</div>

## 7. 推荐论文阅读

<DlPaperCard title="LoRA: Low-Rank Adaptation of Large Language Models" meta="Hu et al. · 2021" priority="必读 · 参数高效适配" question="低秩更新假设怎样验证，target modules/rank 如何影响质量与成本？" reading="精读方法、秩消融与合并；复算可训练参数。" href="https://arxiv.org/pdf/2106.09685" />

<DlPaperCard title="QLoRA: Efficient Finetuning of Quantized LLMs" meta="Dettmers et al. · 2023" priority="必读 · 量化微调" question="NF4、double quantization、paged optimizer 分别解决什么显存问题？" reading="精读内存分解、计算 dtype 与 Guanaco 评价；区分存储位宽和算术精度。" href="https://arxiv.org/pdf/2305.14314" />

<DlPaperCard title="Finetuned Language Models Are Zero-Shot Learners" meta="Wei et al. · 2022" priority="选读 · 指令混合" question="跨任务 instruction tuning 如何改善未见任务，任务/模板混合怎样构造？" reading="读 FLAN 任务簇、模板和 held-out 评价，检查训练任务污染边界。" href="https://arxiv.org/pdf/2109.01652" />

## 8. 练习与门禁

实现 chat labels；计算 full/LoRA 状态；合并数值对照；做 rank×target 消融；量化误差切片；验证 base hash；报告领域收益、通用回归和安全门禁。

## 本章小结

- SFT 首先是模板与 loss mask 合同，然后才是优化问题。
- LoRA 压缩更新子空间，QLoRA 再压缩冻结底座存储；激活、数据与评测不会消失。
- rank、target、量化和 base hash 必须作为制品记录，发布同时检查收益、遗忘、安全和成本。

<DlLessonProgress lesson-id="22" next-href="/courses/deep-learning/23-rlhf-dpo-reasoning" next-label="下一章：RLHF、DPO 与推理模型" />
