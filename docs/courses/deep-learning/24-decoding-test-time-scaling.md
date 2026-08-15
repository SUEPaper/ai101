---
course: deep-learning
edition: industrial
title: 解码、投机解码与 Test-Time Scaling
description: 从 greedy、beam、temperature、top-k/top-p 进入重复控制、投机解码、自一致性和统一推理预算下的 test-time scaling。
outline: [2, 3]
---

<DlLessonHero number="24" module="大语言模型" title="解码、投机解码与 Test-Time Scaling" question="训练好的条件分布怎样变成一条输出；我们又该怎样用更多采样、搜索或验证换质量，而不掩盖延迟与 token 成本？" duration="建议 300–400 分钟" :outcomes="['区分 greedy、beam 与随机采样的目标和失效', '推导 temperature、top-k、top-p 对分布支持和熵的影响', '解释投机解码为何保持 target 分布及何时无加速', '为 self-consistency、reranking 与 test-time scaling 建立预算合同', '按质量、TTFT、TPOT、token、重复和终止评价解码']" />

训练给出 $p_\theta(x_t|x_{<t})$，解码把每步分布、停止条件和外部验证器变成最终序列。相同 checkpoint 仅改变 decoder，就可能显著改变准确率、创造性、重复、延迟和安全；因此解码配置属于模型制品。

## 贯穿案例：维修步骤生成

系统要生成最多 12 步维修流程。高风险步骤要求稳定、有证据、终止明确；创意多样性不是首要目标。我们在同一 300-output-token 上限下比较 greedy、nucleus、beam、投机解码与多样本验证。

<DlQuiz kind="先判断" question="temperature=0 是否等价于从一个合法概率分布随机采样？" :options="['严格等价', '通常实现为 greedy/argmax 极限或特殊分支，不应假设数值 softmax 可直接除以 0', '会得到均匀分布', '会扩大 top-p']" :answer="1" explanation="实现需显式处理零温度。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>给定 logits `[3,2,1,0]`，手算 $T=1$ 与 $T=0.5$ 的概率排序、熵变化；top-k=2、top-p=0.8 各保留什么？</div>

## 1. Greedy 与 beam：局部最优不等于序列最优

Greedy 每步取 argmax，确定且快，却可能早期走进差前缀。Beam search 同时保留 $B$ 个累计 log-prob 前缀，适合翻译等相对约束任务；长度惩罚不足会偏短，beam 过大可能降低开放生成质量。最大似然序列也不等于最有用或最真实答案。

```python
import math

paths = {"短": [-.1, -.2], "长": [-.08, -.12, -.18, -.2]}
raw = {k: sum(v) for k,v in paths.items()}
normalized = {k: sum(v)/(len(v)**.7) for k,v in paths.items()}
assert max(raw, key=raw.get) == "短"
assert max(normalized, key=normalized.get) in paths
```

<DlQuiz kind="Beam" question="增大 beam width 是否保证业务正确率单调提升？" :options="['保证', '不保证；搜索更接近模型高概率序列，但模型概率和业务效用可能错位', '只会降低内存', '等价于 temperature 增大']" :answer="1" explanation="搜索误差与模型/效用误差不同。" />

## 2. Temperature、top-k、top-p 与重复控制

<DecodingDistributionLab />

temperature 在截断前重标 logits；top-k 固定候选数，top-p/nucleus 选择累计概率达到阈值的最小集合，候选数随分布尖锐度变化。典型实现先处理 repetition/frequency/presence penalties，再 temperature、截断、重归一化与采样；顺序不同会产生不同输出。

```python
import math

def softmax(logits, temperature=1.0):
    z = [x/temperature for x in logits]; m = max(z)
    ex = [math.exp(x-m) for x in z]; s = sum(ex)
    return [x/s for x in ex]

p1, p05 = softmax([3,2,1],1), softmax([3,2,1],.5)
assert p05[0] > p1[0] and abs(sum(p05)-1) < 1e-12
```

<DlQuiz kind="Nucleus" question="top-p=0.9 每步是否总保留相同数量 token？" :options="['是', '否，集合大小随当前概率分布变化', '只保留一个', '保留整个词表']" :answer="1" explanation="这正是 nucleus 与固定 top-k 的差异。" />

重复惩罚可能阻止循环，也会破坏代码变量、术语和合法列表重复。EOS、stop strings、最大 token、JSON grammar 和安全终止必须一起测试；按字符截断可能切坏 UTF-8 或 JSON。

## 3. 投机解码：草稿提议，目标验证

<SpeculativeDecodeLab />

draft model 一次生成多个候选，target 对整段并行评分，并按接受—校正规则保留 target 的原始采样分布。draft 越接近 target，接受率越高；太大则 draft 成本抵消收益，太小/异域则频繁拒绝。加速取决于 batch、带宽、长度和 kernel，不是只看接受率。

```python
def speedup_proxy(draft_tokens, acceptance, draft_cost=.15):
    accepted = draft_tokens*acceptance
    return (accepted+1)/(draft_tokens*draft_cost+1)

assert speedup_proxy(5,.8) > speedup_proxy(5,.2)
assert speedup_proxy(12,.05) < 1
```

<DlQuiz kind="投机解码" question="正确实现的 speculative decoding 会把 target 分布近似成 draft 分布吗？" :options="['会', '不会，接受与校正规则保持 target 分布；draft 只提速', '只保留 draft top-1', '删除 target']" :answer="1" explanation="质量差异应来自实现/数值/配置，而不是算法目标。" />

## 4. Self-consistency、reranking 与 Test-Time Scaling

对可验证任务可采样 $K$ 条推理路径，按最终答案投票、verifier 或 reward rerank。更多样本提高找到正确解的机会，但高度相关样本收益递减；verifier 可能被欺骗。比较必须固定总生成 token、墙钟或费用，不能让一个模型无限采样。

<DlQuiz kind="计算预算" question="模型 A pass@1=60%，模型 B 用 64 次采样得到 pass@64=90%，能直接称 B 单次推理更强吗？" :options="['能', '不能，需统一采样/token/墙钟预算并同时报告 pass@k 曲线', '只需看输出长度', 'pass@k 与预算无关']" :answer="1" explanation="test-time compute 是评价条件。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>在 10k token 总预算下比较 1×长推理、10×短采样+投票、5×采样+verifier；定义成功、成本和错误相关性。</div>

## 5. 在线评价与可复现配置

记录 tokenizer/checkpoint、随机种子、temperature、top-k/p、penalties、beam、max tokens、stop、draft/target 版本和 structured decoding。质量报告至少包含任务成功、校准、重复率、长度、无效终止；系统报告 TTFT、TPOT、P50/P95、吞吐和成本。

<div class="dl-stop"><strong>停下来推一推 3</strong>构造“安全但重复”“流畅但提前 EOS”“正确但超预算”“JSON 合法但字段错误”四个失败样本，说明单一平均准确率漏掉什么。</div>

## 6. 推荐论文阅读

<DlPaperCard title="The Curious Case of Neural Text Degeneration" meta="Holtzman et al. · 2020" priority="必读 · Nucleus sampling" question="最大似然/beam 为何产生退化文本，top-p 怎样适应分布尾部？" reading="精读退化指标、人类评价和 nucleus 算法；复核不同任务边界。" href="https://arxiv.org/pdf/1904.09751" />

<DlPaperCard title="Fast Inference from Transformers via Speculative Decoding" meta="Leviathan et al. · 2023" priority="必读 · 无损加速" question="接受—拒绝与修正分布怎样保持 target 采样，速度模型依赖哪些量？" reading="读算法、正确性和 wall-time 实验；复算接受率敏感性。" href="https://arxiv.org/pdf/2211.17192" />

<DlPaperCard title="Self-Consistency Improves Chain of Thought Reasoning in Language Models" meta="Wang et al. · 2023" priority="选读 · 多路径推理" question="多样推理路径与答案投票为何可能优于 greedy CoT，成本如何报告？" reading="读采样配置、任务、投票和增益随样本数变化。" href="https://arxiv.org/pdf/2203.11171" />

## 7. 练习与门禁

实现稳定 temperature/top-p；写 beam 长度惩罚；复算投机接受；画 pass@k–token 曲线；测试 EOS/stop/JSON；报告质量—延迟—成本 Pareto。

## 本章小结

- 解码把模型概率变成产品行为，配置必须版本化。
- top-k/p、temperature 和 penalties 改变支持与熵；beam 更强搜索不保证更高业务效用。
- 投机解码保持 target 分布；test-time scaling 的收益必须与采样、token、延迟和 verifier 风险一起报告。

<DlLessonProgress lesson-id="24" next-href="/courses/deep-learning/25-rag" next-label="下一章：RAG 与知识边界" />
