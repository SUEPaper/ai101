---
course: deep-learning
edition: industrial
title: BERT、GPT 与 T5 预训练范式
description: 比较 encoder-only、decoder-only、encoder–decoder 及 MLM、causal LM、span corruption 的信息边界、损失位置、任务接口与成本。
outline: [2, 3]
---

<DlLessonHero number="19" module="大语言模型" title="BERT、GPT 与 T5 预训练范式" question="同一批无标注文本经过不同 corruption、visibility 和 loss mask 后，为何会得到擅长理解、续写或条件生成的三类模型？" duration="建议 300–390 分钟" :outcomes="['用信息可见性和损失位置区分 MLM、causal LM 与 span corruption', '比较 encoder-only、decoder-only、encoder–decoder 的任务接口和注意力成本', '解释 BERT、GPT、T5 的预训练—适配路径', '审计 special token、attention mask、loss mask 与有效 token 预算', '为分类、检索、对话和文本转换选择可证伪的预训练范式']" />

自注意力只是算子，预训练目标才规定模型看见什么、预测什么。BERT、GPT 与 T5 的差异不应背成模型清单；它们分别代表双向编码、因果生成和输入—输出转换三种概率与产品接口。

## 贯穿案例：维修知识库的分类、补全与规范化

同一批维修记录要支持故障分类、工单续写和口语描述到规范步骤的转换。我们统一 tokenizer、时间切分和总训练 FLOPs，只改变目标与架构；比较时同时报告目标任务、有效监督 token、输入/输出长度和端到端延迟。

<DlQuiz kind="先判断" question="只把 causal mask 改成双向可见，就能把 GPT 等价变成 BERT 吗？" :options="['能，其他完全相同', '不能；corruption、loss 位置、输入特殊 token 和适配接口也不同', '只能在英文中能', '只需删除 softmax']" :answer="1" explanation="可见性只是预训练合同的一部分。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>对“泵温升高立即停机”画出 causal LM、15% MLM 和一个被遮 span 的输入、可见边与 loss 位置。</div>

## 1. 三种目标：预测事件单位不同

<PretrainingObjectiveLab />

causal LM 最小化每个下一 token 的 NLL，几乎每个位置都是监督；MLM 用 `[MASK]`、随机 token 或保留原 token 替换被抽中位置，只在这些位置计 loss；T5 把连续 span 换成 sentinel，并让 decoder 生成被删除 span。三者的“15% corruption”不能与“15% 有效训练 token”混为一谈。

```python
def causal_mask(n):
    return [[int(j <= i) for j in range(n)] for i in range(n)]

m = causal_mask(5)
assert sum(map(sum, m)) == 15
assert m[1] == [1, 1, 0, 0, 0]
```

<DlQuiz kind="监督密度" question="在长度相同且不重采样的序列上，MLM 通常比 causal LM 的直接 loss 位置怎样？" :options="['更多', '更少，只在抽中的遮蔽位置', '完全一样', '没有 loss']" :answer="1" explanation="这会影响样本效率与训练预算的公平比较。" />

## 2. BERT：双向编码与下游 head

BERT 将 `[CLS]`、segment/type 与 position 表示相加，通过深层双向 encoder 得到每个位置的上下文状态。MLM 缓解静态词表示限制；原始 NSP 试图区分句对关系，但后续工作表明数据与目标设计可替代它。分类可接 `[CLS]` head，抽取任务使用 token head，双编码检索则需重新规定 pooling 与负例。

MLM 的 `[MASK]` 在下游不存在，产生 pretrain–finetune mismatch；动态 masking 能增加同一文本的监督覆盖，却不消除数据重复和评测污染。

```python
tokens = list("轴承温度升高")
chosen = [1, 4]
corrupted = ["[MASK]" if i in chosen else t for i, t in enumerate(tokens)]
labels = [t if i in chosen else None for i, t in enumerate(tokens)]
assert sum(x is not None for x in labels) == 2
```

<DlQuiz kind="信息边界" question="BERT 做离线分类时能否读取目标 token 右侧？" :options="['可以，encoder self-attention 通常双向可见', '不可以，永远因果', '只在训练时不可以', '只看 `[CLS]` 本身']" :answer="0" explanation="这也意味着它不能直接无泄漏地承担左到右在线生成。" />

## 3. GPT：统一为下一 token 预测

GPT 系列把自然语言任务格式化为前缀续写。架构简单统一，预训练监督稠密，能自然进行 few-shot/in-context learning；代价是分类、抽取也必须转成生成或另接 head，长输入—输出共同进入因果上下文和 KV cache。

训练时要 shift labels，SFT 时还可能只对 assistant span 计 loss。若 prompt template、BOS/EOS 或角色 token 在部署变化，概率合同已经改变。

<ArchitectureInterfaceLab />

<DlQuiz kind="接口选择" question="需要流式生成长答案时，三类架构中哪个最直接？" :options="['Encoder-only', 'Decoder-only causal LM', '只用静态 embedding', 'MLM 随机反复填空一定最直接']" :answer="1" explanation="因果分解原生支持逐 token 解码与 KV cache。" />

## 4. T5：把任务统一为 text-to-text

T5 encoder 读取破坏后的完整输入，decoder 自回归生成 sentinel 标记的缺失 span。下游分类也输出文本标签，翻译、摘要和问答共享接口。优点是输入理解与输出生成解耦；成本包含 encoder self-attention、decoder causal attention 和 cross-attention。

```python
source = ["泵", "温", "<extra_id_0>", "停", "机"]
target = ["<extra_id_0>", "升", "高", "<extra_id_1>"]
assert source.count("<extra_id_0>") == 1
assert target[0] == "<extra_id_0>" and target[-1] == "<extra_id_1>"
```

<DlQuiz kind="Span corruption" question="T5 sentinel token 的作用是什么？" :options="['表示并排序被删除 span 的边界，使 decoder 重建多个片段', '把所有输入变成 padding', '替代位置编码', '计算优化器状态']" :answer="0" explanation="sentinel 把输入中的缺口与输出中的恢复片段对齐。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>同样 512 输入、128 输出，估算三种架构 attention 矩阵项；哪些常数被这个 $n^2$ 代理忽略？</div>

## 5. 选择范式而不是追随名称

编码检索偏好稳定、低延迟的双向表示；开放生成偏好 decoder-only；高质量条件转换可能受益于 encoder–decoder。选择还受预训练 checkpoint、serving 生态、序列长度、数据量和授权影响。必须用同一业务切分比较，而不是引用不同论文表格。

<div class="dl-stop"><strong>停下来推一推 3</strong>为维修知识库建立三个冻结评测：分类、引用式补全、口语到规范步骤。定义每个任务的质量、吞吐、显存和泄漏门禁。</div>

## 6. 推荐论文阅读

<DlPaperCard title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" meta="Devlin et al. · 2019" priority="必读 · Encoder-only" question="MLM/NSP 的输入破坏、loss 位置与下游适配分别怎样实现？" reading="精读第 3 节与任务图；把 `[MASK]` mismatch、数据和总 FLOPs 记入复核表。" href="https://arxiv.org/pdf/1810.04805" />

<DlPaperCard title="Language Models are Unsupervised Multitask Learners" meta="Radford et al. · 2019" priority="必读 · Decoder-only" question="纯 causal LM 如何把任务条件放进文本前缀，并表现出 zero-shot 迁移？" reading="读训练目标、WebText 数据与任务格式；区分生成接口与事实保证。" href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" />

<DlPaperCard title="Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" meta="Raffel et al. · 2020" priority="必读 · Encoder–decoder" question="统一 text-to-text 框架怎样比较目标、数据混合、模型规模与任务迁移？" reading="精读 span corruption、mixture 与 scaling 实验；不要只读榜单。" href="https://arxiv.org/pdf/1910.10683" />

## 7. 练习与发布门禁

实现三种 corruption；核对 attention/loss mask；用相同 token/FLOPs 比较；记录 special-token 协议；做分类、生成、转换切片；检查 prompt template 漂移与 benchmark 重复。

## 本章小结

- BERT、GPT、T5 的核心差异是 corruption、visibility、loss 和适配接口。
- 双向编码适合理解，因果分解适合生成，encoder–decoder 适合条件转换；没有脱离任务和成本的普遍赢家。
- 有效监督 token、special token、mask 和数据切分是可复现预训练合同。

<DlLessonProgress lesson-id="19" next-href="/courses/deep-learning/20-modern-llm-architecture" next-label="下一章：现代 LLM 架构" />
