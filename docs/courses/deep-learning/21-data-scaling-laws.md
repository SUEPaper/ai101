---
course: deep-learning
edition: industrial
title: 预训练数据工程与 Scaling Law
description: 建立来源、许可、清洗、去重、混合、污染和 token lineage，并用 Kaplan/Chinchilla 讨论参数—数据—计算最优边界。
outline: [2, 3]
---

<DlLessonHero number="21" module="大语言模型" title="预训练数据工程与 Scaling Law" question="万亿 token 从哪里来、彼此有多独立、是否污染评测；模型参数与 token 又怎样在固定计算下分配？" duration="建议 330–430 分钟" :outcomes="['设计含来源、许可、hash、过滤版本和删除链路的数据 lineage', '解释精确/近似去重、质量过滤和混合权重的失效条件', '审计 benchmark contamination 与时间泄漏', '使用经验 scaling law 做固定算力比较而非外推真理', '估算训练 FLOPs、MFU、墙钟和失败实验总成本']" />

模型架构是显式代码，数据管线往往是隐式模型。来源混合、清洗、去重和 tokenizer 会改变训练分布；若只记录“训练了 1T token”，实验无法复现，也无法处理版权删除、隐私或污染。

## 贯穿案例：构建 140B-token 工业语料

来源包括公开手册、维修工单、代码和合成问答。发布前必须回答：每个 token 的来源/许可是什么，近重复怎样处理，工单如何脱敏，评测集何时冻结，混合采样怎样复现，删除请求如何传播到派生 shard。

<DlQuiz kind="先判断" question="训练 token 翻倍是否等于独立信息翻倍？" :options="['总是', '不一定，近重复、模板、低质量和重采样会降低有效独立信息', '只要文件名不同就等于', 'tokenizer 会保证']" :answer="1" explanation="raw token count 不是信息量证书。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>画出 source→snapshot→normalize→PII filter→dedup→quality→mixture→shard→checkpoint 的 lineage；每层记录哪些 hash 和版本？</div>

## 1. 数据合同：来源、许可、隐私与删除

每个样本至少携带 source URI/collection、抓取时间、许可/授权、内容 hash、语言、质量/安全标记和管线版本。个人/企业敏感数据需要最小化、脱敏、访问控制与审计；“网页可访问”不等于可任意训练。删除必须能定位原始项、近重复与派生 shard，并明确已训练 checkpoint 的处理政策。

## 2. 清洗、去重与质量过滤

精确 hash 去重只能删相同字节；规范化后 hash 能处理轻微格式差异，却可能误并业务不同版本。MinHash/LSH、n-gram 或 embedding 可做近重复候选，再设阈值和域内保留规则。文档、段落、跨数据集去重作用不同。

<DedupContaminationLab />

```python
import hashlib
import unicodedata

def normalized_hash(text):
    clean = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()

assert normalized_hash("ＡI  手册") == normalized_hash("ai 手册")
```

<DlQuiz kind="去重" question="为什么跨训练集与评测集去重比只在训练集内部去重更关键？" :options="['能发现潜在评测答案泄漏', '能增大词表', '会自动提高所有任务', '能替代时间切分']" :answer="0" explanation="训练内部去重控制频次，跨集合去重还控制污染。" />

质量分类器常偏向主流语言和既有写作风格；困惑度过滤可能删除罕见但重要的专业文档。必须按来源/语言/时间/长度/主题审计保留率，并保留原始统计而非只留筛后数据。

## 3. 混合权重与有效采样

不同域规模相差巨大，直接按 token 比例会淹没小而高价值的代码、数学或领域语料；温度采样、上采样和 curriculum 改变每个样本的期望重复次数。报告应同时给原始 token、采样概率、epoch/repetition 和最终见到的 token。

```python
sizes = {"web": 1000, "manual": 20, "code": 100}
temperature = .5
weights = {k: v**temperature for k, v in sizes.items()}
z = sum(weights.values())
probs = {k: v/z for k, v in weights.items()}
assert probs["manual"] > sizes["manual"] / sum(sizes.values())
assert abs(sum(probs.values()) - 1) < 1e-12
```

<DlQuiz kind="混合" question="小语料上采样的主要风险是什么？" :options="['重复记忆、过拟合与域偏置', '一定减少训练步数', '让 tokenizer 失效', '完全没有风险']" :answer="0" explanation="采样概率必须与独立样本数一起看。" />

## 4. 评测污染与时间切分

污染包括 benchmark 原文、答案、解题讲解、近似翻译和模板变体。仅 substring 搜索会漏掉改写，embedding 近邻又有误报；应结合 exact/n-gram/MinHash/语义候选和人工复核。冻结 cutoff 后出现的新题能降低直接泄漏，但不消除同模板或人工记忆。

<DlQuiz kind="污染" question="模型在公开 benchmark 得分很高但私有同分布测试下降，首先应查什么？" :options="['评测污染、模板重合与选择偏差', 'GPU 风扇', '只调 temperature', '把测试并入训练']" :answer="0" explanation="公开集长期曝光会破坏独立性。" />

## 5. Scaling Law：经验关系，不是自然常数

Kaplan 等观察到 loss 随参数、数据和计算呈幂律区间；Chinchilla 强调许多大模型在固定计算下 token 不足，主张更平衡地扩大 $N$ 与 $D$。经验式依赖模型族、数据质量、优化、tokenizer 和训练区间，不能把拟合指数外推到任意规模。

<ScalingBudgetLab />

训练 dense Transformer 常用 $\mathrm{FLOPs}\approx6ND$ 粗估，其中 $N$ 为非 embedding 参数、$D$ 为训练 token。墙钟还取决于集群峰值和 Model FLOPs Utilization；数据加载、通信、故障与 checkpoint 都会降低实际利用率。

```python
N, D = 7e9, 140e9
training_flops = 6*N*D
peak, mfu = 100e15, .40
ideal_days = training_flops/(peak*mfu)/86400
assert training_flops == 5.88e21
assert ideal_days > 1
```

<DlQuiz kind="计算最优" question="固定训练 FLOPs 下，只增参数不增 token 的典型风险是什么？" :options="['模型欠训练，参数容量未被数据充分利用', '一定得到最低 loss', '推理成本下降', '数据污染消失']" :answer="0" explanation="计算最优讨论 N 与 D 的联合分配。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>同为 $6ND$，大模型少 token 与小模型多 token 的训练成本相近；为什么推理成本、数据重复和下游能力仍不同？</div>

## 6. 规模化实验与经济性

pilot run 要验证 loss 曲线、数据配比、梯度稳定、吞吐和 checkpoint 恢复，再外推全量。总预算必须包含数据处理、失败 runs、超参、评测和长期推理。最小训练 loss 不一定最小全生命周期成本。

<div class="dl-stop"><strong>停下来推一推 3</strong>构造 1%、5%、20% 规模 pilot，哪些量可外推，哪些量（涌现能力、尾部安全、通信故障）不能只靠幂律？</div>

## 7. 推荐论文阅读

<DlPaperCard title="Scaling Laws for Neural Language Models" meta="Kaplan et al. · 2020" priority="必读 · 幂律观测" question="参数、数据、计算受限区间的 loss 拟合怎样建立，哪些条件固定？" reading="精读拟合对象、计算估算和 optimal allocation；记录外推假设。" href="https://arxiv.org/pdf/2001.08361" />

<DlPaperCard title="Training Compute-Optimal Large Language Models" meta="Hoffmann et al. · 2022" priority="必读 · Chinchilla" question="为何 Gopher 类模型被判断为欠训练，新的 N–D 配比如何得到？" reading="读 IsoFLOP profiles、实验拟合与下游评价；区分训练最优和推理最优。" href="https://arxiv.org/pdf/2203.15556" />

<DlPaperCard title="Deduplicating Training Data Makes Language Models Better" meta="Lee et al. · 2022" priority="选读 · 数据去重" question="近重复怎样影响记忆、训练效率和 benchmark 污染？" reading="精读 suffix-array/MinHash 管线、memorization 与 eval overlap 实验。" href="https://arxiv.org/pdf/2107.06499" />

## 8. 练习与门禁

建立 lineage；实现 exact/MinHash 候选；审计各域保留率；复算混合重复；拟合小规模幂律并画残差；估算 FLOPs/MFU；对 benchmark 做跨集合去重和时间切分。

## 本章小结

- 数据来源、许可、去重、质量、混合和删除链路共同定义模型训练分布。
- raw token 不等于独立有效信息；公开 benchmark 必须做污染审计。
- Scaling Law 是绑定配方和区间的经验模型，固定计算要联合选择参数与 token，并报告总生命周期成本。

<DlLessonProgress lesson-id="21" next-href="/courses/deep-learning/22-sft-lora-qlora" next-label="下一章：SFT、LoRA 与 QLoRA" />
