---
course: deep-learning
edition: industrial
title: 大模型评估、可靠性、安全与可解释性
description: 建立 benchmark、污染、配对置信区间、LLM judge、校准、幻觉、偏见、隐私、红队和解释边界的发布证据体系。
outline: [2, 3]
---

<DlLessonHero number="28" module="大语言模型" title="大模型评估、可靠性、安全与可解释性" question="怎样把一个漂亮总分拆成可复现、带不确定性、抗污染并覆盖真实风险的发布证据？" duration="建议 370–480 分钟" :outcomes="['编写任务、数据、解码、成本与统计单位明确的 eval contract', '用配对差值、bootstrap/随机化和多重比较报告不确定性', '校准并审计 LLM-as-a-judge 的顺序、风格和注入偏置', '设计幻觉、偏见、隐私、越狱、拒答与高风险切片', '说明 attribution/probe 等解释工具能回答和不能证明什么']" />

评估是模型系统的一部分。benchmark 分数若没有数据版本、污染审计、解码预算、配对原始结果和置信区间，不能支持发布。安全也不是一个“毒性分数”，而是威胁模型下的多层门禁。

## 贯穿案例：维修助手发布审查

候选 B 相比 A 总体准确率高 4 个百分点。审查必须回答：差值是否稳定，是否来自污染或更多 test-time token，罕见危险故障是否退步，引用是否支持结论，judge 是否偏好更长回答，是否泄露工单隐私，越狱能否调用高风险工具。

<DlQuiz kind="先判断" question="A=72%、B=76% 是否足以宣布 B 显著更好？" :options="['足够', '不足，需要同一独立样本上的配对差值、不确定性、切片与预算合同', '只需样本数大于 10', '百分比无需统计']" :answer="1" explanation="两个点估计没有差值分布。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>写 eval card：模型/adapter、prompt、decoder、工具、数据 cutoff、独立单位、指标、切片、成本、拒答与失败样本各填什么？</div>

## 1. Benchmark 合同与污染

明确任务输入输出、评分器、few-shot prompt、最大 token、采样次数、工具/检索和超时。benchmark 版本、题目来源、许可、语言和数据 cutoff 必须固定；公开题及讲解应与训练/后训练做 exact、near-duplicate、语义和时间污染审计。

HELM 强调 scenario、adaptation 与 metric 的多维组合。leaderboard 总分会掩盖群体/任务 trade-off；选择 benchmark 本身也是多重尝试，不能反复看测试集后仍称其冻结。

<DlQuiz kind="污染" question="只从最终 SFT 数据删除 benchmark 是否足够？" :options="['足够', '不足，预训练、合成 teacher、偏好、RAG 索引和 prompt 示例都可能泄漏', '只要模型不开源就够', '污染只影响数学题']" :answer="1" explanation="需审计完整数据与工具链。" />

## 2. 配对比较与不确定性

同一题/用户/文档上的 A/B 结果相关，应对差值做 paired bootstrap、随机化或适当模型；独立单位不是 token 行数。报告区间、效应量、原始 wins/ties/losses，并对大量模型/切片选择做多重比较与确认集。

<BootstrapEvaluationLab />

```python
import random

diffs = [1,0,-1,1,1,0,-1,1]
rng = random.Random(0)
means = []
for _ in range(2000):
    sample = [rng.choice(diffs) for _ in diffs]
    means.append(sum(sample)/len(sample))
lo, hi = sorted(means)[50], sorted(means)[1950]
assert lo <= sum(diffs)/len(diffs) <= hi
```

<DlQuiz kind="统计单位" question="1000 个 token 来自 10 个文档，bootstrap 应优先按什么重采样？" :options="['独立文档/任务单位', '每个 token 当独立样本', '每个字符', '模型参数']" :answer="0" explanation="token 内相关会虚假缩窄区间。" />

## 3. LLM-as-a-Judge 不是神谕

Judge 可扩大开放式评价，但会有位置、冗长、风格、自我偏好、模型名和提示注入偏差。应盲化来源、随机并双向换序、标准化长度/格式、隔离被评文本中的指令，用多 judge/人工子集校准，并报告一致率与无法判定。

<JudgeBiasLab />

```python
pairs = [("A","B","B"),("B","A","A")]  # 展示顺序交换，赢家语义应一致
semantic_winners = [winner for _,_,winner in pairs]
position_consistent = semantic_winners[0] == semantic_winners[1]
assert not position_consistent  # 这个玩具 judge 暴露顺序偏差
```

<DlQuiz kind="Judge" question="交换 A/B 后 judge 总选第二个，说明什么？" :options="['B 内容一定更好', '存在强位置偏差，需校准或拒用该判定', '模型更安全', '置信区间变窄']" :answer="1" explanation="换序是最低成本的偏置探针。" />

## 4. 可靠性：校准、幻觉、拒答与 OOD

准确率不说明置信可靠。可用 ECE/Brier、选择性风险—覆盖、温度校准和 conformal/集合方法，但生成文本的“置信”定义需明确（答案 token、verifier、self-consistency 或 judge）。幻觉要按无证据 claim、错误引用、矛盾和不可回答分类。

```python
pred = [.9,.8,.6,.4]; correct = [1,0,1,0]
bins = [(0,.5),(.5,.75),(.75,1.01)]
ece = 0
for lo,hi in bins:
    ids=[i for i,p in enumerate(pred) if lo<=p<hi]
    if ids:
        ece += len(ids)/len(pred)*abs(sum(pred[i] for i in ids)/len(ids)-sum(correct[i] for i in ids)/len(ids))
assert 0 <= ece <= 1
```

<DlQuiz kind="拒答" question="提高拒答率一定提升安全和有用性吗？" :options="['一定', '不一定，要看风险—覆盖：危险错误可能下降，但可回答请求也可能被过度拒绝', '拒答无成本', '只与温度有关']" :answer="1" explanation="应同时报告覆盖率与接受集合风险。" />

## 5. 安全、偏见、隐私与红队

威胁模型包括恶意用户、恶意检索文档、越权工具、训练数据提取、成员推断、PII 泄漏、越狱、提示注入、拒绝服务和供应链。红队要记录攻击预算、适应性、成功标准和修复回归集；模型层 refusal 之外还需权限、sandbox、速率、内容/输出验证和人工升级。

偏见评价按语言、地域、性别/职业、设备/客户群体和高风险后果切片，区分代表性、刻板印象与性能差异。隐私测试用 canary/PII/secret 提取，不把“模型说没有”当证明。安全 benchmark 也会被过拟合，应保留私有动态集。

<DlQuiz kind="红队" question="模型通过固定越狱题库是否证明安全？" :options="['证明', '不证明，只说明对该版本/攻击预算的已知样本表现', '证明不会泄密', '等同形式化验证']" :answer="1" explanation="攻击会适应，需纵深防御和持续监控。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>将“错误停机建议”画成 hazard tree：模型、检索、工具、权限、人机界面和监控各有哪些防线与共同失效？</div>

## 6. 可解释性边界

attention map、saliency、integrated gradients、probe、activation patching、causal tracing 和 sparse autoencoder 回答的问题不同。相关性热图不是因果解释；probe 可读出信息不等于主模型使用它；自然语言自我解释可能是事后合理化。解释结论必须有反事实/干预、随机基线和预测性验证。

可解释性适合生成假设、定位电路和支持审计，不能单独证明公平、安全或无后门。对高风险发布，行为测试和系统硬边界仍是主要证据。

<DlQuiz kind="解释" question="某 attention head 高权重指向引用句，能否证明答案因该句而生成？" :options="['能', '不能，仅相关；需遮蔽/patching 等干预验证因果作用', 'attention 永远无意义', '只看颜色即可']" :answer="1" explanation="权重可视化不是充分因果证据。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>选择一个解释假设，设计 activation patching/输入反事实；写出若结果不变时如何推翻假设。</div>

## 7. 推荐论文阅读

<DlPaperCard title="Holistic Evaluation of Language Models" meta="Liang et al. · 2022" priority="必读 · 多维评估" question="scenario、adaptation、metric 和 robustness/fairness/efficiency 怎样组成矩阵？" reading="读 taxonomy、标准化与透明度；用本章案例复写一个 scenario card。" href="https://arxiv.org/pdf/2211.09110" />

<DlPaperCard title="Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" meta="Zheng et al. · 2023" priority="必读 · Judge 与人类偏好" question="位置、冗长和自增强偏差如何测试，pairwise judge 与众包怎样对照？" reading="精读 bias、agreement 与 arena 统计；设计双向换序复验。" href="https://arxiv.org/pdf/2306.05685" />

<DlPaperCard title="TruthfulQA: Measuring How Models Mimic Human Falsehoods" meta="Lin, Hilton & Evans · 2022" priority="选读 · 真实性反例" question="对抗性问题怎样区分信息性与真实性，自动 judge 有什么局限？" reading="读数据构造、人工/自动指标和模型规模结果；检查公开题污染风险。" href="https://arxiv.org/pdf/2109.07958" />

## 8. 练习与发布门禁

写 eval card；做配对 bootstrap；审计污染；双向 judge；画风险—覆盖/ECE；建立 hazard tree；跑提示注入/隐私红队；用干预验证一个解释假设；提交失败样本而非只交总分。

## 本章小结

- 可发布评估必须绑定数据、解码、成本、独立单位、污染和置信区间。
- LLM judge 可扩展但有位置、风格和注入偏差，需要人工校准，不能独占真值。
- 可靠性、安全、偏见、隐私和解释是不同证据层；纵深系统防线优先于单一模型分数。

<DlLessonProgress lesson-id="28" />
