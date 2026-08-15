---
course: deep-learning
edition: industrial
title: RAG、向量检索与知识边界
description: 从 sparse/dense/hybrid retrieval、chunk 与 rerank 进入证据装配、引用、冲突、拒答、索引更新和端到端 RAG 评价。
outline: [2, 3]
---

<DlLessonHero number="25" module="大语言模型" title="RAG、向量检索与知识边界" question="检索命中怎样变成可引用答案；当证据缺失、过期或冲突时，系统如何拒答而不是让参数记忆补空？" duration="建议 330–430 分钟" :outcomes="['区分 BM25、dense、hybrid 与 late-interaction 检索接口', '设计 chunk、metadata、embedding、索引和删除版本合同', '比较 bi-encoder 召回与 cross-encoder rerank', '建立证据冲突、引用 span、不可回答和更新门禁', '分解 retrieval、grounding、answer 与系统成本指标']" />

RAG 把一部分记忆从参数移到可更新索引，但并未自动消除幻觉。检索相关不等于足以回答，生成引用不等于引用支持结论；端到端系统必须能定位失败在 query、召回、重排、装配还是生成。

## 贯穿案例：设备手册问答

索引包含多个型号和修订版手册。用户问“X3 变频器 E17 如何复位？”系统必须过滤型号/版本、找到复位步骤和安全前置，引用页/段；若只命中 X2 或旧版，应提示冲突或拒答。

<DlQuiz kind="先判断" question="top-1 chunk 与问题语义相似，是否证明答案有证据？" :options="['证明', '不证明；相似 chunk 可能缺少答案、版本错误或与其他证据冲突', '向量检索永不出错', '只需相似度大于 0']" :answer="1" explanation="relevance、answerability 与 authority 是不同标签。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>给 query、document、chunk、embedding、index、reranker、prompt、citation 分别写版本 ID；一次错误答案怎样回溯？</div>

## 1. Sparse、dense 与 hybrid retrieval

BM25 利用词频/逆文档频率，精确型号和罕见术语强；dense bi-encoder 把 query/document 映射到向量，语义改写召回强但可能忽略数字；hybrid 用加权或 Reciprocal Rank Fusion 合并。ColBERT 等 late interaction 保留 token 级表示，在成本与细粒度匹配间折中。

```python
def reciprocal_rank_fusion(rankings, k=60):
    score = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            score[doc] = score.get(doc, 0) + 1/(k+rank)
    return sorted(score, key=score.get, reverse=True)

fused = reciprocal_rank_fusion([["X3", "X2"], ["X2", "X3"]])
assert set(fused[:2]) == {"X2", "X3"}
```

<DlQuiz kind="混合检索" question="设备型号 `X3-E17` 查询为何常保留 sparse 通路？" :options="['稀有精确字符串可能被 dense 表示平滑掉', 'BM25 会生成答案', '向量不能存储', 'sparse 不需索引']" :answer="0" explanation="词法与语义通路互补。" />

## 2. Chunk、overlap、metadata 与索引生命周期

chunk 太小丢上下文，太大稀释相关性并占满 LLM context；overlap 可保护跨边界 span，也制造重复召回。结构感知切块应保留标题、表格、页码和层级。metadata filter 不是 prompt 建议，而应在检索层强制型号、租户、权限和时间。

<RetrievalPipelineLab />

```python
def chunks(tokens, size=6, overlap=2):
    step = size-overlap
    return [tokens[i:i+size] for i in range(0, len(tokens), step)]

parts = chunks(list(range(14)))
assert parts[0][-2:] == parts[1][:2]
assert all(len(x) <= 6 for x in parts)
```

<DlQuiz kind="切块" question="overlap 增大最直接的代价是什么？" :options="['索引/上下文重复、存储和召回冗余上升', '一定漏掉跨边界答案', '文档数量归零', '无需去重']" :answer="0" explanation="重叠保护边界但增加冗余。" />

索引制品要记录 embedding model、维度、归一化、distance、chunker、corpus snapshot 和 tombstone。更新文档需删除旧 chunk，防止不同版本并存；权限变化必须同步索引，不能只靠生成提示阻止泄漏。

## 3. 召回、重排与上下文装配

bi-encoder 可高吞吐召回，cross-encoder 让 query 与候选联合编码，精度高但成本随候选数增长。装配阶段要去近重复、保留来源、按多样性/权威/时效排序，并避免“lost in the middle”。top-k 不是越大越好：噪声和冲突会削弱 grounding。

<DlQuiz kind="重排" question="为什么 cross-encoder 通常用于 rerank 而非全库暴力检索？" :options="['每个 query–doc 需联合前向，成本太高', '它没有 tokenizer', '只能输出文本', '不支持相关性']" :answer="0" explanation="bi-encoder 可预计算文档向量。" />

## 4. 参数记忆、证据冲突与拒答

<EvidenceConflictLab />

Prompt 应明确只依据给定证据、逐项引用并在不足时拒答，但提示不是强保证。可以先做 answerability/entailment，生成后检查每个 claim 的 citation span。参数知识与检索冲突时，应优先可信最新来源并显式报告，而非无声融合。

```python
claims = {"断电后等待5分钟": {"doc":"X3-v4", "span":"p12#L4-L6"},
          "短接端子": None}
unsupported = [c for c,e in claims.items() if e is None]
assert unsupported == ["短接端子"]
```

<DlQuiz kind="引用" question="答案附了一个 URL，是否等于 grounded？" :options="['等于', '不等于，引用必须指向支持具体 claim 的可核对 span', '只有 PDF 才需要核对', 'URL 越长越可靠']" :answer="1" explanation="citation correctness 与 citation presence 不同。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>构造可回答、不可回答、旧版冲突、恶意文档注入四类 query；分别定义检索、回答、引用和拒答期望。</div>

## 5. RAG 评价分层

检索层：Recall@k、MRR/nDCG、answer-span recall、过滤正确性；重排层：pairwise/NDCG 与 latency；生成层：claim correctness、citation precision/recall、faithfulness、拒答；系统层：端到端成功、TTFT/P95、索引新鲜度、成本、权限泄漏。只报答案 BLEU/ROUGE 或“看起来有引用”不够。

<DlQuiz kind="分层诊断" question="gold evidence 未进入 top-k，但回答碰巧正确，RAG 系统应如何记账？" :options="['检索失败，答案可能来自参数记忆；两层分别记录', '检索成功', '删除 gold', '只记总分正确']" :answer="0" explanation="分层指标才能定位改进。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>在 top-k=4/8/20、rerank on/off、chunk=200/500 的网格上，先写预期，再报告 answer recall、citation correctness、P95 和 context token。</div>

## 6. 推荐论文阅读

<DlPaperCard title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" meta="Lewis et al. · 2020" priority="必读 · RAG" question="parametric 与 non-parametric memory 怎样联合训练/边缘化，检索文档如何影响生成？" reading="精读 RAG-Sequence/Token、DPR 接口和知识任务评价。" href="https://arxiv.org/pdf/2005.11401" />

<DlPaperCard title="Dense Passage Retrieval for Open-Domain Question Answering" meta="Karpukhin et al. · 2020" priority="必读 · Dense retrieval" question="in-batch negatives、bi-encoder 和 BM25 对照怎样构造？" reading="读训练数据、负例、索引与 top-k recall；检查 benchmark 域。" href="https://arxiv.org/pdf/2004.04906" />

<DlPaperCard title="ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction" meta="Khattab & Zaharia · 2020" priority="选读 · Late interaction" question="MaxSim token 交互怎样交换向量存储、检索效率与细粒度匹配？" reading="精读表示、索引和 latency/quality 对照。" href="https://arxiv.org/pdf/2004.12832" />

## 7. 练习与门禁

实现 RRF；切块消融；构造 hard negatives；版本化索引；测试权限/删除；逐 claim 引用；四类 answerability；报告检索—生成—系统三层指标。

## 本章小结

- RAG 是版本化的检索、重排、装配、生成和引用系统，不是一条“向量库”提示词。
- 相关性、可回答性、权威、时效和引用支持必须分别评价。
- 参数记忆与证据冲突时要显式仲裁、拒答和升级，不能用流畅答案掩盖检索失败。

<DlLessonProgress lesson-id="25" next-href="/courses/deep-learning/26-tools-agents" next-label="下一章：工具调用与 Agent" />
