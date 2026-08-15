---
course: deep-learning
edition: industrial
title: Word2Vec、GloVe 与词向量评估
description: 从分布假设进入 Skip-gram/CBOW、负采样、共现矩阵加权分解与词向量几何，建立内在、外在、偏置和稳定性评价。
outline: [2, 3]
---

<DlLessonHero number="17" module="NLP 基础" title="Word2Vec、GloVe 与词向量评估" question="一个词只有一个向量时，它从局部窗口和全局共现中保留了什么，又系统性丢失了哪些词义、频率与社会偏置？" duration="建议 280–370 分钟" :outcomes="['从分布假设构造 Skip-gram/CBOW 训练对', '推导 negative sampling 的正负点积梯度', '解释 GloVe 的加权 log-count 目标与矩阵观点', '用相似度、类比、下游迁移、偏置与种子稳定性评价词向量']" />

词向量把离散词 ID 映射到 $d$ 维连续空间。其核心假设是 distributional hypothesis：相似上下文中的词往往有相似含义。它不是“词义真值”，而是语料、窗口、采样与目标共同产生的统计表示。

## 贯穿案例：维修术语检索与同义词扩展

系统要把“轴承异响”“滚动体噪声”“bearing noise”连接起来，并避免把不同型号或否定表达混为一谈。我们使用工单语料训练静态 embedding，再通过人工术语集、检索召回和故障分类外在任务评价。

<DlQuiz kind="分布假设" question="两个词在相似上下文中频繁出现，词向量训练通常倾向于什么？" :options="['让它们表示完全相反方向', '让表示更相似，但是否真同义仍需任务证据', '删除两个词', '保证所有词义都相同']" :answer="1" explanation="上下文相似可来自同义、相关或可替换结构，并不保证严格同义。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>“高温”和“低温”出现在几乎相同句式中。分布表示为何可能很近？检索和故障决策应怎样处理反义关系？</div>

## 1. 从 one-hot 到可学习 lookup

embedding matrix $E\in\mathbb R^{|V|\times d}$ 的第 $i$ 行就是 token $i$ 的向量。one-hot 乘矩阵等价于查表。Word2Vec 通常保留 input 与 output 两套向量，训练后取其中一套或组合；报告必须说明。

## 2. Skip-gram 与 CBOW 的训练对

Skip-gram 用中心词预测窗口内上下文；CBOW 用上下文聚合预测中心词。窗口可固定或随机，距离可加权。动态窗口会让近邻更常被采样，隐式改变关系类型。

```python
def skipgram_pairs(tokens, window=2):
    pairs = []
    for center in range(len(tokens)):
        left, right = max(0, center-window), min(len(tokens), center+window+1)
        for context in range(left, right):
            if context != center:
                pairs.append((tokens[center], tokens[context]))
    return pairs

pairs = skipgram_pairs("轴承 温度 升高 触发 告警".split(), 1)
assert ("温度", "轴承") in pairs and ("温度", "升高") in pairs
```

<DlQuiz kind="窗口" question="增大上下文窗口通常更偏向捕获什么？" :options="['更局部的句法关系', '更宽主题/语义共现，同时引入更多噪声', '完全不改变训练对', '只改变词表编码']" :answer="1" explanation="小窗口常偏句法和局部替换，大窗口更偏主题关联，但这是统计趋势。" />

## 3. 全 softmax 与 negative sampling

全 softmax 的分母遍历整个词表。Negative sampling 把每个真实 pair 与 $K$ 个噪声词做二分类：

$$L=-\log\sigma(u_o^Tv_i)-\sum_{k=1}^K\log\sigma(-u_k^Tv_i).$$

正 pair 的点积被推高，负 pair 被推低。噪声常按 unigram$^{0.75}$ 采样，减少最高频词垄断并保留一定常见词概率。

<NegativeSamplingLab />

```python
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
positive, negatives = 1.2, np.array([-0.4, -1.0, 0.2])
loss = -np.log(sigmoid(positive)) - np.log(sigmoid(-negatives)).sum()
grad_positive = sigmoid(positive) - 1
grad_negatives = sigmoid(negatives)
assert loss > 0 and grad_positive < 0 and np.all(grad_negatives > 0)
```

<DlQuiz kind="负采样" question="负采样中的“负词”是语义反义词吗？" :options="['总是', '不是，通常从噪声分布抽取的非观测上下文词', '只在中文是', '只从同义词抽取']" :answer="1" explanation="negative 表示该训练窗口未观测为正 pair，不等于语言学反义。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>若漏标/稀疏语料中某个真实相关词常被当负样本，梯度会怎样？in-batch negatives 在哪些任务中尤其危险？</div>

## 4. 高频词 subsampling 与 PMI 视角

`the/的` 等高频词产生大量低信息 pair。Word2Vec 可按频率随机丢弃。Negative sampling 的最优点积与 shifted PMI 有联系，说明它隐式分解共现统计，但有限维、采样和正则会改变精确等式。

## 5. GloVe：显式拟合全局共现

GloVe 构造共现计数 $X_{ij}$，优化

$$J=\sum_{ij}f(X_{ij})(w_i^T\tilde w_j+b_i+\tilde b_j-\log X_{ij})^2.$$

加权函数压低极罕见噪声和极高频 pair 的主导。词向量常取 $w_i+\tilde w_i$。共现矩阵可以流式/稀疏构建，否则 $|V|^2$ 不可承受。

```python
import numpy as np

counts = np.array([1., 10., 100., 1000.])
xmax, alpha = 100., 0.75
weights = np.where(counts < xmax, (counts/xmax)**alpha, 1.0)
targets = np.log(counts)
assert np.all(np.diff(weights) >= 0)
assert np.isclose(weights[-1], 1.0) and np.all(np.isfinite(targets))
```

<DlQuiz kind="GloVe" question="GloVe 目标为何拟合 log X_ij 而不是原始计数？" :options="['为了产生负计数', '压缩巨大频率动态范围，并让共现比的加性关系更易由点积表达', '删除 bias', '只为画图']" :answer="1" explanation="对数把乘法共现比例转为加性差异。" />

## 6. 几何、相似度与类比

<EmbeddingGeometryLab />

余弦消除范数，点积保留方向与长度。频率常影响范数，因此检索选择会改变结果。类比 $v_{king}-v_{man}+v_{woman}$ 展示某些局部线性规律，但对词表、预处理和评价排除规则敏感。

多义词只有一个向量，会把“银行/河岸”等用法平均。子词 embedding 改善形态和 OOV，却不能完全解决上下文词义；第 18 章用 contextual representation 重写接口。

<DlQuiz kind="相似度" question="何时点积与余弦排名可能不同？" :options="['向量范数不同且范数影响点积', '维度相同就永不不同', '只在一维', '只在训练前']" :answer="0" explanation="点积等于范数乘积与余弦；范数差会重排。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>术语检索 top-10 在一个种子很好，另一个种子变化大。设计 Procrustes 对齐、邻域重叠和下游 bootstrap 稳定性检查。</div>

## 7. 评价与偏置

内在评价包括相似度与类比，但人工词对小且领域有限；外在评价把 embedding 放入冻结/微调任务。还应检查频率、语言、性别/职业等偏置、隐私记忆和 OOV 切片。静态 embedding 反映语料社会结构，去偏方法可能只隐藏一个线性方向，不能保证下游公平。

## 8. 推荐论文阅读

<DlPaperCard title="Efficient Estimation of Word Representations in Vector Space" meta="Mikolov et al. · 2013" priority="必读 · Word2Vec" question="CBOW 与 Skip-gram 怎样改变输入输出预测任务和计算成本？" reading="读架构、复杂度和语义/句法评价；再与后续 negative sampling 论文连接。" href="https://arxiv.org/pdf/1301.3781" />

<DlPaperCard title="Distributed Representations of Words and Phrases and their Compositionality" meta="Mikolov et al. · 2013" priority="必读 · 负采样" question="negative sampling、subsampling 与 phrase detection 分别改变什么数据和目标？" reading="精读负采样公式、噪声分布与 subsampling。" href="https://arxiv.org/pdf/1310.4546" />

<DlPaperCard title="GloVe: Global Vectors for Word Representation" meta="Pennington, Socher & Manning · 2014" priority="必读 · 全局共现" question="共现概率比为何比原始概率更能区分关系，目标函数怎样从该直觉产生？" reading="读第 2 节推导、加权函数和 word analogy 对照。" href="https://nlp.stanford.edu/pubs/glove.pdf" />

## 9. 练习与门禁

1. 生成 Skip-gram 动态窗口 pair；2. 有限差分检查负采样梯度；3. 稀疏构建共现矩阵；4. 比较 input/output/sum embedding；5. 按频率与 OOV 切片检索；6. 多种子邻域稳定性与偏置审计。

## 本章小结

- Word2Vec 从局部预测学习表示，GloVe 显式拟合加权全局共现。
- Negative sampling 的负例来自噪声分布，不等于反义；窗口与采样定义表示偏好。
- 词向量几何受范数、频率、语料和随机种子影响，内在指标不能替代下游与偏置评价。

<DlLessonProgress lesson-id="17" next-href="/courses/deep-learning/18-neural-language-models" next-label="下一章：神经语言模型与上下文表示" />
