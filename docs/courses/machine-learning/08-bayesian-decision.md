---
course: machine-learning
title: 分类首先是一个决策问题
description: 从后验概率和代价矩阵推导 Bayes 最优分类器，区分概率估计、阈值决策、生成式与判别式路线。
outline: [2, 3]
---

<MlLessonHero number="08" title="分类首先是一个决策问题" question="模型给出 0.37 的患病概率后，真正应该采取什么行动？" duration="建议 180–230 分钟" :outcomes="['从条件风险推出 Bayes 最优动作与代价敏感阈值', '区分概率估计、类别预测和实际决策', '比较生成式与判别式建模的信息需求']" />

## 诊断

<MlQuiz question="分类器输出 0.8，最合理的解释是什么？" :options="['一定正确', '在模型与校准成立时，是条件概率估计', '损失等于 0.8', '必须预测负类']" :answer="1" explanation="分数只有经过概率模型或校准才有概率语义；它仍不是最终动作。" />
<MlQuiz question="0-1 损失下，二分类 Bayes 阈值是多少？" :options="['0', '0.5', '1', '由样本量直接决定']" :answer="1" explanation="两种错误同价时，选择后验概率更大的类别，二分类分界为 0.5。" />
<MlQuiz question="训练类别比例能否直接当作部署时先验？" :options="['总能', '不能，抽样和部署人群可能不同', '只有回归不能', '只要 n>100 就能']" :answer="1" explanation="病例对照抽样、过采样和时间变化都会让训练先验与部署先验不同。" />

## 1. 预测只是决策的中间量

给定输入 $x$，动作集合为 $\mathcal A$，真实类别为 $Y$。动作 $a$ 的条件风险是

$$
R(a|x)=\sum_{y\in\mathcal Y}L(y,a)P(Y=y|X=x).
$$

Bayes 决策规则逐个输入选择条件风险最小的动作：

$$
\boxed{a^*(x)\in\arg\min_{a\in\mathcal A}R(a|x)}.
$$

它被称为 Bayes 最优，是因为在已知真实后验与损失的理想条件下，没有其他规则能有更低期望风险。现实中后验未知，要用数据估计，所以“Bayes 最优”不是说某个训练模型自动最优。

## 2. 二分类代价阈值完整推导

令动作 1 的误报代价为 $C_{10}=L(0,1)$，动作 0 的漏报代价为 $C_{01}=L(1,0)$，正确动作代价为 0。记 $p(x)=P(Y=1|x)$：

$$
R(0|x)=C_{01}p(x),\qquad
R(1|x)=C_{10}[1-p(x)].
$$

选 1 当且仅当

$$
C_{10}(1-p)<C_{01}p
\iff
\boxed{p>\frac{C_{10}}{C_{01}+C_{10}}}.
$$

若拒绝、复检或转人工也是合法动作，就为它们分别写条件风险，而不是把复杂流程硬压成一个阈值。

<MlLab kind="decision" />

<div class="ml-stop"><strong>停下来推一推 1</strong>加入固定复检成本 $c$，复检后可得到真实标签。复检的条件风险是 $c$。写出它优于直接判 0 和判 1 的概率区间。</div>

<MlQuiz question="漏报代价是误报的 9 倍，最优阈值是多少？" :options="['0.9', '0.5', '0.1', '9']" :answer="2" explanation="令 C₀₁=9、C₁₀=1，阈值为 1/(9+1)=0.1。" />

## 3. 多分类 Bayes 规则

对 $K$ 类，后验向量 $p_k(x)=P(Y=k|x)$。0-1 损失下，若动作 $a$ 正确损失 0、错误损失 1：

$$
R(a|x)=\sum_{y\ne a}p_y(x)=1-p_a(x).
$$

最小化风险等价于最大化后验：

$$
a^*(x)=\arg\max_k p_k(x).
$$

一般代价矩阵 $C_{ya}$ 下则比较

$$
R(a|x)=\sum_y C_{ya}p_y(x).
$$

“取最大概率类别”只是 0-1 损失的特例。

## 4. 为什么直接用线性回归做分类有问题

用 $x^T\beta$ 拟合 0/1 标签会遇到：

- 输出不受 $[0,1]$ 约束，不能直接解释为概率。
- 高 leverage 点可把拟合线拉出概率范围。
- Bernoulli 条件方差是 $p(1-p)$，天然异方差。
- 平方损失的条件最优预测确实是 $\mathbb E[Y|x]=P(Y=1|x)$，但线性函数未必适合这个有界均值。

最后一点很重要：问题不在平方损失完全错误，而在对条件均值施加了不合适的无界线性形式。逻辑回归用 Sigmoid 把线性 log odds 映射到概率区间。

## 5. 生成式与判别式两条路线

Bayes 公式为

$$
P(Y=k|x)=\frac{p(x|Y=k)P(Y=k)}{\sum_jp(x|Y=j)P(Y=j)}.
$$

生成式分类器建模类条件分布 $p(x|y)$ 和先验 $P(y)$，再用 Bayes 规则得到后验。判别式方法直接建模 $P(y|x)$ 或决策边界。

| 路线 | 建模对象 | 可能优势 | 主要代价 |
|---|---|---|---|
| 生成式 | $p(x|y),P(y)$ | 可生成/处理缺失，结构假设正确时样本效率高 | 必须解释输入分布，错误假设会伤害后验 |
| 判别式 | $P(y|x)$ 或边界 | 聚焦预测，通常更灵活 | 不直接给出输入生成机制 |

23 章会从高斯类条件分布推导 LDA/QDA；09 章直接推导判别式逻辑回归。

<div class="ml-stop"><strong>停下来推一推 2</strong>证明两类后验 odds 等于 likelihood ratio 乘 prior odds：$P(1|x)/P(0|x)=[p(x|1)/p(x|0)][P(1)/P(0)]$。</div>

## 6. 先验漂移下修正 odds

若训练与部署的 $p(x|y)$ 不变，但类别先验从 $\pi_{train}$ 变为 $\pi_{deploy}$，可以修正后验 odds：

$$
\frac{p_{deploy}(1|x)}{p_{deploy}(0|x)}
=\frac{p_{train}(1|x)}{p_{train}(0|x)}
\frac{\pi_{deploy}/(1-\pi_{deploy})}{\pi_{train}/(1-\pi_{train})}.
$$

这依赖“类条件分布不变”的 label shift 假设。若输入机制也变了，简单先验修正不够。

## 7. NumPy：从概率到动作

```python
import numpy as np

def bayes_binary_action(probability, false_negative_cost, false_positive_cost):
    probability = np.asarray(probability, dtype=float)
    threshold = false_positive_cost / (
        false_negative_cost + false_positive_cost
    )
    return (probability > threshold).astype(int), threshold

probability = np.array([.05, .18, .41, .82])
action, threshold = bayes_binary_action(probability, 8, 2)
assert np.isclose(threshold, .2)
assert np.array_equal(action, [0, 0, 1, 1])
```

## 8. 练习：三动作分诊

设计“回家、留观、立即治疗”三个动作的代价矩阵。输入后验向量后计算每个动作条件风险，返回最小风险动作。再改变治疗副作用成本，观察动作区域怎样变化。

<details><summary>代码骨架</summary>

```python
def minimum_risk_action(posteriors, cost_matrix):
    posteriors = np.asarray(posteriors, dtype=float)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    risks = posteriors @ cost_matrix
    return np.argmin(risks, axis=1), risks
```

</details>

<div class="ml-stop"><strong>停下来推一推 3</strong>一个分类器 A 的准确率高于 B，但 B 的概率更校准。在哪些决策场景中 B 可能更有价值？考虑代价会变化或需要多种阈值。</div>

## 9. 常见误解

- 后验概率、阈值和行动是三层，不要混写。
- 准确率隐含两类错误同价且部署类别比例与评价集一致。
- 代价不是为了调出更高分数，而是实际决策定义的一部分。
- 生成式不等于一定生成图像；它指建模联合分布或类条件分布。
- Bayes 最优依赖真实后验与真实代价，估计误差仍需评价。

<MlQuiz question="多分类一般代价矩阵下，最佳动作是什么？" :options="['最大后验类别', '最小条件期望代价的动作', '标签编号最小的类', '训练样本最多的类']" :answer="1" explanation="最大后验只在 0-1 损失时等价；一般情况必须对代价加权求和。" />
<MlQuiz question="判别式模型直接学习什么？" :options="['只学习 p(x)', 'P(y|x) 或决策边界', '必须学习 p(x|y)', '只学习类别先验']" :answer="1" explanation="它避免完整建模输入分布，直接针对条件预测或边界。" />

## 10. 小结

分类的核心不是输出类别名，而是在后验不确定性下最小化行动代价。下一章将从 Bernoulli 分布推出一个可训练、可校准的后验模型。

<MlLessonProgress lesson-id="08" next-label="第 09 章：逻辑回归为什么是这个形式" next-href="/courses/machine-learning/09-logistic-regression" />
