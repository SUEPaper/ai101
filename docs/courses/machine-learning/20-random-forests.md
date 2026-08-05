---
title: 20. Bagging 与随机森林为什么稳定
description: 从 Bootstrap、相关误差与袋外样本推导随机森林
course: machine-learning
---

<MlLessonHero
  eyebrow="第五模块 · 非参数与集成"
  title="Bagging 与随机森林为什么稳定"
  subtitle="平均不能消灭共同偏差；随机森林的关键是让许多强而不完全相同的树一起投票。"
  :diagnostics="['能算 Bootstrap 的袋外比例', '能推导相关基学习器平均后的方差', '能识别特征重要性的常见陷阱']"
/>

## 1. Bootstrap 制造许多训练集

从 $n$ 个样本中有放回抽取 $n$ 次，得到一个 Bootstrap 样本。某个固定样本一次未被抽中的概率为

$$
\left(1-\frac1n\right)^n
\xrightarrow[n\to\infty]{}e^{-1}\approx0.368.
$$

因此每棵树平均看到约 $63.2\%$ 的不同训练样本，其余约 $36.8\%$ 是该树的袋外（OOB）样本。

Bagging 对每个 Bootstrap 样本训练一棵通常不剪枝的树。回归平均：

$$
\hat f_{\rm bag}(\mathbf x)
=\frac1B\sum_{b=1}^{B}\hat f_b(\mathbf x),
$$

分类则投票或平均概率。

::: ml-stop 停下来推一推 1
注意“抽取了 n 次”与“看到 n 个不同样本”不同。用 $n=3$ 枚举或模拟，检查不同样本数。
:::

<MlQuiz
  question="大样本极限下，一个训练样本未进入某次大小为 n 的 Bootstrap 样本的概率约是多少？"
  :options="['0', '0.368', '0.5', '0.632']"
  :answer="1"
  explanation="极限为 (1-1/n)^n→e^-1≈0.368；进入至少一次的概率约 0.632。"
/>

## 2. 平均为什么降方差

设每棵树的预测误差方差都是 $\sigma^2$，任意两棵误差相关系数为 $\rho$。平均预测误差

$$
\bar\varepsilon=\frac1B\sum_{b=1}^{B}\varepsilon_b.
$$

展开方差：

$$
\begin{aligned}
\operatorname{Var}(\bar\varepsilon)
&=\frac1{B^2}\left[
\sum_b\operatorname{Var}(\varepsilon_b)
+\sum_{b\ne b'}\operatorname{Cov}(\varepsilon_b,\varepsilon_{b'})
\right]\\
&=\frac1{B^2}\left[B\sigma^2+B(B-1)\rho\sigma^2\right]\\
&=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2.
\end{aligned}
$$

$B\to\infty$ 时，独立部分消失，但共同相关部分 $\rho\sigma^2$ 留下。只增加相同的树不够，还要降低树之间的相关性。

<MlLab
  kind="forest"
  title="实验：树数与相关性如何决定集成方差"
  description="调节树数和误差相关性，观察方差下降的极限。"
/>

<MlQuiz
  question="当基学习器误差完全相关，即 rho=1 时，增加树数会怎样影响上述方差？"
  :options="['降为 0', '保持 sigma²', '按 1/B 下降', '变为负数']"
  :answer="1"
  explanation="完全相同的误差无法靠平均抵消，方差公式退化为 sigma²。"
/>

## 3. 随机森林比 Bagging 多了一层随机性

若一个特征非常强，所有 Bootstrap 树都可能在根节点选择它，树仍高度相似。随机森林在每个节点只从随机抽取的 $m_{\rm try}$ 个特征中寻找最佳切分。

- $m_{\rm try}$ 大：单树更强，但树更相关；
- $m_{\rm try}$ 小：单树偏差可能上升，但相关性下降；
- 集成效果取决于“单树强度—树间相关性”的平衡。

分类常以 $\sqrt d$ 为起点，回归常以 $d/3$ 或框架默认值为起点，但都应验证，不能当成定律。

::: ml-stop 停下来推一推 2
如果数据里只有一个真正有用的特征，把 $m_{\rm try}$ 设得极小会发生什么？说明去相关也有代价。
:::

<MlQuiz
  question="随机森林在节点处随机抽取候选特征的主要目标是什么？"
  :options="['让每棵树完全相同', '降低树间相关性', '消除所有偏差', '让特征无需编码']"
  :answer="1"
  explanation="限制候选特征迫使树探索不同切分，从而提升平均的方差缩减效果。"
/>

## 4. OOB 评价

对训练样本 $i$，收集所有未在各自 Bootstrap 样本中包含 $i$ 的树：

$$
\mathcal B_i^{\rm OOB}
=\{b:i\notin\mathcal D_b^{\rm boot}\}.
$$

OOB 预测为

$$
\hat f_i^{\rm OOB}
=\frac1{|\mathcal B_i^{\rm OOB}|}
\sum_{b\in\mathcal B_i^{\rm OOB}}\hat f_b(\mathbf x_i).
$$

因为这些树训练时没见过样本 $i$，OOB 误差能近似独立验证误差，常可省下一次交叉验证。但所有上游预处理和调参若偷看完整数据，OOB 仍可能乐观。

<MlQuiz
  question="计算样本 i 的 OOB 预测时，应使用哪些树？"
  :options="['所有树', '只用训练 Bootstrap 中不含 i 的树', '只用第一棵树', '只用最深的树']"
  :answer="1"
  explanation="对 i 而言，只有没用它训练的树才能提供近似样本外预测。"
/>

## 5. 特征重要性并不天然可信

不纯度下降重要性把某特征参与切分带来的加权下降求和。它计算快，但偏爱：

- 连续或高基数特征，因为候选阈值更多；
- 可替代相关特征中的某一个，重要性被任意分摊；
- 训练分布里的预测关联，而非因果效应。

置换重要性在验证集上打乱某特征，观察性能下降：

$$
I_j
=\operatorname{score}(\mathcal D_{\rm val})
-\operatorname{score}(\operatorname{permute}_j(\mathcal D_{\rm val})).
$$

它更接近样本外预测依赖，但相关特征可互相补偿，使单个特征重要性被低估；分组置换有时更合适。

::: ml-stop 停下来推一推 3
复制一个强特征得到两个完全相同的列。置换其中一列时模型仍可用另一列，单列置换重要性会怎样变化？
:::

<MlQuiz
  question="把不纯度重要性直接解释为因果效应，问题在哪里？"
  :options="['树没有输出', '它衡量训练预测关联，受候选数和相关特征影响', '重要性总是负数', 'Bootstrap 禁止解释']"
  :answer="1"
  explanation="预测重要性既不等于干预效应，也可能因切分机会与特征相关性产生偏差。"
/>

## 6. 为什么通常让单树长得很深

Bagging 主要降低方差。若先强剪枝，每棵树偏差较高，平均不能消掉共同偏差。深树单独看方差高，但 Bootstrap 和特征随机化制造差异，再由平均稳定。

不过叶节点过小会使概率极端，内存与预测延迟也会增加。实际仍需调节最小叶样本、深度和树数。

<MlQuiz
  question="Bagging 最擅长改善哪类基学习器？"
  :options="['对样本扰动敏感的高方差模型', '完全固定不变的模型', '没有预测输出的模型', '只含常数的模型']"
  :answer="0"
  explanation="不同重采样产生不同高方差模型，平均才能明显稳定；稳定模型的收益较小。"
/>

## 7. OOB 投票的核心伪实现

~~~python
import numpy as np

def bootstrap_indices(n, rng):
    train = rng.integers(0, n, size=n)
    used = np.zeros(n, dtype=bool)
    used[train] = True
    oob = np.flatnonzero(~used)
    return train, oob
~~~

生产实现还要保存每棵树的 OOB 索引，按样本累计预测并记录有效投票数。树很少时，某些样本可能没有 OOB 预测。

<MlQuiz
  question="随机森林继续增加树数时，训练成本增加，但泛化误差通常会怎样？"
  :options="['必然无限恶化', '通常趋于稳定极限，而非因树多自动过拟合', '必然降到零', '与树相关性无关']"
  :answer="1"
  explanation="平均的蒙特卡洛波动随树数下降，最终受单树偏差和共同相关误差限制。"
/>

## 8. 动手练习：看见相关性

在含 2 个强特征和 18 个噪声特征的数据上：

1. 比较单树、Bagging、随机森林；
2. 改变 $m_{\rm try}$，记录单树准确率与树间预测相关性；
3. 画树数—OOB 误差曲线；
4. 复制一个强特征，比较不纯度与置换重要性；
5. 报告准确率、训练时间、预测延迟和模型大小。

<MlLessonProgress
  storage-key="20-random-forests"
  next-href="/courses/machine-learning/21-boosting"
  next-label="下一章：Boosting 为什么能把弱学习器变强"
/>
