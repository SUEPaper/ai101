---
course: machine-learning
title: 训练误差为什么不等于未来误差
description: 完整推导平方误差的偏差—方差分解，理解数据切分、交叉验证、学习曲线、超参数选择与泄漏。
outline: [2, 3]
---

<MlLessonHero number="03" title="训练误差为什么不等于未来误差" question="一个模型已经记住所有训练样本，为什么可能仍然不会预测明天？" duration="建议 220–280 分钟" :outcomes="['区分训练、验证、测试各自回答的问题', '逐步推导平方误差的偏差—方差—噪声分解', '用学习曲线和交叉验证诊断欠拟合、过拟合与泄漏']" />

## 先诊断：数据被你用了几次

<MlQuiz question="反复查看测试分数并据此调参，测试集还在估计什么？" :options="['完全独立的未来误差', '被选择流程间接优化后的误差，通常偏乐观', '训练误差', 'Bayes 误差']" :answer="1" explanation="一旦选择依赖测试反馈，测试集就参与了训练流程，不再提供一次性独立评价。" />

<MlQuiz question="训练误差很高、验证误差也很高且二者接近，首先怀疑什么？" :options="['高方差', '高偏差或优化不足', '测试泄漏', '模型一定完美']" :answer="1" explanation="两者都高且接近说明模型连训练规律都没拟合好；要区分表达不足与优化未收敛。" />

<MlQuiz question="增加训练样本最直接缓解哪类问题？" :options="['不可约噪声', '估计方差', '错误损失定义', '标签含义不清']" :answer="1" explanation="更多独立样本通常让拟合结果对抽样更稳定；它不能消除数据生成过程本身的噪声。" />

## 1. 三份数据回答三个不同问题

- 训练集：估计模型参数，最小化经验目标。
- 验证集：选择超参数、特征处理、阈值和停止时刻。
- 测试集：在完整流程冻结后，只做一次最终审计。

若数据有时间、用户或群组结构，随机逐行切分可能破坏独立性。预测未来应按时间向前切；同一患者多次记录应放在同一分区；预处理参数只能从训练折估计。

<div class="ml-warning"><strong>流水线之外的计算也会泄漏</strong>先用全体数据填补缺失、标准化、筛特征或做 PCA，再交叉验证，验证折的信息已经进入训练表示。正确做法是让每个折内部重新拟合全部数据依赖步骤。</div>

## 2. 泛化间隙从哪里来

真实风险与经验风险之差是

$$
R(h)-\widehat R_n(h).
$$

对训练前固定的 $h$，它是抽样波动；对通过数据挑选的 $\widehat h$，还包含选择效应。模型空间越大、尝试次数越多，越可能找到一个恰好迎合训练噪声的候选。

训练误差通常对真实风险偏乐观，因为同一数据既决定参数又评价参数。验证集把评价与参数拟合分开，但若反复根据验证集尝试方案，验证集也会逐渐被“学会”。

## 3. 偏差—方差分解：完整推导

设数据由

$$
Y=f(x)+\varepsilon,
\qquad
\mathbb E[\varepsilon\mid x]=0,
\qquad
\operatorname{Var}(\varepsilon\mid x)=\sigma^2
$$

生成。训练集 $D$ 是随机的，所以拟合器 $\widehat f_D(x)$ 也随抽样变化。对固定 $x$，考虑新标签 $Y$ 的期望平方误差：

$$
\mathbb E_{D,Y}\left[(Y-\widehat f_D(x))^2\right].
$$

记所有训练集上预测的平均为

$$
\bar f(x)=\mathbb E_D[\widehat f_D(x)].
$$

在误差中加减 $f(x)$ 与 $\bar f(x)$：

$$
Y-\widehat f_D
=\underbrace{Y-f}_{\varepsilon}
+\underbrace{f-\bar f}_{\text{bias}}
+\underbrace{\bar f-\widehat f_D}_{\text{sampling variation}}.
$$

平方并取期望。先写出全部项：

$$
\begin{aligned}
\mathbb E[(Y-\widehat f_D)^2]
=&\ \mathbb E[(Y-f)^2]+(f-\bar f)^2
+\mathbb E[(\bar f-\widehat f_D)^2]\\
&+2\mathbb E[(Y-f)(f-\bar f)]\\
&+2\mathbb E[(Y-f)(\bar f-\widehat f_D)]\\
&+2\mathbb E[(f-\bar f)(\bar f-\widehat f_D)].
\end{aligned}
$$

三个交叉项为零：

1. $f-\bar f$ 对固定 $x$ 是常数，而 $\mathbb E[Y-f\mid x]=0$。
2. 新测试噪声与训练集独立，且条件均值为零。
3. $\mathbb E_D[\bar f-\widehat f_D]=\bar f-\bar f=0$。

因此

$$
\boxed{
\mathbb E[(Y-\widehat f_D(x))^2]
=\sigma^2
+\left(f(x)-\mathbb E_D\widehat f_D(x)\right)^2
+\mathbb E_D\left[\left(\widehat f_D(x)-\mathbb E_D\widehat f_D(x)\right)^2\right]
}
$$

即

$$
\text{Expected test error}=\text{noise}+\text{bias}^2+\text{variance}.
$$

<div class="ml-stop"><strong>停下来推一推 1</strong>为什么第二个交叉项需要“新测试噪声与训练集独立”？如果同一条样本同时参与训练和评价，这一步是否还成立？</div>

<MlLab kind="generalization" />

<MlQuiz question="偏差—方差分解中的 variance 对什么取方差？" :options="['固定模型内的标签噪声', '训练集重新抽样导致预测 f̂_D(x) 的变化', '输入 x 的单位', '损失函数的数值范围']" :answer="1" explanation="这里的随机性来自训练集 D；同一 x 在不同训练样本上拟合出的预测会变化。" />

## 4. 复杂度怎样改变三项

更灵活的模型通常能让 $\bar f(x)$ 更接近真实 $f(x)$，降低偏差；但它也更容易随训练样本波动，提高方差。不可约噪声 $\sigma^2$ 来自相同输入下结果仍随机，不能通过模型复杂度消除。

“偏差低、方差高”不是某个算法永恒属性。增加正则、剪枝、增大 kNN 的 $k$、限制网络宽度，都可能让同一算法沿偏差—方差轴移动。

## 5. 交叉验证估计的是什么

K 折交叉验证把数据分成 $K$ 个互斥折。第 $k$ 次用其余折训练，折 $k$ 评价：

$$
\widehat R_{CV}=\frac1K\sum_{k=1}^K
\frac1{|I_k|}\sum_{i\in I_k}L(y_i,\widehat h^{(-k)}(x_i)).
$$

这里每个样本只在没有参与该轮参数拟合时被评价。CV 估计的是“用约 $(K-1)n/K$ 个样本训练该流程”的表现，并非完整训练集模型的完全无偏风险。

- $K$ 小：每轮训练数据少，计算便宜，估计可能偏悲观。
- $K$ 大：每轮更接近全数据训练，但各折估计高度相关，计算增加。
- 分类不平衡可用分层折；群组数据用 Group K-fold；时间数据用前向扩展窗口。

### 嵌套交叉验证

外层只估计最终表现，内层选择超参数。对每个外层训练区，内层重新选择超参数，再在外层验证区评价。这样避免“用同一 CV 分数既选方案又报告方案”的选择偏差。

<div class="ml-stop"><strong>停下来推一推 2</strong>5 折 CV 中，标准化器应拟合几次？至少每个训练折一次。若还在内层搜索超参数，外层的每个候选流程都必须在对应内折重新拟合。</div>

## 6. 学习曲线把诊断变成可验证假设

学习曲线横轴是训练样本量，纵轴画训练与验证误差：

- 两条都高且接近：高偏差、特征不足，或优化未完成。
- 训练低、验证高且差距大：高方差、泄漏或分布不一致。
- 样本增加时验证继续改善：收集更多相似数据可能有效。
- 验证误差早早平台：需要改变表示、假设或数据质量，不只是加样本。

曲线必须在多个随机划分上给出分布或误差条，只画一个种子的折线可能把抽样噪声误当趋势。

## 7. 数值验证偏差—方差

下面固定真实函数，重复抽训练集，估计同一个 $x_0$ 处的预测分布：

```python
import numpy as np

rng = np.random.default_rng(42)
x0 = .35
true_value = np.sin(2 * np.pi * x0)

def fit_and_predict(degree, sample_size=25, noise=.25):
    x = rng.uniform(0, 1, sample_size)
    y = np.sin(2 * np.pi * x) + rng.normal(0, noise, sample_size)
    coefficients = np.polyfit(x, y, degree)
    return np.polyval(coefficients, x0)

def bias_variance_at_x(degree, repeats=500):
    predictions = np.array([fit_and_predict(degree) for _ in range(repeats)])
    mean_prediction = predictions.mean()
    bias_squared = (mean_prediction - true_value) ** 2
    variance = predictions.var(ddof=0)
    return bias_squared, variance

for degree in [1, 3, 9]:
    print(degree, bias_variance_at_x(degree))
```

这个实验只估计固定 $x_0$ 处的偏差和方差。完整风险还要对输入分布 $P_X$ 再取期望。

## 8. 练习：从零实现 K 折流程

不用 `cross_val_score`，实现索引切分。每个折内重新计算训练均值与标准差，再拟合多项式 Ridge。比较次数 1、3、9 的平均验证 MSE。

<details><summary>关键检查</summary>

每个样本恰好进入一次验证折；缩放参数来自对应训练折；所有次数使用相同折，减少比较噪声。

</details>

<details><summary>索引骨架</summary>

```python
def kfold_indices(n, folds, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return np.array_split(indices, folds)

folds = kfold_indices(23, 5)
assert sorted(np.concatenate(folds).tolist()) == list(range(23))
```

</details>

<details><summary>评价循环参考答案</summary>

```python
def cross_validate(model_factory, X, y, fold_indices):
    scores = []
    all_indices = np.arange(len(y))
    for validation_indices in fold_indices:
        train_mask = ~np.isin(all_indices, validation_indices)
        train_indices = all_indices[train_mask]
        model = model_factory()
        model.fit(X[train_indices], y[train_indices])
        prediction = model.predict(X[validation_indices])
        scores.append(np.mean((y[validation_indices] - prediction) ** 2))
    return np.asarray(scores)
```

把缩放放进 `model_factory()` 返回的流水线中，而不是在函数外对全体 `X` 处理。

</details>

<div class="ml-stop"><strong>停下来推一推 3</strong>若你比较 200 个超参数组合并选择 CV 最低者，最小 CV 误差为什么仍可能偏乐观？怎样用嵌套 CV 或独立测试集审计这一选择流程？</div>

## 9. 常见误解

- “验证集越大越好”：它与训练数据争夺有限样本，需要权衡估计精度和拟合质量。
- “K 折解决所有分布偏移”：CV 只重采样已有分布；未来机制改变仍会失效。
- “训练/验证接近就说明好”：也可能二者都很差。
- “固定随机种子消除不确定性”：它只固定一次抽样，不能描述抽样变化。
- “测试集不参与梯度就没有泄漏”：人工选择同样会把测试信息传回流程。

<MlQuiz question="时间序列预测最常见的安全验证方式是什么？" :options="['随机打乱逐行 K 折', '用未来训练、过去验证', '按时间前向切分，只用过去预测未来', '把全部数据标准化后切分']" :answer="2" explanation="时间顺序是数据生成机制的一部分；验证时不能让未来信息进入过去。" />

<MlQuiz question="不可约噪声 σ² 能通过无限复杂模型消除吗？" :options="['能', '不能，它是给定 x 后 Y 的随机性', '只要训练误差为 0 就能', '只对分类不能']" :answer="1" explanation="模型可以减少偏差，却不能预测在相同信息下仍然随机的那部分结果。" />

## 10. 本章小结

- 参数拟合、超参数选择和最终审计必须使用不同信息边界。
- 期望测试平方误差可分成不可约噪声、平方偏差和估计方差。
- 交叉验证与学习曲线是估计工具，不会自动修复泄漏、错误切分或分布偏移。

<MlLessonProgress lesson-id="03" next-label="第 04 章：最小二乘的几何本质" next-href="/courses/machine-learning/04-ols-geometry" />
