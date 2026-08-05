---
course: machine-learning
title: 学习问题怎样被数学化
description: 从现实决策写出输入空间、输出空间、假设空间、损失、风险与经验风险，理解归纳偏置为何不可避免。
outline: [2, 3]
---

<MlLessonHero number="01" title="学习问题怎样被数学化" question="一个现实愿望，怎样变成算法能够优化、他人能够审查的问题？" duration="建议 180–220 分钟" :outcomes="['把任务写成输入空间、输出空间、假设空间、损失和数据分布', '从实际错误代价构造损失并区分风险与经验风险', '解释为什么有限样本学习一定需要归纳偏置']" />

## 先诊断：你在优化什么

在数据科学课中，你已经把数据分成训练集和测试集，也调用过 `fit`。本章不再讨论 API，而是检查 `fit` 背后究竟承诺了什么。

<MlQuiz question="房价预测的 MAE 很低，能否说明这个模型适合决定贷款额度？" :options="['能，预测误差低就足够', '不能，还要把贷款决策及其错误代价写进问题', '能，只要测试集足够大', '不能，因为回归模型不能做决策']" :answer="1" explanation="预测质量只回答数值接近程度；决策还要定义高估、低估分别造成什么后果。" />

<MlQuiz question="训练集固定后，经验风险 R̂(h) 是随机变量吗？" :options="['对这份固定数据和固定 h，它是确定数', '永远是随机变量', '只在分类中确定', '取决于优化器']" :answer="0" explanation="抽样前它随数据变化；给定数据集与模型后，有限和已经确定。区分条件是关键。" />

<MlQuiz question="把假设空间设成所有可能函数，会自动得到最好泛化吗？" :options="['会，因为真函数一定在其中', '不会，有限样本无法区分大量同样拟合训练集的函数', '会，只要用梯度下降', '只对回归会']" :answer="1" explanation="更大的空间提高表达能力，也让有限样本留下更多无法排除的解释；必须加入归纳偏置。" />

## 1. 从一次实际决策开始

假设医院要根据入院时的测量判断患者是否进入高风险观察区。模糊目标“预测病情”至少缺少五项：输入是什么、输出是什么、允许什么模型、哪类错误更严重、病例怎样产生。

一个监督学习问题可以写成六元组：

$$
(\mathcal X,\mathcal Y,\mathcal H,L,P_{X,Y},D).
$$

- 输入空间 $\mathcal X$：所有合法特征向量，例如年龄、体温和化验结果。
- 输出空间 $\mathcal Y$：要预测的结果，例如 $\{0,1\}$。
- 假设空间 $\mathcal H$：算法可以选择的函数集合，$h:\mathcal X\to\mathcal A$。
- 损失 $L(y,a)$：真实结果为 $y$、采取动作 $a$ 时付出的代价。
- 数据分布 $P_{X,Y}$：未来病例和结果的未知联合分布。
- 样本 $D=\{(x_i,y_i)\}_{i=1}^n$：从未知分布取得的有限证据。

这里故意把动作空间 $\mathcal A$ 与标签空间 $\mathcal Y$ 区分开。概率预测输出可以是 $[0,1]$，最终动作却可能只有“普通观察”和“高风险观察”。

<div class="ml-stop"><strong>停下来推一推 1</strong>为“预测明天是否会缺货”写出六元组。尤其要写清：把缺货预测成不缺货，与把不缺货预测成缺货，代价为什么不相同。</div>

## 2. 损失来自决策，不来自算法名字

二分类的 0–1 损失是

$$
L(y,a)=\mathbf 1[y\ne a].
$$

它把两种错误视为同价。如果漏诊代价为 $C_{01}$，误报代价为 $C_{10}$，更诚实的损失矩阵是：

| 真实结果 | 动作 0 | 动作 1 |
|---|---:|---:|
| $Y=0$ | $0$ | $C_{10}$ |
| $Y=1$ | $C_{01}$ | $0$ |

给定输入 $x$，采取动作 0 的条件风险为

$$
R(0\mid x)=C_{01}P(Y=1\mid x),
$$

动作 1 的条件风险为

$$
R(1\mid x)=C_{10}P(Y=0\mid x)=C_{10}[1-P(Y=1\mid x)].
$$

选择动作 1 当且仅当第二个风险更小：

$$
\begin{aligned}
C_{10}(1-p)&<C_{01}p\\
C_{10}&<(C_{01}+C_{10})p\\
p&>\frac{C_{10}}{C_{01}+C_{10}}.
\end{aligned}
$$

阈值不是天生的 $0.5$。只有 $C_{01}=C_{10}$ 时才得到 $0.5$。

<MlLab kind="risk" />

<MlQuiz question="漏报代价 C₀₁ 增大，最优正类阈值怎样变化？" :options="['升高', '降低', '不变', '一定变成 0']" :answer="1" explanation="阈值为 C₁₀/(C₀₁+C₁₀)。分母增加而分子不变，所以阈值降低，模型更愿意判为正类。" />

## 3. 真实风险是我们想要、却不能直接计算的量

模型 $h$ 在未来分布上的真实风险定义为

$$
R(h)=\mathbb E_{(X,Y)\sim P_{X,Y}}[L(Y,h(X))]
=\int L(y,h(x))\,dP(x,y).
$$

期望表示：若不断从同一生成过程遇到新样本，平均损失趋向多少。问题在于 $P_{X,Y}$ 未知，否则可以直接积分，不需要从数据学习。

训练数据让我们计算经验风险：

$$
\widehat R_n(h)=\frac1n\sum_{i=1}^{n}L(y_i,h(x_i)).
$$

经验风险最小化（ERM）选择

$$
\widehat h\in\arg\min_{h\in\mathcal H}\widehat R_n(h).
$$

这里的 $\arg\min$ 返回“使目标最小的参数或函数”，不是最小值本身。若有多个同样最优的模型，它返回一个集合，因此写 $\in$ 比写等号更严谨。

### 经验风险为什么有希望

对固定模型 $h$，若样本独立同分布且损失期望存在，大数定律给出

$$
\widehat R_n(h)\xrightarrow[]{P}R(h).
$$

但训练会在看过数据后选择 $\widehat h$。对每个固定 $h$ 都收敛，不自动推出被数据挑中的 $\widehat h$ 也可靠；还需要控制整个假设空间的一致偏差。第 28 章会用 Uniform Convergence、PAC 与 VC 维解释这一步。

<div class="ml-stop"><strong>停下来推一推 2</strong>解释为什么“在测试集上尝试 1000 个模型，再报告最好的一个”会让测试误差不再是诚实的未来风险估计。被选择的模型已经依赖这份测试数据。</div>

## 4. 归纳偏置不是缺陷，而是学习成立的条件

只有三个训练点时，无穷多个函数都能穿过它们。要对新点做预测，算法必须偏好某些解释：线性模型偏好线性关系，决策树偏好轴对齐的分段常数，kNN 偏好邻近输入有相似输出，神经网络的架构与优化也带来偏好。

归纳偏置可以进入四个位置：

1. 假设空间：只允许 $h_\theta(x)=\theta^\top x$。
2. 目标函数：加入 $\lambda\lVert\theta\rVert_2^2$。
3. 优化算法：梯度下降从特定初值出发，可能偏好某类解。
4. 数据与表示：选择哪些特征、怎样采样，已经规定算法能看见什么。

没有“完全无偏”的学习器。真正的问题是偏置是否与任务结构相符，以及它是否被清楚记录。

## 5. 一个完整例子：明日峰值用电

令 $x\in\mathbb R^5$ 包含温度、湿度、工作日、前一日峰值和截距，$y\in\mathbb R$ 是明日峰值。先选线性假设空间与平方损失：

$$
\mathcal H=\{x\mapsto \beta^\top x:\beta\in\mathbb R^5\},\qquad
L(y,\widehat y)=(y-\widehat y)^2.
$$

经验风险是

$$
\widehat R_n(\beta)=\frac1n\sum_{i=1}^n(y_i-\beta^\top x_i)^2
=\frac1n\lVert y-X\beta\rVert_2^2.
$$

形状检查：$X$ 是 <span class="ml-shape">n × 5</span>，$\beta$ 是 <span class="ml-shape">5 × 1</span>，$X\beta$ 与 $y$ 都是 <span class="ml-shape">n × 1</span>。

```python
import numpy as np

X = np.array([
    [1, 22, 48, 1, 46],
    [1, 25, 55, 1, 49],
    [1, 28, 61, 1, 53],
], dtype=float)
y = np.array([47, 51, 56], dtype=float)
beta = np.array([4.0, .7, .1, 1.5, .5])

prediction = X @ beta
losses = (y - prediction) ** 2
empirical_risk = losses.mean()

assert prediction.shape == y.shape
print(prediction, empirical_risk)
```

平方损失是否合适仍需回到行动：如果低估峰值会造成停电，而高估只增加少量备用成本，就应考虑非对称损失。本章的形式化不是文书工作，而是防止算法优化错目标的第一道检查。

## 6. 解是否存在、是否唯一

ERM 的写法不保证最优解存在。若假设空间不闭、目标不连续，可能只有越来越小的值而没有达到最小值的点。即使存在，也可能不唯一。

- 存在性常由紧致参数域与连续目标保证。
- 唯一性常由严格凸性保证，但非凸模型不具备这个便利。
- 可辨识性问的是不同参数是否代表同一数据分布或同一函数。

第 04 章会看到列不满秩时 OLS 预测可以唯一，而系数不唯一；第 10 章会看到 Softmax 给所有类别参数加同一向量时概率不变。

<div class="ml-stop"><strong>停下来推一推 3</strong>若两个参数 $\theta_1\ne\theta_2$ 对所有输入都给出同一预测，它们在函数层面是否不同？参数估计是否可辨识？分别回答。</div>

## 7. 代码练习：把任务写成可计算合同

实现一个函数，接收预测、真实值和漏报/误报代价，返回每个样本的损失与平均经验风险。先自己写，再展开答案。

<details><summary>思路与公式骨架</summary>

对每个样本判断四种情况。只有 `y=1, prediction=0` 付出 $C_{01}$，只有 `y=0, prediction=1` 付出 $C_{10}$。

</details>

<details><summary>代码骨架</summary>

```python
def asymmetric_risk(y, prediction, false_negative_cost, false_positive_cost):
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    losses = np.zeros_like(y, dtype=float)
    # TODO: 写入两种错误的损失
    return losses, losses.mean()
```

</details>

<details><summary>参考答案与边界检查</summary>

```python
def asymmetric_risk(y, prediction, false_negative_cost, false_positive_cost):
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    if y.shape != prediction.shape:
        raise ValueError("y and prediction must have the same shape")
    if not np.isin(y, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("binary labels must be 0 or 1")

    losses = np.zeros_like(y, dtype=float)
    losses[(y == 1) & (prediction == 0)] = false_negative_cost
    losses[(y == 0) & (prediction == 1)] = false_positive_cost
    return losses, losses.mean()

losses, risk = asymmetric_risk([1, 0, 1, 0], [0, 1, 1, 0], 8, 2)
assert np.allclose(losses, [8, 2, 0, 0])
assert risk == 2.5
```

</details>

## 8. 常见误解与调试清单

- “标签就是动作”：概率估计、排序和资源分配的动作空间可能完全不同。
- “损失越数学越客观”：损失编码价值选择，必须说明错误代价来自哪里。
- “训练集平均代表未来”：只有数据生成机制稳定、抽样合理且没有选择偏差时才有根据。
- “假设空间越大越好”：表达能力与可学习性、计算成本同时变化。
- “优化成功就是学习成功”：优化只说明找到低经验风险，泛化是另一个问题。

<MlQuiz question="argminₕ R̂(h) 返回什么？" :options="['最小风险数值', '使风险达到最小的模型集合', '损失函数本身', '测试集']" :answer="1" explanation="min 返回最小值，argmin 返回达到最小值的自变量；有并列解时它是集合。" />

<MlQuiz question="对每个固定 h 都有 R̂(h) 接近 R(h)，为何仍不足以保证 ERM 泛化？" :options="['因为 ĥ 是看过数据后选择的，需要同时控制整个假设空间', '因为经验风险不能计算', '因为期望不存在', '因为训练集总是有标签错误']" :answer="0" explanation="数据依赖的选择会寻找偶然偏差最大的模型之一，因此需要一致收敛或复杂度控制。" />

## 9. 本章小结与术语

- 学习问题不是算法名称，而是输入、输出、假设、损失、分布与样本的组合。
- 真实风险是目标，经验风险是有限数据给出的可计算替代。
- ERM 需要归纳偏置与泛化控制；低训练误差不是终点。

<dl class="ml-terms"><div><dt>risk</dt><dd>未来数据分布上损失的期望。</dd></div><div><dt>ERM</dt><dd>在假设空间中最小化经验风险。</dd></div><div><dt>inductive bias</dt><dd>算法从有限证据外推时使用的结构偏好。</dd></div></dl>

## 参考资料

- CMU 10-301/10-601：学习问题、函数逼近与经验风险的知识结构。
- *Probabilistic Machine Learning: An Introduction*，Ch.1、Ch.5：概率建模与决策论主题。
- 本页医院与用电案例、推导、代码、问题和交互说明均为本站独立创作。

<MlLessonProgress lesson-id="01" next-label="第 02 章：参数为什么能被算法学出来" next-href="/courses/machine-learning/02-optimization" />
