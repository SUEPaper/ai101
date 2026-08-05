---
course: machine-learning
title: 线性模型怎样表达非线性
description: 用基函数、特征映射、多项式与局部加权回归理解“对参数线性”，并推导平滑矩阵和有效自由度。
outline: [2, 3]
---

<MlLessonHero number="07" title="线性模型怎样表达非线性" question="一条关于参数的直线公式，怎样画出弯曲、周期甚至局部变化的函数？" duration="建议 220–280 分钟" :outcomes="['区分对输入线性与对参数线性', '构造多项式、RBF 和分段基函数', '用平滑矩阵迹解释正则化模型的有效自由度']" />

## 诊断

<MlQuiz question="f(x)=β₀+β₁x+β₂x² 是线性模型吗？" :options="['不是，因为曲线会弯', '对参数 β 是线性的，因此属于线性模型', '只有 β₂=0 才是', '取决于 y 是否连续']" :answer="1" explanation="线性模型中的“线性”指参数以线性组合出现；输入可先经过非线性特征映射。" />

<MlQuiz question="把多项式次数不断提高，训练误差通常怎样？" :options="['不增', '一定增大', '完全不变', '只在分类中下降']" :answer="0" explanation="更高次数包含低次数模型作为特例，假设空间嵌套，所以最小训练误差不能变差。" />

<MlQuiz question="基函数中心和宽度可以怎样得到？" :options="['只能由梯度下降学习', '可预先设定、由数据估计或与参数一起学习', '必须等于标签', '不能改变']" :answer="1" explanation="固定基函数保持线性求解；若中心/宽度也学习，整体优化通常变成非线性。" />

## 1. 线性组合的是参数，不一定是原始输入

先定义特征映射

$$
\phi(x)=[\phi_0(x),\phi_1(x),\ldots,\phi_{m-1}(x)]^T.
$$

模型为

$$
f_\beta(x)=\beta^T\phi(x)=\sum_{j=0}^{m-1}\beta_j\phi_j(x).
$$

只要 $\phi$ 固定，$f$ 对 $\beta$ 线性，仍可使用 OLS/Ridge 的全部推导。非线性被放进表示，而不是求解器。

常见基函数：

- 多项式：$1,x,x^2,\ldots,x^d$。
- Fourier：$1,\sin(\omega x),\cos(\omega x),\ldots$。
- RBF：$\exp[-(x-c_j)^2/(2s_j^2)]$。
- 分段线性 hinge：$(x-c_j)_+=\max(x-c_j,0)$。

## 2. 多项式回归仍是一次矩阵求解

一维输入的次数 $d$ 设计矩阵是 Vandermonde 结构：

$$
\Phi=
\begin{bmatrix}
1&x_1&x_1^2&\cdots&x_1^d\\
1&x_2&x_2^2&\cdots&x_2^d\\
\vdots&\vdots&\vdots&&\vdots\\
1&x_n&x_n^2&\cdots&x_n^d
\end{bmatrix}.
$$

目标仍是

$$
\min_\beta\lVert y-\Phi\beta\rVert^2+\lambda\lVert\beta\rVert^2.
$$

高次幂会造成严重尺度差异与共线，尤其当 $x$ 离 0 很远。应先把 $x$ 中心化/缩放，或使用更稳定的正交多项式基。

<MlLab kind="basis" />

<div class="ml-stop"><strong>停下来推一推 1</strong>为什么次数 9 的假设空间包含次数 3？把 $x^4$ 到 $x^9$ 的系数设为 0 即可。因此训练误差不增，但验证误差可能上升。</div>

<MlQuiz question="高次多项式在训练区间外为何常失控？" :options="['高次项随 |x| 快速增长，外推主要由边界和最高次项决定', '因为 OLS 不能预测', '因为标签必须为 0', '因为正则一定发散']" :answer="0" explanation="区间内的小误差可对应区间外巨大变化；拟合好插值不等于有可信外推结构。" />

## 3. RBF 基函数表达局部影响

RBF 基

$$
\phi_j(x)=\exp\left[-\frac{\lVert x-c_j\rVert^2}{2s^2}\right]
$$

在中心 $c_j$ 附近接近 1，远处衰减。宽度 $s$ 小时模型局部灵活但方差高；$s$ 大时多个基函数相似，模型平滑但设计矩阵可能共线。

固定中心与宽度时，只学习外层 $\beta$，仍是凸问题。若中心也优化，参数进入指数内部，目标一般非凸。

## 4. 局部加权回归：每个查询点重新定义目标

对查询点 $x_0$，给训练样本权重

$$
w_i(x_0)=\exp\left[-\frac{\lVert x_i-x_0\rVert^2}{2\tau^2}\right].
$$

拟合局部线性模型：

$$
\widehat\beta(x_0)
=\arg\min_\beta\sum_iw_i(x_0)(y_i-x_i^T\beta)^2.
$$

写成 $W(x_0)=\operatorname{diag}(w_i)$，正规方程为

$$
X^TWX\widehat\beta=X^TWy,
$$

所以

$$
\widehat\beta(x_0)=(X^TWX)^{-1}X^TWy.
$$

预测 $\widehat y(x_0)=x_0^T\widehat\beta(x_0)$。带宽 $\tau$ 是主要复杂度参数；每个查询都要局部求解，预测成本高于全局线性模型。

<div class="ml-stop"><strong>停下来推一推 2</strong>当 $\tau\to\infty$，所有权重趋近什么？局部加权回归会退化成怎样的全局模型？</div>

## 5. 平滑矩阵与有效自由度

Ridge 基函数模型的预测为

$$
\widehat y
=\Phi(\Phi^T\Phi+\lambda I)^{-1}\Phi^Ty
=S_\lambda y.
$$

$S_\lambda$ 称平滑矩阵。与 OLS 投影矩阵不同，$\lambda>0$ 时通常不幂等。有效自由度定义为

$$
\operatorname{df}_{eff}=\operatorname{tr}(S_\lambda).
$$

用 SVD $\Phi=U\Sigma V^T$：

$$
S_\lambda
=U\operatorname{diag}\left(\frac{s_j^2}{s_j^2+\lambda}\right)U^T,
$$

因此

$$
\boxed{\operatorname{df}_{eff}
=\sum_j\frac{s_j^2}{s_j^2+\lambda}}.
$$

当 $\lambda=0$ 且满秩，等于参数数 $m$；$\lambda$ 增大时，每个方向贡献从 1 向 0 收缩。模型可以有许多基函数，但正则让有效复杂度远小于名义参数数。

## 6. 近似误差与估计误差

扩大基函数集合降低“最好可表达函数”与真实函数之间的近似误差；但需要估计更多方向，提高抽样方差。正则、样本量和基函数设计共同控制总风险。

这也说明为什么不能只用“参数数量”衡量复杂度：相同 $m$，强 Ridge 的有效自由度更低；相同次数，不同输入范围导致条件数完全不同。

## 7. NumPy：基函数与验证选择

```python
import numpy as np

rng = np.random.default_rng(9)
x = np.sort(rng.uniform(-1, 1, 80))
y = np.sin(3 * x) + rng.normal(0, .15, len(x))

def polynomial_design(x, degree):
    x = np.asarray(x)
    return np.column_stack([x ** power for power in range(degree + 1)])

def ridge_fit(Phi, y, regularization):
    penalty = np.eye(Phi.shape[1])
    penalty[0, 0] = 0  # 不惩罚截距
    return np.linalg.solve(Phi.T @ Phi + regularization * penalty, Phi.T @ y)

Phi = polynomial_design(x, degree=7)
beta = ridge_fit(Phi, y, regularization=.2)
prediction = Phi @ beta
print(np.mean((y - prediction) ** 2))
```

真正选择次数与 $\lambda$ 时，把 `polynomial_design`、缩放和拟合都放进交叉验证折内部。

## 8. 练习：实现 RBF Ridge

在 $[-1,1]$ 均匀放置 15 个中心，实现 RBF 设计矩阵。用验证集比较宽度 $s\in\{0.05,0.15,0.4\}$ 与 $\lambda\in\{10^{-4},10^{-2},1\}$，报告验证最优组合和有效自由度。

<details><summary>参考实现</summary>

```python
def rbf_design(x, centers, width):
    x = np.asarray(x)[:, None]
    centers = np.asarray(centers)[None, :]
    features = np.exp(-((x - centers) ** 2) / (2 * width ** 2))
    return np.column_stack([np.ones(len(x)), features])

def effective_degrees_of_freedom(Phi, regularization):
    penalty = np.eye(Phi.shape[1])
    penalty[0, 0] = 0
    smoother = Phi @ np.linalg.solve(
        Phi.T @ Phi + regularization * penalty,
        Phi.T,
    )
    return np.trace(smoother)
```

</details>

<div class="ml-stop"><strong>停下来推一推 3</strong>同时把 RBF 宽度减小、正则减弱，会让偏差和方差朝哪个方向变化？写出你的预测，再用重复抽样验证，不要只看一次曲线。</div>

## 9. 常见误解与边界

- 输入非线性不等于参数非线性；固定特征映射仍可凸优化。
- 多项式适合局部插值，不自带可信外推。
- 基函数越多不等于有效自由度越高，正则会连续收缩各方向。
- 验证必须把基函数中心、缩放和超参数选择放入信息边界。
- 局部加权方法的训练看似便宜，但查询时需要保留数据并重新求解。

<MlQuiz question="Ridge 平滑矩阵 Sλ 在 λ>0 时通常是否幂等？" :options="['是', '否，重复平滑还会继续改变结果', '只要有截距就是', '只有分类时不是']" :answer="1" explanation="其谱值 s²/(s²+λ) 位于 0 与 1 之间，平方后一般不等于自身。" />

<MlQuiz question="有效自由度 tr(Sλ) 随 λ 增大怎样变化？" :options="['不增', '必然增大', '完全不变', '变成负数']" :answer="0" explanation="每个项 s²/(s²+λ) 都随 λ 增大而下降。" />

## 10. 本章小结

- 线性模型可以通过固定非线性基函数表达复杂曲线，关键是对参数仍线性。
- 局部加权回归让每个查询点拥有自己的加权最小二乘问题。
- $\operatorname{tr}(S_\lambda)$ 把正则后的复杂度写成连续有效自由度，比只数参数更精确。

<MlLessonProgress lesson-id="07" next-label="第 08 章：分类首先是一个决策问题" next-href="/courses/machine-learning/08-bayesian-decision" />
