---
course: machine-learning
title: 为什么高斯噪声导出平方损失
description: 从条件高斯模型完整推导 OLS 的最大似然解、噪声方差估计与系数协方差，并分析同方差和重尾噪声失效。
outline: [2, 3]
---

<MlLessonHero number="05" title="为什么高斯噪声导出平方损失" question="最小二乘只是几何距离，还是某个数据生成假设的概率结论？" duration="建议 260–320 分钟" :outcomes="['从独立高斯噪声似然推出 OLS', '推导噪声方差与系数估计的抽样协方差', '识别同方差、正态性、独立性与线性均值假设的不同作用']" />

## 先修诊断

<MlQuiz question="似然 p(y|X,β) 中，哪些量被视为变量？" :options="['观察到的 y 固定，把 β 当作待比较参数', 'β 固定，只让 y 改变', 'X 必须随机积分掉', '似然与概率分布完全同义']" :answer="0" explanation="同一个表达式作为 y 的函数是分布；观察 y 后作为 β 的函数是似然。" />

<MlQuiz question="高斯噪声假设最直接导出哪种训练损失？" :options="['绝对误差', '平方误差', '0-1 损失', 'hinge loss']" :answer="1" explanation="高斯对数密度的参数相关项是负的平方残差除以 2σ²。" />

<MlQuiz question="残差图呈明显漏斗形，首先违反什么？" :options="['零均值', '同方差', '样本数为整数', '矩阵可乘']" :answer="1" explanation="预测水平越大残差散布越宽，说明条件方差随 x 改变。" />

## 1. 把线性均值与随机噪声分开

条件高斯模型写作

$$
Y_i\mid x_i\sim\mathcal N(x_i^T\beta,\sigma^2),
$$

等价地

$$
Y_i=x_i^T\beta+\varepsilon_i,
\qquad \varepsilon_i\overset{iid}{\sim}\mathcal N(0,\sigma^2).
$$

这句话同时包含：

- 条件均值线性：$\mathbb E[Y_i|x_i]=x_i^T\beta$。
- 零均值噪声：$\mathbb E[\varepsilon_i|X]=0$。
- 同方差：每个样本条件方差相同。
- 条件独立：给定 $X$ 后，各 $Y_i$ 独立。
- 正态形状：用于有限样本精确分布和似然形式。

不同结论需要不同强度假设。OLS 系数无偏不要求噪声正态，但精确的 t/F 推断通常需要更强条件。

## 2. 从联合似然逐步得到 SSE

单个样本密度为

$$
p(y_i|x_i,\beta,\sigma^2)
=\frac1{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(y_i-x_i^T\beta)^2}{2\sigma^2}\right].
$$

条件独立使联合密度可以相乘：

$$
p(y|X,\beta,\sigma^2)
=\prod_{i=1}^n p(y_i|x_i,\beta,\sigma^2).
$$

取对数把乘积变成求和：

$$
\begin{aligned}
\ell(\beta,\sigma^2)
&=\sum_{i=1}^n\log p(y_i|x_i,\beta,\sigma^2)\\
&=-\frac n2\log(2\pi)-\frac n2\log\sigma^2
-\frac1{2\sigma^2}\sum_{i=1}^n(y_i-x_i^T\beta)^2.
\end{aligned}
$$

固定 $\sigma^2$ 时，前两项与 $\beta$ 无关，$1/(2\sigma^2)>0$。因此

$$
\arg\max_\beta\ell(\beta,\sigma^2)
=\arg\min_\beta\sum_i(y_i-x_i^T\beta)^2.
$$

于是高斯 MLE 与几何 OLS 给出同一个 $\widehat\beta$。

<div class="ml-stop"><strong>停下来推一推 1</strong>为什么固定 σ 后除以 $2\sigma^2$ 不改变最优 β？因为对所有候选乘同一个正数只缩放目标；若 σ 依赖样本，结论会变成加权最小二乘。</div>

<MlLab kind="likelihood" />

## 3. 同时估计噪声方差

把 $\widehat\beta$ 代回 log-likelihood，对 $s=\sigma^2$ 求导：

$$
\ell(s)=C-\frac n2\log s-\frac{\operatorname{SSE}}{2s}.
$$

所以

$$
\frac{d\ell}{ds}=-\frac n{2s}+\frac{\operatorname{SSE}}{2s^2}=0.
$$

乘以 $2s^2$：

$$
-ns+\operatorname{SSE}=0
\quad\Longrightarrow\quad
\boxed{\widehat\sigma^2_{MLE}=\frac{\operatorname{SSE}}n}.
$$

但估计 $p$ 个独立系数消耗了 $p$ 个自由度。在标准满秩线性模型中，

$$
\mathbb E[\operatorname{SSE}]=(n-p)\sigma^2,
$$

因此无偏估计是

$$
\widehat\sigma^2=\frac{\operatorname{SSE}}{n-p}.
$$

MLE 的目标是最大化似然，不保证有限样本无偏；两者回答不同准则。

<MlQuiz question="为什么 σ² 的 MLE 分母是 n，而常见无偏估计分母是 n−p？" :options="['两个公式必有一个错误', 'MLE 与无偏性是不同准则，拟合 p 个系数消耗自由度', '因为 n−p 总等于 n', '只有分类才需要自由度']" :answer="1" explanation="SSE 的期望是 (n−p)σ²；除以 n 的 MLE 因此向下偏。" />

## 4. 系数估计为什么会波动

在满列秩且模型正确时，$y=X\beta+\varepsilon$。OLS 为

$$
\begin{aligned}
\widehat\beta
&=(X^TX)^{-1}X^Ty\\
&=(X^TX)^{-1}X^T(X\beta+\varepsilon)\\
&=\beta+(X^TX)^{-1}X^T\varepsilon.
\end{aligned}
$$

给定 $X$，若 $\mathbb E[\varepsilon|X]=0$：

$$
\mathbb E[\widehat\beta|X]=\beta.
$$

若 $\operatorname{Cov}(\varepsilon|X)=\sigma^2I$：

$$
\begin{aligned}
\operatorname{Cov}(\widehat\beta|X)
&=(X^TX)^{-1}X^T(\sigma^2I)X(X^TX)^{-1}\\
&=\boxed{\sigma^2(X^TX)^{-1}}.
\end{aligned}
$$

近共线让 $X^TX$ 有很小特征值，其逆在相应方向很大，所以系数方差爆炸。这与第 04 章的数值条件问题是同一谱结构的统计版本。

<div class="ml-stop"><strong>停下来推一推 2</strong>若把样本量增大且 $X^TX$ 近似按 n 线性增长，系数协方差大致怎样缩放？约为 $1/n$，标准误约为 $1/\sqrt n$。</div>

## 5. 高斯性到底负责什么

在高斯噪声下，$\widehat\beta$ 是正态变量的线性变换：

$$
\widehat\beta|X\sim\mathcal N\left(\beta,\sigma^2(X^TX)^{-1}\right).
$$

这给出有限样本精确区间的起点。若噪声不是正态，但零条件均值、有限方差和适当正则条件成立，OLS 仍可无偏/一致，并在大样本下近似正态；小样本精确分布则改变。

### 异方差

若 $\operatorname{Cov}(\varepsilon|X)=\Omega$ 而不是 $\sigma^2I$：

$$
\operatorname{Cov}(\widehat\beta|X)
=(X^TX)^{-1}X^T\Omega X(X^TX)^{-1}.
$$

使用同方差公式会得到错误标准误。若 $\Omega$ 已知，可最小化加权残差

$$
(y-X\beta)^T\Omega^{-1}(y-X\beta),
$$

得到广义最小二乘。

### 重尾与离群点

平方损失的梯度对残差 $r$ 是 $2r$，大残差影响线性增长。Laplace 噪声

$$
p(\varepsilon)\propto\exp(-|\varepsilon|/b)
$$

的负对数似然对应绝对损失。Huber 损失在中心使用平方、尾部使用线性，是效率与稳健性的折中。

## 6. NumPy：似然、OLS 与不确定性

```python
import numpy as np

rng = np.random.default_rng(7)
n = 80
x = np.linspace(-2, 2, n)
X = np.column_stack([np.ones(n), x])
beta_true = np.array([1.5, -0.8])
sigma = .6
y = X @ beta_true + rng.normal(0, sigma, n)

beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
residual = y - X @ beta_hat
sigma2_mle = residual @ residual / n
sigma2_unbiased = residual @ residual / (n - X.shape[1])
covariance = sigma2_unbiased * np.linalg.inv(X.T @ X)
standard_error = np.sqrt(np.diag(covariance))

print(beta_hat)
print(sigma2_mle, sigma2_unbiased, standard_error)
assert sigma2_mle < sigma2_unbiased
```

教学代码用 `inv` 展示协方差公式；生产数值计算应通过 Cholesky/QR 求解，不主动求逆。

## 7. 练习：噪声假设压力测试

固定 $X$ 和真实 $\beta$，分别生成高斯、Laplace、异方差噪声。每种重复 1000 次，记录 $\widehat\beta$ 的均值、经验协方差和 95% 正态区间覆盖率。

<details><summary>实验骨架</summary>

```python
def simulate(noise_factory, repeats=1000):
    estimates = []
    for _ in range(repeats):
        y = X @ beta_true + noise_factory()
        estimates.append(np.linalg.lstsq(X, y, rcond=None)[0])
    estimates = np.asarray(estimates)
    return estimates.mean(axis=0), np.cov(estimates, rowvar=False)
```

</details>

<details><summary>你应看到什么</summary>

零条件均值下，三种情形系数均值仍接近真实值；重尾使抽样分布尾部更厚；异方差下，同方差协方差公式与经验协方差不匹配。若噪声均值依赖 $x$，系数会系统偏离。

</details>

<div class="ml-stop"><strong>停下来推一推 3</strong>“残差近似正态”与“条件均值线性”哪一个对系数无偏更关键？后者。正态形状主要服务似然与精确推断，零条件均值才让误差项不系统进入系数。</div>

## 8. 条件与失效清单

- 线性是对条件均值的假设，不要求每个观测落在直线上。
- 正态性不是 OLS 代数解存在的条件。
- 同方差错误主要伤害标准误与效率，不必然使系数有偏。
- $\mathbb E[\varepsilon|X]\ne0$ 常由遗漏变量、反向因果或选择机制导致，会使系数有偏。
- 独立性破坏常见于时间序列、空间数据与同一主体重复测量。

<MlQuiz question="若 E[ε|X]=0 但误差异方差，OLS 系数通常怎样？" :options="['必然有偏', '仍可无偏，但常规同方差标准误错误且效率可能降低', '不存在', '自动等于 Ridge']" :answer="1" explanation="无偏推导只用零条件均值；协方差简化为 σ²(XᵀX)⁻¹ 才需要同方差。" />

<MlQuiz question="Laplace 噪声的 MLE 对应什么损失？" :options="['绝对误差', '平方误差', '交叉熵', '0-1 损失']" :answer="0" explanation="Laplace 负对数密度与 |y−μ| 成正比。" />

## 9. 本章小结

- 独立同方差高斯条件模型的负对数似然与 SSE 只差正缩放和常数。
- $\widehat\beta$ 的协方差是 $\sigma^2(X^TX)^{-1}$，共线在统计上放大估计方差。
- 应把线性均值、零条件均值、同方差、独立和正态分别检查，不能统称“线性回归假设”。

<MlLessonProgress lesson-id="05" next-label="第 06 章：正则化究竟改变了什么" next-href="/courses/machine-learning/06-regularization" />
