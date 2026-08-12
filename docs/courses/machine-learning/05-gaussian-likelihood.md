---
course: machine-learning
edition: industrial
title: 线性回归：高斯似然与统计推断
description: 以压力变送器多点校准为贯穿案例，从条件高斯模型推导 OLS、噪声方差与系数抽样分布，区分参数区间、均值区间和单次预测区间，并用加权最小二乘与 HC1 稳健协方差处理异方差诊断。
outline: [2, 3]
---

<MlLessonHero number="05" title="线性回归：高斯似然与统计推断" question="最小二乘给出了一条校准线；要声称灵敏度可信、下一次读数落在某个范围内，还需要哪些概率假设与证据？" duration="建议 420–540 分钟" :outcomes="['把线性均值、零条件均值、同方差、独立性和高斯形状拆成不同假设', '从条件联合似然完整推导 OLS 与噪声方差 MLE', '推导系数无偏性、一般夹心协方差和同方差简化式', '构造有限样本 t 区间并区分均值响应与单次观测预测区间', '实现 HC1 异方差稳健协方差与已知方差下的加权最小二乘', '审计相关重复测量、重尾噪声、输入测量误差和模型选择后推断']" />

第 04 章只用几何与凸优化就得到了 OLS 点估计。几何结论不需要高斯噪声，也不会自动给出标准误、置信区间或未来读数分布。

本章增加一层可被检验的概率模型：它能解释平方损失为什么出现，也让“不确定性”成为可计算对象。代价是每个区间都带着假设；公式运行成功不等于假设成立。

## 贯穿案例：压力变送器多点校准

某过程装置使用 0–10 bar、4–20 mA 压力变送器。计量工程师在恒温校准台上施加多个参考压力，等待稳定后记录输出电流，并估计：

- 零点电流是否接近 4 mA；
- 灵敏度是否接近 1.6 mA/bar；
- 温度变化是否产生可辨识漂移；
- 给定参考压力时，平均输出和下一次单独读数的不确定范围；
- 量程两端噪声变大时，常规标准误是否仍可信。

本章数据、仪器和验收数值均为**教学合成数据**，不代表真实校准规范。正式计量还要纳入参考标准证书、分辨率、迟滞、重复性、环境和不确定度传播，本章只研究条件线性回归这一层。

### 数据合同先于回归公式

| 字段 | 样本语义 | 单位 | 进入模型的形式 | 关键边界 |
|---|---|---:|---|---|
| `reference_pressure` | 稳定后参考标准压力 | bar | 原值 | 暂按误差可忽略的条件输入处理 |
| `ambient_temperature` | 稳定窗口平均环境温度 | °C | 减去 23°C | 与压力程序不能完全混淆 |
| `output_current` | 同一稳定窗口平均电流 | mA | 响应 $Y$ | 原始瞬时点不能伪装成独立重复 |
| `run_id` | 独立校准循环 | — | 分组/追溯 | 同循环相关性不能由 HC1 修复 |
| `sequence_index` | 循环内施压顺序 | — | 诊断 | 用于检查迟滞和时间漂移 |

本章拟合的是“给定参考压力，电流如何分布”。若实际任务是根据带误差的电流反推压力，直接代数求逆并不等于重新拟合了正确的反向条件模型；参考压力也有明显误差时，则进入 errors-in-variables 问题。

```python
import numpy as np

rng = np.random.default_rng(20260805)
pressure_levels = np.linspace(0.0, 10.0, 14)
reference_pressure = np.tile(pressure_levels, 12)
n_observations = reference_pressure.size
ambient_temperature = rng.normal(23.0, 1.8, n_observations)
run_id = np.repeat(np.arange(12), pressure_levels.size)

# 打乱展示顺序，避免后续代码依赖“压力已经排序”的偶然事实。
order = rng.permutation(n_observations)
reference_pressure = reference_pressure[order]
ambient_temperature = ambient_temperature[order]
run_id = run_id[order]

X = np.column_stack([
    np.ones(n_observations),
    reference_pressure,
    ambient_temperature - 23.0,
])
beta_true = np.array([4.01, 1.598, 0.006])
sigma_true = 0.045
output_current = X @ beta_true + rng.normal(0.0, sigma_true, n_observations)

assert X.shape == (168, 3)
assert output_current.shape == (168,)
assert np.linalg.matrix_rank(X) == X.shape[1]
assert np.isfinite(X).all() and np.isfinite(output_current).all()
```

<MlQuiz kind="数据契约" question="团队希望根据现场输出电流反推压力。为什么不能把本章拟合的 current=a+b×pressure 直接求逆后就声称得到无偏压力校准？" :options="['除法在机器学习中不允许', '反向条件分布不同，且电流与参考压力的测量误差会改变估计问题', '因为截距必须为 0', '只有神经网络能反推压力']" :answer="1" explanation="回归方向由条件对象决定。输入测量误差、响应误差和验收损失在反向任务中都不同；代数求逆没有自动传播这些不确定性。" />

## 1. 一句话中其实有五个假设

经典条件高斯线性模型写作

$$
Y_i\mid x_i
\sim\mathcal N(x_i^T\beta,\sigma^2),
$$

等价地

$$
Y_i=x_i^T\beta+\varepsilon_i,
\qquad
\varepsilon_i\mid X\overset{iid}{\sim}\mathcal N(0,\sigma^2).
$$

这段简写应拆成可分别审计的命题：

1. **条件均值线性**：$\mathbb E[Y_i\mid X]=x_i^T\beta$；
2. **零条件均值**：$\mathbb E[\varepsilon_i\mid X]=0$；
3. **同方差**：$\operatorname{Var}(\varepsilon_i\mid X)=\sigma^2$；
4. **条件不相关/独立**：不同样本误差没有共同波动；
5. **高斯形状**：给定 $X$ 后误差分布为高斯。

满列秩是设计矩阵条件，不属于噪声分布。参考压力在经典推导中被视为固定，或我们对随机获得的 $X$ 条件化；它不意味着现场输入真的没有测量误差。

### 不同结论使用的假设并不相同

| 结论 | 需要的核心条件 | 不必要求 |
|---|---|---|
| OLS 代数解与投影 | 有限矩阵、平方损失；系数唯一还需满列秩 | 高斯、同方差、独立 |
| $\widehat\beta$ 条件无偏 | 均值模型正确且 $\mathbb E[\varepsilon\mid X]=0$ | 高斯、同方差 |
| $\sigma^2(X^TX)^{-1}$ 协方差 | 在无偏条件上再加 $\operatorname{Cov}(\varepsilon\mid X)=\sigma^2I$ | 高斯 |
| 有限样本精确 t/F 推断 | 经典线性模型和条件高斯 | 仅靠大样本近似 |
| 大样本稳健区间 | 适当矩条件、独立或正确处理相关结构、均值模型正确 | 同方差、高斯 |

“残差看起来像钟形”只碰到第五项；它不能证明第一项和第二项，而后两项对系数目标是否正确更关键。

<MlQuiz kind="假设映射" question="若 E[ε|X]=0，但误差方差随压力改变，哪项结论仍通常成立？" :options="['OLS 系数条件无偏', '常规同方差标准误一定正确', '所有有限样本 t 区间精确覆盖', '每个观测权重都应相同且最有效']" :answer="0" explanation="无偏推导只使用零条件均值。异方差会改变协方差与效率，因而伤害常规标准误和精确 t 推断。" />

## 2. 密度与似然是同一公式的两种阅读方式

对单个读数，条件密度是

$$
p(y_i\mid x_i,\beta,\sigma^2)
=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[
-\frac{(y_i-x_i^T\beta)^2}{2\sigma^2}
\right].
$$

训练前，把 $y_i$ 当变量、参数固定，它描述可能观测的概率密度。观察数据后，把 $y$ 与 $X$ 固定、比较不同参数，它被称为似然：

$$
L(\beta,\sigma^2;y,X)
=p(y\mid X,\beta,\sigma^2).
$$

似然不是“参数的概率分布”；频率学派中参数在这里不是随机变量。若要得到参数后验分布，还需要先验与 Bayes 规则。

给定 $X$ 后条件独立，联合似然可以相乘：

$$
L(\beta,\sigma^2)
=\prod_{i=1}^n
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left[
-\frac{(y_i-x_i^T\beta)^2}{2\sigma^2}
\right].
$$

计算时使用对数似然，避免大量小密度相乘导致下溢：

$$
\ell(\beta,\sigma^2)
=-\frac n2\log(2\pi)
-\frac n2\log\sigma^2
-\frac{1}{2\sigma^2}\lVert y-X\beta\rVert_2^2.
$$

<MlQuiz kind="概念辨析" question="观察到 y 后，L(β)=p(y|X,β) 应怎样理解？" :options="['β 的概率密度，积分必为 1', '固定数据后用于比较不同 β 对这份数据的支持程度，不要求对 β 积分为 1', '新的测试标签', '经验风险的单位换算']" :answer="1" explanation="同一个条件密度表达式在参数方向上称为似然。归一化约束针对 y 的分布，不针对 β。" />

## 3. 高斯最大似然为什么落在 OLS 点上

固定 $\sigma^2>0$，对数似然前两项与 $\beta$ 无关，最后一项只是 RSS 乘以负的正常数。因此

$$
\begin{aligned}
\widehat\beta_{MLE}
&\in\arg\max_\beta\ell(\beta,\sigma^2)\\
&=\arg\min_\beta\lVert y-X\beta\rVert_2^2\\
&=\widehat\beta_{OLS}.
\end{aligned}
$$

若 $X$ 满列秩，

$$
\widehat\beta
=(X^TX)^{-1}X^Ty.
$$

若 $X$ 秩亏，最大似然预测仍是唯一投影，但系数 MLE 不是唯一；伪逆只是从等价最大值中选最小范数解。

<GaussianLikelihoodLab />

### 未知 $\sigma$ 不会改变 $\widehat\beta$，但会改变似然高度

给定任意 $\beta$，对 $s=\sigma^2$ 求导：

$$
\frac{\partial\ell}{\partial s}
=-\frac{n}{2s}
+\frac{\operatorname{RSS}(\beta)}{2s^2}.
$$

令其为零，得到剖面 MLE：

$$
\widehat\sigma^2(\beta)
=\frac{\operatorname{RSS}(\beta)}{n}.
$$

代回对数似然后，最大化仍等价于最小化 RSS，所以联合 MLE 的 $\widehat\beta$ 与 OLS 相同。

```python
def gaussian_log_likelihood(beta, sigma, design, target):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    residual = target - design @ beta
    n = target.size
    return (
        -0.5 * n * np.log(2.0 * np.pi)
        -n * np.log(sigma)
        -0.5 * (residual @ residual) / sigma ** 2
    )

beta_hat = np.linalg.lstsq(X, output_current, rcond=None)[0]
residual = output_current - X @ beta_hat
rss = residual @ residual
sigma2_mle = rss / n_observations
sigma_mle = np.sqrt(sigma2_mle)

nearby_beta = beta_hat + np.array([0.02, -0.004, 0.001])
assert gaussian_log_likelihood(beta_hat, sigma_mle, X, output_current) > gaussian_log_likelihood(
    nearby_beta, sigma_mle, X, output_current
)
assert np.linalg.norm(X.T @ residual) < 1e-9
print("MLE beta:", np.round(beta_hat, 6), "sigma:", round(sigma_mle, 6))
```

<div class="ml-stop"><strong>停下来推一推 1</strong>固定同一个 $\sigma$，把 RSS 乘以 $1/(2\sigma^2)$ 为什么不改变最优 $\beta$？如果每个样本拥有不同的已知 $\sigma_i$，哪一项会进入每个残差的权重？</div>

<MlQuiz kind="推导补全" question="在同方差高斯模型中，把假设 σ 从 0.05 改为 0.10，会怎样影响 β 的 MLE？" :options="['最优 β 不变，但似然曲面的尖锐程度改变', 'β 必须减半', '截距消失', 'OLS 变成 Lasso']" :answer="0" explanation="所有残差平方被同一个正数缩放，最小点不变；参数偏离最优点的对数似然代价会按 1/σ² 缩放。" />

## 4. 方差 MLE 为什么除以 n，而常用估计除以 n−p

联合最大似然给出

$$
\widehat\sigma^2_{MLE}=\frac{\operatorname{RSS}}{n}.
$$

MLE 优化的是观测数据在模型下的似然，不承诺有限样本无偏。带截距、满列秩的正确同方差模型中，残差为

$$
r=(I-H)y=(I-H)\varepsilon=M\varepsilon.
$$

因为 $M$ 对称且幂等，$\operatorname{tr}(M)=n-p$。利用 $\mathbb E[\varepsilon\varepsilon^T\mid X]=\sigma^2I$：

$$
\begin{aligned}
\mathbb E[\operatorname{RSS}\mid X]
&=\mathbb E[\varepsilon^TM\varepsilon\mid X]\\
&=\operatorname{tr}\left(M\,\mathbb E[\varepsilon\varepsilon^T\mid X]\right)\\
&=\sigma^2\operatorname{tr}(M)\\
&=(n-p)\sigma^2.
\end{aligned}
$$

因此经典无偏估计为

$$
s^2=\frac{\operatorname{RSS}}{n-p}.
$$

$p$ 必须是设计矩阵的有效秩，不是“业务字段数量”；重复列不应被重复计算自由度。

```python
p = X.shape[1]
H = X @ np.linalg.pinv(X)
M = np.eye(n_observations) - H
sigma2_unbiased = rss / (n_observations - p)

noise_draws = rng.normal(0.0, sigma_true, size=(n_observations, 600))
simulated_residuals = M @ noise_draws
simulated_rss = np.sum(simulated_residuals ** 2, axis=0)
expected_rss = (n_observations - p) * sigma_true ** 2

assert np.isclose(np.trace(M), n_observations - p)
assert sigma2_mle < sigma2_unbiased
assert np.isclose(simulated_rss.mean(), expected_rss, rtol=0.03)
print("E[RSS] simulation / theory:", simulated_rss.mean(), expected_rss)
```

<MlQuiz kind="数值判断" question="n=100、设计矩阵有效秩 p=5、RSS=95。σ² 的 MLE 与经典无偏估计分别是多少？" :options="['0.95 与 1.00', '1.00 与 0.95', '95 与 5', '二者都为 1.05']" :answer="0" explanation="MLE 为 95/100=0.95；无偏估计为 95/(100−5)=1。" />

## 5. 系数为什么波动：线性变换与夹心协方差

在正确均值模型下

$$
y=X\beta+\varepsilon.
$$

满列秩 OLS 可写为

$$
\begin{aligned}
\widehat\beta
&=(X^TX)^{-1}X^T(X\beta+\varepsilon)\\
&=\beta+(X^TX)^{-1}X^T\varepsilon.
\end{aligned}
$$

给定 $X$，若 $\mathbb E[\varepsilon\mid X]=0$：

$$
\mathbb E[\widehat\beta\mid X]=\beta.
$$

令 $\Omega=\operatorname{Cov}(\varepsilon\mid X)$，一般协方差为

$$
\boxed{
\operatorname{Cov}(\widehat\beta\mid X)
=(X^TX)^{-1}X^T\Omega X(X^TX)^{-1}
}.
$$

两侧的 $(X^TX)^{-1}$ 像面包，中间的 $X^T\Omega X$ 像夹心，因此常被称为 sandwich covariance。若 $\Omega=\sigma^2I$，简化为

$$
\boxed{
\operatorname{Cov}(\widehat\beta\mid X)
=\sigma^2(X^TX)^{-1}
}.
$$

第 04 章的小奇异值在这里再次出现：沿弱奇异向量 $v_j$ 的系数方差与 $\sigma^2/\sigma_j^2(X)$ 成正比。数值不稳定和统计不确定性来自同一设计几何，但前者是舍入/求解误差，后者是重复采样波动，不能混为一谈。

```python
gram = X.T @ X
gram_inverse = np.linalg.solve(gram, np.eye(p))
theoretical_covariance = sigma_true ** 2 * gram_inverse

noise_for_coefficients = rng.normal(0.0, sigma_true, size=(n_observations, 3000))
beta_repeats = beta_true[:, None] + np.linalg.solve(
    gram,
    X.T @ noise_for_coefficients,
)
empirical_covariance = np.cov(beta_repeats, rowvar=True, ddof=0)

assert np.allclose(
    np.diag(empirical_covariance),
    np.diag(theoretical_covariance),
    rtol=0.12,
)
print("empirical SE:", np.sqrt(np.diag(empirical_covariance)))
print("theoretical SE:", np.sqrt(np.diag(theoretical_covariance)))
```

### Gauss–Markov 定理没有要求高斯

在零条件均值、同方差不相关误差和满列秩下，OLS 是所有**线性无偏估计量**中协方差最小的，即 BLUE。设另一个线性无偏估计量为 $Cy$，满足 $CX=I$。写成

$$
C=(X^TX)^{-1}X^T+D,
\qquad DX=0.
$$

交叉项因 $DX=0$ 消失，协方差差为

$$
\operatorname{Cov}(Cy\mid X)
-\operatorname{Cov}(\widehat\beta\mid X)
=\sigma^2DD^T\succeq0.
$$

“最优”只限定在线性无偏类中，不代表 OLS 对异常值最稳健、预测风险最低或所有估计器中最好。

<MlQuiz kind="定理边界" question="Gauss–Markov 定理中的 BLUE 最准确的含义是什么？" :options="['OLS 在所有可能算法中测试误差最低', '在经典条件下，OLS 在线性无偏估计量类中协方差最小', 'OLS 的每个预测都等于真值', '必须先证明误差高斯']" :answer="1" explanation="定理不要求高斯，也不比较有偏估计、稳健损失或非线性算法。BLUE 的比较类和假设都不能省略。" />

<div class="ml-stop"><strong>停下来推一推 2</strong>样本覆盖相似且 $X^TX$ 随 $n$ 近似线性增长时，系数方差与标准误大致怎样随 $n$ 缩放？为什么“标准误减半”通常需要约四倍独立信息，而不是两倍原始行数？</div>

## 6. 高斯假设把标准化系数变成 t 分布

若误差条件高斯，$\widehat\beta$ 是高斯向量的线性变换：

$$
\widehat\beta\mid X
\sim\mathcal N\left(
\beta,
\sigma^2(X^TX)^{-1}
\right).
$$

同时，在经典线性模型中

$$
\frac{(n-p)s^2}{\sigma^2}\sim\chi^2_{n-p},
$$

并且 $\widehat\beta$ 与 $s^2$ 独立。于是第 $j$ 个系数的统计量

$$
T_j=
\frac{\widehat\beta_j-\beta_j}
{s\sqrt{[(X^TX)^{-1}]_{jj}}}
\sim t_{n-p}.
$$

双侧 $1-\alpha$ 置信区间为

$$
\widehat\beta_j
\pm
t_{1-\alpha/2,n-p}
s\sqrt{[(X^TX)^{-1}]_{jj}}.
$$

频率学派 95% 置信区间的覆盖含义是：若完整采样和拟合流程无限重复，约 95% 这样构造的区间会覆盖固定真参数。它不是“给定这份数据，真参数有 95% 概率位于此区间”。

```python
degrees_of_freedom = n_observations - p
t_critical_95 = 1.974  # df=165 的双侧 95% 临界值，固定写入以避免新增 SciPy 依赖。
estimated_covariance = sigma2_unbiased * gram_inverse
coefficient_se = np.sqrt(np.diag(estimated_covariance))
coefficient_interval = np.column_stack([
    beta_hat - t_critical_95 * coefficient_se,
    beta_hat + t_critical_95 * coefficient_se,
])

assert np.all(coefficient_interval[:, 0] < coefficient_interval[:, 1])
assert np.all(coefficient_se > 0)
print("coefficient intervals:\n", np.round(coefficient_interval, 6))
```

### 单个系数区间不是整个报告的万能通行证

- 同时检查很多系数时，“每个 95%”不等于“全部同时 95%”；
- 看完 p 值再删变量、挑交互、换分组后，原区间没有计算选择过程；
- 实际意义不能由统计显著性替代：0.0001 mA/bar 的偏差即使显著，也可能低于计量分辨率；
- 不显著不等于无效应，可能只是覆盖不足、共线或区间太宽；
- 系数区间只在模型定义的条件目标内成立，不自动支持因果结论。

<MlQuiz kind="解释审计" question="“灵敏度 95% 置信区间为 [1.596,1.600]”的正确频率解释是什么？" :options="['真灵敏度有 95% 后验概率位于区间内', '重复执行同一采样与构造流程时，长期约 95% 的区间覆盖固定真灵敏度', '未来 95% 的电流都位于该区间', '模型正确的概率为 95%']" :answer="1" explanation="置信水平描述区间构造程序的重复采样覆盖率，不是给定数据后对固定参数赋概率。" />

## 7. 参数区间、均值响应区间与单次预测区间

给定新工况向量 $x_0$，平均响应为 $\mu_0=x_0^T\beta$，估计为 $\widehat\mu_0=x_0^T\widehat\beta$。其估计方差是

$$
\operatorname{Var}(\widehat\mu_0\mid X)
=\sigma^2x_0^T(X^TX)^{-1}x_0.
$$

平均响应的置信区间使用

$$
s\sqrt{x_0^T(X^TX)^{-1}x_0}.
$$

下一次单独读数

$$
Y_0=x_0^T\beta+\varepsilon_0
$$

还包含新的不可约测量噪声，预测标准误为

$$
s\sqrt{1+x_0^T(X^TX)^{-1}x_0}.
$$

所以单次预测区间必然比同一点的均值响应区间宽。两者都会在远离设计中心、杠杆值较高的外推点变宽；但若模型结构在外推区错误，公式变宽也未必足够保护风险。

```python
def response_intervals(x0, beta, sigma2, gram_inv, critical):
    x0 = np.asarray(x0, dtype=float)
    mean = float(x0 @ beta)
    parameter_variance = float(sigma2 * x0 @ gram_inv @ x0)
    mean_half_width = critical * np.sqrt(parameter_variance)
    prediction_half_width = critical * np.sqrt(sigma2 + parameter_variance)
    return {
        "mean": mean,
        "mean_interval": (mean - mean_half_width, mean + mean_half_width),
        "prediction_interval": (
            mean - prediction_half_width,
            mean + prediction_half_width,
        ),
        "mean_half_width": mean_half_width,
        "prediction_half_width": prediction_half_width,
    }

center_condition = np.array([1.0, 5.0, 0.0])
outside_condition = np.array([1.0, 13.0, 0.0])
center_report = response_intervals(
    center_condition, beta_hat, sigma2_unbiased, gram_inverse, t_critical_95
)
outside_report = response_intervals(
    outside_condition, beta_hat, sigma2_unbiased, gram_inverse, t_critical_95
)

assert center_report["prediction_half_width"] > center_report["mean_half_width"]
assert outside_report["mean_half_width"] > center_report["mean_half_width"]
print(center_report)
```

<MlQuiz kind="区间选择" question="验收人员问“同一压力和温度下，下一次单独电流读数大约落在哪里”。应优先报告哪种区间？" :options="['β₁ 的系数区间', '平均响应置信区间', '包含新观测噪声的预测区间', '训练 RSS 区间']" :answer="2" explanation="问题针对下一次单独观测，既包含参数估计误差，也包含新的测量噪声；均值区间只描述条件均值估计。" />

## 8. 异方差：点估计可能还对，常规区间已经错

若

$$
\operatorname{Cov}(\varepsilon\mid X)
=\Omega
=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_n^2),
$$

零条件均值下 OLS 仍条件无偏，但真实协方差是夹心式，而不是 $\sigma^2(X^TX)^{-1}$。

### 方差已知或有可信模型：加权最小二乘

条件高斯但每个样本方差不同，负对数似然中与 $\beta$ 有关的部分是

$$
\frac12\sum_{i=1}^n
\frac{(y_i-x_i^T\beta)^2}{\sigma_i^2}.
$$

令 $W=\operatorname{diag}(1/\sigma_i^2)$，得到

$$
\widehat\beta_{WLS}
=(X^TWX)^{-1}X^TWy.
$$

小方差读数权重大，大方差读数权重小。权重必须来自噪声方差或明确业务目标；把“重要样本权重”“抽样权重”和“逆方差权重”混用，会改变估计对象。

### 方差未知：HC0/HC1 夹心估计

用 OLS 残差 $r_i$ 估计未知中间层，HC0 为

$$
\widehat V_{HC0}
=(X^TX)^{-1}
X^T\operatorname{diag}(r_i^2)X
(X^TX)^{-1}.
$$

HC1 乘有限样本修正 $n/(n-p)$。它在适当大样本条件下对异方差稳健，但不是有限样本精确 t 理论，也不会改变 OLS 点估计。

<HeteroskedasticCoverageLab />

```python
edge_distance = np.abs(reference_pressure - 5.0) / 5.0
sigma_by_observation = 0.02 * (1.0 + 0.7 * 12.0 * edge_distance ** 3)
heteroskedastic_current = X @ beta_true + rng.normal(
    0.0, sigma_by_observation, n_observations
)

beta_heteroskedastic = np.linalg.lstsq(X, heteroskedastic_current, rcond=None)[0]
heteroskedastic_residual = heteroskedastic_current - X @ beta_heteroskedastic
heteroskedastic_rss = heteroskedastic_residual @ heteroskedastic_residual
classic_sigma2 = heteroskedastic_rss / (n_observations - p)
classic_covariance = classic_sigma2 * gram_inverse

meat = X.T @ (heteroskedastic_residual[:, None] ** 2 * X)
hc1_covariance = (
    n_observations / (n_observations - p)
    * gram_inverse @ meat @ gram_inverse
)

weighted_X = X / sigma_by_observation[:, None]
weighted_y = heteroskedastic_current / sigma_by_observation
beta_wls = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)[0]

assert np.allclose(hc1_covariance, hc1_covariance.T)
assert np.all(np.linalg.eigvalsh(hc1_covariance) >= -1e-12)
assert np.isfinite(beta_wls).all()
print("classic SE:", np.sqrt(np.diag(classic_covariance)))
print("HC1 SE:", np.sqrt(np.diag(hc1_covariance)))
print("OLS / WLS sensitivity:", beta_heteroskedastic, beta_wls)
```

### HC1 不能修复相关重复测量

同一 `run_id` 内的零点漂移会让 $\Omega$ 出现非对角块。逐点 HC1 仍假设样本之间足够独立，通常低估相关组带来的不确定性。应按独立校准循环使用 cluster-robust covariance、组 bootstrap 或显式层级/相关模型；有效独立信息更接近循环数，而不是瞬时读数行数。

<div class="ml-stop"><strong>停下来推一推 3</strong>把每个稳定窗口复制 100 次，OLS 系数不会改变。若软件把 100 份复制当独立样本，常规标准误会怎样变化？真实独立信息真的增加了吗？</div>

<MlQuiz kind="错误诊断" question="残差呈漏斗形，团队只把 covariance_type 改成 HC1 后宣称模型已经修复。还缺什么？" :options="['HC1 已同时修复均值非线性、相关性和外推', 'HC1 只修正异方差下的协方差估计；仍需检查均值模型、相关组、影响点和目标域', '必须删除所有大残差', '只要区间变宽就无需验证']" :answer="1" explanation="稳健标准误不是稳健模型。它不改变点预测，也不纠正遗漏变量、非线性、组相关或数据泄漏。" />

## 9. 高斯尾部不合适时，损失也应接受压力测试

高斯负对数似然使单点贡献随残差平方增长：

$$
-\log p_{Gaussian}(r)=C+\frac{r^2}{2\sigma^2}.
$$

这对真正的高斯噪声有效率，但一个极端计量错误可能主导拟合。替代观测模型对应不同损失：

| 噪声/方法 | 单点损失核心 | 工程含义 | 边界 |
|---|---|---|---|
| Gaussian | $r^2$ | 中心误差高效，解析简单 | 对极端值敏感 |
| Laplace | $|r|$ | 中位数目标，尾部影响较小 | 在 0 不光滑；参数目标改变 |
| Student-t | $\log(1+r^2/c)$ 类重尾 | 给极端读数更高概率 | 自由度选择与优化更复杂 |
| Huber | 小残差平方，大残差线性 | 保留中心效率并限制影响 | 阈值需冻结或验证选择 |

稳健损失不等于自动删除异常点。异常可能来自传感器故障，也可能是真实迟滞、饱和或遗漏工况；应保留行标识、回看计量链并报告敏感性。

<MlQuiz kind="模型选择" question="若把高斯观测模型改为 Laplace，最大似然对应的主要残差目标是什么？" :options="['绝对残差和', '残差平方和', '0–1 错误数', '系数 L2 范数']" :answer="0" explanation="Laplace 负对数密度与 |r| 成正比。改变观测分布会改变训练损失与中心目标。" />

## 数值验证

用重复采样验证公式时，必须明确哪些量固定、哪些量随机。下面固定设计 $X$ 与真实 $\beta$，只重抽噪声；它验证条件协方差，不代表随机工况覆盖或跨仪器泛化。

```python
def slope_coverage_experiment(sample_size=60, strength=0.6, repeats=800, seed=31):
    local_rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, sample_size)
    centered_x = x - x.mean()
    sxx = centered_x @ centered_x
    classic_cover = 0
    hc1_cover = 0
    slope_estimates = []

    for _ in range(repeats):
        sigma = 0.04 * (
            1.0 + strength * 12.0 * (np.abs(x - 0.5) / 0.5) ** 3
        )
        error = local_rng.normal(0.0, sigma)
        slope_error = centered_x @ error / sxx
        intercept_error = error.mean() - x.mean() * slope_error
        fitted_residual = error - intercept_error - slope_error * x
        classic_se = np.sqrt(
            (fitted_residual @ fitted_residual) / (sample_size - 2) / sxx
        )
        hc1_se = np.sqrt(
            sample_size / (sample_size - 2)
            * np.sum(centered_x ** 2 * fitted_residual ** 2) / sxx ** 2
        )
        classic_cover += abs(slope_error) <= 1.96 * classic_se
        hc1_cover += abs(slope_error) <= 1.96 * hc1_se
        slope_estimates.append(2.0 + slope_error)

    return {
        "classic_coverage": classic_cover / repeats,
        "hc1_coverage": hc1_cover / repeats,
        "empirical_slope_sd": float(np.std(slope_estimates, ddof=0)),
    }

coverage_report = slope_coverage_experiment()
assert coverage_report["hc1_coverage"] > coverage_report["classic_coverage"]
assert coverage_report["hc1_coverage"] > 0.90
print(coverage_report)
```

本章验证链至少应回答：

1. `lstsq` 得到的 $\widehat\beta$ 是否同时最大化高斯剖面对数似然；
2. $X^Tr$ 是否接近 0，数值秩与条件数是否已记录；
3. 同方差仿真中经验系数协方差是否匹配 $\sigma^2(X^TX)^{-1}$；
4. 模拟 RSS 均值是否匹配 $(n-p)\sigma^2$；
5. 单次预测区间是否严格宽于同工况均值区间；
6. 异方差压力测试中常规与 HC1 覆盖率是否分离；
7. 按 `run_id` 重采样后，区间是否比逐行独立假设更宽；
8. 高杠杆、迟滞、饱和、时间漂移和温度—压力混淆是否单独检查。

一次数据上“HC1 比常规 SE 大”不是稳健性的定义；覆盖率是重复采样性质，真实项目只能通过设计、理论条件、模拟和独立复现实验共同建立证据。

## 复杂度与工程边界

令 $n$ 为样本数、$p$ 为参数数，下面假设稠密设计矩阵：

| 操作 | 典型时间 | 主要存储 | 工业注意点 |
|---|---:|---:|---|
| QR/SVD OLS | 约 $O(np^2)$ | $O(np)$ 或分块表示 | 沿用第 04 章稳定求解器，不显式求逆 |
| 经典协方差 | 已有分解后约 $O(p^2)$ | $O(p^2)$ | 仅在同方差相关条件可信时使用 |
| HC0/HC1 meat | $O(np^2)$ | 可流式累计 $O(p^2)$ | 大样本可分块累加 $x_ix_i^Tr_i^2$ |
| WLS | 与 OLS 同量级 | 权重向量 + 分解 | 极端权重会造成新病态，应审计权重范围 |
| cluster-robust | 约 $O(np+Gp^2)$ | 每组 score 和 | 组数 $G$ 小时渐近近似脆弱 |
| 参数 bootstrap | 约为 $B$ 次拟合 | 可并行 | 只复现设定分布，不能暴露模型错误 |
| 组/非参数 bootstrap | 约为 $B$ 次拟合 | 分组索引 | 重采样单位必须对应独立信息单位 |

### 上线前必须写进报告的边界

- **固定设计条件性**：本章多数公式给出给定当前 $X$ 的不确定性，不包含未来压力/温度分布改变；
- **参考标准误差**：若参考压力不再近似精确，应使用测量误差模型或计量不确定度传播；
- **校准方向**：预测电流的模型不能无条件当作预测压力的模型；
- **相关结构**：同一次循环、同一仪器或同一环境段的重复读数必须按组处理；
- **模型选择后推断**：用同一数据筛变量、选变换、挑异常处理后，普通区间通常偏乐观；
- **覆盖域**：0–10 bar 内的校准不支持 13 bar 外推，公式给出的有限区间不代表物理有效；
- **统计与计量容差**：区间、最大允许误差和决策风险是不同对象，验收规则应预先冻结；
- **因果边界**：温度系数是给定模型下的条件关联，压力程序与温度时间趋势混淆时不能解释为温度因果效应。

<MlQuiz kind="工程审计" question="团队在同一份数据上尝试 40 种特征和异常值规则，只报告最终模型的普通 95% 区间。最大的问题是什么？" :options="['区间小数位不足', '区间没有计算模型选择与规则搜索带来的额外随机性，通常偏乐观', '尝试越多区间越精确', '只需把置信水平改名为准确率']" :answer="1" explanation="最终模型是数据驱动选择的随机结果。选择后仍套用预先固定模型的区间，覆盖率通常不再是标称值。" />

## 10. 章节练习：提交一份可复核校准推断报告

### 推导练习

1. 从联合高斯似然同时求 $\widehat\beta$ 和 $\widehat\sigma^2_{MLE}$，说明为何二者可分步优化。
2. 使用迹恒等式证明 $\mathbb E[\operatorname{RSS}\mid X]=(n-p)\sigma^2$。
3. 从 $\widehat\beta=\beta+(X^TX)^{-1}X^T\varepsilon$ 推导一般夹心协方差。
4. 证明已知 $\Omega$ 时，对 $\Omega^{-1/2}y$ 与 $\Omega^{-1/2}X$ 做 OLS 等价于 GLS。
5. 推导新工况下均值区间与单次预测区间的方差差，标出独立新噪声项。

### 实现练习

扩展本章数据：加入 12 个 `run_id` 的随机零点漂移、量程两端异方差和 3 个计量异常。比较常规、HC1、按循环 cluster-robust 与按循环 bootstrap 的灵敏度区间，并报告覆盖率模拟。

```python
# TODO：由学习者完成；本代码块不参与自动执行。
def calibration_inference_report(X, y, run_ids, prediction_points):
    # 1. 稳定拟合 OLS，记录秩、条件数、残差和杠杆值
    # 2. 返回 classic、HC1、cluster-robust 三种协方差
    # 3. 按 run_id 做组 bootstrap，禁止逐行重采样
    # 4. 区分系数、均值响应和单次观测区间
    # 5. 返回异常/高影响行标识，但不自动删除
    raise NotImplementedError
```

### 报告审计练习

审查以下结论：“温度系数 p<0.05，因此升温必然导致零点漂移；95% 预测区间覆盖了训练数据的 96%，所以未来所有仪器都满足计量要求。”

答案至少应指出：条件关联不是因果；温度可能与校准顺序混淆；训练内覆盖不是未来覆盖；预测区间与计量容差不是同一对象；单台仪器不能代表仪器总体；选择过程、相关循环、异方差和目标压力范围都必须进入证据链。

## 11. 本章小结与术语

- 同方差条件高斯模型的负对数似然与 RSS 只差正缩放及常数，因此 $\widehat\beta_{MLE}=\widehat\beta_{OLS}$。
- 方差 MLE 使用 $n$；经典无偏估计使用有效残差自由度 $n-p$，二者优化准则不同。
- 零条件均值保证 OLS 条件无偏；同方差把夹心协方差简化为 $\sigma^2(X^TX)^{-1}$；高斯使有限样本精确 t 推断成立。
- 参数置信区间、平均响应置信区间和单次观测预测区间回答三个不同问题。
- 异方差下可在方差模型可信时使用 WLS；HC1 修正未知异方差下的大样本协方差，但不改变点估计，也不处理组相关。
- 重尾、相关重复、输入测量误差、外推、模型选择和因果解释都超出一条普通 OLS 汇总表能够证明的范围。

<dl class="ml-terms">
  <div><dt>likelihood</dt><dd>观察数据固定后，把条件密度作为参数函数来比较数据对参数的支持。</dd></div>
  <div><dt>maximum likelihood estimate</dt><dd>使观察数据似然最大的参数值；它不是参数后验均值。</dd></div>
  <div><dt>homoskedasticity</dt><dd>给定设计后，每个观测误差拥有相同条件方差。</dd></div>
  <div><dt>heteroskedasticity</dt><dd>条件误差方差随输入或样本改变。</dd></div>
  <div><dt>standard error</dt><dd>估计量重复采样标准差的估计，不是观测值自身标准差。</dd></div>
  <div><dt>confidence interval</dt><dd>具有标称重复采样覆盖率的参数或均值区间构造程序。</dd></div>
  <div><dt>prediction interval</dt><dd>同时纳入参数不确定性和新观测噪声的未来观测区间。</dd></div>
  <div><dt>sandwich covariance</dt><dd>允许一般误差协方差结构的 $A\Omega A^T$ 形式。</dd></div>
  <div><dt>HC1</dt><dd>用逐点残差平方估计异方差中间层并乘 $n/(n-p)$ 修正的稳健协方差。</dd></div>
  <div><dt>weighted least squares</dt><dd>按已知或可信逆方差对残差平方加权的最大似然/广义最小二乘方法。</dd></div>
</dl>

## 来源与许可

- Stanford CS229 Lecture Notes，PDF 第 17–19 页，§1.3：用于核对独立高斯误差、条件似然、对数似然与 OLS 最大似然等价。本站以压力变送器案例重新推导，不复用讲义住房案例、图形或措辞。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 401–406 页，§11.1–11.2.2.4：用于核对条件线性高斯模型、NLL、OLS、异方差模型和加权最小二乘。本站代码、实验和统计推断链独立创作。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 89–90 页，§2.6.3–2.6.4：用于核对同方差/异方差回归以及均值不确定性与观测预测不确定性的区分；不复用原图。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 432–435 页，§11.6：用于核对 Gaussian、Laplace、Student-t 与 Huber 的重尾/稳健回归边界；本站表格、案例和题目独立创作。
- 本章 HC1、t 区间、Gauss–Markov 与夹心协方差推导属于标准统计知识，公式均由本站独立推导和数值验证；未复制外部课程题目或表述。
- 本章压力变送器 168 条读数、校准循环、异方差压力测试、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实计量规范、仪器性能或验收限值。详细登记见[来源与许可登记](./sources.md) 。

<MlLessonProgress lesson-id="05" next-label="第 06 章：Ridge 回归与 Lasso" next-href="/courses/machine-learning/06-regularization" />
