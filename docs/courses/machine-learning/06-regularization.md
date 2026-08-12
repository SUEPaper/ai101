---
course: machine-learning
edition: industrial
title: Ridge 回归与 Lasso
description: 以电机绕组热点温度软测量为贯穿案例，从病态设计矩阵出发，推导 Ridge 的闭式解、谱收缩、有效自由度和高斯 MAP，再由 L1 几何、次梯度、软阈值与坐标下降推导 Lasso，并建立可部署的调参与稳定性审计流程。
outline: [2, 3]
---

<MlLessonHero number="06" title="Ridge 回归与 Lasso" question="一组冗余传感器让 OLS 系数剧烈摆动：应当稳定所有方向，还是把一部分系数压成精确的 0？" duration="建议 480–600 分钟" :outcomes="['从多重共线性与小奇异值解释 OLS 的噪声放大', '从惩罚式、约束式、增广最小二乘和 Gaussian MAP 四条路径推导 Ridge', '使用 SVD 解释方向相关收缩、有效自由度与偏差—方差变化', '从 Laplace MAP、L1 几何、KKT 次梯度和近端映射推导 Lasso 的精确稀疏', '实现 Ridge 稳定求解、Lasso 坐标下降、正则化路径与 KKT 验收', '按测试循环切分数据，联合审计预测误差、系数稳定性、活跃集合与传感器成本']" />

第 04 章已经说明：近共线让残差平方和形成狭长山谷，小标签扰动可以换来巨大系数变化。第 05 章又说明：在正确同方差模型下，OLS 方差包含 $(X^TX)^{-1}$，因此小特征值会直接放大估计不确定性。

正则化不是在 OLS 后面加一条经验规则，而是**主动允许训练误差上升，以换取参数空间中的稳定、结构或更小目标风险**。Ridge 与 Lasso 使用相同平方损失，却通过不同几何改变解：

- Ridge 连续收缩所有受惩罚方向，适合预测稳定和保留成组信息；
- Lasso 对弱坐标执行阈值化，能够产生精确零系数，适合稀疏近似和测量成本探索；
- 两者都不自动证明被选变量有因果作用，也不自动授权拆除传感器。

## 贯穿案例：电机绕组热点温度软测量

大型电机绕组内部热点难以长期直接测量。试验台可以在少数测试循环中布置参考光纤温度计，生产环境则只能持续获得壳体温度、电流、功率、冷却液和振动等信号。团队希望训练软测量模型，在参考探头撤下后估计热点温度。

这个任务天然包含两类正则化动机：

1. 壳体温度 A/B、电流/功率、冷却液入口/出口高度相关，OLS 会在这些字段之间任意分摊系数；
2. 边缘设备希望减少采集通道，但“某次 Lasso 拟合为 0”并不足以证明拆掉该传感器不会伤害其他工况。

本章数据、设备参数、温度关系和阈值均为**教学合成**，不代表任何电机型号、绝缘等级或保护设定。真实热点保护必须遵循设备规范、传感器诊断和独立安全链；机器学习软测量不能替代硬件保护。

### 数据合同：一行是稳定窗口，不是一条瞬时采样

| 字段 | 单位 | 角色 | 主要风险 |
|---|---:|---|---|
| `surface_temp_a/b` | °C | 两只相邻壳体温度传感器 | 高度冗余；单个系数不稳定 |
| `phase_current` | A | 负载代理 | 与有功功率近共线 |
| `active_power` | kW | 负载代理 | 量程必须按训练集标准化 |
| `coolant_in/out` | °C | 冷却边界与温升 | 入口/出口同时含环境共模 |
| `speed` | rpm | 运行点 | 测试程序可能与负载混淆 |
| `vibration_rms` | mm/s | 机械状态代理 | 异常值和漂移需单独治理 |
| `hotspot_temperature` | °C | 训练期参考响应 | 上线后不可持续获得 |
| `run_id` | — | 独立测试循环 | 切分、交叉验证、重采样单位 |

相邻窗口共享热惯性，不能随机逐行打散。这里按完整 `run_id` 切出训练、验证、测试循环，并且只用训练集计算标准化参数；测试集只在最终比较中打开一次。

```python
import numpy as np

rng = np.random.default_rng(20260805)
n_runs, windows_per_run = 24, 14
run_id = np.repeat(np.arange(n_runs), windows_per_run)
n_observations = run_id.size

# 每个测试循环有自己的环境与冷却偏置，窗口内再扫描负载。
run_ambient = rng.normal(25.0, 3.2, n_runs)
run_cooling = rng.normal(0.0, 0.7, n_runs)
load = np.tile(np.linspace(0.15, 1.0, windows_per_run), n_runs)
load += rng.normal(0.0, 0.035, n_observations)
ambient = run_ambient[run_id] + rng.normal(0.0, 0.25, n_observations)
cooling = run_cooling[run_id] + rng.normal(0.0, 0.18, n_observations)

phase_current = 82.0 + 76.0 * load + rng.normal(0.0, 1.8, n_observations)
active_power = 10.5 + 0.083 * phase_current + rng.normal(0.0, 0.32, n_observations)
coolant_in = 20.0 + 0.55 * ambient - 1.7 * cooling + rng.normal(0.0, 0.35, n_observations)
coolant_out = coolant_in + 3.0 + 8.8 * load - 2.4 * cooling + rng.normal(0.0, 0.45, n_observations)
surface_temp_a = 28.0 + 34.0 * load + 0.48 * ambient - 5.2 * cooling + rng.normal(0.0, 0.55, n_observations)
surface_temp_b = surface_temp_a + rng.normal(0.0, 0.42, n_observations)
speed = 1320.0 + 310.0 * load + rng.normal(0.0, 16.0, n_observations)
vibration_rms = 1.1 + 0.55 * load + rng.normal(0.0, 0.16, n_observations)

raw_X = np.column_stack([
    surface_temp_a, surface_temp_b, phase_current, active_power,
    coolant_in, coolant_out, speed, vibration_rms,
])
feature_names = np.array([
    "surface_a", "surface_b", "current", "power",
    "coolant_in", "coolant_out", "speed", "vibration",
])
hotspot_temperature = (
    31.0 + 0.34 * surface_temp_a + 0.15 * phase_current
    + 0.48 * (coolant_out - coolant_in) + 0.004 * speed
    + rng.normal(0.0, 1.15, n_observations)
)

train_mask = run_id < 16
valid_mask = (run_id >= 16) & (run_id < 20)
test_mask = run_id >= 20
X_mean = raw_X[train_mask].mean(axis=0)
X_scale = raw_X[train_mask].std(axis=0, ddof=1)
y_mean = hotspot_temperature[train_mask].mean()

X_train = (raw_X[train_mask] - X_mean) / X_scale
X_valid = (raw_X[valid_mask] - X_mean) / X_scale
X_test = (raw_X[test_mask] - X_mean) / X_scale
y_train = hotspot_temperature[train_mask] - y_mean
y_valid = hotspot_temperature[valid_mask] - y_mean
y_test = hotspot_temperature[test_mask] - y_mean

assert X_train.shape == (224, 8)
assert set(run_id[train_mask]).isdisjoint(set(run_id[test_mask]))
assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-12)
assert np.allclose(X_train.std(axis=0, ddof=1), 1.0)
```

<MlQuiz kind="数据切分" question="相邻窗口共享热惯性。哪种验证方式最能避免同一测试循环的信息泄漏？" :options="['逐行随机切分', '按完整 run_id 分组，并在每个训练折内重新计算标准化参数', '先用全数据标准化再随机切分', '只比较训练 RSS']" :answer="1" explanation="独立信息单位是测试循环。逐行切分会让同一热过程同时出现在训练与验证；全数据标准化也会提前使用验证分布。" />

## 1. 正则化要解决的是放大，不只是“系数太大”

中心化后的 OLS 为

$$
\widehat\beta_{OLS}=V\,\operatorname{diag}\left(\frac{1}{s_j}\right)U^Ty,
\qquad X=U\operatorname{diag}(s_j)V^T.
$$

响应在左奇异方向 $u_j$ 上的一点噪声 $u_j^T\varepsilon$，会被除以 $s_j$ 后写进右奇异方向 $v_j$ 的系数。$s_j$ 很小时，问题不是某个原始系数字面上大，而是数据几乎没有能力辨认某个线性组合。

设 $X^TX$ 的特征值为 $s_j^2$。满列秩时

$$
\kappa_2(X^TX)=\frac{s_{\max}^2}{s_{\min}^2}=\kappa_2(X)^2.
$$

因此用正规方程求解还会把设计矩阵的条件数平方。正则化可以改善统计稳定性，但实现仍不应显式计算矩阵逆；数值稳定与统计正则化是两层问题。

### 标准化不是可选装饰

在惩罚 $\sum_j\beta_j^2$ 或 $\sum_j|\beta_j|$ 中，所有系数被放在同一单位下比较。若 `speed` 以 rpm、`power` 以 kW、振动以 mm/s 直接进入模型，那么同样的物理效应会因单位换算得到不同惩罚。

工程上必须冻结：

- 每列中心和尺度只由当前训练折计算；
- 常数列、近零方差列如何处理；
- 截距通常不惩罚；
- 类别哑变量、周期基函数或成组变量是否使用相同惩罚；
- 部署端必须复用训练产物中的同一组变换参数。

<MlQuiz kind="单位审计" question="把转速单位从 rpm 改成 krpm，但保持 λ 和其他流程不变，未经标准化的 Lasso 会怎样？" :options="['解必然不变', '惩罚的相对尺度改变，变量进入路径的顺序可能改变', '只改变截距', 'L1 自动识别物理单位']" :answer="1" explanation="系数会按单位反向缩放，而 L1 惩罚直接作用于系数绝对值；因此变量单位能改变优化问题。" />

## 2. Ridge：在平方损失上增加二次曲率

本章统一使用

$$
J_{ridge}(\beta;\lambda)
=\frac12\lVert y-X\beta\rVert_2^2
+\frac\lambda2\lVert\beta\rVert_2^2,
\qquad \lambda\ge 0.
$$

梯度和 Hessian 分别为

$$
\nabla J=(X^TX+\lambda I)\beta-X^Ty,
\qquad
\nabla^2J=X^TX+\lambda I.
$$

当 $\lambda>0$ 时，对任意非零 $a$：

$$
a^T(X^TX+\lambda I)a
=\lVert Xa\rVert_2^2+\lambda\lVert a\rVert_2^2>0.
$$

所以目标严格凸，即使 $p>n$ 或 $X$ 秩亏也有唯一受惩罚系数解：

$$
\widehat\beta_{ridge}(\lambda)
=(X^TX+\lambda I)^{-1}X^Ty.
$$

这里的逆用于数学表达；代码应通过线性方程、Cholesky、QR 或 SVD 求解。

### 截距为何不惩罚

若原始模型为 $\beta_0+x^T\beta$，惩罚截距会让整体响应均值被向 0 拉回，而且结果依赖于响应量纲的零点。常见做法有两种：

1. 训练折内中心化 $X$ 与 $y$，只拟合受惩罚斜率，最后恢复截距；
2. 在惩罚矩阵中把截距位置设为 0：$P=\operatorname{diag}(0,1,\ldots,1)$。

本章代码使用第一种。若业务确实拥有截距先验，惩罚它并非数学错误，但必须作为显式建模决定记录。

### 增广最小二乘避免手写正规方程

Ridge 等价于

$$
\left\lVert
\begin{bmatrix}y\\0\end{bmatrix}
-
\begin{bmatrix}X\\\sqrt\lambda I\end{bmatrix}\beta
\right\rVert_2^2.
$$

于是可把它交给稳定的最小二乘求解器：

```python
def fit_ridge_qr(design, target, lambda_value):
    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative")
    n_features = design.shape[1]
    if lambda_value == 0:
        return np.linalg.lstsq(design, target, rcond=None)[0]
    augmented_X = np.vstack([design, np.sqrt(lambda_value) * np.eye(n_features)])
    augmented_y = np.concatenate([target, np.zeros(n_features)])
    return np.linalg.lstsq(augmented_X, augmented_y, rcond=None)[0]

def mse(target, prediction):
    return np.mean((target - prediction) ** 2)

beta_ols = fit_ridge_qr(X_train, y_train, 0.0)
beta_ridge_demo = fit_ridge_qr(X_train, y_train, 12.0)
gram_condition = np.linalg.cond(X_train.T @ X_train)
regularized_condition = np.linalg.cond(
    X_train.T @ X_train + 12.0 * np.eye(X_train.shape[1])
)

assert regularized_condition < gram_condition
assert np.linalg.norm(beta_ridge_demo) < np.linalg.norm(beta_ols)
print("Gram condition:", round(gram_condition, 1), "->", round(regularized_condition, 1))
print("validation MSE, OLS/Ridge:", round(mse(y_valid, X_valid @ beta_ols), 3), round(mse(y_valid, X_valid @ beta_ridge_demo), 3))
```

<div class="ml-stop"><strong>停下来推一推 1</strong>证明增广设计的 Gram 矩阵正好是 $X^TX+\lambda I$。如果截距列不应受惩罚，增广矩阵最后一部分应怎样改？</div>

<MlQuiz kind="解的性质" question="当 X 秩亏且 λ>0 时，Ridge 为何仍有唯一解？" :options="['因为 y 必须为 0', 'XᵀX+λI 正定，目标严格凸', '因为 L2 会删除重复列', '因为验证集补齐了秩']" :answer="1" explanation="任意非零方向的二次型都等于 ‖Xa‖²+λ‖a‖²>0，因此 Hessian 正定。" />

## 3. SVD 揭示 Ridge 不是统一打折

将紧致 SVD $X=USV^T$ 代入闭式解：

$$
\begin{aligned}
\widehat\beta_{ridge}
&=V(S^2+\lambda I)^{-1}SU^Ty\\
&=\sum_j\frac{s_j}{s_j^2+\lambda}(u_j^Ty)v_j.
\end{aligned}
$$

与 OLS 的方向系数 $(u_j^Ty)/s_j$ 相比，Ridge 乘上

$$
h_j(\lambda)=\frac{s_j^2}{s_j^2+\lambda}.
$$

- $s_j^2\gg\lambda$：数据强方向几乎保留；
- $s_j^2\approx\lambda$：保留约一半；
- $s_j^2\ll\lambda$：弱方向接近关闭。

拟合值为

$$
\widehat y_\lambda
=U\operatorname{diag}(h_j)U^Ty.
$$

这说明收缩发生在奇异方向，不是每个原始字段独立乘同一个系数。两个冗余温度传感器的“和方向”可能被保留，“差方向”则被强烈压制。

<RidgeSpectralShrinkageLab />

### 有效自由度：模型并没有突然少一列

Ridge 帽子矩阵为

$$
H_\lambda=X(X^TX+\lambda I)^{-1}X^T
=U\operatorname{diag}(h_j)U^T.
$$

定义

$$
\operatorname{df}_{eff}(\lambda)
=\operatorname{tr}(H_\lambda)
=\sum_j\frac{s_j^2}{s_j^2+\lambda}.
$$

它可取非整数。$\lambda=0$ 且满列秩时为 $p$；$\lambda\to\infty$ 时趋近 0。Ridge 通常没有删除列，但每条可拟合方向贡献了 0 到 1 之间的自由度。

```python
U_train, singular_values, Vt_train = np.linalg.svd(X_train, full_matrices=False)
lambda_demo = 12.0
spectral_beta = Vt_train.T @ (
    singular_values / (singular_values ** 2 + lambda_demo) * (U_train.T @ y_train)
)
retention = singular_values ** 2 / (singular_values ** 2 + lambda_demo)
effective_dof = retention.sum()

assert np.allclose(spectral_beta, beta_ridge_demo, atol=1e-10)
assert np.all((retention > 0) & (retention < 1))
assert effective_dof < X_train.shape[1]
print("singular values:", np.round(singular_values, 3))
print("spectral retention:", np.round(retention, 3), "effective df:", round(effective_dof, 3))
```

<MlQuiz kind="谱判断" question="某奇异方向满足 sⱼ²=λ。该方向在 Ridge 拟合值中的保留率是多少？" :options="['0', '1/2', '1', '取决于 uⱼᵀy 的符号']" :answer="1" explanation="hⱼ=sⱼ²/(sⱼ²+λ)=1/2。响应投影决定幅度和符号，不改变保留比例。" />

## 4. Ridge 怎样交换偏差与方差

在固定设计线性模型 $y=X\beta+\varepsilon$、$\mathbb E[\varepsilon\mid X]=0$、$\operatorname{Cov}(\varepsilon\mid X)=\sigma^2I$ 下，记

$$
A_\lambda=(X^TX+\lambda I)^{-1}X^TX.
$$

则

$$
\mathbb E[\widehat\beta_\lambda\mid X]=A_\lambda\beta,
$$

偏差为

$$
\operatorname{Bias}(\widehat\beta_\lambda\mid X)
=-\lambda(X^TX+\lambda I)^{-1}\beta.
$$

协方差为

$$
\operatorname{Cov}(\widehat\beta_\lambda\mid X)
=\sigma^2(X^TX+\lambda I)^{-1}X^TX(X^TX+\lambda I)^{-1}.
$$

在第 $j$ 个奇异方向，OLS 方差为 $\sigma^2/s_j^2$，Ridge 方差为

$$
\sigma^2\frac{s_j^2}{(s_j^2+\lambda)^2}.
$$

弱方向的方差下降最多，但会引入收缩偏差。测试误差是否下降取决于真实信号、噪声、目标输入分布和 $\lambda$；“系数范数更小”本身不是验收指标。

```python
beta_fixed = np.array([1.3, 1.3, 0.85, 0.25, -0.45, 0.65, 0.30, 0.0])
sigma_fixed = 1.1
n_draws = 1400
noise_draws = rng.normal(0.0, sigma_fixed, size=(X_train.shape[0], n_draws))
simulated_targets = X_train @ beta_fixed[:, None] + noise_draws
gram = X_train.T @ X_train
regularized_gram = gram + lambda_demo * np.eye(gram.shape[0])
ridge_inverse = np.linalg.solve(regularized_gram, np.eye(gram.shape[0]))
ridge_operator = ridge_inverse @ X_train.T
simulated_betas = ridge_operator @ simulated_targets

expected_beta = ridge_inverse @ gram @ beta_fixed
theory_covariance = sigma_fixed ** 2 * ridge_inverse @ gram @ ridge_inverse
empirical_mean = simulated_betas.mean(axis=1)
empirical_variance = simulated_betas.var(axis=1, ddof=1)

assert np.allclose(empirical_mean, expected_beta, atol=0.035)
assert np.allclose(empirical_variance, np.diag(theory_covariance), rtol=0.16, atol=0.002)
print("largest absolute bias:", round(np.max(np.abs(expected_beta - beta_fixed)), 3))
print("mean coefficient variance:", round(empirical_variance.mean(), 4))
```

<div class="ml-stop"><strong>停下来推一推 2</strong>当 $\lambda\to\infty$，系数方差趋近什么？若真实 $\beta\ne0$，偏差趋近什么？这解释了为什么验证误差通常不是随 $\lambda$ 单调下降。</div>

<MlQuiz kind="偏差方差" question="把 λ 增大到极大时，下列哪项最准确？" :options="['系数方差趋近 0，但对非零真系数的偏差增大', '偏差与方差都必然为 0', '训练 RSS 必然下降', 'Ridge 变成无偏估计']" :answer="0" explanation="极强惩罚把斜率推向 0，降低抽样波动但牺牲非零信号；训练 RSS 相对 OLS 不会更低。" />

## 5. 惩罚式与约束式是同一条边界的两种参数化

Ridge 也常写成

$$
\min_\beta\frac12\lVert y-X\beta\rVert_2^2
\quad\text{s.t.}\quad
\lVert\beta\rVert_2^2\le t.
$$

拉格朗日函数为

$$
\mathcal L(\beta,\nu)
=\frac12\lVert y-X\beta\rVert_2^2
+\frac\nu2(\lVert\beta\rVert_2^2-t),
\qquad \nu\ge0.
$$

KKT 条件包含：

$$
\begin{cases}
(X^TX+\nu I)\beta=X^Ty, & \text{驻点}\\
\lVert\beta\rVert_2^2\le t, & \text{原始可行}\\
\nu\ge0, & \text{对偶可行}\\
\nu(\lVert\beta\rVert_2^2-t)=0. & \text{互补松弛}
\end{cases}
$$

约束活跃时，$\nu$ 就扮演惩罚式中的 $\lambda$。但 $t$ 与 $\lambda$ 不是通用单位换算：映射依赖 $X$、$y$、损失缩放和是否惩罚截距。若 $t$ 足以包含 OLS 解，约束不活跃，$\nu=0$。

同理 Lasso 可以写成 $\lVert\beta\rVert_1\le t$。L2 可行集是圆球，L1 可行集是带坐标轴拐角的交叉多面体；这一几何差异将产生完全不同的零系数行为。

<MlQuiz kind="KKT 映射" question="约束半径 t 已经包含 OLS 解时，对应最优乘子通常是什么？" :options="['ν=0', 'ν=∞', 'ν必须为负', 'ν等于样本数']" :answer="0" explanation="约束不活跃时由互补松弛得到乘子为 0，惩罚式退化为未正则化最小二乘。" />

## 6. Gaussian 先验给出 Ridge 的 MAP，而不是自动给出可靠先验

设观测模型为

$$
y\mid X,\beta\sim\mathcal N(X\beta,\sigma^2I),
$$

受惩罚斜率的先验为

$$
\beta\sim\mathcal N(0,\tau^2I).
$$

忽略与 $\beta$ 无关的常数，负对数后验为

$$
-\log p(\beta\mid X,y)
=\frac{1}{2\sigma^2}\lVert y-X\beta\rVert_2^2
+\frac{1}{2\tau^2}\lVert\beta\rVert_2^2+C.
$$

乘以 $\sigma^2$ 后，MAP 等价于 Ridge，且在本章目标约定下

$$
\lambda=\frac{\sigma^2}{\tau^2}.
$$

这条关系说明 $\lambda$ 是“观测噪声相对于先验尺度”的比值。但它不意味着任意交叉验证得到的 $\lambda$ 都来自工程上可信的 Gaussian 先验，也不意味着 MAP 已经积分了参数不确定性。

在**线性 Gaussian likelihood + Gaussian prior** 的特殊共轭模型中，后验本身是 Gaussian，MAP 与后验均值恰好重合；离开这个特例，两者通常不同。要报告 Bayesian 区间，还需保留后验协方差并说明超参数来源，而不是只输出 Ridge 点估计。

<MlQuiz kind="Bayes 边界" question="在本章目标约定下，观测噪声 σ² 增大而先验方差 τ² 固定时，λ 应怎样变化？" :options="['增大', '减小', '保持 0', '只改变截距']" :answer="0" explanation="λ=σ²/τ²；数据更噪而先验尺度不变时，MAP 相对更依赖收缩先验。" />

## 7. Lasso：不可导点不是缺陷，而是精确稀疏的来源

Lasso 目标写作

$$
J_{lasso}(\beta;\lambda)
=\frac12\lVert y-X\beta\rVert_2^2
+\lambda\lVert\beta\rVert_1,
\qquad
\lVert\beta\rVert_1=\sum_j|\beta_j|.
$$

它是凸问题，但 $|\beta_j|$ 在 0 处不可导。0 点的次梯度集合为

$$
\partial|\beta_j|
=
\begin{cases}
\{+1\}, & \beta_j>0,\\
[-1,1], & \beta_j=0,\\
\{-1\}, & \beta_j<0.
\end{cases}
$$

记残差 $r=y-X\widehat\beta$。最优 KKT 条件为

$$
X^Tr\in\lambda\,\partial\lVert\widehat\beta\rVert_1.
$$

逐坐标展开：

$$
\begin{cases}
x_j^Tr=\lambda\operatorname{sign}(\widehat\beta_j), & \widehat\beta_j\ne0,\\
|x_j^Tr|\le\lambda, & \widehat\beta_j=0.
\end{cases}
$$

因此，只要某列与当前残差的相关证据没有越过 $\lambda$，0 就能满足最优性条件。Ridge 的导数 $\lambda\beta_j$ 在 0 点也是 0，没有一整段次梯度去吸收小证据，所以它通常只连续接近 0。

### 正交设计下的软阈值解析解

若 $X^TX=I$，令 $z=X^Ty$，目标可分解为

$$
J(\beta)=\sum_j\left[\frac12(\beta_j-z_j)^2+\lambda|\beta_j|\right]+C.
$$

逐坐标解为软阈值：

$$
\widehat\beta_j
=S_\lambda(z_j)
=\operatorname{sign}(z_j)(|z_j|-\lambda)_+.
$$

它分三段：$z_j>\lambda$ 时减去 $\lambda$；$z_j<-\lambda$ 时加上 $\lambda$；$|z_j|\le\lambda$ 时精确为 0。

<LassoGeometryLab />

<div class="ml-stop"><strong>停下来推一推 3</strong>对一维目标 $\frac12(\beta-z)^2+\lambda|\beta|$，分别假设 $\beta>0$、$\beta<0$、$\beta=0$ 写出最优条件，并拼出软阈值函数。</div>

<MlQuiz kind="KKT 判断" question="某标准化列当前满足 |xⱼᵀr|=0.7λ。Lasso 在 βⱼ=0 处是否可能满足最优条件？" :options="['可以，因为相关证据位于 [−λ,λ] 内', '不可以，必须严格等于 λ', '只有 Ridge 可以', '必须先令 λ=0']" :answer="0" explanation="零坐标的次梯度允许整个 [−1,1] 区间，因此 |xⱼᵀr|≤λ 正是非活跃坐标条件。" />

### Laplace 先验给出 Lasso MAP

若每个系数独立服从零中心 Laplace 先验

$$
p(\beta_j)=\frac{1}{2b}\exp\left(-\frac{|\beta_j|}{b}\right),
$$

则负对数先验与 $\lVert\beta\rVert_1/b$ 成正比。在 Gaussian 观测模型下，MAP 目标就是 Lasso，且在本章约定中 $\lambda=\sigma^2/b$。

但“MAP 系数为 0”不等于后验认为该变量绝对无关；它是后验众数在不可导惩罚下的点估计结果。Laplace 后验均值一般不会拥有同样的精确零结构。

## 8. 从软阈值到一般设计：坐标下降与近端梯度

### 坐标下降：把其他列暂时放进偏残差

固定除 $\beta_j$ 外的所有坐标，定义

$$
r_{-j}=y-\sum_{k\ne j}x_k\beta_k.
$$

关于 $\beta_j$ 的子问题为

$$
\frac12\lVert r_{-j}-x_j\beta_j\rVert_2^2+\lambda|\beta_j|,
$$

所以更新为

$$
\beta_j\leftarrow
\frac{S_\lambda(x_j^Tr_{-j})}{\lVert x_j\rVert_2^2}.
$$

循环更新坐标会单调降低凸目标。相关列会相互改变偏残差，因此“先进入路径的字段”可能吸收另一列的信号；活跃集合不一定随 $\lambda$ 单调扩大。

```python
def soft_threshold(value, threshold):
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)

def lasso_coordinate_descent(design, target, lambda_value, beta_init=None, max_iter=8000, tol=1e-10):
    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative")
    beta = np.zeros(design.shape[1]) if beta_init is None else beta_init.copy()
    residual = target - design @ beta
    column_norms = np.sum(design ** 2, axis=0)
    for iteration in range(1, max_iter + 1):
        previous = beta.copy()
        for j in range(design.shape[1]):
            residual += design[:, j] * beta[j]
            partial_correlation = design[:, j] @ residual
            beta[j] = soft_threshold(partial_correlation, lambda_value) / column_norms[j]
            residual -= design[:, j] * beta[j]
        if np.max(np.abs(beta - previous)) < tol:
            return beta, iteration
    raise RuntimeError("coordinate descent did not converge")

def lasso_kkt_violation(design, target, beta, lambda_value, active_tol=1e-8):
    gradient = design.T @ (design @ beta - target)
    active = np.abs(beta) > active_tol
    violations = np.zeros_like(beta)
    violations[active] = np.abs(gradient[active] + lambda_value * np.sign(beta[active]))
    violations[~active] = np.maximum(np.abs(gradient[~active]) - lambda_value, 0.0)
    return violations.max()

lambda_max = np.max(np.abs(X_train.T @ y_train))
lambda_lasso_demo = 0.08 * lambda_max
beta_lasso_demo, lasso_iterations = lasso_coordinate_descent(
    X_train, y_train, lambda_lasso_demo
)
kkt_error = lasso_kkt_violation(
    X_train, y_train, beta_lasso_demo, lambda_lasso_demo
)

assert kkt_error < 1e-7
assert np.count_nonzero(np.abs(beta_lasso_demo) > 1e-8) < X_train.shape[1]
print("lambda_max/demo:", round(lambda_max, 3), round(lambda_lasso_demo, 3))
print("active/iterations/KKT:", np.count_nonzero(beta_lasso_demo), lasso_iterations, f"{kkt_error:.2e}")
```

当所有列与响应都中心化时，$\beta=0$ 成为最优解的最小阈值是

$$
\lambda_{max}=\lVert X^Ty\rVert_\infty.
$$

因此正则化路径通常从 $\lambda_{max}$ 开始按对数网格向下走，并用上一个较大 $\lambda$ 的解 warm start 下一个问题。

### 近端梯度：梯度一步，再执行软阈值

把目标拆为光滑项 $f(\beta)=\frac12\lVert y-X\beta\rVert^2$ 和非光滑项 $g(\beta)=\lambda\lVert\beta\rVert_1$。近端梯度更新为

$$
\begin{aligned}
z^{(t)}&=\beta^{(t)}-\eta X^T(X\beta^{(t)}-y),\\
\beta^{(t+1)}&=S_{\eta\lambda}(z^{(t)}).
\end{aligned}
$$

若固定步长满足 $0<\eta\le1/\lVert X\rVert_2^2$，基本算法具有标准收敛保证。坐标下降适合普通稠密表格；近端方法更容易扩展到矩阵算子和其他可分非光滑惩罚。

<MlQuiz kind="算法验收" question="坐标下降返回一个看似稀疏的系数向量。最直接的数值验收是什么？" :options="['只看迭代次数', '检查活跃坐标等式和非活跃坐标不等式的最大 KKT 违反量', '要求训练 RSS 为 0', '把所有小数截断为 0']" :answer="1" explanation="KKT 条件直接刻画凸问题最优性；稀疏外观或迭代上限都不能替代最优性残差。" />

## 9. 目标函数缩放决定 λ 的数值含义

常见库可能使用以下三种写法：

$$
\frac12\operatorname{RSS}+\lambda\lVert\beta\rVert_1,
$$

$$
\frac{1}{2n}\operatorname{RSS}+\alpha\lVert\beta\rVert_1,
$$

$$
\operatorname{MSE}+\alpha\lVert\beta\rVert_1.
$$

它们可以通过换算超参数对应，但同一个数字 `0.1` 绝不代表同一强度。Ridge 中也常见 $\lambda\lVert\beta\rVert^2$ 与 $\lambda\lVert\beta\rVert^2/2$ 的差异。

模型登记至少保存：

- 完整目标函数及 $1/2$、$1/n$ 因子；
- 标准化定义和自由度约定；
- 截距与各特征惩罚权重；
- λ 网格如何相对 `lambda_max` 生成；
- 求解容差、最大迭代、KKT 或 dual gap；
- 软件和版本。

否则跨语言复现时即使都写着 “Lasso λ=0.3”，也可能得到完全不同的模型。

<MlQuiz kind="协议换算" question="实现 A 最小化 0.5·RSS+λ‖β‖₁，实现 B 最小化 RSS/(2n)+α‖β‖₁。要得到同一解，通常应怎样对应？" :options="['λ=nα', 'λ=α/n', 'λ=α²', '二者永远不能对应']" :answer="0" explanation="把 B 的目标整体乘 n 得到 0.5·RSS+nα‖β‖₁，所以 A 中 λ=nα。" />

## 10. 选 λ：验证预测，也验证模型行为

正则化参数是模型选择的一部分，不能用测试集调。工业流程建议：

1. 冻结完整测试循环；
2. 在开发集内部按 `run_id` 做组交叉验证或按时间滚动验证；
3. 每个训练折内重新拟合标准化器；
4. Ridge 使用对数 λ 网格；Lasso 从各训练折的 $\lambda_{max}$ 向下生成相对网格；
5. 记录每折预测损失、最坏组误差、Ridge 有效自由度、Lasso 活跃数；
6. 若多个参数统计上接近，可使用 one-standard-error 规则选择更强正则；
7. 只在方案冻结后重训，并打开测试集一次。

下面为了保持代码透明，使用预留的四个完整循环作验证；正式小样本项目应改为组交叉验证。

```python
ridge_grid = np.concatenate([[0.0], np.geomspace(0.05, 250.0, 36)])
ridge_validation_mse = []
ridge_betas = []
for lambda_value in ridge_grid:
    beta = fit_ridge_qr(X_train, y_train, lambda_value)
    ridge_betas.append(beta)
    ridge_validation_mse.append(mse(y_valid, X_valid @ beta))

lasso_grid = np.geomspace(lambda_max, 0.01 * lambda_max, 42)
lasso_validation_mse = []
lasso_betas = []
warm_beta = np.zeros(X_train.shape[1])
for lambda_value in lasso_grid:
    warm_beta, _ = lasso_coordinate_descent(
        X_train, y_train, lambda_value, beta_init=warm_beta
    )
    lasso_betas.append(warm_beta.copy())
    lasso_validation_mse.append(mse(y_valid, X_valid @ warm_beta))

ridge_choice = int(np.argmin(ridge_validation_mse))
lasso_choice = int(np.argmin(lasso_validation_mse))
selected_ridge_lambda = ridge_grid[ridge_choice]
selected_lasso_lambda = lasso_grid[lasso_choice]
selected_ridge_beta = ridge_betas[ridge_choice]
selected_lasso_beta = lasso_betas[lasso_choice]

test_results = {
    "OLS": mse(y_test, X_test @ beta_ols),
    "Ridge": mse(y_test, X_test @ selected_ridge_beta),
    "Lasso": mse(y_test, X_test @ selected_lasso_beta),
}
assert selected_ridge_lambda >= 0
assert selected_lasso_lambda > 0
assert all(np.isfinite(value) for value in test_results.values())
print("selected Ridge/Lasso lambda:", round(selected_ridge_lambda, 4), round(selected_lasso_lambda, 4))
print("test MSE:", {name: round(value, 3) for name, value in test_results.items()})
print("Lasso active:", feature_names[np.abs(selected_lasso_beta) > 1e-8].tolist())
```

### 最低验证误差不是唯一答案

软测量模型还应联合观察：

| 维度 | Ridge | Lasso | 不能据此声称 |
|---|---|---|---|
| 测试循环 MSE/MAE | 预测是否稳定 | 稀疏后预测是否仍可接受 | 未来所有工况都覆盖 |
| 系数重采样波动 | 冗余方向是否被抑制 | 非零幅度是否稳定 | 系数是因果效应 |
| 活跃集合频率 | 通常全部保留 | 字段是否反复入选 | 低频字段可以安全拆除 |
| 有效自由度/活跃数 | 连续复杂度 | 离散稀疏度 | 两者可直接等同 |
| 采集成本与失效模式 | 保留冗余可能更可靠 | 有机会减少通道 | 软件选择可绕过安全评审 |

## 11. 稀疏不等于稳定选择，更不等于因果发现

两只壳体温度传感器高度相关时，它们对预测提供近似可替代信息。Lasso 的 L1 球有拐角，确实倾向于选择坐标轴；但究竟选择 A 还是 B，可能由很小的采样扰动决定。

下面按完整训练循环 bootstrap，比较 Ridge 系数标准差与 Lasso 入选频率。标准化器和已选 λ 保持冻结，目的是隔离“训练样本循环变化”带来的拟合波动。

```python
train_run_id = run_id[train_mask]
unique_train_runs = np.unique(train_run_id)
n_bootstrap = 160
ridge_bootstrap = []
lasso_bootstrap = []

for _ in range(n_bootstrap):
    sampled_runs = rng.choice(
        unique_train_runs, size=unique_train_runs.size, replace=True
    )
    sampled_rows = np.concatenate([
        np.flatnonzero(train_run_id == group) for group in sampled_runs
    ])
    boot_X = X_train[sampled_rows]
    boot_y = y_train[sampled_rows]
    ridge_bootstrap.append(
        fit_ridge_qr(boot_X, boot_y, selected_ridge_lambda)
    )
    boot_lasso, _ = lasso_coordinate_descent(
        boot_X, boot_y, selected_lasso_lambda, max_iter=30000, tol=1e-8
    )
    assert lasso_kkt_violation(
        boot_X, boot_y, boot_lasso, selected_lasso_lambda
    ) < 2e-5
    lasso_bootstrap.append(boot_lasso)

ridge_bootstrap = np.asarray(ridge_bootstrap)
lasso_bootstrap = np.asarray(lasso_bootstrap)
ridge_coefficient_sd = ridge_bootstrap.std(axis=0, ddof=1)
lasso_selection_frequency = np.mean(
    np.abs(lasso_bootstrap) > 1e-7, axis=0
)

assert ridge_bootstrap.shape == (n_bootstrap, X_train.shape[1])
assert np.all(
    (lasso_selection_frequency >= 0) & (lasso_selection_frequency <= 1)
)
for name, ridge_sd, selection_frequency in zip(
    feature_names, ridge_coefficient_sd, lasso_selection_frequency
):
    print(
        f"{name:12s} ridge_sd={ridge_sd:6.3f} "
        f"lasso_select={selection_frequency:5.1%}"
    )
```

如果两个温度字段各自只有约 50% 入选，而“至少一个被选”接近 100%，正确结论是这组信息有用、单列身份不稳定；不能挑入选频率略高的一只就宣布另一只冗余。

### 选后再做 OLS 也没有免费午餐

有人会在 Lasso 选择活跃集后，用同一份数据在这些列上重新做 OLS，希望消除收缩偏差。预测上这有时有用，但会引入新的方差，并且普通 OLS 标准误忽略了“活跃集也是数据选择结果”。

若需要解释或区间，应考虑：独立选择/推断样本、嵌套交叉验证、选择后推断方法、稳定性选择或面向预测的 bootstrap。最简单且诚实的做法，常常是把 Lasso 定位为预测/筛选器，不把普通系数表包装成确认性科学结论。

<MlQuiz kind="案例审计" question="壳体温度 A/B 高度相关，100 次组 bootstrap 中 A 入选 54 次、B 入选 48 次、至少一个入选 99 次。最稳妥的解释是什么？" :options="['A 已被证明是唯一原因', '这组温度信息稳定有用，但单个传感器身份不稳定', 'B 可以立即拆除', '两列都没有预测价值']" :answer="1" explanation="相关变量会竞争同一信号。组层面的稳定不等于单列稳定，更不能直接推出因果或硬件删除决定。" />

## 12. Ridge、Lasso 还是 Elastic Net

选择应由目标和失效模式驱动：

| 条件 | 首选起点 | 原因 |
|---|---|---|
| $p\gg n$、大量冗余连续信号、预测优先 | Ridge | 稳定保留成组信息，解唯一 |
| 真正希望稀疏近似、采集成本明确 | Lasso | 能产生精确 0，但必须做稳定性审计 |
| 既要稀疏又希望相关组共同保留 | Elastic Net | L1 负责稀疏，L2 缓和相关变量竞争 |
| 已知传感器必须保留或受不同先验约束 | 加权/分组惩罚 | 业务结构进入惩罚矩阵，而非事后篡改 |
| 变量以自然组出现，如一组 Fourier 对 | Group Lasso 等结构惩罚 | 按组进入/退出比逐列更符合语义 |

Elastic Net 的基本目标为

$$
\frac12\lVert y-X\beta\rVert^2
+\lambda_1\lVert\beta\rVert_1
+\frac{\lambda_2}{2}\lVert\beta\rVert_2^2.
$$

本章不展开其求解；关键是它不是“参数越多越高级”，而是在相关变量竞争明显、纯 Lasso 选择不稳定时提供一条有依据的折中。

## 数值验证

### 四个必须独立通过的不变量

本章实现应至少验证：

1. 增广最小二乘解、线性方程解和 SVD 公式在数值容差内一致；
2. Ridge 的每个谱保留率位于 $(0,1)$，有效自由度随 $\lambda$ 增大而不增；
3. 正交设计的一般坐标下降与软阈值解析解一致；
4. Lasso 输出满足活跃/非活跃坐标的 KKT 条件，而不是只看“迭代结束”。

```python
# 1. Ridge 三种表达一致。
direct_ridge = np.linalg.solve(
    X_train.T @ X_train + lambda_demo * np.eye(X_train.shape[1]),
    X_train.T @ y_train,
)
assert np.allclose(direct_ridge, beta_ridge_demo, atol=1e-10)
assert np.allclose(direct_ridge, spectral_beta, atol=1e-10)

# 2. 有效自由度随 lambda 单调不增。
dof_path = np.array([
    np.sum(singular_values ** 2 / (singular_values ** 2 + value))
    for value in np.geomspace(1e-4, 1e4, 120)
])
assert np.all(np.diff(dof_path) < 0)

# 3. 正交设计上的 Lasso 等于软阈值。
orthogonal_X, _ = np.linalg.qr(rng.normal(size=(80, 6)))
orthogonal_y = rng.normal(size=80)
orthogonal_lambda = 0.18
analytic_lasso = soft_threshold(
    orthogonal_X.T @ orthogonal_y, orthogonal_lambda
)
numeric_lasso, _ = lasso_coordinate_descent(
    orthogonal_X, orthogonal_y, orthogonal_lambda
)
assert np.allclose(analytic_lasso, numeric_lasso, atol=1e-9)

# 4. 选择后的 Lasso 解满足 KKT。
assert lasso_kkt_violation(
    X_train, y_train, selected_lasso_beta, selected_lasso_lambda
) < 1e-7
print("Ridge equivalence, df monotonicity, soft threshold and KKT: passed")
```

数值验证不证明业务假设成立。它只能证明“实现了声称的优化问题”；训练/验证协议、传感器漂移、目标定义和覆盖域仍需独立审计。

## 复杂度与工程边界

设 $X\in\mathbb R^{n\times p}$，以下以稠密矩阵为基准：

| 操作 | 典型时间 | 存储/复用 | 工程关注点 |
|---|---:|---:|---|
| Ridge 单次 QR/SVD | $O(np^2+p^3)$ | $O(np)$ | 不显式求逆；同一设计可复用分解 |
| Ridge 整条路径（已有 SVD） | 初始 $O(np^2+p^3)$，每个 λ 约 $O(p)$–$O(np)$ | 保存 $U^Ty,s,V$ | $n,p$ 很大时不能保存完整 $U$ |
| Ridge Cholesky | 形成 Gram $O(np^2)$，分解 $O(p^3)$ | $O(p^2)$ | 条件性较差时 QR/SVD 更稳 |
| Lasso 坐标下降一轮 | $O(np)$ | 残差缓存 $O(n)$ | 标准化、warm start、活跃集加速 |
| 近端梯度一步 | $O(np)$ | 梯度和参数 | 步长由 $\lVert X\rVert_2^2$ 控制 |
| λ 路径与组交叉验证 | 约“折数 × λ 数 × 拟合成本” | 可按折并行 | 预处理必须在折内完成 |
| 组 bootstrap 稳定性 | $B$ 次拟合 | 可流式汇总 | 重采样单位必须是 `run_id` |

### 求解停止条件

- Ridge 直接法应记录线性系统相对残差与分解告警；
- Lasso 不应只使用“系数变化很小”，还应记录最大 KKT 违反量或 duality gap；
- 强相关下坐标下降可能需要更多轮，达到 `max_iter` 不是收敛；
- 极小 λ 接近 OLS，病态方向与迭代成本都会重新出现；
- 稀疏矩阵应使用保持稀疏性的算子，避免无意转成稠密 Gram。

### 上线前的失效模式清单

- **缩放泄漏**：用全数据均值/方差，使验证集提前参与训练；
- **λ 协议漂移**：训练库和部署重写使用不同的 $1/n$ 或 $1/2$ 约定；
- **截距被误罚**：响应零点改变就改变整体预测基线；
- **相关变量任意竞争**：Lasso 每次选不同传感器，却只汇报一次活跃集；
- **异常点支配**：L1 系数惩罚不等于 L1 残差损失，Lasso 仍使用平方残差并会受响应异常影响；
- **动态泄漏**：随机窗口切分把同一热过程同时放进训练与测试；
- **外推伪稳定**：系数变小不保证超出测试转速/冷却范围后物理合理；
- **保护链越权**：软测量模型不能替代独立超温保护；
- **选择后过度解释**：非零系数、入选频率和因果作用是三种不同命题；
- **漂移后路径失效**：传感器换型、量程或校准改变后，旧标准化器与 λ 都需复验。

<MlQuiz kind="失效诊断" question="Lasso 使用 L1 系数惩罚，因此团队声称它对热点温度标签中的极端异常值天然稳健。这个结论哪里错？" :options="['Lasso 的数据拟合项仍是平方残差，响应异常值仍可能支配目标', 'L1 系数惩罚会自动删除异常行', '只有 λ=0 才受异常值影响', '标准化可以证明没有异常值']" :answer="0" explanation="L1 作用于系数而非残差。若要改变对响应异常的敏感度，需要审计观测模型或使用 Huber、绝对损失等稳健拟合。" />

## 13. 章节练习：交付一份可复核的软测量正则化报告

### 推导练习

1. 从 Ridge 一阶条件推导闭式解，并证明 $\lambda>0$ 时 Hessian 正定。
2. 从 SVD 推导系数方向因子 $s_j/(s_j^2+\lambda)$、拟合值保留率 $s_j^2/(s_j^2+\lambda)$ 和有效自由度。
3. 推导 Ridge 条件偏差与协方差，解释为何弱方向的方差下降最明显。
4. 写出 L2 约束式的完整 KKT 条件，解释何时乘子为 0。
5. 从 Gaussian/Laplace 先验分别推导 Ridge/Lasso MAP，并标出目标函数缩放对 λ 的影响。
6. 从一维次梯度条件推导软阈值，并证明 $\lambda_{max}=\lVert X^Ty\rVert_\infty$。
7. 推导一般坐标下降更新，说明为什么分母是 $\lVert x_j\rVert^2$。

### 实现练习

把验证过程升级为 4 折 `run_id` 组交叉验证。每个折内重新拟合标准化器，同时生成 Ridge 与 Lasso 路径；输出平均误差、标准误、one-standard-error 选择、最坏循环误差、有效自由度/活跃数、KKT 违反量和运行时间。

```python
# TODO：由学习者完成；本代码块不参与自动执行。
def grouped_regularization_report(raw_X, y, groups, ridge_ratios, lasso_ratios):
    # 1. 生成互斥 group folds，任何 group 不得跨训练/验证
    # 2. 每折只用训练部分拟合中心与尺度
    # 3. Lasso 网格按该折 lambda_max 的比例生成
    # 4. 返回逐折预测误差、最坏组误差、df/active count、KKT 与耗时
    # 5. 使用 one-standard-error 规则选择更强正则，并保留全部审计轨迹
    raise NotImplementedError
```

### 稳定性与部署练习

对 160 次组 bootstrap 输出：

- 每个 Ridge 标准化系数的中位数、5%–95% 分位数；
- 每个 Lasso 字段的入选频率、符号一致率；
- `surface_a/b` 至少一个入选的组级频率；
- 删除一只壳体传感器后的独立测试误差与最坏循环误差；
- 传感器缺失、常数漂移和比例漂移压力测试；
- 一页模型卡：目标函数、标准化、λ 约定、训练范围、保护边界和回退策略。

报告必须明确区分“预测可接受”“字段稳定入选”“硬件允许移除”。前两项统计证据不能代替第三项的系统安全验证。

### 反例练习

构造两个数据集：

1. 两列完全相同，证明 Ridge 给出唯一的对称分配，而 Lasso 可能存在多个等价系数解；
2. 两列高度相关但标签信号相同，展示不同 bootstrap 样本中 Lasso 在两列之间切换，而预测近似不变。

解释“凸目标”为什么不自动保证系数唯一：平方损失可能不严格凸，L1 惩罚也不严格凸；唯一性还与设计矩阵和一般位置条件有关。

## 14. 本章小结与术语

- Ridge 在 $X^TX$ 上加 $\lambda I$，使目标拥有额外曲率；它以偏差换方差，并优先收缩小奇异值方向。
- 增广最小二乘、闭式线性系统和 SVD 谱滤波描述同一个 Ridge 解，但适合不同数值与解释任务。
- Ridge 有效自由度是谱保留率之和，通常不是整数，也不表示删除了同等数量的原始字段。
- Lasso 的精确 0 来自 L1 在 0 点的次梯度区间；正交设计下它就是软阈值，一般设计下由 KKT 条件和坐标下降实现。
- Gaussian 与 Laplace 先验分别给出 Ridge 与 Lasso 的 MAP 解释；MAP 点估计不等同于完整后验推断。
- λ 的数值依赖损失缩放、标准化、截距和软件协议，模型产物必须保存完整目标约定。
- 预测误差、系数波动、活跃集合、采集成本和安全边界应分别验收；稀疏不等于稳定选择，也不等于因果发现。

<dl class="ml-terms">
  <div><dt>regularization</dt><dd>通过限制或惩罚参数空间，引入偏差以控制方差、复杂度或结构的建模方法。</dd></div>
  <div><dt>Ridge regression</dt><dd>在线性平方损失上加入 L2 平方惩罚的回归，又称 Tikhonov regularization。</dd></div>
  <div><dt>spectral shrinkage</dt><dd>在奇异向量基底中按 $s_j^2/(s_j^2+\lambda)$ 对拟合方向进行强弱不同的收缩。</dd></div>
  <div><dt>effective degrees of freedom</dt><dd>Ridge 帽子矩阵的迹，等于各谱方向保留率之和。</dd></div>
  <div><dt>Lasso</dt><dd>在线性平方损失上加入 L1 惩罚、能够产生精确零系数的凸回归方法。</dd></div>
  <div><dt>subgradient</dt><dd>凸函数不可导点上推广梯度的集合；L1 在 0 点的次梯度是 $[-1,1]$。</dd></div>
  <div><dt>soft thresholding</dt><dd>把绝对值未超过阈值的坐标置零，其余坐标向零平移阈值距离的近端算子。</dd></div>
  <div><dt>coordinate descent</dt><dd>固定其他坐标并逐个精确或近似最小化当前坐标子问题的优化方法。</dd></div>
  <div><dt>regularization path</dt><dd>系数解随惩罚参数变化形成的轨迹，常由大到小 λ 配合 warm start 计算。</dd></div>
  <div><dt>selection stability</dt><dd>数据扰动或重采样下，变量活跃状态、符号与组级信息是否重复出现。</dd></div>
</dl>

## 来源与许可

- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 412–415 页，§11.3：用于核对 Gaussian 先验到 Ridge MAP、闭式解、增广数据、SVD 谱收缩和有效自由度。本站以电机热点软测量重新推导并独立实现，不复用书中案例、图表或措辞。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 415–422 页，§11.4：用于核对 Laplace 先验、L1 约束几何、次梯度、软阈值、正则化路径和活跃集非单调边界。两项 SVG 实验、代码与题目均为本站独立创作。
- Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 77–81 页（讲义页码 71–75），Chapter 13：用于交叉核对 Ridge 唯一性、偏差—方差、特征标准化、L1/L2 等高线直觉和 Lasso 路径；不复制讲义图片、课堂案例或措辞。
- Stanford CS229 Lecture Notes，PDF 第 143–144 页，§9.4：用于核对从参数先验、后验到 MAP 点估计的边界，以及 Gaussian 先验对过拟合的影响；本站不复用原文示例。
- 本章固定设计偏差—方差、KKT、近端梯度、复杂度与求解验收属于标准机器学习知识；公式由本站独立组织并通过 NumPy 数值验证。
- 本章 336 个稳定窗口、电机参数、温度关系、分组切分、bootstrap、两项交互实验、所有题目和 SVG 图形均为教学合成与本站独立创作，不代表真实设备性能、安全阈值或维护决策。详细登记见[来源与许可登记](./sources.md) 。

<MlLessonProgress lesson-id="06" next-label="第 07 章：基函数回归与局部加权回归" next-href="/courses/machine-learning/07-basis-functions" />
