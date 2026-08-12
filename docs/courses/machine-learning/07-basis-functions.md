---
course: machine-learning
edition: industrial
title: 基函数回归与局部加权回归
description: 以冷站湿球温度—单位冷量功耗性能基线为贯穿案例，从特征映射、样条与 Ridge 平滑矩阵推导全局基函数回归，再从核权重、局部常数和局部线性模型推导局部加权回归，并建立复杂度选择、边界诊断与部署审计流程。
outline: [2, 3]
---

<MlLessonHero number="07" title="基函数回归与局部加权回归" question="冷站效率曲线明显弯曲、边界样本又稀少：应当预先铺设一组全局基函数，还是让每个查询温度拥有自己的局部模型？" duration="建议 480–600 分钟" :outcomes="['区分对输入非线性与对参数线性，并把基函数写成设计矩阵', '比较多项式、Fourier、RBF、hinge 与 B-spline 的支撑范围和工程先验', '推导带一般惩罚矩阵的 Ridge 基函数解、平滑矩阵与有效自由度', '从核加权目标推导 Nadaraya–Watson 与局部线性回归，并解释边界偏差', '在分组交叉验证内部完成缩放、节点构造、次数与正则强度选择', '审计外推、局部有效样本量、维数灾难、查询成本与运行域越界']" />

第 06 章用 Ridge 和 Lasso 稳定了一个给定的线性表示。本章向前走一步：如果湿球温度与单位冷量功耗之间不是直线，应当改变的首先是**表示**，而不一定是优化器。

两条路线都能产生弯曲函数，但计算语义不同：

- **基函数回归**先固定一套特征映射 $\phi(x)$，再一次性学习全局系数；上线只需计算基函数和一次点积；
- **局部加权回归**在每个查询点 $x_0$ 周围重新分配样本权重并拟合局部模型；训练近乎只是保存数据，成本推迟到查询阶段；
- 两者都需要用验证数据选择复杂度，也都不自动拥有可信外推能力。

## 贯穿案例：冷站性能基线不是一条直线

园区冷站需要建立“湿球温度—单位冷量功耗”性能基线。湿球温度升高时，冷却塔散热条件恶化，冷机与水泵的组合效率往往非线性变化。运维团队希望在稳定负荷带内回答：

1. 当前单位冷量功耗是否显著高于健康基线；
2. 高温端样本少时，曲线是否只是模型摆动；
3. 清洗换热器前后，性能曲线在哪些温区发生变化。

本章把响应定义为

$$
y=\frac{\text{冷站总有功功率（kW）}}{\text{制冷量（RT）}},
$$

单位为 kW/RT。只研究部分负荷率 $0.65\le\mathrm{PLR}\le0.75$ 的稳定窗口，避免把负荷结构变化误当成温度曲线。

本章数据、设备参数、阈值和性能曲线均为**教学合成**，不代表任何真实冷站、冷机型号或节能验收标准。真实系统还需核对流量计、温度计、能量平衡、设备启停组合和控制逻辑。

### 数据合同：一行是一段稳定运行窗口

| 字段 | 单位 | 角色 | 工程边界 |
|---|---:|---|---|
| `day_id` | — | 独立运行日 | 切分与交叉验证单位 |
| `window_time` | min | 日内窗口序号 | 用于排序，不作为未来信息特征 |
| `wet_bulb_temperature` | °C | 主要解释变量 | 量程漂移会直接移动曲线 |
| `part_load_ratio` | 1 | 运行域过滤与协变量 | 本章只保留 0.65–0.75 |
| `cooling_tons` | RT | 分母 | 低流量或坏点会放大比值 |
| `electric_power` | kW | 分子 | 必须与制冷量时间对齐 |
| `specific_power` | kW/RT | 响应 | 由对齐后的分子分母计算 |
| `operating_mode` | 类别 | 运行域标签 | 启停、清洗、旁通工况必须剔除或分模 |

相邻窗口共享天气与热惯性，随机逐行切分会让同一天同时出现在训练和验证中。下面按完整 `day_id` 切分：前 24 天训练，中间 6 天验证，最后 6 天测试。测试集只在模型和阈值冻结后打开一次。

```python
import numpy as np

rng = np.random.default_rng(20260805)
n_days, windows_per_day = 36, 8
day_id = np.repeat(np.arange(n_days), windows_per_day)
window_time = np.tile(np.arange(windows_per_day), n_days)
n_observations = day_id.size

# 每天覆盖一个天气中心，日内只有小幅变化；同日窗口显然不是独立样本。
day_wet_bulb = rng.uniform(16.5, 31.5, n_days)
wet_bulb_temperature = (
    day_wet_bulb[day_id]
    + 0.8 * np.sin((window_time - 2.0) / 7.0 * np.pi)
    + rng.normal(0.0, 0.22, n_observations)
)
part_load_ratio = np.clip(
    0.70 + rng.normal(0.0, 0.018, n_observations),
    0.65,
    0.75,
)
cooling_tons = 3100.0 * part_load_ratio + rng.normal(0.0, 24.0, n_observations)

true_specific_power = (
    0.515
    + 0.00145 * (wet_bulb_temperature - 20.0) ** 2
    + 0.012 * np.maximum(wet_bulb_temperature - 25.0, 0.0)
    + 0.38 * (part_load_ratio - 0.70) ** 2
)
day_efficiency_shift = rng.normal(0.0, 0.006, n_days)
specific_power = (
    true_specific_power
    + day_efficiency_shift[day_id]
    + rng.normal(0.0, 0.007, n_observations)
)
electric_power = cooling_tons * specific_power
operating_mode = np.full(n_observations, "stable")

train_mask = day_id < 24
valid_mask = (day_id >= 24) & (day_id < 30)
test_mask = day_id >= 30

x_train = wet_bulb_temperature[train_mask]
y_train = specific_power[train_mask]
x_valid = wet_bulb_temperature[valid_mask]
y_valid = specific_power[valid_mask]
x_test = wet_bulb_temperature[test_mask]
y_test = specific_power[test_mask]

assert x_train.shape == (192,)
assert set(day_id[train_mask]).isdisjoint(set(day_id[test_mask]))
assert np.all((part_load_ratio >= 0.65) & (part_load_ratio <= 0.75))
assert np.allclose(electric_power / cooling_tons, specific_power)
```

<MlQuiz kind="数据边界" question="同一天的 8 个稳定窗口共享天气与设备状态。哪种切分最能评估对未来运行日的泛化？" :options="['逐行随机切分', '按完整 day_id 分组切分，并在训练折内重算所有变换', '按响应大小排序后交替切分', '先用全数据选次数再切分']" :answer="1" explanation="独立信息单位是运行日。逐行切分会把同一天的共同扰动泄漏到验证集；用全数据选表示也属于模型选择泄漏。" />

## 1. “线性”描述参数，不描述曲线外形

给定固定特征映射

$$
\phi(x)=
[\phi_0(x),\phi_1(x),\ldots,\phi_{m-1}(x)]^T,
$$

模型写成

$$
f_\beta(x)=\beta^T\phi(x)
=\sum_{j=0}^{m-1}\beta_j\phi_j(x).
$$

$\phi_j(x)$ 可以是弯曲、周期或局部的；只要它们在拟合前固定，未知参数 $\beta$ 仍以线性组合出现。对 $n$ 个样本构造

$$
\Phi=
\begin{bmatrix}
\phi(x_1)^T\\
\phi(x_2)^T\\
\vdots\\
\phi(x_n)^T
\end{bmatrix}
\in\mathbb R^{n\times m},
$$

于是训练又回到第 04–06 章熟悉的线性代数：

$$
\min_\beta
\frac12\lVert y-\Phi\beta\rVert_2^2
+\frac\lambda2\lVert L\beta\rVert_2^2.
$$

$L$ 是惩罚算子。若截距不惩罚，可令 $L^TL=\operatorname{diag}(0,1,\ldots,1)$；样条还可用二阶导数惩罚对应的非对角矩阵。

### 输入维度与模型维度不是一回事

一维 $d$ 次多项式有 $d+1$ 列。若有 $D$ 个输入并包含总次数不超过 $d$ 的全部交互项，列数为

$$
m=\binom{D+d}{d}.
$$

例如 10 个传感器做到三次全交互，已有 $\binom{13}{3}=286$ 列。多变量基展开会迅速增加存储、条件数和估计方差；“原始输入只有 10 列”并不表示模型简单。

<MlQuiz kind="概念辨析" question="模型 f(x)=β₀+β₁sin(x)+β₂x² 是否属于线性模型？" :options="['不属于，因为曲线会弯', '属于，因为固定特征映射后对 β 线性', '只有 x 为整数时属于', '只有 β₂=0 时属于']" :answer="1" explanation="线性模型的线性指未知参数的组合方式。输入先经过 sin 和平方映射不破坏参数线性。" />

<div class="ml-stop"><strong>停下来推一推 1</strong>写出两个输入 $(x_1,x_2)$ 的二次全多项式设计行。哪些列是主效应，哪一列是交互作用？如果删除交互列，模型失去了什么表达能力？</div>

## 2. 基函数家族编码不同的结构先验

没有“最灵活就最好”的基函数。选择基函数就是声明函数在哪些位置可以变化、变化是否周期、影响是全局还是局部。

| 家族 | 典型形式 | 支撑范围 | 适合的结构 | 主要风险 |
|---|---|---|---|---|
| 多项式 | $1,z,z^2,\ldots,z^d$ | 全局 | 单区间平滑趋势、低阶曲率 | 高阶病态、边界振荡、外推爆炸 |
| Fourier | $\sin(k\omega x),\cos(k\omega x)$ | 全局 | 有已知周期的角度、日周期、转动相位 | 周期边界设错会强行首尾相接 |
| RBF | $\exp[-(x-c_j)^2/(2s^2)]$ | 局部但非紧支撑 | 平滑局部变化 | 中心与宽度也需验证；列可能高度相似 |
| hinge | $(x-\kappa_j)_+$ | 节点右侧 | 连续分段线性、阈值后斜率变化 | 导数在节点跳变 |
| B-spline | 分段多项式递推 | 紧局部支撑 | 平滑曲线、局部可控 | 节点、阶数、边界条件需冻结 |

先把输入中心化和缩放，能显著改善 $z^d$ 的数值尺度，但不能消除高阶多项式固有的全局耦合：改变一个系数仍会影响整个区间。

### 多项式回归与 Vandermonde 设计

令 $z=(x-\mu)/s$，$d$ 次设计矩阵为

$$
\Phi_{ij}=z_i^j,\qquad j=0,1,\ldots,d.
$$

当次数从 $d$ 增加到 $d+1$，旧模型可以把新系数设为 0，因此最小训练 RSS 不会增加。但验证误差没有这个单调保证。高次项还会在训练区间外快速主导：

$$
|z|>1\quad\Longrightarrow\quad |z|^d \text{ 随 }d\text{ 快速增长}.
$$

正交多项式可以改善列的条件数。它改变的是同一函数空间的坐标系，不会凭空给高阶外推增加物理依据。

### hinge 与样条：把变化限制在节点附近

连续分段线性模型可写成

$$
f(x)=\beta_0+\beta_1x
+\sum_{k=1}^{K}\gamma_k(x-\kappa_k)_+.
$$

当 $x$ 穿过节点 $\kappa_k$，斜率增加 $\gamma_k$，函数值保持连续。三次回归样条则让各段为三次多项式，并通过约束使节点处的函数、一阶导数和二阶导数连续。

B-spline 是同一类分段多项式空间的数值稳定基。它的关键工程优势是**局部支撑**：调整一个系数主要影响少数相邻节点区间，而不是整条曲线。

节点不能偷看测试标签。等频节点若从全数据分位数计算，也已使用验证分布；严格验证时，节点位置、输入尺度、RBF 中心和宽度都必须在当前训练折内确定。

<MlQuiz kind="结构选择" question="设备响应已知每 360° 周期重复，且 0° 与 360° 必须连续。哪类基函数最直接编码这个约束？" :options="['Fourier 正弦/余弦基', '未经缩放的高次幂', '只含截距', '把角度当类别且删除一个水平']" :answer="0" explanation="Fourier 基天然首尾周期连续；多项式不会自动把两个端点连接起来。" />

<MlQuiz kind="泄漏审计" question="用全体样本分位数确定 spline 节点，再进行按日交叉验证，问题在哪里？" :options="['没有问题，节点不是标签', '节点构造提前使用了验证折的输入分布，模型选择流程越过信息边界', '只会让训练更慢', '只影响截距']" :answer="1" explanation="部署时不会提前知道未来输入分布。凡由数据估计的表示参数都应在训练折内拟合。" />

## 3. 从设计矩阵到稳定的 Ridge 基函数解

一般二次惩罚目标为

$$
J(\beta)=
\frac12\lVert y-\Phi\beta\rVert_2^2
+\frac\lambda2\lVert L\beta\rVert_2^2.
$$

一阶条件给出

$$
(\Phi^T\Phi+\lambda L^TL)\widehat\beta=\Phi^Ty.
$$

若矩阵可逆，

$$
\widehat\beta=
(\Phi^T\Phi+\lambda L^TL)^{-1}\Phi^Ty.
$$

代码不应显式求逆。把惩罚写进增广最小二乘：

$$
\left\lVert
\begin{bmatrix}y\\0\end{bmatrix}
-
\begin{bmatrix}\Phi\\\sqrt\lambda L\end{bmatrix}\beta
\right\rVert_2^2,
$$

即可交给 QR/SVD 型 `lstsq` 求解。

下面建立四种一维设计矩阵。所有数据驱动参数都显式传入，便于在交叉验证折内重算。

```python
def standardize_1d(x, center=None, scale=None):
    values = np.asarray(x, dtype=float)
    if center is None:
        center = float(values.mean())
    if scale is None:
        scale = float(values.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    return (values - center) / scale, center, scale

def polynomial_design(z, degree):
    values = np.asarray(z, dtype=float)
    return np.column_stack([values ** power for power in range(degree + 1)])

def hinge_design(x, knots):
    values = np.asarray(x, dtype=float)
    knot_values = np.asarray(knots, dtype=float)
    return np.column_stack([
        np.ones(values.size),
        values,
        *[np.maximum(values - knot, 0.0) for knot in knot_values],
    ])

def rbf_design(x, centers, width):
    values = np.asarray(x, dtype=float)[:, None]
    center_values = np.asarray(centers, dtype=float)[None, :]
    if width <= 0:
        raise ValueError("width must be positive")
    features = np.exp(-0.5 * ((values - center_values) / width) ** 2)
    return np.column_stack([np.ones(values.shape[0]), features])

def fourier_design(x, period, harmonics):
    values = np.asarray(x, dtype=float)
    if period <= 0:
        raise ValueError("period must be positive")
    columns = [np.ones(values.size)]
    for order in range(1, harmonics + 1):
        angle = 2.0 * np.pi * order * values / period
        columns.extend([np.sin(angle), np.cos(angle)])
    return np.column_stack(columns)

z_train, x_center, x_scale = standardize_1d(x_train)
Phi_demo = polynomial_design(z_train, degree=5)
assert Phi_demo.shape == (192, 6)
assert np.allclose(Phi_demo[:, 0], 1.0)
```

```python
def fit_basis_ridge(Phi, target, lambda_value):
    design = np.asarray(Phi, dtype=float)
    response = np.asarray(target, dtype=float)
    if lambda_value < 0:
        raise ValueError("lambda_value must be non-negative")
    penalty_root = np.eye(design.shape[1])
    penalty_root[0, 0] = 0.0
    if lambda_value == 0:
        return np.linalg.lstsq(design, response, rcond=None)[0]
    augmented_X = np.vstack([design, np.sqrt(lambda_value) * penalty_root])
    augmented_y = np.concatenate([response, np.zeros(design.shape[1])])
    return np.linalg.lstsq(augmented_X, augmented_y, rcond=None)[0]

def smoother_matrix(Phi, lambda_value):
    design = np.asarray(Phi, dtype=float)
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    system = design.T @ design + lambda_value * penalty
    return design @ np.linalg.solve(system, design.T)

beta_demo = fit_basis_ridge(Phi_demo, y_train, lambda_value=0.1)
S_demo = smoother_matrix(Phi_demo, lambda_value=0.1)
effective_df_demo = float(np.trace(S_demo))

assert beta_demo.shape == (6,)
assert 1.0 < effective_df_demo < Phi_demo.shape[1]
assert np.allclose(S_demo, S_demo.T, atol=1e-10)
print("五次多项式 Ridge 的有效自由度：", round(effective_df_demo, 3))
```

<BasisExpansionWorkbenchLab />

实验 07-A 中，浅蓝点是建模样本，黄色曲线是拟合结果，底部五条“通道”显示不同基函数对当前曲线的加权贡献。观察三个联动：

1. 增加多项式次数会扩大名义维度，但训练误差只会不增；
2. 减小 $\lambda$ 会提高有效自由度，边界曲线更容易摆动；
3. hinge 基把斜率变化放在节点之后，局部解释通常比高次幂直接。

<MlQuiz kind="实验读图" question="实验中把多项式次数提高，而训练 RMSE 下降、验证 RMSE 上升，最合理的解释是什么？" :options="['更大的嵌套函数空间降低训练误差，但抽样方差和边界摆动增大', '优化器一定出错', '验证集标签必然错误', '高次模型参数更少']" :answer="0" explanation="训练拟合能力与总体泛化不是同一目标。次数、正则和样本覆盖共同决定验证风险。" />

<div class="ml-stop"><strong>停下来推一推 2</strong>对 hinge 模型求 $f'(x)$。当查询点从节点左侧越到右侧时，斜率改变多少？为什么函数连续但导数可以不连续？</div>

## 4. 平滑矩阵把“名义维度”变成有效自由度

固定 $\Phi$、$L$ 和 $\lambda$ 时，训练点上的拟合值为

$$
\widehat y
=\Phi(\Phi^T\Phi+\lambda L^TL)^{-1}\Phi^Ty
=S_\lambda y.
$$

$S_\lambda$ 称为平滑矩阵或线性平滑器矩阵。OLS 满列秩时的帽子矩阵是正交投影，满足 $H^2=H$；Ridge 的谱值通常在 0 与 1 之间，因此

$$
S_\lambda^2\ne S_\lambda.
$$

重复平滑仍会继续改变结果。有效自由度定义为

$$
\operatorname{df}_{eff}=\operatorname{tr}(S_\lambda).
$$

若 $L=I$ 且 $\Phi=U\operatorname{diag}(s_j)V^T$，

$$
\operatorname{df}_{eff}
=\sum_j\frac{s_j^2}{s_j^2+\lambda}.
$$

截距不惩罚时，它固定贡献约 1，其余方向按数据强度连续贡献 0 到 1。于是：

- 名义列数 $m$ 只说明最多有多少参数方向；
- 秩 $\operatorname{rank}(\Phi)$ 说明数据能辨认多少方向；
- $\operatorname{tr}(S_\lambda)$ 说明正则后实际用了多少拟合灵活度。

对于一般二阶差分或曲率惩罚，不能直接套用上面的简单奇异值分式，但 $\operatorname{tr}(S_\lambda)$ 的定义仍成立。

### 线性平滑器的留一误差捷径

若表示、带宽和 $\lambda$ 在删去第 $i$ 个样本后仍按同一规则固定，线性平滑器满足

$$
y_i-\widehat y_i^{(-i)}
=\frac{y_i-\widehat y_i}{1-S_{ii}}.
$$

这能避免真的拟合 $n$ 次。但如果节点、尺度、RBF 中心会随删除样本而重估，或者超参数也借留一结果选择，上式的使用边界必须重新检查。工程实现不能为了速度悄悄改变验证协议。

<MlQuiz kind="复杂度判断" question="一个模型有 40 个基函数，但强 Ridge 下 tr(Sλ)=6.4。哪项表述最准确？" :options="['模型只存了 6.4 个系数', '名义参数有 40 个，但响应拟合灵活度约为 6.4', '矩阵秩一定是 6', '训练样本只有 6.4 个']" :answer="1" explanation="有效自由度是平滑器对数据的总敏感度，可以非整数；它不等于实际存储的系数数量。" />

## 5. 复杂度选择：验证整个表示流程

需要一起选择的对象可能包括：

- 多项式次数；
- spline 节点数、位置与阶数；
- RBF 中心数与宽度；
- Fourier 周期与谐波数；
- Ridge 强度 $\lambda$；
- 是否加入负荷率、交互项或分运行模式建模。

这些都属于模型。正确流程是在每个训练折内部：

1. 根据折内运行日估计输入中心和尺度；
2. 根据折内输入确定节点或 RBF 中心；
3. 构建设计矩阵并拟合；
4. 变换验证折，记录日级误差、边界误差和有效自由度；
5. 汇总分组折，冻结选择，再用完整训练区拟合。

下面只演示多项式次数与 Ridge 强度的分组交叉验证。为保持测试集纯净，候选选择仅使用最初 24 个训练日。

```python
def rmse(target, prediction):
    return float(np.sqrt(np.mean((np.asarray(target) - np.asarray(prediction)) ** 2)))

train_days = day_id[train_mask]
degrees = [1, 2, 3, 5, 7]
lambda_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
cv_records = []

for degree in degrees:
    for lambda_value in lambda_grid:
        fold_scores = []
        for fold in range(4):
            fold_valid = train_days % 4 == fold
            fold_train = ~fold_valid
            z_fold_train, fold_center, fold_scale = standardize_1d(x_train[fold_train])
            z_fold_valid, _, _ = standardize_1d(
                x_train[fold_valid],
                center=fold_center,
                scale=fold_scale,
            )
            Phi_fold_train = polynomial_design(z_fold_train, degree)
            Phi_fold_valid = polynomial_design(z_fold_valid, degree)
            beta_fold = fit_basis_ridge(Phi_fold_train, y_train[fold_train], lambda_value)
            fold_scores.append(rmse(y_train[fold_valid], Phi_fold_valid @ beta_fold))
        cv_records.append((float(np.mean(fold_scores)), degree, lambda_value))

cv_records.sort()
best_cv_rmse, selected_degree, selected_lambda = cv_records[0]
z_train_final, final_center, final_scale = standardize_1d(x_train)
z_valid_final, _, _ = standardize_1d(x_valid, final_center, final_scale)
Phi_train_final = polynomial_design(z_train_final, selected_degree)
Phi_valid_final = polynomial_design(z_valid_final, selected_degree)
beta_selected = fit_basis_ridge(Phi_train_final, y_train, selected_lambda)
validation_rmse = rmse(y_valid, Phi_valid_final @ beta_selected)
selected_df = float(np.trace(smoother_matrix(Phi_train_final, selected_lambda)))

assert selected_degree in degrees
assert selected_lambda in lambda_grid
assert np.isfinite(validation_rmse)
print(
    "CV选择：degree=", selected_degree,
    "lambda=", selected_lambda,
    "CV RMSE=", round(best_cv_rmse, 5),
    "独立验证 RMSE=", round(validation_rmse, 5),
    "df=", round(selected_df, 2),
)
```

### one-standard-error 规则与样本量实验

最小平均验证误差常会选择较复杂模型。若每折得分标准误为 $SE$，可以选“平均误差不超过最小值加一个 $SE$”的最简单候选。这里的“简单”应先定义，例如：

1. 更低有效自由度；
2. 若有效自由度近似相同，则更低次数；
3. 若次数相同，则更强正则。

还应画学习曲线：用 6、12、18、24 个训练日重复选择。若增加独立运行日持续降低验证误差，问题主要是估计方差；若训练与验证误差都高且接近，则需要改进表示或加入物理相关协变量，而不是继续提高次数。

<MlQuiz kind="验证策略" question="固定次数后，随着 λ 增大，训练 RSS 与有效自由度通常怎样变化？" :options="['训练 RSS 不减，有效自由度不增', '两者都必然下降', '训练 RSS 下降，有效自由度上升', '两者完全不变']" :answer="0" explanation="更强惩罚限制拟合，因此相对未惩罚最小二乘的训练 RSS不会更低；平滑器谱值随 λ 增大而下降。" />

<div class="ml-stop"><strong>停下来推一推 3</strong>为什么“次数 7、强 Ridge”和“次数 3、弱 Ridge”可能拥有相近有效自由度，却仍产生不同边界形状？有效自由度概括了什么，又遗漏了什么？</div>

## 6. 局部加权回归：查询点决定哪些数据重要

全局基函数在训练时确定表示，局部方法则在查询 $x_0$ 到来时计算权重。常用 Gaussian 形权重为

$$
w_i(x_0)
=\exp\left[-\frac{(x_i-x_0)^2}{2h^2}\right],
$$

其中 $h>0$ 是带宽。这里的 $w_i$ 是由距离确定的**算法权重**，并不是“样本服从 Gaussian 分布”的概率陈述，也无需归一化后才能进入加权最小二乘。

带宽控制邻域：

- $h$ 小：只看极近样本，偏差低但方差高，局部矩阵可能近奇异；
- $h$ 大：很多样本权重相近，曲线更平滑，极限下接近全局拟合；
- 距离必须在有意义的尺度上计算，多变量时尤其如此。

### 局部常数：Nadaraya–Watson

在 $x_0$ 附近只拟合常数 $a$：

$$
\widehat a
=\arg\min_a\sum_iw_i(x_0)(y_i-a)^2.
$$

令导数为 0：

$$
\boxed{
\widehat y_{NW}(x_0)
=\frac{\sum_iw_i(x_0)y_i}{\sum_iw_i(x_0)}
}.
$$

这是邻域响应的归一化加权平均。它直观且稳定，但在边界处邻域只有一侧，容易把斜率拉平。

### 局部线性：用查询点中心化坐标

在查询点周围拟合切线：

$$
y_i\approx a+b(x_i-x_0).
$$

定义

$$
Z(x_0)=
\begin{bmatrix}
1&x_1-x_0\\
\vdots&\vdots\\
1&x_n-x_0
\end{bmatrix},
\qquad
W(x_0)=\operatorname{diag}(w_i(x_0)).
$$

则

$$
\begin{bmatrix}\widehat a\\\widehat b\end{bmatrix}
=(Z^TWZ)^{-1}Z^TWy.
$$

因为坐标以 $x_0$ 为中心，查询点的预测就是截距：

$$
\widehat y_{LL}(x_0)=\widehat a.
$$

局部线性模型能精确再现直线，并显著减小一维边界处的一阶偏差。它的等价样本权重可能出现负值；这不是数值错误，而是为了补偿单侧邻域并保持线性再现。

```python
def gaussian_weights(x, query, bandwidth):
    values = np.asarray(x, dtype=float)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    return np.exp(-0.5 * ((values - query) / bandwidth) ** 2)

def nadaraya_watson(x, y, query, bandwidth):
    weights = gaussian_weights(x, query, bandwidth)
    denominator = weights.sum()
    if denominator <= 1e-12:
        raise ValueError("query has no effective neighbors")
    return float(weights @ np.asarray(y, dtype=float) / denominator)

def local_linear_predict(x, y, query, bandwidth, ridge=1e-10):
    values = np.asarray(x, dtype=float)
    response = np.asarray(y, dtype=float)
    weights = gaussian_weights(values, query, bandwidth)
    centered = values - query
    Z = np.column_stack([np.ones(values.size), centered])
    system = Z.T @ (weights[:, None] * Z)
    penalty = np.diag([0.0, ridge])
    beta = np.linalg.solve(system + penalty, Z.T @ (weights * response))
    prediction = float(beta[0])
    effective_n = float(weights.sum() ** 2 / (weights @ weights))
    return prediction, float(beta[1]), effective_n

query_demo = 29.0
nw_demo = nadaraya_watson(x_train, y_train, query_demo, bandwidth=2.0)
ll_demo, local_slope_demo, local_neff_demo = local_linear_predict(
    x_train, y_train, query_demo, bandwidth=2.0
)
assert np.isfinite([nw_demo, ll_demo, local_slope_demo, local_neff_demo]).all()
assert 1.0 <= local_neff_demo <= x_train.size
print("29°C 局部常数/局部线性：", round(nw_demo, 4), round(ll_demo, 4))
print("局部斜率与有效样本量：", round(local_slope_demo, 4), round(local_neff_demo, 1))
```

<LocallyWeightedRegressionLab />

实验 07-B 把查询点画成可移动的“局部检测光束”。拖动查询温度并调节带宽，观察：

- 样本发光强度对应核权重，而不是标签大小；
- 局部常数只给出加权平均，局部线性还估计当前切线；
- 查询点靠近温区边界时，权重集中在单侧；
- 带宽很小时，名义上有很多历史点，实际有效样本量仍可能很低。

<MlQuiz kind="局部模型" question="为什么局部线性回归使用 xᵢ−x₀ 作为第二列后，预测值直接等于局部截距 â？" :options="['因为查询点处 x−x₀=0', '因为所有权重都等于 0', '因为斜率被强制为 0', '因为响应已标准化']" :answer="0" explanation="在查询点代入局部坐标 0，â+b̂·0=â。中心化还改善了截距与斜率的数值解释。" />

<MlQuiz kind="边界诊断" question="在训练温区左边界，局部常数估计为何常有明显偏差？" :options="['邻域只有右侧样本，加权平均无法表达局部斜率', 'Gaussian 权重一定为负', '局部常数总是外推', '因为样本数越多偏差越大']" :answer="0" explanation="边界缺少对称邻域。若真实函数有斜率，单侧响应平均会偏离查询点；局部线性可用斜率作一阶校正。" />

<div class="ml-stop"><strong>停下来推一推 4</strong>令 $h\to\infty$，Gaussian 权重趋近什么？局部常数和局部线性分别退化为哪两个全局模型？</div>

## 7. 局部方法也是线性平滑器

对固定查询 $x_0$，局部线性预测仍可写成

$$
\widehat y(x_0)=s(x_0)^Ty,
$$

其中

$$
s(x_0)^T
=e_1^T(Z^TWZ)^{-1}Z^TW.
$$

因此它对响应 $y$ 线性，但对输入位置和带宽非线性。若在每个训练输入 $x_i$ 上预测并把 $s(x_i)^T$ 堆成矩阵 $S_h$，同样可定义

$$
\operatorname{df}_{eff}(h)=\operatorname{tr}(S_h).
$$

带宽减小时，对角元素通常增大，模型更接近记忆训练响应，有效自由度上升。局部线性 $S_h$ 一般不对称，也不是正交投影。

### 有效样本量是查询级健康指标

把未归一化权重记为 $w_i$，Kish 型有效样本量为

$$
n_{eff}(x_0)
=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
$$

它不是统计独立样本数的完整替代，但能快速揭示权重是否集中在少数点。上线应同时记录：

- 查询点到训练域的距离；
- $n_{eff}(x_0)$；
- $Z^TWZ$ 的条件数；
- 最大单点权重占比；
- 当前运行模式与训练模式是否一致。

若指标越界，应拒绝静默给出高置信度数值，转入“超出标定域”或回退模型，而不是继续缩小带宽追求贴点。

## 数值验证

教材推导应当转化为可以失败的断言。先验证一个关键性质：局部线性在边界也能再现直线，而局部常数因单侧平均产生偏差。

```python
x_line = np.linspace(0.0, 1.0, 25)
y_line = 2.0 + 3.0 * x_line
boundary_query = 0.03
truth_at_query = 2.0 + 3.0 * boundary_query

nw_boundary = nadaraya_watson(x_line, y_line, boundary_query, bandwidth=0.18)
ll_boundary, ll_slope, ll_neff = local_linear_predict(
    x_line,
    y_line,
    boundary_query,
    bandwidth=0.18,
)

assert abs(ll_boundary - truth_at_query) < 1e-9
assert abs(ll_slope - 3.0) < 1e-8
assert abs(nw_boundary - truth_at_query) > 1e-3
assert ll_neff > 3.0
print("边界真值/NW/局部线性：", truth_at_query, nw_boundary, ll_boundary)
```

再检查实现级不变量：Ridge 有效自由度随 $\lambda$ 不增，训练点上的 OLS 平滑矩阵在满列秩时近似幂等，局部权重非负。

```python
df_path = np.array([
    np.trace(smoother_matrix(Phi_demo, lambda_value))
    for lambda_value in [1e-4, 1e-2, 1.0, 100.0]
])
S_ols = smoother_matrix(Phi_demo, lambda_value=0.0)
local_weights = gaussian_weights(x_train, query=27.0, bandwidth=1.8)

assert np.all(np.diff(df_path) <= 1e-10)
assert np.linalg.norm(S_ols @ S_ols - S_ols) < 1e-8
assert np.all((local_weights >= 0.0) & (local_weights <= 1.0))
assert local_weights.max() > 0.95
print("有效自由度路径：", np.round(df_path, 3))
```

### 最终测试只执行一次

选定表示、次数、$\lambda$ 或带宽后，可合并训练与验证日重新拟合，再对最后 6 天测试。最终报告至少包括：

- 全测试集 RMSE、MAE；
- 按温区分箱的误差与最大绝对误差；
- 测试点落在训练温区之外的比例；
- 每日误差分布，而不只是一项逐行平均；
- 基函数模型的有效自由度，或局部模型的 $n_{eff}$ 分布；
- 与简单线性基线和物理基线的差值。

不能在看到测试高温端误差后返回修改次数、节点或带宽，再继续称它为“测试集”。那已把测试集变成调参集，需要新的未见运行日完成最终验收。

<MlQuiz kind="数值验证" question="哪条断言最能捕获 Ridge 实现中 λ 方向写反的错误？" :options="['df(10⁻⁴)≥df(10⁻²)≥df(1)≥df(100)', '所有 β 都为正', '训练 RMSE 永远为 0', '测试误差等于训练误差']" :answer="0" explanation="固定设计下，每个受惩罚方向的保留率随 λ 增大而下降，因此有效自由度应单调不增。" />

## 复杂度与工程边界

### 计算与存储

设 $n$ 个样本、$m$ 个固定基函数：

| 阶段 | 基函数 Ridge | 一维局部线性（朴素实现） |
|---|---:|---:|
| 构建设计/权重 | $O(nm)$ | 每个查询 $O(n)$ |
| 拟合/求解 | QR 约 $O(nm^2)$ | 每个查询解 $2\times2$ 系统，扫描仍为 $O(n)$ |
| 模型存储 | $O(m)$ 系数及变换参数 | $O(n)$ 历史样本 |
| 单次预测 | $O(m)$ | $O(n)$ |
| 超参数验证 | 候选数 × 折数 × 拟合成本 | 带宽数 × 折数 × 查询成本 |

多变量局部线性若有 $p$ 个局部系数，每个查询约需 $O(np^2+p^3)$。生产优化可以使用近邻索引、紧支撑核、固定网格预计算或批量矩阵运算，但必须验证近似误差，并明确缓存何时因数据、尺度、带宽或运行模式变化而失效。

### 维数灾难

一维区间中“附近”容易定义；维度升高后，固定半径球体占总体积的比例迅速变小。为了收集足够邻居，只能增大带宽，局部模型又变得不再局部。

更严重的是距离尺度：若湿球温度、PLR、流量、时间和设备编号直接拼接，单位最大或噪声最大的维度会支配邻域。标准化只能对齐数值尺度，不能证明某维度在工艺上应进入距离。应优先：

- 用运行机理选择少量局部坐标；
- 对周期变量使用环形距离；
- 对类别运行模式分模，而不是硬塞进欧氏距离；
- 用验证确认加入一个维度是否真的改善跨日风险。

### 外推与运行域

多项式会在域外迅速增长；RBF 在远离中心时可能只剩截距；局部回归在域外只有单侧且极小的权重。三者都不能把“代码返回一个数”当作可信外推。

运行域治理应独立于模型：

1. 冻结训练温区、负荷带、设备组合和传感器质量条件；
2. 请求先通过域检查，再调用预测器；
3. 越界时返回明确状态和可审计的回退值；
4. 新工况积累到足够独立运行日后，按版本重训与再验证。

<MlQuiz kind="部署决策" question="查询温度高于所有训练温度 5°C，局部线性模型仍返回数值。正确处置是什么？" :options="['把该数值当高置信度预测', '标记超出标定域，检查有效邻居与回退策略，不把单侧外推伪装成插值', '继续减小带宽直到贴近最近点', '删除域检查']" :answer="1" explanation="局部拟合能计算不代表数据支持充分。域外只有单侧邻域，结果高度依赖最近样本和局部斜率。" />

## 10. 工业失效模式与诊断表

| 现象 | 可能根因 | 诊断 | 处置 |
|---|---|---|---|
| 训练误差很低，高温端验证误差高 | 次数过高、样本稀疏 | 画温区样本密度、边界误差与 df | 降复杂度、补高温运行日、限制运行域 |
| 改一个节点，整条多项式曲线大变 | 全局基病态 | 查看条件数、系数 bootstrap | 标准化，改用稳定 spline/RBF 表示 |
| 局部预测随单个点跳变 | 带宽过小或坏点高权重 | 查看 $n_{eff}$、最大权重和残差 | 提高带宽、稳健权重、治理传感器坏点 |
| 交叉验证很好，上线恶化 | 同日泄漏、模式漂移 | 按日重切；按设备组合分层 | 修正信息边界，建立模式与漂移监控 |
| 0°/360° 附近断裂 | 普通欧氏距离忽略周期 | 比较环形距离与端点残差 | 使用 Fourier 或周期核 |
| 预测延迟随历史库增长 | 朴素局部查询为 $O(n)$ | 监控 P95/P99 延迟与内存 | 近邻索引、压缩原型或改用全局基函数 |
| 曲线在清洗日突然变好 | 维护事件改变数据生成过程 | 对齐维护日志与日级残差 | 分版本建模，不把维护标签泄漏进历史验证 |

### 负响应和单调性不是自动被禁止

普通多项式、spline 和局部线性并不保证单位功耗为正，也不保证高温端单调增加。若物理约束必须满足，需要显式采用：

- 对响应建模 $\log y$ 以保证正值；
- 带线性不等式约束的系数求解；
- 单调 spline 或 isotonic 方法；
- 预测后安全裁剪，但必须记录裁剪率并认识到它改变统计性质。

不要用“训练区间内刚好没违约”替代约束证明。

## 11. 从案例到上线验收

冷站性能基线的最小可部署流程如下：

1. **定义窗口**：只纳入传感器健康、设备组合稳定、PLR 在目标带内的窗口；
2. **冻结信息边界**：以运行日为切分单位，标准化和节点在折内拟合；
3. **建立基线**：至少比较常数、一次线性、低次 Ridge 基函数和局部线性；
4. **选择复杂度**：联合看日级验证误差、边界误差、df 或 $n_{eff}$，优先 one-SE 简模型；
5. **压力测试**：模拟温区边界、坏点、缺测、传感器偏置和查询延迟；
6. **锁定测试**：只在候选和阈值冻结后打开测试日；
7. **部署监控**：记录运行域越界率、残差分布、局部有效样本量和版本；
8. **受控更新**：新数据先进入影子评估，不能在线静默改变节点或带宽。

模型输出用于性能诊断，不应直接覆盖冷机保护、启停联锁或设备厂商控制边界。

<MlQuiz kind="验收协议" question="局部模型上线后最值得与预测值一起记录的是哪组信息？" :options="['查询点、运行域状态、有效样本量、最大权重与模型版本', '只记录最终小数', '训练集全部标签但不记录版本', '只记录带宽名称']" :answer="0" explanation="局部预测的可信度依赖查询位置和邻域支持；没有域状态、邻域健康和版本，就无法复盘异常结果。" />

## 12. 章节练习

### 练习 1：多项式次数、Ridge 与样本量三因素实验

在教学合成数据上比较次数 $d\in\lbrace 1,2,3,5,7,9\rbrace$、$\lambda\in\lbrace 10^{-4},10^{-2},1,100\rbrace$ 和训练日数 $N\in\lbrace 6,12,18,24\rbrace$。每个组合至少重复 20 组按日重采样，报告：

- 训练、验证和高温端 RMSE；
- 有效自由度分布；
- 曲线在 16°C 与 33°C 的预测分布；
- 选择最小验证误差与 one-SE 规则的差异。

解释“样本量增加”改变的是哪些误差来源，而不是只展示一张最优曲线。

### 练习 2：RBF 与 hinge 的表示比较

只用训练日确定 8 个等频中心/节点。比较：

- 共享宽度 RBF Ridge；
- 连续分段线性 hinge Ridge；
- 五次多项式 Ridge。

对每个模型画单个基函数、加权贡献和最终曲线，讨论局部支撑、条件数、外推和解释方式。不得用测试日挑中心或宽度。

### 练习 3：从零实现局部线性平滑矩阵

对每个训练输入构造一行 $s(x_i)^T$，堆成 $S_h$。验证：

1. $S_hy$ 与逐点局部线性预测一致；
2. 每行权重和接近 1；
3. 对直线响应，预测在数值误差内精确；
4. 带宽减小时 $\operatorname{tr}(S_h)$ 通常上升；
5. 局部线性等价权重可以为负，但原始 Gaussian 核权重非负。

### 练习 4：工程代码骨架

补全按日交叉验证、one-SE 选择与域外拒绝。不要调用测试集参与 `select_model`。

```python
# TODO: 练习骨架，验证脚本会跳过本代码块。
def select_model(train_x, train_y, train_day_id, candidates):
    """返回按运行日分组验证得到的冻结配置与诊断表。"""
    raise NotImplementedError

def predict_with_domain_guard(model, query_x, calibrated_range):
    """域外返回状态，不把可计算数值伪装成受支持预测。"""
    raise NotImplementedError
```

### 练习 5：口头审查

不看公式，用两分钟回答：

1. 为什么固定 RBF 中心时是线性模型，学习中心时通常不是？
2. 为什么多项式训练误差随次数不增，却不能据此选次数？
3. 有效自由度与参数个数、矩阵秩有何区别？
4. 为什么局部线性能减小边界偏差？
5. 哪些对象必须在交叉验证折内重估？
6. 为什么多变量局部回归容易遭遇维数灾难？

## 13. 本章小结

- 固定基函数把非线性输入映射转化为对参数线性的设计矩阵问题；
- 多项式、Fourier、RBF、hinge 和 B-spline 编码不同的全局、周期与局部结构；
- Ridge 平滑矩阵 $S_\lambda$ 把响应映射到拟合值，$\operatorname{tr}(S_\lambda)$ 衡量连续的有效复杂度；
- 局部常数是核加权平均，局部线性通过查询点中心化的切线拟合减小边界偏差；
- 次数、节点、中心、尺度、正则和带宽都属于验证流程，不能提前看验证或测试数据；
- 工业部署必须同时治理运行域、有效邻域、查询延迟、数值条件和数据漂移。

## 来源与许可

- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 401–403 页，第 11.1 节与 11.2.1 节：线性回归、固定特征变换与多项式回归。用于核对“对参数线性”和设计矩阵结构；正文、冷站案例、代码、题目与图形均独立创作。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 429–432 页，第 11.5 节：回归 spline、局部支撑、节点、B-spline 设计矩阵、Ridge 与交叉验证。仅作知识边界核对，不复用原文图表或案例。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 594–596 页，第 16.3.5 节：核回归、Nadaraya–Watson、局部加权线性回归、LOWESS/LOESS。用于核对公式与术语边界，交互实验与实现独立创作。
- Stanford CS229 Lecture Notes，PDF 第 19–20 页，第 1.4 节：多项式欠拟合/过拟合、局部加权回归目标、Gaussian 形权重、带宽与非参数查询成本。仅作推导交叉核对，不复制讲义住房案例、图形或表达。
- Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 65–66 页（讲义页码 59–60）：多项式特征扩展、边界振荡与加权最小二乘。仅作工程边界核对，不复用原文图形或课堂案例。

*Probabilistic Machine Learning: An Introduction* 按其 `CC BY-NC-ND 4.0` 许可边界处理：本章只把它作为知识参考，不翻译、改写或重绘其原文材料。其余讲义未确认可再分发许可，本站只核对公共知识。详细登记见 [机器学习教材来源与许可登记](./sources) 。

<MlLessonProgress lesson-id="07" next-label="第 08 章：贝叶斯决策与最优分类器" next-href="/courses/machine-learning/08-bayesian-decision" />
