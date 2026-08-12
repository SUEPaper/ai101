---
course: machine-learning
edition: industrial
title: 核方法与核 SVM
description: 以滚子终检的环状缺陷分流为贯穿案例，从软间隔原始问题推导 Lagrangian、对偶与 KKT，再建立特征映射、Gram 半正定、Polynomial/RBF 核、SMO 与规模化近似。
outline: [2, 3]
---

<MlLessonHero
  number="12"
  title="核方法与核 SVM"
  question="原空间没有一条直线能分开的样本，怎样在不显式构造高维特征的前提下求出凸、可审计的非线性分类器？"
  duration="建议 460–560 分钟"
  :outcomes="[
    '从软间隔 SVM 的 Lagrangian 逐项消去原始变量，得到对偶目标、盒约束与标签平衡',
    '用完整 KKT 条件、原始/对偶间隙和 α 状态审计支持向量与求解精度',
    '证明 Gram 半正定，解释多项式/RBF 核、核超参数、SMO、预测成本与显式近似'
  ]"
/>

第 11 章在原始特征空间中训练

$$
f(x)=w^Tx+b.
$$

若“合格”位于特征平面中心，而需要复检的样本从多个方向包围它，任何一条直线都会失败。直接堆叠多项式特征是一条路，但高阶组合数会迅速增长。核方法的关键不是一句“升到高维”，而是建立三层严格接口：

$$
\boxed{
\text{SVM 对偶只依赖内积}
\Rightarrow
K(x,z)=\phi(x)^T\phi(z)
\Rightarrow
\text{不显式形成 }\phi\text{ 也能训练与预测}
}
$$

本章必须同时守住：

- 对偶可行不等于原始问题自动可行；
- 相似度函数不一定是有效核；
- 在一批样本上 Gram 近似半正定，不足以证明函数全局有效；
- RBF 的 $\gamma$、特征尺度与 $C$ 必须联合选择；
- “无限维”没有消除 $O(n^2)$ Gram 存储和支持向量预测成本。

## 贯穿案例：滚子终检的环状异常分流

上一章用涡流响应和轮廓缺口深度演示线性边界。本章考虑另一种真实会出现的结构：稳定工艺样本集中在联合特征中心；过热、表层脱碳、裂纹或探头耦合异常可能沿不同方向远离中心，形成近似环状的“需复检”区域。

教学任务仍是：

| 标签 | 动作 | 输出边界 |
|---:|---|---|
| $-1$ | 自动放行 | 中心稳定区 |
| $+1$ | 送复检 | 外围异常区 |

环状几何只是教学合成结构，不代表任何真实滚子缺陷都遵循圆形边界。部署前必须先验证这种相似性先验是否符合设备、材料和工艺知识。

### 数据合同

| 对象 | 教学定义 | 工业系统必须确认 |
|---|---|---|
| 观测单元 | 一只滚子的一次冻结终检记录 | 多探头、多角度和复测如何聚合 |
| 两个特征 | 去基线后的涡流相位、轮廓曲率 | 单位、窗口、滤波和探头版本 |
| 标签 | 独立复核后自动放行/送复检 | 未知、争议与标签修订路径 |
| 分组键 | 热处理批次 | 原料炉次与设备嵌套结构 |
| 相似度先验 | 标准化空间中的径向或局部邻近 | 是否应使用各向异性尺度或结构核 |
| 部署目标 | 新批次同工艺滚子 | 新材料与设备换型必须重验 |

教学数据含 30 个批次、每批 12 只滚子；20 批训练、5 批验证、5 批测试。类别按中心/外围半径生成，并加入批次偏移、测量噪声与少量争议标签。

```python
import numpy as np

rng = np.random.default_rng(20260805)
lot_count = 30
rollers_per_lot = 12
lot_id = np.repeat(np.arange(lot_count), rollers_per_lot)
sample_count = len(lot_id)
lot_shift = rng.normal(0, 0.10, size=(lot_count, 2))

label = np.where(rng.random(sample_count) < 0.38, 1.0, -1.0)
radius = np.where(
    label == 1,
    rng.normal(1.72, 0.22, sample_count),
    rng.normal(0.66, 0.18, sample_count),
)
angle = rng.uniform(-np.pi, np.pi, sample_count)
latent = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
latent += lot_shift[lot_id] + rng.normal(0, 0.045, size=(sample_count, 2))

raw_features = np.column_stack([
    31.0 + 4.8 * latent[:, 0],
    8.0 + 2.6 * latent[:, 1],
])
disputed = rng.choice(sample_count, size=8, replace=False)
label[disputed] *= -1

lot_order = rng.permutation(lot_count)
train_lots = lot_order[:20]
validation_lots = lot_order[20:25]
test_lots = lot_order[25:]
train_mask = np.isin(lot_id, train_lots)
validation_mask = np.isin(lot_id, validation_lots)
test_mask = np.isin(lot_id, test_lots)

train_mean = raw_features[train_mask].mean(axis=0)
train_scale = raw_features[train_mask].std(axis=0, ddof=0)
standardized = (raw_features - train_mean) / train_scale
X_train, y_train = standardized[train_mask], label[train_mask]
X_validation, y_validation = standardized[validation_mask], label[validation_mask]
X_test, y_test = standardized[test_mask], label[test_mask]

assert len(set(train_lots) & set(validation_lots)) == 0
assert len(set(train_lots) & set(test_lots)) == 0
assert len(set(validation_lots) & set(test_lots)) == 0
assert np.allclose(X_train.mean(axis=0), 0, atol=1e-12)
print("训练/验证/测试：", len(y_train), len(y_validation), len(y_test))
print("教学复检率：", round((label == 1).mean(), 4))
print("训练标准化尺度：", np.round(train_scale, 3))
```

<MlQuiz kind="先验合同" question="看到二维训练散点呈环状，就直接选 RBF 核并宣布可上线，缺少的关键证据是什么？" :options="['只缺训练准确率', '还需证明分组外数据保持相似结构、尺度有业务含义，并用独立批次联合验收 C 与 γ', '只需把 γ 设得很大', '核方法不需要标签合同']" :answer="1" explanation="训练图形只是有限样本证据；核与尺度编码结构先验，必须在独立部署层级验证。" />

## 1. 对偶从带约束原始问题开始

把输入先写成一般特征映射 $\phi(x)$。软间隔原始问题是

$$
\begin{aligned}
\min_{w,b,\xi}\quad&
\frac12\lVert w\rVert^2+C\sum_i\xi_i\\
\text{s.t.}\quad&
1-\xi_i-y_i(w^T\phi(x_i)+b)\le0,\\
&-\xi_i\le0.
\end{aligned}
$$

对第一组约束引入 $\alpha_i\ge0$，对非负松弛约束引入 $\mu_i\ge0$：

$$
\begin{aligned}
\mathcal L(w,b,\xi,\alpha,\mu)
=&\frac12\lVert w\rVert^2+C\sum_i\xi_i\\
&+\sum_i\alpha_i[1-\xi_i-y_i(w^T\phi_i+b)]
-\sum_i\mu_i\xi_i.
\end{aligned}
$$

符号不能凭记忆抄写。原约束统一成 $g_i\le0$ 后，最小化问题的 Lagrangian 使用 $+\alpha_i g_i$，并要求 $\alpha_i\ge0$。

### 对原始变量取驻点

对 $w$：

$$
\nabla_w\mathcal L
=w-\sum_i\alpha_i y_i\phi_i=0
\quad\Longrightarrow\quad
\boxed{w=\sum_i\alpha_i y_i\phi_i.}
$$

对 $b$：

$$
\frac{\partial\mathcal L}{\partial b}
=-\sum_i\alpha_i y_i=0
\quad\Longrightarrow\quad
\boxed{\sum_i\alpha_i y_i=0.}
$$

对每个 $\xi_i$：

$$
\frac{\partial\mathcal L}{\partial\xi_i}
=C-\alpha_i-\mu_i=0.
$$

因 $\mu_i\ge0$，得到盒约束

$$
\boxed{0\le\alpha_i\le C.}
$$

这里已经出现三条结构：

1. $w$ 必在训练特征向量张成的空间中；
2. 正负类对偶“质量”必须平衡；
3. 软间隔使每个 $\alpha_i$ 被 $C$ 截住。

<MlQuiz kind="Lagrangian" question="为什么软间隔对偶出现 αᵢ≤C，而硬间隔没有这个上界？" :options="['因为核函数有界', '因为对 ξᵢ 的驻点给出 C−αᵢ−μᵢ=0 且 μᵢ≥0', '因为标签只有两个', '因为 b 被正则化']" :answer="1" explanation="盒上界来自松弛变量及其非负约束；硬间隔原问题没有 ξ 与 CΣξ。" />

## 2. 消去原始变量得到核化入口

记 $\phi_i=\phi(x_i)$。由

$$
w=\sum_i\alpha_i y_i\phi_i
$$

可得

$$
\lVert w\rVert^2
=\sum_i\sum_j
\alpha_i\alpha_j y_i y_j\phi_i^T\phi_j.
$$

Lagrangian 中与 $w$ 相关的线性项是

$$
-\sum_i\alpha_i y_i w^T\phi_i=-w^Tw.
$$

与 $\frac12w^Tw$ 合并后留下 $-\frac12w^Tw$。$b$ 项因 $\sum_i\alpha_i y_i=0$ 消失，$\xi$ 项因 $C-\alpha_i-\mu_i=0$ 消失。对偶函数是原始变量上的下确界，最终得到

$$
\boxed{
\begin{aligned}
\max_\alpha\quad&
D(\alpha)=\sum_i\alpha_i
-\frac12\sum_i\sum_j
\alpha_i\alpha_jy_i y_jK(x_i,x_j)\\
\text{s.t.}\quad&
0\le\alpha_i\le C,\\
&\sum_i\alpha_i y_i=0,
\end{aligned}}
$$

其中

$$
K(x_i,x_j)=\phi(x_i)^T\phi(x_j).
$$

令

$$
Q_{ij}=y_i y_j K(x_i,x_j),
$$

对偶目标简写为

$$
D(\alpha)=\mathbf1^T\alpha-\frac12\alpha^TQ\alpha.
$$

若 $K$ 半正定，则 $Q=\operatorname{diag}(y)K\operatorname{diag}(y)$ 也半正定，所以 $D$ 是凹函数；在凸可行集上最大化它没有虚假的局部峰值。

<SvmDualBalanceBenchLab />

<MlQuiz kind="对偶结构" question="核技巧能够进入 SVM 的直接原因是什么？" :options="['对偶目标和预测只通过 φ(xᵢ)ᵀφ(xⱼ) 使用特征', '原始问题没有约束', '核会自动选择 C', '所有相似度都是概率']" :answer="0" explanation="消元后只剩特征空间内积；有效核可以直接提供这些内积而不显式构造 φ。" />

### 四点对称问题的原始—对偶一致性

考虑正类 $(2,\pm1)$、负类 $(-2,\pm1)$。规范尺度下最优边界为 $x_1=0$，$w=(1/2,0)$。四个点共用 $\alpha=1/16$：

```python
dual_X = np.array([[-2., -1.], [-2., 1.], [2., -1.], [2., 1.]])
dual_y = np.array([-1., -1., 1., 1.])
dual_alpha = np.full(4, 1 / 16)
dual_gram = dual_X @ dual_X.T
dual_weight = (dual_alpha * dual_y) @ dual_X
dual_intercept = 0.
dual_margin = dual_y * (dual_X @ dual_weight + dual_intercept)
dual_objective = dual_alpha.sum() - 0.5 * (
    (dual_alpha * dual_y) @ dual_gram @ (dual_alpha * dual_y)
)
primal_objective = 0.5 * (dual_weight @ dual_weight)

print("w / margins：", dual_weight, dual_margin)
print("原始/对偶目标：", primal_objective, dual_objective)
assert np.allclose(dual_weight, [0.5, 0])
assert np.allclose(dual_margin, 1)
assert abs(dual_alpha @ dual_y) < 1e-12
assert np.isclose(primal_objective, dual_objective)
```

## 3. KKT 条件是支持向量与求解误差的审计协议

软间隔 SVM 是凸问题，并且总能用足够大的 $\xi$ 构造严格可行点；在标准条件下强对偶成立。最优解满足四组 KKT 条件。

### 3.1 原始可行

$$
y_if_i\ge1-\xi_i,
\qquad
\xi_i\ge0.
$$

### 3.2 对偶可行

$$
0\le\alpha_i\le C,
\qquad
\mu_i=C-\alpha_i\ge0.
$$

### 3.3 驻点

$$
w=\sum_i\alpha_i y_i\phi_i,
\qquad
\sum_i\alpha_i y_i=0,
\qquad
C-\alpha_i-\mu_i=0.
$$

### 3.4 互补松弛

$$
\alpha_i[1-\xi_i-y_if_i]=0,
\qquad
\mu_i\xi_i=0.
$$

由此得到支持向量状态：

| $\alpha_i$ | 典型 margin/slack 状态 | 含义 |
|---:|---|---|
| $0$ | $y_if_i\ge1$ | 间隔外，不参与核展开 |
| $0<\alpha_i<C$ | $y_if_i=1,\ \xi_i=0$ | 自由支持向量，贴住间隔 |
| $\alpha_i=C$ | $y_if_i\le1$ | 有界支持向量，可在间隔内或错分 |

退化问题中等号状态可能重叠，因此生产代码应使用容差，不要用浮点严格相等分类。

对任意自由支持向量：

$$
b=y_i-\sum_j\alpha_j y_jK(x_j,x_i).
$$

对多个自由支持向量的 $b$ 取平均，可降低单点浮点误差。

```python
C_demo = 0.20
dual_mu = C_demo - dual_alpha
dual_slack = np.zeros(4)
stationarity_w = dual_weight - (dual_alpha * dual_y) @ dual_X
stationarity_b = dual_alpha @ dual_y
complementarity_margin = dual_alpha * (1 - dual_slack - dual_margin)
complementarity_slack = dual_mu * dual_slack
b_recovered = np.mean(
    dual_y - dual_gram.T @ (dual_alpha * dual_y)
)

print("w/b 驻点残差：", np.linalg.norm(stationarity_w), abs(stationarity_b))
print("两组互补松弛残差：", np.max(np.abs(complementarity_margin)), np.max(np.abs(complementarity_slack)))
print("恢复截距：", b_recovered)
assert np.linalg.norm(stationarity_w) < 1e-12
assert abs(stationarity_b) < 1e-12
assert np.max(np.abs(complementarity_margin)) < 1e-12
assert np.max(np.abs(complementarity_slack)) < 1e-12
assert abs(b_recovered) < 1e-12
```

<div class="ml-stop"><strong>停下来推一推 1</strong>只用 $\alpha_i[1-\xi_i-y_if_i]=0$，能否推出 $0&lt;\alpha_i&lt;C$ 时 $\xi_i=0$？指出还必须使用哪一条互补松弛和哪一个驻点关系。</div>

<MlQuiz kind="KKT 状态" question="若数值解中 0&lt;αᵢ&lt;C，但 yᵢfᵢ=0.73，最合理的判断是什么？" :options="['这是自由支持向量的正常状态', 'KKT 残差明显，求解尚未收敛或 b/核实现有误', '应把它当概率 0.73', '说明 Gram 必为零矩阵']" :answer="1" explanation="自由支持向量应满足 ξ=0 且 y f=1；偏离 1 是直接的最优性审计信号。" />

## 4. 特征映射、核与表示形式

若显式映射为

$$
\phi:\mathcal X\to\mathcal H,
$$

核定义为

$$
\boxed{K(x,z)=\langle\phi(x),\phi(z)\rangle_{\mathcal H}.}
$$

由驻点关系

$$
w=\sum_i\alpha_i y_i\phi(x_i),
$$

预测可完全写成

$$
\boxed{
f(x)=\sum_{i:\alpha_i>0}\alpha_i y_iK(x_i,x)+b.
}
$$

这既是 KKT 推导的直接结果，也是表示定理的一个具体实例：带 RKHS 范数正则的最优函数可以写成训练样本核截面的线性组合。

核技巧**没有**：

- 自动寻找正确相似度；
- 自动减少训练样本；
- 自动产生概率；
- 让高维计算免费。

它只把“构造 $\phi$ 后做内积”的接口换成“直接计算 $K$”。

<MlQuiz kind="表示辨析" question="核 SVM 预测为什么只需要支持向量？" :options="['因为 αᵢ=0 的训练样本在展开式中系数为零', '因为非支持向量没有标签', '因为 Gram 只有一列', '因为支持向量一定是最近邻']" :answer="0" explanation="预测展开是 ΣαᵢyᵢK(xᵢ,x)+b；α=0 的项严格消失。" />

## 5. Gram 半正定是凸性的接口

给定任意有限样本 $x_1,\ldots,x_n$，Gram 矩阵为

$$
K_{ij}=K(x_i,x_j).
$$

若 $K$ 来自内积，则对任意 $c\in\mathbb R^n$：

$$
\begin{aligned}
c^TKc
&=\sum_i\sum_j c_i c_j\phi(x_i)^T\phi(x_j)\\
&=\left\lVert\sum_i c_i\phi(x_i)\right\rVert^2\\
&\ge0.
\end{aligned}
$$

因此 Gram 必须对称半正定。反过来，若一个核函数对**任意有限样本集**都产生对称半正定 Gram，它对应某个 Hilbert 特征空间内积。

重要边界：

- 在一个样本集上检查 PSD 是必要的单元测试，不是对任意输入的证明；
- 有限精度下可能出现 $-10^{-12}$ 量级特征值，应结合矩阵尺度和对称误差判断；
- 明显负特征值会使 $Q$ 不再 PSD，对偶目标不再保证凹；
- 给 Gram 对角线加 jitter 改变了核矩阵，不能用来悄悄掩盖无效相似度。

```python
def linear_kernel(X, Z):
    return X @ Z.T

def polynomial_kernel(X, Z, degree=2, offset=1.0):
    return (offset + X @ Z.T) ** degree

def rbf_kernel(X, Z, gamma):
    X_squared = np.sum(X ** 2, axis=1)[:, None]
    Z_squared = np.sum(Z ** 2, axis=1)[None, :]
    squared_distance = np.maximum(X_squared + Z_squared - 2 * X @ Z.T, 0)
    return np.exp(-gamma * squared_distance)

def quadratic_features(X):
    return np.column_stack([
        np.ones(len(X)),
        np.sqrt(2) * X[:, 0],
        np.sqrt(2) * X[:, 1],
        X[:, 0] ** 2,
        np.sqrt(2) * X[:, 0] * X[:, 1],
        X[:, 1] ** 2,
    ])

gram_explicit = quadratic_features(X_train) @ quadratic_features(X_train).T
gram_implicit = polynomial_kernel(X_train, X_train, degree=2, offset=1)
gram_eigenvalues = np.linalg.eigvalsh((gram_implicit + gram_implicit.T) / 2)

print("显式/隐式 Gram 最大差：", np.max(np.abs(gram_explicit - gram_implicit)))
print("二次核 Gram 最小特征值：", gram_eigenvalues.min())
assert np.allclose(gram_explicit, gram_implicit)
assert gram_eigenvalues.min() > -1e-8
```

对比一个对称但无效的“负内积相似度”：

```python
bad_gram = -linear_kernel(X_train[:20], X_train[:20])
bad_eigenvalues = np.linalg.eigvalsh((bad_gram + bad_gram.T) / 2)
print("负内积 Gram 最小特征值：", bad_eigenvalues.min())
assert np.allclose(bad_gram, bad_gram.T)
assert bad_eigenvalues.min() < -1e-3
```

<KernelFeatureLiftLab />

<MlQuiz kind="核有效性" question="某自定义相似度在当前 50 个样本上的 Gram 对称且最小特征值为 −2×10⁻¹³。可以得出什么？" :options="['已证明它对所有输入都是有效核', '当前矩阵在浮点容差内与 PSD 一致，但仍需解析证明或更强的核构造依据', '它一定无效', '它一定等于 RBF']" :answer="1" explanation="单个有限样本检查只能发现反例，不能证明对所有有限集合都 PSD；极小负值还可能是舍入误差。" />

## 6. 多项式核显式编码交互阶数

非齐次多项式核为

$$
K(x,z)=(c+x^Tz)^d,
\qquad c\ge0.
$$

当 $d=2$、$c=1$、输入二维时：

$$
\begin{aligned}
(1+x^Tz)^2
=&1+2x_1z_1+2x_2z_2\\
&+x_1^2z_1^2
+2x_1x_2z_1z_2
+x_2^2z_2^2.
\end{aligned}
$$

对应一个显式映射

$$
\phi(x)=
[1,\sqrt2x_1,\sqrt2x_2,x_1^2,\sqrt2x_1x_2,x_2^2]^T.
$$

环状分类能利用

$$
r^2=x_1^2+x_2^2
$$

在提升空间中形成线性阈值。次数 $d$、offset $c$ 和输入尺度共同决定包含哪些低/高阶项及其相对权重。高次数不是“更精确”，而是更强的交互先验与更差的数值范围。

<MlQuiz kind="显式映射" question="二维二次核能把环状边界线性化，关键新增了哪类坐标？" :options="['只有常数 1', 'x₁²、x₁x₂、x₂² 等二次坐标', '样本编号', '类别概率']" :answer="1" explanation="圆或椭圆边界在二次特征中是线性方程；这不是原空间的一条直线。" />

## 7. RBF 核把尺度变成局部相似度

RBF 核写成

$$
K_\gamma(x,z)=
\exp(-\gamma\lVert x-z\rVert^2),
\qquad \gamma>0.
$$

也常写成

$$
\exp\left(-\frac{\lVert x-z\rVert^2}{2\sigma^2}\right),
\qquad
\gamma=\frac1{2\sigma^2}.
$$

其隐式特征空间无限维，但单次核计算只需 $O(d)$。$\gamma$ 的几何意义：

- $\gamma$ 小：相似度衰减慢，每个支持向量影响范围宽，边界更平滑；
- $\gamma$ 大：相似度衰减快，影响更局部，容易围绕训练点形成复杂边界；
- 当 $K=1/2$ 时，半相似半径为

$$
r_{1/2}=\sqrt{\frac{\log2}{\gamma}}.
$$

若把所有标准化坐标再乘 $a$：

$$
\exp[-\gamma\lVert a(x-z)\rVert^2]
=\exp[-(a^2\gamma)\lVert x-z\rVert^2].
$$

所以特征尺度和 $\gamma$ 不可分开讨论。

```python
gamma = 0.8
probe_X = X_train[:30]
rbf_original = rbf_kernel(probe_X, probe_X, gamma)
scale_factor = 3.0
rbf_scaled = rbf_kernel(scale_factor * probe_X, scale_factor * probe_X, gamma)
rbf_equivalent = rbf_kernel(probe_X, probe_X, gamma * scale_factor ** 2)
half_similarity_radius = np.sqrt(np.log(2) / gamma)

print("缩放/γ 等价最大差：", np.max(np.abs(rbf_scaled - rbf_equivalent)))
print("半相似半径：", half_similarity_radius)
assert np.allclose(rbf_scaled, rbf_equivalent)
assert np.allclose(np.diag(rbf_original), 1)
```

<div class="ml-stop"><strong>停下来推一推 2</strong>若某一传感器方向的标准差被低估 10 倍，RBF 距离和相似度会怎样偏移？为什么单独调小全局 γ 不能精确修复各向异性尺度错误？</div>

<MlQuiz kind="RBF 尺度" question="保持数据不变，把 γ 从 0.2 增至 2，哪项通常成立？" :options="['半相似半径增大', '相似度衰减更快，单个支持向量影响更局部', 'Gram 自动变成单位矩阵且永远不过拟合', 'C 自动减小']" :answer="1" explanation="r₁/₂=√(log2/γ)，γ 增大使有效邻域缩小；是否过拟合仍取决于 C、样本和尺度。" />

## 8. 用简化 SMO 直接求解核 SVM

核 SVM 的对偶有标签平衡等式和盒约束。Sequential Minimal Optimization（SMO）每次只更新两个 $\alpha$，从而保持

$$
\sum_i\alpha_i y_i=0.
$$

下面的教学实现采用确定性最大误差差异选点、误差缓存和 KKT 停止轮数。它用于验证公式，不替代成熟库在收敛、工作集、缓存、容差和大规模优化上的实现。

```python
def fit_smo(K, y, C, tolerance=1e-3, maximum_passes=15, maximum_sweeps=1000):
    n = len(y)
    alpha = np.zeros(n)
    intercept = 0.0
    decision = np.zeros(n)
    passes_without_change = 0
    sweeps = 0
    while passes_without_change < maximum_passes and sweeps < maximum_sweeps:
        changed = 0
        for i in range(n):
            error_i = decision[i] - y[i]
            violates = (
                (y[i] * error_i < -tolerance and alpha[i] < C - 1e-12)
                or (y[i] * error_i > tolerance and alpha[i] > 1e-12)
            )
            if not violates:
                continue
            errors = decision - y
            candidate_order = np.argsort(-np.abs(error_i - errors))
            for j in candidate_order:
                if j == i:
                    continue
                error_j = errors[j]
                old_i, old_j, old_b = alpha[i], alpha[j], intercept

                if y[i] != y[j]:
                    lower = max(0.0, old_j - old_i)
                    upper = min(C, C + old_j - old_i)
                else:
                    lower = max(0.0, old_i + old_j - C)
                    upper = min(C, old_i + old_j)
                if upper - lower < 1e-12:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= -1e-14:
                    continue
                new_j = old_j - y[j] * (error_i - error_j) / eta
                new_j = float(np.clip(new_j, lower, upper))
                if abs(new_j - old_j) < 1e-7:
                    continue
                new_i = old_i + y[i] * y[j] * (old_j - new_j)

                b1 = (
                    old_b - error_i
                    - y[i] * (new_i - old_i) * K[i, i]
                    - y[j] * (new_j - old_j) * K[i, j]
                )
                b2 = (
                    old_b - error_j
                    - y[i] * (new_i - old_i) * K[i, j]
                    - y[j] * (new_j - old_j) * K[j, j]
                )
                if 1e-10 < new_i < C - 1e-10:
                    intercept = b1
                elif 1e-10 < new_j < C - 1e-10:
                    intercept = b2
                else:
                    intercept = 0.5 * (b1 + b2)

                alpha[i], alpha[j] = new_i, new_j
                decision += (
                    y[i] * (new_i - old_i) * K[:, i]
                    + y[j] * (new_j - old_j) * K[:, j]
                    + (intercept - old_b)
                )
                changed += 1
                break
        sweeps += 1
        passes_without_change = passes_without_change + 1 if changed == 0 else 0
    return alpha, intercept, {
        "sweeps": sweeps,
        "passes": passes_without_change,
        "converged": passes_without_change >= maximum_passes,
    }

linear_gram_train = linear_kernel(X_train, X_train)
linear_alpha, linear_b, linear_audit = fit_smo(
    linear_gram_train, y_train, C=1.0
)
linear_score_train = linear_gram_train @ (linear_alpha * y_train) + linear_b
print("线性核 sweeps/支持向量/训练准确率：", linear_audit["sweeps"], np.sum(linear_alpha > 1e-6), np.mean(y_train * linear_score_train > 0))
assert abs(linear_alpha @ y_train) < 1e-7
assert np.all((linear_alpha >= -1e-12) & (linear_alpha <= 1 + 1e-12))
```

## 9. C 与核参数必须在同一验证协议中选择

$C$ 控制违约代价，核参数控制特征空间几何。二者强耦合：

- 高 $\gamma$ 的 RBF 更局部，通常需要更强正则化来限制记忆；
- 高次多项式扩大高阶坐标动态范围，C 的有效尺度随之改变；
- 先用全数据选 $\gamma$、再“单独验证 C”仍会泄漏模型选择。

下面在训练批次拟合 6 个候选，仅用验证批次选择；测试只用于冻结候选的一次最终验收。

```python
def balanced_accuracy(y, score):
    prediction = np.where(score >= 0, 1., -1.)
    positive_recall = np.mean(prediction[y == 1] == 1)
    negative_recall = np.mean(prediction[y == -1] == -1)
    return 0.5 * (positive_recall + negative_recall)

kernel_candidates = [
    ("linear", {"C": 1.0}),
    ("poly2", {"C": 0.5, "degree": 2}),
    ("poly2", {"C": 2.0, "degree": 2}),
    ("rbf", {"C": 0.5, "gamma": 0.35}),
    ("rbf", {"C": 1.0, "gamma": 0.8}),
    ("rbf", {"C": 2.0, "gamma": 1.6}),
]

def kernel_from_spec(name, parameters, X, Z):
    if name == "linear":
        return linear_kernel(X, Z)
    if name == "poly2":
        return polynomial_kernel(X, Z, degree=parameters["degree"], offset=1.0)
    return rbf_kernel(X, Z, parameters["gamma"])

selection_rows = []
fitted_candidates = []
for name, parameters in kernel_candidates:
    K_train = kernel_from_spec(name, parameters, X_train, X_train)
    alpha, intercept, audit = fit_smo(K_train, y_train, C=parameters["C"])
    K_validation = kernel_from_spec(name, parameters, X_validation, X_train)
    validation_score = K_validation @ (alpha * y_train) + intercept
    score = balanced_accuracy(y_validation, validation_score)
    eligible_score = score if audit["converged"] else -np.inf
    selection_rows.append((
        name, parameters, eligible_score, np.sum(alpha > 1e-6),
        audit["sweeps"], audit["converged"]
    ))
    fitted_candidates.append((name, parameters, alpha, intercept, audit))

for row in selection_rows:
    print(row)
selected_index = int(np.argmax([row[2] for row in selection_rows]))
selected_name, selected_parameters, selected_alpha, selected_b, selected_audit = fitted_candidates[selected_index]
print("冻结选择：", selected_name, selected_parameters)
assert selected_audit["converged"]
```

```python
K_test_selected = kernel_from_spec(
    selected_name, selected_parameters, X_test, X_train
)
test_score = K_test_selected @ (selected_alpha * y_train) + selected_b
test_balanced_accuracy = balanced_accuracy(y_test, test_score)
test_prediction = np.where(test_score >= 0, 1., -1.)
test_tp = int(np.sum((y_test == 1) & (test_prediction == 1)))
test_fp = int(np.sum((y_test == -1) & (test_prediction == 1)))
test_tn = int(np.sum((y_test == -1) & (test_prediction == -1)))
test_fn = int(np.sum((y_test == 1) & (test_prediction == -1)))
selected_support = selected_alpha > 1e-6

print("测试 TP/FP/TN/FN：", test_tp, test_fp, test_tn, test_fn)
print("测试 balanced accuracy：", round(test_balanced_accuracy, 4))
print("支持向量：", selected_support.sum(), "/", len(y_train))
```

<MlQuiz kind="选择协议" question="RBF γ 和 C 应如何调参？" :options="['先在测试集选 γ，再在训练集选 C', '在同一个仅属于模型选择的验证协议中联合搜索，测试集只验收冻结组合', 'γ 固定越大越好', 'C 与 γ 独立，可分别看测试集']" :answer="1" explanation="γ 改变特征空间局部性，C 改变违约惩罚，二者耦合；测试集不能参与候选选择。" />

## 10. 用 KKT 与原始—对偶间隙审计 SMO

对冻结模型，在训练 Gram 上计算

$$
f=K(\alpha\odot y)+b,
\qquad
m_i=y_if_i.
$$

RKHS 范数满足

$$
\lVert w\rVert_{\mathcal H}^2
=(\alpha\odot y)^TK(\alpha\odot y).
$$

于是

$$
P=\frac12\lVert w\rVert_{\mathcal H}^2
+C\sum_i\max(0,1-m_i),
$$

$$
D=\mathbf1^T\alpha
-\frac12\lVert w\rVert_{\mathcal H}^2.
$$

弱对偶保证 $P-D\ge0$；收敛时 gap 应趋近 0。还应分别检查：

- $\alpha\approx0$ 时 $m_i\ge1$；
- $0<\alpha_i<C$ 时 $m_i\approx1$；
- $\alpha_i\approx C$ 时 $m_i\le1$；
- $|\alpha^Ty|$ 足够小。

```python
K_train_selected = kernel_from_spec(
    selected_name, selected_parameters, X_train, X_train
)
selected_C = selected_parameters["C"]
train_score = K_train_selected @ (selected_alpha * y_train) + selected_b
train_margin = y_train * train_score
coefficient = selected_alpha * y_train
rkhs_norm_squared = coefficient @ K_train_selected @ coefficient
primal_value = 0.5 * rkhs_norm_squared + selected_C * np.maximum(0, 1 - train_margin).sum()
dual_value = selected_alpha.sum() - 0.5 * rkhs_norm_squared
duality_gap = primal_value - dual_value

zero_mask = selected_alpha <= 1e-5
free_mask = (selected_alpha > 1e-5) & (selected_alpha < selected_C - 1e-5)
cap_mask = selected_alpha >= selected_C - 1e-5
zero_violation = np.maximum(0, 1 - train_margin[zero_mask]).max(initial=0)
free_violation = np.abs(train_margin[free_mask] - 1).max(initial=0)
cap_violation = np.maximum(0, train_margin[cap_mask] - 1).max(initial=0)

print("原始/对偶/gap：", round(primal_value, 5), round(dual_value, 5), round(duality_gap, 5))
print("KKT 分层最大残差：", zero_violation, free_violation, cap_violation)
print("标签平衡残差：", abs(selected_alpha @ y_train))
assert duality_gap > -1e-6
assert duality_gap < 1e-2
assert max(zero_violation, free_violation, cap_violation) < 2e-3
assert abs(selected_alpha @ y_train) < 1e-5
```

有限容差下的 SMO 仍会留下非零 gap，所以本例把候选的收敛状态、gap 小于 `10⁻²` 和分层 KKT 残差小于 `2×10⁻³` 都写进可执行门禁。生产实现还应锁定求解器版本、停止容差、最大迭代和缓存精度，并拒绝悄悄接受未收敛模型。

<MlQuiz kind="求解审计" question="训练准确率已经 100%，但原始—对偶 gap 很大。能否据此宣称优化完成？" :options="['能，准确率就是 KKT', '不能；分类正确不证明 margin、α、盒约束和目标已达到最优', '只要支持向量多即可', '把 gap 取绝对值后忽略']" :answer="1" explanation="0-1 结果只看符号；SVM 最优性还要求 KKT 与 primal-dual gap 收敛。" />

## 11. 核参数也决定概率与开放集边界

核 SVM 输出

$$
f(x)=\sum_{i\in SV}\alpha_i y_iK(x_i,x)+b
$$

仍不是 log odds。即使 RBF 相似度位于 $[0,1]$，加权和 $f(x)$ 也不是概率。若需要概率，必须用独立校准集拟合映射并验收 NLL、Brier、可靠性与先验漂移。

RBF 还存在开放空间风险：离所有支持向量很远时，所有核值趋近 0，分数趋近 $b$。模型可能因截距把完全陌生样本稳定归到某一类。必须额外监控：

- 到训练分布的距离或最大核相似度；
- 支持向量覆盖范围；
- 新批次 kernel similarity 分布；
- 明确的未知/拒绝路径。

<MlQuiz kind="开放集" question="RBF 输入远离所有支持向量时，f(x) 通常趋近什么？" :options="['正类概率 1', '截距 b，因为各 K(xᵢ,x) 趋近 0', '训练准确率', 'Gram 的迹']" :answer="1" explanation="远离训练支撑时核展开项衰减，剩下 b；这不等于模型知道该样本属于哪类。" />

## 12. 计算、内存与显式近似

设训练样本 $n$、原始维度 $d$、支持向量数 $n_{SV}$：

| 环节 | 时间/空间 | 工程含义 |
|---|---:|---|
| 完整 Gram 构造 | $O(n^2d)$ | 核计算可并行，但数据传输与缓存昂贵 |
| Gram 内存 | $O(n^2)$ | float64 在 $n=100000$ 时约 80 GB |
| 通用核 SVM 训练 | 通常至少二次级 | 强依赖求解器、缓存与可分性 |
| 单样本预测 | $O(n_{SV}d)$ | 支持向量接近全部样本时延迟失控 |
| 模型存储 | $O(n_{SV}d)$ | 还需保存 α、标签、缩放与核参数 |

核技巧只避免显式特征维度 $D$，没有避免样本维度 $n$。

### 随机 Fourier 特征近似 RBF

对

$$
K(x,z)=\exp(-\gamma\lVert x-z\rVert^2),
$$

采样

$$
\omega_j\sim\mathcal N(0,2\gamma I),
\qquad
b_j\sim\operatorname{Unif}(0,2\pi),
$$

构造

$$
z_j(x)=\sqrt{\frac2m}\cos(\omega_j^Tx+b_j).
$$

则 $z(x)^Tz(z)$ 近似 RBF 核。之后可训练线性模型，把存储从 $O(n^2)$ 转为 $O(nm)$ 数据变换与 $O(m)$ 参数，但引入随机近似误差。

```python
rff_rng = np.random.default_rng(1205)
rff_dimension = 800
rff_gamma = 0.8
frequencies = rff_rng.normal(
    0, np.sqrt(2 * rff_gamma), size=(X_train.shape[1], rff_dimension)
)
phases = rff_rng.uniform(0, 2 * np.pi, rff_dimension)

def random_fourier_features(X, frequencies, phases):
    return np.sqrt(2 / frequencies.shape[1]) * np.cos(X @ frequencies + phases)

rff_probe = X_train[:80]
exact_probe_gram = rbf_kernel(rff_probe, rff_probe, rff_gamma)
features_probe = random_fourier_features(rff_probe, frequencies, phases)
approximate_probe_gram = features_probe @ features_probe.T
relative_gram_error = np.linalg.norm(
    exact_probe_gram - approximate_probe_gram
) / np.linalg.norm(exact_probe_gram)

gram_bytes_100k = 100_000 ** 2 * 8
print("100k float64 Gram GB：", gram_bytes_100k / 1e9)
print("RFF 相对 Gram 误差：", round(relative_gram_error, 4))
assert relative_gram_error < 0.25
```

近似维度 $m$ 必须进入验证协议；不能只报告内存下降。还要固定随机种子、误差容差、线性求解器、延迟和任务指标。

<div class="ml-stop"><strong>停下来推一推 3</strong>若 $n_{SV}$ 从 8% 上升到 92%，训练准确率不变，核 SVM 的模型大小、单样本延迟和开放集覆盖会怎样变化？哪些量应触发上线阻断？</div>

<MlQuiz kind="复杂度" question="核技巧主要消除了哪项成本，又没有消除哪项成本？" :options="['消除标签；保留概率', '避免显式形成高维 φ；仍可能需要 n×n Gram 与逐支持向量预测', '消除所有训练成本', '只消除截距']" :answer="1" explanation="核计算绕开显式 D 维特征，但样本两两关系和支持向量展开仍可能很贵。" />

## 数值验证：从 Gram 到冻结测试

本章实现至少检查：

1. 原始—对偶解析小例的目标一致；
2. $\alpha^Ty=0$ 与 $0\le\alpha\le C$；
3. 自由/有界/零系数支持向量的 KKT 分层；
4. 显式二次映射与多项式核 Gram 一致；
5. 有效 Gram 的对称误差与最小特征值；
6. 无效相似度能产生明确负特征值反例；
7. 特征尺度与 RBF $\gamma$ 的等价关系；
8. C/核参数只用验证批次选择；
9. 测试只验收冻结候选；
10. RFF 近似误差、随机种子和维度可复现。

```python
selected_gram_eigenvalues = np.linalg.eigvalsh(
    (K_train_selected + K_train_selected.T) / 2
)
selected_train_prediction = np.where(train_score >= 0, 1., -1.)

assert np.all(np.isfinite(K_train_selected))
assert np.allclose(K_train_selected, K_train_selected.T, atol=1e-10)
assert selected_gram_eigenvalues.min() > -1e-7
assert np.all((selected_alpha >= -1e-8) & (selected_alpha <= selected_C + 1e-8))
assert np.all(np.isfinite(test_score))
assert np.all(np.isin(selected_train_prediction, [-1, 1]))
assert set(train_lots).isdisjoint(validation_lots)
assert set(train_lots).isdisjoint(test_lots)
assert set(validation_lots).isdisjoint(test_lots)
print("Gram、盒约束、标签平衡、批次隔离与冻结测试检查通过。")
```

## 复杂度与工程边界

### 工业边界

- **结构先验错误**：核越灵活，越能拟合错误相似度。
- **尺度与 γ 耦合**：单位、标准化或 ARD 尺度变化会重写局部邻域。
- **PSD 责任**：自定义核需要解析构造依据和有限样本反例测试。
- **求解容差**：准确率不能替代 KKT、gap 和收敛状态。
- **规模边界**：Gram、支持向量存储和预测延迟必须压测。
- **概率边界**：decision function 不得冒充概率。
- **开放集边界**：远离支持向量时趋近 b，需要独立拒绝机制。
- **版本边界**：核、C、γ、缩放、支持向量和校准器必须共同版本化。

## 13. 常见失效模式

| 现象 | 根因候选 | 优先证据 | 处理 |
|---|---|---|---|
| 训练很好、跨批次崩溃 | γ 过大或批次泄漏 | 分组验证、支持向量批次分布 | 降低局部性并重做分组选择 |
| Gram 明显负特征值 | 自定义相似度非 PSD 或实现不对称 | 对称误差、最小特征向量 | 修正核构造，不靠 jitter 掩盖 |
| KKT gap 长期不降 | 求解器未收敛、缓存或 b 错误 | α 分层残差、原始/对偶目标 | 阻断模型并修复求解流程 |
| 支持向量接近全部样本 | 重叠、C/γ 不当、错标或漂移 | margin、α、批次与标签审计 | 联合调参并复核数据合同 |
| 推理延迟突然升高 | 支持向量数或核成本上升 | nSV、P95 延迟、模型大小 | 预算约束、显式近似或线性路线 |
| 换单位后边界碎裂 | 缩放与 γ 合同断裂 | 单位版本、核距离分布 | 恢复训练变换并重新验收 |
| 远端未知样本被稳定放行 | RBF 项衰减后由 b 主导 | 最大核相似度、OOD 集 | 增加拒绝/硬规则路径 |

排查顺序：

1. 核对标签、批次、特征截止、单位和缩放；
2. 核对核公式、参数、Gram 对称与 PSD；
3. 核对 α 盒约束、标签平衡、KKT 和 gap；
4. 核对 C/核参数的分组选择与测试冻结；
5. 核对支持向量数量、批次组成、延迟和开放集覆盖；
6. 最后才增加核复杂度或近似维度。

## 14. 上线验收清单

- [ ] 环状/局部相似度先验有工艺依据，而非只来自训练散点。
- [ ] 标签、分组、单位、标准化与未知类路径已冻结。
- [ ] Lagrangian 符号、驻点、盒约束和标签平衡已推导核对。
- [ ] 解析小例通过原始—对偶目标一致性验证。
- [ ] 自定义核有 PSD 构造依据，并有反例/边界单元测试。
- [ ] Polynomial/RBF 参数与 C 在同一分组验证协议中选择。
- [ ] 求解器版本、容差、最大迭代、gap 和收敛状态已锁定。
- [ ] α=0、自由 α、α=C 三类 KKT 残差均达标。
- [ ] 测试集只对冻结模型做一次验收。
- [ ] 支持向量数量、模型大小、P95/P99 延迟满足预算。
- [ ] decision function 未未经校准冒充概率。
- [ ] 低最大核相似度和远端未知样本有拒绝路径。
- [ ] RFF/Nyström 等近似若启用，误差、种子和维度已验收。

## 15. 章节练习

### 练习 A：逐行推导软间隔对偶

从原始问题开始，要求：

1. 把两组不等式统一写成 $g\le0$；
2. 写出含 $\alpha,\mu$ 的 Lagrangian；
3. 分别对 $w,b,\xi$ 取驻点；
4. 解释为什么 $0\le\alpha_i\le C$；
5. 代回并化简二次项；
6. 写出 $Q=\operatorname{diag}(y)K\operatorname{diag}(y)$；
7. 证明 $K\succeq0\Rightarrow Q\succeq0$。

### 练习 B：手算四点 KKT

对本章四点对称数据：

1. 推导共同 $\alpha$ 的一维对偶目标；
2. 求无盒约束最优点；
3. 分别讨论 $C<1/16$ 与 $C\ge1/16$；
4. 恢复 $w,b$；
5. 检查原始/对偶可行、驻点与两组互补松弛；
6. 计算 primal-dual gap。

### 练习 C：实现可审计核 SVM

补全下面的求解器。要求支持核缓存、工作集选择、标签平衡、盒投影、b 恢复、KKT 分层、gap、收敛状态、冻结缩放与预测预算。

```python
# TODO: 练习骨架，验证脚本会跳过本代码块。
def auditable_kernel_svm_fit(X, y, kernel, C):
    alpha = np.zeros(len(y))
    intercept = 0.0
    audit_rows = []
    # 1. 构造并审计 Gram 的对称性与 PSD
    # 2. 成对更新 alpha，保持 y.T @ alpha = 0
    # 3. 用自由支持向量恢复 b
    # 4. 记录 KKT 分层、gap、支持向量数和停止原因
    return alpha, intercept, audit_rows
```

### 练习 D：设计 100k 样本路线

针对 $n=100000,d=40$ 的终检数据，比较：

- 完整 RBF SVM；
- 随机 Fourier 特征 + 线性 SVM；
- Nyström 近似；
- 固定非线性特征 + 线性模型。

必须给出训练时间、峰值内存、模型大小、单样本与批量延迟、近似误差、支持向量/特征维度、重训频率和开放集策略。

<MlQuiz kind="综合验收" question="核 SVM 在测试集 balanced accuracy 很高，是否足以证明工业可用？" :options="['足够', '不够；还需证明相似度先验、PSD、KKT/gap、分组选择、支持向量与延迟预算、概率和开放集边界', '只需 γ 大于 1', '只需 Gram 对角线为 1']" :answer="1" explanation="任务指标不能替代核有效性、求解最优性、数据合同、计算预算和未知样本处置。" />

## 16. 本章结论

核 SVM 的理论闭环是：

$$
\boxed{
\text{软间隔原始问题}
\Rightarrow
\text{Lagrangian/KKT}
\Rightarrow
\text{只剩特征内积的对偶}
\Rightarrow
\text{PSD 核}
\Rightarrow
\text{支持向量核展开}
}
$$

必须记住：

1. **对偶是同一优化问题的另一组坐标**，不是额外模型；
2. **KKT 是可执行的最优性协议**，准确率不能替代；
3. **核是特征空间内积接口**，不是任意相似度；
4. **Polynomial 与 RBF 编码不同先验**，必须和尺度、C 联合验证；
5. **无限维不等于免费**，Gram 与支持向量仍决定成本；
6. **核分数不是概率**，远端样本还需要开放集拒绝。

下一章进入神经网络模块：从线性层、激活函数与复合表示推导多层感知机，并用决策区域和隐藏单元可视化说明它怎样学习特征而不只选择一个固定核。

## 来源与许可

- Stanford CS229 Lecture Notes，PDF 第 54–59 页，§5.4：显式特征映射、Polynomial/Gaussian 核、kernel matrix、对称半正定与 Mercer 条件。用于核接口、PSD 证明和计算边界核对；不复用手写数字或蛋白质示例、原文图形或措辞。
- Stanford CS229 Lecture Notes，PDF 第 66–74 页，§6.5–6.7：Lagrange 对偶、KKT、SVM 对偶、支持向量、核化入口与不可分情形。用于对偶和 KKT 推导交叉核对；本章案例、SMO、题目和实验独立创作。
- Jonathan Richard Shewchuk，*Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 169–173 页，Bonus Lecture B：显式高阶特征、kernelization、Polynomial/Gaussian 核与原始/对偶计算成本。用于复杂度和表示边界核对；不复制课堂图形、示例或表达。
- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 597–603 页，§17.1–17.1.2.3：Mercer kernel、Gram、特征表示、RBF/ARD/Polynomial 与核组合规则。用于术语、PSD 与核构造核对；不复用原书图表或数据。
- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 617–621 页，§17.3.2–17.3.5：SVM 对偶、KKT、软间隔、核技巧、预测展开与概率边界。用于公式和工程边界交叉核对；不复用原文表达或图形。

本章 360 只滚子、30 个热处理批次、环状特征与争议标签、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实轴承缺陷分布、核参数、复检能力、质量成本、放行标准或安全规范。

<MlLessonProgress lesson-id="12" next-label="第 13 章：多层感知机（MLP）" next-href="/courses/machine-learning/13-mlp" />
