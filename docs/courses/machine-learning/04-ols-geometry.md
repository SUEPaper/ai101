---
course: machine-learning
edition: industrial
title: 普通最小二乘法（OLS）
description: 以空压站小时能耗基线标定为贯穿案例，从列空间投影完整推导正规方程、帽子矩阵与伪逆，解释秩亏和近共线下的可辨识性，并建立 QR、SVD、杠杆值与残差诊断的工程实现规范。
outline: [2, 3]
---

<MlLessonHero number="04" title="普通最小二乘法（OLS）" question="平方误差最小为什么等价于正交投影；当两只传感器几乎记录同一件事时，模型究竟识别出了什么？" duration="建议 420–540 分钟" :outcomes="['把业务样本、设计矩阵、参数和预测的形状与单位写成可审计契约', '从最短距离和矩阵微分两条路径推导正规方程', '证明投影矩阵的对称、幂等与残差正交性质', '区分最优预测唯一、系数唯一和系数可解释三个问题', '用紧致 SVD 推导伪逆与最小范数解', '使用 lstsq、QR 和杠杆值诊断构建数值稳定的 OLS 基线']" />

第 03 章解决“怎样评价一个训练流程”。本章固定平方损失，把注意力收回到一个基础却常被误用的问题：线性模型到底在求什么解？

普通最小二乘法不是一句 `fit()`，也不只是公式 $(X^TX)^{-1}X^Ty$。工业实现必须同时回答四件事：可表达的预测空间是什么、解是否唯一、计算是否稳定、残差还能暴露哪些模型失配。

## 贯穿案例：空压站小时能耗基线标定

一家工厂希望给空压站建立小时能耗基线。基线不是直接控制压缩机，而是在每小时结束后，用该小时的产气量、环境温度和供气压力解释合理能耗；实际能耗显著高于基线时，再结合泄漏巡检和设备状态调查原因。

本章所有工况与数值均为**教学合成数据**，不代表任何真实工厂。特征与目标来自同一小时，因此这个模型只能用于事后基线核算；若改成提前一小时调度，所有特征都必须在预测时刻之前可得，数据契约也要重写。

### 先冻结样本语义与单位

| 符号 | 每行含义 | 单位 | 参考点 | 是否进入模型 |
|---|---|---:|---:|---|
| $q_i$ | 第 $i$ 小时标准产气量 | 千标方/小时 | 33 | 是，使用 $q_i-33$ |
| $t_i$ | 小时平均环境温度 | °C | 23 | 是，使用 $t_i-23$ |
| $p_i$ | 小时平均供气压力 | bar | 6.8 | 是，使用 $p_i-6.8$ |
| $y_i$ | 同小时总电耗 | kWh | — | 目标 |
| 时间戳 | 自然小时结束时刻 | — | — | 用于切分与追溯，不直接拟合 |

把连续量减去有工程含义的参考工况后，截距 $\beta_0$ 表示“产气量 33、温度 23°C、压力 6.8 bar 时的基线能耗”。中心化不是 OLS 的必要条件，但会让截距更可读，并减弱截距列与连续特征的数值耦合。

```python
import numpy as np

rng = np.random.default_rng(20260805)
n_hours = 240

throughput = rng.uniform(18.0, 48.0, n_hours)       # 千标方/小时
ambient_c = rng.uniform(10.0, 36.0, n_hours)        # °C
pressure_bar = rng.normal(6.8, 0.24, n_hours)       # bar

X = np.column_stack([
    np.ones(n_hours),
    throughput - 33.0,
    ambient_c - 23.0,
    pressure_bar - 6.8,
])
beta_process = np.array([580.0, 7.8, 4.1, 32.0])
y = X @ beta_process + rng.normal(0.0, 15.0, n_hours)

assert X.shape == (n_hours, 4)
assert y.shape == (n_hours,)
assert np.allclose(X[:, 0], 1.0)
assert np.isfinite(X).all() and np.isfinite(y).all()
```

这里的噪声只是为了生成可复现实验，不是在本章先验承诺真实能耗服从高斯分布。OLS 的几何解只需要平方损失；概率假设、标准误与统计推断留到第 05 章。

<MlQuiz kind="数据契约" question="当前模型使用同小时平均压力解释同小时电耗。团队现在要在小时开始前预测并调度，最关键的修改是什么？" :options="['保留数据不变，只把模型改名为预测器', '重新定义特征截止时刻，只使用调度时已经可得的信息', '删除截距即可避免泄漏', '把训练误差乘以 60']" :answer="1" explanation="事后解释与事前预测拥有不同的信息边界。同小时结束后才能汇总的压力不能直接用于小时开始前的决策；必须先重写样本和特征截止契约。" />

## 1. 设计矩阵规定模型能够说什么

令 $X\in\mathbb R^{n\times p}$，每一行对应一个小时，每一列对应一个基函数；$\beta\in\mathbb R^p$，$y\in\mathbb R^n$。批量预测为

$$
\widehat y=X\beta\in\mathbb R^n.
$$

本案例 $p=4$。第一列全为 1，其余三列分别是相对参考工况的产气量、温度和压力。第 $i$ 个预测展开为

$$
\widehat y_i
=\beta_0
+\beta_q(q_i-33)
+\beta_t(t_i-23)
+\beta_p(p_i-6.8).
$$

“线性”指参数 $\beta$ 线性出现，不要求输入只能原样进入。若将 $(t_i-23)^2$ 加为一列，模型对温度是二次函数，对扩展后的参数仍是线性模型。第 07 章会系统研究这种基函数扩展。

所有可能预测构成

$$
\mathcal C(X)
=\{X\beta:\beta\in\mathbb R^p\},
$$

即 $X$ 的列空间。训练不是在整个 $\mathbb R^n$ 中任意选择 $\widehat y$，而只能在 $\mathcal C(X)$ 中移动。增加一列特征，真正改变的是可表达预测子空间；若新列已经能由旧列线性组合出来，列空间没有扩大。

普通最小二乘法求解

$$
\widehat\beta
\in\arg\min_{\beta\in\mathbb R^p}
\operatorname{RSS}(\beta),
\qquad
\operatorname{RSS}(\beta)=\lVert y-X\beta\rVert_2^2.
$$

除以 $n$ 得到 MSE，不改变最优解；乘以 $1/2$ 也只会让梯度少一个常数 2。必须在报告里说清楚使用 RSS、MSE 还是 RMSE，因为它们的数值和单位不同。

<MlQuiz kind="形状检查" question="X 的形状是 n×p，y 是长度 n 的向量。为了让 Xβ 与 y 对齐，β 必须是什么形状？" :options="['长度 n', '长度 p', 'n×p', 'p×n']" :answer="1" explanation="X 的 p 列与 β 的 p 个系数收缩，结果是每个样本一个预测的长度 n 向量。" />

## 2. 最近点为什么产生残差正交

定义最优预测 $\widehat y=X\widehat\beta$，残差

$$
r=y-\widehat y.
$$

假设存在某个列空间方向 $Xv$ 与残差不正交，即 $r^TXv\ne0$。沿参数方向 $v$ 移动 $a$：

$$
\begin{aligned}
g(a)
&=\lVert y-X(\widehat\beta+av)\rVert_2^2\\
&=\lVert r-aXv\rVert_2^2\\
&=r^Tr-2a\,r^TXv+a^2\lVert Xv\rVert_2^2.
\end{aligned}
$$

在 $a=0$ 处，$g'(0)=-2r^TXv$。若它不为零，总能选择 $a$ 的符号让 RSS 下降，与 $\widehat\beta$ 最优矛盾。因此

$$
r^TXv=0\quad\text{对所有 }v\in\mathbb R^p.
$$

这等价于

$$
\boxed{X^T(y-X\widehat\beta)=0}.
$$

残差不是只与某一列碰巧正交，而是与 $X$ 列空间中的每个方向都正交。于是 $\widehat y$ 是 $y$ 到 $\mathcal C(X)$ 的正交投影。

<ColumnSpaceProjectionLab />

### 勾股分解需要正交条件

因为 $y=\widehat y+r$ 且 $\widehat y\perp r$，

$$
\lVert y\rVert_2^2
=\lVert\widehat y\rVert_2^2+\lVert r\rVert_2^2.
$$

当模型包含截距时，$r$ 与全 1 列正交，所以

$$
\sum_{i=1}^n r_i=0,
\qquad
\overline{\widehat y}=\bar y.
$$

此时对中心化目标还能写出常见的平方和分解：

$$
\underbrace{\sum_i(y_i-\bar y)^2}_{\mathrm{TSS}}
=
\underbrace{\sum_i(\widehat y_i-\bar y)^2}_{\mathrm{ESS}}
+
\underbrace{\sum_i(y_i-\widehat y_i)^2}_{\mathrm{RSS}}.
$$

若没有截距，训练残差和不必为零，以上围绕 $\bar y$ 的 TSS 分解也不能直接照搬；这就是无截距模型的 $R^2$ 容易被误解的原因之一。

<div class="ml-stop"><strong>停下来推一推 1</strong>把 $X$ 的第一列设为全 1。将残差正交式 $X^Tr=0$ 的第一个分量写开，解释为什么“训练残差均值为零”是带截距 OLS 的代数结果，而不是模型无偏的总体证据。</div>

<MlQuiz kind="推导补全" question="带截距 OLS 的训练残差均值为 0，能推出未来每种工况下误差条件均值也为 0 吗？" :options="['能，这是同一个命题', '不能；前者是有限训练样本上的正交条件，后者是关于总体机制的统计假设', '只有样本数为偶数时能', '只要 RSS 大于 0 就能']" :answer="1" explanation="训练残差与设计矩阵正交是求解器产生的代数事实。总体条件均值是否为零要靠抽样设计、模型设定和独立数据检验。" />

## 3. 矩阵微分导出正规方程

从目标函数展开：

$$
\begin{aligned}
\operatorname{RSS}(\beta)
&=(y-X\beta)^T(y-X\beta)\\
&=y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta\\
&=y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta.
\end{aligned}
$$

因为 $y^TX\beta$ 是标量，等于其转置 $\beta^TX^Ty$。又因为 $X^TX$ 对称，

$$
\nabla_\beta\operatorname{RSS}
=-2X^Ty+2X^TX\beta
=-2X^T(y-X\beta).
$$

令梯度为零，得到与投影推导完全相同的正规方程：

$$
\boxed{X^TX\widehat\beta=X^Ty}.
$$

Hessian 为

$$
\nabla_\beta^2\operatorname{RSS}=2X^TX.
$$

对任意 $v$，

$$
v^TX^TXv=\lVert Xv\rVert_2^2\ge0,
$$

所以目标总是凸的，任何正规方程解都是全局最小。若 $X$ 满列秩，$Xv=0$ 只能推出 $v=0$，Hessian 正定，目标严格凸，系数解唯一：

$$
\widehat\beta=(X^TX)^{-1}X^Ty.
$$

这个逆矩阵公式适合证明，不是推荐实现。$n>p$ 也不保证满列秩：两列完全重复时，即使有一百万行，$X^TX$ 仍然奇异。

```python
beta_hat, residual_sums, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
prediction = X @ beta_hat
residual = y - prediction

assert rank == X.shape[1]
assert np.linalg.norm(X.T @ residual) < 1e-9
assert np.isclose(residual.mean(), 0.0, atol=1e-12)
assert np.allclose(beta_hat, beta_process, atol=6.0)

print("estimated coefficients:", np.round(beta_hat, 3))
print("training RMSE:", np.sqrt(np.mean(residual ** 2)))
```

`np.linalg.lstsq` 同时返回数值秩与奇异值。工程日志应保存这些诊断，而不是只保存四个系数。

<MlQuiz kind="概念辨析" question="哪一项足以保证 OLS 系数唯一？" :options="['n 大于 p', 'X 满列秩', '训练集包含截距', '目标 y 的元素互不相同']" :answer="1" explanation="唯一性取决于 X 的零空间是否只有零向量。n>p 只是满列秩的必要环境之一，不排除重复列或线性组合。" />

## 4. 帽子矩阵把标签投到列空间

满列秩时定义

$$
H=X(X^TX)^{-1}X^T,
\qquad
\widehat y=Hy.
$$

$H$ 被称为帽子矩阵，因为它把 $y$ 变成 $\widehat y$。它不是模型参数；它是由当前设计矩阵决定的 $n\times n$ 线性算子。

### 对称：投影是正交的

$$
H^T
=X\left[(X^TX)^{-1}\right]^TX^T
=H.
$$

### 幂等：投影两次没有新变化

$$
\begin{aligned}
H^2
&=X(X^TX)^{-1}X^TX(X^TX)^{-1}X^T\\
&=X(X^TX)^{-1}X^T=H.
\end{aligned}
$$

### 残差生成矩阵

令 $M=I-H$，则

$$
r=My,
\qquad
M^T=M,
\qquad
M^2=M,
\qquad
HM=0.
$$

$H$ 投到 $\mathcal C(X)$，$M$ 投到其正交补。幂等矩阵的特征值只可能是 0 或 1，因此满列秩时

$$
\operatorname{tr}(H)=\operatorname{rank}(H)=p.
$$

对角元 $h_{ii}$ 称为杠杆值。它只依赖输入位置：某小时的工况若远离大部分训练工况，通常有较高杠杆值；高杠杆不等于该点标签一定异常，却表示它有更大机会拉动拟合。

```python
H = X @ np.linalg.pinv(X)
M = np.eye(n_hours) - H

assert np.allclose(H, H.T)
assert np.allclose(H @ H, H)
assert np.allclose(M, M.T)
assert np.allclose(M @ M, M)
assert np.allclose(H @ M, 0.0, atol=1e-10)
assert np.isclose(np.trace(H), np.linalg.matrix_rank(X))
assert np.allclose(H @ y, prediction)

centered_target = y - y.mean()
centered_prediction = prediction - y.mean()
tss = centered_target @ centered_target
ess = centered_prediction @ centered_prediction
rss = residual @ residual
assert np.isclose(tss, ess + rss)
```

在真实大数据上不要构造完整 $n\times n$ 的 $H$：当 $n=10^6$ 时，仅密集双精度矩阵就约占 8 TB。后文会用 QR 的行范数直接求 $h_{ii}$。

<MlQuiz kind="推导补全" question="H²=H 的直接含义是什么？" :options="['H 的所有元素都是 0 或 1', '已经进入列空间的预测再次投影后不再改变', 'H 必须可逆', '每个杠杆值都等于 p']" :answer="1" explanation="幂等描述算子的重复作用，不要求矩阵元素为 0 或 1。投影一次后已经位于目标子空间。" />

## 5. 秩亏时，预测唯一不等于参数唯一

若 $X$ 不满列秩，存在非零 $v\in\mathcal N(X)$ 使 $Xv=0$。对任意一个最优解 $\widehat\beta$：

$$
X(\widehat\beta+cv)=X\widehat\beta
\qquad\text{对所有 }c\in\mathbb R.
$$

因此有无穷多组系数给出相同预测和 RSS。最优预测仍然唯一，因为它是 $y$ 到固定闭凸子空间 $\mathcal C(X)$ 的最近点；不唯一的是到达这个点的参数坐标。

本案例若同时接入两只校准后读数完全相同的产气量计，模型只能识别两个流量系数的和，无法从数据中区分每只表“贡献”多少。

<CollinearitySolutionsLab />

### 紧致 SVD 给出 Moore–Penrose 伪逆

设 $r=\operatorname{rank}(X)$，紧致奇异值分解为

$$
X=U_r\Sigma_rV_r^T,
$$

其中 $U_r\in\mathbb R^{n\times r}$、$V_r\in\mathbb R^{p\times r}$ 的列正交归一，$\Sigma_r$ 只保留正奇异值。定义

$$
X^+=V_r\Sigma_r^{-1}U_r^T.
$$

伪逆解

$$
\widehat\beta^+=X^+y
$$

满足正规方程，而且预测为

$$
X\widehat\beta^+
=U_rU_r^Ty.
$$

$U_rU_r^T$ 是到 $\mathcal C(X)$ 的正交投影。所有 OLS 解可写成

$$
\widehat\beta^++v,
\qquad v\in\mathcal N(X).
$$

$\widehat\beta^+$ 位于 $X$ 的行空间，行空间与零空间正交，所以

$$
\lVert\widehat\beta^++v\rVert_2^2
=\lVert\widehat\beta^+\rVert_2^2+\lVert v\rVert_2^2.
$$

因此伪逆在所有等价解中选欧氏范数最小者。它是一条坐标选择规则，不会凭空恢复缺失的识别信息。

```python
X_rank_deficient = np.column_stack([
    np.ones(n_hours),
    throughput - 33.0,
    throughput - 33.0,  # 完全重复的备用流量计
    ambient_c - 23.0,
    pressure_bar - 6.8,
])

beta_minimum_norm = np.linalg.lstsq(X_rank_deficient, y, rcond=None)[0]
null_direction = np.array([0.0, 1.0, -1.0, 0.0, 0.0])
beta_alternative = beta_minimum_norm + 25.0 * null_direction

assert np.linalg.matrix_rank(X_rank_deficient) == 4
assert np.allclose(X_rank_deficient @ null_direction, 0.0)
assert np.allclose(
    X_rank_deficient @ beta_minimum_norm,
    X_rank_deficient @ beta_alternative,
)
assert np.linalg.norm(beta_minimum_norm) < np.linalg.norm(beta_alternative)
assert np.allclose(beta_minimum_norm, np.linalg.pinv(X_rank_deficient) @ y)

print("minimum-norm duplicated coefficients:", beta_minimum_norm[1:3])
print("alternative duplicated coefficients:", beta_alternative[1:3])
```

<div class="ml-stop"><strong>停下来推一推 2</strong>两列完全相同时，零空间含有 $(0,1,-1,0,\ldots)^T$。为什么数据只能确定两列系数之和？伪逆把这两个系数分得相等，是物理定律还是最小范数约定？</div>

<MlQuiz kind="错误诊断" question="团队发现重复传感器下 pinv 仍返回了一组系数，于是声称两个传感器的独立边际效应都已识别。错误在哪里？" :options="['pinv 不能处理矩阵', '伪逆只从等价解中选最小范数解，没有增加识别信息', '重复传感器会让预测不存在', '只要系数有限就一定可解释']" :answer="1" explanation="有数值输出不等于参数被数据唯一确定。应检查秩、零空间和系数稳定性，并按业务语义删除或重构冗余特征。" />

## 6. 近共线把小扰动放大

完全重复是秩亏；更常见的是两列几乎相同。此时解在数学上唯一，却可能极不稳定。

SVD 形式下

$$
\widehat\beta
=V\Sigma^{-1}U^Ty
=\sum_{j=1}^p\frac{u_j^Ty}{\sigma_j}v_j.
$$

目标在左奇异方向 $u_j$ 上的一点扰动，会被除以 $\sigma_j$ 后写入系数方向 $v_j$。最小奇异值很小时，系数对标签噪声、舍入误差和样本替换都高度敏感。

二范数条件数为

$$
\kappa_2(X)=\frac{\sigma_{\max}(X)}{\sigma_{\min}(X)}.
$$

若直接形成 Gram 矩阵，

$$
\kappa_2(X^TX)=\kappa_2(X)^2.
$$

这会把原本已经困难的问题再平方。因此不应把 `inv(X.T @ X) @ X.T @ y` 当作生产实现。

近共线不必让训练预测明显变差：系数可以沿几乎零空间的方向大幅移动，$X\beta$ 却只改变一点。这正是“测试分数稳定、系数解释失真”可以同时发生的原因。

<MlQuiz kind="数值判断" question="若 κ₂(X)=10⁶，那么 κ₂(XᵀX) 大约是多少？" :options="['10³', '10⁶', '10¹²', '2×10⁶']" :answer="2" explanation="在二范数下，Gram 矩阵的奇异值是 X 奇异值的平方，因此条件数也平方。" />

## 7. 数值实现：优先求解，不要显式求逆

### `lstsq`：默认基线

`np.linalg.lstsq(X, y, rcond=None)` 直接求最小二乘解，并能根据奇异值阈值处理秩亏。它比手写正规方程更接近生产所需的数值语义。

### QR：高而瘦、满列秩时高效

对 $n\ge p$ 的满列秩矩阵做约化 QR：

$$
X=QR,
\qquad
Q^TQ=I_p,
$$

其中 $Q\in\mathbb R^{n\times p}$，$R\in\mathbb R^{p\times p}$ 为上三角。最小二乘条件变成

$$
R\widehat\beta=Q^Ty,
$$

可用回代求解，不必形成 $X^TX$，更不必求逆。实际库通常使用 Householder QR；经典 Gram–Schmidt 在近相关列上会损失正交性。

```python
Q, R = np.linalg.qr(X, mode="reduced")
beta_qr = np.linalg.solve(R, Q.T @ y)
beta_svd = np.linalg.lstsq(X, y, rcond=None)[0]

condition_x = np.linalg.cond(X)
condition_gram = np.linalg.cond(X.T @ X)

assert np.allclose(Q.T @ Q, np.eye(X.shape[1]))
assert np.allclose(X, Q @ R)
assert np.allclose(beta_qr, beta_svd)
assert np.isclose(condition_gram, condition_x ** 2, rtol=1e-8)

base = np.linspace(-1.0, 1.0, n_hours)
almost_duplicate = base + 1e-7 * rng.normal(size=n_hours)
X_ill = np.column_stack([np.ones(n_hours), base, almost_duplicate])
print("κ(X_ill):", f"{np.linalg.cond(X_ill):.3e}")
print("κ(X_ill.T @ X_ill):", f"{np.linalg.cond(X_ill.T @ X_ill):.3e}")
```

### SVD：秩判断和最小范数语义最清楚

SVD 成本通常高于同规模 QR，但能直接暴露奇异值谱，适合秩亏、近秩亏和宽矩阵。`rcond` 或库内部阈值决定多小的奇异值被视为 0；这不是无关紧要的实现细节，而会改变数值秩和返回系数。

### 迭代法：超大、稀疏或只能做矩阵乘法时

当 $X$ 无法整体放入内存，可使用 LSQR、LSMR 一类只需要 $Xv$ 和 $X^Tu$ 的迭代最小二乘法。此时要记录停止容差、迭代次数、预条件方式和最终残差；“算法停止”不自动等于达到精确 OLS 解。

### 缩放与精度

不同量纲会影响条件数和迭代收敛。中心化、按训练集尺度标准化通常有帮助，但必须在训练分区拟合变换；截距列不做标准化。低精度计算、大动态范围和近共线叠加时，应比较 `float32` 与 `float64`、监控奇异值并做扰动测试。

<MlQuiz kind="工程决策" question="一个 500 万行、30 列、满列秩的稠密设计矩阵需要稳定求 OLS；哪种思路最合理？" :options="['显式构造 500万×500万 的帽子矩阵', '直接计算 inv(XᵀX)', '使用成熟的 QR/lstsq 求解器，并单独计算所需诊断', '逐元素枚举所有 β']" :answer="2" explanation="高而瘦问题适合分解求解。完整帽子矩阵内存不可接受；显式逆会放大数值问题，也没有必要。" />

## 8. 残差诊断：正交不等于模型正确

OLS 无论模型设定是否正确，都会让训练残差与 $X$ 正交。因此以下检查都可能同时成立：残差均值为 0、每个特征与残差样本内积为 0、正规方程误差接近机器精度。它们证明求解器完成了代数任务，不证明业务机制线性、误差独立或未来预测可靠。

### 至少检查四类失配

| 诊断 | 观察对象 | 可能问题 | 下一步 |
|---|---|---|---|
| 残差—拟合值图 | 曲线、漏斗、分组带状 | 非线性、异方差、遗漏工况 | 加入有依据的基函数；检查方差模型与分组 |
| 残差—时间图 | 连续同号、周期、突变 | 自相关、漂移、维护事件 | 按时间评价；加入状态变量或时序模型 |
| 高杠杆点 | 大 $h_{ii}$ | 工况外推、输入错误、稀有组合 | 回看原始记录；报告删除前后敏感性 |
| 大残差点 | 大标准化残差 | 标签异常、未建模故障、重尾噪声 | 核对计量链；不要先删再解释 |

含截距且满列秩时，约化 QR 给出

$$
H=QQ^T,
\qquad
h_{ii}=\sum_{j=1}^p Q_{ij}^2.
$$

只需保存 $Q$ 的逐行平方和，无需构造 $H$。所有杠杆值满足

$$
0\le h_{ii}\le1,
\qquad
\sum_i h_{ii}=p.
$$

删去第 $i$ 个样本后再拟合的留一残差可由一次拟合得到：

$$
r_{(i)}=\frac{r_i}{1-h_{ii}}.
$$

当 $h_{ii}$ 接近 1 时，同一个普通残差会被大幅放大，提示该点几乎在用自己决定自己的预测。

```python
leverage = np.sum(Q ** 2, axis=1)
residual_scale = np.sqrt(rss / (n_hours - X.shape[1]))
standardized_residual = residual / (
    residual_scale * np.sqrt(np.maximum(1.0 - leverage, 1e-12))
)
loo_residual_fast = residual / (1.0 - leverage)

loo_residual_exact = np.empty(n_hours)
for index in range(n_hours):
    keep = np.arange(n_hours) != index
    beta_without_index = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    loo_residual_exact[index] = y[index] - X[index] @ beta_without_index

assert np.all((0.0 <= leverage) & (leverage <= 1.0))
assert np.isclose(leverage.sum(), X.shape[1])
assert np.allclose(loo_residual_fast, loo_residual_exact, rtol=1e-8, atol=1e-8)

most_influential_candidates = np.argsort(
    np.abs(standardized_residual) * np.sqrt(leverage)
)[-5:][::-1]
print("candidate rows for source-data review:", most_influential_candidates)
```

上面的乘积只是用于本章排序演示，不应冒充正式 Cook 距离。生产诊断要预先定义阈值、保留原始行标识，并区分“数据错误”“有效但稀有的工况”和“模型结构不足”。看到异常点后直接删除，会让模型只适用于被人为清理过的世界。

<MlQuiz kind="错误诊断" question="团队用 Xᵀr≈0 证明“残差与工况独立，模型没有遗漏变量”。为什么不成立？" :options="['矩阵乘法不支持独立性', 'Xᵀr≈0 是训练 OLS 的一阶条件，只表示样本线性相关为零，不等于总体独立或模型正确', '残差必须全部等于 0 才能独立', '只有分类模型才需要独立性']" :answer="1" explanation="正交比独立弱得多，而且这里只是同一训练样本上的代数约束。应使用独立评价、残差图、分组与时间诊断检验模型失配。" />

<div class="ml-stop"><strong>停下来推一推 3</strong>一个小时同时具有最高杠杆值和很小的普通残差。为什么不能据此忽略它？利用 $r_i/(1-h_{ii})$ 说明留一预测可能发生什么。</div>

## 数值验证

对本章实现做一次分层审计，不能只检查系数“看起来合理”。

```python
projection_checks = {
    "normal_equation_residual": float(np.linalg.norm(X.T @ residual)),
    "projection_symmetry": float(np.linalg.norm(H - H.T)),
    "projection_idempotence": float(np.linalg.norm(H @ H - H)),
    "pythagorean_gap": float(abs(tss - ess - rss)),
    "qr_reconstruction": float(np.linalg.norm(X - Q @ R)),
    "loo_formula_gap": float(np.max(np.abs(loo_residual_fast - loo_residual_exact))),
}

assert max(projection_checks.values()) < 1e-7
assert np.linalg.matrix_rank(X) == X.shape[1]
assert singular_values[-1] > 0
assert np.isfinite(beta_hat).all()

for name, value in projection_checks.items():
    print(f"{name}: {value:.3e}")
```

一条可复查的 OLS 验证链至少包括：

1. 检查样本单位、时间范围、特征截止、单位、缺失值和重复记录；
2. 保存 $n$、$p$、数值秩、奇异值谱和条件数；
3. 用两种成熟求解路径比较预测，而不是只比较可能不唯一的系数；
4. 检查 $X^Tr$、QR 重构和目标函数值；
5. 画残差—拟合值、残差—时间和输入覆盖图；
6. 审核高杠杆、大残差与高影响候选的原始数据；
7. 在独立时间段或设备组上评价 RMSE、MAE 与业务代价；
8. 对重采样、微小标签扰动和特征删减做系数及预测稳定性分析。

如果两种算法系数差异很大而预测接近，应先怀疑秩、尺度和可辨识性，不要用“平均系数”掩盖问题。

## 复杂度与工程边界

下面给出稠密、$n\ge p$ 情形的典型量级；具体常数取决于实现、硬件与矩阵结构。

| 方法 | 时间量级 | 额外空间/对象 | 优点 | 主要边界 |
|---|---:|---:|---|---|
| 正规方程 + Cholesky | 形成 Gram 约 $O(np^2)$，分解 $O(p^3)$ | $p\times p$ Gram | 快、内存小 | 条件数平方；不适合秩亏；不要显式求逆 |
| 约化 QR | $O(np^2)$ | $Q$ 或 Householder 表示、$R$ | 满列秩时稳定，适合高而瘦 | 成本通常高于形成 Gram；需列主元处理秩判断 |
| 薄 SVD | 典型 $O(np^2+p^3)$ | 奇异向量与奇异值 | 秩、条件数、最小范数语义清楚 | 常数较大；截断阈值影响数值秩 |
| LSQR/LSMR | 每轮约为稀疏矩阵乘法成本 | 若干向量 | 超大、稀疏、矩阵自由 | 近病态时收敛慢；停止容差影响解 |
| 完整帽子矩阵 | 时间和空间至少 $O(n^2)$ | $n\times n$ | 只适合小型教学审计 | 大样本不可行，通常只算对角或矩阵—向量积 |

### 工业实现边界

- **OLS 不是因果估计器**：$\beta_q$ 是给定已纳入特征后的样本线性关联；遗漏变量、反向因果和选择偏差都可能改变它。
- **OLS 不负责发现所有非线性**：残差结构可提示失配，但加入哪个基函数必须由机理、验证和数据覆盖共同约束。
- **平方损失对大残差敏感**：异常点的贡献按残差平方增长。先审计计量链，再决定稳健损失、分层模型或数据修复。
- **同方差不是求点估计的条件**：但经典标准误与区间会依赖更强假设；第 05 章再讨论异方差稳健推断。
- **共线不是靠更多小数位修复**：应从传感器语义、采样覆盖、变量重构或正则化处理；Ridge 会在第 06 章出现。
- **外推风险不会显示在训练 RSS 里**：参考工况之外的预测必须结合输入支持域和独立时间/设备验证。

<MlQuiz kind="工程判断" question="两只高度相关的流量计让单个系数在重采样中大幅变动，但总能耗预测稳定。最诚实的报告是什么？" :options="['报告任意一次系数并解释为各自因果效应', '说明联合预测在当前覆盖域内稳定，但单个边际系数不可可靠解释，并给出秩/条件数与敏感性证据', '把所有系数四舍五入为 0', '只报告训练 R² 就足够']" :answer="1" explanation="预测性能与参数可辨识性是不同问题。需要同时报告目标域预测证据和系数不稳定的限制。" />

## 9. 章节练习：完成一份 OLS 基线审计

### 推导练习

1. 从 $g(a)=\lVert r-aXv\rVert^2$ 出发，不使用矩阵微分证明残差与整个列空间正交。
2. 证明 $H=XX^+$ 在秩亏情形仍然对称、幂等，并且 $\operatorname{tr}(H)=\operatorname{rank}(X)$。
3. 设 $X$ 有两列完全相同，写出所有最优解，并证明伪逆解的范数最小。
4. 从分块矩阵或 Sherman–Morrison 公式推导 $r_{(i)}=r_i/(1-h_{ii})$，说明 $h_{ii}=1$ 时发生了什么。

### 实现练习

在本章合成数据上加入第二只近重复流量计与 5 个计量异常小时，完成求解器、秩、条件数、系数稳定性、残差和杠杆值审计。不得构造完整帽子矩阵。

```python
# TODO：由学习者完成；本代码块不参与自动执行。
def audit_ols_baseline(X_train, y_train, row_ids):
    # 1. 校验形状、有限值和 row_ids 唯一性
    # 2. 用稳定求解器返回最小二乘解、数值秩和奇异值
    # 3. 不构造 H，计算 leverage 对角元
    # 4. 返回正规方程残差、条件数、LOOCV 残差和待复核 row_ids
    # 5. 对微小标签扰动重拟合，分别报告系数与预测变化
    raise NotImplementedError
```

### 案例审计练习

审查以下结论：“温度系数为 4.08，所以升高 1°C 必然导致电耗增加 4.08 kWh；模型 $R^2=0.93$，可直接用于下季度所有机组调度。”

答案至少指出：关联不等于因果、系数依赖已纳入变量和覆盖域、训练拟合度不代表未来风险、事后同小时基线不能直接冒充事前调度模型，并给出需要补充的数据切分与稳定性证据。

<details><summary>审计检查点</summary>

- 明确模型的使用时刻和每个特征的可用时刻；
- 按时间和机组冻结独立评价区，而不是随机拆小时；
- 报告目标域 RMSE/MAE、分组结果和区间，不只报训练 $R^2$；
- 检查温度与负荷、季节、设备组合之间的共线性；
- 通过物理试验或因果设计才可能支持“升温导致”的结论；
- 超出训练温度、压力或负荷覆盖域时标记外推，不静默给出高置信预测。

</details>

## 10. 本章小结与术语

- 设计矩阵的列空间是线性模型能够表达的全部训练集预测。
- OLS 把 $y$ 正交投影到列空间；残差正交与正规方程是同一个最优条件。
- 满列秩使 $X^TX$ 正定、目标严格凸，因此系数唯一；秩亏时最优预测仍唯一，参数不唯一。
- 伪逆由非零奇异值倒数构成，返回最小范数解，但不会恢复不可辨识的物理含义。
- 形成 $X^TX$ 会平方条件数；成熟的 `lstsq`、QR、SVD 或迭代最小二乘求解器优于显式求逆。
- 帽子矩阵对称、幂等，杠杆值来自其对角；大数据中用 QR 行范数求对角，不构造完整矩阵。
- 残差正交只证明训练代数最优。非线性、异方差、自相关、异常值、分布偏移和因果问题仍需独立证据。

<dl class="ml-terms">
  <div><dt>design matrix</dt><dd>每行对应样本、每列对应一个基函数或截距的矩阵。</dd></div>
  <div><dt>column space</dt><dd>所有 $X\beta$ 构成的可表达预测子空间。</dd></div>
  <div><dt>normal equations</dt><dd>$X^TX\widehat\beta=X^Ty$；等价于最优残差与列空间正交。</dd></div>
  <div><dt>hat matrix</dt><dd>$H=XX^+$；把目标向量投到设计矩阵列空间的算子。</dd></div>
  <div><dt>rank deficiency</dt><dd>设计矩阵列线性相关，导致存在非零零空间方向和非唯一系数。</dd></div>
  <div><dt>pseudoinverse</dt><dd>通过非零奇异值倒数定义的广义逆；在 OLS 中给出最小范数解。</dd></div>
  <div><dt>condition number</dt><dd>最大与最小有效奇异值之比，刻画输入扰动可能被放大的程度。</dd></div>
  <div><dt>leverage</dt><dd>帽子矩阵对角元，衡量样本输入位置对自身拟合值的潜在影响。</dd></div>
  <div><dt>identifiability</dt><dd>观测分布是否能唯一确定参数；有数值解不等于参数可辨识。</dd></div>
</dl>

## 来源与许可

- Stanford CS229 Lecture Notes，PDF 第 15–17 页，§1.2：用于核对设计矩阵、平方损失矩阵展开、正规方程及可逆条件。本站使用空压站案例重新推导和表述，不复用讲义图表或住房案例。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 404–406 页，§11.2.2.1–11.2.2.3：用于核对 OLS、正交投影、帽子矩阵、SVD/QR 求解与标准化边界。本站交互、代码、题目和图形均独立创作。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 298–299 页，§7.7.2–7.7.3：用于核对欠定系统的一般解、最小范数解、正规方程和满列秩唯一性证明；不复用原文案例或图形。
- Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 61–63 页：用于交叉核对截距列、正规方程、伪逆、帽子矩阵和“不要显式求逆”的数值提醒。
- Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 140–142 页：用于核对紧致 SVD 伪逆、最小范数解和小奇异值放大效应。
- 本章空压站 240 小时数据、双流量计情境、两项网页交互、全部题目、代码和 SVG 图形均为教学合成与本站独立创作，不代表真实工厂能耗、传感器精度或节能标准。详细登记见[来源与许可登记](./sources.md) 。

<MlLessonProgress lesson-id="04" next-label="第 05 章：线性回归：高斯似然与统计推断" next-href="/courses/machine-learning/05-gaussian-likelihood" />
