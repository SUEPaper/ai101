---
course: machine-learning
title: 最小二乘的几何本质
description: 从列空间投影和残差正交完整推导正规方程、伪逆与帽子矩阵，并区分预测唯一和参数唯一。
outline: [2, 3]
---

<MlLessonHero number="04" title="最小二乘的几何本质" question="为什么最小平方误差的残差必须与每一个特征方向正交？" duration="建议 300–360 分钟" :outcomes="['从列空间投影推出正规方程', '判断 OLS 解的存在性、参数唯一性与预测唯一性', '用 SVD 伪逆解释秩亏和最小范数解']" />

## 先修连接与诊断

数据科学第 12 讲已经把 $X\beta$ 解释为批量预测。本章要证明投影性质、处理不可逆情形，并解释为什么“直接求逆”不是稳健实现。

<MlQuiz question="X 的形状是 n×p，β 和 y 的形状分别是什么？" :options="['β:n×1，y:p×1', 'β:p×1，y:n×1', 'β:p×n，y:1×n', '都必须是 n×p']" :answer="1" explanation="内部维度 p 相消，Xβ 得到 n×1，与每个样本一个响应的 y 对齐。" />

<MlQuiz question="若 X 有两列完全相同，XᵀX 会怎样？" :options="['仍正定可逆', '奇异，系数不唯一', '样本数自动翻倍', '残差一定为零']" :answer="1" explanation="重复列让 X 存在非零零空间向量，因此 XᵀX 也有零特征值。" />

<MlQuiz question="OLS 最优残差与什么正交？" :options="['只与预测向量正交', '与 X 列空间中的所有向量正交', '与 y 正交', '与参数 β 正交']" :answer="1" explanation="最优预测是 y 到列空间的正交投影，残差位于列空间的正交补。" />

## 1. 设计矩阵定义可表达的全部预测

令 $X\in\mathbb R^{n\times p}$，第 $i$ 行是样本 $x_i^T$，$y\in\mathbb R^n$。线性模型的预测集合是

$$
\mathcal C(X)=\{X\beta:\beta\in\mathbb R^p\},
$$

即 $X$ 的列空间。改变 $\beta$ 只能在这个子空间内移动。OLS 要解决

$$
\widehat\beta\in\arg\min_{\beta\in\mathbb R^p}
\lVert y-X\beta\rVert_2^2.
$$

参数空间可能有多组解，但子空间中距离 $y$ 最近的投影点 $\widehat y$ 总是唯一，因为有限维线性子空间是闭凸集。

## 2. 沿任意列空间方向都不能再下降

记残差 $e=y-X\widehat\beta$。若存在某个方向 $Xv$ 与残差不正交，即 $e^TXv\ne0$，沿参数方向 $v$ 做小变化：

$$
\begin{aligned}
g(t)&=\lVert y-X(\widehat\beta+tv)\rVert_2^2\\
&=\lVert e-tXv\rVert_2^2\\
&=e^Te-2t e^TXv+t^2\lVert Xv\rVert_2^2.
\end{aligned}
$$

在 $t=0$ 的导数是 $g'(0)=-2e^TXv$。若它非零，总能选 $t$ 的符号让目标下降，与最优矛盾。因此对所有 $v$：

$$
e^TXv=0.
$$

等价地

$$
\boxed{X^T(y-X\widehat\beta)=0}.
$$

展开得到正规方程：

$$
\boxed{X^TX\widehat\beta=X^Ty}.
$$

这不是凭空“对平方损失求导的口诀”，而是最近投影点的正交条件。

<div class="ml-stop"><strong>停下来推一推 1</strong>若 X 包含一列全 1，正交条件对应哪条残差性质？答案是 $\mathbf1^Te=\sum_i e_i=0$，因此带截距 OLS 的训练残差均值为 0。</div>

<MlLab kind="ols" />

## 3. 用矩阵微分再次得到同一结论

展开目标：

$$
\begin{aligned}
J(\beta)
&=(y-X\beta)^T(y-X\beta)\\
&=y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta\\
&=y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta.
\end{aligned}
$$

因为 $X^TX$ 对称，

$$
\nabla_\beta J=-2X^Ty+2X^TX\beta
=-2X^T(y-X\beta).
$$

Hessian 为

$$
\nabla_\beta^2J=2X^TX\succeq0,
$$

因为对任意 $v$，$v^TX^TXv=\lVert Xv\rVert^2\ge0$。所以目标凸，任一驻点都是全局最小。

若 $X$ 满列秩，$Xv=0$ 只可能 $v=0$，Hessian 正定，目标严格凸，解唯一：

$$
\widehat\beta=(X^TX)^{-1}X^Ty.
$$

这个公式用于理论推导；数值计算优先 QR、SVD 或专门最小二乘求解器。

<MlQuiz question="X 满列秩时，为什么 OLS 系数唯一？" :options="['因为样本误差一定为 0', '因为 XᵀX 正定，目标严格凸', '因为 y 位于列空间', '因为使用了截距']" :answer="1" explanation="满列秩使任意非零 v 都满足 ‖Xv‖²>0，因此 Hessian 正定。" />

## 4. 秩亏时：参数不唯一，预测仍唯一

若存在 $v\ne0$ 使 $Xv=0$，那么对任意解 $\widehat\beta$：

$$
X(\widehat\beta+v)=X\widehat\beta.
$$

两组参数给出完全相同的预测和损失。参数不可辨识，但投影 $\widehat y$ 仍唯一。

对 $X$ 做薄 SVD：

$$
X=U_r\Sigma_rV_r^T,
$$

其中只保留 $r=\operatorname{rank}(X)$ 个正奇异值。Moore–Penrose 伪逆为

$$
X^+=V_r\Sigma_r^{-1}U_r^T.
$$

最小范数 OLS 解是

$$
\widehat\beta^+=X^+y.
$$

验证预测：

$$
X\widehat\beta^+
=U_r\Sigma_rV_r^TV_r\Sigma_r^{-1}U_r^Ty
=U_rU_r^Ty.
$$

$U_rU_r^T$ 正是到 $\mathcal C(X)$ 的正交投影矩阵。伪逆没有让不可辨识参数突然变得可辨识；它从无穷多解中选取欧氏范数最小者。

<div class="ml-stop"><strong>停下来推一推 2</strong>一般解可写成 $\widehat\beta^++v$，其中 $v\in\mathcal N(X)$。利用行空间与零空间正交，证明 $\lVert\widehat\beta^++v\rVert^2=\lVert\widehat\beta^+\rVert^2+\lVert v\rVert^2$。</div>

## 5. 帽子矩阵的三条性质

满列秩时定义

$$
H=X(X^TX)^{-1}X^T,
\qquad \widehat y=Hy.
$$

它把 $y$ “戴上帽子”变成 $\widehat y$。

### 对称

$$
H^T=[X(X^TX)^{-1}X^T]^T
=X[(X^TX)^{-1}]^TX^T=H.
$$

因为 $X^TX$ 及其逆都对称。

### 幂等

$$
\begin{aligned}
H^2
&=X(X^TX)^{-1}X^TX(X^TX)^{-1}X^T\\
&=X(X^TX)^{-1}X^T=H.
\end{aligned}
$$

投影一次已经进入子空间，再投影不会改变。

### 迹等于秩

幂等矩阵的特征值只可能是 0 或 1，因此

$$
\operatorname{tr}(H)=\operatorname{rank}(H)=\operatorname{rank}(X)=p.
$$

这也解释了 OLS 的有效自由度。对角元 $h_{ii}$ 是 leverage，表示第 $i$ 个输入位置对自身拟合值的影响。

## 6. 数值实现：不要显式求逆

```python
import numpy as np

X = np.array([
    [1., 0., 0.],
    [1., 1., 1.],
    [1., 2., 2.],
    [1., 3., 3.],
])
y = np.array([1., 2., 2.8, 4.2])

beta, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
prediction = X @ beta
error = y - prediction

print("rank:", rank, "columns:", X.shape[1])
print("minimum-norm beta:", beta)
assert rank < X.shape[1]
assert np.linalg.norm(X.T @ error) < 1e-10
assert np.allclose(prediction, X @ np.linalg.pinv(X) @ y)
```

`np.linalg.lstsq` 基于分解处理满秩和秩亏，不形成 $X^TX$。形成 $X^TX$ 会把条件数近似平方：

$$
\kappa_2(X^TX)=\kappa_2(X)^2,
$$

因此近共线时数值误差被放大。

## 7. 练习：投影矩阵审计

对一个满列秩 $X$ 和一个秩亏 $X$，分别用 `pinv` 构造 $H=XX^+$。检查对称、幂等、残差正交和 `trace(H)=rank(X)`；再向秩亏解加一个零空间向量，验证预测不变、系数范数变大。

<details><summary>参考断言</summary>

```python
def projection_report(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    H = X @ np.linalg.pinv(X)
    prediction = H @ y
    residual = y - prediction
    return {
        "symmetric": np.allclose(H, H.T),
        "idempotent": np.allclose(H @ H, H),
        "orthogonality": np.linalg.norm(X.T @ residual),
        "trace": np.trace(H),
        "rank": np.linalg.matrix_rank(X),
    }

report = projection_report(X, y)
assert report["symmetric"] and report["idempotent"]
assert report["orthogonality"] < 1e-10
assert np.isclose(report["trace"], report["rank"])
```

</details>

<div class="ml-stop"><strong>停下来推一推 3</strong>若 y 已经位于 X 的列空间，H y 等于什么？残差、SSE 和训练 $R^2$ 分别是多少？这仍不保证未来误差为零。</div>

## 8. 常见误解与条件

- OLS 总有最优预测，但系数不一定唯一。
- $X^TX$ 可逆要求满列秩，不只是 $n>p$；列仍可能相关。
- 残差与特征正交是训练样本上的代数性质，不等于误差与特征在总体上独立。
- 伪逆解决线性代数定义，不修复共线导致的解释不稳定。
- 平方损失的概率来源、同方差和不确定性留到下一章。

<MlQuiz question="帽子矩阵 H 幂等表示什么？" :options="['H 的元素都等于 1', '投影后再次投影不再改变结果', 'H 不可逆', '残差等于预测']" :answer="1" explanation="Hy 已在列空间内，因此 H(Hy)=Hy。" />

<MlQuiz question="为什么不推荐计算 inv(X.T@X)@X.T@y？" :options="['理论公式错误', '形成 XᵀX 会恶化条件数，显式求逆也不必要', 'NumPy 不支持逆', '只能用于分类']" :answer="1" explanation="QR/SVD 或线性求解能避免显式逆，并更好处理近奇异与秩亏。" />

## 9. 本章小结

- OLS 是把 $y$ 正交投影到 $X$ 的列空间，正规方程就是残差正交。
- 满列秩保证系数唯一；秩亏时伪逆选择最小范数解，但预测仍唯一。
- 帽子矩阵对称、幂等，迹等于模型列空间维数。

<MlLessonProgress lesson-id="04" next-label="第 05 章：为什么高斯噪声导出平方损失" next-href="/courses/machine-learning/05-gaussian-likelihood" />
