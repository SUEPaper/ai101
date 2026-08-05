---
title: S02. 矩阵微分
description: 用 differential 与 trace 技巧系统推导机器学习梯度
course: machine-learning
---

<MlLessonHero
  eyebrow="选修 · 数学工具箱"
  title="矩阵微分"
  subtitle="先把标量目标写成 differential，再把 dX 前的系数读成梯度，能避开大量索引混乱。"
  :diagnostics="['能区分梯度、Jacobian 与形状', '能用 trace 技巧推导二次型和最小二乘', '能检查广播与矩阵乘法的反向']"
/>

## 1. 先约定形状

标量函数 $f(\mathbf x)$ 的梯度采用列向量：

$$
df=(\nabla_{\mathbf x}f)^\top d\mathbf x.
$$

矩阵变量 $\mathbf X$：

$$
df
=\operatorname{tr}
\left[
(\nabla_{\mathbf X}f)^\top d\mathbf X
\right].
$$

只要把 $df$ 整理成右侧形式，$d\mathbf X$ 前的矩阵就是梯度。

常用恒等式：

$$
d(\mathbf A\mathbf B)
=(d\mathbf A)\mathbf B+\mathbf A(d\mathbf B),
$$

$$
d(\mathbf A^{-1})
=-\mathbf A^{-1}(d\mathbf A)\mathbf A^{-1},
$$

$$
\operatorname{tr}(\mathbf A\mathbf B\mathbf C)
=\operatorname{tr}(\mathbf C\mathbf A\mathbf B).
$$

## 2. 线性函数

$$
f(\mathbf x)=\mathbf a^\top\mathbf x.
$$

则

$$
df=\mathbf a^\top d\mathbf x,
\qquad
\nabla_{\mathbf x}f=\mathbf a.
$$

对

$$
f(\mathbf X)=\operatorname{tr}(\mathbf A^\top\mathbf X),
$$

有

$$
df=\operatorname{tr}(\mathbf A^\top d\mathbf X),
\qquad
\nabla_{\mathbf X}f=\mathbf A.
$$

它是“读梯度”的模板。

## 3. 二次型

$$
f(\mathbf x)=\mathbf x^\top\mathbf A\mathbf x.
$$

微分：

$$
\begin{aligned}
df
&=(d\mathbf x)^\top\mathbf A\mathbf x
+\mathbf x^\top\mathbf A(d\mathbf x)\\
&=\mathbf x^\top\mathbf A^\top d\mathbf x
+\mathbf x^\top\mathbf A d\mathbf x\\
&=[(\mathbf A+\mathbf A^\top)\mathbf x]^\top d\mathbf x.
\end{aligned}
$$

所以

$$
\boxed{
\nabla_{\mathbf x}f
=(\mathbf A+\mathbf A^\top)\mathbf x
}.
$$

若 $\mathbf A$ 对称，化为 $2\mathbf A\mathbf x$。

<MlQuiz
  question="若 A 不保证对称，x^T A x 对 x 的梯度是什么？"
  :options="['Ax', '(A+A^T)x', 'A^T x', '2x']"
  :answer="1"
  explanation="两个 x 分别产生一项，合并为对称部分作用于 x。"
/>

## 4. 最小二乘

$$
L(\boldsymbol\beta)
=\frac12
\|\mathbf y-\mathbf X\boldsymbol\beta\|_2^2.
$$

令 $\mathbf r=\mathbf y-\mathbf X\boldsymbol\beta$：

$$
dL
=\mathbf r^\top d\mathbf r
=\mathbf r^\top(-\mathbf X\,d\boldsymbol\beta)
=(-\mathbf X^\top\mathbf r)^\top
d\boldsymbol\beta.
$$

所以

$$
\nabla_{\boldsymbol\beta}L
=\mathbf X^\top(\mathbf X\boldsymbol\beta-\mathbf y).
$$

这比逐元素求导更短，并自动保持形状。

::: ml-stop 停下来推一推
在目标中加入 $\lambda\|\beta\|^2/2$，写出新梯度并令零，得到 Ridge 正规方程。
:::

## 5. trace 技巧推导矩阵乘法反向

设

$$
\mathbf Z=\mathbf X\mathbf W,\qquad L=L(\mathbf Z),
$$

上游梯度 $\mathbf G=\nabla_{\mathbf Z}L$。因为

$$
d\mathbf Z=(d\mathbf X)\mathbf W+\mathbf X(d\mathbf W),
$$

$$
\begin{aligned}
dL
&=\operatorname{tr}(\mathbf G^\top d\mathbf Z)\\
&=\operatorname{tr}(\mathbf G^\top(d\mathbf X)\mathbf W)
+\operatorname{tr}(\mathbf G^\top\mathbf X(d\mathbf W))\\
&=\operatorname{tr}((\mathbf G\mathbf W^\top)^\top d\mathbf X)
+\operatorname{tr}((\mathbf X^\top\mathbf G)^\top d\mathbf W).
\end{aligned}
$$

因此

$$
\boxed{
\nabla_{\mathbf X}L=\mathbf G\mathbf W^\top,\qquad
\nabla_{\mathbf W}L=\mathbf X^\top\mathbf G
}.
$$

<MlQuiz
  question="Z=XW，上游梯度 G 与 Z 同形状。对 W 的梯度是什么？"
  :options="['GW^T', 'X^T G', 'G^T X', 'XG']"
  :answer="1"
  explanation="通过 trace 循环置换把 dW 移到末尾，读出 X^T G。"
/>

## 6. log determinant 与逆矩阵

对正定 $\mathbf X$：

$$
d\log|\mathbf X|
=\operatorname{tr}(\mathbf X^{-1}d\mathbf X),
$$

所以

$$
\nabla_{\mathbf X}\log|\mathbf X|
=\mathbf X^{-\top}.
$$

若 $\mathbf X$ 对称，则为 $\mathbf X^{-1}$。再例如

$$
f=\mathbf a^\top\mathbf X^{-1}\mathbf b,
$$

$$
df
=-\mathbf a^\top\mathbf X^{-1}
(d\mathbf X)\mathbf X^{-1}\mathbf b.
$$

用 trace 整理：

$$
\nabla_{\mathbf X}f
=-\mathbf X^{-\top}\mathbf a\mathbf b^\top\mathbf X^{-\top}.
$$

## 7. 元素级、归约与广播

若 $\mathbf Y=\phi(\mathbf X)$ 逐元素，则

$$
\nabla_{\mathbf X}L
=\nabla_{\mathbf Y}L\odot\phi'(\mathbf X).
$$

前向 reduce-sum，反向 broadcast；前向 broadcast，反向在被复制轴 reduce-sum。每个梯度必须与被求导变量同形状，这是最便宜的错误检查。

<MlQuiz
  question="检查矩阵梯度公式的第一道低成本防线是什么？"
  :options="['梯度形状必须与对应变量相同', '所有梯度必须为正', '矩阵必须方形', '损失必须是整数']"
  :answer="0"
  explanation="形状不能证明公式正确，但能立即排除大量转置和广播错误。"
/>

## 8. 有限差分仍是最终核对

对随机方向 $\mathbf D$，方向导数应满足

$$
\frac{f(\mathbf X+\epsilon\mathbf D)
-f(\mathbf X-\epsilon\mathbf D)}{2\epsilon}
\approx
\operatorname{tr}
[(\nabla_{\mathbf X}f)^\top\mathbf D].
$$

方向检查一次能覆盖整个梯度，比逐元素更快；发现问题后再定位元素。

## 9. 动手练习

依次推导并用有限差分验证：

1. $\|\mathbf A\mathbf X-\mathbf B\|_F^2/2$；
2. $\operatorname{tr}(\mathbf X^\top\mathbf A\mathbf X)$；
3. $\log|\mathbf X|$；
4. Softmax 交叉熵对 logits 的梯度；
5. 带批量偏置的矩阵乘法反向。

<MlLessonProgress
  storage-key="s02-matrix-calculus"
  next-href="/courses/machine-learning/s03-probability"
  next-label="下一篇：概率统计"
/>
