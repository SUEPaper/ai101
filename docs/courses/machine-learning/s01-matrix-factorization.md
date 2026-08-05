---
title: S01. 矩阵分解
description: 用特征分解、SVD、QR 与 Cholesky 看懂机器学习中的线性代数计算
course: machine-learning
---

<MlLessonHero
  eyebrow="选修 · 数学工具箱"
  title="矩阵分解"
  subtitle="矩阵分解把一个难操作的线性变换拆成旋转、缩放或三角求解，让结构和数值稳定性同时可见。"
  :diagnostics="['能区分特征分解与 SVD 的适用条件', '能解释最小二乘为何优先 QR/SVD 而非正规方程', '能用条件数判断问题敏感性']"
/>

## 1. 矩阵是线性变换

$\mathbf A\in\mathbb R^{m\times n}$ 把 $\mathbb R^n$ 的向量映射到 $\mathbb R^m$。分解的目的，是选择一组更容易理解的坐标，使变换成为简单的缩放或三角运算。

不要把 $\mathbf A^{-1}$ 当作求解线性方程的默认工具。求

$$
\mathbf A\mathbf x=\mathbf b
$$

时，数值库会通过分解和三角代入直接求 $\mathbf x$，显式逆矩阵通常更慢、更耗内存、误差也更大。

## 2. 特征分解：同一空间中的不变方向

若

$$
\mathbf A\mathbf v=\lambda\mathbf v,
$$

$\mathbf v$ 经变换后方向不变，只缩放 $\lambda$。可对角化方阵满足

$$
\mathbf A=\mathbf V\boldsymbol\Lambda\mathbf V^{-1}.
$$

实对称矩阵更好：

$$
\mathbf A=\mathbf Q\boldsymbol\Lambda\mathbf Q^\top,
\qquad
\mathbf Q^\top\mathbf Q=\mathbf I,
$$

其特征值为实数、特征向量可正交。协方差矩阵还半正定，所以 $\lambda_i\ge0$。

矩阵幂因此容易：

$$
\mathbf A^k
=\mathbf Q\boldsymbol\Lambda^k\mathbf Q^\top.
$$

<MlQuiz
  question="实对称矩阵的特征分解有什么特殊好处？"
  :options="['特征值为实数且可取正交特征向量', '一定可逆', '所有特征值都等于 1', '只能用于方差为零的数据']"
  :answer="0"
  explanation="谱定理保证实对称矩阵可由正交矩阵对角化。"
/>

## 3. SVD：任何矩形矩阵都能分解

奇异值分解：

$$
\mathbf A=\mathbf U\boldsymbol\Sigma\mathbf V^\top,
$$

其中 $\mathbf U,\mathbf V$ 列正交，$\boldsymbol\Sigma$ 对角非负。几何顺序是：

1. $\mathbf V^\top$ 旋转输入；
2. $\boldsymbol\Sigma$ 沿正交轴缩放；
3. $\mathbf U$ 旋转到输出空间。

关系：

$$
\mathbf A^\top\mathbf A
=\mathbf V\boldsymbol\Sigma^2\mathbf V^\top,
$$

$$
\mathbf A\mathbf A^\top
=\mathbf U\boldsymbol\Sigma^2\mathbf U^\top.
$$

因此右奇异向量是 $\mathbf A^\top\mathbf A$ 的特征向量，奇异值平方是特征值。

<MlQuiz
  question="SVD 相比普通特征分解，最重要的适用性差异是什么？"
  :options="['SVD 可用于任意矩形矩阵', 'SVD 只用于对称方阵', 'SVD 不含正交向量', 'SVD 不能表示秩']"
  :answer="0"
  explanation="任意实矩阵都有 SVD，而特征分解只直接定义于方阵且不一定可对角化。"
/>

## 4. 低秩近似

奇异值从大到小 $\sigma_1\ge\cdots$，截断：

$$
\mathbf A_k
=\sum_{j=1}^{k}
\sigma_j\mathbf u_j\mathbf v_j^\top.
$$

Eckart–Young 定理：

$$
\mathbf A_k
=\arg\min_{\operatorname{rank}(\mathbf B)\le k}
\|\mathbf A-\mathbf B\|_F,
$$

且

$$
\|\mathbf A-\mathbf A_k\|_F^2
=\sum_{j>k}\sigma_j^2.
$$

这连接 PCA、矩阵压缩和推荐系统。

::: ml-stop 停下来推一推
奇异值为 $[5,3,1]$。最佳秩 1 近似的 Frobenius 重构误差平方是多少？
:::

## 5. 伪逆与最小二乘

若

$$
\mathbf A=\mathbf U\boldsymbol\Sigma\mathbf V^\top,
$$

伪逆

$$
\mathbf A^+
=\mathbf V\boldsymbol\Sigma^+\mathbf U^\top,
$$

其中非零奇异值取倒数，零保持零。最小二乘最小范数解：

$$
\mathbf x^\star=\mathbf A^+\mathbf b.
$$

很小的奇异值取倒数会放大噪声。截断 SVD 与 Ridge 都通过压制小奇异方向稳定估计。

## 6. QR：最小二乘的稳定路线

满列秩 $\mathbf X=\mathbf Q\mathbf R$，$\mathbf Q^\top\mathbf Q=\mathbf I$，$\mathbf R$ 上三角：

$$
\min_\beta\|\mathbf y-\mathbf X\beta\|^2
=\min_\beta\|\mathbf Q^\top\mathbf y-\mathbf R\beta\|^2+\text{常数}.
$$

解三角系统

$$
\mathbf R\beta=\mathbf Q^\top\mathbf y.
$$

正规方程形成 $\mathbf X^\top\mathbf X$，其 2-范数条件数约平方：

$$
\kappa_2(\mathbf X^\top\mathbf X)=\kappa_2(\mathbf X)^2.
$$

所以数据病态时优先 QR 或 SVD。

<MlQuiz
  question="为什么数值计算中常避免用正规方程直接解最小二乘？"
  :options="['X^T X 会让条件数平方，放大数值问题', '矩阵乘法未定义', 'QR 只能处理整数', '正规方程没有解']"
  :answer="0"
  explanation="形成 Gram 矩阵损失条件数，QR/SVD 通常更稳定。"
/>

## 7. Cholesky：正定矩阵的高效分解

对对称正定矩阵：

$$
\mathbf A=\mathbf L\mathbf L^\top,
$$

$\mathbf L$ 为下三角。先解

$$
\mathbf L\mathbf y=\mathbf b,
$$

再解

$$
\mathbf L^\top\mathbf x=\mathbf y.
$$

它常用于高斯模型、Ridge 与协方差计算。若分解失败，应检查对称性、正定性或加入有依据的 jitter，而不是盲目增大对角线。

## 8. 条件数不是算法误差

可逆矩阵的 2-范数条件数：

$$
\kappa_2(\mathbf A)
=\frac{\sigma_{\max}}{\sigma_{\min}}.
$$

它衡量输入扰动可被解放大的程度。$\kappa$ 大表示问题本身敏感；更稳定算法能避免额外损失，但不能凭空恢复数据中不存在的信息。

## 9. 动手练习

1. 构造奇异值从 $1$ 到 $10^{-10}$ 的矩阵；
2. 比较正规方程、QR、SVD 的最小二乘误差；
3. 扰动 $\mathbf b$ 并观察解变化；
4. 用截断 SVD 稳定解；
5. 验证秩 $k$ 重构误差等于丢弃奇异值平方和。

<MlLessonProgress
  storage-key="s01-matrix-factorization"
  next-href="/courses/machine-learning/s02-matrix-calculus"
  next-label="下一篇：矩阵微分"
/>
