---
title: S05. 凸优化
description: 从凸集、凸函数、一阶条件到梯度法与近端方法理解优化保证
course: machine-learning
---

<MlLessonHero
  eyebrow="选修 · 数学工具箱"
  title="凸优化"
  subtitle="凸性把局部信息升级成全局保证：任意局部最小都是全局最小，一阶切线永远位于函数下方。"
  :diagnostics="['能用定义、Hessian 判断凸性', '能解释光滑与强凸如何决定梯度下降速度', '能推导 L1 的近端软阈值']"
/>

## 1. 凸集

集合 $\mathcal C$ 凸，当任意 $\mathbf x,\mathbf y\in\mathcal C$ 和 $\lambda\in[0,1]$：

$$
\lambda\mathbf x+(1-\lambda)\mathbf y\in\mathcal C.
$$

几何上，集合中任意两点的线段仍在集合内。仿射子空间、范数球、半空间是凸集；环形集合和两个分离区域的并通常不是。

## 2. 凸函数

$f$ 在凸域上凸，当

$$
f(\lambda\mathbf x+(1-\lambda)\mathbf y)
\le
\lambda f(\mathbf x)+(1-\lambda)f(\mathbf y).
$$

图像位于弦下方。可微函数等价的一阶条件：

$$
f(\mathbf y)
\ge
f(\mathbf x)+
\nabla f(\mathbf x)^\top(\mathbf y-\mathbf x).
$$

切线是全局下界。若 $\nabla f(\mathbf x^\star)=0$，立刻有

$$
f(\mathbf y)\ge f(\mathbf x^\star)
$$

对全部 $\mathbf y$ 成立，所以驻点全局最优。

<MlQuiz
  question="可微凸函数在某点梯度为 0，可以得出什么？"
  :options="['该点是全局最小点', '该点必为最大点', '函数必为常数', '只能说明局部信息']"
  :answer="0"
  explanation="凸函数的一阶下界使所有其他点函数值不小于该驻点。"
/>

## 3. Hessian 判据

二次可微函数在凸域上凸，当且仅当

$$
\nabla^2f(\mathbf x)\succeq0
$$

处处成立。例：

$$
f(\mathbf x)
=\frac12\|\mathbf A\mathbf x-\mathbf b\|^2,
$$

$$
\nabla^2f=\mathbf A^\top\mathbf A\succeq0.
$$

若 $\mathbf A$ 满列秩，则 Hessian 正定，解唯一。

## 4. 光滑性与下降引理

若梯度是 $L$-Lipschitz：

$$
\|\nabla f(\mathbf x)-\nabla f(\mathbf y)\|
\le L\|\mathbf x-\mathbf y\|,
$$

则

$$
f(\mathbf y)
\le
f(\mathbf x)+\nabla f(\mathbf x)^\top(\mathbf y-\mathbf x)
+\frac L2\|\mathbf y-\mathbf x\|^2.
$$

取梯度步 $\mathbf y=\mathbf x-\eta\nabla f(\mathbf x)$：

$$
f(\mathbf y)
\le
f(\mathbf x)
-\eta\left(1-\frac{L\eta}{2}\right)
\|\nabla f(\mathbf x)\|^2.
$$

所以 $0<\eta<2/L$ 保证下降（非驻点处）。

::: ml-stop 停下来推一推
取 $\eta=1/L$，下降项系数是多少？得到每步至少下降多少梯度范数平方？
:::

<MlQuiz
  question="L-smooth 凸函数使用固定步长时，哪一范围给出下降引理中的正下降系数？"
  :options="['0<eta<2/L', 'eta<0', 'eta>2L', '任意大步长']"
  :answer="0"
  explanation="系数 eta(1-L eta/2) 在该范围为正。"
/>

## 5. 强凸与收敛速度

$\mu$-强凸：

$$
f(\mathbf y)
\ge
f(\mathbf x)+\nabla f(\mathbf x)^\top(\mathbf y-\mathbf x)
+\frac\mu2\|\mathbf y-\mathbf x\|^2.
$$

它保证唯一最优点，并提供二次下界。对 $L$-smooth、$\mu$-strongly convex 函数，合适梯度下降具有线性（几何）收敛：

$$
f(\mathbf x_t)-f^\star
\le
\left(1-\frac\mu L\right)^t
[f(\mathbf x_0)-f^\star]
$$

（一种常见步长下的界）。条件数 $L/\mu$ 越大，狭长谷越难优化。

## 6. 非光滑凸函数与次梯度

$f(x)=|x|$ 在 0 不可导，但次梯度集合：

$$
\partial|x|=
\begin{cases}
\{1\},&x>0,\\
[-1,1],&x=0,\\
\{-1\},&x<0.
\end{cases}
$$

$\mathbf g\in\partial f(\mathbf x)$ 若

$$
f(\mathbf y)
\ge f(\mathbf x)+\mathbf g^\top(\mathbf y-\mathbf x).
$$

次梯度法能优化凸非光滑目标，但通常比光滑梯度法慢。

## 7. 近端梯度与 L1

复合目标

$$
F(\mathbf x)=f(\mathbf x)+\lambda\|\mathbf x\|_1,
$$

其中 $f$ 光滑。近端梯度先走普通梯度步：

$$
\mathbf v=\mathbf x_t-\eta\nabla f(\mathbf x_t),
$$

再解

$$
\mathbf x_{t+1}
=\arg\min_{\mathbf x}
\left[
\frac1{2\eta}\|\mathbf x-\mathbf v\|^2
+\lambda\|\mathbf x\|_1
\right].
$$

逐坐标解得到软阈值：

$$
x_{t+1,j}
=\operatorname{sign}(v_j)
\max(|v_j|-\eta\lambda,0).
$$

这解释 Lasso 为何产生精确零，而普通梯度加符号项通常只会来回跨过零。

<MlQuiz
  question="L1 近端算子的结果是什么？"
  :options="['软阈值收缩', '矩阵逆', 'Softmax', '硬编码为 1']"
  :answer="0"
  explanation="每个坐标向零收缩 eta lambda，小于阈值者精确变为零。"
/>

## 8. 凸性不等于容易，非凸也不等于不能学

凸问题可能规模巨大、病态或含昂贵约束；神经网络非凸，却可因结构、初始化和随机优化找到好解。凸分析提供可证明基线和局部算子的语言，不应被简化为“凸一定快、非凸一定失败”。

## 9. 动手练习

1. 画凸函数与非凸函数的弦；
2. 数值验证二次目标的下降步长边界；
3. 改变条件数观察梯度下降轨迹；
4. 实现次梯度 Lasso 与近端梯度 Lasso；
5. 比较目标下降、稀疏性和收敛速度。

<MlLessonProgress
  storage-key="s05-convex-optimization"
  next-href="/courses/machine-learning/s06-lagrange-duality"
  next-label="下一篇：Lagrange 对偶"
/>
