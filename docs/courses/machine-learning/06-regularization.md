---
course: machine-learning
title: 正则化究竟改变了什么
description: 从谱收缩、约束优化和高斯先验三条路径推导 Ridge，并由 Laplace 先验与近端更新理解 Lasso 稀疏性。
outline: [2, 3]
---

<MlLessonHero number="06" title="正则化究竟改变了什么" question="模型已经能最小化训练误差，为什么还要故意把目标变差？" duration="建议 300–360 分钟" :outcomes="['推导 Ridge 闭式解与奇异值方向收缩', '连接惩罚、约束与 Gaussian MAP 三种解释', '解释 Lasso 的不可导点、软阈值与稀疏解']" />

## 诊断：正则不是修饰项

<MlQuiz question="给 OLS 加 λ‖β‖² 后，训练 SSE 通常怎样？" :options="['一定降低', '不低于未正则 OLS', '保持完全相同', '自动变成 0']" :answer="1" explanation="OLS 已经在全部 β 中最小化训练 SSE；加入约束后只能相同或更高，收益来自稳定性与泛化。" />

<MlQuiz question="Ridge 能否在 X 秩亏时给出唯一系数？" :options="['λ>0 时通常能', '永远不能', '只有 λ=0 能', '取决于 y 是否为整数']" :answer="0" explanation="XᵀX+λI 的特征值都至少为 λ>0，因此正定可逆。" />

<MlQuiz question="为什么标准化常在正则化前完成？" :options="['让 y 变成正态', '同一 λ 才不会因特征单位不同而产生任意惩罚差异', '增加样本数', '使 L1 等于 L2']" :answer="1" explanation="系数尺度依赖特征单位；不标准化时，大单位特征可用小系数表达同样作用，从而被少罚。" />

## 1. Ridge 闭式解

先假设特征已中心化，截距单独处理且不惩罚。Ridge 目标为

$$
J_\lambda(\beta)=\lVert y-X\beta\rVert_2^2+\lambda\lVert\beta\rVert_2^2,
\qquad\lambda\ge0.
$$

梯度为

$$
\nabla J_\lambda
=-2X^T(y-X\beta)+2\lambda\beta.
$$

令其为零：

$$
\begin{aligned}
-2X^Ty+2X^TX\beta+2\lambda\beta&=0\\
(X^TX+\lambda I)\beta&=X^Ty.
\end{aligned}
$$

所以

$$
\boxed{\widehat\beta_\lambda=(X^TX+\lambda I)^{-1}X^Ty}.
$$

Hessian 是 $2(X^TX+\lambda I)$。当 $\lambda>0$ 时，对任意 $v\ne0$：

$$
v^T(X^TX+\lambda I)v=\lVert Xv\rVert^2+\lambda\lVert v\rVert^2>0,
$$

因此目标严格凸，解唯一。

## 2. SVD 显示每个方向收缩多少

令 $X=U\Sigma V^T$，奇异值为 $s_j$。把 Ridge 解写到右奇异向量基：

$$
\begin{aligned}
\widehat\beta_\lambda
&=V(\Sigma^T\Sigma+\lambda I)^{-1}\Sigma^TU^Ty\\
&=\sum_{j=1}^{r}\frac{s_j}{s_j^2+\lambda}(u_j^Ty)v_j.
\end{aligned}
$$

OLS 对应系数是 $(u_j^Ty)/s_j$。因此相对 OLS 的收缩因子为

$$
\boxed{\frac{s_j^2}{s_j^2+\lambda}}.
$$

当 $s_j$ 很小，OLS 要除以小数，噪声被放大；Ridge 把该方向强烈压缩。大奇异值方向保留更多。

<MlLab kind="regularization" />

<div class="ml-stop"><strong>停下来推一推 1</strong>令 $\lambda\to0$ 与 $\lambda\to\infty$，收缩因子分别趋向什么？在秩亏方向 $s_j=0$ 时又是什么？</div>

<MlQuiz question="Ridge 对哪个谱方向收缩最强？" :options="['奇异值最大的方向', '奇异值最小的方向', '所有方向绝对相同', '截距方向']" :answer="1" explanation="相对保留率 s²/(s²+λ) 随 s 增大；小奇异值方向最不稳定，也被压得最强。" />

## 3. 惩罚形式与约束形式

另一种写法是

$$
\min_\beta\lVert y-X\beta\rVert^2
\quad\text{s.t.}\quad \lVert\beta\rVert_2^2\le t.
$$

Lagrangian 为

$$
\mathcal L(\beta,\lambda)
=\lVert y-X\beta\rVert^2+\lambda(\lVert\beta\rVert^2-t),
\qquad\lambda\ge0.
$$

对固定 $\lambda$，其中与 $\beta$ 有关的部分就是 Ridge 惩罚目标。适当正则条件下，每个有效约束半径 $t$ 对应某个 $\lambda$；但二者没有普遍的简单闭式换算，关系依赖数据。

几何上，OLS 等高椭圆首次接触 L2 球的位置是约束解。L2 球边界光滑，接触点通常不会精确落在坐标轴上，所以 Ridge 缩小系数却通常不产生严格的 0。

## 4. 从高斯先验得到 MAP

仍用高斯观测模型

$$
y|\beta\sim\mathcal N(X\beta,\sigma^2I),
$$

再给参数独立零均值高斯先验

$$
\beta\sim\mathcal N(0,\tau^2I).
$$

Bayes 公式在对数下为

$$
\log p(\beta|y,X)=\log p(y|X,\beta)+\log p(\beta)-\log p(y|X).
$$

最后一项与 $\beta$ 无关。取负并删除常数：

$$
-\log p(\beta|y,X)
=\frac1{2\sigma^2}\lVert y-X\beta\rVert^2
+\frac1{2\tau^2}\lVert\beta\rVert^2+C.
$$

乘以正数 $2\sigma^2$：

$$
\lVert y-X\beta\rVert^2
+\frac{\sigma^2}{\tau^2}\lVert\beta\rVert^2.
$$

因此

$$
\boxed{\lambda=\sigma^2/\tau^2}.
$$

噪声越大或先验越集中，正则越强。MAP 是后验众数，不是完整 Bayes 预测；完整预测还要对参数后验积分。

<div class="ml-stop"><strong>停下来推一推 2</strong>为什么把先验均值改成 $m$ 会得到 $\lVert\beta-m\rVert^2$，而不是 $\lVert\beta\rVert^2$？正则化表达的是向哪个参考参数收缩。</div>

## 5. Lasso：从 Laplace 先验到 L1 尖角

Lasso 目标为

$$
\min_\beta\frac12\lVert y-X\beta\rVert^2+\lambda\lVert\beta\rVert_1.
$$

独立 Laplace 先验 $p(\beta_j)\propto\exp(-|\beta_j|/b)$ 的负对数正比于 $\sum_j|\beta_j|$，因此也有 MAP 解释。

$|\beta_j|$ 在 0 处不可导，但有次梯度：

$$
\partial|\beta_j|=
\begin{cases}
\{1\},&\beta_j>0,\\
[-1,1],&\beta_j=0,\\
\{-1\},&\beta_j<0.
\end{cases}
$$

最优条件是

$$
0\in-X^T(y-X\beta)+\lambda\partial\lVert\beta\rVert_1.
$$

对标准正交设计 $X^TX=I$，令 $z=X^Ty$。每个坐标独立求解

$$
\min_{\beta_j}\frac12(\beta_j-z_j)^2+\lambda|\beta_j|.
$$

分三种情况：

- $\beta_j>0$：$\beta_j-z_j+\lambda=0\Rightarrow\beta_j=z_j-\lambda$，要求 $z_j>\lambda$。
- $\beta_j<0$：$\beta_j-z_j-\lambda=0\Rightarrow\beta_j=z_j+\lambda$，要求 $z_j<-\lambda$。
- $\beta_j=0$：要求 $0\in-z_j+\lambda[-1,1]$，即 $|z_j|\le\lambda$。

合并为软阈值：

$$
\boxed{S_\lambda(z)=\operatorname{sign}(z)\max(|z|-\lambda,0)}.
$$

一整段 $[-\lambda,\lambda]$ 被映射为精确 0，这就是稀疏性的计算来源。

## 6. 近端梯度为何自然处理 L1

把目标写成光滑 $f(\beta)=\frac12\lVert y-X\beta\rVert^2$ 与非光滑 $g(\beta)=\lambda\lVert\beta\rVert_1$。先做梯度步

$$
z_t=\beta_t-\eta X^T(X\beta_t-y),
$$

再做 L1 近端映射

$$
\boxed{\beta_{t+1}=S_{\eta\lambda}(z_t)}.
$$

这不是在 0 处硬凑导数，而是精确解一个带二次邻近项的小问题。

## 7. NumPy：Ridge 路径与谱检查

```python
import numpy as np

rng = np.random.default_rng(4)
n = 80
x1 = rng.normal(size=n)
x2 = x1 + .02 * rng.normal(size=n)
X = np.column_stack([x1, x2])
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = 3 * x1 - 2.8 * x2 + .2 * rng.normal(size=n)

def ridge(X, y, regularization):
    matrix = X.T @ X + regularization * np.eye(X.shape[1])
    return np.linalg.solve(matrix, X.T @ y)

lambdas = np.logspace(-5, 3, 30)
path = np.array([ridge(X, y, value) for value in lambdas])
norms = np.linalg.norm(path, axis=1)

assert np.all(np.diff(norms) <= 1e-10)
print(np.linalg.cond(X.T @ X), np.linalg.cond(X.T @ X + np.eye(2)))
```

系数范数随 $\lambda$ 不增，但单个系数不一定单调；相关特征之间可能重新分配权重。

## 8. 练习：实现 Lasso 近端更新

先把 $X$ 的列标准化。用 Lipschitz 常数 $L=\lVert X\rVert_2^2$ 选择 $\eta=1/L$，实现近端梯度并记录目标函数。

<details><summary>参考实现</summary>

```python
def soft_threshold(values, threshold):
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0)

def lasso_proximal_gradient(X, y, regularization, steps=2000):
    beta = np.zeros(X.shape[1])
    learning_rate = 1 / (np.linalg.norm(X, 2) ** 2)
    objectives = []
    for _ in range(steps):
        gradient = X.T @ (X @ beta - y)
        beta = soft_threshold(
            beta - learning_rate * gradient,
            learning_rate * regularization,
        )
        objectives.append(
            .5 * np.sum((y - X @ beta) ** 2)
            + regularization * np.sum(np.abs(beta))
        )
    return beta, np.asarray(objectives)
```

</details>

<div class="ml-stop"><strong>停下来推一推 3</strong>若两个特征几乎相同，Lasso 可能任意选择其中一个，Ridge 倾向共同分担。哪个更适合“稳定解释”？稀疏不自动等于真实变量选择。</div>

## 9. 常见误解与边界

- 正则化不是为了让训练损失更小，而是用偏差交换方差与稳定性。
- $\lambda$ 必须通过验证流程选择，不能用测试集挑。
- 通常不惩罚截距；应先中心化或使用单独惩罚矩阵。
- Lasso 的 0 不证明因果无效；强相关、尺度和样本扰动都会改变选择。
- MAP 点估计不包含完整后验不确定性。

<MlQuiz question="Lasso 为什么能得到精确零系数？" :options="['因为平方损失为零', 'L1 在 0 的次梯度区间与软阈值把一段输入映射为 0', '因为使用高斯先验', '因为矩阵必须满秩']" :answer="1" explanation="L1 尖角使 |z|≤λ 的最优坐标恰为 0。" />

<MlQuiz question="Gaussian 先验方差 τ² 变大，等价 Ridge λ=σ²/τ² 怎样变？" :options="['增大', '减小', '不变', '变成负数']" :answer="1" explanation="先验更宽表示对大系数约束更弱，因此 λ 减小。" />

## 10. 本章小结

- Ridge 让所有谱方向加上 $\lambda$，重点稳定小奇异值方向。
- L2 惩罚、L2 约束和零均值 Gaussian MAP 在适当条件下对应。
- Lasso 的稀疏来自 L1 的非光滑几何与软阈值更新，不等于自动发现真因果变量。

<MlLessonProgress lesson-id="06" next-label="第 07 章：线性模型怎样表达非线性" next-href="/courses/machine-learning/07-basis-functions" />
