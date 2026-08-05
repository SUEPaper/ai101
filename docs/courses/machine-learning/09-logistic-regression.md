---
course: machine-learning
title: 逻辑回归为什么是这个形式
description: 从 Bernoulli 似然和线性 log odds 推导 Sigmoid、交叉熵、梯度、Hessian、凸性与 Newton 更新。
outline: [2, 3]
---

<MlLessonHero number="09" title="逻辑回归为什么是这个形式" question="为什么二分类概率不是直接等于线性分数，而要先经过 Sigmoid？" duration="建议 320–380 分钟" :outcomes="['从线性 log odds 得到 Sigmoid', '逐步推导 Bernoulli 负对数似然、梯度与 Hessian', '证明目标凸并解释完全分离与正则化']" />

## 诊断

<MlQuiz question="log odds log[p/(1−p)] 的取值范围是什么？" :options="['[0,1]', '整个实数轴', '非负实数', '只取整数']" :answer="1" explanation="p 从 0 到 1 时，odds 从 0 到无穷，log odds 覆盖整个实数轴，适合与线性分数相等。" />
<MlQuiz question="逻辑回归是回归还是分类？" :options="['只能回归连续 y', '它对 log odds 做线性建模，用于概率分类', '名字决定它不能分类', '只做聚类']" :answer="1" explanation="“回归”指对参数化关系的建模；输出是 Bernoulli 概率。" />
<MlQuiz question="二分类交叉熵的最小可能值是多少？" :options="['0', '−1', '1', '无下界']" :answer="0" explanation="当每个正确类别概率趋近 1 时损失趋近 0，但有限参数未必达到。" />

## 1. 线性建模 log odds

直接令 $p(x)=x^T\beta$ 会越出 $[0,1]$。把概率变成 odds：

$$
\operatorname{odds}(x)=\frac{p(x)}{1-p(x)}\in(0,\infty).
$$

再取对数：

$$
\log\frac{p(x)}{1-p(x)}=x^T\beta=z.
$$

反解 $p$：

$$
\begin{aligned}
\frac p{1-p}&=e^z\\
p&=e^z(1-p)\\
p(1+e^z)&=e^z\\
\boxed{p=\frac{e^z}{1+e^z}=\frac1{1+e^{-z}}=\sigma(z)}.
\end{aligned}
$$

因此每个 $\beta_j$ 的解释是：其他特征不变时，$x_j$ 增加 1，log odds 增加 $\beta_j$，odds 乘以 $e^{\beta_j}$。

<MlLab kind="logistic" />

## 2. Bernoulli 似然统一两种标签

$Y_i\in\{0,1\}$ 且 $p_i=\sigma(x_i^T\beta)$。Bernoulli 概率质量函数写成

$$
p(y_i|x_i,\beta)=p_i^{y_i}(1-p_i)^{1-y_i}.
$$

取联合 log-likelihood：

$$
\ell(\beta)=\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)].
$$

最小化负 log-likelihood：

$$
\boxed{J(\beta)=-\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]}.
$$

这是二元交叉熵。若 $y_i=1$，只剩 $-\log p_i$；若 $y_i=0$，只剩 $-\log(1-p_i)$。

利用

$$
\log\sigma(z)=z-\log(1+e^z),
\quad
\log[1-\sigma(z)]=-\log(1+e^z),
$$

单样本损失也可写成稳定的 softplus 形式：

$$
\ell_i=\log(1+e^{z_i})-y_iz_i.
$$

<div class="ml-stop"><strong>停下来推一推 1</strong>把 $y=1$ 和 $y=0$ 分别代入 softplus 形式，验证它与两种交叉熵分支一致。</div>

<MlQuiz question="若 y=1 且模型给 p=0.01，损失为何很大？" :options="['−log(0.01) 很大', '平方概率很大', '因为 p>1', '因为标签不是整数']" :answer="0" explanation="负对数对自信但错误的概率施加强惩罚。" />

## 3. 梯度为何是 Xᵀ(p−y)

先对单样本 logit $z$ 求导。Sigmoid 导数是

$$
\sigma'(z)=\sigma(z)[1-\sigma(z)]=p(1-p).
$$

对交叉熵直接求导：

$$
\begin{aligned}
\frac{d\ell_i}{dz_i}
&=-\left[y_i\frac1{p_i}p_i(1-p_i)
+(1-y_i)\frac{-1}{1-p_i}p_i(1-p_i)\right]\\
&=-[y_i(1-p_i)-(1-y_i)p_i]\\
&=p_i-y_i.
\end{aligned}
$$

又因为 $z_i=x_i^T\beta$，$\nabla_\beta z_i=x_i$：

$$
\nabla_\beta\ell_i=(p_i-y_i)x_i.
$$

累加得到

$$
\boxed{\nabla J(\beta)=X^T(p-y)}.
$$

形状：$X^T$ 为 <span class="ml-shape">p × n</span>，$p-y$ 为 <span class="ml-shape">n × 1</span>，梯度为 <span class="ml-shape">p × 1</span>。

## 4. Hessian、凸性与 Newton

$p_i$ 对 $\beta$ 的梯度是 $p_i(1-p_i)x_i$。令

$$
W=\operatorname{diag}(p_i(1-p_i)),
$$

则

$$
\boxed{\nabla^2J(\beta)=X^TWX}.
$$

对任意 $v$：

$$
v^TX^TWXv=(Xv)^TW(Xv)=\sum_i p_i(1-p_i)(x_i^Tv)^2\ge0.
$$

因此负 log-likelihood 是凸函数。加入 L2 后

$$
J_\lambda=J+\frac\lambda2\lVert\beta\rVert^2,
\quad
g=X^T(p-y)+\lambda\beta,
\quad
H=X^TWX+\lambda I.
$$

Newton 更新解

$$
H\Delta=-g,
\qquad \beta\leftarrow\beta+\Delta.
$$

这也称 IRLS，因为每轮 Hessian 相当于带权最小二乘曲率。

<div class="ml-stop"><strong>停下来推一推 2</strong>$p_i$ 接近 0 或 1 时，权重 $p_i(1-p_i)$ 怎样？这些样本对局部曲率贡献变小，这与完全分离中的曲率退化有关。</div>

## 5. 完全分离：下确界存在，有限最优参数不存在

若存在 $v$ 使所有正类 $x_i^Tv>0$、负类 $x_i^Tv<0$，沿 $\beta=cv$ 让 $c\to\infty$，正确类别概率都趋近 1，损失趋近 0。但任何有限 $c$ 都有正损失。

所以目标的下确界为 0，却没有有限参数达到。优化器表现为系数持续增大、损失缓慢下降、Hessian 越来越病态。L2 正则加入随 $\lVert\beta\rVert^2$ 增大的代价，恢复有限唯一解。

## 6. 稳定 NumPy 实现与梯度检查

```python
import numpy as np

def sigmoid(logits):
    logits = np.asarray(logits, dtype=float)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1 + exp_logits)
    return result

def logistic_loss_gradient(beta, X, y, regularization=0.0):
    logits = X @ beta
    loss = np.logaddexp(0, logits).sum() - y @ logits
    loss += .5 * regularization * beta @ beta
    probability = sigmoid(logits)
    gradient = X.T @ (probability - y) + regularization * beta
    return loss, gradient

def finite_difference(function, beta, epsilon=1e-6):
    result = np.zeros_like(beta)
    for j in range(len(beta)):
        step = np.zeros_like(beta)
        step[j] = epsilon
        result[j] = (function(beta + step) - function(beta - step)) / (2 * epsilon)
    return result
```

`np.logaddexp(0,z)` 稳定计算 $\log(1+e^z)$，避免大正数 `exp(z)` 溢出。

## 7. 练习：实现 Newton 逻辑回归

实现损失、梯度、Hessian 与阻尼 Newton。每步用线搜索确保损失下降；在完全分离数据上比较无正则和 L2。

<details><summary>Hessian 与更新骨架</summary>

```python
probability = sigmoid(X @ beta)
weights = probability * (1 - probability)
hessian = X.T @ (weights[:, None] * X) + regularization * np.eye(X.shape[1])
direction = np.linalg.solve(hessian, -gradient)
```

</details>

<div class="ml-stop"><strong>停下来推一推 3</strong>若同时把所有样本权重复制两份，未正则 MLE 是否改变？加入固定 λ 后呢？数据项翻倍但正则不变，等效正则强度会减半。</div>

## 8. 常见误解

- 线性的是 log odds，概率曲线不是直线。
- 凸不等于一定有有限最优解；完全分离是反例。
- 系数大不等于特征重要，尺度和相关性会影响数值。
- 训练交叉熵低不保证概率校准或阈值决策成本低。
- 正则截距需谨慎；通常只惩罚特征系数。

<MlQuiz question="逻辑回归 Hessian 为什么半正定？" :options="['因为标签都为正', '它是 XᵀWX 且 W 对角非负', '因为 β=0', '因为 Sigmoid 线性']" :answer="1" explanation="任意二次型等于 Σwᵢ(xᵢᵀv)²≥0。" />
<MlQuiz question="完全分离时无正则 MLE 的参数会怎样？" :options="['有限收敛到 0', '范数趋向无穷，损失趋近 0 但不达到', 'Hessian 恒等于 I', '概率都变 0.5']" :answer="1" explanation="沿分离方向持续放大即可让似然继续提高，没有有限最大值。" />

## 9. 小结

逻辑回归把线性分数放在无界 log odds 空间，通过 Sigmoid 得到概率。Bernoulli MLE 自动产生交叉熵、$X^T(p-y)$ 梯度和 $X^TWX$ 曲率。

<MlLessonProgress lesson-id="09" next-label="第 10 章：多分类为什么得到 Softmax" next-href="/courses/machine-learning/10-softmax" />
