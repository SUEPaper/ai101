---
course: machine-learning
title: 多分类为什么得到 Softmax
description: 从 Categorical 似然推导 Softmax、交叉熵和矩阵梯度，解释平移不变性、参数不可辨识与 log-sum-exp。
outline: [2, 3]
---

<MlLessonHero number="10" title="多分类为什么得到 Softmax" question="多个任意实数分数，怎样变成和为 1、可微且可比较的类别概率？" duration="建议 240–300 分钟" :outcomes="['从指数归一化与 Categorical 似然得到 Softmax 交叉熵', '推导矩阵梯度 Xᵀ(P−Y)', '稳定实现 log-sum-exp 并解释参数不可辨识']" />

## 诊断

<MlQuiz question="Softmax 输出的 K 个数满足什么？" :options="['都可为负', '都在 (0,1) 且总和为 1', '最大值一定为 1', '总和等于 K']" :answer="1" explanation="指数为正，归一化后形成 Categorical 概率。" />
<MlQuiz question="给全部 logits 加同一个常数 c，Softmax 怎样变化？" :options="['不变', '每项加 c', '总和变 c', '变成 one-hot']" :answer="0" explanation="分子分母都乘 eᶜ，比例不变。" />
<MlQuiz question="温度 T<1 会让分布怎样？" :options="['更平坦', '更尖锐', '完全不变', '产生负概率']" :answer="1" explanation="用 z/T 放大 logit 差距，最大类别占据更多概率。" />

## 1. 从正分数归一化得到 Softmax

每类线性 logit 为

$$
z_k=x^Tw_k.
$$

用指数把任意实数变成正数，再归一化：

$$
\boxed{p_k=\frac{e^{z_k}}{\sum_{j=1}^Ke^{z_j}}}.
$$

类 $k$ 与类 $r$ 的 log odds 为

$$
\log\frac{p_k}{p_r}=z_k-z_r=x^T(w_k-w_r).
$$

因此多分类仍在线性建模成对 log odds。

<MlLab kind="softmax" />

## 2. Categorical 似然与交叉熵

用 one-hot $y_{ik}\in\{0,1\}$ 表示标签，且 $\sum_ky_{ik}=1$。单样本概率可写成

$$
p(y_i|x_i,W)=\prod_{k=1}^Kp_{ik}^{y_{ik}}.
$$

负 log-likelihood 为

$$
\begin{aligned}
J(W)
&=-\sum_i\sum_ky_{ik}\log p_{ik}\\
&=-\sum_i\sum_ky_{ik}
\left(z_{ik}-\log\sum_je^{z_{ij}}\right)\\
&=\sum_i\left[\log\sum_je^{z_{ij}}-\sum_ky_{ik}z_{ik}\right].
\end{aligned}
$$

因为 one-hot 只选中真实类 $c_i$：

$$
\ell_i=-\log p_{i,c_i}.
$$

<div class="ml-stop"><strong>停下来推一推 1</strong>若模型给真实类概率 0.9、0.5、0.1，对应损失分别是多少？比较概率下降与负对数惩罚的非线性。</div>

<MlQuiz question="one-hot 标签下，单样本交叉熵化简为什么？" :options="['所有概率之和', '真实类别概率的负对数', '最大 logit', '预测类别编号']" :answer="1" explanation="只有真实类位置 y=1，其余项乘 0。" />

## 3. 对 logit 的梯度是 p−y

对单样本，

$$
\ell=-\sum_ky_kz_k+\log\sum_je^{z_j}.
$$

对 $z_r$ 求导：

$$
\frac{\partial\ell}{\partial z_r}
=-y_r+\frac{e^{z_r}}{\sum_je^{z_j}}
=p_r-y_r.
$$

把所有样本堆叠：$X$ 为 <span class="ml-shape">n × d</span>，$W$ 为 <span class="ml-shape">d × K</span>，$Z=XW$、$P$、$Y$ 都是 <span class="ml-shape">n × K</span>。链式法则得到

$$
\boxed{\nabla_WJ=X^T(P-Y)}.
$$

每一行 $P_i-Y_i$ 的和为 0，反映只关心 logit 相对差值。

## 4. Jacobian 揭示类别之间的耦合

Softmax 对 logits 的导数为

$$
\frac{\partial p_k}{\partial z_r}
=p_k(\mathbf1[k=r]-p_r).
$$

矩阵写成

$$
J_{softmax}=\operatorname{diag}(p)-pp^T.
$$

对角为 $p_k(1-p_k)$，非对角为 $-p_kp_r$。提高一个 logit 会提高自身概率并压低其余类别，因为概率总和必须保持 1。

<div class="ml-stop"><strong>停下来推一推 2</strong>证明 $J_{softmax}\mathbf1=0$。这说明沿“所有 logits 同时增加”的方向，概率没有变化。</div>

## 5. 参数不可辨识

给每个类别权重同时加同一向量 $a$：

$$
w_k'=w_k+a.
$$

对输入 $x$，所有 logits 同时加 $x^Ta$，Softmax 不变。因此 $W$ 不唯一。可选择一个参考类令 $w_K=0$，或加约束 $\sum_kw_k=0$。L2 正则也会从等价参数中偏好较小范数代表，但概率模型本身仍有平移对称性。

## 6. log-sum-exp 稳定化

直接计算 $e^{1000}$ 会溢出。令 $m=\max_jz_j$：

$$
\begin{aligned}
\log\sum_je^{z_j}
&=\log\left[e^m\sum_je^{z_j-m}\right]\\
&=m+\log\sum_je^{z_j-m}.
\end{aligned}
$$

最大指数变成 $e^0=1$，其余不超过 1。Softmax 同理：

$$
p_k=\frac{e^{z_k-m}}{\sum_je^{z_j-m}}.
$$

减去 $m$ 是精确恒等变换，不是近似。

## 7. NumPy 实现与梯度检查

```python
import numpy as np

def softmax_cross_entropy(W, X, Y, regularization=0.0):
    logits = X @ W
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probability = shifted - log_normalizer
    loss = -(Y * log_probability).sum()
    loss += .5 * regularization * np.sum(W ** 2)
    probability = np.exp(log_probability)
    gradient = X.T @ (probability - Y) + regularization * W
    return loss, gradient

X = np.array([[1., -1.], [1., 0.], [1., 2.]])
Y = np.eye(3)[[0, 1, 2]]
W = np.array([[1000., 998., 995.], [.2, -.4, .1]])
loss, gradient = softmax_cross_entropy(W, X, Y)
assert np.isfinite(loss) and np.isfinite(gradient).all()
```

## 8. 练习：矩阵有限差分

逐元素扰动 $W_{jk}$，用中心差分检查解析梯度。报告最大相对误差，并测试给所有类别列加同一向量后概率不变。

<details><summary>参考检查</summary>

```python
def check_matrix_gradient(function, W, epsilon=1e-6):
    numerical = np.zeros_like(W)
    for row in range(W.shape[0]):
        for column in range(W.shape[1]):
            step = np.zeros_like(W)
            step[row, column] = epsilon
            numerical[row, column] = (
                function(W + step) - function(W - step)
            ) / (2 * epsilon)
    return numerical
```

</details>

<div class="ml-stop"><strong>停下来推一推 3</strong>温度缩放 $p_k\propto e^{z_k/T}$ 是否改变最大概率类别？$T>0$ 时不改变排序，但改变置信度与交叉熵。</div>

## 9. 常见误解

- Softmax 不会让独立类别变“互斥真实”，互斥性来自 Categorical 标签设定。
- 输出和为 1 不等于概率已校准。
- 减最大值不改变概率，却是必要的数值实现。
- 参数不唯一不妨碍概率唯一；解释单个类别系数必须指定参考。
- 大类别数时完整归一化成本为 $O(K)$，可成为瓶颈。

<MlQuiz question="Softmax 梯度矩阵的形状是什么？" :options="['n×K', 'd×K，与 W 相同', 'K×K', 'd×n']" :answer="1" explanation="Xᵀ(d×n) 乘 (P−Y)(n×K) 得 d×K。" />
<MlQuiz question="减去最大 logit 为什么安全？" :options="['它是近似但误差小', 'Softmax 对共同平移不变', '它让标签变化', '只在二分类安全']" :answer="1" explanation="所有指数共同乘 e⁻ᵐ，归一化比值完全不变。" />

## 10. 小结

Softmax 是多类线性 log odds 的指数归一化；Categorical MLE 给出交叉熵，梯度紧凑地写成 $X^T(P-Y)$。稳定实现必须使用 log-sum-exp。

<MlLessonProgress lesson-id="10" next-label="第 11 章：从犯错更新到最大间隔" next-href="/courses/machine-learning/11-margin-svm" />
