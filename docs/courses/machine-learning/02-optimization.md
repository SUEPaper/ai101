---
course: machine-learning
title: 参数为什么能被算法学出来
description: 从一阶 Taylor 展开推导最速下降，证明二次目标的稳定学习率范围，并连接 Newton、SGD、Momentum 与 Adam。
outline: [2, 3]
---

<MlLessonHero number="02" title="参数为什么能被算法学出来" question="既然目标函数只是一个数，算法怎样知道参数下一步往哪里走？" duration="建议 240–300 分钟" :outcomes="['从 Taylor 展开推导负梯度方向', '推出二次目标的稳定学习率与条件数影响', '区分 GD、Newton、SGD、Momentum 和 Adam 的更新信息']" />

## 先把“下降”说准确

设目标 $J:\mathbb R^p\to\mathbb R$。我们寻找 $\theta^\star\in\arg\min J(\theta)$。优化器只处理已经给定的目标；目标是否代表真实任务，仍是第 01 章的问题。

<MlQuiz question="梯度 ∇J(θ) 指向哪里？" :options="['局部上升最快的方向', '局部下降最快的方向', '全局最优点', 'Hessian 的最小特征向量']" :answer="0" explanation="梯度是局部最陡上升方向，所以欧氏范数约束下的最陡下降方向是负梯度。" />

<MlQuiz question="∇J(θ)=0 能否推出 θ 是全局最小值？" :options="['总能', '只有一维能', '不能，可能是极大值或鞍点；凸性等条件才提供保证', '只要学习率小就能']" :answer="2" explanation="一阶驻点只说局部线性变化为零；还要检查曲率和全局结构。" />

<MlQuiz question="特征缩放为什么会影响梯度下降？" :options="['它改变标签', '它改变损失面的曲率尺度与条件数', '它让样本独立', '它自动增加正则化']" :answer="1" explanation="不同尺度会制造狭长等高线，单一学习率要兼顾陡峭和扁平方向。" />

## 1. 从 Taylor 展开推出负梯度

在 $\theta$ 附近走一个小位移 $\Delta$。一阶 Taylor 展开给出

$$
J(\theta+\Delta)=J(\theta)+\nabla J(\theta)^\top\Delta+o(\lVert\Delta\rVert_2).
$$

忽略高阶小量后，要让目标下降，就要让内积 $\nabla J(\theta)^\top\Delta$ 尽量负。若限制每步长度 $\lVert\Delta\rVert_2\le\varepsilon$，Cauchy–Schwarz 不等式给出

$$
\nabla J^\top\Delta\ge -\lVert\nabla J\rVert_2\lVert\Delta\rVert_2
\ge-\varepsilon\lVert\nabla J\rVert_2.
$$

等号在

$$
\Delta^\star=-\varepsilon\frac{\nabla J}{\lVert\nabla J\rVert_2}
$$

处取得。因此欧氏距离定义下，负梯度是局部最速下降方向。把步长吸收到学习率 $\eta_t$ 中，得到

$$
\boxed{\theta_{t+1}=\theta_t-\eta_t\nabla J(\theta_t)}.
$$

“最速”只在当前点、足够小的邻域和选定范数下成立，不保证一步到全局最优。

<div class="ml-stop"><strong>停下来推一推 1</strong>若把步长限制改为 $\lVert\Delta\rVert_1\le\varepsilon$，最优位移会集中在哪个坐标？提示：与梯度绝对值最大的坐标有关。梯度方向依赖你如何度量“同样大的一步”。</div>

## 2. 一维二次函数给出精确稳定条件

考虑

$$
J(w)=\frac12\lambda w^2,\qquad \lambda>0.
$$

梯度为 $J'(w)=\lambda w$。代入更新：

$$
\begin{aligned}
w_{t+1}&=w_t-\eta\lambda w_t\\
&=(1-\eta\lambda)w_t.
\end{aligned}
$$

递推展开：

$$
w_t=(1-\eta\lambda)^t w_0.
$$

要使 $w_t\to0$，必须且只需

$$
|1-\eta\lambda|<1
\iff -1<1-\eta\lambda<1
\iff \boxed{0<\eta<\frac2\lambda}.
$$

- $0<\eta<1/\lambda$：同侧单调靠近。
- $1/\lambda<\eta<2/\lambda$：跨过最优点振荡，但振幅缩小。
- $\eta=2/\lambda$：在两侧等幅振荡。
- $\eta>2/\lambda$：振幅增长并发散。

<MlLab kind="optimizer" />

<MlQuiz question="对 J(w)=λw²/2，η=1/λ 时会发生什么？" :options="['一步到 w=0', '发散', '等幅振荡', '梯度为 0 但 w 不变']" :answer="0" explanation="更新倍率 1−ηλ=0，所以精确二次的一维问题一步到达最优点。" />

## 3. 多维二次目标与条件数

令

$$
J(\theta)=\frac12\theta^\top H\theta,
$$

其中 $H$ 对称正定。梯度为 $H\theta$。把 $H$ 做特征分解 $H=Q\Lambda Q^\top$，并令 $z_t=Q^\top\theta_t$：

$$
\begin{aligned}
\theta_{t+1}&=(I-\eta H)\theta_t,\\
z_{t+1}&=(I-\eta\Lambda)z_t.
\end{aligned}
$$

这把多维问题拆成 $p$ 个独立的一维递推：

$$
z_{t+1,j}=(1-\eta\lambda_j)z_{t,j}.
$$

所有方向都稳定要求

$$
0<\eta<\frac{2}{\lambda_{\max}}.
$$

但最平缓方向 $\lambda_{\min}$ 收敛最慢。条件数

$$
\kappa(H)=\frac{\lambda_{\max}}{\lambda_{\min}}
$$

越大，损失面越狭长，允许步长由最陡方向限制，而总迭代数由最平方向拖慢。标准化和预条件的本质是改善这些方向的尺度。

## 4. 光滑函数为什么能保证一步下降

若梯度是 $L$-Lipschitz，即

$$
\lVert\nabla J(u)-\nabla J(v)\rVert_2\le L\lVert u-v\rVert_2,
$$

可以得到下降引理：

$$
J(v)\le J(u)+\nabla J(u)^\top(v-u)+\frac L2\lVert v-u\rVert_2^2.
$$

令 $v=u-\eta\nabla J(u)$：

$$
J(v)\le J(u)-\eta\left(1-\frac{L\eta}{2}\right)\lVert\nabla J(u)\rVert_2^2.
$$

所以 $0<\eta<2/L$ 时，非零梯度会让这个上界严格下降。二次函数中 $L=\lambda_{\max}$，与刚才的精确条件吻合。

<div class="ml-stop"><strong>停下来推一推 2</strong>在上式取 $\eta=1/L$，证明每步至少下降 $\lVert\nabla J(u)\rVert^2/(2L)$。注意这仍不说明非凸函数最终到达全局最优。</div>

## 5. Newton 法来自二阶局部模型

二阶 Taylor 近似为

$$
m(\Delta)=J(\theta)+g^\top\Delta+\frac12\Delta^\top H\Delta,
$$

其中 $g=\nabla J(\theta)$、$H=\nabla^2J(\theta)$。对 $\Delta$ 求梯度并令其为零：

$$
\nabla_\Delta m=g+H\Delta=0
\quad\Longrightarrow\quad
\boxed{\Delta_N=-H^{-1}g}.
$$

实现时不求逆，而解线性方程 $H\Delta=-g$。若 $H$ 正定，$g^\top\Delta_N=-g^\top H^{-1}g<0$，它是下降方向。非凸区域的 Hessian 可能不正定，需要阻尼、信赖域或修改曲率。

Newton 用曲率校正坐标尺度，局部收敛快，但构造和求解 $p\times p$ Hessian 通常需要 $O(p^2)$ 内存与 $O(p^3)$ 直接求解成本。

## 6. 从全量梯度到随机梯度

经验风险常写成

$$
J(\theta)=\frac1n\sum_{i=1}^n\ell_i(\theta).
$$

全量梯度每步读取全部样本。均匀抽一个索引 $I$，随机梯度 $g_I=\nabla\ell_I(\theta)$ 满足

$$
\mathbb E_I[g_I]
=\frac1n\sum_{i=1}^n\nabla\ell_i(\theta)
=\nabla J(\theta).
$$

它是无偏估计，但有方差。mini-batch 平均 $B$ 个近似独立梯度后，方差约缩小为原来的 $1/B$，代价是每步计算增加。

### Momentum

一种常见写法是

$$
v_t=\beta v_{t-1}+(1-\beta)g_t,
\qquad
\theta_{t+1}=\theta_t-\eta v_t.
$$

递推展开可见 $v_t$ 是历史梯度的指数加权平均。方向长期一致时累积，方向来回变化时抵消，因此能缓解狭长谷底中的横向振荡。

### Adam

Adam 同时维护一阶与二阶矩：

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\\
\widehat m_t&=m_t/(1-\beta_1^t),\\
\widehat v_t&=v_t/(1-\beta_2^t),\\
\theta_{t+1}&=\theta_t-\eta\widehat m_t/(\sqrt{\widehat v_t}+\epsilon).
\end{aligned}
$$

$g_t^2$、除法和平方根都逐坐标进行。偏差修正来自 $m_0=v_0=0$：早期指数平均被系统性拉向 0。

## 7. NumPy 对照：同一二次目标的三种路径

```python
import numpy as np

H = np.array([[8.0, 0.0], [0.0, 0.5]])

def loss(theta):
    return 0.5 * theta @ H @ theta

def gradient(theta):
    return H @ theta

def gradient_descent(theta, learning_rate, steps):
    path = [theta.copy()]
    for _ in range(steps):
        theta = theta - learning_rate * gradient(theta)
        path.append(theta.copy())
    return np.asarray(path)

stable = gradient_descent(np.array([2.0, 2.0]), 0.20, 30)
unstable = gradient_descent(np.array([2.0, 2.0]), 0.27, 15)

assert 0.20 < 2 / np.linalg.eigvalsh(H).max()
assert 0.27 > 2 / np.linalg.eigvalsh(H).max()
print(loss(stable[-1]), loss(unstable[-1]))
```

这里 $\lambda_{\max}=8$，稳定上界是 $2/8=0.25$。代码不仅“跑优化”，还直接验证推导给出的分界。

<div class="ml-stop"><strong>停下来推一推 3</strong>若把第一个特征除以 4，相应二次曲率从 8 变为多少量级？这为何可能允许更大学习率？注意参数坐标也随特征定义改变。</div>

## 8. 练习：实现优化器并验证公式

实现 GD、Momentum 和 Newton 更新，在 $H=\operatorname{diag}(8,0.5)$ 上记录每步损失。要求用断言验证：GD 在 $\eta=0.2$ 收敛、$\eta=0.27$ 发散；Newton 对精确二次函数一步到达零点。

<details><summary>公式骨架</summary>

GD 使用 $H\theta$；Momentum 对梯度做指数平均；Newton 解 $H\Delta=-H\theta$。

</details>

<details><summary>参考实现</summary>

```python
def run_optimizer(kind, theta, steps=30, learning_rate=.2, momentum=.9):
    theta = np.asarray(theta, dtype=float)
    velocity = np.zeros_like(theta)
    losses = []
    for _ in range(steps):
        losses.append(loss(theta))
        grad = gradient(theta)
        if kind == "gd":
            theta -= learning_rate * grad
        elif kind == "momentum":
            velocity = momentum * velocity + (1 - momentum) * grad
            theta -= learning_rate * velocity
        elif kind == "newton":
            theta += np.linalg.solve(H, -grad)
        else:
            raise ValueError("unknown optimizer")
    return theta, np.asarray(losses)

theta_newton, newton_losses = run_optimizer("newton", [2, 2], steps=2)
assert np.linalg.norm(theta_newton) < 1e-12
```

</details>

## 9. 条件、局限与检查清单

- 梯度正确不等于学习率正确；先检查数值梯度，再检查稳定范围。
- 损失下降不等于验证风险下降。
- SGD 无偏不等于每一步都朝下降方向；方差是它的组成部分。
- Adam 的坐标自适应不能修复错误目标、数据泄漏或坏梯度。
- 非凸目标中驻点可能是局部极小、局部极大或鞍点。

<MlQuiz question="mini-batch 从 B=1 增到 B=16，在近似独立时梯度方差约变成多少？" :options="['16 倍', '1/16', '1/4', '不变']" :answer="1" explanation="独立估计的平均方差除以 B；标准差则约除以 √B。" />

<MlQuiz question="Newton 步为什么通常不显式计算 H⁻¹？" :options="['逆矩阵不存在于数学中', '解线性方程更稳定也更高效', '因为 Newton 不使用 Hessian', '为了让梯度无偏']" :answer="1" explanation="需要的是 HΔ=−g 的解；显式求逆增加计算与数值误差。" />

## 10. 本章小结

- 负梯度来自一阶局部模型与步长范数约束。
- 二次目标的稳定范围由最大曲率决定，速度受条件数限制。
- Newton 校正曲率；SGD 用有噪声的低成本梯度；Momentum/Adam 重组历史梯度，但不改变目标本身。

<MlLessonProgress lesson-id="02" next-label="第 03 章：训练误差为什么不等于未来误差" next-href="/courses/machine-learning/03-generalization" />
