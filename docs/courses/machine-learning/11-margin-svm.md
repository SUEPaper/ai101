---
course: machine-learning
title: 从犯错更新到最大间隔
description: 从感知机更新、函数间隔和几何间隔推导硬间隔与软间隔 SVM、hinge loss 和 C 的意义。
outline: [2, 3]
---

<MlLessonHero number="11" title="从犯错更新到最大间隔" question="许多直线都能分对训练样本，为什么要选择离两类都最远的那一条？" duration="建议 300–360 分钟" :outcomes="['解释感知机更新与线性可分条件', '从几何间隔推出硬间隔 SVM', '由松弛变量消元得到 hinge loss 与软间隔目标']" />

## 诊断

<MlQuiz question="把 (w,b) 同时乘 10，决策边界怎样变化？" :options="['不变', '旋转 10 度', '平移 10', '类别反转']" :answer="0" explanation="sign(10(wᵀx+b)) 与原符号相同；函数间隔缩放，几何边界不变。" />
<MlQuiz question="支持向量为何重要？" :options="['它们离边界最远', '它们贴近或违反间隔，决定最优边界', '它们是所有负类', '它们只用于测试']" :answer="1" explanation="远离间隔且分类正确的样本不会改变硬间隔解的局部约束。" />
<MlQuiz question="C 增大表示什么？" :options="['更重惩罚间隔违约', '更强偏好宽间隔且忽略错误', '删除训练样本', '核宽度增大']" :answer="0" explanation="软间隔目标中 C 乘以违约总量；C 大时更不愿容忍训练错误。" />

## 1. 感知机：只在犯错时移动

标签用 $y_i\in\{-1,+1\}$。线性预测为 $\operatorname{sign}(w^Tx+b)$。若

$$
y_i(w^Tx_i+b)\le0,
$$

样本被错分或恰在边界，感知机更新

$$
w\leftarrow w+\eta y_ix_i,
\qquad b\leftarrow b+\eta y_i.
$$

若 $y_i=+1$，把 $w$ 朝 $x_i$ 推；若 $y_i=-1$，朝反方向推。它等价于对感知机损失

$$
\ell_i(w)=\max(0,-y_iw^Tx_i)
$$

做一次次梯度更新。

若数据存在线性分隔向量 $w^*$、样本范数不超过 $R$、单位化间隔至少 $\gamma>0$，感知机犯错次数可界为 $(R/\gamma)^2$。关键不是数据量，而是可分间隔；不可分数据上它可能不停循环。

## 2. 函数间隔与几何间隔

函数间隔为

$$
\widehat\gamma_i=y_i(w^Tx_i+b).
$$

它随参数缩放，不是实际距离。点 $x_i$ 到超平面 $w^Tx+b=0$ 的有符号距离是

$$
\frac{w^Tx_i+b}{\lVert w\rVert_2},
$$

所以带标签几何间隔为

$$
\gamma_i=\frac{y_i(w^Tx_i+b)}{\lVert w\rVert_2}.
$$

将 $(w,b)$ 乘正数不会改变几何间隔。为消除尺度自由，可规定最靠近样本的函数间隔为 1：

$$
y_i(w^Tx_i+b)\ge1.
$$

此时最近点几何间隔为 $1/\lVert w\rVert$，两侧支撑超平面之间宽度为 $2/\lVert w\rVert$。

<MlLab kind="margin" />

<div class="ml-stop"><strong>停下来推一推 1</strong>证明点 $x$ 沿法向量 $w/\lVert w\rVert$ 移动距离 $d=(w^Tx+b)/\lVert w\rVert$ 后会落在超平面上（注意移动方向的符号）。</div>

<MlQuiz question="最大化 1/‖w‖ 等价于最小化什么？" :options="['‖w‖²/2', '训练样本数', 'b²', '预测概率']" :answer="0" explanation="平方和 1/2 不改变最优点，却让目标光滑且推导方便。" />

## 3. 硬间隔 SVM

最大化最小几何间隔等价于

$$
\boxed{
\begin{aligned}
\min_{w,b}\quad&\frac12\lVert w\rVert^2\\
\text{s.t.}\quad&y_i(w^Tx_i+b)\ge1,\ i=1,\ldots,n.
\end{aligned}}
$$

目标凸，约束是线性不等式，因此是凸二次规划。数据不可线性分时可行域为空，硬间隔没有解。

## 4. 松弛变量得到软间隔

允许第 $i$ 个样本违反间隔 $\xi_i\ge0$：

$$
y_i(w^Tx_i+b)\ge1-\xi_i.
$$

目标同时追求宽间隔和小违约：

$$
\boxed{
\min_{w,b,\xi}\frac12\lVert w\rVert^2+C\sum_i\xi_i
\quad\text{s.t.}\quad
y_if_i\ge1-\xi_i,\ \xi_i\ge0.}
$$

对固定 $w,b$，满足约束的最小 $\xi_i$ 是

$$
\xi_i^*=\max(0,1-y_if_i).
$$

代回得到无显式约束形式：

$$
\boxed{
\min_{w,b}\frac12\lVert w\rVert^2
+C\sum_i\max(0,1-y_i(w^Tx_i+b)).}
$$

第二项就是 hinge loss。

- $y_if_i\ge1$：损失 0，在正确间隔外。
- $0<y_if_i<1$：分类正确但间隔不足。
- $y_if_i\le0$：分类错误，损失至少 1。

<div class="ml-stop"><strong>停下来推一推 2</strong>写出 hinge loss 对 w 的次梯度：间隔外为 0，间隔内为 $-y_ix_i$，在折点取二者凸包。</div>

## 5. C 与常见 λ 写法

有些库最小化

$$
\frac\lambda2\lVert w\rVert^2+\frac1n\sum_i\ell_i.
$$

与上式整体乘正数比较，可见 $C$ 与 $1/(n\lambda)$ 同量级。不同库是否对损失取平均、是否惩罚截距，会改变数值含义，不能只复制同一个超参数。

$C$ 大：重视训练违约，边界可能被离群点拉动；$C$ 小：更宽间隔、更强正则，允许更多违约。

## 6. NumPy：hinge 与次梯度

```python
import numpy as np

def svm_objective_gradient(w, X, y, regularization):
    margins = y * (X @ w)
    active = margins < 1
    hinge = np.maximum(0, 1 - margins)
    objective = .5 * regularization * (w @ w) + hinge.mean()
    gradient = regularization * w - (X[active].T @ y[active]) / len(y)
    return objective, gradient

X = np.array([[1., 2.], [1., 1.], [-1., -1.], [-1., -2.]])
y = np.array([1., 1., -1., -1.])
w = np.zeros(2)
for _ in range(300):
    _, gradient = svm_objective_gradient(w, X, y, regularization=.1)
    w -= .05 * gradient
print(w, y * (X @ w))
```

折点不可导会让中心有限差分在恰好 $y_if_i=1$ 时不稳定；梯度检查应避开折点或检查次梯度不等式。

## 7. 练习：手算支持向量

给定二维点 $(2,0),(2,2)$ 为正类，$(0,0),(0,2)$ 为负类。推导最大间隔边界、$w,b$、间隔宽度，并核对哪些点满足 $y_if_i=1$。

<details><summary>答案骨架</summary>

边界为 $x_1=1$。选择规范尺度 $w=(1,0),b=-1$，正类和负类四点都位于支撑超平面，函数间隔为 1，几何半间隔为 1，总宽度为 2。

</details>

<div class="ml-stop"><strong>停下来推一推 3</strong>给每个特征乘不同尺度会改变欧氏距离和最大间隔边界。为什么 SVM 前的特征缩放不是纯数值技巧，而是在定义几何？</div>

## 8. 常见误解

- 感知机只求一个分隔面，SVM 选择最大间隔面。
- 函数间隔可任意缩放，几何间隔才是距离。
- 硬间隔对不可分数据无解；软间隔不是“错误版 SVM”。
- SVM 分数默认不是概率，需要额外校准。
- 最大间隔依赖选定特征空间和度量。

<MlQuiz question="样本 y f(x)=0.4 的 hinge loss 是多少？" :options="['0', '0.4', '0.6', '1.4']" :answer="2" explanation="max(0,1−0.4)=0.6，分类虽正确但仍在间隔内。" />
<MlQuiz question="硬间隔数据不可分时会怎样？" :options="['自动允许错误', '约束不可行，没有解', '退化为逻辑回归', 'C 自动变小']" :answer="1" explanation="必须引入松弛变量或修改特征空间。" />

## 9. 小结

SVM 从分类正确进一步要求几何安全余量。规范化函数间隔后，最大化间隔变成最小化 $\lVert w\rVert^2/2$；松弛变量消元自然得到 hinge loss。

<MlLessonProgress lesson-id="11" next-label="第 12 章：对偶问题为什么带来核技巧" next-href="/courses/machine-learning/12-kernels" />
