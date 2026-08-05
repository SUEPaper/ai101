---
title: S04. 信息论
description: 从自信息、熵、交叉熵、KL 散度与互信息理解机器学习目标
course: machine-learning
---

<MlLessonHero
  eyebrow="选修 · 数学工具箱"
  title="信息论"
  subtitle="越意外的事件携带越多信息；概率模型的训练，就是尽量减少用错误分布编码真实数据的额外代价。"
  :diagnostics="['能从自信息推到熵与交叉熵', '能证明 KL 非负并说明其不对称性', '能用互信息解释特征和标签的依赖']"
/>

## 1. 自信息为何取负对数

希望事件概率 $p$ 的信息量 $I(p)$ 满足：

- 越罕见，信息越大；
- 独立事件联合发生，信息相加；
- 连续变化。

独立事件概率相乘，所以满足加法的函数是对数：

$$
I(x)=-\log p(x).
$$

底数为 2 时单位是 bit，底数为 $e$ 时是 nat。确定事件 $p=1$ 的信息为 0。

## 2. 熵：平均意外程度

离散分布 $p$ 的熵：

$$
H(p)
=\mathbb E_p[-\log p(X)]
=-\sum_xp(x)\log p(x).
$$

二元熵：

$$
H_{\rm bin}(p)
=-p\log p-(1-p)\log(1-p).
$$

$p=0$ 或 1 时不确定性为 0，$p=1/2$ 时最大。对 $K$ 个类别，均匀分布熵最大为 $\log K$。

<MlQuiz
  question="K 类均匀分布的熵是多少？"
  :options="['0', 'log K', 'K²', '1/K']"
  :answer="1"
  explanation="每类概率 1/K，代入得到 -K(1/K)log(1/K)=log K。"
/>

## 3. 交叉熵：用 q 编码 p 的平均代价

真实分布为 $p$，模型分布为 $q$：

$$
H(p,q)
=-\mathbb E_p[\log q(X)]
=-\sum_xp(x)\log q(x).
$$

监督数据从 $p$ 抽样，经验交叉熵

$$
-\frac1n\sum_i\log q_\theta(y_i\mid x_i)
$$

正是负对数似然。若模型给真实事件概率 0，编码长度和损失趋于无穷。

## 4. KL 散度

$$
\operatorname{KL}(p\|q)
=\sum_xp(x)\log\frac{p(x)}{q(x)}.
$$

展开：

$$
\operatorname{KL}(p\|q)
=H(p,q)-H(p).
$$

训练时 $p$ 固定，最小化交叉熵等价于最小化 $\operatorname{KL}(p\|q)$。

### KL 为什么非负

利用 $\log u\le u-1$：

$$
\begin{aligned}
-\operatorname{KL}(p\|q)
&=\sum_xp(x)\log\frac{q(x)}{p(x)}\\
&\le\sum_xp(x)\left(\frac{q(x)}{p(x)}-1\right)\\
&=\sum_xq(x)-\sum_xp(x)=0.
\end{aligned}
$$

所以 $\operatorname{KL}(p\|q)\ge0$，且在几乎处处 $p=q$ 时取 0。

::: ml-stop 停下来推一推
若 $p$ 在某事件上为正而 $q=0$，$\operatorname{KL}(p\|q)$ 会怎样？这解释了哪种交叉熵惩罚？
:::

<MlQuiz
  question="KL(p||q) 是否等于 KL(q||p)？"
  :options="['总是相等', '一般不相等，所以 KL 不是距离度量', '只要维度相同就相等', '二者都恒为零']"
  :answer="1"
  explanation="期望分别在 p、q 下计算，零概率覆盖和关注区域不同。"
/>

## 5. 联合熵、条件熵与互信息

链式法则：

$$
H(X,Y)=H(X)+H(Y\mid X).
$$

互信息：

$$
I(X;Y)
=\operatorname{KL}
(p(x,y)\|p(x)p(y)).
$$

等价形式：

$$
I(X;Y)
=H(X)-H(X\mid Y)
=H(Y)-H(Y\mid X).
$$

它衡量知道一个变量后另一个变量不确定性减少多少。$I(X;Y)=0$ 当且仅当独立（在常规条件下）。

<MlQuiz
  question="互信息 I(X;Y) 最直接衡量什么？"
  :options="['知道 Y 后 X 不确定性减少多少', 'X 与 Y 的均值差', '样本数量', '梯度范数']"
  :answer="0"
  explanation="I=H(X)-H(X|Y)，也等于联合分布与独立分布乘积的 KL。"
/>

## 6. 决策树的信息增益

节点标签熵为 $H(Y)$，按特征切分 $S$ 后剩余条件熵：

$$
H(Y\mid S)
=\sum_sp(s)H(Y\mid S=s).
$$

信息增益：

$$
H(Y)-H(Y\mid S)=I(Y;S).
$$

所以决策树的熵下降，就是切分结果与标签之间的经验互信息。

## 7. KL 的方向会改变行为

$\operatorname{KL}(p\|q)$ 会强烈惩罚 $p>0,q\approx0$，倾向覆盖 $p$ 的所有模式；$\operatorname{KL}(q\|p)$ 对 $q$ 放质量但 $p$ 很小处惩罚强，常倾向集中于某个高密度模式。这个“覆盖/寻模”描述是典型直觉，不是所有优化条件下的绝对定律。

## 8. 动手练习

1. 画二元熵曲线；
2. 对两个离散分布计算两个方向的 KL；
3. 验证交叉熵等于熵加 KL；
4. 从样本估计类别熵和决策树信息增益；
5. 构造 $X=Y$ 与独立 $X,Y$，比较互信息。

<MlLessonProgress
  storage-key="s04-information-theory"
  next-href="/courses/machine-learning/s05-convex-optimization"
  next-label="下一篇：凸优化"
/>
