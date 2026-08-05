---
title: 22. MLE 与 MAP 为什么反复出现
description: 从数据分布、似然与先验推导最大似然和最大后验估计
course: machine-learning
---

<MlLessonHero
  eyebrow="第六模块 · 概率与潜变量"
  title="MLE 与 MAP 为什么反复出现"
  subtitle="损失函数常不是凭经验挑选的：先写概率模型，负对数似然就会把它推出来。"
  :diagnostics="['能区分概率、似然与对数似然', '能从常见分布推导 MLE', '能说明 MAP、正则化与完整后验的关系']"
/>

## 1. 同一个式子，谁是变量决定它叫什么

概率模型写成

$$
p(\mathcal D\mid\theta).
$$

- 固定参数 $\theta$，把数据 $\mathcal D$ 看成随机变量，它是数据的概率分布；
- 固定已经观察到的数据，把 $\theta$ 看成待比较对象，它是似然
  $L(\theta;\mathcal D)$。

似然不是“参数的概率分布”，不要求对 $\theta$ 积分为 1。最大似然估计选择

$$
\hat\theta_{\rm MLE}
=\arg\max_\theta p(\mathcal D\mid\theta).
$$

若样本条件独立同分布：

$$
p(\mathcal D\mid\theta)
=\prod_{i=1}^{n}p(x_i\mid\theta).
$$

乘积容易下溢，取对数变为求和：

$$
\ell(\theta)
=\log p(\mathcal D\mid\theta)
=\sum_{i=1}^{n}\log p(x_i\mid\theta).
$$

对数单调递增，所以最大似然与最大对数似然解相同。

<MlLab
  kind="estimate"
  title="实验：样本怎样改变似然曲线"
  description="调节样本数量与成功比例，观察 Bernoulli 似然峰值、曲率与 MAP 收缩。"
/>

<MlQuiz
  question="似然 L(theta;D) 是否必须对 theta 积分为 1？"
  :options="['必须', '不必须；它是固定数据后关于参数的评分函数', '只在分类时必须', '只在 n=1 时必须']"
  :answer="1"
  explanation="概率分布对随机变量归一化；似然固定数据后不自动成为参数分布。"
/>

## 2. Bernoulli MLE：样本均值从哪里来

$x_i\in\{0,1\}$，设

$$
p(x_i\mid\theta)
=\theta^{x_i}(1-\theta)^{1-x_i}.
$$

令 $S=\sum_i x_i$，对数似然：

$$
\ell(\theta)
=S\log\theta+(n-S)\log(1-\theta).
$$

一阶导数：

$$
\ell'(\theta)
=\frac{S}{\theta}-\frac{n-S}{1-\theta}.
$$

令其为零：

$$
S(1-\theta)=(n-S)\theta
\quad\Rightarrow\quad
\boxed{\hat\theta_{\rm MLE}=\frac Sn=\bar x}.
$$

二阶导数

$$
\ell''(\theta)
=-\frac{S}{\theta^2}
-\frac{n-S}{(1-\theta)^2}<0
$$

说明内点解是最大值。若全是 0 或全是 1，最大值在边界；小样本下得到概率 0/1 往往过于自信。

::: ml-stop 停下来推一推 1
抛硬币 5 次全为正面，MLE 是多少？若下一次出现反面，该模型给出的概率和对数损失会怎样？
:::

<MlQuiz
  question="Bernoulli 参数的最大似然估计是什么？"
  :options="['样本方差', '样本中 1 的比例', '固定为 0.5', '样本数倒数']"
  :answer="1"
  explanation="对数似然求导得到 theta=S/n。"
/>

## 3. 高斯 MLE：均值和方差

设 $x_i\sim\mathcal N(\mu,\sigma^2)$。对数似然为

$$
\ell(\mu,\sigma^2)
=-\frac n2\log(2\pi)
-\frac n2\log\sigma^2
-\frac1{2\sigma^2}\sum_i(x_i-\mu)^2.
$$

对 $\mu$ 求导：

$$
\frac{\partial\ell}{\partial\mu}
=\frac1{\sigma^2}\sum_i(x_i-\mu)=0,
$$

所以

$$
\hat\mu_{\rm MLE}=\bar x.
$$

对 $v=\sigma^2$ 求导：

$$
\frac{\partial\ell}{\partial v}
=-\frac n{2v}
+\frac1{2v^2}\sum_i(x_i-\mu)^2=0,
$$

得到

$$
\boxed{
\hat\sigma_{\rm MLE}^2
=\frac1n\sum_i(x_i-\bar x)^2
}.
$$

分母是 $n$ 而不是 $n-1$。后者是无偏样本方差；MLE 追求似然最大，不自动追求无偏。

<MlQuiz
  question="高斯方差 MLE 的分母通常是？"
  :options="['n', 'n-1', 'sqrt(n)', '2n']"
  :answer="0"
  explanation="将均值 MLE 代入并最大化似然得到 1/n；1/(n-1) 是无偏校正。"
/>

## 4. 交叉熵也是最大似然

监督学习中假设样本条件独立：

$$
p(\mathbf y\mid\mathbf X,\theta)
=\prod_i p(y_i\mid\mathbf x_i,\theta).
$$

最大化条件似然等价于最小化平均负对数似然：

$$
\hat\theta
=\arg\min_\theta
-\frac1n\sum_i\log p_\theta(y_i\mid\mathbf x_i).
$$

多分类 one-hot 标签时：

$$
-\sum_k y_{ik}\log p_{ik}
=-\log p_{i,y_i},
$$

正是交叉熵。负对数还把独立证据的乘积变成可加损失，并对“给真实事件极小概率”的过度自信施加强惩罚。

<MlQuiz
  question="多分类交叉熵为何等价于类别条件分布的负对数似然？"
  :options="['one-hot 标签只保留真实类别的 log 概率', 'Softmax 输出都是整数', '平方误差与它恒等', '类别必须只有两个']"
  :answer="0"
  explanation="one-hot 求和中只有真实类别项非零，单样本损失即 -log p(y|x)。"
/>

## 5. MAP：让先验参与点估计

Bayes 公式：

$$
p(\theta\mid\mathcal D)
=\frac{p(\mathcal D\mid\theta)p(\theta)}
{p(\mathcal D)}.
$$

最大后验估计

$$
\hat\theta_{\rm MAP}
=\arg\max_\theta p(\theta\mid\mathcal D)
=\arg\max_\theta
\left[
\log p(\mathcal D\mid\theta)+\log p(\theta)
\right],
$$

因为证据 $p(\mathcal D)$ 与 $\theta$ 无关。

若

$$
\theta\sim\mathcal N(0,\tau^2\mathbf I),
$$

则

$$
-\log p(\theta)
=\frac1{2\tau^2}\|\theta\|_2^2+\text{常数}.
$$

所以高斯先验对应 $L_2$ 正则。Laplace 先验

$$
p(\theta_j)\propto e^{-|\theta_j|/b}
$$

对应 $L_1$ 正则。正则化强度是似然噪声尺度与先验尺度的组合，不应脱离损失归一化比较。

::: ml-stop 停下来推一推 2
先验方差 $\tau^2$ 变小，$L_2$ 惩罚变强还是变弱？用负对数先验的系数说明。
:::

<MlQuiz
  question="零均值各向同性高斯先验的 MAP，通常对应哪种参数惩罚？"
  :options="['L0', 'L1', 'L2', '不产生惩罚']"
  :answer="2"
  explanation="高斯负对数密度的参数相关项是平方范数。"
/>

## 6. Beta—Bernoulli：看见平滑与收缩

取共轭先验

$$
\theta\sim\operatorname{Beta}(\alpha,\beta),
\qquad
p(\theta)\propto
\theta^{\alpha-1}(1-\theta)^{\beta-1}.
$$

观察 $S$ 次成功、$n-S$ 次失败后：

$$
p(\theta\mid\mathcal D)
\propto
\theta^{S+\alpha-1}
(1-\theta)^{n-S+\beta-1},
$$

因此

$$
\theta\mid\mathcal D
\sim\operatorname{Beta}(\alpha+S,\beta+n-S).
$$

当 $\alpha,\beta>1$，MAP 为

$$
\hat\theta_{\rm MAP}
=\frac{S+\alpha-1}{n+\alpha+\beta-2}.
$$

后验均值为

$$
\mathbb E[\theta\mid\mathcal D]
=\frac{S+\alpha}{n+\alpha+\beta}.
$$

它们都像加入伪计数，但不是同一个量。MAP 只是后验最高点，完整 Bayes 预测还会对参数不确定性积分。

<MlQuiz
  question="MAP 与完整后验的区别是什么？"
  :options="['MAP 保留全部参数不确定性', 'MAP 用一个后验众数作点估计，完整后验保留分布', '二者永远相同', '完整后验不使用数据']"
  :answer="1"
  explanation="MAP 压缩成单个最可能参数，丢失宽度、偏斜和多峰等不确定性信息。"
/>

## 7. 后验预测

Bayes 预测不是简单代入 MAP：

$$
p(y_\star\mid\mathbf x_\star,\mathcal D)
=\int
p(y_\star\mid\mathbf x_\star,\theta)
p(\theta\mid\mathcal D)\,d\theta.
$$

积分平均了参数可能性。数据少时后验宽，参数不确定性对预测影响明显；数据多且模型正则时，后验可能集中，点估计近似更合理。

::: ml-stop 停下来推一推 3
两个参数值各有后验概率 0.5，对某事件分别预测 0.1 与 0.9。后验预测概率是多少？它与任选一个 MAP 有何不同？
:::

<MlQuiz
  question="数据量增大时，在常见正则条件下先验对后验的相对影响通常怎样？"
  :options="['通常减弱，似然证据累积', '必然增大', '始终完全不变', '让似然消失']"
  :answer="0"
  explanation="对数似然通常随 n 累加，而固定先验项不随样本数同速增长；但弱可识别问题可能例外。"
/>

## 8. 动手练习：硬币的三种答案

对成功数 $S$、总次数 $n$：

1. 画 Bernoulli 似然和对数似然；
2. 计算 MLE；
3. 使用 Beta(2,2) 与 Beta(20,20) 先验，计算 MAP 与后验均值；
4. 画完整后验和 95% 区间；
5. 让 $n$ 从 5 增长到 500，解释先验影响如何变化；
6. 比较 5 次全成功时 MLE 与后验预测。

<MlLessonProgress
  storage-key="22-mle-map"
  next-href="/courses/machine-learning/23-generative-classifiers"
  next-label="下一章：生成式分类器怎样得到决策边界"
/>
