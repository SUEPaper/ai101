---
title: S06. Lagrange 对偶
description: 从约束惩罚、对偶下界、弱强对偶与 KKT 条件理解机器学习中的对偶问题
course: machine-learning
---

<MlLessonHero
  eyebrow="选修 · 数学工具箱"
  title="Lagrange 对偶"
  subtitle="Lagrange 乘子给每条约束定价；对偶问题寻找最紧的全局下界，并在适当条件下与原问题相遇。"
  :diagnostics="['能从原问题构造 Lagrangian 与对偶函数', '能证明弱对偶', '能使用 KKT 条件解释支持向量与互补松弛']"
/>

## 1. 原问题

考虑

$$
\begin{aligned}
\min_{\mathbf x}\quad &f_0(\mathbf x)\\
\text{s.t.}\quad
&f_i(\mathbf x)\le0,\quad i=1,\ldots,m,\\
&h_j(\mathbf x)=0,\quad j=1,\ldots,p.
\end{aligned}
$$

不等式乘子 $\lambda_i\ge0$，等式乘子 $\nu_j$ 无符号约束。Lagrangian：

$$
\mathcal L(\mathbf x,\boldsymbol\lambda,\boldsymbol\nu)
=f_0(\mathbf x)
+\sum_i\lambda_if_i(\mathbf x)
+\sum_j\nu_jh_j(\mathbf x).
$$

若可行点违反不了 $f_i\le0$，则 $\lambda_if_i\le0$。乘子可解释为约束收紧一小单位时最优值的边际价格。

## 2. 对偶函数是下界

定义

$$
g(\boldsymbol\lambda,\boldsymbol\nu)
=\inf_{\mathbf x}
\mathcal L(\mathbf x,\boldsymbol\lambda,\boldsymbol\nu).
$$

对任意原问题可行 $\widetilde{\mathbf x}$ 和 $\boldsymbol\lambda\ge0$：

$$
\begin{aligned}
g(\boldsymbol\lambda,\boldsymbol\nu)
&\le
\mathcal L(\widetilde{\mathbf x},
\boldsymbol\lambda,\boldsymbol\nu)\\
&=f_0(\widetilde{\mathbf x})
+\sum_i\lambda_if_i(\widetilde{\mathbf x})
+0\\
&\le f_0(\widetilde{\mathbf x}).
\end{aligned}
$$

所以 $g$ 是任意可行目标值、特别是最优值 $p^\star$ 的下界：

$$
g(\lambda,\nu)\le p^\star.
$$

<MlQuiz
  question="为什么不等式约束乘子要求 lambda_i≥0？"
  :options="['使可行点的 lambda_i f_i(x)≤0，从而对偶函数保持下界', '让所有约束等号成立', '方便矩阵求逆', '乘子必须是概率']"
  :answer="0"
  explanation="可行点满足 f_i≤0，非负乘子才能保证 Lagrangian 不高于原目标。"
/>

## 3. 对偶问题与弱对偶

为了得到最紧的下界，最大化：

$$
\begin{aligned}
\max_{\boldsymbol\lambda,\boldsymbol\nu}\quad
&g(\boldsymbol\lambda,\boldsymbol\nu)\\
\text{s.t.}\quad&\boldsymbol\lambda\ge0.
\end{aligned}
$$

对偶最优值 $d^\star$ 总满足

$$
d^\star\le p^\star.
$$

这叫弱对偶，不要求原问题凸。差值 $p^\star-d^\star$ 是对偶间隙。

::: ml-stop 停下来推一推
若找到一个原可行解目标 10、一个对偶可行解目标 9.8，最优原目标位于什么区间？最大可能次优差是多少？
:::

<MlQuiz
  question="弱对偶对一般最小化原问题给出什么关系？"
  :options="['d*≤p*', 'd*≥p*', 'd*=p* 永远成立', '二者无定义']"
  :answer="0"
  explanation="任何对偶可行值是原最优值下界，最大下界仍不超过原最优值。"
/>

## 4. 强对偶与 Slater 条件

强对偶是

$$
d^\star=p^\star.
$$

对凸原问题，若存在严格可行点：

$$
f_i(\mathbf x)<0,\qquad h_j(\mathbf x)=0,
$$

且等式约束仿射，Slater 条件通常保证强对偶并保证对偶最优乘子存在。它是充分条件，不是所有问题的必要条件。

## 5. KKT 条件

在凸且满足适当约束资格时，最优解满足：

1. 原可行：
   $f_i(\mathbf x^\star)\le0,\ h_j(\mathbf x^\star)=0$；
2. 对偶可行：
   $\lambda_i^\star\ge0$；
3. 驻点：

$$
\nabla f_0(\mathbf x^\star)
+\sum_i\lambda_i^\star\nabla f_i(\mathbf x^\star)
+\sum_j\nu_j^\star\nabla h_j(\mathbf x^\star)=0;
$$

4. 互补松弛：

$$
\lambda_i^\star f_i(\mathbf x^\star)=0.
$$

互补松弛意味着：约束若不活跃 $f_i<0$，其价格必须 $\lambda_i=0$；乘子若为正，约束必在边界。

<MlQuiz
  question="互补松弛 lambda_i f_i(x)=0 表明什么？"
  :options="['不活跃约束的乘子为 0；正乘子对应活跃约束', '所有约束都必须违反', '所有乘子都等于 1', '目标梯度必为零且无约束项']"
  :answer="0"
  explanation="两个因子至少一个为零，把约束是否卡住最优点与影子价格联系起来。"
/>

## 6. 一个一维例子

$$
\min_x\quad x^2
\qquad\text{s.t.}\quad x\ge1.
$$

写为 $1-x\le0$：

$$
\mathcal L(x,\lambda)
=x^2+\lambda(1-x),\qquad\lambda\ge0.
$$

对 $x$ 求下确界，驻点

$$
2x-\lambda=0
\quad\Rightarrow\quad x=\lambda/2.
$$

代回：

$$
g(\lambda)
=\lambda-\frac{\lambda^2}{4}.
$$

最大化得 $\lambda^\star=2$，$g(2)=1$。原问题最优 $x^\star=1$、$p^\star=1$，强对偶成立。互补松弛 $2(1-1)=0$。

## 7. SVM 的支持向量为何出现

硬间隔 SVM：

$$
\min_{\mathbf w,b}\frac12\|\mathbf w\|^2
\quad\text{s.t.}\quad
1-y_i(\mathbf w^\top\mathbf x_i+b)\le0.
$$

每个样本有乘子 $\alpha_i\ge0$。互补松弛：

$$
\alpha_i
[1-y_i(\mathbf w^\top\mathbf x_i+b)]=0.
$$

严格位于间隔外的样本满足括号 $<0$，所以 $\alpha_i=0$；只有约束活跃的间隔样本可能 $\alpha_i>0$，它们就是支持向量。由驻点条件：

$$
\mathbf w=\sum_i\alpha_i y_i\mathbf x_i.
$$

因此决策方向完全由支持向量组合。

<MlQuiz
  question="硬间隔 SVM 中，严格位于间隔外的样本为何通常 alpha_i=0？"
  :options="['由互补松弛，非活跃约束乘子必须为 0', '它们没有标签', '核函数删除了它们', '所有样本 alpha 都相同']"
  :answer="0"
  explanation="只有卡住边界的约束具有非零影子价格并影响最优方向。"
/>

## 8. 对偶为何有用

- 给原最优值下界和停止证据；
- 揭示活跃约束与稀疏结构；
- 原变量很多、约束少时可能更易求；
- 只通过内积出现时可使用核技巧；
- 分布式优化可通过约束价格协调子问题。

但对偶也可能不更小、不更易数值求解，非凸问题还可能有间隙。

## 9. 动手练习

1. 完整求解一维约束二次问题；
2. 画 $g(\lambda)$ 并验证它始终不超过原最优值；
3. 对二维投影到半空间问题写 KKT；
4. 推导硬间隔 SVM 的驻点和互补松弛；
5. 用数值解检查原目标、对偶目标与间隙。

<MlLessonProgress
  storage-key="s06-lagrange-duality"
  next-href="/courses/machine-learning/assignments"
  next-label="下一站：八次综合作业"
/>
