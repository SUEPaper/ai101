---
title: 21. Boosting 为什么能把弱学习器变强
description: 从指数损失、函数梯度与二阶近似推导 Boosting
course: machine-learning
---

<MlLessonHero
  eyebrow="第五模块 · 非参数与集成"
  title="Boosting 为什么能把弱学习器变强"
  subtitle="Bagging 并行平均许多模型；Boosting 串行地让下一棵树修正当前模型最需要修正的方向。"
  :diagnostics="['能推导 AdaBoost 的样本权重与步长', '能把梯度提升解释为函数空间下降', '能区分学习率、树深与轮数的作用']"
/>

## 1. 加法模型

Boosting 构造

$$
F_M(\mathbf x)
=F_0(\mathbf x)+\sum_{m=1}^{M}\eta\alpha_mh_m(\mathbf x),
$$

其中 $h_m$ 是弱学习器，常为浅树；$\eta$ 是学习率。第 $m$ 轮固定已有 $F_{m-1}$，选择新函数和步长降低损失：

$$
(\alpha_m,h_m)
=\arg\min_{\alpha,h}
\sum_{i=1}^{n}
\ell\left(y_i,F_{m-1}(\mathbf x_i)+\alpha h(\mathbf x_i)\right).
$$

它是前向分阶段加法建模，不会回头同时重新优化全部旧树。

<MlLab
  kind="boosting"
  title="实验：弱学习器怎样逐轮修正残差"
  description="调节轮数和学习率，观察加法模型如何贴近目标以及何时开始拟合噪声。"
/>

## 2. AdaBoost：从指数损失推导

二分类令 $y_i\in\{-1,+1\}$，使用指数损失：

$$
L(F)=\sum_{i=1}^{n}\exp[-y_iF(\mathbf x_i)].
$$

第 $m$ 轮加入 $\alpha h(\mathbf x)$，其中 $h\in\{-1,+1\}$：

$$
\begin{aligned}
L(F_{m-1}+\alpha h)
&=\sum_i
\exp[-y_iF_{m-1}(\mathbf x_i)]
\exp[-\alpha y_ih(\mathbf x_i)]\\
&=\sum_i w_i^{(m)}
\exp[-\alpha y_ih(\mathbf x_i)],
\end{aligned}
$$

其中

$$
w_i^{(m)}=\exp[-y_iF_{m-1}(\mathbf x_i)].
$$

分类正确时 $y_ih_i=+1$，错误时为 $-1$。记加权错误率

$$
\varepsilon_m
=\frac{\sum_iw_i^{(m)}\mathbb I(y_i\ne h_m(\mathbf x_i))}
{\sum_iw_i^{(m)}}.
$$

则与 $\alpha$ 相关的目标为

$$
(1-\varepsilon_m)e^{-\alpha}
+\varepsilon_me^\alpha.
$$

求导并令零：

$$
-(1-\varepsilon_m)e^{-\alpha}
+\varepsilon_me^\alpha=0,
$$

$$
e^{2\alpha}=\frac{1-\varepsilon_m}{\varepsilon_m},
$$

$$
\boxed{\alpha_m=\frac12\log\frac{1-\varepsilon_m}{\varepsilon_m}}.
$$

更新权重：

$$
w_i^{(m+1)}
\propto
w_i^{(m)}e^{-\alpha_my_ih_m(\mathbf x_i)}.
$$

错分样本被相对放大，下一轮更关注它们。

::: ml-stop 停下来推一推 1
分别代入 $\varepsilon=0.1,0.5,0.8$ 计算 $\alpha$ 的正负。若弱学习器错误率大于 0.5，可怎样解释或翻转它？
:::

<MlQuiz
  question="AdaBoost 中某弱分类器加权错误率恰为 0.5 时，其 alpha 是多少？"
  :options="['负无穷', '0', '0.5', '正无穷']"
  :answer="1"
  explanation="它与随机猜测相当，log((1-e)/e)=log 1=0，不应对集成产生贡献。"
/>

## 3. 梯度提升：在函数空间做最速下降

对一般可微损失，第 $m$ 轮在当前预测处计算负梯度（伪残差）：

$$
r_{im}
=-\left[
\frac{\partial\ell(y_i,F(\mathbf x_i))}
{\partial F(\mathbf x_i)}
\right]_{F=F_{m-1}}.
$$

训练回归树 $h_m$ 去拟合 $(\mathbf x_i,r_{im})$，再更新

$$
F_m(\mathbf x)
=F_{m-1}(\mathbf x)+\eta\gamma_mh_m(\mathbf x).
$$

为什么平方损失时叫“拟合残差”？若

$$
\ell(y,F)=\frac12(y-F)^2,
$$

则

$$
-\frac{\partial\ell}{\partial F}=y-F,
$$

恰好是普通残差。对逻辑损失，伪残差不再是 $y-F$，但仍是最能降低当前损失的函数方向。

<MlQuiz
  question="平方损失梯度提升中的负梯度等于什么？"
  :options="['当前预测减标签', '标签减当前预测，即残差', '标签平方', '树的深度']"
  :answer="1"
  explanation="对 1/2(y-F)^2 关于 F 求导为 F-y，取负得到 y-F。"
/>

## 4. 每个叶子的步长

树把样本分到叶区域 $R_{jm}$。更精确的更新是每个叶子单独找常数：

$$
\gamma_{jm}
=\arg\min_\gamma
\sum_{\mathbf x_i\in R_{jm}}
\ell(y_i,F_{m-1}(\mathbf x_i)+\gamma).
$$

平方损失下：

$$
\gamma_{jm}
=\frac1{|R_{jm}|}\sum_{\mathbf x_i\in R_{jm}}
(y_i-F_{m-1}(\mathbf x_i)),
$$

即叶内平均残差。于是浅树每轮在若干区域做分段常数修正。

::: ml-stop 停下来推一推 2
某叶内残差为 $[2,-1,3]$。平方损失的最佳叶更新是多少？加入学习率 $\eta=0.1$ 后实际加多少？
:::

<MlQuiz
  question="梯度提升树通常使用浅树，一个重要原因是？"
  :options="['每轮只需表达当前误差方向的低阶交互，并由多轮累加', '深树无法计算梯度', '浅树参数一定为零', '浅树不需要数据']"
  :answer="0"
  explanation="浅树作为受控弱学习器，逐轮组合可降低方差和过拟合风险；深度控制单轮交互阶数。"
/>

## 5. 二阶近似为何有用

对当前预测增量 $f(\mathbf x_i)$ 做二阶 Taylor 展开：

$$
\ell(y_i,F_i+f_i)
\approx
\ell(y_i,F_i)+g_if_i+\frac12h_if_i^2,
$$

其中

$$
g_i=\frac{\partial\ell}{\partial F_i},
\qquad
h_i=\frac{\partial^2\ell}{\partial F_i^2}.
$$

若叶 $R$ 输出常数 $w$，并加 $L_2$ 惩罚 $\lambda w^2/2$，与 $w$ 有关的目标：

$$
G_Rw+\frac12(H_R+\lambda)w^2,
$$

$$
G_R=\sum_{i\in R}g_i,\qquad H_R=\sum_{i\in R}h_i.
$$

求导得最优叶权重

$$
\boxed{w_R^\star=-\frac{G_R}{H_R+\lambda}}.
$$

代回可得该叶的目标改善量与 $G_R^2/(H_R+\lambda)$ 成正比。这是现代二阶提升树切分评分的核心来源。

<MlQuiz
  question="二阶提升中 Hessian 累计量 H_R 主要提供什么信息？"
  :options="['损失在当前预测附近的曲率', '特征名称', '树的颜色', '标签排列']"
  :answer="0"
  explanation="二阶导数描述局部曲率，使步长相对纯一阶残差更有尺度感。"
/>

## 6. 学习率、轮数和树深共同控制容量

学习率小意味着每轮只走一小步，通常需要更多树：

$$
F_m=F_{m-1}+\eta f_m,\qquad0<\eta\le1.
$$

常见控制手段：

- 小学习率配合较多轮数；
- 用验证集早停选择轮数；
- 限制树深或叶数控制交互阶数；
- 子采样样本或特征增加随机性；
- 正则化叶权重和最小叶 Hessian。

不能单独问“树越多是否过拟合”，因为答案取决于学习率、深度、噪声和正则化。

::: ml-stop 停下来推一推 3
把学习率缩小 10 倍但轮数不变，模型会怎样？为什么通常需要同时增大轮数？
:::

<MlQuiz
  question="Boosting 中使用早停，通常监控什么？"
  :options="['训练损失是否还能下降', '独立验证指标是否连续若干轮不再改善', '树的颜色是否一致', '每轮随机种子是否相同']"
  :answer="1"
  explanation="训练损失可能继续下降而泛化变差，早停应依据未用于拟合的验证数据。"
/>

## 7. 与随机森林的本质区别

| 维度 | 随机森林 | Boosting |
|---|---|---|
| 训练关系 | 树可并行 | 后一棵依赖前一轮 |
| 核心机制 | 重采样、去相关、平均 | 沿损失方向逐步加法 |
| 主要收益 | 降低高方差 | 逐步降低偏差，也需控方差 |
| 噪声敏感性 | 通常较稳健 | 可能持续追逐难例与异常点 |
| 调参敏感度 | 相对低 | 学习率、轮数、深度强耦合 |

<MlQuiz
  question="哪项最能区分 Boosting 与 Bagging？"
  :options="['Boosting 的基学习器按顺序依赖当前误差，Bagging 基学习器可独立训练', 'Boosting 不使用损失', 'Bagging 只能做回归', 'Boosting 没有超参数']"
  :answer="0"
  explanation="串行纠错与并行平均是两类集成的核心结构差异。"
/>

## 8. 动手练习：用树桩拟合正弦

1. 生成带噪声的 $y=\sin x$；
2. 以均值作为 $F_0$；
3. 每轮用深度 1 的回归树拟合残差；
4. 画第 1、5、20、100 轮的函数与残差；
5. 比较 $\eta=1,0.1,0.01$；
6. 加入几个极端离群点，对比平方损失与更稳健损失。

<MlLessonProgress
  storage-key="21-boosting"
  next-href="/courses/machine-learning/22-mle-map"
  next-label="下一章：MLE 与 MAP 为什么反复出现"
/>
