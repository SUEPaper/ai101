---
title: S03. 概率统计
description: 从条件概率、期望方差到大数定律和中心极限定理建立统计直觉
course: machine-learning
---

<MlLessonHero
  eyebrow="选修 · 数学工具箱"
  title="概率统计"
  subtitle="概率描述生成数据前的不确定性；统计从有限样本反推分布，并量化结论会有多不稳定。"
  :diagnostics="['能用条件概率与 Bayes 公式反转证据', '能使用全期望和全方差公式', '能区分大数定律、中心极限定理与置信区间']"
/>

## 1. 条件概率与 Bayes 公式

$$
p(A\mid B)=\frac{p(A\cap B)}{p(B)}.
$$

乘法规则：

$$
p(A\cap B)=p(A\mid B)p(B)=p(B\mid A)p(A).
$$

因此

$$
\boxed{
p(A\mid B)
=\frac{p(B\mid A)p(A)}{p(B)}
}.
$$

若 $\{A_k\}$ 构成划分：

$$
p(B)=\sum_kp(B\mid A_k)p(A_k).
$$

低基率事件即使检测灵敏度高，阳性后的真实概率也可能不高，必须把先验基率纳入。

<MlQuiz
  question="Bayes 公式中的 p(A) 常被称为什么？"
  :options="['先验概率', '似然归一化后的梯度', '样本方差', '损失函数']"
  :answer="0"
  explanation="观察 B 前对 A 的信念是先验，p(B|A) 提供证据。"
/>

## 2. 随机变量、期望和方差

离散期望：

$$
\mathbb E[X]=\sum_xx\,p(x).
$$

线性性不要求独立：

$$
\mathbb E[aX+bY]=a\mathbb E[X]+b\mathbb E[Y].
$$

方差：

$$
\operatorname{Var}(X)
=\mathbb E[(X-\mathbb E[X])^2]
=\mathbb E[X^2]-\mathbb E[X]^2.
$$

和的方差：

$$
\operatorname{Var}(X+Y)
=\operatorname{Var}(X)+\operatorname{Var}(Y)
+2\operatorname{Cov}(X,Y).
$$

只有不相关时协方差项才为 0；独立蕴含不相关，反向一般不成立。

## 3. 全期望与全方差

$$
\mathbb E[X]
=\mathbb E_Y[\mathbb E[X\mid Y]].
$$

全方差：

$$
\boxed{
\operatorname{Var}(X)
=\mathbb E_Y[\operatorname{Var}(X\mid Y)]
+\operatorname{Var}_Y(\mathbb E[X\mid Y])
}.
$$

第一项是组内平均方差，第二项是组间均值方差。偏差—方差分解、混合模型和分层分析都会反复出现这条结构。

::: ml-stop 停下来推一推
两个等概率群体均值为 0、10，组内方差都为 4。总方差是多少？
:::

<MlQuiz
  question="全方差公式把总变化分成哪两部分？"
  :options="['平均组内方差与条件均值的组间方差', '最大值与最小值', '训练与测试样本数', '概率与似然']"
  :answer="0"
  explanation="E[Var(X|Y)] 描述组内随机性，Var(E[X|Y]) 描述组间均值变化。"
/>

## 4. 协方差矩阵

随机向量均值 $\boldsymbol\mu$，协方差：

$$
\boldsymbol\Sigma
=\mathbb E[
(\mathbf X-\boldsymbol\mu)
(\mathbf X-\boldsymbol\mu)^\top
].
$$

对任意 $\mathbf a$：

$$
\mathbf a^\top\boldsymbol\Sigma\mathbf a
=\operatorname{Var}(\mathbf a^\top\mathbf X)\ge0,
$$

所以协方差矩阵半正定。线性变换 $\mathbf Y=\mathbf A\mathbf X+\mathbf b$：

$$
\mathbb E[\mathbf Y]=\mathbf A\boldsymbol\mu+\mathbf b,
$$

$$
\operatorname{Cov}(\mathbf Y)
=\mathbf A\boldsymbol\Sigma\mathbf A^\top.
$$

## 5. 大数定律与中心极限定理

样本均值

$$
\bar X_n=\frac1n\sum_{i=1}^{n}X_i.
$$

大数定律说明在适当条件下

$$
\bar X_n\xrightarrow{p}\mu.
$$

它回答“是否收敛”。中心极限定理进一步描述误差尺度：

$$
\sqrt n\frac{\bar X_n-\mu}{\sigma}
\xrightarrow{d}\mathcal N(0,1).
$$

因此

$$
\bar X_n\approx
\mathcal N\left(\mu,\frac{\sigma^2}{n}\right).
$$

标准误差按 $1/\sqrt n$ 下降。CLT 是渐近结果，重尾、小样本、强依赖时近似可能差。

<MlQuiz
  question="大数定律与中心极限定理的核心差异是什么？"
  :options="['前者说明样本均值收敛，后者描述缩放误差的渐近分布', '二者完全相同', '前者只用于高斯', '后者不需要样本']"
  :answer="0"
  explanation="LLN 给一致性，CLT 给常见条件下的近似正态波动尺度。"
/>

## 6. 置信区间不是参数的后验概率

若方差已知，均值的近似 $1-\alpha$ 置信区间：

$$
\bar x\pm
z_{1-\alpha/2}\frac{\sigma}{\sqrt n}.
$$

频率学解释是：重复采样并按同一规则构造区间，长期有 $1-\alpha$ 比例覆盖固定真值。对已经算出的某个区间，不能直接说“真值有 95% 概率在里面”；那是 Bayes 可信区间式语言。

## 7. 相关不等于因果

$$
\operatorname{Cov}(X,Y)\ne0
$$

只说明共同变化。混杂、反向因果、选择偏差都可能产生相关。预测模型可以利用稳定关联，但干预决策需要额外因果假设和设计。

<MlQuiz
  question="观察到 X 与 Y 高相关，能否仅据此断言干预 X 会改变 Y？"
  :options="['能', '不能；还需排除混杂、反向因果等', '只要样本大就能', '相关系数为正就能']"
  :answer="1"
  explanation="无限数据能精确估计观测关联，却不能自动识别缺失的因果结构。"
/>

## 8. 动手练习

1. 模拟不同 $n$ 下样本均值分布；
2. 验证标准差按 $1/\sqrt n$ 变化；
3. 重复构造 95% 置信区间并计算覆盖率；
4. 用两个混合群体验证全方差公式；
5. 构造一个共同原因导致的虚假相关例子。

<MlLessonProgress
  storage-key="s03-probability"
  next-href="/courses/machine-learning/s04-information-theory"
  next-label="下一篇：信息论"
/>
