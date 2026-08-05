---
title: 28. 有限数据为什么可能学到规律
description: 从 Hoeffding 不等式、并合界和假设空间容量理解 PAC 泛化
course: machine-learning
---

<MlLessonHero
  eyebrow="第八模块 · 学习理论"
  title="有限数据为什么可能学到规律"
  subtitle="泛化不是“训练得好所以相信”，而是数据独立、假设容量受控时，经验误差会一致接近真实误差。"
  :diagnostics="['能从固定假设界推到有限假设空间一致界', '能读懂 PAC 样本复杂度', '能指出理论保证所依赖的分布与独立性条件']"
/>

## 1. 两种风险

数据 $(\mathbf x,y)\sim\mathcal P$，损失 $\ell(h(\mathbf x),y)\in[0,1]$。真实风险：

$$
R(h)
=\mathbb E_{(\mathbf x,y)\sim\mathcal P}
[\ell(h(\mathbf x),y)].
$$

训练集 $S$ 有 $n$ 个独立同分布样本，经验风险：

$$
\hat R_S(h)
=\frac1n\sum_{i=1}^{n}
\ell(h(\mathbf x_i),y_i).
$$

训练只能看到 $\hat R$，部署关心 $R$。学习理论研究二者差距以及选择过 $h$ 后界是否仍成立。

<MlLab
  kind="pac"
  title="实验：样本量、假设数和置信度怎样改变泛化界"
  description="调节 n、假设空间大小与 delta，观察一致偏差上界及其保守性。"
/>

## 2. 固定一个假设：Hoeffding 不等式

先假设 $h$ 在看数据前已经固定。令

$$
Z_i=\ell(h(\mathbf x_i),y_i)\in[0,1].
$$

$Z_i$ 独立同分布，$\mathbb E[Z_i]=R(h)$，
$\hat R(h)=n^{-1}\sum_iZ_i$。Hoeffding 给出

$$
\Pr\left(
|\hat R(h)-R(h)|>\epsilon
\right)
\le2e^{-2n\epsilon^2}.
$$

令右侧为 $\delta$，解出

$$
\boxed{
|R(h)-\hat R(h)|
\le
\sqrt{\frac{\log(2/\delta)}{2n}}
}
$$

以至少 $1-\delta$ 的概率成立。误差随 $1/\sqrt n$ 缩小：想把界宽减半，样本量约需变为四倍。

<MlQuiz
  question="固定假设的 Hoeffding 界中，样本量增为 4 倍时，偏差上界约变为多少？"
  :options="['原来的 1/4', '原来的 1/2', '原来的 2 倍', '不变']"
  :answer="1"
  explanation="主要尺度是 1/√n，样本量乘 4 后上界乘 1/2。"
/>

## 3. 为什么不能直接把固定假设界用于训练结果

训练算法根据同一数据从许多假设中挑了经验风险小的一个。若只对最终 $h$ 套固定假设界，相当于忽略了“反复尝试后挑最好”带来的选择偏差。

假设空间 $\mathcal H$ 有限，$|\mathcal H|=M$。对每个 $h$ 的失败事件 $A_h$，并合界：

$$
\Pr\left(\bigcup_{h\in\mathcal H}A_h\right)
\le\sum_{h\in\mathcal H}\Pr(A_h).
$$

因此

$$
\Pr\left(
\exists h\in\mathcal H:
|R(h)-\hat R(h)|>\epsilon
\right)
\le2M e^{-2n\epsilon^2}.
$$

令其等于 $\delta$：

$$
\boxed{
\forall h\in\mathcal H,\quad
|R(h)-\hat R(h)|
\le
\sqrt{
\frac{\log(2M/\delta)}{2n}
}
}
$$

以至少 $1-\delta$ 同时成立。因为对全部 $h$ 同时成立，数据依赖地选出的 $\hat h$ 也包含在内。

::: ml-stop 停下来推一推 1
假设数 $M$ 从 $10^3$ 增到 $10^6$，复杂度项并非增大 1000 倍。指出它通过什么函数进入界。
:::

<MlQuiz
  question="有限假设空间的泛化界中，模型数量 M 通常怎样进入？"
  :options="['线性 M', '平方 M²', '对数 log M', '完全不进入']"
  :answer="2"
  explanation="并合界产生因子 M，解指数尾界后变为 log M。"
/>

## 4. ERM 的超额风险

令经验风险最小化解

$$
\hat h=\arg\min_{h\in\mathcal H}\hat R(h),
$$

假设空间内真实最优

$$
h^\star=\arg\min_{h\in\mathcal H}R(h).
$$

若一致偏差界为 $\epsilon$，即所有 $h$ 都满足
$|R(h)-\hat R(h)|\le\epsilon$，则

$$
\begin{aligned}
R(\hat h)
&\le\hat R(\hat h)+\epsilon\\
&\le\hat R(h^\star)+\epsilon\\
&\le R(h^\star)+2\epsilon.
\end{aligned}
$$

所以

$$
R(\hat h)-R(h^\star)\le2\epsilon.
$$

第二行使用 $\hat h$ 的 ERM 定义。这个三步链条说明“一致收敛”怎样转化为学习保证。

<MlQuiz
  question="ERM 超额风险推导中，哪一步使用了 hat h 最小化经验风险？"
  :options="['R(hat h)≤R(h*)', 'hat R(hat h)≤hat R(h*)', 'R(h*)≤hat R(h*)', 'epsilon≤0']"
  :answer="1"
  explanation="ERM 只保证训练样本上的经验风险不大于其他候选，不直接比较真实风险。"
/>

## 5. PAC 样本复杂度

若希望一致偏差不超过 $\epsilon$，失败概率不超过 $\delta$：

$$
2M e^{-2n\epsilon^2}\le\delta.
$$

解 $n$：

$$
\boxed{
n\ge
\frac{1}{2\epsilon^2}
\left[
\log(2M)+\log\frac1\delta
\right]
}.
$$

PAC 指 Probably Approximately Correct：

- Probably：至少 $1-\delta$；
- Approximately：误差容忍 $\epsilon$；
- Correct：相对目标风险或假设空间最优的保证。

界常很保守，它的价值是揭示依赖关系，而非精确预测需要多少样本。

::: ml-stop 停下来推一推 2
把允许误差 $\epsilon$ 减半，样本复杂度的主项变为几倍？把失败概率 $\delta$ 减半又只增加什么量？
:::

<MlQuiz
  question="PAC 界中把 epsilon 减半，样本复杂度主项约需变为多少倍？"
  :options="['2 倍', '4 倍', '8 倍', '不变']"
  :answer="1"
  explanation="样本复杂度按 1/epsilon² 增长。"
/>

## 6. 无限假设空间与 VC 维

线性分类器的参数连续，$|\mathcal H|=\infty$，不能直接用 $\log M$。关键不是参数取值无限，而是模型能在有限样本上实现多少种标签模式。

若某组 $m$ 个点的全部 $2^m$ 种二分类标记都能被 $\mathcal H$ 实现，就说它们被打散。VC 维是可被打散的最大点数。

二维平面中的线性分类器 VC 维为 3：

- 任意三个一般位置点的 8 种标记都可由直线实现；
- 某些四点标记（如 XOR）不可线性分开。

增长函数 $\Pi_{\mathcal H}(n)$ 统计 $n$ 个点上最多能实现的标记数。VC 维为 $d_{\rm VC}$ 时，Sauer 引理给出

$$
\Pi_{\mathcal H}(n)
\le
\sum_{j=0}^{d_{\rm VC}}\binom nj
\le
\left(\frac{en}{d_{\rm VC}}\right)^{d_{\rm VC}}
$$

（当 $n\ge d_{\rm VC}$）。它把无限参数空间替换为有限样本上的有效复杂度。

<MlQuiz
  question="VC 维衡量的核心是什么？"
  :options="['参数的数值大小', '假设空间在有限点集上实现二分类标记模式的能力', '训练速度', '输入特征单位']"
  :answer="1"
  explanation="它关注可打散的点数，是组合表达容量而非简单参数计数。"
/>

## 7. 结构风险最小化

更复杂的假设空间通常经验误差更低，但泛化复杂度项更高。可把模型族嵌套：

$$
\mathcal H_1\subset\mathcal H_2\subset\cdots
$$

并平衡

$$
\text{经验风险}+\text{复杂度惩罚}.
$$

这与正则化、验证选模和最小描述长度有共同精神：不能只奖励拟合，也要计入选择自由度。

但现代过参数模型说明“参数越多，界必然越差”过于粗糙。范数、间隔、优化隐式偏置和数据结构可能比参数数量更有解释力。

## 8. 理论保证的边界条件

经典界常依赖：

- 训练样本独立同分布；
- 训练与部署来自同一分布；
- 损失有界或尾部受控；
- 假设空间/算法复杂度被正确计入；
- 测试集没有参与反复选择。

时间序列、群组数据和分布漂移会破坏这些条件。此时要用时间切分、群组切分、漂移监控或专门的依赖数据理论。

::: ml-stop 停下来推一推 3
同一患者有 20 张片子，随机按图片切分会违反哪种独立性直觉？怎样按群组切分更接近部署问题？
:::

<MlQuiz
  question="在测试集上反复挑模型后再报告同一测试分数，主要破坏了什么？"
  :options="['测试集作为独立最终评价的角色', '损失函数可导性', '矩阵乘法结合律', '类别标签定义']"
  :answer="0"
  explanation="反复选择使模型对测试集信息产生适应，测试分数也出现选择偏差。"
/>

## 9. 界为何仍值得学

泛化界可能比实际误差松很多，但它强迫我们问对问题：

- 概率是对什么随机性？
- 保证对固定模型还是选择后的全部模型？
- 置信度与误差怎样交换？
- 数据依赖条件是否成立？
- 模型容量用什么量衡量？

这套语言能识别“训练分数很好”与“对未来有证据”之间缺失的逻辑环节。

<MlQuiz
  question="一个很松的泛化界仍可能有价值，因为它能做什么？"
  :options="['精确等于测试误差', '揭示样本量、置信度、容量与假设条件的依赖', '证明所有模型公平', '替代真实验证']"
  :answer="1"
  explanation="理论界提供结构性依赖和审计条件，不应被误当作精确性能预测。"
/>

## 10. 动手练习：经验覆盖率

固定一个 Bernoulli 风险为 $p$ 的分类器：

1. 重复抽取 10 000 份大小为 $n$ 的样本；
2. 计算 $|\hat R-R|$；
3. 检查 Hoeffding 区间的经验覆盖率；
4. 改变 $n,\delta$；
5. 再从 $M$ 个随机假设中挑经验风险最小者；
6. 对比错误的固定假设界与含 $\log M$ 的一致界。

<MlLessonProgress
  storage-key="28-learning-theory"
  next-href="/courses/machine-learning/29-recommender-systems"
  next-label="下一章：推荐系统怎样从缺失矩阵学习"
/>
