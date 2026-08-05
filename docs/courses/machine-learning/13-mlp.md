---
title: 13. 复合线性层怎样表达复杂函数
description: 从仿射变换、激活函数和参数初始化推导多层感知机
course: machine-learning
---

<MlLessonHero
  eyebrow="第四模块 · 神经网络"
  title="复合线性层怎样表达复杂函数"
  subtitle="神经网络的能力不来自“层数”本身，而来自线性变换之间那一点非线性。"
  :diagnostics="['能解释为什么没有激活函数的深层网络仍是线性模型', '能为任务选择输出层与损失', '能从方差传播推导 Xavier 与 He 初始化']"
/>

## 1. 先拆掉“神经元”的神秘感

一个神经元只完成两步：

$$
z=\mathbf w^\top\mathbf x+b,\qquad h=\phi(z).
$$

第一步把输入投影到一个方向，第二步用激活函数 $\phi$ 改变形状。把 $m$ 个神经元并排写成矩阵，就是一层：

$$
\mathbf h=\phi(\mathbf W^\top\mathbf x+\mathbf b).
$$

若一批数据写成 $\mathbf X\in\mathbb R^{B\times d}$，隐藏宽度为 $m$，则

$$
\mathbf Z=\mathbf X\mathbf W+\mathbf 1\mathbf b^\top,\qquad
\mathbf H=\phi(\mathbf Z),
$$

其中 $\mathbf W\in\mathbb R^{d\times m}$，$\mathbf b\in\mathbb R^m$。激活函数逐元素作用，不改变形状。

::: ml-stop 停下来推一推 1
取 $B=32,d=10,m=64$。写出 $\mathbf X,\mathbf W,\mathbf b,\mathbf Z,\mathbf H$ 的形状，并说明偏置在批维度发生了什么。
:::

## 2. 没有非线性，多深都只是一层

先看两个不含激活函数的仿射层：

$$
\mathbf h=\mathbf W_1^\top\mathbf x+\mathbf b_1,
$$

$$
\mathbf y=\mathbf W_2^\top\mathbf h+\mathbf b_2.
$$

代入第一式：

$$
\begin{aligned}
\mathbf y
&=\mathbf W_2^\top(\mathbf W_1^\top\mathbf x+\mathbf b_1)+\mathbf b_2\\
&=(\mathbf W_1\mathbf W_2)^\top\mathbf x+
(\mathbf W_2^\top\mathbf b_1+\mathbf b_2)\\
&=\widetilde{\mathbf W}^\top\mathbf x+\widetilde{\mathbf b}.
\end{aligned}
$$

复合结果仍是仿射函数。归纳地，任意多个仿射层都能折叠成一个仿射层。因此，网络可以弯折决策边界的关键是 $\phi$。

<MlQuiz
  question="把 20 个不含激活函数的全连接层堆起来，模型表达能力会发生什么变化？"
  :options="['一定得到 20 次多项式', '仍等价于一个仿射变换', '一定可以拟合任意函数', '只要参数多就一定非线性']"
  :answer="1"
  explanation="矩阵乘法的复合仍是矩阵乘法，偏置项的复合仍是偏置项。深度不会凭空产生非线性。"
/>

## 3. 激活函数在做什么

| 激活 | 公式 | 优点 | 常见风险 |
|---|---|---|---|
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$ | 输出可解释为概率 | 两端导数接近 0，且输出非零中心 |
| Tanh | $\tanh z$ | 零中心、平滑 | 大绝对值处梯度消失 |
| ReLU | $\max(0,z)$ | 正区间梯度稳定、计算简单 | 负区间梯度为 0 |
| Leaky ReLU | $\max(\alpha z,z)$ | 负区间保留小梯度 | 多一个斜率选择 |
| GELU | $z\Phi(z)$ | 平滑门控，Transformer 常用 | 计算稍复杂 |

ReLU 网络为什么能表达弯曲边界？一维函数 $\operatorname{ReLU}(x-t)$ 在 $t$ 处制造一个折点。线性组合

$$
f(x)=a_0+a_1x+\sum_{j=1}^{m}c_j\operatorname{ReLU}(x-t_j)
$$

能够构造任意分段线性曲线。高维时，每个 $\mathbf w_j^\top\mathbf x+b_j=0$ 都是一张折叠空间的超平面。

<MlLab
  kind="network"
  title="实验：宽度怎样增加折点"
  description="调整隐藏单元数与输入位置，观察 ReLU 基函数叠加后函数形状和局部斜率的变化。"
/>

## 4. 两层 MLP 的完整表达式

设输入维度 $d$、隐藏宽度 $m$、输出维度 $K$：

$$
\begin{aligned}
\mathbf Z_1&=\mathbf X\mathbf W_1+\mathbf 1\mathbf b_1^\top,
&\mathbf W_1&\in\mathbb R^{d\times m},\\
\mathbf H&=\phi(\mathbf Z_1),
&\mathbf H&\in\mathbb R^{B\times m},\\
\mathbf Z_2&=\mathbf H\mathbf W_2+\mathbf 1\mathbf b_2^\top,
&\mathbf W_2&\in\mathbb R^{m\times K}.
\end{aligned}
$$

$\mathbf Z_2$ 通常称为 logits。输出变换和损失必须与任务匹配：

| 任务 | 输出层 | 单样本损失 |
|---|---|---|
| 实数回归 | 恒等映射 | $\frac12\|\mathbf z-\mathbf y\|_2^2$ |
| 二分类 | Sigmoid | $-[y\log p+(1-y)\log(1-p)]$ |
| 单标签多分类 | Softmax | $-\log p_y$ |
| 多标签分类 | 每一维独立 Sigmoid | 各维二元交叉熵之和 |

多分类不能把每个类别都做独立 Sigmoid，因为类别互斥且概率和应为 1；多标签问题则允许多个标签同时为真。

::: ml-stop 停下来推一推 2
一个模型判断图片里是否同时含“猫、草地、汽车”。它应使用 Softmax 还是三个 Sigmoid？写出理由。
:::

<MlQuiz
  question="三分类单标签任务的最后一层最合适的组合是？"
  :options="['三个 ReLU 与平方损失', '三个独立 Sigmoid 且强制取最大', '三个 logits、Softmax 与交叉熵', '一个 Sigmoid']"
  :answer="2"
  explanation="Softmax 将三个 logits 归一化成和为 1 的类别分布，交叉熵对应类别似然。"
/>

## 5. “万能逼近”不等于“训练一定成功”

万能逼近定理粗略地说：在适当激活函数下，足够宽的单隐层网络可以在紧致区域上一致逼近任意连续函数。它回答的是“是否存在一组参数”，没有保证：

- 需要多少隐藏单元；
- 有限数据能否辨认出那组参数；
- 优化算法能否找到它；
- 区域之外能否泛化；
- 计算和样本成本是否合理。

所以表达能力、可优化性和泛化能力是三个不同问题。扩大网络能降低表达偏差，也可能改变优化轨迹和过拟合风险。

<MlQuiz
  question="万能逼近定理最直接保证了什么？"
  :options="['梯度下降一定找到全局最优', '足够宽时存在能逼近目标函数的参数', '网络在训练区间外一定正确', '小数据上一定不会过拟合']"
  :answer="1"
  explanation="它是存在性结论，不是训练算法、样本效率或外推能力的保证。"
/>

## 6. 初始化：让信号方差别逐层爆炸或消失

设

$$
z_j=\sum_{i=1}^{n_{\text{in}}}w_{ij}x_i,
$$

并近似假设 $w_{ij}$ 与 $x_i$ 独立、均值为 0。则

$$
\operatorname{Var}(z_j)
=\sum_i\operatorname{Var}(w_{ij}x_i)
=n_{\text{in}}\operatorname{Var}(w)\operatorname{Var}(x).
$$

若希望线性层前后方差相近，需要

$$
\operatorname{Var}(w)\approx\frac{1}{n_{\text{in}}}.
$$

这给出 Xavier 初始化的基本尺度。它还兼顾反向梯度，常取

$$
\operatorname{Var}(w)=\frac{2}{n_{\text{in}}+n_{\text{out}}}.
$$

对零均值对称输入，ReLU 大约把一半值置零，使二阶矩约减半。为了补偿，He 初始化取

$$
\operatorname{Var}(w)=\frac{2}{n_{\text{in}}}.
$$

这不是魔法常数，而是“让二阶矩跨层近似守恒”的结果。真实网络不完全满足独立同分布假设，但推导提供了可靠起点。

::: ml-stop 停下来推一推 3
若一个 ReLU 层有 $n_{\text{in}}=200$，He 正态初始化的标准差是多少？注意公式给的是方差。
:::

<MlQuiz
  question="ReLU 网络使用 He 初始化时，权重方差通常为什么是 2/n_in？"
  :options="['让权重之和等于 1', '补偿 ReLU 约丢掉一半二阶矩', '让所有神经元输出相同', '保证损失凸']"
  :answer="1"
  explanation="在线性组合方差守恒的 1/n_in 基础上，乘 2 近似补偿 ReLU 截断负半轴造成的二阶矩下降。"
/>

<MlQuiz
  question="为什么通常不把同一层所有权重都初始化为相同值？"
  :options="['会让数值类型出错', '同层单元保持对称，收到相同梯度，无法分工', '会自动变成卷积', '损失函数无法计算']"
  :answer="1"
  explanation="相同初始化使单元的前向值和更新始终相同，相当于只有一个重复的特征。随机初始化用于打破对称性。"
/>

## 7. 最小前向实现

~~~python
import numpy as np

def relu(z):
    return np.maximum(z, 0.0)

def stable_softmax(z):
    shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def forward(x, w1, b1, w2, b2):
    z1 = x @ w1 + b1
    h = relu(z1)
    logits = h @ w2 + b2
    return z1, h, logits, stable_softmax(logits)
~~~

这里的偏置依赖 NumPy broadcasting：形状 $(m,)$ 会沿批维复制。下一章会反向穿过每一步，并解释广播反向时为什么必须求和。

<MlQuiz
  question="在批量前向中，X 的形状为 (B,d)，W1 为 (d,m)，W2 为 (m,K)。logits 的形状是？"
  :options="['(d,K)', '(B,m)', '(B,K)', '(m,K)']"
  :answer="2"
  explanation="矩阵链为 (B,d)(d,m)(m,K)，最终每个样本得到 K 个 logits。"
/>

<MlQuiz
  question="隐藏层宽度增加，最直接改变的是什么？"
  :options="['数据集标签', '可组合的特征与折叠超平面数量', '输出概率必须变大', '训练损失必然为零']"
  :answer="1"
  explanation="更多隐藏单元提供更多可学习基函数；它不自动保证更好的优化或泛化。"
/>

## 8. 动手练习：看见方差传播

实现一个 30 层、每层宽度 256 的随机 ReLU 网络，对同一批标准正态输入分别使用：

1. 标准差 1 的普通正态初始化；
2. 标准差 $\sqrt{1/n_{\text{in}}}$；
3. 标准差 $\sqrt{2/n_{\text{in}}}$。

记录每层激活的均值、标准差和零值比例。解释哪种方案爆炸、哪种衰减、哪种相对稳定。再把 ReLU 改为 Tanh，检查结论为何变化。

<MlLessonProgress
  storage-key="13-mlp"
  next-href="/courses/machine-learning/14-backpropagation"
  next-label="下一章：反向传播为什么高效"
/>
