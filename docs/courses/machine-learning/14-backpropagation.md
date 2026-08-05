---
title: 14. 反向传播为什么高效
description: 从计算图、链式法则与向量—雅可比积完整推导反向传播
course: machine-learning
---

<MlLessonHero
  eyebrow="第四模块 · 神经网络"
  title="反向传播为什么高效"
  subtitle="反向传播不是新的微积分规则，而是把链式法则按复用中间结果的顺序执行。"
  :diagnostics="['能手算带分支计算图的梯度', '能推导两层 MLP 的全部参数梯度', '能用有限差分检查自己的反向实现']"
/>

## 1. 从一个标量计算图开始

设

$$
a=xy,\qquad b=a+x,\qquad L=b^2.
$$

前向计算保存 $a,b$。反向从 $\bar L=\partial L/\partial L=1$ 开始，其中横杠记号表示“最终损失对该节点的导数”：

$$
\bar b=\bar L\frac{\partial L}{\partial b}=2b,
$$

$$
\bar a=\bar b\frac{\partial b}{\partial a}=2b.
$$

$x$ 通过两条路径影响 $L$：一条经过 $a$，另一条直接进入 $b$。链式法则要求把两条贡献相加：

$$
\bar x
=\bar a\frac{\partial a}{\partial x}
+\bar b\frac{\partial b}{\partial x}
=2by+2b.
$$

同理

$$
\bar y=\bar a\frac{\partial a}{\partial y}=2bx.
$$

“分支处相加”是反向传播最容易漏掉的规则。它来自多元链式法则，不是实现细节。

::: ml-stop 停下来推一推 1
令 $x=2,y=3$，先算 $a,b,L$，再算 $\bar x,\bar y$。用展开式 $L=(xy+x)^2$ 直接求导核对。
:::

<MlQuiz
  question="某变量沿两条路径影响损失，反向传播到该变量时应该怎样处理？"
  :options="['只保留更大的梯度', '把各路径梯度贡献相加', '把各路径梯度相乘', '随机选一条路径']"
  :answer="1"
  explanation="总导数是所有依赖路径贡献之和；这正是多元链式法则。"
/>

## 2. 向量情形：真正传播的是 VJP

若 $\mathbf y=f(\mathbf x)$，其雅可比矩阵为

$$
\mathbf J_f=\frac{\partial\mathbf y}{\partial\mathbf x^\top},
\qquad
(\mathbf J_f)_{ij}=\frac{\partial y_i}{\partial x_j}.
$$

损失 $L$ 是标量。已知上游梯度 $\bar{\mathbf y}=\nabla_{\mathbf y}L$，需要

$$
\bar{\mathbf x}
=\nabla_{\mathbf x}L
=\mathbf J_f^\top\bar{\mathbf y}.
$$

这叫向量—雅可比积（按另一种行列约定也称 VJP）。关键是：无需显式构造可能巨大的 $\mathbf J_f$，只需让每个算子实现“给定上游梯度，返回对输入的梯度”。

若计算图按

$$
\mathbf x\to\mathbf h_1\to\cdots\to\mathbf h_T\to L
$$

排列，反向传播从右到左连续乘雅可比转置：

$$
\nabla_{\mathbf x}L
=\mathbf J_1^\top\mathbf J_2^\top\cdots\mathbf J_T^\top.
$$

中间 VJP 每次只算一次，因此一次反向的计算量通常与一次前向同阶。

<MlLab
  kind="backprop"
  title="实验：梯度怎样沿链相乘"
  description="调整网络深度和局部导数，观察梯度范数为何会指数级消失或爆炸。"
/>

<MlQuiz
  question="反向传播为什么通常不显式构造完整雅可比矩阵？"
  :options="['雅可比不存在', '只需要雅可比转置与上游梯度的乘积', '矩阵不能求导', '只能处理标量参数']"
  :answer="1"
  explanation="标量损失给出一个上游向量，局部算子可直接计算 VJP，避免存储巨大且多余的完整雅可比。"
/>

## 3. 两层分类 MLP：从损失推到每个参数

沿用上一章的批量前向：

$$
\begin{aligned}
\mathbf Z_1&=\mathbf X\mathbf W_1+\mathbf 1\mathbf b_1^\top,\\
\mathbf H&=\operatorname{ReLU}(\mathbf Z_1),\\
\mathbf Z_2&=\mathbf H\mathbf W_2+\mathbf 1\mathbf b_2^\top,\\
\mathbf P&=\operatorname{softmax}(\mathbf Z_2).
\end{aligned}
$$

批大小为 $B$，标签 one-hot 矩阵为 $\mathbf Y$，平均交叉熵为

$$
L=-\frac1B\sum_{i=1}^{B}\sum_{k=1}^{K}Y_{ik}\log P_{ik}.
$$

第 10 章已经推得 Softmax 与交叉熵的合并梯度：

$$
\mathbf G_2
=\frac{\partial L}{\partial\mathbf Z_2}
=\frac{\mathbf P-\mathbf Y}{B}.
$$

对 $\mathbf Z_2=\mathbf H\mathbf W_2+\mathbf b_2$：

$$
\frac{\partial L}{\partial\mathbf W_2}
=\mathbf H^\top\mathbf G_2,
\qquad
\frac{\partial L}{\partial\mathbf b_2}
=\sum_{i=1}^{B}\mathbf G_{2,i:},
$$

$$
\frac{\partial L}{\partial\mathbf H}
=\mathbf G_2\mathbf W_2^\top.
$$

ReLU 的局部导数为

$$
\operatorname{ReLU}'(z)=
\begin{cases}
1,&z>0,\\
0,&z<0.
\end{cases}
$$

$z=0$ 处不可导，工程上任选一个次梯度，通常取 0。因此

$$
\mathbf G_1
=\frac{\partial L}{\partial\mathbf Z_1}
=\frac{\partial L}{\partial\mathbf H}
\odot\mathbb I(\mathbf Z_1>0).
$$

最后

$$
\frac{\partial L}{\partial\mathbf W_1}
=\mathbf X^\top\mathbf G_1,
\qquad
\frac{\partial L}{\partial\mathbf b_1}
=\sum_{i=1}^{B}\mathbf G_{1,i:}.
$$

所有形状都应与参数一致：

| 梯度 | 形状 |
|---|---|
| $\partial L/\partial\mathbf W_2$ | $m\times K$ |
| $\partial L/\partial\mathbf b_2$ | $K$ |
| $\partial L/\partial\mathbf W_1$ | $d\times m$ |
| $\partial L/\partial\mathbf b_1$ | $m$ |

::: ml-stop 停下来推一推 2
逐个检查上面四个矩阵乘法的形状。若无法仅靠形状排除错误，再指出哪个公式必须靠链式法则判断。
:::

<MlQuiz
  question="交叉熵按批次取平均后，为什么 G2 中要除以 B？"
  :options="['Softmax 要求概率和为 B', '损失定义为 B 个样本损失的平均', '防止矩阵不可逆', '让偏置消失']"
  :answer="1"
  explanation="求导保持线性，平均损失前的 1/B 会原样进入所有下游梯度。"
/>

## 4. 广播的反向：复制的逆操作是求和

前向中 $\mathbf b_2$ 被复制到 $B$ 行：

$$
Z_{2,ik}=(\mathbf H\mathbf W_2)_{ik}+b_{2,k}.
$$

每个 $Z_{2,ik}$ 都以导数 1 依赖同一个 $b_{2,k}$，所以

$$
\frac{\partial L}{\partial b_{2,k}}
=\sum_{i=1}^{B}\frac{\partial L}{\partial Z_{2,ik}}.
$$

一般地，前向在某轴上 broadcast，反向就在该轴上 sum。若遗漏求和，梯度形状就不能回到原变量形状。

<MlQuiz
  question="前向把形状 (m,) 的偏置广播到 (B,m)，反向对偏置梯度应执行什么操作？"
  :options="['沿批维求和', '沿特征维求最大', '转置即可', '再广播一次']"
  :answer="0"
  explanation="同一偏置影响了批中每个样本，总导数要累加所有样本的贡献。"
/>

## 5. 一份可检查的 NumPy 反向

~~~python
import numpy as np

def backward(x, y, z1, h, probs, w2):
    batch = x.shape[0]
    g2 = (probs - y) / batch

    dw2 = h.T @ g2
    db2 = g2.sum(axis=0)

    dh = g2 @ w2.T
    g1 = dh * (z1 > 0)

    dw1 = x.T @ g1
    db1 = g1.sum(axis=0)
    return dw1, db1, dw2, db2
~~~

实现时缓存前向中反向需要的值；但缓存越多，显存越高。检查点技术会丢弃部分中间值，在反向时重新计算，以时间换显存。

## 6. 有限差分：先验证局部梯度

对参数 $\theta_j$，中心差分近似为

$$
g_j^{\text{num}}
=\frac{L(\theta_j+\varepsilon)-L(\theta_j-\varepsilon)}{2\varepsilon}
+O(\varepsilon^2).
$$

用相对误差比较解析梯度：

$$
\operatorname{relerr}
=\frac{|g_j^{\text{ana}}-g_j^{\text{num}}|}
{\max(1,|g_j^{\text{ana}}|,|g_j^{\text{num}}|)}.
$$

$\varepsilon$ 太大会有截断误差，太小会有浮点消去误差；双精度下可从 $10^{-5}$ 左右开始。ReLU 折点附近不可直接用差分判断。

~~~python
def finite_difference(loss_fn, theta, index, eps=1e-5):
    old = theta[index]
    theta[index] = old + eps
    plus = loss_fn()
    theta[index] = old - eps
    minus = loss_fn()
    theta[index] = old
    return (plus - minus) / (2 * eps)
~~~

::: ml-stop 停下来推一推 3
为什么只看训练损失下降不能证明反向实现正确？列出一种“梯度错了但损失仍可能下降”的情况。
:::

<MlQuiz
  question="有限差分检查梯度时，epsilon 取得极小为什么反而可能更差？"
  :options="['会改变模型结构', '两次很接近的损失相减会受浮点消去误差支配', '梯度会自动变成整数', 'Softmax 不允许小数']"
  :answer="1"
  explanation="中心差分同时受截断误差与舍入误差影响，epsilon 不是越小越好。"
/>

<MlQuiz
  question="自动微分框架的反向模式本质上在做什么？"
  :options="['符号积分', '按反向拓扑序复用局部 VJP', '搜索所有参数组合', '用随机数近似每个导数']"
  :answer="1"
  explanation="框架记录计算图和局部算子，标量损失反向时逐节点计算并累加 VJP。"
/>

<MlQuiz
  question="对深度为 T 的标量链，局部导数绝对值都约为 0.5，输入梯度量级怎样变化？"
  :options="['约按 0.5 的 T 次方衰减', '保持不变', '约按 T 线性增长', '必定变成负数']"
  :answer="0"
  explanation="链式法则连续相乘；长期小于 1 的局部增益会造成梯度消失。"
/>

## 7. 动手练习：从零训练 XOR

实现一个结构为 $2\to 8\to 2$ 的 MLP：

1. 手写前向、交叉熵和上述反向；
2. 随机抽 10 个参数做中心差分检查；
3. 用全批量梯度下降拟合 XOR；
4. 画出决策区域；
5. 故意删掉偏置梯度的求和，记录形状错误或训练异常。

提交梯度相对误差表，而不只提交最终准确率。

<MlLessonProgress
  storage-key="14-backpropagation"
  next-href="/courses/machine-learning/15-cnn"
  next-label="下一章：CNN 为什么适合网格数据"
/>
