---
title: 17. Transformer 为什么不需要递归
description: 从内容寻址、缩放点积和多头机制推导自注意力
course: machine-learning
---

<MlLessonHero
  eyebrow="第四模块 · 神经网络"
  title="Transformer 为什么不需要递归"
  subtitle="注意力让每个位置按内容直接读取其他位置，把长依赖从 T 步缩短为一次聚合。"
  :diagnostics="['能从查询—键—值解释注意力', '能推导 1/√d_k 缩放', '能正确构造因果掩码与多头形状']"
/>

## 1. 从“按地址读取”到“按内容读取”

数据库按精确键读取；注意力允许模糊的内容匹配。对第 $i$ 个查询 $\mathbf q_i$ 和第 $j$ 个键 $\mathbf k_j$，先计算相似度，再归一化为权重：

$$
a_{ij}
=\frac{\exp(s_{ij})}{\sum_{\ell=1}^{T}\exp(s_{i\ell})}.
$$

输出是值向量的加权平均：

$$
\mathbf o_i=\sum_{j=1}^{T}a_{ij}\mathbf v_j.
$$

因为 $a_{ij}\ge0$ 且 $\sum_j a_{ij}=1$，每个输出是值向量的凸组合。查询决定“我要找什么”，键描述“我有什么”，值携带“被读出的内容”。

<MlQuiz
  question="注意力中的 value 最直接扮演什么角色？"
  :options="['用于计算被聚合的内容', '只决定序列长度', '固定为类别标签', '用于初始化优化器']"
  :answer="0"
  explanation="query 与 key 决定读取权重，value 是按这些权重被汇总的信息。"
/>

## 2. 缩放点积注意力

输入 $\mathbf X\in\mathbb R^{T\times d_{\rm model}}$，线性投影得到

$$
\mathbf Q=\mathbf X\mathbf W_Q,\quad
\mathbf K=\mathbf X\mathbf W_K,\quad
\mathbf V=\mathbf X\mathbf W_V,
$$

其中 $\mathbf Q,\mathbf K\in\mathbb R^{T\times d_k}$，
$\mathbf V\in\mathbb R^{T\times d_v}$。矩阵形式为

$$
\operatorname{Attention}(\mathbf Q,\mathbf K,\mathbf V)
=\operatorname{softmax}\left(
\frac{\mathbf Q\mathbf K^\top}{\sqrt{d_k}}
\right)\mathbf V.
$$

Softmax 对每一行执行：第 $i$ 行表示查询 $i$ 对所有键的分布。

### 为什么除以 $\sqrt{d_k}$

若 $q_r,k_r$ 独立、均值 0、方差 1，则点积

$$
s=\mathbf q^\top\mathbf k=\sum_{r=1}^{d_k}q_rk_r
$$

满足

$$
\mathbb E[s]=0,\qquad
\operatorname{Var}(s)
=\sum_{r=1}^{d_k}\operatorname{Var}(q_rk_r)
\approx d_k.
$$

所以 $s$ 的标准差约为 $\sqrt{d_k}$。除以 $\sqrt{d_k}$ 后，logit 尺度保持为 $O(1)$，避免维度增大使 Softmax 过早饱和、梯度变小。

::: ml-stop 停下来推一推 1
若每个点积项方差为 4，且 $d_k=64$，点积的标准差约是多少？应除以什么量才能把标准差缩回约 1？
:::

<MlLab
  kind="attention"
  title="实验：温度怎样改变注意力"
  description="调节相似度尺度与查询位置，观察注意力分布从平均到尖锐的连续变化。"
/>

<MlQuiz
  question="缩放点积注意力除以 √d_k 的主要目的是什么？"
  :options="['让序列变短', '控制点积方差，避免 Softmax 随维度增大而饱和', '让所有注意力权重相等', '删除位置信息']"
  :answer="1"
  explanation="独立分量点积的方差约随 d_k 线性增长，标准差随 √d_k 增长。"
/>

## 3. 掩码必须加在 Softmax 之前

自回归预测位置 $i$ 时不能看到未来 $j>i$。构造掩码

$$
M_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

然后

$$
\mathbf A
=\operatorname{softmax}\left(
\frac{\mathbf Q\mathbf K^\top}{\sqrt{d_k}}+\mathbf M
\right).
$$

因为 $e^{-\infty}=0$，未来位置概率严格为 0。若先 Softmax 再把未来权重设为 0，剩余权重之和小于 1；除非重新归一化，否则已不是同一分布。

Padding mask 则屏蔽批中补齐位置。两种 mask 目的不同，但都应在 Softmax 前作用于 logits。

<MlQuiz
  question="为什么因果 mask 通常在 Softmax 前把未来 logit 设为负无穷？"
  :options="['使未来位置权重归一化后为 0', '让未来 token 变成正数', '减少模型参数', '替代位置编码']"
  :answer="0"
  explanation="指数化后负无穷对应 0，且合法位置仍在同一次 Softmax 中归一化。"
/>

## 4. 多头：在不同子空间建立多组关系

第 $h$ 个头计算

$$
\operatorname{head}_h
=\operatorname{Attention}
(\mathbf X\mathbf W_Q^{(h)},
 \mathbf X\mathbf W_K^{(h)},
 \mathbf X\mathbf W_V^{(h)}).
$$

拼接并投影：

$$
\operatorname{MHA}(\mathbf X)
=\operatorname{Concat}(
\operatorname{head}_1,\ldots,\operatorname{head}_H)\mathbf W_O.
$$

通常 $d_k=d_v=d_{\rm model}/H$。多头并非保证每个头自动对应某种语言学关系；它只是提供多个独立投影和注意力分布，允许模型并行表达不同关系。

::: ml-stop 停下来推一推 2
$B=8,T=128,d_{\rm model}=512,H=8$。写出每头 Q、K、注意力矩阵和拼接后输出的形状。
:::

<MlQuiz
  question="d_model=768、12 个等宽注意力头时，每个头常用的 d_k 是多少？"
  :options="['12', '64', '256', '768']"
  :answer="1"
  explanation="常见等分设置为 768/12=64。"
/>

## 5. 没有递归，位置从哪里来

纯自注意力对输入排列是等变的：若同时置换所有 token，输出也按同样方式置换。它本身不知道先后顺序，必须加入位置信息。

经典正弦位置编码为

$$
\operatorname{PE}_{p,2i}
=\sin\left(p/10000^{2i/d}\right),
$$

$$
\operatorname{PE}_{p,2i+1}
=\cos\left(p/10000^{2i/d}\right).
$$

不同维度提供不同频率。也可以学习绝对位置向量，或把相对距离/旋转结构直接放入注意力分数。它们有不同的长度外推性质，不能混称为“只是给 token 加编号”。

<MlQuiz
  question="如果不给纯自注意力任何位置信息，它缺少哪种能力？"
  :options="['区分词表大小', '区分相同 token 集合的不同排列', '计算矩阵乘法', '输出连续数值']"
  :answer="1"
  explanation="自注意力本身对排列等变，需要位置机制表达顺序和相对距离。"
/>

## 6. 一个 Transformer 块

现代块通常由注意力和逐位置前馈网络组成，并配合残差连接与归一化。Pre-Norm 形式可写为

$$
\mathbf H'
=\mathbf H+\operatorname{MHA}(\operatorname{LN}(\mathbf H)),
$$

$$
\mathbf H''
=\mathbf H'+\operatorname{FFN}(\operatorname{LN}(\mathbf H')).
$$

前馈网络对每个位置独立共享：

$$
\operatorname{FFN}(\mathbf x)
=\mathbf W_2\phi(\mathbf W_1\mathbf x+\mathbf b_1)+\mathbf b_2.
$$

注意力负责位置间通信，FFN 负责每个位置内部的非线性变换，残差提供近似恒等的梯度路径。

## 7. 复杂度的交换

全注意力分数矩阵大小为 $T\times T$，时间和显存的主项约为

$$
O(T^2d).
$$

RNN 每步计算约为 $O(d^2)$，总计 $O(Td^2)$，但时间步必须串行。Transformer 能并行处理所有位置，长序列时却受二次方注意力矩阵限制。

两者没有无条件胜负：短中序列与并行硬件偏向注意力，极长流式输入可能需要局部、稀疏、线性注意力或状态空间方法。

::: ml-stop 停下来推一推 3
序列长度从 1 024 翻倍到 2 048，保持维度不变，全注意力分数矩阵的元素数变为几倍？这对显存意味着什么？
:::

<MlQuiz
  question="标准全注意力在序列长度 T 上的主要复杂度是多少？"
  :options="['O(log T)', 'O(T)', 'O(T²)', '与 T 无关']"
  :answer="2"
  explanation="每个查询与每个键都计算分数，形成 T×T 矩阵。"
/>

## 8. 稳定的掩码自注意力实现

~~~python
import numpy as np

def causal_attention(q, k, v):
    dk = q.shape[-1]
    scores = q @ k.T / np.sqrt(dk)

    future = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)

    row_max = np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores - row_max)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v, weights
~~~

减去每行最大值不会改变 Softmax，却避免指数溢出。真实批量多头实现还要处理 batch 与 head 维，并确保 padding mask 的广播方向正确。

<MlQuiz
  question="Softmax 前减去每一行最大 logit，为什么不改变结果？"
  :options="['所有指数项乘了同一个因子，归一化时抵消', '最大值一定为 0', '这会改变结果，只是误差小', '因为矩阵必须对称']"
  :answer="0"
  explanation="exp(z_i-c)/Σexp(z_j-c) 中公共因子 exp(-c) 在分子分母抵消。"
/>

## 9. 动手练习：检查信息泄漏

1. 实现上面的单头因果注意力；
2. 改变未来 token，验证较早位置输出完全不变；
3. 故意把 mask 放到 Softmax 后，检查权重和；
4. 画出不同缩放温度下的注意力热图；
5. 用有限差分检查 $\mathbf V$ 的梯度；
6. 解释“无递归”为什么不等于“无顺序”。

<MlLessonProgress
  storage-key="17-transformer"
  next-href="/courses/machine-learning/18-knn"
  next-label="下一章：不写参数方程也能学习吗"
/>
