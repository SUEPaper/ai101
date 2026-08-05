---
title: 16. RNN 怎样对序列共享状态
description: 从条件分解、状态递推和时间反向传播推导 RNN、LSTM 与 GRU
course: machine-learning
---

<MlLessonHero
  eyebrow="第四模块 · 神经网络"
  title="RNN 怎样对序列共享状态"
  subtitle="RNN 把过去压进状态，并在每个时间步重复使用同一组参数。"
  :diagnostics="['能写出序列概率的条件分解', '能推导时间反向传播与参数梯度累加', '能解释 LSTM 的加法记忆通路']"
/>

## 1. 序列概率来自链式法则

对序列 $\mathbf x_{1:T}$，概率链式法则恒成立：

$$
p(\mathbf x_{1:T})
=\prod_{t=1}^{T}p(\mathbf x_t\mid\mathbf x_{1:t-1}).
$$

难点是历史长度随 $t$ 增长。RNN 用固定维度状态 $\mathbf h_t$ 汇总历史：

$$
\mathbf h_t
=\phi(\mathbf W_{xh}\mathbf x_t
+\mathbf W_{hh}\mathbf h_{t-1}
+\mathbf b_h),
$$

$$
\mathbf o_t=\mathbf W_{ho}\mathbf h_t+\mathbf b_o.
$$

所有时间步共享 $\mathbf W_{xh},\mathbf W_{hh}$。这既降低参数量，也编码了“同一种状态更新规律可用于不同位置”的时间平移先验。

<MlQuiz
  question="RNN 在不同时间步共享参数，最直接带来什么结果？"
  :options="['序列必须固定长度', '参数量不随序列长度增长，并可处理不同长度', '状态不再依赖过去', '梯度不需要求和']"
  :answer="1"
  explanation="展开图虽然随 T 变长，但重复使用的是同一组转移参数。"
/>

## 2. 时间反向传播：共享参数的贡献要累加

记

$$
\mathbf a_t
=\mathbf W_{xh}\mathbf x_t+\mathbf W_{hh}\mathbf h_{t-1}+\mathbf b_h,
\qquad
\mathbf h_t=\phi(\mathbf a_t).
$$

设损失 $L=\sum_t\ell_t(\mathbf h_t)$。状态 $\mathbf h_t$ 同时影响当前损失和未来状态，因此

$$
\frac{\partial L}{\partial\mathbf h_t}
=\frac{\partial\ell_t}{\partial\mathbf h_t}
+\mathbf W_{hh}^\top
\frac{\partial L}{\partial\mathbf a_{t+1}}.
$$

再乘激活导数：

$$
\boldsymbol\delta_t
=\frac{\partial L}{\partial\mathbf a_t}
=\frac{\partial L}{\partial\mathbf h_t}
\odot\phi'(\mathbf a_t).
$$

因为 $\mathbf W_{hh}$ 在所有时刻出现，总梯度是

$$
\frac{\partial L}{\partial\mathbf W_{hh}}
=\sum_{t=1}^{T}\boldsymbol\delta_t\mathbf h_{t-1}^\top.
$$

同理

$$
\frac{\partial L}{\partial\mathbf W_{xh}}
=\sum_t\boldsymbol\delta_t\mathbf x_t^\top,
\qquad
\frac{\partial L}{\partial\mathbf b_h}
=\sum_t\boldsymbol\delta_t.
$$

这就是 BPTT（通过时间的反向传播）：先把循环展开成有向无环图，再从 $T$ 到 1 应用普通反向传播。

::: ml-stop 停下来推一推 1
一个参数在 10 个时间步被使用。为什么不能只用最后一步产生的参数梯度？把它与卷积核的空间共享联系起来。
:::

<MlLab
  kind="recurrence"
  title="实验：递推增益如何控制记忆"
  description="调节递推系数与序列长度，观察早期输入对最终状态和梯度的影响。"
/>

<MlQuiz
  question="BPTT 对共享的循环权重求梯度时，需要怎样组合各时刻贡献？"
  :options="['只取第一步', '只取最后一步', '对所有使用该权重的时刻求和', '取各时刻最大值']"
  :answer="2"
  explanation="同一参数通过多个计算路径影响总损失，总导数是各路径贡献之和。"
/>

## 3. 梯度为什么消失或爆炸

暂时忽略每步直接损失。早期状态对晚期状态的影响包含雅可比连乘：

$$
\frac{\partial\mathbf h_T}{\partial\mathbf h_t}
=\prod_{k=t+1}^{T}
\left[
\operatorname{diag}\bigl(\phi'(\mathbf a_k)\bigr)\mathbf W_{hh}
\right].
$$

取范数上界：

$$
\left\|
\frac{\partial\mathbf h_T}{\partial\mathbf h_t}
\right\|
\le
\prod_{k=t+1}^{T}
\|\operatorname{diag}(\phi'(\mathbf a_k))\|
\|\mathbf W_{hh}\|.
$$

若平均乘子小于 1，影响指数衰减；大于 1，可能指数爆炸。对 Tanh，$|\phi'|\le1$ 且饱和时远小于 1，因此长程梯度尤其困难。

梯度裁剪

$$
\mathbf g\leftarrow
\mathbf g\min\left(1,\frac{\tau}{\|\mathbf g\|_2}\right)
$$

能限制爆炸，但不能恢复已经消失的梯度。截断 BPTT 只反传最近 $K$ 步，降低内存和计算，也会切断更长依赖。

::: ml-stop 停下来推一推 2
若标量递推的局部导数恒为 0.8，跨 50 步的梯度倍率约为多少数量级？这说明什么？
:::

<MlQuiz
  question="梯度裁剪主要针对哪种问题？"
  :options="['梯度爆炸', '标签缺失', '类别不平衡', '输出维度过小']"
  :answer="0"
  explanation="它限制梯度范数，防止一次更新过大；不能从根本上解决梯度消失。"
/>

## 4. LSTM：建立一条近似加法的记忆通路

LSTM 引入细胞状态 $\mathbf c_t$。一套常见定义为

$$
\begin{aligned}
\mathbf i_t&=\sigma(\mathbf W_{xi}\mathbf x_t+\mathbf W_{hi}\mathbf h_{t-1}+\mathbf b_i),\\
\mathbf f_t&=\sigma(\mathbf W_{xf}\mathbf x_t+\mathbf W_{hf}\mathbf h_{t-1}+\mathbf b_f),\\
\mathbf o_t&=\sigma(\mathbf W_{xo}\mathbf x_t+\mathbf W_{ho}\mathbf h_{t-1}+\mathbf b_o),\\
\widetilde{\mathbf c}_t&=\tanh(\mathbf W_{xc}\mathbf x_t+\mathbf W_{hc}\mathbf h_{t-1}+\mathbf b_c),\\
\mathbf c_t&=\mathbf f_t\odot\mathbf c_{t-1}
+\mathbf i_t\odot\widetilde{\mathbf c}_t,\\
\mathbf h_t&=\mathbf o_t\odot\tanh(\mathbf c_t).
\end{aligned}
$$

若暂时只看从 $\mathbf c_{t-1}$ 到 $\mathbf c_t$ 的直接路径：

$$
\frac{\partial\mathbf c_t}{\partial\mathbf c_{t-1}}
\approx\operatorname{diag}(\mathbf f_t).
$$

当遗忘门 $\mathbf f_t$ 接近 1，梯度可沿加法状态通路长期传播；输入门决定写入多少候选记忆，输出门决定暴露多少记忆。它缓解而非数学上消灭梯度问题。

<MlQuiz
  question="LSTM 比普通 Tanh RNN 更利于长程梯度传播的核心结构是什么？"
  :options="['状态更新含受门控制的加法通路', '每步都使用不同权重', '删除了所有非线性', '只保留最后一个输入']"
  :answer="0"
  explanation="细胞状态通过遗忘门缩放后与新记忆相加，可形成接近恒等映射的梯度路径。"
/>

## 5. GRU：把门控写得更紧凑

一套 GRU 约定为

$$
\begin{aligned}
\mathbf z_t&=\sigma(\mathbf W_{xz}\mathbf x_t+\mathbf W_{hz}\mathbf h_{t-1}+\mathbf b_z),\\
\mathbf r_t&=\sigma(\mathbf W_{xr}\mathbf x_t+\mathbf W_{hr}\mathbf h_{t-1}+\mathbf b_r),\\
\widetilde{\mathbf h}_t
&=\tanh(\mathbf W_{xh}\mathbf x_t+
\mathbf W_{hh}(\mathbf r_t\odot\mathbf h_{t-1})+\mathbf b_h),\\
\mathbf h_t
&=(1-\mathbf z_t)\odot\mathbf h_{t-1}
+\mathbf z_t\odot\widetilde{\mathbf h}_t.
\end{aligned}
$$

注意有些资料交换 $\mathbf z_t$ 与 $1-\mathbf z_t$ 的语义；判断时看最终更新式，不要只背“更新门越大保留越多”这句话。GRU 没有单独的细胞状态和输出门，参数更少。

<MlQuiz
  question="阅读不同资料的 GRU 公式时，为什么要先看最终状态更新式？"
  :options="['变量名和门的取反约定可能不同', 'GRU 没有参数', '所有公式都必须相同', '最终状态与门无关']"
  :answer="0"
  explanation="不同实现可能把 z 定义为保留旧状态或写入新状态的比例，最终更新式才决定语义。"
/>

## 6. 参数量与形状

输入维度 $d$、隐藏维度 $h$。普通 RNN 的状态层参数量为

$$
hd+h^2+h.
$$

LSTM 有四组门/候选参数：

$$
4(hd+h^2+h).
$$

GRU 通常为

$$
3(hd+h^2+h).
$$

实际框架常把四组矩阵拼成大矩阵以提高吞吐，数学含义不变。

::: ml-stop 停下来推一推 3
取 $d=100,h=256$，分别计算普通 RNN、GRU、LSTM 状态层参数量。不要漏掉偏置。
:::

<MlQuiz
  question="将 LSTM 四组门矩阵拼接为一个大矩阵，主要改变了什么？"
  :options="['模型表达式本质', '计算实现效率，不改变门的数学含义', '序列标签', '损失函数必须改为平方损失']"
  :answer="1"
  explanation="拼接让一次矩阵乘法同时产生四组预激活，是等价的批量实现。"
/>

## 7. 一个清晰的序列前向

~~~python
import numpy as np

def rnn_forward(xs, wxh, whh, b, h0):
    h = h0
    states = [h0]
    for x_t in xs:
        a = wxh @ x_t + whh @ h + b
        h = np.tanh(a)
        states.append(h)
    return np.stack(states)
~~~

这段实现按时间顺序运行，难以像卷积一样完全并行。下一章的自注意力会用“任意位置直接读取任意位置”换掉递推路径。

## 8. 动手练习：复制记忆任务

构造序列：第一个位置给一个标量，随后填入 $T-1$ 个零，目标是在最后一步输出第一个值。

1. 分别训练普通 Tanh RNN 与 GRU；
2. 让 $T$ 从 10 增加到 100；
3. 记录首个输入处的梯度范数；
4. 比较梯度裁剪前后的训练曲线；
5. 说明裁剪为何不能解决普通 RNN 的长程遗忘。

<MlLessonProgress
  storage-key="16-rnn"
  next-href="/courses/machine-learning/17-transformer"
  next-label="下一章：Transformer 为什么不需要递归"
/>
