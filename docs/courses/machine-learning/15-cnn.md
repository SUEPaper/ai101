---
title: 15. CNN 为什么适合网格数据
description: 从局部连接、参数共享与平移等变推导卷积神经网络
course: machine-learning
---

<MlLessonHero
  eyebrow="第四模块 · 神经网络"
  title="CNN 为什么适合网格数据"
  subtitle="卷积不是因为图像“只能这样处理”，而是把局部性与位置共享写进模型。"
  :diagnostics="['能手算卷积输出尺寸与感受野', '能解释参数共享带来的归纳偏置', '能推导卷积核和输入的梯度']"
/>

## 1. 全连接层忽略了图像的结构

一张 $224\times224$ 的 RGB 图片有 $150\,528$ 个输入。若直接连接到 1 000 个隐藏单元，仅一层就有约 $1.5\times10^8$ 个权重，而且“猫耳探测器”在左上角学到后，不能自动复用于右下角。

图像和其他网格数据通常具有两个可利用的假设：

1. **局部性**：邻近像素的组合比相距很远的像素更常形成局部模式；
2. **位置共享**：同一种边缘或纹理可能出现在任意位置。

卷积层分别用局部连接与参数共享表达这两个归纳偏置。

## 2. 深度学习库实际计算的是互相关

对输入 $\mathbf X\in\mathbb R^{C_{\rm in}\times H\times W}$、卷积核
$\mathbf K\in\mathbb R^{C_{\rm out}\times C_{\rm in}\times K_h\times K_w}$，输出为

$$
Y_{oij}
=b_o+\sum_{c=1}^{C_{\rm in}}\sum_{u=0}^{K_h-1}\sum_{v=0}^{K_w-1}
K_{ocuv}\,X_{c,\,iS_h+u-P_h,\,jS_w+v-P_w}.
$$

越界输入按 0 处理。数学卷积会把核翻转，深度学习库通常不翻转，所以严格说这是互相关；核本来就是学出来的，命名差异不影响表达能力。

输出高度和宽度是

$$
H_{\rm out}
=\left\lfloor
\frac{H+2P_h-D_h(K_h-1)-1}{S_h}+1
\right\rfloor,
$$

$$
W_{\rm out}
=\left\lfloor
\frac{W+2P_w-D_w(K_w-1)-1}{S_w}+1
\right\rfloor,
$$

其中 $S$ 是步幅，$P$ 是填充，$D$ 是膨胀率。普通卷积取 $D=1$。

::: ml-stop 停下来推一推 1
输入 $32\times32$，核 $5\times5$，填充 2、步幅 2、膨胀率 1。手算输出尺寸。为什么公式中必须有向下取整？
:::

<MlQuiz
  question="输入 28×28，使用 3×3 核、步幅 1、填充 1，输出空间尺寸是多少？"
  :options="['26×26', '27×27', '28×28', '30×30']"
  :answer="2"
  explanation="代入公式得到 (28+2-3)/1+1=28，这种填充常称为 same padding。"
/>

<MlLab
  kind="convolution"
  title="实验：核、步幅与填充如何改变特征图"
  description="调节核尺度和步幅，观察输出尺寸、参数量与局部响应。"
/>

## 3. 参数量为什么骤降

全连接层从 $C_{\rm in}HW$ 映射到 $M$ 个单元，参数量近似为

$$
M(C_{\rm in}HW+1).
$$

卷积层的参数量与图像尺寸无关：

$$
C_{\rm out}(C_{\rm in}K_hK_w+1).
$$

以 $C_{\rm in}=3,C_{\rm out}=64,K_h=K_w=3$ 为例，仅有

$$
64(3\cdot3\cdot3+1)=1792
$$

个参数。同一个 $K_{ocuv}$ 被所有空间位置复用，因此其梯度会汇总所有位置的证据。

<MlQuiz
  question="标准卷积层的参数量与下列哪项无关？"
  :options="['输入通道数', '输出通道数', '卷积核尺寸', '输入图像的高和宽']"
  :answer="3"
  explanation="核在空间位置共享，所以参数量不随 H、W 增加；计算量仍会随输出位置数增加。"
/>

## 4. 平移等变，不等于平移不变

令 $T_\Delta$ 表示把输入平移 $\Delta$。忽略边界和步幅问题时，卷积满足

$$
\operatorname{Conv}(T_\Delta\mathbf X,\mathbf K)
=T_\Delta\operatorname{Conv}(\mathbf X,\mathbf K).
$$

输入移动，特征图跟着移动，这叫**平移等变**。而分类输出不随移动变化才叫**平移不变**。全局池化可把空间维汇总，从等变特征得到近似不变表示：

$$
g_c=\frac{1}{HW}\sum_{i,j}X_{cij}.
$$

步幅、边界填充和采样会破坏严格等变性；数据增强则进一步告诉模型哪些变换应保持语义。

<MlQuiz
  question="卷积层最直接保证的结构性质是？"
  :options="['任意旋转不变', '平移等变', '尺度不变', '所有类别概率相同']"
  :answer="1"
  explanation="同一核在每个位置使用，使输入平移时响应也相应平移；不变性通常还需池化或汇总。"
/>

## 5. 堆叠怎样扩大感受野

第 $l$ 层某单元在原输入上的感受野大小记为 $r_l$，相邻单元在原输入上的间距记为 $j_l$。初始 $r_0=j_0=1$。对于核宽 $K_l$、步幅 $S_l$、膨胀率 $D_l$：

$$
j_l=j_{l-1}S_l,
$$

$$
r_l=r_{l-1}+(K_l-1)D_lj_{l-1}.
$$

三个步幅为 1 的 $3\times3$ 卷积依次得到 $3,5,7$ 的理论感受野，却比一个 $7\times7$ 卷积插入更多非线性，参数通常也更少。

::: ml-stop 停下来推一推 2
两层 $3\times3$ 卷积，第一层步幅 2，第二层步幅 1。逐层计算 $j_l$ 和 $r_l$。
:::

<MlQuiz
  question="两个步幅 1、无膨胀的 3×3 卷积连续堆叠，理论感受野边长是多少？"
  :options="['3', '4', '5', '6']"
  :answer="2"
  explanation="第一层感受野为 3，第二层再向两侧覆盖一个第一层步长，总边长为 3+2=5。"
/>

## 6. 卷积的反向传播

为简化记号，考虑单通道、步幅 1、无填充：

$$
Y_{ij}=\sum_{u,v}K_{uv}X_{i+u,j+v}.
$$

设上游梯度 $G_{ij}=\partial L/\partial Y_{ij}$。某个核元素 $K_{uv}$ 影响每个输出位置，因此

$$
\frac{\partial L}{\partial K_{uv}}
=\sum_{i,j}G_{ij}X_{i+u,j+v}.
$$

某个输入 $X_{pq}$ 也影响多个输出，需要收集所有覆盖它的窗口：

$$
\frac{\partial L}{\partial X_{pq}}
=\sum_{u,v}G_{p-u,q-v}K_{uv},
$$

越界的 $G$ 视为 0。这里出现翻转索引，正是“前向互相关、反向对输入像卷积”的来源。多通道时再对对应输入或输出通道求和。

<MlQuiz
  question="为什么卷积核某个元素的梯度要对所有输出位置求和？"
  :options="['核元素只在一个位置使用', '该参数在所有空间位置共享，总导数累加每次使用的贡献', '为了让输出变方形', '因为 Softmax 要求求和']"
  :answer="1"
  explanation="参数共享意味着同一变量出现在许多计算路径上，分支反向规则要求把贡献相加。"
/>

## 7. 从零实现一个二维互相关

~~~python
import numpy as np

def conv2d_single(x, kernel, stride=1, padding=0):
    x_pad = np.pad(x, ((padding, padding), (padding, padding)))
    kh, kw = kernel.shape
    oh = (x_pad.shape[0] - kh) // stride + 1
    ow = (x_pad.shape[1] - kw) // stride + 1
    out = np.empty((oh, ow), dtype=float)
    for i in range(oh):
        for j in range(ow):
            patch = x_pad[i*stride:i*stride+kh,
                          j*stride:j*stride+kw]
            out[i, j] = np.sum(patch * kernel)
    return out
~~~

这段代码适合检查定义，不适合生产性能。真实框架会使用隐式矩阵乘法、Winograd 或专用内核。

## 8. 池化、步幅卷积与失效边界

最大池化保留窗口最大响应，平均池化保留局部均值。池化没有可学习参数，但会丢失精确位置；步幅卷积能学习下采样规则。选择哪一种不是定律，要根据：

- 是否需要像素级定位；
- 是否允许空间信息快速压缩；
- 小目标是否可能被下采样抹掉；
- 计算与显存预算。

CNN 的先验适合局部、近似平移共享的网格。对长距离关系、集合结构或不规则图，它可能需要很深网络或不同算子。

::: ml-stop 停下来推一推 3
语义分割为什么不能像图像分类那样一路把空间尺寸压到 $1\times1$？说明任务输出结构与网络结构的关系。
:::

<MlQuiz
  question="当任务要求每个像素都有预测时，过早使用强下采样的主要风险是？"
  :options="['参数一定变多', '细粒度位置和小目标信息丢失', '激活函数变成线性', '标签数量自动减少']"
  :answer="1"
  explanation="密集预测依赖空间对应关系，强下采样可能造成不可恢复的定位信息损失。"
/>

## 9. 动手练习：边缘核与梯度检查

用上面的实现完成：

1. 对一张灰度小图应用水平、垂直 Sobel 核；
2. 比较零填充与不填充的边界响应；
3. 手写 $\partial L/\partial K$；
4. 对核的每个元素做有限差分检查；
5. 解释为何核梯度看起来像输入窗口与上游梯度的相关。

<MlLessonProgress
  storage-key="15-cnn"
  next-href="/courses/machine-learning/16-rnn"
  next-label="下一章：RNN 怎样对序列共享状态"
/>
