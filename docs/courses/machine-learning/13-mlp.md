---
course: machine-learning
edition: industrial
title: 多层感知机（MLP）
description: 以激光焊接双传感器一致性分流为贯穿案例，从仿射层、非线性表示、激活函数、输出概率模型、对称性与方差传播推导可部署的多层感知机。
outline: [2, 3]
---

<MlLessonHero
  number="13"
  title="多层感知机（MLP）"
  question="当一条直线无法表达工艺规则时，怎样用可审计的神经元、激活函数和初始化，把输入空间折叠成可学习的决策区域？"
  duration="建议 520–640 分钟"
  :outcomes="[
    '从矩阵形状和仿射复合证明没有激活函数的深层网络仍等价于单层线性模型',
    '用四个 ReLU 单元精确构造 XOR 型一致性规则，并区分表示存在性、可训练性与泛化',
    '依据概率模型匹配输出层和损失，从二阶矩传播推导 Xavier 与 He 初始化并建立数值监控'
  ]"
/>

上一章把非线性写进固定核函数。本章换一条路线：让模型自己学习特征映射

$$
h_\theta(x)=\phi_L\!\left(
W_L\phi_{L-1}(\cdots \phi_1(W_1x+b_1)\cdots)+b_L
\right).
$$

多层感知机（multilayer perceptron, MLP）不是“许多线性层”的同义词。它必须同时包含：

1. 可学习的仿射变换，用来旋转、缩放和平移坐标；
2. 非线性激活，用来切换局部计算路径；
3. 与任务概率模型匹配的输出头和损失；
4. 能让信号与梯度跨层传播的初始化；
5. 明确的数据、计算、阈值和监控合同。

本章不把神经网络当黑箱。我们会从一个无法被单条直线分开的工艺规则出发，手工写出能精确实现它的网络，再把每个工程接口拆开。

## 贯穿案例：激光焊接双传感器一致性分流

某激光焊接工位同时记录熔池热像和光电二极管信号。经基线模型校正后，每个焊点得到两个无量纲残差：

| 字段 | 本章含义 | 正值 | 负值 |
|---|---|---|---|
| thermal_residual | 峰值温度相对工艺基线的标准化残差 | 偏热 | 偏冷 |
| optical_residual | 光强积分相对工艺基线的标准化残差 | 偏强 | 偏弱 |

教学规则是：

$$
y=
\begin{cases}
1,&x_1x_2<0\quad\text{两路方向相反，送人工复核},\\
0,&x_1x_2>0\quad\text{两路方向一致，允许进入后续工序}.
\end{cases}
$$

这就是连续输入上的 XOR 几何：第一、第三象限一类，第二、第四象限另一类。它不是任何真实焊接产线可直接采用的标签规则。真实标签必须来自破坏性检测、截面金相、无损检测或经过校准的专家复判；两个代理传感器不能互相证明产品质量。

### 数据合同

| 对象 | 教学定义 | 工业系统必须冻结 |
|---|---|---|
| 观测单元 | 一个焊点的一次同步采集 | 重焊、重复触发和缺帧如何归并 |
| 特征 | 两路经训练集基线校正的残差 | 原始单位、采样窗、滤波、固件版本 |
| 标签 | 符号不一致为复核 | 标签来源、盲评流程、争议样本状态 |
| 分组键 | batch_id | 原料卷、设备、班次和配方的嵌套关系 |
| 切分 | 32/8/8 个批次 | 未来部署层级与时间外推窗口 |
| 决策 | review_label=1 进入人工复核 | 阈值、产能、漏检成本和降级策略 |

教学数据含 48 个批次、每批 24 个焊点。批次偏移和测量噪声会让四个象限形成重叠边缘；切分必须按批次，而不是把同一批次随机打散。

```python
import numpy as np

rng = np.random.default_rng(20260805)
batch_count = 48
points_per_batch = 24
batch_id = np.repeat(np.arange(batch_count), points_per_batch)
sample_count = len(batch_id)

corner_sign = rng.choice([-1.0, 1.0], size=(sample_count, 2))
corner_radius = rng.normal(1.25, 0.24, size=(sample_count, 2))
batch_shift = rng.normal(0.0, 0.10, size=(batch_count, 2))
features = corner_sign * corner_radius
features += batch_shift[batch_id]
features += rng.normal(0.0, 0.10, size=features.shape)
review_label = (features[:, 0] * features[:, 1] < 0).astype(np.int64)

batch_order = rng.permutation(batch_count)
train_batches = batch_order[:32]
validation_batches = batch_order[32:40]
test_batches = batch_order[40:]
train_mask = np.isin(batch_id, train_batches)
validation_mask = np.isin(batch_id, validation_batches)
test_mask = np.isin(batch_id, test_batches)

train_mean = features[train_mask].mean(axis=0)
train_scale = features[train_mask].std(axis=0, ddof=0)
standardized = (features - train_mean) / train_scale
X_train, y_train = standardized[train_mask], review_label[train_mask]
X_validation, y_validation = standardized[validation_mask], review_label[validation_mask]
X_test, y_test = standardized[test_mask], review_label[test_mask]

assert len(X_train) == 32 * points_per_batch
assert len(X_validation) == len(X_test) == 8 * points_per_batch
assert set(train_batches).isdisjoint(validation_batches)
assert set(train_batches).isdisjoint(test_batches)
assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-12)
print("训练/验证/测试：", len(X_train), len(X_validation), len(X_test))
print("复核率：", round(review_label.mean(), 4))
```

<MlQuiz kind="标签合同" question="两路传感器残差方向相反，能否直接当作焊点缺陷的真实标签？" :options="['可以，XOR 规则就是物理定律', '不可以；它只能定义教学分流规则，质量标签仍需独立检测证据', '可以，只要训练准确率高', '只需把残差标准化就可以']" :answer="1" explanation="传感器一致性是代理信号，不是产品质量真值；工业闭环必须保留独立标签来源。" />

## 1. 一个神经元就是仿射投影加非线性

对输入 $x\in\mathbb R^d$，一个神经元计算

$$
z=w^Tx+b,\qquad h=\phi(z).
$$

$z$ 称为 pre-activation，$h$ 称为 activation。$w$ 决定观察方向，$b$ 决定阈值，$\phi$ 决定这个投影通过何种非线性门。

把 $m$ 个神经元并排，就得到全连接层。采用“样本在行、特征在列”的约定：

$$
\begin{aligned}
X&\in\mathbb R^{B\times d},\\
W&\in\mathbb R^{d\times m},\\
b&\in\mathbb R^m,\\
Z&=XW+\mathbf 1_Bb^T\in\mathbb R^{B\times m},\\
H&=\phi(Z)\in\mathbb R^{B\times m}.
\end{aligned}
$$

偏置在数学上被复制到每一行，在 NumPy 中由 broadcasting 完成。广播只是存储和语法便利，不改变导数：反向传播时，复制到 $B$ 行的偏置梯度必须沿批维求和。

```python
import numpy as np

rng = np.random.default_rng(13)
B, d, m = 32, 10, 64
X = rng.normal(size=(B, d))
W = rng.normal(scale=0.1, size=(d, m))
b = np.zeros(m)
Z = X @ W + b
H = np.maximum(Z, 0.0)

assert X.shape == (32, 10)
assert W.shape == (10, 64)
assert b.shape == (64,)
assert Z.shape == H.shape == (32, 64)
assert np.all(H >= 0.0)
```

<div class="ml-stop">
  <strong>停下来推一推 1</strong>
  <p>若 X 的形状是 (32, 10)，隐藏层宽度为 64，请写出 W、b、Z、H 的形状。再说明为什么偏置梯度不能只取某一个样本的贡献。</p>
</div>

<MlQuiz kind="形状合同" question="批量输入 X 为 (B,d)，权重 W 为 (d,m)，偏置 b 为 (m,)。激活 H 的形状是什么？" :options="['(d,m)', '(B,d)', '(B,m)', '(m,B)']" :answer="2" explanation="XW 得到每个样本的 m 个 pre-activation；逐元素激活不改变形状。" />

## 2. 没有非线性，多深仍是一层

考虑两个仿射层：

$$
h=XW_1+\mathbf 1b_1^T,\qquad
Y=hW_2+\mathbf 1b_2^T.
$$

代入第一式：

$$
\begin{aligned}
Y
&=(XW_1+\mathbf 1b_1^T)W_2+\mathbf 1b_2^T\\
&=X(W_1W_2)+\mathbf 1(b_1^TW_2+b_2^T).
\end{aligned}
$$

令 $\widetilde W=W_1W_2$、$\widetilde b=W_2^Tb_1+b_2$，复合仍是一个仿射层。归纳可知，任意有限个仿射层的复合仍是仿射函数。

深层线性网络的参数化和单层线性模型不同，因此优化轨迹、隐式偏置和矩阵秩约束可能不同；但就输入到输出的函数族而言，它没有制造非线性决策边界。

```python
import numpy as np

rng = np.random.default_rng(1301)
X = rng.normal(size=(17, 5))
W1, b1 = rng.normal(size=(5, 7)), rng.normal(size=7)
W2, b2 = rng.normal(size=(7, 3)), rng.normal(size=3)
W3, b3 = rng.normal(size=(3, 2)), rng.normal(size=2)

deep = ((X @ W1 + b1) @ W2 + b2) @ W3 + b3
collapsed_W = W1 @ W2 @ W3
collapsed_b = (b1 @ W2 + b2) @ W3 + b3
shallow = X @ collapsed_W + collapsed_b

assert np.allclose(deep, shallow, atol=1e-12)
```

<MlpCoherenceRouterLab />

在 13-A 中把激活切成“恒等”，四条隐藏路径会重新合并成一条平面，XOR 式象限交替结构消失。切回 ReLU 后，每个门只在一个半空间开启，输出层可以重新组合这些局部响应。

<MlQuiz kind="表示折叠" question="把 20 个全连接层串联，但所有隐藏层都使用恒等激活。输入到输出的函数是什么？" :options="['20 次多项式', '仍是一个仿射变换', '任意连续函数', '分段常数函数']" :answer="1" explanation="矩阵乘法和偏置的有限次复合仍可折叠为一个矩阵与一个偏置。" />

## 3. 为什么一条直线不能解决 XOR

取四个理想点：

$$
(1,1)\mapsto0,\quad(-1,-1)\mapsto0,\quad
(1,-1)\mapsto1,\quad(-1,1)\mapsto1.
$$

若存在线性分数 $s=w_1x_1+w_2x_2+b$，并规定复核点分数为正、一致点为负，则有

$$
\begin{aligned}
w_1+w_2+b&<0,\\
-w_1-w_2+b&<0.
\end{aligned}
$$

相加得 $2b<0$。而另两个点要求

$$
\begin{aligned}
w_1-w_2+b&>0,\\
-w_1+w_2+b&>0,
\end{aligned}
$$

相加得 $2b>0$，矛盾。问题不是线性模型“参数不够多”，而是标签区域不是一个半空间。

```python
import numpy as np

X_xor = np.array([[1., 1.], [-1., -1.], [1., -1.], [-1., 1.]])
y_xor = np.array([0., 0., 1., 1.])
design = np.column_stack([X_xor, np.ones(len(X_xor))])
least_squares_parameter, *_ = np.linalg.lstsq(design, y_xor, rcond=None)
linear_score = design @ least_squares_parameter
linear_prediction = (linear_score >= 0.5).astype(int)

assert (linear_prediction == y_xor).mean() <= 0.5
assert np.allclose(linear_score, 0.5)
```

<MlQuiz kind="线性可分性" question="XOR 四个角点为什么不能被一个仿射二分类器完全分开？" :options="['因为输入维度太高', '因为两类各占交替象限，不存在一个同时满足四组符号约束的半空间', '因为样本太少', '因为标签必须改成 -1 和 1']" :answer="1" explanation="把四个线性不等式两两相加，会同时推出 b 小于 0 和 b 大于 0。" />

## 4. 四个 ReLU 门可以精确表达一致性规则

定义

$$
\begin{aligned}
h_1&=\operatorname{ReLU}(x_1+x_2),\\
h_2&=\operatorname{ReLU}(-x_1-x_2),\\
h_3&=\operatorname{ReLU}(x_1-x_2),\\
h_4&=\operatorname{ReLU}(-x_1+x_2).
\end{aligned}
$$

输出分数取

$$
s(x)=-h_1-h_2+h_3+h_4.
$$

利用 $\operatorname{ReLU}(a)+\operatorname{ReLU}(-a)=|a|$，

$$
s(x)=|x_1-x_2|-|x_1+x_2|.
$$

当 $x_1x_2<0$ 时，$|x_1-x_2|>|x_1+x_2|$，所以 $s>0$；当 $x_1x_2>0$ 时，$s<0$。轴上 $s=0$，正好是教学规则未定义的边界。

这套权重是“存在性证书”：它证明一个宽度为 4 的单隐层 ReLU 网络足以表达规则。它没有证明随机初始化和梯度下降一定会找到这组参数。

```python
import numpy as np

def relu(z):
    return np.maximum(z, 0.0)

W1 = np.array([
    [1.0, -1.0, 1.0, -1.0],
    [1.0, -1.0, -1.0, 1.0],
])
b1 = np.zeros(4)
W2 = np.array([-1.0, -1.0, 1.0, 1.0])
b2 = 0.0

def explicit_xor_score(x):
    hidden = relu(x @ W1 + b1)
    return hidden @ W2 + b2

rng = np.random.default_rng(1302)
sign = rng.choice([-1.0, 1.0], size=(1000, 2))
test_points = sign * rng.uniform(0.1, 2.0, size=(1000, 2))
target = (test_points[:, 0] * test_points[:, 1] < 0).astype(int)
prediction = (explicit_xor_score(test_points) > 0).astype(int)

assert np.array_equal(prediction, target)
assert np.isclose(explicit_xor_score(np.array([[1.0, -1.0]]))[0], 2.0)
assert np.isclose(explicit_xor_score(np.array([[1.0, 1.0]]))[0], -2.0)
```

<MlQuiz kind="构造网络" question="四门网络的输出 s=|x₁−x₂|−|x₁+x₂| 在什么区域为正？" :options="['两个输入同号', '两个输入异号', '只在原点', '所有区域']" :answer="1" explanation="异号时差的绝对值大于和的绝对值，对应送复核的两个象限。" />

## 5. ReLU 网络如何折叠空间

对固定的激活状态，每个 ReLU 要么传递线性值，要么输出 0。设二值对角矩阵

$$
D(x)=\operatorname{diag}\left(
\mathbf 1[z_1>0],\ldots,\mathbf 1[z_m>0]
\right).
$$

单隐层网络在该状态区域内可写成

$$
f(x)=W_2^TD(x)(W_1^Tx+b_1)+b_2,
$$

因此在每个激活区域内部都是仿射函数；不同区域使用不同的有效斜率和偏置。ReLU 网络的非线性来自激活状态跨越超平面时发生切换，而不是每个局部区域都变成高次多项式。

对 $m$ 个一般位置的超平面，二维空间最多被切成

$$
1+m+\binom m2
$$

个区域；高维一般上界为 $\sum_{j=0}^d\binom mj$。深层网络会在前一层已经折叠后的坐标中继续切分，因此区域数可以组合增长，但实际达到多少取决于权重、宽度和输入分布。

```python
import numpy as np

axis = np.linspace(-2.0, 2.0, 81)
grid_x1, grid_x2 = np.meshgrid(axis, axis)
grid = np.column_stack([grid_x1.ravel(), grid_x2.ravel()])
pre_activation = grid @ W1
activation_code = (pre_activation > 0).astype(np.int64)
region_id = activation_code @ (1 << np.arange(4))

unique_regions = np.unique(region_id)
assert 4 <= len(unique_regions) <= 11

probe = np.array([[1.4, -0.7], [1.2, -0.6], [1.0, -0.5]])
probe_code = (probe @ W1 > 0).astype(int)
probe_score = explicit_xor_score(probe)
linear_step = probe_score[1] - probe_score[0]
assert np.all(probe_code == probe_code[0])
assert np.isclose(probe_score[2] - probe_score[1], linear_step)
```

<MlQuiz kind="分段仿射" question="在一个固定 ReLU 激活模式内部，MLP 对输入是什么函数？" :options="['仿射函数', '正弦函数', '任意不连续函数', '概率密度']" :answer="0" explanation="激活开关固定后，ReLU 等价于一个固定对角选择矩阵，整个网络可折叠为仿射映射。" />

## 6. 表示能力、优化能力与泛化能力必须分开

万能逼近定理的常用直觉是：对紧致区域上的连续目标函数，使用合适非多项式激活且隐藏单元足够多的单隐层网络，存在一组参数能把误差压到任意小。

这个结论回答“是否存在”，不回答：

- 需要多宽、多少样本和多少计算；
- 优化器能否在有限时间找到那组参数；
- 噪声标签下会不会记忆训练集；
- 训练区域之外是否正确外推；
- 量化、延迟和内存约束下是否仍可部署；
- 概率是否校准、阈值是否满足产能。

“网络能表示 XOR”与“网络能从 32 个训练批次学出可靠的批次外规则”是两个命题。工业验收关注后者。

### 固定特征与学习特征

核方法写成

$$
f(x)=\sum_i\alpha_iK(x,x_i)+b,
$$

相似度结构由核和训练样本锚定。MLP 写成

$$
f_\theta(x)=g_{\theta_2}(h_{\theta_1}(x)),
$$

特征映射 $h_{\theta_1}$ 与输出头一起从任务损失中学习。

| 维度 | 核方法 | MLP |
|---|---|---|
| 表示先验 | 人工选择核 | 通过架构和数据学习表示 |
| 典型优化 | 核 SVM 可保持凸 | 参数联合优化通常非凸 |
| 样本规模 | Gram 矩阵可能成为瓶颈 | mini-batch 易扩展 |
| 审计入口 | 支持向量、核参数 | 激活、梯度、表示漂移 |
| 外推风险 | 取决于核相似度 | 取决于 learned representation |

<MlQuiz kind="定理边界" question="万能逼近定理最直接保证什么？" :options="['梯度下降必然找到全局最优', '足够宽时存在逼近紧致域连续函数的参数', '训练域外必然正确', '小样本下必然校准']" :answer="1" explanation="它是函数表示的存在性结论，不是优化、样本效率、外推或部署保证。" />

## 7. 激活函数：选择梯度通过的工作区

<ActivationFunctionScopeLab />

### 7.1 Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}},\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z)).
$$

它输出 $(0,1)$，适合二分类概率头。作为深层隐藏激活时，大绝对值输入让导数接近 0，且输出不是零中心，容易造成梯度和优化问题。

### 7.2 Tanh

$$
\tanh'(z)=1-\tanh^2(z).
$$

Tanh 输出 $(-1,1)$ 且零中心，但同样在两端饱和。循环网络中的门控结构仍常使用它，因为有界状态具有明确作用；这不意味着它是所有深层隐藏层的默认选择。

### 7.3 ReLU 与 Leaky ReLU

$$
\operatorname{ReLU}(z)=\max(0,z),\qquad
\operatorname{LeakyReLU}(z)=\max(\alpha z,z).
$$

ReLU 正半轴导数为 1，计算简单；负半轴导数为 0。若一个单元对所有训练样本都落在负侧，它可能长期没有梯度。Leaky ReLU 用小斜率 $\alpha$ 保留负半轴路径，但不能替代学习率、初始化和输入尺度检查。

### 7.4 GELU

$$
\operatorname{GELU}(z)=z\Phi(z).
$$

GELU 是平滑门控，负值附近不会被硬切为 0，在 Transformer 等架构中常见。选择它应基于架构基线和验证结果，而不是因为公式更复杂。

### 7.5 导数实现必须做数值核对

```python
import numpy as np

def stable_sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out

def sigmoid_derivative(z):
    value = stable_sigmoid(z)
    return value * (1.0 - value)

probe = np.linspace(-6.0, 6.0, 31)
epsilon = 1e-5
numeric = (
    stable_sigmoid(probe + epsilon)
    - stable_sigmoid(probe - epsilon)
) / (2.0 * epsilon)

assert np.max(np.abs(numeric - sigmoid_derivative(probe))) < 1e-9
assert stable_sigmoid(np.array([-1000.0, 1000.0])).tolist() == [0.0, 1.0]
```

<MlQuiz kind="激活工作区" question="一个 Sigmoid 隐藏单元长期收到绝对值很大的 pre-activation，最直接的问题是什么？" :options="['输出维度增加', '导数接近 0，传回该路径的梯度很小', '自动变成 Softmax', '参数量翻倍']" :answer="1" explanation="Sigmoid 两端饱和；实验 13-B 会同时显示函数值、局部导数和样本占用区间。" />

## 8. 输出层不是装饰：它定义概率模型

隐藏层产生表示，输出头把表示映射到任务需要的自然参数。损失函数应与观测模型一致：

| 任务 | 输出 logits | 预测变换 | 典型负对数似然 |
|---|---:|---|---|
| 实数回归 | $K$ 个实数 | 恒等 | 高斯假设下的平方误差 |
| 二分类 | 1 个 | Sigmoid | binary cross-entropy |
| 单标签 $K$ 分类 | $K$ 个 | Softmax | categorical cross-entropy |
| 多标签 $K$ 分类 | $K$ 个 | 各维独立 Sigmoid | 各维 BCE 之和 |
| 正计数 | 1 个 | exponential/softplus | Poisson NLL |

单标签多分类要求一个样本只属于一个互斥类别，Softmax 概率和为 1。多标签任务允许“气孔、飞溅、欠熔”同时存在，应使用独立 Sigmoid。

### 直接从 logits 计算稳定损失

二分类的稳定形式是

$$
\ell(z,y)=\max(z,0)-zy+\log(1+e^{-|z|}).
$$

Softmax 交叉熵使用 log-sum-exp：

$$
\ell(z,y)=-z_y+
\left[m+\log\sum_k e^{z_k-m}\right],
\qquad m=\max_kz_k.
$$

先算概率再取对数，会在极端 logit 下产生 0、1、上溢或 $\log0$。训练 API 应优先接收 logits 的 fused loss。

```python
import numpy as np

def binary_cross_entropy_with_logits(logit, target):
    logit = np.asarray(logit, dtype=float)
    target = np.asarray(target, dtype=float)
    return np.maximum(logit, 0.0) - logit * target + np.log1p(np.exp(-np.abs(logit)))

def multiclass_cross_entropy(logits, target_index):
    logits = np.asarray(logits, dtype=float)
    target_index = np.asarray(target_index, dtype=int)
    maximum = logits.max(axis=1, keepdims=True)
    log_normalizer = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
    return log_normalizer - logits[np.arange(len(logits)), target_index]

bce = binary_cross_entropy_with_logits(
    np.array([-1000.0, 1000.0]),
    np.array([0.0, 1.0]),
)
ce = multiclass_cross_entropy(
    np.array([[1000.0, 999.0, -1000.0], [-800.0, -801.0, -799.0]]),
    np.array([0, 2]),
)

assert np.all(np.isfinite(bce))
assert np.all(np.isfinite(ce))
assert np.all(bce < 1e-12)
```

<div class="ml-stop">
  <strong>停下来推一推 2</strong>
  <p>一个视觉模型需要同时报告气孔、飞溅、欠熔三种缺陷是否存在。应使用三个独立 Sigmoid 还是一个三类 Softmax？写出概率事件为什么不同。</p>
</div>

<MlQuiz kind="输出合同" question="一张图可能同时含有三种缺陷，最合适的输出头是什么？" :options="['三类 Softmax', '三个独立 Sigmoid', '一个 ReLU', '一个未经约束的标量']" :answer="1" explanation="三个标签不是互斥事件；每一维需要自己的 Bernoulli 概率。" />

## 9. 一次批量前向的完整数据流

对一层隐藏的二分类 MLP：

$$
\begin{aligned}
Z_1&=XW_1+b_1,\\
H_1&=\operatorname{ReLU}(Z_1),\\
z_2&=H_1w_2+b_2,\\
p&=\sigma(z_2).
\end{aligned}
$$

训练时需要保存能重建局部导数的 cache。保存什么取决于算子：ReLU 至少需要知道 $Z_1$ 的符号，线性层需要输入和权重。盲目缓存所有中间张量会增加内存；过早丢弃则必须重算。

```python
import numpy as np

def mlp_forward(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    Z1 = X @ W1 + b1
    H1 = np.maximum(Z1, 0.0)
    logits = H1 @ W2 + b2
    cache = {"X": X, "Z1": Z1, "H1": H1}
    return logits, cache

rng = np.random.default_rng(1303)
parameters = {
    "W1": rng.normal(scale=np.sqrt(2 / 2), size=(2, 16)),
    "b1": np.zeros(16),
    "W2": rng.normal(scale=np.sqrt(2 / 16), size=(16, 1)),
    "b2": np.zeros(1),
}
mini_batch = rng.normal(size=(64, 2))
logits, cache = mlp_forward(mini_batch, parameters)

assert logits.shape == (64, 1)
assert cache["Z1"].shape == cache["H1"].shape == (64, 16)
assert set(cache) == {"X", "Z1", "H1"}
```

### 形状合同建议

在工业代码边界明确规定：

- 二分类 logits 统一为 $(B,1)$ 还是 $(B,)$，不要混用；
- 类别索引是 $(B,)$，one-hot 是 $(B,K)$；
- 特征 dtype、设备、缺失值策略和 batch 维必须校验；
- 推理单样本也保留 batch 维，避免条件分支；
- 输出先保留 logits，概率和阈值动作分层记录。

<MlQuiz kind="前向缓存" question="为什么训练前向通常需要保存部分中间量？" :options="['为了改变标签', '反向传播要用局部输入或激活状态计算梯度', '为了让模型变成凸函数', '为了增加类别数']" :answer="1" explanation="链式法则需要每个算子的局部信息；下一章将从这些 cache 推导反向传播。" />

## 10. 参数对称性：相同初始化不会自动分工

假设同一隐藏层的两个神经元权重和偏置完全相同。它们对每个输入产生相同输出；连接到下一层的初始条件若也对称，则反向收到相同梯度，更新后仍相同。两个单元始终是重复副本。

随机初始化的首要作用是打破这种对称性。这里的“随机”不等于尺度可以随意。

另一方面，已经训练好的隐藏单元可以任意置换，只要下一层对应连接做同样置换，网络函数不变。因此参数空间存在许多等价点，神经元编号本身没有语义。

```python
import numpy as np

rng = np.random.default_rng(1304)
X = rng.normal(size=(40, 3))
shared_weight = rng.normal(size=(3, 1))
identical_hidden = np.maximum(X @ np.repeat(shared_weight, 4, axis=1), 0.0)
assert np.allclose(identical_hidden, identical_hidden[:, [0]])

random_hidden_weight = rng.normal(size=(3, 4))
random_hidden = np.maximum(X @ random_hidden_weight, 0.0)
readout = rng.normal(size=4)
original_output = random_hidden @ readout

permutation = np.array([2, 0, 3, 1])
permuted_output = random_hidden[:, permutation] @ readout[permutation]
assert np.allclose(original_output, permuted_output)
```

<MlQuiz kind="对称性" question="为什么不能把同一隐藏层所有神经元的权重初始化成完全相同的值？" :options="['矩阵乘法会报错', '它们会产生相同输出并收到相同梯度，无法学习不同特征', '损失会自动变成零', '会增加显存']" :answer="1" explanation="相同初始条件保持置换对称，多个神经元等价于重复同一个特征。" />

## 11. 初始化来自二阶矩传播

考虑

$$
z_j=\sum_{i=1}^{n_{\text{in}}}w_{ij}x_i.
$$

在 $w_{ij}$ 与 $x_i$ 近似独立、均值为 0、各分量同方差时，

$$
\operatorname{Var}(z_j)
=n_{\text{in}}\operatorname{Var}(w)\operatorname{Var}(x).
$$

若希望线性层前后的方差近似不变，需要

$$
\operatorname{Var}(w)\approx\frac1{n_{\text{in}}}.
$$

### 11.1 Xavier / Glorot

同时兼顾前向激活和反向梯度，常取

$$
\operatorname{Var}(w)=\frac{2}{n_{\text{in}}+n_{\text{out}}}.
$$

它适合 Tanh、近线性的对称激活等基线。均匀分布版本若取 $w\sim U(-a,a)$，因方差为 $a^2/3$，得到

$$
a=\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}.
$$

### 11.2 He / Kaiming

对零均值对称输入，ReLU 约保留一半二阶矩，因此

$$
\operatorname E[\operatorname{ReLU}(z)^2]
\approx\frac12\operatorname E[z^2].
$$

要补偿这个因子，取

$$
\operatorname{Var}(w)=\frac{2}{n_{\text{in}}}.
$$

注意推导守恒的是二阶矩或近似方差，不是每层样本均值，也不是保证每一层精确相等。真实网络中的相关性、有限宽度、偏置、归一化和残差连接都会改变传播。

<InitializationSignalTunnelLab />

13-C 把同一批固定信号送入等宽随机网络。观察：

- 固定标准差在 fan-in 增大时可能迅速爆炸；
- Xavier 配 ReLU 往往逐层衰减，因为没有补偿半数截断；
- He 配 ReLU 的 RMS 通常更稳定；
- 换成 Tanh 后，数值可能有界但大量单元进入饱和区。

```python
import numpy as np

def propagation_trace(initialization, activation, depth=30, width=256, seed=19):
    rng = np.random.default_rng(seed)
    value = rng.normal(size=(512, width))
    rms_trace = []
    blocked_trace = []
    for _ in range(depth):
        if initialization == "fixed":
            scale = 0.2
        elif initialization == "xavier":
            scale = np.sqrt(1.0 / width)
        elif initialization == "he":
            scale = np.sqrt(2.0 / width)
        else:
            raise ValueError(initialization)
        weight = rng.normal(scale=scale, size=(width, width))
        pre_activation = value @ weight
        if activation == "relu":
            value = np.maximum(pre_activation, 0.0)
            blocked = np.mean(pre_activation <= 0.0)
        elif activation == "tanh":
            value = np.tanh(pre_activation)
            blocked = np.mean(np.abs(value) > 0.98)
        else:
            raise ValueError(activation)
        rms_trace.append(float(np.sqrt(np.mean(value ** 2))))
        blocked_trace.append(float(blocked))
    return np.array(rms_trace), np.array(blocked_trace)

fixed_rms, _ = propagation_trace("fixed", "relu")
xavier_rms, _ = propagation_trace("xavier", "relu")
he_rms, he_blocked = propagation_trace("he", "relu")

assert fixed_rms[-1] > 1.0
assert xavier_rms[-1] < 0.1
assert 0.1 < he_rms[-1] < 5.0
assert np.all((0.35 < he_blocked) & (he_blocked < 0.65))
```

<div class="ml-stop">
  <strong>停下来推一推 3</strong>
  <p>一个 ReLU 层的 fan-in 为 200。先写出 He 正态初始化的方差，再写标准差。为什么答案不是 2/200？</p>
</div>

<MlQuiz kind="方差传播" question="ReLU 网络的 He 初始化为什么常取 Var(w)=2/fan-in？" :options="['让所有权重之和为 1', '在线性方差尺度上补偿 ReLU 约截掉一半二阶矩', '保证损失凸', '让隐藏单元输出相同']" :answer="1" explanation="1/fan-in 维持线性组合尺度，额外的因子 2 近似补偿 ReLU 的半轴门控。" />

<MlQuiz kind="初始化匹配" question="Xavier 初始化与 He 初始化的区别应如何理解？" :options="['只是两个软件包的不同名字', '它们依据不同激活对信号二阶矩的影响选择尺度', 'He 一定适用于所有激活', 'Xavier 会消除所有梯度消失']" :answer="1" explanation="初始化是激活、fan-in/fan-out 和传播假设共同决定的起点，不是无条件保证。" />

## 12. 输出偏置可以编码训练先验

若二分类训练集正类比例为 $\pi$，并希望初始模型在零特征贡献时输出该比例，可设置

$$
b_{\text{out}}=\log\frac{\pi}{1-\pi}.
$$

因为 $\sigma(b_{\text{out}})=\pi$。这能避免极度不平衡任务从 0.5 起步造成很大的初始误差，但部署基率若不同，不能把训练先验误当成现场概率；仍需校准和阈值验收。

多分类先验 $\pi_k$ 可用 $b_k=\log\pi_k+c$，任意公共常数 $c$ 会被 Softmax 抵消。

```python
import numpy as np

def logit(probability):
    probability = np.asarray(probability, dtype=float)
    return np.log(probability) - np.log1p(-probability)

def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))

binary_prior = 0.08
binary_bias = logit(binary_prior)
assert np.isclose(sigmoid(binary_bias), binary_prior)

class_prior = np.array([0.70, 0.20, 0.10])
class_bias = np.log(class_prior)
softmax_prior = np.exp(class_bias - class_bias.max())
softmax_prior /= softmax_prior.sum()
assert np.allclose(softmax_prior, class_prior)
```

## 复杂度与工程边界：参数、MAC 和激活内存

对于层宽

$$
d_0\rightarrow d_1\rightarrow\cdots\rightarrow d_L,
$$

含偏置参数量为

$$
N_{\text{param}}=\sum_{\ell=1}^{L}(d_{\ell-1}+1)d_\ell.
$$

单样本线性层的乘加次数近似为

$$
N_{\text{MAC}}=\sum_{\ell=1}^{L}d_{\ell-1}d_\ell.
$$

训练激活内存不仅是输出。粗略下界至少包含 batch 内各层 activation、pre-activation 或 mask，以及反向梯度；优化器还可能为每个参数保存一到两个状态。部署前必须在真实 batch、dtype、运行时和硬件上测量峰值。

```python
import numpy as np

def mlp_budget(layer_widths, batch_size, bytes_per_value=4):
    parameter_count = sum(
        (fan_in + 1) * fan_out
        for fan_in, fan_out in zip(layer_widths[:-1], layer_widths[1:])
    )
    mac_per_sample = sum(
        fan_in * fan_out
        for fan_in, fan_out in zip(layer_widths[:-1], layer_widths[1:])
    )
    saved_activation_bytes = batch_size * sum(layer_widths[1:]) * bytes_per_value
    return parameter_count, mac_per_sample, saved_activation_bytes

parameters, macs, activation_bytes = mlp_budget([8, 64, 32, 1], batch_size=256)
assert parameters == 2689
assert macs == 2592
assert activation_bytes == 256 * (64 + 32 + 1) * 4
print("参数：", parameters, "单样本 MAC：", macs, "最低激活字节：", activation_bytes)
```

<MlQuiz kind="部署预算" question="把隐藏宽度整体翻倍，为什么计算量可能不只翻倍？" :options="['因为标签数量变化', '相邻两个隐藏层的矩阵乘法同时扩大输入和输出维，成本近似按宽度平方增长', '因为 Sigmoid 自动增加样本', '因为偏置必须排序']" :answer="1" explanation="隐藏层到隐藏层的 MAC 为 d_in×d_out；两边同时翻倍时该项约变为四倍。" />

## 数值验证：前向与接口不变量

训练循环开始前，先把可自动检查的合同写成断言：

1. 输入、参数、logits、loss 全部有限；
2. 批量维和类别维满足统一约定；
3. 概率位于合法区间，Softmax 行和接近 1；
4. BCE 和交叉熵直接接收 logits；
5. 训练标准化统计不读取验证和测试批次；
6. 固定随机种子可复现实验，但不同种子也要做稳健性检查；
7. 预测动作、原始 logit、概率、阈值版本分开记录。

```python
import numpy as np

def stable_softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)

rng = np.random.default_rng(1305)
X = rng.normal(size=(128, 2))
W1 = rng.normal(scale=1.0, size=(2, 16))
b1 = np.zeros(16)
W2 = rng.normal(scale=np.sqrt(2 / 16), size=(16, 3))
b2 = np.log(np.array([0.7, 0.2, 0.1]))

Z1 = X @ W1 + b1
H1 = np.maximum(Z1, 0.0)
logits = H1 @ W2 + b2
probability = stable_softmax(logits)

assert np.all(np.isfinite(logits))
assert probability.shape == (128, 3)
assert np.all((0.0 <= probability) & (probability <= 1.0))
assert np.allclose(probability.sum(axis=1), 1.0)
```

## 15. 从教学构造到工业训练

手工四门网络给出了可解释基线，但真实系统不会只做一次前向。训练和验收至少要覆盖下列链条：

### 15.1 数据与切分

- 观测单元、时间戳、分组键和标签版本可追溯；
- 标准化、缺失值填补和类别权重只在训练数据拟合；
- 训练、验证、测试按部署层级隔离；
- 争议标签不被强制压成确定真值。

### 15.2 训练

- 从简单线性模型和手工规则建立下限；
- 报告多随机种子分布，而不是只挑最好一次；
- 监控训练/验证 loss、梯度范数、激活 RMS、零值或饱和比例；
- 用早停和明确的选择指标冻结 checkpoint；
- 超参数搜索预算与测试集严格隔离。

### 15.3 验收

仅有 accuracy 不够。复核系统至少报告：

| 指标 | 回答的问题 |
|---|---|
| recall / 漏检率 | 有风险的焊点有多少被放过 |
| precision / 复核命中率 | 人工复核产能是否被无效告警占满 |
| PR 曲线 | 阈值变化下产能与漏检如何交换 |
| calibration | 0.8 概率是否真对应近似 80% 频率 |
| 分组指标 | 设备、材料、班次是否出现性能洼地 |
| abstention rate | 数据异常或低置信度时有多少样本降级 |

### 15.4 部署

- 固化特征转换、权重、阈值和运行时版本；
- 量测真实 p50/p95/p99 延迟与峰值内存；
- 对 NaN、越界、缺传感器和版本不匹配 fail closed；
- 保留规则或人工通道作为可回退路径；
- 同时监控输入漂移、表示漂移、输出基率和延迟。

<MlQuiz kind="工业验收" question="一个 MLP 在随机样本切分上达到 99% accuracy，为什么仍不能证明可部署？" :options="['因为神经网络永远不能部署', '还缺分组外验证、标签证据、成本敏感指标、校准、延迟和失效降级测试', '只需增加一层', '只需把 accuracy 四舍五入']" :answer="1" explanation="部署成功标准跨越数据、统计、计算和运行时合同，单个随机切分指标无法覆盖。" />

## 16. 常见失败模式与定位顺序

| 现象 | 优先检查 | 机制 | 处置 |
|---|---|---|---|
| loss 一开始就是 NaN | 输入、logits、loss 实现 | 溢出、非法值或 $\log0$ | 用 logits 稳定损失，逐层检查有限性 |
| 训练不下降 | 标签、shape、学习率、梯度 | 广播错误、符号错误或步长不合适 | 先在极小数据过拟合，再扩大 |
| 所有 ReLU 为 0 | pre-activation 分布 | 偏置或大步长把单元推入负半轴 | 检查尺度、初始化与学习率 |
| Sigmoid/Tanh 饱和 | 激活直方图与导数 | 输入绝对值过大 | 标准化、调整初始化或激活 |
| 训练好、验证差 | 分组切分和容量 | 泄漏、过拟合或分布偏移 | 修切分、正则化、增数据 |
| 概率极端但错误多 | calibration curve | 似然拟合不等于校准 | 温度缩放并重验阈值 |
| 种子间差异很大 | 多次训练分布 | 非凸优化和小样本不稳定 | 报告分布、简化模型、增数据 |
| 离线正常、线上漂移 | 特征版本和监控 | 预处理偏差或输入迁移 | 固化流水线、影子运行、回滚 |

### 最小定位顺序

1. 在 8–32 个样本上关闭正则化，确认能把训练 loss 压低；
2. 打印每层 shape、有限性、均值、RMS、零值/饱和率；
3. 对一个参数做有限差分梯度检查；
4. 比较线性基线、手工四门网络和学习型 MLP；
5. 再讨论深度、宽度、优化器和正则化。

跳过前四步直接增加层数，通常只会把接口错误藏得更深。

## 17. 本章可执行检查清单

### 表示

- [ ] 我能把任意深的无激活仿射网络折叠成一层。
- [ ] 我能解释 XOR 的线性不可能性。
- [ ] 我能写出四个 ReLU 门及其输出权重。
- [ ] 我能说明 ReLU 网络在固定激活区域内为何仿射。

### 概率与数值

- [ ] 输出头与回归、二分类、单标签或多标签事件匹配。
- [ ] BCE/Softmax 交叉熵直接从 logits 稳定计算。
- [ ] 输出偏置只编码训练先验，不被误作部署校准。
- [ ] shape、dtype、有限性和概率归一化有自动断言。

### 初始化与训练

- [ ] 同层神经元没有完全相同初始化。
- [ ] Xavier/He 的尺度与激活和 fan-in/fan-out 匹配。
- [ ] 监控每层激活 RMS、零值/饱和比例与梯度范数。
- [ ] 训练报告覆盖多个随机种子。

### 工业闭环

- [ ] 标签来自独立质量证据。
- [ ] 切分按未来部署层级隔离。
- [ ] 阈值由漏检成本与复核产能共同决定。
- [ ] 模型、预处理、阈值和运行时可追溯并可回滚。

## 18. 练习与最小实现

### 练习 1：证明线性折叠

对含 $L$ 个仿射层的网络，用归纳法写出折叠后的

$$
\widetilde W=W_1W_2\cdots W_L
$$

以及偏置递推。再说明为什么加入任意一个非线性后，一般不能继续这样合并。

### 练习 2：NumPy 前向

实现任意层宽列表的 MLP 前向函数。要求：

- 每层保存 shape 合同；
- 隐藏层支持 ReLU 或 Tanh；
- 最后一层只输出 logits；
- 不依赖深度写重复分支；
- 对不匹配的权重形状立即报错。

### 练习 3：画二维决策边界

在本章批次数据上比较：

1. 线性 logistic regression；
2. 手工四门 ReLU 网络；
3. 学习得到的 $2\rightarrow8\rightarrow1$ MLP。

除散点和边界外，必须画验证批次上的概率等高线，并报告不同阈值下的复核率与漏检率。

### 练习 4：初始化消融

对 30 层等宽网络比较固定标准差、Xavier 和 He；分别使用 ReLU 与 Tanh。记录每层：

- pre-activation RMS；
- activation RMS；
- ReLU 零值比例或 Tanh 饱和比例；
- 反向梯度 RMS。

最后解释前向稳定是否必然意味着反向也稳定。

```python
# TODO：实现任意层宽 MLP 前向，并为每一层加入 shape 与有限性断言。
#
# def forward_mlp(X, parameters, activation="relu"):
#     ...
#
# 要求至少通过：
# 1. [2, 8, 4, 1] 的批量形状测试；
# 2. ReLU 与 Tanh 分支测试；
# 3. 一个错误权重形状测试；
# 4. 极端 logits 仍有限的测试。
```

## 19. 本章结论

1. 神经元是仿射投影与非线性门，不是神秘的生物类比。
2. 没有激活函数时，任意深网络在函数层面仍是一个仿射模型。
3. 四个 ReLU 单元可以精确构造 XOR 型双传感器一致性规则。
4. ReLU 网络通过激活模式切换得到分段仿射决策面。
5. 万能逼近只给表示存在性，不保证训练、泛化、校准或部署。
6. 输出头和损失必须对应概率事件；稳定实现应直接使用 logits。
7. 随机初始化打破对称，Xavier 与 He 用二阶矩传播控制尺度。
8. 工业 MLP 还必须冻结标签、分组切分、阈值、计算预算、监控和降级合同。

## 来源与许可

本章关于神经元、全连接层、向量化、激活函数和线性激活折叠的推导，参考 CS229 Lecture Notes 的神经网络章节（源 PDF 第 85–95 页，§7.2–7.3）。XOR、可微 MLP、输出概率模型、激活函数和万能逼近边界，参考 Probabilistic Machine Learning: An Introduction（源 PDF 第 454–465 页，§13.1–13.2.5）。隐藏层对称性、ReLU、输出单元与损失，以及 Xavier/He 初始化，参考 CS189/289A 讲义（源 PDF 第 95–104、113 页）。一隐层组件、非线性决策边界和宽度直觉，参考 CMU 10-701 Lecture 11（源 PDF 第 45–73 页）。

激光焊接双传感器数据、四象限规则、批次协议和部署指标是本章为教学独立构造的案例，不来自上述资料，也不应被视为真实工艺规范。

<MlLessonProgress
  lesson-id="13"
  next-href="/courses/machine-learning/14-backpropagation"
  next-label="第 14 章：反向传播为什么高效"
/>
