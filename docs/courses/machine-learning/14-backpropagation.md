---
course: machine-learning
edition: industrial
title: 反向传播
description: 以激光焊接复核网络的梯度审计为贯穿案例，从局部线性化、计算图、VJP 和矩阵链式法则推导两层 MLP 反向传播、反向模式自动微分、广播归约与数值梯度检查。
outline: [2, 3]
---

<MlLessonHero
  number="14"
  title="反向传播"
  question="一个参数可能沿多条路径影响最终损失；怎样只复用局部导数和前向缓存，就在线性于计算图规模的时间内得到全部参数梯度？"
  duration="建议 560–680 分钟"
  :outcomes="[
    '从局部线性化和多元链式法则手算含分支计算图，解释反向伴随量、路径累加和拓扑顺序',
    '用 VJP 与矩阵微分逐层推导两层 MLP 的 dW、db 和输入梯度，并处理 batch mean 与 broadcasting',
    '实现最小反向模式自动微分和 NumPy MLP 反向，用有限差分及可选 PyTorch Autograd 对照建立梯度验收门禁'
  ]"
/>

第 13 章证明了四个 ReLU 单元足以表示双传感器一致性规则，但“存在一组参数”还不是学习算法。训练必须回答：

$$
\boxed{
\text{当前损失改变 1 个微小单位时，每个参数应向哪个方向移动、移动多强？}
}
$$

若参数量是千万级，逐个参数做有限差分需要千万次以上前向，无法用于训练。反向传播（backpropagation）利用计算图结构，把最终标量损失的敏感度从输出向输入传播；它不是新的求导规则，而是链式法则、动态规划和局部算子接口的组合。

本章坚持四个边界：

1. 梯度是当前点的局部一阶敏感度，不是大步更新后的因果保证；
2. 反向传播计算梯度，优化器决定怎样使用梯度；
3. 自动微分精确执行已记录程序的导数，不判断模型、标签或业务目标是否正确；
4. 损失下降不能证明梯度实现正确，必须有独立数值审计。

## 贯穿案例：激光焊接复核网络的梯度审计

沿用第 13 章的双传感器案例。每个焊点有热像残差 $x_1$ 和光强残差 $x_2$；两路方向相反的教学样本标为送复核。现在训练一个

$$
2\rightarrow8\rightarrow2
$$

的 ReLU MLP。输出两个 logits，分别对应“自动放行”和“送人工复核”。本章的重点不是刷新准确率，而是建立一份可追溯的梯度账本：

| 层级 | 必须记录 | 主要故障 |
|---|---|---|
| 输入与标签 | 批次、shape、dtype、有限性、标签版本 | 泄漏、错位、错误代理标签 |
| 前向节点 | pre-activation、activation、logits、loss | 溢出、错误广播、失活 |
| 反向节点 | 上游梯度、局部 VJP、累加次数 | 漏乘、覆盖、错误归约 |
| 参数梯度 | shape、范数、有限性、相对差分误差 | NaN、消失、爆炸、符号错误 |
| 优化步骤 | loss reduction、学习率、累积步数、裁剪顺序 | 梯度尺度漂移、重复更新 |

教学标签仍是合成规则，不是真实焊接质量结论。真实系统必须由独立金相、无损检测或经过校准的复判流程提供标签。

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
raw_features = corner_sign * corner_radius
raw_features += batch_shift[batch_id]
raw_features += rng.normal(0.0, 0.10, size=raw_features.shape)
review_label = (raw_features[:, 0] * raw_features[:, 1] < 0).astype(np.int64)

batch_order = rng.permutation(batch_count)
train_batches = batch_order[:32]
validation_batches = batch_order[32:40]
test_batches = batch_order[40:]
train_mask = np.isin(batch_id, train_batches)
validation_mask = np.isin(batch_id, validation_batches)
test_mask = np.isin(batch_id, test_batches)

train_mean = raw_features[train_mask].mean(axis=0)
train_scale = raw_features[train_mask].std(axis=0, ddof=0)
features = (raw_features - train_mean) / train_scale
X_train, y_train = features[train_mask], review_label[train_mask]
X_validation, y_validation = features[validation_mask], review_label[validation_mask]
X_test, y_test = features[test_mask], review_label[test_mask]

assert X_train.shape == (768, 2)
assert X_validation.shape == X_test.shape == (192, 2)
assert set(train_batches).isdisjoint(validation_batches)
assert set(train_batches).isdisjoint(test_batches)
assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-12)
print("训练/验证/测试：", len(X_train), len(X_validation), len(X_test))
print("教学复核率：", round(review_label.mean(), 4))
```

<MlQuiz kind="职责边界" question="自动微分返回的梯度全部通过数值检查，能否证明焊接标签和部署目标正确？" :options="['可以，梯度正确等于业务正确', '不可以；它只证明程序局部导数与前向目标一致，不能验证标签来源和业务目标', '可以，只要参数很多', '只需再检查一次训练准确率']" :answer="1" explanation="梯度审计验证实现，不验证问题定义；标签合同和行动代价仍需独立证据。" />

## 1. 导数是局部线性模型

标量函数 $f:\mathbb R\to\mathbb R$ 在 $x$ 附近满足

$$
f(x+\Delta x)
=f(x)+f'(x)\Delta x+o(|\Delta x|).
$$

$f'(x)$ 是当前点的局部增益。它回答无穷小扰动怎样传到输出；若更新步很大、跨过 ReLU 折点或进入另一数据区域，一阶近似可能失效。

对多变量标量函数 $L(x_1,\ldots,x_d)$，

$$
L(x+\Delta x)
\approx L(x)+\nabla_xL^T\Delta x.
$$

梯度是使局部增量最大的方向；负梯度是局部最速下降方向。这解释了为什么第 02 章的梯度下降需要小步长、线搜索或自适应尺度。

### 一个带分支的标量图

定义

$$
a=xy,\qquad b=a+x,\qquad e=b-t,\qquad L=\frac12e^2.
$$

使用伴随量记号

$$
\bar v=\frac{\partial L}{\partial v}.
$$

从 $\bar L=1$ 开始：

$$
\bar e=e,\qquad
\bar b=\bar e,\qquad
\bar a=\bar b.
$$

$x$ 沿两条路径影响 $L$：

$$
\boxed{
\bar x
=\underbrace{\bar a\,y}_{x\to a\to b\to e\to L}
+\underbrace{\bar b}_{x\to b\to e\to L}
=e(y+1).
}
$$

另外

$$
\bar y=\bar a x=ex,\qquad
\bar t=-\bar e=-e.
$$

```python
import numpy as np

scalar_x, scalar_y, scalar_target = 1.2, -1.5, 0.4
scalar_a = scalar_x * scalar_y
scalar_b = scalar_a + scalar_x
scalar_error = scalar_b - scalar_target
scalar_loss = 0.5 * scalar_error ** 2

grad_x = scalar_error * (scalar_y + 1.0)
grad_y = scalar_error * scalar_x
grad_target = -scalar_error

def scalar_loss_at(x_value, y_value, target_value):
    return 0.5 * (x_value * y_value + x_value - target_value) ** 2

epsilon = 1e-6
numeric_grad_x = (
    scalar_loss_at(scalar_x + epsilon, scalar_y, scalar_target)
    - scalar_loss_at(scalar_x - epsilon, scalar_y, scalar_target)
) / (2 * epsilon)

assert np.isclose(scalar_loss, 0.5)
assert np.isclose(grad_x, 0.5)
assert np.isclose(grad_y, -1.2)
assert np.isclose(grad_target, 1.0)
assert np.isclose(grad_x, numeric_grad_x, atol=1e-9)
```

<div class="ml-stop">
  <strong>停下来推一推 1</strong>
  <p>令 x=2、y=3、t=1。先做完整前向，再分别写出 x 经乘法支路和跳接支路返回的贡献。若只保留后一条，最终梯度错了多少？</p>
</div>

<MlQuiz kind="局部敏感度" question="梯度 ∂L/∂x=4 最准确的含义是什么？" :options="['x 增加任意大小都会让 L 精确增加四倍', '在当前点附近，x 的微小增量 Δx 使 L 约增加 4Δx', 'x 必须被设成 4', '损失一定为 4']" :answer="1" explanation="导数是局部一阶系数；有限步更新还受高阶项和区域切换影响。" />

## 2. 计算图是有向无环依赖关系

计算图把程序拆成局部算子。节点保存值，边表示依赖。一个合法的前馈反向图需要：

- 前向按拓扑序：所有父节点先于子节点；
- 反向按逆拓扑序：节点收到所有下游贡献后，再把 VJP 分发给父节点；
- 一个变量被多次使用时，伴随量初始化为 0，并使用累加；
- 常量可以参与前向，但不一定需要保存梯度；
- 控制流实际执行了哪条路径，动态图只记录那条路径。

若图有环，必须先按时间展开或定义隐式微分；普通一次逆拓扑遍历不能直接处理循环依赖。

### 最小标量反向模式引擎

```python
class Value:
    def __init__(self, data, parents=(), name=""):
        self.data = float(data)
        self.parents = list(parents)
        self.name = name
        self.grad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, ((self, 1.0), (other, 1.0)))

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data * other.data,
            ((self, other.data), (other, self.data)),
        )

    __rmul__ = __mul__

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, other):
        return self + (-other)

    def backward(self):
        topological = []
        visited = set()

        def build(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent, _ in node.parents:
                build(parent)
            topological.append(node)

        build(self)
        for node in topological:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topological):
            for parent, local_derivative in node.parents:
                parent.grad += node.grad * local_derivative

x_node = Value(1.2, name="x")
y_node = Value(-1.5, name="y")
t_node = Value(0.4, name="target")
a_node = x_node * y_node
b_node = a_node + x_node
error_node = b_node - t_node
loss_node = 0.5 * error_node * error_node
loss_node.backward()

assert np.isclose(loss_node.data, scalar_loss)
assert np.isclose(x_node.grad, grad_x)
assert np.isclose(y_node.grad, grad_y)
assert np.isclose(t_node.grad, grad_target)
```

这里的 parent.grad += ... 是算法正确性的核心，不是语法偏好。若改成赋值，同一父节点后到的路径会覆盖先到贡献。

<ComputationGraphTraceLab />

<MlQuiz kind="分支累加" question="变量 x 沿三条路径影响标量损失，反向到 x 时应怎样处理？" :options="['只保留最后一条', '将三条路径的链式贡献相加', '将三条完整梯度相乘', '取绝对值最大的路径']" :answer="1" explanation="多元链式法则对所有依赖路径求和；覆盖任一路径都会改变总导数。" />

## 3. 向量函数：Jacobian、JVP 与 VJP

设

$$
y=f(x),\qquad x\in\mathbb R^n,\ y\in\mathbb R^m.
$$

采用 numerator layout：

$$
J_f(x)=\frac{\partial y}{\partial x^T}\in\mathbb R^{m\times n},
\qquad
[J_f]_{ij}=\frac{\partial y_i}{\partial x_j}.
$$

### 3.1 JVP：输入方向向前推

给定输入切向量 $v\in\mathbb R^n$，

$$
J_f(x)v
$$

是沿 $v$ 的输出方向导数。前向模式自动微分在每个节点同时传播值和切向量。

### 3.2 VJP：输出伴随量向后拉

给定输出伴随量 $u\in\mathbb R^m$，

$$
J_f(x)^Tu
$$

把最终标量

$$
L=u^Tf(x)
$$

对输出的敏感度拉回输入。深度学习的最终 loss 是标量，所以一次反向 VJP 就能得到全部输入或参数梯度。

```python
import numpy as np

rng = np.random.default_rng(1401)
A = rng.normal(size=(4, 3))
x_vector = rng.normal(size=3)
output_vector = np.tanh(A @ x_vector)
jacobian = np.diag(1.0 - output_vector ** 2) @ A

input_direction = rng.normal(size=3)
output_cotangent = rng.normal(size=4)
jvp = jacobian @ input_direction
vjp = jacobian.T @ output_cotangent

eps = 1e-6
numeric_jvp = (
    np.tanh(A @ (x_vector + eps * input_direction))
    - np.tanh(A @ (x_vector - eps * input_direction))
) / (2 * eps)

def scalarized_output(value):
    return output_cotangent @ np.tanh(A @ value)

numeric_vjp = np.array([
    (
        scalarized_output(x_vector + eps * np.eye(3)[index])
        - scalarized_output(x_vector - eps * np.eye(3)[index])
    ) / (2 * eps)
    for index in range(3)
])

assert np.allclose(jvp, numeric_jvp, atol=1e-9)
assert np.allclose(vjp, numeric_vjp, atol=1e-9)
```

<MlQuiz kind="VJP" question="标量损失 L 依赖向量 y=f(x)，已知上游梯度 ∇yL。反向真正需要什么？" :options="['完整 Jacobian 的所有元素', 'J_f(x)ᵀ∇yL 这个 VJP', 'Jacobian 的行列式', 'J_f(x) 的逆矩阵']" :answer="1" explanation="反向只需把一个输出伴随量拉回输入，无需构造巨大 Jacobian。" />

## 4. 反向模式为什么适合神经网络训练

设函数 $f:\mathbb R^n\to\mathbb R^m$。完整 Jacobian 有 $mn$ 个元素。

- 前向模式一次得到一个输入方向的 JVP；构造全部 Jacobian 通常需 $n$ 次；
- 反向模式一次得到一个输出方向的 VJP；构造全部 Jacobian 通常需 $m$ 次；
- 训练中 $n$ 是海量参数，$m=1$ 是标量 loss，因此一次反向最合适。

若任务是用少量输入控制量影响海量输出，例如求某个方向导数，前向模式可能更合适。模式选择取决于输入/输出维度，不是“反向永远更快”。

对一个含 $E$ 条有效边的标量计算图，只要每个局部 VJP 与对应前向算子同阶，前向加反向通常是 $O(E)$；逐参数重复链式法则会反复经过共享子图，可能达到 $O(E^2)$ 或更差。

```python
parameter_count = 1_000_000
scalar_output_count = 1
graph_operation_count = 8_000_000

forward_mode_sweeps_for_full_gradient = parameter_count
reverse_mode_sweeps_for_full_gradient = scalar_output_count
reverse_operation_order = 2 * graph_operation_count

assert reverse_mode_sweeps_for_full_gradient == 1
assert forward_mode_sweeps_for_full_gradient == 1_000_000
assert reverse_operation_order == 16_000_000
```

<MlQuiz kind="模式选择" question="一个网络有一千万个参数和一个标量 loss，为什么反向模式合适？" :options="['因为参数越多 Jacobian 越容易求逆', '一次输出方向的 VJP 可同时得到所有参数对标量 loss 的梯度', '因为反向不需要前向', '因为它保证全局最优']" :answer="1" explanation="标量输出使反向只需一次伴随传播；前向模式若求完整梯度需按输入方向重复。" />

## 5. 局部算子的反向合同

每个算子只需知道：

1. 前向输入或能恢复局部导数的 cache；
2. 来自所有下游的上游梯度；
3. 怎样计算对每个输入和参数的 VJP；
4. 返回梯度必须与对应输入形状一致。

| 前向 | 给定上游 $g$ | 反向到输入 |
|---|---|---|
| $z=x+y$ | $g=\bar z$ | $\bar x{+}=g,\ \bar y{+}=g$ |
| $z=xy$ | $g=\bar z$ | $\bar x{+}=gy,\ \bar y{+}=gx$ |
| $z=\max(x,y)$ | $g=\bar z$ | 梯度路由到获胜输入；相等处需约定 |
| $z=\operatorname{ReLU}(x)$ | $g=\bar z$ | $\bar x{+}=g\mathbf1(x>0)$ |
| $Z=XW$ | $G=\bar Z$ | $\bar X{+}=GW^T,\ \bar W{+}=X^TG$ |
| $Z=X+b$ | $G=\bar Z$ | $\bar X{+}=G,\ \bar b{+}=\operatorname{sum\_to\_shape}(G)$ |

“加法门复制梯度、乘法门交换输入再乘、max 门路由梯度”只是局部规则。最终梯度仍要乘上游量，并在分叉处累加。

## 6. 矩阵乘法梯度不靠背口诀

设

$$
Z=XW,\qquad
X\in\mathbb R^{B\times d},\
W\in\mathbb R^{d\times m},\
Z\in\mathbb R^{B\times m}.
$$

上游梯度为 $G=\partial L/\partial Z$。用微分：

$$
dZ=dX\,W+X\,dW.
$$

标量损失微分写成 Frobenius 内积：

$$
dL=\langle G,dZ\rangle_F
=\operatorname{tr}(G^TdZ).
$$

对 $dX$ 项：

$$
\operatorname{tr}(G^TdXW)
=\operatorname{tr}((GW^T)^TdX),
$$

因此

$$
\boxed{\frac{\partial L}{\partial X}=GW^T.}
$$

对 $dW$ 项：

$$
\operatorname{tr}(G^TXdW)
=\operatorname{tr}((X^TG)^TdW),
$$

所以

$$
\boxed{\frac{\partial L}{\partial W}=X^TG.}
$$

两个结果的 shape 必须分别回到 $X$ 和 $W$。

```python
import numpy as np

rng = np.random.default_rng(1402)
matrix_X = rng.normal(size=(7, 3))
matrix_W = rng.normal(size=(3, 5))
matrix_bias = rng.normal(size=5)
upstream = rng.normal(size=(7, 5))

matrix_Z = matrix_X @ matrix_W + matrix_bias
grad_matrix_X = upstream @ matrix_W.T
grad_matrix_W = matrix_X.T @ upstream
grad_matrix_bias = upstream.sum(axis=0)

def scalarized_linear():
    return np.sum((matrix_X @ matrix_W + matrix_bias) * upstream)

probe_index = (1, 3)
old_value = matrix_W[probe_index]
matrix_W[probe_index] = old_value + 1e-6
plus = scalarized_linear()
matrix_W[probe_index] = old_value - 1e-6
minus = scalarized_linear()
matrix_W[probe_index] = old_value
numeric = (plus - minus) / (2e-6)

assert grad_matrix_X.shape == matrix_X.shape
assert grad_matrix_W.shape == matrix_W.shape
assert grad_matrix_bias.shape == matrix_bias.shape
assert np.isclose(numeric, grad_matrix_W[probe_index], atol=1e-9)
```

<MlQuiz kind="矩阵反向" question="Z=XW，已知 G=∂L/∂Z。∂L/∂W 是什么？" :options="['GWᵀ', 'XᵀG', 'GXᵀ', 'WᵀG']" :answer="1" explanation="dL=tr(GᵀX dW)=tr((XᵀG)ᵀdW)，且结果形状必须与 W 相同。" />

## 7. Softmax 与交叉熵合并后的起点

对一个样本 logits $z\in\mathbb R^K$：

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},
\qquad
\ell=-\sum_ky_k\log p_k.
$$

第 10 章已证明

$$
\frac{\partial\ell}{\partial z}=p-y.
$$

对 batch mean：

$$
L=\frac1B\sum_{i=1}^{B}\ell_i
\quad\Longrightarrow\quad
\boxed{G_2=\frac{P-Y}{B}.}
$$

除以 $B$ 不是“为了数值小一点”，而是 loss 定义的一部分。若训练从 mean 改为 sum 而学习率不变，梯度会随 batch size 放大。

<MlQuiz kind="损失归约" question="交叉熵按 batch mean 定义时，为什么 logits 梯度是 (P−Y)/B？" :options="['Softmax 要求除以 B', '平均损失前的 1/B 通过线性求导保留下来', '为了让概率和为 B', '只有偏置需要除以 B']" :answer="1" explanation="reduction 是目标函数合同；改变 mean/sum 会改变全部参数梯度尺度。" />

## 8. 两层 MLP 的完整矩阵反向

前向：

$$
\begin{aligned}
Z_1&=XW_1+\mathbf1b_1^T,\\
H&=\operatorname{ReLU}(Z_1),\\
Z_2&=HW_2+\mathbf1b_2^T,\\
P&=\operatorname{softmax}(Z_2),\\
L&=-\frac1B\sum_{i,k}Y_{ik}\log P_{ik}.
\end{aligned}
$$

设 $X:(B,d)$、$W_1:(d,m)$、$W_2:(m,K)$。从输出向输入逆序：

$$
\begin{aligned}
G_2&=(P-Y)/B,\\
\nabla_{W_2}L&=H^TG_2,\\
\nabla_{b_2}L&=\sum_iG_{2,i:},\\
\bar H&=G_2W_2^T,\\
G_1&=\bar H\odot\mathbf1(Z_1>0),\\
\nabla_{W_1}L&=X^TG_1,\\
\nabla_{b_1}L&=\sum_iG_{1,i:},\\
\bar X&=G_1W_1^T.
\end{aligned}
$$

### 形状账本

| 量 | shape | 由谁产生 |
|---|---:|---|
| $G_2$ | $(B,K)$ | Softmax-cross-entropy VJP |
| $\nabla W_2$ | $(m,K)$ | $H^TG_2$ |
| $\nabla b_2$ | $(K,)$ | 沿 batch 归约 |
| $\bar H$ | $(B,m)$ | $G_2W_2^T$ |
| $G_1$ | $(B,m)$ | ReLU mask |
| $\nabla W_1$ | $(d,m)$ | $X^TG_1$ |
| $\nabla b_1$ | $(m,)$ | 沿 batch 归约 |
| $\bar X$ | $(B,d)$ | $G_1W_1^T$ |

形状正确是必要条件，不是充分条件。例如 $W_2^TG_2^T$ 转置后也可能凑出某个合法 shape，但链式方向可能错；还需数值检查。

```python
import numpy as np

def stable_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)

def mlp_forward(X, parameters):
    Z1 = X @ parameters["W1"] + parameters["b1"]
    H = np.maximum(Z1, 0.0)
    logits = H @ parameters["W2"] + parameters["b2"]
    probability = stable_softmax(logits)
    cache = {"X": X, "Z1": Z1, "H": H, "probability": probability}
    return probability, cache

def mean_cross_entropy(probability, target):
    return -np.log(probability[np.arange(len(target)), target]).mean()

def mlp_backward(target, parameters, cache):
    X = cache["X"]
    Z1 = cache["Z1"]
    H = cache["H"]
    probability = cache["probability"]
    batch = len(X)
    one_hot = np.eye(probability.shape[1])[target]

    G2 = (probability - one_hot) / batch
    dW2 = H.T @ G2
    db2 = G2.sum(axis=0)
    dH = G2 @ parameters["W2"].T
    G1 = dH * (Z1 > 0.0)
    dW1 = X.T @ G1
    db1 = G1.sum(axis=0)
    dX = G1 @ parameters["W1"].T
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}, dX

rng = np.random.default_rng(1403)
parameters = {
    "W1": rng.normal(scale=1.0, size=(2, 8)),
    "b1": rng.normal(scale=0.03, size=8),
    "W2": rng.normal(scale=np.sqrt(2 / 8), size=(8, 2)),
    "b2": np.zeros(2),
}

probability, mlp_cache = mlp_forward(X_train, parameters)
initial_loss = mean_cross_entropy(probability, y_train)
gradients, input_gradient = mlp_backward(y_train, parameters, mlp_cache)

assert probability.shape == (768, 2)
assert gradients["W1"].shape == parameters["W1"].shape
assert gradients["b1"].shape == parameters["b1"].shape
assert gradients["W2"].shape == parameters["W2"].shape
assert gradients["b2"].shape == parameters["b2"].shape
assert input_gradient.shape == X_train.shape
assert all(np.all(np.isfinite(value)) for value in gradients.values())
```

<div class="ml-stop">
  <strong>停下来推一推 2</strong>
  <p>设 B=32、d=2、m=8、K=2。逐项写出 G₂、dW₂、db₂、dH、G₁、dW₁、db₁、dX 的 shape；再指出哪些错误可能通过 shape 检查却仍是错误导数。</p>
</div>

<MlQuiz kind="反向顺序" question="为什么必须先得到 G₂，才能计算 dW₂ 和 dH？" :options="['因为 W₂ 必须可逆', 'G₂ 是最终损失对该层输出的上游敏感度，局部 VJP 需要它', '因为 ReLU 只接受整数', '因为 batch 不能并行']" :answer="1" explanation="每个局部 backward 都消费来自下游的伴随量；逆拓扑顺序保证该量已完整累加。" />

## 9. 广播反向：复制的逆操作是求和

前向

$$
Z_{ik}=(HW)_{ik}+b_k
$$

把同一个 $b_k$ 复制给 $B$ 个样本。于是

$$
\frac{\partial L}{\partial b_k}
=\sum_{i=1}^B\frac{\partial L}{\partial Z_{ik}}.
$$

一般规则不是“见广播就固定沿 axis=0 求和”，而是把上游梯度归约回原输入 shape：

1. 去掉前向新增的领先维；
2. 对原 shape 中大小为 1、前向被扩展的轴求和并保留维；
3. 最终 shape 必须与原输入完全一致。

```python
import numpy as np

def sum_to_shape(gradient, target_shape):
    result = np.asarray(gradient)
    while result.ndim > len(target_shape):
        result = result.sum(axis=0)
    for axis, target_size in enumerate(target_shape):
        if target_size == 1 and result.shape[axis] != 1:
            result = result.sum(axis=axis, keepdims=True)
    if result.shape != target_shape:
        raise ValueError((result.shape, target_shape))
    return result

upstream = np.arange(24.0).reshape(2, 4, 3)
bias_vector = np.zeros(3)
bias_row = np.zeros((1, 3))

vector_gradient = sum_to_shape(upstream, bias_vector.shape)
row_gradient = sum_to_shape(upstream, bias_row.shape)

assert vector_gradient.shape == (3,)
assert row_gradient.shape == (1, 3)
assert np.allclose(vector_gradient, upstream.sum(axis=(0, 1)))
assert np.allclose(row_gradient, upstream.sum(axis=(0, 1), keepdims=False)[None, :])
```

<BatchGradientReducerLab />

<MlQuiz kind="广播反向" question="偏置 shape 为 (1,m)，前向扩展成 (B,m)。反向返回偏置梯度的 shape 和操作是什么？" :options="['(B,m)，保持不变', '(1,m)，沿 batch 轴求和并 keepdims', '(m,B)，转置', '(m,)，任取一行']" :answer="1" explanation="所有复制位置依赖同一参数；求和恢复数值贡献，keepdims 恢复原 shape。" />

## 10. 参数共享也是计算图分叉

卷积核跨空间位置复用、RNN 权重跨时间复用、一个 embedding 被多个 token 索引，都是同一参数节点被多次使用。其梯度必须汇总所有使用位置。

简单例子：

$$
\hat y=wx_1+wx_2,\qquad
L=\frac12(\hat y-t)^2.
$$

若 $e=\hat y-t$，

$$
\frac{\partial L}{\partial w}
=ex_1+ex_2
=e(x_1+x_2).
$$

不能因为代码里只有一个变量 $w$ 就只返回一份贡献。

```python
shared_weight = 0.7
shared_x1, shared_x2, shared_target = 1.2, -0.4, 0.3
shared_prediction = shared_weight * shared_x1 + shared_weight * shared_x2
shared_error = shared_prediction - shared_target

path_one = shared_error * shared_x1
path_two = shared_error * shared_x2
shared_gradient = path_one + path_two

def shared_loss(weight):
    prediction = weight * shared_x1 + weight * shared_x2
    return 0.5 * (prediction - shared_target) ** 2

numeric_shared = (
    shared_loss(shared_weight + 1e-6)
    - shared_loss(shared_weight - 1e-6)
) / (2e-6)

assert np.isclose(shared_gradient, numeric_shared, atol=1e-10)
assert not np.isclose(path_two, numeric_shared)
```

<MlQuiz kind="参数共享" question="同一卷积核参数在 100 个空间位置被使用，反向时该参数梯度怎样得到？" :options="['随机选一个位置', '汇总 100 个使用位置的路径贡献', '把 100 个梯度相乘', '只使用最大激活位置']" :answer="1" explanation="参数共享对应计算图中一个参数节点通往多个算子，伴随量必须累加。" />

## 11. 反向模式自动微分做了什么

典型动态图框架在前向时建立 tape：

1. 记录实际执行的算子、输入引用和输出；
2. 保存 backward 所需 cache 或保存重算入口；
3. 标记哪些张量需要梯度；
4. 从标量 loss 注入 1；
5. 逆拓扑调用每个算子的 VJP；
6. 对共享父节点和叶参数累加梯度。

自动微分既不是符号微分，也不是有限差分：

| 方法 | 产生什么 | 主要代价 | 典型用途 |
|---|---|---|---|
| 符号微分 | 新的解析表达式 | 表达式可能膨胀 | 代数系统、闭式推导 |
| 有限差分 | 数值近似 | 每个坐标需额外前向 | 小规模梯度检查 |
| 自动微分 | 已执行程序在当前输入处的机器精度导数 | 记录图、cache 与反向计算 | 模型训练 |

动态图中的 if 分支只对本次走过的路径求导。离散索引、比较、取整和外部 I/O 一般不可微；框架不会自动发明合理替代梯度。

<MlQuiz kind="自动微分" question="反向模式自动微分的本质是什么？" :options="['把整个程序化成一个符号公式', '记录已执行计算图并按逆拓扑序复用局部 VJP', '随机扰动每个参数', '搜索最优参数']" :answer="1" explanation="它精确组合局部导数并累加路径，不负责优化，也不是有限差分近似。" />

## 12. 有限差分：独立验收解析反向

对参数坐标 $\theta_j$，中心差分：

$$
g_j^{\text{num}}
=\frac{L(\theta_j+\varepsilon)-L(\theta_j-\varepsilon)}
{2\varepsilon}
+O(\varepsilon^2).
$$

本章采用相对误差

$$
\operatorname{relerr}
=\frac{|g_j^{\text{ana}}-g_j^{\text{num}}|}
{\max(1,|g_j^{\text{ana}}|,|g_j^{\text{num}}|)}.
$$

分母至少为 1，避免两个接近 0 的正确梯度被微小绝对误差放大。也可采用以两者绝对值之和为分母的定义，但项目必须冻结一种口径。

### ε 为什么不是越小越好

- $\varepsilon$ 太大：高阶项导致截断误差；
- $\varepsilon$ 太小：两次接近的 loss 相减导致消去误差；
- float32 比 float64 更早进入舍入误差区；
- ReLU、max 等折点附近，中心差分跨越两侧，不能裁决单侧次梯度。

<GradientCheckMicroscopeLab />

### 对两层 MLP 抽样检查

```python
import numpy as np

def relative_gradient_error(analytic, numeric):
    return abs(analytic - numeric) / max(1.0, abs(analytic), abs(numeric))

gradient_checks = [
    ("W1", (0, 0)),
    ("W1", (1, 5)),
    ("b1", (3,)),
    ("W2", (2, 1)),
    ("W2", (6, 0)),
    ("b2", (1,)),
]
check_errors = []

for parameter_name, index in gradient_checks:
    parameter_array = parameters[parameter_name]
    original = parameter_array[index]
    parameter_array[index] = original + 1e-6
    plus_probability, _ = mlp_forward(X_train, parameters)
    plus_loss = mean_cross_entropy(plus_probability, y_train)
    parameter_array[index] = original - 1e-6
    minus_probability, _ = mlp_forward(X_train, parameters)
    minus_loss = mean_cross_entropy(minus_probability, y_train)
    parameter_array[index] = original

    numeric_gradient = (plus_loss - minus_loss) / (2e-6)
    analytic_gradient = gradients[parameter_name][index]
    check_errors.append(relative_gradient_error(analytic_gradient, numeric_gradient))

assert max(check_errors) < 1e-7
print("抽样梯度最大相对误差：", f"{max(check_errors):.3e}")
```

<div class="ml-stop">
  <strong>停下来推一推 3</strong>
  <p>为什么训练 loss 连续下降仍不能证明 backward 正确？构造一种“漏掉一条路径，但剩余方向仍暂时下降”的情形，并说明有限差分能发现什么。</p>
</div>

<MlQuiz kind="差分尺度" question="有限差分中的 ε 极小时，误差为什么可能重新上升？" :options="['模型会增加一层', '两个非常接近的浮点 loss 相减，舍入与消去误差占主导', '解析梯度会变成整数', 'batch size 会自动变化']" :answer="1" explanation="中心差分同时承受截断误差和浮点误差，通常存在中间最优尺度。" />

<MlQuiz kind="不可导点" question="在 ReLU 的 θ=0 处用对称中心差分检查框架采用的右侧或左侧次梯度，结论可靠吗？" :options="['可靠，中心差分一定精确', '不可靠；扰动跨越折点，得到的是两侧变化的混合', '可靠，只要用 float32', '只需把 ε 设成 1']" :answer="1" explanation="应把检查点移离折点，或只验证框架明确的单侧约定。" />

## 13. 与 PyTorch Autograd 做数值对照

核心推导和 NumPy 实现不能被框架替代；但成熟框架可作为第二实现交叉核对。对照必须满足：

- 同一组输入、标签和权重；
- 同一 dtype，优先 float64 做梯度审计；
- 相同 loss reduction；
- ReLU 折点样本移开；
- 参数布局与转置约定一致；
- 比较每个参数的最大绝对误差和相对误差。

下面代码需要可选 PyTorch 环境，仓库当前标准验证环境未安装 PyTorch，因此不将它伪装成已执行门禁，也不为一次对照新增大型依赖。

```python
# 不执行：可选 PyTorch Autograd 对照，需要另行提供 torch 环境。
#
# import torch
# torch.set_default_dtype(torch.float64)
#
# tx = torch.tensor(X_train)
# ty = torch.tensor(y_train)
# tw1 = torch.tensor(parameters["W1"], requires_grad=True)
# tb1 = torch.tensor(parameters["b1"], requires_grad=True)
# tw2 = torch.tensor(parameters["W2"], requires_grad=True)
# tb2 = torch.tensor(parameters["b2"], requires_grad=True)
#
# th = torch.relu(tx @ tw1 + tb1)
# tlogits = th @ tw2 + tb2
# tloss = torch.nn.functional.cross_entropy(tlogits, ty, reduction="mean")
# tloss.backward()
#
# for name, torch_gradient in {
#     "W1": tw1.grad, "b1": tb1.grad,
#     "W2": tw2.grad, "b2": tb2.grad,
# }.items():
#     assert np.allclose(
#         gradients[name],
#         torch_gradient.detach().numpy(),
#         atol=1e-9,
#         rtol=1e-7,
#     )
```

## 14. 前向缓存、重算与显存

反向需要局部信息：

- 矩阵乘法对 $W$ 求梯度需要输入 $X$；
- 对输入求梯度需要 $W$；
- ReLU backward 需要输入符号或等价 mask；
- Sigmoid backward 可使用前向输出 $h$ 计算 $h(1-h)$；
- Softmax-cross-entropy fused backward 可保存概率或重新计算稳定 Softmax。

全部保存会提高速度但占内存。gradient checkpointing 只保存部分边界 activation，反向时重跑中间前向，以计算换显存。

若有 $L$ 个等宽层、batch 为 $B$、宽度 $m$、float32：

$$
\text{仅保存一份层输出的字节}
\approx4LBm.
$$

真实训练还包括 pre-activation、梯度、临时 workspace、参数、优化器状态，不能把这个下界当峰值。

```python
def activation_memory_bytes(depth, batch, width, bytes_per_value=4):
    return depth * batch * width * bytes_per_value

depth, batch, width = 48, 256, 2048
save_all = activation_memory_bytes(depth, batch, width)
checkpoint_interval = 8
saved_boundaries = depth // checkpoint_interval + 1
checkpoint_memory = activation_memory_bytes(saved_boundaries, batch, width)
recomputed_layers_upper_bound = depth - saved_boundaries

assert save_all == 100_663_296
assert checkpoint_memory < save_all / 6
assert recomputed_layers_upper_bound > 0
print("全存/检查点激活 MiB：", round(save_all / 2**20, 1), round(checkpoint_memory / 2**20, 1))
```

<MlQuiz kind="缓存权衡" question="gradient checkpointing 的核心交换是什么？" :options="['用更高标签噪声换准确率', '丢弃部分中间激活，反向时重算，以额外计算换显存', '用更多参数换更少数据', '把梯度改成整数']" :answer="1" explanation="它不改变数学梯度，只改变哪些前向中间值保存、哪些在反向重算。" />

## 15. 梯度消失、爆炸与裁剪边界

对标量链

$$
h_\ell=f_\ell(h_{\ell-1}),
$$

输入梯度为

$$
\frac{\partial L}{\partial h_0}
=\frac{\partial L}{\partial h_L}
\prod_{\ell=1}^{L}
\frac{\partial h_\ell}{\partial h_{\ell-1}}.
$$

若局部增益长期小于 1，乘积指数衰减；长期大于 1，指数增长。向量网络对应 Jacobian 连乘，其奇异值谱决定不同方向的放大与压缩。

初始化、归一化、残差连接和门控结构用于改善传播。梯度裁剪

$$
g_{\text{clip}}
=g\min\left(1,\frac{\tau}{\|g\|_2}\right)
$$

限制单次更新，但不修复造成爆炸的模型、数据或实现根因。若每一步都裁剪，必须调查学习率、loss reduction、序列长度和异常 batch。

```python
import numpy as np

depth = 30
local_gains = np.array([0.5, 0.9, 1.0, 1.1, 1.5])
input_gradient_scale = local_gains ** depth

assert input_gradient_scale[0] < 1e-8
assert np.isclose(input_gradient_scale[2], 1.0)
assert input_gradient_scale[-1] > 1e5

raw_gradient = np.array([120.0, -50.0, 30.0])
clip_threshold = 10.0
raw_norm = np.linalg.norm(raw_gradient)
clipped_gradient = raw_gradient * min(1.0, clip_threshold / raw_norm)

assert np.isclose(np.linalg.norm(clipped_gradient), clip_threshold)
assert np.allclose(clipped_gradient / np.linalg.norm(clipped_gradient), raw_gradient / raw_norm)
```

<MlQuiz kind="梯度传播" question="深链中每层局部导数绝对值约为 0.5，输入端梯度量级怎样随深度 L 变化？" :options="['约按 0.5^L 衰减', '保持不变', '按 L 线性增长', '一定变成负数']" :answer="0" explanation="链式法则连续相乘；向量情形由 Jacobian 奇异值沿方向共同决定。" />

## 16. 梯度累积、清零与 loss reduction

许多框架的 parameter.grad 默认做累加，原因是计算图分支和参数共享需要 +=。训练循环因此必须显式管理生命周期：

1. 清零旧梯度；
2. 对一个或多个 micro-batch 前向和 backward；
3. 若要模拟大 batch mean，按总样本数统一缩放；
4. 必要时先 unscale，再检查有限性和裁剪；
5. optimizer step；
6. 记录本步真正使用的梯度范数。

若每个 micro-batch 自己取 mean 后直接相加，最后还要除以 micro-batch 数；当 micro-batch 大小不等时，应按样本数加权，而不是简单平均。

```python
import numpy as np

micro_batch_gradients = [
    (np.array([0.8, -0.2]), 3),
    (np.array([0.1, 0.5]), 1),
]

weighted_large_batch_gradient = sum(
    mean_gradient * sample_count
    for mean_gradient, sample_count in micro_batch_gradients
) / sum(sample_count for _, sample_count in micro_batch_gradients)

naive_equal_micro_batch_average = np.mean(
    [gradient for gradient, _ in micro_batch_gradients],
    axis=0,
)

assert np.allclose(weighted_large_batch_gradient, np.array([0.625, -0.025]))
assert not np.allclose(weighted_large_batch_gradient, naive_equal_micro_batch_average)
```

<MlQuiz kind="梯度生命周期" question="两个 micro-batch 大小分别为 3 和 1，各自返回 mean gradient。怎样得到等价于四样本大 batch mean 的梯度？" :options="['两份梯度简单平均', '按各自样本数加权求和后除以总样本数', '只用更大的 micro-batch', '把两份梯度相乘']" :answer="1" explanation="每份 mean 先还原成样本贡献总和，再除以总样本数；大小不等时简单平均会错。" />

## 17. 混合精度中的 loss scaling

float16 的动态范围可能让很小的梯度下溢为 0。loss scaling 在反向前把 loss 乘 $S$，使梯度也乘 $S$；优化前再用更高精度除以 $S$。

$$
\nabla_\theta(SL)=S\nabla_\theta L.
$$

正确顺序是：

1. scaled backward；
2. unscale gradients；
3. 检查 NaN/Inf；
4. 对未缩放梯度做裁剪；
5. optimizer step。

若在 unscale 前裁剪，阈值语义会随 $S$ 改变。

```python
import numpy as np

tiny_gradient = np.float32(1e-8)
direct_half = np.float16(tiny_gradient)
scale = np.float32(4096.0)
scaled_half = np.float16(tiny_gradient * scale)
recovered = np.float32(scaled_half) / scale

assert direct_half == 0.0
assert scaled_half != 0.0
assert np.isclose(recovered, tiny_gradient, rtol=1e-3)
```

## 数值验证：一次更新必须通过哪些门禁

### 18.1 小数据过拟合

选择 16–32 个可信样本，关闭 dropout 和强正则化。模型应能显著降低训练 loss。若不能，先查数据、前向、反向和优化器，不要继续加深网络。

### 18.2 全参数 shape 与有限性

```python
for name, parameter in parameters.items():
    gradient = gradients[name]
    assert gradient.shape == parameter.shape
    assert np.all(np.isfinite(parameter))
    assert np.all(np.isfinite(gradient))

assert np.isfinite(initial_loss)
assert np.allclose(probability.sum(axis=1), 1.0)
```

### 18.3 从零训练教学 XOR

```python
import numpy as np

rng = np.random.default_rng(1404)
trained_parameters = {
    "W1": rng.normal(scale=1.0, size=(2, 8)),
    "b1": np.zeros(8),
    "W2": rng.normal(scale=np.sqrt(2 / 8), size=(8, 2)),
    "b2": np.zeros(2),
}
training_losses = []

for step in range(500):
    train_probability, train_cache = mlp_forward(X_train, trained_parameters)
    train_loss = mean_cross_entropy(train_probability, y_train)
    train_gradients, _ = mlp_backward(y_train, trained_parameters, train_cache)
    learning_rate = 0.4 if step < 300 else 0.15
    for name in trained_parameters:
        trained_parameters[name] -= learning_rate * train_gradients[name]
    training_losses.append(train_loss)

test_probability, _ = mlp_forward(X_test, trained_parameters)
test_prediction = test_probability.argmax(axis=1)
test_accuracy = np.mean(test_prediction == y_test)

assert training_losses[-1] < training_losses[0] * 0.15
assert test_accuracy > 0.98
print("训练 loss 首/尾：", round(training_losses[0], 5), round(training_losses[-1], 5))
print("独立批次测试准确率：", round(test_accuracy, 4))
```

这个准确率只验证教学规则可学习，不是工业质量验收。真实系统还需成本敏感指标、概率校准、分组外验证和失效降级。

## 复杂度与工程边界

### 19.1 时间

对常规稠密层，前向矩阵乘法和反向的两个矩阵乘法同阶：

$$
O(Bdm).
$$

整个网络的一次 backward 通常是 forward 的常数倍，但具体倍数取决于算子、融合、硬件和是否重算。不能把“同阶”误写成“耗时完全相等”。

### 19.2 内存

主要包括：

- 参数与参数梯度；
- 前向 activation/cache；
- 反向临时张量；
- 优化器状态，例如 Adam 的一阶与二阶矩；
- 框架 workspace 与通信缓冲。

### 19.3 分布式 reduction

数据并行中，各 worker 对不同样本计算局部梯度，再做 all-reduce。系统必须冻结：

- 本地 loss 是 mean 还是 sum；
- all-reduce 是 sum 还是 average；
- 世界大小变化时学习率怎样调整；
- 梯度累积与通信在哪一步发生；
- 异常 worker、NaN 和超时怎样处理。

若本地已经 mean、all-reduce 又 sum，梯度会随 worker 数放大；框架默认值不能代替合同。

### 19.4 高阶导数

普通 backward 得到一阶 VJP。若需要 Hessian-vector product、元学习或梯度惩罚，必须保留用于“对梯度再求导”的图，内存和计算都会增加。训练中无意保留图会造成显存增长。

## 20. 常见失败模式与定位顺序

| 现象 | 优先检查 | 根因候选 | 处置 |
|---|---|---|---|
| loss 有限、某参数 grad=None | 计算图连接 | detach、离散操作、未启用梯度 | 找到断图节点，不用零梯度掩盖 |
| grad shape 正确但数值错 | 差分与分支次数 | 漏乘、覆盖、错误转置 | 小图逐节点 VJP 检查 |
| db 随样本顺序变化 | 广播反向 | 只取一行而非归约 | sum_to_shape 并断言 shape |
| batch 翻倍梯度也翻倍 | reduction | sum/mean 合同不一致 | 固化 loss 与通信归约 |
| ReLU 梯度大面积为 0 | Z 分布 | 失活、偏置漂移、输入尺度 | 查看 pre-activation 直方图 |
| 梯度突然爆炸 | 分层范数 | 异常 batch、长链、尺度错误 | 定位首个异常层，再决定裁剪 |
| float16 梯度为 0 | dtype 与 scaler | 下溢 | loss scaling，unscale 后验收 |
| 每步梯度越来越大 | 清零时机 | 跨 step 意外累积 | 检查 zero_grad 与 accumulation |
| checkpoint 后结果变化 | 重算确定性 | dropout/RNG 状态不一致 | 保存或重放 RNG 状态 |
| 差分在 ReLU 附近失败 | 检查点 | 跨越不可导折点 | 移开检查点或做单侧审计 |

### 推荐定位顺序

1. 验证输入、标签、参数和 loss 有限；
2. 在一个样本上打印计算图与 shape；
3. 在极小平滑网络上做 float64 有限差分；
4. 与第二个独立实现或可选 Autograd 对照；
5. 检查每层 activation/gradient RMS 和分支累加次数；
6. 再检查学习率、初始化、正则化和大规模系统。

## 21. 本章可执行检查清单

### 数学

- [ ] 能从局部线性化解释梯度，不把它当有限步因果保证。
- [ ] 能手算含分支计算图并在分叉处累加。
- [ ] 能区分 Jacobian、JVP 和 VJP。
- [ ] 能从微分推导矩阵乘法的 dX 与 dW。

### 两层 MLP

- [ ] 能从 Softmax-cross-entropy 得到 G₂。
- [ ] 能逐层写出 dW₂、db₂、dH、G₁、dW₁、db₁、dX。
- [ ] 每个梯度 shape 与原变量一致。
- [ ] batch mean 的 1/B 只出现一次且位置明确。

### 实现审计

- [ ] 共享节点使用累加，不覆盖。
- [ ] broadcasting 使用 sum_to_shape。
- [ ] float64 中抽样参数通过中心差分。
- [ ] ReLU 折点不被错误纳入平滑差分门禁。
- [ ] 可选框架对照使用相同权重、dtype 和 reduction。

### 训练系统

- [ ] 明确 zero_grad、micro-batch 累积和 optimizer step 顺序。
- [ ] 混合精度先 unscale 再检查和裁剪。
- [ ] activation、gradient、裁剪比例与非有限值有分层监控。
- [ ] 分布式 all-reduce 的 sum/average 合同已冻结。

## 22. 练习与最小实现

### 练习 1：手推分支图

对

$$
q=x+y,\quad
r=xy,\quad
s=q+r,\quad
L=\log(1+e^s)
$$

完成前向和反向，明确 $x,y$ 的两条路径贡献，并用中心差分核对。

### 练习 2：补全 NumPy MLP

只用 NumPy 实现 $2\to8\to2$ 网络：

- stable Softmax 与 mean cross-entropy；
- 前向 cache；
- 全部参数和输入梯度；
- 抽样差分检查；
- 训练曲线与二维边界；
- shape、有限性和 batch isolation 断言。

### 练习 3：实现最小 DAG 自动微分

在本章 Value 基础上增加：

- subtraction、division、exp、log、Tanh、ReLU；
- 一个节点多次使用时的梯度累加；
- 拓扑排序和循环拒绝；
- zero_grad；
- 与手推和有限差分的测试。

### 练习 4：与 PyTorch 对照

在另行提供的 PyTorch 环境中复现同一组 float64 权重和输入。比较四组参数梯度与输入梯度，报告最大绝对误差、相对误差、dtype 和 reduction，不只报告“看起来一致”。

```python
# TODO：实现支持任意层宽的 NumPy MLP forward/backward。
#
# 要求：
# 1. 参数、cache、gradient 使用稳定命名；
# 2. 每层 gradient shape 与 parameter shape 一致；
# 3. 随机抽取至少 10 个坐标做 float64 中心差分；
# 4. 支持 mean 与 sum reduction，并验证比例；
# 5. 在教学 XOR 上训练并画出独立批次决策区域。
```

## 23. 本章结论

1. 反向传播是链式法则在计算图上的逆拓扑动态规划。
2. 节点伴随量表示最终标量损失对该节点的局部敏感度。
3. 分支和参数共享要求梯度累加；覆盖会静默丢失路径。
4. 深度学习训练需要 VJP，而不是显式构造完整 Jacobian。
5. 矩阵乘法反向由 Frobenius 内积得到 $dX=GW^T$、$dW=X^TG$。
6. 广播的反向是把梯度归约回原输入 shape。
7. Softmax-cross-entropy 的 batch mean 梯度是 $(P-Y)/B$。
8. 有限差分、第二实现对照、shape 与有限性共同构成梯度门禁。
9. cache、checkpoint、混合精度、梯度累积和分布式归约都必须有明确合同。
10. 梯度正确只是训练可信的必要条件，标签、目标、泛化与部署仍需独立验证。

## 来源与许可

Jacobian、JVP、VJP 与复合 Jacobian 的定义参考 *Probabilistic Machine Learning: An Introduction*（源 PDF 第 301–303 页，§7.8.5）；前向/反向模式、MLP VJP、常见层与计算图参考同书源 PDF 第 468–475 页（§13.3–13.3.4）。反向传播线性复杂度、局部 backward、MLP cache 与批量矩阵实现参考 Stanford CS229 Lecture Notes 源 PDF 第 99–111 页（§7.4）。

有限差分、链式法则、标量反向和向量反向的讲解顺序参考 CMU 10-301/10-601 Lecture 12 源 PDF 第 39–53 页及 Lecture 13 源 PDF 第 14–54 页。计算图从朴素重复求导到按边线性复杂度、输出层与 ReLU 网络反向的边界参考 Berkeley CS189/289A 讲义源 PDF 第 98–107 页。前向缓存、分叉处梯度累加、分阶段计算和向量化梯度参考本地保存的 Stanford CS231n *Backpropagation, Intuitions* 页面；该页面随附 MIT License。本章未复用上述资料的原图，三项网页实验、焊接案例、代码和题目均为本站独立创作。

激光焊接双传感器数据、四象限标签、梯度审计账本和部署门禁均为教学合成，不代表真实焊接缺陷机理、传感器性能、质量阈值、放行标准或安全策略。

<MlLessonProgress
  lesson-id="14"
  next-href="/courses/machine-learning/15-cnn"
  next-label="第 15 章：卷积神经网络（CNN）"
/>
