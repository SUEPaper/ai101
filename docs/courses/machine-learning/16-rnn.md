---
course: machine-learning
edition: industrial
title: 循环神经网络（RNN）
description: 以连续退火炉钢卷热历史判读为贯穿案例，从序列条件分解与共享状态递推推导时间反向传播、梯度消失与爆炸、截断 BPTT、LSTM、GRU 及状态化部署，并用 NumPy、有限差分和最小字符模型建立验收门禁。
outline: [2, 3]
---

<MlLessonHero
  number="16"
  title="循环神经网络（RNN）"
  question="一卷钢带当前是否欠热，不只取决于这一秒的炉温，还取决于它此前经历了什么；怎样用固定大小的状态压缩可变长度历史，并让跨时间共享的参数得到完整、稳定、可审计的梯度？"
  duration="建议 660–780 分钟"
  :outcomes="[
    '从概率链式法则、序列任务形状和因果约束建立 RNN 状态递推，手算时间展开、参数量与三步前向',
    '从展开计算图推导 BPTT、Jacobian 连乘、梯度消失与爆炸，并区分裁剪、截断和状态携带的作用边界',
    '逐门实现 LSTM 与 GRU，核对两种 GRU 更新门记号，并用有限差分、字符模型和工业时序合同验收实现'
  ]"
/>

第 15 章利用空间局部性和参数共享处理二维网格。本章把共享方向旋转到时间轴：同一个状态转移函数在每个时刻重复调用，把此前历史压进固定维度的隐藏状态。

RNN 的价值不只是“能接收序列”。它明确提出三个可检验假设：

1. **历史可压缩**：预测所需历史能被有限维状态 $\mathbf h_t$ 充分概括；
2. **转移规律可共享**：不同时间位置使用同一组参数；
3. **顺序有意义**：交换两个观测一般会改变状态和输出。

这些仍是假设。若任务只取决于无序集合，RNN 会平白引入顺序偏置；若设备机理在工艺段之间突变，一套共享转移也可能欠拟合。工业实现必须先定义“一个序列是什么、何时复位状态、未来信息能否使用”，再讨论单元类型。

## 贯穿案例：连续退火炉钢卷热历史判读

连续退火线每秒记录带钢温差、线速偏差和炉内露点偏差。任务是在钢卷离开均热段时识别需要离线复核的热历史，而不是直接闭环控制炉温。

本章使用教学合成数据。每条样本代表一卷钢带在一个炉段内的最多 30 个时刻：

| 字段 | 张量语义 | 单位与边界 |
|---|---|---|
| `temperature_error[t]` | 实测带温减目标带温 | 摄氏度；传感器失效不能填成 0 冒充“恰好达标” |
| `speed_error[t]` | 线速相对设定的偏差 | 标准化比例；影响单位长度的受热时间 |
| `dew_point_error[t]` | 露点相对工艺上限的偏差 | 摄氏度；早期偏差可能经历史累积影响终态 |
| `valid_mask[t]` | 该时刻是否真实存在 | padding 位置不更新状态、不计损失 |
| `campaign_id` | 连续生产批次 | 同一 campaign 不得跨训练、验证和测试 |
| `review_label` | 离线质量复核结果 | 只表示复核目标，不等价于停线或自动放行 |

状态边界是模型合同的一部分：

- 同一钢卷被切成连续窗口时，可以按时间顺序携带末状态；
- 换卷、换炉、维护、时间倒退或记录断档时必须复位；
- shuffle 后的 mini-batch 不能沿用前一批的状态；
- 在线判读只能用 $x_{1:t}$，离线复盘才可考虑双向上下文。

### 1. 按生产 campaign 切分，再构造序列与掩码

下面生成 48 卷教学序列，分属 12 个 campaign。有效长度在 22–30 之间，padding 统一为明显异常的 `99.0`，用来检验实现是否真的应用了 mask。

```python
import numpy as np

rng = np.random.default_rng(20260805)
campaign_count = 12
coils_per_campaign = 4
coil_count = campaign_count * coils_per_campaign
maximum_steps = 30
feature_count = 3

campaign_id = np.repeat(np.arange(campaign_count), coils_per_campaign)
sequence_length = rng.integers(22, maximum_steps + 1, size=coil_count)
sequences = np.full((coil_count, maximum_steps, feature_count), 99.0, dtype=np.float64)
valid_mask = np.zeros((coil_count, maximum_steps), dtype=np.float64)
review_label = np.zeros(coil_count, dtype=np.int64)
thermal_score = np.zeros(coil_count, dtype=np.float64)

campaign_offset = rng.normal(0.0, 0.12, size=(campaign_count, 3))
for coil_index in range(coil_count):
    length = sequence_length[coil_index]
    time = np.arange(length)
    offset = campaign_offset[campaign_id[coil_index]]
    temperature_error = 0.25 * np.sin(time / 4.5) + offset[0]
    speed_error = 0.10 * np.cos(time / 5.0) + offset[1]
    dew_point_error = rng.normal(0.0, 0.12, size=length) + offset[2]

    if coil_index % 3 == 1:
        temperature_error[3:8] -= 1.0
    if coil_index % 5 == 2:
        dew_point_error[1:5] += 0.85
    temperature_error += rng.normal(0.0, 0.08, size=length)
    speed_error += rng.normal(0.0, 0.04, size=length)

    current_state = 0.0
    for step in range(length):
        deficit = max(0.0, -temperature_error[step] + 0.7 * speed_error[step])
        current_state = 0.90 * current_state + deficit + 0.45 * max(0.0, dew_point_error[step])

    sequences[coil_index, :length, 0] = temperature_error
    sequences[coil_index, :length, 1] = speed_error
    sequences[coil_index, :length, 2] = dew_point_error
    valid_mask[coil_index, :length] = 1.0
    thermal_score[coil_index] = current_state

review_threshold = float(np.median(thermal_score))
review_label = (thermal_score > review_threshold).astype(np.int64)

campaign_order = rng.permutation(campaign_count)
train_campaigns = campaign_order[:8]
validation_campaigns = campaign_order[8:10]
test_campaigns = campaign_order[10:]
train_rows = np.isin(campaign_id, train_campaigns)
validation_rows = np.isin(campaign_id, validation_campaigns)
test_rows = np.isin(campaign_id, test_campaigns)

assert sequences.shape == (48, 30, 3)
assert valid_mask.shape == (48, 30)
assert np.all(valid_mask.sum(axis=1) == sequence_length)
assert set(train_campaigns).isdisjoint(validation_campaigns)
assert set(train_campaigns).isdisjoint(test_campaigns)
assert set(validation_campaigns).isdisjoint(test_campaigns)
assert train_rows.sum() == 32
assert validation_rows.sum() == test_rows.sum() == 8
assert set(np.unique(review_label)) == {0, 1}
print("训练/验证/测试钢卷：", train_rows.sum(), validation_rows.sum(), test_rows.sum())
print("有效长度范围：", sequence_length.min(), sequence_length.max())
```

这段数据准备刻意保留三种不同概念：

- `sequence_length` 是每卷真实长度；
- `valid_mask` 是批处理中的数值接口；
- `campaign_id` 是防止跨集合泄漏的分组键。

三者不能相互替代。mask 正确不代表切分正确；按 campaign 切分也不能阻止 padding 污染状态。

<MlQuiz kind="数据合同" question="同一钢卷被切成四个连续窗口训练 stateful RNN，下面哪种做法正确？" :options="['任意打乱窗口并始终携带状态', '只在同卷且时间连续的相邻窗口间携带状态，并在卷边界复位', '把上一批最后一个样本的状态交给下一批第一个样本', '删除所有短序列']" :answer="1" explanation="状态携带必须遵守实体身份与时间连续性；跨卷或乱序携带会制造不可见的数据串扰。" />

## 2. 序列概率：链式法则不等于 RNN 假设

对随机序列 $\mathbf x_{1:T}$，概率链式法则恒成立：

$$
p(\mathbf x_{1:T})
=p(\mathbf x_1)\prod_{t=2}^{T}
p(\mathbf x_t\mid\mathbf x_{1:t-1}).
$$

这是概率恒等式，没有规定如何表示越来越长的历史。RNN 增加了一个建模假设：用固定维度状态递推

$$
\mathbf h_t=f_\theta(\mathbf x_t,\mathbf h_{t-1})
$$

近似保存预测未来所需的信息，并令

$$
p_\theta(\mathbf x_t\mid\mathbf x_{1:t-1})
=p_\theta(\mathbf x_t\mid\mathbf h_{t-1}).
$$

若输出是离散 token，常用 Softmax 得到每步条件分布。联合对数似然变为各步对数概率之和：

$$
\log p_\theta(\mathbf x_{1:T})
=\sum_{t=1}^{T}\log p_\theta(\mathbf x_t\mid\mathbf h_{t-1}).
$$

```python
conditional_probability = np.array([0.42, 0.71, 0.63, 0.80], dtype=np.float64)
joint_probability = float(np.prod(conditional_probability))
joint_log_probability = float(np.log(conditional_probability).sum())

assert np.isclose(np.exp(joint_log_probability), joint_probability)
assert 0.0 < joint_probability < conditional_probability.min()

logits = np.array([1200.0, 1198.0, 1195.0])
shifted = logits - logits.max()
probabilities = np.exp(shifted) / np.exp(shifted).sum()
assert np.isfinite(probabilities).all()
assert np.isclose(probabilities.sum(), 1.0)
```

### 2.1 任务形状先于单元选择

| 任务 | 输入与输出 | 退火炉例子 | 常见读出 |
|---|---|---|---|
| many-to-one | $x_{1:T}\to y$ | 整卷是否复核 | 最终有效状态或带 mask 的汇总 |
| many-to-many 对齐 | $x_{1:T}\to y_{1:T}$ | 每秒异常分数 | 每个有效时刻输出 |
| one-to-many | $x\to y_{1:T}$ | 给定工艺条件生成轨迹 | 自回归解码 |
| sequence-to-sequence | $x_{1:S}\to y_{1:T}$ | 历史窗口到未来轨迹 | 编码器—解码器 |

在线判读的 $y_t$ 不能读取 $x_{t+1:T}$。双向 RNN 把正向与反向状态拼接，适合完整序列已经到齐的离线标注，但会破坏在线因果合同。

<MlQuiz kind="任务边界" question="要在每秒数据到达时立即发出欠热预警，哪种结构会直接泄漏未来？" :options="['单向 RNN', '因果滑动窗口', '双向 RNN', '只读取 h_t 的线性层']" :answer="2" explanation="反向状态读取当前时刻之后的观测；它可用于离线复盘，不能冒充在线结果。" />

## 3. Vanilla RNN：一个转移函数沿时间重复调用

最基本的 Tanh RNN 为

$$
\begin{aligned}
\mathbf a_t
&=\mathbf W_{xh}\mathbf x_t
+\mathbf W_{hh}\mathbf h_{t-1}+\mathbf b_h,\\
\mathbf h_t&=\tanh(\mathbf a_t),\\
\mathbf o_t&=\mathbf W_{hy}\mathbf h_t+\mathbf b_y.
\end{aligned}
$$

设输入维度 $D$、隐藏维度 $H$、输出维度 $O$。含 bias 的参数量为

$$
\underbrace{HD}_{W_{xh}}
+\underbrace{H^2}_{W_{hh}}
+\underbrace{H}_{b_h}
+\underbrace{OH}_{W_{hy}}
+\underbrace{O}_{b_y}.
$$

它与序列长度 $T$ 无关。展开图里虽然出现 $T$ 个方框，每个方框引用的是同一组变量，而不是复制出 $T$ 套参数。

```python
def vanilla_rnn_forward(x, initial_state, weight_xh, weight_hh, bias_h):
    x = np.asarray(x, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.float64).copy()
    states = [state.copy()]
    preactivations = []
    for current_input in x:
        preactivation = weight_xh @ current_input + weight_hh @ state + bias_h
        state = np.tanh(preactivation)
        preactivations.append(preactivation)
        states.append(state.copy())
    return np.asarray(states), np.asarray(preactivations)

input_dimension, hidden_dimension, output_dimension = 3, 5, 2
weight_xh = rng.normal(0.0, 0.2, size=(hidden_dimension, input_dimension))
weight_hh = rng.normal(0.0, 0.2, size=(hidden_dimension, hidden_dimension))
bias_h = np.zeros(hidden_dimension)
weight_hy = rng.normal(0.0, 0.2, size=(output_dimension, hidden_dimension))
bias_y = np.zeros(output_dimension)

example_sequence = sequences[0, :sequence_length[0]]
example_states, example_preactivations = vanilla_rnn_forward(
    example_sequence, np.zeros(hidden_dimension), weight_xh, weight_hh, bias_h,
)
parameter_count = (
    hidden_dimension * input_dimension + hidden_dimension ** 2 + hidden_dimension
    + output_dimension * hidden_dimension + output_dimension
)

assert example_states.shape == (sequence_length[0] + 1, hidden_dimension)
assert example_preactivations.shape == (sequence_length[0], hidden_dimension)
assert parameter_count == 57
assert np.all(np.abs(example_states) < 1.0)
```

<RnnStateUnrollerLab />

### 3.1 顺序敏感来自非交换的递推

均值、最大值等对称汇总会忽略顺序；RNN 一般不会。即使两个序列包含完全相同的值，事件发生在早期还是晚期，最终状态也可能不同。

```python
def scalar_recurrence(values, input_gain=0.9, recurrent_gain=0.78, initial_state=0.0):
    state = float(initial_state)
    history = []
    for value in values:
        state = np.tanh(input_gain * value + recurrent_gain * state)
        history.append(state)
    return np.asarray(history)

early_event = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
late_event = early_event[::-1]
early_history = scalar_recurrence(early_event)
late_history = scalar_recurrence(late_event)

assert np.isclose(early_event.mean(), late_event.mean())
assert not np.isclose(early_history[-1], late_history[-1])
assert late_history[-1] > early_history[-1]
```

这里不是说“越晚越重要”是普适事实，而是说明递推本身会产生时间位置效应。若业务要求绝对时间平移不变，必须通过数据、结构或汇总显式实现，不能从“参数共享”自动推出。

<MlQuiz kind="结构辨析" question="两个序列包含同一组数值但顺序相反，普通 Tanh RNN 的最终状态通常如何？" :options="['必然相同，因为参数共享', '通常不同，因为递推复合一般不可交换', '必然互为相反数', '只由序列长度决定']" :answer="1" explanation="共享意味着每步用同一个函数，不意味着交换函数输入的时间顺序不影响复合结果。" />

## 4. 变长序列：padding 不能悄悄更新状态

把不同长度序列组成 batch 时，常见做法是 padding 加 mask。对样本 $n$ 的第 $t$ 步，令 $m_{nt}\in\{0,1\}$：

$$
\widehat{\mathbf h}_{nt}=f_\theta(\mathbf x_{nt},\mathbf h_{n,t-1}),
\qquad
\mathbf h_{nt}=m_{nt}\widehat{\mathbf h}_{nt}+(1-m_{nt})\mathbf h_{n,t-1}.
$$

当 $m=0$，状态原样穿过。若还计算逐时刻损失，也必须用 mask 排除 padding；最后一步读出则要取“最后一个有效状态”，不是数组最后一行的伪状态。

```python
def masked_rnn_forward(batch_x, mask, initial_state, weight_xh, weight_hh, bias_h):
    batch_size, steps, _ = batch_x.shape
    state = np.broadcast_to(initial_state, (batch_size, len(initial_state))).copy()
    state_history = []
    for step in range(steps):
        proposed = np.tanh(batch_x[:, step] @ weight_xh.T + state @ weight_hh.T + bias_h)
        current_mask = mask[:, step:step + 1]
        state = current_mask * proposed + (1.0 - current_mask) * state
        state_history.append(state.copy())
    return np.stack(state_history, axis=1)

small_batch = sequences[:3].copy()
small_mask = valid_mask[:3].copy()
masked_states = masked_rnn_forward(
    small_batch, small_mask, np.zeros(hidden_dimension), weight_xh, weight_hh, bias_h,
)

changed_padding = small_batch.copy()
changed_padding[small_mask == 0] = -777.0
masked_states_changed = masked_rnn_forward(
    changed_padding, small_mask, np.zeros(hidden_dimension), weight_xh, weight_hh, bias_h,
)

assert masked_states.shape == (3, maximum_steps, hidden_dimension)
assert np.allclose(masked_states, masked_states_changed)
for row, length in enumerate(sequence_length[:3]):
    assert np.allclose(masked_states[row, length - 1], masked_states[row, -1])
```

打包序列（packed sequence）可跳过 padding 计算，但它改变的是执行效率，不改变 mask 和真实长度的语义。导出到其他推理引擎时，必须重新验收排序、长度 dtype、空序列策略和最终状态索引。

<MlQuiz kind="实现定位" question="短序列的最终状态随 padding 填充值变化，最可能的根因是什么？" :options="['隐藏维度太小', 'mask 没有阻止 padding 步更新状态，或最终状态索引取错', 'Softmax 类别太多', '参数共享次数不足']" :answer="1" explanation="正确的 masked recurrence 对无效位置的具体填充值不敏感。" />

## 5. 展开计算图：BPTT 就是时间轴上的反向传播

记总损失

$$
L=\sum_{t=1}^{T}\ell_t(\mathbf h_t).
$$

$\mathbf h_t$ 有两类下游：当前时刻损失和下一时刻状态。因此反向递推为

$$
\frac{\partial L}{\partial\mathbf h_t}
=\frac{\partial\ell_t}{\partial\mathbf h_t}
+\mathbf W_{hh}^{\mathsf T}\boldsymbol\delta_{t+1},
$$

$$
\boldsymbol\delta_t
=\frac{\partial L}{\partial\mathbf a_t}
=\frac{\partial L}{\partial\mathbf h_t}
\odot(1-\mathbf h_t^2).
$$

共享参数在每个时间步都有一条影响损失的路径，总导数必须累加：

$$
\boxed{
\frac{\partial L}{\partial\mathbf W_{hh}}
=\sum_{t=1}^{T}\boldsymbol\delta_t\mathbf h_{t-1}^{\mathsf T}
},
$$

$$
\frac{\partial L}{\partial\mathbf W_{xh}}
=\sum_{t=1}^{T}\boldsymbol\delta_t\mathbf x_t^{\mathsf T},
\qquad
\frac{\partial L}{\partial\mathbf b_h}
=\sum_{t=1}^{T}\boldsymbol\delta_t.
$$

这与卷积核跨空间共享完全同构：前向多处读同一参数，反向就把所有使用位置的贡献收集到同一个梯度槽。

### 5.1 手算三步标量 RNN

令

$$
h_t=\tanh(W_xx_t+W_hh_{t-1}+b),
\qquad
L=\frac12(h_3-y)^2.
$$

取 $x=(0.2,-0.1,0.4)$、$W_x=0.7$、$W_h=0.5$、$b=-0.05$、$h_0=0.1$、$y=0.25$。反向从

$$
\delta_3=(h_3-y)(1-h_3^2)
$$

开始，并递推 $\delta_t=W_h\delta_{t+1}(1-h_t^2)$。

```python
def scalar_three_step_bptt(x, target, input_weight, recurrent_weight, bias, initial_state):
    states = [float(initial_state)]
    preactivations = []
    for value in x:
        preactivation = input_weight * value + recurrent_weight * states[-1] + bias
        preactivations.append(preactivation)
        states.append(float(np.tanh(preactivation)))

    loss = 0.5 * (states[-1] - target) ** 2
    gradient_input_weight = 0.0
    gradient_recurrent_weight = 0.0
    gradient_bias = 0.0
    next_state_gradient = states[-1] - target
    deltas = np.zeros(len(x))
    for step in range(len(x) - 1, -1, -1):
        delta = next_state_gradient * (1.0 - states[step + 1] ** 2)
        deltas[step] = delta
        gradient_input_weight += delta * x[step]
        gradient_recurrent_weight += delta * states[step]
        gradient_bias += delta
        next_state_gradient = recurrent_weight * delta
    return {
        "states": np.asarray(states),
        "preactivations": np.asarray(preactivations),
        "deltas": deltas,
        "loss": loss,
        "dWx": gradient_input_weight,
        "dWh": gradient_recurrent_weight,
        "db": gradient_bias,
        "dh0": next_state_gradient,
    }

three_step_x = np.array([0.2, -0.1, 0.4])
three_step = scalar_three_step_bptt(three_step_x, 0.25, 0.7, 0.5, -0.05, 0.1)

assert three_step["states"].shape == (4,)
assert three_step["deltas"].shape == (3,)
assert np.isclose(three_step["loss"], 0.5 * (three_step["states"][-1] - 0.25) ** 2)
assert np.isfinite(list(three_step.values())[3:]).all()
print("h0…h3：", np.round(three_step["states"], 6))
print("delta1…delta3：", np.round(three_step["deltas"], 6))
print("dWx, dWh, db：", three_step["dWx"], three_step["dWh"], three_step["db"])
```

手算时最常见的两处错误是：只把 $\delta_3$ 写进参数梯度，以及计算 $dW_h$ 时误用 $h_t$ 而不是前向真正读到的 $h_{t-1}$。

<div class="ml-stop">
  <strong>停下来推一推 1</strong>
  <p>不运行代码，按六位小数手算上面的 a1、h1、a2、h2、a3、h3，再从 delta3 倒推到 delta1。分别写出 dWh 的三项贡献，说明为何必须相加。</p>
</div>

<MlQuiz kind="梯度手算" question="在 h_t=tanh(W_x x_t+W_h h_{t-1}+b) 中，第 t 步对共享 W_h 的局部梯度外积使用哪两个量？" :options="['delta_t 与 h_{t-1}', 'delta_t 与 h_t', 'h_t 与 x_{t+1}', '只使用最后一步 delta_T']" :answer="0" explanation="前向在第 t 步由 W_h 乘的是 h_{t-1}，所以局部贡献为 delta_t h_{t-1}^T；共享参数再跨时间求和。" />

## 6. Jacobian 连乘解释长程梯度

忽略中间时刻的直接损失，早期状态到最终状态的导数为

$$
\frac{\partial\mathbf h_T}{\partial\mathbf h_t}
=\prod_{k=t+1}^{T}
\underbrace{
\operatorname{diag}(1-\mathbf h_k^2)\mathbf W_{hh}
}_{\mathbf J_k}.
$$

注意矩阵乘积有顺序，不能把它简化成一个标量特征值的幂。范数上界

$$
\left\|\frac{\partial\mathbf h_T}{\partial\mathbf h_t}\right\|
\le\prod_{k=t+1}^{T}\|\mathbf J_k\|
$$

给出直觉：平均局部增益小于 1 时通常指数衰减，大于 1 时可能爆炸。Tanh 饱和还会令 $1-h_t^2\approx0$，即使 $\|W_{hh}\|$ 不小，梯度仍可能消失。

```python
def scalar_gradient_multiplier(recurrent_gain, preactivation, lag):
    local_derivative = 1.0 - np.tanh(preactivation) ** 2
    return abs(recurrent_gain * local_derivative) ** lag

quiet_multiplier = scalar_gradient_multiplier(0.92, 0.55, 35)
saturated_multiplier = scalar_gradient_multiplier(1.10, 2.0, 35)
linear_explosion = scalar_gradient_multiplier(1.18, 0.0, 35)

assert quiet_multiplier < 1e-3
assert saturated_multiplier < quiet_multiplier
assert linear_explosion > 100.0

matrix_recurrent = np.array([[0.95, 0.20], [-0.10, 0.85]])
state_path = np.array([[0.15, 0.30], [0.20, 0.45], [0.55, 0.60], [0.70, 0.25]])
jacobian_product = np.eye(2)
for state in state_path:
    local_jacobian = np.diag(1.0 - state ** 2) @ matrix_recurrent
    jacobian_product = local_jacobian @ jacobian_product

singular_values = np.linalg.svd(jacobian_product, compute_uv=False)
assert singular_values.shape == (2,)
assert singular_values[0] >= singular_values[1] >= 0.0
```

<BpttGradientScopeLab />

### 6.1 权重谱只是线索，不是完整诊断

只看 $W_{hh}$ 的谱半径会遗漏：

- 激活工作点随样本和时刻变化；
- 非正规矩阵即使特征值不大，也可能产生暂态放大；
- 不同方向的奇异值差异很大；
- 门控、归一化、跳接和多层结构增加其他路径；
- 中间时刻损失会缩短有效反传距离。

因此训练日志至少应记录全局梯度范数、裁剪比例、隐藏状态饱和率、有效长度分布和按序列长度分层的指标。

<div class="ml-stop">
  <strong>停下来推一推 2</strong>
  <p>标量递推每步局部导数恒为 0.8，跨 50 步的倍率是 0.8^50。先估数量级，再想：把裁剪阈值从 5 改成 1，能否恢复这条早期信号？为什么？</p>
</div>

<MlQuiz kind="原理辨析" question="把 Tanh RNN 的 W_hh 谱半径调到 1，是否保证长程梯度不消失？" :options="['保证，因为每步 Jacobian 就是 W_hh', '不保证，激活导数、方向变化和矩阵非正规性仍进入 Jacobian 连乘', '保证，只要 batch 足够大', '只对分类任务保证']" :answer="1" explanation="真实局部 Jacobian 还包含激活导数并随时间变化；谱半径只是一个不完整线索。" />

## 7. 梯度裁剪与截断 BPTT 解决不同问题

### 7.1 全局范数裁剪限制更新，不恢复记忆

对所有参数梯度拼成向量 $\mathbf g$，全局范数裁剪为

$$
\mathbf g\leftarrow\mathbf g
\min\left(1,\frac{\tau}{\|\mathbf g\|_2+\epsilon}\right).
$$

它保持梯度方向，只在范数超过阈值时等比例缩小。逐元素 clamp 会改变方向，除非有明确理由，不应与全局范数裁剪混称。

```python
def clip_by_global_norm(gradients, threshold, epsilon=1e-12):
    global_norm = float(np.sqrt(sum(np.sum(gradient ** 2) for gradient in gradients)))
    scale = min(1.0, threshold / (global_norm + epsilon))
    return [gradient * scale for gradient in gradients], global_norm, scale

raw_gradients = [np.array([3.0, 4.0]), np.array([[0.0, 12.0]])]
clipped_gradients, raw_norm, clip_scale = clip_by_global_norm(raw_gradients, threshold=5.0)
clipped_norm = np.sqrt(sum(np.sum(gradient ** 2) for gradient in clipped_gradients))

assert np.isclose(raw_norm, 13.0)
assert np.isclose(clip_scale, 5.0 / 13.0)
assert np.isclose(clipped_norm, 5.0)
assert np.allclose(clipped_gradients[0] / clipped_gradients[0][0], np.array([1.0, 4.0 / 3.0]))
```

### 7.2 截断 BPTT 限制反向图，不等于清空前向状态

截断窗口为 $K$ 时，序列按块前向。每块可以接收上一块末状态，以保留前向上下文；但要在块边界停止梯度，使反向最多穿过 $K$ 步：

```text
state = initial_state
for chunk in chronological_chunks:
    state = detach(state)        # 数值保留，跨块梯度切断
    outputs, state = rnn(chunk, state)
    loss(outputs).backward()
    clip_grad_norm_(parameters, tau)
    optimizer.step()
```

“携带状态”和“跨边界反传”是两个正交开关：

| 前向状态 | 反向图 | 含义 |
|---|---|---|
| 不携带 | 不跨块 | 每块独立，长上下文完全丢失 |
| 携带 | 不跨块 | 标准截断 BPTT，能用旧状态但不把梯度传回旧块 |
| 携带 | 跨块 | 完整 BPTT，图和内存随总长度增长 |
| 跨实体携带 | 任意 | 合同错误，上一钢卷污染下一钢卷 |

```python
def chunked_forward(values, chunk_size, carry_state):
    chunk_outputs = []
    state = 0.0
    for start in range(0, len(values), chunk_size):
        if not carry_state:
            state = 0.0
        current_outputs = scalar_recurrence(
            values[start:start + chunk_size], initial_state=state,
        )
        state = float(current_outputs[-1])
        chunk_outputs.extend(current_outputs)
    return np.asarray(chunk_outputs)

window_sequence = np.array([0.7, 0.0, 0.0, -0.2, 0.0, 0.0])
full_forward = scalar_recurrence(window_sequence)
carried_forward = chunked_forward(window_sequence, chunk_size=3, carry_state=True)
reset_forward = chunked_forward(window_sequence, chunk_size=3, carry_state=False)

assert np.allclose(full_forward, carried_forward)
assert not np.allclose(full_forward, reset_forward)
```

截断降低显存和计算，代价是无法为跨越 $K$ 步的信用分配提供直接梯度。$K$ 应依据有效依赖长度、吞吐预算和按长度分层的验证结果选择，而不是照搬默认值。

<MlQuiz kind="训练策略" question="截断 BPTT 在块边界 detach 状态后，最准确的描述是什么？" :options="['状态数值和梯度都清零', '状态数值可继续前向，但梯度不再穿过旧块', '只裁剪爆炸梯度，不改变反传距离', '自动把双向 RNN 变成单向']" :answer="1" explanation="detach 保留张量值但切断计算图；它与是否把状态复位为零是两件事。" />

## 8. LSTM：把记忆更新改成受门控制的加法通路

Vanilla RNN 每步把旧状态和新输入一起压进 Tanh。LSTM（long short-term memory）把内部记忆 $\mathbf c_t$ 与对外隐藏状态 $\mathbf h_t$ 分开：

$$
\begin{aligned}
\mathbf i_t&=\sigma(\mathbf W_i[\mathbf x_t;\mathbf h_{t-1}]+\mathbf b_i),\\
\mathbf f_t&=\sigma(\mathbf W_f[\mathbf x_t;\mathbf h_{t-1}]+\mathbf b_f),\\
\mathbf o_t&=\sigma(\mathbf W_o[\mathbf x_t;\mathbf h_{t-1}]+\mathbf b_o),\\
\widetilde{\mathbf c}_t&=\tanh(\mathbf W_c[\mathbf x_t;\mathbf h_{t-1}]+\mathbf b_c),\\
\mathbf c_t&=\mathbf f_t\odot\mathbf c_{t-1}
+\mathbf i_t\odot\widetilde{\mathbf c}_t,\\
\mathbf h_t&=\mathbf o_t\odot\tanh(\mathbf c_t).
\end{aligned}
$$

逐门看语义：

1. 输入门 $i_t$ 决定候选内容写入多少；
2. 遗忘门 $f_t$ 决定旧细胞状态保留多少；
3. 候选 $\widetilde c_t$ 提供可写入的新内容；
4. 输出门 $o_t$ 决定内部记忆向 $h_t$ 暴露多少。

只看从 $c_{t-1}$ 到 $c_t$ 的直接加法路径，局部导数是 $f_t$。当一段时间内 $f_t\approx1$，这条路径可避免每步都穿过 Tanh 与完整循环矩阵。但门本身也依赖 $h_{t-1}$，所以总 Jacobian 还有其他分支；LSTM 是缓解，不是“永不梯度消失”的证明。

```python
def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(
        value >= 0,
        1.0 / (1.0 + np.exp(-value)),
        np.exp(value) / (1.0 + np.exp(value)),
    )

def lstm_step(x, previous_hidden, previous_cell, weight, bias):
    joined = np.concatenate([x, previous_hidden])
    gates = weight @ joined + bias
    hidden_size = len(previous_hidden)
    input_gate = sigmoid(gates[:hidden_size])
    forget_gate = sigmoid(gates[hidden_size:2 * hidden_size])
    output_gate = sigmoid(gates[2 * hidden_size:3 * hidden_size])
    candidate = np.tanh(gates[3 * hidden_size:])
    cell = forget_gate * previous_cell + input_gate * candidate
    hidden = output_gate * np.tanh(cell)
    return hidden, cell, (input_gate, forget_gate, output_gate, candidate)

lstm_input_size, lstm_hidden_size = 2, 3
lstm_weight = rng.normal(0.0, 0.25, size=(4 * lstm_hidden_size, lstm_input_size + lstm_hidden_size))
lstm_bias = np.zeros(4 * lstm_hidden_size)
lstm_bias[lstm_hidden_size:2 * lstm_hidden_size] = 1.0
lstm_hidden, lstm_cell, lstm_gates = lstm_step(
    np.array([0.4, -0.2]), np.zeros(3), np.array([0.2, -0.1, 0.3]), lstm_weight, lstm_bias,
)

assert lstm_hidden.shape == lstm_cell.shape == (3,)
assert all(gate.shape == (3,) for gate in lstm_gates)
assert np.isfinite(lstm_hidden).all()

previous_cell = np.array([0.7, -0.4])
candidate_cell = np.array([-0.2, 0.9])
keep_all = 1.0 * previous_cell + 0.0 * candidate_cell
replace_all = 0.0 * previous_cell + 1.0 * candidate_cell
erase_all = 0.0 * previous_cell + 0.0 * candidate_cell
assert np.allclose(keep_all, previous_cell)
assert np.allclose(replace_all, candidate_cell)
assert np.allclose(erase_all, 0.0)
```

把遗忘门 bias 初始化为正值可让初始 $f_t$ 偏大，但它只是优化先验，不是通用最优。必须同时观察任务依赖长度、门分布、状态漂移和饱和率。

<GatedMemoryFlowLab />

<MlQuiz kind="门控边界" question="在 LSTM 更新 c_t=f_t c_{t-1}+i_t c̃_t 中，哪组边界值会原样保留旧记忆并拒绝候选？" :options="['f=1, i=0', 'f=0, i=1', 'f=0, i=0', 'f=1, i=1']" :answer="0" explanation="直接代入即可：c_t=c_{t-1}。边界测试比只背门名称更可靠。" />

## 9. GRU：合并状态，但先声明更新门记号

GRU（gated recurrent unit）没有独立细胞状态。一套常见、也是本章采用的约定是：$z_t$ 表示**保留旧状态的比例**：

$$
\begin{aligned}
\mathbf z_t&=\sigma(\mathbf W_z\mathbf x_t+\mathbf U_z\mathbf h_{t-1}+\mathbf b_z),\\
\mathbf r_t&=\sigma(\mathbf W_r\mathbf x_t+\mathbf U_r\mathbf h_{t-1}+\mathbf b_r),\\
\widetilde{\mathbf h}_t
&=\tanh(\mathbf W_h\mathbf x_t
+\mathbf U_h(\mathbf r_t\odot\mathbf h_{t-1})+\mathbf b_h),\\
\mathbf h_t&=\mathbf z_t\odot\mathbf h_{t-1}
+(1-\mathbf z_t)\odot\widetilde{\mathbf h}_t.
\end{aligned}
$$

有些资料把最后一式写成

$$
\mathbf h_t=(1-\mathbf z_t)\odot\mathbf h_{t-1}
+\mathbf z_t\odot\widetilde{\mathbf h}_t,
$$

此时 $z_t$ 表示**写入候选的比例**。两者可通过 $z' = 1-z$ 相互转换，不代表算法冲突。阅读实现时不要只看“update gate”这个名字，要代入边界：$z=0$ 或 $1$ 时，究竟保留旧状态还是写入候选。

```python
def gru_step_retain_convention(x, previous_hidden, parameters):
    wz, uz, bz, wr, ur, br, wh, uh, bh = parameters
    update_retain = sigmoid(wz @ x + uz @ previous_hidden + bz)
    reset_gate = sigmoid(wr @ x + ur @ previous_hidden + br)
    candidate = np.tanh(wh @ x + uh @ (reset_gate * previous_hidden) + bh)
    hidden = update_retain * previous_hidden + (1.0 - update_retain) * candidate
    return hidden, update_retain, reset_gate, candidate

gru_input_size, gru_hidden_size = 2, 3
gru_parameters = (
    rng.normal(0.0, 0.2, size=(gru_hidden_size, gru_input_size)),
    rng.normal(0.0, 0.2, size=(gru_hidden_size, gru_hidden_size)),
    np.zeros(gru_hidden_size),
    rng.normal(0.0, 0.2, size=(gru_hidden_size, gru_input_size)),
    rng.normal(0.0, 0.2, size=(gru_hidden_size, gru_hidden_size)),
    np.zeros(gru_hidden_size),
    rng.normal(0.0, 0.2, size=(gru_hidden_size, gru_input_size)),
    rng.normal(0.0, 0.2, size=(gru_hidden_size, gru_hidden_size)),
    np.zeros(gru_hidden_size),
)
gru_previous = np.array([0.3, -0.5, 0.2])
gru_hidden, gru_retain, gru_reset, gru_candidate = gru_step_retain_convention(
    np.array([0.4, -0.1]), gru_previous, gru_parameters,
)

write_convention_z = 1.0 - gru_retain
same_hidden_other_notation = (
    (1.0 - write_convention_z) * gru_previous + write_convention_z * gru_candidate
)
assert np.allclose(gru_hidden, same_hidden_other_notation)
assert np.allclose(1.0 * gru_previous + 0.0 * gru_candidate, gru_previous)
assert np.allclose(0.0 * gru_previous + 1.0 * gru_candidate, gru_candidate)
assert np.all((gru_reset > 0.0) & (gru_reset < 1.0))
```

reset gate 控制生成候选时读取多少旧状态；update gate 控制最终状态在旧值和候选之间插值。它们都不是“删除整个历史”的硬开关，而是逐维、可微的软门。

<div class="ml-stop">
  <strong>停下来推一推 3</strong>
  <p>看到某实现写 h_t=(1-z_t)h_{t-1}+z_t h̃_t。先令 z=0 和 z=1 写出结果，再判断该实现里的 z 表示“保留比例”还是“写入比例”。不要依据变量名猜。</p>
</div>

<MlQuiz kind="记号审计" question="两篇资料的 GRU 更新式把 z 与 1−z 对调，最可靠的核对方法是什么？" :options="['认为其中一篇必错', '比较模型参数文件大小', '代入 z=0 和 z=1，确认门在边界时保留旧状态还是写入候选', '只看作者把它称为 update gate']" :answer="2" explanation="边界行为直接定义门的语义；两套记号可通过 z 与 1−z 互换而等价。" />

## 10. 参数量：门控用更多计算换更可控的状态路径

只计循环层，输入维度 $D$、隐藏维度 $H$：

| 单元 | 仿射组数 | 参数量（含 bias） |
|---|---:|---:|
| Vanilla RNN | 1 | $H(D+H+1)$ |
| GRU | 3 | $3H(D+H+1)$ |
| LSTM | 4 | $4H(D+H+1)$ |

输出头另加 $OH+O$。框架可能把输入权重和循环权重分开，并分别保存两组 bias，因此实际 `state_dict` 计数要按具体 API 核对。

```python
def recurrent_parameter_count(input_size, hidden_size, output_size=0, cell="rnn"):
    gate_factor = {"rnn": 1, "gru": 3, "lstm": 4}[cell]
    recurrent = gate_factor * hidden_size * (input_size + hidden_size + 1)
    output = output_size * hidden_size + output_size if output_size else 0
    return recurrent, output, recurrent + output

rnn_count = recurrent_parameter_count(12, 32, 5, "rnn")
gru_count = recurrent_parameter_count(12, 32, 5, "gru")
lstm_count = recurrent_parameter_count(12, 32, 5, "lstm")

assert rnn_count == (1440, 165, 1605)
assert gru_count == (4320, 165, 4485)
assert lstm_count == (5760, 165, 5925)
assert lstm_count[0] == 4 * rnn_count[0]
assert gru_count[0] == 3 * rnn_count[0]
```

“GRU 参数少所以一定更快”并不成立。目标硬件上的融合内核、序列长度、batch、隐藏维度、数据布局和内存带宽都影响延迟；必须在真实 shape 上 benchmark。

<MlQuiz kind="参数审计" question="D、H 相同且忽略输出头时，标准 LSTM 循环层为何约是 Vanilla RNN 参数量的 4 倍？" :options="['序列固定有四步', '输入门、遗忘门、输出门和候选四组仿射变换各有参数', 'LSTM 必须双向', '细胞状态占四份训练样本']" :answer="1" explanation="倍率来自四组门/候选仿射，不来自序列长度或方向数。" />

## 数值验证

### 11. 对向量 RNN 的完整 BPTT 做有限差分

下面使用最终状态平方误差，同时求 $W_{xh},W_{hh},b,h_0,x_{1:T}$ 的梯度。反向实现按前向实际读取的张量累加，再用独立中心差分检查全部参数坐标。

```python
def rnn_final_loss_and_grad(x, target, parameters, initial_state):
    wxh, whh, bias = parameters
    states, preactivations = vanilla_rnn_forward(x, initial_state, wxh, whh, bias)
    difference = states[-1] - target
    loss = 0.5 * float(difference @ difference)

    gradient_wxh = np.zeros_like(wxh)
    gradient_whh = np.zeros_like(whh)
    gradient_bias = np.zeros_like(bias)
    gradient_x = np.zeros_like(x)
    state_gradient = difference.copy()

    for step in range(len(x) - 1, -1, -1):
        delta = state_gradient * (1.0 - np.tanh(preactivations[step]) ** 2)
        gradient_wxh += np.outer(delta, x[step])
        gradient_whh += np.outer(delta, states[step])
        gradient_bias += delta
        gradient_x[step] = wxh.T @ delta
        state_gradient = whh.T @ delta

    return loss, (gradient_wxh, gradient_whh, gradient_bias), gradient_x, state_gradient

check_x = rng.normal(0.0, 0.3, size=(4, 2))
check_target = rng.normal(0.0, 0.2, size=3)
check_parameters = (
    rng.normal(0.0, 0.25, size=(3, 2)),
    rng.normal(0.0, 0.20, size=(3, 3)),
    rng.normal(0.0, 0.10, size=3),
)
check_initial = rng.normal(0.0, 0.1, size=3)
check_loss, analytic_parameters, analytic_x, analytic_initial = rnn_final_loss_and_grad(
    check_x, check_target, check_parameters, check_initial,
)

def rnn_loss_only(x, target, parameters, initial_state):
    return rnn_final_loss_and_grad(x, target, parameters, initial_state)[0]

epsilon = 1e-6
maximum_parameter_error = 0.0
for parameter_index, parameter in enumerate(check_parameters):
    numerical = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        plus_parameters = [value.copy() for value in check_parameters]
        minus_parameters = [value.copy() for value in check_parameters]
        plus_parameters[parameter_index][index] += epsilon
        minus_parameters[parameter_index][index] -= epsilon
        numerical[index] = (
            rnn_loss_only(check_x, check_target, tuple(plus_parameters), check_initial)
            - rnn_loss_only(check_x, check_target, tuple(minus_parameters), check_initial)
        ) / (2 * epsilon)
    scale = max(1.0, np.max(np.abs(numerical)), np.max(np.abs(analytic_parameters[parameter_index])))
    maximum_parameter_error = max(
        maximum_parameter_error,
        float(np.max(np.abs(numerical - analytic_parameters[parameter_index])) / scale),
    )

numerical_initial = np.zeros_like(check_initial)
for index in np.ndindex(check_initial.shape):
    plus_initial = check_initial.copy()
    minus_initial = check_initial.copy()
    plus_initial[index] += epsilon
    minus_initial[index] -= epsilon
    numerical_initial[index] = (
        rnn_loss_only(check_x, check_target, check_parameters, plus_initial)
        - rnn_loss_only(check_x, check_target, check_parameters, minus_initial)
    ) / (2 * epsilon)

initial_error = np.max(np.abs(numerical_initial - analytic_initial)) / max(
    1.0, np.max(np.abs(numerical_initial)), np.max(np.abs(analytic_initial)),
)
assert maximum_parameter_error < 1e-8
assert initial_error < 1e-8
assert np.isfinite(analytic_x).all()
print("参数 / h0 最大相对误差：", maximum_parameter_error, initial_error)
```

有限差分通过只能证明这个小输入附近的局部导数一致，不能证明长序列训练稳定、mask 正确或数据合同正确。下一组测试应覆盖空白 padding、边界门值、不同 reduction、长短序列和状态复位。

<MlQuiz kind="数值审计" question="RNN 的训练 loss 能下降，但有限差分显示 dW_hh 错误。应相信哪项？" :options="['相信 loss，因为下降就证明导数正确', '相信有限差分，并优先检查跨时刻梯度累加和 h_{t-1} 缓存', '两者都忽略，直接增大隐藏层', '只检查输出 shape']" :answer="1" explanation="错误梯度也可能偶然下降；独立前向扰动能直接审计局部导数。" />

### 11.1 可选 PyTorch 对照

仓库标准验证环境不为本章新增大型框架依赖。若已有 PyTorch，可在 float64 下用相同权重、初始状态和 reduction 对照。PyTorch 的 RNN 权重布局、bias 拆分和输出 shape 必须显式核对。

```python
# 不执行：仅在另行准备的 PyTorch 环境中运行。
import torch

torch_rnn = torch.nn.RNN(input_size=2, hidden_size=3, nonlinearity="tanh", bias=True).double()
with torch.no_grad():
    torch_rnn.weight_ih_l0.copy_(torch.tensor(check_parameters[0]))
    torch_rnn.weight_hh_l0.copy_(torch.tensor(check_parameters[1]))
    torch_rnn.bias_ih_l0.copy_(torch.tensor(check_parameters[2]))
    torch_rnn.bias_hh_l0.zero_()

torch_x = torch.tensor(check_x[:, None, :], dtype=torch.float64, requires_grad=True)
torch_h0 = torch.tensor(check_initial[None, None, :], dtype=torch.float64, requires_grad=True)
torch_target = torch.tensor(check_target, dtype=torch.float64)
torch_output, torch_final = torch_rnn(torch_x, torch_h0)
torch_loss = 0.5 * ((torch_final[0, 0] - torch_target) ** 2).sum()
torch_loss.backward()

assert np.allclose(torch_loss.item(), check_loss, atol=1e-12)
assert np.allclose(torch_rnn.weight_ih_l0.grad.numpy(), analytic_parameters[0], atol=1e-10)
assert np.allclose(torch_rnn.weight_hh_l0.grad.numpy(), analytic_parameters[1], atol=1e-10)
assert np.allclose(torch_h0.grad.numpy()[0, 0], analytic_initial, atol=1e-10)
```

## 12. 最小字符模型：把每步条件分布和 BPTT 跑通

字符模型是检验序列实现的最小案例。训练文本为重复的 `ABCD`，输入字符预测下一字符。它很容易，目的不是语言建模性能，而是同时验收：

- one-hot 输入与每步 Softmax；
- 共享输出头；
- 交叉熵对 logits 的梯度；
- 从最后一步到第一步的状态梯度累加；
- 全局范数裁剪；
- 推理时把上一步预测作为下一步输入。

```python
characters = np.array(list("ABCD"))
character_to_index = {character: index for index, character in enumerate(characters)}
training_text = "ABCD" * 10
training_indices = np.array([character_to_index[value] for value in training_text[:-1]])
training_targets = np.array([character_to_index[value] for value in training_text[1:]])
vocabulary_size = len(characters)
character_hidden_size = 10

character_parameters = {
    "wxh": rng.normal(0.0, 0.12, size=(character_hidden_size, vocabulary_size)),
    "whh": rng.normal(0.0, 0.08, size=(character_hidden_size, character_hidden_size)),
    "bh": np.zeros(character_hidden_size),
    "why": rng.normal(0.0, 0.12, size=(vocabulary_size, character_hidden_size)),
    "by": np.zeros(vocabulary_size),
}

def character_loss_and_grad(input_indices, target_indices, parameters):
    wxh, whh = parameters["wxh"], parameters["whh"]
    bh, why, by = parameters["bh"], parameters["why"], parameters["by"]
    states = [np.zeros(whh.shape[0])]
    probabilities_by_step = []
    one_hot_by_step = []
    loss = 0.0

    for input_index, target_index in zip(input_indices, target_indices):
        one_hot = np.eye(vocabulary_size)[input_index]
        state = np.tanh(wxh @ one_hot + whh @ states[-1] + bh)
        logits = why @ state + by
        shifted_logits = logits - logits.max()
        probabilities = np.exp(shifted_logits) / np.exp(shifted_logits).sum()
        loss -= np.log(probabilities[target_index] + 1e-12)
        one_hot_by_step.append(one_hot)
        states.append(state)
        probabilities_by_step.append(probabilities)

    gradients = {name: np.zeros_like(value) for name, value in parameters.items()}
    next_state_gradient = np.zeros(whh.shape[0])
    normalizer = len(input_indices)
    for step in range(len(input_indices) - 1, -1, -1):
        gradient_logits = probabilities_by_step[step].copy()
        gradient_logits[target_indices[step]] -= 1.0
        gradient_logits /= normalizer
        gradients["why"] += np.outer(gradient_logits, states[step + 1])
        gradients["by"] += gradient_logits
        state_gradient = why.T @ gradient_logits + next_state_gradient
        delta = state_gradient * (1.0 - states[step + 1] ** 2)
        gradients["wxh"] += np.outer(delta, one_hot_by_step[step])
        gradients["whh"] += np.outer(delta, states[step])
        gradients["bh"] += delta
        next_state_gradient = whh.T @ delta
    return loss / normalizer, gradients

initial_character_loss, _ = character_loss_and_grad(
    training_indices, training_targets, character_parameters,
)
accumulators = {name: np.zeros_like(value) for name, value in character_parameters.items()}
gradient_norms = []
for _ in range(260):
    character_loss, gradients = character_loss_and_grad(
        training_indices, training_targets, character_parameters,
    )
    clipped, gradient_norm, _ = clip_by_global_norm(list(gradients.values()), threshold=5.0)
    gradient_norms.append(gradient_norm)
    for (name, parameter), gradient in zip(character_parameters.items(), clipped):
        accumulators[name] += gradient ** 2
        parameter -= 0.25 * gradient / np.sqrt(accumulators[name] + 1e-8)

final_character_loss, _ = character_loss_and_grad(
    training_indices, training_targets, character_parameters,
)

def generate_characters(start_character, count, parameters):
    current = character_to_index[start_character]
    state = np.zeros(parameters["whh"].shape[0])
    generated = [start_character]
    for _ in range(count - 1):
        one_hot = np.eye(vocabulary_size)[current]
        state = np.tanh(parameters["wxh"] @ one_hot + parameters["whh"] @ state + parameters["bh"])
        logits = parameters["why"] @ state + parameters["by"]
        current = int(np.argmax(logits))
        generated.append(characters[current])
    return "".join(generated)

generated_text = generate_characters("A", 16, character_parameters)
assert final_character_loss < 0.15 * initial_character_loss
assert generated_text == "ABCDABCDABCDABCD"
assert np.isfinite(gradient_norms).all()
print("初始/最终字符损失：", initial_character_loss, final_character_loss)
print("生成：", generated_text)
```

训练时把真实前一个字符作为下一步输入，称为 teacher forcing；生成时模型读取自己的预测。二者输入分布不一致会产生 exposure bias。本例是确定性循环，不代表开放文本生成能力，也不能从训练串复现推断泛化。

<MlQuiz kind="案例解释" question="字符模型准确生成训练中的 ABCD 循环，能证明什么？" :options="['证明已获得通用语言理解', '证明这条最小前向、Softmax、BPTT、裁剪和自回归链路能拟合该合成规律', '证明不会出现曝光偏差', '证明隐藏状态保存了无限历史']" :answer="1" explanation="最小合成任务用于实现验收，不提供真实语言泛化或长期记忆证据。" />

## 13. 回到退火炉案例：先建立可审计的递推基线

本章合成标签来自一个指数衰减热负荷。先用同样形状的、但参数并不等同于标签生成器的递推特征建立诊断基线：

$$
s_t=0.86s_{t-1}
+\max(0,-\Delta T_t+0.5\Delta v_t)
+0.35\max(0,\Delta d_t).
$$

它把“历史如何进入判定”暴露给工程审计。阈值只能用训练 campaign 拟合，再冻结到验证和测试。

```python
def thermal_recurrent_feature(sequence, mask, retention=0.86):
    state = 0.0
    for current_input, is_valid in zip(sequence, mask):
        if not is_valid:
            continue
        temperature_error, speed_error, dew_point_error = current_input
        state = (
            retention * state
            + max(0.0, -temperature_error + 0.5 * speed_error)
            + 0.35 * max(0.0, dew_point_error)
        )
    return state

diagnostic_state = np.array([
    thermal_recurrent_feature(sequence, mask)
    for sequence, mask in zip(sequences, valid_mask)
])
candidate_thresholds = np.unique(diagnostic_state[train_rows])
training_errors = np.array([
    np.mean((diagnostic_state[train_rows] > threshold) != review_label[train_rows])
    for threshold in candidate_thresholds
])
selected_threshold = float(candidate_thresholds[np.argmin(training_errors)])

validation_prediction = (diagnostic_state[validation_rows] > selected_threshold).astype(np.int64)
test_prediction = (diagnostic_state[test_rows] > selected_threshold).astype(np.int64)
validation_accuracy = float(np.mean(validation_prediction == review_label[validation_rows]))
test_accuracy = float(np.mean(test_prediction == review_label[test_rows]))

assert validation_accuracy >= 0.75
assert test_accuracy >= 0.75
assert selected_threshold in diagnostic_state[train_rows]
print("验证/测试准确率：", validation_accuracy, test_accuracy)
```

这个基线的作用是暴露数据、时间方向、mask、状态复位和阈值冻结问题，不是声称真实炉温机理就是一阶指数。换成可学习 RNN/LSTM/GRU 后仍应保留它作回归基线，并报告：

- 按 campaign、钢种、规格、有效长度和时间缺口分层的混淆矩阵；
- 复核召回、误报工作量、校准与置信区间；
- 打乱时间、抹去早期窗口和改变 padding 的受控测试；
- 状态复位/携带日志，以及断档、换卷和维护事件；
- 传感器缺失、饱和、漂移和乱序压力测试；
- 模型、预处理、阈值、状态协议与推理运行时版本。

<MlQuiz kind="案例审计" question="递推基线在测试集上达到较高准确率，下面哪项结论成立？" :options="['可立即接管炉温闭环控制', '合成数据主链和阈值冻结流程可运行；真实机理、外部泛化与行动风险仍未被证明', '无需记录状态复位', '可把测试集继续用于选阈值']" :answer="1" explanation="合成验收不等于产线有效性，更不授权高风险控制动作；测试集也不能反复参与选择。" />

## 复杂度与工程边界

### 14. 计算、激活与延迟预算

对 batch $B$、长度 $T$、输入维度 $D$、隐藏维度 $H$，单层 Vanilla RNN 前向乘加量约为

$$
BT(HD+H^2),
$$

GRU 和 LSTM 主干约乘 3 和 4。训练需要保存每步状态、预激活和门值，激活内存约随 $BTH$ 线性增长；完整 BPTT 的串行依赖还限制时间维并行。

```python
def recurrent_system_estimate(batch, steps, input_size, hidden_size, cell="rnn", bytes_per_value=4):
    gate_factor = {"rnn": 1, "gru": 3, "lstm": 4}[cell]
    macs = batch * steps * gate_factor * (hidden_size * input_size + hidden_size ** 2)
    saved_values = batch * steps * hidden_size * gate_factor
    activation_bytes = saved_values * bytes_per_value
    return macs, activation_bytes

rnn_budget = recurrent_system_estimate(64, 240, 48, 128, "rnn")
gru_budget = recurrent_system_estimate(64, 240, 48, 128, "gru")
lstm_budget = recurrent_system_estimate(64, 240, 48, 128, "lstm")

assert gru_budget[0] == 3 * rnn_budget[0]
assert lstm_budget[0] == 4 * rnn_budget[0]
assert lstm_budget[1] == 4 * rnn_budget[1]
print("RNN/GRU/LSTM MAC：", rnn_budget[0], gru_budget[0], lstm_budget[0])
print("估算保存激活 MiB：", *(round(value[1] / 2**20, 2) for value in (rnn_budget, gru_budget, lstm_budget)))
```

该估算忽略输出头、反向临时量、优化器状态、框架 workspace 和 kernel 融合，只用于数量级比较。真实性能验收必须报告硬件、运行时、dtype、batch、长度分布、padding 比例、warm-up、同步方式、P50/P95/P99 延迟和峰值内存。

### 14.1 状态化服务比无状态请求多一层系统风险

若在线服务保存隐藏状态，模型接口不再只是 `request -> response`。至少要定义：

| 风险 | 需要冻结的合同 |
|---|---|
| 状态串线 | 状态键必须包含设备、钢卷、工艺段和模型版本 |
| 乱序/重放 | 每条消息有单调时间戳或序号，明确迟到策略 |
| 服务重启 | 状态快照、重建窗口或明确复位，不能静默换策略 |
| 模型热更新 | 旧状态是否与新参数兼容；通常按版本隔离并复位 |
| 长时间断档 | 超时阈值和复位原因写日志 |
| 多副本路由 | 粘性路由或共享状态存储，并测并发一致性 |

RNN 状态不是可随意缓存的性能优化，而是预测语义的一部分。状态 ID、时间范围、复位原因和模型版本必须可追溯。

### 14.2 双向、多层与 dropout 的边界

- **双向 RNN**：拼接正向和反向状态，增加完整上下文，但不满足在线因果性；
- **多层 RNN**：上层读取下层整条状态序列，增加表示能力和优化难度；
- **dropout**：常加在层间或输出；时间轴上是否共享 mask 会改变噪声过程；
- **归一化**：batch 统计在变长、在线和小 batch 时尤其敏感；
- **混合精度**：需检查 Sigmoid/Tanh 饱和、累加精度、梯度下溢和裁剪前后范数。

<MlQuiz kind="系统边界" question="stateful RNN 服务热更新模型后继续复用旧模型产生的隐藏状态，主要风险是什么？" :options="['参数量自动翻倍', '隐藏状态语义与新参数不兼容，可能造成无提示预测漂移', '序列会自动变成无序集合', 'Softmax 不再归一化']" :answer="1" explanation="隐藏状态由旧转移函数编码；模型版本切换必须隔离、迁移或复位状态。" />

## 15. 失效模式与定位顺序

| 现象 | 优先检查 | 典型根因 | 验证与处置 |
|---|---|---|---|
| 短序列随 padding 值改变 | mask 与最终索引 | padding 更新了状态或计入 loss | 把 padding 改成极端值做不变性测试 |
| 随机切分高、换 campaign 骤降 | 分组键 | 同批漂移泄漏 | 按 campaign 重切并冻结统计量 |
| 换卷首段异常 | 状态键与复位日志 | 上一卷状态串入 | 构造两卷相反历史做隔离测试 |
| 时间打乱性能不变 | 数据或读出 | 模型只用静态均值、时间特征失效 | 受控 permutation / reversal 测试 |
| dW shape 对但差分失败 | 共享累加与缓存 | 漏时刻、错用 $h_t$、转置错误 | 三步 float64 全坐标差分 |
| 长序列 loss 不降 | 梯度和饱和率 | Jacobian 连乘消失 | 记录按 lag 的灵敏度，试门控/短路径 |
| loss 突然 NaN | 全局梯度范数 | 爆炸、学习率或数值溢出 | 先定位首个非有限值，再裁剪和调参 |
| 截断后长期依赖丢失 | $K$ 与 detach | 信用分配跨越窗口 | 按依赖长度验证，增大 K 或改结构 |
| GRU 与参考实现方向相反 | $z$ 约定 | 保留/写入记号对调 | 用 $z=0,1$ 边界测试 |
| LSTM 记忆总被擦除 | forget gate | bias、饱和或输入尺度 | 记录门分布和细胞状态轨迹 |
| 离线很好、在线下降 | 因果合同 | 训练使用了双向或未来特征 | 做严格前缀回放 |
| P99 延迟随长度飙升 | 长度分布与 padding | batch 被最长序列拖累 | bucketing、packing 与真实流量压测 |

推荐定位顺序：

1. 固定三步标量例子，手算前向与 BPTT；
2. 用极端 padding 验证 mask 和最终状态索引；
3. 对 $W_{xh},W_{hh},b,h_0$ 做 float64 中心差分；
4. 用门的 0/1 边界核对 LSTM/GRU 公式和记号；
5. 记录梯度范数、裁剪率、饱和率和长度分层指标；
6. 回放状态复位、乱序、断档、换卷和模型更新；
7. 最后再增加隐藏维度、层数或替换单元。

## 16. 本章可执行检查清单

### 数据与任务

- [ ] 序列实体、时间方向、采样频率、缺失值和复位边界已版本化。
- [ ] 相关窗口按 campaign/设备/实体分组切分。
- [ ] 在线任务没有使用双向状态或未来统计量。
- [ ] padding、mask、length 和目标对齐已分别测试。

### 前向与形状

- [ ] 能区分 many-to-one、对齐 many-to-many 与自回归任务。
- [ ] 参数量不随 $T$ 增长，前向状态数组含明确的 $h_0$。
- [ ] 时间反转和打乱测试能揭示模型是否真正使用顺序。
- [ ] batch-first/time-first、隐藏层数和方向轴均有 shape 断言。

### 反向与稳定性

- [ ] 共享参数梯度跨全部有效时刻累加。
- [ ] 三步手算与 float64 有限差分通过。
- [ ] 梯度裁剪报告裁剪前范数和裁剪比例。
- [ ] 截断长度、状态 carry 与 detach 语义相互独立。

### 门控与部署

- [ ] LSTM 的 $i,f,o,\widetilde c$ 形状和边界行为通过测试。
- [ ] GRU 明确声明 $z$ 是保留还是写入比例。
- [ ] 状态键包含实体和模型版本，复位原因可追溯。
- [ ] 目标硬件上按真实长度分布报告延迟与内存。

## 17. 练习与最小实现

### 练习 1：三步前向与 BPTT

对本章三步标量 RNN：

1. 手算 $a_1,h_1,a_2,h_2,a_3,h_3,L$；
2. 手算 $\delta_3,\delta_2,\delta_1$；
3. 分项写出 $dW_x,dW_h,db,dh_0$；
4. 用 $\epsilon=10^{-6}$ 对全部标量参数做中心差分；
5. 把损失改成 $\sum_t\frac12(h_t-y_t)^2$，说明反向递推多出哪一项。

### 练习 2：变长 batch 与状态边界

实现支持 batch 的 `RNNCell`：

- 输入 `(B,T,D)` 和 mask `(B,T)`；
- 返回全部状态和最后有效状态；
- padding 分别填 `0、99、NaN` 时给出明确策略；
- 测试同卷连续窗口 carry 与跨卷 reset；
- 乱序消息必须拒绝或按已声明策略重放。

### 练习 3：LSTM/GRU 单步与参数对照

为两个单元分别：

1. 写出所有门、候选和状态 shape；
2. 测试门值 0/1 边界；
3. 对照两种 GRU $z$ 记号的等价变换；
4. 手算 $D=2,H=3$ 的参数量；
5. 在已有框架环境对照前向和梯度，记录 bias 布局差异。

### 练习 4：字符模型与梯度轨迹

扩展本章字符模型：

- 把序列长度从 8 扫到 128；
- 每次记录裁剪前全局梯度范数；
- 比较 Vanilla RNN、GRU、LSTM 的损失和长程记忆；
- 加入一个需要跨 40 步复制首字符的任务；
- 分别报告训练时 teacher forcing 和自由生成结果。

### 练习 5：退火炉受控压力测试

对案例基线或你训练的门控模型执行：

1. 时间反转；
2. 早期 5 步抹除；
3. 有效区后追加极端 padding；
4. 在卷边界错误携带状态；
5. 制造 3 步乱序和 10 步断档；
6. 按有效长度和 campaign 报告性能变化。

```python
# TODO：实现支持 mask 的批量 LSTM forward/backward，并完成以下验收。
#
# 1. 输入 (B,T,D)，初始 h0/c0 为 (B,H)，返回全部有效 h/c；
# 2. 四组门一次矩阵乘法完成，并记录切分顺序；
# 3. padding 不更新 h/c、不计入 loss；
# 4. dW 跨 batch 与全部有效时间累加；
# 5. 用 float64 有限差分检查 W、b、x、h0、c0；
# 6. 用 f/i/o 的边界值检查保留、替换、擦除和输出屏蔽。
```

## 18. 本章结论

1. 概率链式法则分解序列联合分布；RNN 额外假设固定维状态足以概括相关历史。
2. 时间展开增加计算节点，不增加参数副本；$W_{xh},W_{hh}$ 在所有时刻共享。
3. RNN 通常顺序敏感；参数共享不等于时间排列不变。
4. 变长 batch 必须同时管理长度、mask、padding 和最终有效状态索引。
5. BPTT 是展开图上的普通反向传播，共享参数梯度必须跨所有有效时刻累加。
6. 长程梯度由局部 Jacobian 连乘决定；激活饱和、矩阵方向和时间变化都影响消失与爆炸。
7. 全局范数裁剪限制爆炸更新，不能恢复消失梯度；截断 BPTT 限制反向距离，不能与状态复位混为一谈。
8. LSTM 用输入、遗忘、输出门和细胞状态构造可调的加法记忆路径，但不保证梯度永不消失。
9. GRU 用 reset 和 update 门直接更新隐藏状态；不同资料的 $z$ 约定必须通过 0/1 边界行为核对。
10. Vanilla RNN、GRU、LSTM 循环层参数倍率约为 1:3:4，真实速度仍需目标硬件实测。
11. 三步手算、有限差分、门边界、字符任务和状态回放共同构成实现验收门禁。
12. stateful 服务的状态键、顺序、断档、复位和模型版本都是预测语义，不是外围缓存细节。

## 来源与许可

序列映射、RNN 状态递推、生成式序列模型、teacher forcing、双向 RNN、时间反向传播、截断 BPTT、梯度消失/爆炸、梯度裁剪、GRU 与 LSTM 的知识边界，参考 Kevin P. Murphy 的 *Probabilistic Machine Learning: An Introduction* 源 PDF 第 533–545 页（§15.1–15.2.7），其中 BPTT 与截断训练集中在源 PDF 第 540–542 页，门控单元集中在源 PDF 第 542–545 页。

从序列标注和时间序列任务到共享递推、时间展开、BPTT、双向/多层 RNN 与 LSTM 的教学线索，交叉参考 CMU 10-301/10-601 Lecture 17 *MAP, CNN, RNN* 源 PDF 第 107–127 页，以及 Lecture 18 *RNN, Transformer* 源 PDF 第 2–15 页。

Christopher Olah 的 *Understanding LSTM Networks* 本地网页快照与 Simeon Kostadinov、Raimi Karim / Vihar Kurama 的 GRU 文章快照，仅用于交叉核对门控直觉、图示信息层级和两类 GRU 更新门记号。快照未提供足以复用正文或图像的明确再分发许可，因此本章没有复制其文字、图形、代码或案例；公式以 Murphy 教材和独立边界推导为准。

连续退火炉 48 卷、12 个 campaign、三通道热历史、标签生成规则、递推诊断基线、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作。教学合成数据不代表真实炉温机理、材料性能、传感器精度、质量标准、复核阈值、放行规范或闭环控制策略；PyTorch 对照明确标为可选未执行内容，仓库标准验证环境未为此新增大型依赖。

<MlLessonProgress
  lesson-id="16"
  next-href="/courses/machine-learning/17-transformer"
  next-label="第 17 章：Transformer 与自注意力"
/>
