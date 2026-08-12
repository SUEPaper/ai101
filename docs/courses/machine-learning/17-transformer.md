---
course: machine-learning
edition: industrial
title: Transformer 与自注意力
description: 以压缩机站保护事件根因分诊为贯穿案例，从内容寻址推导缩放点积、多头自注意力、因果与 padding Mask、位置编码、残差和 LayerNorm，并用 NumPy、有限差分、泄漏测试与复杂度预算建立实现验收门禁。
outline: [2, 3]
---

<MlLessonHero
  number="17"
  title="Transformer 与自注意力"
  question="一次跳闸前可能同时出现润滑压低、振动升高、电流突增和远程复位；怎样让模型按当前诊断问题直接读取整条事件链，又保证在线位置绝不偷看未来、不同注意力头不混淆、长序列成本可预算？"
  duration="建议 720–840 分钟"
  :outcomes="[
    '从内容寻址推导 Query、Key、Value、缩放点积与逐行 Softmax，手算单头注意力并标注全部矩阵形状',
    '正确合并因果与 padding Mask，推导注意力反向传播，用有限差分和未来改写测试排除梯度错误与信息泄漏',
    '实现多头自注意力、正弦位置编码、Pre-Norm Transformer 层和工业事件分诊基线，完成复杂度、数据合同与部署验收'
  ]"
/>

第 16 章用递推状态把历史逐步压缩为 $\mathbf h_t$。Transformer 选择另一条路：每个位置都可以对可见位置做一次**按内容寻址**，直接聚合当前任务需要的信息。路径变短、训练可并行，但代价是显式构造 $T\times T$ 的关系矩阵。

本章不把“没有递归”误写成“没有顺序”，也不把彩色注意力热图当作因果解释。我们将反复核对四个问题：

1. 当前 Query 在找什么，Key 用什么特征接受匹配，Value 最终携带什么；
2. 哪些位置在业务上可见，Mask 是否在 Softmax 前生效；
3. token 内容相同但位置不同，模型凭什么区分先后；
4. 训练、离线复盘和在线自回归的计算合同是否一致。

## 贯穿案例：压缩机站保护事件根因分诊

天然气压缩机站的一次保护动作会产生多源异步事件：PLC 启停、润滑系统低压、振动监测告警、电机电流突增、转速下降、保护跳闸、远程复位和人工确认。任务不是让模型直接下达控制命令，而是对一个已闭合事件窗口做**离线根因分诊**，将工单优先路由到润滑、机械、电气或控制逻辑四个专业组。

同一模型若被改造成在线告警辅助，则位置 $t$ 只能读取 $1{:}t$；事故结束后的“人工确认”不能反向帮助事故发生前的判断。这一差别正是因果 Mask 的业务含义。

### 事件合同

| 字段 | 张量形状与语义 | 生产约束 |
|---|---|---|
| `event_type[t]` | 离散事件类型 | 词表版本固定；未知事件进入 `UNK`，不能静默映射成正常 |
| `source_id[t]` | 设备或子系统标识 | 资产更名和设备替换必须经过主数据映射 |
| `severity[t]` | 标准化严重度 | 量程、方向和缺失策略随模型版本固化 |
| `relative_time[t]` | 相对窗口起点的时间 | 真实时间间隔不能只靠序号假装等距 |
| `valid_mask[t]` | 是否为真实 token | padding 不参与注意力、损失和指标 |
| `campaign_id` | 连续运行批次 | 同一 campaign 的相似事故不可跨数据集 |
| `root_cause` | 离线复核标签 | 允许“待确认”；教学数据暂用四类完备标签 |

窗口从“启动或上次复位”开始，到“人工确认或超时”结束。过长窗口不能任意截掉早期先兆；过短窗口也不能把跳闸后的处理事件拼进下一次事故。真正上线前还要记录事件时钟源、乱序容忍区间、重复消息去重键和窗口修订策略。

### 1. 先按 campaign 切分，再做 padding

下面生成 18 个运行 campaign、72 条教学事件链。四类模式仅用于验证算法，不代表真实设备机理、保护定值或维修规则。

```python
import numpy as np

rng = np.random.default_rng(20260805)
event_names = np.array([
    "START", "VIB_HIGH", "LUBE_LOW", "TEMP_HIGH",
    "RPM_DROP", "TRIP", "ACK", "CURRENT_SURGE",
    "VOLTAGE_DIP", "REMOTE_RESET", "HEARTBEAT", "UNK",
])
cause_names = np.array(["润滑", "机械", "电气", "控制逻辑"])
patterns = [
    [0, 2, 2, 1, 4, 5, 6],
    [0, 1, 1, 3, 4, 5, 6],
    [0, 7, 8, 7, 4, 5, 6],
    [0, 10, 9, 10, 5, 9, 6],
]

campaign_count = 18
incidents_per_campaign = 4
incident_count = campaign_count * incidents_per_campaign
maximum_tokens = 10

campaign_id = np.repeat(np.arange(campaign_count), incidents_per_campaign)
root_cause = np.tile(np.arange(4), campaign_count)
event_type = np.full((incident_count, maximum_tokens), -1, dtype=np.int64)
source_id = np.full_like(event_type, -1)
severity = np.zeros((incident_count, maximum_tokens), dtype=np.float64)
relative_time = np.zeros_like(severity)
valid_mask = np.zeros_like(severity, dtype=bool)

base_severity = np.array([.20, .84, .96, .68, .61, 1.00, .25, .98, .88, .72, .08, .15])
source_lookup = np.array([0, 2, 1, 1, 0, 0, 4, 3, 3, 4, 0, 5])

for row in range(incident_count):
    sequence = list(patterns[root_cause[row]])
    if row % 3 == 0:
        sequence.insert(2, 10)
    if row % 5 == 0:
        sequence.insert(-1, 11)
    length = len(sequence)
    event_type[row, :length] = sequence
    source_id[row, :length] = source_lookup[sequence]
    severity[row, :length] = np.clip(
        base_severity[sequence] + rng.normal(0.0, 0.025, size=length), 0.0, 1.0
    )
    relative_time[row, :length] = np.cumsum(rng.uniform(0.4, 2.2, size=length))
    valid_mask[row, :length] = True

campaign_order = rng.permutation(campaign_count)
train_campaigns = campaign_order[:12]
validation_campaigns = campaign_order[12:15]
test_campaigns = campaign_order[15:]
train_rows = np.isin(campaign_id, train_campaigns)
validation_rows = np.isin(campaign_id, validation_campaigns)
test_rows = np.isin(campaign_id, test_campaigns)

assert event_type.shape == valid_mask.shape == (72, 10)
assert np.all(event_type[~valid_mask] == -1)
assert set(train_campaigns).isdisjoint(validation_campaigns)
assert set(train_campaigns).isdisjoint(test_campaigns)
assert set(validation_campaigns).isdisjoint(test_campaigns)
assert (train_rows.sum(), validation_rows.sum(), test_rows.sum()) == (48, 12, 12)
print("训练/验证/测试事件链：", train_rows.sum(), validation_rows.sum(), test_rows.sum())
print("有效长度范围：", valid_mask.sum(axis=1).min(), valid_mask.sum(axis=1).max())
```

这里有三层不同的边界：`campaign_id` 防止切分泄漏，`valid_mask` 防止 padding 污染，因果 Mask 防止未来事件污染过去位置。任何一层正确都不能替代另外两层。

<MlQuiz kind="数据合同" question="离线事故复盘允许读取完整事件窗，但同一模型还要输出每分钟在线风险。最关键的训练合同是什么？" :options="['所有任务都使用双向全注意力', '在线输出位置只能读取当时已经到达的事件，离线任务可另设全上下文头或模型', '只要最终测试集独立就可读取未来', '把未来事件严重度归零即可']" :answer="1" explanation="切分独立不能消除单条序列内部的未来泄漏；在线位置必须使用因果可见域，离线复盘可以声明另一套可见域。" />

### 2. token 不是一个整数：表示层必须保留字段语义

`event_type=5` 只是查表地址，不含“跳闸”的自然距离。工程上通常把事件类型嵌入、设备嵌入、严重度投影、时间间隔编码相加或拼接，再投影到统一的 $d_{\text{model}}$。无论哪种方案，都要先声明 padding 行如何处理。

```python
event_embedding = rng.normal(0.0, 0.20, size=(len(event_names) + 1, 8))
source_embedding = rng.normal(0.0, 0.15, size=(6 + 1, 8))
event_padding_index = len(event_names)
source_padding_index = 6

safe_event_type = np.where(valid_mask, event_type, event_padding_index)
safe_source_id = np.where(valid_mask, source_id, source_padding_index)
severity_direction = rng.normal(0.0, 0.12, size=8)
time_direction = rng.normal(0.0, 0.08, size=8)

token_representation = (
    event_embedding[safe_event_type]
    + source_embedding[safe_source_id]
    + severity[..., None] * severity_direction
    + np.log1p(relative_time)[..., None] * time_direction
)
token_representation = np.where(valid_mask[..., None], token_representation, 0.0)

assert token_representation.shape == (72, 10, 8)
assert np.all(token_representation[~valid_mask] == 0.0)
```

把 padding 表示置零仍不够：线性层可能有 bias，位置编码也会让 padding 行再次非零。因此 Key 侧 Mask 与损失 Mask 仍然不可省略。

## 2. 内容寻址：Query、Key、Value 各司其职

精确地址读取要求调用者已经知道位置；注意力则用“我要找什么”去匹配“每个位置提供什么”。对查询 $\mathbf q_i$ 和键 $\mathbf k_j$，相似度生成读取权重：

$$
s_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_k}},\qquad
a_{ij}=\frac{\exp(s_{ij})}{\sum_{\ell\in\mathcal V_i}\exp(s_{i\ell})},
$$

其中 $\mathcal V_i$ 是查询 $i$ 的可见 Key 集合。输出为

$$
\mathbf z_i=\sum_{j\in\mathcal V_i}a_{ij}\mathbf v_j.
$$

只要该行至少有一个合法 Key，就有 $a_{ij}\ge 0$ 且 $\sum_j a_{ij}=1$，所以 $\mathbf z_i$ 位于 Value 的凸包内。

- **Query**：当前计算需要什么，例如“这次 TRIP 之前是否有润滑证据”；
- **Key**：每个事件愿意用什么特征接受匹配，例如设备域、事件类型和时间邻近性；
- **Value**：匹配成功后真正被搬运的内容，例如根因证据向量或处置紧迫度；
- **Attention 权重**：当前前向计算中的路由系数，不自动等于因果贡献、特征重要性或人的诊断理由。

<AttentionRoutingConsoleLab />

<MlQuiz kind="概念辨析" question="固定 Q 与 K，只把 V 从“根因证据”通道切换成“处置紧迫度”通道，会发生什么？" :options="['注意力权重不变，输出内容改变', '注意力权重和输出都不变', '注意力权重改变，输出一定不变', '序列长度改变']" :answer="0" explanation="QKᵀ 决定地址分布，V 决定按该分布搬运的内容；这正是 Key 与 Value 分工的意义。" />

<div class="ml-stop"><strong>停下来推一推 1</strong>有两个 Value：v₁=(2,0)、v₂=(0,4)，注意力权重为 (0.75,0.25)。先手算输出，再说明为什么它不可能超出两个 Value 形成的线段。</div>

## 3. 从单个查询到矩阵：先把形状写全

设一条事件链有 $T$ 个 token，输入

$$
\mathbf X\in\mathbb R^{T\times d_{\text{model}}}.
$$

单头投影为

$$
\mathbf Q=\mathbf X\mathbf W_Q\in\mathbb R^{T\times d_k},\quad
\mathbf K=\mathbf X\mathbf W_K\in\mathbb R^{T\times d_k},\quad
\mathbf V=\mathbf X\mathbf W_V\in\mathbb R^{T\times d_v}.
$$

于是

$$
\underbrace{\mathbf S}_{T\times T}
=\frac{
\overbrace{\mathbf Q}^{T\times d_k}
\overbrace{\mathbf K^\top}^{d_k\times T}
}{\sqrt{d_k}},\qquad
\underbrace{\mathbf Z}_{T\times d_v}
=\underbrace{\operatorname{softmax}_{\text{row}}(\mathbf S)}_{T\times T}
\underbrace{\mathbf V}_{T\times d_v}.
$$

Softmax 必须逐行执行：每一行属于一个 Query，对全部可见 Key 归一化。按列归一化会回答完全不同的问题。

### 3.1 三个事件的手算

```python
manual_q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
manual_k = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
manual_v = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

manual_scores = manual_q @ manual_k.T / np.sqrt(2.0)
manual_exp = np.exp(manual_scores - manual_scores.max(axis=1, keepdims=True))
manual_weights = manual_exp / manual_exp.sum(axis=1, keepdims=True)
manual_output = manual_weights @ manual_v

assert manual_scores.shape == manual_weights.shape == (3, 3)
assert manual_output.shape == (3, 2)
assert np.allclose(manual_weights.sum(axis=1), 1.0)
assert np.all(manual_weights >= 0.0)
print(np.round(manual_weights, 3))
print(np.round(manual_output, 3))
```

第一行 Query 与第一个 Key 同向，因而第一列权重较大；但输出第一行不是 Key，而是三个 **Value** 的加权和。把 Key 当输出内容是实现注意力时最常见的概念混淆之一。

<MlQuiz kind="手算" question="Q 为 5×8，K 为 7×8，V 为 7×3，这个交叉注意力的权重矩阵和输出形状分别是什么？" :options="['5×7 与 5×3', '7×5 与 7×3', '5×8 与 7×3', '8×8 与 5×7']" :answer="0" explanation="QKᵀ 得到每个 5 个查询对 7 个键的分数，再乘 7×3 的 V，输出沿用查询长度 5。" />

### 3.2 稳定 Softmax 是算子合同，不是优化技巧

对一行 logits $\mathbf s$，减去任意公共常数不改变 Softmax：

$$
\frac{e^{s_j-c}}{\sum_\ell e^{s_\ell-c}}
=\frac{e^{-c}e^{s_j}}{e^{-c}\sum_\ell e^{s_\ell}}
=\frac{e^{s_j}}{\sum_\ell e^{s_\ell}}.
$$

令 $c=\max_j s_j$ 可使最大指数为 1，避免大正数溢出。

```python
def row_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    row_maximum = np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(logits - row_maximum)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)

large_logits = np.array([[1200.0, 1198.0, 1195.0]])
stable_probabilities = row_softmax(large_logits)
assert np.isfinite(stable_probabilities).all()
assert np.allclose(stable_probabilities.sum(axis=1), 1.0)
```

若一整行都被 Mask 成 $-\infty$，`max` 和减法会产生未定义结果。工业实现应在数据合同处保证每个有效 Query 至少有一个合法 Key，或显式定义空行输出；不能靠 `nan_to_num` 掩盖错误。

## 4. 为什么要除以 $\sqrt{d_k}$

设 $q_r,k_r$ 独立、均值为 0、方差为 1。未缩放点积为

$$
u=\mathbf q^\top\mathbf k=\sum_{r=1}^{d_k}q_rk_r.
$$

因为 $\mathbb E[q_rk_r]=0$、$\operatorname{Var}(q_rk_r)=1$，在独立近似下

$$
\mathbb E[u]=0,\qquad
\operatorname{Var}(u)=d_k,\qquad
\operatorname{Std}(u)=\sqrt{d_k}.
$$

维度增大时，未缩放 logits 会把 Softmax 推向近似 one-hot。此时大部分位置的梯度接近零。除以 $\sqrt{d_k}$ 后，logit 标准差回到 $O(1)$；它控制的是统计尺度，不是为了“让权重更平均”的硬约束。

```python
rng_scale = np.random.default_rng(1704)
scale_statistics = []
for key_dimension in (4, 16, 64, 256):
    query_samples = rng_scale.normal(size=(30000, key_dimension))
    key_samples = rng_scale.normal(size=(30000, key_dimension))
    raw_dot = np.sum(query_samples * key_samples, axis=1)
    scaled_dot = raw_dot / np.sqrt(key_dimension)
    scale_statistics.append((key_dimension, raw_dot.std(), scaled_dot.std()))

for key_dimension, raw_std, scaled_std in scale_statistics:
    assert 0.90 < raw_std / np.sqrt(key_dimension) < 1.10
    assert 0.90 < scaled_std < 1.10
print([(d, round(raw, 2), round(scaled, 2)) for d, raw, scaled in scale_statistics])
```

“除以 $\sqrt{d_k}$”与可学习温度、Query/Key 范数和 LayerNorm 共同影响最终熵。它不是对尖锐注意力的充分防护，也不保证训练稳定。

<MlQuiz kind="原理推导" question="若 Q、K 每个分量独立且方差都是 4，点积含 64 项，点积标准差约为多少？要缩回约 1 应除以多少？" :options="['8，除以 8', '32，除以 32', '64，除以 64', '256，除以 16']" :answer="1" explanation="每个乘积的方差为 4×4=16，64 项之和方差为 1024，标准差为 32；一般缩放还取决于投影后的实际方差。" />

## 5. Mask：先定义可见域，再做概率归一化

Mask 不是绘图层上的灰色格子，而是概率分布的支持集。令 $M_{ij}$ 为

$$
M_{ij}=
\begin{cases}
0,&j\in\mathcal V_i,\\
-\infty,&j\notin\mathcal V_i.
\end{cases}
$$

缩放点积注意力写成

$$
\operatorname{Attention}(\mathbf Q,\mathbf K,\mathbf V;\mathbf M)
=\operatorname{softmax}_{\text{row}}\left(
\frac{\mathbf Q\mathbf K^\top}{\sqrt{d_k}}+\mathbf M
\right)\mathbf V.
$$

### 5.1 两类 Mask 不能混为一个名字

- **因果 Mask**：对查询 $i$ 屏蔽 $j>i$，保证在线前缀不读未来；
- **padding Mask**：对所有查询屏蔽不存在的 Key；
- **Query 有效性**：padding Query 的输出通常还要在层外清零，并从损失中排除；
- **任务 Mask**：某些业务还会屏蔽跨设备、跨窗口或不可信来源，需要单独说明语义。

```python
def scaled_dot_product_attention(query, key, value, allow_mask):
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    allow_mask = np.asarray(allow_mask, dtype=bool)
    if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("教学实现要求 Q、K、V 都是二维矩阵")
    if query.shape[1] != key.shape[1] or key.shape[0] != value.shape[0]:
        raise ValueError("Q/K 特征维或 K/V token 数不一致")
    if allow_mask.shape != (query.shape[0], key.shape[0]):
        raise ValueError("Mask 应为 (query_tokens, key_tokens)")
    if np.any(allow_mask.sum(axis=1) == 0):
        raise ValueError("每个 Query 至少需要一个合法 Key")

    scores = query @ key.T / np.sqrt(query.shape[1])
    masked_scores = np.where(allow_mask, scores, -np.inf)
    row_maximum = np.max(masked_scores, axis=1, keepdims=True)
    exponentials = np.where(allow_mask, np.exp(masked_scores - row_maximum), 0.0)
    weights = exponentials / exponentials.sum(axis=1, keepdims=True)
    return weights @ value, weights

mask_rng = np.random.default_rng(1705)
mask_q = mask_rng.normal(size=(6, 4))
mask_k = mask_rng.normal(size=(6, 4))
mask_v = mask_rng.normal(size=(6, 3))
causal_allow = np.tril(np.ones((6, 6), dtype=bool))
mask_output, mask_weights = scaled_dot_product_attention(
    mask_q, mask_k, mask_v, causal_allow
)

assert mask_output.shape == (6, 3)
assert np.allclose(mask_weights.sum(axis=1), 1.0)
assert np.all(mask_weights[~causal_allow] == 0.0)
```

### 5.2 未来改写测试比“看起来是三角形”更强

正确的因果实现应满足前缀不变性：任意改写位置 $m$ 之后的 Key 和 Value，位置 $i<m$ 的输出都不变。若模型输入投影逐 token 独立，改写未来 Query 也不影响过去位置。

```python
future_q = mask_q.copy()
future_k = mask_k.copy()
future_v = mask_v.copy()
future_q[4:] += 100.0
future_k[4:] -= 80.0
future_v[4:] *= -50.0

changed_output, changed_weights = scaled_dot_product_attention(
    future_q, future_k, future_v, causal_allow
)
assert np.allclose(changed_output[:4], mask_output[:4], atol=1e-12)
assert np.allclose(changed_weights[:4], mask_weights[:4], atol=1e-12)
assert not np.allclose(changed_output[4:], mask_output[4:])
```

这个测试应在完整模型上重复，包括 token 化、位置编码、归一化、缓存和输出头。单独测试注意力核通过，不代表上游特征工程没有用全窗口均值或事故后字段。

<CausalMaskLeakScannerLab />

### 5.3 为什么 Softmax 后清零仍然错误

设原始概率为 $p_j=e^{s_j}/\sum_k e^{s_k}$。若事后把未来位置置零，合法位置保留的仍是 $p_j$，其和为 $1-\sum_{j\notin\mathcal V_i}p_j<1$。未来 logit 已经进入分母，改变未来仍会改变过去合法权重。

```python
bad_scores = mask_q @ mask_k.T / np.sqrt(mask_q.shape[1])
bad_weights = row_softmax(bad_scores)
post_mask_weights = np.where(causal_allow, bad_weights, 0.0)

assert np.all(post_mask_weights[~causal_allow] == 0.0)
assert np.any(post_mask_weights.sum(axis=1) < 0.999)
assert np.allclose(post_mask_weights[-1].sum(), 1.0)
```

事后再重新归一化可以在纯数学上恢复同一行结果，但既浪费计算，也容易让全遮罩行、低精度下溢和广播错误漏过测试。直接在 logits 上定义支持集更清晰。

<MlQuiz kind="代码审查" question="一段代码先对完整 logits 做 Softmax，再把上三角权重乘 0。热图中未来格全为 0，为什么仍不合格？" :options="['因为合法权重已经被未来 logits 污染，且行和可能小于 1', '因为 Softmax 不能用于矩阵', '因为因果 Mask 应屏蔽过去', '因为 Value 必须全部为正']" :answer="0" explanation="未来项参与了 Softmax 分母；只看最终灰格无法发现归一化污染，必须检查行和与未来改写不变性。" />

<div class="ml-stop"><strong>停下来推一推 2</strong>某行未遮罩 Softmax 权重为 (0.2,0.3,0.5)，最后一项是未来。先清零未来得到什么、行和是多少？若重新归一化，前两项变成多少？</div>

## 6. 注意力如何反向传播

把一行 Softmax 的上游梯度记作 $\mathbf g_a=\partial L/\partial\mathbf a$。Softmax Jacobian 为

$$
\frac{\partial a_i}{\partial s_j}=a_i(\delta_{ij}-a_j).
$$

不必显式构造 $T\times T$ Jacobian，可以写成向量形式

$$
\mathbf g_s
=\mathbf a\odot\left(
\mathbf g_a-(\mathbf g_a^\top\mathbf a)\mathbf 1
\right).
$$

对整批行，若 $\mathbf G=\partial L/\partial\mathbf Z$，则

$$
\frac{\partial L}{\partial\mathbf V}=\mathbf A^\top\mathbf G,
\qquad
\frac{\partial L}{\partial\mathbf A}=\mathbf G\mathbf V^\top,
$$

$$
\frac{\partial L}{\partial\mathbf Q}
=\frac{\mathbf G_S\mathbf K}{\sqrt{d_k}},
\qquad
\frac{\partial L}{\partial\mathbf K}
=\frac{\mathbf G_S^\top\mathbf Q}{\sqrt{d_k}}.
$$

被 Mask 的概率为 0，相应 score 梯度也应为 0。下面用中心差分同时核对 Q、K、V。

```python
gradient_rng = np.random.default_rng(1706)
gradient_q = gradient_rng.normal(size=(4, 3))
gradient_k = gradient_rng.normal(size=(4, 3))
gradient_v = gradient_rng.normal(size=(4, 2))
gradient_allow = np.tril(np.ones((4, 4), dtype=bool))
upstream = gradient_rng.normal(size=(4, 2))

gradient_z, gradient_a = scaled_dot_product_attention(
    gradient_q, gradient_k, gradient_v, gradient_allow
)
gradient_dv = gradient_a.T @ upstream
gradient_da = upstream @ gradient_v.T
gradient_ds = gradient_a * (
    gradient_da - np.sum(gradient_da * gradient_a, axis=1, keepdims=True)
)
gradient_dq = gradient_ds @ gradient_k / np.sqrt(gradient_q.shape[1])
gradient_dk = gradient_ds.T @ gradient_q / np.sqrt(gradient_q.shape[1])

def attention_scalar_loss(query, key, value):
    output, _ = scaled_dot_product_attention(query, key, value, gradient_allow)
    return float(np.sum(output * upstream))

def central_difference(array, loss_function, epsilon=1e-6):
    numerical = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        positive = array.copy()
        negative = array.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        numerical[index] = (loss_function(positive) - loss_function(negative)) / (2 * epsilon)
    return numerical

numeric_dq = central_difference(
    gradient_q, lambda candidate: attention_scalar_loss(candidate, gradient_k, gradient_v)
)
numeric_dk = central_difference(
    gradient_k, lambda candidate: attention_scalar_loss(gradient_q, candidate, gradient_v)
)
numeric_dv = central_difference(
    gradient_v, lambda candidate: attention_scalar_loss(gradient_q, gradient_k, candidate)
)

assert np.max(np.abs(numeric_dq - gradient_dq)) < 1e-7
assert np.max(np.abs(numeric_dk - gradient_dk)) < 1e-7
assert np.max(np.abs(numeric_dv - gradient_dv)) < 1e-7
assert np.all(gradient_ds[~gradient_allow] == 0.0)
```

有限差分应使用 float64、小尺寸和远离分段边界的随机数。它是局部实现门禁，不替代端到端任务测试。

<MlQuiz kind="原理推导" question="在 Z=AV 中，若某个 Value 被多个 Query 读取，它的梯度为什么会累加？" :options="['因为 dV=AᵀG 汇总所有 Query 路径', '因为 V 自动等于 Q', '因为 Softmax 按列归一化', '因为 Mask 会复制 Value']" :answer="0" explanation="共享 Value 节点通过多条加权路径影响损失；矩阵乘法 AᵀG 正好把这些路径贡献求和。" />

## 7. 自注意力：同一序列同时产生 Q、K、V

自注意力令

$$
\mathbf Q=\mathbf X\mathbf W_Q,\qquad
\mathbf K=\mathbf X\mathbf W_K,\qquad
\mathbf V=\mathbf X\mathbf W_V.
$$

“自”只说明三者来自同一组 token，不表示 $\mathbf W_Q=\mathbf W_K=\mathbf W_V$。三套投影允许同一事件分别表达提问、匹配和被搬运内容。

若没有位置编码且没有特殊 Mask，对 token 同时施加置换矩阵 $\mathbf P$：

$$
\mathbf X'=\mathbf P\mathbf X,
$$

则注意力满足

$$
\operatorname{Attn}(\mathbf P\mathbf X)
=\mathbf P\operatorname{Attn}(\mathbf X).
$$

这是**置换等变**：输入换序，输出跟着换序，而不是输出完全不变。

```python
permutation_rng = np.random.default_rng(1707)
permutation_x = permutation_rng.normal(size=(6, 5))
permutation_wq = permutation_rng.normal(size=(5, 4))
permutation_wk = permutation_rng.normal(size=(5, 4))
permutation_wv = permutation_rng.normal(size=(5, 3))
full_allow = np.ones((6, 6), dtype=bool)

def unpositioned_self_attention(inputs):
    return scaled_dot_product_attention(
        inputs @ permutation_wq,
        inputs @ permutation_wk,
        inputs @ permutation_wv,
        full_allow,
    )[0]

permutation = np.array([3, 0, 5, 2, 1, 4])
original_self_output = unpositioned_self_attention(permutation_x)
permuted_self_output = unpositioned_self_attention(permutation_x[permutation])
assert np.allclose(permuted_self_output, original_self_output[permutation])
```

因果 Mask 本身已经注入“前后”结构，所以因果注意力即使不加位置向量也不再对任意置换等变；但仅凭“前缀大小”通常不足以精确表达距离、周期和绝对位置。

## 8. 位置编码：把顺序提供给内容算子

经典正弦位置编码对位置 $p$ 和偶数通道 $2i$ 定义

$$
\operatorname{PE}(p,2i)=\sin\left(p/10000^{2i/d}\right),
$$

$$
\operatorname{PE}(p,2i+1)=\cos\left(p/10000^{2i/d}\right).
$$

每对正弦和余弦的相位随位置平滑变化。固定偏移 $r$ 的位置向量关系只依赖 $r$；这为模型表达相对位移提供了结构线索。输入通常写成

$$
\mathbf H^{(0)}=\mathbf E_{\text{token}}+\mathbf E_{\text{position}},
$$

两者必须同为 $d_{\text{model}}$ 维。相加不是唯一选择，旋转位置编码、相对位置偏置和可学习绝对嵌入各有外推与实现边界；本章先掌握固定正弦方案。

```python
def sinusoidal_position_encoding(length, model_dimension):
    if model_dimension % 2 != 0:
        raise ValueError("教学实现要求 d_model 为偶数")
    position = np.arange(length, dtype=np.float64)[:, None]
    inverse_frequency = 1.0 / (
        10000.0 ** (np.arange(0, model_dimension, 2) / model_dimension)
    )
    angles = position * inverse_frequency[None, :]
    encoding = np.zeros((length, model_dimension), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)
    return encoding

position_encoding = sinusoidal_position_encoding(16, 8)
assert position_encoding.shape == (16, 8)
assert np.allclose(np.sum(position_encoding ** 2, axis=1), 4.0)
offset_two_dot_products = np.array([
    position_encoding[position] @ position_encoding[position + 2]
    for position in range(14)
])
assert np.max(offset_two_dot_products) - np.min(offset_two_dot_products) < 1e-12
```

工业事件通常不等距。仅用 token 序号会把相隔 20 ms 和 20 min 的事件当成同一步长；可额外编码真实时间差、班次或工况阶段，但这些字段必须只来自当时已知信息。

<MultiHeadPositionBoardLab />

<MlQuiz kind="概念辨析" question="不加位置编码的全自注意力看到两个 token 被交换时，最准确的描述是什么？" :options="['输出按同一置换跟着交换', '输出一定完全不变', '模型自动从数组下标读取位置', 'Q 与 K 会自动变成递归状态']" :answer="0" explanation="共享逐 token 投影与全注意力对置换等变；若任务依赖顺序，需要位置结构或有序 Mask。" />

## 9. 多头注意力：在多个投影子空间并行寻址

单头只能用一套相似度组织所有关系。多头注意力为每个头 $r$ 使用独立投影：

$$
\operatorname{head}_r
=\operatorname{Attention}(
\mathbf X\mathbf W_Q^{(r)},
\mathbf X\mathbf W_K^{(r)},
\mathbf X\mathbf W_V^{(r)}).
$$

拼接后再用输出矩阵混合：

$$
\operatorname{MHA}(\mathbf X)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)\mathbf W_O.
$$

常见设置为 $d_k=d_v=d_{\text{model}}/h$。若总维度固定，增加头数会减小每头维度，Q/K/V/O 的主参数量仍约为 $4d_{\text{model}}^2$，并不会按头数线性增加。

### 9.1 批量多头形状

设输入为 $(B,T,d)$，投影并转轴后 Q、K、V 为 $(B,h,T,d_h)$，分数和权重为 $(B,h,T,T)$，拼接后回到 $(B,T,h d_h)$。

```python
multihead_rng = np.random.default_rng(1709)
batch_size, token_count, model_dimension, head_count = 2, 6, 8, 2
head_dimension = model_dimension // head_count
multihead_x = multihead_rng.normal(size=(batch_size, token_count, model_dimension))
multihead_wq = multihead_rng.normal(scale=.2, size=(head_count, model_dimension, head_dimension))
multihead_wk = multihead_rng.normal(scale=.2, size=(head_count, model_dimension, head_dimension))
multihead_wv = multihead_rng.normal(scale=.2, size=(head_count, model_dimension, head_dimension))
multihead_wo = multihead_rng.normal(scale=.2, size=(model_dimension, model_dimension))

def multihead_self_attention(inputs, wq, wk, wv, wo, allow_mask):
    batch, tokens, model = inputs.shape
    heads, model_again, head_dim = wq.shape
    if model != model_again or model != heads * head_dim:
        raise ValueError("要求 d_model = h × d_head")
    query = np.einsum("btd,hdk->bhtk", inputs, wq)
    key = np.einsum("btd,hdk->bhtk", inputs, wk)
    value = np.einsum("btd,hdk->bhtk", inputs, wv)
    scores = np.einsum("bhik,bhjk->bhij", query, key) / np.sqrt(head_dim)
    allowed = np.asarray(allow_mask, dtype=bool)[None, None, :, :]
    masked_scores = np.where(allowed, scores, -np.inf)
    row_maximum = np.max(masked_scores, axis=-1, keepdims=True)
    exponentials = np.where(allowed, np.exp(masked_scores - row_maximum), 0.0)
    weights = exponentials / exponentials.sum(axis=-1, keepdims=True)
    per_head = np.einsum("bhij,bhjk->bhik", weights, value)
    concatenated = per_head.transpose(0, 2, 1, 3).reshape(batch, tokens, model)
    return concatenated @ wo, weights

multihead_allow = np.tril(np.ones((token_count, token_count), dtype=bool))
multihead_output, multihead_weights = multihead_self_attention(
    multihead_x, multihead_wq, multihead_wk, multihead_wv, multihead_wo, multihead_allow
)

assert multihead_output.shape == (2, 6, 8)
assert multihead_weights.shape == (2, 2, 6, 6)
assert np.allclose(multihead_weights.sum(axis=-1), 1.0)
assert np.all(multihead_weights[..., ~multihead_allow] == 0.0)
```

最后一条布尔索引断言刻意检查所有 batch 和 head 的禁止格。生产实现还要分别测试 `(B,1,1,T)` padding Mask、`(1,1,T,T)` 因果 Mask 以及组合后的广播形状。

<MlQuiz kind="手算" question="输入形状为 B=4、T=128、d_model=256，使用 8 个等宽头。每头 Q 的形状与注意力权重总形状是什么？" :options="['4×8×128×32；4×8×128×128', '4×128×256；4×128×8', '8×4×32×128；8×128×256', '4×8×256×32；4×8×32×32']" :answer="0" explanation="每头维度为 256/8=32；每个头的每个 Query 都与 128 个 Key 配对。" />

### 9.2 头的名称是分析假设，不是训练保证

某个头可能偏向同设备事件，另一个偏向时间邻近，但训练目标没有强制它们可读、互异或稳定。头可能冗余，也可能在不同样本上承担不同功能。删头、置换头和干预 Value 后观察任务变化，比给热图写故事更可靠。

## 10. Transformer 层：通信、逐位置计算与稳定梯度路径

一个 Transformer 层通常由两个子层构成：

1. 多头自注意力，让位置之间交换信息；
2. 前馈网络（FFN），对每个位置独立使用同一组非线性参数。

FFN 可写为

$$
\operatorname{FFN}(\mathbf x)
=\mathbf W_2\phi(\mathbf W_1\mathbf x+\mathbf b_1)+\mathbf b_2,
$$

其中 $\mathbf W_1$ 常把 $d_{\text{model}}$ 扩到 $d_{\text{ff}}$，再由 $\mathbf W_2$ 压回。注意力负责 token 间通信，FFN 负责每个 token 内部的通道混合。

### 10.1 LayerNorm 归一化的是单个 token 的特征维

对 $d$ 维 token $\mathbf x$：

$$
\mu=\frac1d\sum_{j=1}^d x_j,qquad
\sigma^2=\frac1d\sum_{j=1}^d(x_j-\mu)^2,
$$

$$
\operatorname{LN}(\mathbf x)
=\boldsymbol\gamma\odot\frac{\mathbf x-\mu}{\sqrt{\sigma^2+\epsilon}}
+\boldsymbol\beta.
$$

它不跨 batch 汇总统计量，因此训练和推理不需要 BatchNorm 那样的运行均值。但 $\epsilon$、低精度累加和近零方差 token 仍需测试。

```python
def layer_norm(inputs, epsilon=1e-5, gamma=None, beta=None):
    inputs = np.asarray(inputs, dtype=np.float64)
    mean = inputs.mean(axis=-1, keepdims=True)
    variance = np.mean((inputs - mean) ** 2, axis=-1, keepdims=True)
    normalized = (inputs - mean) / np.sqrt(variance + epsilon)
    if gamma is None:
        gamma = np.ones(inputs.shape[-1])
    if beta is None:
        beta = np.zeros(inputs.shape[-1])
    return normalized * gamma + beta

layer_norm_input = np.array([[[2.0, 4.0, 6.0, 8.0], [3.0, 3.0, 5.0, 9.0]]])
layer_norm_output = layer_norm(layer_norm_input, epsilon=0.0)
affine_layer_norm_output = layer_norm(3.5 * layer_norm_input + 17.0, epsilon=0.0)
assert np.allclose(layer_norm_output.mean(axis=-1), 0.0)
assert np.allclose(layer_norm_output.var(axis=-1), 1.0)
assert np.allclose(layer_norm_output, affine_layer_norm_output)
```

### 10.2 Pre-Norm 与 Post-Norm 要写清顺序

Pre-Norm 层为

$$
\mathbf H' = \mathbf H + \operatorname{MHA}(\operatorname{LN}(\mathbf H)),
$$

$$
\mathbf H'' = \mathbf H' + \operatorname{FFN}(\operatorname{LN}(\mathbf H')).
$$

原始 Transformer 常见 Post-Norm：先残差相加再归一化。两者函数不同，不能只写“有残差和 LayerNorm”而省略顺序。Pre-Norm 通常为深层网络提供更直接的恒等梯度路径，但最终选择仍要与既有架构、初始化和训练配方一致。

```python
transformer_rng = np.random.default_rng(1710)
transformer_w1 = transformer_rng.normal(scale=.15, size=(8, 16))
transformer_b1 = np.zeros(16)
transformer_w2 = transformer_rng.normal(scale=.15, size=(16, 8))
transformer_b2 = np.zeros(8)

def pre_norm_transformer_layer(inputs, allow_mask):
    normalized_attention_input = layer_norm(inputs)
    attention_output, _ = multihead_self_attention(
        normalized_attention_input,
        multihead_wq,
        multihead_wk,
        multihead_wv,
        multihead_wo,
        allow_mask,
    )
    after_attention = inputs + attention_output
    normalized_ffn_input = layer_norm(after_attention)
    hidden = np.maximum(0.0, normalized_ffn_input @ transformer_w1 + transformer_b1)
    feed_forward_output = hidden @ transformer_w2 + transformer_b2
    return after_attention + feed_forward_output

transformer_output = pre_norm_transformer_layer(multihead_x, multihead_allow)
assert transformer_output.shape == multihead_x.shape == (2, 6, 8)
assert np.isfinite(transformer_output).all()
```

残差要求子层输入输出形状一致。若多头拼接后忘记 $W_O$，恰好维度相等也可能数值上能相加，却失去跨头通道混合；shape 通过不代表语义完整。

<MlQuiz kind="架构辨析" question="Pre-Norm 层中第一条残差支路最准确的表达是哪一个？" :options="['H + MHA(LN(H))', 'LN(H + MHA(H))', 'MHA(H) + FFN(H)', 'LN(MHA(H))']" :answer="0" explanation="Pre-Norm 在每个子层进入前归一化，子层输出再与未经归一化改写的残差主路相加。" />

## 11. 编码器、解码器与因果语言模型

Transformer 是一类层结构，不等于单一任务：

- **编码器式**：常用全双向自注意力，为每个输入位置生成上下文表示；适合离线分类、检索表示和序列标注；
- **解码器式**：使用因果自注意力，以 $x_{<t}$ 预测下一个 token；适合自回归生成和在线前缀预测；
- **编码器—解码器式**：解码器先对已生成前缀做因果自注意力，再用交叉注意力读取编码器输出；适合输入序列到输出序列的条件生成。

自回归链式分解为

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

训练时可并行计算所有位置的 logits，因为完整目标序列已知且因果 Mask 保证位置 $t$ 只读前缀。推理时第 $t$ 个 token 未知，必须生成后才能继续；“训练无时间递归”不等于“自回归推理一次完成”。

### 11.1 标签位移

若输入是 `[BOS, START, LUBE_LOW, TRIP]`，目标应是 `[START, LUBE_LOW, TRIP, EOS]`。损失位置、padding Mask 与因果 Mask 必须一起核对。常见 off-by-one 错误会让模型学习复制当前 token，而日志中的 loss 仍可能下降。

<MlQuiz kind="工程判断" question="因果语言模型训练时为什么可以一次计算整段序列，而不违反因果性？" :options="['因为因果 Mask 让每个位置只读取其前缀，矩阵运算可并行完成所有位置', '因为训练时允许合法偷看答案', '因为位置编码删除了未来', '因为 Softmax 自动按时间执行循环']" :answer="0" explanation="并行是计算调度，因果性是依赖图约束；两者不是对立概念。" />

## 12. 案例实现：用注意力池化分诊根因

先建立一个可审计基线，而不是直接堆叠深层模型。我们给四类专业组定义四维教学证据 Value：润滑低压指向润滑组，振动和高温指向机械组，电流/电压异常指向电气组，远程复位和心跳异常指向控制逻辑组。Key 使用严重度，固定诊断 Query 偏向高严重度事件。

这个基线没有训练 Q/K/V，只训练每类的注意力池化质心。它的目的不是追求真实精度，而是检验数据、Mask、池化和分组切分是否通畅。

```python
evidence_table = np.zeros((len(event_names), 4), dtype=np.float64)
evidence_table[2] = [1.0, 0.0, 0.0, 0.0]
evidence_table[1] = [0.0, 1.0, 0.0, 0.0]
evidence_table[3] = [0.0, 0.7, 0.0, 0.0]
evidence_table[7] = [0.0, 0.0, 1.0, 0.0]
evidence_table[8] = [0.0, 0.0, 0.9, 0.0]
evidence_table[9] = [0.0, 0.0, 0.0, 1.0]
evidence_table[10] = [0.0, 0.0, 0.0, 0.5]

def diagnostic_attention_pool(event_row, severity_row, mask_row):
    valid_events = event_row[mask_row]
    valid_severity = severity_row[mask_row]
    logits = 2.5 * valid_severity
    weights = row_softmax(logits[None, :])[0]
    values = evidence_table[valid_events]
    return weights @ values, weights

pooled_evidence = np.vstack([
    diagnostic_attention_pool(event_type[row], severity[row], valid_mask[row])[0]
    for row in range(incident_count)
])
class_centroids = np.vstack([
    pooled_evidence[train_rows & (root_cause == class_index)].mean(axis=0)
    for class_index in range(4)
])
distance_to_centroid = np.sum(
    (pooled_evidence[:, None, :] - class_centroids[None, :, :]) ** 2,
    axis=-1,
)
predicted_cause = np.argmin(distance_to_centroid, axis=1)
test_accuracy = np.mean(predicted_cause[test_rows] == root_cause[test_rows])

assert pooled_evidence.shape == (72, 4)
assert test_accuracy > 0.95
print("教学注意力分诊测试准确率：", round(float(test_accuracy), 3))
```

这个结果不能作为上线证据，因为标签生成规则直接决定了 Value 表。它只证明实现能从合成模式恢复预设规则。下一步应把固定表替换为可学习嵌入和投影，并在真实、冻结的 campaign 外测试集上与计数规则、RNN、树模型比较。

### 12.1 从热图到可验证诊断

对每个测试样本至少记录：

- 预测类、校准置信度和拒绝判定；
- 有效 token 数、被屏蔽 token 数和是否发生截断；
- 每层每头注意力熵，而不是只保存一张平均热图；
- 删除最高权重事件、置换同类事件、改写未来事件后的预测变化；
- 设备、campaign、窗口长度和根因类别分层指标。

注意力权重可以帮助定位模型读取路径，但高权重事件不一定是充分原因。Value 方向、后续层、残差支路和输出头都会改变最终决策。

<MlQuiz kind="案例判断" question="某头对 `LUBE_LOW` 给出最高权重，能否直接宣称它是跳闸根因？" :options="['不能；权重只描述该计算中的路由，还需干预、数据与机理证据', '能；最高注意力等于因果概率', '能；只要权重超过 0.5', '不能；注意力永远不可分析']" :answer="0" explanation="注意力可作为审计线索，但不是因果识别结果；需要结合 Value、后续层、删除/替换干预和工程证据。" />

## 数值验证

工业级实现不能只验证“输出 shape 对了”。下面把核心不变量收成最小门禁。

### 13.1 算子级门禁

1. 每个有效 Query 的注意力权重非负且行和为 1；
2. 禁止位置权重严格为 0；
3. 全遮罩行被显式拒绝；
4. Q/K/V 与投影参数梯度通过 float64 中心差分；
5. 无位置全注意力通过置换等变测试；
6. 因果模型通过未来改写前缀不变测试；
7. padding 内容从 0 改成极端值时，有效输出不变；
8. 多头 reshape 前后 token 与 head 次序没有串轴。

```python
# 组合回归门禁：故意给 padding 放入极端值，合法输出仍应不变。
padding_q = mask_rng.normal(size=(5, 4))
padding_k = mask_rng.normal(size=(7, 4))
padding_v = mask_rng.normal(size=(7, 3))
padding_allow = np.zeros((5, 7), dtype=bool)
padding_allow[:, :5] = True

padding_output, padding_weights = scaled_dot_product_attention(
    padding_q, padding_k, padding_v, padding_allow
)
corrupted_padding_k = padding_k.copy()
corrupted_padding_v = padding_v.copy()
corrupted_padding_k[5:] = 1e6
corrupted_padding_v[5:] = -1e9
corrupted_output, corrupted_weights = scaled_dot_product_attention(
    padding_q, corrupted_padding_k, corrupted_padding_v, padding_allow
)

assert np.allclose(corrupted_output, padding_output)
assert np.allclose(corrupted_weights, padding_weights)
assert np.all(padding_weights[:, 5:] == 0.0)

try:
    scaled_dot_product_attention(
        padding_q[:1], padding_k, padding_v, np.zeros((1, 7), dtype=bool)
    )
except ValueError as error:
    assert "至少需要一个合法 Key" in str(error)
else:
    raise AssertionError("全遮罩行必须被拒绝")
```

### 13.2 数据与任务级门禁

- **分组切分**：设备、campaign、相邻窗口的交集为零；
- **时间回放**：按真实到达顺序重放，晚到消息遵守已声明水位线策略；
- **未来金丝雀**：在事故结束后加一个与标签强相关的假字段，在线指标若上升说明管线泄漏；
- **截断压力**：分别删除早期先兆和晚期确认，报告精度、拒绝率与校准变化；
- **长度分层**：短、中、长事件链分别评估，避免平均值掩盖长窗退化；
- **跨工况**：检修后、设备改造后和新固件版本单独报告。

<MlQuiz kind="测试设计" question="哪项测试最直接验证因果 Mask 的业务效果，而不只验证矩阵外观？" :options="['任意改写未来 token，检查所有过去输出保持不变', '检查模型参数数量为偶数', '只看训练 loss 是否下降', '把注意力图换成蓝色']" :answer="0" explanation="前缀不变性直接对应“过去不能依赖未来”的功能合同，可覆盖归一化污染和多处管线泄漏。" />

<div class="ml-stop"><strong>停下来推一推 3</strong>设计一个长度为 6 的测试：第 5、6 个事件携带标签答案。分别写出正确因果模型和无 Mask 模型在第 3 个位置应满足的断言。测试为什么还要覆盖位置编码和归一化层？</div>

## 复杂度与工程边界

### 14.1 全注意力的二次项

QKᵀ 形成 $(B,h,T,T)$ 分数矩阵，时间主项约为

$$
O(BT^2d),
$$

权重显存主项约为

$$
O(BhT^2).
$$

Q/K/V 投影和 FFN 还包含 $O(BTd^2)$ 或 $O(BTd\,d_{\text{ff}})$。因此“注意力复杂度是 $O(T^2)$”只是针对序列长度的主关系，不代表所有设备、维度和长度下都由分数矩阵主导。

```python
def attention_matrix_megabytes(batch, heads, tokens, bytes_per_value=2):
    return batch * heads * tokens * tokens * bytes_per_value / (1024 ** 2)

memory_budget = {
    tokens: attention_matrix_megabytes(batch=8, heads=8, tokens=tokens)
    for tokens in (512, 1024, 2048, 4096)
}
assert np.isclose(memory_budget[1024] / memory_budget[512], 4.0)
assert np.isclose(memory_budget[4096] / memory_budget[1024], 16.0)
print({tokens: round(megabytes, 1) for tokens, megabytes in memory_budget.items()})

model_dimension_budget = 512
ffn_dimension_budget = 2048
attention_parameter_main = 4 * model_dimension_budget ** 2
ffn_parameter_main = 2 * model_dimension_budget * ffn_dimension_budget
assert attention_parameter_main == 1_048_576
assert ffn_parameter_main == 2_097_152
```

表中只计算一张半精度注意力矩阵，不含梯度、Softmax 中间量、dropout、各层激活和框架工作区。训练峰值必须在目标硬件、真实 batch 与序列长度分布上实测。

<MlQuiz kind="容量估算" question="batch、头数和精度不变，序列长度从 1024 增到 4096，仅注意力权重矩阵元素数变为多少倍？" :options="['4 倍', '8 倍', '16 倍', '与长度无关']" :answer="2" explanation="矩阵含 T² 个位置对，长度扩大 4 倍会让元素数扩大 4²=16 倍。" />

### 14.2 训练并行、推理串行与 KV Cache

因果模型训练时一次计算全部 token；自回归推理仍逐 token 生成。若每步重新计算完整前缀，会重复生成旧 Key/Value。KV Cache 保存各层历史 Key/Value，使新步只投影新 token 并对缓存读取。

缓存不会消除以下成本：

- 新 Query 仍需与全部历史 Key 做点积，单步读取成本随上下文增长；
- 每层每头的 K/V 占用显存，长上下文和大 batch 很快受内存限制；
- beam search、请求批处理、前缀复用和缓存淘汰改变服务语义；
- 模型版本、位置索引、Mask 规则或量化格式改变时，旧缓存必须失效。

对本章离线根因分类，一次全窗口前向通常不需要 KV Cache。为“流式”名义引入缓存只会增加状态边界和回放复杂度。

### 14.3 RNN 与 Transformer 的选择不是口号

| 维度 | RNN | 标准全注意力 |
|---|---|---|
| 训练时间依赖 | 时间步串行 | token 维可并行 |
| 两位置最短交互路径 | 随距离增长 | 单层可直接连接 |
| 流式状态 | 固定维隐藏状态 | 通常增长的 KV Cache |
| 长度成本 | 约线性步数 | 分数矩阵二次增长 |
| 顺序偏置 | 递推天然有序 | 需要位置或结构 Mask |

短中事件链、GPU 批训练和跨位置关系复杂时，Transformer 很有吸引力；极长、高频、严格定内存流式任务可能更适合 RNN、局部注意力、稀疏注意力或状态空间模型。最终判断要看目标硬件的延迟、吞吐、峰值内存和任务精度，而不是只比较大 O。

<div class="ml-stop"><strong>停下来推一推 4</strong>一个服务把最大事件数从 256 提到 1024，batch 与头数不变。估算注意力矩阵增长倍数；再列出两种不简单“加显存”的业务或模型改法。</div>

## 15. 生产失败模式与监控

### 15.1 未来泄漏

- 用完整窗口的均值、最大值或总事件数作为每个在线 token 的输入；
- 标签位移错误，让位置 $t$ 直接看到目标 token；
- Mask 广播轴错误，只遮住某些 head 或某些 batch；
- Softmax 后清零，合法权重仍依赖未来；
- 事故后人工字段通过资产表回填到事故发生前。

### 15.2 padding 与空序列

- padding Key 未遮罩，位置编码或 bias 让其变成高分；
- padding Query 虽不参与损失，却进入后续池化；
- 全遮罩行产生 NaN，随后被静默替换为 0；
- `PAD`、`UNK` 和真实的“心跳为空”共用一个词表 id。

### 15.3 长度、词表与时间漂移

- 新固件产生未知事件码，`UNK` 比例激增；
- 时钟同步故障导致相对时间为负或跨天跳变；
- 窗口策略变化使长度分布和截断率改变；
- 长序列注意力熵塌缩或过度平均；
- 设备改造后，相同事件码对应的机理含义改变。

### 15.4 最低监控面板

至少按模型版本、设备族和根因类监控：

1. 输入事件率、`UNK` 率、空窗率、乱序率、重复率；
2. 序列长度分位数、padding 比例、截断率；
3. 注意力 logits 范围、熵、Mask 后行和、NaN/Inf；
4. 预测类比例、置信度、拒绝率和延迟分位数；
5. 标签成熟后统计分层混淆矩阵、校准误差和工单转派成本；
6. 未来改写、padding 污染和缓存回放三类在线金丝雀。

<MlQuiz kind="故障定位" question="线上只有长事件链出现 NaN，短链正常。第一轮最有价值的联合排查是什么？" :options="['检查全遮罩行、Mask 广播、logit 范围和低精度 Softmax', '把所有 NaN 直接改成 0', '删除 LayerNorm', '只增加注意力头数']" :answer="0" explanation="长链会放大 Mask 边界、极端 logit 和低精度稳定性问题；先定位数值与可见域合同，不能静默吞掉 NaN。" />

## 16. 验收清单

### 数据与任务

- [ ] 离线复盘、在线前缀预测和自回归生成分别声明可见域。
- [ ] 设备、campaign、相邻窗口按实体分组切分。
- [ ] 事件词表、设备主数据、严重度量程和时间语义版本化。
- [ ] `PAD`、`UNK`、缺失和真实零值互不混淆。
- [ ] 标签成熟时间、回填路径与预测时间点可追溯。

### 算子与梯度

- [ ] Q/K/V、每头、拼接和输出 shape 有断言。
- [ ] Softmax 逐 Query 行执行且减去合法最大值。
- [ ] 因果与 padding Mask 在 logits 上合并。
- [ ] 全遮罩行显式拒绝或有文档化语义。
- [ ] Q/K/V 和投影梯度通过 float64 有限差分。
- [ ] 禁止格概率与 score 梯度严格为 0。

### 结构与部署

- [ ] 位置编码覆盖真实最大长度与时间间隔需求。
- [ ] Pre-Norm/Post-Norm、激活、dropout 和 bias 布局与配置一致。
- [ ] 未来改写、padding 极值、置换和截断测试进入 CI。
- [ ] 在目标硬件实测长度分层的延迟、吞吐和峰值内存。
- [ ] KV Cache 的请求、模型版本、位置和失效边界明确。
- [ ] 注意力图只作为路由证据，不在报告中冒充因果解释。

## 17. 练习与最小实现

### 练习 1：手算单头注意力

给定

$$
\mathbf Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
\mathbf K=\begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix},\quad
\mathbf V=\begin{bmatrix}1&0\\2&1\\0&3\end{bmatrix},
$$

完成：

1. 写出 Q、K、V、score、weight 和输出形状；
2. 手算除以 $\sqrt2$ 后的 scores；
3. 逐行稳定 Softmax；
4. 计算输出并验证在 Value 凸包内；
5. 对第二个 Key 做屏蔽，重新计算并解释变化。

### 练习 2：Mask 与泄漏对抗测试

实现同时支持 causal 和 padding 的 batch 多头注意力：

- 输入 `(B,h,T_q,d_k)`、`(B,h,T_k,d_k)`、`(B,h,T_k,d_v)`；
- 接受 query mask、key mask 和 causal offset；
- 测试不同 batch 长度、广播轴和全遮罩行；
- 将未来 K/V 改成 `1e9`，过去输出必须不变；
- 故意实现 Softmax 后清零，确保测试能抓住错误。

### 练习 3：多头与位置消融

在合成事件数据上比较：

1. 单头与 2、4、8 头；
2. 无位置、正弦位置和真实时间差编码；
3. 原始顺序、随机置换和时间反转；
4. 删除最高权重头与随机删头；
5. 固定 $d_{\text{model}}$ 时核对参数量是否随头数改变。

### 练习 4：最小 Transformer 根因分类器

构建两层 Pre-Norm 编码器：事件字段嵌入、位置编码、多头注意力、FFN、masked mean pooling 和四类 Softmax。必须：

- 按 campaign 切分；
- 仅用训练集估计标准化和词表；
- 用有限差分抽查一层注意力；
- 报告宏 F1、混淆矩阵、校准误差和拒绝率；
- 与事件计数、逻辑回归、树模型和 RNN 基线比较；
- 执行未来金丝雀、padding 极值和长窗压力测试。

### 练习 5：框架对照

下面的 PyTorch 片段仅是可选对照，仓库验证环境未为此新增大型依赖。核对不同版本中布尔 Mask 的真值语义、`is_causal` 与显式 Mask 是否可同时使用，以及 dropout 在训练/评估模式下的差异。

```python
# 不执行：需要可选的 PyTorch 环境。
import torch
import torch.nn.functional as F

torch.manual_seed(17)
q = torch.randn(2, 3, 6, 4, dtype=torch.float64, requires_grad=True)
k = torch.randn(2, 3, 6, 4, dtype=torch.float64, requires_grad=True)
v = torch.randn(2, 3, 6, 5, dtype=torch.float64, requires_grad=True)

output = F.scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
)
assert output.shape == (2, 3, 6, 5)
output.square().mean().backward()
assert q.grad is not None and k.grad is not None and v.grad is not None
```

```python
# TODO：实现带 padding、因果 Mask 和完整 backward 的批量多头注意力，并完成以下验收。
#
# 1. 输入 Q/K/V 分别为 (B,h,Tq,dk)、(B,h,Tk,dk)、(B,h,Tk,dv)；
# 2. Mask 在 Softmax 前进入 logits，全遮罩 Query 明确报错；
# 3. 返回输出、权重及可选 KV Cache，缓存带模型版本与位置偏移；
# 4. dQ/dK/dV 与投影矩阵梯度通过 float64 中心差分；
# 5. 未来改写、padding 极值、置换等变和多头串轴测试全部通过；
# 6. 在 128、512、2048 token 上记录延迟和峰值内存。
```

## 18. 本章结论

1. 注意力是按内容寻址：Q 提问、K 接受匹配、V 搬运内容，权重是路由系数而非天然因果解释。
2. 缩放点积除以 $\sqrt{d_k}$，因为独立单位方差分量的点积标准差随 $\sqrt{d_k}$ 增长。
3. Softmax 按 Query 行归一化；稳定实现减去每行合法最大值，并拒绝没有合法 Key 的行。
4. 因果 Mask、padding Mask 和 Query 有效性解决不同问题，必须分别建模和组合测试。
5. Mask 必须在 Softmax 前定义概率支持集；未来格事后清零仍可能留下归一化污染。
6. 未来改写前缀不变测试能把业务因果合同转成可执行断言。
7. 注意力反向传播可由矩阵乘法和 Softmax 向量—Jacobian 积完成，Q/K/V 都应通过有限差分。
8. 无位置的全自注意力对 token 置换等变；顺序来自位置编码、结构 Mask 或其他显式机制。
9. 多头使用多套投影子空间；固定总维度时，增加头数通常不增加 Q/K/V/O 的主参数量。
10. Transformer 层用注意力做位置间通信、FFN 做逐位置通道变换，并通过残差和 LayerNorm 稳定深层优化。
11. Pre-Norm 与 Post-Norm 的运算顺序不同，必须在配置、代码和权重加载中一致。
12. 因果模型训练可以并行计算全部位置，自回归推理仍逐 token 生成；KV Cache 减少重复投影但不消除历史读取成本。
13. 标准全注意力在长度上的时间主项为 $O(T^2d)$、权重内存主项为 $O(hT^2)$，真实峰值必须在目标硬件实测。
14. 工业事件模型的可用性取决于窗口、时钟、词表、Mask、切分、标签成熟和回放合同，而不只取决于网络结构。
15. 手算、有限差分、未来改写、padding 极值、置换、长窗压力和分组外测试共同构成实现验收门禁。

## 来源与许可

内容寻址、缩放点积注意力、自注意力、多头注意力、位置编码、Transformer 编码器—解码器结构与复杂度边界，参考 Kevin P. Murphy 的 *Probabilistic Machine Learning: An Introduction* 源 PDF 第 549–563 页（§15.4–15.5）；其中缩放点积与 Q/K/V 的矩阵定义集中在源 PDF 第 551–552 页，自注意力、多头与位置表示集中在源 PDF 第 556–560 页，编码器、解码器和复杂度讨论集中在源 PDF 第 560–562 页。

从 RNN 到内容注意力、Q/K/V 和 Transformer 的教学顺序，交叉参考 CMU 10-301/10-601 Lecture 18 *RNN, Transformer* 源 PDF 第 62–77 页；多头注意力、语言模型、残差、LayerNorm、矩阵化因果 Mask 与批处理实践，交叉参考 CMU Lecture 19 *Autodiff, In-context Learning* 源 PDF 第 3–33 页。

自注意力的 token 表示、缩放、Q/K/V、多头、层结构与位置编码，交叉参考 UC Berkeley CS 189/289A Lecture 21 源 PDF 第 60–118 页；自回归任务中的 masked attention 交叉参考 Lecture 23 源 PDF 第 17–34 页。不同课程对层顺序、符号和实现接口的表达不完全一致，本章以明确 shape、可见域和 Pre-Norm 公式消除歧义。

Jay Alammar 的 *The Illustrated Transformer* 本地网页快照采用 CC BY-NC-SA 4.0，用于交叉核对 Q/K/V、矩阵化、多头、位置、残差和因果遮罩的视觉信息顺序。本章未复制其图像、代码或表述；压缩机站案例、三项网页交互、SVG 图形、NumPy 实现、题目与测试均为本站独立创作。

教学合成事件链不代表真实天然气站、压缩机、电机、PLC、保护继电器、报警优先级、维修根因或安全联锁规则。本章模型只用于离线分诊教学，不得据此旁路保护、自动复位、修改定值或直接闭环控制；PyTorch 片段明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

<MlLessonProgress
  lesson-id="17"
  next-href="/courses/machine-learning/18-knn"
  next-label="第 18 章：k 近邻算法（kNN）"
/>
