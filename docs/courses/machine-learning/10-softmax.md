---
course: machine-learning
edition: industrial
title: Softmax 回归
description: 以电机台架主导故障分流为贯穿案例，从 Categorical 分布与相对 log odds 推导 Softmax、交叉熵、矩阵梯度、Hessian、凸性、参数不可辨识、温度缩放与成本动作。
outline: [2, 3]
---

<MlLessonHero
  number="10"
  title="Softmax 回归"
  question="四路任意实数证据怎样竞争同一份概率质量，又怎样在平移零空间、数值溢出和非对称代价下保持可审计？"
  duration="建议 430–520 分钟"
  :outcomes="[
    '从 Categorical 分布、相对 log odds 与指数归一化推导 Softmax 交叉熵',
    '推导 Softmax Jacobian、矩阵梯度、Hessian、凸性与 Hessian-vector product',
    '解释参考类、平移不变性、温度、label smoothing、多标签边界与成本决策'
  ]"
/>

二分类逻辑回归只有一条独立 log-odds 轴。现在输出变成 $C$ 个互斥类别：

$$
Y\in\{1,\ldots,C\}.
$$

模型必须返回概率向量

$$
p(x)=[p_1(x),\ldots,p_C(x)]^T,
\qquad
p_c>0,
\qquad
\sum_{c=1}^Cp_c=1.
$$

Softmax 回归不是“给每个类别各训练一个互不相干的 Sigmoid”。所有类别共享一个归一化分母，因此任何一类概率上升都要从其他类别回收概率质量。本章关心的不只是公式，还包括：

- 互斥标签假设是否符合真实处置；
- logits 的共同平移为何不可辨识；
- 极端 logits 怎样稳定归一化；
- 完整 Hessian 为什么半正定却必然有零方向；
- 类别温度、label smoothing、类别权重和成本矩阵分别改变哪一层。

## 贯穿案例：电机台架主导故障分流

某电机装配线在下线测试台采集一次转频、二次转频、轴承包络谱与轴向振动。每台电机由复核工程师给出一个**主导处置标签**：

| 类别 | 教学标签 | 处置含义 |
|---:|---|---|
| 0 | 合格 | 自动进入下一工序 |
| 1 | 不平衡 | 转子平衡复核 |
| 2 | 不对中 | 装配同轴度复核 |
| 3 | 轴承异常 | 轴承与装配应力复核 |

这里的“互斥”来自标签合同：工程师必须选择主导处置路径。它不代表物理故障不能共存。一台电机完全可能同时存在不对中与轴承异常；若任务要求同时输出多个故障标签，输出空间应是 $\{0,1\}^C$，通常使用 $C$ 个 Sigmoid 与逐类二元损失，而不是 Softmax。

### 数据合同

| 对象 | 本章教学定义 | 工业系统必须确认 |
|---|---|---|
| 观测单元 | 一台电机的一次冻结台架窗口 | 重测、返修后复测如何关联 |
| 标签 | 专家裁决的主导处置 | 多故障、未知、待复核是否另设状态 |
| 特征截止 | 首次处置前 | 禁止使用拆机结论和返修结果 |
| 分组键 | 生产批次 | 供应商、设备、班次是否还需更高层分组 |
| 类别集合 | 四类封闭集合 | 新故障类型出现时如何拒绝而非硬归类 |
| 部署目标 | 新批次同工艺电机 | 新材料、新轴承或新台架需重新验收 |

教学数据含 60 个生产批次、每批 40 台电机。前 40 批训练，中间 10 批验证，最后 10 批测试。批次效应会同时改变传感器和类别先验，因此不能把同批样本随机拆到三侧。

```python
import numpy as np

rng = np.random.default_rng(20260805)
class_names = np.array(["合格", "不平衡", "不对中", "轴承异常"])
class_count = len(class_names)
lot_count = 60
motors_per_lot = 40
lot_id = np.repeat(np.arange(lot_count), motors_per_lot)
lot_shift = rng.normal(0, 0.28, size=(lot_count, 2))

one_x = rng.normal(lot_shift[lot_id, 0], 1.0)
two_x = rng.normal(0.20 * one_x + lot_shift[lot_id, 1], 1.0)
envelope = rng.normal(0.15 * two_x - 0.2 * lot_shift[lot_id, 0], 1.0)
axial = rng.normal(0.30 * two_x + 0.2 * lot_shift[lot_id, 1], 1.0)
raw_features = np.column_stack([one_x, two_x, envelope, axial])

true_weight = np.array([
    [1.25, -0.70, -0.85, -1.00],
    [-0.85, 1.65, 0.15, 0.05],
    [-0.70, 0.10, 1.45, 0.15],
    [-0.95, 0.05, 0.20, 1.75],
    [-0.45, 0.10, 1.00, 0.35],
])
raw_design = np.column_stack([np.ones(len(raw_features)), raw_features])
true_logits = raw_design @ true_weight
true_logits[:, 1] += 0.45 * lot_shift[lot_id, 0]
true_logits[:, 2] += 0.45 * lot_shift[lot_id, 1]
true_logits[:, 3] -= 0.30 * lot_shift[lot_id, 0]
shifted_true_logits = true_logits - true_logits.max(axis=1, keepdims=True)
true_probability = np.exp(shifted_true_logits)
true_probability /= true_probability.sum(axis=1, keepdims=True)
uniform_draw = rng.random(len(true_probability))
label = (uniform_draw[:, None] > np.cumsum(true_probability, axis=1)).sum(axis=1)

train_mask = lot_id < 40
validation_mask = (lot_id >= 40) & (lot_id < 50)
test_mask = lot_id >= 50
print("训练/验证/测试：", train_mask.sum(), validation_mask.sum(), test_mask.sum())
print("四类总体计数：", dict(zip(class_names, np.bincount(label, minlength=class_count))))
print("三侧批次数：",
      np.unique(lot_id[train_mask]).size,
      np.unique(lot_id[validation_mask]).size,
      np.unique(lot_id[test_mask]).size)
```

<MlQuiz kind="标签合同" question="一台电机可同时存在不对中和轴承异常，业务又要求同时报告两项。最合适的输出建模是什么？" :options="['仍强制四类 Softmax', '每类独立二元标签与 Sigmoid，多标签评价', '把两个故障概率相加后裁剪', '只保留样本较多的故障']" :answer="1" explanation="Softmax/Categorical 假设单次观测只有一个互斥结果。多故障并存应使用多标签输出，除非业务明确要求裁决唯一主导处置。" />

## 1. Categorical 分布定义了互斥输出空间

把类别 $c_i$ 编码为 one-hot 向量 $y_i\in\{0,1\}^C$：

$$
y_{ic}=\mathbf 1(c_i=c),
\qquad
\sum_{c=1}^Cy_{ic}=1.
$$

若概率参数为 $p_i\in\Delta^{C-1}$，其中

$$
\Delta^{C-1}
=\left\{
p\in\mathbb R^C:
p_c\ge0,\ \sum_cp_c=1
\right\},
$$

则

$$
Y_i\mid x_i\sim\operatorname{Categorical}(p_i),
$$

概率质量函数可写为

$$
p(y_i\mid p_i)=\prod_{c=1}^Cp_{ic}^{y_{ic}}.
$$

Categorical 描述一次试验的一个结果；Multinomial 描述固定次数试验后各类别的计数。机器学习文献常把多类逻辑回归称作 multinomial logistic regression，但单个样本的条件分布是 Categorical。

<MlQuiz kind="分布辨析" question="一台电机只产生一个主导处置标签，最直接的观测分布是什么？" :options="['Gaussian', 'Categorical', 'Poisson', '四个相互独立 Bernoulli']" :answer="1" explanation="一次观测从互斥类别集合中取一个值，对应 Categorical；批量类别计数才对应 Multinomial。" />

## 2. 从相对 log odds 推导 Softmax

每类先产生任意实数 logit：

$$
z_c(x)=\tilde x^Tw_c.
$$

多分类没有唯一的“绝对 log odds”，有意义的是成对 log odds。选参考类 $r$：

$$
\log\frac{p_c(x)}{p_r(x)}
=z_c(x)-z_r(x)
=\tilde x^T(w_c-w_r).
$$

若令非归一化正权为 $e^{z_c}$，归一化得到

$$
\boxed{
p_c
=\frac{e^{z_c}}{\sum_{j=1}^Ce^{z_j}}
}.
$$

它满足：

$$
p_c>0,
\qquad
\sum_cp_c=1.
$$

Softmax 不是逐类 Sigmoid 再归一化。指数有关键性质

$$
\frac{p_c}{p_r}=e^{z_c-z_r},
$$

使成对 log odds 恰好等于 logit 差。

### 二分类是 Softmax 的一个坐标选择

当 $C=2$：

$$
p_1
=\frac{e^{z_1}}{e^{z_0}+e^{z_1}}
=\frac1{1+e^{-(z_1-z_0)}}
=\sigma(z_1-z_0).
$$

二分类只需要一个差值参数。保留两列权重不会改变预测能力，却引入共同平移冗余。

<MlQuiz kind="相对证据" question="若不平衡与合格的 logits 差从 0.4 增加到 1.1，它们的概率比发生什么变化？" :options="['固定增加 0.7', '从 exp(0.4) 变为 exp(1.1)', '两类概率都固定不变', '概率比一定等于 1.1']" :answer="1" explanation="Softmax 的成对 odds ratio 是 exp(z_c−z_r)。其他类别影响各自绝对概率，但不改变这两个类别的概率比公式。" />

<div class="ml-stop"><strong>停下来推一推 1</strong> 对 logits $[2,1,0]$ 手算三个相对指数权。无需先算巨大精度：减最大值后得到 $[1,e^{-1},e^{-2}]$，再除以它们的和。</div>

<SoftmaxCompetitionConsoleLab />

<MlQuiz kind="竞争机制" question="只把“轴承异常”logit 提高 1，其余不变。哪项必然成立？" :options="['轴承异常概率上升，其余每类概率下降', '所有概率同时上升', '概率和大于 1', '其余 logits 自动减少 1']" :answer="0" explanation="共同分母增大，目标类分子也增大；目标类概率上升，其余类别分子不变但分母变大。" />

## 3. log-sum-exp：先稳定坐标，再做指数

直接计算 $e^{1000}$ 会溢出，$e^{-1000}$ 会下溢。定义

$$
\operatorname{LSE}(z)
=\log\sum_{c=1}^Ce^{z_c}.
$$

对任意常数 $m$：

$$
\begin{aligned}
\operatorname{LSE}(z)
&=\log\left(e^m\sum_ce^{z_c-m}\right)\\
&=m+\log\sum_ce^{z_c-m}.
\end{aligned}
$$

取 $m=\max_cz_c$，最大的指数变成 1，其余不超过 1：

$$
\log p_c=z_c-\operatorname{LSE}(z).
$$

减最大值是精确恒等变换，不是近似，也不是“裁剪极端值”。

```python
def logsumexp(logits, axis=-1, keepdims=False):
    logits = np.asarray(logits, dtype=float)
    maximum = np.max(logits, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.exp(logits - maximum).sum(axis=axis, keepdims=True)
    )
    return result if keepdims else np.squeeze(result, axis=axis)

def stable_softmax(logits):
    logits = np.asarray(logits, dtype=float)
    return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))

extreme_logits = np.array([
    [1000.0, 1001.0, 999.0, 998.0],
    [-1000.0, -999.0, -1002.0, -1001.0],
])
extreme_probability = stable_softmax(extreme_logits)
print("极端 logits 的稳定概率：")
print(np.round(extreme_probability, 6))
print("概率和：", extreme_probability.sum(axis=1))
```

<MlQuiz kind="数值稳定" question="对每行 logits 减去该行最大值，为什么不会改变 Softmax？" :options="['因为所有分子与分母共同乘同一个正因子', '因为最大值总是 0', '这是近似但误差很小', '只对四分类成立']" :answer="0" explanation="共同平移使所有指数同乘 e^{-m}，归一化比值严格不变；选择最大值只是让指数范围安全。" />

## 4. Categorical 似然为什么得到交叉熵

对独立样本：

$$
\mathcal L(W)
=\prod_{i=1}^n\prod_{c=1}^C
p_{ic}^{y_{ic}}.
$$

平均负 log-likelihood：

$$
\begin{aligned}
J(W)
&=-\frac1n\sum_i\sum_cy_{ic}\log p_{ic}\\
&=\frac1n\sum_i
\left[
\operatorname{LSE}(z_i)-\sum_cy_{ic}z_{ic}
\right].
\end{aligned}
$$

对 one-hot 标签 $c_i$：

$$
\boxed{
\ell_i=\operatorname{LSE}(z_i)-z_{i,c_i}
=-\log p_{i,c_i}
}.
$$

交叉熵直接从 Categorical MLE 产生。若目标是一个可信标签分布 $q_i$ 而非 one-hot，仍可用

$$
\ell_i=-\sum_cq_{ic}\log p_{ic},
\qquad \sum_cq_{ic}=1.
$$

但“多个类别都为真”的多标签不是标签分布不确定性：多标签每一位都有独立真假语义，总和不要求为 1。

<MlQuiz kind="似然推导" question="one-hot 标签下，单样本多分类交叉熵等于什么？" :options="['所有 logits 的平均值', '真实类别概率的负对数', '最大概率本身', '预测类别编号']" :answer="1" explanation="one-hot 只保留真实类别对应的 log probability，因此损失是 −log p_true。" />

## 5. 对 logits 的梯度为何仍是预测减标签

单样本损失：

$$
\ell(z,y)=\operatorname{LSE}(z)-y^Tz.
$$

因为

$$
\frac{\partial\operatorname{LSE}(z)}{\partial z_c}
=\frac{e^{z_c}}{\sum_je^{z_j}}
=p_c,
$$

所以

$$
\boxed{\nabla_z\ell=p-y}.
$$

设

$$
X\in\mathbb R^{n\times d},
\qquad
W\in\mathbb R^{d\times C},
\qquad
Z=XW,
$$

且 $P,Y\in\mathbb R^{n\times C}$。矩阵链式法则给出

$$
\boxed{
\nabla_WJ=\frac1nX^T(P-Y)
}.
$$

形状闭环：

$$
(d\times n)(n\times C)=d\times C.
$$

每行误差满足

$$
\mathbf1^T(p_i-y_i)=0,
$$

因此数据梯度的类别列和为 0。训练只关心相对 logits。

### 只用训练批次拟合标准化

```python
feature_mean = raw_features[train_mask].mean(axis=0)
feature_scale = raw_features[train_mask].std(axis=0)
feature_scale = np.where(feature_scale > 0, feature_scale, 1.0)
standardized_features = (raw_features - feature_mean) / feature_scale
design = np.column_stack([np.ones(len(standardized_features)), standardized_features])

X_train, y_train = design[train_mask], label[train_mask]
X_validation, y_validation = design[validation_mask], label[validation_mask]
X_test, y_test = design[test_mask], label[test_mask]
lot_train = lot_id[train_mask]
Y_train = np.eye(class_count)[y_train]
Y_validation = np.eye(class_count)[y_validation]
Y_test = np.eye(class_count)[y_test]

penalty_mask = np.ones((X_train.shape[1], 1))
penalty_mask[0, 0] = 0.0

def softmax_objective_gradient(
    weight, X, target, regularization=0.0
):
    logits = X @ weight
    log_probability = logits - logsumexp(
        logits, axis=1, keepdims=True
    )
    probability = np.exp(log_probability)
    loss = -np.sum(target * log_probability) / len(X)
    loss += 0.5 * regularization * np.sum(
        penalty_mask * weight**2
    )
    gradient = X.T @ (probability - target) / len(X)
    gradient += regularization * penalty_mask * weight
    return loss, gradient, probability
```

### 矩阵有限差分

```python
probe_weight = rng.normal(0, 0.25, size=(X_train.shape[1], class_count))
probe_weight -= probe_weight.mean(axis=1, keepdims=True)
probe_loss, analytic_gradient, _ = softmax_objective_gradient(
    probe_weight, X_train, Y_train, regularization=0.09
)

epsilon = 1e-6
numeric_gradient = np.zeros_like(probe_weight)
for row in range(probe_weight.shape[0]):
    for column in range(probe_weight.shape[1]):
        step = np.zeros_like(probe_weight)
        step[row, column] = epsilon
        plus = softmax_objective_gradient(
            probe_weight + step,
            X_train,
            Y_train,
            regularization=0.09,
        )[0]
        minus = softmax_objective_gradient(
            probe_weight - step,
            X_train,
            Y_train,
            regularization=0.09,
        )[0]
        numeric_gradient[row, column] = (plus - minus) / (2 * epsilon)

print("矩阵梯度最大误差：",
      np.max(np.abs(analytic_gradient - numeric_gradient)))
print("数据梯度类别列和：",
      np.max(np.abs(analytic_gradient.sum(axis=1))))
```

<MlQuiz kind="矩阵形状" question="X 为 n×d、P−Y 为 n×C，权重 W 为 d×C。梯度应写成什么？" :options="['(P−Y)X', 'Xᵀ(P−Y)', 'X(P−Y)ᵀ', '(P−Y)ᵀXᵀ']" :answer="1" explanation="Xᵀ 是 d×n，乘 n×C 得 d×C，与 W 形状一致。" />

<div class="ml-stop"><strong>停下来推一推 2</strong> 若某样本真实类是轴承异常，当前概率为 $[0.6,0.2,0.15,0.05]$，logit 梯度为 $[0.6,0.2,0.15,-0.95]$。一次下降步会压低前三类并强烈抬高真实类。</div>

## 6. Softmax Jacobian 是 Categorical 协方差

对 logits 求导：

$$
\frac{\partial p_c}{\partial z_r}
=p_c(\mathbf1[c=r]-p_r).
$$

矩阵形式：

$$
\boxed{
J_{\mathrm{softmax}}(p)
=\operatorname{diag}(p)-pp^T
}.
$$

如果随机 one-hot 向量 $Y\sim\operatorname{Categorical}(p)$，那么

$$
\operatorname{Cov}(Y)
=\operatorname{diag}(p)-pp^T.
$$

所以 Jacobian 天然半正定。对任意 $a$：

$$
a^TJa
=\sum_cp_ca_c^2-\left(\sum_cp_ca_c\right)^2
=\operatorname{Var}_{c\sim p}(a_c)\ge0.
$$

并且

$$
J\mathbf1=0.
$$

共同抬高所有 logits 是零方向。

### 完整 Hessian

按类别列优先向量化 $W$：

$$
\boxed{
H(W)
=\frac1n\sum_{i=1}^n
\left[
\operatorname{diag}(p_i)-p_ip_i^T
\right]
\otimes
(x_ix_i^T)
}.
$$

Kronecker 两因子都半正定，因此 Hessian 半正定，未正则多类负 log-likelihood 对 $W$ 凸。但由于共同平移零空间，它在完整参数化下不严格凸。

```python
example_probability = np.array([0.52, 0.23, 0.17, 0.08])
softmax_jacobian = (
    np.diag(example_probability)
    - np.outer(example_probability, example_probability)
)
print("Jacobian 特征值：", np.round(np.linalg.eigvalsh(softmax_jacobian), 8))
print("J·1 范数：", np.linalg.norm(softmax_jacobian @ np.ones(class_count)))

small_hessian = np.zeros((
    X_train.shape[1] * class_count,
    X_train.shape[1] * class_count,
))
small_probability = softmax_objective_gradient(
    probe_weight, X_train[:80], Y_train[:80]
)[2]
for row, probability in zip(X_train[:80], small_probability):
    class_curvature = np.diag(probability) - np.outer(probability, probability)
    small_hessian += np.kron(class_curvature, np.outer(row, row)) / 80
print("完整 Hessian 最小/最大特征值：",
      np.linalg.eigvalsh(small_hessian)[[0, -1]])
```

<MlQuiz kind="曲率判断" question="为什么完整 Softmax 参数化的 Hessian 必然存在共同平移零方向？" :options="['因为概率可以为负', '每个样本的类别 Jacobian 满足 J·1=0，共同 logit 变化不影响概率', '因为特征必须为零', '只有两分类才有零方向']" :answer="1" explanation="共同平移对所有类别加同一数，Softmax 不变；对应的参数方向不会改变任何预测，曲率为零。" />

## 7. 参数不可辨识与参考坐标

本章使用 $W\in\mathbb R^{d\times C}$。对任意 $v\in\mathbb R^d$：

$$
W'=W+v\mathbf1^T.
$$

则

$$
x^TW'=x^TW+(x^Tv)\mathbf1^T,
$$

所有类别 logits 共同增加 $x^Tv$，概率完全相同。于是：

- 概率模型可辨识；
- 完整 $W$ 的绝对列值不可辨识；
- 只解释 $w_c-w_r$ 才有坐标无关意义；
- 优化器返回的某个 $W$ 只是等价类中的代表。

### 三种常见定规

1. **参考类**：令 $w_C=0$，估计其余 $C-1$ 列；
2. **和为零**：令 $\sum_cw_c=0$；
3. **完整 L2**：在所有等价参数中选择最小范数代表。

完整 L2 会打破 gauge 自由度，但若截距不惩罚，共同截距平移仍未被惩罚，最好显式中心化或使用参考类。参考类为 0 只是坐标选择，不表示参考类“没有特征效应”。

<SoftmaxReferenceGaugeLab />

```python
common_vector = np.array([2.0, -0.4, 0.8, -0.2, 0.5])
shifted_weight = probe_weight + common_vector[:, None]
probe_probability = softmax_objective_gradient(
    probe_weight, X_validation, Y_validation
)[2]
shifted_probability = softmax_objective_gradient(
    shifted_weight, X_validation, Y_validation
)[2]

reference_weight = probe_weight - probe_weight[:, [-1]]
centered_weight = probe_weight - probe_weight.mean(axis=1, keepdims=True)
reference_probability = stable_softmax(X_validation @ reference_weight)
centered_probability = stable_softmax(X_validation @ centered_weight)

print("共同平移概率最大差：",
      np.max(np.abs(probe_probability - shifted_probability)))
print("参考类/和为零概率最大差：",
      np.max(np.abs(probe_probability - reference_probability)),
      np.max(np.abs(probe_probability - centered_probability)))
print("参考类最后一列范数：", np.linalg.norm(reference_weight[:, -1]))
print("和为零定规的列和范数：", np.linalg.norm(centered_weight.sum(axis=1)))
```

<MlQuiz kind="坐标解释" question="把轴承异常设为参考类并令其权重列为 0，能否解释为轴承异常不受任何特征影响？" :options="['能', '不能；这是相对 log-odds 的坐标选择，其他列都相对于参考类解释', '只有截距不受影响', '只在测试集不能']" :answer="1" explanation="参考类参数固定为零是消除冗余的定规。实际概率仍受所有类别相对分数影响。" />

## 8. 优化：矩阵梯度、HVP 与 gauge

### 8.1 L2 目标

使用平均损失：

$$
J_\lambda(W)
=-\frac1n\sum_i\sum_cy_{ic}\log p_{ic}
+\frac\lambda2\lVert P\odot W\rVert_F^2,
$$

其中 $P$ 的截距行为 0，其余为 1。训练循环每步把类别列中心化到和为零 gauge：

$$
W\leftarrow W-\frac1C(W\mathbf1)\mathbf1^T.
$$

这一步不改变概率，只固定参数坐标。

```python
def center_class_gauge(weight):
    return weight - weight.mean(axis=1, keepdims=True)

def fit_softmax_gradient_descent(
    X,
    target,
    regularization=0.08,
    maximum_iterations=1200,
    tolerance=1e-8,
):
    weight = np.zeros((X.shape[1], target.shape[1]))
    history = []
    for _ in range(maximum_iterations):
        loss, gradient, _ = softmax_objective_gradient(
            weight, X, target, regularization
        )
        history.append(loss)
        if np.linalg.norm(gradient, ord=np.inf) < tolerance:
            break
        direction = -gradient
        directional_derivative = np.sum(gradient * direction)
        step_size = 1.0
        while step_size > 1e-9:
            candidate = center_class_gauge(weight + step_size * direction)
            candidate_loss = softmax_objective_gradient(
                candidate, X, target, regularization
            )[0]
            if candidate_loss <= loss + 1e-4 * step_size * directional_derivative:
                weight = candidate
                break
            step_size *= 0.5
        else:
            if -directional_derivative < 1e-14:
                break
            raise RuntimeError("Softmax 回溯线搜索未找到下降步")
    return weight, np.array(history)

regularization = 0.08
weight_fit, training_history = fit_softmax_gradient_descent(
    X_train, Y_train, regularization=regularization
)
validation_loss, _, validation_probability = softmax_objective_gradient(
    weight_fit, X_validation, Y_validation, regularization=0.0
)
validation_prediction = validation_probability.argmax(axis=1)
validation_accuracy = np.mean(validation_prediction == y_validation)
print("训练迭代数与目标首/尾：",
      len(training_history),
      round(training_history[0], 6),
      round(training_history[-1], 6))
print("验证交叉熵/准确率：",
      round(validation_loss, 5), round(validation_accuracy, 4))
print("gauge 列和范数：", np.linalg.norm(weight_fit.sum(axis=1)))
```

### 8.2 为什么不显式形成 Hessian

完整 Hessian 维度是 $dC\times dC$，存储为 $O(d^2C^2)$。但对方向 $V\in\mathbb R^{d\times C}$：

1. 先算 logit 方向 $S=XV$；
2. 对每行应用 $J_iS_i$；
3. 左乘 $X^T$。

$$
H[V]
=\frac1nX^T
\left[
P\odot
\left(
S-(P\odot S)\mathbf1\mathbf1^T
\right)
\right]
+\lambda P_{\text{reg}}\odot V.
$$

实现时行均值项是 $\sum_cp_{ic}S_{ic}$。

```python
def softmax_hessian_vector_product(
    weight, direction, X, regularization=0.0
):
    probability = stable_softmax(X @ weight)
    logit_direction = X @ direction
    weighted_mean = np.sum(
        probability * logit_direction, axis=1, keepdims=True
    )
    class_response = probability * (logit_direction - weighted_mean)
    product = X.T @ class_response / len(X)
    product += regularization * penalty_mask * direction
    return product

direction_rng = np.random.default_rng(410)
direction = direction_rng.normal(size=weight_fit.shape)
direction = center_class_gauge(direction)
hvp_analytic = softmax_hessian_vector_product(
    weight_fit, direction, X_train, regularization
)
hvp_epsilon = 1e-5
gradient_plus = softmax_objective_gradient(
    weight_fit + hvp_epsilon * direction,
    X_train,
    Y_train,
    regularization,
)[1]
gradient_minus = softmax_objective_gradient(
    weight_fit - hvp_epsilon * direction,
    X_train,
    Y_train,
    regularization,
)[1]
hvp_numeric = (gradient_plus - gradient_minus) / (2 * hvp_epsilon)
print("HVP 最大误差：", np.max(np.abs(hvp_analytic - hvp_numeric)))
```

### 8.3 多类完全分离

若存在 $W$ 让每个样本真实类 logit 严格大于其他类，沿这些差值方向放大参数会让训练交叉熵趋近 0，有限 MLE 仍可能不存在。L2、label smoothing 或不可分样本会限制无穷放大，但它们解决的问题不同：

- L2 给参数范数明确代价；
- label smoothing 改变训练目标分布；
- 不可分数据通过冲突样本形成有限折中；
- gauge 只去除不改变预测的共同平移，不限制类别间隔。

<div class="ml-stop"><strong>停下来推一推 3</strong> 共同平移的参数发散与类别间隔发散有什么不同？前者从头到尾不改变概率，是纯坐标冗余；后者让预测越来越 one-hot，会持续降低可分数据的交叉熵。</div>

## 9. 温度缩放改变置信度，不改变赢家

定义

$$
p_c(T)=\frac{e^{z_c/T}}{\sum_je^{z_j/T}},
\qquad T>0.
$$

- $T<1$ 放大 logit 差，分布更尖锐；
- $T>1$ 缩小差值，分布更平坦；
- 任意 $T>0$ 保持 argmax 不变；
- $T\to0^+$ 接近 one-hot argmax；
- $T\to\infty$ 接近均匀分布。

### 冻结模型上的温度校准

训练完成后，用独立验证批次选择单个 $T$ 最小化 NLL：

$$
T^*=\arg\min_{T>0}
-\sum_i\log
\operatorname{softmax}(z_i/T)_{y_i}.
$$

它不改变类别排序和 top-1 预测，只调整置信度。温度校准不能修复类别特异偏差、输入相关错校准或标签漂移。

```python
validation_logits = X_validation @ weight_fit
temperature_grid = np.linspace(0.35, 3.0, 266)

def multiclass_log_loss_from_logits(logits, integer_label, temperature=1.0):
    scaled = logits / temperature
    log_probability = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return -np.mean(log_probability[np.arange(len(integer_label)), integer_label])

validation_temperature_loss = np.array([
    multiclass_log_loss_from_logits(
        validation_logits, y_validation, temperature
    )
    for temperature in temperature_grid
])
selected_temperature = temperature_grid[
    int(np.argmin(validation_temperature_loss))
]

test_logits = X_test @ weight_fit
test_probability_raw = stable_softmax(test_logits)
test_probability_scaled = stable_softmax(
    test_logits / selected_temperature
)
test_log_loss_raw = multiclass_log_loss_from_logits(
    test_logits, y_test, temperature=1.0
)
test_log_loss_scaled = multiclass_log_loss_from_logits(
    test_logits, y_test, temperature=selected_temperature
)
test_brier_raw = np.mean(np.sum((test_probability_raw - Y_test) ** 2, axis=1))
test_brier_scaled = np.mean(np.sum((test_probability_scaled - Y_test) ** 2, axis=1))
print("验证选择温度：", round(selected_temperature, 3))
print("测试 NLL 原始/温度缩放：",
      round(test_log_loss_raw, 5), round(test_log_loss_scaled, 5))
print("测试 Brier 原始/温度缩放：",
      round(test_brier_raw, 5), round(test_brier_scaled, 5))
```

<MlQuiz kind="温度辨析" question="对冻结 logits 使用正温度 T=2，哪项一定不变？" :options="['每类概率', '预测熵', 'argmax 类别与 logit 排序', '交叉熵']" :answer="2" explanation="除以正数保持大小顺序，但概率差、熵和 NLL 都会改变。" />

## 10. Label smoothing 改变训练目标

标准 one-hot 标签 $y$ 可替换为

$$
\tilde y
=(1-\varepsilon)y+\frac{\varepsilon}{C}\mathbf1.
$$

真实类目标从 1 变为

$$
1-\varepsilon+\frac{\varepsilon}{C},
$$

其他类目标为 $\varepsilon/C$。梯度仍是

$$
\nabla_z\ell=p-\tilde y.
$$

它会抑制无限增大的类别间隔，并可作为正则化手段，但要明确：

- 它不是自动得到的真实标签不确定性；
- $\varepsilon$ 是训练超参数；
- 它改变最优预测目标；
- 过度 smoothing 会伤害少数类与可分辨性；
- 验证仍要使用真实标签和部署指标。

```python
smoothing = 0.10
smoothed_target = (
    (1 - smoothing) * Y_train
    + smoothing / class_count
)
smoothed_loss, smoothed_gradient, smoothed_probability = (
    softmax_objective_gradient(
        weight_fit, X_train, smoothed_target, regularization
    )
)
hard_loss, hard_gradient, _ = softmax_objective_gradient(
    weight_fit, X_train, Y_train, regularization
)
print("平滑目标行和范围：",
      smoothed_target.sum(axis=1).min(),
      smoothed_target.sum(axis=1).max())
print("硬/平滑目标 loss：", round(hard_loss, 5), round(smoothed_loss, 5))
print("两种梯度差范数：", np.linalg.norm(hard_gradient - smoothed_gradient))
```

<MlQuiz kind="平滑语义" question="label smoothing ε=0.1 的正确描述是什么？" :options="['把模型输出概率统一乘 0.9', '把训练目标从 one-hot 改为带少量均匀质量的分布', '允许多个标签同时为真', '只改变部署阈值']" :answer="1" explanation="它修改交叉熵目标分布，不等同于多标签，也不是仅在推理时调整概率。" />

## 11. Argmax 只对应对称 0-1 损失

若动作就是选择一个类别，且所有误分类代价相同：

$$
a^*(x)=\arg\max_cp_c(x).
$$

工业处置通常有非对称代价。设损失矩阵 $L_{y,a}$ 表示真实类 $y$ 下采取动作 $a$ 的损失，条件风险：

$$
R(a\mid x)=\sum_yp_y(x)L_{y,a}.
$$

最优动作：

$$
\boxed{
a^*(x)=\arg\min_a p(x)^TL_{:,a}
}.
$$

Softmax 概率层与成本动作层必须分开。提高少数类训练权重和修改处置成本也不是同一件事。

```python
cost_matrix = np.array([
    [0.0, 5.0, 6.0, 9.0],
    [28.0, 0.0, 9.0, 14.0],
    [34.0, 10.0, 0.0, 16.0],
    [48.0, 15.0, 13.0, 0.0],
])
argmax_action = test_probability_scaled.argmax(axis=1)
conditional_risk = test_probability_scaled @ cost_matrix
minimum_risk_action = conditional_risk.argmin(axis=1)

argmax_realized_cost = np.mean(cost_matrix[y_test, argmax_action])
risk_realized_cost = np.mean(cost_matrix[y_test, minimum_risk_action])
print("argmax 与最小风险动作不同的样本数：",
      np.sum(argmax_action != minimum_risk_action))
print("测试实际平均成本 argmax/风险动作：",
      round(argmax_realized_cost, 4), round(risk_realized_cost, 4))
```

<MlQuiz kind="决策规则" question="轴承异常漏放代价远高于把合格件送复核的代价。最终动作是否仍应固定使用 argmax？" :options="['是，Softmax 只允许 argmax', '否，应把后验概率与损失矩阵相乘后选择最低条件风险动作', '只需给轴承类别改名', '把所有概率平方即可']" :answer="1" explanation="argmax 是对称 0-1 损失的特例。非对称工业代价属于独立决策层。" />

## 12. 评价：不只看整体准确率

四分类至少报告：

- 每类 support、召回率和精确率；
- 混淆矩阵，尤其是危险类别被错到哪里；
- macro 指标，避免多数类遮蔽少数类；
- NLL 和 multiclass Brier，评价概率质量；
- top-$k$ 指标，仅在后续流程真的允许多个候选时使用；
- 置信度—正确率可靠性；
- 拒绝/复核覆盖率与成本账本；
- 按批次、供应商和工况切片。

Multiclass Brier 可写为

$$
\frac1n\sum_i\sum_c(p_{ic}-y_{ic})^2.
$$

它同时惩罚真实类概率不足和错误类概率过高。NLL 对“自信但错”更敏感。

```python
test_prediction = test_probability_scaled.argmax(axis=1)
confusion = np.zeros((class_count, class_count), dtype=int)
for truth, prediction in zip(y_test, test_prediction):
    confusion[truth, prediction] += 1
per_class_recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
per_class_precision = np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1)
macro_recall = per_class_recall.mean()

confidence = test_probability_scaled.max(axis=1)
correct = (test_prediction == y_test).astype(float)
bin_edges = np.linspace(0, 1, 6)
calibration_rows = []
for left, right in zip(bin_edges[:-1], bin_edges[1:]):
    member = (confidence >= left) & (
        confidence < right if right < 1 else confidence <= right
    )
    if member.any():
        calibration_rows.append((
            int(member.sum()),
            float(confidence[member].mean()),
            float(correct[member].mean()),
        ))

print("测试混淆矩阵：\n", confusion)
print("每类召回：", np.round(per_class_recall, 3))
print("每类精确率：", np.round(per_class_precision, 3))
print("macro recall：", round(macro_recall, 4))
print("置信度箱 (n, mean_confidence, accuracy)：", calibration_rows)
```

## 数值验证：从概率单纯形到部署账本

本章实现必须满足：

1. 极端 logits 下概率有限且行和为 1；
2. 共同平移不改变概率；
3. 参考类与和为零定规不改变概率；
4. 解析矩阵梯度通过有限差分；
5. Softmax Jacobian 半正定且 $J\mathbf1=0$；
6. HVP 与梯度方向差分一致；
7. 训练目标单调不增；
8. label smoothing 目标仍位于概率单纯形；
9. 成本动作与混淆矩阵账本守恒。

```python
assert np.isfinite(extreme_probability).all()
assert np.allclose(extreme_probability.sum(axis=1), 1.0, atol=1e-14)
assert np.max(np.abs(analytic_gradient - numeric_gradient)) < 1e-7
assert np.linalg.eigvalsh(softmax_jacobian).min() > -1e-12
assert np.linalg.norm(softmax_jacobian @ np.ones(class_count)) < 1e-12
assert np.max(np.abs(probe_probability - shifted_probability)) < 1e-12
assert np.max(np.abs(probe_probability - reference_probability)) < 1e-12
assert np.max(np.abs(probe_probability - centered_probability)) < 1e-12
assert np.max(np.abs(hvp_analytic - hvp_numeric)) < 1e-7
assert np.all(np.diff(training_history) <= 1e-12)
assert np.allclose(smoothed_target.sum(axis=1), 1.0)
assert confusion.sum() == len(y_test)
assert minimum_risk_action.shape == y_test.shape
print("Softmax 稳定性、梯度、曲率、定规、HVP 与部署账本检查通过。")
```

## 复杂度与工程边界

设样本数 $n$、参数特征数 $d$、类别数 $C$：

| 操作 | 时间 | 主要内存 | 说明 |
|---|---:|---:|---|
| logits $XW$ | $O(ndC)$ | $O(nC)$ | 可分批计算 |
| Softmax + CE | $O(nC)$ | $O(nC)$ | 必须稳定 LSE |
| 完整梯度 | $O(ndC)$ | $O(dC)$ 附加 | 适合 GD/L-BFGS |
| 完整 Hessian | 最高 $O(nd^2C^2)$ | $O(d^2C^2)$ | 通常不显式形成 |
| HVP | $O(ndC)$ | $O(nC+dC)$ | 适合 Newton-CG |
| 推理单样本 | $O(dC)$ | $O(C)$ | 还要做稳定归一化 |

当 $C$ 很大时，完整归一化成为瓶颈：

- hierarchical softmax 可把单标签路径计算降到约 $O(\log C)$；
- sampled softmax / negative sampling 用采样近似训练归一化；
- 两阶段召回—排序把大标签空间拆开；
- 类别树与采样分布会引入新的偏差和维护成本。

本章只有 4 类，不应为了“可扩展”引入这些近似。精确 Softmax 更简单、更可审计。

### 工业边界

- Softmax 不能自动识别“以上皆非”；封闭类别集外的新故障会被硬分到某个已知类。
- 最大概率低可触发拒绝，但阈值必须在未知/漂移样本上验证。
- 输出和为 1 不代表概率已校准。
- 类别先验随供应商或设备变化时，截距与概率都会漂移。
- 类别合并、拆分和命名变化属于标签模式变更，需要模型与评价版本迁移。
- 训练标签由主导处置裁决产生时，模型学习的是裁决规则，不一定是纯物理根因。

## 13. 常见失效模式

| 现象 | 首要假设 | 证据 | 修复方向 |
|---|---|---|---|
| loss 出现 `nan` | 直接指数溢出 | 极端 logits、朴素 exp | 使用 LSE/减最大值 |
| 参数列整体漂移 | gauge 未固定 | 概率不变、列共同变化 | 参考类或和为零约束 |
| 训练准确率 100%、范数继续增长 | 多类完全分离 | 间隔、范数、loss 轨迹 | L2、smoothing、数据审计 |
| 多故障样本总被迫选一个 | 标签合同错误 | 专家多标签记录 | 改为 multi-label Sigmoid |
| 整体准确率高、轴承召回低 | 类别不平衡或代价错位 | 混淆矩阵、macro 指标 | 数据、阈值/成本层、权重语义 |
| AUC/准确率不变但 NLL 变差 | 置信度失准 | 可靠性图、NLL、Brier | 独立温度校准或重训 |
| 新故障被高置信归到旧类 | open-set 漂移 | 距离、低密度、审计样本 | 拒绝机制、异常检测、扩充类别 |
| 不同库系数无法直接比较 | 参考类/方向/正则不同 | 还原概率与 pairwise 差 | 统一坐标和目标协议 |

排查顺序：

1. 核对互斥/多标签合同、类别版本和特征截止时间；
2. 核对矩阵形状、one-hot 顺序、参考类和 gauge；
3. 用极端 logits 测稳定 LSE；
4. 做矩阵梯度与 HVP 有限差分；
5. 检查分离、参数范数和正则化尺度；
6. 在独立批次验证概率、每类指标和成本动作；
7. 最后才更换优化器或模型。

## 14. 上线验收清单

- [ ] 类别互斥性、主导处置规则和未知类处理已定义。
- [ ] 类别编号与 one-hot 列顺序被版本化。
- [ ] 分组切分与部署批次层级一致。
- [ ] 标准化只在训练批次拟合。
- [ ] Softmax 与交叉熵从 logits 稳定计算。
- [ ] 参考类或和为零 gauge 已写入模型合同。
- [ ] 目标 sum/mean、L2、截距惩罚和类别权重已锁定。
- [ ] 矩阵梯度、Jacobian 零方向和 HVP 已数值验证。
- [ ] 已检查多类完全分离与参数范数轨迹。
- [ ] 温度只在独立验证批次选择，测试只做一次验收。
- [ ] 每类召回、精确率、NLL、Brier、校准和成本均达标。
- [ ] 低置信、未知故障和强制工艺规则有独立路径。

## 15. 章节练习

### 练习 A：手算三类 Softmax

给定

$$
z=[2,0,-1]^T,
\qquad
y=[0,1,0]^T.
$$

完成：

1. 用减最大值计算概率；
2. 计算交叉熵；
3. 计算 $\nabla_z\ell$；
4. 写出 $J_{\mathrm{softmax}}$；
5. 验证 $J\mathbf1=0$；
6. 解释为什么给 $z$ 加 1000 不改变结果。

### 练习 B：从零实现矩阵训练器

补全下面的求解器。要求支持：

- 稳定 log-sum-exp；
- one-hot 与 smoothed target；
- 截距不惩罚；
- 和为零 gauge；
- 回溯线搜索；
- 梯度与 HVP 数值检查；
- 每轮 loss、梯度范数、参数范数和最大 logit 审计。

```python
# TODO: 练习骨架，验证脚本会跳过本代码块。
def auditable_softmax_fit(X, target, regularization, smoothing=0.0):
    weight = np.zeros((X.shape[1], target.shape[1]))
    audit_rows = []
    # 1. 构造平滑目标并检查每行和为 1
    # 2. 稳定计算 loss、概率和矩阵梯度
    # 3. 固定 gauge，执行带回溯的下降步
    # 4. 记录审计量并判断停止
    return weight, audit_rows
```

### 练习 C：证明完整 Hessian 半正定

从

$$
J_i=\operatorname{diag}(p_i)-p_ip_i^T
$$

和

$$
x_ix_i^T
$$

均半正定出发，证明

$$
J_i\otimes(x_ix_i^T)
$$

半正定。再构造 $d$ 个线性无关的共同类平移方向，说明完整参数化为何至少有这些零方向。

### 练习 D：多分类还是多标签

为以下任务分别选择 Softmax、独立 Sigmoid 或其他输出，并写出理由：

1. 每台电机唯一主导处置工位；
2. 每台电机可能同时命中多个故障标签；
3. 一个已知总数的批次中各缺陷类型计数；
4. 允许“未知故障”的开放集识别；
5. 专家给出四类概率分布而不是硬标签。

<MlQuiz kind="综合验收" question="Softmax 模型测试准确率很高，是否足以证明可上线？" :options="['足够', '不够；还需验证标签互斥合同、稳定性、gauge、分离、每类概率质量、成本动作和未知类路径', '只需再看训练 loss', '把温度固定为 1 即可']" :answer="1" explanation="准确率只覆盖对称 argmax 结果，不能证明概率、数值实现、参数坐标和工业决策边界正确。" />

## 16. 本章结论

Softmax 回归的理论闭环是：

$$
\boxed{
\text{Categorical 互斥标签}
\Rightarrow
\text{相对 log odds}
\Rightarrow
\text{Softmax}
\Rightarrow
\text{交叉熵}
\Rightarrow
X^T(P-Y)
}
$$

同时要记住：

1. **所有类别共享概率质量**：它们不是独立 Sigmoid；
2. **共同平移是零方向**：概率可辨识，完整参数不唯一；
3. **稳定 LSE 是必要实现**：减最大值是精确变换；
4. **温度、smoothing 与成本属于不同层**：不能把旋钮混用；
5. **互斥性来自标签合同**：物理故障可共存时应改为多标签。

下一章从概率似然转向几何间隔：感知机只在犯错时更新，支持向量机则把“分对”升级为“以最大几何间隔分对”。

## 来源与许可

- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 84–87 页，§2.5.1–2.5.4：Categorical/Multinomial、Softmax、温度、二分类约化与 log-sum-exp。用于概率结构、术语和数值稳定性核对；不复用原书 Iris 图、骰子示例或表达。
- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 381–385 页，§10.3–10.3.3：多类逻辑回归、互斥/多标签边界、交叉熵、梯度、Hessian、凸性与优化。用于推导和算法边界交叉核对；本章电机案例、代码和实验独立创作。
- Stanford CS229 Lecture Notes，PDF 第 26–28 页，§2.3：多分类 logits、Softmax、条件似然、交叉熵与逐类梯度。用于矩阵推导顺序核对；不复用邮件分类示例或原文图形。
- Jonathan Richard Shewchuk，*Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 104–105 页（讲义页码 98–99），Lecture 17：Softmax 输出、one-hot/软目标、交叉熵与输出层误差梯度。用于输出耦合和神经网络前置关系核对；不复制课堂图形、MNIST 示例或措辞。

本章 2,400 台电机、60 个生产批次、四类主导处置标签、传感器与成本定义、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实电机故障率、台架性能、维修策略、放行阈值或安全规范。

<MlLessonProgress lesson-id="10" next-label="第 11 章：感知机与支持向量机" next-href="/courses/machine-learning/11-margin-svm" />
