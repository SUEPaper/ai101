---
course: machine-learning
edition: industrial
title: k 近邻算法（kNN）
description: 以离心泵历史检维修案例匹配为贯穿案例，从局部常数估计推导 kNN 分类、回归、距离加权、Voronoi、偏差—方差和一致性直觉，并用分组验证、距离集中、开放集拒绝和查询预算建立工业验收门禁。
outline: [2, 3]
---

<MlLessonHero
  number="18"
  title="k 近邻算法（kNN）"
  question="新泵组出现振动和温升异常时，能否不先拟合一条全局方程，而是直接调取最相似的历史检维修案例？如果可以，谁定义‘相似’，邻域多大才可信，又怎样避免把同一台设备的重复工单当成泛化能力？"
  duration="建议 600–720 分钟"
  :outcomes="[
    '从局部常数假设推导 kNN 分类后验、回归均值、距离加权与确定性平票协议，手算邻居和 Voronoi 边界',
    '解释距离度量、特征尺度、k 与邻域半径如何共同控制偏差—方差，并用分组验证冻结预处理和超参数',
    '从零实现批量 kNN、开放集拒绝和缺失距离，验证高维距离集中、查询复杂度与精确/近似索引边界'
  ]"
/>

第 17 章通过可学习投影决定“读取哪些 token”。本章把表示先固定下来，直接询问：与当前样本最相似的历史样本是什么，它们的标签能否代表当前局部？

kNN 看起来没有训练循环，却绝不等于“没有模型选择”。距离、特征变换、邻居数 $k$、投票权重、平票规则、拒绝阈值和样本库版本共同定义了预测函数。训练成本被推迟到查询时，统计假设也被浓缩成一句必须审计的话：

> 在所选表示和距离下，足够接近的历史工况应具有相近的目标。

## 贯穿案例：离心泵历史检维修案例分诊

一家流程工厂积累了多台离心泵的状态监测窗口与检维修结论。新告警到达后，可靠性工程师希望完成两个辅助任务：

1. 将工单初步分到“叶轮不平衡”“联轴器不对中”或“润滑不足”专业路线；
2. 估计本次检修的停机小时数，用于排班和备件准备。

模型只提供历史相似案例和分诊建议，不直接停泵、不修改保护值，也不代替工程师确认。特别是，新型故障或越界工况不能被迫归入三个已知类别。

### 数据合同与错误代价

| 字段 | 物理含义 | 边界与风险 |
|---|---|---|
| `vibration_rms` | 泵壳振动速度 RMS，mm/s | 采样位置和带宽必须一致 |
| `axial_ratio` | 轴向能量占总振动能量比例 | 比值无量纲，但受测点方向影响 |
| `temperature_rise` | 轴承温升，°C | 环境基准不能使用事故后的回填值 |
| `current_imbalance` | 三相电流不平衡比例 | 负载变化会形成混杂 |
| `load_fraction` | 额定流量归一化负载 | 超出训练运行域应拒绝 |
| `pump_id` | 物理设备标识 | 同一泵的重复窗口不得跨数据集 |
| `root_cause` | 检维修确认的主因 | 复合故障、待确认和误标需单独登记 |
| `maintenance_hours` | 从隔离到恢复的停机时长 | 同时受班次、备件和审批影响，不是纯机理量 |

错把润滑不足分到叶轮路线可能延误供油检查；错把不平衡分到润滑路线会造成无效拆检。开放集样本若被强制贴上已知标签，风险通常高于“转人工复核”。因此最终系统应评价分层召回、转派成本、拒绝率与查询延迟，而不只看总体准确率。

### 1. 先按泵组切分，再建立案例库

下面生成 30 台泵、每台 5 次重复窗口，共 150 条教学记录。泵级偏移刻意大于单次测量噪声，用来检验随机按行切分造成的近重复泄漏。

```python
import numpy as np

rng = np.random.default_rng(20260805)
pump_count = 30
records_per_pump = 5
record_count = pump_count * records_per_pump
feature_names = np.array([
    "vibration_rms", "axial_ratio", "temperature_rise",
    "current_imbalance", "load_fraction",
])
cause_names = np.array(["叶轮不平衡", "联轴器不对中", "润滑不足"])

pump_id = np.repeat(np.arange(pump_count), records_per_pump)
pump_cause = np.arange(pump_count) % 3
root_cause = pump_cause[pump_id]
class_centers = np.array([
    [4.6, 0.25, 10.0, 0.080, 0.76],
    [6.1, 0.68, 14.0, 0.105, 0.78],
    [5.2, 0.40, 23.0, 0.070, 0.72],
])
pump_offset_scale = np.array([0.75, 0.11, 2.2, 0.018, 0.075])
record_noise_scale = np.array([0.18, 0.025, 0.65, 0.006, 0.020])
pump_offsets = rng.normal(0.0, pump_offset_scale, size=(pump_count, 5))

measurements = (
    class_centers[root_cause]
    + pump_offsets[pump_id]
    + rng.normal(0.0, record_noise_scale, size=(record_count, 5))
)
measurements[:, 1] = np.clip(measurements[:, 1], 0.02, 0.98)
measurements[:, 3] = np.clip(measurements[:, 3], 0.001, None)
measurements[:, 4] = np.clip(measurements[:, 4], 0.30, 1.10)
maintenance_hours = (
    np.array([5.0, 7.5, 4.0])[root_cause]
    + 0.18 * measurements[:, 2]
    + 1.6 * measurements[:, 3] * 10
    + rng.normal(0.0, 0.35, size=record_count)
)

train_pumps, validation_pumps, test_pumps = [], [], []
for class_index in range(3):
    candidates = np.where(pump_cause == class_index)[0]
    ordered = rng.permutation(candidates)
    train_pumps.extend(ordered[:6])
    validation_pumps.extend(ordered[6:8])
    test_pumps.extend(ordered[8:])
train_pumps = np.array(train_pumps)
validation_pumps = np.array(validation_pumps)
test_pumps = np.array(test_pumps)
train_rows = np.isin(pump_id, train_pumps)
validation_rows = np.isin(pump_id, validation_pumps)
test_rows = np.isin(pump_id, test_pumps)

assert measurements.shape == (150, 5)
assert (train_rows.sum(), validation_rows.sum(), test_rows.sum()) == (90, 30, 30)
assert set(train_pumps).isdisjoint(validation_pumps)
assert set(train_pumps).isdisjoint(test_pumps)
assert set(validation_pumps).isdisjoint(test_pumps)
assert np.all(np.bincount(root_cause[train_rows]) == 30)
print("训练/验证/测试案例：", train_rows.sum(), validation_rows.sum(), test_rows.sum())
print("训练/验证/测试泵数：", len(train_pumps), len(validation_pumps), len(test_pumps))
```

这不是一个“训练集、测试集随机分 8:2”的普通表格问题。同一台泵的五个窗口共享装配偏差、传感器安装、工况和维护历史。若近重复样本跨集合，kNN 会直接检索到同一设备的近邻，验证分数近似变成身份识别能力。

<MlQuiz kind="数据合同" question="同一台泵有五个相邻状态窗口，评估模型迁移到未见泵组的能力时应怎样切分？" :options="['五个窗口随机分到不同集合', '以 pump_id 为组，让同一泵只出现在一个集合', '只删除标签重复的窗口', '把测试窗口加入案例库但不显示标签']" :answer="1" explanation="相邻窗口共享设备特征，必须按部署时真正未见的实体边界切分；否则 kNN 尤其容易用近重复样本获得虚高分数。" />

### 2. 预处理本身就是距离模型的一部分

欧氏距离把不同特征的数值差直接平方相加。温升的数值范围远大于电流不平衡比例，若使用原值，单位而非业务语义会决定邻居。标准化参数必须只由训练案例库估计。

```python
training_mean = measurements[train_rows].mean(axis=0)
training_scale = measurements[train_rows].std(axis=0)
if np.any(training_scale <= 0):
    raise ValueError("训练特征包含零方差列，不能直接用于标准化距离")

scaled_measurements = (measurements - training_mean) / training_scale
assert np.allclose(scaled_measurements[train_rows].mean(axis=0), 0.0, atol=1e-12)
assert np.allclose(scaled_measurements[train_rows].std(axis=0), 1.0, atol=1e-12)
assert not np.allclose(scaled_measurements[validation_rows].mean(axis=0), 0.0)
```

最后一条断言很重要：验证集没有义务在训练统计量下恰好均值为 0。若它被强制标准化到自身均值为 0，验证阶段已经使用了验证分布信息。

## 2. kNN 是局部常数估计器

给定训练集

$$
\mathcal D=\{(\mathbf x_i,y_i)\}_{i=1}^n,
$$

按距离 $d$ 选出查询 $\mathbf x$ 的 $k$ 个近邻索引集合 $\mathcal N_k(\mathbf x)$。

### 2.1 分类：局部类别频率

对类别 $c$，等权 kNN 估计

$$
\widehat p(y=c\mid\mathbf x,\mathcal D)
=\frac1k\sum_{i\in\mathcal N_k(\mathbf x)}\mathbb I(y_i=c),
$$

并预测

$$
\widehat y(\mathbf x)
=\arg\max_c\widehat p(y=c\mid\mathbf x,\mathcal D).
$$

它假设条件类别概率在查询附近近似不变。票数 $3/5$ 是邻域内样本频率，不自动是校准后的 60% 故障概率：邻域由数据自适应选择，样本可能相关，$k$ 也经过模型选择。

### 2.2 回归：局部目标均值

回归使用

$$
\widehat f(\mathbf x)
=\frac1k\sum_{i\in\mathcal N_k(\mathbf x)}y_i.
$$

这同样是局部常数假设：在邻域内用一个均值近似未知回归函数。若目标随距离方向快速变化，局部线性回归可能比简单平均更合适。

“非参数”表示模型复杂度可随样本库增长，不表示没有选择：$k$、距离、表示、权重和拒绝协议都是超参数或拟合产物。

<KnnVoteInspectionBayLab />

<MlQuiz kind="概念辨析" question="一个 5-NN 分类器在当前邻域发现 3 个不对中、2 个不平衡案例。最准确的解释是什么？" :options="['局部经验频率为 3/5，但不保证是校准概率', '不对中的真实概率已被严格证明为 60%', '全局训练集中不对中比例一定是 60%', '模型已经拟合了五个连续参数']" :answer="0" explanation="邻域投票是局部频率估计；选择偏差、样本相关性和邻域大小都会影响其概率解释。" />

<div class="ml-stop"><strong>停下来推一推 1</strong>查询点的五个邻居标签依次为 A、B、A、C、B。写出三类等权票数和预测；若 A 与 B 的票数仍相同，设计一个与输入顺序无关的确定性平票规则。</div>

## 3. 距离先定义“谁有资格提供证据”

### 3.1 Minkowski 距离

常见族为

$$
d_p(\mathbf x,\mathbf z)
=\left(\sum_{j=1}^d w_j|x_j-z_j|^p\right)^{1/p},
\qquad p\ge1,\;w_j\ge0.
$$

- $p=2$：欧氏距离，等距面是椭球；
- $p=1$：曼哈顿距离，单个大差异不会被平方进一步放大；
- $p=\infty$：Chebyshev 距离，只看最大加权坐标差；
- $w_j$：领域权重，既表达量纲也表达认为哪些方向重要。

下面实现系统边界检查，不接受 NaN、负权重、空特征或不一致维度。

```python
def pairwise_minkowski(queries, references, p=2, feature_weights=None):
    queries = np.asarray(queries, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    if queries.ndim != 2 or references.ndim != 2:
        raise ValueError("queries 和 references 必须是二维矩阵")
    if queries.shape[1] != references.shape[1] or queries.shape[1] == 0:
        raise ValueError("查询与案例库的特征维必须一致且非空")
    if not np.isfinite(queries).all() or not np.isfinite(references).all():
        raise ValueError("普通 Minkowski 距离不接受 NaN 或 Inf")
    if p < 1 and p != np.inf:
        raise ValueError("p<1 不满足度量的三角不等式")
    if feature_weights is None:
        weights = np.ones(queries.shape[1])
    else:
        weights = np.asarray(feature_weights, dtype=np.float64)
    if weights.shape != (queries.shape[1],) or np.any(weights < 0):
        raise ValueError("feature_weights 必须为非负 d 维向量")
    difference = np.abs(queries[:, None, :] - references[None, :, :])
    if p == np.inf:
        return np.max(difference * weights[None, None, :], axis=-1)
    return np.sum(weights[None, None, :] * difference ** p, axis=-1) ** (1 / p)

distance_smoke = pairwise_minkowski(
    np.array([[0.0, 0.0], [1.0, 1.0]]),
    np.array([[3.0, 4.0], [1.0, 2.0]]),
    p=2,
)
assert np.allclose(distance_smoke[0], [5.0, np.sqrt(5.0)])
assert distance_smoke.shape == (2, 2)
```

### 3.2 一个手算邻域

```python
manual_cases = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 2.0],
    [2.0, 2.0],
])
manual_labels = np.array([0, 0, 1, 1])
manual_query = np.array([[0.8, 0.6]])
manual_distance = pairwise_minkowski(manual_query, manual_cases, p=2)[0]
manual_order = np.argsort(manual_distance, kind="stable")

assert manual_order.tolist() == [1, 0, 2, 3]
assert np.allclose(manual_distance, [1.0, np.sqrt(0.4), np.sqrt(2.6), np.sqrt(3.4)])
assert np.bincount(manual_labels[manual_order[:3]], minlength=2).tolist() == [2, 1]
```

相同坐标、有限精度或离散特征可能产生距离平局。若实现让 `argsort`、线程调度或案例写入顺序决定结果，线上重放会不一致。应以稳定样本键排序，分类票数平局再按总距离、先验或转人工规则确定。

<MlQuiz kind="结果预测" question="查询 (0.8,0.6) 到案例 (1,0) 的欧氏距离是多少？" :options="['√0.4', '0.4', '√1.4', '1.4']" :answer="0" explanation="两维差为 −0.2 和 0.6，平方和为 0.04+0.36=0.4，欧氏距离为 √0.4。" />

### 3.3 Mahalanobis 距离：旋转并缩放空间

若特征相关，可用

$$
d_M^2(\mathbf x,\mathbf z)
=(\mathbf x-\mathbf z)^\top\mathbf M(\mathbf x-\mathbf z),
\qquad \mathbf M\succeq0.
$$

$\mathbf M=\mathbf I$ 退化为欧氏距离。若 $\mathbf M=\boldsymbol\Sigma^{-1}$，相关且高方差方向被降权。协方差必须只在训练折估计；当 $d\ge n$ 或特征近共线时，样本协方差可能奇异，需要收缩、降维或受约束的度量学习。

```python
training_covariance = np.cov(scaled_measurements[train_rows], rowvar=False, ddof=0)
shrinkage = 0.15
shrunk_covariance = (
    (1 - shrinkage) * training_covariance
    + shrinkage * np.eye(training_covariance.shape[0])
)
mahalanobis_matrix = np.linalg.inv(shrunk_covariance)
mahalanobis_cholesky = np.linalg.cholesky(mahalanobis_matrix)
mahalanobis_delta = scaled_measurements[validation_rows][0] - scaled_measurements[train_rows][0]
mahalanobis_squared = mahalanobis_delta @ mahalanobis_matrix @ mahalanobis_delta
whitened_squared = np.sum((mahalanobis_delta @ mahalanobis_cholesky) ** 2)

assert np.all(np.linalg.eigvalsh(mahalanobis_matrix) > 0)
assert np.allclose(mahalanobis_squared, whitened_squared)
```

## 4. 特征尺度：单位不能偷偷变成重要性

若温升从 °C 攂成 0.1°C 计数，数值差扩大十倍。未标准化的欧氏距离会改变，即使物理状态完全相同。训练折 Z-score

$$
x_{ij}'=\frac{x_{ij}-\mu_j^{\text{train}}}{s_j^{\text{train}}}
$$

可以抵消平移和正比例单位变换，但有三条边界：

1. 它假设每维同等重要，不会自动删除无关特征；
2. 异常值会扭曲均值与标准差，必要时使用稳健尺度；
3. 类别变量、周期角度、波形和文本需要语义匹配的表示，不能直接套欧氏距离。

<DistanceScaleCalibrationBayLab />

<MlQuiz kind="错误诊断" question="把温升单位从 °C 改为 0.1°C 计数后，未标准化 kNN 的最近案例改变。根因是什么？" :options="['数值尺度被当成了距离权重', '类别标签被自动重编码', 'kNN 只能使用整数特征', '欧氏距离不满足对称性']" :answer="0" explanation="单位倍率改变了该维对距离的贡献；这不是设备关系变化，而是表示与度量合同没有固定。" />

<div class="ml-stop"><strong>停下来推一推 2</strong>候选 A 与查询的两维差为 (0.1,10)，候选 B 为 (1,1)。分别计算原始欧氏距离；若第一维乘 100，最近邻是否改变？这说明特征缩放与领域权重有什么区别？</div>

## 5. 从 1-NN 到 Voronoi 分区

当 $k=1$ 时，每个训练样本 $\mathbf x_i$ 拥有一个 Voronoi 单元

$$
V_i=\{\mathbf x: d(\mathbf x,\mathbf x_i)\le d(\mathbf x,\mathbf x_j),\;\forall j\ne i\}.
$$

单元内所有查询都复制样本 $i$ 的标签。欧氏距离下，两点单元的公共边界位于垂直平分面；二维可画成多边形拼接。决策边界只保留不同标签单元的接触边。

一维排序样本 $x_{(1)}<\cdots<x_{(n)}$ 的相邻 Voronoi 边界就是中点：

$$
b_i=\frac{x_{(i)}+x_{(i+1)}}2.
$$

```python
voronoi_points = np.array([-2.0, -0.5, 1.5, 4.0])
voronoi_labels = np.array([0, 1, 1, 0])
voronoi_boundaries = (voronoi_points[:-1] + voronoi_points[1:]) / 2

def one_nearest_on_line(query_values, points, labels):
    distance = np.abs(np.asarray(query_values)[:, None] - points[None, :])
    return labels[np.argmin(distance, axis=1)]

assert np.allclose(voronoi_boundaries, [-1.25, 0.5, 2.75])
assert one_nearest_on_line(np.array([-1.5, 0.0, 2.0, 3.0]), voronoi_points, voronoi_labels).tolist() == [0, 1, 1, 0]
```

若每个训练坐标唯一，1-NN 通常训练误差为 0，因为样本是自己的最近邻。这不是泛化证据。若相同坐标存在冲突标签，任何确定性函数都无法同时正确分类，训练误差也未必为 0。

<MlQuiz kind="推导补全" question="一维欧氏 1-NN 中，训练点位于 2 和 8，它们的 Voronoi 分界点在哪里？" :options="['5', '3', '6', '取决于标签编码']" :answer="0" explanation="令 |x−2|=|x−8|，得到中点 x=(2+8)/2=5；标签只决定边界两侧的类别，不改变几何分界。" />

## 6. k 是偏差—方差与空间分辨率旋钮

$k=1$ 时邻域极小，模型可以跟随细碎局部结构：偏差低、方差高、对误标和离群点敏感。增大 $k$ 会扩大邻域并平均更多样本：方差通常下降，但真实局部边界也可能被抹平。

回归中，若暂时把邻域视为固定，噪声独立且方差为 $\sigma^2$：

$$
\operatorname{Var}\left(\frac1k\sum_{i\in\mathcal N_k}\varepsilon_i\right)
=\frac{\sigma^2}{k}.
$$

真实 kNN 的邻域本身由随机训练点决定，完整方差不只有这一项。增大 $k$ 还会增加第 $k$ 近邻半径 $r_k(\mathbf x)$，局部常数假设在更大区域内成立得更差，产生平滑偏差。

```python
def one_dimensional_knn_regression(x_train, y_train, query, k):
    distances = np.abs(x_train - query)
    neighbors = np.argpartition(distances, k - 1)[:k]
    return float(np.mean(y_train[neighbors]))

bias_rng = np.random.default_rng(1806)
query_location = 0.85
true_query_value = query_location ** 3
replications = 700
predictions_by_k = {}
for k_value in (1, 5, 25):
    predictions = []
    for _ in range(replications):
        x_sample = bias_rng.uniform(-1.0, 1.0, size=50)
        y_sample = x_sample ** 3 + bias_rng.normal(0.0, 0.20, size=50)
        predictions.append(one_dimensional_knn_regression(x_sample, y_sample, query_location, k_value))
    predictions_by_k[k_value] = np.array(predictions)

bias_squared = {
    k_value: (values.mean() - true_query_value) ** 2
    for k_value, values in predictions_by_k.items()
}
prediction_variance = {
    k_value: values.var()
    for k_value, values in predictions_by_k.items()
}
assert prediction_variance[25] < prediction_variance[1]
assert bias_squared[25] > bias_squared[1]
print("bias²：", {k: round(v, 4) for k, v in bias_squared.items()})
print("variance：", {k: round(v, 4) for k, v in prediction_variance.items()})
```

因此不能用训练误差选择 $k$：1-NN 会被系统性偏爱。应在冻结的分组验证集或内层交叉验证中联合选择预处理、距离、$k$、权重与拒绝阈值。

<MlQuiz kind="设计判断" question="为什么用训练准确率选择 k 几乎总会偏爱 k=1？" :options="['每个唯一训练样本通常是自己的最近邻', 'k=1 的查询复杂度恒为零', 'k=1 自动进行特征选择', '训练标签不会参与投票']" :answer="0" explanation="1-NN 会记住唯一训练坐标，训练误差缺少对方差和新样本表现的惩罚。" />

## 7. 从零实现分类与回归

### 7.1 批量分类、零距离与平票

下面的分类器使用 `argpartition` 找出前 $k$ 个候选，再在候选内部按距离与稳定索引排序。票重平局时，依次比较该类邻居总距离和类别编号；生产系统也可以把平局直接转人工。

```python
def knn_classify(reference_x, reference_y, queries, k, p=2, distance_weighted=False):
    reference_x = np.asarray(reference_x, dtype=np.float64)
    reference_y = np.asarray(reference_y)
    queries = np.asarray(queries, dtype=np.float64)
    if not 1 <= k <= len(reference_x):
        raise ValueError("k 必须位于 [1, 案例数]")
    distances = pairwise_minkowski(queries, reference_x, p=p)
    candidate_indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
    predictions, vote_rows, ordered_indices, ordered_distances = [], [], [], []
    classes = np.unique(reference_y)
    for row, candidates in enumerate(candidate_indices):
        order = np.lexsort((candidates, distances[row, candidates]))
        neighbors = candidates[order]
        neighbor_distance = distances[row, neighbors]
        neighbor_labels = reference_y[neighbors]
        if distance_weighted:
            zero = neighbor_distance <= 1e-12
            weights = zero.astype(float) if np.any(zero) else 1.0 / neighbor_distance
        else:
            weights = np.ones(k)
        votes = np.array([weights[neighbor_labels == class_value].sum() for class_value in classes])
        distance_sum = np.array([
            neighbor_distance[neighbor_labels == class_value].sum()
            if np.any(neighbor_labels == class_value) else np.inf
            for class_value in classes
        ])
        winner_order = np.lexsort((classes, distance_sum, -votes))
        predictions.append(classes[winner_order[0]])
        vote_rows.append(votes)
        ordered_indices.append(neighbors)
        ordered_distances.append(neighbor_distance)
    return (
        np.asarray(predictions), np.vstack(vote_rows),
        np.vstack(ordered_indices), np.vstack(ordered_distances),
    )

manual_prediction, manual_votes, manual_neighbors, _ = knn_classify(
    manual_cases, manual_labels, manual_query, k=3
)
assert manual_prediction.tolist() == [0]
assert manual_votes.tolist() == [[2.0, 1.0]]
assert manual_neighbors.tolist() == [[1, 0, 2]]
```

### 7.2 回归与距离加权

距离加权回归为

$$
\widehat f(\mathbf x)
=\frac{\sum_{i\in\mathcal N_k(\mathbf x)}w_i(\mathbf x)y_i}
{\sum_{i\in\mathcal N_k(\mathbf x)}w_i(\mathbf x)},
\qquad
w_i(\mathbf x)=d(\mathbf x,\mathbf x_i)^{-q}.
$$

若查询与一个或多个案例距离为 0，应只在零距离集合内平均，不能让 `1/0` 变成 Inf 后继续归一化。

```python
def knn_regress(reference_x, reference_y, queries, k, p=2, distance_power=0):
    distances = pairwise_minkowski(queries, reference_x, p=p)
    neighbor_indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
    predictions = []
    for row, neighbors in enumerate(neighbor_indices):
        neighbor_distance = distances[row, neighbors]
        targets = np.asarray(reference_y)[neighbors]
        zero = neighbor_distance <= 1e-12
        if np.any(zero):
            predictions.append(np.mean(targets[zero]))
        elif distance_power == 0:
            predictions.append(np.mean(targets))
        else:
            weights = neighbor_distance ** (-distance_power)
            predictions.append(np.sum(weights * targets) / np.sum(weights))
    return np.asarray(predictions)

regression_reference_x = np.array([[0.0], [1.0], [3.0]])
regression_reference_y = np.array([2.0, 4.0, 10.0])
assert np.allclose(knn_regress(regression_reference_x, regression_reference_y, np.array([[1.0]]), 2, distance_power=2), [4.0])
assert np.allclose(knn_regress(regression_reference_x, regression_reference_y, np.array([[2.0]]), 2), [7.0])
```

<MlQuiz kind="代码审查" question="反距离加权时查询点与两个训练案例完全重合，最稳妥的处理是什么？" :options="['只在所有零距离案例的目标之间平均', '给距离统一加一个巨大常数', '删除查询点', '任意选择最后写入的案例']" :answer="0" explanation="零距离案例应支配预测；若有多个重合案例，必须以确定规则合并它们，避免除零和输入顺序依赖。" />

## 8. 用分组验证冻结整个距离管线

### 8.1 在泵级验证集选择 k

```python
candidate_k = np.arange(1, 22, 2)
validation_accuracy_by_k = []
for k_value in candidate_k:
    prediction, _, _, _ = knn_classify(
        scaled_measurements[train_rows], root_cause[train_rows],
        scaled_measurements[validation_rows], k=int(k_value), p=2,
    )
    validation_accuracy_by_k.append(np.mean(prediction == root_cause[validation_rows]))
validation_accuracy_by_k = np.asarray(validation_accuracy_by_k)
selected_k = int(candidate_k[np.argmax(validation_accuracy_by_k)])

test_prediction, test_votes, test_neighbors, test_neighbor_distances = knn_classify(
    scaled_measurements[train_rows], root_cause[train_rows],
    scaled_measurements[test_rows], k=selected_k, p=2,
)
group_test_accuracy = float(np.mean(test_prediction == root_cause[test_rows]))
assert selected_k in candidate_k
assert group_test_accuracy >= 0.70
print("冻结 k / 泵外测试准确率：", selected_k, round(group_test_accuracy, 3))
```

验证集用于选择 $k$ 后不能继续用于无次数限制的特征试错；测试集只在完整配置冻结后打开一次。若数据量不足，应使用按泵分组的嵌套交叉验证。

### 8.2 随机按行切分的乐观偏差

```python
row_split_rng = np.random.default_rng(1808)
row_order = row_split_rng.permutation(record_count)
row_train_indices = row_order[:90]
row_validation_indices = row_order[90:120]
row_test_indices = row_order[120:]
row_mean = measurements[row_train_indices].mean(axis=0)
row_scale = measurements[row_train_indices].std(axis=0)
row_scaled = (measurements - row_mean) / row_scale

row_validation_scores = []
for k_value in candidate_k:
    prediction, _, _, _ = knn_classify(
        row_scaled[row_train_indices], root_cause[row_train_indices],
        row_scaled[row_validation_indices], int(k_value),
    )
    row_validation_scores.append(np.mean(prediction == root_cause[row_validation_indices]))
row_selected_k = int(candidate_k[np.argmax(row_validation_scores)])
row_test_prediction, _, row_test_neighbors, _ = knn_classify(
    row_scaled[row_train_indices], root_cause[row_train_indices],
    row_scaled[row_test_indices], row_selected_k,
)
row_test_accuracy = float(np.mean(row_test_prediction == root_cause[row_test_indices]))
same_pump_neighbor_rate = np.mean(
    pump_id[row_train_indices][row_test_neighbors[:, 0]] == pump_id[row_test_indices]
)
print("随机行切分测试准确率 / 最近邻同泵率：", round(row_test_accuracy, 3), round(float(same_pump_neighbor_rate), 3))
assert same_pump_neighbor_rate > 0.70
```

这条同泵率是泄漏证据，不以“随机切分分数是否恰好更高”作为唯一判断。即使一次随机种子下总体准确率没有明显上升，验证问题已经从“未见泵组泛化”变成“同泵重复窗口识别”。

<MlQuiz kind="错误诊断" question="随机行切分后准确率很高，且 85% 测试样本的最近邻来自同一 pump_id。应怎样解释？" :options="['评估被设备近重复泄漏污染，不能证明迁移到新泵', '说明 kNN 已学会因果机理', '只需把 k 增大到偶数', '说明不需要测试集']" :answer="0" explanation="最近邻正在利用同一实体的稳定指纹；若部署目标是新泵组，这种切分改变了要回答的问题。" />

<div class="ml-stop"><strong>停下来推一推 3</strong>列出一个五折分组交叉验证流程：每折中何时估计标准化参数、何时建立案例库、何时选择 k。若还要选择拒绝阈值，最终测试集可以被使用几次？</div>

## 9. 缺失值与混合字段不是填零就结束

标准欧氏距离不能直接处理 NaN。把缺失填 0 会把“未测量”变成“恰好为训练均值”（标准化后）或物理零值。可选策略包括：

- 在训练折拟合插补器并增加缺失指示；
- 只在双方共同观测维度上计算归一化距离，并要求最小重叠数；
- 对类别和数值字段使用 Gower 一类混合距离；
- 对波形、频谱或文本先学习/设计表示，再在表示空间检索；
- 关键测点缺失时直接拒绝，不让算法替业务决定可比性。

共同观测距离的一种形式为

$$
d^2(\mathbf x,\mathbf z)
=\frac d{|O(\mathbf x,\mathbf z)|}
\sum_{j\in O(\mathbf x,\mathbf z)}(x_j-z_j)^2,
$$

其中 $O$ 是双方共同有效的维度。乘 $d/|O|$ 让不同重叠数的量级大致可比，但缺失机制若非随机，仍可能产生偏差。

```python
def overlap_normalized_distance(query, reference, minimum_overlap=2):
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    observed = np.isfinite(query) & np.isfinite(reference)
    overlap = int(observed.sum())
    if overlap < minimum_overlap:
        return np.inf
    return float(np.sqrt(query.size / overlap * np.sum((query[observed] - reference[observed]) ** 2)))

missing_query = np.array([0.0, np.nan, 2.0, 1.0])
missing_reference = np.array([1.0, 9.0, 2.0, np.nan])
assert np.isclose(overlap_normalized_distance(missing_query, missing_reference), np.sqrt(2.0))
assert np.isinf(overlap_normalized_distance(np.array([np.nan, 1.0]), np.array([0.0, np.nan])))
```

<MlQuiz kind="设计判断" question="标准化后的关键温升特征缺失，直接填 0 的隐含语义是什么？" :options="['把未知当成训练均值附近', '把样本自动拒绝', '删除该维对所有样本的影响', '保证缺失完全随机']" :answer="0" explanation="Z-score 的 0 对应训练均值；若不加缺失指示或拒绝规则，未知值会伪装成典型正常值。" />

## 10. 维数灾难：高维中的“局部”可能覆盖全域

在 $d$ 维单位超立方体中，若轴对齐小立方体要覆盖总体积比例 $p$，每维边长比例为

$$
\ell(p,d)=p^{1/d}.
$$

当 $p=0.01$：二维只需每维跨 10%，十维要跨约 63%，一百维要跨约 95.5%。有限样本下，为收集少量邻居也可能横跨几乎整个运行域，局部常数假设失去意义。

高维还有**距离集中**：独立无关维度不断累加后，最近、典型与最远样本的相对差距缩小，近邻排序更容易被噪声扰动。

<DistanceConcentrationTunnelLab />

### 10.1 固定种子模拟距离集中

```python
concentration_rng = np.random.default_rng(1810)
concentration_points = concentration_rng.uniform(0.0, 1.0, size=(5000, 128))
concentration_ratios = []
concentration_contrasts = []
for dimension in (2, 8, 32, 128):
    distances = np.linalg.norm(concentration_points[:, :dimension] - 0.5, axis=1)
    nearest = np.min(distances)
    median = np.median(distances)
    farthest = np.max(distances)
    concentration_ratios.append(nearest / median)
    concentration_contrasts.append((farthest - nearest) / nearest)

assert concentration_ratios[-1] > concentration_ratios[0]
assert concentration_contrasts[-1] < concentration_contrasts[0]
print("最近/中位：", np.round(concentration_ratios, 3))
print("相对反差：", np.round(concentration_contrasts, 3))
```

```python
dimensions_for_volume = np.array([2, 10, 100], dtype=np.float64)
one_percent_edge = 0.01 ** (1.0 / dimensions_for_volume)
assert np.allclose(one_percent_edge[0], 0.1)
assert one_percent_edge[-1] > 0.95
print("覆盖 1% 体积所需每维边长：", np.round(one_percent_edge, 4))
```

解决方案不是笼统地“多收数据”：可以删除无关特征、利用领域不变性、用 PCA/监督表示降低有效维数、学习度量，或改用带结构假设的模型。若数据实际落在低维流形，环境维数大并不必然等于有效维数大。

<MlQuiz kind="结果预测" question="要在 100 维单位超立方体覆盖 1% 体积，每维边长大约需要达到全范围的多少？" :options="['95.5%', '1%', '10%', '0.01%']" :answer="0" explanation="边长为 0.01^(1/100)≈0.955；高维中少量体积也需要跨过几乎完整的每维范围。" />

### 10.2 无关维度可以改写最近邻身份

```python
irrelevant_rng = np.random.default_rng(1811)
base_train = scaled_measurements[train_rows]
base_validation = scaled_measurements[validation_rows]
_, _, base_neighbor, _ = knn_classify(
    base_train, root_cause[train_rows], base_validation, k=1
)
noise_train = irrelevant_rng.normal(size=(base_train.shape[0], 30))
noise_validation = irrelevant_rng.normal(size=(base_validation.shape[0], 30))
augmented_train = np.column_stack([base_train, noise_train])
augmented_validation = np.column_stack([base_validation, noise_validation])
_, _, augmented_neighbor, _ = knn_classify(
    augmented_train, root_cause[train_rows], augmented_validation, k=1
)
neighbor_identity_retention = np.mean(base_neighbor[:, 0] == augmented_neighbor[:, 0])
assert neighbor_identity_retention < 0.50
print("加入 30 个无关维后最近邻身份保留率：", round(float(neighbor_identity_retention), 3))
```

这些噪声维都已经是单位尺度，标准化无法删除它们。标准化修复“单位不公平”，特征选择解决“方向无关”，两者不是同一件事。

## 11. 一致性结论：只保留使用条件和量级直觉

最近邻并非只能靠经验。经典结论给出两层直觉：

1. 在独立同分布、合适度量等技术条件下，样本量趋于无穷时，1-NN 的错误率受 Bayes 风险的常数倍控制；二分类下经典上界可写成 $R_{1NN}\le2R^*(1-R^*)$；
2. 若 $n\to\infty$ 的同时让 $k\to\infty$ 且 $k/n\to0$，邻域样本数增长以降低方差，但邻域所占总体比例趋于 0 以保持局部，kNN 可趋向 Bayes 最优风险。

第二个条件解释了看似冲突的要求：

- $k\to\infty$：邻域频率有足够样本稳定下来；
- $k/n\to0$：邻域半径仍能收缩到查询点，而不是变成全局平均。

它们是渐近结论，不承诺有限样本、高维、错度量、组相关、标签噪声或分布偏移下表现良好。工业判断仍依赖验证和运行域审计。

<MlQuiz kind="推导补全" question="kNN 一致性直觉为什么同时要求 k→∞ 且 k/n→0？" :options="['前者稳定局部频率，后者让邻域占比和半径收缩', '前者让训练误差恒为零，后者删除距离', '两者只为降低查询时间', '为了保证 k 始终等于 1']" :answer="0" explanation="邻居数要增长以降低抽样波动，但相对样本库比例要下降，才能保持真正的局部估计。" />

## 12. 开放集拒绝：最近不等于足够近

kNN 在闭集分类中总能返回一个已知类别，即使查询远离全部训练数据。可以把第 $k$ 近邻半径

$$
r_k(\mathbf x)=\max_{i\in\mathcal N_k(\mathbf x)}d(\mathbf x,\mathbf x_i)
$$

作为覆盖指标，在独立验证集上冻结阈值 $\tau$：若 $r_k(\mathbf x)>\tau$，转人工或新故障流程。阈值不是概率保证；类别密度不同可使用分层阈值，并同时监控邻域纯度。

```python
validation_prediction, _, _, validation_neighbor_distances = knn_classify(
    scaled_measurements[train_rows], root_cause[train_rows],
    scaled_measurements[validation_rows], selected_k,
)
validation_radius = validation_neighbor_distances[:, -1]
radius_threshold = float(np.quantile(validation_radius, 0.95))

out_of_domain_raw = np.array([[13.0, 0.96, 42.0, 0.28, 0.20]])
out_of_domain_scaled = (out_of_domain_raw - training_mean) / training_scale
_, _, _, out_of_domain_distances = knn_classify(
    scaled_measurements[train_rows], root_cause[train_rows],
    out_of_domain_scaled, selected_k,
)
out_of_domain_radius = float(out_of_domain_distances[0, -1])
test_radius = test_neighbor_distances[:, -1]
test_acceptance_rate = float(np.mean(test_radius <= radius_threshold))

assert out_of_domain_radius > radius_threshold
assert test_acceptance_rate >= 0.70
print("半径阈值 / OOD 半径 / 测试接受率：", round(radius_threshold, 3), round(out_of_domain_radius, 3), round(test_acceptance_rate, 3))
```

若拒绝样本之后再用人工标签增量更新案例库，必须防止反馈回路：易确认样本先被写回会改变局部类别密度，困难样本长期缺失则让模型越来越相信已有模式。

<MlQuiz kind="工程判断" question="一个查询的最近邻仍然比验证集中 99% 的第 k 邻居都远，最稳妥的动作是什么？" :options="['按开放集协议拒绝并转复核', '仍强制输出最近邻标签且标记 100% 置信', '把距离清零', '临时把测试样本加入训练集']" :answer="0" explanation="“最近”只表示相对排序，不表示绝对相似；超出冻结覆盖阈值时应拒绝，而不是强制闭集分类。" />

## 数值验证

### 13.1 最小验收矩阵

一个可靠的 kNN 实现至少应通过：

1. 手算小样本的距离、邻居顺序、票数与回归均值；
2. 训练行置换后，非距离平局样本的预测不变；
3. 单位正比例变化经训练折标准化后，邻居排名不变；
4. 查询等于训练样本时，反距离权重只使用全部零距离案例；
5. `argpartition` 与完整排序得到相同前 $k$ 集合；
6. 同泵窗口不跨训练、验证、测试；
7. 缺失重叠不足、全零尺度、NaN 和越界 $k$ 被明确拒绝；
8. 开放集金丝雀的第 $k$ 半径超过冻结阈值。

```python
partition_rng = np.random.default_rng(1813)
partition_values = partition_rng.normal(size=1000)
partition_k = 17
partition_set = set(np.argpartition(partition_values, partition_k - 1)[:partition_k])
sorted_set = set(np.argsort(partition_values)[:partition_k])
assert partition_set == sorted_set

permutation_indices = partition_rng.permutation(len(manual_cases))
permuted_prediction, _, _, _ = knn_classify(
    manual_cases[permutation_indices], manual_labels[permutation_indices], manual_query, 3
)
assert permuted_prediction.tolist() == manual_prediction.tolist()
```

### 13.2 案例误差切片

```python
test_confusion = np.zeros((3, 3), dtype=int)
for truth, prediction in zip(root_cause[test_rows], test_prediction):
    test_confusion[truth, prediction] += 1
test_recall = np.diag(test_confusion) / np.maximum(test_confusion.sum(axis=1), 1)

test_hours_prediction = knn_regress(
    scaled_measurements[train_rows], maintenance_hours[train_rows],
    scaled_measurements[test_rows], selected_k, distance_power=1,
)
test_hours_mae = float(np.mean(np.abs(test_hours_prediction - maintenance_hours[test_rows])))
assert test_confusion.sum() == test_rows.sum()
assert np.all((test_recall >= 0.0) & (test_recall <= 1.0))
assert np.isfinite(test_hours_mae)
print("泵外测试混淆矩阵：\n", test_confusion)
print("每类召回：", np.round(test_recall, 3), "停机小时 MAE：", round(test_hours_mae, 3))
```

停机时长受组织流程影响，误差应按班次、备件可用性和泵型切片。分类也要报告复合故障、短窗口、关键传感器缺失和低覆盖半径样本，而不是只给一个平均数。

<MlQuiz kind="测试设计" question="下面哪组测试最能覆盖 kNN 的核心实现风险？" :options="['手算邻居、单位不变性、输入置换、零距离、分组切分和 OOD 半径', '只看训练准确率', '只确认函数能返回整数', '只把 k 从 1 调到 3']" :answer="0" explanation="kNN 的主要风险来自距离语义、排序边界、重复实体泄漏和覆盖范围，必须用性质测试而不是单一分数覆盖。" />

## 复杂度与工程边界

### 14.1 朴素精确查询

对 $n$ 个 $d$ 维案例和一个查询：

- 计算全部距离：$O(nd)$ 时间；
- 完整排序：$O(n\log n)$；
- `argpartition` 平均可避免完整排序，只需定位前 $k$；
- 维护大小为 $k$ 的堆：约 $O(n\log k)$；
- 案例库存储：$O(nd)$，还要加标签、元数据和索引副本。

训练不是 $O(0)$ 的真实系统动作：需要清洗、分组切分、预处理拟合、超参数验证、索引构建、版本化和回放测试。只是核心统计模型没有梯度优化。

```python
def brute_force_budget(reference_count, feature_count, query_count, bytes_per_value=4):
    distance_components = reference_count * feature_count * query_count
    reference_bytes = reference_count * feature_count * bytes_per_value
    distance_bytes = reference_count * query_count * bytes_per_value
    return distance_components, reference_bytes, distance_bytes

budget = brute_force_budget(2_000_000, 64, 128)
assert budget[0] == 16_384_000_000
assert np.isclose(budget[1] / 1024**3, 0.476837158203125)
assert np.isclose(budget[2] / 1024**3, 0.95367431640625)
print("距离分量 / 案例库 GiB / 距离矩阵 GiB：", budget[0], round(budget[1]/1024**3, 3), round(budget[2]/1024**3, 3))
```

距离矩阵可以按查询和案例分块，避免一次物化 $Q\times n$；但分块不会减少总算术量。

### 14.2 Voronoi、k-d tree 与近似近邻

- **Voronoi 图**：适合解释低维 1-NN；高维结构规模迅速增长，order-$k$ 更不实用；
- **k-d tree / ball tree**：低到中维可能大量剪枝；高维距离集中时容易退化到访问多数节点；
- **近似近邻（ANN）**：以邻居召回换取延迟和内存，必须在任务指标上验证近似误差；
- **PCA/随机投影/学习表示**：降低搜索维数，但改变了邻居语义，需要与 $k$ 联合验证；
- **原型删减**：压缩案例库时可能优先删除少数类或边界样本，应按类和运行域审计。

ANN 上线至少记录 `recall@k`、端到端分类/回归变化、P50/P95/P99 延迟、索引内存、构建时间、增量写入可见性和删除一致性。不能用“向量库返回了结果”代替算法验收。

<MlQuiz kind="容量估算" question="朴素 kNN 对一个查询计算全部距离的主时间复杂度是什么？" :options="['O(nd)', 'O(k)', 'O(log d)', 'O(1)']" :answer="0" explanation="需要扫描 n 个 d 维案例；选择前 k 的额外成本取决于完整排序、分区或堆实现。" />

### 14.3 何时不该使用 kNN

- 有效维数高且没有可信表示；
- 查询必须严格定时，而案例库巨大且近似误差不可接受；
- 敏感训练记录不能在预测服务中保留；
- 目标需要远距离外推，局部均值只能停留在历史范围；
- 类别随时间产生新语义，而案例库版本和删除机制不可控；
- 简单物理规则或线性模型已经达到相同性能且更易维护。

<div class="ml-stop"><strong>停下来推一推 4</strong>案例库从 10 万条扩到 200 万条，维度从 16 增到 64。估算朴素查询算术量增长倍数；再给出一种减少 n 和一种减少有效 d 的方案，并说明各自可能损伤哪些少数案例。</div>

## 15. 生产失败模式与监控

### 距离语义失效

- 传感器换量程或单位，邻居排名突变；
- 新增 30 个“都已标准化”的无关特征，距离反差下降；
- 泵型、转速或介质不同却未进入距离或分层案例库；
- 把缺失填成均值，未知工况伪装成典型样本。

### 案例库污染

- 同一泵重复窗口跨集合；
- 检修结论被后续复核修改，但索引未重建；
- 重复写入一条工单，相当于无意增加其票重；
- 自动预测被写回成“标签”，形成自我确认循环；
- 删除请求只删元数据，没有从向量索引移除。

### 最低监控面板

1. 案例总量、每类/泵型/工况覆盖和标签成熟度；
2. 查询第 1 与第 $k$ 距离、距离反差、邻域纯度和拒绝率；
3. 邻居来自同设备、同批次、同时间段的比例；
4. `UNK`、缺失、量程越界和标准化后极值比例；
5. ANN `recall@k`、索引新鲜度、P95/P99 延迟和内存；
6. 标签成熟后的分层召回、转派成本、停机时长 MAE 和校准；
7. 模型版本、案例库快照、预处理统计量和索引构建哈希。

<MlQuiz kind="故障定位" question="线上类别比例未变，但第 1/第 k 距离比持续升高、拒绝率下降。优先检查什么？" :options="['特征尺度、案例重复写入和邻域密度是否改变', '只把 k 固定为 1', '删除所有拒绝规则', '把票数解释成校准概率']" :answer="0" explanation="距离结构变化可能来自单位漂移或案例密度污染；类别比例稳定不能证明邻域仍有相同语义。" />

## 16. 验收清单

### 数据与表示

- [ ] 部署目标对应的实体、时间或 campaign 分组切分已冻结。
- [ ] 标准化、插补、降维和度量学习只在训练折拟合。
- [ ] 单位、量程、采样位置、泵型和工况版本可追溯。
- [ ] 重复案例、冲突标签、复合故障和删除请求有确定协议。

### 算法

- [ ] 距离函数的输入形状、有限值、权重和 $p$ 范围被验证。
- [ ] 距离平局、票数平局和零距离集合有确定行为。
- [ ] 分类、回归、等权和距离加权分别通过手算。
- [ ] $k$、距离、特征和拒绝阈值在独立验证流程中联合选择。
- [ ] 票数频率没有未经校准就作为故障概率发布。

### 系统与运行域

- [ ] 精确或近似索引的 `recall@k` 与任务指标同时验收。
- [ ] 峰值内存、批量查询、P99 延迟和索引构建时间实测。
- [ ] 第 $k$ 半径、距离反差、邻域纯度和拒绝率进入监控。
- [ ] 案例新增、修订、删除和回滚在元数据与索引间原子一致。
- [ ] 开放集样本可以拒绝并进入独立人工闭环。

## 17. 练习与最小实现

### 练习 1：手算分类、回归与平票

给定二维案例

$$
(0,0,A),\;(1,0,A),\;(0,2,B),\;(2,2,B),
$$

查询 $(0.8,0.6)$：

1. 手算 L1 与 L2 距离及完整排序；
2. 给出 $k=1,2,3,4$ 的分类结果；
3. 为 $k=2$ 设计确定性平票规则；
4. 若目标改成 $2,4,8,10$，计算等权与反距离回归；
5. 交换训练行顺序，验证所有非平局结果不变。

### 练习 2：从零实现工业 kNN

实现支持 `(Q,d)` 批量查询的分类与回归器：

- Minkowski $p=1,2,\infty$ 和非负特征权重；
- 等权与反距离权重；
- 零距离、距离平局、票数平局和无效 $k$；
- 分块计算，不能一次物化超预算距离矩阵；
- 返回邻居 id、距离、票重和拒绝原因；
- 与完整排序、手算和有限数据压力样例对照。

### 练习 3：无泄漏模型选择

在案例数据上完成按 `pump_id` 的五折外层评估和三折内层选择：

1. 内层联合选择标准/稳健尺度、L1/L2、$k$ 和距离权重；
2. 只用内层训练折拟合预处理；
3. 外层报告分类宏召回、转派成本、回归 MAE 和拒绝率；
4. 与错误的随机按行切分对照同泵最近邻率；
5. 给每个外层折提供置信区间或成对差异。

### 练习 4：维数灾难与索引预算

- 将 0、10、30、100 个标准化无关维加入数据；
- 记录邻居身份保留率、距离反差和验证指标；
- 比较穷举、树索引和一个 ANN 配置的 `recall@k` 与延迟；
- 分别用 PCA 和领域特征选择降维；
- 解释为什么 PCA 保方差不保证保类别近邻。

### 练习 5：框架对照

下面的 scikit-learn 片段仅作可选接口对照，仓库验证环境未为此新增依赖。重点核对 pipeline 是否在折内拟合、`weights='distance'` 的零距离行为、算法选择与返回邻居顺序。

```python
# 不执行：需要可选的 scikit-learn 环境。
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

model = Pipeline([
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=selected_k, weights="distance", p=2)),
])
model.fit(measurements[train_rows], root_cause[train_rows])
library_prediction = model.predict(measurements[test_rows])
assert library_prediction.shape == (test_rows.sum(),)
```

```python
# TODO：实现可分块、可拒绝的工业 kNN，并完成以下验收。
#
# 1. 输入案例库、查询、特征合同、k、p、权重和最大内存预算；
# 2. 返回预测、稳定排序的邻居 id/距离/票重、半径与拒绝原因；
# 3. 零距离、票数平局、全缺失、重叠不足和非法 k 都有确定行为；
# 4. 分块结果与完整距离矩阵在 float64 下等价；
# 5. 分组嵌套 CV 冻结预处理、k、距离和拒绝阈值；
# 6. 对 ANN 索引报告 recall@k、任务差异、P99 延迟与索引一致性。
```

## 18. 本章结论

1. kNN 用查询附近的标签频率或目标均值做局部常数估计，模型复杂度随案例库增长。
2. Query 的“邻近”完全由表示、距离、尺度和权重定义；单位选择不能偷偷成为业务重要性。
3. 分类票数是局部经验频率，不自动等于校准概率；回归均值也不能可靠外推到训练域之外。
4. 1-NN 在唯一坐标上通常训练误差为 0，对应 Voronoi 分区，但这不是泛化证据。
5. 增大 $k$ 会扩大邻域、降低局部噪声方差并增加平滑偏差；应使用分组验证选择。
6. 距离加权必须单独处理零距离集合，距离与票数平局必须与输入顺序无关。
7. 标准化只解决尺度，不会删除已标准化的无关维；类别、缺失、波形和文本需要匹配的距离合同。
8. 同一设备的重复窗口跨集合会把评估变成身份检索，kNN 对这种泄漏尤其敏感。
9. 高维下局部覆盖需要跨越大部分坐标范围，最近与典型距离趋于集中，邻居排序变脆弱。
10. 一致性直觉要求 $k\to\infty$ 同时 $k/n\to0$：样本数增长但邻域比例收缩。
11. 最近邻始终存在，却可能绝对很远；开放集任务应基于冻结覆盖指标允许拒绝。
12. 朴素查询为 $O(nd)$，树索引在高维可能退化，ANN 必须用 `recall@k` 和端到端指标共同验收。
13. 案例库新增、修订、删除、版本和索引新鲜度都是模型语义，不是外围存储细节。
14. 手算、单位不变性、输入置换、零距离、分组切分、距离集中和 OOD 金丝雀共同构成实现门禁。

## 来源与许可

kNN 分类的局部类别分布、Mahalanobis 距离、1-NN Voronoi、$k$ 的平滑效果、维数灾难、查询与存储成本、近似近邻和开放集识别，参考 Kevin P. Murphy 的 *Probabilistic Machine Learning: An Introduction* 源 PDF 第 577–581 页（§16.1–16.1.4）；距离学习的知识边界交叉参考源 PDF 第 582–583 页（§16.2–16.2.1）。本章不复用其中标注为改编或第三方许可的图形。

最近邻算法、L1/L2 距离、$O(nd)$ 查询、距离平票、特征尺度归纳偏置、Cover–Hart 大样本结论及模型/超参数区分，参考 CMU 10-301/10-601 Lecture 5 *k-Nearest Neighbors + Model Selection* 源 PDF 第 5–20、67–72 页。分类平票、距离加权/等权回归和三折验证例题交叉参考 CMU HW3 Recitation Solutions 源 PDF 第 5–9 页；本章没有复制其题图或解答表述。

kNN 分类/回归、1-NN 与 Bayes 风险关系、$k\to\infty$ 且 $k/n\to0$ 的一致性条件、穷举查询、Voronoi、k-d tree、精确与近似搜索边界，交叉参考 Jonathan Richard Shewchuk 的 CS 189/289A 讲义 Lecture 24–25 源 PDF 第 156–161 页。讲义只用于核对定理条件、复杂度和索引边界，不复用课堂图形、案例或措辞。

离心泵 30 台、150 条重复窗口、三类教学根因、停机时长生成规则、三项网页交互、全部 NumPy 代码、题目与 SVG 图形均为本站独立创作。教学数据不代表真实泵型、传感器量程、振动标准、保护阈值、检维修结论或停机计划；模型不得直接用于停泵、修改保护、备件采购或安全决策。scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

<MlLessonProgress
  lesson-id="18"
  next-href="/courses/machine-learning/19-decision-trees"
  next-label="第 19 章：决策树"
/>
