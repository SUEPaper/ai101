---
course: machine-learning
edition: industrial
title: 决策树
description: 以注塑连接器壳体质量分流为贯穿案例，从分段常数模型推导分类不纯度、回归平方误差、连续阈值扫描、递归生长与代价复杂度剪枝，并用批次隔离、缺失默认方向、概率平滑和规则回放建立工业验收门禁。
outline: [2, 3]
---

<MlLessonHero
  number="19"
  title="决策树"
  question="一件注塑壳体为什么被送去返工？如果模型只需依次核对几道压力、时间和含水阈值，怎样证明每一道切分确实降低了不确定性，又怎样阻止深树把偶然噪声刻成生产规则？"
  duration="建议 660–780 分钟"
  :outcomes="[
    '把决策树写成轴对齐分区上的分段常数函数，推导 Gini、熵、信息增益、回归叶均值和平方误差下降',
    '从零实现稳定的连续阈值扫描、分类树与回归树，解释贪心生长、平票协议、类别字段和缺失默认方向',
    '推导代价复杂度与弱链接剪枝，用批次分组验证、概率平滑、路径回放和容量预算验收工业树模型'
  ]"
/>

第 18 章把预测交给查询点周围的历史案例，本章改用一组层级规则压缩训练数据。树不要求欧氏空间，也不需要标准化；它通过一连串集合成员测试，把输入空间切成若干叶区域，再在每个区域内输出常数。

“能打印成规则”不等于“天然可靠”。训练样本的轻微扰动可能改变根节点，缺失字段可能把产品送往另一条路径，高基数类别可能制造虚假的不纯度下降，叶内频率也不是自动校准的概率。工业级实现必须同时管理统计目标、确定性协议和线上路由语义。

## 贯穿案例：注塑连接器壳体质量分流

一家电子连接器工厂在注塑机末端采集每模过程摘要，并由尺寸、外观和重量复检给出“合格”“欠注”或“飞边”标签。质量工程师希望模型完成两项辅助任务：

1. 在首件与巡检阶段建议放行、调机复核或隔离路线；
2. 估计壳体关键尺寸的绝对偏差，用于安排量测资源。

树只辅助分流，不直接修改注塑压力、锁模力、模温或安全联锁。任何新材料、模具换版、传感器失效和训练运行域外产品都必须进入明确的复核流程。

### 数据合同与决策代价

| 字段 | 工艺含义 | 主要边界 |
|---|---|---|
| `peak_pressure` | 型腔峰值压力，MPa | 传感器位置、量程与零点版本必须固定 |
| `fill_time` | 充填阶段持续时间，s | 控制器阶段定义变化会造成语义漂移 |
| `mold_temperature` | 模具表面温度，°C | 测点与红外发射率不可混用 |
| `resin_moisture` | 树脂含水率，% | 离线抽检与在线估计的来源要区分 |
| `clamp_reserve` | 锁模力余量，% | 负值、饱和和设备换算错误必须拒绝 |
| `cavity_id` | 模穴编号 | 高基数编号可能记住模穴而不是工艺机理 |
| `batch_id` | 材料批次与连续生产 campaign | 同一批次的重复模次不得跨数据集 |
| `quality_class` | 复检主导缺陷 | 复合缺陷、待判与返修后标签不能静默覆盖 |
| `dimension_deviation` | 关键尺寸绝对偏差，mm | 量具分辨率与温度补偿版本必须记录 |

漏放飞边可能影响装配和爬电距离，误报会增加隔离与复检成本；欠注与飞边也对应完全不同的调机方向。因此要报告每类召回、代价矩阵下的动作风险和拒绝率，而不是只给总体准确率。

### 1. 先按材料批次切分，再生长任何节点

下面构造 42 个批次、每批 5 个重复模次，共 210 条教学记录。批次级偏移刻意大于单模噪声，用来模拟原料、环境与设备状态的共同影响。

```python
import numpy as np

rng = np.random.default_rng(20260819)
batch_count = 42
shots_per_batch = 5
record_count = batch_count * shots_per_batch
batch_id = np.repeat(np.arange(batch_count), shots_per_batch)
batch_mode = np.arange(batch_count) % 3
quality_class = batch_mode[batch_id].copy()
class_names = np.array(["合格", "欠注", "飞边"])
feature_names = np.array([
    "peak_pressure", "fill_time", "mold_temperature",
    "resin_moisture", "clamp_reserve",
])

class_centers = np.array([
    [72.0, 1.05, 80.0, 0.060, 12.0],
    [62.0, 1.27, 76.0, 0.070, 14.0],
    [82.0, 0.98, 88.0, 0.100, 6.0],
])
batch_scale = np.array([2.2, 0.045, 1.8, 0.009, 1.4])
shot_scale = np.array([1.1, 0.025, 0.8, 0.004, 0.7])
batch_offset = rng.normal(0.0, batch_scale, size=(batch_count, 5))
process_features = (
    class_centers[quality_class]
    + batch_offset[batch_id]
    + rng.normal(0.0, shot_scale, size=(record_count, 5))
)
process_features[:, 1] = np.clip(process_features[:, 1], 0.75, 1.55)
process_features[:, 3] = np.clip(process_features[:, 3], 0.015, 0.160)
process_features[:, 4] = np.clip(process_features[:, 4], 0.0, 20.0)

# 约 8% 的教学标签噪声模拟复合缺陷、复检分歧与录入错误。
label_noise = rng.random(record_count) < 0.08
quality_class[label_noise] = (quality_class[label_noise] + rng.integers(1, 3, label_noise.sum())) % 3
cavity_id = np.array([f"C{value + 1:02d}" for value in batch_id % 12])
dimension_deviation = (
    0.018
    + 0.0030 * np.abs(process_features[:, 0] - 72.0)
    + 0.0040 * np.abs(process_features[:, 2] - 80.0)
    + 0.45 * np.maximum(process_features[:, 3] - 0.060, 0.0)
    + rng.normal(0.0, 0.006, record_count)
)
dimension_deviation = np.clip(dimension_deviation, 0.0, None)

train_batches, validation_batches, test_batches = [], [], []
for mode in range(3):
    candidates = np.where(batch_mode == mode)[0]
    ordered = rng.permutation(candidates)
    train_batches.extend(ordered[:8])
    validation_batches.extend(ordered[8:11])
    test_batches.extend(ordered[11:])
train_batches = np.array(train_batches)
validation_batches = np.array(validation_batches)
test_batches = np.array(test_batches)
train_rows = np.isin(batch_id, train_batches)
validation_rows = np.isin(batch_id, validation_batches)
test_rows = np.isin(batch_id, test_batches)

assert process_features.shape == (210, 5)
assert (train_rows.sum(), validation_rows.sum(), test_rows.sum()) == (120, 45, 45)
assert set(train_batches).isdisjoint(validation_batches)
assert set(train_batches).isdisjoint(test_batches)
assert set(validation_batches).isdisjoint(test_batches)
print("训练/验证/测试模次：", train_rows.sum(), validation_rows.sum(), test_rows.sum())
print("训练/验证/测试批次：", len(train_batches), len(validation_batches), len(test_batches))
print("总体教学标签分布：", np.bincount(quality_class, minlength=3))
```

同批次五个模次共享材料、烘料、环境和设备状态。若随机按行切分，模型可能把某个批次的阈值模式带到测试集，评估的是批次记忆而不是对新批次的迁移。

<MlQuiz kind="数据合同" question="目标是把模型部署到后续未见材料批次，五个连续模次应怎样分配？" :options="['每个模次独立随机切分', '以 batch_id 为组整体进入同一集合', '只把缺陷件放入测试集', '训练完成后再删除 batch_id']" :answer="1" explanation="批次是重复测量与共同偏移的来源，必须按真实部署边界隔离；训练后删除标识不能撤销已经发生的数据泄漏。" />

## 2. 决策树是分区上的分段常数函数

二叉树内部节点选择特征 $j$ 和阈值 $s$，把当前区域 $R$ 分成

$$
R_L(j,s)=\{\mathbf x\in R:x_j\le s\},\qquad
R_R(j,s)=\{\mathbf x\in R:x_j>s\}.
$$

从根到叶的一条路径是多个集合条件的交集。若路径依次检查“充填时间不超过 1.18 s”“峰值压力不超过 78 MPa”“含水率不超过 0.09%”，对应叶区域就是一个轴对齐长方体：

$$
R_m=\{\mathbf x:x_{\text{fill}}\le1.18,
x_{\text{pressure}}\le78,
x_{\text{moisture}}\le0.09\}.
$$

分类树在叶 $R_m$ 输出多数类或类别频率；回归树输出常数 $c_m$：

$$
f(\mathbf x)=\sum_{m=1}^{M}c_m\,\mathbf 1[\mathbf x\in R_m].
$$

树的非线性来自区域组合，而不是每个叶内的复杂函数。轴对齐切分能自然表达阈值与交互，却可能需要许多台阶才能逼近斜边界或平滑曲线。

<MlQuiz kind="结果预测" question="一棵标准二叉树依次对单个特征做阈值测试，它在二维空间形成什么边界？" :options="['任意圆弧', '由水平和垂直线段组成的轴对齐分区', '必然是一条直线', '只形成 Voronoi 多边形']" :answer="1" explanation="单变量阈值把当前矩形区域沿坐标轴切开；多层组合得到轴对齐矩形单元。" />

## 3. 分类切分：先定义节点的不纯度

节点 $R$ 中类别 $k$ 的经验比例为

$$
\hat p_k(R)=\frac{1}{n_R}\sum_{i:\mathbf x_i\in R}\mathbf 1[y_i=k].
$$

三种常见不纯度为

$$
I_{\text{error}}(R)=1-\max_k\hat p_k,
$$

$$
I_{\text{Gini}}(R)=1-\sum_{k=1}^{K}\hat p_k^2,
$$

$$
I_{\text{entropy}}(R)=-\sum_{k=1}^{K}\hat p_k\log_2\hat p_k,
\qquad 0\log 0\equiv0.
$$

Gini 可解释为：先按节点分布随机抽一个真实类别，再独立按同一分布随机给出类别，两者不同的概率。熵是平均信息量；纯节点的两者都为 0。误分类率只看最大类别，对少数类组成变化不敏感，常不适合作为生长阶段的切分准则。

### 3.1 加权子节点与不纯度下降

候选切分产生 $R_L,R_R$ 后，必须按样本量加权。记父节点有 $n$ 个样本，左右子节点分别有 $n_L,n_R$ 个样本：

$$
I_{\text{after}}=\frac{n_L}{n}I(R_L)+\frac{n_R}{n}I(R_R).
$$

$$
\Delta I=I(R)-I_{\text{after}}.
$$

对熵而言，$\Delta I$ 就是标签与二叉分支之间的经验互信息。熵与 Gini 都是类别概率的凹函数，所以把父分布写成子分布的加权平均后，Jensen 不等式给出 $\Delta I\ge0$；空子节点或两个子节点类别分布完全相同时取等号。

```python
def impurity_from_counts(counts, criterion="gini"):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts / total
    if criterion == "gini":
        return float(1.0 - probabilities @ probabilities)
    if criterion == "entropy":
        positive = probabilities > 0
        return float(-np.sum(probabilities[positive] * np.log2(probabilities[positive])))
    if criterion == "error":
        return float(1.0 - probabilities.max())
    raise ValueError("criterion 必须是 gini、entropy 或 error")

def label_impurity(labels, criterion="gini", class_count=3):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=class_count)
    return impurity_from_counts(counts, criterion)

manual_parent = np.array([0] * 6 + [1] * 4)
manual_left = np.array([0] * 4)
manual_right = np.array([0] * 2 + [1] * 4)
parent_gini = label_impurity(manual_parent)
after_gini = (
    len(manual_left) * label_impurity(manual_left)
    + len(manual_right) * label_impurity(manual_right)
) / len(manual_parent)
manual_gain = parent_gini - after_gini
assert np.allclose([parent_gini, after_gini, manual_gain], [0.48, 4/15, 16/75])
assert label_impurity(np.array([2, 2, 2])) == 0.0
print("父/切分后/Gini 下降：", round(parent_gini, 4), round(after_gini, 4), round(manual_gain, 4))
```

<TreeSplitFoundryLab />

<div class="ml-stop"><strong>停下来推一推 1</strong>父节点有 6 个合格、4 个欠注。候选切分得到左侧 4 个合格，右侧 2 个合格、4 个欠注。分别手算误分类率、Gini 和熵的父节点值、加权子节点值与下降量；解释为什么不能把左右不纯度直接相加。</div>

<MlQuiz kind="推导补全" question="为什么分类切分后的左右不纯度必须按子节点样本数加权？" :options="['为了让阈值一定取整数', '它们共同表示从父节点随机抽一个样本时所遇到的期望不纯度', '为了让深树训练误差升高', '因为左右节点必须一样大']" :answer="1" explanation="子节点大小就是随机样本落入该分支的经验概率；不加权会让很小的纯节点与大节点拥有相同影响。" />

## 4. 连续特征：只扫描相邻不同值的中点

对节点中的某个连续特征排序为

$$
x_{(1)j}\le x_{(2)j}\le\cdots\le x_{(n)j}.
$$

只需在 $x_{(i)j}<x_{(i+1)j}$ 时检查

$$
s_i=\frac{x_{(i)j}+x_{(i+1)j}}{2}.
$$

同一相邻区间内移动阈值不会改变左右样本集合，因此不纯度完全相同。扫描时维护左右类别累计计数，每移动一个样本只更新两个计数桶，而不必为每个阈值重新切片统计。

### 4.1 稳定的最佳切分实现

下面的实现采用稳定排序，要求左右至少有 `min_leaf` 个样本；完全同分时保留先出现的较小特征索引与较小阈值，使结果不依赖哈希或并行归约顺序。

```python
def best_classification_split(X, y, min_leaf=1, criterion="gini"):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or y.shape != (len(X),):
        raise ValueError("X 必须是二维数组，y 必须与 X 行数一致")
    if not np.isfinite(X).all():
        raise ValueError("基础切分器不接受 NaN/Inf；缺失路由必须显式处理")
    if min_leaf < 1 or 2 * min_leaf > len(y):
        return None
    class_count = int(y.max()) + 1
    parent_counts = np.bincount(y, minlength=class_count).astype(np.float64)
    parent_impurity = impurity_from_counts(parent_counts, criterion)
    best = None

    for feature_index in range(X.shape[1]):
        order = np.argsort(X[:, feature_index], kind="mergesort")
        values = X[order, feature_index]
        labels = y[order]
        left_counts = np.zeros(class_count, dtype=np.float64)
        right_counts = parent_counts.copy()
        for position in range(len(y) - 1):
            label = labels[position]
            left_counts[label] += 1.0
            right_counts[label] -= 1.0
            left_size = position + 1
            right_size = len(y) - left_size
            if values[position] == values[position + 1]:
                continue
            if left_size < min_leaf or right_size < min_leaf:
                continue
            after = (
                left_size * impurity_from_counts(left_counts, criterion)
                + right_size * impurity_from_counts(right_counts, criterion)
            ) / len(y)
            gain = parent_impurity - after
            threshold = float((values[position] + values[position + 1]) / 2.0)
            candidate = (float(gain), feature_index, threshold, float(after))
            if best is None or gain > best[0] + 1e-15:
                best = candidate
    return best

root_split = best_classification_split(
    process_features[train_rows], quality_class[train_rows], min_leaf=8, criterion="gini"
)
assert root_split is not None and root_split[0] > 0
print("根节点 gain/特征/阈值：", round(root_split[0], 4), feature_names[root_split[1]], round(root_split[2], 4))
```

数值实现还要防止中点溢出。对极大同号浮点数，可用 `a + (b - a) / 2`；若输入允许接近浮点上界，还应在特征合同处拒绝不合理量级。阈值序列与平票协议必须版本化，因为它们会改变整棵树。

<MlQuiz kind="代码阅读" question="为什么连续特征只需在相邻不同排序值之间检查一个阈值？" :options="['中点总能让左右样本数相等', '同一相邻区间内任意阈值产生完全相同的左右样本集合', '中点能消除标签噪声', '其他阈值会产生斜边界']" :answer="1" explanation="树的目标只由分到左右两侧的样本决定；样本集合不变时，任何区间内阈值的目标都相同。" />

## 5. 递归生长一棵分类树

节点的生长流程是：

1. 统计当前标签、默认预测与叶概率；
2. 检查纯度、最大深度、最小样本数等停止条件；
3. 扫描全部特征与阈值，选择当前不纯度下降最大的切分；
4. 若下降不足则保留为叶，否则递归生长左右子树。

这是贪心算法。根节点只优化第一刀，不会为了后面两刀的组合收益主动接受较差的当前切分。搜索所有树结构会产生组合爆炸，实际 CART 以可计算性换取全局最优保证。

```python
def build_classification_tree(X, y, max_depth=4, min_leaf=5, min_gain=1e-12, depth=0):
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=3)
    node = {
        "leaf": True,
        "depth": depth,
        "n": len(y),
        "counts": counts,
        "prediction": int(np.flatnonzero(counts == counts.max())[0]),
        "probabilities": counts / len(y),
        "impurity": label_impurity(y, "gini", 3),
    }
    if depth >= max_depth or len(y) < 2 * min_leaf or node["impurity"] <= 1e-15:
        return node
    split = best_classification_split(X, y, min_leaf=min_leaf, criterion="gini")
    if split is None or split[0] < min_gain:
        return node
    gain, feature_index, threshold, _ = split
    left_mask = X[:, feature_index] <= threshold
    node.update({
        "leaf": False,
        "gain": gain,
        "feature": feature_index,
        "threshold": threshold,
        "left": build_classification_tree(
            X[left_mask], y[left_mask], max_depth, min_leaf, min_gain, depth + 1
        ),
        "right": build_classification_tree(
            X[~left_mask], y[~left_mask], max_depth, min_leaf, min_gain, depth + 1
        ),
    })
    return node

def classification_leaf(tree, row):
    node = tree
    path = []
    while not node["leaf"]:
        go_left = row[node["feature"]] <= node["threshold"]
        path.append((node["feature"], node["threshold"], bool(go_left)))
        node = node["left"] if go_left else node["right"]
    return node, path

def predict_classification_tree(tree, X):
    return np.array([classification_leaf(tree, row)[0]["prediction"] for row in X], dtype=int)

teaching_tree = build_classification_tree(
    process_features[train_rows], quality_class[train_rows], max_depth=4, min_leaf=6
)
teaching_prediction = predict_classification_tree(teaching_tree, process_features[validation_rows])
assert teaching_prediction.shape == (validation_rows.sum(),)
assert teaching_tree["depth"] == 0
print("深度 4 教学树验证准确率：", round(float(np.mean(teaching_prediction == quality_class[validation_rows])), 3))
```

`np.flatnonzero(...)[0]` 明确规定类别票数平局时选择编码最小的类别。若业务要求“平局一律复核”，就不应把它藏在类别顺序里，而应让预测接口返回 `TIE` 状态。

<MlQuiz kind="工程判断" question="为什么贪心树不保证找到训练误差最低且叶子最少的全局最优结构？" :options="['因为 Gini 可能为负', '每个节点只优化当前切分，没有联合搜索后续子树组合', '因为树不能处理连续特征', '因为叶节点不存标签']" :answer="1" explanation="局部最佳第一刀可能阻断另一条需要先做弱切分、后做强切分的结构；全局树搜索通常不可承受。" />

<TreeRouteManifoldLab />

## 6. 树深不是唯一复杂度旋钮

常见预剪枝参数包括：

- `max_depth`：限制最长路径与推理延迟；
- `min_samples_leaf`：保证每个叶有最低证据量；
- `min_samples_split`：不足样本时不尝试切分；
- `min_impurity_decrease`：当前收益不足时停止；
- `max_leaf_nodes`：直接限制区域数量；
- 类别权重或样本权重：改变训练目标中的业务重要性。

这些参数相互作用。相同深度下，`min_samples_leaf=1` 与 30 的统计行为完全不同；只报告“深度 5”不足以复现实验。

### 6.1 用批次验证冻结整棵树

```python
classification_candidates = [
    {"max_depth": 2, "min_leaf": 12},
    {"max_depth": 3, "min_leaf": 10},
    {"max_depth": 4, "min_leaf": 6},
    {"max_depth": 6, "min_leaf": 3},
    {"max_depth": 10, "min_leaf": 1},
]
candidate_trees = []
validation_scores = []
for configuration in classification_candidates:
    candidate_tree = build_classification_tree(
        process_features[train_rows], quality_class[train_rows], **configuration
    )
    prediction = predict_classification_tree(candidate_tree, process_features[validation_rows])
    candidate_trees.append(candidate_tree)
    validation_scores.append(float(np.mean(prediction == quality_class[validation_rows])))

selected_candidate = int(np.argmax(validation_scores))
selected_configuration = classification_candidates[selected_candidate]
selected_tree = candidate_trees[selected_candidate]
test_prediction = predict_classification_tree(selected_tree, process_features[test_rows])
test_accuracy = float(np.mean(test_prediction == quality_class[test_rows]))
assert 0.0 <= test_accuracy <= 1.0
assert selected_configuration in classification_candidates
print("验证准确率路径：", np.round(validation_scores, 3))
print("冻结配置 / 批次外测试准确率：", selected_configuration, round(test_accuracy, 3))
```

测试集只在配置冻结后使用一次。若发现测试结果不理想，应回到新的开发周期，而不是继续用测试标签微调深度或叶大小。

<div class="ml-stop"><strong>停下来推一推 2</strong>一棵深度 3、`min_samples_leaf=20` 的树与深度 3、`min_samples_leaf=1` 的树，最大叶数上界相同吗？每个叶的概率稳定性相同吗？分别说明结构容量和叶内估计方差。</div>

<MlQuiz kind="故障诊断" question="训练准确率随深度从 2 到 10 单调上升，批次隔离验证准确率在深度 4 后下降，最直接的解释是什么？" :options="['深树的训练偏差更高', '更细分区开始追随训练批次和标签噪声，方差上升', 'Gini 不能用于多分类', '特征必须先标准化']" :answer="1" explanation="继续切分通常不增加训练不纯度，却会减少叶内样本并放大样本扰动；验证下降是过拟合证据。" />

## 7. 单棵树为何高方差

若两个候选根切分的 gain 很接近，删除少量样本、修正一个标签或换一个批次就可能让赢家互换。根节点变化后，左右子树看到的数据都不同，整棵结构会级联重建。

下面按训练批次做 bootstrap，记录每次根节点使用的特征。抽样单位仍是批次，而不是把重复模次当作独立样本。

```python
stability_rng = np.random.default_rng(1919)
train_batch_values = np.unique(batch_id[train_rows])
root_feature_counts = np.zeros(process_features.shape[1], dtype=int)
root_thresholds_by_feature = [[] for _ in range(process_features.shape[1])]
for _ in range(120):
    sampled_batches = stability_rng.choice(train_batch_values, size=len(train_batch_values), replace=True)
    sampled_indices = np.concatenate([np.flatnonzero(batch_id == value) for value in sampled_batches])
    split = best_classification_split(
        process_features[sampled_indices], quality_class[sampled_indices], min_leaf=8
    )
    if split is not None:
        root_feature_counts[split[1]] += 1
        root_thresholds_by_feature[split[1]].append(split[2])

root_feature_share = root_feature_counts / root_feature_counts.sum()
dominant_root_feature = int(np.argmax(root_feature_counts))
dominant_threshold_quantiles = np.quantile(
    root_thresholds_by_feature[dominant_root_feature], [.05, .5, .95]
)
assert np.isclose(root_feature_share.sum(), 1.0)
print("根特征 bootstrap 占比：", dict(zip(feature_names, np.round(root_feature_share, 3))))
print(
    "主导根特征及其阈值 5%/50%/95%：",
    feature_names[dominant_root_feature], np.round(dominant_threshold_quantiles, 3),
)
```

稳定性不是要求每棵树完全相同，而是判断一条被当作作业标准的规则是否对抽样、标签修订和测量误差过于敏感。第 20 章会利用多棵树平均来降低这种方差，但会牺牲单棵树的直接可读性。

<MlQuiz kind="结果预测" question="两个根切分的 gain 几乎相同，重新抽样后根特征频繁互换，这说明什么？" :options="['模型一定存在代码错误', '根规则证据脆弱，单棵树结构稳定性不足', '训练误差必为 100%', '树已经完成概率校准']" :answer="1" explanation="近似平局使小扰动足以改变根节点，随后整棵子树都可能重构；应报告稳定性并避免把单一路径过度解释。" />

## 8. 回归树：叶均值来自平方损失最小化

回归叶 $R$ 输出常数 $c$，平方误差为

$$
J_R(c)=\sum_{i:\mathbf x_i\in R}(y_i-c)^2.
$$

求导并令零：

$$
\frac{dJ_R}{dc}=-2\sum_{i\in R}(y_i-c)=0
\quad\Longrightarrow\quad
c^\star=\bar y_R.
$$

进一步利用平方和分解，任意常数都满足

$$
\sum_{i\in R}(y_i-c)^2
=\sum_{i\in R}(y_i-\bar y_R)^2+n_R(c-\bar y_R)^2,
$$

所以均值是唯一最优常数（节点非空时）。若损失改成绝对误差，最优叶常数是中位数；Huber、Poisson 或分位数目标也会改变叶统计量和切分准则。

```python
leaf_targets = np.array([0.020, 0.031, 0.028, 0.047, 0.034])
leaf_mean = float(leaf_targets.mean())
leaf_sse = float(np.sum((leaf_targets - leaf_mean) ** 2))
alternative_sse = float(np.sum((leaf_targets - (leaf_mean + .01)) ** 2))
assert alternative_sse > leaf_sse
assert np.isclose(alternative_sse - leaf_sse, len(leaf_targets) * .01 ** 2)
print("叶均值 / 最小 SSE：", round(leaf_mean, 5), round(leaf_sse, 7))
```

<MlQuiz kind="推导补全" question="平方损失回归树的叶节点为什么输出训练目标均值？" :options="['均值让叶样本数最少', '均值是常数预测下平方误差的唯一最小点', '均值不受离群点影响', '均值保证所有残差同号']" :answer="1" explanation="平方和可分解为围绕样本均值的不可约项加 n(c−均值)²，因此常数取均值时最小。" />

### 8.1 用累计和扫描回归阈值

排序后维护左侧的 $\sum y_i$、$\sum y_i^2$ 和样本数，就能用

$$
\operatorname{SSE}(R)=\sum_{i\in R}y_i^2-\frac{(\sum_{i\in R}y_i)^2}{n_R}
$$

在每个候选点 $O(1)$ 更新左右误差。

```python
def best_regression_split(X, y, min_leaf=1):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total_sum = float(y.sum())
    total_square = float(y @ y)
    parent_sse = max(0.0, total_square - total_sum * total_sum / len(y))
    best = None
    for feature_index in range(X.shape[1]):
        order = np.argsort(X[:, feature_index], kind="mergesort")
        values = X[order, feature_index]
        targets = y[order]
        left_sum = 0.0
        left_square = 0.0
        for position in range(len(y) - 1):
            value = float(targets[position])
            left_sum += value
            left_square += value * value
            left_size = position + 1
            right_size = len(y) - left_size
            if values[position] == values[position + 1]:
                continue
            if left_size < min_leaf or right_size < min_leaf:
                continue
            right_sum = total_sum - left_sum
            right_square = total_square - left_square
            left_sse = max(0.0, left_square - left_sum * left_sum / left_size)
            right_sse = max(0.0, right_square - right_sum * right_sum / right_size)
            reduction = parent_sse - left_sse - right_sse
            threshold = float((values[position] + values[position + 1]) / 2.0)
            candidate = (float(reduction), feature_index, threshold)
            if best is None or reduction > best[0] + 1e-15:
                best = candidate
    return best

regression_root = best_regression_split(
    process_features[train_rows], dimension_deviation[train_rows], min_leaf=8
)
assert regression_root is not None and regression_root[0] > 0
print("回归根节点 SSE 下降/特征/阈值：", round(regression_root[0], 5), feature_names[regression_root[1]], round(regression_root[2], 4))
```

### 8.2 从零生长回归树

```python
def build_regression_tree(X, y, max_depth=4, min_leaf=6, min_reduction=1e-12, depth=0):
    y = np.asarray(y, dtype=np.float64)
    mean = float(y.mean())
    sse = float(np.sum((y - mean) ** 2))
    node = {"leaf": True, "depth": depth, "n": len(y), "prediction": mean, "sse": sse}
    if depth >= max_depth or len(y) < 2 * min_leaf or sse <= 1e-18:
        return node
    split = best_regression_split(X, y, min_leaf=min_leaf)
    if split is None or split[0] < min_reduction:
        return node
    reduction, feature_index, threshold = split
    left_mask = X[:, feature_index] <= threshold
    node.update({
        "leaf": False, "reduction": reduction, "feature": feature_index, "threshold": threshold,
        "left": build_regression_tree(X[left_mask], y[left_mask], max_depth, min_leaf, min_reduction, depth + 1),
        "right": build_regression_tree(X[~left_mask], y[~left_mask], max_depth, min_leaf, min_reduction, depth + 1),
    })
    return node

def predict_regression_tree(tree, X):
    predictions = []
    for row in X:
        node = tree
        while not node["leaf"]:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        predictions.append(node["prediction"])
    return np.array(predictions)

regression_tree = build_regression_tree(
    process_features[train_rows], dimension_deviation[train_rows], max_depth=4, min_leaf=8
)
regression_prediction = predict_regression_tree(regression_tree, process_features[test_rows])
regression_mae = float(np.mean(np.abs(regression_prediction - dimension_deviation[test_rows])))
assert np.isfinite(regression_mae) and regression_mae < 0.08
print("批次外关键尺寸偏差 MAE：", round(regression_mae, 5), "mm")
```

回归树不会在叶内外推。若训练叶均值最大为 0.12 mm，输入再远也只能返回某个历史叶常数。需要连续外推或物理单调性时，应选择带结构约束的模型，或在叶内拟合受控局部模型。

## 9. 不用标准化，但仍要冻结表示语义

标准树只比较顺序和阈值。对单个连续特征施加严格单调变换，候选左右样本集合不变，只是阈值数值随坐标改变。因此把 MPa 线性换成 bar 不会像 kNN 那样改变邻近关系。

```python
raw_pressure = process_features[train_rows, [0]].reshape(-1, 1)
log_pressure = np.log(raw_pressure)
raw_pressure_split = best_classification_split(raw_pressure, quality_class[train_rows], min_leaf=8)
log_pressure_split = best_classification_split(log_pressure, quality_class[train_rows], min_leaf=8)
raw_partition = raw_pressure[:, 0] <= raw_pressure_split[2]
log_partition = log_pressure[:, 0] <= log_pressure_split[2]
assert np.array_equal(raw_partition, log_partition)
assert np.isclose(raw_pressure_split[0], log_pressure_split[0])
print("原值/log 压力 gain：", round(raw_pressure_split[0], 6), round(log_pressure_split[0], 6))
```

“无需标准化”不等于“无需数据治理”。字段单位变换后阈值必须同步迁移；若线上仍用旧阈值，整条路径会失效。非单调变换、裁剪、分箱、插补和类别编码也可能改变候选分区。

<MlQuiz kind="概念辨析" question="把压力从 MPa 线性换算为 bar，并同步换算所有树阈值，标准轴对齐树的路由会怎样？" :options="['通常保持不变', '一定变成斜边界', '必须重新标准化才能预测', '叶节点全部互换']" :answer="0" explanation="严格单调换算保持样本顺序；同步变换阈值后，每个 ≤ 判断的真假不变。" />

## 10. 类别特征：候选集合与高基数偏差

类别字段不能按任意整数编码后直接解释成连续顺序。一个有 $m$ 个取值的类别特征若允许任意二分，去除左右对称和空集合后有

$$
2^{m-1}-1
$$

个候选子集。高基数特征因此拥有更多“试彩票”机会，更容易在训练噪声上获得较大 gain。

```python
category_levels = np.array([2, 4, 8, 12, 20, 40])
binary_subset_candidates = 2 ** (category_levels - 1) - 1
assert binary_subset_candidates[0] == 1
assert binary_subset_candidates[-1] > 10 ** 11
print("类别数 -> 任意二分候选：", dict(zip(category_levels, binary_subset_candidates)))
```

工程选择包括：

- 低基数、确有业务语义时枚举子集；
- one-hot 后逐个测试“是否为某类别”，但只能一步隔离一个取值；
- 用训练折目标统计排序类别，再扫描前缀，但统计量必须折内计算并平滑；
- 把高基数标识保留为审计字段，不允许进入模型；
- 使用支持原生类别处理且有明确未知类别协议的算法。

`cavity_id` 很可能吸收模穴磨损，也可能只记住某批训练数据。若目标是跨模具泛化，直接使用模穴编号尤其危险。必须比较保留、分层和移除它的批次外结果。

<MlQuiz kind="故障诊断" question="一个 500 取值的设备序列号在训练中获得极高信息增益，最应先怀疑什么？" :options="['序列号一定是物理因果变量', '高基数带来大量候选并可能记忆设备或批次', '熵无法处理类别特征', '树只能使用二值标签']" :answer="1" explanation="候选越多，偶然找到高 gain 的机会越大；标识字段还可能直接编码实体身份和数据泄漏。" />

## 11. 缺失值必须有训练时冻结的路由

线上压力缺失时，`NaN <= threshold` 在许多语言中返回 `false`，但把所有缺失静默送右侧只是编程副作用，不是学习算法。常见协议有：

- 在每个节点比较“缺失全部向左”与“缺失全部向右”的训练目标，冻结较优默认方向；
- 学习替代切分：主特征缺失时使用与其分区最一致的第二特征；
- 训练折插补并增加缺失指示器；
- 缺失本身具有安全语义时直接转人工或专用分支。

### 11.1 学习默认方向

```python
def missing_default_direction(values, labels, threshold, criterion="gini"):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    missing = np.isnan(values)
    known_left = (~missing) & (values <= threshold)
    known_right = (~missing) & (values > threshold)
    candidates = []
    for direction in ("left", "right"):
        left = known_left | (missing if direction == "left" else False)
        right = known_right | (missing if direction == "right" else False)
        if not left.any() or not right.any():
            continue
        after = (
            left.sum() * label_impurity(labels[left], criterion, 3)
            + right.sum() * label_impurity(labels[right], criterion, 3)
        ) / len(labels)
        candidates.append((float(after), direction, int(left.sum()), int(right.sum())))
    if not candidates:
        raise ValueError("缺失路由无法形成两个非空子节点")
    return min(candidates, key=lambda item: (item[0], item[1]))

missing_rng = np.random.default_rng(1911)
root_feature = root_split[1]
root_threshold = root_split[2]
root_values_with_missing = process_features[train_rows, root_feature].copy()
root_values_with_missing[missing_rng.random(len(root_values_with_missing)) < .12] = np.nan
default_route = missing_default_direction(
    root_values_with_missing, quality_class[train_rows], root_threshold
)
assert default_route[2] + default_route[3] == train_rows.sum()
print("根节点缺失默认方向/切分后不纯度：", default_route[1], round(default_route[0], 4))
```

默认方向、替代特征顺序、插补统计量和缺失拒绝原因都必须进入模型制品。训练时没有缺失、线上突然缺失时，不能声称模型已经学习了合理路由。

<div class="ml-stop"><strong>停下来推一推 3</strong>压力传感器缺失恰好多发生在设备过量程时。若训练时用全局中位数插补并未保留缺失指示器，树会把这些样本送到哪里？说明这为何不是“随机缺失”问题，以及安全策略应如何覆盖。</div>

<MlQuiz kind="工程判断" question="线上主切分特征缺失时，最不可靠的做法是什么？" :options="['使用训练时冻结的默认方向', '使用训练时验证过的替代切分', '依赖 NaN 比较恰好返回 false，未记录即全部向右', '按安全协议拒绝高风险样本']" :answer="2" explanation="语言比较行为不是业务或统计协议；它必须被显式学习、验证、记录并跨实现复现。" />

## 12. 代价复杂度剪枝：先长树，再选择子树

预剪枝可能在一个当前 gain 很小的节点提前停止，却错过下一层很有价值的切分。后剪枝先长出较大的树，再在其嵌套子树中权衡经验误差与叶子数：

$$
R_\alpha(T)=R(T)+\alpha|\widetilde T|,
$$

其中 $R(T)$ 是所有叶内训练损失之和，$|\widetilde T|$ 是叶节点数。$\alpha=0$ 只看训练拟合；$\alpha$ 增大时，更小的子树逐步成为最优。

### 12.1 固定 alpha 的自底向上剪枝

对某内部节点 $t$，有两种选择：保留已剪好的左右子树，或把整个子树折叠成一个叶。比较两者的 $R_\alpha$ 即可做固定 $\alpha$ 的动态规划。

```python
def leaf_training_error(node):
    return int(node["n"] - np.max(node["counts"]))

def prune_for_alpha(node, alpha_value):
    if node["leaf"]:
        copied = dict(node)
        return copied, leaf_training_error(node) + alpha_value
    left, left_cost = prune_for_alpha(node["left"], alpha_value)
    right, right_cost = prune_for_alpha(node["right"], alpha_value)
    subtree_cost = left_cost + right_cost
    collapsed_cost = leaf_training_error(node) + alpha_value
    if collapsed_cost <= subtree_cost + 1e-12:
        collapsed = {key: node[key] for key in (
            "leaf", "depth", "n", "counts", "prediction", "probabilities", "impurity"
        )}
        collapsed["leaf"] = True
        return collapsed, collapsed_cost
    kept = dict(node)
    kept["left"], kept["right"] = left, right
    return kept, subtree_cost

def tree_leaf_count(node):
    return 1 if node["leaf"] else tree_leaf_count(node["left"]) + tree_leaf_count(node["right"])

def tree_max_depth(node):
    return node["depth"] if node["leaf"] else max(tree_max_depth(node["left"]), tree_max_depth(node["right"]))

large_tree = build_classification_tree(
    process_features[train_rows], quality_class[train_rows], max_depth=12, min_leaf=1, min_gain=0.0
)
alpha_candidates = np.array([0.0, .25, .5, 1.0, 2.0, 4.0, 8.0, 12.0])
pruned_trees, pruning_validation_scores = [], []
for alpha_value in alpha_candidates:
    pruned_tree, _ = prune_for_alpha(large_tree, float(alpha_value))
    pruned_trees.append(pruned_tree)
    prediction = predict_classification_tree(pruned_tree, process_features[validation_rows])
    pruning_validation_scores.append(float(np.mean(prediction == quality_class[validation_rows])))

pruning_choice = int(np.argmax(pruning_validation_scores))
pruned_tree = pruned_trees[pruning_choice]
assert all(
    tree_leaf_count(pruned_trees[index]) >= tree_leaf_count(pruned_trees[index + 1])
    for index in range(len(pruned_trees) - 1)
)
print("alpha / 叶数：", list(zip(alpha_candidates, [tree_leaf_count(tree) for tree in pruned_trees])))
print("验证选择 alpha/叶数：", alpha_candidates[pruning_choice], tree_leaf_count(pruned_tree))
```

### 12.2 弱链接临界值

若 $T_t$ 是以内部节点 $t$ 为根的当前子树，把它折叠为单叶所增加的训练误差为 $R(t)-R(T_t)$，减少的叶子数为 $|\widetilde T_t|-1$。每删除一个叶子的平均代价是

$$
g(t)=\frac{R(t)-R(T_t)}{|\widetilde T_t|-1}.
$$

$g(t)$ 最小的子树是当前最弱链接；从最小 $g(t)$ 开始依次剪除，可得到嵌套的 cost-complexity 路径。

```python
def subtree_error_and_leaves(node, candidates):
    if node["leaf"]:
        return leaf_training_error(node), 1
    left_error, left_leaves = subtree_error_and_leaves(node["left"], candidates)
    right_error, right_leaves = subtree_error_and_leaves(node["right"], candidates)
    subtree_error = left_error + right_error
    leaves = left_leaves + right_leaves
    effective_alpha = (leaf_training_error(node) - subtree_error) / (leaves - 1)
    candidates.append((float(effective_alpha), node["depth"], leaves))
    return subtree_error, leaves

weak_links = []
subtree_error_and_leaves(large_tree, weak_links)
weak_links.sort(key=lambda item: (item[0], -item[1]))
assert weak_links and weak_links[0][0] >= -1e-12
print("最小三个弱链接 (alpha, depth, leaves)：", np.round(weak_links[:3], 4))
```

同一临界值可能对应多段子树；实现必须规定同时剪除还是稳定排序逐个剪除。框架中的 `ccp_alpha` 若按总样本归一化，数值尺度会与这里的原始错误数不同，不能直接搬参数。

<CostComplexityPruningRailLab />

<MlQuiz kind="推导补全" question="子树折叠成一个叶时，弱链接值 g(t) 的分母为什么是叶数减 1？" :options="['因为根节点不计训练误差', '折叠后从 L 个叶变成 1 个叶，实际减少 L−1 个叶', '为了保证 g(t) 小于 1', '因为二叉树深度少 1']" :answer="1" explanation="g(t) 衡量每减少一个叶所付出的训练误差增量，折叠 L 叶子树只剩一个叶，因此减少 L−1。" />

## 13. 叶概率：频率、平滑与校准

分类叶最直接的概率估计是

$$
\hat p_k=\frac{n_k}{n_{\text{leaf}}}.
$$

小叶很容易给出 0 或 1。对称 Dirichlet/Laplace 平滑给出

$$
\widetilde p_k=\frac{n_k+\lambda}{n_{\text{leaf}}+K\lambda}.
$$

它能避免零概率，却不能自动解决分区选择偏差或跨批次校准。概率系统仍要在独立验证集上评估 NLL、Brier、可靠性图和动作阈值。

```python
def smoothed_leaf_probabilities(tree, X, smoothing=1.0):
    probabilities = []
    for row in X:
        leaf, _ = classification_leaf(tree, row)
        counts = leaf["counts"].astype(np.float64)
        probabilities.append((counts + smoothing) / (counts.sum() + smoothing * len(counts)))
    return np.vstack(probabilities)

test_probabilities = smoothed_leaf_probabilities(pruned_tree, process_features[test_rows], smoothing=1.0)
test_truth = quality_class[test_rows]
test_nll = float(-np.mean(np.log(test_probabilities[np.arange(len(test_truth)), test_truth])))
test_brier = float(np.mean(np.sum((test_probabilities - np.eye(3)[test_truth]) ** 2, axis=1)))
assert np.allclose(test_probabilities.sum(axis=1), 1.0)
assert np.all((test_probabilities > 0.0) & (test_probabilities < 1.0))
print("平滑叶概率测试 NLL / Brier：", round(test_nll, 4), round(test_brier, 4))
```

<MlQuiz kind="概念辨析" question="给叶频率加入 Laplace 平滑后，可以直接宣称概率已校准吗？" :options="['可以，非零概率等于校准', '不可以，平滑只收缩极端频率，仍需独立数据验证校准', '可以，只要树足够深', '不可以，因为树不能输出概率']" :answer="1" explanation="平滑处理小样本与零概率，但没有证明预测频率与真实长期频率一致；分区选择和分布偏移仍会影响校准。" />

## 14. 类别代价与样本权重

若漏放飞边的代价高，可以为训练样本赋权 $w_i$，用权重而不是样本数计算类别比例：

$$
\hat p_k^{(w)}(R)=
\frac{\sum_{i\in R}w_i\mathbf1[y_i=k]}{\sum_{i\in R}w_i}.
$$

切分后的左右权重也必须按总权重加权。类别权重改变的是生长目标，不等于已经选择了最优动作阈值；若叶输出要解释为原部署先验下的概率，还要处理权重引入的先验偏移。

```python
def weighted_gini(labels, weights, class_count=3):
    labels = np.asarray(labels, dtype=int)
    weights = np.asarray(weights, dtype=np.float64)
    if len(labels) != len(weights) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("权重必须非负、总和为正且与标签等长")
    counts = np.bincount(labels, weights=weights, minlength=class_count)
    probabilities = counts / counts.sum()
    return float(1.0 - probabilities @ probabilities)

unit_weights = np.ones(train_rows.sum())
cost_weights = np.where(quality_class[train_rows] == 2, 5.0, 1.0)
ordinary_root_gini = weighted_gini(quality_class[train_rows], unit_weights)
cost_root_gini = weighted_gini(quality_class[train_rows], cost_weights)
assert np.isclose(ordinary_root_gini, label_impurity(quality_class[train_rows]))
assert not np.isclose(ordinary_root_gini, cost_root_gini)
print("普通/飞边五倍权重根 Gini：", round(ordinary_root_gini, 4), round(cost_root_gini, 4))
```

## 数值验证

### 15.1 最小验收矩阵

一个可靠的树实现至少应验证：

- 纯节点的 Gini 与熵为 0，空计数的约定明确；
- 手算小样本的父不纯度、加权子不纯度和 gain 一致；
- 同一相邻区间内阈值移动不改变目标；
- 行置换后，非平局数据得到等价分区和预测；
- 分类票数、gain 和特征平局有确定行为；
- 回归叶均值与 SSE 累计公式等于直接计算；
- 固定 $\alpha$ 增大时叶数不增加；
- 缺失默认方向覆盖全部样本，训练与推理实现一致；
- 单调单位换算并同步阈值后路径完全相同；
- 未知类别、NaN、Inf、空节点和非法参数被显式拒绝。

```python
verification_rng = np.random.default_rng(1915)
permutation = verification_rng.permutation(train_rows.sum())
permuted_split = best_classification_split(
    process_features[train_rows][permutation], quality_class[train_rows][permutation], min_leaf=8
)
assert root_split[1] == permuted_split[1]
assert np.isclose(root_split[0], permuted_split[0])
assert np.isclose(root_split[2], permuted_split[2])

direct_sse = np.sum((dimension_deviation[train_rows] - dimension_deviation[train_rows].mean()) ** 2)
sum_sse = (
    dimension_deviation[train_rows] @ dimension_deviation[train_rows]
    - dimension_deviation[train_rows].sum() ** 2 / train_rows.sum()
)
assert np.isclose(direct_sse, sum_sse)
assert tree_max_depth(pruned_tree) <= tree_max_depth(large_tree)
print("置换、累计 SSE 与剪枝深度检查通过。")
```

### 15.2 误差切片与混淆矩阵

```python
pruned_test_prediction = predict_classification_tree(pruned_tree, process_features[test_rows])
test_confusion = np.zeros((3, 3), dtype=int)
for truth, prediction in zip(quality_class[test_rows], pruned_test_prediction):
    test_confusion[truth, prediction] += 1
test_recall = np.diag(test_confusion) / np.maximum(test_confusion.sum(axis=1), 1)
test_precision = np.diag(test_confusion) / np.maximum(test_confusion.sum(axis=0), 1)
assert test_confusion.sum() == test_rows.sum()
assert np.all((test_recall >= 0) & (test_recall <= 1))
print("剪枝树批次外混淆矩阵：\n", test_confusion)
print("每类召回 / 精确率：", np.round(test_recall, 3), np.round(test_precision, 3))
```

至少按模具版本、材料牌号、模穴、缺失状态、边界阈值附近和标签成熟度切片。总体准确率稳定时，某个模穴可能已经集中进入一条高风险叶。

<MlQuiz kind="测试设计" question="下面哪组回归测试最能覆盖决策树实现的核心风险？" :options="['手算 gain、行置换、平票、累计 SSE、缺失路由、剪枝单调性和单位路径回放', '只检查训练准确率', '只确认树能打印文本', '只把最大深度从 3 改成 4']" :answer="0" explanation="树的风险集中在切分算术、确定性、递归状态、缺失协议和剪枝路径；必须逐项建立可执行不变量。" />

### 15.3 阈值量化与跨语言回放

生产推理器可能把 float64 阈值导出为 float32、定点数或 PLC 可表示小数。边界样本会因量化改变分支，必须用金丝雀记录逐节点真假结果。

```python
root_feature_index = selected_tree.get("feature", 0)
root_threshold_float64 = float(selected_tree.get("threshold", 0.0))
root_threshold_float32 = float(np.float32(root_threshold_float64))
boundary_canaries = np.array([
    np.nextafter(root_threshold_float64, -np.inf),
    root_threshold_float64,
    np.nextafter(root_threshold_float64, np.inf),
])
route_float64 = boundary_canaries <= root_threshold_float64
route_float32 = boundary_canaries <= root_threshold_float32
route_difference = int(np.sum(route_float64 != route_float32))
assert route_difference in (0, 1, 2, 3)
print("根阈值 float64/float32/边界路由差异数：", root_threshold_float64, root_threshold_float32, route_difference)
```

不能用“最终类别恰好一样”掩盖中间路径差异。今天两个叶输出相同，明天叶概率、监控计数或下游动作不同，潜伏的阈值不一致就会暴露。

## 复杂度与工程边界

### 16.1 训练、预测与内存

若每个节点都从头排序，连续特征树训练会非常昂贵。典型分析为：

- 根节点预排序：约 $O(dn\log n)$；
- 利用排序索引或直方图复用后，扫描工作与样本参与的节点深度有关，约 $O(dnD)$；
- 朴素逐阈值重新切片可能退化到 $O(dn^2)$ 甚至更差；
- 单样本预测：$O(D)$ 次比较；
- 批量预测：$O(qD)$；
- 树结构存储：$O(M)$，$M$ 为节点数；
- 完整二叉树深度 $D$ 的叶数最多 $2^D$，但 `min_leaf` 又给出不超过 $\lfloor n/m\rfloor$ 的数据上界。

```python
training_records = 5_000_000
feature_count = 120
depth_budget = 12
histogram_bins = 256
float32_matrix_gib = training_records * feature_count * 4 / 2 ** 30
histogram_cells_per_node = feature_count * histogram_bins
batch_comparisons = 200_000 * depth_budget
max_leaves_by_depth = 2 ** depth_budget
max_leaves_by_minimum = training_records // 500
assert min(max_leaves_by_depth, max_leaves_by_minimum) == max_leaves_by_depth
print("训练矩阵 GiB：", round(float32_matrix_gib, 3))
print("每节点直方图单元 / 批量比较：", histogram_cells_per_node, batch_comparisons)
print("深度/最小叶样本上界：", max_leaves_by_depth, max_leaves_by_minimum)
```

大规模实现常采用量化直方图、并行特征扫描、稀疏感知、外存排序和叶优先生长。近似切分会改变阈值与树结构，必须把 bin 边界、缺失桶、确定性归约和线程数纳入复现合同。

### 16.2 何时不该使用单棵树

- 决策边界主要是平滑斜面，轴对齐台阶需要大量叶；
- 需要稳定、校准且低方差的概率，而可用叶样本很少；
- 规则必须满足严格单调性，但普通树没有结构约束；
- 训练样本轻微变化就不能接受规则重排；
- 类别字段极高基数、未知类别频繁且没有可信编码协议；
- 需要可靠外推连续目标；
- 单棵树性能不足，而又不能接受集成后的解释与延迟成本。

<div class="ml-stop"><strong>停下来推一推 4</strong>五百万行、120 个连续特征、最大深度 12 的训练任务中，估算 float32 设计矩阵内存、一次对 20 万样本批量推理的比较次数，以及深度给出的最大叶数。再说明为何这些上界仍不足以估算训练峰值内存。</div>

<MlQuiz kind="容量估算" question="深度为 D 的二叉树对单个样本预测，主要比较次数量级是什么？" :options="['O(D)', 'O(2^D) 必须访问全部叶', 'O(nD)', 'O(dn²)']" :answer="0" explanation="预测只沿一条根到叶路径前进；训练需要搜索许多候选，二者复杂度不能混为一谈。" />

## 17. 生产失败模式与监控

### 数据与标签失效

- 同一材料批次跨训练和测试，验证分数变成批次记忆；
- 复检后标签覆盖原标签，却没有保留成熟时间与版本；
- 模穴编号或订单号高基数泄漏实体身份；
- 缺失、饱和、量程变更被当成普通数值；
- 返工后的过程记录与返工前标签时间错位。

### 路由与模型制品失效

- float64 阈值导出为低精度后，边界金丝雀换路；
- 训练使用 `<=`，服务实现使用 `<`；
- NaN 默认方向、未知类别和替代特征未随模型发布；
- 类别编码字典更新，旧整数不再代表同一模穴；
- 剪枝后节点 id 重排，监控仍聚合旧叶 id；
- 文本规则展示经过四舍五入，与真实预测阈值不一致。

### 最低监控面板

- 每个内部节点的左右路由比例、缺失比例和边界邻近比例；
- 每个叶的流量、类别频率、平滑概率、标签成熟后误差；
- 深度分布、未知类别、拒绝原因和替代切分使用率；
- 根与浅层节点的 bootstrap 稳定性、阈值漂移和 gain；
- 按材料、模具、模穴、班次和版本切分的召回与动作成本；
- 模型哈希、树结构、类别字典、缺失协议、阈值精度与训练数据快照。

<MlQuiz kind="故障定位" question="线上总体类别比例稳定，但某浅层节点的缺失默认路由比例从 2% 升到 35%，应优先检查什么？" :options="['只增加最大深度', '传感器、字段映射、NaN 协议与该节点默认方向是否变化', '把所有叶概率设为 0.5', '删除节点监控']" :answer="1" explanation="浅层缺失路由会影响大批样本的后续路径；应先定位数据与协议变化，而不是用更深树掩盖。" />

## 18. 验收清单

### 数据与目标

- [ ] 实体、批次、时间和 campaign 切分与部署边界一致。
- [ ] 复合缺陷、待判、返工后标签和标签成熟时间有明确协议。
- [ ] 数值单位、量程、类别字典、未知类别和缺失语义已版本化。
- [ ] 标识字段与高基数类别经过泄漏和候选偏差审计。

### 算法

- [ ] Gini、熵、信息增益、SSE 与叶统计量通过手算。
- [ ] 连续阈值只扫描有效中点，累计计数/和与直接计算一致。
- [ ] 特征、阈值、类别票数和平票规则不依赖输入行顺序。
- [ ] 最大深度、最小叶、最小 gain、权重和停止条件被共同验证。
- [ ] 代价复杂度尺度、弱链接平票和验证选择流程可复现。
- [ ] 缺失默认方向、替代切分和未知类别在训练与推理一致。

### 系统与运行域

- [ ] 节点比较符、阈值精度、类别编码和路径日志跨语言回放一致。
- [ ] 峰值训练内存、节点数、批量延迟和最坏路径深度已实测。
- [ ] 浅层路由比例、叶流量、缺失率、边界样本和分层误差进入监控。
- [ ] 叶概率经过独立校准评价，未把训练频率直接当作置信保证。
- [ ] 新模具、材料换版、越界输入和高风险缺失可以拒绝并转人工。

## 19. 练习与最小实现

### 练习 1：手算三种分类准则

给定父节点三类计数 $(8,5,3)$，两个候选切分分别产生：

- A：$(7,1,0)$ 与 $(1,4,3)$；
- B：$(4,4,0)$ 与 $(4,1,3)$。

手算误分类率、Gini、熵及加权下降，解释三种准则是否选择同一切分，并写出所有平票处理。

### 练习 2：实现工业分类树

从零实现支持批量预测的树：

- 连续阈值累计扫描与稳定平票；
- Gini、熵和样本权重；
- 最大深度、最小叶、最小 gain；
- 冻结缺失默认方向和未知类别拒绝；
- 返回预测、叶概率、节点路径、拒绝原因和模型版本；
- 用手算、行置换、单位换算和跨语言金丝雀验收。

### 练习 3：回归树与不同叶损失

分别实现平方损失均值叶、绝对损失中位数叶和分位数叶；对含离群点的尺寸偏差数据比较切分、MAE、RMSE、Pinball loss 和叶预测变化。

### 练习 4：完整弱链接剪枝路径

从最大树开始，计算每个内部节点的 $g(t)$，处理临界值平票并生成嵌套子树序列。用批次分组嵌套验证选择 $\alpha$，证明验证和测试只在各自允许的层级使用。

### 练习 5：框架对照

下面的 scikit-learn 片段只作可选接口对照，仓库验证环境没有为此新增依赖。重点核对 `criterion`、`min_samples_leaf`、`ccp_alpha`、类别编码、缺失能力与框架版本。

```python
# 不执行：需要可选的 scikit-learn 环境。
from sklearn.tree import DecisionTreeClassifier

library_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=selected_configuration["max_depth"],
    min_samples_leaf=selected_configuration["min_leaf"],
    ccp_alpha=0.0,
    random_state=19,
)
library_tree.fit(process_features[train_rows], quality_class[train_rows])
library_prediction = library_tree.predict(process_features[test_rows])
assert library_prediction.shape == (test_rows.sum(),)
```

```python
# TODO：实现可审计、可剪枝、可拒绝的工业决策树，并完成以下验收。
#
# 1. 连续、低基数类别与缺失字段都有明确候选和路由合同；
# 2. 累计扫描与暴力枚举在小样本上得到相同最优分区；
# 3. 分类、回归、权重、平票、空节点和非法参数行为确定；
# 4. 生成完整 weakest-link 路径，并只用分组验证选择 alpha；
# 5. 导出模型能逐节点回放比较符、阈值精度、默认方向和类别字典；
# 6. 报告节点/叶流量、路径深度、校准、分层误差、延迟和峰值内存。
```

## 20. 本章结论

- 决策树把输入空间递归切成轴对齐区域，并在叶内输出类别分布或数值常数。
- Gini 与熵使用完整类别比例；误分类率只看多数类，切分分辨率较弱。
- 不纯度下降必须按子节点样本量或权重加权；熵下降就是经验信息增益。
- 连续特征只需扫描相邻不同排序值的中点，累计计数/和避免重复统计。
- 分类树输出多数类，平方损失回归树的最优叶常数是均值；损失改变时叶统计量也改变。
- 贪心生长只优化当前切分，不保证全局最优树；深树通常低偏差、高方差。
- 最大深度、最小叶、最小 gain 与权重共同定义容量，必须按部署实体分组验证。
- 单调单位变换不会改变候选分区，但字段语义、阈值和比较符仍须同步版本化。
- 高基数类别拥有大量候选，容易记忆实体或批次；类别编码不是无害预处理。
- 缺失默认方向必须在训练中显式学习或按安全协议拒绝，不能依赖 NaN 比较副作用。
- 代价复杂度剪枝在误差与叶数之间选择嵌套子树；$\alpha$ 仍由独立验证确定。
- 叶频率与 Laplace 平滑都不自动等于校准概率，仍需 NLL、Brier 与可靠性审计。
- 生产模型不仅是树结构，还包括类别字典、缺失路由、阈值精度、比较符和路径日志协议。
- 训练复杂度与排序/深度有关，预测只访问一条路径；近似直方图与并行归约也会改变结构复现性。
- 节点路由比例、叶覆盖、缺失率、边界金丝雀和标签成熟后的分层误差构成最低监控面。

## 来源与许可

决策树的递归生长、分类不纯度、熵/信息增益、连续特征中点扫描、累计计数和 $O(nd\,\text{depth})$ 训练复杂度，参考 Jonathan Richard Shewchuk 的 CS 189/289A 讲义源 PDF 第 82–86 页；回归叶均值与方差、早停、叶概率、验证剪枝、轴对齐边界和高方差边界参考源 PDF 第 87–90 页。本章不复用讲义中的课堂图形、案例或措辞。

CMU 10-301/10-601 Lecture 3 *Decision Trees* 源 PDF 第 18–33 页用于核对切分准则、递归训练、分类错误率的分辨率不足、熵与互信息；Lecture 4 *Decision Tree Properties* 源 PDF 第 3–46 页用于核对轴对齐边界、贪心非全局最优、停止条件、模型选择与优缺点。HW3 Recitation Solutions 源 PDF 第 1–4 页用于交叉核对连续二维分区、深度过拟合和验证集剪枝的知识粒度。本章没有复制其题图、数据、答案或表达。

注塑连接器壳体的 42 个教学批次、210 条重复模次、三类质量标签、关键尺寸偏差生成规则、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作。数值不代表真实材料、模具、注塑机、传感器、工艺窗口、质量标准或安全阈值；模型不得直接修改压力、锁模力、模温、安全联锁或自动放行高风险产品。scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

<MlLessonProgress
  lesson-id="19-decision-trees"
  next-href="/courses/machine-learning/20-random-forests"
  next-label="第 20 章：Bagging 与随机森林"
/>
