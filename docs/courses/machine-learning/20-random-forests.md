---
course: machine-learning
edition: industrial
title: Bagging 与随机森林
description: 以半导体等离子刻蚀腔体的批次异常分诊为贯穿案例，从 Bootstrap 权重推导 OOB、相关预测平均方差与随机特征去相关机制，并从零实现支持分组重采样、OOB 调参与解释审计的最小随机森林。
outline: [2, 3]
---

<MlLessonHero
  number="20"
  title="Bagging 与随机森林"
  question="一棵深树能把刻蚀过程切得很细，却会因换几片晶圆就改写根节点。把很多棵树平均为什么能稳定？如果它们总沿同一强传感器切分，树再多又为何无济于事？"
  duration="建议 720–840 分钟"
  :outcomes="[
    '把 Bootstrap 写成多项式权重，推导唯一入袋率、OOB 率、相关基学习器平均方差及其不可消除的相关误差地板',
    '区分 Bagging、随机子空间与随机森林，从零实现节点级特征抽样、分组 Bootstrap、概率聚合和逐样本 OOB 预测',
    '用分组 OOB 选择 mtry 与叶容量，审计树间相关、特征重要性、概率锐度、延迟、复现制品和生产漂移'
  ]"
/>

第 19 章得到了一棵可审计的决策树，但也暴露了树最难处理的性质：训练集只改变几条记录，第一刀就可能换特征，整棵树随之重排。Bagging 不试图把每棵树变稳定，而是让许多不稳定树在不同训练样本上生长，再聚合它们的预测。随机森林又在每个节点限制候选特征，主动削弱树之间的同步。

“树多”不是随机森林的充分条件。若抽样单位错了，OOB 会泄漏；若一个强特征和它的复制列统治所有树，平均会撞上相关误差地板；若用 OOB 反复选参后还把同一 OOB 分数当最终无偏成绩，也是在偷看验证信号。工业实现的对象不是一袋树，而是一整套抽样、随机性、聚合、评价和版本协议。

## 贯穿案例：等离子刻蚀腔体批次异常分诊

一家晶圆厂希望用每片晶圆的刻蚀过程摘要，辅助判断一个批次更像“正常”“腔壁聚合物累积”还是“质量流量控制器漂移”。模型服务于离线工程复核和量测排程，不直接修改 RF 功率、气体流量、腔压、端点条件或设备联锁。

同一晶圆批次中的四片晶圆共享配方、腔体状态和前序工艺。部署目标是诊断未来新批次，因此抽样与评价的最小独立单位是 `lot_id`，不是单片晶圆。

### 数据合同与动作边界

| 字段 | 工艺含义 | 主要边界 |
|---|---|---|
| `endpoint_slope` | 光学端点信号归一化斜率 | 光谱波段、滤波窗口和端点算法必须版本化 |
| `endpoint_slope_copy` | 同源管线的斜率副本 | 故意保留，用于暴露相关特征的解释陷阱 |
| `reflected_power_ratio` | 反射功率与前向功率比 | 匹配网络量程、采样阶段和饱和标记不可混用 |
| `pressure_cv` | 主刻蚀阶段腔压变异系数 | 平均窗口与异常点清洗会改变语义 |
| `wall_temp_delta` | 腔壁温度相对基准偏移 | 腔体位置、测点和基准配方必须冻结 |
| `clean_age` | 上次湿洗后的累计标准化片数 | 维护记录回填和重置延迟会造成时间穿越 |
| `mfc_offset` | 质量流量控制器零点偏移代理 | 只作诊断特征，不代表可直接调整的设定值 |
| `bias_arc_rate` | 偏置 RF 弧光事件率 | 事件定义、去抖和设备固件要随模型发布 |
| `lot_id` | 晶圆批次 | 同批晶圆不得跨训练、OOB、验证或测试边界 |
| `chamber_campaign` | 清洗到清洗之间的运行周期 | 更严格部署时应以 campaign 为最外层分组 |
| `fault_class` | 复核后的主导异常类别 | 待判、复合异常和标签成熟时间必须保留 |

错把质量流量漂移判成腔壁累积会把工程师引向不同的检查路线；漏报又会延迟复核。因此总体准确率只用于讲解，验收还要报告每类召回、拒绝率、复核成本和跨腔体 campaign 的稳定性。

### 1. 先按批次冻结训练、验证与测试

下面生成 60 个教学批次、每批 4 片晶圆。批次共同偏移明显大于片内噪声，强迫评价流程尊重部署实体。

```python
import numpy as np

rng = np.random.default_rng(20260820)
lot_count = 60
wafers_per_lot = 4
record_count = lot_count * wafers_per_lot
lot_id = np.repeat(np.arange(lot_count), wafers_per_lot)
lot_mode = np.arange(lot_count) % 3
fault_class = lot_mode[lot_id].copy()
class_names = np.array(["正常", "腔壁累积", "MFC 漂移"])
feature_names = np.array([
    "endpoint_slope", "endpoint_slope_copy", "reflected_power_ratio",
    "pressure_cv", "wall_temp_delta", "clean_age",
    "mfc_offset", "bias_arc_rate",
])

class_centers = np.array([
    [-0.20, -0.20, 0.08, 0.10, 0.00, -0.35, 0.00, 0.08],
    [ 0.75,  0.75, 0.58, 0.48, 0.62,  1.00, 0.12, 0.42],
    [ 0.35,  0.35, 0.26, 0.95, 0.12,  0.05, 1.18, 0.24],
])
lot_scale = np.array([0.22, 0.22, 0.12, 0.16, 0.18, 0.24, 0.20, 0.10])
wafer_scale = np.array([0.18, 0.05, 0.09, 0.12, 0.12, 0.15, 0.14, 0.08])
lot_offset = rng.normal(0.0, lot_scale, size=(lot_count, len(feature_names)))
process_features = (
    class_centers[fault_class]
    + lot_offset[lot_id]
    + rng.normal(0.0, wafer_scale, size=(record_count, len(feature_names)))
)
# 第二列来自第一列的同源管线，不是独立证据。
process_features[:, 1] = process_features[:, 0] + rng.normal(0.0, 0.045, record_count)
label_noise = rng.random(record_count) < 0.09
fault_class[label_noise] = (
    fault_class[label_noise] + rng.integers(1, 3, label_noise.sum())
) % 3

train_lots, validation_lots, test_lots = [], [], []
for mode in range(3):
    candidates = np.where(lot_mode == mode)[0]
    ordered = rng.permutation(candidates)
    train_lots.extend(ordered[:12])
    validation_lots.extend(ordered[12:16])
    test_lots.extend(ordered[16:20])
train_lots = np.array(train_lots)
validation_lots = np.array(validation_lots)
test_lots = np.array(test_lots)
train_rows = np.isin(lot_id, train_lots)
validation_rows = np.isin(lot_id, validation_lots)
test_rows = np.isin(lot_id, test_lots)

assert process_features.shape == (240, 8)
assert (train_rows.sum(), validation_rows.sum(), test_rows.sum()) == (144, 48, 48)
assert set(train_lots).isdisjoint(validation_lots)
assert set(train_lots).isdisjoint(test_lots)
assert set(validation_lots).isdisjoint(test_lots)
print("训练/验证/测试晶圆：", train_rows.sum(), validation_rows.sum(), test_rows.sum())
print("训练/验证/测试批次：", len(train_lots), len(validation_lots), len(test_lots))
print("三类教学标签：", np.bincount(fault_class, minlength=3))
```

若未来要迁移到从未见过的腔体 campaign，外层测试还要以 `chamber_campaign` 分组；批次级 OOB 只能回答“同类 campaign 中的新批次”问题，不能自动证明跨设备泛化。

<MlQuiz kind="数据合同" question="模型目标是诊断未来新批次，同批四片晶圆应怎样参与 Bootstrap？" :options="['四片各自独立抽样', '以 lot_id 为单位整组入袋或整组 OOB', '只抽异常晶圆', '先按行抽样，计算 OOB 时再删除 lot_id']" :answer="1" explanation="抽样单位必须匹配部署独立单位；按行抽样会让同批兄弟晶圆同时出现在树的训练与 OOB 评价中。" />

## 2. Bootstrap 不是“删掉 36.8% 再训练”

从 $n$ 个训练单位中独立、有放回抽取 $n$ 次。令 $W_i^{(b)}$ 表示单位 $i$ 在第 $b$ 个 Bootstrap 样本中出现的次数，则

$$
(W_1^{(b)},\ldots,W_n^{(b)})
\sim \operatorname{Multinomial}\left(n;\frac1n,\ldots,\frac1n\right).
$$

每棵树看到的总抽签数仍是 $n$，但有些单位权重为 0，有些为 2、3 甚至更高。固定单位 $i$ 一次都没有抽到的精确概率为

$$
p_{\mathrm{OOB}}(n)=P(W_i^{(b)}=0)
=\left(1-\frac1n\right)^n.
$$

它至少出现一次的概率为

$$
p_{\mathrm{unique}}(n)=1-\left(1-\frac1n\right)^n
\longrightarrow 1-e^{-1}\approx0.632.
$$

所以“63.2%”指不同训练单位的期望比例，不是抽样数组长度。有限 $n$ 时应使用精确式，而不是把 0.632 当硬约束。

```python
def bootstrap_indices(n, random_state):
    generator = np.random.default_rng(random_state)
    sampled = generator.integers(0, n, size=n)
    multiplicity = np.bincount(sampled, minlength=n)
    oob = np.flatnonzero(multiplicity == 0)
    return sampled, multiplicity, oob

sampled, multiplicity, oob = bootstrap_indices(40, 20)
exact_oob_probability = (1 - 1 / 40) ** 40
assert sampled.shape == (40,)
assert multiplicity.sum() == 40
assert np.intersect1d(np.unique(sampled), oob).size == 0
print("本轮唯一/OOB 数：", np.count_nonzero(multiplicity), len(oob))
print("n=40 的精确 OOB 概率：", round(exact_oob_probability, 4))
```

```python
simulation_rng = np.random.default_rng(2001)
simulation_count = 6000
oob_rates = []
for _ in range(simulation_count):
    draw = simulation_rng.integers(0, 40, size=40)
    oob_rates.append(np.mean(np.bincount(draw, minlength=40) == 0))
simulated_oob_rate = float(np.mean(oob_rates))
assert abs(simulated_oob_rate - exact_oob_probability) < 0.01
print("模拟/精确 OOB 比例：", round(simulated_oob_rate, 4), round(exact_oob_probability, 4))
```

<BootstrapOobCarouselLab />

<div class="ml-stop"><strong>停下来推一推 1</strong>对 $n=3$ 枚举全部 $3^3$ 个有放回抽样序列，分别统计唯一单位数为 1、2、3 的序列数。解释为什么“抽三次”并不等于“看见三个不同单位”。</div>

<MlQuiz kind="公式辨析" question="从 n 个单位有放回抽 n 次，某单位进入样本至少一次的概率是什么？" :options="['(1−1/n)^n', '1−(1−1/n)^n', '1/n', '必然为 1']" :answer="1" explanation="一次未中的概率为 1−1/n，n 次都未中是其 n 次方；至少一次进入是补事件。" />

### 2.1 行 Bootstrap 为什么会污染批次 OOB

若把 144 片训练晶圆当独立单位，某片虽然可能 OOB，但它同批的三片兄弟通常已经入袋。树可借共享批次偏移间接认出这片晶圆。下面量化这种暴露。

```python
X_train = process_features[train_rows]
y_train = fault_class[train_rows]
groups_train = lot_id[train_rows]
row_rng = np.random.default_rng(2020)
row_sample = row_rng.integers(0, len(X_train), size=len(X_train))
row_inbag = np.zeros(len(X_train), dtype=bool)
row_inbag[row_sample] = True
row_oob = np.flatnonzero(~row_inbag)
sampled_groups = set(groups_train[row_sample].tolist())
leaked_oob = np.array([groups_train[index] in sampled_groups for index in row_oob])
assert leaked_oob.size > 0
print("行级 OOB 片数：", len(row_oob))
print("其中批次兄弟已入袋的比例：", round(float(leaked_oob.mean()), 3))
```

分组 Bootstrap 从 36 个训练批次中有放回抽 36 次，每次把整批四片加入训练树。一个批次只可能处于“至少入袋一次”或“整批 OOB”两种状态。组大小不等时，这也会让大组对叶计数贡献更多；若业务希望每组等权，还需显式设计组权重，不能把两种目标混为一谈。

## 3. Bagging 聚合的是预测，不是树结构

给定 $B$ 个在不同 Bootstrap 样本上训练的基学习器 $\hat f_b$，回归 Bagging 输出

$$
\hat f_{\mathrm{bag}}(\mathbf x)
=\frac1B\sum_{b=1}^{B}\hat f_b(\mathbf x).
$$

二分类可多数投票，也可先平均类别概率再阈值。多分类概率平均为

$$
\bar p_k(\mathbf x)=\frac1B\sum_{b=1}^{B}\hat p_{bk}(\mathbf x),
\qquad
\hat y=\arg\max_k\bar p_k(\mathbf x).
$$

硬投票只保留每棵树的赢家；概率平均还保留叶内类别比例，但仍不保证校准。框架若把 `predict_proba` 与 `predict` 定义在不同聚合路径上，必须分别回放。

```python
tree_probabilities = np.array([
    [0.48, 0.47, 0.05],
    [0.48, 0.46, 0.06],
    [0.02, 0.96, 0.02],
])
hard_votes = np.argmax(tree_probabilities, axis=1)
majority_prediction = np.bincount(hard_votes, minlength=3).argmax()
mean_probability = tree_probabilities.mean(axis=0)
probability_prediction = mean_probability.argmax()
assert majority_prediction == 0
assert probability_prediction == 1
print("硬投票/概率平均类别：", majority_prediction, probability_prediction)
print("平均概率：", np.round(mean_probability, 3))
```

这不是谁永远更正确，而是两个不同的接口合同。类别权重、叶平滑和平票规则也会改变聚合结果，必须写进模型制品。

<MlQuiz kind="边界反例" question="三棵树的硬标签多数为正常，但少数树给异常类极高概率。概率平均的最终类别能否与硬投票不同？" :options="['不能，两者恒等', '能，因为硬投票丢弃了每棵树的概率幅度', '只有回归才可能', '只要树数为奇数就不能']" :answer="1" explanation="多数投票聚合 argmax，概率平均先聚合完整向量；两个运算不交换。" />

## 4. 平均为什么降方差，又为什么有地板

在固定输入 $\mathbf x$ 处，把第 $b$ 棵树相对其期望预测的随机偏差记为 $\varepsilon_b$。先作一个诊断性简化：每棵树方差都是 $\sigma^2$，任意两棵树相关系数都是 $\rho$。则

$$
\bar\varepsilon=\frac1B\sum_{b=1}^{B}\varepsilon_b.
$$

逐项展开：

$$
\begin{aligned}
\operatorname{Var}(\bar\varepsilon)
&=\frac1{B^2}\left[
\sum_{b=1}^{B}\operatorname{Var}(\varepsilon_b)
+\sum_{b\ne b'}\operatorname{Cov}(\varepsilon_b,\varepsilon_{b'})
\right]\\
&=\frac1{B^2}\left[B\sigma^2+B(B-1)\rho\sigma^2\right]\\
&=\sigma^2\left[\rho+\frac{1-\rho}{B}\right].
\end{aligned}
$$

第一项 $\rho\sigma^2$ 是所有树共同摆动的部分；第二项才会随 $B$ 消失。若 $\rho=0$，方差为 $\sigma^2/B$；若 $\rho=1$，所有树像复制件，方差始终为 $\sigma^2$。增加树数只消除有限森林的蒙特卡洛波动，不能消除共同偏差、共同数据缺口或共同传感器错误。

```python
def ensemble_variance_ratio(tree_count, correlation):
    if tree_count < 1 or not 0 <= correlation <= 1:
        raise ValueError("tree_count>=1 且 correlation 位于 [0,1]")
    return correlation + (1 - correlation) / tree_count

variance_rng = np.random.default_rng(2040)
replications, trees, rho = 30000, 50, 0.24
shared = variance_rng.normal(size=(replications, 1))
private = variance_rng.normal(size=(replications, trees))
errors = np.sqrt(rho) * shared + np.sqrt(1 - rho) * private
empirical_ratio = float(np.var(errors.mean(axis=1), ddof=0))
theoretical_ratio = ensemble_variance_ratio(trees, rho)
assert abs(empirical_ratio - theoretical_ratio) < 0.015
print("模拟/理论方差比：", round(empirical_ratio, 4), round(theoretical_ratio, 4))
```

<ForestCorrelationRackLab />

<div class="ml-stop"><strong>停下来推一推 2</strong>令 $B=100$。分别计算 $\rho=0.02$、$0.2$、$0.8$ 时的方差比，再计算把树数翻到 200 的收益。哪种情形应优先投入到降低相关性，而不是继续堆树？</div>

<MlQuiz kind="结果预测" question="B→∞ 且树间误差相关系数 rho=0.2 时，等方差模型中的集成方差会趋向哪里？" :options="['0', '0.2σ²', 'σ²/B', '负无穷']" :answer="1" explanation="独立部分随 1/B 消失，相关部分留下 rho·σ²。" />

### 4.1 为什么深树适合 Bagging

深决策树对样本扰动敏感，通常低训练偏差、高方差，正好提供“平均可以消掉”的随机波动。稳定的线性模型在 Bootstrap 样本间变化较小，Bagging 收益往往有限。反过来，若先把每棵树剪成同一个高偏差树桩，平均也不能恢复被所有树共同错过的交互。

这不等于叶越小越好。极小叶会产生极端概率、巨大节点数和较高推理成本；`max_depth`、`min_leaf`、类别权重和缺失路由仍需一起验证。

<MlQuiz kind="机制判断" question="Bagging 通常最适合哪类基学习器？" :options="['对训练样本扰动敏感的高方差学习器', '每次都输出同一常数的学习器', '没有预测接口的算法', '偏差极高且完全相同的树桩']" :answer="0" explanation="只有重采样能制造足够不同的预测时，平均才有明显的方差缩减空间。" />

## 5. 随机森林：每个节点重新抽候选特征

样本 Bagging 只改变行。如果 `endpoint_slope` 在所有 Bootstrap 样本中都很强，许多树仍会在根节点选择它，预测高度相关。随机森林训练第 $b$ 棵树时：

1. 对训练单位作 Bootstrap；
2. 在当前节点从 $d$ 个特征中无放回抽取 $m_{\mathrm{try}}$ 个；
3. 只在这组候选中寻找最优切分；
4. 每到一个新节点都重新抽候选集；
5. 生长到停止条件，再聚合所有树。

必须区分三个名称：

| 方法 | 行随机性 | 特征随机性 |
|---|---|---|
| 样本 Bagging | 每个学习器重采样行 | 每个节点看全部特征 |
| 随机子空间 | 可选 | 通常为每个学习器固定一组特征 |
| 随机森林 | 每棵树 Bootstrap 行 | 每个节点重新抽候选特征 |

若无放回抽 $m_{\mathrm{try}}$ 个候选，固定特征在某节点入围的概率是 $m_{\mathrm{try}}/d$。减小 $m_{\mathrm{try}}$ 往往降低树间相关，却也可能让关键特征错过浅层节点，降低单树强度。分类的 $\sqrt d$ 只是起点，不是定律。

<RandomSubspaceSwitchyardLab />

<MlQuiz kind="概念辨析" question="随机森林与经典随机子空间法最关键的区别是什么？" :options="['随机森林在每个节点重新抽候选特征，随机子空间通常为整棵学习器固定子集', '随机森林完全不抽样本', '随机子空间只能回归', '两者都要求所有节点使用同一特征']" :answer="0" explanation="节点级重抽让同一棵树的不同分支也能看到不同候选集，是随机森林的核心去相关机制。" />

### 5.1 强度—相关性不是单调竞赛

`mtry=d` 时退化为树 Bagging；`mtry=1` 时节点几乎不优化特征选择。中间值能否更好，取决于冗余特征、噪声维度、样本量、叶容量和损失。

复制强特征尤其危险。即使原强特征未入围，它的副本仍可能入围，使“强特征族”继续统治浅层节点；同时单列置换重要性会因另一个副本可替代而下降。工程上应同时查看树预测相关、浅层切分占比和相关特征组的重要性。

<div class="ml-stop"><strong>停下来推一推 3</strong>数据有 100 个特征，其中只有一个强特征。比较 $m_{\mathrm{try}}=1,10,100$ 时强特征在某节点入围的概率。再加入 9 个几乎相同的副本，说明为什么“缩小 mtry 就必然去相关”不再成立。</div>

## 6. 从零实现最小随机森林

下面只实现连续特征、多分类 Gini、批次 Bootstrap 和概率平均，用于暴露算法合同。生产库还需要缺失方向、类别字段、样本权重、直方图切分、并行归约、模型序列化和跨语言回放。

### 6.1 节点只扫描随机候选列

对每个候选特征排序后，切分只需检查相邻不同值之间的中点。左、右类别计数逐步更新，避免为每个阈值重新统计。

```python
def gini_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts / total
    return float(1.0 - probabilities @ probabilities)

def best_random_split(X, y, candidate_features, min_leaf, class_count=3):
    n = len(y)
    parent_counts = np.bincount(y, minlength=class_count).astype(float)
    parent_impurity = gini_from_counts(parent_counts)
    best = None
    tolerance = 1e-12
    for feature in np.sort(candidate_features):
        order = np.argsort(X[:, feature], kind="mergesort")
        values = X[order, feature]
        labels = y[order]
        left_counts = np.zeros(class_count, dtype=float)
        right_counts = parent_counts.copy()
        for split_position in range(1, n):
            label = labels[split_position - 1]
            left_counts[label] += 1
            right_counts[label] -= 1
            left_size, right_size = split_position, n - split_position
            if left_size < min_leaf or right_size < min_leaf:
                continue
            if values[split_position - 1] == values[split_position]:
                continue
            child_impurity = (
                left_size * gini_from_counts(left_counts)
                + right_size * gini_from_counts(right_counts)
            ) / n
            gain = parent_impurity - child_impurity
            threshold = (values[split_position - 1] + values[split_position]) / 2
            candidate = (float(gain), int(feature), float(threshold))
            if best is None or gain > best[0] + tolerance:
                best = candidate
            elif abs(gain - best[0]) <= tolerance and candidate[1:] < best[1:]:
                best = candidate
    return best

tiny_X = np.array([[0.0, 2.0], [0.2, 1.8], [0.8, 0.4], [1.0, 0.2]])
tiny_y = np.array([0, 0, 1, 1])
tiny_split = best_random_split(tiny_X, tiny_y, np.array([0, 1]), min_leaf=1, class_count=2)
assert tiny_split[0] == 0.5 and tiny_split[1] == 0
print("手算切分 gain/feature/threshold：", tiny_split)
```

稳定平票协议先比 gain，再比特征编号和阈值。否则线程调度、列顺序或排序实现会改变整棵树。

### 6.2 递归生长与叶概率

每个节点用自己的随机数流抽候选列。叶概率使用简单 Laplace 平滑，避免未见类别恰好得到 0；平滑不等于校准。

```python
def build_random_tree(X, y, generator, max_depth=7, min_leaf=4, mtry=3,
                      class_count=3, depth=0):
    counts = np.bincount(y, minlength=class_count).astype(float)
    leaf = {
        "leaf": True,
        "probability": (counts + 1.0) / (counts.sum() + class_count),
        "samples": int(len(y)),
    }
    if depth >= max_depth or len(y) < 2 * min_leaf or np.count_nonzero(counts) == 1:
        return leaf
    candidate_features = generator.choice(
        X.shape[1], size=min(mtry, X.shape[1]), replace=False
    )
    split = best_random_split(X, y, candidate_features, min_leaf, class_count)
    if split is None or split[0] <= 1e-12:
        return leaf
    gain, feature, threshold = split
    left_mask = X[:, feature] <= threshold
    if left_mask.sum() < min_leaf or (~left_mask).sum() < min_leaf:
        return leaf
    return {
        "leaf": False,
        "feature": feature,
        "threshold": threshold,
        "gain": gain,
        "samples": int(len(y)),
        "left": build_random_tree(
            X[left_mask], y[left_mask], generator, max_depth, min_leaf,
            mtry, class_count, depth + 1,
        ),
        "right": build_random_tree(
            X[~left_mask], y[~left_mask], generator, max_depth, min_leaf,
            mtry, class_count, depth + 1,
        ),
    }

def predict_tree_probability(tree, X):
    predictions = []
    for row in X:
        node = tree
        while not node["leaf"]:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        predictions.append(node["probability"])
    return np.asarray(predictions)

tree_rng = np.random.default_rng(2061)
teaching_tree = build_random_tree(X_train, y_train, tree_rng, max_depth=5, min_leaf=4, mtry=3)
teaching_probability = predict_tree_probability(teaching_tree, X_train[:7])
assert teaching_probability.shape == (7, 3)
assert np.allclose(teaching_probability.sum(axis=1), 1.0)
print("根节点是否为叶 / 前两行概率：", teaching_tree["leaf"], np.round(teaching_probability[:2], 3))
```

### 6.3 分组 Bootstrap 与 OOB 账本

每棵树保存它未见过哪些训练批次。随机种子由森林主种子和树编号确定；增加树数时，已有树不应改变。

```python
def train_group_forest(X, y, groups, tree_count, mtry, seed,
                       max_depth=7, min_leaf=4, class_count=3):
    unique_groups = np.unique(groups)
    forest = []
    for tree_index in range(tree_count):
        tree_generator = np.random.default_rng(seed + 1009 * tree_index)
        sampled_groups = tree_generator.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        sampled_rows = np.concatenate([
            np.flatnonzero(groups == group) for group in sampled_groups
        ])
        inbag_groups = np.unique(sampled_groups)
        oob_mask = ~np.isin(groups, inbag_groups)
        tree = build_random_tree(
            X[sampled_rows], y[sampled_rows], tree_generator,
            max_depth=max_depth, min_leaf=min_leaf, mtry=mtry,
            class_count=class_count,
        )
        forest.append({"tree": tree, "oob_mask": oob_mask, "inbag_groups": inbag_groups})
    return forest

def forest_probability(forest, X):
    predictions = [predict_tree_probability(item["tree"], X) for item in forest]
    return np.mean(predictions, axis=0)

def oob_probability(forest, X, class_count=3):
    probability_sum = np.zeros((len(X), class_count), dtype=float)
    vote_count = np.zeros(len(X), dtype=int)
    for item in forest:
        mask = item["oob_mask"]
        if np.any(mask):
            probability_sum[mask] += predict_tree_probability(item["tree"], X[mask])
            vote_count[mask] += 1
    probabilities = np.full_like(probability_sum, np.nan)
    valid = vote_count > 0
    probabilities[valid] = probability_sum[valid] / vote_count[valid, None]
    return probabilities, vote_count

smoke_forest = train_group_forest(
    X_train, y_train, groups_train, tree_count=7, mtry=3, seed=2062,
    max_depth=5, min_leaf=4,
)
smoke_probability = forest_probability(smoke_forest, X_train[:9])
smoke_oob, smoke_votes = oob_probability(smoke_forest, X_train)
assert smoke_probability.shape == (9, 3)
assert np.allclose(smoke_probability.sum(axis=1), 1.0)
assert np.all(smoke_votes >= 0)
print("7 棵树下 OOB 零票晶圆：", int(np.sum(smoke_votes == 0)))
```

<MlQuiz kind="代码审计" question="为什么每棵树要保存 in-bag/OOB 批次，而不能训练完只留树结构？" :options="['为了让树更深', '逐样本 OOB 聚合、重要性与审计需要知道哪些树没有见过该批次', '为了改变测试标签', '因为 Bootstrap 后不能预测新样本']" :answer="1" explanation="OOB 是树—样本资格关系；丢掉抽样账本后无法重建合法的 OOB 预测。" />

## 7. Bagging 与随机森林的实证对照

令 `mtry=d`，节点总能看到所有特征，得到树 Bagging；令 `mtry=3`，得到节点级随机特征森林。两者使用相同分组抽样、树数和容量。

```python
def accuracy_from_probability(probability, labels):
    return float(np.mean(np.argmax(probability, axis=1) == labels))

bagged_forest = train_group_forest(
    X_train, y_train, groups_train, tree_count=48, mtry=X_train.shape[1],
    seed=2070, max_depth=7, min_leaf=4,
)
random_forest = train_group_forest(
    X_train, y_train, groups_train, tree_count=48, mtry=3,
    seed=2070, max_depth=7, min_leaf=4,
)
bagged_validation_probability = forest_probability(bagged_forest, process_features[validation_rows])
random_validation_probability = forest_probability(random_forest, process_features[validation_rows])
bagged_validation_accuracy = accuracy_from_probability(
    bagged_validation_probability, fault_class[validation_rows]
)
random_validation_accuracy = accuracy_from_probability(
    random_validation_probability, fault_class[validation_rows]
)
assert 0 <= bagged_validation_accuracy <= 1
assert 0 <= random_validation_accuracy <= 1
print("验证准确率 Bagging / RF：", round(bagged_validation_accuracy, 3), round(random_validation_accuracy, 3))
```

单次验证分数不能证明随机森林总优于 Bagging。要验证去相关机制，还要在同一批样本上比较树预测。这里使用“腔壁累积”概率，并排除概率恒定的树。

```python
def mean_pairwise_tree_correlation(forest, X, class_index=1):
    matrix = np.vstack([
        predict_tree_probability(item["tree"], X)[:, class_index]
        for item in forest
    ])
    variable = np.std(matrix, axis=1) > 1e-12
    matrix = matrix[variable]
    if len(matrix) < 2:
        return 0.0
    correlation_matrix = np.corrcoef(matrix)
    upper = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    return float(np.mean(upper))

bagging_correlation = mean_pairwise_tree_correlation(
    bagged_forest, process_features[validation_rows]
)
forest_correlation = mean_pairwise_tree_correlation(
    random_forest, process_features[validation_rows]
)
assert -1 <= bagging_correlation <= 1
assert -1 <= forest_correlation <= 1
print("树间平均相关 Bagging / RF：", round(bagging_correlation, 3), round(forest_correlation, 3))
```

相关性应与单树强度一起读。只追求低相关，随机输出的树相关性也低，却没有预测价值。可同时报告平均单树 OOB 分数、树间相关、集成 OOB 分数和推理成本。

<MlQuiz kind="指标诊断" question="一个配置的树间相关性很低，但单树接近随机猜测，应怎样判断？" :options="['低相关已经证明配置最好', '去相关过度可能损伤单树强度，必须看集成 OOB/验证损失', '继续把 mtry 降到 0', '只报告相关性，不报告误差']" :answer="1" explanation="随机森林优化的是最终泛化，不是单独最小化相关系数；强度与相关性必须联合评估。" />

## 8. OOB：每个样本只有部分树有资格投票

对训练单位 $i$，合法树集合为

$$
\mathcal B_i^{\mathrm{OOB}}=\{b:W_i^{(b)}=0\}.
$$

分类 OOB 概率为

$$
\hat{\mathbf p}_i^{\mathrm{OOB}}
=\frac{1}{|\mathcal B_i^{\mathrm{OOB}}|}
\sum_{b\in\mathcal B_i^{\mathrm{OOB}}}\hat{\mathbf p}_b(\mathbf x_i).
$$

若每棵树独立抽样，单位 $i$ 的 OOB 票数近似服从

$$
V_i\sim\operatorname{Binomial}\left(B,p_{\mathrm{OOB}}(n)\right),
$$

所以期望约为 $B/e$，零票概率为 $(1-p_{\mathrm{OOB}})^B$。树很少时必须显式标记零票，不能拿全森林预测补洞。

```python
oob_forest = train_group_forest(
    X_train, y_train, groups_train, tree_count=100, mtry=3,
    seed=2080, max_depth=7, min_leaf=4,
)
train_oob_probability, train_oob_votes = oob_probability(oob_forest, X_train)
valid_oob = train_oob_votes > 0
oob_accuracy = accuracy_from_probability(
    train_oob_probability[valid_oob], y_train[valid_oob]
)
group_count = len(np.unique(groups_train))
exact_group_oob = (1 - 1 / group_count) ** group_count
expected_votes = 100 * exact_group_oob
assert valid_oob.all()
assert abs(train_oob_votes.mean() - expected_votes) < 8
print("OOB 准确率 / 最小-平均-最大票数：", round(oob_accuracy, 3), int(train_oob_votes.min()), round(float(train_oob_votes.mean()), 1), int(train_oob_votes.max()))
```

OOB 只免去“每组超参数都重新做一次 K 折”的成本，不免除以下边界：

- 标准化、缺失填补、目标编码和特征筛选必须在每棵树允许的数据内拟合；
- 重复测量要用分组 OOB，时间预测要用前向评价，空间/设备迁移要按目标域分组；
- 反复用 OOB 选几十个特征、阈值和配置后，最佳 OOB 分数已带选择偏差；
- 最终测试集仍应冻结，只在选择完成后使用一次；
- 类别极不平衡时，要检查每个 OOB 子集的类别覆盖与每类有效票数。

<MlQuiz kind="泄漏定位" question="随机森林用全体训练数据先做目标编码，再计算 OOB 误差，为什么仍会乐观？" :options="['OOB 禁止类别特征', '编码器已经从该样本及其标签聚合中偷看信息，树没见过该行也不能撤销上游泄漏', '树数太多', '只有回归会泄漏']" :answer="1" explanation="OOB 资格只约束树训练样本；任何在全量数据上拟合的监督预处理都绕过了这个边界。" />

### 8.1 用 OOB 选择 mtry，不用测试集挑赢家

下面固定种子、树容量和候选集合，只比较 `mtry`。真实项目还应重复多个森林种子，给出 OOB 差异的不确定性，并把业务成本作为主指标。

```python
mtry_candidates = [1, 2, 3, 5, 8]
oob_scores = []
candidate_forests = []
for candidate_mtry in mtry_candidates:
    candidate_forest = train_group_forest(
        X_train, y_train, groups_train, tree_count=42,
        mtry=candidate_mtry, seed=2090,
        max_depth=7, min_leaf=4,
    )
    candidate_oob, candidate_votes = oob_probability(candidate_forest, X_train)
    eligible = candidate_votes > 0
    score = accuracy_from_probability(candidate_oob[eligible], y_train[eligible])
    oob_scores.append(score)
    candidate_forests.append(candidate_forest)

best_index = int(np.argmax(oob_scores))
selected_mtry = mtry_candidates[best_index]
selected_oob_score = oob_scores[best_index]
assert selected_mtry in mtry_candidates
print("mtry→OOB：", dict(zip(mtry_candidates, np.round(oob_scores, 3))))
print("冻结 mtry：", selected_mtry, "OOB：", round(selected_oob_score, 3))
```

随后用训练集拟合更大的最终森林，用验证集选择动作阈值或概率校准；所有选择冻结后，测试集只使用一次。

```python
final_forest = train_group_forest(
    X_train, y_train, groups_train, tree_count=96,
    mtry=selected_mtry, seed=2100,
    max_depth=7, min_leaf=4,
)
validation_probability = forest_probability(final_forest, process_features[validation_rows])
test_probability = forest_probability(final_forest, process_features[test_rows])
validation_accuracy = accuracy_from_probability(validation_probability, fault_class[validation_rows])
test_accuracy = accuracy_from_probability(test_probability, fault_class[test_rows])
assert test_probability.shape == (48, 3)
assert np.allclose(test_probability.sum(axis=1), 1.0)
print("冻结后的验证/测试准确率：", round(validation_accuracy, 3), round(test_accuracy, 3))
```

### 8.2 树数是稳定预算，不是常规正则化旋钮

沿同一棵树序列做前缀聚合，才能画可比的 OOB 收敛曲线；为每个 $B$ 重新换种子，会把树数效应和随机波动混在一起。

```python
long_forest = train_group_forest(
    X_train, y_train, groups_train, tree_count=120,
    mtry=selected_mtry, seed=2110,
    max_depth=7, min_leaf=4,
)
checkpoints = [5, 10, 20, 40, 80, 120]
oob_curve = []
coverage_curve = []
for checkpoint in checkpoints:
    prefix_probability, prefix_votes = oob_probability(long_forest[:checkpoint], X_train)
    eligible = prefix_votes > 0
    oob_curve.append(accuracy_from_probability(prefix_probability[eligible], y_train[eligible]))
    coverage_curve.append(float(np.mean(eligible)))
assert all(later + 1e-12 >= earlier for earlier, later in zip(coverage_curve, coverage_curve[1:]))
print("B→OOB 准确率：", dict(zip(checkpoints, np.round(oob_curve, 3))))
print("B→OOB 覆盖率：", dict(zip(checkpoints, np.round(coverage_curve, 3))))
```

误差曲线不必单调，但通常逐渐稳定。继续加树一般不会像加深单树那样自动提高模型容量；它主要减少对有限随机树样本的依赖。不过若持续盯着测试曲线决定何时停止，仍会发生测试集选择泄漏。

<MlQuiz kind="实验设计" question="比较 B=20、40、80 的 OOB 曲线时，哪种做法最能隔离树数效应？" :options="['为每个 B 重新生成完全不同的森林', '训练一条固定随机种子的 80 树序列，评估其 20/40/80 前缀', '每次看测试集直到满意', '只比较训练准确率']" :answer="1" explanation="前缀比较共享前面的树，避免把随机种子差异误当成树数效应；测试集不能参与停止选择。" />

## 9. 特征重要性：解释森林最容易走偏的地方

### 9.1 不纯度下降重要性（MDI）

把所有使用特征 $j$ 的节点加权 gain 相加：

$$
I_j^{\mathrm{MDI}}
=\frac1B\sum_{b=1}^{B}\sum_{t\in T_b:\,v(t)=j}
\frac{n_t}{n}\,\Delta I_t.
$$

它计算便宜，却偏爱拥有更多候选阈值的连续/高基数字段；相关特征会竞争同一份 gain；训练不纯度也只描述训练分布内的预测关联，不是因果效应。

### 9.2 验证集分组置换重要性

在冻结的验证集上打乱特征 $j$，性能下降定义为

$$
I_j^{\mathrm{perm}}
=S(\mathcal D_{\mathrm{val}})
-S(\pi_j(\mathcal D_{\mathrm{val}})).
$$

重复测量应按批次块置换，保留片内结构。若打乱后造出物理上不可能的特征组合，结果只说明模型在该人工扰动上的依赖。

```python
def permute_feature_by_group(X, groups, feature_index, generator):
    result = X.copy()
    unique_groups = np.unique(groups)
    donor_groups = generator.permutation(unique_groups)
    for receiver, donor in zip(unique_groups, donor_groups):
        receiver_rows = np.flatnonzero(groups == receiver)
        donor_rows = np.flatnonzero(groups == donor)
        if len(receiver_rows) != len(donor_rows):
            raise ValueError("教学实现要求每组行数相同")
        result[receiver_rows, feature_index] = X[donor_rows, feature_index]
    return result

X_validation = process_features[validation_rows]
y_validation = fault_class[validation_rows]
groups_validation = lot_id[validation_rows]
baseline_validation_score = accuracy_from_probability(validation_probability, y_validation)
permutation_importance = []
for feature_index in range(X_validation.shape[1]):
    drops = []
    for repeat in range(8):
        permuted = permute_feature_by_group(
            X_validation, groups_validation, feature_index,
            np.random.default_rng(2120 + 31 * feature_index + repeat),
        )
        score = accuracy_from_probability(forest_probability(final_forest, permuted), y_validation)
        drops.append(baseline_validation_score - score)
    permutation_importance.append(float(np.mean(drops)))

importance_order = np.argsort(permutation_importance)[::-1]
print("验证集分组置换重要性前五：")
for index in importance_order[:5]:
    print(feature_names[index], round(permutation_importance[index], 3))
```

### 9.3 相关特征要联合置换

`endpoint_slope` 与 `endpoint_slope_copy` 可互相替代。单独打乱一列，森林仍可走另一列；联合打乱更接近“移除这组证据”的问题。

```python
def permute_feature_group_jointly(X, groups, feature_indices, generator):
    result = X.copy()
    unique_groups = np.unique(groups)
    donor_groups = generator.permutation(unique_groups)
    for receiver, donor in zip(unique_groups, donor_groups):
        receiver_rows = np.flatnonzero(groups == receiver)
        donor_rows = np.flatnonzero(groups == donor)
        result[np.ix_(receiver_rows, feature_indices)] = X[np.ix_(donor_rows, feature_indices)]
    return result

joint_drops = []
for repeat in range(12):
    jointly_permuted = permute_feature_group_jointly(
        X_validation, groups_validation, [0, 1], np.random.default_rng(2140 + repeat)
    )
    joint_score = accuracy_from_probability(
        forest_probability(final_forest, jointly_permuted), y_validation
    )
    joint_drops.append(baseline_validation_score - joint_score)
joint_importance = float(np.mean(joint_drops))
individual_sum = permutation_importance[0] + permutation_importance[1]
print("两个单列重要性之和 / 联合重要性：", round(individual_sum, 3), round(joint_importance, 3))
```

重要性还不能回答“把 MFC 设定值改动 1% 会发生什么”。那是干预与因果问题，需要过程知识、实验设计和安全审查。

<div class="ml-stop"><strong>停下来推一推 4</strong>复制一个强特征形成两列。分别预测 MDI、单列置换和联合置换会怎样变化。哪些变化来自模型替代性，哪些并不能说明物理机理？</div>

<MlQuiz kind="解释审计" question="两个高度相关特征的单列置换重要性都很低，最合理的下一步是什么？" :options="['断言两者都无用', '检查联合/条件置换、浅层切分占比与领域机理', '把重要性解释成因果效应', '删除验证集']" :answer="1" explanation="相关特征能互相补偿，单列置换会低估特征组的预测依赖；仍需结合运行域和机理审计。" />

## 数值验证

### 10.1 概率、票差与不确定性不是同一件事

分类森林可输出平均概率、硬票比例、第一与第二类的 margin，以及树预测分歧。它们都能帮助排序复核，但都不自动等于真实错误概率。

多分类 Brier score 为

$$
\operatorname{Brier}=\frac1n\sum_{i=1}^{n}
\sum_{k=1}^{K}(\bar p_{ik}-\mathbf1[y_i=k])^2.
$$

```python
def multiclass_brier(probability, labels, class_count=3):
    one_hot = np.eye(class_count)[labels]
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))

sorted_probability = np.sort(test_probability, axis=1)
vote_margin = sorted_probability[:, -1] - sorted_probability[:, -2]
test_brier = multiclass_brier(test_probability, fault_class[test_rows])
low_margin_rate = float(np.mean(vote_margin < 0.20))
assert 0 <= test_brier <= 2
assert 0 <= low_margin_rate <= 1
print("测试 Brier / margin<0.20 比例：", round(test_brier, 3), round(low_margin_rate, 3))
```

低 margin 可以触发复核，高 margin 也不能越过运行域门禁。所有树在未见的新配方上可能一致地外推错误；集成一致不等于认知不确定性已经量化。校准器必须只在验证数据上拟合，并在独立测试与未来批次上复核。

<MlQuiz kind="部署判断" question="一批新配方晶圆得到 0.98 的森林最大类概率，能否自动放行？" :options="['能，高概率就是安全保证', '不能，先检查运行域、配方版本、校准与动作风险', '只要树数超过 100 就能', 'OOB 会自动覆盖所有新配方']" :answer="1" explanation="森林概率是训练分布内的预测汇总；运行域外的一致错误仍可能给出极高概率。" />

### 10.2 边界与退化验证

一套最小测试至少覆盖：

- `B=1` 时森林退化为单树；
- `mtry=d` 时节点特征机制退化为树 Bagging；
- 相同种子与数据得到逐节点相同的模型；
- 加树只追加新种子，不改写已有树；
- 每棵树的 in-bag 与 OOB 组互斥；
- OOB 票数为 0 时输出缺失而不是全森林结果；
- 类别平票、概率平票、阈值相等和重复特征都有固定规则；
- 非法 `B`、`mtry`、空数据和 NaN 不会静默训练。

```python
forest_a = train_group_forest(
    X_train, y_train, groups_train, tree_count=12, mtry=3,
    seed=2150, max_depth=5, min_leaf=4,
)
forest_b = train_group_forest(
    X_train, y_train, groups_train, tree_count=20, mtry=3,
    seed=2150, max_depth=5, min_leaf=4,
)
prefix_a = forest_probability(forest_a, X_validation[:6])
prefix_b = forest_probability(forest_b[:12], X_validation[:6])
assert np.allclose(prefix_a, prefix_b)
for item in forest_b:
    oob_groups = set(groups_train[item["oob_mask"]].tolist())
    assert oob_groups.isdisjoint(set(item["inbag_groups"].tolist()))
print("随机种子前缀与 OOB 互斥测试通过")
```

## 复杂度与工程边界

### 11.1 训练、内存与推理

设样本数 $n$、特征数 $d$、树数 $B$、平均深度 $D$、每节点候选数 $m$。若每节点高效扫描已排序或直方图特征，粗略量级为

$$
T_{\mathrm{train}}=O(Bm nD),
\qquad
T_{\mathrm{infer}}=O(BD) \text{每样本},
\qquad
M=O(BN_{\mathrm{node}}).
$$

朴素实现每节点重新排序会额外付出对数因子。树之间天然可并行，但并行会增加数据副本、直方图、线程栈和归约峰值内存。平均延迟不是容量门禁，必须实测高分位、最坏深度和批量大小。

```python
production_rows = 8_000_000
production_features = 180
production_trees = 600
average_depth = 14
mtry_budget = 16
matrix_gib = production_rows * production_features * 4 / 2 ** 30
comparisons_per_record = production_trees * average_depth
feature_scans_proxy = production_trees * mtry_budget * production_rows * average_depth
oob_vote_cells = production_rows * 4  # 仅计一个 int32 票数向量
print("float32 设计矩阵 GiB：", round(matrix_gib, 2))
print("每条记录树节点比较：", comparisons_per_record)
print("训练候选扫描代理量：", f"{feature_scans_proxy:.3e}")
print("单个 OOB 票数向量 MiB：", round(oob_vote_cells / 2 ** 20, 2))
```

生产实现常用量化直方图、特征分箱、列采样、树并行、叶并行、批量向量化和模型压缩。分箱边界、浮点精度、线程归约顺序与随机数算法都会改变树结构，必须进入复现合同。

### 11.2 何时不该选随机森林

- 需要平滑外推连续目标；树只能在训练范围内作分段常数延伸；
- 概率必须高度校准，而可用校准集很小；
- 延迟或模型内存无法容纳数百棵树；
- 特征主要是超高维稀疏文本，线性模型可能更直接；
- 需要严格单调、可验证的少量规则，普通森林不满足结构约束；
- 数据强时间依赖，却只能使用普通 Bootstrap/OOB；
- 要回答干预因果问题，而不是预测关联；
- 新设备/新配方远离训练运行域，森林的高置信投票没有外推保证。

<MlQuiz kind="容量估算" question="一座 600 棵、平均深度 14 的二叉森林对单条记录推理，主要节点比较量级是多少？" :options="['约 14 次', '约 600 次', '约 8400 次', '必须访问全部叶，无法估算']" :answer="2" explanation="每棵树沿一条约 14 层的路径，600 棵树合计约 600×14 次比较；实际延迟还取决于内存局部性与批处理。" />

## 12. 工业训练与发布协议

### 12.1 训练制品不是一个 `forest.pkl`

最低制品应包含：

- 原始训练快照、标签成熟时间、排除规则与特征 schema；
- 抽样单位、每棵树的 in-bag/OOB 组或可重建种子；
- `B`、`mtry`、深度、最小叶、类别权重、平票和缺失协议；
- 特征顺序、单位、分箱、类别字典和相关特征组；
- 随机数算法、主种子、树编号到子种子的映射；
- 树结构、阈值精度、比较符、叶统计量与概率平滑；
- OOB 选择轨迹、验证阈值、最终测试报告和拒绝规则；
- 训练库、编译器、线程数、模型哈希和跨语言金丝雀。

仅保存主种子而不冻结随机数算法与并行调度，跨版本仍可能生成不同森林。若要增量追加树，已有树的子种子必须只依赖主种子与树编号。

### 12.2 线上最低监控面板

- 输入缺失、饱和、量程、配方族、腔体与 campaign 覆盖；
- 每类概率、第一—第二类 margin、拒绝率与人工复核队列；
- 树投票熵、平均树相关、叶覆盖和极小叶命中率；
- 浅层特征使用占比、相关特征组联合重要性与重要性漂移；
- 成熟标签后的每类召回、Brier/NLL、动作成本与批次 bootstrap 区间；
- OOB 训练基线与线上新批次性能的差异；
- P50/P95/P99 延迟、模型常驻内存、树数和节点数；
- 模型哈希、schema、随机种子、阈值精度与回退版本。

### 12.3 常见失败模式

| 现象 | 更可能的根因 | 首要检查 |
|---|---|---|
| OOB 很高，未来批次明显下降 | 行 Bootstrap、预处理泄漏或 campaign 漂移 | 抽样单位、编码器拟合边界、外层分组测试 |
| 加树后误差很快不再改善 | 相关误差地板或共同偏差 | 树间相关、mtry、特征冗余、叶容量 |
| 单列重要性突然下降 | 相关副本接入或分布变化 | 联合置换、schema diff、浅层切分占比 |
| 高置信错误集中在新配方 | 运行域外一致外推 | 配方门禁、距离/密度、人工复核 |
| 相同模型包在两服务输出不同 | 阈值精度、比较符或类别顺序不同 | 逐树/逐节点金丝雀回放 |
| 延迟随树数线性上升并抖动 | 内存访问与批量不足 | 树布局、批量、线程、P99 路径 |

<MlQuiz kind="故障定位" question="OOB 指标稳定，但新腔体 campaign 上三类召回同时下降，优先做什么？" :options="['继续增加同分布树数', '核对 campaign 运行域、特征语义与外层分组评价', '删除 OOB 日志', '把所有概率乘以 2']" :answer="1" explanation="OOB 估计的是当前抽样单位和训练域内泛化；新 campaign 的共同漂移不会被普通批次 OOB 自动覆盖。" />

## 13. 验收清单

### 数据与抽样

- [ ] Bootstrap 单位与部署实体一致，重复测量不会跨 in-bag/OOB。
- [ ] 时间、设备、campaign 和批次的层级关系已写进切分协议。
- [ ] 上游填补、编码、筛选和校准没有使用不允许的 OOB/验证信息。
- [ ] 相关副本、高基数身份字段、维护后写字段和标签成熟时间已审计。

### 算法与统计

- [ ] 有限 $n$ 唯一率/OOB 率与模拟一致，重复次数总和等于抽样次数。
- [ ] 相关平均方差公式通过模拟，树数与相关性分开解释。
- [ ] `mtry=d`、`mtry=1`、`B=1`、纯节点、重复阈值和平票均有测试。
- [ ] 每个节点独立抽候选特征，不把整树固定子空间误叫随机森林。
- [ ] OOB 预测只聚合未见该组的树，零票与低票样本显式标记。
- [ ] OOB 用于选择后，最终测试仍保持冻结。
- [ ] MDI、单列置换和联合置换的解释边界已区分。

### 系统与运行域

- [ ] 树编号—子种子映射允许只追加树，不改写已有树。
- [ ] 训练峰值内存、并行度、模型大小与 P99 延迟已实测。
- [ ] 比较符、阈值精度、类别顺序和叶概率可跨语言逐树回放。
- [ ] 运行域外配方、腔体、缺失组合和高风险动作可拒绝。
- [ ] 标签成熟后持续监控分层召回、成本、校准、投票分歧与漂移。

## 14. 练习与最小实现

### 练习 1：枚举 Bootstrap 权重

对 $n=4$ 枚举全部 $4^4$ 个抽样序列，统计唯一单位数、最大重复次数和每个单位的 OOB 频率。与精确式比较，并解释这些事件为什么不相互独立。

### 练习 2：实现回归 Bagging

把本章分类叶替换成平方损失均值叶：

- 输出每棵树、森林均值和 OOB 均值；
- 比较单树、Bagging 与随机森林的 RMSE/MAE；
- 用多个独立训练集分离估计偏差和方差；
- 检查训练范围外的分段常数外推；
- 报告树间残差相关与 $B$ 的边际收益。

### 练习 3：不等长组的分组 Bootstrap

允许每个批次晶圆数不同，分别实现“每组等权”和“每片等权”的目标。说明抽组后复制整组会怎样改变叶计数，并设计与业务风险一致的加权规则。

### 练习 4：完整 OOB 重要性

为每棵树只在它自己的 OOB 组上计算基线和置换损失，再跨树/样本聚合。实现：

- 单列置换与相关特征组联合置换；
- 每组同一置换映射，避免打散重复测量；
- 置信区间与负重要性保留；
- 与冻结验证集重要性比较；
- 禁止测试集参与特征筛选。

### 练习 5：框架对照

下面的 scikit-learn 片段只作可选接口核对。它的普通 `bootstrap=True` 按行抽样，不能替代本章的批次级 Bootstrap；若业务需要分组抽样，要在外部生成每棵树的组样本或使用明确支持该协议的实现。

```python
# 不执行：需要可选的 scikit-learn 环境；普通 RandomForestClassifier 按行 Bootstrap。
from sklearn.ensemble import RandomForestClassifier

library_forest = RandomForestClassifier(
    n_estimators=400,
    max_features=selected_mtry,
    min_samples_leaf=4,
    bootstrap=True,
    oob_score=True,
    random_state=20,
    n_jobs=-1,
)
library_forest.fit(X_train, y_train)
library_probability = library_forest.predict_proba(process_features[test_rows])
assert library_probability.shape == (test_rows.sum(), 3)
```

```python
# TODO：完成支持分组抽样、OOB 审计和稳定导出的工业随机森林。
#
# 1. 实现分类 Gini 与回归 SSE、节点级候选抽样和确定性平票；
# 2. 支持按批次/campaign Bootstrap、不等长组权重和逐树 OOB 账本；
# 3. 用多个森林种子联合选择 mtry、叶容量与树数稳定预算；
# 4. 实现概率平均、硬投票、OOB 覆盖、校准和动作阈值冻结；
# 5. 实现 MDI、单列/联合 OOB 置换重要性并保留不确定性；
# 6. 导出特征 schema、阈值精度、比较符、类别顺序和子种子映射；
# 7. 报告分层误差、树间相关、P99 延迟、内存、节点数和运行域拒绝。
```

## 15. 本章结论

- Bootstrap 从 $n$ 个单位有放回抽 $n$ 次，抽样长度仍为 $n$，不同单位期望比例趋近 63.2%。
- 固定单位 OOB 的精确概率是 $(1-1/n)^n$，OOB 资格属于“树—单位”关系，不是全局数据切分。
- 重复测量必须按部署实体分组 Bootstrap；按行 OOB 会因同组兄弟入袋而泄漏。
- Bagging 聚合预测，不聚合树结构；硬投票与概率平均是不同接口，可能给出不同类别。
- 等方差、等相关诊断模型给出 $\sigma^2[\rho+(1-\rho)/B]$，相关误差形成树数无法消除的地板。
- Bagging 最适合低偏差、高方差且对样本扰动敏感的基学习器；共同偏差不会被平均消除。
- 随机森林把 Bootstrap 与节点级特征候选抽样结合；随机子空间通常为整棵学习器固定特征集。
- `mtry` 同时改变单树强度和树间相关性；默认经验值只是搜索起点。
- 复制强特征可能恢复浅层统治、提高树相关，并让单列重要性相互补偿。
- OOB 每个样本约获得 $B/e$ 张票；低票和零票必须显式记录。
- OOB 能降低模型选择成本，但不免除预处理泄漏、分组/时间边界、选择偏差和最终测试冻结。
- 树数主要控制有限森林的蒙特卡洛稳定性；公平比较应使用同一树序列的前缀。
- MDI 受候选数与相关性影响；置换重要性衡量预测依赖，不是因果效应。
- 森林概率、票差与树一致性都不自动提供运行域外保证，仍需校准和拒绝策略。
- 生产制品必须保存抽样、随机数、特征、阈值、类别、OOB 与验证协议，而不只是树结构。
- 训练和推理成本近似随树数线性增长；并行、分箱和量化同时影响容量与复现。

## 来源与许可

Jonathan Richard Shewchuk 的 CS 189/289A 课程讲义源 PDF 第 90–94 页用于核对单树高方差、平均动机、Bagging、约 63.2% 唯一入袋比例、随机森林节点级特征抽样、分类常用 $\sqrt d$ / 回归常用 $d/3$ 起点及单树强度—相关性权衡。本章没有复用其课堂图形、案例或措辞。

CMU 10-301/10-601 Lecture 24 *Ensemble Methods: Bagging* 源 PDF 第 29–32 页用于核对样本 Bagging、分类投票、回归平均与随机子空间；源 PDF 第 34–41 页用于核对随机森林和逐节点特征随机化；源 PDF 第 43–48 页用于核对 OOB 误差、OOB 选参与不纯度重要性。HW9 Recitation Solutions 源 PDF 第 8–9 页用于交叉核对只用未见样本的树形成 OOB 预测和节点候选特征随机化。本章不复制课件表格、题图、解答、数据或表达。

等离子刻蚀腔体批次异常分诊的 60 个教学批次、240 片合成晶圆、三类标签、同源斜率副本、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作。数值不代表真实晶圆厂、腔体、配方、传感器、缺陷率、工艺窗口、维修策略或安全阈值；模型不得直接调整 RF 功率、气体流量、腔压、端点条件、设备联锁或自动放行产品。scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

<MlLessonProgress
  lesson-id="20-random-forests"
  next-href="/courses/machine-learning/21-boosting"
  next-label="第 21 章：AdaBoost 与梯度提升"
/>
