---
title: 聚类评估与模型选择
description: 从压缩空气站复核队列验收案例推导层次聚类、linkage、Dendrogram、内部指标、稳定性、外部指标、图拉普拉斯与谱聚类
course: machine-learning
edition: industrial
---

<MlLessonHero
  number="27"
  title="聚类评估与模型选择"
  question="没有唯一真值，怎样证明三个复核队列不是一条漂亮的散点图，而是经得住重采样、用途约束和非凸反例的工程划分？"
  duration="建议 720–840 分钟"
  :outcomes="[
    '把聚类评价写成用途、几何、稳定性、外部审计与运行容量组成的证据链，而不是追逐一项最高分',
    '从簇间距离定义实现 single、complete、average linkage，逐步生成并正确读取 Dendrogram',
    '从样本级轮廓系数推导 WCSS、Silhouette、Calinski–Harabasz 与 Davies–Bouldin 的偏好和失效边界',
    '实现分组重采样稳定性、ARI 与 NMI，并解释聚类目标为什么不等于所谓真实类别',
    '从图切割能量推导图拉普拉斯、归一化谱嵌入，并在非凸工况轨迹上比较 K-means 与谱聚类'
  ]"
/>

上一章回答“给定 $K$，怎样把窗口压缩成 $K$ 个欧氏原型”。这一章回答更危险的问题：为什么是这个 $K$、这个距离和这个划分？

没有标签不等于没有评价标准，也不等于存在一项可以代替工程判断的“无监督准确率”。聚类算法总在某个表示、距离和目标下制造结构；评价指标又把某种结构偏好编码成分数。工业验收必须让这些假设逐项见光。

> 本章延续压缩空气站教学场景，但使用一个独立生成、可完整重放的冻结快照。隐藏模式只用于合成器单元测试和候选冻结后的少量审计，不参与距离、层次树、稳定性或 $K$ 的预选。统计簇仍不是故障根因，不得直接启停设备、改变压力设定、旁路保护或生成维修结论。

## 贯穿案例：三个复核队列能否签字发布

### 1.1 这不是“找真实类别”

工程组准备把 528 个小时窗口分到若干复核队列，每周从各队列抽取代表记录和边界记录。候选方案包括 $K=2,ldots,6$，上线系统最多能稳定维护三个队列。

这里至少存在四种都合理、却不等价的分法：

- 按欧氏量测模式分组，服务于原型压缩；
- 按时间轨迹分组，服务于启停过程复盘；
- 按维护动作分组，服务于工单路由；
- 按已确认根因分组，服务于监督诊断。

本章目标只是第一种。即使某个统计簇与历史工单高度一致，也不能据此宣称算法“发现了真实故障”。真实世界可以同时有多种有效分区；有时一个物理状态还会在不同负荷下形成多个几何子簇。

<MlQuiz
  kind="目标合同"
  question="某聚类与故障标签的 ARI 只有 0.35，但能稳定覆盖三种人工复核队列。最准确的结论是什么？"
  :options="['它没有复现故障标签，但是否适合复核队列仍要按预先声明的用途判断', '算法必然错误', '故障标签必然错误', '只要把簇编号重排，ARI 就会变成 1']"
  :answer="0"
  explanation="ARI 只衡量两个分区的一致性，不宣布哪一个是唯一真理。用途合同与外部标签回答的是不同问题。"
/>

### 1.2 发布前冻结五层证据

| 层次 | 本章问题 | 不能替代什么 |
|---|---|---|
| 目标与用途 | 分组后允许什么动作，人工能维护几个队列 | 不能由内部指标临时决定 |
| 几何 | 簇内是否紧、簇间是否远、树在何高度合并 | 不能证明稳定或有物理意义 |
| 稳定性 | 换一批 campaign 后分区是否近似重现 | 稳定不等于有用 |
| 外部审计 | 与少量独立复核标签是否一致 | 标签不是免费真值，也不能反复调参 |
| 运行风险 | 极小簇、漂移、拒绝率和人员负荷是否可控 | 不能被一个平均分掩盖 |

验收顺序也要冻结：先用训练集几何和稳定性形成短名单，再检查队列容量，最后只打开一次独立审计标签。若先看标签再改 $K$、距离或特征，所谓“外部评价集”已经变成调参集。

## 2. 数据合同：campaign 是隔离单元，隐藏模式不是训练字段

### 2.1 五项无量纲过程指数

每个 production campaign 含 8 个相邻窗口，共 66 个 campaign。输入是已经过量测系统确认、方向统一到“数值越大代表该维度越强”的五项教学指数：

| 字段 | 工程含义 | 风险 |
|---|---|---|
| `load_ratio` | 负荷占比 | 时间相邻窗口强相关 |
| `specific_power_index` | 比功率指数 | 不能自动解释成泄漏 |
| `cooling_stress_index` | 冷却负担指数 | 受季节和水侧工况影响 |
| `dewpoint_risk_index` | 露点裕量风险指数 | 只能触发复核，不能旁路联锁 |
| `cycling_index` | 加卸载循环强度 | 与控制策略版本相关 |

尽管原始字段已无量纲，不同维度的样本方差仍不同，因此距离空间继续只用训练 campaign 的均值和标准差冻结。

```python
import numpy as np

SEED = 2707
FEATURE_NAMES = np.array([
    "load_ratio",
    "specific_power_index",
    "cooling_stress_index",
    "dewpoint_risk_index",
    "cycling_index",
])

MODE_MEANS = np.array([
    [0.82, 0.22, 0.34, 0.28, 0.31],
    [0.47, 0.48, 0.60, 0.54, 0.57],
    [0.21, 0.78, 0.83, 0.76, 0.80],
])

def make_review_dataset(seed=SEED):
    rng = np.random.default_rng(seed)
    campaign_mode = rng.choice(3, size=66, p=[0.39, 0.36, 0.25])
    rows, groups, audit = [], [], []
    for campaign, mode in enumerate(campaign_mode):
        campaign_offset = rng.normal(
            0,
            [0.025, 0.020, 0.025, 0.020, 0.020],
        )
        for window in range(8):
            within_campaign_drift = (window - 3.5) * np.array([
                0.006, 0.002, -0.003, 0.004, -0.002,
            ])
            row = (
                MODE_MEANS[mode]
                + campaign_offset
                + within_campaign_drift
                + rng.normal(0, 0.045, 5)
            )
            rows.append(np.clip(row, 0, 1))
            groups.append(campaign)
            audit.append(mode)
    return np.asarray(rows), np.asarray(groups), np.asarray(audit)

X_RAW, CAMPAIGN, MODE_AUDIT = make_review_dataset()

assert X_RAW.shape == (528, 5)
assert CAMPAIGN.shape == MODE_AUDIT.shape == (528,)
assert np.isfinite(X_RAW).all()
print("记录 / 特征 / campaign：", X_RAW.shape, len(np.unique(CAMPAIGN)))
```

输出为：

```text
记录 / 特征 / campaign： (528, 5) 66
```

### 2.2 训练几何与外部审计必须隔离

前 48 个 campaign 用于构造候选、拟合尺度和评估重采样稳定性；后 18 个只用于冻结后的复核。其中每四条才有一条合成审计标签，模拟昂贵的工程复核。

```python
train_campaigns = np.unique(CAMPAIGN)[:48]
train_mask = np.isin(CAMPAIGN, train_campaigns)
eval_mask = ~train_mask

scale_mean = X_RAW[train_mask].mean(axis=0)
scale_std = X_RAW[train_mask].std(axis=0, ddof=1)
if np.any(scale_std <= 0):
    raise ValueError("训练集存在零方差特征，无法冻结欧氏距离空间")

X_TRAIN = (X_RAW[train_mask] - scale_mean) / scale_std
X_EVAL = (X_RAW[eval_mask] - scale_mean) / scale_std
GROUP_TRAIN = CAMPAIGN[train_mask]
AUDIT_EVAL_ALL = MODE_AUDIT[eval_mask]
AUDIT_REVIEWED = np.arange(len(X_EVAL)) % 4 == 0

assert X_TRAIN.shape == (384, 5)
assert X_EVAL.shape == (144, 5)
assert AUDIT_REVIEWED.sum() == 36
assert set(GROUP_TRAIN).isdisjoint(CAMPAIGN[eval_mask])
assert np.allclose(X_TRAIN.mean(axis=0), 0, atol=1e-12)
print("train / eval / 独立审计：", 384, 144, AUDIT_REVIEWED.sum())
```

输出为：

```text
train / eval / 独立审计： 384 144 36
```

<MlQuiz
  kind="评价隔离"
  question="为什么 36 条审计标签不能在候选 K=2…6 之间反复试到 ARI 最高？"
  :options="['这样会把独立审计集变成模型选择集，最终一致性估计偏乐观', '因为 ARI 只能用于训练数据', '因为标签编号必须与簇编号相同', '因为无监督学习禁止保存任何标签']"
  :answer="0"
  explanation="标签可以用于外部评价，但其使用时序必须像测试集一样受控；反复查看并调整会消耗独立性。"
/>

## 3. 层次聚类先回答：簇与簇怎样算距离

K-means 直接优化固定 $K$ 的原型。层次聚类（hierarchical agglomerative clustering, HAC）从 $N$ 个单样本簇开始，每一步合并当前距离最近的两个簇，直到只剩一个簇。整条合并历史形成一棵树；在不同高度截断，同一棵树可以得到不同 $K$。

关键不在“每次找最近”，而在两个集合 $A,B$ 之间的距离怎样定义。给定点距离 $d(mathbf x_i,mathbf x_j)$：

$$
D_{\text{single}}(A,B)
=\min_{i\in A,j\in B}d(\mathbf x_i,\mathbf x_j),
$$

$$
D_{\text{complete}}(A,B)
=\max_{i\in A,j\in B}d(\mathbf x_i,\mathbf x_j),
$$

$$
D_{\text{average}}(A,B)
=\frac{1}{|A||B|}
\sum_{i\in A}\sum_{j\in B}d(\mathbf x_i,\mathbf x_j).
$$

- single linkage 只要两簇各有一个点相近就愿意合并，能沿弯曲形状延伸，却容易被桥接点串成链；
- complete linkage 要求最远点也不能太远，偏好直径较小、较均衡的簇，但对离群点敏感；
- average linkage 平均所有跨簇点对，介于两者之间，仍受距离尺度和样本密度影响。

这三者不是“同一算法的三个速度选项”，而是三种不同的集合几何。

<MlQuiz
  kind="linkage 几何"
  question="两个大簇之间只有一对桥接点很近，其余跨簇点都很远。哪种 linkage 最可能最早合并它们？"
  :options="['single linkage', 'complete linkage', 'average linkage 必然更早', '三者距离永远相同']"
  :answer="0"
  explanation="single linkage 只取最小跨簇距离，因此一对近邻就能触发合并，这也是 chaining 的来源。"
/>

### 3.1 六个代表窗口的手算量规

先用一维坐标隔离 linkage 本身。六个代表窗口的过程指数为

$$
A=0.0, B=0.7, C=1.5, D=3.1, E=4.0, F=6.3.
$$

第一步三种 linkage 都把 $A,B$ 以高度 $0.7$ 合并，因为单样本簇之间没有定义差异。第二步开始，single 对 $\{A,B\}$ 与 $C$ 的距离是 $\min(1.5,0.8)=0.8$；complete 是 $\max(1.5,0.8)=1.5$；average 是 $(1.5+0.8)/2=1.15$。

::: ml-stop 停下来推一推 1
若把 $C$ 从 1.5 移到 2.4，先不要运行代码：分别计算 $D_{\text{single}}(\{A,B\},C)$、$D_{\text{complete}}(\{A,B\},C)$ 和 $D_{\text{average}}(\{A,B\},C)$，再判断 $D,E$ 是否会抢先合并。
:::

```python
HAC_X = np.array([[0.0], [0.7], [1.5], [3.1], [4.0], [6.3]])
HAC_NAMES = np.array(list("ABCDEF"))

def pairwise_euclidean(x):
    squared = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.maximum(squared, 0))

def linkage_distance(distance_matrix, left, right, method):
    cross = distance_matrix[np.ix_(left, right)]
    if method == "single":
        return float(cross.min())
    if method == "complete":
        return float(cross.max())
    if method == "average":
        return float(cross.mean())
    raise ValueError(f"未知 linkage：{method}")

HAC_D = pairwise_euclidean(HAC_X)
AB, C = [0, 1], [2]
assert np.isclose(linkage_distance(HAC_D, AB, C, "single"), 0.8)
assert np.isclose(linkage_distance(HAC_D, AB, C, "complete"), 1.5)
assert np.isclose(linkage_distance(HAC_D, AB, C, "average"), 1.15)
print("AB 到 C：single / complete / average = 0.80 / 1.50 / 1.15")
```

### 3.2 从零生成合并历史

实现必须保存左右子树、合并高度和成员数，而不是只返回最终标签。距离相同时还要有确定性 tie-break，否则同一数据在不同遍历顺序下可能画出不同但等价的树。

```python
def agglomerative_history(x, method):
    distance_matrix = pairwise_euclidean(x)
    clusters = {index: [index] for index in range(len(x))}
    next_cluster_id = len(x)
    history = []

    while len(clusters) > 1:
        candidates = []
        cluster_ids = sorted(clusters)
        for left_index, left_id in enumerate(cluster_ids):
            for right_id in cluster_ids[left_index + 1:]:
                distance = linkage_distance(
                    distance_matrix,
                    clusters[left_id],
                    clusters[right_id],
                    method,
                )
                candidates.append((distance, left_id, right_id))

        distance, left_id, right_id = min(candidates)
        members = sorted(clusters.pop(left_id) + clusters.pop(right_id))
        clusters[next_cluster_id] = members
        history.append({
            "id": next_cluster_id,
            "left": left_id,
            "right": right_id,
            "height": distance,
            "members": members,
        })
        next_cluster_id += 1
    return history

for method in ["single", "complete", "average"]:
    tree = agglomerative_history(HAC_X, method)
    heights = np.array([merge["height"] for merge in tree])
    assert len(tree) == len(HAC_X) - 1
    assert np.all(np.diff(heights) >= -1e-12)
    print(method, np.round(heights, 3))
```

输出为：

```text
single [0.7 0.8 0.9 1.6 2.3]
complete [0.7 0.9 1.5 3.2 6.3]
average [0.7   0.9   1.15  2.75  3.733]
```

这里的高度单调断言适用于本章三种 linkage。若使用 centroid linkage，两个簇合并后的新质心可能比原来的合并高度更靠近别的簇，Dendrogram 会出现 inversion；不能把所有“层次聚类”都默认为单调树。

<LinkageDendrogramBenchLab />

## 4. Dendrogram 怎样读，怎样被误读

Dendrogram 的叶子是样本，内部结点是一次合并，纵坐标是该次合并的 linkage 距离。水平切一刀，穿过多少条仍未合并的树枝，就得到多少个簇。

必须避免四种常见误读：

1. **横向距离没有量纲。** 为了不让枝条交叉，绘图库可以交换左右子树；叶子在屏幕上的左右间隔不等于样本距离。
2. **长竖线只是候选证据。** 某两个连续合并高度之间存在大跳跃，说明该 linkage 下有明显尺度分隔，不自动证明这个 $K$ 有业务意义。
3. **切树不重新优化。** 从完整树切出的 $K$ 簇受前面所有贪心合并约束，不等于 $K$-means 或全局最优 $K$ 分区。
4. **树不稳定时，枝条故事也不稳定。** 微小量测扰动可能交换早期合并，尤其是许多距离接近时。

若只为浏览 384 个训练窗口，完整距离矩阵需要 $O(N^2)$ 内存，朴素重复搜索可达 $O(N^3)$ 时间。成熟实现会使用 Lance–Williams 更新、最小堆或针对 linkage 的专用算法，但工业审计仍要记录距离、linkage、tie-break、截断高度和软件版本。

<MlQuiz
  kind="树图读取"
  question="Dendrogram 中 A 与 B 的叶子在水平方向画得很近，能否据此断言原始距离很小？"
  :options="['不能；横向叶序主要为排版，合并高度才承载 linkage 距离', '能，水平像素就是欧氏距离', '只有 complete linkage 可以', '只要树没有交叉就可以']"
  :answer="0"
  explanation="左右子树可交换而不改变层次结构。将横向排版解释成样本距离是典型误读。"
/>

## 5. WCSS 与肘部：只回答欧氏重构代价

对候选分配 $z_i$ 和质心 $\boldsymbol\mu_k$，K-means 的簇内平方和是

$$
\operatorname{WCSS}(K)
=\sum_{k=1}^{K}\sum_{i:z_i=k}
\|\mathbf x_i-\boldsymbol\mu_k\|_2^2.
$$

只要允许增加质心，旧解总可以被复制成新解，所以最优 WCSS 随 $K$ 单调不增；$K=N$ 时可降到 0。肘部法观察相对边际改善

$$
g_K=
\frac{\operatorname{WCSS}(K-1)-\operatorname{WCSS}(K)}
{\operatorname{WCSS}(K-1)}.
$$

它寻找“继续增加原型后收益明显变小”的位置，却没有统计显著性保证，也不规定怎样处理平滑曲线。更重要的是，WCSS 只能在相同样本、特征变换、权重、距离和优化质量下比较；把某个候选只跑一次、另一个跑 32 个起点，比较的可能是初始化运气。

下面为本章自包含地实现确定性 K-means++ 与多起点。它只是候选生成器，不是评价裁判。

```python
def assign_to_centers(x, centers):
    squared = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return squared.argmin(axis=1), squared.min(axis=1)

def run_kmeans(x, k, seed, restarts=12, max_iter=120):
    if not 1 <= k <= len(x):
        raise ValueError("K 必须位于 1 与样本数之间")
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(restarts):
        centers = [x[rng.integers(len(x))]]
        while len(centers) < k:
            current = np.asarray(centers)
            nearest = np.sum(
                (x[:, None, :] - current[None, :, :]) ** 2,
                axis=2,
            ).min(axis=1)
            total = nearest.sum()
            if total <= 0:
                raise ValueError("可区分样本少于 K，K-means++ 无法继续")
            centers.append(x[rng.choice(len(x), p=nearest / total)])
        centers = np.asarray(centers)

        for _ in range(max_iter):
            labels, nearest = assign_to_centers(x, centers)
            updated = centers.copy()
            for cluster in range(k):
                members = x[labels == cluster]
                updated[cluster] = (
                    members.mean(axis=0)
                    if len(members)
                    else x[nearest.argmax()]
                )
            if np.allclose(updated, centers, atol=1e-8):
                centers = updated
                break
            centers = updated

        labels, nearest = assign_to_centers(x, centers)
        candidate = (float(nearest.sum()), labels, centers)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best

KMEANS_FITS = {
    k: run_kmeans(X_TRAIN, k, seed=2700 + k)
    for k in range(2, 7)
}
WCSS = {k: fit[0] for k, fit in KMEANS_FITS.items()}

assert all(WCSS[k] < WCSS[k - 1] for k in range(3, 7))
print({k: round(value, 1) for k, value in WCSS.items()})
```

输出为：

```text
{2: 516.3, 3: 133.6, 4: 120.6, 5: 111.7, 6: 103.8}
```

从 $K=2$ 到 $K=3$ 的 WCSS 降幅约 74.1%，之后第一次只再下降约 9.7%。这支持把 $K=3$ 放进短名单，却不能单独签字：数据本来就是用三个紧凑云团合成的，欧氏指标与 K-means 共享同一偏好。

<MlQuiz
  kind="肘部边界"
  question="为什么 WCSS 最小不能作为选择 K 的规则？"
  :options="['WCSS 随 K 增大单调不增，K=N 时可为 0', 'WCSS 必为负数', 'WCSS 使用真实标签', 'K 不影响质心数']"
  :answer="0"
  explanation="增加中心扩大了可行解集合。必须再引入复杂度、稳定性、用途和运行约束。"
/>

## 6. Silhouette：把每个样本放在本簇与最近邻簇之间

对样本 $i$，定义：

$$
a(i)=\frac{1}{|C_i|-1}
\sum_{j\in C_i,j\ne i}d(i,j),
$$

$$
b(i)=\min_{C\ne C_i}
\frac{1}{|C|}\sum_{j\in C}d(i,j).
$$

$a(i)$ 是本簇凝聚代价，$b(i)$ 是它转投最近其他簇后的平均距离。轮廓系数为

$$
\boxed{
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}
}
\in[-1,1].
$$

- $s(i)\approx1$：本簇内近，最近其他簇远；
- $s(i)\approx0$：位于两个簇的几何边界；
- $s(i)<0$：按该距离看，它平均更接近另一个簇；
- 单样本簇没有可计算的 $a(i)$，本章按常见约定记为 0，并必须在报告里另列 singleton 数量。

若 $a=2,b=5$，则 $s=0.6$；若交换为 $a=5,b=2$，则 $s=-0.6$。分母取最大值正是为了把正负两侧都压到 $[-1,1]$。

```python
def silhouette_samples(x, labels):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("Silhouette 至少需要两个簇")
    distance_matrix = pairwise_euclidean(x)
    scores = np.zeros(len(x))

    for index in range(len(x)):
        own = labels == labels[index]
        if own.sum() == 1:
            scores[index] = 0.0
            continue
        a_i = distance_matrix[index, own].sum() / (own.sum() - 1)
        b_i = min(
            distance_matrix[index, labels == cluster].mean()
            for cluster in unique
            if cluster != labels[index]
        )
        scores[index] = (b_i - a_i) / max(a_i, b_i)
    return scores

SILHOUETTE = {
    k: silhouette_samples(X_TRAIN, fit[1])
    for k, fit in KMEANS_FITS.items()
}

assert np.allclose((5 - 2) / max(2, 5), 0.6)
assert np.allclose((2 - 5) / max(5, 2), -0.6)
assert all(np.all((-1 <= values) & (values <= 1)) for values in SILHOUETTE.values())
print({k: round(values.mean(), 3) for k, values in SILHOUETTE.items()})
```

输出为：

```text
{2: 0.666, 3: 0.736, 4: 0.489, 5: 0.264, 6: 0.263}
```

不要只保存总体均值。工业报告至少要列每簇中位数、低分位和负值率，因为一个巨大、清晰的簇可以掩盖另一个小而混乱的簇。

```python
labels_k3 = KMEANS_FITS[3][1]
silhouette_k3 = SILHOUETTE[3]
SILHOUETTE_BY_CLUSTER = {
    int(cluster): {
        "count": int((labels_k3 == cluster).sum()),
        "median": float(np.median(silhouette_k3[labels_k3 == cluster])),
        "p10": float(np.quantile(silhouette_k3[labels_k3 == cluster], 0.10)),
        "negative_rate": float((silhouette_k3[labels_k3 == cluster] < 0).mean()),
    }
    for cluster in np.unique(labels_k3)
}

assert sum(row["count"] for row in SILHOUETTE_BY_CLUSTER.values()) == len(X_TRAIN)
assert max(row["negative_rate"] for row in SILHOUETTE_BY_CLUSTER.values()) == 0
print({k: (v["count"], round(v["p10"], 3)) for k, v in SILHOUETTE_BY_CLUSTER.items()})
```

Silhouette 仍不是算法无关裁判。它用相同距离奖励“簇内紧、簇间远”，因此偏好凸而分离的结构；同心环或月牙的物理分区可能具有较低轮廓。用欧氏 K-means 造簇，再用欧氏 Silhouette 验收，会形成同源偏好闭环。

<MlQuiz
  kind="样本级解释"
  question="一个样本的 Silhouette 为负，最精确的解释是什么？"
  :options="['按当前距离，它到最近其他簇的平均距离小于到本簇的平均距离', '它必定是传感器故障', '整个候选 K 必须作废', '它一定是单样本簇']"
  :answer="0"
  explanation="负值是几何边界信号，不直接给出物理原因，也不能仅凭一个样本否决全部分区。"
/>

## 7. CH 与 DB：同样的“紧与远”，不同的聚合方式

Calinski–Harabasz（CH）把簇间离差与簇内离差做自由度修正：

$$
\operatorname{CH}
=\frac{B_K/(K-1)}{W_K/(N-K)},
$$

其中

$$
B_K=\sum_{k=1}^{K}n_k\|\boldsymbol\mu_k-\bar{\mathbf x}\|^2,
\qquad
W_K=\sum_{k=1}^{K}\sum_{i:z_i=k}
\|\mathbf x_i-\boldsymbol\mu_k\|^2.
$$

CH 越大通常越好。Davies–Bouldin（DB）先定义簇内平均半径

$$
S_k=\frac1{n_k}\sum_{i:z_i=k}
\|\mathbf x_i-\boldsymbol\mu_k\|,
$$

再让每个簇寻找最容易混淆的另一个簇：

$$
\operatorname{DB}
=\frac1K\sum_{k=1}^{K}
\max_{\ell\ne k}
\frac{S_k+S_\ell}{\|\boldsymbol\mu_k-\boldsymbol\mu_\ell\|}.
$$

DB 越小通常越好。二者都依赖中心与欧氏散度，对非凸、不同密度或严重不平衡结构仍可能误导。

```python
def calinski_harabasz(x, labels):
    unique = np.unique(labels)
    if not 1 < len(unique) < len(x):
        raise ValueError("CH 需要 1 < K < N")
    overall = x.mean(axis=0)
    between = sum(
        (labels == cluster).sum()
        * np.sum((x[labels == cluster].mean(axis=0) - overall) ** 2)
        for cluster in unique
    )
    within = sum(
        np.sum((x[labels == cluster] - x[labels == cluster].mean(axis=0)) ** 2)
        for cluster in unique
    )
    return between / (len(unique) - 1) / (within / (len(x) - len(unique)))

def davies_bouldin(x, labels):
    unique = np.unique(labels)
    centers = np.asarray([x[labels == cluster].mean(axis=0) for cluster in unique])
    scatter = np.asarray([
        np.linalg.norm(x[labels == cluster] - centers[index], axis=1).mean()
        for index, cluster in enumerate(unique)
    ])
    center_distance = pairwise_euclidean(centers)
    ratios = np.divide(
        scatter[:, None] + scatter[None, :],
        center_distance,
        out=np.zeros_like(center_distance),
        where=center_distance > 0,
    )
    return np.max(ratios, axis=1).mean()

CH = {k: calinski_harabasz(X_TRAIN, fit[1]) for k, fit in KMEANS_FITS.items()}
DB = {k: davies_bouldin(X_TRAIN, fit[1]) for k, fit in KMEANS_FITS.items()}

assert max(CH, key=CH.get) == 3
assert min(DB, key=DB.get) == 3
print("CH：", {k: round(v, 1) for k, v in CH.items()})
print("DB：", {k: round(v, 3) for k, v in DB.items()})
```

<MlQuiz
  kind="指标方向"
  question="在其适用几何下，CH 与 DB 的常见优化方向分别是什么？"
  :options="['CH 越大、DB 越小', '两者都越小', '两者都越大', 'CH 只看标签，DB 只看时间']"
  :answer="0"
  explanation="CH 奖励自由度修正后的簇间/簇内离差比；DB 惩罚每簇与最相似邻簇的散度—间距比。"
/>

## 8. 比较两个分区：先丢掉任意的簇编号

聚类标签的数值没有顺序。把 `[0,0,1,1]` 改名为 `[7,7,3,3]`，分区完全没变，直接标签准确率却失去意义。

### 8.1 Rand 与 Adjusted Rand

对所有 $\binom N2$ 个样本对，只问它们在两个分区中是否同组：

- $a$：两个分区都判为同组；
- $b$：两个分区都判为异组；
- $c,d$：一个同组、另一个异组。

Rand Index 为

$$
\operatorname{RI}=\frac{a+b}{a+b+c+d}.
$$

但当簇很多时，大量样本对天然都是“异组—异组”，RI 会偏高。Adjusted Rand Index（ARI）用超几何随机分区的期望校正共同配对数。若列联表元素为 $n_{uv}$、行和为 $a_u$、列和为 $b_v$：

$$
\operatorname{ARI}=
\frac{
\sum_{uv}\binom{n_{uv}}2
-\frac{\sum_u\binom{a_u}2\sum_v\binom{b_v}2}{\binom N2}
}{
\frac12\left[
\sum_u\binom{a_u}2+\sum_v\binom{b_v}2
\right]
-\frac{\sum_u\binom{a_u}2\sum_v\binom{b_v}2}{\binom N2}
}.
$$

完全一致时为 1，随机基线附近为 0，也可能略小于 0。

```python
def contingency(left, right):
    _, left_inverse = np.unique(left, return_inverse=True)
    _, right_inverse = np.unique(right, return_inverse=True)
    table = np.zeros((left_inverse.max() + 1, right_inverse.max() + 1), dtype=int)
    np.add.at(table, (left_inverse, right_inverse), 1)
    return table

def adjusted_rand(left, right):
    table = contingency(left, right)
    choose2 = lambda values: np.sum(values * (values - 1) / 2)
    index = choose2(table)
    row_pairs = choose2(table.sum(axis=1))
    column_pairs = choose2(table.sum(axis=0))
    total_pairs = len(left) * (len(left) - 1) / 2
    expected = row_pairs * column_pairs / total_pairs
    maximum = 0.5 * (row_pairs + column_pairs)
    if np.isclose(maximum, expected):
        return 1.0 if np.array_equal(left, right) else 0.0
    return float((index - expected) / (maximum - expected))

partition = np.array([0, 0, 1, 1, 2, 2])
renamed = np.array([7, 7, 3, 3, 9, 9])
assert adjusted_rand(partition, renamed) == 1.0
assert np.mean(partition == renamed) == 0
print("编号置换后的直接相等率 / ARI：", 0.0, 1.0)
```

ARI 不只用于“聚类对真实标签”。它更重要的工业用途是比较重采样、不同 seed、不同时间窗和不同算法得到的两个无监督分区。

::: ml-stop 停下来推一推 2
分区 $U=[0,0,1,1]$ 与 $V=[1,1,0,0]$ 的直接标签相等率为多少？列出六个样本对的同组/异组关系，解释为什么 ARI 应为 1。
:::

### 8.2 互信息与 NMI

两个分区 $U,V$ 的互信息为

$$
I(U;V)=\sum_{u,v}p_{uv}
\log\frac{p_{uv}}{p_u p_v}.
$$

它衡量知道一个分区能减少多少另一个分区的不确定性。归一化互信息的一种约定是

$$
\operatorname{NMI}(U,V)
=\frac{2I(U;V)}{H(U)+H(V)}.
$$

不同软件还可能使用几何平均、最大熵或 adjusted mutual information；报告时必须写明定义。未调整 NMI 对簇数仍有偏好，不能因为范围在 $[0,1]$ 就当作“无偏准确率”。

```python
def normalized_mutual_information(left, right):
    table = contingency(left, right).astype(float)
    probability = table / table.sum()
    row = probability.sum(axis=1)
    column = probability.sum(axis=0)
    positive = probability > 0
    denominator = row[:, None] * column[None, :]
    mutual_information = np.sum(
        probability[positive]
        * np.log(probability[positive] / denominator[positive])
    )
    entropy_left = -np.sum(row[row > 0] * np.log(row[row > 0]))
    entropy_right = -np.sum(column[column > 0] * np.log(column[column > 0]))
    return float(2 * mutual_information / (entropy_left + entropy_right))

assert np.isclose(normalized_mutual_information(partition, renamed), 1)
assert np.isclose(
    normalized_mutual_information(partition, renamed),
    normalized_mutual_information(renamed, partition),
)
```

### 8.3 Purity 为什么会奖励碎片化

Purity 为每个簇选择其中最多的参考类别，再把命中数相加：

$$
\operatorname{purity}(\Omega,\mathcal C)
=\frac1N\sum_k\max_j|\omega_k\cap c_j|.
$$

它直观，却在每个样本单独成簇时达到 1。因此 purity 只能与簇数、簇大小分布和其他指标一起报告；它不适合独自选择 $K$。

<MlQuiz
  kind="外部指标"
  question="把 N 个样本各自放进一个单样本簇，Purity 通常会怎样？"
  :options="['达到 1，因此它会奖励过度碎片化', '必为 0', '等于 ARI', '无法计算任何参考类别']"
  :answer="0"
  explanation="每个 singleton 都能选择自己的参考类别作为多数类。高 purity 不等于得到有用的聚类数。"
/>

## 9. 稳定性：换一批 campaign，结构还能不能重现

没有外部标签时，可以问一个更弱、却很有价值的问题：训练数据轻微变化后，分区是否大致一致？

本案例不能逐窗口普通 bootstrap，因为相邻窗口共享 campaign 偏移。正确的重采样单元是 campaign。为了避免 bootstrap 重复项和缺席样本难以一一对应，本章采用固定评价框架：

1. 在完整训练集上拟合参考候选；
2. 无放回抽取 80% 的训练 campaign；
3. 在子样本上重新拟合相同 $K$；
4. 用新质心预测**同一个完整训练输入框架**；
5. 用 ARI 比较参考分区与新分区；
6. 重复 40 次，报告中位数、P10、最差值和逐样本共分配不确定性。

这种“固定框架预测稳定性”适用于有 `predict` 接口的原型模型。对于 DBSCAN 等无法自然预测新点的算法，可以在共同出现样本上比较，或构造共识矩阵；不能为了凑 ARI 强行按数组位置对齐 bootstrap 副本。

```python
def campaign_stability(x, groups, k, reference_centers, seed=901, repeats=40):
    rng = np.random.default_rng(seed + k)
    reference_labels, _ = assign_to_centers(x, reference_centers)
    group_values = np.unique(groups)
    scores = []

    for repeat in range(repeats):
        picked = rng.choice(
            group_values,
            size=int(0.8 * len(group_values)),
            replace=False,
        )
        sample_mask = np.isin(groups, picked)
        _, _, centers = run_kmeans(
            x[sample_mask],
            k,
            seed + 100 * k + repeat,
            restarts=6,
        )
        candidate_labels, _ = assign_to_centers(x, centers)
        scores.append(adjusted_rand(reference_labels, candidate_labels))
    return np.asarray(scores)

STABILITY = {
    k: campaign_stability(X_TRAIN, GROUP_TRAIN, k, fit[2])
    for k, fit in KMEANS_FITS.items()
}

assert all(values.shape == (40,) for values in STABILITY.values())
print({
    k: (round(np.median(v), 3), round(np.quantile(v, 0.10), 3))
    for k, v in STABILITY.items()
})
```

输出为：

```text
{2: (1.0, 1.0), 3: (1.0, 1.0), 4: (0.879, 0.817), 5: (0.717, 0.599), 6: (0.696, 0.596)}
```

$K=2$ 很稳定，却把两个生成模式合并了；这说明稳定性只排除“结构主要由抽样偶然性造成”的部分风险，不证明粒度足够，也不证明分组有用。反过来，真实存在但样本很少的工况也可能不稳定。工程上应把不稳定当作调查信号，而不是把稳定当作真值证书。

<MlQuiz
  kind="稳定性边界"
  question="一个算法每次都把全部样本放进同一簇，因此重采样 ARI 恒为 1。可以据此签字吗？"
  :options="['不可以；稳定只说明可重复，退化分区仍可能毫无用途', '可以，稳定性等于业务价值', '可以，因为 K=1 一定最优', '不可以，因为 ARI 不能比较两个分区']"
  :answer="0"
  explanation="稳定性是必要证据之一，不评价粒度、语义、运行容量或下游收益。"
/>

### 9.1 共识矩阵定位不稳定样本

单个 ARI 把整次重训压成一个数。更细的诊断是共识矩阵

$$
C_{ij}=\frac1B\sum_{b=1}^{B}
\mathbb I\{z_i^{(b)}=z_j^{(b)}\},
$$

它记录两个样本在多少次重训中同簇。清晰块对角意味着稳定群组；大量接近 0.5 的格子说明边界或多解。$N$ 很大时不应保存完整 $N^2$ 矩阵，可以只对代表样本、边界样本或分层抽样子集计算，并保存不稳定样本清单。

## 10. 模型选择不是把分数相加，而是按合同逐项过门

### 10.1 候选证据表

本章预先冻结四个硬门：

- 平均 Silhouette 至少 0.45；
- campaign 重采样 ARI 的 P10 至少 0.80；
- 最小簇占训练集至少 15%；
- 人工最多维护三个复核队列。

在全部硬门通过的候选中，再取平均 Silhouette 更高者。这个规则不是普遍定理，而是本案例的发布合同。换成异常发现任务时，15% 最小簇门槛就可能完全不合理。

```python
CANDIDATES = []
previous_wcss = None
for k in range(2, 7):
    objective, labels, centers = KMEANS_FITS[k]
    counts = np.bincount(labels, minlength=k)
    row = {
        "k": k,
        "wcss": objective,
        "relative_gain": (
            np.nan
            if previous_wcss is None
            else (previous_wcss - objective) / previous_wcss
        ),
        "silhouette": float(SILHOUETTE[k].mean()),
        "ch": float(CH[k]),
        "db": float(DB[k]),
        "stability_median": float(np.median(STABILITY[k])),
        "stability_p10": float(np.quantile(STABILITY[k], 0.10)),
        "min_share": float(counts.min() / len(labels)),
    }
    row["passes"] = (
        row["silhouette"] >= 0.45
        and row["stability_p10"] >= 0.80
        and row["min_share"] >= 0.15
        and k <= 3
    )
    CANDIDATES.append(row)
    previous_wcss = objective

eligible = [row for row in CANDIDATES if row["passes"]]
SELECTED = max(eligible, key=lambda row: row["silhouette"])

assert [row["k"] for row in eligible] == [2, 3]
assert SELECTED["k"] == 3
for row in CANDIDATES:
    print(
        row["k"],
        round(row["silhouette"], 3),
        round(row["stability_p10"], 3),
        round(row["min_share"], 3),
        row["passes"],
    )
```

输出为：

```text
2 0.666 1.0 0.375 True
3 0.736 1.0 0.167 True
4 0.489 0.817 0.167 False
5 0.264 0.599 0.167 False
6 0.263 0.596 0.143 False
```

### 10.2 候选冻结后，才打开少量外部审计

```python
selected_centers = KMEANS_FITS[SELECTED["k"]][2]
eval_labels, _ = assign_to_centers(X_EVAL, selected_centers)
FINAL_AUDIT_ARI = adjusted_rand(
    AUDIT_EVAL_ALL[AUDIT_REVIEWED],
    eval_labels[AUDIT_REVIEWED],
)

assert np.isclose(FINAL_AUDIT_ARI, 1.0)
print("冻结 K=3 后的 36 条独立审计 ARI：", round(FINAL_AUDIT_ARI, 3))
```

这里 ARI 为 1 是合成器刻意制造的清晰教学结果，不是工业项目应期待的常态。真实审计还要给出标签来源、复核者一致性、盲法、缺失标签机制和置信区间。36 条记录更不支持把 1.000 写成对未来总体的精确保证。

<ClusterEvidenceLedgerLab />

<MlQuiz
  kind="决策规则"
  question="为什么本章用硬门禁形成可行集合，而不是把 Silhouette、稳定性和队列数归一化后加权求和？"
  :options="['硬门表达不可补偿约束，避免极高几何分数抵消不可运行的队列负荷', '因为加法在数学上不存在', '因为所有指标量纲相同', '因为门禁能证明存在真实类别']"
  :answer="0"
  explanation="人员容量、最小簇负荷等是不可补偿的运行约束；权重和归一化还会引入隐藏的任意性。"
/>

## 11. 非凸反例：高 Silhouette 可以偏离目标结构

压缩空气站的某类控制器存在升载与卸载滞回。把两个过程量画在平面上，两种控制模式各沿一条弯曲轨迹运行，形成互相环抱的“双月牙”。

欧氏 K-means 必须用一条 Voronoi 直线切分两个质心，因此更容易按左/右空间切开。若工程目标是沿轨迹连通性分组，原空间的球形几何就错了。

```python
def make_hysteresis_moons(seed=27, n=180, noise=0.055):
    rng = np.random.default_rng(seed)
    half = n // 2
    theta_top = np.linspace(0, np.pi, half) + rng.normal(0, 0.025, half)
    theta_bottom = np.linspace(0, np.pi, n - half) + rng.normal(0, 0.025, n - half)
    top = np.column_stack([np.cos(theta_top), np.sin(theta_top)])
    bottom = np.column_stack([
        1 - np.cos(theta_bottom),
        0.48 - np.sin(theta_bottom),
    ])
    top += rng.normal(0, noise, top.shape)
    bottom += rng.normal(0, noise, bottom.shape)
    return (
        np.vstack([top, bottom]),
        np.concatenate([
            np.zeros(half, dtype=int),
            np.ones(n - half, dtype=int),
        ]),
    )

X_MOON, MOON_AUDIT = make_hysteresis_moons()
_, KMEANS_MOON_LABELS, _ = run_kmeans(X_MOON, 2, seed=27, restarts=30)
KMEANS_MOON_ARI = adjusted_rand(MOON_AUDIT, KMEANS_MOON_LABELS)
KMEANS_MOON_SIL = silhouette_samples(X_MOON, KMEANS_MOON_LABELS).mean()

assert KMEANS_MOON_ARI < 0.30
print("原空间 K-means：ARI / Silhouette =", round(KMEANS_MOON_ARI, 3), round(KMEANS_MOON_SIL, 3))
```

输出为：

```text
原空间 K-means：ARI / Silhouette = 0.246 0.484
```

0.484 的轮廓并不难看，但它奖励的是左右两个凸块，而不是两条过程轨迹。这不是 Silhouette 算错，而是评价目标与工程目标不一致。

## 12. 从相似图到图拉普拉斯

### 12.1 把样本改写成图

构造无向加权图 $G=(V,E)$：每个样本是顶点，相似样本之间连边。常见 Gaussian 权重为

$$
w_{ij}=\exp\left(
-\frac{\|\mathbf x_i-\mathbf x_j\|^2}{2\sigma^2}
\right),
$$

但通常只保留 $k$ 近邻或半径邻域，避免远距离弱边把图变成稠密完全图。定义权重矩阵 $\mathbf W$、度矩阵

$$
D_{ii}=\sum_j w_{ij},
$$

以及非归一化图拉普拉斯

$$
\boxed{\mathbf L=\mathbf D-\mathbf W}.
$$

对任意顶点信号 $\mathbf f$：

$$
\mathbf f^\top\mathbf L\mathbf f
=\frac12\sum_{i,j}w_{ij}(f_i-f_j)^2\ge0.
$$

所以 $\mathbf L$ 半正定。这条式子的工程直觉是：一条权重大、代表强相似性的边，如果两端信号差异很大，就要支付高能量。低特征值对应在强边上变化缓慢的“图低频”方向。

```python
W_TOY = np.array([
    [0.0, 1.0, 0.2],
    [1.0, 0.0, 0.4],
    [0.2, 0.4, 0.0],
])
D_TOY = np.diag(W_TOY.sum(axis=1))
L_TOY = D_TOY - W_TOY
F_TOY = np.array([0.2, 0.4, 1.3])

quadratic = F_TOY @ L_TOY @ F_TOY
edge_energy = 0.5 * np.sum(W_TOY * (F_TOY[:, None] - F_TOY[None, :]) ** 2)
eigenvalues = np.linalg.eigvalsh(L_TOY)

assert np.allclose(quadratic, edge_energy)
assert eigenvalues.min() >= -1e-12
assert np.allclose(L_TOY @ np.ones(3), 0)
print("拉普拉斯能量 / 最小特征值：", round(quadratic, 3), round(eigenvalues[0], 12))
```

### 12.2 零特征值数等于连通分量数

因为 $\mathbf L\mathbf 1=0$，常数向量总是零特征向量。若图有 $c$ 个连通分量，每个分量的指示向量都在零空间中，因此零特征值的重数是 $c$。这把一个离散连通问题转成了线性代数问题。

图几乎断开时，第二小特征值 $\lambda_2$ 很小，其特征向量在两个稠密区域内近似常数、在稀疏桥边处改变。这就是 Fiedler 向量的切割直觉。

<MlQuiz
  kind="拉普拉斯直觉"
  question="一个无向图有三个完全断开的连通分量，非归一化拉普拉斯 L 有多少个零特征值？"
  :options="['3 个', '1 个', '0 个', '等于边数']"
  :answer="0"
  explanation="每个连通分量的指示向量都是独立零特征向量，零空间维数等于连通分量数。"
/>

## 13. 从 RatioCut、Ncut 到归一化谱嵌入

### 13.1 只最小化 cut 会制造孤立点

把顶点分成 $A$ 与 $\bar A$，跨集合边权为

$$
\operatorname{cut}(A,\bar A)
=\sum_{i\in A,j\in\bar A}w_{ij}.
$$

只最小化 cut 容易切掉一个度很小的孤立点。RatioCut 用顶点数平衡：

$$
\operatorname{RatioCut}(A,\bar A)
=\operatorname{cut}(A,\bar A)
\left(\frac1{|A|}+\frac1{|\bar A|}\right).
$$

Normalized Cut（Ncut）用总体连接强度

$$
\operatorname{vol}(A)=\sum_{i\in A}D_{ii}
$$

做平衡：

$$
\operatorname{Ncut}(A,\bar A)
=\operatorname{cut}(A,\bar A)
\left(\frac1{\operatorname{vol}(A)}
+\frac1{\operatorname{vol}(\bar A)}\right).
$$

离散最优切割通常难解。谱方法放松离散指示向量约束，求连续 Rayleigh quotient 的低特征向量，再用阈值或 K-means 离散化。

### 13.2 为什么要归一化

归一化拉普拉斯为

$$
\mathbf L_{\text{sym}}
=\mathbf I-\mathbf D^{-1/2}\mathbf W\mathbf D^{-1/2}.
$$

它降低高连接度区域对未归一化能量的支配。对 $K$ 个簇的常见算法：

1. 构造并对称化稀疏相似图；
2. 计算 $\mathbf L_{\text{sym}}$ 最小的 $K$ 个特征向量；
3. 把它们按列组成 $\mathbf U\in\mathbb R^{N\times K}$；
4. 对每一行做 $\ell_2$ 归一化；
5. 在这 $N$ 个谱坐标上运行 K-means；
6. 把谱空间标签映回原样本。

第一特征向量主要编码度结构；后续低频方向展开图上的大尺度连通关系。行归一化能减少度大小对方向的影响，但不是所有谱聚类变体都使用同一规范。报告必须写清图构造、拉普拉斯版本、特征向量数量、行归一化、离散化方法和随机种子。

::: ml-stop 停下来推一推 3
若相似图已经恰好由两个互不连通的子图组成，写出两个分量指示向量。它们经过怎样的线性组合，可以得到一个在两侧取相反符号的零能量切割方向？
:::

<MlQuiz
  kind="谱算法顺序"
  question="标准归一化谱聚类在得到最小 K 个特征向量后，通常对什么对象运行 K-means？"
  :options="['特征向量矩阵逐行归一化后的 N 个谱坐标', '拉普拉斯的 K 个特征值', '原始距离矩阵的列名', '真实类别编号']"
  :answer="0"
  explanation="每个样本对应 U 的一行；谱嵌入先改变表示，再在这些行向量上做普通聚类。"
/>

## 14. 从零实现谱聚类，并让指标公开意见不一

下面用对称 $k$ 近邻图、Gaussian 权重、$\mathbf L_{\text{sym}}$、最小两个特征向量、逐行归一化和 K-means 实现完整流程。

```python
def spectral_clustering(x, k, neighbors=9, sigma=0.25, seed=2727):
    if not 1 <= neighbors < len(x):
        raise ValueError("neighbors 必须位于 1 与 N-1 之间")
    squared = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    nearest = np.argsort(squared, axis=1)[:, 1:neighbors + 1]
    weights = np.zeros_like(squared)

    for row in range(len(x)):
        columns = nearest[row]
        weights[row, columns] = np.exp(
            -squared[row, columns] / (2 * sigma ** 2)
        )
    weights = np.maximum(weights, weights.T)
    degree = weights.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("相似图存在零度顶点，请检查图构造")

    inverse_sqrt_degree = np.diag(1 / np.sqrt(degree))
    laplacian = np.eye(len(x)) - inverse_sqrt_degree @ weights @ inverse_sqrt_degree
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    embedding = eigenvectors[:, :k]
    row_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(row_norm, 1e-12)
    _, labels, centers = run_kmeans(
        embedding,
        k,
        seed=seed,
        restarts=20,
    )
    return {
        "labels": labels,
        "centers": centers,
        "embedding": embedding,
        "eigenvalues": eigenvalues,
        "weights": weights,
    }

SPECTRAL_MOON = spectral_clustering(X_MOON, 2)
SPECTRAL_MOON_ARI = adjusted_rand(MOON_AUDIT, SPECTRAL_MOON["labels"])
SPECTRAL_MOON_SIL = silhouette_samples(X_MOON, SPECTRAL_MOON["labels"]).mean()

assert SPECTRAL_MOON_ARI > 0.95
assert SPECTRAL_MOON_SIL < KMEANS_MOON_SIL
print(
    "谱聚类：ARI / 原空间 Silhouette =",
    round(SPECTRAL_MOON_ARI, 3),
    round(SPECTRAL_MOON_SIL, 3),
)
print("最小四个特征值：", np.round(SPECTRAL_MOON["eigenvalues"][:4], 5))
```

输出为：

```text
谱聚类：ARI / 原空间 Silhouette = 0.978 0.338
最小四个特征值： [-0.       0.00131  0.00457  0.00797]
```

这个反例必须同时报告两种结论：

- 按合成器保存的轨迹模式，谱聚类 ARI 从 0.246 提升到 0.978；
- 按原空间欧氏凸性，谱聚类 Silhouette 反而从 0.484 降到 0.338。

二者都没有算错。它们回答不同问题：一个比较轨迹身份，另一个奖励原空间紧凑凸簇。若项目没有预先声明要保持图连通结构，不能看到 ARI 后再倒推“谱聚类必然更正确”。

<SpectralGraphCutLab />

<MlQuiz
  kind="指标冲突"
  question="双月牙案例中谱聚类 ARI 更高、原空间 Silhouette 更低，最合理的解释是什么？"
  :options="['两个指标编码不同目标：轨迹身份与原空间凸紧凑性发生冲突', '至少有一个实现必然出错', 'ARI 与 Silhouette 理论上必须同向', '只需把簇编号交换即可同时提高两者']"
  :answer="0"
  explanation="评价冲突正是模型假设冲突的可观测表现，应回到任务合同决定优先级。"
/>

## 15. 图构造是模型的一部分，不是预处理细节

谱聚类把原空间建模问题转移到了图：

- `neighbors` 太小，图可能碎成许多连通分量，零空间方向不唯一；
- `neighbors` 太大，会出现跨轨迹捷径，非凸结构被抹平；
- $\sigma$ 太小，除最近边外权重近乎 0，数值上接近断图；
- $\sigma$ 太大，保留边的权重趋同，局部距离差异消失；
- 量纲、缺失填补和无关特征会先污染近邻，再污染整个谱嵌入。

对近邻数做压力测试时，必须在同一冻结输入上报告连通分量、低端谱、分区稳定性和任务证据，而不是挑最好看的图。

```python
SPECTRAL_SENSITIVITY = []
for neighbors in [5, 7, 9, 12, 16, 24]:
    fit = spectral_clustering(
        X_MOON,
        2,
        neighbors=neighbors,
        sigma=0.25,
        seed=2727,
    )
    SPECTRAL_SENSITIVITY.append({
        "neighbors": neighbors,
        "ari": adjusted_rand(MOON_AUDIT, fit["labels"]),
        "lambda2": float(fit["eigenvalues"][1]),
    })

assert SPECTRAL_SENSITIVITY[0]["ari"] < 0.70
assert max(row["ari"] for row in SPECTRAL_SENSITIVITY[1:]) > 0.95
print([
    (row["neighbors"], round(row["ari"], 3), round(row["lambda2"], 5))
    for row in SPECTRAL_SENSITIVITY
])
```

输出为：

```text
[(5, 0.569, 0.0), (7, 0.978, 0.00086), (9, 0.978, 0.00131), (12, 0.978, 0.00173), (16, 0.978, 0.00259), (24, 0.978, 0.00675)]
```

$k=5$ 时第二特征值显示为 0，但这不能简单读成“恰好两个理想簇”：图可能有多于两个近断开方向，特征子空间基底可以旋转，离散化结果因此不稳。应检查接近零的特征值个数、连通分量和不同扰动下的子空间，而不是只看 $\lambda_2$。

<MlQuiz
  kind="图超参数"
  question="k 近邻图的 neighbors 设得过大，最典型的结构风险是什么？"
  :options="['跨越原本稀疏区域形成捷径，使不同非凸轨迹在图上强连接', '所有边权自动变成负数', '拉普拉斯不再是方阵', 'K-means 的质心数量自动减少']"
  :answer="0"
  explanation="图构造决定“局部相似”边界；过密图会抹平流形或轨迹之间的低密度间隔。"
/>

## 16. 三类方法各自在承诺什么

| 方法 | 核心承诺 | 强项 | 主要风险 | 新样本接口 |
|---|---|---|---|---|
| K-means | 到欧氏质心的平方误差小 | 快、可解释、原型可压缩 | 球形偏好、尺度与离群点敏感、局部最优 | 最近质心自然可用 |
| HAC | 在某个 linkage 下按距离逐级合并 | 可浏览多尺度，不必先冻结 K | 贪心不可撤销、$O(N^2)$ 存储、桥接/离群点敏感 | 通常没有唯一自然规则 |
| 谱聚类 | 相似图可被低代价切开 | 非凸连通结构、全局图信息 | 图构造敏感、特征分解昂贵、样本外扩展困难 | 需 Nyström、图扩展或重新拟合 |

选择算法时不要先问“哪一个分数高”，先问相似性是由质心、集合直径还是局部图连通表达。若未来数据必须低延迟路由，HAC 与标准谱聚类还要额外设计样本外分配；训练集上漂亮的树或嵌入不能自动变成部署接口。

<MlQuiz
  kind="方法选择"
  question="任务要求保留弯曲工况轨迹的局部连通性，而不是用一个均值代表每簇。哪种候选最值得优先压力测试？"
  :options="['基于局部相似图的谱聚类', '只增加 K-means 的 max_iter', '直接按簇编号排序', '把 WCSS 乘以 -1']"
  :answer="0"
  explanation="谱方法能把局部图连通信息转换为低频坐标；是否可用仍取决于图稳定性、规模和部署接口。"
/>

## 17. 工业验收：把“选了 K=3”变成可复核制品

### 17.1 发布包至少包含什么

| 制品 | 必须保存的内容 |
|---|---|
| 问题合同 | 样本单元、允许动作、禁止动作、队列容量、责任人 |
| 数据快照 | campaign 范围、去重、缺失、异常处理、特征顺序与单位 |
| 几何合同 | 标准化参数、业务权重、距离、算法、linkage 或图构造 |
| 候选账本 | 每个 $K$ 的全部起点、WCSS、Silhouette 分布、CH、DB、簇大小 |
| 稳定性账本 | 重采样单元、比例、重复数、seed、ARI 分布、共识诊断 |
| 外部审计 | 标签来源、盲法、覆盖率、复核者一致性、只读时间点 |
| 选择规则 | 硬门、可行集合、排序规则、未选候选及原因 |
| 运行接口 | 新样本分配、拒绝门、版本对齐、漂移与回滚策略 |

不能只保存一张肘部图。没有输入快照、尺度、seed、候选账本和审计时序，那张图无法重放，也不能证明结论没有事后选择。

### 17.2 簇编号必须版本化对齐

重训后的 `cluster_0` 可能对应旧版 `cluster_2`。对 K-means 可用冻结尺度下的质心匹配，再由工程人员确认；对谱聚类，特征向量的符号可翻转、重复特征值子空间还可旋转，更不能把谱坐标列或簇编号当永久身份。

对齐失败时应发布新版本并让下游显式迁移，不能偷偷复用旧语义。任何“高风险”“异常”“需维修”命名都要附上独立证据与审核责任，不得从质心排序自动生成。

<MlQuiz
  kind="版本治理"
  question="谱聚类重训后第二特征向量整体乘以 -1，是否代表过程结构反转？"
  :options="['不代表；特征向量符号本来就不唯一，分区结构可完全相同', '代表所有设备状态互换', '代表拉普拉斯出现负特征值', '必然导致 ARI 为 0']"
  :answer="0"
  explanation="若 v 是特征向量，-v 也是。重复特征值时整个特征子空间的基底还可能旋转。"
/>

### 17.3 监控分三层

上线后至少分开监控：

1. **输入层**：缺失、单位、范围、campaign 构成、特征相关与近邻距离漂移；
2. **分区层**：簇占比、极小簇、边界率、质心残差、共识不确定性、跨版本匹配；
3. **用途层**：各队列实际复核量、命中类型、人工退回、未覆盖新模式和处理时延。

若队列占比漂移，可能是真实工况变化，也可能是上游单位或控制版本变化。监控只能触发调查；在原因确认前，不应自动重训并覆盖线上语义。

## 18. 失效排查：先定位哪一层假设破了

| 观察 | 优先检查 | 不应立刻做什么 |
|---|---|---|
| WCSS 无明显肘部 | 是否存在连续谱、尺度错误、K 范围过窄 | 不要肉眼硬圈一个 K |
| 平均 Silhouette 高但小簇大量负值 | 分簇分布、singleton、密度和离群点 | 不要只汇报平均数 |
| 不同 seed 分区变化大 | 局部盆地、重叠结构、初始化、K 是否过大 | 不要只保留最好 seed |
| campaign 重采样不稳 | 有效样本量、时间漂移、小簇支持度 | 不要改成逐窗口 bootstrap 粉饰结果 |
| 外部 ARI 低 | 聚类目标与标签任务是否一致、标签噪声与覆盖 | 不要先宣判算法或标签谁错 |
| 谱聚类对近邻数敏感 | 图连通分量、跨模式捷径、量纲和无关特征 | 不要只截取最好看的参数 |
| 树出现长链 | single-link bridge、离群点、距离集中 | 不要把链条自动命名为演化路径 |
| 新样本无法自然分配 | 算法是否有样本外接口、是否需要拒绝机制 | 不要用最近训练样本冒充既定协议 |

<MlQuiz
  kind="重采样审计"
  question="逐小时窗口 bootstrap 显示很稳定，但按完整 campaign 重采样明显不稳。应信哪一个更接近未来 campaign 风险？"
  :options="['按 campaign 的结果，因为它尊重相关结构与实际泛化单元', '逐窗口结果，因为样本数看起来更多', '两者取最大值', '删掉不稳定 campaign']"
  :answer="0"
  explanation="相关窗口不能冒充独立样本。重采样单元应匹配部署时真正会变化和到来的群组。"
/>

## 数值验证：可执行回归测试与成熟库对照

### 19.1 本章关键不变量

```python
# 距离矩阵必须对称、对角为 0。
assert np.allclose(HAC_D, HAC_D.T)
assert np.allclose(np.diag(HAC_D), 0)

# 簇编号置换不能改变分区比较。
for k, fit in KMEANS_FITS.items():
    labels = fit[1]
    renamed = 100 + 7 * labels
    assert np.isclose(adjusted_rand(labels, renamed), 1)

# 选择规则在冻结输入与 seed 下必须可重放。
repeat_fit = run_kmeans(X_TRAIN, 3, seed=2703)
assert np.isclose(repeat_fit[0], KMEANS_FITS[3][0])
assert np.isclose(adjusted_rand(repeat_fit[1], KMEANS_FITS[3][1]), 1)

# 图、拉普拉斯和谱嵌入必须满足基本数值合同。
moon_weights = SPECTRAL_MOON["weights"]
assert np.allclose(moon_weights, moon_weights.T)
assert np.all(moon_weights >= 0)
assert SPECTRAL_MOON["embedding"].shape == (180, 2)
assert np.allclose(
    np.linalg.norm(SPECTRAL_MOON["embedding"], axis=1),
    1,
)

print("距离、置换、重放、图对称与谱嵌入门禁：全部通过")
```

### 19.2 scikit-learn 只作可选交叉核对

仓库不为教材新增该依赖，因此下面固定跳过。成熟库能提供高效 HAC、指标和稀疏谱求解，却不知道 campaign 是否泄漏、审计标签何时打开、三个队列是否可运行。

```python
# 不执行：可选环境对照，需要 scikit-learn。
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

hac = AgglomerativeClustering(n_clusters=3, linkage="average").fit(X_TRAIN)
spectral = SpectralClustering(
    n_clusters=2,
    affinity="nearest_neighbors",
    n_neighbors=9,
    assign_labels="kmeans",
    random_state=2727,
).fit(X_MOON)

print(silhouette_score(X_TRAIN, hac.labels_))
print(adjusted_rand_score(MOON_AUDIT, spectral.labels_))
```

## 复杂度与工程边界

| 操作 | 朴素时间 | 主要内存 | 规模化方向 |
|---|---:|---:|---|
| 全成对距离 | $O(N^2D)$ | $O(N^2)$ | 分块、近似近邻、代表样本 |
| 朴素 HAC | 最坏 $O(N^3)$ | $O(N^2)$ | Lance–Williams、专用 linkage 算法 |
| Silhouette | $O(N^2D)$ | 可达 $O(N^2)$ | 分层抽样、分块精确计算 |
| $B$ 次 K-means 稳定性 | $O(BINKD)$ | 取决于批大小 | 并行重放、固定评价子集 |
| 稠密谱分解 | $O(N^3)$ | $O(N^2)$ | 稀疏图、Lanczos、Nyström |

抽样近似必须保存抽样设计和误差检查。不能因为全量 Silhouette 太贵，就只挑最清晰的样本计算；那会系统性抬高分数。

## 21. 章节练习

### 基础推导

1. 对 $A=\{0,2\},B=\{5,9\}$ 手算 single、complete、average linkage，并解释哪一个最受点 9 影响。
2. 证明 $\mathbf f^\top(\mathbf D-\mathbf W)\mathbf f=\frac12\sum_{ij}w_{ij}(f_i-f_j)^2$。
3. 从 $\mathbf L\mathbf 1=0$ 出发，构造有两个连通分量时的两个独立零特征向量。
4. 给定一个 $3\times4$ 列联表，逐项计算 ARI 分子中的共同配对数和随机期望。

### 实现与反例

5. 为 `agglomerative_history` 实现按目标 $K$ 切树，验证 $K=N$ 与 $K=1$ 两个边界。
6. 构造一个桥接点，让 single linkage 产生 chaining；移走桥接点后比较三棵树。
7. 把 Silhouette 改成按簇等权平均，比较它与按样本平均对小簇的敏感性。
8. 为稳定性实验保存共识矩阵，找出共分配概率最接近 0.5 的十个窗口。
9. 在双月牙上扫描 `neighbors` 与 $\sigma$，同时画 ARI、连通分量、$\lambda_2$ 和跨模式边数，不准只选最好点。
10. 生成同心圆、不同密度高斯和连续梯度三种数据，分别说明“适合谱聚类”“适合概率模型”“根本不应硬分簇”的证据。

### 工业审计

11. 将三个复核队列的产能改成不均等上限，设计不可补偿门禁，不要把人员超载变成一个可被高 Silhouette 抵消的软分数。
12. 写一份外部标签使用协议，包含盲法、复核者分歧、打开次数、缺失机制和禁止事后调参条款。
13. 设计一次按季度回放的聚类漂移审计，区分标签置换、质心移动、簇占比变化和新工况拒绝。
14. 比较 K-means、HAC 与谱聚类的样本外分配方案，明确每种方案何时必须拒绝自动路由。

练习骨架：

```python
# TODO：实现按目标 K 切 Dendrogram，并返回与叶子顺序无关的规范化标签。
def cut_dendrogram(history, n_samples, k):
    raise NotImplementedError
```

## 来源与许可

- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 745–747 页，§21.1–§21.1.1.3：用于核对内部/外部评价边界、Purity、Rand、ARI、互信息与 NMI。
- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 748–752 页，§21.2–§21.2.3：用于核对 HAC、Dendrogram、single/complete/average linkage、复杂度与几何偏好。
- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 757–760 页，§21.3.7：用于核对 WCSS 随 $K$ 单调、Silhouette 定义、样本级图示含义与球形偏好边界。
- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 764–768 页，§21.5–§21.5.4：用于核对图相似度、图拉普拉斯、连通分量、归一化谱嵌入、非凸示例与随机游走联系。
- Jonathan Richard Shewchuk，UC Berkeley CS 189/289A 合订讲义，PDF 第 137–140 页：用于交叉核对 HAC、三种 linkage、Dendrogram 切割、横向叶序、chaining、outlier 与稳定性边界。
- Jonathan Richard Shewchuk，UC Berkeley CS 189/289A 合订讲义，PDF 第 175–184 页，Bonus Lecture C：用于交叉核对拉普拉斯二次型、Fiedler 向量、sweep cut、Ncut、多特征向量谱嵌入、逐行归一化与非凸比较。

本章压缩空气站复核队列验收案例的 66 个教学 campaign、528 条合成窗口、五项过程指数、三个隐藏生成模式、36 条冻结后审计记录、非凸滞回轨迹压力测试、四层硬门协议、全部 NumPy 代码、17 道即时测验和三项专用 SVG 网页实验均为本站独立创作。它们不代表真实企业、空压机性能、能耗基准、控制器逻辑、故障根因、复核产能、安全阈值或监管结论。没有复制、改编或再分发参考资料中的图形、数据、代码、案例和课堂测验。

完整来源登记见 [课程来源与许可](./sources.md) 。

<MlLessonProgress
  storage-key="27-clustering-evaluation"
  lesson-id="27"
  next-href="/courses/machine-learning/28-learning-theory"
  next-label="下一章：PAC 学习与 VC 维"
/>
