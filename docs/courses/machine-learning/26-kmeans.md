---
title: K-means 聚类
description: 从铸造车间压缩空气站工况归档案例推导 WCSS、Voronoi 分配、均值更新、Lloyd 交替最小化、K-means++、多起点、空簇处置与部署审计
course: machine-learning
edition: industrial
---

<MlLessonHero
  number="26"
  title="K-means 聚类"
  question="没有工况标签，怎样把 528 个运行窗口压缩成三个可复核原型，并证明每一次分流与料斗移动都在降低同一笔平方代价？"
  duration="建议 660–780 分钟"
  :outcomes="[
    '从簇内平方和推导最近质心分配、Voronoi 超平面与簇内均值更新，而不是背诵两步流程',
    '写出 Lloyd 交替最小化的不等式链，区分目标收敛、固定点、局部最优和全局最优',
    '从零实现确定性 K-means++、多起点、标签规范化、空簇门禁和完整迭代账本',
    '用分组留出、尺度合同、失效反例与部署监控判断 K-means 是否适合工业工况归档'
  ]"
/>

上一章用 PCA 把相关量测压缩成低维坐标，本章处理另一类无监督任务：把相似记录归到同一组，并用一个原型代表这一组。

K-means 的名字很短，工程合同却不短。它不是“让库自动找出三类故障”，而是在给定特征、距离和簇数 $K$ 后，寻找 $K$ 个欧氏原型，使样本到所属原型的平方距离和尽量小。这个目标同时决定了分配规则、质心更新、适用簇形状、离群点敏感性和量纲风险。

> 本章中的“工况 A/B/C”只是合成数据上的几何分组，不是已确认的设备状态或故障根因。聚类结果只用于抽样、归档和工程复核，不得直接启停压缩机、修改压力设定、旁路露点或温度保护、自动生成维修结论。

## 1. 问题合同：先定义压缩任务，再运行算法

给定无标签样本

$$
\mathcal D=\{\mathbf x_i\}_{i=1}^{N},
\qquad \mathbf x_i\in\mathbb R^D,
$$

以及预先确定的簇数 $K$，K-means 输出两类对象：

- 分配 $z_i\in\{0,\ldots,K-1\}$；
- 质心 $\boldsymbol\mu_k\in\mathbb R^D$。

上线前至少要冻结六项合同：

1. 一行样本代表什么时间窗口或资产单元；
2. 哪些连续特征进入欧氏空间；
3. 均值、尺度与业务权重从哪一组数据估计；
4. $K$ 由什么外部容量或模型选择证据确定；
5. 空簇、平票、缺失值和离群点怎样处置；
6. 聚类编号怎样版本化，允许哪些后续动作。

若这六项没有写清，`cluster_2` 只是一次运行中的数组索引。它既不天然代表“高耗能”，也不会跨重训保持同一含义。

<MlQuiz
  kind="对象辨析"
  question="一次 K-means 训练输出 cluster 0、1、2。哪项解释正确？"
  :options="['编号只标识本次解中的三个质心，除非另有对齐合同，否则没有固定物理语义', 'cluster 2 必然比 cluster 1 更危险', '编号天然按样本数量排序', '编号在所有随机种子和重训版本间保持不变']"
  :answer="0"
  explanation="目标函数对质心排列完全对称；交换两列质心及相应标签不会改变 WCSS。物理命名必须来自额外证据和版本化对齐。"
/>

## 贯穿案例：铸造车间压缩空气站工况归档

### 2.1 业务边界与五路量测

某铸造车间有多台并联空压机。工程组希望从历史小时窗口中建立三个“代表性工况篮”，每周从各篮抽样原始趋势供人工复核，检查群控策略是否出现新的运行组合。这里的 $K=3$ 来自三个固定复核队列的容量合同，不是从当前样本偷偷看出的真值。

每个窗口包含五项连续量测：

| 字段 | 单位 | 教学含义 | 为什么不能直接混入平方距离 |
|---|---:|---|---|
| `header_pressure_bar` | bar | 母管平均压力 | 数值约 7，量纲是压力 |
| `specific_power_kwh_per_knm3` | kWh/kNm³ | 比功率 | 数值约 0.1，却可能有强业务意义 |
| `loaded_unit_ratio` | 1 | 加载机组比例 | 数值位于 0–1 |
| `cooling_delta_c` | °C | 冷却水进出口温差 | 数值约 7–16 |
| `dewpoint_margin_c` | °C | 相对露点要求的裕量 | 数值约 7–13 |

允许动作只有：把窗口分配到原型篮、显示最近原型与残差、抽取代表样本和边界样本供复核。模型不得把统计簇自动命名为“泄漏”“结垢”或“故障”，也不得直接改变机组调度和保护逻辑。

### 2.2 合成数据：campaign 是最小隔离单元

教学数据含 66 个 production campaign，每个 campaign 8 个小时窗口，共 528 条记录。生成器内部有三个 `REGIME_AUDIT` 值，只用于课后核对几何分组；K-means 训练不读取它。验证与测试阶段另注入四条受控高残差记录，用来检查“离质心远”是否能进入人工复核，而不是给它们强行分配故障名。

```python
import itertools
import numpy as np

SEED = 2606
N_CAMPAIGNS = 66
WINDOWS_PER_CAMPAIGN = 8

FEATURE_NAMES = np.array([
    "header_pressure_bar",
    "specific_power_kwh_per_knm3",
    "loaded_unit_ratio",
    "cooling_delta_c",
    "dewpoint_margin_c",
])

REGIME_MEAN = np.array([
    [6.82, 0.102, 0.34, 7.4, 12.5],
    [7.16, 0.112, 0.63, 9.5, 10.0],
    [7.48, 0.137, 0.84, 15.2, 7.3],
])
REGIME_SD = np.array([
    [0.08, 0.0040, 0.055, 1.00, 1.15],
    [0.10, 0.0048, 0.060, 1.15, 1.05],
    [0.13, 0.0065, 0.055, 1.65, 1.35],
])
CAMPAIGN_LOADING = np.array([0.035, 0.0018, 0.018, 0.45, -0.38])

def make_compressor_dataset(seed=SEED):
    rng = np.random.default_rng(seed)
    rows, groups, audit, outlier = [], [], [], []
    for campaign in range(N_CAMPAIGNS):
        shift = rng.normal() * CAMPAIGN_LOADING
        mix = np.array([0.38, 0.40, 0.22])
        if campaign % 6 == 0:
            mix += np.array([0.08, -0.03, -0.05])
        mix = mix / mix.sum()
        for window in range(WINDOWS_PER_CAMPAIGN):
            regime = rng.choice(3, p=mix)
            common = rng.normal()
            noise = rng.normal(size=5)
            correlation = np.array([0.25, 0.35, 0.30, 0.45, -0.25]) * common
            row = (
                REGIME_MEAN[regime]
                + shift
                + REGIME_SD[regime] * (0.72 * noise + correlation)
            )
            is_outlier = (
                campaign >= 44
                and window == 0
                and campaign in {47, 53, 58, 63}
            )
            if is_outlier:
                row += np.array([0.05, 0.028, -0.02, 4.8, -4.0])
            rows.append(row)
            groups.append(campaign)
            audit.append(regime)
            outlier.append(is_outlier)
    return (
        np.asarray(rows),
        np.asarray(groups),
        np.asarray(audit),
        np.asarray(outlier),
    )

X_RAW, CAMPAIGN, REGIME_AUDIT, OUTLIER_AUDIT = make_compressor_dataset()

assert X_RAW.shape == (528, 5)
assert CAMPAIGN.shape == REGIME_AUDIT.shape == OUTLIER_AUDIT.shape == (528,)
assert np.isfinite(X_RAW).all()
print("记录 / 特征 / 三种审计状态：", X_RAW.shape, np.bincount(REGIME_AUDIT))
```

输出为：

```text
记录 / 特征 / 三种审计状态： (528, 5) [203 217 108]
```

训练、验证、测试必须按 campaign 切分。前 44 个 campaign 拟合尺度和质心，中间 11 个用于冻结运行合同，最后 11 个只做一次最终审计。

```python
train_mask = CAMPAIGN < 44
val_mask = (CAMPAIGN >= 44) & (CAMPAIGN < 55)
test_mask = CAMPAIGN >= 55

scale_mean = X_RAW[train_mask].mean(axis=0)
scale_std = X_RAW[train_mask].std(axis=0, ddof=0)
if np.any(scale_std <= 0):
    raise ValueError("训练集存在零方差特征，不能执行标准化")

X = (X_RAW - scale_mean) / scale_std
X_TRAIN, X_VAL, X_TEST = X[train_mask], X[val_mask], X[test_mask]
AUDIT_TRAIN = REGIME_AUDIT[train_mask]
AUDIT_VAL, AUDIT_TEST = REGIME_AUDIT[val_mask], REGIME_AUDIT[test_mask]
OUTLIER_VAL, OUTLIER_TEST = OUTLIER_AUDIT[val_mask], OUTLIER_AUDIT[test_mask]

assert (len(X_TRAIN), len(X_VAL), len(X_TEST)) == (352, 88, 88)
assert set(CAMPAIGN[train_mask]).isdisjoint(CAMPAIGN[val_mask])
assert set(CAMPAIGN[train_mask]).isdisjoint(CAMPAIGN[test_mask])
assert np.allclose(X_TRAIN.mean(axis=0), 0, atol=1e-12)
assert np.allclose(X_TRAIN.std(axis=0), 1, atol=1e-12)
print("train / validation / test：", 352, 88, 88)
print("受控高残差记录 validation / test：", OUTLIER_VAL.sum(), OUTLIER_TEST.sum())
```

输出为：

```text
train / validation / test： 352 88 88
受控高残差记录 validation / test： 2 2
```

这不是监督学习切分的机械照搬。即便没有标签，若先用全部数据估计均值、尺度或质心，测试 campaign 的输入分布也已经进入模型。未来工况回放会因此过于乐观。

<MlQuiz
  kind="切分审计"
  question="为什么无监督 K-means 也必须只用训练 campaign 估计标准化参数？"
  :options="['验证和测试输入的分布信息也会泄漏进距离几何与质心', '因为 K-means 实际读取监督标签', '因为测试集不能包含连续变量', '因为标准差只能在单个样本上计算']"
  :answer="0"
  explanation="均值和尺度决定欧氏空间；用未来 campaign 拟合它们，相当于让评价对象参与模型训练。"
/>

## 3. WCSS：算法真正支付的成本

定义 one-hot 分配矩阵 $\mathbf Z\in\{0,1\}^{N\times K}$，每行恰有一个 1：

$$
z_{ik}=1\iff \mathbf x_i\text{ assigned to cluster }k,
\qquad
\sum_{k=1}^{K}z_{ik}=1.
$$

K-means 最小化簇内平方和（within-cluster sum of squares, WCSS），也常称 distortion 或 inertia：

$$
\boxed{
J(\mathbf Z,\mathbf M)
=\sum_{i=1}^{N}\sum_{k=1}^{K}
z_{ik}\|\mathbf x_i-\boldsymbol\mu_k\|_2^2
}
$$

其中 $\mathbf M\in\mathbb R^{K\times D}$ 的第 $k$ 行是 $\boldsymbol\mu_k^\top$。矩阵形式为

$$
J(\mathbf Z,\mathbf M)=\|\mathbf X-\mathbf Z\mathbf M\|_F^2.
$$

这个式子可读成一个离散编码器—解码器：每个样本编码为原型索引 $z_i$，再解码为 $\boldsymbol\mu_{z_i}$；WCSS 就是总重构误差。

```python
X_TOY = np.array([[0.0, 0.0], [0.4, 0.2], [2.8, 3.0], [3.2, 2.7]])
M_TOY = np.array([[0.2, 0.1], [3.0, 2.85]])
LABEL_TOY = np.array([0, 0, 1, 1])
Z_TOY = np.eye(2)[LABEL_TOY]

direct_wcss = np.sum((X_TOY - M_TOY[LABEL_TOY]) ** 2)
matrix_wcss = np.linalg.norm(X_TOY - Z_TOY @ M_TOY, ord="fro") ** 2
assert np.allclose(direct_wcss, matrix_wcss)
print("逐样本形式 / 矩阵形式：", direct_wcss, matrix_wcss)
```

输出为：

```text
逐样本形式 / 矩阵形式： 0.22500000000000012 0.22500000000000003
```

平方不是装饰：它让最优代表点成为算术均值，并让远距离样本获得二次增长的影响。若改成一般距离却仍更新均值，已经不再精确优化上面的目标。

<MlQuiz
  kind="目标识别"
  question="把每个簇的代表点限制为某个真实样本，并最小化一般成对距离和，得到的还是标准 K-means 吗？"
  :options="['不是，更接近 K-medoids', '是，因为所有聚类都等价', '是，只要 K 相同', '不是，因为它变成监督分类']"
  :answer="0"
  explanation="K-means 的中心可落在样本之外，且均值更新与平方欧氏损失绑定；medoid 则必须是实际样本，可接受更一般的距离。"
/>

<KMeansVoronoiSwitchyardLab />

## 4. 固定质心：最近中心与 Voronoi 边界

固定 $\boldsymbol\mu_1,\ldots,\boldsymbol\mu_K$ 后，$z_i$ 只出现在第 $i$ 项：

$$
J=\sum_{i=1}^{N}
\underbrace{\|\mathbf x_i-\boldsymbol\mu_{z_i}\|_2^2}_{\text{depends only on }z_i}.
$$

因此每个样本都能独立选择：

$$
\boxed{
z_i^{\star}=\arg\min_k
\|\mathbf x_i-\boldsymbol\mu_k\|_2^2
}.
$$

比较两个质心 $a,b$，平方距离之差为

$$
\begin{aligned}
&\|\mathbf x-\boldsymbol\mu_a\|^2
-\|\mathbf x-\boldsymbol\mu_b\|^2\\
&=2\mathbf x^\top(\boldsymbol\mu_b-\boldsymbol\mu_a)
+\|\boldsymbol\mu_a\|^2-\|\boldsymbol\mu_b\|^2.
\end{aligned}
$$

$\|\mathbf x\|^2$ 在相减时消失。令差为 0，得到关于 $\mathbf x$ 的线性方程；两质心的分界面是垂直平分超平面。所有最近中心区域因此构成 Voronoi 剖分，每个单元都是凸集。

```python
probe = np.array([0.4, -0.2])
centers_2d = np.array([[-1.0, 0.2], [1.2, -0.3]])

lhs = np.sum((probe - centers_2d[0]) ** 2) - np.sum((probe - centers_2d[1]) ** 2)
rhs = (
    2 * probe @ (centers_2d[1] - centers_2d[0])
    + centers_2d[0] @ centers_2d[0]
    - centers_2d[1] @ centers_2d[1]
)
assert np.allclose(lhs, rhs)
print("平方距离差 / 展开后的线性式：", lhs, rhs)
```

输出为：

```text
平方距离差 / 展开后的线性式： 1.4699999999999998 1.4700000000000002
```

### 4.1 平票不是数学遗漏，而是接口合同

若两个距离完全相同，`argmin` 有多个合法答案。常见确定性规则有：

- 选择最小质心索引；
- 若旧标签也是最优解，保留旧标签；
- 使用稳定排序后的第一个质心。

平票规则不改变当步最小 WCSS，却可能改变后续空簇和最终编号。浮点计算中还要定义“相等”是精确相等还是容差内相等，并记录实现版本。

::: ml-stop 停下来推一推 1
一维质心为 $\mu_0=-2,\mu_1=4$。先手算 Voronoi 边界，再判断 $x=1$ 在“最小索引”与“保留旧标签”两种规则下可能进入哪个簇。
:::

<MlQuiz
  kind="推导判断"
  question="欧氏 K-means 中任意两个质心之间的分配边界为什么是超平面？"
  :options="['两项平方距离相减后，x 的二次项抵消，只剩线性式', '因为所有样本已被中心化', '因为均值一定在原点', '因为 K 必须等于 2']"
  :answer="0"
  explanation="共同的 ‖x‖² 被消掉，等距条件对 x 是一次方程；多质心时每个 Voronoi 单元是这些半空间的交。"
/>

## 5. 固定分配：为什么代表点恰好是均值

固定分配后，不同簇的中心也能分别优化。设

$$
S_k=\{i:z_i=k\},\qquad n_k=|S_k|.
$$

非空簇 $k$ 的目标是

$$
J_k(\boldsymbol\mu_k)
=\sum_{i\in S_k}\|\mathbf x_i-\boldsymbol\mu_k\|^2.
$$

对向量求梯度：

$$
\nabla_{\boldsymbol\mu_k}J_k
=-2\sum_{i\in S_k}(\mathbf x_i-\boldsymbol\mu_k)
=2n_k\boldsymbol\mu_k-2\sum_{i\in S_k}\mathbf x_i.
$$

Hessian 是 $2n_k\mathbf I\succ0$，所以唯一全局极小值为

$$
\boxed{
\boldsymbol\mu_k^\star
=\frac{1}{n_k}\sum_{i\in S_k}\mathbf x_i
}.
$$

还可用平方和分解得到更强的结论。记簇均值为 $\bar{\mathbf x}_k$，对任意候选 $\mathbf m$：

$$
\sum_{i\in S_k}\|\mathbf x_i-\mathbf m\|^2
=\sum_{i\in S_k}\|\mathbf x_i-\bar{\mathbf x}_k\|^2
+n_k\|\mathbf m-\bar{\mathbf x}_k\|^2.
$$

右侧第二项非负，且只在 $\mathbf m=\bar{\mathbf x}_k$ 时为 0。这同时给出最优性、唯一性和偏离均值要多付的精确代价。

```python
cluster = X_TRAIN[:11]
cluster_mean = cluster.mean(axis=0)
candidate = cluster_mean + np.linspace(-0.2, 0.2, cluster.shape[1])

decomposition_left = np.sum((cluster - candidate) ** 2)
decomposition_right = (
    np.sum((cluster - cluster_mean) ** 2)
    + len(cluster) * np.sum((candidate - cluster_mean) ** 2)
)
assert np.allclose(decomposition_left, decomposition_right)
print("候选中心损失 / 均值损失加惩罚：", decomposition_left, decomposition_right)
```

输出为：

```text
候选中心损失 / 均值损失加惩罚： 69.98370482583383 69.98370482583383
```

### 5.1 为什么绝对距离不能继续取均值

在一维，最小化 $\sum_i(x_i-m)^2$ 得到均值；最小化 $\sum_i|x_i-m|$ 得到中位数集合。高维欧氏距离和对应 geometric median，也不是逐列均值。损失、距离与中心更新必须成套改变。

::: ml-stop 停下来推一推 2
对一维样本 $[0,2,10]$，分别计算平方损失在均值 $4$ 与中位数 $2$ 处的值，再计算绝对损失在两处的值。哪个中心对应哪种目标？
:::

<MlQuiz
  kind="反例诊断"
  question="把平方欧氏距离改成 L1 距离，却仍把质心更新为均值，会失去什么性质？"
  :options="['更新步不再保证精确最小化固定分配下的目标', '分配步不能计算距离', '簇数会自动变成 1', '所有 Voronoi 单元都会消失']"
  :answer="0"
  explanation="算术均值是平方损失的最优代表；改变损失后必须重新推导代表点，不能只换距离函数。"
/>

## 6. Lloyd 算法：两次条件最优不等于一次联合最优

记第 $t$ 轮开始时为 $(\mathbf Z^{(t)},\mathbf M^{(t)})$。

分配步在固定质心下精确最小化：

$$
J(\mathbf Z^{(t+1)},\mathbf M^{(t)})
\le J(\mathbf Z^{(t)},\mathbf M^{(t)}).
$$

均值步在固定分配下精确最小化：

$$
J(\mathbf Z^{(t+1)},\mathbf M^{(t+1)})
\le J(\mathbf Z^{(t+1)},\mathbf M^{(t)}).
$$

合起来就是

$$
\boxed{
J^{(t)}
\ge J_{\text{assign}}^{(t+1)}
\ge J^{(t+1)}
}.
$$

WCSS 非负且单调不增，因此目标值收敛。在没有空簇、平票按稳定规则处理且每次分配真正改变时，可能的分配只有有限个，算法最终到达不再变化的固定点。

但这不等于全局最优：$\mathbf Z$ 是离散变量，联合目标非凸；“固定另一块时最优”只说明坐标块最优。不同初始中心可以进入不同吸引域。

```python
def squared_distances(x, centers):
    x = np.asarray(x, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if x.ndim != 2 or centers.ndim != 2 or x.shape[1] != centers.shape[1]:
        raise ValueError("x 与 centers 必须是特征维相同的二维数组")
    if not np.isfinite(x).all() or not np.isfinite(centers).all():
        raise ValueError("K-means 不接受 NaN 或 inf")
    return np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)

def assign_nearest(x, centers):
    # np.argmin 在平票时返回最小索引，形成确定性合同。
    return squared_distances(x, centers).argmin(axis=1)

def wcss(x, labels, centers):
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=int)
    centers = np.asarray(centers, dtype=float)
    if labels.shape != (len(x),):
        raise ValueError("labels 形状错误")
    if np.any(labels < 0) or np.any(labels >= len(centers)):
        raise ValueError("labels 超出质心索引范围")
    return float(np.sum((x - centers[labels]) ** 2))

def update_nonempty_means(x, labels, k):
    counts = np.bincount(labels, minlength=k)
    if np.any(counts == 0):
        empty = np.flatnonzero(counts == 0).tolist()
        raise RuntimeError(f"发现空簇：{empty}")
    return np.vstack([x[labels == cluster].mean(axis=0) for cluster in range(k)])
```

先用一个小数据集把两段不等式分别验算：

```python
toy_start = np.array([[-0.5, 0.5], [2.4, 2.2]])
toy_old_labels = np.array([0, 1, 0, 1])

j_before = wcss(X_TOY, toy_old_labels, toy_start)
toy_new_labels = assign_nearest(X_TOY, toy_start)
j_after_assign = wcss(X_TOY, toy_new_labels, toy_start)
toy_updated = update_nonempty_means(X_TOY, toy_new_labels, k=2)
j_after_update = wcss(X_TOY, toy_new_labels, toy_updated)

assert j_before >= j_after_assign >= j_after_update
print("旧分配 → 最近中心 → 均值更新：", j_before, j_after_assign, j_after_update)
```

输出为：

```text
旧分配 → 最近中心 → 均值更新： 26.53 3.09 0.2250000000000001
```

<KMeansIterationPlaybackLab />

<MlQuiz
  kind="证明边界"
  question="WCSS 每轮单调不增，能够推出哪项结论？"
  :options="['目标值会收敛，但一般只能保证到局部固定点', '一定到达全局最小值', '所有初始化产生相同质心', '簇编号具有物理顺序']"
  :answer="0"
  explanation="单调有下界保证的是目标序列收敛；非凸联合问题仍可能存在多个固定点。"
/>

## 7. 从零实现：初始化、迭代账本与停止规则

### 7.1 Forgy 与随机分区不是同一种“随机初始化”

Forgy 方法从样本中无放回抽取 $K$ 行作为初始质心，然后先做分配。随机分区则先给每个样本随机标签，再计算各簇均值。两者的起点分布和空簇风险不同，实验报告不能都写成含糊的 `random`。

```python
def init_forgy(x, k, rng):
    x = np.asarray(x, dtype=float)
    if not 1 <= k <= len(x):
        raise ValueError("必须满足 1 <= K <= N")
    indices = rng.choice(len(x), size=k, replace=False)
    return x[indices].copy(), indices

def init_random_partition(x, k, rng):
    x = np.asarray(x, dtype=float)
    if not 1 <= k <= len(x):
        raise ValueError("必须满足 1 <= K <= N")
    # 先保证每簇至少一个样本，再随机分配其余样本。
    order = rng.permutation(len(x))
    labels = np.empty(len(x), dtype=int)
    labels[order[:k]] = np.arange(k)
    labels[order[k:]] = rng.integers(0, k, size=len(x) - k)
    return update_nonempty_means(x, labels, k), labels

rng_init = np.random.default_rng(2606)
forgy_centers, forgy_indices = init_forgy(X_TRAIN, 3, rng_init)
partition_centers, partition_labels = init_random_partition(X_TRAIN, 3, rng_init)
assert len(np.unique(forgy_indices)) == 3
assert np.all(np.bincount(partition_labels, minlength=3) > 0)
print("Forgy 样本索引：", forgy_indices.tolist())
print("随机分区初始计数：", np.bincount(partition_labels, minlength=3).tolist())
```

### 7.2 K-means++：按最近中心平方距离采样

先均匀抽取一个样本。已有 $t-1$ 个中心时，定义

$$
D_{t-1}(\mathbf x_i)^2
=\min_{1\le k<t}\|\mathbf x_i-\boldsymbol\mu_k\|^2,
$$

再按

$$
\Pr(\boldsymbol\mu_t=\mathbf x_i)
=\frac{D_{t-1}(\mathbf x_i)^2}
{\sum_jD_{t-1}(\mathbf x_j)^2}
$$

抽取下一个中心。远离当前覆盖的样本更可能被选，但不是无条件选择最远点；这种随机化降低了单个离群点必然接管中心的风险。

```python
def init_kmeans_pp(x, k, rng):
    x = np.asarray(x, dtype=float)
    if not 1 <= k <= len(x):
        raise ValueError("必须满足 1 <= K <= N")
    chosen = [int(rng.integers(len(x)))]
    for _ in range(1, k):
        nearest_d2 = squared_distances(x, x[chosen]).min(axis=1)
        nearest_d2[chosen] = 0.0
        total = nearest_d2.sum()
        if total <= 0:
            remaining = np.setdiff1d(np.arange(len(x)), np.asarray(chosen))
            if len(remaining) == 0:
                raise ValueError("不同样本位置不足以初始化 K 个中心")
            chosen.append(int(rng.choice(remaining)))
        else:
            chosen.append(int(rng.choice(len(x), p=nearest_d2 / total)))
    return x[chosen].copy(), np.asarray(chosen)

distance_example = np.array([1.0, 2.0, 3.0])
probability_example = distance_example**2 / np.sum(distance_example**2)
assert np.allclose(probability_example, np.array([1, 4, 9]) / 14)
print("距离 1:2:3 对应概率：", np.round(probability_example, 4))
```

K-means++ 给的是初始化质量保证，不是最终全局最优证书。经典结论约束初始化后的期望目标相对全局最优为 $O(\log K)$ 量级；其前提、常数和后续 Lloyd 过程不能被简化成“必然最优”。生产中仍应使用多起点。

::: ml-stop 停下来推一推 3
已有一个中心，三个候选点到最近中心的距离分别为 $1,2,3$。先写出 $D^2$ 权重 $1:4:9$，再说明为什么这与“永远选距离 3 的点”不是同一个算法。
:::

<MlQuiz
  kind="算法辨析"
  question="K-means++ 的第二个及后续中心如何抽取？"
  :options="['按样本到最近已选中心的平方距离归一化采样', '总是选当前均值', '按监督标签分层采样', '均匀抽取并允许重复同一行']"
  :answer="0"
  explanation="D² 采样偏向未覆盖区域但保留随机性；实现还要处理重复点导致总 D² 为 0 的退化情形。"
/>

### 7.3 明确停止条件，不用“直到收敛”掩盖细节

可审计实现至少记录：

- 每轮更新后的 WCSS；
- 最大质心位移；
- 改变标签的样本数；
- 是否触发空簇处理；
- 达到 `max_iter` 还是正常停止。

标签不再变化是清晰的离散停止条件。质心位移或相对目标变化小于容差也可使用，但容差必须绑定数据尺度；不能把绝对 `1e-4` 跨标准化与原始单位直接复用。

```python
def lloyd_kmeans(x, initial_centers, max_iter=100, tol=1e-10):
    x = np.asarray(x, dtype=float)
    centers = np.asarray(initial_centers, dtype=float).copy()
    if max_iter < 1 or tol < 0:
        raise ValueError("max_iter 必须为正，tol 不能为负")
    if len(centers) < 1 or len(centers) > len(x):
        raise ValueError("质心数量必须位于 [1, N]")

    trace = []
    previous_labels = None
    for iteration in range(1, max_iter + 1):
        labels = assign_nearest(x, centers)
        counts = np.bincount(labels, minlength=len(centers))
        if np.any(counts == 0):
            raise RuntimeError(
                f"第 {iteration} 轮出现空簇 {np.flatnonzero(counts == 0).tolist()}"
            )

        updated = update_nonempty_means(x, labels, len(centers))
        objective = wcss(x, labels, updated)
        shift = float(np.linalg.norm(updated - centers, axis=1).max())
        moved = (
            len(x)
            if previous_labels is None
            else int(np.count_nonzero(labels != previous_labels))
        )
        trace.append({
            "iteration": iteration,
            "objective": objective,
            "max_shift": shift,
            "moved": moved,
        })

        same_assignment = previous_labels is not None and np.array_equal(labels, previous_labels)
        centers = updated
        previous_labels = labels.copy()
        if same_assignment or shift <= tol:
            break
    else:
        raise RuntimeError("达到 max_iter 仍未满足停止条件")

    final_labels = assign_nearest(x, centers)
    final_counts = np.bincount(final_labels, minlength=len(centers))
    if np.any(final_counts == 0):
        raise RuntimeError("最终最近中心分配再次产生空簇")
    return {
        "labels": final_labels,
        "centers": centers,
        "objective": wcss(x, final_labels, centers),
        "trace": trace,
        "converged": True,
    }
```

对每条轨迹验证 WCSS 不增：

```python
initial_pp, initial_indices = init_kmeans_pp(
    X_TRAIN, k=3, rng=np.random.default_rng(2606)
)
single_run = lloyd_kmeans(X_TRAIN, initial_pp)
single_history = np.array([row["objective"] for row in single_run["trace"]])

assert np.all(np.diff(single_history) <= 1e-10)
assert np.isclose(single_history[-1], single_run["objective"])
print("初始化索引：", initial_indices.tolist())
print("迭代轮数 / 首末 WCSS：", len(single_history), single_history[0], single_history[-1])
```

### 7.4 多起点不是重复劳动，而是非凸优化预算

每个起点独立运行到停止，只比较最终 WCSS；不能把不同运行的质心混在一起，也不能在看到测试结果后挑 seed。并列时按运行序号选择，可保证确定性重放。

```python
def run_multistart(x, k, n_init, seed, method="kmeans++"):
    if n_init < 1:
        raise ValueError("n_init 必须至少为 1")
    if method not in {"kmeans++", "forgy"}:
        raise ValueError("method 必须是 'kmeans++' 或 'forgy'")
    rng = np.random.default_rng(seed)
    runs = []
    for run_id in range(n_init):
        if method == "kmeans++":
            initial, initial_indices = init_kmeans_pp(x, k, rng)
        else:
            initial, initial_indices = init_forgy(x, k, rng)
        fitted = lloyd_kmeans(x, initial)
        fitted["run_id"] = run_id
        fitted["initial_indices"] = initial_indices
        runs.append(fitted)
    best = min(runs, key=lambda run: (run["objective"], run["run_id"]))
    return best, runs

best_case, case_runs = run_multistart(
    X_TRAIN, k=3, n_init=32, seed=2606, method="kmeans++"
)
case_objectives = np.array([run["objective"] for run in case_runs])
assert best_case["objective"] == case_objectives.min()
print("32 个起点的 WCSS min / median / max：",
      case_objectives.min(), np.median(case_objectives), case_objectives.max())
```

本案例的三团几何结构较清楚，32 个 K-means++ 起点都到达 `220.918958`。这只能说明这批数据对当前预算稳定，不能证明算法一般无局部最优。下面故意构造四个子群、却只给 $K=3$ 的布局：

```python
rng_local = np.random.default_rng(2617)
local_parts = [
    rng_local.normal([-3.2, 0.0], [0.34, 0.45], size=(42, 2)),
    rng_local.normal([-0.8, 0.0], [0.38, 0.48], size=(36, 2)),
    rng_local.normal([1.8, -1.5], [0.42, 0.38], size=(28, 2)),
    rng_local.normal([2.1, 1.8], [0.46, 0.40], size=(18, 2)),
]
X_LOCAL = np.vstack(local_parts)
_, local_runs = run_multistart(X_LOCAL, 3, 80, 2617, method="forgy")
local_objectives = np.array([run["objective"] for run in local_runs])
basins, basin_counts = np.unique(np.round(local_objectives, 3), return_counts=True)

assert len(basins) >= 3
print("局部盆地 WCSS：", basins)
print("各盆地命中次数：", basin_counts)
```

输出包含多个明显不同的固定点：

```text
局部盆地 WCSS： [146.074 165.333 172.334 172.631 200.012 200.121]
各盆地命中次数： [27 42  1  7  2  1]
```

<MlQuiz
  kind="工程选择"
  question="多起点运行后应如何冻结最终 K-means 解？"
  :options="['在训练目标上选最终 WCSS 最小者，并保存 seed、起点、轨迹和并列规则', '在测试集上挑最像预期结论的 seed', '把不同运行的每个最好质心拼成一组', '总是选择迭代轮数最多者']"
  :answer="0"
  explanation="多起点是在同一训练目标上的非凸优化预算；测试集不能参与 seed 选择，质心也必须来自同一次一致运行。"
/>

## 8. 冻结案例模型：标签规范化与样本外分配

### 8.1 质心排列对称，部署接口不能随 seed 漂移

对任意置换 $\pi$，同时做

$$
\boldsymbol\mu_k' = \boldsymbol\mu_{\pi(k)},
\qquad
z_i'=\pi^{-1}(z_i),
$$

所有样本对应的质心位置不变，WCSS 也不变。因此训练算法没有理由把“工况 A”永远放在索引 0。若下游看板、抽样队列或人工注释依赖编号，就必须在训练后做规范化。

本案例按原始单位下的比功率质心从小到大排序。这只是稳定接口，不是把高比功率簇宣判为故障。

```python
def canonicalize_by_feature(labels, centers, feature_index):
    labels = np.asarray(labels, dtype=int)
    centers = np.asarray(centers, dtype=float)
    order = np.argsort(centers[:, feature_index], kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return inverse[labels], centers[order], order

CASE_LABELS, CASE_CENTERS, CASE_ORDER = canonicalize_by_feature(
    best_case["labels"], best_case["centers"], feature_index=1
)
CASE_CENTERS_RAW = CASE_CENTERS * scale_std + scale_mean

assert np.all(np.diff(CASE_CENTERS_RAW[:, 1]) >= 0)
assert np.isclose(wcss(X_TRAIN, CASE_LABELS, CASE_CENTERS), best_case["objective"])
print("训练簇计数：", np.bincount(CASE_LABELS, minlength=3))
print("原始单位质心：\n", np.round(CASE_CENTERS_RAW, 4))
```

输出为：

```text
训练簇计数： [138 142  72]
原始单位质心：
 [[ 6.8209  0.1021  0.3426  7.4601 12.4762]
  [ 7.1567  0.1121  0.6345  9.3907  9.9834]
  [ 7.4828  0.1357  0.832  15.3677  7.1979]]
```

新样本不再更新质心，只使用冻结的均值、尺度、特征顺序和质心执行最近中心分配：

```python
def predict_kmeans(x, centers):
    labels = assign_nearest(x, centers)
    residual_d2 = squared_distances(x, centers).min(axis=1)
    return labels, residual_d2

VAL_LABELS, VAL_RESIDUAL_D2 = predict_kmeans(X_VAL, CASE_CENTERS)
TEST_LABELS, TEST_RESIDUAL_D2 = predict_kmeans(X_TEST, CASE_CENTERS)

assert np.bincount(VAL_LABELS, minlength=3).sum() == 88
assert np.bincount(TEST_LABELS, minlength=3).sum() == 88
print("validation 簇计数：", np.bincount(VAL_LABELS, minlength=3))
print("test 簇计数：", np.bincount(TEST_LABELS, minlength=3))
```

输出为：

```text
validation 簇计数： [33 34 21]
test 簇计数： [33 39 16]
```

### 8.2 合成状态只能做教材审计，不能反向命名簇

聚类标签允许置换，所以不能直接计算 `mean(cluster == audit)`。下面枚举三个编号的全部置换，只回答“几何分组是否找回生成器中的三团”；真实现场没有这个隐藏答案。

```python
def best_permuted_accuracy(audit, cluster, k):
    audit = np.asarray(audit, dtype=int)
    cluster = np.asarray(cluster, dtype=int)
    scores = []
    for permutation in itertools.permutations(range(k)):
        mapped = np.asarray(permutation)[cluster]
        scores.append(np.mean(mapped == audit))
    return float(max(scores))

train_audit_accuracy = best_permuted_accuracy(AUDIT_TRAIN, CASE_LABELS, 3)
val_audit_accuracy = best_permuted_accuracy(AUDIT_VAL, VAL_LABELS, 3)
test_audit_accuracy = best_permuted_accuracy(AUDIT_TEST, TEST_LABELS, 3)

print("合成审计准确率 train / validation / test：",
      np.round([train_audit_accuracy, val_audit_accuracy, test_audit_accuracy], 4))
```

输出为：

```text
合成审计准确率 train / validation / test： [0.9972 0.9886 0.9773]
```

这个结果说明教学数据的三团几何结构可恢复，不说明三团等于三种真实故障。下一章会用置换不变指标、稳定性、内部指标和外部业务证据系统评价聚类。

<MlQuiz
  kind="标签审计"
  question="为什么不能直接把本次 cluster 2 永久命名为“热交换受限”？"
  :options="['目标对标签置换不变，且几何相似不等于物理根因已确认', '因为 cluster 2 没有质心', '因为无监督算法不能输出任何编号', '因为 K-means 只能用于图像']"
  :answer="0"
  explanation="编号本身没有语义；命名需要原始趋势、工艺记录和专家复核，并要在重训时做版本对齐。"
/>

## 9. 尺度与特征权重：欧氏距离就是你的业务假设

原始单位下的目标为

$$
\sum_i\sum_{j=1}^{D}(x_{ij}-\mu_{z_i,j})^2.
$$

仅把压力从 bar 改报 kPa，就会把该列差异放大 $100^2$ 倍。算法无法知道这是单位换算而非业务重要性变化。

### 9.1 标准化必须训练期拟合并随模型冻结

用训练均值与标准差做

$$
\tilde x_{ij}=\frac{x_{ij}-m_j}{s_j}
$$

后，线性单位变换不会改变标准化坐标。下面同时验证“正确管线不变”和“裸欧氏距离会变”：

```python
def pair_partition_agreement(labels_a, labels_b):
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape != labels_b.shape:
        raise ValueError("两组标签形状必须相同")
    same_a = labels_a[:, None] == labels_a[None, :]
    same_b = labels_b[:, None] == labels_b[None, :]
    upper = np.triu_indices(len(labels_a), k=1)
    return float(np.mean(same_a[upper] == same_b[upper]))

X_RAW_KPA = X_RAW.copy()
X_RAW_KPA[:, 0] *= 100.0
kpa_mean = X_RAW_KPA[train_mask].mean(axis=0)
kpa_std = X_RAW_KPA[train_mask].std(axis=0, ddof=0)
X_KPA_STANDARDIZED = (X_RAW_KPA - kpa_mean) / kpa_std
assert np.allclose(X_KPA_STANDARDIZED, X)

raw_fit, _ = run_multistart(X_RAW[train_mask], 3, 16, 2606, method="kmeans++")
kpa_fit, _ = run_multistart(X_RAW_KPA[train_mask], 3, 16, 2606, method="kmeans++")
raw_unit_agreement = pair_partition_agreement(raw_fit["labels"], kpa_fit["labels"])
print("不标准化时，bar→kPa 前后成对分区一致率：", round(raw_unit_agreement, 4))
```

输出为：

```text
不标准化时，bar→kPa 前后成对分区一致率： 0.8585
```

### 9.2 标准化不是中立选择

标准化让训练期每列方差约为 1，但“波动小却关键”的比功率与“波动大但噪声多”的温差是否应等权，仍是业务决定。若工程组给出非负权重 $w_j$，目标变为

$$
J_w=\sum_i\sum_jw_j(x_{ij}-\mu_{z_i,j})^2.
$$

令 $x_{ij}'=\sqrt{w_j}x_{ij}$，普通欧氏 K-means 就能实现这个加权目标：

```python
FEATURE_WEIGHT = np.array([1.0, 2.0, 0.8, 1.0, 1.4])
sample_x = X_TRAIN[:17]
sample_centers = CASE_CENTERS
sample_labels = assign_nearest(sample_x * np.sqrt(FEATURE_WEIGHT),
                               sample_centers * np.sqrt(FEATURE_WEIGHT))

weighted_direct = np.sum(
    FEATURE_WEIGHT * (sample_x - sample_centers[sample_labels]) ** 2
)
transformed_euclidean = np.sum(
    (sample_x * np.sqrt(FEATURE_WEIGHT)
     - sample_centers[sample_labels] * np.sqrt(FEATURE_WEIGHT)) ** 2
)
assert np.allclose(weighted_direct, transformed_euclidean)
print("加权目标 / 变换后欧氏目标：", weighted_direct, transformed_euclidean)
```

权重必须进入模型制品和验证计划。依据测试效果事后调权，或让每个班次手工改权而不留版本，都会破坏可比性。

<MlQuiz
  kind="尺度审计"
  question="把一列从 bar 换成 kPa 后，未标准化 K-means 为什么可能改变分区？"
  :options="['该列的平方距离贡献被放大约 10000 倍', '质心数量自动增加', '标签变成监督标签', '均值更新不再存在']"
  :answer="0"
  explanation="欧氏距离依赖坐标单位；训练期标准化可消除线性单位换算，但是否等权仍需业务合同。"
/>

## 10. 空簇：均值公式的分母为零

若某轮没有样本分到簇 $k$，则

$$
n_k=0,
\qquad
\boldsymbol\mu_k
=\frac{1}{n_k}\sum_{i:z_i=k}\mathbf x_i
$$

未定义。空簇常由重复质心、重复样本、$K$ 过大、极端不平衡或数值平票触发。生产实现不能用 `NaN` 悄悄继续，也不能把旧中心原地保留后假装仍有 $K$ 个有效簇。

<KMeansEmptyClusterRescueLab />

### 10.1 三类可审计处置

常见策略包括：

1. **整次重启**：放弃当前 run，换 seed；轨迹解释最清楚，但会消耗优化预算。
2. **最大残差点接管**：从拥有至少两个样本的供体簇中，拿走当前残差最大的样本作为空簇种子。
3. **拆分最大簇**：选择 WCSS 最大或样本最多的簇，在其内部放置两个中心后继续。

后两项改变了标准 Lloyd 更新，不能继续引用“这一干预半步必定不增”的原证明。正确做法是记录事件，把修复后的状态当作新轨迹起点，再从下一次完整分配—更新开始检查单调性。

```python
EMPTY_X = np.array([[0.0], [0.1], [0.2], [4.0], [4.1], [8.0], [8.1]])
EMPTY_START = np.array([[0.0], [0.0], [8.0]])
empty_labels = assign_nearest(EMPTY_X, EMPTY_START)
empty_counts = np.bincount(empty_labels, minlength=3)

assert empty_counts.tolist() == [4, 0, 3]
print("重复中心后的首次分配计数：", empty_counts)
```

下面实现“最大残差点接管”。它先确保供体簇至少剩一个样本，再计算新均值：

```python
def repair_empty_by_farthest(x, labels, centers):
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=int).copy()
    centers = np.asarray(centers, dtype=float)
    k = len(centers)
    events = []
    counts = np.bincount(labels, minlength=k)

    for empty_cluster in np.flatnonzero(counts == 0):
        residual = np.sum((x - centers[labels]) ** 2, axis=1)
        eligible = counts[labels] > 1
        if not np.any(eligible):
            raise RuntimeError("没有可安全拆出的供体样本")
        candidate_scores = np.where(eligible, residual, -np.inf)
        sample_index = int(np.argmax(candidate_scores))
        donor = int(labels[sample_index])
        labels[sample_index] = empty_cluster
        counts[donor] -= 1
        counts[empty_cluster] += 1
        events.append({
            "empty_cluster": int(empty_cluster),
            "sample_index": sample_index,
            "donor_cluster": donor,
        })

    repaired_centers = update_nonempty_means(x, labels, k)
    return labels, repaired_centers, events

repaired_labels, repaired_centers, repair_events = repair_empty_by_farthest(
    EMPTY_X, empty_labels, EMPTY_START
)
assert np.all(np.bincount(repaired_labels, minlength=3) > 0)
assert np.isfinite(repaired_centers).all()
print("修复后计数：", np.bincount(repaired_labels, minlength=3))
print("修复事件：", repair_events)
```

输出为：

```text
修复后计数： [3 1 3]
修复事件： [{'empty_cluster': 1, 'sample_index': 3, 'donor_cluster': 0}]
```

::: ml-stop 停下来推一推 4
两个初始中心完全重合，而平票规则总选较小索引。解释为什么较大索引簇必为空；再说明“保留旧中心”为什么只是避免除零，不保证下一轮会恢复。
:::

<MlQuiz
  kind="故障处置"
  question="一次空簇修复后，关于 WCSS 单调性最严谨的说法是什么？"
  :options="['修复是额外干预，应记录并把修复后状态作为新轨迹起点重新检查', '任何修复都自动属于均值步，所以必定不增', '可以忽略该 run 的空簇记录', '把 NaN 质心写成 0 即可延续原证明']"
  :answer="0"
  explanation="强制拿走样本或重置中心未必是原子问题的精确最小化；不能把干预跨过的不等式当作已证明。"
/>

## 11. 残差不是概率，最近簇也不是“正常”证书

对新样本定义最近质心平方残差

$$
r^2(\mathbf x)=\min_k\|\mathbf x-\boldsymbol\mu_k\|^2.
$$

无论样本多远，K-means 都会返回一个最近簇；它没有内建的“以上皆非”。若要触发复核，应另设距离或密度门禁，并用验证数据冻结。

本案例仅用训练残差的 99.5% 分位作教学阈值：

```python
TRAIN_LABELS, TRAIN_RESIDUAL_D2 = predict_kmeans(X_TRAIN, CASE_CENTERS)
RESIDUAL_THRESHOLD = float(np.quantile(TRAIN_RESIDUAL_D2, 0.995))
VAL_REVIEW = VAL_RESIDUAL_D2 > RESIDUAL_THRESHOLD
TEST_REVIEW = TEST_RESIDUAL_D2 > RESIDUAL_THRESHOLD

assert VAL_REVIEW[OUTLIER_VAL].sum() == OUTLIER_VAL.sum() == 2
assert TEST_REVIEW[OUTLIER_TEST].sum() == OUTLIER_TEST.sum() == 2
print("训练 99.5% 残差阈值：", RESIDUAL_THRESHOLD)
print("validation 复核 / 受控记录检出 / 常规误触发：",
      VAL_REVIEW.sum(), VAL_REVIEW[OUTLIER_VAL].sum(), VAL_REVIEW[~OUTLIER_VAL].sum())
print("test 复核 / 受控记录检出 / 常规误触发：",
      TEST_REVIEW.sum(), TEST_REVIEW[OUTLIER_TEST].sum(), TEST_REVIEW[~OUTLIER_TEST].sum())
```

输出为：

```text
训练 99.5% 残差阈值： 2.703788810337902
validation 复核 / 受控记录检出 / 常规误触发： 3 2 1
test 复核 / 受控记录检出 / 常规误触发： 3 2 1
```

这只是受控数据上的验收：每组 88 条中触发 3 条，两个注入点全部进入复核，同时有一条常规记录越线。真实门限必须考虑 campaign 相关性、复核容量、漂移和代价，不能把 `99.5%` 当成通用标准。

<MlQuiz
  kind="输出解释"
  question="一个新样本离所有质心都很远，标准 K-means 的 `predict` 会怎样？"
  :options="['仍返回最近质心的编号，因此需要独立残差/拒绝门禁', '自动返回 unknown', '自动增加一个质心', '把所有簇概率设为 0']"
  :answer="0"
  explanation="最近中心规则总能给出 argmin；K-means 没有概率归一化或开放集拒绝机制。"
/>

## 12. $K$ 不是算法自己学出的数字

对同一数据，最优训练 WCSS 随 $K$ 单调不增。原因很直接：$K+1$ 个中心至少可以复制一个旧中心，复现 $K$ 个中心的解；多出的自由度只可能继续降低重构误差。

```python
K_GRID = np.arange(1, 8)
K_WCSS = []
for k_value in K_GRID:
    best_k, _ = run_multistart(
        X_TRAIN, int(k_value), n_init=10, seed=2800 + int(k_value), method="kmeans++"
    )
    K_WCSS.append(best_k["objective"])
K_WCSS = np.asarray(K_WCSS)

assert np.all(np.diff(K_WCSS) <= 1e-10)
print("K=1..7 的训练 WCSS：", np.round(K_WCSS, 3))
```

输出为：

```text
K=1..7 的训练 WCSS： [1760.     644.27   220.919  192.886  166.399  151.539  143.268]
```

因此“选择验证 WCSS 最小的 $K$”也不成立：若目标和样本相同，它通常继续偏向更大的 $K$。肘部图只是启发式，拐点可能含糊。还需结合：

- 业务上可执行的原型数量；
- bootstrap / campaign 重采样下的稳定性；
- silhouette 等内部结构指标及其球形偏好；
- 有限外部审计标签；
- GMM 的 BIC 或其他概率模型证据；
- 新数据上的可复现性与人工解释成本。

本章的 $K=3$ 来自三个固定复核队列的预先合同。如何不靠标签系统选 $K$，是下一章“聚类评估与模型选择”的主线。

<MlQuiz
  kind="模型选择"
  question="为什么不能选择训练 WCSS 最小的 K？"
  :options="['WCSS 随 K 增大单调不增，极端时 K=N 可把训练误差降为 0', 'WCSS 只能为负', 'K 不影响模型容量', 'K-means 没有目标函数']"
  :answer="0"
  explanation="更多原型带来更多重构自由度；模型选择必须加入稳定性、复杂度、业务容量或概率模型证据。"
/>

## 13. 与球形 GMM 的联系：硬分配是一个极限，不是完全等价

设 GMM 的混合权重相等，所有协方差都是 $\sigma^2\mathbf I$。责任度满足

$$
\gamma_{ik}
\propto
\exp\left(
-\frac{\|\mathbf x_i-\boldsymbol\mu_k\|^2}{2\sigma^2}
\right).
$$

当 $\sigma^2\to0$ 且最近中心唯一，最近项相对其他项的指数比趋于无穷，软责任度集中到最近中心。GMM 的均值更新

$$
\boldsymbol\mu_k
=\frac{\sum_i\gamma_{ik}\mathbf x_i}{\sum_i\gamma_{ik}}
$$

也退化为硬成员均值。

```python
gmm_probe = np.array([0.1, 0.2])
gmm_centers = np.array([[-0.8, 0.0], [1.1, 0.4]])
gmm_d2 = np.sum((gmm_probe - gmm_centers) ** 2, axis=1)

def equal_spherical_responsibility(distance2, variance):
    logits = -np.asarray(distance2) / (2 * variance)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()

for variance in [1.0, 0.1, 0.01]:
    print(variance, np.round(equal_spherical_responsibility(gmm_d2, variance), 6))
```

但二者仍有边界差异：

- K-means 不估计混合权重与方差；
- WCSS 不是归一化概率密度；
- GMM 可表达椭圆协方差和软重叠，K-means 的单元仍是最近中心 Voronoi 区；
- 若最近距离平票，零方差极限并不自动选定某个唯一编号；
- GMM 似然本身也会有局部最优和协方差坍缩风险。

<MlQuiz
  kind="模型联系"
  question="等权、同球形协方差 GMM 在方差趋近 0 且最近中心唯一时，责任度趋向什么？"
  :options="['最近中心的硬分配', '所有成分均匀分配', 'PCA 的第一主成分', '随机森林投票']"
  :answer="0"
  explanation="距离差被 1/σ² 放大，最近中心的指数项支配；这解释了联系，但 K-means 本身仍不是完整密度模型。"
/>

## 14. 结构性失效：不是多调几次 seed 就能修好

### 14.1 非凸簇与相互包围的簇

每个 K-means 簇是 Voronoi 单元与样本集的交，偏好凸、近似球形的区域。两个月牙、同心环或沿曲线延伸的工况轨迹，即使重复初始化也无法由少量凸单元完整表达。可考虑谱聚类、密度聚类或先学习合适表示。

### 14.2 方差、密度和样本量差异

一个稠密小簇靠近宽松大簇时，平方误差可能宁愿拆开大簇、吞并小簇。K-means 不是“寻找所有自然群体”的公理，而是在当前欧氏表示中做向量量化。

### 14.3 离群点：平方损失和均值双重放大影响

```python
one_dimensional = np.array([0.0, 0.2, -0.1, 0.1, -0.2])
contaminated = np.append(one_dimensional, 8.0)

mean_shift = contaminated.mean() - one_dimensional.mean()
median_shift = np.median(contaminated) - np.median(one_dimensional)
print("加入 8.0 后均值位移 / 中位数位移：", mean_shift, median_shift)
```

输出为：

```text
加入 8.0 后均值位移 / 中位数位移： 1.3333333333333333 0.05
```

稳健替代包括先做受控异常审核、trimmed K-means、K-medians、K-medoids 或带重尾成分的概率模型。不能先随意删掉“不好看”的点，再把稳定结果写成模型证据。

### 14.4 高维距离集中与无关列

高维中大量无关坐标会累加噪声平方差，使最近与次近距离差变小。PCA、随机投影或监督/度量学习可以改善表示，但任何表示都必须在训练折内拟合；PCA 保留高方差也不保证保留聚类相关结构。

### 14.5 类别、序列和缺失值不是普通连续坐标

- 无序类别做整数编码后使用欧氏距离，会捏造 `A` 到 `B` 比到 `C` 更近的顺序。
- 时间序列逐点欧氏距离对相位偏移敏感，可能需要对齐距离、特征摘要或专门序列模型。
- NaN 会传播到距离和质心，必须先定义缺失机制、训练期插补和缺失指示，而不是依赖 `nanmean` 默默改变每列成员集。

<MlQuiz
  kind="适用边界"
  question="两个同心圆数据即使增加 K-means 重启次数，为什么仍难按内外环分成两个簇？"
  :options="['两个 Voronoi 单元是凸半空间式区域，不能表达相互包围的非凸结构', '圆上没有均值', '平方距离不能用于二维', '初始化次数越多 K 越小']"
  :answer="0"
  explanation="这是模型归纳偏置不匹配，不只是局部最优；更多 seed 不能改变最近中心分区的几何族。"
/>

## 15. 扩展算法：改变数据量、权重或代表点

### 15.1 样本加权 K-means

若窗口代表的持续时间、产量或可信度不同，可定义非负样本权重 $a_i$：

$$
J_a=\sum_i a_i\|\mathbf x_i-\boldsymbol\mu_{z_i}\|^2,
\qquad
\boldsymbol\mu_k
=\frac{\sum_{i:z_i=k}a_i\mathbf x_i}
{\sum_{i:z_i=k}a_i}.
$$

分配步在权重只依赖样本且 $a_i>0$ 时仍选择最近质心；更新步变成加权均值。零权重样本不应决定任何中心，负权重则破坏目标解释。

```python
weighted_x = np.array([[0.0], [2.0], [10.0]])
sample_weight = np.array([1.0, 3.0, 0.5])
weighted_center = np.sum(sample_weight[:, None] * weighted_x, axis=0) / sample_weight.sum()

gradient = 2 * np.sum(
    sample_weight[:, None] * (weighted_center - weighted_x), axis=0
)
assert np.allclose(gradient, 0)
print("加权中心：", weighted_center)
```

### 15.2 Mini-batch K-means

全量 Lloyd 每轮读取全部 $N$ 个样本。Mini-batch 每次只抽小批量，并按在线均值更新被命中的中心。例如质心 $k$ 已累计权重 $c_k$，收到样本 $\mathbf x$ 后：

$$
c_k\leftarrow c_k+1,
\qquad
\boldsymbol\mu_k
\leftarrow\boldsymbol\mu_k+\frac{1}{c_k}(\mathbf x-\boldsymbol\mu_k).
$$

它降低单步成本并适合流式近似，但轨迹不再是每半步精确最小化全数据 WCSS；要单独报告吞吐、最终全量 WCSS、seed 和批次顺序敏感性。

### 15.3 K-medoids 与向量量化

K-medoids 把代表点限制为真实样本，可使用一般成对距离，通常比均值更抗极端值，但计算成本更高。K-means 也可解释为向量量化：用 $K$ 个 codeword 替代 $N$ 个连续向量。若每个样本只存原型索引，大致需要 $N\log_2K$ 位索引外加 $KD$ 个浮点数的 codebook；代价正是重构 distortion。

### 15.4 不能混淆的几个相近名字

| 方法 | 分配 | 代表点 / 模型 | 主要变化 |
|---|---|---|---|
| K-means | 最近中心硬分配 | 簇内均值 | 平方欧氏 WCSS |
| K-medians | 最近中心硬分配 | 中位数型代表 | L1，更抗单维尖峰 |
| K-medoids | 最近 medoid | 真实样本 | 支持一般距离 |
| GMM | 软责任度 | 均值、协方差、权重 | 概率密度与椭圆形状 |
| Mini-batch K-means | 小批近似分配 | 在线均值 | 速度换精度 |

## 数值验证

### 16.1 一组不依赖“图看起来对”的回归门禁

工业实现应把数学性质写成可执行断言：

```python
# 1. 每条正常 Lloyd 轨迹的 WCSS 不增。
for fitted_run in case_runs:
    history = np.array([row["objective"] for row in fitted_run["trace"]])
    assert np.all(np.diff(history) <= 1e-10)

# 2. 最终质心等于对应成员均值。
for cluster_id in range(3):
    assert np.allclose(
        CASE_CENTERS[cluster_id],
        X_TRAIN[CASE_LABELS == cluster_id].mean(axis=0),
    )

# 3. 预测标签确实选择逐行最小平方距离。
train_d2 = squared_distances(X_TRAIN, CASE_CENTERS)
assert np.array_equal(CASE_LABELS, train_d2.argmin(axis=1))

# 4. WCSS 与逐行最小距离和一致。
assert np.isclose(best_case["objective"], train_d2.min(axis=1).sum())

# 5. 质心置换并同步重映射标签，目标不变。
permutation = np.array([2, 0, 1])
permuted_centers = CASE_CENTERS[permutation]
inverse = np.empty_like(permutation)
inverse[permutation] = np.arange(3)
permuted_labels = inverse[CASE_LABELS]
assert np.isclose(
    wcss(X_TRAIN, permuted_labels, permuted_centers),
    best_case["objective"],
)

print("目标单调、均值固定点、最近中心、目标恒等式、标签置换：全部通过")
```

### 16.2 确定性、边界输入与失败要显式

```python
# 同 seed、同实现必须精确重放。
repeat_best, repeat_runs = run_multistart(
    X_TRAIN, 3, 32, 2606, method="kmeans++"
)
assert np.array_equal(repeat_best["labels"], best_case["labels"])
assert np.array_equal(repeat_best["centers"], best_case["centers"])

# K=1、K=N 的基本边界。
one_center, _ = init_kmeans_pp(X_TOY, 1, np.random.default_rng(1))
one_fit = lloyd_kmeans(X_TOY, one_center)
assert np.allclose(one_fit["centers"][0], X_TOY.mean(axis=0))

all_center_fit = lloyd_kmeans(X_TOY, X_TOY.copy())
assert np.isclose(all_center_fit["objective"], 0)

# 重复中心必须失败，而不是产生 NaN。
try:
    lloyd_kmeans(EMPTY_X, EMPTY_START)
except RuntimeError as error:
    assert "空簇" in str(error)
else:
    raise AssertionError("重复中心场景应触发空簇门禁")

# 非有限值必须在距离边界被拒绝。
try:
    squared_distances(np.array([[np.nan, 0.0]]), np.zeros((1, 2)))
except ValueError as error:
    assert "NaN" in str(error)
else:
    raise AssertionError("NaN 输入应被拒绝")

print("确定性重放、K 边界、空簇与非有限值门禁：全部通过")
```

### 16.3 成熟库对照只核接口，不替代问题合同

下面给出 scikit-learn 对照，但仓库不为教材新增该依赖，因此固定验证会跳过。注意不同版本对 `n_init='auto'`、容差和算法实现的默认值可能不同；生产代码应显式填写并记录版本。

```python
# 不执行：可选环境对照，需要 scikit-learn。
from sklearn.cluster import KMeans

reference = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=32,
    max_iter=100,
    tol=1e-10,
    random_state=2606,
    algorithm="lloyd",
).fit(X_TRAIN)

print(reference.inertia_, reference.n_iter_)
```

成熟库能提供优化过的距离计算、并行和参数检查，却不知道一行样本是否泄漏、单位是否合理、三个簇是否可执行、空簇策略是否符合审计要求。

## 复杂度与工程边界

### 17.1 时间、内存和吞吐

朴素 Lloyd 每轮计算 $N\times K$ 个 $D$ 维距离：

$$
T_{\text{one run}}=O(I N K D),
$$

其中 $I$ 是迭代轮数。使用 $R$ 个起点后为 $O(RINKD)$。若显式保存距离矩阵，额外内存为 $O(NK)$；分块计算可以把峰值内存降到 $O(BK)$，但仍要准确累计 WCSS 与簇和。

三类常见加速不能混成同一承诺：

- 向量化/BLAS 改变常数，不改变目标；
- 利用三角不等式跳过不可能获胜的中心，在精确条件下不改变结果；
- mini-batch、近似近邻或低精度计算改变优化路径，需重新验证质量。

<MlQuiz
  kind="复杂度估算"
  question="全量 Lloyd 使用 R 个起点、I 轮、N 个样本、K 个中心和 D 个特征时，主导时间量级是什么？"
  :options="['O(RINKD)', 'O(N+K+D)', 'O(K²)', 'O(log N)']"
  :answer="0"
  explanation="每轮每个样本要与 K 个 D 维中心比较，多起点再乘 R；实际实现可降低常数或跳过部分距离。"
/>

### 17.2 训练制品清单

一次可重放的 K-means 发布至少保存：

| 制品 | 必要内容 |
|---|---|
| 样本合同 | 窗口定义、资产范围、campaign 切分、去重规则 |
| 特征合同 | 名称、顺序、单位、变换、缺失和裁剪策略 |
| 尺度合同 | 训练均值、训练尺度、可选业务权重 |
| 优化合同 | $K$、初始化、`n_init`、seed、停止规则、空簇与平票策略 |
| 结果制品 | 规范化质心、训练计数、WCSS、每个 run 的轨迹与未选原因 |
| 路由合同 | 原型编号、最近距离、拒绝阈值、人工复核动作 |
| 版本证据 | 数据快照、代码版本、库版本、构建与回归测试结果 |

### 17.3 上线监控不是只看平均 WCSS

至少监控以下切片，并按 campaign 或资产聚合置信区间：

- 各簇占比与连续空篮；
- 最近质心平方残差的中位数、高分位和越线率；
- 第一、第二近质心距离差，识别边界样本；
- 特征缺失率、单位和尺度漂移；
- 质心与冻结版本的匹配距离；
- 人工复核后的新工况、误命名和漏覆盖记录；
- 计算时延、重试、空簇事件与 seed 重放一致性。

若簇占比改变，可能是真实工况构成变化，也可能是传感器单位、缺失插补或上游过滤变化。监控只能触发调查，不能自动宣判原因。

<MlQuiz
  kind="部署审计"
  question="重训后质心索引顺序改变，下游仍把 cluster 0 当旧工况 A，最直接的问题是什么？"
  :options="['标签置换被误当成真实工况漂移，需要先做质心匹配与版本对齐', 'K-means 会自动修正数据库字段', 'WCSS 必然变成负数', '所有样本都会成为离群点']"
  :answer="0"
  explanation="簇编号没有天然身份。跨版本比较必须先按冻结特征空间匹配质心，并保留人工确认和回滚。"
/>

### 17.4 何时应停止使用 K-means

出现以下情况时，继续堆 seed 或增大 `max_iter` 通常不是正确方向：

- 工程相似性无法用连续欧氏坐标表达；
- 目标结构明显非凸、密度差异巨大或簇严重不平衡；
- 需要成员概率、置信度或生成密度；
- 离群点决定均值而无法先行处置；
- $K$ 个原型没有可执行的业务动作；
- 重采样和时间回放下分区不稳定；
- 下游把统计簇当作未经复核的故障真值。

此时应回到问题定义：更换表示、距离、模型或任务，而不是把算法输出包装得更像答案。

## 18. 章节练习

### 基础推导

1. 从 one-hot 目标推导最近中心分配，并明确每行和为 1 的约束在何处使用。
2. 用梯度与平方和分解两种方法证明均值更新，说明 $n_k=0$ 时各自在哪一步失效。
3. 推导加权样本的质心更新，并写出权重全为正时分配步不变的理由。
4. 对两个二维质心写出精确边界方程，验证法向量与质心连线平行。

### 实现与反例

5. 在 `lloyd_kmeans` 中加入“相对 WCSS 变化”停止条件，同时保留标签稳定门禁；比较两者的停止轮次。
6. 实现随机分区、Forgy 和 K-means++ 三种初始化，在四子群反例上运行 200 次，报告最终盆地分布。
7. 构造重复点使 K-means++ 的剩余 $D^2$ 总和为 0，验证实现给出明确错误或受控回退。
8. 生成同心圆和不同方差高斯数据，说明哪一种是结构失配，哪一种可能通过更换概率模型改善。
9. 对本案例逐 campaign bootstrap，比较质心匹配距离和分区稳定性；不要只画一条 WCSS 曲线。

### 工程审计

10. 设计一张模型卡，写清簇编号对齐、空簇事件、拒绝门禁、人工复核和回滚责任人。
11. 模拟把 `specific_power_kwh_per_knm3` 误报为 Wh/Nm³，验证输入单位门禁能在分配前阻断。
12. 比较全量、分块精确和 mini-batch 三种实现的峰值内存、耗时与最终 WCSS。

练习骨架：

```python
# TODO：实现按 campaign 重采样、质心最优匹配与稳定性区间。
def bootstrap_centroid_stability(x, groups, k, n_boot, seed):
    raise NotImplementedError
```

## 来源与许可

- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 752–758 页，§21.3–§21.3.7：用于核对 WCSS 矩阵形式、交替最小化、Voronoi、空簇实例、向量量化、K-means++、K-medoids、mini-batch 与 WCSS 随 $K$ 单调的边界。
- Stanford CS229 Lecture Notes，PDF 第 146–149 页，Chapter 10：用于核对 Lloyd 两步、distortion、坐标下降、单调收敛、非凸局部最优与多起点。
- CMU 10-301/10-601 Lecture 25，PDF 第 10–42 页：用于交叉核对目标—交替最小化—初始化反例—K-means++—失效形状的讲解范围。
- Jonathan Richard Shewchuk，UC Berkeley CS 189/289A 合订讲义，PDF 第 134–137 页：用于核对 NP-hard 背景、均值/最近中心两步、平票、有限分配停止、Forgy/随机分区/K-means++ 区分、尺度与局部盆地。

本章铸造车间压缩空气站案例的 66 个教学 campaign、528 条合成窗口、五路量测、三个隐藏生成状态、四条受控高残差记录、分组留出合同、全部 NumPy 代码、17 道即时测验和三项 SVG 网页实验均为本站独立创作。它们不代表真实企业、铸造工艺、空压机性能、能耗基准、露点要求、群控策略、维修根因、安全阈值或监管结论。没有复制、改编或再分发参考资料中的图形、代码、案例和课堂测验。

完整来源登记见 [课程来源与许可](./sources.md) 。

<MlLessonProgress
  storage-key="26-kmeans"
  lesson-id="26"
  next-href="/courses/machine-learning/27-clustering-evaluation"
  next-label="下一章：聚类评估与模型选择"
/>
