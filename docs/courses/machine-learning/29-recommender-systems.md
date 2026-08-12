---
title: 矩阵分解与 BPR 推荐
description: 从检修知识卡曝光日志出发，推导偏置矩阵分解、SGD、ALS、加权隐式反馈、BPR 与全候选集 Top-K 评价
course: machine-learning
edition: industrial
---

<MlLessonHero
  number="29"
  title="矩阵分解与 BPR 推荐"
  question="评分矩阵里的空白既不是零分，也不自动是负例：怎样从被历史策略选择过的稀疏反馈中学习下一张知识卡，同时把冷启动、曝光偏差和反馈回路写进工程合同？"
  duration="建议 660–780 分钟"
  :outcomes="[
    '从观测集上的高斯似然推导全局、用户、物品偏置与低秩交互项，并手写可梯度检查的 SGD',
    '解释直接 SVD、观测掩码矩阵分解与 PCA 的关系，识别零填充、不可识别变换和冷 ID 失效',
    '推导显式反馈 ALS 与置信加权隐式 ALS，说明半步凸、联合非凸以及稀疏加速的来源',
    '从成对伯努利似然推导 BPR 目标和全部更新式，在同一数据与候选池上比较热门、pointwise 与 BPR',
    '用全库 Recall@K、NDCG@K、覆盖率、头尾切片和曝光审计设计可上线但不越权的推荐资格包'
  ]"
/>

> 本章的推荐结果只用于给可靠性工程师排列可选知识卡，不得自动修改检修步骤、跳过安全规程、变更设备参数或替代专业判断。历史点击能说明“在旧策略展示后发生了什么”，不能证明知识卡正确，更不能证明未展示内容无用。

## 贯穿案例：检修知识库下一张卡推荐

### 1. 决策、反馈和不可越过的边界

某制造企业的检修知识库有 64 张卡片，覆盖润滑、振动、热像和电气四类主题。72 名可靠性工程师在处理工单时会看见候选卡片，随后可能跳过、收藏、完成学习并给出 1–5 分。产品问题不是“补完整张评分表”，而是：

$$
\text{给定工程师 }u\text{ 在时刻 }t\text{ 可见且合规的候选集 }\mathcal C_{u,t},
\quad
\text{返回前 }K\text{ 张供人工选择。}
$$

`候选集` 先执行权限、设备型号、语言、版本有效期和安全规则，排序器只在合法集合内决定次序。把已废止规程放进高分榜，再靠模型“学会别推荐”，属于架构错误。

教学数据采用以下冻结合同：

| 对象 | 定义 | 数量/口径 |
|---|---|---:|
| 用户 | 可靠性工程师 ID | 72 |
| 物品 | 检修知识卡 ID | 64 |
| warm 物品 | 训练期已经出现 | 56 |
| cold 物品 | 新导入，只在未来测试窗口出现 | 8 |
| 单用户事件序列 | 12 次训练曝光、3 次验证曝光、3 次测试曝光 | 18 |
| 显式反馈 | 曝光后的 1–5 分 | 只在曝光格可见 |
| 隐式正反馈 | 训练期评分不低于 4，解释为收藏/完成 | 二值 |
| 主排序指标 | 全合法候选池上的 Recall@5、NDCG@5 | 按用户平均 |
| 审计指标 | catalog coverage、head/tail/cold Recall@5 | 不得被主指标抵消 |

这里的 8 张 cold 卡片故意具有内容分布变化；纯 ID 模型既没有它们的训练交互，也没有内容特征，测试必须单列而不能藏进总体平均。

```python
import math
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def make_maintenance_case(seed=2908, n_users=72, n_items=64, true_dim=4):
    rng = np.random.default_rng(seed)
    user_group = np.arange(n_users) % 4
    item_topic = np.arange(n_items) % 4
    user_true = 0.18 * rng.normal(size=(n_users, true_dim))
    item_true = 0.18 * rng.normal(size=(n_items, true_dim))
    user_true[np.arange(n_users), user_group] += 1.05
    item_true[np.arange(n_items), item_topic] += 1.0
    user_bias_true = rng.normal(0, 0.28, n_users)
    item_quality = rng.normal(0, 0.34, n_items)
    item_quality[:8] += 0.35
    item_quality[-8:] -= 0.85  # 新导入卡片形成可审计的 cold cohort shift
    utility = (
        3.0
        + user_bias_true[:, None]
        + item_quality[None, :]
        + 1.15 * (user_true @ item_true.T)
    )
    warm = np.arange(n_items - 8)
    cold = np.arange(n_items - 8, n_items)
    rows = []
    for user in range(n_users):
        exposure_logit = (
            1.1 * item_quality[warm]
            + 0.75 * (item_topic[warm] == user_group[user])
            + 0.12 * utility[user, warm]
        )
        probability = np.exp(exposure_logit - exposure_logit.max())
        probability /= probability.sum()
        selected_warm = rng.choice(warm, size=17, replace=False, p=probability)
        ordered = rng.permutation(selected_warm)
        chosen = list(ordered) + [int(rng.choice(cold))]
        parts = ["train"] * 12 + ["valid"] * 3 + ["test"] * 3
        for position, (item, part) in enumerate(zip(chosen, parts)):
            noisy = utility[user, item] + rng.normal(0, 0.35)
            rating = float(np.clip(np.rint(noisy), 1, 5))
            rows.append((user, item, position, part, rating, item >= n_items - 8))
    return {
        "rows": rows,
        "utility": utility,
        "user_group": user_group,
        "item_topic": item_topic,
        "item_quality": item_quality,
        "n_users": n_users,
        "n_items": n_items,
    }


CASE = make_maintenance_case()
ROWS = CASE["rows"]
N_USERS = CASE["n_users"]
N_ITEMS = CASE["n_items"]


def arrays(part):
    selected = [row for row in ROWS if row[3] == part]
    return (
        np.array([row[0] for row in selected], dtype=int),
        np.array([row[1] for row in selected], dtype=int),
        np.array([row[4] for row in selected], dtype=float),
    )


TRAIN_U, TRAIN_I, TRAIN_R = arrays("train")
VALID_U, VALID_I, VALID_R = arrays("valid")
TEST_U, TEST_I, TEST_R = arrays("test")
print(len(TRAIN_R), len(VALID_R), len(TEST_R))
print("cold share:", round(np.mean(TEST_I >= N_ITEMS - 8), 3))
print("rating mean:", np.round([TRAIN_R.mean(), VALID_R.mean(), TEST_R.mean()], 3))
```

预期输出：

```text
864 216 216
cold share: 0.333
rating mean: [3.617 3.606 3.167]
```

这 1296 条记录是**曝光事件**，不是从完整偏好矩阵均匀抽到的单元格。训练均值比测试均值高，不只是随机噪声：新卡片队列改变了数据分布。

<MlQuiz
  kind="决策合同"
  question="历史日志里工程师 E-17 与卡片 C60 没有交互，当前唯一合法的结论是什么？"
  :options="['E-17 不喜欢 C60', 'C60 的评分是 0', '没有观测到该组合的反馈，是否曝光还需查日志', 'C60 必须从候选集删除']"
  :answer="2"
  explanation="缺少行为可能来自未曝光、未记录或曝光后未行动；没有曝光字段时连“看过但没点”也无法确认。"
/>

<ExposureMatrixLightTableLab />

### 1.1 两张表不能合并成一个布尔值

工业日志至少保留：

1. `impression(user_id, item_id, timestamp, request_id, position, propensity, policy_version)`；
2. `action(request_id, action_type, value, timestamp)`；
3. `catalog(item_id, valid_from, valid_to, eligibility, content_version)`。

曝光表回答“旧策略允许我们观察哪里”；行为表回答“曝光后发生了什么”。只存点击会让未点击与未展示不可区分；只存最终 Top-K 又无法重建位置和展示概率。

## 2. 显式反馈、隐式反馈与目标错位

令

$$
\Omega=\{(u,i):\text{在训练窗口内观测到评分 }r_{ui}\}.
$$

显式反馈提供带尺度的 $r_{ui}\in\{1,\ldots,5\}$，适合建模评分条件均值。隐式反馈只有点击、收藏、播放或完成，通常只可靠地告诉我们正事件；行为次数更像信心或强度，不是“偏好值可以无限增大”。

| 日志 | 可直接构造的监督信号 | 不能默认为 |
|---|---|---|
| 1–5 分评分 | 观测格上的有序/数值反馈 | 未评分格为 0 分 |
| 收藏、完成 | 曝光后的正行为 | 所有未收藏卡都是负例 |
| 曝光后跳过 | 弱负信号，受位置和上下文影响 | 永久厌恶 |
| 重复打开 20 次 | 较高行为强度/信心 | 偏好标签等于 20 |
| 未曝光 | 没有行为可观察 | 负例 |

推荐排序最终关心 Top-K。用 RMSE 训练的评分模型可以成为候选，但低 RMSE 不保证前五名次序最好；BPR 则直接针对成对次序。目标不同，评价必须回到同一未来决策。

<MlQuiz
  kind="反馈语义"
  question="一张卡被同一工程师打开 20 次，在置信加权隐式矩阵分解中最常见的处理是什么？"
  :options="['令偏好标签 p=20', '令偏好指示 p=1，并提高置信权重 c', '令该用户向量固定为 20', '把另外 63 张卡都标成强负例']"
  :answer="1"
  explanation="二值偏好表示是否发生正行为，次数进入置信权重；还需防刷量、自动刷新和重复会话污染。"
/>

## 3. 先做偏置基线：别让潜因子替均值加班

评分存在三个容易解释的主效应：

$$
\widehat r_{ui}^{\text{bias}}=\mu+b_u+b_i.
$$

$\mu$ 是全局均值，$b_u$ 表示某工程师整体打分偏高或偏低，$b_i$ 表示某知识卡普遍更受欢迎。正则目标为

$$
J_{\text{bias}}
=\frac12\sum_{(u,i)\in\Omega}
\left(r_{ui}-\mu-b_u-b_i\right)^2
+\frac{\lambda}{2}
\left(\sum_u b_u^2+\sum_i b_i^2\right).
$$

对单条记录定义 $e_{ui}=r_{ui}-\mu-b_u-b_i$，则 SGD 更新为

$$
b_u\leftarrow b_u+\eta(e_{ui}-\lambda b_u),\qquad
b_i\leftarrow b_i+\eta(e_{ui}-\lambda b_i).
$$

```python
def fit_bias_baseline(
    users, items, ratings, n_users, n_items,
    learning_rate=0.025, regularization=0.15,
    epochs=80, seed=2909,
):
    rng = np.random.default_rng(seed)
    model = {
        "mu": float(ratings.mean()),
        "bu": np.zeros(n_users),
        "bi": np.zeros(n_items),
    }
    for _ in range(epochs):
        for index in rng.permutation(len(ratings)):
            u, i, rating = users[index], items[index], ratings[index]
            error = rating - model["mu"] - model["bu"][u] - model["bi"][i]
            model["bu"][u] += learning_rate * (
                error - regularization * model["bu"][u]
            )
            model["bi"][i] += learning_rate * (
                error - regularization * model["bi"][i]
            )
    return model


def predict_bias(model, users, items):
    return model["mu"] + model["bu"][users] + model["bi"][items]


BIAS = fit_bias_baseline(TRAIN_U, TRAIN_I, TRAIN_R, N_USERS, N_ITEMS)
bias_valid_rmse = np.sqrt(
    np.mean((VALID_R - predict_bias(BIAS, VALID_U, VALID_I)) ** 2)
)
print("bias validation RMSE:", round(bias_valid_rmse, 4))
```

预期输出：

```text
bias validation RMSE: 0.7832
```

基线必须进入后续对比。否则“矩阵分解有效”可能只是它偷偷重学了一遍用户和物品均值。

<MlQuiz
  kind="偏置项"
  question="加入物品偏置 b_i 后，潜向量 q_i 最应该保留哪类信息？"
  :options="['该卡无条件的平均受欢迎度', '超出全局与主效应后，和用户潜偏好的匹配结构', '卡片数据库主键', '所有未观测格都为零的事实']"
  :answer="1"
  explanation="全局、用户和物品主效应先被偏置吸收，点积更集中描述条件交互。"
/>

## 4. 从观测似然到带偏置低秩矩阵分解

令工程师与知识卡各有 $k$ 维潜向量：

$$
\mathbf p_u,\mathbf q_i\in\mathbb R^k,\qquad
\widehat r_{ui}
=\mu+b_u+b_i+\mathbf p_u^\top\mathbf q_i.
$$

只对已观测评分作同方差高斯假设：

$$
r_{ui}\mid u,i,\Theta
\sim
\mathcal N\!\left(
\mu+b_u+b_i+\mathbf p_u^\top\mathbf q_i,\sigma^2
\right),
\quad (u,i)\in\Omega.
$$

独立高斯先验会把负对数后验化成

$$
\boxed{
J(\Theta)
=\frac12\sum_{(u,i)\in\Omega}
\left(r_{ui}-\mu-b_u-b_i-\mathbf p_u^\top\mathbf q_i\right)^2
+\frac{\lambda}{2}
\left(
\|\mathbf P\|_F^2+\|\mathbf Q\|_F^2
+\|\mathbf b_u\|_2^2+\|\mathbf b_i\|_2^2
\right)
}.
$$

用掩码 $\mathbf M_{ui}=\mathbb I[(u,i)\in\Omega]$，交互项可以写成

$$
\left\|
\mathbf M\odot
\left(
\mathbf R-\mu\mathbf 1\mathbf 1^\top
-\mathbf b_u\mathbf1^\top
-\mathbf1\mathbf b_i^\top
-\mathbf P\mathbf Q^\top
\right)
\right\|_F^2.
$$

掩码进入损失，不是先把 $\mathbf R$ 的空白改成数字。低秩 $k$ 太小会混合不同专业，太大会记住稀少 ID；$\lambda$ 太小使范数膨胀，太大则把交互项压回偏置基线。

<MlQuiz
  kind="观测掩码"
  question="为什么目标函数只对 (u,i)∈Ω 求和？"
  :options="['为了让矩阵变成方阵', '因为只有这些格子的评分似然有观测证据，空白格不是数值标签', '为了强制所有预测为正', '因为潜因子不能处理稠密数据']"
  :answer="1"
  explanation="未观测格没有显式评分似然；把它们纳入平方误差必须先另行声明负例或缺失机制。"
/>

## 5. 与 SVD/PCA 的关系：完整矩阵结论不能越界搬运

对一个**完整且已适当中心化**的矩阵 $\mathbf X$，

$$
\min_{\operatorname{rank}(\widehat{\mathbf X})\le k}
\|\mathbf X-\widehat{\mathbf X}\|_F^2
$$

由截断 SVD 给出；PCA 的低维重构也落在同一代数对象上。推荐矩阵则只在 $\Omega$ 上有标签，缺失又受历史曝光策略影响，还要处理偏置、置信度、Top-K 和冷启动。

```python
TRUTH = np.array([
    [5, 4, 1, 1],
    [4, 5, 1, 1],
    [1, 1, 5, 4],
    [1, 1, 4, 5],
], dtype=float)
MASK = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 0],
], dtype=bool)
ZERO_FILLED = np.where(MASK, TRUTH, 0.0)
left, singular_values, right_t = np.linalg.svd(
    ZERO_FILLED, full_matrices=False
)
ZERO_RANK2 = (left[:, :2] * singular_values[:2]) @ right_t[:2]
print(
    "observed RMSE:",
    round(np.sqrt(np.mean((TRUTH[MASK] - ZERO_RANK2[MASK]) ** 2)), 3),
)
print(
    "missing mean, truth vs zero-SVD:",
    round(TRUTH[~MASK].mean(), 3),
    round(ZERO_RANK2[~MASK].mean(), 3),
)
```

预期输出：

```text
observed RMSE: 0.743
missing mean, truth vs zero-SVD: 2.0 0.51
```

零填充改变了优化问题：算法现在被要求重构大量伪造的 0。即使观测 RMSE 尚可，空白区也被系统性压低。

::: warning 停下来推一推 1
<!-- ml-stop -->
假设评分范围是 1–5，矩阵的 99.8% 为空。把空白填 0 后，rank-10 SVD 的总体平方误差很低。请分别写出它真正优化的目标、业务真正有标签的目标，并解释为什么“总体误差低”主要奖励预测 0。
:::

<MlQuiz
  kind="SVD 边界"
  question="什么时候截断 SVD 可以直接声称解决了最佳 rank-k 平方重构？"
  :options="['任意有选择偏差的稀疏推荐矩阵', '完整矩阵上的 Frobenius 范数问题，或已明确采用相应完整数据目标', '只要用户数大于物品数', '只要先把空白填成全局均值']"
  :answer="1"
  explanation="Eckart–Young 结论针对完整矩阵的 Frobenius 目标；观测掩码会改变问题。"
/>

## 6. SGD：从一条评分推到四组更新

定义

$$
e_{ui}=r_{ui}-\mu-b_u-b_i-\mathbf p_u^\top\mathbf q_i.
$$

单条记录的正则损失为

$$
\ell_{ui}
=\frac12e_{ui}^2
+\frac\lambda2
\left(
b_u^2+b_i^2+\|\mathbf p_u\|^2+\|\mathbf q_i\|^2
\right).
$$

逐项求导：

$$
\frac{\partial\ell}{\partial b_u}=-e_{ui}+\lambda b_u,\qquad
\frac{\partial\ell}{\partial b_i}=-e_{ui}+\lambda b_i,
$$

$$
\frac{\partial\ell}{\partial\mathbf p_u}
=-e_{ui}\mathbf q_i+\lambda\mathbf p_u,\qquad
\frac{\partial\ell}{\partial\mathbf q_i}
=-e_{ui}\mathbf p_u+\lambda\mathbf q_i.
$$

沿负梯度移动得到

$$
\begin{aligned}
b_u&\leftarrow b_u+\eta(e_{ui}-\lambda b_u),\\
b_i&\leftarrow b_i+\eta(e_{ui}-\lambda b_i),\\
\mathbf p_u&\leftarrow\mathbf p_u+\eta(e_{ui}\mathbf q_i-\lambda\mathbf p_u),\\
\mathbf q_i&\leftarrow\mathbf q_i+\eta(e_{ui}\mathbf p_u-\lambda\mathbf q_i).
\end{aligned}
$$

同一步中 $\mathbf q_i$ 的梯度必须使用更新前的 $\mathbf p_u$；否则实现的是另一个带顺序效应的算法。

```python
def predict_mf(model, users, items):
    return (
        model["mu"]
        + model["bu"][users]
        + model["bi"][items]
        + np.sum(model["p"][users] * model["q"][items], axis=1)
    )


def fit_explicit_mf(
    users, items, ratings, n_users, n_items,
    factors=4, learning_rate=0.018,
    regularization=0.035, epochs=90, seed=2911,
):
    rng = np.random.default_rng(seed)
    model = {
        "mu": float(ratings.mean()),
        "bu": np.zeros(n_users),
        "bi": np.zeros(n_items),
        "p": rng.normal(0, 0.08, size=(n_users, factors)),
        "q": rng.normal(0, 0.08, size=(n_items, factors)),
    }
    history = []
    for _ in range(epochs):
        for index in rng.permutation(len(ratings)):
            u, i, rating = users[index], items[index], ratings[index]
            prediction = (
                model["mu"] + model["bu"][u] + model["bi"][i]
                + model["p"][u] @ model["q"][i]
            )
            error = rating - prediction
            old_p = model["p"][u].copy()
            model["bu"][u] += learning_rate * (
                error - regularization * model["bu"][u]
            )
            model["bi"][i] += learning_rate * (
                error - regularization * model["bi"][i]
            )
            model["p"][u] += learning_rate * (
                error * model["q"][i] - regularization * model["p"][u]
            )
            model["q"][i] += learning_rate * (
                error * old_p - regularization * model["q"][i]
            )
        train_error = ratings - predict_mf(model, users, items)
        history.append(float(np.sqrt(np.mean(train_error ** 2))))
    model["history"] = np.asarray(history)
    return model


MF_CANDIDATES = {}
for regularization in [0.005, 0.02, 0.05, 0.15]:
    candidate = fit_explicit_mf(
        TRAIN_U, TRAIN_I, TRAIN_R, N_USERS, N_ITEMS,
        regularization=regularization,
    )
    valid_rmse = np.sqrt(
        np.mean((VALID_R - predict_mf(candidate, VALID_U, VALID_I)) ** 2)
    )
    MF_CANDIDATES[regularization] = (candidate, valid_rmse)
    print(
        regularization,
        round(candidate["history"][-1], 4),
        round(valid_rmse, 4),
    )

SELECTED_LAMBDA = min(
    MF_CANDIDATES,
    key=lambda value: MF_CANDIDATES[value][1],
)
MF = MF_CANDIDATES[SELECTED_LAMBDA][0]
print("selected lambda:", SELECTED_LAMBDA)
```

预期输出中的三列依次是 $\lambda$、训练 RMSE、验证 RMSE：

```text
0.005 0.2713 0.6869
0.02 0.2793 0.6632
0.05 0.302 0.6313
0.15 0.4125 0.6157
selected lambda: 0.15
```

训练误差随正则变强而上升，但验证误差下降。超参数用验证集选择后，测试集只打开一次。

```python
for name, mask in [
    ("warm", TEST_I < N_ITEMS - 8),
    ("cold", TEST_I >= N_ITEMS - 8),
    ("all", np.ones(len(TEST_I), dtype=bool)),
]:
    rmse = np.sqrt(
        np.mean(
            (
                TEST_R[mask]
                - predict_mf(MF, TEST_U[mask], TEST_I[mask])
            ) ** 2
        )
    )
    print(name, round(rmse, 4), int(mask.sum()))
```

预期输出：

```text
warm 0.6895 144
cold 1.4845 72
all 1.0254 216
```

总体 RMSE 把 144 个 warm 与 72 个 cold 混在一起。cold 卡片的 $\mathbf q_i$ 和 $b_i$ 从未被训练数据更新，纯 ID 模型没有资格声称理解它们。

<MlQuiz
  kind="SGD 实现"
  question="同一条评分里先更新 p_u，再用新 p_u 更新 q_i，会发生什么？"
  :options="['数学上仍严格等于同时 SGD', '引入顺序依赖；若要实现推导出的同一步梯度，应保存 old_p', '自动变成 ALS', '消除非凸性']"
  :answer="1"
  explanation="两个梯度都在更新前参数点求值；原地覆盖会混用两个时间点。"
/>

### 6.1 有限差分不是装饰

对 $\mathbf p_u$ 第 $d$ 维，中心差分

$$
\frac{\ell(\mathbf p_u+\epsilon\mathbf e_d)
-\ell(\mathbf p_u-\epsilon\mathbf e_d)}{2\epsilon}
$$

应与解析梯度一致。检查单样本损失时，正则项的计数必须和解析式相同。

```python
def explicit_single_loss(
    pu, qi, bu, bi, mu, rating, regularization
):
    error = rating - mu - bu - bi - pu @ qi
    return (
        0.5 * error ** 2
        + 0.5 * regularization
        * (pu @ pu + qi @ qi + bu ** 2 + bi ** 2)
    )


u0, i0, r0 = int(TRAIN_U[0]), int(TRAIN_I[0]), float(TRAIN_R[0])
pu0 = MF["p"][u0].copy()
qi0 = MF["q"][i0].copy()
error0 = (
    r0 - MF["mu"] - MF["bu"][u0] - MF["bi"][i0] - pu0 @ qi0
)
analytical = -error0 * qi0 + SELECTED_LAMBDA * pu0
numeric = np.empty_like(pu0)
epsilon = 1e-6
for dimension in range(len(pu0)):
    plus, minus = pu0.copy(), pu0.copy()
    plus[dimension] += epsilon
    minus[dimension] -= epsilon
    numeric[dimension] = (
        explicit_single_loss(
            plus, qi0, MF["bu"][u0], MF["bi"][i0],
            MF["mu"], r0, SELECTED_LAMBDA,
        )
        - explicit_single_loss(
            minus, qi0, MF["bu"][u0], MF["bi"][i0],
            MF["mu"], r0, SELECTED_LAMBDA,
        )
    ) / (2 * epsilon)

EXPLICIT_GRADIENT_ERROR = np.max(np.abs(analytical - numeric))
print("explicit gradient max error:", EXPLICIT_GRADIENT_ERROR)
```

预期误差约为 $1.23\times10^{-11}$，远小于教学门限 $10^{-6}$。

## 7. 潜空间的几何与不可识别性

若 $\mathbf A$ 可逆，

$$
\mathbf P\mathbf Q^\top
=
(\mathbf P\mathbf A)
\left(\mathbf Q\mathbf A^{-\top}\right)^\top.
$$

所以同一预测矩阵对应无穷多组坐标。特别地，正交 $\mathbf A$ 表示共同旋转或反射，向量范数与点积都不变；还可以交换维度或同时翻转一维符号。潜因子可以帮助发现结构，却不能未经外部验证就命名为确定的物理概念。

尺度也有耦合：不加正则时，把 $\mathbf P$ 放大 $c$ 倍、把 $\mathbf Q$ 缩小 $c$ 倍，乘积不变。对称的 L2 正则会偏好更平衡的分解，但仍不消除旋转不确定性。

<LatentFactorAlignmentBenchLab />

<MlQuiz
  kind="可识别性"
  question="共同对 P、Q 做同一个正交旋转后，哪一项一定保持不变？"
  :options="['每个潜因子的业务名称', '所有用户—物品点积和由此得到的排序', '每个坐标的数值', '训练过程的每一步轨迹']"
  :answer="1"
  explanation="正交旋转满足 A Aᵀ=I，因此点积不变；坐标解释并不唯一。"
/>

::: warning 停下来推一推 2
<!-- ml-stop -->
报告把四维矩阵分解的第 2 维命名为“电气安全意识”，并据此给工程师画像。请构造一个共同旋转，使预测完全不变但第 2 维数值改变；再写出要获得可解释声明至少需要的外部锚点、稳定性检验和用途限制。
:::

## 8. ALS：固定一侧后是许多独立 Ridge 问题

先忽略偏置以看清主线。固定物品矩阵 $\mathbf Q$，对用户 $u$ 收集已评分物品 $\Omega_u$，令 $\mathbf Q_u$ 的每一行是对应 $\mathbf q_i^\top$，中心化评分向量为 $\widetilde{\mathbf r}_u$：

$$
\min_{\mathbf p_u}
\frac12\|\widetilde{\mathbf r}_u-\mathbf Q_u\mathbf p_u\|_2^2
+\frac\lambda2\|\mathbf p_u\|_2^2.
$$

令梯度为 0：

$$
-\mathbf Q_u^\top
(\widetilde{\mathbf r}_u-\mathbf Q_u\mathbf p_u)
+\lambda\mathbf p_u=0,
$$

于是

$$
\boxed{
\mathbf p_u
=
(\mathbf Q_u^\top\mathbf Q_u+\lambda\mathbf I)^{-1}
\mathbf Q_u^\top\widetilde{\mathbf r}_u
}.
$$

固定 $\mathbf P$ 后，对每个物品完全对称。偏置可作为设计矩阵中的常数列一起求解。工程实现使用 `solve(A, b)` 或 Cholesky，不显式计算 $A^{-1}$。

```python
def explicit_objective(model, users, items, ratings, regularization):
    errors = ratings - predict_mf(model, users, items)
    penalty = (
        np.sum(model["bu"] ** 2)
        + np.sum(model["bi"] ** 2)
        + np.sum(model["p"] ** 2)
        + np.sum(model["q"] ** 2)
    )
    return (
        0.5 * np.sum(errors ** 2)
        + 0.5 * regularization * penalty
    )


def als_sweep(model, users, items, ratings, regularization):
    updated = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in model.items()
    }
    factors = updated["p"].shape[1]
    for u in range(len(updated["bu"])):
        mask = users == u
        if not np.any(mask):
            continue
        design = np.column_stack([
            np.ones(mask.sum()), updated["q"][items[mask]]
        ])
        target = ratings[mask] - updated["mu"] - updated["bi"][items[mask]]
        system = design.T @ design + regularization * np.eye(factors + 1)
        solution = np.linalg.solve(system, design.T @ target)
        updated["bu"][u], updated["p"][u] = solution[0], solution[1:]
    for i in range(len(updated["bi"])):
        mask = items == i
        if not np.any(mask):
            continue
        design = np.column_stack([
            np.ones(mask.sum()), updated["p"][users[mask]]
        ])
        target = ratings[mask] - updated["mu"] - updated["bu"][users[mask]]
        system = design.T @ design + regularization * np.eye(factors + 1)
        solution = np.linalg.solve(system, design.T @ target)
        updated["bi"][i], updated["q"][i] = solution[0], solution[1:]
    return updated


rng = np.random.default_rng(2912)
ALS0 = {
    "mu": float(TRAIN_R.mean()),
    "bu": np.zeros(N_USERS),
    "bi": np.zeros(N_ITEMS),
    "p": rng.normal(0, 0.08, (N_USERS, 4)),
    "q": rng.normal(0, 0.08, (N_ITEMS, 4)),
}
ALS1 = als_sweep(
    ALS0, TRAIN_U, TRAIN_I, TRAIN_R, regularization=0.05
)
ALS_OBJECTIVES = [
    explicit_objective(ALS0, TRAIN_U, TRAIN_I, TRAIN_R, 0.05),
    explicit_objective(ALS1, TRAIN_U, TRAIN_I, TRAIN_R, 0.05),
]
print(*np.round(ALS_OBJECTIVES, 3))
```

预期输出：

```text
354.126 142.248
```

一次用户半步与物品半步都精确最小化各自条件目标，因此目标不增；但 $\mathbf P,\mathbf Q$ 联合目标仍非凸，不保证到达全局最优。多起点、验证曲线和数值条件数仍需审查。

<MlQuiz
  kind="ALS 边界"
  question="ALS 每一半步是凸 Ridge 问题，为什么仍不能宣布联合全局最优？"
  :options="['因为 Ridge 没有闭式解', 'P 与 Q 同时未知时含双线性乘积，联合目标非凸', '因为 ALS 不使用评分', '因为矩阵必须是对称的']"
  :answer="1"
  explanation="固定一块参数后条件凸，不等于两块参数联合凸；不同初始化可到不同驻点。"
/>

## 9. 置信加权隐式 ALS：偏好与证据强度分开

对于行为计数 $r_{ui}\ge0$，定义

$$
p_{ui}=\mathbb I[r_{ui}>0],\qquad
c_{ui}=1+\alpha r_{ui}.
$$

$p_{ui}$ 是二值偏好指示，$c_{ui}$ 是对这个指示的信心。未交互格也进入目标，但只有基准权重 1：

$$
J
=\sum_{u,i}
c_{ui}(p_{ui}-\mathbf x_u^\top\mathbf y_i)^2
+\lambda
\left(
\sum_u\|\mathbf x_u\|^2+\sum_i\|\mathbf y_i\|^2
\right).
$$

令 $\mathbf C_u=\operatorname{diag}(c_{u1},\ldots,c_{um})$，$\mathbf p(u)$ 收集用户 $u$ 的二值偏好。固定 $\mathbf Y$：

$$
\boxed{
\mathbf x_u
=
(\mathbf Y^\top\mathbf C_u\mathbf Y+\lambda\mathbf I)^{-1}
\mathbf Y^\top\mathbf C_u\mathbf p(u)
}.
$$

乍看要遍历所有 $m$ 个物品。利用

$$
\mathbf Y^\top\mathbf C_u\mathbf Y
=
\mathbf Y^\top\mathbf Y
+\mathbf Y^\top(\mathbf C_u-\mathbf I)\mathbf Y,
$$

第一项可全局预计算，第二项只在 $c_{ui}-1\ne0$ 的交互物品上累加；这正是稀疏隐式 ALS 能扩展的关键。

```python
Y_SMALL = np.array([
    [0.8, 0.1],
    [0.3, 0.7],
    [-0.2, 0.9],
    [0.6, -0.4],
    [0.1, 0.8],
])
COUNTS = np.array([3, 0, 1, 0, 4], dtype=float)
PREFERENCE = (COUNTS > 0).astype(float)
CONFIDENCE = 1 + 2 * COUNTS
DIRECT_SYSTEM = (
    Y_SMALL.T @ (CONFIDENCE[:, None] * Y_SMALL)
    + 0.2 * np.eye(2)
)
RIGHT_SIDE = Y_SMALL.T @ (CONFIDENCE * PREFERENCE)
DIRECT_USER = np.linalg.solve(DIRECT_SYSTEM, RIGHT_SIDE)
FAST_SYSTEM = (
    Y_SMALL.T @ Y_SMALL
    + Y_SMALL.T @ ((CONFIDENCE - 1)[:, None] * Y_SMALL)
    + 0.2 * np.eye(2)
)
FAST_USER = np.linalg.solve(FAST_SYSTEM, RIGHT_SIDE)
print(np.round(DIRECT_USER, 4))
print("direct-fast max error:", np.max(np.abs(DIRECT_USER - FAST_USER)))
```

预期输出：

```text
[0.9601 1.0887]
direct-fast max error: 0.0
```

这个目标仍把未交互项当低置信负偏好。若大量未交互其实从未曝光，权重虽低，数量优势仍可能压过正例；曝光建模不能被“低权重”四个字消除。

<MlQuiz
  kind="隐式 ALS"
  question="c_ui=1+αr_ui 中的 1 有什么作用？"
  :options="['把所有未交互项设成高置信正例', '给未交互项保留基准权重，使全矩阵都有弱偏好约束', '保证矩阵秩等于 1', '替代正则项']"
  :answer="1"
  explanation="r=0 时 c=1、p=0；正行为随次数获得更高信心，但未行为仍作为弱约束存在。"
/>

## 10. BPR：从“正例应在未观察项前”推导成对似然

### 10.1 训练样本不是单个格，而是三元组

对用户 $u$，令 $I_u^+$ 是训练期正行为物品集合。BPR 构造

$$
\mathcal D_S
=
\{(u,i,j):i\in I_u^+,\ j\in\mathcal I\setminus I_u^+\}.
$$

三元组表达相对偏好 $i>_u j$。它仍包含假设：未观察的 $j$ 被当作**采样得到的暂定负例**，而不是已证实厌恶。若 $j$ 从未曝光，三元组可能是假负例。

定义分数与成对间隔：

$$
s_{ui}=b_i+\mathbf p_u^\top\mathbf q_i,\qquad
\widehat x_{uij}=s_{ui}-s_{uj}
=b_i-b_j+\mathbf p_u^\top(\mathbf q_i-\mathbf q_j).
$$

用 logistic 链接表示成对事件：

$$
\Pr(i>_u j\mid\Theta)=\sigma(\widehat x_{uij}).
$$

假设采样三元组条件独立，并给参数零均值高斯先验，MAP 目标为

$$
\boxed{
\max_\Theta
\sum_{(u,i,j)\in\mathcal D_S}
\log\sigma(\widehat x_{uij})
-\frac{\lambda}{2}\|\Theta\|_2^2
}.
$$

这和按用户计算 AUC 的“正例分数高于负例”结构相近，但训练时的用户、正例和负例采样分布决定了每类 pair 的实际权重。

<MlQuiz
  kind="BPR 样本"
  question="BPR 三元组 (u,i,j) 直接声明了什么？"
  :options="['i 的绝对评分必须是 5', '对用户 u，已观测正例 i 应排在采样未观察项 j 前', 'j 已经被用户明确否决', '所有用户共享同一排序']"
  :answer="1"
  explanation="BPR 优化相对次序；未观察项只是训练假设下的暂定负例，不是可靠的显式负标签。"
/>

### 10.2 为什么用户偏置消失、物品偏置保留

若原分数含 $\mu+b_u+b_i+\mathbf p_u^\top\mathbf q_i$，做差时

$$
(\mu+b_u+b_i+\mathbf p_u^\top\mathbf q_i)
-
(\mu+b_u+b_j+\mathbf p_u^\top\mathbf q_j)
$$

中的全局均值和用户偏置完全抵消；物品偏置 $b_i-b_j$ 仍影响次序。这不是“BPR 忘了用户”，而是同一用户的常数不能改变该用户内部排序。

令

$$
\gamma_{uij}=\sigma(-\widehat x_{uij}).
$$

因为

$$
\frac{\partial}{\partial x}\log\sigma(x)=\sigma(-x),
$$

单个三元组的对数后验上升方向为

$$
\begin{aligned}
\Delta\mathbf p_u
&\propto
\gamma_{uij}(\mathbf q_i-\mathbf q_j)-\lambda\mathbf p_u,\\
\Delta\mathbf q_i
&\propto
\gamma_{uij}\mathbf p_u-\lambda\mathbf q_i,\\
\Delta\mathbf q_j
&\propto
-\gamma_{uij}\mathbf p_u-\lambda\mathbf q_j,\\
\Delta b_i&\propto\gamma_{uij}-\lambda b_i,\\
\Delta b_j&\propto-\gamma_{uij}-\lambda b_j.
\end{aligned}
$$

若间隔为负或很小，$\gamma$ 大，更新强；若正例已远高于负例，$\gamma$ 接近 0，只剩正则收缩。

<MlQuiz
  kind="成对抵消"
  question="在同一用户的 BPR 分数差 s_ui−s_uj 中，哪一项必然抵消？"
  :options="['物品偏置 b_i', '用户偏置 b_u 和全局常数 μ', '物品向量 q_i', '负例分数 s_uj']"
  :answer="1"
  explanation="对同一用户，两边共享 μ+b_u；它们不影响用户内部物品排序。"
/>

<BprPairwiseSorterLab />

### 10.3 负采样定义了模型看见的反例

不可能枚举全部 $\mathcal D_S$。常见策略：

| 负采样 | 优点 | 风险 |
|---|---|---|
| 对未观察项均匀采样 | 简单、覆盖长尾 | 大量样本太容易，梯度信息弱 |
| 按热门度采样 | 更常比较竞争激烈物品 | 强化头部；把高曝光未点击和未曝光混在一起 |
| 困难负例 | 训练集中于当前高分错误 | 假负例破坏更大，且需额外检索 |
| 只从“已曝光未行动”采样 | 语义更接近弱负反馈 | 覆盖旧策略支持集，仍有位置和上下文偏差 |

负例分布、每个正例的采样数、是否排除同会话未来正例都必须版本化。训练目标的期望是对**采样分布**取的，不同 sampler 不是同一个实验细节。

<MlQuiz
  kind="负采样"
  question="把均匀负采样改成困难负例后，最需要同步记录什么？"
  :options="['只记录最终 loss', 'sampler 版本、候选域、曝光条件与假负例审计', '删除所有热门物品', '把学习率固定为 1']"
  :answer="1"
  explanation="采样分布改变优化权重和风险；必须能重放候选域并审计未曝光正潜力项。"
/>

## 11. 同一日志上的热门、pointwise 与 BPR

先把训练期评分 $\ge4$ 的卡视为正行为；所有训练曝光都从未来候选集中排除。测试相关项同样由未来评分 $\ge4$ 定义。没有未来相关项的用户不进入 Recall/NDCG 平均，但数量必须报告。

```python
POSITIVE = np.zeros((N_USERS, N_ITEMS), dtype=bool)
POSITIVE[TRAIN_U[TRAIN_R >= 4], TRAIN_I[TRAIN_R >= 4]] = True
TRAIN_OBSERVED = np.zeros_like(POSITIVE)
TRAIN_OBSERVED[TRAIN_U, TRAIN_I] = True
TEST_RELEVANT = np.zeros_like(POSITIVE)
TEST_RELEVANT[TEST_U[TEST_R >= 4], TEST_I[TEST_R >= 4]] = True
print(
    "positive/user quantiles:",
    np.quantile(POSITIVE.sum(axis=1), [0, 0.5, 1]),
)
print("evaluation users:", int(np.sum(TEST_RELEVANT.sum(axis=1) > 0)))
```

预期输出：

```text
positive/user quantiles: [ 1.  7. 11.]
evaluation users: 59
```

pointwise 基线把正例与从未观察集合抽到的暂定负例做 logistic 分类；BPR 用同样潜因子容量，但一次优化一个正负 pair。

```python
def all_scores(model):
    return (
        model["p"] @ model["q"].T
        + model.get(
            "bi", np.zeros(model["q"].shape[0])
        )[None, :]
    )


def fit_pointwise(
    positive, steps=45000, factors=8,
    learning_rate=0.035, regularization=0.003, seed=2921,
):
    rng = np.random.default_rng(seed)
    n_users, n_items = positive.shape
    p = rng.normal(0, 0.08, (n_users, factors))
    q = rng.normal(0, 0.08, (n_items, factors))
    bi = np.zeros(n_items)
    positives = [np.flatnonzero(positive[u]) for u in range(n_users)]
    negatives = [np.flatnonzero(~positive[u]) for u in range(n_users)]
    eligible = np.flatnonzero([len(values) > 0 for values in positives])
    for _ in range(steps):
        u = int(rng.choice(eligible))
        label = int(rng.random() < 0.5)
        i = int(rng.choice(
            positives[u] if label else negatives[u]
        ))
        score = bi[i] + p[u] @ q[i]
        gradient = sigmoid(score) - label
        old_p = p[u].copy()
        p[u] -= learning_rate * (
            gradient * q[i] + regularization * p[u]
        )
        q[i] -= learning_rate * (
            gradient * old_p + regularization * q[i]
        )
        bi[i] -= learning_rate * (
            gradient + regularization * bi[i]
        )
    return {"p": p, "q": q, "bi": bi}


def fit_bpr(
    positive, steps=90000, factors=8,
    learning_rate=0.01, regularization=0.003,
    seed=2922, negative_mode="uniform",
):
    rng = np.random.default_rng(seed)
    n_users, n_items = positive.shape
    p = rng.normal(0, 0.08, (n_users, factors))
    q = rng.normal(0, 0.08, (n_items, factors))
    bi = np.zeros(n_items)
    positives = [np.flatnonzero(positive[u]) for u in range(n_users)]
    negatives = [np.flatnonzero(~positive[u]) for u in range(n_users)]
    eligible = np.flatnonzero([len(values) > 0 for values in positives])
    popularity = positive.sum(axis=0).astype(float) + 1
    for _ in range(steps):
        u = int(rng.choice(eligible))
        i = int(rng.choice(positives[u]))
        negative_pool = negatives[u]
        if negative_mode == "popular":
            weights = popularity[negative_pool]
            j = int(rng.choice(
                negative_pool, p=weights / weights.sum()
            ))
        else:
            j = int(rng.choice(negative_pool))
        gap = bi[i] - bi[j] + p[u] @ (q[i] - q[j])
        multiplier = sigmoid(-gap)
        old_p = p[u].copy()
        old_qi = q[i].copy()
        old_qj = q[j].copy()
        p[u] += learning_rate * (
            multiplier * (old_qi - old_qj)
            - regularization * old_p
        )
        q[i] += learning_rate * (
            multiplier * old_p - regularization * old_qi
        )
        q[j] += learning_rate * (
            -multiplier * old_p - regularization * old_qj
        )
        bi[i] += learning_rate * (
            multiplier - regularization * bi[i]
        )
        bi[j] += learning_rate * (
            -multiplier - regularization * bi[j]
        )
    return {"p": p, "q": q, "bi": bi}


POINT = fit_pointwise(POSITIVE)
BPR = fit_bpr(POSITIVE)
BPR_POPNEG = fit_bpr(POSITIVE, negative_mode="popular")
POPULARITY = POSITIVE.sum(axis=0).astype(float)
```

### 11.1 先把评价函数冻结

对用户 $u$，推荐列表 $\pi_u(1{:}K)$ 与未来相关集合 $\mathcal R_u$：

$$
\operatorname{Recall@K}_u
=
\frac{|\pi_u(1{:}K)\cap\mathcal R_u|}
{|\mathcal R_u|}.
$$

二值相关性的

$$
\operatorname{DCG@K}_u
=
\sum_{r=1}^{K}
\frac{\mathbb I[\pi_u(r)\in\mathcal R_u]}{\log_2(r+1)},
\qquad
\operatorname{NDCG@K}_u
=\frac{\operatorname{DCG@K}_u}{\operatorname{IDCG@K}_u}.
$$

catalog coverage 是所有用户 Top-K 并集中不同物品数除以全库物品数。它不衡量相关性，却能发现榜单塌到少数头部。

```python
def rank_metrics(scores, relevant, excluded, k=5):
    recalls, ndcgs, recommended = [], [], set()
    per_user = []
    for u in range(scores.shape[0]):
        truth = np.flatnonzero(relevant[u] & ~excluded[u])
        if len(truth) == 0:
            continue
        candidate_scores = scores[u].copy()
        candidate_scores[excluded[u]] = -np.inf
        order = np.argsort(
            -candidate_scores, kind="stable"
        )[:k]
        hits = np.isin(order, truth)
        recall = hits.sum() / len(truth)
        dcg = np.sum(hits / np.log2(np.arange(2, k + 2)))
        ideal = np.sum(
            1 / np.log2(np.arange(2, min(k, len(truth)) + 2))
        )
        recalls.append(recall)
        ndcgs.append(dcg / ideal)
        recommended.update(order.tolist())
        per_user.append((u, recall, dcg / ideal, order))
    return {
        "recall": float(np.mean(recalls)),
        "ndcg": float(np.mean(ndcgs)),
        "coverage": len(recommended) / scores.shape[1],
        "per_user": per_user,
    }


MODELS = {
    "popular": np.tile(POPULARITY, (N_USERS, 1)),
    "pointwise": all_scores(POINT),
    "bpr_uniform": all_scores(BPR),
    "bpr_popneg": all_scores(BPR_POPNEG),
}
FULL_METRICS = {}
for name, scores in MODELS.items():
    metric = rank_metrics(
        scores, TEST_RELEVANT, TRAIN_OBSERVED, k=5
    )
    FULL_METRICS[name] = metric
    print(name, {
        key: round(metric[key], 4)
        for key in ["recall", "ndcg", "coverage"]
    })
```

预期输出：

```text
popular {'recall': 0.3305, 'ndcg': 0.227, 'coverage': 0.1875}
pointwise {'recall': 0.3305, 'ndcg': 0.2366, 'coverage': 0.6094}
bpr_uniform {'recall': 0.3446, 'ndcg': 0.2645, 'coverage': 0.5781}
bpr_popneg {'recall': 0.2853, 'ndcg': 0.1977, 'coverage': 0.7812}
```

在这一次冻结数据和超参数下，均匀 BPR 的 Recall/NDCG 最好；这不是“BPR 永远胜出”。热门加权负采样覆盖更多卡片却降低相关性，说明 sampler 改变了训练重点。pointwise 与热门 Recall 相同但 NDCG 更高，意味着命中的未来相关卡被排得更靠前。

<MlQuiz
  kind="同源对比"
  question="看到本表中 BPR 的 NDCG@5 最高，可以得出的最强结论是什么？"
  :options="['BPR 对所有推荐任务都最好', '在冻结数据、候选池、超参数和本次随机种子下，均匀 BPR 的该指标最高', 'BPR 已消除曝光偏差', '线上安全性已经证明']"
  :answer="1"
  explanation="算法比较只覆盖当前协议；统计不确定性、线上因果效果、冷启动和安全门仍需另行验证。"
/>

### 11.2 少量采样负例会制造漂亮的离线数字

若每个未来正例只与 5 或 10 个随机负例竞争，任务远比全库 Top-5 简单。下面保持模型不变，只缩小评价候选池。

```python
def sampled_candidate_recall(
    scores, relevant, excluded,
    negatives=10, k=5, seed=2944,
):
    rng = np.random.default_rng(seed)
    recalls = []
    for u in range(scores.shape[0]):
        truth = np.flatnonzero(relevant[u] & ~excluded[u])
        if len(truth) == 0:
            continue
        pool = np.flatnonzero(~excluded[u] & ~relevant[u])
        sampled = rng.choice(
            pool, size=min(negatives, len(pool)), replace=False
        )
        candidates = np.unique(np.concatenate([truth, sampled]))
        order = candidates[np.argsort(
            -scores[u, candidates], kind="stable"
        )[:k]]
        recalls.append(np.isin(order, truth).sum() / len(truth))
    return float(np.mean(recalls))


SAMPLED_RECALLS = {}
for negative_count in [5, 10, 30]:
    value = sampled_candidate_recall(
        MODELS["bpr_uniform"],
        TEST_RELEVANT,
        TRAIN_OBSERVED,
        negatives=negative_count,
    )
    SAMPLED_RECALLS[negative_count] = value
    print(negative_count, round(value, 4))
```

预期输出：

```text
5 0.9605
10 0.7938
30 0.5226
```

同一个 BPR 模型的全库 Recall@5 是 0.3446；只抽 5 个负例时变成 0.9605。报告若不写候选池，就没有可比较的 Recall。

<MlQuiz
  kind="候选池"
  question="为什么“1 个正例 + 5 个随机负例”的 Recall@5 通常严重偏高？"
  :options="['因为 Recall 不能用于二值相关性', '几乎所有候选都能进入前 5，且避开了全库中大量强竞争项', '因为随机数一定泄漏', '因为 K 必须等于 1']"
  :answer="1"
  explanation="缩小候选池改变了任务难度；指标名称相同也不能和全库检索直接比较。"
/>

### 11.3 总体指标之后必须看头部、长尾和冷启动

```python
item_popularity = POSITIVE.sum(axis=0)
warm_popularity = item_popularity[:N_ITEMS - 8]
head_threshold = np.median(warm_popularity)
STRATUM_METRICS = {}
for label, item_mask in [
    (
        "head",
        (item_popularity > head_threshold)
        & (np.arange(N_ITEMS) < N_ITEMS - 8),
    ),
    (
        "tail",
        (item_popularity <= head_threshold)
        & (np.arange(N_ITEMS) < N_ITEMS - 8),
    ),
    ("cold", np.arange(N_ITEMS) >= N_ITEMS - 8),
]:
    stratum_relevant = TEST_RELEVANT & item_mask[None, :]
    metric = rank_metrics(
        MODELS["bpr_uniform"],
        stratum_relevant,
        TRAIN_OBSERVED,
        k=5,
    )
    STRATUM_METRICS[label] = metric
    print(
        label,
        int(stratum_relevant.sum()),
        round(metric["recall"], 4),
    )
```

预期输出：

```text
head 58 0.4545
tail 20 0.1
cold 3 0.0
```

BPR 的总体 0.3446 主要来自头部。cold Recall 为 0 不只是“数据少”：训练时所有未观察项都可能被抽作负例，新卡甚至会在没有曝光证据时被向下推。此处必须触发内容模型、受控探索或热门回退，不能发布一个总体数字掩盖盲区。

<MlQuiz
  kind="切片门禁"
  question="总体 Recall@5 提升，但 cold Recall@5 仍为 0，最合适的处理是什么？"
  :options="['用总体均值抵消 cold 失败', '把 cold 从报告删除', '为 cold 流量设置独立回退、内容特征/探索方案与发布门禁', '把所有 cold 卡标为负例']"
  :answer="2"
  explanation="冷启动是结构性适用域缺口，需要独立策略；总体平均不能创造不存在的 ID 证据。"
/>

### 11.4 再检查一次 BPR 梯度

```python
def bpr_single_loss(
    pu, qi, qj, bi, bj, regularization=0.003
):
    pair_gap = bi - bj + pu @ (qi - qj)
    return (
        -math.log(float(sigmoid(pair_gap)))
        + 0.5 * regularization
        * (
            pu @ pu + qi @ qi + qj @ qj
            + bi ** 2 + bj ** 2
        )
    )


check_u = int(np.flatnonzero(POSITIVE.sum(axis=1) > 0)[0])
check_i = int(np.flatnonzero(POSITIVE[check_u])[0])
check_j = int(np.flatnonzero(~POSITIVE[check_u])[0])
check_p = BPR["p"][check_u].copy()
check_qi = BPR["q"][check_i].copy()
check_qj = BPR["q"][check_j].copy()
check_bi = float(BPR["bi"][check_i])
check_bj = float(BPR["bi"][check_j])
check_gap = (
    check_bi - check_bj
    + check_p @ (check_qi - check_qj)
)
ascent_analytical = (
    sigmoid(-check_gap) * (check_qi - check_qj)
    - 0.003 * check_p
)
ascent_numeric = np.empty_like(check_p)
for dimension in range(len(check_p)):
    plus, minus = check_p.copy(), check_p.copy()
    plus[dimension] += 1e-6
    minus[dimension] -= 1e-6
    ascent_numeric[dimension] = -(
        bpr_single_loss(
            plus, check_qi, check_qj, check_bi, check_bj
        )
        - bpr_single_loss(
            minus, check_qi, check_qj, check_bi, check_bj
        )
    ) / (2e-6)

BPR_GRADIENT_ERROR = np.max(
    np.abs(ascent_analytical - ascent_numeric)
)
print("BPR gradient max error:", BPR_GRADIENT_ERROR)
```

预期误差约为 $7.83\times10^{-11}$。这里比较的是**对数后验上升方向**，所以数值梯度对负损失又取了一次负号；符号必须在测试名里写清。

## 12. 离线切分：模拟下一次请求，而不是打乱历史

推荐交互有时间方向。随机按行切分会产生四类泄漏：

1. 同一会话前半段进测试、后半段进训练；
2. 未来形成的用户向量帮助预测过去；
3. 新版本卡片的未来受欢迎度泄漏到训练；
4. 用未来目录状态过滤过去候选集。

本案例把每名用户的事件序列按时间切成 12/3/3，并把 cold 物品只放入测试。真实系统还应做多个滚动窗口：

$$
\text{train}_{\le t_1}\to\text{valid}_{(t_1,t_2]}
\to\text{test}_{(t_2,t_3]},
$$

检查节假日、班次、设备代际和策略版本。若产品是“下一会话推荐”，同一会话必须整体属于一个集合；若产品是“下一动作推荐”，可保留会话前缀，但特征只能使用预测时刻之前的信息。

测试相关性仍是历史策略选择后的结果。时间切分避免未来信息泄漏，却不会自动消除曝光选择偏差。

<MlQuiz
  kind="时间切分"
  question="同一工单会话的前 5 次卡片交互进训练、最后 1 次进测试，何时最危险？"
  :options="['特征只含会话开始前的静态字段', '训练特征或用户向量汇总了整段会话，包括测试动作', '测试只计算 Recall@5', '候选集少于 64']"
  :answer="1"
  explanation="整会话汇总会把未来动作泄漏进训练表示；切分粒度和特征截止时间必须一起冻结。"
/>

## 13. 冷启动不是一个问题，而是三种适用域缺口

| 缺口 | 纯 ID 模型缺少什么 | 可用方案 |
|---|---|---|
| 新用户 | $\mathbf p_u$ 没有交互可估计 | onboarding、角色/设备上下文、热门与受控探索 |
| 新物品 | $\mathbf q_i,b_i$ 没有交互可估计 | 文本/元数据编码、专家先验、内容近邻、探索流量 |
| 短历史用户/物品 | 估计方差很大 | 更强正则、层级收缩、fold-in、混合排序 |

`fold-in` 指固定已训练好的另一侧因子，用少量新交互解一个 Ridge 用户向量；它不会更新整个模型，适合快速接入已有少量历史的用户。对完全无交互的新物品，仍需要内容映射 $f_\phi(\text{content}_i)\approx\mathbf q_i$ 或独立的内容召回。

混合分数可以写成

$$
s_{ui}
=
\alpha_i\,s_{ui}^{\text{collab}}
+(1-\alpha_i)\,s_{ui}^{\text{content}},
$$

其中 $\alpha_i$ 随物品交互量增长。这个门控函数需要验证，不能用“冷启动时自动切换”一句话替代。

本案例 cold RMSE 1.4845、cold Recall@5 为 0，是适用域报警，不是调大潜因子维数就能修复的训练不足。

## 14. 曝光偏差与反事实评价

### 14.1 观察到的是旧策略选择后的条件分布

令 $E_{ui}$ 表示是否曝光，$Y_{ui}$ 表示曝光后是否产生正行为。日志主要提供

$$
P(Y_{ui}=1\mid E_{ui}=1,\text{旧策略}),
$$

而我们想知道新策略让不同物品曝光时会获得什么结果。热门、位置靠前和旧模型高分物品更容易进入 $E=1$；因此“被点击次数多”混合了偏好与曝光机会。

若旧策略 $\mu(a\mid x)$ 的每次展示概率被准确记录，新策略 $\pi(a\mid x)$ 的期望价值可用逆倾向加权估计：

$$
\widehat V_{\mathrm{IPS}}(\pi)
=
\frac1n\sum_{t=1}^{n}
\frac{\pi(A_t\mid X_t)}
{\mu(A_t\mid X_t)}
R_t.
$$

对确定性新策略，分子在日志动作等于新策略动作时为 1，否则为 0。这个公式不是免费去偏器，它至少要求：

- **一致性**：日志奖励对应实际执行的动作；
- **条件可交换性**：给定 $X$ 后，没有遗漏同时影响动作和奖励的混杂；
- **positivity/support**：新策略可能选择的动作在旧策略下有正概率；
- **倾向概率正确**：记录的是当时实际随机化概率，不是事后拟合分数；
- **稳定日志**：请求、位置、候选集、策略版本和奖励窗口能正确关联。

极小 $\mu$ 会产生巨大权重和方差。裁剪权重、自归一化 IPS 或 doubly robust 方法可减小方差，却引入额外偏差/模型假设；报告必须同时给权重分布、有效样本量和支持集覆盖。

<MlQuiz
  kind="反事实支持"
  question="旧策略从未向某类工程师展示电气卡，新策略却总想展示。为什么 IPS 无法可靠评价？"
  :options="['因为奖励必须连续', '该动作在这类上下文的旧策略概率为 0，违反 positivity，权重没有定义', '因为 IPS 只适合分类', '因为新策略太简单']"
  :answer="1"
  explanation="日志在该上下文没有动作支持，任何加权都不能凭空制造反事实结果；需要受控探索或新实验。"
/>

::: warning 停下来推一推 3
<!-- ml-stop -->
旧策略只在 Top-1 位置以 0.98 概率展示热门卡，以 0.02 概率探索长尾。新策略大量选择长尾。请写出 IPS 权重范围、说明方差为何暴涨，并比较三种选择：扩大安全探索、裁剪权重、拒绝在无支持区域发布。指出每种选择引入的成本或偏差。
:::

### 14.2 反馈回路会让模型预测改变训练分布

$$
\text{模型}\rightarrow\text{候选与排序}
\rightarrow\text{曝光}\rightarrow\text{行为}
\rightarrow\text{下一版训练数据}\rightarrow\text{模型}.
$$

如果热门卡先得到更多曝光，它就更容易积累正例，下一版又更热门；长尾卡没有数据，不代表没有价值。缓解方案包括：

- 保留按安全约束分层的随机探索桶，并记录真实 propensity；
- 把探索与生产主流量分开监控，设置最大风险预算和人工撤回；
- 对位置偏差建模或随机化少量位置，而不是把 `position` 当普通特征蒙混过去；
- 监控曝光覆盖、重复率、主题集中度、用户/设备群体差异和新物品存活率；
- 定期用冻结策略或 interleaving/A/B 实验确认线上因果效果。

探索不能越过合规候选集，也不能在安全关键检修步骤上随机试错。可探索的是**合法知识卡的展示次序**，不是设备控制动作。

## 15. 从召回到排序：工业系统不是一次全库点积

典型请求路径：

```text
请求上下文
  → 合规/权限过滤
  → 多路召回（协同、内容、热门、探索）
  → 去重与版本过滤
  → 精排（MF/BPR/混合模型）
  → 多样性与硬约束重排
  → Top-K 曝光及 propensity 日志
```

候选生成决定“模型有机会看见什么”，精排只决定候选内部次序。若 relevant 卡没有被召回，ranker 再强也无法命中。离线评价因此至少拆成：

1. retrieval Recall@L：真相关项是否进入 $L$ 个候选；
2. ranking NDCG@K：进入候选后是否排到前面；
3. end-to-end Recall@K：完整链路结果；
4. eligibility violation：硬约束违规必须为 0。

近似最近邻可以加速 $\mathbf p_u^\top\mathbf q_i$ 检索，但物品偏置、实时上下文和业务约束可能需要改写索引或二阶段重排。ANN 的近似误差也要独立测量，不能归咎于排序模型。

## 16. 一份可审计的推荐资格包

发布前至少保存：

| 工件 | 必填内容 |
|---|---|
| 决策卡 | 用户、时刻、合法候选集、Top-K、人工权限和禁止自动化边界 |
| 数据卡 | 曝光/行为 schema、时间窗口、会话定义、去重、迟到事件、删除请求 |
| 缺失机制卡 | 未曝光、曝光未行动、日志丢失的区分与比例 |
| 模型卡 | 偏置、因子维数、正则、SGD/ALS、负采样器、seed、停止条件 |
| 评价卡 | 全库/召回候选池、排除规则、无相关用户处理、时间切分和滚动窗口 |
| 切片卡 | head、tail、cold、新用户、主题、角色、设备代际、班次 |
| 反事实卡 | logging policy、propensity、支持率、权重分布、裁剪规则 |
| 发布门禁 | 主指标下限、硬约束 0 违规、cold 回退、延迟/容量和撤回条件 |
| 运行监控 | 覆盖、集中度、重复率、漂移、反馈延迟、策略版本与探索预算 |

“Top-K 更好”不构成完整发布结论。至少还要验证：

- 与热门和现网基线做按用户/会话成对比较；
- 多个时间窗口方向一致，并报告置信区间；
- cold、新用户和长尾有明确回退，不因总体平均过门；
- 候选集与日志能重放，任何曝光能追到策略版本；
- 推荐不可直接触发维修结论或设备动作。

## 数值验证：把推导变成回归门禁

```python
# 数据与时间合同。
assert len(ROWS) == 72 * 18
assert len(TRAIN_R) == 864
assert len(VALID_R) == 216
assert len(TEST_R) == 216
assert np.all(TRAIN_I < N_ITEMS - 8)
assert np.isclose(np.mean(TEST_I >= N_ITEMS - 8), 1 / 3)
for user in range(N_USERS):
    user_rows = [row for row in ROWS if row[0] == user]
    train_positions = [row[2] for row in user_rows if row[3] == "train"]
    valid_positions = [row[2] for row in user_rows if row[3] == "valid"]
    test_positions = [row[2] for row in user_rows if row[3] == "test"]
    assert max(train_positions) < min(valid_positions) < min(test_positions)

# 优化与梯度合同。
assert SELECTED_LAMBDA == 0.15
assert EXPLICIT_GRADIENT_ERROR < 1e-6
assert BPR_GRADIENT_ERROR < 1e-6
assert ALS_OBJECTIVES[1] <= ALS_OBJECTIVES[0] + 1e-10
assert np.allclose(DIRECT_USER, FAST_USER)

# 评价候选池与切片合同。
assert len(FULL_METRICS["bpr_uniform"]["per_user"]) == 59
assert FULL_METRICS["bpr_uniform"]["ndcg"] > FULL_METRICS["popular"]["ndcg"]
assert SAMPLED_RECALLS[5] > SAMPLED_RECALLS[10] > SAMPLED_RECALLS[30]
assert STRATUM_METRICS["cold"]["recall"] == 0
assert np.isfinite(MODELS["bpr_uniform"]).all()

print("时间、冷启动、优化、梯度、候选池与切片门禁：全部通过")
```

预期输出：

```text
时间、冷启动、优化、梯度、候选池与切片门禁：全部通过
```

这些断言验证实现和本章数据合同，不证明真实线上偏好、因果收益或安全性。数据生成器中的 `utility` 只用于教学造数，训练和评价代码没有读取它。

## 复杂度与工程边界

| 操作 | 时间复杂度 | 主要内存 | 工业边界 |
|---|---:|---:|---|
| 显式 MF 单轮 SGD | $O(\lvert\Omega\rvert k)$ | $O((\lvert\mathcal U\rvert+\lvert\mathcal I\rvert)k)$ | 热门 ID 更新冲突、乱序和学习率需审计 |
| 显式 ALS 一轮 | 约 $O(\lvert\Omega\rvert k^2+(\lvert\mathcal U\rvert+\lvert\mathcal I\rvert)k^3)$ | 因子与局部法方程 | 条件数差时用稳定求解；联合仍非凸 |
| 隐式 ALS 一轮 | 稀疏优化后约 $O(\lvert S\rvert k^2+(\lvert\mathcal U\rvert+\lvert\mathcal I\rvert)k^3)$ | 因子、Gram 矩阵 | “所有未交互为弱负”仍是建模假设 |
| BPR 的 $T$ 次更新 | $O(Tk)$ | 因子与采样索引 | sampler 决定权重；并行原地更新可能冲突 |
| 单用户全库打分 | $O(\lvert\mathcal I\rvert k)$ | 可流式 $O(K)$ | 大库需召回/ANN，必须测召回损失 |
| 全库 Top-K 评价 | $O(\lvert\mathcal U\rvert\lvert\mathcal I\rvert)$ | 分块到 $O(B\lvert\mathcal I\rvert)$ | 少量 sampled negatives 不能替代全库指标 |
| IPS 评价 | $O(n)$ | 可流式 | 支持集缺口无法靠计算修复，小 propensity 方差大 |

还要注意：

- 评分是有序离散值，高斯平方损失是近似；若阈值语义重要，可用 ordinal 模型；
- 用户兴趣会漂移，静态 ID 向量不含时序状态；
- 共享账号、机器人行为、批量导入和重复刷新会污染隐式信号；
- 删除用户数据后，因子、缓存、ANN 索引和训练快照都要执行一致的数据治理；
- 把 embedding 当用户画像可能引出隐私和劳动管理风险，应最小化用途与访问权限。

## 17. 章节练习

### 基础推导

1. 从观测高斯似然和各向同性高斯先验完整推导偏置 MF 的正则平方目标，写出 $\lambda$ 与方差比的关系。
2. 对 $e_{ui}=r_{ui}-\widehat r_{ui}$ 重新推导 $b_u,b_i,\mathbf p_u,\mathbf q_i$ 的四组梯度，逐项核对符号。
3. 固定 $\mathbf Q$，从显式目标推导含用户偏置常数列的 ALS 法方程。
4. 从 $\mathbf Y^\top\mathbf C_u\mathbf Y$ 推导隐式 ALS 的稀疏分解，标明可预计算项和用户特有项。
5. 从 Bernoulli 成对似然推导 $\partial\log\sigma(x)/\partial x=\sigma(-x)$ 以及 BPR 五组更新。
6. 证明全局均值和用户偏置在同用户 pair 中抵消，解释为什么物品偏置不抵消。

### 实现与反例

7. 在显式 MF 中故意删除 `old_p`，用多个学习率比较梯度检查、收敛曲线和顺序敏感性。
8. 给零填充 SVD 反例增加缺失率扫描，画空白区预测均值与观测 RMSE，说明两者为何可能背离。
9. 对同一 $\mathbf P,\mathbf Q$ 施加随机正交旋转，验证预测不变；再比较各维与人工主题标签的相关系数如何变化。
10. 把 BPR 负采样改成曝光未行动、均匀未观察和困难负例三种版本，保持更新次数一致比较。
11. 实现 Precision@K、MAP@K 与 graded NDCG@K，明确无相关项用户如何进入分母。
12. 将 sampled negatives 从 5 扫到全库，画 Recall@5 随候选池增长的曲线。

### 工业资格审查

13. 为曝光与行为日志写一份 schema，必须支持请求级候选集、位置、propensity、策略版本、迟到事件和删除请求。
14. 设计三个滚动时间窗口，要求目录版本、会话边界和特征截止时间可重放。
15. 为新卡片设计混合冷启动：内容召回、协同分数门控、探索预算、人工撤回和独立 cold 门禁。
16. 给定一组极端 propensity，计算 IPS、自归一化 IPS 和裁剪 IPS；报告权重分布、有效样本量与偏差来源。
17. 写一页上线评审结论：即使 NDCG 提升，也必须单独处理候选硬约束、cold=0、长尾、隐私、延迟和反馈回路。

练习骨架：

```python
# TODO：保持全候选池，不得把未曝光项直接改成标签 0。
def evaluate_full_catalog(scores, relevant, excluded, k):
    raise NotImplementedError
```

## 来源与许可

- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 741–746 页，§22.1–§22.1.4：用于核对显式反馈、稀疏观测集、偏置矩阵分解、正则目标、SGD、ALS 与概率解释。
- 同书 PDF 第 747–751 页，§22.2–§22.4：用于核对隐式反馈、BPR 入口、侧信息、反馈回路与探索—利用边界。
- Steffen Rendle 等，*BPR: Bayesian Personalized Ranking from Implicit Feedback*，PDF 第 2–7 页，§3–§4.3：用于核对个性化排序三元组、成对 logistic 似然、MAP 目标、LearnBPR 采样与 BPR-MF 更新。
- 同文 PDF 第 8–10 页，§5–§7：用于核对与 pointwise/WR-MF 的评价差异、目标对齐和实验结论适用范围。
- Yifan Hu、Yehuda Koren、Chris Volinsky，*Collaborative Filtering for Implicit Feedback Datasets*，PDF 第 1–5 页：用于核对隐式反馈的无可靠负例、偏好/置信分离、加权 ALS 法方程和稀疏复杂度化简。
- 同文 PDF 第 7–10 页：用于核对全排序评价、热门/邻域/因子基线、head-tail 差异以及离线行为预测不等于发现价值的边界。

本章检修知识库案例的 72 名教学工程师、64 张合成知识卡、1296 条曝光评分、四类主题、8 张 cold 卡、时间切分、全部 NumPy 代码、17 道即时测验和三项专用 SVG 网页实验均为本站独立创作。它们不代表真实企业、人员能力、设备状态、检修规程、卡片质量、推荐收益或安全结论。没有复制、改编或再分发参考资料中的图形、数据、代码、案例和实验结果。

完整来源登记见 [课程来源与许可](./sources.md) 。

<MlLessonProgress
  storage-key="29-recommender-systems"
  lesson-id="29"
  next-href="/courses/machine-learning/30-mdp-bellman"
  next-label="下一章：马尔可夫决策过程与动态规划"
/>
