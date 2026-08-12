---
course: machine-learning
edition: industrial
title: AdaBoost 与梯度提升
description: 以锂离子电池极片涂布批次质量复核为贯穿案例，从指数损失完整推导 AdaBoost、训练误差界与间隔，再把梯度提升写成函数空间下降，并从一阶伪残差推进到二阶正则化树。
outline: [2, 3]
---

<MlLessonHero
  number="21"
  title="AdaBoost 与梯度提升"
  question="随机森林让许多树彼此不同后再平均；如果产线误差不是随机抖动，而是每一轮都留下有结构的欠拟合，下一棵树能否只修正当前模型最欠缺的方向？"
  duration="建议 780–900 分钟"
  :outcomes="[
    '从指数损失逐步推导 AdaBoost 的加权树桩、弱学习器票重、样本权重更新、归一化常数、训练错误上界和归一化间隔',
    '把 Gradient Boosting 写成训练样本函数值空间中的下降，从零实现回归树桩、平方/Huber 伪残差、学习率、前缀验证和早停',
    '由二阶 Taylor 近似推导正则化叶权重与切分增益，审计标签噪声、时间切分、概率校准、资源预算、复现制品和线上漂移'
  ]"
/>

第 20 章用 Bootstrap 和节点级特征抽样让树彼此不同，再通过平均降低高方差。Boosting 走的是另一条路：模型按顺序生长，第 $m$ 个弱学习器必须读取前 $m-1$ 轮留下的错误状态。它不是“把很多树投票”这么宽泛，而是**前向分阶段地优化一个加法函数**。

这条顺序依赖同时带来能力和风险。它能把许多粗糙树桩组合成细致边界，也会把错误标签、量测尖峰和验证集复用逐轮放大。工业实现必须能回答：这一轮在修谁、为什么走这一步、验证集在哪一轮要求停止、部署时如何重放完全相同的树序列。

## 贯穿案例：锂离子电池极片涂布批次质量复核

一条锂离子电池极片涂布线希望用每卷极片的过程摘要完成两项离线辅助任务：

1. 分类任务：预测该卷是否需要进入人工复检与返工评审；
2. 回归任务：估计离线量测得到的面密度偏差，用于安排补充取样。

模型不直接修改浆料配比、泵速、模头间隙、烘箱温度、张力或安全联锁，也不自动放行产品。分类标签来自完成量测和质量评审后的冻结结论；回归标签来自离线面密度测量。任何在裁切、复检或返工之后才产生的字段都不能回填到训练特征。

### 数据合同、时间边界与动作代价

| 字段 | 工艺语义 | 冻结边界 |
|---|---|---|
| `solids_z` | 浆料固含量相对配方中心的标准化偏差 | 取样时刻、测量方法和配方族必须版本化 |
| `viscosity_z` | 上机前黏度偏差 | 温度补偿和等待时间改变语义 |
| `line_speed_z` | 主涂布段线速度偏差 | 只能使用预测时已经发生的设定与实测 |
| `die_gap_z` | 模头间隙偏差 | 校准批次和机械维护后基准不可混用 |
| `zone2_temp_z` | 第二烘箱区温度偏差 | 传感器位置、滤波和停线段剔除规则要冻结 |
| `web_tension_z` | 极片张力偏差 | 饱和、断带和换卷状态必须单独标记 |
| `dew_point_z` | 干燥环境露点偏差 | 新风系统和季节改变会造成分布漂移 |
| `filter_age_z` | 过滤器累计使用强度 | 更换记录延迟会形成时间穿越 |
| `campaign_id` | 同一配方与连续生产周期 | 训练、验证、测试按时间连续 campaign 切分 |
| `rework_required` | 完成复核后的返工评审标签 | 待判、复合缺陷和标签成熟时间必须保留 |
| `coat_weight_deviation` | 面密度相对目标的离线偏差 | 仪器、取样位置和重复测量聚合规则要冻结 |

漏报返工风险可能让异常卷进入后续工序；误报会增加复检与停留时间。因此本章用准确率演示算法，但验收还要报告召回率、精确率、复检容量、单位卷成本和按 campaign 的区间。面密度回归同时报告 RMSE、MAE、尾部误差与运行域拒绝率。

### 1. 按未来 campaign 冻结数据

下面生成 40 个连续教学 campaign、每个 6 卷，共 240 条合成记录。前 24 个 campaign 用于拟合，接着 8 个用于轮数和超参数选择，最后 8 个只做一次冻结测试。

```python
import numpy as np

rng = np.random.default_rng(20260821)
campaign_count = 40
rolls_per_campaign = 6
roll_count = campaign_count * rolls_per_campaign
campaign_id = np.repeat(np.arange(campaign_count), rolls_per_campaign)
feature_names = np.array([
    "solids_z", "viscosity_z", "line_speed_z", "die_gap_z",
    "zone2_temp_z", "web_tension_z", "dew_point_z", "filter_age_z",
])

campaign_shift = rng.normal(
    0.0, 0.24, size=(campaign_count, len(feature_names))
)
process_features = (
    rng.normal(0.0, 1.0, size=(roll_count, len(feature_names)))
    + campaign_shift[campaign_id]
)

quality_latent = (
    1.45 * (process_features[:, 0] > 0.10)
    + 1.10 * (process_features[:, 1] > 0.25)
    - 1.05 * (process_features[:, 2] > 0.05)
    + 0.90 * (process_features[:, 3] > 0.45)
    + 0.55 * process_features[:, 4]
    + 0.35 * process_features[:, 4] * process_features[:, 6]
    + 0.22 * np.sin(1.6 * process_features[:, 7])
    + 0.05 * campaign_id / campaign_count
    + rng.normal(0.0, 0.42, roll_count)
)
rework_required = np.where(
    quality_latent > np.quantile(quality_latent, 0.58), 1, -1
)
label_flip = rng.random(roll_count) < 0.035
rework_required[label_flip] *= -1

coat_weight_deviation = (
    1.85 * process_features[:, 0]
    - 1.30 * process_features[:, 2]
    + 0.95 * process_features[:, 3]
    + 1.05 * np.maximum(process_features[:, 1], 0.0) * process_features[:, 4]
    + 0.45 * np.sin(1.7 * process_features[:, 7])
    + 0.03 * campaign_id
    + rng.normal(0.0, 0.48, roll_count)
)

train_rows = campaign_id < 24
validation_rows = (campaign_id >= 24) & (campaign_id < 32)
test_rows = campaign_id >= 32

assert process_features.shape == (240, 8)
assert (train_rows.sum(), validation_rows.sum(), test_rows.sum()) == (144, 48, 48)
assert campaign_id[train_rows].max() < campaign_id[validation_rows].min()
assert campaign_id[validation_rows].max() < campaign_id[test_rows].min()
print("训练/验证/测试卷数：", train_rows.sum(), validation_rows.sum(), test_rows.sum())
print("训练返工/放行：", np.sum(rework_required[train_rows] == 1), np.sum(rework_required[train_rows] == -1))
```

这个切分模拟“用过去 campaign 预测未来 campaign”。随机打散会让相邻生产周期同时落入训练和测试，掩盖季节、耗材老化与设备维护带来的共同漂移。若一卷被切成多个位置样本，还要以 `roll_id` 为最小分组，不能让同卷片段跨边界。

```python
X_train = process_features[train_rows]
y_train = rework_required[train_rows]
X_validation = process_features[validation_rows]
y_validation = rework_required[validation_rows]
X_test = process_features[test_rows]
y_test = rework_required[test_rows]

y_reg_train = coat_weight_deviation[train_rows]
y_reg_validation = coat_weight_deviation[validation_rows]
y_reg_test = coat_weight_deviation[test_rows]

majority_label = 1 if np.mean(y_train == 1) >= 0.5 else -1
classification_baseline = np.mean(y_test == majority_label)
regression_baseline = np.sqrt(np.mean((y_reg_test - np.mean(y_reg_train)) ** 2))
assert 0 <= classification_baseline <= 1
assert regression_baseline > 0
print("多数类测试准确率：", round(classification_baseline, 3))
print("训练均值测试 RMSE：", round(regression_baseline, 3))
```

<MlQuiz kind="设计判断" question="若部署目标是预测下一个生产 campaign，第 21 章应怎样选择最终测试集？" :options="['随机抽每个 campaign 的一半记录', '使用时间上更晚且全过程冻结的 campaign', '用训练误差最低的轮次', '把验证集和测试集合并以增加样本']" :answer="1" explanation="未来 campaign 是真实部署单位；测试必须位于时间之后并在所有轮数、损失、阈值选择完成前保持冻结。" />

## 2. Boosting 的对象是一个有顺序的加法函数

令弱学习器集合为 $\mathcal H$。Boosting 构造

$$
F_M(\mathbf x)
=F_0(\mathbf x)+\sum_{m=1}^{M}\eta\alpha_mh_m(\mathbf x),
\qquad h_m\in\mathcal H.
$$

不同算法对 $F_0$、$h_m$、$\alpha_m$ 和损失的定义不同，但共有一个关键约束：第 $m$ 轮固定 $F_{m-1}$，只选择新的方向与步长，降低

$$
\mathcal L(F)=\sum_{i=1}^{n}\ell(y_i,F(\mathbf x_i)).
$$

更具体地，前向分阶段更新求解

$$
(\alpha_m,h_m)
=\arg\min_{\alpha,h\in\mathcal H}
\sum_{i=1}^{n}
\ell\bigl(y_i,F_{m-1}(\mathbf x_i)+\alpha h(\mathbf x_i)\bigr).
$$

旧树不回头联合重训，所以它通常不是给定 $M$ 个基函数后的全局最优系数解。它用贪心顺序换取可扩展性、正则化路径和可解释的逐轮状态。

### 2.1 与 Bagging 的差异不只是“串行/并行”

| 问题 | Bagging / 随机森林 | Boosting |
|---|---|---|
| 下一棵树看什么 | 一个重采样数据集和随机特征候选 | 当前集成留下的权重、伪残差或梯度/Hessian |
| 树间依赖 | 给定随机种子后可独立训练 | 第 $m$ 棵依赖 $F_{m-1}$，不能任意并行 |
| 主要作用 | 平均不稳定模型，重点降方差 | 逐轮补充表示，常先降偏差，也可能增方差 |
| 常用基学习器 | 可生长得较深 | 常用树桩或浅树控制单轮交互 |
| 树数语义 | 控制平均的蒙特卡洛稳定性 | 同时是优化步数和容量路径 |
| 异常点反应 | 单点只影响部分树 | 可被连续轮次反复关注 |

“Boosting 一定降低偏差、随机森林一定降低方差”只是方向性诊断，不是无条件定理。有限样本、树深、损失、学习率和噪声会同时改变偏差与方差。

## 3. AdaBoost：指数损失怎样生成整套更新

二分类标签取 $y_i\in\{-1,+1\}$，树桩输出 $h_m(\mathbf x_i)\in\{-1,+1\}$。AdaBoost 的集成分数为

$$
F_M(\mathbf x)=\sum_{m=1}^{M}\alpha_mh_m(\mathbf x),
\qquad \hat y=\operatorname{sign}F_M(\mathbf x).
$$

它使用指数损失

$$
\mathcal L(F)=\sum_{i=1}^{n}\exp[-y_iF(\mathbf x_i)].
$$

定义未归一化间隔 $m_i=y_iF(\mathbf x_i)$。若 $m_i<0$，分类错误；若 $m_i>0$，符号正确，绝对值表示投票离边界有多远。指数损失 $e^{-m_i}$ 在负间隔区快速增长，所以严重错分样本会获得极大影响力。

### 3.1 固定旧模型后，样本权重自然出现

第 $m$ 轮加入 $\alpha h$：

$$
\begin{aligned}
\mathcal L(F_{m-1}+\alpha h)
&=\sum_i
\exp[-y_iF_{m-1}(\mathbf x_i)]
\exp[-\alpha y_ih(\mathbf x_i)]\\
&=\sum_i w_i^{(m)}\exp[-\alpha y_ih(\mathbf x_i)],
\end{aligned}
$$

其中

$$
w_i^{(m)}=\exp[-y_iF_{m-1}(\mathbf x_i)].
$$

所以权重不是额外拍脑袋加上的启发式，而是旧模型在指数损失下留下的逐样本代价。实现时通常把它归一化成分布 $D_m(i)=w_i^{(m)}/\sum_jw_j^{(m)}$。

### 3.2 固定步长时，先找最小加权错误的弱学习器

因为 $y_ih(\mathbf x_i)=+1$ 表示正确，$-1$ 表示错误，目标可拆为

$$
e^{-\alpha}\sum_{i:y_i=h_i}w_i^{(m)}
+e^{\alpha}\sum_{i:y_i\ne h_i}w_i^{(m)}.
$$

只要 $\alpha>0$，错分组乘的 $e^{\alpha}$ 大于正确组乘的 $e^{-\alpha}$。因此应选择加权错误

$$
\varepsilon_m
=\frac{\sum_iw_i^{(m)}\mathbb I(y_i\ne h_m(\mathbf x_i))}
{\sum_iw_i^{(m)}}
$$

最小的 $h_m$。树桩中的类别计数、Gini 或误分类率都必须改成加权版本；把权重只用于最后投票、却用普通未加权树训练，不是同一个算法。

```python
def stump_predict(X_value, feature, threshold, polarity):
    base = np.where(X_value[:, feature] > threshold, 1, -1)
    return polarity * base


def fit_weighted_stump(X_value, y_value, weights):
    n, d = X_value.shape
    if y_value.shape != (n,) or weights.shape != (n,):
        raise ValueError("X、y、weights 形状不一致")
    if not np.all(np.isin(y_value, (-1, 1))):
        raise ValueError("AdaBoost 标签必须是 -1/+1")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights 必须非负且归一化")

    best = None
    for feature in range(d):
        values = np.unique(X_value[:, feature])
        thresholds = np.r_[-np.inf, (values[:-1] + values[1:]) / 2, np.inf]
        for threshold in thresholds:
            for polarity in (-1, 1):
                prediction = stump_predict(
                    X_value, feature, threshold, polarity
                )
                error = float(weights[prediction != y_value].sum())
                candidate = (
                    error, feature, float(threshold), polarity, prediction
                )
                if best is None or candidate[:4] < best[:4]:
                    best = candidate
    return best


uniform_weights = np.full(len(y_train), 1 / len(y_train))
first_stump = fit_weighted_stump(X_train, y_train, uniform_weights)
assert first_stump is not None
assert first_stump[0] < 0.5
print("首轮 ε/特征/阈值/方向：", round(first_stump[0], 4), feature_names[first_stump[1]], round(first_stump[2], 4), first_stump[3])
```

候选中加入 $\pm\infty$ 是为了让常数分类器也进入搜索；重复值之间没有有效中点。平票按“错误率、特征编号、阈值、方向”稳定决定，保证跨运行重放。

### 3.3 固定弱学习器后，闭式解出它的票重

权重归一化后，与 $\alpha$ 有关的目标为

$$
J(\alpha)
=(1-\varepsilon_m)e^{-\alpha}
+\varepsilon_me^{\alpha}.
$$

一阶导数和二阶导数分别是

$$
J'(\alpha)=-(1-\varepsilon_m)e^{-\alpha}+\varepsilon_me^{\alpha},
$$

$$
J''(\alpha)=(1-\varepsilon_m)e^{-\alpha}+\varepsilon_me^{\alpha}>0.
$$

所以只要 $0<\varepsilon_m<1$，这个一维目标严格凸，驻点就是唯一全局最小值：

$$
e^{2\alpha_m}=\frac{1-\varepsilon_m}{\varepsilon_m},
$$

$$
\boxed{\alpha_m=\frac12\log\frac{1-\varepsilon_m}{\varepsilon_m}}.
$$

- $\varepsilon_m<1/2$ 时 $\alpha_m>0$；
- $\varepsilon_m=1/2$ 时 $\alpha_m=0$，该学习器没有信息；
- $\varepsilon_m>1/2$ 时可翻转 $h_m$，错误变成 $1-\varepsilon_m<1/2$；
- $\varepsilon_m=0$ 时理论票重发散，有限精度实现必须停止、裁剪或改用带正则的路线，不能直接计算 `log(1/0)`。

<div class="ml-stop"><strong>停下来推一推 1</strong>分别计算 $\varepsilon=0.1,0.5,0.8$ 对应的 $\alpha$。第三个弱学习器若整体翻转，新的错误率与票重是多少？说明为什么“错误率大于 0.5”不一定表示完全无用。</div>

<MlQuiz kind="推导补全" question="AdaBoost 树桩的加权错误率从 0.30 降到 0.10，最直接发生什么？" :options="['alpha 从正数变成负数', 'alpha 增大，该树桩在最终分数中票重更高', '样本权重全部恢复均匀', '训练顺序变得可并行']" :answer="1" explanation="alpha=0.5 log((1−ε)/ε) 随 ε 在 (0,0.5) 内下降而增大；更可靠的弱学习器获得更大票重。" />

### 3.4 权重更新与归一化常数

加入新树桩后

$$
D_{m+1}(i)
=\frac{D_m(i)\exp[-\alpha_my_ih_m(\mathbf x_i)]}{Z_m},
$$

其中

$$
Z_m=\sum_iD_m(i)\exp[-\alpha_my_ih_m(\mathbf x_i)].
$$

正确样本乘 $e^{-\alpha_m}$，错误样本乘 $e^{\alpha_m}$。代入最优 $\alpha_m$ 可得

$$
Z_m=2\sqrt{\varepsilon_m(1-\varepsilon_m)}.
$$

更精细的恒等式是：归一化后，本轮错分组的总权重恰为 $1/2$，正确组也为 $1/2$：

$$
\frac{\varepsilon_me^{\alpha_m}}{Z_m}
=\frac{(1-\varepsilon_m)e^{-\alpha_m}}{Z_m}
=\frac12.
$$

这不是说每个错分样本权重相同；组内仍按上一轮权重成比例。

<AdaBoostWeightRelayLab />

## 4. 从零实现 AdaBoost，并保存逐轮账本

训练函数要保存每轮的输入权重、树桩、$\varepsilon_m$、$\alpha_m$、$Z_m$、输出权重、分数和间隔。只有最终树列表不足以解释标签污染或复现早停选择。

```python
def fit_adaboost(X_value, y_value, rounds=40):
    if rounds < 1:
        raise ValueError("rounds 必须为正整数")
    n = len(y_value)
    weights = np.full(n, 1 / n)
    score = np.zeros(n)
    model = []
    states = []
    bound = 1.0

    for round_index in range(rounds):
        error, feature, threshold, polarity, prediction = fit_weighted_stump(
            X_value, y_value, weights
        )
        if error >= 0.5 - 1e-12:
            break

        clipped_error = np.clip(error, 1e-12, 1 - 1e-12)
        alpha = 0.5 * np.log((1 - clipped_error) / clipped_error)
        raw_weights = weights * np.exp(
            -alpha * y_value * prediction
        )
        normalizer = float(raw_weights.sum())
        next_weights = raw_weights / normalizer
        score = score + alpha * prediction
        bound *= normalizer
        alpha_sum = sum(abs(item[3]) for item in model) + abs(alpha)

        states.append({
            "round": round_index + 1,
            "weights_before": weights.copy(),
            "weights_after": next_weights.copy(),
            "error": error,
            "alpha": alpha,
            "normalizer": normalizer,
            "bound": bound,
            "train_error": float(np.mean(np.sign(score) != y_value)),
            "margin": y_value * score / alpha_sum,
        })
        model.append((feature, threshold, polarity, alpha))
        weights = next_weights
        if error <= 1e-12:
            break
    return model, states


def adaboost_score(model, X_value):
    score = np.zeros(len(X_value))
    for feature, threshold, polarity, alpha in model:
        score += alpha * stump_predict(
            X_value, feature, threshold, polarity
        )
    return score
```

```python
adaboost_model, adaboost_states = fit_adaboost(
    X_train, y_train, rounds=60
)
assert adaboost_states
assert all(np.isclose(state["weights_after"].sum(), 1.0) for state in adaboost_states)
assert all(np.all(state["weights_after"] >= 0) for state in adaboost_states)

validation_errors = []
for prefix in range(1, len(adaboost_model) + 1):
    validation_score = adaboost_score(
        adaboost_model[:prefix], X_validation
    )
    validation_prediction = np.where(validation_score >= 0, 1, -1)
    validation_errors.append(
        np.mean(validation_prediction != y_validation)
    )

selected_adaboost_round = int(np.argmin(validation_errors)) + 1
frozen_adaboost = adaboost_model[:selected_adaboost_round]
test_adaboost_score = adaboost_score(frozen_adaboost, X_test)
test_adaboost_prediction = np.where(test_adaboost_score >= 0, 1, -1)
test_adaboost_accuracy = np.mean(test_adaboost_prediction == y_test)

print("训练轮数/冻结前缀：", len(adaboost_model), selected_adaboost_round)
print("冻结测试准确率：", round(test_adaboost_accuracy, 3))
print("末轮训练错误/指数上界：", round(adaboost_states[-1]["train_error"], 4), round(adaboost_states[-1]["bound"], 4))
```

本例 60 轮训练的验证前缀选择第 26 轮，冻结测试准确率约为 0.833。数字只证明教学实现的闭环，不代表真实涂布线性能。若多个轮次验证错误相同，应预先规定选择最早轮次，避免无意义增加延迟。

## 5. 弱学习器为什么能把训练错误压下去

### 5.1 0-1 错误被指数损失逐点上界

对每个样本，若 $y_iF_M(\mathbf x_i)\le0$，则 $e^{-y_iF_M(\mathbf x_i)}\ge1$；若分类正确，错误指示量为 0。因此

$$
\mathbb I[y_iF_M(\mathbf x_i)\le0]
\le e^{-y_iF_M(\mathbf x_i)}.
$$

平均后有

$$
\widehat R_{0/1}(F_M)
\le\frac1n\sum_{i=1}^{n}e^{-y_iF_M(\mathbf x_i)}.
$$

### 5.2 指数损失恰好分解为每轮归一化常数之积

初始 $D_1(i)=1/n$，递推展开得

$$
D_{M+1}(i)
=\frac{\frac1n\exp[-y_i\sum_{m=1}^{M}\alpha_mh_m(\mathbf x_i)]}
{\prod_{m=1}^{M}Z_m}.
$$

对 $i$ 求和并使用 $\sum_iD_{M+1}(i)=1$：

$$
\frac1n\sum_i e^{-y_iF_M(\mathbf x_i)}
=\prod_{m=1}^{M}Z_m.
$$

于是

$$
\boxed{
\widehat R_{0/1}(F_M)
\le\prod_{m=1}^{M}2\sqrt{\varepsilon_m(1-\varepsilon_m)}
}.
$$

若每轮弱学习器都有优势 $\gamma_m>0$，即 $\varepsilon_m=1/2-\gamma_m$，则

$$
Z_m=\sqrt{1-4\gamma_m^2}
\le e^{-2\gamma_m^2},
$$

从而

$$
\widehat R_{0/1}(F_M)
\le\exp\left(-2\sum_{m=1}^{M}\gamma_m^2\right).
$$

如果存在统一 $\gamma_m\ge\gamma>0$，训练错误上界至多为 $e^{-2M\gamma^2}$。这就是“弱学习器持续略优于随机猜测可形成强训练分类器”的核心结论。

```python
for state in adaboost_states:
    assert state["train_error"] <= state["bound"] + 1e-12
    exact_z = 2 * np.sqrt(state["error"] * (1 - state["error"]))
    assert np.isclose(state["normalizer"], exact_z)

final_product = np.prod([
    state["normalizer"] for state in adaboost_states
])
assert np.isclose(final_product, adaboost_states[-1]["bound"])
print("训练错误/乘积上界：", round(adaboost_states[-1]["train_error"], 4), round(final_product, 4))
```

这个界只约束训练错误。它没有自动证明测试错误随轮数单调下降，也没有处理自适应选择基学习器带来的完整复杂度。泛化还与弱学习器集合、间隔分布、样本量、噪声和停止规则有关。

### 5.3 间隔比“是否判对”保留更多状态

为了消除整体票重尺度，可定义归一化间隔

$$
\rho_i
=\frac{y_i\sum_m\alpha_mh_m(\mathbf x_i)}
{\sum_m|\alpha_m|}\in[-1,1].
$$

$\rho_i<0$ 表示错分；接近 0 表示投票脆弱；大正值表示弱学习器在该样本上较一致。训练错误已经为 0 后，Boosting 仍可能继续把小正间隔推大，因此“训练错误归零后测试仍改善”并不矛盾。但间隔不是运行域检测，也不是校准概率。

```python
selected_state = adaboost_states[selected_adaboost_round - 1]
selected_margins = selected_state["margin"]
assert np.all(selected_margins >= -1 - 1e-12)
assert np.all(selected_margins <= 1 + 1e-12)
print("训练归一化间隔 5%/50%/95%：", np.round(np.quantile(selected_margins, [0.05, 0.5, 0.95]), 3))

raw_probability = 1 / (1 + np.exp(-2 * np.clip(test_adaboost_score, -350, 350)))
assert np.all((0 <= raw_probability) & (raw_probability <= 1))
print("未校准概率范围：", round(raw_probability.min(), 3), round(raw_probability.max(), 3))
```

指数损失在总体最优条件下给出 $P(Y=1\mid x)=\sigma(2F(x))$ 的联系，但有限树、正则化、数据漂移和模型错设都会破坏直接概率解释。需要概率时，应在独立校准集上拟合校准映射，并把校准方案与动作阈值一同冻结。

## 6. 标签噪声为何会持续吸走权重

把一个训练特征向量复制两次并赋予相反标签，任何确定性分类器都不可能同时判对。这是标签冲突、错误实体合并或时间错配的最小反例。

```python
X_conflict = np.vstack([X_train, X_train[[0]], X_train[[0]]])
y_conflict = np.r_[y_train, -1, 1]
conflict_model, conflict_states = fit_adaboost(
    X_conflict, y_conflict, rounds=60
)

initial_pair_mass = 2 / len(y_conflict)
final_pair_mass = conflict_states[-1]["weights_after"][-2:].sum()
clean_effective_n = 1 / np.sum(adaboost_states[-1]["weights_after"] ** 2)
conflict_effective_n = 1 / np.sum(conflict_states[-1]["weights_after"] ** 2)

assert final_pair_mass > initial_pair_mass
assert conflict_effective_n < clean_effective_n
print("矛盾对初始/末轮总权重：", round(initial_pair_mass, 4), round(final_pair_mass, 4))
print("干净/冲突有效样本数：", round(clean_effective_n, 2), round(conflict_effective_n, 2))
```

本例冲突后的有效样本数由约 50 降至约 38。它说明权重集中，却不能把高权重样本直接判成“坏数据”：真正困难、业务代价高且标签正确的边界样本也会获得高权重。正确流程是回查标签成熟时间、重复实体、特征缺失与工艺上下文，再决定修标签、合并实体、改稳健损失或设置停止门禁。

<MlQuiz kind="错误诊断" question="少数记录的 AdaBoost 权重连续升高，最合理的第一步是什么？" :options="['直接删除所有高权重记录', '审计标签冲突、重复实体、缺失与运行域，再判断是真难例还是数据错误', '把 alpha 全部改成 1', '改用测试集决定是否删除']" :answer="1" explanation="高权重只是当前模型在指数损失下难以解释的信号；它可能来自真难例、错误标签或缺失上下文，不能自动等同异常数据。" />

## 7. 梯度提升：把“修正错误”推广到任意可微损失

AdaBoost 借指数损失得到漂亮的权重闭式解，但真实任务可能需要平方损失、Huber、分位数损失、逻辑损失或 Poisson 偏差。Gradient Boosting 不为每个损失重新发明一套投票规则，而是把训练集上的函数值

$$
\mathbf f=(F(\mathbf x_1),\ldots,F(\mathbf x_n))^\top\in\mathbb R^n
$$

看成优化变量。在当前 $\mathbf f_{m-1}$ 处，经验损失对第 $i$ 个函数值的梯度为

$$
g_{im}
=\left.
\frac{\partial\ell(y_i,F(\mathbf x_i))}
{\partial F(\mathbf x_i)}
\right|_{F=F_{m-1}}.
$$

若能自由修改每个训练点，最速下降会令 $\mathbf f_m=\mathbf f_{m-1}-\eta\mathbf g_m$。但这只记住 $n$ 个坐标，无法预测新样本。于是用一个弱回归器逼近负梯度：

$$
h_m
=\arg\min_{h\in\mathcal H}
\sum_{i=1}^{n}\bigl(-g_{im}-h(\mathbf x_i)\bigr)^2.
$$

再更新

$$
F_m(\mathbf x)
=F_{m-1}(\mathbf x)+\eta\gamma_mh_m(\mathbf x).
$$

这里有两个不同的近似：负梯度是在训练样本函数值空间中的理想方向，$h_m$ 是弱学习器集合对这个方向的投影；$\gamma_m$ 可通过一维线搜索决定，$\eta$ 再做 shrinkage。许多实现把 $\gamma_m$ 吸收到叶输出中。

### 7.1 常见损失的伪残差

伪残差定义为 $r_{im}=-g_{im}$：

| 任务与损失 | $\ell(y,F)$ | 伪残差 $-\partial\ell/\partial F$ | 关键边界 |
|---|---|---|---|
| 平方回归 | $\frac12(y-F)^2$ | $y-F$ | 大残差影响无界 |
| 绝对误差 | $|y-F|$ | $\operatorname{sign}(y-F)$（零点为次梯度） | 方向不含误差大小 |
| Huber | 小残差二次、大残差线性 | $\operatorname{clip}(y-F,-\delta,\delta)$ | $\delta$ 决定限幅区 |
| 二分类逻辑损失 | $-y\log p-(1-y)\log(1-p)$ | $y-p$，$p=\sigma(F)$ | 分数是 logit，不是类别 |
| 多分类交叉熵 | $-\sum_cy_c\log p_c$ | $y_c-p_c$ | 每类梯度和为 0，要处理共同平移 |

平方损失下“拟合残差”不是类比，而是梯度计算的精确结果：

$$
-\frac{\partial}{\partial F}\frac12(y-F)^2=y-F.
$$

### 7.2 从零实现能被逐轮调用的回归树桩

树桩把一个特征按阈值切成左右两叶，并在叶内输出伪残差均值。与第 19 章相同，可用排序后的前缀和在线性扫描中计算所有候选 SSE。

```python
def fit_regression_stump(X_value, target):
    n, d = X_value.shape
    if target.shape != (n,):
        raise ValueError("target 形状错误")
    best = None

    for feature in range(d):
        order = np.argsort(X_value[:, feature], kind="mergesort")
        x_sorted = X_value[order, feature]
        y_sorted = target[order]
        prefix = np.cumsum(y_sorted)
        prefix_sq = np.cumsum(y_sorted ** 2)

        for split in range(1, n):
            if x_sorted[split] == x_sorted[split - 1]:
                continue
            left_n, right_n = split, n - split
            left_sum = prefix[split - 1]
            right_sum = prefix[-1] - left_sum
            left_mean = left_sum / left_n
            right_mean = right_sum / right_n
            left_sse = prefix_sq[split - 1] - left_sum ** 2 / left_n
            right_sq = prefix_sq[-1] - prefix_sq[split - 1]
            right_sse = right_sq - right_sum ** 2 / right_n
            threshold = (x_sorted[split - 1] + x_sorted[split]) / 2
            candidate = (
                float(left_sse + right_sse), feature, float(threshold),
                float(left_mean), float(right_mean),
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate

    if best is None:
        mean = float(np.mean(target))
        return (0.0, 0, np.inf, mean, mean)
    return best


def regression_stump_predict(stump, X_value):
    _, feature, threshold, left_value, right_value = stump
    return np.where(
        X_value[:, feature] <= threshold, left_value, right_value
    )
```

常数特征没有合法中点，函数退化为一个常数叶。实现不允许空叶，也不能在重复值内部切开。真实多叶树应递归复用同样的累计统计，并同时约束深度、最小样本数与最小有效权重。

### 7.3 最小 Gradient Boosting 训练器

平方损失的最优初始常数是训练标签均值；绝对或对称 Huber 路线通常以中位数作为稳健起点。每一轮只在训练集上拟合伪残差，同时把同一棵树追加到验证预测，形成可选择的前缀路径。

```python
def fit_gradient_boosting(
    X_fit, y_fit, X_valid, y_valid,
    rounds=160, learning_rate=0.08,
    loss="squared", delta=1.0,
):
    if not 0 < learning_rate <= 1:
        raise ValueError("learning_rate 必须位于 (0,1]")
    if loss not in {"squared", "huber"}:
        raise ValueError("不支持的损失")

    initial = float(
        np.mean(y_fit) if loss == "squared" else np.median(y_fit)
    )
    fit_prediction = np.full(len(y_fit), initial)
    valid_prediction = np.full(len(y_valid), initial)
    model = []
    history = []

    for round_index in range(rounds):
        raw_residual = y_fit - fit_prediction
        pseudo_residual = (
            raw_residual
            if loss == "squared"
            else np.clip(raw_residual, -delta, delta)
        )
        stump = fit_regression_stump(X_fit, pseudo_residual)
        fit_prediction += learning_rate * regression_stump_predict(
            stump, X_fit
        )
        valid_prediction += learning_rate * regression_stump_predict(
            stump, X_valid
        )
        model.append(stump)
        history.append({
            "round": round_index + 1,
            "train_rmse": float(np.sqrt(np.mean(
                (y_fit - fit_prediction) ** 2
            ))),
            "validation_rmse": float(np.sqrt(np.mean(
                (y_valid - valid_prediction) ** 2
            ))),
            "validation_mae": float(np.mean(
                np.abs(y_valid - valid_prediction)
            )),
            "max_pseudo_residual": float(
                np.max(np.abs(pseudo_residual))
            ),
        })
    return initial, model, history


def gradient_boosting_predict(initial, model, X_value, learning_rate):
    prediction = np.full(len(X_value), initial)
    for stump in model:
        prediction += learning_rate * regression_stump_predict(
            stump, X_value
        )
    return prediction
```

<GradientResidualCoaterLab />

<div class="ml-stop"><strong>停下来推一推 2</strong>当前三个样本的平方损失残差为 $[2,-1,3]$，且它们落在同一叶。推导该叶拟合伪残差时的最优常数；学习率 $\eta=0.1$ 时实际加入 $F$ 的量是多少？如果改为 $\delta=1$ 的 Huber 伪残差，方向怎样变化？</div>

<MlQuiz kind="公式辨析" question="逻辑损失的 Gradient Boosting 为什么不能总把 y−F 当伪残差？" :options="['分类没有梯度', 'F 是 logit，负梯度应为 y−σ(F)', '树不能输出实数', '只有 AdaBoost 可以分类']" :answer="1" explanation="对二分类交叉熵关于 logit F 求导得到 σ(F)−y，负梯度是 y−p；直接用 y−F 混淆了 logit 与概率。" />

## 8. 用冻结验证前缀选择轮数

```python
gb_initial, gb_model, gb_history = fit_gradient_boosting(
    X_train, y_reg_train,
    X_validation, y_reg_validation,
    rounds=260, learning_rate=0.15,
)
selected_gb_round = int(np.argmin([
    row["validation_rmse"] for row in gb_history
])) + 1
frozen_gb_model = gb_model[:selected_gb_round]
gb_test_prediction = gradient_boosting_predict(
    gb_initial, frozen_gb_model, X_test, learning_rate=0.15
)
gb_test_rmse = np.sqrt(np.mean(
    (y_reg_test - gb_test_prediction) ** 2
))

assert gb_test_rmse < regression_baseline
print("训练轮数/验证冻结轮：", len(gb_model), selected_gb_round)
print("GB/均值基线测试 RMSE：", round(gb_test_rmse, 3), round(regression_baseline, 3))
```

固定种子下，验证集在第 242 轮选择前缀；测试 RMSE 约 1.422，低于训练均值基线约 2.753。若早停设置 `patience=20`，应保存“截至当前最佳轮”的制品，而不是循环停止时最后一轮。验证指标的微小改善还要与重复 campaign bootstrap 区间、延迟预算和最小实际收益联合判断。

### 8.1 学习率与轮数是耦合路径，不是两个独立旋钮

$\eta$ 小意味着每一轮只实现弱学习器方向的一部分，通常需要更多轮；$\eta$ 大使训练损失下降更快，也更容易沿噪声方向过冲。不能比较相同轮数下的不同学习率后就宣布某个学习率更好，公平预算至少要比较：

- 各自验证最优前缀；
- 相同训练时间或节点数；
- 相同推理延迟与模型体积；
- 多个 campaign 切分或种子下的稳定性。

```python
learning_rate_paths = {}
for rate, rounds in [(0.30, 140), (0.15, 260), (0.06, 520)]:
    path_initial, path_model, path_history = fit_gradient_boosting(
        X_train, y_reg_train,
        X_validation, y_reg_validation,
        rounds=rounds, learning_rate=rate,
    )
    best_index = int(np.argmin([
        row["validation_rmse"] for row in path_history
    ]))
    learning_rate_paths[rate] = {
        "round": best_index + 1,
        "rmse": path_history[best_index]["validation_rmse"],
        "node_visits": 3 * (best_index + 1),
    }

assert set(learning_rate_paths) == {0.30, 0.15, 0.06}
print("学习率路径：", {
    rate: (row["round"], round(row["rmse"], 3))
    for rate, row in learning_rate_paths.items()
})
```

这里树桩推理可粗略按每轮一个比较加两个叶值读取计为 3 个节点访问。真实实现还受树布局、缓存、批量推理与分支预测影响。

### 8.2 树深控制单轮交互阶数

树桩每轮只表达一个特征的一次轴对齐切分；深度 2 的树可以在一轮表达条件于前一切分的二阶交互。更深的树降低单轮偏差，却会减少“弱”约束、加速记忆局部噪声，并增大模型和延迟。树深与 `min_samples_leaf`、学习率、轮数必须联合选择。

## 9. 叶输出不是对所有损失都等于平均残差

树先用伪残差找到区域 $R_{jm}$ 后，更精确的做法是为每个叶单独线搜索：

$$
\gamma_{jm}
=\arg\min_{\gamma}
\sum_{\mathbf x_i\in R_{jm}}
\ell(y_i,F_{m-1}(\mathbf x_i)+\gamma).
$$

- 平方损失：$\gamma_{jm}$ 是叶内原始残差均值；
- 绝对误差：$\gamma_{jm}$ 是叶内残差中位数，可不唯一；
- Huber：最优值满足裁剪残差之和为 0，通常需一维求根；
- 逻辑损失：叶值与梯度和曲率有关，不是标签均值。

本章最小 Huber 实现直接拟合裁剪伪残差并把树输出作为一步近似，没有再做叶内精确线搜索；这是明确的简化，不应把它误称成某个库的完整实现。

### 9.1 量测尖峰：平方损失与 Huber 的差异

```python
y_outlier = y_reg_train.copy()
y_outlier[[3, 47, 88]] += np.array([12.0, -11.0, 10.0])

sq_initial, sq_model, sq_history = fit_gradient_boosting(
    X_train, y_outlier,
    X_validation, y_reg_validation,
    rounds=320, learning_rate=0.10,
    loss="squared",
)
hu_initial, hu_model, hu_history = fit_gradient_boosting(
    X_train, y_outlier,
    X_validation, y_reg_validation,
    rounds=320, learning_rate=0.10,
    loss="huber", delta=1.8,
)

sq_best = int(np.argmin([
    row["validation_mae"] for row in sq_history
])) + 1
hu_best = int(np.argmin([
    row["validation_mae"] for row in hu_history
])) + 1
sq_validation = gradient_boosting_predict(
    sq_initial, sq_model[:sq_best], X_validation, 0.10
)
hu_validation = gradient_boosting_predict(
    hu_initial, hu_model[:hu_best], X_validation, 0.10
)
sq_mae = np.mean(np.abs(y_reg_validation - sq_validation))
hu_mae = np.mean(np.abs(y_reg_validation - hu_validation))

assert hu_mae < sq_mae
print("平方/Huber 冻结轮：", sq_best, hu_best)
print("平方/Huber 验证 MAE：", round(sq_mae, 3), round(hu_mae, 3))
```

在这个合成污染下，Huber 验证 MAE 约 0.915，平方损失约 1.212。它不证明 Huber 普遍更好：若大残差都是真实高风险卷，过强限幅会损失召回。$\delta$ 必须用业务单位和冻结验证设计，而不是在测试集上挑选。

<MlQuiz kind="设计判断" question="面密度大残差既可能是量测尖峰，也可能是真实严重偏差，Huber 的 delta 应怎样确定？" :options="['固定为 1，不考虑单位', '依据量测重复性、业务代价和冻结验证共同选择', '用测试集逐点调到最好', '越小越稳健，因此取 0']" :answer="1" explanation="delta 决定何时从二次惩罚转为线性惩罚；它必须对应量测噪声尺度和真实大偏差代价，并保持测试集冻结。" />

## 10. 二阶提升树：梯度告诉方向，Hessian 告诉局部尺度

对第 $m$ 轮增量 $f_m(\mathbf x_i)$ 做二阶 Taylor 展开：

$$
\ell(y_i,F_{m-1,i}+f_{m,i})
\approx
\ell(y_i,F_{m-1,i})
+g_if_{m,i}+\frac12h_if_{m,i}^2,
$$

其中

$$
g_i=\left.\frac{\partial\ell}{\partial F_i}\right|_{F_{m-1}},
\qquad
h_i=\left.\frac{\partial^2\ell}{\partial F_i^2}\right|_{F_{m-1}}.
$$

假设新树有 $J$ 个叶，样本映射 $q(\mathbf x)\in\{1,\ldots,J\}$，叶权重为 $w_j$：

$$
f_m(\mathbf x)=w_{q(\mathbf x)}.
$$

加入树复杂度正则

$$
\Omega(f_m)=\gamma J+\frac{\lambda}{2}\sum_{j=1}^{J}w_j^2.
$$

令 $I_j=\{i:q(\mathbf x_i)=j\}$、$G_j=\sum_{i\in I_j}g_i$、$H_j=\sum_{i\in I_j}h_i$，忽略与新树无关的常数后：

$$
\widetilde{\mathcal L}_m
=\sum_{j=1}^{J}
\left[G_jw_j+\frac12(H_j+\lambda)w_j^2\right]
+\gamma J.
$$

对 $w_j$ 求导：

$$
G_j+(H_j+\lambda)w_j=0,
$$

得到正则化 Newton 叶权重

$$
\boxed{w_j^\star=-\frac{G_j}{H_j+\lambda}}.
$$

只要 $H_j+\lambda>0$，二次近似对叶权重严格凸且最优值唯一。若损失在局部非凸、$H_j$ 为负，或 $H_j=\lambda=0$，这套闭式解不再安全。

### 10.1 从叶最优值推导切分增益

代回单叶目标得到改善量 $\frac12G_j^2/(H_j+\lambda)$。把父叶切成左、右两叶，净收益为

$$
\boxed{
\operatorname{gain}
=\frac12\left[
\frac{G_L^2}{H_L+\lambda}
+\frac{G_R^2}{H_R+\lambda}
-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]-\gamma
}.
$$

$\gamma$ 是新增一个叶子的结构成本；增益不大于 0 时保留父叶。工业实现还常要求 $H_L$、$H_R$ 均超过 `min_child_weight`，避免在曲率证据极少的区域产生大步长叶。

<SecondOrderSplitFoundryLab />

<div class="ml-stop"><strong>停下来推一推 3</strong>某叶有 $G=-6,H=2$。分别计算 $\lambda=0$ 与 $\lambda=4$ 的最优叶权重，并解释为什么 $\lambda$ 不只控制模型大小，也给低曲率 Newton 步加了制动。</div>

### 10.2 逻辑损失的梯度与 Hessian

标签取 $y_i\in\{0,1\}$、$p_i=\sigma(F_i)$，单样本负对数似然可写成稳定形式

$$
\ell_i=\log(1+e^{F_i})-y_iF_i.
$$

因此

$$
g_i=p_i-y_i,
\qquad
h_i=p_i(1-p_i).
$$

概率接近 0 或 1 时 $h_i$ 很小。如果此时标签与高置信预测冲突，$|g_i|$ 可接近 1，裸 Newton 步 $-G/H$ 会极大；$\lambda$、最小子叶 Hessian、学习率和 logit 裁剪共同构成数值护栏。

```python
def sigmoid(value):
    value = np.asarray(value, dtype=float)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1 / (1 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


y01_train = (y_train == 1).astype(float)
prior = np.clip(y01_train.mean(), 1e-6, 1 - 1e-6)
initial_logit = np.log(prior / (1 - prior))
logits = np.full(len(y01_train), initial_logit)
probability = sigmoid(logits)
gradient = probability - y01_train
hessian = probability * (1 - probability)

assert np.all(hessian > 0)
assert np.all(hessian <= 0.25 + 1e-12)
assert abs(gradient.sum()) < 1e-10
print("训练先验/logit/Hessian：", round(prior, 3), round(initial_logit, 3), round(hessian[0], 3))
```

初始常数 logit 取训练正类比例的 log odds，使全体梯度和为 0；这只是最佳常数模型，不表示任何特征切分都没有增益，因为局部梯度和仍可不为 0。

```python
def second_order_split_gain(g, h, left, l2=1.0, gamma=0.0):
    if l2 < 0 or gamma < 0:
        raise ValueError("正则化参数必须非负")
    right = ~left
    GL, GR = g[left].sum(), g[right].sum()
    HL, HR = h[left].sum(), h[right].sum()
    parent = (GL + GR) ** 2 / (HL + HR + l2)
    gain = 0.5 * (
        GL ** 2 / (HL + l2)
        + GR ** 2 / (HR + l2)
        - parent
    ) - gamma
    return (
        float(gain),
        float(-GL / (HL + l2)),
        float(-GR / (HR + l2)),
        float(HL), float(HR),
    )


feature = 0
values = np.unique(X_train[:, feature])
thresholds = (values[:-1] + values[1:]) / 2
gain_rows = []
for threshold in thresholds:
    left = X_train[:, feature] <= threshold
    if left.sum() < 5 or (~left).sum() < 5:
        continue
    result = second_order_split_gain(
        gradient, hessian, left, l2=1.0, gamma=0.02
    )
    gain_rows.append((result[0], threshold, *result[1:]))

best_second_order_split = max(
    gain_rows, key=lambda row: (row[0], -row[1])
)
assert best_second_order_split[0] > 0
print("最佳 gain/阈值/左右叶权：", np.round(best_second_order_split[:4], 3))
```

<MlQuiz kind="故障定位" question="二分类二阶树在高置信错标样本附近得到很小 Hessian、很大叶权重，优先检查什么？" :options="['关闭所有正则化', '标签冲突、logit 饱和、lambda、最小子叶 Hessian 与学习率', '把 Hessian 固定为负数', '只增加树数']" :answer="1" explanation="高置信错误会出现大梯度和小曲率，裸 Newton 步可能爆大；数据审计与曲率、步长护栏必须一起检查。" />

## 11. 一阶 Gradient Boosting、Newton Boosting 与 XGBoost 不能混为一谈

本章一阶实现用回归树拟合伪残差，再以学习率追加；二阶路线用 $G/H$ 决定叶权并以正则化增益找切分。XGBoost 是一套具体的二阶正则化树系统，还包含稀疏感知切分、近似/直方图候选、列采样、并行与外存等工程机制。仅写出二阶 Taylor 式不等于实现了 XGBoost。

同样，Gradient Boosting 是优化框架，GBDT 是以决策树为弱学习器的常见实例；AdaBoost 是指数损失下的特定离散分类算法。术语边界要进入模型卡和接口名，避免把完全不同的分数、概率与超参数语义混在一起。

## 数值验证

推导必须同时通过导数、等价式和边界行为验证。只打印一条损失曲线无法证明梯度、Hessian 或权重更新正确。

### 12.1 逻辑损失梯度与 Hessian 的有限差分

```python
def logistic_loss(logit, label):
    return np.logaddexp(0.0, logit) - label * logit


probe_logit = np.array([0.7])
probe_label = np.array([1.0])
epsilon = 1e-5
loss_plus = logistic_loss(probe_logit + epsilon, probe_label)
loss_zero = logistic_loss(probe_logit, probe_label)
loss_minus = logistic_loss(probe_logit - epsilon, probe_label)

finite_gradient = (loss_plus - loss_minus) / (2 * epsilon)
finite_hessian = (
    loss_plus - 2 * loss_zero + loss_minus
) / epsilon ** 2
analytic_probability = sigmoid(probe_logit)
analytic_gradient = analytic_probability - probe_label
analytic_hessian = analytic_probability * (1 - analytic_probability)

gradient_error = np.max(np.abs(finite_gradient - analytic_gradient))
hessian_error = np.max(np.abs(finite_hessian - analytic_hessian))
assert gradient_error < 1e-9
assert hessian_error < 2e-6
print("梯度/Hessian 有限差分误差：", gradient_error, hessian_error)
```

二阶差分会把浮点消去误差除以 $\epsilon^2$，所以 Hessian 容差不能照搬一阶导数。应扫描多个 $\epsilon$，确认存在先下降后上升的可信区间，而不是只挑一个碰巧通过的步长。

### 12.2 AdaBoost 更新恒等式与边界

```python
for state in adaboost_states:
    before = state["weights_before"]
    after = state["weights_after"]
    error = state["error"]
    alpha = state["alpha"]
    assert np.isclose(after.sum(), 1.0)
    assert np.isclose(
        alpha, 0.5 * np.log((1 - error) / error)
    )
    assert np.isclose(
        state["normalizer"],
        2 * np.sqrt(error * (1 - error)),
    )
    assert np.all(np.isfinite(before))
    assert np.all(np.isfinite(after))

assert np.isclose(0.5 * np.log((1 - 0.5) / 0.5), 0.0)
assert 0.5 * np.log((1 - 0.1) / 0.1) > 0
assert 0.5 * np.log((1 - 0.8) / 0.8) < 0
print("AdaBoost 权重、Z 与 ε 边界测试通过")
```

### 12.3 树桩搜索、常数目标与输入拒绝

```python
constant_X = np.ones((6, 2))
constant_target = np.full(6, 3.5)
constant_stump = fit_regression_stump(
    constant_X, constant_target
)
constant_prediction = regression_stump_predict(
    constant_stump, constant_X
)
assert np.allclose(constant_prediction, constant_target)

try:
    fit_gradient_boosting(
        X_train, y_reg_train,
        X_validation, y_reg_validation,
        rounds=2, learning_rate=0.0,
    )
    raise AssertionError("零学习率应被拒绝")
except ValueError:
    pass

try:
    fit_weighted_stump(
        X_train, np.zeros(len(y_train)),
        np.full(len(y_train), 1 / len(y_train)),
    )
    raise AssertionError("0/1 标签不应进入离散 AdaBoost")
except ValueError:
    pass

gain_probe = second_order_split_gain(
    np.array([-0.4, 0.6]),
    np.array([0.2, 0.2]),
    np.array([True, False]),
    l2=1.0, gamma=100.0,
)
assert gain_probe[0] < 0
print("常数树桩、无效输入与负增益拒绝测试通过")
```

### 12.4 稳定指数与可复现前缀

```python
extreme_logits = np.array([-1000.0, 0.0, 1000.0])
stable_probability = sigmoid(extreme_logits)
assert np.all(np.isfinite(stable_probability))
assert np.allclose(stable_probability, [0.0, 0.5, 1.0], atol=1e-15)

repeat_model, repeat_states = fit_adaboost(
    X_train, y_train, rounds=60
)
assert len(repeat_model) == len(adaboost_model)
for first, second in zip(repeat_model, adaboost_model):
    assert first[:3] == second[:3]
    assert np.isclose(first[3], second[3])
assert np.allclose(
    repeat_states[-1]["weights_after"],
    adaboost_states[-1]["weights_after"],
)
print("极端 logit 与确定性重放测试通过")
```

## 复杂度与工程边界

### 13.1 训练、内存与推理复杂度

设样本数 $n$、特征数 $d$、提升轮数 $M$、每棵树叶数 $J$、平均深度 $D$。

| 环节 | 典型时间 | 主要内存 | 说明 |
|---|---:|---:|---|
| 单轮暴力树桩分类 | $O(dn\log n)$ | $O(n)$ | 每轮重新排序；若特征顺序预存，可降为扫描 $O(dn)$ |
| $M$ 轮 AdaBoost 树桩 | $O(Mdn)$ 加一次预排序 | $O(nd+Mn)$ 或流式 $O(nd+n)$ | 完整权重账本占 $O(Mn)$，生产可压缩但要能审计 |
| 一棵深度受限回归树 | 近似 $O(dnD)$ 到 $O(dn\log n)$ | 与候选直方图、节点队列相关 | 精确排序、直方图和稀疏策略不同 |
| $M$ 棵 GBDT | 近似 $O(MdnD)$ | 模型约 $O(MJ)$，训练还需梯度/Hessian 与候选统计 | 训练严格顺序依赖，单棵内部可并行 |
| 单条 AdaBoost 树桩推理 | $O(M)$ | 模型 $O(M)$ | 每轮一次比较与加权累加 |
| 单条树提升推理 | $O(MD)$ | 模型 $O(MJ)$ | 批量、树布局、量化影响实际 P99 |

下面把冻结模型的粗略节点访问和制品字段写成可执行预算，避免只谈渐近符号。

```python
adaboost_comparisons = selected_adaboost_round
gb_node_visits = selected_gb_round * 3  # 教学树桩：比较、叶读取、累加
bytes_per_stump = 4 + 8 + 1 + 8  # feature, threshold, polarity/branch, weight
adaboost_payload = selected_adaboost_round * bytes_per_stump
gb_payload = selected_gb_round * (4 + 8 + 8 + 8)

assert adaboost_comparisons == selected_adaboost_round
assert gb_node_visits == 3 * selected_gb_round
print("AdaBoost 比较/估算字节：", adaboost_comparisons, adaboost_payload)
print("GB 节点访问/估算字节：", gb_node_visits, gb_payload)
```

这些字节数没有包含对象头、对齐、索引、缺失方向和元数据，不能代替真实序列化与基准测试。发布门禁要实测批量 1 与目标批量的 P50/P95/P99、峰值内存、模型加载时间和跨语言一致性。

### 13.2 什么时候优先用更简单的模型

- 关系接近线性、要稳定系数与外推时，Ridge/逻辑回归更容易审计；
- 表格样本很少、标签噪声高且复核昂贵时，先做数据合同和简单树，不要用轮数掩盖问题；
- 需要快速强基线、训练可并行且高方差突出时，随机森林通常更省调参；
- 需要概率决策时，先比较校准逻辑回归；Boosting 原始分数不自动校准；
- 需要严格单调、因果或安全约束时，必须使用明确支持约束的实现与验证，普通 GBDT 不会自动满足；
- 运行域经常出现新配方、新设备或新材料时，树的分段常数外推很危险，必须设置门禁和人工复核。

### 13.3 随机子采样不是 OOB 的直接替代

Stochastic Gradient Boosting 每轮无放回抽取部分训练样本，可降低计算并增加随机正则化。但未参与某轮的样本不能直接像随机森林 OOB 那样组成一个无偏、独立的完整预测器：每棵后续树仍依赖此前所有轮次的模型状态。评价仍应使用冻结验证或明确的交叉拟合协议。

<MlQuiz kind="容量估算" question="一座 500 棵、平均深度 4 的 GBDT 对单条记录推理，主要节点比较量级是多少？" :options="['约 4 次', '约 500 次', '约 2000 次', '必须遍历全部叶']" :answer="2" explanation="每棵树只沿一条约 4 层路径，500 棵合计约 2000 次比较；实际延迟还受内存布局、批量和分支预测影响。" />

## 14. 工业训练、发布与监控协议

### 14.1 训练制品必须保存顺序

Boosting 的树序列不可交换。最低制品包括：

- 原始训练快照、标签成熟时间、campaign/卷级切分与排除规则；
- 特征 schema、单位、缺失、饱和、类别字典、编码器和拟合边界；
- 损失名称及参数、初始化常数、类别/样本权重语义；
- 每轮树结构、叶值、学习率、行/列子采样种子与确定性平票；
- AdaBoost 的 $\varepsilon_m$、$\alpha_m$、$Z_m$、权重摘要和间隔分布；
- GBDT 的训练/验证损失路径、最佳轮、patience 与最小改善门槛；
- 二阶模型的 $\lambda$、$\gamma$、最小子叶 Hessian、最大叶步长和 logit 约束；
- 校准映射、动作阈值、测试报告、拒绝规则、模型哈希和回退版本；
- 库、编译器、线程、浮点精度、直方图边界和跨语言金丝雀。

只保存“树数 242、学习率 0.15”无法重建模型。追加一轮会读取此前完整分数；改变早期树的阈值会重写所有后续伪残差，因此所谓 warm start 也必须验证前缀字节完全一致。

### 14.2 线上最低监控面板

- 输入配方族、campaign、设备、缺失、饱和和标准化范围覆盖；
- 分类分数、校准概率、间隔、拒绝率、复检容量和动作成本；
- 回归残差成熟后的 RMSE、MAE、尾部误差与分位数切片；
- 浅层特征使用、叶覆盖、极小 Hessian 叶命中和最大叶贡献；
- AdaBoost 高权重样本类型与重复实体比例；
- 前 $k$ 轮贡献、最佳轮附近的验证稳定性和线上截断敏感性；
- P50/P95/P99 延迟、模型大小、加载失败和数值溢出；
- 成熟标签延迟、人工复核反馈与可能的选择偏差。

### 14.3 常见失败模式

| 现象 | 更可能的根因 | 首要检查 |
|---|---|---|
| 训练错误很快归零，未来 campaign 召回下降 | 深树/大步长、时间漂移或标签泄漏 | 时间切分、特征可用时刻、学习率—深度路径 |
| 少数样本权重吞噬全局 | 错标、重复矛盾、缺失上下文或真实难例 | 标签成熟、实体键、冲突对和工艺复核 |
| 训练损失下降而验证恶化 | 轮数过多或验证分布变化 | 最佳前缀、patience、campaign 切片 |
| Huber 漏掉严重偏差卷 | $\delta$ 太小，把真风险当尖峰限幅 | 业务代价、重复量测和尾部召回 |
| 二阶叶值突然极大 | 概率饱和、小 Hessian、弱正则 | $G/H$、$\lambda$、最小子叶 Hessian、logit |
| 相同模型包跨服务输出不同 | 阈值精度、比较符、类别编码或树顺序不同 | 逐树前缀金丝雀与序列化合同 |
| 校准概率普遍偏高 | 原始 boosting 分数直接当概率或先验漂移 | 独立校准、Brier/NLL 与部署先验 |
| 线下优化导致线上标签分布改变 | 复检策略形成选择反馈 | 未复检样本标签缺失机制与探索审计 |

### 14.4 反馈回路：只复检高分卷会让低分区失去标签

若上线后只对高风险卷做完整离线量测，低分卷的真实标签逐渐缺失。之后拿“有标签的卷”重训会产生选择偏差：模型只看见自己过去选中的区域。最低缓解是保留按风险层分层的随机审计样本、记录动作概率和标签缺失原因，并在评价中区分全量可观测标签与选择性标签。

<MlQuiz kind="故障定位" question="上线后只复检模型高分卷，半年后训练数据中的低分卷几乎没有成熟标签，这是什么问题？" :options="['树数不足', '模型动作造成选择性标签反馈', 'Hessian 必然为负', 'Bootstrap 唯一率太低']" :answer="1" explanation="模型决定谁被复检，进而决定谁能产生标签；未来训练集不再是被动观测样本，需要随机审计或显式选择机制。" />

## 15. 验收清单

### 数据与评价

- [ ] campaign、卷、位置片段和标签成熟时间已经写进切分合同。
- [ ] 训练、验证、校准、测试各自职责明确，测试没有参与轮数或阈值选择。
- [ ] 后工序、复检结论和返工字段没有时间穿越。
- [ ] 类别代价、回归尾部风险、复检容量与拒绝策略已冻结。
- [ ] 选择性标签和人工复核反馈回路有随机审计样本。

### 算法与数值

- [ ] AdaBoost 标签域、加权树桩、$\varepsilon$、$\alpha$、$Z$ 与权重更新逐项通过断言。
- [ ] $\varepsilon=0,0.5,>0.5$、重复阈值、常数特征和平票协议有明确行为。
- [ ] 训练错误不超过 $\prod_mZ_m$，但报告中没有把它误写成测试错误界。
- [ ] 伪残差与实际损失一致，平方、Huber 和逻辑损失没有混用标签域。
- [ ] 学习率、轮数、树深、最小叶和子采样联合选择。
- [ ] 二阶梯度/Hessian 通过有限差分，叶权重与切分增益含 $\lambda,\gamma$。
- [ ] 小 Hessian、极端 logit、负增益和无效输入被拒绝或稳定处理。

### 系统与运行域

- [ ] 树的顺序、阈值比较符、缺失方向、类别顺序和浮点精度可逐前缀回放。
- [ ] 最佳验证前缀而不是最后训练轮被发布。
- [ ] 批量 1 与目标批量的 P99、峰值内存、模型大小和加载时间已实测。
- [ ] 新配方、新设备、新材料和超范围传感器可拒绝进入自动动作。
- [ ] 校准、间隔、叶覆盖、高权重样本、延迟和成熟标签误差持续监控。

## 16. 30–60 分钟推导与代码练习

### 任务：实现带验证早停的树桩 AdaBoost

目标不是调用库，而是完成以下闭环：

1. 枚举所有特征中点和两个方向，返回稳定平票的加权最优树桩；
2. 实现 $\alpha_t$、$Z_t$ 与归一化权重；
3. 保存训练错误、$\prod Z_t$、最小/中位间隔和验证错误；
4. 用 `patience=8` 与最小改善 `1e-4` 选择最早最佳前缀；
5. 对 $\varepsilon=0,0.5,>0.5$、常数特征、重复值和矛盾标签写测试；
6. 最终测试只调用一次，并输出推理比较次数。

#### 思路与公式骨架

先用当前 $D_t$ 选

$$
h_t=\arg\min_h\sum_iD_t(i)\mathbb I[y_i\ne h(\mathbf x_i)],
$$

再计算

$$
\alpha_t=\underline{\hspace{4cm}},
\qquad
D_{t+1}(i)=\underline{\hspace{6cm}},
$$

并维护

$$
B_t=\prod_{s=1}^{t}Z_s,
\qquad
\rho_i^{(t)}=\underline{\hspace{6cm}}.
$$

早停只读取验证错误序列，遇到平票选择更早轮次。

#### 代码骨架

```python
# TODO：完成支持验证早停与完整账本的 AdaBoost 树桩。
#
# def fit_stump(X, y, weight):
#     # 1. 枚举有效中点与 polarity；
#     # 2. 稳定平票；
#     # 3. 返回 prediction 和 weighted_error。
#
# def fit_adaboost_with_early_stopping(X_train, y_train, X_valid, y_valid, ...):
#     # 1. 初始化 D_1；
#     # 2. 逐轮训练、计算 alpha/Z、更新权重和分数；
#     # 3. 记录训练错误、上界、间隔和验证错误；
#     # 4. 按 patience 返回最早最佳前缀；
#     # 5. 对 eps=0/0.5/>0.5 给出明确策略。
```

#### 参考答案的关键不变量

本章 `fit_weighted_stump` 与 `fit_adaboost` 已给出完整核心实现。加入早停时，正确答案至少应通过：

```python
reference_model, reference_states = fit_adaboost(
    X_train, y_train, rounds=30
)
reference_validation = []
for prefix in range(1, len(reference_model) + 1):
    score = adaboost_score(reference_model[:prefix], X_validation)
    reference_validation.append(
        np.mean(np.where(score >= 0, 1, -1) != y_validation)
    )

reference_best = int(np.argmin(reference_validation)) + 1
assert reference_best >= 1
assert all(
    state["train_error"] <= state["bound"] + 1e-12
    for state in reference_states
)
assert np.isclose(reference_states[-1]["weights_after"].sum(), 1.0)
assert np.all(np.isfinite(reference_states[-1]["weights_after"]))
print("30 轮内最早最佳前缀：", reference_best)
```

边界测试不能靠注释：应实际构造常数特征、全错预测的翻转、完全正确树桩和同坐标反标签对。若实现选择对 $\varepsilon=0$ 裁剪而不是立即停止，必须在模型制品中记录裁剪常数。

### 练习 2：把树桩扩展到深度 2

复用第 19 章的递归树结构拟合伪残差。比较深度 1/2/3 在相同节点预算下的验证性能、最佳轮数、特征交互和 P99 延迟。不得用更多总节点给深树路线额外预算。

### 练习 3：实现精确 Huber 叶线搜索

先用裁剪伪残差寻找区域，再对每个叶求解

$$
\sum_{i\in R}\operatorname{clip}(y_i-F_i-\gamma,-\delta,\delta)=0.
$$

用二分法或分段线性结构求根，并与“直接使用伪残差叶均值”的简化版比较。

### 练习 4：概率校准与动作阈值

在独立校准 campaign 上比较原始 $\sigma(2F)$、Platt scaling 和等距分箱可靠性图。阈值由漏报/误报成本与复检容量确定；禁止用测试集拟合校准或阈值。

### 练习 5：框架接口对照

下面只作可选接口核对，不替代本章从零实现。不同 scikit-learn 版本的损失名称、默认值和 AdaBoost 参数可能变化，运行前必须核对本地版本文档；仓库标准环境不为此新增依赖。

```python
# 不执行：需要可选的 scikit-learn 环境，并应先核对本地版本 API。
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier

library_adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=21),
    n_estimators=selected_adaboost_round,
    learning_rate=1.0,
    random_state=21,
)
library_gbdt = GradientBoostingRegressor(
    n_estimators=selected_gb_round,
    learning_rate=0.15,
    max_depth=1,
    random_state=21,
)
library_adaboost.fit(X_train, y_train)
library_gbdt.fit(X_train, y_reg_train)
```

## 17. 术语卡片

| 术语 | 本章精确定义 | 不要混淆 |
|---|---|---|
| 前向分阶段加法建模 | 固定旧函数，只贪心追加新方向与步长 | 不是联合重训全部树 |
| AdaBoost 样本权重 | 指数损失在当前集成下留下的逐样本代价 | 不是异常概率或业务成本本身 |
| 弱学习条件 | 当前加权分布下错误率严格低于 1/2 | 不是原始未加权准确率略高于 50% |
| margin | 标签与集成分数乘积；可再按总票重归一化 | 不是校准概率或 OOD 距离 |
| pseudo-residual | 损失关于当前函数值的负梯度 | 只有平方损失时才等于普通残差 |
| shrinkage | 用 $\eta$ 缩小每轮函数更新 | 不是直接缩放标签 |
| Newton leaf | 用 $-G/(H+\lambda)$ 得到的二阶正则叶值 | 不是简单叶均值 |
| early stopping | 用独立验证路径选择最早最佳前缀 | 不是训练损失不再下降才停止 |

## 18. 本章结论

- Boosting 是有顺序的前向加法优化，树的排列不能交换。
- AdaBoost 的样本权重来自当前指数损失，弱学习器通过最小加权错误选择。
- 最优票重为 $\frac12\log[(1-\varepsilon)/\varepsilon]$；$\varepsilon=0,0.5,>0.5$ 必须有明确数值策略。
- 最优归一化常数为 $2\sqrt{\varepsilon(1-\varepsilon)}$，训练错误不超过这些常数的乘积。
- 训练错误界不等于测试错误界；间隔保留投票强度，但不自动给出校准概率或运行域保证。
- 指数损失会持续放大严重错分，矛盾标签和重复实体必须在加轮数之前审计。
- Gradient Boosting 在训练样本函数值空间取负梯度，再用弱学习器逼近可泛化方向。
- 平方损失的伪残差才等于普通残差；Huber、逻辑和多分类损失各有不同梯度。
- 学习率、轮数、树深、最小叶和子采样共同定义容量与资源路径。
- 二阶树由 Taylor 近似得到叶权 $-G/(H+\lambda)$ 和含 $\gamma$ 的切分增益。
- 小 Hessian 会放大 Newton 步，必须联合标签审计、L2、最小子叶 Hessian、学习率和 logit 护栏。
- 最佳验证前缀、校准映射、动作阈值和最终测试各自职责不能混用。
- 生产制品必须保存树序、逐轮状态、损失、随机性、阈值精度和运行域，而不只是一个模型文件。

## 来源与许可

Kevin P. Murphy 的 *Probabilistic Machine Learning: An Introduction* 解析自源 PDF 第 644–646 页，用于核对指数损失、margin、离散 AdaBoost、$\alpha$ 与样本权重更新、异常点敏感性；源 PDF 第 646–649 页用于核对 LogitBoost、函数值空间负梯度、常见损失伪残差、shrinkage、随机子采样与梯度树叶线搜索；源 PDF 第 649–650 页用于核对二阶正则化树、叶权重和切分增益。该书为 CC BY-NC-ND 4.0；本章只据其知识边界独立推导，没有改编或复用原书图表、案例、代码和措辞。

Jonathan Richard Shewchuk 的 Berkeley CS 189/289A 课程讲义源 PDF 第 152–155 页用于交叉核对 AdaBoost 指数损失、加权错误、票重、权重递推、短树动机、训练错误趋零条件、margin 概率联系与离群点脆弱性。本章不复用课堂图形、数据、案例或表达。

CMU 10-301/10-601 Lecture 24 *Ensemble Methods: Boosting* 源 PDF 第 13–20 页用于核对弱/强学习器、逐轮加权示例和 AdaBoost 算法；源 PDF 第 21–25 页用于核对票重符号、训练错误分析与 margin 分布。本章不复制课件图形、课堂投票、数据、测验或措辞。

锂离子电池极片涂布批次质量复核案例的 40 个教学 campaign、240 卷合成记录、返工标签、面密度偏差、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作。数值不代表真实电池企业、配方、极片、量测系统、质量标准、复检容量或安全阈值；模型不得直接修改浆料、泵速、模头、烘箱、张力、联锁或自动放行产品。scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

<MlLessonProgress
  lesson-id="21-boosting"
  next-href="/courses/machine-learning/22-mle-map"
  next-label="第 22 章：最大似然估计与最大后验估计"
/>
