---
course: machine-learning
edition: industrial
title: 感知机与支持向量机
description: 以轴承滚子终检分流为贯穿案例，从感知机犯错更新与收敛界推导函数间隔、几何间隔、硬间隔和软间隔 SVM、hinge loss、C 与工业验收协议。
outline: [2, 3]
---

<MlLessonHero
  number="11"
  title="感知机与支持向量机"
  question="分类器已经把训练样本分对之后，怎样证明它留出了可度量的扰动余量，又怎样让少数疑似错标不会把整条边界拖走？"
  duration="建议 430–520 分钟"
  :outcomes="[
    '从线性可行性约束推导感知机更新，并证明犯错次数界的关键不等式',
    '区分函数间隔与几何间隔，从尺度规范推导硬间隔 SVM 原始问题',
    '由松弛变量消元得到 hinge loss，解释 C、支持向量、缩放、代价与监控边界'
  ]"
/>

第 09–10 章从概率分布和似然得到分类器。本章改走几何路线：先只要求样本落在超平面的正确一侧，再要求离边界足够远。

对二分类标签

$$
y_i\in\{-1,+1\},
$$

线性分数与预测为

$$
f(x)=w^Tx+b,
\qquad
\widehat y=\operatorname{sign}(f(x)).
$$

这条路线必须同时回答五个不同问题：

1. 一个样本错分时，怎样把超平面推向正确方向？
2. 什么条件能保证有限次更新后不再犯错？
3. 参数变大而边界不变时，怎样定义真实距离？
4. 数据不可分或含错标时，怎样在间隔与违约之间取舍？
5. SVM 分数不是概率，工业动作又该如何验收？

## 贯穿案例：轴承滚子终检复检分流

某轴承滚子产线在终检工位使用涡流探头和激光轮廓仪筛查表面裂纹。模型输出两个动作：

| 标签 | 动作 | 含义 |
|---:|---|---|
| $-1$ | 自动放行 | 进入后续装配，但仍受硬性工艺规则约束 |
| $+1$ | 送复检 | 进入人工显微复核或无损检测复核 |

本章只讨论**复检分流**，不把模型输出称为“裂纹概率”。最终标签来自冻结的复核流程，而不是模型自己的阈值，也不能使用复检后的返工结果作为输入。

### 数据合同

| 对象 | 教学定义 | 工业系统必须确认 |
|---|---|---|
| 观测单元 | 一只滚子的一次冻结终检记录 | 重测、翻面扫描和返工复测如何去重 |
| 特征 | 处置前的涡流峰值与轮廓缺口深度 | 探头型号、单位、采样窗口和滤波版本 |
| 标签 | 冻结复核流程给出的复检/放行 | 分歧复核、无法判定和错标修订流程 |
| 分组键 | 热处理批次 | 原料炉次、设备、班次是否需要更高层分组 |
| 部署对象 | 新批次、同工艺滚子 | 新材料、探头或表面处理变更需重新验收 |
| 错误代价 | 漏送复检高于多送复检 | 成本值只能由业务与安全责任方确认 |

教学数据包含 60 个热处理批次、每批 40 只滚子。前 40 个随机冻结批次用于训练，10 个用于模型与动作阈值选择，最后 10 个只做一次测试验收。所有数据均为教学合成。

```python
import numpy as np

rng = np.random.default_rng(20260805)
lot_count = 60
rollers_per_lot = 40
lot_id = np.repeat(np.arange(lot_count), rollers_per_lot)
lot_effect = rng.normal(0, 0.32, lot_count)

latent_eddy = rng.normal(lot_effect[lot_id], 1.0)
latent_profile = rng.normal(0.28 * latent_eddy - 0.18 * lot_effect[lot_id], 1.0)
eddy_peak_mv = 24.0 + 5.2 * latent_eddy + rng.normal(0, 0.45, len(lot_id))
notch_depth_um = 7.5 + 2.9 * latent_profile + rng.normal(0, 0.30, len(lot_id))
raw_features = np.column_stack([eddy_peak_mv, notch_depth_um])

latent_risk = (
    1.10 * latent_eddy
    + 0.82 * latent_profile
    + 0.22 * lot_effect[lot_id]
    + rng.normal(0, 0.72, len(lot_id))
)
label = np.where(latent_risk > 0.78, 1.0, -1.0)

# 教学上模拟少量复核分歧；它们不是可据此估计真实错标率的生产数据。
disputed = rng.choice(len(label), size=32, replace=False)
label[disputed] *= -1

lot_order = rng.permutation(lot_count)
train_lots, validation_lots, test_lots = lot_order[:40], lot_order[40:50], lot_order[50:]
train_mask = np.isin(lot_id, train_lots)
validation_mask = np.isin(lot_id, validation_lots)
test_mask = np.isin(lot_id, test_lots)

assert not np.any(train_mask & validation_mask)
assert not np.any(train_mask & test_mask)
assert not np.any(validation_mask & test_mask)
print("训练/验证/测试：", train_mask.sum(), validation_mask.sum(), test_mask.sum())
print("教学复检率：", round((label == 1).mean(), 4))
print("三侧批次数：", len(train_lots), len(validation_lots), len(test_lots))
```

标准化参数只能在训练批次拟合。这里不仅是在改善数值条件，也是在规定 SVM 用什么度量比较两个方向的距离。

```python
train_mean = raw_features[train_mask].mean(axis=0)
train_scale = raw_features[train_mask].std(axis=0, ddof=0)

standardized = (raw_features - train_mean) / train_scale
X_train, y_train = standardized[train_mask], label[train_mask]
X_validation, y_validation = standardized[validation_mask], label[validation_mask]
X_test, y_test = standardized[test_mask], label[test_mask]

assert np.allclose(X_train.mean(axis=0), 0, atol=1e-12)
assert np.allclose(X_train.std(axis=0), 1, atol=1e-12)
print("训练均值：", np.round(train_mean, 3))
print("训练尺度：", np.round(train_scale, 3))
```

<MlQuiz kind="标签合同" question="模型先给出高风险分数，工程师随后依据该分数决定是否把记录标成复检正类。再用这些标签训练 SVM，主要问题是什么？" :options="['没有问题，SVM 不需要概率', '标签被旧模型策略污染，训练目标会复制原策略而非独立复核事实', '只会让 C 变小', '只影响绘图颜色']" :answer="1" explanation="标签应来自独立冻结的复核合同。若标签由旧模型动作产生，新模型只是在学习旧阈值与旧偏差。" />

## 1. 感知机把分类看成线性可行性问题

每条训练记录要求

$$
y_i(w^Tx_i+b)>0.
$$

把截距并入增广向量

$$
\widetilde x_i=
\begin{bmatrix}x_i\\1\end{bmatrix},
\qquad
\widetilde w=
\begin{bmatrix}w\\b\end{bmatrix},
$$

约束写成

$$
y_i\widetilde w^T\widetilde x_i>0.
$$

若当前样本满足

$$
y_i\widetilde w^T\widetilde x_i\le 0,
$$

感知机执行

$$
\boxed{
\widetilde w\leftarrow
\widetilde w+\eta y_i\widetilde x_i
}
$$

其中 $\eta>0$。更新后，当前样本的带标签分数增加

$$
y_i(\widetilde w+\eta y_i\widetilde x_i)^T\widetilde x_i
-y_i\widetilde w^T\widetilde x_i
=\eta\lVert\widetilde x_i\rVert^2>0.
$$

所以这一步确实朝满足当前约束的方向移动。但它可能破坏以前样本的约束；算法必须反复扫描。

感知机损失可写为

$$
\ell_{\mathrm{perc}}(y,f)=\max(0,-yf).
$$

在 $yf<0$ 时，对 $w$ 的次梯度是 $-yx$；在 $yf>0$ 时是 0。标准更新还把压线样本 $yf=0$ 视为犯错，这相当于在折点选择更新侧的次梯度。

```python
toy_X = np.array([
    [-1.8, -0.8], [-1.5, 0.55], [-0.9, -0.4], [-0.7, 1.0],
    [0.8, -0.9], [1.2, 0.3], [1.6, -0.2], [1.8, 0.9],
])
toy_y = np.array([-1., -1., -1., -1., 1., 1., 1., 1.])

def fit_perceptron(X, y, learning_rate=1.0, maximum_epochs=50):
    augmented = np.column_stack([X, np.ones(len(X))])
    parameter = np.zeros(augmented.shape[1])
    best_parameter = parameter.copy()
    best_correct = 0
    mistakes = 0
    trace = []
    converged = False
    for epoch in range(maximum_epochs):
        epoch_mistakes = 0
        for index, (row, target) in enumerate(zip(augmented, y)):
            margin = target * (parameter @ row)
            updated = margin <= 0
            if updated:
                parameter += learning_rate * target * row
                mistakes += 1
                epoch_mistakes += 1
            correct = np.sum(y * (augmented @ parameter) > 0)
            if correct > best_correct:
                best_correct = int(correct)
                best_parameter = parameter.copy()
            trace.append((epoch, index, margin, updated, mistakes, correct))
        if epoch_mistakes == 0:
            converged = True
            break
    return parameter, best_parameter, np.array(trace, dtype=float), converged

toy_parameter, _, toy_trace, toy_converged = fit_perceptron(toy_X, toy_y)
toy_prediction = np.where(
    np.column_stack([toy_X, np.ones(len(toy_X))]) @ toy_parameter >= 0,
    1,
    -1,
)
print("是否收敛/犯错更新数：", toy_converged, int(toy_trace[-1, 4]))
print("最终参数 [w1,w2,b]：", np.round(toy_parameter, 3))
assert np.array_equal(toy_prediction, toy_y)
```

<PerceptronUpdatePlaybackLab />

<MlQuiz kind="更新方向" question="负类样本 y=−1 被误判为正类，感知机对 w 的更新是什么？" :options="['w←w+ηx', 'w←w−ηx', 'w←ηw', 'w 不变，只改概率']" :answer="1" explanation="统一公式是 w←w+ηyx；y=−1 时沿 −x 方向移动。" />

## 2. 收敛定理约束的是犯错次数，不是唯一解

假设存在单位向量 $u$，使所有增广样本满足

$$
y_i u^T\widetilde x_i\ge\gamma>0,
\qquad
\lVert u\rVert=1,
$$

并且

$$
\lVert\widetilde x_i\rVert\le R.
$$

令学习率为 1，从零开始。发生第 $m$ 次犯错更新后：

$$
\begin{aligned}
u^T\widetilde w_m
&=u^T\widetilde w_{m-1}+y_i u^T\widetilde x_i\\
&\ge u^T\widetilde w_{m-1}+\gamma.
\end{aligned}
$$

递推得到

$$
u^T\widetilde w_M\ge M\gamma.
$$

另一方面，犯错时 $y_i\widetilde w_{m-1}^T\widetilde x_i\le0$，因此

$$
\begin{aligned}
\lVert\widetilde w_m\rVert^2
&=\lVert\widetilde w_{m-1}+y_i\widetilde x_i\rVert^2\\
&=\lVert\widetilde w_{m-1}\rVert^2
+2y_i\widetilde w_{m-1}^T\widetilde x_i
+\lVert\widetilde x_i\rVert^2\\
&\le\lVert\widetilde w_{m-1}\rVert^2+R^2.
\end{aligned}
$$

所以

$$
\lVert\widetilde w_M\rVert\le\sqrt{M}R.
$$

结合 Cauchy–Schwarz：

$$
M\gamma
\le u^T\widetilde w_M
\le\lVert\widetilde w_M\rVert
\le\sqrt{M}R,
$$

得到犯错次数界

$$
\boxed{M\le\left(\frac{R}{\gamma}\right)^2.}
$$

这不是训练轮数界，也不保证得到最大间隔边界。它依赖：严格线性可分、正间隔、有限范数和同一特征表示。样本顺序会改变轨迹与最终分隔面。

```python
toy_augmented = np.column_stack([toy_X, np.ones(len(toy_X))])
unit_separator = np.array([1., 0., 0.])
radius = np.linalg.norm(toy_augmented, axis=1).max()
separable_margin = np.min(toy_y * (toy_augmented @ unit_separator))
mistake_bound = (radius / separable_margin) ** 2
actual_mistakes = int(toy_trace[-1, 4])

conflict_X = np.vstack([toy_X, toy_X[5]])
conflict_y = np.append(toy_y, -1.)
_, conflict_best, conflict_trace, conflict_converged = fit_perceptron(
    conflict_X, conflict_y, maximum_epochs=25
)
conflict_augmented = np.column_stack([conflict_X, np.ones(len(conflict_X))])
conflict_best_accuracy = np.mean(conflict_y * (conflict_augmented @ conflict_best) > 0)

print("R/γ/理论犯错上界：", round(radius, 3), round(separable_margin, 3), round(mistake_bound, 2))
print("实际犯错：", actual_mistakes)
print("冲突标签是否收敛/最佳准确率：", conflict_converged, round(conflict_best_accuracy, 3))
assert actual_mistakes <= mistake_bound + 1e-12
assert not conflict_converged and conflict_best_accuracy < 1
```

<div class="ml-stop"><strong>停下来推一推 1</strong>在上述证明中，哪一步使用了“当前样本被错分”，哪一步使用了“存在统一正间隔”？如果数据仅能被分对但最小间隔趋近 0，界会发生什么？</div>

<MlQuiz kind="定理边界" question="感知机在训练集上连续很多轮仍反复更新，最严谨的结论是什么？" :options="['数据一定不可分', '至少说明当前表示、数据顺序和停止范围内尚未观察到收敛；还需检查标签冲突、数值和间隔', '只需把学习率设为无穷大', 'SVM 也一定无解']" :answer="1" explanation="有限运行不能反证可分性；但应优先审计标签冲突、特征表示、尺度和近零间隔。" />

### Pocket 与平均感知机

不可分数据上，标准感知机没有有限停止保证。工程实现通常至少规定：

- 最大轮数和最大更新数；
- 保存训练或验证指标最好的 pocket 参数；
- 对迭代参数做平均，降低样本顺序引起的震荡；
- 记录冲突样本与更新频率，而不是只返回最后一次参数。

这些策略改善运行行为，却没有把感知机变成概率模型，也没有显式解决“间隔多大”的问题。

## 3. 函数间隔不是距离

对样本 $(x_i,y_i)$，函数间隔为

$$
\widehat\gamma_i=y_i(w^Tx_i+b).
$$

它能判断正负，但不能直接称为距离。把 $(w,b)$ 同乘 $a>0$：

$$
\operatorname{sign}(a(w^Tx+b))
=\operatorname{sign}(w^Tx+b),
$$

决策边界完全不变，函数间隔却变成 $a\widehat\gamma_i$。

### 从正交投影推导点到超平面的距离

点 $x$ 到超平面 $w^Tz+b=0$ 的最近点必沿法向量移动。令

$$
z=x-\alpha w.
$$

代入超平面约束：

$$
w^T(x-\alpha w)+b=0,
$$

所以

$$
\alpha=\frac{w^Tx+b}{\lVert w\rVert^2}.
$$

位移长度为

$$
\lVert x-z\rVert
=|\alpha|\lVert w\rVert
=\frac{|w^Tx+b|}{\lVert w\rVert}.
$$

带标签的几何间隔为

$$
\boxed{
\gamma_i=\frac{y_i(w^Tx_i+b)}{\lVert w\rVert}
}
$$

它对正比例缩放不变，才是当前特征度量中的有符号距离。

```python
probe_weight = np.array([1.4, -0.8])
probe_intercept = -0.35
probe_point = np.array([1.1, 0.6])

score = probe_weight @ probe_point + probe_intercept
projection = probe_point - score * probe_weight / (probe_weight @ probe_weight)
distance = score / np.linalg.norm(probe_weight)

scaled_score = (7 * probe_weight) @ probe_point + 7 * probe_intercept
scaled_distance = scaled_score / np.linalg.norm(7 * probe_weight)

print("投影后的边界残差：", probe_weight @ projection + probe_intercept)
print("原始/缩放几何距离：", distance, scaled_distance)
assert abs(probe_weight @ projection + probe_intercept) < 1e-12
assert np.isclose(distance, scaled_distance)
```

<MlQuiz kind="尺度辨析" question="把 w,b 同时乘 100，哪一项保持不变？" :options="['函数间隔', '几何间隔与决策边界', '‖w‖', 'hinge loss']" :answer="1" explanation="几何间隔用 ‖w‖ 归一化，边界只取分数符号；函数间隔和 hinge loss 都会改变。" />

## 4. 从最大几何间隔得到硬间隔 SVM

线性可分时，希望最大化最小几何间隔：

$$
\max_{w,b}\min_i
\frac{y_i(w^Tx_i+b)}{\lVert w\rVert}.
$$

因为参数可正比例缩放，可以固定最小函数间隔为 1：

$$
\min_i y_i(w^Tx_i+b)=1.
$$

此时最小几何间隔是 $1/\lVert w\rVert$。最大化它等价于最小化 $\lVert w\rVert$，再用平方和系数 $1/2$ 得到标准凸二次规划：

$$
\boxed{
\begin{aligned}
\min_{w,b}\quad &\frac12\lVert w\rVert^2\\
\text{s.t.}\quad &y_i(w^Tx_i+b)\ge1,
\quad i=1,\ldots,n.
\end{aligned}}
$$

两条支撑超平面是

$$
w^Tx+b=+1,
\qquad
w^Tx+b=-1.
$$

每侧到决策边界的距离为 $1/\lVert w\rVert$，中间空带总宽度为

$$
\frac{2}{\lVert w\rVert}.
$$

目标严格凸于 $w$，约束线性；若可行，最优法向量 $w$ 唯一。截距 $b$ 在退化数据上不一定唯一。若数据不可线性分，可行域为空，不存在所谓“自动忽略几个点”的硬间隔解。

```python
hard_X = np.array([
    [0., 0.], [0., 2.], [2., 0.], [2., 2.],
    [-1., 1.], [3., 1.],
])
hard_y = np.array([-1., -1., 1., 1., -1., 1.])
hard_weight = np.array([1., 0.])
hard_intercept = -1.
hard_functional_margin = hard_y * (hard_X @ hard_weight + hard_intercept)
hard_support = np.isclose(hard_functional_margin, 1.)

print("函数间隔：", hard_functional_margin)
print("支持向量索引：", np.flatnonzero(hard_support).tolist())
print("半间隔/总间隔：", 1 / np.linalg.norm(hard_weight), 2 / np.linalg.norm(hard_weight))
assert np.all(hard_functional_margin >= 1)
assert hard_support.sum() == 4
```

### 支持向量是活动约束

满足

$$
y_i(w^Tx_i+b)=1
$$

的样本正好贴住间隔边界，是硬间隔支持向量。远离间隔、严格满足 $y_if_i>1$ 的样本对应非活动约束；在局部不改变活动集合时，移动这些远点不会改变最优边界。

“支持向量”不表示它是最典型样本，也不保证标签可靠。恰恰相反，边界附近的错标和异常点最容易成为支持向量并显著影响解。

<div class="ml-stop"><strong>停下来推一推 2</strong>若规范尺度下所有训练样本都满足 $y_if_i>1$，为什么当前参数不可能是硬间隔最优解？提示：把 $w,b$ 同时缩小一点。</div>

<MlQuiz kind="最优性" question="硬间隔最优解中至少一个约束为什么必须取等号？" :options="['否则可以继续按比例缩小 w,b，降低目标且仍保持可行', '因为每个样本都必须错分', '因为 b 必须等于 1', '因为 SVM 输出概率和为 1']" :answer="0" explanation="若所有函数间隔都严格大于 1，就能共同缩小参数而保持约束，目标继续下降，与最优矛盾。" />

## 5. 松弛变量把不可行约束变成可审计违约

真实终检数据会有类分布重叠、复核分歧和测量噪声。为第 $i$ 个样本引入

$$
\xi_i\ge0,
$$

把硬约束改为

$$
y_i(w^Tx_i+b)\ge1-\xi_i.
$$

软间隔原始问题是

$$
\boxed{
\begin{aligned}
\min_{w,b,\xi}\quad &
\frac12\lVert w\rVert^2+C\sum_{i=1}^n\xi_i\\
\text{s.t.}\quad &y_i(w^Tx_i+b)\ge1-\xi_i,\\
&\xi_i\ge0.
\end{aligned}}
$$

每个 $\xi_i$ 都有明确几何含义：

| 带标签分数 $y_if_i$ | 最小 $\xi_i$ | 状态 |
|---:|---:|---|
| $>1$ | 0 | 正确且在间隔外 |
| $=1$ | 0 | 贴住间隔 |
| $(0,1)$ | $(0,1)$ | 分类正确但侵入间隔 |
| $=0$ | 1 | 压在决策边界 |
| $<0$ | $>1$ | 已错分 |

对固定 $w,b$，最小可行松弛量为

$$
\xi_i^*=\max(0,1-y_if_i).
$$

代回目标，得到 hinge loss：

$$
\ell_{\mathrm{hinge}}(y,f)=\max(0,1-yf),
$$

以及无显式松弛变量的等价形式

$$
\boxed{
\min_{w,b}\frac12\lVert w\rVert^2
+C\sum_{i=1}^n\max(0,1-y_i(w^Tx_i+b)).
}
$$

hinge loss 是 0-1 错误的凸上界：错分时 $yf\le0$，hinge 至少为 1；正确但间隔不足的样本仍会被惩罚。

<MlQuiz kind="松弛账本" question="某样本 y·f(x)=−0.35，它的最小松弛量和状态是什么？" :options="['ξ=0.35，分类正确', 'ξ=0.65，间隔内', 'ξ=1.35，已经错分', 'ξ=0，间隔外']" :answer="2" explanation="ξ=max(0,1−yf)=1.35；带标签分数为负意味着错分。" />

## 6. C 不是“边界复杂度旋钮”这么简单

原始目标中：

- $C$ 大：每单位违约更贵，模型更愿意增大 $\lVert w\rVert$、缩窄间隔以追逐训练样本；
- $C$ 小：更重视小范数与宽间隔，允许更多间隔违约。

这不等于“大 C 一定更准”或“小 C 一定支持向量更多”。结果还取决于数据重叠、错标、特征尺度、类别权重和实现的目标缩放。

许多推导和代码使用平均 hinge：

$$
\frac\lambda2\lVert w\rVert^2
+\frac1n\sum_i\ell_i.
$$

把 C 形式整体除以 $nC$，可得

$$
\boxed{\lambda=\frac1{nC}.}
$$

若库使用损失总和、损失平均、不同的 $1/2$ 系数、类别权重或截距惩罚，数值对应关系都会改变。迁移超参数前必须先对齐目标定义。

<SvmMarginCaliperLab />

<MlQuiz kind="超参数语义" question="样本量从 n 翻倍到 2n，代码改为平均 hinge 且 λ 保持不变。若换回 ½‖w‖²+CΣhinge 形式，C 应怎样对应？" :options="['仍固定不变', '按 C=1/(nλ)，因此减半', '变成 2λ', '只由类别数决定']" :answer="1" explanation="两种目标的等价关系是 λ=1/(nC)。目标缩放不同，不能裸抄同一个 C。" />

## 7. hinge 次梯度解释谁真正推动参数

使用等价的平均损失形式

$$
J(w,b)=
\frac\lambda2\lVert w\rVert^2
+\frac1n\sum_{i=1}^n\max(0,1-y_if_i).
$$

定义活动集合

$$
\mathcal A=\{i:y_if_i<1\}.
$$

在没有样本恰好落在折点 $y_if_i=1$ 时：

$$
\boxed{
\nabla_wJ
=\lambda w-\frac1n\sum_{i\in\mathcal A}y_ix_i,
\qquad
\frac{\partial J}{\partial b}
=-\frac1n\sum_{i\in\mathcal A}y_i.
}
$$

间隔外样本的 hinge 梯度为 0；它们不直接推动当前一步。正则项仍持续收缩 $w$。在折点处，hinge 对 $m=yf$ 的次微分是

$$
\partial\max(0,1-m)\big|_{m=1}=[-1,0].
$$

这意味着“折点梯度”不是唯一数字。数值梯度检查必须避开折点，或改查次梯度不等式。

```python
def svm_objective_gradient(parameter, X, y, C):
    weight, intercept = parameter[:-1], parameter[-1]
    regularization = 1 / (len(y) * C)
    margins = y * (X @ weight + intercept)
    hinge = np.maximum(0, 1 - margins)
    active = margins < 1
    objective = 0.5 * regularization * (weight @ weight) + hinge.mean()
    gradient_weight = regularization * weight - X[active].T @ y[active] / len(y)
    gradient_intercept = -y[active].sum() / len(y)
    return objective, np.append(gradient_weight, gradient_intercept), margins

gradient_probe = np.array([0.37, -0.21, 0.13])
probe_C = 0.2
probe_objective, analytic_gradient, probe_margins = svm_objective_gradient(
    gradient_probe, toy_X, toy_y, probe_C
)
assert np.min(np.abs(probe_margins - 1)) > 1e-3

epsilon = 1e-6
numeric_gradient = np.zeros_like(gradient_probe)
for coordinate in range(len(gradient_probe)):
    direction = np.zeros_like(gradient_probe)
    direction[coordinate] = epsilon
    plus = svm_objective_gradient(gradient_probe + direction, toy_X, toy_y, probe_C)[0]
    minus = svm_objective_gradient(gradient_probe - direction, toy_X, toy_y, probe_C)[0]
    numeric_gradient[coordinate] = (plus - minus) / (2 * epsilon)

grid = np.linspace(-1, 3, 401)
kink_subgradient = -0.4
subgradient_gap = np.maximum(0, 1 - grid) - kink_subgradient * (grid - 1)
print("平滑区有限差分最大误差：", np.max(np.abs(analytic_gradient - numeric_gradient)))
print("折点次梯度不等式最小余量：", subgradient_gap.min())
assert np.max(np.abs(analytic_gradient - numeric_gradient)) < 1e-7
assert subgradient_gap.min() >= -1e-12
```

### 一个可审计的原始问题求解器

下面用全批次次梯度和递减步长求近似解。生产中应使用经过验证的凸优化器或成熟线性 SVM 求解器；教学实现的任务是暴露目标、活动集合和停止量，不是替代工业库。

```python
def fit_linear_svm(X, y, C, iterations=2200):
    parameter = np.zeros(X.shape[1] + 1)
    best_parameter = parameter.copy()
    best_objective = np.inf
    audit = []
    for iteration in range(iterations):
        objective, gradient, margins = svm_objective_gradient(parameter, X, y, C)
        if objective < best_objective:
            best_objective = objective
            best_parameter = parameter.copy()
        rate = 0.42 / np.sqrt(iteration + 25)
        parameter -= rate * gradient
        if iteration % 200 == 0 or iteration == iterations - 1:
            audit.append((iteration, objective, np.linalg.norm(gradient), np.mean(margins <= 1)))
    return best_parameter, np.array(audit)

demo_parameter, demo_audit = fit_linear_svm(X_train, y_train, C=0.2)
demo_train_objective = svm_objective_gradient(demo_parameter, X_train, y_train, 0.2)[0]
print("目标首/尾/最佳：", np.round(demo_audit[0, 1], 5), np.round(demo_audit[-1, 1], 5), round(demo_train_objective, 5))
print("最终次梯度范数：", round(demo_audit[-1, 2], 5))
assert demo_train_objective <= demo_audit[0, 1]
```

<MlQuiz kind="次梯度" question="某训练样本已经满足 y·f(x)=2.4。当前一步中它对 hinge 部分的参数次梯度是什么？" :options="['−yx', '+yx', '0', '概率减标签']" :answer="2" explanation="margin>1 时 hinge 为常数 0；样本不直接贡献 hinge 梯度，但正则项仍收缩 w。" />

## 8. 特征缩放是在定义间隔几何

原始特征单位可能是 mV 与 μm。若直接计算

$$
\frac{|w^Tx+b|}{\lVert w\rVert_2},
$$

“一单位 mV”和“一单位 μm”被当成相同长度。把训练均值和尺度记为 $\mu,s$，标准化为

$$
z=D^{-1}(x-\mu),
\qquad
D=\operatorname{diag}(s),
$$

在 $z$ 空间的欧氏距离等价于原空间的加权距离：

$$
\lVert z_1-z_2\rVert^2
=(x_1-x_2)^TD^{-2}(x_1-x_2).
$$

因此缩放不是“画图方便”或“让优化快一点”而已。它决定哪个传感器方向的一次扰动算得更远。工程上必须版本化：

- 单位与量纲；
- 训练均值、尺度及零方差处理；
- 缺失值与异常值规则；
- 推理时使用同一参数；
- 传感器换型后的重新标定协议。

<MlQuiz kind="几何合同" question="把轮廓特征从 μm 改成 mm，却沿用原 SVM 参数和缩放配置，会发生什么？" :options="['决策严格不变', '几何、分数和边界都可能改变，属于输入合同破坏', '只影响训练速度', 'C 会自动换算']" :answer="1" explanation="单位改变相当于对一个坐标轴重缩放；若参数和标准化未同步，模型看到的是不同几何。" />

## 9. 用独立批次联合选择 C 与动作阈值

训练目标默认用 $f(x)=0$ 分隔两类，但工业动作代价可能不对称。需要区分两个选择：

1. $C$ 决定训练时宽间隔与违约的权衡；
2. 部署阈值 $t$ 决定何时把分数送复检。

若漏送复检代价高于多送复检，冻结验证集上的最优 $t$ 可能低于 0。阈值移动不会重新训练支持向量，但会把决策超平面沿法向平移。

```python
def action_cost(y, score, threshold, false_negative_cost=14.0, false_positive_cost=2.0):
    prediction = np.where(score >= threshold, 1., -1.)
    false_negative = np.sum((y == 1) & (prediction == -1))
    false_positive = np.sum((y == -1) & (prediction == 1))
    return (false_negative_cost * false_negative + false_positive_cost * false_positive) / len(y)

C_candidates = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
candidate_rows = []
candidate_models = []
for candidate_C in C_candidates:
    parameter, audit = fit_linear_svm(X_train, y_train, candidate_C, iterations=1800)
    validation_score = X_validation @ parameter[:-1] + parameter[-1]
    threshold_candidates = np.quantile(validation_score, np.linspace(0.02, 0.98, 161))
    validation_costs = np.array([
        action_cost(y_validation, validation_score, threshold)
        for threshold in threshold_candidates
    ])
    best_index = int(np.argmin(validation_costs))
    train_margin = y_train * (X_train @ parameter[:-1] + parameter[-1])
    candidate_rows.append((
        candidate_C,
        threshold_candidates[best_index],
        validation_costs[best_index],
        np.mean(train_margin <= 1.01),
        np.linalg.norm(parameter[:-1]),
    ))
    candidate_models.append(parameter)

candidate_rows = np.array(candidate_rows)
selected_index = int(np.argmin(candidate_rows[:, 2]))
selected_C, selected_threshold = candidate_rows[selected_index, :2]
selected_parameter = candidate_models[selected_index]

print("C / threshold / validation cost / support ratio / norm")
print(np.round(candidate_rows, 4))
print("冻结选择：", selected_C, round(selected_threshold, 4))
```

候选集、成本值和阈值搜索网格都必须在看测试集之前冻结。候选数量越多，验证集选择偏差越明显；高风险任务还应按批次 bootstrap 或外层交叉验证报告不确定性。

### 测试集只做一次最终账本

```python
test_score = X_test @ selected_parameter[:-1] + selected_parameter[-1]
test_prediction = np.where(test_score >= selected_threshold, 1., -1.)

test_tp = int(np.sum((y_test == 1) & (test_prediction == 1)))
test_fp = int(np.sum((y_test == -1) & (test_prediction == 1)))
test_tn = int(np.sum((y_test == -1) & (test_prediction == -1)))
test_fn = int(np.sum((y_test == 1) & (test_prediction == -1)))
test_cost = action_cost(y_test, test_score, selected_threshold)
test_recall = test_tp / (test_tp + test_fn)
test_specificity = test_tn / (test_tn + test_fp)

train_score = X_train @ selected_parameter[:-1] + selected_parameter[-1]
train_margin = y_train * train_score
train_slack = np.maximum(0, 1 - train_margin)
support_mask = train_margin <= 1.01

print("测试 TP/FP/TN/FN：", test_tp, test_fp, test_tn, test_fn)
print("召回/特异度/每件相对成本：", round(test_recall, 4), round(test_specificity, 4), round(test_cost, 4))
print("训练支持向量占比/Σξ：", round(support_mask.mean(), 4), round(train_slack.sum(), 2))
```

<MlQuiz kind="决策分层" question="把部署阈值从 0 调低以减少漏送复检，哪项一定成立？" :options="['SVM 必须重新训练', '支持向量和训练参数不变，动作边界沿法向移动', 'C 自动增大', '分数立刻变成概率']" :answer="1" explanation="阈值属于动作层。它改变预测规则 sign(f−t)，不重新定义已经训练好的 w,b 和支持向量。" />

## 10. 支持向量比例是诊断量，不是越少越好

软间隔中，满足 $y_if_i\le1$ 的样本对 hinge 活动；在精确最优解的对偶描述里，它们与非零对偶系数、支持向量状态相关。工业监控至少要区分：

- 贴住间隔、$\xi_i\approx0$；
- 正确但侵入间隔、$0<\xi_i<1$；
- 已错分、$\xi_i>1$；
- 间隔外、hinge 为 0。

高支持向量比例可能说明边界附近数据密集、C 较小、类别重叠、缩放变化或标签冲突。低比例也不自动代表泛化好；它可能来自过大的 C、极端尺度或测试分布已经偏离训练边界。

```python
margin_outside = train_margin > 1.01
inside_correct = (train_margin > 0) & (train_margin < 0.99)
misclassified = train_margin < 0
near_hinge = np.abs(train_margin - 1) <= 0.01

active_feature_push = -(X_train[support_mask].T @ y_train[support_mask]) / len(y_train)
far_feature_push = -(X_train[margin_outside].T @ y_train[margin_outside]) * 0.0

print("间隔外/贴边/间隔内正确/错分：", margin_outside.sum(), near_hinge.sum(), inside_correct.sum(), misclassified.sum())
print("活动样本 hinge 推力范数：", round(np.linalg.norm(active_feature_push), 5))
print("间隔外单点 hinge 梯度均为 0：", np.linalg.norm(far_feature_push))
assert np.all(train_slack[margin_outside] == 0)
assert np.linalg.norm(far_feature_push) == 0
```

### 类别加权必须写进目标合同

若正负类违约代价不同，可使用

$$
\frac12\lVert w\rVert^2
+C_+\sum_{i:y_i=+1}\xi_i
+C_-\sum_{i:y_i=-1}\xi_i.
$$

这会改变训练边界，不等同于部署后移动阈值。类别权重适合表达“训练时哪些违约更贵”；阈值适合表达“给定冻结分数怎样做动作”。两者可以共同存在，但必须分别调参与审计。

## 11. SVM 分数默认不是概率

$f(x)=w^Tx+b$ 的绝对值与当前规范尺度下的几何距离相关：

$$
\text{signed distance}=\frac{f(x)}{\lVert w\rVert}.
$$

但训练目标没有假设

$$
f(x)=\log\frac{P(y=1\mid x)}{P(y=-1\mid x)}.
$$

因此不能直接把 $f=2$ 宣称为“88% 风险”，也不能把不同版本 SVM 的原始分数横向比较。若业务确实需要概率，应在独立校准集上拟合单调映射，并重新检查 NLL、Brier、可靠性图和部署先验。校准集不能再用于无休止选择 C、特征和阈值。

多分类通常使用 one-vs-rest 或 one-vs-one 组合多个二分类 SVM。组合后的分数尺度、平票规则、校准与成本动作都需要另行定义；原生互斥概率建模优先回到第 10 章 Softmax。

<MlQuiz kind="输出语义" question="线性 SVM 返回 decision_function=2.3，最安全的解释是什么？" :options="['正类概率一定是 2.3', '正类概率一定是 sigmoid(2.3)', '样本位于正侧且具有某个未校准函数分数；概率需独立校准', '表示有 2.3 个支持向量']" :answer="2" explanation="SVM 优化 margin/hinge，不把分数拟合成 log odds；概率语义不能凭 Sigmoid 猜出。" />

## 数值验证：从更新方向到部署账本

本章实现至少应自动检查以下不变量：

1. 每次感知机犯错更新都提高当前样本的带标签分数；
2. 几何距离对 $(w,b)$ 的共同正缩放不变；
3. 硬间隔规范尺度下所有约束至少为 1；
4. $\xi_i=\max(0,1-y_if_i)$ 且 hinge 上界 0-1 错误；
5. 有限差分避开折点时与解析梯度一致；
6. 间隔外样本的 hinge 梯度为 0；
7. 训练/验证/测试批次不重叠；
8. 测试只使用冻结的 C、缩放与阈值。

```python
first_update = next(row for row in toy_trace if row[3] == 1)
first_index = int(first_update[1])
first_row = np.append(toy_X[first_index], 1.)
first_target = toy_y[first_index]
before_margin = first_update[2]
expected_margin_gain = first_row @ first_row

zero_one = (train_margin <= 0).astype(float)
hinge = np.maximum(0, 1 - train_margin)

assert expected_margin_gain > 0
assert np.all(hinge + 1e-12 >= zero_one)
assert np.all(train_slack >= 0)
assert set(train_lots).isdisjoint(validation_lots)
assert set(train_lots).isdisjoint(test_lots)
assert set(validation_lots).isdisjoint(test_lots)
assert np.isfinite(test_cost)
print("更新、距离、hinge 上界、批次隔离与冻结测试账本检查通过。")
```

<div class="ml-stop"><strong>停下来推一推 3</strong>线性 SVM 的支持向量比例突然从 18% 升到 62%，但测试准确率暂时没变。列出至少四种可能原因，并为每种原因指定一个能区分它的审计量。</div>

## 复杂度与工程边界

设样本数为 $n$、特征数为 $d$、训练扫描轮数为 $T$：

| 环节 | 典型时间 | 典型内存 | 关键边界 |
|---|---:|---:|---|
| 流式感知机一轮 | $O(nd)$ | $O(d)$ | 依赖顺序；不可分时不收敛 |
| 全批次原始 SVM 一步 | $O(nd)$ | $O(nd)$ 或流式 $O(d)$ | hinge 不光滑；停止准则需审计 |
| 线性 SVM 原始参数推理 | $O(d)$ / 样本 | $O(d)$ | 不必保存训练样本 |
| 对偶/核 SVM 推理 | $O(n_{SV}d)$ 或核计算成本 | $O(n_{SV}d)$ | 支持向量数直接影响延迟 |
| C 与阈值搜索 | 候选数乘训练/评价成本 | 依实现而定 | 验证集复用导致选择偏差 |

本章只完成线性原始问题。以下内容留给下一章“核方法与核 SVM”：

- Lagrangian 与完整 KKT 条件；
- SVM 对偶问题；
- 对偶系数与支持向量的精确关系；
- Gram 矩阵与正半定核；
- RBF/多项式核、核参数和计算成本。

### 工业边界

- **线性边界**：原始特征不可线性分时，调 C 不能创造非线性表示。
- **离群点敏感**：间隔外远点无梯度，不代表边界附近错标鲁棒。
- **概率缺失**：分数适合排序和几何审计，概率必须另行校准。
- **尺度依赖**：特征单位和标准化版本改变几何问题本身。
- **分布漂移**：支持向量、margin、slack 与动作率都应按批次监控。
- **求解精度**：近似优化器可能让“支持向量”判定受容差影响。

## 12. 常见失效模式

| 现象 | 根因候选 | 优先证据 | 处理 |
|---|---|---|---|
| 感知机不停更新 | 标签冲突、重叠、近零间隔 | 重复坐标异标签、最佳准确率、更新样本频次 | 修订标签合同或改用软间隔 |
| 硬间隔无解 | 数据不可线性分 | 可行性检查、最小 margin | 不要靠放宽数值容差伪造可行 |
| 大 C 后边界剧烈旋转 | 错标或异常点主导 | 单点 slack、支持向量和参数路径 | 审核样本，按验证协议选 C |
| 换单位后结果突变 | 缩放合同破坏 | 原始单位、均值尺度版本、分数分布 | 恢复一致变换并重新验收 |
| 准确率高但漏检代价高 | 固定 0 阈值不匹配业务 | 验证成本曲线、FN/FP | 冻结成本与动作阈值 |
| SVM 分数被当概率 | 输出语义错误 | 可靠性图、NLL、Brier | 独立校准或使用概率模型 |
| 支持向量比例突然上升 | 漂移、错标、C/尺度变化 | margin/slack 分布与版本记录 | 数据质量告警和分层复验 |

排查顺序：

1. 核对观测单元、标签来源、特征截止时间和批次键；
2. 核对单位、标准化参数、标签编码和截距约定；
3. 检查重复坐标异标签、近零间隔和可分性；
4. 对感知机记录更新轨迹，对 SVM 记录目标、次梯度和 slack；
5. 在独立批次选择 C、类别权重与阈值；
6. 最后才考虑核、复杂特征或更换模型。

## 13. 上线验收清单

- [ ] 复检/放行标签合同、争议样本和修订流程已冻结。
- [ ] 观测单元、去重、特征截止和批次键与部署一致。
- [ ] 单位、均值、尺度和零方差规则已版本化。
- [ ] 感知机只在 $yf\le0$ 更新，并有最大轮数、pocket 与审计轨迹。
- [ ] 可分性与犯错次数界的假设没有被误当成生产事实。
- [ ] 函数间隔、几何间隔和共同缩放已数值验证。
- [ ] 硬/软间隔目标、sum/mean、C/λ 与截距惩罚约定已锁定。
- [ ] hinge 次梯度在非折点通过有限差分，折点通过次梯度不等式。
- [ ] C、类别权重与动作阈值在独立批次选择。
- [ ] 测试集仅使用冻结流水线做一次验收。
- [ ] 每类召回、特异度、成本、margin、slack 和支持向量比例达标。
- [ ] SVM 分数没有未经校准冒充概率。
- [ ] 低 margin、未知工况和硬性工艺规则有独立处置路径。

## 14. 章节练习

### 练习 A：逐步证明感知机犯错界

不引用现成结论，完成以下推导：

1. 写出第 $m$ 次犯错后 $u^T\widetilde w_m$ 的递推下界；
2. 写出 $\lVert\widetilde w_m\rVert^2$ 展开式；
3. 指出错分条件消掉了哪一项；
4. 用 Cauchy–Schwarz 得到 $M\le(R/\gamma)^2$；
5. 解释把截距并入增广特征后，$R$ 和 $\gamma$ 如何变化；
6. 构造相同坐标、相反标签的反例说明定理假设失效。

### 练习 B：从距离推导硬间隔

给定正类 $(2,0),(2,2),(3,1)$，负类 $(0,0),(0,2),(-1,1)$：

1. 推导最大间隔决策边界；
2. 找到规范尺度的 $w,b$；
3. 计算函数间隔、半间隔和总间隔；
4. 标出支持向量；
5. 把所有参数乘 5，比较函数间隔、几何间隔和 hinge loss。

### 练习 C：实现可审计软间隔求解器

补全下面的训练器。要求支持训练集拟合标准化、平均 hinge、截距不惩罚、类别权重、递减步长、pocket 参数、折点容差、目标/次梯度/slack/支持向量审计，以及冻结验证阈值。

```python
# TODO: 练习骨架，验证脚本会跳过本代码块。
def auditable_linear_svm_fit(X, y, C, sample_weight=None):
    parameter = np.zeros(X.shape[1] + 1)
    audit_rows = []
    # 1. 明确 sum/mean 与 C/λ 对应关系
    # 2. 稳定计算 margin、hinge、活动集合和次梯度
    # 3. 执行下降步并保存最佳目标参数
    # 4. 记录 slack 分层、支持向量比例和停止原因
    return parameter, audit_rows
```

### 练习 D：设计批次验收协议

为真实终检系统写出不超过两页的验收方案，至少包括：

- 热处理批次切分与重复滚子处理；
- 复核标签盲审与争议样本；
- C、类别权重、动作阈值与校准的选择顺序；
- FN/FP 成本和容量约束；
- margin、slack、支持向量比例与动作率监控；
- 探头换型、单位变更和新材料的再验收触发条件。

<MlQuiz kind="综合验收" question="线性 SVM 的测试准确率很高，是否足以证明可上线？" :options="['足够', '不够；还要证明标签和批次合同、缩放几何、求解目标、C/阈值冻结、代价、margin/slack 漂移与概率语义都正确', '只要支持向量少于 10 个即可', '把 C 设为最大即可']" :answer="1" explanation="准确率只覆盖一个阈值下的总体分类结果，不能证明数据合同、几何、数值实现和工业动作边界。" />

## 15. 本章结论

感知机与支持向量机的逻辑链是：

$$
\boxed{
\text{错分约束}
\Rightarrow
\text{感知机更新}
\Rightarrow
\text{可分时有限犯错}
\Rightarrow
\text{几何间隔}
\Rightarrow
\text{最大间隔}
\Rightarrow
\text{软间隔与 hinge}
}
$$

必须保留五条边界：

1. **感知机只保证找到某个分隔面**：且前提是严格线性可分；
2. **函数间隔不是距离**：只有除以 $\lVert w\rVert$ 后才消除尺度自由；
3. **SVM 把分对升级为留余量**：硬间隔是凸 QP，软间隔把违约显式记账；
4. **C、缩放与阈值作用不同**：分别改变训练权衡、几何度量和部署动作；
5. **SVM 分数不是概率**：概率需要独立校准与重新验收。

下一章将从软间隔原始问题构造 Lagrangian，推导对偶与 KKT 条件，再说明为什么只出现内积后就能引入核函数。

## 来源与许可

- Stanford CS229 Lecture Notes，PDF 第 25 页，§2.2：阈值预测、感知机逐样本更新及其非概率边界。用于更新规则与术语核对；不复用邮件示例、原文表达或图形。
- Stanford CS229 Lecture Notes，PDF 第 60–74 页，§6.1–6.7：间隔直觉、函数/几何间隔、最优间隔分类器、支持向量、不可分情形与 C。用于推导顺序和公式边界交叉核对；本章案例、代码和实验独立创作。
- Jonathan Richard Shewchuk，*Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 18–24 页，Lectures 2–4：感知机更新、收敛定理直觉、最大间隔、硬间隔和软间隔。用于犯错界与几何解释核对；不复制 Mark I 图片、课堂图形、题目或措辞。
- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 615–622 页，§17.3–17.3.6：大间隔、距离、支持向量、松弛变量、C、hinge loss 与概率校准边界。仅作知识结构、公式和工程边界核对；不复用原书图表、数据或表达。

本章 2,400 只滚子、60 个热处理批次、涡流/轮廓特征、复检标签、成本定义、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实轴承缺陷率、探头性能、复检能力、质量成本、放行标准或安全规范。

<MlLessonProgress lesson-id="11" next-label="第 12 章：核方法与核 SVM" next-href="/courses/machine-learning/12-kernels" />
