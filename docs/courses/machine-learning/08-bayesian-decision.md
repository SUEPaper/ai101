---
course: machine-learning
edition: industrial
title: 贝叶斯决策与最优分类器
description: 以动力电池模组端检缺陷分流为贯穿案例，从后验期望损失推导 Bayes 决策、0-1 分类器、非对称代价阈值、拒绝动作与似然比检验，再建立 ROC、部署先验、校准、产能约束和运行点验收流程。
outline: [2, 3]
---

<MlLessonHero number="08" title="贝叶斯决策与最优分类器" question="端检模型给出 18% 的潜在缺陷概率后，模组应该放行、送离线复检，还是立即隔离？" duration="建议 480–600 分钟" :outcomes="['把真实状态、观测、后验、动作和损失拆成可审计的决策链', '从条件风险逐点最小化推导 Bayes 最优规则与 Bayes 风险', '推导一般二分类代价阈值、后验 odds 与似然比判据', '处理多分类代价矩阵、拒绝或复检动作和有限处理产能', '从冻结分数生成混淆矩阵、ROC、AUC、PR 与每千件绝对告警账本', '区分排序、概率校准、先验漂移和最终运行点，并建立上线验收协议']" />

第 07 章解决了怎样估计一条非线性条件均值。本章开始分类主线，但不急着选择逻辑回归、SVM 或神经网络。工业系统首先要回答的是：**看见不确定证据后，允许采取哪些动作，每种真实状态与动作组合会造成什么损失？**

概率、类别与动作是三层不同对象：

$$
x
\longrightarrow
\widehat p(y\mid x)
\longrightarrow
\widehat R(a\mid x)
\longrightarrow
a.
$$

一个模型可以拥有不错的排序能力，却因概率失准而给出错误成本阈值；也可以拥有相同 ROC，却因部署缺陷率变化而产生完全不同的误报数量。把“分类准确率”当作最终目标，会掩盖这条决策链中的关键假设。

## 贯穿案例：动力电池模组端检缺陷分流

动力电池模组下线前经过热像、超声和电参数检查。冻结模型输出潜在内部连接缺陷的后验估计。工厂允许三种动作：

1. **放行**：进入后续 Pack 装配；
2. **离线复检**：送 CT/拆检工位，付出等待与检验成本；
3. **隔离**：退出正常物流，等待质量工程师处置。

隐藏状态只有“合格”和“潜在缺陷”，但动作不必等于类别。教学损失矩阵为：

| 真实状态 $y$ | 放行 | 离线复检 | 隔离 |
|---|---:|---:|---:|
| 合格 $y=0$ | 0 | 8 | 30 |
| 潜在缺陷 $y=1$ | 120 | 8 | 3 |

矩阵中的数字是**相对成本单位**：缺陷件误放最严重；复检固定占用资源；隔离缺陷件仍有处置成本；良品误隔离造成较大报废与节拍损失。它们不是任何真实工厂的报价、安全阈值或放行规范。

真实项目必须由质量、安全、制造、财务、售后和法规责任人共同定义损失，并保留审批版本。模型输出不得绕过硬件保护、强制检验和法定质量门。

### 数据合同：一行是一只完成端检的模组

| 字段 | 单位 | 角色 | 主要风险 |
|---|---:|---|---|
| `production_lot_id` | — | 生产批次 | 切分、重采样和漂移监控单位 |
| `line_id` | 类别 | 产线 | 与班次、供应批可能混淆 |
| `inspection_time` | min | 批内顺序 | 未来信息和维护事件泄漏 |
| `thermal_gradient` | 标准分 | 热像特征 | 相机校准与环境温度漂移 |
| `ultrasound_energy` | 标准分 | 超声特征 | 耦合压力与探头磨损 |
| `line_speed` | 标准分 | 工艺协变量 | 改线后分布变化 |
| `model_probability` | 0–1 | 冻结模型输出 | 只有通过校准审计才有后验语义 |
| `ct_defect_label` | 0/1 | 参考标签 | 只复检高分件会造成验证偏差 |

本章假设所有教学样本最终都获得 CT/拆检参考标签，以便讨论决策评价；这在真实生产中可能过于昂贵。若只有被模型拦截的模组才有真值，直接计算的 FPR、精确率和校准曲线都会遭受**选择性验证偏差**。应设计随机审计样本或使用有明确假设的缺失标签方法。

相邻模组共享供应批、设备状态和环境条件，不能随机逐行切分。下面按完整生产批次划分开发、冻结验证和最终测试；本章不训练分类器，只把模型输出当作上游冻结产物。

```python
import numpy as np

rng = np.random.default_rng(20260805)
n_lots, units_per_lot = 40, 50
production_lot_id = np.repeat(np.arange(n_lots), units_per_lot)
inspection_time = np.tile(np.arange(units_per_lot), n_lots)
n_units = production_lot_id.size
line_id = production_lot_id % 2

lot_shift = rng.normal(0.0, 0.45, n_lots)
thermal_gradient = rng.normal(0.0, 1.0, n_units) + 0.35 * lot_shift[production_lot_id]
ultrasound_energy = rng.normal(0.0, 1.0, n_units) + 0.45 * lot_shift[production_lot_id]
line_speed = rng.normal(0.0, 1.0, n_units) + 0.12 * line_id

def sigmoid(value):
    value = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-value))

true_logit = (
    -3.85
    + 1.05 * thermal_gradient
    + 0.85 * ultrasound_energy
    - 0.20 * line_speed
    + 0.35 * lot_shift[production_lot_id]
)
true_defect_probability = sigmoid(true_logit)
ct_defect_label = rng.binomial(1, true_defect_probability)

# 冻结模型没有直接看到批次随机效应，因此存在有限的估计误差。
model_logit = (
    -3.72
    + 0.98 * thermal_gradient
    + 0.72 * ultrasound_energy
    - 0.18 * line_speed
    + rng.normal(0.0, 0.30, n_units)
)
model_probability = sigmoid(model_logit)

development_mask = production_lot_id < 26
validation_mask = (production_lot_id >= 26) & (production_lot_id < 33)
test_mask = production_lot_id >= 33

assert n_units == 2000
assert set(production_lot_id[validation_mask]).isdisjoint(
    set(production_lot_id[test_mask])
)
assert set(np.unique(ct_defect_label[validation_mask])) == {0, 1}
assert np.all((model_probability > 0) & (model_probability < 1))
print("开发/验证/测试：", development_mask.sum(), validation_mask.sum(), test_mask.sum())
print("总体教学缺陷率：", round(ct_defect_label.mean(), 4))
```

<MlQuiz kind="数据边界" question="只对模型拦截的高分模组做 CT，然后用这些 CT 标签估计精确率。最核心的问题是什么？" :options="['样本太多', '参考标签是否可见由模型动作决定，评价样本不是部署总体的无偏切片', 'CT 标签一定比模型差', '精确率只能用于回归']" :answer="1" explanation="被验证概率依赖模型分数，未拦截区域的真值系统性缺失。若不做随机审计或偏差校正，混淆矩阵和校准结论都不可代表总体。" />

## 1. 决策问题由五个对象组成

定义：

- $X\in\mathcal X$：可观测证据；
- $Y\in\mathcal Y$：真实但未知的状态；
- $A\in\mathcal A$：系统允许执行的动作；
- $L(y,a)$：真实状态为 $y$、执行动作 $a$ 时的损失；
- $\delta:\mathcal X\to\mathcal A$：把观测映射到动作的决策规则。

给定观测 $x$，动作 $a$ 的条件风险为

$$
R(a\mid x)
=\mathbb E[L(Y,a)\mid X=x]
=\sum_{y\in\mathcal Y}L(y,a)p(y\mid x).
$$

Bayes 决策规则逐点选择风险最小动作：

$$
\boxed{
\delta^*(x)\in\arg\min_{a\in\mathcal A}R(a\mid x)
}.
$$

若多个动作风险相同，需要预先定义确定的 tie-break，例如优先安全动作或优先较低产能占用；不要依赖数组顺序形成隐蔽业务规则。

### 为什么逐点最小就是总体最优

任意决策规则的总体风险为

$$
\mathcal R(\delta)
=\mathbb E[L(Y,\delta(X))]
=\mathbb E_X[R(\delta(X)\mid X)].
$$

对每个 $x$ 都有

$$
R(\delta^*(x)\mid x)
\le R(\delta(x)\mid x).
$$

两边对 $X$ 取期望：

$$
\mathcal R(\delta^*)\le\mathcal R(\delta).
$$

这证明了在**真实后验和真实损失已知**的理想条件下，$\delta^*$ 的期望风险不高于其他规则。现实系统只有估计后验 $\widehat p$、估计损失 $\widehat L$ 和有限验证样本，因此“套用 Bayes 公式”不等于已经达到 Bayes 最优。

<MlQuiz kind="概念辨析" question="Bayes 最优分类器的“最优”建立在哪个条件上？" :options="['任何训练算法都会自动满足', '真实后验、动作集合和损失函数已知且决策假设成立', '只要训练准确率最高', '只要阈值等于 0.5']" :answer="1" explanation="Bayes 规则是相对于指定信息、动作与损失的理论最优。后验估计误差、成本误设和分布漂移都会让实际规则偏离它。" />

<div class="ml-stop"><strong>停下来推一推 1</strong>证明给所有动作损失同时加上只依赖真实状态的常数 $c(y)$ 不改变最优动作。为什么给某一个动作加常数通常会改变决策？</div>

## 2. 0-1 损失下才是“取最大后验类别”

若动作就是 $K$ 个类别，0-1 损失为

$$
L(y,a)=\mathbf 1(y\ne a).
$$

动作 $a$ 的条件风险：

$$
\begin{aligned}
R(a\mid x)
&=\sum_y\mathbf 1(y\ne a)p(y\mid x)\\
&=1-p(Y=a\mid x).
\end{aligned}
$$

因此

$$
\delta_{01}^*(x)=\arg\max_{a\in\mathcal Y}p(Y=a\mid x).
$$

这是最大后验类别，也称 MAP 分类规则。它不是一般 Bayes 决策的定义，而是“动作等于类别、所有错误同价、正确动作零损失”的特例。

最小可达的 0-1 条件风险为

$$
R^*(x)=1-\max_y p(y\mid x),
$$

总体 Bayes error 为

$$
\mathcal R^*_{01}
=\mathbb E_X[1-\max_y p(y\mid X)].
$$

即使真实后验已知，只要不同类别的输入分布重叠，Bayes error 仍可能大于 0。这是观测信息不足产生的不可约分类不确定性，不应全部归咎于优化器。

<MlQuiz kind="损失判断" question="三分类中后验为 [0.42,0.35,0.23]。在哪种情况下直接选择第一类有理论依据？" :options="['任何代价矩阵', '0-1 损失且动作就是三个类别', '第一类编号最小', '第三类样本最少']" :answer="1" explanation="最大后验只最小化 0-1 条件风险。一般代价下必须计算每个动作的后验期望损失。" />

## 3. 二分类非对称代价阈值

记 $p=p(Y=1\mid x)$，损失按“真实状态为行、动作为列”排列：

$$
L=
\begin{bmatrix}
L_{00}&L_{01}\\
L_{10}&L_{11}
\end{bmatrix}.
$$

两个动作的条件风险为

$$
R(0\mid x)=L_{00}(1-p)+L_{10}p,
$$

$$
R(1\mid x)=L_{01}(1-p)+L_{11}p.
$$

定义相对误报损失和相对漏报损失：

$$
C_{FP}=L_{01}-L_{00},
\qquad
C_{FN}=L_{10}-L_{11}.
$$

当两者都为正时，选择动作 1 当且仅当

$$
C_{FP}(1-p)<C_{FN}p,
$$

即

$$
\boxed{
p>t^*=
\frac{C_{FP}}{C_{FP}+C_{FN}}
}.
$$

正确动作成本不为 0 时，阈值依赖的是**相对增量**，不是直接把 $L_{01}$ 与 $L_{10}$ 填入公式。若 $C_{FP}$ 或 $C_{FN}$ 非正，说明损失定义可能让“犯错”不比“做对”更差，最优规则可能退化，必须回到原始风险比较。

```python
def binary_bayes_threshold(loss_matrix):
    losses = np.asarray(loss_matrix, dtype=float)
    if losses.shape != (2, 2):
        raise ValueError("loss_matrix must have shape (2, 2)")
    false_positive_increment = losses[0, 1] - losses[0, 0]
    false_negative_increment = losses[1, 0] - losses[1, 1]
    denominator = false_positive_increment + false_negative_increment
    if false_positive_increment <= 0 or false_negative_increment <= 0:
        raise ValueError("error increments must both be positive")
    return false_positive_increment / denominator

binary_loss = np.array([
    [0.0, 4.0],
    [120.0, 3.0],
])
binary_threshold = binary_bayes_threshold(binary_loss)
assert np.isclose(binary_threshold, 4.0 / 121.0)
print("二动作缺陷阈值：", round(binary_threshold, 4))
```

<MlQuiz kind="阈值推导" question="正确隔离缺陷件仍有成本 3，误放缺陷件成本 120；误隔离良品成本 4、正确放行成本 0。Bayes 阈值是多少？" :options="['4/(4+120)', '4/(4+117)', '0.5', '117/(4+117)']" :answer="1" explanation="相对漏报损失是 120−3=117，相对误报损失是 4−0=4，因此阈值为 4/(4+117)。" />

### 后验 odds 与似然比判据

将阈值不等式改写为 odds：

$$
\frac{p(Y=1\mid x)}{p(Y=0\mid x)}
>
\frac{C_{FP}}{C_{FN}}.
$$

Bayes 公式给出

$$
\frac{p(Y=1\mid x)}{p(Y=0\mid x)}
=
\frac{p(x\mid Y=1)}{p(x\mid Y=0)}
\frac{\pi}{1-\pi},
$$

其中 $\pi=p(Y=1)$。于是似然比判据为

$$
\boxed{
\frac{p(x\mid Y=1)}{p(x\mid Y=0)}
>
\frac{C_{FP}}{C_{FN}}
\frac{1-\pi}{\pi}
}.
$$

这把三类信息分开：

- 似然比：当前证据更支持哪种状态；
- prior odds：部署前缺陷有多常见；
- cost ratio：两种错误的相对后果。

罕见缺陷的 prior odds 很小，需要更强证据才能在同成本下判正；高漏报代价又会降低证据门槛。两种效应不能混成“默认 0.5”。

<div class="ml-stop"><strong>停下来推一推 2</strong>从后验 odds 判据推导上式。固定类条件分布与成本时，部署缺陷率从 10% 降到 1%，似然比门槛怎样变化？</div>

## 4. 动作不必等于类别：复检与隔离

对一批后验矩阵

$$
P=
\begin{bmatrix}
p(y=0\mid x_1)&p(y=1\mid x_1)\\
\vdots&\vdots\\
p(y=0\mid x_n)&p(y=1\mid x_n)
\end{bmatrix},
$$

以及状态 × 动作损失矩阵 $L\in\mathbb R^{2\times3}$，全部条件风险为

$$
\mathcal C=PL.
$$

第 $i$ 行的最小元素给出第 $i$ 个模组的最小风险动作。

```python
action_names = np.array(["release", "offline_review", "quarantine"])
triage_loss = np.array([
    [0.0, 8.0, 30.0],
    [120.0, 8.0, 3.0],
])

def minimum_risk_actions(positive_probability, loss_matrix):
    probability = np.asarray(positive_probability, dtype=float)
    losses = np.asarray(loss_matrix, dtype=float)
    if losses.ndim != 2 or losses.shape[0] != 2:
        raise ValueError("loss_matrix rows must correspond to states 0 and 1")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probability must lie in [0, 1]")
    posterior = np.column_stack([1.0 - probability, probability])
    conditional_risks = posterior @ losses
    action_index = np.argmin(conditional_risks, axis=1)
    return action_index, conditional_risks

demo_probability = np.array([0.02, 0.18, 0.92])
demo_action_index, demo_risks = minimum_risk_actions(
    demo_probability,
    triage_loss,
)
assert np.array_equal(action_names[demo_action_index], [
    "release", "offline_review", "quarantine",
])
print(np.round(demo_risks, 2))
```

<BayesRiskDispatchLab />

实验 08-A 的三条线分别是放行、复检和隔离的条件风险。底部色带不是人工设置的概率区间，而是逐点比较三条风险线得到的动作区域。

在默认成本下：

$$
R_{release}(p)=120p,
\qquad
R_{review}(p)=8,
\qquad
R_{quarantine}(p)=30(1-p)+3p.
$$

放行与复检交点为 $p=8/120$；复检与隔离交点为

$$
p=\frac{30-8}{30-3}.
$$

复检不是“模型不确定时固定取 0.5 附近”的启发式，它必须作为一个有成本、有容量、有时延和后续信息质量的正式动作。

### 完美复检是假设，不是事实

实验把复检风险简化为常数 8，相当于复检最终揭示真值且不造成状态相关后果。真实复检可能：

- 仍有漏检与误检；
- 对良品造成拆解损伤；
- 延迟交付并占用 WIP；
- 在高负荷时成本上升；
- 只给出另一份相关而非独立证据。

应把这些结果展开成更细状态或后续决策树。若复检后还要根据新观测继续行动，问题进入“信息价值”和序贯决策，而不是简单三列矩阵。

<MlQuiz kind="动作设计" question="为什么“离线复检”不应被当作没有成本的万能拒绝动作？" :options="['复检不产生任何信息', '它有产能、时延、检测误差和潜在拆解损伤，应进入损失或约束', 'Bayes 决策只允许两个动作', '加入复检会使概率失效']" :answer="1" explanation="拒绝或复检是合法动作，但必须描述其真实后果。成本设为 0 会让系统几乎总选复检，掩盖资源约束。" />

## 5. 多分类与一般代价矩阵

若有 $K$ 个状态、$M$ 个动作，后验向量为

$$
p(x)=[p_1(x),\ldots,p_K(x)]^T,
$$

损失矩阵 $L\in\mathbb R^{K\times M}$。全部动作风险为

$$
r(x)^T=p(x)^TL.
$$

最优动作：

$$
a^*(x)\in\arg\min_{a=1,\ldots,M}r_a(x).
$$

需要先冻结矩阵方向。若把“真实状态为行、动作为列”误写成相反方向，代码仍可能矩阵乘法成功，却把业务语义完全颠倒。工业接口应同时保存：

- 状态名及顺序；
- 动作名及顺序；
- 损失单位、来源和批准版本；
- tie-break 规则；
- 生效日期与适用产品范围。

### Chow reject rule 的特殊情形

若 $K$ 类误分类统一成本为 $\lambda_e$，拒绝统一成本为 $\lambda_r$，且 $0<\lambda_r<\lambda_e$，最大后验为

$$
p^*(x)=\max_kp(Y=k\mid x).
$$

选最可能类别的风险是 $\lambda_e[1-p^*(x)]$，拒绝风险是 $\lambda_r$。因此拒绝当且仅当

$$
p^*(x)<1-\frac{\lambda_r}{\lambda_e}.
$$

真实多动作问题的成本通常不对称，不能把这条简式机械套到每个类别。

## 6. Bayes 分类器依赖后验从哪里来

### 生成式路线

生成式分类器建模

$$
p(x,y)=p(x\mid y)p(y),
$$

再由 Bayes 公式计算

$$
p(y=k\mid x)
=\frac{p(x\mid y=k)\pi_k}
{\sum_jp(x\mid y=j)\pi_j}.
$$

它需要描述每一类输入怎样生成。结构假设正确时，可利用无标签结构、处理某些缺失模式、生成样本或在小样本中获得稳定估计；结构假设错误时，密度误差会传入后验。

### 判别式路线

判别式概率模型直接学习 $p(y\mid x)$，逻辑回归属于此类；另一些方法只学习区分边界或排序分数 $s(x)$，不承诺概率语义。

| 问题 | 生成式 | 判别式概率 | 纯分数/边界 |
|---|---|---|---|
| 建模对象 | $p(x\mid y),p(y)$ | $p(y\mid x)$ | $s(x)$ 或边界 |
| 是否天然给后验 | 经 Bayes 公式 | 是 | 否，需额外校准 |
| 是否解释输入分布 | 是 | 否 | 否 |
| 典型风险 | 类条件分布误设 | 条件形式或优化误设 | 分数尺度没有成本语义 |
| 本课程后续 | 第 23 章 LDA/QDA/Naive Bayes | 第 09–10 章 | 第 11–12 章间隔与核 |

生成式与判别式是建模分解，不是“能否生成图片”的区别。两条路线最终都必须经过外部验证，才能把输出交给损失矩阵。

### 为什么直接线性回归 0/1 标签不够

平方损失在无限函数空间中的条件最优解确实是

$$
f^*(x)=\mathbb E[Y\mid X=x]=p(Y=1\mid x).
$$

问题不在平方损失完全没有概率意义，而在受限线性形式 $x^T\beta$：

- 输出不受 $[0,1]$ 约束；
- Bernoulli 方差 $p(1-p)$ 天然异方差；
- 高 leverage 样本可把直线拉出概率域；
- 线性均值形式常不适合 S 形后验；
- 即使阈值分类尚可，数值也不能未经校准直接进入成本公式。

下一章从 Bernoulli likelihood 推导逻辑回归，给出受约束概率和可优化的条件模型。

<MlQuiz kind="路线辨析" question="一个 SVM 输出 margin 分数 2.7。能否直接把它当作 97% 概率送入损失矩阵？" :options="['可以，任何正分数都是概率', '不可以；margin 有排序和间隔语义，需独立概率校准与验证', '只要 AUC 高就可以', '只在训练集可以']" :answer="1" explanation="分数尺度通常不是后验概率。单调变换不改变 ROC，却会改变固定数值阈值和期望风险计算。" />

## 7. 分数、概率与校准

一个二分类模型可能只输出排序分数 $s(x)$。若 $g$ 严格单调，则 $g(s)$ 与 $s$ 的样本排序相同，所以 ROC 理论上相同；但 $g(s)$ 的数值不一定满足

$$
p(Y=1\mid\widehat p(X)=q)=q.
$$

上式是概率校准的理想定义。校准与区分能力是两条轴：

- **区分**：正例是否通常排在负例前面，可用 ROC-AUC；
- **校准**：预测 10% 的样本中是否约 10% 为正，可用可靠性图、Brier score、log loss；
- **决策价值**：在指定先验、代价与容量下是否降低风险。

Brier score 为

$$
\operatorname{Brier}
=\frac1n\sum_{i=1}^n(\widehat p_i-y_i)^2.
$$

它是 proper scoring rule：在期望意义下，报告真实概率最优。但一项平均分数仍会掩盖关键阈值附近和不同产线的局部失准。

```python
def brier_score(labels, probabilities):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    return float(np.mean((probabilities - labels) ** 2))

validation_y = ct_defect_label[validation_mask]
validation_probability = model_probability[validation_mask]
validation_brier = brier_score(validation_y, validation_probability)

# 严格单调的过度自信变换保持排序，却改变概率尺度。
validation_odds = validation_probability / (1.0 - validation_probability)
overconfident_probability = validation_odds ** 1.6 / (
    1.0 + validation_odds ** 1.6
)
overconfident_brier = brier_score(validation_y, overconfident_probability)

assert np.array_equal(
    np.argsort(validation_probability),
    np.argsort(overconfident_probability),
)
assert not np.isclose(validation_brier, overconfident_brier)
print("原始/过度自信 Brier：", round(validation_brier, 5), round(overconfident_brier, 5))
```

<MlQuiz kind="校准判断" question="对所有分数做严格单调变换后，哪项通常保持不变？" :options="['概率阈值 0.2 对应的动作', '样本排序与 ROC-AUC', 'Brier score', '每个概率箱的真实缺陷率']" :answer="1" explanation="严格单调变换保持两两排序，因此 ROC 不变；但概率数值、校准指标和成本阈值动作都可能改变。" />

## 8. 混淆矩阵与 ROC 运行轨道

对冻结阈值 $t$，预测

$$
\widehat y_i(t)=\mathbf 1[s_i\ge t].
$$

混淆矩阵包含 TP、FP、TN、FN。行内条件率为

$$
\operatorname{TPR}(t)
=\frac{TP}{TP+FN},
\qquad
\operatorname{FPR}(t)
=\frac{FP}{FP+TN}.
$$

阈值从 $+\infty$ 降到 $-\infty$，得到 ROC 曲线

$$
(\operatorname{FPR}(t),\operatorname{TPR}(t)).
$$

有限验证集上，它是由水平和垂直线段组成的阶梯。阈值选择必须在冻结验证集完成；测试集只用于一次最终运行点验收。

```python
def confusion_at_threshold(labels, scores, threshold_value):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(scores, dtype=float) >= threshold_value
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    tn = int(np.sum(~predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    return tp, fp, tn, fn

def roc_curve_from_scores(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.concatenate([
        [np.inf],
        np.sort(np.unique(scores))[::-1],
        [-np.inf],
    ])
    rows = []
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires both classes")
    for threshold_value in thresholds:
        tp, fp, tn, fn = confusion_at_threshold(labels, scores, threshold_value)
        rows.append((threshold_value, tp / positives, fp / negatives))
    return np.asarray(rows, dtype=float)

validation_roc = roc_curve_from_scores(validation_y, validation_probability)
roc_thresholds = validation_roc[:, 0]
validation_tpr = validation_roc[:, 1]
validation_fpr = validation_roc[:, 2]
validation_auc = float(np.sum(
    np.diff(validation_fpr)
    * (validation_tpr[:-1] + validation_tpr[1:])
    / 2.0
))

assert np.all(np.diff(validation_tpr) >= -1e-12)
assert np.all(np.diff(validation_fpr) >= -1e-12)
assert 0.5 < validation_auc <= 1.0
print("冻结验证 ROC-AUC：", round(validation_auc, 4))
```

### AUC 的含义与边界

无并列分数时，

$$
\operatorname{AUC}
=P(s(X^+)>s(X^-)).
$$

它是随机抽取一个正例和负例时，正例排在前面的概率。AUC 汇总全部阈值，不知道：

- 部署缺陷率；
- 误报和漏报代价；
- 每班绝对告警量；
- 复检工位容量；
- 概率是否校准；
- 工厂只关心的极低 FPR 区域。

一个全局 AUC 更高的模型，可能在允许 FPR≤0.5% 的局部区域更差。验收必须声明目标运行区。

## 9. 部署先验把率转换为绝对账本

设部署缺陷率为 $\pi$、每期产量为 $N$。期望计数为

$$
TP=N\pi\operatorname{TPR},
\qquad
FN=N\pi(1-\operatorname{TPR}),
$$

$$
FP=N(1-\pi)\operatorname{FPR},
\qquad
TN=N(1-\pi)(1-\operatorname{FPR}).
$$

部署精确率为

$$
\operatorname{Precision}
=\frac{\pi\operatorname{TPR}}
{\pi\operatorname{TPR}+(1-\pi)\operatorname{FPR}}.
$$

所以 ROC 的 TPR/FPR 在不同类别先验下可以不变，而精确率和绝对误报量显著变化。罕见缺陷下，即使 FPR 很小，误报仍可能多于真阳性。

忽略正确动作的共同成本，单件期望决策成本为

$$
C(t)=
\pi C_{FN}[1-\operatorname{TPR}(t)]
+(1-\pi)C_{FP}\operatorname{FPR}(t).
$$

固定等成本得到 ROC 平面上的直线，斜率为

$$
m=
\frac{(1-\pi)C_{FP}}
{\pi C_{FN}}.
$$

改变先验或代价不会改变冻结验证集 ROC 轨道，但会改变等成本线斜率与切点。

```python
deployment_prevalence = 0.04
false_negative_cost = 120.0
false_positive_cost = 4.0

validation_cost = (
    deployment_prevalence * false_negative_cost * (1.0 - validation_tpr)
    + (1.0 - deployment_prevalence) * false_positive_cost * validation_fpr
)
finite_threshold = np.isfinite(roc_thresholds)
candidate_index = np.flatnonzero(finite_threshold)
best_local_index = candidate_index[np.argmin(validation_cost[finite_threshold])]
selected_threshold = float(roc_thresholds[best_local_index])
selected_validation_cost = float(validation_cost[best_local_index])

assert 0.0 < selected_threshold < 1.0
assert selected_validation_cost <= validation_cost[finite_threshold].min() + 1e-12
print("成本选择阈值/每千件成本：", round(selected_threshold, 4), round(1000 * selected_validation_cost, 2))
```

<RocOperatingPointLab />

实验 08-B 中，阈值滑块只让黄色触点沿同一条 ROC 轨道移动。改变部署缺陷率和漏放代价，会移动红色等成本线与成本最优触点，但不会重新训练模型或改变验证样本排序。

右侧“每千件账本”把率翻译成可排产的数量。工程评审不应只说“FPR=1% 很低”，而要说“在每班 20,000 件、缺陷率 0.5% 时，预期多少误报进入 CT 工位，是否超过班次容量”。

<MlQuiz kind="ROC 判断" question="部署缺陷率从 10% 降到 1%，类条件分数分布不变。哪项最准确？" :options="['ROC 必然下降，精确率不变', 'ROC 可保持不变，但精确率通常下降且绝对误报占比上升', 'AUC 自动变成 0', '最优阈值必然仍是 0.5']" :answer="1" explanation="TPR/FPR 是类内条件率，不直接依赖先验；精确率和期望成本含部署先验，因此会变化。" />

<div class="ml-stop"><strong>停下来推一推 3</strong>从期望成本公式推导等成本线斜率。为什么漏报代价增大时斜率下降，最优切点通常向更高 TPR 移动？</div>

## 10. ROC、PR 与容量约束回答不同问题

### PR 曲线

Recall 就是 TPR，Precision 依赖 $\pi$：

$$
\operatorname{Precision}(t)
=\frac{TP(t)}{TP(t)+FP(t)}.
$$

罕见事件中，PR 曲线通常比全范围 ROC 更直接显示“被拦截样本中有多少是真缺陷”。但跨不同先验的数据集比较 PR-AUC 时，必须注明基准缺陷率；先验不同本身就会改变精确率。

### 固定 FPR 或固定召回

有时业务不愿把后果压成单一货币成本：

- 安全规范要求召回率至少 99%；
- CT 工位要求每千件误报不超过 12；
- 人工复核每日最多 300 件；
- 某类严重缺陷必须零容忍并走独立规则。

此时应解约束问题，例如

$$
\max_t \operatorname{TPR}(t)
\quad\text{s.t.}\quad
\operatorname{FPR}(t)\le\alpha.
$$

这与“用一组主观成本加权”是不同规范。Neyman–Pearson 框架在已知分布的简单假设下说明 likelihood ratio test 对固定一类错误约束的最强性；有限工业数据仍需给 TPR/FPR 置信区间，而不是只用点估计碰线。

### 容量约束是群体耦合

逐件 Bayes 规则假设每个动作成本独立。如果一个班次只能复检 $K$ 件，模组之间通过共享容量耦合。常见策略是在冻结候选池上按“复检相对直接动作的风险降低量”排序，再在容量内选择，而不是简单拿概率最高的 $K$ 件：

$$
\Delta_i=
\min(R_{release,i},R_{quarantine,i})
-R_{review,i}.
$$

只有 $\Delta_i>0$ 才有正的信息/行动价值。若处理时延会改变缺陷后果，还需要排程模型。

<MlQuiz kind="容量约束" question="复检工位每班最多处理 100 件。为什么逐件选择最低风险动作可能不可执行？" :options="['Bayes 风险不能用于多件样本', '每件独立选择可能同时把超过 100 件送复检，共享容量使动作耦合', '阈值只能是整数', 'ROC 不允许复检']" :answer="1" explanation="无容量约束时可逐点最小化；共享资源使联合动作集合受限，需要在批次层面优化风险降低量与排程。" />

## 11. 训练先验、部署先验与 label shift

过采样缺陷件、病例对照采样或人工构造平衡训练集会改变训练类别比例。训练集正例占 50%，不代表部署缺陷率是 50%。

若满足 label shift：

$$
p_{deploy}(x\mid y)=p_{train}(x\mid y),
$$

但先验从 $\pi_{train}$ 变到 $\pi_{deploy}$，后验 odds 可修正为

$$
\frac{p_{deploy}(1\mid x)}{p_{deploy}(0\mid x)}
=
\frac{p_{train}(1\mid x)}{p_{train}(0\mid x)}
\frac{\pi_{deploy}/(1-\pi_{deploy})}
{\pi_{train}/(1-\pi_{train})}.
$$

```python
def adjust_probability_for_prior_shift(probability, training_prior, deployment_prior):
    probability = np.asarray(probability, dtype=float)
    if not (0 < training_prior < 1 and 0 < deployment_prior < 1):
        raise ValueError("priors must lie strictly between zero and one")
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    training_odds = clipped / (1.0 - clipped)
    prior_odds_ratio = (
        deployment_prior / (1.0 - deployment_prior)
        / (training_prior / (1.0 - training_prior))
    )
    deployment_odds = training_odds * prior_odds_ratio
    return deployment_odds / (1.0 + deployment_odds)

balanced_training_probability = np.array([0.2, 0.5, 0.8])
shifted_probability = adjust_probability_for_prior_shift(
    balanced_training_probability,
    training_prior=0.5,
    deployment_prior=0.04,
)
assert np.all(shifted_probability < balanced_training_probability)
assert np.array_equal(
    np.argsort(shifted_probability),
    np.argsort(balanced_training_probability),
)
print("先验修正：", np.round(shifted_probability, 4))
```

这项修正不改变排序，却改变概率和决策阈值。它依赖类条件分布不变；若原材料、设备、传感器或缺陷机制也变了，简单 prior correction 不足。应同时检查：

- $p(x\mid y)$ 稳定性；
- 标签定义与检验流程是否变化；
- 部署先验估计是否及时且无选择偏差；
- 修正后概率是否在独立样本上重新校准。

<MlQuiz kind="漂移判断" question="生产设备更新后，合格与缺陷模组的超声特征分布都改变。只做类别先验 odds 修正是否充分？" :options="['充分，因为所有漂移都是 label shift', '不充分；类条件分布已变，label-shift 假设被破坏', '只要 AUC 不看就行', '把训练先验设为 0.5 即可']" :answer="1" explanation="先验修正只处理 p(y) 变化且要求 p(x|y) 稳定。设备更新造成类条件漂移，需要重新验证或重训。" />

## 数值验证

首先验证风险矩阵乘法与解析阈值的一致性。对二动作损失，在阈值两侧，直接比较条件风险应给出相反动作。

```python
epsilon = 1e-5
probe_probability = np.array([
    binary_threshold - epsilon,
    binary_threshold + epsilon,
])
binary_action, binary_risk = minimum_risk_actions(
    probe_probability,
    binary_loss,
)
assert np.array_equal(binary_action, [0, 1])
assert binary_risk[0, 0] < binary_risk[0, 1]
assert binary_risk[1, 1] < binary_risk[1, 0]

# 在冻结阈值上只打开一次测试批次，形成最终运行点证据。
test_y = ct_defect_label[test_mask]
test_probability = model_probability[test_mask]
test_tp, test_fp, test_tn, test_fn = confusion_at_threshold(
    test_y,
    test_probability,
    selected_threshold,
)
test_tpr = test_tp / (test_tp + test_fn)
test_fpr = test_fp / (test_fp + test_tn)
assert test_tp + test_fp + test_tn + test_fn == test_mask.sum()
assert 0.0 <= test_tpr <= 1.0
assert 0.0 <= test_fpr <= 1.0
print("测试 TP/FP/TN/FN：", test_tp, test_fp, test_tn, test_fn)
print("测试 TPR/FPR：", round(test_tpr, 4), round(test_fpr, 4))
```

再验证部署账本恒等式：TP+FN 等于期望真实缺陷数，FP+TN 等于期望合格数；改变先验不会改变固定类条件率，却会改变精确率。

```python
def deployment_ledger(population, prevalence, tpr, fpr):
    tp = population * prevalence * tpr
    fn = population * prevalence * (1.0 - tpr)
    fp = population * (1.0 - prevalence) * fpr
    tn = population * (1.0 - prevalence) * (1.0 - fpr)
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    return np.array([tp, fp, tn, fn]), precision

ledger_10pct, precision_10pct = deployment_ledger(10000, 0.10, test_tpr, test_fpr)
ledger_1pct, precision_1pct = deployment_ledger(10000, 0.01, test_tpr, test_fpr)

assert np.isclose(ledger_10pct[0] + ledger_10pct[3], 1000)
assert np.isclose(ledger_1pct[0] + ledger_1pct[3], 100)
assert precision_1pct <= precision_10pct
assert np.isclose(validation_roc[0, 1:].sum(), 0.0)
assert np.isclose(validation_roc[-1, 1:].sum(), 2.0)
print("10%/1% 先验精确率：", round(precision_10pct, 4), round(precision_1pct, 4))
```

数值通过不证明成本矩阵正确。验收必须分别覆盖：

- 数学实现：矩阵方向、阈值、率和计数恒等式；
- 统计估计：跨批次置信区间、校准误差、分层稳定性；
- 决策假设：成本、容量、时延、复检准确度；
- 运行治理：域外输入、模型版本、强制规则和人工覆盖。

<MlQuiz kind="数值验证" question="测试集上的阈值成本高于验证集后，哪种做法正确？" :options="['回到测试集继续搜索阈值直到成本最低', '保留冻结阈值并报告差异；若据此改阈值，需要新的未见测试批次', '删除高成本测试批次', '把测试集并入训练后仍称最终测试']" :answer="1" explanation="测试集用于一次无偏验收。看过测试结果再调阈值，会把测试集变成模型选择数据。" />

## 复杂度与工程边界

### 计算复杂度

Bayes 动作层本身通常不昂贵。对 $n$ 个样本、$K$ 个状态、$M$ 个动作：

| 操作 | 时间 | 存储 | 主要风险 |
|---|---:|---:|---|
| 后验 × 损失矩阵 | $O(nKM)$ | $O(nM)$ 或流式 $O(M)$ | 状态/动作顺序错位 |
| 二分类阈值动作 | $O(n)$ | $O(1)$ 额外 | 把分数误当概率 |
| 排序生成完整 ROC | $O(n\log n)$ | $O(n)$ | 在测试集选点 |
| 已排序阈值扫描 | $O(n)$ | $O(n)$ | ties 与样本权重处理 |
| 多候选成本/容量选择 | 候选数 × 验证成本 | 依候选而定 | 反复复用验证集 |

真正复杂的是上游后验模型、标签取得、成本治理和批次级约束，不是最后一次矩阵乘法。

### 估计不确定性

TPR、FPR、Precision 和成本都是有限样本估计。罕见缺陷下，测试集只有 7 个正例时，“检出 7/7”不是已经证明召回率 100%。至少应：

- 报告正负样本数与批次数；
- 对批次而非逐件 bootstrap；
- 给关键率的置信区间；
- 对目标 FPR 极低区积累足够负例；
- 预先规定零事件时的处理方式。

### 成本不确定性

误放代价可能是分布而非常数：后果取决于缺陷严重度、发现阶段、产品用途和召回范围。可以做情景分析：

1. 定义乐观、基准、悲观成本矩阵；
2. 对每个矩阵重新选择运行点；
3. 查看动作是否稳定；
4. 若决策对成本微小变化极敏感，要求更强证据或保守回退。

这不等同于随意挑一个让模型获胜的成本。

### 风险敏感不等于期望损失

本章采用风险中性期望损失。若极端事故的尾部风险不能被平均值接受，可能需要 chance constraint、CVaR、鲁棒优化或强制安全规则。不要把巨大但有限的成本塞进矩阵后声称已经覆盖所有安全治理。

## 12. 工业失效模式与诊断表

| 现象 | 根因 | 诊断证据 | 处置 |
|---|---|---|---|
| 验证 AUC 高，CT 工位仍爆量 | 缺陷率低、FPR 对绝对量仍大 | 每千件/每班账本 | 在目标先验和容量下选点 |
| 0.2 概率实际缺陷率只有 5% | 概率失准或先验变化 | 可靠性图、Brier、分批校准 | 独立重校准并查漂移 |
| 阈值上线后性能突然变好 | 只验证被拦截样本 | 随机审计低分区 | 修复选择性验证偏差 |
| 同批随机切分成绩异常好 | 批次泄漏 | 按 `production_lot_id` 重切 | 按独立生产批评估 |
| 不同产线动作率差异巨大 | 类条件漂移或传感器偏置 | 分线 ROC、校准与特征漂移 | 分层校准、修传感器或分模 |
| 复检区堆积 | 忽略共享容量与时延 | 队列长度、P95 等待时间 | 批次级受约束分配 |
| 成本最优阈值频繁跳动 | 正例少、成本不稳 | 批次 bootstrap 阈值分布 | 用稳健区间和变更审批 |
| 模型低风险却触发硬件告警 | 动作层与安全链冲突 | 规则优先级审计 | 强制安全规则覆盖模型 |

### 标签延迟与反馈回路

缺陷可能在售后才暴露，标签延迟导致近期批次无法用于完整评价。模型动作还会改变未来可见标签：

- 被隔离件更容易获得真值；
- 放行件只有部分在后续暴露；
- 调整阈值改变样本组成；
- 人工复检结果可能被当成“真值”而共享同一偏差。

必须记录“标签来源、可见时间、验证机制”，并把延迟成熟窗口与实时监控分开。

## 13. 从案例到上线验收

一个可部署的端检分流流程至少包含：

1. **定义状态**：明确“缺陷”依据、严重度和参考检验；
2. **冻结动作**：放行、复检、隔离以及硬规则优先级；
3. **批准损失**：记录相对量纲、区间、负责人和版本；
4. **建立信息边界**：按生产批切分，保留随机审计真值；
5. **验证模型输出**：同时检查排序、校准、分层误差和域外输入；
6. **选择运行点**：在部署先验、每班产量、成本与容量下冻结；
7. **锁定测试**：只做一次跨批次最终验收，报告区间；
8. **影子上线**：先记录建议动作，不自动改变物流；
9. **受控切换**：质量门审批后才允许模型参与分流；
10. **持续监控**：告警率、真值成熟度、队列、漂移、人工覆盖和版本。

模型不可删除法规规定的全检项目，也不可单独授权缺陷件放行。

<MlQuiz kind="上线治理" question="模型成本风险低于硬件规则，但硬件规则要求隔离。工业系统应怎样处理？" :options="['模型覆盖硬件规则', '按预先批准的优先级由强制安全规则覆盖模型，并记录冲突', '取两个动作平均', '随机选择']" :answer="1" explanation="模型决策层必须处于安全与法规边界之内。冲突应可审计，不能让统计成本静默绕过强制规则。" />

## 14. 章节练习

### 练习 1：从原始损失推导一般阈值

给定

$$
L=
\begin{bmatrix}
2&9\\
40&5
\end{bmatrix},
$$

不要先把对角线改成 0。直接比较两个条件风险，推导阈值，再用相对损失 $C_{FP},C_{FN}$ 复核。解释为什么给每一行同时减去行最小值不改变动作。

### 练习 2：多严重度状态

把隐藏状态扩展为“合格、一般缺陷、严重缺陷”，动作仍为“放行、复检、隔离”。设计一个 $3\times3$ 损失矩阵，使：

- 严重缺陷误放远高于一般缺陷误放；
- 一般缺陷进入复检优于直接隔离；
- 高严重缺陷后验时隔离占优。

在概率三角形上画出动作区域，并检查每条边界为何是直线。

### 练习 3：ROC 左上角局部验收

在冻结验证批次上计算完整 ROC，但只允许 FPR≤1%。报告：

- 可行阈值集合；
- 最大 TPR 及批次 bootstrap 置信区间；
- 每班 20,000 件、缺陷率 0.5% 的 TP/FP/FN；
- 若 CT 容量为 250 件，运行点是否可执行。

不要用全局 AUC 代替局部结论。

### 练习 4：校准不改变排序的实验

对同一分数应用 $g_\gamma(s)=\sigma(\gamma\operatorname{logit}(s))$，比较 $\gamma\in\lbrace0.6,1,1.8\rbrace$ 下的：

- ROC-AUC；
- Brier score；
- 0.1 概率阈值动作率；
- 指定成本下的经验风险。

解释为什么 AUC 相同而决策结果不同。

### 练习 5：受容量约束的复检分配

计算每件模组送复检相对直接动作的风险降低量 $\Delta_i$。只允许复检 50 件，选择总风险最低的联合动作。比较“按概率最高选 50 件”和“按 $\Delta_i$ 最高选 50 件”。

### 练习 6：工程代码骨架

补齐分组 bootstrap、阈值稳定性和运行域拒绝。不得在函数内部读取测试集。

```python
# TODO: 练习骨架，验证脚本会跳过本代码块。
def grouped_threshold_bootstrap(
    labels,
    probabilities,
    group_id,
    deployment_prevalence,
    loss_matrix,
    n_resamples=500,
):
    """按完整生产批重采样，返回阈值、TPR、FPR 和成本分布。"""
    raise NotImplementedError

def guarded_decision(probability, domain_status, loss_matrix):
    """域外样本不得伪装成已校准的最小风险动作。"""
    raise NotImplementedError
```

### 练习 7：口头审查

不看公式，用两分钟回答：

1. 为什么最大后验类别只是 Bayes 决策的特例？
2. 正确动作成本非零时，二分类阈值怎样计算？
3. 同一条 ROC 为什么能对应多个最优运行点？
4. AUC 高为什么不证明概率可直接进入成本矩阵？
5. 部署先验怎样影响精确率和每千件误报？
6. 为什么只复检高分件会破坏评价？
7. 复检容量怎样使逐件决策变成联合优化？

## 15. 本章小结

- Bayes 决策以真实状态、后验、动作和损失为核心，逐点最小化条件风险；
- 最大后验类别只对应动作等于类别且采用 0-1 损失的特殊情况；
- 非对称二分类阈值由相对误报与漏报损失决定，也可写成 prior odds 加 likelihood ratio 判据；
- 拒绝、复检和隔离必须作为正式动作写入损失或容量约束；
- ROC 描述阈值排序轨道，校准决定概率尺度，部署先验把条件率转成绝对账本；
- 工业验收必须治理选择性验证、批次泄漏、标签延迟、成本不确定性、产能和强制安全规则。

## 来源与许可

- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 197–201 页，第 5.1.1–5.1.2 节：后验期望损失、Bayes 决策、0-1 损失、代价敏感分类与 reject option。用于核对推导边界；电池端检案例、损失矩阵、代码、题目和风险继电器均独立创作。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 201–206 页，第 5.1.3–5.1.4 节：混淆矩阵、TPR/FPR、ROC、AUC、类别不平衡与 PR 曲线。仅作知识与术语核对，不复用原图、案例或表达。
- Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 38–41 页（讲义页码 32–35）：Bayes classifier、非对称损失、Bayes risk、先验加权类条件分布、生成式与判别式路线。用于交叉核对，不复制癌症案例、图形或课堂措辞。
- Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 70–71 页（讲义页码 64–65）：验证/测试集 ROC、TPR/FPR、阈值轨道、AUC 与实际运行点选择。仅作评价边界核对，网页 ROC 验收轨道独立创作。

*Probabilistic Machine Learning: An Introduction* 按其 `CC BY-NC-ND 4.0` 许可边界处理：本章只把它作为知识参考，不翻译、改写或重绘原文材料。CS 189/289A 讲义未确认可再分发许可，本站只核对公共知识。详细登记见 [机器学习教材来源与许可登记](./sources) 。

<MlLessonProgress lesson-id="08" next-label="第 09 章：逻辑回归" next-href="/courses/machine-learning/09-logistic-regression" />
