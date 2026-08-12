---
title: PAC 学习与 VC 维
description: 从连接器视觉终检资格审查出发，推导 Hoeffding、有限假设类 PAC 保证、增长函数与 VC 维，并用成对 Bootstrap 审查真实改进
course: machine-learning
edition: industrial
---

<MlLessonHero
  number="28"
  title="PAC 学习与 VC 维"
  question="训练误差、验证集冠军和冻结测试点估计都很好看时，怎样证明候选模型的改进不是多重尝试、批次相关、概率失准或稀少规格恶化制造的假象？"
  duration="建议 600–720 分钟"
  :outcomes="[
    '区分固定假设界、选择后统一界与冻结测试置信区间，写清每种概率声明的随机对象和适用条件',
    '完整推导有限假设类的 realizable 与 agnostic PAC 样本复杂度，并用一致收敛证明 ERM 超额风险界',
    '用增长函数、shattering、Sauer 引理与 VC 维解释无限假设类的有效容量和最坏情况边界',
    '把正则化、结构风险最小化、多重尝试、验证集复用和校准纳入可审计的模型选择协议',
    '按生产 lot 成对比较模型，并把分布偏移与关键群体错误写成不可补偿的发布门禁'
  ]"
/>

> 本章的概率保证回答的是：在明确的数据生成假设下，重复抽取训练集时，坏事件有多罕见。它不证明当前这一次部署必然安全，也不替代冻结测试、工艺极限验证、失效模式分析和上线监控。

## 贯穿案例：高压连接器压接截面视觉终检资格审查

### 1. 业务决策不是“谁的验证集分数更高”

某高压连接器产线用截面图像识别压接缺陷。旧规则只看空洞率、毛刺率和填充比；算法组准备提交三个候选：加入工艺特征的 `process`、加入照明特征的 `appearance`、再加入交互项的 `wide`。候选一旦放行，就会影响人工复检和批次隔离，资格审查必须回答四个问题：

1. 从多个候选中挑出的最低经验误差，是否只是一次幸运波动？
2. 训练、验证、冻结测试是否按独立生产 lot 切分，而不是把同一 lot 的相邻图像拆散？
3. 总体误差略降时，稀少规格 C 是否反而恶化？
4. 二分类错误率相同的概率输出，是否具有相同校准质量？

教学数据包含 72 个生产 lot，每 lot 12 张图，共 864 张；48/12/12 个 lot 分别用于训练、验证和冻结测试。三种规格 A/B/C 的占比不均，规格 C 的照明分布又不同。下面只依赖 NumPy，生成过程和结果完全可重放。

```python
import math
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_connector_case(seed=2808):
    rng = np.random.default_rng(seed)
    rows = []
    for lot_id in range(72):
        slot = lot_id % 6
        variant = 0 if slot < 3 else (1 if slot < 5 else 2)
        lot_shift = rng.normal(0, 0.035, 4)
        for _ in range(12):
            void = np.clip(
                rng.normal([0.22, 0.28, 0.34][variant] + lot_shift[0], 0.10),
                0,
                1,
            )
            burr = np.clip(
                rng.normal([0.20, 0.25, 0.31][variant] + lot_shift[1], 0.09),
                0,
                1,
            )
            fill = np.clip(
                rng.normal([0.72, 0.70, 0.67][variant] + lot_shift[2], 0.075),
                0.35,
                0.95,
            )
            lighting = np.clip(
                rng.normal([0.18, 0.24, 0.45][variant] + lot_shift[3], 0.08),
                0,
                1,
            )
            latent = (
                -4.25
                + 5.0 * void
                + 3.0 * burr
                + 5.2 * abs(fill - 0.72)
                + 0.55 * (variant == 2)
            )
            defect = int(rng.random() < sigmoid(latent))
            rows.append((lot_id, variant, void, burr, fill, lighting, defect))
    return np.asarray(rows, dtype=float)


CASE = make_connector_case()
LOT = CASE[:, 0].astype(int)
VARIANT = CASE[:, 1].astype(int)
VOID, BURR, FILL, LIGHT = CASE[:, 2:6].T
Y = CASE[:, 6].astype(int)

TRAIN = LOT < 48
VALID = (LOT >= 48) & (LOT < 60)
TEST = LOT >= 60

print(TRAIN.sum(), VALID.sum(), TEST.sum())
print(np.round([Y[TRAIN].mean(), Y[VALID].mean(), Y[TEST].mean()], 3))
```

预期输出：

```text
576 144 144
[0.170 0.111 0.153]
```

缺陷率本身已经随 lot 切分而波动。若把 864 张图像当成 864 次完全独立的工艺重复，后续置信度会被夸大；本章在模型选择时先讲样本级理论，在最终比较时回到独立 lot。

<MlQuiz
  kind="抽样单位"
  question="同一生产 lot 的 12 张相邻截面图高度相关，资格审查最稳妥的切分单位是什么？"
  :options="['像素', '单张图像', '生产 lot', '随机数据行']"
  :answer="2"
  explanation="训练、验证和测试应以最接近独立生成机制的 lot 为单位切分，避免同批工艺状态跨集合泄漏。"
/>

### 2. 先冻结候选协议，再看验证结果

每个分数函数只在训练集的 121 个预先规定阈值中做经验风险最小化；验证集只负责从四个分数家族中选一个，冻结测试在选择完成前不可见。

```python
def fit_threshold(score, y, mask):
    grid = np.linspace(
        np.quantile(score[mask], 0.05),
        np.quantile(score[mask], 0.95),
        121,
    )
    errors = np.array([
        np.mean((score[mask] >= threshold) != y[mask])
        for threshold in grid
    ])
    best = int(np.argmin(errors))
    return float(grid[best]), float(errors[best])


SCORES = {
    "legacy": 4.0 * VOID + 1.7 * BURR + 1.4 * np.abs(FILL - 0.72),
    "process": 4.8 * VOID + 2.6 * BURR + 4.1 * np.abs(FILL - 0.72),
    "appearance": (
        4.8 * VOID + 2.6 * BURR + 4.1 * np.abs(FILL - 0.72) + 1.6 * LIGHT
    ),
    "wide": (
        5.2 * VOID + 3.1 * BURR + 5.0 * np.abs(FILL - 0.72)
        + 0.7 * LIGHT + 0.7 * VOID * BURR
    ),
}

FITS = {}
for name, score in SCORES.items():
    threshold, train_error = fit_threshold(score, Y, TRAIN)
    prediction = (score >= threshold).astype(int)
    FITS[name] = {
        "threshold": threshold,
        "train_error": train_error,
        "valid_error": float(np.mean(prediction[VALID] != Y[VALID])),
        "prediction": prediction,
    }

for name, fit in FITS.items():
    print(
        f"{name:10s}",
        f"train={fit['train_error']:.3f}",
        f"valid={fit['valid_error']:.3f}",
    )

SELECTED_NAME = min(FITS, key=lambda name: FITS[name]["valid_error"])
print("selected:", SELECTED_NAME)
```

预期输出：

```text
legacy     train=0.163 valid=0.139
process    train=0.149 valid=0.118
appearance train=0.153 valid=0.104
wide       train=0.153 valid=0.111
selected: appearance
```

这一步只是完成选择，不是宣布 `appearance` 优于旧规则。验证集最小值已经经历 4 个家族、每家族 121 个阈值的比较；如果团队还反复改特征、随机种子和筛选口径，实际尝试数会更大。

## 3. 先声明概率空间：什么在随机，什么被固定

令单个观测 $Z=(X,Y)$ 来自未知分布 $mathcal D$，损失 $ell(h(X),Y)\in[0,1]$。真实风险是对未来同分布样本的期望：

$$
R(h)=\mathbb E_{Z\sim\mathcal D}[\ell(h(X),Y)].
$$

训练集 $S=(Z_1,\ldots,Z_n)\sim\mathcal D^n$ 的经验风险是

$$
\widehat R_S(h)=\frac1n\sum_{i=1}^n\ell(h(X_i),Y_i).
$$

学习算法 $A$ 把随机数据集映射为 $\widehat h=A(S)$。于是要区分三层对象：

| 对象 | 是否依赖当前样本 $S$ | 常见用途 |
|---|---:|---|
| 看数据前固定的 $h$ | 否 | 单模型估计、预注册基线 |
| 候选集合 $\mathcal H$ | 协议冻结后否 | 统一界、ERM、模型选择 |
| 算法输出 $\widehat h=A(S)$ | 是 | 最终选中的模型 |

Hoeffding 概率是对重复抽取 $S$ 而言，不是对一个已经观测到的参数值下注。若 $\mathcal D$ 在部署时改变，或 $Z_i$ 因同 lot 而相关，后面的 i.i.d. 保证不再原样成立。

<MlQuiz
  kind="概率对象"
  question="PAC 表达式里的概率通常对什么取？"
  :options="['已经固定的真实风险本身', '重复抽取训练集及算法由此产生的输出', '工程师的主观信心', '只对模型参数初始化']"
  :answer="1"
  explanation="数据集是随机对象，学习算法的输出随数据集变化；PAC 控制这种重复抽样下的失败概率。"
/>

## 4. 固定假设：Hoeffding 只覆盖“看数据前已经定好”

先固定一个与 $S$ 无关的 $h$，令

$$
U_i=\ell(h(X_i),Y_i)\in[0,1],\qquad
\mathbb E[U_i]=R(h).
$$

若 $U_1,\ldots,U_n$ 独立同分布，Hoeffding 不等式给出双侧尾界：

$$
\Pr\!\left(\left|\widehat R_S(h)-R(h)\right|>\epsilon\right)
\le 2\exp(-2n\epsilon^2).
$$

令右侧等于 $\delta$ 并解 $\epsilon$：

$$
\boxed{
\left|R(h)-\widehat R_S(h)\right|
\le
\sqrt{\frac{\ln(2/\delta)}{2n}}
}
\quad\text{with probability at least }1-\delta.
$$

它有三个值得背后的量纲关系：半宽按 $n^{-1/2}$ 缩小；置信度通过 $\ln(1/\delta)$ 收费；损失必须有界，且独立性不是装饰。

```python
def hoeffding_gap(n, delta):
    return math.sqrt(math.log(2 / delta) / (2 * n))


for n in (100, 400, 1600):
    print(n, round(hoeffding_gap(n, 0.05), 4))


def invert_fixed_hoeffding(epsilon, delta):
    return math.ceil(math.log(2 / delta) / (2 * epsilon**2))


print("epsilon=0.03 requires n =", invert_fixed_hoeffding(0.03, 0.05))
```

预期输出：

```text
100 0.1358
400 0.0679
1600 0.034
epsilon=0.03 requires n = 2050
```

固定模型并不等于“训练完成后把最终模型名字写下来”。如果模型、阈值或特征是看完这批数据才选出的，它就不是独立于 $S$ 的固定 $h$。

<MlQuiz
  kind="界的尺度"
  question="想把 Hoeffding 半宽从 0.06 降到约 0.03，其他条件不变，样本量大约要怎样变化？"
  :options="['减半', '不变', '翻倍', '变为四倍']"
  :answer="3"
  explanation="半宽与 1/√n 同阶；半宽减半需要约四倍独立样本。"
/>

::: warning 停下来推一推 1
<!-- ml-stop -->
团队先在训练集上试 5000 个特征组合，再把最好的一个称为“固定模型”套用上一节公式。指出被偷换的随机对象，并写出至少一种合法修正：预注册单模型、对全部候选做统一控制，或另留完全未参与选择的数据。
:::

## 5. 从一个候选到全部候选：Union Bound 与一致收敛

设有限候选类 $\mathcal H=\{h_1,\ldots,h_M\}$ 在抽样前已固定。为每个候选定义失败事件

$$
A_h=\left\{S:\left|\widehat R_S(h)-R(h)\right|>\epsilon\right\}.
$$

不要求各事件独立，并合界始终有

$$
\Pr\left(\bigcup_{h\in\mathcal H}A_h\right)
\le\sum_{h\in\mathcal H}\Pr(A_h)
\le 2M e^{-2n\epsilon^2}.
$$

因此，以至少 $1-\delta$ 的概率，以下命题对所有候选**同时**成立：

$$
\boxed{
\sup_{h\in\mathcal H}
\left|R(h)-\widehat R_S(h)\right|
\le
\sqrt{\frac{\ln(2M/\delta)}{2n}}
}.
$$

数据依赖地挑出的 $\widehat h$ 也是 $\mathcal H$ 的成员，所以统一事件自动覆盖它。这里的复杂度是 $\ln M$，但“只按对数收费”绝不等于尝试免费：当 $n$ 不大、$M$ 很大时，上界仍会空泛到接近 1。

```python
def finite_class_gap(n, hypotheses, delta):
    return math.sqrt(math.log(2 * hypotheses / delta) / (2 * n))


print("fixed:", round(hoeffding_gap(1000, 0.10), 4))
print("M=5000:", round(finite_class_gap(1000, 5000, 0.10), 4))
print("M=10^8:", round(finite_class_gap(1000, 10**8, 0.10), 4))
```

预期输出：

```text
fixed: 0.0387
M=5000: 0.0759
M=10^8: 0.1035
```

<PacConfidenceCorridorLab />

<MlQuiz
  kind="并合条件"
  question="并合界需要假设不同候选的失败事件彼此独立吗？"
  :options="['需要完全独立', '只需两两独立', '不需要', '只有 M<10 时不需要']"
  :answer="2"
  explanation="并合界 Pr(∪Aᵢ)≤ΣPr(Aᵢ) 对任意事件成立；它可能保守，但不要求事件独立。"
/>

### 5.1 多重尝试如何制造“冠军偏差”

下面让 5000 个候选的真实错误率都等于 0.15，只保留每轮经验误差最低者。固定假设半宽原本只有 0.0387，但“冠军”的乐观差距经常超过它；把 $M$ 纳入统一界后，模拟失败率回到 $\delta$ 以下。

```python
def simulate_winner_gap(
    n=1000,
    hypotheses=5000,
    true_risk=0.15,
    repeats=4000,
    seed=2811,
):
    rng = np.random.default_rng(seed)
    selected_gaps = []
    # 分块只为控制内存；统计问题没有改变。
    for _ in range(0, repeats, 500):
        empirical = rng.binomial(
            n,
            true_risk,
            size=(min(500, repeats - len(selected_gaps)), hypotheses),
        ) / n
        selected_gaps.extend(true_risk - empirical.min(axis=1))
    return np.asarray(selected_gaps)


WINNER_GAPS = simulate_winner_gap()
FIXED_GAP = hoeffding_gap(1000, 0.10)
UNIFORM_GAP = finite_class_gap(1000, 5000, 0.10)
print("mean winner gap:", round(WINNER_GAPS.mean(), 4))
print("naive failure:", round(np.mean(WINNER_GAPS > FIXED_GAP), 3))
print("uniform failure:", round(np.mean(WINNER_GAPS > UNIFORM_GAP), 3))
print("winner gap quantiles:", np.round(np.quantile(WINNER_GAPS, [0.5, 0.9, 0.99]), 3))
```

预期输出：

```text
mean winner gap: 0.0400
naive failure: 0.644
uniform failure: 0.000
winner gap quantiles: [0.040 0.044 0.050]
```

模拟不是 Hoeffding 的证明；它展示了一个机制：即使所有候选同样差，只要尝试足够多，也会出现一个看起来异常好的经验冠军。

<MlQuiz
  kind="选择偏差"
  question="5000 个真实风险相同的候选中，经验风险最低者通常会怎样？"
  :options="['经验风险系统性偏高', '经验风险系统性偏低', '必然等于真实风险', '不受候选数影响']"
  :answer="1"
  explanation="选择最小值会收集负向采样波动，形成 winner's curse；选择协议必须纳入统计控制。"
/>

## 6. 一致收敛如何变成 ERM 的学习保证

令经验风险最小化解与类内真实最优分别为

$$
\widehat h\in\arg\min_{h\in\mathcal H}\widehat R(h),\qquad
h^\star\in\arg\min_{h\in\mathcal H}R(h).
$$

若事件

$$
\sup_{h\in\mathcal H}|R(h)-\widehat R(h)|\le\gamma
$$

成立，则有一条不能省略中间理由的链：

$$
\begin{aligned}
R(\widehat h)
&\le \widehat R(\widehat h)+\gamma
&&\text{一致偏差，对 }\widehat h\\
&\le \widehat R(h^\star)+\gamma
&&\text{ERM 定义}\\
&\le R(h^\star)+2\gamma
&&\text{一致偏差，对 }h^\star.
\end{aligned}
$$

因此

$$
\boxed{R(\widehat h)-R(h^\star)\le 2\gamma}.
$$

注意两个层次：这是相对 $\mathcal H$ 内最优的**估计误差**；若 $\mathcal H$ 本身表达不了真实规律，还存在近似误差 $R(h^\star)-R^\star$。把类做大通常降低近似误差，却提高估计难度。

```python
def erm_excess_certificate(n, hypotheses, delta):
    return 2 * finite_class_gap(n, hypotheses, delta)


print(round(erm_excess_certificate(576, 4 * 121, 0.05), 3))
```

这会输出约 `0.185`。它非常保守，却诚实提醒：576 张相关图像不应被包装成强有力的分布无关最坏情况证书。

<MlQuiz
  kind="ERM 推导"
  question="ERM 三步链中，哪一步直接使用了经验风险最小化定义？"
  :options="['R(ĥ)≤R(h*)', 'R̂(ĥ)≤R̂(h*)', 'R̂(h*)≤R(h*)', 'γ≤0']"
  :answer="1"
  explanation="ERM 只承诺 ĥ 的经验风险不高于其他候选；真实风险比较必须借助一致偏差。"
/>

## 7. PAC 可学习：先区分 realizable 与 agnostic

PAC 是 Probably Approximately Correct：给定精度 $\epsilon$ 和失败概率 $\delta$，希望某学习算法在多项式数量样本和计算下，以至少 $1-\delta$ 的概率输出风险足够小的假设。常见的两套目标不同：

| 情形 | 关键假设 | 目标 |
|---|---|---|
| Realizable | 存在 $h^\star\in\mathcal H$ 且 $R(h^\star)=0$ | $R(\widehat h)\le\epsilon$ |
| Agnostic | 不假设类内有零风险规则 | $R(\widehat h)\le\inf_{h\in\mathcal H}R(h)+\epsilon$ |

连接器缺陷含测量噪声、标注分歧和不可观测工艺变量，现实中几乎总是 agnostic；realizable 推导仍重要，因为它最清楚地展示“一致候选存活概率”如何产生 $1/\epsilon$ 样本尺度。

### 7.1 有限类、realizable：完整推导

假设学习器返回任意训练误差为 0 的一致假设。对一个真实风险超过 $\epsilon$ 的坏假设 $h$，单个样本不揭穿它的概率至多 $1-\epsilon$；连续 $n$ 个独立样本都不揭穿它：

$$
\Pr(\widehat R(h)=0)
=(1-R(h))^n
\le(1-\epsilon)^n
\le e^{-n\epsilon}.
$$

只要学习器最终返回坏假设，至少有一个坏假设在训练集上保持一致。对至多 $M$ 个候选并合：

$$
\Pr(R(\widehat h)>\epsilon)
\le M e^{-n\epsilon}.
$$

要求失败概率不超过 $\delta$：

$$
M e^{-n\epsilon}\le\delta
\Longleftrightarrow
\boxed{
n\ge\frac{\ln M+\ln(1/\delta)}{\epsilon}
}.
$$

这条推导依赖零训练误差可实现；不能把它用于有噪声的连接器案例。

### 7.2 有限类、agnostic：用统一偏差得到超额风险

若目标是 $R(\widehat h)-R(h^\star)\le\epsilon$，上一节需要统一偏差 $\gamma=\epsilon/2$：

$$
2M\exp\left(-2n(\epsilon/2)^2\right)\le\delta.
$$

解得

$$
\boxed{
n\ge
\frac{2}{\epsilon^2}
\ln\frac{2M}{\delta}
}.
$$

不同教材可能因单侧/双侧界、目标定义和证明工具得到不同常数，但关键尺度稳定：realizable 有限类约为 $O((\ln M+\ln 1/\delta)/\epsilon)$，agnostic 有限类约为 $O((\ln M+\ln 1/\delta)/\epsilon^2)$。

```python
def realizable_sample_size(hypotheses, epsilon, delta):
    return math.ceil((math.log(hypotheses) + math.log(1 / delta)) / epsilon)


def agnostic_erm_sample_size(hypotheses, epsilon, delta):
    return math.ceil(2 * math.log(2 * hypotheses / delta) / epsilon**2)


print("realizable:", realizable_sample_size(1000, 0.05, 0.05))
print("agnostic ERM excess:", agnostic_erm_sample_size(1000, 0.05, 0.05))
```

预期输出：

```text
realizable: 199
agnostic ERM excess: 8478
```

<MlQuiz
  kind="PAC 情形"
  question="为什么 realizable 与 agnostic 的 ε 依赖分别常见 1/ε 与 1/ε²？"
  :options="['前者用零错误候选被样本揭穿的概率，后者要估计风险均值差', '前者候选更多', '后者不使用概率', '只是记号习惯']"
  :answer="0"
  explanation="realizable 分析追踪坏而一致的候选存活，agnostic 分析需要均值以 O(1/√n) 精度一致收敛。"
/>

::: warning 停下来推一推 2
<!-- ml-stop -->
把 agnostic 证明中统一偏差设成 $\epsilon$ 而不是 $\epsilon/2$，最后只能得到多大的 ERM 超额风险？从三步链逐行指出常数 2 从哪里出现。
:::

## 8. 无限假设类：数参数个数为什么不够

实数参数可以取无限多个值，因此“$M=\infty$”直接代入有限类公式毫无用处。但参数无限并不意味着分类能力无限：一个阈值 $h_t(x)=\mathbb 1[x\ge t]$ 有无穷多个 $t$，在 $n$ 个有序样本上却最多只能产生 $n+1$ 种二分类结果。

反过来，单纯数参数也可能误导。一个参数通过高度振荡的函数编码复杂边界，容量可能远大于“一个参数”的直觉；参数冗余又会让参数数目远大于实际可实现的标记数。学习理论要数的是：**假设类在有限样本上能实现多少种不同标记**。

### 8.1 增长函数

对输入集合 $S_x=\{x_1,\ldots,x_n\}$，记假设类在这些点上诱导的标记集合为

$$
\mathcal H|_{S_x}
=\{(h(x_1),\ldots,h(x_n)):h\in\mathcal H\}.
$$

增长函数取最难点集：

$$
\Pi_{\mathcal H}(n)
=\max_{x_1,\ldots,x_n}|\mathcal H|_{S_x}|,
\qquad
1\le\Pi_{\mathcal H}(n)\le2^n.
$$

三个经典例子：

- 正向射线 $\mathbb 1[x\ge t]$：$\Pi(n)=n+1$；
- 区间 $\mathbb 1[a\le x\le b]$：$\Pi(n)=1+n(n+1)/2$；
- 二维仿射半空间的一般位置点：$\Pi(n)=2\sum_{i=0}^{2}\binom{n-1}{i}$，在 $n=3$ 时为 8，在 $n=4$ 时为 14。

```python
def positive_ray_growth(n):
    return n + 1


def interval_growth(n):
    return 1 + n * (n + 1) // 2


def affine_halfplane_growth(n):
    return min(2**n, 2 * sum(math.comb(n - 1, i) for i in range(3)))


for n in range(1, 7):
    print(
        n,
        positive_ray_growth(n),
        interval_growth(n),
        affine_halfplane_growth(n),
        2**n,
    )
```

预期输出最后三行：

```text
4 5 11 14 16
5 6 16 22 32
6 7 22 32 64
```

<MlQuiz
  kind="增长函数"
  question="增长函数 Π_H(n) 数的是什么？"
  :options="['参数的浮点位数', '最难 n 个点上可实现的不同二分类标记数', '训练样本排列数', '优化器迭代次数']"
  :answer="1"
  explanation="它忽略产生同一标记的参数冗余，直接度量有限样本上的可区分行为。"
/>

## 9. Shattering 与 VC 维：必须同时证明下界和上界

若某一组 $n$ 个点上的全部 $2^n$ 种标记都能被 $\mathcal H$ 实现，就说这组点被 $\mathcal H$ **打散**。VC 维定义为

$$
\operatorname{VC}(\mathcal H)
=\max\{n:\exists\,S_x, |S_x|=n,\ |\mathcal H|_{S_x}|=2^n\}.
$$

量词极其关键：证明 $\operatorname{VC}(\mathcal H)\ge d$，只需找出**一组** $d$ 点能被打散；证明 $\operatorname{VC}(\mathcal H)<d+1$，必须证明**任意** $d+1$ 点都不能被打散。

### 9.1 二维仿射直线分类器为什么 VC 维等于 3

假设类为 $h_{w,b}(x)=\mathbb 1[w^\top x+b\ge0]$。

**下界 $\ge3$。** 取三个非共线点。全同标签可把直线放到三点之外；一个正、两个负可用一条直线隔离单点；两个正、一个负是前一种的补集。因此 8 种标记都可实现。

**上界 $<4$。** 任取四点：

- 若一点在另外三点的凸包内，把内点标正、外部三点标负，线性半空间无法把凸组合与三个顶点分开；
- 若四点构成凸四边形，沿环交替赋正负标签，两个类别的凸包线段相交，不能被一条直线严格分开。

所以任意四点总有至少一种不可实现标记，得到

$$
\boxed{\operatorname{VC}(\text{二维仿射半空间})=3}.
$$

<VcShatteringComparatorLab />

<MlQuiz
  kind="VC 量词"
  question="证明某假设类 VC(H)≥5，需要什么？"
  :options="['任意五点都可打散', '存在一组五点可打散', '存在一种五点标记可实现', '参数至少五个']"
  :answer="1"
  explanation="VC 下界使用存在量词：找出一组五点，使全部 32 种标记都可实现。"
/>

<MlQuiz
  kind="几何反例"
  question="四个凸四边形顶点的哪类标记不能被一条直线实现？"
  :options="['全部为正', '一个正三个负', '相邻两点正、另外两点负', '沿四边形交替正负的 XOR 标记']"
  :answer="3"
  explanation="交替标签的两类凸包相交，不存在仿射直线把两类严格分开。"
/>

## 10. 从 VC 维回到泛化：Sauer 引理与尺度

当 $d=\operatorname{VC}(\mathcal H)<\infty$ 时，Sauer 引理把增长函数从最坏的指数增长压到多项式增长。对 $n\ge d$：

$$
\Pi_{\mathcal H}(n)
\le\sum_{i=0}^{d}\binom ni
\le\left(\frac{en}{d}\right)^d.
$$

一种经典 VC 一致收敛形式是

$$
\Pr\left(
\sup_{h\in\mathcal H}|R(h)-\widehat R(h)|>\epsilon
\right)
\le
4\Pi_{\mathcal H}(2n)e^{-n\epsilon^2/8}.
$$

代入 Sauer 上界并反解，可读出一个代表性尺度：

$$
\epsilon
\lesssim
\sqrt{
\frac{d\ln(2en/d)+\ln(4/\delta)}{n}
}.
$$

这里的 $\lesssim$ 是刻意的：不同的对称化、尾界和损失设定给出不同常数或更紧结果。本章只要求掌握有限类完整推导，以及无限类从“增长函数”到“VC 维控制有效标记数”的关键步骤，不把最松的常数当作工程精度。

```python
def sauer_upper(n, vc_dimension):
    if vc_dimension >= n:
        return 2**n
    return sum(math.comb(n, i) for i in range(vc_dimension + 1))


def representative_vc_gap(n, vc_dimension, delta):
    if vc_dimension > 2 * n:
        return 1.0
    log_term = (
        math.log(4 / delta)
        + vc_dimension * math.log(2 * math.e * n / vc_dimension)
    )
    return math.sqrt(8 * log_term / n)


print("Sauer d=3, n=4:", sauer_upper(4, 3))
print("representative gap:", round(representative_vc_gap(576, 5, 0.05), 3))
```

预期输出：

```text
Sauer d=3, n=4: 15
representative gap: 0.713
```

`15` 是对所有 VC 维不超过 3 的类的通用上界，二维一般位置半空间的精确值是 14；上界不必处处取等。`0.713` 则说明分布无关 VC 界在中小工业样本上常常很松。空泛的理论界不是理论失效，而是它拒绝使用分布形状、间隔、噪声率等额外信息。

<MlQuiz
  kind="证书解读"
  question="某 VC 上界算出风险半宽 0.713，最正确的解释是什么？"
  :options="['模型真实误差恰好增加 0.713', '证明代码一定错了', '该分布无关最坏情况证书对当前样本量很松', '可以忽略测试集']"
  :answer="2"
  explanation="上界是高概率最坏情况保证，不是误差点估计；松界仍能揭示容量、样本量和置信度的尺度。"
/>

## 11. 结构风险最小化：控制的是协议，不只是一项正则

把候选类排成嵌套序列

$$
\mathcal H_1\subset\mathcal H_2\subset\cdots,
$$

并给第 $k$ 个类分配失败预算 $\delta_k$，满足 $\sum_k\delta_k\le\delta$。对每个类建立复杂度惩罚 $\Omega_k(n,\delta_k)$，再选择

$$
(\widehat k,\widehat h)
\in\arg\min_k\min_{h\in\mathcal H_k}
\left[\widehat R(h)+\Omega_k(n,\delta_k)\right].
$$

并合界使所有类的证书以至少 $1-\delta$ 同时成立。一个无限可用的预算方案是 $\delta_k=6\delta/(\pi^2k^2)$。

```python
def confidence_budget(total_delta, classes):
    return np.array([
        6 * total_delta / (math.pi**2 * k**2)
        for k in range(1, classes + 1)
    ])


DELTA_BUDGET = confidence_budget(0.05, 6)
print(np.round(DELTA_BUDGET, 5))
print("allocated <= total:", DELTA_BUDGET.sum() <= 0.05)
```

L1/L2 正则、早停、数据增强、剪枝和优化器隐式偏置都可能降低有效复杂度，但“加了 weight decay”不自动生成一张 VC 证书。尤其对过参数化神经网络，参数数目远超样本数却仍能泛化，经典 VC 维只给出最坏情况容量；间隔、范数、算法稳定性、压缩和数据结构往往给出更贴近实践的解释。

工程上更可执行的 SRM 纪律是：先冻结模型家族、搜索范围、随机种子数、选择指标、停止规则和验证集访问次数，再谈复杂度。没有搜索账本，任何“只试了四个模型”的 $M$ 都不可信。

<MlQuiz
  kind="容量控制"
  question="结构风险最小化最核心的权衡是什么？"
  :options="['训练速度与显存', '经验风险与容量惩罚', '准确率与标签名称', '优化器与文件格式']"
  :answer="1"
  explanation="更大类可能降低经验/近似误差，却增加选择和估计复杂度；SRM 显式平衡两者。"
/>

## 12. 冻结测试：比较差值，而不是比较两张独立成绩单

现在才打开 12 个冻结测试 lot。`appearance` 比旧规则少错 3/144 张，总体错误率从 16.7% 降到 14.6%；但规格 C 从 37.5% 恶化到 41.7%。

```python
LEGACY_PRED = FITS["legacy"]["prediction"]
SELECTED_PRED = FITS[SELECTED_NAME]["prediction"]


def error_rate(prediction, mask):
    return float(np.mean(prediction[mask] != Y[mask]))


print("legacy test:", round(error_rate(LEGACY_PRED, TEST), 3))
print("selected test:", round(error_rate(SELECTED_PRED, TEST), 3))
for group, label in enumerate("ABC"):
    group_mask = TEST & (VARIANT == group)
    print(
        label,
        int(group_mask.sum()),
        round(error_rate(LEGACY_PRED, group_mask), 3),
        round(error_rate(SELECTED_PRED, group_mask), 3),
    )
```

预期输出：

```text
legacy test: 0.167
selected test: 0.146
A 72 0.097 0.056
B 48 0.167 0.146
C 24 0.375 0.417
```

同一张图、同一生产 lot 上的两模型错误高度相关。关心的是损失差

$$
D_g=\frac1{|g|}\sum_{i\in g}
\left[\ell(h_{new}(X_i),Y_i)-\ell(h_{old}(X_i),Y_i)\right],
$$

所以必须在同一个 Bootstrap 样本中一起抽取新旧模型的 lot，保留协方差。若把两模型的 lot 独立重采样，就回答了一个不存在的“两条独立产线”问题。

```python
def paired_lot_bootstrap(
    old_prediction,
    new_prediction,
    mask,
    repeats=5000,
    seed=2818,
):
    rng = np.random.default_rng(seed)
    lot_ids = np.unique(LOT[mask])
    differences = np.empty(repeats)
    for repeat in range(repeats):
        sampled_lots = rng.choice(lot_ids, size=len(lot_ids), replace=True)
        sampled_rows = np.concatenate([
            np.flatnonzero(mask & (LOT == lot_id))
            for lot_id in sampled_lots
        ])
        differences[repeat] = (
            np.mean(new_prediction[sampled_rows] != Y[sampled_rows])
            - np.mean(old_prediction[sampled_rows] != Y[sampled_rows])
        )
    return differences


BOOT_DIFF = paired_lot_bootstrap(LEGACY_PRED, SELECTED_PRED, TEST)
POINT_DIFF = error_rate(SELECTED_PRED, TEST) - error_rate(LEGACY_PRED, TEST)
print("point difference:", round(POINT_DIFF, 4))
print("95% CI:", np.round(np.quantile(BOOT_DIFF, [0.025, 0.5, 0.975]), 4))
print("P(new is better):", round(np.mean(BOOT_DIFF < 0), 3))
```

预期输出：

```text
point difference: -0.0208
95% CI: [-0.0625 -0.0208  0.0208]
P(new is better): 0.805
```

<PairedLotBootstrapRailLab />

95% 区间跨过 0，不能宣称统计上确认的新模型总体改进；即使总体区间完全低于 0，规格 C 的恶化仍可能触发不可补偿的发布门禁。Bootstrap 是有限测试样本上的不确定性分析，不修复验证集反复使用，也不证明未来分布不漂移。

<MlQuiz
  kind="成对比较"
  question="比较同一批测试样本上的新旧模型时，成对 Bootstrap 的关键是什么？"
  :options="['新旧模型分别抽不同 lot', '每次抽样保留同一 lot 上的新旧损失配对', '只重采样预测为正的样本', '把所有图像视作无条件独立']"
  :answer="1"
  explanation="目标是同一生产单元上的损失差；保留配对可保留协方差，并按独立 lot 反映抽样单位。"
/>

::: warning 停下来推一推 3
<!-- ml-stop -->
候选总体少错 3 张，但规格 C 多错 1 张。请分别写出“统计证据不足”和“安全门禁失败”两种拒绝理由；二者不能互相替代，也不能用总体平均把规格 C 的风险抵消。
:::

## 13. 多重比较与验证集复用：测试集也会被训练出来

一次冻结测试只对预先指定的比较有效。如果结果不显著就改模型再打开同一测试集，模型团队会逐渐学习测试集噪声，所谓“冻结”名存实亡。常见控制方法包括：

- 预注册唯一主比较和少量次要切片；
- 为多个确认性比较分配错误预算，如 Bonferroni 或层级检验；
- 把探索性分析明确标为探索性，另收集确认数据；
- 保存模型、数据快照、查询次数和打开人；
- 对时间变化问题使用滚动的未来 lot，而不是永久复用同一静态测试集。

若同时做 $K$ 个确认性比较并给每项 $\delta/K$，族级失败概率至多为 $\delta$。这很保守，但比“每个都写 95%”更诚实。

<MlQuiz
  kind="测试治理"
  question="团队根据冻结测试结果连续修改模型并再次测试，主要破坏了什么？"
  :options="['矩阵乘法速度', '测试集与选择过程的独立性', '标签的数据类型', '模型文件大小']"
  :answer="1"
  explanation="反复查看会把测试集信息反馈进模型选择，使原本单次比较的置信解释失效。"
/>

## 14. 校准是另一项资格，不由 0–1 错误率代办

硬阈值决策只看分数是否跨过阈值。把同一分数通过不同斜率的 sigmoid，可保持全部硬预测不变，却显著改变概率的 Brier 分数和负对数似然：

$$
\operatorname{Brier}=\frac1n\sum_i(p_i-y_i)^2,
\qquad
\operatorname{NLL}=-\frac1n\sum_i[y_i\ln p_i+(1-y_i)\ln(1-p_i)].
$$

```python
SELECTED_SCORE = SCORES[SELECTED_NAME]
SELECTED_THRESHOLD = FITS[SELECTED_NAME]["threshold"]


def probability_metrics(probability, y, mask):
    p = np.clip(probability[mask], 1e-12, 1 - 1e-12)
    target = y[mask]
    brier = np.mean((p - target) ** 2)
    nll = -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))
    return float(brier), float(nll)


for slope in (1.2, 2.5, 5.0, 10.0):
    probability = sigmoid(slope * (SELECTED_SCORE - SELECTED_THRESHOLD))
    brier, nll = probability_metrics(probability, Y, TEST)
    hard_error = np.mean((probability[TEST] >= 0.5) != Y[TEST])
    print(slope, round(hard_error, 4), round(brier, 4), round(nll, 4))
```

预期输出：

```text
1.2 0.1458 0.1145 0.3787
2.5 0.1458 0.1116 0.3733
5.0 0.1458 0.1245 0.5579
10.0 0.1458 0.1358 1.0358
```

四个输出的二分类错误率完全相同，过陡的概率却对错误样本过度自信。若概率用于人工复核优先级或风险累积，必须在独立校准集上拟合温度/等距映射，并用可靠性图、Brier、NLL 和决策成本独立验收。

<MlQuiz
  kind="校准边界"
  question="两个模型 0–1 错误率相同，是否说明其概率质量相同？"
  :options="['是', '否，概率锐度和校准仍可完全不同', '只要阈值为 0.5 就相同', '样本超过 100 就相同']"
  :answer="1"
  explanation="硬决策丢弃了概率幅度信息；Brier、NLL 与可靠性分析可能给出截然不同的结论。"
/>

## 15. 分布偏移与群体误差：PAC 保证的适用域

经典 i.i.d. PAC 假设训练与未来样本来自同一 $\mathcal D$。工业系统常见三类破坏：

1. **协变量偏移**：新相机或照明改变 $P(X)$；
2. **概念偏移**：模具磨损使 $P(Y\mid X)$ 改变；
3. **群体组成偏移**：规格 C 的占比从 1/6 升到 1/2。

总体风险可写成群体风险加权和：

$$
R(h)=\sum_g\pi_g R_g(h).
$$

只要部署权重 $\pi_g$ 改变，总体误差就会改变；更严重的是，总体平均可能掩盖稀少但安全关键群体的高错误率。规格 C 测试只有 24 张，37.5% 与 41.7% 的点估计都很不稳定，却已经足够触发“继续收集 C、不得自动放行”的工程动作，而不是声称两个百分比有精密差异。

```python
def group_error_table(prediction, mask):
    table = []
    for group, label in enumerate("ABC"):
        group_mask = mask & (VARIANT == group)
        table.append((label, int(group_mask.sum()), error_rate(prediction, group_mask)))
    return table


GROUP_TABLE = group_error_table(SELECTED_PRED, TEST)
DEPLOYMENT_WEIGHTS = np.array([0.25, 0.25, 0.50])
REWEIGHTED_RISK = sum(
    weight * risk
    for weight, (_, _, risk) in zip(DEPLOYMENT_WEIGHTS, GROUP_TABLE)
)
print("observed test:", round(error_rate(SELECTED_PRED, TEST), 3))
print("if C becomes 50%:", round(REWEIGHTED_RISK, 3))
```

预期输出：

```text
observed test: 0.146
if C becomes 50%: 0.259
```

因此资格证书必须带适用域：相机版本、材料批次、规格组合、速度范围、时间窗口和数据质量门禁。漂移监控发现超域，只能触发降级、复检或重新确认；不能继续引用旧 PAC 数字。

<MlQuiz
  kind="分布偏移"
  question="训练与测试同分布时得到的 PAC 保证，能否直接覆盖新相机造成的输入分布变化？"
  :options="['能，概率界永久有效', '不能，除非另有偏移假设或新域验证', '只要模型参数不变就能', '只要训练误差低就能']"
  :answer="1"
  explanation="经典保证依赖训练与未来样本来自同一分布；超出适用域需要新的假设、数据和验证。"
/>

## 16. 理论证书、统计证据与运行门禁是三层不同控制

| 层次 | 回答的问题 | 本案例结论 |
|---|---|---|
| 学习理论 | 容量与样本量能否控制选择后的泛化差距？ | 能给尺度，但分布无关界很松，且图像非完全独立 |
| 冻结统计比较 | 当前冻结 lot 是否支持“新模型更好”？ | 点估计 -2.08pp，成对 95% 区间跨 0 |
| 工业运行门禁 | 即使平均更好，是否满足所有不可补偿条件？ | 规格 C 恶化，不满足自动放行条件 |

三层不能互相代办：PAC 上界不能替代冻结测试；显著性不能替代群体安全门；上线监控也不能为一次设计错误的验证补票。

### 16.1 一份可审计的模型资格包

至少保存：

- 数据卡：lot 定义、抽样框、排除规则、标注协议、缺失机制；
- 候选账本：全部特征、阈值、超参数、seed、失败实验和访问时间；
- 理论声明：损失范围、独立单位、$\mathcal H$、$M$ 或容量定义、$\epsilon/\delta$；
- 冻结评估：唯一主比较、成对差值、区间方法、群体切片、校准；
- 发布门禁：总体上限、关键群体上限、拒绝/降级逻辑；
- 运行监控：输入漂移、分群覆盖、延迟标签回放、证书失效条件。

<MlQuiz
  kind="发布门禁"
  question="下面哪一项最不应该被总体平均改进抵消？"
  :options="['日志文件大小', '安全关键规格的硬错误上限', '图表颜色', '训练耗时的轻微波动']"
  :answer="1"
  explanation="关键群体安全门应是不可补偿约束；总体平均更好不能授权局部高风险。"
/>

## 数值验证：可执行回归测试与边界实验

### 17.1 本章关键不变量

```python
# 数据和切分合同。
assert CASE.shape == (864, 7)
assert TRAIN.sum() == 576 and VALID.sum() == 144 and TEST.sum() == 144
assert set(LOT[TRAIN]).isdisjoint(set(LOT[VALID]))
assert set(LOT[TRAIN]).isdisjoint(set(LOT[TEST]))
assert set(LOT[VALID]).isdisjoint(set(LOT[TEST]))

# 冻结选择和误差合同。
assert SELECTED_NAME == "appearance"
assert np.isclose(error_rate(LEGACY_PRED, TEST), 24 / 144)
assert np.isclose(error_rate(SELECTED_PRED, TEST), 21 / 144)
assert np.isclose(POINT_DIFF, -3 / 144)

# 容量和置信度的单调性。
assert hoeffding_gap(400, 0.05) < hoeffding_gap(100, 0.05)
assert finite_class_gap(1000, 5000, 0.10) > hoeffding_gap(1000, 0.10)
assert positive_ray_growth(5) == 6
assert interval_growth(4) == 11
assert affine_halfplane_growth(3) == 8
assert affine_halfplane_growth(4) == 14

# 成对 Bootstrap 必须可重放，且当前 95% 区间跨过 0。
repeat_boot = paired_lot_bootstrap(
    LEGACY_PRED,
    SELECTED_PRED,
    TEST,
    repeats=5000,
    seed=2818,
)
assert np.array_equal(BOOT_DIFF, repeat_boot)
lower, upper = np.quantile(BOOT_DIFF, [0.025, 0.975])
assert lower < 0 < upper

print("切分、选择、容量、重放与成对区间门禁：全部通过")
```

### 17.2 反例：把相关图像当独立样本会怎样

把样本量从 12 个独立 lot 偷换成 144 张图像，固定假设 Hoeffding 半宽从约 0.392 缩到 0.113，看起来精度提高了三倍多；这不是模型变好了，只是独立单位写错了。

```python
print("12 independent lots:", round(hoeffding_gap(12, 0.05), 3))
print("144 correlated images claimed iid:", round(hoeffding_gap(144, 0.05), 3))
```

### 17.3 可选成熟库交叉核对

仓库不为本章新增依赖。若环境已有 SciPy，可用 `scipy.stats.bootstrap` 对“每 lot 的成对损失差均值”做交叉核对；调用时传入差值向量或保持 `paired=True`，不能把新旧损失拆成两组独立样本。理论界不需要库函数才能成立，资格协议也不能由库默认值替写。

```python
# 不执行：可选环境对照，需要 SciPy。
from scipy.stats import bootstrap

test_lots = np.unique(LOT[TEST])
lot_differences = np.array([
    np.mean(SELECTED_PRED[TEST & (LOT == lot_id)] != Y[TEST & (LOT == lot_id)])
    - np.mean(LEGACY_PRED[TEST & (LOT == lot_id)] != Y[TEST & (LOT == lot_id)])
    for lot_id in test_lots
])
result = bootstrap(
    (lot_differences,),
    np.mean,
    confidence_level=0.95,
    n_resamples=5000,
    random_state=2818,
)
print(result.confidence_interval)
```

## 复杂度与工程边界

| 操作 | 时间复杂度 | 主要内存 | 工业边界 |
|---|---:|---:|---|
| 评估有限类 $M$ 个候选 | $O(nM)$ | 可流式到 $O(n)$ | 必须记录真实尝试数，不只记录最终入围者 |
| 网格阈值 ERM | $O(nT)$，排序后可降 | $O(n+T)$ | 同一验证集反复改网格会增加自适应选择成本 |
| 多重尝试模拟 | $O(BnM)$ | 分块为 $O(cM)$ | 只用于机制演示，不是证明 |
| 计算增长函数 | 最坏需枚举 $2^n$ 标记 | 依算法而定 | 小规模可视化；大类依赖组合论上界 |
| lot Bootstrap | $O(BGm)$ | $O(B)$ | $G$ 为 lot 数、$m$ 为每 lot 样本；必须保持配对 |
| 群体评估 | $O(n)$ | $O(G)$ | 小群体区间宽，不能因不显著就判定无风险 |

理论边界还包括：

- 0–1 损失有界，但 NLL 若概率可到 0/1 就无界，需裁剪或改用相应尾界；
- Hoeffding 是分布无关界，没有利用方差，Bernstein/经验 Bernstein 可能更紧；
- VC 维刻画最坏情况分类容量，不等于某训练算法在某数据分布上的实际有效容量；
- Bootstrap 依赖观测单位能代表抽样机制，只有 12 个 lot 时尾部分位数很粗；
- 任何证书都只覆盖声明的适用域、数据版本、候选协议和发布规则。

## 18. 章节练习

### 基础推导

1. 从双侧 Hoeffding 界出发，推导 $M=2000,\delta=0.01$ 时统一偏差不超过 0.04 所需的 $n$。
2. 不看正文，重新写出 ERM 三步链，并在每个不等号旁标明依据。
3. 完整推导 finite-realizable 的 $M e^{-n\epsilon}$ 界，指出哪里使用了训练误差为 0。
4. 将 agnostic 目标从超额风险 $\epsilon$ 改为统一偏差 $\epsilon$，比较两个样本复杂度常数。
5. 证明正向射线增长函数为 $n+1$，区间增长函数为 $1+n(n+1)/2$。
6. 为一维区间类证明 VC 维为 2：给出能打散的两点，并证明任意三点不能全部打散。

### 实现与反例

7. 扩展 `simulate_winner_gap`，扫描 $M=1,10,10^3,10^5$，画冠军差距中位数与 $\sqrt{\ln M/n}$ 的关系。
8. 构造 20 个真实风险不同的候选，比较经验冠军、带统一惩罚的冠军和冻结验证冠军。
9. 枚举平面四点的 16 种标记，用线性规划或穷举角度核对凸四边形只有 14 种可实现。
10. 将 Bootstrap 的单位故意改成图像，比较区间宽度；解释为什么更窄的区间反而错误。
11. 给同一硬预测构造三组概率，使错误率相同但 Brier 和 NLL 排名不同。
12. 把规格 C 的部署权重从 0.1 扫到 0.8，画新旧模型重加权风险交叉点。

### 工业资格审查

13. 写一份候选搜索预注册表，必须包含模型家族、阈值数、seed、停止条件、主指标、次要切片和验证集访问预算。
14. 为 5 个确认性比较分配总失败预算 0.05，比较等额 Bonferroni 与层级方案。
15. 设计规格 C 的补充抽样方案，明确 lot 数而非图像数、时间跨度、相机版本和可接受误差上限。
16. 为相机更换设计证书失效与再确认流程，区分只需漂移筛查、必须重新校准和必须完整重训三种条件。
17. 审查一个“神经网络参数 100 万，所以 $M=100$ 万”的报告，指出参数计数、有限假设数和 VC 容量被混淆的位置。

练习骨架：

```python
# TODO：返回按 lot 成对重采样的新旧风险差区间，禁止拆开索引。
def paired_group_interval(old_loss, new_loss, group_id, repeats, seed):
    raise NotImplementedError
```

## 来源与许可

- Stanford CS229 课程讲义，PDF 第 127–132 页，§8.3–§8.3.2：用于核对并合界、Hoeffding、经验/真实风险、有限假设类一致收敛、样本复杂度与 ERM 超额风险链。
- Stanford CS229 课程讲义，PDF 第 132–136 页，§8.3.3：用于核对无限假设类、shattering、VC 维、二维半空间 VC 维和 VC 泛化尺度。
- Stanford CS229 课程讲义，PDF 第 136–143 页，第 9 章 §9.1–§9.3：用于核对显式正则、范数、优化器隐式偏置、holdout、交叉验证和模型选择边界。
- CMU 10-601/10-701，Lecture 15 *PAC Learning*，PDF 第 15–41 页：用于交叉核对 PAC 判据、有限/无限与 realizable/agnostic 四种情形、有限类样本复杂度和模型复杂度权衡。
- CMU 10-601/10-701，Lecture 16 *PAC, MAP*，PDF 第 3–39 页：用于交叉核对 VC 维量词、射线与区间示例、结构风险最小化、正则化和过参数化深度学习边界。
- Jonathan Richard Shewchuk，UC Berkeley CS 189/289A 合订讲义，PDF 第 163–168 页，Bonus Lecture A：用于交叉核对风险、ERM、Hoeffding、统一界、增长函数、二维半空间标记数与 VC 维。

本章高压连接器压接截面视觉终检资格审查案例的 72 个教学 lot、864 条合成记录、四个分数家族、冻结 lot 切分、搜索账本、规格 A/B/C、校准反例、全部 NumPy 代码、17 道即时测验和三项专用 SVG 网页实验均为本站独立创作。它们不代表真实企业、连接器结构、缺陷机理、检测性能、放行阈值、安全结论或监管要求。没有复制、改编或再分发参考资料中的图形、数据、代码、案例和课堂测验。

完整来源登记见 [课程来源与许可](./sources.md) 。

<MlLessonProgress
  storage-key="28-learning-theory"
  lesson-id="28"
  next-href="/courses/machine-learning/29-recommender-systems"
  next-label="下一章：矩阵分解与 BPR 推荐"
/>
