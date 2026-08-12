---
course: machine-learning
edition: industrial
title: 监督学习与经验风险最小化
description: 以循环水泵故障预警为贯穿案例，从实际行动和错误代价建立监督学习问题，推导总体风险、经验风险与经验风险最小化，并用数值实验检查有限样本选择误差。
outline: [2, 3]
---

<MlLessonHero number="01" title="监督学习与经验风险最小化" question="模型应该优化一个容易计算的分数，还是系统真正承担的错误代价？" duration="建议 300–420 分钟" :outcomes="['把工业任务写成包含数据、假设、损失和评价协议的学习合同', '从条件风险推出代价敏感行动阈值', '区分总体风险、经验风险与经验风险最小化', '用 NumPy 实现并验证有限候选集上的 ERM']" />

本章不从算法清单开始。我们先接手一个可能停线的设备预警任务，明确系统要做什么决定、错误怎样计价、训练数据从哪里来，再让经验风险最小化自然出现。

你已经在数据科学课中使用过训练集、测试集和 `fit`。这里要进一步回答：`fit` 到底最小化了什么，这个量与未来生产风险是什么关系，以及为什么更低的训练损失仍可能选出更差的系统。

## 贯穿案例：循环水泵 24 小时故障预警

一家晶圆厂有多台循环水泵。非计划停机可能让整条产线等待，因此设备团队希望每小时根据传感器窗口，判断是否在下一班次安排检修。

本章使用的是**教学合成案例**，不是任何工厂的真实记录。案例的作用是把学习问题写完整，而不是给出可直接上线的维护策略。

### 业务动作不是“输出一个概率”

模型可以输出未来 24 小时发生故障的概率，但值班工程师最终只有两个动作：

| 动作 | 含义 | 主要代价 |
|---|---|---|
| $a=0$ | 继续运行 | 若设备随后故障，会产生停线与抢修代价 |
| $a=1$ | 触发计划检修 | 若设备本来正常，会浪费检修窗口和备件 |

因此预测与决策要分成两层：

$$
x \xrightarrow{f} p(x)=P(Y=1\mid X=x)
\xrightarrow{\delta} a\in\{0,1\}.
$$

$f$ 估计风险，$\delta$ 根据代价采取行动。模型概率的 AUROC 很高，并不能自动说明检修决策省钱；阈值、校准和错误代价都在第二层起作用。

### 数据合同

每个样本对应某台泵在时刻 $t$ 的一个特征窗口：

| 字段 | 数学对象 | 案例定义 |
|---|---|---|
| 输入 $X$ | $\mathcal X\subseteq\mathbb R^5$ | 振动 RMS、轴承温度、电流波动、距上次检修小时数、当前负载 |
| 标签 $Y$ | $\mathcal Y=\{0,1\}$ | 时刻 $t$ 之后 24 小时内是否发生目标故障 |
| 动作 $A$ | $\mathcal A=\{0,1\}$ | 是否触发计划检修 |
| 观测单位 | $(\text{pump_id},t)$ | 一台设备的一个决策时点 |
| 特征截止 | $t$ | 任何 $t$ 之后才产生的信息都不可进入 $X$ |
| 评价切分 | 按时间 | 较早窗口训练，较晚窗口验证和测试，同一故障事件不能跨集合泄漏 |

下面的数组只用于说明边界处的数据形状：

```python
import numpy as np

# 列顺序：振动、温度、电流波动、距检修小时、负载
X = np.array([
    [2.1, 61.0, 0.08, 120.0, 0.72],
    [4.8, 79.0, 0.21, 940.0, 0.91],
    [3.2, 68.0, 0.13, 510.0, 0.84],
    [5.4, 82.0, 0.28, 1080.0, 0.95],
], dtype=float)
y = np.array([0, 1, 0, 1], dtype=int)

assert X.ndim == 2
assert y.shape == (X.shape[0],)
assert np.isin(y, [0, 1]).all()
```

<MlQuiz kind="设计判断" question="若特征包含‘未来 24 小时内是否生成抢修工单’，最大的问题是什么？" :options="['特征维度太少', '它使用了决策时点之后的信息，构成标签泄漏', '它不是连续变量', '它会让训练速度变慢']" :answer="1" explanation="抢修工单通常在故障发生后生成。把它作为时刻 t 的输入，会让离线模型提前看到未来，测试分数失去上线意义。" />

<div class="ml-stop"><strong>停下来推一推 1</strong>把“故障”改成“轴承异常”“保护停机”或“任意维修工单”，会怎样改变标签分布、错误代价和可采取动作？先写出业务定义，再讨论模型。</div>

## 1. 一个监督学习问题需要哪些对象

把案例抽象后，一个可审查的监督学习问题至少包含：

$$
(\mathcal X,\mathcal Y,\mathcal A,\mathcal H,L,P_{X,Y},D,\Pi).
$$

- $\mathcal X$：合法输入的集合。
- $\mathcal Y$：真实结果的集合。
- $\mathcal A$：系统能采取的动作集合。
- $\mathcal H$：学习算法允许选择的预测或决策函数集合。
- $L(y,a)$：真实结果为 $y$、采取动作 $a$ 时的损失。
- $P_{X,Y}$：未来输入与结果的未知联合分布。
- $D=\{(x_i,y_i)\}_{i=1}^n$：从历史过程得到的有限样本。
- $\Pi$：训练、选择和最终评价协议，例如时间切分、评价窗口和冻结规则。

教材有时只写五元组或六元组。本章显式加入动作空间和评价协议，是因为工业系统常在这两个位置失败：模型预测的是一个标签，系统执行的却是另一个动作；或者测试集被反复用于选择，已经不再是独立证据。

### 假设空间与学习算法不是同一对象

假设空间 $\mathcal H$ 是候选函数集合，学习算法 $A$ 是从数据到候选函数的选择规则：

$$
A:D\mapsto \widehat h\in\mathcal H.
$$

若 $\mathcal H$ 是所有阈值规则，候选函数可以写成

$$
h_t(x)=\mathbf 1[p(x)\ge t],\qquad t\in[0,1].
$$

学习算法可能枚举阈值并选择经验损失最低者，也可能使用梯度法训练一个参数模型。第 02 章会研究“怎样找到目标函数的低点”；本章先把“哪个目标值得优化”写清楚。

<MlQuiz kind="概念辨析" question="下列哪一项描述的是学习算法，而不是假设空间？" :options="['所有深度不超过 5 的决策树', '所有阈值 t∈[0,1] 对应的决策规则', '从候选阈值中选择经验代价最低者', '所有宽度为 64 的两层神经网络']" :answer="2" explanation="候选函数的集合是 H；读取数据并从 H 中选出一个函数的过程才是学习算法。" />

## 2. 损失必须对应真实行动

最常见的二分类 0–1 损失是

$$
L_{01}(y,a)=\mathbf 1[y\ne a].
$$

它默认漏检一次与误检一次同价。设备维护通常不满足这个假设。令漏检代价为 $C_{\mathrm{FN}}$，误检代价为 $C_{\mathrm{FP}}$，则

$$
L(y,a)=
\begin{cases}
0,&y=a,\\
C_{\mathrm{FN}},&y=1,a=0,\\
C_{\mathrm{FP}},&y=0,a=1.
\end{cases}
$$

损失的单位也必须明确。它可以是货币、停机分钟、报废晶圆数，也可以是经过组织批准的综合代价分。若把不同单位随意相加，公式虽然能算，决策却无法解释。

### 从条件风险推出行动阈值

设模型给出的故障概率为

$$
p(x)=P(Y=1\mid X=x).
$$

若继续运行，只有未来故障时付出代价：

$$
R(a=0\mid x)=C_{\mathrm{FN}}p(x).
$$

若触发检修，只有设备本来正常时付出代价：

$$
R(a=1\mid x)=C_{\mathrm{FP}}[1-p(x)].
$$

选择检修当且仅当第二项更小：

$$
\begin{aligned}
C_{\mathrm{FP}}(1-p)&<C_{\mathrm{FN}}p\\
C_{\mathrm{FP}}&<(C_{\mathrm{FN}}+C_{\mathrm{FP}})p\\
p&>\frac{C_{\mathrm{FP}}}{C_{\mathrm{FN}}+C_{\mathrm{FP}}}.
\end{aligned}
$$

当漏检代价是误检的 8 倍时，理论阈值为 $1/9\approx0.111$，不是 0.5。这个结论依赖两个关键条件：模型概率已经校准；两类错误代价足以概括当前行动后果。

<RiskDecisionLab />

<MlQuiz kind="结果预测" question="保持误检代价不变，把漏检代价从 4 提高到 12，理论检修阈值如何变化？" :options="['升高，模型应更谨慎', '降低，系统应更早触发检修', '保持 0.5', '只由样本量决定']" :answer="1" explanation="阈值为 C_FP/(C_FN+C_FP)。漏检代价增大使分母增大，因此阈值降低。" />

### 预测损失与决策损失不要混为一谈

训练概率模型时可能使用对数损失：

$$
L_{\log}(y,p)=-y\log p-(1-y)\log(1-p).
$$

它鼓励概率估计，而检修代价决定最终阈值。一个系统可以用对数损失训练概率模型，再用代价矩阵选择动作；也可以直接优化代价敏感代理损失。无论采用哪条路线，都要分别报告：

1. 概率质量，例如校准误差或对数损失；
2. 固定行动规则后的业务代价；
3. 阈值和代价假设变化时的敏感性。

## 3. 总体风险：真正想优化但无法直接计算

给定决策函数 $h:\mathcal X\to\mathcal A$，总体风险定义为

$$
R(h)=\mathbb E_{(X,Y)\sim P_{X,Y}}[L(Y,h(X))].
$$

它回答：如果未来不断从同一生产过程遇到新设备窗口，平均每次决策付出多少损失。

问题是 $P_{X,Y}$ 未知。如果能够精确积分总体风险，也就不需要从历史样本学习。我们只能计算经验风险：

$$
\widehat R_n(h;D)=\frac1n\sum_{i=1}^{n}L(y_i,h(x_i)).
$$

这里要区分抽样前后：

- 抽样前，$D$ 是随机变量，因此 $\widehat R_n(h;D)$ 也是随机变量。
- 给定一份固定数据和固定 $h$ 后，这个有限和是确定数。

对一个预先固定、没有看过数据的 $h$，若样本独立同分布且损失期望存在，大数定律支持

$$
\widehat R_n(h)\xrightarrow[]{P}R(h).
$$

但训练得到的 $\widehat h$ 看过数据。训练会主动寻找经验风险偶然偏低的候选，因此“每个固定模型都能估准”还不足以保证“被选中的模型也能估准”。第 28 章会用一致收敛、PAC 和 VC 维给出正式条件。

<MlQuiz kind="错误诊断" question="工程师在同一测试集上比较 500 个模型并报告最小误差。为什么这个误差偏乐观？" :options="['模型数量多会让 CPU 变慢', '最终模型已经依赖测试集，最小值会吸收多次尝试中的偶然低估', '测试集不能计算损失', '只有神经网络存在这个问题']" :answer="1" explanation="选择过程使用了测试数据，测试集不再是独立的最终证据。需要独立验证集做选择，并把测试集冻结到最后一次评价。" />

<div class="ml-stop"><strong>停下来推一推 2</strong>“这 500 个模型都没有直接用测试样本更新参数”能否排除测试泄漏？说明超参数、特征方案和阈值为何也属于模型选择。</div>

## 4. 经验风险最小化

经验风险最小化（empirical risk minimization, ERM）在假设空间中选择经验风险最低的候选：

$$
\widehat h\in\arg\min_{h\in\mathcal H}\widehat R_n(h).
$$

`min` 返回最小风险数值，`argmin` 返回达到最小值的自变量集合。多个候选并列时，$\arg\min$ 可能包含多个函数，因此严格写法使用 $\in$；实际系统还需要一个确定的并列规则。

### ERM 是一个原则，不是一种具体模型

下面这些训练过程都可以放进 ERM 框架：

- 线性回归在所有线性函数中最小化平均平方损失；
- 逻辑回归最小化平均负对数似然；
- SVM 最小化 hinge loss 与正则项；
- 神经网络用随机优化近似最小化批量经验损失。

差异来自 $\mathcal H$、$L$、复杂度控制和求解方法，而不是是否叫“ERM”。

<MlQuiz kind="推导补全" question="表达式 argminₕ R̂(h) 的结果是什么？" :options="['一个风险数值', '达到最低经验风险的候选函数集合', '损失函数本身', '总体分布 P']" :answer="1" explanation="min 返回最小目标值，argmin 返回达到该值的自变量；存在并列时结果是集合。" />

### 受限与正则化 ERM

直接扩大 $\mathcal H$ 能降低训练风险，却可能增加有限样本下的选择误差。常见控制方式有两种：

$$
\widehat h\in\arg\min_{h\in\mathcal H_k}\widehat R_n(h)
$$

使用受限假设空间，或

$$
\widehat h_\lambda\in\arg\min_{h\in\mathcal H}
\left[\widehat R_n(h)+\lambda\Omega(h)\right]
$$

给复杂度定价。第 03 章研究如何用验证证据选择复杂度，第 06 章推导 Ridge 与 Lasso。

<ErmSamplingLab />

<MlQuiz kind="结果预测" question="扩大候选集合 H 后，最低训练风险会怎样？" :options="['只可能降低或保持不变', '一定升高', '一定等于总体风险', '与候选集合无关']" :answer="0" explanation="旧候选仍在更大的集合中，因此最小经验风险不会升高；但总体风险可能因为选择误差增大而变差。" />

## 5. 归纳偏置决定怎样从有限证据外推

有限样本不能唯一决定训练点之外的函数。即使所有训练点都拟合正确，点与点之间、观测范围之外仍有无数种延伸方式。学习器必须偏好某些解释，这种偏好称为归纳偏置。

偏置可以进入多个位置：

| 位置 | 例子 | 实际含义 |
|---|---|---|
| 假设空间 | 只允许线性边界 | 偏好全局平滑、加性方向 |
| 正则项 | 惩罚大系数 | 偏好参数范数较小的解 |
| 架构 | CNN 的局部连接和参数共享 | 偏好局部模式可跨位置复用 |
| 优化 | 特定初始化与梯度路径 | 在多个低损失解之间产生隐式偏好 |
| 数据 | 时间窗口、采样和特征定义 | 规定模型能看见什么证据 |

没有“完全不带偏置”的有限样本学习。工程问题不是消除偏置，而是让偏置与设备机理、数据生成过程和决策目标相符，并通过失败案例检验它。

### 近似误差与估计误差

令 $h^{**}$ 是所有可能函数中的总体风险最优者，$h^*_{\mathcal H}$ 是假设空间内的总体风险最优者，$\widehat h$ 是 ERM 选择。则超额风险可以拆为

$$
R(\widehat h)-R(h^{**})
=\underbrace{R(h^*_{\mathcal H})-R(h^{**})}_{\text{近似误差}}
+\underbrace{R(\widehat h)-R(h^*_{\mathcal H})}_{\text{估计误差}}.
$$

- 更丰富的 $\mathcal H$ 通常能减小近似误差。
- 在样本量固定时，更丰富的 $\mathcal H$ 往往更难可靠选择，估计误差可能增大。
- 求解器没有真正达到最低经验风险时，还存在优化间隙
  $\widehat R_n(\widetilde h)-\widehat R_n(\widehat h)$；它不能与总体风险差异混为一谈。

<MlQuiz kind="概念辨析" question="训练损失高、验证损失也高，最直接支持哪种诊断？" :options="['一定是测试泄漏', '当前表示、假设空间或优化不足，尚未把训练数据解释好', '一定需要更多正则化', '说明总体风险已经最低']" :answer="1" explanation="训练集本身都没有拟合好，应先区分表示能力不足和优化未收敛；增加正则化通常不会直接解决这一现象。" />

<div class="ml-stop"><strong>停下来推一推 3</strong>一个巨大网络把训练风险降到 0，并不能单独告诉你近似误差、估计误差和优化误差各是多少。分别说明还需要什么证据。</div>

## 6. 从公式到可审查实现

先实现代价敏感经验风险。边界检查属于算法合同的一部分：若概率越界、标签不是二元值、数组形状不一致或成本为负，应在进入内部逻辑前失败。

```python
def cost_sensitive_risk(
    y_true,
    probability,
    threshold,
    false_negative_cost,
    false_positive_cost,
):
    y_true = np.asarray(y_true)
    probability = np.asarray(probability, dtype=float)

    if y_true.ndim != 1 or probability.ndim != 1:
        raise ValueError("y_true and probability must be one-dimensional")
    if y_true.shape != probability.shape or y_true.size == 0:
        raise ValueError("arrays must be non-empty and have the same shape")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
        raise ValueError("probability must be finite and lie in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    if false_negative_cost < 0 or false_positive_cost < 0:
        raise ValueError("costs must be non-negative")

    action = (probability >= threshold).astype(int)
    false_negative = (y_true == 1) & (action == 0)
    false_positive = (y_true == 0) & (action == 1)
    loss = (
        false_negative_cost * false_negative
        + false_positive_cost * false_positive
    ).astype(float)
    return loss, loss.mean()
```

有限候选集上的 ERM 只需枚举并明确并列规则：

```python
def select_threshold(
    y_true,
    probability,
    thresholds,
    false_negative_cost,
    false_positive_cost,
):
    candidates = []
    for threshold in np.asarray(thresholds, dtype=float):
        _, empirical_risk = cost_sensitive_risk(
            y_true,
            probability,
            threshold,
            false_negative_cost,
            false_positive_cost,
        )
        candidates.append((float(empirical_risk), float(threshold)))

    if not candidates:
        raise ValueError("thresholds must not be empty")

    # 风险并列时选择更低阈值：在本案例中偏向减少漏检。
    return min(candidates, key=lambda item: (item[0], item[1]))
```

<MlQuiz kind="代码预测" question="`min(candidates, key=lambda item: (item[0], item[1]))` 在风险相同时怎样选择？" :options="['随机选择', '选择阈值更低的候选', '选择阈值更高的候选', '抛出异常']" :answer="1" explanation="排序键先比较风险 item[0]，相同后比较阈值 item[1]，因此选择更低阈值。并列规则已经成为可测试的系统行为。" />

## 数值验证

代码能运行不代表公式正确。下面用三个层次验证实现。

### 验证一：手算样本与程序一致

两个漏检、一个误检，若 $C_{\mathrm{FN}}=8$、$C_{\mathrm{FP}}=2$，总损失应为 $18$，平均损失为 $4.5$：

```python
loss, risk = cost_sensitive_risk(
    y_true=[1, 0, 1, 0],
    probability=[0.2, 0.8, 0.4, 0.1],
    threshold=0.5,
    false_negative_cost=8,
    false_positive_cost=2,
)

assert np.array_equal(loss, [8.0, 2.0, 8.0, 0.0])
assert risk == 4.5
```

### 验证二：最优阈值接近条件风险推导

若模型概率就是数据生成概率，大样本上使代价最低的阈值应接近
$C_{\mathrm{FP}}/(C_{\mathrm{FN}}+C_{\mathrm{FP}})$：

```python
rng = np.random.default_rng(20260805)
probability = rng.beta(2.0, 5.0, size=200_000)
y_simulated = rng.binomial(1, probability)

false_negative_cost = 8
false_positive_cost = 1
theoretical_threshold = false_positive_cost / (
    false_negative_cost + false_positive_cost
)
grid = np.linspace(0.01, 0.50, 100)
best_risk, best_threshold = select_threshold(
    y_simulated,
    probability,
    grid,
    false_negative_cost,
    false_positive_cost,
)

assert abs(best_threshold - theoretical_threshold) < 0.02
```

### 验证三：无效输入必须被拒绝

```python
try:
    cost_sensitive_risk([0, 2], [0.2, 0.8], 0.5, 8, 1)
except ValueError as error:
    assert "only 0 and 1" in str(error)
else:
    raise AssertionError("invalid labels were accepted")
```

这些断言分别检查具体数值、理论性质和系统边界。三者互补：单个手算例子可能漏掉普遍错误，大样本模拟可能漏掉输入校验，而只有异常测试无法证明数学更新正确。

## 7. 贯穿案例的评价协议

水泵预警不能随机拆分相邻时间窗口后就声称“测试独立”。同一台设备的重叠窗口高度相关，同一次故障前的多个窗口共享事件信息。一个最低限度的协议是：

1. 以故障事件为分组单位，禁止同一事件的窗口跨集合。
2. 按时间排序，早期数据训练，中期数据选择模型和阈值，最后时间段只评价一次。
3. 在验证前冻结标签窗口、缺失值处理、特征截止时间和代价矩阵。
4. 同时报告概率质量、业务代价、漏检/误检数量和设备/工况切片。
5. 对阈值、代价比和时间漂移做敏感性分析。

### 基线必须与行动目标一致

至少比较三个无需复杂模型的基线：

- 从不检修：误检为 0，但承担全部故障漏检代价；
- 全部检修：漏检为 0，但承担全部正常设备误检代价；
- 固定机理阈值：例如振动超过经批准的工程阈值时检修。

如果机器学习模型不能稳定优于这些基线，就没有证据证明新增复杂度值得。

## 复杂度与工程边界

### 计算复杂度

给定 $n$ 个样本，计算一个固定阈值的经验风险需要 $O(n)$ 时间和 $O(n)$ 输出空间；若只累计总风险，可以把额外空间降到 $O(1)$。

直接枚举 $m$ 个阈值需要 $O(nm)$ 时间。二分类阈值只会在排序后的唯一概率之间改变预测，因此先按概率排序，再累计错误计数，可把搜索降到 $O(n\log n)$ 时间和 $O(n)$ 空间。第 02 章会比较枚举、闭式解和迭代优化。

### 上线前必须回答的边界问题

| 风险 | 失败机制 | 最低检查 |
|---|---|---|
| 标签延迟 | 最近数据的“未故障”可能只是尚未走完 24 小时窗口 | 丢弃标签尚未成熟的样本 |
| 时间泄漏 | 维修工单、停机代码或未来聚合进入特征 | 逐字段记录可用时间并做 cutoff 测试 |
| 选择偏差 | 只有被检修设备才会拆机确认真实状态 | 分析标签缺失机制，保留随机抽检或审计样本 |
| 概率失准 | 训练分布变化后，0.2 不再代表约 20% 故障率 | 按时间和工况检查校准曲线 |
| 代价漂移 | 产能、备件和班次改变漏检/误检代价 | 版本化代价矩阵并重新做阈值敏感性分析 |
| 反馈回路 | 触发检修后故障被避免，观测标签受模型动作影响 | 记录策略与反事实不可观测性，避免把“未故障”直接当模型正确 |
| 群组风险 | 新旧设备、厂区和负载区间表现不同 | 报告设备代际与工况切片，不只看总体平均 |

<MlQuiz kind="错误诊断" question="模型上线后，被预警的泵都及时检修，因此它们没有发生故障。能否直接把这些样本标成模型误报？" :options="['能，未故障就一定是误报', '不能，检修动作改变了结果，未观测到不检修时的反事实', '能，只要样本量足够', '不能，因为分类任务没有标签']" :answer="1" explanation="动作干预了未来结果。若没有随机抽检、对照策略或因果假设，不能知道这些设备在不检修时是否会故障。" />

## 章节练习：写出一份可执行学习合同

选择“产品外观缺陷复检”“仓库缺货预警”或“泵故障预警”之一，完成 45–60 分钟练习。

### 任务要求

1. 写出 $\mathcal X,\mathcal Y,\mathcal A,\mathcal H,L,P,D,\Pi$。
2. 给出两种错误的实际单位与代价来源，不允许只写“错误率”。
3. 从条件风险推导行动阈值。
4. 写出一个最简单基线和一个机器学习候选。
5. 设计时间或分组切分，指出至少一种泄漏路径。
6. 用 `cost_sensitive_risk` 比较两个候选阈值，并添加一个无效输入测试。

<details><summary>公式骨架</summary>

先分别写两个动作的条件风险：

$$
R(a=0\mid x)=\underline{\qquad},\qquad
R(a=1\mid x)=\underline{\qquad}.
$$

再求使 $R(a=1\mid x)<R(a=0\mid x)$ 成立的 $p(x)$ 区间。最后写出经验风险：

$$
\widehat R_n(h)=\underline{\qquad}.
$$

</details>

<details><summary>代码骨架</summary>

```python
# TODO：替换成你的标签、概率、成本和候选阈值
y_exercise = np.array([0, 1, 0, 1])
p_exercise = np.array([0.1, 0.7, 0.4, 0.6])
thresholds_exercise = [0.3, 0.5, 0.7]
```

</details>

<details><summary>参考答案检查点</summary>

- 输入只能包含决策时点已经可用的信息。
- 标签窗口和动作窗口必须具体，例如“未来 24 小时内发生目标故障”和“下一班次安排检修”。
- 代价要能追溯到业务假设；不确定时给出区间并做敏感性分析。
- 阈值推导依赖概率校准，不能把任意模型分数直接当概率。
- 验证集用于选择阈值，测试集在所有选择冻结后只评价一次。
- 代码应拒绝形状不一致、概率越界、非二元标签和负成本。

</details>

## 8. 本章小结与术语

- 监督学习系统从实际行动开始，不从算法名称开始。
- 总体风险是未来分布上的目标，经验风险是有限样本上的可计算估计。
- ERM 在给定假设空间内最小化经验风险；它不自动保证总体风险最低。
- 假设空间、正则化、架构、优化和数据定义都包含归纳偏置。
- 工业评价必须把标签时间、行动代价、模型选择和反馈回路写进协议。

<dl class="ml-terms">
  <div><dt>population risk</dt><dd>未来联合分布上损失的期望。</dd></div>
  <div><dt>empirical risk</dt><dd>固定数据集上逐样本损失的平均。</dd></div>
  <div><dt>ERM</dt><dd>在假设空间中选择经验风险最低候选的原则。</dd></div>
  <div><dt>hypothesis space</dt><dd>学习算法允许考虑的函数集合。</dd></div>
  <div><dt>inductive bias</dt><dd>从有限样本向未见输入外推时使用的结构偏好。</dd></div>
  <div><dt>evaluation protocol</dt><dd>规定数据切分、模型选择、阈值选择和最终测试的合同。</dd></div>
</dl>

## 来源与许可

- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 224–225 页，§5.4.1：用于核对总体风险、经验风险、ERM、近似误差与估计误差的知识结构。本站独立推导和表述，不复用原文、图表或案例。
- Stanford CS229 Lecture Notes，PDF 第 128–130 页，§8.3.1–8.3.2：用于核对训练误差、总体误差、假设类与有限假设类 ERM 的推导边界。本站不复制讲义表达、图表或题目。
- 林轩田《机器学习基石》课程笔记 1：*The Learning Problem*，PDF 第 3–4 页：用于核对数据、假设集合、学习算法与最终假设的术语关系；不复用原图、原文或案例。
- 本章水泵案例、数据、代码、交互、题目和图形均为本站独立创作；全部设备数据为教学合成数据。详细登记见[来源与许可登记](./sources.md) 。

<MlLessonProgress lesson-id="01" next-label="第 02 章：梯度下降、牛顿法与随机优化" next-href="/courses/machine-learning/02-optimization" />
