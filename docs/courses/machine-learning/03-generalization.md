---
course: machine-learning
edition: industrial
title: 泛化、偏差—方差与模型选择
description: 以跨产线表面缺陷密度预报为贯穿案例，区分训练目标与目标域风险，完整推导平方损失的偏差—方差—噪声分解，并建立防止验证集过拟合的切分、交叉验证、嵌套选择与学习曲线协议。
outline: [2, 3]
---

<MlLessonHero number="03" title="泛化、偏差—方差与模型选择" question="模型已经把训练损失降得很低，凭什么相信它能预测下一批、下一周和另一条产线？" duration="建议 360–480 分钟" :outcomes="['把训练误差、同分布总体风险与目标域风险写成不同对象', '逐项推导平方损失的噪声、平方偏差与方差分解', '根据样本依赖结构设计训练、验证与最终测试边界', '从零实现折内预处理、K 折与嵌套模型选择', '用学习曲线、重复切分和独立审计诊断欠拟合、过拟合与选择偏差']" />

第 02 章保证优化器能降低给定训练目标。本章研究更棘手的证据问题：训练目标来自已经见过的有限数据，而系统要承担的是尚未发生、甚至来自另一条产线的数据。

“训练集准确率 99%”不是泛化证据；“测试集准确率 99%”也只有在测试集从未参与任何选择、且代表部署目标时才有意义。

## 贯穿案例：跨产线表面缺陷密度预报

一家板材工厂希望在带钢进入终检前，根据张力、速度和温度等上游工艺量，预测单位面积的表面缺陷密度。预测值用于安排复检资源，不直接闭环控制设备。

本章把多个工艺量压缩成一个已经冻结的“张力偏离指数” $x$，专门观察模型复杂度与数据切分。真实项目不能随意压缩成一维；这里的案例、数值与生产线均为**教学合成数据**。

### 部署问题决定切分，而不是随机数生成器

| 数据区 | 业务范围 | 允许用途 |
|---|---|---|
| 训练集 | A 线较早批次 | 拟合参数与所有数据依赖预处理 |
| 验证集 | A 线较晚批次 | 选择多项式次数、正则强度、特征方案和停止点 |
| 最终测试集 | B 线冻结批次 | 完整流程冻结后做一次跨产线审计 |
| 上线监控 | B 线未来批次 | 监测漂移与真实表现，不反写历史测试结论 |

如果目标只是“A 线下一周”，随机逐行切分已经忽略时间依赖；如果目标是“迁移到 B 线”，只在 A 线随机 K 折更无法模拟目标域。切分协议必须跟随预期部署边界。

```python
import numpy as np

rng = np.random.default_rng(20260805)

def line_a_process(x):
    return 0.52 + 0.20 * x + 0.31 * np.sin(2.35 * x)

def line_b_process(x):
    # B 线有未被现有特征完全解释的基线与斜率偏移。
    return line_a_process(x) + 0.055 + 0.035 * x

x_train = rng.uniform(-1.0, 1.0, 160)
x_valid = rng.uniform(-1.0, 1.0, 80)
x_test = rng.uniform(-0.9, 1.1, 100)

y_train = line_a_process(x_train) + rng.normal(0, 0.16, x_train.size)
y_valid = line_a_process(x_valid) + rng.normal(0, 0.16, x_valid.size)
y_test = line_b_process(x_test) + rng.normal(0, 0.18, x_test.size)

for features, labels in ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)):
    assert features.ndim == labels.ndim == 1
    assert features.shape == labels.shape
    assert np.isfinite(features).all() and np.isfinite(labels).all()
```

<MlQuiz kind="设计判断" question="目标是评估模型迁移到 B 线的表现。下列哪种最终测试最能支持这一结论？" :options="['从 A 线训练样本随机留出 20%', '使用参与特征筛选的 B 线样本', '在所有方案冻结后，使用从未参与选择的 B 线代表性批次', '报告 A 线训练误差']" :answer="2" explanation="最终证据必须同时满足信息独立与目标代表性。A 线随机留出不能回答跨产线问题；B 线样本若参与了筛选，也失去独立审计作用。" />

## 1. 训练误差、总体风险与目标域风险

对固定模型 $h$，训练集 $D=\{z_i\}_{i=1}^n$ 上的经验风险是

$$
\widehat R_D(h)=\frac1n\sum_{i=1}^n L(h,z_i).
$$

若未来数据仍来自训练分布 $P$，总体风险为

$$
R_P(h)=\mathbb E_{Z\sim P}[L(h,Z)].
$$

跨产线部署来自目标分布 $Q$，真正关心的则是

$$
R_Q(h)=\mathbb E_{Z\sim Q}[L(h,Z)].
$$

即使独立测试集能准确估计 $R_P(h)$，也不自动估计 $R_Q(h)$。这一区分把两类失败分开：

- **有限样本泛化问题**：训练与未来都来自 $P$，但模型迎合了训练样本的偶然波动；
- **分布偏移问题**：未来来自不同的 $Q$，即使在 $P$ 上估计无偏，也可能迁移失败。

学习算法 $A$ 读取随机训练集并输出 $\widehat h_D=A(D)$。算法的期望目标风险还要对训练集随机性取期望：

$$
\mathcal R_Q(A,n)
=\mathbb E_{D\sim P^n}\left[R_Q(\widehat h_D)\right].
$$

因此泛化不是某个静态模型的单一属性，还依赖样本量、抽样机制、训练算法和选择流程。

<MlQuiz kind="概念辨析" question="“泛化间隙”与“目标域偏移”有什么不同？" :options="['二者完全相同', '泛化间隙比较经验风险与同一机制下的未来风险；目标域偏移还改变了评价分布', '前者只用于分类，后者只用于回归', '只要训练集足够大，二者都必然为零']" :answer="1" explanation="有限样本选择会产生同分布泛化间隙；跨产线时还要面对 P 与 Q 的差异。更多 P 数据不一定覆盖 Q。" />

### 训练误差为什么通常偏乐观

对训练前固定的 $h$，$\widehat R_D(h)$ 可以是 $R_P(h)$ 的无偏估计。但训练后的 $\widehat h_D$ 正是用同一份 $D$ 选择出来的：

$$
\widehat h_D\in\arg\min_{h\in\mathcal H}\widehat R_D(h).
$$

最小化操作会偏好那些在当前样本上偶然表现好的候选。不能把

$$
\mathbb E\left[\min_h\widehat R_D(h)\right]
$$

与

$$
\min_h\mathbb E\left[\widehat R_D(h)\right]
$$

互换；一般有“期望的最小值不大于最小期望值”。候选越多，吸收偶然低估的机会越大。

<div class="ml-stop"><strong>停下来推一推 1</strong>两个硬币真实正面率都为 0.5，各抛 10 次后选择正面率更低的硬币。被选硬币的样本正面率为何会低于 0.5，而它下一次抛掷的真实概率仍是 0.5？把“硬币”替换为“候选模型”。</div>

## 2. 复杂度不是参数数量的同义词

模型复杂度描述候选函数对数据变化的响应能力。它可能由多项式次数、树深、正则强度、核带宽、网络范数、训练轮数和优化器的隐式偏好共同决定。

更大的参数量不必然意味着更高的有效复杂度：强正则的大网络可能比未正则的小网络稳定；两个都达到零训练误差的解，也可能因参数范数与决策边界不同而拥有不同目标风险。

在经典有限样本图景中，复杂度增加通常让训练误差不升高：

$$
\mathcal H_1\subseteq\mathcal H_2
\Longrightarrow
\min_{h\in\mathcal H_2}\widehat R_D(h)
\le
\min_{h\in\mathcal H_1}\widehat R_D(h).
$$

但未来误差可能先降后升。欠拟合是表示或优化连训练规律都未解释好；过拟合是模型对训练数据很好，却对独立目标数据明显变差。二者必须用至少两条曲线判断，不能只看训练误差。

## 3. 平方损失的偏差—方差—噪声分解

偏差—方差分解是对**训练集重抽样后的算法预测**做分析，不是把一份数据上的误差随意拆成三列。

固定输入 $x$，假设标签由

$$
Y=f(x)+\varepsilon,
\qquad
\mathbb E[\varepsilon\mid x]=0,
\qquad
\operatorname{Var}(\varepsilon\mid x)=\sigma^2(x)
$$

生成。随机训练集 $D$ 产生预测 $\widehat f_D(x)$。新的测试噪声与训练集独立。定义平均预测

$$
\overline f(x)=\mathbb E_D[\widehat f_D(x)].
$$

我们关心固定 $x$ 处的新样本平方误差：

$$
\mathbb E_{D,Y}\left[(Y-\widehat f_D(x))^2\right].
$$

在误差中加减 $f(x)$ 与 $\overline f(x)$：

$$
Y-\widehat f_D
=\underbrace{Y-f}_{\varepsilon}
+\underbrace{f-\overline f}_{\text{系统偏离}}
+\underbrace{\overline f-\widehat f_D}_{\text{抽样波动}}.
$$

平方展开有三项平方和三项交叉乘积。交叉项为零的理由分别是：

1. $\mathbb E[\varepsilon\mid x]=0$；
2. 新测试噪声独立于训练集，因此与 $\widehat f_D$ 的乘积期望为零；
3. $\mathbb E_D[\overline f-\widehat f_D]=0$。

最终得到

$$
\boxed{
\mathbb E_{D,Y}[(Y-\widehat f_D(x))^2]
=\sigma^2(x)
+\left(f(x)-\mathbb E_D\widehat f_D(x)\right)^2
+\mathbb E_D\left[(\widehat f_D(x)-\mathbb E_D\widehat f_D(x))^2\right]
}
$$

即

$$
\text{期望测试 MSE}
=\text{噪声}
+\text{平方偏差}
+\text{方差}.
$$

对整个输入分布的风险，还要对 $X\sim P_X$ 再取期望。若噪声方差随 $x$ 改变，噪声项是 $\mathbb E_X[\sigma^2(X)]$。

<MlQuiz kind="推导补全" question="分解中的 variance 对什么随机性取方差？" :options="['固定训练集内标签的离散程度', '输入单位变化', '训练集 D 重抽样后，同一 x 上预测 f̂_D(x) 的变化', '损失函数在一个 batch 内的均值']" :answer="2" explanation="预测器是训练集的函数。方差衡量重复采样和重新训练后预测有多不稳定，不是单份训练集标签的样本方差。" />

<BiasVarianceTradeoffLab />

### 三个术语的工程含义

- **平方偏差高**：即使能重复收集大量同机制数据并平均模型，平均预测仍系统性偏离真实函数；常见原因是表示不足、过强正则或遗漏关键特征。
- **方差高**：换一批训练样本，模型预测明显变化；常见原因是样本少、候选空间灵活、共线性或选择流程过度搜索。
- **噪声高**：给定当前可用信息，结果仍有随机性或未观测因素；增加模型复杂度不能预测定义上不可知的部分。

增加数据通常降低方差，却不自动降低由错误特征或目标域偏移造成的偏差；增加复杂度可能降低偏差、提高方差，但这不是不可违背的单调定律。

### 这个分解不能被滥用

上述加法形式依赖平方损失和条件均值假设。对 0–1 分类损失，没有同样通用的“噪声 + 平方偏差 + 方差”加法分解。分类工程中更稳妥的做法是直接估计目标损失、校准误差和决策代价。

现代过参数模型还可能出现双下降：测试误差在插值阈值附近上升，随后随参数继续增加再次下降。它提醒我们，“参数数量”不是完整复杂度指标，优化器与正则化会选择某类插值解。经典偏差—方差仍是诊断起点，不是所有曲线形状的定理。

<MlQuiz kind="错误诊断" question="团队把二分类准确率直接写成 noise+bias²+variance，并声称三项可相加。问题是什么？" :options="['准确率不能超过 1', '平方偏差加法分解针对平方损失回归；0–1 损失没有这个通用形式', '分类没有训练集', '只需把方差开平方即可']" :answer="1" explanation="分解来自平方展开和交叉项消失。更换为 0–1 损失后代数结构改变，不能只保留术语而丢掉假设。" />

<div class="ml-stop"><strong>停下来推一推 2</strong>若同一条带钢记录同时参与训练与“测试”，新测试噪声独立于训练集的条件还成立吗？这会破坏推导中的哪一个交叉项？</div>

## 4. 数值估计偏差与方差

真实项目通常只有一份历史数据，无法直接观测无限次训练集重抽样。仿真可以验证公式，却不能把合成的真实函数当成生产真相。

```python
def polynomial_design(x, degree):
    x = np.asarray(x, dtype=float)
    return np.column_stack([x ** power for power in range(degree + 1)])

def fit_polynomial_ridge(x, y, degree, ridge=1e-4):
    design = polynomial_design(x, degree)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)

def predict_polynomial(theta, x):
    return polynomial_design(x, theta.size - 1) @ theta

def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

degrees = [1, 3, 5, 9]
records = []
for candidate_degree in degrees:
    theta = fit_polynomial_ridge(x_train, y_train, candidate_degree)
    records.append({
        "degree": candidate_degree,
        "train_mse": mse(y_train, predict_polynomial(theta, x_train)),
        "valid_mse": mse(y_valid, predict_polynomial(theta, x_valid)),
    })

selected_degree = min(records, key=lambda item: item["valid_mse"])["degree"]
assert selected_degree in degrees
assert all(item["train_mse"] >= 0 and item["valid_mse"] >= 0 for item in records)
```

注意这里只允许验证集选择次数。`y_test` 仍未读取；否则最终测试就被纳入了选择回路。

```python
def estimate_bias_variance(degree, x0=0.65, repeats=400, sample_size=24, noise_std=0.16):
    simulation_rng = np.random.default_rng(3103 + degree)
    predictions = []
    for _ in range(repeats):
        x_sample = simulation_rng.uniform(-1, 1, sample_size)
        y_sample = line_a_process(x_sample) + simulation_rng.normal(0, noise_std, sample_size)
        theta = fit_polynomial_ridge(x_sample, y_sample, degree)
        predictions.append(float(predict_polynomial(theta, np.array([x0]))[0]))
    predictions = np.asarray(predictions)
    truth = float(line_a_process(np.array([x0]))[0])
    return {
        "bias_squared": float((predictions.mean() - truth) ** 2),
        "variance": float(predictions.var(ddof=0)),
        "noise": noise_std ** 2,
    }

decomposition = {candidate: estimate_bias_variance(candidate) for candidate in (1, 3, 9)}
assert decomposition[1]["bias_squared"] > decomposition[3]["bias_squared"]
assert decomposition[9]["variance"] > decomposition[3]["variance"]
```

这是 Monte Carlo 估计：`repeats` 越大，模拟误差通常越小，但它只验证指定生成机制。真实 B 线的未知偏移仍不能由 A 线仿真证明。

## 5. 参数、超参数与完整训练流程

信息边界要覆盖所有会被数据决定的选择：

| 类型 | 案例 | 应使用的数据 |
|---|---|---|
| 参数 | 固定次数多项式的系数 | 训练集 |
| 超参数 | 次数、Ridge 强度、早停轮数 | 验证集或内层 CV |
| 数据处理 | 缺失填补、标准化、特征筛选、PCA | 每个训练折内部拟合 |
| 决策规则 | 缺陷密度触发复检的阈值 | 验证集，且按业务损失选择 |
| 最终报告 | 冻结流程在 B 线的风险与区间 | 最终测试集 |

“没有用测试集反向传播”远远不够。人根据测试分数选择特征、模型、阈值或论文版本，同样把测试信息传回了流程。

<ValidationSelectionLab />

<MlQuiz kind="结果预测" question="每个候选的验证误差都是近似无偏估计。比较 500 个候选并报告其中最小值时，最小值通常怎样？" :options="['仍严格无偏', '偏乐观，因为选择偏向偶然低估的候选', '一定偏悲观', '只由训练误差决定']" :answer="1" explanation="单个估计无偏不代表取最小值后无偏。选择操作会吸收验证噪声；尝试越多、验证集越小，问题通常越明显。" />

### 防止验证集被“看熟”

- 在实验开始前定义主指标、候选空间与停止规则；
- 把特征工程、数据清洗和阈值也纳入同一选择预算；
- 保存所有尝试，而不是只保留最佳结果；
- 对重要结论使用重复切分或嵌套 CV 量化选择稳定性；
- 最终测试集只在流程冻结后打开一次；若据此修改，必须获得新的独立审计数据。

## 6. K 折交叉验证估计的是什么

把训练开发数据索引分成 $K$ 个互斥折 $I_1,\ldots,I_K$。第 $k$ 轮在 $D_{-k}$ 上拟合完整流程 $A_\lambda$，在 $I_k$ 上评价：

$$
\widehat R_{CV}(\lambda)
=\frac1n\sum_{k=1}^{K}\sum_{i\in I_k}
L\left(y_i,A_\lambda(D_{-k})(x_i)\right).
$$

这种按样本加权的写法在折大小不等时仍正确。简单平均 $K$ 个折均值只在折大小相同或明确接受等折权重时等价。

CV 估计的是“用约 $(K-1)n/K$ 个样本训练的流程”的表现，不是最后用全部数据重训后那个参数向量的完全无偏风险。各折模型共享大量训练数据，折分数也不是相互独立样本。

### 折的单位必须匹配独立单位

- 同一带钢的相邻窗口：整条带钢放在同一折，避免近重复泄漏；
- 时间预测：用过去训练、较晚时间验证的前向切分；
- 类别不平衡且样本独立：可分层，使各折类别比例稳定；
- 跨产线泛化：可用 leave-one-line-out 或指定目标产线外层测试；
- 空间相关数据：按区域分块，不能随机打散邻近像素。

### 预处理必须在折内拟合

若先在全体数据上计算均值、选择特征或做 PCA，再运行 CV，验证折已经影响训练表示。正确顺序是每轮：

$$
D_{-k}
\xrightarrow{\text{fit transform}}
T_k
\xrightarrow{\text{transform train/valid}}
\xrightarrow{\text{fit model}}
\xrightarrow{\text{score }I_k}.
$$

```python
def kfold_indices(sample_count, folds, seed=0):
    if not 2 <= folds <= sample_count:
        raise ValueError("folds 必须位于 2 与样本数之间")
    indices = np.random.default_rng(seed).permutation(sample_count)
    return [part.astype(int) for part in np.array_split(indices, folds)]

def cross_validated_mse(x, y, degree, folds=5, ridge=1e-4, seed=0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x 与 y 必须是等长一维数组")
    fold_indices = kfold_indices(x.size, folds, seed)
    squared_errors = []
    all_indices = np.arange(x.size)

    for validation_indices in fold_indices:
        train_mask = ~np.isin(all_indices, validation_indices)
        train_indices = all_indices[train_mask]

        # 标准化参数只来自当前训练折。
        mean = x[train_indices].mean()
        scale = x[train_indices].std(ddof=0)
        if scale == 0:
            raise ValueError("训练折特征没有变化")
        x_train_fold = (x[train_indices] - mean) / scale
        x_valid_fold = (x[validation_indices] - mean) / scale

        theta = fit_polynomial_ridge(x_train_fold, y[train_indices], degree, ridge)
        prediction = predict_polynomial(theta, x_valid_fold)
        squared_errors.extend((y[validation_indices] - prediction) ** 2)

    assert len(squared_errors) == x.size
    return float(np.mean(squared_errors))

cv_scores = {candidate: cross_validated_mse(x_train, y_train, candidate, seed=9) for candidate in degrees}
assert all(np.isfinite(score) and score >= 0 for score in cv_scores.values())
```

<MlQuiz kind="代码预测" question="为什么上面每个折都重新计算 mean 与 scale？" :options="['为了让数组更短', '避免验证折参与预处理参数估计，否则 CV 风险偏乐观', '因为多项式不能处理负数', '只为提高运行速度']" :answer="1" explanation="数据依赖预处理是训练流程的一部分。全体标准化会让验证折的信息进入训练表示，尤其在时间、群组或强漂移数据中影响明显。" />

## 7. 嵌套交叉验证分离选择与评价

若同一组 CV 分数既用于选择超参数，又被当作最终性能报告，报告仍受选择偏差影响。嵌套 CV 用两层信息边界：

1. 外层划出完全不参与选择的评价折；
2. 在每个外层训练区内，内层 CV 选择超参数；
3. 用选出的设置重训外层训练区，在外层评价折打分；
4. 汇总外层样本的预测损失。

伪代码中的关键不是“循环两次”，而是外层验证样本不能影响内层任何候选选择。

```python
def nested_degree_selection(x, y, candidate_degrees, outer_folds=4, inner_folds=4, seed=0):
    outer = kfold_indices(len(y), outer_folds, seed)
    all_indices = np.arange(len(y))
    outer_losses = []
    chosen_degrees = []

    for outer_valid in outer:
        outer_train = all_indices[~np.isin(all_indices, outer_valid)]
        inner_scores = {
            candidate: cross_validated_mse(
                x[outer_train], y[outer_train], candidate,
                folds=inner_folds, seed=seed + int(outer_valid[0]) + candidate
            )
            for candidate in candidate_degrees
        }
        chosen = min(inner_scores, key=inner_scores.get)
        chosen_degrees.append(chosen)

        theta = fit_polynomial_ridge(x[outer_train], y[outer_train], chosen)
        prediction = predict_polynomial(theta, x[outer_valid])
        outer_losses.extend((y[outer_valid] - prediction) ** 2)

    return float(np.mean(outer_losses)), chosen_degrees

nested_mse, chosen_by_outer_fold = nested_degree_selection(x_train, y_train, degrees, seed=17)
assert np.isfinite(nested_mse) and nested_mse >= 0
assert len(chosen_by_outer_fold) == 4
```

上面的最终外层拟合为了突出嵌套边界，未做特征标准化；一维多项式仍可运行。真实流水线必须让外层重训使用与内层候选完全相同的预处理定义。

### 何时不需要 K 折

数据量足够大且存在自然的时间外、客户外或产线外验证区时，单个大规模 holdout 往往更贴近部署、更容易审计。K 折不是仪式；它以多次训练成本换取有限样本利用率，不能修复错误目标域。

<div class="ml-stop"><strong>停下来推一推 3</strong>若 5 折 CV 搜索 30 组超参数，再做 4 折外层评价，最朴素实现至少训练多少次候选模型？外层最终重训还要加多少次？说明为何要先缩小合理候选空间。</div>

## 8. 学习曲线把“多收数据”变成可检验假设

学习曲线横轴是训练样本量，纵轴同时画训练与独立验证误差。每个样本量都应重复抽样并报告分布，而不是只画一个幸运种子。

| 曲线现象 | 可能原因 | 下一条证据 |
|---|---|---|
| 训练与验证都高且接近 | 表示偏差高、特征不足或优化未收敛 | 检查训练目标、梯度与更强表示 |
| 训练低、验证高且差距大 | 方差高、泄漏或切分机制不同 | 重复切分、群组审计、正则与更多独立样本 |
| 样本增大时验证持续下降 | 更多相似独立数据可能有价值 | 外推数据收益并给不确定区间 |
| 验证早早平台 | 当前信息或目标定义形成上限 | 错误分层、补特征、检查标签噪声 |
| 训练很好，B 线始终差 | 目标域偏移 | 分产线误差、协变量/概念漂移检查 |

```python
def learning_curve(x, y, degree, sizes, repeats=20, seed=0):
    curve_rng = np.random.default_rng(seed)
    result = []
    for size in sizes:
        train_scores = []
        valid_scores = []
        for _ in range(repeats):
            subset = curve_rng.choice(len(y), size=size, replace=False)
            theta = fit_polynomial_ridge(x[subset], y[subset], degree)
            train_scores.append(mse(y[subset], predict_polynomial(theta, x[subset])))
            valid_scores.append(mse(y_valid, predict_polynomial(theta, x_valid)))
        result.append({
            "size": size,
            "train_mean": float(np.mean(train_scores)),
            "valid_mean": float(np.mean(valid_scores)),
            "valid_std": float(np.std(valid_scores, ddof=0)),
        })
    return result

curve = learning_curve(x_train, y_train, degree=9, sizes=[20, 40, 80, 120], seed=23)
assert [point["size"] for point in curve] == [20, 40, 80, 120]
assert all(point["valid_std"] >= 0 for point in curve)
```

同一验证集被用于整条学习曲线的诊断，不能再把曲线上最低点当作无偏最终报告。它仍属于开发信息。

<MlQuiz kind="证据判断" question="高次数模型的验证误差随训练样本量增加持续下降，但 120 个样本处仍明显高于训练误差。最稳妥的结论是什么？" :options="['更多同机制独立样本可能继续降低方差，但应带区间验证，不能保证解决目标域偏移', '模型一定存在标签泄漏', '不可约噪声已经为零', '应立即打开测试集反复调参']" :answer="0" explanation="曲线支持“更多相似数据可能有帮助”，不是保证。仍要区分有限样本方差与 B 线分布偏移，并保护最终测试边界。" />

## 数值验证

现在才打开冻结的 B 线测试集：先用验证集选出的次数，在 A 线训练与验证数据合并后重训，再做一次目标域评价。

```python
x_development = np.concatenate([x_train, x_valid])
y_development = np.concatenate([y_train, y_valid])
final_theta = fit_polynomial_ridge(x_development, y_development, selected_degree)
final_test_mse = mse(y_test, predict_polynomial(final_theta, x_test))

assert np.isfinite(final_test_mse) and final_test_mse >= 0
assert x_development.size == 240
print(f"冻结流程：degree={selected_degree}，B 线测试 MSE={final_test_mse:.4f}")
```

验证链应至少回答：

1. 候选的训练误差是否随复杂度合理变化；
2. 偏差—方差仿真是否在已知机制上复现推导；
3. 每个 CV 样本是否恰好被折外预测一次；
4. 预处理是否只在对应训练折拟合；
5. 内层选择是否完全看不到外层评价折；
6. 最终 B 线测试是否只在流程冻结后执行。

一次测试分数仍有抽样不确定性。分类错误率可用二项区间；MSE 可按独立单位 bootstrap，但若同一带钢窗口相关，必须按带钢成组重采样。

## 复杂度与工程边界

令 $C_{fit}(n,p)$ 为一次训练成本，$M$ 为候选数，$K$ 为折数：

| 流程 | 训练次数量级 | 主要用途 | 边界 |
|---|---:|---|---|
| 单一 holdout | $M$，选定后再重训 1 次 | 数据充足、有自然目标验证区 | 单次切分方差，开发集会被反复使用 |
| K 折选择 | $MK$，选定后再重训 1 次 | 数据较少、独立同分布近似合理 | 计算高，各折相关，不代表新目标域 |
| 重复 K 折 | $MKR$ | 估计切分稳定性 | 成本线性增加，重复结果也参与开发 |
| 嵌套 CV | 约 $K_{out}(MK_{in}+1)$ | 选择流程的近无泄漏评价 | 成本高，最终仍受数据代表性限制 |
| 时间前向验证 | 每个时间窗口重训 | 未来预测、漂移评估 | 早期训练集小，窗口选择影响结论 |
| Leave-one-group-out | 组数次 | 客户、设备、产线外泛化 | 组很少时区间宽，组间差异可能主导 |

### 工业模型选择清单

- 在实验前冻结样本单位、特征截止、标签窗口和目标域；
- 给数据集版本与每个样本分区生成可复查清单；
- 把预处理、数据增强、特征选择、阈值和早停封装进折内流程；
- 记录全部候选与失败运行，不只保存最佳分数；
- 同时报均值、离散程度、样本/组数量和逐组结果；
- 检查训练、验证、测试之间的实体重复、时间重叠和近重复内容；
- 最终测试结果若触发任何修改，明确把它降级为开发结果并获取新审计数据；
- 上线后按时间、产线和关键工况监控，离线测试不是永久认证。

### 常见误判

- 训练与验证都低，不等于部署风险低：两者可能共享泄漏或同一偏差机制。
- 固定随机种子不等于消除抽样不确定性：它只让一次切分可复现。
- 更多数据不等于更多信息：同一带钢切成大量相邻窗口会制造伪样本量。
- K 越大不等于估计越好：训练成本和折间相关性上升，方差性质并非单调改善。
- 测试集很大不等于代表目标：抽样机制错误时，窄区间只是精确估计了错误分布。

<MlQuiz kind="错误诊断" question="团队有 50 万个窗口，随机切分后区间极窄；但窗口来自仅 12 条带钢，同一带钢跨越训练和测试。主要问题是什么？" :options="['窗口数量还不够', '独立单位其实接近 12，随机窗口切分造成组泄漏和虚假的精度', '置信区间越窄说明泛化越好', '只需增加 K 折的 K']" :answer="1" explanation="相邻窗口共享材料、工况和噪声，不能当作 50 万个独立单位。应按带钢分组切分并按组估计不确定性。" />

## 9. 章节练习：设计一份不可偷看的评价协议

### 推导练习

1. 不省略交叉项，完整展开偏差—方差分解，并逐项标注使用了哪个零均值或独立性条件。
2. 证明对任意随机候选风险估计 $\widehat R_j$，通常有 $\mathbb E[\min_j\widehat R_j]\le\min_j\mathbb E[\widehat R_j]$；说明何时取等号。
3. 对大小不等的 K 折，比较“折均值简单平均”与“所有折外损失按样本平均”的差异。

### 实现练习

扩展 `nested_degree_selection`：在内层与外层都执行训练折标准化，返回每个外层样本的折外预测、候选稳定性和按产线分组的 MSE。

```python
# TODO：完成带折内预处理与折外预测的嵌套选择；本代码块由学习者执行。
def nested_out_of_fold_predictions(x, y, groups, candidates):
    # 1. 外层按 groups 留组
    # 2. 内层只在外层训练组中选择候选
    # 3. 所有数据依赖变换只拟合当前训练区
    # 4. 返回与 y 等长的折外预测和每折元数据
    raise NotImplementedError
```

### 协议审计练习

找出以下流程中的全部信息泄漏：先在全体 A/B 线数据上删除“与标签相关性低”的特征；对全体数据标准化；随机窗口 K 折选择模型；查看 B 线测试分数后调整阈值；只报告最佳种子。

答案必须给出修订后的数据流、每一层允许使用的信息，以及最终结论究竟能推广到哪个时间、哪条产线和哪种工况。

<details><summary>检查点</summary>

- 特征筛选与标准化都必须在对应训练折内拟合。
- 同一带钢的窗口保持在同一分区。
- 若目标是 B 线，至少保留未参与任何选择的 B 线审计批次。
- 阈值与随机种子选择都属于模型选择，不能查看最终测试反馈。
- 报告候选空间、全部运行与不确定性，避免只展示幸运结果。

</details>

## 10. 本章小结与术语

- 经验风险、同分布总体风险和目标域风险是三个不同对象。
- 平方损失下，期望测试误差可分为噪声、平方偏差和训练集抽样导致的方差。
- 参数数量只是复杂度线索；正则化、优化和数据流程共同决定有效复杂度。
- 验证集用于选择，一旦反复使用也会过拟合；最终测试必须同时独立且代表目标域。
- K 折提高有限数据利用率，却不能修复时间、群组和目标域设计错误。
- 嵌套 CV 分离内层选择与外层评价；学习曲线把“增加数据是否有用”变成可检验假设。

<dl class="ml-terms">
  <div><dt>generalization</dt><dd>训练流程在未参与拟合和选择的数据上的表现能力。</dd></div>
  <div><dt>target risk</dt><dd>目标部署分布上损失的期望。</dd></div>
  <div><dt>bias</dt><dd>训练集重抽样后的平均预测与真实条件均值之间的系统偏离。</dd></div>
  <div><dt>variance</dt><dd>训练集重抽样引起的同一点预测波动。</dd></div>
  <div><dt>cross-validation</dt><dd>轮流用折外训练、折内评价来估计训练流程风险的方法。</dd></div>
  <div><dt>nested CV</dt><dd>用内层选择超参数、外层评价选择流程的信息隔离方案。</dd></div>
  <div><dt>selection bias</dt><dd>从多个带噪估计中选最优后产生的系统性乐观。</dd></div>
  <div><dt>learning curve</dt><dd>训练与验证误差随训练样本量变化的诊断曲线。</dd></div>
</dl>

## 来源与许可

- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 155–159 页，§4.5.4–4.5.7：用于核对验证集、K 折、one-standard-error 规则、早停与学习曲线的知识边界。本站案例、代码与评价协议独立创作。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 190–194 页，§4.7.6：用于核对估计量偏差、方差、MSE 分解及分类损失下的适用边界；不复用原文图表或案例。
- Stanford CS229 Lecture Notes，PDF 第 114–124 页，Ch.8：用于交叉核对训练/测试误差、回归偏差—方差推导与双下降边界。本站重新推导和表述，不复制讲义图形。
- Stanford CS229 Lecture Notes，PDF 第 140–143 页，§9.3：用于核对 holdout 与 K 折模型选择流程；嵌套选择、跨产线协议、题目和实现均为本站独立创作。
- 本章 A/B 产线、340 条合成样本、两项网页交互、全部题目、代码和 SVG 图形均为教学合成与本站独立创作，不代表真实工厂统计或质量标准。详细登记见[来源与许可登记](./sources.md) 。

<MlLessonProgress lesson-id="03" next-label="第 04 章：普通最小二乘法（OLS）" next-href="/courses/machine-learning/04-ols-geometry" />
