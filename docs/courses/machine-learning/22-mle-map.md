---
course: machine-learning
edition: industrial
title: 最大似然估计与最大后验估计
description: 以多工位机器视觉漏检率资格审核为贯穿案例，从生成故事、似然与充分统计量推导 Bernoulli、Categorical、Gaussian 的 MLE，再由先验、共轭更新和重复抽样风险建立 MAP、后验预测与生产审计边界。
outline: [2, 3]
---

<MlLessonHero
  number="22"
  title="最大似然估计与最大后验估计"
  question="一台视觉检验机审核了 36 个已知缺陷件，只漏掉 1 个；我们真正知道的是漏检率恰好为 1/36，还是只知道哪些参数能较好解释这批有限证据？"
  duration="建议 780–900 分钟"
  :outcomes="[
    '从生成故事、iid 分解和充分统计量完整推导 Bernoulli、Categorical 与 Gaussian 的 MLE，并解释边界解、曲率、可辨识性和数值稳定性',
    '从 Bayes 规则推导 MAP、正则化对应、Beta–Bernoulli、Dirichlet–Categorical 与 Gaussian–Gaussian 更新，严格区分 MAP、后验均值、可信区间和后验预测',
    '用重复抽样拆解偏差、方差与 MSE，审计先验错配、重复挑战样本、同批数据复用、工位异质性、选择性审核和上线反馈回路'
  ]"
/>

前 21 章已经多次用到平方损失、交叉熵、正则化和概率输出。本章把这些看似分散的做法放回同一个源头：先写清数据如何由参数生成，再问哪组参数最能解释已经观察到的数据；若还有冻结在数据之前的知识，再问数据与先验合在一起后，哪个参数位置最受支持。

这不是“频率学派和 Bayes 学派名词对照”一章。工业系统最终必须回答更具体的问题：零次漏检是否意味着漏检不可能发生；36 条审核记录是否真的有 36 份独立信息；先验来自哪里、是否和目标批次重复；发布动作需要一个点估计，还是需要“超过资格上限的后验概率”。

## 贯穿案例：多工位机器视觉漏检率资格审核

一条精密装配线有 16 个独立机器视觉检验工位。每个工位在启用新光学配置后，都要接受一批“挑战件”审核：挑战件的缺陷已经由离线计量确认，工位若没有触发拒绝或人工复核，就记为一次漏检。

本章估计的是**给定挑战件分布、审核规程和工位版本时的条件漏检率**，不是现场产品缺陷率，也不是整条产线的事故概率。统计结果只用于安排扩大审核、工程复核和监测频率；不得直接关闭安全联锁、自动放行产品或修改工艺参数。

### 数据合同、参数边界和动作

| 字段 | 工程语义 | 冻结与审计要求 |
|---|---|---|
| `station_id` | 视觉工位与光学配置版本 | 镜头、光源、相机、模型或阈值变化后必须换版本 |
| `challenge_id` | 已知缺陷挑战件的实体键 | 同一图像重放或同一实体重复通过不能伪装成独立样本 |
| `defect_family` | 划痕、污染、缺口等挑战类型 | 资格集配比必须与验收规程一起版本化 |
| `hard_mix` | 低对比度、边缘位置等困难挑战占比 | 不同工位审核配比不同会造成不可比 |
| `optics_age` | 光学组件服役状态摘要 | 维护、更换和清洁后语义重新冻结 |
| `missed` | 已知缺陷没有触发拒绝或复核 | 结果在动作发生后产生，不能回填为预测输入 |
| `review_outcome` | 自动拒绝、人工复核、漏检等互斥结果 | Categorical 类别字典和未决状态必须固定 |
| `focus_residual` | 标准靶标焦点评分相对标称值的残差 | 仪器、单位、温漂补偿和重复测量协议要固定 |

资格上限暂以 5% 作为教学阈值。示例动作协议是：若

$$
P(\theta_j>0.05\mid\mathcal D_j)>0.8,
$$

则第 $j$ 个工位进入扩大审核和工程复核。这个阈值只是演示；真实门槛必须来自产品风险、法规、量测能力、抽样方案和复核容量。点估计本身不直接触发放行。

### 1. 生成带有工位异质性的教学审核数据

下面构造 16 个工位。每个工位的真实漏检率只在合成实验中可见，用于检查估计风险；真实生产中当然看不到它。审核量、困难挑战占比和光学状态在工位之间不同。

```python
import math
import numpy as np

rng = np.random.default_rng(20260822)
station_count = 16
station_id = np.arange(station_count)
audit_size = np.array([
    24, 30, 36, 42, 48, 54, 60, 72,
    28, 40, 52, 64, 32, 44, 56, 68,
])
optics_age = (
    np.linspace(-1.1, 1.2, station_count)
    + rng.normal(0.0, 0.18, station_count)
)
hard_mix = np.array([
    0.18, 0.24, 0.31, 0.42, 0.27, 0.36, 0.49, 0.33,
    0.22, 0.45, 0.39, 0.52, 0.29, 0.41, 0.55, 0.47,
])

true_logit = -3.55 + 0.48 * optics_age + 1.30 * (hard_mix - 0.35)
true_miss_rate = 1.0 / (1.0 + np.exp(-true_logit))
miss_count = rng.binomial(audit_size, true_miss_rate)

assert station_id.shape == audit_size.shape == miss_count.shape
assert np.all((0 <= miss_count) & (miss_count <= audit_size))
assert np.all((0 < true_miss_rate) & (true_miss_rate < 1))
print("审核总量 / 漏检总数：", audit_size.sum(), miss_count.sum())
print("各工位漏检数：", miss_count.tolist())
```

本章不把 16 个工位随机拆成训练集和测试集，因为这里不是学习一个从特征到标签的预测函数，而是在给定审核规程下估计每个工位自己的参数。若要学习跨工位回归模型或从历史工位估计先验超参数，仍必须按时间和工位版本冻结开发、验证与最终审计边界。

### 2. 先建立不带先验的基线

每个工位最直接的估计是“漏检数除以审核数”。把全部工位合并得到的单一比例更简单，但它假设所有工位共享同一个参数，会掩盖工位与挑战配比差异。

```python
station_mle = miss_count / audit_size
pooled_mle = miss_count.sum() / audit_size.sum()
zero_count_stations = station_id[miss_count == 0]

station_mle_mse = np.mean((station_mle - true_miss_rate) ** 2)
pooled_mse = np.mean((pooled_mle - true_miss_rate) ** 2)

assert np.isclose(pooled_mle, 37 / 750)
assert zero_count_stations.tolist() == [2]
assert station_mle_mse > 0
print("合并漏检率：", round(pooled_mle, 4))
print("零漏检工位：", zero_count_stations.tolist())
print("逐工位 MLE / 合并常数的合成 MSE：", round(station_mle_mse, 6), round(pooled_mse, 6))
```

合并估计在这一次合成样本上的 MSE 可能很低，因为各工位真实参数彼此接近；它仍然不能回答“哪个工位需要扩大审核”。更简单的基线是否够用，取决于最终动作粒度，而不是只看一个全局误差数字。

<MlQuiz kind="设计判断" question="若同一张低对比度缺陷图被重放 20 次，得到 20 条完全一致的检测结果，能否在 Bernoulli iid 似然中直接当作 20 份独立证据？" :options="['可以，只要记录数是 20', '不可以；共享同一挑战实体会让似然曲率虚假变尖', '只有全部检出时可以', '只要先取对数就可以']" :answer="1" explanation="iid 分解要求条件独立。重复同一实体主要测量重放稳定性，不等于覆盖 20 个独立缺陷工况；实体键、聚类结构或有效样本量必须显式进入协议。" />

## 2. 概率、似然与生成故事

设一个挑战件的漏检结果为 $X\in\{0,1\}$，其中 $X=1$ 表示漏检。给定漏检率参数 $\theta$，模型写为

$$
p(x\mid\theta)=\theta^x(1-\theta)^{1-x}.
$$

同一个式子有两个阅读方向：

- 固定 $\theta$，让 $x$ 变化，$p(x\mid\theta)$ 是随机变量 $X$ 的概率质量函数，必须对 $x\in\{0,1\}$ 求和为 1；
- 固定已经观察到的数据 $x$，让 $\theta$ 变化，$L(\theta;x)=p(x\mid\theta)$ 是关于参数的似然，不要求对 $\theta$ 积分为 1。

“参数出现在哪个密度里”还不够，必须先写生成故事：工位版本冻结后，先存在一个未知漏检率 $\theta$；审核规程从目标挑战分布抽取样本；每个挑战件在给定 $\theta$ 和协变量后产生检测结果。若困难类型、光学状态或实体重复仍影响结果，单一 iid Bernoulli 故事就是错的。

### 2.1 iid 分解不是排版技巧，而是模型假设

若 $x_1,\ldots,x_n$ 在给定 $\theta$ 后独立同分布，则

$$
p(\mathcal D\mid\theta)
=\prod_{i=1}^n p(x_i\mid\theta).
$$

最大似然估计（maximum likelihood estimation, MLE）选择

$$
\hat\theta_{\mathrm{MLE}}
=\arg\max_{\theta\in\Theta}p(\mathcal D\mid\theta).
$$

乘积容易下溢，且乘法不便求导，所以计算对数似然

$$
\ell(\theta)
=\log p(\mathcal D\mid\theta)
=\sum_{i=1}^n\log p(x_i\mid\theta).
$$

对数严格单调递增，因此在似然为正的同一可行域上，最大化 $L$ 和 $\ell$ 的解相同。工程实现应保存总对数似然、平均负对数似然和样本数，不能只保存一个没有归一化口径的“loss”。

```python
def bernoulli_log_likelihood(successes, trials, theta):
    if not (isinstance(successes, (int, np.integer)) and isinstance(trials, (int, np.integer))):
        raise TypeError("successes 和 trials 必须是整数")
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("必须满足 0 <= successes <= trials 且 trials > 0")
    theta = np.asarray(theta, dtype=float)
    if np.any((theta <= 0) | (theta >= 1)):
        raise ValueError("本函数只计算开区间 (0, 1) 内的对数似然")
    return successes * np.log(theta) + (trials - successes) * np.log1p(-theta)

huge_trials = 100_000
huge_successes = 4_000
candidate = 0.04
raw_likelihood = candidate ** huge_successes * (1 - candidate) ** (huge_trials - huge_successes)
stable_log_likelihood = bernoulli_log_likelihood(huge_successes, huge_trials, candidate)

assert raw_likelihood == 0.0
assert np.isfinite(stable_log_likelihood)
print("直接乘积：", raw_likelihood)
print("稳定对数似然：", round(float(stable_log_likelihood), 2))
```

### 2.2 条件似然与联合似然回答不同问题

若目标是学习给定挑战属性 $z_i$ 时的漏检概率，可写

$$
p(\mathbf x\mid\mathbf Z,\theta)=\prod_i p(x_i\mid\mathbf z_i,\theta),
$$

把挑战属性视为给定条件。若连挑战类型如何被抽入审核也要建模，则需要联合模型 $p(x,z\mid\theta)$。两者对选择偏差的含义不同：条件模型不能自动修复“只挑容易样本审核”的挑战集设计。

<MlQuiz kind="错误诊断" question="某实现把 100000 个 Bernoulli 概率直接相乘得到 0，于是断言模型认为数据不可能发生。根因是什么？" :options="['MLE 只能用于高斯', '浮点下溢；应累计 log p 并使用 log1p', '样本数必须小于 100', '先验太弱']" :answer="1" explanation="大量小于 1 的因子相乘会下溢到 0。对数域把乘积变成有限求和；这修复数值表示，不会修复错误的 iid 假设。" />

<LikelihoodProfileScannerLab />

## 3. Bernoulli MLE：样本比例为什么出现

记漏检总数

$$
S=\sum_{i=1}^n x_i,
$$

则联合似然与对数似然为

$$
L(\theta)=\theta^S(1-\theta)^{n-S},
\qquad
\ell(\theta)=S\log\theta+(n-S)\log(1-\theta).
$$

只要 $0<S<n$，一阶导数为

$$
\ell'(\theta)
=\frac{S}{\theta}-\frac{n-S}{1-\theta}.
$$

令它为 0：

$$
S(1-\theta)=(n-S)\theta
\quad\Longrightarrow\quad
\boxed{\hat\theta_{\mathrm{MLE}}=\frac Sn}.
$$

二阶导数

$$
\ell''(\theta)
=-\frac{S}{\theta^2}-\frac{n-S}{(1-\theta)^2}<0
$$

说明开区间内的驻点是唯一全局最大值。若 $S=0$，似然随 $\theta$ 增大而下降，最大值在边界 $\theta=0$；若 $S=n$，最大值在 $\theta=1$。边界解不是代码错误，但“观察为零”绝不等于“未来事件不可能”。

```python
def bernoulli_mle(successes, trials):
    if not (isinstance(successes, (int, np.integer)) and isinstance(trials, (int, np.integer))):
        raise TypeError("successes 和 trials 必须是整数")
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("必须满足 0 <= successes <= trials 且 trials > 0")
    return successes / trials

example_successes, example_trials = 3, 60
closed_form_mle = bernoulli_mle(example_successes, example_trials)
theta_grid = np.linspace(0.0001, 0.9999, 20_001)
grid_log_likelihood = bernoulli_log_likelihood(
    example_successes, example_trials, theta_grid
)
grid_mle = theta_grid[np.argmax(grid_log_likelihood)]

assert abs(grid_mle - closed_form_mle) <= theta_grid[1] - theta_grid[0]
assert bernoulli_mle(0, 36) == 0.0
assert bernoulli_mle(36, 36) == 1.0
print("闭式解 / 网格解：", round(closed_form_mle, 5), round(float(grid_mle), 5))
```

### 3.1 $S$ 是充分统计量

似然只通过 $(S,n)$ 依赖原始序列。只要审核数和漏检数相同，漏检出现在第 1 件还是第 36 件，对这个 iid Bernoulli 参数模型没有区别。于是 $(S,n)$ 是推断 $\theta$ 的充分统计量。

这不意味着可以删除原始记录。`challenge_id`、缺陷族、顺序、工位状态和重复实体是检验模型假设所必需的审计信息；“对参数估计充分”不等于“对数据质量审计充分”。

```python
audit_sequence = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=int)
permuted_sequence = audit_sequence[[2, 5, 1, 9, 0, 7, 3, 8, 6, 4]]
theta_check = 0.08

sequence_log_likelihood = np.sum(
    audit_sequence * np.log(theta_check)
    + (1 - audit_sequence) * np.log1p(-theta_check)
)
summary_log_likelihood = bernoulli_log_likelihood(
    int(audit_sequence.sum()), len(audit_sequence), theta_check
)
permuted_log_likelihood = np.sum(
    permuted_sequence * np.log(theta_check)
    + (1 - permuted_sequence) * np.log1p(-theta_check)
)

assert np.isclose(sequence_log_likelihood, summary_log_likelihood)
assert np.isclose(sequence_log_likelihood, permuted_log_likelihood)
print("原序列 / 计数 / 重排后的 log L：", round(float(sequence_log_likelihood), 6))
```

### 3.2 曲率给出局部辨识强度

在内点 MLE 处，观测负曲率为

$$
-\ell''(\hat\theta)
=\frac{n}{\hat\theta(1-\hat\theta)}.
$$

同一漏检比例下，$n$ 越大，峰越尖；局部二次近似给出标准误

$$
\operatorname{se}(\hat\theta)
\approx\sqrt{\frac{\hat\theta(1-\hat\theta)}{n}}.
$$

这个近似依赖内点、独立性与大样本。$S=0$ 时把 $\hat\theta=0$ 代入会得到标准误 0，这是边界下失效的近似，不是“百分之百确定”。

```python
def bernoulli_score(successes, trials, theta):
    return successes / theta - (trials - successes) / (1 - theta)

def bernoulli_hessian(successes, trials, theta):
    return -successes / theta ** 2 - (trials - successes) / (1 - theta) ** 2

theta0 = closed_form_mle
step = 1e-5
numeric_score_derivative = (
    bernoulli_score(example_successes, example_trials, theta0 + step)
    - bernoulli_score(example_successes, example_trials, theta0 - step)
) / (2 * step)
analytic_hessian = bernoulli_hessian(example_successes, example_trials, theta0)

assert abs(bernoulli_score(example_successes, example_trials, theta0)) < 1e-12
assert np.isclose(numeric_score_derivative, analytic_hessian, rtol=2e-7)
print("解析 / 数值 Hessian：", round(analytic_hessian, 4), round(numeric_score_derivative, 4))
```

::: ml-stop 停下来推一推 1
工位 A 审核 20 件漏检 1 件，工位 B 审核 200 件漏检 10 件。两者 MLE 都是 5%。分别写出峰顶负曲率，说明为什么“点估计相同”不代表证据强度相同；再解释若 B 的 200 件其实来自 10 个缺陷实体各重放 20 次，公式中的 $n=200$ 为什么可疑。
:::

<MlQuiz kind="结果预测" question="36 个挑战件全部检出、漏检数 S=0 时，Bernoulli MLE 与‘下一件绝不漏检’之间是什么关系？" :options="['MLE=0，因此未来漏检概率已被证明为 0', 'MLE 在边界 0，但有限样本不能排除正漏检率', 'MLE 不存在', 'MLE 必须改成 1/36']" :answer="1" explanation="似然在可行域边界取最大值，所以点估计是 0；但参数不确定性仍存在，边界正态近似也失效。后续应看区间、先验敏感性和审核设计。" />

## 4. Categorical MLE：互斥结果如何分配概率质量

审核动作可能有 $K$ 个互斥类别，例如自动拒绝、人工复核、漏检。设 $Y_i\in\{1,\ldots,K\}$，类别概率向量

$$
\boldsymbol\theta=(\theta_1,\ldots,\theta_K),
\qquad \theta_k\ge 0,
\qquad \sum_{k=1}^K\theta_k=1.
$$

令 $N_k=\sum_i\mathbb I(Y_i=k)$，则

$$
\ell(\boldsymbol\theta)=\sum_{k=1}^K N_k\log\theta_k.
$$

用 Lagrange 乘子处理和为 1 的约束：

$$
\mathcal J(\boldsymbol\theta,\lambda)
=\sum_kN_k\log\theta_k
+\lambda\left(1-\sum_k\theta_k\right).
$$

驻点条件

$$
\frac{N_k}{\theta_k}-\lambda=0
\quad\Longrightarrow\quad
\theta_k=\frac{N_k}{\lambda}.
$$

求和并使用约束得到 $\lambda=\sum_kN_k=n$，所以

$$
\boxed{\hat\theta_k=\frac{N_k}{n}}.
$$

若某类从未出现，MLE 给它概率 0；未来一旦出现该类，插件模型的对数损失为无穷。这与 Bernoulli 零计数是同一个边界问题。

```python
def categorical_mle(counts):
    counts = np.asarray(counts)
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError("counts 必须是一维且至少包含两个类别")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("类别计数必须是非负整数")
    total = int(counts.sum())
    if total == 0:
        raise ValueError("总计数必须大于 0")
    return counts / total

review_counts = np.array([90, 21, 9, 0])
review_mle = categorical_mle(review_counts)

assert np.isclose(review_mle.sum(), 1.0)
assert np.all(review_mle >= 0)
assert review_mle[-1] == 0.0
assert np.allclose(review_mle[:3], [0.75, 0.175, 0.075])
print("四类动作 MLE：", np.round(review_mle, 4).tolist())
```

<MlQuiz kind="推导补全" question="Categorical MLE 的 Lagrange 驻点给出 theta_k=N_k/lambda。代入 sum theta_k=1 后，lambda 等于什么？" :options="['类别数 K', '总样本数 n=sum N_k', '最大类别计数', '1/n']" :answer="1" explanation="求和得到 1=(sum N_k)/lambda=n/lambda，所以 lambda=n，最终 theta_k=N_k/n。" />

## 5. Gaussian MLE：均值、方差与退化解

标准靶标的焦点评分残差可先用一维 Gaussian 建模：

$$
R_i\overset{\mathrm{iid}}\sim\mathcal N(\mu,\sigma^2).
$$

对数似然为

$$
\ell(\mu,\sigma^2)
=-\frac n2\log(2\pi)
-\frac n2\log\sigma^2
-\frac{1}{2\sigma^2}\sum_{i=1}^n(r_i-\mu)^2.
$$

固定 $\sigma^2$，对 $\mu$ 求导：

$$
\frac{\partial\ell}{\partial\mu}
=\frac1{\sigma^2}\sum_i(r_i-\mu)=0
\quad\Longrightarrow\quad
\boxed{\hat\mu_{\mathrm{MLE}}=\bar r}.
$$

令 $v=\sigma^2$，再对 $v$ 求导：

$$
\frac{\partial\ell}{\partial v}
=-\frac n{2v}
+\frac1{2v^2}\sum_i(r_i-\mu)^2=0,
$$

所以

$$
\boxed{
\hat\sigma^2_{\mathrm{MLE}}
=\frac1n\sum_i(r_i-\bar r)^2
}.
$$

分母是 $n$，而不是无偏样本方差使用的 $n-1$。MLE 最大化观测数据的似然，不自动保证估计量无偏。若所有残差完全相同，则令 $\mu$ 等于该值并让 $\sigma^2\to0$，密度可以无限增大；可行域若要求 $\sigma^2>0$，就没有有限最大值。工程实现必须识别这种退化，而不是返回“零噪声、百分百精确”。

```python
score_rng = np.random.default_rng(2206)
focus_residual = score_rng.normal(0.12, 0.42, size=48)
gaussian_mean_mle = float(np.mean(focus_residual))
gaussian_variance_mle = float(np.mean((focus_residual - gaussian_mean_mle) ** 2))
gaussian_variance_unbiased = float(np.var(focus_residual, ddof=1))

assert np.isclose(
    gaussian_variance_unbiased,
    gaussian_variance_mle * len(focus_residual) / (len(focus_residual) - 1),
)
assert gaussian_variance_mle > 0
print("均值 MLE：", round(gaussian_mean_mle, 4))
print("方差 MLE / 无偏方差：", round(gaussian_variance_mle, 4), round(gaussian_variance_unbiased, 4))
```

### 5.1 用 $\rho=\log\sigma^2$ 保证方差为正

通用优化器不应在每一步直接猜一个可能为负的方差。令 $\rho=\log\sigma^2$，则 $\sigma^2=e^\rho>0$。在 $(\mu,\rho)$ 参数化下，梯度为

$$
\frac{\partial\ell}{\partial\mu}
=e^{-\rho}\sum_i(r_i-\mu),
\qquad
\frac{\partial\ell}{\partial\rho}
=-\frac n2+\frac{e^{-\rho}}2\sum_i(r_i-\mu)^2.
$$

```python
def gaussian_log_likelihood(parameters, values):
    mu, log_variance = np.asarray(parameters, dtype=float)
    variance = np.exp(log_variance)
    residual = np.asarray(values, dtype=float) - mu
    return -0.5 * (
        len(residual) * (math.log(2 * math.pi) + log_variance)
        + np.sum(residual ** 2) / variance
    )

def gaussian_log_likelihood_gradient(parameters, values):
    mu, log_variance = np.asarray(parameters, dtype=float)
    variance = np.exp(log_variance)
    residual = np.asarray(values, dtype=float) - mu
    return np.array([
        np.sum(residual) / variance,
        -0.5 * len(residual) + 0.5 * np.sum(residual ** 2) / variance,
    ])

gaussian_parameters = np.array([0.05, math.log(0.25)])
finite_step = 1e-6
numeric_gradient = np.empty(2)
for parameter_index in range(2):
    direction = np.zeros(2)
    direction[parameter_index] = finite_step
    numeric_gradient[parameter_index] = (
        gaussian_log_likelihood(gaussian_parameters + direction, focus_residual)
        - gaussian_log_likelihood(gaussian_parameters - direction, focus_residual)
    ) / (2 * finite_step)

analytic_gradient = gaussian_log_likelihood_gradient(gaussian_parameters, focus_residual)
assert np.allclose(analytic_gradient, numeric_gradient, rtol=2e-7, atol=2e-7)
print("Gaussian 梯度最大误差：", float(np.max(np.abs(analytic_gradient - numeric_gradient))))
```

<MlQuiz kind="错误诊断" question="只有一个焦点残差样本时，同时估计 Gaussian 均值和方差，代码返回 mu=x、sigma²=0 并称为高置信度解。主要问题是什么？" :options="['Gaussian 均值不能用 MLE', '似然沿 sigma²→0 退化，没有有限正方差最大值', '方差必须固定为 1', '应把分母改成 n+1']" :answer="1" explanation="均值贴住唯一观测后，方差越小密度越高；这不是高置信度，而是不可识别/退化。需要更多独立重复、方差下界、结构共享或先验。" />

## 6. 充分统计量与指数族：为什么很多 MLE 是“经验矩匹配”

若密度可写成

$$
p(x\mid\boldsymbol\eta)
=h(x)\exp\left[
\boldsymbol\eta^\top T(x)-A(\boldsymbol\eta)
\right],
$$

就属于指数族。$\boldsymbol\eta$ 是自然参数，$T(x)$ 是充分统计量，$A(\boldsymbol\eta)$ 是对数配分函数。iid 样本的对数似然为

$$
\ell(\boldsymbol\eta)
=\boldsymbol\eta^\top\sum_{i=1}^nT(x_i)
-nA(\boldsymbol\eta)
+\sum_i\log h(x_i).
$$

对自然参数求梯度：

$$
\nabla_{\boldsymbol\eta}\ell
=\sum_iT(x_i)-n\nabla A(\boldsymbol\eta).
$$

而指数族恒有

$$
\nabla A(\boldsymbol\eta)
=\mathbb E_{\boldsymbol\eta}[T(X)].
$$

于是内点 MLE 满足

$$
\boxed{
\mathbb E_{\hat{\boldsymbol\eta}}[T(X)]
=\frac1n\sum_iT(x_i)
},
$$

即模型充分统计量的期望匹配经验充分统计量。

| 分布 | 充分统计量 | 经验匹配结果 | 边界风险 |
|---|---|---|---|
| Bernoulli | $x$ | $E[X]=S/n$ | 全 0/全 1 时自然参数趋于无穷 |
| Categorical | one-hot 向量 | 每类概率等于频率 | 未见类别概率为 0 |
| Gaussian，方差已知 | $x$ | 均值等于样本均值 | 相关重复会夸大信息量 |
| Gaussian，均值方差未知 | $(x,x^2)$ | 一阶、二阶经验矩匹配 | 完全相同样本导致方差退化 |

Bernoulli 的自然参数是 log odds $\eta=\log[\theta/(1-\theta)]$，$A(\eta)=\log(1+e^\eta)$。分数方程 $S-n\sigma(\eta)=0$ 再次给出 $\sigma(\hat\eta)=S/n$。

```python
def stable_sigmoid(value):
    value = np.asarray(value, dtype=float)
    return np.where(
        value >= 0,
        1 / (1 + np.exp(-value)),
        np.exp(value) / (1 + np.exp(value)),
    )

moment_rate = 7 / 80
natural_parameter = math.log(moment_rate / (1 - moment_rate))
natural_score = 7 - 80 * float(stable_sigmoid(natural_parameter))

assert abs(natural_score) < 1e-12
assert np.isclose(float(stable_sigmoid(natural_parameter)), moment_rate)
print("Bernoulli 自然参数 / 匹配均值：", round(natural_parameter, 4), round(moment_rate, 4))
```

“有充分统计量”不等于“模型一定正确”。它只说明在已经选定的概率族和条件独立假设下，原始数据关于参数的信息可被压缩；模型外的相关性、漂移和选择机制仍然可能决定成败。

## 7. MLE 的统计性质、存在性与错设边界

在可辨识、真分布属于模型族、参数位于内点且满足一组正则条件时，MLE 常具有：

1. **一致性**：$n\to\infty$ 时趋向真实参数；
2. **渐近正态性**：$\sqrt n(\hat\theta-\theta^*)$ 近似正态；
3. **渐近有效性**：在正则估计量中达到 Fisher 信息界；
4. **一一变换等变性**：若 $\phi=g(\theta)$，则 $\hat\phi_{\rm MLE}=g(\hat\theta_{\rm MLE})$。

这些结论都不是“小样本无条件保证”。本章已经见到三类反例：Bernoulli 边界解、Gaussian 方差退化、重复挑战件破坏独立性。若真分布不在模型族中，MLE 通常趋向最小化

$$
D_{\mathrm{KL}}(p^*\Vert p_\theta)
$$

的伪真参数，而不是真实世界中并不存在的“正确参数”。

### 7.1 Fisher 信息与重复抽样

对单个 Bernoulli 观测，Fisher 信息为

$$
\mathcal I_1(\theta)
=-\mathbb E\left[
\frac{\partial^2}{\partial\theta^2}
\log p(X\mid\theta)
\right]
=\frac1{\theta(1-\theta)}.
$$

$n$ 个独立观测的信息相加，因此 MLE 渐近方差为

$$
\operatorname{Var}(\hat\theta_{\rm MLE})
\approx\frac{1}{n\mathcal I_1(\theta)}
=\frac{\theta(1-\theta)}n.
$$

Bernoulli 样本比例实际上在有限样本下就严格无偏，方差也恰好等于上式。

```python
sampling_rng = np.random.default_rng(2207)
sampling_trials = 200_000
sampling_theta = 0.06
sampling_n = 24
sampling_mle = sampling_rng.binomial(
    sampling_n, sampling_theta, size=sampling_trials
) / sampling_n

theoretical_mean = sampling_theta
theoretical_variance = sampling_theta * (1 - sampling_theta) / sampling_n
simulation_mean = float(np.mean(sampling_mle))
simulation_variance = float(np.var(sampling_mle))

assert abs(simulation_mean - theoretical_mean) < 5e-4
assert abs(simulation_variance - theoretical_variance) < 2e-5
print("模拟 / 理论均值：", round(simulation_mean, 6), theoretical_mean)
print("模拟 / 理论方差：", round(simulation_variance, 6), round(theoretical_variance, 6))
```

::: ml-stop 停下来推一推 2
证明一一变换下的 MLE 等变性：若 $g$ 可逆，说明最大化 $L(\theta)$ 与在 $\phi=g(\theta)$ 中最大化 $L(g^{-1}(\phi))$ 为什么选择对应点。然后先不要把这个结论套给 MAP；等到第 13 节，检查参数密度变换中的 Jacobian 为什么会改变众数。
:::

## 8. MAP：让冻结在数据之前的先验参与点估计

Bayes 规则为

$$
p(\theta\mid\mathcal D)
=\frac{p(\mathcal D\mid\theta)p(\theta)}{p(\mathcal D)}.
$$

最大后验估计（maximum a posteriori estimation, MAP）选择后验密度的众数：

$$
\begin{aligned}
\hat\theta_{\mathrm{MAP}}
&=\arg\max_\theta p(\theta\mid\mathcal D)\\
&=\arg\max_\theta
\left[
\log p(\mathcal D\mid\theta)+\log p(\theta)
\right],
\end{aligned}
$$

因为证据 $p(\mathcal D)$ 与 $\theta$ 无关。MAP 仍然只返回一个点，不保留后验宽度、偏斜、多峰或尾部概率。

### 8.1 Gaussian、Laplace 先验与正则化

若回归噪声为

$$
\mathbf y\mid\mathbf w
\sim\mathcal N(\mathbf X\mathbf w,\sigma^2\mathbf I),
$$

并给权重先验

$$
\mathbf w\sim\mathcal N(\mathbf 0,\tau^2\mathbf I),
$$

则忽略常数后的负对数后验为

$$
\frac1{2\sigma^2}\|\mathbf y-\mathbf X\mathbf w\|_2^2
+\frac1{2\tau^2}\|\mathbf w\|_2^2.
$$

乘以 $2\sigma^2$ 不改变最优点：

$$
\hat{\mathbf w}_{\rm MAP}
=\arg\min_{\mathbf w}
\|\mathbf y-\mathbf X\mathbf w\|_2^2
+\underbrace{\frac{\sigma^2}{\tau^2}}_{\lambda}
\|\mathbf w\|_2^2.
$$

所以零均值 Gaussian 先验对应 $L_2$；独立 Laplace 先验 $p(w_j)\propto e^{-|w_j|/b}$ 对应 $L_1$。对应关系包含噪声尺度、先验尺度，以及损失使用“总和”还是“平均”的约定。只说“Gaussian 先验就是 weight decay 0.01”是不完整的。

```python
map_rng = np.random.default_rng(2208)
map_design = map_rng.normal(size=(80, 3))
map_true_weights = np.array([1.2, -0.7, 0.25])
map_noise_sigma = 0.4
map_prior_tau = 1.2
map_response = (
    map_design @ map_true_weights
    + map_rng.normal(0.0, map_noise_sigma, size=80)
)
map_penalty = map_noise_sigma ** 2 / map_prior_tau ** 2
map_closed_form = np.linalg.solve(
    map_design.T @ map_design + map_penalty * np.eye(map_design.shape[1]),
    map_design.T @ map_response,
)
map_gradient = (
    map_design.T @ (map_design @ map_closed_form - map_response)
    + map_penalty * map_closed_form
)

assert np.linalg.norm(map_gradient) < 1e-10
assert np.all(np.isfinite(map_closed_form))
print("Gaussian-prior MAP 权重：", np.round(map_closed_form, 4).tolist())
print("驻点梯度范数：", float(np.linalg.norm(map_gradient)))
```

<MlQuiz kind="设计判断" question="两个团队都声称使用 Gaussian 先验和 L2 正则，但一个最小化总 NLL，另一个最小化平均 NLL，却直接复制同一个 lambda。问题在哪里？" :options="['没有问题，lambda 与样本数永远无关', '目标缩放不同；同一先验强度对应的正则系数要随口径换算', 'Gaussian 先验只能用于分类', 'MAP 不允许优化']" :answer="1" explanation="总似然随 n 累加，平均 NLL 除以 n；若先验项不同时缩放，同一个数值 lambda 表示不同先验强度。模型制品必须记录完整目标口径。" />

## 9. Beta–Bernoulli：共轭更新、MAP 与后验均值

对漏检率使用 Beta 先验：

$$
\theta\sim\operatorname{Beta}(\alpha,\beta),
\qquad
p(\theta)\propto
\theta^{\alpha-1}(1-\theta)^{\beta-1}.
$$

观察 $S$ 次漏检和 $n-S$ 次检出后：

$$
\begin{aligned}
p(\theta\mid\mathcal D)
&\propto
\theta^S(1-\theta)^{n-S}
\theta^{\alpha-1}(1-\theta)^{\beta-1}\\
&=\theta^{\alpha+S-1}
(1-\theta)^{\beta+n-S-1}.
\end{aligned}
$$

因此

$$
\boxed{
\theta\mid\mathcal D
\sim\operatorname{Beta}(\alpha+S,\beta+n-S)
}.
$$

先验和后验属于同一分布族，所以 Beta 是 Bernoulli/Binomial 的共轭先验。它让批量更新和流式更新都只需维护两个计数参数。

### 9.1 三个常被混在一起的点估计

记 $\alpha'=\alpha+S,\beta'=\beta+n-S$。若 $\alpha',\beta'>1$，后验众数即 MAP：

$$
\hat\theta_{\rm MAP}
=\frac{\alpha'-1}{\alpha'+\beta'-2}
=\frac{S+\alpha-1}{n+\alpha+\beta-2}.
$$

后验均值和方差是

$$
\mathbb E[\theta\mid\mathcal D]
=\frac{\alpha'}{\alpha'+\beta'},
$$

$$
\operatorname{Var}(\theta\mid\mathcal D)
=\frac{\alpha'\beta'}
{(\alpha'+\beta')^2(\alpha'+\beta'+1)}.
$$

对下一件挑战件，完整 Bayes 后验预测漏检概率恰好等于后验均值：

$$
P(X_{n+1}=1\mid\mathcal D)
=\int\theta\,p(\theta\mid\mathcal D)d\theta
=\frac{\alpha'}{\alpha'+\beta'}.
$$

但 MAP、后验均值和后验预测并非一般情况下都相同。尤其要记住：

- `Beta(1,1)` 先验下，MAP 等于 MLE；
- `Beta(1,1)` 先验下，后验均值为 $(S+1)/(n+2)$；
- `Beta(2,2)` 先验下，MAP 也为 $(S+1)/(n+2)$。

所以“add-one smoothing”必须说明是哪个先验配哪个估计量。

```python
def beta_posterior_parameters(successes, trials, alpha, beta):
    bernoulli_mle(successes, trials)  # 复用边界校验
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta 参数必须严格为正")
    return alpha + successes, beta + trials - successes

def beta_mode(alpha, beta):
    if alpha > 1 and beta > 1:
        return (alpha - 1) / (alpha + beta - 2)
    if alpha < 1 and beta < 1:
        return (0.0, 1.0)  # 两个边界都是无限高的众数
    if alpha <= 1 and beta > 1:
        return 0.0
    if alpha > 1 and beta <= 1:
        return 1.0
    return None  # Beta(1,1) 平坦，所有点都是众数

zero_successes, small_trials = 0, 5
uniform_posterior = beta_posterior_parameters(
    zero_successes, small_trials, 1.0, 1.0
)
beta22_posterior = beta_posterior_parameters(
    zero_successes, small_trials, 2.0, 2.0
)
uniform_posterior_mean = uniform_posterior[0] / sum(uniform_posterior)
beta22_map = beta_mode(*beta22_posterior)
uniform_map = beta_mode(*uniform_posterior)

assert np.isclose(uniform_posterior_mean, 1 / 7)
assert np.isclose(beta22_map, 1 / 7)
assert uniform_map == 0.0
print("Uniform 后验均值 / Beta(2,2) MAP / Uniform MAP：", uniform_posterior_mean, beta22_map, uniform_map)
```

### 9.2 用先验众数和 MAP 等效票数解释收缩

为直观展示 MAP，令冻结先验的众数为 $m_0\in(0,1)$，定义

$$
c=\alpha+\beta-2,
\qquad
\alpha=1+cm_0,
\qquad
\beta=1+c(1-m_0).
$$

则

$$
\hat\theta_{\rm MAP}
=\frac{S+cm_0}{n+c}
=\frac{n}{n+c}\hat\theta_{\rm MLE}
+\frac{c}{n+c}m_0.
$$

$c$ 在这里是**MAP 众数公式中的等效票数**，不是 Beta 分布通常定义的总浓度 $\alpha+\beta$；两者相差 2。不同参数化必须写清。

```python
prior_mode = 0.035
map_strength = 36.0
alpha0 = 1.0 + map_strength * prior_mode
beta0 = 1.0 + map_strength * (1.0 - prior_mode)

posterior_alpha = alpha0 + miss_count
posterior_beta = beta0 + audit_size - miss_count
station_map = (posterior_alpha - 1) / (posterior_alpha + posterior_beta - 2)
station_posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

map_as_weighted_average = (
    audit_size * station_mle + map_strength * prior_mode
) / (audit_size + map_strength)

assert np.allclose(station_map, map_as_weighted_average)
assert np.all((0 < station_map) & (station_map < 1))
assert np.all((0 < station_posterior_mean) & (station_posterior_mean < 1))
print("前 4 个工位 MLE：", np.round(station_mle[:4], 4).tolist())
print("前 4 个工位 MAP：", np.round(station_map[:4], 4).tolist())
print("前 4 个工位后验均值：", np.round(station_posterior_mean[:4], 4).tolist())
```

### 9.3 后验区间与超过资格线的概率

生产动作不应只读取一个点。下面在稳定 log density 下做确定性网格积分，计算 95% 等尾可信区间和 $P(\theta>0.05\mid D)$。网格法只用于本章的一维可视化；高维模型不能靠均匀网格扩展。

```python
def beta_grid_summary(alpha, beta, threshold=0.05, grid_size=20_001):
    if alpha <= 0 or beta <= 0 or not (0 < threshold < 1):
        raise ValueError("Beta 参数必须为正，threshold 必须在 (0,1)")
    if grid_size < 1001:
        raise ValueError("grid_size 太小，不能用于本章的区间近似")
    grid = np.linspace(1e-6, 1 - 1e-6, grid_size)
    log_density = (
        (alpha - 1) * np.log(grid)
        + (beta - 1) * np.log1p(-grid)
        - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
    )
    mass = np.exp(log_density - np.max(log_density))
    mass /= mass.sum()
    cumulative = np.cumsum(mass)
    lower = grid[np.searchsorted(cumulative, 0.025)]
    upper = grid[np.searchsorted(cumulative, 0.975)]
    tail = mass[grid > threshold].sum()
    return lower, upper, tail

station_posterior_summary = np.array([
    beta_grid_summary(alpha, beta)
    for alpha, beta in zip(posterior_alpha, posterior_beta)
])
tail_probability = station_posterior_summary[:, 2]
expanded_audit_stations = station_id[tail_probability > 0.8]

assert expanded_audit_stations.tolist() == [6, 11]
assert np.all(station_posterior_summary[:, 0] < station_posterior_summary[:, 1])
assert np.all((0 <= tail_probability) & (tail_probability <= 1))
print("进入扩大审核的工位：", expanded_audit_stations.tolist())
print("对应 P(theta>5%|D)：", np.round(tail_probability[expanded_audit_stations], 3).tolist())
```

<BetaEvidenceMixerLab />

<MlQuiz kind="结果预测" question="S=0、n=5 时，哪一组陈述正确？" :options="['Uniform Beta(1,1) 的 MAP 与后验均值都为 0', 'Uniform 后验均值为 1/7；Beta(2,2) 的 MAP 也为 1/7', 'Beta(2,2) 的 MAP 为 0', '所有 Beta 先验的 MAP 都相同']" :answer="1" explanation="Uniform 先验的后验是 Beta(1,6)，其均值为 1/7、众数在 0；Beta(2,2) 更新为 Beta(2,7)，众数为 1/7。" />

## 10. 共轭更新不只用于二分类

共轭的价值不是“Bayes 一定要选能手算的先验”，而是提供一个完全透明的参照系：数据如何修改参数、顺序更新为何等价、哪些量被保留下来，都能逐项核对。复杂模型没有共轭闭式解时，仍然要遵守同一概率合同，只是改用优化、采样或变分近似。

### 10.1 Beta 更新的批量—流式等价

对每条 Bernoulli 结果 $x_t$，递推

$$
\alpha_t=\alpha_{t-1}+x_t,
\qquad
\beta_t=\beta_{t-1}+1-x_t.
$$

加法可交换，所以逐条更新、按日分批更新和一次性汇总，只要没有丢记录或重复计算，最终后验完全相同。

```python
stream_results = np.array([
    0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
])
sequential_alpha, sequential_beta = alpha0, beta0
for result in stream_results:
    sequential_alpha += int(result)
    sequential_beta += int(1 - result)

batch_alpha, batch_beta = beta_posterior_parameters(
    int(stream_results.sum()), len(stream_results), alpha0, beta0
)
reversed_alpha, reversed_beta = alpha0, beta0
for result in stream_results[::-1]:
    reversed_alpha += int(result)
    reversed_beta += int(1 - result)

assert np.isclose(sequential_alpha, batch_alpha)
assert np.isclose(sequential_beta, batch_beta)
assert np.isclose(reversed_alpha, batch_alpha)
assert np.isclose(reversed_beta, batch_beta)
print("流式 / 批量后验参数：", (sequential_alpha, sequential_beta), (batch_alpha, batch_beta))
```

工程上“顺序不影响后验”不等于顺序字段可以删除。若时间顺序暴露出光源衰减、热漂移或连续失效，静态 $\theta$ 模型已经被证伪，应改成时间分段或动态模型。

### 10.2 Dirichlet–Categorical：多类伪计数

对 $K$ 类概率向量使用 Dirichlet 先验：

$$
p(\boldsymbol\theta)
\propto\prod_{k=1}^K\theta_k^{\alpha_k-1}.
$$

观察类别计数 $N_k$ 后：

$$
\boldsymbol\theta\mid\mathcal D
\sim\operatorname{Dirichlet}
(\alpha_1+N_1,\ldots,\alpha_K+N_K).
$$

后验均值和内点 MAP 分别为

$$
E[\theta_k\mid D]
=\frac{N_k+\alpha_k}{n+\sum_j\alpha_j},
$$

$$
\hat\theta_{k,\rm MAP}
=\frac{N_k+\alpha_k-1}
{n+\sum_j\alpha_j-K}.
$$

第二式要求所有后验参数大于 1；否则众数可能在单纯形边界。Uniform `Dirichlet(1,…,1)` 的后验均值与 `Dirichlet(2,…,2)` 的 MAP 都产生 add-one 平滑，再次说明“平滑公式”不能脱离估计量来命名。

```python
category_count = review_counts.copy()
category_count_total = int(category_count.sum())
uniform_dirichlet = np.ones_like(category_count, dtype=float)
dirichlet2 = np.full_like(category_count, 2.0, dtype=float)

uniform_posterior_parameters = uniform_dirichlet + category_count
uniform_posterior_predictive = (
    uniform_posterior_parameters / uniform_posterior_parameters.sum()
)
dirichlet2_posterior_parameters = dirichlet2 + category_count
dirichlet2_map = (
    dirichlet2_posterior_parameters - 1
) / np.sum(dirichlet2_posterior_parameters - 1)

assert np.allclose(uniform_posterior_predictive, dirichlet2_map)
assert uniform_posterior_predictive[-1] > 0
assert review_mle[-1] == 0
assert np.isclose(uniform_posterior_predictive.sum(), 1.0)
print("未见类别：MLE / Uniform 后验预测：", review_mle[-1], round(float(uniform_posterior_predictive[-1]), 5))
```

### 10.3 Gaussian–Gaussian：按精度加权

若观测方差 $\sigma^2$ 已知，

$$
r_i\mid\mu\sim\mathcal N(\mu,\sigma^2),
\qquad
\mu\sim\mathcal N(m_0,\tau_0^2),
$$

则后验仍是 Gaussian：

$$
\tau_n^{-2}=\tau_0^{-2}+n\sigma^{-2},
$$

$$
m_n
=\tau_n^2\left(
\tau_0^{-2}m_0+n\sigma^{-2}\bar r
\right).
$$

均值是按精度（方差倒数）加权，而不是机械地把“历史均值”和“本批均值”各占一半。Gaussian 后验对称，所以后验均值和 MAP 相同；这是分布形状带来的特例，不是一般规律。

```python
def gaussian_mean_posterior(values, observation_variance, prior_mean, prior_variance):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values 必须是一维非空数组")
    if observation_variance <= 0 or prior_variance <= 0:
        raise ValueError("方差必须严格为正")
    posterior_precision = 1 / prior_variance + values.size / observation_variance
    posterior_variance = 1 / posterior_precision
    posterior_mean = posterior_variance * (
        prior_mean / prior_variance + values.sum() / observation_variance
    )
    return posterior_mean, posterior_variance

known_observation_variance = 0.42 ** 2
focus_prior_mean = 0.0
focus_prior_variance = 0.30 ** 2
focus_batch_mean, focus_batch_variance = gaussian_mean_posterior(
    focus_residual,
    known_observation_variance,
    focus_prior_mean,
    focus_prior_variance,
)

focus_stream_mean, focus_stream_variance = focus_prior_mean, focus_prior_variance
for residual_value in focus_residual:
    next_variance = 1 / (
        1 / focus_stream_variance + 1 / known_observation_variance
    )
    next_mean = next_variance * (
        focus_stream_mean / focus_stream_variance
        + residual_value / known_observation_variance
    )
    focus_stream_mean, focus_stream_variance = next_mean, next_variance

assert np.isclose(focus_stream_mean, focus_batch_mean)
assert np.isclose(focus_stream_variance, focus_batch_variance)
assert focus_batch_variance < focus_prior_variance
print("Gaussian 批量 / 流式后验均值：", round(focus_batch_mean, 6), round(focus_stream_mean, 6))
```

## 11. 重复抽样风险：收缩为什么可能更好，也可能更坏

点估计要在“重复做同样审核”这个外层随机实验中评价。设真实漏检率为 $\theta^*$，每次审核 $n$ 件，$S\sim\operatorname{Binomial}(n,\theta^*)$。

MLE

$$
\hat\theta_{\rm MLE}=S/n
$$

满足

$$
\operatorname{Bias}(\hat\theta_{\rm MLE})=0,
\qquad
\operatorname{Var}(\hat\theta_{\rm MLE})
=\frac{\theta^*(1-\theta^*)}{n}.
$$

对先验众数 $m_0$、MAP 等效票数 $c$，

$$
\hat\theta_{\rm MAP}
=\frac{S+cm_0}{n+c}.
$$

其重复抽样偏差和方差是

$$
\operatorname{Bias}(\hat\theta_{\rm MAP})
=\frac{c(m_0-\theta^*)}{n+c},
$$

$$
\operatorname{Var}(\hat\theta_{\rm MAP})
=\frac{n\theta^*(1-\theta^*)}{(n+c)^2}.
$$

平方误差风险分解为

$$
\operatorname{MSE}=\operatorname{Bias}^2+\operatorname{Variance}.
$$

先验越强，方差单调下降；若先验中心错了，偏差绝对值却会上升。没有“MAP 一定优于 MLE”或“无偏一定优于有偏”的无条件结论。

```python
def bernoulli_map_risk(theta, trials, mode, strength):
    if not (0 < theta < 1 and 0 <= mode <= 1 and trials > 0 and strength >= 0):
        raise ValueError("参数超出风险公式的可行域")
    denominator = trials + strength
    bias = strength * (mode - theta) / denominator
    variance = trials * theta * (1 - theta) / denominator ** 2
    return {
        "bias": bias,
        "variance": variance,
        "mse": bias ** 2 + variance,
    }

def exact_binomial_estimator_risk(theta, trials, mode, strength):
    counts = np.arange(trials + 1)
    probabilities = np.array([
        math.comb(trials, int(count))
        * theta ** count
        * (1 - theta) ** (trials - count)
        for count in counts
    ])
    estimates = (counts + strength * mode) / (trials + strength)
    mean = np.sum(probabilities * estimates)
    variance = np.sum(probabilities * (estimates - mean) ** 2)
    bias = mean - theta
    return {"bias": bias, "variance": variance, "mse": bias ** 2 + variance}

risk_theta, risk_n = 0.06, 24
mle_risk = bernoulli_map_risk(risk_theta, risk_n, 0.0, 0.0)
aligned_map_risk = bernoulli_map_risk(risk_theta, risk_n, 0.035, 120.0)
mismatched_map_risk = bernoulli_map_risk(risk_theta, risk_n, 0.12, 120.0)
exact_mismatched_risk = exact_binomial_estimator_risk(
    risk_theta, risk_n, 0.12, 120.0
)

assert np.allclose(
    list(mismatched_map_risk.values()),
    list(exact_mismatched_risk.values()),
    rtol=1e-12,
    atol=1e-12,
)
assert aligned_map_risk["mse"] < mle_risk["mse"]
assert mismatched_map_risk["mse"] > mle_risk["mse"]
print("MLE / 对齐 MAP / 错配 MAP 的 MSE：", round(mle_risk["mse"], 6), round(aligned_map_risk["mse"], 6), round(mismatched_map_risk["mse"], 6))
```

<MapSamplingRiskArrayLab />

## 12. 后验预测：不要把一个点塞回模型就称为完整 Bayes

插件预测先选一个点 $\hat\theta$，再使用 $p(y_*\mid\hat\theta)$。完整后验预测则对参数不确定性积分：

$$
p(y_*\mid\mathcal D)
=\int p(y_*\mid\theta)p(\theta\mid\mathcal D)d\theta.
$$

对单个 Bernoulli 未来事件，积分结果等于后验均值。对未来 $M$ 件中的漏检总数 $K$，结果不是简单的 Binomial 插件分布，而是 Beta–Binomial：

$$
p(K=k\mid\mathcal D)
=\binom Mk
\frac{B(k+\alpha',M-k+\beta')}{B(\alpha',\beta')}.
$$

它的均值仍为 $M\alpha'/(\alpha'+\beta')$，但方差为

$$
\operatorname{Var}(K\mid D)
=M\frac{\alpha'\beta'}{(\alpha'+\beta')^2}
\frac{\alpha'+\beta'+M}{\alpha'+\beta'+1},
$$

大于把后验均值当固定参数的 Binomial 方差。原因是未来事件共享同一个未知 $\theta$；边缘化后它们不再相互独立。

```python
def log_beta(alpha, beta):
    return math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)

def beta_binomial_pmf(count, future_trials, alpha, beta):
    if not (0 <= count <= future_trials and alpha > 0 and beta > 0):
        raise ValueError("Beta-Binomial 参数无效")
    log_combination = (
        math.lgamma(future_trials + 1)
        - math.lgamma(count + 1)
        - math.lgamma(future_trials - count + 1)
    )
    return math.exp(
        log_combination
        + log_beta(count + alpha, future_trials - count + beta)
        - log_beta(alpha, beta)
    )

future_trials = 20
risk_station = 6
risk_alpha = float(posterior_alpha[risk_station])
risk_beta = float(posterior_beta[risk_station])
future_pmf = np.array([
    beta_binomial_pmf(count, future_trials, risk_alpha, risk_beta)
    for count in range(future_trials + 1)
])
future_counts = np.arange(future_trials + 1)
predictive_mean = risk_alpha / (risk_alpha + risk_beta)
beta_binomial_variance = float(np.sum(
    future_pmf * (future_counts - future_trials * predictive_mean) ** 2
))
plugin_binomial_variance = future_trials * predictive_mean * (1 - predictive_mean)

assert np.isclose(future_pmf.sum(), 1.0, atol=1e-12)
assert np.isclose(np.sum(future_pmf * future_counts), future_trials * predictive_mean)
assert beta_binomial_variance > plugin_binomial_variance
print("未来 20 件的 Beta-Binomial / 插件 Binomial 方差：", round(beta_binomial_variance, 4), round(plugin_binomial_variance, 4))
```

::: ml-stop 停下来推一推 3
设后验只在 $\theta=0.01$ 和 $\theta=0.09$ 两点各有 0.5 权重。下一件漏检的后验预测概率是多少？若未来两件在给定 $\theta$ 后独立，先算“二者都漏检”的后验预测概率，再与把平均概率 0.05 代入得到的 $0.05^2$ 比较。说明参数不确定性为什么会让批量尾部更厚。
:::

<MlQuiz kind="设计判断" question="资格动作直接关心‘真实漏检率超过 5% 的可能性’，最直接匹配的问题量是什么？" :options="['只看 MLE 是否超过 5%', '只看 MAP 是否超过 5%', '计算 P(theta>5%|D)，并结合先验敏感性和动作代价', '把审核数改成 100 后固定通过']" :answer="2" explanation="点估计丢失宽度和偏斜；后验尾概率直接对应参数超过资格线的命题，但仍依赖模型、先验和审核设计，必须做敏感性与数据质量审计。" />

## 13. 通用优化、参数化与 MAP 的众数陷阱

闭式解只存在于少数简单模型。一般 MLE/MAP 需要：

1. 在系统边界验证数据形状、参数可行域和缺失机制；
2. 用 log probability 累计目标；
3. 对概率用 logit、对正尺度用 log/softplus、对协方差用 Cholesky 参数化；
4. 用解析梯度与有限差分或自动微分对照；
5. 区分局部驻点、边界最优、不可辨识平坦方向与真正收敛；
6. 保存初始值、优化器、容差、目标缩放和随机种子。

### 13.1 MLE 等变，MAP 众数却依赖参数化

MLE 比较的是同一个数据似然。若 $\phi=g(\theta)$ 是一一变换，最大似然点直接映射，所以 MLE 等变。

MAP 最大化的是**相对于某个参考测度的密度高度**。变量变换时必须乘 Jacobian：

$$
p_\phi(\phi\mid D)
=p_\theta(\theta\mid D)
\left|\frac{d\theta}{d\phi}\right|.
$$

取 $\theta\sim\operatorname{Beta}(a,b)$，其在 $\theta$ 坐标中的众数为 $(a-1)/(a+b-2)$。令 $\phi=\operatorname{logit}(\theta)$，则 $|d\theta/d\phi|=\theta(1-\theta)$；在 $\phi$ 坐标中最大化密度等价于最大化 $\theta^a(1-\theta)^b$，映回 $\theta$ 后的众数变成 $a/(a+b)$。两者通常不同。

```python
parameterization_alpha = 2.0
parameterization_beta = 5.0
parameterization_grid = np.linspace(1e-5, 1 - 1e-5, 200_001)
theta_log_density = (
    (parameterization_alpha - 1) * np.log(parameterization_grid)
    + (parameterization_beta - 1) * np.log1p(-parameterization_grid)
)
logit_log_density = (
    theta_log_density
    + np.log(parameterization_grid)
    + np.log1p(-parameterization_grid)
)
theta_coordinate_mode = parameterization_grid[np.argmax(theta_log_density)]
logit_coordinate_mode_mapped_back = parameterization_grid[np.argmax(logit_log_density)]

assert abs(theta_coordinate_mode - 0.2) < 1e-5
assert abs(logit_coordinate_mode_mapped_back - 2 / 7) < 1e-5
assert not np.isclose(theta_coordinate_mode, logit_coordinate_mode_mapped_back)
print("theta 密度众数 / logit 密度众数映回：", round(float(theta_coordinate_mode), 5), round(float(logit_coordinate_mode_mapped_back), 5))
```

因此 MAP 作为计算方便的点近似很有用，但不能被描述成“参数空间中坐标无关的最可能真值”。若动作对不确定性敏感，应尽量使用后验期望、分位数、尾概率或对决策损失积分。

## 14. 生产审计：先验、相关性与动作会改变数据

### 14.1 先验必须先于目标批次冻结

可接受的先验来源包括：同版本历史资格批次、经过迁移审计的相邻产线、物理量程或专家约束。制品至少记录：

- 来源工位、时间窗口、挑战集版本和排除规则；
- $\alpha,\beta$ 原值，以及先验均值、众数、浓度或等效票数的参数化口径；
- 为什么可迁移、在哪些配方/光学版本上不可迁移；
- 是否由数据估计超参数，开发集与目标审核集是否严格分离；
- 先验敏感性和错配压力测试。

若用目标批次先估 $m_0,c$，又把同一批数据乘进似然，相当于重复使用证据。Empirical Bayes 可以从许多历史工位估超参数，但目标工位的最终审核必须留出，或使用正确的联合层级模型传播超参数不确定性。

### 14.2 工位异质性不能靠一个合并参数消失

困难挑战占比、光学老化和审核量都影响观察。下面只做描述性切片；它不证明因果，也不取代分层模型。

```python
hard_station_mask = hard_mix >= np.median(hard_mix)
aged_station_mask = optics_age >= np.median(optics_age)

hard_observed_rate = miss_count[hard_station_mask].sum() / audit_size[hard_station_mask].sum()
easy_observed_rate = miss_count[~hard_station_mask].sum() / audit_size[~hard_station_mask].sum()
aged_observed_rate = miss_count[aged_station_mask].sum() / audit_size[aged_station_mask].sum()
fresh_observed_rate = miss_count[~aged_station_mask].sum() / audit_size[~aged_station_mask].sum()

assert hard_station_mask.sum() == 8
assert aged_station_mask.sum() == 8
assert all(0 <= value <= 1 for value in [
    hard_observed_rate, easy_observed_rate,
    aged_observed_rate, fresh_observed_rate,
])
print("困难/容易挑战配比工位观察率：", round(hard_observed_rate, 4), round(easy_observed_rate, 4))
print("较老/较新光学工位观察率：", round(aged_observed_rate, 4), round(fresh_observed_rate, 4))
```

若同一工位不同缺陷族有不同漏检率，应按预先规定的目标混合权重分层估计，而不是让审核样本的偶然配比定义总体参数。若挑战件在批次内相关，可使用 cluster bootstrap、Beta-Binomial 过度离散模型、随机效应或按独立实体聚合；不能只把标准误乘一个经验常数而不说明来源。

### 14.3 先验敏感性不是可选图表

对一个被标记为扩大审核的工位，同时改变冻结先验中心和强度，检查 MAP 与尾概率是否改变动作。如果结论只在一个极强先验下成立，应把它报告为“由先验驱动”，而不是伪装成数据结论。

```python
sensitivity_station = 6
sensitivity_rows = []
for candidate_mode in [0.02, 0.035, 0.08]:
    for candidate_strength in [0.0, 12.0, 36.0, 120.0]:
        candidate_alpha = 1 + candidate_strength * candidate_mode
        candidate_beta = 1 + candidate_strength * (1 - candidate_mode)
        candidate_posterior_alpha = candidate_alpha + miss_count[sensitivity_station]
        candidate_posterior_beta = (
            candidate_beta
            + audit_size[sensitivity_station]
            - miss_count[sensitivity_station]
        )
        candidate_map = (
            candidate_posterior_alpha - 1
        ) / (candidate_posterior_alpha + candidate_posterior_beta - 2)
        candidate_tail = beta_grid_summary(
            candidate_posterior_alpha,
            candidate_posterior_beta,
        )[2]
        sensitivity_rows.append((
            candidate_mode, candidate_strength, candidate_map, candidate_tail
        ))

sensitivity_rows = np.array(sensitivity_rows)
no_prior_row = sensitivity_rows[
    (sensitivity_rows[:, 0] == 0.035)
    & (sensitivity_rows[:, 1] == 0.0)
][0]
assert np.isclose(no_prior_row[2], station_mle[sensitivity_station])
assert np.all((0 <= sensitivity_rows[:, 3]) & (sensitivity_rows[:, 3] <= 1))
print("工位 6 的 MAP 范围：", round(float(sensitivity_rows[:, 2].min()), 4), round(float(sensitivity_rows[:, 2].max()), 4))
print("工位 6 的尾概率范围：", round(float(sensitivity_rows[:, 3].min()), 3), round(float(sensitivity_rows[:, 3].max()), 3))
```

### 14.4 选择性扩大审核会形成反馈回路

上线后若只对高尾概率工位追加离线确认，低风险工位的挑战覆盖和成熟标签会越来越少。以后用“有追加审核的数据”更新先验，会把模型过去的选择策略写进数据分布。最低缓解措施包括：

- 在每个风险层保留固定比例的随机审核；
- 记录工位被抽中、扩审和停止审核的概率与原因；
- 把例行资格、故障追查和模型触发的审核分开统计；
- 先冻结新先验版本，再用于未来批次，禁止回写重算历史动作；
- 监控未决标签、挑战集覆盖、重复实体和人工改判。

### 14.5 最低监控面板

| 监控层 | 至少记录什么 | 触发复核的信号 |
|---|---|---|
| 数据生成 | 工位版本、挑战族、实体键、审核量、重复率、缺失原因 | 挑战配比漂移、重复重放增加、标签未成熟 |
| 似然 | $S,n$、总/平均 NLL、边界解、观测曲率 | 零计数骤增、曲率异常尖锐、同批相关 |
| 先验 | 来源版本、$\alpha,\beta$、中心、强度、敏感性 | 先验贡献长期压过新数据、来源域失效 |
| 后验 | MAP、均值、区间、$P(\theta>\theta_{max})$ | 区间变宽、尾概率越线、不同先验翻转动作 |
| 动作 | 扩审率、复核产能、最终判定、随机审核覆盖 | 只剩高风险样本有标签、动作概率不再可追溯 |
| 系统 | 更新幂等键、批次/流式对账、浮点与版本 | 重复消费、漏消费、同一输入重放结果不同 |

## 复杂度与工程边界

### 15.1 时间、空间与资源瓶颈

| 任务 | 训练/更新时间 | 状态空间 | 主要瓶颈 |
|---|---:|---:|---|
| Bernoulli MLE | 扫描原始记录 $O(n)$；已有 $(S,n)$ 后 $O(1)$ | $O(1)$ | 独立实体计数与边界区间，不是算术 |
| Categorical MLE | $O(n+K)$ | $O(K)$ | 类别字典、零计数和未决类别 |
| 一维 Gaussian MLE | $O(n)$ | 流式 $O(1)$ | 退化方差、异常点、相关重复 |
| $d$ 维 Gaussian 均值/协方差 | $O(nd^2)$；求逆/分解约 $O(d^3)$ | $O(d^2)$ | $n\le d$ 时奇异、条件数与内存 |
| 共轭 Beta/Dirichlet 更新 | 每批 $O(K)$ | $O(K)$ | 幂等、重复消费和先验版本 |
| 一维后验网格 | $O(G)$ | $O(G)$ | 分辨率与尾部截断；不可扩展到高维 |
| 通用梯度 MLE/MAP | 每轮通常 $O(nd)$，共 $T$ 轮 | 取决于优化器 | 局部最优、约束参数化和停止判据 |

推理阶段若只发布点估计，读取概率是 $O(1)$；若每次都重新做后验网格则是设计错误。应预计算所需分位数或提供稳定的分布接口，并在新批次到达时做版本化更新。

### 15.2 适用、不可用与更简单基线

**适用：** 数据生成故事清楚、独立单位可定义、参数维度与样本量匹配、先验来源可审计、动作能明确对应点估计或后验量。

**不可直接用：** 挑战集选择未知、重复实体无键、漏检标签只在部分动作后可见、工位参数快速漂移、单一 Bernoulli 无法解释过度离散、Gaussian 残差明显多峰/重尾、先验来自目标批次的重复使用。

**更简单基线：** 报告 $(S,n)$、样本比例和精确/稳健区间；若动作只需要全线总体水平且工位确实可交换，先使用合并比例。只有当工位级动作、异质性和先验借力带来可验证价值时，才增加层级模型或复杂后验推断。

### 15.3 置信区间和可信区间不能互换解释

频率学派 95% 置信区间保证：若无限次重复抽样并每次按同一规则构造区间，约 95% 的区间覆盖固定真参数。Bayes 95% 可信区间在给定模型和先验下回答：后验对参数落在该区间的概率为 95%。两者可能数值接近，但概率对象和保证不同。报告必须写明构造方法，不能只显示“95% CI”。

## 数值验证：边界、等价与确定性门禁

本章的闭式解也需要测试。最低门禁包括：无效计数拒绝、概率/方差可行域、边界 MLE、充分统计量重排不变、解析梯度有限差分、共轭流式—批量等价、风险公式精确枚举对照、后验质量归一化和退化 Gaussian 识别。

```python
invalid_count_cases = [(-1, 10), (11, 10), (0, 0)]
for invalid_successes, invalid_trials in invalid_count_cases:
    try:
        bernoulli_mle(invalid_successes, invalid_trials)
    except ValueError:
        pass
    else:
        raise AssertionError("无效 Bernoulli 计数没有被拒绝")

try:
    beta_posterior_parameters(1, 10, 0.0, 2.0)
except ValueError:
    pass
else:
    raise AssertionError("非正 Beta 参数没有被拒绝")

constant_values = np.ones(4)
wide_variance_log_likelihood = gaussian_log_likelihood(
    np.array([1.0, math.log(1e-2)]), constant_values
)
narrow_variance_log_likelihood = gaussian_log_likelihood(
    np.array([1.0, math.log(1e-8)]), constant_values
)
assert narrow_variance_log_likelihood > wide_variance_log_likelihood

replayed_summary = np.array([
    beta_grid_summary(alpha, beta)
    for alpha, beta in zip(posterior_alpha, posterior_beta)
])
assert np.array_equal(replayed_summary, station_posterior_summary)
assert np.all(np.isfinite(replayed_summary))
print("无效输入、Gaussian 退化方向与确定性重放门禁全部通过")
```

这里特意没有把“Gaussian 退化方向”当作应通过的合法模型；测试只是确认实现能稳定复现似然继续上升的事实。生产接口应据此拒绝发布零方差点估计，转入更多数据、方差下界、共享结构或先验路线。

## 17. 30–60 分钟推导与代码练习

### 任务：实现可审计的 Beta–Bernoulli 资格更新器

实现一个只接受聚合计数的函数，但要求调用方同时传入 `station_version`、`audit_batch_id` 和幂等键。核心数学函数应：

1. 验证 $0\le S\le n$、$n>0$、$\alpha,\beta>0$；
2. 返回 MLE、后验参数、MAP 的边界状态、后验均值与方差；
3. 用稳定网格返回 95% 等尾可信区间和 $P(\theta>\theta_{max}\mid D)$；
4. 验证分批更新与一次更新相同；
5. 对 $S=0,S=n,\alpha\le1,\beta\le1$ 给出明确行为；
6. 拒绝重复 `audit_batch_id`，并说明这属于系统幂等而不是概率公式。

#### 思路与公式骨架

先更新

$$
\alpha'=\underline{\hspace{3cm}},
\qquad
\beta'=\underline{\hspace{3cm}}.
$$

再分别计算

$$
\hat\theta_{\rm MLE}=\underline{\hspace{3cm}},
\quad
\hat\theta_{\rm MAP}=\underline{\hspace{5cm}},
\quad
E[\theta\mid D]=\underline{\hspace{4cm}}.
$$

尾概率不能用“MAP 大于阈值”代替，应归一化后计算阈值右侧质量。

#### 代码骨架

```python
# TODO：完成带边界状态、可信区间、尾概率和幂等账本的资格更新器。
#
# def update_audit(successes, trials, alpha, beta, threshold, batch_id, seen_ids):
#     # 1. 验证计数、先验、阈值和幂等键；
#     # 2. 计算 MLE 与 alpha_prime/beta_prime；
#     # 3. 显式处理 MAP 的内点、左边界、右边界、双边界和平坦状态；
#     # 4. 计算 posterior mean/variance、equal-tail interval 和 tail probability；
#     # 5. 只在全部计算成功后提交 batch_id。
#     raise NotImplementedError
```

#### 参考答案的数学内核与边界测试

```python
def audit_beta_summary(successes, trials, alpha, beta, threshold=0.05):
    mle = bernoulli_mle(successes, trials)
    posterior_a, posterior_b = beta_posterior_parameters(
        successes, trials, alpha, beta
    )
    posterior_mode = beta_mode(posterior_a, posterior_b)
    posterior_mean = posterior_a / (posterior_a + posterior_b)
    posterior_variance = (
        posterior_a * posterior_b
        / ((posterior_a + posterior_b) ** 2 * (posterior_a + posterior_b + 1))
    )
    lower, upper, tail = beta_grid_summary(
        posterior_a, posterior_b, threshold=threshold
    )
    return {
        "mle": mle,
        "posterior_parameters": (posterior_a, posterior_b),
        "map": posterior_mode,
        "mean": posterior_mean,
        "variance": posterior_variance,
        "interval": (lower, upper),
        "tail_probability": tail,
    }

reference_summary = audit_beta_summary(0, 5, 1.0, 1.0)
assert reference_summary["mle"] == 0.0
assert reference_summary["map"] == 0.0
assert np.isclose(reference_summary["mean"], 1 / 7)
assert reference_summary["interval"][0] < reference_summary["interval"][1]
assert 0 <= reference_summary["tail_probability"] <= 1

left = beta_posterior_parameters(1, 4, 2.0, 3.0)
right = beta_posterior_parameters(2, 6, *left)
whole = beta_posterior_parameters(3, 10, 2.0, 3.0)
assert np.allclose(right, whole)
print("参考更新器边界与分批等价测试通过")
```

幂等账本属于 I/O 边界，不能藏进纯数学函数。正确实现应先检查 `batch_id`，完成全部数值验证后再以事务方式提交后验状态和消费键；失败重试不得把同一计数加两次。

### 练习 2：Categorical 零计数与类别字典变化

对四类结果实现 MLE、Uniform Dirichlet 后验预测和 `Dirichlet(2,…,2)` MAP。再在第五类上线时演示：直接把新类追加为 0 会怎样破坏已冻结参数和序列化合同；设计显式的类别字典版本迁移。

### 练习 3：Gaussian 退化与方差下界

构造全部相同、几乎相同和含一个尖峰的残差。比较无约束 MLE、固定量测方差、$\sigma^2\ge\epsilon$ 约束和适当方差先验。解释每条路线改变的是模型、参数空间还是先验，不能只写“加 epsilon 防数值错误”。

### 练习 4：相关挑战件的覆盖率

模拟 20 个独立缺陷实体，每个重放 10 次，实体内相关。比较把 200 条记录视为 iid 的 Wald 区间、按实体聚合的区间和 cluster bootstrap 覆盖率。固定真参数，至少重复 2,000 次审核实验。

### 练习 5：框架接口对照

下面只核对 Gaussian-prior MAP 与 scikit-learn Ridge 的目标缩放接口，不替代本章闭式推导。不同版本 API 运行前必须核对本地文档；仓库标准环境不为此新增依赖。

```python
# 不执行：需要可选的 scikit-learn 环境，并应先核对本地版本 API。
from sklearn.linear_model import Ridge

library_ridge = Ridge(
    alpha=map_penalty,
    fit_intercept=False,
    solver="svd",
)
library_ridge.fit(map_design, map_response)
assert np.allclose(library_ridge.coef_, map_closed_form)
```

## 18. 验收清单

### 数据与生成故事

- [ ] 随机变量、参数、条件变量、独立单位和目标挑战分布已经写清。
- [ ] `station_id`、版本、`challenge_id`、缺陷族、时间和标签成熟状态可追溯。
- [ ] 重放、重复实体、批内相关和选择性审核没有被当成 iid 新证据。
- [ ] 点估计、区间、尾概率与最终动作的关系已经冻结。

### 估计与数值

- [ ] Bernoulli/Categorical/Gaussian MLE 可从生成故事和 log likelihood 推出。
- [ ] 边界解、Gaussian 方差退化、零计数和不可辨识方向有明确策略。
- [ ] 总/平均 NLL、正则项、噪声尺度和先验尺度的换算可回放。
- [ ] 共轭批量—流式更新、解析梯度—有限差分和风险公式—精确枚举全部一致。
- [ ] 后验网格在 log 域归一化，尾部、区间和重放结果有限且确定。

### 先验与系统

- [ ] 先验早于目标批次冻结，来源域、参数化和适用范围有版本记录。
- [ ] Empirical Bayes 的超参数开发数据与最终审核数据职责分离。
- [ ] 先验中心/强度敏感性不会被单个默认值掩盖。
- [ ] 更新接口具有批次幂等键，失败重试不会重复累计证据。
- [ ] 例行、随机和模型触发审核均保留，反馈回路可估计。

## 19. 术语卡片

| 术语 | 本章精确定义 | 不要混淆 |
|---|---|---|
| 概率 | 固定参数后，关于随机结果的归一化分布 | 不是固定数据后的参数评分 |
| 似然 | 固定观测数据后，关于参数的函数 | 不自动是参数概率密度 |
| 充分统计量 | 在给定模型内保留关于参数的全部样本信息 | 不等于可删除原始审计记录 |
| MLE | 最大化数据似然的点估计 | 不保留参数不确定性 |
| Fisher 信息 | log likelihood 曲率的期望度量 | 重复相关记录不能简单相加信息 |
| 先验 | 看见目标数据之前对参数的分布 | 不能由同一目标批次调好后再次更新 |
| MAP | 后验密度的众数 | 依赖参数化，不等于后验均值 |
| 共轭先验 | 更新后与先验属于同一参数分布族 | 不代表它必然是最合适的领域先验 |
| 后验预测 | 对参数后验积分后的未来结果分布 | 不是把 MAP 塞回似然的插件近似 |
| 可信区间 | 给定模型、先验和数据后的参数后验区间 | 不是频率学派重复抽样覆盖保证 |
| Empirical Bayes | 用一组数据估计先验超参数后再条件推断 | 不是允许同一批数据无账本地计算两次 |

## 20. 本章结论

- 概率和似然使用同一公式，但固定的对象不同；似然不需要对参数归一化。
- iid 乘积是生成假设，不是方便记号；重复挑战实体会虚假放大曲率和信息量。
- Bernoulli、Categorical 和 Gaussian 的 MLE 都可由 log likelihood 推导，经验比例、均值和方差不是凭经验指定的损失结果。
- Bernoulli/Categorical 零计数产生边界概率，Gaussian 完全相同样本会让方差似然退化。
- 指数族的内点 MLE可解释为模型充分统计量期望与经验统计量匹配。
- MLE 的一致、渐近正态和有效性质需要可辨识、内点、正确模型族与独立性等正则条件。
- MAP 最大化 log likelihood 加 log prior；Gaussian/Laplace 先验分别对应带正确尺度的 $L_2/L_1$ 惩罚。
- Beta–Bernoulli、Dirichlet–Categorical 和 Gaussian–Gaussian 展示了计数或精度如何透明更新。
- MLE、MAP、后验均值、可信区间、尾概率和后验预测回答不同问题，不能互相代替。
- 收缩以偏差换方差；正确先验可能降低 MSE，强错配先验也可能让风险反超 MLE。
- MLE 在一一变换下等变；MAP 众数因密度 Jacobian 而依赖参数化。
- 先验必须早于目标批次冻结；同批数据复用、选择性审核和动作反馈会制造过度自信。
- 生产发布必须保存生成故事、数据实体键、目标缩放、先验版本、幂等账本、区间/尾概率与敏感性，而不只是一个点。

## 来源与许可

Kevin P. Murphy 的 *Probabilistic Machine Learning: An Introduction* 源 PDF 第 137–144 页（第 4.1–4.2.6 节）用于核对 MLE 定义、iid/条件似然、Bernoulli、Categorical、Gaussian 估计、充分统计量与 $n$/$n-1$ 区别；源 PDF 第 151–154 页（第 4.5.1–4.5.3 节）用于核对 Bernoulli MAP、零计数、收缩和 Gaussian 先验—$L_2$ 对应；源 PDF 第 159–166 页（第 4.6–4.6.2.10 节）用于核对 Bayes 后验、Beta–Bernoulli、MAP、后验均值/方差和 Beta–Binomial 后验预测；源 PDF 第 168–173 页（第 4.6.3–4.6.4.1 节）用于核对 Dirichlet–Categorical 与 Gaussian–Gaussian 共轭更新。该书为 CC BY-NC-ND 4.0；本章只据知识边界独立推导，没有改编或复用原书图表、案例、代码和措辞。

UC Berkeley CS 189/289A Lecture 5 *MLE and Multivariate Gaussians* 源 PDF 第 16–30 页用于交叉核对 MLE 生成故事、定义、独立性与统计性质；源 PDF 第 31–42 页用于核对一维 Gaussian MLE 与点估计/后验分布区别；源 PDF 第 43–52 页用于核对 Categorical/Multinomial 约束 MLE、Lagrange 乘子和一般优化接口。本章不复用课件图形、数据、练习或表达。

CMU 10-301/10-601 Lecture 16 *PAC Learning + MLE/MAP* 源 PDF 第 42–61 页用于交叉核对概率学习、MLE/MAP 定义、闭式求解步骤和 Bayes 目标；源 PDF 第 62–69 页用于核对 Beta 先验、Bernoulli MLE/MAP、共轭与正则化联系。Lecture 17 *Wrap up MAP + CNNs + RNNs* 源 PDF 第 3–16 页用于核对 MAP 回顾、共轭先验表和 Bernoulli 更新边界。本章不复制课堂图形、示例、测验或措辞。

多工位机器视觉漏检率资格审核案例的 16 个教学工位、750 件合成审核量、37 次合成漏检、挑战配比、光学状态、资格阈值、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作。数值不代表真实企业、产品、相机、光源、缺陷率、量测系统、质量标准、法规门槛或安全策略；模型不得直接修改设备、旁路联锁或自动放行产品。scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

<MlLessonProgress
  lesson-id="22-mle-map"
  next-href="/courses/machine-learning/23-generative-classifiers"
  next-label="第 23 章：高斯判别分析与朴素贝叶斯"
/>
