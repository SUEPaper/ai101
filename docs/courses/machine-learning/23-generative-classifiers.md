---
title: 高斯判别分析与朴素贝叶斯
description: 以数控主轴 EOL 异常根因分诊为案例，从类条件生成故事推导 GDA、LDA、QDA、Gaussian/Bernoulli/Multinomial Naive Bayes，并建立协方差、平滑、校准与生产审计协议。
course: machine-learning
edition: industrial
---

<MlLessonHero
  number="23"
  title="高斯判别分析与朴素贝叶斯"
  question="同一台主轴的连续量测和报警日志都指向某类故障时，我们得到的是独立证据的累积，还是同一物理现象被协方差与词计数重复解释？"
  duration="建议 720–840 分钟"
  :outcomes="[
    '从多元高斯密度、Mahalanobis 距离与 Bayes 规则完整推导 GDA、LDA 和 QDA，解释边界阶数、协方差自由度与白化几何',
    '从条件独立和事件模型推导 Gaussian、Bernoulli、Categorical 与 Multinomial Naive Bayes，实现平滑、log-space 评分与概率评价',
    '建立 campaign 隔离、协方差收缩、缺失边缘化、先验漂移、OOV、相关证据去重、校准和安全动作的生产审计协议'
  ]"
/>

上一章把 MLE、MAP、共轭更新和后验预测写成了可执行估计流程。本章把这些估计器放进一个真正的分类系统：给定一台 EOL（End of Line）测试异常的数控主轴，模型应把它送往哪条人工复检路线？

这不是“看到两个点云，套一个分类器”的章节。我们会先写清楚数据怎样生成，再逐项推导：

1. 类别先验怎样进入决策；
2. 类条件均值、协方差怎样决定概率椭圆；
3. 为什么共享协方差给出 LDA 的线性边界；
4. 为什么类别专属协方差给出 QDA 的二次边界；
5. 为什么对角协方差就是一种 Gaussian Naive Bayes；
6. 为什么离散报警词更适合 Bernoulli 或 Multinomial Naive Bayes；
7. 为什么“条件独立”能降低估计成本，也会把相关证据重复记账。

> 本章所有主轴、批次、量测、报警词和维修结论均为教学合成。模型只用于**安排复检顺序**，不得自动放行、自动返修、修改测试程序、改变机床参数或旁路安全联锁。

## 贯穿案例：数控主轴 EOL 异常根因分诊

### 1.1 业务动作先于算法

一条主轴装配线每天产生多类 EOL 报警。每条异常记录经工程师复核后标成三种教学根因：

| 标签 | 教学语义 | 模型建议动作 | 不允许的自动动作 |
|---|---|---|---|
| 转子不平衡 | 同步振动显著，包络与温升相对温和 | 进入动平衡复测 | 自动修正配重或放行 |
| 轴承预紧异常 | 包络能量、温升和轴向证据共同升高 | 进入轴承拆检 | 自动调预紧力或继续高速运行 |
| 传感器装夹异常 | 振动通道相关异常或掉线，热信号不支持机械故障 | 重装传感器后复测 | 屏蔽通道或把报警判成“假警”后放行 |

模型输出的是三个后验概率和一条复检路由。最终处置仍由工程师结合波形、装配记录和安全规程确认。

### 1.2 两种观测合同

同一台主轴有两类观测，不能混成一个含糊的“特征表”：

| 观测族 | 数据形状 | 本章模型 | 生成故事 |
|---|---|---|---|
| 连续量测 | 同步振动证据、包络能量证据、温升斜率 | GDA / LDA / QDA / Gaussian NB | 给定根因后，连续向量来自一个高斯分布 |
| 离散事件日志 | `1x_peak`、`envelope_rise`、`sensor_dropout` 等词的计数 | Multinomial NB | 给定根因后，日志中的每个词从类别专属词分布抽取 |

“连续量测近似高斯”和“日志词条件独立”都是**模型假设**，不是设备物理定律。后文会用残差、校准和相关性审计专门检验它们。

### 1.3 构造可重放的教学数据

案例包含 54 个生产 campaign，每个 campaign 有 6 台主轴，共 324 条记录。campaign 级漂移模拟同一装配批次共享的环境与夹具偏差。

```python
import numpy as np

rng = np.random.default_rng(2301)
n_campaigns, units_per_campaign = 54, 6
campaign_id = np.repeat(np.arange(n_campaigns), units_per_campaign)
n = campaign_id.size

class_names = np.array(["转子不平衡", "轴承预紧异常", "传感器装夹异常"])
actions = np.array(["动平衡复测", "轴承拆检", "传感器重装后复测"])
class_prior = np.array([0.44, 0.36, 0.20])
y = rng.choice(3, size=n, p=class_prior)

means_true = np.array([
    [2.35, 0.75, 0.55],
    [0.95, 2.25, 1.55],
    [1.55, 1.50, 0.35],
])
covs_true = np.array([
    [[0.42, 0.12, 0.06], [0.12, 0.24, 0.02], [0.06, 0.02, 0.16]],
    [[0.30, -0.08, 0.04], [-0.08, 0.48, 0.17], [0.04, 0.17, 0.28]],
    [[0.38, 0.31, -0.02], [0.31, 0.40, -0.01], [-0.02, -0.01, 0.11]],
])

campaign_shift = rng.normal(0, [0.10, 0.10, 0.06], size=(n_campaigns, 3))
X = np.vstack([
    rng.multivariate_normal(means_true[label], covs_true[label])
    + campaign_shift[campaign]
    for label, campaign in zip(y, campaign_id)
])

vocab = np.array([
    "1x_peak", "envelope_rise", "temperature_rise", "phase_stable",
    "sensor_dropout", "repeat_trigger", "axial_coincidence", "baseline_ok",
])
token_prob = np.array([
    [.22, .10, .09, .17, .07, .13, .08, .14],
    [.08, .20, .17, .08, .06, .15, .14, .12],
    [.10, .10, .07, .10, .22, .19, .08, .14],
])
doc_len = rng.integers(8, 19, size=n)
token_counts = np.vstack([
    rng.multinomial(length, token_prob[label])
    for length, label in zip(doc_len, y)
])

assert X.shape == (324, 3)
assert token_counts.shape == (324, 8)
assert np.all(token_counts.sum(axis=1) == doc_len)
print("三类记录数：", np.bincount(y, minlength=3).tolist())
```

固定随机种子不是为了让结果“看起来稳定”，而是让数据生成、模型选择和验收结果可以逐行重放。

### 1.4 campaign 级隔离

随机按行切分会把同一 campaign 的共享漂移同时送入训练和测试。这里先打乱 campaign，再冻结 36/9/9 个 campaign 作为训练、验证和测试。

```python
campaign_order = rng.permutation(n_campaigns)
train_campaigns = campaign_order[:36]
val_campaigns = campaign_order[36:45]
test_campaigns = campaign_order[45:]

train_mask = np.isin(campaign_id, train_campaigns)
val_mask = np.isin(campaign_id, val_campaigns)
test_mask = np.isin(campaign_id, test_campaigns)

assert not set(train_campaigns) & set(val_campaigns)
assert not set(train_campaigns) & set(test_campaigns)
assert not set(val_campaigns) & set(test_campaigns)
assert (train_mask.sum(), val_mask.sum(), test_mask.sum()) == (216, 54, 54)
print("训练/验证/测试：", train_mask.sum(), val_mask.sum(), test_mask.sum())
```

<MlQuiz
  kind="设计判断"
  question="同一 campaign 的 6 台主轴共享夹具漂移。若随机按行切分，最直接的问题是什么？"
  :options="['训练和测试会共享 campaign 指纹，测试结果偏乐观', '类别先验一定变成均匀分布', '协方差必然变成单位阵', 'Multinomial NB 无法计算词频']"
  :answer="0"
  explanation="数据的独立单位是 campaign，而不是单台记录。分组隔离要先于模型拟合和超参数选择。"
/>

## 2. 从生成故事到分类规则

### 2.1 联合分布的分解

生成式分类器从联合分布开始：

$$
p(\mathbf x,y=c)=p(y=c)p(\mathbf x\mid y=c)
=\pi_c p_c(\mathbf x).
$$

Bayes 规则把它翻译成后验：

$$
p(y=c\mid\mathbf x)
=\frac{\pi_c p_c(\mathbf x)}
{\sum_{k=1}^{C}\pi_k p_k(\mathbf x)}.
$$

如果动作损失是 0–1 损失，最大后验类别就是 Bayes 动作：

$$
\hat y(\mathbf x)=\arg\max_c p(y=c\mid\mathbf x)
=\arg\max_c\bigl[\log\pi_c+\log p_c(\mathbf x)\bigr].
$$

因此可以定义未归一化判别分数

$$
\delta_c(\mathbf x)=\log\pi_c+\log p_c(\mathbf x).
$$

分类只需比较 $\delta_c$，输出概率才需要做 log-sum-exp 归一化。工程实现中先在对数域累加，能避免许多小密度直接相乘后下溢为 0。

### 2.2 生成式与判别式不是“能否画边界”的区别

逻辑回归直接指定 $p(y\mid\mathbf x)$；GDA 和朴素贝叶斯指定 $p(y)$ 与 $p(\mathbf x\mid y)$，再推导 $p(y\mid\mathbf x)$。两者都能输出边界和概率，差别在**训练目标与假设位置**：

| 路线 | 拟合对象 | 优势 | 主要风险 |
|---|---|---|---|
| 生成式 | 联合似然 $p(\mathbf x,y)$ | 可单独拟合各类、处理某些缺失结构、做生成与先验替换 | 类条件密度错设会浪费信息甚至扭曲概率 |
| 判别式 | 条件似然 $p(y\mid\mathbf x)$ | 只为决策边界分配容量，通常对输入分布错设更稳健 | 不能直接回答“某类怎样生成观测” |

生成式模型并不自动具有因果含义。$p(\mathbf x\mid y)$ 描述的是训练流程中的统计条件分布，不等于“根因物理上独立生成了所有特征”。

::: ml-stop 停下来推一推 1
证明比较 $p(y=c\mid\mathbf x)$ 时可以删除共同分母 $p(\mathbf x)$。再说明：为什么在输出经过校准的概率时，不能把这个分母永远忽略？
:::

<MlQuiz
  kind="推导补全"
  question="生成式分类器比较类别时，为什么可以使用 log πc + log p(x|y=c) 而不显式计算 p(x)？"
  :options="['p(x) 对同一查询点的所有候选类别相同，不改变 argmax', 'p(x) 恒等于 1', '只有二分类才可以删除', '取对数会把分母变成零']"
  :answer="0"
  explanation="共同的正分母不改变类别排序；但若要输出归一化后验，仍需对所有类别分数做稳定归一化。"
/>

## 3. 多元高斯：均值之外还要描述方向

### 3.1 密度、形状与合法性

$d$ 维多元高斯密度为

$$
\mathcal N(\mathbf x;\boldsymbol\mu,\boldsymbol\Sigma)
=(2\pi)^{-d/2}|\boldsymbol\Sigma|^{-1/2}
\exp\left[-\frac12
(\mathbf x-\boldsymbol\mu)^\top
\boldsymbol\Sigma^{-1}
(\mathbf x-\boldsymbol\mu)\right].
$$

三个部分各有不同职责：

- $\boldsymbol\mu$ 定位概率椭圆的中心；
- $|\boldsymbol\Sigma|^{1/2}$ 描述椭圆总体积，密度必须相应归一化；
- 二次型按各方向的方差与相关性衡量离中心有多远。

严格密度要求 $\boldsymbol\Sigma$ 对称正定。半正定但奇异的矩阵只在低维子空间上有概率质量，不能直接使用普通的 $d$ 维密度公式。生产实现应显式验证形状、有限值、对称性和最小特征值，不能等到矩阵求解器给出晦涩错误。

### 3.2 协方差不是三个独立旋钮

二维协方差

$$
\boldsymbol\Sigma=
\begin{bmatrix}
\sigma_1^2 & \rho\sigma_1\sigma_2\\
\rho\sigma_1\sigma_2 & \sigma_2^2
\end{bmatrix}
$$

把两个边缘尺度与相关系数绑在一起。合法性要求 $|\rho|<1$；当 $|\rho|\to1$ 时，椭圆被压成接近一条线，条件数迅速上升。

### 3.3 Mahalanobis 距离

定义

$$
d_{\boldsymbol\Sigma}^2(\mathbf x,\boldsymbol\mu)
=(\mathbf x-\boldsymbol\mu)^\top
\boldsymbol\Sigma^{-1}
(\mathbf x-\boldsymbol\mu).
$$

它解决欧氏距离的两个盲点：量纲不同，以及特征相关。沿高方差方向移动同样的物理距离应当较不意外，沿低方差方向移动则更异常。

若 $\boldsymbol\Sigma=\mathbf U\boldsymbol\Lambda\mathbf U^\top$，定义白化矩阵

$$
\mathbf W=\boldsymbol\Lambda^{-1/2}\mathbf U^\top,
$$

则

$$
\mathbf W\boldsymbol\Sigma\mathbf W^\top=\mathbf I,
\qquad
d_{\boldsymbol\Sigma}^2
=\|\mathbf W(\mathbf x-\boldsymbol\mu)\|_2^2.
$$

```python
shared_covariance_demo = np.array([
    [0.37, 0.08, 0.05],
    [0.08, 0.31, 0.07],
    [0.05, 0.07, 0.20],
])
eigenvalues_demo, eigenvectors_demo = np.linalg.eigh(shared_covariance_demo)
whitener_demo = np.diag(eigenvalues_demo ** -0.5) @ eigenvectors_demo.T

query_demo = np.array([1.55, 1.25, 0.80])
mean_demo = np.array([1.30, 1.00, 0.55])
delta_demo = query_demo - mean_demo
mahalanobis_sq_demo = delta_demo @ np.linalg.solve(shared_covariance_demo, delta_demo)
whitened_sq_demo = np.sum((whitener_demo @ delta_demo) ** 2)

assert np.allclose(
    whitener_demo @ shared_covariance_demo @ whitener_demo.T,
    np.eye(3),
    atol=1e-12,
)
assert np.isclose(mahalanobis_sq_demo, whitened_sq_demo)
print("Mahalanobis² / 白化欧氏²：", mahalanobis_sq_demo, whitened_sq_demo)
```

<MahalanobisWhiteningGaugeLab />

<MlQuiz
  kind="结果预测"
  question="两个查询点与类别均值的欧氏距离相同；A 沿协方差长轴，B 沿短轴。哪个点的 Mahalanobis 距离通常更大？"
  :options="['B，因为短轴方向允许的方差更小', 'A，因为长轴在图上更长', '两者必然相等', '只由类别先验决定']"
  :answer="0"
  explanation="Mahalanobis 距离用方向标准差作单位。短轴方向相同的原始位移更罕见，因此标准化距离更大。"
/>

## 4. 高斯判别分析（GDA）

### 4.1 类条件模型

GDA 假设每个类别的连续观测服从多元高斯：

$$
p(\mathbf x\mid y=c)
=\mathcal N(\mathbf x;\boldsymbol\mu_c,\boldsymbol\Sigma_c).
$$

把高斯对数密度代入生成式判别分数，删除与类别无关的 $-\frac d2\log(2\pi)$，得到

$$
\boxed{
\delta_c(\mathbf x)
=\log\pi_c
-\frac12\log|\boldsymbol\Sigma_c|
-\frac12(\mathbf x-\boldsymbol\mu_c)^\top
\boldsymbol\Sigma_c^{-1}
(\mathbf x-\boldsymbol\mu_c)
}.
$$

它同时比较三件事：类别有多常见、该类分布体积有多大、查询点离该类中心有多少个协方差单位。

### 4.2 参数的 MLE

记类别 $c$ 有 $n_c$ 个样本，总样本数为 $n$。类别先验和均值 MLE 为

$$
\hat\pi_c=\frac{n_c}{n},
\qquad
\hat{\boldsymbol\mu}_c
=\frac1{n_c}\sum_{i:y_i=c}\mathbf x_i.
$$

QDA 的类别专属协方差 MLE 为

$$
\hat{\boldsymbol\Sigma}_c
=\frac1{n_c}\sum_{i:y_i=c}
(\mathbf x_i-\hat{\boldsymbol\mu}_c)
(\mathbf x_i-\hat{\boldsymbol\mu}_c)^\top.
$$

注意分母是 $n_c$：这是高斯类条件联合似然的 MLE。若目标是估计总体协方差并要求无偏，会使用 $n_c-1$；“用于分类的 MLE”和“无偏统计量”不能混用名字。

LDA 的共享协方差把**类内散度**池化：

$$
\hat{\boldsymbol\Sigma}_{\text{shared}}
=\frac1n\sum_c\sum_{i:y_i=c}
(\mathbf x_i-\hat{\boldsymbol\mu}_c)
(\mathbf x_i-\hat{\boldsymbol\mu}_c)^\top.
$$

不能围绕全局均值中心化；那会把类别均值之间的差异错误塞进类内噪声。

```python
def fit_gda(X_fit, y_fit, covariance="shared", shrinkage=0.0):
    X_fit = np.asarray(X_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=int)
    if X_fit.ndim != 2 or y_fit.shape != (len(X_fit),):
        raise ValueError("X must be 2D and y must align")

    classes = np.unique(y_fit)
    if not np.array_equal(classes, np.arange(classes.size)):
        raise ValueError("classes must be contiguous from zero")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1]")

    n_fit, d = X_fit.shape
    counts = np.bincount(y_fit, minlength=classes.size)
    priors = counts / n_fit
    means = np.vstack([X_fit[y_fit == c].mean(axis=0) for c in classes])

    class_covs = []
    for c in classes:
        centered = X_fit[y_fit == c] - means[c]
        class_covs.append(centered.T @ centered / counts[c])
    class_covs = np.stack(class_covs)

    if covariance == "shared":
        pooled = sum(counts[c] * class_covs[c] for c in classes) / n_fit
        covs = np.repeat(pooled[None], classes.size, axis=0)
    elif covariance == "class":
        covs = class_covs.copy()
    elif covariance == "diagonal":
        covs = np.stack([np.diag(np.diag(cov)) for cov in class_covs])
    else:
        raise ValueError("unknown covariance mode")

    for c in classes:
        scale = np.trace(covs[c]) / d
        covs[c] = (1 - shrinkage) * covs[c] + shrinkage * scale * np.eye(d)
        eigenvalues = np.linalg.eigvalsh(covs[c])
        tolerance = np.finfo(float).eps * max(1.0, np.max(np.abs(eigenvalues))) * d
        if eigenvalues[0] <= tolerance:
            raise ValueError("covariance is not positive definite; use shrinkage")

    return {
        "priors": priors,
        "means": means,
        "covs": covs,
        "classes": classes,
    }
```

### 4.3 评分时不要显式求逆

公式写 $\boldsymbol\Sigma^{-1}$，实现不应写 `np.linalg.inv(cov)` 再相乘。线性求解 `solve(cov, delta)` 更稳定，也避免显式构造逆矩阵。

```python
def gda_log_scores(model, X_query):
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    scores = np.empty((len(X_query), len(model["classes"])))

    for c in model["classes"]:
        delta = X_query - model["means"][c]
        sign, logdet = np.linalg.slogdet(model["covs"][c])
        if sign <= 0:
            raise ValueError("covariance must be positive definite")
        solved = np.linalg.solve(model["covs"][c], delta.T).T
        scores[:, c] = (
            np.log(model["priors"][c])
            - 0.5 * logdet
            - 0.5 * np.sum(delta * solved, axis=1)
        )
    return scores

def stable_softmax(scores):
    scores = np.asarray(scores, dtype=float)
    shifted = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)

def gda_predict_proba(model, X_query):
    return stable_softmax(gda_log_scores(model, X_query))
```

如果同一个模型需要大量推理，生产实现通常预先缓存每类 Cholesky 分解和 log determinant，而不是每个查询点重新分解协方差。

## 5. LDA：共享协方差为何给出线性边界

### 5.1 展开并删除共同二次项

LDA（Linear Discriminant Analysis）约束

$$
\boldsymbol\Sigma_c=\boldsymbol\Sigma
\quad\text{for all }c.
$$

展开二次型：

$$
\begin{aligned}
-\frac12(\mathbf x-\boldsymbol\mu_c)^\top
\boldsymbol\Sigma^{-1}(\mathbf x-\boldsymbol\mu_c)
=&-\frac12\mathbf x^\top\boldsymbol\Sigma^{-1}\mathbf x\\
&+\mathbf x^\top\boldsymbol\Sigma^{-1}\boldsymbol\mu_c\\
&-\frac12\boldsymbol\mu_c^\top
\boldsymbol\Sigma^{-1}\boldsymbol\mu_c.
\end{aligned}
$$

$-\frac12\mathbf x^\top\boldsymbol\Sigma^{-1}\mathbf x$ 和 $-\frac12\log|\boldsymbol\Sigma|$ 对所有类别相同，比较时抵消：

$$
\boxed{
\delta_c^{\text{LDA}}(\mathbf x)
=\mathbf x^\top\boldsymbol\Sigma^{-1}\boldsymbol\mu_c
-\frac12\boldsymbol\mu_c^\top
\boldsymbol\Sigma^{-1}\boldsymbol\mu_c
+\log\pi_c
}.
$$

这是 $\mathbf x$ 的仿射函数，所以任意两类满足 $\delta_a=\delta_b$ 的集合是超平面。

### 5.2 二分类 LDA 的 log odds

两类情形下

$$
\log\frac{p(y=1\mid\mathbf x)}{p(y=0\mid\mathbf x)}
=\mathbf w^\top\mathbf x+b,
$$

其中

$$
\mathbf w=\boldsymbol\Sigma^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0),
$$

$$
b=-\frac12\boldsymbol\mu_1^\top\boldsymbol\Sigma^{-1}\boldsymbol\mu_1
+\frac12\boldsymbol\mu_0^\top\boldsymbol\Sigma^{-1}\boldsymbol\mu_0
+\log\frac{\pi_1}{\pi_0}.
$$

因此 LDA 推导出的后验具有 logistic 形式。但它和逻辑回归不是同一个估计器：LDA 最大化联合似然并通过均值、协方差得到 $\mathbf w,b$；逻辑回归直接最大化条件似然。

```python
lda_binary = fit_gda(X[train_mask][y[train_mask] != 2], y[train_mask][y[train_mask] != 2], "shared")
sigma_binary = lda_binary["covs"][0]
mu0_binary, mu1_binary = lda_binary["means"]
w_binary = np.linalg.solve(sigma_binary, mu1_binary - mu0_binary)
b_binary = (
    -0.5 * mu1_binary @ np.linalg.solve(sigma_binary, mu1_binary)
    + 0.5 * mu0_binary @ np.linalg.solve(sigma_binary, mu0_binary)
    + np.log(lda_binary["priors"][1] / lda_binary["priors"][0])
)

probe_binary = X[val_mask][:12]
score_difference = (
    gda_log_scores(lda_binary, probe_binary)[:, 1]
    - gda_log_scores(lda_binary, probe_binary)[:, 0]
)
assert np.allclose(score_difference, probe_binary @ w_binary + b_binary)
print("LDA 二分类线性恒等式最大误差：", np.max(np.abs(score_difference - (probe_binary @ w_binary + b_binary))))
```

::: ml-stop 停下来推一推 2
在一维、等方差、等先验、两类均值为 $\mu_0,\mu_1$ 时，把 $\delta_0(x)=\delta_1(x)$ 展开，证明边界是 $(\mu_0+\mu_1)/2$。再说明先验 $\pi_1$ 增大后边界向哪一类均值移动。
:::

## 6. QDA：类别专属协方差为何保留二次项

### 6.1 两类分数差

QDA（Quadratic Discriminant Analysis）允许每类有自己的协方差。令两类分数差为 $g_{10}(\mathbf x)=\delta_1(\mathbf x)-\delta_0(\mathbf x)$，展开可得

$$
g_{10}(\mathbf x)
=\frac12\mathbf x^\top
(\boldsymbol\Sigma_0^{-1}-\boldsymbol\Sigma_1^{-1})\mathbf x
+(\boldsymbol\Sigma_1^{-1}\boldsymbol\mu_1
-\boldsymbol\Sigma_0^{-1}\boldsymbol\mu_0)^\top\mathbf x
+k,
$$

其中 $k$ 汇总均值二次项、log determinant 和 log prior。只要 $\boldsymbol\Sigma_0\ne\boldsymbol\Sigma_1$，二次矩阵通常不为 0，边界就可以是椭圆、抛物线、双曲线或退化二次曲线。

### 6.2 灵活性要用参数和样本支付

一个 $d\times d$ 对称协方差有 $d(d+1)/2$ 个自由参数。忽略均值和先验，三种协方差合同的参数量是：

| 合同 | 协方差参数量 | 边界 | 典型交换 |
|---|---:|---|---|
| 共享全协方差（LDA） | $d(d+1)/2$ | 线性 | 较低方差，无法表达类间形状差异 |
| 类别专属全协方差（QDA） | $C d(d+1)/2$ | 二次 | 更灵活，小样本下更易病态 |
| 类别专属对角协方差（Gaussian NB） | $Cd$ | 轴对齐二次 | 忽略类内相关，估计成本低 |

Gaussian NB 不是“另一个完全无关的算法”：对连续特征采用类别专属的一维 Gaussian 并假设条件独立，恰好等价于对角协方差 QDA。

```python
rng_boundary = np.random.default_rng(2310)
probe_boundary = rng_boundary.normal(size=(500, 3))
polynomial_design = np.column_stack([
    np.ones(len(probe_boundary)),
    probe_boundary,
    probe_boundary[:, 0] ** 2,
    probe_boundary[:, 1] ** 2,
    probe_boundary[:, 2] ** 2,
    probe_boundary[:, 0] * probe_boundary[:, 1],
    probe_boundary[:, 0] * probe_boundary[:, 2],
    probe_boundary[:, 1] * probe_boundary[:, 2],
])

lda_check = fit_gda(X[train_mask], y[train_mask], "shared")
qda_check = fit_gda(X[train_mask], y[train_mask], "class")
for name, model in [("LDA", lda_check), ("QDA", qda_check)]:
    difference = (
        gda_log_scores(model, probe_boundary)[:, 1]
        - gda_log_scores(model, probe_boundary)[:, 0]
    )
    coefficients = np.linalg.lstsq(polynomial_design, difference, rcond=None)[0]
    residual = np.max(np.abs(polynomial_design @ coefficients - difference))
    quadratic_norm = np.linalg.norm(coefficients[4:])
    print(name, "二次系数范数/拟合残差：", quadratic_norm, residual)

assert np.linalg.norm(np.linalg.lstsq(polynomial_design, gda_log_scores(lda_check, probe_boundary)[:, 1] - gda_log_scores(lda_check, probe_boundary)[:, 0], rcond=None)[0][4:]) < 1e-10
```

<GaussianBoundaryProbeLab />

<MlQuiz
  kind="推导补全"
  question="QDA 何时会退化为 LDA 的线性边界？"
  :options="['所有类别协方差相同，使 xᵀΣc⁻¹x 的类别差为零', '所有类别先验相同', '特征维数等于 2', '每类样本数相同']"
  :answer="0"
  explanation="决定边界阶数的是类别间二次矩阵之差。先验和样本数会移动边界，但不会单独消掉二次项。"
/>

## 7. 从零实现、冻结选择与概率评价

### 7.1 不能只报准确率

后验模型至少同时记录：

- accuracy：最终类别是否正确；
- negative log-likelihood（NLL）：给真实类别分配了多少概率；
- multiclass Brier：整个概率向量离 one-hot 目标多远。

```python
def multiclass_metrics(y_true, proba):
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if proba.shape != (len(y_true), len(class_names)):
        raise ValueError("probability shape mismatch")
    if not np.allclose(proba.sum(axis=1), 1.0):
        raise ValueError("probabilities must sum to one")

    onehot = np.eye(proba.shape[1])[y_true]
    pred = proba.argmax(axis=1)
    return {
        "accuracy": np.mean(pred == y_true),
        "nll": -np.mean(np.log(np.clip(proba[np.arange(len(y_true)), y_true], 1e-15, 1))),
        "brier": np.mean(np.sum((proba - onehot) ** 2, axis=1)),
    }
```

### 7.2 同一数据上的三个协方差合同

```python
models = {
    "LDA": fit_gda(X[train_mask], y[train_mask], "shared", 0.0),
    "QDA": fit_gda(X[train_mask], y[train_mask], "class", 0.0),
    "Gaussian NB": fit_gda(X[train_mask], y[train_mask], "diagonal", 0.0),
}

for name, model in models.items():
    val_result = multiclass_metrics(y[val_mask], gda_predict_proba(model, X[val_mask]))
    print(name, "验证：", {key: round(float(value), 4) for key, value in val_result.items()})
```

本次冻结数据上，QDA 验证 accuracy 为 0.8704、NLL 为 0.2541；LDA 和 Gaussian NB 的验证 accuracy 都是 0.8333，但 NLL 分别为 0.3815 和 0.3697。只看 accuracy 会漏掉 QDA 概率排序更好的证据。

### 7.3 收缩协方差

高维或少样本时，可把每类协方差向球形目标收缩：

$$
\widetilde{\boldsymbol\Sigma}_c(\lambda)
=(1-\lambda)\hat{\boldsymbol\Sigma}_c
+\lambda\frac{\operatorname{tr}(\hat{\boldsymbol\Sigma}_c)}{d}\mathbf I,
\qquad 0\le\lambda\le1.
$$

$\lambda$ 是超参数，必须在验证集冻结，不能在测试结果上挑选。

```python
shrinkage_grid = [0.0, 0.05, 0.15, 0.35, 0.65, 1.0]
qda_val_path = []

for lam in shrinkage_grid:
    candidate = fit_gda(X[train_mask], y[train_mask], "class", lam)
    result = multiclass_metrics(y[val_mask], gda_predict_proba(candidate, X[val_mask]))
    qda_val_path.append((lam, result["nll"]))

best_shrinkage = min(qda_val_path, key=lambda item: (item[1], item[0]))[0]
qda_frozen = fit_gda(X[train_mask], y[train_mask], "class", best_shrinkage)
qda_test = multiclass_metrics(y[test_mask], gda_predict_proba(qda_frozen, X[test_mask]))

print("λ→验证 NLL：", [(lam, round(float(nll), 4)) for lam, nll in qda_val_path])
print("冻结 λ：", best_shrinkage)
print("测试：", {key: round(float(value), 4) for key, value in qda_test.items()})
```

验证集冻结 $\lambda=0.05$；独立测试 accuracy 为 0.8704、NLL 为 0.2795、Brier 为 0.1673。这里收缩改善了验证 NLL，却没有保证测试 accuracy 一定超过 LDA。这正是独立测试集的意义。

### 7.4 路由账本要保留逐类召回

```python
qda_test_proba = gda_predict_proba(qda_frozen, X[test_mask])
qda_test_pred = qda_test_proba.argmax(axis=1)
confusion = np.zeros((3, 3), dtype=int)
for truth, prediction in zip(y[test_mask], qda_test_pred):
    confusion[truth, prediction] += 1

recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
route_counts = np.bincount(qda_test_pred, minlength=3)
assert confusion.sum() == test_mask.sum()
print("测试混淆矩阵：\n", confusion)
print("逐类召回：", np.round(recall, 3))
print("三条复检路线负载：", dict(zip(actions, route_counts)))
```

总体准确率不能替代“传感器装夹异常是否被错误送往机械拆检”等逐类问题。若动作代价不对称，还应把后验送入第 8 章的期望损失决策，而不是固定使用 argmax。

<MlQuiz
  kind="设计判断"
  question="验证集上 QDA 的 NLL 最低，而测试集上 LDA 的 accuracy 略高。正确结论是什么？"
  :options="['按预先冻结的选择协议报告 QDA 测试结果，同时如实给出 LDA 对照', '查看测试集后改选 LDA 并删除 QDA 结果', '反复修改 λ 直到 QDA 测试 accuracy 最高', '说明 NLL 指标无效']"
  :answer="0"
  explanation="测试集不能参与模型选择。不同指标可能排序不同，应在协议中提前声明主指标，并完整报告冻结模型。"
/>

## 8. GDA 的工程边界

### 8.1 小样本高维：协方差秩先于求逆

类别 $c$ 的中心化数据矩阵最多只有 $n_c-1$ 个独立方向，因此样本协方差满足

$$
\operatorname{rank}(\hat{\boldsymbol\Sigma}_c)
\le \min(d,n_c-1).
$$

只要 $d\ge n_c$，QDA 的 MLE 协方差必然奇异。即使 $d<n_c$，强共线也会产生极高条件数。下面构造每类 8 个样本、40 个特征的反例。

```python
rng_hd = np.random.default_rng(2323)
X_hd = rng_hd.normal(size=(24, 40))
y_hd = np.repeat(np.arange(3), 8)

singular_ranks = []
for c in range(3):
    centered = X_hd[y_hd == c] - X_hd[y_hd == c].mean(axis=0)
    singular_ranks.append(np.linalg.matrix_rank(centered.T @ centered / 8))

try:
    fit_gda(X_hd, y_hd, "class", 0.0)
except ValueError as exc:
    rejection = str(exc)
else:
    raise AssertionError("singular covariance should be rejected")

repaired_hd = fit_gda(X_hd, y_hd, "class", 0.25)
condition_numbers_hd = [np.linalg.cond(cov) for cov in repaired_hd["covs"]]

assert singular_ranks == [7, 7, 7]
assert all(np.isfinite(condition_numbers_hd))
print("原始秩：", singular_ranks)
print("拒绝原因：", rejection)
print("收缩后条件数：", np.round(condition_numbers_hd, 2))
```

收缩不是“给对角线随便加一个小数”。应记录目标矩阵、$\lambda$ 的选择数据、最小特征值和条件数。若系统跨语言部署，还要固定 float 精度、特征顺序和求解器容差。

### 8.2 类别先验变化：只在 label shift 下替换

若部署域满足

$$
p_{\text{deploy}}(\mathbf x\mid y)
=p_{\text{train}}(\mathbf x\mid y),
$$

只是 $p(y)$ 改变，那么可以重加权旧后验：

$$
p_{\text{deploy}}(y=c\mid\mathbf x)
\propto
p_{\text{train}}(y=c\mid\mathbf x)
\frac{\pi_c^{\text{deploy}}}{\pi_c^{\text{train}}}.
$$

```python
test_probabilities = gda_predict_proba(qda_frozen, X[test_mask])
ambiguous_index = np.argmin(test_probabilities.max(axis=1))
old_posterior = test_probabilities[ambiguous_index]

deployment_prior = np.array([0.25, 0.55, 0.20])
adjusted_posterior = old_posterior * (deployment_prior / qda_frozen["priors"])
adjusted_posterior /= adjusted_posterior.sum()

print("训练先验后验：", np.round(old_posterior, 4))
print("部署先验后验：", np.round(adjusted_posterior, 4))
assert np.isclose(adjusted_posterior.sum(), 1.0)
```

这个样本的轴承预紧异常后验从约 0.404 上升到 0.595。若量测硬件、转速程序或装配工艺同时变化，$p(\mathbf x\mid y)$ 也会漂移，仅替换先验不够。

### 8.3 缺失连续特征：边缘化，不是偷偷补零

高斯分布的任意子向量仍是高斯。若查询只观测到索引集合 $O$，应使用对应均值 $\boldsymbol\mu_{c,O}$ 和协方差子矩阵 $\boldsymbol\Sigma_{c,OO}$：

$$
p(\mathbf x_O\mid y=c)
=\mathcal N(\mathbf x_O;\boldsymbol\mu_{c,O},\boldsymbol\Sigma_{c,OO}).
$$

```python
def gda_log_scores_observed(model, x_query, observed):
    observed = np.asarray(observed, dtype=int)
    x_query = np.asarray(x_query, dtype=float)[observed]
    scores = []

    for c in model["classes"]:
        mean = model["means"][c, observed]
        covariance = model["covs"][c][np.ix_(observed, observed)]
        delta = x_query - mean
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("observed covariance must be positive definite")
        scores.append(
            np.log(model["priors"][c])
            - 0.5 * logdet
            - 0.5 * delta @ np.linalg.solve(covariance, delta)
        )
    return np.array(scores)

missing_query = X[test_mask][0].copy()
observed_columns = np.array([0, 2])  # 包络能量列缺失
missing_scores = gda_log_scores_observed(qda_frozen, missing_query, observed_columns)
missing_posterior = stable_softmax(missing_scores[None])[0]

assert np.isclose(missing_posterior.sum(), 1.0)
print("缺一列后的后验：", np.round(missing_posterior, 4))
```

补零只有在“零”是已定义的真实物理量且训练时也采用同一协议时才合法。缺失模式本身若与故障有关，还要建模 missingness 或增加经过验证的缺失指示，而不是假装随机缺失。

### 8.4 离群点与厚尾

均值和协方差 MLE 对极端量测敏感。一个饱和脉冲可能同时拖动类别中心、拉长协方差椭圆并降低自身 Mahalanobis 距离。工业流程至少要：

1. 在模型前区分传感器越界、物理极值和数据损坏；
2. 报告每类稳健位置/尺度对照和影响点；
3. 比较 Student-$t$ 类条件、收缩协方差或稳健协方差；
4. 不因模型“不喜欢”就删除真实极端故障。

<MlQuiz
  kind="错误诊断"
  question="某类只有 8 个训练样本，却要估计 40 维 QDA 全协方差。最根本的问题是什么？"
  :options="['中心化协方差秩至多为 7，普通高斯密度不可直接求解', '类别先验一定等于零', 'QDA 会自动变成 kNN', '只要使用 float64 就没有问题']"
  :answer="0"
  explanation="这是结构性不可辨识，不是浮点精度问题。应降低协方差自由度、收缩、降维或增加独立样本。"
/>

## 9. 朴素贝叶斯：用条件独立换取可计数的模型

### 9.1 “朴素”指的是给定类别后的独立

朴素贝叶斯假设

$$
p(\mathbf x\mid y=c)
=\prod_{j=1}^{d}p(x_j\mid y=c).
$$

它不是说特征边缘独立：两个报警词可以都因为“轴承预紧异常”而频繁出现，因此在不条件于类别时强相关；假设只声称知道 $y$ 后，一个词不再提供另一个词的额外信息。

对数判别分数变成

$$
\delta_c(\mathbf x)
=\log\pi_c+\sum_{j=1}^{d}\log p(x_j\mid y=c).
$$

联合分布的指数级状态空间被拆成许多低维估计问题。代价是相关证据可能被重复记账，后验常比实际更极端。

### 9.2 分布必须匹配数据语义

“Naive Bayes”不是单一输入格式。必须先回答 $x_j$ 是什么：

| 变体 | $x_j$ 的语义 | 类条件模型 | 缺席是否贡献证据 |
|---|---|---|---|
| Gaussian NB | 连续量测值 | 每类每列一个一维 Gaussian | 是，由连续密度决定 |
| Bernoulli NB | 某报警词是否出现 | Bernoulli | 是，$x_j=0$ 有 $\log(1-\theta_{jc})$ |
| Categorical NB | 每个字段取一个离散状态 | Categorical | 每个字段恰有一个状态 |
| Multinomial NB | 一份日志中各词出现次数 | Multinomial 词概率 | 未出现词的计数为 0，不直接加项 |

不能把 word presence 的 Bernoulli 公式和 token count 的 Multinomial 公式拼在一起。相同词出现 10 次，在 Bernoulli 模型中仍是“出现”，在 Multinomial 模型中则贡献 10 倍 log probability。

### 9.3 Multinomial 事件模型

给定类别 $c$，一份长度为 $L$ 的日志包含词计数 $\mathbf x=(x_1,\ldots,x_V)$，满足 $\sum_j x_j=L$。忽略与类别无关的多项式系数：

$$
p(\mathbf x\mid y=c)
\propto\prod_{j=1}^{V}\theta_{cj}^{x_j},
\qquad
\sum_j\theta_{cj}=1.
$$

判别分数为

$$
\delta_c(\mathbf x)
=\log\pi_c+\sum_{j=1}^{V}x_j\log\theta_{cj}.
$$

词概率的 MLE 是

$$
\hat\theta_{cj}^{\text{MLE}}
=\frac{N_{cj}}{\sum_{k=1}^{V}N_{ck}},
$$

其中 $N_{cj}$ 是训练集中类别 $c$ 的词 $j$ 总计数。

### 9.4 Laplace / additive smoothing

若 $N_{cj}=0$，MLE 给出 $\hat\theta_{cj}=0$。查询日志只要含该词，该类整个似然就变成 0。加性平滑使用

$$
\hat\theta_{cj}
=\frac{N_{cj}+\alpha}
{\sum_k N_{ck}+\alpha V},
\qquad \alpha>0.
$$

这等价于对每个类别词概率放置对称 Dirichlet 先验并使用相应后验估计。分母必须加 $\alpha V$，否则词概率不再和为 1。

```python
def fit_multinomial_nb(counts, labels, alpha=1.0):
    counts = np.asarray(counts, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if counts.ndim != 2 or labels.shape != (len(counts),):
        raise ValueError("counts must be 2D and labels must align")
    if np.any(counts < 0) or alpha <= 0:
        raise ValueError("counts must be nonnegative and alpha must be positive")

    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(classes.size)):
        raise ValueError("classes must be contiguous from zero")

    class_counts = np.bincount(labels, minlength=classes.size)
    token_by_class = np.vstack([
        counts[labels == c].sum(axis=0)
        for c in classes
    ])
    token_probabilities = (
        (token_by_class + alpha)
        / (token_by_class.sum(axis=1, keepdims=True) + alpha * counts.shape[1])
    )

    assert np.allclose(token_probabilities.sum(axis=1), 1.0)
    return {
        "log_prior": np.log(class_counts / len(labels)),
        "log_token_prob": np.log(token_probabilities),
        "classes": classes,
    }
```

```python
def multinomial_nb_predict_proba(model, counts):
    counts = np.atleast_2d(np.asarray(counts, dtype=float))
    if np.any(counts < 0):
        raise ValueError("counts must be nonnegative")
    scores = counts @ model["log_token_prob"].T + model["log_prior"]
    return stable_softmax(scores)
```

### 9.5 平滑强度也要冻结

```python
nb_alpha_path = []
for alpha_candidate in [0.05, 0.2, 1.0, 5.0, 20.0]:
    candidate = fit_multinomial_nb(
        token_counts[train_mask],
        y[train_mask],
        alpha_candidate,
    )
    result = multiclass_metrics(
        y[val_mask],
        multinomial_nb_predict_proba(candidate, token_counts[val_mask]),
    )
    nb_alpha_path.append((alpha_candidate, result["nll"]))

best_alpha = min(nb_alpha_path, key=lambda item: (item[1], item[0]))[0]
nb_frozen = fit_multinomial_nb(token_counts[train_mask], y[train_mask], best_alpha)
nb_test = multiclass_metrics(
    y[test_mask],
    multinomial_nb_predict_proba(nb_frozen, token_counts[test_mask]),
)

print("α→验证 NLL：", [(alpha, round(float(nll), 4)) for alpha, nll in nb_alpha_path])
print("冻结 α / 测试指标：", best_alpha, {key: round(float(value), 4) for key, value in nb_test.items()})
```

本次验证冻结 $\alpha=1$；测试 accuracy 为 0.7778、NLL 为 0.5276。与连续量测模型的结果不能直接解释成“日志不重要”：两种模型看的是不同观测合同，且是否融合应在独立验证协议中决定。

::: ml-stop 停下来推一推 3
某类共有 90 次词事件，词表大小 $V=8$，其中 `sensor_dropout` 从未出现。分别写出 $\alpha=0$ 和 $\alpha=1$ 时该词的概率。为什么分母是 $90+8$ 而不是 $90+1$？
:::

<MlQuiz
  kind="推导补全"
  question="Multinomial NB 对词表大小 V 使用加性平滑 α 后，为什么分母要加 αV？"
  :options="['每个词槽都增加 α 个伪计数，新增总质量是 αV', '为了让文档长度恒等于 V', '因为类别数等于 V', '只是数值优化器的约定']"
  :answer="0"
  explanation="分子对 V 个词各加 α，分母必须加入同样的总伪计数，才能保持一行概率和为 1。"
/>

## 10. 朴素贝叶斯的四类典型失效

### 10.1 零计数与 OOV 是不同问题

零计数指词在固定词表中存在，但某类训练数据没有见过；平滑可以给它非零概率。OOV（out of vocabulary）指部署日志出现了词表之外的新 token；模型根本没有对应列，必须通过冻结的解析规则映射到 `<UNK>`、丢弃或触发 schema 告警。

```python
toy_counts = np.array([
    [4, 1, 0],
    [3, 2, 0],
    [0, 4, 1],
    [0, 3, 2],
])
toy_y = np.array([0, 0, 1, 1])

unsmoothed = np.vstack([
    toy_counts[toy_y == c].sum(axis=0) / toy_counts[toy_y == c].sum()
    for c in range(2)
])
smoothed_toy = fit_multinomial_nb(toy_counts, toy_y, alpha=1.0)
smoothed_probabilities = np.exp(smoothed_toy["log_token_prob"])

assert unsmoothed[0, 2] == 0.0
assert smoothed_probabilities[0, 2] > 0.0
print("类 0 第三个词，平滑前/后：", unsmoothed[0, 2], smoothed_probabilities[0, 2])
```

### 10.2 直接连乘会下溢

```python
tiny_probabilities = np.full(1000, 1e-3)
direct_product = np.prod(tiny_probabilities)
log_product = np.log(tiny_probabilities).sum()

assert direct_product == 0.0
assert np.isfinite(log_product)
print("直接乘积 / 对数和：", direct_product, log_product)
```

“最后再取对数”已经太晚：乘积先变成 0，信息就丢失了。必须从单项 log probability 开始累计，再用 log-sum-exp 归一化。

### 10.3 相关证据会被重复记账

设 `1x_peak` 和 `mirror_1x_peak` 来自同一只物理峰值，只是两条软件馈线。若把两列都送进朴素贝叶斯，条件独立模型会把相同的 log likelihood ratio 加两次。

```python
base_query = np.array([[5, 1, 0, 3, 0, 1, 0, 2]])
duplicated_query = base_query.copy()
duplicated_query[0, 0] += 5  # 模拟同一 1× 峰被镜像馈线再次计数

base_probability = multinomial_nb_predict_proba(nb_frozen, base_query)[0]
duplicated_probability = multinomial_nb_predict_proba(nb_frozen, duplicated_query)[0]

print("单路证据：", np.round(base_probability, 4))
print("镜像重复：", np.round(duplicated_probability, 4))
assert duplicated_probability[0] > base_probability[0]
```

在这条日志上，“转子不平衡”的后验从 0.9946 进一步升至约 0.9999，但没有新增独立物理量测。高置信度不等于高信息量。

<NaiveBayesEvidenceLedgerLab />

### 10.4 用重复抽样量化过度自信

下面先生成 4 个有信息的二值报警，再各复制一个 96% 概率与原报警相同的镜像通道。去重模型与重复模型的分类准确率近似，但重复模型的 NLL 和 Brier 更差。

```python
def fit_bernoulli_nb(features, labels, alpha=1.0):
    features = np.asarray(features, dtype=int)
    labels = np.asarray(labels, dtype=int)
    if np.any((features != 0) & (features != 1)):
        raise ValueError("Bernoulli features must be binary")

    classes = np.unique(labels)
    class_sizes = np.bincount(labels, minlength=len(classes))
    positive_counts = np.vstack([
        features[labels == c].sum(axis=0)
        for c in classes
    ])
    probabilities = (positive_counts + alpha) / (class_sizes[:, None] + 2 * alpha)
    return {
        "log_prior": np.log(class_sizes / len(labels)),
        "p": probabilities,
        "classes": classes,
    }

def bernoulli_nb_predict_proba(model, features):
    features = np.atleast_2d(np.asarray(features, dtype=int))
    probabilities = model["p"]
    scores = (
        features @ np.log(probabilities).T
        + (1 - features) @ np.log(1 - probabilities).T
        + model["log_prior"]
    )
    return stable_softmax(scores)
```

```python
rng_dependency = np.random.default_rng(2324)
n_dependency = 10_000
y_dependency = rng_dependency.integers(0, 2, size=n_dependency)

positive_rates = np.where(
    y_dependency[:, None] == 1,
    np.array([.78, .68, .58, .48]),
    np.array([.22, .32, .42, .52]),
)
latent_alarms = rng_dependency.binomial(1, positive_rates)
mirror_alarms = np.where(
    rng_dependency.random(latent_alarms.shape) < .96,
    latent_alarms,
    1 - latent_alarms,
)

X_deduplicated = latent_alarms
X_with_mirrors = np.column_stack([latent_alarms, mirror_alarms])
dependency_train = np.arange(n_dependency) < 5_000

deduplicated_model = fit_bernoulli_nb(
    X_deduplicated[dependency_train],
    y_dependency[dependency_train],
)
mirror_model = fit_bernoulli_nb(
    X_with_mirrors[dependency_train],
    y_dependency[dependency_train],
)

def binary_probability_metrics(y_true, proba):
    onehot = np.eye(2)[y_true]
    return {
        "accuracy": np.mean(proba.argmax(axis=1) == y_true),
        "nll": -np.mean(np.log(np.clip(proba[np.arange(len(y_true)), y_true], 1e-15, 1))),
        "brier": np.mean(np.sum((proba - onehot) ** 2, axis=1)),
    }

deduplicated_metrics = binary_probability_metrics(
    y_dependency[~dependency_train],
    bernoulli_nb_predict_proba(deduplicated_model, X_deduplicated[~dependency_train]),
)
mirror_metrics = binary_probability_metrics(
    y_dependency[~dependency_train],
    bernoulli_nb_predict_proba(mirror_model, X_with_mirrors[~dependency_train]),
)

print("去重：", {key: round(float(value), 4) for key, value in deduplicated_metrics.items()})
print("镜像重复：", {key: round(float(value), 4) for key, value in mirror_metrics.items()})
```

去重模型 accuracy/NLL/Brier 为 0.7800/0.4730/0.3045；镜像重复模型为 0.7764/0.5522/0.3255。相关列没有带来可重复的准确率收益，却让概率质量明显恶化。

<MlQuiz
  kind="结果预测"
  question="把同一只物理报警复制成两列后，朴素贝叶斯常出现什么现象？"
  :options="['分类可能几乎不变，但后验更极端、校准更差', '后验必然回到 0.5', 'Laplace α 自动识别两列同源', '模型会严格拒绝重复列']"
  :answer="0"
  explanation="条件独立分解会把两列的 log evidence 分别累计。去重、分组或显式依赖模型需要在输入协议中处理。"
/>

## 11. Gaussian NB、LDA、QDA 与逻辑回归的关系

### 11.1 Gaussian NB 等于对角 QDA

对连续特征，Gaussian NB 使用

$$
p(\mathbf x\mid y=c)
=\prod_{j=1}^{d}\mathcal N(x_j;\mu_{cj},\sigma_{cj}^2).
$$

这正是协方差为

$$
\boldsymbol\Sigma_c
=\operatorname{diag}(\sigma_{c1}^2,\ldots,\sigma_{cd}^2)
$$

的 QDA。

```python
gaussian_nb_model = models["Gaussian NB"]
gaussian_nb_query = X[val_mask][:20]
matrix_scores = gda_log_scores(gaussian_nb_model, gaussian_nb_query)

manual_scores = np.empty_like(matrix_scores)
for c in gaussian_nb_model["classes"]:
    variances = np.diag(gaussian_nb_model["covs"][c])
    manual_scores[:, c] = (
        np.log(gaussian_nb_model["priors"][c])
        - 0.5 * np.sum(np.log(variances))
        - 0.5 * np.sum(
            (gaussian_nb_query - gaussian_nb_model["means"][c]) ** 2 / variances,
            axis=1,
        )
    )

assert np.allclose(matrix_scores, manual_scores)
print("Gaussian NB / 对角 QDA 最大分数差：", np.max(np.abs(matrix_scores - manual_scores)))
```

### 11.2 模型族不是按名字排等级

| 模型 | 类条件形状 | 边界能力 | 样本效率 | 主要审计项 |
|---|---|---|---|---|
| LDA | 每类 Gaussian，共享全协方差 | 多类线性 | 较高 | 共协方差是否合理、类内离群点 |
| QDA | 每类 Gaussian，类别专属全协方差 | 二次 | 较低 | 每类样本量、条件数、收缩 |
| Gaussian NB | 每类每列一维 Gaussian | 轴对齐二次 | 高 | 条件相关、单变量厚尾 |
| Bernoulli NB | 二值出现/缺席 | 特征空间中的线性 log odds | 高 | absence 语义、重复报警 |
| Multinomial NB | 词计数 | 计数空间中的线性 log odds | 高 | 词表、长度、平滑、同源词 |
| 逻辑/Softmax 回归 | 直接条件模型 | 线性 | 视正则而定 | 特征尺度、分离、校准 |

若 Gaussian 假设接近真实且样本少，GDA 可能利用结构提高样本效率；若类条件分布明显错设且样本充足，条件模型通常更稳健。正确做法不是凭模型名字判断，而是对同一分组切分报告 discrimination、calibration、稳定性和运行域。

### 11.3 先验、阈值和动作代价不是同一个旋钮

- 训练先验 $\hat\pi_c$ 是数据生成模型的一部分；
- 部署先验用于 label shift 修正；
- 动作阈值或损失矩阵把后验转换为处置；
- 类别权重若直接加入训练，会改变拟合目标。

把四者都叫“调阈值”会让概率失去可追溯语义。生产工件必须分别版本化。

## 数值验证与拒绝门禁

### 12.1 充分统计量可以流式累计

Multinomial NB 的训练只需每类样本数和每类词计数。批量与流式累计必须严格等价。

```python
train_token_counts = token_counts[train_mask]
train_labels = y[train_mask]

batch_token_sum = np.vstack([
    train_token_counts[train_labels == c].sum(axis=0)
    for c in range(3)
])
stream_token_sum = np.zeros_like(batch_token_sum)
stream_class_count = np.zeros(3, dtype=int)

for row, label in zip(train_token_counts, train_labels):
    stream_token_sum[label] += row
    stream_class_count[label] += 1

assert np.array_equal(batch_token_sum, stream_token_sum)
assert np.array_equal(stream_class_count, np.bincount(train_labels, minlength=3))
print("流式类别计数：", stream_class_count.tolist())
```

### 12.2 确定性、无效输入与概率合同

```python
replay_a = fit_gda(X[train_mask], y[train_mask], "class", best_shrinkage)
replay_b = fit_gda(X[train_mask], y[train_mask], "class", best_shrinkage)
replay_probability_a = gda_predict_proba(replay_a, X[test_mask])
replay_probability_b = gda_predict_proba(replay_b, X[test_mask])

assert np.array_equal(replay_probability_a, replay_probability_b)
assert np.all(np.isfinite(replay_probability_a))
assert np.allclose(replay_probability_a.sum(axis=1), 1.0)

for bad_counts, bad_alpha in [
    (np.array([[-1, 2]]), 1.0),
    (np.array([[1, 2]]), 0.0),
]:
    try:
        fit_multinomial_nb(bad_counts, np.array([0]), bad_alpha)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid NB input should be rejected")

print("确定性重放、概率和、有限值与无效输入门禁通过")
```

上线前还应覆盖：

- 空类别、单样本类别、全常数列；
- 非数值、无穷值、负计数、小数计数是否符合事件合同；
- 协方差非对称、非正定、极端条件数；
- 词表列顺序变化、重复 token、OOV 暴增；
- 全部类别 log score 极小的稳定归一化；
- train/serve 预处理版本和特征单位一致；
- 批次顺序打乱后输出不变；
- float32/float64 与跨语言实现的概率金丝雀。

### 12.3 可选库对照

下面代码只用于与成熟库交叉检查，不替代本章从零实现；仓库标准验证环境不为它新增依赖。

```python
# 不执行：可选 scikit-learn 对照
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB

lda_library = LinearDiscriminantAnalysis().fit(X[train_mask], y[train_mask])
qda_library = QuadraticDiscriminantAnalysis(reg_param=best_shrinkage).fit(X[train_mask], y[train_mask])
nb_library = MultinomialNB(alpha=best_alpha).fit(token_counts[train_mask], y[train_mask])

print(lda_library.predict_proba(X[test_mask])[:2])
print(qda_library.predict_proba(X[test_mask])[:2])
print(nb_library.predict_proba(token_counts[test_mask])[:2])
```

## 复杂度与工程边界

### 13.1 时间和空间预算

设训练样本数 $n$、连续维数 $d$、类别数 $C$、词表大小 $V$、日志非零词数 $s$：

| 模型 | 训练时间主项 | 单样本推理 | 模型存储主项 |
|---|---:|---:|---:|
| LDA | $O(nd^2+d^3)$ | $O(Cd^2)$，预白化后可降为 $O(Cd)$ | $O(d^2+Cd)$ |
| QDA | $O(nd^2+Cd^3)$ | $O(Cd^2)$ | $O(Cd^2)$ |
| Gaussian NB | $O(nd)$ | $O(Cd)$ | $O(Cd)$ |
| Multinomial NB | $O(\operatorname{nnz}(X))$ | $O(Cs)$ | $O(CV)$ |

```python
n_budget, d_budget, c_budget, v_budget = 2_000_000, 64, 12, 50_000
qda_covariance_bytes = c_budget * d_budget * d_budget * 8
nb_parameter_bytes = c_budget * v_budget * 8
qda_score_multiply_adds = c_budget * d_budget * d_budget
nb_dense_score_multiply_adds = c_budget * v_budget

print("QDA 协方差 MiB：", round(qda_covariance_bytes / 2**20, 2))
print("NB 稠密词参数 MiB：", round(nb_parameter_bytes / 2**20, 2))
print("QDA 单样本二次型乘加代理量：", qda_score_multiply_adds)
print("NB 稠密单样本乘加代理量：", nb_dense_score_multiply_adds)
```

Multinomial NB 的实际推理应利用稀疏日志，只遍历出现词；QDA 应缓存 Cholesky 分解与 log determinant。若模型服务每次都显式求逆或扫描 50,000 个零计数词，复杂度预算就没有落地。

### 13.2 生产工件合同

一个可上线审计的模型包至少包含：

1. `feature_schema_version`：连续列顺序、单位、量程、缺失协议；
2. `vocabulary_version`：tokenizer、规范化、词表、`<UNK>` 与重复事件规则；
3. `class_order`：类别编号、显示名称、动作映射；
4. `priors`、`means`、`covariances` 或 `log_token_probabilities`；
5. `covariance_mode`、收缩目标与冻结 $\lambda$；
6. 平滑 $\alpha$ 与其选择数据；
7. 训练 campaign 清单、切分摘要和数据时间窗；
8. 校准方法、动作损失矩阵和冻结阈值；
9. 训练/验证/测试指标、逐类召回与概率分箱；
10. 数值金丝雀、拒绝门禁和回滚版本。

### 13.3 在线监控不是只看预测占比

| 监控层 | GDA | Naive Bayes |
|---|---|---|
| 输入 | 均值、协方差、范围、缺失模式 | OOV、日志长度、词频、重复事件率 |
| 模型 | Mahalanobis 距离尾部、log determinant、条件数 | 单词 log likelihood ratio、熵、极端后验比例 |
| 结果 | 后验分布、逐类路由负载、人工复核一致率 | 同左，另加校准与镜像证据审计 |
| 漂移定位 | $p(\mathbf x\mid y)$ 与 $p(y)$ 分开看 | 类条件词频与类别先验分开看 |

没有延迟真值时，只能发现“输入或输出变了”，不能宣称模型性能仍然正常。维修复核标签回来后，必须按 campaign 和时间窗重算校准、NLL、逐类召回和动作成本。

<MlQuiz
  kind="错误诊断"
  question="Multinomial NB 上线后 OOV 比例突然上升，但已知词上的后验仍很稳定。最合理的处置是什么？"
  :options="['触发词表/schema 漂移告警并检查日志解析版本', '把所有 OOV 自动映射为最危险类别', '增大 α 就视为问题解决', '只要 accuracy 尚未回传就忽略']"
  :answer="0"
  explanation="OOV 是输入合同变化，不是固定词表内的零计数。平滑不能为模型未定义的列恢复语义。"
/>

## 14. 章节练习

### 练习 1：手推 LDA 的两类边界

给定

$$
\boldsymbol\mu_0=\begin{bmatrix}0\\1\end{bmatrix},
\quad
\boldsymbol\mu_1=\begin{bmatrix}2\\2\end{bmatrix},
\quad
\boldsymbol\Sigma=\begin{bmatrix}2&1\\1&2\end{bmatrix},
\quad
\pi_0=0.7,\ \pi_1=0.3.
$$

1. 计算 $\mathbf w=\boldsymbol\Sigma^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_0)$；
2. 计算截距 $b$；
3. 写出 $\mathbf w^\top\mathbf x+b=0$；
4. 把先验改成 $\pi_0=\pi_1=0.5$，说明边界如何平移；
5. 验证法向量不因先验变化而旋转。

### 练习 2：找出 QDA 的二次矩阵

对两类一般 QDA，把 $\delta_1(\mathbf x)-\delta_0(\mathbf x)$ 写成

$$
\mathbf x^\top\mathbf A\mathbf x+\mathbf b^\top\mathbf x+c.
$$

明确给出 $\mathbf A,\mathbf b,c$，并回答：

- 什么条件使 $\mathbf A=\mathbf0$？
- 仅让 $|\boldsymbol\Sigma_0|=|\boldsymbol\Sigma_1|$ 是否足以得到线性边界？
- 两个对角协方差不相同时，为什么仍可能有轴对齐二次边界？

### 练习 3：Bernoulli 与 Multinomial 不能互换

考虑两份日志：

- A：`1x_peak` 出现 1 次；
- B：`1x_peak` 出现 12 次。

分别在 Bernoulli 和 Multinomial 事件模型下写出该词对 log score 的贡献。再构造一个“词是否缺席很重要”的例子，说明 Bernoulli NB 为什么必须保留 $x_j=0$ 的项。

### 练习 4：30–60 分钟实现——带拒绝门禁的分组 GDA

#### 目标

在不使用机器学习库的前提下，实现 LDA、QDA 与对角 Gaussian NB，并用 campaign 分组验证冻结协方差合同和收缩参数。

#### 必做步骤

1. 接收 `X: (n,d)`、`y: (n,)`、`group: (n,)`；
2. 在训练折估计先验、均值和三种协方差；
3. 使用 `slogdet` 与 `solve`，禁止显式 `inv`；
4. 对 $\lambda\in\{0,0.05,0.2,0.5,1\}$ 做验证 NLL 选择；
5. 若任一最小特征值低于冻结容差，拒绝未收缩模型；
6. 输出 probability shape、行和、accuracy、NLL、Brier、逐类召回；
7. 记录 group 交集断言、随机种子和配置；
8. 构造 $d\ge n_c$ 的反例，确认未收缩 QDA 被拒绝。

```python
# TODO：练习骨架，不执行
def fit_grouped_gda(X, y, groups, covariance_mode, shrinkage):
    # 1. 校验形状、有限值、类别和 group
    # 2. 只用训练 group 拟合
    # 3. 估计 MLE 参数并做收缩
    # 4. 检查最小特征值和条件数
    # 5. 返回冻结顺序的模型工件
    ...

def score_grouped_gda(model, X_query):
    # 使用 slogdet + solve，返回稳定后验
    ...
```

#### 参考验收

- 共享协方差下，两类 score difference 对二次基的系数范数小于 $10^{-10}$；
- 类别专属协方差下，手工二次式与实现最大误差小于 $10^{-10}$；
- probability 行和误差小于 $10^{-12}$；
- 查询顺序打乱再复原后，概率逐元素一致；
- 同一 group 不跨训练、验证、测试；
- 测试集只运行一次，不参与模型或 $\lambda$ 选择。

### 练习 5：相关报警的校准审计

在本章 4 个潜在报警的模拟上，把镜像一致概率从 0.5 调到 0.99，每个设置重复 30 次：

1. 比较去重模型和镜像模型的 accuracy、NLL、Brier；
2. 画出置信度分箱；
3. 记录“accuracy 变化小但 NLL 变差”的区域；
4. 对镜像列做物理来源分组，再重新训练；
5. 解释为什么增大 Laplace $\alpha$ 不是通用的依赖修复方法。

## 15. 本章验收清单

完成本章后，应能逐项给出证据：

- [ ] 能从联合分布和 Bayes 规则写出生成式判别分数；
- [ ] 能解释 determinant、Mahalanobis 距离和类别先验的不同作用；
- [ ] 能完整推导 LDA 共同二次项为何抵消；
- [ ] 能写出 QDA 两类分数差的二次矩阵；
- [ ] 能区分共享、类别专属、对角协方差的参数量和适用边界；
- [ ] 能用 `solve`、`slogdet` 和稳定 softmax 实现 GDA；
- [ ] 能检测小样本高维协方差奇异并用验证集冻结收缩；
- [ ] 能区分 Gaussian、Bernoulli、Categorical、Multinomial NB；
- [ ] 能推导 additive smoothing 并保持词概率和为 1；
- [ ] 能区分固定词表零计数与 OOV schema 漂移；
- [ ] 能用重复抽样证明相关证据会恶化概率校准；
- [ ] 能把先验修正、动作阈值和类别权重分别版本化；
- [ ] 能按 campaign 隔离训练、验证和测试；
- [ ] 能明确模型只做复检路由，不自动放行或修改联锁。

## 16. 小结：边界是生成假设的投影

GDA、LDA、QDA 和朴素贝叶斯的共同主线不是“都用了 Bayes 公式”，而是先把类别内部的数据形状说清楚：

- LDA 认为各类共享同一协方差单位，因此共同二次项抵消；
- QDA 允许每类拥有自己的概率椭圆，用更多参数换取二次边界；
- Gaussian NB 把椭圆限制为轴对齐，用条件独立降低估计成本；
- Bernoulli 和 Multinomial NB 把离散证据写成可累计的 log likelihood ratio；
- 平滑修复零计数，但不修复分布错设、相关证据或输入 schema 漂移。

工业系统真正需要冻结的，不只是一个 `predict` 函数，而是一整套生成假设、数据独立单位、数值求解、先验版本、词表合同、校准证据和动作边界。

下一章将移除“根因标签已知”的条件：当高斯分量存在、但每条样本来自哪个分量不可见时，期望最大化算法（EM）怎样用责任度在隐变量与参数之间交替更新？

## 来源与许可

本章在以下本地资料的知识边界内独立重写：

1. Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 353–358 页（§9.2–§9.2.5）：GDA、QDA/LDA 判别式、共享/类别专属/对角协方差、模型拟合、收缩与最近中心解释。
2. Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 362–367 页（§9.3–§9.4）：Gaussian/Bernoulli/Categorical Naive Bayes、参数估计、Bayesian smoothing、与逻辑回归的联系及生成式/判别式交换。
3. Stanford CS229 Lecture Notes，PDF 第 36–41 页（§4.1）：多元 Gaussian、GDA 参数估计、线性后验与逻辑回归对照。
4. Stanford CS229 Lecture Notes，PDF 第 42–49 页（§4.2）：条件独立、Bernoulli/Multinomial 事件模型、Laplace smoothing 与零计数。
5. UC Berkeley CS 189/289A Lecture 11，PDF 第 18–50 页：Gaussian 类条件几何、共享/独立协方差、白化、LDA/QDA 边界与 Naive Bayes 入口。

本章数控主轴案例的 54 个教学 campaign、324 条合成记录、三类根因、连续量测、报警词表、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实企业、主轴、轴承、传感器、振动标准、装配参数、测试规范、维修策略或安全阈值。PML 资料按其 CC BY-NC-ND 4.0 边界仅用于知识核对，不复用或改编原书图表、数据、案例、代码和表达。

<MlLessonProgress
  lesson-id="23-generative-classifiers"
  next-href="/courses/machine-learning/24-em"
  next-label="下一章：期望最大化算法（EM）"
/>
