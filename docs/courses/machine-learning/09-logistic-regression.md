---
course: machine-learning
edition: industrial
title: 逻辑回归
description: 以焊缝在线缺陷分流为贯穿案例，从 log odds、条件最大熵与 Bernoulli 似然推导 Sigmoid、交叉熵、梯度、Hessian、凸性、Newton/IRLS、完全分离和正则化。
outline: [2, 3]
---

<MlLessonHero
  number="09"
  title="逻辑回归"
  question="一条线性证据怎样成为可信概率，又怎样在完全分离、类别加权和部署代价下保持可解释？"
  duration="建议 430–520 分钟"
  :outcomes="[
    '从 log odds、条件最大熵和 Bernoulli 似然三条路径得到 Sigmoid 与交叉熵',
    '逐项推导梯度、Hessian、凸性、Lipschitz 上界和 Newton/IRLS 更新',
    '诊断完全分离、正则化尺度、类别加权、校准与阈值部署失效'
  ]"
/>

上一章把“预测概率”和“采取动作”分开：模型估计 $p(Y=1\mid x)$，决策层再根据损失选择动作。本章只解决前半件事，但要求做到工业可用：

> 从一组传感器特征估计可审计的二分类概率，并明确这个概率何时不能直接进入放行决策。

逻辑回归不是“把线性回归的输出压进 0 到 1”。它是一个完整的条件概率模型：

$$
Y_i\mid x_i\sim\operatorname{Bernoulli}(p_i),
\qquad
\operatorname{logit}(p_i)=\log\frac{p_i}{1-p_i}=\tilde x_i^T\beta.
$$

模型形式、训练目标、优化性质和部署语义必须同时成立。只记住 Sigmoid 图形，无法解释系数为什么发散、类别权重为什么破坏原始校准，也无法判断优化器的 `success` 是否可信。

## 贯穿案例：焊缝在线缺陷分流

某自动焊接工位在每条焊缝完成后采集电流波形、声发射、行走速度和表面视觉信号。逻辑回归输出“需要离线无损复核”的概率，随后由成本阈值和强制工艺规则决定：

- **自动放行**：概率低且不触发硬规则；
- **离线复核**：概率超过当前成本阈值；
- **强制隔离**：由独立工艺规则触发，不允许模型覆盖。

本章把“存在内部缺陷或需要离线复核”记为 $Y=1$。它不是在预测某种抽象类别，而是在估计一个有明确参考标签、时间窗口和处置后果的条件概率。

| 对象 | 本章教学定义 | 工业系统必须补充 |
|---|---|---|
| 观测单元 | 一条焊缝的稳定焊接窗口 | 返修后重焊是否算新样本 |
| 正类 | 离线相控阵检测确认需复核 | 缺陷等级、盲区与检测下限 |
| 特征截止 | 焊缝完成后、处置前 | 禁止使用返修结果和终检结论 |
| 分组键 | 生产批次 | 同工件多焊缝、设备和班次层级 |
| 概率用途 | 进入复核分流 | 不覆盖法规和强制工艺规则 |
| 目标分布 | 新批次同工艺窗口 | 新材料、新焊机和新探头需重新验收 |

### 数据生成与标签可见性

教学数据由 48 个生产批次、每批 50 条焊缝组成。批次效应会同时影响特征和缺陷率，因此随机拆散单条焊缝会泄漏批次指纹。本章固定前 32 批训练、后 8 批验证、最后 8 批测试；任何阈值与超参数只在验证批次选择。

```python
import numpy as np

rng = np.random.default_rng(20260805)
batch_count = 48
welds_per_batch = 50
batch_id = np.repeat(np.arange(batch_count), welds_per_batch)
batch_shift = rng.normal(0, 0.32, size=batch_count)

current_shift = rng.normal(batch_shift[batch_id], 1.0)
acoustic_anomaly = rng.normal(0.35 * current_shift + 0.5 * batch_shift[batch_id], 0.9)
travel_speed = rng.normal(-0.2 * batch_shift[batch_id], 1.0)
surface_texture = rng.normal(0.25 * acoustic_anomaly, 1.0)
interaction = current_shift * acoustic_anomaly

raw_features = np.column_stack([
    current_shift,
    acoustic_anomaly,
    travel_speed,
    interaction,
])
true_logit = (
    -3.10
    + 0.85 * current_shift
    + 1.10 * acoustic_anomaly
    - 0.55 * travel_speed
    + 0.32 * interaction
    + 0.55 * batch_shift[batch_id]
)
true_probability = 1 / (1 + np.exp(-true_logit))
label = rng.binomial(1, true_probability)

train_mask = batch_id < 32
validation_mask = (batch_id >= 32) & (batch_id < 40)
test_mask = batch_id >= 40

print("训练/验证/测试：", train_mask.sum(), validation_mask.sum(), test_mask.sum())
print("教学总体复核率：", round(label.mean(), 4))
print("三个集合的批次数：",
      np.unique(batch_id[train_mask]).size,
      np.unique(batch_id[validation_mask]).size,
      np.unique(batch_id[test_mask]).size)
```

这段合成代码不是“真实缺陷规律”。它刻意加入批次随机效应和交互项，用来验证分组切分、非线性特征、正则化和部署审计。

<MlQuiz kind="数据边界" question="把同一生产批次的焊缝随机分到训练集和测试集，最主要的风险是什么？" :options="['测试样本一定减少', '批次共享的设备与材料信号进入两侧，测试结果高估新批次泛化', '逻辑回归不能使用批次数据', 'Sigmoid 会变成线性函数']" :answer="1" explanation="同批样本并非独立的新生产环境。若部署目标是新批次，切分单元也应至少到批次层级。" />

## 1. 为什么不直接把线性分数当概率

设 $\tilde x=[1,x_1,\ldots,x_d]^T$ 已包含截距列。线性分数

$$
z=\tilde x^T\beta
$$

可以取任意实数，而概率必须位于 $[0,1]$。裁剪 $z$ 不是概率模型：

$$
\hat p=\min(1,\max(0,z))
$$

在区间外梯度为零，在折点不可导，也没有自然的 Bernoulli 似然解释。更重要的是，线性结构真正适合的对象不是 $p$，而是无界的 **log odds**。

概率 $p\in(0,1)$ 的 odds 是

$$
\operatorname{odds}(p)=\frac{p}{1-p}\in(0,\infty),
$$

取对数后覆盖整个实数轴：

$$
\operatorname{logit}(p)=\log\frac{p}{1-p}\in\mathbb R.
$$

于是可以令

$$
\log\frac{p(x)}{1-p(x)}=\tilde x^T\beta=z.
$$

反解：

$$
\begin{aligned}
\frac p{1-p}&=e^z,\\
p&=e^z(1-p),\\
p(1+e^z)&=e^z,\\
\boxed{p=\frac{e^z}{1+e^z}=\frac1{1+e^{-z}}=\sigma(z)}.
\end{aligned}
$$

因此逻辑回归“线性”的是 log odds，不是概率曲线。

### 系数解释必须带单位

保持其他特征不变，$x_j$ 增加一单位：

$$
\operatorname{logit}(p')-\operatorname{logit}(p)=\beta_j,
$$

所以 odds 乘以

$$
\frac{\operatorname{odds}(p')}{\operatorname{odds}(p)}=e^{\beta_j}.
$$

若 $x_j$ 已标准化，一单位表示训练分布中的一个标准差；若仍使用安培、毫米或分贝，解释必须写清物理单位。$e^{\beta_j}$ 是条件 odds ratio，不自动等于因果效应。

<MlQuiz kind="系数解释" question="标准化后的声发射特征系数为 0.7。其他输入不变时增加一个训练标准差，正确解释是什么？" :options="['概率固定增加 0.7', 'log odds 增加 0.7，odds 乘以 exp(0.7)', '缺陷率必然翻倍', '这是声发射的因果效应']" :answer="1" explanation="概率增量依赖当前基准概率；系数给出条件 log-odds 变化。观察数据中的关联也不能自动解释为因果。" />

<div class="ml-stop"><strong>停下来推一推 1</strong> 当 $p=0.2$ 时 odds 为多少？若 log odds 增加 $\log 2$，新概率不是 $0.4$，而是把 odds 从 $0.25$ 加倍到 $0.5$，得到 $p=1/3$。</div>

## 2. Sigmoid 的三条来源

### 2.1 log odds 的逆函数

上一节已经得到最直接的路径：

$$
\text{线性 log odds}\quad\Longrightarrow\quad p=\sigma(\tilde x^T\beta).
$$

Sigmoid 不是任意“压缩函数”，而是 logit link 的逆函数。若换成 probit 等 link，就得到不同的二项响应模型。

### 2.2 Bernoulli 广义线性模型

Bernoulli 分布属于指数族：

$$
p(y\mid\eta)=\exp\{y\eta-\log(1+e^\eta)\},
\qquad y\in\{0,1\}.
$$

自然参数 $\eta$ 与均值的关系是

$$
\mathbb E[Y\mid x]=\frac{e^\eta}{1+e^\eta}=\sigma(\eta).
$$

选 canonical link $\eta=\tilde x^T\beta$，就得到逻辑回归。这个视角解释了为什么后面的梯度会简化为“预测减标签”。

### 2.3 条件最大熵直觉

对训练点 $x_i$，考虑在所有二分类条件分布 $q_i(y)$ 中最大化总条件熵：

$$
\max_q -\sum_i\sum_{y\in\{0,1\}}q_i(y)\log q_i(y),
$$

满足归一化和特征矩约束：

$$
\sum_y q_i(y)=1,
\qquad
\sum_i q_i(1)x_i=\sum_i y_i x_i.
$$

对矩约束引入乘子 $\beta$。Lagrange 一阶条件给出

$$
q_i(y)\propto\exp(y\,\beta^Tx_i).
$$

归一化后：

$$
q_i(1)=\frac{\exp(\beta^Tx_i)}{1+\exp(\beta^Tx_i)}
=\sigma(\beta^Tx_i).
$$

最大熵不是说“现实标签最随机”，而是：在只强制模型匹配指定特征矩的前提下，不额外捏造更多结构。特征集合改变，约束和最终模型也会改变。

<MlQuiz kind="原理辨析" question="条件最大熵推导中，Sigmoid 形式来自什么？" :options="['先把概率裁剪到 0 和 1', '熵目标、归一化约束与线性特征矩约束的 Lagrange 解', '把平方误差改名为交叉熵', '要求训练数据完全可分']" :answer="1" explanation="指数形式来自最大熵的驻点条件，二分类归一化产生 Sigmoid。最大熵仍依赖所选特征矩约束。" />

## 3. Bernoulli 似然为什么产生交叉熵

对独立训练样本，

$$
Y_i\mid x_i\sim\operatorname{Bernoulli}(p_i),
\qquad p_i=\sigma(z_i),
\qquad z_i=\tilde x_i^T\beta.
$$

单样本概率质量函数可把两类统一为

$$
p(y_i\mid x_i,\beta)=p_i^{y_i}(1-p_i)^{1-y_i}.
$$

联合 log-likelihood：

$$
\ell(\beta)=\sum_{i=1}^n
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

最小化平均负 log-likelihood：

$$
\boxed{
J(\beta)=
-\frac1n\sum_{i=1}^n
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
}.
$$

这正是二元交叉熵。若 $y_i=1$，损失为 $-\log p_i$；若 $y_i=0$，损失为 $-\log(1-p_i)$。

### 为什么交叉熵能学习概率

设某个 $x$ 的真实正类概率为 $\eta=P(Y=1\mid x)$，模型报告 $q$。条件期望 log loss 为

$$
R(q\mid x)=-\eta\log q-(1-\eta)\log(1-q).
$$

对 $q$ 求导：

$$
\frac{\partial R}{\partial q}
=-\frac{\eta}{q}+\frac{1-\eta}{1-q}
=\frac{q-\eta}{q(1-q)}.
$$

唯一驻点是 $q=\eta$，二阶导为正。交叉熵因此是严格 proper scoring rule：在模型容量和数据足够的理想条件下，诚实报告真实概率能最小化期望损失。

0-1 loss 只关心阈值两侧，不能区分 $0.51$ 与 $0.99$ 的置信度；这就是概率训练不能只看准确率的原因。

### 用 logit 直接算稳定损失

利用

$$
\log\sigma(z)=z-\log(1+e^z),
\qquad
\log[1-\sigma(z)]=-\log(1+e^z),
$$

单样本损失可写为

$$
\boxed{\ell_i=\operatorname{softplus}(z_i)-y_i z_i},
\qquad
\operatorname{softplus}(z)=\log(1+e^z).
$$

工程实现不应先算概率再做 `log(p)`；大绝对值 logit 会让概率舍入为 0 或 1。`np.logaddexp(0, z)` 能稳定计算 softplus。

```python
def stable_sigmoid(logits):
    logits = np.asarray(logits, dtype=float)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1 + exp_logits)
    return result

def binary_log_loss_from_logits(logits, targets, sample_weight=None):
    losses = np.logaddexp(0, logits) - targets * logits
    if sample_weight is None:
        return losses.mean()
    return np.average(losses, weights=sample_weight)

extreme_logits = np.array([-1000.0, -8.0, 0.0, 8.0, 1000.0])
print("稳定概率：", stable_sigmoid(extreme_logits))
print("稳定损失：", binary_log_loss_from_logits(
    extreme_logits, np.array([0, 0, 1, 1, 1])
))
```

<MlQuiz kind="损失判断" question="正类样本的 logit 从 4 增加到 8，准确率没有变化。为什么交叉熵仍会变化？" :options="['交叉熵使用完整预测概率而不只使用阈值标签', '交叉熵等于准确率', 'logit 大于 1 非法', '正类样本不进入似然']" :answer="0" explanation="两者都被判为正类，但模型给真实类别的概率不同。交叉熵对概率质量进行评分。" />

## 4. 梯度、Hessian 与凸性

### 4.1 梯度中的 Sigmoid 导数为什么消失

Sigmoid 导数：

$$
\sigma'(z)=\sigma(z)[1-\sigma(z)]=p(1-p).
$$

单样本损失对 logit 求导：

$$
\begin{aligned}
\frac{d\ell_i}{dz_i}
&=-\left[
y_i\frac{p_i(1-p_i)}{p_i}
 +(1-y_i)\frac{-p_i(1-p_i)}{1-p_i}
\right]\\
&=-[y_i(1-p_i)-(1-y_i)p_i]\\
&=p_i-y_i.
\end{aligned}
$$

链式法则 $\nabla_\beta z_i=\tilde x_i$：

$$
\nabla_\beta\ell_i=(p_i-y_i)\tilde x_i.
$$

平均目标的矩阵梯度：

$$
\boxed{\nabla J(\beta)=\frac1n\tilde X^T(p-y)}.
$$

它有清晰的“矩匹配”解释。无正则最优点若存在，则

$$
\tilde X^Tp=\tilde X^Ty.
$$

模型预测的特征加权正类总量与观测总量匹配。

### 4.2 Hessian

令

$$
W=\operatorname{diag}\left(p_i(1-p_i)\right).
$$

则

$$
\boxed{\nabla^2J(\beta)=\frac1n\tilde X^TW\tilde X}.
$$

对任意向量 $v$：

$$
v^T\nabla^2J(\beta)v
=\frac1n\sum_i p_i(1-p_i)(\tilde x_i^Tv)^2\ge0.
$$

因此平均负 log-likelihood 是凸函数。要说“严格凸”，还需要 $\tilde X$ 满列秩且参数有限，使 $0<p_i<1$。即使严格凸，也不自动保证有限极小值存在；完全分离正是反例。

### 4.3 全局曲率上界

因为

$$
0<p_i(1-p_i)\le\frac14,
$$

所以

$$
\nabla^2J(\beta)\preceq\frac1{4n}\tilde X^T\tilde X.
$$

梯度 Lipschitz 常数可取

$$
L\le\frac{\lambda_{\max}(\tilde X^T\tilde X)}{4n}
$$

；加入强度 $\lambda$ 的 L2 后再加 $\lambda$。固定步长梯度下降可据此得到安全量级，但实践仍应监控目标下降。

### 数值构造：只用训练批次拟合标准化

截距列不标准化，也通常不惩罚。均值和标准差只能在训练批次估计，再原样作用于验证和测试。

```python
feature_mean = raw_features[train_mask].mean(axis=0)
feature_scale = raw_features[train_mask].std(axis=0)
feature_scale = np.where(feature_scale > 0, feature_scale, 1.0)

standardized = (raw_features - feature_mean) / feature_scale
design = np.column_stack([np.ones(len(standardized)), standardized])
X_train, y_train = design[train_mask], label[train_mask]
X_validation, y_validation = design[validation_mask], label[validation_mask]
X_test, y_test = design[test_mask], label[test_mask]
group_train = batch_id[train_mask]

penalty_mask = np.ones(X_train.shape[1])
penalty_mask[0] = 0.0

def objective_gradient_hessian(
    beta, X, y, regularization=0.0, sample_weight=None
):
    logits = X @ beta
    probability = stable_sigmoid(logits)
    if sample_weight is None:
        normalized_weight = np.full(len(y), 1 / len(y))
    else:
        normalized_weight = sample_weight / np.sum(sample_weight)
    loss = np.sum(
        normalized_weight * (np.logaddexp(0, logits) - y * logits)
    )
    loss += 0.5 * regularization * np.sum(penalty_mask * beta**2)
    residual = normalized_weight * (probability - y)
    gradient = X.T @ residual + regularization * penalty_mask * beta
    curvature = normalized_weight * probability * (1 - probability)
    hessian = X.T @ (curvature[:, None] * X)
    hessian += regularization * np.diag(penalty_mask)
    return loss, gradient, hessian
```

### 有限差分不是装饰

推导中的符号、平均因子和正则化常数很容易错。上线前至少用中心差分检查梯度，并检查 Hessian 对称与半正定。

```python
beta_probe = np.array([-1.4, 0.35, -0.2, 0.15, 0.1])
probe_loss, analytic_gradient, analytic_hessian = objective_gradient_hessian(
    beta_probe, X_train, y_train, regularization=0.17
)

epsilon = 1e-6
numeric_gradient = np.zeros_like(beta_probe)
for column in range(len(beta_probe)):
    step = np.zeros_like(beta_probe)
    step[column] = epsilon
    plus = objective_gradient_hessian(
        beta_probe + step, X_train, y_train, regularization=0.17
    )[0]
    minus = objective_gradient_hessian(
        beta_probe - step, X_train, y_train, regularization=0.17
    )[0]
    numeric_gradient[column] = (plus - minus) / (2 * epsilon)

print("梯度最大误差：", np.max(np.abs(analytic_gradient - numeric_gradient)))
print("Hessian 最小特征值：", np.linalg.eigvalsh(analytic_hessian).min())
```

<MlQuiz kind="凸性条件" question="哪项足以推出有限参数处的 Hessian 正定？" :options="['训练准确率为 100%', '设计矩阵满列秩且所有 Sigmoid 概率严格位于 0 与 1 之间', '样本量大于 1', '所有系数为正']" :answer="1" explanation="W 的对角元为正时，XᵀWX 是否正定取决于 X 是否没有非零零空间方向。正定仍不单独保证有限极小值存在。" />

<div class="ml-stop"><strong>停下来推一推 2</strong> 若所有 logit 绝对值都很大，$p_i(1-p_i)$ 会接近 0。梯度可能仍指向下降方向，但 Hessian 的数据曲率会退化，这就是极端置信和完全分离时 Newton 系统病态的根源。</div>

<LogitEvidenceConsoleLab />

<MlQuiz kind="边界辨析" question="把所有特征权重和截距同时乘以 3，0.5 决策边界与概率会怎样？" :options="['边界不变，但远离边界的概率更接近 0 或 1', '边界和概率都完全不变', '边界一定平移 3 个单位', '概率全部变成 0.5']" :answer="0" explanation="wᵀx+b=0 的零集合不变，但 logit 绝对值放大，Sigmoid 更陡。排序、边界与校准不是同一对象。" />

## 5. 从梯度下降到 Newton 与 IRLS

### 5.1 梯度下降

平均目标带 L2 惩罚：

$$
J_\lambda(\beta)=
\frac1n\sum_i\ell_i(\beta)
+\frac\lambda2\lVert P\beta\rVert_2^2,
$$

其中 $P$ 的截距位置为 0，其余为 1。梯度下降：

$$
\beta_{t+1}=\beta_t-\eta\nabla J_\lambda(\beta_t).
$$

每步主要成本是计算 $\tilde X\beta$ 和 $\tilde X^T(p-y)$，稠密数据为 $O(nd)$。它不需要保存 $d\times d$ Hessian，适合大样本或高维稀疏特征。

### 5.2 Newton 更新

带惩罚梯度和 Hessian：

$$
g=\frac1n\tilde X^T(p-y)+\lambda P\beta,
$$

$$
H=\frac1n\tilde X^TW\tilde X+\lambda P.
$$

不要显式求逆。解线性系统

$$
H\Delta=-g,
\qquad
\beta_{\text{new}}=\beta+\alpha\Delta.
$$

纯 Newton 常写 $\alpha=1$，但工业实现应加入阻尼或回溯线搜索，确保目标实际下降。特征尺度不良、近分离和低曲率都会让无阻尼步不可靠。

```python
def fit_logistic_newton(
    X,
    y,
    regularization=0.0,
    sample_weight=None,
    maximum_iterations=80,
    tolerance=1e-9,
):
    beta = np.zeros(X.shape[1])
    history = []
    for iteration in range(maximum_iterations):
        loss, gradient, hessian = objective_gradient_hessian(
            beta, X, y, regularization, sample_weight
        )
        history.append(loss)
        if np.linalg.norm(gradient, ord=np.inf) < tolerance:
            break
        direction = np.linalg.solve(
            hessian + 1e-12 * np.eye(X.shape[1]), -gradient
        )
        step_size = 1.0
        directional_derivative = gradient @ direction
        if -directional_derivative < 1e-14:
            break
        while step_size > 1e-8:
            candidate = beta + step_size * direction
            candidate_loss = objective_gradient_hessian(
                candidate, X, y, regularization, sample_weight
            )[0]
            if candidate_loss <= loss + 1e-4 * step_size * directional_derivative:
                beta = candidate
                break
            step_size *= 0.5
        else:
            raise RuntimeError("回溯线搜索未找到下降步")
    return beta, np.array(history)

regularization = 0.12
beta_newton, newton_history = fit_logistic_newton(
    X_train, y_train, regularization=regularization
)
validation_logits = X_validation @ beta_newton
validation_probability = stable_sigmoid(validation_logits)
validation_log_loss = binary_log_loss_from_logits(
    validation_logits, y_validation
)
validation_brier = np.mean((validation_probability - y_validation) ** 2)

print("Newton 迭代数：", len(newton_history))
print("训练目标首/尾：", round(newton_history[0], 6), round(newton_history[-1], 6))
print("验证 log loss / Brier：",
      round(validation_log_loss, 5), round(validation_brier, 5))
print("标准化系数：", np.round(beta_newton, 4))
```

### 5.3 Newton 为什么又叫 IRLS

定义当前工作响应

$$
z^{(t)}
=\tilde X\beta^{(t)}
+W_t^{-1}(y-p_t).
$$

下一步等价于最小化局部加权平方目标：

$$
\frac1{2n}
\left\lVert
W_t^{1/2}\left(z^{(t)}-\tilde X\beta\right)
\right\rVert_2^2
+\frac\lambda2\lVert P\beta\rVert_2^2.
$$

正规方程：

$$
\left(\frac1n\tilde X^TW_t\tilde X+\lambda P\right)\beta^{(t+1)}
=\frac1n\tilde X^TW_tz^{(t)}.
$$

这就是 iteratively reweighted least squares。这里的权重是 Bernoulli 局部曲率，不是“这个样本更可信”的业务权重，二者不能混为一谈。

```python
beta_irls_probe = np.array([-1.2, 0.2, -0.1, 0.05, 0.08])
probe_probability = stable_sigmoid(X_train @ beta_irls_probe)
probe_weight = probe_probability * (1 - probe_probability)
working_response = (
    X_train @ beta_irls_probe
    + (y_train - probe_probability) / probe_weight
)

irls_hessian = (
    X_train.T @ (probe_weight[:, None] * X_train) / len(y_train)
    + regularization * np.diag(penalty_mask)
)
irls_rhs = (
    X_train.T @ (probe_weight * working_response) / len(y_train)
)
beta_from_irls = np.linalg.solve(irls_hessian, irls_rhs)

_, probe_gradient, probe_hessian = objective_gradient_hessian(
    beta_irls_probe, X_train, y_train, regularization
)
beta_from_newton = beta_irls_probe - np.linalg.solve(
    probe_hessian, probe_gradient
)
print("IRLS 与 Newton 更新差：",
      np.linalg.norm(beta_from_irls - beta_from_newton))
```

<MlQuiz kind="优化辨析" question="IRLS 中的 p(1−p) 权重代表什么？" :options="['样本的业务重要性', 'Bernoulli 负 log-likelihood 在当前 logit 处的局部曲率', '标签是否正确', '部署时的类别先验']" :answer="1" explanation="IRLS 权重来自 Hessian。样本权重、类别权重和局部曲率权重来源不同，不能因名称相同而互换。" />

### 5.4 同一目标下比较 GD 与 Newton

比较优化器必须冻结数据、目标缩放、正则化和停止标准。下面用曲率上界给 GD 一个固定步长，再与阻尼 Newton 比较最终目标。

```python
def fit_logistic_gradient_descent(
    X, y, regularization, maximum_iterations=300
):
    beta = np.zeros(X.shape[1])
    spectral_bound = np.linalg.eigvalsh(X.T @ X / len(X)).max() / 4
    learning_rate = 1 / (spectral_bound + regularization)
    history = []
    for _ in range(maximum_iterations):
        loss, gradient, _ = objective_gradient_hessian(
            beta, X, y, regularization
        )
        history.append(loss)
        beta -= learning_rate * gradient
    return beta, np.array(history), learning_rate

beta_gd, gd_history, gd_learning_rate = fit_logistic_gradient_descent(
    X_train, y_train, regularization, maximum_iterations=300
)
gd_final = objective_gradient_hessian(
    beta_gd, X_train, y_train, regularization
)[0]
newton_final = objective_gradient_hessian(
    beta_newton, X_train, y_train, regularization
)[0]
print("GD 学习率：", round(gd_learning_rate, 4))
print("GD 300 步 / Newton 终值：", round(gd_final, 8), round(newton_final, 8))
print("两种系数差：", np.linalg.norm(beta_gd - beta_newton))
```

## 6. 完全分离：凸目标也可能没有有限解

把标签改写为 $\tilde y_i\in\{-1,+1\}$，单样本损失：

$$
\ell_i(\beta)=\log\left(1+e^{-\tilde y_i\tilde x_i^T\beta}\right).
$$

若存在方向 $v$ 使所有样本满足

$$
\tilde y_i\tilde x_i^Tv>0,
$$

数据完全线性可分。沿 $\beta=cv$ 且 $c\to\infty$：

$$
\ell_i(cv)\to0,
\qquad
J(cv)\to0.
$$

但任何有限 $c$ 的损失都严格大于 0。于是：

- 目标下确界是 0；
- 有限参数空间内没有点达到它；
- 系数范数持续增大；
- 概率越来越接近 0 或 1；
- $W$ 和 Hessian 数据曲率逐步退化。

**凸**意味着任一已存在的局部极小点都是全局极小点，不意味着极小点一定存在。严格凸也不能弥补非紧参数空间中的逃逸。

### 准完全分离

若一部分样本被某方向完美隔开、另一部分恰在边界上，部分系数仍可能发散。工程信号通常是：

- 系数对迭代上限高度敏感；
- loss 缓慢改善但参数范数持续增长；
- 标准误巨大或 Hessian 条件数恶化；
- 换一个批次后极端概率大幅翻转。

<SeparationRegularizationLab />

```python
separable_x = np.array([-1.8, -1.2, -0.7, -0.3, 0.3, 0.7, 1.2, 1.8])
separable_y = np.where(separable_x > 0, 1.0, 0.0)

def one_dimensional_path(x, y, regularization, steps, learning_rate=0.42):
    weight = 0.0
    losses = []
    weights = []
    signed_label = 2 * y - 1
    for _ in range(steps):
        margin = signed_label * weight * x
        loss = np.logaddexp(0, -margin).mean() + 0.5 * regularization * weight**2
        gradient = (
            -np.mean(signed_label * x * stable_sigmoid(-margin))
            + regularization * weight
        )
        losses.append(loss)
        weights.append(weight)
        weight -= learning_rate * gradient
    return np.array(weights), np.array(losses)

unregularized_w, unregularized_loss = one_dimensional_path(
    separable_x, separable_y, regularization=0.0, steps=1200
)
regularized_w, regularized_loss = one_dimensional_path(
    separable_x, separable_y, regularization=0.12, steps=1200
)
print("无正则 w(200/1200)：",
      round(unregularized_w[199], 3), round(unregularized_w[-1], 3))
print("有正则 w(200/1200)：",
      round(regularized_w[199], 3), round(regularized_w[-1], 3))
print("无正则 loss(200/1200)：",
      round(unregularized_loss[199], 5), round(unregularized_loss[-1], 5))
```

<MlQuiz kind="存在性判断" question="完全可分数据上，无正则逻辑回归的训练损失不断下降。哪项判断正确？" :options="['参数必然有限收敛', '损失可趋近 0，但有限 MLE 不存在，系数范数可能发散', '凸性已经失效', 'Sigmoid 输出会超过 1']" :answer="1" explanation="存在分离方向时，沿该方向放大参数可持续提高似然；下确界在无穷远处。" />

<div class="ml-stop"><strong>停下来推一推 3</strong> 在完全可分数据上，训练准确率可能很早就达到 100%，此后系数仍增长几百轮。准确率无法作为参数收敛证据，也无法说明概率已经校准。</div>

## 7. L2 正则化：从 Gaussian MAP 到数值制动

给斜率参数独立零均值 Gaussian 先验：

$$
\beta_j\sim\mathcal N(0,\tau^2),
$$

负 log prior 与 $\beta_j^2/(2\tau^2)$ 成正比。MAP 目标成为

$$
J_\lambda(\beta)
=\frac1n\sum_i\ell_i(\beta)
+\frac\lambda2\sum_{j=1}^d\beta_j^2.
$$

它同时带来三种效果：

1. **统计收缩**：降低小样本与共线特征的方差；
2. **存在性**：若所有逃逸方向都被惩罚，目标变得 coercive，有限最优点出现；
3. **数值曲率**：Hessian 加上 $\lambda I$，低曲率方向被托起。

### 为什么默认不惩罚截距

截距编码基准 log odds。惩罚它会把基准概率拉向 0.5，这通常不是合理先验。若确有先验基准率，应明确给截距单独的先验，而不是无意套用与斜率相同的惩罚。

### $\lambda$ 没有脱离目标缩放的绝对含义

以下两个目标不同：

$$
\sum_i\ell_i+\frac\lambda2\lVert\beta\rVert^2,
\qquad
\frac1n\sum_i\ell_i+\frac\lambda2\lVert\beta\rVert^2.
$$

同一个数值 $\lambda$ 在前者中相对弱 $n$ 倍。复制所有数据不会改变平均损失目标，却会改变求和损失中数据项与固定正则项的比例。文档、实验和生产代码必须锁定缩放协议。

<MlQuiz kind="正则化协议" question="训练代码从平均交叉熵改为交叉熵求和，却保持同一个 λ。会发生什么？" :options="['完全等价', '数据项相对放大约 n 倍，正则化有效强度显著减弱', '截距自动消失', 'Hessian 一定变成单位阵']" :answer="1" explanation="λ 的含义依赖目标缩放。比较实现前必须先统一 sum/mean、样本权重归一化和惩罚系数定义。" />

## 8. 非线性边界仍可保持凸训练

逻辑回归对参数线性，不要求对原始输入线性。用固定特征映射 $\phi(x)$：

$$
p(Y=1\mid x)=\sigma(\beta^T\phi(x)).
$$

例如加入 $x_1x_2$、平方项、样条或固定 RBF 后，原空间边界可弯曲；只要 $\phi$ 不随 $\beta$ 学习，负 log-likelihood 对 $\beta$ 仍凸。

但复杂特征会带来：

- 维度上升与共线性；
- 分离更容易发生；
- 边界外推不稳定；
- 系数单独解释困难；
- 必须在每个验证折内重做标准化和特征选择。

本章合成案例显式加入“电流 × 声发射”交互，是基于工艺假设的固定特征，不是事后查看测试集再补出的项。

## 9. 类别不平衡、样本权重与概率语义

### 9.1 少数类不等于必须加权

无权 Bernoulli 似然在正确设定和代表性抽样下估计部署后验。正类少会增加估计方差，却不意味着模型目标错误。动作不对称应优先在上一章的决策阈值中表达。

### 9.2 类别权重改变了隐含分布

给正类损失乘 $\alpha>1$：

$$
J_\alpha
=-\mathbb E\left[
\alpha Y\log q(X)+(1-Y)\log(1-q(X))
\right].
$$

逐点最优报告不再是原始 $\eta=P(Y=1\mid X)$，而是

$$
q^*(x)=\frac{\alpha\eta(x)}
{\alpha\eta(x)+1-\eta(x)}.
$$

因此

$$
\operatorname{logit}(q^*)=\operatorname{logit}(\eta)+\log\alpha.
$$

类别加权可改善优化关注或特定排序，但原始输出不再直接是部署后验。若抽样和权重机制已知，可做 odds 修正；仍需在代表部署分布的独立数据上验证校准。

```python
positive_weight = 6.0
training_weight = np.where(y_train == 1, positive_weight, 1.0)
beta_weighted, weighted_history = fit_logistic_newton(
    X_train,
    y_train,
    regularization=regularization,
    sample_weight=training_weight,
)

weighted_validation_logit = X_validation @ beta_weighted
weighted_probability_raw = stable_sigmoid(weighted_validation_logit)
weighted_probability_corrected = stable_sigmoid(
    weighted_validation_logit - np.log(positive_weight)
)

raw_weighted_brier = np.mean(
    (weighted_probability_raw - y_validation) ** 2
)
corrected_weighted_brier = np.mean(
    (weighted_probability_corrected - y_validation) ** 2
)
print("类别加权原始/odds 修正 Brier：",
      round(raw_weighted_brier, 5), round(corrected_weighted_brier, 5))
print("验证平均概率 原始/修正/真实率：",
      round(weighted_probability_raw.mean(), 4),
      round(weighted_probability_corrected.mean(), 4),
      round(y_validation.mean(), 4))
```

<MlQuiz kind="权重语义" question="正类交叉熵乘以 6 后，模型原始输出能否默认解释为部署后验？" :options="['能，权重只影响训练速度', '不能；权重改变逐点最优概率，需按抽样机制修正并在部署分布复验', '只要 AUC 高就能', '把阈值固定为 0.5 就能']" :answer="1" explanation="正类权重相当于改变隐含类别 odds。它不仅改变梯度大小，也改变目标最优概率。" />

### 9.3 业务样本权重也要有生成解释

以下权重不是同一种东西：

| 权重 | 目的 | 是否改变目标分布 |
|---|---|---|
| 逆抽样概率权重 | 还原被抽样总体 | 试图恢复原总体 |
| 类别代价权重 | 强调某类训练损失 | 通常改变概率语义 |
| 时间衰减权重 | 偏向近期工况 | 定义新的时间目标 |
| IRLS 曲率权重 | Newton 局部二次近似 | 不是业务重加权 |

权重必须进入数据合同：来源、归一化、极端值裁剪和最终要估计的分布都要明确。

## 10. 校准、阈值和强制规则必须分层

逻辑回归输出 $p_\beta(x)$；它并不自动决定 0.5 阈值。若误放成本为 $C_{\mathrm{FN}}$、误复核成本为 $C_{\mathrm{FP}}$，且对角损失为 0，动作阈值是

$$
\tau=\frac{C_{\mathrm{FP}}}
{C_{\mathrm{FP}}+C_{\mathrm{FN}}}.
$$

训练、校准和阈值选择要用不同证据：

1. 训练批次估计 $\beta$；
2. 独立验证批次检查 log loss、Brier 和可靠性；
3. 在验证批次上按成本或容量冻结阈值；
4. 测试批次只做一次最终验收；
5. 硬工艺规则始终位于模型动作之上。

### 只在验证批次选择运行阈值

```python
false_negative_cost = 70.0
false_positive_cost = 5.0
threshold_grid = np.linspace(0.01, 0.99, 199)

def confusion_at_threshold(y, probability, threshold):
    prediction = probability >= threshold
    tp = int(np.sum((prediction == 1) & (y == 1)))
    fp = int(np.sum((prediction == 1) & (y == 0)))
    tn = int(np.sum((prediction == 0) & (y == 0)))
    fn = int(np.sum((prediction == 0) & (y == 1)))
    return tp, fp, tn, fn

validation_costs = []
for threshold in threshold_grid:
    tp, fp, tn, fn = confusion_at_threshold(
        y_validation, validation_probability, threshold
    )
    validation_costs.append(
        (false_negative_cost * fn + false_positive_cost * fp)
        / len(y_validation)
    )
selected_threshold = threshold_grid[int(np.argmin(validation_costs))]

test_probability = stable_sigmoid(X_test @ beta_newton)
test_confusion = confusion_at_threshold(
    y_test, test_probability, selected_threshold
)
test_tp, test_fp, test_tn, test_fn = test_confusion
test_cost = (
    false_negative_cost * test_fn + false_positive_cost * test_fp
) / len(y_test)

print("冻结阈值：", round(selected_threshold, 4))
print("测试 TP/FP/TN/FN：", test_confusion)
print("测试每件相对成本：", round(test_cost, 4))
```

<MlQuiz kind="部署协议" question="看过测试批次成本后继续在同一测试集搜索阈值，会把测试集变成什么？" :options="['仍是一次性无偏验收集', '模型选择数据，最终成本会产生乐观偏差', '训练标签', '无监督数据']" :answer="1" explanation="阈值也是超参数。测试结果参与阈值选择后，需要新的未见批次才能恢复最终验收语义。" />

## 11. 参数不确定性与批次稳定性

在模型设定正确、样本独立且 MLE 条件成立时，参数协方差常用

$$
\widehat{\operatorname{Cov}}(\hat\beta)
\approx H(\hat\beta)^{-1}/n
$$

（取决于 $H$ 是平均还是求和尺度）。但焊缝同批相关、模型可能错设，简单逆 Hessian 会过度乐观。

工程上至少同时做：

- 按生产批次 bootstrap，而不是逐条焊缝 bootstrap；
- 报告系数或关键概率的分位区间；
- 检查换批次后符号、排序和校准是否稳定；
- 将大系数与高方差特征送回测量系统分析；
- 不把正则化 MAP 的逆 Hessian直接当成无偏频率学标准误。

```python
bootstrap_rng = np.random.default_rng(904)
unique_training_batches = np.unique(group_train)
bootstrap_coefficients = []

for _ in range(30):
    sampled_batches = bootstrap_rng.choice(
        unique_training_batches,
        size=len(unique_training_batches),
        replace=True,
    )
    sampled_rows = np.concatenate([
        np.flatnonzero(group_train == selected_batch)
        for selected_batch in sampled_batches
    ])
    beta_bootstrap, _ = fit_logistic_newton(
        X_train[sampled_rows],
        y_train[sampled_rows],
        regularization=regularization,
        maximum_iterations=50,
    )
    bootstrap_coefficients.append(beta_bootstrap)

bootstrap_coefficients = np.asarray(bootstrap_coefficients)
coefficient_interval = np.quantile(
    bootstrap_coefficients[:, 1:], [0.05, 0.5, 0.95], axis=0
)
print("斜率 5%/50%/95% 分位：")
print(np.round(coefficient_interval, 3))
```

## 数值验证：把公式变成可执行断言

工业教材不能以“曲线看起来对”结束。至少验证：

1. Sigmoid 对称性 $\sigma(-z)=1-\sigma(z)$；
2. 稳定 loss 在极端 logit 下有限；
3. 梯度有限差分误差足够小；
4. Hessian 对称且半正定；
5. 阻尼 Newton 目标单调不增；
6. Newton 与 IRLS 一步等价；
7. 完全分离下无正则系数继续增长；
8. 验证阈值冻结后测试混淆账本守恒。

```python
symmetry_logits = np.linspace(-30, 30, 121)
assert np.allclose(
    stable_sigmoid(-symmetry_logits),
    1 - stable_sigmoid(symmetry_logits),
    atol=1e-14,
)
assert np.isfinite(binary_log_loss_from_logits(
    np.array([-1000.0, 1000.0]), np.array([0, 1])
))
assert np.max(np.abs(analytic_gradient - numeric_gradient)) < 1e-7
assert np.allclose(analytic_hessian, analytic_hessian.T, atol=1e-12)
assert np.linalg.eigvalsh(analytic_hessian).min() > -1e-10
assert np.all(np.diff(newton_history) <= 1e-12)
assert np.linalg.norm(beta_from_irls - beta_from_newton) < 1e-9
assert unregularized_w[-1] > unregularized_w[199]
assert test_tp + test_fp + test_tn + test_fn == len(y_test)
assert 0 < selected_threshold < 1
print("逻辑回归的稳定性、梯度、曲率、IRLS 与部署账本检查通过。")
```

## 复杂度与工程边界

设样本数 $n$、含截距参数数 $d$：

| 方法 | 单轮时间 | 主要内存 | 适用边界 |
|---|---:|---:|---|
| 全批 GD | $O(nd)$ | $O(nd)$ 或流式 $O(d)$ | 大样本、稀疏特征 |
| mini-batch SGD | $O(bd)$ | $O(bd)$ | 在线或超大规模，但需学习率协议 |
| Newton/IRLS | $O(nd^2+d^3)$ | $O(nd+d^2)$ | $d$ 较小、需要少轮高精度 |
| L-BFGS | 约 $O(nd)$ | $O(nd+md)$ | 中高维稠密问题，不显式存 Hessian |
| 坐标/方差缩减法 | 依实现与稀疏度 | 依活跃特征 | 大规模稀疏线性分类 |

真正的成本不只在训练：

- 标准化器、特征顺序、缺失值协议必须版本化；
- 推理要稳定计算 logit 与 Sigmoid；
- 概率、阈值、动作和硬规则都要单独记录；
- 监控不能只看正类率，还要看输入漂移、校准与标签延迟；
- 参考标签只在被模型拦截样本可见时，需要随机审计，否则评价有选择偏差。

### 哪些问题不该硬塞给逻辑回归

- 决策边界高度局部且无法通过可信固定特征表达；
- 标签由未来处置决定，存在严重反事实缺失；
- 需要直接建模空间、序列或图结构，而手工汇总会丢失关键信息；
- 正类定义在不同工厂或检测设备间不一致；
- 关键动作受强制安全规范约束，模型概率没有决策权限。

逻辑回归仍是重要基线：目标凸、概率语义清晰、训练快、可做系数和误差审计。它的价值来自边界清楚，而不是“永远足够复杂”。

## 14. 常见失效模式与定位顺序

| 现象 | 首要假设 | 证据 | 修复方向 |
|---|---|---|---|
| loss 下降、系数持续增长 | 完全/准分离 | 范数轨迹、边际与 Hessian 条件数 | L2、合并稀有水平、重新定义特征 |
| 训练概率极端，测试校准差 | 过拟合或类别权重改变语义 | 可靠性图、Brier、分组测试 | 正则化、odds 修正、独立校准 |
| 不同求解器结果差很多 | 目标缩放或停止协议不同 | 对齐 sum/mean、λ、截距惩罚 | 固定目标合同与 KKT/梯度检查 |
| 换批次系数符号翻转 | 共线、批次混杂或漂移 | 组 bootstrap、相关结构、批次分层 | 重做测量与特征设计 |
| AUC 尚可但成本很差 | 阈值、先验或校准不匹配 | 成本曲线、每千件账本 | 验证集重选阈值并复验校准 |
| 线上正类率突然下降 | 输入/先验/标签管道变化 | 原始特征、缺失率、规则命中率 | 区分模型漂移与数据故障 |
| 训练指标异常优秀 | 标签泄漏 | 特征可用时间与消融 | 删除处置后特征，重做切分 |

建议排查顺序：

1. 先核对观测单元、标签、特征截止时间和分组键；
2. 再核对目标缩放、样本权重、正则化与截距；
3. 执行梯度/Hessian 数值检查；
4. 画系数范数、loss 和曲率轨迹识别分离；
5. 在独立批次检查排序、校准与成本；
6. 最后才更换优化器或模型家族。

## 15. 上线验收清单

- [ ] 正类参考检测、检测下限和标签延迟已定义。
- [ ] 训练、验证、测试按真实部署单元分组。
- [ ] 标准化和特征构造只在训练数据拟合。
- [ ] 目标使用 sum 还是 mean、λ 和截距惩罚已版本化。
- [ ] 梯度有限差分、Hessian 对称性和目标下降已验证。
- [ ] 已检查完全分离、准分离、秩和条件数。
- [ ] 类别/样本权重有明确抽样或目标分布解释。
- [ ] 概率在代表部署分布的独立批次上校准验收。
- [ ] 阈值只在验证批次选择并在测试批次一次冻结验收。
- [ ] 模型动作不覆盖强制工艺与安全规则。
- [ ] 线上同时记录模型版本、logit、概率、阈值、动作和规则命中。
- [ ] 有随机审计样本处理选择性标签问题。

## 16. 章节练习

### 练习 A：手推一个焊缝批次

给定

$$
X=
\begin{bmatrix}
1&-1\\
1&0\\
1&2
\end{bmatrix},
\quad
y=
\begin{bmatrix}
0\\0\\1
\end{bmatrix},
\quad
\beta=
\begin{bmatrix}
0\\\log2
\end{bmatrix},
$$

完成：

1. 计算三个 logit、概率和平均交叉熵；
2. 计算 $X^T(p-y)/3$；
3. 写出 $W$ 与 Hessian；
4. 证明 Hessian 半正定；
5. 做一步阻尼 Newton 并验证目标下降。

### 练习 B：从零实现可审计求解器

补全下面的训练器，不使用 scikit-learn。要求：

- 稳定 softplus；
- 截距不惩罚；
- 支持样本权重且明确归一化；
- 回溯线搜索；
- 梯度停止和最大迭代双重条件；
- 输出 loss、梯度范数、系数范数与最小 Hessian 特征值轨迹。

```python
# TODO: 练习骨架，验证脚本会跳过本代码块。
def auditable_logistic_fit(X, y, regularization, sample_weight=None):
    beta = np.zeros(X.shape[1])
    audit_rows = []
    # 1. 稳定计算目标、梯度与 Hessian
    # 2. 解 Newton 方向，不显式求逆
    # 3. 回溯到满足充分下降
    # 4. 写入每轮审计行并判断停止
    return beta, audit_rows
```

### 练习 C：证明类别权重的 odds 偏移

从

$$
-\alpha\eta\log q-(1-\eta)\log(1-q)
$$

的一阶条件推导

$$
\operatorname{logit}(q^*)=\operatorname{logit}(\eta)+\log\alpha.
$$

再说明以下三种情况下为什么不能只减 $\log\alpha$ 就宣布校准恢复：

1. 还有特征相关抽样；
2. 使用了强正则化；
3. 部署类先验又发生变化。

### 练习 D：构造失效反例

分别构造并数值验证：

1. 完全分离但设计矩阵满列秩；
2. 非分离但设计矩阵秩亏；
3. AUC 不变、校准显著变差的单调 logit 缩放；
4. 随机行切分优秀、按批次切分明显下降的数据。

<MlQuiz kind="综合审计" question="一个模型 AUC 高、训练 loss 低、优化器返回成功。哪项仍是上线前必需证据？" :options="['只需要训练准确率', '独立部署批次上的校准、冻结阈值成本、分离/权重语义和强制规则边界', '把系数都四舍五入', '确认 Sigmoid 图像是 S 形']" :answer="1" explanation="排序、概率、优化和动作是不同层。任何一层的成功都不能替代其余层的验收。" />

## 17. 本章结论

逻辑回归的闭环是：

$$
\boxed{
\text{线性 log odds}
\Rightarrow
\text{Sigmoid 概率}
\Rightarrow
\text{Bernoulli 似然}
\Rightarrow
\text{交叉熵}
\Rightarrow
X^T(p-y),\ X^TWX
}
$$

需要同时记住四条边界：

1. **凸性不保证有限解存在**：完全分离会把最优点推向无穷；
2. **正则化数值依赖缩放协议**：sum/mean、样本权重和截距惩罚必须统一；
3. **类别加权会改变概率语义**：排序改善不等于部署后验校准；
4. **概率不等于动作**：阈值、容量与强制规则属于独立决策层。

下一章把 Bernoulli 扩展为互斥多分类 Categorical 分布，推导 Softmax、矩阵交叉熵、参数不可辨识与稳定 log-sum-exp。

## 来源与许可

- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 369–380 页，§10.1–10.2.8：二分类逻辑回归、线性/非线性边界、Bernoulli NLL、梯度、Hessian、SGD、Newton/IRLS、MAP 与标准化。用于公式、术语和边界交叉核对；不复用原书 Iris 图、案例或表达。
- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 208–209 页，§5.1.6.1–5.1.6.3：KL、交叉熵、log loss、proper scoring rule 与 Brier score。用于概率评分语义核对；本章推导、题目和焊缝案例独立创作。
- Stanford CS229 Lecture Notes，PDF 第 21–24 页，§2.1：二分类、Sigmoid、Bernoulli likelihood 与梯度；PDF 第 29–30 页，§2.4：Newton–Raphson 与 Hessian 更新。用于推导顺序交叉核对；不复用垃圾邮件示例或原文图形。
- Jonathan Richard Shewchuk，*Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 63–69 页（讲义页码 57–63）：logistic loss、梯度、完全分离、Newton、凸性、IRLS 与 LDA/逻辑回归比较。用于存在性和优化边界交叉核对；不复制课堂图形、案例或措辞。

本章 2,400 条焊缝记录、48 个生产批次、传感器定义、成本值、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实焊接缺陷率、无损检测能力、放行阈值、工艺规范或安全决策。

<MlLessonProgress lesson-id="09" next-label="第 10 章：Softmax 回归" next-href="/courses/machine-learning/10-softmax" />
