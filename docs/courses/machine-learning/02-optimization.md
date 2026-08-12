---
course: machine-learning
edition: industrial
title: 梯度下降、牛顿法与随机优化
description: 以轧机厚度偏差校正为贯穿案例，从局部 Taylor 模型推导梯度下降与 Newton 法，证明学习率稳定区间，解释条件数、mini-batch 方差、Momentum 和 Adam，并从零实现与验证优化器。
outline: [2, 3]
---

<MlLessonHero number="02" title="梯度下降、牛顿法与随机优化" question="目标函数已经确定后，怎样用有限计算可靠地找到低损失参数，而不是只看到一个不断变化的 loss？" duration="建议 360–480 分钟" :outcomes="['把工业拟合任务写成有限和优化问题并检查数据形状', '从 Taylor 展开推导梯度方向、Newton 步和阻尼形式', '证明正定二次目标的学习率稳定区间与条件数效应', '解释 mini-batch 梯度的无偏性、方差和计算权衡', '用 NumPy 从零实现 GD、Newton、Momentum 与 Adam，并建立可诊断的停止协议']" />

第 01 章回答“系统真正应该最小化什么”。本章假定训练目标已经通过业务审查，集中处理另一个独立问题：在计算、内存和时间都有限的条件下，怎样找到这个目标的低点。

我们不会把优化器当成一串 API 参数。整章围绕一条冷轧生产线展开：先从厚度偏差校正模型得到最小二乘目标，再让梯度、曲率、随机近似和工程监控从案例中自然出现。

## 贯穿案例：轧机厚度偏差校正

冷轧线上，入口厚度、轧制力、带钢速度和温度会共同影响出口厚度。控制团队希望用上游测量预测下游厚度偏差，并把预测结果作为前馈校正的一项输入。

本章使用**教学合成数据**。它不代表任何钢厂的真实工艺窗口，也不能直接用于闭环控制。真实上线还需要控制稳定性、执行器约束、传感器延迟和安全联锁审查；本章只研究离线参数拟合。

### 先写清样本时序与数据边界

每个样本对应一段已经完成的带钢窗口：

| 对象 | 案例定义 | 形状 |
|---|---|---|
| 输入 $x_i$ | 校正动作前已可用的入口厚度、轧制力、速度、温度 | $mathbb R^4$ |
| 标签 $y_i$ | 同一窗口经过轧机后，由下游测厚仪测得的出口厚度偏差 | 标量 |
| 参数 $	heta$ | 截距与 4 个线性系数 | $mathbb R^5$ |
| 训练矩阵 $X$ | 第一列为 1，其余列为标准化特征 | $mathbb R^{n\times5}$ |
| 预测 | $widehat y_i=x_i^\top\theta$ | 标量 |

“下游实测偏差”可以作为历史训练标签，却不能作为当前窗口的输入。否则模型在离线评价时提前看到了部署时尚未产生的信息。

```python
import numpy as np

rng = np.random.default_rng(20260805)
n = 240

# 教学合成原始量：入口厚度 mm、轧制力 MN、速度 m/s、温度 ℃。
entry_thickness = rng.normal(2.05, 0.08, n)
rolling_force = 8.4 + 3.1 * (entry_thickness - 2.05) + rng.normal(0, 0.42, n)
speed = 12.5 - 0.65 * (rolling_force - 8.4) + rng.normal(0, 0.55, n)
temperature = 880 + 7.5 * (speed - 12.5) + rng.normal(0, 10.0, n)
X_raw = np.column_stack([entry_thickness, rolling_force, speed, temperature])

feature_mean = X_raw.mean(axis=0)
feature_scale = X_raw.std(axis=0, ddof=0)
X_scaled = (X_raw - feature_mean) / feature_scale
X = np.column_stack([np.ones(n), X_scaled])

true_theta = np.array([0.012, 0.024, 0.019, -0.016, 0.011])
y = X @ true_theta + rng.normal(0, 0.008, n)

assert X_raw.shape == (n, 4)
assert X.shape == (n, 5)
assert y.shape == (n,)
assert np.isfinite(X).all() and np.isfinite(y).all()
assert np.allclose(X[:, 1:].mean(axis=0), 0, atol=1e-12)
```

<MlQuiz kind="设计判断" question="离线表中同时有‘校正动作前入口厚度’与‘动作后下游实测偏差’，部署时模型的输入可以包含哪一项？" :options="['两项都可以，因为都在历史表中', '只能包含动作前已经可用的入口厚度；下游偏差只能作为标签或事后监控', '只能包含下游偏差，因为它更接近目标', '是否可用只由相关系数决定']" :answer="1" explanation="训练表的列存在，不等于推理时可用。特征必须服从决策时点；下游实测值发生在模型输出之后，作为当前输入会构成时间泄漏。" />

### 从案例得到有限和目标

采用平方损失后，经验目标为

$$
J(\theta)
=\frac{1}{2n}\sum_{i=1}^{n}(x_i^\top\theta-y_i)^2
=\frac{1}{2n}\lVert X\theta-y\rVert_2^2.
$$

前面的 $1/2$ 只是让求导后的 2 消失，不改变最小值位置。除以 $n$ 让梯度尺度不随样本量线性增长，也便于不同数据量之间比较学习率。

这个目标是第 01 章经验风险最小化的一个实例，也是典型的**有限和问题**：

$$
J(\theta)=\frac1n\sum_{i=1}^{n}\ell_i(\theta),
\qquad
\ell_i(\theta)=\frac12(x_i^\top\theta-y_i)^2.
$$

优化器能可靠降低 $J$，不代表平方损失、线性表示或训练数据一定适合真实控制目标。目标设计错误与目标求解失败必须分别诊断。

<div class="ml-stop"><strong>停下来推一推 1</strong>如果把目标从“样本平均平方误差”改为“样本平方误差之和”，最优参数是否改变？梯度尺度和可复用的学习率会怎样改变？</div>

## 1. 梯度是局部上升率，不是“通往答案的箭头”

设 $J:\mathbb R^p\to\mathbb R$ 可微，在当前参数 $\theta$ 附近走位移 $\Delta$。一阶 Taylor 展开为

$$
J(\theta+\Delta)
=J(\theta)+\nabla J(\theta)^\top\Delta+o(\lVert\Delta\rVert_2).
$$

记 $g=\nabla J(\theta)$。若只保留局部线性项，并限制 $\lVert\Delta\rVert_2\le\varepsilon$，Cauchy–Schwarz 不等式给出

$$
g^\top\Delta
\ge -\lVert g\rVert_2\lVert\Delta\rVert_2
\ge -\varepsilon\lVert g\rVert_2.
$$

等号在

$$
\Delta^\star=-\varepsilon\frac{g}{\lVert g\rVert_2}
$$

处取得。因此，在欧氏距离定义的等长小步中，$-g$ 是局部最速下降方向。把归一化步长吸收到学习率 $\eta_t$ 中，得到梯度下降：

$$
\boxed{\theta_{t+1}=\theta_t-\eta_t\nabla J(\theta_t)}.
$$

这里有三个容易被省略的限定：

1. “最速”只针对当前点的一阶模型，不是到全局最优点的最短路径；
2. 步子必须足够小，使被忽略的高阶项不主导变化；
3. 方向依赖“步长”的度量。若约束换成 $\ell_1$ 范数，最优小步会集中在绝对梯度最大的坐标，而不是沿完整负梯度方向。

<MlQuiz kind="概念辨析" question="在一个非凸目标上观察到 ∇J(θ)=0，能够直接得出什么？" :options="['θ 一定是全局最小值', 'θ 一定是局部最小值', 'θ 是一阶驻点，仍可能是局部极大值或鞍点', '学习率一定过小']" :answer="2" explanation="零梯度只说明局部线性项消失。还需要 Hessian、凸性或其他结构判断驻点类型；在高维非凸问题中，鞍点尤其常见。" />

### 案例目标的梯度与 Hessian

令残差 $r(\theta)=X\theta-y$。对平方目标求导：

$$
\begin{aligned}
J(\theta)&=\frac1{2n}r(\theta)^\top r(\theta),\\
\nabla J(\theta)&=\frac1nX^\top(X\theta-y),\\
\nabla^2J(\theta)&=\frac1nX^\top X.
\end{aligned}
$$

平方目标的 Hessian 与 $\theta$ 无关。它总是半正定；当 $X$ 列满秩时为正定，最小值唯一。若特征完全共线，$X^\top X$ 奇异，可能有多个参数得到相同预测。

```python
def least_squares_loss(theta, X, y):
    residual = X @ theta - y
    return float(residual @ residual / (2 * X.shape[0]))

def least_squares_gradient(theta, X, y):
    return X.T @ (X @ theta - y) / X.shape[0]

def least_squares_hessian(X):
    return X.T @ X / X.shape[0]

def finite_difference_gradient(function, theta, step=1e-6):
    estimate = np.empty_like(theta, dtype=float)
    for index in range(theta.size):
        direction = np.zeros_like(theta, dtype=float)
        direction[index] = step
        estimate[index] = (function(theta + direction) - function(theta - direction)) / (2 * step)
    return estimate

theta_probe = np.array([0.01, -0.02, 0.015, 0.005, -0.01])
analytic = least_squares_gradient(theta_probe, X, y)
numeric = finite_difference_gradient(lambda value: least_squares_loss(value, X, y), theta_probe)

assert analytic.shape == theta_probe.shape
assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-8)
assert np.all(np.linalg.eigvalsh(least_squares_hessian(X)) > 0)
```

数值梯度不是训练算法，而是边界测试。它用额外函数计算换取一条独立证据，适合在低维抽样参数上检查符号、平均因子和索引错误。

## 2. 一维二次函数给出精确学习率边界

先看最简单的曲率：

$$
J(w)=\frac12\lambda(w-w^\star)^2,
\qquad \lambda>0.
$$

令误差 $e_t=w_t-w^\star$。梯度下降更新为

$$
\begin{aligned}
w_{t+1}&=w_t-\eta\lambda(w_t-w^\star),\\
e_{t+1}&=(1-\eta\lambda)e_t,\\
e_t&=(1-\eta\lambda)^t e_0.
\end{aligned}
$$

收敛要求误差倍率的绝对值小于 1：

$$
|1-\eta\lambda|<1
\iff
\boxed{0<\eta<\frac2\lambda}.
$$

这个边界还能解释不同轨迹：

| 学习率 | 每步倍率 | 轨迹 |
|---|---:|---|
| $0<\eta<1/\lambda$ | $0<1-\eta\lambda<1$ | 在同一侧单调靠近 |
| $\eta=1/\lambda$ | $0$ | 一步到达精确二次函数的最优点 |
| $1/\lambda<\eta<2/\lambda$ | $-1<1-\eta\lambda<0$ | 跨越最优点振荡，振幅缩小 |
| $\eta=2/\lambda$ | $-1$ | 两侧等幅振荡，不收敛 |
| $\eta>2/\lambda$ | 绝对值大于 1 | 振幅增长并发散 |

<MlQuiz kind="结果预测" question="对 J(w)=5(w−2)²/2 使用固定学习率 η=0.4，会发生什么？" :options="['一步到 w=2', '同侧单调收敛', '在最优点两侧等幅振荡', '振荡但振幅缩小']" :answer="2" explanation="此时 η=2/λ=0.4，误差倍率为 1−ηλ=−1，所以误差只改变符号，不缩小。稳定条件是严格不等式。" />

## 3. 多维二次目标、特征值与条件数

在最优点附近，正定二次目标可写成

$$
J(\theta)=J(\theta^\star)+\frac12(\theta-\theta^\star)^\top H(\theta-\theta^\star),
\qquad H\succ0.
$$

令 $e_t=\theta_t-\theta^\star$，则

$$
e_{t+1}=(I-\eta H)e_t.
$$

对称矩阵可分解为 $H=Q\Lambda Q^\top$。在特征向量坐标 $z_t=Q^\top e_t$ 中，问题被拆成 $p$ 个一维递推：

$$
z_{t+1,j}=(1-\eta\lambda_j)z_{t,j}.
$$

所有方向同时稳定，当且仅当

$$
\boxed{0<\eta<\frac{2}{\lambda_{\max}(H)}}.
$$

最大特征值限制学习率，最小特征值决定最平缓方向的进展。二者之比

$$
\kappa(H)=\frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}
$$

称为 2-范数条件数。对正定二次目标，选择最优固定学习率

$$
\eta^\star=\frac{2}{\lambda_{\max}+\lambda_{\min}}
$$

时，最坏参数误差方向的单步收缩因子为

$$
\rho^\star=\frac{\kappa-1}{\kappa+1}.
$$

$\kappa$ 很大时，$\rho^\star$ 接近 1：算法必须在陡峭方向避免发散，同时在平缓方向缓慢爬行，于是出现狭长谷底中的“之”字形路径。

### 标准化为什么是优化操作

轧机特征的单位从毫米到摄氏度，原始坐标中的 $X^\top X$ 会把单位差异混入曲率。标准化不会解决所有共线性，但通常能消除纯粹由单位造成的病态尺度。

```python
X_raw_with_bias = np.column_stack([np.ones(n), X_raw])
H_raw = least_squares_hessian(X_raw_with_bias)
H_scaled = least_squares_hessian(X)

condition_raw = np.linalg.cond(H_raw)
condition_scaled = np.linalg.cond(H_scaled)

assert condition_scaled < condition_raw
assert condition_raw / condition_scaled > 100
print(f"原始条件数: {condition_raw:.2e}; 标准化后: {condition_scaled:.2f}")
```

训练时保存的 `feature_mean` 和 `feature_scale` 必须原样用于验证、测试和线上推理。若每个集合分别标准化，参数面对的坐标系已经不同，也会引入未来信息。

<OptimizationPathLab />

<MlQuiz kind="推导补全" question="若 H 的特征值为 1 和 20，固定学习率要使 GD 在两个方向都收敛，必须满足哪一项？" :options="['0<η<2', '0<η<0.1', 'η>20', 'η 只需小于 1']" :answer="1" explanation="稳定上界由最大特征值决定：2/λ_max=2/20=0.1。最小特征值影响慢方向的速度，不放宽上界。" />

<div class="ml-stop"><strong>停下来推一推 2</strong>当 $H$ 只有半正定而不是正定时，零特征值方向的误差倍率是多少？这说明目标值可以已经最优，但参数为什么仍不唯一？</div>

## 4. 从二次边界推广到光滑目标

若 $J$ 的梯度是 $L$-Lipschitz 连续，即

$$
\lVert\nabla J(u)-\nabla J(v)\rVert_2
\le L\lVert u-v\rVert_2,
$$

则下降引理给出一个二次上界：

$$
J(v)\le J(u)+\nabla J(u)^\top(v-u)+\frac L2\lVert v-u\rVert_2^2.
$$

令 $v=u-\eta\nabla J(u)$，得到

$$
J(v)
\le J(u)-\eta\left(1-\frac{L\eta}{2}\right)\lVert\nabla J(u)\rVert_2^2.
$$

所以当 $0<\eta<2/L$ 时，只要梯度非零，这个上界就保证一步下降。取 $\eta=1/L$ 时，至少有

$$
J(u)-J(v)\ge\frac{1}{2L}\lVert\nabla J(u)\rVert_2^2.
$$

对平方目标，$L=\lambda_{\max}(X^\top X/n)$，与前面的精确二次边界一致。对一般神经网络，局部曲率随参数变化，没有一个容易得到且全程适用的 $L$；学习率预热、衰减和回退因此是工程策略，不是对固定二次证明的简单照搬。

### 线搜索与回退

当 $L$ 未知时，可沿下降方向 $d_t$ 搜索步长。Armijo 回退从候选 $\eta$ 开始，反复乘 $0<c<1$，直到满足

$$
J(\theta_t+\eta d_t)
\le J(\theta_t)+\alpha\eta\nabla J(\theta_t)^\top d_t,
$$

其中 $\alpha$ 是很小的正数。它要求“下降得足够多”，避免为了找到一维精确最小点耗费过多函数计算。随机 mini-batch 会让条件两边都带噪声，直接照搬全量线搜索并不稳健。

## 5. 可执行的全量梯度下降

一个可审查的优化器至少要明确输入验证、学习率、停止条件和返回的历史记录。下面的实现只面向稠密最小二乘，不试图包装成通用框架。

```python
def validate_regression_data(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X 必须为二维矩阵，y 必须为一维向量")
    if X.shape[0] == 0 or X.shape[0] != y.shape[0]:
        raise ValueError("X 与 y 的样本数必须相同且非空")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X 与 y 不能包含 NaN 或无穷值")
    return X, y

def gradient_descent(X, y, *, learning_rate=None, max_steps=10000, tolerance=1e-9):
    X, y = validate_regression_data(X, y)
    if max_steps < 1 or tolerance <= 0:
        raise ValueError("max_steps 和 tolerance 必须为正")

    theta = np.zeros(X.shape[1], dtype=float)
    H = least_squares_hessian(X)
    largest_eigenvalue = float(np.linalg.eigvalsh(H).max())
    if largest_eigenvalue <= 0:
        raise ValueError("目标没有正曲率，无法由该规则选择学习率")
    eta = 1 / largest_eigenvalue if learning_rate is None else float(learning_rate)
    if not 0 < eta < 2 / largest_eigenvalue:
        raise ValueError("learning_rate 超出该二次目标的稳定区间")

    history = []
    for step in range(max_steps):
        gradient = least_squares_gradient(theta, X, y)
        history.append({
            "step": step,
            "loss": least_squares_loss(theta, X, y),
            "gradient_norm": float(np.linalg.norm(gradient)),
        })
        if np.linalg.norm(gradient) <= tolerance * (1 + np.linalg.norm(theta)):
            break
        theta = theta - eta * gradient
    return theta, history

theta_gd, history_gd = gradient_descent(X, y)
theta_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]

assert len(history_gd) < 10000
assert np.linalg.norm(theta_gd - theta_lstsq) < 1e-7
assert history_gd[-1]["loss"] <= history_gd[0]["loss"]
```

停止条件使用缩放后的梯度范数，而不是比较一次 `loss` 是否下降。仍需设置最大步数，因为容差可能永远达不到；真实训练还要监控验证指标、计算预算和时间预算。

<MlQuiz kind="代码预测" question="若平方损失实现使用样本平均，而梯度实现忘记除以 n，最可能发生什么？" :options="['最优参数一定改变', '梯度方向不变，但有效步长被放大 n 倍，原学习率可能发散', '梯度自动变成无偏估计', 'Hessian 会变成零矩阵']" :answer="1" explanation="常数缩放不改变 argmin，却会缩放梯度和曲率。若仍沿用平均损失对应的学习率，更新会被放大 n 倍。数值梯度检查可以捕获这种不一致。" />

## 6. Newton 法：让曲率校正坐标

一阶方法只用斜率。Newton 法在当前点建立二阶局部模型：

$$
m_t(\Delta)
=J(\theta_t)+g_t^\top\Delta+\frac12\Delta^\top H_t\Delta,
$$

其中 $g_t=\nabla J(\theta_t)$，$H_t=\nabla^2J(\theta_t)$。对 $\Delta$ 求导并令其为零：

$$
g_t+H_t\Delta_t=0
\quad\Longrightarrow\quad
\boxed{H_t\Delta_t=-g_t}.
$$

若 $H_t\succ0$，则

$$
g_t^\top\Delta_t=-g_t^\top H_t^{-1}g_t<0,
$$

所以 Newton 步是下降方向。实现时应求解线性方程 $H_t\Delta_t=-g_t$，而不是显式构造 $H_t^{-1}$：求逆做了不需要的工作，通常也放大数值误差。

### 为什么最小二乘一步到达

平方目标本身就是精确二次函数，局部二次模型等于原目标。其梯度满足

$$
g(\theta)=H\theta-\frac1nX^\top y.
$$

若 $H$ 可逆，完整 Newton 步给出

$$
\theta_1
=\theta_0-H^{-1}g(\theta_0)
=H^{-1}\frac1nX^\top y,
$$

与正规方程解相同，与初始点无关。

```python
def newton_step_for_least_squares(theta, X, y, damping=0.0):
    if damping < 0:
        raise ValueError("damping 不能为负")
    gradient = least_squares_gradient(theta, X, y)
    H = least_squares_hessian(X)
    system = H + damping * np.eye(H.shape[0])
    delta = np.linalg.solve(system, -gradient)
    return theta + delta

theta_start = np.array([0.5, -0.4, 0.3, -0.2, 0.1])
theta_newton = newton_step_for_least_squares(theta_start, X, y)

assert np.allclose(theta_newton, theta_lstsq, rtol=1e-10, atol=1e-10)
assert least_squares_loss(theta_newton, X, y) < least_squares_loss(theta_start, X, y)
```

### 一般模型为什么没有“一步神话”

对逻辑回归或神经网络，Hessian 随参数变化，二次模型只在局部近似原目标。还会出现三类问题：

- $H_t$ 不正定：Newton 步可能指向局部极大或鞍点；
- $H_t$ 近奇异：线性方程对数值误差极敏感；
- 参数维度巨大：存储 $p\times p$ Hessian 已不可行。

常见修正是阻尼：

$$
(H_t+\lambda I)\Delta_t=-g_t,
\qquad \lambda>0.
$$

$\lambda$ 大时，更新更接近缩小后的梯度步；$\lambda$ 小时，更接近 Newton 步。信赖域方法进一步根据局部模型对真实目标的预测质量调整允许步长。

<MlQuiz kind="错误诊断" question="某非凸模型在一次完整 Newton 步后 loss 上升。最合理的首要解释是什么？" :options="['Newton 法不使用梯度', '当前 Hessian 可能不正定，或局部二次近似在完整步长范围内失效', '训练样本一定有标签泄漏', '只要把 batch size 设为 1 就能修复']" :answer="1" explanation="Newton 步依赖局部曲率。非正定 Hessian 不能保证下降，完整步也可能走出二次近似可信区域；应检查特征值、阻尼、线搜索或信赖域。" />

## 7. 随机梯度：用计算换噪声

全量梯度

$$
g(\theta)=\frac1n\sum_{i=1}^{n}\nabla\ell_i(\theta)
$$

每一步都读取全部样本。数据量很大时，早期参数还很差，却花费一次完整遍历来精确计算方向，可能并不划算。

若从 $\{1,\ldots,n\}$ 中均匀、有放回地抽取索引 $I$，单样本梯度满足

$$
\mathbb E_I[\nabla\ell_I(\theta)]
=\frac1n\sum_{i=1}^{n}\nabla\ell_i(\theta)
=\nabla J(\theta).
$$

它是全量梯度的无偏估计。mini-batch $B_t$ 的平均梯度为

$$
g_t=\frac1{|B_t|}\sum_{i\in B_t}\nabla\ell_i(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta_tg_t.
$$

若每个样本梯度独立、协方差为 $\Sigma$，则

$$
\operatorname{Cov}(g_t)=\frac{\Sigma}{B}.
$$

方差按 $1/B$ 缩小，标准差按 $1/\sqrt B$ 缩小。有限数据无放回抽样还会出现有限总体修正；时间连续、同一设备相邻窗口或按类别成批的数据通常强相关，不能机械套用独立假设。

<StochasticGradientLab />

<div class="ml-stop"><strong>停下来推一推 3</strong>保持总样本梯度计算次数不变，把 batch size 从 8 增到 64，会让参数更新次数变成原来的多少？梯度噪声、硬件吞吐与“看过相同样本数后的进展”为什么不能只用单步 loss 比较？</div>

### epoch、shuffle 与“无偏”不是同一件事

- **iteration**：完成一次参数更新；
- **mini-batch**：一次更新读取的样本集合；
- **epoch**：训练流程按定义让每个训练样本参与一次；
- **shuffle**：每个 epoch 前重新打乱索引，减少固定顺序带来的系统性偏差。

有放回独立抽样最容易分析，但一个 epoch 内可能漏掉或重复样本。工程实现通常采用“每轮随机打乱后无放回分批”，单步之间不独立，却能保证一轮覆盖所有样本。理论假设和实际采样协议要分开写清。

<MlQuiz kind="结果预测" question="在独立同分布近似下，把 mini-batch 从 B=4 增到 B=64，梯度估计的协方差与标准差大约怎样变化？" :options="['协方差变为 1/16，标准差变为 1/4', '协方差变为 1/4，标准差变为 1/16', '二者都变为 1/16', '二者都不变']" :answer="0" explanation="batch 增大 16 倍，均值的协方差除以 16；标准差是方差的平方根，因此除以 4。" />

## 8. Momentum：对历史方向做低通滤波

在狭长谷底中，梯度的陡峭分量可能每步反号，平缓分量却长期同向。Momentum 保存历史梯度的指数平均：

$$
\begin{aligned}
m_t&=\beta m_{t-1}+(1-\beta)g_t,\\
\theta_{t+1}&=\theta_t-\eta m_t,
\qquad 0\le\beta<1.
\end{aligned}
$$

展开递推：

$$
m_t=(1-\beta)\sum_{k=0}^{t-1}\beta^k g_{t-k}
$$

（假定 $m_0=0$）。持续同向的分量会积累，来回变号的分量会抵消，因此轨迹可能更快沿谷底前进。

不同框架有两种常见约定：有的公式省略 $(1-\beta)$，有的把学习率并入速度。它们不能在相同 $\eta$ 下直接比较；迁移超参数时必须先核对更新方程。

Momentum 不保证每一步损失下降。惯性会让参数越过谷底；若学习率和 $\beta$ 同时过大，振荡甚至会放大。

## 9. Adam：一阶矩、二阶矩与逐坐标缩放

Adam 同时维护梯度的一阶矩和平方梯度的二阶原点矩：

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
s_t&=\beta_2s_{t-1}+(1-\beta_2)g_t^2.
\end{aligned}
$$

因为 $m_0=s_0=0$，训练早期两个指数平均都偏向 0。偏差修正为

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat s_t=\frac{s_t}{1-\beta_2^t}.
$$

参数按坐标更新：

$$
\boxed{
\theta_{t+1}
=\theta_t-eta\frac{\widehat m_t}{\sqrt{\widehat s_t}+\epsilon}
}.
$$

$g_t^2$、平方根和除法都是逐坐标运算。频繁出现大梯度的坐标会得到较小的有效步长；稀疏或尺度不同的坐标因此可能更容易训练。

但 Adam 不是“免调参优化器”：

- 基础学习率 $\eta$ 仍决定整体步幅；
- $\epsilon$ 不只是防止除零，在低精度训练中也影响有效更新；
- 对角缩放不能消除由参数相关性造成的旋转型病态条件；
- 训练损失下降仍不能修复错误目标、泄漏或失配的验证协议；
- 自适应更新在某些问题上可能不收敛，不能把经验默认值当成普遍证明。

### 从零实现 mini-batch Momentum 与 Adam

下面让两个优化器共用同一打乱顺序和“batch 内取平均”的约定。为了让示例可复现，随机种子是接口的一部分。

```python
def stochastic_least_squares(
    X, y, *, optimizer, batch_size=16, learning_rate=0.03,
    epochs=100, momentum=0.9, beta2=0.999, epsilon=1e-8, seed=0
):
    X, y = validate_regression_data(X, y)
    if optimizer not in {"momentum", "adam"}:
        raise ValueError("optimizer 必须是 momentum 或 adam")
    if not 1 <= batch_size <= X.shape[0]:
        raise ValueError("batch_size 必须位于 1 与样本数之间")
    if learning_rate <= 0 or epochs < 1 or epsilon <= 0:
        raise ValueError("学习率、轮数和 epsilon 必须为正")

    random = np.random.default_rng(seed)
    theta = np.zeros(X.shape[1])
    first_moment = np.zeros_like(theta)
    second_moment = np.zeros_like(theta)
    step = 0
    history = []

    for epoch in range(epochs):
        for start in range(0, X.shape[0], batch_size):
            indices = random.permutation(X.shape[0]) if start == 0 else indices
            batch = indices[start:start + batch_size]
            gradient = least_squares_gradient(theta, X[batch], y[batch])
            step += 1

            first_moment = momentum * first_moment + (1 - momentum) * gradient
            if optimizer == "momentum":
                update = first_moment
            else:
                second_moment = beta2 * second_moment + (1 - beta2) * gradient ** 2
                corrected_first = first_moment / (1 - momentum ** step)
                corrected_second = second_moment / (1 - beta2 ** step)
                update = corrected_first / (np.sqrt(corrected_second) + epsilon)

            theta = theta - learning_rate * update

        history.append(least_squares_loss(theta, X, y))
    return theta, np.asarray(history)

theta_momentum, momentum_history = stochastic_least_squares(
    X, y, optimizer="momentum", batch_size=16, learning_rate=0.08, epochs=120, seed=7
)
theta_adam, adam_history = stochastic_least_squares(
    X, y, optimizer="adam", batch_size=16, learning_rate=0.02, epochs=120, seed=7
)

initial_loss = least_squares_loss(np.zeros(X.shape[1]), X, y)
assert momentum_history[-1] < initial_loss * 0.1
assert adam_history[-1] < initial_loss * 0.1
assert np.isfinite(theta_momentum).all() and np.isfinite(theta_adam).all()
```

这里每个 epoch 只生成一次排列；最后一个 batch 可以小于指定大小，但梯度函数仍按该 batch 的实际样本数取平均。若误用“求和”，最后一个小 batch 与完整 batch 的有效学习率会不同。

<MlQuiz kind="代码预测" question="mini-batch 损失由平均改为求和，但学习率保持不变，最后一个只有 3 个样本的 batch 与完整 16 个样本的 batch 相比会怎样？" :options="['完整 batch 的更新通常约大 16/3 倍，步幅依赖 batch 实际大小', '两个更新严格相同', '最后一个 batch 一定更大', '只影响日志，不影响参数']" :answer="0" explanation="求和梯度的尺度正比于样本数；平均梯度则把尺度与 batch 大小解耦。变长 batch 时尤其应明确 reduction 约定。" />

## 数值验证

推导必须落到可以失败的检查。下面用一个已知特征值的二维目标验证稳定边界，并确认发散不是绘图错觉。

```python
H_test = np.diag([8.0, 0.5])

def quadratic_path(learning_rate, steps=40):
    theta = np.array([2.0, 2.0])
    losses = []
    for _ in range(steps):
        losses.append(float(0.5 * theta @ H_test @ theta))
        theta = theta - learning_rate * (H_test @ theta)
    return theta, np.asarray(losses)

stable_theta, stable_losses = quadratic_path(0.20)
boundary_theta, boundary_losses = quadratic_path(0.25)
unstable_theta, unstable_losses = quadratic_path(0.27)

assert np.linalg.eigvalsh(H_test).max() == 8.0
assert 0.20 < 2 / 8.0
assert np.isclose(0.25, 2 / 8.0)
assert stable_losses[-1] < stable_losses[0] * 1e-3
assert np.isclose(abs(boundary_theta[0]), 2.0)
assert unstable_losses[-1] > unstable_losses[0] * 100
```

本章的验证链分为四层：

1. **公式检查**：解析梯度与中心差分一致；
2. **边界检查**：在 $2/\lambda_{\max}$ 的两侧观察收敛、等幅振荡和发散；
3. **交叉解检查**：GD 与 Newton 的结果对齐 `np.linalg.lstsq`；
4. **随机检查**：固定种子后，Momentum 与 Adam 在相同数据和采样顺序下显著降低目标。

不能只断言“最终 loss 小于初始 loss”。一个符号或缩放错误可能偶然通过这种弱测试；应尽量把数学性质写成具有分界的断言。

<MlQuiz kind="证据判断" question="下列哪组证据最能说明自写 GD 的梯度与稳定边界都实现正确？" :options="['训练一次后 loss 下降', '只与另一个自写优化器结果一致', '解析梯度通过中心差分，并在理论边界两侧呈现预期的收敛与发散', '把迭代次数增加到一百万']" :answer="2" explanation="中心差分检查局部导数，边界两侧测试检查更新动力学；两条相对独立的证据比单次 loss 下降更有诊断力。" />

## 复杂度与工程边界

以下复杂度针对稠密线性模型，$n$ 为样本数、$p$ 为参数数、$B$ 为 mini-batch 大小。神经网络的一次梯度成本应按一次前向与反向传播估算，不能直接套用矩阵乘法常数。

| 方法 | 单次主要计算 | 额外状态 | 优势 | 主要边界 |
|---|---:|---:|---|---|
| 全量 GD | $O(np)$ | $O(p)$ | 方向确定、便于线搜索与严格监控 | 大数据每步代价高，病态目标迭代慢 |
| mini-batch SGD | $O(Bp)$ | $O(p)$ | 单步便宜、适合流式与硬件并行 | 梯度有噪声，学习率与采样协议敏感 |
| Momentum | $O(Bp)$ | $O(p)$ 速度状态 | 减少振荡、积累稳定方向 | 可能过冲，公式约定影响超参数含义 |
| Adam | $O(Bp)$ | $O(2p)$ 矩状态 | 逐坐标缩放，常用于深度网络 | 内存更高，仍需调学习率，非普遍收敛保证 |
| Newton | 构造 Hessian 常为 $O(np^2)$，稠密求解 $O(p^3)$ | $O(p^2)$ | 低维、良态问题局部很快 | 高维不可直接存 Hessian，非凸曲率可能非正定 |
| L-BFGS | 每步若含全量梯度为 $O(np)$，两层递推约 $O(mp)$ | $O(mp)$ | 用最近 $m$ 组差分近似曲率 | 对随机噪声和非平稳目标更敏感 |

### 训练运行至少记录什么

工业训练不能只保留一条平滑后的 loss 曲线。每个 checkpoint 至少应关联：

- 代码版本、数据快照、随机种子、特征变换版本和优化器完整配置；
- 训练损失与独立验证指标，二者的 reduction 和样本权重定义；
- 学习率、梯度范数、参数范数、更新范数及其比值；
- 吞吐量、每步耗时、显存峰值、NaN/Inf 计数和跳过更新次数；
- epoch、全局 step、累计看过的样本数与停止原因；
- 最佳验证 checkpoint 与最后 checkpoint，不能默认为同一个模型。

### 停止协议

可组合使用以下证据：

1. 梯度范数低于相对容差；
2. 目标的相对改善连续多轮低于阈值；
3. 验证指标在预设耐心窗口内不再改善；
4. 达到时间、样本或算力预算；
5. 出现非有限值、更新异常放大或安全边界触发。

随机目标下，单步 loss 上升很正常。早停应使用固定验证协议和足够长的平滑窗口，不能看到一次波动就停止，也不能反复查看最终测试集来挑停止点。

### 常见故障的证据链

| 现象 | 优先检查 | 不应立即下的结论 |
|---|---|---|
| 第一步就 NaN | 输入有限性、学习率、除法/对数域、混合精度溢出 | “模型结构不适合” |
| loss 周期性剧烈振荡 | 学习率、Momentum 约定、曲率最大方向 | “需要更多数据” |
| 训练慢但稳定 | 条件数、特征尺度、梯度吞吐、batch 利用率 | “迭代次数还不够” |
| 训练降、验证升 | 数据切分、过拟合、分布漂移 | “优化器没有收敛” |
| 换大 batch 后效果变差 | 更新次数、学习率缩放、归一化层、采样组成 | “大 batch 一定泛化差” |
| Newton 步上升 | Hessian 特征值、阻尼、线搜索、求解残差 | “二阶方法永远不可用” |

<MlQuiz kind="错误诊断" question="训练 loss 平稳下降，但验证期轧制批次误差持续升高。最合理的下一步是什么？" :options="['只把 Adam 换成 SGD', '继续训练直到梯度为 0', '检查时间切分、分布漂移与过拟合，并用冻结验证协议定位；这不是单凭训练曲线能证明的优化失败', '把验证集并入训练集后重新报告']" :answer="2" explanation="优化器正在降低训练目标；验证恶化指向泛化或数据协议问题。需要保持验证独立，不能用并入训练来掩盖证据。" />

## 10. 章节练习：从推导到可复现实验

### 基础练习

1. 对 $J(w)=\frac12\lambda(w-a)^2$，分别写出 $\eta=0.5/\lambda$、$1/\lambda$、$1.5/\lambda$ 和 $2/\lambda$ 的误差递推，并画出前 6 步符号。
2. 证明正定二次目标在 $\eta^\star=2/(\lambda_{\max}+\lambda_{\min})$ 下，两个端点方向的误差倍率绝对值相同。
3. 推导无放回抽取 $B$ 个样本均值的有限总体方差修正，并说明 $B=n$ 时为何方差为 0。

### 实现练习

把下面骨架补成带 Armijo 回退的 GD。要求记录每轮函数调用次数，并在二维二次目标和一个非二次凸目标上验证每个接受步都满足充分下降条件。

```python
# TODO：完成 Armijo 回退；本代码块由学习者执行。
def gradient_descent_with_backtracking(function, gradient, theta0):
    theta = np.asarray(theta0, dtype=float).copy()
    # 1. 计算下降方向
    # 2. 从候选学习率开始回退
    # 3. 检查 Armijo 条件并记录函数调用
    # 4. 同时检查梯度容差与最大迭代数
    return theta
```

### 工业诊断练习

给定一份训练日志，设计不超过 6 个只读检查来区分以下原因：

- batch 内损失从平均误改成求和；
- 特征标准化参数与线上版本不一致；
- 学习率跨过稳定边界；
- 训练集下降但时间外验证集发生分布漂移；
- Adam 的 `epsilon` 在低精度计算中下溢。

答案必须写出每个检查预期观察到的证据，不能只列“调小学习率”“换优化器”之类动作。

<details><summary>实现检查点</summary>

- 边界处拒绝形状不一致、空数组、非有限值和非法超参数。
- batch 梯度使用实际 batch 大小取平均。
- 随机种子、打乱策略、全局 step 和 epoch 均可追溯。
- 梯度检查和收敛检查是不同测试，不以一者替代另一者。
- Newton 求解线性方程并检查求解残差，不显式求逆。
- 训练停止原因写入返回结果或日志，而不是只靠循环自然结束。

</details>

## 11. 本章小结与术语

- 负梯度来自一阶局部模型和欧氏步长约束，不是指向全局答案的罗盘。
- 二次目标的稳定学习率由最大特征值限制，收敛速度受条件数控制。
- 标准化与预条件通过改变参数坐标中的曲率改善优化，而不是改变业务目标。
- Newton 法最小化局部二次模型；低维良态问题很快，高维非凸问题需要阻尼、近似或 Hessian-vector product。
- SGD 用有噪声但低成本的梯度估计交换计算；batch size 同时影响方差、吞吐和更新次数。
- Momentum 与 Adam 重组历史梯度和坐标尺度，但都不能替代数据、目标和验证协议审查。

<dl class="ml-terms">
  <div><dt>descent direction</dt><dd>与当前梯度内积为负、在足够小步长下能降低目标的方向。</dd></div>
  <div><dt>learning rate</dt><dd>把更新方向转换为实际参数位移的步长。</dd></div>
  <div><dt>condition number</dt><dd>正定 Hessian 最大与最小特征值之比，刻画曲率尺度不均衡。</dd></div>
  <div><dt>Hessian</dt><dd>目标对参数的二阶偏导矩阵，描述局部曲率。</dd></div>
  <div><dt>mini-batch</dt><dd>一次随机更新所使用的训练样本子集。</dd></div>
  <div><dt>epoch</dt><dd>按训练协议使每个训练样本参与一次的完整数据遍历。</dd></div>
  <div><dt>preconditioning</dt><dd>用坐标变换或矩阵缩放改善不同方向的曲率尺度。</dd></div>
  <div><dt>stationary point</dt><dd>梯度为零的一阶驻点，可能是极小、极大或鞍点。</dd></div>
</dl>

## 来源与许可

- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 314–320 页，§8.2–8.3：用于核对下降方向、固定步长、条件数、Momentum 与 Newton 局部二次模型的知识边界。本站重新组织并独立推导，不复用原文、图表或案例。
- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*，PDF 第 322–331 页，§8.4：用于核对有限和 SGD、mini-batch、学习率日程、预条件方法与 Adam 偏差修正。本站代码、实验和表述均独立创作。
- Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义），PDF 第 110–114 页：用于交叉核对 mini-batch 的方差与硬件权衡、输入标准化和 Momentum 的工程直觉；不复制讲义图片、措辞或课堂案例。
- 本章轧机案例、240 条合成样本、两项网页交互、全部题目、代码和 SVG 图形均为本站独立创作。它们不代表真实产线统计或控制策略。详细登记见[来源与许可登记](./sources.md) 。

<MlLessonProgress lesson-id="02" next-label="第 03 章：泛化、偏差—方差与模型选择" next-href="/courses/machine-learning/03-generalization" />
