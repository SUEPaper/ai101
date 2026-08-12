---
title: 主成分分析（PCA）
description: 从伺服压装多传感器过程指纹案例推导最大方差、最小重构、特征分解、SVD、维数选择、白化与工业过程监测
course: machine-learning
edition: industrial
---

<MlLessonHero
  number="25"
  title="主成分分析（PCA）"
  question="十二路量测中大部分变化其实由三种共同工况驱动：怎样找到这组三维正交坐标，同时保留主要变化、最小化重构误差，并把模型外的新变化单独报警？"
  duration="建议 720–840 分钟"
  :outcomes="[
    '从 Rayleigh 商与 Lagrange 乘子推导第一主成分，并用谱分解证明它是全局最优方向',
    '证明最大投影方差、最小正交重构误差、协方差特征分解与中心化数据 SVD 四条路线等价',
    '从零实现可冻结、可变换、可逆变换的 PCA，并验证符号、子空间、解释方差与重构恒等式',
    '用分组留出、子空间稳定性、T²/Q 双统计量和漂移审计识别量纲、泄漏、离群点与坐标失效'
  ]"
/>

第 24 章的 GMM 用多个概率成分描述数据云；本章先问一个更基础的问题：如果十二路传感器并没有十二个独立自由度，而是被少数共同工况同时驱动，能否用一个低维线性子空间表达它们？

主成分分析（principal component analysis, PCA）给出的答案是：先冻结中心与量纲，再寻找一组两两正交的方向，使投影后保留的方差最大。令人重要的不是“调用一次 SVD”，而是同一组方向还会让正交重构误差最小；协方差矩阵的特征向量、中心化数据的右奇异向量、低秩近似和线性编码器因此连成同一件事。

> PCA 是无监督的几何压缩，不会自动发现物理因果、质量标签或安全边界。本章所有异常名称都只描述合成数据中的几何状态；模型只允许辅助工程复核，不得直接修改压装参数、旁路联锁或自动放行产品。

## 1. 问题合同：压缩的对象到底是什么

设数据矩阵为

$$
\mathbf X=
\begin{bmatrix}
\mathbf x_1^\top\\
\vdots\\
\mathbf x_N^\top
\end{bmatrix}
\in\mathbb R^{N\times D},
$$

其中一行是一件产品，一列是一项量测。PCA 要学习：

- 训练均值 $\boldsymbol\mu\in\mathbb R^D$；
- 可选的逐列尺度 $\mathbf s\in\mathbb R_{>0}^D$；
- 正交基 $\mathbf W_q\in\mathbb R^{D\times q}$，满足 $\mathbf W_q^\top\mathbf W_q=\mathbf I_q$。

标准化后的中心数据记为

$$
\mathbf X_c
=\left(\mathbf X-\mathbf 1\boldsymbol\mu^\top\right)
\operatorname{diag}(\mathbf s)^{-1}.
$$

编码、解码分别是

$$
\mathbf Z=\mathbf X_c\mathbf W_q,
\qquad
\widehat{\mathbf X}_c=\mathbf Z\mathbf W_q^\top.
$$

$\mathbf Z\in\mathbb R^{N\times q}$ 是主成分得分（scores），$\mathbf W_q$ 的列是主方向。工程上还必须明确两件事：

1. **坐标合同**：均值、尺度、方向、顺序和符号如何冻结；
2. **决策合同**：重构误差或得分越界后谁复核，模型不得做什么。

PCA 不是“把列数从 $D$ 改成 $q$”这么简单。若中心、量纲、切分单位或冻结版本改变，坐标含义也随之改变。

<MlQuiz
  kind="对象辨析"
  question="PCA 训练后，哪一组对象必须一起版本化，才能对新样本复现同一坐标？"
  :options="['均值、尺度、主方向及其顺序/符号', '只保存累计解释方差', '只保存训练样本数量', '只保存第一主成分得分']"
  :answer="0"
  explanation="新样本必须先使用训练期均值和尺度，再乘冻结的方向矩阵；方向的顺序与符号也会影响接口坐标。"
/>

## 贯穿案例：伺服压装 EOL 过程指纹

### 2.1 业务边界与十二路量测

某电驱动总成产线用伺服压机完成轴套压装。每件产品结束后，系统保存力—位移曲线、电机电流和夹具振动的摘要。许多列彼此相关：过盈量同时影响峰值力、做功、电流和扭矩；入座过程同时影响位移、时间和力斜率；热状态又会联动电流、扭矩与压头温度。

| 字段 | 单位 | 教学含义 | 预期共同变化 |
|---|---:|---|---|
| `peak_force_kn` | kN | 压装峰值力 | 过盈/负载 |
| `final_force_kn` | kN | 终点保持力 | 过盈/负载 |
| `work_energy_j` | J | 曲线积分做功 | 负载与行程 |
| `seating_depth_mm` | mm | 入座深度 | 行程/定位 |
| `motor_current_a` | A | 峰段驱动电流 | 负载与热态 |
| `drive_torque_nm` | N·m | 等效驱动扭矩 | 负载与热态 |
| `approach_duration_ms` | ms | 接近段时间 | 节拍/行程 |
| `settle_duration_ms` | ms | 保压稳定时间 | 节拍与热态 |
| `head_temperature_c` | °C | 压头温度 | 热态 |
| `fixture_vibration_rms` | mm/s | 夹具振动摘要 | 多因素与独立噪声 |
| `force_slope_kn_per_mm` | kN/mm | 入座段力斜率 | 行程/接触 |
| `hold_current_a` | A | 保持段电流 | 负载与热态 |

允许动作只有：压缩正常过程指纹、辅助工程师查看相关变化、为受控复测提供线索。模型不得据此自动调整速度、力限、位移终点、保持时间、返修策略或安全联锁。

### 2.2 合成数据与 campaign 隔离

教学数据包含 72 个 production campaign，每个 campaign 8 件产品，共 576 条记录。前三个潜在因子只用于生成相关结构；它们不是可直接观测或已确认的物理变量。前 48 个 campaign 是已审核的基线批组，后 24 个批组中注入两种只用于验收的合成状态：

- **子空间内工况远行**：仍沿既有三维相关结构移动；
- **子空间外传感器不一致**：峰值力与终点力的关系偏离训练子空间。

训练算法不读取 `STATE_AUDIT`。它只在验证和最终测试时用于核对双统计量门禁是否识别预设几何差异。

```python
import numpy as np

SEED = 2505
N_CAMPAIGNS = 72
UNITS_PER_CAMPAIGN = 8

FEATURE_NAMES = np.array([
    "peak_force_kn", "final_force_kn", "work_energy_j", "seating_depth_mm",
    "motor_current_a", "drive_torque_nm", "approach_duration_ms",
    "settle_duration_ms", "head_temperature_c", "fixture_vibration_rms",
    "force_slope_kn_per_mm", "hold_current_a",
])
RAW_CENTER = np.array([
    42.0, 35.0, 128.0, 18.4, 14.0, 8.5,
    420.0, 180.0, 46.0, 0.82, 9.6, 8.2,
])
RAW_SCALE = np.array([
    4.8, 4.2, 18.0, 0.55, 1.8, 1.2,
    38.0, 26.0, 4.5, 0.18, 1.25, 1.0,
])
LOADINGS = np.array([
    [ .92,  .15,  .08],
    [ .86,  .24,  .05],
    [ .78,  .48,  .10],
    [-.62,  .70,  .02],
    [ .70,  .12,  .56],
    [ .66,  .20,  .52],
    [ .08,  .90,  .10],
    [ .14,  .78,  .22],
    [ .06,  .14,  .94],
    [ .34,  .30,  .20],
    [-.22,  .84,  .16],
    [ .54, -.08,  .62],
])
NOISE_SD = np.array([
    .16, .18, .18, .20, .18, .20,
    .19, .22, .17, .30, .20, .22,
])

def orthogonal_force_contrast(loadings=LOADINGS):
    contrast = np.zeros(loadings.shape[0])
    contrast[0], contrast[1] = 1.0, -1.0
    basis, _ = np.linalg.qr(loadings)
    residual = contrast - basis @ (basis.T @ contrast)
    return residual / np.linalg.norm(residual)

ORTH_CONTRAST = orthogonal_force_contrast()

def make_pressfit_dataset(seed=SEED):
    rng = np.random.default_rng(seed)
    rows, groups, states = [], [], []
    for campaign in range(N_CAMPAIGNS):
        campaign_latent = rng.normal(0, [.12, .10, .10])
        for _ in range(UNITS_PER_CAMPAIGN):
            latent = rng.normal(size=3) + campaign_latent
            x_std = latent @ LOADINGS.T + rng.normal(0, NOISE_SD)
            state = 0
            if campaign >= 48:
                draw = rng.random()
                if draw < .10:
                    state = 1
                    direction = np.array([1.0, .30, -.18])
                    direction /= np.linalg.norm(direction)
                    jump = rng.choice([-1, 1]) * rng.uniform(3.3, 4.2)
                    x_std += jump * (direction @ LOADINGS.T)
                elif draw < .20:
                    state = 2
                    jump = rng.choice([-1, 1]) * rng.uniform(3.8, 4.8)
                    x_std += jump * ORTH_CONTRAST
            rows.append(RAW_CENTER + RAW_SCALE * x_std)
            groups.append(campaign)
            states.append(state)
    return np.asarray(rows), np.asarray(groups), np.asarray(states)

X_RAW, CAMPAIGN, STATE_AUDIT = make_pressfit_dataset()

assert X_RAW.shape == (576, 12)
assert CAMPAIGN.shape == STATE_AUDIT.shape == (576,)
assert np.isfinite(X_RAW).all()
assert np.linalg.norm(LOADINGS.T @ ORTH_CONTRAST) < 1e-12
print("全部记录 / 正常 / 子空间内 / 子空间外：",
      len(X_RAW), np.bincount(STATE_AUDIT))
```

输出为：

```text
全部记录 / 正常 / 子空间内 / 子空间外： 576 [537  13  26]
```

切分单位必须是 campaign，而不是行。前 48 个审核基线批组用于拟合；接下来的 12 个批组用于选择维数和阈值；最后 12 个批组只打开一次。均值和标准差只能从训练组估计。

```python
train_mask = CAMPAIGN < 48
val_mask = (CAMPAIGN >= 48) & (CAMPAIGN < 60)
test_mask = CAMPAIGN >= 60

scale_mean = X_RAW[train_mask].mean(axis=0)
scale_std = X_RAW[train_mask].std(axis=0, ddof=0)
if np.any(scale_std <= 0):
    raise ValueError("训练集中存在零方差特征")

X = (X_RAW - scale_mean) / scale_std
X_TRAIN, X_VAL, X_TEST = X[train_mask], X[val_mask], X[test_mask]
STATE_VAL, STATE_TEST = STATE_AUDIT[val_mask], STATE_AUDIT[test_mask]

assert (len(X_TRAIN), len(X_VAL), len(X_TEST)) == (384, 96, 96)
assert set(CAMPAIGN[train_mask]).isdisjoint(CAMPAIGN[val_mask])
assert set(CAMPAIGN[train_mask]).isdisjoint(CAMPAIGN[test_mask])
assert np.allclose(X_TRAIN.mean(axis=0), 0, atol=1e-12)
assert np.allclose(X_TRAIN.std(axis=0), 1, atol=1e-12)
print("train / validation / test：", 384, 96, 96)
print("validation audit：", np.bincount(STATE_VAL))
print("test audit：", np.bincount(STATE_TEST))
```

输出为：

```text
train / validation / test： 384 96 96
validation audit： [79  6 11]
test audit： [74  7 15]
```

这套协议刻意区分四类证据：

1. 训练基线只决定坐标和训练期统计分布；
2. 验证批组选择 $q$、报警门限和运行合同；
3. 测试批组只报告冻结方案的样本外表现；
4. 合成状态只核对教材代码，不代表现场已有根因标签。

<MlQuiz
  kind="切分审计"
  question="为什么 PCA 虽然不读取标签，也不能在全数据上先拟合再做验证？"
  :options="['验证批组的均值、尺度和协方差会进入训练变换', 'PCA 只能处理有标签数据', '全数据 SVD 一定不收敛', '验证集必须只有一个特征']"
  :answer="0"
  explanation="无监督预处理同样会从验证输入分布获得信息。评价流水线中的所有拟合步骤都必须留在训练折内。"
/>

## 3. 中心化与尺度：两个不能混为一谈的决定

### 3.1 不中心化时优化的是相对原点的二阶矩

训练均值为

$$
\boldsymbol\mu=\frac1N\sum_{i=1}^{N}\mathbf x_i,
\qquad
\mathbf x_{c,i}=\mathbf x_i-\boldsymbol\mu.
$$

中心化协方差使用分母 $N$：

$$
\mathbf S
=\frac1N\mathbf X_c^\top\mathbf X_c.
$$

若跳过中心化，得到的是

$$
\frac1N\mathbf X^\top\mathbf X
=\mathbf S+\boldsymbol\mu\boldsymbol\mu^\top.
$$

右侧多出的秩一项会让第一方向朝向“原点到均值”，而不是数据围绕均值的主要变化。只有数据根据物理定义已严格零均值时，中心化才可省略。

```python
x_demo = np.array([
    [9.0, 1.0], [10.0, 2.0], [11.0, 3.0],
    [12.0, 4.0], [13.0, 5.0],
])
mu_demo = x_demo.mean(axis=0)
xc_demo = x_demo - mu_demo
second_moment = x_demo.T @ x_demo / len(x_demo)
cov_demo = xc_demo.T @ xc_demo / len(x_demo)

assert np.allclose(second_moment, cov_demo + np.outer(mu_demo, mu_demo))
def canonical_eigen_angle(matrix):
    direction = np.linalg.eigh(matrix)[1][:, -1]
    if direction[0] < 0:
        direction *= -1
    return np.degrees(np.arctan2(direction[1], direction[0]))

angle_raw = canonical_eigen_angle(second_moment)
angle_centered = canonical_eigen_angle(cov_demo)
print("未中心化 / 已中心化方向角：",
      round(angle_raw, 2), round(angle_centered, 2))
```

输出为：

```text
未中心化 / 已中心化方向角： 16.0 45.0
```

### 3.2 标准化改变问题，而不是单纯改善数值

中心化几乎总是 PCA 定义的一部分；标准化则是建模选择：

- **协方差 PCA**：只中心化，原单位中的大方差方向权重更大；
- **相关系数 PCA**：每列再除训练标准差，各列先变成单位方差；
- **工程尺度 PCA**：除以量测不确定度、公差或经批准的参考尺度，目标直接对应工程偏差。

如果同一列从 mm 改报 µm，协方差 PCA 的方向会变化；相关系数 PCA 应保持不变。反过来，若十二个像素使用同一量纲，逐列标准化可能把纯噪声像素放大到与高信噪比像素同等重要。

```python
scale_demo = np.array([
    [1.0, 1.1], [2.0, 1.9], [3.0, 3.2],
    [4.0, 3.8], [5.0, 5.1], [6.0, 6.2],
])

def first_pc_angle(x, standardize=False):
    xc = x - x.mean(axis=0)
    if standardize:
        xc = xc / xc.std(axis=0, ddof=0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0:
        direction = -direction
    return np.degrees(np.arctan2(direction[1], direction[0]))

rescaled_demo = scale_demo * np.array([1000.0, 1.0])
raw_angles = [
    first_pc_angle(scale_demo),
    first_pc_angle(rescaled_demo),
]
standard_angles = [
    first_pc_angle(scale_demo, True),
    first_pc_angle(rescaled_demo, True),
]

assert abs(standard_angles[0] - standard_angles[1]) < 1e-10
assert abs(raw_angles[0] - raw_angles[1]) > 30
print("协方差 PCA 方向角：", np.round(raw_angles, 3))
print("相关系数 PCA 方向角：", np.round(standard_angles, 3))
```

<PcaScalingWorkbenchLab />

上线前不能写“统一先标准化”。应在特征清单中逐列记录单位、参考尺度、零方差处理和更改审批；换传感器量程后，旧 PCA 模型不能静默沿用。

<MlQuiz
  kind="尺度判断"
  question="十二路量测单位不同，因此一定应该逐列标准化。这个结论为什么不成立？"
  :options="['尺度选择取决于要保留原方差、相关结构还是公差意义', '因为 PCA 不允许标准化', '因为标准化会产生负特征值', '只有分类任务才能标准化']"
  :answer="0"
  explanation="标准化会改变目标几何。单位不同是重要信号，但仍需决定单位方差、工程公差或原始能量哪一种才符合任务。"
/>

## 4. 第一主成分：最大方差不是一句直觉

### 4.1 从投影到 Rayleigh 商

对已中心化样本选单位方向 $\mathbf w\in\mathbb R^D$，投影得分为

$$
z_i=\mathbf w^\top\mathbf x_{c,i},
\qquad \|\mathbf w\|_2=1.
$$

因为得分均值为零，投影方差是

$$
\begin{aligned}
\operatorname{Var}(z)
&=\frac1N\sum_i z_i^2\\
&=\frac1N\sum_i
\mathbf w^\top\mathbf x_{c,i}\mathbf x_{c,i}^\top\mathbf w\\
&=\mathbf w^\top\mathbf S\mathbf w.
\end{aligned}
$$

单位约束不能删除：否则把 $\mathbf w$ 放大任意倍就能让目标无界。PCA 的第一方向解决

$$
\max_{\mathbf w}\ \mathbf w^\top\mathbf S\mathbf w
\quad\text{s.t.}\quad
\mathbf w^\top\mathbf w=1.
$$

这就是对称矩阵的 Rayleigh 商优化。

### 4.2 Lagrange 乘子给出特征方程

构造

$$
\mathcal L(\mathbf w,\lambda)
=\mathbf w^\top\mathbf S\mathbf w
-\lambda(\mathbf w^\top\mathbf w-1).
$$

因为 $\mathbf S=\mathbf S^\top$，

$$
\nabla_{\mathbf w}\mathcal L
=2\mathbf S\mathbf w-2\lambda\mathbf w=0,
$$

于是

$$
\boxed{\mathbf S\mathbf w=\lambda\mathbf w}.
$$

驻点是特征向量，目标值正好是对应特征值：

$$
\mathbf w^\top\mathbf S\mathbf w
=\lambda\mathbf w^\top\mathbf w=\lambda.
$$

### 4.3 为什么最大特征值是全局最优

驻点条件本身只说明“候选是特征向量”。要证明全局最优，把任意单位向量展开到 $\mathbf S$ 的正交特征基：

$$
\mathbf w=\sum_{j=1}^{D}a_j\mathbf v_j,
\qquad
\sum_j a_j^2=1,
$$

并令 $\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_D\ge0$。则

$$
\mathbf w^\top\mathbf S\mathbf w
=\sum_j\lambda_j a_j^2
\le\lambda_1\sum_j a_j^2
=\lambda_1.
$$

等号在 $\mathbf w=\pm\mathbf v_1$ 处取得。若 $\lambda_1$ 重复，则对应特征子空间中的任意单位方向都是最优；这不是算法故障，而是数据没有提供唯一轴向。

```python
s_hand = np.array([[4.0, 1.5], [1.5, 2.0]])
eigenvalues_hand, eigenvectors_hand = np.linalg.eigh(s_hand)
order_hand = np.argsort(eigenvalues_hand)[::-1]
eigenvalues_hand = eigenvalues_hand[order_hand]
eigenvectors_hand = eigenvectors_hand[:, order_hand]
w_hand = eigenvectors_hand[:, 0]
rayleigh_hand = w_hand @ s_hand @ w_hand

assert np.allclose(s_hand @ w_hand, eigenvalues_hand[0] * w_hand)
assert np.isclose(rayleigh_hand, eigenvalues_hand[0])
assert np.all(eigenvalues_hand >= -1e-12)
print("特征值：", np.round(eigenvalues_hand, 6))
print("第一方向：", np.round(w_hand, 6))
```

### 4.4 网格搜索把证明变成可观察曲线

二维单位方向可写成 $\mathbf w(\theta)=(\cos\theta,\sin\theta)$。扫描角度得到的方差曲线，应在第一特征向量方向或其反方向达到最大。

```python
angles = np.linspace(-np.pi, np.pi, 20001)
directions = np.column_stack([np.cos(angles), np.sin(angles)])
variances = np.einsum("ni,ij,nj->n", directions, s_hand, directions)
best_grid = directions[np.argmax(variances)]
alignment = abs(best_grid @ w_hand)

assert alignment > 1 - 1e-7
assert np.isclose(variances.max(), eigenvalues_hand[0], atol=1e-7)
print("网格最大方差 / 最大特征值：",
      round(variances.max(), 6), round(eigenvalues_hand[0], 6))
print("方向绝对内积：", round(alignment, 9))
```

<PcaProjectionGantryLab />

::: ml-stop 停下来推一推 1
若 $\mathbf S=\operatorname{diag}(9,4,1)$，第一主成分、第一投影方差和把方向乘以 $-1$ 后的结果分别是什么？再解释为何符号变化不改变重构矩阵 $\mathbf w\mathbf w^\top$。
:::

<MlQuiz
  kind="证明接口"
  question="从 Lagrange 一阶条件得到 S w = λw 后，还缺哪一步才能声称第一主成分是全局最大方差方向？"
  :options="['用谱展开证明任意单位方向的 Rayleigh 商不超过最大特征值', '把所有特征值平方', '改用梯度下降', '假设所有样本服从 Gaussian']"
  :answer="0"
  explanation="一阶条件只给候选。谱展开把目标写成特征值的凸组合，才证明最大特征值给出全局上界。"
/>

## 5. 多个主成分：优化的是子空间

取 $\mathbf W_q=[\mathbf w_1,\ldots,\mathbf w_q]$ 且 $\mathbf W_q^\top\mathbf W_q=\mathbf I_q$。总投影方差是

$$
\operatorname{tr}(\mathbf W_q^\top\mathbf S\mathbf W_q)
=\sum_{j=1}^{q}\mathbf w_j^\top\mathbf S\mathbf w_j.
$$

PCA 解决

$$
\max_{\mathbf W_q^\top\mathbf W_q=\mathbf I_q}
\operatorname{tr}(\mathbf W_q^\top\mathbf S\mathbf W_q).
$$

Ky Fan 最大值原理给出最优值 $\sum_{j=1}^{q}\lambda_j$，最优列空间由前 $q$ 个特征向量张成。可把它理解为顺序选择：先取 $\mathbf v_1$，第二方向必须与它正交，于是只能在剩余子空间选择最大 Rayleigh 商，依次类推。

这里要区分“轴”和“子空间”：

- 单个特征向量至少有 $\pm$ 符号不唯一；
- 若截断边界内有重复特征值，对应轴可以任意正交旋转；
- 只要投影矩阵 $\mathbf P_q=\mathbf W_q\mathbf W_q^\top$ 相同，重构子空间就相同。

因此跨库或跨批次比较时，优先比较投影矩阵、主夹角或重构，而不是逐元素比较每列载荷。

```python
def fit_pca_svd(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2:
        raise ValueError("x 必须是至少两行的二维数组")
    if not np.isfinite(x).all():
        raise ValueError("x 包含非有限值")
    mean = x.mean(axis=0)
    xc = x - mean
    u, singular, vt = np.linalg.svd(xc, full_matrices=False)
    eigenvalues = singular**2 / len(x)
    return {
        "mean": mean,
        "components": vt.T,
        "eigenvalues": eigenvalues,
        "scores": u * singular,
    }

PCA_MODEL = fit_pca_svd(X_TRAIN)
EIGENVALUES = PCA_MODEL["eigenvalues"]
EVR = EIGENVALUES / EIGENVALUES.sum()
CEVR = np.cumsum(EVR)

assert PCA_MODEL["components"].shape == (12, 12)
assert np.allclose(
    PCA_MODEL["components"].T @ PCA_MODEL["components"],
    np.eye(12), atol=1e-12,
)
assert np.all(np.diff(EIGENVALUES) <= 1e-12)
print("前六个特征值：", np.round(EIGENVALUES[:6], 4))
print("前 1/2/3 维累计解释方差：", np.round(CEVR[:3], 4))
```

输出为：

```text
前六个特征值： [6.6505 3.3594 1.4216 0.2092 0.0665 0.0645]
前 1/2/3 维累计解释方差： [0.5542 0.8342 0.9526]
```

解释方差比定义为

$$
\operatorname{EVR}_j
=\frac{\lambda_j}{\sum_{r=1}^{D}\lambda_r},
\qquad
\operatorname{CEVR}(q)
=\sum_{j=1}^{q}\operatorname{EVR}_j.
$$

在训练集使用逐列单位方差且分母为 $N$ 时，特征值之和应等于 $D=12$。这是一条很有用的实现不变量。

```python
train_cov = X_TRAIN.T @ X_TRAIN / len(X_TRAIN)
captured_trace = np.trace(
    PCA_MODEL["components"][:, :3].T
    @ train_cov
    @ PCA_MODEL["components"][:, :3]
)

assert np.isclose(np.trace(train_cov), X_TRAIN.shape[1], atol=1e-10)
assert np.isclose(captured_trace, EIGENVALUES[:3].sum(), atol=1e-10)
print("总标准化方差：", round(np.trace(train_cov), 6))
print("前三维捕获方差：", round(captured_trace, 6))
```

<MlQuiz
  kind="不唯一性"
  question="两个 PCA 实现给出的前三列载荷逐元素不同，但投影矩阵 W Wᵀ 几乎相同。最合理的判断是什么？"
  :options="['它们恢复了同一子空间，差异可能来自符号或重复谱内旋转', '其中一个一定错误', 'PCA 不能有三个分量', '必须把特征值改成负数']"
  :answer="0"
  explanation="PCA 的核心对象是子空间。符号翻转和重复特征值子空间内的正交旋转不会改变投影与重构。"
/>

## 6. 最小重构误差：同一答案的第二条推导

### 6.1 固定正交基时，最优编码就是投影

给定 $\mathbf W_q$，对单个中心样本寻找编码 $\mathbf z$：

$$
\min_{\mathbf z}
\|\mathbf x_c-\mathbf W_q\mathbf z\|_2^2.
$$

展开并求导：

$$
\begin{aligned}
L(\mathbf z)
&=\mathbf x_c^\top\mathbf x_c
-2\mathbf z^\top\mathbf W_q^\top\mathbf x_c
+\mathbf z^\top\mathbf W_q^\top\mathbf W_q\mathbf z,\\
\nabla_{\mathbf z}L
&=-2\mathbf W_q^\top\mathbf x_c+2\mathbf z.
\end{aligned}
$$

所以

$$
\boxed{\mathbf z^*=\mathbf W_q^\top\mathbf x_c},
\qquad
\widehat{\mathbf x}_c
=\mathbf W_q\mathbf W_q^\top\mathbf x_c.
$$

令 $\mathbf P_q=\mathbf W_q\mathbf W_q^\top$。它满足

$$
\mathbf P_q^\top=\mathbf P_q,
\qquad
\mathbf P_q^2=\mathbf P_q,
$$

所以是正交投影矩阵。

### 6.2 Pythagoras 把重构误差变成丢弃方差

投影与残差正交：

$$
(\mathbf P_q\mathbf x_c)^\top
(\mathbf I-\mathbf P_q)\mathbf x_c=0.
$$

因此每个样本都满足

$$
\|\mathbf x_c\|_2^2
=\|\mathbf P_q\mathbf x_c\|_2^2
+\|(\mathbf I-\mathbf P_q)\mathbf x_c\|_2^2.
$$

对所有样本平均，第一项总能量与 $\mathbf W_q$ 无关。最小化残差能量等价于最大化投影能量：

$$
\arg\min_{\mathbf W_q^\top\mathbf W_q=\mathbf I}
\frac1N\|\mathbf X_c-\mathbf X_c\mathbf P_q\|_F^2
=
\arg\max_{\mathbf W_q^\top\mathbf W_q=\mathbf I}
\operatorname{tr}(\mathbf W_q^\top\mathbf S\mathbf W_q).
$$

在最优解处，

$$
\boxed{
\frac1N\|\mathbf X_c-\widehat{\mathbf X}_c\|_F^2
=\sum_{j=q+1}^{D}\lambda_j
}.
$$

这也是 Eckart–Young–Mirsky 定理在中心化数据低秩近似上的表现。

```python
def reconstruct_centered(xc, components, q):
    basis = components[:, :q]
    scores = xc @ basis
    return scores @ basis.T

xc_train = X_TRAIN - PCA_MODEL["mean"]
for q in [1, 2, 3, 6, 12]:
    reconstructed = reconstruct_centered(
        xc_train, PCA_MODEL["components"], q
    )
    mse = np.mean(np.sum((xc_train - reconstructed) ** 2, axis=1))
    discarded = EIGENVALUES[q:].sum()
    assert np.isclose(mse, discarded, atol=1e-10)

q = 3
reconstructed_3 = reconstruct_centered(
    xc_train, PCA_MODEL["components"], q
)
captured = np.sum(reconstructed_3**2, axis=1)
residual = np.sum((xc_train - reconstructed_3)**2, axis=1)
total = np.sum(xc_train**2, axis=1)
assert np.allclose(total, captured + residual, atol=1e-10)
print("q=3 平均重构误差 / 丢弃特征值和：",
      round(residual.mean(), 6), round(EIGENVALUES[3:].sum(), 6))
```

::: ml-stop 停下来推一推 2
特征值为 $[8,3,1,0.5]$ 时，降到二维保留多少总方差比例？平均平方重构误差是多少？为什么把 $q$ 从 2 增加到 3 后，训练与测试重构误差通常都会下降，却不能据此自动认定 $q=3$ 更适合所有任务？
:::

<MlQuiz
  kind="等价关系"
  question="为何最大投影方差与最小正交重构误差得到同一子空间？"
  :options="['总中心能量固定，投影能量与正交残差能量逐样本相加', '因为所有数据都服从球形 Gaussian', '因为特征值之和总是 1', '因为 PCA 使用了标签']"
  :answer="0"
  explanation="正交投影给出 Pythagoras 分解；固定总能量下，多保留一单位投影能量就少一单位残差能量。"
/>

## 7. SVD：不要先把数值问题平方

### 7.1 右奇异向量就是主方向

中心化矩阵的薄 SVD 为

$$
\mathbf X_c=\mathbf U\boldsymbol\Sigma\mathbf V^\top,
$$

其中奇异值 $\sigma_1\ge\cdots\ge0$。代入协方差：

$$
\mathbf S
=\frac1N\mathbf X_c^\top\mathbf X_c
=\mathbf V\frac{\boldsymbol\Sigma^2}{N}\mathbf V^\top.
$$

因此：

$$
\boxed{\mathbf W_q=\mathbf V_{:,1:q}},
\qquad
\boxed{\lambda_j=\frac{\sigma_j^2}{N}},
$$

而主成分得分已经包含在

$$
\mathbf Z_q
=\mathbf X_c\mathbf V_q
=\mathbf U_q\boldsymbol\Sigma_q.
$$

注意术语在不同资料中有歧义：有些资料把方向向量称为 principal components，有些把样本投影得分称为 principal components。本章固定使用“主方向/载荷”指 $\mathbf V$ 的列，“得分”指 $\mathbf Z$。

### 7.2 为什么直接 SVD 通常更稳

显式形成 $\mathbf X_c^\top\mathbf X_c$ 会把二范数条件数平方：

$$
\kappa_2(\mathbf X_c^\top\mathbf X_c)
=\kappa_2(\mathbf X_c)^2.
$$

这会压缩小奇异方向的有效精度。直接对 $\mathbf X_c$ 做 SVD，不需要先形成协方差矩阵，通常是更稳健的默认实现。它不会消除病态性，但避免主动把问题再平方一次。

```python
cov_train = xc_train.T @ xc_train / len(xc_train)
evals_eigh, evecs_eigh = np.linalg.eigh(cov_train)
descending = np.argsort(evals_eigh)[::-1]
evals_eigh = evals_eigh[descending]
evecs_eigh = evecs_eigh[:, descending]

projector_svd = PCA_MODEL["components"][:, :3] @ PCA_MODEL["components"][:, :3].T
projector_eigh = evecs_eigh[:, :3] @ evecs_eigh[:, :3].T

assert np.allclose(evals_eigh, EIGENVALUES, atol=1e-12)
assert np.allclose(projector_svd, projector_eigh, atol=1e-12)
print("特征值最大差：", f"{np.max(np.abs(evals_eigh - EIGENVALUES)):.3e}")
print("前三维投影矩阵最大差：",
      f"{np.max(np.abs(projector_svd - projector_eigh)):.3e}")
```

### 7.3 $D\gg N$、秩与截断算法

中心化后秩最多为 $\min(N-1,D)$，所以最多只有这么多个非零特征值。若 $D\gg N$：

- 薄 SVD 只返回 $r=\min(N,D)$ 个奇异方向；
- 也可对 $N\times N$ Gram 矩阵 $\mathbf X_c\mathbf X_c^\top$ 求解，再映回特征空间；
- 只需要少量方向时，可用 Lanczos、随机 SVD 或增量算法；
- 不要为了“接口固定为 $D$ 维”补造没有数据支撑的零方差轴。

若 $\mathbf u_j$ 是 $\mathbf X_c\mathbf X_c^\top$ 的单位特征向量，特征值为 $\sigma_j^2>0$，则

$$
\mathbf v_j=\frac{\mathbf X_c^\top\mathbf u_j}{\sigma_j}.
$$

除以接近零的奇异值会放大误差，因此必须先做秩门禁。

```python
rng_rank = np.random.default_rng(2511)
x_wide = rng_rank.normal(size=(18, 80))
x_wide -= x_wide.mean(axis=0)
_, singular_wide, vt_wide = np.linalg.svd(x_wide, full_matrices=False)
rank_wide = np.linalg.matrix_rank(x_wide)

assert vt_wide.shape == (18, 80)
assert rank_wide <= 17
assert np.count_nonzero(singular_wide > 1e-10) == rank_wide
print("宽矩阵形状 / 中心化秩：", x_wide.shape, rank_wide)
```

<MlQuiz
  kind="数值实现"
  question="为什么工程实现通常优先直接对中心化数据做 SVD，而不是先形成 XᵀX 再求特征向量？"
  :options="['显式 Gram 矩阵会平方条件数并增加数值损失', 'SVD 不需要中心化', 'SVD 会自动选择业务阈值', 'XᵀX 不是对称矩阵']"
  :answer="0"
  explanation="XᵀX 的谱是奇异值平方，条件数也平方；直接 SVD 避免主动恶化小方向的数值精度。"
/>

## 8. 从零实现：把训练合同写进接口

一个可交付的 PCA 实现至少要处理：二维形状、非有限值、零方差、分量数、训练期中心/尺度、方向规范和逆变换。下面仍只使用 NumPy。

```python
def _as_finite_matrix(x, expected_features=None):
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("输入必须是二维矩阵")
    if len(matrix) == 0 or matrix.shape[1] == 0:
        raise ValueError("输入不能为空")
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise ValueError("特征数量与已拟合模型不一致")
    if not np.isfinite(matrix).all():
        raise ValueError("输入包含 NaN 或无穷大")
    return matrix

def canonicalize_component_signs(components):
    result = np.asarray(components, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = np.argmax(np.abs(result[:, column]))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    return result

def fit_pca(x, n_components, standardize=True):
    matrix = _as_finite_matrix(x)
    n_samples, n_features = matrix.shape
    max_components = min(n_samples - 1, n_features)
    if not 1 <= n_components <= max_components:
        raise ValueError(f"n_components 必须在 1 到 {max_components} 之间")

    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0) if standardize else np.ones(n_features)
    if np.any(scale <= 0):
        raise ValueError("训练数据存在零方差特征")

    centered = (matrix - mean) / scale
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components_all = canonicalize_component_signs(vt.T)
    eigenvalues_all = singular**2 / n_samples
    q = int(n_components)
    return {
        "mean": mean,
        "scale": scale,
        "components": components_all[:, :q],
        "eigenvalues": eigenvalues_all[:q],
        "all_eigenvalues": eigenvalues_all,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_components": q,
        "standardize": bool(standardize),
    }

def transform_pca(x, model):
    matrix = _as_finite_matrix(x, model["n_features"])
    normalized = (matrix - model["mean"]) / model["scale"]
    return normalized @ model["components"]

def inverse_transform_pca(scores, model):
    scores = _as_finite_matrix(scores, model["n_components"])
    normalized = scores @ model["components"].T
    return normalized * model["scale"] + model["mean"]

PRESS_PCA = fit_pca(X_RAW[train_mask], n_components=3, standardize=True)
Z_TRAIN = transform_pca(X_RAW[train_mask], PRESS_PCA)
X_TRAIN_RECON = inverse_transform_pca(Z_TRAIN, PRESS_PCA)

assert Z_TRAIN.shape == (384, 3)
assert X_TRAIN_RECON.shape == X_RAW[train_mask].shape
assert np.allclose(PRESS_PCA["mean"], scale_mean)
assert np.allclose(PRESS_PCA["scale"], scale_std)
print("得分 / 重构形状：", Z_TRAIN.shape, X_TRAIN_RECON.shape)
```

三个实现细节不应被隐藏：

1. 这里使用总体形式 $1/N$，因此 `ddof=0`，特征值是 $\sigma_j^2/N$；
2. 符号规范只稳定接口显示，不增加统计可辨识性；
3. `inverse_transform_pca` 的结果是低秩近似，不是“恢复被删除的信息”。

### 8.1 负面输入和代数不变量

```python
invalid_inputs = [
    np.array([1.0, 2.0]),
    np.empty((0, 3)),
    np.array([[1.0, np.nan], [2.0, 3.0]]),
]
for invalid in invalid_inputs:
    try:
        fit_pca(invalid, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("无效输入没有被拒绝")

try:
    transform_pca(np.ones((2, 11)), PRESS_PCA)
except ValueError:
    pass
else:
    raise AssertionError("错误特征数没有被拒绝")

try:
    fit_pca(np.column_stack([np.arange(4), np.ones(4)]), 1)
except ValueError:
    pass
else:
    raise AssertionError("零方差特征没有被拒绝")

orthogonality_error = np.max(np.abs(
    PRESS_PCA["components"].T @ PRESS_PCA["components"] - np.eye(3)
))
assert orthogonality_error < 1e-12
print("方向正交最大误差：", f"{orthogonality_error:.3e}")
```

### 8.2 符号与子空间不变量

若把第 $j$ 个方向与对应得分同时乘以 $-1$，重构完全不变：

$$
( -\mathbf z_j)(-\mathbf w_j)^\top
=\mathbf z_j\mathbf w_j^\top.
$$

```python
basis_original = PRESS_PCA["components"].copy()
scores_original = Z_TRAIN.copy()
basis_flipped = basis_original.copy()
scores_flipped = scores_original.copy()
basis_flipped[:, 1] *= -1
scores_flipped[:, 1] *= -1

recon_original_std = scores_original @ basis_original.T
recon_flipped_std = scores_flipped @ basis_flipped.T
projector_original = basis_original @ basis_original.T
projector_flipped = basis_flipped @ basis_flipped.T

assert np.allclose(recon_original_std, recon_flipped_std, atol=1e-12)
assert np.allclose(projector_original, projector_flipped, atol=1e-12)
print("符号翻转后的最大重构差：",
      f"{np.max(np.abs(recon_original_std - recon_flipped_std)):.3e}")
```

跨版本接口若直接把 `PC2 > 0` 当作业务规则，会因合法的符号翻转而失效。若下游必须按轴解释，应保存符号锚点，例如“绝对载荷最大的特征取正”；若只关心距离或重构，使用子空间不变量更稳健。

## 9. 维数选择：95% 不是自然定律

### 9.1 三类目标会给出不同的 $q$

常见选择依据包括：

| 目标 | 可用证据 | 主要风险 |
|---|---|---|
| 压缩/重构 | 解释方差、重构预算、存储与延迟 | 高方差不等于高业务价值 |
| 可视化 | 2–3 维可读性、邻域与结构稳定性 | 二维图可能隐藏关键方向 |
| 下游监督任务 | 训练折内 PCA + 验证任务指标 | 全数据预处理泄漏 |
| 过程监测 | 正常误报、受控异常检出、子空间稳定性 | 用测试故障反复调参 |

“累计解释方差达到 95%”只是一个可审计规则，不是统计定理。本案例恰好在 $q=3$ 达到 95.26%，且生成器有三个共同因子；真实现场通常不知道“真维数”。

### 9.2 重构误差不能自己产生 U 形选择曲线

固定同一个训练子空间序列，增加 $q$ 会扩大可表示空间，所以训练和任意固定测试样本的正交重构误差都不会增加。这意味着“选择测试重构误差最小的 $q$”通常只会选到最大允许维数。

```python
def reconstruction_mse(x, mean, components, q):
    centered = x - mean
    basis = components[:, :q]
    residual = centered - (centered @ basis) @ basis.T
    return np.mean(np.sum(residual**2, axis=1))

q_rows = []
for q in range(1, 7):
    train_mse = reconstruction_mse(
        X_TRAIN, PCA_MODEL["mean"], PCA_MODEL["components"], q
    )
    val_mse = reconstruction_mse(
        X_VAL, PCA_MODEL["mean"], PCA_MODEL["components"], q
    )
    q_rows.append((q, CEVR[q - 1], train_mse, val_mse))

q_rows = np.asarray(q_rows)
assert np.all(np.diff(q_rows[:, 2]) <= 1e-12)
assert np.all(np.diff(q_rows[:, 3]) <= 1e-12)
print("q / CEVR / train MSE / validation MSE")
for q, cevr, train_mse, val_mse in q_rows:
    print(int(q), f"{cevr:.4f}", f"{train_mse:.4f}", f"{val_mse:.4f}")
```

验证集含有刻意注入的子空间外变化，所以增加维数还会把一部分“异常”吸收到模型中。用于监测时，不能只追求小重构误差；必须同时审计正常误报和受控偏差的检出。

### 9.3 谱隙、scree 与平行分析

按大小绘制 $\lambda_j$ 得到 scree plot。谱隙

$$
\Delta_q=\lambda_q-\lambda_{q+1}
$$

反映 $q$ 维主子空间与下一方向的分离程度。明显折点是证据，但肉眼折点可能主观。平行分析则把真实特征值与保持样本数、列数和边缘尺度的随机基线比较；只保留显著高于噪声谱的方向。它仍依赖零模型选择，不应包装成唯一真值。

本案例的第三、第四特征值为 1.4216 与 0.2092，边界谱隙明显；第 4 维之后则主要是独立量测噪声。

### 9.4 子空间稳定性比逐列载荷稳定性更可靠

对 campaign 做分组 bootstrap，每次重拟合前三维子空间。两个正交基 $\mathbf W$ 与 $\widetilde{\mathbf W}$ 的奇异值满足

$$
\sigma_j(\mathbf W^\top\widetilde{\mathbf W})
=\cos\theta_j,
$$

其中 $\theta_j$ 是主夹角。最大主夹角接近零表示两个子空间接近。

```python
def largest_principal_angle_deg(basis_a, basis_b):
    singular = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    singular = np.clip(singular, -1.0, 1.0)
    return np.degrees(np.arccos(singular.min()))

rng_boot = np.random.default_rng(2517)
train_groups = np.unique(CAMPAIGN[train_mask])
reference_basis = PCA_MODEL["components"][:, :3]
bootstrap_angles = []

for _ in range(80):
    sampled_groups = rng_boot.choice(
        train_groups, size=len(train_groups), replace=True
    )
    sampled_rows = np.concatenate([
        np.flatnonzero(CAMPAIGN == group) for group in sampled_groups
    ])
    boot_model = fit_pca_svd(X[sampled_rows])
    bootstrap_angles.append(largest_principal_angle_deg(
        reference_basis, boot_model["components"][:, :3]
    ))

bootstrap_angles = np.asarray(bootstrap_angles)
assert np.isfinite(bootstrap_angles).all()
print("最大主夹角 50% / 95% 分位：",
      np.round(np.quantile(bootstrap_angles, [.5, .95]), 3))
```

这里按 campaign 重采样，而不是按行重采样，因为同一批组共享环境偏移。稳定性只说明对当前抽样扰动不敏感，不证明物理语义正确。

<MlQuiz
  kind="模型选择"
  question="为什么不能简单选择验证重构误差最小的 PCA 维数？"
  :options="['随着子空间扩大，固定样本的正交重构误差通常单调下降', '验证集不能包含连续变量', '特征值必须先归一为负数', 'PCA 维数只能取 2']"
  :answer="0"
  explanation="更大的子空间包含更小的子空间，表示能力只增不减。维数必须结合压缩预算、下游任务或监测目标选择。"
/>

## 10. 载荷、得分与解释：不要把方向命名成因果

### 10.1 三种容易混淆的量

- **方向载荷** $w_{dj}$：第 $j$ 个单位主方向在第 $d$ 个标准化特征轴上的坐标；
- **样本得分** $z_{ij}$：样本 $i$ 沿第 $j$ 个方向的位置；
- **相关载荷**：标准化数据中，特征与得分的相关系数，常写为 $w_{dj}\sqrt{\lambda_j}$。

载荷绝对值大只表示该列参与这个统计方向，不代表它是物理根因。相关特征还会分摊载荷；更换尺度、样本域或加入近重复列都可能改变分配。

```python
component_table = []
for j in range(3):
    loading = PRESS_PCA["components"][:, j]
    top = np.argsort(np.abs(loading))[::-1][:4]
    component_table.append([
        (FEATURE_NAMES[index], float(loading[index])) for index in top
    ])

for j, rows in enumerate(component_table, start=1):
    print(f"PC{j} 绝对载荷前四：",
          [(name, round(value, 3)) for name, value in rows])
```

本案例可以把前三维暂时描述为“负载共变方向”“行程—节拍方向”“热—电方向”，但这些名称只是工程阅读标签。只有受控试验、计量溯源和机理分析才能把统计方向连接到物理因果。

### 10.2 biplot 的长度和角度都有前提

biplot 同时画样本得分和特征向量，但它把两个不同空间缩放到同一图上。箭头夹角近似变量相关性，只在特定缩放约定和保留维数足够时成立；二维图中短箭头可能只是该特征主要落在被省略方向，而不是“不重要”。

::: ml-stop 停下来推一推 3
若 PC1 的峰值力载荷由 $+0.42$ 变成 $-0.42$，同时所有 PC1 得分反号，散点图和重构各发生什么变化？如果下游规则写成 `PC1 > 2`，为什么它却会失效？
:::

<MlQuiz
  kind="解释边界"
  question="PC1 上电流与扭矩的绝对载荷最大，可以得出哪项结论？"
  :options="['它们对该统计方向贡献较大，但不能仅凭载荷断言物理因果', '电流一定导致扭矩变化', '其余特征可以删除', 'PC1 就是真实故障标签']"
  :answer="0"
  explanation="PCA 描述相关几何。载荷受尺度、共线性和样本域影响，因果解释需要额外设计与证据。"
/>

## 11. 白化：把方差也抹平的额外变换

普通 PCA 得分满足

$$
\operatorname{Cov}(\mathbf Z_q)
=\operatorname{diag}(\lambda_1,\ldots,\lambda_q).
$$

各维不相关，但方差不同。PCA 白化进一步定义

$$
\widetilde{\mathbf Z}_q
=\mathbf Z_q
\operatorname{diag}\left((\lambda_j+\epsilon)^{-1/2}\right),
$$

使训练白化得分的协方差近似单位阵。$\epsilon$ 是显式的数值/正则合同，不是随手加的小数。

```python
q = 3
scores_3 = xc_train @ PCA_MODEL["components"][:, :q]
whitened_3 = scores_3 / np.sqrt(EIGENVALUES[:q])
whitened_cov = whitened_3.T @ whitened_3 / len(whitened_3)

assert np.allclose(whitened_cov, np.eye(q), atol=1e-12)
print("白化协方差最大单位阵误差：",
      f"{np.max(np.abs(whitened_cov - np.eye(q))):.3e}")
```

白化不是“更好的 PCA”。它会：

- 删除各主方向原本的相对能量；
- 把小特征值方向放大更多；
- 改变欧氏距离和下游正则化的含义；
- 要求推理期使用训练特征值，不能按批次重新缩放。

```python
small_eigenvalues = np.array([9.0, 1.0, 1e-6])
equal_noise = np.array([.01, .01, .01])
whitened_noise = equal_noise / np.sqrt(small_eigenvalues)
amplification = whitened_noise / equal_noise

assert amplification[-1] == 1000.0
print("三个方向的白化噪声放大倍率：", amplification)
```

PCA whitening 的坐标轴是主方向；ZCA whitening 再旋回原特征坐标，得到

$$
\mathbf X_{\text{ZCA}}
=\mathbf X_c\mathbf V
\boldsymbol\Lambda^{-1/2}\mathbf V^\top.
$$

二者都需要截断或正则化小特征值。若目标只是消除相关性而不希望放大小噪声，普通 PCA 得分可能更合适。

<MlQuiz
  kind="白化边界"
  question="为什么把所有 12 个方向都白化可能放大量测噪声？"
  :options="['白化除以 √λ，小特征值方向会被强烈放大', '白化会删除中心化', '白化要求标签', '大特征值会变成无穷']"
  :answer="0"
  explanation="尾部方向通常能量小、信噪比低；除以很小的标准差会把其中噪声放大到单位方差。"
/>

## 12. 从重构到监测：$T^2$ 与 $Q$ 看的是两种偏离

PCA 过程监测常把样本拆成“模型内位置”和“模型外残差”。对训练期标准化的新样本 $\mathbf x_c$：

$$
\mathbf z=\mathbf W_q^\top\mathbf x_c,
\qquad
\mathbf e=(\mathbf I-\mathbf W_q\mathbf W_q^\top)\mathbf x_c.
$$

### 12.1 Hotelling $T^2$：子空间内走了多远

$$
T^2
=\mathbf z^\top\boldsymbol\Lambda_q^{-1}\mathbf z
=\sum_{j=1}^{q}\frac{z_j^2}{\lambda_j}.
$$

它按训练方差归一化每个保留方向。沿已知共变结构走得很远，会使 $T^2$ 大，即使重构很好。

### 12.2 $Q$ 或 SPE：离开子空间多远

$$
Q=\operatorname{SPE}=\|\mathbf e\|_2^2.
$$

它对新型相关关系、传感器不一致和未建模方向更敏感。一个样本可以：

- $T^2$ 高、$Q$ 低：沿已知工况方向远行；
- $T^2$ 低、$Q$ 高：靠近得分中心，但偏离相关结构；
- 两者都高：既远行又离开子空间；
- 两者都低：只表示在当前模型和门限内，不等于产品一定合格。

```python
def pca_diagnostics(x, model, q):
    matrix = _as_finite_matrix(x, model["n_features"])
    normalized = (matrix - model["mean"]) / model["scale"]
    basis = model["components"][:, :q]
    eigenvalues = model["all_eigenvalues"][:q]
    if np.any(eigenvalues <= 1e-12):
        raise ValueError("保留方向含近零特征值，无法稳定计算 T²")
    scores = normalized @ basis
    residual = normalized - scores @ basis.T
    t2 = np.sum(scores**2 / eigenvalues, axis=1)
    spe = np.sum(residual**2, axis=1)
    return scores, residual, t2, spe

_, _, T2_TRAIN, SPE_TRAIN = pca_diagnostics(
    X_RAW[train_mask], PRESS_PCA, 3
)
_, _, T2_VAL, SPE_VAL = pca_diagnostics(
    X_RAW[val_mask], PRESS_PCA, 3
)

T2_LIMIT = float(np.quantile(T2_TRAIN, .99))
SPE_LIMIT = float(np.quantile(SPE_TRAIN, .99))

assert T2_LIMIT > 0 and SPE_LIMIT > 0
print("训练经验 99% 门限 T² / Q：",
      round(T2_LIMIT, 4), round(SPE_LIMIT, 4))
```

这里使用训练经验分位数，是为了让门限假设透明。理论分布门限通常依赖独立、Gaussian、稳态和参数估计条件；实际过程若有自相关、campaign 效应或重尾，直接套公式会失准。更稳妥的做法是冻结基线后，在独立验证批组上校核误报率。

<PcaT2QMonitorLab />

### 12.3 贡献分解是线索，不是根因诊断

$Q$ 可逐列分解为 $e_d^2$；$T^2$ 也可按得分方向或通过线性映射回特征贡献。但相关变量会共享贡献，带符号的分解还可能互相抵消。贡献图回答“哪个量测最影响当前统计量”，不能自动回答“哪个部件先发生故障”。

<MlQuiz
  kind="监测语义"
  question="一个样本 T² 很高而 Q 很低，最准确的解释是什么？"
  :options="['它沿训练子空间中的已知共变方向走得很远，但仍能被子空间较好重构', '它一定是传感器断线', '它已经被证明为不合格品', 'PCA 方向不再正交']"
  :answer="0"
  explanation="T² 衡量模型内距离，Q 衡量模型外残差；两者报警语义不同，不能互相替代。"
/>

## 数值验证

### 13.1 先在验证批组冻结维数与门限

本案例预先规定：选择最小的 $q$ 使训练累计解释方差至少达到 90%，再要求分组 bootstrap 的 95% 最大主夹角不超过 15°，最后用独立验证批组核对正常误报和两类受控偏离。

```python
Q_SELECTED = int(np.flatnonzero(CEVR >= .90)[0] + 1)
ANGLE_P95 = float(np.quantile(bootstrap_angles, .95))

assert Q_SELECTED == 3
assert ANGLE_P95 < 15

val_alarm_t2 = T2_VAL > T2_LIMIT
val_alarm_spe = SPE_VAL > SPE_LIMIT
val_alarm_any = val_alarm_t2 | val_alarm_spe
val_normal = STATE_VAL == 0
val_anomaly = ~val_normal

val_fpr = val_alarm_any[val_normal].mean()
val_tpr = val_alarm_any[val_anomaly].mean()

assert val_fpr <= .08
assert val_tpr >= .90
print("冻结 q / CEVR / 主夹角 P95：",
      Q_SELECTED, round(CEVR[Q_SELECTED - 1], 4), round(ANGLE_P95, 3))
print("验证正常误报 / 受控偏离检出：",
      round(val_fpr, 4), round(val_tpr, 4))
```

这不是“用异常标签训练 PCA”。方向仍只由基线训练数据决定；验证状态只用于验收维数和门限是否满足预先写下的监测合同。若现场没有受控偏差样本，就不能声称已验证检出能力，只能报告基线覆盖和稳定性。

### 13.2 测试批组只打开一次

```python
_, RESIDUAL_TEST, T2_TEST, SPE_TEST = pca_diagnostics(
    X_RAW[test_mask], PRESS_PCA, Q_SELECTED
)

test_alarm_t2 = T2_TEST > T2_LIMIT
test_alarm_spe = SPE_TEST > SPE_LIMIT
test_alarm_any = test_alarm_t2 | test_alarm_spe

test_rows = []
for state, name in [
    (0, "baseline"),
    (1, "in_subspace_excursion"),
    (2, "off_subspace_conflict"),
]:
    mask = STATE_TEST == state
    test_rows.append({
        "state": name,
        "count": int(mask.sum()),
        "median_t2": float(np.median(T2_TEST[mask])),
        "median_q": float(np.median(SPE_TEST[mask])),
        "t2_alarm": float(test_alarm_t2[mask].mean()),
        "q_alarm": float(test_alarm_spe[mask].mean()),
        "either_alarm": float(test_alarm_any[mask].mean()),
    })

for row in test_rows:
    print(row["state"], row["count"],
          f"median T²={row['median_t2']:.3f}",
          f"median Q={row['median_q']:.3f}",
          f"T²/Q/either={row['t2_alarm']:.3f}/"
          f"{row['q_alarm']:.3f}/{row['either_alarm']:.3f}")
```

输出为：

```text
baseline 74 median T²=2.027 median Q=0.525 T²/Q/either=0.027/0.027/0.054
in_subspace_excursion 7 median T²=18.499 median Q=0.736 T²/Q/either=1.000/0.000/1.000
off_subspace_conflict 15 median T²=2.922 median Q=22.394 T²/Q/either=0.000/1.000/1.000
```

结果刻意展示两类几何偏离：子空间内远行只触发 $T^2$，传感器关系冲突只触发 $Q$。若只部署其中一个统计量，必然漏掉另一类。测试基线 5.4% 的任一门报警也说明两个各为 1% 的训练边缘门限，在 campaign 移动和联合使用下不保证总体仍为 1%。

### 13.3 重构不变量与独立参考实现

冻结模型还应通过以下门禁：

- 变换后形状正确、值有限；
- 投影矩阵对称且幂等；
- 直接重构与 `inverse_transform` 一致；
- $Q$ 等于逐列残差平方和；
- 同一输入重复推理完全确定。

```python
test_scores = transform_pca(X_RAW[test_mask], PRESS_PCA)
test_reconstructed = inverse_transform_pca(test_scores, PRESS_PCA)
test_normalized = (X_RAW[test_mask] - PRESS_PCA["mean"]) / PRESS_PCA["scale"]
test_reconstructed_normalized = (
    (test_reconstructed - PRESS_PCA["mean"]) / PRESS_PCA["scale"]
)
projector = PRESS_PCA["components"] @ PRESS_PCA["components"].T

assert np.allclose(projector, projector.T, atol=1e-12)
assert np.allclose(projector @ projector, projector, atol=1e-12)
assert np.allclose(
    test_reconstructed_normalized,
    test_normalized @ projector,
    atol=1e-12,
)
assert np.allclose(SPE_TEST, np.sum(RESIDUAL_TEST**2, axis=1))
assert np.array_equal(
    transform_pca(X_RAW[test_mask], PRESS_PCA), test_scores
)
print("投影幂等最大误差：",
      f"{np.max(np.abs(projector @ projector - projector)):.3e}")
```

<MlQuiz
  kind="验收边界"
  question="测试集上的受控偏离检出率为 100%，可以直接宣布现场未知故障检出率也是 100% 吗？"
  :options="['不能；这里只覆盖生成器规定的两类偏离和当前批组分布', '能；PCA 对所有故障都最优', '能；因为没有使用标签训练', '只要解释方差超过 90% 就能']"
  :answer="0"
  explanation="合成验收只能证明代码对预设几何变化的行为。真实故障谱、传感器失效和分布移动都可能不同。"
/>

## 14. 失效审计一：离群点会旋转主方向

普通 PCA 使用均值和平方损失，单个高杠杆样本能同时移动中心、尺度和协方差。下面在二维稳定数据中加入一个极端点，比较第一方向旋转。

```python
rng_outlier = np.random.default_rng(2521)
base = rng_outlier.normal(size=(120, 2)) @ np.array([[2.2, .0], [.9, .35]])
base -= base.mean(axis=0)

def canonical_first_direction(x):
    model = fit_pca_svd(x)
    direction = model["components"][:, 0].copy()
    if direction[0] < 0:
        direction *= -1
    return direction

direction_clean = canonical_first_direction(base)
direction_contaminated = canonical_first_direction(
    np.vstack([base, [0.0, 35.0]])
)
rotation_deg = np.degrees(np.arccos(np.clip(
    abs(direction_clean @ direction_contaminated), -1.0, 1.0
)))

assert rotation_deg > 20
print("单个高杠杆点造成的第一方向旋转：", round(rotation_deg, 3), "度")
```

可选防护包括：

1. 在数据边界先做物理量程、传感器状态和单位校验；
2. 用独立审核的基线窗口拟合，不把已知故障混入“正常子空间”；
3. 对明显重尾数据比较 robust covariance、robust PCA 或迭代重加权方案；
4. 报告离群点对主夹角和门限的敏感性；
5. 不要先凭同一个 PCA 的 $Q$ 删除样本，再声称该 PCA 证明剩余样本正常。

“robust”不是一个统一算法名。使用任何稳健变体前，都要写明污染模型、损失、收敛条件和新引入的超参数。

## 15. 失效审计二：最大方差可能丢掉最有用的标签方向

PCA 不读取 $y$。若类别差异位于低方差方向，保留最高方差维可能让分类信息消失。

```python
rng_label = np.random.default_rng(2527)
n_label = 400
y_label = np.repeat([-1.0, 1.0], n_label // 2)
x_high_variance = rng_label.normal(0, 8.0, n_label)
x_low_variance_signal = 0.75 * y_label + rng_label.normal(0, .20, n_label)
x_label_demo = np.column_stack([x_high_variance, x_low_variance_signal])

label_pca = fit_pca_svd(x_label_demo)
pc1_score = label_pca["scores"][:, 0]
pc2_score = label_pca["scores"][:, 1]
corr_pc1 = abs(np.corrcoef(pc1_score, y_label)[0, 1])
corr_pc2 = abs(np.corrcoef(pc2_score, y_label)[0, 1])

assert label_pca["eigenvalues"][0] > 50 * label_pca["eigenvalues"][1]
assert corr_pc1 < .15
assert corr_pc2 > .90
print("PC1/PC2 与标签绝对相关：", round(corr_pc1, 3), round(corr_pc2, 3))
```

若 PCA 是监督流水线的一部分，$q$ 必须在训练折内与下游模型共同验证。其他可选方法包括 PLS、监督降维、正则化模型或直接保留原特征。不能把无监督 CEVR 当作标签信息保真率。

<MlQuiz
  kind="反例"
  question="累计解释方差达到 95%，为何仍可能严重损害分类？"
  :options="['类别差异可能位于被丢弃的低方差方向', '解释方差只能用于回归', 'PCA 会随机翻转标签', '95% 必须改为 100% 才能分类']"
  :answer="0"
  explanation="PCA 只优化输入方差，不看标签。方差小的方向完全可能有很高的判别价值。"
/>

## 16. 缺失、漂移与模型版本

### 16.1 缺失值不能用推理批次均值临时填

标准 SVD 需要完整有限矩阵。常见路线包括：

- 使用训练期冻结的逐列填充值，并为缺失机制建立独立特征与验收；
- 迭代低秩补全，但必须在训练折内拟合并验证收敛；
- 使用 probabilistic PCA / factor analysis 对缺失维做条件推断；
- 若关键量测缺失，直接拒绝推理并转人工流程。

用当前批次均值填补会让同一件产品的得分依赖同批其他产品，并把批次分布信息偷偷带入变换。若缺失不是随机的，任何简单填补还可能掩盖真实状态。

### 16.2 重拟合会改变坐标系

新数据到来后直接重拟合 PCA，可能发生：

- 均值和尺度漂移；
- 特征值顺序交换；
- 合法符号翻转；
- 近重复特征值子空间内旋转；
- 异常被吸收到新的“正常”子空间，导致 $Q$ 降低。

因此在线监控至少应同时记录：原始特征质量、均值/尺度漂移、特征值谱、旧新子空间主夹角、$T^2/Q$ 分布、门限覆盖率和下游任务指标。重训必须产生新模型版本，并经过回放和批准；不能在服务中按批静默更新。

### 16.3 坐标对齐只解决接口，不解决语义漂移

若两个版本都保留 $q$ 维，可用正交 Procrustes 寻找

$$
\mathbf R^*
=\arg\min_{\mathbf R^\top\mathbf R=\mathbf I}
\|\mathbf W_{\text{new}}\mathbf R-\mathbf W_{\text{old}}\|_F.
$$

它能对齐旋转与符号，便于仪表连续显示；但若主夹角已经很大，强行对齐只是掩盖模型变化。接口对齐不能替代漂移审批。

## 17. 概率 PCA：给正交子空间加上噪声模型

普通 PCA 是确定性投影。概率 PCA（probabilistic PCA, PPCA）写成线性 Gaussian 潜变量模型：

$$
\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I_q),
\qquad
\mathbf x\mid\mathbf z
\sim\mathcal N(\boldsymbol\mu+\mathbf W\mathbf z,\sigma^2\mathbf I_D).
$$

边缘分布为

$$
p(\mathbf x)
=\mathcal N\left(
\boldsymbol\mu,
\mathbf W\mathbf W^\top+\sigma^2\mathbf I_D
\right).
$$

若样本协方差前 $q$ 个特征向量组成 $\mathbf U_q$，对应特征值为 $\boldsymbol\Lambda_q$，PPCA 的一个 MLE 形式是

$$
\widehat{\mathbf W}
=\mathbf U_q
(\boldsymbol\Lambda_q-\widehat\sigma^2\mathbf I)^{1/2}
\mathbf R,
$$

其中 $\mathbf R$ 是任意正交旋转，丢弃方向的平均方差估计为

$$
\widehat\sigma^2
=\frac1{D-q}\sum_{j=q+1}^{D}\lambda_j.
$$

在 $\sigma^2\to0$ 极限，后验均值回到正交投影。PPCA 的价值在于：

- 能给出观测密度和潜变量后验；
- 可用 EM 处理缺失数据与扩展模型；
- 可用 likelihood 讨论维数；
- 明确区分共同低秩变化与各向同性残余噪声。

但各向同性噪声往往不符合工业传感器实际；factor analysis 用对角残余协方差放宽这一点。第 24 章的 EM 可以用来拟合这些潜变量模型，但标准 PCA 本身不需要迭代 EM。

```python
q_ppca = 3
sigma2_ppca = EIGENVALUES[q_ppca:].mean()
ppca_loading_scales = np.sqrt(EIGENVALUES[:q_ppca] - sigma2_ppca)

assert sigma2_ppca > 0
assert np.all(EIGENVALUES[:q_ppca] > sigma2_ppca)
print("PPCA 各向同性残余方差：", round(sigma2_ppca, 6))
print("前三个 loading scale：", np.round(ppca_loading_scales, 4))
```

## 18. 什么时候普通 PCA 不够

| 现象 | 普通 PCA 的限制 | 更合适的候选 |
|---|---|---|
| 数据沿弯曲流形 | 单一全局线性子空间会切穿曲线 | kernel PCA、局部方法、autoencoder；需单独验证样本外映射 |
| 需要稀疏可读载荷 | 正交特征向量通常稠密 | sparse PCA、旋转后的因子模型 |
| 有大量离群点/稀疏污染 | 平方损失与均值不稳健 | robust covariance、robust PCA |
| 流式超大数据 | 完整 SVD 内存或延迟过高 | incremental/streaming PCA、随机 SVD |
| 重点是标签预测 | 最大输入方差与任务可能错位 | PLS、监督表示或折内联合选择 |
| 观测缺失且需不确定性 | 普通 SVD 无后验 | PPCA、factor analysis 或明确补全模型 |

非线性 autoencoder 若使用线性编码器、线性解码器、平方重构损失、单瓶颈层并找到全局最优，其张成子空间与 PCA 有密切关系；一旦加入非线性、正则化、非正交约束或其他损失，就不再是同一个目标。

## 复杂度与工程边界

### 19.1 训练、推理与存储

对 $N\times D$ 的稠密中心矩阵，完整 SVD 的典型成本可粗略写为

$$
O\bigl(\min(ND^2,N^2D)\bigr),
$$

存储中心数据需要 $O(ND)$。只求前 $q$ 个方向时，迭代/随机方法通常把核心乘法降到约 $O(NDq)$，但还要计入幂迭代、正交化和误差校核。

冻结后，单样本变换成本是 $O(Dq)$，模型至少保存 $2D+Dq+q$ 个浮点数：均值、尺度、方向和特征值。

```python
n_large, d_large, q_large = 1_000_000, 120, 8
float_bytes = 8
data_gib = n_large * d_large * float_bytes / 1024**3
model_kib = (
    2 * d_large + d_large * q_large + q_large
) * float_bytes / 1024
scores_gib = n_large * q_large * float_bytes / 1024**3

print("100 万×120 原矩阵：", round(data_gib, 3), "GiB")
print("q=8 模型：", round(model_kib, 3), "KiB")
print("100 万×8 得分：", round(scores_gib, 3), "GiB")
```

随机 SVD 不是“无损加速”。必须记录随机种子、oversampling、幂迭代次数，并以谱误差、投影矩阵差或重构差对精确小样本基线做交叉核对。

### 19.2 必须交付的模型制品

一个可审计的 PCA 制品至少包含：

1. 特征名、顺序、单位、类型、允许范围和缺失策略；
2. 训练 campaign 清单、时间窗、审核基线定义和排除规则；
3. 均值、尺度类型、尺度数值和 `ddof`；
4. 主方向、特征值、解释方差、$q$ 与符号规范；
5. 训练/验证的谱、主夹角、重构、$T^2/Q$ 门限与覆盖率；
6. 数值库版本、算法、随机参数和模型哈希；
7. 允许动作、人工复核路径、回退策略和重训审批规则。

只保存 `components.npy` 不是可部署模型。缺少特征顺序或尺度时，一个形状仍正确的矩阵乘法也可能产生完全错误但数值有限的结果。

### 19.3 线上监控

| 监控项 | 目的 | 触发后动作 |
|---|---|---|
| schema、单位、非有限值 | 防止数据合同错位 | 拒绝推理并记录输入错误 |
| 均值/尺度漂移 | 发现总体工况变化 | 工程复核，不自动重拟合 |
| 特征值谱与主夹角 | 发现相关结构变化 | 候选模型离线回放 |
| $T^2/Q$ 分布与联合报警率 | 区分模型内/外偏离 | 分层抽样与传感器核验 |
| 重构贡献 | 定位影响统计量的量测 | 作为复核线索，不命名根因 |
| 下游任务指标 | 防止压缩损害最终目标 | 回退原模型或重新选 $q$ |

报警不能直接触发参数修改。特别是模型外 $Q$ 高，可能来自真实新工况，也可能来自单位切换、传感器更换、时间对齐错误或数据管道缺列。

## 20. 与成熟库交叉核对

下面的可选代码只用于本地安装了 scikit-learn 的环境。仓库标准验证不会为一段对照新增大型依赖。

```python
# 不执行：可选 scikit-learn 对照
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=True, with_std=True)
x_train_scaled = scaler.fit_transform(X_RAW[train_mask])
reference = PCA(n_components=3, svd_solver="full", whiten=False)
reference.fit(x_train_scaled)

our_projector = PRESS_PCA["components"] @ PRESS_PCA["components"].T
reference_projector = reference.components_.T @ reference.components_
assert np.allclose(our_projector, reference_projector, atol=1e-10)
assert np.allclose(
    PRESS_PCA["eigenvalues"],
    reference.explained_variance_ * (len(x_train_scaled) - 1) / len(x_train_scaled),
    atol=1e-10,
)
```

scikit-learn 的 `explained_variance_` 使用无偏协方差分母 $N-1$，而本章推导使用 $N$；上面的比例换算正是接口合同的一部分。若忽略 `ddof`，方向仍可能一致，但特征值、白化和 $T^2$ 会出现系统比例差。

## 21. 练习

### 练习 1：手算二维 PCA

给定中心化样本

$$
(-2,-1),\ (-1,0),\ (1,0),\ (2,1),
$$

使用分母 $N$：

1. 写出协方差矩阵；
2. 求特征值和单位特征向量；
3. 写出一维得分与重构；
4. 用丢弃特征值核对平均重构误差；
5. 把所有样本加上 $(100,20)$，解释为何正确中心化后方向不变。

### 练习 2：证明多维重构恒等式

从 $\mathbf P=\mathbf W\mathbf W^\top$ 的对称、幂等性质出发，证明

$$
\|\mathbf X_c-\mathbf X_c\mathbf P\|_F^2
=\|\mathbf X_c\|_F^2
-\operatorname{tr}(\mathbf W^\top\mathbf X_c^\top\mathbf X_c\mathbf W).
$$

要求标出每一步使用的 trace 循环性质，并说明为何正交约束不可删除。

### 练习 3：数组形状与边界

对 $N=40,D=100,q=8$：

1. 写出薄 SVD 中 $\mathbf U,\boldsymbol\Sigma,\mathbf V^\top$ 的形状；
2. 说明中心化后最大秩；
3. 写出得分和重构形状；
4. 解释为何不能请求 40 个非零主成分；
5. 设计对空矩阵、NaN、错误特征顺序和零方差列的拒绝测试。

### 练习 4：失败案例解释

某团队发现换用新传感器后，累计解释方差仍为 96%，但 $Q$ 报警率从 2% 升到 35%。他们准备每天用最新数据自动重拟合 PCA。请分别检查：

- 单位与量程是否变化；
- 新旧均值、尺度和子空间主夹角；
- 时间对齐与缺失填补；
- 正常基线是否被污染；
- 自动重拟合为何可能把异常吸收进模型；
- 回放、审批和回退需要哪些证据。

### 练习 5：30–60 分钟，从公式到 NumPy

补全下面骨架，不得调用 PCA 库。验收要求：

- 训练数据和新数据均做形状/有限值校验；
- 只在训练数据拟合均值和尺度；
- 方向正交误差小于 $10^{-10}$；
- 特征分解与 SVD 的前三维投影矩阵差小于 $10^{-10}$；
- 训练平均重构误差等于丢弃特征值之和；
- 符号翻转不改变重构；
- campaign 分组 bootstrap 输出最大主夹角分位数；
- 对同一批测试样本同时输出 $T^2$ 与 $Q$。

```python
# TODO：练习骨架，不执行
import numpy as np

def fit_pca_from_scratch(x_train, q, standardize=True):
    # 1. 校验二维形状、有限值、q 和零方差
    # 2. 冻结训练均值与尺度
    # 3. 对中心化/标准化矩阵做薄 SVD
    # 4. 保存方向、完整特征值、符号规范和 schema
    raise NotImplementedError

def transform_from_scratch(x, model):
    # 使用冻结均值、尺度和方向；不得按当前批次重估
    raise NotImplementedError

def inverse_transform_from_scratch(scores, model):
    raise NotImplementedError

def t2_q_from_scratch(x, model):
    # 返回得分、重构、T²、Q 和逐列 Q 贡献
    raise NotImplementedError
```

## 22. 术语卡片

| 术语 | 本章中的精确定义 |
|---|---|
| 中心化 | 减去训练均值，使 PCA 描述围绕基线中心的变化 |
| 协方差 PCA | 在原尺度中心化数据上求主方向 |
| 相关系数 PCA | 逐列标准化后求主方向 |
| 主方向/载荷 | 协方差特征向量，即中心化数据的右奇异向量 |
| 得分 | 样本投影到主方向后的低维坐标 |
| 解释方差 | 主方向上的训练期投影方差，即对应特征值 |
| 投影矩阵 | $\mathbf W_q\mathbf W_q^\top$，描述子空间且不受列符号影响 |
| 重构 | 从低维得分投回原特征空间的正交低秩近似 |
| 白化 | 再除以主方向标准差，使保留得分协方差近似单位阵 |
| $T^2$ | 按主方向方差归一化的子空间内平方距离 |
| $Q$/SPE | 样本到保留子空间的平方正交残差 |
| 主夹角 | 比较两个子空间接近程度的旋转/符号不变量 |

## 23. 本章结论

1. PCA 的第一方向最大化 Rayleigh 商；Lagrange 条件给出特征方程，谱展开给出全局最优证明。
2. 前 $q$ 个方向最大化总投影方差，也最小化正交重构误差；最优误差等于丢弃特征值之和。
3. 中心化协方差的特征向量就是中心化数据的右奇异向量，特征值等于奇异值平方除以 $N$；直接 SVD 通常更稳。
4. 标准化不是机械步骤，它决定 PCA 保留原方差、相关结构还是工程公差意义。
5. 解释方差不等于标签信息；维数要按压缩、下游任务或监测合同选择。
6. $T^2$ 与 $Q$ 分别看子空间内远行和子空间外偏离，不能互相替代，也不能直接命名根因。
7. 符号、重复谱内旋转、离群点、缺失、漂移和重拟合都会影响接口；工程交付必须保存完整坐标合同。

## 来源与许可

本章关于最大方差、最小重构、特征分解、SVD、维数选择与概率 PCA 的知识边界，依据本地资料交叉核对：

- Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction*，PDF 第 687–696 页，§20.1–§20.1.4；概率 PCA 见 PDF 第 697–701 页，§20.2–§20.2.3；
- Stanford CS229 Lecture Notes，PDF 第 166–171 页，Chapter 12；
- Jonathan Richard Shewchuk，UC Berkeley CS 189/289A Lecture 20–21，合订讲义 PDF 第 123–131 页。

以上资料只用于核对公共知识、公式、术语和覆盖边界。本章伺服压装案例、576 条合成记录、十二路字段、两类几何偏离、所有 NumPy 代码、题目和三项网页交互均由本站独立创作；没有复制来源中的图形、数据、案例、代码或措辞。详细登记见 [课程资料来源与许可](./sources.md) 。

## 24. 与下一章连接

PCA 为每个样本给出连续低维坐标，但不会把样本划成离散簇。下一章从簇内平方和目标出发推导 K-means：固定质心时为什么选最近簇，固定分配时为什么更新为均值，以及初始化、局部最优和空簇如何破坏“看起来很简单”的两步循环。

<MlLessonProgress
  storage-key="25-pca"
  lesson-id="25"
  next-href="/courses/machine-learning/26-kmeans"
  next-label="下一章：K-means 聚类"
/>
