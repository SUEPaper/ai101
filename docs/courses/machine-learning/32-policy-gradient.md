---
title: REINFORCE 与 Actor–Critic
description: 从晶圆载具随机策略出发，推导轨迹目标、REINFORCE、reward-to-go、baseline、Actor–Critic、GAE 与 PPO 的工程实现边界
course: machine-learning
edition: industrial
---

<MlLessonHero
  number="32"
  title="REINFORCE 与 Actor–Critic"
  question="价值方法先估 Q 再取最大；如果直接输出可采样的动作分布，怎样从轨迹概率得到无模型梯度，并用 Critic 把高方差的回报信号变成可训练的 advantage？"
  duration="建议 840–960 分钟"
  :outcomes="[
    '把工业序贯决策写成带动作概率、log-prob、策略版本、动作 mask、终止与截断语义的 on-policy rollout contract',
    '从轨迹分布和对数导数恒等式逐行推导 REINFORCE、因果 reward-to-go 与折扣策略梯度定理',
    '证明动作无关 baseline 不改变期望梯度，区分无偏方差缩减、近似 Critic 的 bootstrap 偏差和有限批标准化',
    '实现表格 Softmax REINFORCE、一步 Actor–Critic、n-step/GAE，并用解析 oracle、有限差分与多种子评价验收',
    '解释离散/连续 Actor、Critic stop-gradient、熵正则、PPO 裁剪、SAC 边界及真实工业系统的发布禁区'
  ]"
/>

> 本章所有探索、训练与策略采样只允许在合格仿真器、数字孪生或隔离教学环境中发生。代码不得连接真实 AMR、下发路线、修改速度或避障参数；门禁、人员隔离、急停、洁净区许可和动作合法性必须由独立硬安全层强制执行，不能交给奖励函数、Critic 或熵正则“学会遵守”。

## 贯穿案例：让晶圆载具直接学习路线分布

### 1. 同一条走廊，第三种学习接口

第 30 章已知完整转移核，用动态规划求最优策略；第 31 章只看 transition，用 SARSA 或 Q-learning 学动作价值。本章仍使用同一条 `4 × 4` 教学走廊：

- 左下角是任务起点，右上角是交付口；
- 两格隔墙不可穿越，交付口下方是隔离终止区；
- 普通一步奖励为 $-0.45$，进入交付口为 $+14$，进入隔离区为 $-22$；
- 请求方向以 $0.18$ 概率偏到左/右相邻方向；
- 折扣因子 $gamma=0.94$。

本章不再维护 $Q(s,a)$ 后取 $argmax$，而让 Actor 直接输出

$$
\pi_\theta(a\mid s)=\Pr_\theta(A_t=a\mid S_t=s).
$$

同一状态可以保留多个动作的非零概率。随机策略不是装饰：它让动作分布可微，能表达多峰行为，也为仿真探索保留概率质量。但随机性不等于许可；非法动作必须在进入 Softmax 前由外部 `action_mask` 删除。

<MlQuiz
  kind="算法选择"
  question="本章的 Actor 与第 31 章 Q-learning 最直接的输出差异是什么？"
  :options="['Actor 直接输出动作分布；Q-learning 输出动作价值并通常再做贪心或探索选择', 'Actor 不使用状态', 'Q-learning 必须知道完整转移矩阵', 'Actor 的输出一定是确定动作']"
  :answer="0"
  explanation="策略方法参数化 π(a|s)；价值方法参数化 Q(s,a)，动作分布还要由 greedy、epsilon-greedy 等规则另行构造。"
/>

### 2. rollout contract：策略版本是数据的一部分

一条训练记录至少冻结：

| 字段组 | 示例字段 | 为什么不能省 |
|---|---|---|
| 轨迹身份 | `episode_id`、`step_id`、`collector_id` | 还原时间顺序，阻止跨回合拼接 GAE |
| 决策上下文 | `state_id`、`observation_sha`、`action_mask_sha` | 证明策略看到了什么、哪些动作当时合法 |
| 抽样证据 | `action_id`、`behavior_prob`、`behavior_log_prob` | 重算 score、importance ratio 与抽样似然 |
| 策略身份 | `policy_sha`、`policy_version`、`normalizer_sha` | 防止把不同分布的 rollout 当成同一批 on-policy 数据 |
| 结果 | `reward`、`reward_terms`、`next_state_id` | 构造回报、TD target 并审计奖励单位 |
| 边界 | `terminated`、`terminal_reason`、`truncated`、`final_observation` | 决定 bootstrap 与 GAE 是否跨边界传播 |
| 重放 | `rng_seed`、`simulator_sha`、`map_sha`、`schema_version` | 定位环境漂移并做确定性复验 |

`behavior_log_prob` 必须是采样当时、应用同一动作 mask 后的数值，不能用更新后的策略重算。若采集期间 Actor 已变化，还要按 `policy_version` 分批；否则所谓“on-policy batch”其实是未知混合分布。

<MlQuiz
  kind="数据合同"
  question="为什么只记录 action 而不记录 behavior_log_prob 与 policy_version，不足以实现 PPO 或审计 REINFORCE？"
  :options="['因为动作必须存成图片', '因为无法证明样本由哪条分布产生，也无法计算新旧策略概率比', '因为 reward 必须等于 log_prob', '因为策略梯度不需要轨迹']"
  :answer="1"
  explanation="动作相同不代表行为分布相同；概率与版本决定 score、重要性比和样本是否仍接近当前策略。"
/>

### 3. 唯一环境真相源：训练只调用 `env_step`

下面的 $P,R$ 只模拟隐藏环境。REINFORCE 和 Actor–Critic 不接收它们；它们只在本章末作为解析 oracle 验收梯度和冻结策略。

```python
import numpy as np

ROWS, COLS = 4, 4
WALLS = {(1, 1), (2, 1)}
GOAL, HAZARD = (0, 3), (1, 3)
CELLS = [
    (row, col)
    for row in range(ROWS)
    for col in range(COLS)
    if (row, col) not in WALLS
]
INDEX = {cell: index for index, cell in enumerate(CELLS)}
ACTIONS = ("N", "E", "S", "W")
DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
LEFT = {"N": "W", "E": "N", "S": "E", "W": "S"}
RIGHT = {value: key for key, value in LEFT.items()}
TERMINALS = {INDEX[GOAL], INDEX[HAZARD]}
NONTERMINALS = np.array([
    state for state in range(len(CELLS)) if state not in TERMINALS
])
START = INDEX[(3, 0)]
GAMMA = 0.94

def move(cell, action):
    if cell in {GOAL, HAZARD}:
        return cell
    dr, dc = DELTA[action]
    candidate = (cell[0] + dr, cell[1] + dc)
    return candidate if candidate in INDEX else cell

def build_hidden_model(slip=0.18):
    n_states, n_actions = len(CELLS), len(ACTIONS)
    probabilities = np.zeros((n_states, n_actions, n_states))
    rewards = np.zeros_like(probabilities)
    for state, cell in enumerate(CELLS):
        if state in TERMINALS:
            probabilities[state, :, state] = 1.0
            continue
        for action, intended in enumerate(ACTIONS):
            branches = (intended, LEFT[intended], RIGHT[intended])
            weights = (1.0 - slip, slip / 2.0, slip / 2.0)
            for actual, weight in zip(branches, weights):
                next_state = INDEX[move(cell, actual)]
                reward = (
                    14.0 if next_state == INDEX[GOAL]
                    else -22.0 if next_state == INDEX[HAZARD]
                    else -0.45
                )
                probabilities[state, action, next_state] += weight
                rewards[state, action, next_state] = reward
    return probabilities, rewards

HIDDEN_P, HIDDEN_R = build_hidden_model()

def env_step(rng, state, action):
    """训练器可调用的唯一环境接口；不返回概率行。"""
    next_state = int(rng.choice(
        len(CELLS), p=HIDDEN_P[state, action]
    ))
    reward = float(HIDDEN_R[state, action, next_state])
    return next_state, reward, next_state in TERMINALS

assert HIDDEN_P.shape == (14, 4, 14)
assert np.allclose(HIDDEN_P.sum(axis=2), 1.0)
```

<MlQuiz
  kind="信息边界"
  question="为什么训练代码不能读取 HIDDEN_P，但验收代码可以用它？"
  :options="['训练要保持 sample-only 问题定义；验收 oracle 用已知教学真相检查实现，不向更新泄漏', '因为 NumPy 不支持 P', '因为 Actor 必须是线性的', '因为验收不能计算价值']"
  :answer="0"
  explanation="算法声称 model-free，就不能在 target 中偷用转移核；合成环境的隐藏真相仍可作为测试夹具。"
/>

## 1. 目标不是一条最好轨迹，而是轨迹分布的期望

### 1.1 折扣回报与目标

令轨迹

$$
\tau=(s_0,a_0,r_1,s_1,\ldots,a_{T-1},r_T,s_T),
$$

从起点计算的回报为

$$
R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_{t+1},
$$

策略目标是

$$
J(\theta)
=\mathbb E_{\tau\sim p_\theta}[R(\tau)].
$$

这不是“找出训练集中回报最高的一条轨迹”。$	heta$ 同时改变每个决策的动作概率，进而改变以后访问哪些状态；优化对象是整条轨迹分布的期望。

若任务按固定时域、平均奖励或随机终止定义，目标会不同。实现必须把 $J$ 的定义写进实验卡；不能看见代码里有 $gamma$ 就假定所有回报、Critic target 和评价指标都采用同一折扣约定。

### 1.2 轨迹概率中谁依赖参数

在环境动力学不随 $	heta$ 改变时，

$$
p_\theta(\tau)
=\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

若奖励也随机，只需再乘 $p(r_{t+1}\mid s_t,a_t,s_{t+1})$；只要它不含 $	heta$，结论不变。取对数后，乘积变成求和：

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

初始分布、环境转移和奖励条件分布不是“被忽略”，而是它们对策略参数的偏导为零。若策略参数真的进入了可微物理模型，便是另一种 pathwise/model-based 梯度问题，不能继续套用这一消去步骤。

<MlQuiz
  kind="目标函数"
  question="轨迹中的状态分布明明随策略变化，为什么推导里仍不显式计算 ∂P/∂θ？"
  :options="['状态分布变化已通过每一步 πθ 的轨迹似然体现，而环境转移核本身不含 θ', '状态分布其实不变', 'P 恒等于 1', '因为奖励没有噪声']"
  :answer="0"
  explanation="策略改变动作概率，轨迹权重随之改变；对数似然把这条依赖路径汇总到各时刻的 log π 项。"
/>

## 2. 对数导数技巧：从不可枚举轨迹到可采样梯度

### 2.1 逐行推导 REINFORCE

离散轨迹下，

$$
J(\theta)=\sum_\tau p_\theta(\tau)R(\tau).
$$

对参数求导，并乘除 $p_\theta(\tau)$：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\sum_\tau R(\tau)\nabla_\theta p_\theta(\tau)\\
&=\sum_\tau p_\theta(\tau)R(\tau)
\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}\\
&=\mathbb E_{\tau\sim p_\theta}
\left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right]\\
&=\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)\sum_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\right].
\end{aligned}
$$

这就是 score-function estimator。它不对离散的采样动作求导，而对“这次抽到该动作的 log 概率”求导。一次轨迹给出一个无偏但通常很吵的 Monte Carlo 梯度样本。

<MlQuiz
  kind="推导审计"
  question="恒等式 ∇p = p∇log p 的真正作用是什么？"
  :options="['把对未知轨迹总和的概率导数改写成可从当前策略采样的期望', '把概率变成奖励', '保证方差为零', '允许对环境转移求导']"
  :answer="0"
  explanation="乘上 p 后，求和正好成为当前轨迹分布下的期望，因此可以用 rollout 做 Monte Carlo 估计。"
/>

### 2.2 Softmax score 是本章最小机械部件

离散 Actor 输出 logits $z_1,\ldots,z_K$：

$$
\pi(a=k\mid s)
=\frac{e^{z_k}}{\sum_j e^{z_j}}.
$$

对第 $j$ 个 logit，

$$
\frac{\partial\log\pi(a=k\mid s)}{\partial z_j}
=\mathbf 1[j=k]-\pi(a=j\mid s).
$$

所以一次正 advantage 更新会提高采样动作的 logit，同时按当前概率压低其他动作；所有分量之和为零，不会破坏概率归一化。实现中先减最大 logit，再指数化，避免溢出。

在一个双动作分流节点，设北干线概率 $p=\sigma(z)$，两动作期望回报为 $q_N,q_E$，则

$$
J(z)=pq_N+(1-p)q_E,
$$

$$
\boxed{
\frac{dJ}{dz}=p(1-p)(q_N-q_E)
}.
$$

当 $p$ 已接近 0 或 1，$p(1-p)$ 很小；即使另一动作更好，恢复也会很慢。这是过早策略塌缩的机械原因之一。

::: ml-stop 停下来推一推 1
从 $d\sigma(z)/dz=p(1-p)$ 出发，分别写出采样北干线和东捷径时的 score。再证明两种样本梯度按动作概率加权后，恰好得到 $p(1-p)(q_N-q_E)$。
:::

<PolicyGradientSignalBenchLab />

<MlQuiz
  kind="数值直觉"
  question="若北干线回报更高，但其概率已经是 0.999，为什么 logit 梯度仍可能很小？"
  :options="['因为 Softmax 的 p(1−p) 接近 0，策略已靠近概率单纯形边界', '因为高回报动作不能更新', '因为 baseline 一定等于回报', '因为环境必须可微']"
  :answer="0"
  explanation="饱和概率让局部 logit—概率映射变平；稀有动作又很少被采到，恢复信号进一步稀疏。"
/>

## 3. 因果性：当前动作不替过去奖励负责

### 3.1 从整轨迹回报到 reward-to-go

整轨迹估计器把同一个 $R(\tau)$ 乘给每个时刻的 score。但 $A_t$ 不可能改变 $t$ 之前已经发生的奖励。定义

$$
G_t=\sum_{\ell=0}^{T-t-1}\gamma^\ell R_{t+\ell+1}.
$$

对过去奖励项 $C(S_0,A_0,\ldots,S_t)$，条件于 $S_t$ 和过去历史：

$$
\mathbb E_{A_t\sim\pi_\theta}
\left[C\nabla_\theta\log\pi_\theta(A_t\mid S_t)\right]
=C\nabla_\theta\sum_a\pi_\theta(a\mid S_t)=0.
$$

因此对起点折扣目标，因果形式为

$$
\boxed{
\nabla_\theta J
=\mathbb E\left[
\sum_{t=0}^{T-1}
\gamma^tG_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\right].
}
$$

这里外层 $gamma^t$ 不能无声丢掉。很多实现省略它，是因为采用了不同的状态加权目标、把折扣吸收到随机终止，或只把 $gamma$ 当方差控制器。三种写法都可能有合理语境，但推导、代码和评价必须声明同一个目标。

### 3.2 折扣策略梯度定理

定义未归一化折扣访问量

$$
d_\gamma^\pi(s)
=\mathbb E_\pi\left[
\sum_{t=0}^\infty\gamma^t\mathbf 1[S_t=s]
\right].
$$

对 future return 取条件期望，得到

$$
\nabla_\theta J
=\sum_s d_\gamma^\pi(s)
\sum_a \pi_\theta(a\mid s)
Q^\pi(s,a)
\nabla_\theta\log\pi_\theta(a\mid s).
$$

因为 $pi\nabla\log\pi=\nabla\pi$，也可写成

$$
\nabla_\theta J
=\sum_s d_\gamma^\pi(s)
\sum_a Q^\pi(s,a)\nabla_\theta\pi_\theta(a\mid s).
$$

定理没有说状态分布可忽略；$d_\gamma^\pi$ 正是策略诱导的访问权重。妙处在于最终公式不需要显式求 $\nabla d_\gamma^\pi/\nabla\theta$。

<MlQuiz
  kind="折扣约定"
  question="若 Gt 已经从 t 时刻重新折扣，为什么起点折扣目标的因果梯度前仍有 γ^t？"
  :options="['它把 t 时刻的局部回报重新放回从起点计时的权重', '为了让概率和为 1', '为了替代 action mask', '因为 Critic 一定有偏']"
  :answer="0"
  explanation="Gt 的第一项权重为 1；乘 γ^t 后才与原目标中 R_{t+1} 的 γ^t 权重一致。"
/>

## 4. baseline：不改期望，只重新选择零点

### 4.1 无偏性的条件证明

在状态 $s$，从权重中减去不依赖当前动作的 $b(s)$：

$$
(G_t-b(S_t))
\nabla_\theta\log\pi_\theta(A_t\mid S_t).
$$

条件于状态：

$$
\begin{aligned}
&\mathbb E_{A\sim\pi_\theta}
\left[b(s)\nabla_\theta\log\pi_\theta(A\mid s)\right]\\
&=b(s)\sum_a\pi_\theta(a\mid s)
\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}\\
&=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1=0.
\end{aligned}
$$

关键条件不是“baseline 必须准确”，而是它不能看当前采样动作。即使 $b(s)$ 数值有偏，只要在 Actor 梯度中把它视为常量，score 项的期望仍为零；但差得太远可能增大方差。

如果 $b$ 与 Actor 共享参数，surrogate loss 中必须对 advantage 做 `stop_gradient`。否则自动微分还会沿 $b_\theta$ 求导，多出不属于策略梯度定理的项。

::: ml-stop 停下来推一推 2
把 $b(s)$ 改成 $b(s,a)$，指出上面从第二行到第三行为什么不再成立。再考虑一个共享主干网络：若没有 detach advantage，梯度会多走哪条路径？
:::

<MlQuiz
  kind="方差诊断"
  question="下面哪个 baseline 可以直接用于上述无偏证明？"
  :options="['只依赖当前状态和历史、不依赖当前采样动作的值', '读取当前动作后专门给该动作一个偏移的值', '用未来真实动作标签构造的值', '任意对 actor loss 可导的动作函数']"
  :answer="0"
  explanation="证明依赖于 b 能移出对当前动作的求和；动作相关项一般不能任意相减。"
/>

### 4.2 为什么常用 $V^\pi(s)$，但它不总是最优标量 baseline

取 $b(s)=V^\pi(s)$ 时，

$$
G_t-V^\pi(S_t)
$$

是 advantage $A^\pi(S_t,A_t)=Q^\pi(S_t,A_t)-V^\pi(S_t)$ 的单样本估计。它把“绝对回报高不高”改成“比这个状态下的平均行为好多少”。

若目标是最小化单状态 score 向量的平方范数方差，最优标量其实是

$$
b^*(s)=
\frac{
\mathbb E[G_t\|\nabla_\theta\log\pi(A_t\mid s)\|^2\mid s]
}{
\mathbb E[\|\nabla_\theta\log\pi(A_t\mid s)\|^2\mid s]
}.
$$

$V^\pi(s)=\mathbb E[G_t\mid s]$ 是更容易学习、通常很有效的近似，但两者只有在 score 范数与回报权重关系合适时相同。

<MlQuiz
  kind="推导审计"
  question="价值 baseline 预测不准时，哪句话最准确？"
  :options="['只要它不依赖当前动作且 Actor 对其 detach，策略 score 期望仍无偏，但方差可能变好也可能变坏', '策略梯度必定变成零', 'Critic 永远不产生偏差', '可以无限复用旧数据']"
  :answer="0"
  explanation="baseline 的无偏性与数值准确性是两件事；bootstrap Critic 的偏差又是后续另一件事。"
/>

### 4.3 三种估计器的方差不是一句口号

对同一个时刻，比较：

1. 整条轨迹回报乘 score；
2. 删除过去奖励后的 reward-to-go；
3. 再减状态价值后的 advantage。

它们在正确条件下有相同的期望方向，但有限样本的散布不同。方差降低不是“每一批都更接近”；必须在固定策略、固定预算、多个独立批次上报告均值与分布。

<BaselineVarianceStripChartLab />

<MlQuiz
  kind="实验判读"
  question="实验中把 Critic 偏移推得很远后，优势条带方差反而升高，是否推翻 baseline 无偏证明？"
  :options="['不推翻；无偏只约束长期中心，不保证任意 baseline 都降低方差', '推翻，因为无偏等于零方差', '推翻，因为 Critic 必须等于奖励', '不推翻，因为策略不需要概率']"
  :answer="0"
  explanation="baseline 是 control variate；系数或零点选坏时仍可保持期望，却放大波动。"
/>

## 5. 用隐藏模型先验收解析策略梯度

### 5.1 表格 Softmax Actor 与精确 oracle

对教学走廊，令每个非终止状态有四个 logits。解析 oracle 用隐藏模型计算：

$$
V^\pi=(I-\gamma P_\pi)^{-1}r_\pi,
$$

$$
d_\gamma^\pi
=\rho_0^\top(I-\gamma P_\pi)^{-1},
$$

以及 Softmax logit 梯度

$$
\frac{\partial J}{\partial z(s,a)}
=d_\gamma^\pi(s)\pi(a\mid s)
[Q^\pi(s,a)-V^\pi(s)].
$$

```python
def softmax_rows(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)

def exact_policy_quantities(logits):
    """只用于验收；训练函数不得调用。"""
    policy = softmax_rows(logits)
    expected_reward = np.sum(HIDDEN_P * HIDDEN_R, axis=2)
    policy_transition = np.einsum("sa,san->sn", policy, HIDDEN_P)
    policy_reward = np.sum(policy * expected_reward, axis=1)
    transient = policy_transition[np.ix_(NONTERMINALS, NONTERMINALS)]

    value = np.zeros(len(CELLS))
    value[NONTERMINALS] = np.linalg.solve(
        np.eye(len(NONTERMINALS)) - GAMMA * transient,
        policy_reward[NONTERMINALS],
    )
    action_value = np.sum(
        HIDDEN_P * (HIDDEN_R + GAMMA * value[None, None, :]),
        axis=2,
    )
    action_value[list(TERMINALS)] = 0.0

    start_distribution = np.zeros(len(NONTERMINALS))
    start_position = np.where(NONTERMINALS == START)[0][0]
    start_distribution[start_position] = 1.0
    occupancy = np.zeros(len(CELLS))
    occupancy[NONTERMINALS] = start_distribution @ np.linalg.inv(
        np.eye(len(NONTERMINALS)) - GAMMA * transient
    )
    gradient = (
        occupancy[:, None]
        * policy
        * (action_value - value[:, None])
    )
    gradient[list(TERMINALS)] = 0.0
    return value, action_value, policy, occupancy, gradient

rng = np.random.default_rng(3201)
THETA0 = rng.normal(0, 0.22, size=(len(CELLS), len(ACTIONS)))
THETA0[list(TERMINALS)] = 0.0
V0, Q0, PI0, D0, ANALYTIC_GRADIENT = exact_policy_quantities(THETA0)

assert np.allclose(PI0.sum(axis=1), 1.0)
assert np.max(np.abs(ANALYTIC_GRADIENT.sum(axis=1))) < 1e-10
```

### 5.2 中心差分检查整条计算链

```python
NUMERIC_GRADIENT = np.zeros_like(THETA0)
epsilon = 1e-5
for state in NONTERMINALS:
    for action in range(len(ACTIONS)):
        plus, minus = THETA0.copy(), THETA0.copy()
        plus[state, action] += epsilon
        minus[state, action] -= epsilon
        NUMERIC_GRADIENT[state, action] = (
            exact_policy_quantities(plus)[0][START]
            - exact_policy_quantities(minus)[0][START]
        ) / (2 * epsilon)

PG_FD_ERROR = float(np.max(np.abs(
    ANALYTIC_GRADIENT - NUMERIC_GRADIENT
)))
print("start_value", round(float(V0[START]), 6))
print("policy_gradient_fd_max", PG_FD_ERROR)
assert PG_FD_ERROR < 2e-8
```

这个检查同时覆盖 Softmax score、折扣访问量、$Q-V$ 和终止状态处理。只检查 loss 能下降，无法定位 $gamma^t$、状态访问权或 logit Jacobian 漏项。

## 6. Monte Carlo 证据：同方向、不同方差

### 6.1 三个轨迹估计器

```python
def trajectory_gradient(rng, logits, mode="baseline", max_steps=160):
    value, _, policy, _, _ = exact_policy_quantities(logits)
    states, actions, rewards = [], [], []
    state = START
    for _ in range(max_steps):
        action = int(rng.choice(len(ACTIONS), p=policy[state]))
        next_state, reward, terminated = env_step(rng, state, action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        if terminated:
            break

    returns = np.zeros(len(rewards))
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + GAMMA * running
        returns[t] = running

    gradient = np.zeros_like(logits)
    total_return = returns[0]
    for t, (state, action) in enumerate(zip(states, actions)):
        score = -policy[state].copy()
        score[action] += 1.0
        weight = total_return if mode == "full" else returns[t]
        if mode == "baseline":
            weight -= value[state]
        discount = 1.0 if mode == "full" else GAMMA**t
        gradient[state] += discount * weight * score
    return gradient

def draw_gradient_batch(mode, count=5000):
    draw_rng = np.random.default_rng(3202)
    return np.stack([
        trajectory_gradient(draw_rng, THETA0, mode)
        for _ in range(count)
    ])

GRADIENT_SAMPLES = {
    mode: draw_gradient_batch(mode)
    for mode in ("full", "reward_to_go", "baseline")
}

GRADIENT_REPORT = {}
for mode, samples in GRADIENT_SAMPLES.items():
    mean = samples.mean(axis=0)
    centered = samples - mean
    total_variance = np.mean(np.sum(centered**2, axis=(1, 2)))
    cosine = np.vdot(mean, ANALYTIC_GRADIENT) / (
        np.linalg.norm(mean) * np.linalg.norm(ANALYTIC_GRADIENT)
    )
    GRADIENT_REPORT[mode] = (float(total_variance), float(cosine))
    print(mode, "variance/cosine", np.round(GRADIENT_REPORT[mode], 4))

assert GRADIENT_REPORT["baseline"][0] < GRADIENT_REPORT["reward_to_go"][0]
assert GRADIENT_REPORT["reward_to_go"][0] < GRADIENT_REPORT["full"][0]
assert GRADIENT_REPORT["baseline"][1] > 0.98
```

固定结果约为：

| 估计器 | 梯度总方差 | 与解析梯度余弦 |
|---|---:|---:|
| 整轨迹回报 | 约 2081 | 约 0.97 |
| reward-to-go | 约 607 | 约 0.99 |
| 精确价值 baseline | 约 184 | 约 1.00 |

这不是声称任意环境、任意 baseline 都有同样倍率，而是为本章案例建立一份可重放资格证据。严格的均值误差还应随样本数画置信区间；余弦只检查方向，不检查尺度。

<MlQuiz
  kind="验收证据"
  question="为什么“采样梯度和解析梯度余弦很高”仍不能单独证明实现完全正确？"
  :options="['余弦忽略尺度，有限样本也可能偶然对齐，还需有限差分、均值误差与边界断言', '因为余弦只能用于图像', '因为解析梯度没有方向', '因为方差必须为零']"
  :answer="0"
  explanation="方向、尺度、方差和边界语义是不同证据；本章用中心差分和解析 oracle 补齐。"
/>

## 7. REINFORCE：批量收集，再更新当前策略

### 7.1 算法主线

带学习 baseline 的批量 REINFORCE：

1. 冻结策略版本 $	heta_k$；
2. 用 $pi_{\theta_k}$ 收集一批完整 rollout；
3. 反向计算每个时刻的 $G_t$；
4. 用 $G_t-b_\phi(S_t)$ 加权 log-prob；
5. 更新 Actor 一次或少数几次；
6. 用回报回归 baseline；
7. 丢弃或明确校正已离策略的数据，再进入下一轮。

Actor 的最小化 loss 常写成

$$
L_\pi(\theta)
=-\frac1B\sum_t
\gamma^t\operatorname{stopgrad}(\hat A_t)
\log\pi_\theta(a_t\mid s_t).
$$

这里的负号把梯度上升转换为常见优化器的梯度下降。

### 7.2 从样本接口训练，不偷看模型

```python
def sample_episode(rng, logits, max_steps=120):
    policy = softmax_rows(logits)
    states, actions, rewards = [], [], []
    state = START
    for _ in range(max_steps):
        action = int(rng.choice(len(ACTIONS), p=policy[state]))
        next_state, reward, terminated = env_step(rng, state, action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        if terminated:
            break
    return states, actions, rewards

def train_reinforce(seed, iterations=260, batch_size=32):
    rng = np.random.default_rng(seed)
    logits = np.zeros((len(CELLS), len(ACTIONS)))
    baseline = np.zeros(len(CELLS))
    for _ in range(iterations):
        policy = softmax_rows(logits)  # 冻结本批行为策略
        gradient = np.zeros_like(logits)
        value_sum = np.zeros(len(CELLS))
        value_count = np.zeros(len(CELLS))
        transition_count = 0
        for _ in range(batch_size):
            states, actions, rewards = sample_episode(rng, logits)
            running = 0.0
            for t in range(len(rewards) - 1, -1, -1):
                running = rewards[t] + GAMMA * running
                state, action = states[t], actions[t]
                score = -policy[state].copy()
                score[action] += 1.0
                advantage = running - baseline[state]
                gradient[state] += GAMMA**t * advantage * score
                value_sum[state] += running
                value_count[state] += 1
                transition_count += 1
        logits += 0.75 * gradient / max(transition_count, 1)
        seen = value_count > 0
        baseline[seen] = (
            0.85 * baseline[seen]
            + 0.15 * value_sum[seen] / value_count[seen]
        )
    return logits, baseline

REINFORCE_VALUES = []
for seed in range(5):
    trained_logits, learned_baseline = train_reinforce(4100 + seed)
    exact_value = exact_policy_quantities(trained_logits)[0][START]
    REINFORCE_VALUES.append(float(exact_value))

print("reinforce_start_values", np.round(REINFORCE_VALUES, 3))
assert np.mean(REINFORCE_VALUES) > 6.0
```

5 个固定种子的冻结策略起点价值约为 `[6.196, 6.183, 6.203, 6.225, 6.206]`。训练函数没有接收 $P$ 或 $R$；只有最后的 `exact_policy_quantities` 用隐藏模型做资格审查。

<MlQuiz
  kind="实现边界"
  question="为什么不能在每收集一条 transition 后立刻大幅更新 Actor，却仍把回合剩余样本当作同一 on-policy batch？"
  :options="['因为后续样本来自变化后的策略，batch 已混合多个行为分布', '因为 Actor 一次只能处理一条轨迹', '因为 reward-to-go 不允许批量', '因为 Softmax 不可微']"
  :answer="0"
  explanation="on-policy 不是标签；它要求数据生成分布与更新公式假设一致。在线小步更新需采用相应算法和版本控制。"
/>

### 7.3 动作 mask 必须在分布内部

对非法动作集合 $\mathcal I(s)$，应构造

$$
\tilde z_a=
\begin{cases}
z_a,&a\notin\mathcal I(s),\\
-\infty,&a\in\mathcal I(s),
\end{cases}
$$

再计算 Softmax。采样、`log_prob`、熵、PPO ratio 和评价都必须使用同一个 masked distribution。先从完整分布采样、再把非法动作替换为安全动作，会让日志动作与 log-prob 不匹配，梯度有系统偏差。

硬 mask 也不能由 Actor 任意学习开关；其版本和来源属于安全合同。若某状态没有合法动作，应显式拒绝并让环境进入受控终止，而不是对全 `-inf` logits 继续 Softmax。

<MlQuiz
  kind="安全合同"
  question="为什么“先采样非法动作，再由安全层替换成合法动作”不能直接拿替换后的动作训练？"
  :options="['替换后的动作不是记录 log-prob 对应的随机变量，score 与行为分布失配', '因为安全层不能记录日志', '因为合法动作没有概率', '因为 reward 必须为正']"
  :answer="0"
  explanation="梯度需要实际生成动作的分布证据；动作被事后改写后，必须重新定义执行策略和记录其真实概率。"
/>

## 8. 从 baseline 到 Actor–Critic

### 8.1 Monte Carlo Critic 与 bootstrap Critic 不是同一件事

若只用 Monte Carlo 回报训练 $V_\phi(s)$，再把它作为 baseline，Actor 的 score 估计仍可无偏；Critic 主要降低方差。一步 Actor–Critic 则把 future return 替换为 bootstrap：

$$
\delta_t
=R_{t+1}
+\gamma(1-d_{t+1})V_\phi(S_{t+1})
-V_\phi(S_t).
$$

用真实 $V^\pi$ 时，$\mathbb E[\delta_t\mid S_t,A_t]=A^\pi(S_t,A_t)$。用近似 $V_\phi$ 时，target 更低方差，但建模误差会把 Actor 梯度推偏。这就是 Actor–Critic 的偏差—方差交换。

Actor 更新：

$$
\theta\leftarrow\theta
+\alpha_\pi\delta_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t).
$$

Critic 做 semi-gradient TD：

$$
y_t=R_{t+1}+\gamma(1-d_{t+1})
\operatorname{stopgrad}(V_\phi(S_{t+1})),
$$

$$
L_V(\phi)=\frac12[y_t-V_\phi(S_t)]^2.
$$

target 分支必须 detach；否则在随机转移下，最小化的是另一种 residual-gradient 目标，不再是这里的 TD semi-gradient。

<MlQuiz
  kind="偏差方差"
  question="哪项最准确地区分“价值 baseline”和“一步 bootstrap Critic”？"
  :options="['前者从 MC 权重减动作无关零点可保持 score 无偏；后者用近似 V(s′) 替代未来回报，可能引入偏差', '两者永远完全相同', 'baseline 必须依赖动作', 'bootstrap 一定没有方差']"
  :answer="0"
  explanation="Lecture 6 与 Section 3 都强调：近似 Critic 降方差的代价是模型误差带来的偏差。"
/>

### 8.2 神经 Actor 与 Critic 的形状合同

在表格之外，可把 AMR 状态编码成特征向量

$$
x(s)=[r/3,c/3,\text{goal distance},
\text{neighbor occupancy},\text{task phase},\ldots].
$$

一种最小双头网络为

$$
h=\operatorname{ReLU}(xW_h+b_h),
$$

$$
z=hW_\pi+b_\pi\in\mathbb R^{K},
\qquad
V=hW_V+b_V\in\mathbb R.
$$

```python
def masked_softmax(logits, legal_mask):
    logits = np.asarray(logits, dtype=float)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if logits.shape != legal_mask.shape or not legal_mask.any(axis=1).all():
        raise ValueError("每个样本必须至少有一个合法动作")
    masked = np.where(legal_mask, logits, -np.inf)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    weights = np.where(legal_mask, np.exp(shifted), 0.0)
    return weights / weights.sum(axis=1, keepdims=True)

def actor_critic_forward(features, parameters, legal_mask):
    hidden = np.maximum(
        features @ parameters["hidden_weight"] + parameters["hidden_bias"],
        0.0,
    )
    logits = hidden @ parameters["actor_weight"] + parameters["actor_bias"]
    values = hidden @ parameters["critic_weight"] + parameters["critic_bias"]
    return masked_softmax(logits, legal_mask), values[:, 0]

shape_rng = np.random.default_rng(3203)
PARAMETERS = {
    "hidden_weight": shape_rng.normal(size=(6, 12)) / np.sqrt(6),
    "hidden_bias": np.zeros(12),
    "actor_weight": shape_rng.normal(size=(12, 4)) / np.sqrt(12),
    "actor_bias": np.zeros(4),
    "critic_weight": shape_rng.normal(size=(12, 1)) / np.sqrt(12),
    "critic_bias": np.zeros(1),
}
features = shape_rng.normal(size=(8, 6))
legal_mask = np.ones((8, 4), dtype=bool)
legal_mask[0, 3] = False
policy_batch, value_batch = actor_critic_forward(
    features, PARAMETERS, legal_mask
)

assert policy_batch.shape == (8, 4)
assert value_batch.shape == (8,)
assert policy_batch[0, 3] == 0.0
assert np.allclose(policy_batch.sum(axis=1), 1.0)
```

共享 trunk 节省计算并可能共享有用表示，但 Actor 与 Critic 的梯度会在同一参数上竞争。工程上至少分别记录 actor loss、value loss、entropy、explained variance、两头梯度范数和 trunk 梯度夹角；出现 Critic loss 下降而策略退化时，不要只继续增大 value-loss 系数。

<MlQuiz
  kind="网络接口"
  question="共享 trunk 的 Actor–Critic 中，为什么仅看总 loss 下降不够？"
  :options="['Actor、Critic 和熵项尺度不同、梯度可能冲突，总和下降不能证明策略回报改善', '因为共享网络不能反向传播', '因为 Critic 不需要数据', '因为 Softmax 输出标量']"
  :answer="0"
  explanation="多目标共享参数会隐藏头间竞争；必须分项监控并用独立策略评价确认。"
/>

### 8.3 在线一步 Actor–Critic

```python
def train_actor_critic(seed, episodes=3500):
    rng = np.random.default_rng(seed)
    logits = np.zeros((len(CELLS), len(ACTIONS)))
    values = np.zeros(len(CELLS))
    for episode in range(episodes):
        state = START
        actor_rate = 0.055 / (1 + episode / 1800)
        critic_rate = 0.18 / (1 + episode / 2500)
        for _ in range(120):
            policy = softmax_rows(logits)
            action = int(rng.choice(len(ACTIONS), p=policy[state]))
            next_state, reward, terminated = env_step(rng, state, action)
            target = reward if terminated else reward + GAMMA * values[next_state]
            td_error = target - values[state]

            values[state] += critic_rate * td_error
            score = -policy[state].copy()
            score[action] += 1.0
            logits[state] += actor_rate * td_error * score

            if terminated:
                break
            state = next_state
    return logits, values

ACTOR_CRITIC_VALUES = []
for seed in range(5):
    trained_logits, learned_values = train_actor_critic(5200 + seed)
    exact_value = exact_policy_quantities(trained_logits)[0][START]
    ACTOR_CRITIC_VALUES.append(float(exact_value))

UNIFORM_VALUE = float(exact_policy_quantities(
    np.zeros_like(THETA0)
)[0][START])
print("uniform_start_value", round(UNIFORM_VALUE, 3))
print("actor_critic_start_values", np.round(ACTOR_CRITIC_VALUES, 3))
assert np.mean(ACTOR_CRITIC_VALUES) > 6.4
assert np.mean(ACTOR_CRITIC_VALUES) > np.mean(REINFORCE_VALUES)
```

固定结果中，均匀策略起点价值约为 $-8.308$，5 个 Actor–Critic 种子约为 `[6.508, 6.513, 6.515, 6.511, 6.513]`。这不证明 Actor–Critic 在所有任务都优于 REINFORCE；两者样本预算、更新频率和步长不同。这里只证明实现能在冻结教学环境中稳定学出高价值策略。

<ActorCriticSignalRoomLab />

## 9. n-step 与 GAE：在 Monte Carlo 和一步 TD 之间接线

### 9.1 n-step return

$n$ 步 target 为

$$
\hat G_t^{(n)}
=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}
+\gamma^nV_\phi(S_{t+n}).
$$

$n=1$ 依赖 bootstrap，方差低但对 Critic 误差敏感；走到自然终止的完整回报不 bootstrap，方差高但不含终点后的价值模型误差。$n$ 不是越大越好，且 rollout 长度不等于任务自然终止长度。

### 9.2 GAE 是 TD residual 的指数滤波

定义

$$
\delta_t
=R_{t+1}+\gamma V_\phi(S_{t+1})-V_\phi(S_t).
$$

广义优势估计为

$$
\hat A_t^{\operatorname{GAE}(\gamma,\lambda)}
=\sum_{\ell=0}^{T-t-1}
(\gamma\lambda)^\ell\delta_{t+\ell}.
$$

倒序递推只需 $O(T)$：

$$
\hat A_t=\delta_t+\gamma\lambda c_{t+1}\hat A_{t+1}.
$$

$\lambda=0$ 接近一步 TD；$\lambda\to1$ 接近长回报 advantage。近似 Critic 下，增大 $\lambda$ 通常降低 bootstrap 偏差、提高采样方差，但边界还受时域、折扣和截断处理影响。

### 9.3 terminated 与 truncated 要用两个门

自然终止：下一状态没有任务内未来回报，删除 bootstrap，也停止 GAE carry。

时间截断：任务本身未结束，最终观测仍可 bootstrap；但环境重置后不能把下一回合的 advantage carry 接回来。因此需要两个 mask：

$$
m_t^{\text{boot}}=1-\texttt{terminated}_t,
$$

$$
m_t^{\text{carry}}
=1-(\texttt{terminated}_t\lor\texttt{truncated}_t).
$$

```python
def generalized_advantage(
    rewards,
    values,
    next_values,
    terminated,
    truncated,
    gamma=0.94,
    lam=0.90,
):
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    next_values = np.asarray(next_values, dtype=float)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    bootstrap = (~terminated).astype(float)
    continuation = (~(terminated | truncated)).astype(float)
    deltas = rewards + gamma * bootstrap * next_values - values
    advantage = np.zeros_like(rewards)
    carry = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        carry = deltas[t] + gamma * lam * continuation[t] * carry
        advantage[t] = carry
    return advantage, advantage + values

GAE_ADVANTAGE, GAE_TARGET = generalized_advantage(
    rewards=[-0.45, -0.45, 14.0],
    values=[3.0, 4.2, 8.0],
    next_values=[4.2, 8.0, 0.0],
    terminated=[False, False, True],
    truncated=[False, False, False],
)
TRUNCATED_ADVANTAGE, _ = generalized_advantage(
    rewards=[-0.45],
    values=[3.0],
    next_values=[4.2],
    terminated=[False],
    truncated=[True],
)

print("gae_advantage", np.round(GAE_ADVANTAGE, 6))
assert np.isclose(GAE_ADVANTAGE[-1], 14.0 - 8.0)
assert np.isclose(
    TRUNCATED_ADVANTAGE[0], -0.45 + GAMMA * 4.2 - 3.0
)
```

::: ml-stop 停下来推一推 3
同样在时间上限结束：若这是“电量耗尽”的任务自然终止，target 是什么？若只是训练器每 64 步切一段、但返回了 `final_observation`，target 又是什么？分别指出 bootstrap mask 与 carry mask。
:::

<MlQuiz
  kind="边界语义"
  question="时间截断时为什么通常保留 final_observation 的 bootstrap，却停止 GAE carry？"
  :options="['任务未来价值仍存在，但 reset 后的新回合不是原轨迹的连续未来', '因为截断等同安全事故', '因为 Critic 必须输出 0', '因为 λ 只能用于终止状态']"
  :answer="0"
  explanation="bootstrap 回答任务是否结束；carry 回答当前存储段是否还能沿同一轨迹递推，是两个不同问题。"
/>

<MlQuiz
  kind="偏差方差"
  question="在近似 Critic 下，把 GAE λ 从 0 调到接近 1，通常发生什么？"
  :options="['更依赖长程样本、bootstrap 偏差通常下降而采样方差上升', '偏差和方差都必为 0', '不再需要 reward', '自动变成 off-policy']"
  :answer="0"
  explanation="GAE 是多种 n-step advantage 的指数混合；λ 控制短 bootstrap 与长回报的交换。"
/>

### 9.4 advantage 标准化不是无害代数恒等式

实践常做

$$
\tilde A_t
=\frac{\hat A_t-\bar A}{s_A+\epsilon}.
$$

除以标准差主要改变有效学习率；减去 batch 均值像一个批级 baseline。但均值和标准差由当前有限批的所有动作共同估计，会引入样本间耦合，不等同于前面逐状态、条件期望下的精确证明。它通常有用，却应作为明确的优化启发式记录，而不是宣称“理论上完全相同”。小 batch、重尾回报或多任务混批时尤其要审计。

## 10. 熵、连续动作与策略分布的工程细节

### 10.1 熵正则的目标要单独写

离散策略熵：

$$
\mathcal H(\pi(\cdot\mid s))
=-\sum_a\pi(a\mid s)\log\pi(a\mid s).
$$

常见目标：

$$
J_{\text{total}}
=J_{\text{return}}
+\beta\mathbb E[\mathcal H(\pi(\cdot\mid S))].
$$

$\beta$ 过小可能过早塌缩，过大则长期随机。熵梯度是目标中的独立项；把 `beta * entropy` 随意塞进每步环境 reward 会改变 Critic 学的量，并可能与评价回报混淆。训练日志应分别记录 extrinsic return、entropy bonus 和 total surrogate。

### 10.2 高斯 Actor 不是“把离散 Softmax 换个名字”

连续动作常令网络输出 $\mu_\theta(s)$ 与 $\log\sigma_\theta(s)$：

$$
A=\mu_\theta(S)+\sigma_\theta(S)\odot\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

REINFORCE 可以用高斯 `log_prob` 的 score-function 梯度。若动作需限制在 $[-1,1]$，常取 $u\sim\mathcal N$、$a=\tanh u$；此时执行动作的 log-density 必须包含变量变换的 log-Jacobian，并对执行器再做物理量映射。直接 `clip(action)` 会在边界堆积概率质量，记录的 Gaussian log-prob 不再对应实际执行分布。

标准差应有数值上下界，且探索尺度不能越过真实安全层。对高风险控制，不应把神经策略的 $sigma$ 当作安全裕量。

<MlQuiz
  kind="连续策略"
  question="把 Gaussian 样本 clip 到动作边界后，为什么不能继续使用未截断 Gaussian 的 log_prob？"
  :options="['实际执行分布已被裁剪改写，边界出现概率质量，似然与动作不再对应', '因为 Gaussian 没有均值', '因为连续动作不能做策略梯度', '因为熵恒为零']"
  :answer="0"
  explanation="score 必须来自真实采样/执行分布；tanh 变换也要计入 Jacobian。"
/>

## 11. PPO：限制旧数据上的代理收益，不是安全证明

### 11.1 新旧策略概率比

PPO 用旧策略 $\pi_{\text{old}}$ 收集 rollout，计算

$$
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\text{old}}(a_t\mid s_t)}
=\exp(\log\pi_\theta-\log\pi_{\text{old}}).
$$

裁剪代理目标为

$$
L^{\text{clip}}(\theta)
=\mathbb E\left[
\min\left(
r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

符号决定哪一侧受限：

- $\hat A_t>0$：把好动作概率提高到 $1+\epsilon$ 以上，不再得到额外代理收益；
- $\hat A_t<0$：把坏动作概率降到 $1-\epsilon$ 以下，不再得到额外代理收益；
- 反方向移动仍会让目标继续变差，因此不是简单把 ratio 永久截到区间内。

```python
def ppo_clipped_term(ratio, advantage, clip_epsilon=0.2):
    ratio = np.asarray(ratio, dtype=float)
    unclipped = ratio * advantage
    clipped = np.clip(
        ratio, 1 - clip_epsilon, 1 + clip_epsilon
    ) * advantage
    return np.minimum(unclipped, clipped)

RATIOS = np.array([0.6, 0.9, 1.1, 1.5])
POSITIVE_TERMS = ppo_clipped_term(RATIOS, advantage=2.0)
NEGATIVE_TERMS = ppo_clipped_term(RATIOS, advantage=-2.0)

print("positive_advantage", POSITIVE_TERMS)
print("negative_advantage", NEGATIVE_TERMS)
assert np.allclose(POSITIVE_TERMS, [1.2, 1.8, 2.2, 2.4])
assert np.allclose(NEGATIVE_TERMS, [-1.6, -1.8, -2.2, -3.0])
```

<MlQuiz
  kind="PPO 判读"
  question="当 advantage 为负、ratio 从 1 降到 0.6、epsilon=0.2 时，PPO clip 为什么使用 0.8×advantage 而不是 0.6×advantage？"
  :options="['它不再奖励继续压低坏动作概率，负 advantage 下 min 会选择更负的裁剪项', '因为 ratio 不能小于 1', '因为 advantage 会自动变正', '因为旧策略概率为 0']"
  :answer="0"
  explanation="负数会反转大小关系；PPO 的 min 必须结合 advantage 符号理解。"
/>

### 11.2 PPO 仍需要的护栏

裁剪不是 trust-region 的硬保证。训练至少监控：

- approximate KL 与目标 KL；
- clip fraction；
- 新旧策略 entropy；
- importance ratio 的分位数与非有限值；
- value loss、explained variance 与 value clipping；
- 每个 epoch 的独立评价回报；
- 同一 rollout 被重复训练的 epoch 数。

若 KL 超限，应停止当前 batch 的更新，而不是认为 ratio 已在 loss 中裁剪就万事大吉。旧 log-prob、action mask、观测归一化器和 recurrent hidden state 必须与采集版本一致。

### 11.3 SAC 只作边界定位

Soft Actor-Critic 是 off-policy 最大熵 Actor–Critic：使用 replay buffer、学习 $Q$、连续动作重参数化、目标网络和温度系数。它不是“REINFORCE 加一个 entropy bonus”，也不能把本章 on-policy rollout 无限丢进 replay 就自然得到 SAC。要实现 SAC，必须重新建立 off-policy Bellman target、双 Q、行为覆盖和数据新鲜度合同；本章不展开其完整推导。

## 12. 完整训练损失与更新次序

以 on-policy GAE/PPO 为例，一个 batch 常有：

$$
L_{\text{actor}}
=-\mathbb E[L^{\text{clip}}],
$$

$$
L_{\text{critic}}
=\frac12\mathbb E[(\hat V_t-V_\phi(S_t))^2],
$$

$$
L_{\text{entropy}}
=-\mathbb E[\mathcal H(\pi_\theta(\cdot\mid S_t))],
$$

$$
L=L_{\text{actor}}+c_vL_{\text{critic}}+c_eL_{\text{entropy}}.
$$

工业实现的次序合同：

1. 冻结 `policy_old`、观测归一化器和 action-mask 版本；
2. 收集完整 rollout，并保存 old log-prob、old value 与 final observation；
3. 用冻结 old value 或明确的 current value 计算 GAE；
4. 对 advantage detach，构造 actor loss；
5. 对 value target detach，构造 critic loss；
6. 分项检查非有限值、梯度范数和 KL；
7. 按 minibatch/epoch 更新，超 KL 提前停止；
8. 独立 evaluation policy 不加训练探索噪声；
9. 冻结候选策略后才与基线做多种子资格比较。

<MlQuiz
  kind="训练管线"
  question="为什么 old_log_prob 与 old_value 应在采集时或更新前冻结？"
  :options="['否则 ratio 和 GAE 的参照会随优化移动，代理目标失去固定基准', '因为它们不能存浮点数', '因为 Critic 不允许更新', '因为 PPO 只训练一次']"
  :answer="0"
  explanation="PPO 用固定旧策略定义重要性比；target 也必须有清楚的版本边界。"
/>

## 数值验证

本章的自动执行建立了四层证据：

| 层级 | 验证对象 | 固定结果 / 判据 |
|---|---|---|
| 解析层 | 折扣访问量、$Q-V$、Softmax logit 梯度 | 与中心差分最大误差约 $3.7\times10^{-10}$ |
| 采样层 | 整轨迹、reward-to-go、baseline 三种估计器 | 方差约 `2081 / 607 / 184`，baseline 均值方向余弦 $>0.98$ |
| 学习层 | REINFORCE 五种子 | 冻结起点价值均值约 `6.20` |
| 学习层 | 一步 Actor–Critic 五种子 | 冻结起点价值均值约 `6.51`，均匀策略约 `−8.31` |
| 边界层 | 终止、截断与 GAE | 自然终止去 bootstrap；截断保留 final-value bootstrap、阻断 carry |
| 代理层 | PPO 正负 advantage | ratio 四点的符号相关裁剪断言通过 |

### 1. 还应加入的生产级回归测试

- **概率合同**：合法概率和为 1，非法动作概率精确为 0，全非法行被拒绝；
- **似然合同**：采样时 log-prob 与离线重算一致，策略版本不混批；
- **梯度合同**：同一 batch 的 autograd 与手推 score 梯度一致；
- **边界合同**：自然终止、时间截断、collector 切片和异常终止分别测试；
- **不变量**：共同平移 logits 不改变概率，动作重排同时重排概率和梯度；
- **数值合同**：极端 logits、极小标准差、全零 advantage、长时域均无 NaN/Inf；
- **恢复合同**：中断后恢复的优化器、归一化器、RNG、policy version 连续；
- **评价合同**：训练采样与冻结评价隔离，评价种子和预算独立。

### 2. 多种子报告不能只给最终均值

至少报告：

- x 轴为真实环境步数，不只写 optimizer iteration；
- 每个种子的学习曲线和聚合置信带；
- 最终回报、样本效率、成功率、隔离率和最长回合；
- 最佳 checkpoint 的选择规则与验证预算；
- 随机、启发式、动态规划 oracle（若合成环境可得）、Q-learning 基线；
- 训练期间的 entropy、KL、clip fraction、value explained variance；
- 失败种子，不删除“难看曲线”；
- 硬约束违反必须始终为 0，不能与回报加权平均。

<MlQuiz
  kind="实验设计"
  question="为什么挑选回报最高的单个种子作为最终证据不合格？"
  :options="['它混入随机选择偏差，隐藏算法方差和失败概率', '因为强化学习不能设种子', '因为均值必须等于 0', '因为训练曲线不能画图']"
  :answer="0"
  explanation="高方差算法必须预先固定种子集合、预算和聚合规则，完整呈现失败分布。"
/>

## 复杂度与工程边界

### 1. 时间、内存和通信预算

设 rollout 环境步数为 $B$，Actor/Critic 前向代价为 $C_f$，反向代价为 $C_b$，PPO 做 $E$ 个 epoch、minibatch 大小 $M$：

| 环节 | 时间量级 | 主要内存 | 工程瓶颈 |
|---|---:|---:|---|
| 环境采集 | $O(BC_f)$ 加环境成本 | $O(Bd)$ | 仿真吞吐、同步等待、策略版本陈旧 |
| 回报 / GAE | $O(B)$ | $O(B)$ | 边界 mask、final observation、长回合 |
| REINFORCE 更新 | $O(BC_b)$ | 激活与 rollout | 高方差要求大 batch |
| 一步 Actor–Critic | 每步 $O(C_f+C_b)$ | 在线状态或短 rollout | 非平稳 Critic、更新耦合 |
| PPO 多 epoch | $O(EBC_b)$ | rollout + minibatch 激活 | 旧数据过度复用、KL 漂移 |
| 离散全动作头 | 输出 $O(K)$ | logits $O(BK)$ | 动作数大时 Softmax 成本 |
| Gaussian 对角头 | 输出 $O(d_a)$ | 均值/方差 $O(Bd_a)$ | 边界变换与协方差表达能力 |

在分布式采集中，`policy lag` 是算法变量：collector 使用的参数落后 learner 越多，数据越不像当前 on-policy。不能只靠扩大机器数假定线性加速；要记录采集版本分布、丢弃阈值和 learner/collector 吞吐比。

### 2. 典型失效模式与定位顺序

| 症状 | 首查证据 | 常见根因 | 不应使用的绕过 |
|---|---|---|---|
| 回报不升、entropy 很快归零 | logits、最小动作概率、熵系数 | 初始化过尖、步长过大、稀疏回报 | 无上限增大随机噪声 |
| value loss 很低但策略退化 | explained variance、独立回报、共享梯度 | Critic 过拟合或主导共享 trunk | 只继续降总 loss |
| PPO clip fraction 长期很高 | KL、ratio 分位数、epoch 数 | 步长大、旧 batch 重用太多 | 认为 clip 自动保证安全 |
| GAE 在回合边界跳变 | terminated/truncated/final observation | 两类边界混为一个 done | 全部 target 置零 |
| 动作日志与 log-prob 对不上 | action mask 与 policy SHA | 事后动作替换、归一化版本漂移 | 用新策略重算旧 log-prob |
| 某些种子完全失败 | 初期成功轨迹、梯度 SNR | 稀疏回报、概率塌缩、环境随机性 | 删除失败种子 |
| 连续动作卡边界 | pre-tanh 样本、Jacobian、执行分布 | 直接 clip、标准差失控 | 只改显示值 |
| 恢复训练后突变 | optimizer/RNG/normalizer 版本 | checkpoint 不完整 | 重置优化器后称为无缝恢复 |

### 3. 真实系统发布边界

本章策略最多进入：离线回放、隔离仿真、shadow recommendation 和人工审阅的候选路线比较。进入真实设备前至少需要：

- 独立 hard constraint / action shield；
- 覆盖地图版本、传感器失效、通信延迟和交通密度的场景库；
- 与现行确定性规则和价值方法的冻结对照；
- 分布外状态拒绝与安全回退策略；
- 仿真—实机差异、奖励代理失真和观测延迟审计；
- 明确的停机、回滚、人工接管和责任链；
- 变更控制、签名模型包、配置 hash 与审计留痕。

“训练中隔离率为 0”不是安全证明：罕见状态可能从未访问，奖励可能漏掉危险，策略网络也可能被观测漂移推到训练支持之外。

## 13. 一页算法对照

| 方法 | Actor 权重信号 | Critic | 典型偏差 / 方差 | 数据分布 |
|---|---|---|---|---|
| REINFORCE | 完整 $G_t$ | 无 | 无偏、高方差 | on-policy |
| REINFORCE + baseline | $G_t-b(S_t)$ | MC baseline | score 可无偏、方差通常下降 | on-policy |
| 一步 Actor–Critic | $r+\gamma V(s')-V(s)$ | TD value | 近似 Critic 带偏、方差低 | 通常 on-policy |
| n-step A2C | $\hat G_t^{(n)}-V(s)$ | n-step value | $n$ 控制交换 | on-policy |
| GAE | TD residual 指数和 | value | $\lambda$ 控制交换 | on-policy rollout |
| PPO | clipped ratio × GAE | value | 启发式保守代理、仍需 KL | 近 on-policy、小量重复 |
| SAC | 最大熵 Q / 重参数化 | 双 Q | off-policy bootstrap 偏差 | replay / off-policy |

<MlQuiz
  kind="方法边界"
  question="哪种说法准确？"
  :options="['PPO 可对同一 on-policy rollout 做有限次受控更新；SAC 则需要完整 off-policy Q-learning 合同', 'PPO 与 SAC 只是名字不同', 'REINFORCE 可以无限重放旧轨迹而无需校正', 'Actor–Critic 不需要策略概率']"
  :answer="0"
  explanation="数据分布、target 和梯度估计不同；不能只复制 replay buffer 或 entropy 项就互相转换。"
/>

## 章节练习：从手算梯度到 A8 资格包

### 练习 1：双动作分流节点手算

设 $p(N)=0.3$，$q_N=6$，$q_E=-2$：

1. 计算 $J$ 与 $dJ/dz$；
2. 分别计算采样 $N/E$ 的 score；
3. 取 baseline $b=0,2,8$，计算两动作样本信号；
4. 证明三种 baseline 下的期望梯度相同；
5. 计算三种 baseline 的梯度方差，找出最优标量。

### 练习 2：只靠样本实现 REINFORCE

不能向训练函数传入 `HIDDEN_P` 或 `HIDDEN_R`：

1. 实现 batched reward-to-go；
2. 实现状态表 baseline 和一层 MLP baseline；
3. 比较无 baseline、均值 baseline、学习 baseline；
4. 固定环境步数预算，用 20 个预注册种子报告曲线；
5. 用隐藏模型仅在最后验收冻结策略；
6. 说明 batch size、学习率与熵系数如何选择，不能用测试种子选参。

### 练习 3：Actor–Critic 与 GAE

1. 实现一步 Actor–Critic；
2. 为 Critic target 和 Actor advantage 添加正确 detach；
3. 实现 `terminated` / `truncated` 双 mask；
4. 比较 $\lambda\in\{0,0.5,0.9,1\}$ 的偏差、方差与样本效率；
5. 加入共享 trunk，记录两头梯度范数和夹角；
6. 构造一个错误的 `done` mask，让回归测试能稳定抓住它。

### 练习 4：PPO 符号金丝雀

枚举 $\hat A\in\{-2,+2\}$ 与 $r\in\{0.6,0.9,1.1,1.5\}$，手算 clipped surrogate；实现 ratio、KL、clip fraction 与 early-stop。解释为什么只查看 ratio 是否落在 `[0.8,1.2]` 不能证明实现正确。

### 练习 5：A8 强化学习综合作业

在同一个 Gridworld 中提交策略评价、策略迭代、价值迭代、Monte Carlo、SARSA、Q-learning、REINFORCE 与带 baseline 的 Actor–Critic。报告：

- Bellman residual 与隐藏模型 oracle；
- SARSA / Q-learning 的探索风险路径；
- REINFORCE 三种方差缩减器；
- Actor–Critic 的 Critic 误差、advantage 分布和梯度 SNR；
- 多种子回报均值、标准差和置信带；
- 数据合同、动作 mask、终止/截断、版本键与安全边界。

```python
# TODO：完成 A8 时实现；本代码块保留为练习骨架，不参与自动执行。
def train_policy_gradient_agent(env, config):
    """返回冻结策略、训练账本、多种子评价与安全审计证据。"""
    raise NotImplementedError
```

## 来源与许可

本章依据本地解析语料交叉核对公式、算法边界和讲解顺序，正文推导、晶圆载具案例、NumPy 实现、数值结果、题目及三项网页交互均为本站独立创作：

1. Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 5，PDF 第 2–7 页用于 REINFORCE 与轨迹目标，PDF 第 15–18 页用于方差、baseline 和 reward-to-go，PDF 第 20–23 页用于离散策略梯度实现与高方差边界；本地解析稿为 `ref/ml101/opendataloader-pdf/corpus/berkeley-deep-rl/slides/lec-5/document.md` 。
2. Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 6，PDF 第 3–10 页用于价值 baseline 与策略评价，PDF 第 20–25 页用于 on-policy Actor–Critic、n-step 和 GAE，PDF 第 32–33 页仅用于重参数化与 SAC 的边界定位；本地解析稿为 `ref/ml101/opendataloader-pdf/corpus/berkeley-deep-rl/slides/lec-6/document.md` 。
3. UC Berkeley Deep Reinforcement Learning Section 3，PDF 第 5–8 页用于 Actor–Critic 偏差—方差、Critic stop-gradient、n-step 和 GAE 递推的交叉核对；本地解析稿为 `ref/ml101/opendataloader-pdf/corpus/berkeley-deep-rl/sections/section-3/document.md` 。
4. CMU 10-601/10-701 Lecture 22，*Policy Gradient + Deep Reinforcement Learning*，PDF 第 26–33 页用于策略梯度定理与 REINFORCE，PDF 第 40–47 页用于失效模式、baseline、Actor–Critic、A2C 与 n-step 对照；本地解析稿为 `ref/ml101/opendataloader-pdf/corpus/cmu-ml/slides/lecture22-policy-gradient-2/document.md` 。
5. Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 9，PDF 第 14–16 页用于 clipped importance ratio、熵正则与 PPO 边界；本地解析稿为 `ref/ml101/opendataloader-pdf/corpus/berkeley-deep-rl/slides/lec-9/document.md` 。

本章不复制课件图形、机器人案例、课堂题目、作业答案、代码或措辞。晶圆载具走廊的 14 个状态、4 个动作、随机偏移、奖励量规、训练结果、三项交互视觉和安全合同均为教学合成，不代表真实晶圆厂、AMR、地图、交通规则、设备能力、损失量规或发布标准。随机策略不得连接真实设备或安全控制链；仓库标准验证环境未为本章新增依赖。

<MlLessonProgress
  storage-key="32-policy-gradient"
  lesson-id="32"
  next-href="/courses/machine-learning/assignments"
  next-label="下一站：八次综合作业"
/>
