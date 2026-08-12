---
title: SARSA 与 Q-learning
description: 从晶圆载具样本流出发，推导 Monte Carlo、TD、n-step、SARSA、Expected SARSA、Q-learning、Double Q-learning 与 DQN 的工业实现边界
course: machine-learning
edition: industrial
---

<MlLessonHero
  number="31"
  title="SARSA 与 Q-learning"
  question="没有转移矩阵、只能看到一条条运行记录时，怎样让延迟回报沿轨迹传播，并分清‘评价实际探索策略’与‘学习贪心目标策略’？"
  duration="建议 780–900 分钟"
  :outcomes="[
    '把模型未知的工业序贯决策写成带行为概率、终止/截断、动作 mask 与版本键的 transition contract',
    '从回报恒等式推导 Monte Carlo、TD(0) 和 n-step 目标，解释 bootstrap 带来的偏差—方差交换',
    '逐项推导 SARSA、Expected SARSA 与 Q-learning，精确区分行为策略、目标策略和实际下一动作',
    '实现带平票语义、合法动作过滤、时间上限 bootstrap 和逐访问步长的表格控制算法',
    '解释最大化偏差、Double Q-learning、DQN 的回放/目标网络/stop-gradient，以及离线数据的分布外动作风险'
  ]"
/>

> 本章的随机探索只允许发生在合格仿真器、数字孪生或隔离的教学环境中。代码不得连接真实 AMR、下发路线、修改速度或避障参数；门禁、人员隔离、急停、洁净区许可和动作合法性由独立安全层强制执行，不能靠负奖励“学会遵守”。

## 贯穿案例：没有转移矩阵的晶圆载具走廊

### 1. 同一决策，换一种可见信息

第 30 章已知完整的转移核 $P(s'\mid s,a)$，所以一次 Bellman backup 可以对所有落点精确求和。本章仍使用同一条 `4 × 4` 教学走廊：

- 左下角是任务起点，右上角是交付口；
- 两格隔墙不可穿越，交付口下方是隔离终止区；
- 普通一步奖励为 $-0.45$，进入交付口为 $+14$，进入隔离区为 $-22$；
- 请求方向可能受地坪偏移影响；
- 折扣因子 $\gamma=0.94$。

变化只有一个：学习器不能读取 $P$ 和 $R$ 张量，只能依次收到

$$
(S_t,A_t,R_{t+1},S_{t+1},
\texttt{terminated}_{t+1},\texttt{truncated}_{t+1}).
$$

仿真器内部仍有隐藏的随机机制，否则无法生成样本；但算法接口只暴露 `step(action)`。隐藏模型只在本章末作为资格审查的 oracle，不能泄漏进训练 target。

### 2. transition contract 比四元组更长

工业日志至少要冻结：

| 字段组 | 示例字段 | 为什么进入合同 |
|---|---|---|
| 当前决策 | `episode_id`、`step_id`、`state_id`、`action_id` | 还原样本次序和访问频次 |
| 结果 | `reward`、`next_state_id` | 构造 MC/TD target |
| 边界 | `terminated`、`terminal_reason`、`truncated`、`time_limit` | 决定是否删除 bootstrap |
| 策略 | `behavior_policy_sha`、`behavior_prob`、`epsilon` | 判断 on/off-policy 与重要性权重 |
| 可行域 | `legal_action_mask_sha`、`legal_actions` | 证明探索没有越过硬约束 |
| 环境 | `simulator_sha`、`map_sha`、`reward_sha` | 避免把非平稳版本混为一个 MDP |
| 审计 | `rng_seed`、`collector_clock`、`schema_version` | 重放、去重和定位奖励错位 |

`behavior_prob` 必须记录“当时实际用于抽样的概率”，不能事后用新模型重算。若日志只留下贪心动作而没有探索概率，很多 off-policy 校正就失去依据。

<MlQuiz
  kind="数据合同"
  question="为什么仅保存 (state, action, reward, next_state) 仍不足以审计本章算法？"
  :options="['因为 Q 表必须存成图片', '因为还要区分终止与截断、重建行为概率、动作可行域和环境版本', '因为奖励必须是字符串', '因为每条记录都必须包含完整转移矩阵']"
  :answer="1"
  explanation="相同四元组在自然终止和时间截断下 target 不同；没有行为概率与版本键，也无法判断数据来自哪条策略、能否安全复放。"
/>

### 3. 唯一环境真相源：训练只能调用 `env_step`

下面的 $P,R$ 只模拟隐藏环境。后续学习函数不会接收它们；只有精确验收函数会显式使用。

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
                next_cell = move(cell, actual)
                next_state = INDEX[next_cell]
                reward = (
                    14.0 if next_cell == GOAL
                    else -22.0 if next_cell == HAZARD
                    else -0.45
                )
                probabilities[state, action, next_state] += weight
                rewards[state, action, next_state] = reward
    return probabilities, rewards

HIDDEN_P, HIDDEN_R = build_hidden_model()

def env_step(rng, state, action):
    """学习器可调用的唯一环境接口；不返回概率行。"""
    next_state = int(rng.choice(len(CELLS), p=HIDDEN_P[state, action]))
    reward = float(HIDDEN_R[state, action, next_state])
    terminated = next_state in TERMINALS
    return next_state, reward, terminated

assert HIDDEN_P.shape == (14, 4, 14)
assert np.allclose(HIDDEN_P.sum(axis=2), 1.0)
```

<MlQuiz
  kind="信息边界"
  question="“model-free” 最准确的含义是什么？"
  :options="['环境内部不存在动力学', '训练更新不需要显式给出或学习完整转移模型，但仍从环境样本获得信息', '算法不使用状态', '算法不允许做任何预测']"
  :answer="1"
  explanation="真实或模拟环境当然有转移机制；model-free 指算法不通过显式 P 做规划。样本本身仍携带关于该机制的信息。"
/>

## 1. 先做策略预测：同一价值的三种 target

### 1.1 回报恒等式是共同起点

固定行为策略 $\pi$，折扣回报为

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}.
$$

把第一项拆出：

$$
G_t=R_{t+1}+\gamma G_{t+1}.
$$

三类算法不是在定义不同的价值，而是在选择不同的随机 target 来估计同一个

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s].
$$

动态规划使用模型中的完整条件期望；Monte Carlo 等到未来真的发生；TD 则把未知的未来换成当前价值估计。

### 1.2 先冻结一条可解释行为策略

教学策略沿左侧主干北上，再沿顶行向东；若偏到右侧区域，就优先向南、向西返回。我们加入 $\epsilon=0.08$ 的合法动作探索，用来审计访问覆盖。

```python
def epsilon_greedy_probabilities(q_row, epsilon, legal_mask=None):
    if legal_mask is None:
        legal_mask = np.ones(len(q_row), dtype=bool)
    legal = np.flatnonzero(legal_mask)
    if len(legal) == 0:
        raise ValueError("当前状态没有合法动作")
    probabilities = np.zeros(len(q_row), dtype=float)
    probabilities[legal] = epsilon / len(legal)
    legal_values = q_row[legal]
    greedy = legal[np.isclose(legal_values, legal_values.max())]
    probabilities[greedy] += (1.0 - epsilon) / len(greedy)
    return probabilities

def heuristic_action(cell):
    row, col = cell
    if col == 0 and row > 0:
        return ACTIONS.index("N")
    if row == 0:
        return ACTIONS.index("E")
    if col >= 2 and row < 3:
        return ACTIONS.index("S")
    return ACTIONS.index("W")

def heuristic_policy(epsilon=0.08):
    policy = np.zeros((len(CELLS), len(ACTIONS)))
    for state, cell in enumerate(CELLS):
        if state in TERMINALS:
            policy[state] = 0.25
            continue
        preferences = np.zeros(len(ACTIONS))
        preferences[heuristic_action(cell)] = 1.0
        policy[state] = epsilon_greedy_probabilities(preferences, epsilon)
    return policy

def evaluate_policy_exact(policy, gamma=GAMMA):
    """仅用于资格审查：显式读取隐藏模型。"""
    p_policy = np.einsum("sa,san->sn", policy, HIDDEN_P)
    r_policy = np.einsum("sa,san,san->s", policy, HIDDEN_P, HIDDEN_R)
    value = np.linalg.solve(
        np.eye(len(CELLS)) - gamma * p_policy,
        r_policy,
    )
    return value

BEHAVIOR_POLICY = heuristic_policy()
ORACLE_V_BEHAVIOR = evaluate_policy_exact(BEHAVIOR_POLICY)
assert np.allclose(BEHAVIOR_POLICY.sum(axis=1), 1.0)
```

平票不是实现细节。若两个合法动作同时最大，常用语义是把 $1-\epsilon$ 在所有最大动作间均分；若实现总选第一个动作，行为概率、覆盖与可复现性都会改变。

<MlQuiz
  kind="概率语义"
  question="某状态有 4 个合法动作，其中 2 个并列最大，epsilon=0.2。按本章平票规则，每个最大动作概率是多少？"
  :options="['0.45', '0.25', '0.80', '0.10']"
  :answer="0"
  explanation="每个动作先分到 0.2/4=0.05；剩余 0.8 在两个最大动作间均分，所以各为 0.45。"
/>

### 1.3 Monte Carlo：把整段未来压回访问点

对一条已经自然终止的轨迹，Monte Carlo target 就是实际回报：

$$
V(S_t)\leftarrow V(S_t)+\alpha_t(S_t)\,[G_t-V(S_t)].
$$

若用样本均值，某状态第 $N$ 次更新的步长取 $1/N$，在线递推与保存全部回报再求均值等价。两种访问约定必须写清：

- **first-visit MC**：每回合只更新某状态第一次出现的位置；
- **every-visit MC**：该状态每次出现都更新。

两者在合适条件下都收敛，但有限样本的相关结构和权重不同。本章实现 first-visit，并丢弃时间上限截断的回合，避免把不完整尾部冒充完整 $G_t$。

<MlQuiz
  kind="估计量"
  question="为什么时间上限截断的轨迹不能不加说明地作为完整 Monte Carlo 回报？"
  :options="['因为截断前奖励都无效', '因为被删掉的尾部仍可能有价值，直接当完整回报会系统性改变 target', '因为 MC 只能用于连续任务', '因为 gamma 必须等于 1']"
  :answer="1"
  explanation="自然终止后未来回报为零；时间截断只是记录停止。二者混同会把未知尾部强行设为零。"
/>

### 1.4 TD(0)：一次转移就能更新

Bellman 期望方程给出

$$
V^\pi(S_t)
=\mathbb E_\pi[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t].
$$

用一次样本替代环境期望、用当前估计替代未知真值：

$$
\delta_t
=R_{t+1}+\gamma(1-\mathbb 1_{\text{terminated}})V(S_{t+1})-V(S_t),
$$

$$
V(S_t)\leftarrow V(S_t)+\alpha_t\delta_t.
$$

这叫 bootstrap：target 的一部分来自正在学习的估计。它通常比完整 MC 方差低、能在线更新，也因此在有限训练中带有估计误差造成的偏差。

<ReturnTargetMixerLab />

<MlQuiz
  kind="目标拆解"
  question="TD error 为正说明什么？"
  :options="['当前一步样本 target 高于旧价值，应把 V(S_t) 向上移', '策略已经全局最优', '奖励一定为正', '下一状态必须终止']"
  :answer="0"
  explanation="delta 是局部惊讶量，只比较当前 target 与旧估计；它不单独证明策略最优或模型正确。"
/>

::: warning 停下来推一推 1
<!-- ml-stop -->
已记录奖励为 $[-0.45,-0.45,14]$，$\gamma=0.94$，当前 $V(S_{t+1})=4.2$。分别计算 $t$ 时刻的 1-step TD target、2-step target 和完整 MC target。若第二步后只是时间截断，哪个 target 仍应保留 bootstrap？
:::

### 1.5 用同一批交互比较 MC 与 TD

```python
def generate_episode(rng, policy, max_steps=160):
    trace = []
    state = START
    for step in range(max_steps):
        action = int(rng.choice(len(ACTIONS), p=policy[state]))
        next_state, reward, terminated = env_step(rng, state, action)
        truncated = (step + 1 == max_steps) and not terminated
        trace.append((state, action, reward, next_state, terminated, truncated))
        if terminated or truncated:
            break
        state = next_state
    return trace

def first_visit_mc(policy, episodes=5000, seed=3101):
    rng = np.random.default_rng(seed)
    value = np.zeros(len(CELLS))
    counts = np.zeros(len(CELLS), dtype=int)
    discarded = 0
    for _ in range(episodes):
        trace = generate_episode(rng, policy)
        if trace[-1][5]:
            discarded += 1
            continue
        returns = np.zeros(len(trace))
        running = 0.0
        for index in range(len(trace) - 1, -1, -1):
            running = trace[index][2] + GAMMA * running
            returns[index] = running
        seen = set()
        for item, target in zip(trace, returns):
            state = item[0]
            if state in seen:
                continue
            seen.add(state)
            counts[state] += 1
            value[state] += (target - value[state]) / counts[state]
    return value, counts, discarded

def td_zero(policy, episodes=5000, seed=3102):
    rng = np.random.default_rng(seed)
    value = np.zeros(len(CELLS))
    visits = np.zeros(len(CELLS), dtype=int)
    for _ in range(episodes):
        for state, _, reward, next_state, terminated, truncated in generate_episode(rng, policy):
            visits[state] += 1
            alpha = 0.35 / (1.0 + 0.02 * visits[state]) ** 0.55
            bootstrap = 0.0 if terminated else value[next_state]
            target = reward + GAMMA * bootstrap
            value[state] += alpha * (target - value[state])
            if truncated:
                break
    return value, visits

MC_V, MC_COUNTS, MC_DISCARDED = first_visit_mc(BEHAVIOR_POLICY)
TD_V, TD_VISITS = td_zero(BEHAVIOR_POLICY)
NONTERMINAL = np.array([state not in TERMINALS for state in range(len(CELLS))])
MC_MAE = np.mean(np.abs(MC_V[NONTERMINAL] - ORACLE_V_BEHAVIOR[NONTERMINAL]))
TD_MAE = np.mean(np.abs(TD_V[NONTERMINAL] - ORACLE_V_BEHAVIOR[NONTERMINAL]))

assert MC_COUNTS[START] > 4500
assert MC_DISCARDED < 20
assert np.isfinite(MC_MAE + TD_MAE)
```

这里的精确值只能用于教学仿真器回归测试。真实环境没有 oracle，必须改用冻结仿真基准、置信区间、多随机种子、覆盖率和真实业务切片验收，不能把一条训练曲线当真值。

## 2. n-step：在偏差与方差之间移动撕纸线

### 2.1 一般 target

从 $t$ 开始看 $n$ 步：

$$
G_t^{(n)}
=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}
+\gamma^nV(S_{t+n}).
$$

若在第 $n$ 步前自然终止，后续项为零；若只是时间截断且 $S_{t+n}$ 仍有合理的延续价值，则保留 bootstrap。$n=1$ 是 TD(0)，$n$ 走到自然终点是 MC。

### 2.2 “更大的 n 更好”是错误结论

- 小 $n$：传播快、方差较低，但更依赖当前 $V$ 的误差；
- 大 $n$：依赖更多真实奖励、bootstrap 偏差较少，但累积更多环境随机性；
- 稀疏奖励下，过小 $n$ 可能让信号传播很慢；
- off-policy 下，跨越更多行为动作会增加分布校正难度；
- 并行环境中，$n$ 还决定 rollout 缓冲长度与更新延迟。

```python
def n_step_target(rewards, bootstrap_value, gamma=GAMMA):
    target = 0.0
    for offset, reward in enumerate(rewards):
        target += (gamma ** offset) * reward
    return target + (gamma ** len(rewards)) * bootstrap_value

sample_rewards = [-0.45, -0.45, 14.0]
one_step = n_step_target(sample_rewards[:1], bootstrap_value=4.2)
two_step = n_step_target(sample_rewards[:2], bootstrap_value=8.0)
monte_carlo = n_step_target(sample_rewards, bootstrap_value=0.0)

assert np.isclose(one_step, -0.45 + GAMMA * 4.2)
assert np.isclose(monte_carlo, -0.45 - 0.45 * GAMMA + 14 * GAMMA ** 2)
```

<MlQuiz
  kind="偏差方差"
  question="在其他条件相同、价值估计早期很差时，增大 n 通常带来什么交换？"
  :options="['减少对 bootstrap 的依赖，但引入更多轨迹随机性', '同时消除偏差和方差', '必须已知完整 P', '让所有奖励不再折扣']"
  :answer="0"
  explanation="n-step 在一步 bootstrap 与完整样本回报之间插值，不存在对所有任务都占优的固定 n。"
/>

## 3. 从状态价值到动作价值：控制问题的接口

### 3.1 为什么要学习 $Q$

只知道 $V(s)$ 仍需要环境模型才能比较“现在先做哪个动作”。动作价值把当前动作也纳入条件：

$$
Q^\pi(s,a)
=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

一旦有 $Q$，策略改进可以直接在合法动作内做

$$
\pi_{\text{greedy}}(s)\in\arg\max_{a\in\mathcal A(s)}Q(s,a).
$$

采样控制交替执行：

1. 用行为策略收集 transition；
2. 用某种 target 更新 $Q$；
3. 让行为或目标策略相对 $Q$ 更贪心；
4. 继续覆盖尚不确定的合法动作。

这就是 generalized policy iteration 的样本版。SARSA 与 Q-learning 的关键差异只在第 2 步 target，却对应不同的策略问题。

<MlQuiz
  kind="接口选择"
  question="没有环境模型时，为什么控制算法通常直接学习 Q 而不是只学习 V？"
  :options="['Q 不需要奖励', 'Q 已按当前动作分解未来回报，可以直接比较合法动作', 'V 只能是负数', 'Q 自动解决安全约束']"
  :answer="1"
  explanation="V 只评价状态；在未知模型下不能用它前瞻每个动作。Q 将当前动作纳入条件，因此可直接做策略改进。"
/>

### 3.2 行为策略与目标策略必须分栏

- **行为策略** $\mu(a\mid s)$：真正生成动作和数据；
- **目标策略** $\pi(a\mid s)$：当前更新希望评价或逼近的策略。

若 $\mu=\pi$，是 on-policy；若允许不同，是 off-policy。这里的“off-policy”不等于“离线”：在线 Q-learning 也可以一边执行 $\epsilon$-greedy 行为策略，一边学习贪心目标策略。

<MlQuiz
  kind="概念辨析"
  question="哪种描述是正确的？"
  :options="['off-policy 必然只读历史文件', 'on-policy 不允许随机探索', '在线收集 epsilon-greedy 数据，同时用 max 目标学习贪心策略，仍是 off-policy', '行为策略和目标策略永远相同']"
  :answer="2"
  explanation="在线/离线描述数据何时收集；on/off-policy 描述生成数据的策略与被评价策略是否一致，是两条不同轴。"
/>

## 4. SARSA：让实际下一动作进入 target

### 4.1 五元组与更新

SARSA 名称来自

$$
(S_t,A_t,R_{t+1},S_{t+1},A_{t+1}).
$$

行为策略先在 $S_{t+1}$ 实际抽出 $A_{t+1}$，再构造

$$
Y_t^{\text{SARSA}}
=R_{t+1}
+\gamma(1-\mathbb 1_{\text{terminated}})
Q(S_{t+1},A_{t+1}).
$$

更新为

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)
+\alpha_t\,[Y_t^{\text{SARSA}}-Q(S_t,A_t)].
$$

若行为保持 $\epsilon$-greedy，实际下一动作偶尔会探索，所以 SARSA 学到的是这条带探索策略的动作价值，而不是假装下一步总会贪心。

<MlQuiz
  kind="算法识别"
  question="SARSA 为什么是 on-policy？"
  :options="['因为它不使用 Q', '因为 target 使用行为策略实际抽出的下一动作，并继续评价这条行为策略', '因为它需要完整转移矩阵', '因为它永远取 max']"
  :answer="1"
  explanation="数据与 target 都指向同一行为策略；当 epsilon 改变时，SARSA 正在评价的策略也改变。"
/>

### 4.2 Expected SARSA：对下一动作求策略期望

既然已知行为策略概率，就可以把下一动作的抽样换成解析期望：

$$
Y_t^{\text{Exp}}
=R_{t+1}
+\gamma(1-\mathbb 1_{\text{terminated}})
\sum_{a'}\mu(a'\mid S_{t+1})Q(S_{t+1},a').
$$

它仍评价 $\mu$，但消除了“恰好抽到哪个 $A_{t+1}$”这一层方差；环境的下一状态和奖励仍是随机样本。若动作数很大或行为概率不可得，求和也可能昂贵或不可信。

<MlQuiz
  kind="目标比较"
  question="Expected SARSA 相对 SARSA 消除了哪一层随机性？"
  :options="['环境转移的全部随机性', '下一状态行为动作的单次抽样随机性', '奖励定义的不确定性', '参数初始化的随机性']"
  :answer="1"
  explanation="它仍使用样本得到的 r 和 s′，只把给定 s′ 后的下一动作按行为分布精确平均。"
/>

<SarsaQTargetRelayLab />

::: warning 停下来推一推 2
<!-- ml-stop -->
某次 transition 的奖励为 $-0.45$，下一状态四个 $Q$ 值为 `[8, 5, 1, -2]`；行为策略以概率 `[0.70, 0.10, 0.10, 0.10]` 选动作，这次实际选到第二个动作。取 $\gamma=0.94$，分别计算 SARSA、Expected SARSA 和 Q-learning target，并说明三者正在回答什么策略问题。
:::

## 5. Q-learning：样本来自行为策略，target 指向贪心策略

### 5.1 从 Bellman 最优方程到随机逼近

最优动作价值满足

$$
Q^\star(s,a)
=\mathbb E\left[
R_{t+1}+\gamma\max_{a'}Q^\star(S_{t+1},a')
\mid S_t=s,A_t=a
\right].
$$

观察一次 transition 后，Q-learning 用

$$
Y_t^Q
=R_{t+1}
+\gamma(1-\mathbb 1_{\text{terminated}})
\max_{a'\in\mathcal A(S_{t+1})}Q(S_{t+1},a').
$$

环境期望由样本近似，下一动作却不取行为实际动作，而取贪心目标。这正是 off-policy 的来源。

### 5.2 三个 target 只差最后一项

令 $q_{t+1}(a)=Q(S_{t+1},a)$：

| 算法 | bootstrap 项 | 评价的策略 | 下一动作噪声 |
|---|---|---|---|
| SARSA | $q_{t+1}(A_{t+1})$ | 当前行为策略 $\mu$ | 有 |
| Expected SARSA | $\sum_a\mu(a\mid S_{t+1})q_{t+1}(a)$ | 当前行为策略 $\mu$ | 被积分掉 |
| Q-learning | $\max_a q_{t+1}(a)$ | 贪心目标策略 | 无，但有最大化偏差 |

当 $\epsilon=0$ 且贪心动作唯一时，三者 target 相同；这不代表它们在探索期间相同。

```python
def control_targets(reward, next_q, next_action, epsilon, terminated=False):
    if terminated:
        return {"sarsa": reward, "expected": reward, "q_learning": reward}
    probabilities = epsilon_greedy_probabilities(next_q, epsilon)
    return {
        "sarsa": reward + GAMMA * next_q[next_action],
        "expected": reward + GAMMA * float(probabilities @ next_q),
        "q_learning": reward + GAMMA * float(np.max(next_q)),
    }

TARGET_EXAMPLE = control_targets(
    reward=-0.45,
    next_q=np.array([8.0, 5.0, 1.0, -2.0]),
    next_action=1,
    epsilon=0.40,
)
assert TARGET_EXAMPLE["sarsa"] < TARGET_EXAMPLE["expected"] < TARGET_EXAMPLE["q_learning"]
```

<MlQuiz
  kind="算法识别"
  question="同一次样本中，行为在 s′ 选了非贪心动作，但 target 仍使用 max Q(s′,a′)。这是哪种更新？"
  :options="['first-visit MC', 'SARSA', 'Q-learning', '策略评价的动态规划']"
  :answer="2"
  explanation="是否 off-policy 看 target 指向谁；使用 max 表示目标策略贪心，与实际下一行为动作可以不同。"
/>

## 6. 探索不是给 epsilon 一个滑块

### 6.1 覆盖与利用的双重要求

若一开始 $\epsilon=0$，未访问动作可能永远没有机会纠正错误初值。经典表格 Q-learning 的收敛条件通常包括：

- 有限 MDP、奖励有界、$\gamma<1$；
- 环境和奖励语义平稳；
- 每个合法状态—动作对被访问无穷多次；
- 对每个 $(s,a)$，步长满足
  $\sum_k\alpha_k(s,a)=\infty$、
  $\sum_k\alpha_k^2(s,a)<\infty$；
- target、终止和平票实现与理论合同一致。

GLIE（greedy in the limit with infinite exploration）要求探索持续提供无限覆盖，同时策略最终趋于贪心。真实项目预算有限，做不到“无穷”，所以应报告每个状态动作的访问数、有效样本量和最小覆盖，而不是声称满足渐近定理。

<MlQuiz
  kind="收敛条件"
  question="固定常数学习率 alpha 能否满足经典 Robbins–Monro 的平方可和条件？"
  :options="['能，因为 alpha 大于 0', '不能；常数步长的平方和也发散', '只要 epsilon 固定就能', '只要奖励为负就能']"
  :answer="1"
  explanation="常数步长适合追踪漂移，但通常不会收敛到无噪声定点；工程上要明确是在追踪还是在做渐近估计。"
/>

### 6.2 安全探索先经过 action mask

探索分布必须定义在 $\mathcal A_{\text{legal}}(s)$ 上：

$$
\mu(a\mid s)=0,\qquad a\notin\mathcal A_{\text{legal}}(s).
$$

正确顺序是先由独立规则删除非法动作，再在剩余动作上分配 $\epsilon$。把非法动作的 $Q$ 设为一个很小值不够：数值溢出、初始化、mask 漏传或模型外推都可能让它重获概率。

### 6.3 固定 epsilon 下，SARSA 与 Q-learning 的政策口径不同

SARSA 评价持续探索的行为策略，因此会把未来探索风险计入当前动作；Q-learning 学贪心目标策略，却仍由探索行为产生训练数据。部署验收要区分：

- `greedy_target_return`：关闭探索后的目标策略回报；
- `behavior_return_at_epsilon`：保持训练探索时的实际回报；
- `hazard_rate` 与尾部分位数：不能被均值掩盖；
- `coverage_min`：最少访问状态动作对；
- `seed_dispersion`：不同随机种子间波动。

<MlQuiz
  kind="验收口径"
  question="训练时 epsilon=0.2，只报告关闭探索后的贪心回报有什么问题？"
  :options="['没有问题，两者必然相等', '它隐藏了数据收集期间的探索成本与风险，不能代表实际行为策略表现', '贪心策略不能计算回报', 'epsilon 只影响学习率']"
  :answer="1"
  explanation="Q-learning 的目标策略和行为策略本就不同；必须分开报告，否则会用理想部署策略掩盖训练行为。"
/>

## 7. 表格控制实现：同一个循环，只换 target

### 7.1 终止与截断进入代码分支

自然终止：

$$
Y=R_{t+1}.
$$

时间截断并不把 MDP 未来变成零，因此若 `next_state` 可用于价值估计，应计算 bootstrap 后再结束当前采集回合。SARSA 在截断边界还需从延续行为策略抽一个 `next_action` 供 target 使用，但不再把该动作下发给已重置环境。

```python
def train_control(algorithm, episodes=7000, epsilon=0.20, seed=0):
    if algorithm not in {"sarsa", "expected_sarsa", "q_learning"}:
        raise ValueError("未知算法")
    rng = np.random.default_rng(seed)
    q = np.zeros((len(CELLS), len(ACTIONS)))
    visits = np.zeros_like(q, dtype=int)
    episode_returns = []

    for _ in range(episodes):
        state = START
        probabilities = epsilon_greedy_probabilities(q[state], epsilon)
        action = int(rng.choice(len(ACTIONS), p=probabilities))
        discounted_return = 0.0
        discount = 1.0

        for step in range(160):
            next_state, reward, terminated = env_step(rng, state, action)
            truncated = (step + 1 == 160) and not terminated
            discounted_return += discount * reward
            discount *= GAMMA
            visits[state, action] += 1
            alpha = 0.45 / (1.0 + 0.015 * visits[state, action]) ** 0.55

            if terminated:
                target = reward
                next_action = 0
            else:
                next_probabilities = epsilon_greedy_probabilities(q[next_state], epsilon)
                next_action = int(rng.choice(len(ACTIONS), p=next_probabilities))
                if algorithm == "sarsa":
                    future = q[next_state, next_action]
                elif algorithm == "expected_sarsa":
                    future = float(next_probabilities @ q[next_state])
                else:
                    future = float(np.max(q[next_state]))
                target = reward + GAMMA * future

            q[state, action] += alpha * (target - q[state, action])
            if terminated or truncated:
                break
            state, action = next_state, next_action

        episode_returns.append(discounted_return)

    return q, visits, np.asarray(episode_returns)

CONTROL_RUNS = {
    name: train_control(name, seed=3110 + index)
    for index, name in enumerate(("sarsa", "expected_sarsa", "q_learning"))
}

for q, visits, returns in CONTROL_RUNS.values():
    assert np.isfinite(q).all()
    assert visits[START].sum() >= 7000
    assert len(returns) == 7000
```

<QRouteLearningPlaybackLab />

<MlQuiz
  kind="边界语义"
  question="时间上限截断后，Q-learning target 的常见正确处理是什么？"
  :options="['无条件令 future=0', '若过程语义仍继续且 next_state 有效，保留 max Q(next_state) 的 bootstrap，再结束本采集回合', '删除当前奖励', '把 truncated 改写成 terminated']"
  :answer="1"
  explanation="截断是采集边界，不是环境吸收态。是否 bootstrap 由任务语义决定，不能只看循环是否 break。"
/>

### 7.2 Expected SARSA 不等于“总比 SARSA 好”

Expected SARSA 在动作数较少且行为概率精确时，常能降低 target 方差；但它也有边界：

- 行为概率日志错了，解析期望会稳定地算错；
- 动作空间巨大时，完整求和昂贵；
- 动作 mask 若参与概率归一化却未版本化，target 无法重放；
- 行为策略在一批数据内快速变化，旧 transition 与新概率不能混用；
- 分布式采集器延迟同步时，必须记录 actor 当时的策略版本。

SARSA 的实际下一动作反而是直接可观察事实。算法选择不能脱离日志质量。

## 8. 最大化偏差与 Double Q-learning

### 8.1 `max` 同时做选择和评价

假设两个动作真实价值相同，估计都是真值加零均值噪声。即使每个估计无偏，

$$
\mathbb E[\max_a\widehat Q_a]
\ge \max_a\mathbb E[\widehat Q_a].
$$

`max` 更容易选中被正噪声推高的动作，所以赢家估计偏乐观。动作越多、估计越噪，偏差通常越明显。

### 8.2 双表把选择和评价拆开

Double Q-learning 维护 $Q^A,Q^B$。随机选择更新 $Q^A$ 时：

$$
a^\star=\arg\max_a Q^A(s',a),
$$

$$
Y=R+\gamma Q^B(s',a^\star).
$$

更新 $Q^B$ 时对调角色。独立并不完美，但选择噪声与评价噪声不再完全是同一份。

```python
rng = np.random.default_rng(3188)
n_trials = 200_000
estimate_a = rng.normal(0.0, 1.0, size=(n_trials, 2))
estimate_b = rng.normal(0.0, 1.0, size=(n_trials, 2))

single_q_bias = np.max(estimate_a, axis=1).mean()
selected_actions = np.argmax(estimate_a, axis=1)
double_q_estimate = estimate_b[np.arange(n_trials), selected_actions].mean()

assert single_q_bias > 0.5
assert abs(double_q_estimate) < 0.02
```

<MlQuiz
  kind="统计偏差"
  question="Double Q-learning 的核心拆分是什么？"
  :options="['一张表存正奖励，一张表存负奖励', '一套估计选动作，另一套估计评价被选动作', '一张表用于终止，一张表用于截断', '把动作随机分成合法与非法']"
  :answer="1"
  explanation="最大化偏差来自同一噪声既决定赢家又给赢家打分；双估计器拆开这两个角色。"
/>

::: warning 停下来推一推 3
<!-- ml-stop -->
三个动作真实价值都为 0，每个估计独立服从 $N(0,1)$。不必积分，先用对称性解释为什么单个估计器的最大值期望为正；再说明“用 A 选动作、用独立 B 评价”为什么把期望拉回 0。若 A、B 高度相关，结论会怎样变化？
:::

## 9. 从表格到 DQN：不是把数组换成网络就结束

### 9.1 函数逼近改变了收敛故事

状态是图像、点云或高维设备上下文时，无法存完整表格，改用 $Q_\theta(s,a)$。经典 DQN 对 replay buffer $\mathcal B$ 中的小批量最小化

$$
L(\theta)
=\mathbb E_{\mathcal B}
\left[
\ell\left(Q_\theta(s,a)-Y\right)
\right],
$$

$$
Y=r+\gamma(1-d)
\max_{a'}Q_{\bar\theta}(s',a').
$$

$\bar\theta$ 是目标网络参数，$d$ 只表示真正终止。$\ell$ 可用平方损失；Huber loss 在大 TD error 下转为线性增长，降低异常 target 对梯度的支配。

### 9.2 经验回放解决什么，不解决什么

经验回放：

- 打散相邻 transition 的强相关；
- 让一条昂贵样本被复用；
- 让小批量更接近稳定回归数据；
- 可以混合多个采集 actor 的数据。

它不自动修复：

- 旧数据与新地图/奖励混用；
- replay 中非法或 schema 错位动作；
- 离线数据没有覆盖却对分布外动作取 `max`；
- 罕见严重事件被均匀采样淹没；
- 大 buffer 导致策略版本过旧。

<MlQuiz
  kind="组件职责"
  question="DQN 的 replay buffer 主要缓解哪项问题？"
  :options="['相邻样本相关且每条交互只用一次', '硬安全约束缺失', '奖励定义错误', '所有分布外动作']"
  :answer="0"
  explanation="回放改善优化数据流和样本复用；它不能替代动作安全层、版本治理或离线 RL 的支持约束。"
/>

### 9.3 目标网络与 stop-gradient

若 online network 同时产生预测和 target，每次参数更新都会让追赶目标立即移动。DQN 用延迟复制或 Polyak 平均更新 $\bar\theta$：

$$
\bar\theta\leftarrow\theta\quad\text{每 }C\text{ 步},
$$

或

$$
\bar\theta\leftarrow\tau\theta+(1-\tau)\bar\theta.
$$

计算梯度时把 $Y$ 当常数，不让梯度穿过 target 分支。这是半梯度 TD，不是把 Bellman 平方误差对同一随机样本做普通全梯度回归。

<MlQuiz
  kind="组件职责"
  question="目标网络的直接作用是什么？"
  :options="['让回归 target 在一段更新内相对稳定', '保证训练样本独立同分布', '删除探索', '证明神经网络一定收敛']"
  :answer="0"
  explanation="延迟目标减少“预测追逐自己”的速度，但函数逼近、off-policy 与 bootstrap 的组合仍可能不稳定。"
/>

### 9.4 Double DQN：在线网选、目标网评

Double DQN target 为

$$
a^\star=\arg\max_a Q_\theta(s',a),
$$

$$
Y_{\text{DDQN}}
=r+\gamma(1-d)Q_{\bar\theta}(s',a^\star).
$$

在线网络负责选择，目标网络负责评价。它与双表思想一致，但两网络因周期复制而相关，所以是降低偏差，不是数学上彻底消除偏差。

```python
def dqn_targets(rewards, terminated, online_next_q, target_next_q):
    rewards = np.asarray(rewards, dtype=float)
    terminated = np.asarray(terminated, dtype=bool)
    dqn = rewards + GAMMA * (~terminated) * np.max(target_next_q, axis=1)
    selected = np.argmax(online_next_q, axis=1)
    double = rewards + GAMMA * (~terminated) * target_next_q[
        np.arange(len(rewards)), selected
    ]
    return dqn, double

batch_rewards = np.array([-0.45, 14.0, -0.45])
batch_terminated = np.array([False, True, False])
online_next = np.array([[8.0, 7.0], [2.0, 3.0], [5.0, 6.0]])
target_next = np.array([[6.0, 7.5], [9.0, 10.0], [5.5, 5.0]])
DQN_Y, DDQN_Y = dqn_targets(
    batch_rewards, batch_terminated, online_next, target_next
)

assert DQN_Y[1] == 14.0
assert DDQN_Y[1] == 14.0
assert DQN_Y[0] != DDQN_Y[0]
```

### 9.5 deadly triad 是风险标签，不是单一故障

下面三项同时出现时尤其危险：

1. **函数逼近**：不同状态动作共享参数；
2. **bootstrap**：target 依赖当前估计；
3. **off-policy**：训练分布与目标策略不同。

DQN 的 replay、目标网络、Huber、梯度裁剪、Double DQN、探索调度和多随机种子都是工程缓解，不构成“必收敛”证明。先在可精确求解的小 MDP 上验证 target、终止、mask 与平票，再扩展到神经网络，比直接在大任务上看平均回报更可靠。

## 10. 离线 replay 不是免费 off-policy 学习

在线 Q-learning 能通过持续探索补齐动作覆盖；纯离线数据不能再向环境查询。若某状态只有动作 A 的数据，target 却对未见动作 B 取 `max Q(s,B)`，函数逼近误差可能把 B 推得很高，再通过 bootstrap 反复放大。

因此普通 DQN 不能因为“本来就是 off-policy”就直接获得可靠离线 RL 能力。至少要检查：

- 行为支持：目标动作在数据中是否出现；
- 状态动作密度与切片覆盖；
- 数据集版本、重复 transition 和泄漏；
- OOD action 的 Q 分布与不确定性；
- 保守目标、行为约束或专门离线 RL 方法的必要性；
- 任何线上试验是否经过独立安全与变更审批。

<MlQuiz
  kind="分布边界"
  question="为什么标准 Q-learning 在固定离线数据上可能对未见动作产生灾难性乐观估计？"
  :options="['因为离线数据没有奖励', 'max 会查询缺乏支持的动作，函数逼近误差再经 bootstrap 传播', '因为 off-policy 只能用于表格', '因为 replay buffer 太小必然为零']"
  :answer="1"
  explanation="off-policy 不等于可以无限外推；没有交互纠错时，分布外动作的高估会自举放大。"
/>

## 复杂度与工程边界

### 11.1 表格算法成本

设一共收集 $T$ 条 transition：

- SARSA 与 Q-learning 每条更新若维护最大值，时间通常为 $O(|A|)$；
- Expected SARSA 要对合法动作求和，也是 $O(|A|)$；
- 表格存储为 $O(|S||A|)$；
- first-visit MC 需保留一回合，额外空间 $O(H)$；
- n-step 在线实现需长度 $O(n)$ 的队列；
- Double Q-learning 把 Q 表存储常数翻倍；
- 精确资格 oracle 的稠密模型存储为 $O(|S|^2|A|)$，但不属于训练依赖。

若动作数极大，完整 `max` 和策略期望都会成为瓶颈，需要候选生成、分层动作或 actor；这会改变算法接口，应记录架构决策，不能藏在一个近似 `argmax` 中。

### 11.2 神经网络训练成本

一次 DQN 更新主要由网络前后向决定，约为 $O(B\cdot C_\theta)$；replay 存储约为 $O(N_{\mathcal B}\cdot d_{\text{transition}})$。真正的工程瓶颈常在：

- 图像/传感器 observation 的序列化与传输；
- replay 采样和版本过滤；
- 多 actor 产生速度与 learner 消耗速度不匹配；
- target network 同步和检查点一致性；
- 评估环境吞吐与多随机种子成本。

### 11.3 发布前的不可省略项

1. 状态、动作、奖励时点、终止与截断 schema 单测；
2. 合法动作 mask 在行为抽样、target `max` 和部署抽取三处一致；
3. 行为概率、策略版本与环境版本可重放；
4. 每状态动作访问数和未覆盖切片；
5. MC、TD 和至少一种精确小模型基准交叉核对；
6. SARSA/Expected SARSA/Q-learning 的 target 单元测试；
7. 常数 epsilon 与关闭探索两种回报分开报告；
8. 多随机种子均值、分位数、失败率和 hazard rate；
9. DQN target stop-gradient、终止 mask 和 Double 选择/评价分支测试；
10. 回放数据的时间、版本和 OOD action 审计；
11. 撤回条件、模型卡和策略 hash；
12. 明确输出仅为离线建议，不接入真实安全控制链。

## 数值验证

### 12.1 隐藏模型只在验收时出现

为了验证表格实现，先用第 30 章的动态规划 oracle 求 $Q^\star$。注意 `train_control` 的签名没有 `HIDDEN_P,HIDDEN_R`；只有这里的资格函数能访问它们。

```python
def q_from_value(value):
    return np.sum(
        HIDDEN_P * (HIDDEN_R + GAMMA * value[None, None, :]),
        axis=2,
    )

def value_iteration(tolerance=1e-12):
    value = np.zeros(len(CELLS))
    for _ in range(10_000):
        next_value = np.max(q_from_value(value), axis=1)
        next_value[list(TERMINALS)] = 0.0
        if np.max(np.abs(next_value - value)) < tolerance:
            return next_value
        value = next_value
    raise RuntimeError("价值迭代未收敛")

def policy_from_q(q, epsilon=0.0):
    return np.vstack([
        np.full(len(ACTIONS), 0.25)
        if state in TERMINALS
        else epsilon_greedy_probabilities(q[state], epsilon)
        for state in range(len(CELLS))
    ])

ORACLE_V_STAR = value_iteration()
ORACLE_Q_STAR = q_from_value(ORACLE_V_STAR)
ORACLE_Q_STAR[list(TERMINALS)] = 0.0

QUALIFICATION = {}
for name, (q, visits, returns) in CONTROL_RUNS.items():
    greedy_value = evaluate_policy_exact(policy_from_q(q, epsilon=0.0))
    behavior_value = evaluate_policy_exact(policy_from_q(q, epsilon=0.20))
    q_mae = np.mean(np.abs(q[NONTERMINAL] - ORACLE_Q_STAR[NONTERMINAL]))
    QUALIFICATION[name] = {
        "greedy_start": float(greedy_value[START]),
        "behavior_start": float(behavior_value[START]),
        "q_mae_to_optimal": float(q_mae),
        "min_state_action_visits": int(visits[NONTERMINAL].min()),
        "last_1000_sample_return": float(returns[-1000:].mean()),
    }

assert np.isclose(ORACLE_V_STAR[START], 6.550967338, atol=1e-8)
assert all(item["min_state_action_visits"] > 0 for item in QUALIFICATION.values())
assert QUALIFICATION["q_learning"]["q_mae_to_optimal"] < 2.5
```

这里不要求三种算法学到相同 $Q$：固定 $\epsilon=0.20$ 时，SARSA 与 Expected SARSA 的定点对应探索行为策略，Q-learning 才以 $Q^\star$ 为目标。更公平的比较是：

- 对 Q-learning 报 `q_mae_to_optimal`；
- 对 on-policy 方法报相对其行为策略 Bellman 方程的 residual；
- 对所有算法同时报贪心部署值、保持探索的行为值、覆盖和风险；
- 若 $\epsilon$ 按 GLIE 衰减，再讨论三者是否趋向相同贪心极限。

### 12.2 关键不变量汇总

```python
assert np.allclose(HIDDEN_P.sum(axis=2), 1.0)
assert np.all(HIDDEN_P >= 0.0)
assert MC_MAE < 1.0
assert TD_MAE < 1.5
assert TARGET_EXAMPLE["sarsa"] < TARGET_EXAMPLE["expected"]
assert TARGET_EXAMPLE["expected"] < TARGET_EXAMPLE["q_learning"]
assert single_q_bias > 25 * abs(double_q_estimate)
assert np.all(DQN_Y[batch_terminated] == batch_rewards[batch_terminated])
assert np.all(DDQN_Y[batch_terminated] == batch_rewards[batch_terminated])

print({
    "mc_mae": round(float(MC_MAE), 4),
    "td_mae": round(float(TD_MAE), 4),
    "single_q_bias": round(float(single_q_bias), 4),
    "double_q_mean": round(float(double_q_estimate), 4),
    "qualification": QUALIFICATION,
})
```

若断言失败，先定位语义而不是放宽阈值：

- MC 偏差大：检查回报方向、折扣指数和截断尾部；
- TD 偏差大：检查终止 mask、步长、状态索引和奖励错位；
- Q-learning 偏差大：检查 `max` 是否只在合法动作上、探索覆盖和平票；
- SARSA 异常：检查 `next_action` 是否真由同一行为策略抽出；
- DQN 终止样本不等于即时奖励：检查 `(1-d)` 广播；
- Double DQN 与 DQN 永远相同：检查是否真的拆开了选择和评价网络。

## 13. 章末工程清单

- [ ] 学习器接口只接收 transition，没有偷读隐藏 $P,R$；
- [ ] 行为概率、策略版本、动作 mask 和环境版本随样本冻结；
- [ ] first/every-visit、MC 截断尾部和奖励时点已定义；
- [ ] TD、n-step 的 bootstrap 边界有单元测试；
- [ ] epsilon-greedy 在所有合法平票动作间按合同分配概率；
- [ ] SARSA 使用实际下一动作，Expected SARSA 使用行为期望，Q-learning 使用合法动作最大值；
- [ ] 步长是在每个状态动作对上计数，不是只看全局 step；
- [ ] 常数 epsilon 的行为值与关闭探索的贪心值分开报告；
- [ ] 访问覆盖、多随机种子、风险尾部和失败率进入验收；
- [ ] Double Q 的选择与评价分支没有误用同一表；
- [ ] replay、目标网络和 stop-gradient 各自有职责测试；
- [ ] 纯离线数据没有被当作“任意 off-policy 外推许可”；
- [ ] 随机探索只在合格仿真/隔离环境，发布物不接设备控制链。

## 14. 与下一章的接口

本章通过 $Q$ 间接得到策略：先估计动作价值，再做 `argmax` 或 $\epsilon$-greedy。连续动作、高维策略或随机策略优化中，`max_a Q(s,a)` 可能难算，也可能需要直接表达动作分布。

第 32 章将把策略参数 $\theta$ 直接放进目标：

$$
\max_\theta J(\theta)
=\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_t\gamma^tR_{t+1}\right],
$$

从 REINFORCE 的 Monte Carlo policy gradient 出发，再用本章 TD error 训练 critic，形成 Actor–Critic。接口上将继续复用：

- 行为/目标策略与日志概率；
- reward-to-go、baseline 与 n-step；
- terminated/truncated；
- on/off-policy 与数据支持；
- 多随机种子和安全探索边界。

## 15. 章节练习

### 基础推导

1. 从 $G_t=R_{t+1}+\gamma G_{t+1}$ 推导 TD(0) target，并解释为什么 target 的条件期望在真值处等于 $V^\pi$。
2. 对奖励 `[-1, 2, 5]`、$\gamma=0.9$ 和指定 bootstrap 值，手算所有 $n=1,2,3$ target。
3. 证明在给定 $S_{t+1}$ 时，Expected SARSA target 对下一动作抽样的条件方差不大于 SARSA。
4. 写出有多个贪心动作时 $\epsilon$-greedy 的精确概率，比较“固定取第一个平票动作”的覆盖差异。
5. 从 Bellman 期望方程推导 SARSA，从 Bellman 最优方程推导 Q-learning，逐个标出被样本近似和被解析求值的项。
6. 构造一个 $\epsilon>0$ 时 SARSA 与 Q-learning target 不同、$\epsilon=0$ 时相同的数值例子。

### 实现与反例

7. 实现 every-visit MC，与本章 first-visit 版本比较有限样本偏差、方差和每回合更新次数。
8. 给 `train_control` 增加状态依赖合法动作 mask，并写测试证明非法动作抽样概率和 `max` 参与度都为零。
9. 实现在线 n-step SARSA 环形缓冲，覆盖自然终止发生在 $n$ 步窗口内部的情形。
10. 实现 GLIE 调度和逐状态动作步长，画覆盖最小值、greedy value 与 behavior value。
11. 实现 Double Q-learning，验证更新 A 时由 A 选、B 评，更新 B 时角色对调。
12. 注入 `terminated/truncated` 混淆，测量长路线价值偏差并写回归测试。
13. 构造 10 个等价值动作加噪声的实验，画单 Q 最大化偏差随动作数和噪声标准差的变化。

### 工业资格审查

14. 为多 actor 采集系统设计 transition schema，处理 actor 策略滞后、重复样本和环境版本切换。
15. 设计一份探索许可证：明确仿真资格、合法动作 mask 责任方、风险预算、停止条件与人工审批。
16. 为 DQN replay 设计版本过滤、稀有严重事件抽样和训练/资格数据隔离方案。
17. 审计一个离线数据集的行为支持，列出 `max` 可能查询的 OOD action，并提出不上线时的保守处置。
18. 写一页发布评审，分开报告行为策略、目标策略、关闭探索策略以及失败随机种子。

练习骨架：

```python
# TODO：实现 Double Q-learning；更新 A 时必须由 A 选动作、B 评价，反之亦然。
def double_q_update(q_a, q_b, transition, rng):
    raise NotImplementedError
```

## 来源与许可

- Sergey Levine，UC Berkeley Deep Reinforcement Learning Section 3，PDF 第 5–8 页：用于核对 Monte Carlo return、TD bootstrap、stop-gradient、n-step return 与偏差—方差边界。
- Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 7，PDF 第 14–20 页：用于核对 fitted/online Q-learning、off-policy、epsilon-greedy 探索与 replay 入口。
- Sergey Levine，UC Berkeley Deep Reinforcement Learning Section 4，PDF 第 1–3 页：用于核对表格 Q-learning、DQN 的经验回放/目标网络、终止 mask、epsilon-greedy 与 Double DQN。
- Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 8，PDF 第 3–8 页、14–22 页：用于核对 target stop-gradient、经典 DQN、目标网络、最大化过估计、Double Q-learning 和实践稳定化建议。

本章的经典 SARSA 与 Expected SARSA 部分依据 Bellman 期望方程独立推导；本地 PDF 语料没有单独覆盖它们的经典表格算法页，因此不把邻近资料冒充直接来源。晶圆载具样本流、14 状态教学仿真器、行为策略合同、全部 NumPy 代码、18 道即时测验和三项专用网页实验均为本站独立创作，不代表真实晶圆厂、AMR、地图、交通规则、损失量规、安全规范或控制策略。没有复制、改编或再分发参考课件中的图形、案例、数据、代码、题目和实验结果。

完整来源登记见 [课程来源与许可](./sources.md) 。

<MlLessonProgress
  storage-key="31-value-learning"
  lesson-id="31"
  next-href="/courses/machine-learning/32-policy-gradient"
  next-label="下一章：REINFORCE 与 Actor–Critic"
/>
