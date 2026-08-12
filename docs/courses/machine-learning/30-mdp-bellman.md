---
title: 马尔可夫决策过程与动态规划
description: 从晶圆载具转运走廊的随机路径规划出发，推导 MDP、Bellman 期望与最优方程、策略评价、策略迭代、价值迭代和 residual 证书
course: machine-learning
edition: industrial
---

<MlLessonHero
  number="30"
  title="马尔可夫决策过程与动态规划"
  question="一次动作会改变下一时刻可见的选择，环境又带有随机性：怎样把长周期决策压成可审计的一步递推，并在已知模型下算出可复验的策略？"
  duration="建议 720–840 分钟"
  :outcomes="[
    '把工业序贯决策写成包含初始分布、终止语义和可行动作集的有限 MDP 合同，并用条件分布审计状态是否近似 Markov',
    '从路径级回报恒等式推导 V、Q、优势函数以及 Bellman 期望方程，区分奖励时点、状态价值和动作价值',
    '把固定策略评价写成线性方程与迭代 backup，解释同步、原地和截断评价的差异',
    '证明策略改进与 Bellman 最优算子的压缩性，手写策略迭代和价值迭代并用 residual 给出后验误差界',
    '识别终止与截断、折扣与有限时域、奖励塑形与硬约束、模型误差与仿真资格之间的工程边界'
  ]"
/>

> 本章把晶圆载具转运写成离线教学规划器，只用于解释已知转移模型下的决策计算。它不得连接真实 AMR、下发路线、修改速度或避障参数，也不得用有限奖励替代门禁、人员隔离、急停、洁净区许可等硬安全约束。

## 贯穿案例：晶圆载具转运走廊

### 1. 先定义决策，而不是先选算法

某教学晶圆厂有一条 `4 × 4` 的载具转运走廊。AMR 从左下角出发，把密封载具送到右上角交付口；中间两格是不可通行隔墙，交付口下方是隔离区。地坪可能打滑或受局部交通影响：请求“北移”时，实际有一定概率向西或向东偏移；撞到边界或隔墙则留在原地。

业务问题不是“预测下一格”，而是：

$$
\text{在每个可决策格位选择一个方向，最大化从该格位开始的期望折扣回报。}
$$

本章采用以下教学量规：

- 普通一步奖励为 $-0.45$，表示时间与调度占用；
- 首次进入交付口奖励为 $+14$；
- 首次进入隔离区奖励为 $-22$；
- 请求方向以 $0.82$ 概率执行，以 $0.09$ 概率分别偏到左右方向；
- 折扣因子 $gamma=0.94$；
- 交付口和隔离区都是吸收终止状态，进入后的后续奖励为零。

这里的数字只定义教学目标，不代表真实生产损失。真实系统首先由路径许可、隔离门、速度包络、碰撞避免和人工放行删除非法动作；优化器只能在剩余合法动作中排序。

<MlQuiz
  kind="决策合同"
  question="为什么不能把‘闯入人员隔离区’只写成一个足够大的负奖励，然后仍允许规划器选择？"
  :options="['因为负奖励不能参与 Bellman 方程', '因为安全禁令应先从可行动作集合中删除，有限惩罚仍可能在异常量纲或模型误差下被交换掉', '因为动态规划只支持正奖励', '因为状态数会变多']"
  :answer="1"
  explanation="硬安全约束定义可行域，奖励只在可行域内表达偏好。把禁令降格为有限代价，会让优化器在折扣、尺度漂移或模型误差下进行本不允许的交换。"
/>

### 2. 数据与版本合同

动态规划不需要历史轨迹就能更新价值，但它需要一个已知模型。每次发布至少冻结：

| 对象 | 本章字段 | 必须回答的问题 |
|---|---|---|
| 状态字典 | `cell_id`、`seal_state`、`mission_phase`、`traffic_mode` | 是否足以预测下一状态与奖励 |
| 动作字典 | `requested_direction`、`legal_mask` | 谁负责删除非法动作 |
| 转移核 | `model_version`、`P[next_state | state, action]` | 概率来自何处、在哪些切片有效 |
| 奖励核 | `reward_version`、`R[state, action, next_state]` | 奖励发生在离开、途中还是落点 |
| 时域 | `gamma` 或 `horizon`、`terminal_reason` | 折扣任务和有限期任务是否混用 |
| 初始分布 | `start_zone`、`mission_type` | 报告的是单点价值还是任务组合价值 |
| 审计键 | `simulator_sha`、`map_sha`、`policy_sha` | 能否重放同一规划结果 |

只有 `P` 的行和为 1 还不够。还要检查状态字典是否把会改变未来的密封告警、载荷类型、班次交通和剩余时限保留下来。

## 1. 从工业合同到 MDP

### 1.1 七个对象比“五元组”更适合工程交接

教科书常写

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma).
$$

工程实现还需要初始分布 $ho_0$、终止集合 $mathcal T$，并允许动作集合依赖状态：

$$
\mathcal M=(\mathcal S,\{\mathcal A(s)\}_{s\in\mathcal S},P,R,\rho_0,\gamma,\mathcal T).
$$

- $mathcal S$ 是决策时刻可见的状态；
- $mathcal A(s)$ 是状态 $s$ 下经过硬约束过滤后的动作；
- $P(s'\mid s,a)$ 是动作后下一状态的条件分布；
- $R(s,a,s')$ 是该次转移产生的期望奖励；
- $ho_0(s)=\Pr(S_0=s)$ 定义任务从哪里开始；
- $gamma$ 定义未来奖励相对当前的权重；
- $mathcal T$ 定义自然终止，而不是采集器恰好停止记录。

若奖励本身随机，可以写联合核 $p(s',r\mid s,a)$；本章为简洁，把它的条件均值记为 $R(s,a,s')$。

### 1.2 用 NumPy 建立唯一转移真相源

下面的 `P[s, a, s_next]` 与 `R[s, a, s_next]` 会贯穿全部算法。终止奖励发生在“进入终点”的那次转移上，终止状态之后自循环且奖励为零。

```python
import numpy as np

ROWS, COLS = 4, 4
WALLS = {(1, 1), (2, 1)}
GOAL, HAZARD = (0, 3), (1, 3)
CELLS = [(r, c) for r in range(ROWS) for c in range(COLS)
         if (r, c) not in WALLS]
INDEX = {cell: i for i, cell in enumerate(CELLS)}
ACTIONS = ["N", "E", "S", "W"]
DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
LEFT = {"N": "W", "E": "N", "S": "E", "W": "S"}
RIGHT = {value: key for key, value in LEFT.items()}

def move(cell, action):
    if cell in {GOAL, HAZARD}:
        return cell
    dr, dc = DELTA[action]
    candidate = (cell[0] + dr, cell[1] + dc)
    return candidate if candidate in INDEX else cell

def build_model(slip=0.18):
    n_states, n_actions = len(CELLS), len(ACTIONS)
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros_like(P)
    for s, cell in enumerate(CELLS):
        if cell in {GOAL, HAZARD}:
            P[s, :, s] = 1.0
            continue
        for a, intended in enumerate(ACTIONS):
            actuals = (intended, LEFT[intended], RIGHT[intended])
            probabilities = (1 - slip, slip / 2, slip / 2)
            for actual, probability in zip(actuals, probabilities):
                next_cell = move(cell, actual)
                ns = INDEX[next_cell]
                reward = (14.0 if next_cell == GOAL
                          else -22.0 if next_cell == HAZARD
                          else -0.45)
                P[s, a, ns] += probability
                R[s, a, ns] = reward
    return P, R

P, R = build_model()
gamma = 0.94
assert P.shape == (14, 4, 14)
assert np.allclose(P.sum(axis=2), 1.0)
assert np.all(P >= 0)
print(P.shape, P.sum(axis=2).min(), P.sum(axis=2).max())
```

### 1.3 合并重复落点，不是覆盖概率

在墙边请求“北移”，意图方向和某个偏移方向可能都留在原格。它们是两个随机分支，却指向同一个 `next_state`，所以概率必须用 `+=` 聚合。奖励若只由落点决定，重复写同一个奖励无害；若奖励还依赖“实际执行方向”，就必须先定义联合分支再求条件期望，不能用三维 `R` 草率覆盖。

```python
s = INDEX[(2, 2)]
a = ACTIONS.index("N")
nonzero = np.flatnonzero(P[s, a])
audit = [(CELLS[ns], round(P[s, a, ns], 3), R[s, a, ns]) for ns in nonzero]
assert np.isclose(sum(item[1] for item in audit), 1.0)
print(audit)
```

<MlQuiz
  kind="转移核"
  question="两个物理分支都落回同一个状态时，P[s,a,s'] 应如何处理？"
  :options="['保留最后一个分支', '取两个概率的平均', '把指向同一 s′ 的概率相加', '随机删除一个分支']"
  :answer="2"
  explanation="状态转移核记录的是落点事件的总概率；互斥分支指向同一落点时必须求和，否则整行不再表示完整概率分布。"
/>

## 2. Markov 性是状态设计要求，不是世界假设

### 2.1 条件独立的正确读法

记完整历史为

$$
H_t=(S_0,A_0,R_1,\ldots,S_t).
$$

Markov 条件是

$$
p(S_{t+1},R_{t+1}\mid H_t,A_t)
=p(S_{t+1},R_{t+1}\mid S_t,A_t).
$$

它不是说过去没有影响，而是说过去对未来的相关信息已经被 $S_t$ 总结。若把状态只写成格位 `C2`，却漏掉会改变隔离概率的“密封告警已锁存”，两段历史会被错误合并，位置状态就不是充分摘要。

<MarkovStateAuditConsoleLab />

### 2.2 怎样审计状态别名

对候选状态编码 $phi(H_t)$，可以在同一动作下比较历史切片的下一状态分布：

$$
\Delta(a)=
\max_{h_i,h_j:\,\phi(h_i)=\phi(h_j)}
D\!\left(
\widehat p(S_{t+1}\mid h_i,a),
\widehat p(S_{t+1}\mid h_j,a)
\right).
$$

$D$ 可以是总变差距离、Jensen–Shannon divergence，或对关键事件概率的绝对差。工业上还要：

1. 先限制到有共同动作支持的历史，避免把旧策略选择偏差误判成非 Markov；
2. 按载荷、班次、地图版本和传感器健康切片；
3. 对差异给置信区间，不用小样本“证明相等”；
4. 发现别名后补状态字段、转为 belief state，或缩短模型有效期；
5. 把“未发现反例”写成审计结论，不写成严格证明。

若真实状态不可见，只收到观测 $O_t$，问题更接近 POMDP。此时用最近 $k$ 帧拼接只是启发式；严格做法是维护 belief：

$$
b_t(s)=\Pr(S_t=s\mid O_{0:t},A_{0:t-1}).
$$

### 2.3 Markov 与平稳不是一回事

Markov 性讨论“给定现在后是否还需要更早历史”；平稳性讨论转移核是否随时间变化：

$$
P_t(s'\mid s,a)=P(s'\mid s,a).
$$

换班、地图封控、轮胎磨损和交通规则更新都可能破坏平稳性，即使每个时段内部仍是 Markov。可以把班次或地图版本并入状态，也可以建立分时模型；不能默默把漂移平均成一个永远有效的核。

<MlQuiz
  kind="状态充分性"
  question="两段历史编码成同一个状态、执行同一动作，但下一步进入隔离区的概率显著不同。最直接的结论是什么？"
  :options="['奖励函数一定错误', '当前状态编码存在别名，未总结影响未来的历史信息', '策略必须是确定性的', '折扣因子必须设为 1']"
  :answer="1"
  explanation="同状态同动作却保留历史依赖，正是状态摘要不充分的证据；先修状态或观测模型，再讨论求解器。"
/>

<MlQuiz
  kind="概念辨析"
  question="把地图版本加入状态主要是在处理什么问题？"
  :options="['让所有奖励变成正数', '让不同版本的转移规律在扩展状态中可区分', '保证策略必然随机', '消除所有部分可观测性']"
  :answer="1"
  explanation="时间变化的核可通过上下文字段被条件化；这可能恢复扩展状态下的平稳近似，但不会自动补齐所有隐藏状态。"
/>

::: warning 停下来推一推 1
<!-- ml-stop -->
两个任务都在 `C2`，一个剩余 5 秒、一个剩余 50 秒；超时后任务强制结束。若状态只有格位，有限时域最优策略能否保持平稳？写出最少需要补入的状态字段。
:::

## 3. 轨迹、奖励时点与折扣回报

### 3.1 一条轨迹不是一个独立同分布样本表

在策略 $pi$ 与环境核 $P$ 下，轨迹为

$$
\tau=(S_0,A_0,R_1,S_1,A_1,R_2,\ldots).
$$

它的概率分解是

$$
p_\pi(\tau)
=\rho_0(s_0)
\prod_{t\ge 0}
\pi(a_t\mid s_t)
p(s_{t+1},r_{t+1}\mid s_t,a_t).
$$

相邻记录由状态递推相连，不能把每一行当作 iid 监督学习样本。第 31 章会处理未知模型下如何从轨迹采样学习；本章假设 $P,R$ 已经通过独立资格审查。

### 3.2 路径级恒等式先于 Bellman 方程

从时刻 $t$ 开始的折扣回报为

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

拆出第一项：

$$
\boxed{G_t=R_{t+1}+\gamma G_{t+1}}.
$$

这是对每条路径都成立的代数恒等式，还没有取期望，也没有最优化。若 $|R_t|\le R_{\max}$ 且 $gamma<1$，则

$$
|G_t|\le \frac{R_{\max}}{1-\gamma}.
$$

$\gamma$ 有三种常见解释：时间偏好、任务继续概率，或无限和的数值正则。它不是“每一步固定损失 6%”的物理定律。若时步长度从 1 秒改成 10 秒，要按时间常数重算折扣，而不能保留同一个数字。

### 3.3 用模拟轨迹核对奖励时点

```python
rng = np.random.default_rng(30)
start = INDEX[(3, 0)]
terminal_indices = {INDEX[GOAL], INDEX[HAZARD]}

def simulate(policy, episodes=2000, max_steps=200):
    returns = []
    terminal_counts = {"goal": 0, "hazard": 0, "truncated": 0}
    for _ in range(episodes):
        s, discount, total = start, 1.0, 0.0
        for _ in range(max_steps):
            a = int(policy[s])
            ns = rng.choice(len(CELLS), p=P[s, a])
            reward = R[s, a, ns]
            total += discount * reward
            discount *= gamma
            s = ns
            if s in terminal_indices:
                key = "goal" if s == INDEX[GOAL] else "hazard"
                terminal_counts[key] += 1
                break
        else:
            terminal_counts["truncated"] += 1
        returns.append(total)
    return np.asarray(returns), terminal_counts

always_north = np.full(len(CELLS), ACTIONS.index("N"), dtype=int)
sample_returns, counts = simulate(always_north)
assert sum(counts.values()) == len(sample_returns)
print(sample_returns.mean(), counts)
```

如果错误地让终止状态每步继续发放 $+14$，交付价值会被放大成约 $14/(1-\gamma)$，算法会优化一个完全不同的业务合同。奖励落在转移上还是状态上不是排版差异，而是必须版本化的语义。

<MlQuiz
  kind="回报递推"
  question="路径级恒等式 G_t=R_{t+1}+γG_{t+1} 本身是否依赖 Markov 性？"
  :options="['依赖，非 Markov 时不成立', '不依赖；它只是把折扣级数拆出第一项', '只在确定性环境成立', '只在最优策略下成立']"
  :answer="1"
  explanation="递推是代数恒等式；Markov 性在把条件期望压到当前状态、得到价值函数方程时才关键。"
/>

## 4. 策略、占用与三种价值

### 4.1 策略定义动作分布

平稳随机策略写为

$$
\pi(a\mid s)=\Pr(A_t=a\mid S_t=s),
\qquad \sum_{a\in\mathcal A(s)}\pi(a\mid s)=1.
$$

确定性策略是特例，可写成 $a=\pi(s)$。在有限时域任务中，最优策略通常依赖剩余时间，写成 $\pi_t(a\mid s)$；把时间 $t$ 或剩余步数并入状态后，又可转成扩展状态上的平稳策略。

部署指标不能只报起点价值。折扣状态占用

$$
d_\rho^\pi(s)
=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t
\Pr_\pi(S_t=s\mid S_0\sim\rho)
$$

说明策略实际会把质量放到哪些状态。两个平均回报相近的策略，可能一个频繁接近隔离区，另一个稳定走长路；切片占用会暴露这种差异。

### 4.2 $V^\pi$、$Q^\pi$ 与优势

状态价值：

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s].
$$

动作价值：

$$
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

二者关系：

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a).
$$

优势函数：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s),
\qquad \sum_a\pi(a\mid s)A^\pi(s,a)=0.
$$

$A^\pi>0$ 只表示“比当前策略在该状态的平均动作好”，不表示奖励为正，也不表示动作满足外部安全约束。

<MlQuiz
  kind="策略语义"
  question="有限时域任务中，同一物理格位在剩余 2 步和剩余 20 步可能需要不同动作。怎样仍使用平稳策略记号？"
  :options="['删除时域限制', '把剩余步数并入状态', '强制 gamma=0', '把策略改成价值函数']"
  :answer="1"
  explanation="扩展状态 (s, remaining_steps) 能保留决策所需的时间信息，平稳策略在扩展状态上仍可表达时变行为。"
/>

<MlQuiz
  kind="价值辨析"
  question="若 A^π(s,a)>0，能推出什么？"
  :options="['该动作的一步奖励必为正', '该动作价值高于当前策略在 s 的平均价值', '该动作一定是全局最优', '该动作一定安全']"
  :answer="1"
  explanation="优势是相对当前策略基线的量；它不提供绝对回报、全局最优或安全可行性保证。"
/>

## 5. Bellman 期望方程：固定策略的一步分解

### 5.1 从路径递推到条件期望

对 $G_t=R_{t+1}+\gamma G_{t+1}$ 在 $S_t=s$ 条件下取期望：

$$
\begin{aligned}
V^\pi(s)
&=\mathbb E_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s]\\
&=\sum_a\pi(a\mid s)
\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma V^\pi(s')\right].
\end{aligned}
$$

定义对任意候选函数 $V$ 的一步动作值：

$$
Q_V(s,a)=\sum_{s'}P(s'\mid s,a)
[R(s,a,s')+\gamma V(s')].
$$

那么固定策略 Bellman 算子是

$$
(T^\pi V)(s)=\sum_a\pi(a\mid s)Q_V(s,a),
$$

而真实策略价值是它的固定点：

$$
V^\pi=T^\pi V^\pi.
$$

动作价值形式为

$$
Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\right].
$$

<BellmanBackupSwitchboardLab />

### 5.2 backup 的四层账本

一次状态 backup 必须按以下顺序：

1. 对请求动作枚举随机实际方向；
2. 映射到落点 $s'$ 并读取转移奖励；
3. 形成分支目标 $R+\gamma V(s')$；
4. 先按 $P$ 对落点求期望，再按固定策略求和，或在最优算子中对动作取最大。

把 `max` 移到转移期望里面会假设动作能在随机落点揭晓后重选，产生不可能的信息优势：

$$
\max_a\mathbb E_{S'}[X(a,S')]
\ne
\mathbb E_{S'}[\max_a X(a,S')].
$$

<MlQuiz
  kind="Bellman 顺序"
  question="在标准 MDP 中，为什么最优 backup 是 max_a Σ_s′ P(s′|s,a)[r+γV(s′)]，而不是 Σ_s′P(s′|s,a) max_a[r+γV(s′)]？"
  :options="['因为奖励不能求和', '因为动作在随机落点揭晓前选择，不能针对每个已知落点事后重选', '因为 max 只能放在最外层', '因为策略必须随机']"
  :answer="1"
  explanation="决策时的信息集决定运算顺序。先看到落点再选动作会引入标准 MDP 不具备的预知信息。"
/>

<MlQuiz
  kind="奖励时点"
  question="终点奖励定义为‘进入交付口时 +14’。到达后终止状态的自循环奖励应是多少？"
  :options="['每步继续 +14', '0', '-0.45', '由 gamma 决定']"
  :answer="1"
  explanation="奖励已经在进入终点的转移上发放；吸收状态之后奖励为零，避免把一次交付重复计价。"
/>

## 6. 策略评价：线性系统与迭代 backup

### 6.1 矩阵形式

固定策略诱导 Markov 链：

$$
P^\pi_{ss'}=\sum_a\pi(a\mid s)P(s'\mid s,a),
$$

以及一步期望奖励

$$
r^\pi_s=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)R(s,a,s').
$$

于是

$$
V^\pi=r^\pi+\gamma P^\pi V^\pi,
$$

即

$$
\boxed{(I-\gamma P^\pi)V^\pi=r^\pi}.
$$

当有限状态、$gamma<1$ 时，$I-\gamma P^\pi$ 可逆。数值实现应使用 `np.linalg.solve`，不要显式计算逆矩阵。

```python
def q_from_v(P, R, value, gamma):
    return np.sum(P * (R + gamma * value[None, None, :]), axis=2)

def evaluate_policy_exact(P, R, policy, gamma):
    rows = np.arange(len(policy))
    P_pi = P[rows, policy]
    r_pi = np.sum(P_pi * R[rows, policy], axis=1)
    value = np.linalg.solve(np.eye(len(policy)) - gamma * P_pi, r_pi)
    residual = np.max(np.abs(value - (r_pi + gamma * P_pi @ value)))
    return value, residual

v_north, linear_residual = evaluate_policy_exact(P, R, always_north, gamma)
assert linear_residual < 1e-10
print(v_north[start], linear_residual)
```

这也给了模拟器单元测试：前节 Monte Carlo 回报均值应在采样误差内接近 `v_north[start]`。模型求解与轨迹采样若长期不一致，优先检查奖励时点、终止处理和转移行索引。

### 6.2 迭代策略评价

不解线性系统，也可以从任意 $V_0$ 反复应用固定策略算子：

$$
V_{k+1}=T^\pi V_k.
$$

```python
def evaluate_policy_iterative(P, R, policy, gamma, tolerance=1e-10):
    value = np.zeros(len(policy))
    rows = np.arange(len(policy))
    deltas = []
    for _ in range(10000):
        q = q_from_v(P, R, value, gamma)
        new_value = q[rows, policy]
        delta = np.max(np.abs(new_value - value))
        deltas.append(delta)
        value = new_value
        if delta < tolerance:
            break
    return value, np.asarray(deltas)

v_north_iter, policy_deltas = evaluate_policy_iterative(
    P, R, always_north, gamma
)
assert np.max(np.abs(v_north_iter - v_north)) < 2e-9
print(len(policy_deltas), policy_deltas[-1])
```

同步更新使用完整旧向量计算完整新向量，证明和日志最清楚。原地 Gauss–Seidel 更新在扫描到后续状态时立即读取本轮新值，常常更快，但结果依赖扫描顺序；并行实现也更难复现。工业日志必须记录更新语义、状态顺序、容差和最大轮数。

<MlQuiz
  kind="线性评价"
  question="固定策略评价为何能写成线性系统，而 Bellman 最优方程通常不能？"
  :options="['固定策略没有奖励', '固定策略把动作概率固定，方程中不再含对动作的 max', '线性系统只支持两个状态', '最优方程不含转移概率']"
  :answer="1"
  explanation="给定策略后，P^π 和 r^π 都是固定量；最优算子中的 max 会随价值改变活跃动作，使方程分段线性而非单一线性系统。"
/>

<MlQuiz
  kind="实现语义"
  question="同步策略评价与原地评价的主要区别是什么？"
  :options="['同步更新不使用奖励', '原地更新会在同一轮读取已经更新过的状态值，传播与扫描顺序有关', '同步更新只能用于确定性环境', '原地更新一定不收敛']"
  :answer="1"
  explanation="两者固定点相同，但每轮读取快照不同；这会改变收敛轨迹、并行方式和可复现日志。"
/>

::: warning 停下来推一推 2
<!-- ml-stop -->
一个单状态系统每步奖励 1，以概率 1 回到自身，$gamma=0.9$。分别用标量 Bellman 方程、无穷级数和线性系统求 $V$，再说明若终止概率为 0.2，方程中的有效继续系数如何变化。
:::

## 7. 策略改进与策略迭代

### 7.1 一步贪心为何不会变差

已知 $V^\pi$，定义新策略在每个状态选择

$$
\pi'(s)\in\arg\max_a Q^\pi(s,a).
$$

于是

$$
Q^\pi(s,\pi'(s))\ge V^\pi(s).
$$

把右边继续展开：

$$
\begin{aligned}
V^\pi(s)
&\le \mathbb E[R_1+\gamma V^\pi(S_1)\mid s,\pi']\\
&\le \mathbb E[R_1+\gamma R_2+\gamma^2V^\pi(S_2)\mid s,\pi']\\
&\le \cdots \le V^{\pi'}(s).
\end{aligned}
$$

因此 $\pi'$ 不劣于 $\pi$。若所有状态的贪心动作都没有变化，策略已满足最优 Bellman 条件；有限 MDP 的确定性策略数量有限，所以精确策略迭代会在有限次改进后停止。

### 7.2 算法：评价到底，再整体换路

```python
def improve_policy(P, R, value, gamma):
    return np.argmax(q_from_v(P, R, value, gamma), axis=1)

def policy_iteration(P, R, gamma):
    policy = np.full(P.shape[0], ACTIONS.index("N"), dtype=int)
    change_log = []
    for _ in range(100):
        value, _ = evaluate_policy_exact(P, R, policy, gamma)
        improved = improve_policy(P, R, value, gamma)
        changes = int(np.sum(improved != policy))
        change_log.append(changes)
        if changes == 0:
            return value, policy, change_log
        policy = improved
    raise RuntimeError("policy iteration did not converge")

v_pi, pi_star_by_pi, pi_changes = policy_iteration(P, R, gamma)
assert pi_changes[-1] == 0
print(pi_changes, v_pi[start])
```

默认模型从“永远北移”开始，策略变化记录为 `6 → 2 → 0`。策略评价可能做很多便宜的单动作 backup，也可以解线性系统；策略改进每轮要比较全部动作。

改进时的平票需要固定协议。`np.argmax` 会选择第一个最大值，所以动作字典顺序会影响等价策略的字节级结果。发布包应记录动作排序或使用显式平票规则，例如优先保持旧动作以减少路线抖动。

<MlQuiz
  kind="策略改进"
  question="策略迭代中，若对精确 V^π 做一步贪心后所有状态动作都不变，可以得出什么？"
  :options="['策略只在起点最优', '策略满足最优性条件，可停止', '奖励函数必须为零', '还必须随机探索才能判断']"
  :answer="1"
  explanation="固定策略已经对自己的价值函数贪心，策略改进无法再严格提升；有限折扣 MDP 中这正是最优策略条件。"
/>

## 8. Bellman 最优方程与价值迭代

### 8.1 “第一步正确，之后继续正确”

最优状态价值定义为

$$
V^\star(s)=\max_\pi V^\pi(s).
$$

如果第一步选动作 $a$，从下一状态起继续最优，则

$$
Q^\star(s,a)=\sum_{s'}P(s'\mid s,a)
[R(s,a,s')+\gamma V^\star(s')].
$$

第一步也取最优：

$$
\boxed{
V^\star(s)=\max_a\sum_{s'}P(s'\mid s,a)
[R(s,a,s')+\gamma V^\star(s')]
}.
$$

定义最优 Bellman 算子

$$
(TV)(s)=\max_a Q_V(s,a),
$$

则 $V^\star=TV^\star$。从最优价值抽取策略仍需一次模型前瞻：

$$
\pi^\star(s)\in\arg\max_a Q_{V^\star}(s,a).
$$

这说明只存 $V$ 时，决策仍需要 $P,R$；若直接存 $Q^\star$，动作选择只需 `argmax`。第 31 章会利用这个区别在未知模型时直接学习动作价值。

### 8.2 价值迭代

```python
def value_iteration(P, R, gamma, tolerance=1e-11):
    value = np.zeros(P.shape[0])
    residuals = []
    for _ in range(10000):
        new_value = np.max(q_from_v(P, R, value, gamma), axis=1)
        residual = np.max(np.abs(new_value - value))
        residuals.append(residual)
        value = new_value
        if residual < tolerance:
            break
    policy = improve_policy(P, R, value, gamma)
    return value, policy, np.asarray(residuals)

v_star, pi_star_by_vi, vi_residuals = value_iteration(P, R, gamma)
assert np.array_equal(pi_star_by_vi, pi_star_by_pi)
assert np.max(np.abs(v_star - v_pi)) < 1e-8
print(len(vi_residuals), v_star[start], vi_residuals[-1])
```

价值迭代不把中间的贪心动作当作已完成策略，只反复做最优 backup。策略常在价值完全收敛前稳定，但如果没有间隔证书，不能因为连续几轮动作没变就宣布最优。

<DynamicProgrammingPulseGridLab />

<MlQuiz
  kind="最优算子"
  question="只保存 V* 而不保存 Q* 时，从某状态选动作还需要什么？"
  :options="['只看 V*(s) 的正负', '用 P、R 和下一状态价值做一次 one-step lookahead', '重新采样训练集标签', '把 gamma 设为 1']"
  :answer="1"
  explanation="V*(s) 没有按动作分解；必须计算各动作的 ΣP[R+γV*] 才能抽取贪心策略。"
/>

## 9. 为什么价值迭代收敛

### 9.1 上确界范数下的压缩

对任意两个有界价值函数 $U,V$，利用不等式

$$
|\max_a x_a-\max_a y_a|\le \max_a|x_a-y_a|,
$$

有

$$
\begin{aligned}
|(TU)(s)-(TV)(s)|
&\le \max_a\left|
\gamma\sum_{s'}P(s'\mid s,a)(U(s')-V(s'))
\right|\\
&\le \gamma\|U-V\|_\infty.
\end{aligned}
$$

再对 $s$ 取最大：

$$
\boxed{\|TU-TV\|_\infty\le\gamma\|U-V\|_\infty}.
$$

$T$ 是压缩映射，因此：

1. 存在唯一固定点 $V^\star$；
2. 从任意 $V_0$ 出发，$V_{k+1}=TV_k$ 都收敛到它；
3. 先验误差按几何速度衰减：

$$
\|V_k-V^\star\|_\infty
\le \gamma^k\|V_0-V^\star\|_\infty.
$$

### 9.2 数值验证压缩不等式

```python
rng = np.random.default_rng(300)
u, v = rng.normal(size=(2, len(CELLS)))
Tu = np.max(q_from_v(P, R, u, gamma), axis=1)
Tv = np.max(q_from_v(P, R, v, gamma), axis=1)
ratio = np.max(np.abs(Tu - Tv)) / np.max(np.abs(u - v))
assert ratio <= gamma + 1e-12
print(ratio, "<=", gamma)
```

一次随机验证不是证明；它只检查实现没有明显违反理论不变量。证明使用了转移概率非负且行和为 1。若模型张量行和漂移到 1.03，代码仍可能输出数字，但压缩论证已经不适用。

<MlQuiz
  kind="收敛证明"
  question="Bellman 最优算子在上确界范数下的压缩系数是什么？"
  :options="['状态数 |S|', '动作数 |A|', '折扣因子 γ', '最大奖励 Rmax']"
  :answer="2"
  explanation="概率加权不会放大上确界差异，未来价值项只剩 γ 倍；奖励在两个算子的差中抵消。"
/>

### 9.3 residual 是可计算的后验误差证书

真实误差 $\|V-V^\star\|_\infty$ 不能直接算，因为 $V^\star$ 正是未知量；Bellman residual 可以算：

$$
\delta(V)=\|TV-V\|_\infty.
$$

利用三角不等式和压缩性：

$$
\begin{aligned}
\|V-V^\star\|_\infty
&\le \|V-TV\|_\infty+\|TV-TV^\star\|_\infty\\
&\le \delta(V)+\gamma\|V-V^\star\|_\infty,
\end{aligned}
$$

所以

$$
\boxed{
\|V-V^\star\|_\infty
\le \frac{\delta(V)}{1-\gamma}
}.
$$

若想保证价值误差不超过 $epsilon$，停止条件应是

$$
\delta(V)\le(1-\gamma)\epsilon.
$$

```python
def fixed_value_steps(P, R, gamma, steps):
    value = np.zeros(P.shape[0])
    for _ in range(steps):
        value = np.max(q_from_v(P, R, value, gamma), axis=1)
    return value

v_12 = fixed_value_steps(P, R, gamma, 12)
residual_12 = np.max(np.abs(
    np.max(q_from_v(P, R, v_12, gamma), axis=1) - v_12
))
actual_error = np.max(np.abs(v_12 - v_star))
certificate = residual_12 / (1 - gamma)
assert actual_error <= certificate + 1e-12
print(actual_error, certificate)
```

当 $gamma$ 接近 1，$1/(1-\gamma)$ 很大，证书会变松、迭代也变慢。不要把“相邻两轮只差 $10^{-4}$”直接写成“距最优只差 $10^{-4}$”。

<MlQuiz
  kind="停止证书"
  question="γ=0.95，希望保证 ||V−V*||∞≤0.02。按 residual 界，至少应要求 ||TV−V||∞ 不超过多少？"
  :options="['0.02', '0.019', '0.001', '0.4']"
  :answer="2"
  explanation="需要 residual≤(1−γ)ε=0.05×0.02=0.001。相邻迭代差不能不经系数换算就当成价值误差。"
/>

## 10. 有限时域、终止与截断

### 10.1 有限时域用后向归纳

若任务最多剩 $H$ 步，价值依赖时间：

$$
V_t^\star(s)=\max_a\sum_{s'}P_t(s'\mid s,a)
[R_t(s,a,s')+V_{t+1}^\star(s')],
$$

终点条件为 $V_H^\star(s)=g(s)$。此时可以取 $gamma=1$，因为和只有有限项。算法从最后一步往前算，不是让无限时域价值迭代在 $gamma=1$ 下碰运气。

```python
def finite_horizon_dp(P, R, horizon):
    value = np.zeros(P.shape[0])
    policies = []
    for _ in range(horizon):
        q = q_from_v(P, R, value, gamma=1.0)
        policies.append(np.argmax(q, axis=1))
        value = np.max(q, axis=1)
    return value, policies[::-1]

v_h6, policy_by_time = finite_horizon_dp(P, R, horizon=6)
assert len(policy_by_time) == 6
print(v_h6[start], ACTIONS[policy_by_time[0][start]])
```

这里循环变量从“离终点还有 1 步”逐渐扩到 6 步，最后把策略列表反转，得到从任务开始到截止时刻的次序。

### 10.2 `terminated` 与 `truncated` 必须分开

- `terminated=True`：任务自然进入交付、隔离或业务定义的终止状态，后续价值按合同为零；
- `truncated=True`：日志因 200 步上限、仿真超时或采集窗口结束而被截断，真实任务可能继续。

把截断当终止，会系统性低估长任务尾部价值。第 31 章做 TD target 时，这个区别会直接决定是否 bootstrap；本章建立模型时就必须保留 `terminal_reason`。

### 10.3 持续任务的平均奖励边界

有些设备调度没有自然 episode。折扣目标仍可用，但当业务真正关心长期单位时间产出时，平均奖励

$$
\bar r^\pi=\lim_{T\to\infty}\frac1T
\mathbb E_\pi\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

可能更贴合目标。它需要关于链的遍历性和相对价值的新理论，不能把本章 $gamma<1$ 的压缩证明原封不动搬过去。

<MlQuiz
  kind="终止语义"
  question="仿真器因 `max_steps=200` 停止，但任务在真实环境会继续。这条记录应标为什么？"
  :options="['自然终止 terminated', '时间截断 truncated', '吸收奖励', '非法状态']"
  :answer="1"
  explanation="采集窗口结束不等于业务终止；保留截断标记才能决定后续算法是否对尾部价值 bootstrap。"
/>

::: warning 停下来推一推 3
<!-- ml-stop -->
若终点奖励定义为“每次处于交付口都得 +14”，终点又是自循环状态，$gamma=0.94$ 时交付口价值是多少？再把语义改成“首次进入时 +14、之后 0”，比较两个合同。
:::

## 11. 奖励不是安全规范

### 11.1 奖励量纲与可交换性

线性奖励

$$
R=w_1\cdot\text{交付}-w_2\cdot\text{时长}-w_3\cdot\text{隔离}
$$

隐含这些量可以按权重交换。真实工程要先问：

- 哪些是不可交换的硬约束；
- 时间、能耗、质量损失是否已换成可解释共同单位；
- 奖励尺度改变时，折扣与停止阈值是否一起校准；
- 是否存在绕过定义、反复刷奖励或延迟终止的路径；
- 稀有严重事件是否被平均回报掩盖。

可行域过滤、风险预算、人工复核和独立安全控制器不应藏在一个标量奖励里。

### 11.2 potential-based shaping 的有限许可

稀疏终点奖励会让后续采样学习困难，可以加入

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s).
$$

在折扣 MDP、边界条件一致时，这类 potential-based shaping 改变学习信号但保持最优策略集合。随意加“离终点更近就 +1”不一定属于该形式，可能直接改写目标。

```python
phi = np.array([
    0.0 if cell in {GOAL, HAZARD}
    else -abs(cell[0] - GOAL[0]) - abs(cell[1] - GOAL[1])
    for cell in CELLS
])
R_shaped = R + gamma * phi[None, None, :] - phi[:, None, None]
v_shaped, pi_shaped, _ = value_iteration(P, R_shaped, gamma)
assert np.array_equal(pi_shaped, pi_star_by_vi)
print(np.max(np.abs(v_shaped - (v_star - phi))))
```

验证中的价值关系是 $V'_\pi(s)=V_\pi(s)-\Phi(s)$。策略不变不代表 shaped reward 可以替代原业务报表；发布时仍应同时报告原始回报。

<MlQuiz
  kind="奖励边界"
  question="哪种做法最符合本章的工程边界？"
  :options="['把人员隔离禁令设成 -100 后允许策略权衡', '先用硬规则删除非法动作，再在合法动作内优化可交换代价', '只要平均回报高就忽略稀有严重事件', '用 shaped reward 替代原始业务指标']"
  :answer="1"
  explanation="硬约束先定义可行域；奖励只负责可行方案间的偏好。塑形奖励是算法工具，不能改写业务验收口径。"
/>

## 12. 模型误差：动态规划只会精确求解给定模型

### 12.1 估计误差与结构误差

价值迭代 residual 很小，只说明 $V$ 接近当前模型的固定点，不说明当前模型接近真实环境。误差来源包括：

- 稀有动作没有共同支持，转移概率靠外推；
- 状态别名使历史依赖被平均；
- 地图、轮胎和交通策略改变导致核漂移；
- 奖励记录延迟或终止时点错误；
- 仿真器遗漏人员、设备和网络联锁。

因此至少同时报告：数值 residual、模型校准误差、切片覆盖、策略占用范围和仿真到现实偏差。前者不能替代后四者。

### 12.2 对偏移概率做敏感性扫描

```python
sensitivity = []
for slip_value in (0.05, 0.18, 0.35, 0.50):
    P_s, R_s = build_model(slip=slip_value)
    v_s, pi_s, _ = value_iteration(P_s, R_s, gamma)
    sensitivity.append((
        slip_value,
        float(v_s[start]),
        ACTIONS[int(pi_s[INDEX[(2, 2)]])]
    ))
print(sensitivity)
```

若一个动作只在偏移概率精确等于 0.18 时胜出，而在合理置信区间内频繁翻转，就不应发布成“稳定最优”。可选做法包括保守 MDP、分布鲁棒规划、风险敏感目标或把该状态送人工规则；这些方法改变了优化问题，必须单独记录决策依据。

<MlQuiz
  kind="模型资格"
  question="价值迭代的 Bellman residual 已小于 10^-10，能否推出真实 AMR 环境中的策略也近似最优？"
  :options="['能，residual 已足够小', '不能；它只证明对当前 P、R 的数值固定点精度，不能消除模型偏差', '只要状态少于 20 个就能', '只要策略确定就能']"
  :answer="1"
  explanation="求解误差和建模误差是两张证书。动态规划可以高精度地解错模型，必须另做转移核、覆盖和仿真资格审查。"
/>

## 复杂度与工程边界

### 13.1 表格动态规划的成本

若转移张量稠密：

- 一次所有状态动作的 $Q_V$ 计算是 $O(|S|^2|A|)$；
- 一轮价值迭代是 $O(|S|^2|A|)$；
- 一轮迭代策略评价是 $O(|S|^2)$，策略改进是 $O(|S|^2|A|)$；
- 直接解固定策略线性系统通常是 $O(|S|^3)$；
- 存储完整 $P$ 是 $O(|S|^2|A|)$。

局部走廊的每个动作只到少数邻居，用稀疏后继列表可把一次 sweep 降到 $O(E)$，其中 $E$ 是非零转移分支数。不要为 14 个状态引入复杂稀疏框架；当状态扩成格位、载荷、密封、交通、时间的笛卡尔积后，才需要系统评估稀疏矩阵、近似动态规划或模型分解。

### 13.2 数值实现清单

1. `P` 使用非负浮点，逐 `(s,a)` 检查行和；
2. 终止状态的后续奖励与 bootstrap 语义单测；
3. 奖励张量广播维度显式写出，防止把 `R[s,a]` 错当 `R[s,a,s′]`；
4. 同步和原地更新使用不同函数名与日志字段；
5. `argmax` 平票规则固定；
6. 同时报迭代差、Bellman residual 和换算后的误差界；
7. 策略迭代与价值迭代在小模型上交叉核对；
8. 轨迹模拟均值与线性策略评价交叉核对；
9. 对 $gamma$、转移概率和奖励范围做敏感性扫描；
10. 任何模型版本变化都使旧策略证书失效。

### 13.3 发布包与运行边界

一个可审计的离线规划发布包至少包含：

- 状态、动作、终止、截断和奖励时点的 schema；
- `P`、`R`、$ho_0$、$gamma$ 的版本与哈希；
- 非法动作 mask 的外部责任方；
- 模型来源、共同支持和未覆盖切片；
- 求解器版本、更新语义、容差、最大轮数与平票规则；
- 最终价值、策略、residual、误差界和敏感性扫描；
- 仿真轨迹、线性评价与双算法同解的测试报告；
- 明确声明只输出建议，不接入真实控制链。

动态规划适合“模型已知、状态动作有限、目标合同明确”的规划层。状态巨大、模型未知或持续漂移时，应转向采样学习、函数逼近或在线系统辨识；但硬约束、状态审计和奖励语义不会因此消失。

## 数值验证

### 14.1 统一验收：同一模型、三条求解路径

下面把本章关键不变量集中成可执行验收。`v_star` 来自价值迭代，`v_pi` 来自策略迭代，固定策略的模拟均值与线性解则核对奖励时点。

```python
bellman_residual = np.max(np.abs(
    np.max(q_from_v(P, R, v_star, gamma), axis=1) - v_star
))
policy_value_check, policy_residual = evaluate_policy_exact(
    P, R, pi_star_by_vi, gamma
)
mc_standard_error = sample_returns.std(ddof=1) / np.sqrt(len(sample_returns))

assert bellman_residual < 1e-10
assert policy_residual < 1e-10
assert np.max(np.abs(policy_value_check - v_star)) < 1e-8
assert abs(sample_returns.mean() - v_north[start]) < 5 * mc_standard_error
assert np.array_equal(pi_star_by_vi, pi_star_by_pi)
assert np.allclose(P.sum(axis=2), 1.0)

print({
    "states": len(CELLS),
    "value_iterations": len(vi_residuals),
    "policy_changes": pi_changes,
    "start_value": round(float(v_star[start]), 6),
    "bellman_residual": float(bellman_residual),
    "mc_z": float((sample_returns.mean() - v_north[start]) / mc_standard_error),
})
```

默认环境的应有结果是：

- 14 个有效状态、4 个请求动作；
- 价值迭代在严格容差下约 58 轮收敛；
- 策略迭代的变化格数为 `[6, 2, 0]`；
- 两种动态规划得到相同动作表；
- 左下起点最优价值约为 `6.550967`；
- 最终最优 Bellman residual 小于 $10^{-10}$。

这些值既是教材答案，也是回归测试的黄金摘要。若更换地图、奖励、转移核或平票顺序，必须显式更新版本和黄金值，而不是放宽断言掩盖差异。

### 14.2 小模型失效注入

资格测试还应主动制造错误：

- 把某行概率改成 1.02，确认 schema 校验失败；
- 把终点自循环奖励改成 14，确认终点价值异常膨胀；
- 把 `+= probability` 改成覆盖，确认墙边行和失败；
- 把 `max` 放进落点期望内部，确认出现信息泄漏优势；
- 把截断当终止，确认长路线被系统性低估；
- 删除 `seal_state`，确认历史切片的下一状态分布重新分裂；
- 调换动作字典，确认平票策略哈希变化而价值可能不变。

测试目标不是让程序“跑完”，而是让错误的决策合同无法悄悄进入求解器。

## 15. 章末工程清单

在把 MDP 交给下一章的采样学习算法前，逐项签字：

- [ ] 决策时刻、状态字段、动作、终止与截断已定义；
- [ ] 状态别名按共同动作支持与关键切片审计；
- [ ] 转移核非负、归一、版本化并有覆盖说明；
- [ ] 奖励时点、单位、硬约束边界和原始业务指标已冻结；
- [ ] 有限时域与折扣无限时域没有混用；
- [ ] 固定策略的线性解、迭代解和 Monte Carlo 均值相互一致；
- [ ] 策略迭代与价值迭代在小模型上同解；
- [ ] 停止条件由 Bellman residual 换算为误差界；
- [ ] $gamma$、转移与奖励敏感性已报告；
- [ ] residual 证书没有被误写成真实环境最优性证书；
- [ ] 发布物不接入真实设备控制或安全联锁。

## 16. 与下一章的接口

本章依赖完整 $P,R$ 做每一次期望 backup。真实系统往往没有可枚举模型，只能观察

$$
(S_t,A_t,R_{t+1},S_{t+1}).
$$

第 31 章将用样本替代对 $P$ 的精确求和：Monte Carlo 用完整回报，TD 用一步 bootstrap，SARSA 学当前策略的动作价值，Q-learning 用样本近似最优 Bellman backup。无论换成哪一种，以下合同原样保留：

- 状态是否近似 Markov；
- 动作是否经过硬约束过滤；
- 终止与截断是否分开；
- 奖励时点是否唯一；
- target 中的期望、采样和 `max` 顺序是否符合决策信息集。

## 17. 章节练习

### 基础推导

1. 从 $G_t=R_{t+1}+\gamma G_{t+1}$ 完整推导 $V^\pi$ 与 $Q^\pi$ 的 Bellman 期望方程，标明对策略和环境分别求和的位置。
2. 对一个两状态两动作 MDP 写出 $P^\pi,r^\pi$，手算 $(I-\gamma P^\pi)V=r^\pi$。
3. 证明 $\sum_a\pi(a\mid s)A^\pi(s,a)=0$，并给出所有 $Q$ 都为负但某动作优势为正的数值例子。
4. 逐行证明 Bellman 最优算子的上确界压缩性，指出每一步用了什么条件。
5. 从三角不等式推导 $\|V-V^\star\|_\infty\le\|TV-V\|_\infty/(1-\gamma)$。
6. 对有限时域 $H=3$ 的小 MDP 手工执行后向归纳，解释策略为何依赖剩余步数。

### 实现与反例

7. 为本章模型实现稀疏后继列表，与稠密张量逐状态逐动作比较 $Q_V$。
8. 实现原地价值迭代，比较四种状态扫描顺序的迭代数、最终 residual 和策略哈希。
9. 把终点奖励分别实现为进入时奖励和驻留奖励，写回归测试阻止二者混用。
10. 构造一个例子，证明 $\max_a\mathbb E[X_a]\ne\mathbb E[\max_aX_a]$，并解释后者泄露了什么信息。
11. 给策略评价增加截断 sweep，比较 modified policy iteration 与精确策略迭代的外层轮数和总 backup 数。
12. 扫描 $\gamma\in[0.5,0.99]$，报告迭代数、residual 证书和起点最优动作翻转点。

### 工业资格审查

13. 设计 `Markov state audit`：明确共同动作支持、历史切片、距离指标、置信区间和无法证明的部分。
14. 写一份 MDP schema，覆盖状态版本、动作 mask、转移/奖励核、初始分布、终止原因、截断原因和哈希。
15. 将隔离区从负奖励改为硬禁行，比较可行策略集合与最优价值，说明二者不是“惩罚足够大”的等价实现。
16. 给偏移概率建立置信区间，在区间端点重算策略；为动作翻转状态设计保守处置。
17. 写一页发布评审：分别给出数值求解证书、模型资格证书、安全边界和撤回条件。

练习骨架：

```python
# TODO：实现稀疏后继表版本，并逐 (s,a) 与稠密 q_from_v 对齐。
def sparse_q_from_v(successors, value, gamma):
    raise NotImplementedError
```

## 来源与许可

- Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 4，PDF 第 5–7 页：用于核对 Markov chain、MDP、POMDP 与“完全可观测”边界。
- 同课 Lecture 4，PDF 第 14–15 页、23–25 页：用于核对无限/有限时域的期望目标、状态价值、动作价值及其使用方式。
- CMU 10-301/10-601 Lecture 21，*Reinforcement Learning: Markov Decision Processes*，PDF 第 45–62 页：用于核对价值迭代、Bellman 最优方程、策略抽取与每轮复杂度。
- 同课 PDF 第 78–90 页：用于核对固定策略评价、线性系统、策略改进、策略迭代及其与价值迭代的关系。
- Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 7，PDF 第 5–9 页：用于交叉核对动态规划和“评价—改进”策略迭代结构。

本章晶圆载具转运走廊的 14 个有效状态、4 个请求动作、随机偏移模型、奖励量规、密封状态别名审计、全部 NumPy 代码、18 道即时测验和三项专用网页实验均为本站独立创作。它们不代表真实晶圆厂、AMR、洁净区、地图、交通规则、设备能力、损失量规、安全规范或控制策略。没有复制、改编或再分发参考课件中的图形、案例、数据、代码、题目和实验结果。

完整来源登记见 [课程来源与许可](./sources.md) 。

<MlLessonProgress
  storage-key="30-mdp-bellman"
  lesson-id="30"
  next-href="/courses/machine-learning/31-value-learning"
  next-label="下一章：SARSA 与 Q-learning"
/>
