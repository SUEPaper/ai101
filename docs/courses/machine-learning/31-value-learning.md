---
title: 31. 没有环境模型怎样学习价值
description: 从 Monte Carlo、TD、SARSA、Q-learning 到 DQN 理解无模型价值学习
course: machine-learning
---

<MlLessonHero
  eyebrow="第九模块 · 推荐与强化学习"
  title="没有环境模型怎样学习价值"
  subtitle="不知道转移概率时，用一次真实转移替代 Bellman 期望；目标带着当前估计，就产生了 TD 学习。"
  :diagnostics="['能区分 Monte Carlo 与 TD 目标', '能推导 Q-learning 和 SARSA 的更新差异', '能解释经验回放、目标网络与 Double DQN']"
/>

## 1. 三种估计思想

固定策略 $\pi$，要估计 $V^\pi(s)$。

**动态规划**使用已知模型的完整期望：

$$
V(s)\leftarrow
\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)
[r+\gamma V(s')].
$$

**Monte Carlo** 等一条轨迹结束，用真实回报：

$$
V(S_t)\leftarrow
V(S_t)+\alpha[G_t-V(S_t)].
$$

**TD(0)** 只走一步，用奖励与下一状态估计：

$$
V(S_t)\leftarrow
V(S_t)+\alpha
\underbrace{
[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]
}_{\delta_t}.
$$

$\delta_t$ 是 TD error。

<MlLab
  kind="qlearning"
  title="实验：探索率与学习率如何改变 Q 学习"
  description="调节 epsilon、alpha 和奖励延迟，观察价值传播、探索覆盖与收敛波动。"
/>

## 2. MC 与 TD 的偏差—方差

Monte Carlo 目标 $G_t$ 在正确采样下对 $V^\pi(S_t)$ 无偏，但包含整条未来随机性，方差高，且通常要等回合终止。

TD 目标

$$
R_{t+1}+\gamma V(S_{t+1})
$$

使用当前估计 bootstrap，早期可能有偏，却只含一步随机性、能在线更新、适用于持续任务。随着学习收敛，目标的自举偏差可消失。

n-step 回报连接二者：

$$
G_t^{(n)}
=R_{t+1}+\gamma R_{t+2}
+\cdots+\gamma^{n-1}R_{t+n}
+\gamma^nV(S_{t+n}).
$$

$n=1$ 是 TD(0)，走到回合末且不 bootstrap 接近 MC。

<MlQuiz
  question="相对 Monte Carlo，TD(0) 的典型特点是什么？"
  :options="['目标方差较低但因 bootstrap 可能有偏，可在线更新', '必须等回合结束且完全无偏', '需要已知转移矩阵', '不能用于持续任务']"
  :answer="0"
  explanation="TD 用一步样本和当前价值估计，不需完整回报，形成偏差—方差交换。"
/>

::: ml-stop 停下来推一推 1
奖励序列为 $1,2,3$，末状态价值为 0，$\gamma=0.9$。计算 $t=0$ 的 1-step、2-step 和完整 MC 目标，设当前 $V(S_1)=5,V(S_2)=4$。
:::

## 3. Q-learning：对 Bellman 最优方程做采样近似

Bellman 最优动作价值：

$$
Q^\star(s,a)
=\mathbb E[
R_{t+1}+\gamma\max_{a'}Q^\star(S_{t+1},a')
\mid S_t=s,A_t=a].
$$

观察一次转移 $(S_t,A_t,R_{t+1},S_{t+1})$，用括号内样本作为目标：

$$
y_t
=R_{t+1}+\gamma(1-d_t)
\max_{a'}Q(S_{t+1},a'),
$$

$d_t=1$ 表示真正终止。更新：

$$
\boxed{
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)+\alpha[y_t-Q(S_t,A_t)]
}.
$$

行为可用 $\epsilon$-greedy：

$$
A_t=
\begin{cases}
\text{随机动作},&\text{概率 }\epsilon,\\
\arg\max_aQ(S_t,a),&\text{概率 }1-\epsilon.
\end{cases}
$$

目标仍使用最大动作，不一定是实际下一动作，所以 Q-learning 是 off-policy。

<MlQuiz
  question="Q-learning 为什么称为 off-policy？"
  :options="['行为可含探索，但更新目标按贪心目标策略取 max', '它没有策略', '只能离线学习', '奖励来自另一个环境']"
  :answer="0"
  explanation="生成数据的行为策略与更新所评估的贪心目标策略可以不同。"
/>

## 4. SARSA：学习实际行为策略的价值

SARSA 使用实际选出的下一动作 $A_{t+1}$：

$$
y_t^{\rm SARSA}
=R_{t+1}+\gamma(1-d_t)
Q(S_{t+1},A_{t+1}),
$$

$$
Q(S_t,A_t)\leftarrow
Q(S_t,A_t)+\alpha
[y_t^{\rm SARSA}-Q(S_t,A_t)].
$$

若行为是 $\epsilon$-greedy，SARSA 评价包含探索风险的策略，是 on-policy。悬崖行走中，Q-learning 可能学到贴崖最短路径；SARSA 会考虑探索时跌落的风险而保持距离。

::: ml-stop 停下来推一推 2
下一状态两个动作 Q 值为 10 和 0，行为因探索选了值 0 的动作。忽略奖励，Q-learning 与 SARSA 的目标分别使用哪个值？
:::

<MlQuiz
  question="SARSA 更新名称中的五元组是什么？"
  :options="['S_t,A_t,R_{t+1},S_{t+1},A_{t+1}', '状态、状态、状态、状态、状态', '参数的五个导数', '五个独立环境']"
  :answer="0"
  explanation="更新使用当前状态动作、奖励、下一状态以及行为实际选择的下一动作。"
/>

## 5. 表格 Q-learning 何时收敛

有限 MDP 中，经典条件包括：

- 每个状态—动作对被访问无穷多次；
- 学习率满足
  $\sum_t\alpha_t=\infty$、
  $\sum_t\alpha_t^2<\infty$；
- 奖励有界；
- 环境平稳；
- 折扣 $\gamma<1$。

固定非零探索率保证覆盖但最终行为不完全贪心；GLIE 策略让所有对持续被访问，同时 $\epsilon_t\to0$。实际有限预算下只能近似这些条件。

<MlQuiz
  question="若 epsilon 从第一步就设为 0，Q-learning 的主要风险是什么？"
  :options="['未探索动作可能永远没有数据，贪心于错误初值', '更新必然无偏', '状态数变为零', '奖励自动最大']"
  :answer="0"
  explanation="纯利用可能锁定早期偶然看起来好的动作，无法满足充分探索。"
/>

## 6. 从表格到 DQN

用网络 $Q_\theta(s,a)$ 逼近动作价值，最小化 TD 平方误差：

$$
L(\theta)
=\mathbb E_{(s,a,r,s',d)\sim\mathcal B}
\left[
\left(
y-Q_\theta(s,a)
\right)^2
\right],
$$

$$
y=r+\gamma(1-d)
\max_{a'}Q_{\bar\theta}(s',a').
$$

两个稳定化组件：

1. **经验回放** $\mathcal B$：打乱相邻转移相关性并重复利用数据；
2. **目标网络** $\bar\theta$：延迟复制在线参数，让回归目标短期相对稳定。

若目标也立即使用正在更新的同一网络，预测和目标同时移动，容易振荡或发散。

<MlQuiz
  question="DQN 使用目标网络的主要目的是什么？"
  :options="['让 TD 目标在一段时间内更稳定', '删除所有旧数据', '保证 Q 值恒为正', '替代探索']"
  :answer="0"
  explanation="延迟更新目标参数，减弱追逐自身快速变化预测的移动目标问题。"
/>

## 7. 最大化偏差与 Double DQN

若各动作估计含零均值噪声，最大值满足一般关系

$$
\mathbb E[\max_a\hat Q_a]
\ge
\max_a\mathbb E[\hat Q_a],
$$

产生过估计。Double DQN 用在线网络选动作、目标网络评价：

$$
a^\star
=\arg\max_aQ_\theta(s',a),
$$

$$
y^{\rm DDQN}
=r+\gamma(1-d)
Q_{\bar\theta}(s',a^\star).
$$

分离选择与评价，降低同一噪声同时主导二者造成的乐观偏差。

::: ml-stop 停下来推一推 3
两个动作真实价值都为 0，估计误差独立对称。为什么取两个估计的最大值期望会大于 0？
:::

<MlQuiz
  question="Double DQN 如何减轻最大化过估计？"
  :options="['在线网络选动作，目标网络评价该动作', '删除 max 并随机预测', '把奖励取负', '只保留一个动作']"
  :answer="0"
  explanation="用不同参数承担选择与评价，减少同一估计噪声的赢家偏差。"
/>

## 8. 终止与截断不能混淆

环境返回可能包含：

- **terminated**：MDP 真正终止，未来价值为 0；
- **truncated**：因时间上限等外部原因截断，真实过程可能继续。

若把截断都当终止，会错误删除 bootstrap 项，低估时间上限附近价值。具体处理要依据状态是否仍有合理后继价值。

## 9. 表格实现骨架

~~~python
import numpy as np

def q_update(q, state, action, reward, next_state,
             terminated, alpha, gamma):
    future = 0.0 if terminated else np.max(q[next_state])
    target = reward + gamma * future
    td_error = target - q[state, action]
    q[state, action] += alpha * td_error
    return td_error
~~~

## 10. 动手练习：悬崖行走

1. 实现 $\epsilon$-greedy 表格 Q-learning 与 SARSA；
2. 用相同随机种子跑 100 次；
3. 画训练回报和最终策略；
4. 比较训练时保持 $\epsilon>0$ 的实际回报；
5. 检查终止与时间截断处理；
6. 报告均值、分位数和失败率，而不只展示最好曲线。

<MlLessonProgress
  storage-key="31-value-learning"
  next-href="/courses/machine-learning/32-policy-gradient"
  next-label="下一章：策略怎样直接沿回报方向更新"
/>
