---
title: 30. Bellman 方程为什么成立
description: 从马尔可夫决策过程、回报递推与压缩映射推导 Bellman 方程
course: machine-learning
---

<MlLessonHero
  eyebrow="第九模块 · 推荐与强化学习"
  title="Bellman 方程为什么成立"
  subtitle="未来回报等于眼前奖励加折扣后的剩余回报；Bellman 方程只是把这条恒等式按状态取期望。"
  :diagnostics="['能完整写出 MDP 五元组', '能推导策略价值与最优价值方程', '能用压缩性解释值迭代收敛']"
/>

## 1. 强化学习的对象：MDP

马尔可夫决策过程（MDP）通常写成

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma).
$$

- $\mathcal S$：状态；
- $\mathcal A$：动作；
- $P(s'\mid s,a)$：转移概率；
- $R(s,a,s')$：一步奖励；
- $\gamma\in[0,1)$：折扣因子。

马尔可夫性：

$$
p(s_{t+1}\mid s_0,a_0,\ldots,s_t,a_t)
=p(s_{t+1}\mid s_t,a_t).
$$

它不表示“未来与过去无关”，而是当前状态已经包含预测未来所需的历史信息。若状态遗漏关键历史，问题对学习器而言不再马尔可夫。

<MlLab
  kind="bellman"
  title="实验：折扣与转移如何改变状态价值"
  description="调节奖励、折扣和转移概率，观察即时收益与长期价值的交换。"
/>

<MlQuiz
  question="MDP 的马尔可夫性最准确的表述是？"
  :options="['未来完全不受过去影响', '给定当前状态和动作，下一状态分布不再依赖更早历史', '奖励必须固定', '策略不能随机']"
  :answer="1"
  explanation="过去可以通过当前状态影响未来；关键是当前状态是充分的预测摘要。"
/>

## 2. 回报为何递推

从时刻 $t$ 开始的折扣回报：

$$
G_t
=R_{t+1}+\gamma R_{t+2}
+\gamma^2R_{t+3}+\cdots.
$$

把第一项拆出：

$$
\begin{aligned}
G_t
&=R_{t+1}
+\gamma(R_{t+2}+\gamma R_{t+3}+\cdots)\\
&=R_{t+1}+\gamma G_{t+1}.
\end{aligned}
$$

这条路径级恒等式是全部 Bellman 方程的起点。若奖励有界 $|R_t|\le R_{\max}$ 且 $\gamma<1$：

$$
|G_t|\le\frac{R_{\max}}{1-\gamma}.
$$

折扣既表达时间偏好，也让无限期回报收敛。有限期任务或保证终止的任务也可取 $\gamma=1$。

::: ml-stop 停下来推一推 1
每一步奖励恒为 1，$\gamma=0.9$，无限期回报是多少？若 $\gamma=1$ 又会发生什么？
:::

<MlQuiz
  question="折扣回报 G_t 与 G_{t+1} 的递推关系是什么？"
  :options="['G_t=R_tG_{t+1}', 'G_t=R_{t+1}+gamma G_{t+1}', 'G_t=G_{t+1}/gamma', '二者无关']"
  :answer="1"
  explanation="将回报级数拆出第一项，剩余级数正是 gamma 倍的下一时刻回报。"
/>

## 3. 策略与两个价值函数

随机策略

$$
\pi(a\mid s)=\Pr(A_t=a\mid S_t=s).
$$

状态价值：

$$
V^\pi(s)
=\mathbb E_\pi[G_t\mid S_t=s].
$$

动作价值：

$$
Q^\pi(s,a)
=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

二者关系：

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a).
$$

优势函数：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s),
$$

表示动作 $a$ 相对该状态策略平均水平好多少。按策略取期望为零：

$$
\sum_a\pi(a\mid s)A^\pi(s,a)=0.
$$

<MlQuiz
  question="优势函数 A^pi(s,a)>0 表示什么？"
  :options="['该动作回报必为正', '该动作价值高于策略在该状态的平均价值', '状态一定终止', '动作概率等于 1']"
  :answer="1"
  explanation="优势是 Q-V 的相对量；即使二者都为负，动作仍可能有正优势。"
/>

## 4. Bellman 期望方程

从 $G_t=R_{t+1}+\gamma G_{t+1}$ 出发：

$$
\begin{aligned}
V^\pi(s)
&=\mathbb E_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s]\\
&=\sum_a\pi(a\mid s)
\sum_{s'}P(s'\mid s,a)
\left[
R(s,a,s')+\gamma V^\pi(s')
\right].
\end{aligned}
$$

动作价值：

$$
Q^\pi(s,a)
=\sum_{s'}P(s'\mid s,a)
\left[
R(s,a,s')
+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')
\right].
$$

它们是自洽方程：当前价值由一步奖励和下一状态价值组成。

::: ml-stop 停下来推一推 2
一个状态只有一个动作：以 0.5 概率终止并得 2，以 0.5 概率回到原状态并得 0，$\gamma=0.9$。列方程并解 $V$。
:::

<MlQuiz
  question="Bellman 期望方程中的期望对哪些随机性求？"
  :options="['策略选动作与环境转移', '只对初始参数', '只对奖励单位', '不含任何随机性']"
  :answer="0"
  explanation="在状态价值中，动作由策略分布产生，下一状态由转移分布产生。"
/>

## 5. Bellman 最优方程

最优价值：

$$
V^\star(s)=\max_\pi V^\pi(s),
\qquad
Q^\star(s,a)=\max_\pi Q^\pi(s,a).
$$

若第一步选最优动作，之后也最优：

$$
\boxed{
V^\star(s)
=\max_a\sum_{s'}P(s'\mid s,a)
[R(s,a,s')+\gamma V^\star(s')]
}.
$$

动作价值形式：

$$
\boxed{
Q^\star(s,a)
=\sum_{s'}P(s'\mid s,a)
[R(s,a,s')+\gamma\max_{a'}Q^\star(s',a')]
}.
$$

最优策略可取

$$
\pi^\star(s)\in\arg\max_aQ^\star(s,a).
$$

期望方程对固定策略是线性的；最优方程含 max，一般非线性。

<MlQuiz
  question="Bellman 最优方程与固定策略期望方程的关键差异是什么？"
  :options="['把下一动作按固定策略平均改为选择最大价值动作', '删除奖励', '必须令 gamma=0', '不再使用状态']"
  :answer="0"
  explanation="最优性要求每个后续状态也采用最优动作，因此出现 max。"
/>

## 6. 为什么值迭代会收敛

定义 Bellman 最优算子

$$
(\mathcal TV)(s)
=\max_a\sum_{s'}P(s'\mid s,a)
[R(s,a,s')+\gamma V(s')].
$$

对任意 $V,W$，利用
$|\max_af_a-\max_ag_a|\le\max_a|f_a-g_a|$：

$$
\begin{aligned}
|(\mathcal TV)(s)-(\mathcal TW)(s)|
&\le
\gamma\max_a
\left|
\sum_{s'}P(s'\mid s,a)(V(s')-W(s'))
\right|\\
&\le
\gamma\|V-W\|_\infty.
\end{aligned}
$$

所以

$$
\|\mathcal TV-\mathcal TW\|_\infty
\le\gamma\|V-W\|_\infty.
$$

$\gamma<1$ 时它是压缩映射，有唯一不动点 $V^\star$，迭代

$$
V_{k+1}=\mathcal TV_k
$$

从任意有界初值收敛，误差至多按 $\gamma^k$ 收缩。

::: ml-stop 停下来推一推 3
$\gamma=0.99$ 与 $0.5$ 哪个理论收敛更慢？为何大折扣同时意味着更长规划视野和更差的收缩系数？
:::

<MlQuiz
  question="折扣 MDP 的 Bellman 最优算子在无穷范数下的压缩系数是什么？"
  :options="['0', 'gamma', '1/gamma', '状态数']"
  :answer="1"
  explanation="转移概率加权不放大无穷范数，未来项再乘 gamma。"
/>

## 7. 值迭代与策略迭代

值迭代：

~~~python
def value_iteration(states, actions, transition, reward, gamma, tol):
    value = {s: 0.0 for s in states}
    while True:
        updated = {}
        for s in states:
            candidates = []
            for a in actions[s]:
                q = sum(
                    prob * (reward(s, a, nxt) + gamma * value[nxt])
                    for nxt, prob in transition(s, a)
                )
                candidates.append(q)
            updated[s] = max(candidates)
        if max(abs(updated[s] - value[s]) for s in states) < tol:
            return updated
        value = updated
~~~

策略迭代交替：

1. **策略评价**：求固定 $\pi$ 的 $V^\pi$；
2. **策略改进**：对 $Q^\pi$ 贪心更新策略。

有限 MDP 中，若策略不再变化就达到最优。模型已知时可用动态规划；下一章讨论转移未知、只能采样时如何学习。

<MlQuiz
  question="动态规划求解 MDP 的关键前提通常是什么？"
  :options="['已知或可枚举转移与奖励模型', '没有状态', '只能有一个动作', '奖励必须全为正']"
  :answer="0"
  explanation="Bellman 备份需要对下一状态分布求和；未知模型时需用采样近似。"
/>

## 8. 动手练习：小型 Gridworld

构造 $4\times4$ 网格，设终点奖励 0、每步奖励 -1：

1. 手算终点附近两层价值；
2. 实现策略评价；
3. 实现值迭代并画价值热图和箭头策略；
4. 比较 $\gamma=0.5,0.9,0.99$；
5. 加一个高奖励但高风险动作；
6. 检查状态是否真的包含预测转移所需信息。

<MlLessonProgress
  storage-key="30-mdp-bellman"
  next-href="/courses/machine-learning/31-value-learning"
  next-label="下一章：没有环境模型怎样学习价值"
/>
