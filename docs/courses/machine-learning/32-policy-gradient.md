---
title: 32. 策略怎样直接沿回报方向更新
description: 从轨迹概率、对数导数技巧与基线推导策略梯度和 Actor–Critic
course: machine-learning
---

<MlLessonHero
  eyebrow="第九模块 · 推荐与强化学习"
  title="策略怎样直接沿回报方向更新"
  subtitle="价值学习先估动作价值再取最大；策略梯度直接调整动作概率，让高回报轨迹更可能出现。"
  :diagnostics="['能从轨迹分布推导 REINFORCE', '能证明与动作无关的基线不引入偏差', '能解释 Actor–Critic、熵奖励与 PPO 裁剪']"
/>

## 1. 为什么直接参数化策略

令策略 $\pi_\theta(a\mid s)$ 可微，目标是期望折扣回报：

$$
J(\theta)
=\mathbb E_{\tau\sim p_\theta(\tau)}[R(\tau)].
$$

直接策略方法适合：

- 连续动作，无法枚举 $\arg\max_aQ(s,a)$；
- 需要随机策略；
- 动作分布有明确结构；
- 策略梯度比价值最大化更平滑。

代价是轨迹梯度方差较高，且策略数据分布会随参数变化。

<MlLab
  kind="policy"
  title="实验：策略概率如何沿回报更新"
  description="调节奖励差、基线与熵系数，观察动作概率、梯度方差和探索塌缩。"
/>

## 2. 轨迹概率

轨迹

$$
\tau=(s_0,a_0,r_1,s_1,\ldots,s_T)
$$

在策略下的概率：

$$
p_\theta(\tau)
=\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)
P(s_{t+1}\mid s_t,a_t).
$$

环境初始分布 $\rho_0$ 和转移 $P$ 不依赖 $\theta$。取对数：

$$
\log p_\theta(\tau)
=\log\rho_0(s_0)
+\sum_t\log\pi_\theta(a_t\mid s_t)
+\sum_t\log P(s_{t+1}\mid s_t,a_t).
$$

因此

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_t\nabla_\theta
\log\pi_\theta(a_t\mid s_t).
$$

未知环境动力学在梯度中消失，这是无模型策略梯度的关键。

<MlQuiz
  question="策略梯度推导中为什么不需要对环境转移概率求导？"
  :options="['假设环境转移不依赖策略参数 theta', '环境没有状态', '转移概率恒为 1', '奖励等于零']"
  :answer="0"
  explanation="轨迹 log 概率中只有策略项依赖 theta，环境项对 theta 的梯度为零。"
/>

## 3. 对数导数技巧推导 REINFORCE

从

$$
J(\theta)=\sum_\tau p_\theta(\tau)R(\tau)
$$

求导：

$$
\begin{aligned}
\nabla_\theta J
&=\sum_\tau
\nabla_\theta p_\theta(\tau)R(\tau)\\
&=\sum_\tau
p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)R(\tau)\\
&=\mathbb E_{\tau\sim p_\theta}
\left[
R(\tau)\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
\end{aligned}
$$

第二行用了

$$
\nabla p=p\nabla\log p.
$$

一次轨迹给出无偏 Monte Carlo 估计。高回报轨迹中的动作 log 概率被提高，低回报轨迹的动作概率被压低。

<MlQuiz
  question="REINFORCE 使用 log 概率梯度的原因是什么？"
  :options="['对数导数技巧把轨迹概率梯度写成可采样期望', '概率不能求导', '只为数值显示', '让奖励自动归一化']"
  :answer="0"
  explanation="∇p=p∇log p 使求和变成对当前策略轨迹分布的期望。"
/>

## 4. 因果性：过去动作不该为过去奖励负责

完整回报 $R(\tau)$ 可替换为从时刻 $t$ 开始的 reward-to-go：

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_{k+1}.
$$

早于 $t$ 的奖励不受 $a_t$ 影响，其期望梯度为零：

$$
\mathbb E[
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
b(S_t)
]=0
$$

对任意不依赖当前动作的 $b(S_t)$ 都成立。于是

$$
\boxed{
\nabla_\theta J
=\mathbb E
\left[
\sum_t
\gamma^tG_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\right]
}
$$

（折扣因子的精确放置随目标定义约定变化）。使用 reward-to-go 删除与当前动作无关的过去噪声。

::: ml-stop 停下来推一推 1
时刻 5 的动作能否改变时刻 2 已经得到的奖励？为什么仍把整条轨迹总回报乘到它的梯度上会增加方差？
:::

<MlQuiz
  question="用 reward-to-go 替代整条轨迹回报的主要收益是什么？"
  :options="['删除当前动作不可能影响的过去奖励噪声，降低方差', '让梯度恒为零', '获得环境模型', '使动作变离散']"
  :answer="0"
  explanation="因果上动作只影响之后回报，过去奖励作为权重只增添随机波动。"
/>

## 5. 基线为何不引入偏差

从回报减去只依赖状态的基线：

$$
(G_t-b(S_t))
\nabla_\theta\log\pi_\theta(A_t\mid S_t).
$$

条件于状态：

$$
\begin{aligned}
&\mathbb E_{A\sim\pi_\theta}
[b(s)\nabla_\theta\log\pi_\theta(A\mid s)]\\
&=b(s)\sum_a
\pi_\theta(a\mid s)
\frac{\nabla_\theta\pi_\theta(a\mid s)}
{\pi_\theta(a\mid s)}\\
&=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta1=0.
\end{aligned}
$$

所以基线不改变期望梯度，却能显著降低方差。常取

$$
b(s)=V^\pi(s),
$$

此时 $G_t-V^\pi(s_t)$ 是优势的样本估计。

::: ml-stop 停下来推一推 2
如果基线也依赖当前动作 $a$，上面的哪一步不再成立？它是否仍能任意相减？
:::

<MlQuiz
  question="为什么状态价值 V(s) 可作为策略梯度基线而不引入偏差？"
  :options="['它不依赖采样的当前动作，基线项的期望 score 为零', '它永远等于回报', '它没有参数', '它使环境确定']"
  :answer="0"
  explanation="对动作分布的 log 概率梯度期望为 0，乘以只依赖状态的量仍为 0。"
/>

## 6. Actor–Critic

Actor 是策略 $\pi_\theta$，Critic 估计 $V_\phi$ 或 $Q_\phi$。一步 TD 优势：

$$
\hat A_t
=r_{t+1}+\gamma(1-d_t)V_\phi(s_{t+1})
-V_\phi(s_t).
$$

Actor 更新方向：

$$
\nabla_\theta J
\approx
\frac1B\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\hat A_t.
$$

Critic 最小化 TD 目标误差：

$$
L_V(\phi)
=\frac12
\left[
r_{t+1}+\gamma(1-d_t)
V_{\bar\phi}(s_{t+1})-V_\phi(s_t)
\right]^2.
$$

Critic 的 bootstrap 降低方差但引入估计偏差；好的 Critic 是方差缩减器，不要求在训练早期完全准确。

<MlQuiz
  question="Actor–Critic 中 Critic 的主要作用是什么？"
  :options="['估计价值/优势，为策略更新提供较低方差信号', '直接替代环境', '固定所有动作概率', '生成监督标签']"
  :answer="0"
  explanation="Critic 学习回报基准，Actor 用优势决定提高或降低已采样动作概率。"
/>

## 7. 广义优势估计

定义 TD 残差

$$
\delta_t
=r_{t+1}+\gamma V(s_{t+1})-V(s_t).
$$

GAE：

$$
\hat A_t^{\rm GAE(\gamma,\lambda)}
=\sum_{l=0}^{\infty}
(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda$ 小更依赖短步 bootstrap，方差低、偏差可能高；$\lambda$ 接近 1 更接近 Monte Carlo reward-to-go，偏差低、方差高。

<MlQuiz
  question="GAE 中增大 lambda 通常产生什么交换？"
  :options="['更接近长回报，偏差降低而方差上升', '偏差方差都必为零', '删除折扣', '不再需要价值函数']"
  :answer="0"
  explanation="lambda 控制多步 TD 残差的衰减，连接短步 bootstrap 与长程回报。"
/>

## 8. 熵奖励防止过早塌缩

策略熵

$$
\mathcal H(\pi(\cdot\mid s))
=-\sum_a\pi(a\mid s)\log\pi(a\mid s).
$$

优化时加入

$$
J_{\rm total}
=J_{\rm return}
+\beta\mathbb E[\mathcal H(\pi(\cdot\mid S))].
$$

$\beta>0$ 鼓励分布保持分散，改善探索。过大则策略长期随机，过小可能过早锁定次优动作。连续动作常用高斯策略，网络输出均值与对数标准差。

## 9. PPO 的保守更新直觉

用旧策略收集数据，概率比

$$
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\rm old}}(a_t\mid s_t)}.
$$

朴素代理目标为 $r_t\hat A_t$。PPO 裁剪目标：

$$
L^{\rm clip}
=\mathbb E\left[
\min\left(
r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

当更新把有利动作概率推得过远，或把不利动作概率压得过远时，裁剪限制额外收益，形成保守更新压力。它不是严格保证每次策略都改善，仍需监控 KL、回报与数值稳定性。

::: ml-stop 停下来推一推 3
若 $\hat A_t>0$ 且概率比从 1 增到 2，裁剪上界为 1.2，代理目标从何处开始不再奖励继续增大？
:::

<MlQuiz
  question="PPO 裁剪概率比的主要意图是什么？"
  :options="['限制单批数据上的策略更新幅度', '保证环境确定', '把连续动作变离散', '令优势恒为正']"
  :answer="0"
  explanation="旧策略数据只在新旧策略接近时可靠，裁剪抑制过大的概率变化。"
/>

## 10. 最小离散策略梯度

~~~python
import numpy as np

def policy_loss(log_probs, returns, baseline):
    advantage = returns - baseline
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    return -np.mean(log_probs * advantage)
~~~

真实自动微分实现中 <code>log_probs</code> 必须来自当前策略计算图；采样动作本身不可导并不妨碍 score-function 梯度。

<MlQuiz
  question="离散动作采样不可导，REINFORCE 为什么仍可训练？"
  :options="['它对采样结果求普通导数', '它使用动作 log 概率的 score-function 梯度估计期望导数', '它忽略策略参数', '它把动作当连续值']"
  :answer="1"
  explanation="对数导数技巧绕过对随机采样操作本身求导，梯度落在可微的 log 概率上。"
/>

## 11. 强化学习实验的最低证据

强化学习回报方差大，单条最好曲线没有说服力。至少报告：

- 多个随机种子的均值、置信区间或分位数；
- 训练环境步数而非仅迭代数；
- 训练回报与独立评价回报；
- 成功率、失败类型和约束违反；
- 超参数选择所用预算；
- 与随机、启发式和简单价值方法的基线；
- 终止、截断、奖励缩放的处理。

## 12. 动手练习：两臂 Bandit 到 Actor–Critic

1. 用一个 logit 参数表示两动作 Softmax 策略；
2. 手推并实现 REINFORCE；
3. 比较无基线、均值基线和学习价值基线的梯度方差；
4. 加熵奖励并观察探索；
5. 扩展到短链 MDP，实现一步 Actor–Critic；
6. 用至少 20 个随机种子报告学习曲线；
7. 解释为何在线策略改变后，旧轨迹不能无限重复当作同分布数据。

<MlLessonProgress
  storage-key="32-policy-gradient"
  next-href="/courses/machine-learning/assignments"
  next-label="下一站：八次综合作业"
/>
