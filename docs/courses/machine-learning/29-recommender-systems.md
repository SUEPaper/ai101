---
title: 29. 推荐系统怎样从缺失矩阵学习
description: 从矩阵分解、隐式反馈、排序指标与反馈回路理解推荐系统
course: machine-learning
---

<MlLessonHero
  eyebrow="第九模块 · 推荐与强化学习"
  title="推荐系统怎样从缺失矩阵学习"
  subtitle="评分矩阵里的空白不是零分：它可能表示没看见、没兴趣，或系统从未展示。"
  :diagnostics="['能推导带偏置矩阵分解的梯度', '能区分显式与隐式反馈目标', '能设计符合时间与曝光机制的离线评价']"
/>

## 1. 推荐不是普通的完整矩阵回归

设用户集合 $\mathcal U$、物品集合 $\mathcal I$，观测交互

$$
\Omega=\{(u,i):r_{ui}\text{ 被观察到}\}.
$$

矩阵 $\mathbf R$ 极度稀疏。未观察的 $r_{ui}$ 通常不能填成 0：

- 用户可能不知道物品；
- 系统可能没有展示；
- 用户看见但没有行动；
- 行为记录可能丢失。

这叫曝光和选择偏差：我们只在系统历史策略展示过的区域观察反馈。

<MlLab
  kind="recommend"
  title="实验：潜因子怎样补全偏好"
  description="调节潜因子维数与正则强度，观察用户—物品向量的匹配、训练误差和冷启动。"
/>

## 2. 从低秩假设到矩阵分解

假设用户和物品都能嵌入 $k$ 维潜空间：

$$
\hat r_{ui}
=\mu+b_u+c_i+\mathbf p_u^\top\mathbf q_i.
$$

- $\mu$：全局平均；
- $b_u$：用户偏高或偏低打分；
- $c_i$：物品普遍受欢迎程度；
- $\mathbf p_u,\mathbf q_i\in\mathbb R^k$：潜在偏好与属性。

显式评分的正则平方目标：

$$
\begin{aligned}
J
&=\frac12\sum_{(u,i)\in\Omega}
(r_{ui}-\hat r_{ui})^2\\
&\quad+\frac\lambda2
\left(
\sum_u\|\mathbf p_u\|^2
+\sum_i\|\mathbf q_i\|^2
+\sum_ub_u^2+\sum_ic_i^2
\right).
\end{aligned}
$$

定义误差 $e_{ui}=r_{ui}-\hat r_{ui}$。对一次交互：

$$
\frac{\partial J}{\partial\mathbf p_u}
=-e_{ui}\mathbf q_i+\lambda\mathbf p_u,
$$

$$
\frac{\partial J}{\partial\mathbf q_i}
=-e_{ui}\mathbf p_u+\lambda\mathbf q_i,
$$

$$
\frac{\partial J}{\partial b_u}
=-e_{ui}+\lambda b_u,\qquad
\frac{\partial J}{\partial c_i}
=-e_{ui}+\lambda c_i.
$$

SGD 就沿这些梯度更新。训练时必须用更新前的 $\mathbf p_u$ 计算 $\mathbf q_i$ 梯度，或接受明确的顺序更新算法。

::: ml-stop 停下来推一推 1
若 $e_{ui}>0$，说明当前低估评分。忽略正则，$\mathbf p_u$ 沿 $\mathbf q_i$ 的哪个方向移动？点积会怎样变化？
:::

<MlQuiz
  question="矩阵分解中加入用户偏置 b_u 主要表示什么？"
  :options="['某用户整体更倾向打高分或低分', '物品所属类别', '交互发生时间', '潜因子维数']"
  :answer="0"
  explanation="它吸收用户总体评分尺度差异，让潜向量更专注于用户—物品匹配。"
/>

## 3. 交替最小二乘

固定全部物品向量，用户 $u$ 的目标是 Ridge 回归：

$$
\min_{\mathbf p_u}
\frac12\sum_{i\in\Omega_u}
(r_{ui}-\mathbf p_u^\top\mathbf q_i)^2
+\frac\lambda2\|\mathbf p_u\|^2.
$$

把该用户已评分物品堆成 $\mathbf Q_u$，评分为 $\mathbf r_u$：

$$
\boxed{
\mathbf p_u
=(\mathbf Q_u^\top\mathbf Q_u+\lambda\mathbf I)^{-1}
\mathbf Q_u^\top\mathbf r_u
}.
$$

再固定用户向量更新每个物品向量。ALS 两组更新可各自并行，也方便处理加权隐式反馈。

<MlQuiz
  question="矩阵分解的 ALS 为什么每一半步容易求解？"
  :options="['固定一侧潜向量后，另一侧变成带正则的线性最小二乘', '目标自动变凸到联合全局', '不再使用数据', '矩阵没有缺失']"
  :answer="0"
  explanation="目标对 P、Q 联合非凸，但固定一侧后对另一侧是独立的 Ridge 问题。"
/>

## 4. 隐式反馈：信心不等于偏好值

点击、播放和购买只给出正向行为，没有可靠负标签。常构造

$$
p_{ui}=\mathbb I(r_{ui}>0),
\qquad
c_{ui}=1+\alpha r_{ui},
$$

并最小化加权目标：

$$
\sum_{u,i}
c_{ui}(p_{ui}-\mathbf p_u^\top\mathbf q_i)^2
+\lambda(\|\mathbf P\|_F^2+\|\mathbf Q\|_F^2).
$$

$p_{ui}$ 是偏好指示，$c_{ui}$ 是我们对该指示的信心。未交互项不是强负例，只给低权重。

另一条路线直接优化排序。BPR 对用户 $u$ 的正物品 $i$ 与未观察物品 $j$：

$$
\max
\sum_{(u,i,j)}
\log\sigma(\hat s_{ui}-\hat s_{uj})
-\lambda\|\theta\|^2.
$$

它鼓励正物品分数高于采样负物品，但负采样分布会改变学习重点。

::: ml-stop 停下来推一推 2
某用户听了一首歌 100 次。$p_{ui}$ 是否应该是 100？在上面的隐式目标里，次数进入哪个量？
:::

<MlQuiz
  question="隐式反馈加权矩阵分解中，重复行为次数常用于什么？"
  :options="['直接作为偏好标签的任意大值', '提高对正偏好指示的置信权重', '删除该用户', '固定潜因子为次数']"
  :answer="1"
  explanation="偏好通常二值化，次数增加信心；具体设计仍需考虑刷量和曝光。"
/>

## 5. Top-K 排序指标

对用户的推荐列表，Precision@K：

$$
\operatorname{P@K}
=\frac{\#\{\text{前 K 中相关物品}\}}{K}.
$$

Recall@K：

$$
\operatorname{R@K}
=\frac{\#\{\text{前 K 中相关物品}\}}
{\#\{\text{全部相关物品}\}}.
$$

NDCG 让靠前的相关结果贡献更大：

$$
\operatorname{DCG@K}
=\sum_{j=1}^{K}
\frac{2^{rel_j}-1}{\log_2(j+1)},
$$

$$
\operatorname{NDCG@K}
=\frac{\operatorname{DCG@K}}
{\operatorname{IDCG@K}}.
$$

离线指标要定义候选集。只从少量随机负例中排名会显著高估相对全库检索的性能。

<MlQuiz
  question="NDCG 相比 Recall@K 额外关注什么？"
  :options="['相关物品在列表中的位置与分级相关性', '训练时间', '用户总数', '矩阵是否方形']"
  :answer="0"
  explanation="DCG 使用位置折扣并可接收分级相关性，归一化后便于用户间比较。"
/>

## 6. 时间切分与冷启动

随机切分交互可能让未来行为进入训练。更接近上线的方案：

1. 按时间训练早期交互；
2. 用之后交互验证；
3. 为每个用户保留未来物品；
4. 防止同一次会话跨集合泄漏。

矩阵分解无法给从未出现的新用户或新物品学习潜向量。冷启动可利用：

- 用户画像与物品内容；
- 热门和探索策略；
- onboarding 偏好；
- 混合模型与元学习。

::: ml-stop 停下来推一推 3
若随机把同一用户未来评分放进训练集，模型可能通过什么信息间接“预知”该用户在测试时的偏好向量？
:::

<MlQuiz
  question="纯协同矩阵分解为何难以处理全新物品？"
  :options="['新物品没有历史交互，无法可靠估计其潜向量', '新物品一定没有内容', '矩阵分解只能做分类', '评分必须是负数']"
  :answer="0"
  explanation="协同信号来自共同行为；无交互物品需要内容特征、先验或探索来启动。"
/>

## 7. 反馈回路：预测会改变未来数据

推荐系统选择展示内容，展示又决定能观察到哪些点击：

$$
\text{模型}\to\text{曝光}\to\text{行为}\to\text{训练数据}\to\text{模型}.
$$

热门物品因曝光更多获得更多数据，可能继续变热门。离线历史只覆盖旧策略选择过的动作，不能无偏评价全新策略。需要：

- 保留随机探索流量；
- 记录展示概率；
- 使用反事实评价；
- 线上 A/B 实验；
- 监控覆盖、多样性、新颖性和长期满意度。

这也连接到后面的强化学习：推荐动作会改变后续状态和回报，但并非所有推荐任务都需要强化学习。

<MlQuiz
  question="推荐系统中的反馈回路为什么使普通监督学习评价不充分？"
  :options="['模型的展示选择会改变之后可观察的数据分布', '评分矩阵必须完整', '用户不会变化', '离线数据包含所有反事实']"
  :answer="0"
  explanation="行为数据受历史策略干预，未展示物品的反馈通常不可观察，存在选择偏差。"
/>

## 8. 最小 SGD 更新

~~~python
def mf_step(rating, user_vec, item_vec, user_bias, item_bias,
            global_mean, learning_rate, regularization):
    prediction = global_mean + user_bias + item_bias + user_vec @ item_vec
    error = rating - prediction
    old_user = user_vec.copy()

    user_vec += learning_rate * (error * item_vec - regularization * user_vec)
    item_vec += learning_rate * (error * old_user - regularization * item_vec)
    user_bias += learning_rate * (error - regularization * user_bias)
    item_bias += learning_rate * (error - regularization * item_bias)
    return error
~~~

## 9. 动手练习：从评分到 Top-K

1. 对小型评分矩阵实现带偏置矩阵分解；
2. 按用户时间切分最后一次交互；
3. 对比 RMSE 与 Recall@10；
4. 比较全库候选和 100 个随机负例候选；
5. 模拟一个新物品并设计冷启动特征；
6. 报告热门度分组下的覆盖率，检查长尾是否被完全忽略。

<MlLessonProgress
  storage-key="29-recommender-systems"
  next-href="/courses/machine-learning/30-mdp-bellman"
  next-label="下一章：Bellman 方程为什么成立"
/>
