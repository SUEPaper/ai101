---
title: 26. K-means 为什么交替两步
description: 从簇内平方和推导分配、均值更新、收敛与初始化
course: machine-learning
---

<MlLessonHero
  eyebrow="第七模块 · 降维与聚类"
  title="K-means 为什么交替两步"
  subtitle="固定中心时选最近中心，固定分配时取均值；两步都精确降低同一个目标。"
  :diagnostics="['能从目标函数推导两步更新', '能说明单调下降不等于全局最优', '能识别尺度、离群点和簇形状的限制']"
/>

## 1. 先写目标，而不是先背算法

给定 $K$ 个中心 $\boldsymbol\mu_1,\ldots,\boldsymbol\mu_K$，样本分配 $z_i\in\{1,\ldots,K\}$。K-means 最小化簇内平方和：

$$
J(\mathbf z,\boldsymbol\mu)
=\sum_{i=1}^{n}
\|\mathbf x_i-\boldsymbol\mu_{z_i}\|_2^2.
$$

联合优化离散分配和连续中心是非凸组合问题。Lloyd 算法交替固定一组变量、精确优化另一组。

<MlLab
  kind="kmeans"
  title="实验：分配与中心如何交替收敛"
  description="调节 K 和初始中心，逐步执行 E-like 分配与 M-like 均值更新，观察目标下降。"
/>

## 2. 固定中心：为什么分到最近簇

中心固定时，目标对每个样本独立：

$$
J=\sum_i
\underbrace{
\|\mathbf x_i-\boldsymbol\mu_{z_i}\|_2^2
}_{\text{只由 }z_i\text{ 决定}}.
$$

所以

$$
\boxed{
z_i
=\arg\min_k
\|\mathbf x_i-\boldsymbol\mu_k\|_2^2
}.
$$

这一步生成 Voronoi 区域，欧氏距离下边界是中心连线的垂直平分超平面。

<MlQuiz
  question="K-means 分配步骤为何可逐样本独立选择最近中心？"
  :options="['固定中心后目标是各样本距离项之和，分配之间无耦合', '中心始终相同', '样本必须排序', '距离不能比较']"
  :answer="0"
  explanation="每个 z_i 只出现在自己的平方距离项中，逐项最小即可最小化总和。"
/>

## 3. 固定分配：为什么中心是均值

对簇 $k$ 的目标：

$$
J_k(\boldsymbol\mu_k)
=\sum_{i:z_i=k}
\|\mathbf x_i-\boldsymbol\mu_k\|_2^2.
$$

求梯度：

$$
\nabla_{\boldsymbol\mu_k}J_k
=-2\sum_{i:z_i=k}
(\mathbf x_i-\boldsymbol\mu_k).
$$

令其为 0：

$$
\boxed{
\boldsymbol\mu_k
=\frac1{n_k}\sum_{i:z_i=k}\mathbf x_i
}.
$$

均值是平方欧氏损失的最优代表。若使用绝对距离，对应中心会变成中位数；改变距离不能继续无条件使用均值更新。

::: ml-stop 停下来推一推 1
一维簇样本为 $[0,2,10]$。平方损失最优中心与绝对损失最优中心分别是什么？
:::

<MlQuiz
  question="K-means 的中心更新为何取簇内均值？"
  :options="['均值最小化簇内平方欧氏距离和', '均值一定是数据点', '均值不受异常值影响', '为了让 K 变小']"
  :answer="0"
  explanation="对中心求导并令零得到均值；它会受异常值影响。"
/>

## 4. 为什么目标单调下降

记一次迭代前为 $(\mathbf z^{(t)},\boldsymbol\mu^{(t)})$。

分配步选固定中心下的最优分配：

$$
J(\mathbf z^{(t+1)},\boldsymbol\mu^{(t)})
\le
J(\mathbf z^{(t)},\boldsymbol\mu^{(t)}).
$$

更新步选固定分配下的最优中心：

$$
J(\mathbf z^{(t+1)},\boldsymbol\mu^{(t+1)})
\le
J(\mathbf z^{(t+1)},\boldsymbol\mu^{(t)}).
$$

所以每轮目标不增。分配组合有限，算法最终到达不再变化的局部固定点；但不同初始化可能得到不同局部最优。

<MlQuiz
  question="K-means 每轮目标不增，为什么仍不能保证全局最优？"
  :options="['目标含离散分配且非凸，可能有多个局部固定点', '均值公式错误', '距离必为负', '迭代次数太多']"
  :answer="0"
  explanation="坐标交替只保证相对当前另一组变量最优，初始化会决定落入哪个吸引域。"
/>

## 5. K-means++ 初始化

先随机选一个中心。之后每个样本按

$$
\Pr(\mathbf x_i\text{ 被选})
=\frac{D(\mathbf x_i)^2}
{\sum_jD(\mathbf x_j)^2}
$$

抽为新中心，其中 $D(\mathbf x_i)$ 是它到最近已选中心的距离。远离现有中心的点更可能被选中，使初始中心分散。

实践中仍应多次初始化，保留最终 $J$ 最小者。极端离群点也会因距离平方获得很高选择概率。

::: ml-stop 停下来推一推 2
已有一个中心，三个样本到它的距离为 $1,2,3$。K-means++ 选择概率之比是多少？
:::

<MlQuiz
  question="K-means++ 相比完全随机初始化，核心策略是什么？"
  :options="['更偏向选择远离已有中心的样本', '总选均值', '先使用标签', '让所有中心重合']"
  :answer="0"
  explanation="按最近中心距离平方采样，通常得到更分散的初始覆盖。"
/>

## 6. 失效边界

K-means 隐含偏好近似球形、尺度相近、欧氏可分的簇。它常在以下情况失效：

- 月牙或环形非凸簇；
- 密度和方差差异巨大；
- 类别大小严重不平衡；
- 含离群点；
- 特征单位不可比；
- 高维距离集中；
- 类别是离散语义而非欧氏向量。

空簇会导致均值分母为 0，可重置到远点、拆分大簇或重新启动。重复点和完全对称也要定义平票规则。

<MlQuiz
  question="K-means 对两个同心圆数据表现通常不佳，主要原因是什么？"
  :options="['它的 Voronoi 区域偏向凸区域，难表达环形簇', '圆没有均值', 'K 必须等于 1', '数据不能标准化']"
  :answer="0"
  explanation="最近中心分配产生凸 Voronoi 单元，非凸同心结构与该归纳偏置不匹配。"
/>

## 7. 与球形 GMM 的关系

若 GMM 各分量先验相等、协方差均为 $\sigma^2\mathbf I$，责任度

$$
\gamma_{ik}\propto
\exp\left(
-\frac{\|\mathbf x_i-\boldsymbol\mu_k\|^2}{2\sigma^2}
\right).
$$

当 $\sigma^2\to0$，责任度集中到最近中心，软分配趋于 K-means 硬分配。因此 K-means 可视为特定极限，但它本身不给出一般概率密度或不确定性。

::: ml-stop 停下来推一推 3
当两个中心到样本距离相等时，$\sigma^2\to0$ 的责任度是否唯一趋于某一簇？这对应 K-means 的哪项实现约定？
:::

<MlQuiz
  question="球形等方差 GMM 在方差趋近 0 时，软责任度趋向什么？"
  :options="['均匀分配', '最近中心的硬分配', 'PCA 投影', '随机森林投票']"
  :answer="1"
  explanation="距离差经极小方差放大，最近中心的指数项支配分母。"
/>

## 8. 最小实现

~~~python
import numpy as np

def kmeans(x, centers, max_iter=100):
    centers = centers.copy()
    for _ in range(max_iter):
        distance2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        labels = distance2.argmin(axis=1)
        updated = centers.copy()
        for k in range(len(centers)):
            members = x[labels == k]
            if len(members):
                updated[k] = members.mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return labels, centers
~~~

## 9. 动手练习：记录每一步

1. 对二维数据手选三个初始中心；
2. 每轮保存分配、中心和 $J$；
3. 验证 $J$ 不增加；
4. 从 30 个随机起点运行，画最终 $J$ 分布；
5. 比较随机初始化与 K-means++；
6. 在月牙数据上展示结构性失败，而不靠调参掩盖。

<MlLessonProgress
  storage-key="26-kmeans"
  next-href="/courses/machine-learning/27-clustering-evaluation"
  next-label="下一章：聚类没有标签怎样判断好坏"
/>
