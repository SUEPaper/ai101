---
title: 18. 不写参数方程也能学习吗
description: 从距离、局部平均与偏差—方差理解 K 近邻
course: machine-learning
---

<MlLessonHero
  eyebrow="第五模块 · 非参数与集成"
  title="不写参数方程也能学习吗"
  subtitle="KNN 几乎不训练：它把计算推迟到预测时，用附近样本回答局部问题。"
  :diagnostics="['能解释尺度为何会改变邻居', '能用 K 分析偏差—方差', '能正确设计无泄漏的距离模型验证流程']"
/>

## 1. 最近邻就是一种局部估计

给定训练集 $\mathcal D=\{(\mathbf x_i,y_i)\}_{i=1}^n$。对查询点 $\mathbf x$，记按距离最近的 $K$ 个样本索引集合为 $\mathcal N_K(\mathbf x)$。

回归使用局部平均：

$$
\hat f(\mathbf x)
=\frac1K\sum_{i\in\mathcal N_K(\mathbf x)}y_i.
$$

分类使用多数票，也可写成类别后验的局部估计：

$$
\hat p(y=c\mid\mathbf x)
=\frac1K\sum_{i\in\mathcal N_K(\mathbf x)}
\mathbb I(y_i=c),
$$

$$
\hat y(\mathbf x)=\arg\max_c\hat p(y=c\mid\mathbf x).
$$

它没有固定长度的参数向量，而是保留训练数据，因此称为非参数方法。“非参数”不等于没有超参数，$K$、距离和特征变换都决定结果。

<MlLab
  kind="knn"
  title="实验：K 怎样平滑决策边界"
  description="调节邻居数和查询位置，观察局部投票、边界粗糙度与离群点影响。"
/>

## 2. 距离先定义了什么叫相似

常见 Minkowski 距离为

$$
d_p(\mathbf x,\mathbf z)
=\left(\sum_{j=1}^{d}|x_j-z_j|^p\right)^{1/p}.
$$

$p=2$ 是欧氏距离，$p=1$ 是曼哈顿距离。若身高以米计、收入以元计，收入差会支配欧氏距离。标准化

$$
x_{ij}'=\frac{x_{ij}-\mu_j}{s_j}
$$

把各维放到可比尺度；但 $\mu_j,s_j$ 必须只用训练折估计，否则验证数据的信息泄漏进了距离。

更一般的 Mahalanobis 距离

$$
d_M^2(\mathbf x,\mathbf z)
=(\mathbf x-\mathbf z)^\top\mathbf M(\mathbf x-\mathbf z),
\qquad \mathbf M\succeq0
$$

允许不同方向有不同尺度，甚至可以学习 $\mathbf M$。类别、文本、序列则需要与语义匹配的距离，不能默认欧氏距离有意义。

::: ml-stop 停下来推一推 1
两个特征分别是年龄（20–80）和年收入（0–1 000 000）。不缩放时，欧氏最近邻主要由哪个特征决定？这属于算法错误还是表示选择？
:::

<MlQuiz
  question="交叉验证 KNN 时，标准化参数应怎样估计？"
  :options="['用全部数据一次估计', '每一折只用该折训练部分估计', '只用测试集估计', '不论数据尺度都不标准化']"
  :answer="1"
  explanation="预处理属于模型拟合的一部分，必须在每个训练折内部估计，避免验证信息泄漏。"
/>

## 3. K 是偏差—方差旋钮

$K=1$ 时，训练样本通常能被完美记住，局部区域很小：

- 偏差低；
- 方差高；
- 对标签噪声和离群点敏感。

$K$ 增大时平均更多样本：

- 决策边界更平滑；
- 方差降低；
- 过大的邻域会混合不同区域，偏差升高。

回归中若邻域内噪声独立、方差为 $\sigma^2$，仅看噪声平均有

$$
\operatorname{Var}\left(
\frac1K\sum_{i\in\mathcal N_K}\varepsilon_i
\right)
=\frac{\sigma^2}{K}.
$$

但邻域半径也随 $K$ 墠大，引入函数不恒定造成的偏差。因此最优 $K$ 应由验证数据选择。

<MlQuiz
  question="在其他条件相同的理想情形下，增大 K 通常怎样改变 KNN？"
  :options="['偏差下降、方差上升', '偏差上升、方差下降', '偏差方差都必为零', '完全没有影响']"
  :answer="1"
  explanation="更大的局部平均更平滑、更稳定，但可能抹去真实的局部结构。"
/>

## 4. 距离加权与分类细节

距离加权回归可写成

$$
\hat f(\mathbf x)
=\frac{\sum_{i\in\mathcal N_K}w_i(\mathbf x)y_i}
{\sum_{i\in\mathcal N_K}w_i(\mathbf x)},
\qquad
w_i(\mathbf x)=\frac{1}{(d(\mathbf x,\mathbf x_i)+\epsilon)^q}.
$$

它让近点贡献更大。若查询点与训练点距离为 0，应直接处理零距离集合，避免除零。分类还要明确：

- 平票如何打破；
- 类别不平衡时是否使用类别权重；
- 邻域中概率 $0$ 或 $1$ 是否真的可靠；
- 缺失值怎样进入距离。

::: ml-stop 停下来推一推 2
二分类取偶数 K 为什么更容易平票？随机打破平票会给可复现性带来什么问题？
:::

<MlQuiz
  question="距离加权 KNN 中加入很小的 epsilon，主要防止什么？"
  :options="['邻居数变大', '查询点与训练点重合时除以零', '类别数减少', '特征标准化']"
  :answer="1"
  explanation="距离为 0 时反距离权重发散；实际还可对所有零距离样本单独平均。"
/>

## 5. 维数灾难

在 $d$ 维单位超立方体中，要覆盖体积比例 $r$ 的轴对齐小立方体，其边长比例为

$$
\ell=r^{1/d}.
$$

例如只想覆盖 1% 体积，$d=2$ 时 $\ell=0.1$；$d=100$ 时

$$
\ell=0.01^{1/100}\approx0.955.
$$

也就是说，为收集少量邻居，每一维几乎要跨越整个范围，“局部”不再局部。高维中距离还会趋于集中，无关特征不断给距离加噪声。

解决思路不是盲目增加数据而已，还包括特征选择、降维、学习距离、利用领域不变性。

<MlQuiz
  question="高维下 KNN 变难的主要原因之一是什么？"
  :options="['距离一定变为负数', '要包含足够样本的邻域往往不再局部', 'K 只能取 1', '不能计算平均值']"
  :answer="1"
  explanation="空间体积随维度指数增长，有限样本极度稀疏，近邻可能仍然很远。"
/>

## 6. 计算代价与索引

朴素预测一个查询点需要：

$$
O(nd)
$$

计算全部距离，再选择前 $K$ 个。存储为 $O(nd)$。训练几乎只是记忆数据，但预测昂贵。

KD-tree、ball tree 可在低维加速精确查询；高维时剪枝效果退化，常用近似近邻索引。近似查询还引入“召回率—延迟—内存”的系统权衡。

<MlQuiz
  question="朴素 KNN 的主要计算成本发生在哪里？"
  :options="['训练时求矩阵逆', '预测时与大量训练样本计算距离', '标签编码时', '初始化权重时']"
  :answer="1"
  explanation="KNN 把拟合成本推迟到查询阶段，并保留全部或大部分训练数据。"
/>

## 7. 最小实现

~~~python
import numpy as np

def knn_predict(x_train, y_train, query, k):
    distance2 = np.sum((x_train - query) ** 2, axis=1)
    neighbors = np.argpartition(distance2, k - 1)[:k]
    labels, counts = np.unique(y_train[neighbors], return_counts=True)
    return labels[np.argmax(counts)]
~~~

<MlQuiz
  question="np.argpartition 相比完整排序的意义是什么？"
  :options="['只定位前 K 个元素，无需把其余距离完全排序', '自动标准化数据', '学习距离矩阵', '保证类别平衡']"
  :answer="0"
  explanation="只需要近邻集合时，无需支付完整排序的额外开销。"
/>

::: ml-stop 停下来推一推 3
若把同一人的重复记录随机分散到训练折和验证折，KNN 的验证分数会怎样偏移？为什么它对这种泄漏尤其敏感？
:::

## 8. 动手练习：从可视边界到无泄漏验证

在二维双月数据上：

1. 比较 $K=1,5,25$ 的决策区域；
2. 加入一个尺度大 1 000 倍的无关特征；
3. 在每折内部标准化并选择 $K$；
4. 对比错误地全数据标准化的结果；
5. 画验证误差随 $K$ 的曲线，并解释两端。

<MlLessonProgress
  storage-key="18-knn"
  next-href="/courses/machine-learning/19-decision-trees"
  next-label="下一章：决策树的划分准则从哪里来"
/>
