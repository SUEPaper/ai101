---
title: 19. 决策树的划分准则从哪里来
description: 从不纯度下降、平方误差分解与剪枝理解 CART
course: machine-learning
---

<MlLessonHero
  eyebrow="第五模块 · 非参数与集成"
  title="决策树的划分准则从哪里来"
  subtitle="树在每个节点寻找一个问题，让分开后的标签比原来更纯。"
  :diagnostics="['能手算熵、Gini 与信息增益', '能推导回归叶节点的最优常数', '能解释深树高方差与剪枝']"
/>

## 1. 一棵树是一组分段常数规则

每个内部节点提出问题，例如 $x_j\le s$，把区域 $R$ 分成

$$
R_L(j,s)=\{\mathbf x:x_j\le s\},\qquad
R_R(j,s)=\{\mathbf x:x_j>s\}.
$$

叶节点输出常数：分类输出类别分布，回归输出数值均值。因此树的决策边界由轴对齐切分组成，天然能表达特征交互和非线性，又无需标准化。

## 2. 分类不纯度

节点中类别 $k$ 的比例为 $p_k$。三种常见指标：

$$
\text{误分类率}=1-\max_k p_k,
$$

$$
\operatorname{Gini}=1-\sum_{k=1}^{K}p_k^2,
$$

$$
\operatorname{Entropy}=-\sum_{k=1}^{K}p_k\log_2p_k.
$$

Gini 还能解释为：按节点分布随机抽真实类别，再按同一分布随机预测，二者不同的概率：

$$
1-\Pr(\hat y=y)=1-\sum_kp_k^2.
$$

父节点样本数 $N$，左右子节点为 $N_L,N_R$。切分后的加权不纯度为

$$
I_{\rm split}
=\frac{N_L}{N}I(R_L)+\frac{N_R}{N}I(R_R).
$$

不纯度下降

$$
\Delta I
=I(R)-I_{\rm split}
$$

越大越好。用熵时，这就是信息增益。

<MlLab
  kind="tree"
  title="实验：阈值怎样改变不纯度"
  description="拖动切分阈值，观察左右节点的类别比例、加权 Gini 与信息增益。"
/>

::: ml-stop 停下来推一推 1
父节点有 6 个正类、4 个负类。某切分得到左侧 4 正 0 负、右侧 2 正 4 负。手算父、子 Gini 和下降量。
:::

<MlQuiz
  question="一个节点中所有样本都属于同一类别时，Gini 与熵分别是多少？"
  :options="['都是 0', '都是 1', 'Gini 为 1、熵为 0', '取决于样本数']"
  :answer="0"
  explanation="纯节点的类别分布在一个类别上为 1，其余为 0，两种不纯度都为 0。"
/>

## 3. 回归切分来自平方误差

一个叶区域 $R$ 输出常数 $c$，平方损失为

$$
J(c)=\sum_{i:\mathbf x_i\in R}(y_i-c)^2.
$$

求导：

$$
\frac{dJ}{dc}
=-2\sum_{i\in R}(y_i-c)
=-2\sum_{i\in R}y_i+2N_Rc.
$$

令其为 0：

$$
c^\star=\frac1{N_R}\sum_{i\in R}y_i=\bar y_R.
$$

因此回归树叶节点输出均值。候选切分 $(j,s)$ 最小化

$$
\sum_{\mathbf x_i\in R_L}(y_i-\bar y_L)^2
+\sum_{\mathbf x_i\in R_R}(y_i-\bar y_R)^2.
$$

等价地，最大化切分前后的平方误差下降。

<MlQuiz
  question="平方损失回归树的叶节点为什么输出该叶样本均值？"
  :options="['均值让样本数量最小', '均值是常数预测下平方误差的最小点', '均值保证树深为 1', '均值不受离群点影响']"
  :answer="1"
  explanation="对叶常数求导并令零，得到样本均值。均值实际上会受离群点影响。"
/>

## 4. 为什么只能贪心生长

若同时搜索所有树结构，特征、阈值与后续子树组合呈指数爆炸。CART 采用递归贪心：

1. 当前节点枚举候选特征与阈值；
2. 选择当前不纯度下降最大的切分；
3. 在左右子节点重复；
4. 满足停止条件时成为叶节点。

贪心切分不保证全局最优，早期错误无法由后续节点撤销。但算法简单、可扩展，并能被集成方法显著增强。

对连续特征排序后，只需检查相邻不同值的中点。对高基数类别特征，若允许任意子集切分，候选数会爆炸，也容易偏向选择多的特征。

<MlQuiz
  question="决策树通常采用贪心切分的主要原因是什么？"
  :options="['全局搜索所有树结构组合代价极高', '不纯度不能计算', '叶节点不能输出概率', '树只能有两层']"
  :answer="0"
  explanation="树结构的全局组合搜索难以承受，实践中逐节点选择当前最佳切分。"
/>

## 5. 深树为什么高方差

只要继续切分，训练不纯度通常不会增加。叶子足够小甚至能记住训练集。问题是样本轻微变化可能改变早期最佳阈值，继而重构整棵子树。

控制复杂度的方法包括：

- 最大深度；
- 叶节点最小样本数；
- 切分所需最小不纯度下降；
- 最大叶节点数；
- 先长大再剪枝。

代价复杂度剪枝最小化

$$
R_\alpha(T)
=R(T)+\alpha|T_{\rm leaf}|,
$$

其中 $R(T)$ 是叶内训练误差，$|T_{\rm leaf}|$ 是叶数。$\alpha$ 越大，新增叶子的误差收益越难抵消复杂度惩罚。

::: ml-stop 停下来推一推 2
一棵子树比把它合并成单叶少 12 单位训练误差，却多 4 个叶子。$\alpha$ 大于多少时，合并后的目标更小？
:::

<MlQuiz
  question="增大代价复杂度剪枝参数 alpha 通常会怎样？"
  :options="['鼓励更多叶子', '更强地惩罚叶子数，得到更小的树', '改变特征单位', '保证训练误差为零']"
  :answer="1"
  explanation="每个叶子额外增加 alpha 的代价，复杂子树需要更大的误差降低才值得保留。"
/>

## 6. 概率、缺失值与可解释性边界

分类叶的朴素概率为

$$
\hat p_k=\frac{n_k}{n_{\rm leaf}}.
$$

小叶会给出极端的 0 或 1，可用 Laplace 平滑：

$$
\widetilde p_k
=\frac{n_k+\alpha}{n_{\rm leaf}+K\alpha}.
$$

缺失值可在预处理阶段插补、增加缺失指示器，或使用能学习默认方向/替代切分的实现。必须在部署时保持相同规则。

一条路径能给出“这次预测用了哪些条件”，但不等于因果解释；相关特征可互相替代，树的早期切分也可能非常不稳定。

<MlQuiz
  question="单棵树的路径可读，为什么仍不能直接视为因果解释？"
  :options="['树没有特征', '切分基于观测相关性，且相关特征和样本扰动会改变路径', '叶节点没有输出', '因果关系只存在于线性模型']"
  :answer="1"
  explanation="预测规则描述模型如何使用数据，不自动识别干预下的因果效应。"
/>

## 7. 一个透明的最佳阈值搜索

~~~python
import numpy as np

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)

def best_split_one_feature(x, y):
    order = np.argsort(x)
    x, y = x[order], y[order]
    parent = len(y) * gini(y)
    best = None
    for i in range(1, len(y)):
        if x[i] == x[i - 1]:
            continue
        cost = i * gini(y[:i]) + (len(y)-i) * gini(y[i:])
        gain = parent - cost
        if best is None or gain > best[0]:
            best = (gain, (x[i - 1] + x[i]) / 2)
    return best
~~~

真实实现会通过累计计数避免每个阈值重复扫描标签。

::: ml-stop 停下来推一推 3
为什么只在相邻不同特征值的中点检查阈值就够了？同一区间内移动阈值会改变左右样本集合吗？
:::

<MlQuiz
  question="训练集轻微变化可能让单棵深树结构大幅改变，这主要体现了什么？"
  :options="['高偏差', '高方差', '低维数', '概率归一化']"
  :answer="1"
  explanation="模型对样本扰动敏感是高方差表现，也是 Bagging 要解决的核心。"
/>

## 8. 动手练习：手算并验证第一刀

构造 12 个一维二分类样本：

1. 枚举所有相邻中点；
2. 手算每个阈值的加权 Gini；
3. 与代码选择核对；
4. 加入一个标签离群点，观察最佳阈值变化；
5. 比较最大深度 1、3 和不限深度的训练/验证误差。

<MlLessonProgress
  storage-key="19-decision-trees"
  next-href="/courses/machine-learning/20-random-forests"
  next-label="下一章：Bagging 与随机森林为什么稳定"
/>
