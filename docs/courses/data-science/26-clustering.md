---
course: data-science
title: 聚类
description: 从距离、K-means 目标、初始化、层次聚类与 Silhouette 理解无标签分组的用途和边界。
outline: [2, 3]
---

<DsLessonHero
  number="26"
  title="聚类"
  eyebrow="聚类找到的是所选表示与距离下的分组，不是世界唯一正确的类别"
  duration="建议 120–150 分钟"
  :outcomes="[
    '解释距离与缩放如何定义聚类结果',
    '手工执行 K-means 的分配和质心更新',
    '比较 K-means、层次聚类与 Silhouette 的信息和局限'
  ]"
/>

## 本讲要解决的问题

运营团队没有站点类型标签，希望根据早晚高峰、周末变化和周转率发现运营模式。聚类可以提出候选分组，但这些组是否有业务意义仍需回到地图、设施和调度过程验证。

## 1. 聚类从表示和距离开始

同一批站点，用原始骑行量会按规模分组；改成各时段比例可能按时间结构分组。欧氏距离又会让大尺度列支配结果。

```python
import numpy as np
import pandas as pd

stations = pd.DataFrame({
    "morning": [120, 105, 38, 42, 88, 91, 28, 31],
    "evening": [130, 118, 44, 48, 92, 97, 35, 34],
    "weekend_ratio": [.55, .60, 1.25, 1.18, .82, .79, 1.45, 1.38],
})

X_raw = stations.to_numpy(dtype=float)
X_scaled = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=0)

print("原始标准差：", X_raw.std(axis=0).round(2))
print("缩放后标准差：", X_scaled.std(axis=0).round(2))
```

缩放不是自动正确。如果骑行量绝对规模正是业务重点，就不应无条件抹去它。选择表示就是定义“相似”的含义。

## 2. K-means 最小化簇内平方距离

给定 K 个质心，K-means 交替执行：

1. Assignment：每个样本分给最近质心。
2. Update：每个质心移动到所属样本的均值。

目标函数 inertia 是每个点到所属质心的平方距离之和。

<ClusteringLab />

每一步都不会增加 inertia，所以算法会停止；但它只保证到达局部最低点，不保证全局最好。

## 3. 从零实现 K-means

```python
def assign_clusters(X, centroids):
    squared_distances = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return squared_distances.argmin(axis=1)

def update_centroids(X, labels, k, previous):
    centroids = []
    for cluster in range(k):
        members = X[labels == cluster]
        centroids.append(members.mean(axis=0) if len(members) else previous[cluster])
    return np.vstack(centroids)

def kmeans(X, k, seed=42, max_steps=100):
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(max_steps):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, centroids)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    labels = assign_clusters(X, centroids)
    inertia = np.sum((X - centroids[labels]) ** 2)
    return labels, centroids, inertia

labels, centroids, inertia = kmeans(X_scaled, 3)
print(labels, centroids.round(2), round(inertia, 3))
```

<QuizBlock question="K-means 更新质心时为什么取簇内均值？" :options="['均值最小化簇内平方距离和', '均值总是样本点', '为了增加 inertia', '只因为实现简单']" :answer="0" explanation="固定簇成员后，平方距离和的最优常数中心是各维均值。" />

## 4. 初始化与簇形状是重要限制

不同起点可能收敛到不同局部解。实践中会多次初始化，保留 inertia 最小的结果；K-means++ 让初始质心彼此更分散。

```python
runs = []
for seed in range(20):
    labels, centroids, inertia = kmeans(X_scaled, 3, seed=seed)
    runs.append((inertia, seed, labels))

best_inertia, best_seed, best_labels = min(runs, key=lambda row: row[0])
print("最佳 seed：", best_seed, "inertia：", round(best_inertia, 3))
```

K-means 偏好近似球形、大小和密度相近的簇，对长条、环形、不同密度与异常点表现可能很差。

## 5. 层次聚类保留多尺度结构

凝聚式层次聚类从每个点一个簇开始，反复合并最近的两簇，生成树状图 Dendrogram。切树高度决定最终簇数。

<figure class="ds-figure">
  <div class="ds-dendrogram" aria-label="原创层次聚类树状图">
    <div><span>A</span><span>B</span><i class="low"></i></div>
    <div><span>C</span><span>D</span><i class="mid"></i></div>
    <div><span>E</span><span>F</span><i class="high"></i></div>
    <b>低切割 → 更多小簇</b><b>高切割 → 更少大簇</b>
  </div>
  <figcaption>树状图展示合并顺序和距离，而不只给出一个固定 K。</figcaption>
</figure>

簇间距离的定义称为 linkage：single 看最近点，complete 看最远点，average 看平均距离，Ward 关注合并后平方误差增加。不同 linkage 会产生不同形状偏好。

## 6. Silhouette 比较簇内紧密与簇间分离

对一个点：`a` 是到同簇其他点的平均距离，`b` 是到最近其他簇的平均距离；Silhouette `s=(b-a)/max(a,b)`。

- 接近 1：同簇紧密且远离其他簇。
- 接近 0：位于簇边界。
- 小于 0：可能更接近另一个簇。

```python
def silhouette_values(X, labels):
    X = np.asarray(X, dtype=float)
    distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    values = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        a = distances[index, same].mean() if same.sum() else 0
        other_means = [distances[index, labels == other].mean() for other in np.unique(labels) if other != label]
        b = min(other_means) if other_means else 0
        values.append((b - a) / max(a, b) if max(a, b) else 0)
    return np.asarray(values)

silhouettes = silhouette_values(X_scaled, best_labels)
print(silhouettes.round(3), "mean=", round(silhouettes.mean(), 3))
```

最高 Silhouette 不自动等于最有用的业务分组。它只评价所选距离下的几何分离。

## 7. 聚类后的验证工作

- 稳定性：换样本、初始化或时间窗口，分组是否大致保留。
- 可解释性：各簇在原始变量和地图上有何区别。
- 可行动性：分组是否支持不同调度策略。
- 风险：分组是否成为对人或片区的未经验证标签。
- 漂移：新站点或季节变化后，质心与簇大小是否改变。

<div class="ds-warning"><strong>聚类没有天然真值</strong>“算法发现了三类站点”只表示在当前特征、缩放、距离、K 和初始化下出现三组。不要把编号 0、1、2 当作客观身份。</div>

## 8. 知识检查

<QuizBlock question="标准化特征会改变 K-means 结果吗？" :options="['可能，因为它改变距离中各列权重', '不会', '只改变簇编号', '只在 K=1 时改变']" :answer="0" explanation="欧氏距离直接受列尺度影响，缩放重新定义相似性。" />

<QuizBlock question="K-means 是否保证找到全局最小 inertia？" :options="['保证', '不保证，依赖初始化并可能停在局部解', 'K=2 时总保证', '样本大时保证']" :answer="1" explanation="交替优化单调下降，但目标非凸，可能有多个局部解。" />

<QuizBlock question="层次聚类相对 K-means 的一个特点是什么？" :options="['树状图保留多种切割尺度', '不需要距离', '一定更准确', '只能得到两个簇']" :answer="0" explanation="Dendrogram 展示逐层合并结构，可以事后选择切割高度。" />

<QuizBlock question="Silhouette 接近 -1 通常说明什么？" :options="['该点可能更接近其他簇', '该点非常居中', 'K 一定正确', '没有距离信息']" :answer="0" explanation="负值表示最近其他簇平均距离小于本簇平均距离。" />

## 9. 代码练习：选择候选 K

对 `K=2...5` 各运行 20 个初始化，保留最低 inertia 的结果，计算平均 Silhouette。返回比较表，并说明为什么不能只按 inertia 选择 K。

<details><summary>提示</summary>

inertia 随 K 增加不会上升，所以不能直接选最小值；Silhouette 提供一个分离度参考，但还需业务解释和稳定性。

</details>

<details><summary>参考实现</summary>

```python
comparison = []
for k in range(2, 6):
    candidates = [kmeans(X_scaled, k, seed=seed) for seed in range(20)]
    labels, centroids, inertia = min(candidates, key=lambda result: result[2])
    silhouette = silhouette_values(X_scaled, labels).mean()
    comparison.append({"k": k, "inertia": inertia, "silhouette": silhouette})

comparison = pd.DataFrame(comparison)
print(comparison.round(3))
```

</details>

## 10. 无监督学习模块收束

从第 01 讲开始，我们把问题变成数据；检查、清洗和表达数据；用抽样与概率描述不确定性；训练、评价并约束模型；最后在无标签数据中寻找候选结构。

<div class="ds-flow">
  <div><b>QUESTION</b><span>定义任务与后果</span><small>谁需要什么决策</small></div>
  <div><b>DATA</b><span>理解生成与粒度</span><small>缺失、偏差、连接</small></div>
  <div><b>MODEL</b><span>拟合可重复结构</span><small>损失、优化、验证</small></div>
  <div><b>UNCERTAINTY</b><span>量化结论边界</span><small>抽样、Bootstrap</small></div>
  <div><b>DECISION</b><span>沟通、监控、纠错</span><small>指标、公平、反馈</small></div>
</div>

完成 26 讲主线后，学习者已经能说明：数据从哪里来、结论依赖什么、哪里可能失败、谁会受到影响。接下来的图像数据桥接模块把同一套审计方法应用到非表格数据，并为机器学习 CNN 与深度学习计算机视觉建立输入接口。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 26：K-means、层次聚类、Dendrogram、linkage 与 Silhouette 的知识边界。
- 本页站点运营画像案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="26" next-label="V1：数字图像与数组合同" next-href="/courses/data-science/v1-digital-images" />
