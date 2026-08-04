---
course: data-science
title: 主成分分析：SVD
description: 拆解 X=UΣVᵀ，连接奇异值、主成分、解释方差、低秩重构与 Biplot。
outline: [2, 3]
---

<DsLessonHero
  number="25"
  title="主成分分析：SVD"
  eyebrow="SVD 把一个矩阵拆成正交方向、方向强度和样本坐标"
  duration="建议 120–150 分钟"
  :outcomes="[
    '判断 U、Σ、Vᵀ 的形状与角色',
    '从 SVD 获得 PCA Scores、Loadings 和解释方差',
    '用截断 SVD 构造低秩重构并选择成分数'
  ]"
/>

## 本讲要解决的问题

上一讲通过协方差矩阵特征分解求 PCA。本讲直接分解中心化数据矩阵。SVD 数值上更常用，也揭示“每个主成分是一层 rank-1 图案”的压缩视角。

## 1. SVD 的三块各自做什么

对 `n × p` 矩阵 `X`，经济型 SVD 写为 `X = UΣVᵀ`。令 `r = min(n,p)`：

<div class="ds-matrix-figure">
  <div><b>X</b><span>n × p</span><small>中心化数据</small></div><i>=</i>
  <div><b>U</b><span>n × r</span><small>样本方向</small></div><i>×</i>
  <div><b>Σ</b><span>r × r</span><small>奇异值强度</small></div><i>×</i>
  <div><b>Vᵀ</b><span>r × p</span><small>特征方向</small></div>
</div>

```python
import numpy as np
import pandas as pd

profiles = pd.DataFrame({
    "morning": [120, 98, 145, 72, 88, 156],
    "midday": [70, 65, 82, 58, 60, 90],
    "evening": [135, 110, 158, 80, 93, 171],
})
X = profiles.to_numpy(dtype=float)
mean = X.mean(axis=0)
centered = X - mean

U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
print("U", U.shape, "s", singular_values.shape, "Vt", Vt.shape)
print("重构一致：", np.allclose(centered, U @ np.diag(singular_values) @ Vt))
```

Σ 在 NumPy 中以一维奇异值数组返回，按从大到小排序。

## 2. SVD 与 PCA 的直接对应

- PCA loadings / directions：`V` 的列，也就是 `Vt` 的行。
- PCA scores：`UΣ`，也等于 `X_centered @ V`。
- 第 `j` 个方向的样本方差：`σⱼ²/(n-1)`。

```python
scores_from_u = U * singular_values
scores_from_projection = centered @ Vt.T
explained_variance = singular_values ** 2 / (len(centered) - 1)
explained_ratio = explained_variance / explained_variance.sum()

print("Scores 一致：", np.allclose(scores_from_u, scores_from_projection))
print("解释方差比例：", explained_ratio.round(3))
```

<QuizBlock question="中心化 X 的 PCA 方向位于 SVD 的哪一部分？" :options="['V 的列 / Vᵀ 的行', 'U 的行', 'Σ 的对角线本身', '均值向量']" :answer="0" explanation="右奇异向量位于特征空间，正是主成分方向。" />

## 3. 每个成分都是一层 rank-1 矩阵

`X = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + ...`。只保留前 `k` 层得到 rank-k 近似。

<SvdLayersLab />

```python
def reconstruct_rank_k(U, singular_values, Vt, k):
    return (U[:, :k] * singular_values[:k]) @ Vt[:k, :]

for k in [1, 2, 3]:
    approximation = reconstruct_rank_k(U, singular_values, Vt, k)
    error = np.mean((centered - approximation) ** 2)
    print(k, "MSE：", round(error, 4))
```

在所有 rank-k 矩阵中，截断 SVD 给出平方重构误差最小的近似。这让它自然适合压缩和降噪。

## 4. Scree plot 帮助选择成分数

Scree plot 把成分编号与特征值或解释方差画出来。常见选择方法：

- 累计解释方差达到预设门槛，如 90%。
- 曲线从陡降转为平缓的“肘部”。
- 在下游交叉验证中选择能改善任务的 k。
- 结合可解释性和存储/计算预算。

```python
cumulative = np.cumsum(explained_ratio)
minimum_k = int(np.searchsorted(cumulative, 0.9) + 1)
print("累计解释方差：", cumulative.round(3))
print("达到 90% 所需 k：", minimum_k)
```

90% 不是自然定律。无监督方差与预测价值不同，低方差方向也可能承载稀有但重要的信号。

## 5. Biplot 同时展示样本与特征

Biplot 通常在 PC1–PC2 平面画样本 Scores，并用箭头表示原始特征 loadings。

```python
score_table = pd.DataFrame(scores_from_u[:, :2], columns=["PC1", "PC2"])
loading_table = pd.DataFrame(Vt[:2, :].T, index=profiles.columns, columns=["PC1", "PC2"])

print("样本坐标：")
print(score_table.round(2))
print("特征方向：")
print(loading_table.round(3))
```

箭头方向相近表示变量在这两个主成分上共同变化；接近反向表示负关联；接近直角表示在二维投影中关联较弱。Biplot 只显示保留的两个方向，不能代表所有高维关系。

## 6. 图像压缩也是同一个矩阵问题

灰度图可以看成行×列矩阵。保留前 k 个奇异值，会先保留大范围亮暗结构，再逐步加入细节。

```python
image = np.array([
    [8, 8, 8, 2, 2],
    [8, 8, 7, 2, 2],
    [7, 7, 6, 3, 3],
    [2, 2, 3, 8, 8],
    [2, 2, 3, 8, 9],
], dtype=float)

Ui, si, Vti = np.linalg.svd(image, full_matrices=False)
for k in [1, 2, 5]:
    approximation = reconstruct_rank_k(Ui, si, Vti, k)
    print(k, "重构误差：", np.linalg.norm(image - approximation).round(3))
```

## 7. 数值与解释注意事项

- 重复或非常接近的奇异值会使对应方向不稳定，但它们张成的子空间仍可稳定。
- 符号翻转完全等价。
- 异常值可能主导大奇异值。
- 稀疏大矩阵通常使用截断算法，不必计算全部 SVD。

<div class="ds-warning"><strong>解释的是方向，不是隐藏实体</strong>PC1 只是使方差最大的线性组合。给它命名需要载荷、样本分布与领域证据，不应把数学轴自动人格化或因果化。</div>

## 8. 知识检查

<QuizBlock question="第 j 个 PCA 解释方差与奇异值有什么关系？" :options="['σⱼ²/(n-1)', 'σⱼ/(n-1)²', '1/σⱼ', '与 σ 无关']" :answer="0" explanation="中心化矩阵协方差特征值等于奇异值平方除以 n-1。" />

<QuizBlock question="只保留前 k 个奇异值会得到什么？" :options="['rank-k 低秩近似', '原矩阵行数减少为 0', '所有列独立', '分类阈值']" :answer="0" explanation="前 k 个 rank-1 层相加，秩最多为 k。" />

<QuizBlock question="达到 90% 解释方差是否保证下游预测最好？" :options="['保证', '不保证；PCA 不使用目标变量', '只对分类保证', 'k 越小越保证']" :answer="1" explanation="高方差方向不一定是目标最相关方向，应结合验证任务选择。" />

<QuizBlock question="两个奇异值非常接近时，什么可能不稳定？" :options="['各自方向的具体旋转，但联合子空间可稳定', '样本量', '原始数据类型', 'SQL 主键']" :answer="0" explanation="接近重根允许子空间内部方向发生明显旋转。" />

## 9. 代码练习：SVD 压缩报告

写函数接收矩阵和阈值，返回达到累计解释方差阈值的最小 k、rank-k 重构、相对 Frobenius 误差。

<details><summary>提示</summary>

解释方差比例使用 `s**2 / sum(s**2)`；相对误差是 `norm(X-Xk)/norm(X)`。

</details>

<details><summary>参考实现</summary>

```python
def svd_compression_report(matrix, threshold=0.9):
    matrix = np.asarray(matrix, dtype=float)
    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)
    ratio = singular_values ** 2 / np.sum(singular_values ** 2)
    k = int(np.searchsorted(np.cumsum(ratio), threshold) + 1)
    reconstruction = reconstruct_rank_k(U, singular_values, Vt, k)
    relative_error = np.linalg.norm(matrix - reconstruction) / np.linalg.norm(matrix)
    return k, reconstruction, relative_error

k, reconstruction, relative_error = svd_compression_report(image, 0.9)
print(k, round(relative_error, 4))
assert reconstruction.shape == image.shape
```

</details>

## 10. 本讲小结

- SVD 把矩阵拆成样本方向 U、强度 Σ 和特征方向 Vᵀ。
- 中心化数据的 PCA directions 是右奇异向量，scores 是 UΣ。
- 截断前 k 个成分得到最小平方误差的 rank-k 近似。
- Scree、累计方差、下游验证与解释成本共同决定保留成分数。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 25：SVD、主成分、解释方差、Scree、Biplot 与低秩重构的知识边界。
- *Learning Data Science*，Appendix A2：SVD 与线性代数主题。
- 本页站点画像和图像矩阵、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="25" next-label="第 26 讲：聚类" next-href="/courses/data-science/26-clustering" />
