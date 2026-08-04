---
course: data-science
title: 主成分分析：投影
description: 从中心化、点积和重构误差理解 PCA 如何寻找高方差投影方向并压缩高维数据。
outline: [2, 3]
---

<DsLessonHero
  number="24"
  title="主成分分析：投影"
  eyebrow="PCA 不删除某几列，而是寻找所有列的新组合方向"
  duration="建议 120–150 分钟"
  :outcomes="[
    '用点积计算样本在单位方向上的投影',
    '解释最大投影方差与最小重构误差的等价性',
    '从协方差矩阵特征向量计算主成分方向'
  ]"
/>

## 本讲要解决的问题

每个单车站点有 24 个小时骑行量、天气、容量和周边设施等几十个特征。许多列共同变化。PCA 尝试用少数新坐标保留主要变化结构，以便可视化、压缩和后续建模。

## 1. 先中心化，再谈方向

PCA 关心围绕均值的变化。若不中心化，第一方向可能主要指向数据离原点的位置，而不是点云展开方向。

```python
import numpy as np
import pandas as pd

station_profiles = pd.DataFrame({
    "morning": [120, 98, 145, 72, 88, 156],
    "midday": [70, 65, 82, 58, 60, 90],
    "evening": [135, 110, 158, 80, 93, 171],
})

X = station_profiles.to_numpy(dtype=float)
means = X.mean(axis=0)
X_centered = X - means

print("中心化后列均值：", X_centered.mean(axis=0).round(10))
```

若各列单位或尺度差异很大，可以进一步标准化。是否标准化会改变 PCA 问题：协方差 PCA 让高方差原始列更有影响，相关矩阵 PCA 让标准化列处于相近尺度。

## 2. 投影分数是点积

给一个长度为 1 的方向向量 `v`，样本行 `xᵢ` 在该方向的坐标是 `zᵢ = xᵢ·v`；投影回原空间是 `zᵢv`。

```python
direction = np.array([1.0, 0.0, 1.0])
direction = direction / np.linalg.norm(direction)

scores = X_centered @ direction
reconstruction = np.outer(scores, direction)
error = np.mean(np.sum((X_centered - reconstruction) ** 2, axis=1))

print("scores：", scores.round(2))
print("平均重构误差：", round(error, 2))
```

方向必须归一化；否则把 `v` 乘 100 会人为放大分数，无法公平比较方向。

<QuizBlock question="为什么 PCA 方向通常要求长度为 1？" :options="['避免通过任意缩放方向来放大投影方差', '让样本数变成 1', '删除所有负数', '保证列名相同']" :answer="0" explanation="单位长度约束让方向只比较朝向，不比较人为倍率。" />

## 3. 第一主成分寻找最展开的方向

对中心化数据，第一主成分选择使投影分数方差最大的单位向量。这个方向也使一维投影后的平方重构误差最小。

<PcaProjectionLab />

直觉上，点云总平方距离可以拆成“沿投影轴的平方距离”和“垂直轴的平方距离”。总量固定，所以保留得越多，丢失得越少。

<details><summary>第三层：最大方差为何变成特征向量问题？</summary>

投影方差正比于 `vᵀXᵀXv`，约束为 `vᵀv=1`。使用拉格朗日乘子可得 `XᵀXv=λv`。因此最优方向是协方差矩阵最大特征值对应的特征向量。

</details>

## 4. 用协方差矩阵计算主成分

```python
covariance = X_centered.T @ X_centered / (len(X_centered) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
components = eigenvectors[:, order]

pc1 = components[:, 0]
pc1_scores = X_centered @ pc1

print("特征值：", eigenvalues.round(2))
print("PC1 方向：", pc1.round(3))
print("PC1 分数：", pc1_scores.round(2))
```

使用 `eigh` 是因为协方差矩阵对称，返回的特征向量相互正交。第二主成分在与第一主成分正交的方向中最大化剩余方差。

## 5. Scores 与 Loadings 回答不同问题

- Loadings / directions：原始列怎样组合成主成分。
- Scores：每个样本在新主成分坐标中的位置。

```python
score_frame = pd.DataFrame(
    X_centered @ components[:, :2],
    columns=["PC1", "PC2"],
)
loading_frame = pd.DataFrame(
    components[:, :2],
    index=station_profiles.columns,
    columns=["PC1", "PC2"],
)

print(score_frame.round(2))
print(loading_frame.round(3))
```

如果 morning 与 evening 在 PC1 上同号且较大，PC1 可暂时描述为“通勤高峰强度方向”。这是基于载荷与领域知识的解释，不是算法自动赋予的真名。

## 6. 主成分符号可以整体翻转

若 `v` 是主成分，`-v` 也是同一方向；分数也会整体翻转，但重构完全相同。比较不同软件或不同运行结果时，不应把符号翻转误认为模型改变。

```python
reconstruction_a = np.outer(X_centered @ pc1, pc1)
reconstruction_b = np.outer(X_centered @ (-pc1), -pc1)
print(np.allclose(reconstruction_a, reconstruction_b))
```

## 7. PCA 的用途与限制

适合：

- 把高维连续特征压到 2–3 维做探索图。
- 去除强相关列中的冗余方向。
- 压缩、去噪或为模型构造正交特征。

限制：

- 完全忽略目标变量，高方差方向未必对预测有用。
- 只能寻找线性子空间。
- 对缩放、异常值和缺失处理敏感。
- 主成分是混合列，业务解释可能比原始特征困难。

<div class="ds-warning"><strong>不要用全数据拟合 PCA 再划分</strong>PCA 的均值与方向都是从数据学习的状态。监督学习中应只在训练集拟合，再变换验证和测试集。</div>

## 8. 知识检查

<QuizBlock question="第一主成分最大化什么？" :options="['投影分数方差', '原始列数', '样本均值', '分类 Accuracy']" :answer="0" explanation="在单位长度约束下，PC1 选择投影方差最大的方向。" />

<QuizBlock question="对中心化数据，一维投影方差最大等价于什么？" :options="['重构平方误差最小', '样本量最大', '所有特征独立', '平均值为 1']" :answer="0" explanation="总平方距离固定，沿轴保留越多，垂直方向丢失越少。" />

<QuizBlock question="PCA Scores 描述什么？" :options="['每个样本在主成分坐标中的位置', '每列数据类型', '数据库索引', '分类阈值']" :answer="0" explanation="Scores 是样本投影后的新坐标；loadings 描述原始特征组合。" />

<QuizBlock question="把主成分向量整体乘 -1 会怎样？" :options="['表示同一方向，分数翻转但重构不变', '方差归零', '样本消失', '变成第二主成分']" :answer="0" explanation="方向轴没有固有正端，符号整体翻转是等价表示。" />

## 9. 代码练习：封装二维 PCA

写函数 `fit_pca(X, n_components)`，返回训练均值、主成分、分数、解释方差比例和重构；验证重构误差随保留成分增加不升高。

<details><summary>提示</summary>

中心化后计算协方差矩阵，用 `np.linalg.eigh` 并按特征值降序排列。重构是 `scores @ components.T + mean`。

</details>

<details><summary>参考实现</summary>

```python
def fit_pca(X, n_components):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (len(X) - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    components = vectors[:, order[:n_components]]
    scores = centered @ components
    reconstruction = scores @ components.T + mean
    explained_ratio = values[:n_components] / values.sum()
    return mean, components, scores, explained_ratio, reconstruction

errors = []
for count in [1, 2, 3]:
    *_, reconstructed = fit_pca(X, count)
    errors.append(np.mean((X - reconstructed) ** 2))
print(errors)
assert all(left >= right - 1e-12 for left, right in zip(errors, errors[1:]))
```

</details>

## 10. 本讲小结

- PCA 先围绕均值观察变化，缩放选择决定各列相对影响。
- 投影分数是点积；第一主成分最大化投影方差并最小化一维重构误差。
- 协方差矩阵最大特征向量给出 PC1，后续方向相互正交。
- Scores 描述样本，loadings 描述特征组合；主成分符号可等价翻转。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 24：高维数据、投影、方差、重构误差与 PCA 几何的知识边界。
- *Learning Data Science*，Ch.13 案例、Appendix A2：降维与线性代数主题。
- 本页站点画像案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="24" next-label="第 25 讲：PCA 与 SVD" next-href="/courses/data-science/25-pca-svd" />
