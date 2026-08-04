---
course: data-science
title: 交叉验证与正则化
description: 用训练、验证、测试和 K-fold 估计泛化表现，用 L1/L2 在拟合与复杂度之间建立约束。
outline: [2, 3]
---

<DsLessonHero
  number="16"
  title="交叉验证与正则化"
  eyebrow="训练误差回答模型记住了多少，验证误差回答它能带走多少"
  duration="建议 120–140 分钟"
  :outcomes="[
    '区分训练集、验证集与测试集的职责',
    '解释 K-fold 如何提高有限数据的利用率',
    '比较 L1 与 L2，并说明正则化前为何需要缩放特征'
  ]"
/>

## 本讲要解决的问题

能源站团队尝试了 20 组特征和 12 个多项式次数。如果每次都看测试集再调整，最终的测试分数已经被选择过程污染。如何在有限数据下选择模型，又保留一次可信的最终评估？

## 1. 三份数据承担不同职责

<div class="ds-compare">
  <div><b>训练集</b><span>学习预处理与模型参数</span><small>系数、均值、类别词表</small></div>
  <div><b>验证集</b><span>选择特征与超参数</span><small>次数、λ、停止轮数</small></div>
  <div><b>测试集</b><span>最后估计泛化表现</span><small>决策完成后只看一次</small></div>
</div>

```python
import numpy as np

rng = np.random.default_rng(42)
indices = rng.permutation(100)
train_idx = indices[:60]
valid_idx = indices[60:80]
test_idx = indices[80:]

print(len(train_idx), len(valid_idx), len(test_idx))
print("是否重叠：", bool(set(train_idx) & set(valid_idx)))
```

时间序列不能随意随机打散未来和过去。若目标是预测明天，应按时间前后划分，或使用滚动验证。

<QuizBlock question="为什么测试集不能反复用于选择多项式次数？" :options="['测试集不能包含数字', '选择过程会针对测试集偶然性过拟合', '训练集会自动消失', '会让样本量翻倍']" :answer="1" explanation="反复比较并据此调整，相当于把测试信息反馈给模型开发。" />

## 2. K-fold 轮流让每一折做验证集

K-fold 把训练数据分成 `K` 份。每轮用一份验证、其余训练，最后平均 K 个验证分数。

```python
def kfold_indices(n, k, seed=42):
    order = np.random.default_rng(seed).permutation(n)
    return [part for part in np.array_split(order, k)]

folds = kfold_indices(23, 5)
for fold_number, validation_indices in enumerate(folds, start=1):
    training_indices = np.concatenate([
        fold for index, fold in enumerate(folds) if index != fold_number - 1
    ])
    print(f"第 {fold_number} 轮：train={len(training_indices)}, valid={len(validation_indices)}")
```

每一轮都必须重新拟合预处理和模型。若先在全训练数据上做特征选择，再交叉验证，验证折已经影响了特征空间。

## 3. 正则化给复杂系数收取费用

普通 OLS 只最小化训练 MSE。正则化在后面加惩罚：

- Ridge / L2：`MSE + λ Σβⱼ²`，平滑地收缩系数，通常不会精确变成 0。
- LASSO / L1：`MSE + λ Σ|βⱼ|`，会把一部分系数压到精确的 0，可产生稀疏模型。

截距通常不惩罚，因为它只控制整体水平，不代表特征复杂度。

<CrossValidationLab />

当 `λ = 0` 时回到未正则化模型；`λ` 越大，模型越偏好小系数。λ 是超参数，应在交叉验证中选择，而不是用测试集选择。

## 4. Ridge 有稳定的矩阵形式

给定已标准化且不含截距的 `X`，Ridge 解为 `(XᵀX + λI)⁻¹Xᵀy`。代码中仍然应使用线性方程求解而不是显式求逆。

```python
X_demo = np.array([
    [1.0, 0.9, 1.1], [1.0, 1.8, 2.0], [1.0, 2.7, 2.9],
    [1.0, 3.6, 3.7], [1.0, 4.5, 4.6], [1.0, 5.4, 5.7],
])
y_demo = np.array([2.2, 3.7, 5.5, 6.8, 8.2, 10.1])

def fit_ridge(X, y, strength):
    penalty = np.eye(X.shape[1])
    penalty[0, 0] = 0  # 不惩罚截距
    return np.linalg.solve(X.T @ X + strength * penalty, X.T @ y)

for strength in [0, 0.5, 5, 50]:
    print(strength, fit_ridge(X_demo, y_demo, strength).round(3))
```

后两列高度相关。未正则化系数可能互相补偿；Ridge 倾向把权重更平滑地分配给相关特征。

## 5. 不缩放，惩罚就不公平

同样的系数数值，在不同特征单位下代表完全不同的变化。如果面积用平方米，范围约 100；换成平方厘米后范围变成 1,000,000，系数会相应缩小。

```python
raw = np.array([
    [20, 18000], [24, 23000], [28, 31000], [32, 39000], [35, 45000]
], dtype=float)
means = raw.mean(axis=0)
scales = raw.std(axis=0, ddof=0)
scaled = (raw - means) / scales

print("原尺度：", raw.std(axis=0, ddof=0))
print("标准化：", scaled.std(axis=0, ddof=0))
```

L1/L2 直接惩罚系数大小，所以通常先标准化数值特征。标准化参数仍然只能由每一轮训练折学习。

## 6. 一条完整而安全的管道

```python
# tutorial-validation: syntax-only
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric = make_pipeline(SimpleImputer(), StandardScaler())
categorical = OneHotEncoder(handle_unknown="ignore")
preprocess = ColumnTransformer([
    ("number", numeric, ["temp", "humidity"]),
    ("category", categorical, ["day_type"]),
])
model = make_pipeline(preprocess, Ridge(alpha=1.0))
```

Pipeline 的价值不只是少写几行代码，而是把“哪些状态只能从训练数据学习”变成可执行边界。

## 7. 选择 K 与评价指标

- K 较小：每轮训练快，但验证集较大、轮次较少。
- K 较大：每轮训练数据更多，但计算更贵，各折分数相关性更高。
- 常见的 5-fold 或 10-fold 只是起点；时间、群组和稀有类别可能要求专门划分。

评价指标也要和部署目标一致。若大误差特别昂贵，可用 RMSE；若需要稳健，可同时报告 MAE；若群体后果重要，应增加分群指标。

## 8. 知识检查

<QuizBlock question="K-fold 中每个样本通常扮演几次验证样本？" :options="['0 次', '1 次', 'K-1 次', '每轮都是']" :answer="1" explanation="标准 K-fold 中，每个样本属于一个折，因此恰好做一次验证样本。" />

<QuizBlock question="L1 与 L2 的典型差异是什么？" :options="['L1 更容易产生精确为 0 的系数', 'L2 不能处理数字', 'L1 不需要 λ', 'L2 总能消除过拟合']" :answer="0" explanation="L1 的几何形状使最优点更容易落在坐标轴上，从而得到稀疏系数。" />

<QuizBlock question="λ 过大最可能导致什么？" :options="['高方差', '系数被过度收缩并欠拟合', '测试信息泄漏自动消失', '类别数量增加']" :answer="1" explanation="过强约束限制模型表达能力，训练和验证误差都可能升高。" />

<QuizBlock question="为什么每折都要重新拟合 StandardScaler？" :options="['让每折列名不同', '避免验证折参与均值和标准差估计', '减少 K', '让 λ 等于 0']" :answer="1" explanation="预处理也是训练的一部分，验证折不能参与任何已学习状态。" />

## 9. 代码练习：用 K-fold 选择 Ridge 强度

实现 `cross_validate_ridge(X, y, strengths, k)`，对每个强度计算平均验证 MSE，返回最佳强度与结果表。数据 `X` 已含截距。

<details><summary>提示</summary>

使用本讲 `kfold_indices`。每轮用布尔掩码或集合差得到训练索引；调用 `fit_ridge` 后计算验证 MSE。

</details>

<details><summary>参考实现</summary>

```python
def cross_validate_ridge(X, y, strengths, k=3):
    folds = kfold_indices(len(X), k)
    results = {}
    for strength in strengths:
        fold_losses = []
        for fold_index, valid in enumerate(folds):
            train = np.concatenate([fold for index, fold in enumerate(folds) if index != fold_index])
            beta = fit_ridge(X[train], y[train], strength)
            prediction = X[valid] @ beta
            fold_losses.append(np.mean((y[valid] - prediction) ** 2))
        results[strength] = float(np.mean(fold_losses))
    best = min(results, key=results.get)
    return best, results

best_strength, cv_results = cross_validate_ridge(X_demo, y_demo, [0, 0.1, 1, 10], k=3)
print(best_strength, cv_results)
```

</details>

## 10. 本讲小结

- 训练、验证、测试集分别服务于参数学习、模型选择和最终评估。
- K-fold 让有限训练数据轮流参与验证，但每折都要重训完整管道。
- L2 平滑收缩系数，L1 还可能产生稀疏解。
- 正则化强度要通过验证选择，数值特征通常应先缩放。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 16：数据划分、K-fold、超参数、L1/L2、Ridge 与 LASSO 的知识边界。
- *Learning Data Science*，Ch.16：泛化、验证与正则化主题。
- 本页能源站案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="16" next-label="第 17 讲：随机变量" next-href="/courses/data-science/17-random-variables" />
