---
course: data-science
title: 多元回归与最小二乘
description: 用设计矩阵组织多个特征，从矩阵乘法、向量投影和正交残差理解普通最小二乘。
outline: [2, 3]
---

<DsLessonHero
  number="12"
  title="多元回归与最小二乘"
  eyebrow="把每个样本写成一行，把每个可用信号写成一列"
  duration="建议 120–140 分钟"
  :outcomes="[
    '根据样本数和特征数判断 X、β、y 的形状',
    '把线性预测写成矩阵乘法 Xβ',
    '从投影与正交理解 OLS、残差和 R²'
  ]"
/>

## 本讲要解决的问题

温度不是峰值用电的唯一线索。能源站还记录湿度、是否工作日和滞后一天的用电量。如何让四个信号共同参与预测，同时不把公式写成一团？

## 1. 设计矩阵是一份模型输入合同

假设有 `n` 天、`p` 个模型系数：

<figure class="ds-figure">
  <div class="ds-matrix-figure">
    <div><b>X</b><span>n × p</span><small>每行一个观测<br>每列一个特征</small></div>
    <i>×</i>
    <div><b>β</b><span>p × 1</span><small>每个特征<br>一个系数</small></div>
    <i>=</i>
    <div><b>ŷ</b><span>n × 1</span><small>每个观测<br>一个预测</small></div>
  </div>
  <figcaption>矩阵乘法的内部维度必须一致：`X` 的列数等于 `β` 的长度。</figcaption>
</figure>

```python
import numpy as np
import pandas as pd

days = pd.DataFrame({
    "temp": [22, 25, 28, 30, 32, 26, 24, 31],
    "humidity": [48, 55, 61, 68, 72, 58, 51, 70],
    "workday": [1, 1, 1, 1, 1, 0, 0, 1],
    "previous_peak": [46, 49, 53, 58, 62, 60, 51, 55],
    "peak_mwh": [47, 51, 56, 61, 66, 57, 49, 64],
})

feature_names = ["temp", "humidity", "workday", "previous_peak"]
X_features = days[feature_names].to_numpy(dtype=float)
X = np.column_stack([np.ones(len(days)), X_features])
y = days["peak_mwh"].to_numpy(dtype=float)

print("X:", X.shape, "y:", y.shape)
```

第一列全 1，用于承载截距。此时 `X` 有 5 列，系数向量也必须有 5 个数。

<QuizBlock question="X 的形状是 (80, 5)，β 的长度应该是多少？" :options="['5', '16', '80', '400']" :answer="0" explanation="每一列对应一个系数，所以 β 的长度等于 X 的列数 5。" />

## 2. 一行乘以系数，就是一次预测

矩阵乘法没有神秘的新规则。第 `i` 行预测就是该行所有特征与系数逐项相乘再相加。

```python
beta_guess = np.array([5.0, 0.8, 0.12, 2.0, 0.45])
all_predictions = X @ beta_guess
first_prediction = np.sum(X[0] * beta_guess)

print(all_predictions[:3])
print("第一行两种算法一致：", np.isclose(first_prediction, all_predictions[0]))
```

系数解释必须带上“其他特征保持不变”。例如湿度系数为 `0.12`，表示温度、工作日和前一天用电相同时，湿度增加 1 个百分点，预测增加 `0.12 MWh`。

## 3. OLS 是把 y 投影到 X 的列空间

`X` 的每一列都是一个长度为 `n` 的向量。它们能够线性组合出的全部预测，构成 `X` 的列空间。OLS 从这个空间中找到离观测向量 `y` 最近的 `ŷ`。

<ProjectionLab />

几何上的最近点有一个关键性质：残差 `e = y - ŷ` 与列空间中的每个方向正交。因此：

```python
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
prediction = X @ beta
residual = y - prediction

print("beta：", beta.round(3))
print("X.T @ residual：", (X.T @ residual).round(10))
```

因为 `X` 包含全 1 列，残差也与全 1 向量正交，所以残差和为 0。其余特征与残差的点积也为 0。

<details><summary>第三层：从正交得到正规方程</summary>

正交条件是 `Xᵀ(y - Xβ) = 0`。展开后得到 `XᵀXβ = Xᵀy`。若 `XᵀX` 可逆，可以写成 `β = (XᵀX)⁻¹Xᵀy`。实际计算不要主动求逆；`np.linalg.lstsq` 或数值分解通常更稳定。

</details>

## 4. 共线性让单个系数变得不稳定

如果两列几乎重复，例如温度摄氏度与由它线性换算的华氏度，模型可以用许多系数组合产生几乎相同的预测。

```python
X_collinear = np.column_stack([
    np.ones(len(days)),
    days["temp"],
    days["temp"] * 1.8 + 32,
])

print("设计矩阵秩：", np.linalg.matrix_rank(X_collinear))
print("列数：", X_collinear.shape[1])
```

当秩小于列数时，系数没有唯一解。即使只是“接近”共线，单个系数也可能因样本轻微变化而大幅摆动。

<div class="ds-warning"><strong>预测稳定不等于解释稳定</strong>两个共线模型可能给出相似预测，却对“哪一个变量起作用”给出完全不同的系数。解释系数前要检查特征定义、相关结构和研究设计。</div>

## 5. R² 比较模型与均值基线

R² 把模型残差平方和与“只预测 y 均值”的残差平方和比较：`R² = 1 - SSE / SST`。

```python
sse = np.sum((y - prediction) ** 2)
sst = np.sum((y - y.mean()) ** 2)
r_squared = 1 - sse / sst

print("R²：", round(r_squared, 3))
```

- `R² = 0`：在训练样本上不比均值基线更好。
- `R² = 1`：训练样本被完全拟合。
- 在新数据上，R² 可以小于 0，表示比均值基线还差。

R² 高不代表因果正确、误差很小或模型公平；它只描述相对于该样本总变异，线性模型解释了多少平方误差。

## 6. 标准化让系数尺度更可比较

不同列的单位差异会让系数数值难以比较，并影响后续梯度下降与正则化。

```python
means = X_features.mean(axis=0)
scales = X_features.std(axis=0, ddof=0)
X_scaled = (X_features - means) / scales

X_scaled_with_intercept = np.column_stack([np.ones(len(days)), X_scaled])
beta_scaled, *_ = np.linalg.lstsq(X_scaled_with_intercept, y, rcond=None)
print(beta_scaled.round(3))
```

标准化参数必须只从训练集计算，再原样应用到验证集和测试集；否则会泄漏未来信息。

## 7. 知识检查

<QuizBlock question="OLS 几何解释中的 ŷ 位于哪里？" :options="['X 的列空间', '残差空间外的任意点', '只位于 y 轴', '总是等于 y']" :answer="0" explanation="所有线性预测都是 X 各列的线性组合，因此 ŷ 位于 X 的列空间。" />

<QuizBlock question="最小二乘解处，Xᵀe 应该接近什么？" :options="['全 1', '0 向量', 'β', 'y 的均值']" :answer="1" explanation="最优残差与 X 的每一列正交，所以 Xᵀe=0。" />

<QuizBlock question="加入一列与现有列完全重复的特征，主要风险是什么？" :options="['样本数翻倍', '系数解不唯一', '目标变量消失', 'MSE 自动为 0']" :answer="1" explanation="重复列使设计矩阵列不满秩，多个系数组合可产生同样预测。" />

<QuizBlock question="R²=0.8 可以直接推出什么？" :options="['模型是因果模型', '训练样本中相对均值基线减少了约 80% 的平方误差', '测试误差一定很小', '所有群体表现相同']" :answer="1" explanation="R² 是相对均值基线的样本内平方误差比例，不自动说明因果或泛化。" />

## 8. 代码练习：实现 OLS 诊断表

写函数接收 `X`、`y`，返回系数、RMSE、R²、最大绝对正交误差 `max(abs(X.T @ residual))`。用本讲数据验证。

<details><summary>提示</summary>

使用 `np.linalg.lstsq(X, y, rcond=None)[0]`。R² 的分母是 `sum((y-y.mean())**2)`。

</details>

<details><summary>参考实现</summary>

```python
def ols_report(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    prediction = X @ beta
    residual = y - prediction
    rmse = np.sqrt(np.mean(residual ** 2))
    r_squared = 1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)
    orthogonality_error = np.max(np.abs(X.T @ residual))
    return beta, rmse, r_squared, orthogonality_error

report = ols_report(X, y)
print("beta：", report[0].round(3))
print("RMSE、R²、正交误差：", report[1:])
assert report[3] < 1e-8
```

</details>

## 9. 本讲小结

- 设计矩阵的行对应观测，列对应模型使用的特征。
- `Xβ` 同时生成全部预测，形状检查是第一道防错线。
- OLS 是把 `y` 投影到 `X` 的列空间，残差与所有特征正交。
- R² 比较模型与均值基线；共线性会伤害系数稳定性。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 12：多元回归、设计矩阵、投影、正规方程、残差性质与 R² 的知识边界。
- *Learning Data Science*，Ch.15、Appendix A2：线性模型和矩阵代数主题。
- 本页能源站案例、数据、文字、矩阵图、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="12" next-label="第 13 讲：sklearn 与梯度下降" next-href="/courses/data-science/13-gradient-descent" />
