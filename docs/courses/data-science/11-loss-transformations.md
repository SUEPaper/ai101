---
course: data-science
title: 损失函数与数据变换
description: 从常数模型理解 MSE、MAE、均值和中位数，检查异常值敏感性并用变换重塑关系。
outline: [2, 3]
---

<DsLessonHero
  number="11"
  title="损失函数与数据变换"
  eyebrow="所谓最佳模型，永远是相对于某个损失标准而言"
  duration="建议 100–120 分钟"
  :outcomes="[
    '解释为什么 MSE 对应均值、MAE 对应中位数',
    '比较异常值对平方损失和绝对损失的影响',
    '判断对数与平方根变换何时有助于建模'
  ]"
/>

## 本讲要解决的问题

在增加更多特征前，先研究最简单的模型：无论输入是什么，只预测同一个常数。它看似笨拙，却能回答一个关键问题——“最佳”到底由谁定义？

## 1. 常数模型是所有模型的基线

假设能源站最近六天的预测误差修正量为 `5, 6, 6, 7, 8, 12`。常数模型 `ŷ = c` 需要选一个数代表全部观测。

```python
import numpy as np

adjustments = np.array([5, 6, 6, 7, 8, 12], dtype=float)

def constant_predictions(values, constant):
    return np.full_like(values, constant, dtype=float)

for constant in [6, 7, 8]:
    errors = adjustments - constant_predictions(adjustments, constant)
    print(constant, "MSE=", np.mean(errors ** 2), "MAE=", np.mean(np.abs(errors)))
```

基线模型有两种用途：为复杂模型提供最低参照；让我们单独研究损失函数而不被模型结构干扰。

## 2. 平方损失选择均值

MSE 把每个偏差平方。随着 `c` 移动，损失是一条开口向上的抛物线，最低点恰好在样本均值。

```python
mean_constant = adjustments.mean()
grid = np.linspace(4, 13, 181)
mse_grid = [np.mean((adjustments - constant) ** 2) for constant in grid]
best_mse_constant = grid[np.argmin(mse_grid)]

print("均值：", mean_constant)
print("网格搜索的 MSE 最优常数：", best_mse_constant)
```

<details><summary>第三层：为什么导数给出均值？</summary>

把损失写为 `(1/n) Σ(yᵢ-c)²`。对 `c` 求导得到 `(2/n) Σ(c-yᵢ)`。令导数为 0，可得 `nc = Σyᵢ`，所以 `c = ȳ`。

</details>

## 3. 绝对损失选择中位数

MAE 使用距离的绝对值。只要 `c` 位于排序后数据的中间位置，把它稍微向左或向右移动，都不会让总距离继续下降。

```python
median_constant = np.median(adjustments)
mae_grid = [np.mean(np.abs(adjustments - constant)) for constant in grid]
best_mae_constant = grid[np.argmin(mae_grid)]

print("中位数：", median_constant)
print("网格搜索的 MAE 最优常数：", best_mae_constant)
```

<LossComparisonLab />

平方损失让一个 10 单位误差的代价成为 1 单位误差的 100 倍；绝对损失只放大到 10 倍。因此 MSE 更关注大错，MAE 对异常值更稳健。

<QuizBlock question="用常数预测并最小化 MSE，最优常数是什么？" :options="['众数', '均值', '中位数', '最大值']" :answer="1" explanation="平方损失关于常数求导，最低点位于样本均值。" />

## 4. 损失函数表达业务代价

损失不是纯数学装饰。若低估峰值负荷会引发停电，而高估只多购买一点备用容量，对称的 MSE/MAE 可能都不合适。

```python
def asymmetric_loss(actual, predicted):
    error = np.asarray(actual) - np.asarray(predicted)
    return np.mean(np.where(error > 0, 3 * error, -error))

for constant in [7, 8, 9, 10]:
    print(constant, asymmetric_loss(adjustments, constant_predictions(adjustments, constant)))
```

<figure class="ds-figure">
  <div class="ds-flow">
    <div><b>TASK</b><span>预测服务什么决策</span><small>调度、报价还是预警</small></div>
    <div><b>ERROR</b><span>哪类错误更昂贵</span><small>高估与低估是否对称</small></div>
    <div><b>LOSS</b><span>把代价写成函数</span><small>MSE、MAE 或自定义</small></div>
    <div><b>MODEL</b><span>在该标准下优化</span><small>“最佳”由损失定义</small></div>
  </div>
  <figcaption>同一组预测，在不同决策场景中可能得到不同评价。</figcaption>
</figure>

## 5. 变换可以让关系更适合建模

有些变量跨越多个数量级，或者增长速度与当前规模成比例。对数变换会把乘法关系转成加法关系，并压缩极大的值。

```python
temperature = np.array([18, 21, 24, 27, 30, 33], dtype=float)
cooling_load = np.array([4, 6, 10, 17, 29, 48], dtype=float)

raw_correlation = np.corrcoef(temperature, cooling_load)[0, 1]
log_correlation = np.corrcoef(temperature, np.log(cooling_load))[0, 1]

print("原尺度相关：", round(raw_correlation, 3))
print("log(y) 相关：", round(log_correlation, 3))
```

如果拟合 `log(y) = b₀ + b₁x`，那么 `x` 增加 1 时，原尺度的预测约乘以 `exp(b₁)`。解释必须回到业务尺度，而不是停在变换后的数字。

<div class="ds-warning"><strong>变换不是自动清洗</strong>对数要求值为正；给所有值随意加常数会改变解释。是否变换应由数据生成机制、残差诊断和任务目标共同决定。</div>

## 6. 多元模型从“变换后的列”开始

特征工程的本质，是从原始输入构造模型更容易使用的表示。哪怕只拟合一条直线，也可以先把温度变成“超过舒适阈值的度数”。

```python
temperature = np.array([18, 21, 24, 27, 30, 33], dtype=float)
cooling_degrees = np.maximum(temperature - 22, 0)

design_preview = np.column_stack([
    np.ones_like(temperature),
    temperature,
    cooling_degrees,
])
print(design_preview)
```

下一讲会把这些列组成设计矩阵，让多个特征同时参与 OLS。

## 7. 知识检查

<QuizBlock question="把一个极端大值继续增大，哪个统计量通常移动得更快？" :options="['中位数', '均值', '样本量', '列名']" :answer="1" explanation="均值直接包含每个数值的大小，中位数主要由排序位置决定。" />

<QuizBlock question="什么时候更有理由选择 MAE？" :options="['希望大误差受到平方惩罚', '误差代价近似线性且希望对异常值更稳健', '目标是类别标签', '数据只有一行']" :answer="1" explanation="MAE 对误差按绝对值计价，对少数极端值的敏感度低于 MSE。" />

<QuizBlock question="对 y 取对数后，模型预测的是什么？" :options="['仍直接预测原始 y', '预测 log(y)，解释时需变回原尺度', '预测 x 的对数', '自动消除所有异常值']" :answer="1" explanation="目标变量的尺度已改变，预测与误差首先存在于 log 尺度。" />

<QuizBlock question="自定义不对称损失的主要原因是什么？" :options="['让代码更长', '表达高估与低估的业务代价不同', '保证相关等于 1', '避免划分数据集']" :answer="1" explanation="损失函数应反映任务中不同错误带来的真实后果。" />

## 8. 代码练习：比较稳健性

从正态分布生成 100 个观测，分别求均值和中位数；再把最大值替换为 100，比较两个统计量移动了多少，并报告替换前后的最小 MSE 与最小 MAE。

<details><summary>提示</summary>

固定随机种子。MSE 使用均值作为常数，MAE 使用中位数作为常数；比较时不要混用最优点。

</details>

<details><summary>参考实现</summary>

```python
rng = np.random.default_rng(42)
clean = rng.normal(loc=10, scale=2, size=100)
contaminated = clean.copy()
contaminated[np.argmax(contaminated)] = 100

def best_constant_report(values):
    mean = values.mean()
    median = np.median(values)
    return {
        "mean": mean,
        "median": median,
        "min_mse": np.mean((values - mean) ** 2),
        "min_mae": np.mean(np.abs(values - median)),
    }

clean_report = best_constant_report(clean)
dirty_report = best_constant_report(contaminated)
print(clean_report)
print(dirty_report)
print("均值移动：", dirty_report["mean"] - clean_report["mean"])
print("中位数移动：", dirty_report["median"] - clean_report["median"])
```

</details>

## 9. 本讲小结

- 常数模型是理解损失与建立基线的最小模型。
- MSE 的最优常数是均值，MAE 的最优常数是中位数。
- MSE 更强调大误差，MAE 对极端值更稳健。
- 数据变换改变模型看到的关系，也改变系数与误差的解释。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 11：常数模型、MSE、MAE、异常值与变换的知识边界。
- *Learning Data Science*，Ch.4、Ch.15：损失和模型变换主题。
- 本页能源站案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="11" next-label="第 12 讲：多元回归与最小二乘" next-href="/courses/data-science/12-ordinary-least-squares" />
