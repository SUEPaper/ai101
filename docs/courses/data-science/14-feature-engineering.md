---
course: data-science
title: 特征工程与随机梯度下降
description: 比较 Batch、Mini-batch 与 SGD，用 one-hot、多项式和交互特征改变模型能表达的关系。
outline: [2, 3]
---

<DsLessonHero
  number="14"
  title="特征工程与随机梯度下降"
  eyebrow="模型只会使用你交给它的表示，优化器只会看到当前批次的坡度"
  duration="建议 120–140 分钟"
  :outcomes="[
    '比较 Batch、Mini-batch 和 SGD 的更新成本与噪声',
    '为类别、曲率和交互关系构造设计矩阵',
    '识别模型复杂度增加带来的欠拟合与过拟合'
  ]"
/>

## 本讲要解决的问题

同一个线性模型，通过更换输入列，可以表达季节差异、弯曲趋势和变量间的联合作用。与此同时，当记录从几十行增长到几百万行，我们也不必在每一步都扫描全部数据。

## 1. 三种梯度只差“每一步看多少数据”

| 策略 | 单次更新使用 | 路径特点 | 常见用途 |
|---|---:|---|---|
| Batch GD | 全部训练样本 | 平稳、单步较贵 | 中小数据与教学验证 |
| Mini-batch GD | 一小批样本 | 效率与稳定折中 | 大多数深度学习训练 |
| SGD | 1 个样本 | 抖动最大、更新最频繁 | 在线学习或极大数据 |

```python
import numpy as np

rng = np.random.default_rng(42)
x = np.linspace(-2, 2, 80)
y = 1.5 + 2.2 * x + rng.normal(0, 0.45, size=len(x))
X = np.column_stack([np.ones(len(x)), x])

def mini_batch_gradient(X_batch, y_batch, beta):
    residual = y_batch - X_batch @ beta
    return -(2 / len(y_batch)) * X_batch.T @ residual

beta = np.zeros(2)
rate = 0.08
for epoch in range(30):
    order = rng.permutation(len(X))
    for start in range(0, len(X), 10):
        batch = order[start:start + 10]
        beta -= rate * mini_batch_gradient(X[batch], y[batch], beta)

print(beta.round(3))
```

SGD 的单步梯度不是全数据梯度，但在随机抽样条件下，它是全梯度的有噪声估计。训练时通常按 epoch 打乱数据，避免固定顺序制造系统偏差。

## 2. 类别必须先变成可计算列

能源站把日类型记录为“工作日、周末、节假日”。直接编码为 `0, 1, 2` 会暗示类别有顺序和等距关系。One-hot 为每个类别建立指示列。

```python
import pandas as pd

calendar = pd.DataFrame({
    "day_type": ["工作日", "周末", "节假日", "工作日", "周末"],
    "temp": [28, 30, 27, 32, 29],
})

encoded = pd.get_dummies(calendar, columns=["day_type"], dtype=int)
print(encoded)
```

若模型同时包含截距和全部 `k` 个 one-hot 列，就会产生完全共线。常见做法是保留截距并把一个类别作为参考组，或取消截距。

<QuizBlock question="为什么不能简单把三个无序类别编码为 0、1、2？" :options="['NumPy 不支持整数', '会人为加入顺序与等距假设', '会增加样本数', '会自动删除截距']" :answer="1" explanation="数值编码使模型把类别间差异解释为沿一条数轴的固定步长。" />

## 3. 多项式特征让线性参数表达弯曲关系

把 `x²` 作为新列后，模型仍然对系数线性，却可以在原始坐标中画出抛物线：`ŷ = β₀ + β₁x + β₂x²`。

```python
temperature = np.array([18, 21, 24, 27, 30, 33], dtype=float)
load = np.array([41, 43, 47, 54, 64, 77], dtype=float)

X_linear = np.column_stack([np.ones(len(temperature)), temperature])
X_quadratic = np.column_stack([np.ones(len(temperature)), temperature, temperature ** 2])

beta_linear, *_ = np.linalg.lstsq(X_linear, load, rcond=None)
beta_quadratic, *_ = np.linalg.lstsq(X_quadratic, load, rcond=None)

def training_mse(X, y, beta):
    return np.mean((y - X @ beta) ** 2)

print("线性：", training_mse(X_linear, load, beta_linear))
print("二次：", training_mse(X_quadratic, load, beta_quadratic))
```

增加列后，训练误差不会变差，因为模型至少可以把新系数设为 0。但新数据误差可能先降后升。

<FeatureEngineeringLab />

## 4. 交互特征表达“一个变量改变另一个变量的作用”

湿度可能只在高温时显著增加体感负担。加入 `temp × humidity` 后，温度斜率会随湿度改变。

```python
weather = pd.DataFrame({
    "temp": [24, 28, 32, 24, 28, 32],
    "humidity": [40, 40, 40, 75, 75, 75],
    "peak_mwh": [48, 54, 62, 50, 60, 74],
})

weather["temp_x_humidity"] = weather["temp"] * weather["humidity"]
print(weather)
```

交互项改变了系数解释：加入交互后，`temp` 的单独系数表示湿度为 0 时的温度效应，可能不在合理数据范围。中心化变量常让截距和主效应更易解释。

## 5. 特征工程的边界必须从训练集学

以下信息都可能泄漏验证或测试集：

- 用全数据决定缺失值填充值。
- 看过全数据后选择最有用的类别或词汇。
- 根据全数据目标均值编码类别。
- 用未来记录构造当前时刻的滞后特征。

```python
train_categories = ["工作日", "周末"]
valid_categories = ["节假日", "工作日"]

known = set(train_categories)
valid_safe = [value if value in known else "其他" for value in valid_categories]
print(valid_safe)
```

生产系统还要预先定义“训练时没见过的新类别”如何处理，不能等线上报错后再临时决定。

## 6. 复杂度由整个特征管道决定

“线性回归很简单”可能是假象。若输入包含 1000 个词袋特征、六次多项式与大量交互项，模型已经具有很高灵活性。

<div class="ds-compare">
  <div><b>欠拟合</b><span>训练误差与验证误差都高</span><small>表示能力不足或信号缺失</small></div>
  <div><b>合适区域</b><span>验证误差最低</span><small>抓住可重复结构</small></div>
  <div><b>过拟合</b><span>训练误差继续降，验证误差上升</span><small>开始记忆训练集偶然性</small></div>
</div>

特征选择必须在验证过程内完成。看测试集后再删改特征，会让测试集变成事实上的验证集。

## 7. 知识检查

<QuizBlock question="Mini-batch 相比 Batch GD 的直接变化是什么？" :options="['每步只用部分样本估计梯度', '不再需要损失函数', '参数数量变少', '总能一步到最优']" :answer="0" explanation="Mini-batch 用部分样本降低单步计算量，同时引入梯度噪声。" />

<QuizBlock question="加入 x² 后，模型是否仍对参数线性？" :options="['是，β 仍以加法线性组合', '否，任何曲线都不是线性模型', '只在 x>0 时是', '取决于样本量']" :answer="0" explanation="线性模型指对系数 β 线性；输入可以先经过非线性变换。" />

<QuizBlock question="加入新特征后，训练 OLS 的 MSE 通常如何变化？" :options="['不会上升', '一定上升', '必定等于测试 MSE', '自动变成 MAE']" :answer="0" explanation="新模型包含旧模型作为系数为 0 的特例，所以训练最优 MSE 不会更差。" />

<QuizBlock question="验证误差先降后升最常对应什么？" :options="['从欠拟合走向合适，再走向过拟合', '样本量不断增加', '损失函数消失', '所有特征都无效']" :answer="0" explanation="复杂度增加先补足表达能力，随后开始拟合不可重复的样本噪声。" />

## 8. 代码练习：构造可复用设计矩阵

写函数接收包含 `temp`、`humidity`、`day_type` 的 DataFrame，返回含截距、中心化温度、中心化湿度、二次温度、交互项和 day_type one-hot（去掉参考组）的设计矩阵。

<details><summary>提示</summary>

先复制数据；中心化使用列均值；`pd.get_dummies(..., drop_first=True, dtype=float)`；最后用 `np.column_stack` 或 `pd.concat`。

</details>

<details><summary>参考实现</summary>

```python
def build_design(frame):
    work = frame.copy()
    work["temp_c"] = work["temp"] - work["temp"].mean()
    work["humidity_c"] = work["humidity"] - work["humidity"].mean()
    work["temp_squared"] = work["temp_c"] ** 2
    work["interaction"] = work["temp_c"] * work["humidity_c"]
    categories = pd.get_dummies(work["day_type"], prefix="day", drop_first=True, dtype=float)
    numeric = work[["temp_c", "humidity_c", "temp_squared", "interaction"]].astype(float)
    design = pd.concat([numeric, categories], axis=1)
    design.insert(0, "intercept", 1.0)
    return design

exercise_frame = pd.DataFrame({
    "temp": [24, 28, 32, 27],
    "humidity": [45, 60, 75, 55],
    "day_type": ["工作日", "周末", "工作日", "节假日"],
})
design = build_design(exercise_frame)
print(design)
assert design.shape[0] == len(exercise_frame)
```

</details>

## 9. 本讲小结

- Batch、Mini-batch 与 SGD 在每步样本量和梯度噪声上不同。
- One-hot 避免为无序类别强加数值顺序。
- 多项式与交互特征让线性参数表达非线性和条件关系。
- 复杂度属于整个特征管道；特征选择和预处理都必须留在验证边界内。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 14：Mini-batch、SGD、one-hot、多项式特征、复杂度与过拟合的知识边界。
- *Learning Data Science*，Ch.15、Ch.20：特征工程与优化主题。
- 本页能源站案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="14" next-label="第 15 讲：房产估价与公平性" next-href="/courses/data-science/15-fairness-case" />
