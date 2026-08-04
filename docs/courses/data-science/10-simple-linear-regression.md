---
course: data-science
title: 建模与简单线性回归
description: 把模型理解为可检验的近似，用相关系数、残差和均方误差构造并检查简单线性回归。
outline: [2, 3]
---

<DsLessonHero
  number="10"
  title="建模与简单线性回归"
  eyebrow="模型不是现实的缩小版，而是为一个任务保留必要关系的函数"
  duration="建议 110–130 分钟"
  :outcomes="[
    '区分相关性描述与回归预测任务',
    '解释斜率、截距、预测值和残差的含义',
    '用 NumPy 从零计算 OLS 直线并检查残差'
  ]"
/>

## 本讲要解决的问题

河川市的一座社区能源站需要提前一天预测晚高峰用电量。历史记录显示，白天最高温度越高，空调用电通常越大。我们能否把温度变成一条足够实用的预测规则？

这不是寻找一条“绝对正确”的自然定律，而是在明确的输入、输出和误差标准下，构造一个可以被数据检验的近似。

## 1. 模型是一台有边界的预测机器

一个监督学习模型接收特征 `x`，输出对目标 `y` 的预测 `ŷ`。简单线性模型只允许输出落在一条直线上：

<div class="ds-equation"><span>输入</span><b>x = 最高温度</b><i>→</i><span>模型</span><b>ŷ = b₀ + b₁x</b><i>→</i><span>输出</span><b>晚高峰用电量</b></div>

- `b₀` 是截距：当 `x = 0` 时模型给出的基准预测。
- `b₁` 是斜率：`x` 每增加 1 个单位，预测平均变化多少。
- “线性”描述的是参数以加法组合，不代表现实关系永远是一条直线。

```python
import numpy as np
import pandas as pd

energy = pd.DataFrame({
    "max_temp": [18, 20, 22, 24, 27, 29, 31, 33],
    "peak_mwh": [42, 44, 47, 49, 55, 58, 62, 65],
})

def predict(x, intercept, slope):
    return intercept + slope * np.asarray(x)

energy["first_guess"] = predict(energy["max_temp"], 12, 1.55)
print(energy.head())
```

## 2. 相关系数先描述共同变化

相关系数把两个变量的线性共同变化压缩到 `[-1, 1]`。正值表示同向，负值表示反向，绝对值越接近 1，点云越接近某条直线。

```python
x = energy["max_temp"].to_numpy()
y = energy["peak_mwh"].to_numpy()

x_standard = (x - x.mean()) / x.std(ddof=0)
y_standard = (y - y.mean()) / y.std(ddof=0)
correlation = np.mean(x_standard * y_standard)

print(round(correlation, 3))
```

相关不回答三个问题：谁预测谁、预测错了多少、关系是否由第三个变量造成。回归需要把这三件事明确下来。

<QuizBlock
  question="温度与用电量相关系数为 0.88，能够直接推出什么？"
  :options="['温度必然导致全部用电变化', '两者在样本中有较强的正线性关系', '任意温度下预测都准确', '截距一定是 0']"
  :answer="1"
  explanation="相关系数描述样本中的线性共同变化，不单独证明因果，也不保证所有区间都适合线性外推。"
/>

## 3. 残差把“预测不好”变成可计算对象

第 `i` 个观测的残差是 `eᵢ = yᵢ - ŷᵢ`。正残差表示模型低估，负残差表示模型高估。均方误差把全部残差平方后取平均：

```python
def mse(actual, predicted):
    residuals = np.asarray(actual) - np.asarray(predicted)
    return np.mean(residuals ** 2)

guess_a = predict(x, intercept=12, slope=1.55)
guess_b = predict(x, intercept=18, slope=1.35)

print("方案 A：", round(mse(y, guess_a), 3))
print("方案 B：", round(mse(y, guess_b), 3))
```

<RegressionLineLab />

平方有两个作用：避免正负残差互相抵消，并让较大的错误受到更强惩罚。MSE 的单位是目标单位的平方；如果想回到原单位，可以取平方根得到 RMSE。

## 4. OLS 选择残差平方和最小的直线

普通最小二乘（OLS）寻找让训练数据 MSE 最小的 `b₀` 与 `b₁`。单特征情形有直接解：

```python
x_centered = x - x.mean()
y_centered = y - y.mean()

slope = np.sum(x_centered * y_centered) / np.sum(x_centered ** 2)
intercept = y.mean() - slope * x.mean()

energy["prediction"] = predict(x, intercept, slope)
energy["residual"] = y - energy["prediction"]

print(f"ŷ = {intercept:.2f} + {slope:.2f}x")
print("MSE：", round(mse(y, energy["prediction"]), 3))
```

这条公式还揭示了一件事：斜率由 `x` 与 `y` 的共同变化除以 `x` 自己的变化得到。若所有温度都相同，分母为 0，数据就无法告诉我们斜率。

<details><summary>第二层：为什么最优直线经过均值点？</summary>

固定斜率后，改变截距会让所有残差整体平移。最小值出现于残差和为 0，因此 `ȳ - b₀ - b₁x̄ = 0`，整理后得到 `b₀ = ȳ - b₁x̄`。所以带截距的 OLS 直线必然经过 `(x̄, ȳ)`。

</details>

## 5. 拟合之后必须看残差

只报告一个低 MSE 不够。残差图会暴露弯曲趋势、变化不均匀、异常点和时间结构。

```python
residual_summary = energy["residual"].agg(["mean", "std", "min", "max"])
print(residual_summary.round(3))
print("残差与温度相关：", round(np.corrcoef(x, energy["residual"])[0, 1], 6))
```

<figure class="ds-figure">
  <div class="ds-compare">
    <div><b>健康信号</b><span>残差围绕 0 无规律散开</span><small>线性结构已被模型解释了一部分</small></div>
    <div><b>弯曲图案</b><span>低温和高温同号，中间反号</span><small>可能需要非线性特征</small></div>
    <div><b>漏斗图案</b><span>x 越大，残差波动越大</span><small>误差尺度可能随输入变化</small></div>
  </div>
  <figcaption>残差不是拟合结束后的废料，它是下一轮建模最重要的诊断数据。</figcaption>
</figure>

<div class="ds-warning"><strong>不要随意外推</strong>样本温度只覆盖 18–33°C。把直线用于 45°C 或 -10°C，是在没有数据支持的区域假设关系仍然不变。</div>

## 6. 知识检查

<QuizBlock question="残差定义为 y - ŷ。某天残差为 +4，表示什么？" :options="['模型高估 4', '模型低估 4', '斜率为 4', '相关系数为 4']" :answer="1" explanation="实际值比预测值大 4，所以模型低估。" />

<QuizBlock question="为什么不能直接让残差平均值作为损失？" :options="['残差没有单位', '正负误差会互相抵消', '只能处理类别变量', '无法使用 NumPy']" :answer="1" explanation="高估和低估可能使平均残差接近 0，即使每个预测都很差。" />

<QuizBlock question="带截距的一元 OLS 直线经过哪个点？" :options="['原点', '(x 的均值, y 的均值)', '最大值点', '第一个观测点']" :answer="1" explanation="OLS 的最优截距使残差和为 0，因此直线经过均值点。" />

<QuizBlock question="残差图出现明显 U 形，优先说明什么？" :options="['样本一定有偏', '线性形式可能漏掉曲率', '相关就是因果', 'MSE 必然为 0']" :answer="1" explanation="系统弯曲说明残差仍含可预测结构，线性形式可能不足。" />

## 7. 代码练习：从零实现直线拟合

写函数 `fit_line(x, y)`，返回截距与斜率；再写 `rmse`，比较你的结果与 `np.polyfit(x, y, 1)`。

<details><summary>提示</summary>

先中心化 `x`、`y`；斜率等于中心化乘积之和除以 `x` 中心化平方和。注意 `np.polyfit` 返回顺序是斜率、截距。

</details>

<details><summary>参考实现</summary>

```python
def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    intercept = y.mean() - slope * x.mean()
    return intercept, slope

def rmse(actual, predicted):
    return np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2))

b0, b1 = fit_line(x, y)
numpy_slope, numpy_intercept = np.polyfit(x, y, 1)

assert np.allclose([b0, b1], [numpy_intercept, numpy_slope])
print(b0, b1, rmse(y, predict(x, b0, b1)))
```

</details>

## 8. 本讲小结

- 相关系数描述线性共同变化；回归进一步指定预测方向与损失。
- 残差是实际值减预测值，MSE 是残差平方的平均。
- 一元 OLS 的斜率由共同变化除以特征自身变化得到。
- 拟合之后要检查残差结构和数据支持范围。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 10：模型、相关、简单线性回归、损失与残差的知识边界。
- *Learning Data Science*，Ch.4、Ch.15：建模与线性模型主题。
- 本页能源站案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="10" next-label="第 11 讲：损失函数与数据变换" next-href="/courses/data-science/11-loss-transformations" />
