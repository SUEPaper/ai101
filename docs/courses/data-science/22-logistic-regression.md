---
course: data-science
title: 逻辑回归
description: 用 Sigmoid 把线性分数转成概率，以 Log odds 和交叉熵理解逻辑回归及其梯度训练。
outline: [2, 3]
---

<DsLessonHero
  number="22"
  title="逻辑回归"
  eyebrow="先预测正类概率，再把概率交给具体决策场景"
  duration="建议 120–140 分钟"
  :outcomes="[
    '解释 Sigmoid 如何把任意分数压到 0–1',
    '把逻辑回归系数解释为 Log odds 的变化',
    '从零实现交叉熵和逻辑回归梯度下降'
  ]"
/>

## 本讲要解决的问题

运维团队希望预测一个单车站点明天是否会发生“无车或满桩超过 30 分钟”的服务故障。目标只有 `0/1`，但模型最好先输出概率，让第 23 讲根据不同后果选择阈值。

## 1. 线性回归不适合直接输出概率

线性模型 `β₀ + β₁x` 可以小于 0 或大于 1；误差方差也会随概率改变。逻辑回归保留线性分数，再用 Sigmoid 映射：

`p = σ(z) = 1 / (1 + exp(-z))`，其中 `z = Xβ`。

```python
import numpy as np
import pandas as pd

def sigmoid(score):
    score = np.clip(score, -500, 500)
    return 1 / (1 + np.exp(-score))

scores = np.array([-4, -1, 0, 1, 4], dtype=float)
print(pd.DataFrame({"score": scores, "probability": sigmoid(scores)}))
```

Sigmoid 单调递增，`z=0` 时概率为 0.5；分数很大或很小时，概率逐渐靠近 1 或 0，但不会越界。

<LogisticCurveLab />

<QuizBlock question="逻辑回归的线性部分 Xβ 可以取什么范围？" :options="['只能 0–1', '任意实数，再经 Sigmoid 变成概率', '只能正数', '只能整数']" :answer="1" explanation="受限到 0–1 的是 Sigmoid 输出，不是线性分数。" />

## 2. Log odds 让概率重新变成线性

Odds（胜算）是 `p/(1-p)`；Log odds 是其对数。逻辑回归假设：

`log(p/(1-p)) = β₀ + β₁x₁ + ... + βₚxₚ`

当其他特征保持不变，`xⱼ` 增加 1，Log odds 增加 `βⱼ`，Odds 乘以 `exp(βⱼ)`。

```python
beta = np.array([-2.0, 0.8])
x_value = 5.0
score = beta[0] + beta[1] * x_value
probability = sigmoid(score)
odds = probability / (1 - probability)

print("score / log odds：", score)
print("probability：", probability)
print("odds：", odds)
print("x 增加 1，odds 倍数：", np.exp(beta[1]))
```

概率变化不是固定的：同一个 β 在 p 接近 0.5 时改变概率最多，在两端改变较少。因此不要把 β 直接说成“概率增加 β”。

## 3. 交叉熵奖励正确且自信，重罚错误且自信

对单个标签 `y ∈ {0,1}`，二元交叉熵为：

`- [y log(p) + (1-y) log(1-p)]`

```python
def binary_cross_entropy(y, probability):
    probability = np.clip(np.asarray(probability), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return -np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))

y_example = np.array([1, 1, 0, 0])
good = np.array([0.9, 0.7, 0.2, 0.1])
confident_wrong = np.array([0.1, 0.2, 0.8, 0.9])

print("合理概率：", binary_cross_entropy(y_example, good))
print("自信但错误：", binary_cross_entropy(y_example, confident_wrong))
```

对数在 `p→0` 时趋向负无穷，所以把真实正类预测为几乎 0 会受到巨大惩罚。代码中裁剪概率用于避免 `log(0)`。

## 4. 梯度仍是“预测减真实”

逻辑回归平均交叉熵对 β 的梯度为 `(1/n) Xᵀ(p-y)`。形式与线性回归相似，但 `p` 先经过 Sigmoid。

```python
service = pd.DataFrame({
    "imbalance": [0.1, 0.2, 0.35, 0.4, 0.55, 0.6, 0.72, 0.85, 0.9, 0.95],
    "nearby_event": [0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
    "failure": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
})

X = np.column_stack([
    np.ones(len(service)),
    service["imbalance"],
    service["nearby_event"],
])
y = service["failure"].to_numpy(dtype=float)

beta = np.zeros(X.shape[1])
rate = 0.8
for _ in range(3000):
    probability = sigmoid(X @ beta)
    gradient = X.T @ (probability - y) / len(y)
    beta -= rate * gradient

print("beta：", beta.round(3))
print("loss：", binary_cross_entropy(y, sigmoid(X @ beta)))
```

这份小数据近似可分，未正则化系数可能继续增大，把训练概率推向 0 和 1。实践中通常用 L2 正则化稳定系数。

## 5. 二维决策边界是一条直线

若用阈值 0.5，边界满足 `p=0.5`，也就是 `Xβ=0`。两个特征下：

`β₀ + β₁x₁ + β₂x₂ = 0`

这是特征空间中的直线；更多维时是超平面。逻辑回归的概率曲面非线性，但 0.5 决策边界对原始线性特征仍是线性的。

<figure class="ds-figure">
  <div class="ds-compare">
    <div><b>低分区</b><span>z &lt; 0，p &lt; 0.5</span><small>默认判为负类</small></div>
    <div><b>边界</b><span>z = 0，p = 0.5</span><small>系数决定位置与方向</small></div>
    <div><b>高分区</b><span>z &gt; 0，p &gt; 0.5</span><small>默认判为正类</small></div>
  </div>
</figure>

加入多项式特征后，模型仍对参数线性，但在原始坐标中可以得到弯曲边界。

## 6. 概率质量与分类正确是两件事

两个模型可能在 0.5 阈值下预测相同类别，但一个给 0.51，另一个给 0.99。若真实正类比例只有约一半，后者明显过度自信。

```python
probability = sigmoid(X @ beta)
prediction = (probability >= 0.5).astype(int)
accuracy = np.mean(prediction == y)

print(pd.DataFrame({"probability": probability, "prediction": prediction, "actual": y}).round(3))
print("accuracy：", accuracy)
```

交叉熵评价概率；Accuracy 评价给定阈值后的标签。下一讲会系统拆解阈值指标与校准。

## 7. 知识检查

<QuizBlock question="Sigmoid 输入为 0 时输出多少？" :options="['0', '0.5', '1', '无穷大']" :answer="1" explanation="1/(1+exp(0))=1/2。" />

<QuizBlock question="β=0.7 表示 x 增加 1 时什么乘以 exp(0.7)？" :options="['概率', 'Odds', '样本量', '交叉熵']" :answer="1" explanation="逻辑回归系数在线性 Log odds 尺度上相加，因此 Odds 按 exp(β) 倍增。" />

<QuizBlock question="真实 y=1，哪个概率会产生最大交叉熵？" :options="['0.99', '0.8', '0.5', '0.01']" :answer="3" explanation="对正类损失是 -log(p)，接近 0 的错误自信预测受到最大惩罚。" />

<QuizBlock question="两个原始特征、无多项式变换时，0.5 决策边界是什么？" :options="['直线', '圆', '任意曲线', '一个概率值']" :answer="0" explanation="p=0.5 等价于 Xβ=0，是二维特征空间中的直线。" />

## 8. 代码练习：带 L2 的逻辑回归

实现 `fit_logistic_l2(X, y, strength)`。交叉熵梯度后增加 L2 梯度 `strength * beta / n`，但不惩罚截距；返回 β 和损失历史。

<details><summary>提示</summary>

复制 beta 为 penalty，令 `penalty[0]=0`。训练损失应同时记录交叉熵和 `strength/(2n)*sum(beta[1:]**2)`。

</details>

<details><summary>参考实现</summary>

```python
def fit_logistic_l2(X, y, strength=1.0, rate=0.5, steps=3000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    history = []
    for _ in range(steps):
        probability = sigmoid(X @ beta)
        penalty = beta.copy()
        penalty[0] = 0
        gradient = X.T @ (probability - y) / len(y) + strength * penalty / len(y)
        beta -= rate * gradient
        objective = binary_cross_entropy(y, probability) + strength * np.sum(beta[1:] ** 2) / (2 * len(y))
        history.append(objective)
    return beta, np.asarray(history)

beta_l2, history = fit_logistic_l2(X, y, strength=1.0)
print(beta_l2.round(3), history[-1])
assert history[-1] < history[0]
```

</details>

## 9. 本讲小结

- 逻辑回归用 Sigmoid 把线性分数映射成正类概率。
- 系数在线性 Log odds 尺度上解释，`exp(β)` 是 Odds 倍数。
- 交叉熵重罚自信且错误的概率，梯度可用下降法优化。
- 概率预测与阈值后的类别决策需要分别评价。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 22：分类、Sigmoid、Log odds、交叉熵、梯度和决策边界的知识边界。
- *Learning Data Science*，Ch.19–20：逻辑回归与模型训练主题。
- 本页站点服务故障案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="22" next-label="第 23 讲：分类指标与阈值" next-href="/courses/data-science/23-classification-metrics" />
