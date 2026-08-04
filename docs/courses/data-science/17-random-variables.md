---
course: data-science
title: 随机变量
description: 从随机过程到数值结果，理解分布、期望、方差、协方差、样本均值与中心极限定理。
outline: [2, 3]
---

<DsLessonHero
  number="17"
  title="随机变量"
  eyebrow="随机变量不是一个不断乱跳的数，而是把随机结果映射成数值的规则"
  duration="建议 120–150 分钟"
  :outcomes="[
    '区分随机变量、观测值与概率分布',
    '计算离散变量的期望和方差',
    '解释独立样本均值为何随 n 增大而稳定，并理解 CLT 的适用条件'
  ]"
/>

## 本讲要解决的问题

第 09 讲看到，不同随机样本会给出不同满意度均值。本讲把这种“重复做同一随机过程可能得到什么”形式化，为估计量与 Bootstrap 奠定语言。

## 1. 三层对象不要混在一起

假设随机抽一户，记录其是否在晚高峰减少用电：

<div class="ds-equation"><span>随机过程</span><b>从总体抽一户</b><i>→</i><span>随机变量 X</span><b>节电记 1，否则 0</b><i>→</i><span>一次观测</span><b>X = 1</b></div>

- 随机变量 `X`：结果到数值的映射规则。
- 分布：每个可能数值及其概率。
- 观测值 `x`：某次过程真正得到的数。

一次观测后数字已经确定，但在重复过程之前，我们用分布描述不确定性。

## 2. Bernoulli 是最小的离散随机变量

若节电概率为 `p`，则 `X=1` 的概率是 `p`，`X=0` 的概率是 `1-p`。

```python
import numpy as np
import pandas as pd

p = 0.35
values = np.array([0, 1])
probabilities = np.array([1 - p, p])

expectation = np.sum(values * probabilities)
variance = np.sum((values - expectation) ** 2 * probabilities)

print("E[X]：", expectation)
print("Var(X)：", variance)
print("p(1-p)：", p * (1 - p))
```

期望是长期重复的概率加权平均，不保证随机变量一定能取到该值。公平骰子的期望是 3.5，但任何一次都不会掷出 3.5。

<QuizBlock question="随机变量的期望必须是它可能取得的值吗？" :options="['必须', '不必须，例如骰子期望 3.5', '只对 Bernoulli 必须', '取决于列名']" :answer="1" explanation="期望是概率加权中心，可以位于可能值之间。" />

## 3. 方差测量离期望的平方距离

方差是 `E[(X-E[X])²]`。平方使两侧偏差不抵消，并更强调远离中心的结果。标准差是方差的平方根，回到原单位。

```python
demand_values = np.array([40, 50, 70], dtype=float)
demand_probabilities = np.array([0.2, 0.5, 0.3])

demand_mean = np.sum(demand_values * demand_probabilities)
demand_variance = np.sum((demand_values - demand_mean) ** 2 * demand_probabilities)

print("期望：", demand_mean)
print("标准差：", np.sqrt(demand_variance))
```

`E[X²] - E[X]²` 是同一个方差公式的展开形式，但在浮点计算中，对极大且彼此接近的数直接相减可能损失精度。

## 4. 随机变量相加时，期望总能相加

无论 `X` 与 `Y` 是否独立，`E[X+Y] = E[X] + E[Y]`。方差相加则需要考虑协方差：

`Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)`

```python
rng = np.random.default_rng(7)
temperature = rng.normal(28, 4, size=5000)
cooling = 2.2 * temperature + rng.normal(0, 3, size=5000)

covariance = np.mean((temperature - temperature.mean()) * (cooling - cooling.mean()))
correlation = covariance / (temperature.std(ddof=0) * cooling.std(ddof=0))

print("协方差：", round(covariance, 2))
print("相关系数：", round(correlation, 3))
```

协方差保留单位，改变单位会改变数值；相关系数经过标准化，没有单位。

## 5. 样本均值是随机变量

抽样前，未来样本中的每个 `Xᵢ` 和它们的均值都不确定。若 `X₁...Xₙ` 独立同分布，均值为 `μ`、方差为 `σ²`，则：

- `E[X̄] = μ`
- `Var(X̄) = σ²/n`
- `SD(X̄) = σ/√n`

样本量扩大 4 倍，样本均值标准差约缩小一半，而不是四分之一。

<RandomVariableLab />

滑动 `n` 时，直方图会变窄；中心仍在 `p` 附近。这里展示的是 60 次“样本均值”，不是 60 个单次 Bernoulli 结果。

## 6. 中心极限定理描述均值分布的形状

在独立同分布、方差有限等条件下，当 `n` 足够大时，标准化的样本均值分布趋近正态，即使原始变量不是正态。

```python
rng = np.random.default_rng(123)

def repeated_exponential_means(sample_size, repeats=5000):
    samples = rng.exponential(scale=2.0, size=(repeats, sample_size))
    return samples.mean(axis=1)

for sample_size in [1, 5, 30, 100]:
    means = repeated_exponential_means(sample_size)
    print(sample_size, round(means.mean(), 3), round(means.std(ddof=0), 3))
```

原始指数分布明显右偏；随着 `n` 增大，均值分布会更对称、更集中。

<div class="ds-warning"><strong>CLT 不是“任何数据都自动正态”</strong>强依赖、无限方差、极端重尾或样本量太小都会让正态近似变差。它描述统计量在重复抽样中的分布，不会把原始数据改造成正态。</div>

## 7. iid 是强假设，不是默认事实

“独立同分布”包含两部分：

- 同分布：每个观测来自相同概率机制。
- 独立：知道一个观测不会改变另一个观测的分布。

同一栋楼住户的用电、连续日期的负荷、同一个用户的重复点击往往相关。把它们当 iid 会低估均值的不确定性。

```python
daily = np.array([52, 55, 57, 58, 61, 64, 66, 65], dtype=float)
lag_correlation = np.corrcoef(daily[:-1], daily[1:])[0, 1]
print("相邻日期相关：", round(lag_correlation, 3))
```

实际分析可按楼栋聚类抽样、按时间分块或显式建立依赖模型。

## 8. 知识检查

<QuizBlock question="Bernoulli(p) 的期望是什么？" :options="['0', 'p', '1-p', 'p²']" :answer="1" explanation="取值 1 的概率为 p、取值 0 的概率为 1-p，所以加权平均为 p。" />

<QuizBlock question="独立同分布样本量从 25 增加到 100，样本均值标准差约怎样变化？" :options="['变为一半', '变为四分之一', '增加四倍', '不变']" :answer="0" explanation="标准差按 1/√n 缩放，样本量四倍对应标准差一半。" />

<QuizBlock question="X 与 Y 不独立时，E[X+Y] 是否仍等于 E[X]+E[Y]？" :options="['是', '否', '只有协方差为 1 时', '只有正态时']" :answer="0" explanation="期望的线性不要求独立；方差相加才需要协方差项。" />

<QuizBlock question="中心极限定理主要描述什么？" :options="['原始数据一定正态', '适当条件下样本均值的标准化分布趋近正态', '所有异常值消失', '随机变量变成常数']" :answer="1" explanation="CLT 的对象是和或均值等统计量的重复抽样分布。" />

## 9. 代码练习：验证均值方差公式

模拟 Bernoulli(0.3)，对 `n = 4, 16, 64, 256` 各重复 10,000 次。报告样本均值的模拟均值、模拟标准差和理论标准差 `sqrt(p(1-p)/n)`。

<details><summary>提示</summary>

一次生成形状为 `(10000, n)` 的二项分布数组，再沿 `axis=1` 求均值。

</details>

<details><summary>参考实现</summary>

```python
rng = np.random.default_rng(2026)
p = 0.3
reports = []

for n in [4, 16, 64, 256]:
    samples = rng.binomial(1, p, size=(10_000, n))
    sample_means = samples.mean(axis=1)
    reports.append({
        "n": n,
        "simulated_mean": sample_means.mean(),
        "simulated_sd": sample_means.std(ddof=0),
        "theoretical_sd": np.sqrt(p * (1 - p) / n),
    })

report = pd.DataFrame(reports)
print(report.round(4))
```

</details>

## 10. 本讲小结

- 随机变量是从随机结果到数值的映射，分布描述其可能值与概率。
- 期望是概率加权中心，方差测量围绕期望的平方波动。
- 期望总能线性相加；方差相加还要考虑协方差。
- iid 样本均值的标准差按 `1/√n` 缩小，CLT 描述其渐近形状。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 17：随机变量、分布、期望、方差、和、协方差、iid、样本均值与 CLT 的知识边界。
- *Learning Data Science*，Ch.3、Appendix A1：概率与抽样分布主题。
- 本页能源与节电案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="17" next-label="第 18 讲：估计量、偏差与方差" next-href="/courses/data-science/18-estimators" />
