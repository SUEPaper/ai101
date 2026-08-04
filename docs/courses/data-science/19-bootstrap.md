---
course: data-science
title: Bootstrap 与参数推断
description: 用有放回重抽样近似估计量的抽样分布，构造不确定性区间并审查回归系数解释边界。
outline: [2, 3]
---

<DsLessonHero
  number="19"
  title="Bootstrap 与参数推断"
  eyebrow="看不到重复总体时，让样本扮演一个不完美的临时总体"
  duration="建议 120–150 分钟"
  :outcomes="[
    '解释 Bootstrap 为何必须等量、有放回地重抽样',
    '用 Bootstrap 分布估计标准误和百分位区间',
    '区分预测目标与系数推断，并识别 Bootstrap 的失败边界'
  ]"
/>

## 本讲要解决的问题

我们只有一次居民节电调查，却想知道样本均值或回归斜率换一份样本会变化多少。理想做法是从总体重复抽样；现实中总体不可见，Bootstrap 用观测样本近似它。

## 1. Bootstrap 的三步循环

给定大小为 `n` 的原样本：

1. 从这 `n` 行中有放回地抽 `n` 行。
2. 在重抽样本上重新计算目标统计量。
3. 重复 `B` 次，得到统计量的 Bootstrap 分布。

```python
import numpy as np
import pandas as pd

observed = np.array([12, 15, 15, 18, 21, 24], dtype=float)
rng = np.random.default_rng(42)

one_bootstrap_sample = rng.choice(observed, size=len(observed), replace=True)
print("原样本：", observed)
print("重抽样：", one_bootstrap_sample)
print("均值：", one_bootstrap_sample.mean())
```

有放回意味着某些原记录会重复，另一些不会出现；等量抽样让重抽统计量保持与原估计量相同的样本量条件。

<BootstrapLab />

## 2. Bootstrap 分布近似抽样分布

```python
def bootstrap_statistic(values, statistic, repetitions=5000, seed=42):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        resample = rng.choice(values, size=len(values), replace=True)
        estimates.append(statistic(resample))
    return np.asarray(estimates)

bootstrap_means = bootstrap_statistic(observed, np.mean)
print("原样本均值：", observed.mean())
print("Bootstrap 中心：", bootstrap_means.mean())
print("Bootstrap 标准误：", bootstrap_means.std(ddof=1))
```

Bootstrap 标准误是重抽统计量的标准差。它估计“若从原总体重新抽同样大小的样本，估计量会波动多少”。

<QuizBlock question="Bootstrap 为什么使用有放回抽样？" :options="['为了让样本变小', '让观测样本近似可重复抽取的总体分布', '保证每行恰好一次', '避免随机性']" :answer="1" explanation="若无放回地抽相同 n 行，每次都只是原样本的排列，统计量不会变化。" />

## 3. 百分位区间读取中间 95%

最直接的 95% Bootstrap 百分位区间取分布的 2.5% 与 97.5% 分位点。

```python
lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
print(f"95% Bootstrap 区间：[{lower:.2f}, {upper:.2f}]")
```

区间不是“参数有 95% 概率在里面”的贝叶斯概率陈述。在频率学派解释下，如果重复整个抽样和建区间流程，约 95% 的区间会覆盖固定参数——前提是方法与假设适用。

## 4. 对整行重抽，保留特征与目标对应关系

回归推断时不能分别抽 `X` 与 `y`。必须按观测行抽索引，再在每个重抽数据集上重新拟合。

```python
energy = pd.DataFrame({
    "temp": [18, 20, 22, 24, 27, 29, 31, 33, 26, 30],
    "peak_mwh": [42, 44, 47, 49, 55, 58, 62, 65, 54, 61],
})

def fit_slope(frame):
    x = frame["temp"].to_numpy(dtype=float)
    y = frame["peak_mwh"].to_numpy(dtype=float)
    return np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)

rng = np.random.default_rng(2026)
slopes = []
for _ in range(4000):
    rows = rng.integers(0, len(energy), size=len(energy))
    slopes.append(fit_slope(energy.iloc[rows]))

slope_interval = np.quantile(slopes, [0.025, 0.975])
print("样本斜率：", round(fit_slope(energy), 3))
print("斜率区间：", slope_interval.round(3))
```

## 5. 预测与推断问的是不同问题

| 目标 | 主要问题 | 评价重点 |
|---|---|---|
| 预测 | 新一天的峰值用电是多少 | 测试误差、校准、稳定性 |
| 参数推断 | 控制其他条件后，温度系数可能是多少 | 标准误、区间、模型假设 |
| 因果推断 | 如果主动改变温度或政策，结果会怎样 | 识别策略、混杂、干预定义 |

一个模型可以预测很好，但系数不适合因果解释；也可能系数研究设计清晰，但个体预测误差不低。

## 6. 混杂与共线性会改变系数含义

若夏季工作时长与温度同时增加，漏掉工作时长可能让温度系数吸收其影响。加入高度相关特征后，共线性又会放大系数不确定性。

```python
rng = np.random.default_rng(8)
temperature = rng.normal(28, 4, size=300)
operating_hours = 7 + 0.18 * temperature + rng.normal(0, 0.7, size=300)
load = 15 + 1.1 * temperature + 3.2 * operating_hours + rng.normal(0, 2, size=300)

X_omitted = np.column_stack([np.ones(len(load)), temperature])
X_full = np.column_stack([np.ones(len(load)), temperature, operating_hours])

beta_omitted, *_ = np.linalg.lstsq(X_omitted, load, rcond=None)
beta_full, *_ = np.linalg.lstsq(X_full, load, rcond=None)
print("遗漏工作时长：", beta_omitted.round(3))
print("包含工作时长：", beta_full.round(3))
```

Bootstrap 可以显示在现有数据与重抽方案下系数如何波动，但不能自动修复遗漏变量、错误测量或选择偏差。

## 7. Bootstrap 的失败边界

<div class="ds-compare">
  <div><b>样本不代表总体</b><span>重抽只能重复已有偏差</span><small>便利样本仍是便利样本</small></div>
  <div><b>样本太小或稀有事件缺失</b><span>经验分布看不到尾部</span><small>重抽不出从未观察的情况</small></div>
  <div><b>观测存在依赖</b><span>逐行重抽破坏时序或群组</span><small>需要 block / cluster bootstrap</small></div>
</div>

最大值、极端分位数、边界参数和非平滑统计量也可能得到糟糕近似。应结合问题结构、诊断和理论方法选择推断工具。

<div class="ds-warning"><strong>计算次数不是可信度</strong>把 B 从 5,000 增加到 500,000 只能降低 Monte Carlo 抖动，不能扩大原样本覆盖的信息。</div>

## 8. 知识检查

<QuizBlock question="无放回地从 n 行抽 n 行做 Bootstrap 会怎样？" :options="['每次统计量基本相同', '得到更多总体信息', '样本量翻倍', '自动纠正偏差']" :answer="0" explanation="无放回等量抽样只是重排全部原记录，均值等统计量不会变化。" />

<QuizBlock question="回归 Bootstrap 应如何抽样？" :options="['独立抽 X 和 y', '按行索引同时重抽特征和目标', '只抽残差且无需假设', '只重抽系数']" :answer="1" explanation="按行重抽保留每个观测内部的特征—目标配对。" />

<QuizBlock question="Bootstrap 能否修复便利样本的覆盖偏差？" :options="['能', '不能，它只重用观测到的样本', 'B 足够大就能', '只对均值能']" :answer="1" explanation="从未进入原样本的人在任何重抽中都不会出现。" />

<QuizBlock question="把重复次数 B 增加 100 倍，主要减少什么？" :options="['原始抽样偏差', 'Bootstrap 近似自身的 Monte Carlo 波动', '模型混杂', '标签错误']" :answer="1" explanation="更多重复让重抽分布的数值近似更平滑，但不增加原数据的信息。" />

<QuizBlock question="预测准确是否足以证明回归系数有因果含义？" :options="['足以', '不足，因果还需要识别假设与研究设计', '只要 R²>0.5 就足以', 'Bootstrap 后足以']" :answer="1" explanation="预测关联可以来自混杂或代理变量，因果需要额外设计与假设。" />

## 9. 代码练习：Bootstrap 中位数与均值

对一个右偏样本同时 Bootstrap 均值和中位数，报告原估计、标准误、95% 百分位区间，并比较哪个统计量的重抽分布更稳定。

<details><summary>提示</summary>

复用 `bootstrap_statistic`，分别传入 `np.mean` 和 `np.median`。稳定性可以用 Bootstrap 标准误比较。

</details>

<details><summary>参考实现</summary>

```python
right_skewed = np.array([2, 3, 3, 4, 4, 5, 6, 8, 13, 28], dtype=float)

reports = []
for name, statistic in [("mean", np.mean), ("median", np.median)]:
    distribution = bootstrap_statistic(right_skewed, statistic, repetitions=6000, seed=19)
    interval = np.quantile(distribution, [0.025, 0.975])
    reports.append({
        "statistic": name,
        "estimate": statistic(right_skewed),
        "standard_error": distribution.std(ddof=1),
        "lower": interval[0],
        "upper": interval[1],
    })

report = pd.DataFrame(reports)
print(report.round(3))
```

</details>

## 10. 本讲小结

- Bootstrap 从原样本等量、有放回重抽，近似估计量的抽样分布。
- 重抽分布的标准差估计标准误，分位点可构造百分位区间。
- 回归时按整行重抽；预测、参数推断与因果推断不能混为一谈。
- Bootstrap 重用已有信息，不能修复覆盖偏差、混杂、极小样本或错误依赖结构。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 19：Bootstrap、标准误、区间、回归系数推断与模型解释边界。
- *Learning Data Science*，Ch.17：Bootstrap 与推断主题。
- 本页能源与节电案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="19" next-label="第 20 讲：SQL 查询一张表" next-href="/courses/data-science/20-sql-basics" />
