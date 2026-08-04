---
course: data-science
title: 抽样与偏差
description: 区分目标总体、抽样框和样本，比较简单随机、分层和便利抽样，并理解样本量无法修复系统偏差。
outline: [2, 3]
---

<DsLessonHero
  number="09"
  title="抽样与偏差"
  eyebrow="你能对谁下结论，首先取决于谁有机会进入样本"
  duration="建议 100–120 分钟"
  :outcomes="[
    '区分目标总体、抽样框、样本、参数和统计量',
    '判断一个方案是否为概率样本或简单随机样本',
    '解释覆盖偏差、无应答偏差与便利抽样为何不能靠扩大样本自动修复'
  ]"
/>

## 本讲要解决的问题

城市希望估计全体居民对公共交通的满意度。运营方在地铁中心站放置二维码，收到了 20,000 份问卷。样本很大，但它可能主要代表经常乘地铁、经过中心站、愿意扫码的人。

数据量解决的是随机波动，不自动解决“谁根本没有机会被抽到”的系统偏差。

## 1. 抽样问题的五个对象

| 对象 | 满意度调查中的含义 |
|---|---|
| 目标总体 | 希望下结论的全体城市居民 |
| 抽样框 | 实际可以被抽取的居民名单或住址清单 |
| 样本 | 真正被选中并进入调查的人 |
| 参数 | 全体居民的真实平均满意度，通常未知 |
| 统计量 | 样本平均满意度，用来估计参数 |

<figure class="ds-figure">
  <div class="ds-flow">
    <div><b>TARGET</b><span>想了解的总体</span><small>结论希望覆盖谁</small></div>
    <div><b>FRAME</b><span>实际可抽取名单</span><small>谁没有进入名单</small></div>
    <div><b>INVITED</b><span>被选中的单位</span><small>抽取概率是否已知</small></div>
    <div><b>RESPONDED</b><span>真正回应的人</span><small>无应答是否有规律</small></div>
    <div><b>ESTIMATE</b><span>从样本计算统计量</span><small>波动、权重与范围</small></div>
  </div>
  <figcaption>偏差可以出现在抽样框、邀请和回应的任何一步。最终表格再大，也无法显示从未进入数据的人。</figcaption>
</figure>

## 2. 概率样本与简单随机样本

概率样本要求总体中的每个单位都有已知且非零的入样概率。简单随机样本（SRS）更严格：所有相同大小的样本集合有相同概率被选中。

```python
import numpy as np
import pandas as pd

population = pd.DataFrame({
    "resident_id": np.arange(1, 101),
    "district": ["中心区"] * 50 + ["近郊"] * 30 + ["远郊"] * 20,
    "satisfaction": [84] * 50 + [72] * 30 + [56] * 20,
})

srs = population.sample(n=20, random_state=42)

print("总体均值：", population["satisfaction"].mean())
print("SRS 估计：", srs["satisfaction"].mean())
```

随机化不会保证某一次样本完美代表总体。它的价值是让抽样误差可以用概率描述，并避免研究者按结果挑选样本。

<SamplingLab />

## 3. 三类常见偏差

### 覆盖偏差

抽样框漏掉目标总体的一部分。例如只使用地铁乘客名单，会漏掉主要使用公交、步行或私家车的人。

### 选择偏差

样本进入机制与研究变量相关。例如只在中心站扫码，位置本身就改变了谁更容易进入样本。

### 无应答偏差

被抽中的人与真正回应的人存在系统差异。满意或极度不满的人可能更愿意回答。

<div class="ds-warning">
  <strong>更大样本不会修复固定方向的偏差</strong>
  如果便利样本持续过度代表中心区，样本量从 2,000 增加到 20,000 只会更精确地估计“中心区便利样本”的均值。
</div>

## 4. 分层抽样保障关键群体被覆盖

分层抽样先按已知特征划分总体，再在每层内随机抽样。它适合确保小群体进入样本，也可能降低估计方差。

```python
stratified = (
    population
    .groupby("district", group_keys=False)
    .sample(frac=0.2, random_state=42)
)

print(stratified["district"].value_counts())
print(stratified["satisfaction"].mean())
```

这里每个区域抽取 20%，所以样本结构与总体比例一致。如果各层使用不同抽样率，计算总体估计时需要使用权重。

<QuizBlock
  question="分层抽样最直接保证了什么？"
  :options="['每次样本均值都等于总体均值', '预先定义的关键群体按方案进入样本', '不存在无应答', '所有变量都服从正态分布']"
  :answer="1"
  explanation="分层设计控制各层抽取数量或比例，但仍可能有随机误差和无应答问题。"
/>

## 5. 事后分层可以校正已知构成差异

如果样本的区域构成与已知总体构成不同，可以按总体比例重新加权：

```python
sample_summary = pd.DataFrame({
    "district": ["中心区", "近郊", "远郊"],
    "sample_mean": [84, 72, 56],
    "sample_share": [0.80, 0.15, 0.05],
    "population_share": [0.50, 0.30, 0.20],
})

sample_summary["weight"] = (
    sample_summary["population_share"] / sample_summary["sample_share"]
)
weighted_mean = (
    sample_summary["sample_mean"] * sample_summary["population_share"]
).sum()

print(sample_summary)
print("事后分层估计：", weighted_mean)
```

事后分层只能按已经知道且已测量的变量校正。若远郊样本内部仍只包含最活跃的受访者，区域权重不能消除这种层内偏差。

## 6. 抽样误差与非抽样误差

- 抽样误差：随机抽取不同样本造成的统计量波动，可通过概率模型和更大样本减小。
- 非抽样误差：覆盖、无应答、测量、录入或问题措辞造成的系统问题，扩大相同机制的样本通常无效。

```python
estimates = []
for seed in range(200):
    sample = population.sample(n=20, random_state=seed)
    estimates.append(sample["satisfaction"].mean())

print(pd.Series(estimates).describe())
```

这组 200 个估计值展示了“如果重复抽样，样本均值会如何变化”。第 17–19 讲会用随机变量、估计量和 Bootstrap 更正式地描述这种变化。

<div class="ds-stop">
  <strong>停下来想一想</strong>
  如果调查问题写成“你是否支持继续改善广受欢迎的公交服务？”，即使抽样设计完美，问题措辞仍会引入测量偏差。
</div>

## 7. 知识检查

<QuizBlock
  question="目标总体是全体居民，但抽样框只有注册公交卡用户。主要风险是什么？"
  :options="['覆盖偏差', '分箱太多', '正则表达式错误', '过拟合']"
  :answer="0"
  explanation="没有公交卡的居民没有机会被抽到，抽样框未覆盖目标总体。"
/>

<QuizBlock
  question="简单随机样本是否保证每个个体入样概率相同？"
  :options="['是', '否', '只在样本量大于 1000 时', '只对类别变量']"
  :answer="0"
  explanation="SRS 的每个个体入样概率相同，并且所有同样大小的样本集合等概率。"
/>

<QuizBlock
  question="便利样本从 2,000 扩大到 20,000，什么最可能下降？"
  :options="['由固定覆盖机制产生的偏差', '相同机制下的随机波动', '所有测量错误', '目标总体规模']"
  :answer="1"
  explanation="更大样本通常降低随机波动，但不会自动改变谁更容易进入样本。"
/>

<QuizBlock
  question="事后分层能够校正什么？"
  :options="['所有未知偏差', '样本与总体在已知分层变量上的构成差异', '错误的问题措辞', '伪造记录']"
  :answer="1"
  explanation="权重依赖已知总体比例，只能校正所使用分层变量上的构成差异。"
/>

<QuizBlock
  question="随机抽样的核心价值是什么？"
  :options="['保证一次样本与总体完全相同', '让入样机制可描述，并可量化抽样波动', '消除所有缺失值', '让任何相关都成为因果']"
  :answer="1"
  explanation="随机化使样本选择机制透明且可用概率研究，但单次样本仍有波动。"
/>

## 8. 代码练习：比较三种抽样设计

基于本讲的 `population`，分别构造：

1. 前 20 行组成的便利样本。
2. 20 人的 SRS。
3. 每区各抽 10 人的等额分层样本。
4. 比较三个样本的原始均值；为等额分层样本按总体区域比例加权。

<details><summary>提示</summary>

等额分层样本的原始均值会让三个区域各占三分之一，但总体中三个区域比例是 0.5、0.3、0.2，需要对各区样本均值按总体比例求加权和。

</details>

<details><summary>参考实现</summary>

```python
convenience = population.head(20)
srs = population.sample(n=20, random_state=7)
equal_strata = (
    population
    .groupby("district", group_keys=False)
    .sample(n=10, random_state=7)
)

stratum_means = equal_strata.groupby("district")["satisfaction"].mean()
shares = pd.Series({"中心区": 0.5, "近郊": 0.3, "远郊": 0.2})
weighted = (stratum_means * shares).sum()

print("总体：", population["satisfaction"].mean())
print("便利样本：", convenience["satisfaction"].mean())
print("SRS：", srs["satisfaction"].mean())
print("等额分层原始均值：", equal_strata["satisfaction"].mean())
print("等额分层加权均值：", weighted)
```

</details>

## 9. 本讲小结

- 结论范围由目标总体、抽样框和实际回应机制共同决定。
- 概率样本让入样概率可知，SRS 进一步让同样大小的样本集合等概率。
- 样本量主要减少随机波动，不能自动修复系统偏差。
- 分层和事后加权需要明确层构成与权重依据。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 09：总体、调查、概率抽样、分层和事后分层的知识边界。
- *Learning Data Science*，Ch.3：抽样设计与抽样波动主题。
- 本页城市交通满意度案例、文字、图解、代码、问题和练习均为本课程原创。

<DsLessonProgress lesson-id="09" next-label="第 10 讲：建模与简单线性回归" next-href="/courses/data-science/10-simple-linear-regression" />
