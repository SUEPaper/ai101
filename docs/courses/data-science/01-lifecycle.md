---
course: data-science
title: 数据科学生命周期
description: 从一个共享单车问题出发，理解数据科学如何在问题、数据、模型和决策之间循环。
outline: [2, 3]
---

<DsLessonHero
  number="01"
  title="数据科学生命周期"
  eyebrow="先问清楚要改变什么，再决定计算什么"
  duration="建议 75–90 分钟"
  :outcomes="[
    '把模糊需求改写成对象、指标和行动都明确的问题',
    '识别数据获取、探索、建模和沟通之间的反馈关系',
    '区分预测、解释和因果问题，并说明分析的证据边界'
  ]"
/>

## 本讲要解决的问题

一座城市收到大量投诉：“晚高峰总是借不到共享单车。”如果你直接打开数据画图，很可能得到很多图，却没有得到可以执行的答案。数据科学首先要做的不是选择算法，而是把抱怨变成清楚的问题。

<div class="ds-stop">
  <strong>先写下你的版本</strong>
  “改善晚高峰体验”至少缺少三个信息：在哪里、什么时间、用什么指标判断改善。先尝试补全，再继续阅读。
</div>

一个可行动的版本是：

> 在未来四周，哪些站点在工作日 18:00–19:00 出现“可借车辆少于 2 辆”的比例最高？如果提前 30 分钟调入车辆，这个比例是否下降？

这句话限定了观察对象、时间范围、指标和可能行动。后续每一步都可以回到它检查方向是否偏离。

## 1. 生命周期不是流水线

<figure class="ds-figure">
  <div class="ds-flow">
    <div><b>QUESTION</b><span>定义问题</span><small>谁、何时、什么指标、为了什么行动</small></div>
    <div><b>DATA</b><span>取得数据</span><small>来源、粒度、权限与质量</small></div>
    <div><b>UNDERSTAND</b><span>理解数据</span><small>分布、异常、关系与缺口</small></div>
    <div><b>MODEL</b><span>预测或推断</span><small>基线、误差、稳定性与限制</small></div>
    <div><b>ACT</b><span>形成行动</span><small>沟通、部署、监控并重新提问</small></div>
  </div>
  <figcaption>箭头表示最常见的推进方向，不代表只能向前。探索中发现关键字段缺失，就必须回到数据获取；结果无法支持决策，就必须重写问题。</figcaption>
</figure>

真实项目经常出现以下回路：

- 数据粒度太粗，无法判断具体时段，于是返回数据获取。
- 预测准确但无法提前干预，于是返回问题定义。
- 总体指标改善，但某些区域变差，于是重新分组探索。
- 模型上线后行为发生变化，于是重新采集数据。

<LifecycleLab />

## 2. 把问题拆成可验证的部件

一个数据问题至少需要四个部件：

| 部件 | 共享单车案例 | 如果缺失会怎样 |
|---|---|---|
| 对象 | 每个站点的每个工作日晚高峰 | 不知道一行数据代表什么 |
| 指标 | 可借车辆少于 2 辆的分钟占比 | “体验差”无法计算 |
| 比较 | 调度前后，或相似站点之间 | 无法判断变化是否有意义 |
| 行动 | 提前调车、调整运维班次 | 分析结果无法落地 |

这四个部件还能帮助你拒绝伪问题。例如“哪个模型最先进”不是一个完整的数据问题，因为它没有说明服务对象和成功标准。

<QuizBlock
  question="下面哪个问题最适合直接进入数据获取阶段？"
  :options="['怎样让城市交通更好', '哪个机器学习模型最强', '过去四周哪些站点在工作日 18:00–19:00 的缺车分钟占比最高', '数据里有什么有趣的东西']"
  :answer="2"
  explanation="它明确了对象、时间窗口和可计算指标。其他选项仍需要继续收窄。"
/>

## 3. 一行数据代表什么

数据的**粒度**是每一行代表的观察单位。共享单车系统可能导出三种表：

- 每次骑行一行：适合分析出发地、目的地和时长。
- 每个站点每分钟一行：适合分析缺车持续时间。
- 每个区域每天一行：适合看长期趋势，但无法定位具体站点和时段。

如果研究问题关注“缺车分钟占比”，使用每日区域汇总表就不够。数据量大不能弥补粒度错误。

```python
import pandas as pd

station_minutes = pd.DataFrame({
    "station": ["湖畔", "湖畔", "剧场", "剧场"],
    "time": ["18:00", "18:01", "18:00", "18:01"],
    "bikes_available": [1, 0, 5, 4],
})

station_minutes["is_shortage"] = station_minutes["bikes_available"] < 2
shortage_rate = station_minutes.groupby("station")["is_shortage"].mean()

print(shortage_rate)
```

这段代码把业务定义直接写进数据：`bikes_available < 2`。如果运营团队后来认为少于 3 辆才算风险，定义和结果都要重新计算。

<div class="ds-stop">
  <strong>停下来想一想</strong>
  `mean()` 为什么能计算布尔列的占比？在 Python 数值运算中，`True` 可以作为 1，`False` 可以作为 0；平均值因此等于 `True` 的比例。
</div>

## 4. 探索、预测与推断回答不同问题

### 探索：发生了什么

探索性数据分析关注分布、异常和关系。例如：缺车是否集中在少数站点？雨天的记录是否明显更少？探索可以发现线索，但不能自动证明原因。

### 预测：接下来可能发生什么

预测问题是：“根据现在的信息，某站点 30 分钟后缺车的概率是多少？”评价重点是对新数据的表现，而不是某个系数是否显著。

### 推断：关系有多大、有多确定

推断问题是：“在其他条件相似时，提前调入 5 辆车与缺车比例下降多少相关？”如果进一步问“调车是否导致下降”，就进入因果问题，需要更强的研究设计。

<figure class="ds-figure">
  <div class="ds-anatomy">
    <span class="is-index">问题类型</span><span class="is-column">探索</span><span class="is-column">预测</span><span class="is-column">推断</span>
    <span class="is-index">核心动词</span><span>发现</span><span>预估</span><span>量化关系</span>
    <span class="is-index">主要证据</span><span>图形与统计量</span><span>留出数据误差</span><span>估计值与不确定性</span>
    <span class="is-index">常见误区</span><span>把相关当原因</span><span>只看训练误差</span><span>忽略混杂因素</span>
  </div>
  <figcaption>同一个数据集可以支持多种问题，但每种问题需要不同的评价标准。</figcaption>
</figure>

## 5. 数据科学结论必须带着限制一起交付

假设模型发现大学城站点最容易缺车。你仍然要追问：

- 传感器故障是否更常发生在这个站点？
- 暑假数据能否代表开学后的需求？
- 调度优先照顾骑行量大的区域，会不会让边缘区域更难借车？
- 模型预测“高风险”之后，谁负责行动，多久行动一次？

一份可信的分析报告至少包含：数据来自哪里、如何定义指标、使用了什么比较、结果对哪些范围有效、有哪些替代解释，以及下一步如何监控。

## 6. 知识检查

<QuizBlock
  question="分析时发现站点状态每 30 分钟才记录一次，无法计算缺车分钟占比。下一步最合理的动作是什么？"
  :options="['换一个更复杂的模型', '返回数据获取，寻找更细粒度记录', '删除缺失的分钟', '扩大图表尺寸']"
  :answer="1"
  explanation="问题需要分钟级指标，而现有粒度不支持。算法无法从粗粒度数据中恢复从未记录的信息。"
/>

<QuizBlock
  question="模型能准确预测某站点缺车，是否证明该站点位置导致了缺车？"
  :options="['是，准确率高就能证明因果', '否，预测关系不自动等于因果关系', '是，只要样本足够大', '取决于是否使用 Pandas']"
  :answer="1"
  explanation="预测可以利用相关关系；因果结论还需要排除混杂因素并依赖合适的研究设计。"
/>

<QuizBlock
  question="“每个区域每天一行”的数据最不适合回答哪个问题？"
  :options="['哪个区域月均骑行量最高', '每天骑行量是否有长期趋势', '某站点 18:07 是否无车可借', '周末和工作日的区域骑行量是否不同']"
  :answer="2"
  explanation="区域日表已经丢失具体站点和分钟信息，无法回答站点在某一分钟的状态。"
/>

<QuizBlock
  question="为什么行动阶段仍可能回到问题定义？"
  :options="['因为数据科学永远没有结果', '因为结果可能无法支持原定行动或暴露了新的问题', '因为代码必须重复运行两次', '因为所有模型都需要换成深度学习']"
  :answer="1"
  explanation="生命周期是反馈系统。无法执行、效果不均或环境变化都会要求重新定义问题。"
/>

## 7. 代码练习：定义一个可计算指标

下面的表记录了 6 个站点时刻。请计算每个站点的缺车比例，并找出比例最高的站点。

```python
import pandas as pd

status = pd.DataFrame({
    "station": ["湖畔", "湖畔", "湖畔", "剧场", "剧场", "剧场"],
    "bikes_available": [0, 1, 4, 2, 1, 5],
})
```

<details><summary>提示 1</summary>

先创建布尔列 `is_shortage`，规则是可借车辆少于 2 辆。

</details>

<details><summary>提示 2</summary>

按 `station` 分组，对 `is_shortage` 求均值。

</details>

<details><summary>参考实现</summary>

```python
import pandas as pd

status = pd.DataFrame({
    "station": ["湖畔", "湖畔", "湖畔", "剧场", "剧场", "剧场"],
    "bikes_available": [0, 1, 4, 2, 1, 5],
})

status["is_shortage"] = status["bikes_available"] < 2
rates = status.groupby("station")["is_shortage"].mean()

print(rates)
print("最高风险站点：", rates.idxmax())
```

</details>

## 8. 常见错误

- 先选模型，后找问题。
- 用现有字段决定研究问题，而不是先定义真正需要的证据。
- 把相关关系写成因果结论。
- 只报告一个平均数，隐藏不同群体或区域的差异。
- 交付结果时不说明数据范围和失效条件。

## 9. 本讲小结

- 数据科学从可行动的问题开始，以可以复查的证据结束。
- 生命周期会反复回到前面的阶段，不是一次性流水线。
- 数据粒度决定了哪些问题能够回答。
- 探索、预测、推断和因果问题需要不同证据。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 01：课程主题与数据生命周期知识边界。
- *Learning Data Science*，Ch.1–2：数据生命周期与问题范围主题。
- 本页共享单车案例、文字、图解、代码、问题和练习均为本课程原创。

<DsLessonProgress lesson-id="01" next-href="/courses/data-science/02-pandas-structures" next-label="第 02 讲：Pandas 表、列与索引" />
