---
course: data-science
title: 可视化：看见分布
description: 根据变量类型选择图形，并理解直方图、箱线图、Violin plot 与 KDE 各自保留和隐藏的信息。
outline: [2, 3]
---

<DsLessonHero
  number="07"
  title="可视化：看见分布"
  eyebrow="图不是数据的装饰，而是一个带参数的观察仪器"
  duration="建议 90–110 分钟"
  :outcomes="[
    '根据定量、名义和有序变量选择合适的分布图',
    '解释直方图箱宽和 KDE bandwidth 如何影响看到的形状',
    '结合直方图、箱线图和 Violin plot 判断中心、离散、偏斜与多峰'
  ]"
/>

## 本讲要解决的问题

只看平均骑行时长，会把短途通勤、休闲长骑和异常未还车混在一起。分布图的任务是展示值出现在哪里、出现多少，以及整体形状中是否存在偏斜、多峰、长尾或异常值。

```python
import numpy as np
import pandas as pd

ride_minutes = np.array([8, 9, 11, 12, 13, 15, 16, 18, 19, 22, 25, 28, 31, 35, 44, 57, 88])
rides = pd.DataFrame({"duration_min": ride_minutes})

print(rides["duration_min"].describe())
```

## 1. 变量类型决定图形语法

- 名义类别，如区域、天气类型：关注每类数量或比例，常用条形图。
- 有序类别，如低、中、高：保留类别顺序的条形图。
- 定量变量，如时长、温度：关注数轴上的形状，常用直方图、箱线图、KDE。

<figure class="ds-figure">
  <div class="ds-anatomy">
    <span class="is-index">问题</span><span class="is-column">类别有多少</span><span class="is-column">数值分布如何</span><span class="is-column">组间分布有何不同</span>
    <span class="is-index">优先图形</span><span>Bar</span><span>Histogram / ECDF</span><span>Box / Violin / 分面</span>
    <span class="is-index">不要混淆</span><span>类别顺序</span><span>箱宽与尺度</span><span>样本量与重叠</span>
  </div>
  <figcaption>条形图的每根柱代表独立类别；直方图的柱表示连续数轴上的区间，柱之间通常不留缝。</figcaption>
</figure>

## 2. 直方图通过分箱近似分布

直方图先把数轴切成区间，再统计每个区间中的观测数。箱宽太大可能掩盖多峰，太小则可能把随机波动误认成结构。

<HistogramLab />

使用 Matplotlib 绘制：

```python
# tutorial-validation: syntax-only
import matplotlib.pyplot as plt

plt.hist(ride_minutes, bins=6, edgecolor="white", color="#2f7c7a")
plt.xlabel("骑行时长（分钟）")
plt.ylabel("记录数")
plt.title("骑行时长分布")
plt.show()
```

选择分箱时至少检查两到三种合理箱宽。如果“两个峰”的结论只在一个特殊箱宽下出现，它还不是稳定证据。

<QuizBlock
  question="直方图的箱数从 5 增加到 50，通常会发生什么？"
  :options="['图一定更准确', '局部波动更明显，也更容易把噪声看成结构', '所有异常值消失', '横轴变成类别变量']"
  :answer="1"
  explanation="更多、更窄的箱保留局部细节，同时提高随机波动的可见度；需要结合样本量判断。"
/>

## 3. 箱线图是五数概括，不是完整形状

箱线图强调中位数、四分位距和可能的异常点，适合紧凑比较多个组。但两个形状完全不同的分布可能拥有相同箱线图。

```python
quartiles = np.quantile(ride_minutes, [0, 0.25, 0.5, 0.75, 1])
q1, median, q3 = quartiles[1:4]
iqr = q3 - q1

print("五数概括：", quartiles)
print("上侧异常阈值：", q3 + 1.5 * iqr)
```

箱线图中被标为“异常”的点并不自动是错误数据。它只表示点远离中间 50% 的范围，需要结合业务判断。

## 4. KDE 用平滑核叠出连续曲线

KDE 可以想象成在每个观测点放一个小山丘，再把所有山丘相加。`bandwidth` 控制山丘宽度：

- 太小：曲线出现许多尖峰，追随样本噪声。
- 太大：曲线过度平滑，吞掉真实结构。

```python
# tutorial-validation: syntax-only
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(data=rides, x="duration_min", bw_adjust=0.7, fill=True, color="#d76d50")
plt.xlabel("骑行时长（分钟）")
plt.show()
```

KDE 曲线是密度，不是“这个位置有多少个人”。曲线下面积为 1，高度会随平滑参数变化。

## 5. 比较多个群体

```python
grouped_rides = pd.DataFrame({
    "day_type": ["工作日"] * 8 + ["周末"] * 8,
    "duration_min": [8, 9, 11, 12, 14, 17, 19, 25, 12, 15, 18, 22, 27, 34, 46, 61],
})

print(grouped_rides.groupby("day_type")["duration_min"].describe())
```

```python
# tutorial-validation: syntax-only
import seaborn as sns
import matplotlib.pyplot as plt

sns.violinplot(data=grouped_rides, x="day_type", y="duration_min", inner="quartile", color="#86aaa0")
plt.xlabel("")
plt.ylabel("骑行时长（分钟）")
plt.show()
```

Violin plot 结合了平滑形状和组间比较，但小样本的平滑轮廓可能显得比数据更确定。最好同时标注样本量，必要时叠加原始点。

<figure class="ds-figure">
  <div class="ds-flow">
    <div><b>CENTER</b><span>中心在哪里</span><small>中位数、均值</small></div>
    <div><b>SPREAD</b><span>波动有多大</span><small>IQR、标准差、范围</small></div>
    <div><b>SHAPE</b><span>对称、偏斜或多峰</span><small>Histogram、KDE、ECDF</small></div>
    <div><b>TAIL</b><span>极端区域如何</span><small>长尾、异常点</small></div>
    <div><b>GROUP</b><span>不同群体是否重叠</span><small>分面、Box、Violin</small></div>
  </div>
  <figcaption>阅读分布时按固定问题扫描，比只说“这个图很偏”更容易形成可复查结论。</figcaption>
</figure>

## 6. 知识检查

<QuizBlock
  question="展示不同区域的站点数量，首选哪类图？"
  :options="['条形图', 'KDE', '散点图', '等高线图']"
  :answer="0"
  explanation="区域是类别变量，每类计数适合用条形长度编码。"
/>

<QuizBlock
  question="箱线图最容易隐藏哪类信息？"
  :options="['中位数', '四分位距', '多峰等具体分布形状', '最大值']"
  :answer="2"
  explanation="箱线图把数据压缩成少量统计量，无法完整表达多峰或局部空缺。"
/>

<QuizBlock
  question="KDE bandwidth 太大时，主要风险是什么？"
  :options="['曲线过度平滑，真实局部结构被隐藏', '数据自动增加', '面积不再为 1', '横轴变成文本']"
  :answer="0"
  explanation="宽核会把相邻结构混合，曲线更平滑但可能失去真实峰值。"
/>

<QuizBlock
  question="箱线图之外的点应如何解释？"
  :options="['一定是录入错误', '一定要删除', '是基于 IQR 规则的远端点，需要业务复核', '证明分布是正态']"
  :answer="2"
  explanation="统计规则只标记远端程度，不能判断值是否真实或应删除。"
/>

## 7. 代码练习：比较工作日和周末

使用 `grouped_rides`：

1. 分别计算两组的数量、中位数和四分位距。
2. 画并排箱线图。
3. 叠加原始点。
4. 用一句话描述中心、波动和尾部差异，不下因果结论。

<details><summary>参考计算</summary>

```python
summary = grouped_rides.groupby("day_type")["duration_min"].agg(
    count="size",
    median="median",
    q1=lambda values: values.quantile(0.25),
    q3=lambda values: values.quantile(0.75),
)
summary["iqr"] = summary["q3"] - summary["q1"]

print(summary)
```

</details>

<details><summary>参考绘图</summary>

```python
# tutorial-validation: syntax-only
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(data=grouped_rides, x="day_type", y="duration_min", color="#c9ddd6")
sns.stripplot(data=grouped_rides, x="day_type", y="duration_min", color="#b95f43", alpha=0.75)
plt.xlabel("")
plt.ylabel("骑行时长（分钟）")
plt.show()
```

</details>

## 8. 本讲小结

- 图形选择首先取决于变量类型和问题。
- 直方图和 KDE 都带有平滑参数，结论应检查参数稳定性。
- 箱线图适合紧凑比较，但会隐藏具体形状。
- 异常点是复核入口，不是自动删除指令。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 07：分布可视化的知识边界。
- *Learning Data Science*，Ch.10–11：EDA 与可视化主题。
- 本页数据、文字、图解、代码、问题和练习均为本课程原创。

<DsLessonProgress lesson-id="07" next-href="/courses/data-science/08-visualization-relationships" next-label="第 08 讲：关系与表达" />
