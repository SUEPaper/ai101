---
course: data-science
title: 可视化：关系与表达
description: 用散点图、变换、分组与视觉编码呈现变量关系，并识别过度绘制和误导性尺度。
outline: [2, 3]
---

<DsLessonHero
  number="08"
  title="可视化：关系与表达"
  eyebrow="让读者准确看见差异，而不是被图形效果牵着走"
  duration="建议 100–120 分钟"
  :outcomes="[
    '识别散点图中的方向、形状、强度、异常点与分组',
    '使用 jitter、透明度、Hexbin 或分面处理过度绘制',
    '根据人类感知选择位置、长度、面积和颜色编码'
  ]"
/>

## 本讲要解决的问题

单变量分布回答“每个变量长什么样”，关系图回答“两个或更多变量如何一起变化”。但图上没有明显模式，可能是确实没有关系，也可能只是点重叠、尺度错误或群体混在一起。

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
temperature = rng.integers(18, 33, size=80)
rides = 35 + temperature * 3 + rng.normal(0, 13, size=80)

weather = pd.DataFrame({"temperature": temperature, "rides": rides})
print(weather.corr(numeric_only=True))
```

## 1. 阅读散点图的五个问题

1. 方向：一个变量增加时，另一个通常增加还是减少？
2. 形状：关系近似直线、弯曲，还是存在阈值？
3. 强度：点围绕主要形状聚集得多紧？
4. 异常：是否有远离主要模式的点？
5. 分组：颜色或形状是否暗示多个子群体？

```python
# tutorial-validation: syntax-only
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=weather, x="temperature", y="rides", alpha=0.55, color="#2f7c7a")
plt.xlabel("气温（°C）")
plt.ylabel("骑行量")
plt.show()
```

相关系数只能概括线性关系，无法代替散点图。弯曲关系、异常点和多个群体都可能让一个相关系数产生误导。

## 2. 过度绘制会隐藏频数

当坐标取值重复时，多个点会完全重叠。图上看到一个点，不代表只有一条记录。

<ScatterLab />

常用处理方式：

- 调低点的 `alpha`，重叠区域会更深。
- 对离散坐标加入少量 jitter，只用于展示密度。
- 数据很多时使用 Hexbin 或二维密度。
- 使用分面拆开关键群体。
- 直接汇总成每个区间的数量或均值。

```python
# tutorial-validation: syntax-only
import matplotlib.pyplot as plt

plt.hexbin(weather["temperature"], weather["rides"], gridsize=12, cmap="viridis", mincnt=1)
plt.colorbar(label="记录数")
plt.xlabel("气温（°C）")
plt.ylabel("骑行量")
plt.show()
```

<div class="ds-warning">
  <strong>Jitter 只改变显示位置</strong>
  扰动后的点不能用于读取精确坐标，也不能拿来重新计算相关或回归。计算必须继续使用原始数据。
</div>

## 3. 变换可以揭示被压缩的结构

订单金额、城市人口等变量常跨越多个数量级。在线性坐标上，大值会挤压小值；对数变换可以把“倍数差异”展开。

```python
growth = pd.DataFrame({
    "marketing": [1, 2, 4, 8, 16, 32],
    "new_users": [14, 22, 41, 78, 153, 311],
})
growth["log_marketing"] = np.log2(growth["marketing"])
growth["log_users"] = np.log2(growth["new_users"])

print(growth)
```

变换后的直线关系描述的是原尺度上的乘法关系。解释时必须回到原单位，不能只说“log_users 增加 1”。

<QuizBlock
  question="对数坐标最适合揭示哪类差异？"
  :options="['固定加法差异', '跨多个数量级的倍数差异', '纯类别变量', '缺失值类型']"
  :answer="1"
  explanation="对数把乘法关系转成加法距离，适合跨数量级的正值数据。"
/>

## 4. 人眼最准确地比较位置和长度

常见视觉通道的定量比较能力大致是：共同尺度上的位置最准确，其次是长度；角度、面积、体积和颜色明度更难精确比较。

<figure class="ds-figure">
  <div class="ds-flow">
    <div><b>POSITION</b><span>共同坐标位置</span><small>最适合精确比较</small></div>
    <div><b>LENGTH</b><span>条形长度</span><small>直观、可排序</small></div>
    <div><b>ANGLE</b><span>扇形角度</span><small>细小差异难判断</small></div>
    <div><b>AREA</b><span>圆面积</span><small>容易低估或高估</small></div>
    <div><b>COLOR</b><span>明度与色相</span><small>适合模式，不适合精确读数</small></div>
  </div>
  <figcaption>需要读者比较精确数值时，优先给出共同尺度上的位置或长度，不要让面积和三维体积承担主要信息。</figcaption>
</figure>

### 颜色应匹配数据语义

- 顺序型色带：从低到高，如骑行量。
- 发散型色带：围绕有意义中点向两边变化，如相对基线的增减。
- 类别型色板：区分少量无序类别。

彩虹色带的亮度并不单调，可能制造不存在的边界。颜色也不能成为唯一编码，应配合标签、位置或形状。

## 5. 比较必须保持尺度诚实

条形图用长度表达数量，通常应从零开始；折线图关注位置变化，可以根据问题缩放，但必须清晰显示范围。任何截断都应问：它帮助看清变化，还是夸大差异？

```python
district = pd.DataFrame({
    "district": ["北区", "南区", "西区"],
    "satisfaction": [78, 81, 83],
})

print(district.assign(distance_from_zero=district["satisfaction"]))
```

如果条形轴从 75 开始，83 的柱可能看起来是 78 的数倍，虽然实际只高 5 个百分点。

## 6. 图中应直接提供阅读上下文

一张可发表的图至少需要：

- 说明“谁和什么”的标题，而不是只写“结果”。
- 带单位的轴标签。
- 必要的数据来源和时间范围。
- 清楚的颜色含义。
- 对关键点的直接标注。
- 样本量或不确定性信息。

<div class="ds-note">
  <strong>图表结论写成观察，不写成宣传语</strong>
  “南区样本满意度比北区高 3 个百分点”比“南区遥遥领先”更可核查，也保留了进一步讨论不确定性的空间。
</div>

## 7. 知识检查

<QuizBlock
  question="散点图中的多个点完全重叠，首要问题是什么？"
  :options="['相关系数必定为零', '图形隐藏了局部频数', '横轴必须删除', '数据一定重复']"
  :answer="1"
  explanation="相同或相近坐标会让多条记录显示成一个点，需要透明度、jitter、分箱或密度图。"
/>

<QuizBlock
  question="精确比较三个类别的数值，通常优先使用什么视觉通道？"
  :options="['三维球体体积', '扇形角度', '共同基线上的条形长度', '随机颜色']"
  :answer="2"
  explanation="共同基线上的长度更容易准确比较，面积和角度更适合粗略比例。"
/>

<QuizBlock
  question="展示相对零增长的正负变化，哪类颜色最合适？"
  :options="['发散型色带，并以零为中点', '无序类别色板', '彩虹色带', '所有值同一种颜色且无标签']"
  :answer="0"
  explanation="发散色带围绕有意义中点表达方向和幅度，零增长正好是自然中点。"
/>

<QuizBlock
  question="条形图截断纵轴的主要风险是什么？"
  :options="['文件体积变大', '用长度夸大较小的数值差异', '颜色自动消失', '不能显示类别名']"
  :answer="1"
  explanation="条形长度从截断基线开始，会让读者把局部差异误解为整体比例差异。"
/>

<QuizBlock
  question="为什么相关系数不能替代散点图？"
  :options="['相关系数只能用于文本', '它可能隐藏非线性、异常点和子群体', '散点图永远没有误差', '相关系数不能是负数']"
  :answer="1"
  explanation="同一个相关系数可以对应形状完全不同的数据关系，必须结合图形检查。"
/>

## 8. 代码练习：制作一张可审查的关系图

使用本讲的 `weather`：

1. 画气温与骑行量散点图。
2. 设置透明度并添加趋势线。
3. 给坐标轴添加单位。
4. 标出骑行量最高的记录。
5. 写一句只描述观察、不声称因果的结论。

<details><summary>参考绘图</summary>

```python
# tutorial-validation: syntax-only
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.regplot(
    data=weather,
    x="temperature",
    y="rides",
    scatter_kws={"alpha": 0.5, "color": "#2f7c7a"},
    line_kws={"color": "#d76d50"},
)
highest = weather.loc[weather["rides"].idxmax()]
ax.annotate(
    "最高骑行量",
    xy=(highest["temperature"], highest["rides"]),
    xytext=(8, 10),
    textcoords="offset points",
)
ax.set(xlabel="气温（°C）", ylabel="骑行量", title="较暖天气通常伴随更高骑行量")
plt.show()
```

</details>

## 9. 本讲小结

- 关系图需要检查方向、形状、强度、异常和分组。
- 过度绘制隐藏频数，jitter 只用于显示，不改变计算数据。
- 变换可以展开尺度，但解释必须回到原单位。
- 位置和长度适合精确比较；面积、角度和颜色需要谨慎使用。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 08：关系可视化、变换和视觉编码的知识边界。
- *Learning Data Science*，Ch.11：尺度、比较、颜色与上下文主题。
- 本页数据、文字、图解、代码、问题和练习均为本课程原创。

<DsLessonProgress lesson-id="08" next-href="/courses/data-science/09-sampling" next-label="第 09 讲：抽样与偏差" />
