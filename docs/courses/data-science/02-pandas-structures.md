---
course: data-science
title: Pandas：表、列与索引
description: 建立 DataFrame、Series、Index 的清晰心智模型，并掌握 loc、iloc 和方括号选择。
outline: [2, 3]
---

<DsLessonHero
  number="02"
  title="Pandas：表、列与索引"
  eyebrow="先看懂表格的坐标系，再写选择表达式"
  duration="建议 90 分钟"
  :outcomes="[
    '解释 DataFrame、Series 与 Index 的关系',
    '根据标签和位置选择正确的 loc 或 iloc',
    '在运行前判断一次选择会返回标量、Series 还是 DataFrame'
  ]"
/>

## 本讲要解决的问题

很多 Pandas 错误不是 API 记不住，而是没有先回答三个问题：你选择的是行还是列？你使用的是标签还是位置？你希望结果保持二维表还是缩成一维列？

本讲继续使用原创的共享单车站点日报。先构造一张小表：

```python
import pandas as pd

stations = pd.DataFrame(
    {
        "station": ["湖畔", "剧场", "大学城", "体育馆"],
        "district": ["北区", "南区", "北区", "西区"],
        "rides": [128, 94, 173, 81],
    },
    index=["S01", "S02", "S03", "S04"],
)

print(stations)
```

## 1. DataFrame 是对齐的 Series 集合

DataFrame 可以看成多列共享同一个 Index 的 Series。每列有自己的名称和数据类型；Index 为每行提供标签。

<figure class="ds-figure">
  <div class="ds-anatomy">
    <span class="is-index">Index</span><span class="is-column">station</span><span class="is-column">district</span><span class="is-column">rides</span>
    <span class="is-index">S01</span><span>湖畔</span><span>北区</span><span>128</span>
    <span class="is-index">S02</span><span>剧场</span><span>南区</span><span>94</span>
    <span class="is-index">S03</span><span>大学城</span><span>北区</span><span>173</span>
    <span class="is-index">S04</span><span>体育馆</span><span>西区</span><span>81</span>
  </div>
  <figcaption>Index 是行标签，不一定等于行号；列名是列标签。位置和标签恰好相同时最容易让初学者误判。</figcaption>
</figure>

```python
print(type(stations))
print(type(stations["rides"]))
print(stations.index)
print(stations.columns)
print(stations.shape)
```

`stations["rides"]` 返回 Series，因为只选择了一列。Series 仍然保留原表的 Index，所以每个骑行量仍能对应站点行。

<QuizBlock
  question="`stations['rides']` 返回什么类型？"
  :options="['整数', 'List', 'Series', '只有一列的 DataFrame']"
  :answer="2"
  explanation="单个列名放在方括号中会返回 Series。若要保持 DataFrame，需要使用列名列表：stations[['rides']]。"
/>

## 2. Index 是标签，不是隐形行号

Index 可以是字符串、日期或其他可哈希值，也不要求连续。它的任务是让行可以被稳定对齐和按标签引用。

```python
print(stations.loc["S02"])
print(stations.iloc[1])
```

这两行当前返回相同记录，但理由完全不同：

- `.loc["S02"]` 寻找标签为 `S02` 的行。
- `.iloc[1]` 寻找位置为 1，也就是第二行的记录。

如果表格重新排序，`S02` 的位置可能变化；`.loc["S02"]` 仍然跟随标签，`.iloc[1]` 只认位置。

<div class="ds-warning">
  <strong>Index 不要求唯一</strong>
  标签重复时，`.loc[标签]` 可能返回多行。是否允许重复取决于业务语义，不应默认 Index 一定等同于数据库主键。
</div>

## 3. 三种常用选择方式

### 方括号：主要选择列或按布尔条件筛行

```python
rides = stations["rides"]
small_table = stations[["station", "rides"]]

print(rides)
print(small_table)
```

单个列名返回 Series，列名 List 返回 DataFrame。这两个表达式只差一层方括号，结果维度却不同。

### `.loc`：按标签选择

```python
row = stations.loc["S02"]
block = stations.loc[["S01", "S04"], ["station", "rides"]]

print(row)
print(block)
```

`.loc[行标签, 列标签]` 的左右两部分都是标签。标签切片通常包含右端点，这是它和普通 Python 位置切片的重要差异。

### `.iloc`：按整数位置选择

```python
first_two_rows = stations.iloc[0:2]
middle_columns = stations.iloc[:, 1:3]

print(first_two_rows)
print(middle_columns)
```

`.iloc` 遵守 Python 切片规则：左端包含，右端不包含。`0:2` 选择位置 0 和 1。

<DataFrameExplorer />

## 4. 先预测返回形状

在运行代码前，先问“选中了几个轴”：

| 表达式 | 行选择 | 列选择 | 典型返回 |
|---|---|---|---|
| `df['rides']` | 全部行 | 一个列标签 | Series |
| `df[['rides']]` | 全部行 | 一个列名组成的 List | DataFrame |
| `df.loc['S02']` | 一个行标签 | 全部列 | Series |
| `df.loc[['S02']]` | 一个行标签组成的 List | 全部列 | DataFrame |
| `df.loc['S02', 'rides']` | 一个行标签 | 一个列标签 | 标量 |
| `df.iloc[1:3, 0:2]` | 两个行位置 | 两个列位置 | DataFrame |

<div class="ds-stop">
  <strong>停下来想一想</strong>
  为什么 `df.loc[['S02']]` 保持二维？因为传入的是“标签列表”，Pandas 把任务理解为选择一组行，即使列表中只有一个标签。
</div>

## 5. 检查表格，而不是把整张表打印出来

处理中等数据集时，先查看形状、列名、数据类型和少量首尾记录：

```python
print(stations.shape)
print(stations.columns.tolist())
print(stations.dtypes)
print(stations.head(2))
print(stations.tail(2))
```

这些检查回答了四个基础问题：数据有多大、字段叫什么、字段类型是什么、记录大致长什么样。

## 6. 知识检查

<QuizBlock
  question="表格重新按 rides 降序排列后，哪个表达式仍然稳定找到站点 S02？"
  :options="['stations.iloc[1]', 'stations.loc[\'S02\']', 'stations[1]', 'stations.iloc[\'S02\']']"
  :answer="1"
  explanation="loc 使用标签；只要 S02 标签仍存在，就不受行位置变化影响。"
/>

<QuizBlock
  question="哪个表达式返回只有 rides 一列的 DataFrame，而不是 Series？"
  :options="['stations[\'rides\']', 'stations.rides', 'stations[[\'rides\']]', 'stations.loc[\'rides\']']"
  :answer="2"
  explanation="列名列表表示选择一组列，因此返回二维 DataFrame。"
/>

<QuizBlock
  question="`stations.iloc[1:3]` 选择哪些行位置？"
  :options="['只选择 1', '选择 1、2', '选择 1、2、3', '选择标签 S01 到 S03']"
  :answer="1"
  explanation="iloc 使用 Python 位置切片，右端 3 不包含，因此选择位置 1 和 2。"
/>

<QuizBlock
  question="如果 Index 中有两个 S02，`stations.loc['S02']` 最可能返回什么？"
  :options="['必定报错', '第一行', '所有标签为 S02 的行', '最后一行']"
  :answer="2"
  explanation="Pandas 允许重复 Index；按重复标签选择时会返回所有匹配行。"
/>

<QuizBlock
  question="哪个属性最直接回答 DataFrame 有多少行、多少列？"
  :options="['df.size', 'df.shape', 'df.index', 'df.head']"
  :answer="1"
  explanation="shape 返回 (行数, 列数)。size 返回所有单元格总数。"
/>

## 7. 代码练习：精确取出数据

使用本讲的 `stations` 表完成：

1. 取出 `station` 与 `rides` 两列并保持 DataFrame。
2. 按标签取出 `S01` 和 `S03` 两行。
3. 用 `.iloc` 取出最后两行的前两列。
4. 取出 `S03` 行、`rides` 列的单个值。

<details><summary>提示</summary>

保持二维时，把标签放进 List。单个值需要同时给 `.loc` 一个行标签和一个列标签。

</details>

<details><summary>参考实现</summary>

```python
import pandas as pd

stations = pd.DataFrame(
    {
        "station": ["湖畔", "剧场", "大学城", "体育馆"],
        "district": ["北区", "南区", "北区", "西区"],
        "rides": [128, 94, 173, 81],
    },
    index=["S01", "S02", "S03", "S04"],
)

print(stations[["station", "rides"]])
print(stations.loc[["S01", "S03"]])
print(stations.iloc[-2:, :2])
print(stations.loc["S03", "rides"])
```

</details>

## 8. 常见错误

- 把 `.loc` 的标签切片当成右端不包含的位置切片。
- 表格排序后仍用固定 `.iloc` 位置指代某个业务对象。
- 需要 DataFrame 却意外选择成 Series，导致后续形状不匹配。
- 把 Index 当成自动保证唯一的主键。
- 用完整打印代替 `shape`、`dtypes` 和少量抽样检查。

## 9. 本讲小结

- DataFrame 是共享 Index 的多列 Series。
- `.loc` 按标签，`.iloc` 按位置。
- 单个标签常压缩维度，标签 List 常保持二维。
- 每次选择前先预测结果的行、列和数据类型。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 02：Pandas 数据结构和选择操作的知识边界。
- *Learning Data Science*，Ch.6：DataFrame、Index 与子集选择主题。
- 本页案例、文字、图解、代码、问题和练习均为本课程原创。

<DsLessonProgress lesson-id="02" next-href="/courses/data-science/03-pandas-transform" next-label="第 03 讲：Pandas 筛选与变换" />
