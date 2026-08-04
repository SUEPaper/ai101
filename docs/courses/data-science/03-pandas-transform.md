---
course: data-science
title: Pandas：筛选与变换
description: 使用布尔掩码筛选行，安全地修改列，并把一连串数据操作组织成可读的变换管道。
outline: [2, 3]
---

<DsLessonHero
  number="03"
  title="Pandas：筛选与变换"
  eyebrow="把选择条件写成数据，让每一步变化都能被检查"
  duration="建议 90 分钟"
  :outcomes="[
    '把业务条件转换成与 DataFrame 等长的布尔掩码',
    '正确组合多个条件并解释 &、|、~ 的作用',
    '安全地新增、修改、重命名和删除列，并组织方法链'
  ]"
/>

## 本讲要解决的问题

上一讲解决“按标签或位置拿什么”。现实分析更常见的任务是“拿出所有满足某种条件的行”：高需求站点、雨天记录、异常订单或超过阈值的测量值。

```python
import pandas as pd

daily = pd.DataFrame({
    "station": ["湖畔", "剧场", "大学城", "体育馆", "公园东门"],
    "district": ["北区", "南区", "北区", "西区", "南区"],
    "rides": [128, 94, 173, 81, 116],
    "capacity": [20, 16, 24, 15, 18],
    "is_weekend": [False, False, True, False, True],
})
```

## 1. 布尔筛选分成两步

第一步计算条件，得到与原表行数相同的布尔 Series；第二步用它打开或关闭每一行。

```python
mask = daily["rides"] >= 100
selected = daily[mask]

print(mask)
print(selected)
```

<figure class="ds-figure">
  <div class="ds-pipeline">
    <span>原始列 rides</span><i>→</i><span>逐项判断 ≥ 100</span><i>→</i><span>True / False 掩码</span><i>→</i><span>只保留 True 行</span>
  </div>
  <figcaption>布尔掩码不是“筛选后的表”，而是一列判断结果。把判断和筛选拆开打印，是调试复杂条件最有效的方法之一。</figcaption>
</figure>

<BooleanMaskLab />

也可以把两步写在一起：

```python
high_demand = daily[daily["rides"] >= 100]
print(high_demand)
```

当条件较长时，建议先给掩码命名。`high_demand_mask` 比一串重复表达式更容易检查。

## 2. 组合条件时使用 `&`、`|`、`~`

Pandas 的条件是 Series，不能直接使用 Python 的 `and`、`or`、`not`。应使用逐元素运算符：

- `&`：两个条件同时成立。
- `|`：至少一个条件成立。
- `~`：条件取反。

```python
busy_weekday = daily[
    (daily["rides"] >= 100) & (~daily["is_weekend"])
]

north_or_south = daily[
    (daily["district"] == "北区") | (daily["district"] == "南区")
]

print(busy_weekday)
print(north_or_south)
```

每个比较条件都使用括号。原因是 `&` 和比较运算的优先级容易产生与阅读直觉不同的解释。

<div class="ds-warning">
  <strong>不要写 `series_a and series_b`</strong>
  Python 的 `and` 需要把整个对象判断为一个真或假，但一列中可能同时存在 True 和 False。Pandas 会拒绝这种含糊判断。
</div>

### 更适合列表条件的 `.isin()`

```python
target_districts = ["北区", "西区"]
target_rows = daily[daily["district"].isin(target_districts)]

print(target_rows)
```

当条件是“属于这几个值之一”时，`.isin()` 通常比多个 `|` 更清楚。

## 3. 新增和修改列

列运算会自动按 Index 对齐并逐行计算：

```python
daily["rides_per_slot"] = daily["rides"] / daily["capacity"]
daily["demand_level"] = daily["rides_per_slot"].map(
    lambda value: "高" if value >= 7 else "常规"
)

print(daily[["station", "rides_per_slot", "demand_level"]])
```

`rides_per_slot` 把不同容量的站点放到更可比的尺度上。这里的指标仍只是一个探索工具，不代表容量越小就一定需要扩建。

重命名和删除列应明确指定：

```python
renamed = daily.rename(columns={"rides": "ride_count"})
trimmed = renamed.drop(columns=["is_weekend"])

print(trimmed.columns.tolist())
```

大多数 Pandas 方法默认返回新表，不修改原表。把结果赋给新变量通常比依赖 `inplace=True` 更容易追踪。

## 4. 切片后修改前先 `.copy()`

下面的意图是修改高需求子表，而不是修改原表：

```python
high_demand = daily.loc[daily["rides"] >= 100].copy()
high_demand["priority"] = "需要复核"

print(high_demand[["station", "priority"]])
```

显式 `.copy()` 表达了数据所有权：`high_demand` 是一个可以独立修改的新表。这样能够避免链式赋值带来的含糊行为。

<div class="ds-stop">
  <strong>停下来想一想</strong>
  如果你的目标是直接修改原表中符合条件的行，更清晰的写法是 `daily.loc[mask, "priority"] = "需要复核"`，同时指定行和列。
</div>

## 5. 用小型方法回答表格问题

```python
print(daily.shape)
print(daily.size)
print(daily["district"].unique())
print(daily["district"].value_counts())
print(daily["rides"].describe())
print(daily.sort_values("rides", ascending=False).head(3))
```

- `shape` 是 `(行数, 列数)`。
- `size` 是单元格总数。
- `unique()` 返回出现过的不同值。
- `value_counts()` 计算每个值出现的次数。
- `describe()` 给出一组快速描述统计。
- `sort_values()` 按值排序。

## 6. 方法链是一条可以从左到右阅读的数据管道

```python
result = (
    daily
    .loc[lambda frame: frame["rides"] >= 100]
    .assign(rides_per_slot=lambda frame: frame["rides"] / frame["capacity"])
    .sort_values("rides_per_slot", ascending=False)
    .reset_index(drop=True)
)

print(result[["station", "rides_per_slot"]])
```

<figure class="ds-figure">
  <div class="ds-pipeline">
    <span>daily</span><i>→</i><span>筛选高需求</span><i>→</i><span>计算标准化指标</span><i>→</i><span>排序</span><i>→</i><span>重置 Index</span>
  </div>
  <figcaption>好的方法链像处理记录：每一步只有一个明确动作。链条过长时，应保存中间结果并检查形状，而不是追求“一行代码”。</figcaption>
</figure>

## 7. 知识检查

<QuizBlock
  question="`daily['rides'] >= 100` 的结果是什么？"
  :options="['一个布尔值', '与 daily 等长的布尔 Series', '筛选后的 DataFrame', '一列字符串']"
  :answer="1"
  explanation="比较会逐元素执行，得到每行一个 True 或 False；把这个 Series 放入 daily[...] 才会筛行。"
/>

<QuizBlock
  question="同时选择北区且 rides 大于 120 的正确写法是哪一个？"
  :options="['两个比较分别加括号，再用 & 连接', '两个比较直接用 and 连接', '用一个等号给 district 赋值', '两个比较分别加括号，再用 | 连接']"
  :answer="0"
  explanation="两个 Series 条件用 & 逐元素组合，并分别加括号。| 表示或，不符合“且”。"
/>

<QuizBlock
  question="什么情况下 `.isin([...])` 最合适？"
  :options="['检查数值是否缺失', '检查一列的值是否属于给定集合', '计算平均值', '修改 Index']"
  :answer="1"
  explanation="isin 会为每个值检查成员关系，适合“地区属于北区或西区”这类集合条件。"
/>

<QuizBlock
  question="为什么对子表修改前常使用 `.copy()`？"
  :options="['让表自动排序', '明确创建独立数据，避免链式赋值含糊', '让列名变短', '保证没有重复值']"
  :answer="1"
  explanation="copy 明确子表拥有独立数据；否则一次切片可能是视图或副本，修改意图不清楚。"
/>

<QuizBlock
  question="一个 5 行 4 列 DataFrame 的 `size` 是多少？"
  :options="['5', '4', '9', '20']"
  :answer="3"
  explanation="size 是所有单元格数量，即 5 × 4 = 20。shape 才返回 (5, 4)。"
/>

## 8. 代码练习：生成调度候选表

基于本讲的 `daily`：

1. 选出工作日且骑行量至少为 90 的站点。
2. 创建独立副本。
3. 新增 `load = rides / capacity`。
4. 按 `load` 从高到低排列。
5. 最终只保留 `station`、`district` 和 `load`。

<details><summary>提示 1</summary>

工作日条件是 `~daily["is_weekend"]`。

</details>

<details><summary>提示 2</summary>

可以先筛选并 `.copy()`，再新增列；也可以使用 `.assign()` 方法链。

</details>

<details><summary>参考实现</summary>

```python
import pandas as pd

daily = pd.DataFrame({
    "station": ["湖畔", "剧场", "大学城", "体育馆", "公园东门"],
    "district": ["北区", "南区", "北区", "西区", "南区"],
    "rides": [128, 94, 173, 81, 116],
    "capacity": [20, 16, 24, 15, 18],
    "is_weekend": [False, False, True, False, True],
})

candidate = daily.loc[
    (~daily["is_weekend"]) & (daily["rides"] >= 90)
].copy()
candidate["load"] = candidate["rides"] / candidate["capacity"]
candidate = candidate.sort_values("load", ascending=False)
candidate = candidate[["station", "district", "load"]]

print(candidate)
```

</details>

## 9. 本讲小结

- 布尔筛选先产生掩码，再按掩码保留行。
- Pandas 条件使用 `&`、`|`、`~`，并为单个比较加括号。
- 新增列通常是向量化运算；子表修改前用 `.copy()` 明确意图。
- 方法链应保持短小，每一步都能用业务语言解释。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 03：条件选择、列变换和常用方法的知识边界。
- *Learning Data Science*，Ch.6：筛选与变换主题。
- 本页案例、文字、图解、代码、问题和练习均为本课程原创。

<DsLessonProgress lesson-id="03" next-href="/courses/data-science/04-pandas-aggregate" next-label="第 04 讲：Pandas 分组、透视与连接" />
