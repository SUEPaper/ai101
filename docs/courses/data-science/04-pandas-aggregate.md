---
course: data-science
title: Pandas：分组、透视与连接
description: 使用 groupby、agg、pivot_table 和 merge，把行级记录汇总成可以比较的证据。
outline: [2, 3]
---

<DsLessonHero
  number="04"
  title="Pandas：分组、透视与连接"
  eyebrow="从一堆行里提炼群体差异，再把分散信息拼回完整语境"
  duration="建议 100–120 分钟"
  :outcomes="[
    '用 Split–Apply–Combine 解释 groupby 的计算过程',
    '使用 agg 和 pivot_table 生成可比较的汇总表',
    '根据键和保留范围选择合适的 merge 方式'
  ]"
/>

## 本讲要解决的问题

逐行查看数据很难回答群体问题：“哪个区域骑行量最高？”“周末和工作日的模式是否不同？”“站点容量与需求是否匹配？”这类问题需要先分组，再对每组应用同一个统计量。

```python
import pandas as pd

daily = pd.DataFrame({
    "station_id": ["S01", "S02", "S03", "S04", "S05"],
    "district": ["北区", "南区", "北区", "西区", "南区"],
    "day_type": ["工作日", "工作日", "周末", "工作日", "周末"],
    "rides": [128, 94, 173, 81, 116],
    "shortage_minutes": [18, 7, 24, 5, 13],
})
```

## 1. GroupBy 是 Split–Apply–Combine

执行 `groupby` 时，可以把过程想成三步：

1. **Split**：按照键把行分到不同桶里。
2. **Apply**：在每个桶上执行同一种计算。
3. **Combine**：把每组结果拼成新的 Series 或 DataFrame。

<GroupByLab />

```python
district_rides = daily.groupby("district")["rides"].sum()
print(district_rides)
```

`groupby("district")` 本身只是描述分组规则；直到调用 `.sum()`，Pandas 才知道每组要留下什么结果。

<div class="ds-stop">
  <strong>停下来想一想</strong>
  分组后原始行通常被压缩。你必须明确“每组最后留下一个什么值”：总量、均值、最大值、行数，还是多种统计量。
</div>

## 2. `agg` 一次生成多种统计量

```python
summary = (
    daily
    .groupby("district")
    .agg(
        station_count=("station_id", "nunique"),
        total_rides=("rides", "sum"),
        average_rides=("rides", "mean"),
        max_shortage=("shortage_minutes", "max"),
    )
    .reset_index()
)

print(summary)
```

命名聚合把“输出列名、输入列、聚合函数”写在一起。结果比先计算四张表再拼接更容易审查。

### `.size()` 和 `.count()` 不完全相同

- `.size()` 计算每组行数，包括字段中存在缺失值的行。
- `.count()` 对每一列计算非缺失值数量。

如果你要问“这一组有多少条记录”，优先考虑 `.size()`；如果问“这个字段有多少个有效观测”，使用 `.count()`。

<QuizBlock
  question="`daily.groupby('district')` 已经完成了什么？"
  :options="['已经计算每区总量', '定义了按 district 分组的对象，但还没决定聚合方式', '删除了 district 列', '自动绘制了柱状图']"
  :answer="1"
  explanation="GroupBy 对象保存分组规则；还需要 sum、mean、agg 等操作才能生成汇总结果。"
/>

## 3. 多列分组与透视表

按区域和日期类型同时分组：

```python
by_district_day = (
    daily
    .groupby(["district", "day_type"], as_index=False)
    .agg(total_rides=("rides", "sum"))
)

print(by_district_day)
```

长表适合继续计算；透视表适合把一个分类变量展开为列，便于比较：

```python
pivot = daily.pivot_table(
    index="district",
    columns="day_type",
    values="rides",
    aggfunc="mean",
    fill_value=0,
)

print(pivot)
```

<figure class="ds-figure">
  <div class="ds-pipeline">
    <span>长表：district / day_type / rides</span><i>→</i><span>指定 index 与 columns</span><i>→</i><span>交叉单元格执行 mean</span><i>→</i><span>宽表：每种 day_type 成为一列</span>
  </div>
  <figcaption>透视表没有创造新信息，它只是重新排列并聚合。必须明确重复组合出现时使用什么 `aggfunc`。</figcaption>
</figure>

## 4. Join：通过键把不同表的语境拼起来

骑行日报记录需求，站点表记录容量和区域属性。两张表通过 `station_id` 连接：

```python
stations = pd.DataFrame({
    "station_id": ["S01", "S02", "S03", "S04", "S06"],
    "station_name": ["湖畔", "剧场", "大学城", "体育馆", "火车北站"],
    "capacity": [20, 16, 24, 15, 30],
})

merged = daily.merge(stations, on="station_id", how="left", validate="many_to_one")
print(merged)
```

这里选择 `left`，因为分析主体是 `daily` 的骑行记录：即使某个站点在站点表中缺少信息，也要保留记录并暴露缺失值。

`validate="many_to_one"` 把业务假设写进代码：日报中一个站点可以出现多次，但站点表中每个 `station_id` 应只出现一次。如果右表键重复，连接会把行意外倍增，验证可以尽早报错。

### 常见连接方式

| `how` | 保留范围 | 典型问题 |
|---|---|---|
| `inner` | 两边键都出现的行 | 哪些记录拥有完整匹配信息？ |
| `left` | 左表全部行 | 为主分析表补充属性时，哪些没有匹配？ |
| `right` | 右表全部行 | 是否覆盖了完整参考清单？ |
| `outer` | 两边所有键 | 两张清单有哪些差异？ |

<div class="ds-warning">
  <strong>连接后立刻检查行数和未匹配键</strong>
  `merge` 能运行成功不代表结果正确。至少比较连接前后行数，并检查新增字段的缺失值。
</div>

```python
print("连接前：", len(daily))
print("连接后：", len(merged))
print("未匹配站点：", merged.loc[merged["station_name"].isna(), "station_id"].tolist())
```

## 5. 从汇总表回到业务问题

汇总不是终点。假设北区总骑行量最高，至少有两种解释：北区站点更多，或单站需求更高。因此要同时查看 `station_count`、`total_rides` 和 `average_rides`。

```python
report = (
    merged
    .groupby("district", as_index=False)
    .agg(
        stations=("station_id", "nunique"),
        total_rides=("rides", "sum"),
        average_capacity=("capacity", "mean"),
        total_shortage_minutes=("shortage_minutes", "sum"),
    )
    .sort_values("total_shortage_minutes", ascending=False)
)

print(report)
```

<figure class="ds-figure">
  <div class="ds-anatomy">
    <span class="is-index">分析层级</span><span class="is-column">行级</span><span class="is-column">组级</span><span class="is-column">决策级</span>
    <span class="is-index">对象</span><span>站点日报</span><span>区域汇总</span><span>调度优先级</span>
    <span class="is-index">风险</span><span>噪声多</span><span>掩盖组内差异</span><span>指标不等于行动</span>
  </div>
  <figcaption>聚合会压缩信息。每次从行级上升到组级，都要记录哪些差异被隐藏了。</figcaption>
</figure>

## 6. 知识检查

<QuizBlock
  question="要计算每个区域的记录行数，并且缺失值行也要计入，应优先使用什么？"
  :options="['groupby().size()', 'groupby().count() 的任意列', 'dropna()', 'pivot_table()']"
  :answer="0"
  explanation="size 直接计算每组行数；count 针对列计算非缺失值数量。"
/>

<QuizBlock
  question="透视表中同一个 district 与 day_type 组合有多行时，为什么需要 aggfunc？"
  :options="['为了修改列名', '为了决定多行如何压缩成一个单元格', '为了删除 Index', '为了选择连接方式']"
  :answer="1"
  explanation="一个交叉单元格只能放一个汇总值，aggfunc 决定使用均值、总和或其他聚合方式。"
/>

<QuizBlock
  question="以日报表为分析主体，为它补充站点容量，同时保留所有日报记录，应该使用哪种连接？"
  :options="['inner', 'left', 'right', 'cross']"
  :answer="1"
  explanation="left merge 保留左侧日报表的全部行，未匹配的站点属性会显示为缺失值。"
/>

<QuizBlock
  question="连接后行数意外翻倍，最可能的原因是什么？"
  :options="['两边连接键都存在重复，形成多对多匹配', '使用了 mean', '列名太长', 'Index 从 0 开始']"
  :answer="0"
  explanation="重复键会产生组合匹配。使用 validate 声明预期关系，并在连接后检查行数。"
/>

<QuizBlock
  question="某区域 total_rides 最高，为什么不能立刻断言单站需求最高？"
  :options="['总量可能只是因为该区域站点更多', 'sum 不能用于整数', '区域名称不是数字', '必须先使用 SQL']"
  :answer="0"
  explanation="总量同时受站点数量和单站水平影响，应结合组规模和均值解释。"
/>

## 7. 代码练习：构造区域调度报告

使用本讲的 `daily` 和 `stations`：

1. 左连接两张表，并验证关系是 `many_to_one`。
2. 找出未匹配的 `station_id`。
3. 按区域计算站点数、总骑行量、平均容量和总缺车分钟。
4. 新增 `shortage_per_station`。
5. 按该指标从高到低排列。

<details><summary>提示</summary>

聚合后再计算 `shortage_per_station = total_shortage_minutes / stations`，避免在行级重复除法后错误汇总。

</details>

<details><summary>参考实现</summary>

```python
import pandas as pd

daily = pd.DataFrame({
    "station_id": ["S01", "S02", "S03", "S04", "S05"],
    "district": ["北区", "南区", "北区", "西区", "南区"],
    "day_type": ["工作日", "工作日", "周末", "工作日", "周末"],
    "rides": [128, 94, 173, 81, 116],
    "shortage_minutes": [18, 7, 24, 5, 13],
})

stations = pd.DataFrame({
    "station_id": ["S01", "S02", "S03", "S04", "S06"],
    "station_name": ["湖畔", "剧场", "大学城", "体育馆", "火车北站"],
    "capacity": [20, 16, 24, 15, 30],
})

merged = daily.merge(stations, on="station_id", how="left", validate="many_to_one")
unmatched = merged.loc[merged["station_name"].isna(), "station_id"].tolist()

report = (
    merged
    .groupby("district", as_index=False)
    .agg(
        stations=("station_id", "nunique"),
        total_rides=("rides", "sum"),
        average_capacity=("capacity", "mean"),
        total_shortage_minutes=("shortage_minutes", "sum"),
    )
)
report["shortage_per_station"] = report["total_shortage_minutes"] / report["stations"]
report = report.sort_values("shortage_per_station", ascending=False)

print("未匹配：", unmatched)
print(report)
```

</details>

## 8. 常见错误

- 分组前没有明确每组最终需要留下什么统计量。
- 用 `.count()` 计算行数，却忽略该列中的缺失值。
- 透视表没有明确重复组合的聚合规则。
- 连接前不检查键是否唯一，连接后不检查行数。
- 用总量直接比较规模不同的群体。

## 9. 本讲小结

- `groupby` 的本质是 Split–Apply–Combine。
- 命名聚合让输入列、函数和输出列更容易审查。
- 透视表重新排列并聚合信息，适合比较但会压缩细节。
- `merge` 必须明确连接键、保留范围和预期键关系。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 04：分组、透视和连接的知识边界。
- *Learning Data Science*，Ch.6：聚合与连接主题。
- 本页案例、文字、图解、代码、问题和练习均为本课程原创。

<DsLessonProgress lesson-id="04" next-label="第 05 讲：数据清洗与探索性分析" />
