---
course: data-science
title: 数据清洗与探索性分析
description: 从文件格式、粒度、变量类型和数据约束出发，建立可追踪的数据清洗与 EDA 流程。
outline: [2, 3]
---

<DsLessonHero
  number="05"
  title="数据清洗与探索性分析"
  eyebrow="清洗不是把表变得好看，而是让每个值重新获得可信含义"
  duration="建议 100–120 分钟"
  :outcomes="[
    '识别 CSV、JSON 等表示方式与表格粒度的差异',
    '为字段建立类型、范围、唯一性和跨列一致性检查',
    '根据缺失机制选择保留、删除、填补或重新获取数据'
  ]"
/>

## 本讲要解决的问题

拿到一个文件，并不等于拿到了可以分析的数据。你需要确认一行代表什么、每列应当是什么类型、记录是否忠实于现实，以及清洗动作会不会悄悄改变研究对象。

本讲处理一份城市共享单车状态日志。原始数据可能来自多个设备，因此时间格式、站点编码和缺失方式并不统一。

## 1. 文件格式只是容器

CSV 用分隔符表示二维表，适合结构稳定的矩形数据；JSON 可以表达嵌套结构，适合 API 返回值；数据库则能维护类型、约束和多表关系。

```python
from io import StringIO
import pandas as pd

raw_csv = StringIO("""station_id,time,bikes_available,capacity
S01,2026-08-01 18:00,4,20
S02,2026/08/01 18:00,2,16
S02,2026/08/01 18:00,2,16
S03,2026-08-01 18:03,-1,24
,2026-08-01 18:04,6,18
S04,18点05分,3,15
""")

status = pd.read_csv(raw_csv)
print(status)
print(status.dtypes)
```

CSV 不会自动保证 `station_id` 唯一，也不会阻止负数进入车辆列。读取成功只说明文本能被解析，不说明数据可信。

<figure class="ds-figure">
  <div class="ds-flow">
    <div><b>BYTES</b><span>文件能否正确解码</span><small>编码、分隔符、换行</small></div>
    <div><b>SHAPE</b><span>一行一列代表什么</span><small>粒度、表头、嵌套结构</small></div>
    <div><b>TYPE</b><span>值能否按预期计算</span><small>数值、类别、时间、文本</small></div>
    <div><b>RULE</b><span>值是否满足现实约束</span><small>范围、唯一、跨列关系</small></div>
    <div><b>SCOPE</b><span>数据能代表谁</span><small>覆盖、缺失与采集偏差</small></div>
  </div>
  <figcaption>清洗应从底层表示逐步走向业务语义。越靠右的检查越依赖领域知识。</figcaption>
</figure>

## 2. 先声明粒度

本表预期粒度是：**一个站点在一个采集时刻一行**。因此 `(station_id, time)` 组合应当唯一。

```python
duplicate_key = status.duplicated(subset=["station_id", "time"], keep=False)
print(status.loc[duplicate_key])
```

如果同一组合出现两行，不能立即执行 `drop_duplicates()`。重复可能来自完全重复上传，也可能是两个传感器的独立测量。先查明生成机制，才能决定如何合并。

<DataQualityLab />

## 3. 变量类型来自问题语义

常见类型包括：

- 定量变量：骑行量、容量、温度。
- 名义类别：区域、站点类型，没有自然顺序。
- 有序类别：低、中、高，有顺序但间距未必相等。
- 时间变量：时间戳、日期、时长。
- 标识符：`station_id` 看起来像文本，但主要作用是区分对象。

同一串字符可以有不同语义。例如 `20260801` 既可能是整数，也可能是日期编码。类型选择应服从要执行的比较和运算。

```python
status["parsed_time"] = pd.to_datetime(status["time"], errors="coerce")
status["bikes_available"] = pd.to_numeric(status["bikes_available"], errors="coerce")

print(status[["time", "parsed_time"]])
```

`errors="coerce"` 会把无法解析的值转换为缺失值。它不会自动修好数据，但会把解析失败变成可以统计和定位的结果。

## 4. 用约束找问题，不靠肉眼扫表

至少建立四类检查：

```python
checks = pd.DataFrame({
    "missing_station": status["station_id"].isna(),
    "invalid_time": status["parsed_time"].isna(),
    "negative_bikes": status["bikes_available"] < 0,
    "over_capacity": status["bikes_available"] > status["capacity"],
    "duplicate_key": duplicate_key,
})

print(checks.sum())
```

- 完整性：关键字段是否缺失。
- 合法范围：车辆数是否在 0 到容量之间。
- 唯一性：粒度对应的键是否重复。
- 跨列一致性：可借车辆不能大于容量。

<div class="ds-note">
  <strong>保留质量标记</strong>
  不要清洗后只留下“干净表”。保留诸如 `invalid_time`、`was_imputed` 的标记，后续才能比较清洗是否改变了结论。
</div>

## 5. 缺失值没有统一答案

缺失可能表示：设备没有上报、字段不适用、真实值为零但被空白表示、或数据在连接时没有匹配。处理方式必须与缺失原因和分析目标一致。

| 处理方式 | 适用条件 | 主要风险 |
|---|---|---|
| 保留缺失 | 模型或汇总能够显式处理 | 后续代码可能误用 |
| 删除记录 | 数量少且近似随机，不改变对象范围 | 样本变小或产生选择偏差 |
| 填补 | 有可信规则或预测依据 | 低估不确定性，制造虚假精度 |
| 重新获取 | 关键字段且可以追溯来源 | 成本和时间增加 |
| 增加缺失类别 | 缺失本身具有业务含义 | 可能混合不同缺失机制 |

```python
clean = status.copy()
clean["quality_issue"] = checks.any(axis=1)
clean.loc[clean["bikes_available"] < 0, "bikes_available"] = pd.NA

print(clean[["station_id", "bikes_available", "quality_issue"]])
```

把负值改成缺失值只是“拒绝使用不可能的测量”，不是恢复真实车辆数。分析报告必须说明这一区别。

## 6. EDA 的第一轮问题

一轮基础探索通常回答：

1. 数据有多大，时间和对象覆盖到哪里？
2. 每列是什么类型，缺失比例是多少？
3. 数值分布是否有不可能值、长尾或异常峰？
4. 类别是否存在拼写变体或极小组？
5. 关键字段之间是否满足业务约束？

```python
print(clean.shape)
print(clean.isna().mean().sort_values(ascending=False))
print(clean["bikes_available"].describe())
print(clean["station_id"].value_counts(dropna=False))
```

EDA 不是为了“把所有图都画一遍”，而是用低成本检查不断修正对数据生成过程的理解。

## 7. 知识检查

<QuizBlock
  question="CSV 文件可以被 Pandas 成功读取，最能说明什么？"
  :options="['数据一定可信', '文本结构可以被解析', '所有列类型都正确', '不存在重复记录']"
  :answer="1"
  explanation="读取成功只证明格式层面可以解析，业务约束、粒度和代表性仍需检查。"
/>

<QuizBlock
  question="粒度为“站点—时刻”时，最自然的唯一性检查键是什么？"
  :options="['station_id', 'time', '(station_id, time)', 'bikes_available']"
  :answer="2"
  explanation="单独站点或时间都可以重复；二者组合才对应一条预期观测。"
/>

<QuizBlock
  question="`pd.to_datetime(..., errors='coerce')` 遇到无法解析的文本时会怎样？"
  :options="['自动猜出正确日期', '删除整行', '转换为缺失时间值', '转换为当前时间']"
  :answer="2"
  explanation="coerce 把失败显式变成 NaT，便于统计和定位，但不代表问题已经修复。"
/>

<QuizBlock
  question="把负的可借车辆数改成缺失值，最准确的描述是什么？"
  :options="['恢复了真实值', '标记该测量不可用于数值分析', '证明车辆数是零', '消除了所有偏差']"
  :answer="1"
  explanation="规则只知道负值不可能，不知道真实值是多少。缺失保留了这种不确定性。"
/>

<QuizBlock
  question="为什么清洗后应保留质量标记？"
  :options="['让表格颜色更丰富', '便于评估清洗记录是否系统性影响结论', '让文件自动压缩', '替代原始数据备份']"
  :answer="1"
  explanation="质量标记让你能比较问题记录与正常记录，审查删除或填补带来的影响。"
/>

## 8. 代码练习：建立质量报告

为本讲的 `status` 创建一张质量报告，至少包含：总行数、重复键行数、站点缺失行数、时间解析失败行数、车辆数越界行数。最后给原表增加 `is_valid`。

<details><summary>提示</summary>

合法车辆数需要同时满足 `bikes_available >= 0` 和 `bikes_available <= capacity`。任何一项检查失败都应让 `is_valid` 为 False。

</details>

<details><summary>参考实现</summary>

```python
from io import StringIO
import pandas as pd

raw_csv = StringIO("""station_id,time,bikes_available,capacity
S01,2026-08-01 18:00,4,20
S02,2026/08/01 18:00,2,16
S02,2026/08/01 18:00,2,16
S03,2026-08-01 18:03,-1,24
,2026-08-01 18:04,6,18
S04,18点05分,3,15
""")
status = pd.read_csv(raw_csv)
status["parsed_time"] = pd.to_datetime(status["time"], errors="coerce")

duplicate = status.duplicated(["station_id", "time"], keep=False)
missing_station = status["station_id"].isna()
invalid_time = status["parsed_time"].isna()
out_of_range = (status["bikes_available"] < 0) | (status["bikes_available"] > status["capacity"])

report = {
    "rows": len(status),
    "duplicate_key_rows": int(duplicate.sum()),
    "missing_station_rows": int(missing_station.sum()),
    "invalid_time_rows": int(invalid_time.sum()),
    "out_of_range_rows": int(out_of_range.sum()),
}
status["is_valid"] = ~(duplicate | missing_station | invalid_time | out_of_range)

print(report)
print(status)
```

</details>

## 9. 本讲小结

- 文件格式定义数据如何保存，不保证数据是否可信。
- 粒度决定一行代表什么，也决定唯一键和可回答的问题。
- 清洗以明确约束为依据，并保留问题与处理记录。
- 缺失值处理取决于缺失机制，填补不是默认答案。
- EDA 是对数据生成过程的连续审查。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 05：文件、粒度、变量类型与数据质量的知识边界。
- *Learning Data Science*，Ch.8–10：文件、数据整理与探索主题。
- 本页案例、文字、图解、代码、问题和练习均为本课程原创。

<DsLessonProgress lesson-id="05" next-href="/courses/data-science/06-regex" next-label="第 06 讲：文本与正则表达式" />
