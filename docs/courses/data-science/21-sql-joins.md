---
course: data-science
title: SQL：分组与连接
description: 使用 GROUP BY、HAVING、CASE、Join、CTE 和窗口函数，在多张关系表中构造可审计分析。
outline: [2, 3]
---

<DsLessonHero
  number="21"
  title="SQL：分组与连接"
  eyebrow="连接之前先写清每张表的粒度，否则行数会替你做出错误决定"
  duration="建议 110–130 分钟"
  :outcomes="[
    '使用 GROUP BY 与 HAVING 生成分组统计',
    '根据保留未匹配行的需要选择 Join 类型',
    '用 CTE 拆分多阶段查询并检查连接基数'
  ]"
/>

## 本讲要解决的问题

站点日报只包含骑行量；站点维表保存区域；维修工单表一站可能多行。要回答“各区骑行量与维修负担是否有关”，必须先理解三张表各自的粒度，再聚合和连接。

## 1. GROUP BY 把多行压成每组一行

```python
import sqlite3
import pandas as pd

connection = sqlite3.connect(":memory:")
connection.executescript("""
CREATE TABLE stations (station_id TEXT PRIMARY KEY, station TEXT, district TEXT);
CREATE TABLE station_day (service_date TEXT, station_id TEXT, rides INTEGER);
CREATE TABLE repairs (ticket_id TEXT PRIMARY KEY, station_id TEXT, severity INTEGER, closed INTEGER);
INSERT INTO stations VALUES ('S01','湖畔','北区'),('S02','剧场','南区'),('S03','大学城','北区'),('S04','体育馆','西区');
INSERT INTO station_day VALUES
('2026-07-01','S01',128),('2026-07-01','S02',94),('2026-07-01','S03',173),('2026-07-01','S04',81),
('2026-07-02','S01',135),('2026-07-02','S02',101),('2026-07-02','S03',165),('2026-07-02','S04',89);
INSERT INTO repairs VALUES ('R7','S01',2,1),('R9','S01',3,0),('R10','S03',1,1),('R12','S05',2,0);
""")

query = """
SELECT station_id, COUNT(*) AS days, AVG(rides) AS avg_rides, SUM(rides) AS total_rides
FROM station_day
GROUP BY station_id
"""
print(pd.read_sql_query(query, connection))
```

聚合后粒度从“站点×日期一行”变成“站点一行”。`COUNT(*)` 数行；`COUNT(column)` 只数非 NULL 值。

## 2. WHERE 过滤行，HAVING 过滤组

```sql
SELECT station_id, AVG(rides) AS avg_rides
FROM station_day
WHERE service_date >= '2026-07-01'
GROUP BY station_id
HAVING AVG(rides) >= 120;
```

逻辑上 `WHERE` 在分组前工作，`HAVING` 在聚合后工作。能在分组前排除的行尽量用 WHERE，语义更清楚，也可能减少计算。

<QuizBlock question="筛选平均 rides 大于 120 的组，应该使用什么？" :options="['WHERE AVG(rides)>120', 'HAVING AVG(rides)>120', 'ORDER BY only', 'LIMIT']" :answer="1" explanation="平均值在 GROUP BY 后产生，组级条件使用 HAVING。" />

## 3. CASE 把业务规则写入结果列

```python
case_query = """
SELECT
  station_id,
  SUM(CASE WHEN rides >= 120 THEN 1 ELSE 0 END) AS busy_days,
  COUNT(*) AS observed_days
FROM station_day
GROUP BY station_id
ORDER BY busy_days DESC
"""
print(pd.read_sql_query(case_query, connection))
```

条件聚合把“符合条件的行记 1”再求和。生产查询中应把 120 的来源和单位写进数据合同，避免阈值变成无人知晓的魔法数字。

## 4. Join 的核心是匹配关系与保留规则

- `INNER JOIN`：只保留两侧匹配行。
- `LEFT JOIN`：保留左表所有行，右侧未匹配填 NULL。
- `FULL OUTER JOIN`：两侧未匹配行都保留；不同数据库支持情况不同。

<SqlJoinLab />

一对多连接会复制左表行。S01 有两个工单，连接后 S01 出现两行；如果随后直接汇总 rides，就会重复计算。

```python
duplicated = pd.read_sql_query("""
SELECT d.station_id, d.rides, r.ticket_id
FROM station_day AS d
JOIN repairs AS r ON d.station_id = r.station_id
WHERE d.service_date = '2026-07-01'
""", connection)
print(duplicated)
```

<div class="ds-warning"><strong>Join 前先预测行数</strong>写下左右表粒度、连接键是否唯一、未匹配行如何处理。运行后检查总行数、键重复和 NULL 数量。</div>

## 5. 先聚合，再连接，控制基数

要把“每站工单数”连接到“每站平均骑行量”，先把两边都变成每站一行。

```python
safe_join = """
WITH ride_summary AS (
  SELECT station_id, AVG(rides) AS avg_rides
  FROM station_day
  GROUP BY station_id
),
repair_summary AS (
  SELECT station_id, COUNT(*) AS repair_count, AVG(severity) AS avg_severity
  FROM repairs
  GROUP BY station_id
)
SELECT
  s.station_id,
  s.station,
  s.district,
  r.avg_rides,
  COALESCE(p.repair_count, 0) AS repair_count,
  p.avg_severity
FROM stations AS s
LEFT JOIN ride_summary AS r ON s.station_id = r.station_id
LEFT JOIN repair_summary AS p ON s.station_id = p.station_id
ORDER BY r.avg_rides DESC
"""
print(pd.read_sql_query(safe_join, connection))
```

`COALESCE` 把“没有匹配工单”转换为 0，但平均严重度仍保留 NULL：没有工单时，严重度不是 0，而是没有定义。

## 6. CTE 给复杂查询命名阶段

`WITH ... AS (...)` 创建只在当前查询内存在的命名结果。它适合表达“先按站点汇总，再连接，再按区域汇总”的数据流。

<div class="ds-equation"><b>原始明细</b><i>→</i><b>ride_summary</b><i>→</i><b>joined</b><i>→</i><b>district_report</b></div>

CTE 提高可读性，不保证物化或性能。性能仍要看数据库执行计划、索引、统计信息和数据规模。

## 7. 窗口函数保留原行粒度

`GROUP BY` 折叠行；窗口函数在不折叠原行的情况下计算组内排名、累计值或移动统计。

```python
ranked = pd.read_sql_query("""
SELECT
  service_date,
  station_id,
  rides,
  RANK() OVER (PARTITION BY service_date ORDER BY rides DESC) AS daily_rank
FROM station_day
ORDER BY service_date, daily_rank
""", connection)
print(ranked)
```

同一日期的各行仍然存在，只新增了组内排名。窗口函数通常在 WHERE 之后计算；要按窗口结果筛选，可放进 CTE 再筛。

## 8. 知识检查

<QuizBlock question="LEFT JOIN 中右表未匹配时会怎样？" :options="['左行被删除', '左行保留，右侧列为 NULL', '查询报错', '自动复制两次']" :answer="1" explanation="LEFT JOIN 的保留侧是左表。" />

<QuizBlock question="一对多 Join 的常见风险是什么？" :options="['左表行被复制，后续聚合可能重复计数', '所有 NULL 消失', '主键自动修改', '结果一定为空']" :answer="0" explanation="每个右侧匹配会生成一行，左侧数值也随之重复。" />

<QuizBlock question="COALESCE(repair_count,0) 的含义是什么？" :options="['删除工单', '用 0 替代 NULL 工单数', '把所有数改为 0', '连接主键']" :answer="1" explanation="COALESCE 返回第一个非 NULL 参数。" />

<QuizBlock question="窗口函数与 GROUP BY 的关键差异是什么？" :options="['窗口函数保留原行粒度', '窗口函数不能排序', 'GROUP BY 保留所有明细行', '两者完全相同']" :answer="0" explanation="窗口结果附加到每一原始行，GROUP BY 则把多行折叠为组。" />

## 9. 代码练习：区域运营报告

写一条 SQL：先计算每站平均 rides 与未关闭工单数，再按区域汇总站点数、平均站点 rides、未关闭工单总数；保留没有工单的站点，只显示至少 1 个站点的区域。

<details><summary>提示</summary>

建立两个 CTE：`rides_by_station` 与 `open_repairs_by_station`。连接到 stations 时用 LEFT JOIN 和 COALESCE，最后按 district 分组。

</details>

<details><summary>参考实现</summary>

```python
district_report_sql = """
WITH rides_by_station AS (
  SELECT station_id, AVG(rides) AS avg_rides
  FROM station_day
  GROUP BY station_id
),
open_repairs_by_station AS (
  SELECT station_id, SUM(CASE WHEN closed = 0 THEN 1 ELSE 0 END) AS open_repairs
  FROM repairs
  GROUP BY station_id
)
SELECT
  s.district,
  COUNT(*) AS station_count,
  ROUND(AVG(r.avg_rides), 1) AS avg_station_rides,
  SUM(COALESCE(p.open_repairs, 0)) AS open_repairs
FROM stations AS s
LEFT JOIN rides_by_station AS r ON s.station_id = r.station_id
LEFT JOIN open_repairs_by_station AS p ON s.station_id = p.station_id
GROUP BY s.district
HAVING COUNT(*) >= 1
ORDER BY avg_station_rides DESC
"""
print(pd.read_sql_query(district_report_sql, connection))
```

</details>

## 10. 本讲小结

- GROUP BY 改变结果粒度，WHERE 过滤行，HAVING 过滤聚合后的组。
- Join 类型决定未匹配行是否保留；连接基数决定行是否复制。
- 先按目标粒度聚合再连接，可以避免重复统计。
- CTE 命名查询阶段，窗口函数在保留明细行的同时计算组内统计。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 21：聚合、HAVING、CASE、Join、CTE 和窗口计算的知识边界。
- *Learning Data Science*，Ch.7：多表 SQL 分析主题。
- 本页共享单车数据库、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="21" next-label="第 22 讲：逻辑回归" next-href="/courses/data-science/22-logistic-regression" />
