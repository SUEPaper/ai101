---
course: data-science
title: SQL：查询一张表
description: 从关系表、Schema 和查询逻辑顺序出发，掌握 SELECT、WHERE、NULL、ORDER BY 与 LIMIT。
outline: [2, 3]
---

<DsLessonHero
  number="20"
  title="SQL：查询一张表"
  eyebrow="SQL 描述想得到的结果，不要求你逐行安排执行过程"
  duration="建议 100–120 分钟"
  :outcomes="[
    '读懂关系表、行、列、Schema 与主键',
    '使用 SELECT、WHERE、ORDER BY、LIMIT 构造单表查询',
    '正确处理 NULL、别名、表达式和逻辑顺序'
  ]"
/>

## 本讲要解决的问题

共享单车系统每天产生数百万行记录。把整张表加载进 Pandas 再筛选既慢又浪费内存。SQL 可以让数据库在存储附近完成筛选、排序和计算，只返回真正需要的结果。

## 1. 表不只是网格，还有结构合同

一张关系表由行和有名称、类型的列组成。Schema 记录表名、列名、类型、是否允许空值及约束。

```python
import sqlite3
import pandas as pd

connection = sqlite3.connect(":memory:")
connection.executescript("""
CREATE TABLE station_day (
    service_date TEXT NOT NULL,
    station_id TEXT NOT NULL,
    station TEXT NOT NULL,
    district TEXT,
    rides INTEGER CHECK (rides >= 0),
    PRIMARY KEY (service_date, station_id)
);
INSERT INTO station_day VALUES
('2026-07-01','S01','湖畔','北区',128),
('2026-07-01','S02','剧场','南区',94),
('2026-07-01','S03','大学城','北区',173),
('2026-07-01','S04','体育馆','西区',81),
('2026-07-01','S05','书城','南区',116),
('2026-07-01','S06','港口',NULL,142);
""")

schema = pd.read_sql_query("PRAGMA table_info(station_day)", connection)
print(schema[["name", "type", "notnull", "pk"]])
```

这里的主键由日期和站点编号共同组成：同一站点每天一行。主键约束防止无意重复，也让连接关系更清楚。

## 2. SELECT 选择列，FROM 选择来源

```sql
SELECT station, rides
FROM station_day;
```

```python
query = """
SELECT station, rides
FROM station_day
"""
print(pd.read_sql_query(query, connection))
```

`SELECT *` 适合探索小表，不适合作为稳定生产接口：Schema 增加列会改变返回形状，还可能读取无用的大字段或敏感字段。

## 3. WHERE 保留条件为 TRUE 的行

```sql
SELECT station, district, rides
FROM station_day
WHERE rides >= 120 AND district = '北区';
```

字符串用单引号；双引号在标准 SQL 中用于标识符。组合条件时明确加括号，避免误解 `AND` 与 `OR` 的优先级。

```python
query = """
SELECT station, district, rides
FROM station_day
WHERE rides >= 120 AND district = '北区'
"""
print(pd.read_sql_query(query, connection))
```

<QuizBlock question="WHERE rides >= 100 AND district='北区' 保留哪类行？" :options="['满足任一条件', '同时满足两个条件', '所有行', '只保留 NULL']" :answer="1" explanation="AND 要求左右条件都为 TRUE。" />

## 4. NULL 代表未知，不等于空字符串或 0

`district = NULL` 不会得到 TRUE，因为未知值参与比较会产生 UNKNOWN。应使用 `IS NULL` 或 `IS NOT NULL`。

```sql
SELECT station, district
FROM station_day
WHERE district IS NULL;
```

```python
missing_district = pd.read_sql_query(
    "SELECT station, district FROM station_day WHERE district IS NULL",
    connection,
)
print(missing_district)
```

`WHERE` 只保留 TRUE，FALSE 和 UNKNOWN 都被过滤。这个三值逻辑是 SQL 中许多“明明不等于却没选中”的根源。

## 5. SELECT 可以构造新列

```sql
SELECT
    station,
    rides,
    ROUND(rides / 24.0, 1) AS rides_per_hour,
    CASE WHEN rides >= 120 THEN '繁忙' ELSE '一般' END AS load_level
FROM station_day;
```

别名 `AS` 只改变结果列名，不改变源表。`24.0` 明确要求小数除法，避免某些数据库方言执行整数除法。

<SqlQueryLab />

## 6. ORDER BY 与 LIMIT 控制结果顺序和规模

关系表没有默认行顺序。需要“最高的三个站点”时必须明确排序。

```python
top_three = pd.read_sql_query("""
SELECT station, rides
FROM station_day
WHERE rides IS NOT NULL
ORDER BY rides DESC, station ASC
LIMIT 3
""", connection)
print(top_three)
```

第二个排序键解决 rides 相同时的稳定顺序。没有 `ORDER BY` 的 `LIMIT 3` 只代表数据库任选三行，不代表最早或最高。

## 7. 查询书写顺序与逻辑顺序不同

书写顺序是 `SELECT → FROM → WHERE → ORDER BY → LIMIT`。用于理解的逻辑顺序更接近：

<div class="ds-equation"><b>FROM</b><i>→</i><b>WHERE</b><i>→</i><b>SELECT</b><i>→</i><b>ORDER BY</b><i>→</i><b>LIMIT</b></div>

因此 `WHERE` 通常不能直接引用同层 `SELECT` 中刚定义的别名；可以使用子查询或 CTE 把结果变成下一层输入。

<div class="ds-warning"><strong>不要用字符串拼接用户输入</strong>参数化查询会把数据与 SQL 结构分开，降低 SQL 注入风险。表名和列名不能直接当普通参数，应使用白名单映射。</div>

```python
minimum_rides = 110
safe_query = "SELECT station, rides FROM station_day WHERE rides >= ? ORDER BY rides DESC"
print(pd.read_sql_query(safe_query, connection, params=[minimum_rides]))
```

## 8. 知识检查

<QuizBlock question="没有 ORDER BY 的查询是否保证返回插入顺序？" :options="['保证', '不保证', '只对 SQLite 保证', '有主键就保证']" :answer="1" explanation="关系结果没有固有顺序，执行计划或存储变化都可能改变返回顺序。" />

<QuizBlock question="筛选缺失 district 的正确写法是什么？" :options="['district = NULL', 'district IS NULL', 'district == None', 'NULL(district)']" :answer="1" explanation="SQL 用 IS NULL 判断未知值。" />

<QuizBlock question="为什么稳定接口通常避免 SELECT *？" :options="['星号不能执行', 'Schema 变化会意外改变返回列并读取无用数据', '会自动删除行', '只能选择数字']" :answer="1" explanation="显式列清单让数据合同和读取成本更可控。" />

<QuizBlock question="LIMIT 5 放在无排序查询中表示什么？" :options="['最高 5 行', '数据库返回任意 5 行', '最早 5 行', '随机样本']" :answer="1" explanation="没有排序标准，“前 5”没有业务意义，也不是概率随机样本。" />

## 9. 代码练习：写一条参数化查询

写函数 `busy_stations(connection, minimum, district=None)`：返回 `rides >= minimum` 的站点；若给出区域再增加区域条件；结果按 rides 降序、station 升序，最多 5 行。不得拼接参数值。

<details><summary>提示</summary>

SQL 结构可以按是否有 district 选择两个固定模板，值统一放在 `params` 列表中。

</details>

<details><summary>参考实现</summary>

```python
def busy_stations(connection, minimum, district=None):
    if district is None:
        sql = """
        SELECT station, district, rides
        FROM station_day
        WHERE rides >= ?
        ORDER BY rides DESC, station ASC
        LIMIT 5
        """
        params = [minimum]
    else:
        sql = """
        SELECT station, district, rides
        FROM station_day
        WHERE rides >= ? AND district = ?
        ORDER BY rides DESC, station ASC
        LIMIT 5
        """
        params = [minimum, district]
    return pd.read_sql_query(sql, connection, params=params)

print(busy_stations(connection, 100))
print(busy_stations(connection, 100, "北区"))
```

</details>

## 10. 本讲小结

- Schema 定义表的列、类型和约束；主键定义行的身份。
- SQL 通过 SELECT、FROM、WHERE 描述所需结果，NULL 使用三值逻辑。
- 排名必须显式 `ORDER BY`，`LIMIT` 本身不代表最高、最早或随机。
- 用户值使用参数化查询，不与 SQL 字符串直接拼接。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 20：关系数据库、Schema、SELECT、WHERE、NULL、ORDER BY 与 LIMIT 的知识边界。
- *Learning Data Science*，Ch.7：关系数据与 SQL 主题。
- 本页共享单车数据库、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="20" next-label="第 21 讲：SQL 分组与连接" next-href="/courses/data-science/21-sql-joins" />
