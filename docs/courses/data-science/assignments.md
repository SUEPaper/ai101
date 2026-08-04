---
course: data-science
title: 数据科学模块作业
description: 七次原创综合作业，覆盖 Pandas、EDA、回归、推断、SQL、分类以及 PCA 与聚类。
outline: [2, 3]
---

<DsLessonHero
  number="HW"
  title="七次模块综合作业"
  eyebrow="每次作业都把几讲知识连成一条小型数据科学闭环"
  duration="每次建议 90–150 分钟"
  :outcomes="[
    '把单讲概念组合成可复现分析',
    '用断言、图表与文字共同验证结论',
    '在提交中主动记录数据、模型和推断边界'
  ]"
/>

## 使用方式

每次作业都使用本页固定随机种子生成的原创合成数据，不依赖外部下载，也不包含第三方数据版权。建议新建独立 Notebook，按“问题 → 数据检查 → 实现 → 验证 → 局限”组织。

统一提交内容：

- 一个可从头运行的 `.ipynb`。
- 至少 3 个能防止静默错误的 `assert`。
- 一段不超过 200 字的结论与一段局限说明。
- 图表必须有标题、轴标签、单位；不得只贴输出不解释。

## A1 · Pandas 数据整理：站点日报

**覆盖讲次：** 01–04　　**建议用时：** 120 分钟

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(101)
raw_trips = pd.DataFrame({
    "station_id": rng.choice([" s01", "S02", "s03 ", "S04"], 240),
    "started_at": pd.Timestamp("2026-06-01") + pd.to_timedelta(rng.integers(0, 7 * 24 * 60, 240), unit="m"),
    "duration_min": np.round(rng.gamma(2.5, 7, 240), 1),
    "member_type": rng.choice(["member", "casual", None], 240, p=[.62, .34, .04]),
})
station_meta = pd.DataFrame({
    "station_id": ["S01", "S02", "S03", "S04"],
    "station": ["湖畔", "剧场", "大学城", "体育馆"],
    "district": ["北区", "南区", "北区", "西区"],
})
```

任务：

1. 统一 `station_id` 大小写与空白，验证全部能连接元数据。
2. 构造 `service_date`、`hour`、是否高峰与时长分组。
3. 生成“站点×日期”粒度日报：骑行量、时长中位数、会员比例。
4. 用 `pivot_table` 比较各区工作日与周末骑行量。
5. 找出骑行量最高的三个站点日，并保留并列规则说明。

<details><summary>参考验收点</summary>

- 清洗后只有 `S01...S04` 四个站点，连接后 `district` 不缺失。
- 日报主键 `station_id + service_date` 无重复。
- 会员比例计算时明确处理缺失 member_type 的分母。
- 解释 `groupby(...).size()` 与 `.count()` 在缺失值下的区别。

</details>

## A2 · EDA 与可视化：楼宇能耗

**覆盖讲次：** 05–09　　**建议用时：** 150 分钟

```python
rng = np.random.default_rng(202)
n = 500
buildings = pd.DataFrame({
    "building_id": np.arange(n),
    "district": rng.choice(["中心区", "近郊", "远郊"], n, p=[.5, .3, .2]),
    "area_m2": np.round(rng.lognormal(7.2, .55, n)),
    "age_years": rng.integers(1, 55, n).astype(float),
    "energy_kwh": np.round(rng.lognormal(9.1, .48, n)),
})
buildings.loc[rng.choice(n, 22, replace=False), "age_years"] = np.nan
buildings.loc[rng.choice(n, 5, replace=False), "energy_kwh"] *= 8
```

任务：

1. 写数据质量报告：粒度、类型、缺失、重复、范围与极端值。
2. 构造单位面积能耗，不得在分母为 0 时静默生成无穷大。
3. 比较原尺度与对数尺度分布，说明异常值处理方案但保留原始列。
4. 至少完成直方图、箱线图、散点图和分区小 multiples 四张图。
5. 模拟“只调查中心区”的便利样本，比较其均值与分层加权估计。

```python
# tutorial-validation: syntax-only
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data=buildings, x="energy_kwh", hue="district", element="step", common_norm=False)
plt.xlabel("年用电量（kWh）")
plt.title("不同区域楼宇能耗分布")
```

<details><summary>参考验收点</summary>

- 把“真实异常”与“录入错误”分开讨论，不因箱线图越界自动删除。
- 图形比较时分箱规则一致，或明确说明不同分箱的理由。
- 便利样本误差应联系区域构成，而不是只归因于样本量。
- 加权估计必须报告使用的总体区域比例。

</details>

## A3 · 回归与模型评估：峰值负荷

**覆盖讲次：** 10–16　　**建议用时：** 150 分钟

```python
rng = np.random.default_rng(303)
n = 360
temperature = rng.normal(26, 6, n)
humidity = np.clip(rng.normal(62, 14, n), 20, 98)
workday = rng.binomial(1, .72, n)
peak_load = 18 + .7 * temperature + .012 * temperature ** 2 + .08 * humidity + 3.5 * workday + rng.normal(0, 3.2, n)
load_data = pd.DataFrame({"temperature": temperature, "humidity": humidity, "workday": workday, "peak_load": peak_load})
```

任务：

1. 按前 70%、中间 15%、最后 15% 划分训练、验证、测试，说明为什么模拟时间任务不随机打散。
2. 建立均值基线、线性模型和含二次温度的模型。
3. 只用训练集均值/标准差缩放；报告各模型验证 RMSE 与 MAE。
4. 对 Ridge 强度执行 5-fold 选择，再只在测试集评价一次。
5. 画残差对预测图，分析弯曲、异方差和异常点。

<details><summary>参考验收点</summary>

- 设计矩阵形状与系数长度一致，截距不受正则惩罚。
- 二次模型验证误差通常优于纯线性模型，但以实际运行结果为准。
- 任何看过测试集后的模型调整都要明确标记为流程违规实验。
- 报告系数时使用“其他特征保持不变”，不写因果结论。

</details>

## A4 · 抽样与 Bootstrap：交通满意度

**覆盖讲次：** 09、17–19　　**建议用时：** 120 分钟

```python
rng = np.random.default_rng(404)
population_size = 20_000
district = rng.choice(["中心区", "近郊", "远郊"], population_size, p=[.5, .3, .2])
district_mean = pd.Series(district).map({"中心区": 82, "近郊": 72, "远郊": 59}).to_numpy()
population = pd.DataFrame({
    "district": district,
    "satisfaction": np.clip(rng.normal(district_mean, 11), 0, 100),
})
```

任务：

1. 构造中心区过度代表的 400 人便利样本与 400 人 SRS。
2. 重复 1,000 次 SRS，验证均值标准差约按 `1/√n` 缩放。
3. 从一次 SRS 构造均值和中位数的 95% Bootstrap 区间。
4. 计算中心区与远郊均值差的 Bootstrap 区间，必须按行重抽。
5. 解释 Bootstrap 为什么不能修复便利样本覆盖偏差。

<details><summary>参考验收点</summary>

- 报告总体真值只用于模拟验收；现实调查中它通常未知。
- 每次 Bootstrap 重抽样本大小与原样本相同且有放回。
- 固定随机种子，并报告重复次数 B。
- 区间解释使用重复抽样覆盖率语言，不写成参数的后验概率。

</details>

## A5 · SQL 数据分析：运营数据库

**覆盖讲次：** 20–21　　**建议用时：** 120 分钟

```python
import sqlite3

sql_connection = sqlite3.connect(":memory:")
sql_connection.executescript("""
CREATE TABLE stations (station_id TEXT PRIMARY KEY, station TEXT, district TEXT);
CREATE TABLE trips (trip_id INTEGER PRIMARY KEY, station_id TEXT, service_date TEXT, duration_min REAL);
CREATE TABLE repairs (ticket_id INTEGER PRIMARY KEY, station_id TEXT, opened_date TEXT, closed_date TEXT);
INSERT INTO stations VALUES ('S01','湖畔','北区'),('S02','剧场','南区'),('S03','大学城','北区'),('S04','体育馆','西区');
INSERT INTO trips VALUES
(1,'S01','2026-07-01',12),(2,'S01','2026-07-01',25),(3,'S02','2026-07-01',18),(4,'S03','2026-07-01',9),
(5,'S03','2026-07-02',31),(6,'S03','2026-07-02',14),(7,'S04','2026-07-02',20),(8,'S01','2026-07-02',16);
INSERT INTO repairs VALUES (101,'S01','2026-06-29','2026-07-01'),(102,'S01','2026-07-02',NULL),(103,'S03','2026-07-01',NULL);
""")
```

任务：

1. 查询每区站点数，保留没有 trips 的站点。
2. 计算每站每天 trip 数与时长中位替代指标（SQLite 可先用平均值并说明限制）。
3. 找出 trip 数前两名的站点日，处理并列。
4. 计算每站未关闭工单数，不得因一对多 Join 重复 trip。
5. 使用至少两个 CTE 生成区域运营报告，并用断言检查行数和主键。

<details><summary>参考验收点</summary>

- Join 前写出每张表粒度和预计连接基数。
- “无工单”用 0，“平均严重度未知”保留 NULL，不混为一谈。
- 查询不使用 `SELECT *`，用户值使用参数化绑定。
- 用 Pandas 复算一个汇总结果作为交叉检查。

</details>

## A6 · 分类诊断：站点故障预警

**覆盖讲次：** 22–23　　**建议用时：** 150 分钟

```python
rng = np.random.default_rng(606)
n = 1200
imbalance = rng.beta(2, 3, n)
event = rng.binomial(1, .18, n)
rain = rng.binomial(1, .25, n)
logit = -4.2 + 5.4 * imbalance + 1.1 * event + .7 * rain
failure_probability = 1 / (1 + np.exp(-logit))
failure = rng.binomial(1, failure_probability)
failure_data = pd.DataFrame({"imbalance": imbalance, "event": event, "rain": rain, "failure": failure})
```

任务：

1. 实现带 L2 的逻辑回归，绘制训练交叉熵。
2. 报告验证集 Accuracy、Precision、Recall、Specificity、F1。
3. 绘制 ROC 与 PR，并说明基准率对 PR 的影响。
4. 在“FN 成本 8、FP 成本 2”下选择阈值；再加入每日容量约束比较。
5. 按 rain 分群报告 FPR/FNR，并讨论差异来源和不确定性。

<details><summary>参考验收点</summary>

- 概率裁剪避免 `log(0)`，训练/验证/测试边界清楚。
- 阈值只能在验证集选择，测试集只用于最终报告。
- 同时报告混淆矩阵绝对数与比率。
- 不把分群差异直接等同于歧视结论，要检查样本量、标签和数据生成机制。

</details>

## A7 · PCA 与聚类：站点画像

**覆盖讲次：** 24–26　　**建议用时：** 150 分钟

```python
rng = np.random.default_rng(707)
base_profiles = np.array([
    [1.8, .7, .6, 1.9],   # 通勤型
    [.8, 1.3, 1.4, .9],   # 日间型
    [.6, .9, 1.1, .7],    # 社区型
])
true_group = rng.integers(0, 3, 180)
profiles = base_profiles[true_group] + rng.normal(0, .18, size=(180, 4))
profile_data = pd.DataFrame(profiles, columns=["morning", "midday", "weekend", "evening"])
```

任务：

1. 比较不缩放与标准化后的 PCA，解释方向差异。
2. 报告 Scree、累计解释方差、PC1/PC2 loadings 和二维 scores 图。
3. 用 K-means 比较 K=2...6，每个 K 运行 20 次初始化。
4. 报告 inertia、Silhouette 与初始化稳定性，选择候选 K。
5. 用原始变量解释每个簇，并说明不能把簇编号当作客观站点身份。

<details><summary>参考验收点</summary>

- PCA 和缩放器只在分析数据上拟合；若用于监督下游任务则必须只拟合训练集。
- 验证 `scores @ components.T + mean` 的重构形状正确。
- 不只按最小 inertia 选择 K，因为 inertia 随 K 增加不会上升。
- 报告不同随机种子结果的一致程度。

</details>

## 统一评分量规

| 维度 | 权重 | 达标证据 |
|---|---:|---|
| 正确性 | 35% | 数据粒度、公式、代码和指标正确，关键结果有断言 |
| 推理过程 | 25% | 选择方法前说明任务、假设与替代方案 |
| 可复现性 | 15% | 从头运行、固定种子、无隐藏状态、环境说明清楚 |
| 表达 | 15% | 图表可读，结论回答问题，不罗列输出 |
| 边界意识 | 10% | 讨论数据质量、泛化、推断或社会影响限制 |

## 参考资料与原创性

- 作业知识范围与对应讲次一致，未翻译或复刻 Berkeley Data 100 原作业。
- 所有案例、数据生成代码、任务、验收点和评分量规均为本课程原创。
- 学习者可讨论思路，但提交必须注明协作、外部代码与生成式工具使用情况。

<DsLessonProgress lesson-id="HW" next-label="结课项目：完成一次数据科学闭环" next-href="/courses/data-science/project" />
