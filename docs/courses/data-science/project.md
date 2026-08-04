---
course: data-science
title: 数据科学结课项目
description: 从问题定义、数据生成与审计、EDA、建模或推断，到评价局限和沟通决策的完整原创项目。
outline: [2, 3]
---

<DsLessonHero
  number="PROJECT"
  title="结课项目：从问题走到证据"
  eyebrow="完成一次可以被别人复现、质疑和改进的数据科学闭环"
  duration="建议 12–20 小时"
  :outcomes="[
    '把开放问题收敛成可验证的分析任务',
    '交付可复现的数据、代码、模型与沟通材料',
    '明确结论适用范围、失败模式和受影响者'
  ]"
/>

## 项目命题

选择一个真实但范围可控的问题，完成：

<div class="ds-flow">
  <div><b>QUESTION</b><span>提出可检验问题</span><small>用户、决策、结果</small></div>
  <div><b>DATA</b><span>获得并审计数据</span><small>来源、粒度、偏差</small></div>
  <div><b>EXPLORE</b><span>描述结构与异常</span><small>图形、分组、缺失</small></div>
  <div><b>ANALYZE</b><span>建模或推断</span><small>基线、验证、不确定性</small></div>
  <div><b>DECIDE</b><span>沟通与限制</span><small>行动、风险、监控</small></div>
</div>

可选方向：城市交通、公共能源、环境监测、教育资源、开源软件生态、个人运动与健康记录。若使用真实数据，必须确认公开许可、个人信息和使用范围。

## 自带数据选项：河川市公共服务

不希望寻找外部数据时，可以使用以下固定种子生成原创合成数据。它包含时间趋势、区域差异、缺失、类别不平衡和一个需要审查的反馈变量。

```python
import numpy as np
import pandas as pd

def make_rivercity_data(seed=10101, days=365, stations=24):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days)
    station_ids = [f"S{index:02d}" for index in range(1, stations + 1)]
    meta = pd.DataFrame({
        "station_id": station_ids,
        "district": np.repeat(["中心区", "近郊", "远郊"], [10, 8, 6]),
        "capacity": rng.integers(18, 48, stations),
    })
    frame = pd.MultiIndex.from_product([dates, station_ids], names=["date", "station_id"]).to_frame(index=False)
    frame = frame.merge(meta, on="station_id", validate="many_to_one")
    day_index = (frame["date"] - frame["date"].min()).dt.days.to_numpy()
    weekend = (frame["date"].dt.dayofweek >= 5).astype(int).to_numpy()
    temperature = 22 + 10 * np.sin(2 * np.pi * day_index / 365) + rng.normal(0, 3, len(frame))
    district_effect = frame["district"].map({"中心区": 22, "近郊": 9, "远郊": -3}).to_numpy()
    rides = 35 + district_effect + 1.3 * temperature - 12 * weekend + .7 * frame["capacity"].to_numpy() + rng.normal(0, 12, len(frame))
    imbalance_score = np.clip(rng.beta(2, 5, len(frame)) + .004 * np.maximum(rides - 90, 0), 0, 1)
    failure_probability = 1 / (1 + np.exp(-(-4.4 + 5.2 * imbalance_score + .5 * weekend)))
    frame["temperature"] = temperature.round(1)
    frame["rides"] = np.maximum(0, np.round(rides)).astype(int)
    frame["imbalance_score"] = imbalance_score.round(3)
    frame["service_failure"] = rng.binomial(1, failure_probability)
    missing = rng.choice(len(frame), size=int(.025 * len(frame)), replace=False)
    frame.loc[missing, "temperature"] = np.nan
    return frame, meta

rivercity, station_meta = make_rivercity_data()
print(rivercity.shape)
print(rivercity.head())
```

这是模拟数据，不能用于描述任何真实城市。它的价值是让你完整练习工作流，而不是产出现实政策结论。

## 阶段 1 · 项目契约

提交一页提案，必须回答：

- 谁会使用结论，做什么决策？
- 分析单位和时间范围是什么？
- 目标是描述、预测、参数推断还是因果推断？
- 至少一个可证伪的成功标准和一个停止条件是什么？
- 哪些人或群体可能因错误结论受到影响？

**闸门：** 如果问题只能写成“看看数据里有什么”，尚未形成可验收项目。

## 阶段 2 · 数据说明与质量审计

建立数据卡：来源与许可、采集或生成方式、时间范围、粒度、主键、字段定义、缺失机制、覆盖偏差和禁止用途。

最低要求：

- 自动检查主键、类型、范围、缺失和重复。
- 至少追查一个异常模式的可能生成原因。
- 所有删除、插补或截尾同时保留原始列与理由。
- 若连接多表，写出连接前后粒度与行数预测。

## 阶段 3 · EDA 与问题修订

至少完成六张必要图，而不是六张装饰图：目标分布、关键特征分布、两变量关系、分群比较、时间/空间结构、缺失或异常模式。

在 EDA 后写“问题修订记录”：哪些初始假设被支持、被否定或仍无法判断；项目范围是否改变，以及为什么。

## 阶段 4 · 分析主线

至少选择一种主线：

### 建模路线

- 简单基线与至少两个候选模型。
- 清楚的训练/验证/测试边界或时间滚动验证。
- 至少一次特征工程与一次正则化/超参数选择。
- 报告总体、分群、概率或残差诊断。

### 推断路线

- 明确总体、抽样框、参数和估计量。
- 说明抽样或观测机制及 iid 是否合理。
- Bootstrap 或其他适用区间，同时报告点估计。
- 讨论混杂、选择偏差、测量误差与区间解释。

### 无监督路线

- 解释缩放、距离和特征表示的选择。
- PCA 或聚类至少比较两种设置。
- 稳定性、重构、Silhouette 或下游效用检查。
- 用原始变量解释结构，不把算法编号当真实身份。

## 阶段 5 · 结论、影响与监控

最终结论分四栏：

| 栏目 | 必须回答 |
|---|---|
| 我们观察到什么 | 数据直接支持的描述 |
| 我们可以做什么 | 与证据强度匹配的行动建议 |
| 我们不能声称什么 | 因果、外推或代表性边界 |
| 上线后看什么 | 漂移、错误、分群表现、申诉或反馈 |

至少写一个“如果结论错了”的失败情景，并给出发现和撤回机制。

## 交付物

1. `README.md`：问题、环境、运行方式、目录。
2. `data-card.md`：数据来源、许可、字段、质量与禁止用途。
3. `analysis.ipynb` 或 `src/`：可从干净环境运行。
4. `report.md`：1,500–2,500 字，面向技术审阅者。
5. `brief.md`：不超过 500 字，面向决策者。
6. `figures/`：最终图表与可编辑生成代码。
7. `limitations.md`：失败模式、影响、监控与后续数据需求。

## 评分量规（100 分）

| 维度 | 分值 | 优秀表现 |
|---|---:|---|
| 问题与决策 | 12 | 目标具体、可证伪，分析类型与行动匹配 |
| 数据与许可 | 15 | 来源清楚、粒度正确、质量审计可复现 |
| EDA | 13 | 图形解决实际疑问，并推动问题修订 |
| 方法正确性 | 20 | 基线、验证、推断或无监督方法使用正确 |
| 诊断与稳健性 | 15 | 比较替代方案，检查残差、分群和敏感性 |
| 沟通 | 10 | 技术报告与决策简报各自适配读者 |
| 局限与影响 | 10 | 清楚说明不能声称什么及潜在伤害 |
| 可复现性 | 5 | 环境、种子、路径和运行顺序完整 |

## 参考实施框架

<details><summary>展开项目结构与验收顺序</summary>

```text
project/
├── README.md
├── data-card.md
├── requirements.txt
├── analysis.ipynb
├── report.md
├── brief.md
├── limitations.md
└── figures/
```

验收顺序：先在新环境按 README 运行；检查数据卡与主键断言；核对每张最终图是否由代码生成；核对测试集是否只在最终模型后使用；最后把报告中的每个数字追溯到 Notebook 输出。

</details>

## 学术诚信与来源

- 可以使用第三方库和公开数据，但必须记录作者、链接、版本、许可证和访问日期。
- 不得把不可公开的教材、PPT、题目或数据复制进项目仓库。
- 使用他人代码、同学讨论或生成式工具时，说明使用范围和人工验证方式。
- 本页命题、合成数据生成器、阶段闸门、交付物和评分量规均为本课程原创。

<DsLessonProgress lesson-id="PROJECT" />
