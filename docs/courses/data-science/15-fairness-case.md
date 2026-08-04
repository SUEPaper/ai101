---
course: data-science
title: 案例：房产估价与公平性
description: 用虚构的保障房租金评估案例审查历史数据、群体误差、人类影响、反馈回路与申诉机制。
outline: [2, 3]
---

<DsLessonHero
  number="15"
  title="案例：房产估价与公平性"
  eyebrow="模型误差进入制度后，不再只是一个数字"
  duration="建议 100–120 分钟"
  :outcomes="[
    '把公平性审查放回完整数据科学生命周期',
    '分群计算误差与严重低估率，识别总体指标的遮蔽',
    '提出数据、模型、流程和申诉四层缓解措施'
  ]"
/>

## 案例说明

本讲使用完全虚构的“河川市保障房租金评估”案例。市住房服务中心希望用模型估计片区参考租金，辅助审核补贴额度。模型不会直接作出最终决定，但估计结果会影响材料复核优先级。

案例中的城市、机构、数据、群体和数值均为本课程原创，不复用 Berkeley 课程案例的机构、数据或叙事。

## 1. 从“能不能预测”退回“该不该建模”

在写第一行代码前，要回答：

<div class="ds-flow">
  <div><b>PURPOSE</b><span>模型服务谁</span><small>辅助审核还是自动拒绝</small></div>
  <div><b>HARM</b><span>错误伤害谁</span><small>少补、延迟或重复举证</small></div>
  <div><b>ALTERNATIVE</b><span>有没有更低风险方案</span><small>规则、抽查、增员</small></div>
  <div><b>RECOURSE</b><span>人如何纠错</span><small>解释、复核、申诉</small></div>
</div>

若目标含糊地写成“提高效率”，团队很容易只优化平均误差，而忽略效率收益和错误代价分别落在谁身上。

## 2. 历史成交价不是天然的真实标签

历史数据记录了过去的市场与制度，不是无偏的“房屋价值”。老城区可能因为交易稀少、房龄记录缺失、公共设施更新未入库而长期被低估。

```python
import pandas as pd
import numpy as np

audit = pd.DataFrame({
    "district_group": ["新城区"] * 4 + ["老城区"] * 4,
    "actual_rent": [92, 105, 118, 110, 82, 96, 110, 104],
    "predicted_rent": [95, 102, 121, 108, 68, 80, 91, 88],
    "record_age_missing": [0, 0, 0, 0, 1, 0, 1, 1],
})

audit["error"] = audit["predicted_rent"] - audit["actual_rent"]
audit["absolute_error"] = audit["error"].abs()
audit["severe_underestimate"] = audit["predicted_rent"] < 0.9 * audit["actual_rent"]
print(audit)
```

代理变量也值得审查。即使删除片区名称，建筑年代、通勤时间、维修记录缺失率仍可能高度指示片区。

## 3. 总体 MAE 会遮住方向与群体

```python
overall_mae = audit["absolute_error"].mean()
group_report = audit.groupby("district_group").agg(
    count=("actual_rent", "size"),
    mae=("absolute_error", "mean"),
    mean_error=("error", "mean"),
    severe_under_rate=("severe_underestimate", "mean"),
)

print("总体 MAE：", overall_mae)
print(group_report)
```

`mean_error` 保留方向：负值表示系统低估。`severe_under_rate` 接近政策后果，因为超过 10% 的低估更可能触发补充材料或减少初始补贴额度。

<FairnessLab />

滑块展示“总体优化”可能掩盖的权衡，但真实系统不能简单按群体给预测加一个补丁。受保护属性的使用、差别待遇、合理便利与纠偏措施需要法律、政策和社区共同判断。

<QuizBlock question="总体 MAE 较低，为什么仍要分群检查？" :options="['分群能保证因果', '相反方向或少数群体的大误差可能在总体平均中被稀释', '总体 MAE 没有单位', '模型只能处理一个群体']" :answer="1" explanation="总体平均会按样本构成加权，人数少或方向相反的错误可能被遮蔽。" />

## 4. 公平性不只有一个数学定义

不同目标可能冲突：

| 关注点 | 示例问题 | 可能的指标 |
|---|---|---|
| 预测准确性 | 各群体平均偏差是否相近 | MAE、RMSE、均值误差 |
| 错误方向 | 是否持续低估某群体 | 严重低估率 |
| 决策结果 | 谁更常进入人工复核 | 复核率、拒绝率 |
| 过程公平 | 是否能够理解和纠错 | 解释覆盖率、申诉成功率 |
| 长期影响 | 系统是否加剧资料缺失 | 重复举证、退出率、反馈回路 |

指标不能替代价值选择。团队应先写清楚伤害与权利，再选择能观察这些问题的指标。

## 5. 模型会制造新的训练数据

如果被低估的申请需要更多证明，部分人可能放弃申请；下一轮数据就更少包含他们。模型不是被动观察者，而是改变未来可见数据的行动者。

<figure class="ds-figure">
  <div class="ds-cycle">
    <span>历史记录不完整</span><i>→</i><span>模型在老城区误差更大</span><i>→</i><span>更多人被要求补材料</span><i>→</i><span>部分申请退出</span><i>→</i><span>未来记录更不完整</span>
  </div>
  <figcaption>反馈回路使一次部署改变下一批数据的生成过程。</figcaption>
</figure>

因此上线监控不能只看输入分布漂移，还要看流程结果、申诉、退出和群体覆盖。

## 6. 四层缓解比“换模型”更完整

### 数据层

- 与一线审核员和居民核实字段含义与缺失原因。
- 主动补采长期缺失的维修、面积和设施数据。
- 记录数据时间、覆盖范围和测量误差。

### 模型层

- 报告分群误差和方向，不只报告总体分数。
- 检查代理变量、样本量和不确定性。
- 对高风险、低置信度输入设置人工接管。

### 流程层

- 模型只排序待复核材料，不自动拒绝。
- 审核员看到证据与不确定性，不只看到一个分数。
- 记录人工覆盖模型建议的原因，用于审计而非惩罚。

### 权利层

- 告知申请人模型参与了哪个环节。
- 提供可理解的理由、人工复核和时间明确的申诉。
- 定期邀请受影响群体参与目标与指标复审。

<div class="ds-warning"><strong>“人工在环”不是万能保证</strong>如果人员没有时间、证据或推翻权限，人工确认只会把自动化偏差盖上人工印章。</div>

## 7. 用数据卡记录可审计边界

```python
data_card = {
    "purpose": "辅助确定保障房租金材料复核优先级",
    "not_for": ["自动拒绝", "跨城市外推"],
    "time_range": "2024-01 至 2025-12",
    "known_gaps": ["老城区房龄缺失较多", "短租不在样本中"],
    "monitor": ["分群 MAE", "严重低估率", "申诉率", "退出率"],
    "owner": "住房服务中心模型治理小组",
}

for key, value in data_card.items():
    print(f"{key}: {value}")
```

数据卡和模型卡不是一次性合规附件。它们应随数据、用途和流程变化而更新，并明确谁对异常负责。

## 8. 知识检查

<QuizBlock question="历史成交价为何不能自动视为无偏真值？" :options="['价格不是数字', '历史市场、覆盖与制度差异会进入标签', '模型不能预测价格', '只要样本大就无偏']" :answer="1" explanation="标签由现实过程生成，过去的不平等和测量缺口会被数据继承。" />

<QuizBlock question="删除片区名称后是否一定消除了片区信息？" :options="['是', '否，其他特征可能成为代理变量', '只对线性回归是', '只要 R² 高就是']" :answer="1" explanation="建筑年代、通勤或缺失模式等变量仍可能高度指示片区。" />

<QuizBlock question="以下哪项最接近有效申诉机制？" :options="['展示一个模型分数', '提供理由、人工复核、证据补充和明确时限', '让同一模型再次评分', '只允许开发者查看']" :answer="1" explanation="申诉需要知情、纠错渠道、真正有权改变结果的人和可预期流程。" />

<QuizBlock question="反馈回路为何重要？" :options="['模型部署会改变未来行为和可见数据', '反馈只影响网页颜色', '它保证公平', '它让样本量固定']" :answer="0" explanation="系统决策改变申请、退出和记录过程，未来训练数据不再独立于模型。" />

## 9. 代码练习：生成群体审计表

写函数 `fairness_report(frame, group, actual, predicted)`，对每组报告样本量、MAE、RMSE、均值误差、严重低估率，并标记样本量小于 3 的组。

<details><summary>提示</summary>

先复制数据并构造 `error`、`squared_error`、`absolute_error` 与布尔列，再 `groupby().agg()`。

</details>

<details><summary>参考实现</summary>

```python
def fairness_report(frame, group, actual, predicted):
    work = frame.copy()
    work["error"] = work[predicted] - work[actual]
    work["absolute_error"] = work["error"].abs()
    work["squared_error"] = work["error"] ** 2
    work["severe_under"] = work[predicted] < 0.9 * work[actual]
    report = work.groupby(group).agg(
        count=(actual, "size"),
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        mean_error=("error", "mean"),
        severe_under_rate=("severe_under", "mean"),
    )
    report["rmse"] = np.sqrt(report.pop("mse"))
    report["small_group_warning"] = report["count"] < 3
    return report

report = fairness_report(audit, "district_group", "actual_rent", "predicted_rent")
print(report)
```

</details>

## 10. 本讲小结

- 先判断任务是否值得建模，再讨论怎样提高分数。
- 历史标签和代理变量会携带既有制度与覆盖差异。
- 公平性需要结合分群误差、方向、决策结果、程序权利和长期反馈。
- 数据、模型、流程与申诉必须一起设计，不能把责任交给单一指标。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 15：房产估价案例、数据科学生命周期、公平性与人类影响的主题边界。
- 本页“河川市保障房租金评估”案例、机构、数据、文字、图解、代码、问题和实验均为本课程原创。
- 本页内容是课程中的技术与治理分析，不构成针对具体制度的法律意见。

<DsLessonProgress lesson-id="15" next-label="第 16 讲：交叉验证与正则化" next-href="/courses/data-science/16-cross-validation" />
