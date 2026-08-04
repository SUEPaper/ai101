---
course: data-science
title: 分类指标与阈值
description: 用混淆矩阵、Precision、Recall、ROC、PR 和校准评价概率模型，并按错误代价选择阈值。
outline: [2, 3]
---

<DsLessonHero
  number="23"
  title="分类指标与阈值"
  eyebrow="模型给出风险排序，阈值把排序变成资源与错误的分配"
  duration="建议 110–130 分钟"
  :outcomes="[
    '从混淆矩阵计算 Accuracy、Precision、Recall 与 Specificity',
    '解释阈值移动如何交换假阳性与假阴性',
    '根据类不平衡、错误代价和概率校准选择评价方式'
  ]"
/>

## 本讲要解决的问题

站点故障预警可以提前派出调度车辆。漏报会造成服务中断，误报会浪费调度资源。0.5 只是数学默认值，不知道两种错误的代价，也不知道每天可处理多少预警。

## 1. 混淆矩阵记录四种结果

```python
import numpy as np
import pandas as pd

probability = np.array([.94,.82,.76,.67,.58,.49,.41,.33,.22,.12])
actual = np.array([1,1,0,1,0,1,0,0,1,0])

def confusion_counts(actual, probability, threshold):
    predicted = probability >= threshold
    actual = np.asarray(actual, dtype=bool)
    return {
        "tp": int(np.sum(predicted & actual)),
        "fp": int(np.sum(predicted & ~actual)),
        "tn": int(np.sum(~predicted & ~actual)),
        "fn": int(np.sum(~predicted & actual)),
    }

print(confusion_counts(actual, probability, 0.5))
```

- TP：正确预警故障。
- FP：预警但未故障，消耗资源。
- TN：正确不预警。
- FN：未预警但故障，造成漏报。

<MetricThresholdLab />

阈值下降时，更多样本判为正类：Recall 通常上升，FP 也通常增加。阈值不是模型内部真理，而是部署决策。

## 2. 四个常用比率回答不同问题

```python
def classification_metrics(counts):
    tp, fp, tn, fn = (counts[key] for key in ["tp", "fp", "tn", "fn"])
    safe = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    return {
        "accuracy": safe(tp + tn, tp + fp + tn + fn),
        "precision": safe(tp, tp + fp),
        "recall": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "f1": safe(2 * tp, 2 * tp + fp + fn),
    }

for threshold in [0.3, 0.5, 0.8]:
    print(threshold, classification_metrics(confusion_counts(actual, probability, threshold)))
```

- Precision：所有预警中，多少真的故障。
- Recall / TPR：所有真实故障中，多少被预警。
- Specificity / TNR：所有真实正常中，多少没有误报。
- F1：Precision 与 Recall 的调和平均，但不包含 TN。

<QuizBlock question="漏掉真实故障非常昂贵时，应优先关注哪个指标？" :options="['Recall', 'Specificity only', '样本量', 'R²']" :answer="0" explanation="Recall 的分母是真实正类，直接衡量有多少故障被检出。" />

## 3. 类不平衡会让 Accuracy 失去警觉

若 10,000 个站点日中只有 100 次故障，全部预测“正常”也有 99% Accuracy，却完全没有预警价值。

```python
rare_actual = np.array([1] * 100 + [0] * 9900)
always_negative = np.zeros_like(rare_actual)
accuracy = np.mean(always_negative == rare_actual)
recall = np.sum((always_negative == 1) & (rare_actual == 1)) / np.sum(rare_actual == 1)

print("Accuracy：", accuracy)
print("Recall：", recall)
```

评价报告应同时给出类别基准率、绝对数量和多个指标，不能只给一个百分比。

## 4. ROC 与 PR 扫描全部阈值

ROC 曲线绘制 TPR 对 FPR；PR 曲线绘制 Precision 对 Recall。二者都评价概率排序，不固定单一阈值。

```python
def threshold_curve(actual, probability):
    rows = []
    for threshold in np.r_[1.01, np.sort(np.unique(probability))[::-1], -0.01]:
        counts = confusion_counts(actual, probability, threshold)
        metrics = classification_metrics(counts)
        fpr = counts["fp"] / (counts["fp"] + counts["tn"])
        rows.append({"threshold": threshold, "tpr": metrics["recall"], "fpr": fpr, "precision": metrics["precision"]})
    return pd.DataFrame(rows)

curve = threshold_curve(actual, probability)
print(curve.round(3))
```

ROC-AUC 可解释为随机取一个正类和一个负类，模型给正类更高分的概率。极不平衡场景中，PR 曲线通常更直观，因为 Precision 会直接受到误报数量影响。

## 5. 排序好不代表概率校准好

校准问的是：所有预测约 0.7 的事件，长期是否约 70% 发生。一个模型可能 AUC 很高，却把 0.7 风险系统报成 0.95。

```python
calibration = pd.DataFrame({"probability": probability, "actual": actual})
calibration["bin"] = pd.cut(calibration["probability"], bins=[0, .3, .6, 1], include_lowest=True)
calibration_report = calibration.groupby("bin", observed=True).agg(
    count=("actual", "size"),
    mean_probability=("probability", "mean"),
    observed_rate=("actual", "mean"),
)
print(calibration_report)
```

若概率用于资源预算、风险沟通或期望代价计算，校准尤其重要。

## 6. 用错误代价和容量选择阈值

假设漏报成本记为 8，误报成本记为 2，总成本为 `8×FN + 2×FP`。

```python
candidate_rows = []
for threshold in np.linspace(0.1, 0.9, 17):
    counts = confusion_counts(actual, probability, threshold)
    cost = 8 * counts["fn"] + 2 * counts["fp"]
    candidate_rows.append({"threshold": threshold, "cost": cost, **counts})

cost_table = pd.DataFrame(candidate_rows)
print(cost_table.sort_values(["cost", "threshold"]).head())
```

真实成本往往不易量化，且会随群体、时段和系统容量变化。可以同时建立约束，例如“Recall 至少 0.8 且每日最多处理 20 个预警”。

<div class="ds-warning"><strong>阈值也是公平性决策</strong>各群体基准率、测量误差和概率分布不同，同一阈值可能产生不同 FPR/FNR。分群指标能暴露差异，但采用群体特定阈值涉及政策与法律判断。</div>

## 7. 知识检查

<QuizBlock question="降低分类阈值通常会怎样？" :options="['预测正类增多，Recall 上升且 FP 可能增加', '预测正类减少', '所有指标同时变好', '概率本身改变']" :answer="0" explanation="阈值只改变概率到标签的映射，更多样本跨入正类。" />

<QuizBlock question="Precision 的分母是什么？" :options="['所有真实正类', '所有预测正类', '所有真实负类', '全部样本']" :answer="1" explanation="Precision=TP/(TP+FP)，问预警中多少是真的。" />

<QuizBlock question="极不平衡时为什么要警惕 Accuracy？" :options="['它无法计算', '多数类基线可能已经很高，掩盖少数类完全漏检', '它总等于 0', '它只看概率']" :answer="1" explanation="全部预测多数类也可能得到很高 Accuracy。" />

<QuizBlock question="AUC 高是否保证 0.8 概率真的约 80% 发生？" :options="['保证', '不保证；AUC 评价排序，校准是另一性质', '只要样本大就保证', 'AUC 就是概率均值']" :answer="1" explanation="排序可很好但概率尺度仍系统过高或过低。" />

## 8. 代码练习：按容量选择阈值

每天最多处理 4 个预警。写函数在所有候选概率处选择能使 Recall 最高、且预测正类不超过容量的阈值；并报告混淆矩阵与指标。

<details><summary>提示</summary>

候选阈值可用唯一概率值加上 1.01。对每个阈值计算 `tp+fp`，先过滤容量，再按 Recall 降序、阈值降序选择。

</details>

<details><summary>参考实现</summary>

```python
def choose_threshold_by_capacity(actual, probability, capacity):
    candidates = np.r_[1.01, np.sort(np.unique(probability))[::-1]]
    feasible = []
    for threshold in candidates:
        counts = confusion_counts(actual, probability, threshold)
        predicted_positive = counts["tp"] + counts["fp"]
        if predicted_positive <= capacity:
            metrics = classification_metrics(counts)
            feasible.append((metrics["recall"], threshold, counts, metrics))
    if not feasible:
        raise ValueError("没有满足容量约束的阈值")
    _, threshold, counts, metrics = max(feasible, key=lambda row: (row[0], row[1]))
    return threshold, counts, metrics

threshold, counts, metrics = choose_threshold_by_capacity(actual, probability, capacity=4)
print(threshold, counts, metrics)
```

</details>

## 9. 本讲小结

- 混淆矩阵记录 TP、FP、TN、FN，指标从不同分母观察错误。
- 阈值降低通常提高 Recall 并增加误报；0.5 并非通用最优。
- 类不平衡时要报告基准率、绝对数量、PR 指标和多个阈值。
- 排序、校准和决策成本是三个不同层次，必须分别评价。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 23：阈值、混淆矩阵、Precision、Recall、ROC/PR 和模型评价的知识边界。
- *Learning Data Science*，Ch.19：分类评价主题。
- 本页站点服务故障案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="23" next-label="第 24 讲：PCA 投影" next-href="/courses/data-science/24-pca-projection" />
