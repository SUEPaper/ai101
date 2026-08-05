---
course: machine-learning
title: 机器学习教材来源与许可登记
description: 记录机器学习教材的知识参考、原创边界、第三方许可和案例素材使用规则。
outline: [2, 3]
---

# 来源与许可登记

本教材的中文正文、原创案例、合成数据、公式组织、代码、练习和 Vue 交互按本站课程目标独立创作。公共算法、数学事实和论文结论会参考下列资料核对，但公开可见不等于允许复制原文、原图或原题。

## 采用原则

1. 公共知识可以用本站自己的结构、语言、数据和图形重新表达。
2. 有明确许可的材料只在许可范围内复用，并保留作者、来源、许可链接与修改说明。
3. 未声明再分发许可的讲义、题目、答案、图和独创案例只用于内部学习，不进入站点。
4. 每个章节末记录主要知识参考；若实际使用第三方原样素材，再逐项登记精确页码、使用方式与归属文案。

## 主要知识参考

| 资料 | 用途 | 公开使用边界 |
|---|---|---|
| CMU 10-301/10-601 | 学习问题、优化、树、近邻、感知机、PAC 与作业粒度 | 只在对应版本明确许可范围内复制；其余独立重写 |
| Berkeley CS 189/289A | 概率、回归、优化、生成式模型、神经网络与信息论 | 未找到统一再分发许可的课件和练习不复制 |
| Stanford CS229 Notes | 回归、逻辑回归、SVM、核、EM、PCA 与学习理论 | 公式和事实独立表达；原文、图和题需许可 |
| Berkeley Deep RL Spring 2026 | MDP、价值学习、策略梯度、Actor–Critic 与延伸路线 | starter code 按 MIT License；PDF 内容不直接复制 |
| *Probabilistic Machine Learning: An Introduction* | 概率建模、决策论、信息论和模型假设查阅 | CC BY-NC-ND 4.0；只在符合 NonCommercial 时原样分享，改编需另行授权 |
| *Concise Machine Learning* | 经典模型的理论边界 | 按文档自身许可逐项判断 |
| 林轩田“基石/技法” | 泛化、SVM、核、Bagging 与 Boosting 的理解参考 | 不复制翻译讲义、原题或独创叙事 |
| Coursera Machine Learning Specialization | 分步实验和即时反馈粒度 | 不复制课程测验、截图、实验文本和答案 |

## 已确认可复用的特殊资料

### CS231n Backpropagation 页面

页面随仓库采用 MIT License。若后续直接复用或改编正文、图或代码，保留 `Copyright (c) 2015 Andrej Karpathy` 与完整 MIT License。本教材当前反向传播正文与交互为独立创作。

### *The Illustrated Transformer*

采用 CC BY-NC-SA 4.0。若非商业复用或改编，必须署名 Jay Alammar、链接原文与许可、说明修改，并把改编材料按相同许可发布。本教材当前注意力图与实验为独立创作，不复制原图。

### Berkeley Deep RL starter code

starter code 使用 MIT License，可在保留通知后复用。课程 PDF、作业说明、图表和评分标准未声明统一再分发许可，不因本地保存而获得公开发布权。

## PML 的 CC BY-NC-ND 边界

本地 `ref/ml101/mlapp_book1.pdf` 为 2025-04-18 版本，共 860 页，版权页声明 CC BY-NC-ND。符合 NonCommercial 时可分享未改动的原样材料并完整署名；公开翻译、改写、重绘、替换图中标签或重组片段属于改编，不由 ND 条款授权。

若后续使用原样节选，统一归属格式为：

> 来源：Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*, §X.Y / Figure X.Y, © 2022 Massachusetts Institute of Technology, CC BY-NC-ND 4.0，原始材料未作修改。

同时链接原书页面与许可文本，并逐项检查图注中的第三方权利声明。

## 课程原创素材登记

- 页面首屏、证明路径、公式工作台与课程配色：本站原创。
- `MlLab` 的 32 个参数化实验、数据生成和图形：本站原创。
- 各章 NumPy 最小实现、检查题、练习和答案：本站原创。
- 数据科学课程内部链接与自有案例可复用；其中第三方素材仍继承原始限制。

详细参考资料本地清单见 `ref/ml101/README.md`。`ref/` 不进入 VitePress 发布产物。
