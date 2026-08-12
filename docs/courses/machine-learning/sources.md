---
course: machine-learning
title: 机器学习教材来源与许可登记
description: 逐章记录机器学习教材采用的知识来源、精确页码、使用方式和许可边界。
outline: [2, 3]
---

# 机器学习教材来源与许可登记

本页是网页版教材的来源审计记录。参考某份资料的知识结构，不等于复制它的文字、图片、案例或题目；每章细化完成时，必须登记精确 PDF 页码或网页小节、实际采用内容、许可判断和本站处理方式。

## 使用规则

- `ref/ml101/opendataloader-pdf/` 只用于检索。公式、页码和图表在采用前必须回看源 PDF，因为本地解析可能打乱复杂公式的字符顺序。
- 未找到明确再分发许可的讲义、网页、课程题目和解答，只用于核对公共知识与章节结构，不复制原文、原图、独创案例或题目。
- *Probabilistic Machine Learning: An Introduction* 按 `CC BY-NC-ND 4.0` 处理。只有实际用途符合非商业条件时，才能原样分享未修改材料并完整署名；翻译、改写或重绘不属于“原样分享”。本教材当前只把它作为知识参考，正文、案例和交互均独立创作。
- 本站独立创作的案例必须明确数据是实测、公开数据还是合成数据；合成数据不能被描述成真实生产统计。
- 每个章节的“来源与许可”小节与本页登记必须一致。

## 已完成章节

### 第 01 章：监督学习与经验风险最小化

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 224–225 页，§5.4.1 | 总体风险、经验风险、ERM、近似误差与估计误差 | 仅作知识结构核对；正文公式重新推导，案例、代码和图形由本站独立创作，不复用原文图表 |
| Stanford CS229 Lecture Notes | PDF 第 128–130 页，§8.3.1–8.3.2 | 训练误差、总体误差、假设类和有限假设类上的 ERM | 仅作推导边界参考；不复制讲义表达、图表或题目 |
| 林轩田《机器学习基石》课程笔记 1：*The Learning Problem* | PDF 第 3–4 页 | 数据、假设集合、学习算法与最终假设的关系 | 仅作中文术语和知识顺序参考；不复用原图、原文或案例 |

第 01 章“循环水泵故障预警”案例、20 台设备的演示样本、12,000 台设备的模拟总体、两项网页交互、全部题目和代码均为本站独立创作。所有数值都是教学合成数据，不代表任何工厂的真实故障率、损失或检修策略。

### 第 02 章：梯度下降、牛顿法与随机优化

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 314–320 页，§8.2–8.3 | 下降方向、固定步长、条件数、Momentum、Newton 局部二次模型 | 仅作知识边界核对；正文重新推导，轧机案例与优化路径交互由本站独立创作，不复用原文图表 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 322–331 页，§8.4 | 有限和 SGD、mini-batch、学习率日程、预条件方法、Adam 偏差修正 | 仅作公式与适用边界核对；代码、题目、随机梯度实验和工程诊断均独立创作 |
| Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 110–114 页 | mini-batch 的方差与硬件权衡、输入标准化、Momentum | 仅作工程直觉交叉核对；不复制讲义图片、措辞或课堂案例 |

第 02 章“轧机厚度偏差校正”案例、240 条训练样本、二维优化目标、两项网页交互、全部题目和代码均为本站独立创作。所有数据都是教学合成数据，不代表真实产线统计、工艺窗口或控制策略。

### 第 03 章：泛化、偏差—方差与模型选择

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 155–159 页，§4.5.4–4.5.7 | 验证集、K 折、one-standard-error 规则、早停、学习曲线 | 仅作知识边界核对；跨产线案例、嵌套选择实现和评价协议由本站独立创作 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 190–194 页，§4.7.6 | 估计量偏差、方差、MSE 分解、分类损失下的适用边界 | 仅作推导边界核对；正文重新推导，不复用原文图表或案例 |
| Stanford CS229 Lecture Notes | PDF 第 114–124 页，Ch.8 | 训练/测试误差、回归偏差—方差推导、双下降 | 用于交叉核对假设与现代边界；不复制讲义表达或图形 |
| Stanford CS229 Lecture Notes | PDF 第 140–143 页，§9.3 | holdout 与 K 折模型选择 | 仅作基本流程核对；验证集复用交互、题目和代码均为本站独立创作 |

第 03 章 A/B 产线、340 条合成样本、两项网页交互、全部题目、代码和 SVG 图形均为本站独立创作。所有数据都是教学合成数据，不代表真实工厂统计、产线差异或质量标准。

### 第 04 章：普通最小二乘法（OLS）

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Stanford CS229 Lecture Notes | PDF 第 15–17 页，§1.2 | 设计矩阵、平方损失矩阵展开、正规方程、可逆条件 | 仅作基本推导核对；正文以空压站案例独立重写，不复用讲义图表、措辞或住房案例 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 404–406 页，§11.2.2.1–11.2.2.3 | OLS、正交投影、帽子矩阵、SVD/QR 求解和标准化 | 仅作知识与算法边界核对；交互、代码、题目和图形均由本站独立创作 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 298–299 页，§7.7.2–7.7.3 | 欠定系统的一般解、最小范数解、正规方程与满列秩唯一性 | 仅作证明结构核对；不复用原文案例、图形或表达 |
| Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 61–63 页 | 截距列、正规方程、伪逆、帽子矩阵与数值求解提醒 | 用于交叉核对工程边界；不复制讲义图形、课堂案例或措辞 |
| Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 140–142 页 | 紧致 SVD 伪逆、最小范数解、小奇异值放大 | 仅作秩亏与病态问题核对；交互实验与推导均独立创作 |

第 04 章空压站 240 小时数据、双流量计情境、两项网页交互、全部题目、代码和 SVG 图形均为本站独立创作。所有数值都是教学合成数据，不代表真实工厂能耗、传感器精度或节能标准。

### 第 05 章：线性回归：高斯似然与统计推断

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Stanford CS229 Lecture Notes | PDF 第 17–19 页，§1.3 | 独立高斯误差、条件似然、对数似然、OLS 与最大似然等价 | 仅作基本概率推导核对；正文使用压力变送器案例独立重写，不复用讲义住房案例、图形或措辞 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 401–406 页，§11.1–11.2.2.4 | 条件线性高斯模型、NLL、OLS、异方差模型、加权最小二乘 | 仅作知识与算法边界核对；统计推断链、代码、实验和题目均由本站独立创作 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 89–90 页，§2.6.3–2.6.4 | 同方差/异方差回归、均值不确定性与观测预测不确定性 | 仅作术语和区间对象核对；不复用原图或案例 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 432–435 页，§11.6 | Gaussian、Laplace、Student-t 与 Huber 的稳健回归边界 | 仅作损失与观测模型映射核对；本站表格、案例和题目独立创作 |

第 05 章 HC1、t 区间、Gauss–Markov 和夹心协方差均按标准统计定义独立推导并通过合成重复采样验证。压力变送器 168 条读数、校准循环、两项网页交互、全部代码、题目和 SVG 图形均为本站独立创作，不代表真实计量规范、仪器性能或验收限值。

### 第 06 章：Ridge 回归与 Lasso

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 412–415 页，§11.3 | Gaussian 先验与 Ridge MAP、闭式解、增广最小二乘、谱收缩、有效自由度 | 仅作推导边界核对；电机案例、代码、谱滤波实验和题目均由本站独立创作 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 415–422 页，§11.4 | Laplace 先验、L1 约束几何、次梯度、软阈值、正则化路径 | 仅作知识与算法边界核对；不复用原文案例、图表或措辞 |
| Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 77–81 页（讲义页码 71–75），Chapter 13 | Ridge 唯一性与偏差—方差、标准化、L1/L2 几何和 Lasso 路径 | 用于交叉核对；网页几何实验、SVG 图形和工业审计流程均独立创作 |
| Stanford CS229 Lecture Notes | PDF 第 143–144 页，§9.4 | 参数先验、后验、完整 Bayesian 预测与 MAP 点估计边界 | 仅作 Bayesian 解释边界核对；不复用讲义示例或表达 |

第 06 章电机绕组热点软测量的 336 个稳定窗口、24 个测试循环、分组切分、组 bootstrap、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实设备性能、安全阈值、保护策略或硬件移除决策。

### 第 07 章：基函数回归与局部加权回归

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 401–403 页，§11.1、§11.2.1 | 固定特征变换、对参数线性、多项式回归 | 仅作知识结构核对；冷站案例、推导、代码、题目和基函数混合实验均由本站独立创作 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 429–432 页，§11.5 | 回归 spline、局部支撑、节点、B-spline 设计矩阵、Ridge 与交叉验证 | 仅作样条知识边界核对；不复用原文图表、案例或表达 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 594–596 页，§16.3.5 | 核回归、Nadaraya–Watson、局部加权线性回归、LOWESS/LOESS | 仅作公式与术语核对；局部检测光束实验、有效样本量诊断和实现均独立创作 |
| Stanford CS229 Lecture Notes | PDF 第 19–20 页，§1.4 | 多项式欠拟合/过拟合、局部加权目标、Gaussian 形权重、带宽与查询成本 | 用于交叉核对推导和非参数方法边界；不复制讲义住房案例、图形或措辞 |
| Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 65–66 页（讲义页码 59–60） | 多项式特征扩展、边界振荡、加权最小二乘 | 仅作工程风险核对；不复用原文图形、课堂案例或表达 |

第 07 章冷站湿球温度—单位冷量功耗案例的 288 个稳定窗口、36 个运行日、分组切分、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实冷站性能、节能阈值、运行策略或设备控制边界。

### 第 08 章：贝叶斯决策与最优分类器

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 197–201 页，§5.1.1–5.1.2 | 后验期望损失、Bayes 决策、0-1 损失、代价敏感分类、reject option | 仅作推导与术语核对；动力电池端检案例、损失矩阵、代码、题目和风险继电器实验均由本站独立创作 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 201–206 页，§5.1.3–5.1.4 | 混淆矩阵、TPR/FPR、ROC、AUC、类别不平衡与 PR 曲线 | 仅作评价边界核对；不复用原文图表、案例或表达 |
| Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 38–41 页（讲义页码 32–35） | Bayes classifier、非对称损失、Bayes risk、先验加权类条件分布、生成式与判别式路线 | 用于交叉核对；不复制讲义癌症案例、图形或课堂措辞 |
| Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 70–71 页（讲义页码 64–65） | 验证/测试集 ROC、TPR/FPR、阈值轨道、AUC 和实际运行点选择 | 仅作评价流程核对；网页 ROC 验收轨道、部署账本和交互均独立创作 |

第 08 章动力电池模组端检缺陷分流案例的 2,000 只模组合成记录、40 个生产批次、三动作损失矩阵、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实电池缺陷率、端检性能、安全阈值、质量成本或放行规范。

### 第 09 章：逻辑回归

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 369–380 页，§10.1–10.2.8 | 二分类逻辑回归、线性/非线性边界、Bernoulli NLL、梯度、Hessian、SGD、Newton/IRLS、MAP 与标准化 | 仅作公式、术语和算法边界核对；焊缝案例、代码、题目和两项网页实验均由本站独立创作 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 208–209 页，§5.1.6.1–5.1.6.3 | KL、交叉熵、log loss、proper scoring rule 与 Brier score | 仅作概率评分语义核对；不复用原文图表、案例或表达 |
| Stanford CS229 Lecture Notes | PDF 第 21–24 页，§2.1；PDF 第 29–30 页，§2.4 | Sigmoid、Bernoulli likelihood、梯度、Newton–Raphson 与 Hessian 更新 | 用于交叉核对推导顺序；不复制讲义垃圾邮件示例、图形或措辞 |
| Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 63–69 页（讲义页码 57–63） | logistic loss、梯度、完全分离、Newton、凸性、IRLS 与 LDA/逻辑回归比较 | 用于核对存在性与优化边界；不复用课堂图形、示例或表达 |

第 09 章焊缝在线缺陷分流案例的 2,400 条合成记录、48 个生产批次、传感器定义、成本值、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实焊接缺陷率、无损检测能力、放行阈值、工艺规范或安全决策。

### 第 10 章：Softmax 回归

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 84–87 页，§2.5.1–2.5.4 | Categorical/Multinomial 分布、Softmax、温度、二分类退化与 log-sum-exp | 仅作概率模型、归一化与数值稳定边界核对；正文推导、代码与两项网页实验均由本站独立创作 |
| Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction* | PDF 第 381–385 页，§10.3–10.3.3 | 多项逻辑回归、互斥分类与多标签区别、交叉熵、梯度、Hessian、凸性与优化 | 仅作公式、术语和算法边界核对；不复用原文图表、案例或表达 |
| Stanford CS229 Lecture Notes | PDF 第 26–28 页，§2.3 | 多分类 logits、Softmax、条件似然、交叉熵与参数梯度 | 用于交叉核对推导顺序和矩阵形状；不复制讲义案例、图形、题目或措辞 |
| Jonathan Richard Shewchuk, *Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 104–105 页（讲义页码 98–99），Lecture 17 | Softmax 输出、one-hot/软目标、交叉熵与输出层梯度 | 用于核对神经网络输出层接口；不复用课堂图形、示例或表达 |

第 10 章电机台架主导故障分流案例的 2,400 台合成记录、60 个生产批次、四类互斥处置标签、损失矩阵、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实电机故障率、台架诊断能力、质量成本、放行规范或维修策略。

### 第 11 章：感知机与支持向量机

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Stanford CS229 Lecture Notes | PDF 第 25 页，§2.2 | 阈值预测、感知机逐样本更新与非概率边界 | 仅作更新规则和术语核对；滚子案例、证明、代码和逐件播放实验均由本站独立创作 |
| Stanford CS229 Lecture Notes | PDF 第 60–74 页，§6.1–6.7 | 间隔直觉、函数/几何间隔、最优间隔分类器、支持向量、不可分情形与 `C` | 用于推导顺序和公式边界交叉核对；不复用原文图形、案例、题目或措辞 |
| Jonathan Richard Shewchuk，*Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 18–24 页，Lectures 2–4 | 感知机更新、犯错次数界、最大间隔、硬间隔和软间隔 | 仅作收敛证明与几何解释核对；不复制 Mark I 图片、课堂图形、示例或表达 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 615–622 页，§17.3–17.3.6 | 大间隔、点到超平面距离、支持向量、松弛变量、`C`、hinge loss 与概率校准边界 | 仅作知识结构、公式和工程边界核对；不复用原书图表、数据或表达 |

第 11 章轴承滚子终检复检分流案例的 2,400 只合成记录、60 个热处理批次、涡流/轮廓特征、争议标签、成本定义、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实轴承缺陷率、探头性能、复检能力、质量成本、放行标准或安全规范。

### 第 12 章：核方法与核 SVM

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Stanford CS229 Lecture Notes | PDF 第 54–59 页，§5.4 | 显式特征映射、多项式核、Gaussian 核、核矩阵、半正定性与 Mercer 条件 | 仅作核函数合法性、表示方式和公式边界核对；正文证明、滚子案例、代码和核特征实验均由本站独立创作 |
| Stanford CS229 Lecture Notes | PDF 第 66–74 页，§6.5–6.7 | 对偶性、KKT 条件、SVM 对偶、支持向量、核化与不可分情形 | 用于交叉核对推导链与盒约束；不复用原文图形、案例、题目或措辞 |
| Jonathan Richard Shewchuk，*Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 169–173 页，Bonus Lecture B | 高阶特征、核化、多项式核、Gaussian 核与计算代价 | 仅作计算直觉和知识边界核对；网页升维实验、SVG 图形和工程审计流程均独立创作 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 597–603 页，§17.1–17.1.2.3 | Mercer 核、Gram 矩阵、RBF/ARD/多项式核及核的组合 | 仅作术语、核族和半正定边界核对；不复用原书图表、数据或表达 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 617–621 页，§17.3.2–17.3.5 | SVM 对偶、KKT、软间隔、核技巧、预测与概率校准边界 | 仅作优化与部署边界核对；对偶平衡实验、实现、题目和运行协议均由本站独立创作 |

第 12 章滚子终检环形异常分流案例的 360 只合成记录、30 个生产批次、中心放行区与外圈复检区、争议标签、两项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实轴承缺陷率、探头性能、核参数、复检能力、放行标准或安全规范。

### 第 13 章：多层感知机（MLP）

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Stanford CS229 Lecture Notes | PDF 第 85–95 页，§7.2–7.3 | 神经元、全连接层、批量向量化、激活函数、线性激活折叠、学习型特征映射与 MLP 组件 | 仅作公式、术语和知识边界核对；正文推导、焊接案例、代码和表示折叠实验均由本站独立创作 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 454–465 页，§13.1–13.2.5 | 表格数据 MLP、XOR、可微 MLP、激活函数、输出概率模型、万能逼近与深度边界 | 仅作知识结构和概率接口核对；不复用原书图表、数据、案例或表达 |
| Jonathan Richard Shewchuk，*Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 95–104 页（Lectures 16–17）；PDF 第 113 页（Lecture 18） | 单隐层网络、隐藏单元置换对称、ReLU、输出单元与损失、Xavier/He 初始化 | 用于交叉核对表示、对称性和初始化边界；不复制课堂图形、示例、题目或措辞 |
| CMU 10-701 Lecture 11，*Neural Networks* | PDF 第 45–73 页 | 神经网络组件、一隐层结构、非线性决策边界、宽度与架构直觉 | 仅作讲解顺序与表示直觉核对；三项网页交互、SVG 图形和工程审计流程均独立创作 |

第 13 章激光焊接双传感器一致性分流案例的 1,152 条合成记录、48 个生产批次、四象限教学规则、三项网页交互、全部代码、题目和 SVG 图形均为教学合成与本站独立创作，不代表真实焊接缺陷机理、传感器性能、复核阈值、质量标准、放行规范或安全策略。

### 第 14 章：反向传播

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 301–303 页，§7.8.5；PDF 第 468–475 页，§13.3–13.3.4 | Jacobian、JVP、VJP、复合 Jacobian、前向/反向模式、MLP VJP、常见层和任意 DAG 计算图 | 仅作术语、公式和算法边界核对；正文推导、最小自动微分引擎、代码和三项网页实验均由本站独立创作 |
| Stanford CS229 Lecture Notes | PDF 第 99–111 页，§7.4 | 反向传播线性复杂度、局部 backward function、矩阵乘法/激活/损失模块、MLP cache 与批量矩阵实现 | 用于交叉核对复杂度、模块接口和推导链；不复用讲义图形、案例、题目或措辞 |
| CMU 10-301/10-601 Lecture 12，*Neural Networks + Backpropagation*；Lecture 13，*Backpropagation II* | Lecture 12 PDF 第 39–53 页；Lecture 13 PDF 第 14–54 页 | 有限差分、链式法则、标量反向、向量反向、多元链式法则和任意层 backward | 仅作讲解顺序与公式边界核对；不复制课堂图形、测验或表达 |
| Jonathan Richard Shewchuk，*Machine Learning*（Berkeley CS 189/289A 课程讲义） | PDF 第 98–107 页 | 从朴素重复求导到按边线性复杂度、反向传播算法、输出层与 ReLU 网络反向 | 仅作复杂度和网络实例交叉核对；不复用课堂图形、示例或措辞 |
| Stanford CS231n，*Backpropagation, Intuitions* | 本地快照的 Staged computation、Patterns in backward flow、Gradients for vectorized operations 小节；随附 MIT License | 前向缓存、分叉处梯度累加、局部门解释、分阶段计算与向量化反向 | 在 MIT License 边界内仅作知识与实现习惯参考；本章未复用原页面 SVG，网页实验和视觉语言均独立设计 |

第 14 章沿用的 1,152 条激光焊接双传感器合成记录、48 个生产批次、梯度审计账本、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实焊接缺陷机理、传感器性能、梯度阈值、质量标准、放行规范或安全策略。PyTorch 对照明确标为可选未执行内容，仓库标准验证环境未为此新增大型依赖。

### 第 15 章：卷积神经网络（CNN）

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 498–506 页，§14.1–14.2.3 | 图像网络、二维互相关、卷积的稀疏与权重绑定矩阵、padding、stride、多输入/输出通道、1×1 卷积、池化、平移等变与全局平均池化 | 仅作术语、公式、几何和知识边界核对；钢带案例、代码、题目与三项网页实验均由本站独立创作 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 516–519 页，§14.4.1–14.4.3 | 膨胀卷积、转置卷积与深度可分离卷积 | 仅作变体定义与效率边界核对；不复用原书图表、案例或表达 |
| CMU 10-301/10-601 Lecture 17，*MAP, CNN, RNN* | PDF 第 31–89 页 | 从全连接视觉模型的参数问题到局部共享、padding、stride、下采样、训练、多通道卷积和池化 | 用于交叉核对讲解顺序与张量接口；不复制课堂图形、测验或措辞 |
| Jonathan Richard Shewchuk，*Machine Learning*（Berkeley CS189/289A 课程讲义） | PDF 第 115–119 页，Lecture 19 | 局部连接、共享权重、通道参数量、池化、步幅卷积及反向传播提示 | 用于核对参数计数与采样边界；不复用课堂图形、示例或表达 |

第 15 章冷轧钢带线扫表面缺陷分流案例的 864 个合成小块、36 卷分组、三类教学标签、固定核基线、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实钢带缺陷机理、相机性能、质量标准、复核阈值、放行规范或停线策略。PyTorch 对照明确标为可选未执行内容，仓库标准验证环境未为此新增大型依赖。

### 第 16 章：循环神经网络（RNN）

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 533–545 页，§15.1–15.2.7；其中 BPTT 与截断训练见 PDF 第 540–542 页，门控单元见 PDF 第 542–545 页 | 序列映射、RNN 状态递推、条件生成、teacher forcing、双向 RNN、BPTT、截断 BPTT、梯度消失/爆炸、梯度裁剪、GRU 与 LSTM | 仅作术语、公式、算法和知识边界核对；退火炉案例、代码、题目与三项网页实验均由本站独立创作 |
| CMU 10-301/10-601 Lecture 17，*MAP, CNN, RNN* | PDF 第 107–127 页 | 序列标注、时间序列、共享递推、时间展开、BPTT、双向/多层 RNN 与 LSTM 结果 | 用于交叉核对讲解顺序、计算图与任务边界；不复制课堂图形、测验、案例或措辞 |
| CMU 10-301/10-601 Lecture 18，*RNN, Transformer* | PDF 第 2–15 页 | RNN 回顾、共享参数、BPTT、双向和深层 RNN | 用于核对章节衔接与展开图边界；不复用课堂图形或表达 |
| Christopher Olah，*Understanding LSTM Networks*；Simeon Kostadinov、Raimi Karim / Vihar Kurama 的 GRU 文章本地快照 | 本地快照的 LSTM 门控、cell state、GRU update/reset gate 与两套更新门记号相关小节；未发现明确再分发许可 | 门控直觉、信息路径、GRU 更新门记号差异与边界行为 | 仅作内部交叉参考；不复制原文、图像、代码或案例，公式以 Murphy 教材和本站独立推导为准 |

第 16 章连续退火炉热历史判读案例的 48 卷合成序列、12 个生产 campaign、三通道观测、标签生成规则、递推诊断基线、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实炉温机理、材料性能、传感器精度、质量标准、复核阈值、放行规范或闭环控制策略。PyTorch 对照明确标为可选未执行内容，仓库标准验证环境未为此新增大型依赖。

### 第 17 章：Transformer 与自注意力

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 549–563 页，§15.4–15.5；缩放点积与 Q/K/V 见 PDF 第 551–552 页，自注意力、多头与位置表示见 PDF 第 556–560 页，编码器、解码器和复杂度见 PDF 第 560–562 页 | 内容寻址、缩放点积、自注意力、多头、位置编码、Transformer 结构与复杂度 | 仅作术语、公式、算法和知识边界核对；压缩机站案例、代码、题目与三项网页实验均由本站独立创作 |
| CMU 10-301/10-601 Lecture 18，*RNN, Transformer*；Lecture 19，*Autodiff, In-context Learning* | Lecture 18 PDF 第 62–77 页；Lecture 19 PDF 第 3–33 页 | 从 RNN 到注意力、Q/K/V、多头、语言模型、残差、LayerNorm、因果 Mask、矩阵化与批处理 | 用于交叉核对推导顺序、矩阵形状和任务边界；不复制课堂图形、测验、案例或措辞 |
| UC Berkeley CS 189/289A Lecture 21；Lecture 23 | Lecture 21 PDF 第 60–118 页；Lecture 23 PDF 第 17–34 页 | 内容注意力、自注意力、Q/K/V、缩放、多头、层结构、位置编码和 masked attention | 用于交叉核对概念覆盖、层结构与自回归遮罩；不复用课堂图形或表达 |
| Jay Alammar，*The Illustrated Transformer* 本地网页快照 | Q/K/V、矩阵化、多头、位置、残差和因果遮罩相关小节；页面声明 CC BY-NC-SA 4.0 | 视觉信息顺序与概念连接 | 仅作内部交叉参考；本章没有复制原文、图像或代码，所有 SVG 交互均为本站独立创作 |

第 17 章压缩机站保护事件根因分诊案例的 72 条合成事件链、18 个运行 campaign、四类教学标签、固定证据池化基线、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实天然气站、压缩机、电机、PLC、保护继电器、报警优先级、维修根因或安全联锁规则。该模型不得用于旁路保护、自动复位、修改定值或直接闭环控制；PyTorch 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 18 章：k 近邻算法（kNN）

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 577–581 页，§16.1–16.1.4；距离学习边界见 PDF 第 582–583 页，§16.2–16.2.1 | 局部类别分布、Mahalanobis 距离、Voronoi、平滑效果、维数灾难、查询/存储成本、近似近邻与开放集 | 仅作术语、公式、算法与知识边界核对；不复用教材中标注为改编或第三方许可的图形 |
| CMU 10-301/10-601 Lecture 5，*k-Nearest Neighbors + Model Selection* | PDF 第 5–20、67–72 页 | 最近邻算法、L1/L2、查询复杂度、平票、尺度归纳偏置、Cover–Hart 结论及模型/超参数区分 | 用于交叉核对讲解顺序、距离语义和模型选择边界；不复制课堂图形、测验、案例或措辞 |
| CMU 10-301/10-601 HW3 Recitation Solutions | PDF 第 5–9 页 | 分类平票、距离加权/等权回归与三折验证 | 仅作内部知识粒度与边界核对；不复制题图、解答或表达 |
| Jonathan Richard Shewchuk，CS 189/289A Lecture 24–25 | PDF 第 156–161 页 | 分类/回归、Bayes 风险关系、一致性条件、穷举查询、Voronoi、k-d tree、精确与近似搜索 | 用于核对定理条件、复杂度和索引边界；不复用课堂图形、案例或措辞 |

第 18 章离心泵历史检维修案例分诊的 30 台泵、150 条重复窗口、三类教学根因、停机时长生成规则、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实泵型、传感器量程、振动标准、保护阈值、检维修结论或停机计划。该模型不得直接用于停泵、修改保护、备件采购或安全决策；scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 19 章：决策树

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Jonathan Richard Shewchuk，CS 189/289A Lecture 14–15 | PDF 第 82–86 页；回归、早停、概率与剪枝见 PDF 第 87–90 页 | 递归树生长、熵/信息增益、连续中点扫描、累计计数、复杂度、回归叶均值、早停、叶概率、验证剪枝与高方差 | 用于核对公式、算法、复杂度与知识边界；不复用课堂图形、案例或措辞 |
| CMU 10-301/10-601 Lecture 3，*Decision Trees* | PDF 第 18–33 页 | 切分准则、递归训练、分类错误率的分辨率不足、熵与互信息 | 用于交叉核对推导顺序和算法粒度；不复制课堂图形、测验、数据或表达 |
| CMU 10-301/10-601 Lecture 4，*Decision Tree Properties* | PDF 第 3–46 页 | 轴对齐边界、贪心非全局最优、停止条件、模型选择、树的优缺点与高方差边界 | 用于核对模型选择和失效条件；不复用课堂案例、图形或措辞 |
| CMU 10-301/10-601 HW3 Recitation Solutions | PDF 第 1–4 页 | 连续二维分区、树深过拟合和验证集剪枝 | 仅作内部知识粒度与边界核对；不复制题图、数据、答案或表达 |

第 19 章注塑连接器壳体质量分流的 42 个教学批次、210 条重复模次、三类质量标签、关键尺寸偏差生成规则、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实材料、模具、注塑机、传感器、工艺窗口、质量标准或安全阈值。该模型不得直接修改压力、锁模力、模温、安全联锁或自动放行高风险产品；scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 20 章：Bagging 与随机森林

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Jonathan Richard Shewchuk，CS 189/289A Lecture 15–16 | PDF 第 90–94 页 | 单树高方差、集成平均、Bagging、63.2% 唯一入袋比例、节点级特征抽样、`mtry` 起点与强度—相关性权衡 | 用于核对公式、算法和知识边界；不复用课堂图形、案例或措辞 |
| CMU 10-301/10-601 Lecture 24，*Ensemble Methods: Bagging* | PDF 第 29–32、34–48 页 | 样本 Bagging、投票/平均、随机子空间、随机森林、逐节点特征随机化、OOB 误差、选参与不纯度重要性 | 用于交叉核对算法顺序、术语与评价边界；不复制课件表格、图形、数据、测验或表达 |
| CMU 10-301/10-601 HW9 Recitation Solutions | PDF 第 8–9 页 | 逐样本 OOB 聚合、未见样本资格与节点候选特征随机化 | 仅作内部知识粒度与边界核对；不复制题图、答案、数据或表达 |

第 20 章等离子刻蚀腔体批次异常分诊的 60 个教学批次、240 片合成晶圆、三类教学标签、同源端点斜率副本、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实晶圆厂、腔体、配方、传感器、缺陷率、工艺窗口、维修策略或安全阈值。该模型不得直接调整 RF 功率、气体流量、腔压、端点条件、设备联锁或自动放行产品；scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 21 章：AdaBoost 与梯度提升

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 644–646 页，§18.5.3–18.5.4 | margin、指数损失、离散 AdaBoost、弱学习器加权错误、票重、样本权重更新、LogitBoost 与异常点敏感性 | 依据 CC BY-NC-ND 4.0 只作术语、公式、算法和知识边界核对；未改编或复用原书图表、案例、代码和措辞 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 646–650 页，§18.5.5–18.5.5.2 | 函数空间梯度、常见损失伪残差、shrinkage、随机子采样、梯度树叶线搜索，以及二阶正则叶权与切分增益 | 仅作推导与工程边界核对；涂布案例、代码、题目和三项网页实验均由本站独立创作 |
| Jonathan Richard Shewchuk，CS 189/289A Lecture 24 | PDF 第 152–155 页 | AdaBoost 指数损失、权重递推、弱学习器票重、短树动机、训练错误趋零条件、margin 概率联系与离群点脆弱性 | 用于交叉核对公式与失效条件；不复用课堂图形、数据、案例或表达 |
| CMU 10-301/10-601 Lecture 24，*Ensemble Methods: Boosting* | PDF 第 13–20、21–25 页 | 弱/强学习器、AdaBoost 逐轮算法、票重符号、训练错误分析与 margin 分布 | 仅作讲解顺序和算法粒度核对；不复制课件图形、课堂投票、数据、测验或措辞 |

第 21 章锂离子电池极片涂布批次质量复核案例的 40 个教学 campaign、240 卷合成记录、返工标签、面密度偏差、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实电池企业、配方、极片、量测系统、质量标准、复检容量或安全阈值。该模型不得直接修改浆料、泵速、模头、烘箱、张力、联锁或自动放行产品；scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 22 章：最大似然估计与最大后验估计

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 137–144 页，§4.1–4.2.6 | MLE、iid/条件似然、Bernoulli、Categorical、Gaussian、充分统计量、方差 MLE 与无偏估计区别 | 依据 CC BY-NC-ND 4.0 只作术语、公式和知识边界核对；未改编或复用原书图表、案例、代码和措辞 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 151–154 页，§4.5.1–4.5.3 | Bernoulli MAP、零计数、收缩估计、Gaussian 先验与 L2 正则 | 仅作推导与边界核对；机器视觉案例、目标缩放说明和数值验证均由本站独立创作 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 159–166 页，§4.6–4.6.2.10；PDF 第 168–173 页，§4.6.3–4.6.4.1 | Bayes 后验、Beta–Bernoulli、后验均值/方差、Beta–Binomial 后验预测、Dirichlet–Categorical、Gaussian–Gaussian | 仅作共轭更新、后验预测和公式边界核对；不复用原书图形、数据或表达 |
| UC Berkeley CS 189/289A Lecture 5，*MLE and Multivariate Gaussians* | PDF 第 16–30、31–42、43–52 页 | MLE 生成故事、独立性、统计性质、Gaussian MLE、点估计与后验分布、Categorical/Multinomial 约束 MLE | 用于交叉核对推导顺序、Lagrange 约束与优化接口；不复制课件图形、数据、练习或措辞 |
| CMU 10-301/10-601 Lecture 16，*PAC Learning + MLE/MAP*；Lecture 17，*Wrap up MAP + CNNs + RNNs* | Lecture 16 PDF 第 42–61、62–69 页；Lecture 17 PDF 第 3–16 页 | MLE/MAP 定义与闭式步骤、Bayes 目标、Beta 先验、Bernoulli 更新、共轭与正则化联系 | 仅作讲解顺序、术语和算法粒度核对；不复制课堂图形、示例、测验或表达 |

第 22 章多工位机器视觉漏检率资格审核案例的 16 个教学工位、750 件合成审核量、37 次合成漏检、挑战配比、光学状态、资格阈值、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实企业、产品、相机、光源、缺陷率、量测系统、质量标准、法规门槛或安全策略。该模型不得直接修改设备、旁路联锁或自动放行产品；scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 23 章：高斯判别分析与朴素贝叶斯

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 353–358 页，§9.2–§9.2.5 | GDA 类条件模型、QDA/LDA 判别式、共享/类别专属/对角协方差、MLE、收缩与 Mahalanobis 最近中心解释 | 依据 CC BY-NC-ND 4.0 只作术语、公式、算法和知识边界核对；未复用或改编原书图表、数据、案例、代码和措辞 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 362–367 页，§9.3–§9.4 | Gaussian、Bernoulli、Categorical Naive Bayes，参数估计、Bayesian smoothing、与逻辑回归的联系及生成式/判别式交换 | 仅作模型族、估计公式和失效边界核对；主轴案例、日志词表、重复证据审计和三项网页实验均为本站独立创作 |
| Stanford CS229 Lecture Notes | PDF 第 36–41 页，§4.1 | 多元 Gaussian、GDA 参数估计、共享协方差、线性后验及与逻辑回归的对照 | 用于交叉核对推导顺序、参数公式和错设边界；不复制讲义图形、示例或表达 |
| Stanford CS229 Lecture Notes | PDF 第 42–49 页，§4.2 | 条件独立、Bernoulli/Multinomial 事件模型、Laplace smoothing、零计数和文本分类 | 用于核对事件模型、平滑分母和 log-space 实现；不复用讲义案例、练习或措辞 |
| UC Berkeley CS 189/289A Lecture 11 | PDF 第 18–50 页 | Gaussian 类条件几何、共享/独立协方差、白化、LDA/QDA 边界与 Naive Bayes 入口 | 用于核对协方差几何、白化和边界阶数；不复制课件图形、课堂数据、测验或表达 |

第 23 章数控主轴 EOL 异常根因分诊案例的 54 个教学 campaign、324 条合成记录、三类根因、连续量测、报警词表、复检动作、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实企业、主轴、轴承、传感器、振动标准、装配参数、测试规范、维修策略或安全阈值。该模型只用于教学中的复检路由，不得自动放行、自动返修、修改测试程序、改变机床参数或旁路安全联锁；scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 24 章：期望最大化算法（EM）

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 342–345 页，§8.7–§8.7.2.2 | MM 紧下界、EM、Jensen 下界、ELBO、KL 分解、E 步与下界相切 | 依据 CC BY-NC-ND 4.0 只作术语、公式、算法和知识边界核对；未复用或改编原书图表、数据、案例、代码和措辞 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 345–349 页，§8.7.2.3–§8.7.3.4 | 期望充分统计量、GMM 责任度、混合权重/均值/协方差 M 步、协方差坍缩与 MAP 入口 | 仅作推导、数值门禁和失效边界核对；主轴无标签案例、重启账本和三项网页实验均为本站独立创作 |
| Stanford CS229 Lecture Notes | PDF 第 149–152 页，§11.1–§11.2 | GMM 潜变量、软分配、E/M 更新、与 K-means 的关系和 Jensen 不等式 | 用于交叉核对推导顺序、软计数公式和算法边界；不复制讲义图形、示例、练习或表达 |
| Stanford CS229 Lecture Notes | PDF 第 153–160 页，§11.3–§11.4 | 一般 EM、ELBO、相切条件、似然单调性、KL 解释与 GMM 回代 | 用于核对不等式链、收敛声明和一般化边界；不复用讲义图表、案例或措辞 |
| UC Berkeley CS 189/289A Lecture 6，*Multivariate Gaussians & Mixture of Gaussians* | PDF 第 45–56 页 | GMM 生成过程、边缘似然、混合权重、密度估计与 K-means 零噪声联系 | 用于核对模型入口和软/硬分配联系；不复制课件图形、课堂数据、测验或表达 |

第 24 章主轴 EOL 无标签量测案例的 60 个教学 campaign、360 条合成记录、三种隐藏生成机制、两项连续量测、分组留出协议、人工复核动作、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实企业、主轴、轴承、传感器、振动标准、装配参数、试车程序、根因比例或安全阈值。统计分量不得直接命名为已确认物理根因；模型只用于教学中的复核路由，不得自动放行、自动返修、改变设备参数或旁路安全联锁。scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 25 章：主成分分析（PCA）

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 687–693 页，§20.1–§20.1.3.3 | 线性编码/解码、重构损失、最大方差、Lagrange 特征方程、多主成分、尺度选择、高维 Gram 路线及 SVD 等价 | 依据 CC BY-NC-ND 4.0 只作术语、公式、算法和知识边界核对；未复用或改编原书图表、数据、案例、代码和措辞 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 694–696 页，§20.1.4；PDF 第 697–701 页，§20.2–§20.2.3.3 | 重构误差随维数单调、scree/解释方差、profile likelihood、factor analysis、PPCA、残余方差、后验投影与 EM 入口 | 仅作维数选择、概率模型联系和扩展边界核对；压装案例、监测门禁、数值实现和三项网页实验均由本站独立创作 |
| Stanford CS229 Lecture Notes | PDF 第 166–171 页，Chapter 12 | PCA 的冗余检测动机、中心化/标准化、最大投影方差、特征向量、多维表示、最小重构入口与应用边界 | 用于交叉核对推导顺序、术语和尺度说明；不复制讲义图形、汽车/飞行员/人脸案例或表达 |
| Jonathan Richard Shewchuk，UC Berkeley CS 189/289A Lecture 20–21 | 合订讲义 PDF 第 123–131 页 | 中心化、协方差/相关系数选择、最大方差、最小正交距离、特征分解、解释方差、SVD 数值动机、得分与重构 | 用于核对三条推导、SVD 接口、复杂度和失效边界；不复用课堂图形、数据、案例或措辞 |

第 25 章伺服压装 EOL 过程指纹案例的 72 个教学 campaign、576 条合成记录、十二路量测、三个合成共同因子、子空间内远行与子空间外传感器关系冲突、分组留出协议、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实企业、电驱动总成、压机、轴套、量测系统、工艺窗口、质量规范、报警阈值或安全策略。PCA 只用于教学中的压缩与复核触发，不得自动修改速度、力限、位移终点、保压时间、返修策略、放行结果或安全联锁；scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 26 章：K-means 聚类

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 752–758 页，§21.3–§21.3.7；K-means++ 见 PDF 第 755–756 页，模型选择边界见 PDF 第 757–758 页 | WCSS 与矩阵形式、交替最小化、Voronoi、初始化敏感性、空簇实例、向量量化、K-means++、K-medoids、mini-batch、WCSS 随 K 单调与 silhouette 入口 | 依据 CC BY-NC-ND 4.0 只作术语、公式、算法和知识边界核对；未改编或复用原书图表、数据、代码、案例和措辞 |
| Stanford CS229 Lecture Notes | PDF 第 146–149 页，Chapter 10 | Lloyd 分配/均值两步、distortion、坐标下降、目标单调、局部最优、多起点 | 用于交叉核对推导顺序、收敛声明和非凸边界；不复制讲义图形、示例、练习或表达 |
| CMU 10-301/10-601 Lecture 25，*K-means, PCA* | PDF 第 10–42 页；初始化与 K-means++ 见 PDF 第 23–40 页 | 目标函数、block coordinate descent、随机初始化反例、farthest-point 风险、D² 采样、肘部启发与非球形失效 | 用于核对知识覆盖与初始化反例边界；不复用课堂图形、投票、表格、数据或措辞 |
| Jonathan Richard Shewchuk，UC Berkeley CS 189/289A 合订讲义 | PDF 第 134–137 页 | Lloyd/NP-hard 背景、均值与最近中心两步、平票、有限分配停止、局部最优、Forgy/随机分区/K-means++、多起点和尺度 | 用于核对术语区分、停止论证、尺度与工程边界；不复制课堂图形、案例或表达 |

第 26 章铸造车间压缩空气站工况归档案例的 66 个教学 campaign、528 条合成窗口、五路量测、三个隐藏生成状态、四条受控高残差记录、分组留出协议、三项网页交互、全部 NumPy 代码、题目和 SVG 图形均为本站独立创作，不代表真实企业、铸造工艺、空压机性能、能耗基准、露点要求、群控策略、维修根因、安全阈值或监管结论。统计簇不得直接命名为已确认物理根因；模型只用于教学中的归档、抽样和复核触发，不得启停设备、修改压力设定、旁路保护或自动生成维修结论。scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 27 章：聚类评估与模型选择

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 745–747 页，§21.1–§21.1.1.3 | 内部/外部评价边界、Purity、Rand、ARI、互信息与 NMI | 依据 CC BY-NC-ND 4.0 只作术语、公式、算法和知识边界核对；未改编或复用原书图表、数据、代码、案例和措辞 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 748–752 页，§21.2–§21.2.3 | HAC、Dendrogram、single/complete/average linkage、复杂度与几何偏好 | 用于核对层次聚类定义与边界；手算量规、实现、题目和网页实验均由本站独立创作 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 757–760 页，§21.3.7 | WCSS 随 K 单调、Silhouette 定义、样本级解释和球形簇偏好 | 用于核对指标定义和失效边界；不复用原书图示、示例或表达 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 764–768 页，§21.5–§21.5.4 | 图相似度、图拉普拉斯、连通分量、归一化谱嵌入、非凸示例与随机游走联系 | 用于核对谱聚类推导链与知识范围；不复用原书图形、数据、案例或措辞 |
| Jonathan Richard Shewchuk，UC Berkeley CS 189/289A 合订讲义 | PDF 第 137–140 页 | HAC、三种 linkage、Dendrogram 切割、横向叶序、chaining、outlier 与稳定性 | 用于交叉核对几何解释和失败边界；不复制课堂图形、示例或表达 |
| Jonathan Richard Shewchuk，UC Berkeley CS 189/289A 合订讲义 | PDF 第 175–184 页，Bonus Lecture C | 拉普拉斯二次型、Fiedler 向量、sweep cut、Ncut、多特征向量谱嵌入、逐行归一化与非凸比较 | 用于交叉核对图切割和谱方法边界；三项网页交互、SVG 图形与工程审计流程均独立创作 |

第 27 章压缩空气站复核队列验收案例的 66 个教学 campaign、528 条合成窗口、五项过程指数、三个隐藏生成模式、36 条冻结后审计记录、非凸滞回轨迹压力测试、四层硬门协议、全部 NumPy 代码、17 道即时测验和三项专用 SVG 网页实验均为本站独立创作，不代表真实企业、空压机性能、能耗基准、控制器逻辑、故障根因、复核产能、安全阈值或监管结论。统计簇只用于教学中的归档、抽样与复核触发，不得直接启停设备、修改压力设定、旁路保护或自动生成维修结论；scikit-learn 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 28 章：PAC 学习与 VC 维

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Stanford CS229 Lecture Notes | PDF 第 127–132 页，§8.3–§8.3.2 | 并合界、Hoeffding、经验/真实风险、有限假设类一致收敛、样本复杂度与 ERM 超额风险链 | 仅作定义、公式、推导顺序和知识边界核对；连接器案例、代码、题目与三项网页实验均为本站独立创作 |
| Stanford CS229 Lecture Notes | PDF 第 132–136 页，§8.3.3 | 无限假设类、shattering、VC 维、二维半空间 VC 维与 VC 泛化尺度 | 用于交叉核对增长函数到 VC 容量的关键步骤；不复制讲义图形、示例、练习或表达 |
| Stanford CS229 Lecture Notes | PDF 第 136–143 页，第 9 章 §9.1–§9.3 | 显式正则、范数、优化器隐式偏置、holdout、交叉验证与模型选择 | 用于核对结构风险控制和验证边界；不复用讲义案例、代码或措辞 |
| CMU 10-601/10-701 Lecture 15，*PAC Learning* | PDF 第 15–41 页 | PAC 判据、有限/无限与 realizable/agnostic 四种情形、有限类样本复杂度和复杂度权衡 | 用于交叉核对 $1/\epsilon$ 与 $1/\epsilon^2$ 两类样本尺度、模型选择边界；不复制课件图形、课堂测验或表达 |
| CMU 10-601/10-701 Lecture 16，*PAC, MAP* | PDF 第 3–39 页 | VC 维量词、射线与区间、结构风险最小化、正则化和过参数化深度学习边界 | 用于核对 VC 下界/上界的量词与 SRM 入口；网页光学台、证书走廊和批次差值尺均为本站独立设计 |
| Jonathan Richard Shewchuk，UC Berkeley CS 189/289A 合订讲义 | PDF 第 163–168 页，Bonus Lecture A | 风险、ERM、Hoeffding、统一界、增长函数、二维半空间标记数与 VC 维 | 用于交叉核对统一收敛链、14/16 标记反例和容量解释；不复用课堂图形、数据、案例或措辞 |

第 28 章高压连接器压接截面视觉终检资格审查案例的 72 个教学 lot、864 条合成记录、四个分数家族、冻结 lot 切分、搜索账本、规格 A/B/C、概率校准反例、全部 NumPy 代码、17 道即时测验和三项专用 SVG 网页实验均为本站独立创作，不代表真实企业、连接器结构、缺陷机理、检测性能、放行阈值、安全结论或监管要求。模型只用于教学中的资格审查，不得自动放行、修改工艺参数、旁路人工复核或替代安全验证；SciPy 对照明确标为可选未执行内容，仓库标准验证环境未为此新增依赖。

### 第 29 章：矩阵分解与 BPR 推荐

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 741–746 页，§22.1–§22.1.4 | 显式反馈、稀疏观测集、偏置矩阵分解、正则目标、SGD、ALS 与概率解释 | 依据 CC BY-NC-ND 4.0 只作术语、公式、推导顺序和知识边界核对；未改编或复用原书图表、数据、代码、案例和措辞 |
| Kevin P. Murphy，*Probabilistic Machine Learning: An Introduction* | PDF 第 747–751 页，§22.2–§22.4 | 隐式反馈、BPR 入口、侧信息、反馈回路与探索—利用 | 用于核对推荐系统从观测行为到闭环部署的知识范围；不复用原书图形、示例、实验或表达 |
| Steffen Rendle 等，*BPR: Bayesian Personalized Ranking from Implicit Feedback* | PDF 第 2–7 页，§3–§4.3 | 个性化排序三元组、成对 logistic 似然、MAP 目标、LearnBPR 采样、BPR-MF 更新 | 用于核对原始目标、梯度和采样算法；检修案例、实现、题目与成对分拣器均为本站独立创作 |
| Steffen Rendle 等，*BPR: Bayesian Personalized Ranking from Implicit Feedback* | PDF 第 8–10 页，§5–§7 | 与 pointwise/WR-MF 的实验比较、评价协议、目标与指标对齐 | 用于核对结论适用范围和评价边界；不复制论文数据、结果表、图形、代码或措辞 |
| Yifan Hu、Yehuda Koren、Chris Volinsky，*Collaborative Filtering for Implicit Feedback Datasets* | PDF 第 1–5 页 | 无可靠负例、偏好/置信分离、加权 ALS 法方程、稀疏复杂度化简 | 用于核对置信加权隐式矩阵分解的推导和工程复杂度；不复用论文案例、数据、图表或表达 |
| Yifan Hu、Yehuda Koren、Chris Volinsky，*Collaborative Filtering for Implicit Feedback Datasets* | PDF 第 7–10 页 | 全排序评价、热门/邻域/因子基线、head-tail 差异、离线行为预测边界 | 用于核对切片评价和“预测历史行为不等于发现价值”的限制；网页实验和资格协议均独立设计 |

第 29 章检修知识库推荐案例的 72 名教学工程师、64 张合成知识卡、1296 条曝光评分、四类主题、8 张 cold 卡、时间切分、全部 NumPy 代码、17 道即时测验和三项专用 SVG 网页实验均为本站独立创作，不代表真实企业、人员能力、设备状态、检修规程、卡片质量、推荐收益或安全结论。推荐只用于教学中的知识卡排序，不得自动修改检修步骤、跳过安全规程、变更设备参数或替代专业判断；仓库标准验证环境未为本章新增依赖。

### 第 30 章：马尔可夫决策过程与动态规划

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 4 | PDF 第 5–7 页 | Markov chain、MDP、POMDP 与完全可观测边界 | 仅作术语、状态充分性和知识范围核对；晶圆载具案例、状态审计与网页事件记录器均由本站独立创作 |
| Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 4 | PDF 第 14–15 页、23–25 页 | 无限/有限时域的期望目标、状态价值、动作价值与算法接口 | 用于核对目标与价值函数定义；不复制课件图形、机器人案例、推导版式或表达 |
| CMU 10-301/10-601 Lecture 21，*Reinforcement Learning: Markov Decision Processes* | PDF 第 45–62 页 | 价值迭代、Bellman 最优方程、策略抽取和每轮复杂度 | 用于核对推导顺序、算法边界与复杂度；Bellman 配电盘、数值模型、代码和题目均由本站独立创作 |
| CMU 10-301/10-601 Lecture 21，*Reinforcement Learning: Markov Decision Processes* | PDF 第 78–90 页 | 固定策略评价、线性系统、策略改进、策略迭代及与价值迭代的关系 | 用于交叉核对两类动态规划；不复用课件 Gridworld 图形、课堂问题、数值或措辞 |
| Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 7 | PDF 第 5–9 页 | 动态规划和评价—改进式策略迭代 | 仅作算法结构交叉核对；动态规划脉冲地坪与工业审计流程均独立设计 |

第 30 章晶圆载具转运走廊案例的 14 个有效状态、4 个请求动作、随机偏移模型、奖励量规、密封状态别名审计、全部 NumPy 代码、18 道即时测验和三项专用网页实验均为本站独立创作，不代表真实晶圆厂、AMR、洁净区、地图、交通规则、设备能力、损失量规、安全规范或控制策略。规划器只用于离线教学，不得连接真实设备、下发路线、修改速度或避障参数，也不得替代门禁、人员隔离、急停或人工放行；仓库标准验证环境未为本章新增依赖。

### 第 31 章：SARSA 与 Q-learning

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Sergey Levine，UC Berkeley Deep Reinforcement Learning Section 3 | PDF 第 5–8 页 | Monte Carlo return、TD bootstrap、stop-gradient、n-step return 与偏差—方差交换 | 用于核对回报 target、TD 估计和 n-step 边界；晶圆载具样本带、代码、题目和回报撕纸线均为本站独立创作 |
| Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 7 | PDF 第 14–20 页 | fitted/online Q-learning、off-policy、epsilon-greedy 探索与 replay 入口 | 用于核对 Q-learning 的样本更新、行为/目标策略和探索范围；不复制课件图形、机器人案例或表达 |
| Sergey Levine，UC Berkeley Deep Reinforcement Learning Section 4 | PDF 第 1–3 页 | 表格 Q-learning、经验回放、目标网络、终止 mask、epsilon-greedy、Double DQN | 用于核对从表格到 DQN 的算法接口和终止语义；继电器柜与路线回放器均由本站独立设计 |
| Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 8 | PDF 第 3–8 页、14–22 页 | target stop-gradient、经典 DQN、目标网络、最大化过估计、Double Q-learning 与实践稳定化 | 用于交叉核对深度 Q-learning 的故障机制和缓解组件；不复用课件图形、实验、代码或措辞 |

第 31 章的经典 SARSA 与 Expected SARSA 由 Bellman 期望方程独立推导；本地 PDF 语料没有单独覆盖其经典表格算法页，因此未把邻近内容登记成直接来源。本章沿用第 30 章的 14 状态教学环境，但学习器只接收 transition；隐藏模型只用于离线资格审查。晶圆载具样本流、行为策略合同、全部 NumPy 代码、18 道即时测验和三项专用网页实验均为本站独立创作，不代表真实晶圆厂、AMR、地图、交通规则、损失量规、安全规范或控制策略。随机探索只允许在合格仿真器、数字孪生或隔离教学环境中运行，不得连接真实设备或安全控制链；仓库标准验证环境未为本章新增依赖。

### 第 32 章：REINFORCE 与 Actor–Critic

| 来源 | 精确位置 | 采用的知识点 | 使用方式 |
|---|---|---|---|
| Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 5 | PDF 第 2–7 页、15–18 页、20–23 页 | 轨迹目标、对数导数、REINFORCE、方差问题、baseline 无偏性、reward-to-go 与离散实现 | 用于核对推导链、因果性与实现边界；晶圆载具案例、代码、题目和策略梯度信号台均为本站独立创作 |
| Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 6 | PDF 第 3–10 页、20–25 页、32–33 页 | 价值 baseline、on-policy Actor–Critic、Critic 偏差、n-step、GAE、重参数化和 SAC 边界 | 用于核对从 Monte Carlo baseline 到 bootstrap Critic 的偏差—方差交换；不复制课件图形、案例、代码或表达 |
| UC Berkeley Deep Reinforcement Learning Section 3 | PDF 第 5–8 页 | Actor–Critic、Critic target stop-gradient、n-step return、GAE 定义与倒序递推 | 用于交叉核对 target、递推与终止边界；方差条带记录仪、双 mask 实现和数值金丝雀均独立设计 |
| CMU 10-601/10-701 Lecture 22，*Policy Gradient + Deep Reinforcement Learning* | PDF 第 26–33 页、40–47 页 | 策略梯度定理、Softmax score、REINFORCE、失效模式、baseline、Actor–Critic、A2C 与 n-step 对照 | 用于独立核对公式、算法粒度和失效条件；不复用课堂例题、投票题、图形、代码或措辞 |
| Sergey Levine，UC Berkeley Deep Reinforcement Learning Lecture 9 | PDF 第 14–16 页 | clipped importance ratio、熵正则与 PPO | 仅用于现代稳定化直觉和符号边界；PPO 正负 advantage 金丝雀及工程监控协议均由本站独立创作 |

第 32 章沿用第 30–31 章的 14 状态教学走廊，但 Actor 直接学习动作分布，训练器仍只接收 sample-only rollout；隐藏转移核仅用于解析策略梯度、有限差分与冻结策略资格审查。轨迹合同、Softmax/Actor–Critic/GAE/PPO NumPy 实现、22 道即时测验、三项专用网页实验和全部数值结果均为本站独立创作，不代表真实晶圆厂、AMR、地图、交通规则、设备能力、损失量规、安全规范或发布标准。随机策略只允许在合格仿真器、数字孪生或隔离教学环境中运行，不得连接真实设备或安全控制链；仓库标准验证环境未为本章新增依赖。

## 资料总览

本地语料的范围、解析方法和校验结果见仓库内 `ref/ml101/README.md` 与 `ref/ml101/opendataloader-pdf/README.md`。教材章节安排见 [机器学习教材章节命名与模块划分](../machine-learning-chapter-structure.md) 。
