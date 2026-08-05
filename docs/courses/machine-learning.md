---
course: machine-learning
title: 机器学习
description: 从建模假设、目标函数和完整推导出发，用 NumPy 与交互实验理解经典机器学习和强化学习。
outline: false
---

<div class="ml-course-hero">
  <p class="ml-course-hero__label">AI101 · MACHINE LEARNING</p>
  <h1>不要背算法。<br>把它推出来。</h1>
  <p>这是一门从“会调用模型”走向“知道模型为什么成立”的网页版教材。每个核心算法都从可观察问题出发，写清假设，逐步推出目标函数和更新规则，再用 NumPy、数值检查与交互实验验证。</p>
  <div class="ml-stat-strip">
    <div><b>32 章</b><span>九个模块的理论主线</span></div>
    <div><b>32 个实验</b><span>公式、曲线与训练状态联动</span></div>
    <div><b>126–163h</b><span>正文、手推、代码与作业</span></div>
    <div><b>6 个选修章</b><span>数学工具不阻断核心学习</span></div>
  </div>
</div>

## 这门课解决什么问题

在数据科学课中，你已经用过线性回归、逻辑回归、PCA 和 K-means。现在要追问更深一层：平方损失从哪里来？正规方程为什么成立？一个解什么时候唯一？Softmax 为什么要减去最大值？随机森林为何能降低方差？Bellman backup 为什么会收敛？

课程把算法写成同一种可审查结构：

<div class="ml-proof-map">
  <div><b>PROBLEM</b><span>先定义预测或决策</span><small>输入、输出、未知分布与错误代价</small></div>
  <div><b>ASSUMPTION</b><span>明确允许模型相信什么</span><small>函数空间、概率分布、独立性或 Markov 性</small></div>
  <div><b>DERIVATION</b><span>从目标逐行推出算法</span><small>梯度、闭式解、对偶、动态规划或随机估计</small></div>
  <div><b>CHECK</b><span>让公式接受数值检验</span><small>有限差分、等价实现、反例和失败条件</small></div>
</div>

你不会看到只给结论的“公式墙”。新符号在第一次出现时定义；矩阵运算同时标注形状；长推导先给路线图，再逐行说明用了哪个事实。每章还有三次“停下来推一推”、至少五道即时反馈题，以及一个 30–60 分钟的推导与代码练习。

## 开始前需要会什么

这门课面向已经完成本站[数据科学课程](./data-science.md)的学习者。你应能使用 Python、NumPy、Pandas、Matplotlib 和 scikit-learn，理解训练/验证/测试划分、均值、方差、条件概率，以及矩阵的行列含义。

不要求先完成一整套高等数学课程。矩阵微分会在反向传播处就地解释，Lagrange 乘子会在 PCA 与 SVM 处就地使用，Jensen 不等式会在 EM 处从图形和切线开始。想看系统证明时，再进入 S01–S06 数学工具箱。

<div class="ml-note"><strong>推荐学习节奏</strong>每次安排 90–150 分钟：先读到核心推导结束，合上答案手算关键一步；再拖动实验参数预测曲线变化；最后运行 NumPy 代码并完成检查题。看懂动画不等于会推导，真正的完成证据是你能解释每一项为什么出现。</div>

## 课程地图

<MlCourseMap />

## 作业、项目与资料边界

- [八次综合作业](./machine-learning/assignments.md)把相邻章节组合成从推导到失效分析的完整任务。
- [机器学习结课项目](./machine-learning/project.md)提供监督预测、推荐系统和强化学习三条路线。
- [来源与许可登记](./machine-learning/sources.md)说明原创内容、参考资料和可复用素材的边界。
- [教材大纲与完成标准](./machine-learning-textbook-outline.md)记录 32 章的范围、练习结构和验收条件。

## 第一条主线

从[第 01 章：学习问题怎样被数学化](./machine-learning/01-learning-problem.md)开始。你会先把一个模糊任务写成输入空间、输出空间、假设空间、损失和数据分布，再看经验风险如何把“想学会”变成“可以计算”。
