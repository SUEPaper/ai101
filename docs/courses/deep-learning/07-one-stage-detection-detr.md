---
course: deep-learning
edition: industrial
title: YOLO、RetinaNet 与 DETR
description: 比较单阶段密集预测、Focal Loss 与 Transformer 集合预测，拆解标签分配、解码、匹配和端到端延迟的差异。
outline: [2, 3]
---

<DlLessonHero number="07" module="计算机视觉" title="YOLO、RetinaNet 与 DETR" question="去掉 proposal stage 之后，检测问题真的更简单了吗；如果再去掉 anchor 和 NMS，新复杂度转移到了哪里？" duration="建议 280–370 分钟" :outcomes="['解释 YOLO 式密集预测的网格、尺度与解码接口', '推导 Focal Loss 对易样本的连续降权', '构造 DETR 的 object query、集合损失与二分匹配', '按精度、收敛、吞吐和后处理比较三类检测器']" />

两阶段方法把候选生成与精分类分开；单阶段方法直接在密集位置预测类别和框，减少不规则 RoI 处理。DETR 更进一步，把检测写成固定长度集合预测，通过一对一匹配训练，通常不依赖 NMS。复杂度没有消失，只是从 proposal/NMS 转移到标签分配、object query、匹配代价与训练收敛。

## 贯穿案例：高速传送带异物检测

相机每秒 120 帧，绝大多数像素是背景，异物尺寸从 8 到 300 像素。服务要求单帧端到端 P99 低于 7 ms，允许合并很少的相邻框，但不允许漏掉刀片状细长物。我们将比较 YOLO 类实时检测器、RetinaNet 与小型 DETR 变体。

<DlQuiz kind="系统边界" question="评价实时检测器时，端到端延迟应包含什么？" :options="['只包含 backbone kernel', '至少包含预处理、模型、解码、阈值/NMS 与必要的数据传输', '只包含文件读取', '只包含后处理']" :answer="1" explanation="不同架构把工作放在模型内外不同位置；只测 forward 会产生不公平比较。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>若模型 forward 为 4 ms、CPU NMS 为 5 ms，而另一模型 forward 为 7 ms 且无需 NMS，哪个更“实时”？还缺哪些并发和尾延迟信息？</div>

## 1. YOLO：把检测写成规则密集张量

早期 YOLO 将图像划分为网格，每个 cell 预测有限框与类别；现代 YOLO 系列通常在多尺度 feature map 上输出形如

$$
B\times H\times W\times A\times(4+1+C)
$$

的张量，或使用 anchor-free 变体。四个框参数、objectness 与类别分数的具体语义随版本而变，不能只写“用了 YOLO”。

以 anchor-based 解码为例，中心偏移经过 sigmoid 限制到 cell 周围，宽高由 anchor 与指数或其他有界变换得到。训练/推理解码公式必须完全一致。

```python
import math

def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))

def decode_center(raw_x, raw_y, cell_x, cell_y, stride):
    return ((sigmoid(raw_x) + cell_x) * stride,
            (sigmoid(raw_y) + cell_y) * stride)

center = decode_center(0.0, 0.0, 4, 7, 8)
assert center == (36.0, 60.0)
```

单阶段速度来自规则张量和共享计算，但候选数量巨大。多尺度 head、标签分配、正负样本不平衡、框损失和 NMS 都影响结果。不同 YOLO 版本在这些接口上差异很大，报告必须写版本、实现、输入和导出方式。

<DlQuiz kind="张量接口" question="多尺度检测 head 的主要目的是什么？" :options="['让每个尺度预测完全相同的像素', '让细分辨率层处理小目标、粗分辨率层覆盖大目标，并共享类别语义', '删除所有下采样', '只提高参数量']" :answer="1" explanation="不同 stride 对应不同空间粒度与感受野，目标分配要与尺度匹配。" />

## 2. 密集预测的正负失衡

一张图可能有数万个候选位置，绝大多数是易分类背景。普通交叉熵对每个易负样本贡献虽小，数量累积后仍可能主导梯度。硬负样本挖掘离散选择一部分；Focal Loss 连续降低易样本权重：

$$
\operatorname{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t.
$$

当 $p_t\to1$，调制因子趋近 0；难样本保持较大贡献。$\gamma=0$ 时退化为带 $\alpha$ 的交叉熵。

<FocalLossLab />

```python
import numpy as np

def focal_loss(correct_probability, gamma=2.0, alpha=0.25):
    probability = np.clip(correct_probability, 1e-8, 1.0)
    return -alpha * (1.0 - probability) ** gamma * np.log(probability)

easy = focal_loss(np.array([0.99]))[0]
hard = focal_loss(np.array([0.20]))[0]
assert hard > easy * 1000
```

<DlQuiz kind="损失判断" question="Focal Loss 的 γ 增大时，最直接的作用是什么？" :options="['提高所有样本同样权重', '更强地压低高 p_t 易样本的贡献', '删除框回归损失', '自动修复漏标']" :answer="1" explanation="调制因子 (1-p_t)^γ 对易样本更小；漏标正例仍可能被当作难负例放大。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>数据中漏标了许多小目标。Focal Loss 会让这些候选作为“难负样本”获得怎样的权重？应先改损失还是先审计标注？</div>

## 3. RetinaNet：强 FPN + Focal Loss 的受控对照

RetinaNet 在 FPN 每层放置共享结构的分类与回归子网，使用 anchors 和 Focal Loss。论文的重要证据是：单阶段检测器曾落后并非必然因为“一阶段表达力差”，而是极端前景/背景失衡破坏了训练。

分类 prior bias 初始化也关键。若正样本先验 $\pi$ 很小，可将输出 bias 初始化为

$$
b=-\log\frac{1-\pi}{\pi},
$$

避免训练初始所有位置给出高正类概率、产生巨大损失。这个细节属于算法合同，不是可忽略工程参数。

<DlQuiz kind="初始化" question="密集检测分类头为何常按低正样本先验初始化 bias？" :options="['让初始预测符合稀疏前景，避免大量背景产生失控损失', '让所有框一开始满分', '删除类别不平衡', '只为节省参数']" :answer="0" explanation="合理先验让训练起点的 objectness/类别概率与数据稀疏性一致。" />

## 4. Anchor-free 并不等于 label-free

FCOS 类方法直接预测位置到四边距离和中心度，不使用预定义 anchor 尺度/长宽比，但仍必须决定哪些 feature 位置负责哪个目标、重叠目标怎样消歧、多尺度怎样分配。去掉 anchor 超参数后，中心采样半径、范围和动态分配成为新接口。

任何“anchor-free 更简单”的结论都应列出被移除和新增的超参数，而不是只比较配置文件长度。

## 5. DETR：检测作为集合预测

DETR 将骨干特征展平并加入位置编码，Transformer encoder 建模全局关系；固定数量的 learned object query 经 decoder 产生类别（含 no-object）和框。训练时用 Hungarian algorithm 在预测集合与 GT 集合之间找最小代价一对一匹配：

$$
\hat\sigma=\arg\min_{\sigma}
\sum_i\left[-\log p_{\sigma(i)}(c_i)
+\lambda_1\lVert b_i-\hat b_{\sigma(i)}\rVert_1
+\lambda_{\mathrm{giou}}L_{\mathrm{GIoU}}\right].
$$

一对一匹配使同一 GT 只监督一个 query，其余 query 学 no-object，因此通常不需要 NMS。

<SetMatchingLab />

```python
import itertools

cost = [[0.3, 1.4, 0.9], [1.2, 0.2, 0.8]]  # two targets, three queries
candidates = []
for assignment in itertools.permutations(range(3), 2):
    total = sum(cost[target][query] for target, query in enumerate(assignment))
    candidates.append((total, assignment))
best = min(candidates)
assert best == (0.5, (0, 1))
```

<DlQuiz kind="集合预测" question="DETR 中未匹配的 object query 学习什么？" :options="['随机目标框', 'no-object 类别', '复制最高分目标', '不参与任何损失']" :answer="1" explanation="未匹配 query 通常接受 no-object 分类监督，类别权重需处理数量不平衡。" />

### 5.1 DETR 的失效与改进方向

原始 DETR 收敛慢，对小目标较弱。全局 attention 的高分辨率成本也高。Deformable DETR 等方法让每个 query 只采样少量多尺度关键点，加快收敛并改善小目标。它说明“端到端”不是没有结构先验，而是把先验从 anchor/NMS 换成 query、attention 采样和匹配。

<div class="dl-stop"><strong>停下来推一推 3</strong>GT 数量偶尔超过 object query 数会怎样？query 数过多又如何改变 no-object 比例、显存与解码成本？</div>

## 6. 三类系统的公平比较

| 维度 | YOLO 类 | RetinaNet | DETR 类 |
|---|---|---|---|
| 输出 | 密集多尺度 | anchor 密集预测 | 固定 query 集合 |
| 标签分配 | 版本相关动态/静态 | IoU anchor | 一对一匹配 |
| 不平衡 | objectness/采样/损失 | Focal Loss | no-object 权重 |
| 去重 | 通常 NMS | NMS | 通常无需 NMS |
| 常见弱点 | 版本接口复杂 | 候选多、后处理 | 收敛与小目标 |

公平协议固定骨干规模、输入、增强、训练 token/图像数、预训练、精度和端到端计时边界。报告参数/FLOPs 之外，还要报告 kernel、后处理、峰值内存和 batch=1 尾延迟。

## 7. 推荐论文阅读

<DlPaperCard title="You Only Look Once: Unified, Real-Time Object Detection" meta="Redmon et al. · 2015" priority="必读 · 单阶段起点" question="把检测统一为单个规则输出张量带来了什么速度收益，又牺牲了哪些定位与小目标能力？" reading="读统一预测公式、误差分析和速度表；不要把 2015 结构等同于所有现代 YOLO。" href="https://arxiv.org/pdf/1506.02640" />

<DlPaperCard title="Focal Loss for Dense Object Detection" meta="Lin et al. · 2017" priority="必读 · 类别失衡" question="单阶段检测差距能否由易负样本主导梯度解释，γ 与 α 分别做什么？" reading="精读调制因子、prior bias、RetinaNet 对照与 loss 消融。" href="https://arxiv.org/pdf/1708.02002" />

<DlPaperCard title="End-to-End Object Detection with Transformers" meta="Carion et al. · 2020" priority="必读 · 集合预测" question="二分匹配如何让检测摆脱 anchor 与 NMS，代价为何必须同时包含类别和几何？" reading="读 set prediction loss、object queries、训练 schedule 和小目标误差。" href="https://arxiv.org/pdf/2005.12872" />

## 8. 练习与门禁

1. 为一个多尺度 YOLO head 写出每层输出 shape、解码和原图坐标逆变换测试。
2. 画 $\gamma\in\{0,1,2,4\}$ 下 Focal Loss 曲线，分别分析易正、难正、易负和漏标正例。
3. 构造小型匹配代价矩阵，手算最优 assignment，并与穷举程序核对。
4. 固定 forward 规模，对 CPU NMS、GPU NMS 和无 NMS 系统测端到端 P99。
5. 按小/中/大与细长目标切片比较 YOLO、RetinaNet、DETR。
6. 注入 object query 不足、NMS 过强和漏标三种故障，记录错误指纹。

## 本章小结

- 单阶段方法用规则密集预测换取吞吐，但标签分配、不平衡与后处理仍复杂。
- Focal Loss 连续压低易样本贡献，不能修复漏标或目标覆盖问题。
- anchor-free 仍需要位置到目标的责任分配。
- DETR 用 object query 和一对一集合匹配减少重复预测，复杂度转移到匹配与收敛。
- 公平检测比较必须包含端到端后处理、尾延迟、目标尺寸和错误类型。

<DlLessonProgress lesson-id="07" next-href="/courses/deep-learning/08-semantic-segmentation" next-label="下一章：语义分割" />
