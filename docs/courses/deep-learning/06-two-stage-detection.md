---
course: deep-learning
edition: industrial
title: R-CNN 系列与两阶段目标检测
description: 从 proposal、anchor 匹配、RoI 对齐、分类回归多任务损失到 NMS，建立 Faster R-CNN 与 Mask R-CNN 的训练和评价合同。
outline: [2, 3]
---

<DlLessonHero number="06" module="计算机视觉" title="R-CNN 系列与两阶段目标检测" question="检测器输出的不只是类别，而是一组数量可变、需要去重、必须与标注匹配的空间对象；监督信号怎样进入网络？" duration="建议 280–360 分钟" :outcomes="['解释 R-CNN 到 Faster R-CNN 的共享计算演进', '逐步构造 anchor 标签、框参数化与多任务损失', '说明 RoI Pooling 与 RoIAlign 的量化差异', '用 AP、尺寸切片、NMS 和错误分解审计两阶段检测器']" />

本章不重复卷积基础，也不把检测简化成“分类加四个数字”。检测输出是集合：每张图的目标数不同，框之间会重复，背景远多于前景，评价还要跨置信度阈值和 IoU 阈值积分。

## 贯穿案例：产线托盘上的密集零件检测

每张 1920×1080 图像包含 5–80 个金属零件，小目标宽度可能只有 12 像素。业务要求漏检代价远高于重复框，但机械臂只能接收互斥实例。训练标注来自多批次相机，边界框紧致程度并不一致。

检测合同先固定：坐标是 `xyxy` 还是 `xywh`；是否包含右下边界；原图到 resize/letterbox 的变换；空图如何表示；ignore/iscrowd 如何处理；类别是否包含显式背景；NMS 和 score 阈值放在模型内还是服务层。

<DlQuiz kind="数据合同" question="水平翻转检测图像时，只翻图像而不更新框会怎样？" :options="['只是轻微噪声', '监督输入与目标不再对应，模型会学习错误定位信号', '只影响分类不影响回归', '框会由 NMS 自动修复']" :answer="1" explanation="几何增强必须对图像和所有空间标注做同一变换，并裁剪、过滤无效框。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>为一张没有任何目标的合法背景图写出 target shape、RPN 标签、RoI 采样和损失行为。哪些实现会因空张量而崩溃？</div>

## 1. 从逐 proposal 卷积到共享特征

R-CNN 用外部 selective search 生成约 2000 个 region proposal，每个 proposal 单独裁剪并经过 CNN，计算重复。Fast R-CNN 先整图计算特征图，再把 proposal 映射到特征图做 RoI Pooling，共享骨干。Faster R-CNN 进一步用 Region Proposal Network（RPN）学习候选区域，与检测头共享特征。

| 系统 | proposal 来源 | 骨干计算 | 端到端训练接口 |
|---|---|---|---|
| R-CNN | 外部算法 | 每个区域重复 | 多阶段 |
| Fast R-CNN | 外部算法 | 整图一次 | 分类回归头联合 |
| Faster R-CNN | RPN 学习 | RPN 与 RoI 共享 | 近似统一 |

“two-stage”指先 proposal、再按 proposal 精分类/回归，不是训练一定分两次。

## 2. Anchor 是候选坐标系，不是目标

在特征图每个位置放置多种尺度和长宽比的 anchor。训练时按 IoU 与 ground truth 匹配：高于正阈值为正样本，低于负阈值为负样本，中间忽略；还常保证每个 GT 的最佳 anchor 为正。

$$
\operatorname{IoU}(A,G)=\frac{|A\cap G|}{|A\cup G|}.
$$

若输入 stride 为 16，一个 12 像素目标可能只覆盖不到一个 feature cell。增加更细 FPN 层、调整 anchor 尺度或提高分辨率，比盲目降低 IoU 阈值更直接。

```python
def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0

assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
```

<AnchorMatchLab />

<DlQuiz kind="标签分配" question="为什么在正负 IoU 阈值之间设置 ignore 区间？" :options="['节省标签文件空间', '避免把边界模糊的中间候选强制当作清晰正负样本', '让所有 anchor 都成为正样本', '只用于推理']" :answer="1" explanation="中间 IoU 的语义不稳定，忽略可减少冲突监督；但也会改变有效样本数。" />

## 3. 框回归学习相对变换

对 anchor/proposal $A=(x_a,y_a,w_a,h_a)$ 和目标 $G=(x_g,y_g,w_g,h_g)$，常见编码为

$$
t_x=\frac{x_g-x_a}{w_a},\quad t_y=\frac{y_g-y_a}{h_a},\quad
t_w=\log\frac{w_g}{w_a},\quad t_h=\log\frac{h_g}{h_a}.
$$

平移按 anchor 尺度归一化，尺寸用对数比，缓解绝对像素范围差异。训练只对正样本计算回归损失；否则背景没有合法目标框。

```python
import math

def encode(anchor, target):
    xa, ya, wa, ha = anchor
    xg, yg, wg, hg = target
    return ((xg-xa)/wa, (yg-ya)/ha, math.log(wg/wa), math.log(hg/ha))

delta = encode((50, 40, 100, 80), (60, 36, 120, 72))
assert all(math.isfinite(value) for value in delta)
assert delta[0] == 0.1
```

分类损失和回归损失的归一化分母必须明确：按采样 anchor 数、正样本数还是 batch 图像数？空图下除以正样本数会产生 NaN，应使用明确的零回归损失和安全分母。

<DlQuiz kind="损失接口" question="RPN 中负 anchor 是否应计算框回归损失？" :options="['应，因为每个 anchor 都有坐标', '不应，背景没有对应 GT 几何目标，通常只贡献 objectness 分类', '只在推理时计算', '必须回归到整张图']" :answer="1" explanation="回归监督只对匹配到 GT 的正 anchor 定义。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>一张图只有 2 个正 anchor，另一张有 200 个。按全 batch 正样本平均和按每图平均会怎样改变两张图的梯度权重？</div>

## 4. RoI Pooling 与 RoIAlign

Fast/Faster R-CNN 需要把大小不同的 proposal 变成固定尺寸特征。RoI Pooling 会量化 proposal 边界和 bin，坐标误差对分类可能可忍受，对像素级 mask 边界则明显。RoIAlign 不量化边界，在每个 bin 的采样点用双线性插值。

设连续坐标 $(x,y)$ 落在四个特征点之间，双线性插值是四点按距离加权：

$$
f(x,y)=\sum_{i\in\{0,1\}}\sum_{j\in\{0,1\}}
w_{ij}f(x_i,y_j).
$$

坐标对齐还依赖 resize 比例、feature stride 和框坐标约定。`aligned=True` 之类的半像素规则必须纳入复现指纹。

## 5. FPN 把语义送回高分辨率

Feature Pyramid Network 用 top-down 路径和 lateral connection 把深层语义与浅层高分辨率结合，产生 $P_2,P_3,\dots$。小目标送往细层，大目标送往粗层。分层规则通常依赖 proposal 面积；输入尺度变化会改变分配分布。

检查 FPN 时记录每层 stride、通道、anchor 尺度、正样本数和召回率。若 $P_2$ 有巨大激活却几乎没有正样本，它只在消耗预算。

<DlQuiz kind="多尺度" question="FPN 的 lateral connection 主要连接什么？" :options="['不同图像样本', '同一骨干的相邻尺度，把高层语义与低层空间细节融合', '分类器与优化器', 'CPU 与 GPU']" :answer="1" explanation="横向连接投影对应 stage 特征，top-down 上采样后相加或融合。" />

## 6. 推理：分数门、解码、裁剪与 NMS

典型推理链是：预测 objectness/类别与 delta；解码为框；裁剪到图像；删除过小框；按分数取 top-k；按类别或类无关执行 NMS。顺序变化会改变输出和延迟。

<NmsWorkbenchLab />

硬 NMS 从最高分框开始，删除与其 IoU 超阈值的同类框。密集相邻零件可能被误删；过高阈值又留下重复框。score 与 NMS 阈值必须在验证集按业务代价选择，不能只复制 COCO 默认。

```python
def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall

strict = precision_recall(tp=72, fp=8, fn=28)
loose = precision_recall(tp=89, fp=31, fn=11)
assert strict[0] > loose[0]
assert loose[1] > strict[1]
```

<DlQuiz kind="阈值决策" question="降低 score 阈值通常对 precision 和 recall 有何方向性影响？" :options="['precision 和 recall 都必然提高', '候选更多，recall 常提高而 precision 可能下降', '两者完全不变', '只改变框坐标']" :answer="1" explanation="更低阈值接纳更多真阳性和假阳性，具体幅度需从验证数据测量。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>密集零件场景中 recall 高但重复框多。分别说明调 score、调 NMS IoU、改 Soft-NMS、改标注一致性会产生什么可观测后果。</div>

## 7. AP 不是一个阈值下的 F1

预测按分数排序，在指定 IoU 匹配阈值下逐个标记 TP/FP，得到 precision-recall 曲线并积分为 AP。COCO AP 通常跨 IoU 0.50:0.95 平均，还按 small/medium/large 切片。必须同时看：

- AP50 与 AP75 的差距：定位是否粗糙；
- APS/APM/APL：尺度是否失衡；
- 每类 AP 与业务召回；
- proposal recall：问题在 RPN 还是 RoI head；
- 分类、定位、重复、背景假阳性分解。

## 8. 推荐论文阅读

<DlPaperCard title="Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" meta="Ren et al. · 2015" priority="必读 · 两阶段核心" question="RPN 怎样共享骨干特征，anchor 分类与回归如何被采样和联合优化？" reading="精读 RPN 标签定义、多任务损失、proposal 数和速度表；逐项写 shape。" href="https://arxiv.org/pdf/1506.01497" />

<DlPaperCard title="Feature Pyramid Networks for Object Detection" meta="Lin et al. · 2016" priority="必读 · 多尺度" question="top-down 与 lateral path 如何让每一尺度都拥有强语义？" reading="读架构图、层级分配和消融；把 stride、anchor 尺度与目标尺寸对齐。" href="https://arxiv.org/pdf/1612.03144" />

<DlPaperCard title="Mask R-CNN" meta="He et al. · 2017" priority="必读 · 对齐接口" question="RoIAlign 为什么对 mask AP 关键，分类与 mask 分支为何解耦？" reading="精读 RoIAlign 量化差异与消融；第 09 章再读 mask head 和实例输出。" href="https://arxiv.org/pdf/1703.06870" />

## 9. 练习与门禁

1. 实现 box encode/decode 往返测试，覆盖零面积、越界和极端长宽比。
2. 统计每张图、每个 FPN level 的正/负/忽略 anchor，画目标尺寸条件分布。
3. 对 NMS IoU 与 score 做二维扫描，报告机械臂重复抓取与漏检代价。
4. 比较 RoI Pooling 与 RoIAlign 在小目标和大目标上的 AP75。
5. 做 100 proposals 到 1000 proposals 的 recall/延迟 Pareto 曲线。
6. 构造空图、单目标、完全重叠双目标和 100 目标的回归测试。

## 本章小结

- 两阶段检测器通过共享特征的 RPN 生成 proposal，再精分类与回归。
- anchor 匹配、回归参数化、采样和损失归一化共同定义监督接口。
- RoIAlign 通过连续坐标插值避免量化，对实例 mask 尤其关键。
- FPN 建立多尺度语义接口，但每层预算必须与真实尺寸分布核对。
- NMS 与 score 阈值是业务决策的一部分；AP 需要尺寸、IoU 和错误类型分解。

<DlLessonProgress lesson-id="06" next-href="/courses/deep-learning/07-one-stage-detection-detr" next-label="下一章：单阶段检测与 DETR" />
