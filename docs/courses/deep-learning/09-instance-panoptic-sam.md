---
course: deep-learning
edition: industrial
title: 实例分割、全景分割与 SAM
description: 从 Mask R-CNN 的实例掩码、全景装配与 PQ，到 SAM 的提示编码、掩码歧义和数据引擎，建立可提示分割的能力边界。
outline: [2, 3]
---

<DlLessonHero number="09" module="计算机视觉" title="实例分割、全景分割与 SAM" question="像素类别、对象身份和用户提示分别提供什么监督；一个基础分割模型又不能替代哪些任务模型？" duration="建议 280–370 分钟" :outcomes="['区分 semantic、instance 与 panoptic 输出及评价', '解释 Mask R-CNN 的类条件 mask 与 RoIAlign', '构造全景互斥 ID、匹配和 PQ 账本', '审计 SAM 的图像编码、提示编码、候选 mask 与领域失效']" />

语义分割回答“每个像素是什么类”，实例分割还要回答“属于哪个对象”，全景分割要求所有像素都有语义且 thing 类分实例。SAM 则引入新的任务接口：给定点、框或粗 mask 提示，输出一个或多个合理掩码。

## 贯穿案例：混料托盘的可交互标注系统

标注员先点击或框选零件，系统给出候选 mask，再由人工修正并用于下游实例分割训练。托盘含重叠金属件、透明薄膜和无固定类别的异物。目标不是“零样本自动上线”，而是降低标注时间并保留审计轨迹。

<DlQuiz kind="任务边界" question="同一类别的两个相邻零件在语义分割输出中通常怎样表示？" :options="['必须有不同实例 ID', '可属于同一类别区域，不保证被分开', '都变成背景', '只保留一个']" :answer="1" explanation="语义分割只预测类别；实例身份是额外输出。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>为 semantic、instance、panoptic 三种输出分别写一个最小 JSON/张量合同，说明类别 ID、实例 ID、void 和重叠如何表示。</div>

## 1. Mask R-CNN：在检测器上增加并行 mask 分支

Mask R-CNN 基于 Faster R-CNN。每个正 RoI 同时进入分类、框回归与 mask 分支。mask 分支输出 $K\times m\times m$ 类条件 logits；训练时只对 GT 类别对应的一个通道计算二值 mask loss。分类竞争与像素 mask 因而解耦。

$$
L=L_{cls}+L_{box}+L_{mask}.
$$

RoIAlign 对实例 mask 关键，因为 RoI Pooling 的坐标量化会直接破坏边界对齐。预测的固定 $m\times m$ mask 最后缩放回检测框区域，而不是直接生成整图 mask。

```python
def select_class_mask(mask_logits, labels):
    # mask_logits shape: B x K x M x M, labels shape: B
    selected = []
    for batch_index, class_id in enumerate(labels):
        selected.append(mask_logits[batch_index][class_id])
    return selected

toy = [[[[0]], [[1]], [[2]]], [[[3]], [[4]], [[5]]]]
selected = select_class_mask(toy, [2, 0])
assert selected == [[[2]], [[3]]]
```

<DlQuiz kind="Mask head" question="Mask R-CNN 为何通常不在每个像素对 K 类做 softmax？" :options="['因为没有类别', '分类 head 已决定 RoI 类别，mask head 对每类独立二值预测可避免类间像素竞争', '因为 mask 没有空间', '因为只能预测背景']" :answer="1" explanation="论文采用每类独立 sigmoid，训练只监督 GT 类别通道。" />

## 2. 实例输出仍需装配

推理产生检测框、类别、分数和每个实例 mask。多个 mask 可以重叠；若下游要求互斥，需要按分数、面积、类别优先级或学习到的规则解决冲突。实例 AP 以 mask IoU 匹配，不应拿 box AP 代替。

小对象的 $28\times28$ RoI mask 可能足够，细长对象缩放进正方形网格后却易断裂。应按框长宽比、mask 面积和边界复杂度切片。

<DlQuiz kind="评价" question="box AP 很高是否保证 mask AP 同样高？" :options="['保证', '不保证，框可以定位正确而内部边界、孔洞或细结构错误', '只要类别数小就保证', 'Mask R-CNN 不评价 AP']" :answer="1" explanation="box IoU 与 mask IoU 约束不同，必须分别报告。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>两个实例框重叠但真实 mask 不重叠。类无关 box NMS 可能删掉什么？mask-aware 去重又需要哪些额外成本？</div>

## 3. 全景分割：每个像素一个互斥片段

thing 类有实例 ID，stuff 类通常按类别合并区域。常见编码把 `category_id` 与 `instance_id` 合成单个整数，同时在 sidecar JSON 保存 segment 元数据。ID divisor 必须足够大且版本固定。

全景质量 Panoptic Quality 为

$$
PQ=\underbrace{\frac{\sum_{(p,g)\in TP}\operatorname{IoU}(p,g)}{|TP|}}_{SQ}
\cdot
\underbrace{\frac{|TP|}{|TP|+\frac12|FP|+\frac12|FN|}}_{RQ}.
$$

PQ 同时惩罚没识别到的实例和已匹配实例的掩码质量。匹配阈值和 void/crowd 处理属于评价协议。

<PanopticAssemblyLab />

```python
def panoptic_id(category_id, instance_id, divisor=1000):
    if not 0 <= instance_id < divisor:
        raise ValueError("instance id exceeds divisor")
    return category_id * divisor + instance_id

segment_id = panoptic_id(category_id=17, instance_id=23)
assert segment_id == 17023
assert divmod(segment_id, 1000) == (17, 23)
```

<DlQuiz kind="全景合同" question="一个合法 panoptic 输出像素能否同时属于两个实例？" :options="['可以，任意重叠', '不可以，最终 panoptic map 要给每个有效像素唯一 segment ID', '只在 thing 类不可以', '只在训练时可以']" :answer="1" explanation="实例预测可以先重叠，但全景装配后的像素标签必须互斥。" />

## 4. SAM：把分割变成可提示接口

Segment Anything Model 包含三个接口：重型 image encoder 计算可复用图像 embedding；prompt encoder 把正/负点、框或 mask 编码成稀疏/稠密提示；轻量 mask decoder 结合 prompt token 与图像 embedding 输出候选 masks 和质量分数。

同一个点可能指物体局部、完整物体或其组成部分，因此提示存在语义歧义。SAM 可输出多个候选 mask，不代表它知道业务类别。它是 category-agnostic segmentation，不是开放词表分类器。

<PromptMaskStabilityLab />

<DlQuiz kind="能力边界" question="只给 SAM 一个正点后得到 mask，是否同时获得了可靠业务类别？" :options="['是，mask 自带任意类别', '否，SAM 主要提供可提示掩码；类别语义需要其他模型、提示或人工', '只有小目标自带类别', '只有负点时自带类别']" :answer="1" explanation="分割几何与业务语义是不同接口。" />

### 4.1 数据引擎比模型图更关键

SAM 的 SA-1B 数据由模型辅助人工标注和自动 mask 生成逐步扩展。阅读时要区分模型结构贡献、数据规模贡献和交互式数据引擎。把一个 promptable model 迁移到医疗或工业域时，重新建立高质量提示/修正循环往往比直接微调更重要。

## 5. 提示稳定性与标注系统设计

标注工具应保存原图 hash、模型版本、提示序列、每次候选、人工编辑、最终 mask 和耗时。只保存最终 mask 会丢失模型辅助是否真的节省时间的证据。

可测指标包括：首个提示 IoU、达到目标 IoU 的点击数、每实例秒数、候选翻转率、不同标注员一致性，以及按尺寸/细长/遮挡/透明度切片。模型自动生成的 mask 不应未经复核回流为训练真值，否则错误会自我强化。

```python
def clicks_to_target(iou_trace, target=0.9):
    for index, value in enumerate(iou_trace, start=1):
        if value >= target:
            return index
    return None

trace = [0.52, 0.73, 0.88, 0.93]
assert clicks_to_target(trace) == 4
assert clicks_to_target([0.3, 0.4]) is None
```

<div class="dl-stop"><strong>停下来推一推 3</strong>基础模型在 90% 样本上一点完成，但细长裂纹平均需 12 点。总体点击数会掩盖什么？怎样决定自动接受、人工复核和拒绝的门槛？</div>

## 6. 领域失效与适配

常见失效包括小目标、低对比、透明/反光、细长结构、医学成像通道、极高分辨率切片和与自然图像不同的对象层级。适配选择有：改提示策略、框前置检测、adapter/LoRA、领域 mask decoder、重新训练 encoder，或用 SAM 生成候选再训练专用轻模型。

任何适配都要防止 benchmark 泄漏：若用同一测试集交互选点再报告结果，用户策略已经看到了答案。应冻结提示生成规则，或由独立标注员在盲测集操作。

## 7. 推荐论文阅读

<DlPaperCard title="Mask R-CNN" meta="He et al. · 2017" priority="必读 · 实例分割" question="并行 mask 分支与 RoIAlign 分别解决什么问题，为什么 mask 预测与类别解耦？" reading="读三分支损失、RoIAlign 消融和 keypoint 泛化；核对固定 RoI mask 到原图的映射。" href="https://arxiv.org/pdf/1703.06870" />

<DlPaperCard title="Panoptic Segmentation" meta="Kirillov et al. · 2018" priority="必读 · 统一任务" question="如何在互斥输出中统一 things 与 stuff，PQ 为什么能分解为 SQ 与 RQ？" reading="精读任务定义、匹配唯一性、PQ 与错误分析；特别记录 void/crowd 规则。" href="https://arxiv.org/pdf/1801.00868" />

<DlPaperCard title="Segment Anything" meta="Kirillov et al. · 2023" priority="必读 · 基础分割模型" question="promptable task、模型接口与数据引擎怎样共同产生零样本迁移能力？" reading="分开读 task/model/data engine 三部分，再看歧义 mask、数据规模和领域外限制。" href="https://arxiv.org/pdf/2304.02643" />

## 8. 练习与门禁

1. 实现类条件 mask 选择与粘贴，测试越界框、零面积和长条 RoI。
2. 为全景 ID 编码写碰撞、divisor 越界和 JSON/map 一致性测试。
3. 计算一个小例子的 SQ、RQ、PQ，并解释 FP/FN 各自怎样影响。
4. 建立提示序列日志，比较点、框、点+框达到 IoU 0.9 的交互成本。
5. 按细长、低对比、重叠、透明切片评估 SAM，禁止只报告总体 mIoU。
6. 设计模型自动 mask 回流的数据治理：抽检、版本、撤回和污染检测。

## 本章小结

- 实例分割在像素类别之外增加对象身份；Mask R-CNN 以检测 RoI 为单位预测类条件 mask。
- 全景输出要求像素互斥，PQ 将匹配后的分割质量与识别质量相乘。
- SAM 提供图像 embedding、提示编码和候选 mask 接口，不自动提供业务类别。
- 可提示模型的价值要用点击数、编辑时间、稳定性和切片衡量。
- 自动生成 mask 回流必须有人类复核与版本治理，避免错误闭环。

<DlLessonProgress lesson-id="09" next-href="/courses/deep-learning/10-vision-transformer" next-label="下一章：Vision Transformer" />
