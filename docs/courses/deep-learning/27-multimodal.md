---
course: deep-learning
edition: industrial
title: 多模态大模型与统一表示
description: 从 CLIP 对齐、视觉 encoder 与 projector/cross-attention 进入视觉 token、指令微调、OCR/grounding、文本捷径和统一生成。
outline: [2, 3]
---

<DlLessonHero number="27" module="大语言模型" title="多模态大模型与统一表示" question="图像怎样被编译成 LLM 可消费的 token；模型又是否真的依据视觉证据，而不是沿文本先验作答？" duration="建议 330–430 分钟" :outcomes="['解释 CLIP 式对比对齐与零样本分类接口', '比较 projector、cross-attention/resampler 与统一 token 序列', '计算分辨率、patch、crop、视觉 token 和延迟', '设计 OCR、spatial grounding、模态冲突和图像消融评价', '区分理解、生成、对齐与统一多模态建模的边界']" />

多模态模型不等于在语言模型前接一个图片上传框。视觉 encoder、connector、token budget、预训练配对和指令数据共同决定模型能看见哪些细节；流畅描述可能来自语言先验和数据捷径。

## 贯穿案例：设备铭牌与故障照片诊断

输入包含整机照片、铭牌小字和用户描述。任务是识别型号、定位漏油区域、引用手册并生成维修建议。我们分开评价 OCR、对象/空间 grounding、证据冲突、建议正确性和高风险拒答。

<DlQuiz kind="先判断" question="多模态模型能描述图片，是否证明其空间定位准确？" :options="['证明', '不证明；描述可能依赖全局语义/文本先验，需坐标、遮挡和反事实测试', '只要输出长就证明', 'CLIP loss 会保证像素级定位']" :answer="1" explanation="任务接口与粒度不同。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>将“读铭牌型号”“框出漏油点”“解释原因”“生成步骤”拆成四个标签与指标；哪些需要像素/坐标监督？</div>

## 1. CLIP：双编码对齐与零样本接口

CLIP 分别编码图像与文本，在 batch 内用对称对比损失拉近配对、推远非配对。零样本分类把类别写成 prompt，比较图像与类别文本 embedding。它学习全局对齐，不天然给出细粒度 OCR、计数或定位。

```python
import numpy as np

image = np.array([.8,.6]); texts = np.array([[.75,.65],[-.8,.2]])
image = image/np.linalg.norm(image); texts = texts/np.linalg.norm(texts,axis=1,keepdims=True)
scores = texts @ image
assert scores.argmax() == 0 and scores[0] > scores[1]
```

<DlQuiz kind="CLIP" question="CLIP batch 中 in-batch negative 一定是语义无关吗？" :options="['一定', '不一定，可能存在多个正确描述或相近图像形成 false negatives', '只有一张图', 'negative 是反义词']" :answer="1" explanation="数据重复与多正例会影响对比目标。" />

## 2. 将视觉接入 LLM

常见路线：用线性/MLP projector 将视觉 patch embedding 投到语言维度；用 Q-Former/resampler 压缩视觉 token；或在语言层插入 gated cross-attention（Flamingo）。LLaVA 等以视觉 encoder + projector + 指令微调建立对话接口。冻结/解冻哪些模块决定数据与计算需求。

<VisualTokenBudgetLab />

```python
import math

def visual_tokens(height, width, patch, crops=1):
    return math.ceil(height/patch)*math.ceil(width/patch)*crops

assert visual_tokens(336,336,14) == 576
assert visual_tokens(672,672,14) == 4*576
```

<DlQuiz kind="视觉预算" question="图像边长翻倍、patch 不变时，patch token 约怎样变化？" :options="['约 4 倍', '约 2 倍', '不变', '减半']" :answer="0" explanation="二维网格两轴都翻倍。" />

高分辨率、multi-crop、dynamic tiling 改善小字潜力，但视觉 token 与文本共享 context/KV，TTFT 与显存上升。Resampler 固定输出长度，可能丢小目标。应报告原图、预处理、crop 坐标和缩放，不只报“支持 4K”。

## 3. 多模态预训练与指令微调

数据从 image-caption 对扩展到交错图文、OCR、图表、区域指代和多轮问答。网页 alt text 有噪声和社会偏置；合成 QA 继承 teacher 错误。阶段常包括对齐 connector、联合预训练、视觉指令微调和偏好优化，每阶段的数据/冻结策略需分开记录。

<DlQuiz kind="数据" question="仅用图像 caption 对训练，最可能缺少哪类监督？" :options="['精确空间关系、OCR、细粒度计数与交互指令', '全局语义', '文本 token', '图像文件']" :answer="0" explanation="caption 常描述显著对象而忽略细节。" />

## 4. 文本捷径与模态冲突

<ModalityConflictLab />

若问题“这只香蕉是什么颜色？”训练中几乎总答黄色，模型可能忽略反事实蓝香蕉。测试应交换图片、遮掉图片、改写问题、加入矛盾 caption，并测答案是否随视觉证据变化。OCR 应按字符/字段准确率和尺寸切片；定位按 IoU/pointing；图表还要检查轴与单位。

```python
pred_with_image = "X3"
pred_without_image = "X2"
gold = "X3"
grounding_gain = int(pred_with_image==gold)-int(pred_without_image==gold)
assert grounding_gain == 1
```

<DlQuiz kind="消融" question="遮掉图片后答案完全不变，最合理的第一解释是什么？" :options="['任务可能由文本先验解决或模型忽略视觉，需要更多反事实检查', '视觉理解完美', '图像 token 更多', '证明没有偏置']" :answer="0" explanation="模态消融是 grounding 的基本证据。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>设计同问题配原图、反事实图、纯色图、无图四联测试；预期答案和拒答怎样变化？</div>

## 5. 统一表示与多模态生成

统一模型可把图像/音频离散为 token，与文本共享自回归序列；或用 diffusion/flow decoder 从语言/视觉表示生成连续模态。统一接口简化任务，但词表、序列长度、不同模态损失尺度和生成安全变复杂。理解 benchmark 高分不等于图像生成质量，反之亦然。

<DlQuiz kind="统一模型" question="把图像量化为离散 token 后最直接的新成本是什么？" :options="['视觉序列可能很长，码本误差与语言 token 竞争容量', '不再需要数据', '自动消除版权风险', '不需评测']" :answer="0" explanation="离散化交换统一接口与码本/序列成本。" />

## 6. 安全与部署

图片可能含隐私、恶意文字提示、二维码和隐写信息；OCR 结果是未信任数据。模型可能泄露人脸/地址、做高风险医疗/工业判断，或在裁剪中遗漏警告标志。应做内容过滤、最小分辨率保存、EXIF 清理、租户隔离、视觉 prompt injection 和人工复核。

<div class="dl-stop"><strong>停下来推一推 3</strong>为铭牌诊断建立分辨率×字号×旋转×反光×语言切片，同时记录视觉 token、TTFT、字段准确率和危险建议率。</div>

## 7. 推荐论文阅读

<DlPaperCard title="Learning Transferable Visual Models From Natural Language Supervision" meta="Radford et al. · 2021" priority="必读 · CLIP" question="双编码对比目标、prompt ensemble 与零样本分类怎样连接？" reading="精读数据规模、InfoNCE 目标、prompt 和分布偏移；核对偏置/失败。" href="https://arxiv.org/pdf/2103.00020" />

<DlPaperCard title="Flamingo: a Visual Language Model for Few-Shot Learning" meta="Alayrac et al. · 2022" priority="必读 · Cross-attention" question="Perceiver Resampler 与 gated cross-attention 如何把交错图文接入冻结 LLM？" reading="读架构、训练数据混合、few-shot 接口与视觉 token 压缩。" href="https://arxiv.org/pdf/2204.14198" />

<DlPaperCard title="Visual Instruction Tuning" meta="Liu et al. · 2023" priority="必读 · LLaVA" question="视觉指令数据怎样合成，projector/冻结策略与对话评价如何设计？" reading="读两阶段训练、数据构造和 GPT judge 限制；复核视觉 grounding 证据。" href="https://arxiv.org/pdf/2304.08485" />

## 8. 练习与门禁

实现 CLIP cosine；计算视觉 token；比较 projector/resampler；做图像消融/反事实；OCR/定位切片；注入视觉 prompt；报告质量—token—TTFT—安全。

## 本章小结

- 多模态能力来自视觉 encoder、connector、数据和任务监督，不由“能上传图片”证明。
- 分辨率与 patch 决定视觉 token、上下文和延迟；压缩会丢细节。
- 必须用遮挡、反事实、空间/OCR 指标和模态冲突验证 grounding，并把图片内容视为未信任输入。

<DlLessonProgress lesson-id="27" next-href="/courses/deep-learning/28-evaluation-safety" next-label="下一章：评估、可靠性与安全" />
