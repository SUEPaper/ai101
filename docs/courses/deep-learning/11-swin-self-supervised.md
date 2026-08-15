---
course: deep-learning
edition: industrial
title: Swin Transformer 与视觉自监督学习
description: 从窗口注意力、移位连通、patch merging 与层级特征出发，比较 SimCLR、DINO、MAE 和 CLIP 的视觉预训练目标与迁移证据。
outline: [2, 3]
---

<DlLessonHero number="11" module="计算机视觉" title="Swin Transformer 与视觉自监督学习" question="若全局注意力太贵，局部窗口怎样交换信息；若标签太贵，预训练目标又迫使表示保留什么？" duration="建议 300–390 分钟" :outcomes="['推导窗口注意力相对全局注意力的复杂度', '解释 shifted window、attention mask 与 patch merging', '比较对比学习、自蒸馏、掩码重建和图文对齐目标', '建立 linear probe、fine-tune 与 dense transfer 的表示评价矩阵']" />

本章把两个问题连起来：Swin 用局部窗口和层级下采样让 Transformer 适配高分辨率密集视觉；自监督学习用数据自身、teacher 或跨模态配对代替人工类别。结构决定信息怎样流，目标决定什么信息值得保留。

## 贯穿案例：千万张未标注产线图像预训练

企业有 1000 万张无标签图，来自 20 条产线和多个相机；只有 2 万张分类标签、3000 张检测框和 800 张像素 mask。我们要选预训练目标，并验证迁移到分类、检测、分割和跨产线异常发现，而不是只做一个线性探测分数。

<DlQuiz kind="目标边界" question="自监督学习是否意味着没有任何监督信号？" :options="['是，loss 不存在', '否，监督目标由数据变换、掩码、teacher 或配对关系构造，而非人工类别', '只有文本任务有监督', '只在推理时有监督']" :answer="1" explanation="自监督仍有明确预测目标；关键是目标从数据结构自动产生。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>同一零件的旋转是否应被视为“同一实例视图”？若方向决定装配正确性，SimCLR 的旋转增强会删除什么信息？</div>

## 1. Swin 的窗口复杂度

全局 MSA 对 $h\times w$ token 的 pairwise 成本为 $O((hw)^2C)$。将 token 划分为 $M\times M$ 不重叠窗口，每窗局部 attention，总成本约

$$
\frac{hw}{M^2}\cdot M^4C=hwM^2C,
$$

对固定窗口大小随图像面积线性增长。代价是单层中不同窗口互不通信。

```python
def attention_pairs(height, width, window):
    tokens = height * width
    global_pairs = tokens * tokens
    windows = (height // window) * (width // window)
    local_pairs = windows * (window * window) ** 2
    return global_pairs, local_pairs

global_pairs, local_pairs = attention_pairs(56, 56, 7)
assert global_pairs / local_pairs == 64
```

<DlQuiz kind="复杂度" question="固定窗口边长 M 时，窗口 attention 的 pairwise 成本随 token 总数 N 怎样增长？" :options="['约 O(N²)', '约 O(NM²)，对固定 M 近似线性', '与 N 无关', '约 O(log N)']" :answer="1" explanation="窗口数与 N 成正比，每窗有 M² token 和 M⁴ pair。" />

## 2. Shifted window 建立跨窗连接

Swin 交替使用 regular window MSA（W-MSA）与平移半个窗口的 shifted window MSA（SW-MSA）。第二个 block 的分区边界错开，使上一层不同窗口的 token 在新窗口相遇。

高效实现常用 cyclic shift 保持规则内存，再用 attention mask 阻止环绕后本不相邻的 token 互相注意。若 mask 方向或区域 ID 错，模型会产生图像左右/上下边界泄漏。

<ShiftedWindowLab />

```python
import numpy as np

def cyclic_shift(tokens, shift):
    return np.roll(tokens, shift=(-shift, -shift), axis=(0, 1))

grid = np.arange(16).reshape(4, 4)
shifted = cyclic_shift(grid, 2)
restored = np.roll(shifted, shift=(2, 2), axis=(0, 1))
assert np.array_equal(restored, grid)
```

<DlQuiz kind="连通性" question="为什么只使用固定不移位窗口会限制表示？" :options="['窗口内不能计算 attention', '不同窗口 token 没有跨边界通道，堆很多层仍按同一分区隔离', '参数会自动变为零', '位置编码失效']" :answer="1" explanation="交替移位让边界改变，跨窗信息才能逐层传播。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>若窗口 7×7、feature map 不是 7 的倍数，padding token 如何进入 attention mask？池化和裁剪时又怎样排除它们？</div>

## 3. Patch merging 与层级视觉骨干

每到新 stage，Swin 将相邻 $2\times2$ token 拼接，空间边长减半、通道经线性层调整。它生成类似 CNN 的多尺度 $C_2$–$C_5$ 特征，易接 FPN、检测和分割 head。

若输入为 $B\times H\times W\times C$，拼接后为 $B\times H/2\times W/2\times4C$，再投影到常用的 $2C$。奇数尺寸需要 pad，输出元数据必须记录有效区域。

<DlQuiz kind="层级 shape" question="2×2 patch merging 在投影前怎样改变 shape？" :options="['空间不变，通道不变', 'H/W 各减半，四个邻居在通道轴拼接为 4C', 'H/W 各翻倍', '只删除 CLS']" :answer="1" explanation="先重排四个相邻 token，再做 LayerNorm/linear projection。" />

## 4. 相对位置偏置

窗口内 attention score 可加由相对位移索引得到的 bias。相对位置比绝对表更贴合局部窗口共享，但窗口尺寸变化仍需插值或重新初始化。位置 bias 是每个 head 的可学习先验，不等于完整几何理解。

## 5. 对比学习：增强定义不变性

SimCLR 对同一图像做两次随机增强，得到正对；batch 中其他图像为负对。NT-Xent loss 让正对相似度高于负对：

$$
\ell_{i,j}=-\log\frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}
{\sum_{k\ne i}\exp(\operatorname{sim}(z_i,z_k)/\tau)}.
$$

温度 $\tau$ 控制分布尖锐度，大 batch 提供更多负样本。最关键的设计是增强：随机裁剪、颜色变化被声明为“身份无关”。若颜色或方向是缺陷语义，错误不变性会伤害下游。

```python
import numpy as np

def normalized(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)

anchor = normalized([1.0, 0.2, 0.0])
positive = normalized([0.9, 0.3, 0.0])
negative = normalized([-0.2, 0.1, 0.9])
assert anchor @ positive > anchor @ negative
```

<DlQuiz kind="增强语义" question="SimCLR 中把两种增强视图当正对，隐含了什么声明？" :options="['两图必须逐像素相同', '下游希望表示对这些增强保持语义一致', '所有颜色都应删除', '负样本没有作用']" :answer="1" explanation="增强定义了模型应忽略或保持的变化，必须与领域标签语义一致。" />

## 6. DINO：无标签自蒸馏

DINO 用 student 匹配 momentum teacher 的输出分布。teacher 接受全局 crop，student 接受全局和局部 crop；centering 与 sharpening 防止所有样本坍缩为相同分布。teacher 参数由 student 指数滑动平均更新，不通过梯度直接训练。

“没有负样本”不等于不会坍缩。stop-gradient、teacher momentum、温度和 centering 共同建立非平凡目标。DINO 的 attention 常呈现对象区域，但仍不能把可视化当作像素真值。

## 7. MAE：只编码少量可见 patch

MAE 随机 mask 高比例 patch，encoder 只处理可见 token，轻量 decoder 接收 mask token 重建像素。图像空间冗余高，因此 75% mask 仍可恢复宏观结构；高 mask ratio 也显著减少 encoder 预训练计算。

像素重建偏好颜色、纹理和局部统计，不一定直接优化语义分离。MAE 往往 fine-tune 很强而 linear probe 相对不同，这提醒我们评价协议会改变“表示好坏”的结论。

<VisionPretrainLab />

<DlQuiz kind="目标比较" question="MAE 为什么能在高 mask ratio 下仍有效训练？" :options="['因为 decoder 看到了真值标签', '自然图像具有空间冗余，encoder 只处理可见 token，decoder 学习重建被遮挡内容', '因为不计算 loss', '因为每个 patch 独立无关']" :answer="1" explanation="高遮挡让任务非平凡并降低 encoder 计算，轻量 decoder 承担像素重建。" />

## 8. CLIP：用语言扩大监督空间

CLIP 对 batch 内图像和文本编码，优化正确图文对的双向对比目标。文本让视觉表示连接开放词汇语义，支持 zero-shot 分类；但网页语料偏见、文本泄漏和提示词敏感性进入系统。

零样本评价必须冻结 prompt 模板或预注册 prompt ensemble，不能在测试集上反复调词。工业图像若缺少对应自然语言描述，通用 CLIP 可能更擅长宏观物体而非细粒度缺陷。

<div class="dl-stop"><strong>停下来推一推 3</strong>选择 SimCLR、DINO、MAE 或 CLIP 时，为分类、检测、分割、跨域检索分别写一个首要评价，并说明为什么不能只用 linear probe。</div>

## 9. 预训练评价矩阵

至少包含：冻结 linear probe 衡量线性可分性；k-NN/检索衡量局部几何；全量 fine-tune 衡量适配上限；低样本曲线衡量数据效率；检测/分割迁移衡量空间信息；领域外切片衡量不变性；训练计算和数据治理衡量成本。

若下游最终是 Swin 检测器，预训练对照应固定下游 head、schedule、增强和随机种子。不同目标若使用不同数据量或分辨率，要报告总图像/像素/token 和计算量。

## 10. 推荐论文阅读

<DlPaperCard title="Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" meta="Liu et al. · 2021" priority="必读 · 层级 Transformer" question="窗口与移位怎样把复杂度变为线性并建立跨窗连接，层级特征如何服务密集预测？" reading="精读 W-MSA/SW-MSA 图、复杂度公式、patch merging 与检测分割迁移。" href="https://arxiv.org/pdf/2103.14030" />

<DlPaperCard title="Emerging Properties in Self-Supervised Vision Transformers" meta="Caron et al. · 2021" priority="必读 · DINO" question="无负样本自蒸馏如何避免坍缩，multi-crop 与 teacher 温度分别做什么？" reading="读 loss、centering/sharpening、teacher momentum 和 attention 可视化的证据边界。" href="https://arxiv.org/pdf/2104.14294" />

<DlPaperCard title="Masked Autoencoders Are Scalable Vision Learners" meta="He et al. · 2021" priority="必读 · 掩码预训练" question="非对称 encoder-decoder 与高 mask ratio 如何同时改变任务难度和计算？" reading="精读架构、mask ratio、decoder 消融和迁移协议。" href="https://arxiv.org/pdf/2111.06377" />

<DlPaperCard title="Learning Transferable Visual Models From Natural Language Supervision" meta="Radford et al. · 2021" priority="选读 · 图文对齐" question="自然语言监督怎样支持 zero-shot transfer，数据与 prompt 又引入哪些偏差？" reading="读对比目标、数据规模、zero-shot prompt 和分布偏移结果。" href="https://arxiv.org/pdf/2103.00020" />

## 11. 练习与门禁

1. 为 W-MSA 与 SW-MSA 写区域 ID 和 mask 单元测试，检测 cyclic wrap 泄漏。
2. 计算四个 stage 的 token、窗口数、通道、pairwise cost 和激活峰值。
3. 构造方向敏感任务，对比允许/禁止旋转增强的 SimCLR 迁移结果。
4. 监控 DINO teacher/student entropy，注入错误 stop-gradient 并识别坍缩。
5. 扫描 MAE mask ratio，报告预训练吞吐、linear probe 和 full fine-tune。
6. 冻结 CLIP prompt 集，在全新产线测试 zero-shot 与检索，禁止测试集调词。
7. 用统一下游 schedule 比较四种目标在分类/检测/分割的 Pareto 前沿。

## 本章小结

- Swin 用固定局部窗口把高分辨率 attention 变为近似线性成本。
- 交替 shifted window 建立跨窗通信，cyclic shift 必须配正确 mask。
- patch merging 提供检测和分割需要的层级多尺度特征。
- SimCLR 的增强定义不变性，DINO 用 teacher-student 防坍缩，MAE 重建高比例 mask，CLIP 对齐图文语义。
- 预训练目标保留的信息不同，必须用 linear probe、fine-tune、密集迁移和领域外切片共同评价。

<DlLessonProgress lesson-id="11" />
