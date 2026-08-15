---
course: deep-learning
edition: industrial
title: Vision Transformer（ViT）与 DeiT
description: 从 patch embedding、位置编码、CLS token、全局注意力预算与训练配方出发，解释 ViT 的数据规模、归纳偏置和蒸馏接口。
outline: [2, 3]
---

<DlLessonHero number="10" module="计算机视觉" title="Vision Transformer（ViT）与 DeiT" question="把图像切成 token 后，Transformer 获得了全局关系，但在 patch 边界、二维位置和数据效率上失去了什么？" duration="建议 280–370 分钟" :outcomes="['写出 patch embedding、CLS 与位置编码的完整 shape', '计算分辨率和 patch size 对注意力成本的平方影响', '比较卷积局部偏置与 ViT 的数据/正则依赖', '解释 DeiT 蒸馏 token 和现代 ViT 训练 recipe']" />

本章假设已经学过自注意力的 Q/K/V 与缩放点积公式，不再从序列 Transformer 基础推导。重点是图像怎样被转换成 token、二维几何怎样进入模型、全局注意力怎样改写计算预算，以及为什么原始 ViT 需要大规模预训练而 DeiT 能在 ImageNet-1K 上有效训练。

## 贯穿案例：高分辨率晶圆缺陷分类

输入是 1024×1024 图像，微小缺陷宽度约 6–20 像素。候选是 ViT-B/16。若直接用 16 像素 patch，每个微小缺陷可能只影响少数 token；若改为 8 像素 patch，token 数从 $64^2$ 增至 $128^2$，全局注意力矩阵约扩大 16 倍。

<DlQuiz kind="预算直觉" question="输入边长不变，patch size 从 16 减为 8，token 数和全局注意力矩阵元素数近似怎样变化？" :options="['token 2 倍，矩阵 4 倍', 'token 4 倍，矩阵 16 倍', '都 2 倍', '都不变']" :answer="1" explanation="二维每边 patch 数翻倍，总 token 约四倍；N×N 注意力矩阵约十六倍。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>对 1024 输入分别计算 P=32、16、8 的 token 数与单层每头 attention score 数。哪一个在你的显存上可行？</div>

## 1. Patch embedding 是带步长卷积

把 $H\times W\times C$ 图像分成不重叠 $P\times P$ patch，每个 patch 展平为 $P^2C$ 向量，再线性投影到 $D$ 维：

$$
N=\frac HP\frac WP,\qquad
X_p\in\mathbb R^{B\times N\times(P^2C)},\qquad
Z=X_pE,\ E\in\mathbb R^{P^2C\times D}.
$$

它等价于 kernel=$P$、stride=$P$、输出通道 $D$ 的卷积。等价性提醒我们：patch 化不是“没有卷积”，而是一种不重叠、早期下采样很强的卷积 stem。

```python
def vit_shapes(batch, height, width, channels, patch, width_d):
    if height % patch or width % patch:
        raise ValueError("image must be divisible by patch size")
    tokens = (height // patch) * (width // patch)
    flattened = (batch, tokens, patch * patch * channels)
    embedded = (batch, tokens + 1, width_d)  # plus CLS
    return flattened, embedded

flat, embedded = vit_shapes(2, 224, 224, 3, 16, 768)
assert flat == (2, 196, 768)
assert embedded == (2, 197, 768)
```

<PatchTokenLab />

<DlQuiz kind="Patch 接口" question="ViT-B/16 中“16”通常表示什么？" :options="['16 个类别', 'patch 边长 16 像素', '16 个 attention head', '16 层卷积']" :answer="1" explanation="输入通常被分成 16×16 的不重叠 patch；具体 hidden size 和层数由 B 配置决定。" />

### 1.1 Patch 边界与别名

原始 patch projection 没有重叠，细线跨边界时证据被分散。可用更小 patch、overlapping stem、卷积 stem 或多尺度结构改善，但都改变预训练权重兼容与成本。输入 resize 也应先做抗混叠审计。

## 2. CLS token 与位置编码

ViT 在 patch token 前加入可学习的 `[CLS]` token，最后用其表示分类。所有 token 再加位置编码：

$$
Z_0=[x_{cls};x_p^1E;\dots;x_p^NE]+E_{pos}.
$$

没有位置编码，自注意力对 token 排列是置换等变的，模型无法区分左上与右下。原始 ViT 使用一维可学习绝对位置表，但其顺序来自二维网格展平。

当 fine-tune 分辨率变化时，位置表通常先还原成二维网格，再做二维插值；CLS 位置单独保留。直接把整条一维序列插值会混淆二维邻域。

```python
import numpy as np

def split_cls_grid(position, grid_side):
    cls = position[:, :1, :]
    grid = position[:, 1:, :]
    assert grid.shape[1] == grid_side * grid_side
    return cls, grid.reshape(position.shape[0], grid_side, grid_side, -1)

position = np.zeros((1, 197, 32), dtype=np.float32)
cls, grid = split_cls_grid(position, 14)
assert cls.shape == (1, 1, 32)
assert grid.shape == (1, 14, 14, 32)
```

<DlQuiz kind="位置接口" question="fine-tune 到更高分辨率时，为什么常对 patch 位置编码做二维插值？" :options="['因为类别数增加', 'token 网格尺寸改变，需要保留二维空间邻近关系并适配新位置数', '为了删除 CLS', '因为注意力不需要位置']" :answer="1" explanation="输入网格改变会使位置表长度不匹配；二维插值与图像网格结构一致。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>224/P16 的位置表迁移到 384/P16 时，旧新网格分别多大？CLS token 应不应该参与二维插值？</div>

## 3. 全局注意力的预算

每层多头注意力主要有两类成本：QKV/输出投影约 $O(ND^2)$，attention score 与加权约 $O(N^2D)$。当 $N\ll D$ 时投影可能占主导；高分辨率使 $N$ 增大后，平方项迅速成为瓶颈。

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V.
$$

仅报告 $N^2$ 不足以代表总 FLOPs，仍应分开统计线性层、MLP、激活与 attention matrix 内存。

```python
def attention_terms(tokens, width, layers):
    projections = layers * 4 * tokens * width * width
    pairwise = layers * 2 * tokens * tokens * width
    return projections, pairwise

projection, pairwise = attention_terms(197, 768, 12)
assert projection > pairwise
_, pairwise_large = attention_terms(4097, 768, 12)
assert pairwise_large > pairwise
```

<DlQuiz kind="复杂度" question="在高分辨率 ViT 中，attention score 内存为何容易成为瓶颈？" :options="['它只随 D 增长', '每头需表示 N×N token 关系，随 token 数平方增长', '它与输入无关', '因为 CLS 数量平方增长']" :answer="1" explanation="空间分辨率提高会快速增加 N，训练还需保存中间量用于反向。" />

## 4. 归纳偏置与数据规模

卷积显式编码局部连接与平移等变，参数在空间共享。原始 ViT 的全局 attention 和位置表假设更弱，数据少时较难学到稳健局部结构；大规模预训练下，更弱偏置提供更大表示自由度。

<VitInductiveBiasLab />

这个实验是概念响应面，不是 benchmark。真实比较必须固定参数/FLOPs、增强、优化器、训练 epoch、预训练数据和评价协议。只把 ResNet 的旧 recipe 与 ViT 的强 recipe 对比，会把训练配方误归因于架构。

## 5. ViT 的训练 recipe

ViT 常用 AdamW、warmup、cosine decay、较强增强、label smoothing、stochastic depth、Mixup/CutMix 和大规模预训练。LayerNorm 放置、pre-norm 与 residual scale 影响深层稳定性。

预训练/微调时要检查：输入归一化；位置插值；分类头是否重新初始化；weight decay 是否排除 bias/LayerNorm/position token；分层学习率衰减是否启用；drop path 在评估时是否关闭。

<DlQuiz kind="公平实验" question="比较 ResNet 与 ViT 时，下面哪项最会破坏归因？" :options="['记录参数量', '给 ViT 使用强增强和长训练，却让 ResNet 沿用短旧配方', '固定数据切分', '报告多随机种子']" :answer="1" explanation="训练 recipe 与架构同时改变，差异无法只归于架构。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>ViT 训练 loss 很低但小目标召回差。列出 patch size、输入 resize、位置编码、标签偏差和增强五条竞争假设及证据。</div>

## 6. DeiT：数据高效训练与蒸馏 token

DeiT 展示了在 ImageNet-1K 上通过强 recipe 训练 ViT，并引入 distillation token。teacher 的预测监督 distill token，分类 token 接受真值监督；推理时组合二者。蒸馏不是把 CNN 结构复制进 ViT，而是通过 teacher 输出传递归纳偏置与决策信息。

应比较 hard/soft distillation、teacher 类型、训练时长与无蒸馏强基线。若 teacher 使用额外数据或更大输入，必须计入资源与数据边界。

## 7. 可解释性边界

attention map 不是因果解释。CLS 到 patch 的 attention 可视化能显示路由权重，但多层残差、MLP 和 value 内容都会影响输出。attention rollout、gradient attribution 和遮挡试验应交叉验证，并用受控背景替换检验伪相关。

## 8. 推荐论文阅读

<DlPaperCard title="An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" meta="Dosovitskiy et al. · 2020" priority="必读 · ViT 起点" question="纯 Transformer 在什么数据规模和训练条件下超越卷积基线，patch/位置接口怎样定义？" reading="精读模型公式、预训练规模对照、数据量曲线和 fine-tune 分辨率处理。" href="https://arxiv.org/pdf/2010.11929" />

<DlPaperCard title="Training Data-Efficient Image Transformers & Distillation through Attention" meta="Touvron et al. · 2020" priority="必读 · 训练配方" question="ImageNet-1K 条件下，强 recipe 和 distillation token 分别贡献什么？" reading="读 augmentation/regularization 配方、teacher 对照和 distill token 消融。" href="https://arxiv.org/pdf/2012.12877" />

<DlPaperCard title="Masked Autoencoders Are Scalable Vision Learners" meta="He et al. · 2021" priority="选读 · 预训练桥梁" question="高 mask ratio 为什么适合冗余图像，非对称 encoder-decoder 如何降低预训练成本？" reading="重点读 mask ratio、decoder 规模和 fine-tune/linear probe；第 11 章放入目标比较。" href="https://arxiv.org/pdf/2111.06377" />

## 9. 练习与门禁

1. 写 patchify/unpatchify 往返测试，覆盖 RGB/灰度与不可整除尺寸。
2. 生成分辨率、patch size、token、注意力内存的预算表。
3. 实现位置表二维插值的 shape 测试，确保 CLS 不参与网格插值。
4. 固定现代 recipe 比较 ResNet 与 ViT 的小/中/大数据曲线。
5. 按目标像素面积比较 P=32/16/8，记录精度、显存和延迟。
6. 用 attention、gradient 与遮挡三种方法检查同一失败样本，记录不一致。

## 本章小结

- patch embedding 是强下采样的带步长投影；patch size 同时决定细节和 token 预算。
- ViT 需要位置接口表达二维几何，分辨率迁移时要二维插值位置表。
- 全局 attention 的平方项在高分辨率下迅速增长，但投影和 MLP 也要纳入账本。
- 卷积的局部偏置更省样本；ViT 更依赖规模、增强、正则和预训练。
- DeiT 说明训练配方与蒸馏能改变数据效率，公平比较必须控制 recipe。

<DlLessonProgress lesson-id="10" next-href="/courses/deep-learning/11-swin-self-supervised" next-label="下一章：Swin 与视觉自监督" />
