---
course: deep-learning
edition: industrial
title: FCN、U-Net 与语义分割
description: 从全卷积预测、转置卷积与插值、跳跃连接、空洞卷积、损失和 mIoU 出发，建立像素标签、边界与部署评价合同。
outline: [2, 3]
---

<DlLessonHero number="08" module="计算机视觉" title="FCN、U-Net 与语义分割" question="把分类器改成逐像素输出后，怎样同时保留深层语义、细小边界和可审计的像素坐标？" duration="建议 280–360 分钟" :outcomes="['写出 FCN、U-Net 编解码器与 skip connection 的 shape 合同', '比较转置卷积、插值后卷积与空洞卷积的边界行为', '正确处理 ignore label、类别不平衡和空类', '区分像素准确率、IoU、Dice 与边界指标的证据范围']" />

语义分割为每个像素预测类别，不区分同类的不同实例。输出 logits 通常为 $B\times C\times H\times W$，标签为 $B\times H\times W$ 的整数类别 ID。最常见的严重错误不是网络层写错，而是 resize、插值、ignore label 或颜色映射破坏了标签。

## 贯穿案例：焊缝缺陷像素分割

相机输出 1536×512 灰度图，需要分出背景、正常焊缝、气孔和裂纹。裂纹只占 0.3% 像素，宽度可能 2–4 像素。总体 pixel accuracy 接近 99% 仍可能完全漏掉裂纹，因此评价必须包含每类 IoU、Dice、边界误差和连通缺陷召回。

<DlQuiz kind="标签合同" question="缩放整数语义 mask 时应优先使用什么插值？" :options="['双线性，让边界更平滑', '最近邻，避免产生不存在的中间类别 ID', '任意插值都相同', '只用傅里叶插值']" :answer="1" explanation="类别 ID 不是连续物理量；双线性会混出非法标签。软概率图才可使用连续插值。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>一个 crop 中没有裂纹。Dice loss 的分子分母、类别平均和梯度应该怎样定义？写清平滑项和空类策略。</div>

## 1. FCN：把分类头变成空间预测头

分类网络的全连接层可视为覆盖整张特征图的卷积。FCN 删除固定尺寸全连接接口，用 $1\times1$ 卷积产生每个低分辨率位置的类别 logits，再上采样回输入大小。网络因而接受可变空间尺寸，并保留粗位置。

若骨干总 stride 为 32，输出边界很粗。FCN-32s 直接上采样；FCN-16s/8s 把浅层特征融合回来。关键不是“skip 越多越好”，而是深浅特征的空间对齐、通道投影与语义差距。

```python
def segmentation_shapes(batch, classes, height, width, output_stride):
    assert height % output_stride == 0 and width % output_stride == 0
    low = (batch, classes, height // output_stride, width // output_stride)
    full = (batch, classes, height, width)
    target = (batch, height, width)
    return low, full, target

low, full, target = segmentation_shapes(2, 4, 512, 1536, 32)
assert low == (2, 4, 16, 48)
assert target == (2, 512, 1536)
```

<DlQuiz kind="输出接口" question="语义分割的 softmax 通常沿哪个维度计算？" :options="['batch 维', '类别 C 维，对每个像素形成类别分布', '高度维', '所有像素一起']" :answer="1" explanation="每个像素位置有 C 个 logits，softmax 在类别轴归一化。" />

## 2. 上采样不是一个无害细节

双线性插值无参数且稳定，之后可接卷积细化。转置卷积可学习上采样，但 kernel/stride 重叠不均时容易产生 checkerboard artifact。可分离的 resize-conv 通常更易审计：先固定插值，再做普通卷积。

输出尺寸公式必须测试。对于转置卷积，一维输出为

$$
L_{out}=(L_{in}-1)s-2p+d(k-1)+o+1,
$$

其中 $o$ 是 output padding。靠手调 `output_padding` 对齐某个样本会在另一尺寸失效，应从 shape 合同推导。

```python
def transposed_output(length, kernel, stride, padding, dilation=1, output_padding=0):
    return ((length - 1) * stride - 2 * padding
            + dilation * (kernel - 1) + output_padding + 1)

assert transposed_output(16, kernel=4, stride=2, padding=1) == 32
assert transposed_output(17, kernel=3, stride=2, padding=1, output_padding=1) == 34
```

<DlQuiz kind="伪影诊断" question="转置卷积出现棋盘格伪影的常见结构原因是什么？" :options="['颜色类别太多', 'kernel 与 stride 造成不均匀重叠，使输出位置收到不同次数贡献', 'softmax 太稳定', '标签使用最近邻']" :answer="1" explanation="不均匀覆盖会周期性改变输出幅度；resize 后卷积是常见替代。" />

## 3. U-Net：对称路径与高分辨率跳连

U-Net 的 encoder 逐步下采样获得语义，decoder 逐步上采样，并把对应尺度的 encoder 特征拼接回来。原论文使用 valid convolution，导致 skip feature 需要裁剪；现代实现常用 same padding，但仍可能因奇数尺寸产生一像素错位。

拼接与残差相加不同：拼接后通道数相加，后续卷积决定如何融合；相加要求通道一致且默认同一语义坐标系。医疗/工业小数据中 U-Net 强大，但高分辨率 skip 也可能把纹理噪声直接送入 decoder。

<DlQuiz kind="跳连辨析" question="U-Net 常见 skip connection 与 ResNet shortcut 的主要区别是什么？" :options="['完全相同', 'U-Net 通常跨 encoder-decoder 同尺度拼接空间细节；ResNet 通常在块内相加以改善优化', 'U-Net 不传特征', 'ResNet 只在推理使用']" :answer="1" explanation="二者连接跨度、融合操作和目的不同。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>输入高宽为奇数时，多次 stride=2 下采样后再上采样可能差一像素。给出 crop、pad 或显式 size 插值三种策略的风险。</div>

## 4. 空洞卷积与多尺度上下文

空洞卷积在核元素间插入间隔，以相同参数数扩大采样跨度：

$$
K_{eff}=K+(K-1)(d-1).
$$

DeepLab 使用 atrous spatial pyramid pooling（ASPP）并行多个 dilation rate，加上全局或图像级上下文。较大 dilation 会稀疏采样并产生 gridding；output stride、输入 crop 和 dilation 必须共同设计。

<DilationFieldLab />

<DlQuiz kind="感受野" question="3×3 卷积、dilation=2 的有效核跨度是多少？" :options="['3×3', '4×4', '5×5', '6×6']" :answer="2" explanation="Keff=3+(3-1)(2-1)=5，实际仍只采样 9 个位置。" />

## 5. 损失：像素交叉熵之外

多类交叉熵对有效像素求和或平均，ignore label 必须排除分子和分母。类别权重可缓解频率失衡，但过大权重会放大错标。Dice 对重叠敏感：

$$
\operatorname{Dice}=\frac{2\sum_i p_iy_i+\epsilon}{\sum_i p_i+\sum_i y_i+\epsilon}.
$$

训练时的 soft Dice 与评价时阈值后二值 Dice 不是同一量。边界损失、Focal/Tversky loss 可针对细结构，但每种都引入空类和权重选择。

```python
import numpy as np

prediction = np.array([1, 1, 0, 0, 1], dtype=np.float64)
target = np.array([1, 0, 0, 0, 1], dtype=np.float64)
intersection = (prediction * target).sum()
dice = 2 * intersection / (prediction.sum() + target.sum())
iou = intersection / ((prediction + target) > 0).sum()
assert np.isclose(dice, 0.8)
assert np.isclose(iou, 2 / 3)
```

## 6. 指标反例与边界证据

<SegmentationMetricLab />

pixel accuracy 被背景主导。每类 IoU 再平均为 mIoU，但应说明是否包含背景、空类如何处理。Dice 与 IoU 单调相关，却不能直接说明边界偏移；可增加 boundary F-score、Hausdorff distance 或连通组件级召回。

滑窗推理还会引入重叠融合和边缘 padding。若训练 crop 中目标居中、推理窗口边缘目标缺上下文，指标会产生位置偏差。测试应按距窗口边界距离切片。

<DlQuiz kind="指标反例" question="前景占 1%，模型把所有像素预测为背景时，pixel accuracy 与前景 IoU 分别接近什么？" :options="['都接近 99%', 'accuracy 约 99%，前景 IoU 为 0', 'accuracy 为 0，IoU 为 99%', '都为 0']" :answer="1" explanation="背景正确占绝大多数像素，但没有任何前景 TP。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>两个模型 mIoU 相同，一个边界毛刺多、另一个整体偏移 3 像素。为焊缝机械定位选择额外指标和失败样本可视化。</div>

## 7. 推荐论文阅读

<DlPaperCard title="Fully Convolutional Networks for Semantic Segmentation" meta="Long, Shelhamer & Darrell · 2014" priority="必读 · 任务接口" question="分类网络怎样被重写为任意尺寸的密集预测，浅层 skip 提供了什么？" reading="读 fully convolutional conversion、FCN-32s/16s/8s 与逐步融合结果。" href="https://arxiv.org/pdf/1411.4038" />

<DlPaperCard title="U-Net: Convolutional Networks for Biomedical Image Segmentation" meta="Ronneberger et al. · 2015" priority="必读 · 编解码器" question="小数据条件下，重叠 tile、强增强和高分辨率 skip 如何共同工作？" reading="精读架构图、裁剪对齐、weighted loss 与推理 tiling；不要只复制 U 形外观。" href="https://arxiv.org/pdf/1505.04597" />

<DlPaperCard title="Rethinking Atrous Convolution for Semantic Image Segmentation" meta="Chen et al. · 2017" priority="必读 · 多尺度上下文" question="output stride、ASPP rate 与图像尺度怎样共同决定有效采样？" reading="读 atrous 公式、ASPP 与 decoder 消融，注意训练 crop 对 dilation 的约束。" href="https://arxiv.org/pdf/1706.05587" />

## 8. 练习与门禁

1. 为 U-Net 每一层写 shape 断言，覆盖奇数输入和最小合法尺寸。
2. 对 mask resize 写属性测试：输出只能包含原有类别和 ignore ID。
3. 比较转置卷积与 resize-conv 的棋盘格频谱和边界 F-score。
4. 扫描 dilation/output stride，报告裂纹宽度切片与激活峰值。
5. 同时报告 pixel accuracy、per-class IoU、Dice、boundary F-score 与组件召回。
6. 构造空 mask、全前景、单像素前景、ignore 边界和断裂细线回归集。

## 本章小结

- FCN 把分类器转换为保留空间坐标的密集预测；输出 stride 决定粗糙程度。
- U-Net 通过 encoder-decoder 和同尺度 skip 融合语义与细节，shape 对齐是硬合同。
- 空洞卷积扩大采样跨度但可能产生稀疏网格伪影。
- mask 插值、ignore label 与空类策略比“换一个 loss”更基础。
- pixel accuracy 不能代表稀有前景；mIoU 还需边界与连通组件指标补充。

<DlLessonProgress lesson-id="08" next-href="/courses/deep-learning/09-instance-panoptic-sam" next-label="下一章：实例、全景与 SAM" />
