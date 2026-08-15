---
course: deep-learning
edition: industrial
title: 数据增强、迁移学习与实验诊断
description: 把数据增强写成标签保持合同，比较线性探测、部分解冻和全量微调，建立从过拟合一个 batch 到曲线指纹、checkpoint 与 profiler 的训练诊断顺序。
outline: [2, 3]
---

<DlLessonHero number="03" module="现代训练" title="数据增强、迁移学习与实验诊断" question="预训练模型能跑、训练曲线也在降，怎样证明模型学到的是目标任务，而不是增强伪影、泄漏或错误的运行模式？" duration="建议 260–360 分钟" :outcomes="['把数据增强写成显式标签保持或标签变换合同', '根据数据量与领域偏移设计冻结、部分解冻和全量微调消融', '按最小测试阶梯定位训练故障，而不是盲目调参', '建立可恢复 checkpoint、实验指纹和 profiler 测量协议']" />

前两章建立了优化器和稳定性接口。本章把它们接入完整实验：数据怎样变化、预训练表示怎样复用、运行怎样复现、异常怎样定位。目标不是整理“训练技巧大全”，而是让每个技巧都留下可以审计的输入、状态和反例。

## 贯穿案例：少样本工业表面缺陷分类

工厂新上线一条金属表面检测线，只有 1200 张已复核图片，分为划痕、压痕、污染和正常四类。已有一个在自然图像上预训练的视觉骨干。团队希望一周内建立可靠基线。

关键难点：

- 缺陷方向可能有业务含义，任意旋转不一定保持标签；
- 同一卷材的相邻图片高度相关，按图片随机切分会泄漏；
- 自然图像与灰度近摄纹理存在领域偏移；
- 全量微调参数多，少样本下容易过拟合；
- 训练吞吐受图像解码和增强影响，不一定由 GPU 计算决定。

数据合同先于模型：

| 对象 | 合同 |
|---|---|
| 样本单位 | 一张经过设备校准的表面图像 |
| 分组单位 | `coil_id`，同一卷材只进入一个集合 |
| 标签 | 四类互斥结果，由双人复核与争议队列产生 |
| 特征截止 | 拍摄时可得的像素和设备元数据 |
| 冻结测试 | 最后 20% 卷材，模型选择期间不可查看 |
| 主指标 | macro F1；同时报告每类召回与拒绝率 |
| 成本指标 | GPU 小时、峰值显存、单图延迟 |

<DlQuiz kind="切分审计" question="同一卷材的相邻图片随机分到训练和测试，会造成什么风险？" :options="['只会让训练变慢', '高度相似纹理跨集合泄漏，测试分数可能估计记忆而非跨卷材泛化', '会自动增加类别数量', '只影响回归任务']" :answer="1" explanation="分组单位必须匹配未来泛化单位。相邻窗口共享材料、设备与缺陷纹理，按行随机切分会破坏独立证据。" />

## 1. 数据增强是一条标签语义声明

增强从分布 $q(a)$ 采样变换 $a$，训练目标变为

$$
\widehat R_{\mathrm{aug}}(\theta)=
\frac1n\sum_{i=1}^{n}
\mathbb E_{a\sim q}[L(f_\theta(a(x_i)),\widetilde y_i(a))].
$$

分类中常假设 $\widetilde y_i(a)=y_i$，即变换保持标签。检测、分割和关键点任务则必须同步变换框、mask 或坐标。若图像翻转了而框没有翻，代码能运行，监督信号却已经错误。

### 1.1 不变性与等变性

分类希望

$$
f(a(x))\approx f(x),
$$

称为对变换不变。分割希望输出随空间变换同步变化：

$$
f(a(x))\approx a(f(x)),
$$

称为等变。混淆二者会产生“增强后标签不变”的错误假设。

### 1.2 常见增强的业务反例

| 增强 | 可能有效的先验 | 反例 |
|---|---|---|
| 水平翻转 | 左右方向不影响类别 | 文字、交通方向、器件极性 |
| 随机旋转 | 目标朝向任意 | 重力方向、加工纹理方向 |
| 颜色抖动 | 光照变化不影响目标 | 颜色本身决定等级或温度 |
| 随机裁剪 | 局部仍含目标证据 | 小缺陷被裁掉却保留正标签 |
| Mixup | 类别间可线性插值 | 精确定位、稀有安全事件 |
| CutMix | 局部区域可组合 | 标签比例与真实语义不一致 |

增强必须在训练集内完成，验证和测试只做确定性预处理。若每次验证随机裁剪，指标同时包含模型变化和采样噪声。

```python
def assert_classification_augmentation(image, label, transform):
    augmented = transform(image)
    assert augmented.shape == image.shape
    assert augmented.dtype == image.dtype
    assert label in {0, 1, 2, 3}
    return augmented, label
```

这段断言只能检查形状和类型，不能证明语义保持。语义必须由领域审查、可视化抽样和消融实验共同支持。

<DlQuiz kind="增强语义" question="对语义分割图像做随机水平翻转时，正确做法是什么？" :options="['只翻转输入图像', '图像与 mask 使用同一随机参数同步翻转', '保持图像不变只翻 mask', '重新随机生成类别标签']" :answer="1" explanation="分割目标对空间变换等变。输入与像素标签必须共享同一次随机变换。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>为金属划痕任务分别判断水平翻转、90° 旋转、颜色抖动、随机裁剪是否保持标签。若信息不足，写出必须询问的工艺事实，不要猜。</div>

## 2. 迁移学习的三个基线

把预训练模型拆成表征骨干 $h_{\phi}$ 与任务头 $g_w$：

$$
f_{\phi,w}(x)=g_w(h_\phi(x)).
$$

### 2.1 线性探测或只训练任务头

冻结 $\phi$，只优化 $w$。它回答“现有表示是否已经线性暴露目标信息”。优点是快、方差小、显存低；缺点是无法适配大的领域偏移。

线性探测不是最终模型的自动选择，而是诊断基线。若线性探测已经很好，全量微调收益很小且方差更大，部署时可以优先简单方案。

### 2.2 部分解冻

先训练任务头，再解冻骨干后部若干层。低层通常保留边缘和局部纹理，后层更任务特化，但这只是经验倾向，工业近摄图像可能恰好需要重新学习低层统计。

解冻时要重新构造或更新优化器参数组。把 `requires_grad=True` 改回来却没有加入 optimizer，是常见的“看似解冻、实际不更新”。

### 2.3 全量微调

所有参数参与训练，适应能力最强，也最容易在小数据上遗忘预训练表示或过拟合。常用较小 backbone 学习率：

$$
\eta_{\mathrm{backbone}}=\alpha\eta_{\mathrm{head}},\qquad 0<\alpha<1.
$$

还可以按层深设置 layer-wise learning-rate decay。它增加了参数组复杂度，必须记录每组参数数量、学习率和衰减。

<TransferFreezeLab />

<DlQuiz kind="策略选择" question="只有 300 个样本、领域偏移较小，第一条有信息量的迁移基线通常是什么？" :options="['直接从零训练最大模型', '冻结骨干，只训练任务头并建立线性探测基线', '删除验证集以增加训练样本', '只比较训练准确率']" :answer="1" explanation="线性探测能快速判断预训练表示是否可复用，并提供低方差、低成本基线；之后再决定是否值得解冻。" />

### 2.4 冻结不等于推理模式

`requires_grad=False` 只阻止参数梯度。BatchNorm 的 running statistics 和 Dropout 行为由 `train()`/`eval()` 控制。一个冻结骨干若仍处于 train mode，BatchNorm 状态可能继续漂移；若整个模型 `eval()`，新任务头中的 Dropout 也会被关闭。

需要按模块明确：

```python
# tutorial-validation: syntax-only
backbone.requires_grad_(False)
backbone.eval()
head.requires_grad_(True)
head.train()

assert not any(parameter.requires_grad for parameter in backbone.parameters())
assert all(parameter.requires_grad for parameter in head.parameters())
```

不同架构没有完全相同的冻结规则。没有 BatchNorm 的 Transformer 骨干仍可能含 Dropout 或 stochastic depth。

## 3. 一个可靠的训练诊断阶梯

出现异常时，从最小、最确定的测试开始。直接启动大规模超参数搜索会把代码、数据和优化错误混在一起。

### 3.1 边界 0：静态合同

- 输入 shape、dtype、范围、设备；
- 标签取值、ignore index、mask 数量；
- loss reduction 是 mean 还是 sum；
- 可训练参数数量与 optimizer 参数组；
- train/eval 模式；
- 一个 batch 内是否含非有限值。

### 3.2 边界 1：过拟合一个小 batch

关闭强增强和随机正则，固定一个小 batch，尝试把训练 loss 降到接近 0。若做不到，优先检查模型—损失接口、标签、学习率、梯度、冻结状态和实现错误，而不是泛化。

这个测试不是发布证据。能记住 16 个样本只证明训练路径具有最小一致性，不证明验证集有效。

### 3.3 边界 2：小数据端到端运行

用少量真实 batch 跑完整训练、验证、checkpoint、恢复和日志。目标是验证状态机，不追求准确率。

### 3.4 边界 3：健康基线

选择最简单、已知可训练的配置，记录训练/验证曲线、梯度范数、吞吐和显存。任何新技巧都与这条基线单独消融。

<TrainingDiagnosisLab />

<DlQuiz kind="调试顺序" question="新训练脚本在完整数据上 loss 不降，最先应做哪项测试？" :options="['直接搜索 200 组超参数', '关闭随机增强，在一个小 batch 上尝试过拟合并检查梯度与标签', '立即换更大模型', '只延长训练时间']" :answer="1" explanation="单 batch 测试能快速隔离实现与优化路径问题；若最小样本都记不住，先谈泛化没有意义。" />

## 4. 训练曲线是指纹，不是根因证明

常见指纹：

| 曲线 | 候选原因 | 下一条证据 |
|---|---|---|
| 训练和验证都不降 | 学习率、冻结、标签、容量、梯度中断 | 梯度非零、参数更新量、单 batch 测试 |
| 训练降，验证上升 | 过拟合、切分偏移、train/eval 错位 | 增强消融、分组切片、运行模式 |
| 两条曲线锯齿尖峰 | 学习率过高、坏 batch、overflow | 按 step 对齐梯度范数、batch id、loss scale |
| 验证异常完美 | 泄漏、重复样本、后验特征 | 实体/时间重叠、近重复检索、特征截止 |
| 恢复后轨迹跳变 | 状态未恢复、数据游标变化 | optimizer/scheduler/RNG/scaler 指纹 |

同一指纹可能来自多个原因，所以曲线只能提出候选集合。诊断需要“若原因 A 为真，还应看到什么”这样的可证伪预测。

<div class="dl-stop"><strong>停下来推一推 2</strong>训练 loss 平滑下降，验证 loss 从第 3 个 epoch 开始上升。分别为过拟合、BatchNorm 模式错误、训练/验证预处理不一致写出一个额外可观测预测。</div>

## 5. 可复现实验不等于只设置 seed

一个实验指纹至少包含：

```text
git_commit
config_hash
dataset_version + split_hash
model_init_source
optimizer/scheduler/precision
library + driver + device versions
random seeds
checkpoint parent
```

设置 seed 仍可能受非确定性 kernel、数据加载顺序、不同 GPU、编译器选择和浮点归约顺序影响。应区分两种要求：

- **精确重放**：调试或恢复测试要求逐步相同；可能牺牲性能。
- **统计复现**：科学比较要求多个种子的分布结论一致；不要求 bitwise 相同。

只报告一个“最好 seed”会把随机性当作模型能力。

### 5.1 Checkpoint 是完整状态快照

```python
# tutorial-validation: syntax-only
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict() if scaler is not None else None,
    "global_step": global_step,
    "tokens_seen": tokens_seen,
    "data_cursor": data_cursor,
    "rng_state": capture_rng_state(),
    "config_hash": config_hash,
}
```

生产中还要原子写入、校验和、版本兼容与保留策略。保存成功日志不能代替恢复测试；至少定期在隔离进程中加载并运行一步。

<DlQuiz kind="复现边界" question="两次运行设置了相同 seed，但使用不同 GPU kernel，结果有微小差异。这一定说明代码错误吗？" :options="['一定', '不一定；浮点归约顺序和非确定性 kernel 可导致差异，应按精确重放或统计复现目标判断', 'seed 从不影响训练', '只要最终准确率相同就无需记录环境']" :answer="1" explanation="seed 只是随机状态的一部分。需要记录设备、库、kernel 与确定性设置，并明确允许的复现容差。" />

## 6. Profiler 先测时间线，再猜瓶颈

训练 step 可拆为：数据读取与解码、CPU 增强、host-to-device、forward、backward、optimizer、通信和同步。GPU 利用率低并不自动说明模型太小；可能是输入管线让 GPU 等待。

测量协议：

1. 预热，排除编译、缓存和首次分配；
2. 固定 shape、batch、精度与增强；
3. 使用正确同步边界；
4. 采样多步并报告分位数；
5. 区分 CPU wall time 与 GPU kernel time；
6. 保存 profiler trace 和环境指纹；
7. 优化后用相同协议复测，并验证数值正确性。

典型判断：

| 观察 | 候选瓶颈 | 验证实验 |
|---|---|---|
| GPU 时间线有周期性空洞 | DataLoader 或 CPU 增强 | 使用预解码数据/增加 worker 的 A/B |
| 很多短小 kernel | launch overhead、缺少融合 | 编译/融合前后 kernel 数与端到端时间 |
| H2D 与计算不重叠 | pinned memory/异步复制失效 | 时间线检查 stream 重叠 |
| optimizer step 占比高 | 状态更新带宽 | fused optimizer 与普通实现正确性/时间对比 |
| OOM 但理论账本可放下 | 激活峰值、workspace、碎片 | memory snapshot 与逐阶段峰值 |

## 7. 数据增强与迁移实验矩阵

最小实验矩阵不是全组合爆炸，而是逐条回答问题：

1. 无增强、冻结骨干：表示可用性基线；
2. 业务批准的弱增强、冻结骨干：增强是否改善头部泛化；
3. 弱增强、部分解冻：适配后层是否有收益；
4. 弱增强、全量微调：额外自由度是否值得；
5. 强增强、最佳解冻策略：收益是否来自正则而非标签破坏；
6. 从零训练：预训练到底贡献多少。

每次只改变一个主因，使用同一分组切分和冻结评价。报告总体 macro F1 之外，还要按缺陷类型、卷材、相机、亮度和缺陷尺寸切片。

<DlQuiz kind="消融设计" question="为了判断强增强是否有帮助，最小可比实验应保持什么不变？" :options="['只保持模型名称', '数据切分、初始化来源、解冻策略、训练预算、评价协议，仅改变增强策略', '可以同时更换优化器与 batch', '只看增强后的训练准确率']" :answer="1" explanation="消融要让主要差异可归因。若同时改变优化器、预算或解冻策略，就无法判断收益来自哪里。" />

## 8. 失效模式与发布门禁

| 失效 | 离线表现 | 上线风险 | 门禁 |
|---|---|---|---|
| 卷材泄漏 | 测试分数异常高 | 新卷材性能骤降 | 分组 hash 无交集、近重复审计 |
| 非标签保持增强 | 训练更难或表面更稳健 | 系统性错判特定方向 | 领域批准、增强样本与标签联检 |
| 冻结骨干仍更新 BN | 训练验证波动 | 推理统计失配 | 模块模式单元测试、状态 diff |
| 全量微调遗忘 | 训练集极好 | 少数类和域外退化 | 线性探测/部分/全量消融与切片 |
| checkpoint 不完整 | 平时无影响 | 故障恢复轨迹跳变 | 隔离恢复下一步一致性测试 |
| profiler 测量错误 | 宣称“提速” | 实际吞吐或延迟无收益 | warmup、同步、重复和正确性基线 |

发布门禁：训练数据与冻结测试按实体隔离；所有增强有标签语义说明；至少比较三种解冻策略；恢复测试通过；最终报告包含分布、切片、成本和失败样本，而不是单一最佳分数。

<div class="dl-stop"><strong>停下来推一推 3</strong>模型总体 macro F1 提升 2 pp，但“污染”类召回下降 8 pp，训练时间翻倍。给出发布、拒绝或继续实验所需的决策信息。</div>

## 9. 推荐论文阅读

<DlPaperCard title="ImageNet Classification with Deep Convolutional Neural Networks" meta="Krizhevsky, Sutskever & Hinton · 2012" priority="必读 · 历史入口" question="数据增强、Dropout、GPU 训练和模型规模如何共同组成结果，而不是单一架构故事？" reading="先读训练细节与数据增强，再看消融和双 GPU 约束。第 04 章会从架构演进重新阅读。" href="https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf" />

<DlPaperCard title="A Simple Framework for Contrastive Learning of Visual Representations" meta="Chen et al. · 2020" priority="选读" question="为什么增强组合本身决定了自监督任务中什么信息被视为不变？" reading="重点读数据增强组合与 projection head 消融；把“更强增强”理解成预训练目标设计，而不是通用图像技巧。" href="https://arxiv.org/pdf/2002.05709" />

<DlPaperCard title="Masked Autoencoders Are Scalable Vision Learners" meta="He et al. · 2021" priority="选读 · 迁移入口" question="预训练与下游微调的输入、编码器和评价协议怎样连接？" reading="读方法图、mask ratio 消融和 fine-tuning/linear probing 结果；关注表示评价为何需要多种适配方式。" href="https://arxiv.org/pdf/2111.06377" />

## 10. 练习

1. 为一个检测任务写出随机裁剪后的图像、框、类别、空目标和最小可见比例合同。
2. 实现分组切分检查，断言 `coil_id` 在 train/validation/test 之间无交集。
3. 比较只训练任务头、解冻最后一 stage、全量微调的可训练参数数、峰值显存和验证分布。
4. 构造一个单 batch 过拟合测试，并说明关闭了哪些随机增强与正则化。
5. 创建 checkpoint 后在新进程恢复，固定下一 batch，比较 loss、梯度摘要和一步更新后的参数 hash。
6. 对数据加载瓶颈做 profiler：固定 GPU 计算，改变 worker 数和预取，报告 P50/P95 step time。
7. 选择本章故障注入台中的两个曲线指纹，为每个指纹写出三个竞争假设和各自的下一条证据。

## 本章小结

- 数据增强是标签不变或等变声明，必须由业务语义、同步变换和消融共同验证。
- 线性探测、部分解冻和全量微调回答不同问题，应构成连续证据链。
- 冻结梯度与切换 train/eval 不是同一个操作，BatchNorm 与 Dropout 状态必须单独控制。
- 调试从静态合同、单 batch、小数据完整运行到健康基线逐层推进。
- 曲线是候选根因的指纹，不是证明；诊断要提出额外可观测预测。
- 可复现训练需要数据、环境、状态和父 checkpoint 指纹，不能只记录 seed。
- Profiler 结论绑定 warmup、同步、shape、精度和正确性基线。

<DlLessonProgress lesson-id="03" />
