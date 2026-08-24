---
course: deep-learning
title: 深度学习与大模型
description: 从现代训练、计算机视觉和生成模型进入大语言模型、GPU 算子、分布式训练与推理服务。
outline: [2, 3]
---

# 深度学习与大模型

> **状态：已开放。** 全部 32 章正文、64 个专用交互实验、8 次综合作业、结课项目与推荐论文导读均已完成，并通过内容、代码、构建、内部链接及桌面/移动端验收。

课程接续本站[数据科学图像桥接模块](./data-science/v1-digital-images.md)与机器学习课，不重复讲解图像数组和 OpenCV 基础、MLP、反向传播、CNN 基础、RNN 时间展开和自注意力基础推导。视觉部分直接从 AlexNet、VGG、ResNet 开始，进入目标检测、语义分割、ViT 与 Swin Transformer；语言部分从文本、分词和词向量建立新数据接口，再系统进入现代大语言模型。GPU、CUDA 生态、AI 算子、分布式训练、vLLM 和 SGLang 是正式课程内容，不作为附录处理。

<figure class="dl-figure">
  <img src="/images/deep-learning/training-signal-lab.png" alt="抽象训练系统剖面：前向信号穿过多层网络，反向梯度沿相反方向返回，优化器状态、学习率、混合精度和诊断仪表分布在信号链周围">
  <figcaption>原创课程插图：把深度学习训练看成一条需要持续观测的信号链。青色表示前向表示与有效梯度，琥珀色表示优化器状态和训练预算，红色表示需要诊断的异常路径。插图由 GPT 图像生成工具生成，正文中的公式图和实验图使用可交互 SVG 独立实现。</figcaption>
</figure>

## 课程规模

- 16 周，32 讲，预计核心学习时间 90–120 小时。
- 6 个模块，8 次综合作业，1 个结课项目。
- 86 份公开课件和 61 篇推荐论文已经下载并完成 PDF 完整性校验。
- 核心实验支持 CPU、浏览器或单张消费级 GPU；多卡和服务实验使用小模型、模拟器、日志或可选云 GPU。

## 六个模块

| 模块 | 讲次 | 核心内容 |
|---|---:|---|
| 现代深度学习训练 | 01–03 | Adam、AdamW、归一化、混合精度、迁移学习与训练诊断 |
| 计算机视觉 | 04–11 | AlexNet、VGG、ResNet、检测、分割、ViT、Swin 与视觉自监督 |
| 生成模型 | 12–15 | GAN、WGAN、StyleGAN、VAE、Diffusion 与 Flow Matching |
| NLP 基础 | 16–18 | Unicode、Tokenization、BPE、Word2Vec、GloVe 与神经语言模型 |
| 大语言模型 | 19–28 | 预训练、现代架构、Scaling Law、后训练、RAG、Agent、多模态与评估 |
| GPU 与 AI Infra | 29–32 | 训练/推理硬件、CUDA/Triton/TileLang、分布式训练、vLLM 与 SGLang |

## 已开放章节

1. [Adam、AdamW 与学习率调度](./deep-learning/01-adam-adamw.md)：矩估计、偏差修正、解耦权重衰减、固定预算调度与优化器显存账本。
2. [初始化、归一化与稳定训练](./deep-learning/02-stable-training.md)：信号传播、残差路径、BatchNorm/LayerNorm/RMSNorm、FP16/BF16 与 loss scaling。
3. [数据增强、迁移学习与实验诊断](./deep-learning/03-transfer-diagnostics.md)：增强语义、冻结/解冻、单 batch 调试、曲线指纹、checkpoint 与 profiler。
4. [AlexNet、VGG、GoogLeNet 与 ResNet](./deep-learning/04-classic-vision-architectures.md)：经典架构的参数/FLOPs/激活账本与残差优化路径。
5. [MobileNet、EfficientNet 与 ConvNeXt](./deep-learning/05-efficient-convnets.md)：深度可分离卷积、倒残差、复合缩放和设备级延迟。
6. [R-CNN 系列与两阶段目标检测](./deep-learning/06-two-stage-detection.md)：proposal、anchor、RoIAlign、FPN、NMS 与 AP。
7. [YOLO、RetinaNet 与 DETR](./deep-learning/07-one-stage-detection-detr.md)：密集预测、Focal Loss、object query 与集合匹配。
8. [FCN、U-Net 与语义分割](./deep-learning/08-semantic-segmentation.md)：编解码器、空洞卷积、像素损失与分割指标反例。
9. [实例分割、全景分割与 SAM](./deep-learning/09-instance-panoptic-sam.md)：Mask R-CNN、全景装配、PQ 和提示稳定性。
10. [Vision Transformer（ViT）与 DeiT](./deep-learning/10-vision-transformer.md)：patch token、位置编码、全局注意力预算与数据效率。
11. [Swin Transformer 与视觉自监督学习](./deep-learning/11-swin-self-supervised.md)：移位窗口、层级特征、SimCLR、DINO、MAE 与 CLIP。
12. [GAN 与 DCGAN](./deep-learning/12-gan-dcgan.md)：密度比、非饱和目标、交替优化、DCGAN 与模式坍缩诊断。
13. [WGAN、StyleGAN 与生成质量评估](./deep-learning/13-wgan-stylegan.md)：Lipschitz critic、梯度惩罚、层级样式与质量—覆盖—记忆评价。
14. [VAE、VQ-VAE 与自回归生成](./deep-learning/14-vae-vqvae-autoregressive.md)：ELBO、重参数化、离散码本与因果序列生成。
15. [Diffusion、Latent Diffusion 与 Flow Matching](./deep-learning/15-diffusion-flow-matching.md)：噪声日程、采样器、潜空间扩散、CFG 与连续输运。
16. [文本数据、分词与子词建模](./deep-learning/16-tokenization.md)：Unicode 边界、规范化合同、word/character/byte/subword 接口与 BPE。
17. [Word2Vec、GloVe 与词向量评估](./deep-learning/17-word-embeddings.md)：局部预测、负采样、全局共现、表示几何与偏置评价。
18. [神经语言模型与上下文表示](./deep-learning/18-neural-language-models.md)：序列概率、teacher forcing、perplexity、ELMo 与双向预训练桥梁。
19. [BERT、GPT 与 T5 预训练范式](./deep-learning/19-pretraining-paradigms.md)：三种目标的信息边界、损失位置、模型接口与任务选择。
20. [现代 LLM 架构、MoE 与长上下文](./deep-learning/20-modern-llm-architecture.md)：RMSNorm、SwiGLU、RoPE、GQA/KV、FlashAttention 与专家路由。
21. [预训练数据工程与 Scaling Law](./deep-learning/21-data-scaling-laws.md)：来源 lineage、去重、混合、污染、参数—token—计算预算。
22. [指令微调、LoRA 与 QLoRA](./deep-learning/22-sft-lora-qlora.md)：模板/loss 合同、低秩更新、4-bit 底座与遗忘门禁。
23. [RLHF、PPO、DPO 与推理模型](./deep-learning/23-rlhf-dpo-reasoning.md)：偏好数据、奖励—KL、直接偏好优化与可验证推理奖励。
24. [解码、投机解码与 Test-Time Scaling](./deep-learning/24-decoding-test-time-scaling.md)：分布裁剪、beam、无损草稿验证与统一推理预算。
25. [RAG、向量检索与知识边界](./deep-learning/25-rag.md)：混合召回、切块、重排、证据冲突、引用和拒答。
26. [工具调用、结构化输出与 Agent](./deep-learning/26-tools-agents.md)：Schema、约束解码、状态机、幂等、权限和故障注入。
27. [多模态大模型与统一表示](./deep-learning/27-multimodal.md)：CLIP 对齐、视觉 connector/token、grounding 与模态冲突。
28. [大模型评估、可靠性、安全与可解释性](./deep-learning/28-evaluation-safety.md)：配对统计、Judge 偏置、校准、红队、隐私与解释边界。
29. [AI 训练硬件、推理硬件与 CUDA 生态](./deep-learning/29-hardware-cuda.md)：SIMT/Tensor Core、Roofline、HBM/互联、训练/prefill/decode 与软件栈。
30. [AI 算子与 CUDA、Triton、TileLang](./deep-learning/30-cuda-triton-tilelang.md)：Fusion、tiling、online reduction、四类 kernel 案例与选择边界。
31. [分布式训练、ZeRO、FSDP 与 Megatron](./deep-learning/31-distributed-training.md)：DP/TP/PP/CP/EP、状态分片、重算、拓扑与扩展效率。
32. [vLLM、SGLang 与大模型推理服务](./deep-learning/32-vllm-sglang.md)：Continuous batching、PagedAttention、RadixAttention、SLO 与公平基准。

三十二章共提供 64 个专用交互实验。所有实验都在浏览器本地运行，用于改变训练状态、结构预算、视觉/生成/文本表示、预训练与适配、解码与 RAG、工具/多模态/评估、Roofline、kernel tile、并行拓扑或服务调度；它们不上传数据，也不把教学响应面冒充真实模型 benchmark。

<figure class="dl-figure">
  <img src="/images/deep-learning/vision-model-observatory.png" alt="工业零件在视觉模型观测台中依次变成卷积特征、特征金字塔、检测框、语义和实例掩码、图像 patch token、移位窗口注意力与自监督共享表示">
  <figcaption>原创视觉模块插图：同一工业目标沿分类、检测、分割、视觉 Transformer 和自监督学习五类接口被重新表示。图中只承担模块关系总览，精确张量、匹配、指标和成本由第 04–11 章的 16 个交互实验给出。</figcaption>
</figure>

<figure class="dl-figure">
  <img src="/images/deep-learning/generative-model-observatory.png" alt="深色生成模型观测台中，对抗生成回路、连续潜空间、离散码本序列和噪声去除与流输运四条路径生成同一组抽象样本分布">
  <figcaption>原创生成模型模块插图：红色闭环表示生成器与判别器的对抗博弈，紫色瓶颈表示连续潜变量，离散方格表示 VQ 码本与自回归 prior，琥珀时间片和青色流线表示扩散去噪与连续输运。插图由 GPT 图像生成工具生成；第 12–15 章的 8 个交互实验负责给出可计算的目标、梯度、瓶颈和噪声日程。</figcaption>
</figure>

<figure class="dl-figure">
  <img src="/images/deep-learning/llm-infra-observatory.png" alt="大语言模型与 AI Infra 观测台从左到右展示文档与多色 token 流、Transformer 和稀疏专家路由、GPU tiled compute 与 HBM、多加速器互联，以及持续进入服务队列并复用分页 KV 块和共享前缀树的请求">
  <figcaption>原创大模型与 AI Infra 总览：左侧把数据 lineage 与 token 接口保留到模型入口，紫色分支表示 MoE 专家路由，中部青色矩阵与琥珀分块表示 tiled compute 和分层内存，互联网络把计算扩展到多个设备；右侧循环队列、分页块表和共享前缀树对应 continuous batching、PagedAttention 与 RadixAttention。插图由 GPT 图像生成工具生成；第 19–32 章的交互实验给出架构、显存、并行、算子和服务 SLO 的可计算边界。</figcaption>
</figure>

## 学习边界

本课程假设学习者已经理解梯度下降、反向传播、卷积、自注意力和基础策略梯度。每讲会提供前置诊断和回链，但不会复制机器学习课的基础推导。模型章节必须同时解释结构、张量形状、计算/内存成本、实验依据和失效条件；系统章节必须用延迟、吞吐、显存、带宽或正确性检查给出可证伪结论。

## 已确认资料

- [32 讲章节结构与作业映射](./deep-learning-chapter-structure.md)
- [完整内容建设计划](./deep-learning-content-plan.md)
- [参考课程、资料快照与 61 篇推荐论文](./deep-learning-sources.md)

## 实践与交付

- [八次综合作业](./deep-learning/assignments.md)：从训练系统、视觉、生成和 Tiny GPT 进入 LoRA/RAG、GPU kernel 与 vLLM/SGLang 服务基准；每次都要求原始结果、成本、失败分析和发布判断。
- [结课项目](./deep-learning/project.md)：模型、应用、系统三条路线，包含 12 问提案、五个里程碑、冻结预算、必做实验矩阵、评分量规与发布审查门禁。

课程参考 Stanford CS231n、Stanford CS224n、CMU 11-785 和 CMU Advanced NLP 的公开资料，但不会翻译或复刻第三方 slides。所有中文正文、案例、图形、代码、测验和作业均独立创作。

## 开放标准

课程已完成 32 个章节、8 次作业和结课项目的原创内容建设；内容检查覆盖章节结构、交互组件、即时检查、论文导读、图片与内部链接，Python 正文代码、生产构建和桌面/移动端页面也已通过验收。后续改动进入试学反馈驱动的维护阶段，不再以目录或资料下载代替完整课程交付。
