# 深度学习与大模型课程网站自学内容建设计划

## 1. 建设结论

深度学习与大模型课程采用“**16 周、32 讲、8 次综合作业、1 个结课项目**”的结构，预计核心学习时间为 90–120 小时。课程以 Stanford CS231n、Stanford CS224n、CMU 11-785 和 CMU Advanced NLP 的公开课程资料为知识范围与论文入口，同时加入 GPU、算子、分布式训练和大模型推理服务模块。

这里的“参考公开课程”是指核对主题覆盖、模型演进、术语、公式和论文入口，不是翻译、重排或复刻第三方 slides。所有中文正文、案例、数据、图形、代码、题目和网页交互必须独立创作。

课程接续本站机器学习课程。学习者已经学过 MLP、反向传播、CNN、RNN、Transformer、自注意力、基础优化和策略梯度，因此本课程不再重复这些内容的基础推导。卷积部分直接从 AlexNet、VGG、GoogLeNet 和 ResNet 开始；Transformer 部分直接从现代视觉架构、预训练范式和 LLM 工程结构开始。

## 2. 目标学习者与先修要求

目标学习者应当已经具备以下能力：

- 能使用 Python、NumPy 和基础 PyTorch 训练小型神经网络。
- 理解训练集、验证集、测试集和冻结评测的区别。
- 理解梯度下降、反向传播、交叉熵、正则化和基本概率模型。
- 理解 CNN、RNN、Transformer 和自注意力的输入输出形状与最小实现。
- 能阅读基础英文论文的摘要、方法图、实验表和消融结论。

课程不要求学习者已经拥有多卡 GPU。核心概念实验必须能在浏览器、CPU 或单张消费级 GPU 上完成；分布式训练和大模型服务章节使用小模型、模拟器、运行日志或可选云 GPU 实验。任何真实硬件基准都必须报告设备型号、软件版本、数据类型、输入形状和测量协议。

## 3. 与机器学习课程的边界

| 已在机器学习课完整讲解 | 本课程只做的事情 |
|---|---|
| MLP、激活函数和表示折叠 | 比较现代网络中的归一化、残差路径、激活与稳定训练配置 |
| 反向传播、VJP、计算图和梯度检查 | 使用 profiler、混合精度、梯度缩放和检查点诊断训练系统 |
| CNN 的局部连接、参数共享、卷积形状和反向传播 | 从 AlexNet 开始分析架构演进、检测、分割和视觉 Transformer |
| RNN、LSTM、GRU、BPTT 和梯度问题 | 只在语言模型演进中说明它们为何被现代预训练架构替代或保留 |
| Q/K/V、自注意力、多头、位置编码和因果 Mask | 研究 BERT/GPT/T5、RoPE、GQA、KV Cache、FlashAttention、MoE 和长上下文 |
| REINFORCE、Actor–Critic、PPO 基础 | 研究 RLHF、偏好优化、推理训练中的奖励与验证接口 |

每章开头设置 5–10 分钟前置诊断。未通过时链接回机器学习课程对应章节，不在本课程复制同一套讲解。

## 4. 课程总体结构

| 模块 | 讲次 | 核心问题 |
|---|---:|---|
| 现代深度学习训练 | 01–03 | 怎样让深层模型在固定预算下稳定、可复现地训练？ |
| 计算机视觉 | 04–11 | 架构怎样从卷积层堆叠演进到残差、密集预测和视觉 Transformer？ |
| 生成模型 | 12–15 | 不同生成目标怎样建模数据分布，它们各自如何失败？ |
| NLP 基础 | 16–18 | 文本怎样变成可训练的离散序列和连续表示？ |
| 大语言模型 | 19–28 | 模型、数据、计算、后训练、检索和评估怎样组成完整 LLM 系统？ |
| GPU 与 AI Infra | 29–32 | 硬件、算子、并行策略和服务调度怎样决定成本与性能？ |

## 5. 每讲页面结构

每讲页面使用相同的证据结构，但交互形式必须由本讲问题决定：

1. 本讲要解决的问题、建议用时和先修诊断。
2. 3–5 个可验证学习目标。
3. 一个贯穿案例及其数据、形状、预算和评价合同。
4. 模型或系统的核心机制，不从模型名称倒推空泛介绍。
5. 必要公式、张量形状和计算/内存复杂度。
6. 至少一个可运行的最小实现或实验驱动代码片段。
7. 至少两个与本讲机制直接相关的交互实验。
8. 三次“先预测，再运行”的推导或系统停顿。
9. 至少五道即时反馈题。
10. 训练、评估、部署或安全中的失效条件。
11. 必读论文、选读论文和对应的精确章节或页码。
12. 本讲小结、术语卡片和下一讲接口。

模型章节不能只展示网络结构图；系统章节不能只展示组件名。每章必须给出一个可以被日志、指标、形状检查或反例推翻的判断。

## 6. 三十二讲内容与实验规划

| 讲次 | 自学内容重点 | 不重复的边界 | 核心实验或证据 |
|---:|---|---|---|
| 01 | Adam 的一阶/二阶矩、偏差修正、AdamW 解耦权重衰减、warmup、cosine、one-cycle、优化器状态成本 | 不重新推导 SGD 和 Momentum 基础 | 在同一梯度序列上比较 Adam/L2 与 AdamW；固定 token 或 step 预算比较调度策略 |
| 02 | Xavier/He、残差路径、BatchNorm、LayerNorm、RMSNorm、激活、梯度裁剪、FP16/BF16 与 loss scaling | 不重讲链式法则 | 逐层记录激活/梯度分布；制造 FP16 overflow 并用动态缩放恢复 |
| 03 | 数据增强、迁移学习、冻结与解冻、checkpoint、随机性、训练曲线诊断、过拟合一个 batch、profiler | 不重讲训练/验证/测试定义 | 故障注入训练控制室；比较错误学习率、标签、归一化和数据泄漏的曲线指纹 |
| 04 | ImageNet、AlexNet、VGG、GoogLeNet、ResNet，深度、宽度、感受野、残差优化与计算预算 | 不重讲卷积公式 | 架构账本比较参数、FLOPs、激活和准确率；残差块 Jacobian 路径实验 |
| 05 | Depthwise separable convolution、MobileNet、EfficientNet 复合缩放、ConvNeXt 的现代化卷积设计 | 不重讲普通卷积核计算 | 同一延迟预算下比较标准卷积和 depthwise；精度—延迟—显存 Pareto 前沿 |
| 06 | R-CNN、Fast R-CNN、Faster R-CNN、anchor、RPN、RoI Pool/Align、NMS 与 mAP | 不重讲二分类器基础 | Anchor 匹配与 NMS 工作台；按尺寸和遮挡切片分析 AP |
| 07 | YOLO、RetinaNet、Focal Loss、DETR 集合预测、Hungarian matching、单阶段与两阶段取舍 | 不重新推导 Softmax | 类别不平衡下比较 CE/Focal；同一场景比较 NMS 与一对一匹配 |
| 08 | FCN、encoder–decoder、skip connection、U-Net、DeepLab、dilation、IoU/Dice 与边界误差 | 不重讲卷积上采样基础 | 分辨率与感受野联动实验；像素准确率与 IoU 反例 |
| 09 | Mask R-CNN、实例/语义/全景分割、mask quality、promptable segmentation、SAM 数据引擎 | 不把 SAM 当作通用零误差工具 | 实例粘连与小物体错误分析；提示点/框变化对 mask 稳定性的影响 |
| 10 | patch embedding、class token、位置表示、ViT、DeiT、数据规模、归纳偏置和迁移 | 不重讲 Q/K/V 与多头机制 | Patch 大小对 token 数、FLOPs 和细粒度识别的影响；CNN/ViT 数据效率对照 |
| 11 | Swin 窗口注意力、shifted window、层次特征、SimCLR、DINO、MAE、CLIP | 不重复自注意力矩阵推导 | 窗口移动的跨区通信实验；线性探测与微调的表示质量对比 |
| 12 | minimax GAN、非饱和生成器目标、DCGAN、判别器—生成器动态、模式坍缩 | 不重讲二元交叉熵定义 | 二维混合分布 GAN；观察判别器过强、梯度消失和模式覆盖 |
| 13 | Wasserstein 距离、Lipschitz 约束、WGAN-GP、Conditional GAN、StyleGAN、FID 与 precision/recall | 不把单张好看样本当评价 | 梯度惩罚消融；构造 FID 相近但覆盖率不同的反例 |
| 14 | VAE、ELBO、reparameterization、posterior collapse、VQ-VAE、离散码本和自回归生成 | 概率基础链接回机器学习课 | KL 权重和重构质量权衡；码本使用率与 dead code 诊断 |
| 15 | DDPM 前向/反向过程、噪声预测、classifier-free guidance、Latent Diffusion、Flow Matching | 不追求完整图像产品实现 | 一维/二维扩散轨迹；guidance 对质量、多样性和成本的影响 |
| 16 | Unicode、规范化、语料切分、word/character/subword tokenization、BPE、WordPiece、Unigram、SentencePiece | 这是新数据接口，不假设学习者已掌握 | 在中英混合、代码和 emoji 上训练 tokenizer；比较 token 数、UNK 和公平性成本 |
| 17 | distributional hypothesis、Word2Vec、negative sampling、GloVe、内在/外在评估、多义词与偏见 | 不把词向量降维图当作完整评价 | 负采样梯度实验；相似度、类比和下游任务评价冲突 |
| 18 | n-gram 到神经语言模型、teacher forcing、ELMo 与上下文表示、perplexity、exposure bias | RNN 机制只复习接口 | 同一序列比较静态/上下文表示；困惑度与生成质量不一致反例 |
| 19 | encoder-only、decoder-only、encoder–decoder，MLM、causal LM、span corruption，BERT、GPT、T5 | 自注意力结构视为已知 | 同一文本比较三类预训练目标的可见信息与损失位置 |
| 20 | pre-norm、RMSNorm、SwiGLU、RoPE、ALiBi、MQA/GQA、KV Cache、MoE、长上下文与状态空间模型边界 | 不重讲标准 Transformer block | GQA 的 KV 内存计算器；RoPE 长度外推与 MoE 路由负载实验 |
| 21 | 数据获取、清洗、去重、质量过滤、混合权重、污染、token 预算、Kaplan/Chinchilla Scaling Law | 不把更多数据默认视为更好 | 计算最优模型—token 预算台；近重复导致评测污染的审计实验 |
| 22 | SFT 数据合同、prompt template、full fine-tuning、LoRA、QLoRA、adapter 合并与遗忘 | 不重复矩阵分解基础 | LoRA rank/target module/显存对照；量化底座上的误差和收敛检查 |
| 23 | 偏好数据、reward model、RLHF、PPO、DPO、KL 约束、可验证奖励与推理模型 | PPO 基础链接回机器学习课 | 偏好对噪声实验；DPO 温度和 reference policy 敏感性 |
| 24 | greedy、beam、top-k、top-p、temperature、重复控制、speculative decoding、self-consistency、test-time scaling | 不把 Chain-of-Thought 当作事实保证 | 解码树与校准实验；draft/target 接受率对吞吐的影响 |
| 25 | dense/sparse retrieval、embedding、chunk、rerank、RAG、引用、更新与 parametric/non-parametric memory | 不把检索命中等同于答案正确 | Chunk/rerank 消融；可回答、不可回答和证据冲突三类评测 |
| 26 | function calling、JSON Schema、约束解码、ReAct、工具权限、状态、记忆和 Agent 评测 | 不把提示词流程称为自治智能 | 工具调用状态机；错误参数、超时、重复副作用和权限拒绝注入 |
| 27 | CLIP 对齐、视觉编码器接入 LLM、projector、cross-attention、视觉 token、指令微调与统一多模态生成 | 视觉基础链接回第 10–11 讲 | 图像分辨率—视觉 token—延迟计算器；文本捷径与视觉证据冲突测试 |
| 28 | benchmark、数据污染、置信区间、LLM-as-a-judge、切片、红队、幻觉、校准、偏见、隐私与可解释性边界 | 不发布无误差条的单一总分 | 配对 Bootstrap 评测；judge 偏置、顺序偏置和提示注入实验 |
| 29 | CPU/GPU/TPU/NPU，训练/推理硬件，SIMT、Tensor Core、HBM、PCIe/NVLink、CUDA runtime、cuBLAS、cuDNN、NCCL | 不讲成硬件型号营销页 | 算力/带宽/容量约束计算器；训练与 decode 的瓶颈分类 |
| 30 | 算子、kernel、fusion、tiling、memory coalescing、Roofline、profiling，CUDA、Triton、TileLang 的抽象层与取舍 | 不声称三者互相完全替代 | Vector Add、Fused Softmax、GEMM/Dequant GEMM、FlashAttention 四个案例 |
| 31 | data/tensor/pipeline/context/expert parallel，通信集合，activation checkpoint，ZeRO、FSDP、Megatron | 不把多卡数量等同于线性加速 | 模型状态与通信账本；不同并行组合的显存、气泡和 all-reduce 成本 |
| 32 | prefill/decode、continuous batching、PagedAttention、RadixAttention、prefix cache、quantization、structured output、vLLM 与 SGLang | 不以单次离线生成代表在线服务 | 同模型同请求集服务基准；TTFT、TPOT、吞吐、P50/P95/P99 和显存对比 |

## 7. GPU 与 AI Infra 的讲授边界

### 7.1 硬件层

硬件部分回答“模型为什么在某种设备上快或慢”，不做芯片采购排行榜。必须区分：

- 训练偏重吞吐、互联、优化器与激活内存。
- 在线推理同时受权重带宽、KV Cache、batch 形状和尾延迟约束。
- Prefill 通常比 decode 更偏计算密集，decode 更容易受内存带宽和调度影响。
- 峰值 FLOPS 不能代替端到端吞吐；数据类型、稀疏条件和利用率必须一起报告。

### 7.2 CUDA、Triton 与 TileLang

三者不处在完全相同的层级：

| 技术 | 主要抽象 | 优势 | 代价 | 课程案例 |
|---|---|---|---|---|
| CUDA | thread/block、memory hierarchy、runtime 与完整 NVIDIA 生态 | 控制最细、库和工具成熟、可访问新硬件能力 | 代码量和硬件细节最多 | 归约、共享内存、CUDA Graph、Tensor Core |
| Triton | Python 中的 program/block 级张量计算 | 适合快速实现融合算子，编译器承担大量线程映射 | 仍需理解 tile、layout、occupancy 和形状约束 | Vector Add、Fused Softmax、LayerNorm、Matmul |
| TileLang | 基于 TVM IR 的 tile/dataflow、shared/fragment 与 pipeline | 更显式表达数据搬运、GEMM 和流水线，适合复杂 AI kernel | 生态较新，调优与后端能力仍需逐设备验证 | GEMM、Dequant GEMM、FlashAttention、MLA Decode |

所有性能案例先给 PyTorch 或库实现作为正确性基线，再比较自定义 kernel。基准必须包含 warm-up、同步、重复次数、分位数、正确性容差、输入形状和设备信息。

### 7.3 vLLM 与 SGLang

这部分讲推理系统的共同问题和关键设计，不做简单的框架胜负表：

- vLLM 以 PagedAttention 和分页 KV Cache 管理为核心入口。
- SGLang 以 RadixAttention、前缀复用、结构化生成和复杂语言模型程序执行为重要入口。
- 两者都在快速演进，功能与性能结论必须绑定版本、模型、硬件和工作负载。
- 服务实验统一报告 TTFT、TPOT、端到端延迟、请求/秒、token/秒、显存和失败率。

## 8. 练习与项目体系

每讲配置：

- 5–10 道概念、形状、成本或失效条件检查题。
- 至少 1 个代码阅读或日志诊断题。
- 1 个 30–90 分钟的小实验。
- 1 个论文图表复核任务，要求说明实验假设而不是复述结论。

结课项目从以下三条路线中选择一条：

1. **模型路线**：复现并改进一个视觉、生成或语言模型，在冻结预算和评测合同下完成消融。
2. **应用路线**：完成一个 RAG、多模态或工具调用系统，重点评价证据、失败率、成本和安全边界。
3. **系统路线**：完成一个 GPU 算子、训练并行或推理服务项目，重点评价正确性、性能、可扩展性和瓶颈。

所有项目必须提交环境锁定、数据来源、训练/推理配置、随机种子、原始指标、失败样本、成本估计和复现步骤。只提交演示视频或最佳样本不构成完成。

## 9. 实施与依赖策略

课程正文优先使用当前站点的 Vue、SVG 和 Canvas 完成交互。训练代码以 PyTorch 为主，但本阶段不新增大型依赖；每个模块开始实施时再根据真实实验决定环境文件，并先记录依赖选择理由。

建议把运行环境拆成三层：

- 核心环境：Python、NumPy、PyTorch、torchvision 和基础数据工具。
- NLP/LLM 环境：tokenizers、transformers、datasets、evaluate 和 PEFT，按章节需要加入。
- GPU/Infra 环境：CUDA、Triton、TileLang、vLLM、SGLang，使用独立 Linux GPU 环境和锁定镜像，不要求网站构建机安装。

不同 GPU 架构、CUDA 版本和驱动组合不能共享未经验证的性能结论。课程提供的数值结果必须保留环境指纹，并把无法在标准构建环境复验的 GPU 实验明确标为“硬件环境验证”。

## 10. 版权与资料使用边界

第三方 slides 和论文下载到 `ref/dl101/`，该目录已被 `.gitignore` 排除，只作为本地研究资料。公开网站不得直接提供这些文件的下载链接或重新分发副本，除非逐项确认许可允许这样做。

公开内容采用以下方式：

- 链接课程主页、原始 PDF 或论文出版页。
- 记录实际参考的讲次、页码、论文与访问日期。
- 独立撰写中文解释并制作原创图形、数据、题目、代码和交互。
- 不复制 slides 的版式、插图、动画顺序、课堂案例、题目或连续表达。
- 第三方图像即使出现在公开 slides 中，也不能推定具有再分发许可。

课程资料同步脚本会记录 URL、重定向结果、文件大小、页数和 SHA-256，但这些技术记录不构成版权许可。许可状态必须在真正使用具体表达前单独核对。

## 11. 完成标准

只有同时满足以下条件，课程才能从“内容建设中”改为“已开放”：

- 32 个正式章节全部具有原创正文、实验、测验和来源登记。
- 8 次综合作业和结课项目具有清晰输入、评价合同和完成标准。
- 正文代码在声明的环境中运行通过，GPU 实验具有环境指纹和原始日志。
- 课程结构检查、内部链接检查、站点构建和差异检查通过。
- 桌面端和移动端没有页面级横向溢出，复杂图只在自身容器内滚动。
- 每章的论文引用、第三方资料使用和版权边界完成复核。
