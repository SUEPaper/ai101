---
course: deep-learning
title: 深度学习与大模型八次综合作业
description: 八次从模型目标、数据、实现到质量、成本与失败分析的综合作业，覆盖训练、视觉、生成、语言模型、后训练和 AI Infra。
outline: [2, 3]
---

# 八次综合作业

八次作业要求完成“合同 → 基线 → 实现 → 数值/系统验证 → 消融 → 失败分析 → 发布判断”闭环。目标不是复刻最大模型，而是在有限 CPU、单张消费级 GPU 或模拟环境中给出可审计证据。每次作业对应课程章节，但题目、中文说明、数据合同和验收均为本站独立设计。

## 统一提交与运行合同

每次提交以下目录，名称不得省略：

```text
assignment-Ax/
├── README.md              # 环境、数据、命令、预计时间与硬件
├── report.pdf             # 6–10 页，正文中文
├── src/                   # 核心实现，不把 Notebook 当唯一源码
├── notebooks/             # 探索与可视化，可从头运行
├── configs/               # 基线/候选/消融配置
├── tests/                 # 形状、数值、确定性或协议回归
├── results/raw/           # 原始 JSON/CSV，不只交截图
├── results/figures/       # 由原始结果生成的图
└── environment.*          # lockfile 或容器定义
```

统一要求：

1. 固定数据 snapshot、训练/验证/冻结测试切分和随机种子；任何样本删除都留下 manifest。
2. 先运行简单基线，再运行候选；共享 tokenizer、预处理、预算和评价合同。
3. 至少 3 个随机种子；生成/后训练任务若成本受限，至少报告采样不确定性并说明限制。
4. 每个关键结论能由 `results/raw/` 的表格或断言复核，不能只交最好截图。
5. 记录 wall time、峰值内存/显存、训练/推理 token 或 FLOPs 代理、设备与软件版本。
6. 主动制造至少一个失败条件，解释机制并给出是否发布的判断。
7. 使用外部代码、模型、数据或图时写明 URL、版本、许可和修改；不得提交第三方课程答案。

### 统一评分量规

| 维度 | 权重 | 达标证据 |
|---|---:|---|
| 问题、数据与信息边界 | 15% | 输入/输出、切分、mask、来源、许可和泄漏风险清楚 |
| 模型/系统推导 | 20% | 目标、形状、状态或成本公式与实现逐项对应 |
| 实现与正确性 | 20% | 核心测试、数值对照、边界输入和可复现命令通过 |
| 实验设计与统计 | 20% | 基线、预算公平、种子/配对不确定性和消融完整 |
| 失败、安全与工程边界 | 15% | 可复现反例、风险切片、成本与发布门禁 |
| 报告、可视化与引用 | 10% | 图表有单位/数据表替代，结论和来源可核对 |

只要出现冻结测试参与选型、无法从空环境运行、伪造/选择性删除失败结果或未声明使用他人答案，本次作业不得超过 59 分。

## A1 · AdamW 与训练系统消融

对应 [第 01–03 章](./01-adam-adamw.md) 。目标是在固定 step/token 预算下训练一个小型 MLP、CNN 或 Transformer，并证明训练系统而非模型名导致的差异。

### 必做任务

1. 用同一初始权重和 gradient trace 比较 Adam+L2 与 AdamW；逐步记录参数范数、$m_t$、$v_t$、effective step 和解耦衰减。
2. 比较 constant、warmup+cosine 和 one-cycle 中至少两种调度，固定总 optimizer steps。
3. 比较一种归一化/初始化或 Pre/Post-Norm 选择；记录逐层 activation/gradient RMS。
4. 在支持的设备上比较 FP32 与 BF16/FP16；FP16 必须制造 underflow/overflow，并展示 static/dynamic loss scaling 的恢复或失败。
5. 做 overfit-one-batch、checkpoint 中断恢复和数据泄漏故障注入。

### 必交证据与验收

- AdamW state update 的小数组手算与 NumPy 对照，误差小于 $10^{-10}$。
- 同预算 loss/主指标、参数范数、梯度范数、tokens/s、峰值显存和 3 种子区间。
- 从 checkpoint 恢复后的下一步 loss/学习率/参数与不中断基线在定义容差内一致。
- 一页“曲线指纹 → 假设 → 下一检查”的诊断记录。

## A2 · ResNet 分类、迁移学习与设备预算

对应 [第 04–05 章](./04-classic-vision-architectures.md) 。使用 CIFAR-10/100、Tiny ImageNet 子集或自建许可明确的小型图像集；单卡上限建议 8GB，训练不超过 2 GPU-hours。

### 必做任务

1. 实现 basic residual block，并与 plain VGG-style block 在近似参数/FLOPs 下比较优化曲线。
2. 建立 zero-rule/线性分类或小 CNN 基线，再训练 ResNet-18 级候选。
3. 选择 MobileNet/ConvNeXt tiny 之一做精度—延迟—峰值激活对照；不得只比较参数量。
4. 比较 frozen backbone、partial unfreeze、full fine-tune；所有预处理只从训练集拟合。
5. 对分辨率、类别频率、模糊/噪声和域外采集至少四个切片做错误审计。

### 必交证据与验收

- 每 stage 张量形状、参数、MACs/FLOPs 口径和 activation bytes 表。
- Residual gate 置零后等价恒等路径的数值断言。
- 同 batch/设备下 warm-up 后 P50/P95 推理延迟与峰值内存；说明预处理是否计时。
- 冻结测试只在最终一次运行，报告总体与切片置信区间。

## A3 · 目标检测或分割的密集预测合同

对应 [第 06–09 章](./06-two-stage-detection.md) 。在检测路线和分割路线中选择一条，但必须同时完成数据几何、损失、指标反例和错误类型分析。

### 检测路线

- 比较 Faster R-CNN 与 RetinaNet/小型 YOLO 或 DETR 中一个候选。
- 可视化 anchor/point/query 匹配、NMS 或一对一匹配；手算至少一个 IoU/AP 小例。
- 报告 AP50、AP50:95，并按 small/medium/large、遮挡和类别切片；不能只报 mAP。

### 分割路线

- 比较 U-Net/FCN 与 DeepLab 或 Mask R-CNN 中一个候选。
- 明确 resize/crop 后 mask 插值、ignore index 和 class mapping；检查空前景 batch。
- 报告 IoU/Dice、边界指标和像素准确率反例；实例任务再报 mask AP/PQ。

### 共通门禁

提交 20 个按 FP/FN/localization/class/merge/split 分类的失败样本，附原始标注与预测；说明至少一个标注歧义。所有可视化同时输出机器可读坐标/掩码摘要。

## A4 · ViT、Swin 与视觉表征

对应 [第 10–11 章](./10-vision-transformer.md) 。目标是区分 patch/window 成本、数据效率和预训练表示，而不是训练大规模 foundation model。

### 必做任务

1. 对 224/384 分辨率与 patch 8/16/32 计算 token、attention score 元素、FLOPs 和峰值激活。
2. 训练 tiny ViT，比较绝对位置插值、无位置或另一种位置策略；记录非训练分辨率迁移。
3. 实现 Swin window partition/reverse 与 shifted attention mask 的可逆/无泄漏测试。
4. 选择 SimCLR、MAE 或 DINO 做小规模预训练；同 backbone 比较 random init、linear probe、full fine-tune。
5. 以 CNN 基线做低数据 1%/10%/100% 训练切片。

### 必交证据与验收

- patchify/unpatchify 与 window reverse 精确恢复断言。
- Attention 全局/窗口成本推导与 profiler 实测的差异解释。
- 表征 kNN/linear probe、下游微调、训练吞吐与显存四类证据。

## A5 · 生成模型的覆盖、质量与记忆

对应 [第 12–15 章](./12-gan-dcgan.md) 。在 GAN/WGAN、VAE/VQ-VAE、Diffusion/Flow 中至少实现两条路线，其中一条必须在二维混合分布上从零实现以便审计覆盖。

### 必做任务

1. 在二维 8-mode mixture 上比较样本覆盖、质量和训练动态；记录每 mode 质量/数量，不用单张散点图代替统计。
2. GAN 路线注入判别器过强/模式坍缩；WGAN 路线比较 clipping 与 gradient penalty 或做 Lipschitz 代理审计。
3. VAE 路线扫描 $\beta$ 并报告重构、KL、active units；VQ 路线报告 codebook perplexity/dead codes。
4. Diffusion/Flow 路线画不同 timestep/SNR 或 ODE 步数轨迹，比较质量—多样性—采样步数。
5. 做最近训练样本、membership/duplicate 或 leave-one-group-out 记忆检查。

### 发布门禁

FID 只能作为一项指标，并说明 feature extractor/样本数；至少同时报告 precision/recall 或 mode coverage、最近邻记忆和多种子。不能手挑“最好样本板”。

## A6 · Tokenizer 与 Tiny GPT

对应 [第 16–21 章](./16-tokenization.md) 。从原始文本到训练完成一个 5M–30M 参数的 causal LM；CPU 可完成 tokenizer/数据审计，模型建议单张消费级 GPU 不超过 6 GPU-hours。

### 数据与 tokenizer

1. 建立许可明确、总量可控的中英/代码或领域语料，保留 source、time、hash 和 split manifest。
2. 实现最小 BPE merge learner，并与 SentencePiece/Hugging Face tokenizer 对照。
3. 按语言、emoji、数字、代码、罕见术语报告 bytes/token、tokens/char、UNK/fallback 和 round-trip。
4. 做 exact/near-duplicate 与 benchmark contamination 审计，冻结 validation/test 后不得参与 merge/词表选择。

### Tiny GPT

1. 实现 causal/loss mask、weight tying（可选）、Pre-Norm/RMSNorm、RoPE 或绝对位置中的一种；所有张量写形状。
2. 用固定 token 预算比较两个模型规模或两个数据配比；记录参数、训练 FLOPs 代理、MFU/tokens/s、loss 和 PPL。
3. 验证 label shift、padding、document boundary；单样本 overfit 后再长跑。
4. 在相同 tokenizer/解码下评价 PPL、bits-per-byte、重复、终止和领域切片。

### 验收

最小训练配置 50 step 内 loss 明显下降；causal mask 无未来泄漏；checkpoint 恢复一致；测试集无 exact training duplicate；模型卡明确不能把 Tiny GPT 用作事实系统。

## A7 · LoRA、RAG 与大模型评估

对应 [第 22–28 章](./22-sft-lora-qlora.md) 。在以下两条路线中选择一条，均需小模型/开放权重、许可合规和冻结评测；不要求购买云 API。

### 路线 L · LoRA/QLoRA 适配

1. 构建至少 500 条、含明确 train/validation/frozen-test 的指令数据，审计模板和 assistant-only loss mask。
2. 比较 base、LoRA 两个 rank/target 配方、QLoRA 一个配方；记录 base hash、rank/alpha/targets、量化类型/group/compute dtype。
3. 报告领域成功、格式、安全拒答、通用回归、显存、tokens/s 和 adapter 合并前后误差。

### 路线 R · RAG 系统

1. 建立版本化文档索引，包含 chunk、metadata、embedding、权限和 tombstone。
2. 比较 sparse、dense、hybrid 中至少两种；消融 chunk/overlap/top-k/rerank。
3. 构造可回答、不可回答、证据冲突、恶意文档注入四类评测；逐 claim 核对引用 span。

### 两条路线共同要求

使用 paired bootstrap 或适当配对统计；测试至少一种 LLM judge 换序偏差并用人工子集校准；报告质量、P95 延迟、token/费用、失败和安全。训练/索引数据不得包含冻结题及答案。

## A8 · GPU 算子与推理服务

对应 [第 29–32 章](./29-hardware-cuda.md) 。无 NVIDIA GPU 时可完成 Roofline/状态模拟器、CPU 正确性基线和公开 trace replay；有 GPU 时再做 Triton/TileLang/CUDA 和 vLLM/SGLang 实测。不能伪造硬件结果。

### Part I · 算子

1. 从 Vector Add、Fused Softmax、LayerNorm、GEMM/Dequant 中选一个；先写 PyTorch/库基线和高精度 reference。
2. 用 Triton、TileLang 或 CUDA 实现；解释 program/thread、tile、shared/register、mask 和数值稳定。
3. 覆盖典型、非整除、小 shape、极值和不同 stride；前向相对/绝对容差通过，若有反向也做梯度对照。
4. 用 GPU event、warm-up、同步和至少 100 次报告 P50/P95、GB/s 或 TFLOP/s、峰值内存；给 profiler 瓶颈。

### Part II · 服务

1. 用同一模型/量化/硬件/请求 trace 比较 vLLM 与 SGLang；若某框架不支持，记录能力门禁而非换模型。
2. 请求集包含短/长 prompt、短/长 output、共享前缀、结构化输出和取消；固定解码与输出质量检查。
3. 做 open-loop 到达率扫描，报告 request/s、input/output token/s、TTFT/TPOT/E2E P50/P95/P99、显存、cache hit 和失败率。
4. 注入超长请求、过载、取消和 OOM 临界，检查 admission/backpressure 与 KV 回收。

### 最终验收

微基准加速必须同时报告端到端收益；吞吐提高若使 SLO/质量失败，不判为改进。所有 CUDA/Triton/TileLang/vLLM/SGLang 结论绑定 commit/version、driver、GPU 和 shape/workload。

## 迟交、协作与学术诚信

可以讨论概念、调试策略和公开论文，但每位学习者必须独立完成推导、实现、实验和报告。两人项目式协作需在提交前登记，并在 commit/报告中逐项说明分工；共同代码仍须双方能解释。

使用生成式 AI 辅助时，应在 `README.md` 记录工具、日期、用途和人工验证：可以用于解释错误、提出测试或语言润色，不能直接提交未理解的代码/推导，也不能让工具读取未授权私有数据。任何结论都由提交者负责。

## 作业完成检查

- [ ] 从干净环境按 README 命令可运行，随机种子与数据 manifest 固定。
- [ ] 冻结测试未参与调参，原始结果未选择性删除。
- [ ] 关键形状、数值、mask、协议或服务状态有自动测试。
- [ ] 质量、成本、不确定性和失败切片同时报告。
- [ ] 模型、数据、代码、论文和图片来源/许可完整。
- [ ] 报告明确给出“发布/不发布/需补证据”判断。
