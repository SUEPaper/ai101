---
course: deep-learning
title: 深度学习与大模型课程来源与推荐阅读
description: 登记课程参考课件、推荐论文、本地资料快照、复核方法和独立创作边界。
outline: [2, 3]
---

# 深度学习与大模型课程来源与推荐阅读

## 1. 资料快照

截至 2026-08-14，本地参考资料库共有 **147 份 PDF、9565 页、899490175 字节**，其中包括 **86 份公开课件**和 **61 篇推荐论文**。全部文件已经通过 PDF 文件头、可解析页数和 SHA-256 校验，失败数为 0。

第三方 PDF 只保存在被 Git 忽略的 `ref/dl101/` 中，用于备课、页码复核和离线检索，不随网站发布。公开仓库只保存来源地址、资料目录和同步程序：

- 机器可读目录：`scripts/data/dl101-resources.json`
- 同步与校验程序：`scripts/sync_dl101_resources.py`
- 本地审计清单：`ref/dl101/manifest.json`，包含来源 URL、文件路径、页数、字节数、SHA-256 和抓取状态。

## 2. 参考课程与课件状态

| 课程 | 学期 | 已发现课件 | 使用方式 |
|---|---|---:|---|
| [Stanford CS231n](https://cs231n.stanford.edu/schedule.html) | Spring 2026 | 17 | 视觉架构、检测、分割、生成模型和视觉表征的主题覆盖与论文入口 |
| [Stanford CS224n](https://web.stanford.edu/class/cs224n/) | Winter 2026 | 17 | 词向量、预训练、后训练、RAG、Agent、评估和推理系统的主题覆盖 |
| [CMU Advanced NLP](https://cmu-l3.github.io/anlp-spring2026/) | Spring 2026 | 23 | Tokenization、现代 LLM、数据、并行、推理、评估与安全的补充视角 |
| [CMU 11-785](https://deeplearning.cs.cmu.edu/F26/index.html) | Fall 2026 | 0 | 2026-08-14 检查时课程表已发布日期和主题，但尚未发布 slides；保留自动复查入口 |
| [CMU 11-785](https://deeplearning.cs.cmu.edu/S26/index.html) | Spring 2026 补充 | 29 | 在 Fall 2026 slides 发布前补足训练、视觉、序列和生成模型参考，始终明确标为春季补充版本 |

“已发现课件”只统计课程页面直接公开的 lecture slides，不把作业、项目、复习课或外部链接重复计入。CMU Fall 2026 的 0 份不是下载失败，也不会用 Spring 2026 文件冒充；后续重新运行同步程序时会按原课程页面自动复查。

## 3. 推荐论文

标为“必读”的论文用于课前或课后精读；“选读”用于比较方法演进与工程边界；“前置复习”只用于唤醒机器学习课已经讲过的概念。本清单给出原始公开地址，本地缓存不对外分发。

### 3.1 现代训练

- 必读：[Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980)
- 必读：[Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101)
- 选读：[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167)

### 3.2 计算机视觉

- 必读：[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- 必读：[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)
- 必读：[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)
- 选读：[A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545)
- 必读：[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497)
- 必读：[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640)
- 必读：[End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872)
- 必读：[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038)
- 必读：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)
- 选读：[Mask R-CNN](https://arxiv.org/pdf/1703.06870)
- 选读：[Segment Anything](https://arxiv.org/pdf/2304.02643)
- 必读：[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)
- 必读：[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030)
- 必读：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)
- 选读：[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709)
- 选读：[Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.14294)
- 选读：[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377)

### 3.3 生成模型

- 必读：[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661)
- 选读：[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434)
- 必读：[Wasserstein GAN](https://arxiv.org/pdf/1701.07875)
- 选读：[Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028)
- 选读：[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948)
- 必读：[Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114)
- 选读：[Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937)
- 必读：[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)
- 选读：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)
- 选读：[Flow Matching for Generative Modeling](https://arxiv.org/pdf/2210.02747)

### 3.4 NLP 基础

- 必读：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781)
- 选读：[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546)
- 必读：[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
- 必读：[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909)
- 选读：[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226)

### 3.5 大语言模型

- 前置复习：[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
- 必读：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)
- 必读：[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- 选读：[Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683)
- 选读：[Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467)
- 必读：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864)
- 选读：[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)
- 必读：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
- 必读：[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971)
- 必读：[Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361)
- 必读：[Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)
- 必读：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- 必读：[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)
- 必读：[Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)
- 必读：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)
- 选读：[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948)
- 必读：[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401)
- 必读：[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629)
- 选读：[Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/pdf/2302.04761)
- 选读：[Holistic Evaluation of Language Models](https://arxiv.org/pdf/2211.09110)

### 3.6 GPU 与 AI Infra

- 必读：[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053)
- 必读：[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054)
- 必读：[Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- 选读：[TileLang: A Composable Tiled Programming Model for AI Systems](https://arxiv.org/pdf/2504.17577)
- 必读：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)
- 必读：[SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/pdf/2312.07104)

## 4. 同步与复核方法

安装 `pypdf` 后，在仓库根目录运行：

```bash
python3 scripts/sync_dl101_resources.py
```

只复核现有缓存的文件头、页数和 SHA-256，不访问网络：

```bash
python3 scripts/sync_dl101_resources.py --verify-only
```

需要重新下载已有文件时显式添加 `--refresh`。同步程序不会替代人工来源判断：课程页面结构、论文版本、撤稿或许可证变化仍需在引用前复查。

## 5. 使用与版权边界

1. 公开课件和论文只用于核对知识范围、术语、公式、实验结论与原始引用。
2. 不复制第三方 slides 的页面布局、叙述顺序、图表、截图、习题、讲者备注或大段文字。
3. 课程中文正文、案例、数据、图形、代码、测验、实验和作业必须独立创作；需要复用第三方素材时，先确认许可证并在素材附近精确署名。
4. 每个正式章节完成时登记实际使用的论文页码或章节；只列入资料库但未实际使用的论文，不写成正文依据。
5. 本站许可不自动覆盖第三方资料，第三方作品继续受其各自版权和许可证约束。
