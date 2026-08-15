export type DlLessonStatus = 'available' | 'planned'

export interface DlLesson {
  number: string
  title: string
  href: string
  status: DlLessonStatus
}

export interface DlModule {
  title: string
  summary: string
  lessons: DlLesson[]
}

const lesson = (number: string, title: string, slug: string, status: DlLessonStatus = 'planned'): DlLesson => ({
  number,
  title,
  href: `/courses/deep-learning/${slug}`,
  status
})

export const dlModules: DlModule[] = [
  {
    title: '一 · 现代训练',
    summary: '先把优化器、数值格式和训练诊断连成一套可观测系统。',
    lessons: [
      lesson('01', 'Adam、AdamW 与学习率调度', '01-adam-adamw', 'available'),
      lesson('02', '初始化、归一化与稳定训练', '02-stable-training', 'available'),
      lesson('03', '数据增强、迁移学习与实验诊断', '03-transfer-diagnostics', 'available')
    ]
  },
  {
    title: '二 · 计算机视觉',
    summary: '从经典深层网络走到检测、分割和视觉 Transformer。',
    lessons: [
      lesson('04', 'AlexNet、VGG、GoogLeNet 与 ResNet', '04-classic-vision-architectures', 'available'),
      lesson('05', 'MobileNet、EfficientNet 与 ConvNeXt', '05-efficient-convnets', 'available'),
      lesson('06', 'R-CNN 系列与两阶段目标检测', '06-two-stage-detection', 'available'),
      lesson('07', 'YOLO、RetinaNet 与 DETR', '07-one-stage-detection-detr', 'available'),
      lesson('08', 'FCN、U-Net 与语义分割', '08-semantic-segmentation', 'available'),
      lesson('09', '实例分割、全景分割与 SAM', '09-instance-panoptic-sam', 'available'),
      lesson('10', 'Vision Transformer（ViT）与 DeiT', '10-vision-transformer', 'available'),
      lesson('11', 'Swin Transformer 与视觉自监督学习', '11-swin-self-supervised', 'available')
    ]
  },
  {
    title: '三 · 生成模型',
    summary: '比较对抗、潜变量、扩散与流模型怎样表示数据分布。',
    lessons: [
      lesson('12', 'GAN 与 DCGAN', '12-gan-dcgan', 'available'),
      lesson('13', 'WGAN、StyleGAN 与生成质量评估', '13-wgan-stylegan', 'available'),
      lesson('14', 'VAE、VQ-VAE 与自回归生成', '14-vae-vqvae-autoregressive', 'available'),
      lesson('15', 'Diffusion、Latent Diffusion 与 Flow Matching', '15-diffusion-flow-matching', 'available')
    ]
  },
  {
    title: '四 · NLP 基础',
    summary: '先把文本变成可信 token，再研究静态与上下文表示。',
    lessons: [
      lesson('16', '文本数据、分词与子词建模', '16-tokenization', 'available'),
      lesson('17', 'Word2Vec、GloVe 与词向量评估', '17-word-embeddings', 'available'),
      lesson('18', '神经语言模型与上下文表示', '18-neural-language-models', 'available')
    ]
  },
  {
    title: '五 · 大语言模型',
    summary: '把架构、数据、后训练、检索、工具和评估放进同一系统。',
    lessons: [
      lesson('19', 'BERT、GPT 与 T5 预训练范式', '19-pretraining-paradigms', 'available'),
      lesson('20', '现代 LLM 架构、MoE 与长上下文', '20-modern-llm-architecture', 'available'),
      lesson('21', '预训练数据工程与 Scaling Law', '21-data-scaling-laws', 'available'),
      lesson('22', '指令微调、LoRA 与 QLoRA', '22-sft-lora-qlora', 'available'),
      lesson('23', 'RLHF、PPO、DPO 与推理模型', '23-rlhf-dpo-reasoning', 'available'),
      lesson('24', '解码、投机解码与 Test-Time Scaling', '24-decoding-test-time-scaling', 'available'),
      lesson('25', 'RAG、向量检索与知识边界', '25-rag', 'available'),
      lesson('26', '工具调用、结构化输出与 Agent', '26-tools-agents', 'available'),
      lesson('27', '多模态大模型与统一表示', '27-multimodal', 'available'),
      lesson('28', '大模型评估、可靠性、安全与可解释性', '28-evaluation-safety', 'available')
    ]
  },
  {
    title: '六 · GPU 与 AI Infra',
    summary: '解释硬件、算子、并行和服务调度怎样决定真实成本。',
    lessons: [
      lesson('29', 'AI 训练硬件、推理硬件与 CUDA 生态', '29-hardware-cuda', 'available'),
      lesson('30', 'AI 算子与 CUDA、Triton、TileLang', '30-cuda-triton-tilelang', 'available'),
      lesson('31', '分布式训练、ZeRO、FSDP 与 Megatron', '31-distributed-training', 'available'),
      lesson('32', 'vLLM、SGLang 与大模型推理服务', '32-vllm-sglang', 'available')
    ]
  }
]

export const dlLessons = dlModules.flatMap((module) => module.lessons)
export const dlAvailableLessons = dlLessons.filter((item) => item.status === 'available')

export function normalizeDlPath(path: string) {
  return path.replace(/\.html$/, '').replace(/\/$/, '')
}
