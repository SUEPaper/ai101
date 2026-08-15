---
course: deep-learning
edition: industrial
title: 现代 LLM 架构、MoE 与长上下文
description: 从 Pre-Norm、RMSNorm、SwiGLU、RoPE 进入 GQA/KV Cache、FlashAttention、稀疏专家和长上下文有效性。
outline: [2, 3]
---

<DlLessonHero number="20" module="大语言模型" title="现代 LLM 架构、MoE 与长上下文" question="现代 decoder block 的细节怎样影响稳定性、每 token FLOPs、KV 显存、通信和真正可用的上下文长度？" duration="建议 330–430 分钟" :outcomes="['解释 Pre-Norm、RMSNorm 与 SwiGLU 的信号和参数接口', '计算 MHA/MQA/GQA 的 KV Cache 容量', '区分 FlashAttention 的精确 IO 优化与近似稀疏注意力', '建立 MoE 路由、容量、负载和 expert-parallel 账本', '用检索位置、冲突证据和外推切片评价长上下文']" />

本章不重复标准 Transformer block，而研究现代 LLM 真正改变成本与训练行为的结构细节。每个“更长、更大、更稀疏”的主张都必须落到形状、字节、通信或冻结任务证据。

## 贯穿案例：7B 维修助手从 8k 扩到 32k

团队希望批量读取整本设备手册。我们先算单请求 KV Cache，再决定 GQA、RoPE 外推和推理并发；同时用答案位置、干扰证据和跨文档引用切片验证模型是否真的使用 32k，而不是只支持输入不报错。

<DlQuiz kind="先判断" question="模型配置写 32k context 是否证明能可靠利用第 1k 位置的证据回答第 32k 问题？" :options="['证明', '不证明，需位置、冲突和任务切片验证有效上下文', '只要显存够就证明', 'RoPE 会自动保证']" :answer="1" explanation="可接受输入长度只是接口上限。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>列出 32 层、32 Q heads、8 KV heads、head dim 128、8k token、BF16 的单序列 KV 字节；再乘并发。</div>

## 1. Block 现代化：Pre-Norm、RMSNorm、SwiGLU

Pre-Norm 将归一化放在 attention/MLP 前，使残差主干保留更直接的恒等梯度路径。RMSNorm 只按均方根缩放，不减均值：$y=x/\sqrt{\mathrm{mean}(x^2)+\epsilon}\odot g$。SwiGLU 用门控分支 $\mathrm{SiLU}(xW_g)\odot(xW_u)$ 再投影，通常调整中间宽度保持参数预算。

```python
import numpy as np

def rmsnorm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + eps)

x = np.array([[1., 2., 3., 4.]])
y = rmsnorm(x)
assert np.allclose(np.mean(y*y, axis=-1), 1, atol=1e-5)
```

<DlQuiz kind="归一化" question="RMSNorm 与 LayerNorm 的直接区别是什么？" :options="['RMSNorm 不做均值中心化，只按均方根缩放', 'RMSNorm 没有 epsilon', 'RMSNorm 只用于 CNN', 'LayerNorm 不可学习 scale']" :answer="0" explanation="两者的统计和不变性不同，不能只改类名。" />

## 2. RoPE、ALiBi 与位置外推

RoPE 对每对通道施加位置相关旋转，使 query-key 点积带相对位置相位；ALiBi 直接给注意力分数加与距离成比例的 head-specific bias。超过训练长度时，位置插值、NTK-aware scaling 等会改变频谱，不保证无损外推。

```python
import math

def rotate_pair(x0, x1, position, theta=.01):
    c, s = math.cos(position*theta), math.sin(position*theta)
    return x0*c-x1*s, x0*s+x1*c

u = rotate_pair(1., 0., 10)
assert math.isclose(u[0]**2 + u[1]**2, 1.0)
```

<DlQuiz kind="位置" question="RoPE 旋转保持了每对通道的什么量？" :options="['二维范数', 'token 数量', 'KV head 数', 'MLP 参数总数']" :answer="0" explanation="旋转保持范数，同时让点积包含相对相位。" />

## 3. MHA、MQA、GQA 与 KV Cache

<KvCacheLab />

decode 时历史 K/V 可缓存，避免重算；每层每 token 需要 K 和 V 两份状态。MHA 每个 query head 各有 KV，MQA 所有 Q 共享一组 KV，GQA 介于两者。减少 KV heads 显著节省容量和带宽，但可能降低表示容量，且 checkpoint 转换需分组平均或再训练。

<DlQuiz kind="KV" question="GQA 从 32 个 KV heads 降到 8 个，其他不变时 cache 约缩小多少？" :options="['4 倍', '2 倍', '32 倍', '不变']" :answer="0" explanation="KV 容量对 KV head 数线性。" />

## 4. FlashAttention：IO-aware 的精确注意力

FlashAttention 通过 tiling 和 online softmax 避免把完整 $T\times T$ score 矩阵写回 HBM；它计算与标准 dense attention 数学等价的结果（容差内），不是稀疏或低秩近似。加速取决于形状、dtype、mask、硬件和 kernel 版本。

```python
def score_bytes(tokens, bytes_per=2):
    return tokens*tokens*bytes_per

assert score_bytes(8192) == 134_217_728  # 单 head 一份 128 MiB score
assert score_bytes(16384) == 4*score_bytes(8192)
```

<DlQuiz kind="FlashAttention" question="FlashAttention 的核心收益来自哪里？" :options="['把 dense attention 改成随机采样', '减少 HBM 中间读写并融合分块 softmax', '删除 softmax', '减少模型层数']" :answer="1" explanation="它是 IO-aware 精确算法。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>为什么理论 FLOPs 相同的两种 attention 实现墙钟可能差数倍？用 HBM 读写、SRAM tile、占用率和重算回答。</div>

## 5. MoE：总参数、激活参数与通信分开算

<MoeRouterLab />

router 为每个 token 选择 top-k experts。总参数可增大而每 token 只激活少量 FFN；但 expert parallel 引入 all-to-all，负载倾斜会造成热点、padding 或 token dropping。容量因子越大越不易溢出，却浪费计算；auxiliary loss 可能与主任务梯度冲突。

<DlQuiz kind="MoE" question="MoE 的总参数翻八倍是否意味着每 token FLOPs 必然翻八倍？" :options="['必然', '不必然，取决于 top-k 激活；但通信和路由成本可能上升', 'FLOPs 会归零', '只与词表有关']" :answer="1" explanation="总容量、激活计算与通信必须分别报告。" />

## 6. 长上下文与状态空间边界

长上下文评价至少包含不同答案位置、相似干扰、矛盾证据、多跳引用、真实长文和延迟/显存。needle-in-a-haystack 是必要非充分测试。滑窗、稀疏注意力、压缩记忆和 SSM 改变信息路由；SSM 具有线性序列扫描优势，但内容寻址、并行训练、状态容量和硬件 kernel 决定实际取舍。

<div class="dl-stop"><strong>停下来推一推 3</strong>为“整本手册问答”设计 2k/8k/32k 的位置×干扰×冲突三维矩阵，并同时记录 TTFT、峰值 KV 和准确率。</div>

## 7. 推荐论文阅读

<DlPaperCard title="LLaMA: Open and Efficient Foundation Language Models" meta="Touvron et al. · 2023" priority="必读 · 现代 decoder 配方" question="RMSNorm、SwiGLU、RoPE 与训练 token 选择怎样组成配方？" reading="读架构与训练设置，核对参数、token、上下文和数据来源，不只比较榜单。" href="https://arxiv.org/pdf/2302.13971" />

<DlPaperCard title="GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" meta="Ainslie et al. · 2023" priority="必读 · KV 状态" question="怎样从 MHA checkpoint 构造 GQA，质量—速度权衡如何评价？" reading="精读 uptraining、group 数与推理结果；复算 KV 容量。" href="https://arxiv.org/pdf/2305.13245" />

<DlPaperCard title="FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" meta="Dao et al. · 2022" priority="必读 · 算法—硬件" question="为什么 IO complexity 而非只看 FLOPs 能解释加速？" reading="精读 tiling、online softmax 与 HBM 访问分析；第 30 章再看 kernel。" href="https://arxiv.org/pdf/2205.14135" />

## 8. 练习与门禁

实现 RMSNorm/RoPE；复算 KV；比较 MHA/GQA；审计 MoE overflow；长上下文位置切片；记录 kernel/dtype；区分总参数、激活参数、FLOPs、字节和延迟。

## 本章小结

- 现代 block 的归一化、门控和位置表示共同决定稳定性与外推。
- GQA 降低 KV 容量；FlashAttention 降低 IO；MoE 增加参数容量但引入路由与通信。
- 最大 context 不是有效 context，必须用位置、冲突、成本联合验证。

<DlLessonProgress lesson-id="20" next-href="/courses/deep-learning/21-data-scaling-laws" next-label="下一章：数据工程与 Scaling Law" />
