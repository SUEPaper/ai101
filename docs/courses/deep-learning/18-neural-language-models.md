---
course: deep-learning
edition: industrial
title: 神经语言模型与上下文表示
description: 从序列概率分解进入前馈、循环与双向语言模型，理解困惑度、teacher forcing、上下文表示、ELMo 与 BERT 之间的接口变化。
outline: [2, 3]
---

<DlLessonHero number="18" module="NLP 基础" title="神经语言模型与上下文表示" question="语言模型怎样把一句话的联合概率拆成可训练的逐 token 预测，又怎样从‘一个词一个向量’走向依赖整句的上下文表示？" duration="建议 300–400 分钟" :outcomes="['用链式法则定义自回归语言模型、NLL 与 perplexity', '区分固定窗口前馈模型、RNN/LSTM 状态和 Transformer 上下文接口', '解释 teacher forcing、暴露偏差、因果目标与掩码语言模型目标', '从 ELMo 的层混合连接到 BERT 的双向上下文表示', '用 tokenizer、mask、数据切分和校准约束语言模型评价']" />

第 16 章把原始文本变成 token，第 17 章给每个词表项一个静态向量。本章再改变一次接口：模型不再只回答“这个词是什么”，而要回答“在当前上下文和当前位置，这个 token 表示什么、下一 token 有多可能”。我们不重复机器学习课中的 RNN 展开和自注意力基础推导，而聚焦语言建模目标、上下文状态、评价协议与失效条件。

## 贯穿案例：设备维修记录的补全、检索与风险提示

输入“主泵振动升高，建议立即……”，生成系统应给“停机检查”较高概率；编码系统则要判断“泵”在具体句中指设备而非动作。两项任务可以共用文本和参数，却有不同的信息边界：在线生成只能读取左侧，离线分类可以读取整段。若评价时把未来 token 泄漏给生成模型，结果再高也无效。

<DlQuiz kind="接口" question="为什么静态词向量不足以区分‘河岸’和‘银行’中的相关表面片段？" :options="['向量维度一定太小', '静态查表只依赖 token ID，不依赖当前句子的语境与位置', '中文不能使用 embedding', '余弦相似度没有定义']" :answer="1" explanation="上下文编码器把 token、位置和周围证据共同映射为位置相关表示。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>同一条维修记录同时用于“下一 token 补全”和“故障类别判定”。分别画出每个输出位置允许读取的输入范围，并标出任何未来信息泄漏。</div>

## 1. 链式法则把联合概率变成局部预测

任意序列 $x_{1:T}$ 的联合概率都可精确写为

$$p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).$$

模型用参数 $\theta$ 近似每个条件分布，训练最小化 token 级负对数似然：

$$\mathcal L_{\mathrm{NLL}}=-\sum_{t=1}^{T}m_t\log p_\theta(x_t\mid x_{<t}),$$

其中 $m_t$ 是有效 token mask。padding、提示区、用户不希望监督的片段必须由 mask 明确定义；“batch 里有这个 ID”不等于“它应贡献 loss”。序列起止 token 也属于概率合同，不能在训练和评价时悄悄改变。

<LanguageModelFactorizationLab />

```python
import math

token_nll = [1.2, 0.4, 2.0, 0.8]
mask = [1, 1, 0, 1]  # 第三个位置是 padding，不计入评价
total = sum(loss * keep for loss, keep in zip(token_nll, mask))
count = sum(mask)
mean_nll = total / count
perplexity = math.exp(mean_nll)
assert math.isclose(mean_nll, 0.8)
assert math.isclose(perplexity, math.exp(0.8))
```

<DlQuiz kind="概率分解" question="自回归因子分解中的 p(x_t|x_<t) 是否要求模型只使用固定长度上下文？" :options="['要求恰好两个 token', '不要求；链式法则允许完整前缀，具体模型才可能截断上下文', '要求读到未来', '只适用于英文']" :answer="1" explanation="因子分解是概率恒等式；上下文窗口是模型和系统的近似。" />

## 2. 从 n-gram 到可学习上下文状态

传统 n-gram 直接计数最近 $n-1$ 个 token，稀疏组合导致大量未见上下文。Bengio 等人的神经概率语言模型先查表得到连续 embedding，再把固定窗口拼接送入前馈网络；相似词因共享参数获得统计强度。它仍受固定窗口和位置绑定限制。

RNN/LSTM 把任意长度前缀压进递归状态 $h_t=f(h_{t-1},e(x_t))$。这个状态大小固定，但“理论上看过”不等于“有效保留”：长距离梯度、状态容量和优化都会造成遗忘。Transformer 允许每层直接聚合多个先前位置，但计算、KV cache 和位置外推又形成新的上下文成本。本章只比较接口；第 19–20 章详讲预训练范式与现代架构。

<DlQuiz kind="上下文容量" question="把最大上下文从 2k 扩到 128k 后，哪项结论最可靠？" :options="['模型必然正确使用所有远距证据', '可见范围扩大，但有效利用仍需 needle、冲突证据和位置切片实验验证', '困惑度必然降到 1', 'KV cache 不再占内存']" :answer="1" explanation="窗口长度是可访问上限，不是信息利用能力证书。" />

## 3. Teacher forcing、生成与暴露偏差

训练时通常把真实前缀喂给模型，这叫 teacher forcing；同一序列的所有位置可以并行形成监督。推理时，模型读取自己刚生成的 token。一次早期错误会改变后续条件分布，这种训练—推理前缀差异常称 exposure bias。

它不是简单把 teacher forcing 关闭就能解决。自由运行训练会失去稳定的逐位置监督，还要处理离散采样。实践中应同时检查多步 rollout、错误恢复、长度、重复、终止行为和任务成功率；解码策略将在第 24 章展开。

```python
def training_pairs(tokens, bos="<bos>"):
    inputs = [bos] + tokens[:-1]
    targets = tokens
    return list(zip(inputs, targets))

pairs = training_pairs(["立即", "停机", "检查"])
assert pairs == [("<bos>", "立即"), ("立即", "停机"), ("停机", "检查")]
```

<DlQuiz kind="Teacher forcing" question="teacher forcing 训练和自回归推理的关键输入差异是什么？" :options="['训练前缀来自真实数据，推理前缀逐步包含模型自己的输出', '训练没有 target', '推理可以无条件读取未来答案', '二者的 tokenizer 必须不同']" :answer="0" explanation="前缀分布差异会让局部预测误差在长序列中累积。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>一个模型 teacher-forced perplexity 很低，但生成维修步骤时不断重复“检查”。设计不依赖单一自动指标的复现实验：至少记录哪四种序列级行为？</div>

## 4. 因果语言模型与双向掩码目标

因果语言模型（causal LM）在位置 $t$ 只能读取 $x_{\le t}$，适合直接生成。掩码语言模型（masked LM）随机遮住输入位置，用左右两侧上下文恢复它们；BERT 由此学习双向编码表示，适合分类、抽取和检索，但其输出不是按左到右归一化的序列生成概率。

两者的“mask”不要混淆：attention mask 控制哪些位置能互相读取，loss mask 控制哪些位置计入损失，MLM 的 `[MASK]` 则是输入替换策略。同一 batch 可能同时存在三类 mask。

```python
def causal_visibility(length):
    return [[int(source <= target) for source in range(length)]
            for target in range(length)]

visibility = causal_visibility(4)
assert visibility[0] == [1, 0, 0, 0]
assert visibility[3] == [1, 1, 1, 1]
assert sum(map(sum, visibility)) == 10
```

<DlQuiz kind="目标函数" question="为什么 BERT 式双向编码器不能直接作为无泄漏的左到右生成模型？" :options="['因为它没有 embedding', '其训练表示可以读取目标位置右侧信息，并未定义同样的因果生成分解', '因为它只能处理一个 token', '因为 softmax 只能用于视觉']" :answer="1" explanation="离线双向表示与在线因果生成有不同的信息合同。" />

## 5. ELMo：同一 token 的多层上下文表示

ELMo 用双向语言模型产生每个位置的多层状态，并为下游任务学习层混合：

$$\mathrm{ELMo}_t^{task}=\gamma^{task}\sum_{j=0}^{L}s_j^{task}h_{t,j},\qquad \sum_js_j=1.$$

较低层常保留形态和局部句法，较高层更偏任务语义；这是经验趋势，不是层编号的固定解释。ELMo 仍以字符/词级管线和双向 LSTM 为核心，而 BERT 把深层双向 Transformer 直接放入大规模 masked pretraining。两者共同完成关键转向：表示是 token-in-context，而不是词表行号的永久属性。

<ContextSenseLab />

<DlQuiz kind="上下文表示" question="ELMo 的任务特定层混合 s_j 主要允许下游任务做什么？" :options="['重写 tokenizer 的 Unicode 编码', '选择不同深度的上下文特征组合，而非只取固定最后一层', '保证消除所有偏置', '把双向模型变为因果生成器']" :answer="1" explanation="不同任务可能需要不同层次的局部、句法与语义信息。" />

## 6. 困惑度的可比边界

语料总有效 token 数为 $N$ 时，$\mathrm{PPL}=\exp(-\frac1N\sum_i\log p(x_i))$。它是平均 NLL 的单调变换，不是“模型有多少种困惑”。更低 PPL 只表示在同一概率单位、同一 tokenizer、同一数据切分、同一 BOS/EOS 与 mask 合同下分配了更高概率。

不同 tokenizer 会改变 $N$ 和每步事件空间，因此 token-level PPL 不能直接横比。可以补充 bits-per-byte/character，但仍要统一规范化与字节合同。领域部署还要报告长度、语言、设备类型、时间切片、OOD、置信校准和真实任务损失。

<DlQuiz kind="评价协议" question="两个模型分别用字符 tokenizer 和 50k 子词 tokenizer，哪种比较最可信？" :options="['直接比较 token-level perplexity', '统一原始文本和字节/字符概率单位，公开 tokenizer 与归一化合同，再辅以下游评价', '只比较词表大小', 'PPL 数字小的必然更安全']" :answer="1" explanation="token 事件单位不同会破坏直接 PPL 可比性。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>设计维修语料的时间外测试：如何避免同一设备模板跨集合泄漏？除 PPL 外，如何按罕见故障、长记录和新设备型号切片？</div>

## 7. 工程门禁：形状、数据与概率

训练 batch 通常至少包含 `input_ids[B,T]`、位置/attention mask 与 `labels[B,T]`。做下一个 token 预测时，logits 与 labels 必须对齐错开；padding label 常设为 ignore index。常见静默错误包括：未 shift、把 prompt 也计入 SFT loss、跨文档拼接而没有边界 token、验证集模板泄漏、在 padding 上平均，以及评测阶段仍启用随机 masking。

概率校准也不能由低 NLL 自动推出。对故障风险提示，应画可靠性曲线、检查温度缩放是否只在独立校准集拟合，并允许模型拒答。语言模型可能流畅地给错误建议；流畅度不是事实性或安全性。

## 8. 推荐论文阅读

<DlPaperCard title="A Neural Probabilistic Language Model" meta="Bengio et al. · 2003" priority="必读 · 神经语言模型起点" question="连续词表示如何缓解 n-gram 的维数灾难，固定窗口又留下什么限制？" reading="精读联合学习 embedding 与条件概率的目标、计算瓶颈和实验设置；把它与第 17 章静态词向量区分。" href="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf" />

<DlPaperCard title="Deep Contextualized Word Representations" meta="Peters et al. · 2018" priority="必读 · ELMo" question="双向语言模型的不同层为何要由下游任务学习加权，而不是永久选定最后一层？" reading="精读表示公式、biLM 目标与消融；记录静态 embedding 和 contextual representation 的输入输出接口差异。" href="https://arxiv.org/pdf/1802.05365" />

<DlPaperCard title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" meta="Devlin et al. · 2019" priority="必读 · 双向预训练桥梁" question="masked LM 与 next sentence prediction 怎样构造监督，预训练—微调接口为何有效？" reading="先读目标、输入表示和任务适配，再带着数据污染、mask mismatch 与生成限制审视结果。第 19 章会与 GPT、T5 对照。" href="https://arxiv.org/pdf/1810.04805" />

## 9. 练习与门禁

1. 对含 padding 的 batch 手算 masked NLL 与 PPL；2. 构造因果 attention mask 并验证无未来泄漏；3. 对齐 shifted logits/labels；4. 比较固定窗口和递归状态在远距依赖上的失效；5. 复现 ELMo 层混合的 softmax 权重；6. 用相同原始文本比较 token-PPL 与 bits-per-byte；7. 做模板去重后的时间外切分；8. 为补全系统增加重复、终止、拒答与风险切片。

## 本章小结

- 链式法则定义自回归概率，NLL 与 PPL 的统计单位由 tokenizer、边界和 mask 合同共同决定。
- 固定窗口、递归状态和注意力提供不同上下文接口；可见长上下文不等于有效利用。
- Teacher forcing 提供稳定并行监督，但生成时模型面对自己的前缀，必须做序列级评价。
- ELMo 与 BERT 把表示从静态词表向量推进到 token-in-context；双向编码与因果生成不能混用信息边界。
- 工业评价要检查数据泄漏、padding/shift、切片、校准和任务风险，而不只看平均 perplexity。

<DlLessonProgress lesson-id="18" />
