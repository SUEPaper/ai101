---
course: deep-learning
edition: industrial
title: 文本数据、分词与子词建模
description: 从 UTF-8、Unicode 规范化和 grapheme 边界进入 word/character/byte tokenization、BPE、WordPiece、Unigram 与 SentencePiece 的可复现接口。
outline: [2, 3]
---

<DlLessonHero number="16" module="NLP 基础" title="文本数据、分词与子词建模" question="模型看到的不是“句子”，而是一串由编码、规范化、预分词、子词算法和特殊 token 共同决定的整数；怎样保证这条边界不漂移？" duration="建议 280–370 分钟" :outcomes="['区分 UTF-8 bytes、code points、grapheme clusters 与 model tokens', '审计 NFC/NFKC、大小写、空白与控制字符规范化', '手工执行 BPE 并比较 WordPiece、Unigram 与 byte fallback', '建立 tokenizer 训练、版本、offset 与特殊 token 的发布门禁']" />

机器学习模型不直接接收文本。真正输入是 `input_ids`、attention mask、position 与可能的 segment/type ids。Tokenizer 一旦变化，同一文档会变成不同训练样本，embedding 行、长度预算、截断位置和服务缓存全都失配。

## 贯穿案例：中英混合工业维修日志

语料包含中文、英文型号、全角字符、emoji、单位 `μm`、组合音标、不可见控制符和日志模板。目标是训练可用于后续 LLM 的 tokenizer。门禁不是“能 tokenize”，而是 round-trip、offset、特殊 token、长尾覆盖、领域术语和版本兼容全部明确。

<DlQuiz kind="数据边界" question="Python 的 len(text)、UTF-8 byte 数和用户看到的字符数是否总相同？" :options="['总相同', '不相同，emoji、组合音标和非 ASCII 会使多个层级长度不同', '只在英文中不同', '只在训练后不同']" :answer="1" explanation="字符串实现、Unicode code point、grapheme cluster 和编码 byte 是不同边界。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>“👩🏽‍💻”由哪些 code point 组成？若按 code point 截断到中间，显示、offset 和标签会怎样损坏？</div>

## 1. 从 bytes 到用户感知字符

UTF-8 用 1–4 bytes 编码 Unicode code point；组合字符与 ZWJ 序列又让多个 code point 组成一个 grapheme cluster。JavaScript `length` 计 UTF-16 code units，Python `len` 通常计 code points，都不等于 grapheme 数。

<TextBoundaryLab />

```python
import unicodedata

raw = "ＡI Cafe\u0301 👩🏽‍💻"
nfc = unicodedata.normalize("NFC", raw)
nfkc = unicodedata.normalize("NFKC", raw)
assert "é" in nfc
assert nfkc.startswith("AI")
assert len(raw.encode("utf-8")) > len(raw)
```

NFC 合并规范等价序列；NFKC 还折叠全角、兼容字形等。检索可能受益于 NFKC，但型号中的全角/半角或数学符号可能有业务含义。选择必须通过下游反例，而非统一“清洗”。

<DlQuiz kind="规范化" question="NFKC 相比 NFC 的额外风险是什么？" :options="['无法处理 Unicode', '会折叠兼容字符，可能删除业务有意义的形式差异', '必然增加 byte 数', '只适用于图片']" :answer="1" explanation="兼容等价不等于领域语义等价。" />

## 2. 规范化是一份不可逆合同

应显式登记：解码错误策略；Unicode normalization；大小写；空白折叠；换行；控制字符；数字/单位；HTML；语言标记；隐私替换；文档边界。清洗顺序也重要，例如先 offset 标注再 NFKC 会让字符位置漂移。

```python
def normalize_log(text):
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(text.split())
    return text

assert normalize_log("温度\r\n  80°C") == "温度 80°C"
```

<DlQuiz kind="Offset 合同" question="做 NER span 标注时，最危险的流程是什么？" :options="['先冻结规范化与 tokenizer 再标注', '在原文标注 offset，训练时再做会改变长度的规范化却不重映射', '保存原文 hash', '同时保存 char/token offset']" :answer="1" explanation="文本改变后原 offset 不再指向同一 span。" />

<div class="dl-stop"><strong>停下来推一推 2</strong>日志中多个空格表示表格列。若统一折叠空白，会破坏什么结构？提出保留布局与去噪的双版本策略。</div>

## 3. 四种基本粒度

word tokenizer 序列短但开放词表困难，中文/黏着语语言边界复杂；character 覆盖广但序列长；byte 只有 256 基础单位、无 OOV，却把多字节文字拉长；subword 在词表与长度之间折中。

| 粒度 | 覆盖 | 长度 | 主要风险 |
|---|---|---|---|
| word | 依赖词表 | 短 | OOV、边界语言依赖 |
| character | Unicode 范围 | 中长 | grapheme 被拆、词义组合负担 |
| byte | 完全 | 最长 | 非拉丁文字成本、可读性差 |
| subword | 词表 + fallback | 中 | 语料偏置、边界不直观 |

## 4. BPE：贪心合并频繁相邻对

BPE 从字符/byte 词表开始，反复统计相邻 pair，选择最高频 pair 合并并加入词表。训练结果依赖预分词、空白标记、tie-break、语料顺序和 merge 数。

<BpeMergeLab />

```python
from collections import Counter

corpus = [list("low"), list("lower"), list("newest")]
pairs = Counter()
for word in corpus:
    pairs.update(zip(word, word[1:]))
best = max(pairs, key=lambda pair: (pairs[pair], pair))
assert pairs[best] >= 1

def merge(word, pair):
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and tuple(word[i:i+2]) == pair:
            out.append("".join(pair)); i += 2
        else:
            out.append(word[i]); i += 1
    return out

assert len(merge(list("lower"), ("e", "r"))) == 4
```

<DlQuiz kind="BPE" question="增加 BPE merge 数通常怎样改变词表与序列？" :options="['词表变小、序列变长', '词表变大、常见片段序列变短', '都不变', '只改变 Unicode 编码']" :answer="1" explanation="合并把常见 pair 变成新 token，以更大词表换较短序列。" />

## 5. WordPiece、Unigram 与 SentencePiece

WordPiece 常选择最大化似然收益的合并并用 longest-match-first 编码；Unigram 从大候选词表出发，按概率模型逐步裁剪，编码可用 Viterbi 或采样；SentencePiece 是直接在原始 Unicode 文本上训练/编码的实现框架，可承载 BPE 或 Unigram，并把空白作为显式符号。

算法名不够。必须保存模型文件、normalizer、pre-tokenizer、decoder、added tokens、特殊 token ID 和训练库版本。相同 `.vocab` 不保证相同 offset/decoder 行为。

<DlQuiz kind="实现边界" question="SentencePiece 与 BPE 是同义词吗？" :options="['是', '不是，SentencePiece 是可直接处理原始文本的 tokenizer 框架，可使用 BPE 或 Unigram 模型', 'SentencePiece 只处理句号', 'BPE 只处理图像']" :answer="1" explanation="实现框架与子词学习算法要分开描述。" />

## 6. 特殊 token、padding 与截断

`BOS/EOS/PAD/UNK/MASK` 的字符串和 ID 是模型 ABI。decoder-only 常左 padding 便于批量最后位置生成，但训练可右 padding；position 与 attention mask 必须匹配。截断应保留文档边界、标签 span 和任务所需尾部，不能默默取前 N 个 token。

<div class="dl-stop"><strong>停下来推一推 3</strong>维修结论总在日志末尾。默认右截断会造成什么标签捷径或信息丢失？设计 head+tail、分块和层级汇总对照。</div>

## 7. 推荐论文阅读

<DlPaperCard title="Neural Machine Translation of Rare Words with Subword Units" meta="Sennrich, Haddow & Birch · 2015" priority="必读 · BPE" question="BPE 如何从数据学习可组合子词并缓解稀有词，训练与应用阶段各做什么？" reading="手工重放算法 1，再检查词边界与翻译结果中的复合词。" href="https://arxiv.org/pdf/1508.07909" />

<DlPaperCard title="SentencePiece: A simple and language independent subword tokenizer and detokenizer" meta="Kudo & Richardson · 2018" priority="必读 · 工程接口" question="直接处理 raw sentence 与显式空白如何减少语言依赖预分词？" reading="读 normalizer、lossless tokenization、BPE/Unigram 支持与 subword regularization。" href="https://arxiv.org/pdf/1808.06226" />

<DlPaperCard title="ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models" meta="Xue et al. · 2021" priority="选读 · Byte 对照" question="byte 级模型用更长序列换来了哪些噪声鲁棒性与跨语言覆盖？" reading="重点比较序列长度、模型分配和下游噪声任务。" href="https://arxiv.org/pdf/2105.13626" />

## 8. 练习与门禁

1. 建立 100 个 Unicode/emoji/控制字符 round-trip 测试；2. 手工训练 10 次 BPE merge；3. 比较 BPE/Unigram/byte 的长度分位；4. 对领域术语做完整率切片；5. 验证 encode/decode、offset 与特殊 ID；6. tokenizer 文件 hash 不匹配时拒绝加载模型。

## 本章小结

- bytes、code points、graphemes 与 tokens 是不同边界；规范化可能不可逆。
- 子词算法在词表、序列长度和开放词覆盖间交换，BPE token 不等于语言学词素。
- tokenizer 是模型 ABI，训练、索引、微调与服务必须共享完整版本和特殊 token 合同。

<DlLessonProgress lesson-id="16" next-href="/courses/deep-learning/17-word-embeddings" next-label="下一章：词向量" />
