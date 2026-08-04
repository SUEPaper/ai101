---
course: data-science
title: 文本与正则表达式
description: 用字符串方法和正则表达式规范文本、提取结构化字段，并理解模式匹配的边界。
outline: [2, 3]
---

<DsLessonHero
  number="06"
  title="文本与正则表达式"
  eyebrow="把重复出现的文本结构变成可检验的模式"
  duration="建议 90–110 分钟"
  :outcomes="[
    '使用字符串方法完成大小写、空白和常见别名规范化',
    '解释字符类、量词、锚点和捕获组如何组成正则模式',
    '使用 Pandas str 方法提取字段，并判断何时不应使用 Regex'
  ]"
/>

## 本讲要解决的问题

站点编号本应写成 `N-01`，现实文件中却可能出现 `n-01`、`N 01`、`north-01`。文本清洗有两个目标：把同一对象的不同写法规范成一致形式，以及从半结构化文本中提取新字段。

```python
import pandas as pd

codes = pd.Series([" n-01 ", "N 03", "south-2", "W-12", None], name="raw_code")
normalized = codes.str.strip().str.upper().str.replace(" ", "-", regex=False)

print(normalized)
```

普通字符串方法适合确定的变换。只有当目标由“模式”定义时，才需要正则表达式。

## 1. 正则表达式是一种小型模式语言

以下模式匹配一个大写字母、一个连字符和两位数字：

```text
^[A-Z]-\d{2}$
```

<figure class="ds-figure">
  <div class="ds-anatomy">
    <span class="is-index">片段</span><span class="is-column">^[A-Z]</span><span class="is-column">-</span><span class="is-column">\d{2}$</span>
    <span class="is-index">含义</span><span>从开头匹配一个大写字母</span><span>字面连字符</span><span>两位数字后立即结束</span>
    <span class="is-index">示例</span><span>N</span><span>-</span><span>01</span>
  </div>
  <figcaption>锚点 `^` 和 `$` 约束整个字符串。如果去掉它们，模式可能只匹配一段局部文本。</figcaption>
</figure>

<RegexLab />

## 2. 四组最常用的语法

### 字符类

- `[ABC]`：A、B、C 中任意一个。
- `[A-Z]`：任意大写英文字母。
- `\d`：数字字符。
- `\s`：空白字符。
- `.`：除换行外的任意字符。

### 量词

- `?`：前一个元素出现 0 或 1 次。
- `*`：出现 0 次或多次。
- `+`：出现 1 次或多次。
- `{2}`：恰好 2 次。
- `{2,4}`：出现 2 到 4 次。

### 边界

- `^`：字符串开头。
- `$`：字符串结尾。
- `\b`：单词边界。

### 分组与选择

- `(N|S)`：匹配 N 或 S，并捕获结果。
- `(?:N|S)`：匹配但不捕获。
- `(?P<zone>[A-Z])`：命名捕获组。

```python
import re

pattern = re.compile(r"^(?P<zone>[A-Z])-(?P<number>\d{2})$")
match = pattern.fullmatch("N-03")

print(match.groupdict() if match else None)
```

## 3. Python 中优先使用 Raw String

正则和 Python 字符串都使用反斜杠。如果使用普通字符串，一个反斜杠可能需要重复转义。Raw String 在前面增加 `r`，让正则更接近屏幕上的写法：

```python
digits_pattern = r"\d+"
windows_path_pattern = r"^[A-Z]:\\"

print(re.findall(digits_pattern, "S12 has 8 bikes"))
print(bool(re.search(windows_path_pattern, r"C:\data\stations.csv")))
```

Raw String 不会改变正则语义，只减少 Python 字符串层面的转义负担。

## 4. 在 Pandas 中匹配、提取和替换

### 验证完整格式

```python
clean_codes = pd.Series(["N-01", "S-2", "N-03", "W-12", "N-104"])
valid = clean_codes.str.fullmatch(r"[A-Z]-\d{2}", na=False)

print(valid)
```

### 提取捕获组

```python
parts = clean_codes.str.extract(r"^(?P<zone>[A-Z])-(?P<number>\d+)$")
parts["number"] = pd.to_numeric(parts["number"], errors="coerce")

print(parts)
```

### 替换格式

```python
messy = pd.Series(["N 01", "S_02", "W-12"])
standard = messy.str.replace(r"[ _]", "-", regex=True)

print(standard)
```

<div class="ds-stop">
  <strong>停下来想一想</strong>
  `str.contains()` 问“任意位置是否出现”，`str.fullmatch()` 问“整个字符串是否符合”。做格式验证时通常需要后者。
</div>

## 5. 贪婪匹配与最小匹配

量词默认尽可能多地匹配：

```python
text = "站点[湖畔] 状态[低库存]"

print(re.findall(r"\[(.*)\]", text))
print(re.findall(r"\[(.*?)\]", text))
```

第一种 `.*` 会从第一个左括号吃到最后一个右括号；`.*?` 则使用最小匹配，分别得到两个括号内容。更稳健的写法通常是明确排除右括号：`\[([^]]+)\]`。

## 6. Regex 的边界

正则适合局部、规则相对稳定的文本结构。以下情况不应强行使用：

- 解析任意嵌套 JSON 或 HTML。
- 理解自然语言中的否定、上下文和指代。
- 规则不断变化且需要大量例外。
- 已经存在可靠的专用解析器。

<figure class="ds-figure">
  <div class="ds-flow">
    <div><b>NORMALIZE</b><span>统一大小写与空白</span><small>普通字符串方法</small></div>
    <div><b>MATCH</b><span>定位符合模式的文本</span><small>contains / fullmatch</small></div>
    <div><b>EXTRACT</b><span>把捕获组变成新列</span><small>str.extract</small></div>
    <div><b>VALIDATE</b><span>检查缺失和未匹配比例</span><small>不要静默丢失</small></div>
    <div><b>REVIEW</b><span>抽样查看原文和结果</span><small>规则不能替代语义</small></div>
  </div>
  <figcaption>Regex 管道必须保留原始文本，并统计未匹配案例。否则错误的模式会制造看似整齐的新字段。</figcaption>
</figure>

## 7. 知识检查

<QuizBlock
  question="`^` 和 `$` 同时出现时，主要作用是什么？"
  :options="['忽略大小写', '要求模式覆盖整个字符串的首尾', '匹配任意数字', '开启捕获组']"
  :answer="1"
  explanation="两个锚点把匹配限制在完整字符串，常用于格式验证。"
/>

<QuizBlock
  question="`\d+` 表示什么？"
  :options="['恰好一个数字', '一个或多个数字', '零个或多个字母', '字符串结尾']"
  :answer="1"
  explanation="\d 表示数字字符，+ 表示前一个元素出现一次或多次。"
/>

<QuizBlock
  question="为什么 Python 正则常写成 `r'\d+'`？"
  :options="['让匹配速度翻倍', '减少 Python 字符串层面对反斜杠的转义', '自动忽略空白', '把结果转成整数']"
  :answer="1"
  explanation="Raw String 让反斜杠更直接地传给正则引擎，避免两层转义混淆。"
/>

<QuizBlock
  question="从站点编码中分别取得区域字母和编号，应使用什么功能？"
  :options="['捕获组与 str.extract', 'str.len', 'sort_values', 'groupby.size']"
  :answer="0"
  explanation="捕获组定义需要保留的片段，str.extract 把每个捕获组变成结果列。"
/>

<QuizBlock
  question="以下哪个任务最不适合单独依赖 Regex？"
  :options="['验证站点编码格式', '提取日志中的日期', '解析任意嵌套 HTML 并理解语义', '替换多种分隔符']"
  :answer="2"
  explanation="嵌套结构和自然语言语义需要专用解析器或更强的模型，正则只适合局部稳定模式。"
/>

## 8. 代码练习：规范站点编码

把以下编码转换为标准形式 `区域字母-两位编号`。只接受 `north`、`south`、`east`、`west` 或对应首字母；无法识别的值保留为缺失，并输出未识别比例。

```python
raw = pd.Series(["north 1", "S-02", "east_7", "W 12", "unknown", None])
```

<details><summary>提示 1</summary>

先统一小写和分隔符，再用 Dictionary 把方向词转换为首字母。

</details>

<details><summary>提示 2</summary>

可以先提取方向与数字，再使用字符串格式化生成两位编号。

</details>

<details><summary>参考实现</summary>

```python
import pandas as pd

raw = pd.Series(["north 1", "S-02", "east_7", "W 12", "unknown", None])
prepared = raw.str.strip().str.lower().str.replace(r"[-_]", " ", regex=True)
parts = prepared.str.extract(r"^(north|south|east|west|n|s|e|w)\s+(\d+)$")

zone_map = {
    "north": "N", "n": "N",
    "south": "S", "s": "S",
    "east": "E", "e": "E",
    "west": "W", "w": "W",
}

zone = parts[0].map(zone_map)
number = pd.to_numeric(parts[1], errors="coerce")
standard = zone + "-" + number.map(lambda value: f"{value:02.0f}" if pd.notna(value) else pd.NA)

print(standard)
print("未识别比例：", standard.isna().mean())
```

</details>

## 9. 本讲小结

- 先用普通字符串方法规范确定的差异，再使用 Regex 处理模式。
- 字符类、量词、锚点和捕获组构成大多数实用模式。
- 格式验证优先使用完整匹配，字段提取使用捕获组。
- 保留原文、统计未匹配比例，并知道何时换用专用解析器。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 06：文本处理和正则表达式的知识边界。
- *Learning Data Science*，Ch.13：字符串、正则和文本特征主题。
- 本页站点编码案例、文字、图解、代码、问题和练习均为本课程原创。

<DsLessonProgress lesson-id="06" next-href="/courses/data-science/07-visualization-distributions" next-label="第 07 讲：看见分布" />
