---
course: data-science
title: 开始前的工具与数学桥
description: 用一个预备单元补齐 Jupyter、NumPy、Pandas 和本课程所需的数学直觉。
outline: [2, 3]
---

<DsLessonHero
  number="P0"
  title="开始前的工具与数学桥"
  eyebrow="先建立计算的坐标系，再进入数据世界"
  duration="建议 90–120 分钟"
  :outcomes="[
    '解释 Notebook 为什么可能出现乱序结果',
    '判断数组与表格的 shape，并区分行、列和轴',
    '用均值、方差、点积和梯度描述后续课程中的基本计算'
  ]"
/>

## 本讲要解决的问题

数据科学代码的难点通常不在语法，而在于你是否清楚“数据现在是什么形状、每个值代表什么、下一步运算沿哪个方向发生”。本单元只补齐这些会反复出现的底层概念。

### 准备课程环境

建议在项目根目录创建独立环境，避免与其他 Python 项目互相影响：

```bash
python3 -m venv .venv-ds
source .venv-ds/bin/activate
python -m pip install -r requirements-ds.txt
```

Windows PowerShell 激活命令为 `.venv-ds\Scripts\Activate.ps1`。安装后执行：

```python
import numpy as np
import pandas as pd

print(np.__version__)
print(pd.__version__)
```

版本选择与浏览器交互边界见[数据科学课程 Python 环境决策](/decisions/data-science-python-environment)。

### 先做诊断

<QuizBlock
  question="一个 Notebook 单元格显示 In [8]，它上面的单元格显示 In [3]。最准确的解释是什么？"
  :options="['Notebook 会自动从下往上运行', '两个单元格最近一次执行的顺序不同', '数字代表代码行数', '上方单元格运行失败了']"
  :answer="1"
  explanation="方括号中的数字是内核执行计数，不是页面顺序。Notebook 允许任意顺序执行，因此屏幕上的代码顺序不一定等于内核中的状态顺序。"
/>

<QuizBlock
  question="形状为 (120, 4) 的二维数组通常表示什么？"
  :options="['120 个特征、4 条记录', '120 条记录、4 个特征', '数组只有 124 个值', '数组必定包含 120 个 DataFrame']"
  :answer="1"
  explanation="在本课程的数据表约定中，第一个维度通常对应行或观测，第二个维度对应列或特征。"
/>

如果两题都能解释理由，可以快速阅读本页；如果只是猜对，建议亲手运行全部代码。

## 1. Notebook 是带状态的实验记录本

普通 Python 脚本通常从第一行运行到最后一行。Notebook 则维护一个持续存在的内核：变量一旦创建，会一直留在内存中，直到被覆盖、删除或重启内核。

<figure class="ds-figure">
  <div class="ds-pipeline">
    <span>Cell A：x = 10</span><i>→</i><span>内核记住 x</span><i>→</i><span>Cell B：x + 5</span><i>→</i><span>输出 15</span>
  </div>
  <figcaption>页面记录的是单元格位置，内核记录的是执行历史。遇到无法解释的结果时，重启内核并从上到下运行，是最可靠的排查起点。</figcaption>
</figure>

下面的代码可以直接放进一个空 Notebook：

```python
temperature = 26
rides = 128

print(f"气温 {temperature}°C，骑行量 {rides} 次")
```

输出：

```text
气温 26°C，骑行量 128 次
```

<div class="ds-warning">
  <strong>常见错误：代码看起来没问题，但变量值已经变了</strong>
  先检查执行编号，再选择“Restart Kernel and Run All”。不要通过反复运行不同单元格把状态碰巧调整正确。
</div>

## 2. 从 List 到 NumPy 数组

Python List 可以混合保存多种对象，NumPy 数组则更强调统一的数据类型和向量化计算。

```python
import numpy as np

rides = np.array([128, 94, 173, 81])

print(rides.shape)
print(rides.mean())
print(rides >= 100)
```

输出：

```text
(4,)
119.0
[ True False  True False]
```

最后一行不是一个答案，而是一组和原数组一一对应的判断。第 03 讲会用这组布尔值筛选表格行。

<ArrayShapeLab />

### Axis 是“压缩哪个方向”

对于二维数组，`axis=0` 表示沿着行向下汇总，最后为每一列留下一个结果；`axis=1` 表示横向汇总每一行。

```python
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
])

print(matrix.sum(axis=0))
print(matrix.sum(axis=1))
```

输出：

```text
[50 70 90]
[ 60 150]
```

<div class="ds-stop">
  <strong>停下来想一想</strong>
  不背“0 是列、1 是行”。问自己：这个轴被压缩以后，剩下哪个方向？上例中 `axis=0` 把两行压成一行，所以每列留下一个和。
</div>

## 3. 从数组到 DataFrame

数组主要依靠位置表达含义，DataFrame 则为行和列增加标签。标签让代码更接近问题语言。

```python
import pandas as pd

stations = pd.DataFrame({
    "station": ["湖畔", "剧场", "大学城"],
    "rides": [128, 94, 173],
    "is_weekend": [False, False, True],
})

print(stations.shape)
print(stations["rides"].mean())
```

这里的 `station`、`rides` 和 `is_weekend` 不是装饰，而是数据契约：它们说明每个值应该如何解释。

<figure class="ds-figure">
  <div class="ds-anatomy">
    <span class="is-index">index</span><span class="is-column">station</span><span class="is-column">rides</span><span class="is-column">is_weekend</span>
    <span class="is-index">0</span><span>湖畔</span><span>128</span><span>False</span>
    <span class="is-index">1</span><span>剧场</span><span>94</span><span>False</span>
    <span class="is-index">2</span><span>大学城</span><span>173</span><span>True</span>
  </div>
  <figcaption>DataFrame 同时拥有位置和标签。第 02 讲会专门处理两者的差异。</figcaption>
</figure>

## 4. 后续课程真正需要的数学

### 均值与方差：中心和波动

均值回答“典型水平大约在哪里”，方差回答“数据围绕均值散得有多开”。方差先计算每个值与均值的距离，再平方并取平均。

```python
rides = np.array([80, 100, 120])
center = rides.mean()
variance = ((rides - center) ** 2).mean()

print(center)
print(variance)
```

### 点积：把特征按权重合成一个数

一个线性模型可以写成“每个特征乘以一个权重，再相加”。点积只是这句话的紧凑写法。

```python
features = np.array([1.0, 26.0, 0.0])
weights = np.array([12.0, 4.5, -18.0])

prediction = features @ weights
print(prediction)
```

### 梯度：当前位置最陡的上升方向

如果目标是最小化损失，就沿梯度的反方向移动：

```python
theta = 8.0
learning_rate = 0.1

for _ in range(5):
    gradient = 2 * (theta - 3)
    theta = theta - learning_rate * gradient

print(round(theta, 3))
```

这里的损失函数在 `theta = 3` 时最小。暂时不需要推导，只要看懂“计算坡度 → 反向走一步 → 重复”即可。

## 5. 知识检查

<QuizBlock
  question="`rides = np.array([80, 100, 120])` 的 `shape` 是什么？"
  :options="['(3,)', '(1, 3)', '(3, 1)', '(80, 100, 120)']"
  :answer="0"
  explanation="这是只有一个轴的一维数组，轴的长度为 3，所以 shape 是 (3,)。"
/>

<QuizBlock
  question="二维数组按 `axis=0` 求和后，什么被压缩了？"
  :options="['列方向', '行方向', '所有维度', '数据类型']"
  :answer="1"
  explanation="axis=0 对应第一个维度，也就是行维度。行被压缩后，每列留下一个汇总结果。"
/>

<QuizBlock
  question="为什么 DataFrame 的列名很重要？"
  :options="['让文件体积更小', '让值获得业务含义并支持按标签操作', '保证不会有缺失值', '自动完成统计推断']"
  :answer="1"
  explanation="列名把位置上的数值连接到业务语义，也是筛选、聚合和建模时引用变量的主要方式。"
/>

## 6. 代码练习

创建一个包含 `day`、`temperature`、`rides` 三列的 DataFrame，完成以下任务：

1. 打印表格的 `shape`。
2. 计算平均骑行量。
3. 新增 `rides_per_degree = rides / temperature`。
4. 找出 `rides` 高于平均值的行。

<details><summary>提示 1：从数据结构开始</summary>

先用 Dictionary 构造 DataFrame，每个键是一列，每个值是长度相同的 List。

</details>

<details><summary>提示 2：筛选条件</summary>

`df["rides"] > df["rides"].mean()` 会得到一组布尔值，可以把它放进 `df[...]`。

</details>

<details><summary>参考实现</summary>

```python
import pandas as pd

daily = pd.DataFrame({
    "day": ["周一", "周二", "周三", "周四"],
    "temperature": [24, 27, 21, 25],
    "rides": [118, 142, 96, 131],
})

print(daily.shape)
print(daily["rides"].mean())

daily["rides_per_degree"] = daily["rides"] / daily["temperature"]
above_average = daily[daily["rides"] > daily["rides"].mean()]

print(above_average)
```

</details>

## 7. 本讲小结

- Notebook 的内核状态取决于执行历史。
- `shape` 描述数据结构，是很多计算能否成立的前提。
- NumPy 用数组表达批量计算，Pandas 用标签为数据增加语义。
- 均值、方差、点积和梯度会在后续课程中反复出现。

## 参考资料

- NumPy 官方文档：Array fundamentals。
- Pandas 官方文档：DataFrame basics。
- 本页案例、文字、图解、代码和练习均为本课程原创。

<DsLessonProgress lesson-id="P0" next-href="/courses/data-science/01-lifecycle" next-label="第 01 讲：数据科学生命周期" />
