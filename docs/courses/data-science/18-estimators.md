---
course: data-science
title: 估计量、偏差与方差
description: 区分参数、估计量与估计值，用平行重复抽样理解偏差、方差、噪声和模型复杂度。
outline: [2, 3]
---

<DsLessonHero
  number="18"
  title="估计量、偏差与方差"
  eyebrow="一份数据只能给一个答案，估计量的品质要在许多可能样本中判断"
  duration="建议 110–130 分钟"
  :outcomes="[
    '区分数据生成过程、参数、估计量和估计值',
    '通过重复模拟估计偏差、方差和 MSE',
    '解释模型复杂度如何制造偏差与方差的折中'
  ]"
/>

## 本讲要解决的问题

能源站抽取 30 个家庭，得到平均节电量 2.4 kWh。换一批家庭，答案可能变成 1.9 或 2.8。我们应该怎样评价“样本均值”这条计算规则，而不是只评价眼前的 2.4？

## 1. 四个对象构成推断链

<div class="ds-flow">
  <div><b>DGP</b><span>数据生成过程</span><small>总体、抽样与测量机制</small></div>
  <div><b>PARAMETER</b><span>固定但未知的量</span><small>总体均值 μ</small></div>
  <div><b>ESTIMATOR</b><span>样本到答案的规则</span><small>样本均值 X̄</small></div>
  <div><b>ESTIMATE</b><span>本次算出的数字</span><small>例如 2.4 kWh</small></div>
</div>

参数在给定总体中固定；抽样前估计量是随机变量；观察样本后得到一个确定的估计值。

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
population = rng.gamma(shape=2.2, scale=1.1, size=100_000)
parameter = population.mean()
sample = rng.choice(population, size=30, replace=False)
estimate = sample.mean()

print("总体参数：", round(parameter, 3))
print("本次估计：", round(estimate, 3))
```

## 2. 偏差是估计量长期中心与参数的距离

估计量偏差定义为 `Bias(T) = E[T] - θ`。无偏表示重复抽样中心对准参数，不表示每一次都准确。

```python
def sampling_distribution(population, sample_size, estimator, repeats=4000, seed=7):
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sample = rng.choice(population, size=sample_size, replace=True)
        estimates.append(estimator(sample))
    return np.asarray(estimates)

mean_estimates = sampling_distribution(population, 30, np.mean)
median_estimates = sampling_distribution(population, 30, np.median)

print("均值估计量偏差：", mean_estimates.mean() - parameter)
print("中位数估计总体均值的偏差：", median_estimates.mean() - parameter)
```

中位数可能很好地估计总体中位数，但若目标参数是总体均值，它就是不同任务下的估计量。讨论偏差必须先说清楚目标参数。

<QuizBlock question="无偏估计量是否保证本次估计等于参数？" :options="['保证', '不保证；无偏描述重复抽样的平均', '只要 n>30 就保证', '只对中位数保证']" :answer="1" explanation="无偏是抽样分布中心的性质，单次样本仍有随机误差。" />

## 3. 方差描述估计量在平行样本间的波动

两个估计量可能都无偏，但一个更稳定。方差越小，平行重复抽样得到的数字越集中。

```python
for sample_size in [10, 30, 100, 300]:
    estimates = sampling_distribution(population, sample_size, np.mean, repeats=2000)
    print(sample_size, "bias=", round(estimates.mean() - parameter, 4), "variance=", round(estimates.var(), 4))
```

扩大 iid 样本主要降低方差，不会修复估计规则或采样机制带来的系统偏差。

## 4. MSE 同时惩罚偏差与方差

对参数估计，`MSE(T) = E[(T-θ)²] = Bias(T)² + Var(T)`。

```python
def estimator_report(estimates, parameter):
    bias = estimates.mean() - parameter
    variance = estimates.var(ddof=0)
    mse = np.mean((estimates - parameter) ** 2)
    return {"bias": bias, "variance": variance, "mse": mse, "check": bias ** 2 + variance}

print(estimator_report(mean_estimates, parameter))
```

一个略有偏但方差显著更小的估计量，可能拥有更低 MSE。正则化正是主动引入一点偏差，换取更低方差的典型例子。

## 5. 预测误差还包含不可约噪声

监督学习中，训练数据和未来结果都可能随机变化。常见的概念分解是：

`期望测试误差 = Bias² + Variance + Noise`

- Bias²：模型平均预测与真实规律的系统距离。
- Variance：换一份训练集后，模型预测变化多少。
- Noise：即使知道真实规律，也无法从特征预测的个体随机性。

<BiasVarianceLab />

复杂度低时，不同训练集得到的模型相似，但都漏掉真实结构：高偏差、低方差。复杂度高时，平均形状可能更灵活，但模型开始追随各训练集的噪声：低偏差、高方差。

## 6. 用平行宇宙模拟模型方差

真实世界通常只有一份训练数据；模拟让我们看到“如果重新生成数据并训练，会发生什么”。

```python
def fit_polynomial(sample_x, sample_y, degree, query=0.4):
    design = np.column_stack([sample_x ** power for power in range(degree + 1)])
    beta, *_ = np.linalg.lstsq(design, sample_y, rcond=None)
    query_row = np.array([query ** power for power in range(degree + 1)])
    return query_row @ beta

def true_function(x):
    return 2 + 1.5 * x - 0.8 * x ** 2

rng = np.random.default_rng(2026)
predictions = {degree: [] for degree in [1, 2, 8]}
for _ in range(300):
    sample_x = rng.uniform(-1, 1, size=25)
    sample_y = true_function(sample_x) + rng.normal(0, 0.35, size=25)
    for degree in predictions:
        predictions[degree].append(fit_polynomial(sample_x, sample_y, degree))

truth = true_function(0.4)
for degree, values in predictions.items():
    values = np.asarray(values)
    print(degree, "bias=", round(values.mean() - truth, 3), "variance=", round(values.var(), 3))
```

## 7. 训练误差不能估计模型方差

高复杂度模型可能在每个平行训练集上都得到极低训练误差，同时对同一个新输入给出非常不同的预测。模型选择必须看留出数据或交叉验证。

<div class="ds-warning"><strong>偏差—方差不是二选一口号</strong>数据量、噪声、特征、正则化、优化和评价点都会改变分解。实际工作中通常无法精确计算两项，只能用学习曲线、交叉验证和稳定性实验寻找证据。</div>

## 8. 知识检查

<QuizBlock question="估计值与估计量有什么区别？" :options="['没有区别', '估计量是规则，估计值是规则在本次样本上的结果', '估计值一定随机', '估计量一定是均值']" :answer="1" explanation="抽样前规则的输出不确定；观察数据后得到具体数值。" />

<QuizBlock question="扩大相同机制的 iid 样本最直接降低什么？" :options="['系统偏差', '估计量方差', '参数本身', '标签定义错误']" :answer="1" explanation="样本量增加使随机波动缩小，但不改变固定方向的机制偏差。" />

<QuizBlock question="一个有偏估计量能否比无偏估计量拥有更低 MSE？" :options="['能，如果方差降低得足够多', '不能', '只对类别变量能', 'MSE 与偏差无关']" :answer="0" explanation="MSE=Bias²+Variance，较小方差可以抵消新增的偏差平方。" />

<QuizBlock question="不可约噪声指什么？" :options="['写错代码', '给定可用特征后仍无法预测的结果随机性', '模型参数太少', '验证集太大']" :answer="1" explanation="即使知道条件期望，单个结果仍可能围绕它随机变化。" />

## 9. 代码练习：比较两个方差估计量

正态总体方差为 9。对样本量 `n=8` 重复抽样，比较 `np.var(sample, ddof=0)` 与 `ddof=1` 的偏差、方差和 MSE。

<details><summary>提示</summary>

把两个估计量分别存入数组，再对真实参数 9 调用 `estimator_report`。`ddof=1` 是常见的无偏样本方差。

</details>

<details><summary>参考实现</summary>

```python
rng = np.random.default_rng(88)
biased_variances = []
unbiased_variances = []

for _ in range(10_000):
    sample = rng.normal(loc=4, scale=3, size=8)
    biased_variances.append(np.var(sample, ddof=0))
    unbiased_variances.append(np.var(sample, ddof=1))

biased_report = estimator_report(np.asarray(biased_variances), 9)
unbiased_report = estimator_report(np.asarray(unbiased_variances), 9)
print(pd.DataFrame([biased_report, unbiased_report], index=["ddof=0", "ddof=1"]))
```

</details>

## 10. 本讲小结

- 参数是数据生成过程中的固定目标；估计量是从样本到答案的规则。
- 偏差描述长期中心偏离参数，方差描述平行样本间波动。
- 估计量 MSE 等于偏差平方加方差；预测误差还包含不可约噪声。
- 模型复杂度通常降低偏差、增加方差，验证帮助寻找折中区域。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 18：数据生成过程、估计量、偏差、方差和模型复杂度的知识边界。
- *Learning Data Science*，Ch.16–17：泛化、估计量和推断主题。
- 本页能源与节电案例、数据、文字、图解、代码、问题和实验均为本课程原创。

<DsLessonProgress lesson-id="18" next-label="第 19 讲：Bootstrap 与参数推断" next-href="/courses/data-science/19-bootstrap" />
