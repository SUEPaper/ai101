---
course: data-science
title: sklearn 与梯度下降
description: 建立可复现的 sklearn 训练工作流，并从导数、梯度和学习率理解迭代优化。
outline: [2, 3]
---

<DsLessonHero
  number="13"
  title="sklearn 与梯度下降"
  eyebrow="训练模型，就是让参数沿着损失下降的方向反复更新"
  duration="建议 120–140 分钟"
  :outcomes="[
    '使用 fit、predict、score 描述 sklearn 估计器工作流',
    '根据导数判断一维参数的下降方向',
    '解释学习率、收敛、凸性和特征尺度的作用'
  ]"
/>

## 本讲要解决的问题

OLS 有直接解，但许多模型没有方便的闭式公式；数据很大时，直接解也可能代价过高。我们需要一种通用办法：从一个参数起点出发，用局部坡度一步步靠近更低损失。

## 1. sklearn 把训练过程封装成统一接口

典型估计器遵循三个动作：`fit` 学参数，`predict` 生成预测，评价函数比较预测与真实值。

```python
# tutorial-validation: syntax-only
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
rmse = mean_squared_error(y_valid, prediction) ** 0.5
```

<figure class="ds-figure">
  <div class="ds-flow">
    <div><b>SPLIT</b><span>先划分数据</span><small>测试集保持封存</small></div>
    <div><b>FIT PREP</b><span>只在训练集学预处理</span><small>均值、类别、缺失规则</small></div>
    <div><b>FIT MODEL</b><span>用训练集优化参数</span><small>model.fit</small></div>
    <div><b>VALIDATE</b><span>在验证集做选择</span><small>特征、超参数、阈值</small></div>
    <div><b>TEST</b><span>最后只评估一次</span><small>估计泛化表现</small></div>
  </div>
  <figcaption>接口统一不等于流程自动正确。划分时机和预处理边界仍由使用者负责。</figcaption>
</figure>

如果把全体数据标准化后再划分，验证集的信息已经参与训练。可靠做法是把预处理和模型放进同一 Pipeline。

```python
# tutorial-validation: syntax-only
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

model = make_pipeline(
    StandardScaler(),
    SGDRegressor(loss="squared_error", learning_rate="constant", eta0=0.01, random_state=42),
)
model.fit(X_train, y_train)
```

## 2. 一维梯度下降从坡度开始

考虑损失 `L(θ) = (θ - 2)² + 0.5`。导数是 `2(θ - 2)`：

- 导数为正，向右会升高，应向左走。
- 导数为负，向右会降低，应向右走。
- 导数为 0，处在水平位置；对于这条凸曲线，它就是全局最低点。

```python
import numpy as np

def loss(theta):
    return (theta - 2) ** 2 + 0.5

def gradient(theta):
    return 2 * (theta - 2)

theta = -3.5
learning_rate = 0.18
for step in range(8):
    theta = theta - learning_rate * gradient(theta)
    print(step + 1, round(theta, 4), round(loss(theta), 4))
```

更新规则中的负号不是背诵结果：梯度指向局部上升最快方向，所以减去它就是局部下降方向。

<GradientDescentLab />

<QuizBlock question="当前位置的导数为 -6。若学习率为正，梯度下降下一步向哪里移动？" :options="['参数增大', '参数减小', '保持不变', '无法判断']" :answer="0" explanation="θ_new = θ - η(-6)，因此参数增大。" />

## 3. 学习率决定步长，不决定目标位置

学习率太小会收敛很慢；太大可能跨过谷底、振荡甚至发散。固定学习率不是唯一选择，实际优化器常采用衰减学习率或自适应步长。

```python
def run_descent(start, rate, steps=12):
    theta = float(start)
    history = []
    for _ in range(steps):
        history.append((theta, loss(theta)))
        theta -= rate * gradient(theta)
    return np.array(history)

for rate in [0.05, 0.4, 1.05]:
    history = run_descent(-3.5, rate)
    print(rate, "final loss:", round(history[-1, 1], 3))
```

“损失下降了”也不是充分停止条件。可以同时检查迭代次数、参数变化、梯度范数和验证集表现。

## 4. 多参数时，导数升级为梯度

若 `β` 有多个分量，梯度收集损失对每个分量的偏导。线性回归 MSE 的梯度可以写成：

`gradient = -(2/n) Xᵀ(y - Xβ)`

```python
import numpy as np

X_demo = np.array([
    [1.0, -1.0],
    [1.0,  0.0],
    [1.0,  1.0],
    [1.0,  2.0],
])
y_demo = np.array([1.0, 2.1, 3.0, 4.2])

beta = np.zeros(X_demo.shape[1])
rate = 0.08
for _ in range(200):
    residual = y_demo - X_demo @ beta
    grad = -(2 / len(y_demo)) * X_demo.T @ residual
    beta -= rate * grad

print(beta.round(4))
print(np.linalg.lstsq(X_demo, y_demo, rcond=None)[0].round(4))
```

梯度下降得到的结果接近直接 OLS 解，是对更新公式的有力检查。

## 5. 凸性说明“局部向下”能否找到全局最低点

凸函数像一个没有隐藏山谷的碗。在线性回归的 MSE 中，任何局部最低点也是全局最低点；若设计矩阵满秩，最低点唯一。

神经网络等非凸模型的地形更复杂，可能有鞍点、平坦区和许多等价低谷。梯度下降仍然有用，但“导数接近 0”不再自动意味着找到唯一最好解。

<div class="ds-stop"><strong>停下来想一想</strong>梯度只描述当前位置附近的坡度。学习率决定我们是否仍留在这个局部近似可信的范围内。</div>

## 6. 特征尺度会改变优化地形

如果温度约为 30，而建筑面积约为 30,000，对应梯度尺度可能相差很大。损失等高线会变成狭长椭圆，固定步长容易左右震荡。

```python
features = np.array([
    [22, 18000], [25, 24000], [28, 31000], [31, 40000]
], dtype=float)

scaled = (features - features.mean(axis=0)) / features.std(axis=0, ddof=0)
print("原始标准差：", features.std(axis=0, ddof=0))
print("标准化后标准差：", scaled.std(axis=0, ddof=0))
```

标准化通常让优化更容易，也让下一讲的正则化更加公平地对待不同单位的列。

## 7. 知识检查

<QuizBlock question="fit 与 predict 的主要区别是什么？" :options="['fit 学习参数，predict 使用已学参数生成输出', '两者完全相同', 'predict 会重训模型', 'fit 只处理测试集']" :answer="0" explanation="fit 改变估计器的已学习状态；predict 应使用既有状态。" />

<QuizBlock question="为什么预处理参数只能从训练集估计？" :options="['让代码更短', '避免验证或测试信息泄漏进训练过程', '保证特征全为整数', '消除随机性']" :answer="1" explanation="全数据均值、类别表或填充值都会携带留出集信息。" />

<QuizBlock question="学习率过大时常见现象是什么？" :options="['稳定但很慢', '跨过最低点并振荡或发散', '梯度自动为 0', '测试集变大']" :answer="1" explanation="步长过大使局部下降方向无法保证实际更新后仍下降。" />

<QuizBlock question="凸损失的重要性质是什么？" :options="['每个点梯度都为 0', '局部最低点也是全局最低点', '不需要数据', '只能有一个参数']" :answer="1" explanation="凸性排除了更低的隐藏局部谷底。" />

## 8. 代码练习：带停止条件的梯度下降

实现 `fit_gd(X, y, rate, max_steps, tolerance)`。每次计算 MSE 和梯度；当梯度范数小于 `tolerance` 时提前停止。返回系数和损失历史。

<details><summary>提示</summary>

梯度是 `-(2/n) * X.T @ (y-X@beta)`；使用 `np.linalg.norm(grad)`。初始系数可设为全 0。

</details>

<details><summary>参考实现</summary>

```python
def fit_gd(X, y, rate=0.08, max_steps=5000, tolerance=1e-8):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    history = []
    for _ in range(max_steps):
        residual = y - X @ beta
        history.append(np.mean(residual ** 2))
        grad = -(2 / len(y)) * X.T @ residual
        if np.linalg.norm(grad) < tolerance:
            break
        beta -= rate * grad
    return beta, np.asarray(history)

beta_gd, history = fit_gd(X_demo, y_demo)
beta_exact = np.linalg.lstsq(X_demo, y_demo, rcond=None)[0]
assert np.allclose(beta_gd, beta_exact, atol=1e-5)
print(beta_gd, len(history), history[-1])
```

</details>

## 9. 本讲小结

- sklearn 用统一的 `fit`、`predict` 接口组织估计器，但数据边界仍需主动设计。
- 梯度指向局部上升最快方向，减去梯度得到下降更新。
- 学习率控制步长；凸性决定局部下降与全局最优的关系。
- 多特征尺度差异会拖慢优化，标准化通常是重要准备。

## 参考资料

- Berkeley Data 100 Summer 2025，Lecture 13：sklearn 工作流、一维与多维梯度下降、学习率和凸性的知识边界。
- *Learning Data Science*，Ch.20：模型训练与优化主题。
- 本页能源站案例、文字、图解、NumPy 实现、问题和交互均为本课程原创；sklearn 代码只展示公共 API 用法。

<DsLessonProgress lesson-id="13" next-label="第 14 讲：特征工程与随机梯度下降" next-href="/courses/data-science/14-feature-engineering" />
