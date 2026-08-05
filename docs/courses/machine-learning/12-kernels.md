---
course: machine-learning
title: 对偶问题为什么带来核技巧
description: 从软间隔 SVM 的 Lagrangian 完整推导对偶与 KKT 条件，再用 Gram 矩阵、RBF 和多项式核理解隐式特征空间。
outline: [2, 3]
---

<MlLessonHero number="12" title="对偶问题为什么带来核技巧" question="不显式构造高维特征，为什么只计算样本两两相似度就能得到非线性边界？" duration="建议 360–440 分钟" :outcomes="['从 SVM 原始问题逐步消元得到对偶', '用 KKT 条件解释支持向量状态', '判断 Gram 矩阵半正定并分析核方法复杂度']" />

## 诊断

<MlQuiz question="SVM 对偶中训练样本怎样进入目标？" :options="['只通过内积 xᵢᵀxⱼ', '只通过均值', '完全不使用特征', '只通过标签数量']" :answer="0" explanation="这为把内积替换为核函数 K(xᵢ,xⱼ) 提供入口。" />
<MlQuiz question="有效核的 Gram 矩阵至少应满足什么？" :options="['反对称', '对称半正定', '每项都大于 1', '行列式必须为 0']" :answer="1" explanation="它应能表示某个特征空间中的内积。" />
<MlQuiz question="RBF 的 γ 很大时模型通常怎样？" :options="['相似度衰减很快，边界更局部', '所有点都相似', '退化为常数核', '不再半正定']" :answer="0" explanation="除非常接近，否则 exp(−γ距离²) 接近 0。" />

## 1. 从软间隔原始问题开始

$$
\begin{aligned}
\min_{w,b,\xi}\quad&\frac12\lVert w\rVert^2+C\sum_i\xi_i\\
\text{s.t.}\quad&1-\xi_i-y_i(w^Tx_i+b)\le0,\\
&-\xi_i\le0.
\end{aligned}
$$

为两组约束分别引入乘子 $\alpha_i\ge0$、$\mu_i\ge0$：

$$
\mathcal{L}=\frac{1}{2}\lVert w\rVert^2+C\sum_i\xi_i
+\sum_i\alpha_i\left[1-\xi_i-y_i(w^Tx_i+b)\right]
-\sum_i\mu_i\xi_i.
$$

对原始变量取驻点。

### 对 w

$$
\frac{\partial\mathcal{L}}{\partial w}
=w-\sum_i\alpha_iy_ix_i=0
\Rightarrow
\boxed{w=\sum_i\alpha_iy_ix_i}.
$$

### 对 b

$$
\frac{\partial\mathcal{L}}{\partial b}
=-\sum_i\alpha_iy_i=0
\Rightarrow
\boxed{\sum_i\alpha_iy_i=0}.
$$

### 对 ξᵢ

$$
\frac{\partial\mathcal{L}}{\partial\xi_i}
=C-\alpha_i-\mu_i=0.
$$

因 $\mu_i\ge0$，得到

$$
\boxed{0\le\alpha_i\le C}.
$$

<div class="ml-stop"><strong>停下来推一推 1</strong>为什么硬间隔对偶只有 $\alpha_i\ge0$ 而没有上界 C？因为原问题没有 $C\xi_i$ 与 $\xi_i\ge0$ 这组条件。</div>

## 2. 代回得到对偶目标

把 $w=\sum_i\alpha_iy_ix_i$ 代回。记

$$
\lVert w\rVert^2
=\sum_i\sum_j\alpha_i\alpha_jy_iy_jx_i^Tx_j.
$$

线性 $w$ 项为

$$
-\sum_i\alpha_iy_iw^Tx_i=-w^Tw.
$$

与 $\frac12w^Tw$ 合并为 $-\frac12w^Tw$；$b$ 与 $\xi$ 项由驻点条件消失。得到

$$
\boxed{
\begin{aligned}
\max_\alpha\quad&\sum_i\alpha_i
-\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_jx_i^Tx_j\\
\text{s.t.}\quad&0\le\alpha_i\le C,\\
&\sum_i\alpha_iy_i=0.
\end{aligned}}
$$

对偶只有样本内积，没有单独出现 $w$。

## 3. KKT 条件解释支持向量

凸问题满足适当条件时，KKT 是最优的充要条件：

1. 原始可行：$y_if_i\ge1-\xi_i,\xi_i\ge0$。
2. 对偶可行：$0\le\alpha_i\le C$。
3. 驻点条件：上述三组导数为零。
4. 互补松弛：

$$
\alpha_i[1-\xi_i-y_if_i]=0,
\qquad
\mu_i\xi_i=0.
$$

由此分类：

- $\alpha_i=0$：样本通常在间隔外，不直接决定边界。
- $0<\alpha_i<C$：$\mu_i=C-\alpha_i>0$，故 $\xi_i=0$；又因 $\alpha_i>0$，$y_if_i=1$，恰在间隔上。
- $\alpha_i=C$：可能在间隔内或错分，是有界支持向量。

对任意 $0<\alpha_i<C$ 的样本，

$$
b=y_i-\sum_j\alpha_jy_jK(x_j,x_i).
$$

实践中对多个自由支持向量的 $b$ 取平均更稳定。

<MlQuiz question="若 0<αᵢ<C，该样本满足什么？" :options="['yᵢfᵢ=1 且 ξᵢ=0', '一定错分', 'αᵢ=0', '不受 KKT 约束']" :answer="0" explanation="两组互补松弛共同推出它恰在间隔边界。" />

## 4. 核技巧

若显式特征映射为 $\phi(x)$，对偶只需要

$$
\phi(x_i)^T\phi(x_j).
$$

定义核

$$
K(x,z)=\phi(x)^T\phi(z),
$$

即可不构造 $\phi$ 而计算内积。决策函数为

$$
\boxed{f(x)=\sum_{i\in SV}\alpha_iy_iK(x_i,x)+b}.
$$

只有 $\alpha_i>0$ 的支持向量参与预测。

<MlLab kind="kernel" />

## 5. 常用核对应什么表示

### 多项式核

$$
K(x,z)=(x^Tz+c)^d.
$$

它隐式包含最高 $d$ 次的单项式和交互。以 $d=2,c=1$、二维输入为例：

$$
(x^Tz+1)^2
=1+2x_1z_1+2x_2z_2+x_1^2z_1^2+2x_1x_2z_1z_2+x_2^2z_2^2,
$$

可读出一个带适当缩放的显式特征映射。

### RBF 核

$$
K(x,z)=\exp(-\gamma\lVert x-z\rVert^2).
$$

对应无限维特征空间。$\gamma$ 小时相似度变化缓慢、边界平滑；大时影响范围窄、模型容易局部记忆。

## 6. 为什么要检查半正定

任取样本得到 Gram 矩阵 $K_{ij}=K(x_i,x_j)$。若它来自内积，对任意系数 $c$：

$$
c^TKc
=\sum_{ij}c_ic_j\phi(x_i)^T\phi(x_j)
=\left\lVert\sum_ic_i\phi(x_i)\right\rVert^2\ge0.
$$

所以 Gram 矩阵对称半正定。反过来，满足适当条件的对称半正定核对应某个 Hilbert 空间内积。随意定义“相似度”不一定是有效核；不定 Gram 会破坏凸性保证。

<div class="ml-stop"><strong>停下来推一推 2</strong>证明两个有效核之和、乘非负常数仍是有效核。分别从 Gram 矩阵半正定或拼接特征映射解释。</div>

## 7. NumPy：Gram 矩阵审计

```python
import numpy as np

def rbf_kernel(X, Z, gamma):
    X_squared = np.sum(X ** 2, axis=1)[:, None]
    Z_squared = np.sum(Z ** 2, axis=1)[None, :]
    squared_distance = X_squared + Z_squared - 2 * X @ Z.T
    return np.exp(-gamma * np.maximum(squared_distance, 0))

rng = np.random.default_rng(12)
X = rng.normal(size=(40, 3))
K = rbf_kernel(X, X, gamma=.8)
eigenvalues = np.linalg.eigvalsh((K + K.T) / 2)

assert np.allclose(K, K.T)
assert eigenvalues.min() > -1e-10
assert np.allclose(np.diag(K), 1)
```

浮点误差可能产生极小负特征值；数量级和对称误差要一起判断，不应盲目截断掩盖错误核。

## 8. 计算代价与近似

完整 Gram 矩阵需要 $O(n^2)$ 内存；通用二次规划训练可高于 $O(n^2)$，预测成本与支持向量数成正比。大样本时可考虑：

- 显式随机 Fourier 特征近似 RBF，再训练线性模型。
- Nyström 采样近似 Gram。
- 直接使用适合大规模的线性/神经表示。

核方法的优势是中小数据上强几何先验和凸优化；它不是“免费无限维”。

## 9. 练习：从显式特征验证核

对二维二次齐次核 $K(x,z)=(x^Tz)^2$，构造

$$
\phi(x)=[x_1^2,\sqrt2x_1x_2,x_2^2]^T
$$

并证明 $\phi(x)^T\phi(z)=K(x,z)$。用随机数据数值比较两种 Gram。

<details><summary>参考断言</summary>

```python
def quadratic_features(X):
    return np.column_stack([
        X[:, 0] ** 2,
        np.sqrt(2) * X[:, 0] * X[:, 1],
        X[:, 1] ** 2,
    ])

X2 = rng.normal(size=(12, 2))
explicit = quadratic_features(X2) @ quadratic_features(X2).T
implicit = (X2 @ X2.T) ** 2
assert np.allclose(explicit, implicit)
```

</details>

<div class="ml-stop"><strong>停下来推一推 3</strong>RBF 中同时放大所有特征尺度与放大 γ 有何关系？这说明为什么核方法前必须定义有意义的特征尺度。</div>

## 10. 常见误解

- 核技巧不是把数据“真的搬到电脑里的无限维向量”，而是只计算所需内积。
- 训练凸不等于超参数 $C,\gamma$ 自动正确。
- RBF 高表达力会带来过拟合与 $O(n^2)$ 存储。
- 支持向量稀疏性取决于数据与 C，最坏可接近全部样本。
- 对偶方便核化，原始问题在 $d\ll n$ 的线性场景可能更便宜。

<MlQuiz question="核 SVM 预测成本主要取决于什么？" :options="['训练样本特征均值', '支持向量数量及每次核计算成本', '类别名字长度', 'Hessian 的迹']" :answer="1" explanation="预测需对每个支持向量计算 K(xᵢ,x)。" />
<MlQuiz question="为何自定义相似度不一定能当核？" :options="['相似度必须是整数', '其所有 Gram 矩阵未必对称半正定', '核不能比较样本', '只有 RBF 是核']" :answer="1" explanation="不定 Gram 可能不对应任何内积空间，并破坏 SVM 对偶的凸性。" />

## 11. 小结

SVM 对偶把参数表示成训练样本加权和，目标只依赖两两内积。用有效核替换内积即可获得非线性表示；KKT 条件精确描述哪些样本成为支持向量。

<MlLessonProgress lesson-id="12" next-label="第 13 章：复合线性层怎样表达复杂函数" next-href="/courses/machine-learning/13-mlp" />
