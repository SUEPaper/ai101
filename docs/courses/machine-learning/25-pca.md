---
title: 25. PCA 为什么等于寻找特征向量
description: 从最大方差、最小重构误差与 SVD 三条路径推导 PCA
course: machine-learning
---

<MlLessonHero
  eyebrow="第七模块 · 降维与聚类"
  title="PCA 为什么等于寻找特征向量"
  subtitle="PCA 寻找数据云延展最明显的正交方向；同一组方向也恰好让线性重构误差最小。"
  :diagnostics="['能用 Rayleigh 商推导第一主成分', '能证明最大方差与最小重构等价', '能用 SVD 实现并避免数据泄漏']"
/>

## 1. 第一步永远是中心化

数据矩阵 $\mathbf X\in\mathbb R^{n\times d}$，样本均值

$$
\boldsymbol\mu=\frac1n\sum_{i=1}^{n}\mathbf x_i.
$$

中心化矩阵

$$
\mathbf X_c=\mathbf X-\mathbf 1\boldsymbol\mu^\top.
$$

使用分母 $n$ 的样本协方差：

$$
\mathbf S=\frac1n\mathbf X_c^\top\mathbf X_c.
$$

若不中心化，PCA 会寻找相对原点的最大二阶矩方向，而非围绕数据均值的变化方向。标准化不是必选步骤：它把协方差 PCA 变成相关系数 PCA，适合单位不可比但也会改变问题。

<MlLab
  kind="pca"
  title="实验：主方向如何随数据云旋转"
  description="调节相关性与降维维数，观察特征向量、解释方差和重构误差。"
/>

## 2. 最大方差推导

寻找单位方向 $\mathbf w$，把样本投影为

$$
z_i=\mathbf w^\top(\mathbf x_i-\boldsymbol\mu).
$$

投影方差：

$$
\frac1n\sum_i z_i^2
=\frac1n\sum_i
\mathbf w^\top(\mathbf x_i-\boldsymbol\mu)
(\mathbf x_i-\boldsymbol\mu)^\top\mathbf w
=\mathbf w^\top\mathbf S\mathbf w.
$$

约束 $\|\mathbf w\|_2=1$，Lagrangian：

$$
\mathcal L(\mathbf w,\lambda)
=\mathbf w^\top\mathbf S\mathbf w
-\lambda(\mathbf w^\top\mathbf w-1).
$$

求导：

$$
\nabla_{\mathbf w}\mathcal L
=2\mathbf S\mathbf w-2\lambda\mathbf w=0,
$$

所以

$$
\mathbf S\mathbf w=\lambda\mathbf w.
$$

驻点是协方差的特征向量，目标值

$$
\mathbf w^\top\mathbf S\mathbf w
=\lambda\mathbf w^\top\mathbf w=\lambda.
$$

最大方差方向就是最大特征值对应的特征向量。

::: ml-stop 停下来推一推 1
若协方差矩阵为对角矩阵 $\operatorname{diag}(9,1)$，第一主成分是哪条轴？投影方差是多少？
:::

<MlQuiz
  question="PCA 第一主成分对应协方差矩阵的哪个特征向量？"
  :options="['最小特征值', '最大特征值', '特征值为 0', '任意特征向量']"
  :answer="1"
  explanation="单位方向投影方差是 Rayleigh 商，在最大特征值的特征向量处取最大。"
/>

## 3. 多个主成分

取 $\mathbf W\in\mathbb R^{d\times q}$，列向量正交：

$$
\mathbf W^\top\mathbf W=\mathbf I_q.
$$

总投影方差为

$$
\operatorname{tr}(\mathbf W^\top\mathbf S\mathbf W).
$$

最优 $\mathbf W$ 由前 $q$ 个最大特征值的特征向量组成。解释方差比：

$$
\operatorname{EVR}_j
=\frac{\lambda_j}{\sum_{r=1}^{d}\lambda_r},
$$

累计解释方差：

$$
\operatorname{CEVR}(q)
=\frac{\sum_{j=1}^{q}\lambda_j}
{\sum_{r=1}^{d}\lambda_r}.
$$

它衡量保留了总方差的多少，不等于保留了标签预测信息的多少：低方差方向也可能恰好区分类别。

<MlQuiz
  question="累计解释方差达到 95%，能否保证保留 95% 的分类信息？"
  :options="['能', '不能；PCA 不看标签，低方差方向也可能有判别信息', '只在二分类能', '只要标准化就能']"
  :answer="1"
  explanation="解释方差是无监督几何指标，与下游标签信息不等价。"
/>

## 4. 最小重构误差与最大方差等价

投影后重构：

$$
\hat{\mathbf x}_i
=\boldsymbol\mu+\mathbf W\mathbf W^\top
(\mathbf x_i-\boldsymbol\mu).
$$

因为 $\mathbf P=\mathbf W\mathbf W^\top$ 是正交投影矩阵，中心化样本分解为投影与正交残差：

$$
\|\mathbf x_{c,i}\|_2^2
=\|\mathbf P\mathbf x_{c,i}\|_2^2
+\|(\mathbf I-\mathbf P)\mathbf x_{c,i}\|_2^2.
$$

对样本求和：

$$
\sum_i\|\mathbf x_{c,i}-\hat{\mathbf x}_{c,i}\|_2^2
=\sum_i\|\mathbf x_{c,i}\|_2^2
-\sum_i\|\mathbf W^\top\mathbf x_{c,i}\|_2^2.
$$

第一项与 $\mathbf W$ 无关，因此最小化重构误差等价于最大化投影方差。

进一步，最优 $q$ 维重构的平均平方误差为被丢弃特征值之和：

$$
\frac1n\|\mathbf X_c-\mathbf X_c\mathbf W\mathbf W^\top\|_F^2
=\sum_{j=q+1}^{d}\lambda_j.
$$

::: ml-stop 停下来推一推 2
特征值为 $[10,4,1]$，降到 2 维。保留方差比与平均重构误差分别是多少？
:::

<MlQuiz
  question="PCA 最优 q 维线性重构丢失的平均平方误差等于什么？"
  :options="['保留特征值之和', '被丢弃特征值之和', '最大特征值平方', '样本均值']"
  :answer="1"
  explanation="正交方向能量由特征值分解，保留前 q 个后，残差能量是其余特征值之和。"
/>

## 5. SVD 是更直接的数值实现

中心化矩阵的薄 SVD：

$$
\mathbf X_c=\mathbf U\boldsymbol\Sigma\mathbf V^\top.
$$

则

$$
\mathbf S
=\frac1n\mathbf X_c^\top\mathbf X_c
=\mathbf V\frac{\boldsymbol\Sigma^2}{n}\mathbf V^\top.
$$

所以右奇异向量 $\mathbf V$ 就是主方向，协方差特征值为

$$
\lambda_j=\frac{\sigma_j^2}{n}.
$$

直接对 $\mathbf X_c$ 做 SVD 避免显式形成 $\mathbf X_c^\top\mathbf X_c$ 后条件数平方的问题。若只要少数主成分，可用截断或随机 SVD。

<MlQuiz
  question="对中心化数据矩阵做 SVD 时，PCA 主方向对应什么？"
  :options="['左奇异向量 U', '右奇异向量 V 的列', '奇异值之和', '数据均值']"
  :answer="1"
  explanation="X_c^T X_c=VΣ²V^T，协方差的特征向量是右奇异向量。"
/>

## 6. 白化不是普通 PCA 的必然步骤

PCA 坐标

$$
\mathbf z_i=\mathbf W_q^\top(\mathbf x_i-\boldsymbol\mu)
$$

各维不相关，方差为 $\lambda_j$。白化进一步缩放：

$$
\widetilde z_{ij}
=\frac{z_{ij}}{\sqrt{\lambda_j+\epsilon}},
$$

使保留维的协方差近似单位阵。它可能放大小特征值方向上的噪声，应根据下游模型决定。

## 7. 泄漏与部署

均值、标准差、主方向和维数都必须只用训练数据拟合。正确流程：

1. 在训练折估计 $\boldsymbol\mu$ 与 $\mathbf W$；
2. 用同一参数变换验证折；
3. 用验证选择 $q$；
4. 最终在训练+验证上重拟合，再一次性评价测试集。

<MlQuiz
  question="在交叉验证前先对全部数据做 PCA，主要问题是什么？"
  :options="['PCA 不能用于无标签数据', '验证折的协方差信息泄漏到训练变换', '特征值会变成负数', '样本数一定减少']"
  :answer="1"
  explanation="即使不使用标签，预处理也从验证输入分布获得了信息，评价不再严格样本外。"
/>

::: ml-stop 停下来推一推 3
PCA 特征向量乘以 $-1$ 仍是同一主方向。为什么两个库输出符号相反不代表谁算错了？
:::

## 8. 最小 SVD 实现

~~~python
import numpy as np

def fit_pca(x, components):
    mean = x.mean(axis=0)
    xc = x - mean
    _, singular, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:components].T
    eigenvalues = singular**2 / x.shape[0]
    return mean, basis, eigenvalues

def transform_pca(x, mean, basis):
    return (x - mean) @ basis
~~~

## 9. 动手练习：三条路径核对同一答案

在二维相关高斯数据上：

1. 用协方差特征分解求主方向；
2. 用 SVD 求主方向；
3. 网格搜索单位角度，找最大投影方差；
4. 比较三者，允许符号翻转；
5. 验证丢弃特征值之和等于重构误差；
6. 加一个量纲大但噪声强的特征，对比标准化前后。

<MlLessonProgress
  storage-key="25-pca"
  next-href="/courses/machine-learning/26-kmeans"
  next-label="下一章：K-means 为什么交替两步"
/>
