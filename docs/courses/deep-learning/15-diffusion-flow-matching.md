---
course: deep-learning
edition: industrial
title: Diffusion、Latent Diffusion 与 Flow Matching
description: 从前向加噪闭式、噪声预测、SNR 与采样器进入潜空间扩散、classifier-free guidance 和连续速度场。
outline: [2, 3]
---

<DlLessonHero number="15" module="生成模型" title="Diffusion、Latent Diffusion 与 Flow Matching" question="把数据逐步变成噪声再反演，或直接学习连续输运速度场时，训练目标、采样步数与条件控制怎样连接？" duration="建议 320–420 分钟" :outcomes="['推导 q(x_t|x_0) 闭式与 SNR 日程', '区分 ε、x0、v 参数化及 DDPM/DDIM 采样', '解释 latent diffusion 的压缩收益与重构上限', '写出 classifier-free guidance 与 Flow Matching 的质量/多样性/步数交换']" />

## 贯穿案例：条件缺陷图像生成

目标是按缺陷类别、位置和严重度生成 512×512 图像。系统必须登记 VAE 版本、文本/条件编码器、noise schedule、prediction type、sampler、步数、guidance、seed 与安全过滤；否则“同一个模型”无法复现。

<DlQuiz kind="复现合同" question="只记录扩散 checkpoint，是否足以复现样本？" :options="['足够', '不够，还需 VAE/条件编码、schedule、sampler、步数、guidance 与 seed 等', '只缺图片格式', '只缺 batch size']" :answer="1" explanation="扩散推理是模型与数值求解器共同定义的系统。" />

<div class="dl-stop"><strong>停下来推一推 1</strong>同一 seed 改变 sampler 或 guidance 后是否仍应得到同图？哪些状态才真正定义随机轨迹？</div>

## 1. 前向加噪有闭式边缘分布

定义 $q(x_t|x_{t-1})=\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_tI)$，令 $\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{s=1}^t\alpha_s$，则

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.$$

训练无需逐步加噪，可直接随机采样 $t,\epsilon$ 构造 $x_t$。

```python
import numpy as np

betas = np.linspace(1e-4, 0.02, 1000)
alpha_bar = np.cumprod(1.0 - betas)
assert np.all(np.diff(alpha_bar) < 0)
t = 499
signal, noise = np.sqrt(alpha_bar[t]), np.sqrt(1 - alpha_bar[t])
assert np.isclose(signal**2 + noise**2, 1.0)
```

<NoiseScheduleLab />

<DlQuiz kind="前向过程" question="训练时获得 x_t 是否必须从 x_0 连续模拟 t 次？" :options="['必须', '不必须，可用 ᾱ_t 闭式一次采样', '只在 GAN 中不必', '必须先解 ODE']" :answer="1" explanation="Gaussian 链的边缘分布有闭式，这是随机 t 训练高效的关键。" />

## 2. ε、x0 与 v 参数化

模型可预测噪声 $\epsilon$、干净样本 $x_0$ 或速度组合 $v=\sqrt{\bar\alpha}\epsilon-\sqrt{1-\bar\alpha}x_0$。它们可代数转换，但相同 MSE 会对不同 SNR 区域加不同权重。min-SNR 等策略重新分配梯度。

```python
import numpy as np

x0, eps, alpha_bar = 0.7, -0.4, 0.36
xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1-alpha_bar) * eps
recovered = (xt - np.sqrt(1-alpha_bar) * eps) / np.sqrt(alpha_bar)
assert np.isclose(recovered, x0)
```

<DlQuiz kind="参数化" question="ε-prediction 与 x0-prediction 是否只是改输出名字？" :options="['是', '不是，虽可互换，但损失权重、数值尺度和不同 SNR 梯度会改变', '只影响文件名', '只影响 VAE']" :answer="1" explanation="参数化与 weighting 共同定义训练信号。" />

## 3. 反向采样与步数

DDPM 祖先采样带随机噪声；DDIM 可沿非马尔可夫、可确定路径用更少步。DPM-Solver 等把过程视作 ODE/SDE 数值求解。步数不能脱离 solver 与误差容忍比较，NFE 比“迭代次数”更通用。

<div class="dl-stop"><strong>停下来推一推 2</strong>20 步 sampler 比 50 步快 2.5 倍吗？列出 U-Net/VAE、编译、batch、CFG 双前向与调度开销。</div>

## 4. Latent Diffusion 把去噪搬到压缩空间

预训练 autoencoder 将图像 $x$ 编码为低分辨率 latent $z$，扩散模型在 $z$ 上训练，最后 decoder 还原图像。空间压缩 $f$ 倍使 attention/卷积成本显著下降，但细节上限受 VAE 重构控制。

条件通过 cross-attention 或拼接进入 U-Net/Transformer。文本编码器、tokenizer 与空条件 embedding 都是 checkpoint 接口，不能只换 prompt 而忽略版本。

<DlQuiz kind="潜空间边界" question="Latent Diffusion 的最终细节上限受什么额外部件限制？" :options="['只受随机种子', 'autoencoder 的压缩与重构质量', '只受类别数', '不受任何限制']" :answer="1" explanation="扩散只在 latent 中生成，decoder 无法恢复被编码器系统性丢失的信息。" />

## 5. Classifier-free guidance

训练时随机丢弃条件，同时学习条件与无条件预测。推理组合

$$\hat\epsilon=\epsilon_{uncond}+s(\epsilon_{cond}-\epsilon_{uncond}).$$

$s$ 增大通常强化条件对齐，也可能过饱和、降低多样性并放大伪影。CFG 常需一次批内双分支或两次预测，延迟账本要计入。

```python
import numpy as np

uncond = np.array([0.2, -0.1])
cond = np.array([0.5, 0.3])
guided = uncond + 4.0 * (cond - uncond)
assert np.allclose(guided, [1.4, 1.5])
```

## 6. Flow Matching：回归速度场

选择从简单先验到数据的概率路径 $p_t$，构造条件速度 $u_t$，训练

$$L_{FM}=\mathbb E_{t,x_t}\lVert v_\theta(x_t,t)-u_t(x_t)\rVert^2.$$

生成时求解 $dx/dt=v_\theta(x,t)$。更直的路径可减少数值步数；Rectified Flow 等方法关注直化。它与 diffusion 都学习随时间变化的向量场，但目标构造和采样方程不同。

<FlowTransportLab />

<DlQuiz kind="连续流" question="Flow Matching 采样时主要解什么？" :options="['离散分类树', '由学习速度场定义的 ODE', 'K-means', '码本最近邻']" :answer="1" explanation="连续时间状态沿速度场从先验输运到数据分布。" />

<div class="dl-stop"><strong>停下来推一推 3</strong>更少 NFE 的 flow 模型 FID 略差但 P99 延迟减半。按交互生成与离线批量生成分别制定门禁。</div>

## 7. 推荐论文阅读

<DlPaperCard title="Denoising Diffusion Probabilistic Models" meta="Ho, Jain & Abbeel · 2020" priority="必读 · DDPM" question="变分目标如何化成随机 timestep 的噪声预测，schedule 进入哪些公式？" reading="精读前向闭式、简化目标与采样算法。" href="https://arxiv.org/pdf/2006.11239" />

<DlPaperCard title="High-Resolution Image Synthesis with Latent Diffusion Models" meta="Rombach et al. · 2021" priority="必读 · 潜空间扩散" question="感知压缩与生成建模如何分工，cross-attention 怎样注入条件？" reading="读 autoencoder 压缩消融、latent U-Net 与计算质量曲线。" href="https://arxiv.org/pdf/2112.10752" />

<DlPaperCard title="Flow Matching for Generative Modeling" meta="Lipman et al. · 2022" priority="必读 · 连续流" question="条件概率路径如何给出可监督速度目标，OT 路径为何可能更高效？" reading="先读 vector field 与 conditional flow matching，再看路径对照。" href="https://arxiv.org/pdf/2210.02747" />

## 8. 练习与小结

1. 验证前向闭式；2. 绘制 schedule 的 log-SNR；3. 换算 ε/x0/v；4. 画步数—FID—延迟 Pareto；5. 切片扫描 CFG；6. 比较 diffusion sampler 与 flow ODE 的 NFE。

## 本章小结

- 前向 Gaussian 链允许任意 $t$ 闭式加噪，SNR 而非 timestep 名称决定信号状态。
- prediction type、loss weighting、sampler 与 NFE 共同定义扩散系统。
- Latent Diffusion 用 autoencoder 换取计算效率，也继承其重构上限。
- CFG 交换条件对齐与多样性；Flow Matching 直接回归连续输运速度场。

<DlLessonProgress lesson-id="15" />
