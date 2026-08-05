<script setup lang="ts">
import { computed, reactive, watch } from 'vue'

interface Control { label: string; min: number; max: number; step: number; initial: number; unit?: string }
interface Definition { index: string; title: string; question: string; formula: string; a: Control; b: Control; xLabel: string }
interface Metric { label: string; value: string }
interface Result { series: number[]; compare?: number[]; metrics: Metric[]; note: string }

const props = defineProps<{ kind: string }>()
const c = (label: string, min: number, max: number, step: number, initial: number, unit = ''): Control => ({ label, min, max, step, initial, unit })

const labDefinitions: Record<string, Definition> = {
  risk: { index: '01', title: '错误代价会把最优决策推向哪里', question: '漏报越来越昂贵时，阈值还应该固定在 0.5 吗？', formula: 'R(h)=E[L(Y,h(X))]', a: c('判为正类的阈值', .1, .9, .01, .5), b: c('漏报/误报代价比', 1, 12, .5, 4, '×'), xLabel: '候选阈值' },
  optimizer: { index: '02', title: '学习率与曲率共同决定稳定性', question: '为什么同一个学习率在狭长损失面上会振荡？', formula: 'wₜ₊₁=(1−ηλ)wₜ', a: c('学习率 η', .02, 1.2, .01, .28), b: c('最大曲率 λ', .5, 5, .1, 2.4), xLabel: '迭代步' },
  generalization: { index: '03', title: '训练误差下降不等于风险下降', question: '模型复杂度增加到什么时候应该停？', formula: 'E[(Y−f̂)²]=Bias²+Variance+σ²', a: c('选择的复杂度', 1, 12, 1, 5), b: c('噪声强度', 0, 1, .05, .3), xLabel: '模型复杂度' },
  ols: { index: '04', title: '投影、正交与条件数联动', question: '当特征列接近共线时，预测与系数分别发生什么？', formula: 'Xᵀ(y−Xβ̂)=0', a: c('投影方向', 0, 90, 1, 32, '°'), b: c('列相关系数', 0, .98, .01, .55), xLabel: '候选投影方向' },
  likelihood: { index: '05', title: '从高斯似然看平方损失', question: '噪声尺度改变后，同样的残差有多“不可能”？', formula: '−log p(y|μ,σ²)=Σ(yᵢ−μ)²/(2σ²)+n log σ', a: c('候选均值 μ', -2, 2, .05, .2), b: c('噪声 σ', .2, 2, .05, .8), xLabel: '候选均值' },
  regularization: { index: '06', title: '正则化让不同方向收缩多少', question: '为什么病态方向会被 Ridge 更明显地压缩？', formula: 'β̂λ=(XᵀX+λI)⁻¹Xᵀy', a: c('正则强度 λ', 0, 10, .1, 2), b: c('小特征值', .05, 2, .05, .3), xLabel: '特征值方向' },
  basis: { index: '07', title: '基函数复杂度与验证误差', question: '多项式次数越高，拟合能力一定越好吗？', formula: 'f(x)=Σⱼβⱼφⱼ(x)', a: c('多项式次数', 1, 14, 1, 5), b: c('Ridge λ', 0, 2, .05, .25), xLabel: '多项式次数' },
  decision: { index: '08', title: '代价矩阵决定概率阈值', question: '后验概率相同，业务代价不同会得到同一动作吗？', formula: '预测 1 ⇔ P(Y=1|x) > C₀₁/(C₀₁+C₁₀)', a: c('漏报代价 C₀₁', 1, 20, 1, 8), b: c('误报代价 C₁₀', 1, 20, 1, 3), xLabel: '后验概率' },
  logistic: { index: '09', title: '参数、概率曲线与交叉熵联动', question: '完全分离时，为什么权重会继续变大？', formula: 'p(y=1|x)=σ(wx+b)', a: c('权重 w', -6, 6, .1, 2.2), b: c('截距 b', -4, 4, .1, -.4), xLabel: '输入 x' },
  softmax: { index: '10', title: '温度与 log-sum-exp 稳定化', question: '所有 logit 同时加上常数，概率为什么不变？', formula: 'softmax(z)ᵢ=exp(zᵢ−m)/Σⱼexp(zⱼ−m)', a: c('温度 T', .2, 3, .05, 1), b: c('最大 logit', 10, 1000, 10, 120), xLabel: '类别索引' },
  margin: { index: '11', title: '间隔、违约与 C 的取舍', question: '惩罚违约更重时，间隔为什么可能变窄？', formula: 'min ½‖w‖²+CΣξᵢ', a: c('惩罚 C', .1, 20, .1, 3), b: c('离群点比例', 0, .35, .01, .08), xLabel: '候选间隔' },
  kernel: { index: '12', title: 'RBF 宽度与 Gram 矩阵', question: 'γ 太大时，每个样本为什么像一座孤岛？', formula: 'K(x,z)=exp(−γ‖x−z‖²)', a: c('RBF γ', .05, 8, .05, 1.2), b: c('样本距离', 0, 4, .05, 1), xLabel: '样本距离' },
  network: { index: '13', title: '隐藏单元怎样弯折决策边界', question: '没有非线性激活时，层数增加为什么没有帮助？', formula: 'W₂(W₁x+b₁)+b₂=(W₂W₁)x+b′', a: c('隐藏单元数', 1, 32, 1, 6), b: c('激活斜率', .1, 3, .1, 1), xLabel: '输入位置' },
  backprop: { index: '14', title: '深度、局部导数与梯度流', question: '多个局部 Jacobian 连乘后，梯度会剩多少？', formula: '∂L/∂x=J₁ᵀJ₂ᵀ⋯Jₗᵀ∂L/∂hₗ', a: c('网络深度 L', 1, 40, 1, 12), b: c('平均局部导数', .2, 1.4, .02, .78), xLabel: '层编号' },
  convolution: { index: '15', title: '卷积尺寸、参数共享与感受野', question: '改变核与步幅时，输出和参数量怎样变化？', formula: 'Hout=⌊(H+2P−K)/S⌋+1', a: c('卷积核 K', 1, 11, 2, 3), b: c('步幅 S', 1, 4, 1, 1), xLabel: '堆叠层数' },
  recurrence: { index: '16', title: '时间展开与 Jacobian 连乘', question: '谱半径略小于 1，长序列梯度会怎样？', formula: '∂hₜ/∂hₖ=∏ⱼ₌ₖ₊₁ᵗ Jⱼ', a: c('序列长度 T', 2, 80, 1, 30), b: c('状态 Jacobian 范数', .5, 1.2, .01, .92), xLabel: '时间步' },
  attention: { index: '17', title: '缩放、Mask 与注意力复杂度', question: '为什么点积要除以 √dₖ？', formula: 'Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V', a: c('序列长度 n', 4, 128, 4, 32), b: c('键维度 dₖ', 4, 256, 4, 64), xLabel: '键位置' },
  knn: { index: '18', title: '邻居数与维度灾难', question: '维度增加后，最近和最远样本为什么越来越像？', formula: 'ŷ(x)=mode{yᵢ:i∈Nₖ(x)}', a: c('邻居数 k', 1, 31, 2, 7), b: c('特征维度 d', 1, 100, 1, 8), xLabel: '邻居数' },
  tree: { index: '19', title: '阈值、纯度下降与剪枝路径', question: '继续切分带来的训练收益是否值得复杂度？', formula: 'ΔI=I(parent)−Σₘ(nₘ/n)I(childₘ)', a: c('最大深度', 1, 16, 1, 5), b: c('剪枝 α', 0, .2, .005, .03), xLabel: '树深度' },
  forest: { index: '20', title: '树数、相关性与集成方差', question: '树已经很多时，为什么降低相关性更重要？', formula: 'Var(mean)=σ²[ρ+(1−ρ)/B]', a: c('树数量 B', 1, 300, 1, 80), b: c('树间相关 ρ', 0, 1, .01, .28), xLabel: '树数量' },
  boosting: { index: '21', title: '样本权重、轮数与加性模型', question: '弱学习器怎样把注意力移向被错分样本？', formula: 'Dₜ₊₁(i)∝Dₜ(i)exp(−αₜyᵢhₜ(xᵢ))', a: c('Boosting 轮数', 1, 100, 1, 25), b: c('学习率', .01, 1, .01, .2), xLabel: '迭代轮数' },
  estimate: { index: '22', title: '样本与先验怎样共同决定 MAP', question: '样本很少时，先验相当于多少次额外观察？', formula: 'pMAP=(s+α−1)/(n+α+β−2)', a: c('样本量 n', 2, 200, 1, 20), b: c('先验强度 α+β', 2, 80, 1, 12), xLabel: '观测成功率' },
  gda: { index: '23', title: '协方差假设决定边界形状', question: '共享协方差为什么得到线性边界？', formula: 'log p(x|k)=−½(x−μₖ)ᵀΣₖ⁻¹(x−μₖ)−½log|Σₖ|+c', a: c('协方差差异', 0, 2, .05, .4), b: c('类别先验 P(Y=1)', .05, .95, .01, .5), xLabel: '输入位置' },
  em: { index: '24', title: '责任度与似然单调上升', question: '两个分量接近时，软分配为什么更不确定？', formula: 'rᵢk=πₖN(xᵢ|μₖ,Σₖ)/ΣⱼπⱼN(xᵢ|μⱼ,Σⱼ)', a: c('均值间距', .2, 6, .1, 2.4), b: c('初始偏移', 0, 5, .1, 1.4), xLabel: 'EM 迭代' },
  pca: { index: '25', title: '投影方差与重构误差是一件事', question: '主轴偏离数据方向时，两种目标怎样同步变化？', formula: 'max ‖Xw‖² s.t. ‖w‖=1', a: c('投影角度', 0, 180, 1, 35, '°'), b: c('长短轴方差比', 1, 12, .1, 5), xLabel: '候选方向' },
  kmeans: { index: '26', title: '分配与更新为何让目标不增', question: '不同初值会走到同一个局部最优吗？', formula: 'J=ΣᵢΣₖrᵢk‖xᵢ−μₖ‖²', a: c('随机初值编号', 1, 20, 1, 4), b: c('簇数量 K', 2, 8, 1, 3), xLabel: '交替迭代' },
  cluster: { index: '27', title: '形状、簇数与轮廓系数', question: '没有标签时，一个高分指标能定义“真实簇”吗？', formula: 's(i)=[b(i)−a(i)]/max{a(i),b(i)}', a: c('候选簇数 K', 2, 10, 1, 3), b: c('非凸程度', 0, 1, .05, .5), xLabel: '候选簇数' },
  pac: { index: '28', title: '误差、置信度与样本复杂度', question: '把失败概率再减半，需要增加多少样本？', formula: 'n≥log(2|H|/δ)/(2ε²)', a: c('容许误差 ε', .02, .3, .01, .1), b: c('失败概率 δ', .001, .2, .001, .05), xLabel: '样本量' },
  recommend: { index: '29', title: '潜因子、正则化与排序联动', question: '未观测条目为什么不能直接当成 0 分？', formula: 'min Σ(i,j)∈Ω(rᵢⱼ−μ−bᵢ−bⱼ−pᵢᵀqⱼ)²+λ‖Θ‖²', a: c('潜因子维数 d', 1, 40, 1, 8), b: c('正则 λ', 0, 2, .05, .25), xLabel: '潜因子维数' },
  bellman: { index: '30', title: '折扣、backup 与 Bellman residual', question: '为什么 γ<1 会让反复 backup 收敛？', formula: 'V(s)=Σₐπ(a|s)Σs′P(s′|s,a)[r+γV(s′)]', a: c('折扣 γ', 0, .99, .01, .9), b: c('backup 次数', 1, 60, 1, 15), xLabel: 'backup 次数' },
  qlearning: { index: '31', title: '探索、步长与 TD 目标', question: 'SARSA 和 Q-learning 在危险区域为何学出不同路线？', formula: 'Q←Q+α[r+γ maxₐ′Q(s′,a′)−Q]', a: c('学习率 α', .01, 1, .01, .2), b: c('探索率 ε', 0, 1, .01, .15), xLabel: '更新次数' },
  policy: { index: '32', title: 'baseline 怎样降低梯度方差', question: '减去与动作无关的 baseline 为什么不引入偏差？', formula: '∇J=E[∇logπθ(a|s)(Gₜ−b(s))]', a: c('baseline 拟合度', 0, 1, .01, .7), b: c('轨迹数量', 2, 200, 1, 40), xLabel: '轨迹批次' }
}

const state = reactive({ a: 0, b: 0 })
const definition = computed(() => labDefinitions[props.kind] ?? labDefinitions.risk)
watch(definition, (value) => { state.a = value.a.initial; state.b = value.b.initial }, { immediate: true })

const seq = (count: number, fn: (x: number, i: number) => number) => Array.from({ length: count }, (_, i) => fn(i / Math.max(1, count - 1), i))
const fixed = (value: number, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '∞'
const decay = (start: number, rate: number, floor = 0) => seq(28, (_, i) => floor + start * Math.pow(rate, i))

function evaluate(kind: string, a: number, b: number): Result {
  let series: number[] = []
  let compare: number[] | undefined
  let metrics: Metric[] = []
  let note = ''
  switch (kind) {
    case 'risk': {
      series = seq(31, (x) => .3 * (1 - x) + .04 * b * x * x)
      const optimum = Math.min(.95, .3 / (.08 * b))
      metrics = [{ label: '当前经验代价', value: fixed(.3 * (1 - a) + .04 * b * a * a) }, { label: '示例最优阈值', value: fixed(optimum, 2) }, { label: '代价比', value: `${b.toFixed(1)}×` }]
      note = '漏报代价越高，允许判为正类的门槛应越低；0.5 不是脱离代价矩阵的自然常数。'; break
    }
    case 'optimizer': {
      const factor = 1 - a * b
      series = seq(30, (_, i) => .5 * b * Math.pow(factor, 2 * i))
      metrics = [{ label: '更新倍率 |1−ηλ|', value: fixed(Math.abs(factor)) }, { label: '第 20 步损失', value: fixed(series[20], 5) }, { label: '稳定条件', value: Math.abs(factor) < 1 ? '满足' : '不满足' }]
      note = Math.abs(factor) < 1 ? '每步都把误差乘以小于 1 的因子；接近 1 时收敛很慢。' : '更新倍率不小于 1，参数会振荡或发散。'; break
    }
    case 'generalization': {
      series = seq(12, (_, i) => .72 / (i + 1) + .006 * Math.pow(i + 1, 2) + b)
      compare = seq(12, (_, i) => .7 / (i + 1) + .18 * b)
      const k = Math.round(a) - 1
      metrics = [{ label: '训练误差', value: fixed(compare[k]) }, { label: '验证误差', value: fixed(series[k]) }, { label: '泛化间隙', value: fixed(series[k] - compare[k]) }]
      note = '蓝线是验证误差，青线是训练误差；最低训练误差通常不对应最低未来风险。'; break
    }
    case 'ols': {
      series = seq(37, (x) => Math.pow(Math.sin((x * 90 - 38) * Math.PI / 180), 2))
      const residual = Math.pow(Math.sin((a - 38) * Math.PI / 180), 2)
      const condition = (1 + b) / Math.max(.001, 1 - b)
      metrics = [{ label: '残差平方', value: fixed(residual) }, { label: '近似条件数', value: fixed(condition, 1) }, { label: '正交误差', value: residual < .002 ? '≈ 0' : '尚未为 0' }]
      note = '角度控制投影误差，相关系数控制系数解的稳定性；二者是不同问题。'; break
    }
    case 'likelihood': {
      const ys = [-1.2, -.5, -.2, .1, .4, .8, 1.1]
      const nll = (mu: number) => ys.reduce((sum, y) => sum + (y - mu) ** 2 / (2 * b * b) + Math.log(b), 0)
      series = seq(33, (x) => nll(-2 + 4 * x))
      metrics = [{ label: '负对数似然', value: fixed(nll(a)) }, { label: '样本均值', value: fixed(ys.reduce((s, y) => s + y, 0) / ys.length, 2) }, { label: '平方残差权重', value: fixed(1 / (2 * b * b), 2) }]
      note = '固定 σ 时，最大化高斯似然与最小化平方和具有同一个最优 μ。'; break
    }
    case 'regularization': {
      series = seq(30, (x) => x / (x + a + .001))
      const smallShrink = b / (b + a + .001)
      metrics = [{ label: '病态方向保留率', value: fixed(smallShrink) }, { label: '良好方向保留率', value: fixed(5 / (5 + a)) }, { label: '条件数改善', value: fixed((5 + a) / (b + a), 1) }]
      note = 'Ridge 在小特征值方向加入同样的 λ，因此相对收缩更强，也改善数值条件。'; break
    }
    case 'basis': {
      series = seq(14, (_, i) => .8 / (i + 1) + .008 * (i + 1) ** 2 / (1 + 3 * b))
      compare = seq(14, (_, i) => .72 / (i + 1))
      const k = Math.round(a) - 1
      metrics = [{ label: '验证误差', value: fixed(series[k]) }, { label: '训练误差', value: fixed(compare[k]) }, { label: '参数数量', value: String(Math.round(a) + 1) }]
      note = '基函数让模型对参数仍保持线性，却能对输入形成非线性；正则化控制高阶项。'; break
    }
    case 'decision': {
      const threshold = a / (a + b)
      series = seq(31, (p) => Math.min(a * p, b * (1 - p)))
      metrics = [{ label: '最优概率阈值', value: fixed(threshold, 2) }, { label: 'P(Y=1)=0.5 的动作', value: .5 > threshold ? '预测正类' : '预测负类' }, { label: '代价总量', value: fixed(a + b, 0) }]
      note = '比较两个动作的条件风险，交点就是阈值；阈值由错误代价而不是分类器名字决定。'; break
    }
    case 'logistic': {
      series = seq(41, (x) => 1 / (1 + Math.exp(-(a * (-4 + 8 * x) + b))))
      const p0 = 1 / (1 + Math.exp(-b))
      metrics = [{ label: 'x=0 的概率', value: fixed(p0) }, { label: '决策边界 x', value: Math.abs(a) < .01 ? '不存在' : fixed(-b / a, 2) }, { label: '曲线方向', value: a >= 0 ? '递增' : '递减' }]
      note = '权重控制 log odds 的斜率，截距平移边界；概率始终落在 0 到 1 之间。'; break
    }
    case 'softmax': {
      const logits = [b, b - 2, b - 5].map((z) => z / a)
      const max = Math.max(...logits)
      const exp = logits.map((z) => Math.exp(z - max)); const sum = exp.reduce((s, x) => s + x, 0)
      series = exp.map((x) => x / sum)
      metrics = series.map((value, i) => ({ label: `类别 ${i + 1} 概率`, value: fixed(value) }))
      note = `减去最大 logit ${fixed(max, 1)} 不改变比值，却避免直接计算 exp(${fixed(max, 1)}) 的溢出。`; break
    }
    case 'margin': {
      series = seq(30, (x) => 1 / (1 + a * x) + b * x * 1.6)
      const violations = b * 20 / (1 + a)
      metrics = [{ label: '示例间隔宽度', value: fixed(2 / Math.sqrt(1 + a), 2) }, { label: '平均违约', value: fixed(violations, 2) }, { label: '支持向量压力', value: a > 10 ? '高' : '中低' }]
      note = '增大 C 强迫模型更重视训练违约，但在有离群点时可能用更窄间隔换取更少错误。'; break
    }
    case 'kernel': {
      series = seq(41, (x) => Math.exp(-a * (4 * x) ** 2))
      const sim = Math.exp(-a * b * b)
      metrics = [{ label: '当前相似度', value: fixed(sim) }, { label: '有效半径 K≥0.5', value: fixed(Math.sqrt(Math.log(2) / a), 2) }, { label: '局部程度', value: a > 3 ? '很局部' : '较平滑' }]
      note = 'γ 控制一个样本能影响多远；过大时 Gram 矩阵接近单位阵，训练拟合强而外推弱。'; break
    }
    case 'network': {
      series = seq(41, (x) => Math.tanh(b * a / 8 * (-2 + 4 * x)))
      metrics = [{ label: '隐藏参数量', value: String(Math.round(3 * a + 1)) }, { label: '可用折点上界', value: String(Math.round(a)) }, { label: '激活', value: b < .2 ? '近似常数' : '非线性' }]
      note = '表示能力来自激活函数与宽度的配合；删除激活后，任意多线性层都能合并成一层。'; break
    }
    case 'backprop': {
      series = seq(Math.max(2, Math.round(a)), (_, i) => Math.pow(b, i))
      const grad = Math.pow(b, a)
      metrics = [{ label: '输入端梯度范数', value: fixed(grad, 6) }, { label: '对数梯度', value: fixed(Math.log10(Math.max(1e-12, grad)), 2) }, { label: '状态', value: grad < .01 ? '消失' : grad > 10 ? '爆炸' : '可传播' }]
      note = '反向传播复用前向图中的局部导数；连乘解释梯度尺度，分支处则必须相加。'; break
    }
    case 'convolution': {
      const out = Math.floor((32 - a) / b) + 1
      series = seq(8, (_, i) => 1 + i * (a - 1))
      metrics = [{ label: '输出边长', value: String(out) }, { label: '卷积核参数', value: String(Math.round(a * a * 3 * 16)) }, { label: '第 4 层感受野', value: `${fixed(1 + 3 * (a - 1), 0)}×${fixed(1 + 3 * (a - 1), 0)}` }]
      note = '参数量与空间位置无关，这是共享的收益；步幅改变输出密度，堆叠扩大感受野。'; break
    }
    case 'recurrence': {
      series = seq(Math.round(a), (_, i) => Math.pow(b, i))
      const grad = Math.pow(b, a)
      metrics = [{ label: '跨 T 步梯度', value: fixed(grad, 7) }, { label: '半衰期', value: b < 1 ? fixed(Math.log(.5) / Math.log(b), 1) : '不衰减' }, { label: '状态', value: b < .98 ? '易消失' : b > 1.02 ? '易爆炸' : '临界' }]
      note = 'BPTT 是共享参数计算图上的链式法则；门控提供更接近恒等映射的状态路径。'; break
    }
    case 'attention': {
      series = seq(Math.min(32, Math.round(a)), (x) => Math.exp(-5 * Math.abs(x - .65)))
      const memory = a * a
      metrics = [{ label: '注意力分数个数', value: memory.toLocaleString('zh-CN') }, { label: '未缩放点积标准差', value: fixed(Math.sqrt(b), 1) }, { label: '缩放后标准差', value: '≈ 1.0' }]
      note = '除以 √dₖ 把点积尺度拉回稳定范围；代价是显式注意力矩阵随序列长度二次增长。'; break
    }
    case 'knn': {
      series = seq(16, (_, i) => .8 / (2 * i + 1) + .015 * (2 * i + 1) + .002 * b)
      const concentration = 1 - 1 / Math.sqrt(b)
      metrics = [{ label: '示例验证误差', value: fixed(.8 / a + .015 * a + .002 * b) }, { label: '距离集中指数', value: fixed(concentration) }, { label: '查询比较次数', value: 'O(nd)' }]
      note = 'k 小方差高，k 大偏差高；维度升高时距离的区分度下降，缩放与度量选择变得关键。'; break
    }
    case 'tree': {
      series = seq(16, (_, i) => .65 / (i + 1) + b * (2 ** i) / 20)
      compare = seq(16, (_, i) => .62 / (i + 1))
      const depth = Math.round(a)
      metrics = [{ label: '最大叶节点数', value: String(2 ** depth) }, { label: '剪枝惩罚', value: fixed(b * 2 ** depth, 2) }, { label: '验证目标', value: fixed(series[depth - 1]) }]
      note = '青线表示训练误差，蓝线加入复杂度代价；剪枝不是“修剪不好看的枝”，而是在目标中定价叶节点。'; break
    }
    case 'forest': {
      series = seq(30, (x) => b + (1 - b) / (1 + x * (a - 1)))
      const variance = b + (1 - b) / a
      metrics = [{ label: '相对集成方差', value: fixed(variance) }, { label: '无限树方差下界', value: fixed(b) }, { label: '单样本 OOB 概率', value: '≈ 36.8%' }]
      note = '增加树数只能消掉不相关部分；相关性 ρ 是无法靠无限平均消除的方差下界。'; break
    }
    case 'boosting': {
      series = decay(1, Math.max(.5, 1 - b * .45), .02)
      const rounds = Math.round(a)
      metrics = [{ label: '示例指数损失', value: fixed(Math.pow(Math.max(.5, 1 - b * .45), rounds), 5) }, { label: '有效步长', value: fixed(a * b, 1) }, { label: '噪声风险', value: a * b > 25 ? '偏高' : '可控' }]
      note = '每轮拟合当前负梯度或重加权错误；轮数与学习率共同决定函数空间里走了多远。'; break
    }
    case 'estimate': {
      const successRate = .7; const priorMean = .5
      const map = (successRate * a + priorMean * b) / (a + b)
      series = seq(31, (x) => (x * a + priorMean * b) / (a + b))
      metrics = [{ label: 'MLE', value: fixed(successRate) }, { label: 'MAP', value: fixed(map) }, { label: '先验权重', value: fixed(b / (a + b)) }]
      note = 'MAP 可看成数据统计量与先验统计量的加权；样本量增大时，固定强度先验的影响消退。'; break
    }
    case 'gda': {
      series = seq(41, (x) => 1 / (1 + Math.exp(-((-4 + 8 * x) + a * (-4 + 8 * x) ** 2 / 4 + Math.log(b / (1 - b))))))
      metrics = [{ label: '二次项强度', value: fixed(a) }, { label: '边界类型', value: a < .05 ? '近似线性' : '二次曲线' }, { label: '先验 log odds', value: fixed(Math.log(b / (1 - b)), 2) }]
      note = 'LDA 共享协方差，二次项相消；QDA 各类协方差不同，二次项保留。'; break
    }
    case 'em': {
      const rate = .55 + .35 * Math.exp(-a / 2) + .03 * b
      series = seq(24, (_, i) => 1 - Math.pow(Math.min(.98, rate), i + 1))
      const ambiguity = Math.exp(-a * a / 8)
      metrics = [{ label: '责任度不确定性', value: fixed(ambiguity) }, { label: '第 10 轮相对似然', value: fixed(series[9]) }, { label: '收敛状态', value: rate > .93 ? '很慢' : '正常' }]
      note = 'E 步更新软分配，M 步固定软分配优化参数；每步都不降低观测数据似然，但不保证全局最优。'; break
    }
    case 'pca': {
      const theta = (a - 28) * Math.PI / 180
      const variance = b * Math.cos(theta) ** 2 + Math.sin(theta) ** 2
      const total = b + 1
      series = seq(37, (x) => b * Math.cos((x * 180 - 28) * Math.PI / 180) ** 2 + Math.sin((x * 180 - 28) * Math.PI / 180) ** 2)
      metrics = [{ label: '投影方差', value: fixed(variance) }, { label: '解释比例', value: fixed(variance / total) }, { label: '重构误差', value: fixed(total - variance) }]
      note = '总方差固定，因此最大化投影方差与最小化正交重构误差完全等价。'; break
    }
    case 'kmeans': {
      const rate = .5 + ((Math.round(a) * 17) % 10) / 30
      series = decay(8 + b, Math.min(.9, rate), 1 + ((Math.round(a) + Math.round(b)) % 4) / 5)
      metrics = [{ label: '最终簇内平方和', value: fixed(series.at(-1) ?? 0) }, { label: '迭代次数', value: String(series.findIndex((v, i) => i > 0 && Math.abs(v - series[i - 1]) < .01) + 1 || series.length) }, { label: '局部解编号', value: `#${Math.round(a)}` }]
      note = '每个分配步和更新步都精确最小化当前变量块，所以目标不增；初值不同仍可能到达不同局部解。'; break
    }
    case 'cluster': {
      series = seq(9, (_, i) => .72 - .08 * Math.abs((i + 2) - 3) - b * .22 * Math.sin(i * 1.7) ** 2)
      const idx = Math.round(a) - 2
      metrics = [{ label: '轮廓系数', value: fixed(series[idx]) }, { label: '候选簇数', value: String(Math.round(a)) }, { label: '可信度', value: b > .7 ? '需结合稳定性' : '中等' }]
      note = '指标只评价某种几何目标；对非凸结构、不同密度或业务分组，它不等于“发现真实类别”。'; break
    }
    case 'pac': {
      const h = 100; const n = Math.log(2 * h / b) / (2 * a * a)
      series = seq(30, (x) => Math.sqrt(Math.log(2 * h / b) / (2 * (20 + 20 * x * x * n))))
      metrics = [{ label: '所需样本下界', value: Math.ceil(n).toLocaleString('zh-CN') }, { label: '假设集合大小', value: String(h) }, { label: '置信度', value: `${fixed((1 - b) * 100, 1)}%` }]
      note = '误差 ε 以平方出现在分母，要求误差减半通常约需四倍样本；δ 只以对数影响样本量。'; break
    }
    case 'recommend': {
      series = seq(20, (_, i) => .9 / Math.sqrt(i + 1) + .008 * (i + 1) * a / (1 + b))
      const params = (120 + 80) * a + 200
      metrics = [{ label: '潜因子参数', value: params.toLocaleString('zh-CN') }, { label: '示例验证损失', value: fixed(.9 / Math.sqrt(a) + .008 * a * a / (1 + b)) }, { label: '冷启动能力', value: '仅靠 ID 不足' }]
      note = '目标只在观测集合 Ω 上求和；未观测可能是未曝光，不等于明确负反馈。'; break
    }
    case 'bellman': {
      series = decay(1, a)
      const residual = Math.pow(a, b)
      metrics = [{ label: '第 k 轮 residual 上界', value: fixed(residual, 7) }, { label: '有效规划视野', value: a < 1 ? fixed(1 / (1 - a), 1) : '∞' }, { label: '压缩模', value: fixed(a, 2) }]
      note = 'Bellman 算子把最大差异至多乘以 γ；γ<1 时，反复 backup 逼近唯一不动点。'; break
    }
    case 'qlearning': {
      const contraction = Math.max(.05, 1 - a * (1 - .9))
      series = decay(1, contraction, b * .08)
      metrics = [{ label: '当前 TD 误差示例', value: fixed(series[Math.min(27, Math.round(8 + 20 * a))]) }, { label: '随机动作比例', value: `${fixed(b * 100, 0)}%` }, { label: '目标策略', value: '贪心（off-policy）' }]
      note = 'Q-learning 用 max 构造目标，即使行为由 ε-greedy 产生；探索带来覆盖，也带来回报方差。'; break
    }
    case 'policy': {
      const variance = (1 - .85 * a) / Math.sqrt(b)
      series = seq(30, (_, i) => variance * (1 + .45 * Math.sin(i * 2.1)) / Math.sqrt(i + 1))
      compare = seq(30, (_, i) => (1 / Math.sqrt(b)) * (1 + .45 * Math.sin(i * 2.1)) / Math.sqrt(i + 1))
      metrics = [{ label: '梯度标准误示例', value: fixed(variance) }, { label: '方差降低比例', value: `${fixed(85 * a, 0)}%` }, { label: '估计偏差', value: 'baseline 与动作无关时为 0' }]
      note = '蓝线使用 baseline，青线未使用；baseline 不改变期望梯度，但能显著减少有限轨迹估计的抖动。'; break
    }
    default: series = decay(1, .8); metrics = [{ label: '状态', value: '就绪' }]; note = '拖动参数，观察公式对应的数值变化。'
  }
  return { series, compare, metrics, note }
}

const result = computed(() => evaluate(props.kind, state.a, state.b))
const allValues = computed(() => [...result.value.series, ...(result.value.compare ?? [])].filter(Number.isFinite))
const minValue = computed(() => Math.min(...allValues.value, 0))
const maxValue = computed(() => Math.max(...allValues.value, 1e-6))
function points(values: number[]) {
  const span = Math.max(1e-9, maxValue.value - minValue.value)
  return values.map((value, index) => `${22 + index * 316 / Math.max(1, values.length - 1)},${164 - (value - minValue.value) / span * 132}`).join(' ')
}
function display(control: Control, value: number) {
  const digits = control.step < .01 ? 3 : control.step < 1 ? 2 : 0
  return `${value.toFixed(digits)}${control.unit ?? ''}`
}
</script>

<template>
  <section class="ml-lab" :aria-label="`交互实验 ${definition.index}：${definition.title}`">
    <header>
      <div><span>INTERACTIVE {{ definition.index }}</span><h3>{{ definition.title }}</h3><p>{{ definition.question }}</p></div>
      <code>{{ definition.formula }}</code>
    </header>
    <div class="ml-lab__body">
      <div class="ml-lab__chart">
        <svg viewBox="0 0 360 190" role="img" :aria-label="result.note">
          <g class="grid"><line v-for="y in [32,65,98,131,164]" :key="y" x1="22" :y1="y" x2="338" :y2="y" /></g>
          <polyline v-if="result.compare" class="compare" :points="points(result.compare)" />
          <polyline class="primary" :points="points(result.series)" />
          <line x1="22" y1="164" x2="338" y2="164" class="axis" />
          <text x="22" y="183">{{ definition.xLabel }}</text>
        </svg>
        <div class="ml-lab__metrics" aria-label="实验数值摘要">
          <div v-for="metric in result.metrics" :key="metric.label"><span>{{ metric.label }}</span><strong>{{ metric.value }}</strong></div>
        </div>
      </div>
      <div class="ml-lab__controls">
        <label>
          <span>{{ definition.a.label }} <output>{{ display(definition.a, state.a) }}</output></span>
          <input v-model.number="state.a" type="range" :min="definition.a.min" :max="definition.a.max" :step="definition.a.step">
        </label>
        <label>
          <span>{{ definition.b.label }} <output>{{ display(definition.b, state.b) }}</output></span>
          <input v-model.number="state.b" type="range" :min="definition.b.min" :max="definition.b.max" :step="definition.b.step">
        </label>
        <p role="status">{{ result.note }}</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.ml-lab{margin:34px 0;padding:26px;border:1px solid #334a86;border-radius:14px;color:#eef2ff;background:#17203b;box-shadow:7px 7px 0 #cbd5f2}.ml-lab>header{display:grid;grid-template-columns:minmax(0,1fr) minmax(180px,.65fr);gap:24px;align-items:start}.ml-lab>header span{color:#76d6cf;font:850 10px/1 ui-monospace,monospace;letter-spacing:.12em}.ml-lab h3{margin:8px 0 0;border:0;color:#fff;font-size:20px}.ml-lab>header p{margin:8px 0 0;color:#aebbdc;font-size:12px}.ml-lab>header code{display:block;padding:12px;border:1px solid #3d548d;border-radius:8px;color:#f4c95d;background:#10182e;font:650 11px/1.6 ui-monospace,monospace;white-space:normal}.ml-lab__body{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(210px,.72fr);gap:24px;margin-top:22px}.ml-lab__chart svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:9px;background:#10182e}.ml-lab__chart .grid line{stroke:#2b3963;stroke-width:1}.ml-lab__chart .axis{stroke:#7381a6;stroke-width:1}.ml-lab__chart .primary,.ml-lab__chart .compare{fill:none;stroke:#f06b5b;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.ml-lab__chart .compare{stroke:#72d5ce;stroke-width:2;stroke-dasharray:5 4}.ml-lab__chart text{fill:#8f9cbc;font:10px ui-monospace,monospace}.ml-lab__metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:8px}.ml-lab__metrics>div{min-height:66px;padding:10px;border:1px solid #344675;border-radius:7px;background:#111a31}.ml-lab__metrics span,.ml-lab__metrics strong{display:block}.ml-lab__metrics span{color:#8f9cbc;font-size:9px;line-height:1.35}.ml-lab__metrics strong{margin-top:6px;color:#f4c95d;font:760 13px/1.3 ui-monospace,monospace}.ml-lab__controls{display:grid;align-content:start;gap:20px}.ml-lab__controls label>span{display:flex;justify-content:space-between;gap:12px;color:#c8d1e8;font-size:11px}.ml-lab__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-lab__controls input{width:100%;height:44px;margin:4px 0 0;accent-color:#e15b4b;cursor:pointer}.ml-lab__controls p{margin:0;padding:14px;border-left:3px solid #72d5ce;color:#b9c5e1;background:#10182e;font-size:11px;line-height:1.75}@media(max-width:760px){.ml-lab{padding:22px 18px}.ml-lab>header,.ml-lab__body{grid-template-columns:1fr}.ml-lab__metrics{grid-template-columns:1fr}.ml-lab__metrics>div{min-height:0}}@media(prefers-reduced-motion:reduce){.ml-lab *{transition-duration:.01ms!important}}
</style>
