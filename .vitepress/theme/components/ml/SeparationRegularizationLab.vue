<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const iterations = ref(90)
const regularization = ref(0.12)
const closestGap = ref(0.35)
const contradiction = ref(false)

const sigmoid = (value: number) => value >= 0
  ? 1 / (1 + Math.exp(-value))
  : Math.exp(value) / (1 + Math.exp(value))
const softplus = (value: number) => value > 0
  ? value + Math.log1p(Math.exp(-value))
  : Math.log1p(Math.exp(value))

const result = computed(() => {
  const x = [-1.8, -1.25, -0.8, -closestGap.value, closestGap.value, 0.8, 1.25, 1.8]
  const label = [-1, -1, -1, -1, 1, contradiction.value ? -1 : 1, 1, 1]
  const objective = (weight: number) => (
    x.reduce((sum, value, index) => sum + softplus(-label[index] * weight * value), 0) / x.length
    + 0.5 * regularization.value * weight * weight
  )
  const gradient = (weight: number) => (
    x.reduce((sum, value, index) => (
      sum - label[index] * value * sigmoid(-label[index] * weight * value)
    ), 0) / x.length
    + regularization.value * weight
  )
  const hessian = (weight: number) => (
    x.reduce((sum, value) => {
      const probability = sigmoid(weight * value)
      return sum + value * value * probability * (1 - probability)
    }, 0) / x.length
    + regularization.value
  )

  let weight = 0
  const path = [weight]
  for (let step = 0; step < iterations.value; step++) {
    weight -= 0.42 * gradient(weight)
    path.push(weight)
  }
  const maximumWeight = Math.max(7, weight * 1.16)
  const curve = Array.from({ length: 101 }, (_, index) => {
    const candidate = maximumWeight * index / 100
    return { weight: candidate, objective: objective(candidate) }
  })
  const maximumObjective = Math.max(...curve.map((point) => point.objective), 0.7)
  const saturated = x.filter((value) => {
    const probability = sigmoid(weight * value)
    return probability < 0.01 || probability > 0.99
  }).length
  return {
    x,
    label,
    weight,
    path,
    curve,
    maximumWeight,
    maximumObjective,
    loss: objective(weight),
    gradient: gradient(weight),
    hessian: hessian(weight),
    closestProbability: sigmoid(weight * closestGap.value),
    saturated,
  }
})

const chart = { left: 50, right: 400, top: 42, bottom: 244 }
const xPlot = (weight: number) => chart.left + weight / result.value.maximumWeight * (chart.right - chart.left)
const yPlot = (objective: number) => chart.bottom - objective / result.value.maximumObjective * (chart.bottom - chart.top)
const objectivePoints = computed(() => result.value.curve
  .map((point) => `${xPlot(point.weight)},${yPlot(point.objective)}`)
  .join(' '))
const sampleX = (value: number) => 50 + (value + 2) / 4 * 350
</script>

<template>
  <MlExperimentFrame
    index="09-B"
    title="观察完全分离时系数逃逸，再接入 L2 制动器"
    question="目标凸为什么仍可能没有有限最优点，正则化又怎样把下确界变成可达到的解？"
    formula="Jλ(w)=mean log(1+e^(−ỹwx)) + λw²/2"
  >
    <template #visual>
      <figure class="separation-rig">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`迭代 ${iterations} 次，系数 ${result.weight.toFixed(2)}，损失 ${result.loss.toFixed(3)}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="rig-bg" />
          <text x="50" y="30" class="panel-title">系数方向上的目标函数 / 当前训练轨迹</text>
          <line v-for="tick in [0.25,.5,.75,1]" :key="`y-${tick}`" x1="50" x2="400" :y1="yPlot(tick * result.maximumObjective)" :y2="yPlot(tick * result.maximumObjective)" class="grid" />
          <line v-for="tick in [0,.25,.5,.75,1]" :key="`x-${tick}`" :x1="xPlot(tick * result.maximumWeight)" :x2="xPlot(tick * result.maximumWeight)" y1="42" y2="244" class="grid" />
          <polyline :points="objectivePoints" class="objective-line" />
          <line :x1="xPlot(result.weight)" :x2="xPlot(result.weight)" :y1="yPlot(result.loss)" y2="244" class="weight-line" />
          <circle :cx="xPlot(result.weight)" :cy="yPlot(result.loss)" r="7" class="weight-point" />
          <text x="50" y="263" class="axis-label">0</text>
          <text x="400" y="263" class="axis-label end">{{ result.maximumWeight.toFixed(1) }} → w</text>

          <line x1="50" x2="400" y1="287" y2="287" class="sample-axis" />
          <g v-for="(value,index) in result.x" :key="`${value}-${index}`">
            <circle :cx="sampleX(value)" cy="287" r="7" :class="['training-point', result.label[index] > 0 ? 'positive' : 'negative']" />
            <text :x="sampleX(value)" y="309" class="sample-label">{{ result.label[index] > 0 ? '+' : '−' }}</text>
          </g>

          <text x="434" y="31" class="panel-title">逃逸监控</text>
          <text x="434" y="66" class="ledger-label">迭代步</text>
          <text x="563" y="66" class="ledger-value">{{ iterations }}</text>
          <text x="434" y="99" class="ledger-label">系数 w</text>
          <text x="563" y="99" class="ledger-value weight">{{ result.weight.toFixed(3) }}</text>
          <text x="434" y="132" class="ledger-label">目标 Jλ</text>
          <text x="563" y="132" class="ledger-value">{{ result.loss.toFixed(4) }}</text>
          <text x="434" y="165" class="ledger-label">|梯度|</text>
          <text x="563" y="165" class="ledger-value">{{ Math.abs(result.gradient).toExponential(1) }}</text>
          <text x="434" y="198" class="ledger-label">局部曲率</text>
          <text x="563" y="198" class="ledger-value">{{ result.hessian.toFixed(4) }}</text>
          <line x1="434" x2="563" y1="219" y2="219" class="ledger-rule" />
          <text x="434" y="248" class="ledger-label">饱和样本</text>
          <text x="563" y="248" class="ledger-value warning">{{ result.saturated }} / 8</text>
          <text x="434" y="282" :class="['status-label', regularization > 0 || contradiction ? 'braked' : 'runaway']">{{ contradiction ? '反例形成有限驻点' : regularization > 0 ? 'L2 制动已接入' : '无制动：继续逃逸' }}</text>
        </svg>
        <figcaption>
          <span class="legend objective" />正则化目标
          <span class="legend current" />当前系数
          <span class="legend positive" />正类
          <span class="legend negative" />负类/反例
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>梯度更新次数 <output>{{ iterations }}</output></span>
        <input v-model.number="iterations" type="range" min="0" max="240" step="5">
      </label>
      <label>
        <span>L2 强度 λ <output>{{ regularization.toFixed(2) }}</output></span>
        <input v-model.number="regularization" type="range" min="0" max="1" step="0.02">
      </label>
      <label>
        <span>最近样本间隙 <output>{{ closestGap.toFixed(2) }}</output></span>
        <input v-model.number="closestGap" type="range" min="0.15" max="0.9" step="0.05">
      </label>
      <button type="button" :class="{ active: contradiction }" @click="contradiction = !contradiction">
        {{ contradiction ? '已加入一枚错标反例' : '加入一枚错标反例' }}
      </button>
      <p>完全可分且 λ=0 时，增大迭代次数会让损失继续下降、系数继续增大；加入 L2 或一枚无法同时分对的反例后，有限最优点重新出现。</p>
    </template>
    <template #metrics>
      <div><span>当前系数范数</span><strong>{{ Math.abs(result.weight).toFixed(3) }}</strong></div>
      <div><span>最近正类概率</span><strong>{{ (result.closestProbability * 100).toFixed(1) }}%</strong></div>
      <div><span>Hessian 标量</span><strong>{{ result.hessian.toFixed(4) }}</strong></div>
      <div><span>解的状态</span><strong>{{ regularization > 0 || contradiction ? '有限制动点' : '向无穷逃逸' }}</strong></div>
    </template>
    <template #note>
      “凸”只排除坏的局部极小值，不自动保证极小值存在。工程求解还要检查秩、分离、特征尺度、正则化定义和停止标准，不能只看优化器返回 `success`。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.separation-rig{margin:0}.separation-rig svg{display:block;width:100%;border:1px solid #294764;border-radius:10px;background:#0c1722}.rig-bg{fill:#0c1722}.grid{stroke:#2e485f;stroke-width:1}.objective-line{fill:none;stroke:#63d0e2;stroke-width:4}.weight-line{stroke:#f5c85c;stroke-width:2;stroke-dasharray:5 4}.weight-point{fill:#f5c85c;stroke:#fff;stroke-width:2}.sample-axis{stroke:#49657b;stroke-width:2}.training-point{stroke:#fff;stroke-width:1.5}.training-point.positive{fill:#63d0e2}.training-point.negative{fill:#f58a4b}.separation-rig text{font:700 10px ui-monospace,monospace}.panel-title{fill:#dcecf3}.axis-label,.ledger-label{fill:#8da8bc}.axis-label.end{text-anchor:end}.sample-label{fill:#dcecf3;text-anchor:middle;font-size:9px}.ledger-value{fill:#dcecf3;text-anchor:end;font-size:13px}.ledger-value.weight{fill:#f5c85c}.ledger-value.warning{fill:#f58a4b}.ledger-rule{stroke:#35526c}.status-label{font-size:12px}.status-label.braked{fill:#63d0e2}.status-label.runaway{fill:#f58a4b}.separation-rig figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#91aabd;font-size:10px;line-height:1.6}.legend{display:inline-block;width:14px;height:3px;margin-right:4px;vertical-align:middle}.legend.objective{background:#63d0e2}.legend.current{border-top:2px dashed #f5c85c}.legend.positive,.legend.negative{height:10px;border-radius:50%}.legend.positive{background:#63d0e2}.legend.negative{background:#f58a4b}.ml-experiment__controls label{display:grid;gap:5px;color:#c6d7e4;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0e2;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f58a4b;cursor:pointer}.ml-experiment__controls button{min-height:44px;border:1px solid #47647a;border-radius:6px;background:#142536;color:#c6d7e4;font:700 11px ui-monospace,monospace;cursor:pointer}.ml-experiment__controls button.active{border-color:#f58a4b;background:#3a261e;color:#ffb584}.ml-experiment__controls button:focus-visible{outline:3px solid #f5c85c;outline-offset:2px}.ml-experiment__controls p{margin:0;color:#91aabd;font-size:10px;line-height:1.65}@media(max-width:560px){.separation-rig{overflow-x:auto;overscroll-behavior-inline:contain}.separation-rig svg{width:560px;max-width:none}.separation-rig figcaption{min-width:560px}}
</style>
