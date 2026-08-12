<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const audited = ref(36)
const misses = ref(1)
const priorMode = ref(.035)
const modeStrength = ref(36)
const reused = ref(false)
const maxMisses = computed(() => Math.max(1, Math.floor(audited.value * .2)))
watch(audited, () => { misses.value = Math.min(misses.value, maxMisses.value) })

const mle = computed(() => misses.value / audited.value)
const baseAlpha = computed(() => 1 + modeStrength.value * priorMode.value)
const baseBeta = computed(() => 1 + modeStrength.value * (1 - priorMode.value))
const alpha = computed(() => baseAlpha.value + (reused.value ? misses.value : 0))
const beta = computed(() => baseBeta.value + (reused.value ? audited.value - misses.value : 0))
const activeStrength = computed(() => alpha.value + beta.value - 2)
const activeMode = computed(() => (alpha.value - 1) / activeStrength.value)
const posteriorAlpha = computed(() => alpha.value + misses.value)
const posteriorBeta = computed(() => beta.value + audited.value - misses.value)
const mapEstimate = computed(() => (posteriorAlpha.value - 1) / (posteriorAlpha.value + posteriorBeta.value - 2))
const posteriorMean = computed(() => posteriorAlpha.value / (posteriorAlpha.value + posteriorBeta.value))

const logKernel = (theta: number, a: number, b: number) => (a - 1) * Math.log(theta) + (b - 1) * Math.log1p(-theta)
const plot = { left: 54, right: 592, top: 45, bottom: 274 }
const x = (theta: number) => plot.left + Math.min(.25, Math.max(0, theta)) / .25 * (plot.right - plot.left)
const pathFor = (a: number, b: number, offset = 0) => {
  const values = Array.from({ length: 251 }, (_, index) => {
    const theta = .0005 + index / 250 * .249
    return { theta, log: logKernel(theta, a, b) }
  })
  const maximum = Math.max(...values.map(value => value.log))
  return values.map((value, index) => {
    const height = Math.exp(Math.max(-35, value.log - maximum)) * (plot.bottom - plot.top - 16)
    return `${index === 0 ? 'M' : 'L'}${x(value.theta).toFixed(2)} ${(plot.bottom-height-offset).toFixed(2)}`
  }).join(' ')
}
const priorPath = computed(() => pathFor(alpha.value, beta.value))
const likelihoodPath = computed(() => pathFor(1 + misses.value, 1 + audited.value - misses.value, 3))
const posteriorPath = computed(() => pathFor(posteriorAlpha.value, posteriorBeta.value, 6))

const posteriorGrid = computed(() => {
  const values = Array.from({ length: 2000 }, (_, index) => {
    const theta = .00025 + index / 1999 * .9995
    return { theta, log: logKernel(theta, posteriorAlpha.value, posteriorBeta.value) }
  })
  const maximum = Math.max(...values.map(value => value.log))
  const raw = values.map(value => Math.exp(Math.max(-745, value.log - maximum)))
  const total = raw.reduce((sum, value) => sum + value, 0)
  let cumulative = 0
  return values.map((value, index) => {
    cumulative += raw[index] / total
    return { theta:value.theta, mass:raw[index] / total, cumulative }
  })
})
const tailProbability = computed(() => posteriorGrid.value.filter(value => value.theta > .05).reduce((sum, value) => sum + value.mass, 0))
const lower = computed(() => posteriorGrid.value.find(value => value.cumulative >= .025)?.theta ?? 0)
const upper = computed(() => posteriorGrid.value.find(value => value.cumulative >= .975)?.theta ?? 1)
</script>

<template>
  <MlExperimentFrame
    index="22-B"
    title="把先验与本批证据叠成后验曝光片"
    question="MLE、MAP、后验均值和风险阈值右侧面积为何是四个不同答案？若把同一批审核结果先拿来定先验再更新一次，会发生什么？"
    formula="p(θ|D) ∝ θ^(S+α−1)(1−θ)^(n−S+β−1)"
  >
    <template #visual>
      <div class="mixer-scroll">
        <svg viewBox="0 0 640 340" role="img" :aria-label="`先验众数${(100*activeMode).toFixed(1)}%，MLE${(100*mle).toFixed(1)}%，MAP${(100*mapEstimate).toFixed(1)}%，后验超过百分之五的概率${(100*tailProbability).toFixed(1)}%`">
          <rect width="640" height="340" rx="14" class="panel" />
          <text x="24" y="27" class="eyebrow">EVIDENCE EXPOSURE TABLE · BETA × BERNOULLI</text>
          <rect :x="x(.05)" :width="plot.right-x(.05)" :y="plot.top" :height="plot.bottom-plot.top" class="risk-zone" />
          <line v-for="tick in [0,.05,.10,.15,.20,.25]" :key="tick" :x1="x(tick)" :x2="x(tick)" :y1="plot.top" :y2="plot.bottom" class="grid" />
          <line :x1="plot.left" :x2="plot.right" :y1="plot.bottom" :y2="plot.bottom" class="axis" />
          <path :d="priorPath" class="curve prior" />
          <path :d="likelihoodPath" class="curve likelihood" />
          <path :d="posteriorPath" class="curve posterior" />
          <line :x1="x(mle)" :x2="x(mle)" y1="55" :y2="plot.bottom" class="marker mle" />
          <line :x1="x(mapEstimate)" :x2="x(mapEstimate)" y1="55" :y2="plot.bottom" class="marker map" />
          <line :x1="x(posteriorMean)" :x2="x(posteriorMean)" y1="55" :y2="plot.bottom" class="marker mean" />
          <text :x="x(.05)+7" y="60" class="risk-label">资格上限右侧面积</text>
          <text v-for="tick in [0,.05,.10,.15,.20,.25]" :key="`label-${tick}`" :x="x(tick)" y="295" text-anchor="middle" class="tick">{{ Math.round(tick*100) }}%</text>
          <g transform="translate(54 317)">
            <line x1="0" x2="24" y1="0" y2="0" class="curve prior"/><text x="31" y="3" class="legend prior-text">先验</text>
            <line x1="104" x2="128" y1="0" y2="0" class="curve likelihood"/><text x="135" y="3" class="legend likelihood-text">似然</text>
            <line x1="213" x2="237" y1="0" y2="0" class="curve posterior"/><text x="244" y="3" class="legend posterior-text">后验</text>
            <text x="340" y="3" class="legend reused-text">{{ reused ? '警告：本批证据被计算两次' : '先验来自冻结历史批次' }}</text>
          </g>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>本批审核 n <output>{{ audited }}</output></span><input v-model.number="audited" aria-label="本批审核量" type="range" min="4" max="120" step="4"></label>
      <label class="control"><span>本批漏检 S <output>{{ misses }}</output></span><input v-model.number="misses" aria-label="本批漏检数" type="range" min="0" :max="maxMisses" step="1"></label>
      <label class="control"><span>冻结先验众数 <output>{{ (100*priorMode).toFixed(1) }}%</output></span><input v-model.number="priorMode" aria-label="先验众数" type="range" min="0.005" max="0.12" step="0.005" :disabled="reused"></label>
      <label class="control"><span>MAP 等效票数 <output>{{ modeStrength }}</output></span><input v-model.number="modeStrength" aria-label="MAP等效票数" type="range" min="2" max="120" step="2" :disabled="reused"></label>
      <button type="button" :class="{ active:reused }" :aria-pressed="reused" @click="reused=!reused">{{ reused ? '已违规：本批同时充当先验' : '把本批数据伪装成先验' }}</button>
    </template>
    <template #metrics>
      <div><span>MLE / MAP</span><strong>{{ (100*mle).toFixed(2) }}% / {{ (100*mapEstimate).toFixed(2) }}%</strong></div>
      <div><span>后验均值</span><strong>{{ (100*posteriorMean).toFixed(2) }}%</strong></div>
      <div><span>95% 可信区间</span><strong>{{ (100*lower).toFixed(1) }}–{{ (100*upper).toFixed(1) }}%</strong></div>
      <div><span>P(θ &gt; 5% | D)</span><strong>{{ (100*tailProbability).toFixed(1) }}%</strong></div>
    </template>
    <template #note>MAP 是后验密度最高点，后验均值是平方损失下的 Bayes 点估计，阈值右侧面积才直接回答“真实漏检率超过资格线的概率”。用目标批次调先验后又更新，会把中心重复拉向本批比例，并把后验人为压窄。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.mixer-scroll{overflow-x:auto}.mixer-scroll svg{display:block;width:100%;height:auto}.panel{fill:#111923}.risk-zone{fill:#3b2524;opacity:.78}.grid{stroke:#2a3c46;stroke-width:1}.axis{stroke:#81949c;stroke-width:2}.curve{fill:none;stroke-linejoin:round}.curve.prior{stroke:#a98de5;stroke-width:3;stroke-dasharray:7 5}.curve.likelihood{stroke:#f0bd59;stroke-width:3}.curve.posterior{stroke:#6bc4d8;stroke-width:5}.marker{stroke-width:1.5;stroke-dasharray:4 4}.marker.mle{stroke:#f0bd59}.marker.map{stroke:#e77455}.marker.mean{stroke:#6bc4d8}.eyebrow,.risk-label,.tick,.legend{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#6bc4d8;font-size:10px;letter-spacing:.12em}.risk-label{fill:#f09a7f;font-size:8px}.tick,.legend{fill:#9eb0b8;font-size:8px}.prior-text{fill:#bca9ed}.likelihood-text{fill:#efd078}.posterior-text{fill:#7ed1dc}.reused-text{fill:#e58a6d}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebdc5;font-size:11px}.control output{color:#f0bd59;font:750 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#d8794f}input:disabled{opacity:.45}button{min-height:44px;padding:9px 12px;border:1px solid #526772;border-radius:6px;color:#dce6e9;background:#172733;font-weight:750;cursor:pointer}button.active{border-color:#e77455;color:#ffe8df;background:#472a25}button:focus-visible,input:focus-visible{outline:3px solid #6bc4d8;outline-offset:2px}@media(max-width:560px){.mixer-scroll{margin-right:-18px;padding-right:18px}.mixer-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
