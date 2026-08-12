<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const trueRate = ref(.06)
const sampleSize = ref(24)
const strength = ref(120)
const mismatched = ref(false)
const priorMode = computed(() => mismatched.value ? .12 : .035)

const probabilities = computed(() => {
  const p = trueRate.value
  const n = sampleSize.value
  const values = [Math.pow(1 - p, n)]
  for (let k = 1; k <= n; k += 1) {
    values.push(values[k - 1] * (n - k + 1) / k * p / (1 - p))
  }
  const total = values.reduce((sum, value) => sum + value, 0)
  return values.map(value => value / total)
})
const samplingPoints = computed(() => probabilities.value.map((probability, misses) => ({
  misses,
  probability,
  mle: misses / sampleSize.value,
  map: (misses + strength.value * priorMode.value) / (sampleSize.value + strength.value),
})).filter(point => point.probability > 1e-7))

const mleRisk = computed(() => {
  const variance = trueRate.value * (1 - trueRate.value) / sampleSize.value
  return { mean:trueRate.value, bias:0, variance, mse:variance }
})
const mapRisk = computed(() => {
  const denominator = sampleSize.value + strength.value
  const bias = strength.value * (priorMode.value - trueRate.value) / denominator
  const variance = sampleSize.value * trueRate.value * (1 - trueRate.value) / denominator ** 2
  return { mean:trueRate.value + bias, bias, variance, mse:bias ** 2 + variance }
})
const riskRatio = computed(() => mapRisk.value.mse / mleRisk.value.mse)
const x = (theta: number) => 54 + Math.min(.18, Math.max(0, theta)) / .18 * 530
const radius = (probability: number) => 2.5 + Math.sqrt(probability) * 31
</script>

<template>
  <MlExperimentFrame
    index="22-C"
    title="在重复资格审核中拆开收缩的偏差与方差"
    question="强先验会把每次审核结果压得更集中，但集中是否必然意味着估计更好？把先验中心移错后，风险何时反超 MLE？"
    formula="MSE = Bias² + Variance"
  >
    <template #visual>
      <div class="risk-scroll">
        <svg viewBox="0 0 640 340" role="img" :aria-label="`真实漏检率${(100*trueRate).toFixed(1)}%，MLE均方误差${mleRisk.mse.toFixed(5)}，MAP均方误差${mapRisk.mse.toFixed(5)}，MAP风险比${riskRatio.toFixed(2)}`">
          <rect width="640" height="340" rx="14" class="panel" />
          <text x="24" y="27" class="eyebrow">REPEATED AUDIT RISK ARRAY · EXACT BINOMIAL MASS</text>
          <line v-for="tick in [0,.03,.06,.09,.12,.15,.18]" :key="tick" :x1="x(tick)" :x2="x(tick)" y1="52" y2="276" class="grid" />
          <line :x1="x(trueRate)" :x2="x(trueRate)" y1="45" y2="286" class="truth" />
          <line x1="54" x2="584" y1="118" y2="118" class="rail" />
          <line x1="54" x2="584" y1="232" y2="232" class="rail" />
          <text x="54" y="83" class="row-label mle-label">MLE：每个可能的 S/n</text>
          <text x="54" y="197" class="row-label map-label">MAP：同一 S 经先验收缩后</text>
          <g v-for="point in samplingPoints" :key="point.misses">
            <line :x1="x(point.mle)" :x2="x(point.map)" y1="118" y2="232" class="pair-link" :style="{ opacity:Math.min(.65,.08+point.probability*2.2) }" />
            <circle :cx="x(point.mle)" cy="118" :r="radius(point.probability)" class="mass mle-mass" />
            <circle :cx="x(point.map)" cy="232" :r="radius(point.probability)" class="mass map-mass" />
          </g>
          <circle :cx="x(mleRisk.mean)" cy="118" r="5" class="mean-dot mle-mean" />
          <circle :cx="x(mapRisk.mean)" cy="232" r="5" class="mean-dot map-mean" />
          <line :x1="x(trueRate)" :x2="x(mapRisk.mean)" y1="278" y2="278" class="bias-arrow" />
          <path :d="`M${x(mapRisk.mean)} 278 l${mapRisk.bias>0?-7:7} -5 v10 z`" class="bias-head" />
          <text :x="(x(trueRate)+x(mapRisk.mean))/2" y="270" text-anchor="middle" class="bias-label">MAP bias</text>
          <text v-for="tick in [0,.03,.06,.09,.12,.15,.18]" :key="`label-${tick}`" :x="x(tick)" y="306" text-anchor="middle" class="tick">{{ Math.round(tick*100) }}%</text>
          <rect x="452" y="45" width="132" height="35" rx="5" class="verdict" :class="{ worse:riskRatio>1 }" />
          <text x="518" y="59" text-anchor="middle" class="verdict-title">MAP / MLE 风险</text>
          <text x="518" y="73" text-anchor="middle" class="verdict-value">{{ riskRatio.toFixed(2) }} · {{ riskRatio<=1 ? '较低' : '反超' }}</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>真实漏检率 θ* <output>{{ (100*trueRate).toFixed(1) }}%</output></span><input v-model.number="trueRate" aria-label="真实漏检率" type="range" min="0.01" max="0.15" step="0.005"></label>
      <label class="control"><span>每次审核量 n <output>{{ sampleSize }}</output></span><input v-model.number="sampleSize" aria-label="每次审核量" type="range" min="5" max="100" step="1"></label>
      <label class="control"><span>MAP 等效票数 c <output>{{ strength }}</output></span><input v-model.number="strength" aria-label="MAP等效票数" type="range" min="0" max="160" step="4"></label>
      <button type="button" :class="{ active:mismatched }" :aria-pressed="mismatched" @click="mismatched=!mismatched">{{ mismatched ? '错配先验中心：12%' : '冻结先验中心：3.5%' }}</button>
    </template>
    <template #metrics>
      <div><span>MLE 方差 / MSE</span><strong>{{ mleRisk.variance.toFixed(5) }}</strong></div>
      <div><span>MAP Bias²</span><strong>{{ (mapRisk.bias**2).toFixed(5) }}</strong></div>
      <div><span>MAP 方差</span><strong>{{ mapRisk.variance.toFixed(5) }}</strong></div>
      <div><span>MAP MSE</span><strong>{{ mapRisk.mse.toFixed(5) }}</strong></div>
    </template>
    <template #note>圆面积是精确 Binomial 抽样概率，不是随机动画。收缩让离散结果向先验中心聚拢并降低方差，但错误中心会产生系统偏差；先验越强，方差继续下降，Bias² 却可能主导总风险。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.risk-scroll{overflow-x:auto}.risk-scroll svg{display:block;width:100%;height:auto}.panel{fill:#111923}.grid{stroke:#293b45;stroke-width:1}.truth{stroke:#e8edf0;stroke-width:2;stroke-dasharray:6 4}.rail{stroke:#708690;stroke-width:2}.pair-link{stroke:#667c87}.mass{stroke-width:1.5}.mle-mass{fill:#d19b3f;fill-opacity:.58;stroke:#f0bd59}.map-mass{fill:#765ca7;fill-opacity:.65;stroke:#b49aec}.mean-dot{stroke:#fff;stroke-width:2}.mle-mean{fill:#f0bd59}.map-mean{fill:#a98de5}.bias-arrow{stroke:#e77455;stroke-width:3}.bias-head{fill:#e77455}.eyebrow,.row-label,.bias-label,.tick,.verdict-title,.verdict-value{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#6bc4d8;font-size:10px;letter-spacing:.12em}.row-label{font-size:9px;font-weight:800}.mle-label{fill:#f0bd59}.map-label{fill:#bca8eb}.bias-label{fill:#e98b70;font-size:8px}.tick{fill:#9eb0b8;font-size:8px}.verdict{fill:#163a36;stroke:#63c7bb}.verdict.worse{fill:#442724;stroke:#e77455}.verdict-title{fill:#aebdc5;font-size:7px}.verdict-value{fill:#f2d382;font-size:10px;font-weight:800}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebdc5;font-size:11px}.control output{color:#f0bd59;font:750 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#d8794f}button{min-height:44px;padding:9px 12px;border:1px solid #526772;border-radius:6px;color:#dce6e9;background:#172733;font-weight:750;cursor:pointer}button.active{border-color:#e77455;color:#ffe7df;background:#472923}button:focus-visible,input:focus-visible{outline:3px solid #6bc4d8;outline-offset:2px}@media(max-width:560px){.risk-scroll{margin-right:-18px;padding-right:18px}.risk-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
