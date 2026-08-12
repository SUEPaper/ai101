<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const sampleSize = ref(60)
const heteroskedasticity = ref(0.6)
const seed = ref(17)
const repeats = 300
const criticalValue = 1.96

function makeRandom(initialSeed: number) {
  let state = initialSeed >>> 0
  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 4294967296
  }
}

function normal(random: () => number) {
  const first = Math.max(random(), 1e-12)
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * random())
}

interface IntervalResult {
  classicCover: boolean
  robustCover: boolean
  slope: number
  classicSe: number
  robustSe: number
}

const simulation = computed(() => {
  const n = sampleSize.value
  const x = Array.from({ length: n }, (_, index) => index / (n - 1))
  const meanX = x.reduce((sum, value) => sum + value, 0) / n
  const centered = x.map((value) => value - meanX)
  const sxx = centered.reduce((sum, value) => sum + value ** 2, 0)
  const random = makeRandom(seed.value * 104729 + n * 101)
  const results: IntervalResult[] = []

  for (let repeat = 0; repeat < repeats; repeat += 1) {
    const errors = x.map((value) => {
      const distanceFromCenter = Math.abs(value - 0.5) / 0.5
      const sigma = 0.04 * (1 + heteroskedasticity.value * 12 * distanceFromCenter ** 3)
      return sigma * normal(random)
    })
    const meanError = errors.reduce((sum, value) => sum + value, 0) / n
    const slopeError = centered.reduce((sum, value, index) => sum + value * errors[index], 0) / sxx
    const interceptError = meanError - meanX * slopeError
    const residuals = errors.map((value, index) => value - interceptError - slopeError * x[index])
    const rss = residuals.reduce((sum, value) => sum + value ** 2, 0)
    const classicSe = Math.sqrt(rss / (n - 2) / sxx)
    const robustMeat = residuals.reduce((sum, value, index) => sum + centered[index] ** 2 * value ** 2, 0)
    const robustSe = Math.sqrt(n / (n - 2) * robustMeat / sxx ** 2)
    const slope = 2 + slopeError
    results.push({
      classicCover: Math.abs(slope - 2) <= criticalValue * classicSe,
      robustCover: Math.abs(slope - 2) <= criticalValue * robustSe,
      slope,
      classicSe,
      robustSe
    })
  }

  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length
  const meanSlope = mean(results.map((item) => item.slope))
  const empiricalSd = Math.sqrt(mean(results.map((item) => (item.slope - meanSlope) ** 2)))
  return {
    results,
    classicCoverage: mean(results.map((item) => Number(item.classicCover))),
    robustCoverage: mean(results.map((item) => Number(item.robustCover))),
    meanClassicSe: mean(results.map((item) => item.classicSe)),
    meanRobustSe: mean(results.map((item) => item.robustSe)),
    empiricalSd
  }
})

const gridX = (index: number, panelOffset: number) => panelOffset + (index % 10) * 16
const gridY = (index: number) => 78 + Math.floor(index / 10) * 16
const percent = (value: number) => `${(100 * value).toFixed(1)}%`
</script>

<template>
  <MlExperimentFrame
    index="05-B"
    title="让三百份校准证书接受覆盖率审计"
    question="噪声在量程两端变大时，仍套用同方差标准误会漏掉多少真实灵敏度？"
    formula="HC1 = n/(n−p) · (XᵀX)⁻¹ Xᵀdiag(rᵢ²)X (XᵀX)⁻¹"
  >
    <template #visual>
      <figure class="coverage-ledger">
        <svg viewBox="0 0 600 286" role="img" :aria-label="`常规区间覆盖率 ${percent(simulation.classicCoverage)}，HC1 区间覆盖率 ${percent(simulation.robustCoverage)}`">
          <rect x="16" y="16" width="568" height="254" rx="9" class="plot-bg" />
          <text x="56" y="43" class="panel-title">同方差公式</text>
          <text x="354" y="43" class="panel-title">HC1 稳健公式</text>
          <text x="56" y="61" class="panel-note">前 100 次重复校准</text>
          <text x="354" y="61" class="panel-note">前 100 次重复校准</text>
          <line x1="300" x2="300" y1="34" y2="250" class="divider" />

          <rect
            v-for="(item,index) in simulation.results.slice(0,100)"
            :key="`classic-${index}`"
            :x="gridX(index,56)"
            :y="gridY(index)"
            width="11"
            height="11"
            rx="2"
            :class="item.classicCover ? 'covered' : 'missed'"
          />
          <rect
            v-for="(item,index) in simulation.results.slice(0,100)"
            :key="`robust-${index}`"
            :x="gridX(index,354)"
            :y="gridY(index)"
            width="11"
            height="11"
            rx="2"
            :class="item.robustCover ? 'covered' : 'missed'"
          />

          <text x="56" y="258" class="coverage-value">{{ percent(simulation.classicCoverage) }}</text>
          <text x="354" y="258" class="coverage-value">{{ percent(simulation.robustCoverage) }}</text>
          <text x="128" y="257" class="panel-note">300 次总覆盖</text>
          <text x="426" y="257" class="panel-note">300 次总覆盖</text>
        </svg>
        <figcaption><span class="legend covered" />区间覆盖真实灵敏度　<span class="legend missed" />区间漏掉真实灵敏度　每格是一份独立合成校准研究</figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>每次校准样本数 <output>{{ sampleSize }}</output></span>
        <input v-model.number="sampleSize" type="range" min="30" max="150" step="10">
      </label>
      <label>
        <span>量程两端噪声增强 <output>{{ heteroskedasticity.toFixed(2) }}</output></span>
        <input v-model.number="heteroskedasticity" type="range" min="0" max="1" step="0.05">
      </label>
      <button type="button" @click="seed += 1">重新生成三百次校准</button>
      <p>绿色不是“这次估计正确”，而是该次 95% 区间包含固定真实灵敏度 2。改变噪声结构后，比较长期覆盖率，而不是挑一条幸运区间。</p>
    </template>
    <template #metrics>
      <div><span>常规 95% 覆盖率</span><strong>{{ percent(simulation.classicCoverage) }}</strong></div>
      <div><span>HC1 95% 覆盖率</span><strong>{{ percent(simulation.robustCoverage) }}</strong></div>
      <div><span>灵敏度经验标准差</span><strong>{{ simulation.empiricalSd.toFixed(4) }}</strong></div>
      <div><span>平均 SE：常规 / HC1</span><strong>{{ simulation.meanClassicSe.toFixed(4) }} / {{ simulation.meanRobustSe.toFixed(4) }}</strong></div>
    </template>
    <template #note>
      这里使用 1.96 构造大样本区间。量程两端既是高杠杆区又有更大噪声时，同方差公式会系统低估灵敏度波动；HC1 用逐点残差平方估计“夹心”中间层，覆盖率更接近目标。它不能修复均值模型错误、相关重复测量、小样本或模型选择后的推断。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.coverage-ledger{margin:0}.coverage-ledger svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:10px;background:#10182e}.plot-bg{fill:#10182e}.divider{stroke:#344675}.covered{fill:#72d5ce}.missed{fill:#f06b5b}.coverage-ledger text{font:700 10px ui-monospace,monospace}.panel-title{fill:#eef2ff;font-size:12px!important}.panel-note{fill:#8f9cbc}.coverage-value{fill:#f4c95d;font-size:14px!important}.coverage-ledger figcaption{margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.7}.legend{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px;vertical-align:middle}.legend.covered{background:#72d5ce}.legend.missed{margin-left:12px;background:#f06b5b}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls button{min-height:44px;border:1px solid #6e82bd;border-radius:7px;color:#fff;background:#263d88;font-weight:750;cursor:pointer}.ml-experiment__controls button:hover{background:#3150a3}.ml-experiment__controls button:focus-visible{outline:3px solid #f4c95d;outline-offset:3px}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}
</style>
