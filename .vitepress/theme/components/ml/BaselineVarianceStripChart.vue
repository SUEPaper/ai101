<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const probability = ref(0.38)
const upstreamNoise = ref(4.0)
const criticOffset = ref(0.0)
const seed = ref(32)
const sampleCount = 96
const northMean = 5.4
const eastMean = 1.2

function seededNormal(state: { value: number }) {
  const next = () => {
    state.value = (1664525 * state.value + 1013904223) >>> 0
    return (state.value + 0.5) / 4294967296
  }
  return Math.sqrt(-2 * Math.log(next())) * Math.cos(2 * Math.PI * next())
}

const rows = computed(() => {
  const state = { value: seed.value >>> 0 }
  const p = probability.value
  const baseline = p * northMean + (1 - p) * eastMean + criticOffset.value
  const data = Array.from({ length: sampleCount }, () => {
    const draw = (() => {
      state.value = (1664525 * state.value + 1013904223) >>> 0
      return (state.value + 0.5) / 4294967296
    })()
    const north = draw < p
    const score = north ? 1 - p : -p
    const future = (north ? northMean : eastMean) + 1.1 * seededNormal(state)
    const past = upstreamNoise.value * seededNormal(state)
    return {
      full: score * (past + future),
      rewardToGo: score * future,
      advantage: score * (future - baseline),
    }
  })
  const configs = [
    { key: 'full' as const, label: '整轨迹回报', color: '#c64d3f' },
    { key: 'rewardToGo' as const, label: 'reward-to-go', color: '#f0a83b' },
    { key: 'advantage' as const, label: '优势信号', color: '#3f8f83' },
  ]
  return configs.map(config => {
    const values = data.map(item => item[config.key])
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
    return { ...config, values, mean, variance }
  })
})

const maxAbs = computed(() => Math.max(1, ...rows.value.flatMap(row => row.values.map(Math.abs))))
const exactGradient = computed(() => probability.value * (1 - probability.value) * (northMean - eastMean))
const varianceRatio = computed(() => rows.value[0].variance / Math.max(rows.value[2].variance, 1e-9))
const meanDrift = computed(() => Math.abs(rows.value[2].mean - exactGradient.value))

function x(value: number) {
  return 360 + 306 * value / maxAbs.value
}
</script>

<template>
  <MlExperimentFrame
    index="32-B"
    title="方差条带记录仪"
    question="固定同一策略和同一批伪随机样本，逐层删除过去噪声、再减状态 baseline，比较梯度中心与散布。"
    formula="gₜ = scoreₜ × [Gₜ − b(Sₜ)]"
  >
    <template #visual>
      <div class="recorder">
        <div class="paper-head">
          <span>GRADIENT STRIP / LOT {{ seed.toString().padStart(3, '0') }}</span>
          <b>目标中心 {{ exactGradient.toFixed(3) }}</b>
        </div>
        <svg viewBox="0 0 720 300" role="img" aria-label="三种策略梯度估计器的样本条带和方差">
          <rect x="10" y="10" width="700" height="280" rx="5" class="paper" />
          <g v-for="(row, rowIndex) in rows" :key="row.key" :transform="`translate(0 ${34 + rowIndex * 84})`">
            <text x="28" y="20" class="row-title">{{ row.label }}</text>
            <text x="28" y="39" class="row-stat">var {{ row.variance.toFixed(2) }}</text>
            <line x1="175" y1="26" x2="684" y2="26" class="axis" />
            <line :x1="x(exactGradient)" y1="3" :x2="x(exactGradient)" y2="51" class="target-line" />
            <circle
              v-for="(value, index) in row.values"
              :key="index"
              :cx="x(value)"
              :cy="20 + (index % 7) * 2"
              r="2.3"
              :fill="row.color"
              opacity=".58"
            />
            <polygon :points="`${x(row.mean)},2 ${x(row.mean)-6},-9 ${x(row.mean)+6},-9`" :fill="row.color" />
          </g>
          <text x="175" y="284" class="scale-label">−{{ maxAbs.toFixed(1) }}</text>
          <text x="360" y="284" text-anchor="middle" class="scale-label">0</text>
          <text x="684" y="284" text-anchor="end" class="scale-label">+{{ maxAbs.toFixed(1) }}</text>
        </svg>
      </div>
    </template>

    <template #controls>
      <label>
        <span>北干线概率 p</span>
        <input v-model.number="probability" type="range" min="0.08" max="0.92" step="0.01" />
        <output>{{ probability.toFixed(2) }}</output>
      </label>
      <label>
        <span>过去噪声 σpast</span>
        <input v-model.number="upstreamNoise" type="range" min="0" max="8" step="0.2" />
        <output>{{ upstreamNoise.toFixed(1) }}</output>
      </label>
      <label>
        <span>Critic 偏移</span>
        <input v-model.number="criticOffset" type="range" min="-6" max="6" step="0.2" />
        <output>{{ criticOffset.toFixed(1) }}</output>
      </label>
      <button @click="seed += 1">换一批 96 条轨迹</button>
    </template>

    <template #metrics>
      <div><span>整轨迹方差</span><strong>{{ rows[0].variance.toFixed(2) }}</strong></div>
      <div><span>reward-to-go 方差</span><strong>{{ rows[1].variance.toFixed(2) }}</strong></div>
      <div><span>优势方差</span><strong>{{ rows[2].variance.toFixed(2) }}</strong></div>
      <div><span>全量 / 优势</span><strong>{{ varianceRatio.toFixed(1) }}×</strong></div>
    </template>

    <template #note>
      竖虚线是解析梯度，三条记录的长期中心应相同；有限批次会有 {{ meanDrift.toFixed(3) }} 的抽样漂移。把 Critic 偏移推得很远可看到：动作无关 baseline 仍不改期望，但可能反而增大方差。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.recorder{min-width:560px;color:#17324d}.paper-head{display:flex;justify-content:space-between;padding:9px 12px;border:1px solid #9d9075;border-bottom:0;border-radius:7px 7px 0 0;background:#17324d;color:#ebe4cf;font:800 10px/1.2 ui-monospace,monospace;letter-spacing:.07em}.paper-head b{color:#f0a83b}.recorder svg{display:block;width:100%;border:1px solid #9d9075;border-radius:0 0 7px 7px;background:#d7ceb6}.paper{fill:#ebe4cf;stroke:#a89d82}.axis{stroke:#8d876f;stroke-width:1}.target-line{stroke:#17324d;stroke-width:2;stroke-dasharray:4 4}.row-title{fill:#17324d;font:800 13px "Arial Narrow","PingFang SC",sans-serif}.row-stat,.scale-label{fill:#60645f;font:700 10px ui-monospace,monospace}label{display:grid;grid-template-columns:1fr auto;gap:6px;color:#d6dded;font-size:11px}label input{grid-column:1/-1;width:100%;min-height:44px;accent-color:#f0a83b}label output{color:#f0a83b;font:750 11px ui-monospace,monospace}button{min-height:44px;border:1px solid #f0a83b;border-radius:5px;color:#10182e;background:#f0a83b;font-weight:800}@media(max-width:760px){.recorder{min-width:0;overflow-x:auto}.paper-head,.recorder svg{min-width:560px}}
</style>
