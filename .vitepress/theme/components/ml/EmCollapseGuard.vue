<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const logSigma = ref(-.55)
const logFloor = ref(-1.25)
const guarded = ref(false)
const observations = [-2.4,-2.05,-1.72,-1.36,-1.05,-.74,-.42,-.12,.18,.47,.78,1.12,1.48,1.86,3.6]
const spikeWeight = 1 / observations.length

function normalLogPdf(x: number, mean: number, sigma: number) {
  return -.5 * Math.log(2 * Math.PI * sigma * sigma) - .5 * ((x - mean) / sigma) ** 2
}

function logAddExp(a: number, b: number) {
  const peak = Math.max(a, b)
  return peak + Math.log(Math.exp(a - peak) + Math.exp(b - peak))
}

const state = computed(() => {
  const rawSigma = 10 ** logSigma.value
  const floor = 10 ** logFloor.value
  const effectiveSigma = guarded.value ? Math.max(rawSigma, floor) : rawSigma
  const logLikelihood = observations.reduce((sum, value) => {
    const broad = Math.log(1 - spikeWeight) + normalLogPdf(value, 0, 1.35)
    const spike = Math.log(spikeWeight) + normalLogPdf(value, 3.6, effectiveSigma)
    return sum + logAddExp(broad, spike)
  }, 0)
  const spikePeak = spikeWeight / (Math.sqrt(2 * Math.PI) * effectiveSigma)
  const clipped = guarded.value && rawSigma < floor
  return { rawSigma, floor, effectiveSigma, logLikelihood, spikePeak, clipped }
})

const densityPath = computed(() => {
  const values: string[] = []
  for (let i = 0; i <= 150; i++) {
    const x = -3 + i * 7.2 / 150
    const broad = (1 - spikeWeight) * Math.exp(normalLogPdf(x, 0, 1.35))
    const spike = spikeWeight * Math.exp(normalLogPdf(x, 3.6, state.value.effectiveSigma))
    const density = Math.min(2.4, broad + spike)
    const px = 56 + i * 3.5
    const py = 275 - density * 82
    values.push(`${i ? 'L' : 'M'} ${px.toFixed(1)} ${py.toFixed(1)}`)
  }
  return values.join(' ')
})

const mapX = (value: number) => 56 + (value + 3) / 7.2 * 525
</script>

<template>
  <MlExperimentFrame
    index="24-C"
    title="把协方差坍缩送进限位器"
    question="让一个窄成分钉住孤立记录。没有方差下限时，峰值和似然会怎样变化？接入限位器后，算法优化的可行域发生了什么改变？"
    formula="N(x★|μ=x★,σ²)=(2πσ²)⁻¹ᐟ² → ∞　(σ→0)"
  >
    <template #visual>
      <div class="collapse-scroll">
        <svg viewBox="0 0 640 360" role="img" :aria-label="`原始标准差${state.rawSigma.toFixed(4)}，生效标准差${state.effectiveSigma.toFixed(4)}，对数似然${state.logLikelihood.toFixed(2)}`">
          <rect width="640" height="360" rx="15" class="collapse-panel" />
          <text x="24" y="28" class="collapse-eyebrow">COLLAPSING-VARIANCE TRIP BENCH · SPIKE HEIGHT IS CLIPPED ON SCREEN</text>
          <g class="density-grid"><line x1="56" x2="581" y1="275" y2="275"/><line x1="56" x2="56" y1="64" y2="275"/><line x1="56" x2="581" y1="193" y2="193"/><line x1="56" x2="581" y1="111" y2="111"/></g>
          <path :d="densityPath" class="density-path" />
          <line :x1="mapX(3.6)" :x2="mapX(3.6)" y1="58" y2="275" class="spike-axis" />
          <circle v-for="(value, i) in observations" :key="i" :cx="mapX(value)" cy="284" :r="value === 3.6 ? 7 : 4" :class="value === 3.6 ? 'isolated-point' : 'observation'" />
          <text :x="mapX(3.6) - 5" y="52" text-anchor="end" class="spike-label">孤立记录 x★</text>
          <g transform="translate(44 307)">
            <rect width="552" height="29" rx="5" class="guard-rail" />
            <rect x="3" y="3" :width="guarded ? 546 : 0" height="23" rx="3" class="guard-live" />
            <text x="276" y="19" text-anchor="middle" class="guard-label">{{ guarded ? (state.clipped ? 'VARIANCE FLOOR CLAMPED · σ 停在下限' : 'VARIANCE FLOOR ARMED · 尚未触发') : 'NO FLOOR · 目标允许继续追逐无限尖峰' }}</text>
          </g>
          <g transform="translate(78 70)">
            <rect width="176" height="51" rx="5" class="trip-card" />
            <text x="12" y="18" class="trip-label">SPIKE PEAK DENSITY</text>
            <text x="12" y="39" class="trip-value">{{ state.spikePeak.toExponential(2) }}</text>
          </g>
        </svg>
      </div>
    </template>

    <template #controls>
      <label class="control"><span>原始 log₁₀σ <output>{{ logSigma.toFixed(2) }}</output></span><input v-model.number="logSigma" aria-label="尖峰标准差的十对数" type="range" min="-3" max="0.25" step="0.01"></label>
      <label class="control"><span>下限 log₁₀σmin <output>{{ logFloor.toFixed(2) }}</output></span><input v-model.number="logFloor" aria-label="标准差下限的十对数" type="range" min="-2.4" max="-0.35" step="0.01"></label>
      <div class="guard-buttons" role="group" aria-label="协方差限位器"><button type="button" :aria-pressed="!guarded" @click="guarded=false">旁路限位器</button><button type="button" :aria-pressed="guarded" @click="guarded=true">接入方差下限</button></div>
    </template>

    <template #metrics>
      <div><span>原始 σ</span><strong>{{ state.rawSigma.toFixed(4) }}</strong></div>
      <div><span>生效 σ</span><strong>{{ state.effectiveSigma.toFixed(4) }}</strong></div>
      <div><span>观测 log-likelihood</span><strong>{{ state.logLikelihood.toFixed(3) }}</strong></div>
      <div><span>限位状态</span><strong>{{ state.clipped ? 'CLAMPED' : guarded ? 'ARMED' : 'BYPASS' }}</strong></div>
    </template>

    <template #note>图上的密度峰为了可读性在 2.4 处截顶，指标显示真实峰值。方差下限不是“修复了 MLE”：它改变了可行参数空间，因此必须与模型版本一起冻结。生产实现还应监控最小特征值、有效样本数和重启间稳定性。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.collapse-scroll{overflow-x:auto}.collapse-scroll svg{display:block;width:100%;height:auto}.collapse-panel{fill:#0b1218}.density-grid line{stroke:#2a3b44;stroke-width:1}.density-path{fill:none;stroke:#38d8cc;stroke-width:3}.spike-axis{stroke:#ee725f;stroke-width:2;stroke-dasharray:6 5}.observation{fill:#7f99a4}.isolated-point{fill:#ee725f;stroke:#ffd8d1;stroke-width:2}.collapse-eyebrow,.spike-label,.guard-label,.trip-label,.trip-value{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.collapse-eyebrow{fill:#74a4b1;font-size:9px;letter-spacing:.1em}.spike-label{fill:#ff9b8c;font-size:9px}.guard-rail{fill:#1a2024;stroke:#4a565c}.guard-live{fill:#1d645f}.guard-label{fill:#dce9eb;font-size:9px;font-weight:750}.trip-card{fill:#171b1e;stroke:#8e4d45}.trip-label{fill:#a98781;font-size:8px;letter-spacing:.08em}.trip-value{fill:#ff9989;font-size:15px;font-weight:800}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebdc5;font-size:11px}.control output{color:#ffc864;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#ee725f}input:focus-visible,button:focus-visible{outline:3px solid #d477ff;outline-offset:2px}.guard-buttons{display:grid;grid-template-columns:1fr 1fr;gap:8px}.guard-buttons button{min-height:44px;border:1px solid #65525b;border-radius:6px;color:#dcced3;background:#21191c;font-size:11px;font-weight:760;cursor:pointer}.guard-buttons button[aria-pressed="true"]{border-color:#ee725f;color:#fff;background:#6d3028}@media(max-width:560px){.collapse-scroll{margin-right:-18px;padding-right:18px}.collapse-scroll svg{width:600px}.guard-buttons{grid-template-columns:1fr}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
