<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const round = ref(3)
const contaminated = ref(false)
const positions = [0.07, 0.15, 0.25, 0.35, 0.46, 0.46, 0.61, 0.72, 0.83, 0.93]
const cleanLabels = [-1, -1, 1, 1, 1, 1, -1, 1, 1, -1]

const labels = computed(() => cleanLabels.map((label, index) => contaminated.value && index === 5 ? -1 : label))

const history = computed(() => {
  const n = positions.length
  let weights = Array(n).fill(1 / n)
  let scores = Array(n).fill(0)
  let bound = 1
  let alphaSum = 0
  const unique = [...new Set(positions)]
  const thresholds = [-0.02, ...unique.slice(0, -1).map((value, index) => (value + unique[index + 1]) / 2), 1.02]
  return Array.from({ length: 8 }, (_, index) => {
    let best: { error: number; threshold: number; polarity: number; prediction: number[] } | null = null
    for (const threshold of thresholds) {
      for (const polarity of [-1, 1]) {
        const prediction = positions.map(value => polarity * (value > threshold ? 1 : -1))
        const error = prediction.reduce((sum, value, sample) => sum + (value === labels.value[sample] ? 0 : weights[sample]), 0)
        if (!best || error < best.error - 1e-12 || (Math.abs(error - best.error) < 1e-12 && threshold < best.threshold)) {
          best = { error, threshold, polarity, prediction }
        }
      }
    }
    if (!best || best.error >= .5) return null
    const clipped = Math.min(1 - 1e-12, Math.max(1e-12, best.error))
    const alpha = .5 * Math.log((1 - clipped) / clipped)
    const raw = weights.map((weight, sample) => weight * Math.exp(-alpha * labels.value[sample] * best!.prediction[sample]))
    const normalizer = raw.reduce((sum, value) => sum + value, 0)
    const nextWeights = raw.map(value => value / normalizer)
    scores = scores.map((value, sample) => value + alpha * best!.prediction[sample])
    bound *= normalizer
    alphaSum += Math.abs(alpha)
    const margins = scores.map((value, sample) => labels.value[sample] * value / Math.max(alphaSum, 1e-12))
    const trainError = scores.filter((value, sample) => (value >= 0 ? 1 : -1) !== labels.value[sample]).length / n
    const state = { ...best, alpha, normalizer, weightsBefore: weights, weightsAfter: nextWeights, scores: [...scores], margins, trainError, bound }
    weights = nextWeights
    return state
  }).filter(Boolean) as Array<{
    error: number; threshold: number; polarity: number; prediction: number[]; alpha: number; normalizer: number
    weightsBefore: number[]; weightsAfter: number[]; scores: number[]; margins: number[]; trainError: number; bound: number
  }>
})

const selected = computed(() => history.value[Math.min(round.value, history.value.length) - 1] ?? history.value[0])
const displayWeights = computed(() => selected.value?.weightsAfter ?? Array(positions.length).fill(1 / positions.length))
const gateX = computed(() => 48 + Math.max(0, Math.min(1, selected.value?.threshold ?? .5)) * 530)
const effectiveSamples = computed(() => 1 / displayWeights.value.reduce((sum, value) => sum + value ** 2, 0))
</script>

<template>
  <MlExperimentFrame
    index="21-A"
    title="让复检工单沿弱分类器接力增厚"
    question="每轮树桩只比随机猜测好一点，为什么错分卷材会越来越醒目？接入一条同坐标反标签记录后，权重会流向哪里？"
    formula="wᵢ⁽ᵗ⁺¹⁾ ∝ wᵢ⁽ᵗ⁾ exp(−αₜyᵢhₜ(xᵢ))"
  >
    <template #visual>
      <div class="relay-scroll">
        <svg viewBox="0 0 640 340" role="img" :aria-label="`AdaBoost第${round}轮，加权错误率${selected?.error.toFixed(3)}，训练错误率${selected?.trainError.toFixed(3)}`">
          <rect width="640" height="340" rx="14" class="panel" />
          <text x="24" y="27" class="eyebrow">COATING REVIEW RELAY · ROUND {{ round }}</text>
          <rect x="32" y="54" width="566" height="178" rx="10" class="belt" />
          <line x1="48" y1="166" x2="578" y2="166" class="rail" />
          <line :x1="gateX" y1="68" :x2="gateX" y2="218" class="gate" />
          <path :d="`M${gateX-8} 78 h16 l-8 -14 z`" class="gate-head" />
          <text :x="gateX" y="91" text-anchor="middle" class="gate-label">{{ selected?.polarity === 1 ? '右侧判返工' : '左侧判返工' }}</text>
          <g v-for="(position,index) in positions" :key="index">
            <line :x1="48+position*530" y1="166" :x2="48+position*530" :y2="166-(selected?.margins[index] ?? 0)*34" class="margin-stem" :class="{ wrong:(selected?.margins[index] ?? 0) <= 0 }" />
            <circle :cx="48+position*530" cy="166" :r="8+displayWeights[index]*58" class="sample" :class="labels[index] === 1 ? 'positive' : 'negative'" />
            <text :x="48+position*530" y="170" text-anchor="middle" class="sample-id">{{ index+1 }}</text>
            <rect :x="42+position*530" :y="252-displayWeights[index]*150" width="12" :height="displayWeights[index]*150" class="weight-bar" :class="{ hot:displayWeights[index]>.16 }" />
          </g>
          <line x1="48" y1="252" x2="578" y2="252" class="weight-base" />
          <text x="48" y="278" class="caption">圆片大小 / 下方柱高 = 下一轮归一化样本权重</text>
          <text x="48" y="304" class="legend positive-text">返工 +1</text>
          <text x="124" y="304" class="legend negative-text">放行 −1</text>
          <text x="215" y="304" class="legend hot-text">橙柱 = 权重集中</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>接力轮次 <output>{{ round }}</output></span><input v-model.number="round" aria-label="AdaBoost接力轮次" type="range" min="1" max="8" step="1"></label>
      <button type="button" :class="{ active:contaminated }" :aria-pressed="contaminated" @click="contaminated=!contaminated">
        {{ contaminated ? '已接入：同坐标反标签' : '接入同坐标反标签' }}
      </button>
      <p class="hint">第 5、6 卷坐标相同。污染开关把第 6 卷标签翻转，使任何只看该坐标的树桩都无法同时判对二者。</p>
    </template>
    <template #metrics>
      <div><span>本轮加权错误 εₜ</span><strong>{{ selected?.error.toFixed(3) }}</strong></div>
      <div><span>弱学习器票重 αₜ</span><strong>{{ selected?.alpha.toFixed(3) }}</strong></div>
      <div><span>训练错误 / 上界</span><strong>{{ (100*(selected?.trainError ?? 0)).toFixed(1) }}% / {{ (100*(selected?.bound ?? 0)).toFixed(1) }}%</strong></div>
      <div><span>有效样本数 1/Σw²</span><strong>{{ effectiveSamples.toFixed(1) }} / 10</strong></div>
    </template>
    <template #note>最优 α 会把“本轮错分组”和“本轮正确组”的总权重各推到 1/2，但多轮后的权重不是异常概率。矛盾标签会持续吸走优化预算，工业管线应先审计标签与重复实体，而不是无限加轮数。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.relay-scroll{overflow-x:auto}.relay-scroll svg{display:block;min-width:600px;width:100%;height:auto}.panel{fill:#111923}.belt{fill:#182633;stroke:#4c6170}.rail,.weight-base{stroke:#82939c;stroke-width:2}.eyebrow,.caption,.legend,.gate-label,.sample-id{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#72d4c7;font-size:10px;letter-spacing:.12em}.gate{stroke:#f0be55;stroke-width:3;stroke-dasharray:6 4}.gate-head{fill:#f0be55}.gate-label{fill:#f7d987;font-size:8px}.sample{stroke-width:2}.positive{fill:#b85f3d;stroke:#f4a37d}.negative{fill:#267d84;stroke:#70d1cf}.sample-id{fill:#fff;font-size:8px;font-weight:800}.margin-stem{stroke:#6dd2c8;stroke-width:2}.margin-stem.wrong{stroke:#ef6f62;stroke-width:3}.weight-bar{fill:#6dd2c8}.weight-bar.hot{fill:#e27b47}.caption,.legend{fill:#9fb0ba;font-size:8px}.positive-text{fill:#f0a17e}.negative-text{fill:#73d4d0}.hot-text{fill:#efad73}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebbdc;font-size:11px}.control output{color:#f0c76b;font:750 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#c96d45}button{min-height:44px;padding:9px 12px;border:1px solid #52657a;border-radius:6px;color:#d8e3e8;background:#172634;font-weight:750;cursor:pointer}button.active{border-color:#e47c51;color:#ffe3d2;background:#42271f}.hint{margin:0;padding:11px;border-left:3px solid #69cfc4;color:#a8b9c5;background:#101923;font-size:10px;line-height:1.7}button:focus-visible,input:focus-visible{outline:3px solid #6fd5ca;outline-offset:2px}@media(max-width:560px){.relay-scroll{margin-right:-18px;padding-right:18px}.relay-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
