<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const legacyErrors = [2, 0, 1, 2, 2, 5, 0, 3, 1, 2, 2, 4]
const candidateErrors = [0, 0, 1, 1, 1, 5, 0, 3, 0, 2, 3, 5]
const variants = ['A', 'A', 'A', 'B', 'B', 'C', 'A', 'A', 'A', 'B', 'B', 'C']
const lotSize = 12

const paired = ref(true)
const repeats = ref(4000)
const confidence = ref(0.95)
const showGroup = ref(true)

function seededRandom(seed: number) {
  let state = seed >>> 0
  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 2 ** 32
  }
}

function quantile(sorted: number[], probability: number) {
  const position = (sorted.length - 1) * probability
  const lower = Math.floor(position)
  const fraction = position - lower
  return sorted[lower] * (1 - fraction) + sorted[Math.min(lower + 1, sorted.length - 1)] * fraction
}

const bootstrap = computed(() => {
  const random = seededRandom(2818 + repeats.value + (paired.value ? 1 : 17))
  const values: number[] = []
  for (let repeat = 0; repeat < repeats.value; repeat += 1) {
    let difference = 0
    for (let draw = 0; draw < legacyErrors.length; draw += 1) {
      const candidateIndex = Math.floor(random() * candidateErrors.length)
      const legacyIndex = paired.value ? candidateIndex : Math.floor(random() * legacyErrors.length)
      difference += candidateErrors[candidateIndex] - legacyErrors[legacyIndex]
    }
    values.push(difference / (legacyErrors.length * lotSize))
  }
  values.sort((a, b) => a - b)
  const tail = (1 - confidence.value) / 2
  return {
    values,
    lower: quantile(values, tail),
    median: quantile(values, 0.5),
    upper: quantile(values, 1 - tail),
    betterProbability: values.filter(value => value < 0).length / values.length,
  }
})

const pointDifference = (candidateErrors.reduce((a, b) => a + b, 0) - legacyErrors.reduce((a, b) => a + b, 0)) / (legacyErrors.length * lotSize)
const histogram = computed(() => {
  const minimum = -0.15
  const maximum = 0.12
  const bins = 24
  const counts = Array(bins).fill(0)
  bootstrap.value.values.forEach(value => {
    const index = Math.max(0, Math.min(bins - 1, Math.floor((value - minimum) / (maximum - minimum) * bins)))
    counts[index] += 1
  })
  const peak = Math.max(...counts)
  return counts.map((count, index) => ({
    x: 64 + index * 20,
    y: 266 - count / peak * 142,
    height: count / peak * 142,
  }))
})

const groupC = computed(() => {
  const indices = variants.map((variant, index) => variant === 'C' ? index : -1).filter(index => index >= 0)
  const legacy = indices.reduce((sum, index) => sum + legacyErrors[index], 0) / (indices.length * lotSize)
  const candidate = indices.reduce((sum, index) => sum + candidateErrors[index], 0) / (indices.length * lotSize)
  return { legacy, candidate }
})

function sx(value: number) { return 64 + (value + 0.15) / 0.27 * 480 }
function percent(value: number) { return `${(value * 100).toFixed(1)}%` }
</script>

<template>
  <MlExperimentFrame
    index="28-C"
    title="成对批次差值尺"
    question="候选模型少错了三张图，能否发布？同一批次上的两模型损失为什么必须一起重采样？"
    formula="Δb = meanlot[Lcandidate − Llegacy] · CI = quantile({Δb})"
  >
    <template #visual>
      <div class="rail-scroll">
        <svg viewBox="0 0 630 390" role="img" aria-label="按生产批次成对 Bootstrap 的模型误差差值分布">
          <rect class="case" x="14" y="14" width="602" height="362" rx="3" />
          <text class="caption" x="40" y="42">PAIRED LOT RESAMPLING · Δ ERROR</text>
          <text class="caption" x="590" y="42" text-anchor="end">{{ paired ? 'PAIR LOCKED' : 'PAIR BROKEN · DEMO' }}</text>
          <rect class="plot" x="42" y="66" width="524" height="224" rx="3" />
          <rect v-for="(bar, index) in histogram" :key="index" class="histogram" :x="bar.x" :y="bar.y" width="16" :height="bar.height" />
          <line class="zero" :x1="sx(0)" y1="76" :x2="sx(0)" y2="286" />
          <line class="interval" :x1="sx(bootstrap.lower)" y1="102" :x2="sx(bootstrap.upper)" y2="102" />
          <circle class="interval-end" :cx="sx(bootstrap.lower)" cy="102" r="6" />
          <circle class="interval-end" :cx="sx(bootstrap.upper)" cy="102" r="6" />
          <path class="point-marker" :transform="`translate(${sx(pointDifference)} 92)`" d="M 0 -12 L 9 0 L 0 12 L -9 0 Z" />
          <text class="ci-label" :x="sx(bootstrap.lower)" y="88" text-anchor="middle">{{ percent(bootstrap.lower) }}</text>
          <text class="ci-label" :x="sx(bootstrap.upper)" y="88" text-anchor="middle">{{ percent(bootstrap.upper) }}</text>
          <g class="axis">
            <line x1="64" y1="286" x2="544" y2="286" />
            <g v-for="tick in [-.15, -.10, -.05, 0, .05, .10]" :key="tick">
              <line :x1="sx(tick)" y1="282" :x2="sx(tick)" y2="293" />
              <text :x="sx(tick)" y="309" text-anchor="middle">{{ (tick * 100).toFixed(0) }}pp</text>
            </g>
          </g>
          <g v-if="showGroup" class="group-strip">
            <text x="44" y="337">稀少规格 C</text>
            <rect x="132" y="326" :width="groupC.legacy * 330" height="13" />
            <rect class="candidate" x="132" y="345" :width="groupC.candidate * 330" height="13" />
            <text x="480" y="337">旧 {{ percent(groupC.legacy) }}</text>
            <text class="candidate-text" x="480" y="356">新 {{ percent(groupC.candidate) }}</text>
          </g>
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="mode" aria-label="重采样方式">
        <button type="button" :aria-pressed="paired" @click="paired = true">同批次成对</button>
        <button type="button" :aria-pressed="!paired" class="broken" @click="paired = false">错误示范：拆对</button>
      </div>
      <label class="control">
        <span>Bootstrap 次数 <output>{{ repeats }}</output></span>
        <input v-model.number="repeats" type="range" min="500" max="10000" step="500" aria-label="Bootstrap 重复次数">
      </label>
      <div class="confidence-buttons" aria-label="置信水平">
        <button v-for="level in [.90, .95, .99]" :key="level" type="button" :aria-pressed="confidence === level" @click="confidence = level">{{ (level * 100).toFixed(0) }}%</button>
      </div>
      <button type="button" :aria-pressed="showGroup" @click="showGroup = !showGroup">{{ showGroup ? '隐藏规格 C 切片' : '显示规格 C 切片' }}</button>
    </template>

    <template #metrics>
      <div><span>点估计 新−旧</span><strong>{{ percent(pointDifference) }}</strong></div>
      <div><span>{{ (confidence * 100).toFixed(0) }}% 区间</span><strong>[{{ percent(bootstrap.lower) }}, {{ percent(bootstrap.upper) }}]</strong></div>
      <div><span>Bootstrap P(新更好)</span><strong>{{ percent(bootstrap.betterProbability) }}</strong></div>
      <div><span>规格 C 新−旧</span><strong>+{{ percent(groupC.candidate - groupC.legacy) }}</strong></div>
    </template>

    <template #note>12 个冻结测试 lot 中，候选模型共少错 3/144 张，但成对 lot Bootstrap 区间仍跨过 0，不能宣称 95% 置信的改进；稀少规格 C 还从 37.5% 恶化到 41.7%。拆开两模型的 lot 索引会丢掉协方差，只作为错误示范保留。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.rail-scroll{overflow-x:auto}.rail-scroll svg{display:block;width:100%;height:auto}.case{fill:#0c141a;stroke:#d8b75c;stroke-width:2}.plot{fill:#111f28;stroke:#48616c}.caption,.ci-label,.axis text,.group-strip text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.caption{fill:#d8b75c;font-size:9px;font-weight:850;letter-spacing:.08em}.histogram{fill:#67c8bd;opacity:.68}.zero{stroke:#d65a56;stroke-width:2;stroke-dasharray:7 6}.interval{stroke:#eee9d8;stroke-width:4}.interval-end{fill:#eee9d8;stroke:#7468b4;stroke-width:2}.point-marker{fill:#d8b75c;stroke:#fff0bd;stroke-width:1.5}.ci-label{fill:#eee9d8;font-size:8px}.axis line{stroke:#879ba1}.axis text{fill:#aebdc1;font-size:8px}.group-strip text{fill:#c6d2d4;font-size:9px;font-weight:750}.group-strip rect{fill:#67c8bd}.group-strip rect.candidate{fill:#d65a56}.group-strip .candidate-text{fill:#ffaaa5}.mode,.confidence-buttons{display:grid;grid-template-columns:1fr 1fr;gap:6px}.confidence-buttons{grid-template-columns:repeat(3,1fr)}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#c2cee9;font-size:11px}.control output{color:#f1ce79;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#67c8bd}button{min-height:44px;padding:9px;border:1px solid #536a74;border-radius:4px;color:#dbe6e8;background:#14252e;font:720 10px/1.3 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#d8b75c;color:#ffe9ad;background:#4b3d21}.broken{border-color:#d65a56;color:#ffd3d0}button:focus-visible,input:focus-visible{outline:3px solid #d8b75c;outline-offset:2px}@media(max-width:560px){.rail-scroll{margin-right:-18px;padding-right:18px}.rail-scroll svg{width:690px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
