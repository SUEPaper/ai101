<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Point = { x: number; y: number }

const points: Point[] = [
  {x:-2.05,y:-.62},{x:-1.72,y:-.15},{x:-1.61,y:.38},{x:-1.28,y:-.43},{x:-1.05,y:.25},{x:-.72,y:.58},
  {x:-.25,y:.22},{x:.12,y:.55},{x:.38,y:.92},{x:.62,y:1.28},{x:.88,y:.72},{x:1.08,y:1.12},
  {x:1.36,y:.54},{x:1.58,y:1.04},{x:1.82,y:.38},{x:2.02,y:.84},{x:.02,y:-.52},{x:.54,y:-.22},
]

const means = ref<Point[]>([{ x: -1.85, y: .95 }, { x: 1.72, y: -.65 }])
const variances = ref([1.35, 1.35])
const weights = ref([.5, .5])
const responsibilities = ref(points.map(() => [.5, .5]))
const phase = ref<'ready' | 'expectation' | 'maximization'>('ready')
const iteration = ref(0)
const trace = ref<number[]>([])

function logJoint(point: Point, k: number) {
  const variance = variances.value[k]
  const distanceSq = (point.x - means.value[k].x) ** 2 + (point.y - means.value[k].y) ** 2
  return Math.log(weights.value[k]) - Math.log(2 * Math.PI * variance) - distanceSq / (2 * variance)
}

function posterior(point: Point) {
  const a = logJoint(point, 0)
  const b = logJoint(point, 1)
  const peak = Math.max(a, b)
  const denominator = Math.exp(a - peak) + Math.exp(b - peak)
  return [Math.exp(a - peak) / denominator, Math.exp(b - peak) / denominator]
}

const logLikelihood = computed(() => points.reduce((sum, point) => {
  const a = logJoint(point, 0)
  const b = logJoint(point, 1)
  const peak = Math.max(a, b)
  return sum + peak + Math.log(Math.exp(a - peak) + Math.exp(b - peak))
}, 0))

const effectiveCounts = computed(() => [0, 1].map(k => responsibilities.value.reduce((sum, row) => sum + row[k], 0)))
const meanEntropy = computed(() => responsibilities.value.reduce((sum, row) => {
  return sum - row.reduce((inner, value) => inner + value * Math.log(Math.max(value, 1e-15)), 0)
}, 0) / points.length)

function expectationStep() {
  responsibilities.value = points.map(posterior)
  phase.value = 'expectation'
  if (trace.value.length === 0) trace.value = [logLikelihood.value]
}

function maximizationStep() {
  if (phase.value !== 'expectation') return
  const counts = effectiveCounts.value
  const nextMeans = [0, 1].map(k => ({
    x: points.reduce((sum, point, i) => sum + responsibilities.value[i][k] * point.x, 0) / counts[k],
    y: points.reduce((sum, point, i) => sum + responsibilities.value[i][k] * point.y, 0) / counts[k],
  }))
  const nextVariances = [0, 1].map(k => Math.max(.08, points.reduce((sum, point, i) => {
    const dx = point.x - nextMeans[k].x
    const dy = point.y - nextMeans[k].y
    return sum + responsibilities.value[i][k] * (dx * dx + dy * dy)
  }, 0) / (2 * counts[k])))
  means.value = nextMeans
  variances.value = nextVariances
  weights.value = counts.map(value => value / points.length)
  iteration.value += 1
  phase.value = 'maximization'
  trace.value = [...trace.value, logLikelihood.value].slice(-9)
}

function reset(kind: 'separated' | 'overlap') {
  means.value = kind === 'separated'
    ? [{ x: -1.85, y: .95 }, { x: 1.72, y: -.65 }]
    : [{ x: -.22, y: .12 }, { x: .18, y: .02 }]
  variances.value = kind === 'separated' ? [1.35, 1.35] : [.82, .82]
  weights.value = [.5, .5]
  responsibilities.value = points.map(() => [.5, .5])
  phase.value = 'ready'
  iteration.value = 0
  trace.value = []
}

const mapX = (value: number) => 305 + value * 92
const mapY = (value: number) => 206 - value * 82
const pointColor = (row: number[]) => {
  const a = row[0]
  const r = Math.round(35 * a + 181 * (1 - a))
  const g = Math.round(201 * a + 106 * (1 - a))
  const b = Math.round(189 * a + 240 * (1 - a))
  return `rgb(${r},${g},${b})`
}
const tracePath = computed(() => {
  if (trace.value.length < 2) return ''
  const low = Math.min(...trace.value)
  const high = Math.max(...trace.value)
  const span = Math.max(high - low, 1e-9)
  return trace.value.map((value, index) => {
    const x = 404 + index * (182 / Math.max(trace.value.length - 1, 1))
    const y = 333 - (value - low) / span * 34
    return `${index ? 'L' : 'M'} ${x.toFixed(1)} ${y.toFixed(1)}`
  }).join(' ')
})
</script>

<template>
  <MlExperimentFrame
    index="24-B"
    title="逐拍执行 E 步与 M 步"
    question="先让每条记录形成软分配，再用软计数移动中心和尺度。分开执行两步，观察责任度熵、有效样本数和观测似然怎样联动。"
    formula="E: γᵢₖ←p(zᵢ=k|xᵢ,θᵗ)　　M: θᵗ⁺¹←argmaxθ ∑ᵢₖγᵢₖ log p(xᵢ,zᵢ=k|θ)"
  >
    <template #visual>
      <div class="playback-scroll">
        <svg viewBox="0 0 640 370" role="img" :aria-label="`EM第${iteration}轮，当前阶段${phase}，对数似然${logLikelihood.toFixed(2)}`">
          <rect width="640" height="370" rx="15" class="playback-panel" />
          <text x="24" y="28" class="playback-eyebrow">E/M INDEXING TABLE · SOFT COUNTS IN MOTION</text>
          <g class="playback-grid">
            <line v-for="x in [121,213,305,397,489]" :key="`x-${x}`" :x1="x" :x2="x" y1="48" y2="277" />
            <line v-for="y in [58,140,222]" :key="`y-${y}`" x1="72" x2="538" :y1="y" :y2="y" />
          </g>
          <ellipse v-for="(mean, k) in means" :key="`ellipse-${k}`" :cx="mapX(mean.x)" :cy="mapY(mean.y)" :rx="58 * Math.sqrt(variances[k])" :ry="42 * Math.sqrt(variances[k])" :class="['component-race', `component-${k}`]" />
          <circle v-for="(point, i) in points" :key="`point-${i}`" :cx="mapX(point.x)" :cy="mapY(point.y)" r="7" class="data-point" :style="{ fill: pointColor(responsibilities[i]) }" />
          <g v-for="(mean, k) in means" :key="`mean-${k}`" :transform="`translate(${mapX(mean.x)} ${mapY(mean.y)})`">
            <line x1="-10" x2="10" y1="0" y2="0" :class="['center-mark', `center-${k}`]" />
            <line x1="0" x2="0" y1="-10" y2="10" :class="['center-mark', `center-${k}`]" />
          </g>
          <g transform="translate(30 296)">
            <rect width="342" height="48" rx="5" class="phase-strip" />
            <text x="13" y="18" class="strip-label">CURRENT REGISTER</text>
            <text x="13" y="37" class="strip-value">{{ phase === 'ready' ? '等待 E 步' : phase === 'expectation' ? 'E 步已锁存：可执行 M 步' : `第 ${iteration} 轮 M 步已提交` }}</text>
          </g>
          <g transform="translate(390 293)">
            <rect width="220" height="57" rx="5" class="trace-box" />
            <text x="12" y="16" class="strip-label">OBSERVED LOG-LIKELIHOOD</text>
            <line x1="14" x2="202" y1="42" y2="42" class="trace-axis" />
            <path v-if="tracePath" :d="tracePath" class="trace-line" />
            <text v-else x="110" y="41" text-anchor="middle" class="trace-empty">执行 E 步后开始记录</text>
          </g>
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="button-row"><button type="button" @click="expectationStep">执行 E 步</button><button type="button" :disabled="phase !== 'expectation'" @click="maximizationStep">执行 M 步</button></div>
      <div class="button-row reset-row"><button type="button" @click="reset('separated')">重置：交叉起点</button><button type="button" @click="reset('overlap')">重置：重叠起点</button></div>
    </template>

    <template #metrics>
      <div><span>完整轮次</span><strong>{{ iteration }}</strong></div>
      <div><span>观测 log-likelihood</span><strong>{{ logLikelihood.toFixed(3) }}</strong></div>
      <div><span>软计数 N₁ / N₂</span><strong>{{ effectiveCounts[0].toFixed(1) }} / {{ effectiveCounts[1].toFixed(1) }}</strong></div>
      <div><span>平均责任度熵</span><strong>{{ meanEntropy.toFixed(3) }}</strong></div>
    </template>

    <template #note>颜色混合表示软分配，不是装饰。E 步只改变颜色和软计数；M 步才移动中心、尺度和权重。观测似然要在完整 E/M 轮次的边界比较，不能把“E 步后的责任度变化”误当成一次参数更新。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.playback-scroll{overflow-x:auto}.playback-scroll svg{display:block;width:100%;height:auto}.playback-panel{fill:#0b1218}.playback-grid line{stroke:#23343d;stroke-width:1}.component-race{fill-opacity:.06;stroke-width:2.5;stroke-dasharray:8 5}.component-0{fill:#23c9bd;stroke:#38d8cc}.component-1{fill:#b56af0;stroke:#d477ff}.data-point{stroke:#e7f0f2;stroke-width:1.2}.center-mark{stroke-width:3;stroke-linecap:round}.center-0{stroke:#6be2d9}.center-1{stroke:#dd9eff}.playback-eyebrow,.strip-label,.strip-value,.trace-empty{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.playback-eyebrow{fill:#74a4b1;font-size:9px;letter-spacing:.12em}.phase-strip,.trace-box{fill:#111d25;stroke:#344b56}.strip-label{fill:#77929d;font-size:8px;letter-spacing:.08em}.strip-value{fill:#f2d078;font-size:10px;font-weight:750}.trace-axis{stroke:#3d535e}.trace-line{fill:none;stroke:#38d8cc;stroke-width:2.5}.trace-empty{fill:#6f8791;font-size:8px}.button-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.button-row+*{margin-top:10px}.button-row button{min-height:44px;border:1px solid #42656d;border-radius:6px;color:#dcebed;background:#13232b;font-size:11px;font-weight:760;cursor:pointer}.button-row button:hover:not(:disabled){border-color:#38d8cc}.button-row button:focus-visible{outline:3px solid #d477ff;outline-offset:2px}.button-row button:disabled{cursor:not-allowed;opacity:.42}.reset-row button{border-color:#5a4770;background:#20192a}@media(max-width:560px){.playback-scroll{margin-right:-18px;padding-right:18px}.playback-scroll svg{width:600px}.button-row{grid-template-columns:1fr}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
