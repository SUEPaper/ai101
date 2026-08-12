<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Point = { x: number; y: number }
type Phase = '待分配' | '分配完成' | '均值更新完成'

const points: Point[] = [
  { x: -2.9, y: -0.7 }, { x: -2.7, y: 0.2 }, { x: -2.5, y: 0.9 },
  { x: -2.2, y: -0.3 }, { x: -2.0, y: 0.5 }, { x: -1.8, y: -0.9 },
  { x: -1.6, y: 0.1 }, { x: -1.4, y: 0.8 }, { x: -1.2, y: -0.4 },
  { x: 0.9, y: 1.2 }, { x: 1.1, y: 2.0 }, { x: 1.5, y: 1.5 },
  { x: 1.8, y: 2.2 }, { x: 2.0, y: 1.2 }, { x: 2.4, y: 1.7 },
  { x: 2.7, y: 2.1 }, { x: 1.0, y: -1.8 }, { x: 1.3, y: -1.0 },
  { x: 1.6, y: -2.1 }, { x: 1.9, y: -1.4 }, { x: 2.2, y: -2.2 },
  { x: 2.5, y: -1.1 }, { x: 2.8, y: -1.8 },
]

const palette = ['#f28c42', '#5b8cff', '#a8d85c']
const preset = ref<'crowded' | 'covered'>('crowded')
const centers = ref<Point[]>([])
const labels = ref<number[]>([])
const previousAssignment = ref<number[]>([])
const phase = ref<Phase>('待分配')
const iteration = ref(0)
const lastMoved = ref(0)
const lastDrop = ref(0)
const objective = ref<number | null>(null)
const traces = ref<Point[][]>([])
const ledger = ref<Array<{ label: string; value: number }>>([])

function squaredDistance(a: Point, b: Point) {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2
}

function initialCenters(): Point[] {
  if (preset.value === 'covered') {
    return [{ x: -2.6, y: 0.7 }, { x: 1.1, y: 1.2 }, { x: 2.7, y: -1.8 }]
  }
  return [{ x: -2.9, y: -0.7 }, { x: -2.0, y: 0.5 }, { x: -1.2, y: -0.4 }]
}

function reset() {
  centers.value = initialCenters()
  labels.value = Array(points.length).fill(-1)
  previousAssignment.value = []
  phase.value = '待分配'
  iteration.value = 0
  lastMoved.value = 0
  lastDrop.value = 0
  objective.value = null
  traces.value = centers.value.map(center => [{ ...center }])
  ledger.value = []
}

function assign() {
  const next = points.map(point => {
    const distances = centers.value.map(center => squaredDistance(point, center))
    return distances.indexOf(Math.min(...distances))
  })
  lastMoved.value = previousAssignment.value.length
    ? next.filter((label, index) => label !== previousAssignment.value[index]).length
    : next.length
  labels.value = next
  previousAssignment.value = [...next]
  const nextObjective = points.reduce((sum, point, index) => sum + squaredDistance(point, centers.value[next[index]]), 0)
  if (objective.value !== null) lastDrop.value = objective.value - nextObjective
  objective.value = nextObjective
  phase.value = '分配完成'
  ledger.value.push({ label: `A${iteration.value + 1}`, value: nextObjective })
}

function updateCenters() {
  if (labels.value.some(label => label < 0)) assign()
  const updated = centers.value.map((center, cluster) => {
    const members = points.filter((_, index) => labels.value[index] === cluster)
    if (!members.length) return center
    return {
      x: members.reduce((sum, point) => sum + point.x, 0) / members.length,
      y: members.reduce((sum, point) => sum + point.y, 0) / members.length,
    }
  })
  const nextObjective = points.reduce((sum, point, index) => sum + squaredDistance(point, updated[labels.value[index]]), 0)
  lastDrop.value = (objective.value ?? nextObjective) - nextObjective
  centers.value = updated
  traces.value = traces.value.map((trace, cluster) => [...trace, { ...updated[cluster] }])
  objective.value = nextObjective
  iteration.value += 1
  phase.value = '均值更新完成'
  ledger.value.push({ label: `M${iteration.value}`, value: nextObjective })
}

function completeRound() {
  assign()
  updateCenters()
}

function switchPreset() {
  preset.value = preset.value === 'crowded' ? 'covered' : 'crowded'
  reset()
}

function sx(x: number) { return 42 + (x + 3.5) / 7 * 476 }
function sy(y: number) { return 320 - (y + 2.7) / 5.4 * 280 }

const counts = computed(() => centers.value.map((_, cluster) => labels.value.filter(label => label === cluster).length))
const ledgerMax = computed(() => Math.max(1, ...ledger.value.map(item => item.value)))

reset()
</script>

<template>
  <MlExperimentFrame
    index="26-B"
    title="Lloyd 交替传送带"
    question="分配步和均值步分别降低了多少 WCSS？拥挤起点会让质心走出怎样的轨迹？"
    formula="J(zᵗ⁺¹, μᵗ) ≤ J(zᵗ, μᵗ)；J(zᵗ⁺¹, μᵗ⁺¹) ≤ J(zᵗ⁺¹, μᵗ)"
  >
    <template #visual>
      <div class="playback-scroll">
        <svg viewBox="0 0 560 380" role="img" aria-label="K-means 分配与质心更新逐拍回放">
          <rect class="deck" x="20" y="20" width="520" height="340" rx="8" />
          <g class="grid" aria-hidden="true">
            <line v-for="x in [110, 178, 246, 314, 382, 450]" :key="`x-${x}`" :x1="x" y1="40" :x2="x" y2="320" />
            <line v-for="y in [96, 152, 208, 264]" :key="`y-${y}`" x1="42" :y1="y" x2="518" :y2="y" />
          </g>
          <g v-if="labels.every(label => label >= 0)" class="assignment-wires" aria-hidden="true">
            <line
              v-for="(point, index) in points"
              :key="`wire-${index}`"
              :x1="sx(point.x)"
              :y1="sy(point.y)"
              :x2="sx(centers[labels[index]].x)"
              :y2="sy(centers[labels[index]].y)"
              :stroke="palette[labels[index]]"
            />
          </g>
          <g aria-hidden="true">
            <polyline
              v-for="(trace, cluster) in traces"
              :key="`trace-${cluster}`"
              :points="trace.map(point => `${sx(point.x)},${sy(point.y)}`).join(' ')"
              :stroke="palette[cluster]"
            />
          </g>
          <circle
            v-for="(point, index) in points"
            :key="index"
            :cx="sx(point.x)"
            :cy="sy(point.y)"
            r="6"
            :fill="labels[index] < 0 ? '#d6ded6' : palette[labels[index]]"
          />
          <g v-for="(center, cluster) in centers" :key="`center-${cluster}`" :transform="`translate(${sx(center.x)} ${sy(center.y)})`">
            <path :stroke="palette[cluster]" d="M -11 0 H 11 M 0 -11 V 11" />
            <circle :stroke="palette[cluster]" r="14" />
            <text y="30" text-anchor="middle">μ{{ cluster + 1 }}</text>
          </g>
          <text class="bay-label" x="42" y="34">LLOYD HALF-STEP LEDGER · {{ preset === 'crowded' ? 'CROWDED START' : 'COVERED START' }}</text>
          <g class="ledger" transform="translate(42 329)">
            <rect x="0" y="0" width="476" height="20" rx="3" />
            <rect
              v-if="ledger.length"
              x="0"
              y="0"
              :width="476 * ledger[ledger.length - 1].value / ledgerMax"
              height="20"
              rx="3"
            />
            <text x="8" y="14">{{ ledger.length ? `${ledger[ledger.length - 1].label} · J=${ledger[ledger.length - 1].value.toFixed(2)}` : '等待第一次分配' }}</text>
          </g>
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="button-grid">
        <button type="button" @click="assign">1 · 执行最近中心分配</button>
        <button type="button" @click="updateCenters">2 · 执行簇内均值更新</button>
        <button type="button" class="round" @click="completeRound">执行一整轮</button>
        <button type="button" @click="switchPreset">切换为{{ preset === 'crowded' ? '覆盖' : '拥挤' }}起点</button>
        <button type="button" @click="reset">复位当前起点</button>
      </div>
    </template>

    <template #metrics>
      <div><span>当前半步</span><strong>{{ phase }}</strong></div>
      <div><span>WCSS / J</span><strong>{{ objective === null ? '未计算' : objective.toFixed(3) }}</strong></div>
      <div><span>本半步下降量</span><strong>{{ lastDrop.toFixed(3) }}</strong></div>
      <div><span>分配计数</span><strong>{{ counts.join(' / ') }} · 移动 {{ lastMoved }}</strong></div>
    </template>

    <template #note>细线是当前样本到质心的平方距离路径，粗轨迹记录均值移动。逐个点击 A（分配）与 M（更新）可核对两条不等式；“一整轮”只是把两个精确子问题连续执行，并没有直接求解联合非凸问题。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.playback-scroll{overflow-x:auto}.playback-scroll svg{display:block;width:100%;height:auto}.deck{fill:#0b1418}.grid line{stroke:#d7dfd4;stroke-width:.8;opacity:.12}.assignment-wires line{stroke-width:1.2;opacity:.33}polyline{fill:none;stroke-width:3;stroke-dasharray:5 4}circle{stroke:#f4f6ed;stroke-width:1.4}g>path{fill:none;stroke-width:3}g>circle{fill:#0b1418;stroke-width:2}.bay-label,svg text{fill:#eef3e5;font:750 9px/1 ui-monospace,SFMono-Regular,Menlo,monospace}.bay-label{fill:#94a59a;letter-spacing:.1em}.ledger rect:first-child{fill:#26322c}.ledger rect:nth-child(2){fill:#f28c42}.ledger text{fill:#101817;font-weight:850}.button-grid{display:grid;gap:9px}button{min-height:44px;border:1px solid #53645a;border-radius:6px;color:#e8eee6;background:#17231d;font:720 11px/1.3 ui-monospace,monospace;cursor:pointer}.round{border-color:#ef9b5d;color:#ffd7b7;background:#3a2418}button:focus-visible{outline:3px solid #a8d85c;outline-offset:2px}@media(max-width:560px){.playback-scroll{margin-right:-18px;padding-right:18px}.playback-scroll svg{width:620px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
