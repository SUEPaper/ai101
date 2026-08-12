<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Point = { x: number; y: number; name: string }
type Separator = { nx: number; ny: number; threshold: number }

const pointCount = ref<3 | 4>(3)
const labels = ref([0, 1, 0, 1])
const threePoints: Point[] = [
  { x: -0.95, y: -0.72, name: 'A' },
  { x: 0.95, y: -0.72, name: 'B' },
  { x: 0.0, y: 0.92, name: 'C' },
]
const fourPoints: Point[] = [
  { x: -0.92, y: -0.78, name: 'A' },
  { x: 0.92, y: -0.78, name: 'B' },
  { x: 0.92, y: 0.78, name: 'C' },
  { x: -0.92, y: 0.78, name: 'D' },
]

const points = computed(() => pointCount.value === 3 ? threePoints : fourPoints)
watch(pointCount, value => {
  labels.value = value === 3 ? [0, 1, 0, 1] : [1, 0, 1, 0]
})

function findSeparator(target: number[]): Separator | null {
  const current = points.value
  for (let degree = 0; degree < 720; degree += 1) {
    const angle = degree / 720 * 2 * Math.PI
    const nx = Math.cos(angle)
    const ny = Math.sin(angle)
    const projections = current.map(point => nx * point.x + ny * point.y)
    const sorted = [...projections].sort((a, b) => a - b)
    const thresholds = [sorted[0] - 0.2, ...sorted.slice(0, -1).map((value, index) => (value + sorted[index + 1]) / 2), sorted.at(-1)! + 0.2]
    for (const threshold of thresholds) {
      const predicted = projections.map(value => value >= threshold ? 1 : 0)
      if (predicted.every((value, index) => value === target[index])) return { nx, ny, threshold }
    }
  }
  return null
}

const separator = computed(() => findSeparator(labels.value.slice(0, pointCount.value)))
const patternCount = computed(() => {
  let count = 0
  for (let pattern = 0; pattern < 2 ** pointCount.value; pattern += 1) {
    const target = Array.from({ length: pointCount.value }, (_, index) => (pattern >> index) & 1)
    if (findSeparator(target)) count += 1
  }
  return count
})

const lineEnds = computed(() => {
  if (!separator.value) return []
  const { nx, ny, threshold } = separator.value
  const limit = 1.3
  const candidates: Array<{ x: number; y: number }> = []
  for (const x of [-limit, limit]) {
    if (Math.abs(ny) > 1e-9) {
      const y = (threshold - nx * x) / ny
      if (y >= -limit && y <= limit) candidates.push({ x, y })
    }
  }
  for (const y of [-limit, limit]) {
    if (Math.abs(nx) > 1e-9) {
      const x = (threshold - ny * y) / nx
      if (x >= -limit && x <= limit) candidates.push({ x, y })
    }
  }
  return candidates.filter((point, index) => candidates.findIndex(other => Math.abs(other.x - point.x) < 1e-8 && Math.abs(other.y - point.y) < 1e-8) === index).slice(0, 2)
})

function sx(value: number) { return 310 + value / 1.3 * 208 }
function sy(value: number) { return 205 - value / 1.3 * 142 }
function toggleLabel(index: number) {
  labels.value[index] = 1 - labels.value[index]
  labels.value = [...labels.value]
}
function setPointCount(value: 3 | 4) { pointCount.value = value }
function setXor() {
  pointCount.value = 4
  labels.value = [1, 0, 1, 0]
}
</script>

<template>
  <MlExperimentFrame
    index="28-B"
    title="VC 可打散光学台"
    question="参数连续并不等于容量无限：二维直线能否为这组点实现你指定的全部二分类标记？"
    formula="VC(H)=max{m: ΠH(m)=2ᵐ}"
  >
    <template #visual>
      <div class="comparator-scroll">
        <svg viewBox="0 0 630 390" role="img" aria-label="二维线性分类器对三点和四点标记的可打散实验">
          <rect class="housing" x="14" y="14" width="602" height="362" rx="3" />
          <rect class="pattern-card" x="34" y="48" width="194" height="288" rx="3" />
          <text class="caption" x="48" y="72">DICHOTOMY COUNTER</text>
          <text class="pattern-number" x="48" y="130">{{ patternCount }}</text>
          <text class="pattern-total" x="48" y="154">/ {{ 2 ** pointCount }} 可实现</text>
          <path class="capacity-rule" d="M 48 181 H 210" />
          <text class="status" :class="patternCount === 2 ** pointCount ? 'pass' : 'stop'" x="48" y="216">
            {{ patternCount === 2 ** pointCount ? 'SHATTERED' : 'CAPACITY STOPS' }}
          </text>
          <text class="detail" x="48" y="246">二维仿射半空间</text>
          <text class="detail" x="48" y="266">VC(H) = 3</text>
          <text class="detail" x="48" y="300">当前标记：</text>
          <text class="detail" :class="separator ? 'pass' : 'stop'" x="48" y="320">{{ separator ? '存在分隔直线' : '不存在分隔直线' }}</text>

          <rect class="field" x="250" y="42" width="338" height="308" rx="3" />
          <g class="grid" aria-hidden="true">
            <line v-for="x in [310, 390, 470]" :key="`x-${x}`" :x1="x" y1="58" :x2="x" y2="332" />
            <line v-for="y in [105, 205, 305]" :key="`y-${y}`" x1="270" :y1="y" x2="568" :y2="y" />
          </g>
          <line
            v-if="lineEnds.length === 2"
            class="separator"
            :x1="sx(lineEnds[0].x)"
            :y1="sy(lineEnds[0].y)"
            :x2="sx(lineEnds[1].x)"
            :y2="sy(lineEnds[1].y)"
          />
          <path v-if="!separator && pointCount === 4" class="xor-cross" d="M 300 120 L 520 290 M 520 120 L 300 290" />
          <g
            v-for="(point, index) in points"
            :key="point.name"
            class="sample"
            role="button"
            tabindex="0"
            :aria-label="`切换样本 ${point.name} 的标签`"
            :transform="`translate(${sx(point.x)} ${sy(point.y)})`"
            @click="toggleLabel(index)"
            @keydown.enter.prevent="toggleLabel(index)"
            @keydown.space.prevent="toggleLabel(index)"
          >
            <circle :class="labels[index] ? 'positive' : 'negative'" r="16" />
            <text y="5" text-anchor="middle">{{ labels[index] ? '+' : '−' }}</text>
            <text class="sample-name" y="31" text-anchor="middle">{{ point.name }}</text>
          </g>
          <text class="field-note" x="568" y="342" text-anchor="end">点击样本切换 ± 标签</text>
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="mode" aria-label="样本点数量">
        <button type="button" :aria-pressed="pointCount === 3" @click="setPointCount(3)">三个非共线点</button>
        <button type="button" :aria-pressed="pointCount === 4" @click="setPointCount(4)">四个凸四边形点</button>
      </div>
      <div class="label-grid" aria-label="切换各点标签">
        <button v-for="(point, index) in points" :key="point.name" type="button" :class="labels[index] ? 'label-positive' : 'label-negative'" @click="toggleLabel(index)">{{ point.name }}={{ labels[index] ? '+' : '−' }}</button>
      </div>
      <button type="button" class="xor" @click="setXor">装载四点 XOR 标记</button>
    </template>

    <template #metrics>
      <div><span>点数 m</span><strong>{{ pointCount }}</strong></div>
      <div><span>增长函数 ΠH(m)</span><strong>{{ patternCount }}</strong></div>
      <div><span>全部标记数</span><strong>{{ 2 ** pointCount }}</strong></div>
      <div><span>当前标记</span><strong>{{ separator ? '可实现' : '不可实现' }}</strong></div>
    </template>

    <template #note>三点时 8 种标记全部可实现；四个凸四边形点只有 14 种可实现，交替 XOR 及其补集被排除。VC 维要求“存在一组 d 点可被全部标记打散”，同时还要证明“不存在任何 d+1 点能被打散”。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.comparator-scroll{overflow-x:auto}.comparator-scroll svg{display:block;width:100%;height:auto}.housing{fill:#0c141a;stroke:#d8b75c;stroke-width:2}.pattern-card{fill:#eee9d8;stroke:#7468b4}.field{fill:#111f28;stroke:#465d67}.caption,.pattern-total,.status,.detail,.field-note,.sample-name{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.caption{fill:#596368;font-size:9px;font-weight:850;letter-spacing:.08em}.pattern-number{fill:#1e3038;font:800 62px/1 Georgia,ui-serif,serif}.pattern-total{fill:#5d686b;font-size:11px}.capacity-rule{stroke:#a99a76}.status{font-size:12px;font-weight:900;letter-spacing:.08em}.detail{fill:#4b5b61;font-size:10px;font-weight:700}.pass{fill:#278f86}.stop{fill:#c84c49}.grid line{stroke:#d9ece7;stroke-width:.7;opacity:.11}.separator{stroke:#d8b75c;stroke-width:4;stroke-dasharray:10 6}.xor-cross{stroke:#d65a56;stroke-width:2;stroke-dasharray:7 7;opacity:.45}.sample{cursor:pointer}.sample circle{stroke-width:3}.positive{fill:#67c8bd;stroke:#c8fff7}.negative{fill:#7468b4;stroke:#ded8ff}.sample text{fill:#0b171d;font:900 15px/1 Georgia,ui-serif,serif}.sample .sample-name{fill:#dfe9e8;font-size:9px}.sample:focus-visible{outline:3px solid #d8b75c;outline-offset:4px}.field-note{fill:#9fb0b4;font-size:8px}.mode{display:grid;grid-template-columns:1fr 1fr;gap:6px}.label-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}button{min-height:44px;padding:9px;border:1px solid #536a74;border-radius:4px;color:#dbe6e8;background:#14252e;font:720 10px/1.3 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#d8b75c;color:#ffe9ad;background:#4b3d21}.label-positive{border-color:#67c8bd;color:#c8fff7}.label-negative{border-color:#7468b4;color:#ded8ff}.xor{border-color:#d65a56;color:#ffd3d0;background:#3d2021}button:focus-visible{outline:3px solid #d8b75c;outline-offset:2px}@media(max-width:560px){.comparator-scroll{margin-right:-18px;padding-right:18px}.comparator-scroll svg{width:690px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
