<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Point = { x: number; y: number }

const probeX = ref(0.2)
const probeY = ref(0.4)
const centerBY = ref(0.8)
const swapped = ref(false)

const baseA = computed<Point>(() => ({ x: -1.55, y: -0.55 }))
const baseB = computed<Point>(() => ({ x: 1.45, y: centerBY.value }))
const centers = computed(() => swapped.value ? [baseB.value, baseA.value] : [baseA.value, baseB.value])
const probe = computed<Point>(() => ({ x: probeX.value, y: probeY.value }))

const distanceRows = computed(() => centers.value.map((center, index) => ({
  index,
  value: (probe.value.x - center.x) ** 2 + (probe.value.y - center.y) ** 2,
})))
const gap = computed(() => distanceRows.value[0].value - distanceRows.value[1].value)
const winner = computed(() => Math.abs(gap.value) < 1e-9 ? 0 : (gap.value < 0 ? 0 : 1))
const isTie = computed(() => Math.abs(gap.value) < 1e-9)

const tiles = computed(() => {
  const result: Array<{ x: number; y: number; w: number; h: number; owner: number }> = []
  const cols = 24
  const rows = 14
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const x = -3.5 + (col + 0.5) * 7 / cols
      const y = -2.5 + (row + 0.5) * 5 / rows
      const d0 = (x - centers.value[0].x) ** 2 + (y - centers.value[0].y) ** 2
      const d1 = (x - centers.value[1].x) ** 2 + (y - centers.value[1].y) ** 2
      result.push({
        x: 40 + col * 480 / cols,
        y: 40 + (rows - row - 1) * 280 / rows,
        w: 480 / cols + 0.25,
        h: 280 / rows + 0.25,
        owner: d0 <= d1 ? 0 : 1,
      })
    }
  }
  return result
})

const boundary = computed(() => {
  const [aPoint, bPoint] = centers.value
  const a = 2 * (bPoint.x - aPoint.x)
  const b = 2 * (bPoint.y - aPoint.y)
  const c = aPoint.x ** 2 + aPoint.y ** 2 - bPoint.x ** 2 - bPoint.y ** 2
  const candidates: Point[] = []
  for (const x of [-3.5, 3.5]) {
    if (Math.abs(b) > 1e-12) {
      const y = -(a * x + c) / b
      if (y >= -2.5 && y <= 2.5) candidates.push({ x, y })
    }
  }
  for (const y of [-2.5, 2.5]) {
    if (Math.abs(a) > 1e-12) {
      const x = -(b * y + c) / a
      if (x >= -3.5 && x <= 3.5) candidates.push({ x, y })
    }
  }
  const unique = candidates.filter((point, index) => candidates.findIndex(other =>
    Math.abs(other.x - point.x) < 1e-8 && Math.abs(other.y - point.y) < 1e-8) === index)
  return unique.slice(0, 2)
})

function sx(x: number) { return 40 + (x + 3.5) / 7 * 480 }
function sy(y: number) { return 320 - (y + 2.5) / 5 * 280 }

function snapProbeToBoundary() {
  const [aPoint, bPoint] = centers.value
  const a = 2 * (bPoint.x - aPoint.x)
  const b = 2 * (bPoint.y - aPoint.y)
  const c = aPoint.x ** 2 + aPoint.y ** 2 - bPoint.x ** 2 - bPoint.y ** 2
  const residual = a * probeX.value + b * probeY.value + c
  const denominator = a * a + b * b
  probeX.value -= residual * a / denominator
  probeY.value -= residual * b / denominator
}
</script>

<template>
  <MlExperimentFrame
    index="26-A"
    title="Voronoi 分流台"
    question="最近质心的判定边界为什么一定是一条直线？平票时又是谁在替数学做决定？"
    formula="d₁² − d₂² = 2xᵀ(μ₂−μ₁) + ‖μ₁‖² − ‖μ₂‖²"
  >
    <template #visual>
      <div class="switchyard-scroll">
        <svg viewBox="0 0 560 360" role="img" aria-label="两个质心形成的 Voronoi 分流区域与可移动探针">
          <rect class="deck" x="20" y="20" width="520" height="320" rx="8" />
          <g aria-hidden="true">
            <rect
              v-for="(tile, index) in tiles"
              :key="index"
              :x="tile.x"
              :y="tile.y"
              :width="tile.w"
              :height="tile.h"
              :class="tile.owner === 0 ? 'cell-a' : 'cell-b'"
            />
          </g>
          <g class="grid" aria-hidden="true">
            <line v-for="x in [120, 200, 280, 360, 440]" :key="`x-${x}`" :x1="x" y1="40" :x2="x" y2="320" />
            <line v-for="y in [96, 152, 208, 264]" :key="`y-${y}`" x1="40" :y1="y" x2="520" :y2="y" />
          </g>
          <line
            v-if="boundary.length === 2"
            class="boundary"
            :x1="sx(boundary[0].x)"
            :y1="sy(boundary[0].y)"
            :x2="sx(boundary[1].x)"
            :y2="sy(boundary[1].y)"
          />
          <line class="probe-wire" :x1="sx(probe.x)" :y1="sy(probe.y)" :x2="sx(centers[winner].x)" :y2="sy(centers[winner].y)" />
          <g v-for="(center, index) in centers" :key="index" :transform="`translate(${sx(center.x)} ${sy(center.y)})`">
            <path :class="index === 0 ? 'hopper-a' : 'hopper-b'" d="M -13 -12 H 13 L 8 8 H -8 Z" />
            <text y="27" text-anchor="middle">μ{{ index + 1 }}</text>
          </g>
          <g :transform="`translate(${sx(probe.x)} ${sy(probe.y)})`">
            <circle class="probe-ring" r="11" />
            <circle :class="winner === 0 ? 'probe-a' : 'probe-b'" r="5" />
          </g>
          <text class="bay-label" x="46" y="32">FOUNDRY SORTING FLOOR · SQUARED DISTANCE ROUTING</text>
          <text class="winner-label" x="514" y="335" text-anchor="end">
            {{ isTie ? `平票 → 按索引进入 μ${winner + 1}` : `探针进入 μ${winner + 1}` }}
          </text>
        </svg>
      </div>
    </template>

    <template #controls>
      <label class="control">
        <span>探针 x <output>{{ probeX.toFixed(2) }}</output></span>
        <input v-model.number="probeX" type="range" min="-3" max="3" step="0.05" aria-label="探针横坐标">
      </label>
      <label class="control">
        <span>探针 y <output>{{ probeY.toFixed(2) }}</output></span>
        <input v-model.number="probeY" type="range" min="-2" max="2" step="0.05" aria-label="探针纵坐标">
      </label>
      <label class="control">
        <span>第二质心 y <output>{{ centerBY.toFixed(2) }}</output></span>
        <input v-model.number="centerBY" type="range" min="-1.8" max="1.8" step="0.05" aria-label="第二质心纵坐标">
      </label>
      <button type="button" class="snap" @click="snapProbeToBoundary">把探针吸附到精确边界</button>
      <button type="button" :aria-pressed="swapped" @click="swapped = !swapped">交换质心编号</button>
    </template>

    <template #metrics>
      <div><span>到 μ1 的平方距离</span><strong>{{ distanceRows[0].value.toFixed(3) }}</strong></div>
      <div><span>到 μ2 的平方距离</span><strong>{{ distanceRows[1].value.toFixed(3) }}</strong></div>
      <div><span>d₁² − d₂²</span><strong>{{ gap.toExponential(2) }}</strong></div>
      <div><span>确定性分配</span><strong>μ{{ winner + 1 }}{{ isTie ? '（平票）' : '' }}</strong></div>
    </template>

    <template #note>橙/蓝底色是两个凸 Voronoi 单元。把探针吸到边界后，两项平方距离相等；此时目标函数没有偏好，最小索引、保留旧标签等规则才决定结果。交换编号不改变几何，却可能改变平票标签。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.switchyard-scroll{overflow-x:auto}.switchyard-scroll svg{display:block;width:100%;height:auto}.deck{fill:#0c151a}.cell-a{fill:#422617}.cell-b{fill:#14294b}.grid line{stroke:#d9e1d5;stroke-width:.7;opacity:.12}.boundary{stroke:#dbe5d5;stroke-width:3;stroke-dasharray:8 6}.probe-wire{stroke:#a8d85c;stroke-width:2;stroke-dasharray:4 4}.hopper-a{fill:#f28c42;stroke:#ffd2a8;stroke-width:2}.hopper-b{fill:#5b8cff;stroke:#c7d8ff;stroke-width:2}.probe-ring{fill:#0c151a;stroke:#a8d85c;stroke-width:3}.probe-a{fill:#f28c42}.probe-b{fill:#5b8cff}svg text{fill:#eef3e5;font:750 10px/1 ui-monospace,SFMono-Regular,Menlo,monospace}.bay-label{fill:#98a99e;font-size:9px;letter-spacing:.09em}.winner-label{fill:#a8d85c}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#c2cec5;font-size:11px}.control output{color:#f5aa6f;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#f28c42}button{min-height:44px;border:1px solid #52645a;border-radius:6px;color:#e7eee4;background:#17231d;font:720 11px/1.3 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#5b8cff;color:#c7d8ff;background:#172744}.snap{border-color:#9abb59;color:#d9f4a4;background:#243019}button:focus-visible,input:focus-visible{outline:3px solid #f28c42;outline-offset:2px}@media(max-width:560px){.switchyard-scroll{margin-right:-18px;padding-right:18px}.switchyard-scroll svg{width:620px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
