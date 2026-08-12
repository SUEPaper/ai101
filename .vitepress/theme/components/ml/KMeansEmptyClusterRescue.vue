<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Point = { x: number; y: number }
type Policy = 'stale' | 'farthest' | 'split' | 'restart'

const points: Point[] = [
  { x: -2.8, y: -0.4 }, { x: -2.6, y: 0.4 }, { x: -2.2, y: -0.8 },
  { x: -2.0, y: 0.2 }, { x: -1.7, y: 0.7 },
  { x: -0.2, y: -0.6 }, { x: 0.0, y: 0.2 }, { x: 0.3, y: 0.8 },
  { x: 0.6, y: -0.2 }, { x: 1.8, y: -0.5 }, { x: 2.1, y: 0.3 },
  { x: 2.4, y: -0.8 }, { x: 2.7, y: 0.5 },
]
const initialCenters: Point[] = [
  { x: -2.35, y: 0.0 },
  { x: -2.35, y: 0.0 },
  { x: 2.25, y: -0.1 },
]
const palette = ['#f28c42', '#5b8cff', '#a8d85c']
const policy = ref<Policy>('stale')

function d2(a: Point, b: Point) {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2
}

function assign(centers: Point[]) {
  return points.map(point => {
    const distances = centers.map(center => d2(point, center))
    return distances.indexOf(Math.min(...distances))
  })
}

function mean(members: Point[], fallback: Point) {
  if (!members.length) return { ...fallback }
  return {
    x: members.reduce((sum, point) => sum + point.x, 0) / members.length,
    y: members.reduce((sum, point) => sum + point.y, 0) / members.length,
  }
}

const result = computed(() => {
  let centers = initialCenters.map(center => ({ ...center }))
  let labels = assign(centers)
  const initialCounts = centers.map((_, cluster) => labels.filter(label => label === cluster).length)
  const initialObjective = points.reduce((sum, point, index) => sum + d2(point, centers[labels[index]]), 0)
  let action = '空簇 μ2 保留旧坐标，问题仍未解除'

  if (policy.value !== 'stale') {
    if (policy.value === 'farthest') {
      const residuals = points.map((point, index) => d2(point, centers[labels[index]]))
      const candidate = residuals.indexOf(Math.max(...residuals))
      centers[1] = { ...points[candidate] }
      action = `把最大残差样本 #${candidate + 1} 设为 μ2`
    } else if (policy.value === 'split') {
      const largest = initialCounts.indexOf(Math.max(...initialCounts))
      const members = points.map((point, index) => ({ point, index })).filter(item => labels[item.index] === largest)
      const donorMean = mean(members.map(item => item.point), centers[largest])
      const candidate = members.reduce((best, item) => d2(item.point, donorMean) > d2(best.point, donorMean) ? item : best)
      centers[largest] = donorMean
      centers[1] = { ...candidate.point }
      action = `拆分最大簇 μ${largest + 1}，远端样本 #${candidate.index + 1} 接管 μ2`
    } else {
      centers[1] = { ...points[7] }
      action = '按记录种子重启 μ2 到候选样本 #8'
    }
    labels = assign(centers)
  }

  centers = centers.map((center, cluster) => mean(points.filter((_, index) => labels[index] === cluster), center))
  labels = assign(centers)
  const counts = centers.map((_, cluster) => labels.filter(label => label === cluster).length)
  const objective = points.reduce((sum, point, index) => sum + d2(point, centers[labels[index]]), 0)
  return { centers, labels, counts, initialCounts, initialObjective, objective, action }
})

function sx(x: number) { return 46 + (x + 3.2) / 6.4 * 468 }
function sy(y: number) { return 292 - (y + 1.6) / 3.2 * 220 }
</script>

<template>
  <MlExperimentFrame
    index="26-C"
    title="空簇救援台"
    question="两个初始质心重合时，确定性平票会制造空簇；不同救援策略改变了什么合同？"
    formula="nₖ = 0 ⇒ μₖ = (1/nₖ) Σᵢ:zᵢ=k xᵢ 未定义"
  >
    <template #visual>
      <div class="rescue-scroll">
        <svg viewBox="0 0 560 350" role="img" aria-label="重合质心导致空簇以及三种救援策略">
          <rect class="deck" x="20" y="20" width="520" height="310" rx="8" />
          <g class="grid" aria-hidden="true">
            <line v-for="x in [124, 202, 280, 358, 436]" :key="`x-${x}`" :x1="x" y1="72" :x2="x" y2="292" />
            <line v-for="y in [116, 160, 204, 248]" :key="`y-${y}`" x1="46" :y1="y" x2="514" :y2="y" />
          </g>
          <line
            v-for="(point, index) in points"
            :key="`wire-${index}`"
            :x1="sx(point.x)"
            :y1="sy(point.y)"
            :x2="sx(result.centers[result.labels[index]].x)"
            :y2="sy(result.centers[result.labels[index]].y)"
            :stroke="palette[result.labels[index]]"
            class="wire"
          />
          <circle
            v-for="(point, index) in points"
            :key="index"
            :cx="sx(point.x)"
            :cy="sy(point.y)"
            r="7"
            :fill="palette[result.labels[index]]"
            class="sample"
          />
          <g v-for="(center, cluster) in result.centers" :key="`center-${cluster}`" :transform="`translate(${sx(center.x)} ${sy(center.y)})`">
            <circle :stroke="palette[cluster]" r="16" class="center-ring" />
            <path :stroke="palette[cluster]" d="M -11 0 H 11 M 0 -11 V 11" />
            <text :y="cluster === 1 && policy === 'stale' ? -23 : 31" text-anchor="middle">μ{{ cluster + 1 }} · n={{ result.counts[cluster] }}</text>
          </g>
          <text class="bay-label" x="46" y="42">EMPTY-CLUSTER INCIDENT · POLICY MUST BE LOGGED</text>
          <g class="status" transform="translate(46 302)">
            <rect width="468" height="18" rx="3" :class="result.counts.includes(0) ? 'unsafe' : 'safe'" />
            <text x="9" y="13">{{ result.action }}</text>
          </g>
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="policy-grid" role="group" aria-label="空簇处理策略">
        <button type="button" :aria-pressed="policy === 'stale'" @click="policy = 'stale'">保留旧中心</button>
        <button type="button" :aria-pressed="policy === 'farthest'" @click="policy = 'farthest'">最大残差点接管</button>
        <button type="button" :aria-pressed="policy === 'split'" @click="policy = 'split'">拆分最大簇</button>
        <button type="button" :aria-pressed="policy === 'restart'" @click="policy = 'restart'">确定性重启</button>
      </div>
    </template>

    <template #metrics>
      <div><span>首次分配计数</span><strong>{{ result.initialCounts.join(' / ') }}</strong></div>
      <div><span>救援后计数</span><strong>{{ result.counts.join(' / ') }}</strong></div>
      <div><span>初始 WCSS</span><strong>{{ result.initialObjective.toFixed(3) }}</strong></div>
      <div><span>当前 WCSS / 状态</span><strong>{{ result.objective.toFixed(3) }} · {{ result.counts.includes(0) ? '未解除' : '已解除' }}</strong></div>
    </template>

    <template #note>“保留旧中心”虽然避免除零，却可能永久隐藏少用簇。其余策略都会改变标准 Lloyd 轨迹，因此必须记录触发轮次、供体样本和随机种子，并在救援后重新计算分配与目标；不能把干预前后的单调性混成一个证明。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.rescue-scroll{overflow-x:auto}.rescue-scroll svg{display:block;width:100%;height:auto}.deck{fill:#0b1418}.grid line{stroke:#d7dfd4;stroke-width:.8;opacity:.12}.wire{stroke-width:1.3;opacity:.35}.sample{stroke:#f5f5eb;stroke-width:1.4}.center-ring{fill:#0b1418;stroke-width:2.4}g>path{fill:none;stroke-width:3}.bay-label,svg text{fill:#edf3e7;font:740 9px/1 ui-monospace,SFMono-Regular,Menlo,monospace}.bay-label{fill:#95a59b;letter-spacing:.1em}.status rect.safe{fill:#607f2e}.status rect.unsafe{fill:#8e382d}.status text{fill:#fff;font-weight:850}.policy-grid{display:grid;gap:9px}button{min-height:44px;border:1px solid #53645a;border-radius:6px;color:#e7eee5;background:#17231d;font:720 11px/1.3 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#f28c42;color:#ffd6b7;background:#3d2618}button:focus-visible{outline:3px solid #a8d85c;outline-offset:2px}@media(max-width:560px){.rescue-scroll{margin-right:-18px;padding-right:18px}.rescue-scroll svg{width:620px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
