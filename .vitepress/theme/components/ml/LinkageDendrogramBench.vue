<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Linkage = 'single' | 'complete' | 'average'
type Merge = { id: number; left: number; right: number; height: number; members: number[] }
type DendrogramSegment = { x1: number; y1: number; x2: number; y2: number; mergeIndex: number }

const linkage = ref<Linkage>('single')
const bridgeX = ref(1.5)
const step = ref(0)
const names = ['A', 'B', 'C', 'D', 'E', 'F']
const linkageModes: Linkage[] = ['single', 'complete', 'average']

const points = computed(() => [0, 0.7, bridgeX.value, 3.1, 4.0, 6.3])

function clusterDistance(left: number[], right: number[], mode: Linkage) {
  const values = left.flatMap(a => right.map(b => Math.abs(points.value[a] - points.value[b])))
  if (mode === 'single') return Math.min(...values)
  if (mode === 'complete') return Math.max(...values)
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

const history = computed<Merge[]>(() => {
  const clusters = new Map<number, number[]>(names.map((_, index) => [index, [index]]))
  const merges: Merge[] = []
  let nextId = names.length
  while (clusters.size > 1) {
    const ids = [...clusters.keys()].sort((a, b) => a - b)
    let best: { distance: number; left: number; right: number } | undefined
    ids.forEach((left, leftIndex) => {
      ids.slice(leftIndex + 1).forEach(right => {
        const distance = clusterDistance(clusters.get(left)!, clusters.get(right)!, linkage.value)
        const candidate = { distance, left, right }
        if (!best || distance < best.distance - 1e-10 ||
          (Math.abs(distance - best.distance) < 1e-10 && `${left}-${right}` < `${best.left}-${best.right}`)) {
          best = candidate
        }
      })
    })
    if (!best) break
    const members = [...clusters.get(best.left)!, ...clusters.get(best.right)!].sort((a, b) => a - b)
    clusters.delete(best.left)
    clusters.delete(best.right)
    clusters.set(nextId, members)
    merges.push({ id: nextId, left: best.left, right: best.right, height: best.distance, members })
    nextId += 1
  }
  return merges
})

watch([linkage, bridgeX], () => { step.value = 0 })

const currentClusters = computed(() => {
  const clusters = new Map<number, number[]>(names.map((_, index) => [index, [index]]))
  history.value.slice(0, step.value).forEach(merge => {
    clusters.delete(merge.left)
    clusters.delete(merge.right)
    clusters.set(merge.id, merge.members)
  })
  return [...clusters.values()].sort((a, b) => a[0] - b[0])
})

function memberLabel(members: number[]) {
  return members.map(index => names[index]).join('')
}

const nextMerge = computed(() => history.value[step.value])
const maxHeight = computed(() => Math.max(1, history.value.at(-1)?.height ?? 1))
function leafX(index: number) { return 60 + index * 100 }
function mergeY(height: number) { return 320 - height / maxHeight.value * 235 }

const dendrogram = computed(() => {
  const positions = new Map<number, { x: number; y: number }>()
  names.forEach((_, index) => positions.set(index, { x: leafX(index), y: 320 }))
  const segments: DendrogramSegment[] = []
  history.value.forEach((merge, mergeIndex) => {
    const left = positions.get(merge.left)!
    const right = positions.get(merge.right)!
    const y = mergeY(merge.height)
    segments.push(
      { x1: left.x, y1: left.y, x2: left.x, y2: y, mergeIndex },
      { x1: right.x, y1: right.y, x2: right.x, y2: y, mergeIndex },
      { x1: left.x, y1: y, x2: right.x, y2: y, mergeIndex },
    )
    positions.set(merge.id, { x: (left.x + right.x) / 2, y })
  })
  return segments
})

function setLinkage(value: Linkage) {
  linkage.value = value
}
</script>

<template>
  <MlExperimentFrame
    index="27-B"
    title="Linkage 合并量规"
    question="同一组点，为什么只改“两个簇之间怎样量距离”，树的合并高度和最终切法就会改变？"
    formula="single=min dᵢⱼ · complete=max dᵢⱼ · average=mean dᵢⱼ"
  >
    <template #visual>
      <div class="dendrogram-scroll">
        <svg viewBox="0 0 620 390" role="img" aria-label="六个样本在三种 linkage 下的层次聚类树">
          <rect class="blueprint" x="12" y="12" width="596" height="366" rx="4" />
          <g class="grid" aria-hidden="true">
            <line v-for="y in [85, 132, 179, 226, 273, 320]" :key="y" x1="36" :y1="y" x2="584" :y2="y" />
          </g>
          <text class="caption" x="34" y="38">MERGE HEIGHT</text>
          <text class="caption" x="584" y="38" text-anchor="end">{{ linkage.toUpperCase() }} LINKAGE · STEP {{ step }}/5</text>
          <line
            v-for="(segment, index) in dendrogram"
            :key="index"
            :class="segment.mergeIndex < step ? 'branch active' : segment.mergeIndex === step ? 'branch next' : 'branch future'"
            :x1="segment.x1"
            :y1="segment.y1"
            :x2="segment.x2"
            :y2="segment.y2"
          />
          <g v-for="(name, index) in names" :key="name" :transform="`translate(${leafX(index)} 320)`">
            <circle class="leaf" r="10" />
            <text class="leaf-name" y="4" text-anchor="middle">{{ name }}</text>
            <text class="coordinate" y="26" text-anchor="middle">{{ points[index].toFixed(1) }}</text>
          </g>
          <g class="partition" transform="translate(36 352)">
            <g v-for="(cluster, index) in currentClusters" :key="memberLabel(cluster)" :transform="`translate(${index * 94} 0)`">
              <rect x="0" y="-13" width="84" height="24" rx="3" />
              <text x="42" y="3" text-anchor="middle">{{ memberLabel(cluster) }}</text>
            </g>
          </g>
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="mode" aria-label="linkage 规则">
        <button v-for="mode in linkageModes" :key="mode" type="button" :aria-pressed="linkage === mode" @click="setLinkage(mode)">{{ mode }}</button>
      </div>
      <label class="control">
        <span>桥接样本 C 的位置 <output>{{ bridgeX.toFixed(1) }}</output></span>
        <input v-model.number="bridgeX" type="range" min="1.1" max="2.7" step="0.1" aria-label="桥接样本 C 的坐标">
      </label>
      <label class="control">
        <span>播放到第几次合并 <output>{{ step }}/5</output></span>
        <input v-model.number="step" type="range" min="0" max="5" step="1" aria-label="层次聚类合并步数">
      </label>
      <div class="transport">
        <button type="button" :disabled="step === 0" @click="step -= 1">上一步</button>
        <button type="button" :disabled="step === 5" @click="step += 1">执行下一次合并</button>
      </div>
    </template>

    <template #metrics>
      <div><span>当前簇数</span><strong>{{ currentClusters.length }}</strong></div>
      <div><span>下一次合并</span><strong>{{ nextMerge ? memberLabel(nextMerge.members) : '树已完成' }}</strong></div>
      <div><span>下一高度</span><strong>{{ nextMerge ? nextMerge.height.toFixed(3) : '—' }}</strong></div>
      <div><span>整棵树最高处</span><strong>{{ maxHeight.toFixed(3) }}</strong></div>
    </template>

    <template #note>横向叶序只是画图排版，不代表额外距离；真正有意义的是纵向合并高度。默认点位下，single linkage 会让 C 充当链条，把相邻簇逐段串起；complete linkage 按最远点计价，直到更高位置才允许两个宽簇合并。拖动 C，观察“桥接点”怎样改写整棵树。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.dendrogram-scroll{overflow-x:auto}.dendrogram-scroll svg{display:block;width:100%;height:auto}.blueprint{fill:#10283d;stroke:#c59b48;stroke-width:2}.grid line{stroke:#d8dfd3;stroke-width:.7;opacity:.12}.caption{fill:#d4bb7c;font:800 9px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.09em}.branch{stroke-width:3;fill:none}.branch.active{stroke:#2aa198}.branch.next{stroke:#c94f3d;stroke-dasharray:7 5}.branch.future{stroke:#718294;stroke-width:2;opacity:.32}.leaf{fill:#e9e3d3;stroke:#c59b48;stroke-width:2}.leaf-name{fill:#11283a;font:900 11px/1 Georgia,ui-serif,serif}.coordinate{fill:#bfcad1;font:700 9px/1 ui-monospace,monospace}.partition rect{fill:#e9e3d3;stroke:#c59b48}.partition text{fill:#183047;font:850 10px/1 ui-monospace,monospace}.mode,.transport{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}.transport{grid-template-columns:1fr 1.45fr}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#c2cee9;font-size:11px}.control output{color:#f1ce79;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#c94f3d}button{min-height:44px;padding:8px;border:1px solid #526a7d;border-radius:4px;color:#d8e4eb;background:#122c40;font:720 10px/1.25 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#c59b48;color:#ffe8a9;background:#493b21}button:disabled{cursor:not-allowed;opacity:.42}button:focus-visible,input:focus-visible{outline:3px solid #f0c75e;outline-offset:2px}@media(max-width:560px){.dendrogram-scroll{margin-right:-18px;padding-right:18px}.dendrogram-scroll svg{width:680px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
