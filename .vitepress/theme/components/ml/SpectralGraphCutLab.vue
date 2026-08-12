<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Point = { x: number; y: number; truth: number }
type Edge = { from: number; to: number; weight: number }

const method = ref<'kmeans' | 'spectral'>('spectral')
const neighbors = ref(8)
const sigma = ref(0.25)
const showEdges = ref(true)

const points: Point[] = []
const half = 36
for (let mode = 0; mode < 2; mode += 1) {
  for (let index = 0; index < half; index += 1) {
    const theta = Math.PI * index / (half - 1)
    const dx = 0.035 * Math.sin((index + 31 * mode) * 12.9898)
    const dy = 0.04 * Math.cos((index + 17 * mode) * 7.123)
    points.push({
      x: mode === 0 ? Math.cos(theta) + dx : 1 - Math.cos(theta) + dx,
      y: mode === 0 ? Math.sin(theta) + dy : 0.48 - Math.sin(theta) + dy,
      truth: mode,
    })
  }
}

function squaredDistance(left: Point, right: Point) {
  return (left.x - right.x) ** 2 + (left.y - right.y) ** 2
}

function kmeans2d() {
  let centers = [{ x: -0.7, y: 0 }, { x: 1.7, y: 0 }]
  let labels = points.map(() => 0)
  for (let iteration = 0; iteration < 50; iteration += 1) {
    labels = points.map(point => squaredDistance(point, centers[0] as Point) <= squaredDistance(point, centers[1] as Point) ? 0 : 1)
    const next = centers.map((_, cluster) => {
      const members = points.filter((__, index) => labels[index] === cluster)
      return {
        x: members.reduce((sum, point) => sum + point.x, 0) / members.length,
        y: members.reduce((sum, point) => sum + point.y, 0) / members.length,
      }
    })
    if (next.every((center, index) => Math.hypot(center.x - centers[index].x, center.y - centers[index].y) < 1e-10)) break
    centers = next
  }
  return labels
}

const kmeansLabels = kmeans2d()

function orthogonalize(vector: number[], leading: number[]) {
  const dot = vector.reduce((sum, value, index) => sum + value * leading[index], 0)
  const residual = vector.map((value, index) => value - dot * leading[index])
  const norm = Math.hypot(...residual)
  return residual.map(value => value / Math.max(norm, 1e-12))
}

function kmeans1d(values: number[]) {
  let centers = [Math.min(...values), Math.max(...values)]
  let labels = values.map(() => 0)
  for (let iteration = 0; iteration < 50; iteration += 1) {
    labels = values.map(value => Math.abs(value - centers[0]) <= Math.abs(value - centers[1]) ? 0 : 1)
    const next = centers.map((_, cluster) => {
      const members = values.filter((__, index) => labels[index] === cluster)
      return members.reduce((sum, value) => sum + value, 0) / members.length
    })
    if (next.every((value, index) => Math.abs(value - centers[index]) < 1e-12)) break
    centers = next
  }
  return labels
}

const spectral = computed(() => {
  const n = points.length
  const distances = Array.from({ length: n }, () => Array(n).fill(0))
  for (let left = 0; left < n; left += 1) {
    for (let right = left + 1; right < n; right += 1) {
      distances[left][right] = distances[right][left] = squaredDistance(points[left], points[right])
    }
  }

  const weights = Array.from({ length: n }, () => Array(n).fill(0))
  for (let left = 0; left < n; left += 1) {
    const nearest = Array.from({ length: n }, (_, right) => right)
      .filter(right => right !== left)
      .sort((a, b) => distances[left][a] - distances[left][b])
      .slice(0, neighbors.value)
    nearest.forEach(right => {
      weights[left][right] = Math.exp(-distances[left][right] / (2 * sigma.value ** 2))
    })
  }
  for (let left = 0; left < n; left += 1) {
    for (let right = left + 1; right < n; right += 1) {
      weights[left][right] = weights[right][left] = Math.max(weights[left][right], weights[right][left])
    }
  }

  const degree = weights.map(row => row.reduce((sum, value) => sum + value, 0))
  let leading = degree.map(Math.sqrt)
  const leadingNorm = Math.hypot(...leading)
  leading = leading.map(value => value / Math.max(leadingNorm, 1e-12))
  let vector = orthogonalize(points.map((point, index) => point.y + 0.13 * Math.sin(index * 2.3)), leading)

  function multiplyNormalizedAdjacency(input: number[]) {
    return weights.map((row, left) => row.reduce((sum, weight, right) =>
      sum + weight * input[right] / Math.sqrt(Math.max(degree[left] * degree[right], 1e-24)), 0))
  }

  for (let iteration = 0; iteration < 600; iteration += 1) {
    const propagated = multiplyNormalizedAdjacency(vector)
    vector = orthogonalize(vector.map((value, index) => 0.5 * (value + propagated[index])), leading)
  }
  const propagated = multiplyNormalizedAdjacency(vector)
  const lambda2 = 1 - vector.reduce((sum, value, index) => sum + value * propagated[index], 0)
  const labels = kmeans1d(vector)
  const edges: Edge[] = []
  for (let left = 0; left < n; left += 1) {
    for (let right = left + 1; right < n; right += 1) {
      if (weights[left][right] > 0) edges.push({ from: left, to: right, weight: weights[left][right] })
    }
  }

  const parent = Array.from({ length: n }, (_, index) => index)
  function root(index: number): number {
    if (parent[index] !== index) parent[index] = root(parent[index])
    return parent[index]
  }
  edges.forEach(edge => {
    const leftRoot = root(edge.from)
    const rightRoot = root(edge.to)
    if (leftRoot !== rightRoot) parent[rightRoot] = leftRoot
  })
  const components = new Set(parent.map((_, index) => root(index))).size
  const crossEdges = edges.filter(edge => points[edge.from].truth !== points[edge.to].truth).length
  return { labels, vector, lambda2, edges, components, crossEdges }
})

function adjustedRand(labels: number[]) {
  const table = [[0, 0], [0, 0]]
  points.forEach((point, index) => { table[point.truth][labels[index]] += 1 })
  const choose2 = (value: number) => value * (value - 1) / 2
  const index = table.flat().reduce((sum, value) => sum + choose2(value), 0)
  const rowPairs = table.reduce((sum, row) => sum + choose2(row[0] + row[1]), 0)
  const columnPairs = choose2(table[0][0] + table[1][0]) + choose2(table[0][1] + table[1][1])
  const totalPairs = choose2(points.length)
  const expected = rowPairs * columnPairs / totalPairs
  const maximum = 0.5 * (rowPairs + columnPairs)
  return (index - expected) / (maximum - expected)
}

const activeLabels = computed(() => method.value === 'spectral' ? spectral.value.labels : kmeansLabels)
const activeAri = computed(() => adjustedRand(activeLabels.value))
const vectorRange = computed(() => {
  const values = spectral.value.vector
  return { minimum: Math.min(...values), maximum: Math.max(...values) }
})

function sx(value: number) { return 36 + (value + 1.15) / 3.3 * 300 }
function sy(value: number) { return 335 - (value + 1.10) / 2.25 * 270 }
function vx(value: number) {
  const range = vectorRange.value
  return 386 + (value - range.minimum) / Math.max(range.maximum - range.minimum, 1e-12) * 190
}
</script>

<template>
  <MlExperimentFrame
    index="27-C"
    title="谱图切割台"
    question="两条工况滞回轨迹在原空间互相环抱时，改在邻接图的低频坐标中分组，会发生什么？"
    formula="Lsym = I − D⁻¹ᐟ²WD⁻¹ᐟ² · cluster(rows(U₂))"
  >
    <template #visual>
      <div class="graph-scroll">
        <svg viewBox="0 0 620 390" role="img" aria-label="非凸双月牙数据的原空间 K-means 与谱图切割比较">
          <rect class="deck" x="12" y="12" width="596" height="366" rx="4" />
          <text class="caption" x="30" y="38">ORIGINAL PROCESS PLANE</text>
          <text class="caption" x="374" y="38">SECOND SPECTRAL COORDINATE</text>
          <rect class="plot" x="28" y="48" width="316" height="304" rx="3" />
          <rect class="spectrum" x="366" y="48" width="224" height="304" rx="3" />
          <g v-if="showEdges && method === 'spectral'" class="edges" aria-hidden="true">
            <line
              v-for="(edge, index) in spectral.edges"
              :key="index"
              :x1="sx(points[edge.from].x)"
              :y1="sy(points[edge.from].y)"
              :x2="sx(points[edge.to].x)"
              :y2="sy(points[edge.to].y)"
              :style="{ opacity: Math.max(0.035, edge.weight * 0.22) }"
            />
          </g>
          <circle
            v-for="(point, index) in points"
            :key="`p-${index}`"
            :class="activeLabels[index] === 0 ? 'cluster-a' : 'cluster-b'"
            :cx="sx(point.x)"
            :cy="sy(point.y)"
            r="4.2"
          />
          <line class="zero" :x1="vx(0)" y1="68" :x2="vx(0)" y2="326" />
          <circle
            v-for="(value, index) in spectral.vector"
            :key="`v-${index}`"
            :class="spectral.labels[index] === 0 ? 'cluster-a' : 'cluster-b'"
            :cx="vx(value)"
            :cy="78 + (index % 18) * 13.6"
            r="3.5"
          />
          <text class="axis" :x="vx(0)" y="342" text-anchor="middle">v₂ = 0</text>
          <text class="method-label" x="42" y="344">{{ method === 'spectral' ? '按浏览器实时谱坐标着色' : '按原空间质心着色' }}</text>
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="mode" aria-label="聚类方法">
        <button type="button" :aria-pressed="method === 'kmeans'" @click="method = 'kmeans'">原空间 K-means</button>
        <button type="button" :aria-pressed="method === 'spectral'" @click="method = 'spectral'">谱坐标切分</button>
      </div>
      <label class="control">
        <span>每点近邻数 <output>{{ neighbors }}</output></span>
        <input v-model.number="neighbors" type="range" min="2" max="20" step="1" aria-label="每个样本的近邻数">
      </label>
      <label class="control">
        <span>Gaussian 宽度 σ <output>{{ sigma.toFixed(2) }}</output></span>
        <input v-model.number="sigma" type="range" min="0.12" max="0.60" step="0.01" aria-label="图相似度的高斯宽度">
      </label>
      <button type="button" :aria-pressed="showEdges" @click="showEdges = !showEdges">{{ showEdges ? '隐藏邻接边' : '显示邻接边' }}</button>
    </template>

    <template #metrics>
      <div><span>当前方法审计 ARI</span><strong>{{ activeAri.toFixed(3) }}</strong></div>
      <div><span>图连通分量</span><strong>{{ spectral.components }}</strong></div>
      <div><span>跨隐藏模式边</span><strong>{{ spectral.crossEdges }} / {{ spectral.edges.length }}</strong></div>
      <div><span>λ₂(Lsym)</span><strong>{{ spectral.lambda2.toFixed(5) }}</strong></div>
    </template>

    <template #note>这里的第二谱方向由浏览器实时执行归一化邻接矩阵的幂迭代，再在一维谱坐标上做两中心划分。近邻过多会制造跨轨迹捷径；σ 很小时又会把这些弱边近似忽略。ARI 使用合成器保存的隐藏模式，只用于教学压力测试，不是现实生产中的免费真值。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.graph-scroll{overflow-x:auto}.graph-scroll svg{display:block;width:100%;height:auto}.deck{fill:#10283d;stroke:#c59b48;stroke-width:2}.plot,.spectrum{fill:#0a1b2a;stroke:#657b8c}.caption,.axis,.method-label{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.caption{fill:#d4bb7c;font-size:9px;font-weight:850;letter-spacing:.08em}.edges line{stroke:#91a9b8;stroke-width:1}.cluster-a{fill:#2aa198;stroke:#bdf7ed;stroke-width:.6}.cluster-b{fill:#c94f3d;stroke:#ffd0c9;stroke-width:.6}.zero{stroke:#c59b48;stroke-width:2;stroke-dasharray:7 5}.axis{fill:#d4bb7c;font-size:9px}.method-label{fill:#dbe7eb;font-size:9px;font-weight:750}.mode{display:grid;grid-template-columns:1fr 1fr;gap:6px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#c2cee9;font-size:11px}.control output{color:#f1ce79;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#2aa198}button{min-height:44px;padding:9px;border:1px solid #526a7d;border-radius:4px;color:#d8e4eb;background:#122c40;font:720 10px/1.3 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#c59b48;color:#ffe8a9;background:#493b21}button:focus-visible,input:focus-visible{outline:3px solid #f0c75e;outline-offset:2px}@media(max-width:560px){.graph-scroll{margin-right:-18px;padding-right:18px}.graph-scroll svg{width:690px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
