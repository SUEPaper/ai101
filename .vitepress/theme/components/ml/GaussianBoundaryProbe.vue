<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Matrix2 = [[number, number], [number, number]]

const modelMode = ref<'lda' | 'qda'>('qda')
const correlation = ref(.62)
const varianceRatio = ref(1.25)
const classOnePrior = ref(.36)
const queryX = ref(2.55)
const queryY = ref(1.45)
const means: [number, number][] = [[1.35, 1.02], [3.58, 1.92]]
const cov0: Matrix2 = [[.30, .08], [.08, .18]]

const cov1 = computed<Matrix2>(() => {
  const a = .28 * varianceRatio.value
  const d = .24 / varianceRatio.value
  const b = correlation.value * Math.sqrt(a * d)
  return [[a, b], [b, d]]
})
const sharedCov = computed<Matrix2>(() => {
  const p = classOnePrior.value
  return [
    [(1 - p) * cov0[0][0] + p * cov1.value[0][0], (1 - p) * cov0[0][1] + p * cov1.value[0][1]],
    [(1 - p) * cov0[1][0] + p * cov1.value[1][0], (1 - p) * cov0[1][1] + p * cov1.value[1][1]],
  ]
})

function score(point: [number, number], mean: [number, number], covariance: Matrix2, prior: number) {
  const [[a, b], [, d]] = covariance
  const det = a * d - b * b
  const dx = point[0] - mean[0]
  const dy = point[1] - mean[1]
  const mahalanobis = (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / det
  return Math.log(prior) - .5 * Math.log(det) - .5 * mahalanobis
}
function scores(point: [number, number]) {
  const c0 = modelMode.value === 'lda' ? sharedCov.value : cov0
  const c1 = modelMode.value === 'lda' ? sharedCov.value : cov1.value
  return [
    score(point, means[0], c0, 1 - classOnePrior.value),
    score(point, means[1], c1, classOnePrior.value),
  ]
}
function posterior(point: [number, number]) {
  const raw = scores(point)
  const maximum = Math.max(...raw)
  const weights = raw.map(value => Math.exp(value - maximum))
  return weights[1] / (weights[0] + weights[1])
}

const cells = computed(() => Array.from({ length: 25 * 15 }, (_, index) => {
  const column = index % 25
  const row = Math.floor(index / 25)
  const x = (column + .5) / 5
  const y = (row + .5) / 5
  const probability = posterior([x, y])
  return { column, row, probability }
}))
const queryProbability = computed(() => posterior([queryX.value, queryY.value]))
const queryClass = computed(() => queryProbability.value >= .5 ? 1 : 0)

function ellipse(covariance: Matrix2, mean: [number, number]) {
  const [[a, b], [, d]] = covariance
  const root = Math.sqrt((a - d) ** 2 + 4 * b * b)
  const l1 = Math.max((a + d + root) / 2, 1e-8)
  const l2 = Math.max((a + d - root) / 2, 1e-8)
  const angle = .5 * Math.atan2(2 * b, a - d) * 180 / Math.PI
  return { cx: 70 + mean[0] * 100, cy: 325 - mean[1] * 100, rx: 200 * Math.sqrt(l1), ry: 200 * Math.sqrt(l2), angle: -angle }
}
const ellipse0 = computed(() => ellipse(modelMode.value === 'lda' ? sharedCov.value : cov0, means[0]))
const ellipse1 = computed(() => ellipse(modelMode.value === 'lda' ? sharedCov.value : cov1.value, means[1]))
const queryScreenX = computed(() => 70 + queryX.value * 100)
const queryScreenY = computed(() => 325 - queryY.value * 100)
const boundaryLabel = computed(() => modelMode.value === 'lda' ? '一次分界 · 共享协方差' : '二次分界 · 类别专属协方差')
</script>

<template>
  <MlExperimentFrame
    index="23-A"
    title="让协方差直接决定分界形状"
    question="同一组类别中心，为什么共享协方差只产生直线，而类别专属协方差会把分界弯曲？移动探针并改变相关性，观察 posterior 与边界一起变化。"
    formula="δc(x)=log πc−½log|Σc|−½(x−μc)ᵀΣc⁻¹(x−μc)"
  >
    <template #visual>
      <div class="boundary-scroll">
        <svg viewBox="0 0 640 360" role="img" :aria-label="`${modelMode.toUpperCase()}分界；查询点判为${queryClass === 0 ? '转子不平衡' : '轴承预紧异常'}，类别一后验${(100 * queryProbability).toFixed(1)}%`">
          <rect width="640" height="360" rx="15" class="panel" />
          <text x="24" y="28" class="eyebrow">SPINDLE DIAGNOSTIC RACE · CLASS-CONDITIONAL GEOMETRY</text>
          <g class="decision-field">
            <rect
              v-for="cell in cells"
              :key="`${cell.column}-${cell.row}`"
              :x="70 + cell.column * 20"
              :y="25 + (14 - cell.row) * 20"
              width="20.5"
              height="20.5"
              :class="cell.probability >= .5 ? 'cell cell-one' : 'cell cell-zero'"
              :style="{ opacity: .16 + Math.abs(cell.probability - .5) * .56 }"
            />
          </g>
          <g class="grid">
            <line v-for="tick in [0,1,2,3,4,5]" :key="`x-${tick}`" :x1="70 + tick * 100" :x2="70 + tick * 100" y1="25" y2="325" />
            <line v-for="tick in [0,1,2,3]" :key="`y-${tick}`" x1="70" x2="570" :y1="325 - tick * 100" :y2="325 - tick * 100" />
          </g>
          <ellipse :cx="ellipse0.cx" :cy="ellipse0.cy" :rx="ellipse0.rx" :ry="ellipse0.ry" :transform="`rotate(${ellipse0.angle} ${ellipse0.cx} ${ellipse0.cy})`" class="race race-zero" />
          <ellipse :cx="ellipse1.cx" :cy="ellipse1.cy" :rx="ellipse1.rx" :ry="ellipse1.ry" :transform="`rotate(${ellipse1.angle} ${ellipse1.cx} ${ellipse1.cy})`" class="race race-one" />
          <circle :cx="70 + means[0][0] * 100" :cy="325 - means[0][1] * 100" r="6" class="mean mean-zero" />
          <circle :cx="70 + means[1][0] * 100" :cy="325 - means[1][1] * 100" r="6" class="mean mean-one" />
          <g class="probe">
            <circle :cx="queryScreenX" :cy="queryScreenY" r="10" />
            <line :x1="queryScreenX - 15" :x2="queryScreenX + 15" :y1="queryScreenY" :y2="queryScreenY" />
            <line :x1="queryScreenX" :x2="queryScreenX" :y1="queryScreenY - 15" :y2="queryScreenY + 15" />
          </g>
          <rect x="394" y="38" width="162" height="42" rx="5" class="readout" />
          <text x="475" y="54" text-anchor="middle" class="readout-label">{{ boundaryLabel }}</text>
          <text x="475" y="70" text-anchor="middle" class="readout-value">P(预紧异常 | x) = {{ (100 * queryProbability).toFixed(1) }}%</text>
          <text x="70" y="345" class="axis">同步振动证据 →</text>
          <text x="14" y="205" class="axis" transform="rotate(-90 14 205)">包络能量证据 →</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <button type="button" :class="{ active:modelMode === 'qda' }" :aria-pressed="modelMode === 'qda'" @click="modelMode = modelMode === 'lda' ? 'qda' : 'lda'">{{ modelMode === 'lda' ? '当前：LDA 共享协方差' : '当前：QDA 类别专属协方差' }}</button>
      <label class="control"><span>类别 1 相关系数 ρ <output>{{ correlation.toFixed(2) }}</output></span><input v-model.number="correlation" aria-label="类别一相关系数" type="range" min="-0.85" max="0.85" step="0.01"></label>
      <label class="control"><span>类别 1 方差方向比 <output>{{ varianceRatio.toFixed(2) }}</output></span><input v-model.number="varianceRatio" aria-label="类别一方差方向比" type="range" min="0.55" max="1.8" step="0.01"></label>
      <label class="control"><span>类别 1 先验 π₁ <output>{{ (100 * classOnePrior).toFixed(0) }}%</output></span><input v-model.number="classOnePrior" aria-label="类别一先验" type="range" min="0.12" max="0.72" step="0.01"></label>
      <label class="control"><span>探针 x₁ <output>{{ queryX.toFixed(2) }}</output></span><input v-model.number="queryX" aria-label="查询点横坐标" type="range" min="0.3" max="4.7" step="0.02"></label>
      <label class="control"><span>探针 x₂ <output>{{ queryY.toFixed(2) }}</output></span><input v-model.number="queryY" aria-label="查询点纵坐标" type="range" min="0.2" max="2.8" step="0.02"></label>
    </template>
    <template #metrics>
      <div><span>协方差合同</span><strong>{{ modelMode === 'lda' ? 'Σ₀ = Σ₁' : 'Σ₀ ≠ Σ₁' }}</strong></div>
      <div><span>分界阶数</span><strong>{{ modelMode === 'lda' ? '一次' : '至多二次' }}</strong></div>
      <div><span>探针分类</span><strong>{{ queryClass === 0 ? '转子不平衡' : '轴承预紧异常' }}</strong></div>
      <div><span>类别 1 后验</span><strong>{{ (100 * queryProbability).toFixed(1) }}%</strong></div>
    </template>
    <template #note>彩色方格不是训练样本，而是整张特征平面上的预测。两条椭圆是相同概率密度的协方差滚道：LDA 强制两类滚道同形，二次项比较时抵消；QDA 保留各自的形状、方向与体积，因此分界可以弯曲。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.boundary-scroll{overflow-x:auto}.boundary-scroll svg{display:block;width:100%;height:auto}.panel{fill:#0b1218}.cell{stroke:none}.cell-zero{fill:#1fc7bc}.cell-one{fill:#efb449}.grid line{stroke:#253640;stroke-width:1}.race{fill:none;stroke-width:3;stroke-dasharray:9 5}.race-zero{stroke:#38d8cc}.race-one{stroke:#ffc864}.mean{stroke:#071015;stroke-width:2}.mean-zero{fill:#38d8cc}.mean-one{fill:#ffc864}.probe circle{fill:#d477ff;fill-opacity:.22;stroke:#e5b2ff;stroke-width:2}.probe line{stroke:#e5b2ff;stroke-width:2}.readout{fill:#111e27;stroke:#425865}.eyebrow,.readout-label,.readout-value,.axis{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#70a6b3;font-size:9px;letter-spacing:.12em}.readout-label{fill:#96aab2;font-size:8px}.readout-value{fill:#f1d37a;font-size:10px;font-weight:800}.axis{fill:#78909a;font-size:8px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebdc5;font-size:11px}.control output{color:#ffc864;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#23c9bd}button{min-height:44px;padding:9px 12px;border:1px solid #49606c;border-radius:6px;color:#dce8eb;background:#13232c;font-weight:760;cursor:pointer}button.active{border-color:#efb449;color:#fff0c4;background:#44331c}button:focus-visible,input:focus-visible{outline:3px solid #d477ff;outline-offset:2px}@media(max-width:560px){.boundary-scroll{margin-right:-18px;padding-right:18px}.boundary-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
