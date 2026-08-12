<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const columnAngle = ref(24)
const targetAngle = ref(68)
const targetLength = ref(1.0)

const origin = { x: 300, y: 270 }
const pixelsPerUnit = 176
const radians = (degrees: number) => degrees * Math.PI / 180

const geometry = computed(() => {
  const unit = {
    x: Math.cos(radians(columnAngle.value)),
    y: Math.sin(radians(columnAngle.value))
  }
  const target = {
    x: targetLength.value * Math.cos(radians(targetAngle.value)),
    y: targetLength.value * Math.sin(radians(targetAngle.value))
  }
  const projectionScale = target.x * unit.x + target.y * unit.y
  const prediction = {
    x: projectionScale * unit.x,
    y: projectionScale * unit.y
  }
  const residual = {
    x: target.x - prediction.x,
    y: target.y - prediction.y
  }
  const targetNorm = Math.hypot(target.x, target.y)
  const predictionNorm = Math.hypot(prediction.x, prediction.y)
  const residualNorm = Math.hypot(residual.x, residual.y)
  return {
    unit,
    target,
    prediction,
    residual,
    projectionScale,
    targetNorm,
    predictionNorm,
    residualNorm,
    dot: unit.x * residual.x + unit.y * residual.y,
    pythagoreanGap: targetNorm ** 2 - predictionNorm ** 2 - residualNorm ** 2
  }
})

function point(vector: { x: number; y: number }) {
  return {
    x: origin.x + vector.x * pixelsPerUnit,
    y: origin.y - vector.y * pixelsPerUnit
  }
}

const targetPoint = computed(() => point(geometry.value.target))
const predictionPoint = computed(() => point(geometry.value.prediction))
const lineStart = computed(() => point({ x: -1.55 * geometry.value.unit.x, y: -1.55 * geometry.value.unit.y }))
const lineEnd = computed(() => point({ x: 1.55 * geometry.value.unit.x, y: 1.55 * geometry.value.unit.y }))
const rightAngle = computed(() => {
  const along = 0.085
  const normal = 0.085
  const u = geometry.value.unit
  const residualNorm = Math.max(geometry.value.residualNorm, 1e-12)
  const n = {
    x: geometry.value.residual.x / residualNorm,
    y: geometry.value.residual.y / residualNorm
  }
  const base = geometry.value.prediction
  return [
    point({ x: base.x + along * u.x, y: base.y + along * u.y }),
    point({ x: base.x + along * u.x + normal * n.x, y: base.y + along * u.y + normal * n.y }),
    point({ x: base.x + normal * n.x, y: base.y + normal * n.y })
  ]
})

const format = (value: number) => Math.abs(value) < 5e-10 ? '0.0000' : value.toFixed(4)
</script>

<template>
  <MlExperimentFrame
    index="04-A"
    title="把目标向量投到可表达的预测空间"
    question="当模型只能沿一条列空间直线移动时，最短残差为什么必然与这条线垂直？"
    formula="ŷ = P_X y,   Xᵀ(y − ŷ) = 0"
  >
    <template #visual>
      <figure class="projection-board">
        <svg viewBox="0 0 600 336" role="img" :aria-label="`目标向量长度 ${targetLength.toFixed(2)}，投影长度 ${geometry.predictionNorm.toFixed(3)}，残差长度 ${geometry.residualNorm.toFixed(3)}`">
          <defs>
            <pattern id="ols-grid" width="24" height="24" patternUnits="userSpaceOnUse">
              <path d="M 24 0 L 0 0 0 24" class="minor-grid" />
            </pattern>
            <marker id="arrow-target" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" class="target-head" />
            </marker>
            <marker id="arrow-prediction" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" class="prediction-head" />
            </marker>
            <marker id="arrow-residual" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" class="residual-head" />
            </marker>
          </defs>
          <rect x="16" y="16" width="568" height="304" rx="9" class="plot-bg" />
          <rect x="16" y="16" width="568" height="304" rx="9" fill="url(#ols-grid)" />
          <line :x1="lineStart.x" :y1="lineStart.y" :x2="lineEnd.x" :y2="lineEnd.y" class="column-space" />
          <text :x="lineEnd.x - 84" :y="lineEnd.y - 12" class="space-label">col(X)</text>

          <line :x1="origin.x" :y1="origin.y" :x2="targetPoint.x" :y2="targetPoint.y" class="target-vector" marker-end="url(#arrow-target)" />
          <line :x1="origin.x" :y1="origin.y" :x2="predictionPoint.x" :y2="predictionPoint.y" class="prediction-vector" marker-end="url(#arrow-prediction)" />
          <line :x1="predictionPoint.x" :y1="predictionPoint.y" :x2="targetPoint.x" :y2="targetPoint.y" class="residual-vector" marker-end="url(#arrow-residual)" />
          <polyline :points="rightAngle.map((item) => `${item.x},${item.y}`).join(' ')" class="right-angle" />

          <circle :cx="origin.x" :cy="origin.y" r="4" class="origin-dot" />
          <circle :cx="predictionPoint.x" :cy="predictionPoint.y" r="4" class="prediction-dot" />
          <text :x="targetPoint.x + 9" :y="targetPoint.y - 8" class="target-label">y</text>
          <text :x="predictionPoint.x + 9" :y="predictionPoint.y + 17" class="prediction-label">ŷ</text>
          <text :x="(predictionPoint.x + targetPoint.x) / 2 + 8" :y="(predictionPoint.y + targetPoint.y) / 2" class="residual-label">r</text>
          <text x="34" y="45" class="board-caption">观测空间 R² · 一维列空间</text>
        </svg>
        <figcaption>
          <span class="legend target" />目标 y
          <span class="legend prediction" />预测 ŷ
          <span class="legend residual" />残差 r
          <span class="legend space" />模型可表达的 col(X)
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>列空间方向 <output>{{ columnAngle }}°</output></span>
        <input v-model.number="columnAngle" type="range" min="-10" max="80" step="1">
      </label>
      <label>
        <span>目标向量方向 <output>{{ targetAngle }}°</output></span>
        <input v-model.number="targetAngle" type="range" min="15" max="155" step="1">
      </label>
      <label>
        <span>目标向量长度 <output>{{ targetLength.toFixed(2) }}</output></span>
        <input v-model.number="targetLength" type="range" min="0.55" max="1.25" step="0.05">
      </label>
      <p>拖动方向后，黄色预测始终停在列空间上；红色残差由投影点指向目标。直角标记不是额外假设，而是最短距离的一阶最优条件。</p>
    </template>
    <template #metrics>
      <div><span>投影系数 uᵀy</span><strong>{{ format(geometry.projectionScale) }}</strong></div>
      <div><span>残差长度 ‖r‖₂</span><strong>{{ geometry.residualNorm.toFixed(4) }}</strong></div>
      <div><span>正交审计 uᵀr</span><strong>{{ format(geometry.dot) }}</strong></div>
      <div><span>勾股闭合误差</span><strong>{{ format(geometry.pythagoreanGap) }}</strong></div>
    </template>
    <template #note>
      这里把 n=2、rank(X)=1 的情形画在纸面上。高维 OLS 完全相同：X 的每一列都是观测空间中的方向，最优残差与整个列空间正交。正交只说明训练集上的代数最优，不说明线性关系真实、噪声独立或模型具有因果意义。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.projection-board{margin:0}.projection-board svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:10px;background:#10182e}.plot-bg{fill:#10182e}.minor-grid{fill:none;stroke:#253353;stroke-width:1}.column-space{stroke:#6d7da4;stroke-width:3;stroke-dasharray:9 7}.target-vector{stroke:#72d5ce;stroke-width:4}.prediction-vector{stroke:#f4c95d;stroke-width:5}.residual-vector{stroke:#f06b5b;stroke-width:4}.target-head{fill:#72d5ce}.prediction-head{fill:#f4c95d}.residual-head{fill:#f06b5b}.right-angle{fill:none;stroke:#f3f5ff;stroke-width:2}.origin-dot{fill:#f3f5ff}.prediction-dot{fill:#f4c95d}.projection-board text{font:700 11px ui-monospace,monospace}.board-caption,.space-label{fill:#8f9cbc}.target-label{fill:#72d5ce}.prediction-label{fill:#f4c95d}.residual-label{fill:#f06b5b}.projection-board figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:14px;height:3px;margin-right:4px;vertical-align:middle}.legend.target{background:#72d5ce}.legend.prediction{background:#f4c95d}.legend.residual{background:#f06b5b}.legend.space{border-top:2px dashed #6d7da4}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}
</style>
