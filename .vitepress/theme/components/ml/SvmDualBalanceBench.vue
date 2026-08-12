<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const alpha = ref(0.04)
const cap = ref(0.10)

watch(cap, (value) => {
  if (alpha.value > value) alpha.value = value
})

const optimumAlpha = computed(() => Math.min(1 / 16, cap.value))

const result = computed(() => {
  const weight = 8 * alpha.value
  const minimumMargin = 16 * alpha.value
  const dualObjective = 4 * alpha.value - 32 * alpha.value ** 2
  const optimumObjective = 4 * optimumAlpha.value - 32 * optimumAlpha.value ** 2
  const dualDerivative = 4 - 64 * alpha.value
  const tolerance = 1e-6
  const projectedKktResidual = alpha.value <= tolerance
    ? Math.max(0, dualDerivative)
    : alpha.value >= cap.value - tolerance
      ? Math.max(0, -dualDerivative)
      : Math.abs(dualDerivative)
  return {
    weight,
    minimumMargin,
    dualObjective,
    optimumObjective,
    equalityResidual: 0,
    projectedKktResidual,
    halfMargin: weight > 1e-9 ? 1 / weight : Number.POSITIVE_INFINITY,
    state: Math.abs(alpha.value - optimumAlpha.value) < 1e-4
      ? 'KKT 平衡'
      : alpha.value < optimumAlpha.value
        ? '仍可上升'
        : '越过峰值',
  }
})

const mapPlaneX = (value: number) => 45 + (value + 4) / 8 * 320
const mapPlaneY = (value: number) => 278 - (value + 2) / 4 * 220
const mapCurveX = (value: number) => 415 + value / 0.14 * 145
const mapCurveY = (value: number) => 233 - (value + 0.08) / 0.22 * 150

const curvePath = computed(() => {
  const values = Array.from({ length: 71 }, (_, index) => index * 0.14 / 70)
  return values.map((value, index) => {
    const objective = 4 * value - 32 * value ** 2
    return `${index === 0 ? 'M' : 'L'} ${mapCurveX(value)} ${mapCurveY(objective)}`
  }).join(' ')
})

const snapToOptimum = () => {
  alpha.value = optimumAlpha.value
}
</script>

<template>
  <MlExperimentFrame
    index="12-A"
    title="沿对偶可行轨道寻找 KKT 平衡点"
    question="为什么 α 既要推动对偶目标上升，又必须服从标签平衡、盒约束与互补松弛？"
    formula="max 1ᵀα−½αᵀQα,  0≤α≤C,  yᵀα=0"
  >
    <template #visual>
      <figure class="dual-bench">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`alpha ${alpha.toFixed(4)}，最小函数间隔 ${result.minimumMargin.toFixed(3)}，状态 ${result.state}`">
          <defs>
            <clipPath id="dual-plane-clip"><rect x="45" y="46" width="320" height="232" rx="5" /></clipPath>
          </defs>
          <rect x="15" y="15" width="570" height="296" rx="9" class="bench-bg" />
          <text x="45" y="31" class="panel-title">四个活动约束 / 原始平面</text>
          <text x="414" y="31" class="panel-title">对偶目标轨道</text>
          <rect x="45" y="46" width="320" height="232" rx="5" class="plot-bg" />
          <g class="grid">
            <line v-for="value in [-3,-2,-1,0,1,2,3]" :key="`gx-${value}`" :x1="mapPlaneX(value)" :x2="mapPlaneX(value)" y1="46" y2="278" />
            <line v-for="value in [-1,0,1]" :key="`gy-${value}`" x1="45" x2="365" :y1="mapPlaneY(value)" :y2="mapPlaneY(value)" />
          </g>
          <g clip-path="url(#dual-plane-clip)">
            <rect
              v-if="Number.isFinite(result.halfMargin)"
              :x="mapPlaneX(-result.halfMargin)"
              y="46"
              :width="mapPlaneX(result.halfMargin)-mapPlaneX(-result.halfMargin)"
              height="232"
              class="margin-slab"
            />
            <line :x1="mapPlaneX(0)" :x2="mapPlaneX(0)" y1="46" y2="278" class="decision-line" />
            <line v-if="Number.isFinite(result.halfMargin)" :x1="mapPlaneX(-result.halfMargin)" :x2="mapPlaneX(-result.halfMargin)" y1="46" y2="278" class="jaw negative" />
            <line v-if="Number.isFinite(result.halfMargin)" :x1="mapPlaneX(result.halfMargin)" :x2="mapPlaneX(result.halfMargin)" y1="46" y2="278" class="jaw positive" />
            <g v-for="point in [{x:-2,y:-1,label:-1},{x:-2,y:1,label:-1},{x:2,y:-1,label:1},{x:2,y:1,label:1}]" :key="`${point.x}-${point.y}`">
              <circle :cx="mapPlaneX(point.x)" :cy="mapPlaneY(point.y)" r="11" class="alpha-ring" />
              <circle :cx="mapPlaneX(point.x)" :cy="mapPlaneY(point.y)" r="6.5" :class="point.label === 1 ? 'positive-point' : 'negative-point'" />
              <text :x="mapPlaneX(point.x)+10" :y="mapPlaneY(point.y)-9" class="alpha-label">α={{ alpha.toFixed(3) }}</text>
            </g>
            <circle :cx="mapPlaneX(-3)" :cy="mapPlaneY(0)" r="6" class="negative-point far" />
            <circle :cx="mapPlaneX(3)" :cy="mapPlaneY(0)" r="6" class="positive-point far" />
          </g>
          <text x="205" y="300" class="axis-label center">标准化证据方向 z₁</text>

          <line x1="415" x2="560" y1="233" y2="233" class="curve-axis" />
          <line x1="415" x2="415" y1="70" y2="242" class="curve-axis" />
          <path :d="curvePath" class="dual-curve" />
          <line :x1="mapCurveX(cap)" :x2="mapCurveX(cap)" y1="70" y2="242" class="cap-line" />
          <circle :cx="mapCurveX(alpha)" :cy="mapCurveY(result.dualObjective)" r="6" class="current-point" />
          <circle :cx="mapCurveX(optimumAlpha)" :cy="mapCurveY(result.optimumObjective)" r="4" class="optimum-point" />
          <text x="422" y="64" class="curve-label">D(α)</text>
          <text :x="mapCurveX(cap)-3" y="252" class="curve-label end">C</text>
          <text x="415" y="270" class="curve-label">0</text>
          <text x="560" y="270" class="curve-label end">α</text>
          <text x="415" y="292" class="audit-label">yᵀα</text>
          <text x="560" y="292" class="audit-value safe">{{ result.equalityResidual.toExponential(1) }}</text>
          <text x="415" y="310" class="audit-label">KKT 投影残差</text>
          <text x="560" y="310" :class="['audit-value', result.projectedKktResidual < 1e-4 ? 'safe' : 'danger']">{{ result.projectedKktResidual.toExponential(1) }}</text>
        </svg>
        <figcaption><span><i class="negative" />放行 −1</span><span><i class="positive" />复检 +1</span><span><i class="active" />共同 α</span><span><i class="cap" />盒约束 C</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>共同对偶系数 α <output>{{ alpha.toFixed(4) }}</output></span>
        <input v-model.number="alpha" type="range" min="0" :max="cap" step="0.0025">
      </label>
      <label>
        <span>盒约束 C <output>{{ cap.toFixed(4) }}</output></span>
        <input v-model.number="cap" type="range" min="0.02" max="0.14" step="0.0025">
      </label>
      <button type="button" @click="snapToOptimum">吸附到受约束最优点</button>
      <p>四个活动样本共用 α，因此标签平衡恒为零。C 低于 1/16 时最优点被盒约束截住，投影 KKT 残差按上界处的单侧可行方向计算；C 足够大时，α=1/16 同时满足函数间隔 1 与互补松弛。</p>
    </template>
    <template #metrics>
      <div><span>对偶目标 D(α)</span><strong>{{ result.dualObjective.toFixed(4) }}</strong></div>
      <div><span>最小函数间隔</span><strong>{{ result.minimumMargin.toFixed(3) }}</strong></div>
      <div><span>KKT 投影残差</span><strong>{{ result.projectedKktResidual.toExponential(1) }}</strong></div>
      <div><span>当前状态</span><strong>{{ result.state }}</strong></div>
    </template>
    <template #note>
      这条一维轨道是四点对称问题的完整对偶可行切片，不是一般 SVM 的求解器。它把标签平衡、盒约束、目标最大化和投影 KKT 条件放到同一个可检查坐标中。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.dual-bench{margin:0}.dual-bench svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0c1925}.bench-bg{fill:#0c1925}.dual-bench text{font:700 9px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.plot-bg{fill:#101f2c}.grid line{stroke:#213b4d;stroke-width:1}.margin-slab{fill:#f3c85b;opacity:.08}.decision-line{stroke:#f3c85b;stroke-width:3}.jaw{stroke-width:2;stroke-dasharray:7 5}.jaw.negative{stroke:#63d0c6}.jaw.positive{stroke:#e35d68}.negative-point{fill:#63d0c6;stroke:#dbeaf0;stroke-width:1.5}.positive-point{fill:#e35d68;stroke:#dbeaf0;stroke-width:1.5}.far{opacity:.55}.alpha-ring{fill:none;stroke:#f08a55;stroke-width:2}.alpha-label{fill:#f3c85b}.axis-label,.curve-label,.audit-label{fill:#829eaf}.axis-label.center{text-anchor:middle}.curve-label.end{text-anchor:end}.curve-axis{stroke:#355165}.dual-curve{fill:none;stroke:#63d0c6;stroke-width:2.5}.cap-line{stroke:#e35d68;stroke-width:1.5;stroke-dasharray:4 4}.current-point{fill:#f08a55;stroke:#fff;stroke-width:1.5}.optimum-point{fill:#f3c85b;stroke:#fff;stroke-width:1}.audit-value{fill:#dbeaf0;text-anchor:end}.audit-value.safe{fill:#63d0c6}.audit-value.danger{fill:#e35d68}.dual-bench figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.dual-bench figcaption span{display:inline-flex;align-items:center;gap:5px}.dual-bench figcaption i{width:12px;height:8px;border-radius:2px}.dual-bench figcaption i.negative{background:#63d0c6}.dual-bench figcaption i.positive{background:#e35d68}.dual-bench figcaption i.active{border:2px solid #f08a55;background:transparent}.dual-bench figcaption i.cap{width:3px;background:#e35d68}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 11px ui-monospace,monospace;cursor:pointer}.ml-experiment__controls button:hover{border-color:#f3c85b}.ml-experiment__controls button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.dual-bench{overflow-x:auto;overscroll-behavior-inline:contain}.dual-bench svg{width:560px;max-width:none}.dual-bench figcaption{min-width:560px}}
</style>
