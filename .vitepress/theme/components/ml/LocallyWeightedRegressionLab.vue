<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const queryTemperature = ref(27)
const bandwidth = ref(2.2)
const xValues = [12.0, 13.1, 14.4, 15.8, 17.0, 18.3, 19.1, 20.4, 21.0, 22.2, 23.1, 24.0, 24.8, 25.9, 27.1, 28.2, 29.0, 30.0]
const noise = [0.010, -0.013, 0.008, -0.006, 0.004, -0.011, 0.015, -0.008, 0.005, -0.004, 0.012, -0.010, 0.006, 0.014, -0.009, 0.011, -0.006, 0.008]
const truePerformance = (wetBulb: number) => {
  const centered = wetBulb - 18
  return 0.50 + 0.004 * centered + 0.0009 * centered ** 2 + 0.008 * Math.max(wetBulb - 23, 0) ** 1.3
}
const yValues = xValues.map((value, index) => truePerformance(value) + noise[index])

const chart = { left: 48, right: 570, top: 34, bottom: 262 }
const xPlot = (value: number) => chart.left + (value - 10) / 22 * (chart.right - chart.left)
const yPlot = (value: number) => chart.bottom - (value - 0.42) / 0.46 * (chart.bottom - chart.top)
const truthGrid = Array.from({ length: 121 }, (_, index) => 10 + index * 22 / 120)

const result = computed(() => {
  const weights = xValues.map((value) => Math.exp(-0.5 * ((value - queryTemperature.value) / bandwidth.value) ** 2))
  const dx = xValues.map((value) => value - queryTemperature.value)
  const sumWeight = weights.reduce((sum, value) => sum + value, 0)
  const sumWeightDx = weights.reduce((sum, value, index) => sum + value * dx[index], 0)
  const sumWeightDx2 = weights.reduce((sum, value, index) => sum + value * dx[index] ** 2, 0)
  const sumWeightY = weights.reduce((sum, value, index) => sum + value * yValues[index], 0)
  const sumWeightDxY = weights.reduce((sum, value, index) => sum + value * dx[index] * yValues[index], 0)
  const determinant = sumWeight * sumWeightDx2 - sumWeightDx ** 2 + 1e-10
  const intercept = (sumWeightY * sumWeightDx2 - sumWeightDxY * sumWeightDx) / determinant
  const slope = (sumWeight * sumWeightDxY - sumWeightDx * sumWeightY) / determinant
  const localConstant = sumWeightY / sumWeight
  const effectiveSampleSize = sumWeight ** 2 / weights.reduce((sum, value) => sum + value ** 2, 0)
  const rightWeight = weights.reduce((sum, value, index) => sum + (xValues[index] >= queryTemperature.value ? value : 0), 0)
  const leftWeight = sumWeight - rightWeight
  const lineStart = Math.max(10, queryTemperature.value - 3 * bandwidth.value)
  const lineEnd = Math.min(32, queryTemperature.value + 3 * bandwidth.value)
  return {
    weights,
    intercept,
    slope,
    localConstant,
    effectiveSampleSize,
    imbalance: (rightWeight - leftWeight) / sumWeight,
    lineStart,
    lineEnd,
    lineStartY: intercept + slope * (lineStart - queryTemperature.value),
    lineEndY: intercept + slope * (lineEnd - queryTemperature.value),
  }
})

const truthPoints = truthGrid.map((value) => `${xPlot(value)},${yPlot(truePerformance(value))}`).join(' ')
</script>

<template>
  <MlExperimentFrame
    index="07-B"
    title="移动局部测量灯，重算查询点模型"
    question="查询点靠近边界或进入稀疏温区时，带宽怎样改变有效样本、局部斜率与边界偏差？"
    formula="β̂(x₀) = argminβ Σᵢ Kτ(xᵢ−x₀)[yᵢ−β₀−β₁(xᵢ−x₀)]²"
  >
    <template #visual>
      <figure class="local-inspection">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`查询温度 ${queryTemperature.toFixed(1)} 摄氏度，带宽 ${bandwidth.toFixed(1)}，有效样本 ${result.effectiveSampleSize.toFixed(1)}`">
          <defs>
            <linearGradient id="local-beam" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#f4c95d" stop-opacity=".28"/><stop offset="1" stop-color="#f4c95d" stop-opacity="0"/></linearGradient>
            <clipPath id="local-plot-clip"><rect x="48" y="34" width="522" height="228" rx="5" /></clipPath>
          </defs>
          <rect x="16" y="16" width="568" height="294" rx="9" class="inspection-bg" />
          <text x="48" y="28" class="panel-title">冷站性能局部复核 / 权重越亮，影响越大</text>
          <line v-for="tick in [0.5,0.6,0.7,0.8]" :key="`y-${tick}`" x1="48" x2="570" :y1="yPlot(tick)" :y2="yPlot(tick)" class="grid" />
          <line v-for="tick in [12,16,20,24,28,32]" :key="`x-${tick}`" :x1="xPlot(tick)" :x2="xPlot(tick)" y1="34" y2="262" class="grid" />
          <g clip-path="url(#local-plot-clip)">
            <path :d="`M ${xPlot(queryTemperature - 2.4 * bandwidth)} 34 L ${xPlot(queryTemperature + 2.4 * bandwidth)} 34 L ${xPlot(queryTemperature)} 262 Z`" class="inspection-beam" />
            <polyline :points="truthPoints" class="truth-line" />
            <line :x1="xPlot(result.lineStart)" :y1="yPlot(result.lineStartY)" :x2="xPlot(result.lineEnd)" :y2="yPlot(result.lineEndY)" class="local-line" />
          </g>
          <line :x1="xPlot(queryTemperature)" :x2="xPlot(queryTemperature)" y1="34" y2="262" class="query-line" />
          <g v-for="(value,index) in xValues" :key="`point-${index}`">
            <line :x1="xPlot(value)" :x2="xPlot(value)" :y1="yPlot(yValues[index])" y2="262" :style="{ opacity: 0.08 + 0.42 * result.weights[index] }" class="weight-stem" />
            <circle :cx="xPlot(value)" :cy="yPlot(yValues[index])" :r="3.8 + 5 * result.weights[index]" :style="{ opacity: 0.28 + 0.72 * result.weights[index] }" class="weighted-point" />
          </g>
          <circle :cx="xPlot(queryTemperature)" :cy="yPlot(result.intercept)" r="8" class="query-prediction" />
          <text :x="Math.min(512, xPlot(queryTemperature) + 12)" :y="Math.max(48, yPlot(result.intercept) - 10)" class="query-label">ŷ(x₀)</text>
          <text x="462" y="284" class="axis-label">室外湿球温度 / °C</text>
          <text x="48" y="299" class="footer-label">局部线性模型在中心化坐标 [1, x−x₀] 上拟合，因此截距就是查询点预测。</text>
        </svg>
        <figcaption>
          <span class="legend point" />稳定窗口及局部权重
          <span class="legend local" />局部线性拟合
          <span class="legend truth" />教学真曲线
          <span class="legend query" />查询位置与预测
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>查询湿球温度 x₀ <output>{{ queryTemperature.toFixed(1) }}°C</output></span>
        <input v-model.number="queryTemperature" type="range" min="11" max="31" step="0.2">
      </label>
      <label>
        <span>带宽 τ <output>{{ bandwidth.toFixed(1) }}°C</output></span>
        <input v-model.number="bandwidth" type="range" min="0.5" max="5" step="0.1">
      </label>
      <p>把 x₀ 推到 31°C：有效权重会向左侧失衡。减小 τ 能跟随局部弯曲，却可能只剩少数窗口；增大 τ 会提高稳定性，并最终接近一条全局线性回归。</p>
    </template>
    <template #metrics>
      <div><span>局部线性预测</span><strong>{{ result.intercept.toFixed(3) }}</strong></div>
      <div><span>局部常数预测</span><strong>{{ result.localConstant.toFixed(3) }}</strong></div>
      <div><span>局部斜率 / °C</span><strong>{{ result.slope.toFixed(3) }}</strong></div>
      <div><span>有效样本 / 权重失衡</span><strong>{{ result.effectiveSampleSize.toFixed(1) }} / {{ result.imbalance.toFixed(2) }}</strong></div>
    </template>
    <template #note>
      Gaussian 外形只是一种确定性相似度权重，不表示训练点服从 Gaussian。局部常数等于归一化加权平均；局部线性允许等效权重出现负值来复现斜率，因而通常减小边界偏差，但也更容易在稀疏区失稳。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.local-inspection{margin:0}.local-inspection svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:10px;background:#10182e}.inspection-bg{fill:#10182e}.grid{stroke:#263554;stroke-width:1}.inspection-beam{fill:url(#local-beam)}.truth-line{fill:none;stroke:#8391b2;stroke-width:2;stroke-dasharray:7 5}.local-line{stroke:#f4c95d;stroke-width:4}.query-line{stroke:#e15b4b;stroke-width:2;stroke-dasharray:5 4}.weight-stem{stroke:#72d5ce;stroke-width:2}.weighted-point{fill:#72d5ce;stroke:#fff;stroke-width:1.4}.query-prediction{fill:#f4c95d;stroke:#fff;stroke-width:2}.local-inspection text{font:700 10px ui-monospace,monospace}.panel-title{fill:#dfe5f7}.axis-label,.footer-label{fill:#8f9cbc}.query-label{fill:#f4c95d}.local-inspection figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}.legend.point{height:10px;border-radius:50%;background:#72d5ce}.legend.local{background:#f4c95d}.legend.truth{border-top:2px dashed #8391b2}.legend.query{border-top:2px dashed #e15b4b}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}@media(max-width:560px){.local-inspection{overflow-x:auto;overscroll-behavior-inline:contain}.local-inspection svg{width:560px;max-width:none}.local-inspection figcaption{min-width:560px}}
</style>
