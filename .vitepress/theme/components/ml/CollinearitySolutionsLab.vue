<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const redundancyPercent = ref(92)
const labelPerturbation = ref(0.02)
const betaA = { first: 2.4, second: 0.8 }

const result = computed(() => {
  const correlation = Math.min(redundancyPercent.value / 100, 0.9999)
  const weakSingularValue = Math.sqrt(1 - correlation)
  const shift = labelPerturbation.value / weakSingularValue
  const betaB = {
    first: betaA.first + shift,
    second: betaA.second - shift
  }
  return {
    correlation,
    weakSingularValue,
    condition: (1 + correlation) / (1 - correlation),
    betaB,
    coefficientDistance: Math.sqrt(2) * shift,
    predictionRms: Math.sqrt(2) * labelPerturbation.value,
    coefficientSumA: betaA.first + betaA.second,
    coefficientSumB: betaB.first + betaB.second
  }
})

const chart = { left: 68, top: 30, width: 480, height: 252, minX: -1, maxX: 5, minY: -2, maxY: 4 }
const x = (value: number) => chart.left + (value - chart.minX) / (chart.maxX - chart.minX) * chart.width
const y = (value: number) => chart.top + chart.height - (value - chart.minY) / (chart.maxY - chart.minY) * chart.height
const pointB = computed(() => ({ x: x(result.value.betaB.first), y: y(result.value.betaB.second) }))
const majorRadius = computed(() => Math.min(174, 24 * Math.sqrt((1 + result.value.correlation) / Math.max(1 - result.value.correlation, 1e-4))))
const tickValues = [-1, 0, 1, 2, 3, 4, 5]
const formatCondition = (value: number) => value >= 1000 ? value.toExponential(2) : value.toFixed(1)
</script>

<template>
  <MlExperimentFrame
    index="04-B"
    title="近共线让系数失稳，预测却几乎不动"
    question="两个传感器记录越来越相似时，同样大小的标签扰动会如何沿弱方向放大？"
    formula="κ(XᵀX) = (1 + ρ) / (1 − ρ)"
  >
    <template #visual>
      <figure class="valley-board">
        <svg viewBox="0 0 600 328" role="img" :aria-label="`特征相关系数 ${result.correlation.toFixed(4)}，系数变化范数 ${result.coefficientDistance.toFixed(3)}，预测均方根变化 ${result.predictionRms.toFixed(3)}`">
          <rect x="16" y="16" width="568" height="296" rx="9" class="plot-bg" />
          <line v-for="tick in tickValues" :key="`x-${tick}`" :x1="x(tick)" :x2="x(tick)" :y1="chart.top" :y2="chart.top + chart.height" class="grid" />
          <line v-for="tick in tickValues.filter((tick) => tick <= 4)" :key="`y-${tick}`" :x1="chart.left" :x2="chart.left + chart.width" :y1="y(tick)" :y2="y(tick)" class="grid" />
          <line :x1="x(chart.minX)" :x2="x(chart.maxX)" :y1="y(result.coefficientSumA - chart.minX)" :y2="y(result.coefficientSumA - chart.maxX)" class="equal-prediction" />

          <g :transform="`translate(${x(betaA.first)} ${y(betaA.second)}) rotate(-45)`">
            <ellipse :rx="majorRadius" ry="23" class="contour outer" />
            <ellipse :rx="majorRadius * .67" ry="15" class="contour middle" />
            <ellipse :rx="majorRadius * .36" ry="8" class="contour inner" />
          </g>

          <line :x1="x(betaA.first)" :y1="y(betaA.second)" :x2="pointB.x" :y2="pointB.y" class="coefficient-shift" />
          <circle :cx="x(betaA.first)" :cy="y(betaA.second)" r="8" class="fit-a" />
          <circle :cx="pointB.x" :cy="pointB.y" r="8" class="fit-b" />
          <text :x="x(betaA.first) + 11" :y="y(betaA.second) - 9" class="fit-a-label">拟合 A</text>
          <text :x="Math.min(514, pointB.x + 11)" :y="Math.max(45, pointB.y - 9)" class="fit-b-label">拟合 B</text>
          <text x="434" y="303" class="axis-label">β₁ · 主流量计</text>
          <text x="26" y="46" class="axis-label">β₂</text>
          <text x="26" y="61" class="axis-label">备用表</text>
          <text x="76" y="300" class="valley-label">虚线：β₁ + β₂ 不变的预测等价方向</text>
        </svg>
        <figcaption>
          <span class="legend fit-a" />原始标签拟合
          <span class="legend fit-b" />微扰标签拟合
          <span class="legend contour" />RSS 等高线
          <span class="legend valley" />近等预测方向
        </figcaption>
      </figure>
      <table>
        <thead><tr><th>拟合</th><th>主流量计 β₁</th><th>备用表 β₂</th><th>系数和</th></tr></thead>
        <tbody>
          <tr><td>A</td><td>{{ betaA.first.toFixed(3) }}</td><td>{{ betaA.second.toFixed(3) }}</td><td>{{ result.coefficientSumA.toFixed(3) }}</td></tr>
          <tr><td>B</td><td>{{ result.betaB.first.toFixed(3) }}</td><td>{{ result.betaB.second.toFixed(3) }}</td><td>{{ result.coefficientSumB.toFixed(3) }}</td></tr>
        </tbody>
      </table>
    </template>
    <template #controls>
      <label>
        <span>传感器相似度 ρ <output>{{ result.correlation.toFixed(4) }}</output></span>
        <input v-model.number="redundancyPercent" type="range" min="0" max="99.99" step="0.01">
      </label>
      <label>
        <span>标签微扰幅度 <output>{{ labelPerturbation.toFixed(3) }}</output></span>
        <input v-model.number="labelPerturbation" type="range" min="0.005" max="0.030" step="0.001">
      </label>
      <p>两次拟合只相差一个固定范数的标签扰动。ρ 越接近 1，RSS 山谷越扁，扰动越容易沿 β₁−β₂ 方向变成大系数变化；β₁+β₂ 与批量预测仍接近不变。</p>
    </template>
    <template #metrics>
      <div><span>Gram 条件数</span><strong>{{ formatCondition(result.condition) }}</strong></div>
      <div><span>弱奇异值比例</span><strong>{{ result.weakSingularValue.toFixed(4) }}</strong></div>
      <div><span>系数变化 ‖Δβ‖₂</span><strong>{{ result.coefficientDistance.toFixed(3) }}</strong></div>
      <div><span>预测 RMS 变化</span><strong>{{ result.predictionRms.toFixed(3) }}</strong></div>
    </template>
    <template #note>
      这是标准化二特征的解析实验：XᵀX 的特征值与 1±ρ 成正比。微小扰动沿弱奇异方向被除以 √(1−ρ)，所以“预测稳定”不能推出“每个传感器的边际系数可信”。ρ=1 时会出现一整条等价解，伪逆只选其中的最小范数点。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.ml-experiment__visual th{background:#17203b}
.valley-board{margin:0}.valley-board svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:10px;background:#10182e}.plot-bg{fill:#10182e}.grid{stroke:#263554;stroke-width:1}.equal-prediction{stroke:#6f7da0;stroke-width:2;stroke-dasharray:8 7}.contour{fill:none;stroke:#5a6d9d}.contour.middle{stroke:#7488bd}.contour.inner{stroke:#96a8d6}.coefficient-shift{stroke:#f4c95d;stroke-width:3}.fit-a{fill:#72d5ce;stroke:#fff;stroke-width:2}.fit-b{fill:#f06b5b;stroke:#fff;stroke-width:2}.valley-board text{font:700 10px ui-monospace,monospace}.fit-a-label{fill:#72d5ce}.fit-b-label{fill:#f06b5b}.axis-label,.valley-label{fill:#8f9cbc}.valley-board figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:11px;height:11px;margin-right:4px;vertical-align:middle}.legend.fit-a{border-radius:50%;background:#72d5ce}.legend.fit-b{border-radius:50%;background:#f06b5b}.legend.contour{border:2px solid #7488bd;border-radius:50%}.legend.valley{height:2px;border-top:2px dashed #6f7da0}.ml-experiment__visual table{width:100%;margin:10px 0 0;border-collapse:collapse;color:#c8d1e8;background:#10182e;font-size:10px}.ml-experiment__visual thead tr{background:#17203b}.ml-experiment__visual tbody tr:nth-child(odd){background:#10182e}.ml-experiment__visual tbody tr:nth-child(even){background:#141e37}.ml-experiment__visual th,.ml-experiment__visual td{padding:7px;border:1px solid #2d3c68;color:#c8d1e8;text-align:right}.ml-experiment__visual th:first-child,.ml-experiment__visual td:first-child{text-align:left}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}
</style>
