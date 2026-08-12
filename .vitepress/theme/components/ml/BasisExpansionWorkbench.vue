<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type BasisFamily = 'polynomial' | 'hinge'

const family = ref<BasisFamily>('polynomial')
const complexity = ref(5)
const logLambda = ref(-1)

const trainX = [12.0, 13.1, 14.4, 15.8, 17.0, 18.3, 19.1, 20.4, 21.0, 22.2, 23.1, 24.0, 24.8, 25.9, 27.1, 28.2, 29.0, 30.0]
const noise = [0.010, -0.013, 0.008, -0.006, 0.004, -0.011, 0.015, -0.008, 0.005, -0.004, 0.012, -0.010, 0.006, 0.014, -0.009, 0.011, -0.006, 0.008]
const validX = [12.6, 13.8, 15.1, 16.5, 17.7, 18.8, 19.8, 21.6, 22.7, 23.6, 24.4, 25.3, 26.5, 27.6, 28.6, 29.5]

const truePerformance = (wetBulb: number) => {
  const centered = wetBulb - 18
  return 0.50 + 0.004 * centered + 0.0009 * centered ** 2 + 0.008 * Math.max(wetBulb - 23, 0) ** 1.3
}
const trainY = trainX.map((value, index) => truePerformance(value) + noise[index])
const validY = validX.map((value, index) => truePerformance(value) + 0.008 * Math.sin(index * 1.7))

const scaled = (value: number) => (value - 21) / 9
const knotLocations = (count: number) => count === 1
  ? [0]
  : Array.from({ length: count }, (_, index) => -0.72 + index * 1.44 / (count - 1))
const features = (value: number, selectedFamily: BasisFamily, selectedComplexity: number) => {
  const z = scaled(value)
  if (selectedFamily === 'polynomial') {
    return Array.from({ length: selectedComplexity + 1 }, (_, power) => z ** power)
  }
  return [1, z, ...knotLocations(selectedComplexity).map((knot) => Math.max(z - knot, 0))]
}

const solve = (matrix: number[][], target: number[]) => {
  const a = matrix.map((row, index) => [...row, target[index]])
  const size = a.length
  for (let column = 0; column < size; column++) {
    let pivot = column
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row
    }
    ;[a[column], a[pivot]] = [a[pivot], a[column]]
    const diagonal = a[column][column]
    if (Math.abs(diagonal) < 1e-12) throw new Error('basis system is singular')
    for (let entry = column; entry <= size; entry++) a[column][entry] /= diagonal
    for (let row = 0; row < size; row++) {
      if (row === column) continue
      const factor = a[row][column]
      for (let entry = column; entry <= size; entry++) a[row][entry] -= factor * a[column][entry]
    }
  }
  return a.map((row) => row[size])
}

const dot = (left: number[], right: number[]) => left.reduce((sum, value, index) => sum + value * right[index], 0)
const gridX = Array.from({ length: 121 }, (_, index) => 10 + index * 22 / 120)
const chart = { left: 48, right: 570, top: 35, bottom: 218 }
const xPlot = (value: number) => chart.left + (value - 10) / 22 * (chart.right - chart.left)
const yPlot = (value: number) => chart.bottom - (value - 0.42) / 0.46 * (chart.bottom - chart.top)

const result = computed(() => {
  const lambda = 10 ** logLambda.value
  const design = trainX.map((value) => features(value, family.value, complexity.value))
  const dimension = design[0].length
  const gram = Array.from({ length: dimension }, (_, row) =>
    Array.from({ length: dimension }, (_, column) =>
      design.reduce((sum, values) => sum + values[row] * values[column], 0)
      + (row === column && row > 0 ? lambda : 0)
    )
  )
  const rhs = Array.from({ length: dimension }, (_, column) =>
    design.reduce((sum, values, index) => sum + values[column] * trainY[index], 0)
  )
  const beta = solve(gram, rhs)
  const predict = (value: number) => dot(features(value, family.value, complexity.value), beta)
  const predictions = gridX.map(predict)
  const trainRmse = Math.sqrt(trainX.reduce((sum, value, index) => sum + (predict(value) - trainY[index]) ** 2, 0) / trainX.length)
  const validRmse = Math.sqrt(validX.reduce((sum, value, index) => sum + (predict(value) - validY[index]) ** 2, 0) / validX.length)
  const inverse = Array.from({ length: dimension }, (_, column) => {
    const unit = Array.from({ length: dimension }, (_, index) => Number(index === column))
    return solve(gram, unit)
  })
  const effectiveDof = design.reduce((trace, row) => {
    const transformed = Array.from({ length: dimension }, (_, index) =>
      inverse.reduce((sum, column, columnIndex) => sum + column[index] * row[columnIndex], 0)
    )
    return trace + dot(row, transformed)
  }, 0)
  const contributions = beta.slice(1, 6).map((coefficient, index) =>
    gridX.map((value) => coefficient * features(value, family.value, complexity.value)[index + 1])
  )
  const maxContribution = Math.max(0.01, ...contributions.flat().map(Math.abs))
  return {
    lambda,
    beta,
    predictions,
    trainRmse,
    validRmse,
    effectiveDof,
    boundaryError: Math.abs(predict(32) - truePerformance(32)),
    contributions,
    maxContribution,
    knots: family.value === 'hinge' ? knotLocations(complexity.value).map((value) => value * 9 + 21) : [],
  }
})

const linePoints = (values: number[]) => values.map((value, index) => `${xPlot(gridX[index])},${yPlot(value)}`).join(' ')
const contributionPoints = (values: number[]) => values.map((value, index) => {
  const y = 278 - value / result.value.maxContribution * 24
  return `${xPlot(gridX[index])},${y}`
}).join(' ')
const contributionColors = ['basis-a', 'basis-b', 'basis-c', 'basis-d', 'basis-e']
</script>

<template>
  <MlExperimentFrame
    index="07-A"
    title="在性能曲线调试台混合基函数"
    question="增加全局幂次或局部折点后，训练误差、验证误差、有效自由度和边界外推会怎样分离？"
    formula="f(x) = βᵀφ(x),  df = tr(Sλ)"
  >
    <template #visual>
      <figure class="basis-console">
        <svg viewBox="0 0 600 330" role="img" :aria-label="`${family === 'polynomial' ? '多项式' : '折线样条'}基函数，复杂度 ${complexity}，验证均方根误差 ${result.validRmse.toFixed(3)}`">
          <defs><clipPath id="basis-plot-clip"><rect x="48" y="35" width="522" height="183" rx="5" /></clipPath></defs>
          <rect x="16" y="16" width="568" height="298" rx="9" class="console-bg" />
          <text x="48" y="28" class="panel-title">冷站单位冷量功耗 / kW·RT⁻¹</text>
          <line v-for="tick in [0.5,0.6,0.7,0.8]" :key="`y-${tick}`" x1="48" x2="570" :y1="yPlot(tick)" :y2="yPlot(tick)" class="grid" />
          <line v-for="tick in [12,16,20,24,28,32]" :key="`x-${tick}`" :x1="xPlot(tick)" :x2="xPlot(tick)" y1="35" y2="218" class="grid" />
          <line v-for="knot in result.knots" :key="`knot-${knot}`" :x1="xPlot(knot)" :x2="xPlot(knot)" y1="35" y2="218" class="knot" />
          <g clip-path="url(#basis-plot-clip)">
            <polyline :points="linePoints(gridX.map(truePerformance))" class="truth-line" />
            <polyline :points="linePoints(result.predictions)" class="fit-line" />
          </g>
          <circle v-for="(value,index) in trainX" :key="`train-${index}`" :cx="xPlot(value)" :cy="yPlot(trainY[index])" r="4.5" class="train-point" />
          <circle v-for="(value,index) in validX" :key="`valid-${index}`" :cx="xPlot(value)" :cy="yPlot(validY[index])" r="3.6" class="valid-point" />
          <text x="469" y="235" class="axis-label">室外湿球温度 / °C</text>
          <line x1="48" x2="570" y1="278" y2="278" class="contribution-zero" />
          <text x="48" y="252" class="panel-title">前五条 βⱼφⱼ(x) 通道</text>
          <polyline v-for="(values,index) in result.contributions" :key="`basis-${index}`" :points="contributionPoints(values)" :class="['basis-line', contributionColors[index]]" />
          <text x="48" y="307" class="footer-label">各通道按共同幅值比例绘制；琥珀总线是它们加截距后的和。</text>
        </svg>
        <figcaption>
          <span class="legend train" />拟合窗口
          <span class="legend valid" />冻结验证窗口
          <span class="legend fit" />当前基函数总线
          <span class="legend truth" />教学真曲线
          <span class="legend knot" />折点
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="family-switch" role="group" aria-label="基函数族">
        <button :class="{ active: family === 'polynomial' }" type="button" @click="family = 'polynomial'">全局幂基</button>
        <button :class="{ active: family === 'hinge' }" type="button" @click="family = 'hinge'">连续折线基</button>
      </div>
      <label>
        <span>{{ family === 'polynomial' ? '最高次数' : '内部折点数' }} <output>{{ complexity }}</output></span>
        <input v-model.number="complexity" type="range" min="1" max="9" step="1">
      </label>
      <label>
        <span>log₁₀ λ <output>{{ logLambda.toFixed(1) }}</output></span>
        <input v-model.number="logLambda" type="range" min="-4" max="2" step="0.1">
      </label>
      <p>高次幂会全局改变整条曲线；折线基只在越过折点后改变斜率。把复杂度升高并减弱 Ridge，观察训练误差继续下降时，验证误差和 32°C 边界误差是否同步。</p>
    </template>
    <template #metrics>
      <div><span>训练 RMSE</span><strong>{{ result.trainRmse.toFixed(3) }}</strong></div>
      <div><span>验证 RMSE</span><strong>{{ result.validRmse.toFixed(3) }}</strong></div>
      <div><span>有效自由度</span><strong>{{ result.effectiveDof.toFixed(2) }}</strong></div>
      <div><span>32°C 边界误差</span><strong>{{ result.boundaryError.toFixed(3) }}</strong></div>
    </template>
    <template #note>
      这里的连续折线基是截断幂表示 [1,z,(z−κ₁)₊,…]，便于直接看见“过一个折点多一段斜率”；工业实现常改用数值条件更好、具有局部支撑的 B-spline。两者都先生成设计矩阵，再交给 OLS/Ridge 求解。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.basis-console{margin:0}.basis-console svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:10px;background:#10182e}.console-bg{fill:#10182e}.grid{stroke:#263554;stroke-width:1}.knot{stroke:#e15b4b;stroke-width:1.5;stroke-dasharray:4 4;opacity:.68}.truth-line{fill:none;stroke:#8391b2;stroke-width:2;stroke-dasharray:7 5}.fit-line{fill:none;stroke:#f4c95d;stroke-width:4}.train-point{fill:#72d5ce;stroke:#fff;stroke-width:1.4}.valid-point{fill:#10182e;stroke:#e15b4b;stroke-width:2}.contribution-zero{stroke:#40527e}.basis-line{fill:none;stroke-width:2;opacity:.9}.basis-a{stroke:#72d5ce}.basis-b{stroke:#e15b4b}.basis-c{stroke:#a796d8}.basis-d{stroke:#68a4d9}.basis-e{stroke:#c9d46c}.basis-console text{font:700 10px ui-monospace,monospace}.panel-title{fill:#dfe5f7}.axis-label,.footer-label{fill:#8f9cbc}.basis-console figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}.legend.train{height:10px;border-radius:50%;background:#72d5ce}.legend.valid{height:10px;border:2px solid #e15b4b;border-radius:50%}.legend.fit{background:#f4c95d}.legend.truth{border-top:2px dashed #8391b2}.legend.knot{border-top:2px dashed #e15b4b}.family-switch{display:grid;grid-template-columns:1fr 1fr;gap:6px}.family-switch button{min-height:44px;border:1px solid #40527e;border-radius:7px;color:#b9c4df;background:#10182e;font:750 11px ui-monospace,monospace;cursor:pointer}.family-switch button.active{border-color:#72d5ce;color:#10182e;background:#72d5ce}.family-switch button:focus-visible{outline:3px solid #f4c95d;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}@media(max-width:560px){.basis-console{overflow-x:auto;overscroll-behavior-inline:contain}.basis-console svg{width:560px;max-width:none}.basis-console figcaption{min-width:560px}}
</style>
