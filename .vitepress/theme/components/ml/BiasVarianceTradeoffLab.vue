<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const degrees = [1, 3, 5, 9]
const degree = ref(3)
const sampleSize = ref(24)
const noise = ref(.18)
const probeX = .65
const repeats = 16
const grid = Array.from({ length: 81 }, (_, index) => -1 + 2 * index / 80)

function trueFunction(x: number) {
  return .52 + .20 * x + .31 * Math.sin(2.35 * x)
}

function makeRandom(seed: number) {
  let state = seed >>> 0
  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 4294967296
  }
}

function normal(random: () => number) {
  const first = Math.max(random(), 1e-12)
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * random())
}

function solve(matrix: number[][], vector: number[]) {
  const size = vector.length
  const augmented = matrix.map((row, index) => [...row, vector[index]])
  for (let column = 0; column < size; column += 1) {
    let pivot = column
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row
    }
    ;[augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]]
    const divisor = augmented[column][column]
    if (Math.abs(divisor) < 1e-12) return Array(size).fill(0)
    for (let item = column; item <= size; item += 1) augmented[column][item] /= divisor
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue
      const factor = augmented[row][column]
      for (let item = column; item <= size; item += 1) augmented[row][item] -= factor * augmented[column][item]
    }
  }
  return augmented.map((row) => row[size])
}

function polynomialFeatures(x: number, currentDegree: number) {
  return Array.from({ length: currentDegree + 1 }, (_, power) => x ** power)
}

function fitPolynomial(xs: number[], ys: number[], currentDegree: number) {
  const width = currentDegree + 1
  const gram = Array.from({ length: width }, () => Array(width).fill(0))
  const rhs = Array(width).fill(0)
  for (let index = 0; index < xs.length; index += 1) {
    const features = polynomialFeatures(xs[index], currentDegree)
    for (let row = 0; row < width; row += 1) {
      rhs[row] += features[row] * ys[index]
      for (let column = 0; column < width; column += 1) gram[row][column] += features[row] * features[column]
    }
  }
  for (let index = 1; index < width; index += 1) gram[index][index] += 1e-4
  return solve(gram, rhs)
}

function predict(coefficients: number[], x: number) {
  return coefficients.reduce((sum, coefficient, power) => sum + coefficient * x ** power, 0)
}

const experiment = computed(() => {
  const curves: number[][] = []
  const predictions: number[] = []
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    const random = makeRandom(17011 + repeat * 7919 + sampleSize.value * 31)
    const xs = Array.from({ length: sampleSize.value }, () => -1 + 2 * random())
    const ys = xs.map((x) => trueFunction(x) + noise.value * normal(random))
    const coefficients = fitPolynomial(xs, ys, degree.value)
    curves.push(grid.map((x) => predict(coefficients, x)))
    predictions.push(predict(coefficients, probeX))
  }
  const average = grid.map((_, index) => curves.reduce((sum, curve) => sum + curve[index], 0) / repeats)
  const meanPrediction = predictions.reduce((sum, value) => sum + value, 0) / repeats
  const biasSquared = (meanPrediction - trueFunction(probeX)) ** 2
  const variance = predictions.reduce((sum, value) => sum + (value - meanPrediction) ** 2, 0) / repeats
  return { curves, average, meanPrediction, biasSquared, variance, total: biasSquared + variance + noise.value ** 2 }
})

const xScale = (value: number) => 34 + (value + 1) / 2 * 532
const yScale = (value: number) => 282 - (Math.max(-.22, Math.min(1.28, value)) + .22) / 1.5 * 244
const path = (values: number[]) => values.map((value, index) => `${index === 0 ? 'M' : 'L'} ${xScale(grid[index]).toFixed(1)} ${yScale(value).toFixed(1)}`).join(' ')
const truePath = computed(() => path(grid.map(trueFunction)))
const averagePath = computed(() => path(experiment.value.average))
</script>

<template>
  <MlExperimentFrame
    index="03-A"
    title="复杂度如何重分配偏差与方差"
    question="对同一个真实过程反复抽取训练集，模型平均预测的系统偏离与单次拟合的波动怎样随复杂度改变？"
    formula="E[(Y-f_D)^2] = noise + bias^2 + variance"
  >
    <template #visual>
      <figure class="bv-map">
        <svg viewBox="0 0 600 320" role="img" :aria-label="`${degree} 次多项式在 ${repeats} 份训练集上的预测曲线；探针点平方偏差 ${experiment.biasSquared.toFixed(3)}，方差 ${experiment.variance.toFixed(3)}`">
          <rect class="plot-bg" x="20" y="16" width="560" height="280" rx="8" />
          <line class="axis" x1="34" x2="566" y1="282" y2="282" />
          <line class="axis" x1="34" x2="34" y1="28" y2="282" />
          <line class="probe" :x1="xScale(probeX)" :x2="xScale(probeX)" y1="28" y2="282" />
          <path v-for="(curve, index) in experiment.curves" :key="index" class="fit" :d="path(curve)" />
          <path class="truth" :d="truePath" />
          <path class="average" :d="averagePath" />
          <circle class="truth-point" :cx="xScale(probeX)" :cy="yScale(trueFunction(probeX))" r="5" />
          <circle class="average-point" :cx="xScale(probeX)" :cy="yScale(experiment.meanPrediction)" r="5" />
          <text x="42" y="38">16 份训练集 → 16 条拟合曲线</text>
          <text :x="xScale(probeX) + 7" y="54">探针 x₀</text>
          <text x="516" y="307">过程变量 x</text>
        </svg>
        <figcaption><span class="legend fit" />单次拟合　<span class="legend average" />平均预测　<span class="legend truth" />真实函数；虚线位置用于分解误差</figcaption>
      </figure>
      <table>
        <thead><tr><th>探针 x₀={{ probeX }}</th><th>预测</th></tr></thead>
        <tbody>
          <tr><td>真实函数 f(x₀)</td><td>{{ trueFunction(probeX).toFixed(3) }}</td></tr>
          <tr><td>平均拟合 E[f_D(x₀)]</td><td>{{ experiment.meanPrediction.toFixed(3) }}</td></tr>
        </tbody>
      </table>
    </template>
    <template #controls>
      <fieldset>
        <legend>多项式次数</legend>
        <div class="degree-switch">
          <button v-for="item in degrees" :key="item" type="button" :class="{ active: degree === item }" :aria-pressed="degree === item" @click="degree = item">{{ item }} 次</button>
        </div>
      </fieldset>
      <label>
        <span>每份训练集样本数 <output>{{ sampleSize }}</output></span>
        <input v-model.number="sampleSize" type="range" min="12" max="72" step="4">
      </label>
      <label>
        <span>标签噪声标准差 <output>{{ noise.toFixed(2) }}</output></span>
        <input v-model.number="noise" type="range" min=".05" max=".35" step=".02">
      </label>
    </template>
    <template #metrics>
      <div><span>平方偏差</span><strong>{{ experiment.biasSquared.toFixed(4) }}</strong></div>
      <div><span>预测方差</span><strong>{{ experiment.variance.toFixed(4) }}</strong></div>
      <div><span>噪声方差</span><strong>{{ (noise ** 2).toFixed(4) }}</strong></div>
      <div><span>三项之和</span><strong>{{ experiment.total.toFixed(4) }}</strong></div>
    </template>
    <template #note>
      先固定样本量比较 1 次与 9 次模型，再增加样本。平方偏差看平均预测离真实函数多远；方差看训练集重抽样后同一位置的预测有多不稳定。这里只在一个 x₀ 上用 16 次模拟近似期望，数值本身会有 Monte Carlo 误差。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.bv-map{margin:0}.bv-map svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:9px;background:#10182e}.plot-bg{fill:#10182e}.axis{stroke:#44547e}.probe{stroke:#f4c95d;stroke-dasharray:5 5}.bv-map path{fill:none}.bv-map .fit{stroke:#63739c;stroke-width:1;opacity:.45}.bv-map .truth{stroke:#72d5ce;stroke-width:3}.bv-map .average{stroke:#f4c95d;stroke-width:3}.truth-point{fill:#72d5ce;stroke:#fff;stroke-width:1.5}.average-point{fill:#f4c95d;stroke:#fff;stroke-width:1.5}.bv-map text{fill:#9eabc9;font:10px ui-monospace,monospace}.bv-map figcaption{margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.7}.legend{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}.legend.fit{background:#63739c}.legend.average{background:#f4c95d}.legend.truth{background:#72d5ce}.ml-experiment__visual table{width:100%;margin:10px 0 0;border-collapse:collapse;color:#c8d1e8;background:#10182e;font-size:10px}.ml-experiment__visual th,.ml-experiment__visual td{padding:7px;border:1px solid #2d3c68;text-align:right}.ml-experiment__visual th:first-child,.ml-experiment__visual td:first-child{text-align:left}.ml-experiment__controls fieldset{min-width:0;margin:0;padding:0;border:0}.ml-experiment__controls legend,.ml-experiment__controls label{color:#c8d1e8;font-size:11px}.degree-switch{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;margin-top:7px}.degree-switch button{min-height:44px;padding:6px 3px;border:1px solid #4d6095;border-radius:7px;color:#c8d1e8;background:#111a31;font-size:10px;font-weight:750;cursor:pointer}.degree-switch button.active{border-color:#f4c95d;color:#10182e;background:#f4c95d}.degree-switch button:focus-visible{outline:3px solid #72d5ce;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}@media(max-width:480px){.degree-switch{grid-template-columns:1fr 1fr}}
</style>
