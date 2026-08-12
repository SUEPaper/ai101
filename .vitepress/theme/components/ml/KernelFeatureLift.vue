<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type KernelMode = 'polynomial' | 'rbf'

const mode = ref<KernelMode>('polynomial')
const degree = ref(2)
const gamma = ref(0.8)
const probeRadius = ref(1.15)

const angles = Array.from({ length: 6 }, (_, index) => index * Math.PI / 3)
const points = [
  ...angles.map((angle, index) => ({
    x: (0.58 + 0.04 * (index % 3)) * Math.cos(angle),
    y: (0.58 + 0.04 * (index % 3)) * Math.sin(angle),
    label: -1,
    id: `N${index + 1}`,
  })),
  ...angles.map((angle, index) => ({
    x: (1.62 + 0.08 * (index % 3)) * Math.cos(angle + Math.PI / 6),
    y: (1.62 + 0.08 * (index % 3)) * Math.sin(angle + Math.PI / 6),
    label: 1,
    id: `P${index + 1}`,
  })),
]

const jacobiEigenvalues = (input: number[][]) => {
  const matrix = input.map((row) => [...row])
  const size = matrix.length
  for (let sweep = 0; sweep < 100 * size * size; sweep++) {
    let row = 0
    let column = 1
    let maximum = 0
    for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) {
        if (Math.abs(matrix[i][j]) > maximum) {
          maximum = Math.abs(matrix[i][j])
          row = i
          column = j
        }
      }
    }
    if (maximum < 1e-11) break
    const angle = 0.5 * Math.atan2(
      2 * matrix[row][column],
      matrix[column][column] - matrix[row][row],
    )
    const cosine = Math.cos(angle)
    const sine = Math.sin(angle)
    const diagonalRow = matrix[row][row]
    const diagonalColumn = matrix[column][column]
    const offDiagonal = matrix[row][column]
    matrix[row][row] = cosine ** 2 * diagonalRow - 2 * sine * cosine * offDiagonal + sine ** 2 * diagonalColumn
    matrix[column][column] = sine ** 2 * diagonalRow + 2 * sine * cosine * offDiagonal + cosine ** 2 * diagonalColumn
    matrix[row][column] = 0
    matrix[column][row] = 0
    for (let index = 0; index < size; index++) {
      if (index === row || index === column) continue
      const rowValue = matrix[index][row]
      const columnValue = matrix[index][column]
      matrix[index][row] = cosine * rowValue - sine * columnValue
      matrix[row][index] = matrix[index][row]
      matrix[index][column] = sine * rowValue + cosine * columnValue
      matrix[column][index] = matrix[index][column]
    }
  }
  return matrix.map((row, index) => row[index]).sort((left, right) => left - right)
}

const result = computed(() => {
  const probe = { x: probeRadius.value, y: 0 }
  const kernel = (left: { x: number; y: number }, right: { x: number; y: number }) => {
    const dot = left.x * right.x + left.y * right.y
    if (mode.value === 'polynomial') return (1 + dot) ** degree.value
    const squaredDistance = (left.x - right.x) ** 2 + (left.y - right.y) ** 2
    return Math.exp(-gamma.value * squaredDistance)
  }
  const gram = points.map((left) => points.map((right) => kernel(left, right)))
  const eigenvalues = jacobiEigenvalues(gram)
  const similarities = points.map((point) => kernel(probe, point))
  const maximumSimilarity = Math.max(...similarities.map((value) => Math.abs(value)), 1e-12)
  const lifted = points.map((point, index) => ({
    ...point,
    coordinate: mode.value === 'polynomial'
      ? degree.value === 1 ? point.x : point.x ** 2 + point.y ** 2
      : similarities[index] / maximumSimilarity,
  }))
  const symmetryError = Math.max(...gram.flatMap((row, i) => row.map((value, j) => Math.abs(value - gram[j][i]))))
  const influenceRadius = Math.sqrt(Math.log(2) / gamma.value)
  return {
    gram,
    eigenvalues,
    lifted,
    similarities,
    minimumEigenvalue: eigenvalues[0],
    maximumEigenvalue: eigenvalues.at(-1) ?? 0,
    symmetryError,
    influenceRadius,
    route: mode.value === 'polynomial'
      ? degree.value === 1 ? '原始线性坐标' : '显式含 r² 的提升'
      : '隐式局部相似度',
  }
})

const mapInputX = (value: number) => 42 + (value + 2.2) / 4.4 * 260
const mapInputY = (value: number) => 280 - (value + 2.0) / 4.0 * 220
const mapLiftX = (value: number) => {
  if (mode.value === 'polynomial') {
    const maximum = degree.value === 1 ? 2.2 : 3.6
    const minimum = degree.value === 1 ? -2.2 : 0
    return 348 + (value - minimum) / (maximum - minimum) * 205
  }
  return 348 + value * 205
}
const mapLiftY = (label: number, index: number) => label === 1 ? 105 + (index % 3) * 10 : 224 + (index % 3) * 10
</script>

<template>
  <MlExperimentFrame
    index="12-B"
    title="把环状终检样本送过核特征升降台"
    question="原空间没有直线能分开的两圈样本，何时能在显式提升坐标中线性分开，又怎样审计隐式 RBF 相似度？"
    formula="Kᵢⱼ=φ(xᵢ)ᵀφ(xⱼ),  cᵀKc≥0"
  >
    <template #visual>
      <figure class="kernel-lift">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`当前 ${result.route}，Gram 最小特征值 ${result.minimumEigenvalue.toExponential(1)}`">
          <defs>
            <clipPath id="kernel-input-clip"><rect x="42" y="60" width="260" height="220" rx="5" /></clipPath>
          </defs>
          <rect x="15" y="15" width="570" height="296" rx="9" class="lift-bg" />
          <text x="42" y="31" class="panel-title">原始检验平面</text>
          <text x="348" y="31" class="panel-title">{{ mode === 'polynomial' ? '提升后坐标轨' : '对探针的核相似度轨' }}</text>
          <rect x="42" y="60" width="260" height="220" rx="5" class="plot-bg" />
          <g class="grid">
            <line v-for="value in [-1,0,1]" :key="`gx-${value}`" :x1="mapInputX(value)" :x2="mapInputX(value)" y1="60" y2="280" />
            <line v-for="value in [-1,0,1]" :key="`gy-${value}`" x1="42" x2="302" :y1="mapInputY(value)" :y2="mapInputY(value)" />
          </g>
          <g clip-path="url(#kernel-input-clip)">
            <circle v-if="mode === 'polynomial' && degree >= 2" :cx="mapInputX(0)" :cy="mapInputY(0)" :r="(mapInputX(1.15)-mapInputX(0))" class="radial-boundary" />
            <circle v-if="mode === 'rbf'" :cx="mapInputX(probeRadius)" :cy="mapInputY(0)" :r="(mapInputX(result.influenceRadius)-mapInputX(0))" class="probe-halo" />
            <g v-for="(point,index) in points" :key="point.id">
              <line
                v-if="mode === 'rbf'"
                :x1="mapInputX(probeRadius)"
                :y1="mapInputY(0)"
                :x2="mapInputX(point.x)"
                :y2="mapInputY(point.y)"
                class="similarity-ray"
                :style="{ opacity: Math.min(0.75, 0.08 + Math.abs(result.similarities[index]) * 0.55) }"
              />
              <circle :cx="mapInputX(point.x)" :cy="mapInputY(point.y)" r="6" :class="point.label === 1 ? 'positive-point' : 'negative-point'" />
            </g>
            <circle v-if="mode === 'rbf'" :cx="mapInputX(probeRadius)" :cy="mapInputY(0)" r="8" class="probe-point" />
          </g>
          <path d="M 313 166 L 335 166" class="lift-arrow" />

          <line x1="348" x2="553" y1="105" y2="105" class="rail positive" />
          <line x1="348" x2="553" y1="244" y2="244" class="rail negative" />
          <line v-if="mode === 'polynomial' && degree >= 2" :x1="mapLiftX(1.5)" :x2="mapLiftX(1.5)" y1="70" y2="270" class="lift-separator" />
          <g v-for="(point,index) in result.lifted" :key="`lift-${point.id}`">
            <circle :cx="mapLiftX(point.coordinate)" :cy="mapLiftY(point.label,index)" r="6" :class="point.label === 1 ? 'positive-point' : 'negative-point'" />
          </g>
          <text x="348" y="93" class="rail-label">送复检 +1</text>
          <text x="348" y="232" class="rail-label">自动放行 −1</text>
          <text x="348" y="292" class="audit-label">λmin(K)</text>
          <text x="553" y="292" :class="['audit-value', result.minimumEigenvalue > -1e-7 ? 'safe' : 'danger']">{{ result.minimumEigenvalue.toExponential(1) }}</text>
          <text x="348" y="310" class="audit-label">对称误差</text>
          <text x="553" y="310" class="audit-value safe">{{ result.symmetryError.toExponential(1) }}</text>
        </svg>
        <figcaption><span><i class="negative" />中心合格圈</span><span><i class="positive" />外侧复检圈</span><span><i class="boundary" />提升后的线性阈值</span><span><i class="probe" />RBF 探针半高宽</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="kernel-switch" role="group" aria-label="核函数类型">
        <button type="button" :aria-pressed="mode === 'polynomial'" @click="mode='polynomial'">多项式核</button>
        <button type="button" :aria-pressed="mode === 'rbf'" @click="mode='rbf'">RBF 核</button>
      </div>
      <label>
        <span>多项式次数 d <output>{{ degree }}</output></span>
        <input v-model.number="degree" type="range" min="1" max="4" step="1">
      </label>
      <label>
        <span>RBF γ <output>{{ gamma.toFixed(2) }}</output></span>
        <input v-model.number="gamma" type="range" min="0.10" max="3" step="0.05">
      </label>
      <label>
        <span>探针半径 <output>{{ probeRadius.toFixed(2) }}</output></span>
        <input v-model.number="probeRadius" type="range" min="0.20" max="2.10" step="0.05">
      </label>
      <p>次数 1 只保留原始线性坐标，无法用单一阈值拆开两圈；次数至少 2 时提升空间含 r²。RBF 不显式构造无限维向量，而用 γ 控制局部相似度的衰减范围。</p>
    </template>
    <template #metrics>
      <div><span>当前表示</span><strong>{{ result.route }}</strong></div>
      <div><span>Gram 最小特征值</span><strong>{{ result.minimumEigenvalue.toExponential(1) }}</strong></div>
      <div><span>Gram 最大特征值</span><strong>{{ result.maximumEigenvalue.toExponential(1) }}</strong></div>
      <div><span>RBF 半相似半径</span><strong>{{ result.influenceRadius.toFixed(3) }}</strong></div>
    </template>
    <template #note>
      在这一组有限样本上最小特征值非负是必要审计，但不能单独证明一个自定义函数对所有可能样本都有效。核函数、缩放、d、γ 与 C 必须作为同一个特征空间合同联合选择。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.kernel-lift{margin:0}.kernel-lift svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0c1925}.lift-bg{fill:#0c1925}.kernel-lift text{font:700 9px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.plot-bg{fill:#101f2c}.grid line{stroke:#213b4d;stroke-width:1}.negative-point{fill:#63d0c6;stroke:#dbeaf0;stroke-width:1.5}.positive-point{fill:#e35d68;stroke:#dbeaf0;stroke-width:1.5}.radial-boundary{fill:none;stroke:#f3c85b;stroke-width:2;stroke-dasharray:7 5}.probe-halo{fill:#f3c85b;fill-opacity:.07;stroke:#f3c85b;stroke-width:1.5;stroke-dasharray:5 4}.probe-point{fill:#f3c85b;stroke:#fff;stroke-width:2}.similarity-ray{stroke:#f3c85b;stroke-width:1}.lift-arrow{fill:none;stroke:#829eaf;stroke-width:2}.rail{stroke:#355165;stroke-width:2}.rail.positive{stroke:#6a3c4a}.rail.negative{stroke:#2d5d5b}.lift-separator{stroke:#f3c85b;stroke-width:2;stroke-dasharray:6 4}.rail-label,.audit-label{fill:#829eaf}.audit-value{fill:#dbeaf0;text-anchor:end}.audit-value.safe{fill:#63d0c6}.audit-value.danger{fill:#e35d68}.kernel-lift figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.kernel-lift figcaption span{display:inline-flex;align-items:center;gap:5px}.kernel-lift figcaption i{width:12px;height:8px;border-radius:2px}.kernel-lift figcaption i.negative{background:#63d0c6}.kernel-lift figcaption i.positive{background:#e35d68}.kernel-lift figcaption i.boundary{height:3px;background:#f3c85b}.kernel-lift figcaption i.probe{border:2px solid #f3c85b;background:transparent}.kernel-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.kernel-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 11px ui-monospace,monospace;cursor:pointer}.kernel-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.kernel-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.kernel-lift{overflow-x:auto;overscroll-behavior-inline:contain}.kernel-lift svg{width:560px;max-width:none}.kernel-lift figcaption{min-width:560px}}
</style>
