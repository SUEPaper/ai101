<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type FunctionMode = 'smooth' | 'relu'
type AnalyticMode = 'correct' | 'bug'

const functionMode = ref<FunctionMode>('smooth')
const analyticMode = ref<AnalyticMode>('correct')
const parameter = ref(.7)
const epsilonExponent = ref(-5)

const lossAt = (value: number) => {
  if (functionMode.value === 'smooth') {
    const activation = Math.tanh(1.3 * value - .2)
    return .5 * (activation - .35) ** 2
  }
  const activation = Math.max(0, value)
  return .5 * (activation - .2) ** 2
}

const analyticAt = (value: number) => {
  if (functionMode.value === 'smooth') {
    const activation = Math.tanh(1.3 * value - .2)
    const base = (activation - .35) * (1 - activation ** 2)
    return analyticMode.value === 'correct' ? base * 1.3 : base
  }
  const activation = Math.max(0, value)
  if (analyticMode.value === 'bug') return activation - .2
  return value > 0 ? activation - .2 : 0
}

const numericAt = (value: number, epsilon: number) => (
  lossAt(value + epsilon) - lossAt(value - epsilon)
) / (2 * epsilon)

const relativeError = (analytic: number, numeric: number) => (
  Math.abs(analytic - numeric) / Math.max(1, Math.abs(analytic), Math.abs(numeric))
)

const result = computed(() => {
  const epsilon = 10 ** epsilonExponent.value
  const analytic = analyticAt(parameter.value)
  const numeric = numericAt(parameter.value, epsilon)
  const error = relativeError(analytic, numeric)
  const crossesKink = functionMode.value === 'relu' && Math.abs(parameter.value) <= epsilon
  const verdict = crossesKink
    ? '差分跨越 ReLU 折点'
    : error < 1e-7
      ? '梯度检查通过'
      : error < 1e-4
        ? '误差偏高，检查 ε'
        : '解析反向疑似错误'
  const sweep = Array.from({ length: 45 }, (_, index) => {
    const exponent = -12 + index * 11 / 44
    const currentEpsilon = 10 ** exponent
    const currentNumeric = numericAt(parameter.value, currentEpsilon)
    const currentError = Math.max(1e-16, relativeError(analytic, currentNumeric))
    return { exponent, error: currentError }
  })
  return { epsilon, analytic, numeric, error, crossesKink, verdict, sweep }
})

const mapX = (exponent: number) => 52 + (exponent + 12) / 11 * 495
const mapY = (error: number) => {
  const logError = Math.max(-16, Math.min(0, Math.log10(Math.max(1e-16, error))))
  return 188 - (logError + 16) / 16 * 132
}
const errorPath = computed(() => result.value.sweep.map((point, index) => (
  (index ? 'L ' : 'M ') + mapX(point.exponent) + ' ' + mapY(point.error)
)).join(' '))

const loadSmooth = () => {
  functionMode.value = 'smooth'
  analyticMode.value = 'correct'
  parameter.value = .7
  epsilonExponent.value = -5
}

const loadKink = () => {
  functionMode.value = 'relu'
  analyticMode.value = 'correct'
  parameter.value = 0
  epsilonExponent.value = -5
}
</script>

<template>
  <MlExperimentFrame
    index="14-C"
    title="用有限差分显微镜审计解析反向"
    question="ε 太大有截断误差、太小有消去误差；解析反向漏乘一个局部导数时，误差曲线会留下什么证据？"
    formula="g_num=[L(θ+ε)−L(θ−ε)]/(2ε)"
  >
    <template #visual>
      <figure class="gradient-microscope">
        <svg viewBox="0 0 600 326" role="img" :aria-label="'当前相对误差 '+result.error.toExponential(2)+'，'+result.verdict">
          <rect x="15" y="15" width="570" height="296" rx="9" class="microscope-bg" />
          <text x="52" y="33" class="panel-title">中心差分相对误差扫描</text>
          <text x="547" y="33" class="panel-title end">{{ functionMode==='smooth' ? '平滑 Tanh 单元' : 'ReLU 单元' }}</text>
          <rect x="52" y="48" width="495" height="151" rx="6" class="plot-bg" />
          <rect x="52" y="48" width="495" :height="mapY(1e-4)-48" class="reject-zone" />
          <line x1="52" x2="547" :y1="mapY(1e-4)" :y2="mapY(1e-4)" class="threshold-line" />
          <line x1="52" x2="547" :y1="mapY(1e-8)" :y2="mapY(1e-8)" class="grid-line" />
          <line x1="52" x2="547" :y1="mapY(1e-12)" :y2="mapY(1e-12)" class="grid-line" />
          <path :d="errorPath" :class="['error-path',{wrong:analyticMode==='bug'}]" />
          <circle
            :cx="mapX(epsilonExponent)"
            :cy="mapY(result.error)"
            r="6"
            :class="['probe-point',{wrong:result.error>=1e-4 || result.crossesKink}]"
          />
          <text x="43" :y="mapY(1e-4)+3" class="axis-label end">1e−4</text>
          <text x="43" :y="mapY(1e-8)+3" class="axis-label end">1e−8</text>
          <text x="43" :y="mapY(1e-12)+3" class="axis-label end">1e−12</text>
          <text x="52" y="216" class="axis-label">ε=1e−12</text>
          <text x="547" y="216" class="axis-label end">ε=1e−1</text>

          <rect x="52" y="236" width="151" height="51" rx="6" class="readout-box" />
          <rect x="224" y="236" width="151" height="51" rx="6" class="readout-box" />
          <rect x="396" y="236" width="151" height="51" rx="6" :class="['readout-box','verdict',{wrong:result.error>=1e-4 || result.crossesKink}]" />
          <text x="64" y="253" class="readout-label">解析梯度</text>
          <text x="64" y="275" class="readout-value">{{ result.analytic.toExponential(3) }}</text>
          <text x="236" y="253" class="readout-label">数值梯度</text>
          <text x="236" y="275" class="readout-value">{{ result.numeric.toExponential(3) }}</text>
          <text x="408" y="253" class="readout-label">审计状态</text>
          <text x="408" y="275" :class="['verdict-value',{wrong:result.error>=1e-4 || result.crossesKink}]">{{ result.verdict }}</text>
        </svg>
        <figcaption><span><i class="curve" />误差曲线</span><span><i class="probe" />当前 ε</span><span><i class="reject" />拒绝区</span><span><i class="pass" />可验收区</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="function-switch" role="group" aria-label="梯度检查函数">
        <button type="button" :aria-pressed="functionMode==='smooth'" @click="functionMode='smooth'">平滑 Tanh 单元</button>
        <button type="button" :aria-pressed="functionMode==='relu'" @click="functionMode='relu'">ReLU 单元</button>
      </div>
      <div class="analytic-switch" role="group" aria-label="解析反向实现">
        <button type="button" :aria-pressed="analyticMode==='correct'" @click="analyticMode='correct'">正确解析梯度</button>
        <button type="button" :aria-pressed="analyticMode==='bug'" @click="analyticMode='bug'">注入链式漏乘</button>
      </div>
      <label><span>参数 θ <output>{{ parameter.toFixed(2) }}</output></span><input v-model.number="parameter" type="range" min="-1.5" max="1.5" step=".05"></label>
      <label><span>ε 指数 <output>10^{{ epsilonExponent }}</output></span><input v-model.number="epsilonExponent" type="range" min="-12" max="-1" step="1"></label>
      <div class="sample-switch"><button type="button" @click="loadSmooth">装载平滑检查点</button><button type="button" @click="loadKink">装载 ReLU 折点</button></div>
      <p>通过阈值只适用于平滑点。若 [θ−ε, θ+ε] 跨越 ReLU 折点，中心差分平均了两侧斜率，不能用来裁决框架选取的单侧次梯度。</p>
    </template>
    <template #metrics>
      <div><span>当前 ε</span><strong>{{ result.epsilon.toExponential(1) }}</strong></div>
      <div><span>解析梯度</span><strong>{{ result.analytic.toExponential(3) }}</strong></div>
      <div><span>数值梯度</span><strong>{{ result.numeric.toExponential(3) }}</strong></div>
      <div><span>相对误差</span><strong>{{ result.error.toExponential(2) }}</strong></div>
    </template>
    <template #note>
      有限差分是实现审计器，不是训练算法。它需要两次前向才能检查一个坐标，复杂度随参数量线性增长；工业测试通常固定小网络、双精度和少量抽样坐标，并避开不可导点。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.gradient-microscope{margin:0}.gradient-microscope svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0b1722}.microscope-bg{fill:#0b1722}.gradient-microscope text{font:700 9px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.end{text-anchor:end}.plot-bg{fill:#101f2c}.reject-zone{fill:#e35d68;opacity:.08}.threshold-line{stroke:#e35d68;stroke-width:1.3;stroke-dasharray:5 4}.grid-line{stroke:#294354;stroke-width:1;stroke-dasharray:4 4}.error-path{fill:none;stroke:#63d0c6;stroke-width:2.4}.error-path.wrong{stroke:#e35d68}.probe-point{fill:#f3c85b;stroke:#fff;stroke-width:1.5}.probe-point.wrong{fill:#e35d68}.axis-label{fill:#829eaf}.readout-box{fill:#111e2e;stroke:#3d5870}.readout-box.verdict{stroke:#63d0c6}.readout-box.verdict.wrong{stroke:#e35d68}.readout-label{fill:#829eaf}.readout-value{fill:#f3c85b;font-size:11px}.verdict-value{fill:#63d0c6;font-size:8px}.verdict-value.wrong{fill:#e35d68}.gradient-microscope figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.gradient-microscope figcaption span{display:inline-flex;align-items:center;gap:5px}.gradient-microscope figcaption i{width:12px;height:8px;border-radius:2px}.gradient-microscope figcaption i.curve{height:2px;background:#63d0c6}.gradient-microscope figcaption i.probe{background:#f3c85b}.gradient-microscope figcaption i.reject{background:#e35d68}.gradient-microscope figcaption i.pass{border:1px solid #63d0c6}.function-switch,.analytic-switch,.sample-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.function-switch button,.analytic-switch button,.sample-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 9px ui-monospace,monospace;cursor:pointer}.function-switch button[aria-pressed="true"],.analytic-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.function-switch button:focus-visible,.analytic-switch button:focus-visible,.sample-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.gradient-microscope{overflow-x:auto;overscroll-behavior-inline:contain}.gradient-microscope svg{width:560px;max-width:none}.gradient-microscope figcaption{min-width:560px}}
</style>
