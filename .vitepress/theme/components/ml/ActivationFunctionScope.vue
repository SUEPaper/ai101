<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type ActivationMode = 'sigmoid' | 'tanh' | 'relu' | 'leaky' | 'gelu'

const mode = ref<ActivationMode>('relu')
const gain = ref(1)
const bias = ref(0)
const probe = ref(-0.8)

const modes: Array<{ key: ActivationMode; label: string }> = [
  { key: 'sigmoid', label: 'Sigmoid' },
  { key: 'tanh', label: 'Tanh' },
  { key: 'relu', label: 'ReLU' },
  { key: 'leaky', label: 'Leaky' },
  { key: 'gelu', label: 'GELU' },
]

const sigmoid = (z: number) => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z))

const activate = (z: number) => {
  if (mode.value === 'sigmoid') return sigmoid(z)
  if (mode.value === 'tanh') return Math.tanh(z)
  if (mode.value === 'relu') return Math.max(0, z)
  if (mode.value === 'leaky') return z >= 0 ? z : 0.1 * z
  const inner = Math.sqrt(2 / Math.PI) * (z + 0.044715 * z ** 3)
  return 0.5 * z * (1 + Math.tanh(inner))
}

const derivative = (z: number) => {
  if (mode.value === 'sigmoid') {
    const value = sigmoid(z)
    return value * (1 - value)
  }
  if (mode.value === 'tanh') return 1 - Math.tanh(z) ** 2
  if (mode.value === 'relu') return z > 0 ? 1 : 0
  if (mode.value === 'leaky') return z >= 0 ? 1 : 0.1
  const constant = Math.sqrt(2 / Math.PI)
  const inner = constant * (z + 0.044715 * z ** 3)
  const tanhInner = Math.tanh(inner)
  return 0.5 * (1 + tanhInner) + 0.5 * z * (1 - tanhInner ** 2) * constant * (1 + 3 * 0.044715 * z ** 2)
}

const result = computed(() => {
  const z = gain.value * probe.value + bias.value
  const points = Array.from({ length: 161 }, (_, index) => {
    const x = -4 + index * 8 / 160
    return { x, y: activate(x), derivative: derivative(x) }
  })
  const occupancy = [-2.4, -1.8, -1.3, -.9, -.5, -.15, .15, .5, .9, 1.3, 1.8, 2.4]
    .map((input) => {
      const preactivation = gain.value * input + bias.value
      return { input, preactivation, output: activate(preactivation), derivative: derivative(preactivation) }
    })
  const blocked = occupancy.filter((point) => Math.abs(point.derivative) < .05).length
  const negativeBlocked = occupancy.filter((point) => point.preactivation < 0 && Math.abs(point.derivative) < .05).length
  return {
    z,
    output: activate(z),
    derivative: derivative(z),
    points,
    occupancy,
    blockedRatio: blocked / occupancy.length,
    negativeBlocked,
    state: Math.abs(derivative(z)) < .05 ? '局部信号近乎阻断' : '局部梯度可传递',
  }
})

const yRange = computed(() => mode.value === 'sigmoid' ? [-.25, 1.25] : mode.value === 'tanh' ? [-1.35, 1.35] : [-1.4, 4.4])
const mapX = (value: number) => 48 + (Math.max(-4, Math.min(4, value)) + 4) / 8 * 510
const mapY = (value: number) => {
  const clamped = Math.max(yRange.value[0], Math.min(yRange.value[1], value))
  return 211 - (clamped - yRange.value[0]) / (yRange.value[1] - yRange.value[0]) * 157
}
const mapDerivativeY = (value: number) => 291 - Math.max(-.2, Math.min(1.2, value)) / 1.2 * 58

const curvePath = computed(() => result.value.points.map((point, index) => `${index ? 'L' : 'M'} ${mapX(point.x)} ${mapY(point.y)}`).join(' '))
const derivativePath = computed(() => result.value.points.map((point, index) => `${index ? 'L' : 'M'} ${mapX(point.x)} ${mapDerivativeY(point.derivative)}`).join(' '))
</script>

<template>
  <MlExperimentFrame
    index="13-B"
    title="在激活函数示波器上寻找信号通带"
    question="同一批预激活值经过不同非线性后，哪些区间保留变化，哪些区间把局部敏感度压到接近零？"
    formula="z=gx+b,  h=φ(z),  local gain=φ′(z)"
  >
    <template #visual>
      <figure class="activation-scope">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`当前 ${mode}，预激活 ${result.z.toFixed(2)}，局部导数 ${result.derivative.toFixed(3)}`">
          <defs><clipPath id="activation-curve-clip"><rect x="48" y="46" width="510" height="168" rx="5" /></clipPath></defs>
          <rect x="15" y="15" width="570" height="296" rx="9" class="scope-bg" />
          <text x="48" y="31" class="panel-title">φ(z) 输出轨</text>
          <text x="48" y="229" class="panel-title">φ′(z) 局部增益轨</text>
          <rect x="48" y="46" width="510" height="168" rx="5" class="plot-bg" />
          <g clip-path="url(#activation-curve-clip)">
            <rect :x="mapX(-4)" y="46" :width="mapX(-2.5)-mapX(-4)" height="168" class="saturation-zone" />
            <rect :x="mapX(2.5)" y="46" :width="mapX(4)-mapX(2.5)" height="168" class="saturation-zone" />
            <line v-for="value in [-3,-2,-1,0,1,2,3]" :key="`gx-${value}`" :x1="mapX(value)" :x2="mapX(value)" y1="46" y2="214" class="grid-line" />
            <line x1="48" x2="558" :y1="mapY(0)" :y2="mapY(0)" class="zero-line" />
            <path :d="curvePath" class="activation-curve" />
            <circle v-for="(point,index) in result.occupancy" :key="index" :cx="mapX(Math.max(-4,Math.min(4,point.preactivation)))" :cy="mapY(point.output)" r="3.5" :class="Math.abs(point.derivative)<.05 ? 'blocked-point' : 'open-point'" />
            <line :x1="mapX(result.z)" :x2="mapX(result.z)" y1="46" y2="214" class="probe-line" />
            <circle :cx="mapX(result.z)" :cy="mapY(result.output)" r="7" class="probe-point" />
          </g>
          <line x1="48" x2="558" y1="291" y2="291" class="derivative-axis" />
          <line x1="48" x2="558" :y1="mapDerivativeY(.05)" :y2="mapDerivativeY(.05)" class="blocked-threshold" />
          <path :d="derivativePath" class="derivative-curve" />
          <line :x1="mapX(result.z)" :x2="mapX(result.z)" y1="233" y2="291" class="probe-line" />
          <circle :cx="mapX(result.z)" :cy="mapDerivativeY(result.derivative)" r="6" class="derivative-probe" />
          <text x="48" y="309" class="axis-label">−4</text><text x="303" y="309" class="axis-label center">预激活 z</text><text x="558" y="309" class="axis-label end">+4</text>
          <text x="552" y="61" class="scope-readout end">h={{ result.output.toFixed(3) }}</text>
          <text x="552" y="247" class="scope-readout end">φ′={{ result.derivative.toFixed(3) }}</text>
        </svg>
        <figcaption><span><i class="curve" />激活输出</span><span><i class="derivative" />局部导数</span><span><i class="open" />通带样本</span><span><i class="blocked" />近阻断样本</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="activation-switch" role="group" aria-label="激活函数">
        <button v-for="item in modes" :key="item.key" type="button" :aria-pressed="mode === item.key" @click="mode=item.key">{{ item.label }}</button>
      </div>
      <label><span>输入增益 g <output>{{ gain.toFixed(2) }}</output></span><input v-model.number="gain" type="range" min="0.25" max="3" step="0.05"></label>
      <label><span>偏置 b <output>{{ bias.toFixed(2) }}</output></span><input v-model.number="bias" type="range" min="-2" max="2" step="0.05"></label>
      <label><span>当前探针 x <output>{{ probe.toFixed(2) }}</output></span><input v-model.number="probe" type="range" min="-3" max="3" step="0.05"></label>
      <p>彩点是一批固定输入经 g 与 b 变换后的工作位置。增益过大或偏置整体推向一侧，会让 Sigmoid/Tanh 落入饱和区，也可能让 ReLU 大量进入负半轴。</p>
    </template>
    <template #metrics>
      <div><span>当前预激活 z</span><strong>{{ result.z.toFixed(3) }}</strong></div>
      <div><span>激活输出 h</span><strong>{{ result.output.toFixed(3) }}</strong></div>
      <div><span>局部导数 φ′</span><strong>{{ result.derivative.toFixed(3) }}</strong></div>
      <div><span>近阻断样本占比</span><strong>{{ (result.blockedRatio*100).toFixed(1) }}%</strong></div>
    </template>
    <template #note>
      导数只描述当前点附近的局部敏感度，不等于整个网络的梯度。深层梯度还要乘过后续层的 Jacobian；第 14 章会把这条链完整展开。本实验在 ReLU 的 z=0 处采用导数 0 的实现约定。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.activation-scope{margin:0}.activation-scope svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0c1925}.scope-bg{fill:#0c1925}.activation-scope text{font:700 9px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.plot-bg{fill:#101f2c}.saturation-zone{fill:#e35d68;opacity:.07}.grid-line{stroke:#213b4d;stroke-width:1}.zero-line,.derivative-axis{stroke:#355165;stroke-width:1.5}.activation-curve{fill:none;stroke:#63d0c6;stroke-width:3}.derivative-curve{fill:none;stroke:#f08a55;stroke-width:2.5}.blocked-threshold{stroke:#e35d68;stroke-width:1;stroke-dasharray:4 4}.probe-line{stroke:#f3c85b;stroke-width:1.3;stroke-dasharray:5 4}.probe-point{fill:#f3c85b;stroke:#fff;stroke-width:1.5}.derivative-probe{fill:#f08a55;stroke:#fff;stroke-width:1.5}.open-point{fill:#63d0c6;stroke:#fff;stroke-width:1}.blocked-point{fill:#e35d68;stroke:#fff;stroke-width:1}.axis-label{fill:#829eaf}.center{text-anchor:middle}.end{text-anchor:end}.scope-readout{fill:#f3c85b}.activation-scope figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.activation-scope figcaption span{display:inline-flex;align-items:center;gap:5px}.activation-scope figcaption i{width:12px;height:8px;border-radius:2px}.activation-scope figcaption i.curve{height:3px;background:#63d0c6}.activation-scope figcaption i.derivative{height:3px;background:#f08a55}.activation-scope figcaption i.open{background:#63d0c6}.activation-scope figcaption i.blocked{background:#e35d68}.activation-switch{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.activation-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 9px ui-monospace,monospace;cursor:pointer}.activation-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.activation-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.activation-scope{overflow-x:auto;overscroll-behavior-inline:contain}.activation-scope svg{width:560px;max-width:none}.activation-scope figcaption{min-width:560px}}
</style>
