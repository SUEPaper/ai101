<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type InitMode = 'fixed' | 'xavier' | 'he'
type ActivationMode = 'relu' | 'tanh'

const initMode = ref<InitMode>('he')
const activation = ref<ActivationMode>('relu')
const depth = ref(20)
const width = ref(64)
const seed = ref(13)

const createRandom = (initialSeed: number) => {
  let state = initialSeed >>> 0
  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return (state + .5) / 4294967296
  }
}

const createNormal = (random: () => number) => {
  let spare: number | null = null
  return () => {
    if (spare !== null) {
      const value = spare
      spare = null
      return value
    }
    const radius = Math.sqrt(-2 * Math.log(Math.max(1e-12, random())))
    const angle = 2 * Math.PI * random()
    spare = radius * Math.sin(angle)
    return radius * Math.cos(angle)
  }
}

const result = computed(() => {
  const random = createRandom(seed.value)
  const normal = createNormal(random)
  const batchSize = 8
  const fan = width.value
  let signals = Array.from({ length: batchSize }, () => Array.from({ length: fan }, () => normal()))
  const standardDeviation = initMode.value === 'fixed'
    ? .2
    : initMode.value === 'xavier'
      ? Math.sqrt(1 / fan)
      : Math.sqrt(2 / fan)
  const layers = []
  for (let layer = 1; layer <= depth.value; layer++) {
    const weights = Array.from({ length: fan }, () => Array.from({ length: fan }, () => standardDeviation * normal()))
    const preactivations = signals.map((row) => weights.map((weightRow) => weightRow.reduce((sum, weight, index) => sum + weight * row[index], 0)))
    signals = preactivations.map((row) => row.map((value) => activation.value === 'relu' ? Math.max(0, value) : Math.tanh(value)))
    const flat = signals.flat()
    const rms = Math.sqrt(flat.reduce((sum, value) => sum + value ** 2, 0) / flat.length)
    const mean = flat.reduce((sum, value) => sum + value, 0) / flat.length
    const inactiveRatio = activation.value === 'relu'
      ? flat.filter((value) => value === 0).length / flat.length
      : flat.filter((value) => Math.abs(value) > .98).length / flat.length
    layers.push({ layer, rms, mean, inactiveRatio })
  }
  const rmsValues = layers.map((layer) => layer.rms)
  const final = layers.at(-1) ?? { rms: 0, inactiveRatio: 0, mean: 0 }
  const maximumRms = Math.max(...rmsValues)
  const minimumRms = Math.min(...rmsValues)
  const state = maximumRms > 20
    ? '信号爆炸'
    : final.rms < .03
      ? '信号衰减'
      : activation.value === 'tanh' && final.inactiveRatio > .55
        ? 'Tanh 大面积饱和'
        : '信号量级可用'
  return { layers, standardDeviation, final, maximumRms, minimumRms, state }
})

const layerX = (index: number) => 54 + index / Math.max(1, depth.value - 1) * 495
const rmsY = (rms: number) => {
  const logValue = Math.max(-3, Math.min(3, Math.log10(Math.max(1e-4, rms))))
  return 169 - logValue / 3 * 104
}
const inactivityHeight = (ratio: number) => ratio * 62
const rmsPath = computed(() => result.value.layers.map((layer, index) => `${index ? 'L' : 'M'} ${layerX(index)} ${rmsY(layer.rms)}`).join(' '))

const resample = () => {
  seed.value += 1
}
</script>

<template>
  <MlExperimentFrame
    index="13-C"
    title="让随机信号穿过初始化隧道"
    question="固定尺度、Xavier 与 He 初始化分别会怎样改变深层激活的 RMS、失活率和可用动态范围？"
    formula="Var(z)=fan_in·Var(w)·E[h²]"
  >
    <template #visual>
      <figure class="signal-tunnel">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`当前 ${initMode} 初始化，${activation} 激活，末层 RMS ${result.final.rms.toFixed(3)}，状态 ${result.state}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="tunnel-bg" />
          <text x="54" y="31" class="panel-title">逐层激活 RMS / 对数刻度</text>
          <text x="54" y="216" class="panel-title">{{ activation === 'relu' ? '零激活占比' : '|tanh(h)|>0.98 饱和占比' }}</text>
          <rect x="54" y="48" width="495" height="139" rx="6" class="plot-bg" />
          <rect x="54" y="48" width="495" height="32" class="danger-zone exploding" />
          <rect x="54" y="155" width="495" height="32" class="danger-zone vanishing" />
          <line x1="54" x2="549" :y1="rmsY(10)" :y2="rmsY(10)" class="grid-line" />
          <line x1="54" x2="549" :y1="rmsY(1)" :y2="rmsY(1)" class="unit-line" />
          <line x1="54" x2="549" :y1="rmsY(.1)" :y2="rmsY(.1)" class="grid-line" />
          <path :d="rmsPath" :class="['rms-path', result.state === '信号量级可用' ? 'safe' : 'danger']" />
          <g v-for="(layer,index) in result.layers" :key="layer.layer">
            <circle :cx="layerX(index)" :cy="rmsY(layer.rms)" r="3.5" :class="['layer-node', result.state === '信号量级可用' ? 'safe' : 'danger']" />
            <rect :x="layerX(index)-4" :y="291-inactivityHeight(layer.inactiveRatio)" width="8" :height="inactivityHeight(layer.inactiveRatio)" class="inactive-bar" />
          </g>
          <line x1="54" x2="549" y1="291" y2="291" class="occupancy-axis" />
          <text x="47" :y="rmsY(10)+3" class="axis-label end">10</text>
          <text x="47" :y="rmsY(1)+3" class="axis-label end">1</text>
          <text x="47" :y="rmsY(.1)+3" class="axis-label end">0.1</text>
          <text x="54" y="309" class="axis-label">L1</text>
          <text x="549" y="309" class="axis-label end">L{{ depth }}</text>
          <text x="549" y="45" :class="['status-label', result.state === '信号量级可用' ? 'safe' : 'danger']">{{ result.state }}</text>
        </svg>
        <figcaption><span><i class="safe" />可用量级</span><span><i class="danger" />爆炸或衰减</span><span><i class="inactive" />失活/饱和比例</span><span><i class="unit" />RMS=1 基线</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="init-switch" role="group" aria-label="权重初始化">
        <button type="button" :aria-pressed="initMode === 'fixed'" @click="initMode='fixed'">固定 σ=.20</button>
        <button type="button" :aria-pressed="initMode === 'xavier'" @click="initMode='xavier'">Xavier</button>
        <button type="button" :aria-pressed="initMode === 'he'" @click="initMode='he'">He</button>
      </div>
      <div class="activation-switch" role="group" aria-label="隧道激活函数"><button type="button" :aria-pressed="activation === 'relu'" @click="activation='relu'">ReLU</button><button type="button" :aria-pressed="activation === 'tanh'" @click="activation='tanh'">Tanh</button></div>
      <label><span>网络深度 L <output>{{ depth }}</output></span><input v-model.number="depth" type="range" min="4" max="30" step="1"></label>
      <label><span>每层宽度 fan_in <output>{{ width }}</output></span><input v-model.number="width" type="range" min="16" max="128" step="16"></label>
      <button class="resample" type="button" @click="resample">重采样随机权重 · #{{ seed }}</button>
      <p>隧道用 8 条独立输入、等宽全连接层和固定随机种子做前向试验。它是初始化诊断，不是方差守恒的数学证明，也不包含训练后的权重漂移。</p>
    </template>
    <template #metrics>
      <div><span>权重标准差</span><strong>{{ result.standardDeviation.toFixed(4) }}</strong></div>
      <div><span>末层激活 RMS</span><strong>{{ result.final.rms.toExponential(2) }}</strong></div>
      <div><span>{{ activation === 'relu' ? '末层零激活' : '末层饱和' }}</span><strong>{{ (result.final.inactiveRatio*100).toFixed(1) }}%</strong></div>
      <div><span>传播状态</span><strong>{{ result.state }}</strong></div>
    </template>
    <template #note>
      Xavier 与 He 都来自近似二阶矩账本：前者适合近线性的零中心激活，后者补偿 ReLU 丢掉的负半轴。真实深网还受相关性、残差连接、归一化和训练动力学影响，不能把初始化公式当成永久守恒律。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.signal-tunnel{margin:0}.signal-tunnel svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0c1925}.tunnel-bg{fill:#0c1925}.signal-tunnel text{font:700 9px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.plot-bg{fill:#101f2c}.danger-zone{fill:#e35d68;opacity:.07}.danger-zone.vanishing{fill:#6370d0}.grid-line{stroke:#294354;stroke-width:1;stroke-dasharray:4 4}.unit-line{stroke:#f3c85b;stroke-width:1.5;stroke-dasharray:6 4}.rms-path{fill:none;stroke-width:2.5}.rms-path.safe{stroke:#63d0c6}.rms-path.danger{stroke:#e35d68}.layer-node{stroke:#fff;stroke-width:1}.layer-node.safe{fill:#63d0c6}.layer-node.danger{fill:#e35d68}.inactive-bar{fill:#f08a55;opacity:.72}.occupancy-axis{stroke:#355165;stroke-width:1.5}.axis-label{fill:#829eaf}.end{text-anchor:end}.status-label{text-anchor:end;font-size:10px}.status-label.safe{fill:#63d0c6}.status-label.danger{fill:#e35d68}.signal-tunnel figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.signal-tunnel figcaption span{display:inline-flex;align-items:center;gap:5px}.signal-tunnel figcaption i{width:12px;height:8px;border-radius:2px}.signal-tunnel figcaption i.safe{background:#63d0c6}.signal-tunnel figcaption i.danger{background:#e35d68}.signal-tunnel figcaption i.inactive{background:#f08a55}.signal-tunnel figcaption i.unit{height:2px;background:#f3c85b}.init-switch{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.activation-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.init-switch button,.activation-switch button,.resample{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 9px ui-monospace,monospace;cursor:pointer}.init-switch button[aria-pressed="true"],.activation-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.init-switch button:focus-visible,.activation-switch button:focus-visible,.resample:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.signal-tunnel{overflow-x:auto;overscroll-behavior-inline:contain}.signal-tunnel svg{width:560px;max-width:none}.signal-tunnel figcaption{min-width:560px}}
</style>
