<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type UpstreamMode = 'uniform' | 'center' | 'edge'
type GradientMode = 'sum' | 'single'

const upstreamMode = ref<UpstreamMode>('edge')
const gradientMode = ref<GradientMode>('sum')
const kernelRow = ref(1)
const kernelColumn = ref(1)
const inputRow = ref(2)
const inputColumn = ref(2)

const input = [
  [.2, .1, .0, .7, .1],
  [.1, .3, .2, .9, .0],
  [.0, .2, .4, 1.0, .1],
  [.1, .1, .3, .8, .2],
  [.0, .2, .1, .6, .1],
]
const kernel = [[-.5, 0, .5], [-1, 0, 1], [-.5, 0, .5]]
const upstreamMaps: Record<UpstreamMode, number[][]> = {
  uniform: [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
  center: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  edge: [[.2, .5, .2], [.4, 1, .4], [.1, .3, .1]],
}

const result = computed(() => {
  const upstream = upstreamMaps[upstreamMode.value]
  const kernelContributions = [] as Array<{ i: number; j: number; value: number }>
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      kernelContributions.push({
        i,
        j,
        value: upstream[i][j] * input[i + kernelRow.value][j + kernelColumn.value],
      })
    }
  }
  const correctKernelGradient = kernelContributions.reduce((sum, item) => sum + item.value, 0)
  const centerContribution = kernelContributions.find(item => item.i === 1 && item.j === 1)?.value ?? 0
  const shownKernelGradient = gradientMode.value === 'sum' ? correctKernelGradient : centerContribution

  const inputContributions = [] as Array<{ i: number; j: number; u: number; v: number; value: number }>
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      const u = inputRow.value - i
      const v = inputColumn.value - j
      if (u >= 0 && u < 3 && v >= 0 && v < 3) {
        inputContributions.push({ i, j, u, v, value: upstream[i][j] * kernel[u][v] })
      }
    }
  }
  const inputGradient = inputContributions.reduce((sum, item) => sum + item.value, 0)

  const epsilon = 1e-5
  const lossWithKernel = (candidate: number) => {
    let loss = 0
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        let output = 0
        for (let u = 0; u < 3; u += 1) {
          for (let v = 0; v < 3; v += 1) {
            const weight = u === kernelRow.value && v === kernelColumn.value ? candidate : kernel[u][v]
            output += input[i + u][j + v] * weight
          }
        }
        loss += output * upstream[i][j]
      }
    }
    return loss
  }
  const original = kernel[kernelRow.value][kernelColumn.value]
  const numeric = (lossWithKernel(original + epsilon) - lossWithKernel(original - epsilon)) / (2 * epsilon)
  const relativeError = Math.abs(shownKernelGradient - numeric) / Math.max(1, Math.abs(shownKernelGradient), Math.abs(numeric))
  return { upstream, kernelContributions, correctKernelGradient, shownKernelGradient, inputContributions, inputGradient, numeric, relativeError }
})

const kernelInputCellActive = (row: number, column: number) => result.value.kernelContributions.some(
  item => item.i + kernelRow.value === row && item.j + kernelColumn.value === column && Math.abs(item.value) > 1e-12,
)
const inputUpstreamActive = (row: number, column: number) => result.value.inputContributions.some(
  item => item.i === row && item.j === column && Math.abs(item.value) > 1e-12,
)
const magnitude = (value: number) => Math.min(.9, .14 + Math.abs(value) * .7)
</script>

<template>
  <MlExperimentFrame
    index="15-C"
    title="收集共享卷积核的全部梯度证据"
    question="同一个 K[u,v] 在九个输出位置被重复使用；反向时漏掉任何一条路径，会怎样被有限差分抓住？"
    formula="∂L/∂K[u,v] = Σᵢⱼ G[i,j] X[i+u,j+v]"
  >
    <template #visual>
      <figure class="backward-collector">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`核梯度${result.shownKernelGradient.toFixed(3)}，有限差分误差${result.relativeError.toExponential(2)}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="collector-bg" />
          <text x="31" y="38" class="panel-title">输入 X · 被复用证据</text>
          <text x="268" y="38" class="panel-title">上游梯度 G</text>
          <text x="444" y="38" class="panel-title">核 K / dK</text>
          <g v-for="(values, i) in input" :key="`x-${i}`">
            <g v-for="(value, j) in values" :key="`x-${i}-${j}`">
              <rect :x="31+j*37" :y="54+i*37" width="33" height="33" rx="4" :class="['input-cell',{ active:kernelInputCellActive(i,j), selected:i===inputRow&&j===inputColumn }]" />
              <text :x="47.5+j*37" :y="74+i*37" class="cell-value">{{ value.toFixed(1) }}</text>
            </g>
          </g>
          <g v-for="(values, i) in result.upstream" :key="`g-${i}`">
            <g v-for="(value, j) in values" :key="`g-${i}-${j}`">
              <rect
                :x="267+j*43" :y="70+i*43" width="38" height="38" rx="4"
                :class="['gradient-cell',{ active:inputUpstreamActive(i,j) }]"
                :style="{ fill:`rgba(227,93,104,${magnitude(value)})` }"
              />
              <text :x="286+j*43" :y="93+i*43" class="cell-value">{{ value.toFixed(1) }}</text>
            </g>
          </g>
          <path d="M 408 132 L 431 132" class="flow-arrow" />
          <g v-for="(values, u) in kernel" :key="`k-${u}`">
            <g v-for="(value, v) in values" :key="`k-${u}-${v}`">
              <rect :x="443+v*40" :y="70+u*40" width="35" height="35" rx="4" :class="['kernel-cell',{ active:u===kernelRow&&v===kernelColumn }]" />
              <text :x="460.5+v*40" :y="91+u*40" class="kernel-value">{{ value.toFixed(1) }}</text>
            </g>
          </g>
          <rect x="443" y="205" width="115" height="43" rx="5" :class="['gradient-readout',{ wrong:gradientMode==='single' }]" />
          <text x="454" y="221" class="readout-label">dK[{{ kernelRow }},{{ kernelColumn }}]</text>
          <text x="454" y="239" :class="['readout-value',{ wrong:gradientMode==='single' }]">{{ result.shownKernelGradient.toFixed(3) }}</text>
          <rect x="31" y="263" width="527" height="33" rx="5" class="ledger" />
          <text x="43" y="282" class="ledger-text">dX[{{ inputRow }},{{ inputColumn }}] 收集 {{ result.inputContributions.length }} 条覆盖路径 = {{ result.inputGradient.toFixed(3) }}</text>
          <text x="545" y="282" :class="['audit-text',{ wrong:result.relativeError>1e-6 }]" text-anchor="end">FD {{ result.relativeError<1e-6 ? 'PASS' : 'FAIL' }}</text>
        </svg>
        <figcaption><span><i class="evidence" />参与当前 dK</span><span><i class="selected" />选中输入</span><span><i class="upstream" />上游梯度</span><span><i class="failure" />漏加路径</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="map-switch" role="group" aria-label="上游梯度模式">
        <button type="button" :aria-pressed="upstreamMode==='uniform'" @click="upstreamMode='uniform'">全 1</button>
        <button type="button" :aria-pressed="upstreamMode==='center'" @click="upstreamMode='center'">仅中心</button>
        <button type="button" :aria-pressed="upstreamMode==='edge'" @click="upstreamMode='edge'">边缘加权</button>
      </div>
      <div class="mode-switch" role="group" aria-label="核梯度实现">
        <button type="button" :aria-pressed="gradientMode==='sum'" @click="gradientMode='sum'">累加全部位置</button>
        <button type="button" :aria-pressed="gradientMode==='single'" @click="gradientMode='single'">故障：只取中心</button>
      </div>
      <label><span>核坐标 u <output>{{ kernelRow }}</output></span><input v-model.number="kernelRow" type="range" min="0" max="2" step="1"></label>
      <label><span>核坐标 v <output>{{ kernelColumn }}</output></span><input v-model.number="kernelColumn" type="range" min="0" max="2" step="1"></label>
      <label><span>输入坐标 p <output>{{ inputRow }}</output></span><input v-model.number="inputRow" type="range" min="0" max="4" step="1"></label>
      <label><span>输入坐标 q <output>{{ inputColumn }}</output></span><input v-model.number="inputColumn" type="range" min="0" max="4" step="1"></label>
    </template>
    <template #metrics>
      <div><span>当前 dK</span><strong>{{ result.shownKernelGradient.toFixed(3) }}</strong></div>
      <div><span>核参数使用次数</span><strong>{{ result.kernelContributions.length }}</strong></div>
      <div><span>当前 dX</span><strong>{{ result.inputGradient.toFixed(3) }}</strong></div>
      <div><span>有限差分相对误差</span><strong>{{ result.relativeError.toExponential(2) }}</strong></div>
    </template>
    <template #note>
      参数共享同时影响统计效率和反向实现：一个核坐标从整张特征图收集证据。输入梯度则只从覆盖该像素的输出窗口收集；边界像素被覆盖次数更少，这也是边界梯度统计不同于内部的原因。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.backward-collector{margin:0}.backward-collector svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0b1722}.collector-bg{fill:#0b1722}.backward-collector text{font:700 8px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.input-cell{fill:#132a37;stroke:#315369}.input-cell.active{fill:#284b4e;stroke:#f3c85b;stroke-width:1.5}.input-cell.selected{stroke:#63d0c6;stroke-width:2}.cell-value{fill:#eaf5f4;text-anchor:middle}.gradient-cell{stroke:#743f4d}.gradient-cell.active{stroke:#f3c85b;stroke-width:1.7}.flow-arrow{stroke:#63d0c6;stroke-width:2}.kernel-cell{fill:#14293a;stroke:#4f718a}.kernel-cell.active{fill:#303142;stroke:#f3c85b;stroke-width:2}.kernel-value{fill:#f3c85b;text-anchor:middle}.gradient-readout{fill:#112b2d;stroke:#63d0c6}.gradient-readout.wrong{fill:#321e29;stroke:#e35d68}.readout-label{fill:#829eaf}.readout-value{fill:#63d0c6;font-size:11px}.readout-value.wrong{fill:#e35d68}.ledger{fill:#101f2c;stroke:#29495f}.ledger-text{fill:#bcd0da}.audit-text{fill:#63d0c6}.audit-text.wrong{fill:#e35d68}.backward-collector figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.backward-collector figcaption span{display:inline-flex;align-items:center;gap:5px}.backward-collector figcaption i{width:12px;height:8px;border-radius:2px}.backward-collector figcaption i.evidence{border:2px solid #f3c85b}.backward-collector figcaption i.selected{border:2px solid #63d0c6}.backward-collector figcaption i.upstream{background:#e35d68}.backward-collector figcaption i.failure{background:#6f3445}.map-switch{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.mode-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.map-switch button,.mode-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 8px ui-monospace,monospace;cursor:pointer}.map-switch button[aria-pressed="true"],.mode-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.map-switch button:focus-visible,.mode-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}@media(max-width:560px){.backward-collector{overflow-x:auto;overscroll-behavior-inline:contain}.backward-collector svg{width:560px;max-width:none}.backward-collector figcaption{min-width:560px}}
</style>
