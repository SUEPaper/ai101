<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type KernelMode = 'vertical' | 'horizontal' | 'blur' | 'learned'
type OperatorMode = 'correlation' | 'convolution'

const kernelMode = ref<KernelMode>('vertical')
const operatorMode = ref<OperatorMode>('correlation')
const stride = ref(1)
const padding = ref(0)
const activeRow = ref(1)
const activeColumn = ref(1)

const input = [
  [.08, .12, .18, .82, .16, .10],
  [.10, .14, .20, .94, .18, .11],
  [.12, .16, .24, 1.00, .20, .12],
  [.11, .15, .21, .88, .19, .10],
  [.09, .13, .18, .78, .16, .09],
  [.07, .11, .15, .70, .14, .08],
]

const kernels: Record<KernelMode, number[][]> = {
  vertical: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
  horizontal: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
  blur: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
  learned: [[-.6, -.2, .5], [-1.1, .1, 1.2], [-.5, .2, .7]],
}

const effectiveKernel = computed(() => {
  const raw = kernels[kernelMode.value]
  if (operatorMode.value === 'correlation') return raw
  return [...raw].reverse().map(row => [...row].reverse())
})

const outputSize = computed(() => Math.floor((6 + 2 * padding.value - 3) / stride.value) + 1)
const row = computed(() => Math.min(activeRow.value, outputSize.value - 1))
const column = computed(() => Math.min(activeColumn.value, outputSize.value - 1))

const result = computed(() => {
  const size = outputSize.value
  const output: number[][] = []
  for (let i = 0; i < size; i += 1) {
    const outputRow: number[] = []
    for (let j = 0; j < size; j += 1) {
      let value = 0
      for (let u = 0; u < 3; u += 1) {
        for (let v = 0; v < 3; v += 1) {
          const sourceRow = i * stride.value + u - padding.value
          const sourceColumn = j * stride.value + v - padding.value
          const source = sourceRow >= 0 && sourceRow < 6 && sourceColumn >= 0 && sourceColumn < 6
            ? input[sourceRow][sourceColumn]
            : 0
          value += source * effectiveKernel.value[u][v]
        }
      }
      outputRow.push(value)
    }
    output.push(outputRow)
  }

  const terms = [] as Array<{ input: number; weight: number; product: number; sourceRow: number; sourceColumn: number }>
  for (let u = 0; u < 3; u += 1) {
    for (let v = 0; v < 3; v += 1) {
      const sourceRow = row.value * stride.value + u - padding.value
      const sourceColumn = column.value * stride.value + v - padding.value
      const source = sourceRow >= 0 && sourceRow < 6 && sourceColumn >= 0 && sourceColumn < 6
        ? input[sourceRow][sourceColumn]
        : 0
      const weight = effectiveKernel.value[u][v]
      terms.push({ input: source, weight, product: source * weight, sourceRow, sourceColumn })
    }
  }
  return { output, terms, activeValue: output[row.value][column.value] }
})

const isActiveInput = (inputRow: number, inputColumn: number) => result.value.terms.some(
  term => term.sourceRow === inputRow && term.sourceColumn === inputColumn,
)
const intensity = (value: number) => Math.max(.08, Math.min(.9, value))
const responseOpacity = (value: number) => Math.min(.9, .16 + Math.abs(value) / 3)
const formatWeight = (value: number) => Math.abs(value) < .001 ? '0' : value.toFixed(1)
</script>

<template>
  <MlExperimentFrame
    index="15-A"
    title="在线扫钢带上逐格展开卷积"
    question="同一组核权重滑过每个位置时，当前窗口的九项乘积怎样形成特征图？数学卷积和框架互相关又差在哪一步？"
    formula="Y[i,j] = Σᵤᵥ K[u,v] X[iS+u−P,jS+v−P]"
  >
    <template #visual>
      <figure class="scan-workbench">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`输入六乘六，输出${outputSize}乘${outputSize}，当前响应${result.activeValue.toFixed(3)}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="bench-bg" />
          <text x="31" y="37" class="panel-title">线扫灰度块 X</text>
          <text x="223" y="37" class="panel-title">共享核 K</text>
          <text x="407" y="37" class="panel-title">响应图 Y</text>

          <g v-for="(values, i) in input" :key="`input-${i}`">
            <g v-for="(value, j) in values" :key="`input-${i}-${j}`">
              <rect
                :x="31+j*25" :y="50+i*25" width="22" height="22" rx="3"
                :class="['input-cell',{ active:isActiveInput(i,j), defect:value>.6 }]"
                :style="{ fill: `rgba(99,208,198,${intensity(value)})` }"
              />
              <text :x="42+j*25" :y="64+i*25" class="cell-value">{{ value.toFixed(1) }}</text>
            </g>
          </g>
          <rect
            :x="29+(column*stride-padding)*25"
            :y="48+(row*stride-padding)*25"
            width="72" height="72" rx="5" class="window-frame"
          />

          <g v-for="(values, u) in effectiveKernel" :key="`kernel-${u}`">
            <g v-for="(value, v) in values" :key="`kernel-${u}-${v}`">
              <rect :x="218+v*37" :y="67+u*37" width="33" height="33" rx="4" class="kernel-cell" />
              <text :x="234.5+v*37" :y="87+u*37" class="kernel-value">{{ formatWeight(value) }}</text>
            </g>
          </g>
          <path d="M 378 107 L 398 107" class="flow-arrow" />
          <text x="309" y="199" class="operator-note">{{ operatorMode==='correlation' ? '核保持原方向' : '核已水平+垂直翻转' }}</text>

          <g v-for="(values, i) in result.output" :key="`output-${i}`">
            <g v-for="(value, j) in values" :key="`output-${i}-${j}`">
              <rect
                :x="406+j*(148/outputSize)" :y="50+i*(148/outputSize)"
                :width="Math.max(18,144/outputSize)" :height="Math.max(18,144/outputSize)" rx="3"
                :class="['output-cell',{ active:i===row && j===column, negative:value<0 }]"
                :style="{ fill: value >= 0 ? `rgba(243,200,91,${responseOpacity(value)})` : `rgba(227,93,104,${responseOpacity(value)})` }"
              />
              <text
                v-if="outputSize<=4"
                :x="406+j*(148/outputSize)+72/outputSize"
                :y="50+i*(148/outputSize)+82/outputSize"
                class="output-value"
              >{{ value.toFixed(1) }}</text>
            </g>
          </g>

          <rect x="31" y="223" width="523" height="62" rx="6" class="ledger" />
          <text x="43" y="241" class="ledger-label">当前输出 Y[{{ row }},{{ column }}] 的九项乘积</text>
          <g v-for="(term, index) in result.terms" :key="`term-${index}`">
            <text :x="43+(index%5)*101" :y="259+Math.floor(index/5)*16" class="term-value">
              {{ term.input.toFixed(1) }}×{{ formatWeight(term.weight) }}={{ term.product.toFixed(2) }}
            </text>
          </g>
          <text x="542" y="277" class="sum-value" text-anchor="end">Σ={{ result.activeValue.toFixed(3) }}</text>
        </svg>
        <figcaption><span><i class="window" />当前滑窗</span><span><i class="scratch" />高亮划痕</span><span><i class="positive" />正响应</span><span><i class="negative" />负响应</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="kernel-switch" role="group" aria-label="卷积核">
        <button type="button" :aria-pressed="kernelMode==='vertical'" @click="kernelMode='vertical'">垂直边缘</button>
        <button type="button" :aria-pressed="kernelMode==='horizontal'" @click="kernelMode='horizontal'">水平边缘</button>
        <button type="button" :aria-pressed="kernelMode==='blur'" @click="kernelMode='blur'">均值平滑</button>
        <button type="button" :aria-pressed="kernelMode==='learned'" @click="kernelMode='learned'">学习型核</button>
      </div>
      <div class="operator-switch" role="group" aria-label="运算定义">
        <button type="button" :aria-pressed="operatorMode==='correlation'" @click="operatorMode='correlation'">框架互相关</button>
        <button type="button" :aria-pressed="operatorMode==='convolution'" @click="operatorMode='convolution'">数学卷积</button>
      </div>
      <div class="operator-switch" role="group" aria-label="步幅">
        <button type="button" :aria-pressed="stride===1" @click="stride=1">步幅 1</button>
        <button type="button" :aria-pressed="stride===2" @click="stride=2">步幅 2</button>
      </div>
      <div class="operator-switch" role="group" aria-label="填充">
        <button type="button" :aria-pressed="padding===0" @click="padding=0">无填充</button>
        <button type="button" :aria-pressed="padding===1" @click="padding=1">零填充 1</button>
      </div>
      <label><span>输出行 i <output>{{ row }}</output></span><input v-model.number="activeRow" type="range" min="0" :max="outputSize-1" step="1"></label>
      <label><span>输出列 j <output>{{ column }}</output></span><input v-model.number="activeColumn" type="range" min="0" :max="outputSize-1" step="1"></label>
    </template>
    <template #metrics>
      <div><span>输出空间尺寸</span><strong>{{ outputSize }} × {{ outputSize }}</strong></div>
      <div><span>当前响应</span><strong>{{ result.activeValue.toFixed(3) }}</strong></div>
      <div><span>共享参数</span><strong>9 + 1 bias</strong></div>
      <div><span>窗口起点</span><strong>({{ row*stride-padding }}, {{ column*stride-padding }})</strong></div>
    </template>
    <template #note>
      切换“数学卷积”时只翻转核，不改输入。对固定手工核，方向响应会改变；对可学习核，训练能吸收这次重参数化，所以深度学习接口习惯把互相关也称作卷积。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.scan-workbench{margin:0}.scan-workbench svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0b1722}.bench-bg{fill:#0b1722}.scan-workbench text{font:700 8px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.input-cell{stroke:#29495f;stroke-width:1}.input-cell.active{stroke:#f3c85b;stroke-width:1.8}.input-cell.defect{filter:drop-shadow(0 0 3px rgba(227,93,104,.8))}.cell-value,.output-value{fill:#eaf5f4;text-anchor:middle;font-size:7px}.window-frame{fill:none;stroke:#f3c85b;stroke-width:2;stroke-dasharray:5 3}.kernel-cell{fill:#14293a;stroke:#4f718a}.kernel-value{fill:#f3c85b;text-anchor:middle;font-size:9px}.flow-arrow{fill:none;stroke:#63d0c6;stroke-width:2}.operator-note{fill:#829eaf;text-anchor:middle}.output-cell{stroke:#59633e}.output-cell.negative{stroke:#6f3c4b}.output-cell.active{stroke:#fff;stroke-width:2.3}.ledger{fill:#101f2c;stroke:#29495f}.ledger-label{fill:#829eaf}.term-value{fill:#bcd0da}.sum-value{fill:#f3c85b;font-size:9px}.scan-workbench figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.scan-workbench figcaption span{display:inline-flex;align-items:center;gap:5px}.scan-workbench figcaption i{width:12px;height:8px;border-radius:2px}.scan-workbench figcaption i.window{border:2px solid #f3c85b}.scan-workbench figcaption i.scratch{background:#63d0c6}.scan-workbench figcaption i.positive{background:#f3c85b}.scan-workbench figcaption i.negative{background:#e35d68}.kernel-switch,.operator-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.kernel-switch button,.operator-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 9px ui-monospace,monospace;cursor:pointer}.kernel-switch button[aria-pressed="true"],.operator-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.kernel-switch button:focus-visible,.operator-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}@media(max-width:560px){.scan-workbench{overflow-x:auto;overscroll-behavior-inline:contain}.scan-workbench svg{width:560px;max-width:none}.scan-workbench figcaption{min-width:560px}}
</style>
