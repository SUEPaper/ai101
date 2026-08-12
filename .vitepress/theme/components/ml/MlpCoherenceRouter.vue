<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type ActivationMode = 'relu' | 'linear'

const activation = ref<ActivationMode>('relu')
const unitCount = ref(4)
const thermalResidual = ref(0.9)
const opticalResidual = ref(-0.8)

const gates = [
  { name: '同向 ++', weights: [1, 1], output: -1, color: '#63d0c6' },
  { name: '同向 −−', weights: [-1, -1], output: -1, color: '#63d0c6' },
  { name: '反向 +−', weights: [1, -1], output: 1, color: '#e35d68' },
  { name: '反向 −+', weights: [-1, 1], output: 1, color: '#e35d68' },
]

const activate = (value: number) => activation.value === 'relu' ? Math.max(0, value) : value

const scoreAt = (x1: number, x2: number) => gates.reduce((score, gate, index) => {
  if (index >= unitCount.value) return score
  return score + gate.output * activate(gate.weights[0] * x1 + gate.weights[1] * x2)
}, 0)

const result = computed(() => {
  const hidden = gates.map((gate, index) => {
    const preactivation = gate.weights[0] * thermalResidual.value + gate.weights[1] * opticalResidual.value
    return {
      ...gate,
      enabled: index < unitCount.value,
      preactivation,
      activation: index < unitCount.value ? activate(preactivation) : 0,
    }
  })
  const score = hidden.reduce((sum, gate) => sum + gate.output * gate.activation, 0)
  const action = Math.abs(score) < 1e-8 ? '边界 / 待裁决' : score > 0 ? '送人工复核' : '自动放行'
  const relation = thermalResidual.value * opticalResidual.value < 0 ? '双通道反向' : '双通道同向'
  const grid = Array.from({ length: 14 * 14 }, (_, index) => {
    const column = index % 14
    const row = Math.floor(index / 14)
    const x = -1.5 + (column + 0.5) * 3 / 14
    const y = 1.5 - (row + 0.5) * 3 / 14
    return { column, row, score: scoreAt(x, y) }
  })
  return { hidden, score, action, relation, grid }
})

const samples = [
  [-1.25, -1.05], [-1.05, -1.28], [-0.82, -0.94],
  [1.18, 1.02], [0.92, 1.24], [1.28, 0.78],
  [1.18, -1.03], [0.86, -1.22], [1.30, -0.74],
  [-1.17, 1.06], [-0.83, 1.25], [-1.28, 0.78],
].map(([x, y]) => ({ x, y, mismatch: x * y < 0 }))

const mapX = (value: number) => 38 + (value + 1.5) / 3 * 255
const mapY = (value: number) => 289 - (value + 1.5) / 3 * 239
const scoreY = (value: number) => 170 - Math.max(-3, Math.min(3, value)) / 3 * 102

const loadMismatch = () => {
  thermalResidual.value = 1
  opticalResidual.value = -1
}

const loadAgreement = () => {
  thermalResidual.value = -1
  opticalResidual.value = -1
}
</script>

<template>
  <MlExperimentFrame
    index="13-A"
    title="让四个 ReLU 门接管双传感器一致性路由"
    question="为什么四个线性投影各自只能切一刀，却能在非线性门控后把同向样本与反向样本分开？"
    formula="s(x)=−h₁−h₂+h₃+h₄,  h=φ(Wx)"
  >
    <template #visual>
      <figure class="coherence-router">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`当前 ${result.relation}，网络分数 ${result.score.toFixed(2)}，动作 ${result.action}`">
          <defs>
            <clipPath id="mlp-input-clip"><rect x="38" y="50" width="255" height="239" rx="5" /></clipPath>
          </defs>
          <rect x="15" y="15" width="570" height="296" rx="9" class="router-bg" />
          <text x="38" y="32" class="panel-title">焊接热像残差 × 光强残差</text>
          <text x="316" y="32" class="panel-title">隐藏门总线</text>
          <text x="503" y="32" class="panel-title">处置分数</text>
          <rect x="38" y="50" width="255" height="239" rx="5" class="plot-bg" />
          <g clip-path="url(#mlp-input-clip)">
            <rect
              v-for="cell in result.grid"
              :key="`cell-${cell.row}-${cell.column}`"
              :x="38 + cell.column * 255 / 14"
              :y="50 + cell.row * 239 / 14"
              :width="255 / 14 + .5"
              :height="239 / 14 + .5"
              :class="cell.score > .03 ? 'review-cell' : cell.score < -.03 ? 'pass-cell' : 'boundary-cell'"
              :style="{ opacity: Math.min(.34, .07 + Math.abs(cell.score) * .08) }"
            />
            <line :x1="mapX(0)" :x2="mapX(0)" y1="50" y2="289" class="axis" />
            <line x1="38" x2="293" :y1="mapY(0)" :y2="mapY(0)" class="axis" />
            <line :x1="mapX(-1.5)" :y1="mapY(1.5)" :x2="mapX(1.5)" :y2="mapY(-1.5)" class="gate-line agree" />
            <line :x1="mapX(-1.5)" :y1="mapY(-1.5)" :x2="mapX(1.5)" :y2="mapY(1.5)" class="gate-line mismatch" />
            <circle v-for="(sample,index) in samples" :key="index" :cx="mapX(sample.x)" :cy="mapY(sample.y)" r="5" :class="sample.mismatch ? 'review-point' : 'pass-point'" />
            <circle :cx="mapX(thermalResidual)" :cy="mapY(opticalResidual)" r="9" class="probe-point" />
            <circle :cx="mapX(thermalResidual)" :cy="mapY(opticalResidual)" r="3" class="probe-core" />
          </g>
          <text x="165" y="308" class="axis-label center">热像能量残差 x₁</text>
          <text x="44" y="62" class="axis-label">光强残差 x₂</text>

          <g v-for="(gate,index) in result.hidden" :key="gate.name" :transform="`translate(316 ${51 + index * 57})`">
            <rect width="150" height="43" rx="6" :class="['gate-shell', { disabled: !gate.enabled }]" />
            <rect x="0" y="0" :width="Math.min(150, gate.activation * 44)" height="43" rx="6" :fill="gate.color" class="gate-signal" />
            <text x="10" y="17" class="gate-name">{{ gate.name }}</text>
            <text x="10" y="34" class="gate-value">z={{ gate.preactivation.toFixed(2) }} · h={{ gate.activation.toFixed(2) }}</text>
            <text x="140" y="26" class="gate-sign end">{{ gate.output > 0 ? '+1' : '−1' }}</text>
          </g>

          <line x1="482" x2="482" y1="68" y2="272" class="score-track" />
          <rect x="494" y="67" width="73" height="93" rx="5" class="review-zone" />
          <rect x="494" y="180" width="73" height="93" rx="5" class="pass-zone" />
          <line x1="486" x2="571" y1="170" y2="170" class="score-zero" />
          <circle cx="482" :cy="scoreY(result.score)" r="7" class="score-probe" />
          <text x="530" y="87" class="route-label center">复核 +</text>
          <text x="530" y="255" class="route-label center">放行 −</text>
          <text x="530" y="166" class="score-value center">s={{ result.score.toFixed(2) }}</text>
        </svg>
        <figcaption><span><i class="pass" />通道同向</span><span><i class="review" />通道反向</span><span><i class="probe" />当前焊点</span><span><i class="line" />ReLU 折叠线</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="mode-switch" role="group" aria-label="隐藏层激活函数">
        <button type="button" :aria-pressed="activation === 'relu'" @click="activation='relu'">ReLU 门控</button>
        <button type="button" :aria-pressed="activation === 'linear'" @click="activation='linear'">恒等激活</button>
      </div>
      <label><span>启用隐藏单元 <output>{{ unitCount }} / 4</output></span><input v-model.number="unitCount" type="range" min="1" max="4" step="1"></label>
      <label><span>热像能量残差 x₁ <output>{{ thermalResidual.toFixed(2) }}</output></span><input v-model.number="thermalResidual" type="range" min="-1.4" max="1.4" step="0.05"></label>
      <label><span>光强残差 x₂ <output>{{ opticalResidual.toFixed(2) }}</output></span><input v-model.number="opticalResidual" type="range" min="-1.4" max="1.4" step="0.05"></label>
      <div class="sample-switch"><button type="button" @click="loadMismatch">装载反向样本</button><button type="button" @click="loadAgreement">装载同向样本</button></div>
      <p>四门齐备且使用 ReLU 时，分数等于 |x₁−x₂|−|x₁+x₂|。改成恒等激活后四条线性支路精确抵消，隐藏层再宽也没有产生这组折叠。</p>
    </template>
    <template #metrics>
      <div><span>传感器关系</span><strong>{{ result.relation }}</strong></div>
      <div><span>隐藏向量 h</span><strong>[{{ result.hidden.map(g => g.activation.toFixed(1)).join(', ') }}]</strong></div>
      <div><span>输出分数</span><strong>{{ result.score.toFixed(3) }}</strong></div>
      <div><span>路由动作</span><strong>{{ result.action }}</strong></div>
    </template>
    <template #note>
      这组权重是为解释 XOR 型一致性规则而手工构造的证据，不代表生产网络靠人工指定隐藏单元。真正训练时会从数据学习 W；第 14 章再推导这些参数怎样通过反向传播更新。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.coherence-router{margin:0}.coherence-router svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0c1925}.router-bg{fill:#0c1925}.coherence-router text{font:700 9px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.plot-bg{fill:#101f2c}.pass-cell{fill:#63d0c6}.review-cell{fill:#e35d68}.boundary-cell{fill:#f3c85b}.axis{stroke:#345164;stroke-width:1.2}.gate-line{stroke-width:1.5;stroke-dasharray:6 5;opacity:.7}.gate-line.agree{stroke:#63d0c6}.gate-line.mismatch{stroke:#e35d68}.pass-point{fill:#63d0c6;stroke:#eaf7f5;stroke-width:1.3}.review-point{fill:#e35d68;stroke:#fff0f1;stroke-width:1.3}.probe-point{fill:#f3c85b;stroke:#fff;stroke-width:2}.probe-core{fill:#0c1925}.axis-label{fill:#829eaf}.center{text-anchor:middle}.end{text-anchor:end}.gate-shell{fill:#152838;stroke:#355166}.gate-shell.disabled{opacity:.35}.gate-signal{opacity:.16}.gate-name{fill:#dbeaf0}.gate-value{fill:#829eaf}.gate-sign{fill:#f3c85b}.score-track{stroke:#355166;stroke-width:3}.review-zone{fill:#e35d68;opacity:.12}.pass-zone{fill:#63d0c6;opacity:.12}.score-zero{stroke:#f3c85b;stroke-dasharray:4 4}.score-probe{fill:#f08a55;stroke:#fff;stroke-width:1.5}.route-label{fill:#829eaf}.score-value{fill:#f3c85b}.coherence-router figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.coherence-router figcaption span{display:inline-flex;align-items:center;gap:5px}.coherence-router figcaption i{width:12px;height:8px;border-radius:2px}.coherence-router figcaption i.pass{background:#63d0c6}.coherence-router figcaption i.review{background:#e35d68}.coherence-router figcaption i.probe{background:#f3c85b}.coherence-router figcaption i.line{height:2px;background:#829eaf}.mode-switch,.sample-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.mode-switch button,.sample-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 10px ui-monospace,monospace;cursor:pointer}.mode-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.mode-switch button:focus-visible,.sample-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.coherence-router{overflow-x:auto;overscroll-behavior-inline:contain}.coherence-router svg{width:560px;max-width:none}.coherence-router figcaption{min-width:560px}}
</style>
