<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type ActivationMode = 'linear' | 'tanh'

const sequenceLength = ref(36)
const recurrentGain = ref(.92)
const operatingPoint = ref(.55)
const truncation = ref(16)
const clipThreshold = ref(2)
const activationMode = ref<ActivationMode>('tanh')

const result = computed(() => {
  const localActivationDerivative = activationMode.value === 'linear'
    ? 1
    : 1 - Math.tanh(operatingPoint.value) ** 2
  const localJacobian = recurrentGain.value * localActivationDerivative
  const gradients = Array.from({ length: sequenceLength.value }, (_, lag) => {
    const value = Math.abs(localJacobian) ** lag
    return { lag, value, log: Math.log10(Math.max(1e-14, value)) }
  })
  const earliest = gradients.at(-1)!.value
  const rawAccumulated = gradients.reduce((sum, point) => sum + point.value, 0)
  const clippedAccumulated = Math.min(rawAccumulated, clipThreshold.value)
  const verdict = earliest < 1e-6 ? '早期梯度消失' : earliest > 1e3 ? '早期梯度爆炸' : '梯度仍可观测'
  return { localActivationDerivative, localJacobian, gradients, earliest, rawAccumulated, clippedAccumulated, verdict }
})

const mapX = (lag: number) => 53 + lag * 494 / Math.max(1, sequenceLength.value - 1)
const mapY = (log: number) => 217 - (Math.max(-12, Math.min(4, log)) + 12) / 16 * 154
const gradientPath = computed(() => result.value.gradients.map((point, index) => (
  `${index === 0 ? 'M' : 'L'} ${mapX(point.lag)} ${mapY(point.log)}`
)).join(' '))
const truncationX = computed(() => mapX(Math.max(0, Math.min(sequenceLength.value - 1, truncation.value))))
</script>

<template>
  <MlExperimentFrame
    index="16-B"
    title="在 BPTT 示波器上观察 Jacobian 连乘"
    question="局部 Jacobian 每一步只偏离 1 一点点，跨几十步后为什么会相差多个数量级？截断和裁剪分别改变了什么？"
    formula="∂hT/∂ht = Πₖ₌ₜ₊₁ᵀ [diag(φ′(aₖ)) Wₕ]"
  >
    <template #visual>
      <figure class="gradient-scope">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`局部雅可比${result.localJacobian.toFixed(3)}，最早梯度${result.earliest.toExponential(2)}，${result.verdict}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="scope-bg" />
          <text x="34" y="37" class="panel-title">BPTT 梯度历史 · 横轴为距最终损失的时间间隔</text>
          <rect x="53" y="53" width="494" height="164" rx="6" class="plot-bg" />
          <rect x="53" y="53" width="494" :height="mapY(-6)-53" class="explode-zone" />
          <rect x="53" :y="mapY(-6)" width="494" :height="217-mapY(-6)" class="vanish-zone" />
          <line x1="53" x2="547" :y1="mapY(0)" :y2="mapY(0)" class="unit-line" />
          <line x1="53" x2="547" :y1="mapY(-6)" :y2="mapY(-6)" class="threshold-line" />
          <line x1="53" x2="547" :y1="mapY(-12)" :y2="mapY(-12)" class="grid-line" />
          <rect :x="truncationX" y="53" :width="547-truncationX" height="164" class="truncated-region" />
          <line :x1="truncationX" :x2="truncationX" y1="53" y2="217" class="truncation-line" />
          <path :d="gradientPath" :class="['gradient-path',{danger:result.earliest<1e-6||result.earliest>1e3}]" />
          <circle v-for="point in result.gradients.filter((_,index)=>index%Math.max(1,Math.floor(sequenceLength/12))===0)" :key="point.lag" :cx="mapX(point.lag)" :cy="mapY(point.log)" r="3" class="scope-point" />
          <text x="45" :y="mapY(0)+3" class="axis-label" text-anchor="end">1</text>
          <text x="45" :y="mapY(-6)+3" class="axis-label" text-anchor="end">1e−6</text>
          <text x="45" :y="mapY(-12)+3" class="axis-label" text-anchor="end">1e−12</text>
          <text x="53" y="233" class="axis-label">lag 0 · 最终时刻</text>
          <text x="547" y="233" class="axis-label" text-anchor="end">lag {{ sequenceLength-1 }} · 最早时刻</text>
          <text :x="Math.min(520,truncationX+7)" y="68" class="truncation-label">截断边界 K={{ truncation }}</text>

          <rect x="53" y="254" width="151" height="37" rx="5" class="readout" />
          <rect x="224" y="254" width="151" height="37" rx="5" class="readout" />
          <rect x="395" y="254" width="152" height="37" rx="5" :class="['readout','verdict',{danger:result.earliest<1e-6||result.earliest>1e3}]" />
          <text x="64" y="268" class="readout-label">局部 |J|</text><text x="64" y="283" class="readout-value">{{ result.localJacobian.toFixed(4) }}</text>
          <text x="235" y="268" class="readout-label">最早梯度</text><text x="235" y="283" class="readout-value">{{ result.earliest.toExponential(2) }}</text>
          <text x="406" y="268" class="readout-label">诊断</text><text x="406" y="283" :class="['verdict-value',{danger:result.earliest<1e-6||result.earliest>1e3}]">{{ result.verdict }}</text>
        </svg>
        <figcaption><span><i class="trace" />梯度倍率</span><span><i class="vanish" />消失区</span><span><i class="truncate" />不反传区域</span><span><i class="unit" />单位增益</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="activation-switch" role="group" aria-label="局部激活">
        <button type="button" :aria-pressed="activationMode==='linear'" @click="activationMode='linear'">线性局部导数</button>
        <button type="button" :aria-pressed="activationMode==='tanh'" @click="activationMode='tanh'">Tanh 工作点</button>
      </div>
      <label><span>序列长度 T <output>{{ sequenceLength }}</output></span><input v-model.number="sequenceLength" type="range" min="8" max="60" step="1"></label>
      <label><span>递推增益 ‖Wₕ‖ <output>{{ recurrentGain.toFixed(2) }}</output></span><input v-model.number="recurrentGain" type="range" min=".25" max="1.6" step=".01"></label>
      <label><span>Tanh 预激活 |a| <output>{{ operatingPoint.toFixed(2) }}</output></span><input v-model.number="operatingPoint" type="range" min="0" max="2.4" step=".05" :disabled="activationMode==='linear'"></label>
      <label><span>截断窗口 K <output>{{ Math.min(truncation,sequenceLength-1) }}</output></span><input v-model.number="truncation" type="range" min="2" :max="sequenceLength-1" step="1"></label>
      <label><span>裁剪阈值 τ <output>{{ clipThreshold.toFixed(1) }}</output></span><input v-model.number="clipThreshold" type="range" min=".5" max="10" step=".5"></label>
      <p>灰色区域不是“梯度变小”，而是截断 BPTT 主动停止建图。裁剪只把累计梯度上限压到 τ，不会把已经衰减到 1e−10 的早期信号放大回来。</p>
    </template>
    <template #metrics>
      <div><span>Tanh 局部导数</span><strong>{{ result.localActivationDerivative.toFixed(3) }}</strong></div>
      <div><span>早期梯度倍率</span><strong>{{ result.earliest.toExponential(2) }}</strong></div>
      <div><span>保留反传步数</span><strong>{{ Math.min(truncation,sequenceLength-1) }} / {{ sequenceLength-1 }}</strong></div>
      <div><span>累计范数 / 裁剪后</span><strong>{{ result.rawAccumulated.toFixed(2) }} / {{ result.clippedAccumulated.toFixed(2) }}</strong></div>
    </template>
    <template #note>
      该标量示波器展示的是局部 Jacobian 恒定时的可控切片。真实向量 RNN 的方向会随时间变化，谱范数上界可能很松；应记录逐时刻 Jacobian-vector product、状态饱和率和全局梯度范数，而不是只看 Wₕ 的一个特征值。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.gradient-scope{margin:0}.gradient-scope svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0b1722}.scope-bg{fill:#0b1722}.gradient-scope text{font:700 8px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.plot-bg{fill:#101f2c}.explode-zone{fill:#e35d68;opacity:.05}.vanish-zone{fill:#574c82;opacity:.13}.unit-line{stroke:#f3c85b;stroke-dasharray:5 4}.threshold-line{stroke:#b59cff;stroke-dasharray:4 4}.grid-line{stroke:#29495f;stroke-dasharray:4 4}.truncated-region{fill:#71808b;opacity:.16}.truncation-line{stroke:#b8cbd1;stroke-dasharray:3 3}.gradient-path{fill:none;stroke:#63d0c6;stroke-width:2.5}.gradient-path.danger{stroke:#e35d68}.scope-point{fill:#f3c85b}.axis-label,.truncation-label{fill:#829eaf}.readout{fill:#101f2c;stroke:#365469}.readout.verdict{stroke:#63d0c6}.readout.verdict.danger{stroke:#e35d68}.readout-label{fill:#829eaf}.readout-value{fill:#f3c85b;font-size:10px}.verdict-value{fill:#63d0c6}.verdict-value.danger{fill:#e35d68}.gradient-scope figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.gradient-scope figcaption span{display:inline-flex;align-items:center;gap:5px}.gradient-scope figcaption i{width:12px;height:8px;border-radius:2px}.gradient-scope figcaption i.trace{height:2px;background:#63d0c6}.gradient-scope figcaption i.vanish{background:#574c82}.gradient-scope figcaption i.truncate{background:#71808b}.gradient-scope figcaption i.unit{border-top:2px dashed #f3c85b}.activation-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.activation-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 9px ui-monospace,monospace;cursor:pointer}.activation-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.activation-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls input:disabled{opacity:.45;cursor:not-allowed}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.gradient-scope{overflow-x:auto;overscroll-behavior-inline:contain}.gradient-scope svg{width:560px;max-width:none}.gradient-scope figcaption{min-width:560px}}
</style>
