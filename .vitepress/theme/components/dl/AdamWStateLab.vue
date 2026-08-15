<script setup lang="ts">
import { computed, ref } from 'vue'

const mode = ref<'adam' | 'adamw'>('adamw')
const beta1 = ref(0.9)
const beta2 = ref(0.99)
const decay = ref(0.08)
const steps = ref(12)
const gradients = [0.82, 0.48, -0.14, 1.05, 0.31, -0.42, 0.66, 0.21, -0.57, 0.38, 0.16, -0.28, 0.44, 0.24, -0.19, 0.35, 0.11, -0.22, 0.29, 0.08]

const trace = computed(() => {
  let theta = 1.5
  let m = 0
  let v = 0
  const learningRate = 0.05
  const rows = [{ step: 0, theta, mHat: 0, vHat: 0, raw: 0 }]
  for (let index = 0; index < steps.value; index += 1) {
    const raw = gradients[index]
    const momentGradient = mode.value === 'adam' ? raw + decay.value * theta : raw
    m = beta1.value * m + (1 - beta1.value) * momentGradient
    v = beta2.value * v + (1 - beta2.value) * momentGradient ** 2
    const t = index + 1
    const mHat = m / (1 - beta1.value ** t)
    const vHat = v / (1 - beta2.value ** t)
    theta -= learningRate * mHat / (Math.sqrt(vHat) + 1e-8)
    if (mode.value === 'adamw') theta -= learningRate * decay.value * theta
    rows.push({ step: t, theta, mHat, vHat, raw })
  }
  return rows
})

const last = computed(() => trace.value.at(-1)!)
const points = computed(() => trace.value.map((row, index) => {
  const x = 34 + index * (470 / Math.max(trace.value.length - 1, 1))
  const y = 210 - ((row.theta - 0.6) / 1.1) * 150
  return `${x.toFixed(1)},${Math.max(24, Math.min(218, y)).toFixed(1)}`
}).join(' '))
const decayEffect = computed(() => mode.value === 'adamw' ? 0.05 * decay.value * last.value.theta : decay.value * last.value.theta)
</script>

<template>
  <DlExperimentFrame index="01-A" title="优化器状态示波器" question="L2 正则项进入自适应矩，和 AdamW 直接衰减参数，会留下怎样不同的状态？" formula="mₜ, vₜ → bias correction → adaptive step → decoupled decay">
    <template #visual>
      <svg class="trace" viewBox="0 0 540 250" role="img" aria-label="参数 theta 随优化步变化的折线图">
        <g class="grid"><line v-for="y in [40,80,120,160,200]" :key="y" x1="34" :y1="y" x2="504" :y2="y" /></g>
        <line class="baseline" x1="34" y1="210" x2="504" y2="210" />
        <polyline :points="points" />
        <circle v-for="(row,index) in trace" :key="row.step" :cx="34 + index * (470 / Math.max(trace.length - 1, 1))" :cy="Math.max(24, Math.min(218, 210 - ((row.theta - 0.6) / 1.1) * 150))" r="3.5" />
        <text x="36" y="237">step 0</text><text x="460" y="237">step {{ steps }}</text><text x="39" y="33">θ trajectory</text>
      </svg>
    </template>
    <template #controls>
      <label>衰减方式<select v-model="mode"><option value="adamw">AdamW：解耦衰减</option><option value="adam">Adam + L2：进入矩估计</option></select></label>
      <label>β₁ = {{ beta1.toFixed(2) }}<input v-model.number="beta1" type="range" min="0.5" max="0.98" step="0.01"></label>
      <label>β₂ = {{ beta2.toFixed(3) }}<input v-model.number="beta2" type="range" min="0.9" max="0.999" step="0.001"></label>
      <label>weight decay = {{ decay.toFixed(2) }}<input v-model.number="decay" type="range" min="0" max="0.3" step="0.01"></label>
      <label>观察步数 = {{ steps }}<input v-model.number="steps" type="range" min="2" max="20" step="1"></label>
    </template>
    <template #metrics>
      <div><span>最终参数 θ</span><strong>{{ last.theta.toFixed(4) }}</strong></div>
      <div><span>偏差修正一阶矩</span><strong>{{ last.mHat.toFixed(4) }}</strong></div>
      <div><span>偏差修正二阶矩</span><strong>{{ last.vHat.toFixed(4) }}</strong></div>
      <div><span>当前衰减贡献</span><strong>{{ decayEffect.toFixed(5) }}</strong></div>
    </template>
    <template #note>切换衰减方式时，Adam + L2 会把参数相关项写入一阶、二阶矩；AdamW 的矩只跟踪任务梯度，衰减作为独立参数更新。两者只有在普通 SGD 等特殊条件下才近似等价。</template>
  </DlExperimentFrame>
</template>

<style scoped>
.trace{display:block;width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.grid line{stroke:#223a4f;stroke-width:1}.baseline{stroke:#5b7885}.trace polyline{fill:none;stroke:#58d3c8;stroke-width:3}.trace circle{fill:#f2c14e;stroke:#13213c;stroke-width:1}.trace text{fill:#8ea9b0;font:10px ui-monospace,monospace}
</style>
