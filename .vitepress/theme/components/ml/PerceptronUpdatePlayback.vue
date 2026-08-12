<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Sample = {
  x: number
  y: number
  label: -1 | 1
  id: string
}

const learningRate = ref(0.7)
const step = ref(0)
const hasConflict = ref(false)

const baseSamples: Sample[] = [
  { x: -1.8, y: -0.8, label: -1, id: 'N1' },
  { x: -1.5, y: 0.55, label: -1, id: 'N2' },
  { x: -0.9, y: -0.4, label: -1, id: 'N3' },
  { x: -0.7, y: 1.0, label: -1, id: 'N4' },
  { x: 0.8, y: -0.9, label: 1, id: 'P1' },
  { x: 1.2, y: 0.3, label: 1, id: 'P2' },
  { x: 1.6, y: -0.2, label: 1, id: 'P3' },
  { x: 1.8, y: 0.9, label: 1, id: 'P4' },
]

const samples = computed<Sample[]>(() => hasConflict.value
  ? [...baseSamples, { x: 1.2, y: 0.3, label: -1, id: 'X!' }]
  : baseSamples)

const maximumStep = computed(() => samples.value.length * 4)

watch(maximumStep, (maximum) => {
  if (step.value > maximum) step.value = maximum
})

const playback = computed(() => {
  let weightX = 0
  let weightY = 0
  let intercept = 0
  let mistakes = 0
  const states = [{
    weightX,
    weightY,
    intercept,
    mistakes,
    currentIndex: -1,
    marginBefore: 0,
    updated: false,
    accuracy: 0,
  }]

  for (let presentation = 0; presentation < maximumStep.value; presentation++) {
    const currentIndex = presentation % samples.value.length
    const sample = samples.value[currentIndex]
    const marginBefore = sample.label * (
      weightX * sample.x + weightY * sample.y + intercept
    )
    const updated = marginBefore <= 0
    if (updated) {
      weightX += learningRate.value * sample.label * sample.x
      weightY += learningRate.value * sample.label * sample.y
      intercept += learningRate.value * sample.label
      mistakes += 1
    }
    const correct = samples.value.filter((item) =>
      item.label * (weightX * item.x + weightY * item.y + intercept) > 0
    ).length
    states.push({
      weightX,
      weightY,
      intercept,
      mistakes,
      currentIndex,
      marginBefore,
      updated,
      accuracy: correct / samples.value.length,
    })
  }
  return states
})

const state = computed(() => playback.value[Math.min(step.value, maximumStep.value)])
const currentSample = computed(() => state.value.currentIndex < 0
  ? null
  : samples.value[state.value.currentIndex])

const mapX = (value: number) => 45 + (value + 2.4) / 4.8 * 350
const mapY = (value: number) => 286 - (value + 1.8) / 3.6 * 242

const boundaryPath = computed(() => {
  const norm = Math.hypot(state.value.weightX, state.value.weightY)
  if (norm < 1e-9) return ''
  const centerX = -state.value.intercept * state.value.weightX / (norm * norm)
  const centerY = -state.value.intercept * state.value.weightY / (norm * norm)
  const tangentX = -state.value.weightY / norm
  const tangentY = state.value.weightX / norm
  const length = 6
  return `M ${mapX(centerX - length * tangentX)} ${mapY(centerY - length * tangentY)} L ${mapX(centerX + length * tangentX)} ${mapY(centerY + length * tangentY)}`
})

const updateVector = computed(() => {
  if (!currentSample.value || !state.value.updated) return null
  const scale = learningRate.value * currentSample.value.label
  return {
    x: currentSample.value.x * scale,
    y: currentSample.value.y * scale,
  }
})

const previous = () => {
  step.value = Math.max(0, step.value - 1)
}

const next = () => {
  step.value = Math.min(maximumStep.value, step.value + 1)
}

const toggleConflict = () => {
  hasConflict.value = !hasConflict.value
  step.value = 0
}
</script>

<template>
  <MlExperimentFrame
    index="11-A"
    title="逐件播放滚子终检的感知机更新"
    question="模型为什么只在错分或压线时移动？加入一条互相矛盾的标注后，收敛保证又为何立即失效？"
    formula="yᵢf(xᵢ) ≤ 0 ⇒ (w,b) ← (w,b)+ηyᵢ(xᵢ,1)"
  >
    <template #visual>
      <figure class="perceptron-bench">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`已处理 ${step} 件，累计更新 ${state.mistakes} 次，当前准确率 ${(state.accuracy * 100).toFixed(1)}%`">
          <defs>
            <clipPath id="perceptron-plot-clip"><rect x="45" y="44" width="350" height="242" rx="5" /></clipPath>
            <marker id="perceptron-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" class="arrow-head" />
            </marker>
          </defs>
          <rect x="15" y="15" width="570" height="296" rx="9" class="bench-bg" />
          <text x="45" y="31" class="panel-title">滚子终检特征平面 / 顺序过检</text>
          <text x="421" y="31" class="panel-title">本次更新账本</text>
          <rect x="45" y="44" width="350" height="242" rx="5" class="plot-bg" />
          <g class="grid">
            <line v-for="value in [-2,-1,0,1,2]" :key="`gx-${value}`" :x1="mapX(value)" :x2="mapX(value)" y1="44" y2="286" />
            <line v-for="value in [-1,0,1]" :key="`gy-${value}`" x1="45" x2="395" :y1="mapY(value)" :y2="mapY(value)" />
          </g>
          <g clip-path="url(#perceptron-plot-clip)">
            <path v-if="boundaryPath" :d="boundaryPath" class="decision-line" />
            <g v-for="(sample,index) in samples" :key="sample.id">
              <circle
                v-if="state.currentIndex === index"
                :cx="mapX(sample.x)"
                :cy="mapY(sample.y)"
                r="12"
                :class="['scan-ring', state.updated ? 'is-updated' : '']"
              />
              <circle
                :cx="mapX(sample.x)"
                :cy="mapY(sample.y)"
                r="6.5"
                :class="sample.label === 1 ? 'defect-point' : 'pass-point'"
              />
              <path
                v-if="sample.id === 'X!'"
                :d="`M ${mapX(sample.x)-5} ${mapY(sample.y)-5} L ${mapX(sample.x)+5} ${mapY(sample.y)+5} M ${mapX(sample.x)+5} ${mapY(sample.y)-5} L ${mapX(sample.x)-5} ${mapY(sample.y)+5}`"
                class="conflict-mark"
              />
            </g>
          </g>
          <text x="220" y="303" class="axis-label center">涡流响应 z₁</text>
          <text transform="translate(27 168) rotate(-90)" class="axis-label center">轮廓缺口 z₂</text>

          <text x="421" y="62" class="audit-label">当前样本</text>
          <text x="562" y="62" class="audit-value">{{ currentSample?.id ?? '尚未开始' }}</text>
          <text x="421" y="95" class="audit-label">更新前 y·f(x)</text>
          <text x="562" y="95" :class="['audit-value', state.updated ? 'danger' : 'safe']">{{ step === 0 ? '—' : state.marginBefore.toFixed(3) }}</text>
          <text x="421" y="128" class="audit-label">是否更新</text>
          <text x="562" y="128" :class="['audit-value', state.updated ? 'danger' : 'safe']">{{ step === 0 ? '—' : state.updated ? '执行' : '跳过' }}</text>
          <line x1="421" x2="562" y1="146" y2="146" class="audit-rule" />
          <text x="421" y="174" class="audit-label">w₁ / w₂</text>
          <text x="562" y="174" class="audit-value">{{ state.weightX.toFixed(2) }} / {{ state.weightY.toFixed(2) }}</text>
          <text x="421" y="207" class="audit-label">截距 b</text>
          <text x="562" y="207" class="audit-value">{{ state.intercept.toFixed(2) }}</text>
          <text x="421" y="240" class="audit-label">Δw</text>
          <text x="562" y="240" class="audit-value update">{{ updateVector ? `${updateVector.x.toFixed(2)}, ${updateVector.y.toFixed(2)}` : '0, 0' }}</text>
          <line x1="421" x2="562" y1="258" y2="258" class="audit-rule" />
          <text x="421" y="286" class="audit-label">标签合同</text>
          <text x="562" y="286" :class="['audit-value', hasConflict ? 'danger' : 'safe']">{{ hasConflict ? '存在冲突' : '线性可分' }}</text>
        </svg>
        <figcaption><span><i class="pass" />自动放行 −1</span><span><i class="defect" />送复检 +1</span><span><i class="scan" />当前过检样本</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>处理到第几件 <output>{{ step }} / {{ maximumStep }}</output></span>
        <input v-model.number="step" type="range" min="0" :max="maximumStep" step="1">
      </label>
      <label>
        <span>学习率 η <output>{{ learningRate.toFixed(2) }}</output></span>
        <input v-model.number="learningRate" type="range" min="0.20" max="1.50" step="0.05">
      </label>
      <div class="playback-controls">
        <button type="button" :disabled="step === 0" @click="previous">上一件</button>
        <button type="button" :disabled="step === maximumStep" @click="next">下一件</button>
      </div>
      <button type="button" class="conflict-toggle" :aria-pressed="hasConflict" @click="toggleConflict">
        {{ hasConflict ? '移除冲突标注' : '加入冲突标注' }}
      </button>
      <p>黄色环表示当前样本；红色环表示它触发了更新。冲突标注与 P2 位于同一坐标却给出相反标签，因此任何线性边界都不可能同时分对两者。</p>
    </template>
    <template #metrics>
      <div><span>累计犯错更新</span><strong>{{ state.mistakes }}</strong></div>
      <div><span>当前训练准确率</span><strong>{{ (state.accuracy * 100).toFixed(1) }}%</strong></div>
      <div><span>参数范数 ‖w‖₂</span><strong>{{ Math.hypot(state.weightX, state.weightY).toFixed(3) }}</strong></div>
      <div><span>当前数据状态</span><strong>{{ hasConflict ? '不可分' : state.accuracy === 1 ? '已分开' : '仍有错误' }}</strong></div>
    </template>
    <template #note>
      感知机收敛定理要求标签在所选特征空间中严格线性可分。学习率会改变参数尺度与轨迹，但不会修复相互矛盾的数据合同；不可分时应记录最佳模型、停止协议或改用软间隔目标。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.perceptron-bench{margin:0}.perceptron-bench svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0c1925}.bench-bg{fill:#0c1925}.perceptron-bench text{font:700 10px ui-monospace,monospace}.panel-title{fill:#dbeaf0}.plot-bg{fill:#101f2c}.grid line{stroke:#213b4d;stroke-width:1}.decision-line{fill:none;stroke:#f3c85b;stroke-width:3}.pass-point{fill:#63d0c6;stroke:#dbeaf0;stroke-width:1.5}.defect-point{fill:#e35d68;stroke:#dbeaf0;stroke-width:1.5}.scan-ring{fill:none;stroke:#f3c85b;stroke-width:2}.scan-ring.is-updated{stroke:#f08a55;stroke-width:3}.conflict-mark{fill:none;stroke:#07131c;stroke-width:2}.axis-label,.audit-label{fill:#829eaf}.axis-label.center{text-anchor:middle}.audit-value{fill:#dbeaf0;text-anchor:end;font-size:11px}.audit-value.safe{fill:#63d0c6}.audit-value.danger{fill:#e35d68}.audit-value.update{fill:#f3c85b}.audit-rule{stroke:#355165}.arrow-head{fill:#f3c85b}.perceptron-bench figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.perceptron-bench figcaption span{display:inline-flex;align-items:center;gap:5px}.perceptron-bench figcaption i{width:12px;height:8px;border-radius:2px}.perceptron-bench figcaption i.pass{background:#63d0c6}.perceptron-bench figcaption i.defect{background:#e35d68}.perceptron-bench figcaption i.scan{border:2px solid #f3c85b;background:transparent}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.playback-controls{display:grid;grid-template-columns:1fr 1fr;gap:8px}.playback-controls button,.conflict-toggle{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 11px ui-monospace,monospace;cursor:pointer}.playback-controls button:hover,.conflict-toggle:hover{border-color:#f3c85b}.playback-controls button:focus-visible,.conflict-toggle:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.playback-controls button:disabled{cursor:not-allowed;opacity:.42}.conflict-toggle[aria-pressed="true"]{border-color:#e35d68;color:#ffd7d9;background:#311e29}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.perceptron-bench{overflow-x:auto;overscroll-behavior-inline:contain}.perceptron-bench svg{width:560px;max-width:none}.perceptron-bench figcaption{min-width:560px}}
</style>
