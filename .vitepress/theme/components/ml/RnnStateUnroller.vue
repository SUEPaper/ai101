<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type ResetMode = 'reset' | 'carry'

const sequenceLength = ref(8)
const recurrentGain = ref(.78)
const inputGain = ref(.9)
const eventStep = ref(2)
const activeStep = ref(4)
const resetMode = ref<ResetMode>('reset')

const baseSequence = [.12, .08, 1.0, .16, -.06, .1, .04, -.12, .08, .02, -.05, .06]

const result = computed(() => {
  const length = sequenceLength.value
  const eventIndex = Math.min(eventStep.value, length - 1)
  const activeIndex = Math.min(activeStep.value, length - 1)
  const inputs = baseSequence.slice(0, length)
  inputs[eventIndex] = 1
  const initialState = resetMode.value === 'reset' ? 0 : .62
  const states = [initialState]
  const preactivations = [] as number[]
  for (const input of inputs) {
    const preactivation = inputGain.value * input + recurrentGain.value * states.at(-1)!
    preactivations.push(preactivation)
    states.push(Math.tanh(preactivation))
  }

  const evaluateFinal = (eventValue: number) => {
    const perturbed = [...inputs]
    perturbed[eventIndex] = eventValue
    let state = initialState
    for (const input of perturbed) state = Math.tanh(inputGain.value * input + recurrentGain.value * state)
    return state
  }
  const epsilon = 1e-5
  const eventSensitivity = (evaluateFinal(1 + epsilon) - evaluateFinal(1 - epsilon)) / (2 * epsilon)
  return {
    inputs,
    states,
    preactivations,
    eventIndex,
    activeIndex,
    initialState,
    finalState: states.at(-1)!,
    eventSensitivity,
  }
})

const timelineX = (index: number) => 48 + index * 504 / Math.max(1, sequenceLength.value - 1)
const stateY = (value: number) => 231 - (value + 1) * 33
const statePath = computed(() => result.value.states.slice(1).map((value, index) => (
  `${index === 0 ? 'M' : 'L'} ${timelineX(index)} ${stateY(value)}`
)).join(' '))
</script>

<template>
  <MlExperimentFrame
    index="16-A"
    title="把同一个状态单元沿钢卷时间轴展开"
    question="循环图展开后出现许多单元，为什么参数仍只有一份？跨钢卷保留状态又会把上一卷的历史带到哪里？"
    formula="hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b)"
  >
    <template #visual>
      <figure class="state-unroller">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`序列${sequenceLength}步，最终状态${result.finalState.toFixed(3)}，事件灵敏度${result.eventSensitivity.toExponential(2)}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="unroller-bg" />
          <text x="32" y="37" class="panel-title">连续退火炉历史记录 · 同一转移单元重复调用</text>
          <rect x="32" y="50" width="536" height="112" rx="7" class="timeline-panel" />
          <line x1="48" y1="100" x2="552" y2="100" class="state-rail" />
          <g v-for="(_, index) in result.inputs" :key="`step-${index}`">
            <line v-if="index<result.inputs.length-1" :x1="timelineX(index)+15" y1="100" :x2="timelineX(index+1)-15" y2="100" class="shared-link" />
            <rect :x="timelineX(index)-15" y="82" width="30" height="36" rx="5" :class="['cell',{active:index===result.activeIndex,event:index===result.eventIndex}]" />
            <text :x="timelineX(index)" y="96" class="cell-name">A</text>
            <text :x="timelineX(index)" y="108" class="cell-state">{{ result.states[index+1].toFixed(2) }}</text>
            <line :x1="timelineX(index)" y1="126" :x2="timelineX(index)" y2="141" class="input-link" />
            <text :x="timelineX(index)" y="153" :class="['input-value',{event:index===result.eventIndex}]">x{{ index+1 }}={{ result.inputs[index].toFixed(1) }}</text>
            <text :x="timelineX(index)" y="72" class="step-label">t{{ index+1 }}</text>
          </g>
          <text x="44" y="65" class="initial-label">h₀={{ result.initialState.toFixed(2) }}</text>
          <text x="556" y="65" class="shared-label" text-anchor="end">每一步共享 Wₓ、Wₕ、b</text>

          <rect x="32" y="177" width="536" height="91" rx="7" class="trace-panel" />
          <line x1="48" y1="231" x2="552" y2="231" class="zero-line" />
          <path :d="statePath" class="state-trace" />
          <circle v-for="(value,index) in result.states.slice(1)" :key="`trace-${index}`" :cx="timelineX(index)" :cy="stateY(value)" r="4" :class="['trace-point',{active:index===result.activeIndex}]" />
          <text x="44" y="194" class="trace-label">隐藏状态 hₜ</text>
          <text x="556" y="194" class="trace-label" text-anchor="end">事件位于 t{{ result.eventIndex+1 }}</text>
          <rect x="32" y="282" width="536" height="17" rx="3" class="strip" />
          <text x="44" y="294" class="strip-label">COIL-16 · 时间从左向右 · {{ resetMode==='reset' ? '卷首状态已复位' : '携带上一卷末状态' }}</text>
        </svg>
        <figcaption><span><i class="shared" />共享单元</span><span><i class="event" />热扰动事件</span><span><i class="active" />当前时刻</span><span><i class="trace" />状态轨迹</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <label><span>序列长度 T <output>{{ sequenceLength }}</output></span><input v-model.number="sequenceLength" type="range" min="4" max="12" step="1"></label>
      <label><span>递推增益 Wₕ <output>{{ recurrentGain.toFixed(2) }}</output></span><input v-model.number="recurrentGain" type="range" min=".2" max="1.15" step=".01"></label>
      <label><span>输入增益 Wₓ <output>{{ inputGain.toFixed(2) }}</output></span><input v-model.number="inputGain" type="range" min=".2" max="1.4" step=".02"></label>
      <label><span>扰动时刻 <output>t{{ result.eventIndex+1 }}</output></span><input v-model.number="eventStep" type="range" min="0" :max="sequenceLength-1" step="1"></label>
      <label><span>检查时刻 <output>t{{ result.activeIndex+1 }}</output></span><input v-model.number="activeStep" type="range" min="0" :max="sequenceLength-1" step="1"></label>
      <div class="reset-switch" role="group" aria-label="状态边界策略">
        <button type="button" :aria-pressed="resetMode==='reset'" @click="resetMode='reset'">卷首复位 h₀=0</button>
        <button type="button" :aria-pressed="resetMode==='carry'" @click="resetMode='carry'">跨卷携带 h₀</button>
      </div>
    </template>
    <template #metrics>
      <div><span>最终状态 hT</span><strong>{{ result.finalState.toFixed(3) }}</strong></div>
      <div><span>事件对 hT 的灵敏度</span><strong>{{ result.eventSensitivity.toExponential(2) }}</strong></div>
      <div><span>共享 / 不共享参数</span><strong>3 / {{ 3*sequenceLength }}</strong></div>
      <div><span>当前可见历史</span><strong>x₁…x{{ result.activeIndex+1 }}</strong></div>
    </template>
    <template #note>
      展开图中的 A 是同一个函数实例，不是 T 套参数。状态能跨窗口携带，但只能在同一条连续设备轨迹内按顺序携带；钢卷乱序、设备切换或维护后仍沿用旧状态，会形成隐蔽的数据串扰。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.state-unroller{margin:0}.state-unroller svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0b1722}.unroller-bg{fill:#0b1722}.state-unroller text{font:700 8px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.timeline-panel,.trace-panel{fill:#101f2c;stroke:#29495f}.state-rail{stroke:#4f718a;stroke-width:2}.shared-link{stroke:#63d0c6;stroke-width:2;stroke-dasharray:5 3}.cell{fill:#183044;stroke:#4f718a;stroke-width:1.2}.cell.event{fill:#56302b;stroke:#f0a65a}.cell.active{stroke:#f3c85b;stroke-width:2.2}.cell-name{fill:#63d0c6;text-anchor:middle;font-size:9px}.cell-state{fill:#dbeaf0;text-anchor:middle}.input-link{stroke:#4f718a}.input-value{fill:#829eaf;text-anchor:middle;font-size:7px}.input-value.event{fill:#f0a65a}.step-label{fill:#829eaf;text-anchor:middle}.initial-label,.shared-label,.trace-label{fill:#829eaf}.zero-line{stroke:#29495f;stroke-dasharray:4 4}.state-trace{fill:none;stroke:#63d0c6;stroke-width:2.3}.trace-point{fill:#63d0c6;stroke:#0b1722;stroke-width:1.5}.trace-point.active{fill:#f3c85b}.strip{fill:#1f3035;stroke:#415d66}.strip-label{fill:#b8cbd1}.state-unroller figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.state-unroller figcaption span{display:inline-flex;align-items:center;gap:5px}.state-unroller figcaption i{width:12px;height:8px;border-radius:2px}.state-unroller figcaption i.shared{background:#183044;border:1px solid #63d0c6}.state-unroller figcaption i.event{background:#f0a65a}.state-unroller figcaption i.active{border:2px solid #f3c85b}.state-unroller figcaption i.trace{height:2px;background:#63d0c6}.reset-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.reset-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 9px ui-monospace,monospace;cursor:pointer}.reset-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.reset-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}@media(max-width:560px){.state-unroller{overflow-x:auto;overscroll-behavior-inline:contain}.state-unroller svg{width:560px;max-width:none}.state-unroller figcaption{min-width:560px}}
</style>
