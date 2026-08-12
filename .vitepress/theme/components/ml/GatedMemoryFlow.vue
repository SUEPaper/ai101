<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type CellMode = 'lstm' | 'gru'
type ScenarioMode = 'remember' | 'replace' | 'leak'
type GruConvention = 'retain' | 'write'

const cellMode = ref<CellMode>('lstm')
const scenario = ref<ScenarioMode>('remember')
const retention = ref(.96)
const activeStep = ref(4)
const gruConvention = ref<GruConvention>('retain')

const result = computed(() => {
  const candidate = [0, 1, 0, 0, 0, -.8, 0, 0]
  const states = [0]
  const gates = [] as Array<{ retain: number; write: number; displayedUpdate: number; candidate: number }>
  let directGradient = 1
  for (let index = 0; index < candidate.length; index += 1) {
    const isInitialWrite = index === 1
    const isReplacement = scenario.value === 'replace' && index === 5
    let retain = scenario.value === 'leak' ? Math.min(retention.value, .82) : retention.value
    let write = cellMode.value === 'lstm' ? .02 : 1 - retain
    if (isInitialWrite || isReplacement) {
      retain = .05
      write = .95
    }
    const next = retain * states.at(-1)! + write * candidate[index]
    if (index > 1) directGradient *= retain
    const displayedUpdate = gruConvention.value === 'retain' ? retain : 1 - retain
    gates.push({ retain, write, displayedUpdate, candidate: candidate[index] })
    states.push(next)
  }
  const index = Math.min(activeStep.value, 7)
  return { candidate, states, gates, directGradient, index, finalState: states.at(-1)! }
})

const nodeX = (index: number) => 58 + index * 69
const stateColor = (value: number) => value >= 0 ? '#63d0c6' : '#e35d68'
</script>

<template>
  <MlExperimentFrame
    index="16-C"
    title="沿门控记忆轨道检查保留、写入与替换"
    question="LSTM 的细胞状态和 GRU 的加法更新怎样形成短路径？不同资料把 z 与 1−z 对调时，如何用边界行为判定语义？"
    formula="memoryₜ = retainₜ ⊙ memoryₜ₋₁ + writeₜ ⊙ candidateₜ"
  >
    <template #visual>
      <figure class="memory-flow">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`${cellMode.toUpperCase()}最终记忆${result.finalState.toFixed(3)}，直接梯度${result.directGradient.toExponential(2)}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="flow-bg" />
          <text x="32" y="37" class="panel-title">{{ cellMode.toUpperCase() }} 记忆轨道 · {{ scenario==='remember' ? '写入后保持' : scenario==='replace' ? '新事件替换旧记忆' : '持续泄漏' }}</text>
          <rect x="32" y="50" width="536" height="122" rx="7" class="memory-panel" />
          <line x1="58" y1="96" x2="541" y2="96" class="memory-rail" />
          <g v-for="(_,index) in result.candidate" :key="`memory-${index}`">
            <line v-if="index<7" :x1="nodeX(index)+17" y1="96" :x2="nodeX(index+1)-17" y2="96" :style="{stroke:stateColor(result.states[index+1])}" class="memory-link" />
            <circle :cx="nodeX(index)" cy="96" r="16" :class="['memory-node',{active:index===result.index}]" :style="{stroke:stateColor(result.states[index+1])}" />
            <text :x="nodeX(index)" y="100" class="memory-value">{{ result.states[index+1].toFixed(2) }}</text>
            <text :x="nodeX(index)" y="70" class="step-label">t{{ index+1 }}</text>
            <path :d="`M ${nodeX(index)} 151 L ${nodeX(index)} 116`" class="write-arrow" />
            <text :x="nodeX(index)" y="158" :class="['candidate-value',{event:Math.abs(result.candidate[index])>.1}]">g={{ result.candidate[index].toFixed(1) }}</text>
          </g>

          <rect x="32" y="187" width="164" height="95" rx="7" class="gate-card" />
          <rect x="218" y="187" width="164" height="95" rx="7" class="gate-card" />
          <rect x="404" y="187" width="164" height="95" rx="7" class="gate-card active" />
          <text x="46" y="207" class="card-title">保留旧记忆</text>
          <text x="46" y="234" class="gate-value">{{ result.gates[result.index].retain.toFixed(2) }}</text>
          <rect x="46" y="247" width="134" height="9" rx="4" class="gate-track" /><rect x="46" y="247" :width="134*result.gates[result.index].retain" height="9" rx="4" class="retain-fill" />
          <text x="232" y="207" class="card-title">写入候选</text>
          <text x="232" y="234" class="gate-value">{{ result.gates[result.index].write.toFixed(2) }}</text>
          <rect x="232" y="247" width="134" height="9" rx="4" class="gate-track" /><rect x="232" y="247" :width="134*result.gates[result.index].write" height="9" rx="4" class="write-fill" />
          <text x="418" y="207" class="card-title">当前公式记号</text>
          <text x="418" y="228" class="formula-label">{{ cellMode==='lstm' ? 'fₜ·cₜ₋₁ + iₜ·c̃ₜ' : gruConvention==='retain' ? 'zₜ·hₜ₋₁ + (1−zₜ)·h̃ₜ' : '(1−zₜ)·hₜ₋₁ + zₜ·h̃ₜ' }}</text>
          <text x="418" y="253" class="convention-value">{{ cellMode==='lstm' ? `f=${result.gates[result.index].retain.toFixed(2)} · i=${result.gates[result.index].write.toFixed(2)}` : `显示 z=${result.gates[result.index].displayedUpdate.toFixed(2)}` }}</text>
          <text x="418" y="270" class="card-note">边界值决定门的真实语义</text>
        </svg>
        <figcaption><span><i class="retain" />保留通路</span><span><i class="write" />候选写入</span><span><i class="event" />事件候选</span><span><i class="active" />当前时刻</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="cell-switch" role="group" aria-label="门控单元类型">
        <button type="button" :aria-pressed="cellMode==='lstm'" @click="cellMode='lstm'">LSTM 细胞状态</button>
        <button type="button" :aria-pressed="cellMode==='gru'" @click="cellMode='gru'">GRU 隐藏状态</button>
      </div>
      <div class="scenario-switch" role="group" aria-label="记忆场景">
        <button type="button" :aria-pressed="scenario==='remember'" @click="scenario='remember'">写入后保持</button>
        <button type="button" :aria-pressed="scenario==='replace'" @click="scenario='replace'">t6 替换记忆</button>
        <button type="button" :aria-pressed="scenario==='leak'" @click="scenario='leak'">持续泄漏</button>
      </div>
      <label><span>常态保留比例 <output>{{ retention.toFixed(2) }}</output></span><input v-model.number="retention" type="range" min=".55" max=".995" step=".005"></label>
      <label><span>检查时刻 <output>t{{ result.index+1 }}</output></span><input v-model.number="activeStep" type="range" min="0" max="7" step="1"></label>
      <div v-if="cellMode==='gru'" class="convention-switch" role="group" aria-label="GRU 更新门记号约定">
        <button type="button" :aria-pressed="gruConvention==='retain'" @click="gruConvention='retain'">z 表示保留旧状态</button>
        <button type="button" :aria-pressed="gruConvention==='write'" @click="gruConvention='write'">z 表示写入候选</button>
      </div>
      <p>切换 GRU 记号时，状态轨迹保持不变，只改变显示的 z 与公式位置。这是阅读不同资料时最可靠的核对方法：令 z=0 或 1，观察究竟保留旧状态还是写入候选。</p>
    </template>
    <template #metrics>
      <div><span>最终记忆</span><strong>{{ result.finalState.toFixed(3) }}</strong></div>
      <div><span>写入后直接梯度</span><strong>{{ result.directGradient.toExponential(2) }}</strong></div>
      <div><span>状态层参数倍率</span><strong>{{ cellMode==='lstm' ? '4×' : '3×' }}</strong></div>
      <div><span>当前保留 / 写入</span><strong>{{ result.gates[result.index].retain.toFixed(2) }} / {{ result.gates[result.index].write.toFixed(2) }}</strong></div>
    </template>
    <template #note>
      这里把候选分支视为与旧状态独立，用来隔离加法通路的直接梯度。真实 LSTM/GRU 的门和候选还依赖 hₜ₋₁，因此总 Jacobian 包含额外路径；门控提供更短、更可调的通路，但不保证永不消失。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.memory-flow{margin:0}.memory-flow svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0b1722}.flow-bg{fill:#0b1722}.memory-flow text{font:700 8px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.memory-panel{fill:#101f2c;stroke:#29495f}.memory-rail{stroke:#4f718a;stroke-width:2}.memory-link{stroke-width:3}.memory-node{fill:#142638;stroke-width:2}.memory-node.active{fill:#2c3040;stroke:#f3c85b!important;stroke-width:3}.memory-value{fill:#eaf5f4;text-anchor:middle}.step-label{fill:#829eaf;text-anchor:middle}.write-arrow{fill:none;stroke:#f0a65a;stroke-width:1.4}.candidate-value{fill:#718b98;text-anchor:middle}.candidate-value.event{fill:#f0a65a}.gate-card{fill:#101f2c;stroke:#365469}.gate-card.active{stroke:#f3c85b}.card-title{fill:#829eaf}.gate-value{fill:#f3c85b;font-size:16px}.gate-track{fill:#273b4a}.retain-fill{fill:#63d0c6}.write-fill{fill:#f0a65a}.formula-label{fill:#dbeaf0;font-size:7px}.convention-value{fill:#f3c85b;font-size:10px}.card-note{fill:#829eaf}.memory-flow figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.memory-flow figcaption span{display:inline-flex;align-items:center;gap:5px}.memory-flow figcaption i{width:12px;height:8px;border-radius:2px}.memory-flow figcaption i.retain{background:#63d0c6}.memory-flow figcaption i.write{background:#f0a65a}.memory-flow figcaption i.event{border:2px solid #f0a65a}.memory-flow figcaption i.active{border:2px solid #f3c85b}.cell-switch,.convention-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.scenario-switch{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.cell-switch button,.convention-switch button,.scenario-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 8px ui-monospace,monospace;cursor:pointer}.cell-switch button[aria-pressed="true"],.convention-switch button[aria-pressed="true"],.scenario-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.cell-switch button:focus-visible,.convention-switch button:focus-visible,.scenario-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.memory-flow{overflow-x:auto;overscroll-behavior-inline:contain}.memory-flow svg{width:560px;max-width:none}.memory-flow figcaption{min-width:560px}}
</style>
