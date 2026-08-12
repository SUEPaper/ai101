<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Action = 'N' | 'E' | 'S' | 'W'
type Algorithm = 'sarsa' | 'expected' | 'q'

const actions: Action[] = ['N', 'E', 'S', 'W']
const arrows: Record<Action, string> = { N: '↑', E: '→', S: '↓', W: '←' }
const delta: Record<Action, [number, number]> = { N: [-1, 0], E: [0, 1], S: [1, 0], W: [0, -1] }
const left: Record<Action, Action> = { N: 'W', E: 'N', S: 'E', W: 'S' }
const right: Record<Action, Action> = { N: 'E', E: 'S', S: 'W', W: 'N' }
const wallKeys = new Set(['1-1', '2-1'])
const terminalKeys = new Set(['0-3', '1-3'])
const stateKeys = Array.from({ length: 16 }, (_, index) => `${Math.floor(index / 4)}-${index % 4}`)
  .filter(key => !wallKeys.has(key))
const stateIndex = new Map(stateKeys.map((key, index) => [key, index]))
const displayCells = Array.from({ length: 16 }, (_, index) => {
  const key = `${Math.floor(index / 4)}-${index % 4}`
  return {
    key,
    state: stateIndex.get(key),
    kind: wallKeys.has(key) ? 'wall' : key === '0-3' ? 'goal' : key === '1-3' ? 'hazard' : 'state',
  }
})
const start = stateIndex.get('3-0')!

const algorithm = ref<Algorithm>('sarsa')
const epsilon = ref(0.20)
const alpha = ref(0.18)
const slip = ref(0.18)
const q = ref(stateKeys.map(() => actions.map(() => 0)))
const visits = ref(stateKeys.map(() => actions.map(() => 0)))
const episodes = ref(0)
const successes = ref(0)
const hazards = ref(0)
const returns = ref<number[]>([])
const lastRoute = ref<string[]>([])
let randomState = 310031

function random() {
  randomState = (1664525 * randomState + 1013904223) >>> 0
  return randomState / 4294967296
}

function move(key: string, action: Action) {
  if (terminalKeys.has(key)) return key
  const [row, col] = key.split('-').map(Number)
  const [dr, dc] = delta[action]
  const candidate = `${row + dr}-${col + dc}`
  if (row + dr < 0 || row + dr > 3 || col + dc < 0 || col + dc > 3 || wallKeys.has(candidate)) return key
  return candidate
}

function sampleTransition(state: number, action: number) {
  const requested = actions[action]
  const draw = random()
  const actual = draw < 1 - slip.value
    ? requested
    : draw < 1 - slip.value / 2
      ? left[requested]
      : right[requested]
  const nextKey = move(stateKeys[state], actual)
  const nextState = stateIndex.get(nextKey)!
  const reward = nextKey === '0-3' ? 14 : nextKey === '1-3' ? -22 : -0.45
  return { nextState, reward, terminated: terminalKeys.has(nextKey), nextKey }
}

function greedyActions(row: number[]) {
  const maximum = Math.max(...row)
  return row.map((value, index) => ({ value, index }))
    .filter(item => Math.abs(item.value - maximum) < 1e-9)
    .map(item => item.index)
}

function selectAction(state: number) {
  if (random() < epsilon.value) return Math.floor(random() * actions.length)
  const greedy = greedyActions(q.value[state])
  return greedy[Math.floor(random() * greedy.length)]
}

function expectedQ(state: number) {
  const greedy = new Set(greedyActions(q.value[state]))
  return q.value[state].reduce((sum, value, action) => {
    const probability = epsilon.value / actions.length
      + (greedy.has(action) ? (1 - epsilon.value) / greedy.size : 0)
    return sum + probability * value
  }, 0)
}

function runEpisode() {
  let state = start
  let action = selectAction(state)
  let episodeReturn = 0
  let discount = 1
  const route = [stateKeys[state]]

  for (let step = 0; step < 100; step += 1) {
    const transition = sampleTransition(state, action)
    episodeReturn += discount * transition.reward
    discount *= 0.94
    visits.value[state][action] += 1
    let target = transition.reward
    let nextAction = 0

    if (!transition.terminated) {
      nextAction = selectAction(transition.nextState)
      const future = algorithm.value === 'sarsa'
        ? q.value[transition.nextState][nextAction]
        : algorithm.value === 'expected'
          ? expectedQ(transition.nextState)
          : Math.max(...q.value[transition.nextState])
      target += 0.94 * future
    }

    q.value[state][action] += alpha.value * (target - q.value[state][action])
    route.push(transition.nextKey)
    if (transition.terminated) {
      if (transition.nextKey === '0-3') successes.value += 1
      else hazards.value += 1
      break
    }
    state = transition.nextState
    action = nextAction
  }

  episodes.value += 1
  returns.value.push(episodeReturn)
  returns.value = returns.value.slice(-40)
  lastRoute.value = route.slice(-20)
}

function runBatch(count: number) {
  for (let index = 0; index < count; index += 1) runEpisode()
}

function reset() {
  q.value = stateKeys.map(() => actions.map(() => 0))
  visits.value = stateKeys.map(() => actions.map(() => 0))
  episodes.value = 0
  successes.value = 0
  hazards.value = 0
  returns.value = []
  lastRoute.value = []
  randomState = 310031
}

function setAlgorithm(next: Algorithm) {
  algorithm.value = next
  reset()
}

const averageReturn = computed(() => returns.value.length
  ? returns.value.reduce((sum, value) => sum + value, 0) / returns.value.length
  : 0)
const hazardRate = computed(() => episodes.value ? hazards.value / episodes.value : 0)
const coverage = computed(() => {
  const nonterminal = stateKeys.flatMap((key, state) => terminalKeys.has(key) ? [] : visits.value[state])
  return nonterminal.filter(value => value > 0).length / nonterminal.length
})
const maxAbsQ = computed(() => Math.max(1, ...q.value.flat().map(Math.abs)))
const tracePoints = computed(() => {
  if (!returns.value.length) return ''
  const width = 720
  const height = 90
  const minimum = Math.min(...returns.value, -5)
  const maximum = Math.max(...returns.value, 8)
  return returns.value.map((value, index) => {
    const x = returns.value.length === 1 ? 0 : index * width / (returns.value.length - 1)
    const y = height - (value - minimum) / Math.max(1e-9, maximum - minimum) * height
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
})

function cellArrow(state: number) {
  if (q.value[state].every(value => Math.abs(value) < 1e-10)) return '·'
  return arrows[actions[greedyActions(q.value[state])[0]]]
}

function cellValue(state: number) {
  return Math.max(...q.value[state])
}

function cellStyle(state: number) {
  const value = cellValue(state)
  const strength = Math.min(0.62, 0.10 + Math.abs(value) / maxAbsQ.value * 0.52)
  return { background: value >= 0 ? `rgba(79,156,168,${strength})` : `rgba(216,93,74,${strength})` }
}
</script>

<template>
  <MlExperimentFrame
    index="31-C"
    title="Q 路线学习回放器"
    question="让三种算法在同一随机走廊中逐回合学习；观察价值传播、探索覆盖与隔离事件，而不是只看最后一根箭头。"
    formula="Q(S,A) ← Q(S,A) + α[target − Q(S,A)]"
  >
    <template #visual>
      <div class="playback">
        <div class="fanfold-label">
          <span>WAFER AMR / SAMPLE-ONLY COMMISSIONING</span>
          <b>{{ algorithm.toUpperCase() }}</b>
        </div>
        <div class="route-grid" aria-label="Q 值与贪心路线">
          <div
            v-for="cell in displayCells"
            :key="cell.key"
            :class="['route-cell', cell.kind, { recent: lastRoute.includes(cell.key) }]"
            :style="cell.kind === 'state' && cell.state !== undefined ? cellStyle(cell.state) : undefined"
          >
            <template v-if="cell.kind === 'state' && cell.state !== undefined">
              <span>{{ cell.key }}</span>
              <b>{{ cellArrow(cell.state) }}</b>
              <strong>{{ cellValue(cell.state).toFixed(1) }}</strong>
              <small>n={{ visits[cell.state].reduce((sum, value) => sum + value, 0) }}</small>
            </template>
            <template v-else>
              <b>{{ cell.kind === 'wall' ? '▧' : cell.kind === 'goal' ? '+14' : '−22' }}</b>
              <small>{{ cell.kind === 'wall' ? '隔墙' : cell.kind === 'goal' ? '交付' : '隔离' }}</small>
            </template>
          </div>
        </div>
        <div class="oscilloscope">
          <div class="scope-head">
            <span>LAST 40 DISCOUNTED RETURNS</span>
            <b>{{ averageReturn.toFixed(2) }}</b>
          </div>
          <svg viewBox="0 0 720 100" role="img" aria-label="最近回合折扣回报轨迹">
            <path d="M0 50 H720" class="zero" />
            <polyline v-if="tracePoints" :points="tracePoints" class="return-line" />
          </svg>
        </div>
      </div>
    </template>

    <template #controls>
      <div class="algorithm" role="group" aria-label="价值控制算法">
        <button type="button" :aria-pressed="algorithm === 'sarsa'" @click="setAlgorithm('sarsa')">SARSA</button>
        <button type="button" :aria-pressed="algorithm === 'expected'" @click="setAlgorithm('expected')">Expected</button>
        <button type="button" :aria-pressed="algorithm === 'q'" @click="setAlgorithm('q')">Q-learning</button>
      </div>
      <label class="control">
        <span>探索率 ε <output>{{ epsilon.toFixed(2) }}</output></span>
        <input v-model.number="epsilon" type="range" min="0" max="0.60" step="0.02" aria-label="探索率" @change="reset">
      </label>
      <label class="control">
        <span>学习率 α <output>{{ alpha.toFixed(2) }}</output></span>
        <input v-model.number="alpha" type="range" min="0.02" max="0.50" step="0.02" aria-label="学习率" @change="reset">
      </label>
      <label class="control">
        <span>偏移概率 <output>{{ slip.toFixed(2) }}</output></span>
        <input v-model.number="slip" type="range" min="0" max="0.50" step="0.02" aria-label="偏移概率" @change="reset">
      </label>
      <div class="transport">
        <button type="button" @click="runEpisode">运行 1 回合</button>
        <button type="button" @click="runBatch(50)">快进 50</button>
        <button type="button" @click="runBatch(500)">快进 500</button>
      </div>
      <button type="button" class="reset" @click="reset">清空 Q 记录</button>
    </template>

    <template #metrics>
      <div><span>训练回合</span><strong>{{ episodes }}</strong></div>
      <div><span>最近平均回报</span><strong>{{ averageReturn.toFixed(2) }}</strong></div>
      <div><span>隔离事件率</span><strong>{{ (hazardRate * 100).toFixed(1) }}%</strong></div>
      <div><span>状态动作覆盖</span><strong>{{ (coverage * 100).toFixed(0) }}%</strong></div>
    </template>

    <template #note>
      这是合格仿真器内的教学回放。固定 ε 时，SARSA/Expected SARSA 把持续探索写进价值；Q-learning 的 max 指向关闭探索后的贪心目标，但训练期隔离事件仍属于行为策略成本。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.playback{padding:16px;color:#20272b;background:#deddd3;border:1px solid #7c9198;box-shadow:inset 10px 0 #c6c5bb,inset -10px 0 #c6c5bb,5px 7px 0 #102832}.fanfold-label{display:flex;justify-content:space-between;gap:12px;margin:0 16px 13px;padding-bottom:8px;border-bottom:2px solid #325c70;font:800 9px/1.2 ui-monospace,monospace;letter-spacing:.08em}.fanfold-label b{color:#66517f}.route-grid{display:grid;grid-template-columns:repeat(4,minmax(64px,1fr));gap:5px;margin:0 16px}.route-cell{position:relative;display:grid;grid-template-columns:1fr auto;align-items:center;min-height:76px;padding:8px;border:1px solid #87989c;background:#ebe9de;font-family:ui-monospace,monospace}.route-cell:after{content:"";position:absolute;inset:3px;border:1px dashed rgba(50,92,112,.17);pointer-events:none}.route-cell.recent{outline:3px solid #66517f;outline-offset:-4px}.route-cell span{font-size:7px}.route-cell b{z-index:1;color:#203f4b;font-size:23px}.route-cell strong{z-index:1;font-size:11px}.route-cell small{z-index:1;text-align:right;font-size:7px}.route-cell.wall{color:#737771;background:repeating-linear-gradient(135deg,#bbbcb6 0,#bbbcb6 6px,#929791 7px,#929791 9px)}.route-cell.goal{color:#fff;background:#2f7a76}.route-cell.hazard{color:#fff;background:#b84b3d}.route-cell.wall,.route-cell.goal,.route-cell.hazard{grid-template-columns:1fr;text-align:center}.route-cell.wall b,.route-cell.goal b,.route-cell.hazard b{color:inherit}.oscilloscope{margin:16px 16px 0;padding:9px;border:1px solid #325c70;background:#183039}.scope-head{display:flex;justify-content:space-between;color:#b9d5d8;font:700 8px/1 ui-monospace,monospace;letter-spacing:.07em}.scope-head b{color:#f0c45a}.oscilloscope svg{display:block;width:100%;height:82px;margin-top:8px;background:repeating-linear-gradient(0deg,transparent 0,transparent 19px,rgba(79,156,168,.16) 20px),repeating-linear-gradient(90deg,transparent 0,transparent 35px,rgba(79,156,168,.12) 36px)}.zero{fill:none;stroke:#6d8288;stroke-width:1;stroke-dasharray:5 5}.return-line{fill:none;stroke:#f0c45a;stroke-width:3;stroke-linejoin:round}.algorithm{display:grid;grid-template-columns:1fr;gap:6px}.algorithm button,.transport button,.reset{min-height:36px;border:1px solid #556f79;border-radius:2px;color:#d7e1e5;background:#213741;font:750 9px/1.2 ui-monospace,monospace}.algorithm button[aria-pressed="true"]{border-color:#a9dfe1;color:#15272d;background:#79b9bf}.control{display:grid;gap:6px;color:#c8d6df;font-size:10px}.control span{display:flex;justify-content:space-between}.control output{color:#8dd5dd;font-family:ui-monospace,monospace}.control input{width:100%;accent-color:#4f9ca8}.transport{display:grid;grid-template-columns:1fr 1fr;gap:6px}.transport button:last-child{grid-column:1/-1;color:#fff;background:#66517f}.reset{border-style:dashed}.algorithm button:focus-visible,.transport button:focus-visible,.reset:focus-visible,input:focus-visible{outline:3px solid #f0c45a;outline-offset:2px}@media(max-width:560px){.playback{padding:12px 5px}.fanfold-label{display:grid}.route-grid{grid-template-columns:repeat(4,minmax(50px,1fr));gap:3px}.route-cell{min-height:66px;padding:5px}.route-cell b{font-size:18px}.oscilloscope{margin-top:12px}}
</style>
