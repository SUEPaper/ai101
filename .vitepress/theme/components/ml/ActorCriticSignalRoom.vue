<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Action = 'N' | 'E' | 'S' | 'W'

const actions: Action[] = ['N', 'E', 'S', 'W']
const arrows: Record<Action, string> = { N: '↑', E: '→', S: '↓', W: '←' }
const deltaMove: Record<Action, [number, number]> = { N: [-1, 0], E: [0, 1], S: [1, 0], W: [0, -1] }
const left: Record<Action, Action> = { N: 'W', E: 'N', S: 'E', W: 'S' }
const right: Record<Action, Action> = { N: 'E', E: 'S', S: 'W', W: 'N' }
const wallKeys = new Set(['1-1', '2-1'])
const terminalKeys = new Set(['0-3', '1-3'])
const stateKeys = Array.from({ length: 16 }, (_, index) => `${Math.floor(index / 4)}-${index % 4}`)
  .filter(key => !wallKeys.has(key))
const stateIndex = new Map(stateKeys.map((key, index) => [key, index]))
const displayCells = Array.from({ length: 16 }, (_, index) => {
  const key = `${Math.floor(index / 4)}-${index % 4}`
  return { key, state: stateIndex.get(key), kind: wallKeys.has(key) ? 'wall' : key === '0-3' ? 'goal' : key === '1-3' ? 'hazard' : 'state' }
})
const start = stateIndex.get('3-0')!
const actorRate = ref(0.055)
const criticRate = ref(0.18)
const entropyWeight = ref(0.015)
const slip = ref(0.18)
const logits = ref(stateKeys.map(() => actions.map(() => 0)))
const values = ref(stateKeys.map(() => 0))
const episodes = ref(0)
const successes = ref(0)
const hazards = ref(0)
const returns = ref<number[]>([])
const lastState = ref(start)
const lastAction = ref(0)
const lastReward = ref(0)
const lastDelta = ref(0)
const lastTarget = ref(0)
let randomState = 320032

function random() {
  randomState = (1664525 * randomState + 1013904223) >>> 0
  return randomState / 4294967296
}

function probabilities(state: number) {
  const maximum = Math.max(...logits.value[state])
  const weights = logits.value[state].map(value => Math.exp(value - maximum))
  const total = weights.reduce((sum, value) => sum + value, 0)
  return weights.map(value => value / total)
}

function sample(probability: number[]) {
  const draw = random()
  let cumulative = 0
  for (let index = 0; index < probability.length; index += 1) {
    cumulative += probability[index]
    if (draw <= cumulative) return index
  }
  return probability.length - 1
}

function move(key: string, action: Action) {
  if (terminalKeys.has(key)) return key
  const [row, col] = key.split('-').map(Number)
  const [dr, dc] = deltaMove[action]
  const candidate = `${row + dr}-${col + dc}`
  if (row + dr < 0 || row + dr > 3 || col + dc < 0 || col + dc > 3 || wallKeys.has(candidate)) return key
  return candidate
}

function transition(state: number, action: number) {
  const requested = actions[action]
  const draw = random()
  const actual = draw < 1 - slip.value ? requested : draw < 1 - slip.value / 2 ? left[requested] : right[requested]
  const nextKey = move(stateKeys[state], actual)
  const nextState = stateIndex.get(nextKey)!
  const reward = nextKey === '0-3' ? 14 : nextKey === '1-3' ? -22 : -0.45
  return { nextState, nextKey, reward, terminated: terminalKeys.has(nextKey) }
}

function runEpisode() {
  let state = start
  let episodeReturn = 0
  let discount = 1
  for (let step = 0; step < 120; step += 1) {
    const pi = probabilities(state)
    const action = sample(pi)
    const result = transition(state, action)
    const target = result.reward + (result.terminated ? 0 : 0.94 * values.value[result.nextState])
    const tdError = target - values.value[state]
    values.value[state] += criticRate.value * tdError
    const entropy = -pi.reduce((sum, p) => sum + p * Math.log(Math.max(p, 1e-12)), 0)
    for (let candidate = 0; candidate < actions.length; candidate += 1) {
      const score = (candidate === action ? 1 : 0) - pi[candidate]
      const entropyGradient = -pi[candidate] * (Math.log(Math.max(pi[candidate], 1e-12)) + entropy)
      logits.value[state][candidate] += actorRate.value * (tdError * score + entropyWeight.value * entropyGradient)
    }
    episodeReturn += discount * result.reward
    discount *= 0.94
    lastState.value = state
    lastAction.value = action
    lastReward.value = result.reward
    lastDelta.value = tdError
    lastTarget.value = target
    if (result.terminated) {
      if (result.nextKey === '0-3') successes.value += 1
      else hazards.value += 1
      break
    }
    state = result.nextState
  }
  episodes.value += 1
  returns.value.push(episodeReturn)
  returns.value = returns.value.slice(-50)
}

function runBatch(count: number) {
  for (let index = 0; index < count; index += 1) runEpisode()
}

function reset() {
  logits.value = stateKeys.map(() => actions.map(() => 0))
  values.value = stateKeys.map(() => 0)
  episodes.value = 0
  successes.value = 0
  hazards.value = 0
  returns.value = []
  lastState.value = start
  lastAction.value = 0
  lastReward.value = 0
  lastDelta.value = 0
  lastTarget.value = 0
  randomState = 320032
}

const averageReturn = computed(() => returns.value.length ? returns.value.reduce((sum, value) => sum + value, 0) / returns.value.length : 0)
const successRate = computed(() => episodes.value ? successes.value / episodes.value : 0)
const hazardRate = computed(() => episodes.value ? hazards.value / episodes.value : 0)
const startEntropy = computed(() => -probabilities(start).reduce((sum, p) => sum + p * Math.log(Math.max(p, 1e-12)), 0))
const busState = computed(() => lastDelta.value > 1e-9 ? 'positive' : lastDelta.value < -1e-9 ? 'negative' : 'idle')

function bestAction(state: number) {
  const pi = probabilities(state)
  return pi.indexOf(Math.max(...pi))
}

function cellStyle(state: number) {
  const probability = Math.max(...probabilities(state))
  const alpha = 0.12 + Math.max(0, probability - 0.25) * 0.72
  return { background: `rgba(63,143,131,${alpha.toFixed(3)})` }
}
</script>

<template>
  <MlExperimentFrame
    index="32-C"
    title="Actor–Critic 信号控制室"
    question="让 Actor 负责动作分布、Critic 负责 TD target；观察 advantage 总线如何把正负评价送回本次动作。"
    formula="δ = r + γV(s′) − V(s)"
  >
    <template #visual>
      <div class="control-room">
        <div class="rack-label"><span>WAFER AMR / SIMULATOR ONLY</span><b>A2C SIGNAL ROOM</b></div>
        <div class="room-grid">
          <div class="mimic-panel">
            <div class="route-grid" aria-label="策略箭头与 critic 价值">
              <div
                v-for="cell in displayCells"
                :key="cell.key"
                :class="['route-cell', cell.kind, { active: cell.state === lastState }]"
                :style="cell.kind === 'state' && cell.state !== undefined ? cellStyle(cell.state) : undefined"
              >
                <template v-if="cell.kind === 'state' && cell.state !== undefined">
                  <small>{{ cell.key }}</small>
                  <b>{{ arrows[actions[bestAction(cell.state)]] }}</b>
                  <strong>V {{ values[cell.state].toFixed(1) }}</strong>
                </template>
                <template v-else>
                  <b>{{ cell.kind === 'wall' ? '▧' : cell.kind === 'goal' ? '+14' : '−22' }}</b>
                  <small>{{ cell.kind === 'wall' ? '隔墙' : cell.kind === 'goal' ? '交付' : '隔离' }}</small>
                </template>
              </div>
            </div>
          </div>
          <div class="signal-rack">
            <div class="module critic">
              <span>CRITIC / VALUE HEAD</span>
              <strong>V(s) {{ values[lastState].toFixed(3) }}</strong>
              <code>target {{ lastTarget.toFixed(3) }}</code>
            </div>
            <div :class="['advantage-bus', busState]">
              <i></i><b>δ {{ lastDelta >= 0 ? '+' : '' }}{{ lastDelta.toFixed(3) }}</b><i></i>
            </div>
            <div class="module actor">
              <span>ACTOR / POLICY HEAD</span>
              <strong>动作 {{ actions[lastAction] }} · 奖励 {{ lastReward.toFixed(2) }}</strong>
              <div class="prob-bars">
                <i v-for="(p, index) in probabilities(lastState)" :key="actions[index]"><em>{{ actions[index] }}</em><span :style="{ background: `linear-gradient(90deg,#f0a83b ${p * 100}%,#b9af98 0)` }"></span><b>{{ (p * 100).toFixed(0) }}%</b></i>
              </div>
            </div>
          </div>
        </div>
      </div>
    </template>

    <template #controls>
      <label>
        <span>Actor 步长</span>
        <input v-model.number="actorRate" type="range" min="0.01" max="0.12" step="0.005" />
        <output>{{ actorRate.toFixed(3) }}</output>
      </label>
      <label>
        <span>Critic 步长</span>
        <input v-model.number="criticRate" type="range" min="0.04" max="0.35" step="0.01" />
        <output>{{ criticRate.toFixed(2) }}</output>
      </label>
      <label>
        <span>熵系数 β</span>
        <input v-model.number="entropyWeight" type="range" min="0" max="0.08" step="0.005" />
        <output>{{ entropyWeight.toFixed(3) }}</output>
      </label>
      <label>
        <span>地坪偏移概率</span>
        <input v-model.number="slip" type="range" min="0" max="0.4" step="0.02" />
        <output>{{ slip.toFixed(2) }}</output>
      </label>
      <div class="run-buttons">
        <button @click="runEpisode">运行 1 回合</button>
        <button @click="runBatch(50)">运行 50 回合</button>
        <button class="secondary" @click="reset">复位账本</button>
      </div>
    </template>

    <template #metrics>
      <div><span>训练回合</span><strong>{{ episodes }}</strong></div>
      <div><span>近 50 回报</span><strong>{{ averageReturn.toFixed(2) }}</strong></div>
      <div><span>交付 / 隔离率</span><strong>{{ (successRate * 100).toFixed(1) }}% / {{ (hazardRate * 100).toFixed(1) }}%</strong></div>
      <div><span>起点策略熵</span><strong>{{ startEntropy.toFixed(3) }}</strong></div>
    </template>

    <template #note>
      绿色总线表示本次动作比 Critic 预期好，紫色表示更差；Actor 只沿 score 调整该动作分布，Critic 则向 TD target 回归。这里的随机探索仅是隔离仿真，不代表真实设备控制许可。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.control-room{min-width:580px;color:#17324d;font-family:"Arial Narrow","PingFang SC",sans-serif}.rack-label{display:flex;justify-content:space-between;padding:9px 12px;border:1px solid #9b9179;border-bottom:0;border-radius:7px 7px 0 0;background:#ebe4cf;font:800 10px/1.2 ui-monospace,monospace;letter-spacing:.07em}.rack-label b{color:#a64d79}.room-grid{display:grid;grid-template-columns:1.02fr .98fr;gap:12px;padding:14px;border:1px solid #9b9179;border-radius:0 0 7px 7px;background:#d8d0b9}.mimic-panel,.signal-rack{padding:12px;border:2px solid #17324d;border-radius:5px;background:#ebe4cf;box-shadow:inset 0 0 0 3px rgba(23,50,77,.08)}.route-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:4px}.route-cell{display:grid;place-items:center;min-height:62px;border:1px solid #8f856d;background:#dcd3bd}.route-cell small{font:700 8px ui-monospace,monospace}.route-cell b{font-size:22px}.route-cell strong{font:700 9px ui-monospace,monospace}.route-cell.wall{color:#777;background:repeating-linear-gradient(45deg,#c6bca4 0 5px,#ada48f 5px 10px)}.route-cell.goal{color:#fff;background:#3f8f83}.route-cell.hazard{color:#fff;background:#c64d3f}.route-cell.active{outline:3px solid #f0a83b;outline-offset:-3px}.signal-rack{display:grid;grid-template-rows:auto 48px auto;align-content:center}.module{padding:12px;border:1px solid #8d836b;background:#d9d0ba}.module>span{display:block;font:800 9px/1.2 ui-monospace,monospace;letter-spacing:.08em}.module>strong,.module>code{display:block;margin-top:7px}.module>strong{font-size:13px}.module>code{color:#41525b}.advantage-bus{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:center;color:#6c6c65}.advantage-bus i{height:5px;background:#6c6c65}.advantage-bus b{font:800 11px ui-monospace,monospace}.advantage-bus.positive{color:#287568}.advantage-bus.positive i{background:#3f8f83;box-shadow:0 0 9px #3f8f83}.advantage-bus.negative{color:#a64d79}.advantage-bus.negative i{background:#a64d79;box-shadow:0 0 9px #a64d79}.prob-bars{display:grid;gap:5px;margin-top:10px}.prob-bars i{display:grid;grid-template-columns:14px 1fr 34px;gap:5px;align-items:center;font-style:normal}.prob-bars em,.prob-bars b{font:700 9px ui-monospace,monospace}.prob-bars span{height:6px}label{display:grid;grid-template-columns:1fr auto;gap:6px;color:#d6dded;font-size:11px}label input{grid-column:1/-1;width:100%;min-height:44px;accent-color:#f0a83b}label output{color:#f0a83b;font:750 11px ui-monospace,monospace}.run-buttons{display:grid;grid-template-columns:1fr 1fr;gap:8px}.run-buttons button{min-height:44px;border:1px solid #f0a83b;border-radius:5px;color:#10182e;background:#f0a83b;font-weight:800}.run-buttons .secondary{grid-column:1/-1;color:#dfe6f7;background:#111a31;border-color:#50648b}@media(max-width:760px){.control-room{min-width:0;overflow-x:auto}.rack-label,.room-grid{min-width:580px}}
</style>
