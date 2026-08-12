<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Action = 'N' | 'E' | 'S' | 'W'
type Algorithm = 'value' | 'policy'
type UpdateMode = 'sync' | 'inplace'
type Transition = { next: number; probability: number; reward: number }

const actions: Action[] = ['N', 'E', 'S', 'W']
const arrow: Record<Action, string> = { N: '↑', E: '→', S: '↓', W: '←' }
const delta: Record<Action, [number, number]> = { N: [-1, 0], E: [0, 1], S: [1, 0], W: [0, -1] }
const left: Record<Action, Action> = { N: 'W', E: 'N', S: 'E', W: 'S' }
const right: Record<Action, Action> = { N: 'E', E: 'S', S: 'W', W: 'N' }
const wallKeys = new Set(['1-1', '2-1'])
const goalKey = '0-3'
const hazardKey = '1-3'

const stateKeys = Array.from({ length: 16 }, (_, index) => `${Math.floor(index / 4)}-${index % 4}`).filter(key => !wallKeys.has(key))
const stateIndex = new Map(stateKeys.map((key, index) => [key, index]))
const terminalKeys = new Set([goalKey, hazardKey])
const displayCells = Array.from({ length: 16 }, (_, index) => {
  const key = `${Math.floor(index / 4)}-${index % 4}`
  return { key, index: stateIndex.get(key), kind: wallKeys.has(key) ? 'wall' : key === goalKey ? 'goal' : key === hazardKey ? 'hazard' : 'state' }
})

const algorithm = ref<Algorithm>('value')
const updateMode = ref<UpdateMode>('sync')
const gamma = ref(0.94)
const slip = ref(0.18)
const values = ref<number[]>(Array(stateKeys.length).fill(0))
const policy = ref<Action[]>(Array(stateKeys.length).fill('N'))
const iteration = ref(0)
const policyChanges = ref(0)
const policyStable = ref(false)
const residualHistory = ref<number[]>([])

function move(key: string, action: Action) {
  if (terminalKeys.has(key)) return key
  const [row, col] = key.split('-').map(Number)
  const [dr, dc] = delta[action]
  const nextRow = row + dr
  const nextCol = col + dc
  const candidate = `${nextRow}-${nextCol}`
  if (nextRow < 0 || nextRow > 3 || nextCol < 0 || nextCol > 3 || wallKeys.has(candidate)) return key
  return candidate
}

function reward(nextKey: string) {
  if (nextKey === goalKey) return 14
  if (nextKey === hazardKey) return -22
  return -0.45
}

function transitions(key: string, action: Action): Transition[] {
  if (terminalKeys.has(key)) return [{ next: stateIndex.get(key)!, probability: 1, reward: 0 }]
  const directions = [action, left[action], right[action]]
  const probabilities = [1 - slip.value, slip.value / 2, slip.value / 2]
  return directions.map((direction, index) => {
    const nextKey = move(key, direction)
    return { next: stateIndex.get(nextKey)!, probability: probabilities[index], reward: reward(nextKey) }
  })
}

function qValue(source: number[], state: number, action: Action) {
  const key = stateKeys[state]
  return transitions(key, action).reduce((sum, item) => sum + item.probability * (item.reward + gamma.value * source[item.next]), 0)
}

function greedyAction(source: number[], state: number) {
  return actions.reduce((best, action) => qValue(source, state, action) > qValue(source, state, best) ? action : best)
}

function optimalResidual(source: number[]) {
  let residual = 0
  stateKeys.forEach((key, state) => {
    if (terminalKeys.has(key)) return
    const target = Math.max(...actions.map(action => qValue(source, state, action)))
    residual = Math.max(residual, Math.abs(target - source[state]))
  })
  return residual
}

function evaluateCurrentPolicy() {
  let evaluated = [...values.value]
  for (let sweep = 0; sweep < 500; sweep += 1) {
    const next = [...evaluated]
    let change = 0
    stateKeys.forEach((key, state) => {
      if (terminalKeys.has(key)) {
        next[state] = 0
        return
      }
      next[state] = qValue(evaluated, state, policy.value[state])
      change = Math.max(change, Math.abs(next[state] - evaluated[state]))
    })
    evaluated = next
    if (change < 1e-11) break
  }
  return evaluated
}

function stepOnce() {
  if (algorithm.value === 'value') {
    const source = [...values.value]
    const next = [...source]
    stateKeys.forEach((key, state) => {
      if (terminalKeys.has(key)) {
        next[state] = 0
        return
      }
      const read = updateMode.value === 'sync' ? source : next
      next[state] = Math.max(...actions.map(action => qValue(read, state, action)))
    })
    values.value = next
    policy.value = stateKeys.map((key, state) => terminalKeys.has(key) ? 'N' : greedyAction(next, state))
    policyChanges.value = 0
  } else {
    const evaluated = evaluateCurrentPolicy()
    const improved = stateKeys.map((key, state) => terminalKeys.has(key) ? 'N' : greedyAction(evaluated, state))
    const changes = improved.reduce((count, action, state) => count + Number(action !== policy.value[state]), 0)
    values.value = evaluated
    policy.value = improved
    policyChanges.value = changes
    policyStable.value = changes === 0
  }
  iteration.value += 1
  residualHistory.value.push(optimalResidual(values.value))
  residualHistory.value = residualHistory.value.slice(-18)
}

function advanceBatch() {
  const steps = algorithm.value === 'value' ? 8 : 1
  for (let index = 0; index < steps; index += 1) stepOnce()
}

function runToStop() {
  for (let index = 0; index < 300; index += 1) {
    if (algorithm.value === 'value' && optimalResidual(values.value) < 1e-7) break
    if (algorithm.value === 'policy' && policyStable.value) break
    stepOnce()
  }
}

function resetState() {
  values.value = Array(stateKeys.length).fill(0)
  policy.value = Array(stateKeys.length).fill('N')
  iteration.value = 0
  policyChanges.value = 0
  policyStable.value = false
  residualHistory.value = []
}

function setAlgorithm(next: Algorithm) {
  algorithm.value = next
  resetState()
}

function fullReset() {
  algorithm.value = 'value'
  updateMode.value = 'sync'
  gamma.value = 0.94
  slip.value = 0.18
  resetState()
}

function residualWidth(value: number) {
  return Math.max(3, Math.min(100, 100 * Math.log10(1 + value) / Math.log10(13)))
}

const residual = computed(() => optimalResidual(values.value))
const startValue = computed(() => values.value[stateIndex.get('3-0')!])
const status = computed(() => algorithm.value === 'policy'
  ? policyStable.value ? '策略稳定' : '待改进'
  : residual.value < 1e-7 ? 'residual 达标' : '继续 backup')
</script>

<template>
  <MlExperimentFrame
    index="30-C"
    title="动态规划脉冲地坪"
    question="价值迭代把最优 backup 反复打进每个格位；策略迭代则先把一条路线评到底，再整体换路——它们如何抵达同一固定点？"
    formula="Vₖ₊₁=TVₖ; policy iteration: Vπ ← TπVπ, π ← greedy(Vπ)"
  >
    <template #visual>
      <div class="pulse-scroll">
        <div class="pulse-board">
          <div class="grid-wrap">
            <div class="floor-title">WAFER AMR CORRIDOR · {{ algorithm === 'value' ? 'VALUE PULSE' : 'POLICY PULSE' }}</div>
            <div class="floor-grid" aria-label="动态规划状态价值和当前策略">
              <div v-for="cell in displayCells" :key="cell.key" :class="['cell', cell.kind]">
                <template v-if="cell.kind === 'state' && cell.index !== undefined">
                  <span>{{ cell.key }}</span>
                  <strong>{{ values[cell.index].toFixed(2) }}</strong>
                  <b>{{ arrow[policy[cell.index]] }}</b>
                </template>
                <span v-else>{{ cell.kind === 'wall' ? '隔墙' : cell.kind === 'goal' ? '+14 交付' : '−22 隔离' }}</span>
              </div>
            </div>
          </div>
          <div class="trace-panel">
            <div class="trace-title">BELLMAN RESIDUAL TRACE</div>
            <div class="trace-bars" aria-label="最近 Bellman residual">
              <div v-if="residualHistory.length === 0" class="empty-trace">按“单步脉冲”开始</div>
              <div v-for="(item, index) in residualHistory" :key="`${iteration}-${index}`" class="trace-row">
                <span>k{{ iteration - residualHistory.length + index + 1 }}</span>
                <i><b :style="{ width: `${residualWidth(item)}%` }" /></i>
                <strong>{{ item < 0.001 ? item.toExponential(1) : item.toFixed(3) }}</strong>
              </div>
            </div>
            <div class="route-stamp">{{ status }}</div>
          </div>
        </div>
      </div>
    </template>

    <template #controls>
      <div class="mode" role="group" aria-label="动态规划算法">
        <button type="button" :aria-pressed="algorithm === 'value'" @click="setAlgorithm('value')">价值迭代</button>
        <button type="button" :aria-pressed="algorithm === 'policy'" @click="setAlgorithm('policy')">策略迭代</button>
      </div>
      <div v-if="algorithm === 'value'" class="mode" role="group" aria-label="价值更新方式">
        <button type="button" :aria-pressed="updateMode === 'sync'" @click="updateMode = 'sync'; resetState()">同步副本</button>
        <button type="button" :aria-pressed="updateMode === 'inplace'" @click="updateMode = 'inplace'; resetState()">原地扫描</button>
      </div>
      <label class="control">
        <span>折扣因子 <output>{{ gamma.toFixed(2) }}</output></span>
        <input v-model.number="gamma" type="range" min="0.50" max="0.99" step="0.01" aria-label="动态规划折扣因子" @change="resetState">
      </label>
      <label class="control">
        <span>横向偏移概率 <output>{{ slip.toFixed(2) }}</output></span>
        <input v-model.number="slip" type="range" min="0" max="0.50" step="0.01" aria-label="动态规划横向偏移概率" @change="resetState">
      </label>
      <div class="mode">
        <button type="button" @click="stepOnce">单步脉冲</button>
        <button type="button" @click="advanceBatch">{{ algorithm === 'value' ? '推进 8 步' : '评价并改进' }}</button>
      </div>
      <button type="button" class="run" @click="runToStop">运行到停止条件</button>
      <button type="button" class="reset" @click="fullReset">复位地坪</button>
    </template>

    <template #metrics>
      <div><span>外层迭代</span><strong>{{ iteration }}</strong></div>
      <div><span>最优 Bellman residual</span><strong>{{ residual < 0.001 ? residual.toExponential(2) : residual.toFixed(3) }}</strong></div>
      <div><span>本轮策略变化</span><strong>{{ policyChanges }} 格</strong></div>
      <div><span>起点 V(3-0)</span><strong>{{ startValue.toFixed(3) }}</strong></div>
    </template>

    <template #note>价值迭代每轮只做一次最优 backup；同步副本对应清晰的 $V_{k+1}=TV_k$，原地扫描会立刻利用本轮新值，通常传播更快但日志语义不同。策略迭代的一个外层脉冲包含“固定策略评价到收敛 + 贪心改进”；在默认模型中，它只需少量外层轮次便与价值迭代得到同一路线。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.pulse-scroll{overflow-x:auto}.pulse-board{display:grid;grid-template-columns:390px 294px;gap:14px;min-width:698px;padding:14px;border:2px solid #f2b705;border-radius:4px;background:#242a2e}.grid-wrap{padding:12px;background:#30373b}.floor-title,.trace-title,.empty-trace,.trace-row,.route-stamp,.cell{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.floor-title,.trace-title{margin-bottom:10px;color:#f2b705;font-size:9px;font-weight:900;letter-spacing:.09em}.floor-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}.cell{position:relative;display:grid;place-items:center;min-height:72px;border:1px solid #7a8488;color:#dce2e3;background:#394247}.cell span{font-size:8px}.cell strong{color:#f2b705;font-size:13px}.cell b{position:absolute;right:8px;bottom:5px;color:#79bfd1;font-size:22px}.cell.wall{color:#969ea1;background:repeating-linear-gradient(45deg,#1f2427 0,#1f2427 6px,#596066 6px,#596066 8px)}.cell.goal{border-color:#6f9f58;color:#e7ffe0;background:#35502f}.cell.hazard{border-color:#d64b3c;color:#ffe0da;background:repeating-linear-gradient(135deg,#4a2724 0,#4a2724 7px,#d64b3c 7px,#d64b3c 9px)}.trace-panel{display:flex;flex-direction:column;min-height:332px;padding:12px;border:1px solid #667278;background:#171c1f}.trace-bars{display:grid;align-content:end;gap:4px;min-height:246px}.empty-trace{margin:auto;color:#89969a;font-size:9px}.trace-row{display:grid;grid-template-columns:28px 1fr 54px;gap:6px;align-items:center;color:#9daaad;font-size:7px}.trace-row i{height:7px;background:#30383c}.trace-row i b{display:block;height:100%;background:#3c8ca6}.trace-row strong{color:#e8e2d6;text-align:right}.route-stamp{margin-top:auto;padding:10px;border:2px solid #f2b705;color:#f2b705;font-size:11px;font-weight:900;text-align:center;text-transform:uppercase;transform:rotate(-1deg)}.mode{display:grid;grid-template-columns:1fr 1fr;gap:8px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#c3cee8;font-size:11px}.control output{color:#f7d56a;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#f2b705}button{min-height:44px;padding:9px;border:1px solid #65737a;border-radius:3px;color:#e8e2d6;background:#242a2e;font:720 10px/1.35 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#f2b705;color:#fff1b5;background:#4a401d}.run{border-color:#3c8ca6;color:#bfeefe}.reset{border-color:#6f9f58;color:#cce8bf}button:focus-visible,input:focus-visible{outline:3px solid #f2b705;outline-offset:2px}@media(max-width:560px){.pulse-scroll{margin-right:-18px;padding-right:18px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
