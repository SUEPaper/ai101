<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Action = 'N' | 'E' | 'S' | 'W'
type Cell = { row: number; col: number; key: string; label: string; value: number; kind?: 'wall' | 'goal' | 'hazard' }

const actions: Action[] = ['N', 'E', 'S', 'W']
const actionName: Record<Action, string> = { N: '北移', E: '东移', S: '南移', W: '西移' }
const delta: Record<Action, [number, number]> = { N: [-1, 0], E: [0, 1], S: [1, 0], W: [0, -1] }
const left: Record<Action, Action> = { N: 'W', E: 'N', S: 'E', W: 'S' }
const right: Record<Action, Action> = { N: 'E', E: 'S', S: 'W', W: 'N' }
const policyWeight: Record<Action, number> = { N: 0.70, E: 0.15, S: 0.10, W: 0.05 }

const valueMap: Record<string, number> = {
  '0-0': 10.22, '0-1': 11.74, '0-2': 13.24, '0-3': 0,
  '1-0': 8.94, '1-2': 8.54, '1-3': 0,
  '2-0': 7.76, '2-2': 7.09, '2-3': 4.26,
  '3-0': 6.55, '3-1': 5.54, '3-2': 5.90, '3-3': 4.87,
}
const walls = new Set(['1-1', '2-1'])
const terminals = new Set(['0-3', '1-3'])
const cells: Cell[] = Array.from({ length: 16 }, (_, index) => {
  const row = Math.floor(index / 4)
  const col = index % 4
  const key = `${row}-${col}`
  return {
    row, col, key, label: `R${row + 1}C${col + 1}`,
    value: valueMap[key] ?? 0,
    kind: walls.has(key) ? 'wall' : key === '0-3' ? 'goal' : key === '1-3' ? 'hazard' : undefined,
  }
})

const selectedKey = ref('2-2')
const gamma = ref(0.94)
const slip = ref(0.18)
const mode = ref<'policy' | 'optimal'>('optimal')

function move(key: string, action: Action) {
  if (terminals.has(key)) return key
  const [row, col] = key.split('-').map(Number)
  const [dr, dc] = delta[action]
  const candidate = `${row + dr}-${col + dc}`
  if (row + dr < 0 || row + dr > 3 || col + dc < 0 || col + dc > 3 || walls.has(candidate)) return key
  return candidate
}

function landingReward(key: string) {
  if (key === '0-3') return 14
  if (key === '1-3') return -22
  return -0.45
}

function branchSet(action: Action) {
  const actual = [action, left[action], right[action]]
  const probabilities = [1 - slip.value, slip.value / 2, slip.value / 2]
  return actual.map((direction, index) => {
    const next = move(selectedKey.value, direction)
    const reward = landingReward(next)
    const continuation = valueMap[next] ?? 0
    const target = reward + gamma.value * continuation
    return { direction, next, probability: probabilities[index], reward, continuation, target, contribution: probabilities[index] * target }
  })
}

const qValues = computed(() => actions.map(action => ({
  action,
  value: branchSet(action).reduce((sum, branch) => sum + branch.contribution, 0),
})))
const greedy = computed(() => qValues.value.reduce((best, item) => item.value > best.value ? item : best))
const backup = computed(() => mode.value === 'optimal'
  ? greedy.value.value
  : qValues.value.reduce((sum, item) => sum + policyWeight[item.action] * item.value, 0))
const currentValue = computed(() => valueMap[selectedKey.value] ?? 0)
const residual = computed(() => Math.abs(backup.value - currentValue.value))
const focusAction = computed(() => mode.value === 'optimal' ? greedy.value.action : 'N' as Action)
const branches = computed(() => branchSet(focusAction.value))
const maxMagnitude = computed(() => Math.max(1, ...qValues.value.map(item => Math.abs(item.value))))

function selectCell(cell: Cell) {
  if (!cell.kind) selectedKey.value = cell.key
}

function reset() {
  selectedKey.value = '2-2'
  gamma.value = 0.94
  slip.value = 0.18
  mode.value = 'optimal'
}
</script>

<template>
  <MlExperimentFrame
    index="30-B"
    title="Bellman backup 配电盘"
    question="一次 backup 不是把下一格价值直接抄过来：动作偏差、落点奖励、折扣和策略混合分别进入哪一层？"
    formula="TπV(s)=Σaπ(a|s)Σs′P(s′|s,a)[r+γV(s′)]; TV(s)=maxa QV(s,a)"
  >
    <template #visual>
      <div class="switchboard-scroll">
        <div class="switchboard">
          <div class="floor-grid" role="grid" aria-label="晶圆载具转运走廊状态价值">
            <button
              v-for="cell in cells"
              :key="cell.key"
              type="button"
              role="gridcell"
              :class="['floor-cell', cell.kind, { selected: selectedKey === cell.key }]"
              :disabled="Boolean(cell.kind)"
              :aria-label="cell.kind ? `${cell.label} ${cell.kind}` : `${cell.label}，当前价值 ${cell.value}`"
              @click="selectCell(cell)"
            >
              <span>{{ cell.kind === 'wall' ? '隔墙' : cell.kind === 'goal' ? '交付' : cell.kind === 'hazard' ? '隔离' : cell.label }}</span>
              <strong v-if="!cell.kind">{{ cell.value.toFixed(2) }}</strong>
            </button>
          </div>

          <div class="bus-panel">
            <div class="panel-heading">{{ mode === 'optimal' ? 'OPTIMAL BUS · MAX' : 'POLICY BUS · WEIGHTED SUM' }}</div>
            <div v-for="item in qValues" :key="item.action" class="q-row" :class="{ active: item.action === focusAction }">
              <span>{{ item.action }} · {{ actionName[item.action] }}</span>
              <div class="q-track"><i :style="{ width: `${Math.abs(item.value) / maxMagnitude * 100}%` }" /></div>
              <strong>{{ item.value.toFixed(3) }}</strong>
              <em v-if="mode === 'policy'">× {{ policyWeight[item.action].toFixed(2) }}</em>
            </div>
            <div class="branch-panel">
              <div v-for="branch in branches" :key="branch.direction" class="branch">
                <span>{{ branch.direction }} → {{ branch.next }}</span>
                <b>{{ branch.probability.toFixed(2) }}</b>
                <code>{{ branch.reward.toFixed(2) }} + {{ gamma.toFixed(2) }} × {{ branch.continuation.toFixed(2) }}</code>
                <strong>{{ branch.contribution.toFixed(3) }}</strong>
              </div>
            </div>
          </div>
        </div>
      </div>
    </template>

    <template #controls>
      <div class="mode" role="group" aria-label="Bellman 算子">
        <button type="button" :aria-pressed="mode === 'optimal'" @click="mode = 'optimal'">最优算子 max</button>
        <button type="button" :aria-pressed="mode === 'policy'" @click="mode = 'policy'">固定策略加权</button>
      </div>
      <label class="control">
        <span>折扣因子 <output>{{ gamma.toFixed(2) }}</output></span>
        <input v-model.number="gamma" type="range" min="0" max="0.99" step="0.01" aria-label="折扣因子">
      </label>
      <label class="control">
        <span>横向偏移概率 <output>{{ slip.toFixed(2) }}</output></span>
        <input v-model.number="slip" type="range" min="0" max="0.5" step="0.01" aria-label="横向偏移概率">
      </label>
      <button type="button" class="reset" @click="reset">复位配电盘</button>
    </template>

    <template #metrics>
      <div><span>当前状态</span><strong>{{ selectedKey }}</strong></div>
      <div><span>backup 目标</span><strong>{{ backup.toFixed(3) }}</strong></div>
      <div><span>绝对 residual</span><strong>{{ residual.toFixed(3) }}</strong></div>
      <div><span>贪心动作</span><strong>{{ greedy.action }} · {{ actionName[greedy.action] }}</strong></div>
    </template>

    <template #note>格内数字是固定的旧价值估计，所以本实验只执行“一次” backup。最优算子在动作层取最大值；固定策略算子先对每个动作的转移分支求期望，再按策略概率加权。调高偏移概率时，隔离区附近动作的 $Q$ 值会因小概率大损失明显下降。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.switchboard-scroll{overflow-x:auto}.switchboard{display:grid;grid-template-columns:328px 356px;gap:14px;min-width:698px;padding:14px;border:2px solid #f2b705;border-radius:4px;background:#242a2e}.floor-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:5px;padding:12px;background:repeating-linear-gradient(135deg,#30373b 0,#30373b 14px,#2a3034 14px,#2a3034 28px)}.floor-cell{position:relative;display:grid;place-content:center;gap:4px;min-width:66px;min-height:66px;padding:4px;border:1px solid #778086;border-radius:1px;color:#dde3e4;background:#343c40;font:700 8px/1.2 ui-monospace,monospace;cursor:pointer}.floor-cell strong{color:#f2b705;font-size:13px}.floor-cell.selected{outline:4px solid #3c8ca6;outline-offset:-4px;background:#27434b}.floor-cell.wall{border-color:#596066;color:#949c9f;background:repeating-linear-gradient(45deg,#1f2427 0,#1f2427 6px,#596066 6px,#596066 8px)}.floor-cell.goal{border-color:#6f9f58;color:#e7ffe0;background:#35502f}.floor-cell.hazard{border-color:#d64b3c;color:#ffe0da;background:repeating-linear-gradient(135deg,#4a2724 0,#4a2724 7px,#d64b3c 7px,#d64b3c 9px)}.floor-cell:disabled{opacity:1;cursor:not-allowed}.bus-panel{padding:13px;border:1px solid #778086;background:#171c1f}.panel-heading{margin-bottom:12px;color:#f2b705;font:850 9px/1 ui-monospace,monospace;letter-spacing:.09em}.q-row{display:grid;grid-template-columns:78px 1fr 54px 42px;gap:7px;align-items:center;min-height:32px;color:#bac3c6;font:700 8px/1 ui-monospace,monospace}.q-row.active{color:#fff}.q-row em{color:#9ca9ad;font-style:normal}.q-track{height:7px;background:#343d42}.q-track i{display:block;height:100%;background:#3c8ca6}.q-row.active .q-track i{background:#f2b705}.q-row strong{color:#e8e2d6;text-align:right}.branch-panel{display:grid;gap:5px;margin-top:12px;padding-top:12px;border-top:1px dashed #778086}.branch{display:grid;grid-template-columns:72px 28px 1fr 48px;gap:6px;align-items:center;padding:6px;color:#bbc3c5;background:#242a2e;font:650 8px/1.25 ui-monospace,monospace}.branch b{color:#f2b705}.branch code{color:#a7cdd7;background:transparent}.branch strong{color:#e8e2d6;text-align:right}.mode{display:grid;grid-template-columns:1fr 1fr;gap:8px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#c3cee8;font-size:11px}.control output{color:#f7d56a;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#f2b705}button{min-height:44px;padding:9px;border:1px solid #65737a;border-radius:3px;color:#e8e2d6;background:#242a2e;font:720 10px/1.35 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#f2b705;color:#fff1b5;background:#4a401d}.reset{border-color:#6f9f58;color:#cce8bf}button:focus-visible,input:focus-visible{outline:3px solid #f2b705;outline-offset:2px}@media(max-width:560px){.switchboard-scroll{margin-right:-18px;padding-right:18px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
