<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const engineers = ['E-17', 'E-24', 'E-31', 'E-46', 'E-58', 'E-63']
const cards = ['C07', 'C12', 'C19', 'C26', 'C33', 'C41', 'C52', 'C60']
const stateLabels = ['未曝光', '已曝光·跳过', '已收藏', '已完成']
const initialStates = [
  [3, 0, 2, 0, 1, 0, 0, 0],
  [0, 2, 0, 1, 0, 3, 0, 0],
  [2, 0, 0, 3, 0, 1, 0, 0],
  [0, 1, 3, 0, 2, 0, 0, 0],
  [1, 0, 0, 2, 0, 3, 0, 0],
  [0, 3, 1, 0, 2, 0, 0, 0],
]
const utility = [
  [0.78, 0.31, 0.84, 0.42, 0.24, 0.67, 0.91, 0.58],
  [0.36, 0.88, 0.45, 0.29, 0.74, 0.81, 0.55, 0.92],
  [0.82, 0.37, 0.63, 0.86, 0.41, 0.28, 0.76, 0.53],
  [0.27, 0.45, 0.89, 0.34, 0.85, 0.62, 0.73, 0.49],
  [0.48, 0.71, 0.33, 0.84, 0.39, 0.87, 0.56, 0.79],
  [0.52, 0.86, 0.28, 0.57, 0.82, 0.46, 0.69, 0.75],
]

const states = ref(initialStates.map(row => [...row]))
const selectedUser = ref(0)
const zeroFill = ref(false)
const showUtility = ref(false)

const row = computed(() => states.value[selectedUser.value])
const exposed = computed(() => row.value.filter(value => value > 0).length)
const positive = computed(() => row.value.filter(value => value >= 2).length)
const unexposed = computed(() => row.value.filter(value => value === 0).length)
const candidate = computed(() => {
  const candidates = cards.map((name, index) => ({ name, index, score: utility[selectedUser.value][index] }))
    .filter(item => row.value[item.index] === 0)
    .sort((a, b) => b.score - a.score)
  return candidates[0]
})

function cycleState(user: number, item: number) {
  states.value[user][item] = (states.value[user][item] + 1) % 4
  states.value = states.value.map(rowValue => [...rowValue])
}

function reset() {
  states.value = initialStates.map(rowValue => [...rowValue])
  selectedUser.value = 0
  zeroFill.value = false
  showUtility.value = false
}

function cellClass(value: number) {
  return ['unexposed', 'skipped', 'saved', 'completed'][value]
}
</script>

<template>
  <MlExperimentFrame
    index="29-A"
    title="曝光矩阵灯检台"
    question="同一个空白格，究竟是工程师拒绝了知识卡，还是系统从未让他看见？点击格子改写事件账本。"
    formula="Ω = {(u,i): exposed}; unexposed ≠ negative ≠ rating 0"
  >
    <template #visual>
      <div class="matrix-scroll">
        <svg viewBox="0 0 650 420" role="img" aria-label="工程师与检修知识卡的曝光、跳过、收藏和完成矩阵">
          <defs>
            <pattern id="unseen-hatch-29" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="7" height="7" fill="#26343c" />
              <line x1="0" y1="0" x2="0" y2="7" stroke="#53616a" stroke-width="2" />
            </pattern>
          </defs>
          <rect class="cabinet" x="14" y="14" width="622" height="392" rx="3" />
          <rect class="paper" x="30" y="30" width="590" height="344" rx="2" />
          <text class="kicker" x="50" y="57">WORK-ORDER EXPOSURE LEDGER · EVENT TIME</text>
          <text class="revision" x="596" y="57" text-anchor="end">REV 29-A</text>
          <line class="perforation" x1="34" y1="70" x2="616" y2="70" />

          <g v-for="(card, item) in cards" :key="card">
            <text class="column-label" :x="178 + item * 51" y="91" text-anchor="middle">{{ card }}</text>
          </g>
          <g v-for="(engineer, user) in engineers" :key="engineer">
            <rect v-if="selectedUser === user" class="row-highlight" x="42" :y="104 + user * 42" width="558" height="38" />
            <text class="row-label" x="58" :y="128 + user * 42">{{ engineer }}</text>
            <g
              v-for="(value, item) in states[user]"
              :key="`${user}-${item}`"
              class="cell-button"
              role="button"
              tabindex="0"
              :aria-label="`${engineer} 与 ${cards[item]}：${stateLabels[value]}，点击切换状态`"
              @click="cycleState(user, item)"
              @keydown.enter.prevent="cycleState(user, item)"
              @keydown.space.prevent="cycleState(user, item)"
            >
              <rect :class="cellClass(value)" :x="158 + item * 51" :y="108 + user * 42" width="40" height="30" rx="2" />
              <text v-if="value > 0" class="cell-mark" :x="178 + item * 51" :y="128 + user * 42" text-anchor="middle">{{ ['?', '×', '★', '✓'][value] }}</text>
              <text v-if="showUtility && selectedUser === user" class="utility" :x="178 + item * 51" :y="151 + user * 42" text-anchor="middle">{{ utility[user][item].toFixed(2) }}</text>
            </g>
          </g>

          <line class="tear" x1="34" y1="362" x2="616" y2="362" />
          <g class="legend" transform="translate(50 387)">
            <rect class="unexposed" x="0" y="-10" width="18" height="14" /><text x="24" y="1">未曝光</text>
            <rect class="skipped" x="112" y="-10" width="18" height="14" /><text x="136" y="1">曝光后跳过</text>
            <rect class="saved" x="256" y="-10" width="18" height="14" /><text x="280" y="1">收藏</text>
            <rect class="completed" x="366" y="-10" width="18" height="14" /><text x="390" y="1">完成</text>
          </g>
        </svg>
      </div>
    </template>

    <template #controls>
      <label class="control">
        <span>审查工程师</span>
        <select v-model.number="selectedUser" aria-label="选择要审查的工程师">
          <option v-for="(engineer, index) in engineers" :key="engineer" :value="index">{{ engineer }}</option>
        </select>
      </label>
      <div class="mode" aria-label="空白格解释方式">
        <button type="button" :aria-pressed="!zeroFill" @click="zeroFill = false">保留未曝光</button>
        <button type="button" class="danger" :aria-pressed="zeroFill" @click="zeroFill = true">错误示范：填 0</button>
      </div>
      <button type="button" :aria-pressed="showUtility" @click="showUtility = !showUtility">{{ showUtility ? '隐藏潜在效用' : '揭示教学潜在效用' }}</button>
      <button type="button" class="reset" @click="reset">恢复原始事件账本</button>
    </template>

    <template #metrics>
      <div><span>已曝光 / 全库</span><strong>{{ exposed }} / {{ cards.length }}</strong></div>
      <div><span>正反馈</span><strong>{{ positive }}</strong></div>
      <div><span>{{ zeroFill ? '被强行判成负例' : '保持未知' }}</span><strong>{{ unexposed }}</strong></div>
      <div><span>未知区最高效用</span><strong>{{ candidate?.name ?? '—' }}</strong></div>
    </template>

    <template #note>事件账本必须把“曝光”与“行为”分开记录。填 0 会把 {{ unexposed }} 个未曝光格伪造成明确拒绝；揭示的潜在效用只存在于合成教学世界，真实系统永远看不到它，只能靠探索和未来反馈识别。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.matrix-scroll{overflow-x:auto}.matrix-scroll svg{display:block;width:100%;height:auto}.cabinet{fill:#10222c;stroke:#43b6ad;stroke-width:2}.paper{fill:#efe8d4;stroke:#8b8064}.kicker,.revision,.column-label,.row-label,.cell-mark,.utility,.legend text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.kicker{fill:#17364a;font-size:9px;font-weight:900;letter-spacing:.09em}.revision{fill:#69726d;font-size:8px}.perforation,.tear{stroke:#8b8064;stroke-width:1;stroke-dasharray:3 5}.column-label,.row-label{fill:#344d58;font-size:9px;font-weight:800}.row-highlight{fill:#43b6ad;opacity:.14}.cell-button{cursor:pointer}.cell-button:focus-visible{outline:3px solid #e7762f;outline-offset:2px}.cell-button rect{stroke:#6c7776;stroke-width:1}.unexposed{fill:url(#unseen-hatch-29)}.skipped{fill:#56616a}.saved{fill:#e7762f}.completed{fill:#43b6ad}.cell-mark{fill:#fff8e6;font-size:13px;font-weight:900}.utility{fill:#c44d43;font-size:7px;font-weight:800}.legend text{fill:#3f5258;font-size:8px}.mode{display:grid;grid-template-columns:1fr 1fr;gap:6px}.control{display:grid;gap:8px}.control span{color:#c3cee8;font-size:11px}select,button{min-height:44px;width:100%;padding:9px;border:1px solid #536a74;border-radius:4px;color:#dbe6e8;background:#14252e;font:720 10px/1.35 ui-monospace,monospace}button{cursor:pointer}button[aria-pressed="true"]{border-color:#e7762f;color:#ffe4d2;background:#4b2c20}.danger{border-color:#c44d43;color:#ffd1ca}.reset{border-color:#43b6ad;color:#bbfff8}button:focus-visible,select:focus-visible{outline:3px solid #e7762f;outline-offset:2px}@media(max-width:560px){.matrix-scroll{margin-right:-18px;padding-right:18px}.matrix-scroll svg{width:720px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
