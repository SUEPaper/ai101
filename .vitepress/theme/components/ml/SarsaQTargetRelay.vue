<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const actions = ['N', 'E', 'S', 'W']
const actionNames = ['北移', '东移', '南移', '西移']
const qValues = ref([8, 5, 1, -2])
const nextAction = ref(1)
const epsilon = ref(0.40)
const reward = ref(-0.45)
const gamma = ref(0.94)
const terminated = ref(false)

const greedy = computed(() => qValues.value.reduce(
  (best, value, index, source) => value > source[best] ? index : best,
  0,
))
const probabilities = computed(() => {
  const result = Array(4).fill(epsilon.value / 4)
  const maximum = Math.max(...qValues.value)
  const winners = qValues.value.map((value, index) => ({ value, index }))
    .filter(item => Math.abs(item.value - maximum) < 1e-9)
  winners.forEach(item => { result[item.index] += (1 - epsilon.value) / winners.length })
  return result
})
const sarsaFuture = computed(() => qValues.value[nextAction.value])
const expectedFuture = computed(() => qValues.value.reduce(
  (sum, value, index) => sum + value * probabilities.value[index],
  0,
))
const qFuture = computed(() => Math.max(...qValues.value))
const target = (future: number) => reward.value + (terminated.value ? 0 : gamma.value * future)
const sarsaTarget = computed(() => target(sarsaFuture.value))
const expectedTarget = computed(() => target(expectedFuture.value))
const qTarget = computed(() => target(qFuture.value))
const targetGap = computed(() => qTarget.value - sarsaTarget.value)

function reset() {
  qValues.value = [8, 5, 1, -2]
  nextAction.value = 1
  epsilon.value = 0.40
  reward.value = -0.45
  gamma.value = 0.94
  terminated.value = false
}
</script>

<template>
  <MlExperimentFrame
    index="31-B"
    title="行为—目标继电器柜"
    question="同一条 transition 送进三路 target：实际动作、行为期望和贪心最大值分别接到哪里？"
    formula="SARSA: Q(s′,A′) · Expected: EμQ(s′,·) · Q-learning: max Q(s′,·)"
  >
    <template #visual>
      <div class="relay-cabinet">
        <div class="cabinet-head">
          <span>TRANSITION BUS / s′</span>
          <b :class="{ alarm: terminated }">{{ terminated ? 'TERMINATED' : 'BOOTSTRAP LIVE' }}</b>
        </div>
        <div class="q-bank">
          <div
            v-for="(value, index) in qValues"
            :key="actions[index]"
            :class="['q-relay', { selected: index === nextAction, greedy: index === greedy }]"
          >
            <span>{{ actions[index] }} · {{ actionNames[index] }}</span>
            <strong>{{ value.toFixed(1) }}</strong>
            <small>μ={{ probabilities[index].toFixed(2) }}</small>
            <i v-if="index === nextAction">ACTUAL</i>
            <em v-if="index === greedy">MAX</em>
          </div>
        </div>
        <svg class="wiring" viewBox="0 0 760 120" role="img" aria-label="Q 值到三种 target 的接线">
          <path class="wire sarsa-wire" d="M80 8 C80 50 125 48 125 108" />
          <path class="wire expected-wire" d="M380 8 L380 108" />
          <path class="wire q-wire" d="M680 8 C680 50 635 48 635 108" />
          <circle cx="125" cy="108" r="5" class="sarsa-dot" />
          <circle cx="380" cy="108" r="5" class="expected-dot" />
          <circle cx="635" cy="108" r="5" class="q-dot" />
        </svg>
        <div class="target-bank">
          <div class="target sarsa">
            <span>ACTUAL ACTION</span>
            <b>SARSA</b>
            <strong>{{ sarsaTarget.toFixed(3) }}</strong>
          </div>
          <div class="target expected">
            <span>BEHAVIOR EXPECTATION</span>
            <b>EXPECTED SARSA</b>
            <strong>{{ expectedTarget.toFixed(3) }}</strong>
          </div>
          <div class="target qlearning">
            <span>GREEDY TARGET</span>
            <b>Q-LEARNING</b>
            <strong>{{ qTarget.toFixed(3) }}</strong>
          </div>
        </div>
      </div>
    </template>

    <template #controls>
      <div class="action-select" role="group" aria-label="实际下一动作">
        <span>实际 Aₜ₊₁</span>
        <button
          v-for="(action, index) in actions"
          :key="action"
          type="button"
          :aria-pressed="nextAction === index"
          @click="nextAction = index"
        >{{ action }}</button>
      </div>
      <label v-for="(_, index) in qValues" :key="actions[index]" class="control compact">
        <span>Q({{ actions[index] }}) <output>{{ qValues[index].toFixed(1) }}</output></span>
        <input v-model.number="qValues[index]" type="range" min="-5" max="12" step="0.5" :aria-label="`动作 ${actions[index]} 的 Q 值`">
      </label>
      <label class="control">
        <span>探索率 ε <output>{{ epsilon.toFixed(2) }}</output></span>
        <input v-model.number="epsilon" type="range" min="0" max="1" step="0.05" aria-label="探索率">
      </label>
      <label class="switch">
        <input v-model="terminated" type="checkbox">
        <span>下一状态自然终止</span>
      </label>
      <button type="button" class="reset" @click="reset">复位继电器</button>
    </template>

    <template #metrics>
      <div><span>实际动作 future</span><strong>{{ sarsaFuture.toFixed(3) }}</strong></div>
      <div><span>行为期望 future</span><strong>{{ expectedFuture.toFixed(3) }}</strong></div>
      <div><span>贪心 future</span><strong>{{ qFuture.toFixed(3) }}</strong></div>
      <div><span>Q − SARSA target</span><strong>{{ targetGap.toFixed(3) }}</strong></div>
    </template>

    <template #note>
      将“自然终止”合闸后，三路 bootstrap 同时断开，target 都只剩即时奖励。时间截断不应使用这个开关。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.relay-cabinet{padding:16px;border:2px solid #6f8891;color:#20272b;background:#d9d8cf;box-shadow:inset 0 0 0 4px #b9bab3,5px 7px 0 #102832;font-family:ui-monospace,monospace}.cabinet-head{display:flex;justify-content:space-between;gap:12px;padding:8px 10px;border-bottom:3px double #325c70;font-size:9px;letter-spacing:.09em}.cabinet-head b{padding:4px 7px;color:#2c6872;background:#b8d8d8}.cabinet-head b.alarm{color:#fff;background:#d85d4a}.q-bank,.target-bank{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:18px}.q-relay{position:relative;display:grid;gap:7px;min-height:100px;padding:10px;border:1px solid #7a898e;background:#ebe8db}.q-relay.selected{box-shadow:inset 0 -5px #66517f}.q-relay.greedy{border-color:#d85d4a}.q-relay span{font-size:8px}.q-relay strong{color:#325c70;font-size:20px}.q-relay small{font-size:8px}.q-relay i,.q-relay em{position:absolute;right:5px;padding:2px 4px;color:#fff;font:800 7px/1 ui-monospace,monospace;font-style:normal}.q-relay i{top:5px;background:#66517f}.q-relay em{bottom:5px;background:#d85d4a}.wiring{display:block;width:100%;height:82px;overflow:visible}.wire{fill:none;stroke-width:4;stroke-dasharray:7 5}.sarsa-wire{stroke:#66517f}.expected-wire{stroke:#4f9ca8}.q-wire{stroke:#d85d4a}.sarsa-dot{fill:#66517f}.expected-dot{fill:#4f9ca8}.q-dot{fill:#d85d4a}.target-bank{grid-template-columns:repeat(3,1fr);margin-top:0}.target{display:grid;gap:5px;min-height:92px;padding:10px;border:2px solid;background:#ebe8db}.target span{font-size:7px;letter-spacing:.08em}.target b{font-size:9px}.target strong{font-size:19px}.target.sarsa{border-color:#66517f}.target.sarsa strong{color:#66517f}.target.expected{border-color:#4f9ca8}.target.expected strong{color:#2e7d88}.target.qlearning{border-color:#d85d4a}.target.qlearning strong{color:#b64839}.action-select{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}.action-select span{grid-column:1/-1;color:#c8d6df;font-size:10px}.action-select button,.reset{min-height:34px;border:1px solid #566f79;border-radius:2px;color:#d5e0e4;background:#223841;font:800 10px/1 ui-monospace,monospace}.action-select button[aria-pressed="true"]{border-color:#bda8d2;color:#fff;background:#66517f}.control{display:grid;gap:5px;color:#c8d6df;font-size:10px}.control span{display:flex;justify-content:space-between}.control output{color:#8dd5dd}.control input{width:100%;accent-color:#4f9ca8}.compact{gap:2px}.switch{display:flex;align-items:center;gap:8px;color:#c8d6df;font-size:10px}.switch input{width:17px;height:17px;accent-color:#d85d4a}.reset{border-style:dashed}.action-select button:focus-visible,.reset:focus-visible,input:focus-visible{outline:3px solid #f0c45a;outline-offset:2px}@media(max-width:620px){.q-bank{grid-template-columns:1fr 1fr}.wiring{display:none}.target-bank{grid-template-columns:1fr;margin-top:14px}.q-relay{min-height:86px}}
</style>
