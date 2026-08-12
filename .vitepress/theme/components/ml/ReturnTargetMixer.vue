<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Boundary = 'continuing' | 'terminated' | 'truncated'

const rewards = [-0.45, -0.45, -0.45, -0.45, 14]
const n = ref(2)
const gamma = ref(0.94)
const bootstrapValue = ref(7.2)
const boundary = ref<Boundary>('continuing')

const visibleRewards = computed(() => rewards.slice(0, n.value))
const reachesNaturalEnd = computed(() => n.value >= rewards.length || boundary.value === 'terminated')
const effectiveBootstrap = computed(() => reachesNaturalEnd.value ? 0 : bootstrapValue.value)
const rewardPart = computed(() => visibleRewards.value.reduce(
  (sum, reward, index) => sum + gamma.value ** index * reward,
  0,
))
const bootstrapPart = computed(() => gamma.value ** n.value * effectiveBootstrap.value)
const target = computed(() => rewardPart.value + bootstrapPart.value)
const targetKind = computed(() => {
  if (n.value === rewards.length || reachesNaturalEnd.value) return 'Monte Carlo 尾端'
  if (n.value === 1) return 'TD(0)'
  return `${n.value}-step`
})
const boundaryLabel = computed(() => ({
  continuing: '仍在运行',
  terminated: '自然终止',
  truncated: '时间截断',
})[boundary.value])

function reset() {
  n.value = 2
  gamma.value = 0.94
  bootstrapValue.value = 7.2
  boundary.value = 'continuing'
}
</script>

<template>
  <MlExperimentFrame
    index="31-A"
    title="回报撕纸线"
    question="一条奖励带应该在第几步撕下并接上当前价值？移动撕纸线，观察 MC、TD 与 n-step 如何共享同一条回报恒等式。"
    formula="Gₜ⁽ⁿ⁾ = Σₖ₌₀ⁿ⁻¹ γᵏRₜ₊ₖ₊₁ + γⁿV(Sₜ₊ₙ)"
  >
    <template #visual>
      <div class="recorder-shell">
        <div class="paper" aria-label="n-step 回报事件记录纸">
          <div class="holes holes-left" aria-hidden="true" />
          <div class="holes holes-right" aria-hidden="true" />
          <div class="paper-head">
            <span>AMR EVENT RECORDER · E31-047</span>
            <strong>{{ targetKind }}</strong>
          </div>
          <div class="tape">
            <div
              v-for="(reward, index) in rewards"
              :key="index"
              :class="['event', { active: index < n, future: index >= n }]"
            >
              <small>R{{ index + 1 }}</small>
              <b>{{ reward > 0 ? '+' : '' }}{{ reward.toFixed(2) }}</b>
              <span>× γ<sup>{{ index }}</sup></span>
              <i v-if="index + 1 === n" aria-hidden="true">TEAR</i>
            </div>
            <div :class="['bootstrap', { muted: reachesNaturalEnd }]">
              <small>BOOTSTRAP</small>
              <b>{{ effectiveBootstrap.toFixed(2) }}</b>
              <span>× γ<sup>{{ n }}</sup></span>
            </div>
          </div>
          <div class="trace">
            <span>REWARD PART {{ rewardPart.toFixed(3) }}</span>
            <i />
            <span>BOOTSTRAP {{ bootstrapPart.toFixed(3) }}</span>
          </div>
          <div class="total-stamp">
            <small>TARGET / {{ boundaryLabel }}</small>
            <strong>{{ target.toFixed(3) }}</strong>
          </div>
        </div>
      </div>
    </template>

    <template #controls>
      <label class="control">
        <span>撕纸步数 n <output>{{ n }}</output></span>
        <input v-model.number="n" type="range" min="1" max="5" step="1" aria-label="n-step 步数">
      </label>
      <label class="control">
        <span>折扣 γ <output>{{ gamma.toFixed(2) }}</output></span>
        <input v-model.number="gamma" type="range" min="0.50" max="1" step="0.01" aria-label="折扣因子">
      </label>
      <label class="control">
        <span>V(Sₜ₊ₙ) <output>{{ bootstrapValue.toFixed(1) }}</output></span>
        <input v-model.number="bootstrapValue" type="range" min="-5" max="15" step="0.1" aria-label="bootstrap 价值">
      </label>
      <div class="boundary" role="group" aria-label="轨迹边界">
        <button type="button" :aria-pressed="boundary === 'continuing'" @click="boundary = 'continuing'">仍在运行</button>
        <button type="button" :aria-pressed="boundary === 'truncated'" @click="boundary = 'truncated'">时间截断</button>
        <button type="button" :aria-pressed="boundary === 'terminated'" @click="boundary = 'terminated'">自然终止</button>
      </div>
      <button type="button" class="reset" @click="reset">复位记录纸</button>
    </template>

    <template #metrics>
      <div><span>奖励项</span><strong>{{ rewardPart.toFixed(3) }}</strong></div>
      <div><span>bootstrap 项</span><strong>{{ bootstrapPart.toFixed(3) }}</strong></div>
      <div><span>target 类型</span><strong>{{ targetKind }}</strong></div>
      <div><span>总 target</span><strong>{{ target.toFixed(3) }}</strong></div>
    </template>

    <template #note>
      时间截断只撕断采集记录，不应自动把未来价值清零；自然终止才删除 bootstrap。把 n 推到奖励带末端，则得到完整 Monte Carlo 回报。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.recorder-shell{padding:14px;border:1px solid #567481;background:#253c46;box-shadow:inset 0 0 0 5px #172c34}.paper{position:relative;min-height:330px;padding:28px 40px 26px;overflow:hidden;color:#20272b;background:repeating-linear-gradient(0deg,#e8e5d9 0,#e8e5d9 27px,#d9d7cc 28px);box-shadow:5px 7px 0 #14252c}.paper:before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent 49.7%,rgba(50,92,112,.12) 50%,transparent 50.3%);pointer-events:none}.holes{position:absolute;top:8px;bottom:8px;width:12px;background:radial-gradient(circle,#253c46 0 3px,transparent 3.4px) 0 0/12px 20px}.holes-left{left:8px}.holes-right{right:8px}.paper-head{position:relative;display:flex;justify-content:space-between;gap:14px;padding-bottom:10px;border-bottom:2px solid #325c70;font:700 10px/1.3 ui-monospace,monospace;letter-spacing:.09em}.paper-head strong{color:#66517f}.tape{position:relative;display:grid;grid-template-columns:repeat(5,minmax(62px,1fr)) minmax(80px,1.2fr);gap:6px;margin-top:32px}.event,.bootstrap{position:relative;display:grid;gap:8px;min-height:98px;padding:12px 8px;border:1px solid #86959a;background:rgba(255,255,255,.28);font-family:ui-monospace,monospace}.event.active{border-color:#325c70;background:rgba(79,156,168,.16)}.event.future{opacity:.38}.event small,.bootstrap small{font-size:8px;letter-spacing:.08em}.event b,.bootstrap b{color:#325c70;font-size:16px}.event span,.bootstrap span{font-size:9px}.event i{position:absolute;right:-7px;bottom:-24px;z-index:2;padding:2px 4px;color:#fff;background:#d85d4a;font:800 7px/1 ui-monospace,monospace;letter-spacing:.08em;transform:rotate(-90deg)}.bootstrap{border-style:dashed;border-color:#66517f}.bootstrap b{color:#66517f}.bootstrap.muted{opacity:.3;text-decoration:line-through}.trace{position:relative;display:flex;align-items:center;gap:8px;margin-top:42px;color:#325c70;font:700 8px/1 ui-monospace,monospace;letter-spacing:.05em}.trace i{flex:1;height:2px;background:linear-gradient(90deg,#325c70,#66517f)}.total-stamp{display:flex;align-items:end;justify-content:space-between;margin-top:26px;padding:8px 10px;border:2px solid #d85d4a;color:#b34638;transform:rotate(-1deg)}.total-stamp small{font:800 9px/1 ui-monospace,monospace;letter-spacing:.08em}.total-stamp strong{font:900 24px/1 ui-monospace,monospace}.control{display:grid;gap:7px;color:#c8d6df;font-size:11px}.control span{display:flex;justify-content:space-between}.control output{color:#8dd5dd;font-family:ui-monospace,monospace}.control input{width:100%;accent-color:#4f9ca8}.boundary{display:grid;grid-template-columns:1fr;gap:6px}.boundary button,.reset{min-height:38px;border:1px solid #4f6872;border-radius:3px;color:#cbd9df;background:#20343d;font:700 10px/1 ui-monospace,monospace}.boundary button[aria-pressed="true"]{border-color:#d85d4a;color:#fff;background:#9d463a}.reset{border-style:dashed}.boundary button:focus-visible,.reset:focus-visible,input:focus-visible{outline:3px solid #f0c45a;outline-offset:2px}@media(max-width:820px){.tape{grid-template-columns:repeat(3,1fr)}.event i{display:none}}@media(max-width:520px){.paper{padding:24px 28px}.paper-head{display:grid}.tape{grid-template-columns:repeat(2,1fr)}.trace{align-items:flex-start;flex-direction:column}.trace i{width:100%;flex:none}.total-stamp strong{font-size:19px}}
</style>
