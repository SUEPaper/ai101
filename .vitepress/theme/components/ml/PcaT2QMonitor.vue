<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Scenario = 'baseline' | 'within' | 'outside'

const scenario = ref<Scenario>('baseline')
const strictness = ref(1)
const eigenvalues = [6.65, 3.36]
const scenarios: Record<Scenario, { label: string; score: [number, number]; residual: number; route: string }> = {
  baseline: { label: '基线样本', score: [1.1, .45], residual: .38, route: '保持监控' },
  within: { label: '子空间内远行', score: [5.8, 1.05], residual: .45, route: '工况复核' },
  outside: { label: '传感器关系冲突', score: [1.0, .6], residual: 4.2, route: '量测链复核' },
}

const current = computed(() => scenarios[scenario.value])
const t2 = computed(() => current.value.score.reduce((sum, value, index) => sum + value ** 2 / eigenvalues[index], 0))
const q = computed(() => current.value.residual ** 2)
const t2Limit = computed(() => 3.2 * strictness.value)
const qLimit = computed(() => 2.1 * strictness.value)
const t2Alarm = computed(() => t2.value > t2Limit.value)
const qAlarm = computed(() => q.value > qLimit.value)
const state = computed(() => {
  if (t2Alarm.value && qAlarm.value) return 'T² + Q 双报警'
  if (t2Alarm.value) return 'T² 模型内远行'
  if (qAlarm.value) return 'Q 模型外偏离'
  return '基线窗口内'
})
const scorePoint = computed(() => ({
  x: 186 + current.value.score[0] * 26,
  y: 158 - current.value.score[1] * 30,
}))
const meterWidth = (value: number, limit: number) => Math.min(220, 220 * value / (limit * 1.45))
</script>

<template>
  <MlExperimentFrame
    index="25-C"
    title="用 T²/Q 双表拆开两种越界"
    question="切换同一 PCA 模型下的三种样本。一个统计量看保留子空间内走了多远，另一个看离开子空间多远；单表监控必然有盲区。"
    formula="T² = ∑ⱼ zⱼ²/λⱼ；Q = ‖x − W Wᵀx‖²"
  >
    <template #visual>
      <div class="monitor-scroll">
        <svg viewBox="0 0 640 364" role="img" :aria-label="`${current.label}，T平方${t2.toFixed(2)}，Q${q.toFixed(2)}，状态${state}`">
          <rect width="640" height="364" rx="14" class="deck" />
          <text x="22" y="28" class="eyebrow">ORTHOGONAL WATCHSTAND · DUAL-STATISTIC TRIP</text>
          <g transform="translate(28 52)">
            <rect width="338" height="224" rx="8" class="module" />
            <text x="16" y="26" class="module-title">SCORE PLANE · MODEL SPACE</text>
            <ellipse cx="186" cy="158" rx="106" ry="70" class="limit-ellipse" />
            <line x1="54" x2="316" y1="158" y2="158" class="axis" />
            <line x1="186" x2="186" y1="54" y2="202" class="axis" />
            <line x1="186" y1="158" :x2="scorePoint.x" :y2="scorePoint.y" class="score-vector" />
            <circle :cx="scorePoint.x" :cy="scorePoint.y" r="8" :class="t2Alarm ? 'probe probe-alarm' : 'probe probe-ok'" />
            <text x="306" y="174" class="axis-label">PC1</text>
            <text x="195" y="64" class="axis-label">PC2</text>
            <text x="16" y="208" class="status-text">{{ current.label }} · {{ t2Alarm ? 'OUTSIDE T² LIMIT' : 'INSIDE T² LIMIT' }}</text>
          </g>
          <g transform="translate(386 52)">
            <rect width="226" height="224" rx="8" class="module" />
            <text x="16" y="26" class="module-title">RESIDUAL CHANNEL</text>
            <line x1="113" x2="113" y1="52" y2="178" class="residual-axis" />
            <rect x="89" y="74" width="48" height="104" rx="5" class="residual-well" />
            <rect x="89" :y="178 - Math.min(104, 104 * current.residual / 4.8)" width="48" :height="Math.min(104, 104 * current.residual / 4.8)" rx="5" :class="qAlarm ? 'residual-fill alarm-fill' : 'residual-fill ok-fill'" />
            <line x1="78" x2="148" :y1="178 - 104 * Math.sqrt(qLimit) / 4.8" :y2="178 - 104 * Math.sqrt(qLimit) / 4.8" class="trip-line" />
            <text x="113" y="202" text-anchor="middle" class="status-text">e = {{ current.residual.toFixed(2) }}</text>
          </g>
          <g transform="translate(44 296)">
            <text x="0" y="0" class="meter-label">T²</text>
            <rect x="34" y="-12" width="220" height="16" rx="3" class="meter-track" />
            <rect x="34" y="-12" :width="meterWidth(t2, t2Limit)" height="16" rx="3" :class="t2Alarm ? 'meter alarm-fill' : 'meter ok-fill'" />
            <line :x1="34 + 220 / 1.45" :x2="34 + 220 / 1.45" y1="-16" y2="8" class="trip-line" />
            <text x="264" y="0" class="meter-value">{{ t2.toFixed(2) }} / {{ t2Limit.toFixed(2) }}</text>
          </g>
          <g transform="translate(338 296)">
            <text x="0" y="0" class="meter-label">Q</text>
            <rect x="26" y="-12" width="220" height="16" rx="3" class="meter-track" />
            <rect x="26" y="-12" :width="meterWidth(q, qLimit)" height="16" rx="3" :class="qAlarm ? 'meter alarm-fill' : 'meter ok-fill'" />
            <line :x1="26 + 220 / 1.45" :x2="26 + 220 / 1.45" y1="-16" y2="8" class="trip-line" />
            <text x="254" y="0" text-anchor="end" class="meter-value">{{ q.toFixed(2) }} / {{ qLimit.toFixed(2) }}</text>
          </g>
          <text x="320" y="340" text-anchor="middle" :class="t2Alarm || qAlarm ? 'decision decision-alarm' : 'decision decision-ok'">{{ state }} · {{ current.route }}</text>
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="scenario-bank" role="group" aria-label="选择样本状态">
        <button type="button" :aria-pressed="scenario === 'baseline'" @click="scenario = 'baseline'">基线样本</button>
        <button type="button" :aria-pressed="scenario === 'within'" @click="scenario = 'within'">子空间内远行</button>
        <button type="button" :aria-pressed="scenario === 'outside'" @click="scenario = 'outside'">关系冲突</button>
      </div>
      <label class="control">
        <span>门限倍率 <output>{{ strictness.toFixed(2) }}×</output></span>
        <input v-model.number="strictness" type="range" min="0.7" max="1.4" step="0.01" aria-label="报警门限倍率">
      </label>
    </template>

    <template #metrics>
      <div><span>T² / 门限</span><strong>{{ t2.toFixed(2) }} / {{ t2Limit.toFixed(2) }}</strong></div>
      <div><span>Q / 门限</span><strong>{{ q.toFixed(2) }} / {{ qLimit.toFixed(2) }}</strong></div>
      <div><span>门禁状态</span><strong>{{ state }}</strong></div>
      <div><span>允许路线</span><strong>{{ current.route }}</strong></div>
    </template>

    <template #note>T² 高表示样本沿训练期已知相关结构走远；Q 高表示它离开了该结构。二者都只是复核触发器。即使双表都在限内，也不能替代产品规范、计量确认或安全判定。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.monitor-scroll{overflow-x:auto}.monitor-scroll svg{display:block;width:100%;height:auto}.deck{fill:#091419}.module{fill:#101f25;stroke:#31515a;stroke-width:1.2}.eyebrow,.module-title,.axis-label,.status-text,.meter-label,.meter-value,.decision{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#7ca1aa;font-size:9px;letter-spacing:.12em}.module-title{fill:#43d6c5;font-size:9px;font-weight:800;letter-spacing:.08em}.limit-ellipse{fill:#15312e;fill-opacity:.55;stroke:#43d6c5;stroke-width:1.5;stroke-dasharray:6 4}.axis{stroke:#3e5c64}.axis-label{fill:#718e95;font-size:8px}.score-vector{stroke:#f3ba4f;stroke-width:2}.probe{stroke-width:2}.probe-ok{fill:#43d6c5;stroke:#c9fff8}.probe-alarm{fill:#ff7066;stroke:#ffd5d1}.status-text{fill:#9fb3b8;font-size:8px}.residual-axis{stroke:#45646c}.residual-well{fill:#1c2c31;stroke:#405c64}.residual-fill,.meter{transition:height .18s ease,width .18s ease,y .18s ease}.ok-fill{fill:#43d6c5}.alarm-fill{fill:#ff7066}.trip-line{stroke:#f3ba4f;stroke-width:2}.meter-label{fill:#c8d6d8;font-size:10px;font-weight:800}.meter-track{fill:#24363c}.meter-value{fill:#e8f3f2;font-size:8px}.decision{font-size:11px;font-weight:850;letter-spacing:.04em}.decision-ok{fill:#65ebdb}.decision-alarm{fill:#ff8a81}.scenario-bank{display:grid;gap:8px}button{min-height:44px;border:1px solid #4b6870;border-radius:6px;color:#dbe9e8;background:#122329;font:720 10px/1.25 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#f3ba4f;color:#ffd47b;background:#2b2417}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#b5c4ca;font-size:11px}.control output{color:#f3ba4f;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#43d6c5}button:focus-visible,input:focus-visible{outline:3px solid #f3ba4f;outline-offset:2px}@media(max-width:560px){.monitor-scroll{margin-right:-18px;padding-right:18px}.monitor-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){.residual-fill,.meter{transition:none}}
</style>
