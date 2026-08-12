<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type ActionKey = 'east' | 'inspect'

const includeSeal = ref(false)
const action = ref<ActionKey>('east')

const probabilities: Record<ActionKey, { intact: number; alarm: number }> = {
  east: { intact: 0.09, alarm: 0.62 },
  inspect: { intact: 0.05, alarm: 0.13 },
}

const aliasGap = computed(() => Math.abs(probabilities[action.value].alarm - probabilities[action.value].intact))
const stateA = computed(() => includeSeal.value ? 'C2 · 密封正常' : 'C2')
const stateB = computed(() => includeSeal.value ? 'C2 · 密封告警' : 'C2')
const auditStatus = computed(() => includeSeal.value ? '本样本解除别名' : '状态别名成立')

function reset() {
  includeSeal.value = false
  action.value = 'east'
}
</script>

<template>
  <MlExperimentFrame
    index="30-A"
    title="Markov 状态审计台"
    question="两段历史到达同一格位，却给出不同的下一步风险：问题在环境，还是在状态把关键历史丢掉了？"
    formula="P(Sₜ₊₁|Hₜ,Aₜ) = P(Sₜ₊₁|Sₜ,Aₜ)"
  >
    <template #visual>
      <div class="audit-scroll">
        <svg viewBox="0 0 700 400" role="img" aria-label="两段到达同一格位但密封状态不同的事件历史及下一状态风险">
          <rect class="floor" x="8" y="8" width="684" height="384" rx="4" />
          <path class="lane" d="M 32 96 H 668 M 32 258 H 668" />
          <text class="stencil" x="34" y="42">AMR EVENT RECORDER · STATE SUFFICIENCY AUDIT</text>

          <g transform="translate(36 62)">
            <text class="run-id" x="0" y="17">H-17</text>
            <g v-for="(label, index) in ['装载', '门磁闭合', '密封正常', '到达 C2']" :key="label" :transform="`translate(${78 + index * 108} 0)`">
              <rect class="event normal" width="94" height="54" rx="2" />
              <text class="event-text" x="47" y="31" text-anchor="middle">{{ label }}</text>
            </g>
            <path class="track safe" d="M 80 74 H 504" />
            <rect class="state-tag" x="520" y="0" width="130" height="54" rx="2" />
            <text class="state-text" x="585" y="23" text-anchor="middle">编码状态</text>
            <text class="state-value" x="585" y="40" text-anchor="middle">{{ stateA }}</text>
          </g>

          <g transform="translate(36 224)">
            <text class="run-id" x="0" y="17">H-24</text>
            <g v-for="(label, index) in ['装载', '门磁抖动', '密封告警', '到达 C2']" :key="label" :transform="`translate(${78 + index * 108} 0)`">
              <rect class="event" :class="index >= 1 && index <= 2 ? 'alarm' : 'normal'" width="94" height="54" rx="2" />
              <text class="event-text" x="47" y="31" text-anchor="middle">{{ label }}</text>
            </g>
            <path class="track alarm-track" d="M 80 74 H 504" />
            <rect class="state-tag" x="520" y="0" width="130" height="54" rx="2" />
            <text class="state-text" x="585" y="23" text-anchor="middle">编码状态</text>
            <text class="state-value" x="585" y="40" text-anchor="middle">{{ stateB }}</text>
          </g>

          <g transform="translate(114 155)">
            <text class="risk-label" x="0" y="0">执行「{{ action === 'east' ? '东移' : '原地检查' }}」后进入隔离区的经验概率</text>
            <rect class="risk-track" x="0" y="15" width="430" height="16" />
            <rect class="risk-safe" x="0" y="15" :width="430 * probabilities[action].intact" height="16" />
            <text class="risk-value safe-text" x="448" y="28">H-17 {{ (probabilities[action].intact * 100).toFixed(0) }}%</text>
            <rect class="risk-track" x="0" y="43" width="430" height="16" />
            <rect class="risk-alarm" x="0" y="43" :width="430 * probabilities[action].alarm" height="16" />
            <text class="risk-value alarm-text" x="448" y="56">H-24 {{ (probabilities[action].alarm * 100).toFixed(0) }}%</text>
          </g>
          <text class="footnote" x="36" y="376">相同位置不是充分状态；补入密封告警只能修复已发现的别名，不能证明完整 Markov 性。</text>
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="control-group" role="group" aria-label="状态字段选择">
        <span>状态编码</span>
        <button type="button" :aria-pressed="!includeSeal" @click="includeSeal = false">仅格位</button>
        <button type="button" :aria-pressed="includeSeal" @click="includeSeal = true">格位 + 密封状态</button>
      </div>
      <label class="control">
        <span>待审计动作</span>
        <select v-model="action" aria-label="待审计动作">
          <option value="east">东移至交接区</option>
          <option value="inspect">原地检查</option>
        </select>
      </label>
      <button type="button" class="reset" @click="reset">复位审计台</button>
    </template>

    <template #metrics>
      <div><span>H-17 隔离概率</span><strong>{{ (probabilities[action].intact * 100).toFixed(0) }}%</strong></div>
      <div><span>H-24 隔离概率</span><strong>{{ (probabilities[action].alarm * 100).toFixed(0) }}%</strong></div>
      <div><span>同动作条件差</span><strong>{{ (aliasGap * 100).toFixed(0) }} pp</strong></div>
      <div><span>审计结论</span><strong>{{ auditStatus }}</strong></div>
    </template>

    <template #note>当只编码格位时，两段历史被压成同一个状态，但同一动作的下一状态分布相差 {{ (aliasGap * 100).toFixed(0) }} 个百分点，直接反驳“位置已足够”的假设。加入密封状态后，两段历史不再被错误合并；这只是通过了一项针对性审计，不等于已经证明所有未来都与更早历史条件独立。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.audit-scroll{overflow-x:auto}.audit-scroll svg{display:block;width:100%;height:auto}.floor{fill:#242a2e;stroke:#f2b705;stroke-width:2}.lane{fill:none;stroke:#e8e2d6;stroke-width:1;stroke-dasharray:14 10;opacity:.3}.stencil,.run-id,.event-text,.state-text,.state-value,.risk-label,.risk-value,.footnote{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.stencil{fill:#f2b705;font-size:9px;font-weight:900;letter-spacing:.11em}.run-id{fill:#e8e2d6;font-size:11px;font-weight:900}.event{stroke-width:1.4}.event.normal{fill:#303a3f;stroke:#6f9f58}.event.alarm{fill:#4a2c29;stroke:#d64b3c}.event-text{fill:#f4f0e8;font-size:9px;font-weight:760}.track{fill:none;stroke-width:4;stroke-dasharray:18 7}.track.safe{stroke:#6f9f58}.alarm-track{stroke:#d64b3c}.state-tag{fill:#e8e2d6;stroke:#f2b705;stroke-width:1.5}.state-text{fill:#596168;font-size:7px;letter-spacing:.08em}.state-value{fill:#242a2e;font-size:9px;font-weight:900}.risk-label{fill:#c9d0d2;font-size:9px}.risk-track{fill:#15191c;stroke:#596168}.risk-safe{fill:#6f9f58}.risk-alarm{fill:#d64b3c}.risk-value{font-size:9px;font-weight:850}.safe-text{fill:#a9cc99}.alarm-text{fill:#ff9f91}.footnote{fill:#b8c0c2;font-size:8px}.control-group{display:grid;gap:8px}.control-group>span,.control>span{color:#c3cee8;font-size:11px}.control{display:grid;gap:8px}button,select{width:100%;min-height:44px;padding:9px;border:1px solid #65737a;border-radius:3px;color:#e8e2d6;background:#242a2e;font:720 10px/1.35 ui-monospace,monospace}button{cursor:pointer}button[aria-pressed="true"]{border-color:#f2b705;color:#fff1b5;background:#4a401d}.reset{border-color:#6f9f58;color:#cce8bf}button:focus-visible,select:focus-visible{outline:3px solid #f2b705;outline-offset:2px}@media(max-width:560px){.audit-scroll{margin-right:-18px;padding-right:18px}.audit-scroll svg{width:720px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
