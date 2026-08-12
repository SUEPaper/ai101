<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Candidate = {
  k: number
  wcss: number
  gain: number
  silhouette: number
  stabilityP10: number
  minShare: number
  auditAri: number
}

const candidates: Candidate[] = [
  { k: 2, wcss: 516.3, gain: 0, silhouette: 0.666, stabilityP10: 1.000, minShare: 0.375, auditAri: 0.770 },
  { k: 3, wcss: 133.6, gain: 0.741, silhouette: 0.736, stabilityP10: 1.000, minShare: 0.167, auditAri: 1.000 },
  { k: 4, wcss: 120.6, gain: 0.097, silhouette: 0.489, stabilityP10: 0.817, minShare: 0.167, auditAri: 0.969 },
  { k: 5, wcss: 111.7, gain: 0.074, silhouette: 0.264, stabilityP10: 0.599, minShare: 0.167, auditAri: 0.668 },
  { k: 6, wcss: 103.8, gain: 0.071, silhouette: 0.263, stabilityP10: 0.596, minShare: 0.143, auditAri: 0.684 },
]

const selectedK = ref(3)
const stabilityGate = ref(0.80)
const minimumShare = ref(0.15)
const queueCapacity = ref(3)
const revealAudit = ref(false)

function gates(row: Candidate) {
  return {
    geometry: row.silhouette >= 0.45,
    stability: row.stabilityP10 >= stabilityGate.value,
    load: row.minShare >= minimumShare.value,
    capacity: row.k <= queueCapacity.value,
  }
}

function passCount(row: Candidate) {
  return Object.values(gates(row)).filter(Boolean).length
}

const selected = computed(() => candidates.find(row => row.k === selectedK.value) ?? candidates[1])
const eligible = computed(() => candidates.filter(row => passCount(row) === 4))
const recommendation = computed(() => {
  const pool = eligible.value.length ? eligible.value : candidates
  return [...pool].sort((a, b) => b.silhouette - a.silhouette)[0]
})

function barWidth(value: number, maximum: number) {
  return Math.max(2, value / maximum * 112)
}
</script>

<template>
  <MlExperimentFrame
    index="27-A"
    title="聚类证据审签台"
    question="如果几何、稳定性和复核产能不是同一种证据，怎样在查看少量审计标签之前冻结 K？"
    formula="accept(K) = geometry ∧ stability ∧ load ∧ capacity"
  >
    <template #visual>
      <div class="ledger-scroll">
        <svg viewBox="0 0 640 390" role="img" aria-label="K 等于 2 到 6 的聚类候选证据台账">
          <rect class="paper" x="14" y="14" width="612" height="362" rx="5" />
          <text class="folio" x="32" y="39">EVIDENCE DOCKET · COMPRESSOR REVIEW QUEUES · REV 27</text>
          <text class="column" x="42" y="72">候选</text>
          <text class="column" x="112" y="72">WCSS / 边际降幅</text>
          <text class="column" x="292" y="72">轮廓</text>
          <text class="column" x="404" y="72">稳定 P10</text>
          <text class="column" x="518" y="72">门禁</text>
          <g
            v-for="(row, index) in candidates"
            :key="row.k"
            class="candidate"
            :class="{ selected: selectedK === row.k, recommended: recommendation.k === row.k }"
            :transform="`translate(0 ${88 + index * 54})`"
            role="button"
            tabindex="0"
            :aria-label="`选择 K 等于 ${row.k}`"
            @click="selectedK = row.k"
            @keydown.enter.prevent="selectedK = row.k"
            @keydown.space.prevent="selectedK = row.k"
          >
            <rect class="row" x="27" y="0" width="586" height="44" rx="3" />
            <text class="k" x="48" y="29">K={{ row.k }}</text>
            <rect class="track" x="112" y="10" width="112" height="8" rx="4" />
            <rect class="wcss" x="112" y="10" :width="barWidth(row.wcss, 516.3)" height="8" rx="4" />
            <text class="datum" x="112" y="35">{{ row.wcss.toFixed(1) }} · {{ row.k === 2 ? '基线' : `↓ ${(row.gain * 100).toFixed(1)}%` }}</text>
            <rect class="track" x="292" y="10" width="82" height="8" rx="4" />
            <rect class="silhouette" x="292" y="10" :width="barWidth(row.silhouette, 1) * 82 / 112" height="8" rx="4" />
            <text class="datum" x="292" y="35">{{ row.silhouette.toFixed(3) }}</text>
            <rect class="track" x="404" y="10" width="82" height="8" rx="4" />
            <rect class="stability" x="404" y="10" :width="barWidth(row.stabilityP10, 1) * 82 / 112" height="8" rx="4" />
            <text class="datum" x="404" y="35">{{ row.stabilityP10.toFixed(3) }}</text>
            <g v-for="(passed, gateIndex) in Object.values(gates(row))" :key="gateIndex">
              <circle :class="passed ? 'gate-pass' : 'gate-fail'" :cx="526 + gateIndex * 19" cy="16" r="6" />
            </g>
            <text class="datum" x="518" y="35">{{ passCount(row) }}/4 通过</text>
            <path v-if="recommendation.k === row.k" class="verdict" d="M 600 4 l 8 8 -8 8 -8 -8 z" />
          </g>
          <text class="seal" x="600" y="361" text-anchor="end">{{ revealAudit ? 'AUDIT OPENED AFTER SHORTLIST' : 'AUDIT SEALED' }}</text>
        </svg>
      </div>
    </template>

    <template #controls>
      <label class="control">
        <span>稳定性 P10 门槛 <output>{{ stabilityGate.toFixed(2) }}</output></span>
        <input v-model.number="stabilityGate" type="range" min="0.55" max="0.95" step="0.01" aria-label="稳定性第十百分位门槛">
      </label>
      <label class="control">
        <span>最小簇占比 <output>{{ (minimumShare * 100).toFixed(0) }}%</output></span>
        <input v-model.number="minimumShare" type="range" min="0.10" max="0.22" step="0.01" aria-label="最小簇占比门槛">
      </label>
      <label class="control">
        <span>人工队列上限 <output>{{ queueCapacity }} 个</output></span>
        <input v-model.number="queueCapacity" type="range" min="2" max="6" step="1" aria-label="人工复核队列上限">
      </label>
      <button type="button" :aria-pressed="revealAudit" @click="revealAudit = !revealAudit">
        {{ revealAudit ? '重新封存审计标签' : '候选冻结后揭示审计标签' }}
      </button>
    </template>

    <template #metrics>
      <div><span>当前候选</span><strong>K={{ selected.k }}</strong></div>
      <div><span>通过门禁</span><strong>{{ passCount(selected) }} / 4</strong></div>
      <div><span>规则推荐</span><strong>K={{ recommendation.k }}</strong></div>
      <div><span>少量审计 ARI</span><strong>{{ revealAudit ? selected.auditAri.toFixed(3) : '已封存' }}</strong></div>
    </template>

    <template #note>门禁不是把不同量纲的分数硬加权。默认合同下 K=2 与 K=3 都可执行，再用预先声明的轮廓证据在可行集合中选 K=3；提高最小簇占比或收紧队列上限会合法地改变结论。审计 ARI 只在候选冻结后揭示，防止把外部标签悄悄变成调参集。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.ledger-scroll{overflow-x:auto}.ledger-scroll svg{display:block;width:100%;height:auto}.paper{fill:#eee6d2;stroke:#c69a48;stroke-width:2}.folio,.column,.datum,.seal{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.folio{fill:#17334a;font-size:10px;font-weight:850;letter-spacing:.08em}.column{fill:#674b28;font-size:9px;font-weight:800}.candidate{cursor:pointer}.row{fill:#f8f2e3;stroke:#b9aa8d}.candidate.selected .row{fill:#dce8e4;stroke:#16998c;stroke-width:2}.candidate.recommended .row{stroke:#c94937;stroke-width:2}.k{fill:#12283a;font:800 16px/1 Georgia,ui-serif,serif}.track{fill:#cfc4ad}.wcss{fill:#c49a4a}.silhouette{fill:#1c9c90}.stability{fill:#355f86}.datum{fill:#35434a;font-size:9px;font-weight:700}.gate-pass{fill:#1c9c90;stroke:#0f655d}.gate-fail{fill:#c94937;stroke:#7b2c23}.verdict{fill:#c94937}.seal{fill:#a43d30;font-size:9px;font-weight:900;letter-spacing:.07em}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#c2cee9;font-size:11px}.control output{color:#f1ce79;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#2aa198}button{min-height:44px;padding:10px 12px;border:1px solid #c49a4a;border-radius:4px;color:#f5dfaa;background:#2b261c;font:760 11px/1.35 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#2aa198;color:#c9fff8;background:#173c39}button:focus-visible,input:focus-visible,.candidate:focus-visible{outline:3px solid #f0c75e;outline-offset:2px}@media(max-width:560px){.ledger-scroll{margin-right:-18px;padding-right:18px}.ledger-scroll svg{width:700px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
