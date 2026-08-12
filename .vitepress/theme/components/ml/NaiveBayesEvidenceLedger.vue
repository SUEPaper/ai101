<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const alpha = ref(1)
const oneXCount = ref(2)
const envelopeCount = ref(3)
const temperatureCount = ref(1)
const dropoutCount = ref(0)
const duplicateFeed = ref(false)
const labels = ['先验', '1× 峰', '包络抬升', '温升', '传感器掉线', '镜像 1× 馈线']
const aggregateCounts = [
  [92, 14, 11, 45, 0],
  [18, 87, 76, 20, 3],
]

const tokenProbabilities = computed(() => aggregateCounts.map(row => {
  const denominator = row.reduce((sum, value) => sum + value, 0) + alpha.value * row.length
  return row.map(value => (value + alpha.value) / denominator)
}))
const contributions = computed(() => {
  const probabilities = tokenProbabilities.value
  const logRatio = probabilities[1].map((value, index) => Math.log(value / probabilities[0][index]))
  const prior = Math.log(.45 / .55)
  return [
    prior,
    oneXCount.value * logRatio[0],
    envelopeCount.value * logRatio[1],
    temperatureCount.value * logRatio[2],
    dropoutCount.value * logRatio[4],
    duplicateFeed.value ? oneXCount.value * logRatio[0] : 0,
  ]
})
const logOdds = computed(() => contributions.value.reduce((sum, value) => sum + value, 0))
const posterior = computed(() => logOdds.value >= 0 ? 1 / (1 + Math.exp(-logOdds.value)) : Math.exp(logOdds.value) / (1 + Math.exp(logOdds.value)))
const verdict = computed(() => posterior.value >= .5 ? '轴承预紧异常' : '转子不平衡')
const barWidth = (value: number) => Math.min(176, Math.abs(value) * 34)
const barX = (value: number) => value >= 0 ? 318 : 318 - barWidth(value)
const unseenProbability = computed(() => tokenProbabilities.value[0][4])
</script>

<template>
  <MlExperimentFrame
    index="23-C"
    title="逐项核销朴素贝叶斯证据"
    question="Laplace 平滑怎样解除零概率？若两条馈线其实复制同一只 1× 峰报警，朴素贝叶斯又会怎样把同一证据算两遍？"
    formula="log odds = log(π₁/π₀) + Σj xj log(θj1/θj0)"
  >
    <template #visual>
      <div class="evidence-scroll">
        <svg viewBox="0 0 640 360" role="img" :aria-label="`朴素贝叶斯判为${verdict}，轴承预紧异常后验${(100 * posterior).toFixed(1)}%，对数优势${logOdds.toFixed(2)}`">
          <rect width="640" height="360" rx="15" class="panel" />
          <text x="24" y="27" class="eyebrow">ALARM EVIDENCE LEDGER · LOG-ODDS ACCOUNTING</text>
          <text x="318" y="52" text-anchor="middle" class="zero-label">0 · 决策中性线</text>
          <line x1="318" x2="318" y1="60" y2="291" class="zero-line" />
          <text x="142" y="52" text-anchor="middle" class="side-label side-zero">← 转子不平衡证据</text>
          <text x="494" y="52" text-anchor="middle" class="side-label side-one">轴承预紧证据 →</text>
          <g v-for="(value, index) in contributions" :key="labels[index]">
            <text x="24" :y="87 + index * 34" class="row-label">{{ labels[index] }}</text>
            <rect :x="barX(value)" :y="72 + index * 34" :width="barWidth(value)" height="20" rx="3" :class="value >= 0 ? 'bar bar-one' : 'bar bar-zero'" />
            <text :x="value >= 0 ? barX(value) + barWidth(value) + 7 : barX(value) - 7" :y="87 + index * 34" :text-anchor="value >= 0 ? 'start' : 'end'" class="bar-value">{{ value >= 0 ? '+' : '' }}{{ value.toFixed(2) }}</text>
          </g>
          <rect x="24" y="301" width="592" height="38" rx="5" class="verdict" :class="{ alarm:posterior >= .5 }" />
          <text x="42" y="317" class="verdict-label">冻结账本结论</text>
          <text x="42" y="331" class="verdict-value">{{ verdict }} · P(预紧异常 | 日志) = {{ (100 * posterior).toFixed(1) }}%</text>
          <text x="600" y="324" text-anchor="end" class="log-odds">log odds {{ logOdds.toFixed(2) }}</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>Laplace α <output>{{ alpha.toFixed(2) }}</output></span><input v-model.number="alpha" aria-label="Laplace平滑强度" type="range" min="0.05" max="10" step="0.05"></label>
      <label class="control"><span>1× 峰计数 <output>{{ oneXCount }}</output></span><input v-model.number="oneXCount" aria-label="一倍频峰计数" type="range" min="0" max="8" step="1"></label>
      <label class="control"><span>包络抬升计数 <output>{{ envelopeCount }}</output></span><input v-model.number="envelopeCount" aria-label="包络抬升计数" type="range" min="0" max="8" step="1"></label>
      <label class="control"><span>温升计数 <output>{{ temperatureCount }}</output></span><input v-model.number="temperatureCount" aria-label="温升计数" type="range" min="0" max="8" step="1"></label>
      <label class="control"><span>传感器掉线计数 <output>{{ dropoutCount }}</output></span><input v-model.number="dropoutCount" aria-label="传感器掉线计数" type="range" min="0" max="4" step="1"></label>
      <button type="button" :class="{ active:duplicateFeed }" :aria-pressed="duplicateFeed" @click="duplicateFeed = !duplicateFeed">{{ duplicateFeed ? '镜像馈线已接入：同一 1× 峰重复记账' : '接入镜像 1× 馈线' }}</button>
    </template>
    <template #metrics>
      <div><span>类 0 未见掉线词概率</span><strong>{{ (100 * unseenProbability).toFixed(3) }}%</strong></div>
      <div><span>累计 log odds</span><strong>{{ logOdds.toFixed(3) }}</strong></div>
      <div><span>预紧异常后验</span><strong>{{ (100 * posterior).toFixed(1) }}%</strong></div>
      <div><span>相关证据状态</span><strong>{{ duplicateFeed ? '重复计数' : '单路记账' }}</strong></div>
    </template>
    <template #note>α 只解决有限样本下的零计数，不会让条件独立假设变真。镜像馈线开启后没有新增物理传感器，却再次加入同一项 log likelihood ratio；后验通常更极端，但信息量没有翻倍。生产协议应先按物理来源去重，再做概率校准。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.evidence-scroll{overflow-x:auto}.evidence-scroll svg{display:block;width:100%;height:auto}.panel{fill:#0b1218}.zero-line{stroke:#6f8791;stroke-width:1;stroke-dasharray:4 4}.bar{stroke-width:1}.bar-zero{fill:#1daea6;stroke:#45ded3}.bar-one{fill:#d79a34;stroke:#ffc864}.verdict{fill:#112c2b;stroke:#30c6bb}.verdict.alarm{fill:#3b2c18;stroke:#efb449}.eyebrow,.zero-label,.side-label,.row-label,.bar-value,.verdict-label,.verdict-value,.log-odds{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#70a6b3;font-size:9px;letter-spacing:.12em}.zero-label{fill:#8299a3;font-size:8px}.side-label{font-size:8px;font-weight:800}.side-zero{fill:#54d9d0}.side-one{fill:#ffc864}.row-label{fill:#a7bac1;font-size:8px}.bar-value{fill:#dce7ea;font-size:8px}.verdict-label{fill:#92a7af;font-size:7px}.verdict-value{fill:#f0d27a;font-size:10px;font-weight:800}.log-odds{fill:#d9a4ef;font-size:9px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebdc5;font-size:11px}.control output{color:#ffc864;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#23c9bd}button{min-height:44px;padding:9px 12px;border:1px solid #49606c;border-radius:6px;color:#dce8eb;background:#13232c;font-weight:760;cursor:pointer}button.active{border-color:#ff6b5f;color:#ffe5e1;background:#452722}button:focus-visible,input:focus-visible{outline:3px solid #d477ff;outline-offset:2px}@media(max-width:560px){.evidence-scroll{margin-right:-18px;padding-right:18px}.evidence-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
