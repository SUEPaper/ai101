<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const currentDeviation = ref(0.45)
const acousticAnomaly = ref(0.35)
const evidenceGain = ref(1.4)
const falseNegativeCost = ref(60)

const falsePositiveCost = 6
const intercept = -0.35
const currentWeight = 1.05
const acousticWeight = 0.85
const sigmoid = (value: number) => value >= 0
  ? 1 / (1 + Math.exp(-value))
  : Math.exp(value) / (1 + Math.exp(value))

const chart = { left: 48, right: 390, top: 42, bottom: 276 }
const xPlot = (value: number) => chart.left + (value + 2) / 4 * (chart.right - chart.left)
const yPlot = (value: number) => chart.bottom - (value + 2) / 4 * (chart.bottom - chart.top)
const cells = Array.from({ length: 144 }, (_, index) => {
  const column = index % 12
  const row = Math.floor(index / 12)
  const x = -2 + (column + 0.5) * 4 / 12
  const y = 2 - (row + 0.5) * 4 / 12
  return { column, row, x, y }
})

const result = computed(() => {
  const rawEvidence = intercept
    + currentWeight * currentDeviation.value
    + acousticWeight * acousticAnomaly.value
  const logit = evidenceGain.value * rawEvidence
  const probability = sigmoid(logit)
  const decisionThreshold = falsePositiveCost / (falsePositiveCost + falseNegativeCost.value)
  const thresholdLogit = Math.log(decisionThreshold / (1 - decisionThreshold))
  const action = probability >= decisionThreshold ? '转离线复核' : '自动放行'
  return {
    rawEvidence,
    logit,
    probability,
    odds: Math.exp(logit),
    decisionThreshold,
    thresholdLogit,
    action,
  }
})

const probabilityFor = (x: number, y: number) => sigmoid(
  evidenceGain.value * (intercept + currentWeight * x + acousticWeight * y)
)
const cellFill = (probability: number) => {
  const alpha = 0.10 + 0.72 * probability
  return `rgba(86, 202, 224, ${alpha.toFixed(3)})`
}
const contour = (targetLogit: number) => {
  const rawTarget = targetLogit / evidenceGain.value
  const yAt = (x: number) => (rawTarget - intercept - currentWeight * x) / acousticWeight
  return {
    x1: xPlot(-2),
    y1: yPlot(yAt(-2)),
    x2: xPlot(2),
    y2: yPlot(yAt(2)),
  }
}
</script>

<template>
  <MlExperimentFrame
    index="09-A"
    title="让一枚焊点穿过证据、概率与动作三层仪表"
    question="边界、置信度和业务阈值分别由什么控制，为什么它们不能混成一个旋钮？"
    formula="z = g(b+wᵀx) → p=σ(z) → a=I[p≥τ(C)]"
  >
    <template #visual>
      <figure class="logit-console">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`当前 logit ${result.logit.toFixed(2)}，缺陷概率 ${(result.probability * 100).toFixed(1)}%，动作 ${result.action}`">
          <defs>
            <clipPath id="logit-field-clip"><rect x="48" y="42" width="342" height="234" rx="5" /></clipPath>
          </defs>
          <rect x="15" y="15" width="570" height="296" rx="9" class="console-bg" />
          <text x="48" y="30" class="panel-title">焊缝证据场 / 两个标准化传感器</text>
          <g clip-path="url(#logit-field-clip)">
            <rect
              v-for="cell in cells"
              :key="`${cell.column}-${cell.row}`"
              :x="xPlot(-2 + cell.column * 4 / 12)"
              :y="yPlot(2 - cell.row * 4 / 12)"
              :width="(chart.right - chart.left) / 12 + .5"
              :height="(chart.bottom - chart.top) / 12 + .5"
              :fill="cellFill(probabilityFor(cell.x, cell.y))"
            />
            <line v-for="tick in [-1,0,1]" :key="`x-${tick}`" :x1="xPlot(tick)" :x2="xPlot(tick)" y1="42" y2="276" class="grid" />
            <line v-for="tick in [-1,0,1]" :key="`y-${tick}`" x1="48" x2="390" :y1="yPlot(tick)" :y2="yPlot(tick)" class="grid" />
            <line v-bind="contour(0)" class="probability-boundary" />
            <line v-bind="contour(result.thresholdLogit)" class="action-boundary" />
          </g>
          <circle :cx="xPlot(currentDeviation)" :cy="yPlot(acousticAnomaly)" r="8" class="sample-point" />
          <line :x1="xPlot(currentDeviation)" :x2="xPlot(currentDeviation)" :y1="yPlot(acousticAnomaly) + 12" y2="276" class="query-line" />
          <text x="48" y="297" class="axis-label">电流波形偏移 x₁ →</text>
          <text x="20" y="54" class="axis-label">声发射 x₂</text>

          <text x="424" y="31" class="panel-title">证据链</text>
          <text x="424" y="64" class="ledger-label">原始证据 b+wᵀx</text>
          <text x="563" y="64" class="ledger-value">{{ result.rawEvidence.toFixed(2) }}</text>
          <text x="424" y="95" class="ledger-label">logit z</text>
          <text x="563" y="95" class="ledger-value">{{ result.logit.toFixed(2) }}</text>
          <text x="424" y="126" class="ledger-label">odds</text>
          <text x="563" y="126" class="ledger-value">{{ result.odds.toFixed(2) }} : 1</text>
          <line x1="424" x2="564" y1="145" y2="145" class="ledger-rule" />
          <text x="424" y="174" class="ledger-label">缺陷概率</text>
          <text x="563" y="174" class="ledger-value probability">{{ (result.probability * 100).toFixed(1) }}%</text>
          <rect x="424" y="190" width="140" height="10" rx="5" class="probability-track" />
          <rect x="424" y="190" :width="140 * result.probability" height="10" rx="5" class="probability-fill" />
          <line :x1="424 + 140 * result.decisionThreshold" :x2="424 + 140 * result.decisionThreshold" y1="184" y2="206" class="threshold-mark" />
          <text x="424" y="228" class="ledger-label">成本阈值 τ</text>
          <text x="563" y="228" class="ledger-value threshold">{{ result.decisionThreshold.toFixed(3) }}</text>
          <text x="424" y="267" :class="['action-label', result.action === '转离线复核' ? 'review' : 'release']">{{ result.action }}</text>
          <text x="424" y="291" class="ledger-foot">蓝线 p=0.5 · 橙线 p=τ(C)</text>
        </svg>
        <figcaption>
          <span class="legend field" />Sigmoid 概率场
          <span class="legend probability" />统计边界 p=0.5
          <span class="legend action" />成本动作边界
          <span class="legend sample" />当前焊点
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>电流波形偏移 x₁ <output>{{ currentDeviation.toFixed(2) }}</output></span>
        <input v-model.number="currentDeviation" type="range" min="-2" max="2" step="0.05">
      </label>
      <label>
        <span>声发射异常 x₂ <output>{{ acousticAnomaly.toFixed(2) }}</output></span>
        <input v-model.number="acousticAnomaly" type="range" min="-2" max="2" step="0.05">
      </label>
      <label>
        <span>证据增益 g <output>{{ evidenceGain.toFixed(2) }}</output></span>
        <input v-model.number="evidenceGain" type="range" min="0.35" max="3" step="0.05">
      </label>
      <label>
        <span>漏放代价 C_FN <output>{{ falseNegativeCost }}</output></span>
        <input v-model.number="falseNegativeCost" type="range" min="12" max="150" step="3">
      </label>
      <p>增益只改变概率从 0 到 1 的陡峭程度，不移动蓝色 0.5 边界；改变代价会移动橙色动作边界，却不会重新训练概率模型。</p>
    </template>
    <template #metrics>
      <div><span>原始线性证据</span><strong>{{ result.rawEvidence.toFixed(3) }}</strong></div>
      <div><span>logit / log odds</span><strong>{{ result.logit.toFixed(3) }}</strong></div>
      <div><span>缺陷后验</span><strong>{{ (result.probability * 100).toFixed(1) }}%</strong></div>
      <div><span>当前动作</span><strong>{{ result.action }}</strong></div>
    </template>
    <template #note>
      这里的传感器量、系数和代价都是教学合成值。真实焊缝放行必须由工艺规范、无损检测能力和安全评审共同约束；模型概率不能覆盖强制检验规则。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.logit-console{margin:0}.logit-console svg{display:block;width:100%;border:1px solid #294764;border-radius:10px;background:#0c1722}.console-bg{fill:#0c1722}.grid{stroke:#31506b;stroke-width:1;opacity:.7}.probability-boundary{stroke:#dcecf3;stroke-width:2.2;stroke-dasharray:7 5}.action-boundary{stroke:#f58a4b;stroke-width:3}.sample-point{fill:#f5c85c;stroke:#fff;stroke-width:2}.query-line{stroke:#f5c85c;stroke-width:1.5;stroke-dasharray:4 4}.logit-console text{font:700 10px ui-monospace,monospace}.panel-title{fill:#dcecf3}.axis-label,.ledger-label,.ledger-foot{fill:#8da8bc}.ledger-value{fill:#dcecf3;text-anchor:end;font-size:13px}.ledger-value.probability{fill:#63d0e2}.ledger-value.threshold{fill:#f58a4b}.ledger-rule{stroke:#35526c}.probability-track{fill:#263b4e}.probability-fill{fill:#63d0e2}.threshold-mark{stroke:#f58a4b;stroke-width:3}.action-label{font-size:15px}.action-label.review{fill:#f58a4b}.action-label.release{fill:#63d0e2}.logit-console figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#91aabd;font-size:10px;line-height:1.6}.legend{display:inline-block;width:14px;height:3px;margin-right:4px;vertical-align:middle}.legend.field{background:#63d0e2}.legend.probability{border-top:2px dashed #dcecf3}.legend.action{background:#f58a4b}.legend.sample{height:10px;border-radius:50%;background:#f5c85c}.ml-experiment__controls label{display:grid;gap:5px;color:#c6d7e4;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0e2;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f58a4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#91aabd;font-size:10px;line-height:1.65}@media(max-width:560px){.logit-console{overflow-x:auto;overscroll-behavior-inline:contain}.logit-console svg{width:560px;max-width:none}.logit-console figcaption{min-width:560px}}
</style>
