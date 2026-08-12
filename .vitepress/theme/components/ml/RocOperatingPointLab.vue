<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const threshold = ref(0.55)
const deploymentPrevalence = ref(0.04)
const falseNegativeCost = ref(120)
const falsePositiveCost = 4

const clamp = (value: number) => Math.max(0.001, Math.min(0.999, value))
const positiveScores = Array.from({ length: 48 }, (_, index) =>
  clamp(0.31 + 0.64 * (index / 47) ** 0.72 + 0.035 * Math.sin(index * 1.9))
)
const negativeScores = Array.from({ length: 160 }, (_, index) =>
  clamp(0.02 + 0.72 * (index / 159) ** 1.8 + 0.025 * Math.sin(index * 2.1))
)
const thresholdGrid = Array.from({ length: 101 }, (_, index) => 1 - index / 100)

const ratesAt = (cutoff: number) => {
  const tp = positiveScores.filter((score) => score >= cutoff).length
  const fp = negativeScores.filter((score) => score >= cutoff).length
  return {
    threshold: cutoff,
    tpr: tp / positiveScores.length,
    fpr: fp / negativeScores.length,
  }
}

const rocPoints = thresholdGrid.map(ratesAt)
const auc = (() => {
  let area = 0
  for (let index = 1; index < rocPoints.length; index++) {
    const left = rocPoints[index - 1]
    const right = rocPoints[index]
    area += (right.fpr - left.fpr) * (left.tpr + right.tpr) / 2
  }
  return area
})()

const chart = { left: 55, right: 390, top: 35, bottom: 272 }
const xPlot = (fpr: number) => chart.left + fpr * (chart.right - chart.left)
const yPlot = (tpr: number) => chart.bottom - tpr * (chart.bottom - chart.top)
const rocPolyline = rocPoints.map((point) => `${xPlot(point.fpr)},${yPlot(point.tpr)}`).join(' ')

const result = computed(() => {
  const current = ratesAt(threshold.value)
  const expectedCost = (point: { tpr: number; fpr: number }) =>
    deploymentPrevalence.value * (1 - point.tpr) * falseNegativeCost.value
    + (1 - deploymentPrevalence.value) * point.fpr * falsePositiveCost
  const optimal = rocPoints.reduce((best, point) =>
    expectedCost(point) < expectedCost(best) ? point : best
  )
  const denominator = (
    deploymentPrevalence.value * current.tpr
    + (1 - deploymentPrevalence.value) * current.fpr
  )
  const precision = denominator > 0
    ? deploymentPrevalence.value * current.tpr / denominator
    : 1
  const slope = (
    (1 - deploymentPrevalence.value) * falsePositiveCost
    / (deploymentPrevalence.value * falseNegativeCost.value)
  )
  const xStart = slope > 0 ? Math.max(0, optimal.fpr - optimal.tpr / slope) : 0
  const xEnd = slope > 0 ? Math.min(1, optimal.fpr + (1 - optimal.tpr) / slope) : 1
  const yStart = optimal.tpr + slope * (xStart - optimal.fpr)
  const yEnd = optimal.tpr + slope * (xEnd - optimal.fpr)
  const population = 1000
  return {
    current,
    optimal,
    precision,
    expectedCostPerThousand: expectedCost(current) * population,
    optimalCostPerThousand: expectedCost(optimal) * population,
    truePositives: population * deploymentPrevalence.value * current.tpr,
    falseNegatives: population * deploymentPrevalence.value * (1 - current.tpr),
    falsePositives: population * (1 - deploymentPrevalence.value) * current.fpr,
    isoCost: { xStart, xEnd, yStart, yEnd },
  }
})
</script>

<template>
  <MlExperimentFrame
    index="08-B"
    title="沿 ROC 验收轨道选择运行点"
    question="同一条排序曲线，在缺陷率和漏报代价变化后，为什么最优运行点会移动？"
    formula="cost = π C_FN(1−TPR) + (1−π) C_FP·FPR"
  >
    <template #visual>
      <figure class="roc-console">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`阈值 ${threshold.toFixed(2)}，召回率 ${result.current.tpr.toFixed(2)}，误报率 ${result.current.fpr.toFixed(2)}`">
          <rect x="16" y="16" width="568" height="294" rx="9" class="console-bg" />
          <text x="55" y="27" class="panel-title">冻结验证集 ROC / 部署运行点</text>
          <line v-for="tick in [0,.25,.5,.75,1]" :key="`roc-y-${tick}`" x1="55" x2="390" :y1="yPlot(tick)" :y2="yPlot(tick)" class="grid" />
          <line v-for="tick in [0,.25,.5,.75,1]" :key="`roc-x-${tick}`" :x1="xPlot(tick)" :x2="xPlot(tick)" y1="35" y2="272" class="grid" />
          <line x1="55" y1="272" x2="390" y2="35" class="chance-line" />
          <polyline :points="rocPolyline" class="roc-line" />
          <line
            :x1="xPlot(result.isoCost.xStart)"
            :y1="yPlot(result.isoCost.yStart)"
            :x2="xPlot(result.isoCost.xEnd)"
            :y2="yPlot(result.isoCost.yEnd)"
            class="iso-cost"
          />
          <circle :cx="xPlot(result.current.fpr)" :cy="yPlot(result.current.tpr)" r="8" class="current-point" />
          <circle :cx="xPlot(result.optimal.fpr)" :cy="yPlot(result.optimal.tpr)" r="6" class="optimal-point" />
          <text x="55" y="294" class="axis-label">FPR →</text>
          <text x="24" y="46" class="axis-label">TPR</text>
          <text x="430" y="52" class="ledger-title">每千件部署账本</text>
          <text x="430" y="82" class="ledger-label">真实缺陷</text>
          <text x="558" y="82" class="ledger-value">{{ (deploymentPrevalence * 1000).toFixed(0) }}</text>
          <text x="430" y="111" class="ledger-label">检出 TP</text>
          <text x="558" y="111" class="ledger-value good">{{ result.truePositives.toFixed(1) }}</text>
          <text x="430" y="140" class="ledger-label">漏放 FN</text>
          <text x="558" y="140" class="ledger-value bad">{{ result.falseNegatives.toFixed(1) }}</text>
          <text x="430" y="169" class="ledger-label">误报 FP</text>
          <text x="558" y="169" class="ledger-value warn">{{ result.falsePositives.toFixed(1) }}</text>
          <line x1="430" x2="560" y1="188" y2="188" class="ledger-rule" />
          <text x="430" y="216" class="ledger-label">当前成本</text>
          <text x="558" y="216" class="ledger-value bad">{{ result.expectedCostPerThousand.toFixed(0) }}</text>
          <text x="430" y="245" class="ledger-label">轨道最优</text>
          <text x="558" y="245" class="ledger-value good">{{ result.optimalCostPerThousand.toFixed(0) }}</text>
          <text x="430" y="279" class="ledger-foot">AUC {{ auc.toFixed(3) }} · C_FP={{ falsePositiveCost }}</text>
        </svg>
        <figcaption>
          <span class="legend roc" />阈值轨道
          <span class="legend current" />当前阈值
          <span class="legend optimal" />当前代价下最优
          <span class="legend iso" />等成本线
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>概率/分数阈值 <output>{{ threshold.toFixed(2) }}</output></span>
        <input v-model.number="threshold" type="range" min="0" max="1" step="0.01">
      </label>
      <label>
        <span>部署缺陷率 π <output>{{ (deploymentPrevalence * 100).toFixed(1) }}%</output></span>
        <input v-model.number="deploymentPrevalence" type="range" min="0.01" max="0.20" step="0.01">
      </label>
      <label>
        <span>漏放代价 C_FN <output>{{ falseNegativeCost }}</output></span>
        <input v-model.number="falseNegativeCost" type="range" min="20" max="200" step="5">
      </label>
      <p>阈值只沿同一条 ROC 轨道移动；部署缺陷率与代价不会改变这条验证集排序曲线，却会改变等成本线斜率、精确率和每千件绝对告警量。</p>
    </template>
    <template #metrics>
      <div><span>TPR / FPR</span><strong>{{ result.current.tpr.toFixed(2) }} / {{ result.current.fpr.toFixed(2) }}</strong></div>
      <div><span>部署精确率</span><strong>{{ (result.precision * 100).toFixed(1) }}%</strong></div>
      <div><span>当前每千件成本</span><strong>{{ result.expectedCostPerThousand.toFixed(0) }}</strong></div>
      <div><span>成本最优阈值</span><strong>{{ result.optimal.threshold.toFixed(2) }}</strong></div>
    </template>
    <template #note>
      ROC 只描述排序下的 TPR/FPR 交换，不携带部署先验、绝对产量或概率校准。AUC 适合比较全阈值排序，但不能替代左上角目标区、精确率、产能约束和最终运行点验收。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.roc-console{margin:0}.roc-console svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:10px;background:#10182e}.console-bg{fill:#10182e}.grid{stroke:#263554;stroke-width:1}.chance-line{stroke:#566584;stroke-width:1.5;stroke-dasharray:6 5}.roc-line{fill:none;stroke:#72d5ce;stroke-width:4}.iso-cost{stroke:#e15b4b;stroke-width:2;stroke-dasharray:5 4}.current-point{fill:#f4c95d;stroke:#fff;stroke-width:2}.optimal-point{fill:#10182e;stroke:#e15b4b;stroke-width:3}.roc-console text{font:700 10px ui-monospace,monospace}.panel-title,.ledger-title{fill:#dfe5f7}.axis-label,.ledger-label,.ledger-foot{fill:#8f9cbc}.ledger-value{fill:#dfe5f7;text-anchor:end;font-size:13px}.ledger-value.good{fill:#72d5ce}.ledger-value.warn{fill:#f4c95d}.ledger-value.bad{fill:#e15b4b}.ledger-rule{stroke:#40527e}.roc-console figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:14px;height:3px;margin-right:4px;vertical-align:middle}.legend.roc{background:#72d5ce}.legend.current{height:10px;border-radius:50%;background:#f4c95d}.legend.optimal{height:10px;border:2px solid #e15b4b;border-radius:50%}.legend.iso{border-top:2px dashed #e15b4b}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}@media(max-width:560px){.roc-console{overflow-x:auto;overscroll-behavior-inline:contain}.roc-console svg{width:560px;max-width:none}.roc-console figcaption{min-width:560px}}
</style>
