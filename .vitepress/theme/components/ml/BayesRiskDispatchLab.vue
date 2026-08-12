<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type DispatchAction = '放行' | '离线复检' | '隔离'

const defectProbability = ref(0.18)
const missedDefectCost = ref(120)
const reviewCost = ref(8)
const quarantineGoodCost = ref(30)
const quarantineDefectCost = 3

const chart = { left: 52, right: 570, top: 38, bottom: 222 }
const xPlot = (probability: number) => chart.left + probability * (chart.right - chart.left)
const yPlot = (risk: number, maximum: number) => chart.bottom - risk / maximum * (chart.bottom - chart.top)
const probabilityGrid = Array.from({ length: 101 }, (_, index) => index / 100)

const actionFor = (
  probability: number,
  missCost: number,
  inspectCost: number,
  quarantineCost: number,
): DispatchAction => {
  const risks = [
    { action: '放行' as const, risk: probability * missCost },
    { action: '离线复检' as const, risk: inspectCost },
    {
      action: '隔离' as const,
      risk: (1 - probability) * quarantineCost + probability * quarantineDefectCost,
    },
  ]
  return risks.reduce((best, candidate) => candidate.risk < best.risk ? candidate : best).action
}

const result = computed(() => {
  const releaseRisk = defectProbability.value * missedDefectCost.value
  const inspectionRisk = reviewCost.value
  const quarantineRisk = (
    (1 - defectProbability.value) * quarantineGoodCost.value
    + defectProbability.value * quarantineDefectCost
  )
  const risks = [
    { action: '放行' as const, risk: releaseRisk },
    { action: '离线复检' as const, risk: inspectionRisk },
    { action: '隔离' as const, risk: quarantineRisk },
  ]
  const best = risks.reduce((current, candidate) => candidate.risk < current.risk ? candidate : current)
  const maximumRisk = Math.max(
    missedDefectCost.value,
    reviewCost.value,
    quarantineGoodCost.value,
    1,
  ) * 1.05
  const releaseBoundary = reviewCost.value / missedDefectCost.value
  const quarantineBoundary = (
    (quarantineGoodCost.value - reviewCost.value)
    / (quarantineGoodCost.value - quarantineDefectCost)
  )
  return {
    risks,
    best,
    maximumRisk,
    releaseBoundary,
    quarantineBoundary,
    releaseSeries: probabilityGrid.map((probability) => probability * missedDefectCost.value),
    reviewSeries: probabilityGrid.map(() => reviewCost.value),
    quarantineSeries: probabilityGrid.map((probability) =>
      (1 - probability) * quarantineGoodCost.value + probability * quarantineDefectCost
    ),
    actions: probabilityGrid.slice(0, -1).map((probability) =>
      actionFor(
        probability + 0.005,
        missedDefectCost.value,
        reviewCost.value,
        quarantineGoodCost.value,
      )
    ),
  }
})

const linePoints = (values: number[]) => values
  .map((value, index) => `${xPlot(probabilityGrid[index])},${yPlot(value, result.value.maximumRisk)}`)
  .join(' ')
const actionClass = (action: DispatchAction) => ({
  '放行': 'release',
  '离线复检': 'review',
  '隔离': 'quarantine',
}[action])
</script>

<template>
  <MlExperimentFrame
    index="08-A"
    title="在风险继电器上选择放行、复检或隔离"
    question="同一个缺陷后验概率，在漏放代价、复检成本和良品隔离损失变化后，为什么会触发不同动作？"
    formula="a*(x) = argminₐ Σᵧ L(y,a)p(y|x)"
  >
    <template #visual>
      <figure class="risk-relay">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`缺陷概率 ${defectProbability.toFixed(2)}，最小风险动作 ${result.best.action}`">
          <defs>
            <clipPath id="risk-relay-clip"><rect x="52" y="38" width="518" height="184" rx="5" /></clipPath>
          </defs>
          <rect x="16" y="16" width="568" height="294" rx="9" class="relay-bg" />
          <text x="52" y="29" class="panel-title">单件后验期望损失 / 相对成本单位</text>
          <line v-for="tick in [0.25,0.5,0.75,1]" :key="`risk-y-${tick}`" x1="52" x2="570" :y1="yPlot(tick * result.maximumRisk, result.maximumRisk)" :y2="yPlot(tick * result.maximumRisk, result.maximumRisk)" class="grid" />
          <line v-for="tick in [0,.2,.4,.6,.8,1]" :key="`risk-x-${tick}`" :x1="xPlot(tick)" :x2="xPlot(tick)" y1="38" y2="222" class="grid" />
          <g clip-path="url(#risk-relay-clip)">
            <polyline :points="linePoints(result.releaseSeries)" class="risk-line release-line" />
            <polyline :points="linePoints(result.reviewSeries)" class="risk-line review-line" />
            <polyline :points="linePoints(result.quarantineSeries)" class="risk-line quarantine-line" />
          </g>
          <line :x1="xPlot(defectProbability)" :x2="xPlot(defectProbability)" y1="38" y2="250" class="query-line" />
          <circle
            v-for="item in result.risks"
            :key="item.action"
            :cx="xPlot(defectProbability)"
            :cy="yPlot(item.risk, result.maximumRisk)"
            r="6"
            :class="['risk-point', actionClass(item.action)]"
          />
          <rect
            v-for="(action,index) in result.actions"
            :key="`band-${index}`"
            :x="xPlot(index / 100)"
            y="239"
            :width="xPlot((index + 1) / 100) - xPlot(index / 100) + .3"
            height="12"
            :class="['action-band', actionClass(action)]"
          />
          <text x="52" y="271" class="axis-label">缺陷后验概率 p(Y=1|x)</text>
          <text x="403" y="271" class="boundary-label">放行线 {{ result.releaseBoundary.toFixed(2) }} / 隔离线 {{ result.quarantineBoundary.toFixed(2) }}</text>
          <text x="52" y="297" :class="['decision-label', actionClass(result.best.action)]">当前触点 → {{ result.best.action }}</text>
        </svg>
        <figcaption>
          <span class="legend release" />放行风险
          <span class="legend review" />离线复检风险
          <span class="legend quarantine" />隔离风险
          <span class="legend query" />当前后验
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>缺陷后验概率 <output>{{ defectProbability.toFixed(2) }}</output></span>
        <input v-model.number="defectProbability" type="range" min="0" max="1" step="0.01">
      </label>
      <label>
        <span>缺陷件误放代价 <output>{{ missedDefectCost }}</output></span>
        <input v-model.number="missedDefectCost" type="range" min="60" max="200" step="5">
      </label>
      <label>
        <span>离线复检固定成本 <output>{{ reviewCost }}</output></span>
        <input v-model.number="reviewCost" type="range" min="4" max="15" step="1">
      </label>
      <label>
        <span>良品隔离损失 <output>{{ quarantineGoodCost }}</output></span>
        <input v-model.number="quarantineGoodCost" type="range" min="20" max="50" step="1">
      </label>
      <p>三条风险线的最低者才是动作。降低误放代价会扩大“放行”区；降低复检成本会扩大中间复检区；提高良品隔离损失会把隔离触点推向更高后验。</p>
    </template>
    <template #metrics>
      <div><span>放行风险</span><strong>{{ result.risks[0].risk.toFixed(2) }}</strong></div>
      <div><span>复检风险</span><strong>{{ result.risks[1].risk.toFixed(2) }}</strong></div>
      <div><span>隔离风险</span><strong>{{ result.risks[2].risk.toFixed(2) }}</strong></div>
      <div><span>最小风险动作</span><strong>{{ result.best.action }}</strong></div>
    </template>
    <template #note>
      教学成本是相对量纲，不是财务报价。真实项目必须由质量、安全、生产和售后共同定义损失，并把产能约束另写成约束条件；不能为了让模型“看起来更准”反向编造代价。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.risk-relay{margin:0}.risk-relay svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:10px;background:#10182e}.relay-bg{fill:#10182e}.grid{stroke:#263554;stroke-width:1}.risk-line{fill:none;stroke-width:3}.release-line{stroke:#72d5ce}.review-line{stroke:#f4c95d}.quarantine-line{stroke:#e15b4b}.query-line{stroke:#f4c95d;stroke-width:2;stroke-dasharray:5 4}.risk-point{stroke:#fff;stroke-width:1.8}.risk-point.release,.action-band.release{fill:#72d5ce}.risk-point.review,.action-band.review{fill:#f4c95d}.risk-point.quarantine,.action-band.quarantine{fill:#e15b4b}.risk-relay text{font:700 10px ui-monospace,monospace}.panel-title{fill:#dfe5f7}.axis-label,.boundary-label{fill:#8f9cbc}.boundary-label{text-anchor:end}.decision-label{font-size:15px}.decision-label.release{fill:#72d5ce}.decision-label.review{fill:#f4c95d}.decision-label.quarantine{fill:#e15b4b}.risk-relay figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:14px;height:3px;margin-right:4px;vertical-align:middle}.legend.release{background:#72d5ce}.legend.review{background:#f4c95d}.legend.quarantine{background:#e15b4b}.legend.query{border-top:2px dashed #f4c95d}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}@media(max-width:560px){.risk-relay{overflow-x:auto;overscroll-behavior-inline:contain}.risk-relay svg{width:560px;max-width:none}.risk-relay figcaption{min-width:560px}}
</style>
