<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const pressure = [0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10]
const residualPattern = [0.03, -0.04, 0.02, -0.01, 0.05, -0.03, 0.01, -0.05, 0.02]
const response = pressure.map((value, index) => 4.02 + 1.598 * value + residualPattern[index])
const meanPressure = pressure.reduce((sum, value) => sum + value, 0) / pressure.length
const meanResponse = response.reduce((sum, value) => sum + value, 0) / response.length
const centeredSum = pressure.reduce((sum, value) => sum + (value - meanPressure) ** 2, 0)
const mleSlope = pressure.reduce((sum, value, index) => sum + (value - meanPressure) * (response[index] - meanResponse), 0) / centeredSum
const mleIntercept = meanResponse - mleSlope * meanPressure
const minimumRss = response.reduce((sum, value, index) => sum + (value - mleIntercept - mleSlope * pressure[index]) ** 2, 0)

const candidateIntercept = ref(4.10)
const candidateSlope = ref(1.575)
const assumedSigma = ref(0.08)

const fit = computed(() => {
  const residuals = response.map((value, index) => value - candidateIntercept.value - candidateSlope.value * pressure[index])
  const rss = residuals.reduce((sum, value) => sum + value ** 2, 0)
  const deltaNll = Math.max(0, (rss - minimumRss) / (2 * assumedSigma.value ** 2))
  return {
    residuals,
    rss,
    deltaNll,
    relativeLikelihood: Math.exp(-Math.min(deltaNll, 60))
  }
})

const plot = { left: 38, right: 365, top: 28, bottom: 286 }
const map = { left: 414, right: 574, top: 48, bottom: 272 }
const xPlot = (value: number) => plot.left + value / 10 * (plot.right - plot.left)
const yPlot = (value: number) => plot.bottom - (value - 3.6) / 16.9 * (plot.bottom - plot.top)
const xMap = (value: number) => map.left + (value - 1.55) / 0.10 * (map.right - map.left)
const yMap = (value: number) => map.bottom - (value - 3.84) / 0.34 * (map.bottom - map.top)
const contourScale = computed(() => assumedSigma.value / 0.08)
const formatLikelihood = (value: number) => value < 0.0001 ? value.toExponential(2) : value.toFixed(4)
</script>

<template>
  <MlExperimentFrame
    index="05-A"
    title="在校准证书上寻找最大似然点"
    question="同时改变零点、灵敏度和假设噪声后，哪一项移动最优位置，哪一项只改变似然山峰的尖锐程度？"
    formula="ΔNLL = [RSS(β) − RSS(β̂)] / (2σ²)"
  >
    <template #visual>
      <figure class="likelihood-certificate">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`候选零点 ${candidateIntercept.toFixed(3)}，候选灵敏度 ${candidateSlope.toFixed(3)}，相对似然 ${formatLikelihood(fit.relativeLikelihood)}`">
          <rect x="16" y="16" width="568" height="294" rx="9" class="plot-bg" />

          <text x="38" y="22" class="panel-title">校准读数 / mA</text>
          <line v-for="tick in [4,8,12,16,20]" :key="`y-${tick}`" :x1="plot.left" :x2="plot.right" :y1="yPlot(tick)" :y2="yPlot(tick)" class="grid" />
          <line class="axis" :x1="plot.left" :x2="plot.right" :y1="plot.bottom" :y2="plot.bottom" />
          <line class="axis" :x1="plot.left" :x2="plot.left" :y1="plot.top" :y2="plot.bottom" />
          <text v-for="tick in [0,2,4,6,8,10]" :key="`x-${tick}`" :x="xPlot(tick) - 5" y="303" class="tick">{{ tick }}</text>
          <text x="260" y="317" class="axis-label">参考压力 / bar</text>

          <line :x1="xPlot(0)" :y1="yPlot(mleIntercept)" :x2="xPlot(10)" :y2="yPlot(mleIntercept + 10 * mleSlope)" class="mle-line" />
          <line :x1="xPlot(0)" :y1="yPlot(candidateIntercept)" :x2="xPlot(10)" :y2="yPlot(candidateIntercept + 10 * candidateSlope)" class="candidate-line" />
          <line v-for="(value,index) in response" :key="`res-${index}`" :x1="xPlot(pressure[index])" :x2="xPlot(pressure[index])" :y1="yPlot(value)" :y2="yPlot(candidateIntercept + candidateSlope * pressure[index])" class="residual-stem" />
          <circle v-for="(value,index) in response" :key="`point-${index}`" :cx="xPlot(pressure[index])" :cy="yPlot(value)" r="4.5" class="reading" />

          <line x1="389" x2="389" y1="28" y2="288" class="divider" />
          <text x="414" y="22" class="panel-title">参数似然地形</text>
          <text x="454" y="304" class="axis-label">灵敏度 β₁</text>
          <text x="397" y="44" class="axis-label">零点 β₀</text>
          <line v-for="tick in [1.56,1.58,1.60,1.62,1.64]" :key="`mx-${tick}`" :x1="xMap(tick)" :x2="xMap(tick)" :y1="map.top" :y2="map.bottom" class="map-grid" />
          <line v-for="tick in [3.9,4.0,4.1]" :key="`my-${tick}`" :x1="map.left" :x2="map.right" :y1="yMap(tick)" :y2="yMap(tick)" class="map-grid" />
          <g :transform="`translate(${xMap(mleSlope)} ${yMap(mleIntercept)}) rotate(48)`">
            <ellipse :rx="Math.min(112, 70 * contourScale)" :ry="Math.min(27, 16 * contourScale)" class="contour outer" />
            <ellipse :rx="Math.min(75, 44 * contourScale)" :ry="Math.min(18, 10 * contourScale)" class="contour middle" />
            <ellipse :rx="Math.min(39, 22 * contourScale)" :ry="Math.min(10, 5 * contourScale)" class="contour inner" />
          </g>
          <line :x1="xMap(mleSlope)-7" :x2="xMap(mleSlope)+7" :y1="yMap(mleIntercept)" :y2="yMap(mleIntercept)" class="mle-cross" />
          <line :x1="xMap(mleSlope)" :x2="xMap(mleSlope)" :y1="yMap(mleIntercept)-7" :y2="yMap(mleIntercept)+7" class="mle-cross" />
          <circle :cx="xMap(candidateSlope)" :cy="yMap(candidateIntercept)" r="7" class="candidate-dot" />
          <text :x="Math.min(535, xMap(candidateSlope) + 10)" :y="Math.max(56, yMap(candidateIntercept) - 9)" class="candidate-label">候选</text>
        </svg>
        <figcaption>
          <span class="legend reading" />稳定读数
          <span class="legend candidate" />候选校准线/参数
          <span class="legend mle" />MLE 校准线/十字
          <span class="legend residual" />候选残差
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>候选零点 β₀ <output>{{ candidateIntercept.toFixed(3) }}</output></span>
        <input v-model.number="candidateIntercept" type="range" min="3.84" max="4.18" step="0.005">
      </label>
      <label>
        <span>候选灵敏度 β₁ <output>{{ candidateSlope.toFixed(3) }}</output></span>
        <input v-model.number="candidateSlope" type="range" min="1.55" max="1.65" step="0.001">
      </label>
      <label>
        <span>假设噪声 σ <output>{{ assumedSigma.toFixed(3) }}</output></span>
        <input v-model.number="assumedSigma" type="range" min="0.03" max="0.15" step="0.005">
      </label>
      <p>把候选圆点移向十字，残差茎与 RSS 同时缩短。只改变 σ：十字和最优校准线不移动，但等高线宽度及错误参数的相对似然会改变。</p>
    </template>
    <template #metrics>
      <div><span>候选 RSS</span><strong>{{ fit.rss.toFixed(4) }}</strong></div>
      <div><span>相对 NLL 差</span><strong>{{ fit.deltaNll.toFixed(3) }}</strong></div>
      <div><span>相对似然</span><strong>{{ formatLikelihood(fit.relativeLikelihood) }}</strong></div>
      <div><span>MLE（β₀, β₁）</span><strong>{{ mleIntercept.toFixed(3) }}, {{ mleSlope.toFixed(3) }}</strong></div>
    </template>
    <template #note>
      这是固定设计、同方差高斯模型的条件似然。最大似然点与最小 RSS 点相同；σ 控制“离开最优点要付出多大概率代价”。若每个压力点的 σ 不同，目标会变成按 1/σᵢ² 加权的残差平方和。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.likelihood-certificate{margin:0}.likelihood-certificate svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:10px;background:#10182e}.plot-bg{fill:#10182e}.grid,.map-grid{stroke:#273657;stroke-width:1}.axis{stroke:#65749a}.divider{stroke:#344675}.mle-line{stroke:#72d5ce;stroke-width:3;stroke-dasharray:7 5}.candidate-line{stroke:#f4c95d;stroke-width:4}.residual-stem{stroke:#f06b5b;stroke-width:2;opacity:.72}.reading{fill:#eef2ff;stroke:#72d5ce;stroke-width:2}.contour{fill:none;stroke:#536a9f}.contour.middle{stroke:#7488bd}.contour.inner{stroke:#9aabda}.mle-cross{stroke:#72d5ce;stroke-width:3}.candidate-dot{fill:#f4c95d;stroke:#fff;stroke-width:2}.likelihood-certificate text{font:700 10px ui-monospace,monospace}.panel-title{fill:#dfe5f7}.tick,.axis-label{fill:#8f9cbc}.candidate-label{fill:#f4c95d}.likelihood-certificate figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}.legend.reading{height:10px;border:2px solid #72d5ce;border-radius:50%;background:#eef2ff}.legend.candidate{background:#f4c95d}.legend.mle{border-top:2px dashed #72d5ce}.legend.residual{background:#f06b5b}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}
</style>
