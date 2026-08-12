<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

interface PumpEvent { id: string; score: number; failed: boolean }

const events: PumpEvent[] = [
  { id: 'P01', score: .08, failed: false }, { id: 'P02', score: .12, failed: false },
  { id: 'P03', score: .18, failed: true }, { id: 'P04', score: .23, failed: false },
  { id: 'P05', score: .29, failed: false }, { id: 'P06', score: .34, failed: true },
  { id: 'P07', score: .39, failed: false }, { id: 'P08', score: .44, failed: true },
  { id: 'P09', score: .48, failed: false }, { id: 'P10', score: .53, failed: true },
  { id: 'P11', score: .57, failed: false }, { id: 'P12', score: .61, failed: true },
  { id: 'P13', score: .66, failed: true }, { id: 'P14', score: .71, failed: false },
  { id: 'P15', score: .74, failed: true }, { id: 'P16', score: .79, failed: true },
  { id: 'P17', score: .83, failed: false }, { id: 'P18', score: .87, failed: true },
  { id: 'P19', score: .92, failed: true }, { id: 'P20', score: .96, failed: true }
]

const threshold = ref(.5)
const falseNegativeCost = ref(8)
const falsePositiveCost = 1
const x = (score: number) => 28 + score * 544

const result = computed(() => {
  let falseNegatives = 0
  let falsePositives = 0
  let inspections = 0
  for (const event of events) {
    const inspect = event.score >= threshold.value
    if (inspect) inspections += 1
    if (event.failed && !inspect) falseNegatives += 1
    if (!event.failed && inspect) falsePositives += 1
  }
  return {
    falseNegatives,
    falsePositives,
    inspections,
    cost: falseNegatives * falseNegativeCost.value + falsePositives * falsePositiveCost,
    bayesThreshold: falsePositiveCost / (falseNegativeCost.value + falsePositiveCost)
  }
})
</script>

<template>
  <MlExperimentFrame
    index="01-A"
    title="故障代价如何改变检修阈值"
    question="同一组故障概率，在漏检代价升高后，0.5 还是合理的行动阈值吗？"
    formula="a*(x)=1[p(x) > C_FP / (C_FN + C_FP)]"
  >
    <template #visual>
      <figure class="risk-map">
        <svg viewBox="0 0 600 190" role="img" :aria-label="`20 台泵的故障分数；当前阈值 ${threshold.toFixed(2)}，漏检 ${result.falseNegatives} 台，误检 ${result.falsePositives} 台`">
          <line class="axis" x1="28" x2="572" y1="150" y2="150" />
          <line class="threshold" :x1="x(threshold)" :x2="x(threshold)" y1="22" y2="155" />
          <text :x="Math.min(520, x(threshold) + 7)" y="20">阈值 {{ threshold.toFixed(2) }}</text>
          <text x="28" y="178">0</text><text x="558" y="178">1</text>
          <text x="28" y="38">24h 内故障</text><text x="28" y="104">正常运行</text>
          <g v-for="(event,index) in events" :key="event.id">
            <circle
              :cx="x(event.score)"
              :cy="event.failed ? 56 + (index % 2) * 16 : 116 + (index % 2) * 16"
              r="7"
              :class="[
                event.failed ? 'failed' : 'healthy',
                event.failed && event.score < threshold ? 'missed' : '',
                !event.failed && event.score >= threshold ? 'false-alarm' : ''
              ]"
            />
          </g>
        </svg>
        <figcaption><span class="legend failed" />真实故障　<span class="legend healthy" />真实正常　描边表示当前决策造成的错误</figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>检修阈值 <output>{{ threshold.toFixed(2) }}</output></span>
        <input v-model.number="threshold" type="range" min=".05" max=".95" step=".05">
      </label>
      <label>
        <span>漏检代价 <output>{{ falseNegativeCost }}×</output></span>
        <input v-model.number="falseNegativeCost" type="range" min="1" max="20" step="1">
      </label>
      <p>误检代价固定为 1 个单位。这里的 20 台泵是为教学构造的固定样本，不代表概率已经校准。</p>
    </template>
    <template #metrics>
      <div><span>漏检故障泵</span><strong>{{ result.falseNegatives }}</strong></div>
      <div><span>误触发检修</span><strong>{{ result.falsePositives }}</strong></div>
      <div><span>样本总代价</span><strong>{{ result.cost }}</strong></div>
      <div><span>校准概率的理论阈值</span><strong>{{ result.bayesThreshold.toFixed(3) }}</strong></div>
    </template>
    <template #note>
      理论阈值来自逐台设备的条件风险比较；样本上使总代价最小的阈值可能不同。前者依赖概率校准和代价假设，后者还受有限样本波动影响。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.risk-map{margin:0}.risk-map svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:9px;background:#10182e}.axis{stroke:#7180a9}.threshold{stroke:#f4c95d;stroke-width:2;stroke-dasharray:5 4}.risk-map text{fill:#9eabc9;font:10px ui-monospace,monospace}.risk-map circle{stroke-width:3}.risk-map circle.failed{fill:#f06b5b;stroke:#f06b5b}.risk-map circle.healthy{fill:#72d5ce;stroke:#72d5ce}.risk-map circle.missed,.risk-map circle.false-alarm{stroke:#fff}.risk-map figcaption{margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:9px;height:9px;margin-right:4px;border-radius:50%}.legend.failed{background:#f06b5b}.legend.healthy{background:#72d5ce}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}
</style>
