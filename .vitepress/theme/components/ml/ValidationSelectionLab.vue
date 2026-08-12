<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

interface Candidate { index: number; trueRisk: number; validationRisk: number }

const attempts = ref(40)
const validationSize = ref(100)
const seed = ref(11)

function makeRandom(initialSeed: number) {
  let state = initialSeed >>> 0
  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 4294967296
  }
}

function normal(random: () => number) {
  const first = Math.max(random(), 1e-12)
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * random())
}

const candidates = computed<Candidate[]>(() => {
  const random = makeRandom(seed.value * 104729)
  return Array.from({ length: attempts.value }, (_, index) => {
    const trueRisk = .17 + .018 * Math.sin(index * 1.73) + .012 * (random() - .5)
    const standardError = Math.sqrt(trueRisk * (1 - trueRisk) / validationSize.value)
    const validationRisk = Math.max(.02, Math.min(.34, trueRisk + standardError * normal(random)))
    return { index, trueRisk, validationRisk }
  })
})

const selected = computed(() => candidates.value.reduce((best, item) => item.validationRisk < best.validationRisk ? item : best))
const trulyBest = computed(() => candidates.value.reduce((best, item) => item.trueRisk < best.trueRisk ? item : best))
const x = (index: number) => 38 + index / Math.max(1, attempts.value - 1) * 524
const y = (risk: number) => 274 - (risk - .02) / .30 * 234
const percent = (value: number) => `${(value * 100).toFixed(1)}%`
</script>

<template>
  <MlExperimentFrame
    index="03-B"
    title="反复试验怎样把验证集也调过拟合"
    question="候选方案的真实风险相近时，尝试得越多，最低验证误差里会混入多少偶然好运？"
    formula="selected = argmin_j R_valid(h_j)"
  >
    <template #visual>
      <figure class="selection-map">
        <svg viewBox="0 0 600 316" role="img" :aria-label="`${attempts} 个候选中选择验证误差最低者；验证误差 ${percent(selected.validationRisk)}，真实风险 ${percent(selected.trueRisk)}`">
          <rect class="plot-bg" x="20" y="16" width="560" height="278" rx="8" />
          <line class="axis" x1="38" x2="562" y1="274" y2="274" />
          <line class="axis" x1="38" x2="38" y1="40" y2="274" />
          <line v-for="tick in [.05,.10,.15,.20,.25,.30]" :key="tick" class="grid" x1="38" x2="562" :y1="y(tick)" :y2="y(tick)" />
          <g v-for="candidate in candidates" :key="candidate.index">
            <line class="candidate-link" :x1="x(candidate.index)" :x2="x(candidate.index)" :y1="y(candidate.trueRisk)" :y2="y(candidate.validationRisk)" />
            <circle :cx="x(candidate.index)" :cy="y(candidate.trueRisk)" r="2.3" class="true-dot" />
            <circle :cx="x(candidate.index)" :cy="y(candidate.validationRisk)" r="2.3" class="validation-dot" />
          </g>
          <line class="selected-gap" :x1="x(selected.index)" :x2="x(selected.index)" :y1="y(selected.trueRisk)" :y2="y(selected.validationRisk)" />
          <circle :cx="x(selected.index)" :cy="y(selected.validationRisk)" r="7" class="selected-validation" />
          <circle :cx="x(selected.index)" :cy="y(selected.trueRisk)" r="7" class="selected-true" />
          <text x="44" y="36">误差率</text>
          <text x="474" y="306">候选方案序号</text>
          <text :x="Math.min(500, x(selected.index) + 9)" :y="Math.max(52, y(selected.validationRisk) - 8)">被选中</text>
        </svg>
        <figcaption><span class="legend true" />候选真实风险　<span class="legend validation" />一次验证估计　大圆与粗线表示验证集选出的方案</figcaption>
      </figure>
      <table>
        <thead><tr><th>比较对象</th><th>验证估计</th><th>真实风险</th></tr></thead>
        <tbody>
          <tr><td>验证集选中 #{{ selected.index + 1 }}</td><td>{{ percent(selected.validationRisk) }}</td><td>{{ percent(selected.trueRisk) }}</td></tr>
          <tr><td>真实最佳 #{{ trulyBest.index + 1 }}</td><td>{{ percent(trulyBest.validationRisk) }}</td><td>{{ percent(trulyBest.trueRisk) }}</td></tr>
        </tbody>
      </table>
    </template>
    <template #controls>
      <label>
        <span>尝试方案数 <output>{{ attempts }}</output></span>
        <input v-model.number="attempts" type="range" min="5" max="200" step="5">
      </label>
      <label>
        <span>验证样本数 <output>{{ validationSize }}</output></span>
        <input v-model.number="validationSize" type="range" min="25" max="500" step="25">
      </label>
      <button type="button" @click="seed += 1">重新抽取验证结果</button>
      <p>每个候选的真实风险固定在相近范围；验证误差按有限样本标准误加入随机波动。这里用真实风险代表一个无限大的独立审计集。</p>
    </template>
    <template #metrics>
      <div><span>最低验证误差</span><strong>{{ percent(selected.validationRisk) }}</strong></div>
      <div><span>被选方案真实风险</span><strong>{{ percent(selected.trueRisk) }}</strong></div>
      <div><span>选择乐观偏差</span><strong>{{ percent(selected.trueRisk - selected.validationRisk) }}</strong></div>
      <div><span>错过真实最佳</span><strong>{{ percent(selected.trueRisk - trulyBest.trueRisk) }}</strong></div>
    </template>
    <template #note>
      增加尝试次数并反复抽样。即使每个候选的验证误差都是近似无偏估计，取最小值后也会偏向偶然低估者；验证集被反复用于特征、超参数、阈值和停止点选择时，它已经成为训练流程的一部分。最终测试必须在流程冻结后独立使用。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.selection-map{margin:0}.selection-map svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:9px;background:#10182e}.plot-bg{fill:#10182e}.axis{stroke:#53638b}.grid{stroke:#29395f;stroke-width:1}.candidate-link{stroke:#405077;stroke-width:.7}.true-dot{fill:#72d5ce}.validation-dot{fill:#6f7da0}.selected-gap{stroke:#f4c95d;stroke-width:4}.selected-validation{fill:#f06b5b;stroke:#fff;stroke-width:2}.selected-true{fill:#72d5ce;stroke:#fff;stroke-width:2}.selection-map text{fill:#9eabc9;font:10px ui-monospace,monospace}.selection-map figcaption{margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.7}.legend{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:50%;vertical-align:middle}.legend.true{background:#72d5ce}.legend.validation{background:#6f7da0}.ml-experiment__visual table{width:100%;margin:10px 0 0;border-collapse:collapse;color:#c8d1e8;background:#10182e;font-size:10px}.ml-experiment__visual th,.ml-experiment__visual td{padding:7px;border:1px solid #2d3c68;text-align:right}.ml-experiment__visual th:first-child,.ml-experiment__visual td:first-child{text-align:left}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls button{min-height:44px;border:1px solid #6e82bd;border-radius:7px;color:#fff;background:#263d88;font-weight:750;cursor:pointer}.ml-experiment__controls button:hover{background:#3150a3}.ml-experiment__controls button:focus-visible{outline:3px solid #f4c95d;outline-offset:3px}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}
</style>
