<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

interface Observation { score: number; label: number }
interface Candidate { threshold: number; empirical: number; population: number }

const thresholds = [.25, .4, .55, .7]
const sampleSize = ref(16)
const seed = ref(7)

function makeSample(size: number, initialSeed: number): Observation[] {
  let state = initialSeed >>> 0
  const random = () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 4294967296
  }
  return Array.from({ length: size }, () => {
    const score = random()
    const failureProbability = .08 + .82 * score
    return { score, label: random() < failureProbability ? 1 : 0 }
  })
}

function risk(data: Observation[], threshold: number) {
  return data.reduce((sum, item) => sum + (Number(item.score >= threshold) !== item.label ? 1 : 0), 0) / data.length
}

const population = makeSample(12000, 20260805)
const candidates = computed<Candidate[]>(() => {
  const sample = makeSample(sampleSize.value, seed.value)
  return thresholds.map((threshold) => ({
    threshold,
    empirical: risk(sample, threshold),
    population: risk(population, threshold)
  }))
})
const selected = computed(() => candidates.value.reduce((best, item) => item.empirical < best.empirical ? item : best))
const populationBest = computed(() => candidates.value.reduce((best, item) => item.population < best.population ? item : best))
const percent = (value: number) => `${(value * 100).toFixed(1)}%`
</script>

<template>
  <MlExperimentFrame
    index="01-B"
    title="ERM 会不会选错候选模型"
    question="只观察有限训练样本时，经验风险最小的阈值一定拥有最低总体风险吗？"
    formula="h_hat = argmin_h R_hat_n(h)"
  >
    <template #visual>
      <div class="risk-bars" role="img" :aria-label="`四个候选阈值的经验风险与总体风险对比；ERM 当前选择 ${selected.threshold.toFixed(2)}`">
        <div v-for="candidate in candidates" :key="candidate.threshold" class="risk-bars__group" :class="{ selected: candidate.threshold === selected.threshold }">
          <strong>h{{ candidate.threshold.toFixed(2) }}</strong>
          <div class="risk-bars__track"><span class="empirical" :style="{ height: `${candidate.empirical * 260}px` }" /></div>
          <div class="risk-bars__track"><span class="population" :style="{ height: `${candidate.population * 260}px` }" /></div>
          <small>{{ percent(candidate.empirical) }}<br>{{ percent(candidate.population) }}</small>
        </div>
      </div>
      <p class="risk-bars__legend"><span class="empirical" />经验风险　<span class="population" />总体风险；边框表示 ERM 当前选择</p>
      <table>
        <thead><tr><th>阈值</th><th>经验风险</th><th>总体风险</th></tr></thead>
        <tbody><tr v-for="candidate in candidates" :key="candidate.threshold"><td>{{ candidate.threshold.toFixed(2) }}</td><td>{{ percent(candidate.empirical) }}</td><td>{{ percent(candidate.population) }}</td></tr></tbody>
      </table>
    </template>
    <template #controls>
      <label>
        <span>训练样本量 <output>{{ sampleSize }}</output></span>
        <input v-model.number="sampleSize" type="range" min="8" max="128" step="8">
      </label>
      <button type="button" @click="seed += 1">重新抽取训练集</button>
      <p>总体风险用固定的 12,000 个模拟设备近似，只用于揭示有限样本选择误差；真实项目无法直接看到它。</p>
    </template>
    <template #metrics>
      <div><span>ERM 选择</span><strong>h{{ selected.threshold.toFixed(2) }}</strong></div>
      <div><span>最低总体风险候选</span><strong>h{{ populationBest.threshold.toFixed(2) }}</strong></div>
      <div><span>训练风险</span><strong>{{ percent(selected.empirical) }}</strong></div>
      <div><span>泛化间隙</span><strong>{{ percent(selected.population - selected.empirical) }}</strong></div>
    </template>
    <template #note>
      调小样本量并反复抽样，观察 ERM 的选择怎样变化。经验风险是总体风险的估计，不是总体风险本身；模型又是看过同一份数据后选出的，所以选择过程会放大偶然波动。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.risk-bars{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;min-height:230px;padding:18px 14px 8px;border:1px solid #2d3c68;border-radius:9px;background:#10182e}.risk-bars__group{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto 142px auto;gap:5px;padding:8px;border:2px solid transparent;border-radius:7px}.risk-bars__group.selected{border-color:#f4c95d}.risk-bars__group strong{grid-column:1/-1;color:#dce4fa;text-align:center;font:750 10px ui-monospace,monospace}.risk-bars__track{position:relative;display:flex;align-items:end;justify-content:center;border-bottom:1px solid #526184}.risk-bars__track span{display:block;width:72%;max-height:138px}.risk-bars__track .empirical{background:#f06b5b}.risk-bars__track .population{background:#72d5ce}.risk-bars__group small{grid-column:1/-1;color:#93a0bf;text-align:center;font:9px/1.5 ui-monospace,monospace}.risk-bars__legend{margin:8px 0;color:#9eabc9;font-size:10px}.risk-bars__legend span{display:inline-block;width:10px;height:10px;margin-right:4px}.risk-bars__legend .empirical{background:#f06b5b}.risk-bars__legend .population{background:#72d5ce}.ml-experiment__visual table{width:100%;margin:10px 0 0;border-collapse:collapse;color:#c8d1e8;background:#10182e;font-size:10px}.ml-experiment__visual th,.ml-experiment__visual td{padding:7px;border:1px solid #2d3c68;text-align:right}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls button{min-height:44px;border:1px solid #6e82bd;border-radius:7px;color:#fff;background:#263d88;font-weight:750;cursor:pointer}.ml-experiment__controls button:hover{background:#3150a3}.ml-experiment__controls button:focus-visible{outline:3px solid #f4c95d;outline-offset:3px}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}@media(max-width:520px){.risk-bars{gap:4px;padding-inline:4px}.risk-bars__group{padding:4px}}
</style>
