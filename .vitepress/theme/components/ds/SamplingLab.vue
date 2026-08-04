<script setup lang="ts">
import { computed, ref } from 'vue'

type Method = 'convenience' | 'srs' | 'stratified'
const method = ref<Method>('convenience')
const sampleSize = ref(20)
const run = ref(1)
const population = Array.from({ length: 100 }, (_, index) => {
  const district = index < 50 ? '中心区' : index < 80 ? '近郊' : '远郊'
  const satisfaction = district === '中心区' ? 82 + (index % 7) : district === '近郊' ? 68 + (index % 9) : 51 + (index % 11)
  return { id: index, district, satisfaction }
})
const populationMean = population.reduce((sum, item) => sum + item.satisfaction, 0) / population.length

function shuffled(values: typeof population, seed: number) {
  return [...values].sort((a, b) => ((a.id * 37 + seed * 17) % 101) - ((b.id * 37 + seed * 17) % 101))
}

const sample = computed(() => {
  const size = sampleSize.value
  if (method.value === 'convenience') return population.slice(0, size)
  if (method.value === 'srs') return shuffled(population, run.value).slice(0, size)

  const centerCount = Math.round(size * .5)
  const suburbanCount = Math.round(size * .3)
  const remoteCount = size - centerCount - suburbanCount
  return [
    ...shuffled(population.filter((item) => item.district === '中心区'), run.value).slice(0, centerCount),
    ...shuffled(population.filter((item) => item.district === '近郊'), run.value + 1).slice(0, suburbanCount),
    ...shuffled(population.filter((item) => item.district === '远郊'), run.value + 2).slice(0, remoteCount)
  ]
})
const selectedIds = computed(() => new Set(sample.value.map((item) => item.id)))
const estimate = computed(() => sample.value.reduce((sum, item) => sum + item.satisfaction, 0) / sample.value.length)
</script>

<template>
  <section class="sampling-lab" aria-label="抽样方法实验">
    <header><div><span>INTERACTIVE 09</span><h3>样本大小解决波动，抽样设计解决偏差</h3></div><button type="button" @click="run += 1">重新抽样</button></header>
    <div class="sampling-lab__controls">
      <label>方法<select v-model="method"><option value="convenience">中心区便利样本</option><option value="srs">简单随机样本</option><option value="stratified">按区域分层样本</option></select></label>
      <label>样本量 {{ sampleSize }}<input v-model.number="sampleSize" type="range" min="10" max="50" step="5"></label>
    </div>
    <div class="sampling-lab__dots"><span v-for="item in population" :key="item.id" :class="[`is-${item.district}`, { 'is-selected': selectedIds.has(item.id) }]" :title="`${item.district}，满意度 ${item.satisfaction}`" /></div>
    <div class="sampling-lab__readout"><div><small>总体真实均值</small><b>{{ populationMean.toFixed(1) }}</b></div><div><small>样本估计</small><b>{{ estimate.toFixed(1) }}</b></div><div><small>绝对误差</small><b>{{ Math.abs(estimate - populationMean).toFixed(1) }}</b></div></div>
  </section>
</template>

<style scoped>
.sampling-lab { margin: 32px 0; padding: 25px; border: 1px solid #d4dfd9; border-radius: 16px; background: #fff; }
.sampling-lab header { display: flex; justify-content: space-between; gap: 18px; align-items: start; }
.sampling-lab header span { color: #af6829; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .12em; }
.sampling-lab h3 { margin: 7px 0 0; border: 0; color: #17372f; font-size: 20px; }
.sampling-lab header button { padding: 8px 11px; border: 0; border-radius: 8px; color: #fff; background: #143b37; font-size: 11px; font-weight: 750; cursor: pointer; }
.sampling-lab__controls { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 22px 0; }
.sampling-lab__controls label { color: #65736e; font-size: 11px; }
.sampling-lab select, .sampling-lab input { display: block; width: 100%; margin-top: 7px; }
.sampling-lab select { min-height: 38px; padding: 6px 8px; border: 1px solid #c6d5ce; border-radius: 7px; background: #f6f8f6; }
.sampling-lab input { accent-color: #2f7c7a; }
.sampling-lab__dots { display: grid; grid-template-columns: repeat(20, 1fr); gap: 4px; padding: 16px; border-radius: 10px; background: #eef3f0; }
.sampling-lab__dots span { aspect-ratio: 1; border-radius: 50%; opacity: .18; }
.sampling-lab__dots .is-中心区 { background: #2f7c7a; }
.sampling-lab__dots .is-近郊 { background: #d39a35; }
.sampling-lab__dots .is-远郊 { background: #d76d50; }
.sampling-lab__dots span.is-selected { opacity: 1; box-shadow: 0 0 0 2px #fff, 0 0 0 3px #263d38; }
.sampling-lab__readout { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 14px; }
.sampling-lab__readout > div { padding: 13px; border-radius: 8px; background: #143b37; }
.sampling-lab__readout small, .sampling-lab__readout b { display: block; }
.sampling-lab__readout small { color: #aec5bd; font-size: 9px; }
.sampling-lab__readout b { margin-top: 5px; color: #f3c56f; font: 780 18px/1.1 ui-monospace, monospace; }
@media (max-width: 640px) { .sampling-lab__controls { grid-template-columns: 1fr; } .sampling-lab__dots { grid-template-columns: repeat(10, 1fr); } }
</style>
