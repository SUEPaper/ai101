<script setup lang="ts">
import { computed, ref } from 'vue'

const binCount = ref(6)
const includeOutlier = ref(false)
const baseValues = [42, 48, 51, 52, 55, 57, 61, 62, 63, 65, 68, 69, 71, 72, 74, 78, 82, 88]
const values = computed(() => includeOutlier.value ? [...baseValues, 150] : baseValues)
const bins = computed(() => {
  const minimum = Math.min(...values.value)
  const maximum = Math.max(...values.value)
  const width = (maximum - minimum || 1) / binCount.value
  const result = Array.from({ length: binCount.value }, (_, index) => ({
    start: minimum + index * width,
    end: minimum + (index + 1) * width,
    count: 0
  }))
  values.value.forEach((value) => {
    const index = Math.min(Math.floor((value - minimum) / width), binCount.value - 1)
    result[index].count += 1
  })
  return result
})
const maxCount = computed(() => Math.max(...bins.value.map((bin) => bin.count), 1))
</script>

<template>
  <section class="histogram-lab" aria-label="直方图分箱实验">
    <header><div><span>INTERACTIVE 07</span><h3>同一批数据，不同分箱会讲出不同故事</h3></div><output>{{ binCount }} bins</output></header>
    <div class="histogram-lab__controls"><label>箱数<input v-model.number="binCount" type="range" min="3" max="12"></label><label class="histogram-lab__toggle"><input v-model="includeOutlier" type="checkbox">加入一个 150 的异常值</label></div>
    <div class="histogram-lab__plot">
      <div v-for="(bin, index) in bins" :key="index" class="histogram-lab__bin"><span :style="{ height: `${(bin.count / maxCount) * 150}px` }"><b>{{ bin.count }}</b></span><small>{{ Math.round(bin.start) }}–{{ Math.round(bin.end) }}</small></div>
    </div>
    <p>直方图的箱宽是分析选择，不是数据本身。检查结论是否在合理分箱范围内保持稳定。</p>
  </section>
</template>

<style scoped>
.histogram-lab { margin: 32px 0; padding: 25px; border: 1px solid #d4dfd9; border-radius: 16px; background: #fff; }
.histogram-lab header { display: flex; justify-content: space-between; gap: 20px; }
.histogram-lab header span { color: #ad6729; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .12em; }
.histogram-lab h3 { margin: 7px 0 0; border: 0; color: #17372f; font-size: 20px; }
.histogram-lab output { color: #28675e; font: 800 12px/1 ui-monospace, monospace; }
.histogram-lab__controls { display: grid; grid-template-columns: 1fr auto; align-items: end; gap: 20px; margin: 22px 0 18px; color: #5f6f69; font-size: 11px; }
.histogram-lab__controls input[type='range'] { display: block; width: 100%; margin-top: 7px; accent-color: #2f7c7a; }
.histogram-lab__toggle { display: flex; align-items: center; gap: 7px; }
.histogram-lab__toggle input { accent-color: #d17247; }
.histogram-lab__plot { display: flex; align-items: end; gap: 3px; min-height: 194px; padding: 12px 12px 0; border-left: 1px solid #9eb0a8; border-bottom: 1px solid #9eb0a8; background: #f5f8f6; overflow-x: auto; }
.histogram-lab__bin { display: grid; grid-template-rows: 155px 27px; align-items: end; min-width: 36px; flex: 1; }
.histogram-lab__bin > span { position: relative; display: block; min-height: 2px; background: #2f7c7a; transition: height .2s ease; }
.histogram-lab__bin b { position: absolute; top: -18px; width: 100%; color: #36534c; font: 750 9px/1 ui-monospace, monospace; text-align: center; }
.histogram-lab__bin small { padding-top: 6px; color: #687771; font: 650 8px/1 ui-monospace, monospace; text-align: center; }
.histogram-lab > p { margin: 15px 0 0; color: #5a6964; font-size: 12px; line-height: 1.6; }
@media (max-width: 620px) { .histogram-lab__controls { grid-template-columns: 1fr; } }
</style>
