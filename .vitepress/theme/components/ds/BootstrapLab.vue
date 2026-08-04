<script setup lang="ts">
import { computed, ref } from 'vue'

const observed = [12, 15, 15, 18, 21, 24]
const repetitions = ref(200)
const seed = ref(9)
const draw = (offset: number) => {
  const value = Math.sin((seed.value + offset * 37) * 8.731) * 9713.42
  return observed[Math.floor((value - Math.floor(value)) * observed.length)]
}
const preview = computed(() => observed.map((_, index) => draw(index)))
const means = computed(() => Array.from({ length: repetitions.value }, (_, repeat) => {
  let total = 0
  for (let index = 0; index < observed.length; index += 1) total += draw(repeat * observed.length + index + 20)
  return total / observed.length
}).sort((a, b) => a - b))
const interval = computed(() => [means.value[Math.floor(means.value.length * .025)], means.value[Math.floor(means.value.length * .975)]])
const bins = computed(() => Array.from({ length: 12 }, (_, bin) => means.value.filter(mean => Math.min(11, Math.floor((mean - 12) / 12 * 12)) === bin).length))
</script>

<template>
  <section class="bootstrap-lab" aria-label="Bootstrap 重抽样实验">
    <header><div><span>INTERACTIVE 19</span><h3>把一次样本当作临时总体，有放回地重抽样</h3></div><button type="button" @click="seed += 11">再抽一次</button></header>
    <div class="bootstrap-lab__samples"><div><small>原样本</small><i v-for="(value,index) in observed" :key="index">{{ value }}</i></div><b>→ 有放回 →</b><div><small>一次 Bootstrap 样本</small><i v-for="(value,index) in preview" :key="index">{{ value }}</i></div></div>
    <label>重复次数 B={{ repetitions }}<input v-model.number="repetitions" type="range" min="50" max="500" step="50"></label>
    <div class="bootstrap-lab__chart"><i v-for="(height,index) in bins" :key="index" :style="{height:`${Math.max(2,height/repetitions*420)}px`}"></i></div>
    <p>均值的 95% 百分位区间：<strong>[{{ interval[0]?.toFixed(2) }}, {{ interval[1]?.toFixed(2) }}]</strong></p>
  </section>
</template>

<style scoped>
.bootstrap-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.bootstrap-lab header{display:flex;justify-content:space-between;gap:16px}.bootstrap-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.bootstrap-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.bootstrap-lab button{align-self:start;padding:8px 11px;border:0;border-radius:7px;color:#17302b;background:#f3c56f;font-weight:800;cursor:pointer}.bootstrap-lab__samples{display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center;margin:22px 0}.bootstrap-lab__samples div{display:flex;flex-wrap:wrap;gap:5px;padding:12px;border-radius:9px;background:#0c2420}.bootstrap-lab__samples small{width:100%;margin-bottom:5px;color:#8eaaa1}.bootstrap-lab__samples i{padding:5px 7px;border-radius:4px;color:#17302b;background:#8ebeb1;font:10px ui-monospace,monospace;font-style:normal}.bootstrap-lab__samples div:last-child i{background:#f3c56f}.bootstrap-lab__samples>b{color:#9cb5ad;font-size:9px}.bootstrap-lab>label{display:block;color:#bdd0ca;font-size:11px}.bootstrap-lab input{width:100%;margin-top:8px;accent-color:#f3c56f}.bootstrap-lab__chart{display:flex;height:120px;gap:4px;align-items:end;margin-top:16px;padding:9px 12px;border-bottom:1px solid #55766d;border-radius:9px;background:#0c2420}.bootstrap-lab__chart i{flex:1;max-width:26px;border-radius:3px 3px 0 0;background:#79aa9e}.bootstrap-lab p{margin:14px 0 0;color:#a8bfb7;font-size:11px}.bootstrap-lab p strong{color:#f3c56f;font:12px ui-monospace,monospace}@media(max-width:680px){.bootstrap-lab header{flex-direction:column}.bootstrap-lab__samples{grid-template-columns:1fr}.bootstrap-lab__samples>b{text-align:center}}
</style>
