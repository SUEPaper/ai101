<script setup lang="ts">
import { computed, ref } from 'vue'

const probability = ref(.35)
const sampleSize = ref(5)
const seed = ref(17)
const random = (index: number) => {
  const value = Math.sin((seed.value + index * 91) * 12.9898) * 43758.5453
  return value - Math.floor(value)
}
const means = computed(() => Array.from({ length: 60 }, (_, repeat) => {
  let successes = 0
  for (let index = 0; index < sampleSize.value; index += 1) if (random(repeat * 101 + index) < probability.value) successes += 1
  return successes / sampleSize.value
}))
const bins = computed(() => Array.from({ length: 11 }, (_, bin) => means.value.filter(mean => Math.min(10, Math.round(mean * 10)) === bin).length))
const average = computed(() => means.value.reduce((sum, value) => sum + value, 0) / means.value.length)
</script>

<template>
  <section class="rv-lab" aria-label="随机变量与中心极限定理实验">
    <header><div><span>INTERACTIVE 17</span><h3>重复 60 次抽样，观察样本均值如何聚拢</h3></div><button type="button" @click="seed += 13">重新模拟</button></header>
    <div class="rv-lab__controls"><label>成功概率 p={{ probability.toFixed(2) }}<input v-model.number="probability" type="range" min="0.1" max="0.9" step="0.05"></label><label>每次样本量 n={{ sampleSize }}<input v-model.number="sampleSize" type="range" min="1" max="50" step="1"></label></div>
    <div class="rv-lab__chart"><i v-for="(height,index) in bins" :key="index" :style="{height:`${height*7+2}px`}"><small>{{ (index/10).toFixed(1) }}</small></i></div>
    <div class="rv-lab__stats"><span>理论期望 <b>{{ probability.toFixed(2) }}</b></span><span>60 次均值的平均 <b>{{ average.toFixed(3) }}</b></span><span>单次均值标准差约 <b>{{ Math.sqrt(probability*(1-probability)/sampleSize).toFixed(3) }}</b></span></div>
  </section>
</template>

<style scoped>
.rv-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.rv-lab header{display:flex;justify-content:space-between;gap:16px}.rv-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.rv-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.rv-lab button{align-self:start;padding:8px 11px;border:0;border-radius:7px;color:#17302b;background:#f3c56f;font-weight:800;cursor:pointer}.rv-lab__controls{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:21px 0}.rv-lab label{color:#bdd0ca;font-size:11px}.rv-lab input{width:100%;margin-top:8px;accent-color:#f3c56f}.rv-lab__chart{display:flex;height:140px;align-items:end;justify-content:space-around;gap:5px;padding:12px 14px 26px;border-bottom:1px solid #527269;border-radius:10px;background:#0c2420}.rv-lab__chart i{position:relative;flex:1;max-width:27px;min-height:2px;border-radius:3px 3px 0 0;background:#79aa9e}.rv-lab__chart small{position:absolute;bottom:-19px;left:50%;transform:translateX(-50%);color:#86a49a;font:8px ui-monospace,monospace}.rv-lab__stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:15px}.rv-lab__stats span{padding:10px;border-radius:7px;color:#92ada4;background:#0c2420;font-size:9px}.rv-lab__stats b{display:block;margin-top:4px;color:#f3c56f;font:13px ui-monospace,monospace}@media(max-width:650px){.rv-lab header{flex-direction:column}.rv-lab__controls,.rv-lab__stats{grid-template-columns:1fr}}
</style>
