<script setup lang="ts">
import { computed, ref } from 'vue'

const complexity = ref(3)
const bias = computed(() => 8.5 / (complexity.value + .7))
const variance = computed(() => .32 * complexity.value ** 1.55)
const noise = 1.2
const total = computed(() => bias.value ** 2 / 5 + variance.value + noise)
const predictions = computed(() => Array.from({ length: 9 }, (_, index) => 50 + bias.value + Math.sin(index * 5.7 + complexity.value) * variance.value * 1.6))
</script>

<template>
  <section class="bv-lab" aria-label="偏差方差分解实验">
    <header><div><span>INTERACTIVE 18</span><h3>九个平行世界，训练出九个不完全相同的模型</h3></div><strong>复杂度 {{ complexity }}</strong></header>
    <label>模型复杂度<input v-model.number="complexity" type="range" min="1" max="9" step="1"></label>
    <div class="bv-lab__worlds"><span class="truth">真实规律 50</span><i v-for="(prediction,index) in predictions" :key="index" :style="{left:`${Math.min(96,Math.max(4,(prediction-35)/30*100))}%`}"><small>{{ prediction.toFixed(1) }}</small></i></div>
    <div class="bv-lab__parts"><div><small>Bias²</small><b :style="{height:`${Math.min(95,bias**2*2)}%`}"></b><strong>{{ (bias**2/5).toFixed(2) }}</strong></div><div><small>Variance</small><b :style="{height:`${Math.min(95,variance*10)}%`}"></b><strong>{{ variance.toFixed(2) }}</strong></div><div><small>Noise</small><b :style="{height:`${noise*18}%`}"></b><strong>{{ noise.toFixed(2) }}</strong></div><div class="total"><small>期望误差</small><b :style="{height:`${Math.min(95,total*8)}%`}"></b><strong>{{ total.toFixed(2) }}</strong></div></div>
  </section>
</template>

<style scoped>
.bv-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.bv-lab header{display:flex;justify-content:space-between;gap:16px}.bv-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.bv-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.bv-lab header strong{color:#f3c56f;font:700 12px ui-monospace,monospace}.bv-lab>label{display:block;margin:21px 0;color:#bdd0ca;font-size:11px}.bv-lab input{width:100%;margin-top:8px;accent-color:#f3c56f}.bv-lab__worlds{position:relative;height:85px;border-bottom:2px solid #55766d;background:linear-gradient(90deg,transparent 49.7%,#f3c56f 50%,transparent 50.3%)}.bv-lab__worlds .truth{position:absolute;left:50%;top:0;transform:translateX(-50%);color:#f3c56f;font-size:9px}.bv-lab__worlds i{position:absolute;bottom:-7px;width:13px;height:13px;border-radius:50%;transform:translateX(-50%);background:#df765a}.bv-lab__worlds i small{position:absolute;bottom:18px;left:50%;transform:translateX(-50%);color:#a9c0b8;font:8px ui-monospace,monospace}.bv-lab__parts{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;height:150px;margin-top:18px;align-items:end}.bv-lab__parts div{display:grid;grid-template-rows:16px 1fr 18px;height:100%;text-align:center}.bv-lab__parts small{color:#8eaaa1;font-size:9px}.bv-lab__parts b{align-self:end;width:60%;min-height:4px;justify-self:center;border-radius:5px 5px 0 0;background:#79aa9e}.bv-lab__parts .total b{background:#df765a}.bv-lab__parts strong{color:#f3c56f;font:10px ui-monospace,monospace}@media(max-width:620px){.bv-lab header{flex-direction:column}.bv-lab__parts{grid-template-columns:repeat(4,minmax(58px,1fr));overflow-x:auto}}
</style>
