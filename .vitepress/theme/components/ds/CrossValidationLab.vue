<script setup lang="ts">
import { computed, ref } from 'vue'

const folds = ref(5)
const strength = ref(1.5)
const samples = Array.from({ length: 20 }, (_, index) => index)
const coefficients = [5.8, -4.4, 3.2, 2.5, -1.9, 1.2]
const shrunk = computed(() => coefficients.map(value => value / (1 + strength.value * .32)))
const validation = computed(() => 3.2 + (strength.value - 2.7) ** 2 * .16)
</script>

<template>
  <section class="cv-lab" aria-label="交叉验证与正则化实验">
    <header><div><span>INTERACTIVE 16</span><h3>轮换验证集，再观察惩罚如何收缩系数</h3></div><strong>{{ folds }}-fold · λ={{ strength.toFixed(1) }}</strong></header>
    <div class="cv-lab__controls"><label>K 折数量<input v-model.number="folds" type="range" min="2" max="6" step="1"></label><label>正则强度 λ<input v-model.number="strength" type="range" min="0" max="8" step="0.5"></label></div>
    <div class="cv-lab__folds"><div v-for="fold in folds" :key="fold"><small>轮次 {{ fold }}</small><i v-for="sample in samples" :key="sample" :class="{ validation: sample % folds === fold-1 }"></i></div></div>
    <div class="cv-lab__bottom">
      <div class="coefficients"><span v-for="(value,index) in shrunk" :key="index"><small>β{{ index+1 }}</small><b :style="{width:`${Math.abs(value)*13}%`}" :class="{negative:value<0}"></b><em>{{ value.toFixed(2) }}</em></span></div>
      <div class="score"><small>估计验证误差</small><strong>{{ validation.toFixed(2) }}</strong><p>{{ strength < 1 ? '约束较弱：系数大，方差风险更高。' : strength > 5 ? '约束较强：模型可能欠拟合。' : '当前处在偏差与方差的折中区域。' }}</p></div>
    </div>
  </section>
</template>

<style scoped>
.cv-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.cv-lab header{display:flex;justify-content:space-between;gap:16px}.cv-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.cv-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.cv-lab header strong{color:#f3c56f;font:700 12px ui-monospace,monospace}.cv-lab__controls{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:21px 0}.cv-lab label{color:#bbd0c9;font-size:11px}.cv-lab input{width:100%;margin-top:8px;accent-color:#f3c56f}.cv-lab__folds{display:grid;gap:7px}.cv-lab__folds div{display:grid;grid-template-columns:55px repeat(20,1fr);gap:3px;align-items:center}.cv-lab__folds small{color:#8eaaa1;font-size:8px}.cv-lab__folds i{height:10px;border-radius:2px;background:#497168}.cv-lab__folds i.validation{background:#f3c56f}.cv-lab__bottom{display:grid;grid-template-columns:1.3fr .7fr;gap:16px;margin-top:22px}.coefficients,.score{padding:15px;border-radius:9px;background:#0c2420}.coefficients span{display:grid;grid-template-columns:24px 1fr 34px;gap:7px;align-items:center;margin:7px 0}.coefficients small{color:#8eaaa1}.coefficients b{height:7px;border-radius:3px;background:#79aa9e}.coefficients b.negative{background:#df765a}.coefficients em{color:#dce9e4;font:9px ui-monospace,monospace;font-style:normal}.score small{color:#8eaaa1}.score strong{display:block;margin:5px 0;color:#f3c56f;font-size:28px}.score p{margin:0;color:#a8bfb7;font-size:10px;line-height:1.6}@media(max-width:680px){.cv-lab header{flex-direction:column}.cv-lab__controls,.cv-lab__bottom{grid-template-columns:1fr}.cv-lab__folds{overflow-x:auto}.cv-lab__folds div{min-width:500px}}
</style>
