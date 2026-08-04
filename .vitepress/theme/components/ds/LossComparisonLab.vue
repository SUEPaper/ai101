<script setup lang="ts">
import { computed, ref } from 'vue'

const outlier = ref(12)
const values = computed(() => [5, 6, 6, 7, 8, outlier.value])
const mean = computed(() => values.value.reduce((a, b) => a + b, 0) / values.value.length)
const median = computed(() => {
  const sorted = [...values.value].sort((a, b) => a - b)
  return (sorted[2] + sorted[3]) / 2
})
const mse = computed(() => values.value.reduce((sum, value) => sum + (value - mean.value) ** 2, 0) / values.value.length)
const mae = computed(() => values.value.reduce((sum, value) => sum + Math.abs(value - median.value), 0) / values.value.length)
</script>

<template>
  <section class="loss-lab" aria-label="均值中位数与损失函数实验">
    <header><div><span>INTERACTIVE 11</span><h3>把最右侧观测推远，谁先被拉走？</h3></div><code>异常值 {{ outlier }}</code></header>
    <div class="loss-lab__track">
      <i v-for="(value, index) in values" :key="index" :style="{ left: `${(value - 4) / 28 * 100}%` }"><small>{{ value }}</small></i>
      <b class="mean" :style="{ left: `${(mean - 4) / 28 * 100}%` }">均值</b>
      <b class="median" :style="{ left: `${(median - 4) / 28 * 100}%` }">中位数</b>
    </div>
    <input v-model.number="outlier" aria-label="异常值大小" type="range" min="8" max="32" step="1">
    <div class="loss-lab__stats">
      <div><small>MSE 的最佳常数</small><strong>均值 {{ mean.toFixed(2) }}</strong><span>平方误差 {{ mse.toFixed(2) }}</span></div>
      <div><small>MAE 的最佳常数</small><strong>中位数 {{ median.toFixed(2) }}</strong><span>绝对误差 {{ mae.toFixed(2) }}</span></div>
    </div>
  </section>
</template>

<style scoped>
.loss-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.loss-lab header{display:flex;justify-content:space-between;gap:16px}.loss-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.loss-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.loss-lab header code{align-self:start;padding:8px 10px;border-radius:7px;color:#17302b;background:#f3c56f}.loss-lab__track{position:relative;height:96px;margin:30px 8px 8px;border-bottom:2px solid #5c7c73}.loss-lab__track i{position:absolute;bottom:-7px;width:13px;height:13px;border-radius:50%;transform:translateX(-50%);background:#fff}.loss-lab__track i small{position:absolute;bottom:18px;left:50%;transform:translateX(-50%);color:#abc2ba;font:10px ui-monospace,monospace}.loss-lab__track b{position:absolute;top:0;transform:translateX(-50%);font-size:11px}.loss-lab__track b:after{content:"";position:absolute;top:17px;left:50%;height:57px;border-left:2px dashed currentColor}.loss-lab__track .mean{color:#df765a}.loss-lab__track .median{top:26px;color:#f3c56f}.loss-lab>input{width:100%;accent-color:#f3c56f}.loss-lab__stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:22px}.loss-lab__stats div{padding:15px;border-radius:9px;background:#0c2420}.loss-lab__stats small,.loss-lab__stats strong,.loss-lab__stats span{display:block}.loss-lab__stats small{color:#91afa5}.loss-lab__stats strong{margin:6px 0;color:#fff}.loss-lab__stats span{color:#f3c56f;font:11px ui-monospace,monospace}@media(max-width:620px){.loss-lab header{flex-direction:column}.loss-lab__stats{grid-template-columns:1fr}}
</style>
