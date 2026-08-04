<script setup lang="ts">
import { computed, ref } from 'vue'

const adjustment = ref(0)
const homes = [
  { group: '新城区', actual: 92, predicted: 95 }, { group: '新城区', actual: 105, predicted: 102 }, { group: '新城区', actual: 118, predicted: 121 },
  { group: '老城区', actual: 82, predicted: 68 }, { group: '老城区', actual: 96, predicted: 80 }, { group: '老城区', actual: 110, predicted: 91 }
]
const rows = computed(() => homes.map(home => ({ ...home, adjusted: home.predicted + (home.group === '老城区' ? adjustment.value : 0) })))
const metrics = (group: string) => {
  const groupRows = rows.value.filter(row => row.group === group)
  return {
    mae: groupRows.reduce((sum, row) => sum + Math.abs(row.adjusted - row.actual), 0) / groupRows.length,
    under: groupRows.filter(row => row.adjusted < row.actual * .9).length / groupRows.length
  }
}
const newMetric = computed(() => metrics('新城区'))
const oldMetric = computed(() => metrics('老城区'))
</script>

<template>
  <section class="fair-lab" aria-label="房租评估公平性情景实验">
    <header><div><span>INTERACTIVE 15</span><h3>总体误差相近，不代表每个群体受到相同影响</h3></div><code>老城区修正 {{ adjustment >= 0 ? '+' : '' }}{{ adjustment }}</code></header>
    <label>调整老城区预测值<input v-model.number="adjustment" type="range" min="-8" max="24" step="1"></label>
    <div class="fair-lab__groups">
      <div><small>新城区</small><strong>MAE {{ newMetric.mae.toFixed(1) }}</strong><span>严重低估率 {{ (newMetric.under*100).toFixed(0) }}%</span></div>
      <div><small>老城区</small><strong>MAE {{ oldMetric.mae.toFixed(1) }}</strong><span>严重低估率 {{ (oldMetric.under*100).toFixed(0) }}%</span></div>
    </div>
    <div class="fair-lab__rows"><i v-for="row in rows" :key="row.group+row.actual"><span>{{ row.group }}</span><b :style="{width:`${row.actual/1.3}%`}">实际 {{ row.actual }}</b><em :style="{width:`${Math.max(0,row.adjusted)/1.3}%`}">预测 {{ row.adjusted }}</em></i></div>
    <p>这个滑块只用于暴露群体误差，不能替代政策判断。真实系统还要审查数据来源、申诉渠道、个体伤害和法律边界。</p>
  </section>
</template>

<style scoped>
.fair-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.fair-lab header{display:flex;justify-content:space-between;gap:16px}.fair-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.fair-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.fair-lab header code{align-self:start;padding:8px 10px;border-radius:7px;color:#17302b;background:#f3c56f}.fair-lab>label{display:block;margin:22px 0;color:#bdd1ca;font-size:12px}.fair-lab input{width:100%;margin-top:9px;accent-color:#f3c56f}.fair-lab__groups{display:grid;grid-template-columns:1fr 1fr;gap:12px}.fair-lab__groups div{padding:14px;border-radius:9px;background:#0c2420}.fair-lab__groups small,.fair-lab__groups strong,.fair-lab__groups span{display:block}.fair-lab__groups small{color:#8fafa4}.fair-lab__groups strong{margin:5px 0;color:#fff}.fair-lab__groups span{color:#f3c56f;font:11px ui-monospace,monospace}.fair-lab__rows{margin-top:18px}.fair-lab__rows i{display:grid;grid-template-columns:50px 1fr;gap:5px;margin:10px 0;font-style:normal}.fair-lab__rows i span{grid-row:1/3;align-self:center;color:#8eaaa1;font-size:9px}.fair-lab__rows b,.fair-lab__rows em{box-sizing:border-box;min-width:58px;padding:3px 6px;border-radius:3px;white-space:nowrap;font:8px ui-monospace,monospace}.fair-lab__rows b{color:#17302b;background:#8ebeb1}.fair-lab__rows em{color:#17302b;background:#f3c56f;font-style:normal}.fair-lab p{margin:18px 0 0;padding-top:15px;border-top:1px solid #46675e;color:#a7beb6;font-size:11px;line-height:1.7}@media(max-width:620px){.fair-lab header{flex-direction:column}.fair-lab__groups{grid-template-columns:1fr}}
</style>
