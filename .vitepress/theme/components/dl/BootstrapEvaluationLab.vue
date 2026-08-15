<script setup lang="ts">
import { computed, ref } from 'vue'
const samples=ref(200), delta=ref(.04), disagreement=ref(.3), confidence=ref(1.96)
const se=computed(()=>Math.sqrt(Math.max(.0001,disagreement.value*(1-disagreement.value)/samples.value)))
const low=computed(()=>delta.value-confidence.value*se.value), high=computed(()=>delta.value+confidence.value*se.value)
const verdict=computed(()=>low.value>0?'B 胜':high.value<0?'A 胜':'不确定')
</script>
<template><DlExperimentFrame index="28-A" title="配对评测置信走廊" question="同一题集上的模型差值为何要按样本配对估计，而不是只报两个总分？" formula="CI proxy=Δ±z·sqrt(q(1−q)/n)">
<template #visual><div class="ci"><i :style="{left:`${50+low*400}%`,width:`${Math.max(2,(high-low)*400)}%`}"></i><b style="left:50%"></b></div></template>
<template #controls><label>样本数 = {{samples}}<input v-model.number="samples" type="range" min="50" max="2000" step="50"></label><label>配对差值 = {{delta.toFixed(3)}}<input v-model.number="delta" type="range" min="-.2" max=".2" step=".01"></label><label>不一致率 = {{disagreement.toFixed(2)}}<input v-model.number="disagreement" type="range" min=".05" max=".8" step=".05"></label><label>z = {{confidence.toFixed(2)}}<input v-model.number="confidence" type="range" min="1" max="2.58" step=".01"></label></template>
<template #metrics><div><span>区间下界</span><strong>{{low.toFixed(3)}}</strong></div><div><span>区间上界</span><strong>{{high.toFixed(3)}}</strong></div><div><span>结论</span><strong>{{verdict}}</strong></div><div><span>配对单位</span><strong>题目/任务</strong></div></template>
<template #note>页面使用解析代理；正式报告应按用户/文档/任务的独立单位 bootstrap，保留原始配对结果，并为多重试验校正。</template>
</DlExperimentFrame></template>
<style scoped>.ci{position:relative;min-height:280px;border:1px solid #315765;border-radius:7px;background:#0b172a}.ci i{position:absolute;top:125px;height:30px;background:#58d3c8}.ci b{position:absolute;top:35px;bottom:35px;border-left:2px solid #f2c14e}</style>
