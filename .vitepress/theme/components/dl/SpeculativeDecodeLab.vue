<script setup lang="ts">
import { computed, ref } from 'vue'
const draft=ref(5), acceptance=ref(.72), draftCost=ref(.15), verifyCost=ref(1)
const accepted=computed(()=>draft.value*acceptance.value), cost=computed(()=>draft.value*draftCost.value+verifyCost.value)
const baseline=computed(()=>accepted.value+1), speedup=computed(()=>baseline.value/cost.value)
</script>
<template><DlExperimentFrame index="24-B" title="投机解码验收流水线" question="draft 一次提议多个 token，target 并行验证时，接受率和草稿成本怎样决定加速？" formula="speedup proxy≈(accepted+1)/(draft cost+verify cost)">
<template #visual><div class="draft"><span v-for="i in draft" :key="i" :class="{ok:i<=Math.round(accepted)}">{{i<=Math.round(accepted)?'✓':'×'}}</span></div></template>
<template #controls><label>草稿长度 = {{draft}}<input v-model.number="draft" type="range" min="1" max="12" step="1"></label><label>接受率 = {{acceptance.toFixed(2)}}<input v-model.number="acceptance" type="range" min=".1" max="1" step=".05"></label><label>Draft 相对成本 = {{draftCost.toFixed(2)}}<input v-model.number="draftCost" type="range" min=".02" max=".5" step=".02"></label><label>Verify 成本 = {{verifyCost.toFixed(1)}}<input v-model.number="verifyCost" type="range" min=".5" max="2" step=".1"></label></template>
<template #metrics><div><span>期望接受</span><strong>{{accepted.toFixed(2)}}</strong></div><div><span>轮次成本代理</span><strong>{{cost.toFixed(2)}}</strong></div><div><span>速度提升代理</span><strong>{{speedup.toFixed(2)}}×</strong></div><div><span>分布</span><strong>target exact</strong></div></template>
<template #note>正确投机解码保留 target 分布；代理忽略 memory bandwidth、batch、拒绝后的残余 token 与 kernel。需以 TPOT/P95 实测。</template>
</DlExperimentFrame></template>
<style scoped>.draft{display:flex;flex-wrap:wrap;align-content:center;justify-content:center;gap:10px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.draft span{display:grid;place-items:center;width:48px;height:48px;border:1px solid #df6c5b;color:#df6c5b}.draft .ok{border-color:#58d3c8;color:#58d3c8;background:#173948}</style>
