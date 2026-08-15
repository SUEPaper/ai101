<script setup lang="ts">
import { computed, ref } from 'vue'
const kernel=ref(3), dilation=ref(2), layers=ref(2)
const single=computed(()=>kernel.value+(kernel.value-1)*(dilation.value-1))
const effective=computed(()=>1+layers.value*(single.value-1))
const sampled=(x:number,y:number)=>{const c=5, half=(kernel.value-1)/2;return Math.abs(x-c)%dilation.value===0&&Math.abs(y-c)%dilation.value===0&&Math.abs(x-c)<=half*dilation.value&&Math.abs(y-c)<=half*dilation.value}
</script>
<template><DlExperimentFrame index="08-A" title="空洞卷积视野板" question="不降低特征图分辨率，怎样扩大单层采样跨度；代价又是什么？" formula="Keff = K + (K-1)(d-1)">
<template #visual><div class="grid" role="img" aria-label="空洞卷积采样格点"><span v-for="n in 121" :key="n" :class="{sampled:sampled((n-1)%11,Math.floor((n-1)/11)),center:n===61}"></span></div></template>
<template #controls><label>卷积核 K = {{kernel}}<input v-model.number="kernel" type="range" min="1" max="5" step="2"></label><label>dilation d = {{dilation}}<input v-model.number="dilation" type="range" min="1" max="4" step="1"></label><label>串联层数 = {{layers}}<input v-model.number="layers" type="range" min="1" max="6" step="1"></label></template>
<template #metrics><div><span>单层有效核</span><strong>{{single}}×{{single}}</strong></div><div><span>单层采样点</span><strong>{{kernel**2}}</strong></div><div><span>串联感受野代理</span><strong>{{effective}}×{{effective}}</strong></div><div><span>采样密度</span><strong>{{(kernel**2/single**2*100).toFixed(1)}}%</strong></div></template>
<template #note>空洞卷积扩大跨度但不增加采样点，较大 dilation 可能形成 gridding artifact。DeepLab 用多尺度空洞率并行取证；U-Net 则通过下采样语义与跳连细节共同恢复边界。</template>
</DlExperimentFrame></template>
<style scoped>.grid{display:grid;grid-template-columns:repeat(11,22px);gap:4px;justify-content:center;align-content:center;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.grid span{width:22px;height:22px;border:1px solid #27485b;background:#13263b}.grid .sampled{border-color:#58d3c8;background:#2a7f7a}.grid .center{box-shadow:0 0 0 3px #f2c14e inset}@media(max-width:420px){.grid{grid-template-columns:repeat(11,18px)}.grid span{width:18px;height:18px}}</style>
