<script setup lang="ts">
import { computed, ref } from 'vue'
const gpus=ref(64), tp=ref(8), pp=ref(4), micro=ref(8), sequence=ref(4096)
const dp=computed(()=>Math.max(1,Math.floor(gpus.value/(tp.value*pp.value))))
const bubble=computed(()=>Math.max(0,(pp.value-1)/(micro.value+pp.value-1)))
const comm=computed(()=>Math.log2(tp.value+1)*sequence.value/1024+dp.value*.8+pp.value*.5)
const valid=computed(()=>gpus.value%(tp.value*pp.value)===0)
</script>
<template><DlExperimentFrame index="31-A" title="DP—TP—PP 并行拓扑规划器" question="给定 GPU 数、层切分与 micro-batches，数据/张量/流水线并行怎样决定副本数、通信和气泡？" formula="world=DP·TP·PP; bubble≈(PP−1)/(micro+PP−1)">
<template #visual><div class="topology"><span v-for="i in Math.min(gpus,64)" :key="i" :class="{cut:i%tp===0}"></span></div></template>
<template #controls><label>GPU = {{gpus}}<input v-model.number="gpus" type="range" min="8" max="128" step="8"></label><label>TP = {{tp}}<input v-model.number="tp" type="range" min="1" max="16" step="1"></label><label>PP = {{pp}}<input v-model.number="pp" type="range" min="1" max="16" step="1"></label><label>Micro-batches = {{micro}}<input v-model.number="micro" type="range" min="1" max="64" step="1"></label></template>
<template #metrics><div><span>DP replicas</span><strong>{{dp}}</strong></div><div><span>Pipeline bubble</span><strong>{{(bubble*100).toFixed(1)}}%</strong></div><div><span>通信代理</span><strong>{{comm.toFixed(1)}}</strong></div><div><span>拓扑可整除</span><strong>{{valid?'是':'否'}}</strong></div></template>
<template #note>实际规划还含 context/expert parallel、节点边界、NVLink/网络拓扑、virtual pipeline、激活重算和重叠。整除只是最低门禁。</template>
</DlExperimentFrame></template>
<style scoped>.topology{display:grid;grid-template-columns:repeat(16,1fr);align-content:center;gap:4px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.topology span{aspect-ratio:1;border:1px solid #58d3c8;background:#173948}.topology .cut{border-color:#f2c14e}</style>
