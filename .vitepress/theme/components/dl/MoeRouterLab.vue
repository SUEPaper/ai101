<script setup lang="ts">
import { computed, ref } from 'vue'
const experts=ref(8), topK=ref(2), tokens=ref(1024), capacity=ref(1.25), skew=ref(.2)
const perExpert=computed(()=>Math.ceil(tokens.value*topK.value/experts.value*capacity.value))
const demand=computed(()=>tokens.value*topK.value*(1+skew.value*(experts.value-1))/experts.value)
const overflow=computed(()=>Math.max(0,(demand.value-perExpert.value)/demand.value))
</script>
<template><DlExperimentFrame index="20-B" title="MoE 路由与容量门" question="稀疏专家只激活部分参数时，路由倾斜为何仍会丢 token 或浪费容量？" formula="capacity≈T·top-k/E·factor">
<template #visual><div class="experts"><span v-for="i in experts" :key="i" :class="{hot:i===1}" :style="{opacity:i===1?1:.85-skew*.5}">E{{i}}</span></div></template>
<template #controls><label>专家数 = {{experts}}<input v-model.number="experts" type="range" min="4" max="16" step="4"></label><label>Top-k = {{topK}}<input v-model.number="topK" type="range" min="1" max="4" step="1"></label><label>容量因子 = {{capacity.toFixed(2)}}<input v-model.number="capacity" type="range" min="1" max="2" step=".05"></label><label>路由偏斜 = {{skew.toFixed(2)}}<input v-model.number="skew" type="range" min="0" max=".8" step=".05"></label></template>
<template #metrics><div><span>激活专家/token</span><strong>{{topK}}</strong></div><div><span>每专家容量</span><strong>{{perExpert}}</strong></div><div><span>热点需求</span><strong>{{demand.toFixed(0)}}</strong></div><div><span>溢出代理</span><strong>{{(overflow*100).toFixed(1)}}%</strong></div></template>
<template #note>真实 MoE 还受 auxiliary load-balancing loss、expert parallel 通信、token dropping 与共享专家影响；总参数量大不等于每 token FLOPs 同比增加。</template>
</DlExperimentFrame></template>
<style scoped>.experts{display:grid;grid-template-columns:repeat(4,1fr);align-content:center;gap:12px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.experts span{padding:25px 5px;text-align:center;border:1px solid #58d3c8;color:#dff}.experts .hot{border-color:#df6c5b;color:#fff;background:#4b2434}</style>
