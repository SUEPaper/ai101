<script setup lang="ts">
import { computed, ref } from 'vue'
const requests=ref(24), prompt=ref(1024), output=ref(256), continuous=ref(true), maxTokens=ref(8192)
const active=computed(()=>Math.min(requests.value,Math.max(1,Math.floor(maxTokens.value/(prompt.value*.15+output.value)))))
const utilization=computed(()=>Math.min(.98,(continuous.value?.82:.58)*active.value/Math.max(1,requests.value)+.12))
const ttft=computed(()=>35+prompt.value*.025+active.value*1.5), tpot=computed(()=>continuous.value?12+active.value*.35:18+requests.value*.5)
</script>
<template><DlExperimentFrame index="32-A" title="Continuous Batching 调度台" question="不同 prompt/output 长度的请求如何在 prefill/decode 间动态加入退出，并交换吞吐与尾延迟？" formula="scheduler packs active token budget, not fixed request rectangles">
<template #visual><div class="requests"><span v-for="i in requests" :key="i" :class="{active:i<=active}"></span></div></template>
<template #controls><label>并发请求 = {{requests}}<input v-model.number="requests" type="range" min="1" max="64" step="1"></label><label>Prompt = {{prompt}} tok<input v-model.number="prompt" type="range" min="64" max="8192" step="64"></label><label>Output = {{output}} tok<input v-model.number="output" type="range" min="16" max="2048" step="16"></label><button type="button" @click="continuous=!continuous">{{continuous?'改为静态 batching':'启用 continuous batching'}}</button></template>
<template #metrics><div><span>Active requests</span><strong>{{active}}</strong></div><div><span>利用率代理</span><strong>{{utilization.toFixed(2)}}</strong></div><div><span>TTFT 代理</span><strong>{{ttft.toFixed(1)}} ms</strong></div><div><span>TPOT 代理</span><strong>{{tpot.toFixed(1)}} ms</strong></div></template>
<template #note>响应面不代表框架 benchmark。真实服务要区分 prefill/decode、调度策略、chunked prefill、优先级、取消和 P50/P95/P99。</template>
</DlExperimentFrame></template>
<style scoped>.requests{display:grid;grid-template-columns:repeat(8,1fr);align-content:center;gap:7px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.requests span{height:26px;border:1px solid #536979}.requests .active{border-color:#58d3c8;background:#173948}</style>
