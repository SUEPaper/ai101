<script setup lang="ts">
import { computed, ref } from 'vue'
const layers=ref(32), qHeads=ref(32), kvHeads=ref(8), headDim=ref(128), tokens=ref(8192), bytes=ref(2)
const cacheGiB=computed(()=>2*layers.value*kvHeads.value*headDim.value*tokens.value*bytes.value/2**30)
const ratio=computed(()=>qHeads.value/kvHeads.value), mode=computed(()=>kvHeads.value===qHeads.value?'MHA':kvHeads.value===1?'MQA':'GQA')
</script>
<template><DlExperimentFrame index="20-A" title="KV Cache 显存计算器" question="MQA/GQA 为何减少 decode 状态，而不减少 query heads？" formula="KV bytes=2·L·T·Hkv·Dh·bytes">
<template #visual><div class="cache"><span v-for="i in Math.min(kvHeads,16)" :key="i" :style="{height:`${70+i*5}px`}"></span></div></template>
<template #controls><label>层数 = {{layers}}<input v-model.number="layers" type="range" min="8" max="96" step="8"></label><label>Q heads = {{qHeads}}<input v-model.number="qHeads" type="range" min="8" max="64" step="8"></label><label>KV heads = {{kvHeads}}<input v-model.number="kvHeads" type="range" min="1" max="32" step="1"></label><label>上下文 = {{tokens}}<input v-model.number="tokens" type="range" min="1024" max="32768" step="1024"></label></template>
<template #metrics><div><span>模式</span><strong>{{mode}}</strong></div><div><span>单序列 KV</span><strong>{{cacheGiB.toFixed(2)}} GiB</strong></div><div><span>Q:KV 共享</span><strong>{{ratio.toFixed(1)}}:1</strong></div><div><span>每元素字节</span><strong>{{bytes}}</strong></div></template>
<template #note>公式未含 batch、并行切分、分页碎片与 allocator 余量。KV heads 必须整除 query heads；训练时激活账本与 decode KV 账本不同。</template>
</DlExperimentFrame></template>
<style scoped>.cache{display:flex;align-items:end;justify-content:center;gap:8px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.cache span{width:20px;max-height:210px;border:1px solid #58d3c8;background:linear-gradient(#58d3c8,#193b4b)}</style>
