<script setup lang="ts">
import { computed, ref } from 'vue'
const chunk=ref(320), overlap=ref(48), topK=ref(8), rerank=ref(true)
const chunks=computed(()=>Math.ceil(4000/Math.max(1,chunk.value-overlap.value)))
const recall=computed(()=>Math.min(.98,.45+.07*Math.log2(topK.value)+.0005*chunk.value-.0003*overlap.value+(rerank.value?.08:0)))
const latency=computed(()=>18+topK.value*2+(rerank.value?topK.value*7:0)+chunks.value*.2)
</script>
<template><DlExperimentFrame index="25-A" title="RAG 切块—召回—重排管线" question="chunk、overlap、top-k 与 reranker 如何交换召回、噪声、延迟和上下文占用？" formula="retrieve→rerank→pack→generate">
<template #visual><div class="pipeline"><span>INDEX · {{chunks}}</span><b>→</b><span>TOP {{topK}}</span><b>→</b><span :class="{on:rerank}">RERANK</span></div></template>
<template #controls><label>Chunk token = {{chunk}}<input v-model.number="chunk" type="range" min="80" max="800" step="20"></label><label>Overlap = {{overlap}}<input v-model.number="overlap" type="range" min="0" max="200" step="8"></label><label>Top-k = {{topK}}<input v-model.number="topK" type="range" min="1" max="30" step="1"></label><button type="button" @click="rerank=!rerank">{{rerank?'关闭重排':'启用重排'}}</button></template>
<template #metrics><div><span>文档切块数</span><strong>{{chunks}}</strong></div><div><span>证据召回代理</span><strong>{{recall.toFixed(3)}}</strong></div><div><span>检索延迟代理</span><strong>{{latency.toFixed(1)}} ms</strong></div><div><span>送入上下文</span><strong>{{topK*chunk}} tok</strong></div></template>
<template #note>响应面只示权衡。真实评价需 query-level relevance、答案 span 覆盖、重复 chunk、metadata filter、索引版本和端到端引用正确性。</template>
</DlExperimentFrame></template>
<style scoped>.pipeline{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:12px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.pipeline span{padding:25px 8px;text-align:center;border:1px solid #597181;color:#c9d8dc}.pipeline .on{border-color:#58d3c8;color:#dff}.pipeline b{color:#f2c14e}@media(max-width:520px){.pipeline{grid-template-columns:1fr}.pipeline b{text-align:center}}</style>
