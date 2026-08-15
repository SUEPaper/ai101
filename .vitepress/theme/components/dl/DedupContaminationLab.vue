<script setup lang="ts">
import { computed, ref } from 'vue'
const docs=ref(10), duplicate=ref(.18), overlap=ref(.01), quality=ref(.75)
const unique=computed(()=>docs.value*(1-duplicate.value)), effective=computed(()=>unique.value*quality.value)
const contamination=computed(()=>1-Math.pow(1-overlap.value,Math.max(1,docs.value*100)))
</script>
<template><DlExperimentFrame index="21-B" title="近重复与评测污染审计台" question="原始 token 增长时，有效独立信息为何可能不增，benchmark 泄漏风险为何急剧累计？" formula="effective≈raw·(1−dup)·quality">
<template #visual><div class="docs"><span v-for="i in 24" :key="i" :class="{dup:i%Math.max(2,Math.round(1/duplicate))===0,leak:i===24}"></span></div></template>
<template #controls><label>原始文档 = {{docs}}M<input v-model.number="docs" type="range" min="1" max="50" step="1"></label><label>近重复率 = {{duplicate.toFixed(2)}}<input v-model.number="duplicate" type="range" min="0" max=".7" step=".05"></label><label>评测片段重合率 = {{overlap.toFixed(3)}}<input v-model.number="overlap" type="range" min="0" max=".05" step=".002"></label><label>质量保留 = {{quality.toFixed(2)}}<input v-model.number="quality" type="range" min=".2" max="1" step=".05"></label></template>
<template #metrics><div><span>唯一文档</span><strong>{{unique.toFixed(2)}}M</strong></div><div><span>有效质量代理</span><strong>{{effective.toFixed(2)}}M</strong></div><div><span>污染累计风险</span><strong>{{(contamination*100).toFixed(1)}}%</strong></div><div><span>需保留 lineage</span><strong>是</strong></div></template>
<template #note>重复率与质量不是独立标量，图中风险只示累计效应。生产管线必须保留来源、许可证、hash、过滤版本和 benchmark blocklist。</template>
</DlExperimentFrame></template>
<style scoped>.docs{display:grid;grid-template-columns:repeat(8,1fr);align-content:center;gap:8px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.docs span{height:36px;border:1px solid #58d3c8;background:#173948}.docs .dup{border-color:#f2c14e;background:#4a3b24}.docs .leak{border-color:#df6c5b;background:#542638}</style>
