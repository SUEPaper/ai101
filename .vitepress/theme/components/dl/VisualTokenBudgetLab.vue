<script setup lang="ts">
import { computed, ref } from 'vue'
const resolution=ref(336), patch=ref(14), crops=ref(1), resampler=ref(false)
const raw=computed(()=>Math.ceil(resolution.value/patch.value)**2*crops.value), tokens=computed(()=>resampler.value?Math.min(256,raw.value):raw.value)
const attention=computed(()=>tokens.value*tokens.value), latency=computed(()=>12+tokens.value*.06+attention.value/200000)
</script>
<template><DlExperimentFrame index="27-A" title="视觉 Token 与延迟预算台" question="分辨率、patch、multi-crop 和 resampler 怎样决定进入 LLM 的视觉序列长度？" formula="visual tokens≈crops·ceil(H/P)·ceil(W/P)">
<template #visual><div class="patches" :style="{gridTemplateColumns:`repeat(${Math.min(12,Math.ceil(resolution/patch))},1fr)`}"><i v-for="i in Math.min(tokens,144)" :key="i"></i></div></template>
<template #controls><label>分辨率 = {{resolution}}<input v-model.number="resolution" type="range" min="224" max="896" step="56"></label><label>Patch = {{patch}}<input v-model.number="patch" type="range" min="7" max="32" step="1"></label><label>Crop 数 = {{crops}}<input v-model.number="crops" type="range" min="1" max="8" step="1"></label><button type="button" @click="resampler=!resampler">{{resampler?'关闭 resampler':'压到 256 token'}}</button></template>
<template #metrics><div><span>原始视觉 token</span><strong>{{raw}}</strong></div><div><span>送入 LLM</span><strong>{{tokens}}</strong></div><div><span>注意力代理</span><strong>{{attention.toLocaleString()}}</strong></div><div><span>延迟代理</span><strong>{{latency.toFixed(1)}} ms</strong></div></template>
<template #note>视觉 token 与文本共享上下文和 KV；高分辨率不保证小字正确。真实系统还受视觉 encoder、动态切块、projector 和 batch 影响。</template>
</DlExperimentFrame></template>
<style scoped>.patches{display:grid;align-content:center;gap:3px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.patches i{aspect-ratio:1;border:1px solid #58d3c8;background:#173948}</style>
