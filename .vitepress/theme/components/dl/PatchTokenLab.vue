<script setup lang="ts">
import { computed, ref } from 'vue'
const resolution=ref(224), patch=ref(16), width=ref(768), layers=ref(12)
const side=computed(()=>Math.floor(resolution.value/patch.value)), tokens=computed(()=>side.value**2+1)
const attention=computed(()=>layers.value*tokens.value**2*width.value/1e9)
const patchProjection=computed(()=>patch.value**2*3*width.value/1e6)
</script>
<template><DlExperimentFrame index="10-A" title="Patch Token 预算板" question="减小 patch 获得细节时，注意力矩阵和位置编码要付出多大代价？" formula="N=(H/P)(W/P)+1, attention ∝ L·N²·D">
<template #visual><div class="token-grid" :style="{gridTemplateColumns:`repeat(${Math.min(side,16)},1fr)`}" role="img" :aria-label="`${side} 乘 ${side} 个图像 patch 的抽样网格`"><span v-for="n in Math.min(tokens-1,256)" :key="n" :class="{hot:n%7===0}"></span></div></template>
<template #controls><label>输入边长 = {{resolution}}<input v-model.number="resolution" type="range" min="160" max="384" step="16"></label><label>patch 边长 = {{patch}}<input v-model.number="patch" type="range" min="8" max="32" step="8"></label><label>隐藏宽度 D = {{width}}<input v-model.number="width" type="range" min="192" max="1024" step="64"></label><label>层数 L = {{layers}}<input v-model.number="layers" type="range" min="4" max="24" step="2"></label></template>
<template #metrics><div><span>token 数（含 CLS）</span><strong>{{tokens}}</strong></div><div><span>注意力分数项</span><strong>{{attention.toFixed(2)}} G</strong></div><div><span>patch 投影参数</span><strong>{{patchProjection.toFixed(2)}} M</strong></div><div><span>单 patch 像素</span><strong>{{patch**2}}</strong></div></template>
<template #note>patch 从 16 降到 8 时 token 约增 4 倍，全局注意力项约增 16 倍。ViT 的 patchification 也是一种带步长卷积；小目标可能在投影前就被粗 patch 稀释。</template>
</DlExperimentFrame></template>
<style scoped>.token-grid{display:grid;gap:3px;align-content:center;min-height:280px;padding:24px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.token-grid span{aspect-ratio:1;min-width:4px;border:1px solid #2d5665;background:#173149}.token-grid .hot{border-color:#f2c14e;background:#5f5124}</style>
