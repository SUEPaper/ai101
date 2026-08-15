<script setup lang="ts">
import { computed, ref } from 'vue'
const layers=ref(12), cutoff=ref(5), truncation=ref(.7), noise=ref(.35)
const coarse=computed(()=>cutoff.value/layers.value), fine=computed(()=>1-coarse.value)
const diversity=computed(()=>Math.max(0,truncation.value*(.72+noise.value*.28)))
const fidelity=computed(()=>Math.min(1,1.05-.35*truncation.value-.08*noise.value))
</script>
<template><DlExperimentFrame index="13-B" title="Style Mixing 频段台" question="把两个 w 潜变量注入不同层级时，粗布局与细纹理分别由谁控制？" formula="xₗ = AdaIN/ModConv(xₗ₋₁, A(w)); w₁ before cutoff, w₂ after">
<template #visual><div class="style-stack" role="img" aria-label="StyleGAN 层级样式混合"><span v-for="i in layers" :key="i" :class="i<=cutoff?'coarse':'fine'" :style="{height:`${28+i*5}px`}"></span></div></template>
<template #controls><label>合成层数 = {{layers}}<input v-model.number="layers" type="range" min="6" max="18" step="2"></label><label>style cutoff = {{cutoff}}<input v-model.number="cutoff" type="range" min="1" :max="layers-1" step="1"></label><label>truncation ψ = {{truncation.toFixed(2)}}<input v-model.number="truncation" type="range" min=".2" max="1" step=".05"></label><label>随机噪声 = {{noise.toFixed(2)}}<input v-model.number="noise" type="range" min="0" max="1" step=".05"></label></template>
<template #metrics><div><span>w₁ 粗层占比</span><strong>{{(coarse*100).toFixed(0)}}%</strong></div><div><span>w₂ 细层占比</span><strong>{{(fine*100).toFixed(0)}}%</strong></div><div><span>多样性代理</span><strong>{{diversity.toFixed(3)}}</strong></div><div><span>保真代理</span><strong>{{fidelity.toFixed(3)}}</strong></div></template>
<template #note>Style mixing 揭示层级控制，但“粗/细”是统计趋势而非严格语义轴。truncation 向平均 w 收缩通常提高典型性并降低多样性；噪声主要承载随机微结构。</template>
</DlExperimentFrame></template>
<style scoped>.style-stack{display:flex;align-items:center;justify-content:center;gap:7px;min-height:280px;border:1px solid #493c68;border-radius:7px;background:#0b172a}.style-stack span{width:22px;border-radius:2px;transition:all .2s}.coarse{background:#a88ce8}.fine{background:#f2c14e}</style>
