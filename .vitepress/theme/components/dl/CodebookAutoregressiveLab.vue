<script setup lang="ts">
import { computed, ref } from 'vue'
const size=ref(64), used=ref(22), commitment=ref(.25), context=ref(8)
const perplexity=computed(()=>Math.min(size.value,used.value*(1-.16*Math.abs(commitment.value-.3))))
const dead=computed(()=>Math.max(0,size.value-Math.round(perplexity.value)))
const bits=computed(()=>Math.log2(size.value)), dependencies=computed(()=>Math.min(120,context.value*(context.value+1)/2))
</script>
<template><DlExperimentFrame index="14-B" title="离散码本与序列建模台" question="连续 encoder 输出被量化成 code 后，码本利用率与自回归上下文怎样共同限制生成？" formula="k*=argminₖ||zₑ−eₖ||²; p(k₁:T)=∏ₜp(kₜ|k<t)">
<template #visual><div class="codes" role="img" aria-label="VQ 码本使用与自回归顺序"><span v-for="i in Math.min(size,100)" :key="i" :class="i<=perplexity?'used':'dead'">{{i<=16?i:''}}</span><i :style="{width:`${Math.min(100,context*5)}%`}"></i></div></template>
<template #controls><label>码本大小 K = {{size}}<input v-model.number="size" type="range" min="16" max="256" step="16"></label><label>活跃 code 代理 = {{used}}<input v-model.number="used" type="range" min="1" :max="size" step="1"></label><label>commitment β = {{commitment.toFixed(2)}}<input v-model.number="commitment" type="range" min="0" max="1" step=".05"></label><label>自回归上下文 = {{context}}<input v-model.number="context" type="range" min="1" max="24" step="1"></label></template>
<template #metrics><div><span>码本 perplexity</span><strong>{{perplexity.toFixed(1)}} / {{size}}</strong></div><div><span>死码代理</span><strong>{{dead}}</strong></div><div><span>单 code 上限</span><strong>{{bits.toFixed(1)}} bit</strong></div><div><span>上下文依赖边</span><strong>{{dependencies.toFixed(0)}}</strong></div></template>
<template #note>VQ 把连续潜变量变成离散 token，后续 PixelCNN/Transformer 可按顺序建模。码本坍缩会浪费容量；更大 K 也增加先验建模难度和序列熵。</template>
</DlExperimentFrame></template>
<style scoped>.codes{display:grid;grid-template-columns:repeat(16,1fr);gap:4px;align-content:center;min-height:280px;padding:24px;border:1px solid #493c68;border-radius:7px;background:#0b172a}.codes span{display:grid;place-items:center;aspect-ratio:1;border:1px solid #3a4e63;color:#d9e5e7;font:8px ui-monospace,monospace}.codes .used{background:#6953a0;border-color:#a88ce8}.codes .dead{opacity:.3}.codes i{grid-column:1/-1;height:8px;background:#f2c14e}</style>
