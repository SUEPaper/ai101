<script setup lang="ts">
import { computed, ref } from 'vue'
const mean=ref(.8), logvar=ref(-.7), beta=ref(1), reconstruction=ref(.42)
const variance=computed(()=>Math.exp(logvar.value)), std=computed(()=>Math.sqrt(variance.value))
const kl=computed(()=>.5*(mean.value**2+variance.value-1-logvar.value))
const elbo=computed(()=>reconstruction.value+beta.value*kl.value)
const samples=computed(()=>[-1.5,-.8,0,.8,1.5].map(e=>mean.value+std.value*e))
</script>
<template><DlExperimentFrame index="14-A" title="变分瓶颈压力表" question="重构与先验匹配怎样争夺同一个潜变量通道，posterior collapse 又长什么样？" formula="L=E_q[−log p(x|z)] + β KL(q(z|x)||N(0,I))">
<template #visual><svg viewBox="0 0 540 270" role="img" aria-label="高斯潜变量后验与标准正态先验"><path d="M30 230 C135 230 170 55 270 55 C370 55 405 230 510 230" class="prior"/><path :d="`M30 230 C${190+mean*35-std*55} 230 ${230+mean*35-std*20} ${75+std*20} ${270+mean*35} ${75+std*20} C${310+mean*35+std*20} ${75+std*20} ${350+mean*35+std*55} 230 510 230`" class="posterior"/><circle v-for="(x,i) in samples" :key="i" :cx="270+x*58" cy="235" r="5"/></svg></template>
<template #controls><label>后验均值 μ = {{mean.toFixed(2)}}<input v-model.number="mean" type="range" min="-2" max="2" step=".05"></label><label>log variance = {{logvar.toFixed(2)}}<input v-model.number="logvar" type="range" min="-3" max="2" step=".05"></label><label>β = {{beta.toFixed(2)}}<input v-model.number="beta" type="range" min="0" max="4" step=".1"></label><label>重构 NLL 代理 = {{reconstruction.toFixed(2)}}<input v-model.number="reconstruction" type="range" min=".05" max="2" step=".05"></label></template>
<template #metrics><div><span>后验标准差</span><strong>{{std.toFixed(3)}}</strong></div><div><span>KL</span><strong>{{kl.toFixed(3)}}</strong></div><div><span>β-KL</span><strong>{{(beta*kl).toFixed(3)}}</strong></div><div><span>负 ELBO 代理</span><strong>{{elbo.toFixed(3)}}</strong></div></template>
<template #note>当 μ→0、variance→1 时 KL→0，若 decoder 仍能重构，潜变量可能被忽略。β 增大不是无条件“更解耦”，它会牺牲重构与可用容量。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #493c68;border-radius:7px;background:#0b172a}.prior,.posterior{fill:none;stroke-width:3}.prior{stroke:#54717d;stroke-dasharray:7 5}.posterior{stroke:#a88ce8}circle{fill:#f2c14e}</style>
