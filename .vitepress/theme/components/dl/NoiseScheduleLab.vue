<script setup lang="ts">
import { computed, ref } from 'vue'
const steps=ref(1000), timestep=ref(350), schedule=ref<'linear'|'cosine'>('cosine')
const fraction=computed(()=>Math.min(1,timestep.value/steps.value))
const alphaBar=computed(()=>schedule.value==='linear'?Math.exp(-5*fraction.value**1.35):Math.cos((fraction.value+.008)/1.008*Math.PI/2)**2)
const snr=computed(()=>alphaBar.value/Math.max(1e-6,1-alphaBar.value))
const clean=computed(()=>Math.sqrt(alphaBar.value)), noise=computed(()=>Math.sqrt(1-alphaBar.value))
</script>
<template><DlExperimentFrame index="15-A" title="扩散噪声日程台" question="同一个 timestep 编号为何不能跨 schedule 直接比较；模型实际看到的是怎样的信噪比？" formula="xₜ=√ᾱₜ x₀ + √(1−ᾱₜ) ε, SNR=ᾱₜ/(1−ᾱₜ)">
<template #visual><svg viewBox="0 0 540 270" role="img" aria-label="扩散过程信号与噪声权重"><rect x="40" y="78" :width="440*clean" height="46" class="clean"/><rect x="40" y="154" :width="440*noise" height="46" class="noise"/><text x="42" y="65">clean coefficient</text><text x="42" y="141">noise coefficient</text><line :x1="40+440*fraction" y1="35" :x2="40+440*fraction" y2="225"/></svg></template>
<template #controls><label>总步数 T = {{steps}}<input v-model.number="steps" type="range" min="100" max="2000" step="100"></label><label>当前 t = {{timestep}}<input v-model.number="timestep" type="range" min="0" :max="steps" step="10"></label><label>schedule<select v-model="schedule"><option value="cosine">cosine ᾱ</option><option value="linear">linear β 代理</option></select></label></template>
<template #metrics><div><span>归一化时间</span><strong>{{fraction.toFixed(3)}}</strong></div><div><span>ᾱₜ</span><strong>{{alphaBar.toFixed(4)}}</strong></div><div><span>SNR</span><strong>{{snr.toFixed(3)}}</strong></div><div><span>clean/noise 系数</span><strong>{{clean.toFixed(3)}} / {{noise.toFixed(3)}}</strong></div></template>
<template #note>训练常随机采样 t 并预测 ε、x₀ 或 v。不同参数化和 loss weighting 改变各 SNR 区域的梯度权重；只比较“1000 步”没有定义 schedule 与采样器。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #493c68;border-radius:7px;background:#0b172a}.clean{fill:#58d3c8}.noise{fill:#a88ce8}rect{transition:width .2s}line{stroke:#f2c14e;stroke-width:2}text{fill:#dce7e8;font:10px ui-monospace,monospace}</style>
