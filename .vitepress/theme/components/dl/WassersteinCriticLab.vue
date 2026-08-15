<script setup lang="ts">
import { computed, ref } from 'vue'
const real=ref(1.1), fake=ref(-.8), slope=ref(1), regularizer=ref<'clip'|'gp'>('gp'), bound=ref(.01)
const effectiveSlope=computed(()=>regularizer.value==='clip'?Math.min(slope.value,Math.max(.1,bound.value*40)):slope.value)
const estimate=computed(()=>effectiveSlope.value*(real.value-fake.value))
const penalty=computed(()=>regularizer.value==='gp'?10*(Math.abs(slope.value)-1)**2:Math.max(0,slope.value-effectiveSlope.value))
</script>
<template><DlExperimentFrame index="13-A" title="Wasserstein Critic 量规" question="critic 的函数值差怎样近似分布距离，1-Lipschitz 约束又为何不能省？" formula="W≈E[f(x)]−E[f(G(z))],  ||∇f(x̂)||₂≈1">
<template #visual><svg viewBox="0 0 540 270" role="img" aria-label="critic 函数与真实生成样本位置"><line x1="38" y1="220" x2="500" y2="220"/><line :x1="270+fake*80" y1="45" :x2="270+fake*80" y2="230" class="fake"/><line :x1="270+real*80" y1="45" :x2="270+real*80" y2="230" class="real"/><path :d="`M45 ${185+effectiveSlope*45} L495 ${185-effectiveSlope*45}`"/><text :x="270+fake*80" y="248">fake</text><text :x="270+real*80" y="248">real</text></svg></template>
<template #controls><label>真实中心 = {{real.toFixed(1)}}<input v-model.number="real" type="range" min="-2" max="2" step=".1"></label><label>生成中心 = {{fake.toFixed(1)}}<input v-model.number="fake" type="range" min="-2" max="2" step=".1"></label><label>critic 斜率 = {{slope.toFixed(1)}}<input v-model.number="slope" type="range" min=".1" max="3" step=".1"></label><label>约束<select v-model="regularizer"><option value="gp">Gradient Penalty</option><option value="clip">Weight clipping 代理</option></select></label><label>clip bound = {{bound.toFixed(2)}}<input v-model.number="bound" type="range" min=".005" max=".1" step=".005"></label></template>
<template #metrics><div><span>有效斜率</span><strong>{{effectiveSlope.toFixed(2)}}</strong></div><div><span>W 距离估计</span><strong>{{estimate.toFixed(3)}}</strong></div><div><span>约束代价</span><strong>{{penalty.toFixed(3)}}</strong></div><div><span>中心间隔</span><strong>{{Math.abs(real-fake).toFixed(2)}}</strong></div></template>
<template #note>WGAN critic 不输出概率，也不应使用 sigmoid。原始 weight clipping 容易压坏容量；WGAN-GP 在真实与生成样本插值点约束梯度范数，但也只是 Lipschitz 约束的近似实现。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #493c68;border-radius:7px;background:#0b172a}path{fill:none;stroke:#a88ce8;stroke-width:4}line{stroke:#526d78}.real{stroke:#58d3c8;stroke-width:3}.fake{stroke:#df6c5b;stroke-width:3}text{fill:#e4e8ec;font:10px ui-monospace,monospace;text-anchor:middle}</style>
