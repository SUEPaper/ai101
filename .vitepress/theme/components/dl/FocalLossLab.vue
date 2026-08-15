<script setup lang="ts">
import { computed, ref } from 'vue'
const probability=ref(.82), gamma=ref(2), alpha=ref(.25), positive=ref(true)
const pt=computed(()=>positive.value?probability.value:1-probability.value)
const ce=computed(()=>-Math.log(Math.max(pt.value,1e-6)))
const focal=computed(()=>alpha.value*(1-pt.value)**gamma.value*ce.value)
const modulation=computed(()=>(1-pt.value)**gamma.value)
</script>
<template><DlExperimentFrame index="07-A" title="Focal Loss 梯度聚光器" question="海量易分类背景如何被连续降权，而不是被硬采样全部删除？" formula="FL(pₜ)=-αₜ(1-pₜ)^γ log pₜ">
<template #visual><svg viewBox="0 0 540 270" role="img" aria-label="交叉熵和 focal loss 随正确类别概率变化的曲线"><polyline class="ce" :points="Array.from({length:49},(_,i)=>{const p=.02+i*.02;return `${35+p*470},${230-Math.min(200,-Math.log(p)*44)}`}).join(' ')"/><polyline class="fl" :points="Array.from({length:49},(_,i)=>{const p=.02+i*.02;return `${35+p*470},${230-Math.min(200,(1-p)**gamma*-Math.log(p)*44)}`}).join(' ')"/><line :x1="35+pt*470" y1="28" :x2="35+pt*470" y2="232"/><text x="91" y="36">CE</text><text x="162" y="36">FOCAL</text><text x="460" y="252">pₜ →</text></svg></template>
<template #controls><label>模型给正类概率 p = {{probability.toFixed(2)}}<input v-model.number="probability" type="range" min="0.02" max="0.98" step="0.02"></label><label>γ = {{gamma.toFixed(1)}}<input v-model.number="gamma" type="range" min="0" max="5" step="0.25"></label><label>α = {{alpha.toFixed(2)}}<input v-model.number="alpha" type="range" min="0.05" max="1" step="0.05"></label><button type="button" @click="positive=!positive">当前标签：{{positive?'正类':'负类'}}</button></template>
<template #metrics><div><span>正确类概率 pₜ</span><strong>{{pt.toFixed(3)}}</strong></div><div><span>交叉熵</span><strong>{{ce.toFixed(4)}}</strong></div><div><span>调制因子</span><strong>{{modulation.toFixed(4)}}</strong></div><div><span>Focal loss</span><strong>{{focal.toFixed(4)}}</strong></div></template>
<template #note>γ=0 时退化为带 α 的交叉熵。γ 增大不会凭空创造正样本；标注遗漏、极小目标召回和分类/回归尺度失衡仍需单独诊断。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.ce,.fl{fill:none;stroke-width:3}.ce{stroke:#718d98}.fl{stroke:#f2c14e}line{stroke:#58d3c8;stroke-dasharray:5 4}text{fill:#d7e6e7;font:10px ui-monospace,monospace}</style>
