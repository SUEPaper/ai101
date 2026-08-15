<script setup lang="ts">
import { computed, ref } from 'vue'
const temperature=ref(1), topK=ref(5), topP=ref(.9), repetition=ref(1.1)
const logits=[3.2,2.6,2.1,1.5,.9,.4,-.2]
const probs=computed(()=>{const ex=logits.map(x=>Math.exp(x/temperature.value));const z=ex.reduce((a,b)=>a+b);return ex.map(x=>x/z)})
const kept=computed(()=>{let acc=0,n=0;for(const p of probs.value){if(n>=topK.value||acc>=topP.value)break;acc+=p;n++}return n})
const entropy=computed(()=>-probs.value.reduce((a,p)=>a+p*Math.log(p),0))
</script>
<template><DlExperimentFrame index="24-A" title="解码分布裁剪台" question="temperature、top-k、top-p 与重复惩罚如何先后改变候选分布？" formula="pᵢ∝exp(zᵢ/T); nucleus=min set with Σp≥p">
<template #visual><div class="bars"><i v-for="(p,i) in probs" :key="i" :class="{kept:i<kept}" :style="{height:`${p*500}px`}"></i></div></template>
<template #controls><label>Temperature = {{temperature.toFixed(2)}}<input v-model.number="temperature" type="range" min=".2" max="2" step=".05"></label><label>Top-k = {{topK}}<input v-model.number="topK" type="range" min="1" max="7" step="1"></label><label>Top-p = {{topP.toFixed(2)}}<input v-model.number="topP" type="range" min=".2" max="1" step=".05"></label><label>重复惩罚 = {{repetition.toFixed(2)}}<input v-model.number="repetition" type="range" min="1" max="1.5" step=".05"></label></template>
<template #metrics><div><span>保留候选</span><strong>{{kept}}</strong></div><div><span>分布熵</span><strong>{{entropy.toFixed(3)}}</strong></div><div><span>Top-1 概率</span><strong>{{probs[0].toFixed(3)}}</strong></div><div><span>惩罚协议</span><strong>{{repetition.toFixed(2)}}×</strong></div></template>
<template #note>教学图按已排序 logits 演示；真实实现要明确 temperature、penalty、top-k/top-p 的执行顺序、随机种子和 EOS 处理。</template>
</DlExperimentFrame></template>
<style scoped>.bars{display:flex;align-items:end;justify-content:center;gap:12px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.bars i{width:38px;max-height:220px;background:#34495b}.bars .kept{background:#58d3c8}</style>
