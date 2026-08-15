<script setup lang="ts">
import { computed, ref } from 'vue'
const samples=ref(50), augmentation=ref(.6), locality=ref(.7)
const cnn=computed(()=>Math.min(90,61+11*Math.log10(samples.value)+locality.value*8+augmentation.value*3))
const vit=computed(()=>Math.min(92,47+18*Math.log10(samples.value)+(1-locality.value)*5+augmentation.value*10))
const preferred=computed(()=>vit.value>cnn.value?'ViT':'CNN')
</script>
<template><DlExperimentFrame index="10-B" title="归纳偏置证据秤" question="为什么纯 Transformer 在小数据上曾经落后，而大规模预训练后又能反超？" formula="evidence = architecture bias + data scale + training recipe">
<template #visual><div class="scale" role="img" aria-label="CNN 与 ViT 在数据和增强条件下的教学响应"><section><b>CNN</b><i :style="{height:`${cnn*2}px`}"></i><strong>{{cnn.toFixed(1)}}</strong></section><section><b>ViT</b><i :style="{height:`${vit*2}px`}"></i><strong>{{vit.toFixed(1)}}</strong></section></div></template>
<template #controls><label>训练样本（千）= {{samples}}k<input v-model.number="samples" type="range" min="5" max="500" step="5"></label><label>增强/正则强度 = {{augmentation.toFixed(2)}}<input v-model.number="augmentation" type="range" min="0" max="1" step="0.05"></label><label>任务局部性 = {{locality.toFixed(2)}}<input v-model.number="locality" type="range" min="0" max="1" step="0.05"></label></template>
<template #metrics><div><span>CNN 响应代理</span><strong>{{cnn.toFixed(1)}}</strong></div><div><span>ViT 响应代理</span><strong>{{vit.toFixed(1)}}</strong></div><div><span>当前领先</span><strong>{{preferred}}</strong></div><div><span>差值</span><strong>{{Math.abs(vit-cnn).toFixed(1)}}</strong></div></template>
<template #note>曲线是概念响应面，不是 benchmark。卷积内置局部性与平移等变偏置，数据少时更省样本；ViT 的结构假设更弱，通常更依赖预训练规模、增强、蒸馏和正则化。</template>
</DlExperimentFrame></template>
<style scoped>.scale{display:flex;justify-content:center;align-items:end;gap:70px;min-height:280px;padding:24px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.scale section{display:grid;grid-template-rows:auto 1fr auto;align-items:end;justify-items:center;height:230px;color:#dce9ea;font:11px ui-monospace,monospace}.scale i{display:block;width:76px;max-height:185px;background:#58d3c8;transition:height .2s}.scale section+section i{background:#f2c14e}</style>
