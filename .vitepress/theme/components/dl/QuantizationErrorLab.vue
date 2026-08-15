<script setup lang="ts">
import { computed, ref } from 'vue'
const params=ref(7), bits=ref(4), group=ref(64), outliers=ref(.01)
const memory=computed(()=>params.value*1e9*bits.value/8/2**30), scales=computed(()=>params.value*1e9/group.value*2/2**30)
const error=computed(()=>Math.pow(2,-bits.value/2)*(1+outliers.value*20)*Math.sqrt(group.value/64))
</script>
<template><DlExperimentFrame index="22-B" title="QLoRA 量化误差显微镜" question="4-bit 底座如何节省权重显存；group size、scale 与离群值又在哪里留下误差？" formula="weight bytes≈N·bits/8 + group scales">
<template #visual><div class="quant"><span v-for="i in Math.max(4,bits)" :key="i" :style="{opacity:.25+i/Math.max(4,bits)}"></span></div></template>
<template #controls><label>底座参数 = {{params}}B<input v-model.number="params" type="range" min="1" max="70" step="1"></label><label>位宽<select v-model.number="bits"><option :value="4">4-bit</option><option :value="8">8-bit</option><option :value="16">16-bit</option></select></label><label>Group size = {{group}}<input v-model.number="group" type="range" min="32" max="256" step="32"></label><label>离群比例 = {{outliers.toFixed(3)}}<input v-model.number="outliers" type="range" min="0" max=".1" step=".005"></label></template>
<template #metrics><div><span>权重载荷</span><strong>{{memory.toFixed(2)}} GiB</strong></div><div><span>Scale 载荷</span><strong>{{scales.toFixed(2)}} GiB</strong></div><div><span>误差代理</span><strong>{{error.toFixed(4)}}</strong></div><div><span>LoRA 计算</span><strong>高精度</strong></div></template>
<template #note>QLoRA 通常以 NF4、double quantization 和 paged optimizer 组合；位宽标签不足以复现，必须记录量化类型、group、compute dtype 与 kernel。</template>
</DlExperimentFrame></template>
<style scoped>.quant{display:flex;align-items:center;justify-content:center;gap:9px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.quant span{width:24px;height:160px;border:1px solid #58d3c8;background:#58d3c8}</style>
