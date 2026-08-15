<script setup lang="ts">
import { computed, ref } from 'vue'
const flops=ref(200), bytes=ref(20), peak=ref(1000), bandwidth=ref(3000)
const intensity=computed(()=>flops.value/bytes.value), ridge=computed(()=>peak.value/bandwidth.value)
const attainable=computed(()=>Math.min(peak.value,bandwidth.value*intensity.value)), bottleneck=computed(()=>intensity.value<ridge.value?'内存带宽':'计算')
</script>
<template><DlExperimentFrame index="29-A" title="Roofline 算力—带宽判别台" question="一个算子在给定精度下是受峰值 FLOPS 约束，还是受 HBM 字节搬运约束？" formula="performance≤min(peak FLOP/s, bandwidth·arithmetic intensity)">
<template #visual><div class="roof"><i :style="{width:`${Math.min(100,attainable/peak*100)}%`}"></i><b>{{bottleneck}}</b></div></template>
<template #controls><label>计算 = {{flops}} GFLOP<input v-model.number="flops" type="range" min="10" max="2000" step="10"></label><label>HBM 数据 = {{bytes}} GB<input v-model.number="bytes" type="range" min="1" max="200" step="1"></label><label>峰值 = {{peak}} TFLOP/s<input v-model.number="peak" type="range" min="50" max="2000" step="50"></label><label>带宽 = {{bandwidth}} GB/s<input v-model.number="bandwidth" type="range" min="200" max="6000" step="100"></label></template>
<template #metrics><div><span>算术强度</span><strong>{{intensity.toFixed(2)}} FLOP/B</strong></div><div><span>Ridge point</span><strong>{{ridge.toFixed(3)}}</strong></div><div><span>可达性能代理</span><strong>{{attainable.toFixed(1)}}</strong></div><div><span>瓶颈</span><strong>{{bottleneck}}</strong></div></template>
<template #note>单位在教学响应面中归一到兼容尺度。真实 roofline 要按实际 dtype、读写重用、缓存命中和测得 bandwidth/peak 计算，而非引用营销峰值。</template>
</DlExperimentFrame></template>
<style scoped>.roof{display:grid;align-content:center;gap:28px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.roof i{height:38px;background:#58d3c8}.roof b{color:#f2c14e;font:24px ui-monospace,monospace}</style>
