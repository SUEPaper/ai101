<script setup lang="ts">
import { computed, ref } from 'vue'
const cin=ref(64), cout=ref(128), kernel=ref(3), spatial=ref(56)
const standard= computed(()=>kernel.value**2*cin.value*cout.value)
const depthwise= computed(()=>kernel.value**2*cin.value+cin.value*cout.value)
const ratio=computed(()=>standard.value/depthwise.value)
const macs=computed(()=>depthwise.value*spatial.value**2/1e6)
</script>
<template><DlExperimentFrame index="05-A" title="深度可分离卷积账本" question="把空间混合与通道混合拆开，究竟省掉了哪一项？" formula="K²CᵢCₒ → K²Cᵢ + CᵢCₒ">
  <template #visual><div class="split" role="img" aria-label="标准卷积和深度可分离卷积连接对比"><section><b>STANDARD</b><span class="cube wide"></span><i></i><span class="cube wide"></span><small>空间 × 通道一次完成</small></section><section><b>DEPTHWISE + POINTWISE</b><span class="cube slices"></span><i></i><span class="cube thin"></span><i></i><span class="cube wide"></span><small>逐通道空间混合，再用 1×1 混合通道</small></section></div></template>
  <template #controls><label>输入通道 Cᵢ = {{cin}}<input v-model.number="cin" type="range" min="16" max="256" step="16"></label><label>输出通道 Cₒ = {{cout}}<input v-model.number="cout" type="range" min="16" max="512" step="16"></label><label>核大小 K = {{kernel}}<input v-model.number="kernel" type="range" min="1" max="7" step="2"></label><label>特征图边长 = {{spatial}}<input v-model.number="spatial" type="range" min="14" max="112" step="14"></label></template>
  <template #metrics><div><span>标准卷积参数</span><strong>{{standard.toLocaleString()}}</strong></div><div><span>可分离卷积参数</span><strong>{{depthwise.toLocaleString()}}</strong></div><div><span>理论压缩比</span><strong>{{ratio.toFixed(2)}}×</strong></div><div><span>可分离 MAC</span><strong>{{macs.toFixed(1)}} M</strong></div></template>
  <template #note>理论 MAC 降低不保证端到端同倍率加速。小 batch、内存访问、kernel launch 和硬件对 depthwise kernel 的实现质量会改变真实延迟。</template>
</DlExperimentFrame></template>
<style scoped>.split{display:grid;grid-template-columns:1fr 1.35fr;gap:18px;min-height:280px;padding:24px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.split section{display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;border:1px solid #27485a}.split b,.split small{width:100%;text-align:center;color:#9eb8bd;font:10px ui-monospace,monospace}.cube{display:block;height:80px;border:2px solid #58d3c8;background:rgba(88,211,200,.15);transform:skewY(-8deg)}.wide{width:64px}.thin{width:18px;border-color:#f2c14e}.slices{width:58px;background:repeating-linear-gradient(90deg,rgba(88,211,200,.25) 0 5px,transparent 5px 8px)}i{width:20px;height:2px;background:#df6c5b}@media(max-width:560px){.split{grid-template-columns:1fr}}</style>
