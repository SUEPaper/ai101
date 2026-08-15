<script setup lang="ts">
import { computed, ref } from 'vue'
const windowSize=ref(4), shift=ref(true), stages=ref(2)
const grid=8
const groups=computed(()=>{const offset=shift.value?Math.floor(windowSize.value/2):0;return Array.from({length:grid*grid},(_,n)=>{const x=n%grid,y=Math.floor(n/grid);return (Math.floor(((x+offset)%grid)/windowSize.value)+Math.floor(((y+offset)%grid)/windowSize.value)*Math.ceil(grid/windowSize.value))%5})})
const reachable=computed(()=>shift.value?Math.min(64,windowSize.value**2*stages.value*1.8):windowSize.value**2)
</script>
<template><DlExperimentFrame index="11-A" title="移位窗口连通台" question="局部窗口把注意力从平方复杂度降下来后，跨窗口信息怎样交换？" formula="W-MSA → cyclic shift → SW-MSA → patch merging">
<template #visual><div class="windows" role="img" aria-label="Swin Transformer 窗口分组"><span v-for="(group,n) in groups" :key="n" :class="`g${group}`"></span></div></template>
<template #controls><label>窗口边长 = {{windowSize}}<input v-model.number="windowSize" type="range" min="2" max="4" step="2"></label><label>连续 block = {{stages}}<input v-model.number="stages" type="range" min="1" max="6" step="1"></label><button type="button" @click="shift=!shift">{{shift?'关闭移位':'开启半窗移位'}}</button></template>
<template #metrics><div><span>单窗 token</span><strong>{{windowSize**2}}</strong></div><div><span>窗口数</span><strong>{{(grid/windowSize)**2}}</strong></div><div><span>可达 token 代理</span><strong>{{reachable.toFixed(0)}} / 64</strong></div><div><span>跨窗通道</span><strong>{{shift?'已建立':'未建立'}}</strong></div></template>
<template #note>移位不是把全局注意力恢复回来，而是让相邻 block 的分区边界错开。工程实现常用 cyclic shift 与 attention mask 保持规则张量；层级 patch merging 则产生检测和分割需要的多尺度特征。</template>
</DlExperimentFrame></template>
<style scoped>.windows{display:grid;grid-template-columns:repeat(8,1fr);gap:3px;min-height:280px;padding:24px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.windows span{min-height:24px;border:1px solid #274b5d}.g0{background:#205b61}.g1{background:#665524}.g2{background:#3b4e71}.g3{background:#60383b}.g4{background:#385246}</style>
