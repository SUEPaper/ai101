<script setup lang="ts">
import { computed, ref } from 'vue'
const overlap=ref(35), confidence=ref(.7), voidThreshold=ref(.25)
const conflict=computed(()=>overlap.value*(1-confidence.value))
const voidArea=computed(()=>Math.max(0,conflict.value-voidThreshold.value*20))
const pq=computed(()=>Math.max(0,.88-conflict.value/110-voidArea.value/100))
</script>
<template><DlExperimentFrame index="09-A" title="全景装配台" question="thing 实例与 stuff 语义发生像素冲突时，怎样生成互斥且可评估的全景 ID？" formula="panoptic id = category_id × divisor + instance_id">
<template #visual><svg viewBox="0 0 540 280" role="img" aria-label="全景分割中实例和背景区域装配"><rect x="28" y="30" width="484" height="220" class="stuff"/><path d="M82 208 C96 85 188 66 250 170 L250 232 H70Z" class="thingA"/><path :d="`M${260-overlap} 222 C${250-overlap} 96 ${390-overlap} 66 ${440-overlap} 205 L${440-overlap} 232 H${260-overlap}Z`" class="thingB"/><path v-if="voidArea>0" :d="`M235 118 L${235+voidArea*3} 105 L${248+voidArea*3} 218 L232 223Z`" class="void"/><text x="90" y="58">stuff</text><text x="145" y="188">instance 1</text><text :x="350-overlap" y="188">instance 2</text></svg></template>
<template #controls><label>实例重叠 = {{overlap}}%<input v-model.number="overlap" type="range" min="0" max="80" step="5"></label><label>mask 置信度 = {{confidence.toFixed(2)}}<input v-model.number="confidence" type="range" min="0.2" max="1" step="0.05"></label><label>void 容忍 = {{voidThreshold.toFixed(2)}}<input v-model.number="voidThreshold" type="range" min="0" max="0.8" step="0.05"></label></template>
<template #metrics><div><span>冲突像素代理</span><strong>{{conflict.toFixed(1)}}%</strong></div><div><span>void 区域代理</span><strong>{{voidArea.toFixed(1)}}%</strong></div><div><span>PQ 代理</span><strong>{{pq.toFixed(3)}}</strong></div><div><span>输出 ID 数</span><strong>{{voidArea>0?4:3}}</strong></div></template>
<template #note>全景分割要求每个像素恰属一个片段；PQ 同时包含识别质量与匹配后掩码质量。实际装配必须固定 mask 排序、重叠处理、最小面积和 void 规则。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.stuff{fill:#203b50}.thingA{fill:rgba(88,211,200,.75);stroke:#79ede2}.thingB{fill:rgba(242,193,78,.72);stroke:#ffd86f}.void{fill:#df6c5b;stroke:#ff9c8d}text{fill:#fff;font:11px ui-monospace,monospace}</style>
