<script setup lang="ts">
import { computed, ref } from 'vue'
const parametric=ref(.75), retrieved=ref(.82), authority=ref(.8), fresh=ref(true)
const evidence=computed(()=>retrieved.value*authority.value*(fresh.value?1:.7))
const conflict=computed(()=>Math.abs(parametric.value-evidence.value)), decision=computed(()=>evidence.value>.55?(conflict.value>.25?'引用并提示冲突':'按证据回答'):'拒答/继续检索')
</script>
<template><DlExperimentFrame index="25-B" title="参数记忆—检索证据冲突台" question="模型先验与最新手册矛盾时，系统怎样基于权威、时效和可引用性回答或拒答？" formula="answer=f(retrieval relevance, authority, freshness, conflict)">
<template #visual><div class="conflict"><i :style="{width:`${parametric*100}%`}"></i><i :style="{width:`${evidence*100}%`}"></i></div></template>
<template #controls><label>参数记忆置信 = {{parametric.toFixed(2)}}<input v-model.number="parametric" type="range" min="0" max="1" step=".05"></label><label>检索相关性 = {{retrieved.toFixed(2)}}<input v-model.number="retrieved" type="range" min="0" max="1" step=".05"></label><label>来源权威 = {{authority.toFixed(2)}}<input v-model.number="authority" type="range" min="0" max="1" step=".05"></label><button type="button" @click="fresh=!fresh">{{fresh?'标记为过期':'恢复最新证据'}}</button></template>
<template #metrics><div><span>证据强度</span><strong>{{evidence.toFixed(3)}}</strong></div><div><span>冲突幅度</span><strong>{{conflict.toFixed(3)}}</strong></div><div><span>决策</span><strong>{{decision}}</strong></div><div><span>可引用</span><strong>{{retrieved>.5?'是':'弱'}}</strong></div></template>
<template #note>模型置信不是事实概率。高风险系统应把检索文档、版本、引用 span 和拒答理由作为输出制品，并在冲突时升级人工。</template>
</DlExperimentFrame></template>
<style scoped>.conflict{display:grid;align-content:center;gap:36px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.conflict i{height:30px;background:#a88ce8}.conflict i+i{background:#58d3c8}</style>
