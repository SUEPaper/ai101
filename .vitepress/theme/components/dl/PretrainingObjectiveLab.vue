<script setup lang="ts">
import { computed, ref } from 'vue'
const mode=ref<'causal'|'mlm'|'span'>('causal'), length=ref(12), corrupt=ref(.15)
const visible=computed(()=>mode.value==='causal'?Math.round(length.value*(length.value+1)/2):mode.value==='mlm'?Math.round(length.value*length.value*(1-corrupt.value)):Math.round(length.value*length.value*.72))
const targets=computed(()=>mode.value==='causal'?length.value-1:Math.max(1,Math.round(length.value*corrupt.value)))
const family=computed(()=>mode.value==='causal'?'decoder-only':mode.value==='mlm'?'encoder-only':'encoder–decoder')
</script>
<template><DlExperimentFrame index="19-A" title="预训练目标遮罩台" question="同一段 token 在 causal LM、MLM 和 span corruption 中，哪些位置可见、哪些位置承担 loss？" formula="objective = corruption × visibility × target positions">
<template #visual><div class="mask"><span v-for="i in length" :key="i" :class="{hidden:mode!=='causal'&&i%Math.max(2,Math.round(1/corrupt))===0,target:mode==='causal'&&i===length}">x{{i}}</span></div></template>
<template #controls><label>目标<select v-model="mode"><option value="causal">Causal LM</option><option value="mlm">Masked LM</option><option value="span">Span corruption</option></select></label><label>序列长度 = {{length}}<input v-model.number="length" type="range" min="6" max="24" step="2"></label><label>破坏比例 = {{corrupt.toFixed(2)}}<input v-model.number="corrupt" type="range" min=".05" max=".5" step=".05"></label></template>
<template #metrics><div><span>模型接口</span><strong>{{family}}</strong></div><div><span>Loss 位置</span><strong>{{targets}}</strong></div><div><span>可见关系代理</span><strong>{{visible}}</strong></div><div><span>直接生成</span><strong>{{mode==='causal'?'是':'需改造'}}</strong></div></template>
<template #note>MLM 能读左右文但不是归一化的左到右生成概率；T5 把被遮 span 变成 decoder target。公平比较必须统一有效 token 与计算预算。</template>
</DlExperimentFrame></template>
<style scoped>.mask{display:flex;flex-wrap:wrap;align-content:center;gap:8px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.mask span{padding:12px 9px;border:1px solid #58d3c8;color:#dff}.mask .hidden{border-color:#a88ce8;color:#a88ce8;background:#281f47}.mask .target{border-color:#f2c14e;color:#f2c14e}</style>
