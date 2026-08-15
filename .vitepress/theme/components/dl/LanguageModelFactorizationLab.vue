<script setup lang="ts">
import { computed, ref } from 'vue'
const context=ref(4), vocabulary=ref(32000), entropy=ref(3.2), tokens=ref(24)
const likelihood=computed(()=>Math.exp(-entropy.value*tokens.value)), perplexity=computed(()=>Math.exp(entropy.value))
const edges=computed(()=>Array.from({length:tokens.value},(_,i)=>Math.min(i,context.value)).reduce((a,b)=>a+b,0))
</script>
<template><DlExperimentFrame index="18-A" title="语言模型因子分解轨道" question="联合序列概率如何变成逐 token 条件概率；上下文截断又删除了哪些依赖？" formula="p(x₁:T)=∏ₜp(xₜ|x<t), PPL=exp(mean NLL)">
<template #visual><div class="sequence" role="img" aria-label="自回归语言模型上下文边"><span v-for="i in Math.min(tokens,24)" :key="i" :class="i>Math.min(tokens,24)-context?'context':''">x{{i}}</span></div></template>
<template #controls><label>上下文窗口 = {{context}}<input v-model.number="context" type="range" min="1" max="16" step="1"></label><label>序列 token = {{tokens}}<input v-model.number="tokens" type="range" min="4" max="64" step="4"></label><label>平均 NLL = {{entropy.toFixed(2)}}<input v-model.number="entropy" type="range" min="1" max="8" step=".1"></label><label>词表 = {{vocabulary}}<input v-model.number="vocabulary" type="range" min="1000" max="64000" step="1000"></label></template>
<template #metrics><div><span>Perplexity</span><strong>{{perplexity.toFixed(2)}}</strong></div><div><span>序列 likelihood</span><strong>{{likelihood.toExponential(2)}}</strong></div><div><span>可见依赖边</span><strong>{{edges}}</strong></div><div><span>输出 logits/步</span><strong>{{vocabulary.toLocaleString()}}</strong></div></template>
<template #note>perplexity 只在相同 tokenizer、数据和归一化协议下可比。更长上下文增加可用条件，不保证模型真正利用远距证据。</template>
</DlExperimentFrame></template>
<style scoped>.sequence{display:flex;flex-wrap:wrap;align-content:center;gap:7px;min-height:280px;padding:24px;border:1px solid #2c5f65;border-radius:7px;background:#0b172a}.sequence span{padding:10px 8px;border-bottom:2px solid #3b5665;color:#89a3aa;font:10px ui-monospace,monospace}.sequence .context{border-color:#f2c14e;color:#fff;background:#203a48}</style>
