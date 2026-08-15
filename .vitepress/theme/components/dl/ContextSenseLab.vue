<script setup lang="ts">
import { computed, ref } from 'vue'
const sentence=ref<'river'|'finance'>('river'), layer=ref(4), bidirectional=ref(true)
const sentences={river:'他坐在河岸边看水流',finance:'银行正在审核企业贷款'}
const target=computed(()=>sentence.value==='river'?'岸':'银行')
const senses=computed(()=>sentence.value==='river'?{nature:.78,finance:.08}:{nature:.05,finance:.86})
const contextuality=computed(()=>Math.min(1,.25+layer.value*.07+(bidirectional.value?.18:0)))
</script>
<template><DlExperimentFrame index="18-B" title="上下文词义显微镜" question="静态词向量给一个词一个点，上下文编码器怎样让同一表面形式随句子改变表示？" formula="hₜ=f(x₁:T,t); static e(word) → contextual h(word,context)">
<template #visual><div class="sense" role="img" aria-label="句子上下文与词义响应"><p>{{sentences[sentence]}}</p><div><span>自然/地点</span><i :style="{width:`${senses.nature*contextuality*100}%`}"></i></div><div><span>金融/机构</span><i :style="{width:`${senses.finance*contextuality*100}%`}"></i></div></div></template>
<template #controls><label>语境<select v-model="sentence"><option value="river">河岸语境</option><option value="finance">银行语境</option></select></label><label>编码层 = {{layer}}<input v-model.number="layer" type="range" min="0" max="10" step="1"></label><button type="button" @click="bidirectional=!bidirectional">{{bidirectional?'改为只看左侧':'恢复双向上下文'}}</button></template>
<template #metrics><div><span>目标片段</span><strong>{{target}}</strong></div><div><span>上下文化强度</span><strong>{{contextuality.toFixed(2)}}</strong></div><div><span>主导词义</span><strong>{{senses.finance>senses.nature?'金融':'自然'}}</strong></div><div><span>上下文方向</span><strong>{{bidirectional?'双向':'因果/左侧'}}</strong></div></template>
<template #note>数值是教学响应，不是预训练模型输出。上下文表示依赖层、目标函数和 token 边界；BERT 式双向编码不能直接用作无泄漏的自回归生成。</template>
</DlExperimentFrame></template>
<style scoped>.sense{display:grid;align-content:center;gap:18px;min-height:280px;padding:24px;border:1px solid #2c5f65;border-radius:7px;background:#0b172a}.sense p{color:#fff;font:22px serif}.sense div{display:grid;grid-template-columns:85px 1fr;color:#b7cbce;font-size:11px}.sense i{height:18px;background:#58d3c8}.sense div+div i{background:#a88ce8}</style>
