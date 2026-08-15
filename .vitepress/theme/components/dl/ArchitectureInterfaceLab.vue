<script setup lang="ts">
import { computed, ref } from 'vue'
const kind=ref<'encoder'|'decoder'|'encdec'>('decoder'), input=ref(512), output=ref(128)
const attention=computed(()=>kind.value==='encoder'?input.value**2:kind.value==='decoder'?(input.value+output.value)**2:input.value**2+output.value**2+input.value*output.value)
const use=computed(()=>kind.value==='encoder'?'分类/抽取/检索':kind.value==='decoder'?'开放式续写/对话':'条件生成/转换')
</script>
<template><DlExperimentFrame index="19-B" title="三种架构接口路由器" question="Encoder-only、decoder-only 与 encoder–decoder 怎样分配输入理解、输出生成和注意力成本？" formula="encoder: n²; decoder: (n+m)²; enc-dec: n²+m²+nm">
<template #visual><div class="route"><div>INPUT · {{input}}</div><b>{{kind==='encoder'?'↔':kind==='decoder'?'→':'⇢ ENCODE ⇢ DECODE ⇢'}}</b><div>OUTPUT · {{kind==='encoder'?'HEAD':output}}</div></div></template>
<template #controls><label>接口<select v-model="kind"><option value="encoder">Encoder-only</option><option value="decoder">Decoder-only</option><option value="encdec">Encoder–decoder</option></select></label><label>输入 token = {{input}}<input v-model.number="input" type="range" min="64" max="2048" step="64"></label><label>输出 token = {{output}}<input v-model.number="output" type="range" min="16" max="1024" step="16"></label></template>
<template #metrics><div><span>典型任务</span><strong>{{use}}</strong></div><div><span>Attention 对数代理</span><strong>{{attention.toLocaleString()}}</strong></div><div><span>双向输入</span><strong>{{kind==='decoder'?'否':'是'}}</strong></div><div><span>自回归输出</span><strong>{{kind==='encoder'?'否':'是'}}</strong></div></template>
<template #note>成本只是 dense-attention 代理，不含层数、隐藏维、KV cache 和实现常数；选架构应从信息边界和任务接口出发。</template>
</DlExperimentFrame></template>
<style scoped>.route{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:18px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.route div{padding:32px 10px;text-align:center;border:1px solid #58d3c8;color:#dff}.route b{color:#f2c14e}@media(max-width:520px){.route{grid-template-columns:1fr}.route b{text-align:center}}</style>
