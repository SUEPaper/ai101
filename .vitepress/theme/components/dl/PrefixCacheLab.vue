<script setup lang="ts">
import { computed, ref } from 'vue'
const requests=ref(16), prefix=ref(1800), unique=ref(300), hit=ref(.75), mode=ref<'paged'|'radix'>('radix')
const saved=computed(()=>requests.value*prefix.value*hit.value), compute=computed(()=>requests.value*(prefix.value+unique.value)-saved.value)
const fragmentation=computed(()=>mode.value==='paged'?.08:.04), reuse=computed(()=>saved.value/(requests.value*(prefix.value+unique.value)))
</script>
<template><DlExperimentFrame index="32-B" title="Paged KV—Radix Prefix 复用台" question="分页 KV 管理与共享前缀缓存分别解决碎片、容量和重复 prefill 的什么问题？" formula="saved prefill≈requests·shared prefix·hit rate">
<template #visual><div class="prefix"><div>SHARED · {{prefix}}</div><span v-for="i in Math.min(requests,24)" :key="i">+{{unique}}</span></div></template>
<template #controls><label>请求数 = {{requests}}<input v-model.number="requests" type="range" min="1" max="64" step="1"></label><label>共享前缀 = {{prefix}}<input v-model.number="prefix" type="range" min="0" max="8000" step="100"></label><label>命中率 = {{hit.toFixed(2)}}<input v-model.number="hit" type="range" min="0" max="1" step=".05"></label><label>缓存视角<select v-model="mode"><option value="paged">PagedAttention</option><option value="radix">RadixAttention</option></select></label></template>
<template #metrics><div><span>节省 prefill token</span><strong>{{saved.toFixed(0)}}</strong></div><div><span>剩余计算 token</span><strong>{{compute.toFixed(0)}}</strong></div><div><span>复用比例</span><strong>{{(reuse*100).toFixed(1)}}%</strong></div><div><span>碎片代理</span><strong>{{(fragmentation*100).toFixed(1)}}%</strong></div></template>
<template #note>PagedAttention 核心是近似虚拟内存式 KV 分页；RadixAttention 进一步组织共享前缀。缓存命中还受 tokenizer、模板、adapter、权限、版本和租户隔离约束。</template>
</DlExperimentFrame></template>
<style scoped>.prefix{display:grid;grid-template-columns:repeat(8,1fr);align-content:center;gap:7px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.prefix div{grid-column:1/-1;padding:18px;border:1px solid #f2c14e;color:#f2c14e}.prefix span{padding:13px 3px;text-align:center;border:1px solid #58d3c8;color:#dff}</style>
