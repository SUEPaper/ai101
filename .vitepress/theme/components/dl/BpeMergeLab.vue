<script setup lang="ts">
import { computed, ref } from 'vue'
const merges=ref(4), mode=ref<'bpe'|'byte'>('bpe')
const base=['l','o','w','e','r','▁','n','e','w','e','s','t']
const rules=[['e','r','er'],['n','e','ne'],['ne','w','new'],['l','o','lo'],['lo','w','low'],['low','er','lower'],['new','e','newe'],['newe','s','newes']]
const tokens=computed(()=>{let out=[...base];for(const [a,b,c] of rules.slice(0,merges.value)){for(let i=0;i<out.length-1;i++)if(out[i]===a&&out[i+1]===b){out.splice(i,2,c);i--}}return mode.value==='byte'?[...new TextEncoder().encode('lower newest')].map(x=>x.toString(16)):out})
const compression=computed(()=>base.length/tokens.value.length)
</script>
<template><DlExperimentFrame index="16-B" title="BPE 合并传送带" question="频繁相邻单元逐轮合并时，词表大小、序列长度与未知词处理怎样交换？" formula="vocab←vocab∪{ab}; corpus: a b → ab">
<template #visual><div class="belt" role="img" aria-label="BPE token 合并结果"><span v-for="(token,i) in tokens" :key="i">{{token}}</span></div></template>
<template #controls><label>表示方式<select v-model="mode"><option value="bpe">字符/子词 BPE</option><option value="byte">UTF-8 byte fallback</option></select></label><label>应用 merge 数 = {{merges}}<input v-model.number="merges" type="range" min="0" max="8" step="1"></label></template>
<template #metrics><div><span>原始单元</span><strong>{{base.length}}</strong></div><div><span>当前 token</span><strong>{{tokens.length}}</strong></div><div><span>压缩倍率</span><strong>{{compression.toFixed(2)}}×</strong></div><div><span>已用 merge</span><strong>{{mode==='byte'?0:merges}}</strong></div></template>
<template #note>BPE 学习的是语料统计，不是语言学“正确词”。byte fallback 可保证开放词表覆盖，却可能把罕见文字变成长序列；特殊 token 与空白前缀也必须写进协议。</template>
</DlExperimentFrame></template>
<style scoped>.belt{display:flex;flex-wrap:wrap;align-content:center;gap:8px;min-height:280px;padding:24px;border:1px solid #2c5f65;border-radius:7px;background:#0b172a}.belt span{padding:13px 10px;border:1px solid #58d3c8;border-radius:3px;color:#e2f5f3;background:#173a47;font:12px ui-monospace,monospace}</style>
