<script setup lang="ts">
import { computed, ref } from 'vue'
const position=ref(.12), verbosity=ref(.18), injection=ref(false), swapped=ref(false)
const raw=computed(()=>.52+(swapped.value?-position.value:position.value)+verbosity.value+(injection.value?.25:0))
const choice=computed(()=>raw.value>.5?(swapped.value?'A':'B'):(swapped.value?'B':'A'))
const bias=computed(()=>Math.abs(position.value)+Math.abs(verbosity.value)+(injection.value?.25:0))
</script>
<template><DlExperimentFrame index="28-B" title="LLM-as-a-Judge 偏置扫描台" question="答案顺序、冗长度与提示注入怎样让 judge 偏离内容质量？" formula="observed preference=quality+position+style+injection">
<template #visual><div class="judge"><span :class="{win:choice==='A'}">ANSWER A</span><span :class="{win:choice==='B'}">ANSWER B</span></div></template>
<template #controls><label>位置偏差 = {{position.toFixed(2)}}<input v-model.number="position" type="range" min="-.3" max=".3" step=".02"></label><label>冗长偏差 = {{verbosity.toFixed(2)}}<input v-model.number="verbosity" type="range" min="-.3" max=".4" step=".02"></label><button type="button" @click="swapped=!swapped">{{swapped?'恢复原顺序':'交换 A/B 顺序'}}</button><button type="button" @click="injection=!injection">{{injection?'移除注入':'加入 judge 注入'}}</button></template>
<template #metrics><div><span>Judge 选择</span><strong>{{choice}}</strong></div><div><span>偏置载荷代理</span><strong>{{bias.toFixed(2)}}</strong></div><div><span>顺序已交换</span><strong>{{swapped?'是':'否'}}</strong></div><div><span>注入</span><strong>{{injection?'存在':'无'}}</strong></div></template>
<template #note>LLM judge 要盲化模型名、随机/双向换序、限制格式、隔离被评内容中的指令，并用人工子集校准；它不能成为唯一真值源。</template>
</DlExperimentFrame></template>
<style scoped>.judge{display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:18px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.judge span{padding:55px 8px;text-align:center;border:1px solid #5b7180;color:#aebfc5}.judge .win{border-color:#58d3c8;color:#dff;background:#173948}</style>
