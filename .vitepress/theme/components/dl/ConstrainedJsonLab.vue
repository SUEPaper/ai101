<script setup lang="ts">
import { computed, ref } from 'vue'
const fields=ref(4), enums=ref(3), constrained=ref(true), error=ref(.12)
const valid=computed(()=>constrained.value?1:Math.pow(1-error.value,fields.value+2)), branches=computed(()=>constrained.value?enums.value:32000)
const retries=computed(()=>valid.value===1?0:(1-valid.value)/valid.value)
</script>
<template><DlExperimentFrame index="26-B" title="JSON Schema 约束解码器" question="语法约束怎样在每一步屏蔽非法 token；格式有效又为何不等于参数语义正确？" formula="allowed tokens = grammar(state) ∩ model vocabulary">
<template #visual><div class="json"><code>{ "fault": "bearing", "severity": 3, "stop": true }</code></div></template>
<template #controls><label>必填字段 = {{fields}}<input v-model.number="fields" type="range" min="1" max="12" step="1"></label><label>枚举分支 = {{enums}}<input v-model.number="enums" type="range" min="2" max="20" step="1"></label><label>无约束逐步错误 = {{error.toFixed(2)}}<input v-model.number="error" type="range" min=".01" max=".4" step=".01"></label><button type="button" @click="constrained=!constrained">{{constrained?'关闭语法约束':'启用语法约束'}}</button></template>
<template #metrics><div><span>格式有效率代理</span><strong>{{(valid*100).toFixed(1)}}%</strong></div><div><span>候选分支代理</span><strong>{{branches}}</strong></div><div><span>期望重试</span><strong>{{retries.toFixed(2)}}</strong></div><div><span>语义校验</span><strong>仍需要</strong></div></template>
<template #note>Grammar/schema 保证可解析结构，不保证设备 ID 存在、单位正确或调用获权；语义验证必须在执行前由可信代码完成。</template>
</DlExperimentFrame></template>
<style scoped>.json{display:grid;place-items:center;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.json code{max-width:100%;padding:25px;border:1px solid #58d3c8;color:#f2c14e;white-space:normal}</style>
