<script setup lang="ts">
import { computed, ref } from 'vue'
const fault=ref<'none'|'args'|'timeout'|'duplicate'|'denied'>('args'), retries=ref(1), idempotent=ref(true), permission=ref('read')
const safe=computed(()=>fault.value==='none'||fault.value==='timeout'&&retries.value>0&&idempotent.value||fault.value==='duplicate'&&idempotent.value||fault.value==='denied')
const state=computed(()=>fault.value==='denied'?'DENIED':safe.value?'RECOVERED':'ESCALATE')
</script>
<template><DlExperimentFrame index="26-A" title="工具调用状态机与故障注入" question="参数错误、超时、重复副作用和权限拒绝如何进入显式状态，而不是让模型自由重试？" formula="plan→validate→authorize→execute→observe→commit">
<template #visual><div class="states"><span>VALIDATE</span><b>→</b><span>AUTHORIZE</span><b>→</b><span :class="{bad:!safe}">{{state}}</span></div></template>
<template #controls><label>故障<select v-model="fault"><option value="none">无故障</option><option value="args">参数错误</option><option value="timeout">超时</option><option value="duplicate">重复副作用</option><option value="denied">权限拒绝</option></select></label><label>重试次数 = {{retries}}<input v-model.number="retries" type="range" min="0" max="5" step="1"></label><button type="button" @click="idempotent=!idempotent">{{idempotent?'关闭幂等键':'启用幂等键'}}</button><label>权限<select v-model="permission"><option value="read">只读</option><option value="write">写入</option><option value="admin">管理员</option></select></label></template>
<template #metrics><div><span>终态</span><strong>{{state}}</strong></div><div><span>自动重试</span><strong>{{fault==='timeout'&&safe?'允许':'否/受限'}}</strong></div><div><span>幂等保护</span><strong>{{idempotent?'启用':'关闭'}}</strong></div><div><span>最小权限</span><strong>{{permission}}</strong></div></template>
<template #note>真正系统必须在模型外验证 schema、权限、预算和幂等；超时结果未知时盲目重试写操作可能产生重复副作用。</template>
</DlExperimentFrame></template>
<style scoped>.states{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:12px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.states span{padding:25px 5px;text-align:center;border:1px solid #58d3c8;color:#dff}.states .bad{border-color:#df6c5b;color:#df6c5b}.states b{color:#f2c14e}@media(max-width:520px){.states{grid-template-columns:1fr}.states b{text-align:center}}</style>
