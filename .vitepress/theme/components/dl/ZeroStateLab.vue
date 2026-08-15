<script setup lang="ts">
import { computed, ref } from 'vue'
const params=ref(70), gpus=ref(8), stage=ref(3), activation=ref(12)
const full=computed(()=>params.value*16), shardFactor=computed(()=>stage.value===0?1:stage.value===1?(params.value*12/gpus.value+params.value*4)/full.value:stage.value===2?(params.value*14/gpus.value+params.value*2)/full.value:1/gpus.value)
const stateGiB=computed(()=>full.value*shardFactor.value), total=computed(()=>stateGiB.value+activation.value)
const comm=computed(()=>stage.value===0?1:stage.value===1?1.2:stage.value===2?1.6:2.2)
</script>
<template><DlExperimentFrame index="31-B" title="ZeRO/FSDP 模型状态分片账本" question="optimizer、gradient、parameter 逐级分片时，单卡状态与通信怎样交换？" formula="mixed Adam state≈16 bytes/param before sharding">
<template #visual><div class="shards"><span v-for="i in gpus" :key="i" :class="{active:i<=Math.max(1,gpus/(stage+1))}">G{{i}}</span></div></template>
<template #controls><label>参数 = {{params}}B<input v-model.number="params" type="range" min="1" max="175" step="1"></label><label>GPU = {{gpus}}<input v-model.number="gpus" type="range" min="2" max="64" step="2"></label><label>ZeRO stage<select v-model.number="stage"><option :value="0">0：复制全部</option><option :value="1">1：分 optimizer</option><option :value="2">2：再分 gradient</option><option :value="3">3/FSDP FULL_SHARD</option></select></label><label>激活 = {{activation}} GB<input v-model.number="activation" type="range" min="1" max="80" step="1"></label></template>
<template #metrics><div><span>单卡模型状态</span><strong>{{stateGiB.toFixed(1)}} GB</strong></div><div><span>加激活</span><strong>{{total.toFixed(1)}} GB</strong></div><div><span>通信代理</span><strong>{{comm.toFixed(1)}}×</strong></div><div><span>参数 materialize</span><strong>{{stage===3?'按层':'常驻/部分'}}</strong></div></template>
<template #note>16 B/param 是典型混合精度 Adam 粗估，框架 flat/shard、master weight、offload、padding 和 workspace 会改变账本。FSDP strategy 不应只用 ZeRO stage 名替代。</template>
</DlExperimentFrame></template>
<style scoped>.shards{display:grid;grid-template-columns:repeat(8,1fr);align-content:center;gap:8px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.shards span{padding:18px 2px;text-align:center;border:1px solid #536979;color:#91a6ad}.shards .active{border-color:#58d3c8;color:#dff;background:#173948}</style>
