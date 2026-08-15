<script setup lang="ts">
import { computed, ref } from 'vue'
const params=ref(7), batch=ref(4), sequence=ref(4096), layers=ref(32), mode=ref<'train'|'prefill'|'decode'>('train')
const weights=computed(()=>params.value*2), states=computed(()=>mode.value==='train'?params.value*14:0)
const activation=computed(()=>mode.value==='train'?batch.value*sequence.value*layers.value*.0008:mode.value==='prefill'?batch.value*sequence.value*layers.value*.0003:0)
const kv=computed(()=>mode.value==='decode'?batch.value*sequence.value*layers.value*.00012:0)
const total=computed(()=>weights.value+states.value+activation.value+kv.value)
</script>
<template><DlExperimentFrame index="29-B" title="训练—Prefill—Decode 显存账本" question="相同模型在训练、prefill 和 decode 中为何由不同状态主导容量与带宽？" formula="memory=weights+grad+optimizer+activations+KV+workspace">
<template #visual><div class="memory"><span :style="{flex:weights}">W</span><span :style="{flex:states}">OPT</span><span :style="{flex:activation}">ACT</span><span :style="{flex:kv}">KV</span></div></template>
<template #controls><label>阶段<select v-model="mode"><option value="train">训练</option><option value="prefill">Prefill</option><option value="decode">Decode</option></select></label><label>参数 = {{params}}B<input v-model.number="params" type="range" min="1" max="70" step="1"></label><label>Batch = {{batch}}<input v-model.number="batch" type="range" min="1" max="64" step="1"></label><label>序列 = {{sequence}}<input v-model.number="sequence" type="range" min="512" max="32768" step="512"></label></template>
<template #metrics><div><span>权重代理</span><strong>{{weights.toFixed(1)}} GB</strong></div><div><span>训练状态</span><strong>{{states.toFixed(1)}} GB</strong></div><div><span>激活</span><strong>{{activation.toFixed(1)}} GB</strong></div><div><span>总量代理</span><strong>{{total.toFixed(1)}} GB</strong></div></template>
<template #note>系数是教学代理，不代替精确框架账本；需加入 dtype、tensor parallel、checkpoint、allocator、临时 workspace 与 KV heads。训练/推理硬件选择不能只看同一个“显存”。</template>
</DlExperimentFrame></template>
<style scoped>.memory{display:flex;align-items:stretch;min-width:0;min-height:280px;overflow:hidden;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.memory span{min-width:0;display:grid;place-items:center;overflow:hidden;border:1px solid #0b172a;background:#58d3c8;color:#08212a;font-weight:800}.memory span+span{background:#a88ce8}.memory span+span+span{background:#f2c14e}.memory span+span+span+span{background:#df6c5b}</style>
