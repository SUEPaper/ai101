<script setup lang="ts">
import { computed, ref } from 'vue'
const params=ref(7), tokens=ref(140), utilization=ref(.4), peta=ref(100)
const flops=computed(()=>6*params.value*1e9*tokens.value*1e9), days=computed(()=>flops.value/(peta.value*1e15*utilization.value)/86400)
const ratio=computed(()=>tokens.value/params.value), loss=computed(()=>1.55+.8/Math.pow(params.value,.22)+1.1/Math.pow(tokens.value/20,.2))
</script>
<template><DlExperimentFrame index="21-A" title="Scaling Law 计算预算台" question="参数、训练 token、峰值算力和利用率怎样共同决定训练时间与欠训练风险？" formula="training FLOPs≈6ND; wall≈FLOPs/(peak·MFU)">
<template #visual><div class="scales"><i :style="{width:`${Math.min(100,params/70*100)}%`}"></i><i :style="{width:`${Math.min(100,tokens/1400*100)}%`}"></i></div></template>
<template #controls><label>参数 = {{params}}B<input v-model.number="params" type="range" min="1" max="70" step="1"></label><label>训练 token = {{tokens}}B<input v-model.number="tokens" type="range" min="20" max="1400" step="20"></label><label>MFU = {{utilization.toFixed(2)}}<input v-model.number="utilization" type="range" min=".1" max=".7" step=".05"></label><label>峰值 = {{peta}} PFLOP/s<input v-model.number="peta" type="range" min="10" max="500" step="10"></label></template>
<template #metrics><div><span>Token/参数</span><strong>{{ratio.toFixed(1)}}</strong></div><div><span>训练 FLOPs</span><strong>{{flops.toExponential(2)}}</strong></div><div><span>理想墙钟</span><strong>{{days.toFixed(2)}} 天</strong></div><div><span>Loss 代理</span><strong>{{loss.toFixed(3)}}</strong></div></template>
<template #note>代理只解释等算力权衡，不预测具体模型质量。Chinchilla 结论依赖数据、优化和推理价值；部署总成本还包含失败实验与推理。</template>
</DlExperimentFrame></template>
<style scoped>.scales{display:grid;align-content:center;gap:42px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.scales i{height:35px;background:#58d3c8}.scales i+i{background:#a88ce8}</style>
