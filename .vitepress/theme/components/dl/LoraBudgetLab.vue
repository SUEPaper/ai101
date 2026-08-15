<script setup lang="ts">
import { computed, ref } from 'vue'
const width=ref(4096), layers=ref(32), rank=ref(16), targets=ref(4)
const trainable=computed(()=>layers.value*targets.value*2*width.value*rank.value)
const full=computed(()=>layers.value*targets.value*width.value*width.value)
const percent=computed(()=>100*trainable.value/full.value), adamGiB=computed(()=>trainable.value*12/2**30)
</script>
<template><DlExperimentFrame index="22-A" title="LoRA 参数与状态账本" question="低秩旁路把哪些矩阵设为可训练，rank、目标模块和优化器状态怎样改变成本？" formula="ΔW=BA; trainable≈targets·layers·2dr">
<template #visual><div class="lora"><div>W · frozen</div><b>+</b><div class="rank">B{{rank}} × A{{rank}}</div></div></template>
<template #controls><label>隐藏宽度 = {{width}}<input v-model.number="width" type="range" min="1024" max="8192" step="512"></label><label>层数 = {{layers}}<input v-model.number="layers" type="range" min="8" max="80" step="8"></label><label>Rank = {{rank}}<input v-model.number="rank" type="range" min="2" max="128" step="2"></label><label>目标矩阵/层 = {{targets}}<input v-model.number="targets" type="range" min="1" max="7" step="1"></label></template>
<template #metrics><div><span>可训练参数</span><strong>{{(trainable/1e6).toFixed(1)}}M</strong></div><div><span>相对目标矩阵</span><strong>{{percent.toFixed(3)}}%</strong></div><div><span>Adam 状态代理</span><strong>{{adamGiB.toFixed(2)}} GiB</strong></div><div><span>底座梯度</span><strong>冻结</strong></div></template>
<template #note>LoRA 降低可训练参数和优化器状态，不自动消除底座权重、激活与上下文显存。target modules 和 rank 必须用冻结评测集消融。</template>
</DlExperimentFrame></template>
<style scoped>.lora{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:18px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.lora div{padding:30px;text-align:center;border:1px solid #607487;color:#c8d6da}.lora .rank{border-color:#58d3c8;color:#dff}.lora b{color:#f2c14e}</style>
