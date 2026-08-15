<script setup lang="ts">
import { computed, ref } from 'vue'
const op=ref<'vector'|'softmax'|'gemm'|'attention'>('softmax'), language=ref<'cuda'|'triton'|'tilelang'>('triton'), custom=ref(false)
const traits={cuda:{control:1,dev:.25,portable:.45},triton:{control:.72,dev:.8,portable:.68},tilelang:{control:.82,dev:.68,portable:.62}}
const fit=computed(()=>{const t=traits[language.value];const complex=op.value==='vector'?.2:op.value==='softmax'?.45:op.value==='gemm'?.8:1;return Math.min(1,.35*t.dev+.4*t.control*complex+.25*t.portable+(custom.value?.08:0))})
</script>
<template><DlExperimentFrame index="30-B" title="CUDA—Triton—TileLang 选择矩阵" question="三种编程层级在不同算子上怎样交换控制力、开发速度、布局表达和生态？" formula="choice=f(operation, hardware, control, compiler, team)">
<template #visual><div class="choice"><span :class="{on:language==='cuda'}">CUDA</span><span :class="{on:language==='triton'}">TRITON</span><span :class="{on:language==='tilelang'}">TILELANG</span></div></template>
<template #controls><label>算子<select v-model="op"><option value="vector">Vector Add</option><option value="softmax">Fused Softmax</option><option value="gemm">GEMM/Dequant</option><option value="attention">Flash/MLA</option></select></label><label>实现<select v-model="language"><option value="cuda">CUDA</option><option value="triton">Triton</option><option value="tilelang">TileLang</option></select></label><button type="button" @click="custom=!custom">{{custom?'恢复通用编译':'加入硬件特化'}}</button></template>
<template #metrics><div><span>适配代理</span><strong>{{fit.toFixed(2)}}</strong></div><div><span>控制力</span><strong>{{traits[language].control.toFixed(2)}}</strong></div><div><span>开发速度</span><strong>{{traits[language].dev.toFixed(2)}}</strong></div><div><span>正确性基线</span><strong>库/PyTorch</strong></div></template>
<template #note>数值不是 benchmark。CUDA 是完整平台与底层模型，Triton 是 program/tile 编译抽象，TileLang 强调可组合 tile/dataflow；选择必须绑定版本、硬件、形状和团队。</template>
</DlExperimentFrame></template>
<style scoped>.choice{display:grid;grid-template-columns:repeat(3,1fr);align-items:center;gap:14px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.choice span{padding:42px 4px;text-align:center;border:1px solid #536979;color:#9eb1b8}.choice .on{border-color:#58d3c8;color:#dff;background:#173948}</style>
