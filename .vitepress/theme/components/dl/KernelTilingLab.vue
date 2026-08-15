<script setup lang="ts">
import { computed, ref } from 'vue'
const m=ref(4096), n=ref(4096), k=ref(4096), tile=ref(64), stages=ref(3)
const blocks=computed(()=>Math.ceil(m.value/tile.value)*Math.ceil(n.value/tile.value))
const sharedKiB=computed(()=>2*tile.value*tile.value*2*stages.value/1024)
const intensity=computed(()=>2*tile.value**3/(2*tile.value**2*2))
const occupancy=computed(()=>Math.max(.15,Math.min(1,128/sharedKiB.value)))
</script>
<template><DlExperimentFrame index="30-A" title="GEMM Tile 与流水线调谐台" question="tile、shared memory 和 pipeline stages 怎样交换数据重用、并行块数与 occupancy？" formula="Ctile+=Atile·Btile; AI≈2t³/(4t²)">
<template #visual><div class="tiles" :style="{gridTemplateColumns:`repeat(${Math.min(8,Math.max(2,tile/16))},1fr)`}"><i v-for="i in 32" :key="i"></i></div></template>
<template #controls><label>M/N = {{m}}<input v-model.number="m" type="range" min="512" max="8192" step="512"></label><label>K = {{k}}<input v-model.number="k" type="range" min="512" max="8192" step="512"></label><label>Tile = {{tile}}<input v-model.number="tile" type="range" min="16" max="128" step="16"></label><label>Pipeline stages = {{stages}}<input v-model.number="stages" type="range" min="1" max="6" step="1"></label></template>
<template #metrics><div><span>输出 blocks</span><strong>{{blocks}}</strong></div><div><span>Shared 代理</span><strong>{{sharedKiB.toFixed(1)}} KiB</strong></div><div><span>算术强度代理</span><strong>{{intensity.toFixed(1)}}</strong></div><div><span>Occupancy 代理</span><strong>{{occupancy.toFixed(2)}}</strong></div></template>
<template #note>真实 tile 还含 warps、register fragment、bank conflict、Tensor Core 指令、边界 mask 和硬件上限；最大 tile 不一定最快。</template>
</DlExperimentFrame></template>
<style scoped>.tiles{display:grid;align-content:center;gap:5px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.tiles i{aspect-ratio:1;border:1px solid #58d3c8;background:#173948}</style>
