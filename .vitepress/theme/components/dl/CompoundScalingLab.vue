<script setup lang="ts">
import { computed, ref } from 'vue'
const phi=ref(2), alpha=ref(1.2), beta=ref(1.1), gamma=ref(1.15)
const depth=computed(()=>alpha.value**phi.value), width=computed(()=>beta.value**phi.value), resolution=computed(()=>gamma.value**phi.value)
const compute=computed(()=>depth.value*width.value**2*resolution.value**2)
const balance=computed(()=>Math.min(depth.value,width.value,resolution.value)/Math.max(depth.value,width.value,resolution.value))
</script>
<template><DlExperimentFrame index="05-B" title="复合缩放配方台" question="固定一档额外算力时，只加宽、只加深还是同时提高分辨率？" formula="d=αᵠ, w=βᵠ, r=γᵠ; compute ∝ d·w²·r²">
<template #visual><svg viewBox="0 0 540 260" role="img" aria-label="深度宽度分辨率缩放柱形图"><line x1="45" y1="220" x2="500" y2="220"/><g v-for="(item,index) in [{n:'DEPTH',v:depth,c:'#58d3c8'},{n:'WIDTH',v:width,c:'#f2c14e'},{n:'RESOLUTION',v:resolution,c:'#df6c5b'}]" :key="item.n"><rect :x="85+index*145" :y="220-item.v*70" width="74" :height="item.v*70" :fill="item.c"/><text :x="122+index*145" y="242">{{item.n}}</text><text :x="122+index*145" :y="207-item.v*70">{{item.v.toFixed(2)}}×</text></g></svg></template>
<template #controls><label>缩放等级 φ = {{phi}}<input v-model.number="phi" type="range" min="0" max="6" step="1"></label><label>α（深度）= {{alpha.toFixed(2)}}<input v-model.number="alpha" type="range" min="1" max="1.4" step="0.02"></label><label>β（宽度）= {{beta.toFixed(2)}}<input v-model.number="beta" type="range" min="1" max="1.3" step="0.02"></label><label>γ（分辨率）= {{gamma.toFixed(2)}}<input v-model.number="gamma" type="range" min="1" max="1.3" step="0.02"></label></template>
<template #metrics><div><span>深度倍率</span><strong>{{depth.toFixed(2)}}×</strong></div><div><span>宽度倍率</span><strong>{{width.toFixed(2)}}×</strong></div><div><span>分辨率倍率</span><strong>{{resolution.toFixed(2)}}×</strong></div><div><span>计算倍率代理</span><strong>{{compute.toFixed(1)}}× · 平衡 {{balance.toFixed(2)}}</strong></div></template>
<template #note>EfficientNet 的系数来自基线模型上的搜索，不是普适常数。部署时还要把算子支持、内存峰值和目标尺寸分布加入约束；ConvNeXt 则提醒我们，现代训练配方也能显著提升纯卷积骨干。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}line{stroke:#54717d}rect{transition:all .2s}text{fill:#dce8e9;font:10px ui-monospace,monospace;text-anchor:middle}</style>
