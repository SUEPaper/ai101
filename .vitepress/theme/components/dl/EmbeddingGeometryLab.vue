<script setup lang="ts">
import { computed, ref } from 'vue'
const alpha=ref(.7), normalize=ref(true), query=ref<'king'|'apple'>('king')
const vectors={king:[.9,.7],queen:[.82,.92],man:[.72,.18],woman:[.62,.43],apple:[-.65,.35],orange:[-.48,.52]}
const q=computed(()=>vectors[query.value]), score=(v:number[])=>{const dot=q.value[0]*v[0]+q.value[1]*v[1];return normalize.value?dot/(Math.hypot(...q.value)*Math.hypot(...v)):dot}
const ranked=computed(()=>Object.entries(vectors).filter(([k])=>k!==query.value).map(([k,v])=>[k,score(v)] as const).sort((a,b)=>b[1]-a[1]))
</script>
<template><DlExperimentFrame index="17-A" title="词向量几何台" question="点积同时混合方向与范数；余弦相似度、类比偏移和频率范数分别在说什么？" formula="cos(u,v)=u·v/(||u||||v||); king−man+woman≈queen">
<template #visual><svg viewBox="0 0 540 280" role="img" aria-label="二维教学词向量空间"><line x1="270" y1="25" x2="270" y2="250"/><line x1="35" y1="140" x2="505" y2="140"/><g v-for="(v,k) in vectors" :key="k"><circle :cx="270+v[0]*180" :cy="140-v[1]*100" :r="k===query?9:6"/><text :x="280+v[0]*180" :y="136-v[1]*100">{{k}}</text></g></svg></template>
<template #controls><label>查询词<select v-model="query"><option value="king">king</option><option value="apple">apple</option></select></label><button type="button" @click="normalize=!normalize">{{normalize?'切到点积':'切到余弦'}}</button><label>类比步长 = {{alpha.toFixed(2)}}<input v-model.number="alpha" type="range" min="0" max="1.5" step=".05"></label></template>
<template #metrics><div><span>最相似词</span><strong>{{ranked[0][0]}}</strong></div><div><span>最高分</span><strong>{{ranked[0][1].toFixed(3)}}</strong></div><div><span>查询范数</span><strong>{{Math.hypot(...q).toFixed(3)}}</strong></div><div><span>相似度</span><strong>{{normalize?'cosine':'dot product'}}</strong></div></template>
<template #note>二维点只作几何教学。真实 embedding 的方向、范数和偏置受频率、窗口、负采样与语料影响；类比是局部线性规律，不是词义代数定律。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #2c5f65;border-radius:7px;background:#0b172a}line{stroke:#385665}circle{fill:#58d3c8;stroke:#f2c14e}text{fill:#e7f1f2;font:11px ui-monospace,monospace}</style>
