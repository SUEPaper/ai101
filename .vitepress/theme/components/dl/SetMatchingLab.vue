<script setup lang="ts">
import { computed, ref } from 'vue'
const query2=ref(310), classWeight=ref(1), boxWeight=ref(5)
const targets=[105,360], queries=computed(()=>[85,query2.value,430])
const perms=[[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]]
const costs=computed(()=>perms.map(p=>({p,c:targets.reduce((sum,t,i)=>sum+boxWeight.value*Math.abs(queries.value[p[i]]-t)/400+classWeight.value*(p[i]===2?.55:.12),0)})).sort((a,b)=>a.c-b.c))
const best=computed(()=>costs.value[0])
</script>
<template><DlExperimentFrame index="07-B" title="集合匹配分派台" question="固定数量的 object query 如何通过一对一匹配消除重复预测？" formula="σ*=arg minσ Σᵢ [λcls Lcls + λbox Lbox]">
<template #visual><svg viewBox="0 0 520 260" role="img" aria-label="目标与 object query 的二分图匹配"><g v-for="(x,i) in targets" :key="'t'+i"><circle :cx="x" cy="62" r="18" class="target"/><text :x="x" y="66">T{{i+1}}</text></g><g v-for="(x,i) in queries" :key="'q'+i"><circle :cx="x" cy="205" r="18" :class="best.p.includes(i)?'used':'unused'"/><text :x="x" y="209">Q{{i+1}}</text></g><line v-for="(q,i) in best.p" :key="i" :x1="queries[q]" y1="187" :x2="targets[i]" y2="80"/></svg></template>
<template #controls><label>Query 2 位置 = {{query2}}<input v-model.number="query2" type="range" min="120" max="420" step="10"></label><label>分类代价权重 = {{classWeight.toFixed(1)}}<input v-model.number="classWeight" type="range" min="0" max="3" step="0.25"></label><label>框代价权重 = {{boxWeight.toFixed(1)}}<input v-model.number="boxWeight" type="range" min="1" max="10" step="0.5"></label></template>
<template #metrics><div><span>最优匹配</span><strong>{{best.p.map((q,i)=>`T${i+1}←Q${q+1}`).join(' · ')}}</strong></div><div><span>总代价</span><strong>{{best.c.toFixed(3)}}</strong></div><div><span>未匹配 query</span><strong>{{[0,1,2].filter(q=>!best.p.includes(q)).map(q=>`Q${q+1}`).join('')}}</strong></div><div><span>枚举方案</span><strong>{{costs.length}}</strong></div></template>
<template #note>DETR 训练时用 Hungarian algorithm 在更大的代价矩阵上求一对一分派，未匹配 query 学习“no object”。一对一约束减少重复框，但匹配代价、query 数和训练收敛速度成为新接口。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}circle{stroke-width:3}.target{fill:#1d4c55;stroke:#58d3c8}.used{fill:#4b4020;stroke:#f2c14e}.unused{fill:#2a3445;stroke:#607887}line{stroke:#df6c5b;stroke-width:3}text{fill:#fff;font:10px ui-monospace,monospace;text-anchor:middle}</style>
