<script setup lang="ts">
import { computed, ref } from 'vue'
const threshold=ref(.5), minScore=ref(.35)
const boxes=[{x:76,y:62,w:178,h:116,s:.94},{x:105,y:78,w:176,h:120,s:.83},{x:246,y:90,w:166,h:112,s:.76},{x:267,y:111,w:163,h:108,s:.51},{x:80,y:190,w:105,h:58,s:.31}]
const iou=(a:any,b:any)=>{const x=Math.max(a.x,b.x),y=Math.max(a.y,b.y),r=Math.min(a.x+a.w,b.x+b.w),d=Math.min(a.y+a.h,b.y+b.h);const inter=Math.max(0,r-x)*Math.max(0,d-y);return inter/(a.w*a.h+b.w*b.h-inter)}
const kept=computed(()=>{const out:any[]=[];for(const box of boxes.filter(b=>b.s>=minScore.value).sort((a,b)=>b.s-a.s)){if(out.every(item=>iou(item,box)<=threshold.value))out.push(box)}return out})
const isKept=(box:any)=>kept.value.includes(box)
</script>
<template><DlExperimentFrame index="06-B" title="NMS 候选框工作台" question="一个阈值怎样在重复框与相邻真实目标之间做不可逆取舍？" formula="keep bᵢ iff score≥τₛ and IoU(bᵢ, kept)≤τᵢ">
<template #visual><svg viewBox="0 0 520 300" role="img" aria-label="非极大值抑制候选框"><rect x="28" y="28" width="464" height="244" class="image"/><g v-for="(box,index) in boxes" :key="index"><rect :x="box.x" :y="box.y" :width="box.w" :height="box.h" :class="isKept(box)?'keep':'drop'"/><text :x="box.x+6" :y="box.y+16">{{box.s.toFixed(2)}}</text></g></svg></template>
<template #controls><label>IoU 抑制阈值 = {{threshold.toFixed(2)}}<input v-model.number="threshold" type="range" min="0.1" max="0.9" step="0.05"></label><label>分数阈值 = {{minScore.toFixed(2)}}<input v-model.number="minScore" type="range" min="0.1" max="0.9" step="0.05"></label></template>
<template #metrics><div><span>输入候选</span><strong>{{boxes.length}}</strong></div><div><span>通过分数门</span><strong>{{boxes.filter(b=>b.s>=minScore).length}}</strong></div><div><span>NMS 后保留</span><strong>{{kept.length}}</strong></div><div><span>最高保留分数</span><strong>{{kept[0]?.s.toFixed(2)??'—'}}</strong></div></template>
<template #note>较低 IoU 阈值更积极去重，也更容易误删密集目标。Soft-NMS 会衰减而非直接删除分数；DETR 则用集合匹配绕开手工 anchor 与 NMS，但引入查询和匹配训练问题。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.image{fill:#101f34;stroke:#36566a}.keep{fill:rgba(88,211,200,.09);stroke:#58d3c8;stroke-width:3}.drop{fill:rgba(223,108,91,.06);stroke:#df6c5b;stroke-width:2;stroke-dasharray:6 4;opacity:.55}text{fill:#fff;font:10px ui-monospace,monospace}</style>
