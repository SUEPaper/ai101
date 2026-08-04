<script setup lang="ts">
import { computed, ref, watch } from 'vue'

type Point={x:number,y:number}
const points:Point[]=[{x:1,y:1},{x:1.4,y:1.8},{x:2,y:1.2},{x:2.3,y:2.1},{x:5,y:5.4},{x:5.7,y:4.8},{x:6.2,y:5.7},{x:5.2,y:6.2},{x:8,y:1.3},{x:8.7,y:2},{x:9.2,y:1.1},{x:8.3,y:2.8}]
const k=ref(3)
const step=ref(0)
const initial=(value:number)=>[{x:1,y:6},{x:5,y:2},{x:9,y:6},{x:7,y:4}].slice(0,value)
const centroids=ref<Point[]>(initial(k.value))
const distance=(a:Point,b:Point)=>(a.x-b.x)**2+(a.y-b.y)**2
const assignments=computed(()=>points.map(point=>centroids.value.reduce((best,center,index)=>distance(point,center)<distance(point,centroids.value[best])?index:best,0)))
const iterate=()=>{centroids.value=centroids.value.map((center,index)=>{const members=points.filter((_,pointIndex)=>assignments.value[pointIndex]===index);return members.length?{x:members.reduce((s,p)=>s+p.x,0)/members.length,y:members.reduce((s,p)=>s+p.y,0)/members.length}:center});step.value++}
watch(k,value=>{centroids.value=initial(value);step.value=0})
const colors=['#f3c56f','#df765a','#79aa9e','#b6a4d8']
const sx=(x:number)=>24+x*32
const sy=(y:number)=>200-y*27
const inertia=computed(()=>points.reduce((sum,point,index)=>sum+distance(point,centroids.value[assignments.value[index]]),0))
</script>

<template>
  <section class="cluster-lab" aria-label="K-means 聚类实验">
    <header><div><span>INTERACTIVE 26</span><h3>重复“分配—更新”，观察质心如何移动</h3></div><button type="button" @click="iterate">执行下一轮</button></header>
    <div class="cluster-lab__controls"><label>簇数量 K={{ k }}<input v-model.number="k" type="range" min="2" max="4" step="1"></label><span>迭代 {{ step }} · inertia {{ inertia.toFixed(1) }}</span></div>
    <svg viewBox="0 0 370 220" role="img" aria-label="数据点、簇编号与质心"><g v-for="(point,index) in points" :key="index"><circle :cx="sx(point.x)" :cy="sy(point.y)" r="8" :fill="colors[assignments[index]]"/><text :x="sx(point.x)" :y="sy(point.y)+3">{{ assignments[index]+1 }}</text></g><g v-for="(center,index) in centroids" :key="index" :transform="`translate(${sx(center.x)} ${sy(center.y)})`"><line x1="-10" y1="-10" x2="10" y2="10" :stroke="colors[index]"/><line x1="-10" y1="10" x2="10" y2="-10" :stroke="colors[index]"/></g></svg>
    <p>圆点颜色和数字共同表示当前簇，× 表示质心。初始化不同，结果也可能不同。</p>
  </section>
</template>

<style scoped>
.cluster-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.cluster-lab header{display:flex;justify-content:space-between;gap:16px}.cluster-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.cluster-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.cluster-lab button{align-self:start;padding:8px 11px;border:0;border-radius:7px;color:#17302b;background:#f3c56f;font-weight:800;cursor:pointer}.cluster-lab__controls{display:grid;grid-template-columns:1fr auto;gap:16px;align-items:center;margin:20px 0}.cluster-lab label{color:#bbd0c9;font-size:11px}.cluster-lab input{width:100%;margin-top:8px;accent-color:#f3c56f}.cluster-lab__controls span{color:#f3c56f;font:10px ui-monospace,monospace}.cluster-lab svg{width:100%;border-radius:9px;background:#0c2420}.cluster-lab svg circle{stroke:#edf4f1;stroke-width:1}.cluster-lab svg text{fill:#102b26;font:800 8px ui-monospace,monospace;text-anchor:middle}.cluster-lab svg line{stroke-width:4}.cluster-lab p{margin:12px 0 0;color:#9fb8b0;font-size:10px}@media(max-width:620px){.cluster-lab header{flex-direction:column}.cluster-lab__controls{grid-template-columns:1fr}}
</style>
