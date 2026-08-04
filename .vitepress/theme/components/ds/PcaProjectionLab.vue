<script setup lang="ts">
import { computed, ref } from 'vue'

const angle=ref(32)
const points=[[-3,-1.8],[-2.5,-1.2],[-2,-1.4],[-1,-.3],[-.5,.2],[.4,.1],[1,.9],[1.8,1.1],[2.5,1.9],[3,1.6]]
const axis=computed(()=>({x:Math.cos(angle.value*Math.PI/180),y:Math.sin(angle.value*Math.PI/180)}))
const projected=computed(()=>points.map(([x,y])=>{const score=x*axis.value.x+y*axis.value.y;return{x:score*axis.value.x,y:score*axis.value.y,score}}))
const variance=computed(()=>projected.value.reduce((sum,p)=>sum+p.score**2,0)/points.length)
const error=computed(()=>points.reduce((sum,[x,y],i)=>sum+(x-projected.value[i].x)**2+(y-projected.value[i].y)**2,0)/points.length)
const sx=(x:number)=>180+x*44
const sy=(y:number)=>105-y*44
</script>

<template>
  <section class="pca-lab" aria-label="PCA 投影实验">
    <header><div><span>INTERACTIVE 24</span><h3>旋转一维轴，同时观察投影方差与重构误差</h3></div><strong>{{ angle }}°</strong></header>
    <div class="pca-lab__grid"><svg viewBox="0 0 360 210" role="img" aria-label="二维点云在旋转轴上的投影"><line :x1="sx(-4*axis.x)" :y1="sy(-4*axis.y)" :x2="sx(4*axis.x)" :y2="sy(4*axis.y)" class="axis"/><g v-for="(point,index) in points" :key="index"><line :x1="sx(point[0])" :y1="sy(point[1])" :x2="sx(projected[index].x)" :y2="sy(projected[index].y)" class="residual"/><circle :cx="sx(point[0])" :cy="sy(point[1])" r="5"/><circle :cx="sx(projected[index].x)" :cy="sy(projected[index].y)" r="3" class="shadow"/></g></svg><div><label>投影轴角度<input v-model.number="angle" type="range" min="0" max="180" step="1"></label><p><small>投影方差</small><b>{{ variance.toFixed(2) }}</b></p><p><small>平均重构误差</small><b>{{ error.toFixed(2) }}</b></p><span>PCA 第一主成分选择方差最大、等价地重构误差最小的方向。</span></div></div>
  </section>
</template>

<style scoped>
.pca-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.pca-lab header{display:flex;justify-content:space-between;gap:16px}.pca-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.pca-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.pca-lab header strong{color:#f3c56f;font:700 12px ui-monospace,monospace}.pca-lab__grid{display:grid;grid-template-columns:1.4fr .7fr;gap:18px;margin-top:20px;align-items:center}.pca-lab svg{width:100%;border-radius:9px;background:#0c2420}.pca-lab .axis{stroke:#f3c56f;stroke-width:3}.pca-lab .residual{stroke:#df765a;stroke-width:1;stroke-dasharray:3 2}.pca-lab circle{fill:#edf4f1;stroke:#2b665a;stroke-width:2}.pca-lab circle.shadow{fill:#f3c56f;stroke:none}.pca-lab label{color:#b9cfc7;font-size:11px}.pca-lab input{width:100%;margin:8px 0 14px;accent-color:#f3c56f}.pca-lab p{display:flex;justify-content:space-between;margin:7px 0;padding:9px;border-radius:6px;background:#0c2420}.pca-lab p small{color:#91aaa2}.pca-lab p b{color:#f3c56f}.pca-lab div>span{display:block;margin-top:13px;color:#9fb8b0;font-size:10px;line-height:1.6}@media(max-width:680px){.pca-lab header{flex-direction:column}.pca-lab__grid{grid-template-columns:1fr}}
</style>
