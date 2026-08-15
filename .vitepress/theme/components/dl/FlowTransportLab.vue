<script setup lang="ts">
import { computed, ref } from 'vue'
const time=ref(.35), curvature=ref(.4), guidance=ref(3), steps=ref(20)
const points=Array.from({length:14},(_,i)=>({x:58+(i%7)*58,y:65+Math.floor(i/7)*120}))
const transported=computed(()=>points.map((p,i)=>({x:p.x+(270-p.x)*time.value+Math.sin(i)*curvature.value*38*time.value,y:p.y+(140-p.y)*time.value+Math.cos(i)*curvature.value*28*time.value})))
const error=computed(()=>curvature.value/steps.value*(1+guidance.value*.08)), diversity=computed(()=>Math.max(0,1-guidance.value*.09))
</script>
<template><DlExperimentFrame index="15-B" title="Flow Matching 输运场" question="直接学习概率路径的速度场时，路径曲率、ODE 步数与 guidance 怎样改变采样？" formula="dx/dt=vθ(x,t); L=E||vθ(xₜ,t)−uₜ(xₜ)||²">
<template #visual><svg viewBox="0 0 540 280" role="img" aria-label="粒子随连续速度场输运到目标分布"><path v-for="(p,i) in points" :key="'p'+i" :d="`M${p.x} ${p.y} Q${(p.x+270)/2} ${140+Math.sin(i)*curvature*70} ${transported[i].x} ${transported[i].y}`"/><circle v-for="(p,i) in transported" :key="i" :cx="p.x" :cy="p.y" r="6"/></svg></template>
<template #controls><label>连续时间 t = {{time.toFixed(2)}}<input v-model.number="time" type="range" min="0" max="1" step=".05"></label><label>路径曲率 = {{curvature.toFixed(2)}}<input v-model.number="curvature" type="range" min="0" max="1" step=".05"></label><label>ODE 步数 = {{steps}}<input v-model.number="steps" type="range" min="4" max="80" step="4"></label><label>guidance scale = {{guidance.toFixed(1)}}<input v-model.number="guidance" type="range" min="0" max="10" step=".5"></label></template>
<template #metrics><div><span>路径进度</span><strong>{{(time*100).toFixed(0)}}%</strong></div><div><span>离散误差代理</span><strong>{{error.toFixed(4)}}</strong></div><div><span>多样性代理</span><strong>{{diversity.toFixed(3)}}</strong></div><div><span>函数评估次数</span><strong>{{steps}}</strong></div></template>
<template #note>Flow Matching 回归条件概率路径的速度场，采样解 ODE。更直路径通常允许更少步；classifier-free guidance 提高条件对齐时也可能降低多样性并放大数值误差。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #493c68;border-radius:7px;background:#0b172a}path{fill:none;stroke:#4c6575;stroke-width:1.5}circle{fill:#a88ce8;stroke:#f2c14e}</style>
