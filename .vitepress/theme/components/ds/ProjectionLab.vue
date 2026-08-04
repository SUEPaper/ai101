<script setup lang="ts">
import { computed, ref } from 'vue'

const angle = ref(28)
const target = { x: 3.2, y: 2.3 }
const radians = computed(() => angle.value * Math.PI / 180)
const axis = computed(() => ({ x: Math.cos(radians.value), y: Math.sin(radians.value) }))
const score = computed(() => target.x * axis.value.x + target.y * axis.value.y)
const projection = computed(() => ({ x: score.value * axis.value.x, y: score.value * axis.value.y }))
const residual = computed(() => Math.hypot(target.x - projection.value.x, target.y - projection.value.y))
const sx = (x: number) => 155 + x * 42
const sy = (y: number) => 160 - y * 42
</script>

<template>
  <section class="projection-lab" aria-label="向量投影实验">
    <header><div><span>INTERACTIVE 12</span><h3>转动模型方向，把观测投影到可解释空间</h3></div><strong>残差长度 {{ residual.toFixed(2) }}</strong></header>
    <div class="projection-lab__grid">
      <svg viewBox="0 0 360 210" role="img" aria-label="目标向量、投影与残差">
        <line x1="20" y1="160" x2="340" y2="160" class="grid"/><line x1="155" y1="20" x2="155" y2="195" class="grid"/>
        <line :x1="sx(-3.3*axis.x)" :y1="sy(-3.3*axis.y)" :x2="sx(4.2*axis.x)" :y2="sy(4.2*axis.y)" class="space"/>
        <line :x1="sx(projection.x)" :y1="sy(projection.y)" :x2="sx(target.x)" :y2="sy(target.y)" class="error"/>
        <line :x1="sx(0)" :y1="sy(0)" :x2="sx(target.x)" :y2="sy(target.y)" class="target"/>
        <circle :cx="sx(projection.x)" :cy="sy(projection.y)" r="6"/>
        <text :x="sx(target.x)-28" :y="sy(target.y)-10">观测 y</text>
        <text :x="sx(projection.x)+8" :y="sy(projection.y)+14">ŷ</text>
      </svg>
      <div><label>模型方向 {{ angle }}°<input v-model.number="angle" type="range" min="0" max="90" step="1"></label><p><b>点积：</b>{{ score.toFixed(2) }}</p><p><b>投影：</b>({{ projection.x.toFixed(2) }}, {{ projection.y.toFixed(2) }})</p><small>OLS 选择残差平方和最小的投影。最优点处，残差与模型空间正交。</small></div>
    </div>
  </section>
</template>

<style scoped>
.projection-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.projection-lab header{display:flex;justify-content:space-between;gap:16px}.projection-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.projection-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.projection-lab header strong{color:#f3c56f;font:700 12px ui-monospace,monospace}.projection-lab__grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(180px,.7fr);gap:22px;margin-top:20px;align-items:center}.projection-lab svg{width:100%;border-radius:10px;background:#0c2420}.grid{stroke:#42645b;stroke-width:1}.space{stroke:#f3c56f;stroke-width:3}.target{stroke:#edf4f1;stroke-width:2}.error{stroke:#df765a;stroke-width:3;stroke-dasharray:4 3}.projection-lab circle{fill:#df765a}.projection-lab text{fill:#c8dad4;font:10px sans-serif}.projection-lab label{display:block;color:#c8dad4;font-size:12px}.projection-lab input{width:100%;margin:10px 0 18px;accent-color:#f3c56f}.projection-lab p{margin:8px 0;color:#e8f1ed}.projection-lab small{display:block;margin-top:15px;color:#9db7ae;line-height:1.7}@media(max-width:680px){.projection-lab header{flex-direction:column}.projection-lab__grid{grid-template-columns:1fr}}
</style>
