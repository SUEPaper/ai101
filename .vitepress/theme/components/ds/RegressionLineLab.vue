<script setup lang="ts">
import { computed, ref } from 'vue'

const points = [
  { x: 1, y: 2.1 }, { x: 2, y: 3.1 }, { x: 3, y: 3.8 },
  { x: 4, y: 5.2 }, { x: 5, y: 5.8 }
]
const slope = ref(0.7)
const intercept = ref(1.1)
const prediction = (x: number) => intercept.value + slope.value * x
const mse = computed(() => points.reduce((sum, point) => sum + (point.y - prediction(point.x)) ** 2, 0) / points.length)
const px = (x: number) => 36 + x * 52
const py = (y: number) => 184 - y * 23
</script>

<template>
  <section class="reg-lab" aria-label="回归线与残差实验">
    <header><div><span>INTERACTIVE 10</span><h3>移动直线，观察每个误差如何汇成 MSE</h3></div><strong>MSE {{ mse.toFixed(3) }}</strong></header>
    <div class="reg-lab__grid">
      <svg viewBox="0 0 370 210" role="img" aria-label="数据点、回归线和残差">
        <line x1="36" y1="184" x2="348" y2="184" class="axis" />
        <line x1="36" y1="20" x2="36" y2="184" class="axis" />
        <line :x1="px(0)" :y1="py(prediction(0))" :x2="px(6)" :y2="py(prediction(6))" class="model" />
        <g v-for="point in points" :key="point.x">
          <line :x1="px(point.x)" :x2="px(point.x)" :y1="py(point.y)" :y2="py(prediction(point.x))" class="residual" />
          <circle :cx="px(point.x)" :cy="py(point.y)" r="5" />
        </g>
        <text x="290" y="201">白天最高温度 →</text>
      </svg>
      <div class="reg-lab__controls">
        <label>斜率 b₁ <b>{{ slope.toFixed(2) }}</b><input v-model.number="slope" type="range" min="0.3" max="1.4" step="0.05"></label>
        <label>截距 b₀ <b>{{ intercept.toFixed(2) }}</b><input v-model.number="intercept" type="range" min="0" max="3" step="0.1"></label>
        <code>ŷ = {{ intercept.toFixed(2) }} + {{ slope.toFixed(2) }}x</code>
        <p>珊瑚色竖线是残差。MSE 会特别放大较长的残差。</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.reg-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.reg-lab header{display:flex;justify-content:space-between;gap:18px;align-items:start}.reg-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.reg-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.reg-lab header strong{padding:9px 11px;border-radius:7px;color:#17302b;background:#f3c56f;font:800 13px/1 ui-monospace,monospace}.reg-lab__grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(190px,.7fr);gap:22px;margin-top:22px;align-items:center}.reg-lab svg{width:100%;border-radius:11px;background:#0c2420}.axis{stroke:#527269;stroke-width:1}.model{stroke:#f3c56f;stroke-width:3}.residual{stroke:#df765a;stroke-width:2;stroke-dasharray:3 2}.reg-lab circle{fill:#edf4f1;stroke:#2b665a;stroke-width:2}.reg-lab text{fill:#91afa5;font:9px ui-monospace,monospace}.reg-lab__controls label{display:block;margin-bottom:17px;color:#bdd1ca;font-size:12px}.reg-lab__controls b{float:right;color:#fff}.reg-lab input{width:100%;margin-top:8px;accent-color:#f3c56f}.reg-lab code{display:block;padding:11px;border-radius:8px;color:#f3c56f;background:#0c2420}.reg-lab p{margin:14px 0 0;color:#a9c0b8;font-size:12px;line-height:1.65}@media(max-width:720px){.reg-lab__grid{grid-template-columns:1fr}.reg-lab header{flex-direction:column}.reg-lab header strong{align-self:flex-start}}
</style>
