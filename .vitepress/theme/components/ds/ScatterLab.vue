<script setup lang="ts">
import { computed, ref } from 'vue'

const jitter = ref(0)
const points = Array.from({ length: 42 }, (_, index) => ({
  temperature: [20, 22, 24, 26, 28, 30][index % 6],
  rides: 52 + (index % 6) * 9 + ((index * 17) % 26),
  offset: ((index * 13) % 17) - 8
}))
const plotted = computed(() => points.map((point) => ({
  x: 35 + (point.temperature - 20) * 31 + point.offset * jitter.value / 10,
  y: 178 - (point.rides - 45) * 1.35
})))
</script>

<template>
  <section class="scatter-lab" aria-label="散点图过度绘制实验">
    <header><span>INTERACTIVE 08</span><h3>重叠的点不是不存在，只是看不见</h3></header>
    <label>Jitter 强度：{{ jitter }}<input v-model.number="jitter" type="range" min="0" max="10"></label>
    <svg viewBox="0 0 260 205" role="img" aria-label="气温与骑行量散点图">
      <line x1="28" y1="180" x2="248" y2="180"/><line x1="28" y1="15" x2="28" y2="180"/>
      <circle v-for="(point, index) in plotted" :key="index" :cx="point.x" :cy="point.y" r="4.2"/>
      <text x="138" y="201">气温</text><text x="4" y="100" transform="rotate(-90 9 100)">骑行量</text>
    </svg>
    <p>当 x 只能取少数离散值时，点会竖向堆叠。轻微 jitter 用位置扰动展示密度，但不能把扰动后的坐标当作真实测量值。</p>
  </section>
</template>

<style scoped>
.scatter-lab { margin: 32px 0; padding: 25px; border-radius: 16px; color: #eaf3ef; background: #143b37; }
.scatter-lab header span { color: #f3c56f; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .12em; }
.scatter-lab h3 { margin: 7px 0 18px; border: 0; color: #fff; font-size: 20px; }
.scatter-lab label { display: block; color: #bdd1ca; font-size: 11px; }
.scatter-lab input { width: 100%; margin-top: 7px; accent-color: #f3c56f; }
.scatter-lab svg { width: 100%; max-height: 360px; margin-top: 14px; border-radius: 10px; background: #edf3ef; }
.scatter-lab line { stroke: #779087; stroke-width: 1; }
.scatter-lab circle { fill: #d96f52; fill-opacity: .62; stroke: #fff; stroke-width: .6; transition: cx .18s ease; }
.scatter-lab text { fill: #50635c; font: 9px sans-serif; text-anchor: middle; }
.scatter-lab > p { margin: 13px 0 0; color: #c7d9d2; font-size: 12px; line-height: 1.65; }
</style>
