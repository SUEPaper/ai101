<script setup lang="ts">
import { computed, ref } from 'vue'

const start = ref(-3.5)
const learningRate = ref(0.18)
const steps = computed(() => {
  const result = [start.value]
  for (let index = 0; index < 8; index += 1) result.push(result[index] - learningRate.value * 2 * (result[index] - 2))
  return result
})
const loss = (theta: number) => (theta - 2) ** 2 + 0.5
const px = (theta: number) => 28 + (theta + 5) / 10 * 320
const py = (value: number) => 184 - Math.min(value, 26) / 26 * 158
const curve = computed(() => Array.from({ length: 81 }, (_, index) => {
  const theta = -5 + index / 8
  return `${index ? 'L' : 'M'} ${px(theta)} ${py(loss(theta))}`
}).join(' '))
</script>

<template>
  <section class="gd-lab" aria-label="梯度下降实验">
    <header><div><span>INTERACTIVE 13</span><h3>同一片损失地形，不同学习率会走出不同路径</h3></div><code>第 8 步 θ={{ steps.at(-1)?.toFixed(2) }}</code></header>
    <div class="gd-lab__grid">
      <svg viewBox="0 0 376 210" role="img" aria-label="二次损失曲线与梯度下降路径">
        <path :d="curve" class="curve"/>
        <polyline :points="steps.map(theta => `${px(theta)},${py(loss(theta))}`).join(' ')" class="path"/>
        <circle v-for="(theta,index) in steps" :key="index" :cx="px(theta)" :cy="py(loss(theta))" :r="index===0?6:4"/>
        <text x="292" y="202">参数 θ →</text>
      </svg>
      <div>
        <label>起点 {{ start.toFixed(1) }}<input v-model.number="start" type="range" min="-4.5" max="5" step="0.5"></label>
        <label>学习率 {{ learningRate.toFixed(2) }}<input v-model.number="learningRate" type="range" min="0.03" max="1.05" step="0.03"></label>
        <p :class="{ warn: learningRate >= 1 }">{{ learningRate < .08 ? '步子很稳，但收敛较慢。' : learningRate < 1 ? '当前路径会逐渐靠近最低点。' : '步子过大，路径在最低点两侧振荡。' }}</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.gd-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.gd-lab header{display:flex;justify-content:space-between;gap:16px}.gd-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.gd-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.gd-lab header code{align-self:start;padding:8px 10px;border-radius:7px;color:#17302b;background:#f3c56f}.gd-lab__grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(180px,.7fr);gap:22px;margin-top:20px;align-items:center}.gd-lab svg{width:100%;border-radius:10px;background:#0c2420}.curve{fill:none;stroke:#8aaba0;stroke-width:2}.path{fill:none;stroke:#df765a;stroke-width:2;stroke-dasharray:4 3}.gd-lab circle{fill:#f3c56f;stroke:#122f2a;stroke-width:1}.gd-lab text{fill:#91afa5;font:9px ui-monospace,monospace}.gd-lab label{display:block;margin-bottom:17px;color:#c9dad4;font-size:12px}.gd-lab input{width:100%;margin-top:8px;accent-color:#f3c56f}.gd-lab p{padding:11px;border-radius:8px;color:#bed0ca;background:#0c2420;font-size:12px;line-height:1.6}.gd-lab p.warn{color:#ffd1c5;background:#4b2922}@media(max-width:680px){.gd-lab header{flex-direction:column}.gd-lab__grid{grid-template-columns:1fr}}
</style>
