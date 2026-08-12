<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

interface Vector { x: number; y: number }

const batchSizes = [1, 2, 4, 8, 16, 32, 64]
const batchIndex = ref(2)
const noiseScale = ref(1)
const shuffle = ref(3)
const fullGradient: Vector = { x: 1, y: .55 }
const origin = { x: 245, y: 172 }
const vectorScale = 104

const gradients = computed<Vector[]>(() => Array.from({ length: 64 }, (_, index) => {
  const angle = 2 * Math.PI * index / 64
  const radius = .46 + .18 * Math.sin(5 * angle) + .09 * Math.cos(9 * angle)
  return {
    x: fullGradient.x + noiseScale.value * radius * Math.cos(angle),
    y: fullGradient.y + noiseScale.value * radius * Math.sin(angle)
  }
}))

const order = computed(() => Array.from({ length: 64 }, (_, index) => (index * 17 + shuffle.value * 11) % 64))
const selectedIndices = computed(() => new Set(order.value.slice(0, batchSizes[batchIndex.value])))
const batchGradient = computed<Vector>(() => {
  const selected = order.value.slice(0, batchSizes[batchIndex.value])
  const sum = selected.reduce((value, index) => ({
    x: value.x + gradients.value[index].x,
    y: value.y + gradients.value[index].y
  }), { x: 0, y: 0 })
  return { x: sum.x / selected.length, y: sum.y / selected.length }
})
const error = computed(() => Math.hypot(batchGradient.value.x - fullGradient.x, batchGradient.value.y - fullGradient.y))
const angleError = computed(() => {
  const dot = batchGradient.value.x * fullGradient.x + batchGradient.value.y * fullGradient.y
  const denominator = Math.hypot(batchGradient.value.x, batchGradient.value.y) * Math.hypot(fullGradient.x, fullGradient.y)
  const cosine = Math.max(-1, Math.min(1, dot / denominator))
  return Math.acos(cosine) * 180 / Math.PI
})
const endX = (vector: Vector) => origin.x + vector.x * vectorScale
const endY = (vector: Vector) => origin.y - vector.y * vectorScale
</script>

<template>
  <MlExperimentFrame
    index="02-B"
    title="mini-batch 在省什么，又牺牲什么"
    question="少看一些样本得到的梯度有多嘈杂？增加 batch size 后，它如何靠近全量梯度？"
    formula="g_B = (1/B) sum_i in B grad l_i"
  >
    <template #visual>
      <figure class="gradient-map">
        <svg viewBox="0 0 600 330" role="img" :aria-label="`批量大小 ${batchSizes[batchIndex]}；mini-batch 梯度与全量梯度的夹角误差 ${angleError.toFixed(1)} 度`">
          <defs>
            <marker id="full-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="full-marker" /></marker>
            <marker id="batch-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" class="batch-marker" /></marker>
          </defs>
          <rect class="plot-bg" x="20" y="14" width="560" height="288" rx="8" />
          <line class="axis" x1="36" x2="564" :y1="origin.y" :y2="origin.y" />
          <line class="axis" :x1="origin.x" :x2="origin.x" y1="28" y2="288" />
          <g v-for="(gradient, index) in gradients" :key="index">
            <circle :cx="endX(gradient)" :cy="endY(gradient)" :r="selectedIndices.has(index) ? 5 : 2.5" :class="selectedIndices.has(index) ? 'selected-sample' : 'sample'" />
          </g>
          <line class="full-vector" :x1="origin.x" :y1="origin.y" :x2="endX(fullGradient)" :y2="endY(fullGradient)" marker-end="url(#full-arrow)" />
          <line class="batch-vector" :x1="origin.x" :y1="origin.y" :x2="endX(batchGradient)" :y2="endY(batchGradient)" marker-end="url(#batch-arrow)" />
          <circle class="origin" :cx="origin.x" :cy="origin.y" r="4" />
          <text x="34" y="34">每个点 = 单样本梯度</text>
          <text x="394" y="286">g₁ 坐标</text>
          <text x="253" y="42">g₂ 坐标</text>
        </svg>
        <figcaption><span class="legend sample" />未抽样　<span class="legend selected" />当前 mini-batch　<span class="legend full" />全量梯度　<span class="legend batch" />mini-batch 均值</figcaption>
      </figure>
      <table>
        <thead><tr><th>估计量</th><th>g₁</th><th>g₂</th></tr></thead>
        <tbody>
          <tr><td>全量梯度</td><td>{{ fullGradient.x.toFixed(3) }}</td><td>{{ fullGradient.y.toFixed(3) }}</td></tr>
          <tr><td>mini-batch</td><td>{{ batchGradient.x.toFixed(3) }}</td><td>{{ batchGradient.y.toFixed(3) }}</td></tr>
          <tr><td>误差</td><td>{{ (batchGradient.x - fullGradient.x).toFixed(3) }}</td><td>{{ (batchGradient.y - fullGradient.y).toFixed(3) }}</td></tr>
        </tbody>
      </table>
    </template>
    <template #controls>
      <label>
        <span>批量大小 B <output>{{ batchSizes[batchIndex] }}</output></span>
        <input v-model.number="batchIndex" type="range" min="0" :max="batchSizes.length - 1" step="1">
      </label>
      <label>
        <span>样本梯度噪声 <output>{{ noiseScale.toFixed(2) }}</output></span>
        <input v-model.number="noiseScale" type="range" min="0" max="1.6" step=".1">
      </label>
      <button type="button" @click="shuffle += 1">重新打乱样本顺序</button>
      <p>64 个单样本梯度的平均值被构造成固定全量梯度。每次打乱只改变当前 mini-batch，不改变目标函数。</p>
    </template>
    <template #metrics>
      <div><span>批量大小</span><strong>{{ batchSizes[batchIndex] }} / 64</strong></div>
      <div><span>向量误差 ‖g_B-g‖</span><strong>{{ error.toFixed(3) }}</strong></div>
      <div><span>方向夹角</span><strong>{{ angleError.toFixed(1) }}°</strong></div>
      <div><span>单步样本计算量</span><strong>{{ batchSizes[batchIndex] }}×</strong></div>
    </template>
    <template #note>
      反复打乱并逐步增大 B。mini-batch 梯度不是“错误梯度”，而是全量梯度的随机估计；更大的 B 通常降低方差，却增加单步计算和内存占用。方差按 1/B 缩放需要独立同分布等条件，按顺序采样、类别成批或强相关样本都可能破坏这一直觉。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.gradient-map{margin:0}.gradient-map svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:9px;background:#10182e}.plot-bg{fill:#10182e}.axis{stroke:#344675}.sample{fill:#53638b}.selected-sample{fill:#f06b5b;stroke:#fff;stroke-width:1.5}.full-vector{stroke:#72d5ce;stroke-width:4}.batch-vector{stroke:#f4c95d;stroke-width:3}.full-marker{fill:#72d5ce}.batch-marker{fill:#f4c95d}.origin{fill:#fff}.gradient-map text{fill:#9eabc9;font:10px ui-monospace,monospace}.gradient-map figcaption{margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.7}.legend{display:inline-block;width:10px;height:10px;margin-right:4px;vertical-align:middle}.legend.sample{border-radius:50%;background:#53638b}.legend.selected{border-radius:50%;background:#f06b5b}.legend.full{height:3px;background:#72d5ce}.legend.batch{height:3px;background:#f4c95d}.ml-experiment__visual table{width:100%;margin:10px 0 0;border-collapse:collapse;color:#c8d1e8;background:#10182e;font-size:10px}.ml-experiment__visual th,.ml-experiment__visual td{padding:7px;border:1px solid #2d3c68;text-align:right}.ml-experiment__visual th:first-child,.ml-experiment__visual td:first-child{text-align:left}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls button{min-height:44px;border:1px solid #6e82bd;border-radius:7px;color:#fff;background:#263d88;font-weight:750;cursor:pointer}.ml-experiment__controls button:hover{background:#3150a3}.ml-experiment__controls button:focus-visible{outline:3px solid #f4c95d;outline-offset:3px}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}
</style>
