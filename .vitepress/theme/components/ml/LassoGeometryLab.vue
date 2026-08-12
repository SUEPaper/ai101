<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const lambda = ref(1.0)
const secondarySignal = ref(0.9)
const primarySignal = 1.75
const softThreshold = (value: number, threshold: number) => Math.sign(value) * Math.max(Math.abs(value) - threshold, 0)

const plot = { originX: 63, originY: 280, scale: 130 }
const x = (value: number) => plot.originX + value * plot.scale
const y = (value: number) => plot.originY - value * plot.scale
const path = { left: 382, right: 566, top: 47, bottom: 279 }
const pathX = (value: number) => path.left + value / 2.2 * (path.right - path.left)
const pathY = (value: number) => path.bottom - value / 1.85 * (path.bottom - path.top)

const result = computed(() => {
  const z = [primarySignal, secondarySignal.value]
  const lasso = z.map((value) => softThreshold(value, lambda.value))
  const ridge = z.map((value) => value / (1 + lambda.value))
  const lassoRadius = Math.abs(lasso[0]) + Math.abs(lasso[1])
  const ridgeRadius = Math.hypot(ridge[0], ridge[1])
  const lassoLossRadius = Math.hypot(z[0] - lasso[0], z[1] - lasso[1])
  const ridgeLossRadius = Math.hypot(z[0] - ridge[0], z[1] - ridge[1])
  const lambdaValues = Array.from({ length: 57 }, (_, index) => index * 2.2 / 56)
  const primaryPath = lambdaValues.map((value) => `${pathX(value)},${pathY(softThreshold(primarySignal, value))}`).join(' ')
  const secondaryPath = lambdaValues.map((value) => `${pathX(value)},${pathY(softThreshold(secondarySignal.value, value))}`).join(' ')
  return {
    z,
    lasso,
    ridge,
    lassoRadius,
    ridgeRadius,
    lassoLossRadius,
    ridgeLossRadius,
    primaryPath,
    secondaryPath,
    activeCount: lasso.filter((value) => value !== 0).length,
  }
})

const diamondPoints = computed(() => {
  const radius = result.value.lassoRadius
  return [[radius, 0], [0, radius], [-radius, 0], [0, -radius]]
    .map(([first, second]) => `${x(first)},${y(second)}`)
    .join(' ')
})
</script>

<template>
  <MlExperimentFrame
    index="06-B"
    title="让 L1 解撞上坐标轴拐角"
    question="当第二只温度传感器的证据不足以越过阈值时，为什么 Lasso 会给出精确 0，而 Ridge 只会继续缩小？"
    formula="β̂ⱼ = Sλ(zⱼ) = sign(zⱼ)(|zⱼ| − λ)₊"
  >
    <template #visual>
      <figure class="geometry-board">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`惩罚 ${lambda.toFixed(2)}，Lasso 活跃系数 ${result.activeCount} 个，第二个 Lasso 系数 ${result.lasso[1].toFixed(2)}`">
          <rect x="16" y="16" width="568" height="294" rx="9" class="board-bg" />
          <text x="34" y="38" class="panel-title">系数平面：约束边界与相切点</text>
          <line :x1="x(-0.2)" :x2="x(2.15)" :y1="y(0)" :y2="y(0)" class="axis" />
          <line :x1="x(0)" :x2="x(0)" :y1="y(-0.2)" :y2="y(1.75)" class="axis" />
          <line v-for="tick in [0.5,1,1.5,2]" :key="`gx-${tick}`" :x1="x(tick)" :x2="x(tick)" :y1="y(-0.1)" :y2="y(1.7)" class="grid" />
          <line v-for="tick in [0.5,1,1.5]" :key="`gy-${tick}`" :x1="x(-0.1)" :x2="x(2.1)" :y1="y(tick)" :y2="y(tick)" class="grid" />

          <circle :cx="x(result.z[0])" :cy="y(result.z[1])" :r="result.lassoLossRadius * plot.scale" class="loss-contour lasso-loss" />
          <circle :cx="x(result.z[0])" :cy="y(result.z[1])" :r="result.ridgeLossRadius * plot.scale" class="loss-contour ridge-loss" />
          <polygon :points="diamondPoints" class="l1-boundary" />
          <circle :cx="x(0)" :cy="y(0)" :r="result.ridgeRadius * plot.scale" class="l2-boundary" />
          <circle :cx="x(result.z[0])" :cy="y(result.z[1])" r="6" class="ols-point" />
          <circle :cx="x(result.lasso[0])" :cy="y(result.lasso[1])" r="7" class="lasso-point" />
          <circle :cx="x(result.ridge[0])" :cy="y(result.ridge[1])" r="7" class="ridge-point" />
          <text :x="x(result.z[0]) + 8" :y="y(result.z[1]) - 8" class="ols-label">OLS z</text>
          <text :x="Math.max(70, x(result.lasso[0]) - 34)" :y="Math.max(48, y(result.lasso[1]) - 10)" class="lasso-label">Lasso</text>
          <text :x="x(result.ridge[0]) + 8" :y="y(result.ridge[1]) + 18" class="ridge-label">Ridge</text>
          <text x="279" y="300" class="axis-label">β₁ · 壳体温度 A</text>
          <text x="26" y="57" class="axis-label">β₂</text>

          <line x1="357" x2="357" y1="31" y2="292" class="divider" />
          <text x="382" y="38" class="panel-title">Lasso 正则化路径</text>
          <line :x1="path.left" :x2="path.right" :y1="path.bottom" :y2="path.bottom" class="axis" />
          <line :x1="path.left" :x2="path.left" :y1="path.top" :y2="path.bottom" class="axis" />
          <line v-for="tick in [0.5,1,1.5,2]" :key="`px-${tick}`" :x1="pathX(tick)" :x2="pathX(tick)" :y1="path.top" :y2="path.bottom" class="path-grid" />
          <polyline :points="result.primaryPath" class="path-primary" />
          <polyline :points="result.secondaryPath" class="path-secondary" />
          <line :x1="pathX(lambda)" :x2="pathX(lambda)" :y1="path.top" :y2="path.bottom" class="lambda-cursor" />
          <circle :cx="pathX(lambda)" :cy="pathY(result.lasso[0])" r="5" class="path-primary-dot" />
          <circle :cx="pathX(lambda)" :cy="pathY(result.lasso[1])" r="5" class="path-secondary-dot" />
          <text x="503" y="300" class="axis-label">λ</text>
          <text x="389" y="62" class="path-primary-label">β₁</text>
          <text x="389" :y="Math.min(268, pathY(secondarySignal) + 15)" class="path-secondary-label">β₂</text>
        </svg>
        <figcaption>
          <span class="legend lasso" />L1 菱形 / Lasso
          <span class="legend ridge" />L2 圆 / Ridge
          <span class="legend ols" />无惩罚中心
          <span class="legend cursor" />当前 λ
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>惩罚 λ <output>{{ lambda.toFixed(2) }}</output></span>
        <input v-model.number="lambda" type="range" min="0" max="2.2" step="0.02">
      </label>
      <label>
        <span>第二传感器证据 z₂ <output>{{ secondarySignal.toFixed(2) }}</output></span>
        <input v-model.number="secondarySignal" type="range" min="0.20" max="1.40" step="0.02">
      </label>
      <p>让 λ 穿过 z₂：Lasso 路径会在这一刻落到横轴并保持 β₂=0；Ridge 点仍位于圆边界上，两项只会连续接近 0。λ 再穿过 z₁ 后，Lasso 才把两项都关闭。</p>
    </template>
    <template #metrics>
      <div><span>Lasso 活跃系数</span><strong>{{ result.activeCount }} / 2</strong></div>
      <div><span>β₂：Lasso</span><strong>{{ result.lasso[1].toFixed(3) }}</strong></div>
      <div><span>β₂：Ridge</span><strong>{{ result.ridge[1].toFixed(3) }}</strong></div>
      <div><span>约束半径 L1 / L2</span><strong>{{ result.lassoRadius.toFixed(2) }} / {{ result.ridgeRadius.toFixed(2) }}</strong></div>
    </template>
    <template #note>
      解析式要求列已标准化且正交。一般设计中仍由 KKT 条件决定 0：若 |xⱼᵀr|≤λ，0 就是合法最优坐标。菱形拐角提供几何直觉，但相关传感器会改变椭圆方向，并可能让活跃集合在重采样间不稳定。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.geometry-board{margin:0}.geometry-board svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:10px;background:#10182e}.board-bg{fill:#10182e}.axis{stroke:#69789d;stroke-width:1.4}.grid,.path-grid{stroke:#263554;stroke-width:1}.divider{stroke:#344675}.loss-contour{fill:none;stroke-width:1.5}.lasso-loss{stroke:#e15b4b;stroke-dasharray:5 4}.ridge-loss{stroke:#72d5ce;stroke-dasharray:5 4}.l1-boundary{fill:rgba(225,91,75,.08);stroke:#e15b4b;stroke-width:2.5}.l2-boundary{fill:rgba(114,213,206,.05);stroke:#72d5ce;stroke-width:2}.ols-point{fill:#eef2ff;stroke:#8796ba;stroke-width:2}.lasso-point{fill:#f4c95d;stroke:#fff;stroke-width:2}.ridge-point{fill:#72d5ce;stroke:#fff;stroke-width:2}.path-primary,.path-secondary{fill:none;stroke-width:3}.path-primary{stroke:#f4c95d}.path-secondary{stroke:#e15b4b}.lambda-cursor{stroke:#eef2ff;stroke-width:2;stroke-dasharray:5 4}.path-primary-dot{fill:#f4c95d}.path-secondary-dot{fill:#e15b4b}.geometry-board text{font:700 9.5px ui-monospace,monospace}.panel-title{fill:#dfe5f7;font-size:10.5px}.axis-label{fill:#8f9cbc}.ols-label{fill:#c5cee4}.lasso-label,.path-primary-label{fill:#f4c95d}.ridge-label{fill:#72d5ce}.path-secondary-label{fill:#e15b4b}.geometry-board figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:11px;height:11px;margin-right:4px;vertical-align:middle}.legend.lasso{border:2px solid #e15b4b;transform:rotate(45deg)}.legend.ridge{border:2px solid #72d5ce;border-radius:50%}.legend.ols{border-radius:50%;background:#eef2ff}.legend.cursor{height:2px;border-top:2px dashed #eef2ff}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}
@media(max-width:560px){.geometry-board{overflow-x:auto;overscroll-behavior-inline:contain}.geometry-board svg{width:560px;max-width:none}.geometry-board figcaption{min-width:560px}}
</style>
