<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const angleDeg = ref(18)
const useCenter = ref(true)

const latent = [-2.7, -2.35, -2.05, -1.7, -1.35, -1.05, -.72, -.38, -.08, .25, .58, .94, 1.3, 1.68, 2.08, 2.48, 2.82]
const noise = [.05, -.15, .19, -.08, .13, -.21, .08, -.04, .18, -.12, .04, .16, -.18, .07, -.05, .14, -.09]
const trueAngle = 32 * Math.PI / 180
const rawPoints = latent.map((value, index) => ({
  x: 1.10 + value * Math.cos(trueAngle) - noise[index] * Math.sin(trueAngle),
  y: -1.65 + value * Math.sin(trueAngle) + noise[index] * Math.cos(trueAngle),
}))
const mean = {
  x: rawPoints.reduce((sum, point) => sum + point.x, 0) / rawPoints.length,
  y: rawPoints.reduce((sum, point) => sum + point.y, 0) / rawPoints.length,
}

const points = computed(() => rawPoints.map(point => useCenter.value
  ? { x: point.x - mean.x, y: point.y - mean.y }
  : point))
const direction = computed(() => {
  const angle = angleDeg.value * Math.PI / 180
  return { x: Math.cos(angle), y: Math.sin(angle) }
})
const projected = computed(() => points.value.map(point => {
  const score = point.x * direction.value.x + point.y * direction.value.y
  return {
    source: point,
    score,
    target: { x: score * direction.value.x, y: score * direction.value.y },
  }
}))
const metrics = computed(() => {
  const total = points.value.reduce((sum, point) => sum + point.x ** 2 + point.y ** 2, 0) / points.value.length
  const captured = projected.value.reduce((sum, point) => sum + point.score ** 2, 0) / projected.value.length
  const residual = projected.value.reduce((sum, point) => sum
    + (point.source.x - point.target.x) ** 2
    + (point.source.y - point.target.y) ** 2, 0) / projected.value.length
  return { total, captured, residual, gap: total - captured - residual }
})

function energyAt(angle: number) {
  const radians = angle * Math.PI / 180
  const x = Math.cos(radians)
  const y = Math.sin(radians)
  return points.value.reduce((sum, point) => sum + (point.x * x + point.y * y) ** 2, 0) / points.value.length
}

const bestAngle = computed(() => {
  let best = -90
  let bestEnergy = -Infinity
  for (let angle = -90; angle <= 90; angle += .25) {
    const energy = energyAt(angle)
    if (energy > bestEnergy) {
      bestEnergy = energy
      best = angle
    }
  }
  return best
})

const mapX = (value: number) => 320 + value * 70
const mapY = (value: number) => 190 - value * 70
const lineStart = computed(() => ({ x: mapX(-4 * direction.value.x), y: mapY(-4 * direction.value.y) }))
const lineEnd = computed(() => ({ x: mapX(4 * direction.value.x), y: mapY(4 * direction.value.y) }))

function snapToBest() {
  angleDeg.value = Math.round(bestAngle.value)
}
</script>

<template>
  <MlExperimentFrame
    index="25-B"
    title="把投影龙门转到最小残差位置"
    question="旋转候选轴，观察每个样本到投影线的正交残差。切断中心校准后，龙门会追逐相对原点的能量，而不是围绕均值的变化。"
    formula="总能量 = 投影能量 + 正交残差能量"
  >
    <template #visual>
      <div class="gantry-scroll">
        <svg viewBox="0 0 640 380" role="img" :aria-label="`候选方向${angleDeg}度，投影能量${metrics.captured.toFixed(2)}，残差能量${metrics.residual.toFixed(2)}`">
          <rect width="640" height="380" rx="14" class="deck" />
          <text x="22" y="28" class="eyebrow">ORTHOGONAL METROLOGY GANTRY · ENERGY LEDGER</text>
          <g class="grid">
            <line v-for="x in [110,180,250,320,390,460,530]" :key="`gx-${x}`" :x1="x" :x2="x" y1="48" y2="326" />
            <line v-for="y in [50,120,190,260,330]" :key="`gy-${y}`" x1="74" x2="566" :y1="y" :y2="y" />
          </g>
          <line x1="74" x2="566" y1="190" y2="190" class="datum" />
          <line x1="320" x2="320" y1="48" y2="330" class="datum" />
          <line :x1="lineStart.x" :y1="lineStart.y" :x2="lineEnd.x" :y2="lineEnd.y" class="principal-rail" />
          <g v-for="(point, index) in projected" :key="index">
            <line :x1="mapX(point.source.x)" :y1="mapY(point.source.y)" :x2="mapX(point.target.x)" :y2="mapY(point.target.y)" class="residual-wire" />
            <circle :cx="mapX(point.target.x)" :cy="mapY(point.target.y)" r="3.2" class="projection" />
            <circle :cx="mapX(point.source.x)" :cy="mapY(point.source.y)" r="5" class="sample" />
          </g>
          <g :transform="`translate(${mapX(useCenter ? 0 : mean.x)} ${mapY(useCenter ? 0 : mean.y)})`">
            <path d="M-8 0H8M0-8V8" class="mean-mark" />
          </g>
          <g transform="translate(82 338)">
            <rect width="476" height="24" rx="3" class="energy-track" />
            <rect :width="476 * metrics.captured / Math.max(metrics.total, 1e-9)" height="24" rx="3" class="energy-captured" />
            <text x="12" y="16" class="energy-label">CAPTURED {{ (100 * metrics.captured / metrics.total).toFixed(1) }}%</text>
            <text x="464" y="16" text-anchor="end" class="energy-label">RESIDUAL {{ (100 * metrics.residual / metrics.total).toFixed(1) }}%</text>
          </g>
        </svg>
      </div>
    </template>

    <template #controls>
      <label class="control">
        <span>候选方向 <output>{{ angleDeg }}°</output></span>
        <input v-model.number="angleDeg" type="range" min="-90" max="90" step="1" aria-label="候选投影方向角度">
      </label>
      <button type="button" :aria-pressed="useCenter" @click="useCenter = !useCenter">
        {{ useCenter ? '中心校准：接通' : '中心校准：旁路' }}
      </button>
      <button type="button" class="snap" @click="snapToBest">吸附到当前最优 {{ bestAngle.toFixed(1) }}°</button>
    </template>

    <template #metrics>
      <div><span>总二阶能量</span><strong>{{ metrics.total.toFixed(3) }}</strong></div>
      <div><span>投影能量</span><strong>{{ metrics.captured.toFixed(3) }}</strong></div>
      <div><span>残差能量</span><strong>{{ metrics.residual.toFixed(3) }}</strong></div>
      <div><span>Pythagoras 缺口</span><strong>{{ metrics.gap.toExponential(1) }}</strong></div>
    </template>

    <template #note>青色钢轨是候选主方向，琥珀点是样本，红线是正交残差。中心校准接通时，最大投影能量与最小残差在同一角度出现；旁路后，均值相对原点的能量会主导方向。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.gantry-scroll{overflow-x:auto}.gantry-scroll svg{display:block;width:100%;height:auto}.deck{fill:#091419}.grid line{stroke:#183039;stroke-width:1}.datum{stroke:#42616a;stroke-width:1.2}.principal-rail{stroke:#43d6c5;stroke-width:5;stroke-linecap:round}.residual-wire{stroke:#ff7066;stroke-width:1.5;stroke-dasharray:4 3;opacity:.72}.sample{fill:#f3ba4f;stroke:#fff0bd;stroke-width:1.6}.projection{fill:#43d6c5}.mean-mark{stroke:#e9f6f4;stroke-width:2}.eyebrow,.energy-label{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#7ca1aa;font-size:9px;letter-spacing:.12em}.energy-track{fill:#552a2c}.energy-captured{fill:#168b80}.energy-label{fill:#f2f6f5;font-size:9px;font-weight:750;letter-spacing:.05em}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#b5c4ca;font-size:11px}.control output{color:#f3ba4f;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#43d6c5}button{min-height:44px;border:1px solid #4b6870;border-radius:6px;color:#dbe9e8;background:#122329;font:720 11px/1.25 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#43d6c5;color:#65ebdb;background:#153934}.snap{border-color:#a77d31;color:#ffd47b;background:#2b2417}button:focus-visible,input:focus-visible{outline:3px solid #f3ba4f;outline-offset:2px}@media(max-width:560px){.gantry-scroll{margin-right:-18px;padding-right:18px}.gantry-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
