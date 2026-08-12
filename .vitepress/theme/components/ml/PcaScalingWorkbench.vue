<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type ScaleMode = 'covariance' | 'correlation'

const mode = ref<ScaleMode>('covariance')
const reportInNewtons = ref(false)
const standardizedPairs = [
  [-2.2, -1.8], [-1.9, -1.5], [-1.55, -1.7], [-1.25, -.8],
  [-.9, -1.0], [-.55, -.2], [-.2, -.45], [.1, .35],
  [.45, .1], [.8, .85], [1.15, .65], [1.45, 1.5],
  [1.8, 1.35], [2.15, 2.05],
]

function covariance2d(points: number[][]) {
  const meanX = points.reduce((sum, point) => sum + point[0], 0) / points.length
  const meanY = points.reduce((sum, point) => sum + point[1], 0) / points.length
  const centered = points.map(point => [point[0] - meanX, point[1] - meanY])
  const a = centered.reduce((sum, point) => sum + point[0] ** 2, 0) / points.length
  const b = centered.reduce((sum, point) => sum + point[0] * point[1], 0) / points.length
  const c = centered.reduce((sum, point) => sum + point[1] ** 2, 0) / points.length
  return { a, b, c }
}

function principalDirection(a: number, b: number, c: number) {
  const gap = Math.sqrt((a - c) ** 2 + 4 * b ** 2)
  const first = (a + c + gap) / 2
  const second = (a + c - gap) / 2
  let x = Math.abs(b) > 1e-12 ? b : 1
  let y = Math.abs(b) > 1e-12 ? first - a : 0
  const norm = Math.hypot(x, y)
  x /= norm
  y /= norm
  if (x < 0) {
    x *= -1
    y *= -1
  }
  return { x, y, first, second, angle: Math.atan2(y, x) * 180 / Math.PI }
}

function solve(unitFactor: number, scaleMode: ScaleMode) {
  const raw = standardizedPairs.map(([x, y]) => [
    (42 + 4.8 * x) * unitFactor,
    18.4 + .55 * y,
  ])
  if (scaleMode === 'correlation') {
    const rawCov = covariance2d(raw)
    const sx = Math.sqrt(rawCov.a)
    const sy = Math.sqrt(rawCov.c)
    const normalized = raw.map(point => [point[0] / sx, point[1] / sy])
    const cov = covariance2d(normalized)
    return { cov, direction: principalDirection(cov.a, cov.b, cov.c), sdRatio: 1 }
  }
  const cov = covariance2d(raw)
  return {
    cov,
    direction: principalDirection(cov.a, cov.b, cov.c),
    sdRatio: Math.sqrt(cov.a / cov.c),
  }
}

const result = computed(() => solve(reportInNewtons.value ? 1000 : 1, mode.value))
const baseResult = computed(() => solve(1, mode.value))
const angleDrift = computed(() => Math.abs(result.value.direction.angle - baseResult.value.direction.angle))
const evr = computed(() => result.value.direction.first / (result.value.direction.first + result.value.direction.second))
const loadingBars = computed(() => [
  { label: reportInNewtons.value ? '峰值力（N）' : '峰值力（kN）', value: result.value.direction.x, color: '#43d6c5' },
  { label: '入座深度（mm）', value: result.value.direction.y, color: '#f3ba4f' },
])
</script>

<template>
  <MlExperimentFrame
    index="25-A"
    title="在量纲配电板上切换协方差与相关系数"
    question="同一列只把 kN 改报 N，物理信息没有改变。观察协方差 PCA 为何仍会转向，以及训练标准化如何让相关系数 PCA 保持同一几何。"
    formula="Sraw = XcᵀXc/N；R = D⁻¹/² Sraw D⁻¹/²"
  >
    <template #visual>
      <div class="scale-scroll">
        <svg viewBox="0 0 640 350" role="img" :aria-label="`${mode === 'covariance' ? '协方差' : '相关系数'}PCA，第一方向角${result.direction.angle.toFixed(2)}度`">
          <rect width="640" height="350" rx="14" class="deck" />
          <text x="22" y="28" class="eyebrow">UNIT SWITCHBOARD · COVARIANCE CONTRACT</text>
          <g transform="translate(34 54)">
            <rect width="270" height="244" rx="8" class="module" />
            <text x="18" y="28" class="module-title">{{ mode === 'covariance' ? 'RAW COVARIANCE' : 'TRAIN-FOLD CORRELATION' }}</text>
            <text x="18" y="53" class="matrix-label">{{ mode === 'covariance' ? 'S' : 'R' }} =</text>
            <g transform="translate(72 64)">
              <path d="M8 0H0V82H8M152 0H160V82H152" class="bracket" />
              <text x="44" y="28" text-anchor="middle" class="matrix-value">{{ result.cov.a.toPrecision(3) }}</text>
              <text x="116" y="28" text-anchor="middle" class="matrix-value">{{ result.cov.b.toPrecision(3) }}</text>
              <text x="44" y="66" text-anchor="middle" class="matrix-value">{{ result.cov.b.toPrecision(3) }}</text>
              <text x="116" y="66" text-anchor="middle" class="matrix-value">{{ result.cov.c.toPrecision(3) }}</text>
            </g>
            <line x1="18" x2="252" y1="168" y2="168" class="divider" />
            <text x="18" y="194" class="small-label">STANDARD DEVIATION RATIO</text>
            <text x="252" y="194" text-anchor="end" class="large-value">{{ result.sdRatio.toPrecision(4) }}×</text>
            <text x="18" y="224" class="small-label">UNIT PATH</text>
            <text x="252" y="224" text-anchor="end" class="path-value">kN → {{ reportInNewtons ? 'N' : 'kN' }}</text>
          </g>
          <g transform="translate(324 54)">
            <rect width="282" height="244" rx="8" class="module" />
            <text x="18" y="28" class="module-title">PC1 LOADING BUS</text>
            <g v-for="(bar, index) in loadingBars" :key="bar.label" :transform="`translate(18 ${66 + index * 72})`">
              <text x="0" y="0" class="bar-label">{{ bar.label }}</text>
              <rect x="0" y="14" width="238" height="20" rx="3" class="bar-track" />
              <rect x="0" y="14" :width="238 * Math.abs(bar.value)" height="20" rx="3" :fill="bar.color" />
              <text x="246" y="29" text-anchor="end" class="bar-value">{{ bar.value.toFixed(3) }}</text>
            </g>
            <line x1="18" x2="264" y1="194" y2="194" class="divider" />
            <text x="18" y="221" class="small-label">ANGLE FROM FEATURE 1</text>
            <text x="264" y="221" text-anchor="end" class="large-value">{{ result.direction.angle.toFixed(2) }}°</text>
          </g>
          <g transform="translate(34 316)">
            <circle cx="7" cy="7" r="5" :class="angleDrift < .01 ? 'lamp-ok' : 'lamp-warn'" />
            <text x="22" y="11" class="status-text">单位切换造成方向漂移 {{ angleDrift.toFixed(3) }}° · {{ angleDrift < .01 ? 'INVARIANT' : 'UNIT-DEPENDENT' }}</text>
          </g>
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="button-bank" role="group" aria-label="PCA尺度模式">
        <button type="button" :aria-pressed="mode === 'covariance'" @click="mode = 'covariance'">协方差 PCA</button>
        <button type="button" :aria-pressed="mode === 'correlation'" @click="mode = 'correlation'">相关系数 PCA</button>
      </div>
      <button type="button" class="unit-switch" :aria-pressed="reportInNewtons" @click="reportInNewtons = !reportInNewtons">
        峰值力单位：{{ reportInNewtons ? 'N' : 'kN' }}
      </button>
    </template>

    <template #metrics>
      <div><span>PC1 方向角</span><strong>{{ result.direction.angle.toFixed(2) }}°</strong></div>
      <div><span>PC1 解释方差</span><strong>{{ (100 * evr).toFixed(2) }}%</strong></div>
      <div><span>标准差比</span><strong>{{ result.sdRatio.toPrecision(4) }}×</strong></div>
      <div><span>单位方向漂移</span><strong>{{ angleDrift.toFixed(3) }}°</strong></div>
    </template>

    <template #note>协方差 PCA 直接继承单位尺度；把 kN 改成 N 会把峰值力方差放大一百万倍。相关系数 PCA 用训练期标准差消去这种单位倍率，但也把两列的边缘方差信息一并删除。选择哪条线路必须由任务决定。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.scale-scroll{overflow-x:auto}.scale-scroll svg{display:block;width:100%;height:auto}.deck{fill:#091419}.module{fill:#101f25;stroke:#31515a;stroke-width:1.2}.eyebrow,.module-title,.matrix-label,.matrix-value,.small-label,.large-value,.path-value,.bar-label,.bar-value,.status-text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#7ca1aa;font-size:9px;letter-spacing:.12em}.module-title{fill:#43d6c5;font-size:9px;font-weight:800;letter-spacing:.09em}.matrix-label{fill:#9ab0b6;font-size:12px}.matrix-value{fill:#e8f4f2;font-size:12px;font-weight:700}.bracket{fill:none;stroke:#718f97;stroke-width:1.5}.divider{stroke:#2d4850}.small-label{fill:#78939a;font-size:8px;letter-spacing:.08em}.large-value{fill:#f3ba4f;font-size:13px;font-weight:800}.path-value{fill:#e8f4f2;font-size:11px;font-weight:700}.bar-label{fill:#aabcc0;font-size:9px}.bar-track{fill:#26383d}.bar-value{fill:#eff7f6;font-size:9px;font-weight:750}.lamp-ok{fill:#43d6c5}.lamp-warn{fill:#ff7066}.status-text{fill:#a9bdc1;font-size:9px;letter-spacing:.04em}.button-bank{display:grid;grid-template-columns:1fr 1fr;gap:8px}button{min-height:44px;border:1px solid #4b6870;border-radius:6px;color:#dbe9e8;background:#122329;font:720 10px/1.25 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#43d6c5;color:#65ebdb;background:#153934}.unit-switch{border-color:#a77d31;color:#ffd47b;background:#2b2417}button:focus-visible{outline:3px solid #f3ba4f;outline-offset:2px}@media(max-width:560px){.scale-scroll{margin-right:-18px;padding-right:18px}.scale-scroll svg{width:600px}.button-bank{grid-template-columns:1fr}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
