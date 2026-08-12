<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const queryX = ref(.05)
const queryY = ref(.35)
const priorA = ref(.56)
const overlap = ref(.92)

const componentA = { x: -1.25, y: -.35 }
const componentB = { x: 1.18, y: .82 }

function logGaussian(x: number, y: number, mean: { x: number; y: number }, sigma: number) {
  const distanceSq = (x - mean.x) ** 2 + (y - mean.y) ** 2
  return -Math.log(2 * Math.PI * sigma * sigma) - distanceSq / (2 * sigma * sigma)
}

const evidence = computed(() => {
  const sigmaA = .72 * overlap.value
  const sigmaB = .88 * overlap.value
  const logDensityA = logGaussian(queryX.value, queryY.value, componentA, sigmaA)
  const logDensityB = logGaussian(queryX.value, queryY.value, componentB, sigmaB)
  const logJointA = Math.log(priorA.value) + logDensityA
  const logJointB = Math.log(1 - priorA.value) + logDensityB
  const peak = Math.max(logJointA, logJointB)
  const logEvidence = peak + Math.log(Math.exp(logJointA - peak) + Math.exp(logJointB - peak))
  const responsibilityA = Math.exp(logJointA - logEvidence)
  const responsibilityB = 1 - responsibilityA
  const entropy = -[responsibilityA, responsibilityB]
    .reduce((sum, value) => sum + value * Math.log(Math.max(value, 1e-15)), 0)
  return {
    sigmaA,
    sigmaB,
    logDensityA,
    logDensityB,
    logJointA,
    logJointB,
    logEvidence,
    responsibilityA,
    responsibilityB,
    entropy,
  }
})

const mapX = (value: number) => 320 + value * 92
const mapY = (value: number) => 202 - value * 70
const probe = computed(() => ({ x: mapX(queryX.value), y: mapY(queryY.value) }))
</script>

<template>
  <MlExperimentFrame
    index="24-A"
    title="让两个隐态争夺同一条主轴记录"
    question="责任度不是到中心距离的别名。移动探针并改变先验与成分重叠，观察‘先验 + 类条件密度’如何共同决定 E 步后验。"
    formula="log γᵢₖ = log πₖ + log N(xᵢ|μₖ,Σₖ) − log ∑ⱼ πⱼN(xᵢ|μⱼ,Σⱼ)"
  >
    <template #visual>
      <div class="responsibility-scroll">
        <svg viewBox="0 0 640 368" role="img" :aria-label="`探针由隐态A承担${(evidence.responsibilityA * 100).toFixed(1)}%，隐态B承担${(evidence.responsibilityB * 100).toFixed(1)}%`">
          <defs>
            <linearGradient id="em-flow-a" x1="0" x2="1"><stop stop-color="#23c9bd" stop-opacity=".12"/><stop offset="1" stop-color="#23c9bd" stop-opacity=".9"/></linearGradient>
            <linearGradient id="em-flow-b" x1="1" x2="0"><stop stop-color="#b56af0" stop-opacity=".12"/><stop offset="1" stop-color="#d477ff" stop-opacity=".9"/></linearGradient>
          </defs>
          <rect width="640" height="368" rx="15" class="flow-panel" />
          <text x="24" y="29" class="flow-eyebrow">LATENT-LINEAGE BAY · POSTERIOR RESPONSIBILITY ROUTING</text>
          <g class="flow-grid">
            <line v-for="x in [136,228,320,412,504]" :key="`x-${x}`" :x1="x" :x2="x" y1="54" y2="270" />
            <line v-for="y in [62,132,202,272]" :key="`y-${y}`" x1="90" x2="550" :y1="y" :y2="y" />
          </g>
          <ellipse :cx="mapX(componentA.x)" :cy="mapY(componentA.y)" :rx="68 * evidence.sigmaA" :ry="44 * evidence.sigmaA" class="race race-a" />
          <ellipse :cx="mapX(componentB.x)" :cy="mapY(componentB.y)" :rx="68 * evidence.sigmaB" :ry="44 * evidence.sigmaB" class="race race-b" />
          <line :x1="mapX(componentA.x)" :y1="mapY(componentA.y)" :x2="probe.x" :y2="probe.y" class="route route-a" :style="{ strokeWidth: 2 + 14 * evidence.responsibilityA }" />
          <line :x1="mapX(componentB.x)" :y1="mapY(componentB.y)" :x2="probe.x" :y2="probe.y" class="route route-b" :style="{ strokeWidth: 2 + 14 * evidence.responsibilityB }" />
          <circle :cx="mapX(componentA.x)" :cy="mapY(componentA.y)" r="9" class="hub hub-a" />
          <circle :cx="mapX(componentB.x)" :cy="mapY(componentB.y)" r="9" class="hub hub-b" />
          <circle :cx="probe.x" :cy="probe.y" r="8" class="probe" />
          <text :x="mapX(componentA.x)" :y="mapY(componentA.y) + 27" text-anchor="middle" class="hub-label label-a">隐态 A</text>
          <text :x="mapX(componentB.x)" :y="mapY(componentB.y) - 22" text-anchor="middle" class="hub-label label-b">隐态 B</text>
          <text :x="probe.x" :y="probe.y - 15" text-anchor="middle" class="probe-label">待分配记录</text>
          <g transform="translate(38 295)">
            <rect width="268" height="48" rx="5" class="ledger ledger-a" />
            <text x="12" y="18" class="ledger-title">A · LOG JOINT</text>
            <text x="12" y="36" class="ledger-value">{{ evidence.logJointA.toFixed(3) }} → γ={{ (evidence.responsibilityA * 100).toFixed(1) }}%</text>
          </g>
          <g transform="translate(334 295)">
            <rect width="268" height="48" rx="5" class="ledger ledger-b" />
            <text x="12" y="18" class="ledger-title">B · LOG JOINT</text>
            <text x="12" y="36" class="ledger-value">{{ evidence.logJointB.toFixed(3) }} → γ={{ (evidence.responsibilityB * 100).toFixed(1) }}%</text>
          </g>
        </svg>
      </div>
    </template>

    <template #controls>
      <label class="control"><span>探针 x <output>{{ queryX.toFixed(2) }}</output></span><input v-model.number="queryX" aria-label="探针横坐标" type="range" min="-2.3" max="2.3" step="0.02"></label>
      <label class="control"><span>探针 y <output>{{ queryY.toFixed(2) }}</output></span><input v-model.number="queryY" aria-label="探针纵坐标" type="range" min="-1.8" max="2" step="0.02"></label>
      <label class="control"><span>隐态 A 先验 πₐ <output>{{ priorA.toFixed(2) }}</output></span><input v-model.number="priorA" aria-label="隐态A先验权重" type="range" min="0.08" max="0.92" step="0.01"></label>
      <label class="control"><span>成分重叠倍率 <output>{{ overlap.toFixed(2) }}</output></span><input v-model.number="overlap" aria-label="成分重叠倍率" type="range" min="0.45" max="1.55" step="0.01"></label>
    </template>

    <template #metrics>
      <div><span>责任度 γₐ</span><strong>{{ (evidence.responsibilityA * 100).toFixed(1) }}%</strong></div>
      <div><span>责任度 γᵦ</span><strong>{{ (evidence.responsibilityB * 100).toFixed(1) }}%</strong></div>
      <div><span>责任度熵</span><strong>{{ evidence.entropy.toFixed(3) }}</strong></div>
      <div><span>log evidence</span><strong>{{ evidence.logEvidence.toFixed(3) }}</strong></div>
    </template>

    <template #note>流带宽度就是责任度。探针位于两成分重叠区时，责任度会保留不确定性；提高某成分先验会移动后验，但不会改写该成分对探针的类条件密度。把责任度直接当作已确认故障标签，会丢掉这层区别。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.responsibility-scroll{overflow-x:auto}.responsibility-scroll svg{display:block;width:100%;height:auto}.flow-panel{fill:#0b1218}.flow-grid line{stroke:#23343d;stroke-width:1}.race{stroke-width:2.5;stroke-dasharray:8 5}.race-a{fill:#23c9bd;fill-opacity:.08;stroke:#38d8cc}.race-b{fill:#b56af0;fill-opacity:.08;stroke:#d477ff}.route{stroke-linecap:round;opacity:.7}.route-a{stroke:url(#em-flow-a)}.route-b{stroke:url(#em-flow-b)}.hub{stroke-width:2}.hub-a{fill:#23c9bd;stroke:#c9fff9}.hub-b{fill:#b56af0;stroke:#f1d8ff}.probe{fill:#efb449;stroke:#fff0c6;stroke-width:3}.flow-eyebrow,.hub-label,.probe-label,.ledger-title,.ledger-value{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.flow-eyebrow{fill:#74a4b1;font-size:9px;letter-spacing:.12em}.hub-label{font-size:10px;font-weight:800}.label-a{fill:#72e6dc}.label-b{fill:#dc9bff}.probe-label{fill:#ffd57c;font-size:9px}.ledger{fill:#111d25;stroke-width:1}.ledger-a{stroke:#287c78}.ledger-b{stroke:#764b91}.ledger-title{fill:#809aa5;font-size:8px;letter-spacing:.08em}.ledger-value{fill:#e6eef0;font-size:11px;font-weight:750}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebdc5;font-size:11px}.control output{color:#ffc864;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#23c9bd}input:focus-visible{outline:3px solid #d477ff;outline-offset:2px}@media(max-width:560px){.responsibility-scroll{margin-right:-18px;padding-right:18px}.responsibility-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
