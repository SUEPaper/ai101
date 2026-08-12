<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const audited = ref(40)
const misses = ref(2)
const candidateRate = ref(0.075)
const duplicated = ref(false)

const maxMisses = computed(() => Math.max(1, Math.floor(audited.value * 0.16)))
watch(audited, () => { misses.value = Math.min(misses.value, maxMisses.value) })

const mle = computed(() => misses.value / audited.value)
const effectiveAudits = computed(() => duplicated.value ? Math.max(4, Math.round(audited.value / 6)) : audited.value)
const logLikelihood = (theta: number) => {
  const success = misses.value === 0 ? 0 : misses.value * Math.log(theta)
  const failure = audited.value === misses.value ? 0 : (audited.value - misses.value) * Math.log1p(-theta)
  return success + failure
}
const maximumLogLikelihood = computed(() => logLikelihood(mle.value))
const deltaNll = computed(() => maximumLogLikelihood.value - logLikelihood(candidateRate.value))
const observedCurvature = computed(() => {
  if (misses.value === 0 || misses.value === audited.value) return Number.POSITIVE_INFINITY
  return audited.value / (mle.value * (1 - mle.value))
})

const plot = { left: 54, right: 592, top: 105, bottom: 280 }
const x = (theta: number) => plot.left + Math.min(theta, .18) / .18 * (plot.right - plot.left)
const profilePath = (scale = 1) => Array.from({ length: 181 }, (_, index) => {
  const theta = .0002 + index / 180 * .1798
  const relative = Math.exp(Math.max(-35, scale * (logLikelihood(theta) - maximumLogLikelihood.value)))
  const px = x(theta)
  const py = plot.bottom - relative * (plot.bottom - plot.top - 8)
  return `${index === 0 ? 'M' : 'L'}${px.toFixed(2)} ${py.toFixed(2)}`
}).join(' ')
const iidPath = computed(() => profilePath())
const adjustedPath = computed(() => profilePath(effectiveAudits.value / audited.value))

const plates = computed(() => Array.from({ length: 12 }, (_, index) => ({
  missed: index < Math.round(12 * misses.value / audited.value),
  repeated: duplicated.value && index % 3 !== 0,
  x: 60 + index * 45,
})))
const formatCurvature = computed(() => Number.isFinite(observedCurvature.value) ? observedCurvature.value.toFixed(0) : '边界')
</script>

<template>
  <MlExperimentFrame
    index="22-A"
    title="沿资格样本扫描整条似然剖面"
    question="漏检比例相同但审核量增加时，峰的位置与宽度分别怎样变化？把同一挑战片段重复六次，独立样本假设又会怎样误导曲率？"
    formula="ℓ(θ)=S log θ+(n−S) log(1−θ)"
  >
    <template #visual>
      <div class="scanner-scroll">
        <svg viewBox="0 0 640 340" role="img" :aria-label="`审核${audited}件、漏检${misses}件，最大似然估计${(100*mle).toFixed(2)}%，候选参数负对数似然差${deltaNll.toFixed(2)}`">
          <rect width="640" height="340" rx="14" class="panel" />
          <text x="24" y="27" class="eyebrow">AOI QUALIFICATION · LIKELIHOOD SCAN</text>
          <rect x="36" y="43" width="568" height="44" rx="7" class="carrier" />
          <line x1="48" x2="592" y1="74" y2="74" class="carrier-rail" />
          <g v-for="(plate,index) in plates" :key="index">
            <rect :x="plate.x" y="52" width="26" height="22" rx="3" class="plate" :class="{ missed:plate.missed, repeated:plate.repeated }" />
            <text :x="plate.x+13" y="67" text-anchor="middle" class="plate-id">{{ duplicated ? (index % 3) + 1 : index + 1 }}</text>
          </g>
          <text x="584" y="60" text-anchor="end" class="count">S={{ misses }} / n={{ audited }}</text>

          <line v-for="tick in [0,.03,.06,.09,.12,.15,.18]" :key="tick" :x1="x(tick)" :x2="x(tick)" :y1="plot.top" :y2="plot.bottom" class="grid" />
          <line :x1="plot.left" :x2="plot.right" :y1="plot.bottom" :y2="plot.bottom" class="axis" />
          <path v-if="duplicated" :d="adjustedPath" class="profile adjusted" />
          <path :d="iidPath" class="profile iid" />
          <line :x1="x(mle)" :x2="x(mle)" :y1="plot.top" :y2="plot.bottom" class="mle-marker" />
          <line :x1="x(candidateRate)" :x2="x(candidateRate)" :y1="plot.top" :y2="plot.bottom" class="candidate-marker" />
          <circle :cx="x(candidateRate)" :cy="plot.bottom-Math.exp(Math.max(-35,logLikelihood(candidateRate)-maximumLogLikelihood))*(plot.bottom-plot.top-8)" r="6" class="candidate-dot" />
          <text v-for="tick in [0,.03,.06,.09,.12,.15,.18]" :key="`label-${tick}`" :x="x(tick)" y="300" text-anchor="middle" class="tick">{{ Math.round(tick*100) }}%</text>
          <text x="54" y="322" class="legend iid-text">实线：把每条记录当独立证据</text>
          <text v-if="duplicated" x="344" y="322" class="legend adjusted-text">虚线：按独立挑战族折算</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>审核件数 n <output>{{ audited }}</output></span><input v-model.number="audited" aria-label="审核件数" type="range" min="8" max="120" step="4"></label>
      <label class="control"><span>漏检件数 S <output>{{ misses }}</output></span><input v-model.number="misses" aria-label="漏检件数" type="range" min="0" :max="maxMisses" step="1"></label>
      <label class="control"><span>候选漏检率 θ <output>{{ (100*candidateRate).toFixed(1) }}%</output></span><input v-model.number="candidateRate" aria-label="候选漏检率" type="range" min="0.005" max="0.15" step="0.005"></label>
      <button type="button" :class="{ active:duplicated }" :aria-pressed="duplicated" @click="duplicated=!duplicated">{{ duplicated ? '已接入：重复挑战片段' : '重复每个挑战片段六次' }}</button>
    </template>
    <template #metrics>
      <div><span>MLE 漏检率</span><strong>{{ (100*mle).toFixed(2) }}%</strong></div>
      <div><span>候选 ΔNLL</span><strong>{{ deltaNll.toFixed(3) }}</strong></div>
      <div><span>峰顶负曲率</span><strong>{{ formatCurvature }}</strong></div>
      <div><span>名义 / 折算证据</span><strong>{{ audited }} / {{ effectiveAudits }}</strong></div>
    </template>
    <template #note>比例决定峰顶位置，独立证据量决定峰宽。重复同一缺陷图片不会创造新的工况信息；若仍按 iid 相乘，实线会虚假变尖。S=0 时 MLE 落在边界，基于峰顶二阶近似得到的“标准误为 0”不可信。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.scanner-scroll{overflow-x:auto}.scanner-scroll svg{display:block;width:100%;height:auto}.panel{fill:#111923}.carrier{fill:#1a2a35;stroke:#49606c}.carrier-rail,.axis{stroke:#7f939b;stroke-width:2}.plate{fill:#258594;stroke:#76d0d8;stroke-width:1.5}.plate.missed{fill:#bd6245;stroke:#f19a76}.plate.repeated{stroke:#a98de5;stroke-dasharray:3 2}.plate-id,.eyebrow,.count,.tick,.legend{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.plate-id{fill:#fff;font-size:7px;font-weight:800}.eyebrow{fill:#6bc4d8;font-size:10px;letter-spacing:.12em}.count{fill:#f0bd59;font-size:9px}.grid{stroke:#293b45;stroke-width:1}.profile{fill:none;stroke-linejoin:round}.profile.iid{stroke:#f0bd59;stroke-width:4}.profile.adjusted{stroke:#a98de5;stroke-width:3;stroke-dasharray:7 5}.mle-marker{stroke:#6bc4d8;stroke-width:2;stroke-dasharray:5 4}.candidate-marker{stroke:#e77455;stroke-width:2}.candidate-dot{fill:#e77455;stroke:#fff;stroke-width:2}.tick,.legend{fill:#9eb0b8;font-size:8px}.iid-text{fill:#e5c979}.adjusted-text{fill:#bba6ee}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebdc5;font-size:11px}.control output{color:#f0bd59;font:750 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#d8794f}button{min-height:44px;padding:9px 12px;border:1px solid #526772;border-radius:6px;color:#dce6e9;background:#172733;font-weight:750;cursor:pointer}button.active{border-color:#a98de5;color:#efe8ff;background:#30294a}button:focus-visible,input:focus-visible{outline:3px solid #6bc4d8;outline-offset:2px}@media(max-width:560px){.scanner-scroll{margin-right:-18px;padding-right:18px}.scanner-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
