<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const sampleSize = ref(1000)
const logHypotheses = ref(3)
const delta = ref(0.05)
const empiricalRisk = ref(0.12)

const hypotheses = computed(() => 10 ** logHypotheses.value)
const fixedGap = computed(() => Math.sqrt(Math.log(2 / delta.value) / (2 * sampleSize.value)))
const selectedGap = computed(() => Math.sqrt(Math.log(2 * hypotheses.value / delta.value) / (2 * sampleSize.value)))
const selectedUpper = computed(() => Math.min(1, empiricalRisk.value + selectedGap.value))
const fixedUpper = computed(() => Math.min(1, empiricalRisk.value + fixedGap.value))
const displayMax = computed(() => Math.max(0.35, Math.min(1, selectedUpper.value * 1.15)))

function sx(value: number) {
  return 72 + Math.min(displayMax.value, Math.max(0, value)) / displayMax.value * 484
}

function corridorWidth(gap: number) {
  const lower = Math.max(0, empiricalRisk.value - gap)
  const upper = Math.min(displayMax.value, empiricalRisk.value + gap)
  return Math.max(2, sx(upper) - sx(lower))
}
</script>

<template>
  <MlExperimentFrame
    index="28-A"
    title="PAC 资格证书走廊"
    question="同一份经验误差，在模型预先固定与从 M 个候选中挑选之后，允许写进资格证书的误差走廊相差多少？"
    formula="εfixed=√[ln(2/δ)/(2n)] · εselect=√[ln(2M/δ)/(2n)]"
  >
    <template #visual>
      <div class="certificate-scroll">
        <svg viewBox="0 0 630 390" role="img" aria-label="固定假设与选择后假设的 Hoeffding 置信走廊">
          <rect class="certificate" x="14" y="14" width="602" height="362" rx="3" />
          <path class="perforation" d="M 34 61 H 596 M 34 336 H 596" />
          <text class="serial" x="38" y="43">MODEL QUALIFICATION CERTIFICATE · PAC-28</text>
          <text class="confidence" x="590" y="43" text-anchor="end">CONFIDENCE {{ ((1 - delta) * 100).toFixed(0) }}%</text>

          <g class="axis">
            <line x1="72" y1="294" x2="556" y2="294" />
            <g v-for="tick in [0, .2, .4, .6, .8, 1]" :key="tick" v-show="tick <= displayMax + .01">
              <line :x1="sx(tick)" y1="289" :x2="sx(tick)" y2="300" />
              <text :x="sx(tick)" y="318" text-anchor="middle">{{ tick.toFixed(1) }}</text>
            </g>
          </g>

          <text class="row-label" x="42" y="112">预先固定 h</text>
          <rect class="corridor fixed" :x="sx(Math.max(0, empiricalRisk - fixedGap))" y="125" :width="corridorWidth(fixedGap)" height="34" rx="17" />
          <line class="empirical" :x1="sx(empiricalRisk)" y1="116" :x2="sx(empiricalRisk)" y2="168" />
          <circle class="upper fixed-dot" :cx="sx(fixedUpper)" cy="142" r="7" />
          <text class="limit" :x="sx(fixedUpper)" y="181" text-anchor="middle">上限 {{ fixedUpper.toFixed(3) }}</text>

          <text class="row-label" x="42" y="217">从 M 个候选中选择</text>
          <rect class="corridor selected" :x="sx(Math.max(0, empiricalRisk - selectedGap))" y="230" :width="corridorWidth(selectedGap)" height="34" rx="17" />
          <line class="empirical" :x1="sx(empiricalRisk)" y1="221" :x2="sx(empiricalRisk)" y2="273" />
          <circle class="upper selected-dot" :cx="sx(selectedUpper)" cy="247" r="7" />
          <text class="limit" :x="sx(selectedUpper)" y="286" text-anchor="middle">上限 {{ selectedUpper.toFixed(3) }}</text>

          <g class="stamp" :class="selectedUpper < 1 ? 'usable' : 'vacuous'" transform="translate(497 73) rotate(-7)">
            <rect x="0" y="0" width="92" height="42" rx="2" />
            <text x="46" y="18" text-anchor="middle">{{ selectedUpper < 1 ? 'NON-VACUOUS' : 'VACUOUS' }}</text>
            <text x="46" y="32" text-anchor="middle">BOUND</text>
          </g>
          <text class="foot" x="38" y="358">概率针对重复抽取训练集；不承诺当前部署分布没有漂移</text>
        </svg>
      </div>
    </template>

    <template #controls>
      <label class="control">
        <span>独立样本量 n <output>{{ sampleSize }}</output></span>
        <input v-model.number="sampleSize" type="range" min="100" max="10000" step="100" aria-label="独立样本量">
      </label>
      <label class="control">
        <span>候选数量 M <output>10^{{ logHypotheses }}</output></span>
        <input v-model.number="logHypotheses" type="range" min="0" max="8" step="1" aria-label="候选假设数量的十进制指数">
      </label>
      <label class="control">
        <span>失败概率 δ <output>{{ delta.toFixed(2) }}</output></span>
        <input v-model.number="delta" type="range" min="0.01" max="0.20" step="0.01" aria-label="失败概率">
      </label>
      <label class="control">
        <span>经验风险 R̂ <output>{{ empiricalRisk.toFixed(2) }}</output></span>
        <input v-model.number="empiricalRisk" type="range" min="0.02" max="0.30" step="0.01" aria-label="经验风险">
      </label>
    </template>

    <template #metrics>
      <div><span>固定假设半宽</span><strong>{{ fixedGap.toFixed(3) }}</strong></div>
      <div><span>选择后统一半宽</span><strong>{{ selectedGap.toFixed(3) }}</strong></div>
      <div><span>选择后风险上限</span><strong>{{ selectedUpper.toFixed(3) }}</strong></div>
      <div><span>选择代价</span><strong>+{{ (selectedGap - fixedGap).toFixed(3) }}</strong></div>
    </template>

    <template #note>两条走廊都使用同一 Hoeffding 常数。第一条只对看数据前固定的 h 有效；第二条先对 M 个候选同时成立，因此可以覆盖数据依赖的选择结果。把 M 调大不会线性放大半宽，但会通过 ln M 持续收费。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.certificate-scroll{overflow-x:auto}.certificate-scroll svg{display:block;width:100%;height:auto}.certificate{fill:#eee9d8;stroke:#d8b75c;stroke-width:2}.perforation{fill:none;stroke:#9b9177;stroke-width:1;stroke-dasharray:3 7}.serial,.confidence,.row-label,.axis text,.limit,.foot{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.serial{fill:#26333b;font-size:9px;font-weight:850;letter-spacing:.09em}.confidence{fill:#7468b4;font-size:9px;font-weight:850}.row-label{fill:#253844;font-size:10px;font-weight:800}.axis line{stroke:#4f5a5e}.axis text{fill:#586266;font-size:8px}.corridor.fixed{fill:#67c8bd;opacity:.58}.corridor.selected{fill:#7468b4;opacity:.58}.empirical{stroke:#18252c;stroke-width:3}.upper{stroke:#eee9d8;stroke-width:2}.fixed-dot{fill:#278f86}.selected-dot{fill:#5b4da2}.limit{fill:#36464d;font-size:8px;font-weight:750}.stamp rect{fill:none;stroke-width:2}.stamp text{font:850 8px/1 ui-monospace,monospace;letter-spacing:.06em}.stamp.usable rect{stroke:#258c83}.stamp.usable text{fill:#258c83}.stamp.vacuous rect{stroke:#d65a56}.stamp.vacuous text{fill:#b03b38}.foot{fill:#776d59;font-size:8px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#c2cee9;font-size:11px}.control output{color:#f1ce79;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#67c8bd}input:focus-visible{outline:3px solid #d8b75c;outline-offset:2px}@media(max-width:560px){.certificate-scroll{margin-right:-18px;padding-right:18px}.certificate-scroll svg{width:690px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
