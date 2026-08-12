<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const splitIndex = ref(5)
const l2 = ref(1)
const gamma = ref(.2)
const saturated = ref(false)
const feature = [.08, .16, .25, .34, .43, .54, .63, .72, .84, .93]
const labels = [0, 0, 1, 0, 0, 1, 0, 1, 1, 1]
const sigmoid = (value: number) => value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value))

const ledger = computed(() => {
  const logits = feature.map(value => saturated.value ? (value < .58 ? -5.2 : 5.2) : -.35 + .7 * value)
  const rows = feature.map((value, index) => {
    const probability = sigmoid(logits[index])
    return { value, label: labels[index], probability, gradient: probability - labels[index], hessian: probability * (1 - probability) }
  })
  const left = rows.slice(0, splitIndex.value)
  const right = rows.slice(splitIndex.value)
  const sum = (part: typeof rows, key: 'gradient' | 'hessian') => part.reduce((total, row) => total + row[key], 0)
  const GL = sum(left, 'gradient'), GR = sum(right, 'gradient')
  const HL = sum(left, 'hessian'), HR = sum(right, 'hessian')
  const parent = (GL + GR) ** 2 / (HL + HR + l2.value)
  const gain = .5 * (GL ** 2 / (HL + l2.value) + GR ** 2 / (HR + l2.value) - parent) - gamma.value
  return {
    rows, GL, GR, HL, HR, gain,
    leftWeight: -GL / (HL + l2.value), rightWeight: -GR / (HR + l2.value),
    threshold: (feature[splitIndex.value - 1] + feature[splitIndex.value]) / 2,
  }
})

const mapX = (value: number) => 52 + value * 515
const leftHWidth = computed(() => 180 * ledger.value.HL / Math.max(.001, ledger.value.HL + ledger.value.HR))
</script>

<template>
  <MlExperimentFrame
    index="21-C"
    title="在梯度—曲率称量站决定是否开一道分裂"
    question="同样的梯度总量，在高曲率与低曲率叶里应走多大一步？λ、γ 和饱和概率为什么会改变分裂许可？"
    formula="gain=½[G²L/(HL+λ)+G²R/(HR+λ)−G²/(H+λ)]−γ"
  >
    <template #visual>
      <div class="foundry-scroll">
        <svg viewBox="0 0 640 350" role="img" :aria-label="`二阶分裂阈值${ledger.threshold.toFixed(2)}，增益${ledger.gain.toFixed(3)}，${ledger.gain>0?'允许':'拒绝'}分裂`">
          <rect width="640" height="350" rx="14" class="panel" />
          <text x="24" y="27" class="eyebrow">GRADIENT / HESSIAN WEIGHBRIDGE</text>
          <rect x="32" y="48" width="576" height="150" rx="9" class="track" />
          <line x1="52" y1="145" x2="567" y2="145" class="rail" />
          <line :x1="mapX(ledger.threshold)" y1="58" :x2="mapX(ledger.threshold)" y2="190" class="gate" />
          <g v-for="(row,index) in ledger.rows" :key="index">
            <circle :cx="mapX(row.value)" cy="145" :r="7+row.hessian*34" class="token" :class="row.label ? 'positive':'negative'" />
            <line :x1="mapX(row.value)" y1="137" :x2="mapX(row.value)" :y2="137-row.gradient*42" class="gradient" />
            <text :x="mapX(row.value)" y="173" text-anchor="middle" class="token-id">{{ index+1 }}</text>
          </g>
          <rect x="52" y="228" width="180" height="34" rx="5" class="hopper" />
          <rect x="52" y="228" :width="leftHWidth" height="34" rx="5" class="hessian-fill" />
          <text x="142" y="250" text-anchor="middle" class="hopper-label">LEFT H={{ ledger.HL.toFixed(2) }}</text>
          <rect x="258" y="228" width="180" height="34" rx="5" class="hopper" />
          <rect x="258" y="228" :width="180-leftHWidth" height="34" rx="5" class="hessian-fill right" />
          <text x="348" y="250" text-anchor="middle" class="hopper-label">RIGHT H={{ ledger.HR.toFixed(2) }}</text>
          <path d="M456 218 h132 v58 H456 z" class="permit" :class="{ accepted:ledger.gain>0 }" />
          <text x="522" y="242" text-anchor="middle" class="permit-label">{{ ledger.gain>0 ? 'SPLIT PERMIT' : 'HOLD NODE' }}</text>
          <text x="522" y="260" text-anchor="middle" class="permit-value">{{ ledger.gain.toFixed(3) }}</text>
          <text x="52" y="310" class="caption">圆片面积 = Hessian　竖针 = gradient 方向　料斗 = 子节点曲率预算</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>候选切点 <output>{{ splitIndex }} / 9</output></span><input v-model.number="splitIndex" aria-label="二阶树候选切点" type="range" min="2" max="8" step="1"></label>
      <label class="control"><span>叶权重 L2：λ <output>{{ l2.toFixed(1) }}</output></span><input v-model.number="l2" aria-label="二阶树叶权重L2" type="range" min="0" max="8" step="0.2"></label>
      <label class="control"><span>开叶成本：γ <output>{{ gamma.toFixed(1) }}</output></span><input v-model.number="gamma" aria-label="二阶树开叶成本" type="range" min="0" max="3" step="0.1"></label>
      <button type="button" :class="{ active:saturated }" :aria-pressed="saturated" @click="saturated=!saturated">{{ saturated ? '已切换：概率接近 0/1' : '切换到饱和 logits' }}</button>
    </template>
    <template #metrics>
      <div><span>左 / 右叶权重</span><strong>{{ ledger.leftWeight.toFixed(2) }} / {{ ledger.rightWeight.toFixed(2) }}</strong></div>
      <div><span>左 / 右 Hessian</span><strong>{{ ledger.HL.toFixed(2) }} / {{ ledger.HR.toFixed(2) }}</strong></div>
      <div><span>正则化切分增益</span><strong>{{ ledger.gain.toFixed(3) }}</strong></div>
      <div><span>决策</span><strong>{{ ledger.gain>0 ? '允许分裂' : '保留父叶' }}</strong></div>
    </template>
    <template #note>Hessian 小表示当前局部曲率弱，裸 Newton 步可能很大；λ 给分母加制动，γ 要求新叶先偿还结构成本。真实实现还要门禁最小子叶 Hessian，不能把饱和高置信误差当成“无需样本”。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.foundry-scroll{overflow-x:auto}.foundry-scroll svg{display:block;min-width:600px;width:100%;height:auto}.panel{fill:#111923}.track{fill:#182633;stroke:#4e6370}.rail{stroke:#83949d;stroke-width:2}.eyebrow,.caption,.token-id,.hopper-label,.permit-label,.permit-value{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#72d4c7;font-size:10px;letter-spacing:.12em}.gate{stroke:#f0bd54;stroke-width:3;stroke-dasharray:6 4}.token{stroke-width:2}.positive{fill:#b9613f;stroke:#f3a07a}.negative{fill:#277b83;stroke:#69cfca}.gradient{stroke:#efd36f;stroke-width:2}.token-id{fill:#aebdc5;font-size:8px}.hopper{fill:#233440;stroke:#617582}.hessian-fill{fill:#3eb8ae}.hessian-fill.right{fill:#c96e49}.hopper-label{fill:#e5edf0;font-size:9px}.permit{fill:#3a2827;stroke:#aa5f50}.permit.accepted{fill:#153b38;stroke:#56c8bc}.permit-label{fill:#f1e1c5;font-size:9px;font-weight:800}.permit-value{fill:#f0c56a;font-size:12px}.caption{fill:#9fb0ba;font-size:8px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebbdc;font-size:11px}.control output{color:#f0c76b;font:750 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#c56b46}button{min-height:44px;padding:9px 12px;border:1px solid #52657a;border-radius:6px;color:#d8e3e8;background:#172634;font-weight:750;cursor:pointer}button.active{border-color:#e5a24f;color:#ffe6b5;background:#3f3020}button:focus-visible,input:focus-visible{outline:3px solid #6fd5ca;outline-offset:2px}@media(max-width:560px){.foundry-scroll{margin-right:-18px;padding-right:18px}.foundry-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
