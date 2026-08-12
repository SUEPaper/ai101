<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const treeCount = ref(80)
const correlation = ref(0.22)
const treeSigma = ref(1.0)

const varianceRatio = computed(() => correlation.value + (1 - correlation.value) / treeCount.value)
const ensembleSigma = computed(() => treeSigma.value * Math.sqrt(varianceRatio.value))
const effectiveIndependentTrees = computed(() => 1 / varianceRatio.value)
const marginalGain = computed(() => {
  const next = correlation.value + (1 - correlation.value) / (treeCount.value * 2)
  return 1 - next / varianceRatio.value
})
const curve = computed(() => Array.from({ length: 80 }, (_, index) => {
  const b = 1 + index * 499 / 79
  const ratio = correlation.value + (1 - correlation.value) / b
  return `${48 + index * 5.25},${265 - ratio * 205}`
}).join(' '))
const currentX = computed(() => 48 + ((treeCount.value - 1) / 499) * 415)
const currentY = computed(() => 265 - varianceRatio.value * 205)
const channels = computed(() => Array.from({ length: 14 }, (_, index) => ({
  x: 487 + (index % 2) * 48,
  y: 54 + Math.floor(index / 2) * 34,
  phase: ((index * 37) % 11) / 10,
})))
</script>

<template>
  <MlExperimentFrame
    index="20-B"
    title="在误差母线上找到平均的极限"
    question="树数翻倍会继续稳定预测，但如果所有树都沿同一强特征犯错，为什么方差不会降到零？应该优先增加树，还是先让树彼此不同？"
    formula="Var(f̄)/σ² = ρ + (1−ρ)/B"
  >
    <template #visual>
      <div class="rack-scroll">
        <svg viewBox="0 0 610 330" role="img" :aria-label="`${treeCount}棵树，相关系数${correlation.toFixed(2)}，集成方差比例${varianceRatio.toFixed(3)}`">
          <rect width="610" height="330" rx="14" class="plate" />
          <text x="24" y="24" class="eyebrow">FOREST ERROR BUS · VARIANCE TEST RACK</text>
          <rect x="28" y="42" width="447" height="246" rx="8" class="plot" />
          <line x1="48" y1="265" x2="463" y2="265" class="axis" /><line x1="48" y1="60" x2="48" y2="265" class="axis" />
          <line x1="48" :y1="265-correlation*205" x2="463" :y2="265-correlation*205" class="floor-line" />
          <text x="55" :y="258-correlation*205" class="floor-label">共同误差地板 ρ={{ correlation.toFixed(2) }}</text>
          <polyline :points="curve" class="variance-curve" />
          <line :x1="currentX" y1="265" :x2="currentX" :y2="currentY" class="cursor" />
          <circle :cx="currentX" :cy="currentY" r="7" class="current-point" />
          <text x="52" y="78" class="axis-label">方差比 1.0</text><text x="52" y="281" class="axis-label">B=1</text><text x="421" y="281" class="axis-label">B=500</text>
          <path d="M475 42 H584 V288 H475" class="bus-panel" />
          <text x="493" y="66" class="bus-title">TREE BUS</text>
          <g v-for="(channel,index) in channels" :key="index">
            <rect :x="channel.x" :y="channel.y" width="36" height="20" rx="4" class="channel" />
            <line :x1="channel.x+5" :y1="channel.y+10" :x2="channel.x+31" :y2="channel.y+10" class="channel-base" />
            <path :d="`M${channel.x+5} ${channel.y+10} q7 ${-7*(1-correlation)*(channel.phase-.5)} 13 0 t13 0`" class="private-signal" />
          </g>
          <line x1="492" y1="292" x2="560" y2="292" class="common-bus" />
          <text x="526" y="309" text-anchor="middle" class="bus-note">共享分量不被平均</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>树数 B <output>{{ treeCount }}</output></span><input v-model.number="treeCount" aria-label="树数" type="range" min="1" max="500" step="1"></label>
      <label class="control"><span>树间相关 ρ <output>{{ correlation.toFixed(2) }}</output></span><input v-model.number="correlation" aria-label="树间误差相关系数" type="range" min="0" max="1" step="0.01"></label>
      <label class="control"><span>单树误差标准差 σ <output>{{ treeSigma.toFixed(1) }}</output></span><input v-model.number="treeSigma" aria-label="单树误差标准差" type="range" min="0.4" max="2" step="0.1"></label>
      <p class="hint">公式假设每棵树方差相同、两两相关相同，是诊断模型，不是随机森林误差的完整定理。真实系统应从同一 OOB 样本上的树预测估计相关性。</p>
    </template>
    <template #metrics>
      <div><span>集成方差 / 单树方差</span><strong>{{ varianceRatio.toFixed(3) }}</strong></div>
      <div><span>集成误差标准差</span><strong>{{ ensembleSigma.toFixed(3) }}</strong></div>
      <div><span>等效独立树数</span><strong>{{ effectiveIndependentTrees.toFixed(1) }}</strong></div>
      <div><span>B 再翻倍的方差收益</span><strong>{{ (100 * marginalGain).toFixed(2) }}%</strong></div>
    </template>
    <template #note>当 ρ=0，方差按 1/B 下降；当 ρ=1，所有树像复制件，平均毫无帮助。树数只消除有限森林的蒙特卡洛波动，Bootstrap、节点级特征抽样和结构约束负责改变相关性与单树强度。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.rack-scroll{overflow-x:auto}.rack-scroll svg{display:block;min-width:580px;width:100%;height:auto}.plate{fill:#0f1822}.plot{fill:#13222e;stroke:#405667}.axis{stroke:#718595}.floor-line{stroke:#e36b5d;stroke-dasharray:7 5}.variance-curve{fill:none;stroke:#3bc7d3;stroke-width:3}.cursor{stroke:#e7b857;stroke-dasharray:3 3}.current-point{fill:#e7b857;stroke:#fff1bd;stroke-width:2}.eyebrow,.floor-label,.axis-label,.bus-title,.bus-note{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#71dbe3;font-size:9px;letter-spacing:.1em}.floor-label{fill:#f18a7d;font-size:8px}.axis-label{fill:#8fa2b0;font-size:8px}.bus-panel{fill:#121e29;stroke:#3f5566}.bus-title{fill:#b9adff;font-size:9px;letter-spacing:.12em}.channel{fill:#1d2d3a;stroke:#516877}.channel-base{stroke:#586b78}.private-signal{fill:none;stroke:#8b7cf6;stroke-width:2}.common-bus{stroke:#e36b5d;stroke-width:4}.bus-note{fill:#a9b7c2;font-size:7px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebbdc;font-size:11px}.control output{color:#f0c76b;font:750 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#3bc7d3}.hint{margin:0;padding:11px;border-left:3px solid #8b7cf6;color:#a8b9c5;background:#101b25;font-size:10px;line-height:1.7}input:focus-visible{outline:3px solid #65d8e1;outline-offset:2px}@media(max-width:560px){.rack-scroll{margin-right:-18px;padding-right:18px}.rack-scroll svg{width:580px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
