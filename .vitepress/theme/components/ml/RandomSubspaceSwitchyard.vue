<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const featureCount = 12
const featureNames = ['端点斜率', '斜率副本', '反射功率', '压力波动', '壁温', '清洗龄', 'MFC 偏差', '偏置功率', 'OES 峰', '腔体号', '配方族', '环境温度']
const featureColors = ['#3bc7d3', '#63b5bd', '#8b7cf6', '#e7b857', '#e36b5d', '#9cc36a', '#a88bd8', '#c38b68', '#6cc7a8', '#7f91a1', '#b2a46e', '#7189b7']
const mtry = ref(3)
const treeCount = ref(160)
const duplicateSignal = ref(true)

const nodeCandidates = computed(() => Array.from({ length: 9 }, (_, node) => {
  const chosen: number[] = []
  let cursor = (node * 5 + 2) % featureCount
  while (chosen.length < mtry.value) {
    if (!chosen.includes(cursor)) chosen.push(cursor)
    cursor = (cursor + 7) % featureCount
  }
  const strengths = [1, duplicateSignal.value ? .94 : .16, .76, .61, .55, .48, .45, .38, .35, .18, .25, .12]
  const winner = chosen.reduce((best, index) => strengths[index] > strengths[best] ? index : best, chosen[0])
  return { node, chosen, winner }
}))

const inclusionProbability = computed(() => mtry.value / featureCount)
const winners = computed(() => nodeCandidates.value.map(item => item.winner))
const winnerDiversity = computed(() => new Set(winners.value).size)
const dominantShare = computed(() => winners.value.filter(index => index === 0 || (duplicateSignal.value && index === 1)).length / winners.value.length)
const correlationProxy = computed(() => Math.min(.94, .08 + .73 * Math.pow(mtry.value / featureCount, 1.35) + (duplicateSignal.value ? .08 : 0)))
const strengthProxy = computed(() => Math.min(.93, .48 + .36 * (1 - Math.pow(1 - mtry.value / featureCount, duplicateSignal.value ? 2 : 1)) + .08 * mtry.value / featureCount))
const forestVarianceProxy = computed(() => correlationProxy.value + (1 - correlationProxy.value) / treeCount.value)
</script>

<template>
  <MlExperimentFrame
    index="20-C"
    title="把强特征从每个节点的直通线撤下来"
    question="若端点斜率总能进入候选集，所有树会在相近位置切开。缩小 mtry 会让哪些节点改走别的传感器？复制一列强特征为何会悄悄恢复相关性？"
    formula="P(feature j is offered at a node)=mtry/d"
  >
    <template #visual>
      <div class="switchyard-scroll">
        <svg viewBox="0 0 640 350" role="img" :aria-label="`${featureCount}个特征中每节点抽${mtry}个，九个节点选出${winnerDiversity}种获胜特征`">
          <rect width="640" height="350" rx="14" class="plate" />
          <text x="24" y="24" class="eyebrow">NODE-LEVEL FEATURE SWITCHYARD · 9 个独立切分节点</text>
          <g v-for="(name,index) in featureNames" :key="name">
            <rect x="24" :y="42+index*24" width="113" height="17" rx="3" class="feature-card" />
            <circle cx="34" :cy="50.5+index*24" r="4" :style="{fill:featureColors[index]}" />
            <text x="43" :y="53.5+index*24" class="feature-name">{{ name }}</text>
          </g>
          <g v-for="(item,nodeIndex) in nodeCandidates" :key="item.node">
            <text :x="174+nodeIndex*49" y="50" text-anchor="middle" class="node-id">N{{ nodeIndex+1 }}</text>
            <line :x1="174+nodeIndex*49" y1="61" :x2="174+nodeIndex*49" y2="304" class="node-rail" />
            <g v-for="featureIndex in item.chosen" :key="featureIndex">
              <line x1="137" :y1="50.5+featureIndex*24" :x2="174+nodeIndex*49" :y2="50.5+featureIndex*24" :class="featureIndex===item.winner ? 'route winner-route' : 'route'" />
              <circle :cx="174+nodeIndex*49" :cy="50.5+featureIndex*24" :r="featureIndex===item.winner ? 6 : 3.5" :style="{fill:featureColors[featureIndex]}" :class="featureIndex===item.winner ? 'switch winner' : 'switch'" />
            </g>
          </g>
          <rect x="154" y="315" width="458" height="18" rx="5" class="collector" />
          <text x="383" y="328" text-anchor="middle" class="collector-label">每个节点只在自己的候选支路中选最大 gain</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>每节点候选数 mtry <output>{{ mtry }} / {{ featureCount }}</output></span><input v-model.number="mtry" aria-label="每节点候选特征数" type="range" min="1" :max="featureCount" step="1"></label>
      <label class="control"><span>树数 B <output>{{ treeCount }}</output></span><input v-model.number="treeCount" aria-label="树数" type="range" min="20" max="400" step="20"></label>
      <div class="control"><span>强相关副本</span><button type="button" :aria-pressed="duplicateSignal" @click="duplicateSignal=!duplicateSignal">{{ duplicateSignal ? '已接入：斜率副本可替代' : '已断开：副本只含弱信号' }}</button></div>
      <p class="hint">右侧“相关、强度、方差”是帮助理解方向的教学代理量，不是训练结果。真实 `mtry` 只能通过分组 OOB/验证误差、树预测相关性和成本共同选择。</p>
    </template>
    <template #metrics>
      <div><span>单特征入围概率</span><strong>{{ (100 * inclusionProbability).toFixed(1) }}%</strong></div>
      <div><span>9 节点获胜特征种类</span><strong>{{ winnerDiversity }} / {{ featureCount }}</strong></div>
      <div><span>强特征族获胜占比</span><strong>{{ (100 * dominantShare).toFixed(1) }}%</strong></div>
      <div><span>教学代理：相关 / 强度 / 方差</span><strong>{{ correlationProxy.toFixed(2) }} / {{ strengthProxy.toFixed(2) }} / {{ forestVarianceProxy.toFixed(2) }}</strong></div>
    </template>
    <template #note>随机子空间法通常为每个学习器固定一组特征；随机森林则在每一个节点重新抽候选集。相关副本会提高“强特征族至少有一列入围”的概率，也会让单列置换重要性相互补偿，因此特征去相关和解释审计必须一起做。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.switchyard-scroll{overflow-x:auto}.switchyard-scroll svg{display:block;min-width:610px;width:100%;height:auto}.plate{fill:#0f1822}.eyebrow,.feature-name,.node-id,.collector-label{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#6ed7df;font-size:9px;letter-spacing:.1em}.feature-card{fill:#162630;stroke:#415968}.feature-name{fill:#c9d5dc;font-size:8px}.node-id{fill:#b9adff;font-size:8px;font-weight:800}.node-rail{stroke:#334958;stroke-width:2}.route{stroke:#607481;stroke-width:1;opacity:.55}.winner-route{stroke:#e7b857;stroke-width:2;opacity:1}.switch{stroke:#cbd8df;stroke-width:1}.switch.winner{stroke:#fff0bd;stroke-width:2}.collector{fill:#27213e;stroke:#8b7cf6}.collector-label{fill:#d9d3ff;font-size:8px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebbdc;font-size:11px}.control output{color:#f0c76b;font:750 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#8b7cf6}button{min-height:44px;border:1px solid #4a6170;border-radius:6px;color:#dbe5e9;background:#14232c;font:700 10px/1.3 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#e7b857;color:#ffe4a2;background:#30291e}button:focus-visible,input:focus-visible{outline:3px solid #65d8e1;outline-offset:2px}.hint{margin:0;padding:11px;border-left:3px solid #e36b5d;color:#a8b9c5;background:#101b25;font-size:10px;line-height:1.7}@media(max-width:560px){.switchyard-scroll{margin-right:-18px;padding-right:18px}.switchyard-scroll svg{width:610px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
