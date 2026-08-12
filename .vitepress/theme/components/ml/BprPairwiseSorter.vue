<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Item = { id: string; topic: string; popularity: number; q: number[]; bias: number }

const templates: Item[] = [
  { id: 'C19', topic: '已收藏·热像', popularity: 18, q: [0.72, 0.54], bias: 0.08 },
  { id: 'C07', topic: '润滑', popularity: 31, q: [0.60, 0.18], bias: 0.23 },
  { id: 'C12', topic: '振动', popularity: 46, q: [0.31, 0.71], bias: 0.31 },
  { id: 'C26', topic: '电气', popularity: 11, q: [-0.24, 0.68], bias: -0.02 },
  { id: 'C33', topic: '密封', popularity: 38, q: [0.49, -0.21], bias: 0.27 },
  { id: 'C41', topic: '对中', popularity: 7, q: [-0.37, 0.22], bias: -0.08 },
]

const user = ref([0.24, 0.42])
const items = ref(templates.map(item => ({ ...item, q: [...item.q] })))
const negativeIndex = ref(2)
const sampling = ref<'uniform' | 'popular' | 'hard'>('uniform')
const learningRate = ref(0.08)
const regularization = ref(0.01)
const updates = ref(0)
const cursor = ref(0)

function sigmoid(value: number) { return 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, value)))) }
function itemScore(index: number) {
  const item = items.value[index]
  return item.bias + user.value[0] * item.q[0] + user.value[1] * item.q[1]
}

const scores = computed(() => items.value.map((item, index) => ({ ...item, index, score: itemScore(index) })))
const ranking = computed(() => [...scores.value].sort((a, b) => b.score - a.score))
const gap = computed(() => itemScore(0) - itemScore(negativeIndex.value))
const pairProbability = computed(() => sigmoid(gap.value))
const pairLoss = computed(() => -Math.log(pairProbability.value) + 0.5 * regularization.value * (
  user.value.reduce((sum, value) => sum + value ** 2, 0)
  + items.value[0].q.reduce((sum, value) => sum + value ** 2, 0)
  + items.value[negativeIndex.value].q.reduce((sum, value) => sum + value ** 2, 0)
))

function chooseNegative(index: number) {
  if (index === 0) return
  negativeIndex.value = index
}

function sampleNegative() {
  const candidates = scores.value.filter(item => item.index !== 0)
  if (sampling.value === 'hard') {
    negativeIndex.value = [...candidates].sort((a, b) => b.score - a.score)[0].index
    return
  }
  if (sampling.value === 'popular') {
    negativeIndex.value = [...candidates].sort((a, b) => b.popularity - a.popularity)[cursor.value % 2].index
    cursor.value += 1
    return
  }
  const sequence = [1, 3, 5, 2, 4]
  negativeIndex.value = sequence[cursor.value % sequence.length]
  cursor.value += 1
}

function updatePair() {
  const positive = items.value[0]
  const negative = items.value[negativeIndex.value]
  const multiplier = sigmoid(-gap.value)
  const oldUser = [...user.value]
  const oldPositive = [...positive.q]
  const oldNegative = [...negative.q]
  for (let dimension = 0; dimension < 2; dimension += 1) {
    user.value[dimension] += learningRate.value * (
      multiplier * (oldPositive[dimension] - oldNegative[dimension])
      - regularization.value * oldUser[dimension]
    )
    positive.q[dimension] += learningRate.value * (
      multiplier * oldUser[dimension] - regularization.value * oldPositive[dimension]
    )
    negative.q[dimension] += learningRate.value * (
      -multiplier * oldUser[dimension] - regularization.value * oldNegative[dimension]
    )
  }
  positive.bias += learningRate.value * (multiplier - regularization.value * positive.bias)
  negative.bias += learningRate.value * (-multiplier - regularization.value * negative.bias)
  user.value = [...user.value]
  items.value = items.value.map(item => ({ ...item, q: [...item.q] }))
  updates.value += 1
}

function reset() {
  user.value = [0.24, 0.42]
  items.value = templates.map(item => ({ ...item, q: [...item.q] }))
  negativeIndex.value = 2
  sampling.value = 'uniform'
  learningRate.value = 0.08
  regularization.value = 0.01
  updates.value = 0
  cursor.value = 0
}

function barWidth(score: number) { return Math.max(4, Math.min(214, (score + 0.35) / 1.35 * 214)) }
</script>

<template>
  <MlExperimentFrame
    index="29-C"
    title="BPR 成对分拣器"
    question="一次更新只知道已收藏卡 i 应排在未观察卡 j 前面；负例抽样方式如何改变梯度看到的世界？"
    formula="xuij = sui − suj; ∇ log σ(xuij) = σ(−xuij)∇xuij"
  >
    <template #visual>
      <div class="sorter-scroll">
        <svg viewBox="0 0 650 420" role="img" aria-label="BPR 对正知识卡与采样未观察知识卡进行成对排序更新">
          <rect class="housing" x="14" y="14" width="622" height="392" rx="3" />
          <text class="caption" x="40" y="45">PAIRWISE SORTER · E-17 · POSITIVE &gt; UNOBSERVED</text>
          <rect class="pair-card positive-card" x="40" y="66" width="224" height="92" rx="3" />
          <text class="pair-kind" x="56" y="88">POSITIVE i · OBSERVED</text>
          <text class="pair-id" x="56" y="123">{{ items[0].id }}</text>
          <text class="pair-topic" x="124" y="123">{{ items[0].topic }}</text>
          <text class="pair-score" x="246" y="143" text-anchor="end">sui={{ itemScore(0).toFixed(3) }}</text>
          <rect class="pair-card negative-card" x="366" y="66" width="244" height="92" rx="3" />
          <text class="pair-kind" x="382" y="88">SAMPLED j · UNOBSERVED</text>
          <text class="pair-id" x="382" y="123">{{ items[negativeIndex].id }}</text>
          <text class="pair-topic" x="450" y="123">{{ items[negativeIndex].topic }}</text>
          <text class="pair-score" x="592" y="143" text-anchor="end">suj={{ itemScore(negativeIndex).toFixed(3) }}</text>
          <path class="conveyor" d="M 275 111 H 352" />
          <path class="arrow" d="M 345 103 L 354 111 L 345 119" />
          <text class="gap-label" x="314" y="96" text-anchor="middle">Δ={{ gap.toFixed(2) }}</text>

          <rect class="ranking-card" x="40" y="181" width="570" height="184" rx="3" />
          <text class="rank-title" x="56" y="204">CURRENT FULL-CATALOG RANKING</text>
          <g
            v-for="(item, rank) in ranking"
            :key="item.id"
            class="rank-row"
            role="button"
            tabindex="0"
            :aria-label="item.index === 0 ? `${item.id} 是固定正例` : `选择 ${item.id} 作为负例`"
            :transform="`translate(56 ${229 + rank * 22})`"
            @click="chooseNegative(item.index)"
            @keydown.enter.prevent="chooseNegative(item.index)"
            @keydown.space.prevent="chooseNegative(item.index)"
          >
            <text class="rank-number" x="0" y="0">{{ rank + 1 }}</text>
            <text class="rank-id" :class="{ positive: item.index === 0, selected: item.index === negativeIndex }" x="24" y="0">{{ item.id }}</text>
            <rect class="bar-track" x="78" y="-10" width="214" height="12" rx="2" />
            <rect class="score-bar" :class="{ positive: item.index === 0, selected: item.index === negativeIndex }" x="78" y="-10" :width="barWidth(item.score)" height="12" rx="2" />
            <text class="rank-score" x="307" y="0">{{ item.score.toFixed(3) }}</text>
            <text class="rank-pop" x="382" y="0">曝光 {{ item.popularity }}</text>
          </g>
          <line class="perforation" x1="34" y1="385" x2="616" y2="385" />
        </svg>
      </div>
    </template>

    <template #controls>
      <div class="modes" aria-label="负例采样方式">
        <button type="button" :aria-pressed="sampling === 'uniform'" @click="sampling = 'uniform'">均匀</button>
        <button type="button" :aria-pressed="sampling === 'popular'" @click="sampling = 'popular'">热门加权</button>
        <button type="button" :aria-pressed="sampling === 'hard'" @click="sampling = 'hard'">当前困难项</button>
      </div>
      <label class="control">
        <span>学习率 η <output>{{ learningRate.toFixed(2) }}</output></span>
        <input v-model.number="learningRate" type="range" min="0.01" max="0.20" step="0.01" aria-label="BPR 更新学习率">
      </label>
      <label class="control">
        <span>正则 λ <output>{{ regularization.toFixed(3) }}</output></span>
        <input v-model.number="regularization" type="range" min="0" max="0.08" step="0.005" aria-label="BPR 更新正则强度">
      </label>
      <div class="actions">
        <button type="button" @click="sampleNegative">抽样负例 j</button>
        <button type="button" class="update" @click="updatePair">执行一次 BPR 更新</button>
      </div>
      <button type="button" class="reset" @click="reset">复位分拣器</button>
    </template>

    <template #metrics>
      <div><span>成对间隔 sui−suj</span><strong>{{ gap.toFixed(3) }}</strong></div>
      <div><span>P(i 排在 j 前)</span><strong>{{ (pairProbability * 100).toFixed(1) }}%</strong></div>
      <div><span>单对正则损失</span><strong>{{ pairLoss.toFixed(3) }}</strong></div>
      <div><span>累计更新</span><strong>{{ updates }}</strong></div>
    </template>

    <template #note>间隔小或次序错误时，σ(−Δ) 大，更新更强；间隔已足够大时梯度自然衰减。热门加权和困难负例能改变训练重点，却也更容易把“未曝光但合适”的卡当成负例；负采样策略本身就是模型合同的一部分。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.sorter-scroll{overflow-x:auto}.sorter-scroll svg{display:block;width:100%;height:auto}.housing{fill:#10222c;stroke:#43b6ad;stroke-width:2}.caption,.pair-kind,.pair-id,.pair-topic,.pair-score,.gap-label,.rank-title,.rank-number,.rank-id,.rank-score,.rank-pop{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.caption{fill:#efe8d4;font-size:8px;font-weight:900;letter-spacing:.09em}.pair-card{stroke-width:2}.positive-card{fill:#efe8d4;stroke:#e7762f}.negative-card{fill:#27343b;stroke:#56616a}.pair-kind{fill:#56616a;font-size:8px;font-weight:800}.negative-card+.pair-kind{fill:#aeb9ba}.pair-id{fill:#17364a;font-size:27px;font-weight:900}.negative-card~.pair-id{fill:#efe8d4}.pair-topic{fill:#56616a;font-size:9px}.negative-card~.pair-topic{fill:#b9c2c0}.pair-score{fill:#c44d43;font-size:9px;font-weight:850}.conveyor,.arrow{fill:none;stroke:#e7762f;stroke-width:3}.gap-label{fill:#efe8d4;font-size:8px}.ranking-card{fill:#17364a;stroke:#53616a}.rank-title{fill:#43b6ad;font-size:8px;font-weight:900;letter-spacing:.09em}.rank-row{cursor:pointer}.rank-row:focus-visible{outline:3px solid #e7762f;outline-offset:1px}.rank-number{fill:#9eb1b5;font-size:8px}.rank-id{fill:#e5eeec;font-size:9px;font-weight:850}.rank-id.positive{fill:#ffc18c}.rank-id.selected{fill:#ffaaa2}.bar-track{fill:#26343c}.score-bar{fill:#43b6ad}.score-bar.positive{fill:#e7762f}.score-bar.selected{fill:#c44d43}.rank-score,.rank-pop{fill:#b9c6c8;font-size:8px}.perforation{stroke:#56616a;stroke-dasharray:3 5}.modes{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.actions{display:grid;grid-template-columns:1fr 1fr;gap:6px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#c3cee8;font-size:11px}.control output{color:#ffc28f;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#e7762f}button{min-height:44px;padding:8px;border:1px solid #536a74;border-radius:4px;color:#dbe6e8;background:#14252e;font:720 9px/1.35 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#e7762f;color:#ffe4d2;background:#4b2c20}.update{border-color:#e7762f;color:#ffe4d2}.reset{border-color:#43b6ad;color:#bbfff8}button:focus-visible,input:focus-visible{outline:3px solid #e7762f;outline-offset:2px}@media(max-width:560px){.sorter-scroll{margin-right:-18px;padding-right:18px}.sorter-scroll svg{width:720px}.modes{grid-template-columns:1fr}.actions{grid-template-columns:1fr}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
