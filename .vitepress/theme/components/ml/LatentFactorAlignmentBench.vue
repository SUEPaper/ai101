<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Card = { id: string; topic: string; angle: number; radius: number; bias: number }

const cards: Card[] = [
  { id: 'C07', topic: '润滑', angle: 16, radius: 0.92, bias: 0.08 },
  { id: 'C12', topic: '振动', angle: 52, radius: 1.00, bias: 0.02 },
  { id: 'C19', topic: '热像', angle: 91, radius: 0.82, bias: 0.17 },
  { id: 'C26', topic: '电气', angle: 139, radius: 0.94, bias: -0.03 },
  { id: 'C33', topic: '密封', angle: 205, radius: 0.88, bias: 0.22 },
  { id: 'C41', topic: '对中', angle: 302, radius: 1.04, bias: -0.06 },
]

const userAngle = ref(38)
const commonRotation = ref(0)
const shrinkage = ref(0.12)
const includeBias = ref(true)

function radians(degrees: number) { return degrees * Math.PI / 180 }
function rotate(angle: number, radius: number) {
  const value = radians(angle + commonRotation.value)
  return { x: radius * Math.cos(value), y: radius * Math.sin(value) }
}

const scale = computed(() => 1 - shrinkage.value)
const user = computed(() => rotate(userAngle.value, 1.08 * scale.value))
const projectedCards = computed(() => cards.map(card => ({
  ...card,
  ...rotate(card.angle, card.radius * scale.value),
})))
const ranking = computed(() => projectedCards.value.map(card => ({
  ...card,
  score: user.value.x * card.x + user.value.y * card.y + (includeBias.value ? card.bias : 0),
})).sort((a, b) => b.score - a.score))
const scoreGap = computed(() => ranking.value[0].score - ranking.value[1].score)

function sx(value: number) { return 290 + value * 132 }
function sy(value: number) { return 210 - value * 132 }
function reset() {
  userAngle.value = 38
  commonRotation.value = 0
  shrinkage.value = 0.12
  includeBias.value = true
}
</script>

<template>
  <MlExperimentFrame
    index="29-B"
    title="潜因子对准工作台"
    question="旋转潜空间会改变向量坐标，却为何不改变任何点积分数？正则收缩和物品偏置又分别改了什么？"
    formula="sui = μ + bu + bi + puᵀqi; (PA)(QA)ᵀ = PQᵀ for orthogonal A"
  >
    <template #visual>
      <div class="bench-scroll">
        <svg viewBox="0 0 650 420" role="img" aria-label="用户向量与知识卡向量在二维潜因子空间中的点积和共同旋转">
          <rect class="housing" x="14" y="14" width="622" height="392" rx="3" />
          <rect class="field" x="34" y="42" width="478" height="334" rx="2" />
          <text class="caption" x="52" y="66">LATENT ALIGNMENT BENCH · AXES ARE NOT LABELS</text>
          <g class="grid" aria-hidden="true">
            <circle v-for="radius in [44,88,132]" :key="radius" cx="290" cy="210" :r="radius" />
            <line x1="72" y1="210" x2="500" y2="210" /><line x1="290" y1="78" x2="290" y2="342" />
          </g>
          <line class="user-vector" x1="290" y1="210" :x2="sx(user.x)" :y2="sy(user.y)" />
          <circle class="user-tip" :cx="sx(user.x)" :cy="sy(user.y)" r="8" />
          <text class="user-label" :x="sx(user.x)" :y="sy(user.y) - 13" text-anchor="middle">p(E-17)</text>
          <g v-for="card in projectedCards" :key="card.id">
            <line class="item-vector" x1="290" y1="210" :x2="sx(card.x)" :y2="sy(card.y)" />
            <circle class="item-tip" :cx="sx(card.x)" :cy="sy(card.y)" r="6" />
            <text class="item-label" :x="sx(card.x)" :y="sy(card.y) - 10" text-anchor="middle">{{ card.id }}</text>
          </g>
          <path class="rotation-arrow" d="M 386 117 A 132 132 0 0 1 419 183" />
          <text class="rotation-label" x="430" y="124">共同旋转 {{ commonRotation }}°</text>

          <rect class="rank-card" x="526" y="42" width="92" height="334" rx="2" />
          <text class="rank-title" x="542" y="67">TOP-K</text>
          <g v-for="(card, index) in ranking" :key="card.id" :transform="`translate(542 ${98 + index * 43})`">
            <text class="rank-index" x="0" y="0">{{ index + 1 }}</text>
            <text class="rank-id" x="18" y="0">{{ card.id }}</text>
            <rect class="score-track" x="18" y="9" width="48" height="5" />
            <rect class="score-fill" x="18" y="9" :width="Math.max(2, Math.min(48, (card.score + 1) / 2.4 * 48))" height="5" />
            <text class="rank-score" x="66" y="0" text-anchor="end">{{ card.score.toFixed(2) }}</text>
          </g>
          <line class="perforation" x1="34" y1="389" x2="618" y2="389" />
        </svg>
      </div>
    </template>

    <template #controls>
      <label class="control">
        <span>工程师偏好方向 <output>{{ userAngle }}°</output></span>
        <input v-model.number="userAngle" type="range" min="0" max="359" step="1" aria-label="工程师偏好向量角度">
      </label>
      <label class="control">
        <span>共同旋转 <output>{{ commonRotation }}°</output></span>
        <input v-model.number="commonRotation" type="range" min="0" max="180" step="5" aria-label="用户和物品向量的共同旋转角度">
      </label>
      <label class="control">
        <span>正则收缩 <output>{{ shrinkage.toFixed(2) }}</output></span>
        <input v-model.number="shrinkage" type="range" min="0" max="0.65" step="0.01" aria-label="潜向量收缩强度">
      </label>
      <button type="button" :aria-pressed="includeBias" @click="includeBias = !includeBias">{{ includeBias ? '物品偏置：计入' : '物品偏置：关闭' }}</button>
      <button type="button" class="reset" @click="reset">复位工作台</button>
    </template>

    <template #metrics>
      <div><span>Top-1 知识卡</span><strong>{{ ranking[0].id }} · {{ ranking[0].topic }}</strong></div>
      <div><span>Top-1 分数</span><strong>{{ ranking[0].score.toFixed(3) }}</strong></div>
      <div><span>第一、二名间隔</span><strong>{{ scoreGap.toFixed(3) }}</strong></div>
      <div><span>共同旋转引起的分数差</span><strong>0.000</strong></div>
    </template>

    <template #note>共同旋转只改变潜因子坐标系，不改变点积与榜单，所以不能把某一维事后命名为“振动兴趣”并当成可识别真相。正则收缩会压低交互项；物品偏置则能让普遍受欢迎的卡片越过方向稍近的对手。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.bench-scroll{overflow-x:auto}.bench-scroll svg{display:block;width:100%;height:auto}.housing{fill:#10222c;stroke:#e7762f;stroke-width:2}.field{fill:#17364a;stroke:#53616a}.caption,.user-label,.item-label,.rotation-label,.rank-title,.rank-index,.rank-id,.rank-score{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.caption{fill:#efe8d4;font-size:8px;font-weight:900;letter-spacing:.08em}.grid circle,.grid line{fill:none;stroke:#8fc8c2;stroke-width:.7;opacity:.18}.user-vector{stroke:#e7762f;stroke-width:5}.user-tip{fill:#efe8d4;stroke:#e7762f;stroke-width:3}.user-label{fill:#ffe3ca;font-size:9px;font-weight:850}.item-vector{stroke:#43b6ad;stroke-width:1.6;opacity:.62}.item-tip{fill:#43b6ad;stroke:#c9fff9;stroke-width:1.5}.item-label{fill:#d8eeeb;font-size:8px}.rotation-arrow{fill:none;stroke:#efe8d4;stroke-width:2;stroke-dasharray:6 5}.rotation-label{fill:#efe8d4;font-size:8px}.rank-card{fill:#efe8d4;stroke:#8b8064}.rank-title{fill:#17364a;font-size:9px;font-weight:900;letter-spacing:.1em}.rank-index{fill:#c44d43;font-size:9px;font-weight:900}.rank-id{fill:#17364a;font-size:9px;font-weight:850}.rank-score{fill:#56616a;font-size:7px}.score-track{fill:#c8c0aa}.score-fill{fill:#e7762f}.perforation{stroke:#53616a;stroke-dasharray:3 5}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#c3cee8;font-size:11px}.control output{color:#ffc28f;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#e7762f}button{min-height:44px;padding:9px;border:1px solid #536a74;border-radius:4px;color:#dbe6e8;background:#14252e;font:720 10px/1.35 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#e7762f;color:#ffe4d2;background:#4b2c20}.reset{border-color:#43b6ad;color:#bbfff8}button:focus-visible,input:focus-visible{outline:3px solid #e7762f;outline-offset:2px}@media(max-width:560px){.bench-scroll{margin-right:-18px;padding-right:18px}.bench-scroll svg{width:720px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
