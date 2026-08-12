<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Metric = 'l2' | 'l1'
type VoteMode = 'uniform' | 'distance'

const neighborCount = ref(5)
const queryVibration = ref(5.8)
const queryTemperature = ref(18)
const metric = ref<Metric>('l2')
const voteMode = ref<VoteMode>('uniform')

const classNames = ['叶轮不平衡', '联轴器不对中', '润滑不足']
const classColors = ['#d8a657', '#4db6ac', '#e76f51']
const cases = [
  [3.0, 9, 0], [3.7, 12, 0], [4.2, 8, 0], [5.0, 11, 0], [4.8, 15, 0], [2.8, 14, 0],
  [6.2, 13, 1], [6.8, 17, 1], [7.4, 12, 1], [8.0, 18, 1], [6.0, 21, 1], [8.5, 15, 1],
  [4.8, 23, 2], [5.8, 26, 2], [7.0, 27, 2], [3.8, 25, 2], [6.5, 22, 2], [8.2, 25, 2],
].map(([vibration, temperature, label], index) => ({ id: `WO-${String(index + 1).padStart(3, '0')}`, vibration, temperature, label }))

const vibrationMean = 5.6
const vibrationScale = 2.0
const temperatureMean = 17.5
const temperatureScale = 6.0
const plotX = (vibration: number) => 238 + ((vibration - vibrationMean) / vibrationScale) * 94
const plotY = (temperature: number) => 161 - ((temperature - temperatureMean) / temperatureScale) * 94

const result = computed(() => {
  const ranked = cases.map((item) => {
    const dx = (item.vibration - queryVibration.value) / vibrationScale
    const dy = (item.temperature - queryTemperature.value) / temperatureScale
    const distance = metric.value === 'l2' ? Math.hypot(dx, dy) : Math.abs(dx) + Math.abs(dy)
    return { ...item, distance }
  }).sort((left, right) => left.distance - right.distance || left.id.localeCompare(right.id))
  const selected = ranked.slice(0, neighborCount.value)
  const votes = [0, 0, 0]
  for (const item of selected) {
    const weight = voteMode.value === 'uniform' ? 1 : 1 / Math.max(item.distance, 0.04)
    votes[item.label] += weight
  }
  const winner = votes.reduce((best, value, index) => value > votes[best] ? index : best, 0)
  const total = votes.reduce((sum, value) => sum + value, 0)
  return { ranked, selected, votes, winner, confidence: votes[winner] / total, radius: selected.at(-1)?.distance ?? 0 }
})
</script>

<template>
  <MlExperimentFrame index="18-A" title="在检维修案例台上展开邻域投票" question="当前泵组究竟由哪些历史工单决定？改变 k、查询位置、距离或投票方式时，邻域半径与故障结论怎样一起变化？" formula="ŷ(x) = argmax_c Σᵢ∈Nₖ(x) wᵢ·1[yᵢ=c]">
    <template #visual>
      <figure class="inspection-bay">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`查询泵组预测为${classNames[result.winner]}，置信份额${result.confidence.toFixed(2)}，第k邻居距离${result.radius.toFixed(2)}`">
          <rect x="15" y="15" width="570" height="296" rx="8" class="bench" />
          <text x="31" y="38" class="panel-title">离心泵案例邻域量规 · 坐标已按训练集尺度标定</text>
          <rect x="31" y="51" width="408" height="230" rx="7" class="map-panel" />
          <line x1="48" y1="161" x2="423" y2="161" class="axis" /><line x1="238" y1="66" x2="238" y2="267" class="axis" />
          <text x="302" y="274" class="axis-label">振动 RMS →</text><text x="44" y="80" class="axis-label">温升 ↑</text>
          <circle v-if="metric==='l2'" :cx="plotX(queryVibration)" :cy="plotY(queryTemperature)" :r="result.radius*94" class="caliper-ring" />
          <path v-else :d="`M ${plotX(queryVibration)} ${plotY(queryTemperature)-result.radius*94} L ${plotX(queryVibration)+result.radius*94} ${plotY(queryTemperature)} L ${plotX(queryVibration)} ${plotY(queryTemperature)+result.radius*94} L ${plotX(queryVibration)-result.radius*94} ${plotY(queryTemperature)} Z`" class="caliper-ring" />
          <line v-for="item in result.selected" :key="`lead-${item.id}`" :x1="plotX(queryVibration)" :y1="plotY(queryTemperature)" :x2="plotX(item.vibration)" :y2="plotY(item.temperature)" class="inspection-lead" :style="{stroke:classColors[item.label],strokeWidth:1.2+3*(1-item.distance/(result.radius+.01))}" />
          <g v-for="item in result.ranked" :key="item.id">
            <circle :cx="plotX(item.vibration)" :cy="plotY(item.temperature)" :r="result.selected.includes(item) ? 8 : 5" :class="['case-dot',{selected:result.selected.includes(item)}]" :style="{fill:classColors[item.label]}" />
            <text v-if="result.selected.includes(item)" :x="plotX(item.vibration)" :y="plotY(item.temperature)+2.5" class="case-rank">{{ result.selected.indexOf(item)+1 }}</text>
          </g>
          <g class="query-puck"><circle :cx="plotX(queryVibration)" :cy="plotY(queryTemperature)" r="10" /><line :x1="plotX(queryVibration)-14" :y1="plotY(queryTemperature)" :x2="plotX(queryVibration)+14" :y2="plotY(queryTemperature)" /><line :x1="plotX(queryVibration)" :y1="plotY(queryTemperature)-14" :x2="plotX(queryVibration)" :y2="plotY(queryTemperature)+14" /></g>
          <rect x="451" y="51" width="118" height="230" rx="7" class="ledger-panel" />
          <text x="464" y="72" class="section-label">最近案例清单</text>
          <g v-for="(item,index) in result.ranked.slice(0,5)" :key="`rank-${item.id}`">
            <rect x="462" :y="83+index*34" width="94" height="27" rx="4" :class="['rank-card',{outside:index>=neighborCount}]" />
            <circle cx="473" :cy="96+index*34" r="4" :style="{fill:classColors[item.label]}" />
            <text x="482" :y="94+index*34" class="rank-id">{{ item.id }}</text><text x="482" :y="104+index*34" class="rank-distance">d={{ item.distance.toFixed(2) }}</text>
          </g>
          <text x="464" y="267" class="ledger-note">黄铜圈：第 k 邻域边界</text>
          <g v-for="(name,index) in classNames" :key="name"><rect :x="48+index*124" y="291" width="10" height="8" rx="2" :style="{fill:classColors[index]}" /><text :x="63+index*124" y="298" class="legend-label">{{ name }}</text></g>
        </svg>
      </figure>
    </template>
    <template #controls>
      <label><span>邻居数 k <output>{{ neighborCount }}</output></span><input v-model.number="neighborCount" aria-label="邻居数 k" type="range" min="1" max="11" step="2"></label>
      <label><span>查询振动 RMS <output>{{ queryVibration.toFixed(1) }} mm/s</output></span><input v-model.number="queryVibration" aria-label="查询振动 RMS" type="range" min="2.5" max="8.5" step=".1"></label>
      <label><span>查询轴承温升 <output>{{ queryTemperature.toFixed(0) }} °C</output></span><input v-model.number="queryTemperature" aria-label="查询轴承温升" type="range" min="7" max="28" step="1"></label>
      <div class="switch" role="group" aria-label="距离度量"><button type="button" :aria-pressed="metric==='l2'" @click="metric='l2'">欧氏距离 L2</button><button type="button" :aria-pressed="metric==='l1'" @click="metric='l1'">曼哈顿距离 L1</button></div>
      <div class="switch" role="group" aria-label="邻域投票方式"><button type="button" :aria-pressed="voteMode==='uniform'" @click="voteMode='uniform'">等权投票</button><button type="button" :aria-pressed="voteMode==='distance'" @click="voteMode='distance'">反距离投票</button></div>
      <p>编号是从近到远的检索顺序；圆或菱形量规经过第 k 个邻居。增大 k 不只是“多投几票”，也把查询扩展到更大的物理邻域。</p>
    </template>
    <template #metrics><div><span>分诊结果</span><strong>{{ classNames[result.winner] }}</strong></div><div><span>赢家票重份额</span><strong>{{ result.confidence.toFixed(3) }}</strong></div><div><span>第 k 邻居半径</span><strong>{{ result.radius.toFixed(3) }}</strong></div><div><span>三类票重</span><strong>{{ result.votes.map(v=>v.toFixed(1)).join(' / ') }}</strong></div></template>
    <template #note>二维位置只是训练集标准化后的教学切片。真实工单还包含转速、载荷、泵型和维护状态；若这些字段没有进入距离，图上看似“很近”的案例也可能不具备可比工况。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.inspection-bay{margin:0}.inspection-bay svg{display:block;width:100%;border:1px solid #3a5961;border-radius:10px;background:#10262b}.bench{fill:#10262b}.inspection-bay text{font:700 7px ui-monospace,monospace}.panel-title{fill:#e7eceb;font-size:10px}.map-panel,.ledger-panel{fill:#172f35;stroke:#45656d}.axis{stroke:#526f76;stroke-dasharray:3 4}.axis-label,.section-label,.ledger-note,.legend-label{fill:#91a7ab}.caliper-ring{fill:#d8a65712;stroke:#d8a657;stroke-width:1.5;stroke-dasharray:5 3}.inspection-lead{opacity:.72}.case-dot{stroke:#10262b;stroke-width:2}.case-dot.selected{stroke:#e7eceb;filter:drop-shadow(0 0 3px #d8a657)}.case-rank{fill:#10262b;text-anchor:middle;font-size:6px}.query-puck circle{fill:#e7eceb;stroke:#f47b5f;stroke-width:3}.query-puck line{stroke:#f47b5f;stroke-width:1.4}.rank-card{fill:#203c43;stroke:#4e6f76}.rank-card.outside{opacity:.35}.rank-id{fill:#e7eceb}.rank-distance{fill:#91a7ab;font-size:6px}.switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.switch button{min-height:44px;border:1px solid #54747b;border-radius:7px;color:#dce8e7;background:#173039;font:750 9px ui-monospace,monospace;cursor:pointer}.switch button[aria-pressed="true"]{border-color:#d8a657;color:#f0ca7c;background:#3b3428}.switch button:focus-visible{outline:3px solid #4db6ac;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#cbdcdd;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#d8a657;font:750 10px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e76f51;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9db0b4;font-size:10px;line-height:1.65}@media(max-width:560px){.inspection-bay{overflow-x:auto;overscroll-behavior-inline:contain}.inspection-bay svg{width:560px;max-width:none}}
</style>
