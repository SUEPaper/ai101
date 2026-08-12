<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Feature = 'pressure' | 'fill'
type Criterion = 'gini' | 'entropy' | 'error'

const classNames = ['合格', '欠注', '飞边']
const classColors = ['#6cc7a8', '#eab64d', '#e36552']
const points = [
  { id: 'M01', pressure: 58, fill: 1.34, label: 1 }, { id: 'M02', pressure: 61, fill: 1.28, label: 1 },
  { id: 'M03', pressure: 63, fill: 1.17, label: 1 }, { id: 'M04', pressure: 65, fill: 1.31, label: 1 },
  { id: 'M05', pressure: 66, fill: 1.10, label: 0 }, { id: 'M06', pressure: 68, fill: 1.03, label: 0 },
  { id: 'M07', pressure: 70, fill: 1.14, label: 0 }, { id: 'M08', pressure: 72, fill: 0.98, label: 0 },
  { id: 'M09', pressure: 73, fill: 1.20, label: 0 }, { id: 'M10', pressure: 75, fill: 1.07, label: 0 },
  { id: 'M11', pressure: 76, fill: 0.93, label: 0 }, { id: 'M12', pressure: 77, fill: 1.14, label: 0 },
  { id: 'M13', pressure: 79, fill: 1.02, label: 2 }, { id: 'M14', pressure: 80, fill: 0.91, label: 2 },
  { id: 'M15', pressure: 82, fill: 1.08, label: 2 }, { id: 'M16', pressure: 84, fill: 0.96, label: 2 },
  { id: 'M17', pressure: 85, fill: 1.18, label: 2 }, { id: 'M18', pressure: 74, fill: 1.29, label: 1 },
]

const feature = ref<Feature>('pressure')
const criterion = ref<Criterion>('gini')
const threshold = ref(65)

const impurity = (items: typeof points) => {
  if (!items.length) return 0
  const counts = [0, 0, 0]
  items.forEach(item => counts[item.label]++)
  const probabilities = counts.map(count => count / items.length)
  if (criterion.value === 'gini') return 1 - probabilities.reduce((sum, p) => sum + p * p, 0)
  if (criterion.value === 'error') return 1 - Math.max(...probabilities)
  return -probabilities.reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0)
}

const result = computed(() => {
  const left = points.filter(item => item[feature.value] <= threshold.value)
  const right = points.filter(item => item[feature.value] > threshold.value)
  const parent = impurity(points)
  const children = (left.length * impurity(left) + right.length * impurity(right)) / points.length
  const count = (items: typeof points) => [0, 1, 2].map(label => items.filter(item => item.label === label).length)
  return { left, right, parent, children, gain: parent - children, leftCount: count(left), rightCount: count(right) }
})

const range = computed(() => feature.value === 'pressure'
  ? { min: 57, max: 85, step: 1, unit: 'MPa' }
  : { min: .90, max: 1.35, step: .01, unit: 's' })

watch(feature, value => { threshold.value = value === 'pressure' ? 65 : 1.18 })

const plotX = (value: number) => 42 + ((value - 55) / 32) * 378
const plotY = (value: number) => 274 - ((value - .85) / .55) * 220
const countText = (counts: number[]) => counts.map((count, index) => `${classNames[index]} ${count}`).join(' · ')
</script>

<template>
  <MlExperimentFrame index="19-A" title="在分模线量规上审计第一刀" question="同一批注塑件沿压力或充填时间切开后，左右工位究竟变得多纯？分类错误率为什么可能看不见一个有价值的切分？" formula="ΔI = I(parent) − nL/n·I(L) − nR/n·I(R)">
    <template #visual>
      <div class="foundry-scroll">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`${feature==='pressure'?'压力':'充填时间'}阈值${threshold}${range.unit}，不纯度下降${result.gain.toFixed(3)}`">
          <rect x="0" y="0" width="600" height="326" rx="12" class="plate" />
          <text x="26" y="24" class="eyebrow">MOLD 07 · 18 个首件样本 · 当前候选切分</text>
          <rect x="26" y="38" width="410" height="258" rx="8" class="plot" />
          <line x1="42" y1="274" x2="420" y2="274" class="axis" /><line x1="42" y1="54" x2="42" y2="274" class="axis" />
          <text x="330" y="291" class="axis-label">峰值压力 / MPa →</text><text x="17" y="171" class="axis-label vertical">充填时间 / s →</text>
          <g v-if="feature==='pressure'">
            <rect x="42" y="54" :width="Math.max(0, plotX(threshold)-42)" height="220" class="left-zone" />
            <line :x1="plotX(threshold)" y1="47" :x2="plotX(threshold)" y2="281" class="split-line" />
            <text :x="plotX(threshold)+6" y="68" class="split-tag">≤ {{ threshold }} MPa</text>
          </g>
          <g v-else>
            <rect x="42" :y="plotY(threshold)" width="378" :height="274-plotY(threshold)" class="left-zone" />
            <line x1="35" :y1="plotY(threshold)" x2="427" :y2="plotY(threshold)" class="split-line" />
            <text x="48" :y="plotY(threshold)-7" class="split-tag">≤ {{ threshold.toFixed(2) }} s</text>
          </g>
          <g v-for="item in points" :key="item.id">
            <circle :cx="plotX(item.pressure)" :cy="plotY(item.fill)" r="7" :style="{fill:classColors[item.label]}" class="sample" />
            <text :x="plotX(item.pressure)" :y="plotY(item.fill)+2.5" class="sample-id">{{ item.id.slice(1) }}</text>
          </g>
          <g transform="translate(455 49)">
            <text x="0" y="0" class="ledger-title">CHILD LEDGER</text>
            <rect x="0" y="15" width="122" height="91" rx="6" class="ledger" />
            <text x="10" y="35" class="side">≤ 阈值 · n={{ result.left.length }}</text>
            <g v-for="(count,index) in result.leftCount" :key="`l-${index}`">
              <rect x="10" :y="45+index*15" :width="count*8" height="8" rx="2" :style="{fill:classColors[index]}" />
              <text x="104" :y="52+index*15" class="count">{{ count }}</text>
            </g>
            <rect x="0" y="116" width="122" height="91" rx="6" class="ledger" />
            <text x="10" y="136" class="side">&gt; 阈值 · n={{ result.right.length }}</text>
            <g v-for="(count,index) in result.rightCount" :key="`r-${index}`">
              <rect x="10" :y="146+index*15" :width="count*8" height="8" rx="2" :style="{fill:classColors[index]}" />
              <text x="104" :y="153+index*15" class="count">{{ count }}</text>
            </g>
            <text x="0" y="229" class="ledger-title">GAIN</text>
            <rect x="0" y="239" width="122" height="10" rx="5" class="gain-track" />
            <rect x="0" y="239" :width="Math.min(122, result.gain*220)" height="10" rx="5" class="gain-fill" />
            <text x="0" y="270" class="gain-value">{{ result.gain.toFixed(3) }}</text>
          </g>
        </svg>
      </div>
    </template>
    <template #controls>
      <div class="control"><span>候选特征</span><div class="switch" role="group" aria-label="候选切分特征"><button type="button" :aria-pressed="feature==='pressure'" @click="feature='pressure'">峰值压力</button><button type="button" :aria-pressed="feature==='fill'" @click="feature='fill'">充填时间</button></div></div>
      <label class="control"><span>切分阈值 <output>{{ feature==='pressure' ? threshold.toFixed(0) : threshold.toFixed(2) }} {{ range.unit }}</output></span><input v-model.number="threshold" aria-label="切分阈值" type="range" :min="range.min" :max="range.max" :step="range.step"></label>
      <div class="control"><span>不纯度准则</span><div class="switch three" role="group" aria-label="不纯度准则"><button type="button" :aria-pressed="criterion==='gini'" @click="criterion='gini'">Gini</button><button type="button" :aria-pressed="criterion==='entropy'" @click="criterion='entropy'">熵</button><button type="button" :aria-pressed="criterion==='error'" @click="criterion='error'">错误率</button></div></div>
      <p class="hint">{{ countText(result.leftCount) }}<br>{{ countText(result.rightCount) }}</p>
    </template>
    <template #metrics><div><span>父节点不纯度</span><strong>{{ result.parent.toFixed(3) }}</strong></div><div><span>加权子节点不纯度</span><strong>{{ result.children.toFixed(3) }}</strong></div><div><span>不纯度下降</span><strong>{{ result.gain.toFixed(3) }}</strong></div><div><span>左右样本数</span><strong>{{ result.left.length }} / {{ result.right.length }}</strong></div></template>
    <template #note>误分类率只关心多数类，常对不同切分给出相同分数；Gini 与熵使用完整类别比例，更能分辨“一个纯子节点加一个混合子节点”的早期价值。阈值只在相邻不同观测值之间改变样本集合。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.foundry-scroll{overflow-x:auto}.foundry-scroll svg{display:block;min-width:560px;width:100%;height:auto}.plate{fill:#101923}.plot{fill:#172630;stroke:#47606f}.axis{stroke:#78909e;stroke-width:1}.axis-label,.eyebrow,.ledger-title,.side,.count,.gain-value,.split-tag{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#84cbd4;font-size:9px;letter-spacing:.08em}.axis-label{fill:#8496a5;font-size:8px}.vertical{transform:rotate(-90deg);transform-origin:17px 171px}.left-zone{fill:#3ab5c4;opacity:.10}.split-line{stroke:#f0c15b;stroke-width:2;stroke-dasharray:6 4}.split-tag{fill:#f3ce77;font-size:8px}.sample{stroke:#eef5f3;stroke-width:1.2}.sample-id{fill:#0d1820;font:700 6px/1 ui-monospace,monospace;text-anchor:middle}.ledger{fill:#14222c;stroke:#405867}.ledger-title{fill:#84cbd4;font-size:8px;letter-spacing:.08em}.side{fill:#cbd7df;font-size:8px}.count{fill:#96a8b3;font-size:7px;text-anchor:end}.gain-track{fill:#283a45}.gain-fill{fill:#eab64d}.gain-value{fill:#f3ce77;font-size:18px;font-weight:800}.control{display:grid;gap:8px;margin:0}.control>span{display:flex;justify-content:space-between;color:#aebbdc;font-size:11px}.control output{color:#f3ce77;font:700 11px/1 ui-monospace,monospace}.switch{display:grid;grid-template-columns:1fr 1fr;gap:7px}.switch.three{grid-template-columns:repeat(3,1fr)}button{min-height:44px;border:1px solid #46606f;border-radius:6px;color:#dbe5e9;background:#14232c;font:700 10px/1.2 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#eab64d;color:#f6d983;background:#30291e;box-shadow:inset 0 0 0 1px #9a7935}button:focus-visible,input:focus-visible{outline:3px solid #74d7dc;outline-offset:2px}input[type="range"]{width:100%;min-height:44px;accent-color:#e36552}.hint{margin:0;padding:10px;border-left:3px solid #3ab5c4;color:#9fb0ba;background:#111d26;font:9px/1.7 ui-monospace,monospace}@media(max-width:560px){.foundry-scroll{margin-right:-18px;padding-right:18px}.foundry-scroll svg{width:560px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
