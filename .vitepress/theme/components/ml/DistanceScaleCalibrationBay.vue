<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type ScaleMode = 'raw' | 'standardized'
type Metric = 'l2' | 'l1'

const scaleMode = ref<ScaleMode>('raw')
const metric = ref<Metric>('l2')
const temperatureUnitFactor = ref(1)
const classNames = ['叶轮不平衡', '联轴器不对中', '润滑不足']
const classColors = ['#d8a657', '#4db6ac', '#e76f51']
const query = { sideband: .62, temperature: 12 }
const cases = [
  { id:'C-014', label:1, sideband:.60, temperature:13.0 },
  { id:'C-051', label:0, sideband:.30, temperature:12.1 },
  { id:'C-078', label:1, sideband:.72, temperature:16.0 },
  { id:'C-103', label:2, sideband:.52, temperature:7.0 },
  { id:'C-126', label:2, sideband:.90, temperature:11.5 },
  { id:'C-144', label:0, sideband:.20, temperature:18.0 },
]

const sidebandMean = cases.reduce((sum,item)=>sum+item.sideband,0)/cases.length
const temperatureMean = cases.reduce((sum,item)=>sum+item.temperature,0)/cases.length
const sidebandStd = Math.sqrt(cases.reduce((sum,item)=>sum+(item.sideband-sidebandMean)**2,0)/cases.length)
const temperatureStd = Math.sqrt(cases.reduce((sum,item)=>sum+(item.temperature-temperatureMean)**2,0)/cases.length)

function rankWith(factor: number, mode: ScaleMode, chosenMetric: Metric) {
  return cases.map((item) => {
    const rawSideband = item.sideband-query.sideband
    const rawTemperature = (item.temperature-query.temperature)*factor
    const dx = mode==='standardized' ? rawSideband/sidebandStd : rawSideband
    const dy = mode==='standardized' ? rawTemperature/(temperatureStd*factor) : rawTemperature
    const parts = chosenMetric==='l2' ? [dx**2,dy**2] : [Math.abs(dx),Math.abs(dy)]
    const distance = chosenMetric==='l2' ? Math.sqrt(parts[0]+parts[1]) : parts[0]+parts[1]
    return {...item,dx,dy,parts,distance,tempShare:parts[1]/Math.max(parts[0]+parts[1],1e-12)}
  }).sort((left,right)=>left.distance-right.distance||left.id.localeCompare(right.id))
}

const result = computed(() => {
  const ranked=rankWith(temperatureUnitFactor.value,scaleMode.value,metric.value)
  const low=rankWith(.1,scaleMode.value,metric.value)[0]
  const high=rankWith(10,scaleMode.value,metric.value)[0]
  const maximum=Math.max(...ranked.map(item=>item.distance))
  return {ranked,maximum,unitStable:low.id===high.id,low,high}
})
const barWidth=(distance:number)=>206*distance/Math.max(result.value.maximum,1e-9)
</script>

<template>
  <MlExperimentFrame index="18-B" title="用双单位校准尺审计距离" question="只把温升的记录单位从 °C 改成 0.1°C 计数，最近案例为什么可能换人？训练折标准化后，单位变换为何应当不再改变排名？" formula="d² = ((x₁−z₁)/s₁)² + ((x₂−z₂)/s₂)²">
    <template #visual>
      <figure class="calibration-bay">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`最近案例${result.ranked[0].id}，温升距离贡献${(result.ranked[0].tempShare*100).toFixed(0)}%，单位切换${result.unitStable?'不改排名':'会改排名'}`">
          <rect x="15" y="15" width="570" height="296" rx="8" class="bench"/><text x="31" y="38" class="panel-title">距离校准证书 · 查询 Q-217：边带比 0.62 / 温升 12°C</text>
          <rect x="31" y="51" width="538" height="76" rx="7" class="ruler-panel"/>
          <text x="45" y="72" class="ruler-label">边带能量比</text><line x1="138" y1="68" x2="535" y2="68" class="ruler"/><g v-for="tick in 6" :key="`s-${tick}`"><line :x1="138+(tick-1)*79.4" y1="63" :x2="138+(tick-1)*79.4" y2="74" class="tick"/><text :x="138+(tick-1)*79.4" y="84" class="tick-label">{{ ((tick-1)*.2).toFixed(1) }}</text></g><circle :cx="138+query.sideband*397" cy="68" r="7" class="query-marker"/>
          <text x="45" y="105" class="ruler-label">温升记录值</text><line x1="138" y1="101" x2="535" y2="101" class="ruler"/><g v-for="tick in 6" :key="`t-${tick}`"><line :x1="138+(tick-1)*79.4" y1="96" :x2="138+(tick-1)*79.4" y2="107" class="tick"/><text :x="138+(tick-1)*79.4" y="118" class="tick-label">{{ Math.round((tick-1)*5*temperatureUnitFactor) }}</text></g><circle :cx="138+(query.temperature/25)*397" cy="101" r="7" class="query-marker"/>
          <rect x="31" y="139" width="368" height="151" rx="7" class="ranking-panel"/><text x="45" y="158" class="section-label">候选案例距离 · 越短越相似</text>
          <g v-for="(item,index) in result.ranked.slice(0,5)" :key="item.id"><text x="47" :y="180+index*22" class="rank-number">{{ index+1 }}</text><circle cx="64" :cy="177+index*22" r="4" :style="{fill:classColors[item.label]}"/><text x="75" :y="180+index*22" class="case-id">{{ item.id }}</text><rect x="122" :y="170+index*22" width="218" height="12" rx="3" class="bar-track"/><rect x="122" :y="170+index*22" :width="barWidth(item.distance)" height="12" rx="3" :style="{fill:classColors[item.label]}" class="distance-bar"/><text x="349" :y="180+index*22" class="distance-value">{{ item.distance.toFixed(3) }}</text></g>
          <rect x="413" y="139" width="156" height="151" rx="7" class="certificate-panel"/><text x="491" y="159" class="certificate-title" text-anchor="middle">CALIBRATION</text><circle cx="491" cy="198" r="25" :class="['certificate-stamp',{pass:result.unitStable}]"/><text x="491" y="195" :class="['stamp-main',{pass:result.unitStable}]" text-anchor="middle">{{ result.unitStable?'PASS':'DRIFT' }}</text><text x="491" y="207" class="stamp-note" text-anchor="middle">UNIT TEST</text><text x="429" y="239" class="certificate-line">0.1× → {{ result.low.id }}</text><text x="429" y="255" class="certificate-line">10× → {{ result.high.id }}</text><text x="429" y="275" class="certificate-note">标准化统计只来自训练折</text>
          <g v-for="(name,index) in classNames" :key="name"><rect :x="45+index*122" y="302" width="10" height="8" rx="2" :style="{fill:classColors[index]}"/><text :x="60+index*122" y="309" class="legend-label">{{ name }}</text></g>
        </svg>
      </figure>
    </template>
    <template #controls>
      <div class="switch" role="group" aria-label="特征标定方式"><button type="button" :aria-pressed="scaleMode==='raw'" @click="scaleMode='raw'">直接使用原值</button><button type="button" :aria-pressed="scaleMode==='standardized'" @click="scaleMode='standardized'">训练折 Z-score</button></div>
      <label><span>温升单位倍率 <output>×{{ temperatureUnitFactor.toFixed(1) }}</output></span><input v-model.number="temperatureUnitFactor" aria-label="温升单位倍率" type="range" min=".1" max="10" step=".1"></label>
      <div class="switch" role="group" aria-label="距离范数"><button type="button" :aria-pressed="metric==='l2'" @click="metric='l2'">平方和 L2</button><button type="button" :aria-pressed="metric==='l1'" @click="metric='l1'">绝对和 L1</button></div>
      <p>原始距离会把“单位选择”误当成“业务重要性”。Z-score 能消除线性单位倍率，但不判断特征是否有用，也不能替代领域权重和距离学习。</p>
    </template>
    <template #metrics><div><span>最近历史案例</span><strong>{{ result.ranked[0].id }}</strong></div><div><span>最近案例标签</span><strong>{{ classNames[result.ranked[0].label] }}</strong></div><div><span>温升贡献占比</span><strong>{{ (result.ranked[0].tempShare*100).toFixed(1) }}%</strong></div><div><span>单位倍率压力测试</span><strong>{{ result.unitStable?'排名稳定':'最近案例漂移' }}</strong></div></template>
    <template #note>这里用总体形式展示尺度抵消，生产交叉验证必须在每个训练折单独估计均值与标准差。若先在全数据上标准化，验证折已通过预处理影响了邻居关系。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.calibration-bay{margin:0}.calibration-bay svg{display:block;width:100%;border:1px solid #3a5961;border-radius:10px;background:#10262b}.bench{fill:#10262b}.calibration-bay text{font:700 7px ui-monospace,monospace}.panel-title{fill:#e7eceb;font-size:10px}.ruler-panel,.ranking-panel,.certificate-panel{fill:#172f35;stroke:#45656d}.ruler{stroke:#718b90;stroke-width:2}.tick{stroke:#91a7ab}.tick-label{fill:#91a7ab;text-anchor:middle;font-size:6px}.ruler-label,.section-label,.legend-label{fill:#a7b9bc}.query-marker{fill:#f47b5f;stroke:#e7eceb;stroke-width:2}.rank-number{fill:#d8a657}.case-id{fill:#dbe6e5}.bar-track{fill:#223f46}.distance-bar{opacity:.82}.distance-value{fill:#d8a657}.certificate-title{fill:#d8a657;letter-spacing:.12em}.certificate-stamp{fill:#402a29;stroke:#e76f51;stroke-width:2}.certificate-stamp.pass{fill:#203e39;stroke:#4db6ac}.stamp-main{fill:#e76f51;font-size:10px}.stamp-main.pass{fill:#4db6ac}.stamp-note,.certificate-line,.certificate-note{fill:#a7b9bc}.certificate-note{font-size:6px}.switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.switch button{min-height:44px;border:1px solid #54747b;border-radius:7px;color:#dce8e7;background:#173039;font:750 9px ui-monospace,monospace;cursor:pointer}.switch button[aria-pressed="true"]{border-color:#d8a657;color:#f0ca7c;background:#3b3428}.switch button:focus-visible{outline:3px solid #4db6ac;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#cbdcdd;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#d8a657;font:750 10px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e76f51;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9db0b4;font-size:10px;line-height:1.65}@media(max-width:560px){.calibration-bay{overflow-x:auto;overscroll-behavior-inline:contain}.calibration-bay svg{width:560px;max-width:none}}
</style>
