<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Metric = 'l2' | 'l1'
const dimensionIndex = ref(3)
const sampleCount = ref(500)
const neighborhoodPercent = ref(1)
const metric = ref<Metric>('l2')
const dimensions = [2, 4, 8, 16, 32, 64, 128]

const result = computed(() => {
  const dimension=dimensions[dimensionIndex.value]
  let state=1801
  const random=()=>{state=(1664525*state+1013904223)>>>0;return state/4294967296}
  const distances:number[]=[]
  for(let row=0;row<sampleCount.value;row+=1){
    let accumulator=0
    for(let column=0;column<dimension;column+=1){const delta=Math.abs(random()-.5);accumulator+=metric.value==='l2'?delta*delta:delta}
    distances.push(metric.value==='l2'?Math.sqrt(accumulator):accumulator)
  }
  distances.sort((a,b)=>a-b)
  const nearest=distances[0]
  const median=distances[Math.floor(distances.length/2)]
  const farthest=distances.at(-1)??nearest
  const minimum=nearest, maximum=farthest
  const bins=Array.from({length:20},()=>0)
  for(const distance of distances){const index=Math.min(19,Math.floor((distance-minimum)/Math.max(maximum-minimum,1e-12)*20));bins[index]+=1}
  const maximumBin=Math.max(...bins)
  const edge=Math.pow(neighborhoodPercent.value/100,1/dimension)
  return {dimension,nearest,median,farthest,bins,maximumBin,edge,contrast:(farthest-nearest)/nearest,nearestToMedian:nearest/median,minimum,maximum}
})
const histogramX=(index:number)=>48+index*17
const histogramHeight=(count:number)=>116*count/result.value.maximumBin
const distanceX=(value:number)=>48+340*(value-result.value.minimum)/Math.max(result.value.maximum-result.value.minimum,1e-12)
</script>

<template>
  <MlExperimentFrame index="18-C" title="穿过高维距离公差隧道" question="维度不断增加时，最近点、典型点和最远点为何挤到相似距离？为了覆盖固定样本比例，每一维需要跨过多大范围？" formula="edge(p,d)=p¹ᐟᵈ, contrast=(dₘₐₓ−dₘᵢₙ)/dₘᵢₙ">
    <template #visual>
      <figure class="distance-tunnel">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`${result.dimension}维，最近中位距离比${result.nearestToMedian.toFixed(2)}，相对距离反差${result.contrast.toFixed(2)}`">
          <rect x="15" y="15" width="570" height="296" rx="8" class="bench"/><text x="31" y="38" class="panel-title">高维距离公差隧道 · {{ sampleCount }} 个均匀样本对中心查询</text>
          <rect x="31" y="51" width="377" height="185" rx="7" class="histogram-panel"/><text x="45" y="70" class="section-label">查询距离分布</text>
          <line x1="48" y1="207" x2="388" y2="207" class="axis"/>
          <g v-for="(count,index) in result.bins" :key="index"><rect :x="histogramX(index)" :y="202-histogramHeight(count)" width="13" :height="histogramHeight(count)" rx="2" class="histogram-bar"/></g>
          <line :x1="distanceX(result.nearest)" y1="78" :x2="distanceX(result.nearest)" y2="213" class="marker nearest"/><line :x1="distanceX(result.median)" y1="78" :x2="distanceX(result.median)" y2="213" class="marker median"/><line :x1="distanceX(result.farthest)" y1="78" :x2="distanceX(result.farthest)" y2="213" class="marker farthest"/>
          <text :x="distanceX(result.nearest)+4" y="88" class="marker-label nearest">NEAR</text><text :x="distanceX(result.median)+4" y="102" class="marker-label median">MEDIAN</text><text :x="distanceX(result.farthest)-4" y="88" class="marker-label farthest" text-anchor="end">FAR</text>
          <text x="48" y="224" class="axis-label">{{ result.minimum.toFixed(2) }}</text><text x="388" y="224" class="axis-label" text-anchor="end">{{ result.maximum.toFixed(2) }}</text>
          <rect x="421" y="51" width="148" height="185" rx="7" class="tunnel-panel"/><text x="495" y="70" class="section-label" text-anchor="middle">{{ neighborhoodPercent }}% 体积截面</text>
          <g class="perspective"><path d="M440 89 L550 89 L540 215 L450 215 Z"/><path d="M450 102 L540 102 L532 202 L458 202 Z"/><path d="M460 115 L530 115 L524 189 L466 189 Z"/><path d="M470 128 L520 128 L516 176 L474 176 Z"/></g>
          <rect :x="495-42*result.edge" :y="152-42*result.edge" :width="84*result.edge" :height="84*result.edge" class="coverage-window"/><text x="495" y="148" class="edge-value" text-anchor="middle">{{ (result.edge*100).toFixed(1) }}%</text><text x="495" y="161" class="edge-note" text-anchor="middle">每维边长</text>
          <rect x="31" y="249" width="538" height="43" rx="7" class="contrast-panel"/><text x="46" y="267" class="contrast-label">近 / 中 / 远的可分辨空间</text><rect x="199" y="258" width="342" height="12" rx="6" class="contrast-track"/><rect x="199" y="258" :width="342*Math.min(result.contrast,1)" height="12" rx="6" class="contrast-fill"/><text x="199" y="283" class="contrast-note">维度升高时填充条缩短：最近邻不再显著“近”</text><text x="541" y="283" class="contrast-note" text-anchor="end">{{ metric.toUpperCase() }}</text>
          <g transform="translate(426 302)"><rect width="10" height="8" rx="2" class="legend-near"/><text x="15" y="7" class="legend-label">最近</text><rect x="52" width="10" height="8" rx="2" class="legend-median"/><text x="67" y="7" class="legend-label">中位</text><rect x="104" width="10" height="8" rx="2" class="legend-far"/><text x="119" y="7" class="legend-label">最远</text></g>
        </svg>
      </figure>
    </template>
    <template #controls>
      <label><span>特征维数 d <output>{{ result.dimension }}</output></span><input v-model.number="dimensionIndex" aria-label="特征维数 d" type="range" min="0" max="6" step="1"></label>
      <label><span>历史样本数 n <output>{{ sampleCount }}</output></span><input v-model.number="sampleCount" aria-label="历史样本数 n" type="range" min="100" max="1000" step="100"></label>
      <label><span>目标邻域体积 <output>{{ neighborhoodPercent }}%</output></span><input v-model.number="neighborhoodPercent" aria-label="目标邻域体积" type="range" min="1" max="20" step="1"></label>
      <div class="switch" role="group" aria-label="高维距离范数"><button type="button" :aria-pressed="metric==='l2'" @click="metric='l2'">欧氏距离 L2</button><button type="button" :aria-pressed="metric==='l1'" @click="metric='l1'">曼哈顿距离 L1</button></div>
      <p>样本由固定伪随机序列生成，调节维度时前缀坐标一致。增加样本会改善最近距离，但无法改变覆盖体积随维度指数增长的几何事实。</p>
    </template>
    <template #metrics><div><span>最近 / 中位距离</span><strong>{{ result.nearestToMedian.toFixed(3) }}</strong></div><div><span>相对距离反差</span><strong>{{ result.contrast.toFixed(3) }}</strong></div><div><span>目标体积每维边长</span><strong>{{ (result.edge*100).toFixed(1) }}%</strong></div><div><span>距离分量计算</span><strong>{{ (sampleCount*result.dimension).toLocaleString() }}</strong></div></template>
    <template #note>距离集中并不意味着所有高维任务都失败；若数据落在低维流形、表示已去除无关方向或距离经过任务学习，近邻仍可有效。真正应监控的是邻域纯度、距离反差和运行域覆盖。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.distance-tunnel{margin:0}.distance-tunnel svg{display:block;width:100%;border:1px solid #3a5961;border-radius:10px;background:#10262b}.bench{fill:#10262b}.distance-tunnel text{font:700 7px ui-monospace,monospace}.panel-title{fill:#e7eceb;font-size:10px}.histogram-panel,.tunnel-panel,.contrast-panel{fill:#172f35;stroke:#45656d}.section-label,.axis-label,.contrast-label,.contrast-note,.legend-label{fill:#9fb2b5}.axis{stroke:#617c82}.histogram-bar{fill:#4db6ac;opacity:.72}.marker{stroke-width:2}.marker.nearest{stroke:#d8a657}.marker.median{stroke:#4db6ac}.marker.farthest{stroke:#e76f51}.marker-label.nearest{fill:#d8a657}.marker-label.median{fill:#4db6ac}.marker-label.farthest{fill:#e76f51}.perspective path{fill:none;stroke:#54737a;stroke-width:1}.coverage-window{fill:#d8a65722;stroke:#d8a657;stroke-width:2}.edge-value{fill:#e7eceb;font-size:12px}.edge-note{fill:#d8a657}.contrast-track{fill:#243f45}.contrast-fill{fill:#e76f51}.legend-near{fill:#d8a657}.legend-median{fill:#4db6ac}.legend-far{fill:#e76f51}.switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.switch button{min-height:44px;border:1px solid #54747b;border-radius:7px;color:#dce8e7;background:#173039;font:750 9px ui-monospace,monospace;cursor:pointer}.switch button[aria-pressed="true"]{border-color:#d8a657;color:#f0ca7c;background:#3b3428}.switch button:focus-visible{outline:3px solid #4db6ac;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#cbdcdd;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#d8a657;font:750 10px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e76f51;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9db0b4;font-size:10px;line-height:1.65}@media(max-width:560px){.distance-tunnel{overflow-x:auto;overscroll-behavior-inline:contain}.distance-tunnel svg{width:560px;max-width:none}}
</style>
