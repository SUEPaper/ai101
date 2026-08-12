<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type PositionMode = 'none' | 'sinusoidal'
const headCount = ref(3)
const activeHead = ref(1)
const positionMode = ref<PositionMode>('sinusoidal')
const positionStrength = ref(.65)
const swapped = ref(false)
const tokens = ['润滑低','振动升','转速降','跳闸','复位','确认']
const features = [[1,.1,.9,0],[.8,.3,.7,0],[.4,.8,.6,0],[.9,.7,1,0],[-.4,.4,-.4,1],[-.3,.2,-.2,1]]
const headNames = ['源设备','时间邻近','处置语义','异常强度']
const headColors = ['#5fd8e8','#ffb454','#d486ff','#ff6577']
const queryToken = 3
const originalOrder = [0,1,2,3,4,5]
const swappedOrder = [0,4,2,3,1,5]

const positional = (position: number) => [Math.sin(position),Math.cos(position),Math.sin(position/3),Math.cos(position/3)]
function softmax(values: number[]) { const maximum=Math.max(...values); const e=values.map((v)=>Math.exp(v-maximum)); const total=e.reduce((a,b)=>a+b,0); return e.map((v)=>v/total) }
function project(vector: number[], head: number, kind: 'q'|'k'|'v') {
  const patterns = [[[1,.1,.2,0],[.1,.2,1,0]],[[.2,1,.1,.1],[1,-.1,.2,0]],[[.1,.2,.5,1],[-.2,.1,1,.6]],[[.3,.1,1,-.2],[1,.2,.1,-.1]]]
  const shift = kind==='q' ? .08 : kind==='k' ? -.05 : .14
  return patterns[head].map((row,rowIndex) => row.reduce((sum,value,index) => sum + (value + shift * (rowIndex===index%2 ? 1 : -1)) * vector[index], 0))
}
function evaluateOrder(order: number[], count: number) {
  const encoded = order.map((tokenIndex,position) => features[tokenIndex].map((value,dimension) => value + (positionMode.value==='sinusoidal' ? positionStrength.value * positional(position)[dimension] : 0)))
  const queryPosition = order.indexOf(queryToken)
  return Array.from({length:count},(_,head) => {
    const queries=encoded.map((vector)=>project(vector,head,'q')); const keys=encoded.map((vector)=>project(vector,head,'k')); const values=encoded.map((vector)=>project(vector,head,'v'))
    const scores=keys.map((key)=>key.reduce((sum,value,index)=>sum+value*queries[queryPosition][index],0)/Math.sqrt(2)); const weights=softmax(scores)
    const output=[0,1].map((dimension)=>weights.reduce((sum,weight,index)=>sum+weight*values[index][dimension],0)); return {weights,output}
  })
}
const result = computed(() => {
  const count=headCount.value; const base=evaluateOrder(originalOrder,count); const changed=evaluateOrder(swappedOrder,count); const shownOrder=swapped.value?swappedOrder:originalOrder; const shown=swapped.value?changed:base; const selected=Math.min(activeHead.value,count-1)
  const baseConcat=base.flatMap((head)=>head.output); const changedConcat=changed.flatMap((head)=>head.output); const permutationDelta=Math.sqrt(baseConcat.reduce((sum,value,index)=>sum+(value-changedConcat[index])**2,0))
  let diversity=0,pairs=0; for(let left=0;left<count;left+=1) for(let right=left+1;right<count;right+=1){diversity+=shown[left].weights.reduce((sum,value,index)=>sum+Math.abs(value-shown[right].weights[index]),0);pairs+=1}
  return {shown,shownOrder,selected,permutationDelta,diversity:pairs?diversity/pairs:0}
})
const cellX=(column:number)=>169+column*54
const rowY=(head:number)=>75+head*47
</script>

<template>
  <MlExperimentFrame index="17-C" title="在多头配线板上检查子空间与位置" question="多头只是重复同一张图吗？交换两个事件时，没有位置编码的自注意力为何跟着排列等变，而加入位置后会改变同一查询的读出？" formula="MHA(X) = Concat(head₁,…,headₕ)Wᴼ">
    <template #visual>
      <figure class="head-board">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`${headCount}个注意力头，位置模式${positionMode}，交换敏感度${result.permutationDelta.toFixed(3)}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="board-bg" /><text x="32" y="37" class="panel-title">多头事件配线板 · 查询固定为“跳闸”</text><rect x="32" y="50" width="536" height="217" rx="7" class="heads-panel" />
          <g v-for="(_,head) in result.shown" :key="`head-${head}`">
            <rect x="45" :y="rowY(head)-15" width="103" height="34" rx="5" :class="['head-label-card',{active:head===result.selected}]" :style="{stroke:headColors[head]}" /><text x="57" :y="rowY(head)-2" class="head-index">H{{ head+1 }}</text><text x="57" :y="rowY(head)+11" class="head-name">{{ headNames[head] }}</text>
            <g v-for="(weight,column) in result.shown[head].weights" :key="`weight-${head}-${column}`"><rect :x="cellX(column)" :y="rowY(head)-15" width="43" height="34" rx="4" class="weight-cell" :style="{fill:headColors[head],fillOpacity:.08+weight*2.8}" /><text :x="cellX(column)+21.5" :y="rowY(head)+1" class="weight-value">{{ weight.toFixed(2) }}</text><text v-if="head===result.shown.length-1" :x="cellX(column)+21.5" :y="rowY(head)+31" :class="['token-name',{moved:result.shownOrder[column]!==originalOrder[column]}]">{{ tokens[result.shownOrder[column]] }}</text></g>
          </g>
          <line x1="45" y1="247" x2="552" y2="247" class="concat-rail" /><text x="45" y="260" class="rail-label">Concat</text><g v-for="head in headCount" :key="`concat-${head}`"><rect :x="121+(head-1)*70" y="238" width="56" height="18" rx="3" :style="{fill:headColors[head-1]}" class="concat-block" /><text :x="149+(head-1)*70" y="250" class="concat-label">2 dims</text></g>
          <rect x="469" y="278" width="99" height="19" rx="4" :class="['position-badge',{active:positionMode==='sinusoidal'}]" /><text x="518.5" y="291" class="position-text">{{ positionMode==='sinusoidal' ? 'POSITION ON' : 'POSITION OFF' }}</text><path v-if="positionMode==='sinusoidal'" d="M45 289 C70 273 92 305 117 289 S164 273 188 289" class="position-wave" /><text x="45" y="307" class="footer-note">{{ swapped ? '位置 2 与 5 已交换；橙色 token 为移动事件' : '原始事件顺序；点击交换后比较同一“跳闸”查询' }}</text>
        </svg>
        <figcaption><span><i class="head" />独立子空间</span><span><i class="active" />当前头</span><span><i class="moved" />移动 token</span><span><i class="position" />位置波形</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <label><span>注意力头数 h <output>{{ headCount }}</output></span><input v-model.number="headCount" type="range" min="1" max="4" step="1"></label><label><span>检查头 <output>H{{ result.selected+1 }} {{ headNames[result.selected] }}</output></span><input v-model.number="activeHead" type="range" min="0" :max="headCount-1" step="1"></label>
      <div class="position-switch" role="group" aria-label="位置编码方式"><button type="button" :aria-pressed="positionMode==='none'" @click="positionMode='none'">不加位置</button><button type="button" :aria-pressed="positionMode==='sinusoidal'" @click="positionMode='sinusoidal'">正弦位置</button></div>
      <label><span>位置强度 <output>{{ positionStrength.toFixed(2) }}</output></span><input v-model.number="positionStrength" type="range" min="0" max="1.5" step=".05" :disabled="positionMode==='none'"></label><button type="button" class="swap-button" @click="swapped=!swapped">{{ swapped ? '恢复原始事件顺序' : '交换位置 2 与 5' }}</button><p>不加位置时，交换 Key/Value 的行只会同步重排权重，对同一个内容查询的加权和保持不变；位置向量进入 Q/K/V 后，这种等变性被有意打破。</p>
    </template>
    <template #metrics><div><span>每头 dₖ=dᵥ</span><strong>2</strong></div><div><span>拼接维度</span><strong>{{ 2*headCount }}</strong></div><div><span>头间平均 L1 差异</span><strong>{{ result.diversity.toFixed(3) }}</strong></div><div><span>交换后的读出变化</span><strong>{{ result.permutationDelta.toExponential(2) }}</strong></div></template>
    <template #note>多头提供多套投影和路由分布，不保证每个头自然对应可命名机理；头间冗余也很常见。该实验的“源设备/时间邻近”等名称是预设投影的教学含义，不应类推为训练后模型的自动解释。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.head-board{margin:0}.head-board svg{display:block;width:100%;border:1px solid #284b59;border-radius:10px;background:#091820}.board-bg{fill:#091820}.head-board text{font:700 7px ui-monospace,monospace}.panel-title{fill:#d9edf2;font-size:10px}.heads-panel{fill:#10232c;stroke:#2c5262}.head-label-card{fill:#152d38;stroke-width:1.4}.head-label-card.active{fill:#283044;stroke-width:2.5}.head-index{fill:#d9edf2;font-size:9px}.head-name{fill:#89a9b4}.weight-cell{stroke:#345765}.weight-value{fill:#eaf6f8;text-anchor:middle}.token-name{fill:#89a9b4;text-anchor:middle;font-size:6px}.token-name.moved{fill:#ffb454}.concat-rail{stroke:#537888}.rail-label{fill:#89a9b4}.concat-block{opacity:.78}.concat-label{fill:#091820;text-anchor:middle}.position-badge{fill:#1a3440;stroke:#537888}.position-badge.active{fill:#30283d;stroke:#d486ff}.position-text{fill:#d9edf2;text-anchor:middle}.position-wave{fill:none;stroke:#d486ff;stroke-width:2}.footer-note{fill:#89a9b4}.head-board figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.head-board figcaption span{display:inline-flex;align-items:center;gap:5px}.head-board figcaption i{width:12px;height:8px;border-radius:2px}.head-board figcaption i.head{background:#5fd8e8}.head-board figcaption i.active{border:2px solid #d486ff}.head-board figcaption i.moved{background:#ffb454}.head-board figcaption i.position{height:2px;background:#d486ff}.position-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.position-switch button,.swap-button{min-height:44px;border:1px solid #496a78;border-radius:7px;color:#d9edf2;background:#112833;font:750 9px ui-monospace,monospace;cursor:pointer}.position-switch button[aria-pressed="true"]{border-color:#ffb454;color:#ffb454;background:#302c2b}.swap-button{border-color:#d486ff;color:#d486ff}.position-switch button:focus-visible,.swap-button:focus-visible{outline:3px solid #d486ff;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#5fd8e8;font:750 10px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#d486ff;cursor:pointer}.ml-experiment__controls input:disabled{opacity:.45;cursor:not-allowed}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.head-board{overflow-x:auto;overscroll-behavior-inline:contain}.head-board svg{width:560px;max-width:none}.head-board figcaption{min-width:560px}}
</style>
