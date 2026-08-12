<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type ScaleMode = 'scaled' | 'raw'
type ValueMode = 'cause' | 'urgency'

const queryIndex = ref(4)
const queryGain = ref(2.4)
const scaleMode = ref<ScaleMode>('scaled')
const valueMode = ref<ValueMode>('cause')

const tokens = [
  { label: '振动升高', key: [0.92, 0.08], cause: .82, urgency: .64 },
  { label: '温度偏高', key: [0.48, 0.36], cause: .30, urgency: .72 },
  { label: '润滑压低', key: [1.00, 0.02], cause: 1.00, urgency: .92 },
  { label: '远程复位', key: [-.42, .90], cause: -.52, urgency: -.35 },
  { label: '保护跳闸', key: [.78, .20], cause: .18, urgency: 1.00 },
  { label: '人工确认', key: [-.35, .76], cause: -.28, urgency: -.18 },
]
const queries = [[.82,.10],[.42,.38],[.94,.05],[-.30,.86],[.98,.04],[-.28,.72]]

const result = computed(() => {
  const query = queries[queryIndex.value]
  const divisor = scaleMode.value === 'scaled' ? Math.sqrt(query.length) : 1
  const scores = tokens.map((token) => queryGain.value * token.key.reduce((sum,value,index) => sum + value * query[index], 0) / divisor)
  const maximum = Math.max(...scores)
  const exponentials = scores.map((score) => Math.exp(score - maximum))
  const total = exponentials.reduce((sum,value) => sum + value, 0)
  const weights = exponentials.map((value) => value / total)
  const values = tokens.map((token) => valueMode.value === 'cause' ? token.cause : token.urgency)
  const output = weights.reduce((sum,weight,index) => sum + weight * values[index], 0)
  const entropy = -weights.reduce((sum,weight) => sum + weight * Math.log(weight + 1e-12), 0)
  const maximumIndex = weights.indexOf(Math.max(...weights))
  return { weights, output, entropy, maximumIndex }
})

const tokenX = (index: number) => 64 + index * 94
const lineWidth = (weight: number) => 1 + weight * 12
const barHeight = (weight: number) => 70 * weight
</script>

<template>
  <MlExperimentFrame index="17-A" title="在保护事件总线上分离查询、键和值" question="同一组注意力权重为什么能读取不同信息？点积尺度增大时，内容路由会怎样从混合读取变成近似硬选择？" formula="A = softmax(QKᵀ / √dₖ), Z = AV">
    <template #visual>
      <figure class="routing-console">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`查询${tokens[queryIndex].label}，最关注${tokens[result.maximumIndex].label}，输出${result.output.toFixed(3)}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="console-bg" />
          <text x="32" y="37" class="panel-title">保护事件内容路由 · Q/K 决定地址，V 决定读出</text>
          <rect x="32" y="50" width="536" height="109" rx="7" class="bus-panel" />
          <g v-for="(token,index) in tokens" :key="token.label">
            <line :x1="tokenX(queryIndex)" y1="144" :x2="tokenX(index)" y2="82" :style="{strokeWidth:lineWidth(result.weights[index]),opacity:.22+result.weights[index]*.78}" :class="['route-line',{winner:index===result.maximumIndex}]" />
            <rect :x="tokenX(index)-36" y="61" width="72" height="31" rx="5" :class="['token-card',{query:index===queryIndex,winner:index===result.maximumIndex}]" />
            <text :x="tokenX(index)" y="75" class="token-label">{{ token.label }}</text>
            <text :x="tokenX(index)" y="86" class="token-weight">a={{ result.weights[index].toFixed(2) }}</text>
            <circle :cx="tokenX(index)" cy="144" r="6" :class="['bus-port',{query:index===queryIndex}]" />
          </g>
          <line x1="64" y1="144" x2="534" y2="144" class="event-bus" />
          <text x="44" y="153" class="axis-label">事件总线</text>
          <rect x="32" y="175" width="338" height="117" rx="7" class="weight-panel" />
          <text x="45" y="193" class="section-label">注意力行 · 每列对应一个 Key</text>
          <g v-for="(token,index) in tokens" :key="`bar-${token.label}`">
            <rect :x="51+index*51" y="209" width="31" height="70" rx="3" class="bar-track" />
            <rect :x="51+index*51" :y="279-barHeight(result.weights[index])" width="31" :height="barHeight(result.weights[index])" rx="3" :class="['weight-bar',{winner:index===result.maximumIndex}]" />
            <text :x="66.5+index*51" y="288" class="bar-label">K{{ index+1 }}</text>
          </g>
          <rect x="387" y="175" width="181" height="117" rx="7" class="output-panel" />
          <text x="402" y="194" class="section-label">Value 加权读出</text>
          <text x="402" y="226" class="output-value">{{ result.output.toFixed(3) }}</text>
          <line x1="403" y1="247" x2="550" y2="247" class="output-zero" />
          <circle :cx="476.5+result.output*64" cy="247" r="7" class="output-marker" />
          <text x="402" y="270" class="output-note">{{ valueMode==='cause' ? '根因证据通道' : '处置紧迫度通道' }}</text>
          <text x="402" y="283" class="output-note">权重不随 Value 通道切换</text>
        </svg>
        <figcaption><span><i class="query" />当前 Query</span><span><i class="winner" />最高匹配 Key</span><span><i class="route" />读取强度</span><span><i class="value" />Value 输出</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <label><span>查询事件 <output>{{ tokens[queryIndex].label }}</output></span><input v-model.number="queryIndex" type="range" min="0" max="5" step="1"></label>
      <label><span>查询增益 <output>{{ queryGain.toFixed(1) }}</output></span><input v-model.number="queryGain" type="range" min=".5" max="5" step=".1"></label>
      <div class="mode-switch" role="group" aria-label="点积缩放方式"><button type="button" :aria-pressed="scaleMode==='scaled'" @click="scaleMode='scaled'">除以 √dₖ</button><button type="button" :aria-pressed="scaleMode==='raw'" @click="scaleMode='raw'">不缩放</button></div>
      <div class="mode-switch" role="group" aria-label="Value 数据通道"><button type="button" :aria-pressed="valueMode==='cause'" @click="valueMode='cause'">读取根因证据</button><button type="button" :aria-pressed="valueMode==='urgency'" @click="valueMode='urgency'">读取紧迫度</button></div>
      <p>切换 Value 通道时注意力条形图不变，只改变最终读出；这正是 Key 与 Value 必须分开的原因。增大查询增益或取消缩放会让 Softmax 更尖锐。</p>
    </template>
    <template #metrics>
      <div><span>最高权重事件</span><strong>{{ tokens[result.maximumIndex].label }}</strong></div><div><span>注意力熵</span><strong>{{ result.entropy.toFixed(3) }}</strong></div><div><span>权重行和</span><strong>{{ result.weights.reduce((a,b)=>a+b,0).toFixed(6) }}</strong></div><div><span>Value 读出</span><strong>{{ result.output.toFixed(3) }}</strong></div>
    </template>
    <template #note>该实验把两维点积压成可观察切片。真实模型的 Q/K/V 都是可学习向量；注意力权重表示模型在当前计算中的路由系数，不自动等价于因果贡献或可信解释。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.routing-console{margin:0}.routing-console svg{display:block;width:100%;border:1px solid #284b59;border-radius:10px;background:#091820}.console-bg{fill:#091820}.routing-console text{font:700 8px ui-monospace,monospace}.panel-title{fill:#d9edf2;font-size:10px}.bus-panel,.weight-panel,.output-panel{fill:#10232c;stroke:#2c5262}.event-bus{stroke:#537888;stroke-width:3}.route-line{stroke:#5fd8e8}.route-line.winner{stroke:#ffb454}.token-card{fill:#152d38;stroke:#4d7180}.token-card.query{stroke:#d486ff;stroke-width:2}.token-card.winner{fill:#473326;stroke:#ffb454}.token-label{fill:#d9edf2;text-anchor:middle}.token-weight{fill:#89a9b4;text-anchor:middle}.bus-port{fill:#5fd8e8;stroke:#091820;stroke-width:2}.bus-port.query{fill:#d486ff}.axis-label,.section-label{fill:#89a9b4}.bar-track{fill:#1a3440}.weight-bar{fill:#5fd8e8}.weight-bar.winner{fill:#ffb454}.bar-label{fill:#89a9b4;text-anchor:middle}.output-value{fill:#ffb454;font-size:24px}.output-zero{stroke:#587784}.output-marker{fill:#d486ff;stroke:#091820;stroke-width:2}.output-note{fill:#9ab3bd}.routing-console figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.routing-console figcaption span{display:inline-flex;align-items:center;gap:5px}.routing-console figcaption i{width:12px;height:8px;border-radius:2px}.routing-console figcaption i.query{border:2px solid #d486ff}.routing-console figcaption i.winner{background:#ffb454}.routing-console figcaption i.route{height:2px;background:#5fd8e8}.routing-console figcaption i.value{background:#d486ff}.mode-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.mode-switch button{min-height:44px;border:1px solid #496a78;border-radius:7px;color:#d9edf2;background:#112833;font:750 9px ui-monospace,monospace;cursor:pointer}.mode-switch button[aria-pressed="true"]{border-color:#ffb454;color:#ffb454;background:#302c2b}.mode-switch button:focus-visible{outline:3px solid #d486ff;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#5fd8e8;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#d486ff;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.routing-console{overflow-x:auto;overscroll-behavior-inline:contain}.routing-console svg{width:560px;max-width:none}.routing-console figcaption{min-width:560px}}
</style>
