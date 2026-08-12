<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const passLogit = ref(1.4)
const unbalanceLogit = ref(0.7)
const misalignmentLogit = ref(-0.2)
const bearingLogit = ref(0.9)

const classes = [
  { name: '合格', key: 'pass', color: '#63d0c6' },
  { name: '不平衡', key: 'unbalance', color: '#f3c85b' },
  { name: '不对中', key: 'misalignment', color: '#f08a55' },
  { name: '轴承异常', key: 'bearing', color: '#e35d68' },
]

const result = computed(() => {
  const logits = [
    passLogit.value,
    unbalanceLogit.value,
    misalignmentLogit.value,
    bearingLogit.value,
  ]
  const maximum = Math.max(...logits)
  const weights = logits.map((value) => Math.exp(value - maximum))
  const normalizer = weights.reduce((sum, value) => sum + value, 0)
  const probabilities = weights.map((value) => value / normalizer)
  const order = probabilities
    .map((probability, index) => ({ probability, index }))
    .sort((left, right) => right.probability - left.probability)
  const cumulative = probabilities.reduce<number[]>((values, probability) => {
    values.push((values.at(-1) ?? 0) + probability)
    return values
  }, [])
  const entropy = -probabilities.reduce(
    (sum, probability) => sum + probability * Math.log(probability),
    0,
  )
  return {
    logits,
    weights,
    probabilities,
    cumulative,
    winner: classes[order[0].index].name,
    runnerUp: classes[order[1].index].name,
    logitMargin: logits[order[0].index] - logits[order[1].index],
    entropy,
    probabilitySum: probabilities.reduce((sum, value) => sum + value, 0),
  }
})

const logitX = (value: number) => 153 + (value + 4) / 8 * 188
const busStart = (index: number) => index === 0 ? 0 : result.value.cumulative[index - 1]
</script>

<template>
  <MlExperimentFrame
    index="10-A"
    title="把四路故障证据接入同一条概率歧管"
    question="为什么抬高一类 logit 会同时压低其余三类概率，而总概率质量始终守恒？"
    formula="pₖ = exp(zₖ−m) / Σⱼ exp(zⱼ−m)"
  >
    <template #visual>
      <figure class="competition-console">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`当前预测 ${result.winner}，概率和 ${result.probabilitySum.toFixed(6)}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="console-bg" />
          <text x="38" y="32" class="panel-title">EOL 主导故障证据 / logit 输入</text>
          <text x="412" y="32" class="panel-title">归一化概率</text>
          <line x1="153" x2="341" y1="44" y2="44" class="logit-axis" />
          <text x="153" y="40" class="axis-label">−4</text>
          <text x="247" y="40" class="axis-label center">0</text>
          <text x="341" y="40" class="axis-label end">+4</text>
          <g v-for="(item,index) in classes" :key="item.key">
            <text x="38" :y="74 + index * 46" class="class-label">{{ item.name }}</text>
            <line x1="153" x2="341" :y1="69 + index * 46" :y2="69 + index * 46" class="channel-track" />
            <line x1="247" :x2="logitX(result.logits[index])" :y1="69 + index * 46" :y2="69 + index * 46" :stroke="item.color" class="logit-fill" />
            <circle :cx="logitX(result.logits[index])" :cy="69 + index * 46" r="7" :fill="item.color" class="logit-knob" />
            <text x="361" :y="74 + index * 46" class="logit-value">{{ result.logits[index].toFixed(2) }}</text>
            <path :d="`M 384 ${69 + index * 46} C 400 ${69 + index * 46}, 397 ${69 + index * 46}, 412 ${69 + index * 46}`" :stroke="item.color" class="manifold-link" />
            <rect x="412" :y="58 + index * 46" width="145" height="22" rx="4" class="probability-track" />
            <rect x="412" :y="58 + index * 46" :width="145 * result.probabilities[index]" height="22" rx="4" :fill="item.color" />
            <text x="548" :y="74 + index * 46" class="probability-value">{{ (result.probabilities[index] * 100).toFixed(1) }}%</text>
            <text x="412" :y="89 + index * 46" class="weight-label">相对指数权 {{ result.weights[index].toFixed(3) }}</text>
          </g>
          <text x="38" y="257" class="panel-title">100% 概率母线</text>
          <rect x="38" y="269" width="519" height="24" rx="5" class="bus-track" />
          <rect
            v-for="(item,index) in classes"
            :key="`bus-${item.key}`"
            :x="38 + 519 * busStart(index)"
            y="269"
            :width="519 * result.probabilities[index] + .4"
            height="24"
            :fill="item.color"
          />
          <text x="38" y="310" class="bus-label">Σp = {{ result.probabilitySum.toFixed(6) }}</text>
          <text x="557" y="310" class="bus-label end">赢家 {{ result.winner }}</text>
        </svg>
        <figcaption>
          <span v-for="item in classes" :key="`legend-${item.key}`"><i :style="{ background: item.color }" />{{ item.name }}</span>
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>合格 logit <output>{{ passLogit.toFixed(2) }}</output></span>
        <input v-model.number="passLogit" type="range" min="-4" max="4" step="0.05">
      </label>
      <label>
        <span>不平衡 logit <output>{{ unbalanceLogit.toFixed(2) }}</output></span>
        <input v-model.number="unbalanceLogit" type="range" min="-4" max="4" step="0.05">
      </label>
      <label>
        <span>不对中 logit <output>{{ misalignmentLogit.toFixed(2) }}</output></span>
        <input v-model.number="misalignmentLogit" type="range" min="-4" max="4" step="0.05">
      </label>
      <label>
        <span>轴承异常 logit <output>{{ bearingLogit.toFixed(2) }}</output></span>
        <input v-model.number="bearingLogit" type="range" min="-4" max="4" step="0.05">
      </label>
      <p>Softmax 比较相对证据。抬高任一通道会增加它的指数权重，同时通过共同分母压低所有竞争类别；母线总量始终为 100%。</p>
    </template>
    <template #metrics>
      <div><span>当前主导类别</span><strong>{{ result.winner }}</strong></div>
      <div><span>第二类别</span><strong>{{ result.runnerUp }}</strong></div>
      <div><span>前二 logit 间隔</span><strong>{{ result.logitMargin.toFixed(3) }}</strong></div>
      <div><span>预测熵</span><strong>{{ result.entropy.toFixed(3) }} nat</strong></div>
    </template>
    <template #note>
      四类标签表示“专家裁决的主导处置原因”，不是宣称故障在物理上不能共存。若业务需要同时标记不对中和轴承异常，应改为多标签模型，而不是强迫 Softmax 分掉同一份概率质量。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.competition-console{margin:0}.competition-console svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0c1925}.console-bg{fill:#0c1925}.competition-console text{font:700 10px ui-monospace,monospace}.panel-title{fill:#dbeaf0}.axis-label,.weight-label,.bus-label{fill:#829eaf;font-size:9px}.axis-label.center{text-anchor:middle}.axis-label.end,.bus-label.end{text-anchor:end}.class-label{fill:#c5d7df}.logit-axis,.channel-track{stroke:#385166;stroke-width:2}.logit-fill{stroke-width:5;stroke-linecap:round}.logit-knob{stroke:#fff;stroke-width:1.8}.logit-value{fill:#dbeaf0;text-anchor:end;font-size:12px}.manifold-link{fill:none;stroke-width:1.5;opacity:.65}.probability-track,.bus-track{fill:#1d3040}.probability-value{fill:#dbeaf0;text-anchor:end;font-size:10px}.bus-label{font-size:10px}.competition-console figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.competition-console figcaption span{display:inline-flex;align-items:center;gap:5px}.competition-console figcaption i{width:12px;height:8px;border-radius:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.competition-console{overflow-x:auto;overscroll-behavior-inline:contain}.competition-console svg{width:560px;max-width:none}.competition-console figcaption{min-width:560px}}
</style>
