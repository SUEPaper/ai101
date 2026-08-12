<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const commonShift = ref(0)
const temperature = ref(1)
const labelSmoothing = ref(0.08)
const bearingAdjustment = ref(0)

const names = ['合格', '不平衡', '不对中', '轴承异常']
const colors = ['#63d0c6', '#f3c85b', '#f08a55', '#e35d68']
const baseLogits = [2.2, 1.0, 0.3, -0.6]
const targetIndex = 3

const stableSoftmax = (logits: number[]) => {
  const maximum = Math.max(...logits)
  const weights = logits.map((value) => Math.exp(value - maximum))
  const normalizer = weights.reduce((sum, value) => sum + value, 0)
  return weights.map((value) => value / normalizer)
}

const result = computed(() => {
  const adjusted = baseLogits.map((value, index) =>
    value + (index === 3 ? bearingAdjustment.value : 0)
  )
  const scaled = adjusted.map((value) => (value + commonShift.value) / temperature.value)
  const centered = scaled.map((value) => value - Math.max(...scaled))
  const probability = stableSoftmax(scaled)
  const withoutShift = stableSoftmax(adjusted.map((value) => value / temperature.value))
  const naiveWeights = scaled.map((value) => Math.exp(value))
  const naiveNormalizer = naiveWeights.reduce((sum, value) => sum + value, 0)
  const naiveStatus = !Number.isFinite(naiveNormalizer)
    ? '指数溢出'
    : naiveNormalizer === 0
      ? '指数全下溢'
      : '本次仍可计算'
  const target = names.map((_, index) =>
    (index === targetIndex ? 1 - labelSmoothing.value : 0)
    + labelSmoothing.value / names.length
  )
  const loss = -target.reduce(
    (sum, value, index) => sum + value * Math.log(probability[index]),
    0,
  )
  const entropy = -probability.reduce(
    (sum, value) => sum + value * Math.log(value),
    0,
  )
  const jacobianNull = probability.map((rowProbability, row) => {
    let sum = 0
    for (let column = 0; column < probability.length; column++) {
      sum += rowProbability * ((row === column ? 1 : 0) - probability[column])
    }
    return sum
  })
  return {
    adjusted,
    scaled,
    centered,
    probability,
    target,
    loss,
    entropy,
    naiveStatus,
    drift: Math.max(...probability.map((value, index) => Math.abs(value - withoutShift[index]))),
    jacobianNullNorm: Math.sqrt(jacobianNull.reduce((sum, value) => sum + value * value, 0)),
    winner: names[probability.indexOf(Math.max(...probability))],
  }
})

</script>

<template>
  <MlExperimentFrame
    index="10-B"
    title="升降全部 logits，观察概率仪表拒绝共模漂移"
    question="为什么共同平移会让朴素指数溢出，却既不改变稳定 Softmax，也暴露出参数的零空间？"
    formula="softmax(z+c·1)=softmax(z),  J_softmax·1=0"
  >
    <template #visual>
      <figure class="reference-gauge">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`共模平移 ${commonShift}，稳定概率漂移 ${result.drift.toExponential(1)}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="gauge-bg" />
          <text x="38" y="31" class="panel-title">共模升降台 / 稳定中心化</text>
          <text x="402" y="31" class="panel-title">数值与对称性审计</text>
          <text x="38" y="53" class="column-label">类别</text>
          <text x="139" y="53" class="column-label">原始+共模</text>
          <text x="234" y="53" class="column-label">减最大值</text>
          <text x="333" y="53" class="column-label">概率柱</text>
          <g v-for="(name,index) in names" :key="name">
            <text x="38" :y="82 + index * 43" class="class-label">{{ name }}</text>
            <text x="176" :y="82 + index * 43" class="raw-value">{{ result.scaled[index].toFixed(1) }}</text>
            <line x1="194" x2="289" :y1="77 + index * 43" :y2="77 + index * 43" class="center-track" />
            <circle :cx="289 + Math.max(-1, result.centered[index] / 8) * 82" :cy="77 + index * 43" r="6" :fill="colors[index]" class="center-dot" />
            <text x="296" :y="82 + index * 43" class="center-value">{{ result.centered[index].toFixed(2) }}</text>
            <rect x="326" :y="61 + index * 43" width="50" height="25" rx="4" class="pillar-track" />
            <rect x="326" :y="86 + index * 43 - 25 * result.probability[index]" width="50" :height="25 * result.probability[index]" rx="3" :fill="colors[index]" />
            <text x="383" :y="82 + index * 43" class="probability-value">{{ (result.probability[index] * 100).toFixed(1) }}%</text>
          </g>
          <line x1="38" x2="383" y1="247" y2="247" class="rule" />
          <text x="38" y="267" class="column-label">平滑目标</text>
          <rect x="108" y="255" width="275" height="18" rx="4" class="target-track" />
          <rect
            v-for="(value,index) in result.target"
            :key="`target-${index}`"
            :x="108 + 275 * result.target.slice(0,index).reduce((sum,item) => sum + item,0)"
            y="255"
            :width="275 * value + .3"
            height="18"
            :fill="colors[index]"
          />
          <text x="38" y="299" class="foot-label">真实标签：轴承异常 · ε={{ labelSmoothing.toFixed(2) }}</text>

          <text x="418" y="64" class="audit-label">共模平移 c</text>
          <text x="563" y="64" class="audit-value">{{ commonShift }}</text>
          <text x="418" y="97" class="audit-label">朴素 exp</text>
          <text x="563" y="97" :class="['audit-value', result.naiveStatus === '本次仍可计算' ? 'safe' : 'danger']">{{ result.naiveStatus }}</text>
          <text x="418" y="130" class="audit-label">稳定概率漂移</text>
          <text x="563" y="130" class="audit-value safe">{{ result.drift.toExponential(1) }}</text>
          <text x="418" y="163" class="audit-label">‖J·1‖₂</text>
          <text x="563" y="163" class="audit-value safe">{{ result.jacobianNullNorm.toExponential(1) }}</text>
          <line x1="418" x2="563" y1="184" y2="184" class="rule" />
          <text x="418" y="215" class="audit-label">交叉熵</text>
          <text x="563" y="215" class="audit-value">{{ result.loss.toFixed(3) }}</text>
          <text x="418" y="248" class="audit-label">预测熵</text>
          <text x="563" y="248" class="audit-value">{{ result.entropy.toFixed(3) }}</text>
          <text x="418" y="281" class="audit-label">赢家</text>
          <text x="563" y="281" class="audit-value winner">{{ result.winner }}</text>
        </svg>
        <figcaption>
          <span><i class="raw" />原始/温度缩放 logit</span>
          <span><i class="centered" />减最大值后的稳定坐标</span>
          <span><i class="target" />label smoothing 目标</span>
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>共同平移 c <output>{{ commonShift }}</output></span>
        <input v-model.number="commonShift" type="range" min="-1000" max="1000" step="25">
      </label>
      <label>
        <span>温度 T <output>{{ temperature.toFixed(2) }}</output></span>
        <input v-model.number="temperature" type="range" min="0.25" max="4" step="0.05">
      </label>
      <label>
        <span>label smoothing ε <output>{{ labelSmoothing.toFixed(2) }}</output></span>
        <input v-model.number="labelSmoothing" type="range" min="0" max="0.30" step="0.01">
      </label>
      <label>
        <span>轴承证据修正 <output>{{ bearingAdjustment.toFixed(2) }}</output></span>
        <input v-model.number="bearingAdjustment" type="range" min="-2" max="4" step="0.05">
      </label>
      <p>共同平移只改变坐标原点；温度改变相对差值尺度；轴承修正改变类别竞争；label smoothing 只改变训练目标和 loss。四个旋钮作用在不同层。</p>
    </template>
    <template #metrics>
      <div><span>稳定概率漂移</span><strong>{{ result.drift.toExponential(1) }}</strong></div>
      <div><span>Jacobian 零方向</span><strong>{{ result.jacobianNullNorm.toExponential(1) }}</strong></div>
      <div><span>稳定概率和</span><strong>{{ result.probability.reduce((sum,value) => sum + value,0).toFixed(6) }}</strong></div>
      <div><span>朴素计算状态</span><strong>{{ result.naiveStatus }}</strong></div>
    </template>
    <template #note>
      减去最大 logit 是精确的坐标变换，不是概率近似。共同平移不变性意味着完整 C 类参数存在零空间；参考类或和为零约束用于固定坐标，不能被解释成某一类“没有效应”。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.reference-gauge{margin:0}.reference-gauge svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0c1925}.gauge-bg{fill:#0c1925}.reference-gauge text{font:700 10px ui-monospace,monospace}.panel-title{fill:#dbeaf0}.column-label,.audit-label,.foot-label{fill:#829eaf}.class-label{fill:#c5d7df}.raw-value{fill:#dbeaf0;text-anchor:end;font-size:12px}.center-track{stroke:#324d60;stroke-width:2}.center-dot{stroke:#fff;stroke-width:1.5}.center-value{fill:#8fa8b7}.pillar-track,.target-track{fill:#1d3040}.probability-value{fill:#dbeaf0;text-anchor:end}.rule{stroke:#355165}.audit-value{fill:#dbeaf0;text-anchor:end;font-size:11px}.audit-value.safe{fill:#63d0c6}.audit-value.danger{fill:#e35d68}.audit-value.winner{fill:#f3c85b}.reference-gauge figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.reference-gauge figcaption span{display:inline-flex;align-items:center;gap:5px}.reference-gauge figcaption i{width:14px;height:3px}.reference-gauge figcaption i.raw{background:#dbeaf0}.reference-gauge figcaption i.centered{background:#63d0c6}.reference-gauge figcaption i.target{background:#e35d68}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.reference-gauge{overflow-x:auto;overscroll-behavior-inline:contain}.reference-gauge svg{width:560px;max-width:none}.reference-gauge figcaption{min-width:560px}}
</style>
