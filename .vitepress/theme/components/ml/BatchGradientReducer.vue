<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type ReductionMode = 'mean' | 'sum'
type BackwardMode = 'reduce' | 'last'

const batchSize = ref(5)
const weight = ref(.7)
const bias = ref(.2)
const reduction = ref<ReductionMode>('mean')
const backwardMode = ref<BackwardMode>('reduce')

const allX = [-1.4, -.8, -.2, .45, 1.0, 1.45, 1.9, 2.25]
const allTarget = [-.85, -.35, .1, .62, 1.08, 1.48, 1.72, 2.05]

const result = computed(() => {
  const rows = allX.slice(0, batchSize.value).map((x, index) => {
    const target = allTarget[index]
    const prediction = x * weight.value + bias.value
    const residual = prediction - target
    const gradLogit = reduction.value === 'mean' ? residual / batchSize.value : residual
    return {
      x, target, prediction, residual, gradLogit,
      gradWeight: x * gradLogit,
      gradBias: gradLogit,
    }
  })
  const lossScale = reduction.value === 'mean' ? 1 / batchSize.value : 1
  const loss = lossScale * rows.reduce((sum, row) => sum + .5 * row.residual ** 2, 0)
  const correctWeight = rows.reduce((sum, row) => sum + row.gradWeight, 0)
  const correctBias = rows.reduce((sum, row) => sum + row.gradBias, 0)
  const last = rows.at(-1) ?? { gradWeight: 0, gradBias: 0 }
  const displayedWeight = backwardMode.value === 'reduce' ? correctWeight : last.gradWeight
  const displayedBias = backwardMode.value === 'reduce' ? correctBias : last.gradBias
  const gradientNorm = Math.hypot(displayedWeight, displayedBias)
  return { rows, loss, correctWeight, correctBias, displayedWeight, displayedBias, gradientNorm }
})

const rowY = (index: number) => 77 + index * 27
const contributionWidth = (value: number) => Math.min(72, Math.abs(value) * 115)
</script>

<template>
  <MlExperimentFrame
    index="14-B"
    title="把批量广播的偏置梯度收回原形状"
    question="偏置 b 在前向被复制给每个焊点；反向时，每行贡献怎样归约成唯一的 db？mean loss 又在哪里引入 1/B？"
    formula="z=Xw+b,  db=Σᵢ ∂L/∂zᵢ"
  >
    <template #visual>
      <figure class="gradient-reducer">
        <svg viewBox="0 0 600 326" role="img" :aria-label="'批量大小 '+batchSize+'，偏置梯度 '+result.displayedBias.toFixed(4)">
          <defs>
            <marker id="reducer-arrow" refX="8" refY="4" markerWidth="9" markerHeight="8" orient="auto"><path d="M0,0 L9,4 L0,8 Z" class="reducer-arrow" /></marker>
          </defs>
          <rect x="15" y="15" width="570" height="296" rx="9" class="reducer-bg" />
          <text x="36" y="34" class="panel-title">逐样本前向与上游梯度</text>
          <text x="430" y="34" class="panel-title">参数归约总线</text>
          <text x="36" y="58" class="column-head">i</text>
          <text x="72" y="58" class="column-head">xᵢ</text>
          <text x="128" y="58" class="column-head">zᵢ</text>
          <text x="184" y="58" class="column-head">yᵢ</text>
          <text x="241" y="58" class="column-head">gᵢ=∂L/∂zᵢ</text>
          <text x="350" y="58" class="column-head">xᵢgᵢ</text>
          <line x1="26" x2="404" y1="64" y2="64" class="header-line" />

          <g v-for="(row,index) in result.rows" :key="'batch-row-'+index">
            <rect x="26" :y="rowY(index)-16" width="378" height="23" rx="4" :class="['row-shell',{last:backwardMode==='last' && index===result.rows.length-1}]" />
            <text x="36" :y="rowY(index)" class="row-text">{{ index+1 }}</text>
            <text x="72" :y="rowY(index)" class="row-text">{{ row.x.toFixed(2) }}</text>
            <text x="128" :y="rowY(index)" class="value-text">{{ row.prediction.toFixed(2) }}</text>
            <text x="184" :y="rowY(index)" class="row-text">{{ row.target.toFixed(2) }}</text>
            <rect
              :x="row.gradLogit >= 0 ? 278 : 278-contributionWidth(row.gradLogit)"
              :y="rowY(index)-9"
              :width="contributionWidth(row.gradLogit)"
              height="9"
              rx="2"
              :class="row.gradLogit >= 0 ? 'positive-bar' : 'negative-bar'"
            />
            <line x1="278" x2="278" :y1="rowY(index)-12" :y2="rowY(index)+3" class="zero-line" />
            <text x="241" :y="rowY(index)" class="gradient-text">{{ row.gradLogit.toFixed(3) }}</text>
            <text x="350" :y="rowY(index)" class="gradient-text">{{ row.gradWeight.toFixed(3) }}</text>
            <path :d="'M404 '+(rowY(index)-5)+' C425 '+(rowY(index)-5)+' 417 147 442 147'" :class="['fold-line',{ignored:backwardMode==='last' && index!==result.rows.length-1}]" marker-end="url(#reducer-arrow)" />
          </g>

          <g v-for="index in 8-batchSize" :key="'empty-row-'+index">
            <rect x="26" :y="rowY(batchSize+index-1)-16" width="378" height="23" rx="4" class="row-shell empty" />
          </g>

          <rect x="442" y="105" width="116" height="84" rx="8" :class="['reduce-box',{wrong:backwardMode==='last'}]" />
          <text x="500" y="126" class="reduce-title">{{ backwardMode==='reduce' ? 'Σ batch axis' : '只保留末行' }}</text>
          <text x="500" y="148" class="reduce-value">dW={{ result.displayedWeight.toFixed(3) }}</text>
          <text x="500" y="168" class="reduce-value">db={{ result.displayedBias.toFixed(3) }}</text>
          <text x="500" y="181" :class="['reduce-status',{wrong:backwardMode==='last'}]">{{ backwardMode==='reduce' ? 'shape restored' : 'gradient lost' }}</text>
          <path d="M500 189 L500 236" class="update-line" marker-end="url(#reducer-arrow)" />
          <rect x="431" y="236" width="138" height="50" rx="7" class="parameter-box" />
          <text x="500" y="256" class="parameter-label">参数原形状</text>
          <text x="500" y="274" class="parameter-value">w:(1,) · b:(1,)</text>
        </svg>
        <figcaption><span><i class="forward" />前向预测</span><span><i class="positive" />正梯度</span><span><i class="negative" />负梯度</span><span><i class="ignored" />错误丢弃</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="reduction-switch" role="group" aria-label="批量损失归约">
        <button type="button" :aria-pressed="reduction==='mean'" @click="reduction='mean'">平均损失 mean</button>
        <button type="button" :aria-pressed="reduction==='sum'" @click="reduction='sum'">总损失 sum</button>
      </div>
      <div class="backward-switch" role="group" aria-label="广播反向模式">
        <button type="button" :aria-pressed="backwardMode==='reduce'" @click="backwardMode='reduce'">正确沿批维求和</button>
        <button type="button" :aria-pressed="backwardMode==='last'" @click="backwardMode='last'">错误只取末行</button>
      </div>
      <label><span>批量大小 B <output>{{ batchSize }}</output></span><input v-model.number="batchSize" type="range" min="2" max="8" step="1"></label>
      <label><span>线性权重 w <output>{{ weight.toFixed(2) }}</output></span><input v-model.number="weight" type="range" min="-.5" max="1.5" step=".05"></label>
      <label><span>共享偏置 b <output>{{ bias.toFixed(2) }}</output></span><input v-model.number="bias" type="range" min="-1" max="1" step=".05"></label>
      <p>mean 与 sum 都需要沿批维累加；区别是 mean 的每行上游梯度先带 1/B。把广播的逆操作误写成“取一行”，形状可能侥幸正确，数值却依赖样本顺序。</p>
    </template>
    <template #metrics>
      <div><span>批量损失</span><strong>{{ result.loss.toFixed(5) }}</strong></div>
      <div><span>当前 dW</span><strong>{{ result.displayedWeight.toFixed(5) }}</strong></div>
      <div><span>当前 db</span><strong>{{ result.displayedBias.toFixed(5) }}</strong></div>
      <div><span>参数梯度范数</span><strong>{{ result.gradientNorm.toFixed(5) }}</strong></div>
    </template>
    <template #note>
      广播没有创建新的独立偏置参数。所有样本写入的是同一个 b，因此反向必须把这些依赖路径折回该参数。任何自动微分算子都要保证返回梯度与原输入形状一致。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.gradient-reducer{margin:0}.gradient-reducer svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0b1722}.reducer-bg{fill:#0b1722}.gradient-reducer text{font:700 9px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.column-head{fill:#829eaf}.header-line{stroke:#345164}.row-shell{fill:#122433;stroke:#263f51}.row-shell.last{stroke:#f3c85b}.row-shell.empty{fill:#101d29;opacity:.45}.row-text{fill:#9fb5c1;text-anchor:middle}.value-text{fill:#63d0c6;text-anchor:middle}.gradient-text{fill:#f3c85b;text-anchor:middle}.positive-bar{fill:#e36c65}.negative-bar{fill:#63d0c6}.zero-line{stroke:#8fa8b7;stroke-width:1}.fold-line{fill:none;stroke:#f3c85b;stroke-width:1.4;opacity:.65}.fold-line.ignored{stroke:#e35d68;stroke-dasharray:4 4;opacity:.22}.reducer-arrow{fill:#f3c85b}.reduce-box{fill:#17263b;stroke:#63d0c6;stroke-width:1.5}.reduce-box.wrong{stroke:#e35d68}.reduce-title,.reduce-value,.reduce-status,.parameter-label,.parameter-value{text-anchor:middle}.reduce-title{fill:#dbeaf0}.reduce-value{fill:#f3c85b;font-size:10px}.reduce-status{fill:#63d0c6;font-size:8px}.reduce-status.wrong{fill:#e35d68}.update-line{stroke:#f3c85b;stroke-width:1.5}.parameter-box{fill:#111e2e;stroke:#49627a}.parameter-label{fill:#8fa8b7}.parameter-value{fill:#dbeaf0}.gradient-reducer figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.gradient-reducer figcaption span{display:inline-flex;align-items:center;gap:5px}.gradient-reducer figcaption i{width:12px;height:8px;border-radius:2px}.gradient-reducer figcaption i.forward{background:#63d0c6}.gradient-reducer figcaption i.positive{background:#e36c65}.gradient-reducer figcaption i.negative{background:#63d0c6}.gradient-reducer figcaption i.ignored{background:#e35d68;opacity:.35}.reduction-switch,.backward-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.reduction-switch button,.backward-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 9px ui-monospace,monospace;cursor:pointer}.reduction-switch button[aria-pressed="true"],.backward-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.reduction-switch button:focus-visible,.backward-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.gradient-reducer{overflow-x:auto;overscroll-behavior-inline:contain}.gradient-reducer svg{width:560px;max-width:none}.gradient-reducer figcaption{min-width:560px}}
</style>
