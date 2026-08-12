<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const round = ref(16)
const learningRate = ref(.12)
const robust = ref(false)
const outlier = ref(false)
const x = Array.from({ length: 14 }, (_, index) => .04 + index * .071)
const validationX = x.map(value => value + .018)
const signal = (value: number) => .72 * Math.sin(value * Math.PI * 2) + (value > .57 ? .48 : -.12) - .16 * value
const cleanTarget = x.map((value, index) => signal(value) + .08 * Math.sin(index * 2.3))
const validationTarget = validationX.map((value, index) => signal(value) - .055 * Math.cos(index * 1.7))

function fitStump(target: number[]) {
  let best = { sse: Infinity, threshold: .5, left: 0, right: 0 }
  for (let split = 1; split < x.length; split++) {
    const left = target.slice(0, split).reduce((sum, value) => sum + value, 0) / split
    const right = target.slice(split).reduce((sum, value) => sum + value, 0) / (target.length - split)
    const sse = target.reduce((sum, value, index) => sum + (value - (index < split ? left : right)) ** 2, 0)
    if (sse < best.sse) best = { sse, threshold: (x[split - 1] + x[split]) / 2, left, right }
  }
  return best
}

const run = computed(() => {
  const target = cleanTarget.map((value, index) => outlier.value && index === 10 ? value + 1.65 : value)
  const initial = robust.value ? [...target].sort((a, b) => a - b)[Math.floor(target.length / 2)] : target.reduce((sum, value) => sum + value, 0) / target.length
  let prediction = Array(x.length).fill(initial)
  let validationPrediction = Array(validationX.length).fill(initial)
  const states = []
  let bestRound = 1
  let bestValidation = Infinity
  for (let step = 1; step <= 80; step++) {
    const residual = target.map((value, index) => value - prediction[index])
    const pseudo = robust.value ? residual.map(value => Math.max(-.42, Math.min(.42, value))) : residual
    const stump = fitStump(pseudo)
    prediction = prediction.map((value, index) => value + learningRate.value * (x[index] <= stump.threshold ? stump.left : stump.right))
    validationPrediction = validationPrediction.map((value, index) => value + learningRate.value * (validationX[index] <= stump.threshold ? stump.left : stump.right))
    const validationRmse = Math.sqrt(validationTarget.reduce((sum, value, index) => sum + (value - validationPrediction[index]) ** 2, 0) / validationTarget.length)
    if (validationRmse < bestValidation) { bestValidation = validationRmse; bestRound = step }
    states.push({
      step, stump, prediction: [...prediction], residual: target.map((value, index) => value - prediction[index]), target,
      trainRmse: Math.sqrt(target.reduce((sum, value, index) => sum + (value - prediction[index]) ** 2, 0) / target.length),
      validationRmse, bestRound,
    })
  }
  return states
})

const state = computed(() => run.value[round.value - 1])
const mapX = (value: number) => 46 + value * 520
const mapY = (value: number) => 142 - value * 68
const predictionPoints = computed(() => state.value.prediction.map((value, index) => `${mapX(x[index])},${mapY(value)}`).join(' '))
const maxResidual = computed(() => Math.max(...state.value.residual.map(Math.abs)))
</script>

<template>
  <MlExperimentFrame
    index="21-B"
    title="把下一道补涂量对准当前伪残差"
    question="每轮树桩并不直接拟合最终克重，而是拟合当前模型还欠下的方向。学习率、轮数和稳健损失怎样共同改变补涂轨迹？"
    formula="Fₘ(x)=Fₘ₋₁(x)+ηhₘ(x),  hₘ≈−∂ℓ/∂F"
  >
    <template #visual>
      <div class="coater-scroll">
        <svg viewBox="0 0 640 350" role="img" :aria-label="`梯度提升第${round}轮，训练RMSE${state.trainRmse.toFixed(3)}，验证RMSE${state.validationRmse.toFixed(3)}`">
          <rect width="640" height="350" rx="14" class="panel" />
          <text x="24" y="27" class="eyebrow">RESIDUAL COATER · PASS {{ round }}</text>
          <rect x="30" y="48" width="574" height="180" rx="9" class="plot" />
          <line x1="46" y1="142" x2="566" y2="142" class="zero" />
          <line :x1="mapX(state.stump.threshold)" y1="55" :x2="mapX(state.stump.threshold)" y2="218" class="split" />
          <polyline :points="predictionPoints" class="prediction" />
          <g v-for="(value,index) in state.target" :key="index">
            <line :x1="mapX(x[index])" :y1="mapY(state.prediction[index])" :x2="mapX(x[index])" :y2="mapY(value)" class="residual" />
            <circle :cx="mapX(x[index])" :cy="mapY(value)" r="5" class="target" :class="{ outlier:outlier&&index===10 }" />
            <circle :cx="mapX(x[index])" :cy="mapY(state.prediction[index])" r="3" class="fit" />
            <rect :x="mapX(x[index])-5" y="274" width="10" :height="Math.min(48,Math.abs(state.residual[index])*35)" :transform="state.residual[index]>=0 ? `translate(0 ${-Math.min(48,Math.abs(state.residual[index])*35)})` : ''" class="residual-bar" :class="{ negative:state.residual[index]<0 }" />
          </g>
          <line x1="46" y1="274" x2="566" y2="274" class="residual-base" />
          <text x="46" y="328" class="caption">上：目标铜色 / 当前模型青色 / 竖线为剩余误差　下：送往下一轮的残差方向</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>提升轮次 <output>{{ round }}</output></span><input v-model.number="round" aria-label="梯度提升轮次" type="range" min="1" max="80" step="1"></label>
      <label class="control"><span>学习率 η <output>{{ learningRate.toFixed(2) }}</output></span><input v-model.number="learningRate" aria-label="梯度提升学习率" type="range" min="0.03" max="0.5" step="0.01"></label>
      <button type="button" :class="{ active:robust }" :aria-pressed="robust" @click="robust=!robust">{{ robust ? 'Huber：残差已限幅' : '平方损失：原始残差' }}</button>
      <button type="button" :class="{ danger:outlier }" :aria-pressed="outlier" @click="outlier=!outlier">{{ outlier ? '已接入量测尖峰' : '接入量测尖峰' }}</button>
    </template>
    <template #metrics>
      <div><span>训练 RMSE</span><strong>{{ state.trainRmse.toFixed(3) }}</strong></div>
      <div><span>冻结验证 RMSE</span><strong>{{ state.validationRmse.toFixed(3) }}</strong></div>
      <div><span>截至当前最佳轮</span><strong>{{ state.bestRound }}</strong></div>
      <div><span>最大剩余误差</span><strong>{{ maxResidual.toFixed(3) }}</strong></div>
    </template>
    <template #note>平方损失的伪残差就是普通残差；Huber 在大误差区把方向限幅，避免单个量测尖峰支配下一棵树。验证最优轮只负责选择前缀，最终测试批次仍必须冻结。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.coater-scroll{overflow-x:auto}.coater-scroll svg{display:block;min-width:600px;width:100%;height:auto}.panel{fill:#111a22}.plot{fill:#e6e0d5;stroke:#607078}.eyebrow,.caption{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#70d2c7;font-size:10px;letter-spacing:.12em}.zero{stroke:#9a9890;stroke-dasharray:4 4}.split{stroke:#e2aa42;stroke-width:2;stroke-dasharray:6 4}.prediction{fill:none;stroke:#167985;stroke-width:3}.residual{stroke:#cb714a;stroke-width:1.5}.target{fill:#b85e3b;stroke:#fff1df;stroke-width:1.5}.target.outlier{fill:#e6473c;stroke-width:3}.fit{fill:#0a6d76}.residual-base{stroke:#71838d}.residual-bar{fill:#56c6ba}.residual-bar.negative{fill:#c9784e}.caption{fill:#9fb0ba;font-size:8px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebbdc;font-size:11px}.control output{color:#f0c76b;font:750 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#bc6844}button{min-height:44px;padding:9px 12px;border:1px solid #52657a;border-radius:6px;color:#d8e3e8;background:#172634;font-weight:750;cursor:pointer}button.active{border-color:#5bcbbf;color:#dffff9;background:#183b3b}button.danger{border-color:#e56d58;color:#ffe0d8;background:#42231e}button:focus-visible,input:focus-visible{outline:3px solid #6fd5ca;outline-offset:2px}@media(max-width:560px){.coater-scroll{margin-right:-18px;padding-right:18px}.coater-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
