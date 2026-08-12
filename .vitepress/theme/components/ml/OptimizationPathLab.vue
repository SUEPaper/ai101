<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Optimizer = 'gd' | 'momentum' | 'newton'

interface Point { x: number; y: number }

const optimizer = ref<Optimizer>('gd')
const condition = ref(18)
const learningRateRatio = ref(.72)
const steps = ref(18)
const beta = .8
const start: Point = { x: 1.7, y: 1.35 }

const optimizerNames: Record<Optimizer, string> = {
  gd: '梯度下降',
  momentum: 'Momentum',
  newton: 'Newton'
}

const learningRate = computed(() => 2 * learningRateRatio.value / condition.value)
const stableLimit = computed(() => 2 / condition.value)
const toSvgX = (value: number) => 300 + value * 118
const toSvgY = (value: number) => 164 - value * 91

const result = computed(() => {
  const path: Point[] = [{ ...start }]
  let current = { ...start }
  let velocity = { x: 0, y: 0 }
  let diverged = false

  for (let index = 0; index < steps.value; index += 1) {
    const gradient = { x: condition.value * current.x, y: current.y }
    if (optimizer.value === 'gd') {
      current = {
        x: current.x - learningRate.value * gradient.x,
        y: current.y - learningRate.value * gradient.y
      }
    } else if (optimizer.value === 'momentum') {
      velocity = {
        x: beta * velocity.x + (1 - beta) * gradient.x,
        y: beta * velocity.y + (1 - beta) * gradient.y
      }
      current = {
        x: current.x - learningRate.value * velocity.x,
        y: current.y - learningRate.value * velocity.y
      }
    } else {
      current = { x: 0, y: 0 }
    }

    if (!Number.isFinite(current.x) || !Number.isFinite(current.y) || Math.abs(current.x) > 2.35 || Math.abs(current.y) > 1.78) {
      diverged = true
      current = {
        x: Math.max(-2.35, Math.min(2.35, current.x)),
        y: Math.max(-1.78, Math.min(1.78, current.y))
      }
      path.push(current)
      break
    }
    path.push({ ...current })
  }

  const loss = .5 * (condition.value * current.x ** 2 + current.y ** 2)
  const initialLoss = .5 * (condition.value * start.x ** 2 + start.y ** 2)
  return {
    path,
    current,
    loss,
    initialLoss,
    reduction: loss / initialLoss,
    diverged
  }
})

const polyline = computed(() => result.value.path.map((point) => `${toSvgX(point.x)},${toSvgY(point.y)}`).join(' '))
const contours = computed(() => [1, 3, 7, 15, 30].map((level) => ({
  level,
  rx: Math.sqrt(2 * level / condition.value) * 118,
  ry: Math.sqrt(2 * level) * 91
})))
const percent = (value: number) => `${(value * 100).toFixed(value < .01 ? 2 : 1)}%`
</script>

<template>
  <MlExperimentFrame
    index="02-A"
    title="曲率怎样改变优化路径"
    question="同一个学习率要同时应付陡峭方向和平缓方向时，GD、Momentum 与 Newton 会走出怎样不同的路径？"
    formula="J(x,y) = 1/2 (k x^2 + y^2)"
  >
    <template #visual>
      <figure class="path-map">
        <svg viewBox="0 0 600 328" role="img" :aria-label="`${optimizerNames[optimizer]} 在条件数 ${condition} 的二次目标上迭代 ${result.path.length - 1} 步；最终损失 ${result.loss.toExponential(2)}`">
          <defs>
            <clipPath id="optimization-plot"><rect x="20" y="14" width="560" height="300" rx="8" /></clipPath>
          </defs>
          <g clip-path="url(#optimization-plot)">
            <rect class="plot-bg" x="20" y="14" width="560" height="300" rx="8" />
            <line class="axis" x1="20" x2="580" y1="164" y2="164" />
            <line class="axis" x1="300" x2="300" y1="14" y2="314" />
            <ellipse v-for="contour in contours" :key="contour.level" class="contour" cx="300" cy="164" :rx="contour.rx" :ry="contour.ry" />
            <polyline class="path" :points="polyline" />
            <circle v-for="(point, index) in result.path" :key="index" :cx="toSvgX(point.x)" :cy="toSvgY(point.y)" :r="index === 0 ? 6 : 3.5" :class="index === 0 ? 'start' : 'step'" />
            <circle class="optimum" cx="300" cy="164" r="7" />
          </g>
          <text x="30" y="34">陡峭方向 x</text>
          <text x="475" y="304">平缓方向 y</text>
          <text x="308" y="157">最优点</text>
        </svg>
        <figcaption><span class="legend path" />迭代路径　<span class="legend optimum" />最优点　等高线越狭长，条件数越大</figcaption>
      </figure>
      <table>
        <thead><tr><th>量</th><th>当前值</th><th>判读</th></tr></thead>
        <tbody>
          <tr><td>学习率 η</td><td>{{ learningRate.toFixed(4) }}</td><td>GD 稳定上界 {{ stableLimit.toFixed(4) }}</td></tr>
          <tr><td>最终参数</td><td>({{ result.current.x.toFixed(3) }}, {{ result.current.y.toFixed(3) }})</td><td>{{ result.diverged ? '轨迹已越界' : '仍在绘图区内' }}</td></tr>
        </tbody>
      </table>
    </template>
    <template #controls>
      <fieldset>
        <legend>优化器</legend>
        <div class="optimizer-switch">
          <button v-for="name in (Object.keys(optimizerNames) as Optimizer[])" :key="name" type="button" :class="{ active: optimizer === name }" :aria-pressed="optimizer === name" @click="optimizer = name">{{ optimizerNames[name] }}</button>
        </div>
      </fieldset>
      <label>
        <span>条件数 k <output>{{ condition }}</output></span>
        <input v-model.number="condition" type="range" min="2" max="40" step="1">
      </label>
      <label>
        <span>η / 稳定上界 <output>{{ learningRateRatio.toFixed(2) }}</output></span>
        <input v-model.number="learningRateRatio" type="range" min=".10" max="1.20" step=".05" :disabled="optimizer === 'newton'">
      </label>
      <label>
        <span>最多迭代步 <output>{{ steps }}</output></span>
        <input v-model.number="steps" type="range" min="1" max="40" step="1">
      </label>
    </template>
    <template #metrics>
      <div><span>优化器</span><strong>{{ optimizerNames[optimizer] }}</strong></div>
      <div><span>条件数 κ(H)</span><strong>{{ condition }}</strong></div>
      <div><span>实际迭代</span><strong>{{ result.path.length - 1 }}</strong></div>
      <div><span>剩余损失比例</span><strong>{{ result.diverged ? '越界' : percent(result.reduction) }}</strong></div>
    </template>
    <template #note>
      把条件数调大，再比较三种路径。GD 的学习率受最陡方向限制，却要在平缓方向上慢慢移动；Momentum 用历史方向减少之字形；Newton 直接用 Hessian 校正两个方向的尺度。这里是精确二次目标，不能据此推断 Newton 在一般非凸模型中总能一步收敛。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.path-map{margin:0}.path-map svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:9px;background:#10182e}.plot-bg{fill:#10182e}.axis{stroke:#344675;stroke-width:1}.contour{fill:none;stroke:#405386;stroke-width:1.2}.path{fill:none;stroke:#f4c95d;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}.start{fill:#f06b5b;stroke:#fff;stroke-width:2}.step{fill:#f4c95d}.optimum{fill:#72d5ce;stroke:#eafffc;stroke-width:2}.path-map text{fill:#9eabc9;font:10px ui-monospace,monospace}.path-map figcaption{margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}.legend.path{background:#f4c95d}.legend.optimum{height:9px;border-radius:50%;background:#72d5ce}.ml-experiment__visual table{width:100%;margin:10px 0 0;border-collapse:collapse;color:#c8d1e8;background:#10182e;font-size:10px}.ml-experiment__visual th,.ml-experiment__visual td{padding:7px;border:1px solid #2d3c68;text-align:left}.ml-experiment__controls fieldset{min-width:0;margin:0;padding:0;border:0}.ml-experiment__controls legend,.ml-experiment__controls label{color:#c8d1e8;font-size:11px}.optimizer-switch{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:7px}.optimizer-switch button{min-height:44px;padding:6px;border:1px solid #4d6095;border-radius:7px;color:#c8d1e8;background:#111a31;font-size:10px;font-weight:750;cursor:pointer}.optimizer-switch button.active{border-color:#f4c95d;color:#10182e;background:#f4c95d}.optimizer-switch button:focus-visible{outline:3px solid #72d5ce;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls input:disabled{cursor:not-allowed;opacity:.45}@media(max-width:520px){.optimizer-switch{grid-template-columns:1fr}.path-map text{font-size:9px}}
</style>
