<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const logLambda = ref(0)
const weakSingularValue = ref(0.24)

const singularValues = computed(() => [6.0, 2.4, 0.9, weakSingularValue.value])
const projectedSignals = [3.6, -1.45, 0.42, 0.20]
const labels = ['负载主轴', '冷却主轴', '温升次轴', '冗余弱轴']

const result = computed(() => {
  const lambda = 10 ** logLambda.value
  const directions = singularValues.value.map((singularValue, index) => {
    const retention = singularValue ** 2 / (singularValue ** 2 + lambda)
    const olsCoefficient = projectedSignals[index] / singularValue
    return {
      label: labels[index],
      singularValue,
      retention,
      olsCoefficient,
      ridgeCoefficient: retention * olsCoefficient,
    }
  })
  const largestSquared = singularValues.value[0] ** 2
  const smallestSquared = singularValues.value.at(-1)! ** 2
  return {
    lambda,
    directions,
    effectiveDof: directions.reduce((sum, direction) => sum + direction.retention, 0),
    regularizedCondition: (largestSquared + lambda) / (smallestSquared + lambda),
    olsNorm: Math.sqrt(directions.reduce((sum, direction) => sum + direction.olsCoefficient ** 2, 0)),
    ridgeNorm: Math.sqrt(directions.reduce((sum, direction) => sum + direction.ridgeCoefficient ** 2, 0)),
  }
})

const laneY = (index: number) => 67 + index * 59
const formatCondition = (value: number) => value >= 1000 ? value.toExponential(2) : value.toFixed(1)
</script>

<template>
  <MlExperimentFrame
    index="06-A"
    title="把 Ridge 当作奇异方向滤波器"
    question="同一个惩罚强度，为什么几乎不动主信号，却会关小由冗余传感器形成的弱方向？"
    formula="hⱼ(λ) = sⱼ² / (sⱼ² + λ)"
  >
    <template #visual>
      <figure class="spectral-rack">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`Ridge 惩罚 ${result.lambda.toFixed(3)}，有效自由度 ${result.effectiveDof.toFixed(2)}，弱方向保留率 ${result.directions[3].retention.toFixed(3)}`">
          <rect x="16" y="16" width="568" height="294" rx="9" class="rack-bg" />
          <text x="32" y="39" class="rack-title">电机软测量 · 谱滤波机架</text>
          <text x="224" y="39" class="column-label">奇异强度 sⱼ</text>
          <text x="371" y="39" class="column-label">Ridge 阀门 hⱼ</text>
          <text x="503" y="39" class="column-label">保留输出</text>

          <g v-for="(direction, index) in result.directions" :key="direction.label">
            <rect x="30" :y="laneY(index) - 20" width="535" height="42" rx="7" class="lane" />
            <text x="42" :y="laneY(index) + 4" class="lane-label">{{ direction.label }}</text>
            <line x1="166" x2="282" :y1="laneY(index)" :y2="laneY(index)" class="pipe" />
            <line x1="401" x2="548" :y1="laneY(index)" :y2="laneY(index)" class="pipe muted" />
            <rect x="166" :y="laneY(index) - 7" :width="Math.max(3, 116 * direction.singularValue / 6)" height="14" rx="4" class="signal" />
            <text x="291" :y="laneY(index) + 4" class="value-label">{{ direction.singularValue.toFixed(2) }}</text>
            <circle cx="377" :cy="laneY(index)" r="19" class="valve" />
            <path :d="`M 377 ${laneY(index)} L ${377 + 14 * Math.cos((1 - direction.retention) * Math.PI / 2)} ${laneY(index) - 14 * Math.sin((1 - direction.retention) * Math.PI / 2)}`" class="valve-hand" />
            <rect x="405" :y="laneY(index) - 7" :width="Math.max(2, 140 * direction.retention)" height="14" rx="4" class="retained" />
            <text x="553" :y="laneY(index) + 4" text-anchor="end" class="retention-label">{{ (100 * direction.retention).toFixed(1) }}%</text>
          </g>
          <text x="32" y="298" class="footer-label">λ 对所有方向相同；相对 sⱼ² 越小的方向，阀门关闭得越快。</text>
        </svg>
        <figcaption>
          <span class="legend signal" />设计矩阵提供的方向强度
          <span class="legend retained" />进入拟合值的保留比例
          <span class="legend valve" />谱收缩阀
        </figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>log₁₀ λ <output>{{ logLambda.toFixed(2) }}</output></span>
        <input v-model.number="logLambda" type="range" min="-3" max="3" step="0.05">
      </label>
      <label>
        <span>冗余弱轴 s₄ <output>{{ weakSingularValue.toFixed(2) }}</output></span>
        <input v-model.number="weakSingularValue" type="range" min="0.08" max="0.80" step="0.01">
      </label>
      <p>先把 λ 放在 0.1–1，再降低 s₄。主轴的 sⱼ² 远大于 λ，阀门几乎全开；弱轴的 OLS 系数需要除以很小的 sⱼ，Ridge 会优先切断这条放大噪声的通道。</p>
    </template>
    <template #metrics>
      <div><span>λ</span><strong>{{ result.lambda.toFixed(3) }}</strong></div>
      <div><span>正则化 Gram 条件数</span><strong>{{ formatCondition(result.regularizedCondition) }}</strong></div>
      <div><span>有效自由度</span><strong>{{ result.effectiveDof.toFixed(2) }} / 4</strong></div>
      <div><span>系数范数 OLS → Ridge</span><strong>{{ result.olsNorm.toFixed(2) }} → {{ result.ridgeNorm.toFixed(2) }}</strong></div>
    </template>
    <template #note>
      图中保留率作用于拟合值方向：ŷλ=U diag(hⱼ) Uᵀy。Ridge 不是把每个原始传感器系数乘同一个数，而是先旋转到右奇异向量基底，再按每条方向的数据强度分别收缩；所以“弱方向被压制”不等于“某个原始字段被删除”。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.spectral-rack{margin:0}.spectral-rack svg{display:block;width:100%;border:1px solid #2d3c68;border-radius:10px;background:#10182e}.rack-bg{fill:#10182e}.lane{fill:#141f39;stroke:#2b3c64}.pipe{stroke:#40527e;stroke-width:4}.pipe.muted{stroke:#293858}.signal{fill:#72d5ce}.retained{fill:#f4c95d}.valve{fill:#17203b;stroke:#e15b4b;stroke-width:3}.valve-hand{stroke:#fff;stroke-width:3;stroke-linecap:round}.spectral-rack text{font:700 10px ui-monospace,monospace}.rack-title{fill:#eef2ff;font-size:11px}.column-label,.footer-label{fill:#8391b2}.lane-label{fill:#cbd4e9}.value-label{fill:#72d5ce}.retention-label{fill:#f4c95d}.spectral-rack figcaption{display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:9px;color:#9eabc9;font-size:10px;line-height:1.6}.legend{display:inline-block;width:13px;height:8px;margin-right:4px;vertical-align:middle}.legend.signal{background:#72d5ce}.legend.retained{background:#f4c95d}.legend.valve{width:10px;height:10px;border:2px solid #e15b4b;border-radius:50%}.ml-experiment__controls label{display:grid;gap:5px;color:#c8d1e8;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#76d6cf;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#e15b4b;cursor:pointer}.ml-experiment__controls p{margin:0;color:#9eabc9;font-size:10px;line-height:1.65}
@media(max-width:560px){.spectral-rack{overflow-x:auto;overscroll-behavior-inline:contain}.spectral-rack svg{width:560px;max-width:none}.spectral-rack figcaption{min-width:560px}}
</style>
