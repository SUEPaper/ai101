<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Point = {
  x: number
  y: number
  label: -1 | 1
  id: string
  suspicious?: boolean
}

const logC = ref(0)
const suspiciousX = ref(-1.15)
const secondFeatureScale = ref(1)

const basePoints: Point[] = [
  { x: -1.9, y: -0.8, label: -1, id: 'N1' },
  { x: -1.65, y: 0.2, label: -1, id: 'N2' },
  { x: -1.35, y: -0.45, label: -1, id: 'N3' },
  { x: -1.0, y: 0.75, label: -1, id: 'N4' },
  { x: -0.7, y: 0.05, label: -1, id: 'N5' },
  { x: 0.65, y: -0.55, label: 1, id: 'P1' },
  { x: 0.95, y: 0.5, label: 1, id: 'P2' },
  { x: 1.35, y: -0.2, label: 1, id: 'P3' },
  { x: 1.75, y: 0.75, label: 1, id: 'P4' },
  { x: 2.0, y: -0.65, label: 1, id: 'P5' },
]

const result = computed(() => {
  const c = 10 ** logC.value
  const points: Point[] = [
    ...basePoints,
    { x: suspiciousX.value, y: 0.62, label: 1, id: 'Q?', suspicious: true },
  ].map((point) => ({ ...point, y: point.y * secondFeatureScale.value }))
  const lambda = 1 / (points.length * c)
  let weightX = 0.6
  let weightY = 0
  let intercept = 0
  let best = { objective: Number.POSITIVE_INFINITY, weightX, weightY, intercept }

  for (let iteration = 1; iteration <= 6500; iteration++) {
    let gradientX = lambda * weightX
    let gradientY = lambda * weightY
    let gradientB = 0
    let hingeSum = 0
    for (const point of points) {
      const margin = point.label * (weightX * point.x + weightY * point.y + intercept)
      hingeSum += Math.max(0, 1 - margin)
      if (margin < 1) {
        gradientX -= point.label * point.x / points.length
        gradientY -= point.label * point.y / points.length
        gradientB -= point.label / points.length
      }
    }
    const objective = .5 * lambda * (weightX ** 2 + weightY ** 2) + hingeSum / points.length
    if (objective < best.objective) best = { objective, weightX, weightY, intercept }
    const rate = 0.16 / Math.sqrt(iteration)
    weightX -= rate * gradientX
    weightY -= rate * gradientY
    intercept -= rate * gradientB
  }

  weightX = best.weightX
  weightY = best.weightY
  intercept = best.intercept
  const norm = Math.hypot(weightX, weightY)
  const diagnostics = points.map((point) => {
    const score = weightX * point.x + weightY * point.y + intercept
    const functionalMargin = point.label * score
    const slack = Math.max(0, 1 - functionalMargin)
    return {
      ...point,
      score,
      functionalMargin,
      geometricMargin: functionalMargin / norm,
      slack,
      support: functionalMargin <= 1.025,
      correct: functionalMargin > 0,
    }
  })
  return {
    c,
    lambda,
    weightX,
    weightY,
    intercept,
    norm,
    diagnostics,
    objective: best.objective,
    halfMargin: 1 / norm,
    slackSum: diagnostics.reduce((sum, point) => sum + point.slack, 0),
    supportCount: diagnostics.filter((point) => point.support).length,
    accuracy: diagnostics.filter((point) => point.correct).length / diagnostics.length,
    suspicious: diagnostics.find((point) => point.suspicious)!,
  }
})

const mapX = (value: number) => 43 + (value + 2.6) / 5.2 * 350
const mapY = (value: number) => 286 - (value + 2.05) / 4.1 * 242

const marginPath = (level: number) => {
  const { weightX, weightY, intercept, norm } = result.value
  if (norm < 1e-9) return ''
  const shiftedIntercept = intercept - level
  const centerX = -shiftedIntercept * weightX / (norm * norm)
  const centerY = -shiftedIntercept * weightY / (norm * norm)
  const tangentX = -weightY / norm
  const tangentY = weightX / norm
  const length = 6
  return `M ${mapX(centerX - length * tangentX)} ${mapY(centerY - length * tangentY)} L ${mapX(centerX + length * tangentX)} ${mapY(centerY + length * tangentY)}`
}
</script>

<template>
  <MlExperimentFrame
    index="11-B"
    title="用软间隔量爪夹住真正决定边界的样本"
    question="C、疑似错标位置和特征尺度如何共同改变安全间隔、支持向量与违约账本？"
    formula="min ½‖w‖² + CΣᵢ max(0,1−yᵢfᵢ)"
  >
    <template #visual>
      <figure class="margin-caliper">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`C=${result.c.toFixed(2)}，支持向量 ${result.supportCount} 个，准确率 ${(result.accuracy * 100).toFixed(1)}%`">
          <defs>
            <clipPath id="svm-plot-clip"><rect x="43" y="44" width="350" height="242" rx="5" /></clipPath>
          </defs>
          <rect x="15" y="15" width="570" height="296" rx="9" class="caliper-bg" />
          <text x="43" y="31" class="panel-title">软间隔量爪 / 标准化特征空间</text>
          <text x="420" y="31" class="panel-title">违约与接触点账本</text>
          <rect x="43" y="44" width="350" height="242" rx="5" class="plot-bg" />
          <g class="grid">
            <line v-for="value in [-2,-1,0,1,2]" :key="`gx-${value}`" :x1="mapX(value)" :x2="mapX(value)" y1="44" y2="286" />
            <line v-for="value in [-1,0,1]" :key="`gy-${value}`" x1="43" x2="393" :y1="mapY(value)" :y2="mapY(value)" />
          </g>
          <g clip-path="url(#svm-plot-clip)">
            <path :d="marginPath(-1)" class="margin-jaw negative" />
            <path :d="marginPath(1)" class="margin-jaw positive" />
            <path :d="marginPath(0)" class="decision-line" />
            <g v-for="point in result.diagnostics" :key="point.id">
              <circle
                v-if="point.support"
                :cx="mapX(point.x)"
                :cy="mapY(point.y)"
                r="11"
                :class="['support-ring', point.slack > 1e-3 ? 'is-violating' : '']"
              />
              <circle
                :cx="mapX(point.x)"
                :cy="mapY(point.y)"
                r="6.5"
                :class="point.label === 1 ? 'defect-point' : 'pass-point'"
              />
              <text v-if="point.suspicious" :x="mapX(point.x)+10" :y="mapY(point.y)-8" class="suspicious-label">Q?</text>
            </g>
          </g>
          <text x="218" y="303" class="axis-label center">涡流响应 z₁</text>
          <text transform="translate(25 168) rotate(-90)" class="axis-label center">缩放后的轮廓特征 s·z₂</text>

          <text x="420" y="62" class="audit-label">C / λ</text>
          <text x="562" y="62" class="audit-value">{{ result.c.toFixed(2) }} / {{ result.lambda.toExponential(1) }}</text>
          <text x="420" y="95" class="audit-label">半间隔 1/‖w‖</text>
          <text x="562" y="95" class="audit-value safe">{{ result.halfMargin.toFixed(3) }}</text>
          <text x="420" y="128" class="audit-label">支持向量</text>
          <text x="562" y="128" class="audit-value">{{ result.supportCount }} / {{ result.diagnostics.length }}</text>
          <line x1="420" x2="562" y1="146" y2="146" class="audit-rule" />
          <text x="420" y="174" class="audit-label">Σξ</text>
          <text x="562" y="174" class="audit-value danger">{{ result.slackSum.toFixed(3) }}</text>
          <text x="420" y="207" class="audit-label">疑似件 y·f</text>
          <text x="562" y="207" :class="['audit-value', result.suspicious.functionalMargin > 1 ? 'safe' : 'danger']">{{ result.suspicious.functionalMargin.toFixed(3) }}</text>
          <text x="420" y="240" class="audit-label">疑似件 ξ</text>
          <text x="562" y="240" :class="['audit-value', result.suspicious.slack > 0 ? 'danger' : 'safe']">{{ result.suspicious.slack.toFixed(3) }}</text>
          <line x1="420" x2="562" y1="258" y2="258" class="audit-rule" />
          <text x="420" y="286" class="audit-label">训练准确率</text>
          <text x="562" y="286" class="audit-value winner">{{ (result.accuracy * 100).toFixed(1) }}%</text>
        </svg>
        <figcaption><span><i class="pass" />自动放行 −1</span><span><i class="defect" />送复检 +1</span><span><i class="support" />支持向量</span><span><i class="violation" />间隔违约</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <label>
        <span>log₁₀ C <output>{{ logC.toFixed(2) }} → {{ result.c.toFixed(2) }}</output></span>
        <input v-model.number="logC" type="range" min="-2" max="2" step="0.05">
      </label>
      <label>
        <span>疑似件横坐标 z₁ <output>{{ suspiciousX.toFixed(2) }}</output></span>
        <input v-model.number="suspiciousX" type="range" min="-1.80" max="0.80" step="0.05">
      </label>
      <label>
        <span>第二特征尺度 s <output>{{ secondFeatureScale.toFixed(2) }}</output></span>
        <input v-model.number="secondFeatureScale" type="range" min="0.40" max="2.20" step="0.05">
      </label>
      <p>把疑似件向左拖入自动放行簇，观察高 C 是否牺牲间隔追逐它。改变 s 会改变欧氏几何，因此不仅是“让优化更快”的数值处理。</p>
    </template>
    <template #metrics>
      <div><span>半间隔</span><strong>{{ result.halfMargin.toFixed(3) }}</strong></div>
      <div><span>总松弛量 Σξ</span><strong>{{ result.slackSum.toFixed(3) }}</strong></div>
      <div><span>支持向量占比</span><strong>{{ (100 * result.supportCount / result.diagnostics.length).toFixed(1) }}%</strong></div>
      <div><span>疑似件状态</span><strong>{{ result.suspicious.slack === 0 ? '间隔外' : result.suspicious.correct ? '间隔内' : '已错分' }}</strong></div>
    </template>
    <template #note>
      支持向量是贴住或侵入间隔的训练样本，不等于“最典型样本”。高支持向量占比、总松弛量或疑似错标长期上升，可能意味着批次漂移、标签冲突或缩放合同变化，应进入数据质量告警而非只调 C。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.margin-caliper{margin:0}.margin-caliper svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0c1925}.caliper-bg{fill:#0c1925}.margin-caliper text{font:700 10px ui-monospace,monospace}.panel-title{fill:#dbeaf0}.plot-bg{fill:#101f2c}.grid line{stroke:#213b4d;stroke-width:1}.decision-line{fill:none;stroke:#f3c85b;stroke-width:3}.margin-jaw{fill:none;stroke-width:2;stroke-dasharray:7 5}.margin-jaw.negative{stroke:#63d0c6}.margin-jaw.positive{stroke:#e35d68}.pass-point{fill:#63d0c6;stroke:#dbeaf0;stroke-width:1.5}.defect-point{fill:#e35d68;stroke:#dbeaf0;stroke-width:1.5}.support-ring{fill:none;stroke:#f3c85b;stroke-width:2}.support-ring.is-violating{stroke:#f08a55;stroke-width:3}.suspicious-label{fill:#f3c85b}.axis-label,.audit-label{fill:#829eaf}.axis-label.center{text-anchor:middle}.audit-value{fill:#dbeaf0;text-anchor:end;font-size:11px}.audit-value.safe{fill:#63d0c6}.audit-value.danger{fill:#e35d68}.audit-value.winner{fill:#f3c85b}.audit-rule{stroke:#355165}.margin-caliper figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.margin-caliper figcaption span{display:inline-flex;align-items:center;gap:5px}.margin-caliper figcaption i{width:12px;height:8px;border-radius:2px}.margin-caliper figcaption i.pass{background:#63d0c6}.margin-caliper figcaption i.defect{background:#e35d68}.margin-caliper figcaption i.support{border:2px solid #f3c85b;background:transparent}.margin-caliper figcaption i.violation{border:2px solid #f08a55;background:transparent}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.margin-caliper{overflow-x:auto;overscroll-behavior-inline:contain}.margin-caliper svg{width:560px;max-width:none}.margin-caliper figcaption{min-width:560px}}
</style>
