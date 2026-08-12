<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const correlation = ref(.72)
const varianceRatio = ref(2.2)
const queryAngle = ref(38)
const queryRadius = ref(1.45)

const geometry = computed(() => {
  const a = varianceRatio.value
  const d = 1 / varianceRatio.value
  const b = correlation.value * Math.sqrt(a * d)
  const root = Math.sqrt((a - d) ** 2 + 4 * b * b)
  const lambda1 = (a + d + root) / 2
  const lambda2 = (a + d - root) / 2
  const theta = .5 * Math.atan2(2 * b, a - d)
  const phi = queryAngle.value * Math.PI / 180
  const x = queryRadius.value * Math.cos(phi)
  const y = queryRadius.value * Math.sin(phi)
  const determinant = a * d - b * b
  const mahalanobisSq = (d * x * x - 2 * b * x * y + a * y * y) / determinant
  const principal1 = Math.cos(theta) * x + Math.sin(theta) * y
  const principal2 = -Math.sin(theta) * x + Math.cos(theta) * y
  const whiteX = principal1 / Math.sqrt(lambda1)
  const whiteY = principal2 / Math.sqrt(lambda2)
  return {
    a, b, d, lambda1, lambda2, theta, x, y, whiteX, whiteY,
    euclideanSq: x * x + y * y,
    mahalanobisSq,
    whitenedSq: whiteX * whiteX + whiteY * whiteY,
    condition: lambda1 / lambda2,
  }
})
const rawPoint = computed(() => ({ x: 170 + 56 * geometry.value.x, y: 184 - 56 * geometry.value.y }))
const whitePoint = computed(() => ({ x: 470 + 50 * geometry.value.whiteX, y: 184 - 50 * geometry.value.whiteY }))
const ellipseAngle = computed(() => -geometry.value.theta * 180 / Math.PI)
</script>

<template>
  <MlExperimentFrame
    index="23-B"
    title="把协方差椭圆送入白化量检具"
    question="欧氏距离为何会误判沿高方差方向的正常波动？调节相关性、尺度比和探针方向，验证 Mahalanobis 距离等于白化空间中的欧氏距离。"
    formula="d²Σ(x,μ)=(x−μ)ᵀΣ⁻¹(x−μ)=‖Σ⁻¹ᐟ²(x−μ)‖²₂"
  >
    <template #visual>
      <div class="whitening-scroll">
        <svg viewBox="0 0 640 350" role="img" :aria-label="`原空间欧氏距离平方${geometry.euclideanSq.toFixed(2)}，Mahalanobis距离平方${geometry.mahalanobisSq.toFixed(2)}，白化距离平方${geometry.whitenedSq.toFixed(2)}`">
          <rect width="640" height="350" rx="15" class="panel" />
          <text x="24" y="27" class="eyebrow">COVARIANCE WHITENING GAUGE · SAME DISTANCE, NEW COORDINATES</text>
          <line x1="320" x2="320" y1="46" y2="316" class="separator" />
          <text x="170" y="54" text-anchor="middle" class="bay-title">原始传感器坐标</text>
          <text x="470" y="54" text-anchor="middle" class="bay-title">白化坐标</text>
          <g class="axes">
            <line x1="55" x2="285" y1="184" y2="184" />
            <line x1="170" x2="170" y1="69" y2="299" />
            <line x1="355" x2="585" y1="184" y2="184" />
            <line x1="470" x2="470" y1="69" y2="299" />
          </g>
          <ellipse cx="170" cy="184" :rx="112 * Math.sqrt(geometry.lambda1)" :ry="112 * Math.sqrt(geometry.lambda2)" :transform="`rotate(${ellipseAngle} 170 184)`" class="raw-race" />
          <circle cx="470" cy="184" r="100" class="white-race" />
          <line x1="170" y1="184" :x2="rawPoint.x" :y2="rawPoint.y" class="raw-vector" />
          <line x1="470" y1="184" :x2="whitePoint.x" :y2="whitePoint.y" class="white-vector" />
          <circle :cx="rawPoint.x" :cy="rawPoint.y" r="7" class="raw-point" />
          <circle :cx="whitePoint.x" :cy="whitePoint.y" r="7" class="white-point" />
          <path d="M293 171 L318 184 L293 197" class="transfer" />
          <text x="320" y="218" text-anchor="middle" class="transfer-label">旋转 + 按标准差缩放</text>
          <rect x="50" y="306" width="240" height="28" rx="4" class="meter" />
          <rect x="350" y="306" width="240" height="28" rx="4" class="meter" />
          <text x="170" y="324" text-anchor="middle" class="meter-text">椭圆半径：dΣ = {{ Math.sqrt(geometry.mahalanobisSq).toFixed(2) }}</text>
          <text x="470" y="324" text-anchor="middle" class="meter-text">圆半径：‖z‖₂ = {{ Math.sqrt(geometry.whitenedSq).toFixed(2) }}</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>相关系数 ρ <output>{{ correlation.toFixed(2) }}</output></span><input v-model.number="correlation" aria-label="协方差相关系数" type="range" min="-0.88" max="0.88" step="0.01"></label>
      <label class="control"><span>方差方向比 <output>{{ varianceRatio.toFixed(2) }}</output></span><input v-model.number="varianceRatio" aria-label="协方差方向比" type="range" min="0.5" max="3" step="0.02"></label>
      <label class="control"><span>探针方向 <output>{{ queryAngle }}°</output></span><input v-model.number="queryAngle" aria-label="查询向量方向" type="range" min="0" max="360" step="1"></label>
      <label class="control"><span>探针原始半径 <output>{{ queryRadius.toFixed(2) }}</output></span><input v-model.number="queryRadius" aria-label="查询向量半径" type="range" min="0.2" max="2.5" step="0.02"></label>
    </template>
    <template #metrics>
      <div><span>欧氏距离平方</span><strong>{{ geometry.euclideanSq.toFixed(3) }}</strong></div>
      <div><span>Mahalanobis²</span><strong>{{ geometry.mahalanobisSq.toFixed(3) }}</strong></div>
      <div><span>白化欧氏²</span><strong>{{ geometry.whitenedSq.toFixed(3) }}</strong></div>
      <div><span>协方差条件数</span><strong>{{ geometry.condition.toFixed(1) }}</strong></div>
    </template>
    <template #note>白化不会“删除相关性”，而是把协方差定义的单位换成各主方向的标准差。探针落在椭圆长轴方向时，原始欧氏距离可能很大，但这正是模型认为常见的波动；沿短轴同样的欧氏位移则会得到更大的 Mahalanobis 距离。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.whitening-scroll{overflow-x:auto}.whitening-scroll svg{display:block;width:100%;height:auto}.panel{fill:#0b1218}.separator{stroke:#324650;stroke-width:1;stroke-dasharray:4 5}.axes line{stroke:#2b3f49;stroke-width:1}.raw-race{fill:#23c9bd;fill-opacity:.08;stroke:#38d8cc;stroke-width:3;stroke-dasharray:8 5}.white-race{fill:#efb449;fill-opacity:.07;stroke:#ffc864;stroke-width:3}.raw-vector{stroke:#70e0d8;stroke-width:2}.white-vector{stroke:#ffd57c;stroke-width:2}.raw-point{fill:#38d8cc;stroke:#d7fffb;stroke-width:2}.white-point{fill:#efb449;stroke:#fff0c6;stroke-width:2}.transfer{fill:none;stroke:#d477ff;stroke-width:3}.meter{fill:#121f28;stroke:#3a515d}.eyebrow,.bay-title,.transfer-label,.meter-text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#70a6b3;font-size:9px;letter-spacing:.12em}.bay-title{fill:#dce7ea;font-size:10px;font-weight:800}.transfer-label{fill:#c493df;font-size:7px}.meter-text{fill:#efd27a;font-size:9px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebdc5;font-size:11px}.control output{color:#ffc864;font:760 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#23c9bd}input:focus-visible{outline:3px solid #d477ff;outline-offset:2px}@media(max-width:560px){.whitening-scroll{margin-right:-18px;padding-right:18px}.whitening-scroll svg{width:600px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
