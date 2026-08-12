<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type StridePolicy = 'dense' | 'staged'

const depth = ref(3)
const kernelSize = ref(3)
const dilation = ref(1)
const stridePolicy = ref<StridePolicy>('dense')

const plan = computed(() => {
  let receptiveField = 1
  let jump = 1
  let spatial = 64
  let parameters = 0
  const layers = [] as Array<{ index: number; receptiveField: number; jump: number; stride: number; spatial: number; parameters: number }>
  for (let index = 1; index <= depth.value; index += 1) {
    const stride = stridePolicy.value === 'staged' && index % 2 === 0 ? 2 : 1
    const inputChannels = index === 1 ? 3 : 16
    const layerParameters = 16 * (inputChannels * kernelSize.value ** 2 + 1)
    receptiveField += (kernelSize.value - 1) * dilation.value * jump
    jump *= stride
    spatial = Math.ceil(spatial / stride)
    parameters += layerParameters
    layers.push({ index, receptiveField, jump, stride, spatial, parameters: layerParameters })
  }
  const fullyConnectedParameters = (64 * 64 * 3 + 1) * (64 * 64 * 16)
  return { layers, receptiveField, jump, spatial, parameters, fullyConnectedParameters }
})

const displayCells = computed(() => Math.min(16, Math.ceil(plan.value.receptiveField / plan.value.jump) + 2))
const aperture = computed(() => Math.min(16, Math.max(1, Math.ceil(plan.value.receptiveField / 4))))
const apertureStart = computed(() => 33 + (16 - aperture.value) * 6.5)
const compact = (value: number) => value >= 1e9 ? `${(value / 1e9).toFixed(2)}G` : value >= 1e6 ? `${(value / 1e6).toFixed(2)}M` : `${value.toLocaleString('en-US')}`
</script>

<template>
  <MlExperimentFrame
    index="15-B"
    title="规划多层卷积的理论感受野"
    question="堆叠、步幅和膨胀率怎样改变一个深层单元能看到的原图范围？参数预算与空间分辨率又付出了什么代价？"
    formula="rₗ=rₗ₋₁+(Kₗ−1)Dₗjₗ₋₁；jₗ=jₗ₋₁Sₗ"
  >
    <template #visual>
      <figure class="rf-planner">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`深度${depth}层，理论感受野${plan.receptiveField}，输出跳距${plan.jump}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="planner-bg" />
          <text x="32" y="38" class="panel-title">原始钢带窗口 · 64×64×3</text>
          <text x="309" y="38" class="panel-title">逐层几何账本</text>
          <g v-for="cellRow in 16" :key="`row-${cellRow}`">
            <rect
              v-for="cellColumn in 16" :key="`cell-${cellRow}-${cellColumn}`"
              :x="33+(cellColumn-1)*13" :y="53+(cellRow-1)*13" width="11" height="11" rx="1.5"
              :class="['sensor-cell',{ texture:(cellRow+cellColumn)%5===0 }]"
            />
          </g>
          <rect
            :x="apertureStart" :y="53+(16-aperture)*6.5"
            :width="aperture*13-2" :height="aperture*13-2" rx="5" class="aperture"
          />
          <line x1="242" y1="157" x2="285" y2="157" class="projection" />
          <text x="136" y="279" class="field-label" text-anchor="middle">
            {{ plan.receptiveField>=64 ? '理论窗口已覆盖整幅输入' : `中心单元查看 ${plan.receptiveField}×${plan.receptiveField} 原像素` }}
          </text>

          <g v-for="(layer, index) in plan.layers" :key="layer.index" :transform="`translate(309 ${53+index*38})`">
            <rect width="244" height="30" rx="5" :class="['layer-card',{ active:index===plan.layers.length-1 }]" />
            <text x="12" y="12" class="layer-index">CONV {{ layer.index }}</text>
            <text x="12" y="23" class="layer-rule">K{{ kernelSize }} · D{{ dilation }} · S{{ layer.stride }}</text>
            <text x="116" y="12" class="layer-value">r={{ layer.receptiveField }} · j={{ layer.jump }}</text>
            <text x="232" y="23" class="layer-output" text-anchor="end">{{ layer.spatial }}×{{ layer.spatial }}×16</text>
          </g>
          <g v-if="plan.layers.length<6">
            <rect v-for="index in 6-plan.layers.length" :key="`ghost-${index}`" x="309" :y="53+(plan.layers.length+index-1)*38" width="244" height="30" rx="5" class="layer-ghost" />
          </g>
          <text x="309" y="294" class="budget-label">卷积参数 {{ compact(plan.parameters) }}</text>
          <text x="553" y="294" class="budget-label" text-anchor="end">同形全连接 {{ compact(plan.fullyConnectedParameters) }}</text>
        </svg>
        <figcaption><span><i class="grid" />原像素</span><span><i class="field" />理论感受野</span><span><i class="active" />当前最深层</span><span><i class="stride" />阶段下采样</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <label><span>卷积深度 L <output>{{ depth }} 层</output></span><input v-model.number="depth" type="range" min="1" max="6" step="1"></label>
      <div class="choice-switch" role="group" aria-label="卷积核尺寸">
        <button type="button" :aria-pressed="kernelSize===3" @click="kernelSize=3">3 × 3 核</button>
        <button type="button" :aria-pressed="kernelSize===5" @click="kernelSize=5">5 × 5 核</button>
      </div>
      <label><span>膨胀率 D <output>{{ dilation }}</output></span><input v-model.number="dilation" type="range" min="1" max="3" step="1"></label>
      <div class="choice-switch" role="group" aria-label="步幅策略">
        <button type="button" :aria-pressed="stridePolicy==='dense'" @click="stridePolicy='dense'">全程步幅 1</button>
        <button type="button" :aria-pressed="stridePolicy==='staged'" @click="stridePolicy='staged'">偶数层步幅 2</button>
      </div>
      <p>黄色方框画的是理论覆盖范围，不代表每个像素贡献相同。训练后的有效感受野通常集中在中心，必须借助遮挡、梯度或特征归因另行诊断。</p>
    </template>
    <template #metrics>
      <div><span>理论感受野 r</span><strong>{{ plan.receptiveField }} × {{ plan.receptiveField }}</strong></div>
      <div><span>原图跳距 j</span><strong>{{ plan.jump }} px</strong></div>
      <div><span>末层空间尺寸</span><strong>{{ plan.spatial }} × {{ plan.spatial }}</strong></div>
      <div><span>参数压缩比</span><strong>{{ Math.round(plan.fullyConnectedParameters/plan.parameters).toLocaleString('en-US') }}×</strong></div>
    </template>
    <template #note>
      膨胀卷积在不增加核参数的情况下扩大理论感受野；步幅会同时扩大相邻输出在原图上的跳距并降低空间分辨率。两者都不是免费午餐：网格采样可能遗漏细划痕或产生棋盘式盲区。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.rf-planner{margin:0}.rf-planner svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0b1722}.planner-bg{fill:#0b1722}.rf-planner text{font:700 8px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.sensor-cell{fill:#132a37;stroke:#27495b}.sensor-cell.texture{fill:#1b4650}.aperture{fill:rgba(243,200,91,.13);stroke:#f3c85b;stroke-width:2;stroke-dasharray:5 3}.projection{stroke:#63d0c6;stroke-width:2;stroke-dasharray:6 4}.field-label{fill:#b7cad3;font-size:9px}.layer-card{fill:#102433;stroke:#3d5f73}.layer-card.active{fill:#1c2c39;stroke:#f3c85b;stroke-width:1.5}.layer-index{fill:#63d0c6}.layer-rule{fill:#829eaf}.layer-value{fill:#f3c85b}.layer-output{fill:#b7cad3}.layer-ghost{fill:#0e1c28;stroke:#233d4e;stroke-dasharray:4 4}.budget-label{fill:#829eaf}.rf-planner figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.rf-planner figcaption span{display:inline-flex;align-items:center;gap:5px}.rf-planner figcaption i{width:12px;height:8px;border-radius:2px}.rf-planner figcaption i.grid{background:#1b4650}.rf-planner figcaption i.field{border:2px solid #f3c85b}.rf-planner figcaption i.active{background:#f3c85b}.rf-planner figcaption i.stride{background:#63d0c6}.choice-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.choice-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 9px ui-monospace,monospace;cursor:pointer}.choice-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.choice-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.rf-planner{overflow-x:auto;overscroll-behavior-inline:contain}.rf-planner svg{width:560px;max-width:none}.rf-planner figcaption{min-width:560px}}
</style>
