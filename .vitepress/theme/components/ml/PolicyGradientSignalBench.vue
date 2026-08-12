<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Route = 'north' | 'east'

const logit = ref(-0.35)
const northReturn = ref(5.4)
const eastReturn = ref(1.2)
const baseline = ref(2.0)
const entropyWeight = ref(0.08)
const sampledRoute = ref<Route>('north')

const probability = computed(() => 1 / (1 + Math.exp(-logit.value)))
const expectedReturn = computed(() =>
  probability.value * northReturn.value + (1 - probability.value) * eastReturn.value)
const returnGradient = computed(() =>
  probability.value * (1 - probability.value) * (northReturn.value - eastReturn.value))
const entropy = computed(() => {
  const p = probability.value
  return -(p * Math.log(p) + (1 - p) * Math.log(1 - p))
})
const entropyGradient = computed(() => {
  const p = probability.value
  return p * (1 - p) * Math.log((1 - p) / p)
})
const totalGradient = computed(() => returnGradient.value + entropyWeight.value * entropyGradient.value)
const sampleScore = computed(() => sampledRoute.value === 'north' ? 1 - probability.value : -probability.value)
const sampleReturn = computed(() => sampledRoute.value === 'north' ? northReturn.value : eastReturn.value)
const sampleSignal = computed(() => sampleScore.value * (sampleReturn.value - baseline.value))
const baselineExpectation = computed(() => {
  const p = probability.value
  return p * (1 - p) * (northReturn.value - baseline.value)
    + (1 - p) * (-p) * (eastReturn.value - baseline.value)
})
const needleAngle = computed(() => -112 + probability.value * 224)
const gradientDirection = computed(() => totalGradient.value >= 0 ? '提高北干线概率' : '降低北干线概率')
</script>

<template>
  <MlExperimentFrame
    index="32-A"
    title="策略梯度信号台"
    question="在一个可手算的 AMR 分流节点上，拆开动作概率、score、回报权重、baseline 与熵梯度。"
    formula="∂J/∂z = p(1−p)(q_N−q_E)"
  >
    <template #visual>
      <div class="signal-bench">
        <div class="nameplate">
          <span>AMR JUNCTION / POLICY SIGNAL</span>
          <b>PG-32A</b>
        </div>
        <svg viewBox="0 0 720 310" role="img" aria-label="北干线与东捷径的策略概率和梯度方向">
          <defs>
            <filter id="pg-glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          </defs>
          <rect x="18" y="18" width="684" height="274" rx="8" class="panel" />
          <path d="M100 215 H270 Q310 215 310 175 V85 H610" class="pipe north" />
          <path d="M310 215 H610" class="pipe east" />
          <circle cx="310" cy="215" r="31" class="valve" />
          <path d="M310 215 L310 184" :transform="`rotate(${needleAngle} 310 215)`" class="valve-needle" />
          <text x="54" y="222" class="route-label">决策口</text>
          <text x="494" y="73" class="route-label">北干线 / N</text>
          <text x="494" y="205" class="route-label">东捷径 / E</text>
          <text x="494" y="98" class="route-value">q = {{ northReturn.toFixed(1) }}</text>
          <text x="494" y="230" class="route-value">q = {{ eastReturn.toFixed(1) }}</text>
          <g transform="translate(72 46)">
            <path d="M0 72 A72 72 0 0 1 144 72" class="meter-track" />
            <line x1="72" y1="72" x2="72" y2="12" :transform="`rotate(${needleAngle} 72 72)`" class="meter-needle" />
            <circle cx="72" cy="72" r="7" class="meter-hub" />
            <text x="72" y="96" text-anchor="middle" class="meter-number">p(N) {{ (probability * 100).toFixed(1) }}%</text>
          </g>
          <path d="M350 265 H610" :class="['gradient-bus', { negative: totalGradient < 0 }]" filter="url(#pg-glow)" />
          <polygon :points="totalGradient >= 0 ? '610,265 594,256 594,274' : '350,265 366,256 366,274'" :class="['bus-arrow', { negative: totalGradient < 0 }]" />
          <text x="480" y="284" text-anchor="middle" class="bus-label">{{ gradientDirection }}</text>
        </svg>
        <div class="sample-relay">
          <span>本次采样</span>
          <b>{{ sampledRoute === 'north' ? '北干线 N' : '东捷径 E' }}</b>
          <code>score={{ sampleScore.toFixed(3) }} × (q−b)={{ (sampleReturn - baseline).toFixed(2) }}</code>
          <strong :class="{ negative: sampleSignal < 0 }">样本信号 {{ sampleSignal >= 0 ? '+' : '' }}{{ sampleSignal.toFixed(3) }}</strong>
        </div>
      </div>
    </template>

    <template #controls>
      <label>
        <span>北干线 logit z</span>
        <input v-model.number="logit" type="range" min="-3" max="3" step="0.05" />
        <output>{{ logit.toFixed(2) }}</output>
      </label>
      <label>
        <span>北干线回报 qN</span>
        <input v-model.number="northReturn" type="range" min="-4" max="9" step="0.2" />
        <output>{{ northReturn.toFixed(1) }}</output>
      </label>
      <label>
        <span>东捷径回报 qE</span>
        <input v-model.number="eastReturn" type="range" min="-4" max="9" step="0.2" />
        <output>{{ eastReturn.toFixed(1) }}</output>
      </label>
      <label>
        <span>状态 baseline b</span>
        <input v-model.number="baseline" type="range" min="-6" max="10" step="0.2" />
        <output>{{ baseline.toFixed(1) }}</output>
      </label>
      <label>
        <span>熵系数 β</span>
        <input v-model.number="entropyWeight" type="range" min="0" max="0.5" step="0.01" />
        <output>{{ entropyWeight.toFixed(2) }}</output>
      </label>
      <div class="route-buttons" aria-label="选择要观察的采样动作">
        <button :aria-pressed="sampledRoute === 'north'" @click="sampledRoute = 'north'">采样 N</button>
        <button :aria-pressed="sampledRoute === 'east'" @click="sampledRoute = 'east'">采样 E</button>
      </div>
    </template>

    <template #metrics>
      <div><span>期望回报 J</span><strong>{{ expectedReturn.toFixed(3) }}</strong></div>
      <div><span>回报梯度</span><strong>{{ returnGradient >= 0 ? '+' : '' }}{{ returnGradient.toFixed(3) }}</strong></div>
      <div><span>总梯度（含熵）</span><strong>{{ totalGradient >= 0 ? '+' : '' }}{{ totalGradient.toFixed(3) }}</strong></div>
      <div><span>baseline 后期望</span><strong>{{ baselineExpectation.toFixed(3) }}</strong></div>
    </template>

    <template #note>
      单次样本信号会随 baseline 改变甚至翻号，但两动作加权后的期望恒等于回报梯度；baseline 改的是噪声，不是目标。熵项是另一项目标，其梯度应单独记账。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.signal-bench{color:#17324d;font-family:"Arial Narrow","PingFang SC",sans-serif}.nameplate{display:flex;justify-content:space-between;padding:9px 12px;border:1px solid #a89d82;border-bottom:0;border-radius:8px 8px 0 0;background:#ebe4cf;color:#17324d;font:800 10px/1.2 ui-monospace,monospace;letter-spacing:.08em}.nameplate b{color:#a64d79}.signal-bench svg{display:block;width:100%;min-width:560px;border:1px solid #a89d82;border-radius:0 0 8px 8px;background:#d8d0b9}.panel{fill:#ebe4cf;stroke:#9b9179;stroke-width:2}.pipe{fill:none;stroke-width:16;stroke-linecap:round}.pipe.north{stroke:#3f8f83}.pipe.east{stroke:#f0a83b}.valve{fill:#17324d;stroke:#101f2f;stroke-width:5}.valve-needle{stroke:#ebe4cf;stroke-width:6;stroke-linecap:round}.route-label{fill:#17324d;font:800 15px "Arial Narrow",sans-serif;letter-spacing:.04em}.route-value{fill:#5a5c59;font:700 12px ui-monospace,monospace}.meter-track{fill:none;stroke:#17324d;stroke-width:5;stroke-linecap:round}.meter-needle{stroke:#c64d3f;stroke-width:4;stroke-linecap:round}.meter-hub{fill:#17324d}.meter-number{fill:#17324d;font:800 12px ui-monospace,monospace}.gradient-bus{stroke:#3f8f83;stroke-width:5}.gradient-bus.negative{stroke:#a64d79}.bus-arrow{fill:#3f8f83}.bus-arrow.negative{fill:#a64d79}.bus-label{fill:#17324d;font:800 11px "Arial Narrow",sans-serif}.sample-relay{display:grid;grid-template-columns:auto auto 1fr auto;gap:12px;align-items:center;margin-top:10px;padding:12px;border:1px solid #756d5c;border-radius:6px;background:#ebe4cf}.sample-relay span{font-size:10px;letter-spacing:.08em}.sample-relay b{color:#17324d}.sample-relay code{color:#41525b;background:#d8d0b9}.sample-relay strong{color:#287568;font:800 12px ui-monospace,monospace}.sample-relay strong.negative{color:#a64d79}label{display:grid;grid-template-columns:1fr auto;gap:6px;color:#d6dded;font-size:11px}label input{grid-column:1/-1;width:100%;min-height:44px;accent-color:#f0a83b}label output{color:#f0a83b;font:750 11px ui-monospace,monospace}.route-buttons{display:grid;grid-template-columns:1fr 1fr;gap:8px}.route-buttons button{min-height:44px;border:1px solid #50648b;border-radius:5px;color:#dfe6f7;background:#111a31;font-weight:750}.route-buttons button[aria-pressed="true"]{border-color:#f0a83b;color:#111a31;background:#f0a83b}@media(max-width:760px){.signal-bench{overflow-x:auto}.sample-relay{min-width:560px}}@media(max-width:600px){.sample-relay{grid-template-columns:1fr 1fr}.sample-relay code{grid-column:1/-1}}
</style>
