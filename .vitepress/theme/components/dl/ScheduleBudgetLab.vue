<script setup lang="ts">
import { computed, ref, watch } from 'vue'

const schedule = ref<'constant' | 'cosine' | 'onecycle'>('cosine')
const budget = ref(200)
const warmup = ref(10)
const current = ref(80)
const peak = ref(0.001)

watch(budget, (value) => { if (current.value > value) current.value = value })

function lrAt(step: number) {
  const warmupSteps = Math.max(1, budget.value * warmup.value / 100)
  if (step <= warmupSteps) return peak.value * step / warmupSteps
  const progress = (step - warmupSteps) / Math.max(1, budget.value - warmupSteps)
  if (schedule.value === 'constant') return peak.value
  if (schedule.value === 'onecycle') {
    const phase = Math.min(1, progress)
    return peak.value * (phase < 0.35 ? 0.1 + 0.9 * phase / 0.35 : 0.1 + 0.9 * (1 - phase) / 0.65)
  }
  return peak.value * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, progress)))
}

const samples = computed(() => Array.from({ length: 81 }, (_, index) => {
  const step = index * budget.value / 80
  return { step, lr: lrAt(step) }
}))
const points = computed(() => samples.value.map(({ step, lr }) => `${35 + step / budget.value * 470},${220 - lr / peak.value * 175}`).join(' '))
const currentLr = computed(() => lrAt(current.value))
const exposure = computed(() => Array.from({ length: current.value + 1 }, (_, index) => lrAt(index)).reduce((sum, value) => sum + value, 0))
const state = computed(() => current.value <= budget.value * warmup.value / 100 ? 'warmup' : current.value >= budget.value ? 'finished' : schedule.value)
</script>

<template>
  <DlExperimentFrame index="01-B" title="固定预算学习率台" question="相同峰值学习率并不代表相同优化预算；曲线面积和结束位置怎样改变训练？" formula="ηₜ = warmup(t) × decay(t); compare under a fixed step/token budget">
    <template #visual>
      <svg class="schedule" viewBox="0 0 540 255" role="img" :aria-label="`${schedule} 学习率曲线，当前位于第 ${current} 步`">
        <g class="grid"><line v-for="y in [45,90,135,180,220]" :key="y" x1="35" :y1="y" x2="505" :y2="y" /></g>
        <polyline :points="points" />
        <line class="cursor" :x1="35 + current / budget * 470" y1="34" :x2="35 + current / budget * 470" y2="220" />
        <circle :cx="35 + current / budget * 470" :cy="220 - currentLr / peak * 175" r="6" />
        <text x="35" y="242">0</text><text x="468" y="242">{{ budget }} steps</text><text x="39" y="31">learning rate</text>
      </svg>
    </template>
    <template #controls>
      <label>调度策略<select v-model="schedule"><option value="cosine">Warmup + Cosine</option><option value="constant">Warmup + Constant</option><option value="onecycle">One-cycle</option></select></label>
      <label>总预算 = {{ budget }} steps<input v-model.number="budget" type="range" min="50" max="400" step="10"></label>
      <label>Warmup = {{ warmup }}%<input v-model.number="warmup" type="range" min="0" max="30" step="1"></label>
      <label>峰值 η = {{ peak.toExponential(1) }}<input v-model.number="peak" type="range" min="0.0002" max="0.003" step="0.0001"></label>
      <label>当前 step = {{ current }}<input v-model.number="current" type="range" min="0" :max="budget" step="1"></label>
    </template>
    <template #metrics>
      <div><span>当前学习率</span><strong>{{ currentLr.toExponential(3) }}</strong></div>
      <div><span>累计 LR 暴露量</span><strong>{{ exposure.toFixed(4) }}</strong></div>
      <div><span>Warmup 结束</span><strong>{{ Math.round(budget * warmup / 100) }} step</strong></div>
      <div><span>当前阶段</span><strong>{{ state }}</strong></div>
    </template>
    <template #note>比较调度策略必须固定训练预算、批量、数据顺序和峰值学习率。只截取前半段曲线，可能把尚未进入衰减阶段的策略误判为更差。</template>
  </DlExperimentFrame>
</template>

<style scoped>
.schedule{display:block;width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.grid line{stroke:#223a4f}.schedule polyline{fill:none;stroke:#58d3c8;stroke-width:3}.cursor{stroke:#df6c5b;stroke-width:2;stroke-dasharray:5 4}.schedule circle{fill:#f2c14e;stroke:#13213c;stroke-width:2}.schedule text{fill:#8ea9b0;font:10px ui-monospace,monospace}
</style>
