<script setup lang="ts">
import { computed, ref } from 'vue'

const dtype = ref<'fp16' | 'bf16' | 'fp32'>('fp16')
const gradientExponent = ref(-7)
const lossScale = ref(1024)
const accumulation = ref(4)

const limits = {
  fp16: { min: 6.10e-5, max: 65504 },
  bf16: { min: 1.18e-38, max: 3.39e38 },
  fp32: { min: 1.18e-38, max: 3.39e38 }
}
const raw = computed(() => 10 ** gradientExponent.value)
const scaled = computed(() => raw.value * lossScale.value)
const state = computed(() => scaled.value > limits[dtype.value].max ? 'overflow' : scaled.value < limits[dtype.value].min ? 'underflow' : 'representable')
const recovered = computed(() => state.value === 'representable' ? scaled.value / lossScale.value : state.value === 'underflow' ? 0 : Number.POSITIVE_INFINITY)
const snr = computed(() => Math.sqrt(accumulation.value) * (dtype.value === 'fp16' ? 1 : dtype.value === 'bf16' ? 0.8 : 1.4))
</script>

<template>
  <DlExperimentFrame index="02-B" title="混合精度量程台" question="Loss scaling 如何把小梯度搬进 FP16 的可表示区间，又会在何处引发溢出？" formula="scaled loss → scaled gradients → overflow check → unscale → optimizer">
    <template #visual>
      <div class="range" :class="state" role="img" :aria-label="`当前缩放梯度为 ${scaled.toExponential(2)}，状态 ${state}`">
        <div><span>原始梯度</span><strong>{{ raw.toExponential(2) }}</strong></div><b>× {{ lossScale }}</b><div><span>存储量</span><strong>{{ scaled.toExponential(2) }}</strong></div><b>÷ {{ lossScale }}</b><div><span>恢复梯度</span><strong>{{ Number.isFinite(recovered) ? recovered.toExponential(2) : 'Inf' }}</strong></div>
        <p>{{ state }}</p>
      </div>
    </template>
    <template #controls>
      <label>梯度存储格式<select v-model="dtype"><option value="fp16">FP16</option><option value="bf16">BF16</option><option value="fp32">FP32</option></select></label>
      <label>原始梯度 = 10^{{ gradientExponent }}<input v-model.number="gradientExponent" type="range" min="-40" max="3" step="1"></label>
      <label>Loss scale = {{ lossScale }}<input v-model.number="lossScale" type="range" min="1" max="65536" step="1"></label>
      <label>梯度累积 = {{ accumulation }} micro-batches<input v-model.number="accumulation" type="range" min="1" max="32" step="1"></label>
    </template>
    <template #metrics>
      <div><span>格式最小正规数</span><strong>{{ limits[dtype].min.toExponential(2) }}</strong></div>
      <div><span>格式最大有限数</span><strong>{{ limits[dtype].max.toExponential(2) }}</strong></div>
      <div><span>数值状态</span><strong>{{ state }}</strong></div>
      <div><span>相对聚合信号</span><strong>{{ snr.toFixed(2) }}×</strong></div>
    </template>
    <template #note>真实硬件通常还支持 subnormal、flush-to-zero 和动态 loss scale；本实验用正规数阈值建立第一层判断。BF16 的指数范围接近 FP32，但尾数更短，因此“不过流”不代表精度与 FP32 相同。</template>
  </DlExperimentFrame>
</template>

<style scoped>
.range{position:relative;display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:12px;min-height:285px;padding:22px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.range div{min-width:0;padding:18px 10px;border:1px solid #315267;border-radius:6px;text-align:center;background:#13243a}.range span,.range strong{display:block}.range span{color:#8ba5ad;font-size:10px}.range strong{margin-top:8px;color:#58d3c8;font:750 12px ui-monospace,monospace}.range b{color:#f2c14e;font:700 12px ui-monospace,monospace}.range p{position:absolute;right:15px;top:8px;color:#70dcd3;font:800 9px ui-monospace,monospace}.range.overflow p,.range.underflow p{color:#ff8f81}.range.overflow div:nth-of-type(2),.range.underflow div:nth-of-type(2){border-color:#df6c5b}@media(max-width:620px){.range{grid-template-columns:1fr}.range b{justify-self:center}}
</style>
