<script setup lang="ts">
import { computed, ref } from 'vue'

const depth = ref(24)
const initialization = ref<'small' | 'xavier' | 'he'>('he')
const activation = ref<'relu' | 'gelu' | 'tanh'>('relu')
const normalization = ref<'none' | 'batch' | 'layer' | 'rms'>('none')

const initGain = computed(() => ({ small: 0.55, xavier: 1, he: 1.42 })[initialization.value])
const actFactor = computed(() => ({ relu: 0.5, gelu: 0.43, tanh: 0.72 })[activation.value])
const layers = computed(() => {
  let variance = 1
  const values = []
  for (let index = 1; index <= depth.value; index += 1) {
    variance *= initGain.value ** 2 * actFactor.value
    if (normalization.value !== 'none') variance = 0.82 * variance + 0.18
    variance = Math.max(1e-5, Math.min(1e4, variance))
    values.push({ index, std: Math.sqrt(variance) })
  }
  return values
})
const finalStd = computed(() => layers.value.at(-1)?.std ?? 1)
const gradientRatio = computed(() => Math.min(1e4, Math.max(1e-5, 1 / finalStd.value)))
const unstable = computed(() => layers.value.filter((item) => item.std < 0.1 || item.std > 3).length)
const diagnosis = computed(() => finalStd.value < 0.1 ? '信号衰减' : finalStd.value > 3 ? '信号放大' : '可用区间')
</script>

<template>
  <DlExperimentFrame index="02-A" title="层间信号隧道" question="初始化、激活和归一化怎样共同决定深层网络中的信号尺度？" formula="Var[hˡ] ≈ Var[hˡ⁻¹] × gain² × activation factor">
    <template #visual>
      <div class="tunnel" role="img" :aria-label="`${depth} 层网络的激活标准差条带，诊断为${diagnosis}`">
        <span v-for="item in layers" :key="item.index" :style="{ height: `${Math.max(4, Math.min(100, 18 + Math.log10(item.std + 0.001) * 34))}%` }" :class="{ bad: item.std < .1 || item.std > 3 }"><i>{{ item.index }}</i></span>
      </div>
    </template>
    <template #controls>
      <label>初始化<select v-model="initialization"><option value="small">过小方差</option><option value="xavier">Xavier</option><option value="he">He</option></select></label>
      <label>激活函数<select v-model="activation"><option value="relu">ReLU</option><option value="gelu">GELU</option><option value="tanh">tanh</option></select></label>
      <label>归一化<select v-model="normalization"><option value="none">无归一化</option><option value="batch">BatchNorm</option><option value="layer">LayerNorm</option><option value="rms">RMSNorm</option></select></label>
      <label>深度 = {{ depth }}<input v-model.number="depth" type="range" min="4" max="48" step="2"></label>
    </template>
    <template #metrics>
      <div><span>末层激活 std</span><strong>{{ finalStd.toExponential(2) }}</strong></div>
      <div><span>近似梯度尺度比</span><strong>{{ gradientRatio.toExponential(2) }}</strong></div>
      <div><span>异常层数</span><strong>{{ unstable }} / {{ depth }}</strong></div>
      <div><span>诊断</span><strong>{{ diagnosis }}</strong></div>
    </template>
    <template #note>这是用于形成尺度直觉的均值场近似，不替代真实模型的逐层日志。归一化能拉回尺度，但不能修复错误数据、饱和激活或不合理残差分支。</template>
  </DlExperimentFrame>
</template>

<style scoped>
.tunnel{display:flex;align-items:center;gap:4px;height:285px;padding:20px;overflow-x:auto;border:1px solid #294b5c;border-radius:7px;background:linear-gradient(180deg,#0b172a,#10243a)}.tunnel span{position:relative;flex:1 0 8px;max-width:20px;min-height:4px;border-radius:2px 2px 0 0;background:#58d3c8;box-shadow:0 0 12px rgba(88,211,200,.28);transition:height .2s ease}.tunnel span.bad{background:#df6c5b}.tunnel i{position:absolute;left:50%;bottom:-17px;transform:translateX(-50%);color:#718c95;font:7px/1 ui-monospace,monospace;font-style:normal}@media(prefers-reduced-motion:reduce){.tunnel span{transition:none}}
</style>
