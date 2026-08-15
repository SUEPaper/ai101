<script setup lang="ts">
import { computed, ref } from 'vue'

const samples = ref(1200)
const shift = ref(35)
const strategy = ref<'head' | 'partial' | 'full'>('partial')
const augmentation = ref(40)

const params = computed(() => ({ head: 1.2, partial: 18, full: 100 })[strategy.value])
const fitPower = computed(() => ({ head: 0.46, partial: 0.76, full: 1 })[strategy.value])
const accuracy = computed(() => {
  const data = Math.log10(samples.value) / 4
  const mismatch = shift.value / 100
  const aug = augmentation.value / 100
  const score = 0.55 + 0.33 * data + 0.15 * fitPower.value * mismatch + 0.07 * aug - 0.18 * mismatch * (1 - fitPower.value)
  return Math.max(0.45, Math.min(0.94, score))
})
const gap = computed(() => Math.max(0.01, 0.16 * params.value / 100 * (1 - Math.log10(samples.value) / 5) - augmentation.value / 1500))
const time = computed(() => (0.8 + params.value / 12) * (samples.value / 1000))
const recommendation = computed(() => samples.value < 800 && shift.value < 45 ? '先训练 head' : samples.value < 4000 ? '部分解冻' : shift.value > 60 ? '全量微调并强审计' : '部分或全量消融')
</script>

<template>
  <DlExperimentFrame index="03-A" title="迁移学习解冻控制台" question="数据量、领域偏移和可训练参数比例怎样决定冻结策略？" formula="benefit = transferable features + adaptation − variance − compute cost">
    <template #visual>
      <div class="blocks" role="img" :aria-label="`${strategy} 策略，可训练参数 ${params}%`">
        <div v-for="index in 12" :key="index" :class="{ trainable: index > 12 - Math.ceil(params / 8.34) }"><span>{{ index < 4 ? 'edge' : index < 9 ? 'shape' : 'task' }}</span></div>
        <p>输入域偏移 {{ shift }}% → 表征骨干 → 任务头</p>
      </div>
    </template>
    <template #controls>
      <label>训练样本 = {{ samples }}<input v-model.number="samples" type="range" min="200" max="12000" step="200"></label>
      <label>领域偏移 = {{ shift }}%<input v-model.number="shift" type="range" min="0" max="100" step="5"></label>
      <label>解冻策略<select v-model="strategy"><option value="head">只训练任务头</option><option value="partial">解冻后部层</option><option value="full">全量微调</option></select></label>
      <label>增强强度 = {{ augmentation }}%<input v-model.number="augmentation" type="range" min="0" max="100" step="5"></label>
    </template>
    <template #metrics>
      <div><span>可训练参数</span><strong>{{ params.toFixed(1) }}%</strong></div>
      <div><span>验证准确率估计</span><strong>{{ (accuracy * 100).toFixed(1) }}%</strong></div>
      <div><span>训练—验证 gap</span><strong>{{ (gap * 100).toFixed(1) }} pp</strong></div>
      <div><span>相对训练时间</span><strong>{{ time.toFixed(1) }} h</strong></div>
    </template>
    <template #note>建议：{{ recommendation }}。数值是教学响应面，不是对真实数据的性能承诺；真正选择必须固定验证集并比较至少三种解冻策略。</template>
  </DlExperimentFrame>
</template>

<style scoped>
.blocks{display:grid;grid-template-columns:repeat(12,minmax(24px,1fr));align-items:end;gap:5px;min-height:285px;padding:32px 18px 40px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.blocks div{display:grid;place-items:end center;height:110px;border:1px solid #3c5d6c;border-radius:4px;background:#1a3046}.blocks div:nth-child(n+5){height:150px}.blocks div:nth-child(n+10){height:195px}.blocks div.trainable{border-color:#f2c14e;background:linear-gradient(180deg,#26384b,#7b621c)}.blocks span{padding-bottom:8px;color:#82a0a8;font:7px ui-monospace,monospace;writing-mode:vertical-rl}.blocks p{grid-column:1/-1;margin:16px 0 0;color:#9eb6bc;font-size:10px;text-align:center}
</style>
