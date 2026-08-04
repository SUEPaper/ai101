<script setup lang="ts">
import { computed, ref } from 'vue'

const mode = ref<'sum' | 'mean'>('sum')
const rows = [
  { district: '北区', rides: 128 },
  { district: '南区', rides: 94 },
  { district: '北区', rides: 173 },
  { district: '西区', rides: 81 },
  { district: '南区', rides: 116 }
]

const groups = computed(() => {
  const bucket = new Map<string, number[]>()
  rows.forEach((row) => bucket.set(row.district, [...(bucket.get(row.district) ?? []), row.rides]))
  return [...bucket.entries()].map(([district, values]) => ({
    district,
    values,
    result: mode.value === 'sum'
      ? values.reduce((total, value) => total + value, 0)
      : values.reduce((total, value) => total + value, 0) / values.length
  }))
})
</script>

<template>
  <section class="group-lab" aria-label="分组聚合实验">
    <header><div><span>INTERACTIVE 04</span><h3>观察 Split → Apply → Combine</h3></div><div><button type="button" :class="{ 'is-active': mode === 'sum' }" @click="mode = 'sum'">sum</button><button type="button" :class="{ 'is-active': mode === 'mean' }" @click="mode = 'mean'">mean</button></div></header>
    <div class="group-lab__flow">
      <div class="group-lab__source"><small>原始记录</small><p v-for="(row, index) in rows" :key="index"><b>{{ row.district }}</b><span>{{ row.rides }}</span></p></div>
      <div class="group-lab__arrow" aria-hidden="true">→</div>
      <div class="group-lab__groups"><small>按 district 拆组</small><p v-for="group in groups" :key="group.district"><b>{{ group.district }}</b><span>{{ group.values.join(' + ') }}</span></p></div>
      <div class="group-lab__arrow" aria-hidden="true">→</div>
      <div class="group-lab__result"><small>{{ mode }} 后合并</small><p v-for="group in groups" :key="group.district"><b>{{ group.district }}</b><span>{{ Number(group.result.toFixed(1)) }}</span></p></div>
    </div>
    <code>df.groupby('district')['rides'].{{ mode }}()</code>
  </section>
</template>

<style scoped>
.group-lab { margin: 32px 0; padding: 25px; border-radius: 16px; color: #eff6f2; background: #143b37; }
.group-lab header { display: flex; justify-content: space-between; align-items: start; gap: 20px; }
.group-lab header span { color: #f3c56f; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .12em; }
.group-lab h3 { margin: 7px 0 0; border: 0; color: #fff; font-size: 20px; }
.group-lab header button { padding: 7px 10px; border: 1px solid #68867e; color: #d5e3de; background: transparent; font: 750 10px/1 ui-monospace, monospace; cursor: pointer; }
.group-lab header button:first-child { border-radius: 7px 0 0 7px; }
.group-lab header button:last-child { border-radius: 0 7px 7px 0; }
.group-lab header button.is-active { color: #17342f; background: #f3c56f; border-color: #f3c56f; }
.group-lab__flow { display: grid; grid-template-columns: 1fr 24px 1.2fr 24px 1fr; align-items: center; gap: 8px; margin: 24px 0; }
.group-lab__flow > div:not(.group-lab__arrow) { min-height: 215px; padding: 13px; border-radius: 9px; background: #204b43; }
.group-lab__flow small { display: block; margin-bottom: 10px; color: #a9c5bc; font-size: 9px; font-weight: 750; }
.group-lab__flow p { display: flex; justify-content: space-between; gap: 8px; margin: 5px 0; padding: 6px 8px; border-radius: 5px; color: #dce9e4; background: #2b5b51; font-size: 10px; }
.group-lab__flow p b { color: #f3c56f; }
.group-lab__arrow { color: #e5ae49; font-size: 20px; text-align: center; }
.group-lab > code { display: block; padding: 10px 12px; border-radius: 7px; color: #18332e; background: #edf4f0; font-size: 11px; }
@media (max-width: 700px) { .group-lab__flow { grid-template-columns: 1fr; } .group-lab__arrow { transform: rotate(90deg); } .group-lab__flow > div:not(.group-lab__arrow) { min-height: auto; } }
</style>
