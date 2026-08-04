<script setup lang="ts">
import { computed, ref } from 'vue'

const threshold = ref(100)
const rows = [
  { station: '湖畔', rides: 128 },
  { station: '剧场', rides: 94 },
  { station: '大学城', rides: 173 },
  { station: '体育馆', rides: 81 },
  { station: '公园东门', rides: 116 }
]
const selectedRows = computed(() => rows.filter((row) => row.rides >= threshold.value))
</script>

<template>
  <section class="mask-lab" aria-label="布尔筛选实验">
    <header><span>INTERACTIVE 03</span><h3>布尔掩码像一排闸门</h3><code>df[df['rides'] &gt;= {{ threshold }}]</code></header>
    <label>最低骑行量：<b>{{ threshold }}</b><input v-model.number="threshold" type="range" min="70" max="180" step="5"></label>
    <div class="mask-lab__rows">
      <div v-for="row in rows" :key="row.station" :class="{ 'is-open': row.rides >= threshold }">
        <span>{{ row.station }}</span><b>{{ row.rides }}</b><code>{{ row.rides >= threshold ? 'True' : 'False' }}</code>
      </div>
    </div>
    <p>当前有 <strong>{{ selectedRows.length }}</strong> 行通过闸门：{{ selectedRows.map((row) => row.station).join('、') || '没有站点' }}。</p>
  </section>
</template>

<style scoped>
.mask-lab { margin: 32px 0; padding: 25px; border: 1px solid #d6e1db; border-radius: 16px; background: #fff; }
.mask-lab header { display: grid; grid-template-columns: 1fr auto; align-items: end; gap: 4px 18px; }
.mask-lab header span { grid-column: 1 / -1; color: #b76e25; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .12em; }
.mask-lab h3 { margin: 5px 0 0; border: 0; color: #18352f; font-size: 20px; }
.mask-lab header code { color: #8b4f26; font-size: 11px; }
.mask-lab > label { display: block; margin: 24px 0 16px; color: #61706b; font-size: 12px; }
.mask-lab > label input { width: 100%; margin-top: 8px; accent-color: #2f7c7a; }
.mask-lab__rows { display: grid; gap: 6px; }
.mask-lab__rows > div { display: grid; grid-template-columns: 1fr 60px 70px; gap: 12px; align-items: center; padding: 10px 12px; border-radius: 7px; color: #8a9591; background: #f0f3f1; opacity: .55; }
.mask-lab__rows > div.is-open { color: #183a33; background: #e3f2eb; opacity: 1; }
.mask-lab__rows b, .mask-lab__rows code { font-size: 11px; }
.mask-lab > p { margin: 15px 0 0; color: #56645f; font-size: 12px; }
@media (max-width: 620px) { .mask-lab header { grid-template-columns: 1fr; } .mask-lab__rows > div { grid-template-columns: 1fr 44px 54px; } }
</style>
