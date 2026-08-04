<script setup lang="ts">
import { computed, ref } from 'vue'

const rows = [
  { index: 'S01', station: '湖畔', district: '北区', rides: 128 },
  { index: 'S02', station: '剧场', district: '南区', rides: 94 },
  { index: 'S03', station: '大学城', district: '北区', rides: 173 },
  { index: 'S04', station: '体育馆', district: '西区', rides: 81 }
]

const queries = [
  { code: "df['rides']", note: '按列名取出一列，结果是 Series。', columns: ['rides'], indexes: rows.map((row) => row.index) },
  { code: "df.loc['S02']", note: '按索引标签取出一行，结果是 Series。', columns: ['station', 'district', 'rides'], indexes: ['S02'] },
  { code: 'df.iloc[1:3]', note: '按位置取第 2 到第 3 行，切片右端不包含。', columns: ['station', 'district', 'rides'], indexes: ['S02', 'S03'] },
  { code: "df.loc[['S01', 'S04'], ['station', 'rides']]", note: '同时指定行标签和列标签，结果仍是 DataFrame。', columns: ['station', 'rides'], indexes: ['S01', 'S04'] }
]

const selected = ref(0)
const activeQuery = computed(() => queries[selected.value])
const visibleRows = computed(() => rows.filter((row) => activeQuery.value.indexes.includes(row.index)))
</script>

<template>
  <section class="frame-lab" aria-label="DataFrame 选择实验">
    <header><div><span>INTERACTIVE 02</span><h3>一句选择表达式，究竟拿走了什么？</h3></div><code>{{ activeQuery.code }}</code></header>
    <div class="frame-lab__queries">
      <button v-for="(query, index) in queries" :key="query.code" type="button" :class="{ 'is-active': selected === index }" @click="selected = index">{{ query.code }}</button>
    </div>
    <div class="frame-lab__canvas">
      <table>
        <thead><tr><th>index</th><th v-for="column in ['station', 'district', 'rides']" :key="column" :class="{ 'is-hit': activeQuery.columns.includes(column) }">{{ column }}</th></tr></thead>
        <tbody><tr v-for="row in rows" :key="row.index" :class="{ 'is-muted': !activeQuery.indexes.includes(row.index) }"><th :class="{ 'is-hit': activeQuery.indexes.includes(row.index) }">{{ row.index }}</th><td v-for="column in ['station', 'district', 'rides']" :key="column" :class="{ 'is-hit': activeQuery.indexes.includes(row.index) && activeQuery.columns.includes(column) }">{{ row[column as keyof typeof row] }}</td></tr></tbody>
      </table>
      <div class="frame-lab__result"><small>返回结果</small><b>{{ visibleRows.length }} 行 × {{ activeQuery.columns.length }} 列</b><p>{{ activeQuery.note }}</p></div>
    </div>
  </section>
</template>

<style scoped>
.frame-lab { margin: 32px 0; padding: 25px; border-radius: 16px; color: #edf4f1; background: #122f2a; overflow: hidden; }
.frame-lab header { display: flex; align-items: start; justify-content: space-between; gap: 18px; }
.frame-lab header span { color: #f3c56f; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .12em; }
.frame-lab h3 { margin: 7px 0 0; border: 0; color: #fff; font-size: 19px; }
.frame-lab header code { padding: 9px 11px; border-radius: 7px; color: #192f2b; background: #f3c56f; font-size: 11px; }
.frame-lab__queries { display: flex; flex-wrap: wrap; gap: 7px; margin: 22px 0; }
.frame-lab__queries button { padding: 8px 10px; border: 1px solid #527269; border-radius: 7px; color: #c9dcd5; background: transparent; font: 650 10px/1.2 ui-monospace, monospace; cursor: pointer; }
.frame-lab__queries button.is-active { border-color: #fff; color: #17302b; background: #fff; }
.frame-lab__canvas { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(180px, .7fr); gap: 18px; align-items: center; }
.frame-lab table { width: 100%; margin: 0; border-collapse: collapse; font-size: 11px; }
.frame-lab th, .frame-lab td { padding: 8px; border: 1px solid #45665d; color: #dce8e3; text-align: left; }
.frame-lab th { color: #91afa5; font: 750 10px/1.2 ui-monospace, monospace; }
.frame-lab .is-muted { opacity: .28; }
.frame-lab .is-hit { color: #fff; background: #2b665a; opacity: 1; }
.frame-lab__result { padding: 16px; border-radius: 10px; color: #213a35; background: #edf3ef; }
.frame-lab__result small, .frame-lab__result b { display: block; }
.frame-lab__result small { color: #6a7974; font-size: 10px; }
.frame-lab__result b { margin-top: 6px; color: #b36137; font-size: 17px; }
.frame-lab__result p { margin: 10px 0 0; font-size: 12px; line-height: 1.65; }
@media (max-width: 720px) { .frame-lab header { align-items: stretch; flex-direction: column; } .frame-lab__canvas { grid-template-columns: 1fr; overflow-x: auto; } }
</style>
