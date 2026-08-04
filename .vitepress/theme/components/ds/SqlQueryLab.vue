<script setup lang="ts">
import { computed, ref } from 'vue'

const rows = [
  { station: '湖畔', district: '北区', rides: 128 }, { station: '剧场', district: '南区', rides: 94 },
  { station: '大学城', district: '北区', rides: 173 }, { station: '体育馆', district: '西区', rides: 81 },
  { station: '书城', district: '南区', rides: 116 }, { station: '港口', district: '东区', rides: 142 }
]
const queries = [
  { label: '筛选与排序', sql: 'SELECT station, rides\nFROM station_day\nWHERE rides >= 120\nORDER BY rides DESC;', run: () => rows.filter(r => r.rides >= 120).sort((a,b)=>b.rides-a.rides).map(({station,rides})=>({station,rides})) },
  { label: '计算新列', sql: 'SELECT station, rides,\n       ROUND(rides / 24.0, 1) AS rides_per_hour\nFROM station_day;', run: () => rows.map(r=>({station:r.station,rides:r.rides,rides_per_hour:(r.rides/24).toFixed(1)})) },
  { label: '限制行数', sql: 'SELECT *\nFROM station_day\nORDER BY rides DESC\nLIMIT 3;', run: () => [...rows].sort((a,b)=>b.rides-a.rides).slice(0,3) }
]
const selected = ref(0)
const result = computed(() => queries[selected.value].run())
const columns = computed(() => Object.keys(result.value[0] ?? {}))
</script>

<template>
  <section class="sql-lab" aria-label="SQL 单表查询实验">
    <header><div><span>INTERACTIVE 20</span><h3>选择查询，观察子句怎样改变结果表</h3></div><strong>{{ result.length }} rows</strong></header>
    <div class="sql-lab__buttons"><button v-for="(query,index) in queries" :key="query.label" :class="{active:selected===index}" type="button" @click="selected=index">{{ query.label }}</button></div>
    <div class="sql-lab__grid"><pre>{{ queries[selected].sql }}</pre><div class="table-wrap"><table><thead><tr><th v-for="column in columns" :key="column">{{ column }}</th></tr></thead><tbody><tr v-for="(row,index) in result" :key="index"><td v-for="column in columns" :key="column">{{ row[column as keyof typeof row] }}</td></tr></tbody></table></div></div>
  </section>
</template>

<style scoped>
.sql-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.sql-lab header{display:flex;justify-content:space-between;gap:16px}.sql-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.sql-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.sql-lab header strong{color:#f3c56f;font:700 11px ui-monospace,monospace}.sql-lab__buttons{display:flex;gap:7px;margin:20px 0}.sql-lab button{padding:8px 10px;border:1px solid #55766c;border-radius:7px;color:#b9cec7;background:transparent;font-size:10px;cursor:pointer}.sql-lab button.active{color:#17302b;background:#f3c56f}.sql-lab__grid{display:grid;grid-template-columns:.85fr 1.15fr;gap:14px}.sql-lab pre{margin:0;padding:15px;border-radius:9px;color:#f3c56f;background:#0c2420;font:10px/1.7 ui-monospace,monospace;white-space:pre-wrap}.sql-lab .table-wrap{overflow-x:auto}.sql-lab table{width:100%;margin:0;border-collapse:collapse;font-size:10px}.sql-lab th,.sql-lab td{padding:7px;border:1px solid #486a61;text-align:left}.sql-lab th{color:#9bb6ad}.sql-lab td{color:#e4eeea}@media(max-width:680px){.sql-lab header{flex-direction:column}.sql-lab__buttons{flex-wrap:wrap}.sql-lab__grid{grid-template-columns:1fr}}
</style>
