<script setup lang="ts">
import { computed, ref } from 'vue'

const stations = [{id:'S01',station:'湖畔'},{id:'S02',station:'剧场'},{id:'S03',station:'大学城'}]
const repairs = [{station_id:'S01',ticket:'R7'},{station_id:'S01',ticket:'R9'},{station_id:'S04',ticket:'R12'}]
const joinType = ref<'inner'|'left'|'full'>('inner')
const selectJoin = (type: string) => {
  joinType.value = type as 'inner' | 'left' | 'full'
}
const result = computed(() => {
  const joined = stations.flatMap(station => {
    const matches = repairs.filter(repair=>repair.station_id===station.id)
    if (matches.length) return matches.map(repair=>({id:station.id,station:station.station,ticket:repair.ticket}))
    return joinType.value === 'inner' ? [] : [{id:station.id,station:station.station,ticket:'NULL'}]
  })
  if (joinType.value === 'full') repairs.filter(repair=>!stations.some(station=>station.id===repair.station_id)).forEach(repair=>joined.push({id:repair.station_id,station:'NULL',ticket:repair.ticket}))
  return joined
})
</script>

<template>
  <section class="join-lab" aria-label="SQL 连接实验">
    <header><div><span>INTERACTIVE 21</span><h3>选择 Join，追踪未匹配行是否留下</h3></div><code>{{ joinType.toUpperCase() }} JOIN</code></header>
    <div class="join-lab__buttons"><button v-for="type in ['inner','left','full']" :key="type" :class="{active:joinType===type}" type="button" @click="selectJoin(type)">{{ type }}</button></div>
    <div class="join-lab__sources"><div><small>stations</small><i v-for="row in stations" :key="row.id"><b>{{ row.id }}</b>{{ row.station }}</i></div><span>⟗</span><div><small>repairs</small><i v-for="row in repairs" :key="row.ticket"><b>{{ row.station_id }}</b>{{ row.ticket }}</i></div></div>
    <table><thead><tr><th>id</th><th>station</th><th>ticket</th></tr></thead><tbody><tr v-for="(row,index) in result" :key="index"><td>{{ row.id }}</td><td>{{ row.station }}</td><td>{{ row.ticket }}</td></tr></tbody></table>
  </section>
</template>

<style scoped>
.join-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.join-lab header{display:flex;justify-content:space-between;gap:16px}.join-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.join-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.join-lab header code{align-self:start;padding:8px 10px;border-radius:7px;color:#17302b;background:#f3c56f}.join-lab__buttons{display:flex;gap:7px;margin:20px 0}.join-lab button{padding:7px 12px;border:1px solid #57766d;border-radius:6px;color:#b9cec7;background:transparent;font:700 9px ui-monospace,monospace;cursor:pointer}.join-lab button.active{color:#17302b;background:#f3c56f}.join-lab__sources{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center}.join-lab__sources div{padding:12px;border-radius:8px;background:#0c2420}.join-lab__sources small{display:block;margin-bottom:7px;color:#8faaa2}.join-lab__sources i{display:flex;gap:8px;margin:5px 0;color:#dbe8e3;font-size:10px;font-style:normal}.join-lab__sources i b{color:#f3c56f;font-family:ui-monospace,monospace}.join-lab__sources>span{color:#df765a;font-size:22px}.join-lab table{width:100%;margin:18px 0 0;border-collapse:collapse;font-size:10px}.join-lab th,.join-lab td{padding:7px;border:1px solid #486a61;text-align:left}.join-lab th{color:#9bb6ad}.join-lab td{color:#e4eeea}@media(max-width:620px){.join-lab header{flex-direction:column}.join-lab__sources{grid-template-columns:1fr}.join-lab__sources>span{text-align:center}}
</style>
