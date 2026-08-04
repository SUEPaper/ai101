<script setup lang="ts">
import { ref } from 'vue'

const checks = [
  { key: 'duplicate', label: '唯一性', question: '同一站点时刻是否重复？', rows: [1, 2], finding: 'S02 在 18:00 出现两次，汇总会重复计算。', action: '先核对事件 ID 与数据来源，不能直接随意删一行。' },
  { key: 'range', label: '取值范围', question: '车辆数是否可能为负？', rows: [3], finding: 'S03 的 bikes_available = -1，不符合物理约束。', action: '回查传感器状态；无法恢复时标记缺失并保留质量标记。' },
  { key: 'missing', label: '完整性', question: '关键字段是否缺失？', rows: [4], finding: '最后一行缺少 station_id，无法关联站点信息。', action: '尝试从设备 ID 或原始日志恢复，失败后从站点级分析中排除。' },
  { key: 'time', label: '格式与语义', question: '时间能否统一解析？', rows: [5], finding: '18点05分不是统一的 HH:MM 格式。', action: '显式指定解析规则并验证转换失败的记录。' }
]

const rows = [
  ['S01', '18:00', '4'],
  ['S02', '18:00', '2'],
  ['S02', '18:00', '2'],
  ['S03', '18:03', '-1'],
  ['—', '18:04', '6'],
  ['S04', '18点05分', '3']
]

const active = ref(0)
</script>

<template>
  <section class="quality-lab" aria-label="数据质量检查实验">
    <header><span>INTERACTIVE 05</span><h3>不要急着修：先说明违反了哪条约束</h3></header>
    <div class="quality-lab__checks">
      <button v-for="(check, index) in checks" :key="check.key" type="button" :class="{ 'is-active': active === index }" @click="active = index">{{ check.label }}</button>
    </div>
    <div class="quality-lab__body">
      <table><thead><tr><th>station_id</th><th>time</th><th>bikes</th></tr></thead><tbody><tr v-for="(row, index) in rows" :key="index" :class="{ 'is-hit': checks[active].rows.includes(index) }"><td v-for="cell in row" :key="cell">{{ cell }}</td></tr></tbody></table>
      <div><small>{{ checks[active].question }}</small><strong>{{ checks[active].finding }}</strong><p>{{ checks[active].action }}</p></div>
    </div>
  </section>
</template>

<style scoped>
.quality-lab { margin: 32px 0; padding: 25px; border: 1px solid #d4dfd9; border-radius: 16px; background: #fff; }
.quality-lab header span { color: #b66c28; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .12em; }
.quality-lab h3 { margin: 7px 0 18px; border: 0; color: #17352f; font-size: 20px; }
.quality-lab__checks { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 18px; }
.quality-lab__checks button { padding: 8px 11px; border: 1px solid #cbd8d2; border-radius: 8px; color: #52635d; background: #f4f7f5; font-size: 11px; font-weight: 750; cursor: pointer; }
.quality-lab__checks button.is-active { color: #fff; background: #143b37; border-color: #143b37; }
.quality-lab__body { display: grid; grid-template-columns: 1.25fr .9fr; gap: 18px; align-items: center; }
.quality-lab table { width: 100%; margin: 0; border-collapse: collapse; font-size: 11px; }
.quality-lab th, .quality-lab td { padding: 8px; border: 1px solid #d8e2dd; text-align: left; }
.quality-lab th { color: #6b7a75; background: #edf3ef; font: 750 10px/1.2 ui-monospace, monospace; }
.quality-lab tr.is-hit td { color: #7e3828; background: #ffe8df; font-weight: 750; }
.quality-lab__body > div { padding: 18px; border-radius: 10px; background: #f1f5f2; }
.quality-lab__body small, .quality-lab__body strong { display: block; }
.quality-lab__body small { color: #71807b; font-size: 10px; }
.quality-lab__body strong { margin-top: 8px; color: #8b422e; font-size: 14px; line-height: 1.5; }
.quality-lab__body p { margin: 10px 0 0; color: #50605a; font-size: 12px; line-height: 1.65; }
@media (max-width: 680px) { .quality-lab__body { grid-template-columns: 1fr; overflow-x: auto; } }
</style>
