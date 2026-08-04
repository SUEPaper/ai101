<script setup lang="ts">
import { computed, ref } from 'vue'

const stages = [
  { title: '提出可行动的问题', question: '哪些站点在工作日 18:00–19:00 经常无车可借？', output: '对象、时间窗口、指标与决策动作' },
  { title: '取得并校验数据', question: '调度日志、站点容量和天气记录能否按时间对齐？', output: '字段说明、数据粒度与质量报告' },
  { title: '探索模式', question: '缺车是否集中在少数站点、天气或日期类型？', output: '分布、异常值与候选关系' },
  { title: '建模或推断', question: '哪些信息能提前预警下一小时的缺车风险？', output: '预测、区间与适用边界' },
  { title: '形成决策并复查', question: '提前调度是否真的减少了用户无车可借的次数？', output: '行动方案、监控指标与新问题' }
]

const active = ref(0)
const current = computed(() => stages[active.value])
</script>

<template>
  <section class="lifecycle-lab" aria-label="数据科学生命周期交互图">
    <header><span>INTERACTIVE 01</span><h3>沿着一次真实决策走完数据生命周期</h3></header>
    <div class="lifecycle-lab__rail">
      <button
        v-for="(stage, index) in stages"
        :key="stage.title"
        type="button"
        :class="{ 'is-active': active === index }"
        :aria-pressed="active === index"
        @click="active = index"
      ><b>{{ index + 1 }}</b><span>{{ stage.title }}</span></button>
    </div>
    <div class="lifecycle-lab__detail">
      <div><small>此阶段追问</small><p>{{ current.question }}</p></div>
      <div><small>需要留下的证据</small><p>{{ current.output }}</p></div>
    </div>
  </section>
</template>

<style scoped>
.lifecycle-lab { margin: 32px 0; padding: 26px; border: 1px solid #cbdad3; border-radius: 16px; background: #fff; box-shadow: 0 18px 50px rgba(20,59,55,.08); }
.lifecycle-lab header span { color: #b56d23; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .12em; }
.lifecycle-lab h3 { margin: 8px 0 24px; border: 0; color: #162c27; font-size: 20px; }
.lifecycle-lab__rail { display: grid; grid-template-columns: repeat(5, 1fr); border-top: 2px solid #c9d7d0; }
.lifecycle-lab__rail button { position: relative; padding: 22px 8px 10px; border: 0; color: #67736f; background: transparent; font-size: 11px; line-height: 1.45; cursor: pointer; text-align: left; }
.lifecycle-lab__rail button::before { content: ''; position: absolute; top: -7px; left: 9px; width: 12px; height: 12px; border: 2px solid #aebfb6; border-radius: 50%; background: #fff; }
.lifecycle-lab__rail button.is-active { color: #173f38; font-weight: 750; }
.lifecycle-lab__rail button.is-active::before { border-color: #d47a4d; background: #d47a4d; box-shadow: 0 0 0 5px #f8e7dd; }
.lifecycle-lab__rail b { display: block; margin-bottom: 4px; color: #b56d23; font: 800 10px/1 ui-monospace, monospace; }
.lifecycle-lab__detail { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 18px; }
.lifecycle-lab__detail > div { padding: 16px; border-radius: 10px; background: #edf3ef; }
.lifecycle-lab__detail small { color: #71807a; font-size: 10px; font-weight: 750; }
.lifecycle-lab__detail p { margin: 8px 0 0; color: #243d37; font-size: 13px; line-height: 1.65; }
@media (max-width: 700px) { .lifecycle-lab__rail { grid-template-columns: 1fr; border-top: 0; border-left: 2px solid #c9d7d0; } .lifecycle-lab__rail button { padding: 8px 12px 8px 22px; } .lifecycle-lab__rail button::before { top: 13px; left: -7px; } .lifecycle-lab__detail { grid-template-columns: 1fr; } }
</style>
