<script setup lang="ts">
import { computed, ref } from 'vue'

const rows = ref(2)
const columns = ref(3)
const cells = computed(() => Array.from({ length: rows.value * columns.value }, (_, index) => index + 1))
const kind = computed(() => rows.value === 1 || columns.value === 1 ? '一维视图' : '二维矩阵')
</script>

<template>
  <section class="shape-lab" aria-label="数组形状实验">
    <header>
      <div><span>INTERACTIVE 00</span><h3>形状不是外观，而是计算契约</h3></div>
      <output>shape = ({{ rows }}, {{ columns }})</output>
    </header>
    <div class="shape-lab__controls">
      <label>行数 {{ rows }}<input v-model.number="rows" type="range" min="1" max="4"></label>
      <label>列数 {{ columns }}<input v-model.number="columns" type="range" min="1" max="5"></label>
    </div>
    <div class="shape-lab__stage">
      <div class="shape-lab__grid" :style="{ gridTemplateColumns: `repeat(${columns}, 46px)` }">
        <span v-for="cell in cells" :key="cell">{{ cell }}</span>
      </div>
      <div class="shape-lab__readout">
        <b>{{ kind }}</b>
        <p>共有 {{ rows * columns }} 个值。沿行观察是 {{ rows }} 组记录，沿列观察是 {{ columns }} 个特征。</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.shape-lab { margin: 32px 0; padding: 24px; border-radius: 16px; color: #eaf3ef; background: #112f2a; }
.shape-lab header { display: flex; align-items: start; justify-content: space-between; gap: 20px; }
.shape-lab header span { color: #efbc5c; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .1em; }
.shape-lab h3 { margin: 7px 0 0; border: 0; color: #fff; font-size: 19px; }
.shape-lab output { padding: 8px 10px; border-radius: 7px; color: #132d28; background: #f3c56f; font: 800 12px/1 ui-monospace, monospace; }
.shape-lab__controls { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 24px 0; }
.shape-lab__controls label { color: #bed2ca; font-size: 12px; }
.shape-lab__controls input { display: block; width: 100%; margin-top: 8px; accent-color: #efbc5c; }
.shape-lab__stage { display: grid; grid-template-columns: minmax(230px, 1fr) 1fr; align-items: center; gap: 28px; padding: 22px; border-radius: 12px; background: #1c4039; }
.shape-lab__grid { display: grid; gap: 5px; justify-content: center; }
.shape-lab__grid span { display: grid; place-items: center; width: 46px; height: 38px; border: 1px solid #6a9286; border-radius: 5px; color: #fff; background: #285149; font: 700 12px/1 ui-monospace, monospace; }
.shape-lab__readout b { color: #f3c56f; font-size: 14px; }
.shape-lab__readout p { margin: 8px 0 0; color: #d4e3dd; font-size: 12px; line-height: 1.7; }
@media (max-width: 640px) { .shape-lab header, .shape-lab__stage { align-items: stretch; grid-template-columns: 1fr; flex-direction: column; } .shape-lab__controls { grid-template-columns: 1fr; } }
</style>
