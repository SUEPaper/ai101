<script setup lang="ts">
import { computed, ref } from 'vue'

const degree = ref(1)
const optimizer = ref<'batch' | 'mini' | 'sgd'>('batch')
const trainError = [8.8, 5.1, 3.2, 2.2, 1.5, 1.0]
const validError = [9.4, 5.7, 4.0, 4.3, 6.1, 9.2]
const optimizerNote = computed(() => ({
  batch: '每一步使用全部样本：方向稳定，单步计算最重。',
  mini: '每一步使用一小批样本：速度与稳定性的折中。',
  sgd: '每一步只看一个样本：路径抖动，但更新非常快。'
}[optimizer.value]))
const selectOptimizer = (mode: string) => {
  optimizer.value = mode as 'batch' | 'mini' | 'sgd'
}
const pathPoints = computed(() => {
  const offsets = optimizer.value === 'batch' ? [0,0,0,0,0,0] : optimizer.value === 'mini' ? [0,8,-5,4,-2,0] : [0,18,-13,12,-8,4]
  return offsets.map((offset,index) => `${35+index*55},${175-index*25+offset}`).join(' ')
})
</script>

<template>
  <section class="feature-lab" aria-label="特征工程与随机梯度下降实验">
    <header><div><span>INTERACTIVE 14</span><h3>复杂特征让训练更贴合，也让验证风险先降后升</h3></div><strong>degree={{ degree }}</strong></header>
    <div class="feature-lab__grid">
      <div class="feature-lab__features">
        <label>多项式次数 {{ degree }}<input v-model.number="degree" type="range" min="1" max="6" step="1"></label>
        <code>[x<span v-for="power in degree-1" :key="power">, x<sup>{{ power+2 }}</sup></span>]</code>
        <div class="meters"><span>训练误差 <b :style="{width: `${trainError[degree-1]*8}%`}"></b><em>{{ trainError[degree-1] }}</em></span><span>验证误差 <b :style="{width: `${validError[degree-1]*8}%`}"></b><em>{{ validError[degree-1] }}</em></span></div>
      </div>
      <div class="feature-lab__optimizer">
        <div class="buttons"><button v-for="mode in ['batch','mini','sgd']" :key="mode" :class="{active:optimizer===mode}" type="button" @click="selectOptimizer(mode)">{{ mode }}</button></div>
        <svg viewBox="0 0 330 205" role="img" aria-label="不同批量策略的优化路径"><path d="M20 185 Q155 70 315 28"/><polyline :points="pathPoints"/><circle v-for="point in pathPoints.split(' ')" :key="point" :cx="point.split(',')[0]" :cy="point.split(',')[1]" r="4"/></svg>
        <p>{{ optimizerNote }}</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.feature-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.feature-lab header{display:flex;justify-content:space-between;gap:16px}.feature-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.feature-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.feature-lab header strong{color:#f3c56f;font:700 13px ui-monospace,monospace}.feature-lab__grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:22px}.feature-lab__features,.feature-lab__optimizer{padding:17px;border-radius:10px;background:#0c2420}.feature-lab label{display:block;color:#c8dad4;font-size:12px}.feature-lab input{width:100%;margin:9px 0 17px;accent-color:#f3c56f}.feature-lab code{display:block;min-height:25px;color:#f3c56f}.meters{margin-top:22px}.meters span{position:relative;display:grid;grid-template-columns:70px 1fr 28px;gap:8px;align-items:center;margin:12px 0;color:#9eb7af;font-size:10px}.meters b{height:8px;border-radius:4px;background:#df765a}.meters span:first-child b{background:#6da99a}.meters em{color:#fff;font-style:normal}.buttons{display:flex;gap:5px}.buttons button{flex:1;padding:7px;border:1px solid #55766c;border-radius:6px;color:#b8cec6;background:transparent;font:700 9px ui-monospace,monospace;cursor:pointer}.buttons button.active{color:#17302b;background:#f3c56f}.feature-lab svg{width:100%;height:125px;margin-top:8px}.feature-lab svg path{fill:none;stroke:#45675e;stroke-width:2}.feature-lab svg polyline{fill:none;stroke:#df765a;stroke-width:2;stroke-dasharray:4 3}.feature-lab svg circle{fill:#f3c56f}.feature-lab p{margin:0;color:#9fb8b0;font-size:11px;line-height:1.6}@media(max-width:720px){.feature-lab header{flex-direction:column}.feature-lab__grid{grid-template-columns:1fr}}
</style>
