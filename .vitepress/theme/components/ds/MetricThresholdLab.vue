<script setup lang="ts">
import { computed, ref } from 'vue'

const threshold=ref(.5)
const rows=[{p:.94,y:1},{p:.82,y:1},{p:.76,y:0},{p:.67,y:1},{p:.58,y:0},{p:.49,y:1},{p:.41,y:0},{p:.33,y:0},{p:.22,y:1},{p:.12,y:0}]
const counts=computed(()=>{let tp=0,fp=0,tn=0,fn=0;rows.forEach(r=>{const pred=r.p>=threshold.value?1:0;if(pred&&r.y)tp++;else if(pred&&!r.y)fp++;else if(!pred&&!r.y)tn++;else fn++});return{tp,fp,tn,fn}})
const safe=(a:number,b:number)=>b? a/b:0
const precision=computed(()=>safe(counts.value.tp,counts.value.tp+counts.value.fp))
const recall=computed(()=>safe(counts.value.tp,counts.value.tp+counts.value.fn))
const specificity=computed(()=>safe(counts.value.tn,counts.value.tn+counts.value.fp))
</script>

<template>
  <section class="metric-lab" aria-label="分类阈值与混淆矩阵实验">
    <header><div><span>INTERACTIVE 23</span><h3>移动阈值，把概率排序变成不同的错误组合</h3></div><code>threshold={{ threshold.toFixed(2) }}</code></header>
    <label>判为正类的最低概率<input v-model.number="threshold" type="range" min="0.1" max="0.9" step="0.01"></label>
    <div class="metric-lab__grid"><div class="matrix"><i></i><b>预测正</b><b>预测负</b><b>实际正</b><strong>{{ counts.tp }}<small>TP</small></strong><strong>{{ counts.fn }}<small>FN</small></strong><b>实际负</b><strong>{{ counts.fp }}<small>FP</small></strong><strong>{{ counts.tn }}<small>TN</small></strong></div><div class="metrics"><span>Precision <b>{{ precision.toFixed(2) }}</b></span><span>Recall <b>{{ recall.toFixed(2) }}</b></span><span>Specificity <b>{{ specificity.toFixed(2) }}</b></span><span>Accuracy <b>{{ ((counts.tp+counts.tn)/rows.length).toFixed(2) }}</b></span></div></div>
    <div class="metric-lab__scores"><i v-for="(row,index) in rows" :key="index" :class="{positive:row.y===1}" :style="{left:`${row.p*100}%`}" :title="`实际${row.y ? '正' : '负'}类，概率 ${row.p}`"><small>{{ row.y ? '+' : '−' }} {{ row.p }}</small></i><b :style="{left:`${threshold*100}%`}"></b></div>
  </section>
</template>

<style scoped>
.metric-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.metric-lab header{display:flex;justify-content:space-between;gap:16px}.metric-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.metric-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.metric-lab header code{align-self:start;padding:8px 10px;border-radius:7px;color:#17302b;background:#f3c56f}.metric-lab>label{display:block;margin:20px 0;color:#bbd0c9;font-size:11px}.metric-lab input{width:100%;margin-top:8px;accent-color:#f3c56f}.metric-lab__grid{display:grid;grid-template-columns:1fr .8fr;gap:14px}.matrix{display:grid;grid-template-columns:58px 1fr 1fr;gap:4px}.matrix>*{display:flex;align-items:center;justify-content:center;min-height:31px;border-radius:5px;background:#0c2420}.matrix>b{color:#8eaaa1;font-size:9px}.matrix strong{color:#f3c56f;font-size:18px}.matrix strong small{margin-left:5px;color:#8eaaa1;font-size:8px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:5px}.metrics span{padding:10px;border-radius:5px;color:#8eaaa1;background:#0c2420;font-size:9px}.metrics b{display:block;margin-top:4px;color:#fff;font:15px ui-monospace,monospace}.metric-lab__scores{position:relative;height:58px;margin:18px 6px 0;border-bottom:2px solid #55766d}.metric-lab__scores i{position:absolute;bottom:-6px;width:11px;height:11px;border-radius:2px;transform:translateX(-50%) rotate(45deg);background:#739e93}.metric-lab__scores i.positive{border-radius:50%;transform:translateX(-50%);background:#df765a}.metric-lab__scores i small{position:absolute;bottom:15px;left:50%;transform:translateX(-50%) rotate(-45deg);color:#8eaaa1;font:7px ui-monospace,monospace;white-space:nowrap}.metric-lab__scores i.positive small{transform:translateX(-50%)}.metric-lab__scores>b{position:absolute;top:0;bottom:0;border-left:2px dashed #f3c56f}@media(max-width:650px){.metric-lab header{flex-direction:column}.metric-lab__grid{grid-template-columns:1fr}}
</style>
