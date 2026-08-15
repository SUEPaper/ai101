<script setup lang="ts">
import { computed, ref } from 'vue'
const foreground=ref(10), miss=ref(20), falsePositive=ref(5)
const counts=computed(()=>{const p=foreground.value,n=100-p,tp=p*(1-miss.value/100),fn=p-tp,fp=n*falsePositive.value/100,tn=n-fp;return{tp,fn,fp,tn}})
const accuracy=computed(()=>(counts.value.tp+counts.value.tn)/100)
const iou=computed(()=>counts.value.tp/(counts.value.tp+counts.value.fp+counts.value.fn))
const dice=computed(()=>2*counts.value.tp/(2*counts.value.tp+counts.value.fp+counts.value.fn))
</script>
<template><DlExperimentFrame index="08-B" title="分割指标反例台" question="像素准确率很高时，前景掩码为什么仍可能不可用？" formula="IoU=TP/(TP+FP+FN), Dice=2TP/(2TP+FP+FN)">
<template #visual><div class="matrix" role="img" aria-label="分割混淆矩阵"><div><span>TP</span><b>{{counts.tp.toFixed(1)}}</b></div><div><span>FN</span><b>{{counts.fn.toFixed(1)}}</b></div><div><span>FP</span><b>{{counts.fp.toFixed(1)}}</b></div><div><span>TN</span><b>{{counts.tn.toFixed(1)}}</b></div></div></template>
<template #controls><label>真实前景占比 = {{foreground}}%<input v-model.number="foreground" type="range" min="1" max="50" step="1"></label><label>前景漏分率 = {{miss}}%<input v-model.number="miss" type="range" min="0" max="100" step="5"></label><label>背景误报率 = {{falsePositive}}%<input v-model.number="falsePositive" type="range" min="0" max="50" step="1"></label></template>
<template #metrics><div><span>Pixel accuracy</span><strong>{{(accuracy*100).toFixed(1)}}%</strong></div><div><span>前景 IoU</span><strong>{{(iou*100).toFixed(1)}}%</strong></div><div><span>前景 Dice</span><strong>{{(dice*100).toFixed(1)}}%</strong></div><div><span>前景召回率</span><strong>{{(100-miss).toFixed(0)}}%</strong></div></template>
<template #note>把前景占比调到 1%、漏分率调到 100%，准确率仍接近 99%，但 IoU 与 Dice 为 0。报告 mIoU 时还要声明类别平均、ignore label、空类和边界后处理规则。</template>
</DlExperimentFrame></template>
<style scoped>.matrix{display:grid;grid-template-columns:1fr 1fr;min-height:280px;gap:8px;padding:24px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.matrix div{display:grid;place-items:center;border:1px solid #31556a;background:#10223a}.matrix div:nth-child(1){background:#174941}.matrix div:nth-child(2),.matrix div:nth-child(3){background:#482a2d}.matrix span{color:#9bb4ba;font:10px ui-monospace,monospace}.matrix b{color:#f2c14e;font:25px ui-monospace,monospace}</style>
