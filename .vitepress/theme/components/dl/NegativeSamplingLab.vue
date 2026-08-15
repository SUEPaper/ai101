<script setup lang="ts">
import { computed, ref } from 'vue'
const positive=ref(1.4), negative=ref(-.6), negatives=ref(5), exponent=ref(.75)
const sigmoid=(x:number)=>1/(1+Math.exp(-x))
const loss=computed(()=>-Math.log(sigmoid(positive.value))-negatives.value*Math.log(sigmoid(-negative.value)))
const posGrad=computed(()=>sigmoid(positive.value)-1), negGrad=computed(()=>negatives.value*sigmoid(negative.value))
</script>
<template><DlExperimentFrame index="17-B" title="负采样梯度账本" question="Skip-gram 不对整个词表做 softmax 时，正负点积分别被怎样推动？" formula="−log σ(uₒ·vᵢ)−Σₖ log σ(−uₖ·vᵢ)">
<template #visual><div class="ledger" role="img" aria-label="正样本与负样本 sigmoid 概率"><div><span>positive</span><i :style="{width:`${sigmoid(positive)*100}%`}"></i></div><div><span>negative</span><i :style="{width:`${sigmoid(negative)*100}%`}"></i></div></div></template>
<template #controls><label>正点积 = {{positive.toFixed(2)}}<input v-model.number="positive" type="range" min="-3" max="3" step=".1"></label><label>负点积 = {{negative.toFixed(2)}}<input v-model.number="negative" type="range" min="-3" max="3" step=".1"></label><label>负样本数 K = {{negatives}}<input v-model.number="negatives" type="range" min="1" max="20" step="1"></label><label>unigram 指数 = {{exponent.toFixed(2)}}<input v-model.number="exponent" type="range" min=".3" max="1" step=".05"></label></template>
<template #metrics><div><span>目标 loss</span><strong>{{loss.toFixed(3)}}</strong></div><div><span>正点积梯度</span><strong>{{posGrad.toFixed(3)}}</strong></div><div><span>负项总梯度</span><strong>{{negGrad.toFixed(3)}}</strong></div><div><span>采样分布</span><strong>unigram^{{exponent.toFixed(2)}}</strong></div></template>
<template #note>更多负样本增强排斥也增加计算。常用 0.75 次幂降低高频词垄断，但它是语料/目标相关选择；subsampling 高频词又改变正样本分布。</template>
</DlExperimentFrame></template>
<style scoped>.ledger{display:grid;align-content:center;gap:30px;min-height:280px;padding:24px;border:1px solid #2c5f65;border-radius:7px;background:#0b172a}.ledger div{display:grid;grid-template-columns:80px 1fr;color:#dce9e9}.ledger i{height:22px;background:#58d3c8}.ledger div+div i{background:#df6c5b}</style>
