<script setup lang="ts">
import { computed, ref } from 'vue'

const count=ref(1)
const layers=[
  [1,1,1,1,1,.8,.8,.8,.8,.8,.4,.4,.4,.4,.4,0,0,0,0,0,-.3,-.3,-.3,-.3,-.3],
  [.5,.3,0,-.3,-.5,.5,.3,0,-.3,-.5,.5,.3,0,-.3,-.5,.5,.3,0,-.3,-.5,.5,.3,0,-.3,-.5],
  [0,.3,.6,.3,0,-.2,.1,.4,.1,-.2,-.4,-.1,.2,-.1,-.4,-.2,.1,.4,.1,-.2,0,.3,.6,.3,0]
]
const reconstructed=computed(()=>layers[0].map((_,index)=>layers.slice(0,count.value).reduce((sum,layer)=>sum+layer[index],0)))
const retained=[62,86,100]
const cellStyle=(value:number)=>({background:value>=0?`rgba(243,197,111,${Math.min(.95,.16+Math.abs(value)*.55)})`:`rgba(223,118,90,${Math.min(.95,.16+Math.abs(value)*.7)})`})
</script>

<template>
  <section class="svd-lab" aria-label="SVD 低秩重构实验">
    <header><div><span>INTERACTIVE 25</span><h3>逐层加入 rank-1 图案，重构原矩阵</h3></div><code>k={{ count }}</code></header>
    <label>保留主成分数<input v-model.number="count" type="range" min="1" max="3" step="1"></label>
    <div class="svd-lab__formula"><b>X</b><span>≈</span><i v-for="index in count" :key="index"><em>σ{{ index }}</em>u{{ index }}v{{ index }}ᵀ</i></div>
    <div class="svd-lab__matrix"><i v-for="(value,index) in reconstructed" :key="index" :style="cellStyle(value)"></i></div>
    <div class="svd-lab__stats"><span>保留方差 <b>{{ retained[count-1] }}%</b></span><span>压缩参数 <b>{{ count * (5+5+1) }}</b> / 25*</span><small>*这里只用于展示分层思想，实际存储还取决于矩阵尺寸与格式。</small></div>
  </section>
</template>

<style scoped>
.svd-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.svd-lab header{display:flex;justify-content:space-between;gap:16px}.svd-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.svd-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.svd-lab header code{align-self:start;padding:8px 10px;border-radius:7px;color:#17302b;background:#f3c56f}.svd-lab>label{display:block;margin:20px 0;color:#bbd0c9;font-size:11px}.svd-lab input{width:100%;margin-top:8px;accent-color:#f3c56f}.svd-lab__formula{display:flex;justify-content:center;align-items:center;gap:9px;margin:12px 0 18px}.svd-lab__formula>b{font-size:26px}.svd-lab__formula span{color:#8ea79f}.svd-lab__formula i{padding:7px;border-radius:5px;color:#c7d9d3;background:#0c2420;font:10px ui-monospace,monospace;font-style:normal}.svd-lab__formula em{color:#f3c56f;font-style:normal}.svd-lab__matrix{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;width:min(260px,100%);margin:auto;padding:10px;border-radius:9px;background:#0c2420}.svd-lab__matrix i{aspect-ratio:1;border-radius:3px}.svd-lab__stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:17px}.svd-lab__stats span{padding:10px;border-radius:6px;color:#8fa9a0;background:#0c2420;font-size:10px}.svd-lab__stats b{float:right;color:#f3c56f}.svd-lab__stats small{grid-column:1/-1;color:#819d94;font-size:8px}@media(max-width:580px){.svd-lab header{flex-direction:column}.svd-lab__formula{flex-wrap:wrap}}
</style>
