<script setup lang="ts">
import { computed, ref } from 'vue'

const intercept = ref(-2)
const slope = ref(.8)
const sigmoid = (score:number) => 1/(1+Math.exp(-score))
const probability = (x:number) => sigmoid(intercept.value+slope.value*x)
const px = (x:number) => 32+x/8*314
const py = (p:number) => 180-p*150
const curve = computed(()=>Array.from({length:81},(_,i)=>{const x=i/10;return `${i?'L':'M'} ${px(x)} ${py(probability(x))}`}).join(' '))
const boundary = computed(()=>slope.value===0?null:-intercept.value/slope.value)
const points=[{x:1,y:0},{x:2,y:0},{x:3,y:0},{x:3.8,y:1},{x:4.5,y:0},{x:5,y:1},{x:6,y:1},{x:7,y:1}]
</script>

<template>
  <section class="logit-lab" aria-label="逻辑回归概率曲线实验">
    <header><div><span>INTERACTIVE 22</span><h3>调整系数，把线性分数压进 0–1 概率</h3></div><strong>边界 x={{ boundary?.toFixed(2) ?? '无' }}</strong></header>
    <div class="logit-lab__grid"><svg viewBox="0 0 376 205" role="img" aria-label="Sigmoid 概率曲线和分类样本"><line x1="32" y1="105" x2="346" y2="105"/><path :d="curve"/><circle v-for="(point,index) in points" :key="index" :cx="px(point.x)" :cy="py(point.y)+ (point.y?5:-5)" r="5"/><text x="4" y="35">1.0</text><text x="4" y="108">0.5</text><text x="4" y="184">0.0</text></svg><div><label>截距 β₀ {{ intercept.toFixed(1) }}<input v-model.number="intercept" type="range" min="-6" max="3" step="0.2"></label><label>斜率 β₁ {{ slope.toFixed(1) }}<input v-model.number="slope" type="range" min="-1.5" max="2" step="0.1"></label><code>p(y=1|x)=σ({{ intercept.toFixed(1) }}+{{ slope.toFixed(1) }}x)</code><p>x=5 时，预测概率 {{ probability(5).toFixed(3) }}</p></div></div>
  </section>
</template>

<style scoped>
.logit-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.logit-lab header{display:flex;justify-content:space-between;gap:16px}.logit-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.logit-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.logit-lab header strong{color:#f3c56f;font:700 11px ui-monospace,monospace}.logit-lab__grid{display:grid;grid-template-columns:1.4fr .8fr;gap:20px;margin-top:20px;align-items:center}.logit-lab svg{width:100%;border-radius:9px;background:#0c2420}.logit-lab svg line{stroke:#56776e;stroke-width:1;stroke-dasharray:4 3}.logit-lab svg path{fill:none;stroke:#f3c56f;stroke-width:3}.logit-lab svg circle{fill:#df765a;stroke:#fff;stroke-width:1}.logit-lab svg text{fill:#8eaaa1;font:8px ui-monospace,monospace}.logit-lab label{display:block;margin-bottom:15px;color:#bbd0c9;font-size:11px}.logit-lab input{width:100%;margin-top:7px;accent-color:#f3c56f}.logit-lab code{display:block;padding:9px;border-radius:7px;color:#f3c56f;background:#0c2420;font-size:9px}.logit-lab p{color:#a9c0b8;font-size:11px}@media(max-width:680px){.logit-lab header{flex-direction:column}.logit-lab__grid{grid-template-columns:1fr}}
</style>
