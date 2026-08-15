<script setup lang="ts">
import { computed, ref } from 'vue'
const chosen=ref(-1.2), rejected=ref(-2.0), refGap=ref(.2), beta=ref(.1), noise=ref(.1)
const logit=computed(()=>beta.value*((chosen.value-rejected.value)-refGap.value)), p=computed(()=>1/(1+Math.exp(-logit.value)))
const loss=computed(()=>-(1-noise.value)*Math.log(p.value)-noise.value*Math.log(1-p.value))
</script>
<template><DlExperimentFrame index="23-A" title="DPO 偏好对账台" question="chosen/rejected 相对 reference 的 log-ratio 差怎样变成偏好概率；标签噪声与 β 如何改变梯度？" formula="σ(β[(logπc−logπr)−(logπref,c−logπref,r)])">
<template #visual><div class="pair"><div>CHOSEN<em :style="{width:`${p*100}%`}"></em></div><div>REJECTED<em :style="{width:`${(1-p)*100}%`}"></em></div></div></template>
<template #controls><label>Policy chosen logp = {{chosen.toFixed(1)}}<input v-model.number="chosen" type="range" min="-5" max="0" step=".1"></label><label>Policy rejected logp = {{rejected.toFixed(1)}}<input v-model.number="rejected" type="range" min="-5" max="0" step=".1"></label><label>β = {{beta.toFixed(2)}}<input v-model.number="beta" type="range" min=".02" max="1" step=".02"></label><label>标签噪声 = {{noise.toFixed(2)}}<input v-model.number="noise" type="range" min="0" max=".45" step=".05"></label></template>
<template #metrics><div><span>相对 margin</span><strong>{{logit.toFixed(3)}}</strong></div><div><span>偏好概率</span><strong>{{p.toFixed(3)}}</strong></div><div><span>平滑 loss</span><strong>{{loss.toFixed(3)}}</strong></div><div><span>Reference gap</span><strong>{{refGap.toFixed(2)}}</strong></div></template>
<template #note>DPO 避免显式 reward model/PPO rollout，不等于不需要 reference、偏好质量或离线分布支持。β 的语义与实现缩放需核对。</template>
</DlExperimentFrame></template>
<style scoped>.pair{display:grid;align-content:center;gap:28px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.pair div{display:grid;grid-template-columns:85px 1fr;color:#b9cbd0}.pair em{height:24px;background:#58d3c8}.pair div+div em{background:#df6c5b}</style>
