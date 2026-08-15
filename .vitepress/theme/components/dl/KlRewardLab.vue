<script setup lang="ts">
import { computed, ref } from 'vue'
const reward=ref(1.4), kl=ref(.25), coefficient=ref(.1), verified=ref(.6)
const objective=computed(()=>reward.value+verified.value-coefficient.value*kl.value)
const drift=computed(()=>Math.min(1,kl.value*(1-coefficient.value)*.9)), risk=computed(()=>Math.min(1,drift.value*(1-verified.value/2)))
</script>
<template><DlExperimentFrame index="23-B" title="奖励—KL 推理护栏" question="RLHF/推理强化学习为何同时需要任务奖励、reference KL 与可验证信号？" formula="J≈E[r_task+r_verify−λ KL(π||πref)]">
<template #visual><div class="guard"><span :style="{height:`${Math.min(100,(reward+verified)*35)}%`}"></span><span class="kl" :style="{height:`${Math.min(100,kl*100)}%`}"></span><span class="risk" :style="{height:`${risk*100}%`}"></span></div></template>
<template #controls><label>偏好奖励 = {{reward.toFixed(2)}}<input v-model.number="reward" type="range" min="-1" max="4" step=".1"></label><label>Reference KL = {{kl.toFixed(2)}}<input v-model.number="kl" type="range" min="0" max="2" step=".05"></label><label>KL 系数 = {{coefficient.toFixed(2)}}<input v-model.number="coefficient" type="range" min="0" max="1" step=".05"></label><label>可验证奖励 = {{verified.toFixed(2)}}<input v-model.number="verified" type="range" min="0" max="2" step=".1"></label></template>
<template #metrics><div><span>正则化目标</span><strong>{{objective.toFixed(3)}}</strong></div><div><span>策略漂移代理</span><strong>{{drift.toFixed(3)}}</strong></div><div><span>Reward hacking 风险</span><strong>{{risk.toFixed(3)}}</strong></div><div><span>可自动核验</span><strong>{{verified>0?'部分':'否'}}</strong></div></template>
<template #note>图中风险是教学代理。真实 PPO/GRPO 等必须记录采样策略、advantage、clip、KL 估计、奖励归一化、长度偏差和不可验证任务切片。</template>
</DlExperimentFrame></template>
<style scoped>.guard{display:flex;align-items:end;justify-content:center;gap:42px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.guard span{width:62px;min-height:5px;background:#58d3c8}.guard .kl{background:#f2c14e}.guard .risk{background:#df6c5b}</style>
