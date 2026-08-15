<script setup lang="ts">
import { computed, ref } from 'vue'
const seed=ref(4), blocks=ref(4), base=ref(512), kernel=ref(4)
const stages=computed(()=>Array.from({length:blocks.value+1},(_,i)=>({size:seed.value*2**i,channels:Math.max(3,Math.round(base.value/2**i))})))
const params=computed(()=>stages.value.slice(1).reduce((sum,s,i)=>sum+kernel.value**2*stages.value[i].channels*s.channels,0))
</script>
<template><DlExperimentFrame index="12-B" title="DCGAN 形状升频台" question="转置卷积逐级翻倍空间尺寸时，通道、参数与最终分辨率怎样联动？" formula="Hout=(Hin−1)s−2p+k,  s=2, p=1, k=4">
<template #visual><div class="stages" role="img" aria-label="DCGAN 生成器逐级放大特征图"><div v-for="(stage,i) in stages" :key="i" :style="{width:`${44+i*18}px`,height:`${44+i*18}px`,borderColor:i===stages.length-1?'#f2c14e':'#a88ce8'}"><b>{{stage.size}}²</b><small>{{stage.channels}}ch</small></div></div></template>
<template #controls><label>种子特征边长 = {{seed}}<input v-model.number="seed" type="range" min="2" max="8" step="2"></label><label>上采样 block = {{blocks}}<input v-model.number="blocks" type="range" min="2" max="6" step="1"></label><label>起始通道 = {{base}}<input v-model.number="base" type="range" min="128" max="1024" step="128"></label><label>卷积核 = {{kernel}}<input v-model.number="kernel" type="range" min="2" max="6" step="2"></label></template>
<template #metrics><div><span>最终分辨率</span><strong>{{stages.at(-1)?.size}}×{{stages.at(-1)?.size}}</strong></div><div><span>输出通道</span><strong>{{stages.at(-1)?.channels}}</strong></div><div><span>卷积参数代理</span><strong>{{(params/1e6).toFixed(2)}} M</strong></div><div><span>空间放大</span><strong>{{2**blocks}}× / 边</strong></div></template>
<template #note>DCGAN 的规则来自稳定经验：取消 pooling、用 stride conv/transposed conv，生成器多用 BN+ReLU、输出 tanh，判别器用 LeakyReLU。输出公式必须逐层断言，不能靠裁剪凑尺寸。</template>
</DlExperimentFrame></template>
<style scoped>.stages{display:flex;align-items:center;justify-content:center;gap:12px;min-height:280px;border:1px solid #493c68;border-radius:7px;background:#0b172a;overflow:hidden}.stages div{display:grid;place-items:center;flex:none;border:2px solid #a88ce8;background:rgba(168,140,232,.1);transform:rotate(-4deg)}b,small{color:#e7e0f7;font:10px ui-monospace,monospace}</style>
