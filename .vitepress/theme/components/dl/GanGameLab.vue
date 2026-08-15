<script setup lang="ts">
import { computed, ref } from 'vue'
const realMean=ref(1.2), fakeMean=ref(-.6), sharpness=ref(2.5), loss=ref<'minimax'|'nonsaturating'>('nonsaturating')
const dReal=computed(()=>1/(1+Math.exp(-sharpness.value*realMean.value)))
const dFake=computed(()=>1/(1+Math.exp(-sharpness.value*fakeMean.value)))
const dLoss=computed(()=>-Math.log(Math.max(dReal.value,1e-6))-Math.log(Math.max(1-dFake.value,1e-6)))
const gLoss=computed(()=>loss.value==='minimax'?Math.log(Math.max(1-dFake.value,1e-6)):-Math.log(Math.max(dFake.value,1e-6)))
const signal=computed(()=>loss.value==='minimax'?dFake.value*(1-dFake.value):1-dFake.value)
</script>
<template><DlExperimentFrame index="12-A" title="对抗博弈相图" question="判别器太强时，原始 minimax 与非饱和生成器损失留下怎样不同的学习信号？" formula="min_G max_D E log D(x)+E log(1-D(G(z)))">
<template #visual><svg viewBox="0 0 540 270" role="img" aria-label="真实与生成分布及判别边界"><path :d="`M30 230 C${145+realMean*25} 230 ${170+realMean*25} 48 ${250+realMean*25} 230`" class="real"/><path :d="`M30 230 C${145+fakeMean*25} 230 ${170+fakeMean*25} 70 ${250+fakeMean*25} 230`" class="fake"/><line :x1="270-(realMean+fakeMean)*18" y1="35" :x2="270-(realMean+fakeMean)*18" y2="235"/><text x="95" y="35">p data</text><text x="400" y="35">p generator</text><text x="270" y="255">sample axis</text></svg></template>
<template #controls><label>真实分布中心 = {{realMean.toFixed(1)}}<input v-model.number="realMean" type="range" min="-1" max="2" step="0.1"></label><label>生成分布中心 = {{fakeMean.toFixed(1)}}<input v-model.number="fakeMean" type="range" min="-2" max="2" step="0.1"></label><label>判别器锐度 = {{sharpness.toFixed(1)}}<input v-model.number="sharpness" type="range" min=".5" max="8" step=".5"></label><label>生成器目标<select v-model="loss"><option value="nonsaturating">非饱和：−log D(G(z))</option><option value="minimax">原始 minimax</option></select></label></template>
<template #metrics><div><span>D(real)</span><strong>{{dReal.toFixed(3)}}</strong></div><div><span>D(fake)</span><strong>{{dFake.toFixed(3)}}</strong></div><div><span>判别器 loss</span><strong>{{dLoss.toFixed(3)}}</strong></div><div><span>生成器信号代理</span><strong>{{signal.toFixed(4)}} · L={{gLoss.toFixed(2)}}</strong></div></template>
<template #note>非饱和目标与原 minimax 拥有相同理想固定点，但判别器自信拒绝假样本时，前者给生成器更强信号。曲线是标量教学代理，不能替代高维训练诊断。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #493c68;border-radius:7px;background:#0b172a}.real,.fake{fill:none;stroke-width:5}.real{stroke:#58d3c8}.fake{stroke:#a88ce8}line{stroke:#df6c5b;stroke-width:2;stroke-dasharray:6 4}text{fill:#dce8ea;font:10px ui-monospace,monospace;text-anchor:middle}</style>
