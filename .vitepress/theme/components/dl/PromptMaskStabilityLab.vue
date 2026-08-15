<script setup lang="ts">
import { computed, ref } from 'vue'
const positives=ref(1), negatives=ref(0), boxTightness=ref(.6), ambiguity=ref(.35)
const promptEvidence=computed(()=>positives.value*1.2+negatives.value*.8+boxTightness.value*2)
const stability=computed(()=>Math.max(0,Math.min(1,.45+promptEvidence.value*.09-ambiguity.value*.55)))
const maskRadius=computed(()=>58+ambiguity.value*55-positives.value*3+boxTightness.value*8)
</script>
<template><DlExperimentFrame index="09-B" title="可提示掩码稳定性探针" question="点、负点和框是标签，还是对多个合理掩码施加的条件？" formula="mask = decoder(image_embedding, prompt_tokens)">
<template #visual><svg viewBox="0 0 540 280" role="img" aria-label="点和框提示影响候选掩码"><path d="M115 226 C76 170 108 76 208 62 C305 49 429 91 425 174 C422 241 315 236 246 222Z" class="object"/><ellipse cx="270" cy="148" :rx="maskRadius" :ry="maskRadius*.63" class="mask"/><rect :x="175+(1-boxTightness)*55" :y="82+(1-boxTightness)*35" :width="190-(1-boxTightness)*110" :height="135-(1-boxTightness)*70" class="box"/><circle v-for="i in positives" :key="'p'+i" :cx="210+i*27" :cy="139+(i%2)*20" r="7" class="pos"/><circle v-for="i in negatives" :key="'n'+i" :cx="355-i*19" :cy="123+(i%2)*31" r="7" class="neg"/></svg></template>
<template #controls><label>正点数量 = {{positives}}<input v-model.number="positives" type="range" min="1" max="5" step="1"></label><label>负点数量 = {{negatives}}<input v-model.number="negatives" type="range" min="0" max="4" step="1"></label><label>框紧致度 = {{boxTightness.toFixed(2)}}<input v-model.number="boxTightness" type="range" min="0" max="1" step="0.05"></label><label>边界歧义 = {{ambiguity.toFixed(2)}}<input v-model.number="ambiguity" type="range" min="0" max="1" step="0.05"></label></template>
<template #metrics><div><span>提示证据</span><strong>{{promptEvidence.toFixed(2)}}</strong></div><div><span>稳定性代理</span><strong>{{stability.toFixed(3)}}</strong></div><div><span>候选掩码尺度</span><strong>{{maskRadius.toFixed(0)}} px</strong></div><div><span>提示 token</span><strong>{{positives+negatives+2}}</strong></div></template>
<template #note>这是提示敏感性的教学模拟器，不是 SAM 推理实现。SAM 的价值是可提示分割与大规模 mask 数据引擎；部署时仍需测量领域偏移、小目标、细长结构、提示策略和候选选择稳定性。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.object{fill:#273e51;stroke:#557383}.mask{fill:rgba(88,211,200,.45);stroke:#58d3c8;stroke-width:3}.box{fill:none;stroke:#f2c14e;stroke-width:2;stroke-dasharray:7 5}.pos{fill:#5ee8a0;stroke:#fff}.neg{fill:#df6c5b;stroke:#fff}</style>
