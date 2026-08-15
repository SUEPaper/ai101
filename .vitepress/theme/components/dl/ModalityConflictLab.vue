<script setup lang="ts">
import { computed, ref } from 'vue'
const visual=ref(.75), text=ref(.8), shortcut=ref(.25), imageEnabled=ref(true)
const v=computed(()=>imageEnabled.value?visual.value*(1-shortcut.value):0), gap=computed(()=>Math.abs(v.value-text.value))
const answer=computed(()=>!imageEnabled.value?'文本先验':gap.value<.15?'证据不确定':v.value>text.value?'视觉证据':'文本先验')
</script>
<template><DlExperimentFrame index="27-B" title="视觉证据—文本捷径冲突台" question="图片与问题暗示矛盾时，模型是在看图，还是沿语言先验作答？" formula="decision=f(visual evidence, text prior, shortcut)">
<template #visual><div class="modal"><div :style="{opacity:imageEnabled?1:.2}">IMAGE · {{visual.toFixed(2)}}</div><div>TEXT · {{text.toFixed(2)}}</div></div></template>
<template #controls><label>视觉证据 = {{visual.toFixed(2)}}<input v-model.number="visual" type="range" min="0" max="1" step=".05"></label><label>文本先验 = {{text.toFixed(2)}}<input v-model.number="text" type="range" min="0" max="1" step=".05"></label><label>捷径强度 = {{shortcut.toFixed(2)}}<input v-model.number="shortcut" type="range" min="0" max=".8" step=".05"></label><button type="button" @click="imageEnabled=!imageEnabled">{{imageEnabled?'遮掉图片':'恢复图片'}}</button></template>
<template #metrics><div><span>有效视觉证据</span><strong>{{v.toFixed(3)}}</strong></div><div><span>模态冲突</span><strong>{{gap.toFixed(3)}}</strong></div><div><span>主导答案</span><strong>{{answer}}</strong></div><div><span>图像消融</span><strong>{{imageEnabled?'关闭':'开启'}}</strong></div></template>
<template #note>数值是教学响应。视觉 grounding 要用图像遮挡、问题改写、反事实图片和 OCR/位置切片验证，不能从流畅描述推断模型看懂。</template>
</DlExperimentFrame></template>
<style scoped>.modal{display:grid;grid-template-columns:1fr 1fr;align-items:center;gap:18px;min-height:280px;padding:24px;border:1px solid #315765;border-radius:7px;background:#0b172a}.modal div{padding:50px 10px;text-align:center;border:1px solid #58d3c8;color:#dff}.modal div+div{border-color:#a88ce8;color:#e5dcff}</style>
