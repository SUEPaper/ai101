<script setup lang="ts">
import { computed, ref } from 'vue'
const size=ref(96), aspect=ref(1), positive=ref(.7), negative=ref(.3)
const gt={w:128,h:72}
const anchor=computed(()=>({w:size.value*Math.sqrt(aspect.value),h:size.value/Math.sqrt(aspect.value)}))
const iou=computed(()=>{const inter=Math.min(gt.w,anchor.value.w)*Math.min(gt.h,anchor.value.h);return inter/(gt.w*gt.h+anchor.value.w*anchor.value.h-inter)})
const state=computed(()=>iou.value>=positive.value?'正样本':iou.value<negative.value?'负样本':'忽略')
</script>
<template><DlExperimentFrame index="06-A" title="Anchor 匹配检验台" question="尺度、长宽比与 IoU 阈值怎样共同决定 RPN 的监督标签？" formula="IoU(A,G)=|A∩G|/|A∪G|">
<template #visual><svg viewBox="0 0 540 280" role="img" aria-label="真实框与锚框重叠示意"><rect class="gt" x="206" y="100" :width="gt.w" :height="gt.h"/><rect class="anchor" :x="270-anchor.w/2" :y="136-anchor.h/2" :width="anchor.w" :height="anchor.h"/><text x="270" y="42">center-aligned boxes</text><text x="270" y="246">{{state}} · IoU {{iou.toFixed(3)}}</text></svg></template>
<template #controls><label>anchor 尺度 = {{size}}<input v-model.number="size" type="range" min="32" max="192" step="8"></label><label>长宽比 w/h = {{aspect.toFixed(2)}}<input v-model.number="aspect" type="range" min="0.3" max="3" step="0.1"></label><label>正阈值 = {{positive.toFixed(2)}}<input v-model.number="positive" type="range" min="0.5" max="0.9" step="0.05"></label><label>负阈值 = {{negative.toFixed(2)}}<input v-model.number="negative" type="range" min="0.1" max="0.5" step="0.05"></label></template>
<template #metrics><div><span>IoU</span><strong>{{iou.toFixed(3)}}</strong></div><div><span>监督状态</span><strong>{{state}}</strong></div><div><span>anchor 宽×高</span><strong>{{anchor.w.toFixed(0)}}×{{anchor.h.toFixed(0)}}</strong></div><div><span>阈值间隔</span><strong>{{(positive-negative).toFixed(2)}}</strong></div></template>
<template #note>真实 RPN 会在多个位置、尺度和长宽比上匹配，并常强制每个 GT 至少获得一个正 anchor。阈值改变的不只是样本数量，还会改变正负难度与回归目标分布。</template>
</DlExperimentFrame></template>
<style scoped>svg{width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.gt{fill:rgba(88,211,200,.22);stroke:#58d3c8;stroke-width:3}.anchor{fill:rgba(242,193,78,.14);stroke:#f2c14e;stroke-width:3;stroke-dasharray:8 5}.anchor,.gt{transition:all .2s}text{fill:#d8e8e9;font:12px ui-monospace,monospace;text-anchor:middle}</style>
