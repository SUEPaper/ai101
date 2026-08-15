<script setup lang="ts">
import { computed, ref } from 'vue'

const model = ref<'alexnet' | 'vgg16' | 'googlenet' | 'resnet50'>('resnet50')
const resolution = ref(224)
const specs = {
  alexnet: { label: 'AlexNet', params: 61.1, flops: 0.72, depth: 8, top1: 56.5, color: '#df6c5b' },
  vgg16: { label: 'VGG-16', params: 138.4, flops: 15.5, depth: 16, top1: 71.6, color: '#f2c14e' },
  googlenet: { label: 'GoogLeNet', params: 6.8, flops: 1.5, depth: 22, top1: 69.8, color: '#58d3c8' },
  resnet50: { label: 'ResNet-50', params: 25.6, flops: 4.1, depth: 50, top1: 76.1, color: '#65a9df' }
}
const active = computed(() => specs[model.value])
const scaledFlops = computed(() => active.value.flops * (resolution.value / 224) ** 2)
const activation = computed(() => scaledFlops.value * 0.31)
</script>

<template>
  <DlExperimentFrame index="04-A" title="经典架构预算台" question="同样是提升 ImageNet 表示能力，参数、计算与深度分别花在哪里？" formula="FLOPs(R) ≈ FLOPs₂₂₄ × (R / 224)²">
    <template #visual>
      <div class="budget" role="img" :aria-label="`${active.label} 参数和计算预算对比`">
        <div v-for="(spec,key) in specs" :key="key" class="row" :class="{active:key===model}">
          <span>{{ spec.label }}</span><i :style="{width:`${Math.max(6,spec.params / 1.5)}%`,background:spec.color}"></i><b>{{ spec.params }}M</b>
        </div>
        <div class="pipeline"><span v-for="index in Math.min(12,Math.ceil(active.depth/4))" :key="index" :style="{background:active.color,opacity:String(.35+index*.04)}"></span></div>
      </div>
    </template>
    <template #controls>
      <label>架构<select v-model="model"><option value="alexnet">AlexNet</option><option value="vgg16">VGG-16</option><option value="googlenet">GoogLeNet</option><option value="resnet50">ResNet-50</option></select></label>
      <label>输入分辨率 = {{ resolution }}<input v-model.number="resolution" type="range" min="160" max="384" step="16"></label>
    </template>
    <template #metrics>
      <div><span>参数量</span><strong>{{ active.params }} M</strong></div><div><span>估算计算量</span><strong>{{ scaledFlops.toFixed(2) }} GFLOPs</strong></div><div><span>名义深度</span><strong>{{ active.depth }} 层</strong></div><div><span>激活代理量</span><strong>{{ activation.toFixed(2) }} units</strong></div>
    </template>
    <template #note>柱长只编码参数量，不能当作准确率排名。VGG 的参数主要被全连接层和宽通道消耗；GoogLeNet 用分支与 1×1 降维压缩预算；ResNet 的贡献首先是可优化性，而不是“层数本身”。</template>
  </DlExperimentFrame>
</template>

<style scoped>
.budget{min-height:280px;padding:24px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.row{display:grid;grid-template-columns:84px 1fr 58px;gap:10px;align-items:center;min-height:46px;color:#93aeb5;font:11px ui-monospace,monospace}.row i{display:block;height:15px;min-width:8px;border-radius:2px;transition:width .2s}.row b{text-align:right}.row.active{color:#fff}.pipeline{display:flex;gap:6px;align-items:end;height:70px;margin-top:16px;border-bottom:1px solid #37566b}.pipeline span{flex:1;min-width:5px;height:42px;border-radius:2px 2px 0 0}.pipeline span:nth-child(3n){height:64px}
</style>
