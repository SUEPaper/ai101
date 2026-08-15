<script setup lang="ts">
import { computed, ref } from 'vue'
const blocks = ref(24)
const residualScale = ref(0.1)
const shortcut = ref(true)
const plainGain = computed(() => (1 - residualScale.value) ** blocks.value)
const residualGain = computed(() => shortcut.value ? 1 + residualScale.value * Math.sqrt(blocks.value) : plainGain.value)
const pathCount = computed(() => shortcut.value ? 2 ** Math.min(blocks.value, 30) : 1)
</script>
<template>
  <DlExperimentFrame index="04-B" title="残差路径示波器" question="恒等捷径为什么让深网络拥有一条不会被所有权重连乘吞没的梯度通道？" formula="y = x + αF(x),  ∂y/∂x = I + α∂F/∂x">
    <template #visual>
      <svg viewBox="0 0 560 250" role="img" aria-label="普通堆叠与残差堆叠的信号路径对比">
        <path class="rail" d="M35 126 H520"/><path v-if="shortcut" class="skip" d="M62 126 C120 34 220 34 278 126 C336 218 440 218 500 126"/>
        <g v-for="index in 8" :key="index"><rect :x="48+index*58" y="105" width="34" height="42" rx="3"/><text :x="65+index*58" y="132">F</text></g>
        <circle cx="500" cy="126" r="10"/><text x="496" y="130">+</text>
        <text x="38" y="224">effective gradient</text><rect x="170" y="209" :width="Math.min(330,330*residualGain/2.8)" height="18" class="gain"/>
      </svg>
    </template>
    <template #controls>
      <label>残差块数 = {{ blocks }}<input v-model.number="blocks" type="range" min="2" max="48" step="2"></label>
      <label>残差尺度 α = {{ residualScale.toFixed(2) }}<input v-model.number="residualScale" type="range" min="0.02" max="0.3" step="0.01"></label>
      <button type="button" @click="shortcut=!shortcut">{{ shortcut ? '断开恒等捷径' : '接入恒等捷径' }}</button>
    </template>
    <template #metrics><div><span>普通连乘代理</span><strong>{{ plainGain.toExponential(2) }}</strong></div><div><span>有效梯度代理</span><strong>{{ residualGain.toFixed(3) }}</strong></div><div><span>可选路径数</span><strong>{{ pathCount.toLocaleString() }}</strong></div><div><span>当前拓扑</span><strong>{{ shortcut?'x + F(x)':'F(x)' }}</strong></div></template>
    <template #note>这是解释路径的教学代理，不是精确梯度预测。关键事实是残差块的 Jacobian 含恒等项；真正训练时仍需合适初始化、归一化、残差尺度和学习率。</template>
  </DlExperimentFrame>
</template>
<style scoped>
svg{width:100%;min-height:280px;border:1px solid #294b5c;border-radius:7px;background:#0b172a}.rail{fill:none;stroke:#65808d;stroke-width:3}.skip{fill:none;stroke:#58d3c8;stroke-width:4}.gain{fill:#f2c14e}.gain,rect{transition:all .2s}rect{fill:#19344b;stroke:#65a9df}circle{fill:#f2c14e}text{fill:#d7e7e8;font:11px ui-monospace,monospace;text-anchor:middle}
</style>
