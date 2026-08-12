<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const maxDepth = ref(2)
const pressure = ref(73)
const fillTime = ref(1.08)
const moisture = ref(.065)

const classColors: Record<string, string> = { 合格: '#6cc7a8', 欠注: '#eab64d', 飞边: '#e36552' }
const trainAccuracy = [.72, .92, .98]
const validationAccuracy = [.70, .89, .84]
const leafCounts = [2, 4, 6]

const result = computed(() => {
  const path = ['root']
  const messages = [`充填时间 ${fillTime.value.toFixed(2)} s ${fillTime.value <= 1.18 ? '≤' : '>'} 1.18 s`]
  let prediction = fillTime.value <= 1.18 ? '合格' : '欠注'
  let terminal = fillTime.value <= 1.18 ? 'left' : 'right'
  if (maxDepth.value >= 2) {
    path.push(terminal)
    if (terminal === 'left') {
      messages.push(`峰值压力 ${pressure.value.toFixed(0)} MPa ${pressure.value <= 78 ? '≤' : '>'} 78 MPa`)
      prediction = pressure.value <= 78 ? '合格' : '飞边'
      terminal = pressure.value <= 78 ? 'll' : 'lr'
    } else {
      messages.push(`峰值压力 ${pressure.value.toFixed(0)} MPa ${pressure.value <= 65 ? '≤' : '>'} 65 MPa`)
      prediction = pressure.value <= 65 ? '欠注' : '合格'
      terminal = pressure.value <= 65 ? 'rl' : 'rr'
    }
    path.push(terminal)
  }
  if (maxDepth.value >= 3 && prediction === '合格') {
    const limit = terminal === 'll' ? .09 : .10
    messages.push(`树脂含水 ${moisture.value.toFixed(3)}% ${moisture.value <= limit ? '≤' : '>'} ${limit.toFixed(2)}%`)
    prediction = moisture.value <= limit ? '合格' : '飞边'
    path.push('moisture')
  }
  const support: Record<string, number> = { 合格: 34, 欠注: 21, 飞边: 18 }
  return {
    prediction, path, messages,
    depth: messages.length,
    support: Math.max(7, support[prediction] - (maxDepth.value - 1) * 7),
    train: trainAccuracy[maxDepth.value - 1],
    validation: validationAccuracy[maxDepth.value - 1],
    leaves: leafCounts[maxDepth.value - 1],
  }
})

const active = (id: string) => result.value.path.includes(id)
</script>

<template>
  <MlExperimentFrame index="19-B" title="沿气路式树流形追踪一件产品" question="树加深时，一件产品会多经过哪些质量闸门？训练准确率继续上升，为何验证批次反而可能恶化？" formula="h(x) = leaf reached by sequential threshold tests">
    <template #visual>
      <div class="manifold-scroll">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`深度${maxDepth}决策树把当前产品分为${result.prediction}，路径经过${result.depth}个判断`">
          <rect width="600" height="326" rx="12" class="board" />
          <text x="24" y="24" class="eyebrow">QUALITY AIR MANIFOLD · 当前工件的路由以亮色显示</text>
          <path d="M300 76 V98 H150 V118 M300 98 H450 V118" class="pipe" />
          <path d="M150 160 V188 H72 V218 M150 188 H228 V218 M450 160 V188 H372 V218 M450 188 H528 V218" class="pipe" />
          <path v-if="active('left')" d="M300 76 V98 H150 V118" class="route" />
          <path v-if="active('right')" d="M300 76 V98 H450 V118" class="route" />
          <path v-if="active('ll')" d="M150 160 V188 H72 V218" class="route" />
          <path v-if="active('lr')" d="M150 160 V188 H228 V218" class="route" />
          <path v-if="active('rl')" d="M450 160 V188 H372 V218" class="route" />
          <path v-if="active('rr')" d="M450 160 V188 H528 V218" class="route" />
          <g :class="{lit:active('root')}"><rect x="205" y="36" width="190" height="40" rx="6" class="valve" /><text x="300" y="53" class="node">充填时间 ≤ 1.18 s?</text><text x="300" y="67" class="node-sub">ROOT · n=140</text></g>
          <g :class="{lit:active('left'),locked:maxDepth<2}"><rect x="70" y="118" width="160" height="42" rx="6" class="valve" /><text x="150" y="136" class="node">压力 ≤ 78 MPa?</text><text x="150" y="151" class="node-sub">FAST FILL</text></g>
          <g :class="{lit:active('right'),locked:maxDepth<2}"><rect x="370" y="118" width="160" height="42" rx="6" class="valve" /><text x="450" y="136" class="node">压力 ≤ 65 MPa?</text><text x="450" y="151" class="node-sub">SLOW FILL</text></g>
          <g v-for="leaf in [
            {id:'ll',x:22,label:'合格',hint:'含水复核 ≤ .09%'},
            {id:'lr',x:178,label:'飞边',hint:'高压快速充填'},
            {id:'rl',x:322,label:'欠注',hint:'低压缓慢充填'},
            {id:'rr',x:478,label:'合格',hint:'含水复核 ≤ .10%'},
          ]" :key="leaf.id" :class="{lit:active(leaf.id),locked:maxDepth<2}">
            <rect :x="leaf.x" y="218" width="100" height="58" rx="7" class="leaf" :style="active(leaf.id)?{stroke:classColors[result.prediction]}:{}" />
            <circle :cx="leaf.x+14" cy="235" r="5" :fill="classColors[leaf.label]" />
            <text :x="leaf.x+26" y="239" class="leaf-label">{{ leaf.label }}</text>
            <text :x="leaf.x+10" y="258" class="leaf-hint">{{ maxDepth===3 && active(leaf.id) && leaf.label==='合格' ? '含水阀已执行' : leaf.hint }}</text>
          </g>
          <g transform="translate(22 297)">
            <rect width="556" height="17" rx="4" class="route-ticket" />
            <text x="9" y="12" class="ticket">PATH · {{ result.messages.join(' → ') }}</text>
          </g>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>最大深度 <output>{{ maxDepth }}</output></span><input v-model.number="maxDepth" aria-label="最大深度" type="range" min="1" max="3" step="1"></label>
      <label class="control"><span>峰值压力 <output>{{ pressure }} MPa</output></span><input v-model.number="pressure" aria-label="峰值压力" type="range" min="55" max="88" step="1"></label>
      <label class="control"><span>充填时间 <output>{{ fillTime.toFixed(2) }} s</output></span><input v-model.number="fillTime" aria-label="充填时间" type="range" min=".85" max="1.45" step=".01"></label>
      <label class="control"><span>树脂含水 <output>{{ moisture.toFixed(3) }}%</output></span><input v-model.number="moisture" aria-label="树脂含水" type="range" min=".03" max=".13" step=".005"></label>
      <p class="hint">深度 3 的含水阀只在前两层输出“合格”时启用。它能修复少量边界样本，也最容易追随标签噪声。</p>
    </template>
    <template #metrics><div><span>当前分流结果</span><strong :style="{color:classColors[result.prediction]}">{{ result.prediction }}</strong></div><div><span>路径深度 / 叶数</span><strong>{{ result.depth }} / {{ result.leaves }}</strong></div><div><span>训练 / 验证准确率</span><strong>{{ result.train.toFixed(2) }} / {{ result.validation.toFixed(2) }}</strong></div><div><span>叶内训练样本</span><strong>{{ result.support }}</strong></div></template>
    <template #note>一条路径是模型的计算轨迹，不是因果链。深度增加会缩小叶区域并降低训练偏差，同时减少每个叶的证据量、放大阈值和标签扰动；验证分数必须按材料批次或生产 campaign 隔离计算。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.manifold-scroll{overflow-x:auto}.manifold-scroll svg{display:block;min-width:560px;width:100%;height:auto}.board{fill:#111a24}.eyebrow,.node,.node-sub,.leaf-label,.leaf-hint,.ticket{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#87cbd4;font-size:9px;letter-spacing:.08em}.pipe{fill:none;stroke:#40515e;stroke-width:6;stroke-linecap:round;stroke-linejoin:round}.route{fill:none;stroke:#eab64d;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 4px #eab64d)}.valve,.leaf{fill:#172630;stroke:#526b78;stroke-width:1.2}.node{fill:#d8e4e8;font-size:10px;text-anchor:middle}.node-sub{fill:#78909e;font-size:7px;text-anchor:middle;letter-spacing:.1em}.leaf-label{fill:#dce8eb;font-size:10px;font-weight:800}.leaf-hint{fill:#8799a4;font-size:6.5px}.lit .valve,.lit .leaf{stroke:#eab64d;stroke-width:2}.locked{opacity:.28}.route-ticket{fill:#182a33;stroke:#3a5662}.ticket{fill:#b9cad1;font-size:7px}.control{display:grid;gap:7px;margin:0}.control>span{display:flex;justify-content:space-between;color:#aebbdc;font-size:11px}.control output{color:#f1ce79;font:700 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#e36552}input:focus-visible{outline:3px solid #74d7dc;outline-offset:2px}.hint{margin:0;padding:11px;border-left:3px solid #3ab5c4;color:#a6b7bf;background:#111e27;font-size:10px;line-height:1.65}@media(max-width:560px){.manifold-scroll{margin-right:-18px;padding-right:18px}.manifold-scroll svg{width:560px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
