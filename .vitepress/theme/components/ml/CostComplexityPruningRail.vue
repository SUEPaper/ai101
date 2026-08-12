<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Profile = 'same' | 'shift'
const alpha = ref(.02)
const profile = ref<Profile>('same')
const candidates = [
  { leaves: 8, train: .025, validation: .145, shifted: .205 },
  { leaves: 6, train: .045, validation: .110, shifted: .170 },
  { leaves: 4, train: .075, validation: .080, shifted: .125 },
  { leaves: 3, train: .110, validation: .090, shifted: .100 },
  { leaves: 1, train: .320, validation: .330, shifted: .285 },
]

const result = computed(() => {
  const scored = candidates.map((item, index) => ({ ...item, index, objective: item.train + alpha.value * item.leaves }))
  const selected = scored.reduce((best, item) => item.objective < best.objective - 1e-12 ? item : best)
  const validationKey = profile.value === 'same' ? 'validation' : 'shifted'
  const validationBest = scored.reduce((best, item) => item[validationKey] < best[validationKey] ? item : best)
  return { selected, validationBest, scored, validationKey }
})

const curveX = (index: number) => 402 + index * 40
const curveY = (value: number) => 260 - ((value - .02) / .32) * 150
const cut = (stage: number) => result.value.selected.index >= stage
</script>

<template>
  <MlExperimentFrame index="19-C" title="沿弱链接导轨执行代价复杂度剪枝" question="当每个叶子都收取复杂度费用时，哪段子树最先失去保留资格？训练目标选出的子树是否就是目标生产批次上的最佳子树？" formula="Rα(T)=R(T)+α|leaves(T)|">
    <template #visual>
      <div class="prune-scroll">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`alpha为${alpha.toFixed(3)}，当前子树有${result.selected.leaves}个叶，验证错误率${(result.selected[result.validationKey]*100).toFixed(1)}%`">
          <rect width="600" height="326" rx="12" class="board" />
          <text x="22" y="24" class="eyebrow">WEAKEST-LINK PRUNING RAIL · 亮线为保留结构</text>
          <g transform="translate(18 35)">
            <path d="M165 34 V61 H82 V84 M165 61 H248 V84 M82 119 V146 H40 V170 M82 146 H124 V170 M248 119 V146 H206 V170 M248 146 H290 V170" class="tree-wire" />
            <path v-if="!cut(4)" d="M165 34 V61 H82 V84 M165 61 H248 V84" class="kept" />
            <path v-if="!cut(3)" d="M82 119 V146 H40 V170 M82 146 H124 V170" class="kept" />
            <path v-if="!cut(2)" d="M248 119 V146 H206 V170 M248 146 H290 V170" class="kept" />
            <rect x="105" y="0" width="120" height="34" rx="6" class="node" /><text x="165" y="21" class="node-text">ROOT · fill ≤ 1.18</text>
            <g :class="{pruned:cut(3)}"><rect x="32" y="84" width="100" height="35" rx="6" class="node" /><text x="82" y="106" class="node-text">B · pressure</text></g>
            <g :class="{pruned:cut(2)}"><rect x="198" y="84" width="100" height="35" rx="6" class="node" /><text x="248" y="106" class="node-text">C · pressure</text></g>
            <g v-for="(x,index) in [12,96,178,262]" :key="x" :class="{pruned:cut(index<2?1:2)}">
              <rect :x="x" y="170" width="56" height="34" rx="5" class="leaf" /><text :x="x+28" y="191" class="leaf-text">L{{ index+1 }}</text>
            </g>
            <g v-if="cut(1)" transform="translate(77 143)"><path d="M-8 -8 L8 8 M8 -8 L-8 8" class="cut-mark" /><text x="13" y="3" class="cut-label">CUT</text></g>
            <g v-if="cut(2)" transform="translate(247 143)"><path d="M-8 -8 L8 8 M8 -8 L-8 8" class="cut-mark" /><text x="13" y="3" class="cut-label">CUT</text></g>
            <g v-if="cut(3)" transform="translate(82 60)"><path d="M-8 -8 L8 8 M8 -8 L-8 8" class="cut-mark" /><text x="13" y="3" class="cut-label">CUT</text></g>
            <g v-if="cut(4)" transform="translate(165 59)"><path d="M-8 -8 L8 8 M8 -8 L-8 8" class="cut-mark" /><text x="13" y="3" class="cut-label">CUT TO ROOT</text></g>
            <rect x="0" y="225" width="330" height="44" rx="6" class="ticket" />
            <text x="12" y="243" class="ticket-label">ACTIVE SUBTREE</text>
            <text x="12" y="260" class="ticket-value">{{ result.selected.leaves }} LEAVES · Rα={{ result.selected.objective.toFixed(3) }}</text>
          </g>
          <g>
            <text x="390" y="48" class="chart-title">PRUNING PATH</text>
            <line x1="392" y1="260" x2="575" y2="260" class="axis" /><line x1="392" y1="94" x2="392" y2="260" class="axis" />
            <polyline :points="result.scored.map((item,index)=>`${curveX(index)},${curveY(item[result.validationKey])}`).join(' ')" class="validation-line" />
            <polyline :points="result.scored.map((item,index)=>`${curveX(index)},${curveY(item.train)}`).join(' ')" class="train-line" />
            <g v-for="(item,index) in result.scored" :key="item.leaves">
              <circle :cx="curveX(index)" :cy="curveY(item[result.validationKey])" r="5" :class="['curve-dot',{chosen:index===result.selected.index,best:index===result.validationBest.index}]" />
              <text :x="curveX(index)" y="278" class="leaf-count">{{ item.leaves }}</text>
            </g>
            <text x="397" y="82" class="legend validation">— 验证错误</text><text x="495" y="82" class="legend train">-- 训练错误</text>
            <text x="451" y="298" class="axis-label">叶节点数（剪枝路径）</text>
            <text x="373" y="180" class="axis-label vertical">错误率</text>
          </g>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>复杂度费用 α <output>{{ alpha.toFixed(3) }}</output></span><input v-model.number="alpha" aria-label="复杂度费用 alpha" type="range" min="0" max=".12" step=".005"></label>
      <div class="control"><span>验证批次</span><div class="switch" role="group" aria-label="验证批次"><button type="button" :aria-pressed="profile==='same'" @click="profile='same'">同模具版本</button><button type="button" :aria-pressed="profile==='shift'" @click="profile='shift'">换版批次</button></div></div>
      <p class="hint">黄色环标出当前 α 的训练目标最优子树；青色外环标出当前验证批次错误率最低的候选。两者不一致时，不能在测试集上继续调 α。</p>
    </template>
    <template #metrics><div><span>当前叶节点数</span><strong>{{ result.selected.leaves }}</strong></div><div><span>训练 / 验证错误率</span><strong>{{ (result.selected.train*100).toFixed(1) }}% / {{ (result.selected[result.validationKey]*100).toFixed(1) }}%</strong></div><div><span>代价复杂度目标</span><strong>{{ result.selected.objective.toFixed(3) }}</strong></div><div><span>验证最优叶数</span><strong>{{ result.validationBest.leaves }}</strong></div></template>
    <template #note>代价复杂度路径只提供一列嵌套候选，α 仍须在独立验证折中选择。若生产模具、材料或量测协议发生变化，旧验证集不再代表目标风险；“换版批次”不是允许偷看测试集，而是在提醒重新定义部署域。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.prune-scroll{overflow-x:auto}.prune-scroll svg{display:block;min-width:560px;width:100%;height:auto}.board{fill:#111a24}.eyebrow,.node-text,.leaf-text,.cut-label,.ticket-label,.ticket-value,.chart-title,.leaf-count,.legend,.axis-label{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#87cbd4;font-size:9px;letter-spacing:.08em}.tree-wire{fill:none;stroke:#3d4e5a;stroke-width:5;stroke-linecap:round}.kept{fill:none;stroke:#6cc7a8;stroke-width:3;stroke-linecap:round;filter:drop-shadow(0 0 3px #6cc7a8)}.node,.leaf{fill:#172630;stroke:#516a76}.node-text,.leaf-text{fill:#d9e4e8;font-size:8px;text-anchor:middle}.leaf-text{font-size:9px}.pruned{opacity:.20}.cut-mark{stroke:#e36552;stroke-width:3}.cut-label{fill:#f08b78;font-size:7px}.ticket{fill:#172630;stroke:#425c68}.ticket-label{fill:#7f99a5;font-size:7px;letter-spacing:.1em}.ticket-value{fill:#f2ce78;font-size:13px;font-weight:800}.chart-title{fill:#8bcbd3;font-size:9px;letter-spacing:.08em}.axis{stroke:#607683;stroke-width:1}.validation-line{fill:none;stroke:#3ab5c4;stroke-width:2}.train-line{fill:none;stroke:#eab64d;stroke-width:1.5;stroke-dasharray:4 3}.curve-dot{fill:#3ab5c4;stroke:#111a24;stroke-width:2}.curve-dot.chosen{stroke:#eab64d;stroke-width:4}.curve-dot.best{filter:drop-shadow(0 0 4px #69d7df)}.leaf-count,.legend,.axis-label{fill:#8da0aa;font-size:7px;text-anchor:middle}.legend{text-anchor:start}.legend.validation{fill:#61cbd5}.legend.train{fill:#e8bd5a}.vertical{transform:rotate(-90deg);transform-origin:373px 180px}.control{display:grid;gap:8px;margin:0}.control>span{display:flex;justify-content:space-between;color:#aebbdc;font-size:11px}.control output{color:#f2ce78;font:700 11px/1 ui-monospace,monospace}.switch{display:grid;grid-template-columns:1fr 1fr;gap:7px}button{min-height:44px;border:1px solid #46606f;border-radius:6px;color:#dbe5e9;background:#14232c;font:700 10px/1.2 ui-monospace,monospace;cursor:pointer}button[aria-pressed="true"]{border-color:#eab64d;color:#f6d983;background:#30291e;box-shadow:inset 0 0 0 1px #9a7935}button:focus-visible,input:focus-visible{outline:3px solid #74d7dc;outline-offset:2px}input[type="range"]{width:100%;min-height:44px;accent-color:#e36552}.hint{margin:0;padding:11px;border-left:3px solid #3ab5c4;color:#a6b7bf;background:#111e27;font-size:10px;line-height:1.65}@media(max-width:560px){.prune-scroll{margin-right:-18px;padding-right:18px}.prune-scroll svg{width:560px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
