<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type MaskMode = 'correct' | 'none' | 'post'
const activeQuery = ref(3)
const validTokens = ref(7)
const futureSeverity = ref(1.8)
const maskMode = ref<MaskMode>('correct')
const tokens = ['启动','振动升','润滑低','转速降','跳闸','复位','确认','PAD']

function softmax(values: number[]) {
  const maximum = Math.max(...values.filter(Number.isFinite))
  const exponentials = values.map((value) => Number.isFinite(value) ? Math.exp(value - maximum) : 0)
  const total = exponentials.reduce((sum,value) => sum + value, 0)
  return exponentials.map((value) => value / total)
}

function computeMatrix(severity: number, mode: MaskMode, count: number) {
  const matrix: number[][] = []
  for (let row = 0; row < tokens.length; row += 1) {
    const logits = tokens.map((_,column) => 1.1 - .34 * Math.abs(row-column) + (column===2 ? .72 : column===4 ? .36 : 0) + (column===6 ? severity : 0))
    const paddingMasked = logits.map((value,column) => column >= count ? -Infinity : value)
    if (mode === 'correct') matrix.push(softmax(paddingMasked.map((value,column) => column > row ? -Infinity : value)))
    else if (mode === 'post') matrix.push(softmax(paddingMasked).map((value,column) => column > row ? 0 : value))
    else matrix.push(softmax(paddingMasked))
  }
  return matrix
}

const result = computed(() => {
  const query = Math.min(activeQuery.value, validTokens.value - 1)
  const matrix = computeMatrix(futureSeverity.value, maskMode.value, validTokens.value)
  const baseline = computeMatrix(.2, maskMode.value, validTokens.value)
  const values = tokens.map((_,index) => index === 6 ? futureSeverity.value : .15 * index - .3)
  const baselineValues = tokens.map((_,index) => index === 6 ? .2 : .15 * index - .3)
  const output = matrix[query].reduce((sum,weight,index) => sum + weight * values[index], 0)
  const baselineOutput = baseline[query].reduce((sum,weight,index) => sum + weight * baselineValues[index], 0)
  const row = matrix[query]
  return { query, matrix, row, output, rowSum:row.reduce((a,b)=>a+b,0), futureWeight:row.slice(query+1).reduce((a,b)=>a+b,0), prefixDrift:Math.abs(output-baselineOutput) }
})
const cellX = (column: number) => 128 + column * 43
const cellY = (row: number) => 62 + row * 27
const cellOpacity = (weight: number) => .08 + .92 * Math.min(1, weight * 3.5)
</script>

<template>
  <MlExperimentFrame index="17-B" title="用因果泄漏扫描器逐格核对 Mask" question="未来事件被改写时，过去位置的输出是否保持不变？先 Softmax 再清零为什么仍会留下归一化污染？" formula="A = softmax(QKᵀ / √dₖ + M), Mᵢⱼ = −∞ if j > i">
    <template #visual>
      <figure class="mask-scanner">
        <svg viewBox="0 0 600 326" role="img" :aria-label="`查询${tokens[result.query]}，未来权重${result.futureWeight.toExponential(2)}，前缀漂移${result.prefixDrift.toExponential(2)}`">
          <rect x="15" y="15" width="570" height="296" rx="9" class="scanner-bg" />
          <text x="32" y="37" class="panel-title">因果注意力泄漏扫描 · 行是 Query，列是 Key</text>
          <rect x="104" y="48" width="382" height="231" rx="7" class="matrix-panel" />
          <g v-for="(token,column) in tokens" :key="`column-${token}`"><text :x="cellX(column)+17" y="54" :class="['column-label',{padding:column>=validTokens}]">{{ token }}</text></g>
          <g v-for="(row,rowIndex) in result.matrix" :key="`row-${rowIndex}`">
            <text x="96" :y="cellY(rowIndex)+17" class="row-label" text-anchor="end">Q{{ rowIndex+1 }} {{ tokens[rowIndex] }}</text>
            <g v-for="(weight,columnIndex) in row" :key="`cell-${rowIndex}-${columnIndex}`">
              <rect :x="cellX(columnIndex)" :y="cellY(rowIndex)" width="34" height="21" rx="3" :class="['matrix-cell',{active:rowIndex===result.query,forbidden:columnIndex>rowIndex,padding:columnIndex>=validTokens}]" :style="{fillOpacity:cellOpacity(weight)}" />
              <text :x="cellX(columnIndex)+17" :y="cellY(rowIndex)+14" class="cell-value">{{ weight.toFixed(2) }}</text>
            </g>
          </g>
          <rect x="512" y="48" width="56" height="231" rx="7" class="alarm-panel" />
          <text x="540" y="68" class="alarm-title" text-anchor="middle">扫描</text><circle cx="540" cy="99" r="14" :class="['alarm-lamp',{danger:result.prefixDrift>1e-8}]" /><text x="540" y="128" :class="['alarm-text',{danger:result.prefixDrift>1e-8}]" text-anchor="middle">{{ result.prefixDrift>1e-8 ? 'LEAK' : 'PASS' }}</text>
          <text x="540" y="162" class="alarm-value" text-anchor="middle">Σ={{ result.rowSum.toFixed(2) }}</text><text x="540" y="184" class="alarm-value" text-anchor="middle">Δ={{ result.prefixDrift.toExponential(1) }}</text>
          <text x="540" y="222" class="alarm-note" text-anchor="middle">未来事件</text><text x="540" y="236" class="alarm-note" text-anchor="middle">强度 {{ futureSeverity.toFixed(1) }}</text><text x="540" y="264" class="alarm-note" text-anchor="middle">有效 {{ validTokens }}/8</text>
          <text x="104" y="299" class="footer-note">紫框为当前 Query；灰格为未来或 padding；正确 Mask 必须在 Softmax 之前进入 logits。</text>
        </svg>
        <figcaption><span><i class="weight" />注意力权重</span><span><i class="active" />检查行</span><span><i class="forbidden" />不可见位置</span><span><i class="danger" />检测到泄漏</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="mask-switch" role="group" aria-label="因果遮罩实现方式"><button type="button" :aria-pressed="maskMode==='correct'" @click="maskMode='correct'">Softmax 前 Mask</button><button type="button" :aria-pressed="maskMode==='none'" @click="maskMode='none'">不加因果 Mask</button><button type="button" :aria-pressed="maskMode==='post'" @click="maskMode='post'">Softmax 后清零</button></div>
      <label><span>检查 Query <output>Q{{ result.query+1 }} {{ tokens[result.query] }}</output></span><input v-model.number="activeQuery" type="range" min="0" :max="validTokens-1" step="1"></label>
      <label><span>未来“确认”强度 <output>{{ futureSeverity.toFixed(1) }}</output></span><input v-model.number="futureSeverity" type="range" min=".2" max="4" step=".1"></label>
      <label><span>有效 token 数 <output>{{ validTokens }}</output></span><input v-model.number="validTokens" type="range" min="5" max="8" step="1"></label>
      <p>“Softmax 后清零”虽然让未来格显示为 0，但合法位置的权重已被未来 logit 参与过归一化，行和小于 1，过去输出仍会随未来事件改变。</p>
    </template>
    <template #metrics><div><span>未来权重</span><strong>{{ result.futureWeight.toExponential(2) }}</strong></div><div><span>权重行和</span><strong>{{ result.rowSum.toFixed(6) }}</strong></div><div><span>未来改写导致前缀漂移</span><strong>{{ result.prefixDrift.toExponential(2) }}</strong></div><div><span>当前可见 Key</span><strong>{{ maskMode==='correct' ? result.query+1 : validTokens }}</strong></div></template>
    <template #note>因果 Mask 和 padding Mask 是不同条件，实际实现通常合并后一次加到 logits。全被遮住的一行没有合法概率分布，必须在输入合同处拒绝或定义专门行为，不能让 0/0 变成 NaN 后继续传播。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.mask-scanner{margin:0}.mask-scanner svg{display:block;width:100%;border:1px solid #284b59;border-radius:10px;background:#091820}.scanner-bg{fill:#091820}.mask-scanner text{font:700 7px ui-monospace,monospace}.panel-title{fill:#d9edf2;font-size:10px}.matrix-panel,.alarm-panel{fill:#10232c;stroke:#2c5262}.column-label{fill:#9ab3bd;text-anchor:middle;font-size:6px}.column-label.padding{fill:#657b84}.row-label{fill:#89a9b4}.matrix-cell{fill:#5fd8e8;stroke:#1d4350}.matrix-cell.active{stroke:#d486ff;stroke-width:2}.matrix-cell.forbidden{fill:#51646d}.matrix-cell.padding{fill:#384a52}.cell-value{fill:#eaf6f8;text-anchor:middle;font-size:6px}.alarm-title{fill:#89a9b4}.alarm-lamp{fill:#5fd8e8;filter:drop-shadow(0 0 6px #5fd8e8)}.alarm-lamp.danger{fill:#ff6577;filter:drop-shadow(0 0 6px #ff6577)}.alarm-text{fill:#5fd8e8;font-size:10px}.alarm-text.danger{fill:#ff6577}.alarm-value{fill:#ffb454}.alarm-note,.footer-note{fill:#89a9b4}.mask-scanner figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.mask-scanner figcaption span{display:inline-flex;align-items:center;gap:5px}.mask-scanner figcaption i{width:12px;height:8px;border-radius:2px}.mask-scanner figcaption i.weight{background:#5fd8e8}.mask-scanner figcaption i.active{border:2px solid #d486ff}.mask-scanner figcaption i.forbidden{background:#51646d}.mask-scanner figcaption i.danger{background:#ff6577}.mask-switch{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.mask-switch button{min-height:44px;border:1px solid #496a78;border-radius:7px;color:#d9edf2;background:#112833;font:750 8px ui-monospace,monospace;cursor:pointer}.mask-switch button[aria-pressed="true"]{border-color:#ffb454;color:#ffb454;background:#302c2b}.mask-switch button:focus-visible{outline:3px solid #d486ff;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#5fd8e8;font:750 10px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#d486ff;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.mask-scanner{overflow-x:auto;overscroll-behavior-inline:contain}.mask-scanner svg{width:560px;max-width:none}.mask-scanner figcaption{min-width:560px}}
</style>
