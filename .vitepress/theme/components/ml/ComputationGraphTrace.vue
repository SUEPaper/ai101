<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

type Phase = 'forward' | 'backward'
type AccumulationMode = 'sum' | 'overwrite'

const phase = ref<Phase>('backward')
const accumulationMode = ref<AccumulationMode>('sum')
const x = ref(1.2)
const y = ref(-1.5)
const target = ref(0.4)

const result = computed(() => {
  const a = x.value * y.value
  const b = a + x.value
  const error = b - target.value
  const loss = .5 * error ** 2
  const gradError = error
  const gradB = gradError
  const gradA = gradB
  const xViaMultiply = gradA * y.value
  const xViaSkip = gradB
  const correctGradX = xViaMultiply + xViaSkip
  const displayedGradX = accumulationMode.value === 'sum' ? correctGradX : xViaSkip
  const gradY = gradA * x.value
  const gradTarget = -gradError
  const relativeError = Math.abs(displayedGradX - correctGradX) / Math.max(1, Math.abs(correctGradX), Math.abs(displayedGradX))
  return {
    a, b, error, loss, gradError, gradB, gradA, xViaMultiply, xViaSkip,
    correctGradX, displayedGradX, gradY, gradTarget, relativeError,
  }
})

const loadReinforcing = () => {
  x.value = 1.2
  y.value = 1.5
  target.value = 0
}

const loadCancellation = () => {
  x.value = 1.2
  y.value = -1
  target.value = .4
}

const nodeClass = (kind: 'input' | 'gate' | 'loss') => [
  'node-shell',
  kind,
  phase.value === 'backward' ? 'reverse-active' : 'forward-active',
]
</script>

<template>
  <MlExperimentFrame
    index="14-A"
    title="在无损检测计算图上追踪两条梯度支路"
    question="同一个传感器证据既进入乘法门、又绕过乘法门时，损失对它的两份敏感度为什么必须相加？"
    formula="a=xy, b=a+x, L=½(b−t)²"
  >
    <template #visual>
      <figure class="graph-trace">
        <svg viewBox="0 0 600 326" role="img" :aria-label="'当前损失 '+result.loss.toFixed(3)+'，x 梯度 '+result.displayedGradX.toFixed(3)">
          <defs>
            <marker id="trace-forward-arrow" refX="8" refY="4" markerWidth="9" markerHeight="8" orient="auto"><path d="M0,0 L9,4 L0,8 Z" class="forward-arrow" /></marker>
            <marker id="trace-reverse-arrow" refX="1" refY="4" markerWidth="9" markerHeight="8" orient="auto"><path d="M9,0 L0,4 L9,8 Z" class="reverse-arrow" /></marker>
          </defs>
          <rect x="15" y="15" width="570" height="296" rx="9" class="trace-bg" />
          <text x="35" y="34" class="panel-title">前向数值流</text>
          <text x="565" y="34" class="panel-title end">反向伴随量 ∂L/∂node</text>

          <path d="M82 83 C125 83 130 118 171 118" class="edge forward" marker-end="url(#trace-forward-arrow)" />
          <path d="M82 207 C125 207 130 148 171 148" class="edge forward" marker-end="url(#trace-forward-arrow)" />
          <path d="M229 133 L291 133" class="edge forward" marker-end="url(#trace-forward-arrow)" />
          <path d="M82 83 C190 83 201 170 291 170" class="edge forward skip" marker-end="url(#trace-forward-arrow)" />
          <path d="M349 151 L401 151" class="edge forward" marker-end="url(#trace-forward-arrow)" />
          <path d="M459 151 L512 151" class="edge forward" marker-end="url(#trace-forward-arrow)" />
          <path d="M82 270 C260 270 322 190 401 169" class="edge forward" marker-end="url(#trace-forward-arrow)" />

          <path v-if="phase === 'backward'" d="M512 170 L459 170" class="edge reverse" marker-end="url(#trace-reverse-arrow)" />
          <path v-if="phase === 'backward'" d="M401 170 L349 170" class="edge reverse" marker-end="url(#trace-reverse-arrow)" />
          <path v-if="phase === 'backward'" d="M291 151 L229 151" class="edge reverse" marker-end="url(#trace-reverse-arrow)" />
          <path v-if="phase === 'backward'" d="M171 143 C128 143 125 101 82 101" class="edge reverse" marker-end="url(#trace-reverse-arrow)" />
          <path v-if="phase === 'backward'" d="M171 159 C128 159 126 225 82 225" class="edge reverse" marker-end="url(#trace-reverse-arrow)" />
          <path v-if="phase === 'backward'" d="M291 187 C205 187 183 107 82 107" :class="['edge','reverse','skip',{broken:accumulationMode==='overwrite'}]" marker-end="url(#trace-reverse-arrow)" />
          <path v-if="phase === 'backward'" d="M401 188 C323 211 248 286 82 286" class="edge reverse" marker-end="url(#trace-reverse-arrow)" />

          <g transform="translate(34 55)">
            <rect width="48" height="55" rx="7" :class="nodeClass('input')" />
            <text x="24" y="20" class="node-name">x</text>
            <text x="24" y="38" class="value">{{ x.toFixed(2) }}</text>
            <text v-if="phase==='backward'" x="24" y="51" :class="['gradient',{wrong:accumulationMode==='overwrite'}]">g={{ result.displayedGradX.toFixed(2) }}</text>
          </g>
          <g transform="translate(34 179)">
            <rect width="48" height="55" rx="7" :class="nodeClass('input')" />
            <text x="24" y="20" class="node-name">y</text>
            <text x="24" y="38" class="value">{{ y.toFixed(2) }}</text>
            <text v-if="phase==='backward'" x="24" y="51" class="gradient">g={{ result.gradY.toFixed(2) }}</text>
          </g>
          <g transform="translate(34 242)">
            <rect width="48" height="55" rx="7" :class="nodeClass('input')" />
            <text x="24" y="20" class="node-name">t</text>
            <text x="24" y="38" class="value">{{ target.toFixed(2) }}</text>
            <text v-if="phase==='backward'" x="24" y="51" class="gradient">g={{ result.gradTarget.toFixed(2) }}</text>
          </g>

          <g transform="translate(171 105)">
            <rect width="58" height="58" rx="29" :class="nodeClass('gate')" />
            <text x="29" y="23" class="gate-name">×</text>
            <text x="29" y="41" class="value">a={{ result.a.toFixed(2) }}</text>
            <text v-if="phase==='backward'" x="29" y="54" class="gradient">g={{ result.gradA.toFixed(2) }}</text>
          </g>
          <g transform="translate(291 123)">
            <rect width="58" height="58" rx="29" :class="nodeClass('gate')" />
            <text x="29" y="23" class="gate-name">＋</text>
            <text x="29" y="41" class="value">b={{ result.b.toFixed(2) }}</text>
            <text v-if="phase==='backward'" x="29" y="54" class="gradient">g={{ result.gradB.toFixed(2) }}</text>
          </g>
          <g transform="translate(401 123)">
            <rect width="58" height="58" rx="29" :class="nodeClass('gate')" />
            <text x="29" y="23" class="gate-name">−t</text>
            <text x="29" y="41" class="value">e={{ result.error.toFixed(2) }}</text>
            <text v-if="phase==='backward'" x="29" y="54" class="gradient">g={{ result.gradError.toFixed(2) }}</text>
          </g>
          <g transform="translate(512 123)">
            <rect width="58" height="58" rx="9" :class="nodeClass('loss')" />
            <text x="29" y="22" class="gate-name">½e²</text>
            <text x="29" y="42" class="value">L={{ result.loss.toFixed(2) }}</text>
            <text v-if="phase==='backward'" x="29" y="55" class="gradient">g=1</text>
          </g>

          <rect x="105" y="43" width="151" height="38" rx="6" :class="['branch-ledger',{wrong:accumulationMode==='overwrite'}]" />
          <text x="114" y="58" class="ledger-label">x 的两路贡献</text>
          <text x="114" y="73" :class="['ledger-value',{wrong:accumulationMode==='overwrite'}]">
            {{ result.xViaMultiply.toFixed(2) }} {{ accumulationMode==='sum' ? '+' : '被覆盖 →' }} {{ result.xViaSkip.toFixed(2) }}
          </text>
        </svg>
        <figcaption><span><i class="forward" />前向值</span><span><i class="reverse" />反向梯度</span><span><i class="branch" />分支账本</span><span><i class="broken" />被覆盖的贡献</span></figcaption>
      </figure>
    </template>
    <template #controls>
      <div class="phase-switch" role="group" aria-label="计算图方向">
        <button type="button" :aria-pressed="phase==='forward'" @click="phase='forward'">只看前向</button>
        <button type="button" :aria-pressed="phase==='backward'" @click="phase='backward'">打开反向追溯</button>
      </div>
      <div class="accumulation-switch" role="group" aria-label="分支梯度处理">
        <button type="button" :aria-pressed="accumulationMode==='sum'" @click="accumulationMode='sum'">正确累加 +=</button>
        <button type="button" :aria-pressed="accumulationMode==='overwrite'" @click="accumulationMode='overwrite'">错误覆盖 =</button>
      </div>
      <label><span>热像残差 x <output>{{ x.toFixed(2) }}</output></span><input v-model.number="x" type="range" min="-3" max="3" step=".1"></label>
      <label><span>光强耦合 y <output>{{ y.toFixed(2) }}</output></span><input v-model.number="y" type="range" min="-3" max="3" step=".1"></label>
      <label><span>目标响应 t <output>{{ target.toFixed(2) }}</output></span><input v-model.number="target" type="range" min="-2" max="2" step=".1"></label>
      <div class="sample-switch"><button type="button" @click="loadReinforcing">装载同向放大</button><button type="button" @click="loadCancellation">装载支路抵消</button></div>
      <p>错误覆盖模式模拟把 x 的梯度写成赋值而不是累加：损失仍可计算，部分参数甚至仍会更新，但得到的已不是该计算图的总导数。</p>
    </template>
    <template #metrics>
      <div><span>前向损失 L</span><strong>{{ result.loss.toFixed(4) }}</strong></div>
      <div><span>x 经乘法支路</span><strong>{{ result.xViaMultiply.toFixed(4) }}</strong></div>
      <div><span>x 经跳接支路</span><strong>{{ result.xViaSkip.toFixed(4) }}</strong></div>
      <div><span>当前 x 梯度</span><strong>{{ result.displayedGradX.toFixed(4) }}</strong></div>
    </template>
    <template #note>
      青色记录前向值，琥珀色记录最终损失对节点的伴随量。分支不是“选一条路”，而是多个因果路径对同一变量敏感度的求和；自动微分框架也必须执行同一条累加规则。
    </template>
  </MlExperimentFrame>
</template>

<style scoped>
.graph-trace{margin:0}.graph-trace svg{display:block;width:100%;border:1px solid #29495f;border-radius:10px;background:#0b1722}.trace-bg{fill:#0b1722}.graph-trace text{font:700 9px ui-monospace,monospace}.panel-title{fill:#dbeaf0;font-size:10px}.end{text-anchor:end}.edge{fill:none;stroke-width:2}.edge.forward{stroke:#63d0c6;opacity:.7}.edge.forward.skip{stroke-dasharray:5 4}.edge.reverse{stroke:#f3c85b;opacity:.9}.edge.reverse.broken{stroke:#e35d68;stroke-dasharray:4 5;opacity:.48}.forward-arrow{fill:#63d0c6}.reverse-arrow{fill:#f3c85b}.node-shell{stroke-width:1.5}.node-shell.input{fill:#142736;stroke:#416177}.node-shell.gate{fill:#17263b;stroke:#52708d}.node-shell.loss{fill:#2b2534;stroke:#f3c85b}.node-shell.forward-active{filter:none}.node-shell.reverse-active{stroke-width:2}.node-name,.gate-name,.value,.gradient{text-anchor:middle}.node-name,.gate-name{fill:#e8f1f4;font-size:11px}.value{fill:#63d0c6}.gradient{fill:#f3c85b;font-size:8px}.gradient.wrong{fill:#e35d68}.branch-ledger{fill:#132333;stroke:#425d75}.branch-ledger.wrong{stroke:#e35d68}.ledger-label{fill:#8fa8b7}.ledger-value{fill:#f3c85b}.ledger-value.wrong{fill:#e35d68}.graph-trace figcaption{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:9px;color:#8fa8b7;font-size:10px}.graph-trace figcaption span{display:inline-flex;align-items:center;gap:5px}.graph-trace figcaption i{width:12px;height:8px;border-radius:2px}.graph-trace figcaption i.forward{background:#63d0c6}.graph-trace figcaption i.reverse{background:#f3c85b}.graph-trace figcaption i.branch{border:1px solid #52708d}.graph-trace figcaption i.broken{background:#e35d68}.phase-switch,.accumulation-switch,.sample-switch{display:grid;grid-template-columns:1fr 1fr;gap:8px}.phase-switch button,.accumulation-switch button,.sample-switch button{min-height:44px;border:1px solid #49627a;border-radius:7px;color:#dbeaf0;background:#142638;font:750 9px ui-monospace,monospace;cursor:pointer}.phase-switch button[aria-pressed="true"],.accumulation-switch button[aria-pressed="true"]{border-color:#f3c85b;color:#f3c85b;background:#242942}.phase-switch button:focus-visible,.accumulation-switch button:focus-visible,.sample-switch button:focus-visible{outline:3px solid #f3c85b;outline-offset:2px}.ml-experiment__controls label{display:grid;gap:5px;color:#c5d7df;font-size:11px}.ml-experiment__controls label span{display:flex;justify-content:space-between;gap:12px}.ml-experiment__controls output{color:#63d0c6;font:750 11px ui-monospace,monospace}.ml-experiment__controls input{width:100%;height:44px;margin:0;accent-color:#f08a55;cursor:pointer}.ml-experiment__controls p{margin:0;color:#8fa8b7;font-size:10px;line-height:1.65}@media(max-width:560px){.graph-trace{overflow-x:auto;overscroll-behavior-inline:contain}.graph-trace svg{width:560px;max-width:none}.graph-trace figcaption{min-width:560px}}
</style>
