<script setup lang="ts">
import { computed, ref } from 'vue'

type Mode = 'original' | 'flip' | 'letterbox' | 'rotate'

interface Point { x: number; y: number }

const mode = ref<Mode>('original')
const sourceWidth = 160
const sourceHeight = 110
const sourceBox = { x: 42, y: 30, width: 74, height: 54 }

const matrix = computed(() => {
  if (mode.value === 'flip') return [-1, 0, sourceWidth, 0, 1, 0]
  if (mode.value === 'letterbox') return [.9, 0, 18, 0, .9, 15.5]
  if (mode.value === 'rotate') {
    const angle = 15 * Math.PI / 180
    const cosine = Math.cos(angle)
    const sine = Math.sin(angle)
    const centerX = sourceWidth / 2
    const centerY = sourceHeight / 2
    return [
      cosine,
      -sine,
      centerX - cosine * centerX + sine * centerY,
      sine,
      cosine,
      centerY - sine * centerX - cosine * centerY
    ]
  }
  return [1, 0, 0, 0, 1, 0]
})

const canvas = computed(() => mode.value === 'letterbox' ? { width: 180, height: 130 } : { width: sourceWidth, height: sourceHeight })
const svgTransform = computed(() => `matrix(${matrix.value[0]} ${matrix.value[3]} ${matrix.value[1]} ${matrix.value[4]} ${matrix.value[2]} ${matrix.value[5]})`)

function transformPoint(point: Point): Point {
  const [a, b, c, d, e, f] = matrix.value
  return { x: a * point.x + b * point.y + c, y: d * point.x + e * point.y + f }
}

const transformedBox = computed(() => {
  const corners = [
    { x: sourceBox.x, y: sourceBox.y },
    { x: sourceBox.x + sourceBox.width, y: sourceBox.y },
    { x: sourceBox.x, y: sourceBox.y + sourceBox.height },
    { x: sourceBox.x + sourceBox.width, y: sourceBox.y + sourceBox.height }
  ].map(transformPoint)
  const xs = corners.map(point => point.x)
  const ys = corners.map(point => point.y)
  const x = Math.min(...xs)
  const y = Math.min(...ys)
  return { x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y }
})

const modeNotes: Record<Mode, string> = {
  original: '原图与标签共享同一个坐标系。',
  flip: '水平翻转：x′ = W − x − w，y、w、h 不变。',
  letterbox: '等比例缩放后居中填充：框先缩放，再加左、上偏移。',
  rotate: '旋转后重新变换四个角，再求新的轴对齐外接框。'
}

function rounded(value: number) {
  return Math.round(value * 10) / 10
}
</script>

<template>
  <section class="geometry-lab" aria-label="图像与边界框同步变换实验">
    <header>
      <div><span>INTERACTIVE V2</span><h3>变换坐标台</h3></div>
      <code>{{ mode }}</code>
    </header>

    <div class="geometry-lab__modes" aria-label="选择几何变换">
      <button v-for="item in (['original','flip','letterbox','rotate'] as Mode[])" :key="item" type="button" :class="{ active: mode === item }" :aria-pressed="mode === item" @click="mode = item">
        {{ { original: '原图', flip: '水平翻转', letterbox: '等比填充', rotate: '旋转 15°' }[item] }}
      </button>
    </div>

    <div class="geometry-lab__workbench">
      <div class="geometry-lab__canvas">
        <svg :viewBox="`0 0 ${canvas.width} ${canvas.height}`" role="img" :aria-label="`${modeNotes[mode]} 当前边界框坐标显示在右侧`">
          <rect x="0" y="0" :width="canvas.width" :height="canvas.height" class="canvas" />
          <g :transform="svgTransform">
            <rect x="0" y="0" :width="sourceWidth" :height="sourceHeight" class="image" />
            <path d="M12 92 C38 64 46 86 71 48 S118 25 150 10" class="texture" />
            <rect :x="sourceBox.x" :y="sourceBox.y" :width="sourceBox.width" :height="sourceBox.height" class="part" />
            <path d="M51 71 L104 39" class="scratch" />
          </g>
          <rect :x="transformedBox.x" :y="transformedBox.y" :width="transformedBox.width" :height="transformedBox.height" class="bbox" />
        </svg>
      </div>

      <div class="geometry-lab__ledger">
        <p>{{ modeNotes[mode] }}</p>
        <dl>
          <div><dt>source</dt><dd>[42, 30, 74, 54]</dd></div>
          <div><dt>target x</dt><dd>{{ rounded(transformedBox.x) }}</dd></div>
          <div><dt>target y</dt><dd>{{ rounded(transformedBox.y) }}</dd></div>
          <div><dt>target w</dt><dd>{{ rounded(transformedBox.width) }}</dd></div>
          <div><dt>target h</dt><dd>{{ rounded(transformedBox.height) }}</dd></div>
          <div><dt>canvas</dt><dd>{{ canvas.width }} × {{ canvas.height }}</dd></div>
        </dl>
      </div>
    </div>
  </section>
</template>

<style scoped>
.geometry-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.geometry-lab header{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}.geometry-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.geometry-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.geometry-lab header code{padding:8px 10px;border-radius:7px;color:#17302b;background:#f3c56f}.geometry-lab__modes{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:20px}.geometry-lab__modes button{min-height:44px;padding:8px;border:1px solid #55766c;border-radius:7px;color:#bed1ca;background:transparent;font-size:11px;font-weight:750;cursor:pointer}.geometry-lab__modes button.active{border-color:#f3c56f;color:#17302b;background:#f3c56f}.geometry-lab__workbench{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(210px,.65fr);gap:16px;margin-top:16px}.geometry-lab__canvas,.geometry-lab__ledger{padding:16px;border-radius:10px;background:#0c2420}.geometry-lab svg{display:block;width:100%;height:260px}.geometry-lab svg .canvas{fill:#081b18;stroke:#45675e;stroke-width:1}.geometry-lab svg .image{fill:#1a3b35;stroke:#76988e;stroke-width:1}.geometry-lab svg .texture{fill:none;stroke:#315c53;stroke-width:3}.geometry-lab svg .part{fill:#d88751;stroke:#f1b177;stroke-width:1}.geometry-lab svg .scratch{fill:none;stroke:#eaf2ef;stroke-width:3}.geometry-lab svg .bbox{fill:none;stroke:#f3c56f;stroke-width:2;stroke-dasharray:5 3;vector-effect:non-scaling-stroke}.geometry-lab__ledger>p{min-height:52px;margin:0 0 12px;color:#b9cec7;font-size:11px;line-height:1.6}.geometry-lab dl{margin:0}.geometry-lab dl div{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid #294a43}.geometry-lab dt{color:#91aaa2;font-size:10px}.geometry-lab dd{margin:0;color:#fff;font:700 11px ui-monospace,monospace}@media(max-width:720px){.geometry-lab header{flex-direction:column}.geometry-lab__modes{grid-template-columns:repeat(2,1fr)}.geometry-lab__workbench{grid-template-columns:1fr}.geometry-lab svg{height:230px}}
</style>
