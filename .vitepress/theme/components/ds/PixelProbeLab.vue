<script setup lang="ts">
import { computed, ref } from 'vue'

type DisplayMode = 'RGB' | 'BGR' | 'GRAY'

const row = ref(2)
const column = ref(3)
const mode = ref<DisplayMode>('RGB')

const pixels = Array.from({ length: 48 }, (_, index) => {
  const currentRow = Math.floor(index / 8)
  const currentColumn = index % 8
  if (currentRow >= 1 && currentRow <= 4 && currentColumn >= 2 && currentColumn <= 5) {
    if (currentRow >= 2 && currentRow <= 3 && currentColumn >= 3 && currentColumn <= 4) {
      return { r: 38, g: 92, b: 190 }
    }
    return { r: 226, g: 152, b: 42 }
  }
  return { r: 62, g: 46, b: 28 }
})

const selected = computed(() => pixels[row.value * 8 + column.value])
const gray = computed(() => Math.round(.299 * selected.value.r + .587 * selected.value.g + .114 * selected.value.b))
const value = computed(() => {
  if (mode.value === 'GRAY') return `[${gray.value}]`
  if (mode.value === 'BGR') return `[${selected.value.b}, ${selected.value.g}, ${selected.value.r}]`
  return `[${selected.value.r}, ${selected.value.g}, ${selected.value.b}]`
})

function selectPixel(index: number) {
  row.value = Math.floor(index / 8)
  column.value = index % 8
}
</script>

<template>
  <section class="pixel-probe" aria-label="像素坐标与通道顺序实验">
    <header>
      <div>
        <span>INTERACTIVE V1</span>
        <h3>像素探针板</h3>
      </div>
      <code>image[y, x]</code>
    </header>

    <div class="pixel-probe__layout">
      <div class="pixel-probe__viewport" aria-label="可选择的 6 行 8 列像素图">
        <div class="pixel-probe__grid">
          <button
            v-for="(pixel, index) in pixels"
            :key="index"
            type="button"
            :class="{ 'is-selected': row * 8 + column === index }"
            :style="{ backgroundColor: `rgb(${pixel.r}, ${pixel.g}, ${pixel.b})` }"
            :aria-label="`选择第 ${Math.floor(index / 8)} 行、第 ${index % 8} 列像素`"
            :aria-pressed="row * 8 + column === index"
            @click="selectPixel(index)"
          />
        </div>
      </div>

      <div class="pixel-probe__console">
        <div class="pixel-probe__modes" aria-label="通道解释方式">
          <button
            v-for="item in (['RGB', 'BGR', 'GRAY'] as DisplayMode[])"
            :key="item"
            type="button"
            :class="{ active: mode === item }"
            :aria-pressed="mode === item"
            @click="mode = item"
          >{{ item }}</button>
        </div>
        <dl>
          <div><dt>数组索引</dt><dd>[{{ row }}, {{ column }}]</dd></div>
          <div><dt>图像坐标</dt><dd>(x={{ column }}, y={{ row }})</dd></div>
          <div><dt>{{ mode }} 值</dt><dd>{{ value }}</dd></div>
          <div><dt>shape</dt><dd>(6, 8, {{ mode === 'GRAY' ? 1 : 3 }})</dd></div>
          <div><dt>dtype</dt><dd>uint8</dd></div>
        </dl>
        <p>选择任一像素，比较 <b>[row, column]</b> 与 <b>(x, y)</b>，再切换通道解释。</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.pixel-probe{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.pixel-probe header{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}.pixel-probe header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.pixel-probe h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.pixel-probe header code{padding:8px 10px;border-radius:7px;color:#17302b;background:#f3c56f}.pixel-probe__layout{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(230px,.8fr);gap:18px;margin-top:22px}.pixel-probe__viewport{overflow-x:auto;padding:12px;border-radius:10px;background:#0c2420}.pixel-probe__grid{display:grid;grid-template-columns:repeat(8,44px);width:max-content;margin:auto;border:1px solid #55766c}.pixel-probe__grid button{position:relative;width:44px;height:44px;padding:0;border:1px solid rgba(255,255,255,.2);cursor:pointer}.pixel-probe__grid button::after{content:'';position:absolute;inset:5px;border:2px solid transparent}.pixel-probe__grid button:hover::after,.pixel-probe__grid button:focus-visible::after{border-color:#fff}.pixel-probe__grid button.is-selected::after{border-color:#f3c56f;box-shadow:0 0 0 2px #102d29}.pixel-probe__console{padding:16px;border-radius:10px;background:#0c2420}.pixel-probe__modes{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.pixel-probe__modes button{min-height:44px;border:1px solid #55766c;border-radius:7px;color:#bed1ca;background:transparent;font:750 10px ui-monospace,monospace;cursor:pointer}.pixel-probe__modes button.active{border-color:#f3c56f;color:#17302b;background:#f3c56f}.pixel-probe dl{margin:15px 0}.pixel-probe dl div{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid #294a43}.pixel-probe dt{color:#91aaa2;font-size:10px}.pixel-probe dd{margin:0;color:#fff;font:700 11px ui-monospace,monospace}.pixel-probe p{margin:12px 0 0;color:#9fb8b0;font-size:11px;line-height:1.65}.pixel-probe p b{color:#f3c56f}@media(max-width:720px){.pixel-probe header{flex-direction:column}.pixel-probe__layout{grid-template-columns:1fr}.pixel-probe__grid{margin:0}}
</style>
