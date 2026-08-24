<script setup lang="ts">
import { computed, ref } from 'vue'

type Operation = 'raw' | 'open' | 'close'
type Region = { area: number; minRow: number; maxRow: number; minColumn: number; maxColumn: number }

const rows = 12
const columns = 16
const threshold = ref(128)
const operation = ref<Operation>('raw')

const image = Array.from({ length: rows * columns }, (_, index) => {
  const row = Math.floor(index / columns)
  const column = index % columns
  let value = 38 + column * 2 + row
  if (row >= 2 && row <= 8 && column >= 2 && column <= 6) value = 184
  if ((row - 6) ** 2 + (column - 11) ** 2 <= 9) value = 214
  if (row === 5 && column === 4) value = 62
  if ([[1,14],[9,1],[10,8]].some(([r,c]) => r === row && c === column)) value = 242
  return value
})

function erode(mask: boolean[]) {
  return mask.map((_, index) => {
    const row = Math.floor(index / columns)
    const column = index % columns
    for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
      for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
        const neighborRow = row + rowOffset
        const neighborColumn = column + columnOffset
        if (neighborRow < 0 || neighborRow >= rows || neighborColumn < 0 || neighborColumn >= columns) return false
        if (!mask[neighborRow * columns + neighborColumn]) return false
      }
    }
    return true
  })
}

function dilate(mask: boolean[]) {
  return mask.map((_, index) => {
    const row = Math.floor(index / columns)
    const column = index % columns
    for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
      for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
        const neighborRow = row + rowOffset
        const neighborColumn = column + columnOffset
        if (neighborRow < 0 || neighborRow >= rows || neighborColumn < 0 || neighborColumn >= columns) continue
        if (mask[neighborRow * columns + neighborColumn]) return true
      }
    }
    return false
  })
}

const rawMask = computed(() => image.map(value => value >= threshold.value))
const mask = computed(() => {
  if (operation.value === 'open') return dilate(erode(rawMask.value))
  if (operation.value === 'close') return erode(dilate(rawMask.value))
  return rawMask.value
})

const regions = computed<Region[]>(() => {
  const visited = new Set<number>()
  const result: Region[] = []
  for (let start = 0; start < mask.value.length; start++) {
    if (!mask.value[start] || visited.has(start)) continue
    const queue = [start]
    visited.add(start)
    const cells: number[] = []
    while (queue.length) {
      const current = queue.shift()!
      cells.push(current)
      const row = Math.floor(current / columns)
      const column = current % columns
      for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
        for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
          if (rowOffset === 0 && columnOffset === 0) continue
          const neighborRow = row + rowOffset
          const neighborColumn = column + columnOffset
          if (neighborRow < 0 || neighborRow >= rows || neighborColumn < 0 || neighborColumn >= columns) continue
          const neighbor = neighborRow * columns + neighborColumn
          if (mask.value[neighbor] && !visited.has(neighbor)) {
            visited.add(neighbor)
            queue.push(neighbor)
          }
        }
      }
    }
    const regionRows = cells.map(cell => Math.floor(cell / columns))
    const regionColumns = cells.map(cell => cell % columns)
    result.push({
      area: cells.length,
      minRow: Math.min(...regionRows),
      maxRow: Math.max(...regionRows),
      minColumn: Math.min(...regionColumns),
      maxColumn: Math.max(...regionColumns)
    })
  }
  return result.sort((left, right) => right.area - left.area)
})

const foregroundArea = computed(() => mask.value.filter(Boolean).length)
const operationLabel: Record<Operation, string> = { raw: '原始阈值', open: '开运算', close: '闭运算' }
</script>

<template>
  <section class="segmentation-lab" aria-label="阈值、形态学与连通区域测量实验">
    <header>
      <div><span>INTERACTIVE V3</span><h3>分割测量台</h3></div>
      <code>T={{ threshold }}</code>
    </header>

    <div class="segmentation-lab__controls">
      <label>阈值 {{ threshold }}<input v-model.number="threshold" type="range" min="60" max="220" step="4"></label>
      <div aria-label="选择形态学处理">
        <button v-for="item in (['raw','open','close'] as Operation[])" :key="item" type="button" :class="{ active: operation === item }" :aria-pressed="operation === item" @click="operation = item">{{ operationLabel[item] }}</button>
      </div>
    </div>

    <div class="segmentation-lab__boards">
      <figure>
        <figcaption>输入灰度</figcaption>
        <div class="segmentation-lab__viewport"><div class="segmentation-lab__grid">
          <i v-for="(value,index) in image" :key="index" :style="{ backgroundColor: `rgb(${value},${value},${value})` }" :title="`灰度 ${value}`" />
        </div></div>
      </figure>
      <figure>
        <figcaption>{{ operationLabel[operation] }} mask</figcaption>
        <div class="segmentation-lab__viewport"><div class="segmentation-lab__grid is-mask">
          <i v-for="(value,index) in mask" :key="index" :class="{ foreground: value }"><span v-if="value">·</span></i>
        </div></div>
      </figure>
    </div>

    <div class="segmentation-lab__metrics">
      <span>对象数 <b>{{ regions.length }}</b></span>
      <span>前景面积 <b>{{ foregroundArea }} px</b></span>
      <span>最大区域 <b>{{ regions[0]?.area ?? 0 }} px</b></span>
      <span>最大框 <b v-if="regions[0]">x{{ regions[0].minColumn }} y{{ regions[0].minRow }} w{{ regions[0].maxColumn-regions[0].minColumn+1 }} h{{ regions[0].maxRow-regions[0].minRow+1 }}</b><b v-else>—</b></span>
    </div>
    <p>提高阈值会减少前景；开运算倾向去掉孤立白点，闭运算倾向填补小孔和窄间隙。对象数和面积因此不是固定真值。</p>
  </section>
</template>

<style scoped>
.segmentation-lab{margin:32px 0;padding:25px;border-radius:16px;color:#edf4f1;background:#122f2a}.segmentation-lab header{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}.segmentation-lab header span{color:#f3c56f;font:800 10px/1 ui-monospace,monospace;letter-spacing:.12em}.segmentation-lab h3{margin:7px 0 0;border:0;color:#fff;font-size:19px}.segmentation-lab header code{padding:8px 10px;border-radius:7px;color:#17302b;background:#f3c56f}.segmentation-lab__controls{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:end;margin-top:20px}.segmentation-lab__controls label{color:#bed1ca;font-size:11px}.segmentation-lab__controls input{width:100%;margin-top:9px;accent-color:#f3c56f}.segmentation-lab__controls>div{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.segmentation-lab__controls button{min-height:44px;border:1px solid #55766c;border-radius:7px;color:#bed1ca;background:transparent;font-size:11px;font-weight:750;cursor:pointer}.segmentation-lab__controls button.active{border-color:#f3c56f;color:#17302b;background:#f3c56f}.segmentation-lab__boards{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:18px}.segmentation-lab figure{min-width:0;margin:0;padding:13px;border-radius:10px;background:#0c2420}.segmentation-lab figcaption{margin-bottom:9px;color:#9eb7af;font:750 10px ui-monospace,monospace}.segmentation-lab__viewport{overflow-x:auto}.segmentation-lab__grid{display:grid;grid-template-columns:repeat(16,17px);gap:2px;width:max-content;margin:auto}.segmentation-lab__grid i{display:block;width:17px;height:17px;border-radius:2px}.segmentation-lab__grid.is-mask i{display:grid;place-items:center;color:#143b37;background:#173a34;font-size:13px;font-style:normal}.segmentation-lab__grid.is-mask i.foreground{color:#7b410f;background:#f3c56f}.segmentation-lab__metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}.segmentation-lab__metrics span{padding:10px;border-radius:7px;color:#8fa9a0;background:#0c2420;font-size:10px}.segmentation-lab__metrics b{display:block;margin-top:5px;color:#fff;font:750 11px ui-monospace,monospace}.segmentation-lab>p{margin:13px 0 0;color:#9fb8b0;font-size:11px;line-height:1.65}@media(max-width:720px){.segmentation-lab header{flex-direction:column}.segmentation-lab__controls,.segmentation-lab__boards{grid-template-columns:1fr}.segmentation-lab__metrics{grid-template-columns:1fr 1fr}.segmentation-lab__grid{margin:0}}
</style>
