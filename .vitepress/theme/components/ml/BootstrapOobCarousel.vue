<script setup lang="ts">
import { computed, ref } from 'vue'
import MlExperimentFrame from './MlExperimentFrame.vue'

const sampleCount = ref(40)
const seed = ref(7)
const treeCount = ref(120)

const lcg = (initial: number) => {
  let state = initial >>> 0
  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 4294967296
  }
}

const bootstrap = computed(() => {
  const random = lcg(seed.value * 104729 + sampleCount.value)
  const counts = Array.from({ length: sampleCount.value }, () => 0)
  const tickets: number[] = []
  for (let draw = 0; draw < sampleCount.value; draw += 1) {
    const index = Math.floor(random() * sampleCount.value)
    counts[index] += 1
    if (draw < 18) tickets.push(index)
  }
  const carriers = counts.map((count, index) => {
    const angle = -Math.PI / 2 + (index / sampleCount.value) * Math.PI * 2
    return {
      index,
      count,
      x: 230 + 151 * Math.cos(angle),
      y: 169 + 121 * Math.sin(angle),
    }
  })
  const unique = counts.filter(count => count > 0).length
  return { counts, carriers, tickets, unique, oob: sampleCount.value - unique }
})

const exactUniqueRate = computed(() => 1 - Math.pow(1 - 1 / sampleCount.value, sampleCount.value))
const exactOobRate = computed(() => 1 - exactUniqueRate.value)
const expectedOobVotes = computed(() => treeCount.value * exactOobRate.value)
const zeroVoteRisk = computed(() => Math.pow(1 - exactOobRate.value, treeCount.value))
</script>

<template>
  <MlExperimentFrame
    index="20-A"
    title="让晶圆批次走一圈 Bootstrap 载具环"
    question="有放回抽 n 次，为什么载具仍会空出约 36.8%，而另一些批次会重复进入同一棵树？树数又如何决定每批能拿到多少张 OOB 诊断票？"
    formula="P(i∉Dᵇ)=(1−1/n)ⁿ → e⁻¹"
  >
    <template #visual>
      <div class="carousel-scroll">
        <svg viewBox="0 0 610 342" role="img" :aria-label="`${sampleCount}个批次中，本轮${bootstrap.unique}个至少入袋一次，${bootstrap.oob}个袋外`">
          <rect width="610" height="342" rx="14" class="plate" />
          <text x="24" y="24" class="eyebrow">ETCH LINE E7 · GROUP BOOTSTRAP CAROUSEL</text>
          <circle cx="230" cy="169" r="132" class="outer-ring" />
          <circle cx="230" cy="169" r="92" class="inner-ring" />
          <circle cx="230" cy="169" r="49" class="hub" />
          <text x="230" y="158" text-anchor="middle" class="hub-kicker">BOOT</text>
          <text x="230" y="180" text-anchor="middle" class="hub-value">{{ sampleCount }}×</text>
          <text x="230" y="197" text-anchor="middle" class="hub-note">有放回抽取</text>
          <g v-for="carrier in bootstrap.carriers" :key="carrier.index">
            <circle :cx="carrier.x" :cy="carrier.y" r="9" :class="carrier.count ? 'carrier inbag' : 'carrier oob'" />
            <text :x="carrier.x" :y="carrier.y + 3" text-anchor="middle" class="carrier-count">{{ carrier.count }}</text>
          </g>
          <path d="M392 47 H584 V294 H392" class="ticket-rack" />
          <text x="410" y="70" class="rack-title">前 18 张抽样票</text>
          <g v-for="(ticket, index) in bootstrap.tickets" :key="`${index}-${ticket}`">
            <rect :x="410 + (index % 3) * 54" :y="83 + Math.floor(index / 3) * 30" width="44" height="21" rx="4" class="ticket" />
            <text :x="432 + (index % 3) * 54" :y="97 + Math.floor(index / 3) * 30" text-anchor="middle" class="ticket-id">L{{ String(ticket + 1).padStart(2, '0') }}</text>
          </g>
          <circle cx="420" cy="291" r="6" class="carrier inbag" /><text x="433" y="295" class="legend">入袋，数字为重复次数</text>
          <circle cx="420" cy="315" r="6" class="carrier oob" /><text x="433" y="319" class="legend">OOB，本棵树未见</text>
        </svg>
      </div>
    </template>
    <template #controls>
      <label class="control"><span>训练批次数 n <output>{{ sampleCount }}</output></span><input v-model.number="sampleCount" aria-label="训练批次数" type="range" min="16" max="60" step="4"></label>
      <label class="control"><span>重采样编号 <output>{{ seed }}</output></span><input v-model.number="seed" aria-label="重采样编号" type="range" min="1" max="40" step="1"></label>
      <label class="control"><span>森林树数 B <output>{{ treeCount }}</output></span><input v-model.number="treeCount" aria-label="森林树数" type="range" min="5" max="300" step="5"></label>
      <p class="hint">换一个重采样编号，实际 OOB 比例会波动；公式给的是对重采样随机性的期望。这里一个圆点代表完整晶圆批次，而不是同批次中的单片晶圆。</p>
    </template>
    <template #metrics>
      <div><span>本轮唯一样本</span><strong>{{ bootstrap.unique }} / {{ sampleCount }}</strong></div>
      <div><span>本轮 OOB 比例</span><strong>{{ (100 * bootstrap.oob / sampleCount).toFixed(1) }}%</strong></div>
      <div><span>精确期望 OOB</span><strong>{{ (100 * exactOobRate).toFixed(1) }}%</strong></div>
      <div><span>每批期望 OOB 票</span><strong>{{ expectedOobVotes.toFixed(1) }} · 零票 {{ (100 * zeroVoteRisk).toFixed(3) }}%</strong></div>
    </template>
    <template #note>行级重复测量不能各自抽签。若部署目标是新晶圆批次，就应以批次为抽样单位：某批要么整组进入一棵树，要么整组成为该树的 OOB；否则同批兄弟样本会让 OOB 误差虚假乐观。</template>
  </MlExperimentFrame>
</template>

<style scoped>
.carousel-scroll{overflow-x:auto}.carousel-scroll svg{display:block;min-width:580px;width:100%;height:auto}.plate{fill:#0f1822}.outer-ring{fill:#14232f;stroke:#526679;stroke-width:2}.inner-ring{fill:#101c26;stroke:#2f4657;stroke-dasharray:4 5}.hub{fill:#27213e;stroke:#8b7cf6;stroke-width:2}.eyebrow,.hub-kicker,.hub-value,.hub-note,.carrier-count,.rack-title,.ticket-id,.legend{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.eyebrow{fill:#67d3dc;font-size:9px;letter-spacing:.1em}.hub-kicker{fill:#b9adff;font-size:9px;letter-spacing:.16em}.hub-value{fill:#fff;font-size:20px;font-weight:850}.hub-note{fill:#9aaaba;font-size:8px}.carrier{stroke-width:1.5}.carrier.inbag{fill:#8b7cf6;stroke:#d8d2ff}.carrier.oob{fill:#e7b857;stroke:#fff0bd}.carrier-count{fill:#101722;font-size:7px;font-weight:850}.ticket-rack{fill:#111f2a;stroke:#3d5262;stroke-width:1.5}.rack-title{fill:#81dce3;font-size:9px;letter-spacing:.08em}.ticket{fill:#1c2b37;stroke:#506779}.ticket-id{fill:#dce6ee;font-size:8px}.legend{fill:#9eb0bd;font-size:8px}.control{display:grid;gap:8px}.control>span{display:flex;justify-content:space-between;color:#aebbdc;font-size:11px}.control output{color:#f0c76b;font:750 11px/1 ui-monospace,monospace}input[type="range"]{width:100%;min-height:44px;accent-color:#8b7cf6}.hint{margin:0;padding:11px;border-left:3px solid #3bc7d3;color:#a8b9c5;background:#101b25;font-size:10px;line-height:1.7}input:focus-visible{outline:3px solid #65d8e1;outline-offset:2px}@media(max-width:560px){.carousel-scroll{margin-right:-18px;padding-right:18px}.carousel-scroll svg{width:580px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
