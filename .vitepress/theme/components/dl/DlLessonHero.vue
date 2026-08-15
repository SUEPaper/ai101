<script setup lang="ts">
defineProps<{
  number: string
  module: string
  title: string
  question: string
  duration: string
  outcomes: string[]
}>()
</script>

<template>
  <header class="dl-lesson-hero">
    <div class="dl-lesson-hero__signal" aria-hidden="true">
      <span v-for="index in 7" :key="index" :style="{ '--delay': `${index * 0.16}s` }" />
    </div>
    <div class="dl-lesson-hero__meta"><span>CHAPTER {{ number }} · {{ module }}</span><span>{{ duration }}</span></div>
    <p class="dl-lesson-hero__question">{{ question }}</p>
    <h1>{{ title }}</h1>
    <ol class="dl-lesson-hero__rail" aria-label="本章证据链">
      <li><i>DATA</i>输入合同</li><li><i>STATE</i>内部状态</li><li><i>TRACE</i>观测信号</li><li><i>GATE</i>发布门禁</li>
    </ol>
    <div class="dl-lesson-hero__outcomes">
      <strong>学完后你能够</strong>
      <ul><li v-for="outcome in outcomes" :key="outcome">{{ outcome }}</li></ul>
    </div>
  </header>
</template>

<style scoped>
.dl-lesson-hero{position:relative;margin:0 0 48px;padding:36px;overflow:hidden;border:1px solid #244a59;border-radius:11px;color:#edf8f7;background:#13213c;box-shadow:8px 8px 0 #acd9d5}.dl-lesson-hero:before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(91,168,167,.1) 1px,transparent 1px),linear-gradient(90deg,rgba(91,168,167,.1) 1px,transparent 1px);background-size:24px 24px;mask-image:linear-gradient(90deg,transparent 0,#000 48%)}.dl-lesson-hero__signal{position:absolute;right:2%;top:22%;display:flex;align-items:center;width:22%;height:30%;opacity:.78}.dl-lesson-hero__signal:before{content:'';position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,#58d3c8,#f2c14e,#df6c5b)}.dl-lesson-hero__signal span{position:relative;width:9px;height:36px;margin-right:12px;border:1px solid #55828c;border-radius:2px;background:#1c304d;animation:signal 2.4s ease-in-out infinite;animation-delay:var(--delay)}.dl-lesson-hero__signal span:nth-child(3n){height:58px}.dl-lesson-hero__signal span:nth-child(2n):after{content:'';position:absolute;left:3px;right:3px;bottom:4px;height:12px;background:#f2c14e}.dl-lesson-hero__meta{position:relative;z-index:1;display:flex;justify-content:space-between;width:58%;color:#92b9c2;font:750 10px/1.4 ui-monospace,monospace;letter-spacing:.1em}.dl-lesson-hero__question{position:relative;z-index:1;max-width:62%;margin:36px 0 8px;color:#65d6cd;font-size:13px;font-weight:750}.dl-lesson-hero h1{position:relative;z-index:1;max-width:72%;margin:0;border:0;color:#fff;font:760 clamp(34px,5vw,50px)/1.09 "Songti SC","Noto Serif SC",serif;letter-spacing:-.03em}.dl-lesson-hero__rail{position:relative;z-index:1;display:grid;grid-template-columns:repeat(4,1fr);margin:34px 0 0;padding:18px 0 0;border-top:1px solid #315064;list-style:none}.dl-lesson-hero__rail li{margin:0;color:#d5e4e7;font-size:11px}.dl-lesson-hero__rail i{display:inline-grid;min-width:42px;place-items:center;margin-right:6px;padding:5px;border:1px solid #48717a;border-radius:4px;color:#75ded5;font:800 8px/1 ui-monospace,monospace;font-style:normal}.dl-lesson-hero__outcomes{position:relative;z-index:1;display:grid;grid-template-columns:150px 1fr;gap:20px;margin-top:27px;padding-top:23px;border-top:1px solid #315064}.dl-lesson-hero__outcomes strong{color:#f2c14e;font-size:12px}.dl-lesson-hero__outcomes ul{display:grid;gap:8px;margin:0;padding:0;list-style:none}.dl-lesson-hero__outcomes li{position:relative;margin:0;padding-left:17px;color:#e2ecee;font-size:13px;line-height:1.6}.dl-lesson-hero__outcomes li:before{content:'';position:absolute;left:0;top:.65em;width:7px;height:2px;background:#df6c5b}@keyframes signal{0%,100%{transform:translateY(0);filter:brightness(.8)}50%{transform:translateY(-8px);filter:brightness(1.3)}}@media(max-width:680px){.dl-lesson-hero{padding:27px 22px}.dl-lesson-hero__signal{width:70%;opacity:.12}.dl-lesson-hero__meta,.dl-lesson-hero__question,.dl-lesson-hero h1{max-width:100%;width:100%}.dl-lesson-hero__rail{grid-template-columns:1fr 1fr;gap:11px}.dl-lesson-hero__outcomes{grid-template-columns:1fr;gap:10px}}@media(prefers-reduced-motion:reduce){.dl-lesson-hero__signal span{animation:none}}
</style>
