<script setup lang="ts">
defineProps<{
  number: string
  title: string
  question: string
  duration: string
  outcomes: string[]
}>()
</script>

<template>
  <header class="ml-lesson-hero">
    <div class="ml-lesson-hero__grid" aria-hidden="true">
      <svg viewBox="0 0 290 200">
        <path d="M12 174 C55 171 61 76 111 95 S184 130 210 52 S264 25 282 18" />
        <path d="M12 156 C53 148 78 42 123 72 S192 111 224 40 S270 37 282 35" />
        <circle cx="111" cy="95" r="5" /><circle cx="210" cy="52" r="5" />
        <line x1="111" y1="95" x2="155" y2="62" /><path class="arrow" d="M155 62 l-10 1 m10-1 l-3 10" />
      </svg>
    </div>
    <div class="ml-lesson-hero__meta"><span>CHAPTER {{ number }}</span><span>{{ duration }}</span></div>
    <p class="ml-lesson-hero__question">{{ question }}</p>
    <h1>{{ title }}</h1>
    <ol class="ml-lesson-hero__rail" aria-label="本章学习路径">
      <li><i>1</i>提出问题</li><li><i>2</i>写出假设</li><li><i>3</i>逐步推导</li><li><i>4</i>数值验证</li>
    </ol>
    <div class="ml-lesson-hero__outcomes">
      <strong>学完后你能够</strong>
      <ul><li v-for="outcome in outcomes" :key="outcome">{{ outcome }}</li></ul>
    </div>
  </header>
</template>

<style scoped>
.ml-lesson-hero{position:relative;margin:0 0 48px;padding:36px;overflow:hidden;border-radius:14px;color:#edf2ff;background:#17203b;box-shadow:7px 7px 0 #cbd5f2}.ml-lesson-hero__grid{position:absolute;inset:0 0 auto auto;width:42%;height:100%;opacity:.42}.ml-lesson-hero__grid:before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(112,137,206,.22) 1px,transparent 1px),linear-gradient(90deg,rgba(112,137,206,.22) 1px,transparent 1px);background-size:22px 22px}.ml-lesson-hero__grid svg{position:absolute;z-index:1;right:0;bottom:5%;width:95%}.ml-lesson-hero__grid path{fill:none;stroke:#72d5ce;stroke-width:2}.ml-lesson-hero__grid path:nth-child(2){stroke:#e15b4b}.ml-lesson-hero__grid circle{fill:#f4c95d}.ml-lesson-hero__grid line{stroke:#fff;stroke-width:1.5}.ml-lesson-hero__grid .arrow{stroke:#fff}.ml-lesson-hero__meta{position:relative;z-index:2;display:flex;justify-content:space-between;width:58%;color:#9fb2e7;font:750 10px/1.4 ui-monospace,monospace;letter-spacing:.1em}.ml-lesson-hero__question{position:relative;z-index:2;max-width:58%;margin:36px 0 8px;color:#78d6cf;font-size:13px;font-weight:750}.ml-lesson-hero h1{position:relative;z-index:2;max-width:610px;margin:0;border:0;color:#fff;font:760 clamp(34px,6vw,56px)/1.09 "Songti SC","Noto Serif SC",serif;letter-spacing:-.03em}.ml-lesson-hero__rail{position:relative;z-index:2;display:grid;grid-template-columns:repeat(4,1fr);margin:34px 0 0;padding:18px 0 0;border-top:1px solid #364469;list-style:none}.ml-lesson-hero__rail li{margin:0;color:#dce4fa;font-size:11px}.ml-lesson-hero__rail i{display:inline-grid;place-items:center;width:21px;height:21px;margin-right:6px;border:1px solid #7288c8;border-radius:5px;color:#79d6cf;font:800 9px/1 ui-monospace,monospace;font-style:normal}.ml-lesson-hero__outcomes{position:relative;z-index:2;display:grid;grid-template-columns:150px 1fr;gap:20px;margin-top:27px;padding-top:23px;border-top:1px solid #364469}.ml-lesson-hero__outcomes strong{color:#f4c95d;font-size:12px}.ml-lesson-hero__outcomes ul{display:grid;gap:8px;margin:0;padding:0;list-style:none}.ml-lesson-hero__outcomes li{position:relative;margin:0;padding-left:17px;color:#e6ebf8;font-size:13px;line-height:1.6}.ml-lesson-hero__outcomes li:before{content:'';position:absolute;left:0;top:.65em;width:6px;height:6px;background:#e15b4b}@media(max-width:680px){.ml-lesson-hero{padding:27px 22px}.ml-lesson-hero__grid{width:65%;opacity:.18}.ml-lesson-hero__meta,.ml-lesson-hero__question{max-width:100%;width:100%}.ml-lesson-hero__rail{grid-template-columns:1fr 1fr;gap:11px}.ml-lesson-hero__outcomes{grid-template-columns:1fr;gap:10px}}
</style>
