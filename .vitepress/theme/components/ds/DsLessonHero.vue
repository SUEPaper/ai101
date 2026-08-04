<script setup lang="ts">
defineProps<{
  number: string
  title: string
  eyebrow: string
  duration: string
  outcomes: string[]
}>()
</script>

<template>
  <header class="ds-lesson-hero">
    <div class="ds-lesson-hero__meta">
      <span>LESSON {{ number }}</span>
      <span>{{ duration }}</span>
    </div>
    <p class="ds-lesson-hero__eyebrow">{{ eyebrow }}</p>
    <h1>{{ title }}</h1>
    <div class="ds-lesson-hero__rail" aria-label="本讲学习路径">
      <span><i>1</i>观察现象</span>
      <span><i>2</i>拆开机制</span>
      <span><i>3</i>用代码验证</span>
    </div>
    <div class="ds-lesson-hero__outcomes">
      <strong>完成本讲后，你能够</strong>
      <ul>
        <li v-for="outcome in outcomes" :key="outcome">{{ outcome }}</li>
      </ul>
    </div>
  </header>
</template>

<style scoped>
.ds-lesson-hero { margin: 0 0 46px; padding: 34px; border-radius: 18px; color: #edf5f1; background: #102d29; box-shadow: 0 24px 70px rgba(16,45,41,.14); }
.ds-lesson-hero__meta { display: flex; justify-content: space-between; color: #a9c7bd; font: 750 11px/1.4 ui-monospace, monospace; letter-spacing: .08em; }
.ds-lesson-hero__eyebrow { margin: 36px 0 8px; color: #f3c56f; font-size: 13px; font-weight: 750; }
.ds-lesson-hero h1 { max-width: 680px; margin: 0; border: 0; color: #fff; font: 760 clamp(34px, 6vw, 56px)/1.08 "Songti SC", "Noto Serif SC", serif; letter-spacing: -.03em; }
.ds-lesson-hero__rail { display: grid; grid-template-columns: repeat(3, 1fr); margin: 34px 0 0; border-top: 1px solid #31534d; }
.ds-lesson-hero__rail span { position: relative; padding: 18px 8px 0 0; color: #d7e5df; font-size: 12px; }
.ds-lesson-hero__rail i { display: inline-grid; place-items: center; width: 22px; height: 22px; margin-right: 7px; border-radius: 50%; color: #102d29; background: #f3c56f; font: 800 10px/1 ui-monospace, monospace; font-style: normal; }
.ds-lesson-hero__outcomes { display: grid; grid-template-columns: 150px 1fr; gap: 20px; margin-top: 30px; padding-top: 24px; border-top: 1px solid #31534d; }
.ds-lesson-hero__outcomes strong { color: #f3c56f; font-size: 12px; }
.ds-lesson-hero__outcomes ul { display: grid; gap: 8px; margin: 0; padding: 0; list-style: none; }
.ds-lesson-hero__outcomes li { position: relative; margin: 0; padding-left: 18px; color: #e6efeb; font-size: 13px; line-height: 1.6; }
.ds-lesson-hero__outcomes li::before { content: ''; position: absolute; left: 0; top: .65em; width: 6px; height: 6px; border-radius: 50%; background: #df765b; }
@media (max-width: 640px) {
  .ds-lesson-hero { padding: 26px 22px; }
  .ds-lesson-hero__rail { grid-template-columns: 1fr; padding-top: 12px; }
  .ds-lesson-hero__rail span { padding-top: 9px; }
  .ds-lesson-hero__outcomes { grid-template-columns: 1fr; gap: 10px; }
}
</style>
