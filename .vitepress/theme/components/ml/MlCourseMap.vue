<script setup lang="ts">
import { withBase } from 'vitepress'
import { mlModules, mlPracticeModule, mlSupplementModule } from '../../data/mlCourse'
</script>

<template>
  <div class="ml-map" aria-label="机器学习课程地图">
    <section v-for="(module, index) in mlModules" :key="module.title" class="ml-map__module">
      <header>
        <span>M{{ String(index + 1).padStart(2, '0') }}</span>
        <div><h2>{{ module.title }}</h2><p>{{ module.summary }}</p></div>
      </header>
      <div class="ml-map__lessons">
        <a v-for="lesson in module.lessons" :key="lesson.number" :href="withBase(lesson.href)">
          <b>{{ lesson.number }}</b><span>{{ lesson.title }}</span><small>进入推导 →</small>
        </a>
      </div>
    </section>
    <section v-for="module in [mlSupplementModule, mlPracticeModule]" :key="module.title" class="ml-map__module is-secondary">
      <header><span>＋</span><div><h2>{{ module.title }}</h2><p>{{ module.summary }}</p></div></header>
      <div class="ml-map__lessons">
        <a v-for="lesson in module.lessons" :key="lesson.number" :href="withBase(lesson.href)">
          <b>{{ lesson.number }}</b><span>{{ lesson.title }}</span><small>查看内容 →</small>
        </a>
      </div>
    </section>
  </div>
</template>

<style scoped>
.ml-map{margin:42px 0 60px}.ml-map__module{display:grid;grid-template-columns:230px minmax(0,1fr);gap:30px;padding:30px 0;border-top:1px solid #cfd6e7}.ml-map__module>header{display:grid;grid-template-columns:38px 1fr;gap:12px;align-content:start}.ml-map__module>header>span{color:#e15b4b;font:850 10px/1.7 ui-monospace,monospace}.ml-map__module h2{margin:0;border:0;color:#17203b;font:760 21px/1.25 "Songti SC","Noto Serif SC",serif}.ml-map__module p{margin:8px 0 0;color:#6b758e;font-size:12px;line-height:1.65}.ml-map__lessons{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.ml-map__lessons a{position:relative;min-height:104px;padding:16px;border:1px solid #d9dfed;border-radius:10px;color:#17203b;background:#fff;text-decoration:none;box-shadow:0 8px 24px rgba(31,49,102,.04);transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease}.ml-map__lessons a:before{content:'';position:absolute;inset:0 auto 0 0;width:3px;border-radius:10px 0 0 10px;background:#2d51b8;opacity:0;transition:opacity .18s}.ml-map__lessons a:hover{transform:translateY(-2px);border-color:#8498d1;box-shadow:0 14px 28px rgba(31,49,102,.08)}.ml-map__lessons a:hover:before{opacity:1}.ml-map__lessons b{display:block;color:#2d51b8;font:850 10px/1.2 ui-monospace,monospace}.ml-map__lessons span{display:block;margin-top:8px;font-size:14px;font-weight:760}.ml-map__lessons small{display:block;margin-top:10px;color:#778198;font-size:11px}.ml-map__module.is-secondary .ml-map__lessons b{color:#16847d}@media(max-width:720px){.ml-map__module{grid-template-columns:1fr;gap:18px}.ml-map__lessons{grid-template-columns:1fr}}
</style>
