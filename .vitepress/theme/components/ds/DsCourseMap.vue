<script setup lang="ts">
import { withBase } from 'vitepress'
import { dsModules } from '../../data/dsCourse'
</script>

<template>
  <div class="ds-map" aria-label="数据科学课程地图">
    <section v-for="(module, moduleIndex) in dsModules" :key="module.title" class="ds-map__module">
      <header>
        <span>{{ String(moduleIndex).padStart(2, '0') }}</span>
        <div>
          <h2>{{ module.title }}</h2>
          <p>{{ module.summary }}</p>
        </div>
      </header>
      <div class="ds-map__lessons">
        <template v-for="lesson in module.lessons" :key="lesson.number">
          <a v-if="lesson.available" :href="withBase(lesson.href)">
            <b>{{ lesson.number }}</b>
            <span>{{ lesson.title }}</span>
            <small>开始学习 →</small>
          </a>
          <div v-else class="is-planned">
            <b>{{ lesson.number }}</b>
            <span>{{ lesson.title }}</span>
            <small>建设中</small>
          </div>
        </template>
      </div>
    </section>
  </div>
</template>

<style scoped>
.ds-map { margin: 38px 0 56px; }
.ds-map__module { display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 28px; padding: 28px 0; border-top: 1px solid #cfdad4; }
.ds-map__module > header { display: grid; grid-template-columns: 34px 1fr; gap: 12px; align-content: start; }
.ds-map__module > header > span { color: #c07b29; font: 800 12px/1.6 ui-monospace, monospace; }
.ds-map__module h2 { margin: 0; border: 0; color: #10221f; font: 760 22px/1.2 "Songti SC", "Noto Serif SC", serif; }
.ds-map__module p { margin: 7px 0 0; color: #66736f; font-size: 12px; line-height: 1.65; }
.ds-map__lessons { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.ds-map__lessons a,
.ds-map__lessons > div { min-height: 92px; padding: 15px; border: 1px solid #d6e1db; border-radius: 12px; background: #fff; text-decoration: none; transition: transform .18s ease, border-color .18s ease; }
.ds-map__lessons a:hover { transform: translateY(-2px); border-color: #2f7c7a; }
.ds-map__lessons b { display: block; color: #c07b29; font: 800 11px/1.2 ui-monospace, monospace; }
.ds-map__lessons span { display: block; margin-top: 8px; color: #16302b; font-size: 14px; font-weight: 750; }
.ds-map__lessons small { display: block; margin-top: 9px; color: #65736f; font-size: 11px; }
.ds-map__lessons .is-planned { background: #f1f4f1; }
.ds-map__lessons .is-planned span,
.ds-map__lessons .is-planned small { color: #929c98; }
@media (max-width: 700px) {
  .ds-map__module { grid-template-columns: 1fr; gap: 16px; }
  .ds-map__lessons { grid-template-columns: 1fr; }
}
</style>
