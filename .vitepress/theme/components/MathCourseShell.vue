<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, withBase } from 'vitepress'
import { mathModules } from '../data/mathCourse'

const route = useRoute()
const currentPath = computed(() => normalizePath(route.path))

function normalizePath(path: string) {
  return path.replace(/\.html$/, '').replace(/\/index$/, '').replace(/\/$/, '')
}

function isActive(href: string) {
  return currentPath.value === normalizePath(withBase(href))
}
</script>

<template>
  <div class="math-shell">
    <aside class="math-sidebar" aria-label="数值分析课程目录">
      <a class="math-home" :href="withBase('/courses/numerical-computing')">数值分析</a>
      <nav>
        <details v-for="module in mathModules" :key="module.title" :open="module.lessons.some((lesson) => isActive(lesson.href))">
          <summary>{{ module.title }}</summary>
          <a
            v-for="lesson in module.lessons"
            :key="lesson.href"
            :href="withBase(lesson.href)"
            :aria-current="isActive(lesson.href) ? 'page' : undefined"
          >{{ lesson.title }}</a>
        </details>
      </nav>
    </aside>
    <article class="math-content vp-doc">
      <slot />
    </article>
  </div>
</template>

<style scoped>
.math-shell {
  display: grid;
  grid-template-columns: 250px minmax(0, 820px);
  justify-content: space-between;
  gap: 48px;
  width: min(1240px, 100%);
  margin: 0 auto;
  padding: 28px 24px 88px;
}

.math-sidebar {
  position: sticky;
  top: 108px;
  align-self: start;
  max-height: calc(100vh - 132px);
  overflow: auto;
  scrollbar-width: thin;
}

.math-home {
  display: flex;
  align-items: center;
  min-height: 44px;
  margin-bottom: 16px;
  color: #143b37;
  font-size: 18px;
  font-weight: 800;
  text-decoration: none;
}

.math-sidebar details {
  border-top: 1px solid #dbe5df;
}

.math-sidebar summary {
  min-height: 44px;
  padding: 12px 8px;
  color: #17201f;
  font-size: 14px;
  font-weight: 700;
  cursor: pointer;
}

.math-sidebar nav a {
  display: flex;
  align-items: center;
  min-height: 44px;
  padding: 8px 12px;
  border-radius: 6px;
  color: #52625e;
  font-size: 13px;
  line-height: 1.6;
  text-decoration: none;
}

.math-sidebar nav a:hover,
.math-sidebar nav a[aria-current='page'] {
  color: #143b37;
  background: #e6eee8;
}

.math-sidebar a:focus-visible,
.math-sidebar summary:focus-visible {
  outline: 2px solid #2f7c7a;
  outline-offset: -2px;
}

.math-content {
  min-width: 0;
  color: #2f3d39;
  font-size: 16px;
  line-height: 1.85;
  overflow-wrap: anywhere;
}

.math-content :deep(img) {
  max-width: 100%;
  height: auto;
}

.math-content :deep(iframe) {
  max-width: 100%;
}

.math-content :deep(mjx-container) {
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
}

@media (max-width: 959px) {
  .math-shell {
    grid-template-columns: minmax(0, 1fr);
    gap: 24px;
    padding: 20px 20px 64px;
  }

  .math-sidebar {
    position: static;
    max-height: 320px;
    border-bottom: 1px solid #dbe5df;
  }
}
</style>
