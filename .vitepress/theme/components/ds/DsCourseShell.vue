<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, withBase } from 'vitepress'
import { dsLessons, dsModules, normalizeDsPath } from '../../data/dsCourse'

const route = useRoute()
const completed = ref<string[]>([])
const storageKey = 'ai101-ds-completed'

const currentPath = computed(() => normalizeDsPath(route.path))
const availableLessons = computed(() => dsLessons.filter((lesson) => lesson.available))
const progress = computed(() => {
  if (!availableLessons.value.length) return 0
  const count = availableLessons.value.filter((lesson) => completed.value.includes(lesson.number)).length
  return Math.round((count / availableLessons.value.length) * 100)
})

function readProgress() {
  if (typeof window === 'undefined') return
  try {
    const value = JSON.parse(window.localStorage.getItem(storageKey) ?? '[]')
    completed.value = Array.isArray(value) ? value : []
  } catch {
    completed.value = []
  }
}

onMounted(() => {
  readProgress()
  window.addEventListener('ds-progress-updated', readProgress)
})

onUnmounted(() => {
  window.removeEventListener('ds-progress-updated', readProgress)
})
</script>

<template>
  <div class="ds-shell">
    <aside class="ds-sidebar" aria-label="数据科学课程目录">
      <div class="ds-sidebar__sticky">
        <a class="ds-sidebar__home" :href="withBase('/courses/data-science')">
          <span class="ds-sidebar__mark" aria-hidden="true">DS</span>
          <span>
            <strong>数据科学</strong>
            <small>从问题走到证据</small>
          </span>
        </a>

        <div class="ds-sidebar__progress" aria-label="当前开放课程学习进度">
          <div>
            <span>学习进度</span>
            <strong>{{ progress }}%</strong>
          </div>
          <span class="ds-sidebar__track"><span :style="{ width: `${progress}%` }" /></span>
        </div>

        <nav class="ds-sidebar__nav">
          <section v-for="module in dsModules" :key="module.title" class="ds-sidebar__module">
            <h2>{{ module.title }}</h2>
            <ul>
              <li v-for="lesson in module.lessons" :key="lesson.number">
                <a
                  v-if="lesson.available"
                  :class="{ 'is-active': currentPath === normalizeDsPath(lesson.href) }"
                  :href="withBase(lesson.href)"
                  :aria-current="currentPath === normalizeDsPath(lesson.href) ? 'page' : undefined"
                >
                  <span class="ds-sidebar__number" :class="{ 'is-done': completed.includes(lesson.number) }">
                    {{ completed.includes(lesson.number) ? '✓' : lesson.number }}
                  </span>
                  <span>{{ lesson.title }}</span>
                </a>
                <span v-else class="is-planned">
                  <span class="ds-sidebar__number">{{ lesson.number }}</span>
                  <span>{{ lesson.title }}</span>
                </span>
              </li>
            </ul>
          </section>
        </nav>
      </div>
    </aside>

    <details class="ds-mobile-index">
      <summary>课程目录与进度 · {{ progress }}%</summary>
      <nav aria-label="数据科学移动端课程目录">
        <template v-for="module in dsModules" :key="module.title">
          <strong>{{ module.title }}</strong>
          <a
            v-for="lesson in module.lessons.filter((item) => item.available)"
            :key="lesson.number"
            :href="withBase(lesson.href)"
          >{{ lesson.number }} · {{ lesson.title }}</a>
        </template>
      </nav>
    </details>

    <article class="ds-course-content vp-doc">
      <slot />
    </article>
  </div>
</template>

<style scoped>
.ds-shell {
  width: min(1240px, 100%);
  margin: 0 auto;
  padding: 28px 24px 88px;
  display: grid;
  grid-template-columns: 270px minmax(0, 820px);
  justify-content: space-between;
  gap: 64px;
}

.ds-sidebar__sticky {
  position: sticky;
  top: 108px;
  max-height: calc(100vh - 132px);
  overflow: auto;
  padding-right: 12px;
  scrollbar-width: thin;
}

.ds-sidebar__home {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #10221f;
  text-decoration: none;
}

.ds-sidebar__mark {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 50%;
  color: #f7f9f5;
  background: #143b37;
  font: 800 12px/1 ui-monospace, monospace;
  letter-spacing: .08em;
}

.ds-sidebar__home strong,
.ds-sidebar__home small {
  display: block;
}

.ds-sidebar__home strong { font-size: 16px; }
.ds-sidebar__home small { margin-top: 4px; color: #65736f; font-size: 11px; }

.ds-sidebar__progress {
  margin: 20px 0 24px;
  padding: 14px;
  border: 1px solid #d6e1db;
  border-radius: 12px;
  background: #fff;
}

.ds-sidebar__progress > div {
  display: flex;
  justify-content: space-between;
  color: #5b6965;
  font-size: 12px;
}

.ds-sidebar__progress strong { color: #143b37; }
.ds-sidebar__track { display: block; height: 6px; margin-top: 10px; overflow: hidden; border-radius: 999px; background: #e7ede9; }
.ds-sidebar__track > span { display: block; height: 100%; border-radius: inherit; background: #e1a83b; transition: width .25s ease; }

.ds-sidebar__module { margin-top: 22px; }
.ds-sidebar__module h2 { margin: 0 0 7px; color: #78837f; font-size: 11px; font-weight: 800; letter-spacing: .12em; }
.ds-sidebar__module ul { margin: 0; padding: 0; list-style: none; }
.ds-sidebar__module li { margin: 2px 0; }
.ds-sidebar__module a,
.ds-sidebar__module .is-planned {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr);
  align-items: start;
  gap: 8px;
  min-height: 36px;
  padding: 7px 8px;
  border-radius: 8px;
  color: #44524e;
  font-size: 12px;
  line-height: 1.45;
  text-decoration: none;
}

.ds-sidebar__module a:hover { background: #edf2ee; color: #10221f; }
.ds-sidebar__module a.is-active { background: #143b37; color: #fff; }
.ds-sidebar__module .is-planned { color: #9ca5a1; }
.ds-sidebar__number { font: 750 10px/1.6 ui-monospace, monospace; letter-spacing: .03em; }
.ds-sidebar__number.is-done { color: #9a6910; }
.ds-sidebar__module a.is-active .ds-sidebar__number { color: #f3c56f; }

.ds-course-content { min-width: 0; padding-top: 20px; }
.ds-mobile-index { display: none; }

@media (max-width: 960px) {
  .ds-shell { display: block; padding: 18px 20px 72px; }
  .ds-sidebar { display: none; }
  .ds-mobile-index { display: block; margin: 0 0 24px; border: 1px solid #d6e1db; border-radius: 12px; background: #fff; }
  .ds-mobile-index summary { min-height: 48px; padding: 14px 16px; color: #143b37; font-size: 13px; font-weight: 800; cursor: pointer; }
  .ds-mobile-index nav { display: grid; gap: 4px; padding: 0 16px 16px; }
  .ds-mobile-index strong { margin-top: 12px; color: #7b8783; font-size: 11px; }
  .ds-mobile-index a { padding: 7px 0; color: #263d38; font-size: 13px; text-decoration: none; }
  .ds-course-content { padding-top: 0; }
}
</style>
