<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, withBase } from 'vitepress'
import { mlAllModules, mlCoreLessons, normalizeMlPath } from '../../data/mlCourse'

const route = useRoute()
const completed = ref<string[]>([])
const storageKey = 'ai101-ml-completed'

const currentPath = computed(() => normalizeMlPath(route.path))
const progress = computed(() => {
  const count = mlCoreLessons.filter((lesson) => completed.value.includes(lesson.number)).length
  return Math.round((count / mlCoreLessons.length) * 100)
})

function readProgress() {
  if (typeof window === 'undefined') return
  try {
    const value = JSON.parse(window.localStorage.getItem(storageKey) ?? '[]')
    completed.value = Array.isArray(value) ? value.map(String) : []
  } catch {
    completed.value = []
  }
}

onMounted(() => {
  readProgress()
  window.addEventListener('ml-progress-updated', readProgress)
})

onUnmounted(() => window.removeEventListener('ml-progress-updated', readProgress))
</script>

<template>
  <div class="ml-shell">
    <aside class="ml-sidebar" aria-label="机器学习课程目录">
      <div class="ml-sidebar__sticky">
        <a class="ml-sidebar__home" :href="withBase('/courses/machine-learning')">
          <span class="ml-sidebar__mark" aria-hidden="true">∇</span>
          <span><strong>机器学习</strong><small>从目标函数走到算法</small></span>
        </a>

        <div class="ml-sidebar__progress" aria-label="32 个正式章节学习进度">
          <div><span>核心进度</span><strong>{{ progress }}%</strong></div>
          <span class="ml-sidebar__track"><span :style="{ width: `${progress}%` }" /></span>
          <small>{{ completed.filter((item) => /^\d+$/.test(item)).length }} / 32 章</small>
        </div>

        <nav class="ml-sidebar__nav">
          <section v-for="module in mlAllModules" :key="module.title" class="ml-sidebar__module">
            <h2>{{ module.title }}</h2>
            <ul>
              <li v-for="lesson in module.lessons" :key="lesson.number">
                <a
                  :class="{ 'is-active': currentPath === normalizeMlPath(lesson.href) }"
                  :href="withBase(lesson.href)"
                  :aria-current="currentPath === normalizeMlPath(lesson.href) ? 'page' : undefined"
                >
                  <span class="ml-sidebar__number" :class="{ 'is-done': completed.includes(lesson.number) }">
                    {{ completed.includes(lesson.number) ? '✓' : lesson.number }}
                  </span>
                  <span>{{ lesson.title }}</span>
                </a>
              </li>
            </ul>
          </section>
        </nav>
      </div>
    </aside>

    <details class="ml-mobile-index">
      <summary>课程目录与核心进度 · {{ progress }}%</summary>
      <nav aria-label="机器学习移动端课程目录">
        <template v-for="module in mlAllModules" :key="module.title">
          <strong>{{ module.title }}</strong>
          <a v-for="lesson in module.lessons" :key="lesson.number" :href="withBase(lesson.href)">
            {{ lesson.number }} · {{ lesson.title }}
          </a>
        </template>
      </nav>
    </details>

    <article class="ml-course-content vp-doc">
      <slot />
    </article>
  </div>
</template>

<style scoped>
.ml-shell{width:min(1320px,100%);min-height:calc(100dvh - 80px);margin:0 auto;padding:28px 24px 96px;display:grid;grid-template-columns:292px minmax(0,850px);justify-content:space-between;gap:68px;background:#f5f7fb}.ml-sidebar__sticky{position:sticky;top:108px;max-height:calc(100vh - 132px);overflow:auto;padding-right:13px;scrollbar-width:thin}.ml-sidebar__home{display:flex;align-items:center;gap:12px;color:#17203b;text-decoration:none}.ml-sidebar__mark{display:grid;place-items:center;width:44px;height:44px;border:1px solid #2e4fb5;border-radius:10px;color:#f8faff;background:#263d88;font:800 22px/1 Georgia,serif;box-shadow:5px 5px 0 #b9c7ed}.ml-sidebar__home strong,.ml-sidebar__home small{display:block}.ml-sidebar__home strong{font-size:16px}.ml-sidebar__home small{margin-top:4px;color:#63708d;font-size:11px}.ml-sidebar__progress{margin:22px 0 26px;padding:14px;border:1px solid #d8deec;border-radius:10px;background:#fff}.ml-sidebar__progress>div{display:flex;justify-content:space-between;color:#66718a;font-size:12px}.ml-sidebar__progress strong{color:#263d88}.ml-sidebar__progress small{display:block;margin-top:8px;color:#7d879c;font-size:10px}.ml-sidebar__track{display:block;height:6px;margin-top:10px;overflow:hidden;border-radius:2px;background:#e8ebf3}.ml-sidebar__track>span{display:block;height:100%;background:#e15b4b;transition:width .25s ease}.ml-sidebar__module{margin-top:23px}.ml-sidebar__module h2{margin:0 0 7px;color:#76809a;font-size:10px;font-weight:850;letter-spacing:.1em}.ml-sidebar__module ul{margin:0;padding:0;list-style:none}.ml-sidebar__module li{margin:2px 0}.ml-sidebar__module a{display:grid;grid-template-columns:38px minmax(0,1fr);align-items:start;gap:7px;min-height:36px;padding:7px 8px;border-radius:7px;color:#495570;font-size:12px;line-height:1.45;text-decoration:none}.ml-sidebar__module a:hover{color:#17203b;background:#eaedf7}.ml-sidebar__module a.is-active{color:#fff;background:#263d88}.ml-sidebar__number{color:#7080ad;font:750 9px/1.7 ui-monospace,monospace}.ml-sidebar__number.is-done{color:#15867f}.ml-sidebar__module a.is-active .ml-sidebar__number{color:#83ded5}.ml-course-content{min-width:0;padding-top:20px}.ml-mobile-index{display:none}@media(max-width:1000px){.ml-shell{display:block;padding:18px 20px 76px}.ml-sidebar{display:none}.ml-mobile-index{display:block;margin:0 0 24px;border:1px solid #d8deec;border-radius:10px;background:#fff}.ml-mobile-index summary{min-height:48px;padding:14px 16px;color:#263d88;font-size:13px;font-weight:800;cursor:pointer}.ml-mobile-index nav{display:grid;gap:4px;padding:0 16px 18px}.ml-mobile-index strong{margin-top:13px;color:#78829b;font-size:11px}.ml-mobile-index a{padding:7px 0;color:#35415e;font-size:13px;text-decoration:none}.ml-course-content{padding-top:0}}
</style>
