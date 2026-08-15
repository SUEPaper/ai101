<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRoute, withBase } from 'vitepress'
import { dlLessons, dlModules, normalizeDlPath } from '../../data/dlCourse'

const route = useRoute()
const completed = ref<string[]>([])
const storageKey = 'ai101-dl-completed'

const currentPath = computed(() => normalizeDlPath(route.path))
const progress = computed(() => Math.round((completed.value.filter((item) => /^\d{2}$/.test(item)).length / dlLessons.length) * 100))

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
  window.addEventListener('dl-progress-updated', readProgress)
})

onUnmounted(() => window.removeEventListener('dl-progress-updated', readProgress))
</script>

<template>
  <div class="dl-shell">
    <aside class="dl-sidebar" aria-label="深度学习与大模型课程目录">
      <div class="dl-sidebar__sticky">
        <a class="dl-sidebar__home" :href="withBase('/courses/deep-learning')">
          <span class="dl-sidebar__mark" aria-hidden="true"><i /><i /><i /></span>
          <span><strong>深度学习与大模型</strong><small>追踪信号、预算与系统边界</small></span>
        </a>

        <div class="dl-sidebar__progress" aria-label="32 个正式章节学习进度">
          <div><span>全课程进度</span><strong>{{ progress }}%</strong></div>
          <span class="dl-sidebar__track"><span :style="{ width: `${progress}%` }" /></span>
          <small>{{ completed.filter((item) => /^\d{2}$/.test(item)).length }} / 32 章完成</small>
        </div>

        <nav class="dl-sidebar__nav">
          <section v-for="module in dlModules" :key="module.title" class="dl-sidebar__module">
            <h2>{{ module.title }}</h2>
            <ul>
              <li v-for="item in module.lessons" :key="item.number">
                <a
                  v-if="item.status === 'available'"
                  :class="{ 'is-active': currentPath === normalizeDlPath(item.href) }"
                  :href="withBase(item.href)"
                  :aria-current="currentPath === normalizeDlPath(item.href) ? 'page' : undefined"
                >
                  <span class="dl-sidebar__number" :class="{ 'is-done': completed.includes(item.number) }">{{ completed.includes(item.number) ? '✓' : item.number }}</span>
                  <span>{{ item.title }}</span>
                </a>
                <span v-else class="dl-sidebar__planned" :aria-label="`${item.title}，建设中`">
                  <span class="dl-sidebar__number">{{ item.number }}</span>
                  <span>{{ item.title }}</span>
                </span>
              </li>
            </ul>
          </section>
          <section class="dl-sidebar__module">
            <h2>实践与交付</h2>
            <ul>
              <li>
                <a :class="{ 'is-active': currentPath === '/courses/deep-learning/assignments' }" :href="withBase('/courses/deep-learning/assignments')">
                  <span class="dl-sidebar__number">A</span><span>八次综合作业</span>
                </a>
              </li>
              <li>
                <a :class="{ 'is-active': currentPath === '/courses/deep-learning/project' }" :href="withBase('/courses/deep-learning/project')">
                  <span class="dl-sidebar__number">P</span><span>结课项目</span>
                </a>
              </li>
            </ul>
          </section>
        </nav>
      </div>
    </aside>

    <details class="dl-mobile-index">
      <summary>课程目录与学习进度 · {{ progress }}%</summary>
      <nav aria-label="深度学习与大模型移动端课程目录">
        <template v-for="module in dlModules" :key="module.title">
          <strong>{{ module.title }}</strong>
          <a v-for="item in module.lessons.filter((lesson) => lesson.status === 'available')" :key="item.number" :href="withBase(item.href)">
            {{ item.number }} · {{ item.title }}
          </a>
          <span v-for="item in module.lessons.filter((lesson) => lesson.status === 'planned')" :key="item.number" class="is-planned">
            {{ item.number }} · {{ item.title }} · 建设中
          </span>
        </template>
        <strong>实践与交付</strong>
        <a :href="withBase('/courses/deep-learning/assignments')">A · 八次综合作业</a>
        <a :href="withBase('/courses/deep-learning/project')">P · 结课项目</a>
      </nav>
    </details>

    <article class="dl-course-content vp-doc">
      <slot />
    </article>
  </div>
</template>

<style scoped>
.dl-shell{width:min(1360px,100%);min-height:calc(100dvh - 80px);margin:0 auto;padding:28px 24px 96px;display:grid;grid-template-columns:300px minmax(0,880px);justify-content:space-between;gap:72px;background:#f2f6f6}.dl-sidebar__sticky{position:sticky;top:108px;max-height:calc(100vh - 132px);overflow:auto;padding-right:14px;scrollbar-width:thin}.dl-sidebar__home{display:flex;align-items:center;gap:12px;color:#13213c;text-decoration:none}.dl-sidebar__mark{position:relative;display:grid;place-items:center;width:46px;height:46px;overflow:hidden;border:1px solid #2f6c71;border-radius:7px;background:#13213c;box-shadow:5px 5px 0 #acd9d5}.dl-sidebar__mark i{position:absolute;left:9px;width:28px;height:2px;background:#58d3c8}.dl-sidebar__mark i:nth-child(1){top:13px}.dl-sidebar__mark i:nth-child(2){top:22px;width:18px;background:#f2c14e}.dl-sidebar__mark i:nth-child(3){top:31px;width:24px;background:#df6c5b}.dl-sidebar__home strong,.dl-sidebar__home small{display:block}.dl-sidebar__home strong{font-size:15px}.dl-sidebar__home small{margin-top:4px;color:#5c6f7d;font-size:10px}.dl-sidebar__progress{margin:22px 0 26px;padding:14px;border:1px solid #cfddde;border-radius:8px;background:#fff}.dl-sidebar__progress>div{display:flex;justify-content:space-between;color:#60727c;font-size:12px}.dl-sidebar__progress strong{color:#176b69}.dl-sidebar__progress small{display:block;margin-top:8px;color:#77868e;font-size:10px}.dl-sidebar__track{display:block;height:6px;margin-top:10px;overflow:hidden;border-radius:2px;background:#e2ecec}.dl-sidebar__track>span{display:block;height:100%;background:linear-gradient(90deg,#58d3c8,#f2c14e);transition:width .25s ease}.dl-sidebar__module{margin-top:23px}.dl-sidebar__module h2{margin:0 0 7px;color:#6e7e86;font-size:10px;font-weight:850;letter-spacing:.1em}.dl-sidebar__module ul{margin:0;padding:0;list-style:none}.dl-sidebar__module li{margin:2px 0}.dl-sidebar__module a,.dl-sidebar__planned{display:grid;grid-template-columns:38px minmax(0,1fr);align-items:start;gap:7px;min-height:36px;padding:7px 8px;border-radius:6px;font-size:12px;line-height:1.45;text-decoration:none}.dl-sidebar__module a{color:#405561}.dl-sidebar__module a:hover{color:#13213c;background:#e2eeee}.dl-sidebar__module a.is-active{color:#fff;background:#1b5660}.dl-sidebar__planned{color:#93a0a5}.dl-sidebar__number{color:#3b817e;font:750 9px/1.7 ui-monospace,monospace}.dl-sidebar__number.is-done{color:#c0871a}.dl-sidebar__module a.is-active .dl-sidebar__number{color:#7de0d5}.dl-course-content{min-width:0;padding-top:20px}.dl-course-content:deep(.dl-stop){margin:28px 0;padding:18px 20px;border-left:4px solid #f2c14e;color:#314b57;background:#fff8dc;font-size:14px;line-height:1.75}.dl-course-content:deep(.dl-stop strong){display:block;margin-bottom:5px;color:#79590e}.dl-course-content:deep(.dl-figure){margin:30px 0;padding:10px;border:1px solid #c8d9da;border-radius:10px;background:#fff;box-shadow:7px 7px 0 #dce8e8}.dl-course-content:deep(.dl-figure img){display:block;width:100%;border-radius:6px}.dl-course-content:deep(.dl-figure figcaption){padding:12px 8px 4px;color:#5b6e78;font-size:12px;line-height:1.7}.dl-course-content:deep(.dl-reading-path){margin:28px 0;padding:18px 20px;border:1px solid #b9d7d4;border-radius:9px;background:#edf8f7}.dl-mobile-index{display:none}@media(max-width:1020px){.dl-shell{display:block;padding:18px 20px 76px}.dl-sidebar{display:none}.dl-mobile-index{display:block;margin:0 0 24px;border:1px solid #cbdadb;border-radius:9px;background:#fff}.dl-mobile-index summary{min-height:48px;padding:14px 16px;color:#175f65;font-size:13px;font-weight:800;cursor:pointer}.dl-mobile-index nav{display:grid;gap:4px;padding:0 16px 18px}.dl-mobile-index strong{margin-top:13px;color:#64767f;font-size:11px}.dl-mobile-index a,.dl-mobile-index span{padding:7px 0;font-size:13px;text-decoration:none}.dl-mobile-index a{color:#284e5b}.dl-mobile-index .is-planned{color:#9aa5a9}.dl-course-content{padding-top:0}}@media(prefers-reduced-motion:reduce){.dl-sidebar__track>span{transition:none}}
</style>
