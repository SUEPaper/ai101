<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, withBase } from 'vitepress'

const route = useRoute()
const mobileOpen = ref(false)

const mainCourses = [
  { title: '数据科学', href: '/courses/data-science' },
  { title: '机器学习', href: '/courses/machine-learning' },
  { title: '深度学习', href: '/courses/deep-learning' },
  { title: '机器学习实践课', href: '/courses/machine-learning-practice' }
]

const allCourses = [
  { title: '首页', href: '/' },
  ...mainCourses
]

const activePath = computed(() => route.path.replace(/\.html$/, ''))

function isActive(href: string) {
  if (href === '/') return activePath.value === '/' || activePath.value === ''
  return activePath.value.startsWith(href)
}

watch(
  () => route.path,
  () => {
    mobileOpen.value = false
  }
)
</script>

<template>
  <header class="sticky top-0 z-50 border-b border-[#dbe5df] bg-[#f7f9f5]/95 backdrop-blur-xl">
    <nav class="mx-auto flex min-h-[80px] w-full max-w-[1180px] items-center gap-6 px-4 sm:px-6" aria-label="课程导航">
      <a
        class="group flex min-h-11 shrink-0 items-center gap-3 text-[#0d1514] no-underline focus-visible:rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#2f7c7a]"
        :href="withBase('/')"
      >
        <span class="grid h-11 w-11 place-items-center rounded-lg bg-[#0d1514] text-[13px] font-[820] text-[#f3c56f] transition group-hover:bg-[#193431]">
          智
        </span>
        <span class="leading-none">
          <span class="block text-base font-[820]">人工智能 101</span>
          <span class="mt-1 hidden text-[11px] font-semibold text-[#687773] sm:block">人工智能课程系列</span>
        </span>
      </a>

      <div class="hidden flex-1 items-stretch justify-end lg:flex">
        <div class="flex items-stretch gap-1">
          <a
            v-for="course in mainCourses"
            :key="course.href"
            class="group/nav relative flex min-h-[64px] w-[140px] flex-col justify-center rounded-lg px-3 text-center text-[#32413d] no-underline transition hover:bg-white/80 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2f7c7a]"
            :class="isActive(course.href) ? 'text-[#0d1514]' : ''"
            :href="withBase(course.href)"
            :aria-current="isActive(course.href) ? 'page' : undefined"
          >
            <span class="block text-[15px] font-bold leading-none">{{ course.title }}</span>
            <span
              class="absolute inset-x-3 bottom-2 h-0.5 rounded-full transition"
              :class="isActive(course.href) ? 'bg-[#143b37]' : 'bg-transparent group-hover/nav:bg-[#dbe5df]'"
              aria-hidden="true"
            />
          </a>
        </div>
      </div>

      <button
        class="ml-auto inline-flex min-h-11 items-center gap-3 rounded-lg border border-[#dbe5df] bg-white px-4 text-sm font-bold text-[#17201f] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2f7c7a] lg:hidden"
        type="button"
        :aria-expanded="mobileOpen"
        aria-controls="course-mobile-nav"
        @click="mobileOpen = !mobileOpen"
      >
        课程
        <span class="grid gap-1" aria-hidden="true">
          <span class="block h-0.5 w-4 rounded bg-[#17201f]" />
          <span class="block h-0.5 w-4 rounded bg-[#17201f]" />
        </span>
      </button>
    </nav>

    <div v-if="mobileOpen" id="course-mobile-nav" class="border-t border-[#dbe5df] bg-[#f7f9f5] px-4 pb-4 lg:hidden">
      <div class="mx-auto grid w-full max-w-[1180px] gap-2 pt-3">
        <a
          v-for="course in allCourses"
          :key="course.href"
          class="grid min-h-[58px] grid-cols-[minmax(0,1fr)] rounded-lg border border-[#dbe5df] bg-white px-4 py-3 text-[#17201f] no-underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2f7c7a]"
          :class="isActive(course.href) ? 'border-[#143b37] bg-[#143b37] text-white' : ''"
          :href="withBase(course.href)"
          :aria-current="isActive(course.href) ? 'page' : undefined"
        >
          <span class="text-sm font-bold leading-none">{{ course.title }}</span>
        </a>
      </div>
    </div>
  </header>
</template>
