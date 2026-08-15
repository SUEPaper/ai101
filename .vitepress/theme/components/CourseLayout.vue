<script setup lang="ts">
import { nextTick, onMounted, watch } from 'vue'
import { Content, useData, withBase } from 'vitepress'
import CourseNav from './CourseNav.vue'
import DlCourseShell from './dl/DlCourseShell.vue'
import DsCourseShell from './ds/DsCourseShell.vue'
import MlCourseShell from './ml/MlCourseShell.vue'

const { frontmatter, page } = useData()

function repairMathJaxViewBox() {
  if (typeof document === 'undefined') return
  void nextTick(() => {
    document.querySelectorAll<SVGSVGElement>('mjx-container svg[viewbox]').forEach((svg) => {
      const value = svg.getAttribute('viewbox')
      if (!value) return
      svg.setAttribute('viewBox', value)
      svg.removeAttribute('viewbox')
    })
  })
}

onMounted(repairMathJaxViewBox)
watch(() => page.value.relativePath, repairMathJaxViewBox)
</script>

<template>
  <div class="flex min-h-dvh flex-col bg-[#f7f9f5]">
    <a
      class="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[60] focus:rounded-lg focus:bg-white focus:px-4 focus:py-2 focus:text-sm focus:font-bold focus:text-[#143b37] focus:shadow-[0_12px_30px_rgba(13,21,20,0.16)]"
      href="#VPContent"
    >
      跳到正文
    </a>
    <CourseNav />

    <main id="VPContent" class="flex-1">
      <Content v-if="frontmatter.layout === 'home'" />
      <div v-else-if="page.isNotFound" class="mx-auto w-full max-w-[860px] px-5 py-16 sm:px-6">
        <h1 class="m-0 text-[34px] font-[820] leading-[1.12] text-[#0d1514]">页面不存在</h1>
        <p class="mt-4 text-base leading-[1.8] text-[#52625e]">这个地址没有对应的课程内容。</p>
        <a
          class="mt-8 inline-flex min-h-11 items-center rounded-lg bg-[#143b37] px-5 text-sm font-bold text-white no-underline"
          :href="withBase('/')"
        >
          回到首页
        </a>
      </div>
      <DsCourseShell v-else-if="frontmatter.course === 'data-science'">
        <Content />
      </DsCourseShell>
      <MlCourseShell v-else-if="frontmatter.course === 'machine-learning'">
        <Content />
      </MlCourseShell>
      <DlCourseShell v-else-if="frontmatter.course === 'deep-learning'">
        <Content />
      </DlCourseShell>
      <article v-else class="vp-doc mx-auto w-full max-w-[860px] px-5 py-12 sm:px-6 md:py-16">
        <Content />
      </article>
    </main>

    <footer class="border-t border-[#dbe5df] bg-[#f7f9f5]" aria-label="网站信息">
      <div class="mx-auto flex min-h-20 w-full max-w-[1180px] items-center justify-end px-5 py-4 text-right sm:px-6">
        <p class="m-0 flex flex-wrap items-center justify-end gap-x-2 gap-y-1 text-sm font-semibold text-[#52625e]">
          <span>上海电力大学数理学院 · 作者：章海文</span>
          <span aria-hidden="true">·</span>
          <a
            class="rounded-sm text-[#315f55] underline decoration-[#8aa99f] underline-offset-4 hover:text-[#143b37] hover:decoration-[#143b37] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-3 focus-visible:outline-[#2f7c7a]"
            href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans"
            target="_blank"
            rel="noopener noreferrer license"
            aria-label="CC BY-NC-SA 4.0 许可协议（在新窗口打开）"
          >
            CC BY-NC-SA 4.0
          </a>
        </p>
      </div>
    </footer>
  </div>
</template>
