<script setup lang="ts">
import { withBase } from 'vitepress'

const courses = [
  {
    stage: '第一门课',
    title: '数据科学',
    description: '先学会判断数据是否可信，再进入分析、清洗和可视化表达。',
    href: '/courses/data-science',
    borderClass: 'border-l-[#2f7c7a]',
    textClass: 'text-[#2f7c7a]'
  },
  {
    stage: '第二门课',
    title: '机器学习',
    description: '把问题转成可训练任务，用评估指标和误差分析选择模型。',
    href: '/courses/machine-learning',
    borderClass: 'border-l-[#b0792f]',
    textClass: 'text-[#b0792f]'
  },
  {
    stage: '第三门课',
    title: '深度学习',
    description: '进入神经网络、视觉、文本和预训练模型，理解复杂表示能力。',
    href: '/courses/deep-learning',
    borderClass: 'border-l-[#9b4d42]',
    textClass: 'text-[#9b4d42]'
  },
  {
    stage: '实践课',
    title: '机器学习实践课',
    description: '把数据、模型、实验和报告串成一个可复现、可展示的项目。',
    href: '/courses/machine-learning-practice',
    borderClass: 'border-l-[#4d6f91]',
    textClass: 'text-[#4d6f91]'
  }
]
</script>

<template>
  <main class="min-h-[calc(100dvh-80px)] overflow-hidden bg-[#f7f9f5] text-[#17201f]">
    <section class="relative min-h-[calc(100dvh-80px)]" aria-labelledby="ai-home-title">
      <img
        class="absolute inset-y-0 right-0 hidden h-full w-[58%] object-cover opacity-90 lg:block"
        :src="withBase('/images/ai-course-hero.png')"
        alt=""
      >
      <div class="absolute inset-0 bg-[linear-gradient(90deg,#f7f9f5_0%,#f7f9f5_48%,rgba(247,249,245,0.78)_70%,rgba(247,249,245,0.18)_100%)]" aria-hidden="true" />

      <div class="relative mx-auto grid min-h-[calc(100dvh-80px)] w-full max-w-[1180px] items-center gap-10 px-5 py-10 sm:px-6 md:py-14 lg:grid-cols-[0.92fr_1.08fr] lg:py-16">
        <div>
          <p class="mb-5 inline-flex rounded-full border border-[#cbd9d2] bg-white/70 px-3 py-1 text-xs font-bold uppercase text-[#2f7c7a]">
            人工智能课程体系
          </p>
          <h1 id="ai-home-title" class="m-0 max-w-[720px] text-[42px] font-[820] leading-[1.05] text-[#0d1514] md:text-[66px]">
            一条从数据到项目交付的人工智能学习路线
          </h1>
          <p class="mt-6 max-w-[620px] text-base leading-[1.85] text-[#52625e] md:text-lg">
            四门课不是并列菜单，而是一条能力链：先建立数据判断，再学习经典建模，随后进入深度模型，最后用实践课完成可检查的作品。
          </p>

          <div class="mt-8 flex flex-wrap gap-3" aria-label="首页主要入口">
            <a
              class="inline-flex min-h-11 w-full items-center justify-center rounded-lg bg-[#143b37] px-5 text-[15px] font-bold text-white no-underline transition hover:-translate-y-0.5 hover:bg-[#1d504a] sm:w-auto"
              :href="withBase('/courses/data-science')"
            >
              从第一门课开始
            </a>
            <a
              class="inline-flex min-h-11 w-full items-center justify-center rounded-lg border border-[#cbd9d2] bg-white px-5 text-[15px] font-bold text-[#17201f] no-underline transition hover:-translate-y-0.5 hover:border-[#9fb5ab] sm:w-auto"
              :href="withBase('/courses/machine-learning-practice')"
            >
              查看最终项目课
            </a>
          </div>
        </div>

        <div class="rounded-lg border border-[#dbe5df] bg-white/85 p-4 shadow-[0_24px_80px_rgba(13,21,20,0.12)] backdrop-blur md:p-5">
          <div class="grid gap-3">
            <a
              v-for="course in courses"
              :key="course.href"
              class="group grid min-h-[116px] gap-3 rounded-lg border border-[#dbe5df] border-l-4 bg-[#fbfcf9] p-4 text-[#17201f] no-underline transition hover:-translate-y-0.5 hover:border-[#aabbb3] hover:bg-white md:grid-cols-[86px_minmax(0,1fr)_auto] md:items-center"
              :class="course.borderClass"
              :href="withBase(course.href)"
            >
              <span class="text-sm font-[820]" :class="course.textClass">{{ course.stage }}</span>
              <span>
                <strong class="block text-xl leading-[1.2]">{{ course.title }}</strong>
                <span class="mt-2 block text-[15px] leading-[1.65] text-[#5b6864]">{{ course.description }}</span>
              </span>
              <span class="hidden text-sm font-bold text-[#17201f]/55 transition group-hover:text-[#17201f] md:block" aria-hidden="true">进入 →</span>
            </a>
          </div>
        </div>
      </div>
    </section>
  </main>
</template>
