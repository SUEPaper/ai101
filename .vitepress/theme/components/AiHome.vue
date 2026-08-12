<script setup lang="ts">
import { withBase } from 'vitepress'

const courses = [
  {
    number: '01',
    phase: '数据与证据',
    title: '数据科学',
    status: '已开放',
    available: true,
    href: '/courses/data-science',
    routeDescription: '从问题、数据质量和不确定性出发，建立可信的数据判断。',
    summary: '沿真实数据工作流学习清洗、探索、统计推断、SQL、分类与无监督学习，把计算结果写成可以审查的证据。',
    highlights: ['P0 + 26 讲', '7 次模块作业', '1 个结课项目'],
    outcome: '完成一次从问题定义、数据检查到结论表达的数据分析闭环。',
    action: '进入数据科学课程',
    accentClass: 'text-[#2f7c7a]',
    borderClass: 'border-l-[#2f7c7a]',
    dotClass: 'bg-[#2f7c7a]'
  },
  {
    number: '02',
    phase: '建模与推导',
    title: '机器学习',
    status: '已开放',
    available: true,
    href: '/courses/machine-learning',
    routeDescription: '从假设和目标函数推导算法，再用数值检查验证它。',
    summary: '不把算法当成 API 清单。从学习问题、回归和分类一路走到神经网络、推荐系统与强化学习，逐章推导并验证。',
    highlights: ['32 个核心章', '6 个数学选修章', '8 次综合作业 + 项目'],
    outcome: '能够解释模型为何成立、何时失效，以及怎样用实验检查推导。',
    action: '进入机器学习教材',
    accentClass: 'text-[#a66d20]',
    borderClass: 'border-l-[#b0792f]',
    dotClass: 'bg-[#b0792f]'
  },
  {
    number: '03',
    phase: '表示、生成与规模化',
    title: '深度学习与大模型',
    status: '规划中',
    available: false,
    href: '/courses/deep-learning',
    routeDescription: '在神经网络基础上进入现代架构、生成模型与 LLM。',
    summary: '计划系统展开现代架构变体、大规模训练、自监督学习、生成模型和大语言模型；具体大纲、课时与章节仍待确认。',
    highlights: ['现代深度架构', '自监督与生成模型', '大语言模型'],
    outcome: '计划形成从训练机制到大模型应用边界的系统理解。',
    action: '查看当前课程规划',
    accentClass: 'text-[#93483f]',
    borderClass: 'border-l-[#9b4d42]',
    dotClass: 'bg-[#9b4d42]'
  },
  {
    number: '04',
    phase: '工程与交付',
    title: '人工智能实践课',
    status: '规划中',
    available: false,
    href: '/courses/machine-learning-practice',
    routeDescription: '把分析、训练和评估接入可复现、可运行的工程流程。',
    summary: '计划覆盖问题定义、数据与模型版本、实验追踪、服务化、监控和 MLOps，把前序课程能力收束为完整交付。',
    highlights: ['实验与版本管理', '服务化与部署', '监控与 MLOps'],
    outcome: '计划交付一个可运行、可复现、可审查的人工智能系统。',
    action: '查看当前课程规划',
    accentClass: 'text-[#466b8d]',
    borderClass: 'border-l-[#4d6f91]',
    dotClass: 'bg-[#4d6f91]'
  }
]

const openCourses = courses.filter((course) => course.available)
const plannedCourses = courses.filter((course) => !course.available)
</script>

<template>
  <main class="overflow-hidden bg-[#f7f9f5] text-[#17201f]">
    <section class="relative overflow-hidden border-b border-[#dbe5df]" aria-labelledby="ai-home-title">
      <img
        class="absolute inset-y-0 right-0 h-full w-full object-cover object-center opacity-[0.12] lg:w-[64%] lg:opacity-35"
        :src="withBase('/images/ai-course-hero.png')"
        alt=""
      >
      <div class="absolute inset-0 bg-[linear-gradient(90deg,#f7f9f5_0%,#f7f9f5_54%,rgba(247,249,245,0.82)_74%,rgba(247,249,245,0.34)_100%)]" aria-hidden="true" />

      <div class="relative mx-auto grid w-full max-w-[1180px] gap-12 px-5 py-14 sm:px-6 md:py-20 lg:grid-cols-[0.9fr_1.1fr] lg:items-center lg:gap-16 lg:py-24">
        <div>
          <p class="mb-5 inline-flex rounded-full border border-[#cbd9d2] bg-white/80 px-3 py-1 text-xs font-bold text-[#2f7c7a]">
            2 门已开放 · 2 门规划中
          </p>
          <h1 id="ai-home-title" class="m-0 max-w-[650px] text-[44px] font-[840] leading-[1.04] tracking-[-0.035em] text-[#0d1514] md:text-[64px]">
            从可信数据，<br>
            走到可交付的人工智能系统
          </h1>
          <p class="mt-6 max-w-[610px] text-base leading-[1.85] text-[#52625e] md:text-lg">
            四门课组成一条连续能力链：先建立数据判断，再理解模型为什么成立；随后进入深度学习与大模型，最终在人工智能实践课完成工程交付。
          </p>

          <div class="mt-8 flex flex-wrap gap-3" aria-label="已开放课程入口">
            <a
              class="inline-flex min-h-11 w-full items-center justify-center rounded-lg bg-[#143b37] px-5 text-[15px] font-bold text-white no-underline transition duration-200 hover:-translate-y-0.5 hover:bg-[#1d504a] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#2f7c7a] motion-reduce:transform-none motion-reduce:transition-none sm:w-auto"
              :href="withBase('/courses/data-science')"
            >
              开始数据科学
            </a>
            <a
              class="inline-flex min-h-11 w-full items-center justify-center rounded-lg border border-[#b9c9c1] bg-white/90 px-5 text-[15px] font-bold text-[#17201f] no-underline transition duration-200 hover:-translate-y-0.5 hover:border-[#7f9d90] hover:bg-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#2f7c7a] motion-reduce:transform-none motion-reduce:transition-none sm:w-auto"
              :href="withBase('/courses/machine-learning')"
            >
              进入机器学习教材
            </a>
          </div>
        </div>

        <aside class="rounded-xl border border-[#cad8d1] bg-white/90 p-5 shadow-[0_28px_90px_rgba(13,21,20,0.14)] backdrop-blur md:p-7" aria-labelledby="learning-route-title">
          <div class="flex items-end justify-between gap-4 border-b border-[#dbe5df] pb-5">
            <div>
              <p class="m-0 font-mono text-[11px] font-bold tracking-[0.18em] text-[#6a7d77]">LEARNING ROUTE</p>
              <h2 id="learning-route-title" class="mb-0 mt-2 text-2xl font-[820] text-[#0d1514]">学习路线总线</h2>
            </div>
            <span class="shrink-0 text-xs font-semibold text-[#687773]">按能力递进</span>
          </div>

          <ol class="m-0 mt-6 list-none p-0">
            <li
              v-for="(course, index) in courses"
              :key="course.href"
              class="relative grid grid-cols-[32px_minmax(0,1fr)] gap-4 pb-5 last:pb-0"
            >
              <span
                v-if="index < courses.length - 1"
                class="absolute left-[15px] top-8 h-[calc(100%-12px)] w-px bg-[#cbd9d2]"
                aria-hidden="true"
              />
              <span
                class="relative z-10 grid h-8 w-8 place-items-center rounded-full border-4 border-white text-[10px] font-[850] text-white shadow-[0_0_0_1px_#cbd9d2]"
                :class="course.dotClass"
                aria-hidden="true"
              >
                {{ course.number }}
              </span>
              <a
                class="group rounded-lg border border-[#dbe5df] bg-[#fbfcf9] px-4 py-3 text-[#17201f] no-underline transition duration-200 hover:-translate-y-0.5 hover:border-[#9fb5ab] hover:bg-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#2f7c7a] motion-reduce:transform-none motion-reduce:transition-none"
                :href="withBase(course.href)"
              >
                <span class="flex flex-wrap items-center justify-between gap-2">
                  <span class="font-mono text-[10px] font-bold tracking-[0.12em]" :class="course.accentClass">{{ course.phase }}</span>
                  <span
                    class="rounded-full border px-2 py-0.5 text-[10px] font-bold"
                    :class="course.available ? 'border-[#b9d6ca] bg-[#edf7f2] text-[#256354]' : 'border-[#d7d9d6] bg-[#f1f2ef] text-[#66706c]'"
                  >
                    {{ course.status }}
                  </span>
                </span>
                <strong class="mt-2 block text-lg leading-[1.25] group-hover:text-[#143b37]">{{ course.title }}</strong>
                <span class="mt-1 block text-[13px] leading-[1.6] text-[#66736f]">{{ course.routeDescription }}</span>
              </a>
            </li>
          </ol>
        </aside>
      </div>
    </section>

    <section class="border-b border-[#dbe5df] bg-white/55 py-16 md:py-20" aria-labelledby="open-courses-title">
      <div class="mx-auto w-full max-w-[1180px] px-5 sm:px-6">
        <div class="max-w-[760px]">
          <p class="m-0 font-mono text-[11px] font-bold tracking-[0.16em] text-[#2f7c7a]">OPEN FOR STUDY</p>
          <h2 id="open-courses-title" class="mb-0 mt-3 text-[32px] font-[830] leading-[1.12] tracking-[-0.02em] text-[#0d1514] md:text-[44px]">现在可以完整学习</h2>
          <p class="mb-0 mt-4 text-base leading-[1.8] text-[#5b6864]">两套课程已完成正文、实验、练习和项目建设，当前进入试学反馈驱动的维护阶段。</p>
        </div>

        <div class="mt-10 grid gap-5 lg:grid-cols-2">
          <a
            v-for="course in openCourses"
            :key="course.href"
            class="group flex flex-col rounded-xl border border-[#dbe5df] border-l-4 bg-[#fbfcf9] p-6 text-[#17201f] no-underline shadow-[0_16px_45px_rgba(13,21,20,0.06)] transition duration-200 hover:-translate-y-1 hover:border-[#9fb5ab] hover:bg-white hover:shadow-[0_20px_55px_rgba(13,21,20,0.1)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#2f7c7a] motion-reduce:transform-none motion-reduce:transition-none md:p-8"
            :class="course.borderClass"
            :href="withBase(course.href)"
          >
            <span class="flex items-center justify-between gap-4">
              <span class="font-mono text-xs font-bold tracking-[0.14em]" :class="course.accentClass">COURSE {{ course.number }}</span>
              <span class="rounded-full border border-[#b9d6ca] bg-[#edf7f2] px-2.5 py-1 text-[11px] font-bold text-[#256354]">已开放</span>
            </span>
            <h3 class="mb-0 mt-5 text-[28px] font-[830] leading-tight text-[#0d1514]">{{ course.title }}</h3>
            <p class="mb-0 mt-4 text-[15px] leading-[1.8] text-[#5b6864]">{{ course.summary }}</p>
            <ul class="m-0 mt-6 flex list-none flex-wrap gap-2 p-0" :aria-label="`${course.title}课程规模`">
              <li v-for="highlight in course.highlights" :key="highlight" class="rounded-md border border-[#dbe5df] bg-white px-3 py-1.5 text-xs font-bold text-[#40504b]">
                {{ highlight }}
              </li>
            </ul>
            <p class="mb-0 mt-7 border-t border-[#dbe5df] pt-5 text-sm leading-[1.75] text-[#52625e]">
              <strong class="text-[#17201f]">学习产出：</strong>{{ course.outcome }}
            </p>
            <span class="mt-6 inline-flex min-h-11 items-center justify-between gap-4 self-stretch rounded-lg bg-[#143b37] px-4 text-sm font-bold text-white">
              {{ course.action }}
              <span aria-hidden="true">→</span>
            </span>
          </a>
        </div>
      </div>
    </section>

    <section class="py-16 md:py-20" aria-labelledby="planned-courses-title">
      <div class="mx-auto w-full max-w-[1180px] px-5 sm:px-6">
        <div class="grid gap-6 lg:grid-cols-[0.72fr_1.28fr] lg:items-end">
          <div>
            <p class="m-0 font-mono text-[11px] font-bold tracking-[0.16em] text-[#687773]">NEXT ON THE ROUTE</p>
            <h2 id="planned-courses-title" class="mb-0 mt-3 text-[32px] font-[830] leading-[1.12] tracking-[-0.02em] text-[#0d1514] md:text-[44px]">下一阶段建设</h2>
          </div>
          <p class="m-0 max-w-[700px] text-base leading-[1.8] text-[#5b6864] lg:justify-self-end">
            两门后续课的能力边界已经明确，但具体大纲、课时和每节内容仍待确认。这里展示的是建设方向，不把规划误标成已完成课程。
          </p>
        </div>

        <div class="mt-10 grid gap-5 md:grid-cols-2">
          <a
            v-for="course in plannedCourses"
            :key="course.href"
            class="group rounded-xl border border-[#d8dfdb] border-l-4 bg-[#f1f3ef] p-6 text-[#17201f] no-underline transition duration-200 hover:-translate-y-0.5 hover:border-[#aab8b1] hover:bg-[#f6f8f4] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-[#2f7c7a] motion-reduce:transform-none motion-reduce:transition-none md:p-7"
            :class="course.borderClass"
            :href="withBase(course.href)"
          >
            <span class="flex flex-wrap items-center justify-between gap-3">
              <span class="font-mono text-xs font-bold tracking-[0.12em]" :class="course.accentClass">COURSE {{ course.number }}</span>
              <span class="rounded-full border border-[#d7d9d6] bg-white/75 px-2.5 py-1 text-[11px] font-bold text-[#66706c]">规划中</span>
            </span>
            <h3 class="mb-0 mt-5 text-2xl font-[820] leading-tight text-[#17201f]">{{ course.title }}</h3>
            <p class="mb-0 mt-4 text-[15px] leading-[1.8] text-[#616d69]">{{ course.summary }}</p>
            <ul class="m-0 mt-6 flex list-none flex-wrap gap-2 p-0" :aria-label="`${course.title}计划内容`">
              <li v-for="highlight in course.highlights" :key="highlight" class="rounded-md border border-[#d5ddda] bg-white/70 px-3 py-1.5 text-xs font-semibold text-[#56635f]">
                {{ highlight }}
              </li>
            </ul>
            <span class="mt-6 inline-flex min-h-11 items-center gap-3 text-sm font-bold text-[#33443f] group-hover:text-[#143b37]">
              {{ course.action }}
              <span aria-hidden="true">→</span>
            </span>
          </a>
        </div>
      </div>
    </section>
  </main>
</template>
