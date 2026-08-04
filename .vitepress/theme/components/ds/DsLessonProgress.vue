<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps<{
  lessonId: string
  nextHref?: string
  nextLabel?: string
}>()

const done = ref(false)
const storageKey = 'ai101-ds-completed'

function readCompleted() {
  try {
    const values = JSON.parse(window.localStorage.getItem(storageKey) ?? '[]')
    done.value = Array.isArray(values) && values.includes(props.lessonId)
  } catch {
    done.value = false
  }
}

function toggle() {
  const values = new Set<string>()
  try {
    const stored = JSON.parse(window.localStorage.getItem(storageKey) ?? '[]')
    if (Array.isArray(stored)) stored.forEach((item) => values.add(String(item)))
  } catch {
    // Ignore invalid local data and rebuild it from this action.
  }

  if (values.has(props.lessonId)) values.delete(props.lessonId)
  else values.add(props.lessonId)

  window.localStorage.setItem(storageKey, JSON.stringify([...values]))
  readCompleted()
  window.dispatchEvent(new Event('ds-progress-updated'))
}

onMounted(readCompleted)
</script>

<template>
  <footer class="ds-progress-card">
    <div>
      <span>本讲学习记录</span>
      <strong>{{ done ? '已经完成，可以继续前进' : '检查练习后再标记完成' }}</strong>
    </div>
    <div class="ds-progress-card__actions">
      <button type="button" :aria-pressed="done" @click="toggle">
        {{ done ? '撤销完成' : '标记为已完成' }}
      </button>
      <a v-if="nextHref" :href="withBase(nextHref)">{{ nextLabel ?? '下一讲' }} →</a>
    </div>
  </footer>
</template>

<style scoped>
.ds-progress-card { display: flex; align-items: center; justify-content: space-between; gap: 22px; margin-top: 52px; padding: 24px; border: 1px solid #cbdad2; border-radius: 14px; background: #edf3ef; }
.ds-progress-card span { display: block; color: #71807b; font-size: 11px; font-weight: 750; letter-spacing: .08em; }
.ds-progress-card strong { display: block; margin-top: 5px; color: #173a34; font-size: 15px; }
.ds-progress-card__actions { display: flex; flex-wrap: wrap; gap: 8px; }
.ds-progress-card button,
.ds-progress-card a { min-height: 42px; padding: 10px 14px; border-radius: 9px; font-size: 12px; font-weight: 750; cursor: pointer; }
.ds-progress-card button { border: 1px solid #9fb7ac; color: #173a34; background: #fff; }
.ds-progress-card a { display: inline-flex; align-items: center; color: #fff; background: #143b37; text-decoration: none; }
@media (max-width: 650px) { .ds-progress-card { align-items: stretch; flex-direction: column; } }
</style>
