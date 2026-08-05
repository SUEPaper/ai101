<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps<{ lessonId: string; nextHref?: string; nextLabel?: string }>()
const done = ref(false)
const storageKey = 'ai101-ml-completed'

function readCompleted() {
  try {
    const values = JSON.parse(window.localStorage.getItem(storageKey) ?? '[]')
    done.value = Array.isArray(values) && values.includes(props.lessonId)
  } catch { done.value = false }
}

function toggle() {
  const values = new Set<string>()
  try {
    const stored = JSON.parse(window.localStorage.getItem(storageKey) ?? '[]')
    if (Array.isArray(stored)) stored.forEach((item) => values.add(String(item)))
  } catch { /* Invalid local progress is replaced by this action. */ }
  values.has(props.lessonId) ? values.delete(props.lessonId) : values.add(props.lessonId)
  window.localStorage.setItem(storageKey, JSON.stringify([...values]))
  readCompleted()
  window.dispatchEvent(new Event('ml-progress-updated'))
}

onMounted(readCompleted)
</script>

<template>
  <footer class="ml-progress-card">
    <div><span>本章学习记录</span><strong>{{ done ? '推导与练习已完成' : '完成推导、实验和练习后再记录' }}</strong></div>
    <div class="ml-progress-card__actions">
      <button type="button" :aria-pressed="done" @click="toggle">{{ done ? '撤销完成' : '标记为已完成' }}</button>
      <a v-if="nextHref" :href="withBase(nextHref)">{{ nextLabel ?? '下一章' }} →</a>
    </div>
  </footer>
</template>

<style scoped>
.ml-progress-card{display:flex;align-items:center;justify-content:space-between;gap:22px;margin-top:56px;padding:24px;border:1px solid #cfd7e9;border-radius:11px;background:#e9edf8}.ml-progress-card span{display:block;color:#6f7b96;font-size:10px;font-weight:800;letter-spacing:.09em}.ml-progress-card strong{display:block;margin-top:6px;color:#17203b;font-size:15px}.ml-progress-card__actions{display:flex;flex-wrap:wrap;gap:9px}.ml-progress-card button,.ml-progress-card a{display:inline-flex;min-height:44px;align-items:center;padding:10px 15px;border-radius:8px;font-size:12px;font-weight:780;cursor:pointer}.ml-progress-card button{border:1px solid #aab8db;color:#263d88;background:#fff}.ml-progress-card a{color:#fff;background:#263d88;text-decoration:none}@media(max-width:650px){.ml-progress-card{align-items:stretch;flex-direction:column}}
</style>
