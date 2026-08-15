<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps<{ lessonId: string; nextHref?: string; nextLabel?: string }>()
const done = ref(false)
const storageKey = 'ai101-dl-completed'

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
  window.dispatchEvent(new Event('dl-progress-updated'))
}

onMounted(readCompleted)
</script>

<template>
  <footer class="dl-progress-card">
    <div><span>本章学习记录</span><strong>{{ done ? '正文、实验与论文导读已完成' : '完成实验和练习后再记录' }}</strong></div>
    <div class="dl-progress-card__actions">
      <button type="button" :aria-pressed="done" @click="toggle">{{ done ? '撤销完成' : '标记为已完成' }}</button>
      <a v-if="nextHref" :href="withBase(nextHref)">{{ nextLabel ?? '下一章' }} →</a>
    </div>
  </footer>
</template>

<style scoped>
.dl-progress-card{display:flex;align-items:center;justify-content:space-between;gap:22px;margin-top:56px;padding:24px;border:1px solid #bdd6d5;border-radius:9px;background:#e5f1f0}.dl-progress-card span{display:block;color:#60767e;font-size:10px;font-weight:800;letter-spacing:.09em}.dl-progress-card strong{display:block;margin-top:6px;color:#13213c;font-size:15px}.dl-progress-card__actions{display:flex;flex-wrap:wrap;gap:9px}.dl-progress-card button,.dl-progress-card a{display:inline-flex;min-height:44px;align-items:center;padding:10px 15px;border-radius:7px;font-size:12px;font-weight:780;cursor:pointer}.dl-progress-card button{border:1px solid #7fb4b0;color:#176b69;background:#fff}.dl-progress-card a{color:#fff;background:#175d64;text-decoration:none}@media(max-width:650px){.dl-progress-card{align-items:stretch;flex-direction:column}}
</style>
