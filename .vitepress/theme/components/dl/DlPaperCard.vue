<script setup lang="ts">
defineProps<{
  title: string
  meta: string
  priority: string
  question: string
  reading: string
  href: string
}>()
</script>

<template>
  <article class="dl-paper-card">
    <div class="dl-paper-card__meta"><span>{{ priority }}</span><small>{{ meta }}</small></div>
    <h3>{{ title }}</h3>
    <p><strong>带着问题读：</strong>{{ question }}</p>
    <p><strong>建议阅读：</strong>{{ reading }}</p>
    <a :href="href" target="_blank" rel="noopener noreferrer">打开论文 PDF<span class="sr-only">（在新窗口打开）</span> →</a>
  </article>
</template>

<style scoped>
.dl-paper-card{margin:20px 0;padding:20px;border:1px solid #bed5d4;border-radius:8px;background:#f7fbfb}.dl-paper-card__meta{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px}.dl-paper-card__meta span{padding:4px 7px;border-radius:4px;color:#103f43;background:#bce7e2;font:800 9px/1 ui-monospace,monospace;letter-spacing:.08em}.dl-paper-card__meta small{color:#6b7d84}.dl-paper-card h3{margin:12px 0 10px;border:0;color:#13213c;font-size:18px}.dl-paper-card p{margin:7px 0;color:#4b626c;font-size:13px;line-height:1.7}.dl-paper-card a{display:inline-flex;min-height:44px;align-items:center;margin-top:8px;color:#176b69;font-size:12px;font-weight:800;text-decoration:none}.dl-paper-card a:hover{text-decoration:underline;text-underline-offset:4px}
</style>
