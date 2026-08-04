<script setup lang="ts">
import { computed, ref } from 'vue'

const pattern = ref('^([A-Z])-(\\d{2})$')
const samples = ['N-01', 'S-2', 'N-03', 'W-12', 'N-104', 'E_07']
const error = ref('')

const matches = computed(() => {
  try {
    const expression = new RegExp(pattern.value)
    error.value = ''
    return samples.map((sample) => {
      const match = sample.match(expression)
      return { sample, matched: Boolean(match), groups: match?.slice(1) ?? [] }
    })
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '表达式无法解析'
    return samples.map((sample) => ({ sample, matched: false, groups: [] }))
  }
})
</script>

<template>
  <section class="regex-lab" aria-label="正则表达式实验">
    <header><span>INTERACTIVE 06</span><h3>让模式和样本文本当场对话</h3></header>
    <label>正则表达式<input v-model="pattern" type="text" spellcheck="false" aria-describedby="regex-help"></label>
    <p id="regex-help">尝试删除 <code>^</code> 或 <code>$</code>，把 <code>\d{2}</code> 改成 <code>\d+</code>，观察匹配集合如何变化。</p>
    <p v-if="error" class="regex-lab__error" role="status">{{ error }}</p>
    <div class="regex-lab__samples">
      <div v-for="item in matches" :key="item.sample" :class="{ 'is-match': item.matched }">
        <code>{{ item.sample }}</code><b>{{ item.matched ? 'MATCH' : 'SKIP' }}</b><small v-if="item.groups.length">捕获：{{ item.groups.join(' / ') }}</small>
      </div>
    </div>
  </section>
</template>

<style scoped>
.regex-lab { margin: 32px 0; padding: 25px; border-radius: 16px; color: #ebf4f0; background: #122f2a; }
.regex-lab header span { color: #f2be5f; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .12em; }
.regex-lab h3 { margin: 7px 0 20px; border: 0; color: #fff; font-size: 20px; }
.regex-lab label { display: block; color: #b9cdc6; font-size: 11px; font-weight: 750; }
.regex-lab input { display: block; width: 100%; min-height: 44px; margin-top: 8px; padding: 9px 12px; border: 1px solid #5e8076; border-radius: 8px; color: #17332e; background: #fff; font: 700 13px/1.4 ui-monospace, monospace; }
.regex-lab > p { margin: 9px 0 18px; color: #a9c0b8; font-size: 11px; line-height: 1.6; }
.regex-lab > p code { color: #f2c66f; }
.regex-lab__samples { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.regex-lab__samples > div { min-height: 78px; padding: 13px; border: 1px solid #45685e; border-radius: 9px; background: #1d4039; }
.regex-lab__samples code, .regex-lab__samples b, .regex-lab__samples small { display: block; }
.regex-lab__samples code { color: #d1e1db; font-size: 12px; }
.regex-lab__samples b { margin-top: 8px; color: #829b93; font: 800 9px/1 ui-monospace, monospace; }
.regex-lab__samples small { margin-top: 7px; color: #afc5bd; font-size: 9px; }
.regex-lab__samples > div.is-match { border-color: #f2be5f; background: #28564c; }
.regex-lab__samples > div.is-match b { color: #f2be5f; }
.regex-lab__error { padding: 8px; color: #ffd8ce !important; background: #6c3028; }
@media (max-width: 640px) { .regex-lab__samples { grid-template-columns: repeat(2, 1fr); } }
</style>
