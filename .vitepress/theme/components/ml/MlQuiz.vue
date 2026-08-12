<script setup lang="ts">
import { ref } from 'vue'

withDefaults(defineProps<{
  question: string
  options: string[]
  answer: number
  explanation: string
  kind?: string
}>(), {
  kind: '理解检查'
})
const selected = ref<number | null>(null)
const submitted = ref(false)
</script>

<template>
  <section class="ml-quiz" aria-label="即时检查">
    <p class="ml-quiz__label">{{ kind }}</p>
    <h3>{{ question }}</h3>
    <div class="ml-quiz__options">
      <label
        v-for="(option,index) in options"
        :key="option"
        :class="{
          'is-selected': selected === index,
          'is-correct': submitted && index === answer,
          'is-wrong': submitted && selected === index && index !== answer
        }"
      >
        <input v-model="selected" type="radio" :name="question" :value="index" @change="submitted=false">
        <span>{{ String.fromCharCode(65 + index) }}</span>{{ option }}
      </label>
    </div>
    <button type="button" :disabled="selected===null" @click="submitted=true">提交答案</button>
    <p v-if="submitted" class="ml-quiz__feedback" :class="selected===answer?'is-correct':'is-wrong'" role="status">
      <strong>{{ selected===answer ? '答案正确' : `正确答案是 ${String.fromCharCode(65 + answer)}` }}</strong>{{ explanation }}
    </p>
  </section>
</template>

<style scoped>
.ml-quiz{margin:30px 0;padding:24px;border:1px solid #d6ddec;border-left:4px solid #2d51b8;border-radius:10px;background:#fff;box-shadow:0 10px 28px rgba(31,49,102,.05)}.ml-quiz__label{margin:0 0 9px;color:#2d51b8;font:850 10px/1 ui-monospace,monospace;letter-spacing:.08em}.ml-quiz h3{margin:0 0 16px;border:0;color:#17203b;font-size:17px}.ml-quiz__options{display:grid;gap:8px}.ml-quiz label{display:grid;grid-template-columns:0 27px 1fr;align-items:center;gap:9px;min-height:48px;padding:8px 12px;border:1px solid #dce1ed;border-radius:8px;color:#48536d;cursor:pointer;transition:border-color .18s ease,background-color .18s ease}.ml-quiz label:hover{border-color:#9daacf}.ml-quiz label.is-selected{border-color:#6179ba;background:#f0f3fb}.ml-quiz label.is-correct{border-color:#56a79e;background:#ebf8f5}.ml-quiz label.is-wrong{border-color:#d68b80;background:#fff4f1}.ml-quiz input{width:0;height:0;opacity:0}.ml-quiz label:focus-within{outline:3px solid rgba(45,81,184,.22);outline-offset:2px}.ml-quiz label span{display:grid;place-items:center;width:26px;height:26px;border-radius:6px;color:#566687;background:#e9edf7;font:800 10px/1 ui-monospace,monospace}.ml-quiz button{min-height:44px;margin-top:15px;padding:10px 16px;border:0;border-radius:8px;color:#fff;background:#263d88;font-size:12px;font-weight:780;cursor:pointer}.ml-quiz button:disabled{cursor:not-allowed;opacity:.42}.ml-quiz__feedback{margin:14px 0 0;padding:14px;border-radius:8px;font-size:13px;line-height:1.7}.ml-quiz__feedback strong{display:block}.ml-quiz__feedback.is-correct{color:#0e5c57;background:#e4f5f2}.ml-quiz__feedback.is-wrong{color:#85392f;background:#fff0ed}@media(prefers-reduced-motion:reduce){.ml-quiz label{transition:none}}
</style>
