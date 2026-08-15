<script setup lang="ts">
import { ref } from 'vue'

withDefaults(defineProps<{
  question: string
  options: string[]
  answer: number
  explanation: string
  kind?: string
}>(), { kind: '信号检查' })

const selected = ref<number | null>(null)
const submitted = ref(false)
</script>

<template>
  <section class="dl-quiz" aria-label="即时检查">
    <p class="dl-quiz__label">{{ kind }}</p>
    <h3>{{ question }}</h3>
    <div class="dl-quiz__options">
      <label v-for="(option,index) in options" :key="option" :class="{ 'is-selected': selected === index, 'is-correct': submitted && index === answer, 'is-wrong': submitted && selected === index && index !== answer }">
        <input v-model="selected" type="radio" :name="question" :value="index" @change="submitted=false">
        <span>{{ String.fromCharCode(65 + index) }}</span>{{ option }}
      </label>
    </div>
    <button type="button" :disabled="selected===null" @click="submitted=true">检查判断</button>
    <p v-if="submitted" class="dl-quiz__feedback" :class="selected===answer?'is-correct':'is-wrong'" role="status">
      <strong>{{ selected===answer ? '判断成立' : `应选择 ${String.fromCharCode(65 + answer)}` }}</strong>{{ explanation }}
    </p>
  </section>
</template>

<style scoped>
.dl-quiz{margin:30px 0;padding:24px;border:1px solid #ccdcdd;border-left:4px solid #27877f;border-radius:9px;background:#fff;box-shadow:0 10px 28px rgba(17,70,76,.06)}.dl-quiz__label{margin:0 0 9px;color:#207870;font:850 10px/1 ui-monospace,monospace;letter-spacing:.08em}.dl-quiz h3{margin:0 0 16px;border:0;color:#13213c;font-size:17px}.dl-quiz__options{display:grid;gap:8px}.dl-quiz label{display:grid;grid-template-columns:0 27px 1fr;align-items:center;gap:9px;min-height:48px;padding:8px 12px;border:1px solid #d9e4e4;border-radius:7px;color:#405661;cursor:pointer;transition:border-color .18s ease,background-color .18s ease}.dl-quiz label:hover{border-color:#8cb8b5}.dl-quiz label.is-selected{border-color:#4c9d98;background:#edf8f7}.dl-quiz label.is-correct{border-color:#4a9c8b;background:#e9f8f3}.dl-quiz label.is-wrong{border-color:#d8887d;background:#fff3f0}.dl-quiz input{width:0;height:0;opacity:0}.dl-quiz label:focus-within{outline:3px solid rgba(39,135,127,.22);outline-offset:2px}.dl-quiz label span{display:grid;place-items:center;width:26px;height:26px;border-radius:5px;color:#2c6668;background:#e5f0f0;font:800 10px/1 ui-monospace,monospace}.dl-quiz button{min-height:44px;margin-top:15px;padding:10px 16px;border:0;border-radius:7px;color:#fff;background:#175d64;font-size:12px;font-weight:780;cursor:pointer}.dl-quiz button:disabled{cursor:not-allowed;opacity:.42}.dl-quiz__feedback{margin:14px 0 0;padding:14px;border-radius:7px;font-size:13px;line-height:1.7}.dl-quiz__feedback strong{display:block}.dl-quiz__feedback.is-correct{color:#0e5c57;background:#e4f5f2}.dl-quiz__feedback.is-wrong{color:#85392f;background:#fff0ed}@media(prefers-reduced-motion:reduce){.dl-quiz label{transition:none}}
</style>
