<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  question: string
  options: string[]
  answer: number
  explanation: string
}>()

const selected = ref<number | null>(null)
const submitted = ref(false)

function check() {
  if (selected.value !== null) submitted.value = true
}
</script>

<template>
  <section class="quiz-block" aria-label="知识检查">
    <p class="quiz-block__label">CHECKPOINT</p>
    <h3>{{ question }}</h3>
    <div class="quiz-block__options">
      <label v-for="(option, index) in options" :key="option" :class="{ 'is-selected': selected === index }">
        <input v-model="selected" type="radio" :name="question" :value="index" @change="submitted = false">
        <span>{{ String.fromCharCode(65 + index) }}</span>
        {{ option }}
      </label>
    </div>
    <button type="button" :disabled="selected === null" @click="check">检查答案</button>
    <p v-if="submitted" class="quiz-block__feedback" :class="selected === answer ? 'is-correct' : 'is-wrong'" role="status">
      <strong>{{ selected === answer ? '判断正确。' : '再想一步。' }}</strong>
      {{ explanation }}
    </p>
  </section>
</template>

<style scoped>
.quiz-block { margin: 30px 0; padding: 24px; border: 1px solid #d7e2dc; border-left: 5px solid #2f7c7a; border-radius: 12px; background: #fff; box-shadow: 0 12px 32px rgba(18,47,42,.06); }
.quiz-block__label { margin: 0 0 8px; color: #2f7c7a; font: 800 10px/1 ui-monospace, monospace; letter-spacing: .12em; }
.quiz-block h3 { margin: 0 0 16px; border: 0; color: #162a26; font-size: 17px; }
.quiz-block__options { display: grid; gap: 8px; }
.quiz-block label { display: grid; grid-template-columns: 0 26px 1fr; align-items: center; gap: 9px; min-height: 46px; padding: 8px 12px; border: 1px solid #dce5e0; border-radius: 9px; color: #40504b; cursor: pointer; }
.quiz-block label.is-selected { border-color: #2f7c7a; background: #eff7f3; }
.quiz-block input { width: 0; height: 0; opacity: 0; }
.quiz-block label span { display: grid; place-items: center; width: 25px; height: 25px; border-radius: 50%; color: #5b6965; background: #eef2ef; font: 800 10px/1 ui-monospace, monospace; }
.quiz-block button { min-height: 40px; margin-top: 14px; padding: 9px 15px; border: 0; border-radius: 8px; color: #fff; background: #143b37; font-size: 12px; font-weight: 750; cursor: pointer; }
.quiz-block button:disabled { cursor: not-allowed; opacity: .4; }
.quiz-block__feedback { margin: 14px 0 0; padding: 13px 14px; border-radius: 8px; font-size: 13px; line-height: 1.65; }
.quiz-block__feedback strong { display: block; }
.quiz-block__feedback.is-correct { color: #15513f; background: #e9f5ef; }
.quiz-block__feedback.is-wrong { color: #7c3d2c; background: #fff0eb; }
</style>
