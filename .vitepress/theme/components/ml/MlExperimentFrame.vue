<script setup lang="ts">
defineProps<{
  index: string
  title: string
  question: string
  formula: string
}>()
</script>

<template>
  <section class="ml-experiment" :aria-label="`交互实验 ${index}：${title}`">
    <header>
      <div>
        <span>交互实验 {{ index }}</span>
        <h3>{{ title }}</h3>
        <p>{{ question }}</p>
      </div>
      <code>{{ formula }}</code>
    </header>
    <div class="ml-experiment__body">
      <div class="ml-experiment__visual"><slot name="visual" /></div>
      <div class="ml-experiment__controls"><slot name="controls" /></div>
    </div>
    <div class="ml-experiment__metrics" aria-label="实验数值摘要"><slot name="metrics" /></div>
    <p class="ml-experiment__note"><slot name="note" /></p>
  </section>
</template>

<style scoped>
.ml-experiment{margin:36px 0;padding:26px;border:1px solid #334a86;border-radius:14px;color:#eef2ff;background:#17203b;box-shadow:7px 7px 0 #cbd5f2}.ml-experiment>header{display:grid;grid-template-columns:minmax(0,1fr) minmax(210px,.66fr);gap:24px;align-items:start}.ml-experiment>header span{color:#76d6cf;font:850 10px/1 ui-monospace,monospace;letter-spacing:.1em}.ml-experiment h3{margin:8px 0 0;border:0;color:#fff;font-size:21px}.ml-experiment>header p{margin:8px 0 0;color:#aebbdc;font-size:12px;line-height:1.65}.ml-experiment>header code{display:block;padding:12px;border:1px solid #3d548d;border-radius:8px;color:#f4c95d;background:#10182e;font:650 11px/1.6 ui-monospace,monospace;white-space:normal}.ml-experiment__body{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(220px,.72fr);gap:24px;margin-top:24px}.ml-experiment__visual{min-width:0}.ml-experiment__controls{display:grid;align-content:start;gap:18px}.ml-experiment__metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:12px}.ml-experiment__metrics:deep(>div){min-height:68px;padding:11px;border:1px solid #344675;border-radius:7px;background:#111a31}.ml-experiment__metrics:deep(span),.ml-experiment__metrics:deep(strong){display:block}.ml-experiment__metrics:deep(span){color:#8f9cbc;font-size:9px;line-height:1.4}.ml-experiment__metrics:deep(strong){margin-top:6px;color:#f4c95d;font:760 13px/1.35 ui-monospace,monospace}.ml-experiment__note{margin:14px 0 0;padding:14px;border-left:3px solid #72d5ce;color:#c3cde4;background:#10182e;font-size:12px;line-height:1.75}@media(max-width:760px){.ml-experiment{padding:22px 18px}.ml-experiment>header,.ml-experiment__body{grid-template-columns:1fr}.ml-experiment__metrics{grid-template-columns:1fr 1fr}}@media(max-width:440px){.ml-experiment__metrics{grid-template-columns:1fr}}@media(prefers-reduced-motion:reduce){.ml-experiment *{transition-duration:.01ms!important}}
</style>
