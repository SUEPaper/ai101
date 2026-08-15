<script setup lang="ts">
defineProps<{ index: string; title: string; question: string; formula: string }>()
</script>

<template>
  <section class="dl-experiment" :aria-label="`交互实验 ${index}：${title}`">
    <header>
      <div><span>INTERACTIVE TRACE · {{ index }}</span><h3>{{ title }}</h3><p>{{ question }}</p></div>
      <code>{{ formula }}</code>
    </header>
    <div class="dl-experiment__body">
      <div class="dl-experiment__visual"><slot name="visual" /></div>
      <div class="dl-experiment__controls"><slot name="controls" /></div>
    </div>
    <div class="dl-experiment__metrics" aria-label="实验数值摘要"><slot name="metrics" /></div>
    <p class="dl-experiment__note"><slot name="note" /></p>
  </section>
</template>

<style scoped>
.dl-experiment{margin:36px 0;padding:26px;border:1px solid #315765;border-radius:11px;color:#eef7f7;background:#13213c;box-shadow:8px 8px 0 #acd9d5}.dl-experiment>header{display:grid;grid-template-columns:minmax(0,1fr) minmax(210px,.66fr);gap:24px;align-items:start}.dl-experiment>header span{color:#65d6cd;font:850 10px/1 ui-monospace,monospace;letter-spacing:.1em}.dl-experiment h3{margin:8px 0 0;border:0;color:#fff;font-size:21px}.dl-experiment>header p{margin:8px 0 0;color:#afc5cb;font-size:12px;line-height:1.65}.dl-experiment>header code{display:block;min-width:0;padding:12px;border:1px solid #36586b;border-radius:7px;color:#f2c14e;background:#0c172b;font:650 11px/1.6 ui-monospace,monospace;white-space:normal;overflow-wrap:anywhere;word-break:break-word}.dl-experiment__body{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(220px,.72fr);gap:24px;margin-top:24px}.dl-experiment__visual{min-width:0}.dl-experiment__controls{display:grid;align-content:start;gap:15px;min-width:0}.dl-experiment__controls:deep(label){display:grid;gap:7px;min-width:0;color:#b8ced3;font-size:11px}.dl-experiment__controls:deep(input[type=range]){width:100%;min-width:0;min-height:44px;accent-color:#58d3c8}.dl-experiment__controls:deep(select),.dl-experiment__controls:deep(button){min-width:0;min-height:44px;border:1px solid #537987;border-radius:6px;color:#e9f5f4;background:#1a3048}.dl-experiment__controls:deep(button){padding:8px 12px;font-weight:750;cursor:pointer}.dl-experiment__metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:14px}.dl-experiment__metrics:deep(>div){min-height:68px;padding:11px;border:1px solid #315267;border-radius:6px;background:#0d192e}.dl-experiment__metrics:deep(span),.dl-experiment__metrics:deep(strong){display:block}.dl-experiment__metrics:deep(span){color:#8ca7af;font-size:9px;line-height:1.4}.dl-experiment__metrics:deep(strong){margin-top:6px;color:#f2c14e;font:760 13px/1.35 ui-monospace,monospace}.dl-experiment__note{margin:14px 0 0;padding:14px;border-left:3px solid #58d3c8;color:#bfd1d5;background:#0c172b;font-size:12px;line-height:1.75}@media(max-width:760px){.dl-experiment{padding:22px 18px}.dl-experiment>header,.dl-experiment__body{grid-template-columns:1fr}.dl-experiment__metrics{grid-template-columns:1fr 1fr}}@media(max-width:440px){.dl-experiment__metrics{grid-template-columns:1fr}}@media(prefers-reduced-motion:reduce){.dl-experiment *{transition-duration:.01ms!important}}
</style>
