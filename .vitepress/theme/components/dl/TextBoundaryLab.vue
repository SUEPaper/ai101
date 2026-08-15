<script setup lang="ts">
import { computed, ref } from 'vue'
const sample=ref<'accent'|'emoji'|'mixed'>('mixed'), normalize=ref<'none'|'NFC'|'NFKC'>('NFC')
const samples={accent:'Cafe\u0301',emoji:'👩🏽‍💻',mixed:'ＡI模型 Cafe\u0301 👩🏽‍💻'}
const raw=computed(()=>samples[sample.value]), text=computed(()=>normalize.value==='none'?raw.value:raw.value.normalize(normalize.value))
const codepoints=computed(()=>Array.from(text.value)), bytes=computed(()=>new TextEncoder().encode(text.value).length)
const graphemes=computed(()=>Array.from(new Intl.Segmenter('zh',{granularity:'grapheme'}).segment(text.value),x=>x.segment))
</script>
<template><DlExperimentFrame index="16-A" title="文本边界显微镜" question="同一段可见文字在 UTF-8 bytes、Unicode code points、grapheme clusters 和 token 中为什么长度不同？" formula="bytes ≠ code points ≠ graphemes ≠ model tokens">
<template #visual><div class="microscope" role="img" aria-label="文本不同边界层级"><p>{{text}}</p><section><span v-for="(g,i) in graphemes" :key="i">{{g}}</span></section><code>{{codepoints.map(x=>'U+'+x.codePointAt(0)?.toString(16).toUpperCase()).join(' · ')}}</code></div></template>
<template #controls><label>样例<select v-model="sample"><option value="mixed">全角、组合音标与 emoji</option><option value="accent">组合音标</option><option value="emoji">Emoji ZWJ 序列</option></select></label><label>规范化<select v-model="normalize"><option value="none">不规范化</option><option value="NFC">NFC：规范组合</option><option value="NFKC">NFKC：兼容折叠</option></select></label></template>
<template #metrics><div><span>UTF-8 bytes</span><strong>{{bytes}}</strong></div><div><span>Code points</span><strong>{{codepoints.length}}</strong></div><div><span>Graphemes</span><strong>{{graphemes.length}}</strong></div><div><span>JS UTF-16 units</span><strong>{{text.length}}</strong></div></template>
<template #note>NFKC 会把全角Ａ折为 A，可能有利于检索，却也可能删除业务有意义的兼容差异。规范化规则必须在训练、索引、推理和日志中一致。</template>
</DlExperimentFrame></template>
<style scoped>.microscope{min-height:280px;padding:24px;border:1px solid #2c5f65;border-radius:7px;background:#0b172a}.microscope p{color:#fff;font:30px serif}.microscope section{display:flex;flex-wrap:wrap;gap:6px}.microscope span{padding:10px;border:1px solid #58d3c8;background:#173948;color:#dff}.microscope code{display:block;margin-top:24px;color:#f2c14e;white-space:normal}</style>
