// https://vitepress.dev/guide/custom-theme
import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import AiHome from './components/AiHome.vue'
import CourseLayout from './components/CourseLayout.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(CourseLayout)
  },
  enhanceApp({ app }) {
    app.component('AiHome', AiHome)
  }
} satisfies Theme
