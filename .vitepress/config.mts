import { defineConfig } from 'vitepress'
import tailwindcss from '@tailwindcss/vite'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  srcDir: "docs",
  vite: {
    plugins: [tailwindcss()]
  },
  
  title: "AI101",
  description: "人工智能系列课程",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: []
  }
})
