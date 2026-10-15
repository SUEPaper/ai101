# 机器学习教材公式渲染决策

## 背景

机器学习教材要求在网页中完整呈现矩阵、分段函数、多行等式、梯度、Hessian、期望、条件概率和优化约束。使用 Unicode 或代码块只能近似表达公式，无法稳定呈现上下标、对齐推导和可访问的数学结构，也会让初学者难以逐行核对符号。

## 决策

启用 VitePress 的内置 `markdown.math` 接口，并增加其明确要求的可选开发依赖 `markdown-it-mathjax3@^4`。教材正文继续使用标准 Markdown 数学定界符；交互实验中的短公式使用普通文本，避免在频繁更新的 Vue 状态中重复排版。

## 依据

- 当前 VitePress 版本已经包含 MathJax 接入点，并在本地类型声明和运行时代码中明确要求 `markdown-it-mathjax3@^4`。
- MathJax 输出可缩放，适合长公式、矩阵和移动端换行；也比自行维护公式图片更利于复制、搜索和辅助技术读取。
- 项目现有依赖不能提供等价的 Markdown 数学渲染；自行实现解析器会引入更大的维护边界。

## 影响与边界

- 站点构建时增加数学解析成本，但不在每个页面加载 Python 运行时。
- 公式较长时必须使用多行推导并在公式前后解释符号，不把可读性问题交给横向滚动解决。
- 新增依赖只用于构建教材公式，不改变课程数据、部署方式或后端边界。

## 2026-10-15 依赖升级复核

本次更新沿用已有发布渠道：Tailwind CSS 与 `@tailwindcss/vite` 使用最新正式版 `4.3.3`，VitePress 从 `2.0.0-alpha.18` 更新到 `next` 渠道的 `2.0.0-alpha.20`，继续固定精确版本。

`markdown-it-mathjax3` 的最新正式版为 `5.2.0`，但 VitePress `2.0.0-alpha.20` 的 `peerDependencies` 仍要求 `markdown-it-mathjax3@^4`，因此保留最高兼容版本 `4.3.2`。不使用强制安装或依赖覆盖来跨越这个兼容边界；待 VitePress 声明支持新版插件后再升级。

版本标签与兼容范围已通过 npm 官方仓库查询；VitePress 的限制也可在[官方版本清单](https://github.com/vuejs/vitepress/blob/v2.0.0-alpha.20/package.json) 中核对。
