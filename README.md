# AI101

AI101 是一个基于 VitePress 的人工智能系列课程静态网站，面向四门课程入口：

- 数据科学
- 机器学习
- 深度学习
- 机器学习实践课

站点目标是提供一个清晰、稳定、可扫描的课程主页，让学习者能从首页进入完整学习路线和各课程页面。

## 技术栈

- VitePress
- Vue
- Tailwind CSS
- pnpm

## 目录结构

```text
.
├── docs/                         # 课程内容和首页 Markdown 入口
│   ├── index.md                  # 首页入口，渲染自定义首页组件
│   ├── courses/                  # 四门课程页面
│   └── public/                   # 静态资源
├── .vitepress/
│   ├── config.mts                # VitePress 配置
│   └── theme/                    # 自定义主题、导航和首页组件
├── .github/workflows/deploy.yml  # GitHub Pages 自动部署流程
├── ROADMAP.md                    # 当前进度、待办和验证记录
└── package.json                  # 本地开发和构建命令
```

## 本地开发

安装依赖：

```bash
pnpm install
```

启动本地预览：

```bash
pnpm docs:dev
```

默认本地地址通常是：

```text
http://localhost:5173/
```

## 构建

```bash
pnpm docs:build
```

构建产物输出到：

```text
.vitepress/dist
```

本地预览构建结果：

```bash
pnpm docs:preview
```

## 部署

当前部署目标：

```text
https://suepaper.github.io/ai101/
```

由于站点部署在 GitHub Pages 的仓库子路径 `/ai101/` 下，VitePress 配置中需要保留：

```ts
base: "/ai101/",
```

部署流程使用 GitHub Actions：

```text
.github/workflows/deploy.yml
```

推送到 `main` 分支后，工作流会自动执行：

1. 安装依赖
2. 执行 `pnpm docs:build`
3. 上传 `.vitepress/dist`
4. 发布到 GitHub Pages

GitHub 仓库的 Pages 设置需要选择：

```text
Settings -> Pages -> Build and deployment -> Source -> GitHub Actions
```

## 内容维护

- 首页视觉和课程卡片在 `.vitepress/theme/components/AiHome.vue` 中维护。
- 顶部导航在 `.vitepress/theme/components/CourseNav.vue` 中维护。
- 四门课程页面在 `docs/courses/` 中维护。
- 进度、待办和验证记录维护在 `ROADMAP.md` 中。

所有仓库文档默认使用中文。代码符号、命令、文件名和第三方产品名可保留英文。
