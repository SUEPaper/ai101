# AI101

AI101 是一个基于 VitePress 的人工智能系列课程静态网站，提供五门课程入口：

- 数值分析
- 数据科学
- 机器学习
- 深度学习与大模型
- 人工智能实践课

站点目标是提供一个清晰、稳定、可扫描的课程主页，让学习者能从首页进入完整学习路线和各课程页面。

## 课程状态

| 课程 | 当前状态 |
|---|---|
| 数值分析 | 已开放：系统学习计算方法并结合算法实现与数值实验，包含 85 篇文档、8 个实验及配套工具教程 |
| 数据科学 | 已开放 |
| 机器学习 | 已开放 |
| 深度学习与大模型 | 已开放：32 章、64 个专用交互、8 次综合作业和 1 个结课项目 |
| 人工智能实践课 | 规划中 |

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
│   ├── courses/                  # 五门课程页面及配套内容
│   ├── decisions/                # 环境与架构决策记录
│   └── public/                   # 静态资源
├── .vitepress/
│   ├── config.mts                # VitePress 配置
│   └── theme/                    # 自定义主题、导航和首页组件；数据科学组件位于 components/ds
├── .github/workflows/deploy.yml  # GitHub Pages 自动部署流程
├── ROADMAP.md                    # 当前进度、待办和验证记录
├── requirements-ds.txt           # 数据科学教程 Python 环境
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

## 课程内容验证

创建 Python 环境并安装课程依赖：

```bash
python3 -m venv .venv-ds
source .venv-ds/bin/activate
python -m pip install -r requirements-ds.txt
```

检查数据科学、机器学习以及深度学习与大模型课程规划的结构、内部链接和配套内容：

```bash
pnpm docs:check-content
```

提取并校验教程中的 Python 代码块：

```bash
pnpm docs:check-python
```

代码校验按页面共享命名空间运行；标记为 `tutorial-validation: syntax-only` 的图形或第三方工作流片段只做语法检查。

深度学习与大模型课程的第三方参考 PDF 保存在被 Git 忽略的 `ref/dl101/`，不会随网站发布。安装 `pypdf` 后可同步公开课件与推荐论文，或只复核已有缓存：

```bash
python3 scripts/sync_dl101_resources.py
python3 scripts/sync_dl101_resources.py --verify-only
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
- 五门课程页面在 `docs/courses/` 中维护。
- 数值分析沿用 Math201 的正文与资源，内容位于 `docs/courses/numerical-computing/`，课程目录在 `.vitepress/theme/data/mathCourse.ts` 中维护。
- 进度、待办和验证记录维护在 `ROADMAP.md` 中。

所有仓库文档默认使用中文。代码符号、命令、文件名和第三方产品名可保留英文。
