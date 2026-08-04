# 数据科学课程 Python 环境决策

## 决策

数据科学教程使用独立的本地 Python 环境，不把 Python 解释器打包进 VitePress 首屏。学习者按 `requirements-ds.txt` 安装 NumPy、Pandas、Matplotlib、Seaborn 与 scikit-learn；网页交互继续使用轻量 Vue/SVG 实现。

## 原因

- NumPy 与 Pandas 是课程第 02–19、24–26 讲的数据结构和数值计算基础。
- Matplotlib 与 Seaborn 只服务于第 07–08 讲及作业中的统计图形，现有依赖无法替代 Python 端出图。
- scikit-learn 用于第 13、16、22–25 讲展示可迁移的估计器、Pipeline 和预处理接口；课程同时提供 NumPy 从零实现，避免把 API 调用当作唯一学习路径。
- 浏览器端交互只展示概念机制，不要求每个页面下载大型 Python/WASM 运行时，保证静态网站加载和移动端阅读。

## 版本策略

采用主版本上界与经过课程 API 兼容性验证的最低版本，不逐补丁锁定。课程代码只使用这些项目长期稳定的公共 API。若未来提供可下载 Notebook 或 CI 执行环境，再生成精确锁文件并记录 Python 版本。

## 未采用方案

- 当前不为每个页面加载 Pyodide：首屏成本与缓存体积过高。
- 当前不集成完整 JupyterLite：先完成正文、交互、练习和本地可复现环境；需要浏览器内运行时再作为独立阶段建设。
- 不新增第二套 DataFrame、绘图或机器学习库，避免同类 API 并存。

## 验证

- `scripts/validate_ds_python.py` 提取教程中的 Python 代码块并逐页校验。
- 不依赖 Matplotlib、Seaborn 或 scikit-learn 的代码块在课程运行时中实际执行。
- 需要图形后端、独立训练数据或可选第三方包的片段标记为语法检查，并在完整课程环境中运行 Notebook 时再次验证。
