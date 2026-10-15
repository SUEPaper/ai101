#!/usr/bin/env python3
"""检查深度学习与大模型课程规划、资料目录和入口状态。"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSES = ROOT / "docs" / "courses"
COURSE_HOME = COURSES / "deep-learning.md"
CHAPTER_STRUCTURE = COURSES / "deep-learning-chapter-structure.md"
CONTENT_PLAN = COURSES / "deep-learning-content-plan.md"
SOURCES = COURSES / "deep-learning-sources.md"
CATALOG = ROOT / "scripts" / "data" / "dl101-resources.json"
HOME_COMPONENT = ROOT / ".vitepress" / "theme" / "components" / "AiHome.vue"
COURSE_DIR = COURSES / "deep-learning"
ASSIGNMENTS = COURSE_DIR / "assignments.md"
PROJECT = COURSE_DIR / "project.md"
COURSE_DATA = ROOT / ".vitepress" / "theme" / "data" / "dlCourse.ts"
THEME_INDEX = ROOT / ".vitepress" / "theme" / "index.ts"
COURSE_IMAGES = (
    ROOT / "docs" / "public" / "images" / "deep-learning" / "training-signal-lab.png",
    ROOT / "docs" / "public" / "images" / "deep-learning" / "vision-model-observatory.png",
    ROOT / "docs" / "public" / "images" / "deep-learning" / "generative-model-observatory.png",
    ROOT / "docs" / "public" / "images" / "deep-learning" / "llm-infra-observatory.png",
)

EXPECTED_MODULE_COUNTS = {
    "01-training": 3,
    "02-vision": 17,
    "03-generative": 10,
    "04-nlp": 5,
    "05-llm": 20,
    "06-infra": 6,
}

REQUIRED_PAPER_IDS = {
    "adam",
    "adamw",
    "alexnet",
    "vgg",
    "resnet",
    "faster-rcnn",
    "yolo",
    "fcn",
    "unet",
    "vit",
    "swin-transformer",
    "gan",
    "word2vec",
    "subword-nmt",
    "triton",
    "tilelang",
    "vllm-pagedattention",
    "sglang",
}


def read(path: Path, errors: list[str]) -> str:
    if not path.exists():
        errors.append(f"缺少文件：{path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8")


def table_numbers(text: str) -> list[str]:
    return re.findall(r"^\|(?:\s*[^|]+\s*\|)?\s*(\d{2})\s*\|", text, flags=re.MULTILINE)


def check_internal_links(path: Path, text: str, errors: list[str]) -> None:
    for href in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        clean = href.split("#", 1)[0].split("?", 1)[0]
        if not clean or clean.startswith(("https://", "http://", "mailto:", "#")):
            continue
        target = (path.parent / clean).resolve()
        if not target.exists():
            errors.append(f"{path.relative_to(ROOT)} 的链接不存在：{href}")


def main() -> None:
    errors: list[str] = []
    structure = read(CHAPTER_STRUCTURE, errors)
    plan = read(CONTENT_PLAN, errors)
    course_home = read(COURSE_HOME, errors)
    assignments = read(ASSIGNMENTS, errors)
    project = read(PROJECT, errors)
    sources = read(SOURCES, errors)
    home_component = read(HOME_COMPONENT, errors)
    course_data = read(COURSE_DATA, errors)
    theme_index = read(THEME_INDEX, errors)
    expected_numbers = [f"{number:02d}" for number in range(1, 33)]

    structure_numbers = table_numbers(structure)
    if structure_numbers != expected_numbers:
        errors.append(f"章节结构不是连续 01–32：{structure_numbers}")

    plan_numbers = table_numbers(plan)
    if plan_numbers != expected_numbers:
        errors.append(f"内容计划不是连续 01–32：{plan_numbers}")

    for token in ("16 周", "32 个正式章节", "8 次综合作业", "1 个结课项目"):
        if token not in structure:
            errors.append(f"章节结构缺少课程合同：{token}")

    for link in (
        "./deep-learning-chapter-structure.md",
        "./deep-learning-content-plan.md",
        "./deep-learning-sources.md",
    ):
        if link not in course_home:
            errors.append(f"课程主页缺少链接：{link}")
    if "状态：已开放" not in course_home:
        errors.append("课程主页未标为已开放")

    for token in ("./deep-learning/assignments.md", "./deep-learning/project.md"):
        if token not in course_home:
            errors.append(f"课程主页缺少实践入口：{token}")

    for number in range(1, 9):
        if not re.search(rf"^## A{number} · ", assignments, flags=re.MULTILINE):
            errors.append(f"综合作业缺少 A{number}")
    for token in ("统一提交与运行合同", "统一评分量规", "学术诚信", "作业完成检查"):
        if token not in assignments:
            errors.append(f"综合作业缺少统一门禁：{token}")
    for token in ("路线 M", "路线 A", "路线 S", "五个里程碑", "评分量规", "发布审查门禁"):
        if token not in project:
            errors.append(f"结课项目缺少项目合同：{token}")

    chapter_files = sorted(COURSE_DIR.glob("[0-9][0-9]-*.md"))
    chapter_numbers = [path.name[:2] for path in chapter_files]
    completed_numbers = [f"{number:02d}" for number in range(1, 33)]
    if chapter_numbers != completed_numbers:
        errors.append(f"当前应完成全部第 01–32 章，实际为：{chapter_numbers}")

    experiment_total = 0
    quiz_total = 0
    paper_total = 0
    for path in chapter_files:
        text = read(path, errors)
        number = path.name[:2]
        required_tokens = {
            "深度学习课程 frontmatter": "course: deep-learning",
            "工业级版本标记": "edition: industrial",
            "章节首屏": f'<DlLessonHero number="{number}"',
            "章节进度": f'<DlLessonProgress lesson-id="{number}"',
            "章节小结": "## 本章小结",
        }
        for label, token in required_tokens.items():
            if token not in text:
                errors.append(f"{path.relative_to(ROOT)} 缺少{label}")
        if not re.search(r"^## (?:\d+\. )?推荐论文阅读$", text, flags=re.MULTILINE):
            errors.append(f"{path.relative_to(ROOT)} 缺少论文阅读部分")

        experiments = re.findall(r"<([A-Z][A-Za-z0-9]+Lab)\s*/>", text)
        experiment_total += len(experiments)
        if len(experiments) < 2:
            errors.append(f"{path.relative_to(ROOT)} 少于 2 个专用交互实验")
        for component in experiments:
            if f"app.component('{component}'" not in theme_index:
                errors.append(f"交互组件未注册：{component}")

        quiz_count = text.count("<DlQuiz")
        quiz_total += quiz_count
        if quiz_count < 5:
            errors.append(f"{path.relative_to(ROOT)} 少于 5 道即时检查")
        if text.count("dl-stop") < 3:
            errors.append(f"{path.relative_to(ROOT)} 少于 3 次推导停顿")
        if text.count("```python") < 3:
            errors.append(f"{path.relative_to(ROOT)} 少于 3 个 Python 实现或验证代码块")

        paper_count = text.count("<DlPaperCard")
        paper_total += paper_count
        if paper_count < 3:
            errors.append(f"{path.relative_to(ROOT)} 少于 3 篇带简介的推荐论文")
        check_internal_links(path, text, errors)

    available_entries = re.findall(
        r"^\s*lesson\('(\d{2})',[^\n]*'available'\),?$",
        course_data,
        flags=re.MULTILINE,
    )
    if available_entries != completed_numbers:
        errors.append(f"课程目录开放状态未与第 01–32 章同步：{available_entries}")

    for component in ("DlLessonHero", "DlLessonProgress", "DlQuiz", "DlPaperCard"):
        if f"app.component('{component}'" not in theme_index:
            errors.append(f"主题未注册课程组件：{component}")

    for course_image in COURSE_IMAGES:
        if not course_image.exists() or course_image.stat().st_size < 100_000:
            errors.append(f"缺少有效的原创课程插图：{course_image.name}")
    for token in (
        "training-signal-lab.png",
        "vision-model-observatory.png",
        "generative-model-observatory.png",
        "llm-infra-observatory.png",
        "alt=",
        "64 个专用交互实验",
    ):
        if token not in course_home:
            errors.append(f"课程主页缺少图片或交互说明：{token}")

    try:
        catalog = json.loads(read(CATALOG, errors))
    except json.JSONDecodeError as exc:
        errors.append(f"资料目录不是合法 JSON：{exc}")
        catalog = {}

    slide_sources = catalog.get("slide_sources", [])
    papers = catalog.get("papers", [])
    if len(slide_sources) != 5:
        errors.append(f"课件来源应为 5 个，实际为 {len(slide_sources)} 个")
    if len(papers) != 61:
        errors.append(f"推荐论文应为 61 篇，实际为 {len(papers)} 篇")

    paper_ids = [paper.get("id") for paper in papers]
    paper_urls = [paper.get("url") for paper in papers]
    if len(paper_ids) != len(set(paper_ids)):
        errors.append("资料目录存在重复论文 id")
    if len(paper_urls) != len(set(paper_urls)):
        errors.append("资料目录存在重复论文 URL")

    missing_ids = sorted(REQUIRED_PAPER_IDS - set(paper_ids))
    if missing_ids:
        errors.append(f"资料目录缺少指定主题论文：{missing_ids}")

    module_counts: dict[str, int] = {}
    for paper in papers:
        module = paper.get("module")
        module_counts[module] = module_counts.get(module, 0) + 1
        for field in ("id", "module", "module_title", "level", "title", "url"):
            if not paper.get(field):
                errors.append(f"论文条目 {paper.get('id', '<unknown>')} 缺少字段：{field}")
        if paper.get("title") not in sources or paper.get("url") not in sources:
            errors.append(f"来源登记未列出论文：{paper.get('id')}")
    if module_counts != EXPECTED_MODULE_COUNTS:
        errors.append(f"论文模块数量不符：{module_counts}")

    if "title: '深度学习与大模型',\n    status: '已开放',\n    available: true" not in home_component:
        errors.append("首页深度学习与大模型课程卡未标为已开放")
    if "4 门已开放 · 1 门规划中" not in home_component:
        errors.append("首页课程总状态未同步")

    for path, text in (
        (COURSE_HOME, course_home),
        (ASSIGNMENTS, assignments),
        (PROJECT, project),
        (CHAPTER_STRUCTURE, structure),
        (CONTENT_PLAN, plan),
        (SOURCES, sources),
    ):
        check_internal_links(path, text, errors)

    if errors:
        raise SystemExit("深度学习与大模型内容检查失败：\n- " + "\n- ".join(errors))

    print(
        "深度学习与大模型内容检查通过："
        "6 个模块、32 个正式章节、8 次综合作业、1 个结课项目、"
        f"{len(slide_sources)} 个课件来源和 {len(papers)} 篇推荐论文；"
        f"已完成 {len(chapter_files)} 章、{experiment_total} 个专用交互实验、"
        f"{quiz_total} 道即时检查和 {paper_total} 项论文导读。"
    )


if __name__ == "__main__":
    main()
