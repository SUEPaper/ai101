#!/usr/bin/env python3
"""检查机器学习网页版教材的章节结构、练习数量与内部链接。"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
COURSE = DOCS / "courses" / "machine-learning"
COURSE_HOME = DOCS / "courses" / "machine-learning.md"
CHAPTER_STRUCTURE = DOCS / "courses" / "machine-learning-chapter-structure.md"
COURSE_DATA = ROOT / ".vitepress" / "theme" / "data" / "mlCourse.ts"


def resolve_href(source: Path, href: str) -> Path | None:
    clean = href.split("#", 1)[0].split("?", 1)[0]
    if not clean or clean.startswith(("http://", "https://", "mailto:", "#")):
        return None
    if clean.startswith("/"):
        target = DOCS / clean.lstrip("/")
    else:
        target = source.parent / clean
    if target.suffix:
        return target.resolve()
    return target.with_suffix(".md").resolve()


def check_links(path: Path, text: str, errors: list[str]) -> None:
    hrefs = re.findall(r'(?:\]\(|(?:next-)?href=")([^)"]+)', text)
    for href in hrefs:
        target = resolve_href(path, href)
        if target is not None and not target.exists():
            errors.append(f"{path.relative_to(ROOT)} 的链接不存在：{href}")


def parse_chapter_titles(errors: list[str]) -> dict[str, str]:
    if not CHAPTER_STRUCTURE.exists():
        errors.append(f"缺少章节结构文档：{CHAPTER_STRUCTURE.relative_to(ROOT)}")
        return {}
    titles: dict[str, str] = {}
    for line in CHAPTER_STRUCTURE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and re.fullmatch(r"\d{2}", cells[1]):
            titles[cells[1]] = cells[2]
    expected = [f"{number:02d}" for number in range(1, 33)]
    if list(titles) != expected:
        errors.append(f"章节结构文档未给出连续 01–32 章：实际 {list(titles)}")
    return titles


def frontmatter_value(text: str, key: str) -> str | None:
    match = re.match(r"^---\n(.*?)\n---", text, flags=re.DOTALL)
    if not match:
        return None
    value = re.search(rf"^{re.escape(key)}:\s*(.+)$", match.group(1), flags=re.MULTILINE)
    return value.group(1).strip() if value else None


def main() -> None:
    errors: list[str] = []
    chapter_titles = parse_chapter_titles(errors)
    course_data = COURSE_DATA.read_text(encoding="utf-8")
    core_files = sorted(COURSE.glob("[0-9][0-9]-*.md"))
    core_numbers = [path.name[:2] for path in core_files]
    expected_core = [f"{number:02d}" for number in range(1, 33)]
    if core_numbers != expected_core:
        errors.append(f"核心章节不完整：期望 {expected_core}，实际 {core_numbers}")

    lab_kinds: set[str] = set()
    quiz_total = 0
    experiment_total = 0
    industrial_numbers: list[str] = []
    for path in core_files:
        text = path.read_text(encoding="utf-8")
        number = path.name[:2]
        is_industrial = frontmatter_value(text, "edition") == "industrial"
        required = {
            "课程 frontmatter": "course: machine-learning",
            "章节导读": "<MlLessonHero",
            "章节进度": "<MlLessonProgress",
        }
        for label, token in required.items():
            if token not in text:
                errors.append(f"{path.relative_to(ROOT)} 缺少{label}")

        if not re.search(r"^## .*练习", text, flags=re.MULTILINE):
            errors.append(f"{path.relative_to(ROOT)} 缺少章节练习")

        quiz_count = text.count("<MlQuiz")
        quiz_total += quiz_count
        if quiz_count < 5:
            errors.append(f"{path.relative_to(ROOT)} 只有 {quiz_count} 道即时测验")

        stop_count = text.count("ml-stop")
        if stop_count < 3:
            errors.append(f"{path.relative_to(ROOT)} 只有 {stop_count} 次停下来推一推")

        labs = re.findall(r"<([A-Z][A-Za-z0-9]*Lab)\b", text)
        experiment_total += len(labs)
        if not labs:
            errors.append(f"{path.relative_to(ROOT)} 缺少交互实验")

        kinds = re.findall(r'<MlLab\s+[^>]*kind="([^"]+)"', text)
        lab_kinds.update(kinds)

        if is_industrial:
            industrial_numbers.append(number)
            expected_title = chapter_titles.get(number)
            actual_title = frontmatter_value(text, "title")
            hero = re.search(r'<MlLessonHero\b[\s\S]*?\btitle="([^"]+)"', text)
            hero_title = hero.group(1) if hero else None
            if actual_title != expected_title:
                errors.append(
                    f"{path.relative_to(ROOT)} 的正式标题应为“{expected_title}”，实际为“{actual_title}”"
                )
            if hero_title != expected_title:
                errors.append(
                    f"{path.relative_to(ROOT)} 的首屏标题应为“{expected_title}”，实际为“{hero_title}”"
                )
            if expected_title and f"core('{number}', '{expected_title}'" not in course_data:
                errors.append(f"课程目录中的第 {number} 章名称未同步为“{expected_title}”")
            specialized_labs = [name for name in labs if name != "MlLab"]
            if len(specialized_labs) < 2:
                errors.append(f"{path.relative_to(ROOT)} 只有 {len(specialized_labs)} 个专用交互实验")
            if text.count("```python") < 3:
                errors.append(f"{path.relative_to(ROOT)} 少于 3 个 Python 实现或验证代码块")
            quiz_kinds = set(re.findall(r'<MlQuiz\s+[^>]*kind="([^"]+)"', text))
            if len(quiz_kinds) < 3:
                errors.append(f"{path.relative_to(ROOT)} 的即时题型少于 3 类：{sorted(quiz_kinds)}")
            for heading in ("## 贯穿案例", "## 数值验证", "## 复杂度与工程边界", "## 来源与许可"):
                if heading not in text:
                    errors.append(f"{path.relative_to(ROOT)} 缺少工业级章节结构：{heading}")
            if text.count("PDF 第") < 2:
                errors.append(f"{path.relative_to(ROOT)} 缺少至少 2 项精确 PDF 页码来源")
        elif len(kinds) != 1:
            errors.append(f"{path.relative_to(ROOT)} 的旧版内容应保留恰好 1 个 MlLab，直至完成专用实验迁移")

        check_links(path, text, errors)

    supplement_files = sorted(COURSE.glob("s[0-9][0-9]-*.md"))
    supplement_numbers = [path.name[:3].upper() for path in supplement_files]
    expected_supplements = [f"S{number:02d}" for number in range(1, 7)]
    if supplement_numbers != expected_supplements:
        errors.append(
            f"选修章节不完整：期望 {expected_supplements}，实际 {supplement_numbers}"
        )
    for path in supplement_files:
        text = path.read_text(encoding="utf-8")
        for label, token in {
            "课程 frontmatter": "course: machine-learning",
            "章节导读": "<MlLessonHero",
            "章节进度": "<MlLessonProgress",
        }.items():
            if token not in text:
                errors.append(f"{path.relative_to(ROOT)} 缺少{label}")
        check_links(path, text, errors)

    required_pages = [
        COURSE_HOME,
        COURSE / "assignments.md",
        COURSE / "project.md",
        COURSE / "sources.md",
        DOCS / "courses" / "machine-learning-textbook-outline.md",
    ]
    for path in required_pages:
        if not path.exists():
            errors.append(f"缺少课程配套页面：{path.relative_to(ROOT)}")
            continue
        check_links(path, path.read_text(encoding="utf-8"), errors)

    assignments = COURSE / "assignments.md"
    if assignments.exists():
        text = assignments.read_text(encoding="utf-8")
        for number in range(1, 9):
            if not re.search(rf"\bA{number}\b", text):
                errors.append(f"{assignments.relative_to(ROOT)} 缺少 A{number}")

    component = ROOT / ".vitepress" / "theme" / "components" / "ml" / "MlLab.vue"
    component_text = component.read_text(encoding="utf-8")
    for kind in sorted(lab_kinds):
        if f"{kind}:" not in component_text:
            errors.append(f"MlLab.vue 缺少 {kind} 实验配置")

    quiz_component = ROOT / ".vitepress" / "theme" / "components" / "ml" / "MlQuiz.vue"
    quiz_component_text = quiz_component.read_text(encoding="utf-8")
    for legacy_label in ("CHECK", "先判断，再看解释"):
        if legacy_label in quiz_component_text:
            errors.append(f"MlQuiz.vue 仍包含已取消的题卡模式：{legacy_label}")

    if errors:
        raise SystemExit("机器学习内容检查失败：\n- " + "\n- ".join(errors))

    print(
        "机器学习内容检查通过："
        f"{len(core_files)} 个核心章节、{len(supplement_files)} 个选修章节、"
        f"{len(industrial_numbers)} 个工业级细化章节（{', '.join(industrial_numbers) or '暂无'}）、"
        f"{experiment_total} 个核心交互实验、{quiz_total} 道即时测验、"
        "8 次综合作业与 1 个结课项目。"
    )


if __name__ == "__main__":
    main()
