#!/usr/bin/env python3
"""检查机器学习网页版教材的章节结构、练习数量与内部链接。"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
COURSE = DOCS / "courses" / "machine-learning"
COURSE_HOME = DOCS / "courses" / "machine-learning.md"


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


def main() -> None:
    errors: list[str] = []
    core_files = sorted(COURSE.glob("[0-9][0-9]-*.md"))
    core_numbers = [path.name[:2] for path in core_files]
    expected_core = [f"{number:02d}" for number in range(1, 33)]
    if core_numbers != expected_core:
        errors.append(f"核心章节不完整：期望 {expected_core}，实际 {core_numbers}")

    lab_kinds: set[str] = set()
    quiz_total = 0
    for path in core_files:
        text = path.read_text(encoding="utf-8")
        required = {
            "课程 frontmatter": "course: machine-learning",
            "章节导读": "<MlLessonHero",
            "交互实验": "<MlLab",
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

        kinds = re.findall(r'<MlLab\s+[^>]*kind="([^"]+)"', text)
        if len(kinds) != 1:
            errors.append(f"{path.relative_to(ROOT)} 应恰有 1 个具名交互实验")
        lab_kinds.update(kinds)
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

    if len(lab_kinds) != 32:
        errors.append(f"应有 32 类核心交互实验，实际为 {len(lab_kinds)}：{sorted(lab_kinds)}")

    component = ROOT / ".vitepress" / "theme" / "components" / "ml" / "MlLab.vue"
    component_text = component.read_text(encoding="utf-8")
    for kind in sorted(lab_kinds):
        if f"{kind}:" not in component_text:
            errors.append(f"MlLab.vue 缺少 {kind} 实验配置")

    if errors:
        raise SystemExit("机器学习内容检查失败：\n- " + "\n- ".join(errors))

    print(
        "机器学习内容检查通过："
        f"{len(core_files)} 个核心章节、{len(supplement_files)} 个选修章节、"
        f"{len(lab_kinds)} 类交互实验、{quiz_total} 道即时测验、"
        "8 次综合作业与 1 个结课项目。"
    )


if __name__ == "__main__":
    main()
