#!/usr/bin/env python3
"""Validate the structure and internal links of the data science course."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
COURSE = DOCS / "courses" / "data-science"


def resolve_course_href(href: str) -> Path | None:
    clean = href.split("#", 1)[0].split("?", 1)[0]
    if not clean or clean.startswith(("http://", "https://", "mailto:", "#")):
        return None
    if clean.startswith("/"):
        target = DOCS / clean.lstrip("/")
    else:
        target = COURSE / clean
    if target.suffix == ".md":
        return target.resolve()
    if target.suffix:
        return target.resolve()
    return target.with_suffix(".md").resolve()


def main() -> None:
    errors: list[str] = []
    lesson_files = sorted(COURSE.glob("[0-9][0-9]-*.md"))
    numbers = [path.name[:2] for path in lesson_files]
    expected = [f"{number:02d}" for number in range(27)]
    if numbers != expected:
        errors.append(f"讲次文件不完整：期望 {expected}，实际 {numbers}")

    interaction_names: set[str] = set()
    for path in lesson_files:
        text = path.read_text(encoding="utf-8")
        number = path.name[:2]
        required = {
            "course frontmatter": "course: data-science",
            "lesson hero": "<DsLessonHero",
            "exercise": "代码练习",
            "summary": "本讲小结" if number != "26" else "课程主线收束",
            "references": "## 参考资料",
            "progress": "<DsLessonProgress",
            "originality note": "原创",
        }
        for label, token in required.items():
            if token not in text:
                errors.append(f"{path.relative_to(ROOT)} 缺少 {label}")

        quiz_count = text.count("<QuizBlock")
        if quiz_count < 5:
            errors.append(f"{path.relative_to(ROOT)} 只有 {quiz_count} 个 QuizBlock")

        expected_id = "P0" if number == "00" else number
        if f'lesson-id="{expected_id}"' not in text:
            errors.append(f"{path.relative_to(ROOT)} 的进度 ID 应为 {expected_id}")

        if "ref/ds101" in text:
            errors.append(f"{path.relative_to(ROOT)} 不应把私有参考路径写入教程正文")

        interaction_names.update(re.findall(r"<([A-Z][A-Za-z0-9]*Lab)\s*/>", text))

        hrefs = re.findall(r'(?:\]\(|next-href=")([^)"]+)', text)
        for href in hrefs:
            target = resolve_course_href(href)
            if target is not None and not target.exists():
                errors.append(f"{path.relative_to(ROOT)} 的链接不存在：{href}")

    if len(interaction_names) < 9:
        errors.append(f"核心交互实验不足 9 个：{sorted(interaction_names)}")

    required_pages = [
        COURSE / "assignments.md",
        COURSE / "project.md",
        DOCS / "courses" / "data-science-sources.md",
        DOCS / "courses" / "data-science-content-plan.md",
    ]
    for path in required_pages:
        if not path.exists():
            errors.append(f"缺少课程配套页面：{path.relative_to(ROOT)}")

    if errors:
        raise SystemExit("数据科学内容检查失败：\n- " + "\n- ".join(errors))

    print(
        f"数据科学内容检查通过：{len(lesson_files)} 个预备/讲次页面，"
        f"{len(interaction_names)} 类交互实验，7 次模块作业与 1 个结课项目。"
    )


if __name__ == "__main__":
    main()
