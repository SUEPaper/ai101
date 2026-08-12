#!/usr/bin/env python3
"""执行已标记为工业级的机器学习章节中的可运行 Python 代码块。"""

from __future__ import annotations

import re
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "docs" / "courses" / "machine-learning"


def is_industrial(text: str) -> bool:
    frontmatter = re.match(r"^---\n(.*?)\n---", text, flags=re.DOTALL)
    return bool(frontmatter and re.search(r"^edition:\s*industrial$", frontmatter.group(1), re.MULTILINE))


def main() -> None:
    executed = 0
    skipped = 0
    failures: list[str] = []

    for path in sorted(COURSE.glob("[0-9][0-9]-*.md")):
        text = path.read_text(encoding="utf-8")
        if not is_industrial(text):
            continue

        namespace: dict[str, object] = {"__name__": "__ml_textbook_validation__"}
        blocks = re.findall(r"^```python\n(.*?)^```", text, flags=re.DOTALL | re.MULTILINE)
        for index, code in enumerate(blocks, start=1):
            if "# TODO" in code or "# 不执行" in code:
                skipped += 1
                continue
            try:
                exec(compile(code, f"{path.name}:python-block-{index}", "exec"), namespace)
                executed += 1
            except Exception:
                failures.append(
                    f"{path.relative_to(ROOT)} 第 {index} 个 Python 代码块失败：\n"
                    + traceback.format_exc()
                )

    if failures:
        raise SystemExit("机器学习 Python 验证失败：\n" + "\n".join(failures))
    if executed == 0:
        raise SystemExit("机器学习 Python 验证失败：没有找到可执行的工业级章节代码块")

    print(f"机器学习 Python 验证通过：执行 {executed} 个代码块，跳过 {skipped} 个练习骨架。")


if __name__ == "__main__":
    main()

