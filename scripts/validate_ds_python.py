"""依次执行数据科学教程中的 Python 代码块。"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = ROOT / "docs" / "courses" / "data-science"
COURSE_INDEX = ROOT / "docs" / "courses" / "data-science.md"
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def validate_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    namespace: dict[str, object] = {"__name__": "__tutorial_validation__"}
    failures: list[str] = []

    for index, block in enumerate(PYTHON_BLOCK.findall(source), start=1):
        try:
            compiled = compile(block, f"{path}::block-{index}", "exec")
            if block.lstrip().startswith("# tutorial-validation: syntax-only"):
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compiled, namespace)
        except Exception as error:  # noqa: BLE001 - report tutorial runtime failures together.
            failures.append(f"{path.relative_to(ROOT)} block {index}: {type(error).__name__}: {error}")

    return failures


def main() -> int:
    pages = [COURSE_INDEX, *sorted(COURSE_DIR.glob("*.md"))]
    failures = [failure for page in pages for failure in validate_file(page)]

    if failures:
        print("数据科学教程 Python 代码校验失败：")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1

    blocks = [block for page in pages for block in PYTHON_BLOCK.findall(page.read_text(encoding="utf-8"))]
    syntax_only_count = sum(block.lstrip().startswith("# tutorial-validation: syntax-only") for block in blocks)
    print(
        f"已检查 {len(pages)} 个页面中的 {len(blocks)} 个 Python 代码块："
        f"运行 {len(blocks) - syntax_only_count} 个，仅语法检查 {syntax_only_count} 个。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
