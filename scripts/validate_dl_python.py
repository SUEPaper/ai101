#!/usr/bin/env python3
"""提取并执行深度学习与大模型教材中的 Python 代码块。"""

from __future__ import annotations

import ast
import contextlib
import io
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE_DIR = ROOT / "docs" / "courses" / "deep-learning"
BLOCK = re.compile(r"```python\n(.*?)\n```", flags=re.DOTALL)
SYNTAX_ONLY = "tutorial-validation: syntax-only"


def main() -> None:
    errors: list[str] = []
    executed = 0
    skipped = 0

    for path in sorted(COURSE_DIR.glob("[0-9][0-9]-*.md")):
        namespace: dict[str, object] = {"__name__": "__tutorial__"}
        text = path.read_text(encoding="utf-8")
        for index, code in enumerate(BLOCK.findall(text), start=1):
            label = f"{path.relative_to(ROOT)} 第 {index} 个 Python 代码块"
            try:
                ast.parse(code, filename=label)
            except SyntaxError as exc:
                errors.append(f"{label} 语法错误：{exc}")
                continue

            if SYNTAX_ONLY in code:
                skipped += 1
                continue

            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(code, label, "exec"), namespace)
                executed += 1
            except Exception as exc:  # noqa: BLE001 - 教材检查需要汇总原始异常类型
                errors.append(f"{label} 执行失败：{type(exc).__name__}: {exc}")

    if errors:
        raise SystemExit("深度学习 Python 检查失败：\n- " + "\n- ".join(errors))

    print(f"深度学习 Python 检查通过：执行 {executed} 个代码块，跳过 {skipped} 个环境依赖示例。")


if __name__ == "__main__":
    main()
