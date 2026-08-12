#!/usr/bin/env python3
"""递归解析 ref/ml101 下的 PDF，并生成可校验、可断点续跑的结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream
from reportlab.pdfgen import canvas


PARSER_NAME = "opendataloader-pdf"
PARSER_VERSION = "2.5.0"
PAGE_SEPARATOR = "<!-- local-page %page-number% -->"
LOCAL_PAGE_RE = re.compile(r"<!-- local-page (\d+) -->")
SOURCE_PAGE_RE = re.compile(r"<!-- source-page (\d+) -->")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)")
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

logging.getLogger("pypdf").setLevel(logging.ERROR)


class ParseFailure(RuntimeError):
    """单个页段无法完成解析。"""


@dataclass(frozen=True)
class SourceInfo:
    path: Path
    relative_path: Path
    sha256: str
    page_count: int
    size_bytes: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    sanitized, _ = sanitize_json_strings(data)
    atomic_write_text(path, json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n")


def sanitize_json_strings(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return SURROGATE_RE.subn("�", value)
    if isinstance(value, list):
        sanitized_items = []
        replacement_count = 0
        for item in value:
            sanitized, count = sanitize_json_strings(item)
            sanitized_items.append(sanitized)
            replacement_count += count
        return sanitized_items, replacement_count
    if isinstance(value, dict):
        sanitized_mapping = {}
        replacement_count = 0
        for key, item in value.items():
            sanitized_key, key_count = sanitize_json_strings(key)
            sanitized_item, item_count = sanitize_json_strings(item)
            sanitized_mapping[sanitized_key] = sanitized_item
            replacement_count += key_count + item_count
        return sanitized_mapping, replacement_count
    return value, 0


def replace_in_json_strings(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace_in_json_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            replace_in_json_strings(key, old, new): replace_in_json_strings(item, old, new)
            for key, item in value.items()
        }
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_page_count(pdfinfo: Path, path: Path) -> int:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    result = subprocess.run(
        [str(pdfinfo), str(path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ParseFailure(f"pdfinfo 失败：{detail}")
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        raise ParseFailure("pdfinfo 输出中没有 Pages 字段")
    return int(match.group(1))


def discover_sources(source_root: Path, output_root: Path, pdfinfo: Path) -> list[SourceInfo]:
    output_root_resolved = output_root.resolve()
    paths = []
    for candidate in source_root.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() != ".pdf":
            continue
        try:
            candidate.resolve().relative_to(output_root_resolved)
        except ValueError:
            paths.append(candidate)

    paths.sort(key=lambda path: path.relative_to(source_root).as_posix().casefold())
    sources = []
    for index, path in enumerate(paths, start=1):
        relative_path = path.relative_to(source_root)
        print(f"[inventory {index}/{len(paths)}] {relative_path.as_posix()}", flush=True)
        sources.append(
            SourceInfo(
                path=path,
                relative_path=relative_path,
                sha256=sha256_file(path),
                page_count=pdf_page_count(pdfinfo, path),
                size_bytes=path.stat().st_size,
            )
        )
    return sources


def split_ranges(page_count: int, chunk_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + chunk_size - 1, page_count))
        for start in range(1, page_count + 1, chunk_size)
    ]


def collect_page_numbers(value: Any) -> set[int]:
    pages: set[int] = set()
    if isinstance(value, dict):
        page_number = value.get("page number")
        if isinstance(page_number, int):
            pages.add(page_number)
        for child in value.values():
            pages.update(collect_page_numbers(child))
    elif isinstance(value, list):
        for child in value:
            pages.update(collect_page_numbers(child))
    return pages


def collect_image_sources(value: Any) -> list[str]:
    sources: list[str] = []
    if isinstance(value, dict):
        source = value.get("source")
        if isinstance(source, str) and Path(source).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            sources.append(source)
        for child in value.values():
            sources.extend(collect_image_sources(child))
    elif isinstance(value, list):
        for child in value:
            sources.extend(collect_image_sources(child))
    return sources


def markdown_image_sources(markdown: str) -> list[str]:
    return [(match.group(1) or match.group(2)) for match in MARKDOWN_IMAGE_RE.finditer(markdown)]


def find_single_output(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise ParseFailure(f"期望 1 个 {suffix} 文件，实际得到 {len(matches)} 个")
    return matches[0]


def run_parser(command: list[str], timeout_seconds: int) -> tuple[str, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise ParseFailure(f"超过 {timeout_seconds} 秒未完成")
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        process.communicate()
        raise
    if process.returncode != 0:
        detail = (stderr.strip() or stdout.strip() or "没有错误输出")[-2000:]
        raise ParseFailure(f"解析器退出码 {process.returncode}：{detail}")
    return stdout, stderr


def validate_chunk_files(
    chunks_dir: Path,
    start: int,
    end: int,
    source_page_count: int,
    page_numbering: str = "source",
) -> dict[str, Any]:
    chunk_id = f"p{start:04d}-p{end:04d}"
    markdown_path = chunks_dir / f"{chunk_id}.md"
    json_path = chunks_dir / f"{chunk_id}.json"
    if not markdown_path.is_file() or not json_path.is_file():
        raise ParseFailure(f"{chunk_id} 缺少 Markdown 或 JSON")

    markdown = markdown_path.read_text(encoding="utf-8")
    markers = [int(number) for number in LOCAL_PAGE_RE.findall(markdown)]
    page_span = end - start + 1
    if page_numbering == "local":
        expected_pages = list(range(1, page_span + 1))
        expected_json_page_count = page_span
        minimum_element_page = 1
        maximum_element_page = page_span
    elif page_numbering == "source":
        expected_pages = list(range(start, end + 1))
        expected_json_page_count = source_page_count
        minimum_element_page = start
        maximum_element_page = end
    else:
        raise ParseFailure(f"{chunk_id} 使用未知页码模式：{page_numbering}")
    if markers != expected_pages:
        raise ParseFailure(f"{chunk_id} 页码标记不连续：{markers[:10]}")

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ParseFailure(f"{chunk_id} JSON 无法反序列化：{exc}") from exc
    if not isinstance(data, dict):
        raise ParseFailure(f"{chunk_id} JSON 顶层不是对象")
    if data.get("number of pages") != expected_json_page_count:
        raise ParseFailure(
            f"{chunk_id} JSON 总页数为 {data.get('number of pages')}，"
            f"期望 {expected_json_page_count}"
        )
    element_pages = collect_page_numbers(data.get("kids", []))
    if any(
        page < minimum_element_page or page > maximum_element_page
        for page in element_pages
    ):
        raise ParseFailure(f"{chunk_id} JSON 含页段范围外元素")

    local_image_sources = markdown_image_sources(markdown) + collect_image_sources(data)
    missing_images = [
        source
        for source in local_image_sources
        if not source.startswith(("http://", "https://", "data:"))
        and not (chunks_dir / source).is_file()
    ]
    if missing_images:
        raise ParseFailure(f"{chunk_id} 含失效图片引用：{missing_images[0]}")

    image_directory = chunks_dir / f"{chunk_id}_images"
    images = sorted(path for path in image_directory.rglob("*") if path.is_file()) if image_directory.is_dir() else []
    return {
        "source_page_start": start,
        "source_page_end": end,
        "markdown": f"chunks/{chunk_id}.md",
        "json": f"chunks/{chunk_id}.json",
        "image_directory": f"chunks/{chunk_id}_images" if images else None,
        "image_count": len(images),
        "element_count": len(data.get("kids", [])) if isinstance(data.get("kids"), list) else 0,
        "invalid_surrogate_replacement_count": 0,
        "page_numbering": page_numbering,
        "input_adjustment": None,
    }


def write_physical_chunk(source: Path, destination: Path, start: int, end: int) -> None:
    reader = PdfReader(source, strict=False)
    writer = PdfWriter()
    for page_index in range(start - 1, end):
        writer.add_page(reader.pages[page_index])
    with destination.open("wb") as handle:
        writer.write(handle)


def write_flattened_page(source: Path, destination: Path, page_number: int) -> None:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise ParseFailure("找不到 pdftoppm，无法压平复杂单页")

    raster_prefix = destination.with_suffix("")
    result = subprocess.run(
        [
            pdftoppm,
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            "72",
            "-png",
            "-singlefile",
            str(source),
            str(raster_prefix),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ParseFailure(f"pdftoppm 压平失败：{detail}")
    raster_path = raster_prefix.with_suffix(".png")
    if not raster_path.is_file():
        raise ParseFailure("pdftoppm 没有生成单页图像")

    reader = PdfReader(source, strict=False)
    overlay_writer = PdfWriter()
    overlay_writer.add_page(reader.pages[page_number - 1])
    original_page = overlay_writer.pages[0]
    filtered_content = ContentStream(original_page.get_contents(), overlay_writer)
    filtered_operations = []
    inside_text = False
    for operands, operator in filtered_content.operations:
        if operator == b"BT":
            inside_text = True
            filtered_operations.append((operands, operator))
        elif inside_text:
            filtered_operations.append((operands, operator))
            if operator == b"ET":
                inside_text = False
        elif operator in {b"q", b"Q", b"cm"}:
            filtered_operations.append((operands, operator))
    filtered_content.operations = filtered_operations
    original_page.replace_contents(filtered_content)

    width = float(original_page.mediabox.width)
    height = float(original_page.mediabox.height)
    background_buffer = BytesIO()
    background_canvas = canvas.Canvas(background_buffer, pagesize=(width, height))
    background_canvas.drawImage(
        str(raster_path),
        0,
        0,
        width=width,
        height=height,
        preserveAspectRatio=False,
        mask="auto",
    )
    background_canvas.showPage()
    background_canvas.save()
    background_buffer.seek(0)

    background_page = PdfReader(background_buffer).pages[0]
    background_page.merge_page(original_page)
    writer = PdfWriter()
    writer.add_page(background_page)
    with destination.open("wb") as handle:
        writer.write(handle)


def parse_range(
    parser: Path,
    source: SourceInfo,
    chunks_dir: Path,
    temp_root: Path,
    start: int,
    end: int,
    threads: int,
    timeout_seconds: int,
    physical_split: bool,
    flatten_page: bool = False,
) -> dict[str, Any]:
    chunk_id = f"p{start:04d}-p{end:04d}"
    page_numbering = "local" if physical_split else "source"
    try:
        return validate_chunk_files(
            chunks_dir,
            start,
            end,
            source.page_count,
            page_numbering,
        )
    except ParseFailure:
        pass

    with tempfile.TemporaryDirectory(prefix=f"{chunk_id}-", dir=temp_root) as temporary:
        temporary_path = Path(temporary)
        parser_input = source.path
        if flatten_page:
            if start != end:
                raise ParseFailure("压平降级仅支持单页")
            parser_input = temporary_path / "input.pdf"
            write_flattened_page(source.path, parser_input, start)
        elif physical_split:
            parser_input = temporary_path / "input.pdf"
            write_physical_chunk(source.path, parser_input, start, end)
        command = [
            str(parser),
            str(parser_input),
            "-o",
            str(temporary_path),
            "-f",
            "markdown,json",
        ]
        if not physical_split:
            command.extend(["--pages", f"{start}-{end}"])
        command.extend(
            [
                "--markdown-page-separator",
                PAGE_SEPARATOR,
                "--image-output",
                "external",
                "--threads",
                str(threads),
                "--quiet",
            ]
        )
        run_parser(command, timeout_seconds)
        source_markdown = find_single_output(temporary_path, ".md")
        source_json = find_single_output(temporary_path, ".json")
        markdown = source_markdown.read_text(encoding="utf-8")
        data = json.loads(source_json.read_text(encoding="utf-8"))
        data, invalid_surrogate_count = sanitize_json_strings(data)

        old_image_directories = sorted(
            path for path in temporary_path.iterdir() if path.is_dir() and path.name.endswith("_images")
        )
        if len(old_image_directories) > 1:
            raise ParseFailure(f"{chunk_id} 出现多个图片目录")
        if old_image_directories:
            old_image_directory = old_image_directories[0]
            new_image_name = f"{chunk_id}_images"
            markdown = markdown.replace(old_image_directory.name, new_image_name)
            data = replace_in_json_strings(data, old_image_directory.name, new_image_name)
            destination_images = chunks_dir / new_image_name
            if destination_images.exists():
                shutil.rmtree(destination_images)
            shutil.move(str(old_image_directory), str(destination_images))

        atomic_write_text(chunks_dir / f"{chunk_id}.md", markdown)
        atomic_write_json(chunks_dir / f"{chunk_id}.json", data)

    metadata = validate_chunk_files(
        chunks_dir,
        start,
        end,
        source.page_count,
        page_numbering,
    )
    metadata["invalid_surrogate_replacement_count"] = invalid_surrogate_count
    if flatten_page:
        metadata["input_adjustment"] = {
            "type": "rasterized-visuals-with-filtered-text-layer",
            "render_dpi": 72,
            "source_pdf_modified": False,
        }
    elif physical_split:
        metadata["input_adjustment"] = {
            "type": "physical-page-range-split",
            "source_pdf_modified": False,
        }
    return metadata


def progress_is_compatible(progress: dict[str, Any], source: SourceInfo) -> bool:
    return (
        progress.get("source_sha256") == source.sha256
        and progress.get("source_page_count") == source.page_count
        and progress.get("parser_version") == PARSER_VERSION
    )


def save_progress(progress_path: Path, source: SourceInfo, chunks: list[dict[str, Any]], fallbacks: list[dict[str, Any]]) -> None:
    atomic_write_json(
        progress_path,
        {
            "source_sha256": source.sha256,
            "source_page_count": source.page_count,
            "parser_version": PARSER_VERSION,
            "updated_at": utc_now(),
            "chunks": sorted(chunks, key=lambda item: item["source_page_start"]),
            "fallbacks": fallbacks,
        },
    )


def covered_pages(chunks: list[dict[str, Any]]) -> set[int]:
    pages: set[int] = set()
    for chunk in chunks:
        pages.update(range(chunk["source_page_start"], chunk["source_page_end"] + 1))
    return pages


def parse_range_with_fallback(
    parser: Path,
    source: SourceInfo,
    chunks_dir: Path,
    temp_root: Path,
    start: int,
    end: int,
    threads: int,
    timeout_seconds: int,
    chunks: list[dict[str, Any]],
    fallbacks: list[dict[str, Any]],
    progress_path: Path,
    physical_split: bool = False,
) -> None:
    if set(range(start, end + 1)).issubset(covered_pages(chunks)):
        return
    single_page_failure_count = sum(
        1
        for fallback in fallbacks
        if fallback.get("source_page_start") == fallback.get("source_page_end")
    )
    if physical_split and start == end and single_page_failure_count >= 2:
        print(f"  [flatten] page {start} (document fallback)", flush=True)
        chunk = parse_range(
            parser,
            source,
            chunks_dir,
            temp_root,
            start,
            end,
            threads,
            min(timeout_seconds, 60),
            True,
            True,
        )
        chunks[:] = [
            existing
            for existing in chunks
            if existing["source_page_end"] < start
            or existing["source_page_start"] > end
        ]
        chunks.append(chunk)
        save_progress(progress_path, source, chunks, fallbacks)
        return
    previous_failure = any(
        fallback.get("source_page_start") == start
        and fallback.get("source_page_end") == end
        for fallback in fallbacks
    )
    try_current_range = not previous_failure
    if previous_failure:
        print(f"  [resume retry] pages {start}-{end}", flush=True)
    try:
        if not try_current_range:
            raise ParseFailure("沿用上次失败记录，直接缩小分块")
        print(f"  [parse] pages {start}-{end}", flush=True)
        effective_timeout = (
            min(timeout_seconds, 60)
            if physical_split and start == end
            else timeout_seconds
        )
        chunk = parse_range(
            parser,
            source,
            chunks_dir,
            temp_root,
            start,
            end,
            threads,
            effective_timeout,
            physical_split,
        )
        chunks[:] = [
            existing
            for existing in chunks
            if existing["source_page_end"] < start or existing["source_page_start"] > end
        ]
        chunks.append(chunk)
        save_progress(progress_path, source, chunks, fallbacks)
        return
    except (ParseFailure, UnicodeError, json.JSONDecodeError) as exc:
        reason = str(exc)
        if try_current_range:
            fallbacks.append(
                {"source_page_start": start, "source_page_end": end, "reason": reason}
            )
        save_progress(progress_path, source, chunks, fallbacks)
        if try_current_range:
            print(f"  [retry] pages {start}-{end}: {reason}", flush=True)

    page_span = end - start + 1
    if page_span > 10:
        child_ranges = [
            (child_start, min(child_start + 9, end))
            for child_start in range(start, end + 1, 10)
        ]
    elif page_span > 1:
        child_ranges = [(page, page) for page in range(start, end + 1)]
    else:
        print(f"  [flatten] page {start}", flush=True)
        try:
            chunk = parse_range(
                parser,
                source,
                chunks_dir,
                temp_root,
                start,
                end,
                threads,
                min(timeout_seconds, 60),
                True,
                True,
            )
        except (ParseFailure, UnicodeError, json.JSONDecodeError) as exc:
            raise ParseFailure(f"第 {start} 页压平后仍然失败：{exc}") from exc
        chunks[:] = [
            existing
            for existing in chunks
            if existing["source_page_end"] < start
            or existing["source_page_start"] > end
        ]
        chunks.append(chunk)
        save_progress(progress_path, source, chunks, fallbacks)
        return

    for child_start, child_end in child_ranges:
        parse_range_with_fallback(
            parser,
            source,
            chunks_dir,
            temp_root,
            child_start,
            child_end,
            threads,
            timeout_seconds,
            chunks,
            fallbacks,
            progress_path,
            True,
        )


def validate_document(output_directory: Path, source: SourceInfo) -> dict[str, Any]:
    manifest_path = output_directory / "manifest.json"
    if not manifest_path.is_file():
        raise ParseFailure("缺少 manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ParseFailure("文档清单状态不是 complete")
    if manifest.get("source_sha256") != source.sha256:
        raise ParseFailure("源文件 SHA-256 已变化")
    if manifest.get("source_page_count") != source.page_count:
        raise ParseFailure("源文件页数已变化")

    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ParseFailure("文档清单没有分块")
    validated_chunks = [
        validate_chunk_files(
            output_directory / "chunks",
            chunk["source_page_start"],
            chunk["source_page_end"],
            source.page_count,
            chunk.get("page_numbering", "source"),
        )
        for chunk in chunks
    ]
    ordered_pages = [
        page
        for chunk in sorted(validated_chunks, key=lambda item: item["source_page_start"])
        for page in range(chunk["source_page_start"], chunk["source_page_end"] + 1)
    ]
    if ordered_pages != list(range(1, source.page_count + 1)):
        raise ParseFailure("分块源页码不连续或重复")

    merged_path = output_directory / "document.md"
    if not merged_path.is_file():
        raise ParseFailure("缺少合并 Markdown")
    merged = merged_path.read_text(encoding="utf-8")
    merged_pages = [int(number) for number in SOURCE_PAGE_RE.findall(merged)]
    if merged_pages != list(range(1, source.page_count + 1)):
        raise ParseFailure("合并 Markdown 的源页码不连续或重复")
    missing_images = [
        image
        for image in markdown_image_sources(merged)
        if not image.startswith(("http://", "https://", "data:"))
        and not (output_directory / image).is_file()
    ]
    if missing_images:
        raise ParseFailure(f"合并 Markdown 含失效图片引用：{missing_images[0]}")

    return manifest


def parse_document(
    parser: Path,
    source_root: Path,
    output_root: Path,
    temp_root: Path,
    source: SourceInfo,
    chunk_size: int,
    threads: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    output_directory = output_root / source.relative_path.with_suffix("")
    try:
        return validate_document(output_directory, source)
    except (ParseFailure, json.JSONDecodeError, OSError):
        pass

    progress_path = output_directory / ".progress.json"
    chunks_dir = output_directory / "chunks"
    progress: dict[str, Any] = {}
    if progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            progress = {}
    if output_directory.exists() and not progress_is_compatible(progress, source):
        shutil.rmtree(output_directory)
        progress = {}
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks = progress.get("chunks", []) if isinstance(progress.get("chunks"), list) else []
    fallbacks = progress.get("fallbacks", []) if isinstance(progress.get("fallbacks"), list) else []

    valid_chunks = []
    for chunk in chunks:
        try:
            validated_chunk = validate_chunk_files(
                chunks_dir,
                chunk["source_page_start"],
                chunk["source_page_end"],
                source.page_count,
                chunk.get("page_numbering", "source"),
            )
            validated_chunk["invalid_surrogate_replacement_count"] = chunk.get(
                "invalid_surrogate_replacement_count", 0
            )
            validated_chunk["input_adjustment"] = chunk.get("input_adjustment")
            valid_chunks.append(validated_chunk)
        except (KeyError, ParseFailure, UnicodeError, json.JSONDecodeError, OSError):
            continue
    chunks = valid_chunks
    save_progress(progress_path, source, chunks, fallbacks)

    for start, end in split_ranges(source.page_count, chunk_size):
        ranges = [(start, end)]
        physical_split = False
        single_page_failure_count = sum(
            1
            for fallback in fallbacks
            if fallback.get("source_page_start") == fallback.get("source_page_end")
        )
        if single_page_failure_count >= 2:
            ranges = [(page, page) for page in range(start, end + 1)]
            physical_split = True
        elif fallbacks:
            ranges = [
                (child_start, min(child_start + 9, end))
                for child_start in range(start, end + 1, 10)
            ]
            physical_split = True
        for child_start, child_end in ranges:
            parse_range_with_fallback(
                parser,
                source,
                chunks_dir,
                temp_root,
                child_start,
                child_end,
                threads,
                timeout_seconds,
                chunks,
                fallbacks,
                progress_path,
                physical_split,
            )

    chunks.sort(key=lambda item: item["source_page_start"])
    pages = [
        page
        for chunk in chunks
        for page in range(chunk["source_page_start"], chunk["source_page_end"] + 1)
    ]
    if pages != list(range(1, source.page_count + 1)):
        raise ParseFailure("完成解析后分块页码仍不连续")

    merged_parts = []
    for chunk in chunks:
        chunk_markdown = output_directory / chunk["markdown"]
        markdown = chunk_markdown.read_text(encoding="utf-8")
        image_directory = chunk.get("image_directory")
        if image_directory:
            markdown = markdown.replace(
                Path(image_directory).name,
                Path(image_directory).as_posix(),
            )
        if chunk.get("page_numbering", "source") == "local":
            page_offset = chunk["source_page_start"] - 1
            markdown = LOCAL_PAGE_RE.sub(
                lambda match: f"<!-- source-page {int(match.group(1)) + page_offset} -->",
                markdown,
            )
        else:
            markdown = markdown.replace("<!-- local-page ", "<!-- source-page ")
        merged_parts.append(markdown.strip())
    atomic_write_text(output_directory / "document.md", "\n\n".join(merged_parts) + "\n")

    manifest = {
        "status": "complete",
        "source_pdf": display_path(source_root / source.relative_path),
        "source_relative_path": source.relative_path.as_posix(),
        "source_sha256": source.sha256,
        "source_size_bytes": source.size_bytes,
        "source_page_count": source.page_count,
        "parser": PARSER_NAME,
        "parser_version": PARSER_VERSION,
        "parser_mode": "local",
        "formats": ["markdown", "json"],
        "image_output": "external",
        "merged_markdown": "document.md",
        "chunk_size": chunk_size,
        "chunk_timeout_seconds": timeout_seconds,
        "thread_count": threads,
        "chunk_count": len(chunks),
        "image_count": sum(chunk["image_count"] for chunk in chunks),
        "element_count": sum(chunk["element_count"] for chunk in chunks),
        "invalid_surrogate_replacement_count": sum(
            chunk.get("invalid_surrogate_replacement_count", 0) for chunk in chunks
        ),
        "fallbacks": fallbacks,
        "completed_at": utc_now(),
        "chunks": chunks,
    }
    atomic_write_json(output_directory / "manifest.json", manifest)
    validate_document(output_directory, source)
    progress_path.unlink(missing_ok=True)
    return manifest


def write_corpus_manifest(
    source_root: Path,
    output_root: Path,
    sources: list[SourceInfo],
    completed: list[tuple[SourceInfo, dict[str, Any]]],
    failures: list[dict[str, str]],
) -> None:
    documents = []
    for source, manifest in completed:
        document_output = output_root / source.relative_path.with_suffix("")
        documents.append(
            {
                "source_pdf": manifest["source_pdf"],
                "source_sha256": source.sha256,
                "source_page_count": source.page_count,
                "output_directory": display_path(document_output),
                "manifest": display_path(document_output / "manifest.json"),
                "merged_markdown": display_path(document_output / "document.md"),
                "chunk_count": manifest["chunk_count"],
                "image_count": manifest["image_count"],
                "element_count": manifest["element_count"],
                "invalid_surrogate_replacement_count": manifest.get(
                    "invalid_surrogate_replacement_count", 0
                ),
            }
        )
    atomic_write_json(
        output_root / "manifest.json",
        {
            "status": "complete" if not failures and len(completed) == len(sources) else "incomplete",
            "source_root": display_path(source_root),
            "output_root": display_path(output_root),
            "parser": PARSER_NAME,
            "parser_version": PARSER_VERSION,
            "parser_mode": "local",
            "generated_at": utc_now(),
            "document_count_expected": len(sources),
            "document_count_complete": len(completed),
            "source_page_count": sum(source.page_count for source in sources),
            "source_size_bytes": sum(source.size_bytes for source in sources),
            "chunk_count": sum(manifest["chunk_count"] for _, manifest in completed),
            "image_count": sum(manifest["image_count"] for _, manifest in completed),
            "element_count": sum(manifest["element_count"] for _, manifest in completed),
            "invalid_surrogate_replacement_count": sum(
                manifest.get("invalid_surrogate_replacement_count", 0)
                for _, manifest in completed
            ),
            "documents": documents,
            "failures": failures,
        },
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("ref/ml101"))
    parser.add_argument("--output-root", type=Path, default=Path("ref/ml101/opendataloader-pdf/corpus"))
    parser.add_argument("--parser", type=Path, required=True, help="opendataloader-pdf 2.5.0 可执行文件")
    parser.add_argument("--pdfinfo", type=Path, default=Path(shutil.which("pdfinfo") or "pdfinfo"))
    parser.add_argument("--temp-root", type=Path, default=Path("/private/tmp/ml101-opendataloader-work"))
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180, help="单个页段超时秒数")
    parser.add_argument("--match", action="append", default=[], help="只处理指定的源相对路径，可重复")
    parser.add_argument("--limit", type=int, help="只处理排序后的前 N 份文档")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    source_root = arguments.source_root.resolve()
    output_root = arguments.output_root.resolve()
    parser = arguments.parser.resolve()
    pdfinfo = arguments.pdfinfo.resolve()
    temp_root = arguments.temp_root.resolve()

    if not source_root.is_dir():
        print(f"源目录不存在：{source_root}", file=sys.stderr)
        return 2
    if not parser.is_file():
        print(f"解析器不存在：{parser}", file=sys.stderr)
        return 2
    if not pdfinfo.is_file():
        print(f"pdfinfo 不存在：{pdfinfo}", file=sys.stderr)
        return 2
    if arguments.chunk_size < 1 or arguments.threads < 1 or arguments.timeout < 1:
        print("chunk-size、threads 和 timeout 必须是正整数", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(source_root, output_root, pdfinfo)
    if arguments.match:
        requested = set(arguments.match)
        sources = [source for source in sources if source.relative_path.as_posix() in requested]
        found = {source.relative_path.as_posix() for source in sources}
        missing = sorted(requested - found)
        if missing:
            print(f"未找到指定 PDF：{', '.join(missing)}", file=sys.stderr)
            return 2
    if arguments.limit is not None:
        sources = sources[: arguments.limit]
    if not sources:
        print("没有找到待处理 PDF", file=sys.stderr)
        return 2

    completed: list[tuple[SourceInfo, dict[str, Any]]] = []
    failures: list[dict[str, str]] = []
    for index, source in enumerate(sources, start=1):
        print(
            f"[document {index}/{len(sources)}] {source.relative_path.as_posix()} "
            f"({source.page_count} pages)",
            flush=True,
        )
        try:
            output_directory = output_root / source.relative_path.with_suffix("")
            if arguments.verify_only:
                manifest = validate_document(output_directory, source)
            else:
                manifest = parse_document(
                    parser,
                    source_root,
                    output_root,
                    temp_root,
                    source,
                    arguments.chunk_size,
                    arguments.threads,
                    arguments.timeout,
                )
            completed.append((source, manifest))
            print(f"  [complete] {manifest['chunk_count']} chunks, {manifest['image_count']} images", flush=True)
        except (ParseFailure, UnicodeError, json.JSONDecodeError, OSError) as exc:
            failure = {"source_pdf": source.relative_path.as_posix(), "error": str(exc)}
            failures.append(failure)
            print(f"  [failed] {exc}", file=sys.stderr, flush=True)

    write_corpus_manifest(source_root, output_root, sources, completed, failures)
    print(
        f"[summary] complete={len(completed)} failed={len(failures)} "
        f"pages={sum(source.page_count for source in sources)}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
