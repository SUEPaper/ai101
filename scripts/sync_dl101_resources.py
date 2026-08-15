#!/usr/bin/env python3
"""同步深度学习与大模型课程的本地参考资料，并生成可审计清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader


USER_AGENT = "AI101-course-resource-sync/1.0 (+https://suepaper.github.io/ai101/)"
DEFAULT_CATALOG = Path("scripts/data/dl101-resources.json")
DEFAULT_OUTPUT_ROOT = Path("ref/dl101")
logging.getLogger("pypdf").setLevel(logging.ERROR)


class ResourceFailure(RuntimeError):
    """资料发现、下载或校验失败。"""


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


@dataclass(frozen=True)
class Resource:
    resource_id: str
    kind: str
    collection: str
    title: str
    source_url: str
    relative_path: str
    level: str | None = None
    module: str | None = None
    module_title: str | None = None


@dataclass(frozen=True)
class VerifiedPdf:
    resource_id: str
    kind: str
    collection: str
    title: str
    source_url: str
    resolved_url: str
    relative_path: str
    size_bytes: int
    sha256: str
    page_count: int
    sample_text_characters: int
    content_type: str | None
    etag: str | None
    last_modified: str | None
    level: str | None
    module: str | None
    module_title: str | None
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--refresh", action="store_true", help="重新下载已存在的文件")
    parser.add_argument("--verify-only", action="store_true", help="只校验现有 manifest 中的文件")
    parser.add_argument("--skip-slides", action="store_true")
    parser.add_argument("--skip-papers", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_catalog(path: Path) -> dict[str, Any]:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceFailure(f"无法读取资料目录 {path}: {exc}") from exc
    if catalog.get("version") != 1:
        raise ResourceFailure("资料目录版本不是 1")
    if not isinstance(catalog.get("slide_sources"), list) or not isinstance(catalog.get("papers"), list):
        raise ResourceFailure("资料目录缺少 slide_sources 或 papers")
    return catalog


def fetch_bytes(url: str, *, attempts: int = 4) -> tuple[bytes, dict[str, str | None]]:
    last_error: Exception | None = None
    request_url = quote(url, safe=":/?&=#%")
    for attempt in range(1, attempts + 1):
        request = Request(
            request_url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urlopen(request, timeout=90) as response:
                payload = response.read()
                metadata = {
                    "resolved_url": response.geturl(),
                    "content_type": response.headers.get_content_type(),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
                return payload, metadata
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts:
                break
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else attempt * 3.0
            time.sleep(delay)
        except (TimeoutError, URLError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(attempt * 2.0)
    raise ResourceFailure(f"下载失败 {url}: {last_error}")


def safe_filename(url: str) -> str:
    name = unquote(Path(urlparse(url).path).name)
    if not name.lower().endswith(".pdf"):
        name = f"{name or 'resource'}.pdf"
    name = re.sub(r"\s+", "-", name.strip())
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-_")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name


def fetch_html(url: str) -> str:
    payload, _ = fetch_bytes(url)
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def discover_slides(source: dict[str, Any]) -> tuple[list[Resource], dict[str, Any]]:
    page_url = source["page_url"]
    parser = LinkParser()
    parser.feed(fetch_html(page_url))
    pattern = re.compile(source["link_pattern"], re.IGNORECASE)
    link_base_url = source.get("link_base_url", page_url)
    urls = sorted({urljoin(link_base_url, href) for href in parser.links})
    matched = [url for url in urls if pattern.search(url)]
    if not matched and not source.get("allow_empty", False):
        raise ResourceFailure(f"{source['id']} 没有发现符合规则的 slides")

    resources: list[Resource] = []
    used_names: set[str] = set()
    for index, url in enumerate(matched, start=1):
        filename = safe_filename(url)
        if filename in used_names:
            stem = Path(filename).stem
            filename = f"{stem}-{index:02d}.pdf"
        used_names.add(filename)
        resources.append(
            Resource(
                resource_id=f"{source['id']}:{Path(filename).stem}",
                kind="slide",
                collection=source["id"],
                title=Path(filename).stem,
                source_url=url,
                relative_path=str(Path(source["output_dir"]) / filename),
            )
        )
    snapshot = {
        "id": source["id"],
        "course": source["course"],
        "offering": source["offering"],
        "page_url": page_url,
        "discovered_count": len(resources),
        "allow_empty": bool(source.get("allow_empty", False)),
        "note": source.get("note"),
        "status": "published" if resources else "not_yet_published",
    }
    return resources, snapshot


def catalog_papers(catalog: dict[str, Any]) -> list[Resource]:
    resources: list[Resource] = []
    ids: set[str] = set()
    for paper in catalog["papers"]:
        paper_id = paper["id"]
        if paper_id in ids:
            raise ResourceFailure(f"论文 id 重复: {paper_id}")
        ids.add(paper_id)
        resources.append(
            Resource(
                resource_id=f"paper:{paper_id}",
                kind="paper",
                collection=paper["module"],
                title=paper["title"],
                source_url=paper["url"],
                relative_path=str(Path("papers") / paper["module"] / f"{paper_id}.pdf"),
                level=paper["level"],
                module=paper["module"],
                module_title=paper["module_title"],
            )
        )
    return resources


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pdf(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ResourceFailure(f"不是 PDF 文件: {path}")
    try:
        reader = PdfReader(path, strict=False)
        if reader.is_encrypted:
            raise ResourceFailure(f"PDF 已加密: {path}")
        page_count = len(reader.pages)
        if page_count <= 0:
            raise ResourceFailure(f"PDF 没有页面: {path}")
        sample_text = "".join((reader.pages[index].extract_text() or "") for index in range(min(3, page_count)))
    except ResourceFailure:
        raise
    except Exception as exc:
        raise ResourceFailure(f"PDF 无法解析 {path}: {exc}") from exc
    return page_count, len(sample_text.strip())


def download_and_verify(resource: Resource, output_root: Path, refresh: bool) -> VerifiedPdf:
    target = output_root / resource.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, str | None] = {
        "resolved_url": resource.source_url,
        "content_type": None,
        "etag": None,
        "last_modified": None,
    }
    status = "cached"
    if refresh or not target.is_file():
        payload, metadata = fetch_bytes(resource.source_url)
        if not payload.startswith(b"%PDF-"):
            content_type = metadata.get("content_type")
            raise ResourceFailure(
                f"响应不是 PDF {resource.source_url}（content-type={content_type}, bytes={len(payload)}）"
            )
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(target)
        status = "downloaded"
    page_count, sample_text_characters = verify_pdf(target)
    return VerifiedPdf(
        resource_id=resource.resource_id,
        kind=resource.kind,
        collection=resource.collection,
        title=resource.title,
        source_url=resource.source_url,
        resolved_url=metadata["resolved_url"] or resource.source_url,
        relative_path=resource.relative_path,
        size_bytes=target.stat().st_size,
        sha256=sha256_file(target),
        page_count=page_count,
        sample_text_characters=sample_text_characters,
        content_type=metadata["content_type"],
        etag=metadata["etag"],
        last_modified=metadata["last_modified"],
        level=resource.level,
        module=resource.module,
        module_title=resource.module_title,
        status=status,
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_existing(output_root: Path) -> int:
    manifest_path = output_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceFailure(f"无法读取 {manifest_path}: {exc}") from exc
    failures: list[str] = []
    checked = 0
    for item in manifest.get("resources", []):
        path = output_root / item["relative_path"]
        try:
            page_count, _ = verify_pdf(path)
            digest = sha256_file(path)
            if page_count != item["page_count"]:
                raise ResourceFailure(f"页数变化: {page_count} != {item['page_count']}")
            if digest != item["sha256"]:
                raise ResourceFailure("SHA-256 不匹配")
            checked += 1
        except (OSError, ResourceFailure) as exc:
            failures.append(f"{item['relative_path']}: {exc}")
    if failures:
        for failure in failures:
            print(f"[失败] {failure}", file=sys.stderr)
        return 1
    print(f"校验通过：{checked} 份 PDF，页数和 SHA-256 均与 manifest 一致。")
    return 0


def main() -> int:
    args = parse_args()
    if args.jobs < 1 or args.jobs > 8:
        raise ResourceFailure("--jobs 必须在 1–8 之间")
    output_root = args.output_root.resolve()
    if args.verify_only:
        return verify_existing(output_root)

    catalog = read_catalog(args.catalog)
    resources: list[Resource] = []
    source_snapshots: list[dict[str, Any]] = []
    if not args.skip_slides:
        for source in catalog["slide_sources"]:
            discovered, snapshot = discover_slides(source)
            resources.extend(discovered)
            source_snapshots.append(snapshot)
            print(f"发现 {snapshot['id']}: {snapshot['discovered_count']} 份 slides")
    if not args.skip_papers:
        papers = catalog_papers(catalog)
        resources.extend(papers)
        print(f"登记核心论文: {len(papers)} 份")

    relative_paths = [resource.relative_path for resource in resources]
    if len(relative_paths) != len(set(relative_paths)):
        raise ResourceFailure("资料目录产生了重复目标路径")

    completed: list[VerifiedPdf] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(download_and_verify, resource, output_root, args.refresh): resource
            for resource in resources
        }
        for index, future in enumerate(as_completed(futures), start=1):
            resource = futures[future]
            try:
                result = future.result()
                completed.append(result)
                print(
                    f"[{index:03d}/{len(resources):03d}] {result.status}: "
                    f"{result.relative_path}（{result.page_count} 页）"
                )
            except Exception as exc:
                failures.append({"resource_id": resource.resource_id, "error": str(exc)})
                print(f"[{index:03d}/{len(resources):03d}] 失败: {resource.resource_id}: {exc}", file=sys.stderr)

    completed.sort(key=lambda item: item.relative_path)
    manifest = {
        "version": 1,
        "status": "complete" if not failures else "partial",
        "generated_at": utc_now(),
        "catalog": str(args.catalog),
        "catalog_updated_at": catalog.get("updated_at"),
        "public_distribution": False,
        "copyright_note": "第三方 PDF 仅供本地课程研究；公开站点不得直接分发，除非逐项确认许可。",
        "source_snapshots": source_snapshots,
        "summary": {
            "resource_count": len(completed),
            "slide_count": sum(item.kind == "slide" for item in completed),
            "paper_count": sum(item.kind == "paper" for item in completed),
            "page_count": sum(item.page_count for item in completed),
            "size_bytes": sum(item.size_bytes for item in completed),
            "failure_count": len(failures),
        },
        "resources": [asdict(item) for item in completed],
        "failures": failures,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(
        "清单已写入 "
        f"{output_root / 'manifest.json'}：{len(completed)} 份 PDF，"
        f"{manifest['summary']['page_count']} 页，失败 {len(failures)} 份。"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResourceFailure as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
