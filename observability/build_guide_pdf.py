#!/usr/bin/env python3
"""Build the OpenTelemetry guide PDF using only Python's standard library."""

from __future__ import annotations

import re
import textwrap
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "opentelemetry-guide.md"
OUTPUT = HERE / "opentelemetry-guide.pdf"
PAGE_W, PAGE_H = 612, 792
LEFT, RIGHT, TOP, BOTTOM = 56, 56, 54, 54
CONTENT_W = PAGE_W - LEFT - RIGHT


def parse_markdown(source: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    paragraph: list[str] = []
    code: list[str] = []
    in_code = False

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(("paragraph", " ".join(line.strip() for line in paragraph)))
            paragraph.clear()

    for line in source.splitlines():
        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                blocks.append(("code", "\n".join(code)))
                code.clear()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        if not line.strip():
            flush_paragraph()
        elif line.startswith("#"):
            flush_paragraph()
            level = len(line) - len(line.lstrip("#"))
            blocks.append((f"h{min(level, 3)}", line[level:].strip()))
        elif line.startswith("- "):
            flush_paragraph()
            blocks.append(("bullet", line[2:].strip()))
        elif line.startswith("|"):
            flush_paragraph()
            if not re.match(r"\|[\s:|\-]+\|?$", line):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                blocks.append(("table", "   •   ".join(cells)))
        else:
            paragraph.append(line)
    flush_paragraph()
    if code:
        blocks.append(("code", "\n".join(code)))
    return blocks


def pdf_string(value: str) -> bytes:
    encoded = value.encode("cp1252", errors="replace")
    return b"(" + encoded.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def text_command(x: float, y: float, text: str, font: str, size: float, color: tuple[float, float, float]) -> bytes:
    value = pdf_string(text)
    return (
        f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg\n"
        f"BT /{font} {size:.1f} Tf 1 0 0 1 {x:.1f} {y:.1f} Tm ".encode()
        + value
        + b" Tj ET\n"
    )


def build_pages(blocks: list[tuple[str, str]]) -> list[bytes]:
    pages: list[bytes] = []
    ops = bytearray()
    y = PAGE_H - TOP

    def start_page() -> None:
        nonlocal ops, y
        ops = bytearray()
        y = PAGE_H - TOP
        ops.extend(b"0.075 0.235 0.380 rg\n")
        ops.extend(f"0 {PAGE_H - 35} {PAGE_W} 35 re f\n".encode())
        ops.extend(text_command(LEFT, PAGE_H - 23, "OpenTelemetry in the RAG Chatbot", "F2", 10, (1, 1, 1)))

    def finish_page() -> None:
        ops.extend(b"0.72 0.77 0.82 RG 0.5 w\n")
        ops.extend(f"{LEFT} 38 m {PAGE_W - RIGHT} 38 l S\n".encode())
        pages.append(bytes(ops))

    def ensure_space(height: float) -> None:
        nonlocal y
        if y - height < BOTTOM + 18:
            finish_page()
            start_page()

    start_page()
    for kind, content in blocks:
        if kind in {"h1", "h2", "h3"}:
            size = {"h1": 21, "h2": 15, "h3": 11.5}[kind]
            leading = size + 5
            lines = textwrap.wrap(content, width=74)
            ensure_space(len(lines) * leading + 13)
            y -= 7 if kind != "h1" else 13
            for line in lines:
                ops.extend(text_command(LEFT, y, line, "F2", size, (0.075, 0.235, 0.380)))
                y -= leading
            y -= 5
        elif kind == "code":
            code_lines: list[str] = []
            for source_line in content.splitlines() or [""]:
                code_lines.extend(textwrap.wrap(source_line, width=105, replace_whitespace=False, drop_whitespace=False) or [""])
            height = len(code_lines) * 11 + 12
            ensure_space(height + 8)
            ops.extend(b"0.94 0.96 0.97 rg\n")
            ops.extend(f"{LEFT - 5} {y - height + 2} {CONTENT_W + 10} {height} re f\n".encode())
            y -= 10
            for line in code_lines:
                ops.extend(text_command(LEFT + 4, y, line, "F3", 8.1, (0.12, 0.17, 0.22)))
                y -= 11
            y -= 10
        else:
            if kind == "bullet":
                prefix, text, width, indent = "• ", content, 92, 12
                font, size, color = "F1", 9.5, (0.12, 0.15, 0.18)
            elif kind == "table":
                prefix, text, width, indent = "", content, 91, 0
                font, size, color = "F1", 8.2, (0.12, 0.15, 0.18)
            else:
                prefix, text, width, indent = "", content, 100, 0
                font, size, color = "F1", 9.5, (0.12, 0.15, 0.18)
            wrapped = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""]
            ensure_space(len(wrapped) * 13 + 7)
            for index, line in enumerate(wrapped):
                shown = (prefix if index == 0 else " " * len(prefix)) + line
                ops.extend(text_command(LEFT + indent, y, shown, font, size, color))
                y -= 13
            y -= 5
    finish_page()
    return pages


def write_pdf(pages: list[bytes], destination: Path) -> None:
    objects: list[bytes] = []
    page_ids = [6 + index * 2 for index in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")
    for index, page_content in enumerate(pages):
        page_id = page_ids[index]
        content_id = page_id + 1
        ops = page_content + text_command(PAGE_W - RIGHT - 45, 22, f"Page {index + 1} of {len(pages)}", "F1", 8, (0.35, 0.40, 0.45))
        compressed = zlib.compress(ops)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        objects.append(f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode() + compressed + b"\nendstream")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R /Info << /Title (OpenTelemetry in the RAG Chatbot) /Author (OpenAI Codex) >> >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    destination.write_bytes(output)


def main() -> None:
    blocks = parse_markdown(SOURCE.read_text(encoding="utf-8"))
    pages = build_pages(blocks)
    write_pdf(pages, OUTPUT)
    print(f"Created {OUTPUT} ({len(pages)} pages, {OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
