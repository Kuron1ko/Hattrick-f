from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "results" / "geant" / "8sp" / "0" / "Hattrick复现与改进.md"
OUTPUT_PDF = ROOT / "output" / "pdf" / "Hattrick复现与改进.pdf"


FONT_REGULAR = Path(r"C:\Windows\Fonts\Deng.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\Dengb.ttf")
FONT_MONO = Path(r"C:\Windows\Fonts\consola.ttf")


def register_fonts() -> tuple[str, str, str]:
    regular_name = "DengXian"
    bold_name = "DengXianBold"
    mono_name = "Consolas"
    pdfmetrics.registerFont(TTFont(regular_name, str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont(bold_name, str(FONT_BOLD)))
    pdfmetrics.registerFont(TTFont(mono_name, str(FONT_MONO)))
    return regular_name, bold_name, mono_name


def display_width(text: str) -> int:
    width = 0
    for ch in text:
        width += 2 if ord(ch) > 127 else 1
    return max(width, 1)


def latex_to_readable(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", expr)
    expr = expr.replace(r"\lambda", "\u03bb")
    expr = expr.replace(r"\times", "\u00d7")
    expr = expr.replace(r"\cdot", "\u00b7")
    expr = re.sub(r"_\{([^{}]+)\}", r"_\1", expr)
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.replace("\\", "")
    expr = re.sub(r"\s+", " ", expr)
    return expr


def inline_markdown(text: str, mono_font: str) -> str:
    placeholders: dict[str, str] = {}

    def stash(value: str) -> str:
        token = f"@@INLINE_TOKEN_{len(placeholders)}@@"
        placeholders[token] = value
        return token

    text = re.sub(
        r"`([^`]+)`",
        lambda m: stash(f'<font name="{mono_font}">{escape(m.group(1))}</font>'),
        text,
    )
    text = re.sub(
        r"\$([^$]+)\$",
        lambda m: stash(f'<font name="{mono_font}">{escape(latex_to_readable(m.group(1)))}</font>'),
        text,
    )
    text = escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    for token, value in placeholders.items():
        text = text.replace(token, value)
    return text


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    table_lines: list[str] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
        table_lines.append(lines[i].strip())
        i += 1

    rows: list[list[str]] = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            continue
        rows.append(cells)
    return rows, i


def make_table(rows: list[list[str]], styles: dict[str, ParagraphStyle], usable_width: float) -> Table:
    col_count = max(len(row) for row in rows)
    normalized = [row + [""] * (col_count - len(row)) for row in rows]

    weights = []
    for col in range(col_count):
        weight = max(display_width(row[col]) for row in normalized)
        weights.append(weight)
    total = sum(weights)
    col_widths = [usable_width * weight / total for weight in weights]

    font_size = 7 if col_count >= 8 else 8.5
    cell_style = ParagraphStyle(
        "table_cell",
        parent=styles["body"],
        fontSize=font_size,
        leading=font_size + 2,
        alignment=TA_CENTER,
        wordWrap="CJK",
    )
    header_style = ParagraphStyle(
        "table_header",
        parent=cell_style,
        fontName=styles["bold"].fontName,
        textColor=colors.HexColor("#111827"),
    )
    data = []
    for row_idx, row in enumerate(normalized):
        style = header_style if row_idx == 0 else cell_style
        data.append([Paragraph(inline_markdown(cell, styles["mono"].fontName), style) for cell in row])

    table = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2F7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def image_flowable(image_path: Path, usable_width: float) -> Image:
    with PILImage.open(image_path) as img:
        width_px, height_px = img.size
    width = min(usable_width, 165 * mm)
    height = width * height_px / width_px
    max_height = 115 * mm
    if height > max_height:
        height = max_height
        width = height * width_px / height_px
    return Image(str(image_path), width=width, height=height, hAlign="CENTER")


def build_story() -> list:
    regular, bold, mono = register_fonts()
    sample = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "title",
            parent=sample["Title"],
            fontName=bold,
            fontSize=22,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.HexColor("#111827"),
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName=bold,
            fontSize=15,
            leading=22,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#0F172A"),
        ),
        "h3": ParagraphStyle(
            "h3",
            fontName=bold,
            fontSize=12,
            leading=18,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#1F2937"),
        ),
        "body": ParagraphStyle(
            "body",
            fontName=regular,
            fontSize=10,
            leading=16,
            alignment=TA_LEFT,
            wordWrap="CJK",
            spaceAfter=5,
            textColor=colors.HexColor("#111827"),
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName=regular,
            fontSize=10,
            leading=16,
            leftIndent=14,
            firstLineIndent=-8,
            wordWrap="CJK",
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "code",
            fontName=mono,
            fontSize=8.5,
            leading=12,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#F8FAFC"),
            borderColor=colors.HexColor("#CBD5E1"),
            borderWidth=0.3,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "bold": ParagraphStyle("bold_holder", fontName=bold),
        "mono": ParagraphStyle("mono_holder", fontName=mono),
    }

    lines = SOURCE_MD.read_text(encoding="utf-8").splitlines()
    story: list = []
    base_dir = SOURCE_MD.parent
    usable_width = A4[0] - 2 * 20 * mm
    in_code = False
    code_lines: list[str] = []
    last_heading_level = 0

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(raw)
            i += 1
            continue

        if not stripped:
            story.append(Spacer(1, 3))
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            rows, next_i = parse_table(lines, i)
            if rows:
                story.append(make_table(rows, styles, usable_width))
                story.append(Spacer(1, 8))
            i = next_i
            continue

        image_match = re.fullmatch(r"!\[[^\]]*\]\((.+)\)", stripped)
        if image_match:
            raw_path = image_match.group(1).strip()
            if raw_path.startswith("<") and raw_path.endswith(">"):
                raw_path = raw_path[1:-1]
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = base_dir / image_path
            if image_path.exists():
                story.append(Spacer(1, 4))
                story.append(image_flowable(image_path, usable_width))
                story.append(Spacer(1, 10))
            else:
                story.append(Paragraph(f"[图片未找到: {escape(raw_path)}]", styles["body"]))
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = inline_markdown(heading_match.group(2), mono)
            if level == 1:
                if story:
                    story.append(PageBreak())
                story.append(Paragraph(text, styles["title"]))
            elif level == 2:
                if last_heading_level == 1:
                    story.append(Spacer(1, 2))
                story.append(Paragraph(text, styles["h2"]))
            else:
                story.append(Paragraph(text, styles["h3"]))
            last_heading_level = level
            i += 1
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)
        numbered_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if bullet_match:
            story.append(Paragraph(inline_markdown(bullet_match.group(1), mono), styles["bullet"], bulletText="-"))
        elif numbered_match:
            story.append(
                Paragraph(
                    inline_markdown(numbered_match.group(2), mono),
                    styles["bullet"],
                    bulletText=f"{numbered_match.group(1)}.",
                )
            )
        else:
            story.append(Paragraph(inline_markdown(stripped, mono), styles["body"]))
        i += 1

    return story


def add_page_number(canvas, doc):
    canvas.saveState()
    regular = "DengXian"
    canvas.setFont(regular, 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    page_text = f"{doc.page}"
    canvas.drawCentredString(A4[0] / 2, 10 * mm, page_text)
    canvas.restoreState()


def main() -> None:
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Hattrick复现与改进",
        author="Codex",
    )
    doc.build(build_story(), onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
