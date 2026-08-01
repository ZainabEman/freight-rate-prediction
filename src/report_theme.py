"""Visual system for the technical report.

Holds the palette, paragraph styles, page furniture (headers, footers, page
numbers) and the custom flowables the report is built from: KPI cards, callout
boxes, flow diagrams and the temporal-split diagram.

Typography uses the Type 1 fonts built into every PDF reader (Helvetica for text,
Courier for code). No font file is embedded, so the report cannot silently fall
back to a substitute face on another machine.
"""

from __future__ import annotations

from dataclasses import dataclass

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Flowable, Table, TableStyle

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
# The accent is the teal used by the provided score.py, which visually ties the
# report to the December chart it delivers.
ACCENT = colors.HexColor("#064A56")
ACCENT_MID = colors.HexColor("#3E8896")
ACCENT_SOFT = colors.HexColor("#E4EEF0")
ACCENT_WASH = colors.HexColor("#F2F7F8")

INK = colors.HexColor("#0D1618")
INK_SECONDARY = colors.HexColor("#40565A")
INK_MUTED = colors.HexColor("#6B8085")

RULE = colors.HexColor("#C9D6D9")
RULE_LIGHT = colors.HexColor("#E1E9EA")
BAND = colors.HexColor("#F4F8F8")

GOOD = colors.HexColor("#16794F")
GOOD_BG = colors.HexColor("#E2F3EA")
WARN = colors.HexColor("#8A5A00")
WARN_BG = colors.HexColor("#FBF0D8")
CRIT = colors.HexColor("#B3322F")
CRIT_BG = colors.HexColor("#FBE6E5")

PAGE_W, PAGE_H = A4
MARGIN_L = 20 * mm
MARGIN_R = 18 * mm
MARGIN_T = 22 * mm
MARGIN_B = 20 * mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

DOC_TITLE = "Freight Rate Prediction"
DOC_SUBTITLE = "Machine Learning Engineer Assessment"


def build_styles() -> dict[str, ParagraphStyle]:
    """Return the report's paragraph styles, keyed by role."""
    base = getSampleStyleSheet()
    return {
        # Cover
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=34, leading=39, textColor=ACCENT, alignment=0, spaceAfter=0),
        "cover_sub": ParagraphStyle(
            "cover_sub", parent=base["Normal"], fontName="Helvetica",
            fontSize=13.5, leading=19, textColor=INK_SECONDARY, spaceBefore=8),
        "cover_meta": ParagraphStyle(
            "cover_meta", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, leading=15, textColor=INK_MUTED),
        # Headings
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=17, leading=21, textColor=ACCENT, spaceBefore=0, spaceAfter=9),
        # Visually identical to h1 but excluded from the table of contents, so
        # the Contents page does not list itself.
        "h1_notoc": ParagraphStyle(
            "h1_notoc", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=17, leading=21, textColor=ACCENT, spaceBefore=0, spaceAfter=9),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12.5, leading=16, textColor=ACCENT, spaceBefore=13, spaceAfter=5),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=14, textColor=INK, spaceBefore=10, spaceAfter=4),
        # Body
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.6, leading=14, alignment=TA_JUSTIFY,
            textColor=INK_SECONDARY, spaceAfter=7),
        "lede": ParagraphStyle(
            "lede", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10.8, leading=16, textColor=INK, spaceAfter=10),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9.6, leading=13.8, leftIndent=12, bulletIndent=2,
            textColor=INK_SECONDARY, spaceAfter=3.5),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=8.4, leading=11.5, textColor=INK_MUTED, spaceBefore=5, spaceAfter=12),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.2, leading=11, textColor=INK_SECONDARY),
        "cell_head": ParagraphStyle(
            "cell_head", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.2, leading=11, textColor=colors.white),
        "toc1": ParagraphStyle(
            "toc1", fontName="Helvetica-Bold", fontSize=10, leading=17,
            textColor=INK, spaceBefore=5),
        "toc2": ParagraphStyle(
            "toc2", fontName="Helvetica", fontSize=9.2, leading=14.5,
            textColor=INK_SECONDARY, leftIndent=14),
        "center_note": ParagraphStyle(
            "center_note", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=9, leading=13, textColor=INK_MUTED, alignment=TA_CENTER),
    }


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #

class PageDecorator:
    """Draws the running header, footer and page number on every page.

    Page 1 (the cover) is deliberately left undecorated.
    """

    def __init__(self) -> None:
        self.section = ""

    def set_section(self, name: str) -> None:
        """Record the section name shown in the running header."""
        self.section = name

    def __call__(self, canvas, doc) -> None:
        """ReportLab ``onPage`` hook."""
        page = canvas.getPageNumber()
        if page == 1:
            return

        canvas.saveState()

        # Header: title left, current section right, hairline rule under both.
        canvas.setFont("Helvetica-Bold", 7.6)
        canvas.setFillColor(ACCENT)
        canvas.drawString(MARGIN_L, PAGE_H - MARGIN_T + 8 * mm, DOC_TITLE.upper())
        if self.section:
            canvas.setFont("Helvetica", 7.6)
            canvas.setFillColor(INK_MUTED)
            canvas.drawRightString(
                PAGE_W - MARGIN_R, PAGE_H - MARGIN_T + 8 * mm, self.section)
        canvas.setStrokeColor(RULE_LIGHT)
        canvas.setLineWidth(0.6)
        canvas.line(
            MARGIN_L, PAGE_H - MARGIN_T + 6 * mm,
            PAGE_W - MARGIN_R, PAGE_H - MARGIN_T + 6 * mm)

        # Footer: rule, provenance left, page number right.
        canvas.line(MARGIN_L, MARGIN_B - 6 * mm, PAGE_W - MARGIN_R, MARGIN_B - 6 * mm)
        canvas.setFont("Helvetica", 7.4)
        canvas.setFillColor(INK_MUTED)
        canvas.drawString(
            MARGIN_L, MARGIN_B - 10.5 * mm,
            "Technical Report - generated from measured artifacts")
        canvas.setFont("Helvetica-Bold", 7.8)
        canvas.setFillColor(ACCENT)
        canvas.drawRightString(PAGE_W - MARGIN_R, MARGIN_B - 10.5 * mm, str(page))

        canvas.restoreState()


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def styled_table(
    data: list[list],
    widths: list[float],
    *,
    highlight_rows: tuple[int, ...] = (),
    align_right_from: int = 1,
    font_size: float = 8.2,
) -> Table:
    """Build a table with the report's shared styling.

    Args:
        data: Row data, first row treated as the header.
        widths: Column widths in points.
        highlight_rows: Zero-based *data* row indices to emphasise.
        align_right_from: First column index to right-align.
        font_size: Body font size.

    Returns:
        A styled :class:`~reportlab.platypus.Table`.
    """
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE_LIGHT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("BOX", (0, 0), (-1, -1), 0.7, RULE),
    ]
    for row in highlight_rows:
        index = row + 1
        style += [
            ("BACKGROUND", (0, index), (-1, index), ACCENT_SOFT),
            ("FONTNAME", (0, index), (-1, index), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, index), (-1, index), ACCENT),
        ]
    table.setStyle(TableStyle(style))
    return table


# --------------------------------------------------------------------------- #
# Custom flowables
# --------------------------------------------------------------------------- #

@dataclass
class Kpi:
    """One KPI card."""

    label: str
    value: str
    note: str = ""
    hero: bool = False


class KpiGrid(Flowable):
    """A responsive grid of KPI cards drawn directly on the canvas."""

    def __init__(
        self,
        kpis: list[Kpi],
        *,
        columns: int = 4,
        width: float = CONTENT_W,
        card_h: float = 21 * mm,
        gap: float = 4 * mm,
    ) -> None:
        super().__init__()
        self.kpis = kpis
        self.columns = columns
        self.width = width
        self.card_h = card_h
        self.gap = gap
        self.rows = (len(kpis) + columns - 1) // columns
        self.height = self.rows * card_h + (self.rows - 1) * gap

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        """Report the space this flowable needs."""
        return self.width, self.height

    def _fit(self, text: str, font: str, size: float, limit: float) -> tuple[str, float]:
        """Shrink text until it fits the card, so a long value can never overflow."""
        while size > 6.5 and stringWidth(text, font, size) > limit:
            size -= 0.5
        return text, size

    def draw(self) -> None:
        """Render every card."""
        canvas = self.canv
        card_w = (self.width - (self.columns - 1) * self.gap) / self.columns

        for index, kpi in enumerate(self.kpis):
            col = index % self.columns
            row = index // self.columns
            x = col * (card_w + self.gap)
            y = self.height - (row + 1) * self.card_h - row * self.gap

            canvas.saveState()
            canvas.setFillColor(ACCENT_SOFT if kpi.hero else ACCENT_WASH)
            canvas.setStrokeColor(ACCENT if kpi.hero else RULE)
            canvas.setLineWidth(1.0 if kpi.hero else 0.6)
            canvas.roundRect(x, y, card_w, self.card_h, 2.5, fill=1, stroke=1)

            # Accent rule along the top edge of hero cards only.
            if kpi.hero:
                canvas.setFillColor(ACCENT)
                canvas.rect(x, y + self.card_h - 1.6, card_w, 1.6, fill=1, stroke=0)

            pad = 4.2 * mm
            inner = card_w - 2 * pad

            canvas.setFillColor(INK_MUTED)
            canvas.setFont("Helvetica-Bold", 6.3)
            label, size = self._fit(kpi.label.upper(), "Helvetica-Bold", 6.3, inner)
            canvas.setFont("Helvetica-Bold", size)
            canvas.drawString(x + pad, y + self.card_h - 6.2 * mm, label)

            canvas.setFillColor(ACCENT if kpi.hero else INK)
            value, size = self._fit(kpi.value, "Helvetica-Bold", 15, inner)
            canvas.setFont("Helvetica-Bold", size)
            baseline = y + self.card_h - 12.4 * mm if kpi.note else y + self.card_h - 13.6 * mm
            canvas.drawString(x + pad, baseline, value)

            if kpi.note:
                canvas.setFillColor(INK_MUTED)
                note, size = self._fit(kpi.note, "Helvetica", 6.6, inner)
                canvas.setFont("Helvetica", size)
                canvas.drawString(x + pad, y + 3.4 * mm, note)

            canvas.restoreState()


class Callout(Flowable):
    """A bordered note box with an accent rail and optional title."""

    TONES = {
        "info": (ACCENT, ACCENT_SOFT),
        "good": (GOOD, GOOD_BG),
        "warn": (WARN, WARN_BG),
        "crit": (CRIT, CRIT_BG),
    }

    def __init__(
        self,
        text: str,
        *,
        title: str = "",
        tone: str = "info",
        width: float = CONTENT_W,
        styles: dict | None = None,
    ) -> None:
        super().__init__()
        from reportlab.platypus import Paragraph

        self.width = width
        self.rail, self.background = self.TONES.get(tone, self.TONES["info"])
        self.title = title
        style = ParagraphStyle(
            "callout", fontName="Helvetica", fontSize=9.2, leading=13.2,
            textColor=INK_SECONDARY, alignment=TA_JUSTIFY)
        self.paragraph = Paragraph(text, style)
        self.title_style = ParagraphStyle(
            "callout_title", fontName="Helvetica-Bold", fontSize=9.2, leading=13,
            textColor=self.rail)
        self.title_paragraph = Paragraph(title, self.title_style) if title else None
        self.pad = 4.5 * mm

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        """Measure the box, including its inner paragraphs."""
        inner = self.width - 2 * self.pad - 2.2 * mm
        height = 0.0
        if self.title_paragraph:
            height += self.title_paragraph.wrap(inner, available_height)[1] + 2.5 * mm
        height += self.paragraph.wrap(inner, available_height)[1]
        self.height = height + 2 * self.pad
        return self.width, self.height

    def draw(self) -> None:
        """Render the box."""
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(self.background)
        canvas.setStrokeColor(self.background)
        canvas.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        canvas.setFillColor(self.rail)
        canvas.rect(0, 0, 2.2 * mm, self.height, fill=1, stroke=0)
        canvas.restoreState()

        x = 2.2 * mm + self.pad
        y = self.height - self.pad
        if self.title_paragraph:
            h = self.title_paragraph.height
            self.title_paragraph.drawOn(canvas, x, y - h)
            y -= h + 2.5 * mm
        self.paragraph.drawOn(canvas, x, y - self.paragraph.height)


class FlowDiagram(Flowable):
    """A vertical or horizontal box-and-arrow diagram.

    Each step is ``(title, subtitle, tone)`` where tone selects the accent used
    for the box's left rail: ``io``, ``stateless``, ``fitted`` or ``model``.
    """

    TONES = {
        "io": colors.HexColor("#2A78D6"),
        "stateless": colors.HexColor("#1BAF7A"),
        "fitted": colors.HexColor("#EB6834"),
        "model": ACCENT,
    }

    def __init__(
        self,
        steps: list[tuple[str, str, str]],
        *,
        width: float = CONTENT_W,
        horizontal: bool = False,
        box_h: float = 13 * mm,
        gap: float = 6.5 * mm,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.width = width
        self.horizontal = horizontal
        self.box_h = box_h
        self.gap = gap
        if horizontal:
            self.height = box_h + 6 * mm
        else:
            self.height = len(steps) * box_h + (len(steps) - 1) * gap

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        """Report the space this flowable needs."""
        return self.width, self.height

    def _box(self, x: float, y: float, w: float, h: float, step: tuple[str, str, str]) -> None:
        """Draw one labelled box."""
        title, subtitle, tone = step
        canvas = self.canv
        rail = self.TONES.get(tone, ACCENT)

        canvas.setFillColor(ACCENT_WASH)
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.7)
        canvas.roundRect(x, y, w, h, 2, fill=1, stroke=1)
        canvas.setFillColor(rail)
        canvas.rect(x, y, 1.8 * mm, h, fill=1, stroke=0)

        pad = 3.6 * mm
        inner = w - pad - 2.5 * mm
        canvas.setFillColor(INK)
        size = 8.4
        while size > 6 and stringWidth(title, "Helvetica-Bold", size) > inner:
            size -= 0.3
        canvas.setFont("Helvetica-Bold", size)
        canvas.drawString(x + pad, y + h - 5.2 * mm, title)

        if subtitle:
            canvas.setFillColor(INK_MUTED)
            size = 6.8
            while size > 5 and stringWidth(subtitle, "Helvetica", size) > inner:
                size -= 0.2
            canvas.setFont("Helvetica", size)
            canvas.drawString(x + pad, y + h - 9.2 * mm, subtitle)

    def _arrow_down(self, x: float, y_top: float, y_bottom: float) -> None:
        """Draw a downward connector."""
        canvas = self.canv
        canvas.setStrokeColor(ACCENT_MID)
        canvas.setFillColor(ACCENT_MID)
        canvas.setLineWidth(1.1)
        canvas.line(x, y_top, x, y_bottom + 1.9 * mm)
        path = canvas.beginPath()
        path.moveTo(x, y_bottom)
        path.lineTo(x - 1.5 * mm, y_bottom + 2.2 * mm)
        path.lineTo(x + 1.5 * mm, y_bottom + 2.2 * mm)
        path.close()
        canvas.drawPath(path, fill=1, stroke=0)

    def _arrow_right(self, x_left: float, x_right: float, y: float) -> None:
        """Draw a rightward connector."""
        canvas = self.canv
        canvas.setStrokeColor(ACCENT_MID)
        canvas.setFillColor(ACCENT_MID)
        canvas.setLineWidth(1.1)
        canvas.line(x_left, y, x_right - 1.9 * mm, y)
        path = canvas.beginPath()
        path.moveTo(x_right, y)
        path.lineTo(x_right - 2.2 * mm, y - 1.5 * mm)
        path.lineTo(x_right - 2.2 * mm, y + 1.5 * mm)
        path.close()
        canvas.drawPath(path, fill=1, stroke=0)

    def draw(self) -> None:
        """Render the diagram."""
        if self.horizontal:
            count = len(self.steps)
            box_w = (self.width - (count - 1) * self.gap) / count
            y = 3 * mm
            for index, step in enumerate(self.steps):
                x = index * (box_w + self.gap)
                self._box(x, y, box_w, self.box_h, step)
                if index < count - 1:
                    self._arrow_right(x + box_w, x + box_w + self.gap, y + self.box_h / 2)
            return

        box_w = self.width * 0.66
        x = (self.width - box_w) / 2
        for index, step in enumerate(self.steps):
            y = self.height - (index + 1) * self.box_h - index * self.gap
            self._box(x, y, box_w, self.box_h, step)
            if index < len(self.steps) - 1:
                self._arrow_down(self.width / 2, y, y - self.gap)


class TemporalSplitDiagram(Flowable):
    """A calendar bar showing how the 2025 timeline is partitioned.

    Makes the leakage argument visual: the fitted, holdout and scoring blocks are
    contiguous and non-overlapping, and every arrow points forward in time.
    """

    def __init__(self, *, width: float = CONTENT_W) -> None:
        super().__init__()
        self.width = width
        self.height = 46 * mm

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        """Report the space this flowable needs."""
        return self.width, self.height

    def draw(self) -> None:
        """Render the timeline bar, block labels and the leakage annotation."""
        canvas = self.canv
        # 12 months of 2025 mapped linearly across the full content width.
        month_w = self.width / 12.0
        bar_y = self.height - 22 * mm
        bar_h = 11 * mm

        blocks = [
            (0, 8, "TRAINING", "Jan 1 - Aug 31\n38,477 loads", ACCENT, colors.white),
            (8, 10, "HOLDOUT", "Sep 1 - Oct 31\n9,523 loads", ACCENT_MID, colors.white),
            (10, 12, "SCORING", "Nov 1 - Dec 31\n12,000 loads", colors.HexColor("#EB6834"),
             colors.white),
        ]

        for start, end, label, detail, fill, text_color in blocks:
            x = start * month_w
            w = (end - start) * month_w
            canvas.setFillColor(fill)
            canvas.setStrokeColor(colors.white)
            canvas.setLineWidth(1.2)
            canvas.rect(x, bar_y, w, bar_h, fill=1, stroke=1)

            canvas.setFillColor(text_color)
            canvas.setFont("Helvetica-Bold", 7.6)
            canvas.drawCentredString(x + w / 2, bar_y + bar_h / 2 - 1.0 * mm, label)

            canvas.setFillColor(INK_SECONDARY)
            canvas.setFont("Helvetica", 6.9)
            for offset, line in enumerate(detail.split("\n")):
                canvas.drawCentredString(
                    x + w / 2, bar_y - 4.4 * mm - offset * 3.4 * mm, line)

        # Month ticks along the top edge.
        canvas.setFillColor(INK_MUTED)
        canvas.setFont("Helvetica", 6.2)
        for index, name in enumerate(
                ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]):
            canvas.drawCentredString(index * month_w + month_w / 2, bar_y + bar_h + 2.2 * mm, name)

        # Labelled boundaries.
        canvas.setStrokeColor(INK)
        canvas.setLineWidth(0.8)
        canvas.setDash(2, 2)
        for boundary in (8, 10):
            canvas.line(boundary * month_w, bar_y - 1.5 * mm,
                        boundary * month_w, bar_y + bar_h + 5.5 * mm)
        canvas.setDash()

        # Forward-only annotation beneath the bar.
        arrow_y = 6.5 * mm
        canvas.setStrokeColor(GOOD)
        canvas.setFillColor(GOOD)
        canvas.setLineWidth(1.1)
        canvas.line(2 * mm, arrow_y, self.width - 5 * mm, arrow_y)
        path = canvas.beginPath()
        path.moveTo(self.width - 2 * mm, arrow_y)
        path.lineTo(self.width - 5 * mm, arrow_y - 1.5 * mm)
        path.lineTo(self.width - 5 * mm, arrow_y + 1.5 * mm)
        path.close()
        canvas.drawPath(path, fill=1, stroke=0)

        canvas.setFillColor(GOOD)
        canvas.setFont("Helvetica-Bold", 7.4)
        canvas.drawString(
            2 * mm, arrow_y + 2.6 * mm,
            "Every fit precedes every evaluation - no statistic ever learned from a later block")


class SectionDivider(Flowable):
    """A numbered full-width band that opens a major part of the report."""

    def __init__(self, number: str, title: str, *, width: float = CONTENT_W) -> None:
        super().__init__()
        self.number = number
        self.title = title
        self.width = width
        self.height = 15 * mm

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        """Report the space this flowable needs."""
        return self.width, self.height

    def draw(self) -> None:
        """Render the band."""
        canvas = self.canv
        canvas.setFillColor(ACCENT)
        canvas.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(5 * mm, self.height - 5.6 * mm, f"PART {self.number}")
        canvas.setFont("Helvetica-Bold", 12.5)
        canvas.drawString(5 * mm, self.height - 11.2 * mm, self.title)
