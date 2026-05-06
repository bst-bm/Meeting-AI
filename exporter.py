"""Export meeting minutes to text file and PDF."""

from datetime import datetime
from pathlib import Path
from fpdf import FPDF


# ── Text export ───────────────────────────────────────────────────────────────

def to_text(minutes: dict, source_file: str) -> str:
    sep  = "=" * 64
    thin = "-" * 64
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        sep,
        "  MEETING MINUTES",
        sep,
        f"  Generated : {now}",
        f"  Source    : {Path(source_file).name}",
        f"  Title     : {minutes.get('title', 'N/A')}",
        "",
        "SUMMARY",
        thin,
        minutes.get("summary", ""),
        "",
        "PARTICIPANTS",
        thin,
    ]
    for p in minutes.get("participants", []):
        lines.append(f"  • {p}")

    lines += ["", "TOPICS DISCUSSED", thin]
    for t in minutes.get("topics", []):
        lines.append(f"  ▸ {t.get('title', '')}")
        lines.append(f"    {t.get('summary', '')}")

    lines += ["", "ACTION POINTS", thin]
    for i, ap in enumerate(minutes.get("action_points", []), 1):
        owner    = ap.get("owner") or "TBD"
        deadline = ap.get("deadline") or "—"
        lines.append(f"  {i}. {ap.get('action', '')}")
        lines.append(f"     Owner: {owner}   Deadline: {deadline}")

    lines += ["", "DECISIONS", thin]
    for d in minutes.get("decisions", []):
        lines.append(f"  ✓ {d}")

    if minutes.get("next_meeting"):
        lines += ["", "NEXT MEETING", thin, f"  {minutes['next_meeting']}"]

    lines += ["", sep]
    return "\n".join(lines)


def save_text(minutes: dict, source_file: str, output_path: str) -> str:
    content = to_text(minutes, source_file)
    Path(output_path).write_text(content, encoding="utf-8")
    return output_path


# ── PDF export ────────────────────────────────────────────────────────────────

class _PDF(FPDF):
    def __init__(self, source_file: str):
        super().__init__()
        self._source = Path(source_file).name
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(20, 20, 20)

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(40, 40, 40)
        self.cell(0, 10, "Meeting Minutes", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 8, f"Generated {ts}  |  {self._source}  |  Page {self.page_no()}", align="C")

    def section(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(235, 237, 255)
        self.set_text_color(50, 50, 120)
        self.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(40, 40, 40)
        self.ln(2)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bullet(self, text: str, indent: int = 8):
        self.set_font("Helvetica", "", 10)
        self.set_x(20 + indent)
        self.multi_cell(0, 6, f"- {text}")

    def action(self, num: int, action: str, owner: str, deadline: str):
        self.set_font("Helvetica", "B", 10)
        self.set_x(20)
        self.cell(6, 6, f"{num}.")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, action)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100)
        self.set_x(26)
        self.cell(0, 5, f"Owner: {owner}   Deadline: {deadline}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(40, 40, 40)
        self.ln(1)


def save_pdf(minutes: dict, source_file: str, output_path: str) -> str:
    pdf = _PDF(source_file)
    pdf.add_page()

    # Meta
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, minutes.get("title", "Untitled Meeting"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120)
    pdf.cell(0, 6, datetime.now().strftime("%A, %B %d %Y  %H:%M"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(40, 40, 40)
    pdf.ln(2)

    # Summary
    pdf.section("Summary")
    pdf.body(minutes.get("summary", ""))

    # Participants
    pdf.section("Participants")
    for p in minutes.get("participants", []):
        pdf.bullet(p)

    # Topics
    if minutes.get("topics"):
        pdf.section("Topics Discussed")
        for t in minutes["topics"]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, t.get("title", ""), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(28)
            pdf.multi_cell(0, 6, t.get("summary", ""))
            pdf.ln(1)

    # Action Points
    if minutes.get("action_points"):
        pdf.section("Action Points")
        for i, ap in enumerate(minutes["action_points"], 1):
            pdf.action(
                i,
                ap.get("action", ""),
                ap.get("owner") or "TBD",
                ap.get("deadline") or "n/a",
            )

    # Decisions
    if minutes.get("decisions"):
        pdf.section("Decisions")
        for d in minutes["decisions"]:
            pdf.bullet(d)

    # Next meeting
    if minutes.get("next_meeting"):
        pdf.section("Next Meeting")
        pdf.body(minutes["next_meeting"])

    pdf.output(output_path)
    return output_path
