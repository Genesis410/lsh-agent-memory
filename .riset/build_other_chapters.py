"""Compiler for Bab 1, Bab 2, Bab 3, and Bab 5 into formatted Microsoft Word (.docx) documents.
Uses standard academic margins, Times New Roman 12pt, 1.5 spacing, and clean headers.
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def convert_txt_to_docx(txt_path: Path, docx_path: Path):
    if not txt_path.exists():
        print(f"[SKIP] {txt_path} does not exist.")
        return

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = docx.Document()

    # Standard academic page setup (A4, 1.2 inch left, 1.0 inch others)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.2)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)

    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(12)
    style_normal.font.color.rgb = RGBColor(0, 0, 0)
    style_normal.paragraph_format.line_spacing = 1.5
    style_normal.paragraph_format.space_after = Pt(6)

    lines = text.split("\n")
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue

        if line_s.startswith("# BAB") or line_s.startswith("BAB "):
            clean_t = line_s.lstrip("#").strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(clean_t)
            run.bold = True
            run.font.size = Pt(14)
            p.paragraph_format.space_after = Pt(12)
        elif line_s.startswith("## ") or (len(line_s) >= 4 and line_s[:2].isdigit() and line_s[2] == "." and line_s[3] == " "):
            clean_t = line_s.lstrip("#").strip()
            p = doc.add_paragraph()
            run = p.add_run(clean_t)
            run.bold = True
            run.font.size = Pt(13)
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
        elif line_s.startswith("### ") or (len(line_s) >= 6 and line_s[0].isdigit() and line_s[1] == "." and line_s[3] == "." and line_s[5] == " "):
            clean_t = line_s.lstrip("#").strip()
            p = doc.add_paragraph()
            run = p.add_run(clean_t)
            run.bold = True
            run.font.size = Pt(12)
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
        else:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.first_line_indent = Inches(0.4)
            p.add_run(line_s)

    try:
        doc.save(str(docx_path))
        print(f"[OK] Successfully built {docx_path}")
    except PermissionError:
        alt_path = docx_path.parent / f"{docx_path.stem}_revisi.docx"
        doc.save(str(alt_path))
        print(f"[OK] File was open in Word; saved to {alt_path}")

def main():
    print("=================================================================")
    print("COMPILING DOCX FOR BAB 1, BAB 3, AND BAB 5")
    print("=================================================================")
    base_dir = Path(__file__).resolve().parent
    paper_dir = base_dir / "Paper"
    convert_txt_to_docx(paper_dir / "bab1.txt", paper_dir / "bab1.docx")
    convert_txt_to_docx(paper_dir / "bab3.txt", paper_dir / "bab3.docx")
    convert_txt_to_docx(paper_dir / "bab5.txt", paper_dir / "bab5.docx")
    print("[ALL DONE] All chapter documents compiled.")

if __name__ == "__main__":
    main()
