"""
PDF からプレーンテキストを抽出する薄いラッパー。
RealNetPro 形式のマイソク PDF を想定。
"""

from pathlib import Path

import fitz


def extract_text(pdf_path: Path) -> str:
    """全ページを連結したテキストを返す。"""
    doc = fitz.open(pdf_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()
