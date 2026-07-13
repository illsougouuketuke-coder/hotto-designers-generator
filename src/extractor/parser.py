"""
マイソク PDF のテキストを構造化 PropertyData に変換する。

対応フォーマット:
  - RealNetPro 形式 (CREST TAPP 各物件で確認)
  - エビス・リビング形式 (Log浅草橋 等で確認)
  - その他: フィールド名+正規表現による best-effort 抽出

SPEC §8.1 に従い、抽出した数値は一切手を加えない(PDFの記載をそのまま渡す)。
抽出失敗・想定外の値は warnings に積んで、UI 側で人間に確認させる。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.extractor.pdf_text import extract_text


# 既知のフィールドラベル(2フォーマット混在)。スペース表記のゆらぎ込み。
KNOWN_LABELS: set[str] = {
    # RealNetPro
    "物件名", "号室名", "所在地", "交通", "建築構造", "間取タイプ",
    "専有面積", "開口部方位", "築年", "現況/入居時期", "賃料",
    "共益費・管理費", "敷金", "礼金", "保証金", "償却・敷引", "更新料",
    "契約期間", "町内会費", "駐車場", "備 考", "備考", "設 備", "設備",
    "条　件", "条件", "取引態様", "手数料", "分配", "特記事項", "物件種目",
    # エビス・リビング
    "最寄駅", "間取", "賃貸条件", "建物", "建物名", "規模構造",
    "間取内訳", "築年月", "管理費", "現況", "入居日",
}


@dataclass
class PropertyData:
    name: str = ""
    room: str = ""
    floor: int | None = None
    is_corner: bool = False
    address_full: str = ""
    address_short: str = ""
    station: str = ""
    walk_minutes: int | None = None
    layout: str = ""
    area_sqm: float | None = None
    built_year: int | None = None
    built_month: int | None = None
    rent: int | None = None
    common_fee: int | None = None
    deposit: int = 0
    key_money: int = 0
    pet_allowed: bool = False
    warnings: list[str] = field(default_factory=list)


# ===== 正規化・小道具 =====

_FULLWIDTH_TABLE = str.maketrans({
    **{chr(0xFF10 + i): str(i) for i in range(10)},        # ０-９ → 0-9
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)}, # Ａ-Ｚ → A-Z
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)}, # ａ-ｚ → a-z
    "，": ",", "：": ":", "．": ".", "－": "-", "～": "~",
    "（": "(", "）": ")", "「": "「", "」": "」",  # bracket は維持
    "　": " ",
    # カタカナ間取り全角→半角(よく出るもの)
    "Ｋ": "K", "Ｄ": "D", "Ｌ": "L", "Ｓ": "S", "Ｒ": "R",
})


def _normalize(text: str) -> str:
    """全角→半角の正規化。比較・正規表現を半角ベースで揃える。"""
    return text.translate(_FULLWIDTH_TABLE)


def _extract_int(text: str) -> int | None:
    """'68,200 円' -> 68200, 'なし' -> 0, '' -> None"""
    if not text:
        return None
    if "なし" in text or "無し" in text:
        return 0
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None


def _extract_float(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"(\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def _shorten_address(address: str) -> str:
    """'愛知県名古屋市中川区八熊1丁目5-10' -> '名古屋市中川区八熊'"""
    if not address:
        return ""
    address = re.sub(r"^[^市区町村]*?[県府都道]", "", address).strip()
    address = re.split(r"[0-9]|丁目|番地", address, maxsplit=1)[0]
    return address.strip()


def _parse_field_block(text: str) -> dict[str, str]:
    """ラベル単独行 + 次行以降を値とみなす構造を dict 化。"""
    lines = text.split("\n")
    result: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped in KNOWN_LABELS:
            current_key = stripped.replace(" ", "").replace("　", "")
            result.setdefault(current_key, [])
        elif current_key is not None:
            result[current_key].append(line)
    return {k: "\n".join(v).strip() for k, v in result.items()}


def _join_address_lines(raw: str) -> str:
    """所在地ブロックの複数行を結合。郵便番号(〒)行はスキップ。"""
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    lines = [l for l in lines if not l.startswith("〒") and not re.match(r"\d{3}-\d{4}", l)]
    return "".join(lines)


# ===== メインパーサー =====

def parse_property(pdf_path: Path) -> PropertyData:
    """マイソクPDFをパースして PropertyData を返す。多フォーマット対応。"""
    raw_text = extract_text(pdf_path)
    text = _normalize(raw_text)
    # 改行で分断されたラベル(エビス・リビング形式の「賃貸\n条件」等)を結合
    text = text.replace("賃貸\n条件", "賃貸条件")
    fields = _parse_field_block(text)
    data = PropertyData()

    # 物件名: 物件名 -> 建物名 の順で探す
    data.name = fields.get("物件名", "").strip()
    if not data.name:
        bldg = fields.get("建物名", "").strip()
        if bldg:
            # "Log浅草橋 (ログアサクサバシ)" → 括弧前のみ
            data.name = re.split(r"[(（]", bldg, maxsplit=1)[0].strip()

    # 号室名: '1301（13階部分）' (RealNetPro) または タイトル文字列内 (Ebisu)
    room_text = fields.get("号室名", "").strip()
    if room_text:
        parts = re.split(r"[(（]", room_text, maxsplit=1)
        data.room = parts[0].strip()
        if len(parts) > 1:
            m = re.search(r"(\d+)\s*階", parts[1])
            if m:
                data.floor = int(m.group(1))
    else:
        # 文書内に "Ｌｏｇ浅草橋802号室" のようなパターンがあれば抽出
        m = re.search(r"(\d{2,4})\s*号室", text)
        if m:
            data.room = m.group(1)

    # 階数: 取得できていなければ 号室//100 で推定
    if data.floor is None and data.room.isdigit():
        candidate = int(data.room) // 100
        if 1 <= candidate <= 99:
            data.floor = candidate

    # 所在地: 全行を結合(〒は除外)
    address = _join_address_lines(fields.get("所在地", ""))
    data.address_full = address
    data.address_short = _shorten_address(address)

    # 交通: 「駅」括り(RealNetPro) または 素の「○○駅 徒歩N分」(Ebisu)
    transit = fields.get("交通", "")
    m = re.search(r"「([^」]+)」\s*徒歩\s*(\d+)\s*分", transit)
    if not m:
        m = re.search(r"(\S+?)駅\s*徒歩\s*(\d+)\s*分", transit)
    if m:
        data.station = m.group(1) + "駅"
        data.walk_minutes = int(m.group(2))
    # 最寄駅フィールドからもフォールバック
    if not data.station:
        nearest = fields.get("最寄駅", "").strip()
        if nearest:
            data.station = nearest if nearest.endswith("駅") else nearest + "駅"
    if data.walk_minutes is None:
        m = re.search(r"徒歩\s*(\d+)\s*分", text)
        if m:
            data.walk_minutes = int(m.group(1))

    # 間取り: 値内に余計な文字が混入する場合があるので、パターンだけ抽出する
    layout_raw = (
        fields.get("間取タイプ", "").strip()
        or fields.get("間取", "").strip()
    )
    m = re.search(r"(\d+\s*[KDLSR]+)", layout_raw)
    data.layout = re.sub(r"\s+", "", m.group(1)) if m else layout_raw

    # 専有面積
    data.area_sqm = _extract_float(fields.get("専有面積", ""))

    # 築年: '2025年08月' (RealNetPro) または 築年月 '2024年2月' (Ebisu)
    built = fields.get("築年", "").strip() or fields.get("築年月", "").strip()
    m = re.match(r"(\d{4})年\s*(\d{1,2})月", built)
    if m:
        data.built_year = int(m.group(1))
        data.built_month = int(m.group(2))

    # 賃料: 単独フィールド or 賃貸条件ブロックの中
    rent_field = fields.get("賃料", "")
    if not rent_field:
        rent_field = fields.get("賃貸条件", "")
    if rent_field:
        m = re.search(r"賃料\s*[:：]?\s*([\d,]+)\s*円", rent_field)
        if m:
            data.rent = _extract_int(m.group(1))
        elif "円" in rent_field:
            data.rent = _extract_int(rent_field)
    if data.rent is None:
        # 全文走査の最後の砦
        m = re.search(r"賃料\s*[:：]\s*([\d,]+)\s*円", text)
        if m:
            data.rent = _extract_int(m.group(1))

    # 共益費・管理費: 専用ラベル → 賃貸条件 → 全文 の順
    common = fields.get("共益費・管理費", "") or fields.get("管理費", "")
    if common:
        data.common_fee = _extract_int(common)
    elif fields.get("賃貸条件", ""):
        m = re.search(r"管理費\s*[:：]?\s*([\d,]+)\s*円", fields["賃貸条件"])
        if m:
            data.common_fee = _extract_int(m.group(1))
    if data.common_fee is None:
        m = re.search(r"管理費\s*[:：]\s*([\d,]+)\s*円", text)
        if m:
            data.common_fee = _extract_int(m.group(1))

    # 敷金・礼金 (RealNetPro はフィールドがある、Ebisu は "0ヶ月" 表記)
    data.deposit = _extract_int(fields.get("敷金", "")) or 0
    data.key_money = _extract_int(fields.get("礼金", "")) or 0

    # ペット可否: 全文を走査 (Ebisu は本文に "ペット飼育可能" と書く)
    data.pet_allowed = any(
        kw in text for kw in ("ペット相談", "ペット可", "ペット飼育可", "ペット飼育")
    )

    # 角部屋判定: 設備セクション + 全文(Ebisu は明記しない)
    setsubi = fields.get("設備", "")
    data.is_corner = "角部屋" in setsubi or "角部屋" in text

    # 警告チェック (SPEC §6.1, §6.2)
    if data.rent is None or not (1000 <= data.rent <= 500000):
        data.warnings.append(f"賃料が想定範囲外または抽出失敗: {data.rent}")
    if data.common_fee is None:
        data.warnings.append("共益費・管理費が抽出できませんでした")
    if not data.name:
        data.warnings.append("物件名が空です")
    if not data.room:
        data.warnings.append("号室が空です")
    if data.area_sqm is None:
        data.warnings.append("専有面積が抽出できませんでした")
    if not data.station:
        data.warnings.append("最寄り駅が抽出できませんでした")

    return data


if __name__ == "__main__":
    import json
    import sys
    from dataclasses import asdict

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/Users/matchan/Downloads/CREST TAPP金山_1301_20260423192046.pdf"
    )
    data = parse_property(path)
    print(json.dumps(asdict(data), ensure_ascii=False, indent=2))
