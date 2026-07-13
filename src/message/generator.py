"""
HOTTO designers — LINE配信文章 (テンプレ単案フォールバック版)。

LINE Messaging APIの500文字制限(UTF-16符号単位、絵文字は2文字)に対応。

仕様:
  - 1吹き出し = 500文字以内厳守
  - 安全マージンを取り、目標450文字以内
  - デザイナーズ物件のため「雰囲気・立地・デザイン」を主軸にする
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PropertyForMessage:
    """配信文章生成用の物件情報"""
    name: str                   # "HOTTO RESIDENCE 広尾"
    room: str                   # "302"
    station: str = ""           # "広尾駅"
    walk_minutes: int = 0       # 4
    layout: str = ""            # "1LDK"
    area_sqm: float = 0.0       # 38.50
    rent: int = 0               # 150000
    street_desc: str = ""       # 街紹介 (AI生成 or 手入力)


def count_line_chars(text: str) -> int:
    """LINE方式(UTF-16符号単位)で文字数をカウント。絵文字は2文字。"""
    return len(text.encode("utf-16-le")) // 2


def format_jpy(amount: int) -> str:
    return f"{amount:,}"


def generate_line_message(
    properties: list[PropertyForMessage],
    street_desc: str = "",
) -> str:
    """
    デザイナーズ物件配信のテンプレ単案を生成 (APIキー無し時のフォールバック)。

    Args:
        properties: 配信する物件 (通常1件、複数可)
        street_desc: 街紹介テキスト (1件配信時のみ使用)

    Returns:
        配信文章 (500文字以内)

    Raises:
        ValueError: 500文字を超えた場合
    """
    n = len(properties)
    circled = ["❶", "❷", "❸", "❹", "❺"]

    if n == 1:
        p = properties[0]
        rent_part = f"家賃 ¥{format_jpy(p.rent)}\n" if p.rent else ""
        loc_part = ""
        if p.station and p.walk_minutes:
            loc_part = f"{p.station} 徒歩{p.walk_minutes}分\n"
        layout_part = ""
        if p.layout and p.area_sqm:
            layout_part = f"{p.layout} / {p.area_sqm}㎡\n"

        area_block = ""
        if street_desc:
            area_block = f"\n— About the area —\n{street_desc}\n"

        text = (
            f"『HOTTO designers 配信』\n"
            f"— Designer's Residence Collection —\n\n"
            f"このたびご紹介するのは\n"
            f"設計と素材にこだわった一邸。\n"
            f"\n"
            f"———————\n"
            f"❶【{p.name} {p.room}】\n"
            f"———————\n"
            f"{layout_part}"
            f"{loc_part}"
            f"{rent_part}"
            f"{area_block}\n"
            f"ご内見・資料請求はこちら\n"
            f"→『内見希望』\n"
            f"→『資料が欲しい』\n"
            f"→『相談だけ』\n"
            f"一言お送りください。\n\n"
            f"※ご紹介終了の場合がございます"
        )
    else:
        lines = []
        for i, p in enumerate(properties):
            if i >= len(circled):
                raise ValueError(f"物件数が多すぎます (最大{len(circled)}件)")
            lines.append(f"{circled[i]}【{p.name} {p.room}】")
        property_list = "\n".join(lines)

        text = (
            f"『HOTTO designers 配信』\n"
            f"— Designer's Residence Collection —\n\n"
            f"今回ご紹介するのは\n"
            f"設計にこだわった{n}邸。\n\n"
            f"———————\n"
            f"{property_list}\n"
            f"———————\n\n"
            f"ご内見・資料請求はこちら\n"
            f"→『内見希望』\n"
            f"→『資料が欲しい』\n"
            f"→『相談だけ』\n"
            f"一言お送りください。\n\n"
            f"※ご紹介終了の場合がございます"
        )

    chars = count_line_chars(text)
    if chars > 500:
        raise ValueError(
            f"配信文章が500文字を超えています ({chars}文字)。"
            "物件数を減らすか、街紹介を短くしてください。"
        )
    return text


if __name__ == "__main__":
    sample = PropertyForMessage(
        name="HOTTO RESIDENCE 広尾",
        room="302",
        station="広尾駅",
        walk_minutes=4,
        layout="1LDK",
        area_sqm=38.5,
        rent=150000,
        street_desc="緑豊かな有栖川宮記念公園を中心に、洗練されたカフェや国際色豊かな店舗が立ち並ぶ街。落ち着いた住宅街でありながら、感性を刺激する空気が漂います。",
    )
    msg = generate_line_message([sample], street_desc=sample.street_desc)
    print(msg)
    print(f"\n--- {count_line_chars(msg)}文字 / 500文字 ---")
