"""
HOTTO designers — 画像(マイソクのスクリーンショット等)から物件情報を AI 抽出するモジュール。

Claude Sonnet 4.6 の Vision を使い、画像内のテキストを認識して
PropertyData の各フィールドに構造化して返す。

PDFが入手できないケース (スクショ画像しか手元にない等) のためのフォールバック経路。
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from src.extractor.parser import PropertyData


MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1500


# Pydantic で構造化出力 (PropertyData と一対一対応)
class PropertyExtraction(BaseModel):
    name: str = Field(default="", description="物件名 (例: HOTTO RESIDENCE 広尾)")
    room: str = Field(default="", description="号室番号 (例: 302、1301)")
    floor: int | None = Field(default=None, description="階数 (整数のみ。13階なら13)")
    is_corner: bool = Field(default=False, description="角部屋かどうか")
    address_short: str = Field(default="", description="住所(市区町村+丁目程度の短縮版)")
    station: str = Field(default="", description="最寄り駅名 (例: 広尾駅)")
    walk_minutes: int | None = Field(default=None, description="駅徒歩分数 (整数)")
    layout: str = Field(default="", description="間取り (例: 1LDK、2DK、ワンルーム)")
    area_sqm: float | None = Field(default=None, description="専有面積(㎡、小数2桁まで)")
    built_year: int | None = Field(default=None, description="築年(西暦4桁)")
    built_month: int | None = Field(default=None, description="築月(1-12)")
    rent: int | None = Field(default=None, description="家賃(円、整数)")
    common_fee: int | None = Field(default=None, description="共益費・管理費(円、整数)")
    pet_allowed: bool = Field(default=False, description="ペット相談可かどうか")


SYSTEM_PROMPT = """あなたは不動産マイソク(物件チラシ)画像から物件情報を読み取る専門家です。

与えられた画像から物件情報を読み取り、以下のJSONフォーマットでのみ回答してください。
余計な前置きや説明文は一切書かず、純粋なJSONのみを出力します。

# JSON出力フォーマット

```json
{
  "name": "物件名 (建物名、文字列。読み取れない場合は空文字)",
  "room": "号室番号 (文字列、例: 302・1301)",
  "floor": 13,
  "is_corner": false,
  "address_short": "住所(市区町村+丁目程度)",
  "station": "最寄り駅名 (「駅」を含めて)",
  "walk_minutes": 4,
  "layout": "間取り (例: 1LDK、2DK)",
  "area_sqm": 38.5,
  "built_year": 2025,
  "built_month": 8,
  "rent": 150000,
  "common_fee": 15000,
  "pet_allowed": false
}
```

# 各フィールドのルール

- name: 物件名/建物名 (例: "HOTTO RESIDENCE 広尾"、"CREST TAPP金山")
- room: 号室番号 (文字列。例: "302"、"1301")
- floor: 階数 (整数。"13階" なら 13。複数階なら最も高い階)
- is_corner: 「角部屋」「角住戸」等の記載があれば true
- address_short: 住所(市区町村+丁目程度。番地は不要)
- station: 最寄り駅名 (「駅」を含める、例: "広尾駅")
- walk_minutes: 徒歩分数 (整数。複数駅あれば最寄り)
- layout: 間取り (例: "1LDK"、"2DK"、"ワンルーム")
- area_sqm: 専有面積 (㎡、小数可)
- built_year: 築年 (西暦4桁。"令和7年" → 2025)
- built_month: 築月 (1-12)
- rent: 家賃 (円、整数。"15万円" → 150000)
- common_fee: 共益費・管理費 (円、整数。なしは 0)
- pet_allowed: ペット相談可なら true

# 重要なルール
- 読み取れない値: 文字列は "" 、数値は null、boolean は false
- 推測や創作は絶対にしない。書かれていない情報は埋めない
- 全角数字は半角に変換
- 賃料と共益費が分かれて書かれている場合、それぞれ正しいフィールドに入れる
- 回答は ```json ... ``` のコードブロック1つのみ、それ以外は一切書かない
"""


def _encode_image(image_path: Path) -> tuple[str, str]:
    """画像ファイルをbase64エンコード。(media_type, base64_data) を返す。"""
    data = image_path.read_bytes()
    b64 = base64.standard_b64encode(data).decode("utf-8")
    suffix = image_path.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/png")
    return media_type, b64


def extract_property_from_image(
    image_path: Path,
    api_key: str | None = None,
) -> PropertyData:
    """
    マイソク画像から物件情報を抽出して PropertyData を返す。

    Args:
        image_path: マイソク画像のパス (.png/.jpg/.jpeg/.webp)
        api_key: Anthropic API キー (省略時は環境変数)

    Returns:
        PropertyData (warnings には抽出できなかったフィールドの注意が入る)
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY が設定されていません。"
            "画像からの物件情報抽出にはAPIキーが必要です。"
            "サイドバーで設定してください。"
        )

    client = anthropic.Anthropic(api_key=api_key)
    media_type, b64 = _encode_image(image_path)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "この物件チラシ画像から各フィールドを抽出し、指定のJSONフォーマットでのみ回答してください。",
                    },
                ],
            }
        ],
    )

    # レスポンスから JSON を抽出
    raw_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw_text += block.text

    # ```json ... ``` または {...} を抽出
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if not json_match:
        raise ValueError(
            f"AIレスポンスからJSONを抽出できませんでした。応答: {raw_text[:300]}"
        )
    try:
        parsed = json.loads(json_match.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"AIレスポンスのJSONパースに失敗: {e}。応答: {raw_text[:300]}"
        )

    extracted = PropertyExtraction.model_validate(parsed)

    # PropertyData に変換
    data = PropertyData(
        name=extracted.name or "",
        room=extracted.room or "",
        floor=extracted.floor,
        is_corner=extracted.is_corner,
        address_short=extracted.address_short or "",
        station=extracted.station or "",
        walk_minutes=extracted.walk_minutes,
        layout=extracted.layout or "",
        area_sqm=extracted.area_sqm,
        built_year=extracted.built_year,
        built_month=extracted.built_month,
        rent=extracted.rent,
        common_fee=extracted.common_fee,
        pet_allowed=extracted.pet_allowed,
        warnings=[],
    )

    # 警告: 重要フィールドが空ならユーザーに知らせる
    if not data.name:
        data.warnings.append("画像から物件名を読み取れませんでした。手入力してください。")
    if not data.room:
        data.warnings.append("画像から号室を読み取れませんでした。手入力してください。")
    if not data.rent:
        data.warnings.append("画像から家賃を読み取れませんでした。手入力してください。")
    if not data.station:
        data.warnings.append("画像から最寄り駅を読み取れませんでした。手入力してください。")
    data.warnings.append(
        "※ AIによる画像読み取り結果です。全項目を目視で確認・修正してください。"
    )

    return data


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.extractor.image_extractor <image_path>")
        sys.exit(1)
    result = extract_property_from_image(Path(sys.argv[1]))
    print(result)
