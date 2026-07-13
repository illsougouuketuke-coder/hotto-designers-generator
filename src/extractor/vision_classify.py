"""
Claude Vision による抽出画像の自動分類。

PDF から抽出された画像 (数枚〜20枚) を「1回の API 呼び出し」でまとめて分類し、
ポスターの4スロット (hero / bottom_1 / bottom_2 / floor_plan) に割り当てる。

従来の白率ヒューリスティック (pdf_images.classify_for_poster) では
「洗面所がヒーローに来る」等の事故が起きるため、Vision で
category + hero_score (LINE配信メイン写真としての魅力度 0-100) を判定する。

コスト配慮: 各画像は各辺最大 512px に縮小した JPEG で送信する。

重要な学び (image_extractor.py と同じ):
    anthropic の messages.parse (構造化出力) は 'Grammar compilation timed out'
    エラーを起こすため使用禁止。
    messages.create + プロンプトでJSON要求 + 正規表現でJSON抽出 のパターンを使う。

API 失敗時はこのモジュールは例外を投げる。呼び出し側で
pdf_images.classify_for_poster (ヒューリスティック) にフォールバックすること。
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import anthropic
from PIL import Image

from src.extractor.pdf_images import ExtractedImage

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2000

# 送信前の縮小サイズ (各辺最大)
MAX_SIDE = 512
JPEG_QUALITY = 80

VALID_CATEGORIES = {
    "interior",      # 室内 (リビング・居室)
    "exterior",      # 外観
    "kitchen",       # キッチン
    "bathroom",      # 浴室・洗面・トイレ
    "floor_plan",    # 間取り図
    "map",           # 地図
    "logo_or_text",  # ロゴ・文字・バナー
    "other",         # その他
}

# ヒーロー写真に使ってよいカテゴリ
HERO_CATEGORIES = ("interior", "exterior", "kitchen")
# 下部写真に使ってよいカテゴリ (第一候補)
BOTTOM_CATEGORIES = ("interior", "exterior", "kitchen", "bathroom")
# 絶対にスロットに選ばないカテゴリ
EXCLUDED_CATEGORIES = ("map", "logo_or_text")


SYSTEM_PROMPT = """あなたは不動産の物件写真を分類する専門家です。

複数の画像が「画像1」「画像2」... と番号付きで与えられます。
各画像を分類し、以下のJSONフォーマットでのみ回答してください。
余計な前置きや説明文は一切書かず、純粋なJSONのみを出力します。

# JSON出力フォーマット

```json
{
  "classifications": [
    {"index": 1, "category": "interior", "hero_score": 85},
    {"index": 2, "category": "floor_plan", "hero_score": 0}
  ]
}
```

# category の選択肢 (必ずこの8つのいずれか)

- "interior": 室内写真 (リビング・居室・洋室など)
- "exterior": 建物の外観写真 (エントランス・共用部含む)
- "kitchen": キッチンの写真
- "bathroom": 浴室・洗面所・トイレの写真
- "floor_plan": 間取り図 (部屋の配置を示す線画の図面)
- "map": 地図 (周辺地図・案内図)
- "logo_or_text": ロゴ・文字だけの画像・キャンペーンバナー・帯・QRコード
- "other": 上記に当てはまらないもの

# hero_score (0-100 の整数)

LINE配信のメイン写真 (ヒーロー写真) としての魅力度。
- 高スコア: 明るいリビング、窓からの光が入る構図、部屋が広く見える構図、
  スタイリッシュな外観・キッチン
- 低スコア: 暗い写真、狭く見える構図、洗面所・トイレ、設備のアップ
- floor_plan / map / logo_or_text は必ず 0

# 重要なルール
- 与えられた全画像について、番号の欠けなく1件ずつ分類する
- category は上記8つ以外の文字列を使わない
- 回答は ```json ... ``` のコードブロック1つのみ、それ以外は一切書かない
"""


def _encode_image_resized(image_path: Path) -> tuple[str, str]:
    """画像を各辺最大 MAX_SIDE px に縮小し JPEG で base64 エンコード。

    Returns:
        (media_type, base64_data)
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return "image/jpeg", b64


def _extract_json(raw_text: str) -> dict:
    """AIレスポンスから JSON オブジェクトを抽出してパースする。"""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if not json_match:
        raise ValueError(
            f"AIレスポンスからJSONを抽出できませんでした。応答: {raw_text[:300]}"
        )
    try:
        return json.loads(json_match.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"AIレスポンスのJSONパースに失敗: {e}。応答: {raw_text[:300]}"
        )


def _assign_slots(
    images: list[ExtractedImage],
    labels: list[tuple[str, int]],
) -> dict[str, ExtractedImage | None]:
    """category + hero_score から4スロットへ割り当てる (Python側ロジック)。

    - hero: hero_score 最高の interior/exterior/kitchen
    - bottom_1/bottom_2: 次点2枚 (hero・間取り図と重複しない)
    - floor_plan: floor_plan カテゴリの最大サイズのもの
    - map / logo_or_text は絶対に選ばない
    """
    def area(im: ExtractedImage) -> int:
        return im.width * im.height

    indexed = list(zip(images, labels))

    # 間取り図: floor_plan カテゴリの最大面積
    plans = [im for im, (cat, _) in indexed if cat == "floor_plan"]
    floor_plan = max(plans, key=area) if plans else None

    # ヒーロー候補: hero_score 降順 → 面積降順
    hero_pool = [
        (im, score) for im, (cat, score) in indexed if cat in HERO_CATEGORIES
    ]
    hero_pool.sort(key=lambda t: (-t[1], -area(t[0])))
    hero = hero_pool[0][0] if hero_pool else None

    # 下部候補: 写真系カテゴリから hero を除いた次点
    bottom_pool = [
        (im, score)
        for im, (cat, score) in indexed
        if cat in BOTTOM_CATEGORIES and im is not hero
    ]
    bottom_pool.sort(key=lambda t: (-t[1], -area(t[0])))
    # 足りなければ "other" で補充 (map / logo_or_text は絶対に入れない)
    if len(bottom_pool) < 2:
        others = [
            (im, score)
            for im, (cat, score) in indexed
            if cat == "other" and im is not hero
        ]
        others.sort(key=lambda t: (-t[1], -area(t[0])))
        bottom_pool.extend(others)

    bottom_1 = bottom_pool[0][0] if len(bottom_pool) >= 1 else None
    bottom_2 = bottom_pool[1][0] if len(bottom_pool) >= 2 else None

    return {
        "hero": hero,
        "bottom_1": bottom_1,
        "bottom_2": bottom_2,
        "floor_plan": floor_plan,
    }


def classify_images_with_vision(
    images: list[ExtractedImage],
    api_key: str,
) -> dict[str, ExtractedImage | None]:
    """抽出画像を Claude Vision で分類し、4スロットに割り当てて返す。

    1回の API 呼び出しで全画像 (番号付き・縮小済み) をまとめて送る。

    Args:
        images: pdf_images.extract_images 等で得た抽出画像リスト
        api_key: Anthropic API キー (必須)

    Returns:
        {"hero": ..., "bottom_1": ..., "bottom_2": ..., "floor_plan": ...}
        (該当なしのスロットは None)

    Raises:
        RuntimeError: API キー未設定
        anthropic.APIError / ValueError: API 呼び出し・JSON抽出の失敗。
            呼び出し側で classify_for_poster にフォールバックすること。
    """
    if not images:
        return {"hero": None, "bottom_1": None, "bottom_2": None, "floor_plan": None}
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY が設定されていません。"
            "Vision分類には API キーが必要です。"
        )

    client = anthropic.Anthropic(api_key=api_key)

    content: list[dict] = []
    for i, im in enumerate(images, start=1):
        media_type, b64 = _encode_image_resized(im.path)
        content.append({"type": "text", "text": f"画像{i}"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                f"以上 {len(images)} 枚の画像をすべて分類し、"
                "指定のJSONフォーマットでのみ回答してください。"
            ),
        }
    )

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
        messages=[{"role": "user", "content": content}],
    )

    raw_text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            raw_text += block.text

    parsed = _extract_json(raw_text)
    items = parsed.get("classifications", [])
    if not isinstance(items, list):
        raise ValueError(f"classifications がリストではありません: {type(items)}")

    # index -> (category, hero_score)。欠けは ("other", 0) 扱いで事故らせない。
    by_index: dict[int, tuple[str, int]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index", 0))
        except (TypeError, ValueError):
            continue
        cat = str(item.get("category", "other")).strip().lower()
        if cat not in VALID_CATEGORIES:
            cat = "other"
        try:
            score = int(item.get("hero_score", 0))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))
        if cat in EXCLUDED_CATEGORIES or cat == "floor_plan":
            score = 0
        by_index[idx] = (cat, score)

    labels = [by_index.get(i, ("other", 0)) for i in range(1, len(images) + 1)]
    return _assign_slots(images, labels)


if __name__ == "__main__":
    import os
    import sys

    from src.extractor.pdf_images import _white_ratio

    if len(sys.argv) < 2:
        print("Usage: python -m src.extractor.vision_classify <image1> [image2 ...]")
        sys.exit(1)

    imgs = []
    for p in sys.argv[1:]:
        path = Path(p)
        with Image.open(path) as im:
            w, h = im.size
        imgs.append(
            ExtractedImage(
                path=path, width=w, height=h,
                file_size=path.stat().st_size,
                white_ratio=_white_ratio(path),
                is_likely_floor_plan=False,
                is_likely_photo=True,
            )
        )
    result = classify_images_with_vision(
        imgs, api_key=os.environ.get("ANTHROPIC_API_KEY", "")
    )
    for slot, im in result.items():
        print(f"{slot}: {im.path.name if im else 'NONE'}")
