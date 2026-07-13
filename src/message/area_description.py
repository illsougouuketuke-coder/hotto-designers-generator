"""
HOTTO designers — 駅名から街紹介テキストを Claude API で生成するモジュール。

仕様:
  - 駅名と (任意で) 物件の住所概要を渡すと、雰囲気重視の街紹介文を1〜3案返す
  - 文字数は 80〜140字程度を目標 (配信画像とLINE文章の両方で使えるサイズ)
  - 個別店舗名や数字は出さず、街の空気感・雰囲気・暮らしの距離感を語る
  - 4.6 のキャッシュを使ってシステムプロンプトを永続化
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal

import anthropic
from pydantic import BaseModel, Field


MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1200


class AreaVariant(BaseModel):
    label: Literal["short", "balanced", "poetic"]
    text: str


class AreaSet(BaseModel):
    variants: list[AreaVariant] = Field(min_length=3, max_length=3)


SYSTEM_PROMPT = """あなたは「HOTTO designers」の街紹介ライターです。
駅名を与えられたら、その駅周辺の街の雰囲気・空気感を伝える短い紹介文を
3つの長さ・トーンで書き分けます。

# 目的
- HOTTO designers (デザイナーズ物件の配信LINE・配信画像) で使う街紹介
- 読者は感性に敏感で、都市的な暮らしを志向する層
- 街そのものではなく「そこに暮らす感覚」を喚起する文章

# 必ず守るルール
- 具体的な店舗名・施設名・人物名・数字は出さない (例: 「スターバックスがある」「○○分」など NG)
- 過剰な観光案内・グルメガイド調にしない
- 形容詞を盛らない。短く、削ぎ落とした文章
- 街の有名なランドマーク (公園・川など普遍的なもの) は事実として書いて構わない
- 過度な誇張 ("最高峰" "唯一無二" など) は避ける

# 3つのバリエーション

- **short**: 60〜90字。1〜2文。最小限の言葉で街の核を伝える
- **balanced**: 90〜140字。2〜3文。街の表情・暮らしの距離感まで含める
- **poetic**: 100〜150字。3〜4文の短文を重ねて、詩的・余韻のある書き方

# 出力形式

```json
{
  "variants": [
    {"label": "short", "text": "..."},
    {"label": "balanced", "text": "..."},
    {"label": "poetic", "text": "..."}
  ]
}
```

---

# 参考例

## 例1: 駅名「広尾駅」

### short
緑豊かな住宅街と、国際色豊かな文化が静かに同居する街。落ち着いた空気の中に、感性を刺激する気配があります。

### balanced
緑豊かな有栖川宮記念公園を中心に、洗練されたカフェや国際色豊かな店舗が立ち並ぶ街。落ち着いた住宅街でありながら、感性を刺激する空気が漂います。

### poetic
公園の緑と、夜の静かな灯り。
朝の散歩と、昼下がりのカフェ。
住宅街の落ち着きの中に
都市の余白が広がっている、そんな街です。

---

## 例2: 駅名「中目黒駅」

### short
目黒川沿いの桜並木と、独立系ショップが集まる街。クリエイティブが日常に溶け込む空気が流れています。

### balanced
目黒川沿いの桜並木と、独立系のショップやレストランが点在する街。表通りから一本入ると、静かで落ち着いた住宅街が広がります。日常と文化の距離が近い場所です。

### poetic
桜と、川と、夜の灯り。
路地の奥にある好きな店。
帰り道に立ち寄る場所がある、
そんな日常が成立する街です。

---

## 例3: 駅名「代官山駅」

### short
低層の住宅と、洗練されたショップが共存する街。表通りも路地も、それぞれの表情を持っています。

### balanced
低層の住宅街に、ファッションや書店、カフェが点在する街。喧騒からほどよく離れ、暮らしと文化が自然に結びついた距離感が魅力です。

### poetic
坂のある街。
路地のある街。
誰かの暮らしの気配と
店の灯りが、近い場所にあります。

---

これらの例を参考に、与えられた駅名の街紹介を3パターンで生成してください。
"""


def _build_user_prompt(station: str, hint: str = "") -> str:
    hint_part = f"\n補足情報: {hint}" if hint else ""
    return (
        f"駅名: {station}\n"
        f"この街の紹介を short / balanced / poetic の3パターンで書いてください。"
        f"{hint_part}"
    )


def generate_area_descriptions(
    station: str,
    hint: str = "",
    api_key: str | None = None,
) -> list[AreaVariant]:
    """
    駅名から街紹介を 3 案生成する。

    Args:
        station: 駅名 (例: "広尾駅" や "広尾")
        hint: 住所など補足情報 (任意)
        api_key: Anthropic API キー (省略時は環境変数)

    Returns:
        AreaVariant のリスト (short / balanced / poetic の3案)
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY が設定されていません。"
            "サイドバーまたは環境変数で API キーを指定してください。"
        )

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = _build_user_prompt(station, hint)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
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

    area_set = AreaSet.model_validate(parsed)
    return area_set.variants


if __name__ == "__main__":
    for v in generate_area_descriptions("広尾駅"):
        print(f"\n===== {v.label.upper()} =====")
        print(v.text)
