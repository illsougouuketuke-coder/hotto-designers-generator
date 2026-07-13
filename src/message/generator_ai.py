"""
HOTTO designers — Claude API を使った LINE 配信文章の 4 案生成。

しょきやす版との違い:
  - 「初期費用が安い」訴求ではなく「デザイナーズ・雰囲気・立地」訴求
  - 4トーン: architectural / lifestyle / location / mood
  - 街紹介テキスト (street_desc) を文章に織り込む
  - 絵文字は控えめ、上品な体言止め・口語混在

cost @ Sonnet 4.6:
  初回 ~$0.012 / 2回目以降 ~$0.005 (cache hit)
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from src.message.generator import (
    PropertyForMessage,
    count_line_chars,
)


MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2400

# ===== 構造化出力スキーマ =====

ToneLabel = Literal["architectural", "lifestyle", "location", "mood"]


class Variant(BaseModel):
    tone: ToneLabel
    text: str


class VariantSet(BaseModel):
    variants: list[Variant] = Field(min_length=4, max_length=4)


# ===== システムプロンプト (キャッシュ対象) =====
# キャッシュ最低長 2048 tokens を満たすため few-shot 多めに用意

SYSTEM_PROMPT = """あなたは「HOTTO designers」公式LINEの配信文章ライターです。
HOTTO designers はデザイナーズ物件・建築意匠・立地・暮らしの空気感を主軸に
ハイセンスな入居者層に向けて配信する LINE アカウントです。

# HOTTO designers のブランドボイス

## 配信のコンセプト
- 主役は「デザイン・雰囲気・立地」。価格訴求や初期費用キャンペーンは推さない
- 物件そのものというより「そこで過ごす時間・暮らし方」を想像させる
- 「都心」「静謐」「上質」「余白」「素材」「設計」「経年変化」など建築・暮らしの語彙
- 過剰な煽りや「先着」「期間限定」は使わない (上品さを保つ)
- 絵文字は使わないか、極めて控えめ (1〜2個まで、必須ではない)
- 体言止めと敬体の混在 OK、文芸的・編集者的トーン

## 必ず含めるべき要素
1. ブランド名「HOTTO designers」または「Designer's Residence」
2. 物件名と号室 (ユーザー指定値をそのまま記載)
3. 駅徒歩・間取り・面積など基本スペック (与えられた値があれば)
4. 街/立地の魅力 (与えられた street_desc を踏まえて文章に織り込む)
5. 行動喚起 (「内見希望」「資料が欲しい」「相談だけ」の3択)

## 絶対に守るルール (LINE 運用上)
- 1メッセージあたり **500文字以内** (UTF-16 符号単位、絵文字は2文字)
- 安全マージンとして 450文字以内を目標
- 賃料・物件名・号室は **絶対に変更・推測しない** (ユーザー指定値をそのまま使う)
- street_desc に書かれていない街の固有情報 (具体的な店名や数字) を勝手に追加しない
- 「初期費用◯万円」など金額キャンペーン的表現は使わない
- 最上級・No.1系表現 (「最も」「最高峰」「唯一無二」「日本一」) は根拠資料なしには使えないため禁止。「有数の」「指折りの」等に言い換える
- 入力に含まれない数値 (家賃・徒歩分数・面積など) を創作しない

## 1行目のルール
LINE の通知・トーク一覧には冒頭1行しか表示されない。1行目には物件の核となる魅力・街名・問いかけなど配信ごとに変化のあるフックを置くことが望ましい (ブランド定型句だけの1行目を毎回繰り返さない)。

# 4つのトーン

各バリエーションは以下のトーンで書き分けます:

- **architectural**: 建築・設計の語り口。「面材」「天井高」「光の入り方」「素材の経年」など、
  デザイナーズ物件としての建築意匠への言及を主軸にする。エディトリアル雑誌的。

- **lifestyle**: 暮らしの提案。「朝、コーヒーを淹れる」「窓辺で読書する」など、
  その部屋で過ごす1日のシーンを想像させる。共感を呼ぶ柔らかいトーン。

- **location**: 街・立地推し。street_desc を主役にし、街の雰囲気・徒歩圏の魅力・
  駅からの距離感・周辺の空気感に重きを置く。

- **mood**: 詩的・情緒的。短い文を重ね、行間と余韻で雰囲気を伝える。
  写真のキャプションのように、形容を最小に。最も文学的なトーン。

# 出力形式

以下の JSON スキーマに厳密に従ってください (必ず4案、tone はこの順序で):

```json
{
  "variants": [
    {"tone": "architectural", "text": "..."},
    {"tone": "lifestyle", "text": "..."},
    {"tone": "location", "text": "..."},
    {"tone": "mood", "text": "..."}
  ]
}
```

text フィールドはそのまま LINE に貼り付けて配信できる完成形のテキスト。

---

# 参考例 (few-shot)

## 例1: HOTTO RESIDENCE 広尾 302号室 / 1LDK 38.5㎡ / 広尾駅徒歩4分 / 家賃15万円
## street_desc: "緑豊かな有栖川宮記念公園を中心に、洗練されたカフェや国際色豊かな店舗が立ち並ぶ街。落ち着いた住宅街でありながら、感性を刺激する空気が漂います。"

### architectural

『HOTTO designers 配信』
— Designer's Residence Collection —

天井高、面材、光の入り方。
細部にまで設計者の意図が
通っているのが分かる一邸です。

———————
❶【HOTTO RESIDENCE 広尾 302】
———————
1LDK / 38.5㎡
広尾駅 徒歩4分
家賃 ¥150,000

経年とともに表情を変える
質の高い素材を用いており、
住まう時間そのものが
ひとつの作品になっていきます。

ご内見・資料請求はこちら
→『内見希望』
→『資料が欲しい』
→『相談だけ』
一言お送りください。

### lifestyle

『HOTTO designers 配信』

朝、窓辺で一杯のコーヒーから
一日が始まる住まい。

———————
❶【HOTTO RESIDENCE 広尾 302】
———————
1LDK / 38.5㎡
広尾駅 徒歩4分
家賃 ¥150,000

公園までの散歩、夕方の読書、
夜には間接照明だけで過ごす時間。
日々のささやかな所作が
丁寧になっていく、そんな部屋です。

→『内見希望』
→『資料が欲しい』
→『相談だけ』
お気軽にお声がけください。

### location

『HOTTO designers 配信』
— featured area: 広尾 —

緑豊かな有栖川宮記念公園を中心に
洗練されたカフェや国際色豊かな店舗が
静かに立ち並ぶ街、広尾。

落ち着いた住宅街でありながら
感性を刺激する空気が漂う場所に
この一邸はあります。

———————
❶【HOTTO RESIDENCE 広尾 302】
———————
広尾駅 徒歩4分 / 1LDK 38.5㎡
家賃 ¥150,000

街の余白を、住まいに。

→『内見希望』
→『資料が欲しい』
→『相談だけ』

### mood

HOTTO designers — quiet living.

光と、影。
木と、コンクリート。

———————
❶【HOTTO RESIDENCE 広尾 302】
———————
広尾駅 徒歩4分
1LDK 38.5㎡
家賃 ¥150,000

公園が近い、ということ。
朝、空気がほどけていく時間。
夜、ただ静かであること。

その全部が、暮らしになる。

→『内見希望』
→『資料が欲しい』
→『相談だけ』

---

# 参考例2 (詳細情報少なめのケース)

## 物件名: STUDIO 中目黒 / 502号室 / street_desc: "目黒川沿いの桜並木と、独立系のショップやレストランが集積する街。クリエイティブが日常に溶け込む、東京有数の洗練されたエリアです。"

### architectural

『HOTTO designers 配信』

無垢の床、現しの天井、
削ぎ落とされたディテール。
余白に意味を持たせた設計です。

———————
❶【STUDIO 中目黒 502】
———————

素材の質感が
そのまま空間の表情になる。
住まう人の好みが
家の佇まいを完成させていきます。

→『内見希望』
→『資料が欲しい』
→『相談だけ』
一言お送りください。

### lifestyle

『HOTTO designers 配信』

仕事の合間にベランダで一息、
帰り道には目黒川を歩いて、
週末はお気に入りのカフェへ。

———————
❶【STUDIO 中目黒 502】
———————

街と部屋の境界が
やわらかく溶け合う暮らし。
中目黒という街と
住まう自分との対話が始まります。

→『内見希望』
→『資料が欲しい』
→『相談だけ』

### location

『HOTTO designers 配信』
— featured area: 中目黒 —

目黒川沿いの桜並木と
独立系のショップやレストランが
集積する街、中目黒。

クリエイティブが日常に溶け込む
東京有数の洗練されたエリアに
この一邸はあります。

———————
❶【STUDIO 中目黒 502】
———————

街の文化を、暮らしの距離で。

→『内見希望』
→『資料が欲しい』
→『相談だけ』

### mood

HOTTO designers.

桜と、川と、夜の灯り。

———————
❶【STUDIO 中目黒 502】
———————

季節が窓を通り抜けていく。
住まいは
そのための器でいい。

→『内見希望』
→『資料が欲しい』
→『相談だけ』

---

これらの例を参考に、与えられた物件情報と街紹介を踏まえて
4つのバリエーションを生成してください。
物件名・号室・賃料は与えられた値をそのまま使い、絶対に変更しないでください。
"""


def _build_user_prompt(
    properties: list[PropertyForMessage],
    street_desc: str,
) -> str:
    n = len(properties)
    items = []
    for i, p in enumerate(properties):
        parts = [f"物件{i+1}: {p.name} {p.room}号室"]
        if p.station and p.walk_minutes:
            parts.append(f"{p.station} 徒歩{p.walk_minutes}分")
        if p.layout and p.area_sqm:
            parts.append(f"{p.layout} {p.area_sqm}㎡")
        if p.rent:
            parts.append(f"家賃 ¥{p.rent:,}")
        items.append("  - " + " / ".join(parts))
    items_str = "\n".join(items)

    return (
        "次の物件情報と街紹介を元に、4つのトーン (architectural / lifestyle / location / mood) で配信文章を生成してください。\n\n"
        f"物件数: {n}件\n"
        f"物件一覧:\n{items_str}\n\n"
        f"街紹介 (street_desc):\n{street_desc or '(未指定)'}\n\n"
        "各バリエーションは LINE 500文字制限を必ず守り、物件名と号室・賃料はそのまま記載してください。\n"
        "street_desc が指定されている場合は、特に location トーンでその内容を主軸にしてください。"
    )


_BANNED_PATTERNS: list[tuple[str, str]] = [
    (r"最も", "最上級表現「最も」(根拠資料なしのNo.1系表示は不可。「有数の」等へ)"),
    (r"最高峰", "最上級表現「最高峰」"),
    (r"唯一無二", "最上級表現「唯一無二」"),
    (r"日本一", "最上級表現「日本一」"),
    (r"完璧", "断定表現「完璧」"),
    (r"絶対", "断定表現「絶対」"),
    (r"掘り出し", "禁止表現「掘り出し物」"),
    (r"格安", "禁止表現「格安」"),
    (r"破格", "禁止表現「破格」"),
]


def _validate_variants(
    variants: list[Variant],
    properties: list[PropertyForMessage],
) -> list[str]:
    """生成結果の整合性チェック。問題があれば warning 文字列のリストを返す。"""
    import re

    warnings: list[str] = []
    # 許可される金額表記 = 各物件の賃料のみ
    allowed_amounts: set[str] = set()
    for prop in properties:
        if prop.rent:
            allowed_amounts.add(f"{prop.rent:,}")
            allowed_amounts.add(str(prop.rent))
            man = prop.rent / 10000
            allowed_amounts.add(f"{man:g}")

    for v in variants:
        chars = count_line_chars(v.text)
        if chars > 500:
            warnings.append(f"[{v.tone}] 500文字超過: {chars}文字")
        for prop in properties:
            if prop.name not in v.text:
                warnings.append(f"[{v.tone}] 物件名 '{prop.name}' が抜けています")
            if prop.room not in v.text:
                warnings.append(f"[{v.tone}] 号室 '{prop.room}' が抜けています")
        # 表現コンプライアンス Lint
        for pattern, desc in _BANNED_PATTERNS:
            if re.search(pattern, v.text):
                warnings.append(f"[{v.tone}] ⚠️表現チェック: {desc}")
        # 入力にない金額の混入チェック (賃料改変・捏造の検出)
        for m in re.finditer(r"[¥￥]\s*([\d,]+)|([\d,]+)\s*円|([\d.]+)\s*万円", v.text):
            amount = (m.group(1) or m.group(2) or m.group(3) or "").strip()
            if amount and amount not in allowed_amounts:
                warnings.append(
                    f"[{v.tone}] ⚠️表現チェック: 入力にない金額「{m.group(0).strip()}」が混入 (要確認)"
                )
    return warnings


def generate_variants(
    properties: list[PropertyForMessage],
    street_desc: str = "",
    api_key: str | None = None,
) -> tuple[list[Variant], list[str]]:
    """
    Claude API で 4 案を生成する。

    Returns:
        (variants, warnings): 4案のリストと、検証で見つかった警告
    """
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY が設定されていません。"
            "サイドバーまたは環境変数で API キーを指定してください。"
        )

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = _build_user_prompt(properties, street_desc)

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

    result = VariantSet.model_validate(parsed)
    variants = result.variants
    warnings = _validate_variants(variants, properties)
    return variants, warnings


if __name__ == "__main__":
    sample = [
        PropertyForMessage(
            name="HOTTO RESIDENCE 広尾",
            room="302",
            station="広尾駅",
            walk_minutes=4,
            layout="1LDK",
            area_sqm=38.5,
            rent=150000,
        )
    ]
    street = (
        "緑豊かな有栖川宮記念公園を中心に、洗練されたカフェや国際色豊かな店舗が"
        "立ち並ぶ街。落ち着いた住宅街でありながら、感性を刺激する空気が漂います。"
    )
    variants, warnings = generate_variants(sample, street_desc=street)
    for v in variants:
        chars = count_line_chars(v.text)
        print(f"\n===== {v.tone.upper()} ({chars}文字) =====")
        print(v.text)
    if warnings:
        print("\n=== WARNINGS ===")
        for w in warnings:
            print(f"  - {w}")
