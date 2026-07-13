"""
マイソク画像 (スクリーンショット等) からの写真自動切り出し。

マイソク全体を Claude Vision に送り、「物件写真」「間取り図」の各領域の
バウンディングボックスを JSON で返させ、PIL で切り出して保存する。

座標は画像サイズに対する 0-1000 の正規化整数で要求する (この形式が安定する)。
bbox がズレる可能性があるため、期待どおりでなくてもクラッシュせず、
切り出せた分だけ返す設計。

重要な学び (image_extractor.py と同じ):
    anthropic の messages.parse (構造化出力) は 'Grammar compilation timed out'
    エラーを起こすため使用禁止。
    messages.create + プロンプトでJSON要求 + 正規表現でJSON抽出 のパターンを使う。
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import anthropic
from PIL import Image

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2000

# API 送信用の縮小サイズ (長辺最大)。座標は正規化なので縮小しても元画像に適用できる。
MAX_SEND_SIDE = 1568
JPEG_QUALITY = 85

# 元画像の面積に対する最小領域比 (これ未満は捨てる = アイコン・ノイズ)。
# マイソク右側にグリッド状に並ぶ小さな室内写真は1枚あたり約1.5-2%、
# 小さな外観写真は約0.7%しかないため、3%だと全滅する。0.6%まで下げ、
# 誤検出対策は面積ではなくアスペクト比・枚数キャップ・IoU重複除去で行う。
MIN_AREA_RATIO = 0.006
# 元画像のほぼ全体を覆うボックスは「マイソクそのもの」なので捨てる
MAX_AREA_RATIO = 0.90
# 極端に細長いボックス (文字帯・バナー・罫線) は捨てる (長辺/短辺 がこれ超)
MAX_ASPECT_RATIO = 4.5
# 1枚のマイソクから切り出す最大枚数 (面積の大きい順に残す)
MAX_REGIONS = 16
# ほぼ同じ領域の重複ボックスを除去する IoU 閾値
IOU_DEDUP_THRESHOLD = 0.55
# 切り出し時に bbox を上下左右それぞれ寸法のこの割合だけ内側に縮める安全インセット。
# bbox が写真の縁の外の文字帯・枠線をわずかに含んでも混入しにくくする。
INSET_RATIO = 0.02

# ===== 2段階目リファインパス =====
# 1段階目の bbox は「写真の周囲の表組み・文字帯ごと」広く返ってくることがある
# (実測: 2026-07-12 のマイソクで切り出しの上下に文字帯・右に物件概要表が混入)。
# 全切り出しを1回のAPI呼び出しでまとめて再検査し、混入があれば写真本体のみに絞り込む。
REFINE_PASS = True
REFINE_SEND_SIDE = 640      # リファイン送信用の縮小長辺 (px)
REFINE_MIN_SIDE_PX = 100    # リファイン後にこの短辺未満になる box は不正とみなし元を維持

# ===== 低解像度切り出しのアップスケール =====
# マイソク内の小さな写真 (300px前後) は配信画像で拡大されてぼやけるため、
# 短辺がこの値未満の切り出しは LANCZOS+シャープ化で拡大してから使う。
UPSCALE_MIN_SIDE_PX = 450

VALID_CATEGORIES = {"photo", "floor_plan"}


SYSTEM_PROMPT = """あなたは不動産マイソク(物件チラシ)画像のレイアウトを解析する専門家です。

与えられたマイソク画像の中から、以下2種類の領域を全て見つけて、
バウンディングボックスをJSONで回答してください。
余計な前置きや説明文は一切書かず、純粋なJSONのみを出力します。

# 検出対象

- "photo": 物件写真 (室内・外観・キッチン・浴室などの実写真)
- "floor_plan": 間取り図 (部屋の配置を示す線画の図面)

# 検出対象外 (ボックスに含めない)

- 地図・周辺案内図
- 会社ロゴ・QRコード・キャンペーンバナー
- 文字だけの表・物件概要欄

# 座標系

- 画像の左上が原点 (0, 0)、右下が (1000, 1000)
- 座標は画像の幅・高さに対する 0-1000 の正規化整数
- x0 < x1、y0 < y1 (x0,y0 = 左上、x1,y1 = 右下)
- bbox は写真のピクセルのみを囲む。写真に隣接するキャプション・文字帯
  (日付・専有面積などの記載)・QRコード・表・枠線・余白は絶対に含めない
- 境界が曖昧なときは、はみ出すより少し内側 (写真の内部) に寄せる
- 間取り図 (floor_plan) は図面線画の外周のみを囲む。隣接する写真列・
  QRコード・注意書き・物件概要表は絶対に含めない

# JSON出力フォーマット

```json
{
  "regions": [
    {"x0": 20, "y0": 50, "x1": 480, "y1": 400, "category": "photo"},
    {"x0": 520, "y0": 50, "x1": 980, "y1": 600, "category": "floor_plan"}
  ]
}
```

# 重要なルール
- category は "photo" か "floor_plan" のみ
- 見つかった領域を全て列挙する (写真が12枚あれば12ボックス)
- グリッド状に並ぶ小さな室内写真も、1枚ずつ個別のボックスで返す
  (複数枚を1つの大きなボックスに結合しない)
- 小さくても物件写真であれば必ず含める (ロゴ・アイコン・地図は含めない)
- 回答は ```json ... ``` のコードブロック1つのみ、それ以外は一切書かない
"""


def _encode_for_api(img: Image.Image, max_side: int = MAX_SEND_SIDE) -> tuple[str, str]:
    """PIL Image を長辺 max_side px に縮小し JPEG で base64 エンコード。"""
    send = img.convert("RGB")
    send.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    send.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return "image/jpeg", b64


REFINE_SYSTEM_PROMPT = """あなたは不動産マイソクから切り出された画像パッチの品質検査員です。
複数の番号付き画像が与えられます。それぞれ「物件写真」または「間取り図」の候補ですが、
切り出しが甘く、写真の周囲にマイソクの文字帯・表組み・QRコード・隣の写真の断片が
混入していることがあります。

各画像について以下を判定し、JSONのみで回答してください (前置き・説明は一切不要)。

- verdict:
  - "clean": ほぼ写真 (または間取り図) のピクセルだけで構成されている
  - "refine": 写真本体はあるが、周囲に文字帯・表・別要素が混入している
  - "reject": 写真が主役ではない (大部分が文字・表・ロゴ・QR・地図)、
              または物件写真/間取り図ではない
- verdict が "refine" のときのみ、その画像内で「写真本体のみ」を囲む bbox を
  0-1000 正規化整数 (x0,y0,x1,y1 / 左上原点) で返す。
  文字帯・表・枠線・余白は絶対に含めない。迷ったら内側に寄せる。
  間取り図の場合は図面線画の外周のみを囲む。

# JSON出力フォーマット

```json
{"results": [
  {"index": 0, "verdict": "clean"},
  {"index": 1, "verdict": "refine", "x0": 30, "y0": 120, "x1": 970, "y1": 860},
  {"index": 2, "verdict": "reject"}
]}
```

回答は ```json ... ``` のコードブロック1つのみ。全画像分の results を必ず返す。
"""


def _apply_refine_results(
    crops: list[tuple[Image.Image, str]],
    results: list,
) -> list[tuple[Image.Image, str]]:
    """リファイン判定を切り出し画像リストに適用する (純関数・オフラインテスト可能)。

    - clean / 判定なし: そのまま採用
    - refine: 画像内 0-1000 正規化 bbox で再切り出し。bbox が不正
      (座標矛盾・短辺 REFINE_MIN_SIDE_PX 未満) なら元画像のまま採用
    - reject: 除外
    """
    verdicts: dict[int, dict] = {}
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("index"), (int, float)):
            verdicts[int(r["index"])] = r

    refined: list[tuple[Image.Image, str]] = []
    for i, (img, category) in enumerate(crops):
        v = verdicts.get(i)
        if v is None:
            refined.append((img, category))
            continue
        verdict = str(v.get("verdict", "clean")).strip().lower()
        if verdict == "reject":
            continue
        if verdict != "refine":
            refined.append((img, category))
            continue
        try:
            w, h = img.size
            px0 = max(0, min(w, round(int(v["x0"]) / 1000 * w)))
            py0 = max(0, min(h, round(int(v["y0"]) / 1000 * h)))
            px1 = max(0, min(w, round(int(v["x1"]) / 1000 * w)))
            py1 = max(0, min(h, round(int(v["y1"]) / 1000 * h)))
            if (px1 - px0) < REFINE_MIN_SIDE_PX or (py1 - py0) < REFINE_MIN_SIDE_PX:
                refined.append((img, category))
                continue
            refined.append((img.crop((px0, py0, px1, py1)), category))
        except (KeyError, TypeError, ValueError):
            refined.append((img, category))
    return refined


def _refine_crops(
    client: "anthropic.Anthropic",
    crops: list[tuple[Image.Image, str]],
) -> list[tuple[Image.Image, str]]:
    """全切り出しを1回のAPI呼び出しで再検査し、混入を除去する。

    API・JSON抽出の失敗時は例外を握って元のリストをそのまま返す
    (リファインは品質向上のためのベストエフォート)。
    """
    if not crops:
        return crops
    try:
        content: list = []
        for i, (img, _) in enumerate(crops):
            media_type, b64 = _encode_for_api(img, max_side=REFINE_SEND_SIDE)
            content.append({"type": "text", "text": f"[画像 {i}]"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            })
        content.append({
            "type": "text",
            "text": (
                f"以上 {len(crops)} 枚 (index 0-{len(crops) - 1}) を検査し、"
                "指定のJSONフォーマットでのみ回答してください。"
            ),
        })

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": REFINE_SYSTEM_PROMPT,
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
        results = parsed.get("results", [])
        if not isinstance(results, list):
            return crops
        return _apply_refine_results(crops, results)
    except Exception:
        return crops


def _maybe_upscale(out_path: Path, category: str) -> None:
    """短辺が UPSCALE_MIN_SIDE_PX 未満の切り出しをその場でアップスケールする。

    失敗しても元ファイルが残るだけなので例外は握りつぶす。
    """
    try:
        with Image.open(out_path) as img:
            short = min(img.size)
        if short >= UPSCALE_MIN_SIDE_PX:
            return
        scale = 3 if short < 250 else 2
        from src.enhancer.upscaler import upscale_floor_plan, upscale_photo

        if category == "floor_plan":
            upscale_floor_plan(out_path, out_path, scale=scale)
        else:
            upscale_photo(out_path, out_path, scale=scale)
    except Exception:
        pass


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


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """2つのピクセルbox (x0, y0, x1, y1) の IoU (Intersection over Union)。"""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _select_crop_boxes(
    regions: list,
    width: int,
    height: int,
) -> list[tuple[tuple[int, int, int, int], str]]:
    """AI が返した regions (0-1000 正規化) を切り出し用ピクセルboxに変換・選別する。

    純関数 (API 呼び出しなし) なのでオフラインでユニットテスト可能。

    処理順:
        1. 0-1000 正規化 -> ピクセル座標 (クランプ)
        2. 面積フィルタ (MIN_AREA_RATIO 未満 / MAX_AREA_RATIO 超を除去) ※bbox原寸で判定
        3. アスペクト比フィルタ (長辺/短辺 > MAX_ASPECT_RATIO の文字帯・バナーを除去)
        4. 安全インセット (上下左右それぞれ寸法の INSET_RATIO だけ内側へ縮小)
        5. IoU 重複除去 (面積の大きい順に採用、既採用と IoU > IOU_DEDUP_THRESHOLD は捨てる)
        6. 枚数キャップ (面積の大きい順に MAX_REGIONS 枚まで)

    Returns:
        [((px0, py0, px1, py1), category), ...] 面積の大きい順。
    """
    total_area = width * height
    candidates: list[tuple[tuple[int, int, int, int], str]] = []

    for region in regions:
        if not isinstance(region, dict):
            continue
        try:
            x0 = int(region["x0"])
            y0 = int(region["y0"])
            x1 = int(region["x1"])
            y1 = int(region["y1"])
        except (KeyError, TypeError, ValueError):
            continue

        category = str(region.get("category", "photo")).strip().lower()
        if category not in VALID_CATEGORIES:
            category = "photo"

        # 0-1000 正規化 -> ピクセル座標 (クランプ付き)
        px0 = max(0, min(width, round(x0 / 1000 * width)))
        py0 = max(0, min(height, round(y0 / 1000 * height)))
        px1 = max(0, min(width, round(x1 / 1000 * width)))
        py1 = max(0, min(height, round(y1 / 1000 * height)))
        if px1 <= px0 or py1 <= py0:
            continue

        w = px1 - px0
        h = py1 - py0

        # 面積フィルタ (bbox 原寸で判定)
        area = w * h
        if area < total_area * MIN_AREA_RATIO:
            continue
        if area > total_area * MAX_AREA_RATIO:
            continue

        # アスペクト比フィルタ (文字帯・バナー・罫線)
        if max(w, h) / min(w, h) > MAX_ASPECT_RATIO:
            continue

        # 安全インセット: 各辺を寸法の INSET_RATIO だけ内側へ
        inset_x = round(w * INSET_RATIO)
        inset_y = round(h * INSET_RATIO)
        ix0, iy0 = px0 + inset_x, py0 + inset_y
        ix1, iy1 = px1 - inset_x, py1 - inset_y
        if ix1 <= ix0 or iy1 <= iy0:
            continue

        candidates.append(((ix0, iy0, ix1, iy1), category))

    # IoU 重複除去: 面積の大きい順に走査し、既採用とほぼ同じ領域は捨てる
    candidates.sort(key=lambda c: -((c[0][2] - c[0][0]) * (c[0][3] - c[0][1])))
    kept: list[tuple[tuple[int, int, int, int], str]] = []
    for box, category in candidates:
        if any(_iou(box, kb) > IOU_DEDUP_THRESHOLD for kb, _ in kept):
            continue
        kept.append((box, category))

    # 枚数キャップ
    return kept[:MAX_REGIONS]


def crop_photos_from_mysoku(
    mysoku_path: Path,
    output_dir: Path,
    api_key: str,
) -> list[tuple[Path, str]]:
    """マイソク画像から物件写真・間取り図の領域を切り出して保存する。

    Args:
        mysoku_path: マイソク画像 (.png/.jpg 等) のパス
        output_dir: 切り出し画像の保存先ディレクトリ (自動作成)
        api_key: Anthropic API キー (必須)

    Returns:
        [(切り出し画像パス, category), ...]
        category は "photo" | "floor_plan"。
        bbox 不正・小さすぎる領域はスキップし、切り出せた分だけ返す。

    Raises:
        RuntimeError: API キー未設定
        anthropic.APIError / ValueError: API 呼び出し・JSON抽出の失敗。
            呼び出し側で従来動作 (マイソク全体を使用) にフォールバックすること。
    """
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY が設定されていません。"
            "マイソクからの写真切り出しには API キーが必要です。"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic(api_key=api_key)

    with Image.open(mysoku_path) as original:
        original.load()
        width, height = original.size
        media_type, b64 = _encode_for_api(original)

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
                            "text": (
                                "このマイソク画像から物件写真と間取り図の領域を全て検出し、"
                                "指定のJSONフォーマットでのみ回答してください。"
                            ),
                        },
                    ],
                }
            ],
        )

        raw_text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                raw_text += block.text

        parsed = _extract_json(raw_text)
        regions = parsed.get("regions", [])
        if not isinstance(regions, list):
            regions = []

        boxes = _select_crop_boxes(regions, width, height)

        crops: list[tuple[Image.Image, str]] = []
        for box, category in boxes:
            try:
                crops.append((original.crop(box), category))
            except Exception:
                # 個別の切り出し失敗はスキップし、切り出せた分だけ返す
                continue

        # 2段階目: 混入検査 (文字帯・表組みの除去 / 写真でないものの棄却)
        if REFINE_PASS:
            crops = _refine_crops(client, crops)

        results: list[tuple[Path, str]] = []
        for i, (crop, category) in enumerate(crops):
            try:
                out_path = output_dir / f"crop_{i:02d}_{category}.png"
                crop.save(out_path)
                _maybe_upscale(out_path, category)
                results.append((out_path, category))
            except Exception:
                continue

    return results


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.extractor.mysoku_crop <mysoku_image> [output_dir]")
        sys.exit(1)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / "_mysoku_crops"
    crops = crop_photos_from_mysoku(
        src, out, api_key=os.environ.get("ANTHROPIC_API_KEY", "")
    )
    print(f"Cropped {len(crops)} regions to {out}/")
    for path, category in crops:
        print(f"  [{category}] {path.name}")
