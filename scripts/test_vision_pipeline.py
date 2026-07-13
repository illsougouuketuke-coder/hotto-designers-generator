"""
Vision パイプライン (vision_classify / mysoku_crop) の実走テスト。

環境変数 ANTHROPIC_API_KEY があれば samples/ の画像で分類を実走する。
キーがなければ import チェック + オフライン幾何セルフテスト
(bbox変換・インセット・面積/アスペクトフィルタ・IoU重複除去・枚数キャップ)
を行い終了する (CI・キー未設定環境用)。

使い方:
    # キーなし: import チェック + オフライン幾何セルフテスト
    uv run python scripts/test_vision_pipeline.py

    # 分類テスト (デフォルト: samples/1301/ の画像)
    ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/test_vision_pipeline.py

    # 画像ディレクトリを指定
    ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/test_vision_pipeline.py path/to/images/

    # マイソクスクショの切り出しテストも行う場合 (第2引数)
    # 例: 約12枚の写真を含むマイソクで8枚以上切り出せるか・文字帯混入がないかを確認する
    ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/test_vision_pipeline.py samples/1301 path/to/mysoku.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src.extractor.pdf_images import (  # noqa: E402
    ExtractedImage,
    _white_ratio,
    classify_for_poster,
)
from src.extractor.mysoku_crop import (  # noqa: E402
    MAX_REGIONS,
    _select_crop_boxes,
    crop_photos_from_mysoku,
)
from src.extractor.vision_classify import classify_images_with_vision  # noqa: E402

# テスト対象にする画像の最小辺 (アイコン・帯バナーを除外)
MIN_SIDE = 200
MAX_IMAGES = 20


def _offline_geometry_selftest() -> bool:
    """API キーなしで mysoku_crop の bbox 選別ロジックを検証する。

    実マイソク相当 (1970x1388) のレイアウトを模した bbox セット:
    外観大1 + 外観小3 + 右グリッド室内8 + 間取り図1 = 有効13 (photo12+plan1)、
    さらにノイズ4種 (文字帯・アイコン・重複・全面) が正しく除去されるか。
    """
    w, h = 1970, 1388  # 実ユーザーマイソクの実寸
    valid = [
        # 外観大
        {"x0": 20, "y0": 238, "x1": 300, "y1": 497, "category": "photo"},
        # 外観小3
        {"x0": 20, "y0": 500, "x1": 109, "y1": 576, "category": "photo"},
        {"x0": 112, "y0": 500, "x1": 201, "y1": 576, "category": "photo"},
        {"x0": 203, "y0": 500, "x1": 300, "y1": 576, "category": "photo"},
        # 間取り図
        {"x0": 353, "y0": 371, "x1": 429, "y1": 731, "category": "floor_plan"},
    ]
    # 右グリッド室内写真 8枚 (2列x4行、各約1.5-2%)
    for row in range(4):
        y0 = 241 + row * 135
        y1 = y0 + 130
        for x0, x1 in ((490, 607), (609, 726)):
            valid.append(
                {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "category": "photo"}
            )
    noise = [
        # 文字帯 (アスペクト比 > 4.5 で除去されるべき)
        {"x0": 340, "y0": 195, "x1": 660, "y1": 220, "category": "photo"},
        # アイコンサイズ (面積 0.6% 未満で除去されるべき)
        {"x0": 300, "y0": 250, "x1": 340, "y1": 290, "category": "photo"},
        # グリッド1枚目とほぼ同一の重複 (IoU で除去されるべき)
        {"x0": 492, "y0": 243, "x1": 605, "y1": 369, "category": "photo"},
        # ほぼ全面 (マイソクそのもの、除去されるべき)
        {"x0": 5, "y0": 5, "x1": 995, "y1": 995, "category": "photo"},
    ]
    boxes = _select_crop_boxes(valid + noise, w, h)

    ok = True
    n_photo = sum(1 for _, c in boxes if c == "photo")
    n_plan = sum(1 for _, c in boxes if c == "floor_plan")
    if len(boxes) != 13 or n_photo != 12 or n_plan != 1:
        print(f"  NG: 期待13件(photo12+plan1) に対し {len(boxes)}件 "
              f"(photo{n_photo}+plan{n_plan})")
        ok = False
    total = w * h
    for (x0, y0, x1, y1), _ in boxes:
        area = (x1 - x0) * (y1 - y0)
        bw, bh = x1 - x0, y1 - y0
        if area >= total * 0.90 or max(bw, bh) / min(bw, bh) > 4.5:
            print(f"  NG: フィルタ漏れ box=({x0},{y0},{x1},{y1})")
            ok = False

    # インセット検証: 外観大 (原寸 552x359px) は各辺2%内側に縮む
    big = max(boxes, key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]))
    bx0, by0, bx1, by1 = big[0]
    raw_x0, raw_y0 = round(20 / 1000 * w), round(238 / 1000 * h)
    raw_x1, raw_y1 = round(300 / 1000 * w), round(497 / 1000 * h)
    ins_x = round((raw_x1 - raw_x0) * 0.02)
    ins_y = round((raw_y1 - raw_y0) * 0.02)
    if (bx0, by0, bx1, by1) != (raw_x0 + ins_x, raw_y0 + ins_y,
                                raw_x1 - ins_x, raw_y1 - ins_y):
        print(f"  NG: インセット不一致 got={big[0]}")
        ok = False

    # 枚数キャップ検証: 有効20枚 -> MAX_REGIONS 枚
    many = [
        {"x0": (i % 5) * 200 + 5, "y0": (i // 5) * 200 + 5,
         "x1": (i % 5) * 200 + 195, "y1": (i // 5) * 200 + 195,
         "category": "photo"}
        for i in range(20)
    ]
    capped = _select_crop_boxes(many, w, h)
    if len(capped) != MAX_REGIONS:
        print(f"  NG: 枚数キャップ {MAX_REGIONS} に対し {len(capped)}件")
        ok = False

    return ok


def _collect_images(images_dir: Path) -> list[ExtractedImage]:
    """ディレクトリ内の画像を ExtractedImage に包んで返す。"""
    paths = sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    result: list[ExtractedImage] = []
    for p in paths:
        try:
            with Image.open(p) as img:
                w, h = img.size
        except Exception:
            continue
        if min(w, h) < MIN_SIDE:
            continue
        wr = _white_ratio(p)
        result.append(
            ExtractedImage(
                path=p,
                width=w,
                height=h,
                file_size=p.stat().st_size,
                white_ratio=wr,
                is_likely_floor_plan=wr >= 0.80,
                is_likely_photo=wr < 0.55,
            )
        )
        if len(result) >= MAX_IMAGES:
            break
    return result


def main() -> int:
    print("=== Vision パイプライン テスト ===")
    print("import チェック: OK (vision_classify / mysoku_crop)")

    print("--- オフライン幾何セルフテスト (mysoku_crop._select_crop_boxes) ---")
    if _offline_geometry_selftest():
        print("オフライン幾何セルフテスト: OK (13件選別・フィルタ・インセット・キャップ)")
    else:
        print("オフライン幾何セルフテスト: FAILED")
        return 1

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print()
        print("ANTHROPIC_API_KEY が未設定のため、実走テストはスキップしました。")
        print("実走するには: ANTHROPIC_API_KEY=sk-ant-... uv run python scripts/test_vision_pipeline.py <images_dir> <mysoku.png>")
        return 0

    # ---- 分類テスト ----
    if len(sys.argv) > 1:
        images_dir = Path(sys.argv[1])
    elif (ROOT / "samples" / "1301").is_dir():
        images_dir = ROOT / "samples" / "1301"
    else:
        images_dir = ROOT / "samples"

    if not images_dir.is_dir():
        print(f"画像ディレクトリが見つかりません: {images_dir}")
        print("引数で指定してください: uv run python scripts/test_vision_pipeline.py <images_dir>")
        return 1

    images = _collect_images(images_dir)
    if not images:
        print(f"{images_dir} に対象画像 (最小辺{MIN_SIDE}px以上) がありません。")
        print("引数で画像ディレクトリを指定してください。")
        return 1

    print()
    print(f"--- 分類テスト: {images_dir} の {len(images)} 枚 ---")

    heuristic = classify_for_poster(images)
    print("[従来ヒューリスティック]")
    for slot, im in heuristic.items():
        print(f"  {slot}: {im.path.name if im else 'NONE'}")

    print("[Claude Vision] (API呼び出し中...)")
    vision = classify_images_with_vision(images, api_key=api_key)
    for slot, im in vision.items():
        print(f"  {slot}: {im.path.name if im else 'NONE'}")

    # ---- マイソク切り出しテスト (第2引数があれば) ----
    if len(sys.argv) > 2:
        mysoku = Path(sys.argv[2])
        out_dir = ROOT / "output" / "_mysoku_crops_test"
        print()
        print(f"--- 切り出しテスト: {mysoku} ---")
        crops = crop_photos_from_mysoku(mysoku, out_dir, api_key=api_key)
        print(f"切り出し {len(crops)} 枚 -> {out_dir}/")
        for path, category in crops:
            print(f"  [{category}] {path.name}")
        if len(crops) >= 2:
            crop_imgs = [
                ExtractedImage(
                    path=p, width=Image.open(p).width, height=Image.open(p).height,
                    file_size=p.stat().st_size, white_ratio=0.0,
                    is_likely_floor_plan=(c == "floor_plan"),
                    is_likely_photo=(c != "floor_plan"),
                )
                for p, c in crops
            ]
            print("[切り出し画像のスロット割当] (API呼び出し中...)")
            slots = classify_images_with_vision(crop_imgs, api_key=api_key)
            for slot, im in slots.items():
                print(f"  {slot}: {im.path.name if im else 'NONE'}")

    print()
    print("=== テスト完了 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
