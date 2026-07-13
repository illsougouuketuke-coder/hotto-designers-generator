"""
画像のアップスケーリング処理。
マイソクPDFから抽出した低解像度写真(300x300px程度)を、
LANCZOS補間 + アンシャープマスク + ノイズ除去で配信品質に高める。

依存:
  pip install opencv-python opencv-contrib-python

注意: AI生成ではなく、伝統的な画像処理アルゴリズム。
      物件の実物を改変するリスクがないので、不動産配信用途として安全。
"""

from pathlib import Path
import cv2


def upscale_photo(
    input_path: Path,
    output_path: Path,
    scale: int = 3,
    apply_denoise: bool = True,
) -> Path:
    """
    写真をアップスケール+シャープ化+ノイズ除去。
    マイソクPDFの300x300px写真を900x900pxの配信品質に変換する用途。

    Args:
        input_path: 入力画像
        output_path: 出力画像
        scale: 拡大倍率(デフォルト3倍)
        apply_denoise: ノイズ除去を適用するか(処理時間と引き換え)

    Returns:
        生成された画像のパス
    """
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {input_path}")

    h, w = img.shape[:2]

    # 1. LANCZOS4 でアップスケール(エッジ保持に優れる)
    upscaled = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    # 2. アンシャープマスク(ボケた印象をシャープに)
    blur = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, blur, -0.5, 0)

    # 3. ノイズ除去(配信品質に必要、ただし重い処理)
    if apply_denoise:
        result = cv2.fastNlMeansDenoisingColored(
            sharpened,
            None,
            h=3,                # 輝度のフィルタ強度
            hColor=3,           # 色のフィルタ強度
            templateWindowSize=7,
            searchWindowSize=21,
        )
    else:
        result = sharpened

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path


def upscale_floor_plan(
    input_path: Path,
    output_path: Path,
    scale: int = 2,
) -> Path:
    """
    間取り図用のアップスケール(線画なのでINTER_CUBICで十分・ノイズ除去不要)
    """
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {input_path}")
    h, w = img.shape[:2]
    upscaled = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path

def ensure_delivery_resolution(
    input_path: Path,
    min_long_side: int = 1400,
    max_scale: int = 4,
    is_floor_plan: bool = False,
) -> Path:
    """
    どんな入力画像でも配信品質の解像度に自動正規化する。
    長辺が min_long_side 以上ならそのまま返し、不足していれば
    必要倍率(上限 max_scale)でアップスケールした複製を隣に作って返す。

    冪等: 生成済みの *_norm ファイルがあればそれを再利用する。
    読み込み失敗など異常時は元のパスをそのまま返す(生成を止めない)。
    """
    try:
        img = cv2.imread(str(input_path))
        if img is None:
            return input_path
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side >= min_long_side:
            return input_path

        out_path = input_path.with_name(f"{input_path.stem}_norm{input_path.suffix}")
        if out_path.exists():
            return out_path

        scale = min(max_scale, -(-min_long_side // long_side))  # 切り上げ
        if is_floor_plan:
            upscale_floor_plan(input_path, out_path, scale=scale)
        else:
            # 極小画像はノイズ除去まで、そこそこの画像はシャープ化のみで高速化
            upscale_photo(input_path, out_path, scale=scale, apply_denoise=long_side <= 600)
        return out_path
    except Exception:
        return input_path
