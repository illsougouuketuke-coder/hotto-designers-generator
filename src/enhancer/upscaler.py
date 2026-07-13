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
