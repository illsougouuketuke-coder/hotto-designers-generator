"""
マイソク PDF から画像を抽出してフィルタリングする。

PDFには 物件外観・室内写真・間取り図・アイコン・キャンペーン告知 が
玉石混交で埋め込まれている。サイズ・アスペクト比でフィルタし、
さらに「白率」で間取り図候補を切り分ける。

最終的にどれをヒーロー/下部1/下部2/間取り図に使うかは UI でユーザに選ばせる前提。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
from PIL import Image


# サイズ・アスペクト比のフィルタ閾値
# 230: エビス・リビング形式の 233-256px 写真にも対応するための妥協値
MIN_DIMENSION = 230            # 室内写真最小辺
MIN_ASPECT_RATIO = 0.5         # 縦長すぎるものを除外
MAX_ASPECT_RATIO = 2.0         # 横長バナーを除外

# ピクセル値での「白」判定
WHITE_PIXEL_THRESHOLD = 220    # この輝度以上を白とみなす

# 間取り図候補の判定: 高い白率 + ほぼ正方〜縦長
FLOOR_PLAN_MIN_WHITE = 0.80
FLOOR_PLAN_ASPECT_MIN = 0.6
FLOOR_PLAN_ASPECT_MAX = 1.2

# 写真候補(室内写真)の判定: 中〜低の白率(明るい部屋でも 0.55 を超えない経験則)
PHOTO_MAX_WHITE = 0.55
# 「強い写真候補(=確実に部屋写真)」の閾値。これ未満を満たす画像が3枚未満なら警告。
STRONG_PHOTO_MAX_WHITE = 0.55


@dataclass
class ExtractedImage:
    path: Path
    width: int
    height: int
    file_size: int
    white_ratio: float
    is_likely_floor_plan: bool
    is_likely_photo: bool

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


def _white_ratio(img_path: Path) -> float:
    """白に近いピクセルの割合(0.0-1.0)。"""
    arr = np.array(Image.open(img_path).convert("L"))
    return float((arr > WHITE_PIXEL_THRESHOLD).sum()) / arr.size if arr.size else 0.0


def extract_images(pdf_path: Path, output_dir: Path) -> list[ExtractedImage]:
    """
    PDF 内の全画像を output_dir に保存し、フィルタしたリストを返す。
    返るのは「室内写真候補 or 間取り図候補」と判定したもののみ。
    アイコン・小さな帯バナーは除外する。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    extracted: list[ExtractedImage] = []
    try:
        global_idx = 0
        for page in doc:
            for img in page.get_images():
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.name not in ("DeviceRGB", "DeviceGray"):
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                w, h = pix.width, pix.height
                aspect = w / h if h else 0.0

                # サイズとアスペクトでフィルタ
                if w < MIN_DIMENSION or h < MIN_DIMENSION:
                    pix = None
                    global_idx += 1
                    continue
                if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
                    pix = None
                    global_idx += 1
                    continue

                out_path = output_dir / f"img_{global_idx:02d}_{w}x{h}.png"
                pix.save(str(out_path))
                pix = None

                wr = _white_ratio(out_path)
                is_plan = (
                    wr >= FLOOR_PLAN_MIN_WHITE
                    and FLOOR_PLAN_ASPECT_MIN <= aspect <= FLOOR_PLAN_ASPECT_MAX
                )
                is_photo = wr < PHOTO_MAX_WHITE

                extracted.append(
                    ExtractedImage(
                        path=out_path,
                        width=w,
                        height=h,
                        file_size=out_path.stat().st_size,
                        white_ratio=wr,
                        is_likely_floor_plan=is_plan,
                        is_likely_photo=is_photo,
                    )
                )
                global_idx += 1
    finally:
        doc.close()

    return extracted


def classify_for_poster(
    images: list[ExtractedImage],
) -> dict[str, ExtractedImage | None]:
    """
    抽出済み画像をヒーロー候補・下部写真候補・間取り図候補に振り分ける。
    ユーザの最終確認がない場合のデフォルト推定値として使う。

    Returns:
        {
            "hero": 最大の非間取り図写真,
            "bottom_1": 2番目,
            "bottom_2": 3番目,
            "floor_plan": 間取り図候補(白率最大),
        }
    """
    photos = [im for im in images if im.is_likely_photo]
    plans = [im for im in images if im.is_likely_floor_plan]

    # 写真は「白率の低い順」(コンテンツが濃い=実写真ほど低い)→面積大きい順
    photos.sort(key=lambda x: (x.white_ratio, -x.width * x.height))
    # 間取り図は白率最大を優先
    plans.sort(key=lambda x: x.white_ratio, reverse=True)

    return {
        "hero": photos[0] if len(photos) >= 1 else None,
        "bottom_1": photos[1] if len(photos) >= 2 else None,
        "bottom_2": photos[2] if len(photos) >= 3 else None,
        "floor_plan": plans[0] if plans else None,
    }


if __name__ == "__main__":
    import sys

    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/Users/matchan/Downloads/CREST TAPP金山_1301_20260423192046.pdf"
    )
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("samples/_extracted")
    imgs = extract_images(pdf_path, out_dir)
    print(f"Extracted {len(imgs)} images to {out_dir}/")
    for im in imgs:
        tags = []
        if im.is_likely_photo:
            tags.append("PHOTO")
        if im.is_likely_floor_plan:
            tags.append("PLAN")
        tag_str = f" [{','.join(tags)}]" if tags else ""
        print(
            f"  {im.path.name}: {im.width}x{im.height}, "
            f"white={im.white_ratio:.2%}{tag_str}"
        )
    print()
    classification = classify_for_poster(imgs)
    print("Suggested classification:")
    for role, im in classification.items():
        print(f"  {role}: {im.path.name if im else 'NONE'}")
