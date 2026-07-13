"""
配信画像のレンダリング処理。
HTMLテンプレートに変数を流し込み、Playwright(Chromium headless)でPNG画像化する。

# SHARED: shokiyasu-generator / hotto-designers-generator の両プロジェクトで同一内容を維持。
# 片方を修正したら scripts/check_shared_sync.sh で差分を確認し、もう片方へもコピーすること。

依存:
  pip install playwright jinja2
  playwright install chromium
"""

from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from playwright.sync_api import sync_playwright


def format_jpy(amount: int) -> str:
    """整数の金額をカンマ区切りの文字列に変換 (例: 67200 -> '67,200')"""
    return f"{amount:,}"


def _is_shinchiku(built_year: int, built_month: int) -> bool:
    """「新築」表示の可否。不動産公正競争規約では建築後1年未満かつ未入居が条件。

    未入居かはデータから判定できないため、築1年未満のみで判定する。
    """
    if not built_year:
        return False
    try:
        built = date(built_year, built_month or 1, 1)
    except ValueError:
        return False
    return (date.today() - built).days < 365


def render_poster(
    *,
    # ===== 物件情報(マイソクから抽出した値をそのまま使う) =====
    property_name: str,           # 例: "CREST TAPP 金山"
    room_number: str,             # 例: "1301"
    floor: int,                   # 例: 13
    is_corner: bool,              # 角部屋かどうか
    address_short: str,           # 例: "名古屋市中川区八熊"
    station: str,                 # 例: "金山駅"
    walk_minutes: int,            # 例: 9
    layout: str,                  # 例: "1K"
    area_sqm: float,              # 例: 29.97
    built_year: int,              # 例: 2025
    built_month: int,             # 例: 8
    rent: int,                    # 例: 68200 (円)
    common_fee: int,              # 例: 20000 (円)
    pet_allowed: bool,            # ペット相談可
    # ===== キャンペーン情報 =====
    initial_cost: int,            # 例: 30000 (円)
    campaign_no: int,             # しょきやすキャンペーン通し番号 例: 02
    catchphrase: str,             # ヒーロー写真下のキャッチコピー
    # ===== 画像パス(絶対パス推奨) =====
    hero_photo_path: Path,
    bottom_photo_1_path: Path,
    bottom_photo_2_path: Path,
    floor_plan_path: Path,
    # ===== 出力 =====
    output_path: Path,
    template_dir: Path = Path("templates"),
    template_name: str = "poster.html",
    # ===== 任意の追加変数 (HOTTOデザイナーズの街紹介などで使用) =====
    extras: dict | None = None,
) -> Path:
    """
    配信画像(PNG, 1080x1380px, 2倍解像度)を生成して output_path に保存。

    Returns:
        Path: 生成された画像のパス
    """

    # 1. Jinja2 でHTMLを生成
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template(template_name)

    html_content = template.render(
        property_name=property_name,
        room_number=room_number,
        floor=floor,
        is_corner=is_corner,
        address_short=address_short,
        station=station,
        walk_minutes=walk_minutes,
        layout=layout,
        area_sqm=area_sqm,
        built_year=built_year,
        built_month=built_month,
        rent_formatted=format_jpy(rent),
        common_fee_formatted=format_jpy(common_fee),
        pet_allowed=pet_allowed,
        initial_cost_formatted=format_jpy(initial_cost),
        campaign_no=f"{campaign_no:02d}",
        catchphrase=catchphrase,
        hero_photo_path=Path(hero_photo_path).absolute().as_uri(),
        bottom_photo_1_path=Path(bottom_photo_1_path).absolute().as_uri(),
        bottom_photo_2_path=Path(bottom_photo_2_path).absolute().as_uri(),
        floor_plan_path=Path(floor_plan_path).absolute().as_uri(),
        shinchiku=_is_shinchiku(built_year, built_month),
        extras=extras or {},
    )

    # 2. 一時HTMLファイルとして保存
    tmp_html = output_path.parent / f"_tmp_{output_path.stem}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_html.write_text(html_content, encoding="utf-8")

    # 3. Playwrightで画像化
    # Streamlit Cloud (非root・1GB RAM の Debian コンテナ) では、サンドボックス無効化と
    # 省メモリフラグが無いと Chromium が起動できず描画に失敗する。ローカルでも無害。
    launch_args = [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--single-process",
        "--no-zygote",
        "--disable-gpu",
    ]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=launch_args)
            page = browser.new_page(
                viewport={"width": 1080, "height": 1400},
                device_scale_factor=2,  # Retina相当の高解像度出力
            )
            page.goto(tmp_html.absolute().as_uri())
            page.wait_for_load_state("networkidle")
            elem = page.locator(".card")
            elem.screenshot(path=str(output_path))
            browser.close()
    finally:
        # 一時HTMLは削除(デバッグ時はコメントアウト推奨)
        if tmp_html.exists():
            tmp_html.unlink()

    return output_path


if __name__ == "__main__":
    # 単体テスト用サンプル
    out = render_poster(
        property_name="CREST TAPP 金山",
        room_number="1301",
        floor=13,
        is_corner=True,
        address_short="名古屋市中川区八熊",
        station="金山駅",
        walk_minutes=9,
        layout="1K",
        area_sqm=29.97,
        built_year=2025,
        built_month=8,
        rent=68200,
        common_fee=20000,
        pet_allowed=True,
        initial_cost=30000,
        campaign_no=2,
        catchphrase="13階・角部屋。金山3路線徒歩圏の高層デザイナーズ1K。",
        hero_photo_path=Path("samples/1301/hero.jpg"),
        bottom_photo_1_path=Path("samples/1301/photo1.jpg"),
        bottom_photo_2_path=Path("samples/1301/photo2.jpg"),
        floor_plan_path=Path("samples/1301/plan.jpg"),
        output_path=Path("output/CREST_TAPP金山_1301.png"),
    )
    print(f"Saved: {out}")
