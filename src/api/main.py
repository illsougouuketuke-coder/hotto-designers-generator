"""
HOTTO designers — Streamlit UI

実行:
    uv run streamlit run src/api/main.py

機能:
    - マイソクPDFをアップロード -> 物件情報を自動抽出
    - 抽出データを画面で編集
    - 駅名から街紹介をAIで3案生成、好きな案を選んで配信文章・画像に組み込む
    - 画像をスロット (ヒーロー/下部1/下部2/間取り図) に割り当て
    - レイアウト4種 (monochrome / gallery / architect / mood) を一括生成
    - 配信文章4トーン (architectural / lifestyle / location / mood) をAI生成
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


# ===== Streamlit Cloud 用: Playwright Chromium の自動インストール =====
@st.cache_resource(show_spinner="初回起動中: Playwright Chromium をセットアップ...")
def _ensure_playwright_chromium() -> bool:
    try:
        subprocess.run(
            ["playwright", "install", "chromium"],
            check=True,
            capture_output=True,
            timeout=300,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _get_secret(key: str, default: str = "") -> str:
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


def _check_password() -> bool:
    expected = _get_secret("APP_PASSWORD", "")
    if not expected:
        return True
    if st.session_state.get("auth_ok"):
        return True

    st.title("🔒 HOTTO designers 配信画像ジェネレーター")
    st.caption("ご利用にはパスワードが必要です。")
    pw = st.text_input("パスワード", type="password")
    if pw:
        if pw == expected:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    return False


from src.extractor.parser import PropertyData, parse_property  # noqa: E402
from src.extractor.pdf_images import (  # noqa: E402
    ExtractedImage,
    classify_for_poster,
    extract_images,
)
from src.message.generator import (  # noqa: E402
    PropertyForMessage,
    count_line_chars,
    generate_line_message,
)
from src.enhancer.upscaler import ensure_delivery_resolution  # noqa: E402
from src.renderer.render import render_poster  # noqa: E402
from PIL import Image as PImage  # noqa: E402

ROLES = [
    ("hero", "ヒーロー写真"),
    ("bottom_1", "下部写真1"),
    ("bottom_2", "下部写真2"),
    ("floor_plan", "間取り図"),
]

LAYOUTS = [
    ("monochrome", "Monochrome", "poster_monochrome.html", "白基調・ミニマル・余白多め"),
    ("gallery", "Gallery", "poster_gallery.html", "黒基調・大判ヒーロー・美術館的"),
    ("architect", "Architect", "poster_architect.html", "建築誌風・グリッド・テクニカル"),
    ("mood", "Mood", "poster_mood.html", "全面写真+詩的キャプション・情緒系"),
]

TONE_LABELS = {
    "architectural": "Architectural (建築語り)",
    "lifestyle": "Lifestyle (暮らし提案)",
    "location": "Location (街推し)",
    "mood": "Mood (情緒・詩的)",
}
TONE_ORDER = ("architectural", "lifestyle", "location", "mood")


def _default_catchphrase(data: PropertyData) -> str:
    """HOTTO designers 向けのデフォルトキャッチコピーを生成。"""
    parts = []
    if data.floor:
        parts.append(f"{data.floor}階")
    if data.is_corner:
        parts.append("角部屋")
    line1 = "・".join(parts) if parts else ""

    line2_parts = []
    if data.station and data.walk_minutes:
        line2_parts.append(f"{data.station.replace('駅', '')}徒歩{data.walk_minutes}分")
    if data.layout:
        line2_parts.append(f"のデザイナーズ{data.layout}")
    line2 = "".join(line2_parts)

    if line1 and line2:
        return f"{line1}、{line2}。"
    elif line2:
        return f"{line2}。"
    elif line1:
        return f"{line1}。"
    return "都心の上質な暮らしを、デザインで。"


def _friendly_api_error(e: Exception) -> str:
    """API例外を事務員向けの日本語メッセージに変換する。"""
    try:
        import anthropic

        if isinstance(e, anthropic.AuthenticationError):
            return "APIキーが正しくありません。サイドバーのAPIキーを確認してください。"
        if isinstance(e, anthropic.RateLimitError):
            return "APIの利用制限に達しました。1〜2分おいてから再実行してください。"
        if isinstance(e, anthropic.APIConnectionError):
            return "ネットワークに接続できませんでした。回線を確認して再実行してください。"
        if isinstance(e, anthropic.APIStatusError):
            code = getattr(e, "status_code", None)
            if code == 402:
                return "APIの残高が不足している可能性があります。console.anthropic.com で確認してください。"
            if code == 529:
                return "AIサーバーが混雑しています。少し待ってから再実行してください。"
    except ImportError:
        pass
    return f"エラー詳細: {e}"


def _ensure_workdir() -> Path:
    if "workdir" not in st.session_state:
        st.session_state.workdir = Path(tempfile.mkdtemp(prefix="hotto_designers_"))
    return st.session_state.workdir


def _process_pdf(pdf_path: Path, api_key: str = ""):
    workdir = _ensure_workdir()
    data = parse_property(pdf_path)
    images = extract_images(pdf_path, workdir / "imgs")

    # API キーがあれば Claude Vision で分類 (洗面所ヒーロー事故の防止)。
    # 失敗時は従来の白率ヒューリスティックにフォールバック。
    classification = None
    if api_key and images:
        try:
            from src.extractor.vision_classify import classify_images_with_vision
            classification = classify_images_with_vision(images, api_key=api_key)
        except Exception:
            classification = None
    if classification is None:
        classification = classify_for_poster(images)
    return data, images, classification


def _wrap_as_extracted(image_path: Path, is_plan: bool = False) -> ExtractedImage:
    """任意の画像ファイルを ExtractedImage に包む (画像経路・切り出し画像用)。"""
    with PImage.open(image_path) as img:
        w, h = img.size
    return ExtractedImage(
        path=image_path,
        width=w,
        height=h,
        file_size=image_path.stat().st_size,
        white_ratio=0.0,
        is_likely_floor_plan=is_plan,
        is_likely_photo=not is_plan,
    )


def _process_image(image_path: Path, api_key: str):
    """マイソク画像 (.png/.jpg等) から AI で物件情報を抽出。

    (1) 物件情報を AI 抽出 → (2) マイソクから写真・間取り図を自動切り出し
    → (3) 切り出し写真を Claude Vision で4スロットに割り当て。
    切り出しが2枚未満なら従来どおりマイソク全体画像を全スロットに入れる。

    Returns:
        (data, images, classification, crop_ok)
        crop_ok=False のときは呼び出し側で手動差し替えを案内すること。
    """
    from src.extractor.image_extractor import extract_property_from_image

    workdir = _ensure_workdir()
    data = extract_property_from_image(image_path, api_key=api_key)

    whole = _wrap_as_extracted(image_path)

    # マイソク全体から写真・間取り図の領域を自動切り出し (失敗しても続行)
    crops: list[tuple[Path, str]] = []
    try:
        from src.extractor.mysoku_crop import crop_photos_from_mysoku
        crops = crop_photos_from_mysoku(image_path, workdir / "crops", api_key=api_key)
    except Exception:
        crops = []

    crop_images = [
        _wrap_as_extracted(p, is_plan=(cat == "floor_plan")) for p, cat in crops
    ]

    if len(crop_images) >= 2:
        # 切り出し成功 → Vision で4スロットに割り当て (失敗時はヒューリスティック)
        try:
            from src.extractor.vision_classify import classify_images_with_vision
            classification = classify_images_with_vision(crop_images, api_key=api_key)
        except Exception:
            classification = classify_for_poster(crop_images)
        # マイソク全体も手動選択肢として残す
        images = crop_images + [whole]
        return data, images, classification, True

    # 切り出し失敗 → 従来どおり全スロットにマイソク全体
    images = [whole] + crop_images
    classification = {
        "hero": whole,
        "bottom_1": whole,
        "bottom_2": whole,
        "floor_plan": whole,
    }
    return data, images, classification, False


def main() -> None:
    st.set_page_config(
        page_title="HOTTO designers 配信画像ジェネレーター",
        page_icon="🏛️",
        layout="wide",
    )

    if not _check_password():
        return

    _ensure_playwright_chromium()

    st.title("HOTTO designers 配信画像ジェネレーター")
    st.caption("デザイナーズ物件のマイソクPDFから、配信画像と文章を生成します。")

    # ===== サイドバー: API キー =====
    secret_key = _get_secret("ANTHROPIC_API_KEY", "")
    with st.sidebar:
        st.subheader("🔑 Anthropic API キー")
        if secret_key:
            st.success("✓ サーバー側で設定済み")
            active_key = secret_key
        else:
            st.caption(
                "console.anthropic.com の sk-ant-api03-... を貼ってください。"
                "未設定でも画像生成・テンプレ文章は動作します。"
            )
            sidebar_key = st.text_input(
                "API キー",
                value=st.session_state.get("api_key_override", ""),
                type="password",
                placeholder="sk-ant-api03-...",
            )
            st.session_state.api_key_override = sidebar_key
            active_key = sidebar_key.strip()
            if active_key:
                st.success("✓ API キー設定済み — 文章4案 & 街紹介AI生成が使えます")
            else:
                st.info("API キー未設定 — テンプレ単案・街紹介手入力のみ")
    st.session_state.active_api_key = active_key
    has_api_key = bool(active_key)

    workdir = _ensure_workdir()

    # ===== Step 1: マイソクをアップロード (PDF または 画像) =====
    st.header("1. マイソクをアップロード (PDF または 画像)")
    st.caption(
        "PDF・PNG・JPG いずれも受け付けます。"
        "**画像 (スクショ等) の場合は AI で物件情報を読み取るため、サイドバーでAPIキーの設定が必要です。**"
    )
    uploaded = st.file_uploader(
        "ファイルを選択",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
    )
    if not uploaded:
        st.info("マイソクのPDFか画像をアップロードすると、物件情報が自動抽出されます。")
        return

    file_bytes = uploaded.getbuffer().tobytes()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    force_reparse = st.button("🔄 強制再解析")

    if force_reparse or st.session_state.get("last_hash") != file_hash:
        src_path = workdir / uploaded.name
        src_path.write_bytes(file_bytes)
        suffix = uploaded.name.lower().rsplit(".", 1)[-1]
        try:
            if suffix == "pdf":
                spinner_msg = (
                    "PDFを解析し、写真をAIが選定中..."
                    if active_key
                    else "PDFを解析中..."
                )
                with st.spinner(spinner_msg):
                    data, images, classification = _process_pdf(
                        src_path, api_key=active_key
                    )
            else:
                if not active_key:
                    st.error(
                        "画像からの物件情報抽出にはAPIキーが必要です。"
                        "サイドバーで Anthropic API キーを設定してください。"
                    )
                    return
                with st.spinner(
                    "画像をAI(Claude Vision)で解析中..."
                    " (物件情報の読み取り + 写真の自動切り出し、約30-60秒)"
                ):
                    data, images, classification, crop_ok = _process_image(
                        src_path, active_key
                    )
                if crop_ok:
                    st.info(
                        "✂️ マイソクから写真・間取り図を自動で切り出し、"
                        "AIが各スロットに割り当てました。"
                        "Step 4 で確認し、必要なら差し替えてください。"
                    )
                else:
                    st.info(
                        "📷 写真の自動切り出しが十分にできなかったため、"
                        "写真スロットには全て同じマイソク画像が入っています。"
                        "Step 4 で手動で差し替えてください。"
                    )
        except Exception as e:
            st.error(f"ファイルの解析に失敗しました — {_friendly_api_error(e)}")
            return

        st.session_state.last_hash = file_hash
        st.session_state.pdf_path = str(src_path)
        st.session_state.data = data
        st.session_state.images = images
        st.session_state.classification = {
            k: str(v.path) if v else None for k, v in classification.items()
        }
        for key in (
            "generated_layouts", "generated_messages", "generated_warnings",
            "area_variants", "selected_area",
        ):
            st.session_state.pop(key, None)
        for tone in list(TONE_ORDER) + ["single"]:
            st.session_state.pop(f"text_{tone}", None)
        if force_reparse:
            st.success("再解析完了")

    data: PropertyData = st.session_state.data
    images: list[ExtractedImage] = st.session_state.images

    if data.warnings:
        for w in data.warnings:
            st.warning(w)

    # ===== Step 2: Property data =====
    st.header("2. 物件情報の確認・修正")
    st.caption(
        "PDFから読み取った値です。間違いがあれば直接編集してください。"
        "**家賃・共益費は法的にも重要なので必ず目視確認してください。**"
    )
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("物件名", value=data.name)
        room = st.text_input("号室", value=data.room)
        floor = st.number_input("階数", value=int(data.floor or 1), min_value=1)
        is_corner = st.checkbox("角部屋", value=data.is_corner)
        layout_str = st.text_input("間取り", value=data.layout)
        area_sqm = st.number_input(
            "専有面積(㎡)", value=float(data.area_sqm or 0.0),
            format="%.2f", min_value=0.0,
        )
        pet_allowed = st.checkbox("ペット相談可", value=data.pet_allowed)
    with col2:
        address_short = st.text_input("住所(短縮)", value=data.address_short)
        station = st.text_input("最寄り駅", value=data.station)
        walk_minutes = st.number_input(
            "徒歩(分)", value=int(data.walk_minutes or 0), min_value=0,
        )
        built_year = st.number_input(
            "築年", value=int(data.built_year or 2025),
            min_value=1900, max_value=2100,
        )
        built_month = st.number_input(
            "築月", value=int(data.built_month or 1),
            min_value=1, max_value=12,
        )
        rent = st.number_input(
            "家賃(円)", value=int(data.rent or 0), min_value=0, step=100,
        )
        common_fee = st.number_input(
            "共益費(円)", value=int(data.common_fee or 0), min_value=0, step=100,
        )

    # ===== Step 3: 街紹介 =====
    st.header("3. 街紹介 (駅名からAI生成 or 手入力)")
    st.caption(
        "デザイナーズ物件配信では「立地・街の雰囲気」が主役のひとつです。"
        "駅名からAIで3案生成し、選んでそのまま使うか、編集して使えます。"
    )
    col_a, col_b = st.columns([1, 3])
    with col_a:
        gen_area_clicked = st.button(
            "🪄 街紹介をAI生成",
            disabled=not has_api_key or not station,
            help="駅名からAIで3案生成します。要APIキー。",
        )
    with col_b:
        if not has_api_key:
            st.caption("⚠️ APIキー未設定のため、街紹介は手入力のみ")
        elif not station:
            st.caption("⚠️ 駅名が空のため生成できません")

    if gen_area_clicked:
        try:
            from src.message.area_description import generate_area_descriptions
            with st.spinner("街紹介をAI生成中..."):
                variants = generate_area_descriptions(
                    station=station,
                    hint=address_short,
                    api_key=active_key,
                )
            st.session_state.area_variants = [
                {"label": v.label, "text": v.text} for v in variants
            ]
            st.session_state.selected_area = variants[1].text  # balanced をデフォルト選択
            st.success("3案を生成しました。お好みのものを選んで編集してください。")
        except Exception as e:
            st.error(f"街紹介の生成に失敗 — {_friendly_api_error(e)}")

    if st.session_state.get("area_variants"):
        cols_area = st.columns(len(st.session_state.area_variants))
        for col, v in zip(cols_area, st.session_state.area_variants):
            with col:
                st.markdown(f"**{v['label'].upper()}**")
                st.text_area(
                    v["label"], value=v["text"], height=180,
                    key=f"area_preview_{v['label']}",
                    label_visibility="collapsed",
                )
                if st.button(f"この{v['label']}案を採用", key=f"adopt_{v['label']}"):
                    st.session_state.selected_area = st.session_state[
                        f"area_preview_{v['label']}"
                    ]
                    st.rerun()

    street_desc = st.text_area(
        "街紹介 (画像・文章に使用される最終テキスト)",
        value=st.session_state.get("selected_area", ""),
        height=120,
        help="ここに表示されている内容が、配信画像と文章に組み込まれます。直接編集も可能です。",
    )
    st.session_state.selected_area = street_desc
    if street_desc:
        st.caption(
            "✅ 配信前チェック: 公園名・川名などの固有名詞が実在するか、"
            "この街の実態と合っているかを必ず確認してください(AI生成のため)。"
        )

    # ===== Step 4: Photo selection =====
    st.header("4. 写真の選択")
    image_options = ["(なし)"] + [im.path.name for im in images]
    image_paths = {im.path.name: im.path for im in images}
    classification = st.session_state.classification

    selected_paths: dict[str, Path] = {}
    cols = st.columns(4)
    for col, (role_key, role_label) in zip(cols, ROLES):
        with col:
            st.markdown(f"**{role_label}**")
            default = classification.get(role_key)
            default_name = Path(default).name if default else "(なし)"
            idx = image_options.index(default_name) if default_name in image_options else 0
            selected_name = st.selectbox(
                role_label, image_options, index=idx, key=f"select_{role_key}",
                label_visibility="collapsed",
            )
            uploaded_img = st.file_uploader(
                "または手動アップロード", type=["png", "jpg", "jpeg"],
                key=f"upload_{role_key}",
            )
            if uploaded_img:
                up_path = workdir / f"manual_{role_key}_{uploaded_img.name}"
                up_path.write_bytes(uploaded_img.getbuffer())
                selected_paths[role_key] = ensure_delivery_resolution(
                    up_path, is_floor_plan=(role_key == "floor_plan")
                )
                st.image(str(selected_paths[role_key]), use_container_width=True)
                st.caption("📤 手動アップロード適用")
            elif selected_name != "(なし)":
                selected_paths[role_key] = ensure_delivery_resolution(
                    image_paths[selected_name], is_floor_plan=(role_key == "floor_plan")
                )
                st.image(str(selected_paths[role_key]), use_container_width=True)

    with st.expander(f"PDFから抽出された全 {len(images)} 枚を確認"):
        if not images:
            st.write("画像が抽出できませんでした。")
        else:
            thumb_cols = st.columns(5)
            for i, im in enumerate(images):
                with thumb_cols[i % 5]:
                    tags = []
                    if im.is_likely_photo:
                        tags.append("📷")
                    if im.is_likely_floor_plan:
                        tags.append("📐")
                    caption = f"{im.path.name}\n{im.width}x{im.height} {' '.join(tags)}"
                    st.image(str(im.path), caption=caption, use_container_width=True)

    # ===== Step 5: Caption + 配信メタ =====
    st.header("5. キャッチコピー・配信番号")
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        initial_cost = st.number_input(
            "初期費用(円)", value=0, min_value=0, step=1000,
            help="HOTTOデザイナーズでは初期費用は基本前面に出しません。0でも可。",
        )
    with col2:
        campaign_no = st.number_input("配信番号", value=1, min_value=1, max_value=999)
    with col3:
        tmp_data = PropertyData(
            name=name, room=room, floor=floor, is_corner=is_corner,
            address_short=address_short, station=station,
            walk_minutes=walk_minutes, layout=layout_str, area_sqm=area_sqm,
            built_year=built_year, built_month=built_month,
            rent=rent, common_fee=common_fee, pet_allowed=pet_allowed,
        )
        catchphrase = st.text_input(
            "キャッチコピー",
            value=_default_catchphrase(tmp_data),
            help="画像のメインコピーとして使われます",
        )

    # ===== Step 6: Generate =====
    st.header("6. 生成")
    if has_api_key:
        st.caption("✓ API キー設定済み — 文章は AI 4案生成されます")
    else:
        st.caption("⚠️ API キー未設定 — 文章はテンプレ1案のみ")

    layout_label_map = {k: label for k, label, _, _ in LAYOUTS}
    selected_layouts = st.multiselect(
        "生成するレイアウト",
        options=[k for k, _, _, _ in LAYOUTS],
        default=[k for k, _, _, _ in LAYOUTS],
        format_func=lambda k: layout_label_map[k],
        help="不要なレイアウトを外すと生成が速くなります(1案あたり約10秒)。",
    )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        gen_images_clicked = st.button(
            f"🖼 画像を生成 ({len(selected_layouts)}案)",
            type="primary",
            disabled=not selected_layouts,
        )
    with col_g2:
        gen_messages_clicked = st.button("✍️ 文章を生成")
    st.caption("画像と文章は別々に生成・やり直しできます(キャッチコピーだけ直したい時は画像のみ再生成)。")

    if gen_images_clicked:
        missing = [label for key, label in ROLES if key not in selected_paths]
        if missing:
            st.error(
                f"以下の写真が未選択です: {', '.join(missing)} — ↑ Step 4 で選択してください"
            )
            return
        if not name or not room:
            st.error("物件名と号室は必須です — ↑ Step 2 を確認してください")
            return
        if rent <= 0:
            st.error("家賃が0円です。抽出に失敗している可能性があるため Step 2 を確認してください。")
            return
        if len({str(p) for p in selected_paths.values()}) == 1:
            st.warning(
                "⚠️ 4スロットすべてが同じ画像です。"
                "マイソク全体がそのまま4箇所に使われます。意図していない場合は Step 4 で差し替えてください。"
            )

        layouts_out: dict[str, Path] = {}
        todo = [(k, label, t) for k, label, t, _ in LAYOUTS if k in selected_layouts]
        progress = st.progress(0.0, text="レイアウトを描画中...")
        for i, (layout_key, layout_label, template_name) in enumerate(todo):
            progress.progress(
                i / len(todo), text=f"{i + 1}/{len(todo)} {layout_label} を描画中..."
            )
            out_path = workdir / f"poster_{layout_key}_{room}_{file_hash[:6]}.png"
            render_poster(
                property_name=name,
                room_number=room,
                floor=floor,
                is_corner=is_corner,
                address_short=address_short,
                station=station,
                walk_minutes=walk_minutes,
                layout=layout_str,
                area_sqm=area_sqm,
                built_year=built_year,
                built_month=built_month,
                rent=rent,
                common_fee=common_fee,
                pet_allowed=pet_allowed,
                initial_cost=initial_cost,
                campaign_no=campaign_no,
                catchphrase=catchphrase,
                hero_photo_path=selected_paths["hero"],
                bottom_photo_1_path=selected_paths["bottom_1"],
                bottom_photo_2_path=selected_paths["bottom_2"],
                floor_plan_path=selected_paths["floor_plan"],
                output_path=out_path,
                template_dir=ROOT / "templates",
                template_name=template_name,
                extras={"street_desc": street_desc},
            )
            layouts_out[layout_key] = out_path
        progress.progress(1.0, text="描画完了")
        st.session_state.generated_layouts = {k: str(v) for k, v in layouts_out.items()}
        st.session_state.generated_meta = {"name": name, "room": room}

    if gen_messages_clicked:
        if not name or not room:
            st.error("物件名と号室は必須です — ↑ Step 2 を確認してください")
            return
        # 前回の文章編集内容をリセット(新規生成を編集値で上書きしないため)
        for tone in list(TONE_ORDER) + ["single"]:
            st.session_state.pop(f"text_{tone}", None)

        message_warnings: list[str] = []
        messages: dict[str, str] = {}
        prop_for_msg = PropertyForMessage(
            name=name, room=room, station=station,
            walk_minutes=walk_minutes, layout=layout_str,
            area_sqm=area_sqm, rent=rent, street_desc=street_desc,
        )
        use_ai = has_api_key
        if use_ai:
            with st.spinner("文章4案をClaude APIで生成中..."):
                try:
                    from src.message.generator_ai import generate_variants
                    variants, warnings = generate_variants(
                        properties=[prop_for_msg],
                        street_desc=street_desc,
                        api_key=active_key,
                    )
                    messages = {v.tone: v.text for v in variants}
                    message_warnings = warnings
                except Exception as e:
                    st.error(
                        f"AI生成に失敗したためテンプレ版を表示します — {_friendly_api_error(e)}"
                    )
                    use_ai = False
        if not use_ai or not messages:
            try:
                template_msg = generate_line_message(
                    properties=[prop_for_msg], street_desc=street_desc,
                )
                messages = {"architectural": template_msg}
            except ValueError as e:
                message_warnings.append(str(e))
                messages = {}

        st.session_state.generated_messages = messages
        st.session_state.generated_warnings = message_warnings
        st.session_state.generated_meta = {"name": name, "room": room}

    # ===== Display generated results =====
    has_layouts = "generated_layouts" in st.session_state
    has_messages = "generated_messages" in st.session_state
    if not has_layouts and not has_messages:
        return

    meta = st.session_state.get("generated_meta", {})

    if has_layouts:
        layouts_paths = {k: Path(v) for k, v in st.session_state.generated_layouts.items()}

        st.divider()
        st.header("7. レイアウトを選択")
        shown = [entry for entry in LAYOUTS if entry[0] in layouts_paths]
        for row_start in range(0, len(shown), 2):
            layout_cols = st.columns(2)
            for col, (layout_key, layout_label, _, layout_desc) in zip(
                layout_cols, shown[row_start: row_start + 2]
            ):
                with col:
                    st.markdown(f"**{layout_label}** — {layout_desc}")
                    st.image(str(layouts_paths[layout_key]), use_container_width=True)

        layout_choice = st.radio(
            "ダウンロード対象レイアウト",
            options=[k for k, _, _, _ in shown],
            format_func=lambda k: layout_label_map.get(k, k),
            horizontal=True,
            key="layout_choice",
        )

        if layout_choice in layouts_paths:
            with open(layouts_paths[layout_choice], "rb") as f:
                st.download_button(
                    f"📥 {layout_choice} をダウンロード(PNG)",
                    f.read(),
                    file_name=f"{meta.get('name', 'poster')}_{meta.get('room', '')}_{layout_choice}.png",
                    mime="image/png",
                    key=f"dl_layout_{layout_choice}",
                )

    if not has_messages:
        return

    messages = st.session_state.generated_messages
    message_warnings = st.session_state.get("generated_warnings", [])

    st.divider()
    st.header("8. 配信文章を選択")
    if message_warnings:
        for w in message_warnings:
            st.warning(w)

    if not messages:
        st.error("文章を生成できませんでした。")
        return

    def _char_badge(chars: int) -> None:
        if chars > 500:
            st.error(f"⚠️ 500文字超過: {chars}/500 — LINEで送信できません。削ってください")
        elif chars > 450:
            st.warning(f"文字数: {chars}/500")
        else:
            st.caption(f"文字数: {chars}/500")

    if len(messages) == 1:
        only_tone, only_text = next(iter(messages.items()))
        edited = st.text_area(
            "文章(この欄で編集できます)", value=only_text, height=400, key="text_single",
        )
        _char_badge(count_line_chars(edited))
        st.download_button(
            "📥 文章をダウンロード(編集内容が反映されます)",
            edited,
            file_name=f"{meta.get('name', 'message')}_{meta.get('room', '')}_message.txt",
            mime="text/plain",
        )
    else:
        displayed = [t for t in TONE_ORDER if t in messages]
        msg_cols = st.columns(len(displayed))
        for col, tone in zip(msg_cols, displayed):
            with col:
                st.markdown(f"**{TONE_LABELS[tone]}**")
                edited = st.text_area(
                    f"{tone}-text", value=messages[tone], height=400,
                    label_visibility="collapsed", key=f"text_{tone}",
                )
                _char_badge(count_line_chars(edited))

        message_choice = st.radio(
            "ダウンロード対象文章",
            options=displayed,
            format_func=lambda k: TONE_LABELS.get(k, k),
            horizontal=True,
            key="message_choice",
        )
        chosen_msg = st.session_state.get(
            f"text_{message_choice}", messages[message_choice]
        )
        st.download_button(
            f"📥 {message_choice} をダウンロード(.txt)",
            chosen_msg,
            file_name=f"{meta.get('name', 'message')}_{meta.get('room', '')}_{message_choice}.txt",
            mime="text/plain",
        )
        st.caption("💡 上の欄で編集した内容がそのまま文字数チェック・ダウンロードに反映されます。LINEへはコピー&ペーストでもOKです。")


if __name__ == "__main__":
    main()
