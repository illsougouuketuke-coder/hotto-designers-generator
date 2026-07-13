# HOTTO designers 配信画像ジェネレーター — セッション引き継ぎノート

新セッションで再開するときは「`~/Desktop/hotto-designers-generator/RESUME.md` を読んで状況を把握して」と伝えれば続きから作業できます。

---

## プロジェクト概要

- **目的**: Hotto社のデザイナーズ物件専用LINEアカウント「HOTTO designers」向けに、賃貸マイソクPDFから配信画像と文章を自動生成
- **しょきやす版との違い**: 主役は「デザイン・雰囲気・立地」。価格訴求や初期費用キャンペーンは前面に出さない
- **場所**: `~/Desktop/hotto-designers-generator/`
- **技術**: Python 3.11 + uv + Streamlit + Playwright + Claude API (Sonnet 4.6)

---

## 起動方法

```bash
cd ~/Desktop/hotto-designers-generator
uv run streamlit run src/api/main.py --server.port 8503
# → http://localhost:8503
```

ポートは 8503 を割り当てています (しょきやす版が 8501/8502 を使うため衝突回避)。

---

## 機能

### レイアウト4種 (白黒基調・シンプル&おしゃれ)
1. **Monochrome** — 白基調・ミニマル・余白多め (写真左+右テキスト)
2. **Gallery** — 黒基調・全面ヒーロー・美術館的
3. **Architect** — 建築誌風・グリッド背景・テクニカルキャプション (FIG.01等)
4. **Mood** — 全面写真+詩的キャプション・情緒系

### 配信文章4トーン (AI生成、Claude Sonnet 4.6)
- **architectural** — 建築・設計の語り口 (面材・天井高・素材の経年)
- **lifestyle** — 暮らしの提案 (朝のコーヒー・夜の読書など1日のシーン)
- **location** — 街・立地推し (street_descを主軸)
- **mood** — 詩的・情緒的 (短文を重ねる文学的トーン)

### 街紹介AI生成 (新規機能)
- 駅名を入力 → AIが3パターン (short / balanced / poetic) で街紹介を生成
- お好みの案を選んで採用 → 配信画像と文章の両方に組み込まれる
- 手入力・編集も可

### 共通機能 (しょきやす版から流用)
- マイソクPDF解析 (RealNetPro + エビス・リビング系)
- PDF画像抽出と4スロット自動分類
- LINE 500文字制限ガード + 数値整合性チェック (物件名・号室)

---

## ファイル構成

```
~/Desktop/hotto-designers-generator/
├── RESUME.md                   # ★このファイル
├── pyproject.toml + uv.lock    # 依存(uv 管理)
├── requirements.txt            # Cloud デプロイ用
├── packages.txt                # Linux apt deps
├── .gitignore
├── src/
│   ├── extractor/              # しょきやす版から流用 (PDFパース・画像抽出)
│   │   ├── parser.py
│   │   ├── pdf_text.py
│   │   └── pdf_images.py
│   ├── enhancer/upscaler.py   # しょきやす版から流用 (未組込)
│   ├── renderer/render.py      # しょきやす版を拡張 (extras dict 対応)
│   ├── message/
│   │   ├── generator.py        # ★HOTTO用テンプレ単案 (新規)
│   │   ├── generator_ai.py     # ★HOTTO用AI4案 (新規)
│   │   └── area_description.py # ★駅名→街紹介AI生成 (新規)
│   └── api/main.py             # ★HOTTO用Streamlit UI (新規)
└── templates/
    ├── poster_monochrome.html  # ★白基調ミニマル
    ├── poster_gallery.html     # ★黒基調ギャラリー
    ├── poster_architect.html   # ★建築誌風
    └── poster_mood.html        # ★情緒系
```

---

## 操作フロー

1. PDFアップロード → 自動抽出
2. 物件情報の確認・修正
3. **街紹介の生成 or 手入力** (新規ステップ) — 駅名から3案生成、選んで採用
4. 写真選択 (4スロット)
5. キャッチコピー編集 + 配信番号
6. 「生成する」ボタン → 4レイアウト × 4文章を一括生成
7. レイアウト選択 + ダウンロード
8. 文章選択 + ダウンロード

---

## しょきやす版との設計上の違い

| 項目 | しょきやす | HOTTOデザイナーズ |
|------|-----------|------------------|
| 訴求軸 | 初期費用が安い | デザイン・雰囲気・立地 |
| トーン | キャンペーン・煽り強め | 上品・控えめ・編集者的 |
| 絵文字 | 多め (✨🏠🌷🔥等) | 控えめ・基本なし |
| ブランド名 | しょきやす | HOTTO designers |
| 4トーン | polite/limited/casual/engagement | architectural/lifestyle/location/mood |
| 街紹介 | なし | 駅名からAI生成 (新規) |
| カラー | カラフル (ベージュ・黒・ポラロイド) | 白黒基調 |
| 価格表示 | 初期費用を巨大表示 | rent と initial_cost を等価で控えめ |

---

## 既知の制約・残課題

- **CLI 未実装**: しょきやす版にはCLIがあるが、HOTTO版は Streamlit UI のみ (必要なら後で追加)
- **アップスケール未組込**: `src/enhancer/upscaler.py` はあるが render パイプラインに未組込
- **複数物件まとめ配信**: 現状は1物件想定。文章生成テンプレは複数対応するが、画像は1物件のみ
- **PDFフォーマット**: しょきやす版と同じパーサー (RealNetPro + エビス・リビング)。HOTTO物件のフォーマットが異なれば parser.py の拡張が必要
- **街紹介の検証**: 駅名からAIが生成する街紹介は事実関係を検証していない (具体店名・数字は出さないルールでハルシネーション抑制)

---

## デプロイ

しょきやす版と同じ手順 (DEPLOY.md 参照) でできるはずだが、未着手。
ローカル運用なら同Wi-Fi共有: `http://192.168.x.x:8503`

---

## 関連プロジェクト

- しょきやす版: `~/Desktop/shokiyasu-generator/` (株式会社I'll の別アカウント用)
- 共通: PDFパーサー・画像抽出・renderer のコア処理
- Hotto税務文脈: `tax-advisor` スキル系 (別領域)

---

最終更新: 2026-05-14 (初版作成)


---

## 2026-05-18 大規模アップデート (サブエージェント3班による改善)

### 追加機能
- **写真のAI自動選定** (`src/extractor/vision_classify.py`) と **マイソク画像からの写真自動切り出し** (`src/extractor/mysoku_crop.py`)。スクショアップで写真・間取り図を自動検出→切り出し→4スロット自動配置。フォールバック完備。検証: `ANTHROPIC_API_KEY=... uv run python scripts/test_vision_pipeline.py`
- **ぼかし背景フィル**: 全4テンプレートで縦横比不問・切れない額装風表示(白黒基調維持のため低彩度化)
- **既存バグ修正**: gallery=物件名が最下部写真に重なる位置バグ / monochrome=価格サブテキストが写真ストリップに食い込むバグ
- **タイポグラフィ強化**: 長い物件名(42文字)への耐性(palt+行クランプ)、価格ブロックのヘアライン等
- **部分再生成**: 画像/文章ボタン分離+レイアウトmultiselect+進捗バー
- **文章編集の反映**: text_area編集が文字数・DLに反映(実質バグ修正)

### コンプライアンス修正
- few-shotの「東京で最も洗練された」→「東京有数の」(No.1系表示の排除)
- `_validate_variants` に最上級表現Lint + 入力にない金額(賃料改変)検出を追加
- 街紹介の採用時に固有名詞の実在確認を促す注意書き
- 新築表示は築1年未満のみ(`render.py` の `shinchiku` 変数)
- APIエラーの日本語化 (`_friendly_api_error`)

### 共有モジュール管理
- extractor/renderer/enhancer は shokiyasu-generator とコピー共有。**片方を修正したら必ず `bash scripts/check_shared_sync.sh` で差分確認して両方へ反映すること**

最終更新: 2026-05-18
