#!/bin/bash
# 共有モジュールの同期チェック。
# shokiyasu-generator と hotto-designers-generator は extractor/renderer/enhancer を
# コピーで共有している。片方だけ修正すると、もう片方である日突然壊れるため、
# このスクリプトで差分を確認してから両方へ反映すること。
#
# 使い方: bash scripts/check_shared_sync.sh

A=~/Desktop/shokiyasu-generator
B=~/Desktop/hotto-designers-generator

SHARED=(
  src/extractor/parser.py
  src/extractor/pdf_text.py
  src/extractor/pdf_images.py
  src/extractor/image_extractor.py
  src/extractor/vision_classify.py
  src/extractor/mysoku_crop.py
  src/renderer/render.py
  src/enhancer/upscaler.py
)

status=0
for f in "${SHARED[@]}"; do
  if [ ! -f "$A/$f" ] || [ ! -f "$B/$f" ]; then
    echo "❌ MISSING: $f (片方に存在しません)"
    status=1
  elif diff -q "$A/$f" "$B/$f" > /dev/null 2>&1; then
    echo "✅ OK: $f"
  else
    echo "⚠️  DIFF: $f — 内容が食い違っています。新しい方を確認して両方へコピーしてください"
    status=1
  fi
done

exit $status
