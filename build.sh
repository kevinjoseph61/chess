#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

# Download ONNX Runtime WASM files
WASM_DIR="game/static/engine"
WASM_BASE="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist"
curl -sL "$WASM_BASE/ort-wasm-simd-threaded.jsep.mjs" -o "$WASM_DIR/ort-wasm-simd-threaded.jsep.mjs"
curl -sL "$WASM_BASE/ort-wasm-simd-threaded.jsep.wasm" -o "$WASM_DIR/ort-wasm-simd-threaded.jsep.wasm"

python manage.py migrate
python manage.py collectstatic --no-input
