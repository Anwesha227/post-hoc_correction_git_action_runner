#!/usr/bin/env bash
set -euo pipefail

# Wrapper: calls run_qwen.sh with fixed arguments (Ollama backend)
# Edit BASE_MODEL below to change the backbone tag used in output/error filenames.
# Don't forget: chmod +x run_ollama_7b.sh, chmod +x run_qwen.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${SCRIPT_DIR}/run_qwen.sh"   # generic launcher

# >>> EDIT THIS LINE AS NEEDED <<<
BASE_MODEL="vitb32_openclip_laion400m"   # e.g., dinov2, openclip, resnet, etc.

# Ollama config (can override host via env)
OLLAMA_MODEL="qwen2.5vl:7b"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

# Output + error paths that include BASE_MODEL (match Nebius naming style)
OUTPUT_CSV="${SCRIPT_DIR}/../mllm_output/ollama_${BASE_MODEL}_qwen2.5vl_7b_qwen_top5_sci_withconf.csv"
ERROR_LOG="${SCRIPT_DIR}/../mllm_output/ollama_${BASE_MODEL}_qwen2.5vl_7b_qwen_top5_sci_withconf_errors.txt"

# TOPK JSON — keep identical to Nebius wrapper unless you truly vary it per base model
TOPK_JSON="${SCRIPT_DIR}/../../../data/semi-aves/topk/fewshot_finetune_vitb32_openclip_laion400m_semi-aves_16_1_topk_test_predictions.json"

# Sanity: driver present
[[ -x "$DRIVER" ]] || { echo "Driver missing/not executable: $DRIVER" >&2; exit 1; }

# Optional: quick probe that Ollama is up (fail fast if not)
if command -v curl >/dev/null 2>&1; then
  curl -sSf -m 3 "${OLLAMA_HOST}/api/version" >/dev/null || {
    echo "ERROR: Can't reach Ollama at ${OLLAMA_HOST}. Is 'ollama serve' running?" >&2
    exit 1
  }
fi

# Exec the driver
exec "$DRIVER" \
  --backend ollama \
  --python "${SCRIPT_DIR}/../qwen_top5_sci_withconf.py" \
  --image-dir "${SCRIPT_DIR}/../../../datasets/semi-aves" \
  --image-paths "${SCRIPT_DIR}/../../../data/semi-aves/test.txt" \
  --topk-json "${TOPK_JSON}" \
  --taxonomy-json "${SCRIPT_DIR}/../../../data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json" \
  --ollama-model "${OLLAMA_MODEL}" \
  --ollama-host "${OLLAMA_HOST}" \
  --output-csv "$OUTPUT_CSV" \
  --error-file "$ERROR_LOG"
