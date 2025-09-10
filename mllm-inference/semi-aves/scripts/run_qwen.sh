#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_qwen.sh \
    --backend nebius|ollama \
    --python PATH_TO_PY \
    --image-dir DIR \
    --image-paths LIST_TXT \
    --topk-json TOPK_JSON \
    --taxonomy-json TAXONOMY_JSON \
    [--api-model NAME] [--api-base URL] [--env-file .env] \
    [--output-csv FILE] [--error-file FILE] \
    [--ollama-host URL --ollama-model NAME] \
    [--prompt-file FILE]
EOF
}

BACKEND=""
PY_ENTRY=""
IMAGE_DIR=""
IMAGE_PATHS=""
TOPK_JSON=""
TAXONOMY_JSON=""
API_MODEL=""
API_BASE=""
ENV_FILE=""
OUTPUT_CSV=""
ERROR_FILE=""
OLLAMA_HOST=""
OLLAMA_MODEL=""
PROMPT_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --python) PY_ENTRY="$2"; shift 2;;
    --image-dir) IMAGE_DIR="$2"; shift 2;;
    --image-paths) IMAGE_PATHS="$2"; shift 2;;
    --topk-json) TOPK_JSON="$2"; shift 2;;
    --taxonomy-json) TAXONOMY_JSON="$2"; shift 2;;
    --api-model) API_MODEL="$2"; shift 2;;
    --api-base) API_BASE="$2"; shift 2;;
    --env-file) ENV_FILE="$2"; shift 2;;
    --output-csv) OUTPUT_CSV="$2"; shift 2;;
    --error-file) ERROR_FILE="$2"; shift 2;;
    --ollama-host) OLLAMA_HOST="$2"; shift 2;;
    --ollama-model) OLLAMA_MODEL="$2"; shift 2;;
    --prompt-file) PROMPT_FILE="$2"; shift 2;;
    *) echo "Unknown option: $1"; usage; exit 2;;
  esac
done

[[ -n "$BACKEND" && -n "$PY_ENTRY" && -n "$IMAGE_DIR" && -n "$IMAGE_PATHS" && -n "$TOPK_JSON" && -n "$TAXONOMY_JSON" ]] || {
  echo "ERROR: missing required args"; usage; exit 1; }
[[ -f "$PY_ENTRY" ]] || { echo "ERROR: Python script not found: $PY_ENTRY" >&2; exit 1; }

# Load .env for non-key vars (wrappers already export OPENAI_API_KEY)
if [[ -n "${ENV_FILE:-}" ]]; then
  [[ -f "$ENV_FILE" ]] || { echo "ERROR: --env-file not found: $ENV_FILE" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  echo ">> Loaded env from: $ENV_FILE"
fi

case "$BACKEND" in
  nebius)
    [[ -n "${OPENAI_API_KEY:-}" ]] || { echo "ERROR: OPENAI_API_KEY not set (wrapper should export it)"; exit 1; }
    [[ -n "${API_BASE:-}"  ]] || { [[ -n "$API_BASE"  ]] || { echo "ERROR: --api-base required"; exit 1; }; }
    [[ -n "${API_MODEL:-}" ]] || { [[ -n "$API_MODEL" ]] || { echo "ERROR: --api-model required"; exit 1; }; }

    export OPENAI_API_KEY
    export API_BASE
    export API_MODEL

    echo ">> Backend       : OpenAI-compatible (nebius branch)"
    echo ">> API base      : $API_BASE"
    echo ">> API model     : $API_MODEL"
    echo ">> Python entry  : $PY_ENTRY"

    # *********** FIXED: pass hyphenated flags to Python ***********
    python "$PY_ENTRY" \
      --backend nebius \
      --api-base "$API_BASE" \
      --api-model "$API_MODEL" \
      --image-dir "$IMAGE_DIR" \
      --image-paths "$IMAGE_PATHS" \
      --topk-json "$TOPK_JSON" \
      --taxonomy-json "$TAXONOMY_JSON" \
      --output-csv "${OUTPUT_CSV:-outputs/qwen/results.csv}" \
      --error-file "${ERROR_FILE:-outputs/qwen/errors.txt}" \
      ${PROMPT_FILE:+--prompt-file "$PROMPT_FILE"}
    ;;

  ollama)
    [[ -n "$OLLAMA_HOST" && -n "$OLLAMA_MODEL" ]] || { echo "ERROR: --ollama-host and --ollama-model required"; exit 1; }
    echo ">> Backend       : Ollama"
    echo ">> Host          : $OLLAMA_HOST"
    echo ">> Model         : $OLLAMA_MODEL"
    echo ">> Python entry  : $PY_ENTRY"

    # *********** FIXED: pass hyphenated flags to Python ***********
    python "$PY_ENTRY" \
      --backend ollama \
      --ollama-host "$OLLAMA_HOST" \
      --ollama-model "$OLLAMA_MODEL" \
      --image-dir "$IMAGE_DIR" \
      --image-paths "$IMAGE_PATHS" \
      --topk-json "$TOPK_JSON" \
      --taxonomy-json "$TAXONOMY_JSON" \
      --output-csv "${OUTPUT_CSV:-outputs/ollama/results.csv}" \
      --error-file "${ERROR_FILE:-outputs/ollama/errors.txt}" \
      ${PROMPT_FILE:+--prompt-file "$PROMPT_FILE"}
    ;;

  *)
    echo "ERROR: unknown backend '$BACKEND'"; usage; exit 1;;
esac
