#!/usr/bin/env bash
set -euo pipefail

# A generic launcher script for run_inference.py.
# Its job is to parse arguments, validate them, and construct the correct
# command to execute the Python script for different backends.

usage() {
  cat <<'EOF'
Usage:
  run_qwen.sh \
    --backend nebius|hyperbolic|ollama \
    --python PATH_TO_PY \
    --prompt-template TEMPLATE_NAME \
    --prompt-dir DIR \
    --image-dir DIR \
    --taxonomy-json TAXONOMY_JSON \
    [--image-paths LIST_TXT] \
    [--topk-json TOPK_JSON] \
    [--ref-image-dir REF_IMAGE_DIR_PATH] \
    [--api-model NAME] [--api-base URL] [--env-file .env] \
    [--output-csv FILE] [--error-file FILE] \
    [--ollama-host URL --ollama-model NAME] \
    [--dry-run]
EOF
}

# --- Initialize variables ---
BACKEND=""
PY_ENTRY=""
PROMPT_TEMPLATE=""
PROMPT_DIR=""
IMAGE_DIR=""
IMAGE_PATHS=""
TOPK_JSON=""
TAXONOMY_JSON=""
REF_IMAGE_DIR=""
API_MODEL=""
API_BASE=""
ENV_FILE=""
OUTPUT_CSV=""
ERROR_FILE=""
OLLAMA_HOST=""
OLLAMA_MODEL=""
EXTRA_FLAGS=""

# --- Parse command-line arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2;;
    --python) PY_ENTRY="$2"; shift 2;;
    --prompt-template) PROMPT_TEMPLATE="$2"; shift 2;;
    --prompt-dir) PROMPT_DIR="$2"; shift 2;;
    --image-dir) IMAGE_DIR="$2"; shift 2;;
    --image-paths) IMAGE_PATHS="$2"; shift 2;;
    --topk-json) TOPK_JSON="$2"; shift 2;;
    --taxonomy-json) TAXONOMY_JSON="$2"; shift 2;;
    --ref-image-dir) REF_IMAGE_DIR="$2"; shift 2;;
    --api-model) API_MODEL="$2"; shift 2;;
    --api-base) API_BASE="$2"; shift 2;;
    --env-file) ENV_FILE="$2"; shift 2;;
    --output-csv) OUTPUT_CSV="$2"; shift 2;;
    --error-file) ERROR_FILE="$2"; shift 2;;
    --ollama-host) OLLAMA_HOST="$2"; shift 2;;
    --ollama-model) OLLAMA_MODEL="$2"; shift 2;;
    --dry-run) EXTRA_FLAGS+=" --dry-run"; shift 1;;
    *) echo "Unknown option: $1"; usage; exit 2;;
  esac
done

# --- Validate required arguments ---
[[ -n "$BACKEND" && -n "$PY_ENTRY" && -n "$PROMPT_TEMPLATE" && -n "$PROMPT_DIR" && -n "$IMAGE_DIR" && -n "$TAXONOMY_JSON" ]] || {
  echo "ERROR: missing required args"; usage; exit 1; }
[[ -f "$PY_ENTRY" ]] || { echo "ERROR: Python script not found: $PY_ENTRY" >&2; exit 1; }

# --- Load .env file if provided ---
if [[ -n "${ENV_FILE:-}" ]]; then
  if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
    echo ">> Loaded env from: $ENV_FILE"
  else
    echo "ERROR: --env-file not found: $ENV_FILE" >&2; exit 1;
  fi
fi

# --- Construct base command for Python script ---
CMD=(
  python "$PY_ENTRY"
  "--prompt-template" "$PROMPT_TEMPLATE"
  "--prompt-dir" "$PROMPT_DIR"
  "--image-dir" "$IMAGE_DIR"
  "--taxonomy-json" "$TAXONOMY_JSON"
)

# --- Conditionally add optional arguments ---
if [[ -n "$IMAGE_PATHS" ]]; then CMD+=("--image-paths" "$IMAGE_PATHS"); fi
if [[ -n "$TOPK_JSON" ]]; then CMD+=("--topk-json" "$TOPK_JSON"); fi
if [[ -n "$REF_IMAGE_DIR" ]]; then CMD+=("--ref-image-dir" "$REF_IMAGE_DIR"); fi

# --- Backend-specific logic ---
case "$BACKEND" in
  nebius | hyperbolic)
    BACKEND_NAME_FIRST_LETTER=$(echo "${BACKEND:0:1}" | tr 'a-z' 'A-Z')
    BACKEND_NAME_REST="${BACKEND:1}"
    BACKEND_NAME="${BACKEND_NAME_FIRST_LETTER}${BACKEND_NAME_REST}"
    
    [[ -n "${OPENAI_API_KEY:-}" ]] || { echo "ERROR: OPENAI_API_KEY not set for backend '$BACKEND'"; exit 1; }
    [[ -n "${API_BASE:-$API_BASE}" ]] || { echo "ERROR: --api-base required for backend '$BACKEND'"; exit 1; }
    [[ -n "${API_MODEL:-$API_MODEL}" ]] || { echo "ERROR: --api-model required for backend '$BACKEND'"; exit 1; }

    echo ">> Backend         : $BACKEND_NAME (OpenAI-compatible)"
    echo ">> Python entry    : $PY_ENTRY"
    echo ">> Prompt Template : $PROMPT_TEMPLATE"

    "${CMD[@]}" \
      --backend "$BACKEND" \
      --api-base "${API_BASE:-$API_BASE}" \
      --api-model "${API_MODEL:-$API_MODEL}" \
      --output-csv "${OUTPUT_CSV:-outputs/${BACKEND}/results.csv}" \
      --error-file "${ERROR_FILE:-outputs/${BACKEND}/errors.txt}" \
      $EXTRA_FLAGS
    ;;

  ollama)
    [[ -n "$OLLAMA_HOST" && -n "$OLLAMA_MODEL" ]] || { echo "ERROR: --ollama-host and --ollama-model required"; exit 1; }

    echo ">> Backend         : Ollama"
    echo ">> Python entry    : $PY_ENTRY"
    echo ">> Prompt Template : $PROMPT_TEMPLATE"

    "${CMD[@]}" \
      --backend ollama \
      --ollama-host "$OLLAMA_HOST" \
      --ollama-model "$OLLAMA_MODEL" \
      --output-csv "${OUTPUT_CSV:-outputs/ollama/results.csv}" \
      --error-file "${ERROR_FILE:-outputs/ollama/errors.txt}" \
      $EXTRA_FLAGS
    ;;

  *)
    echo "ERROR: unknown backend '$BACKEND'"; usage; exit 1;;
esac