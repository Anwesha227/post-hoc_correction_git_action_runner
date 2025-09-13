#!/usr/bin/env bash
set -euo pipefail

# This script runs Phase 2 of the Knowledge Extraction Pipeline:
# the LLM-powered information extractor.

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.." 

# --- Define Paths ---
PYTHON_SCRIPT="${SCRIPT_DIR}/2_extract_from_text.py"
ENV_FILE="${PROJECT_ROOT}/.env"
SCHEMA_FILE="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/bird_schema.yml"
SCRAPED_DIR="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/scraped_descriptions"
OUTPUT_JSON="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/species_descriptions.json"
LOG_FILE="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/extraction_log.txt"

# --- API Configuration ---
API_MODEL="Qwen/Qwen2.5-72B-Instruct"
API_BASE="https://api.studio.nebius.com/v1/"

# --- Pre-flight Checks ---
if [[ ! -f "$PYTHON_SCRIPT" ]]; then echo "ERROR: Python script not found: $PYTHON_SCRIPT" >&2; exit 1; fi
if [[ ! -f "$SCHEMA_FILE" ]]; then echo "ERROR: Schema file not found: $SCHEMA_FILE" >&2; exit 1; fi
if [[ ! -d "$SCRAPED_DIR" ]]; then echo "ERROR: Scraped text directory not found: $SCRAPED_DIR" >&2; exit 1; fi

# --- MODIFIED: API Key Setup (Standardized and Robust) ---
# Load the .env file to make API keys available as environment variables.
if [[ -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
  echo ">> Loaded .env file from: $ENV_FILE"
else
  echo "ERROR: .env file not found at: $ENV_FILE" >&2; exit 1;
fi

# Explicitly check for the NEBIUS key and export it as the generic OPENAI_API_KEY
# that the Python script will look for. This matches the pattern of the other working scripts.
: "${NEBIUS_API_KEY:?ERROR: NEBIUS_API_KEY is not set in your .env file}"
export OPENAI_API_KEY="$NEBIUS_API_KEY"
echo ">> DEBUG: Exporting API Key starting with '${OPENAI_API_KEY:0:4}'..."
# --- END OF MODIFICATION ---


# --- Execute the Extractor ---
echo "================================================="
echo "Starting Phase 2: Information Extractor"
echo "================================================="
echo "Schema source   : $SCHEMA_FILE"
echo "Text source dir : $SCRAPED_DIR"
echo "Output JSON     : $OUTPUT_JSON"
echo "Log file        : $LOG_FILE"
echo "API Model       : $API_MODEL"
echo "-------------------------------------------------"

python3 "$PYTHON_SCRIPT" \
  --schema-file "$SCHEMA_FILE" \
  --scraped-dir "$SCRAPED_DIR" \
  --output-json "$OUTPUT_JSON" \
  --log-file "$LOG_FILE" \
  --api-model "$API_MODEL" \
  --api-base "$API_BASE"

echo "-------------------------------------------------"
echo "Phase 2 complete."
echo "================================================="