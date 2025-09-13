#!/usr/bin/env bash
set -euo pipefail

# This script runs Phase 1 of the Knowledge Extraction Pipeline:
# the Wikipedia scraper. It is designed to be run from the 'mllm_inference' directory.

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.." 

# --- Define Paths ---
PYTHON_SCRIPT="${SCRIPT_DIR}/1_scrape_wikipedia.py"
TAXONOMY_JSON="${PROJECT_ROOT}/data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/scraped_descriptions"
LOG_FILE="${PROJECT_ROOT}/data/semi-aves/knowledge_pipeline/scraping_log.txt"


# Ensure the Python script and taxonomy file exist before starting.
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "ERROR: Python script not found at: $PYTHON_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$TAXONOMY_JSON" ]]; then
  echo "ERROR: Taxonomy JSON not found at: $TAXONOMY_JSON" >&2
  exit 1
fi

# Create the output directory and the parent directory for the log file.
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# --- Execute the Scraper ---
echo "================================================="
echo "Starting Phase 1: Wikipedia Scraper"
echo "================================================="
echo "Taxonomy source : $TAXONOMY_JSON"
echo "Output directory: $OUTPUT_DIR"
echo "Log file        : $LOG_FILE"
echo "-------------------------------------------------"

# Call the Python script with all the configured arguments.
python3 "$PYTHON_SCRIPT" \
  --taxonomy-json "$TAXONOMY_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --log-file "$LOG_FILE"

echo "-------------------------------------------------"
echo "Phase 1 complete."
echo "================================================="

