#!/usr/bin/env python3
"""
Qwen VL inference with selectable backend + parameterized dataset paths.

Backends:
- Nebius API (OpenAI-compatible, e.g., Qwen/Qwen2.5-VL-72B-Instruct)
- Local Ollama (e.g., qwen2.5vl:7b)

I/O & behavior:
- Reads image rel-paths from test.txt and intersects with TOPK_JSON.
- ALWAYS prints the first prompt (with optional token diagnostics).
- Writes CSV with columns: Image_path, MODEL_NAME, prompt, response
"""

import os
import base64
import time
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures
from PIL import Image
from pathlib import Path
import argparse
from datetime import datetime
import shutil

# Optional token diagnostics
try:
    import tiktoken
    _HAS_TIKTOKEN = True
    try:
        _ENC = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:
    _HAS_TIKTOKEN = False
    def count_tokens(text: str) -> int:
        return -1

load_dotenv()

# === Defaults (can be overridden by CLI / env) ===
DEF_IMAGE_DIR     = os.getenv("IMAGE_DIR",     "../data/semi-aves/")
DEF_IMAGE_PATHS   = os.getenv("IMAGE_PATHS",   "../data/semi-aves/test.txt")
DEF_TOPK_JSON     = os.getenv("TOPK_JSON",     "../data/semi-aves/topk/swift_stage3_vitb32_openclip_laion400m_semi-aves_16_1_topk_test_predictions.json")
DEF_OUTPUT_CSV    = os.getenv("OUTPUT_CSV",    "../mllm_output/qwen_zeroshot_all200_explanation.csv")
DEF_ERROR_FILE    = os.getenv("ERROR_FILE",    "../error_logs/qwen_error_log_zeroshot_all200_explanation.txt")
DEF_TAXONOMY_JSON = os.getenv("TAXONOMY_JSON", "../data/semi-aves/semi-aves_metrics-LAION400M-taxonomy-enriched.json")

DEF_API_MODEL     = os.getenv("API_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
DEF_API_BASE      = os.getenv("API_BASE",  "https://api.studio.nebius.com/v1/")

THROTTLE_SEC      = float(os.getenv("THROTTLE_SEC", "0.5"))
TIMEOUT_SEC       = int(os.getenv("TIMEOUT_SEC", "60"))

DESIRED_COLUMNS   = ["Image_path", "MODEL_NAME", "prompt", "response"]

# === CLI args ===
def parse_args():
    p = argparse.ArgumentParser(description="Qwen VL inference (Nebius API or local Ollama)")
    # Backend selection
    p.add_argument("--backend", choices=["nebius", "ollama"],
                   default=os.getenv("QWEN_BACKEND", "nebius"),
                   help="Inference backend: 'nebius' (default) or 'ollama'")
    p.add_argument("--api-model", default=DEF_API_MODEL,
                   help="Nebius/OpenAI model name (when --backend=nebius)")
    p.add_argument("--api-base", default=DEF_API_BASE,
                   help="Nebius/OpenAI base URL (when --backend=nebius)")
    p.add_argument("--ollama-model", default=os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b"),
                   help="Ollama model name:tag (when --backend=ollama)")
    p.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
                   help="Ollama server host (when --backend=ollama)")

    # Dataset/config paths
    p.add_argument("--image-dir", default=DEF_IMAGE_DIR, help="Root dir that contains the images (joined with test.txt paths)")
    p.add_argument("--image-paths", default=DEF_IMAGE_PATHS, help="Path to test.txt (lines like 'test/2845.jpg 3 1')")
    p.add_argument("--topk-json", default=DEF_TOPK_JSON, help="REQUIRED predictions JSON (intersected with test.txt)")
    p.add_argument("--output-csv", default=DEF_OUTPUT_CSV, help="Output CSV path")
    p.add_argument("--error-file", default=DEF_ERROR_FILE, help="Error log file path")
    p.add_argument("--taxonomy-json", default=DEF_TAXONOMY_JSON, help="Taxonomy JSON with class names")

    return p.parse_args()

ARGS = parse_args()

# Bind args to variables used below
BACKEND       = ARGS.backend.lower()
API_MODEL     = ARGS.api_model
API_BASE      = ARGS.api_base
OLLAMA_MODEL  = ARGS.ollama_model
OLLAMA_HOST   = ARGS.ollama_host

IMAGE_DIR     = ARGS.image_dir
IMAGE_PATHS   = ARGS.image_paths
TOPK_JSON     = ARGS.topk_json
OUTPUT_CSV    = ARGS.output_csv
ERROR_FILE    = ARGS.error_file
TAXONOMY_JSON = ARGS.taxonomy_json

# === OpenAI-compatible client (Hyperbolic or Nebius) ===
# Wrapper (run_hyperbolic_7b.sh or run_nebius_72b.sh) should export OPENAI_API_KEY.
_api_key = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("HYPERBOLIC_API_KEY")
    or os.getenv("NEBIUS_API_KEY")
)
if BACKEND == "nebius":
    if not _api_key:
        raise RuntimeError("OPENAI_API_KEY (or provider key) not set")
    client = OpenAI(base_url=API_BASE, api_key=_api_key)
else:
    client = None  # ollama path uses its own client


# Ensure output directories exist
Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
Path(ERROR_FILE).parent.mkdir(parents=True, exist_ok=True)

# === Load taxonomy (deterministic species list) ===
with open(TAXONOMY_JSON, "r") as f:
    taxonomy = json.load(f)

classid_to_names = {
    int(cid): (v.get("most_common_name", "Unknown"), v.get("name", "Unknown"))
    for cid, v in taxonomy.items()
}

def build_zero_shot_prompt() -> str:
    species_items = [
        f"{i}. {common} ({sci})"
        for i, (common, sci) in sorted(classid_to_names.items())
    ]
    half = len(species_items) // 2
    species_block = ", ".join(species_items[:half]) + "\n" + ", ".join(species_items[half:])
    return (
        "You are a helpful assistant that identifies bird species from images.\n"
        "What is the common name and scientific name of the bird in this image?\n"
        "Always give the most specific common name and the exact scientific name (genus + species), not a family or a generic name.\n"
        "Respond only in the following format and do not include anything else:\n"
        "Common Name: <your answer here>\n"
        "Scientific Name: <your answer here>"
    )

# === Helpers ===
def read_test_image_relpaths(txt_path: str):
    rels = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rels.append(line.split()[0])
    return rels

def normalize_to_rel(path_abs: str) -> str:
    if not path_abs:
        return ""
    tokens = ["test/", "train/", "val/", "validation/"]
    for tok in tokens:
        pos = path_abs.find(tok)
        if pos != -1:
            return path_abs[pos:]
    try:
        abs_image_dir = os.path.abspath(IMAGE_DIR)
        if path_abs.startswith(abs_image_dir):
            return os.path.relpath(path_abs, abs_image_dir)
    except Exception:
        pass
    return os.path.basename(path_abs)

def encode_image_base64(abs_image_path: str) -> str:
    with open(abs_image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def csv_escape(s: str) -> str:
    s = (s or "").replace('"', '""')
    return f'"{s}"'

def ensure_output_schema():
    """
    If OUTPUT_CSV exists but doesn't match desired schema, add MODEL_NAME and reorder.
    Backup saved as *.bak_YYYYMMDD_HHMMSS.
    """
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        return
    try:
        df = pd.read_csv(OUTPUT_CSV)
    except Exception as e:
        print(f"Warning: could not parse {OUTPUT_CSV} to check schema: {e}")
        return
    cols = list(df.columns)
    if cols == DESIRED_COLUMNS:
        return
    if "MODEL_NAME" not in df.columns:
        df["MODEL_NAME"] = API_MODEL if BACKEND == "nebius" else f"ollama/{OLLAMA_MODEL}"
    for col in ["Image_path", "prompt", "response"]:
        if col not in df.columns:
            df[col] = ""
    df = df[DESIRED_COLUMNS]
    bak = OUTPUT_CSV + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(OUTPUT_CSV, bak)
        print(f"Backed up existing CSV to: {bak}")
    except Exception as e:
        print(f"Warning: could not backup existing CSV: {e}")
    df.to_csv(OUTPUT_CSV, index=False)

# === REQUIRED TOPK_JSON ===
if not os.path.exists(TOPK_JSON):
    raise FileNotFoundError(f"TOPK_JSON is required but not found: {TOPK_JSON}")
with open(TOPK_JSON, "r") as f:
    preds_blob = json.load(f)
if not isinstance(preds_blob, dict) or len(preds_blob) == 0:
    raise RuntimeError(f"TOPK_JSON must be a non-empty dict: {TOPK_JSON}")

preds_by_relpath = {}
for _, entry in preds_blob.items():
    img_abs = str(entry.get("image_path", "")).strip()
    if not img_abs:
        continue
    rel = normalize_to_rel(img_abs)
    if rel:
        preds_by_relpath[rel] = entry

# === Intersect JSON & test.txt ===
rel_paths_txt  = set(read_test_image_relpaths(IMAGE_PATHS))
rel_paths_json = set(preds_by_relpath.keys())
rel_image_paths = sorted(rel_paths_txt & rel_paths_json)

# Log mismatches
missing_in_json = sorted(rel_paths_txt - rel_paths_json)
missing_in_txt  = sorted(rel_paths_json - rel_paths_txt)
if missing_in_json:
    with open(ERROR_FILE, "a") as logf:
        for rp in missing_in_json:
            logf.write(f"{rp}: present in test.txt but missing in TOPK_JSON\n")
if missing_in_txt:
    with open(ERROR_FILE, "a") as logf:
        for rp in missing_in_txt:
            logf.write(f"{rp}: present in TOPK_JSON but missing in test.txt\n")

if len(rel_image_paths) == 0:
    raise RuntimeError(
        "No overlapping images between IMAGE_PATHS and TOPK_JSON after normalization. "
        "Check path roots and normalization rules."
    )

# === ALWAYS print the first prompt (no pause) ===
first_prompt = build_zero_shot_prompt()
print("\n====== FIRST ZERO-SHOT PROMPT SAMPLE ======")
print(first_prompt)
if _HAS_TIKTOKEN:
    print("\n====== TOKEN DIAGNOSTIC ======")
    print(f"Prompt length (chars): {len(first_prompt)}")
    print(f"Prompt token count: {count_tokens(first_prompt)}")
print("===========================================\n")

# === Output schema & de-dup ===
ensure_output_schema()
seen = set()
if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
    try:
        existing = pd.read_csv(OUTPUT_CSV)
        if "Image_path" in existing.columns:
            seen = set(existing["Image_path"].astype(str).tolist())
    except Exception as e:
        print(f"Warning: could not read {OUTPUT_CSV} for dedup; proceeding without it. Error: {e}")

header_written = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0

# === Unified model caller (Nebius API or local Ollama) ===
def call_model(prompt: str, base64_image: str) -> tuple[str, str]:
    """
    Returns (answer_text, model_label_for_csv).
    """
    if BACKEND == "ollama":
        import ollama  # local-only dependency
        client_ol = ollama.Client(host=OLLAMA_HOST)
        res = client_ol.chat(
            model=ARGS.ollama_model,
            messages=[{"role": "user", "content": prompt, "images": [base64_image]}],
            options={"temperature": 0.4, "num_predict": 300},
            keep_alive="30m",
        )
        answer = res["message"]["content"].strip()
        return answer, f"ollama/{ARGS.ollama_model}"

    # Default: Nebius (OpenAI-compatible)
    resp = client.chat.completions.create(
        model=API_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }],
        temperature=0.4,
        max_tokens=300,
    )
    answer = resp.choices[0].message.content.strip()
    return answer, API_MODEL

# === Inference Loop ===
with open(OUTPUT_CSV, "a", newline="") as out_f:
    if not header_written:
        out_f.write(",".join(DESIRED_COLUMNS) + "\n")
        header_written = True

    for idx, rel_path in enumerate(tqdm(rel_image_paths, total=len(rel_image_paths))):
        csv_image_path = rel_path  # store relative path for uniformity
        if csv_image_path in seen:
            continue

        abs_image_path = os.path.join(IMAGE_DIR, rel_path)

        if not os.path.exists(abs_image_path):
            print(f"[{idx}] Missing image: {abs_image_path}")
            with open(ERROR_FILE, "a") as logf:
                logf.write(f"{csv_image_path}: missing image at {abs_image_path}\n")
            continue

        # Corruption check (close handle with context manager)
        try:
            with Image.open(abs_image_path) as im:
                im.verify()
        except Exception:
            print(f"[{idx}] Corrupted image file: {csv_image_path}")
            with open(ERROR_FILE, "a") as logf:
                logf.write(f"{csv_image_path}: corrupted image\n")
            continue

        try:
            b64 = encode_image_base64(abs_image_path)
            prompt = first_prompt  # identical prompt for all images

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(call_model, prompt, b64)
                answer, model_label = future.result(timeout=TIMEOUT_SEC)

            out_f.write(
                f'{csv_escape(csv_image_path)},'
                f'{csv_escape(model_label)},'
                f'{csv_escape(prompt)},'
                f'{csv_escape(answer)}\n'
            )
            out_f.flush()
            seen.add(csv_image_path)

            time.sleep(THROTTLE_SEC)

        except concurrent.futures.TimeoutError:
            print(f"[{idx}] Timeout while calling model for {csv_image_path}")
            with open(ERROR_FILE, "a") as logf:
                logf.write(f"{csv_image_path}: timeout\n")
            continue

        except Exception as e:
            print(f"[{idx}] Error with image {csv_image_path}: {e}")
            with open(ERROR_FILE, "a") as logf:
                logf.write(f"{csv_image_path}: {str(e)}\n")
            continue

print("Done.")
