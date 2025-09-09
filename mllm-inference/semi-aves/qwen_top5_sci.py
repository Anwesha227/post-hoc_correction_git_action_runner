#!/usr/bin/env python3
"""
Qwen VL — Top-5 Explanations from TOPK_JSON (Blueprint-aligned)
Output CSV schema (ONLY): image_path, MODEL_NAME, prompt, response

Works with run_qwen.sh (unchanged). Supports --backend nebius|ollama.
"""

import os
import sys
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Nebius (OpenAI-compatible)
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

# Ollama HTTP client (stdlib)
import urllib.request
import urllib.error

load_dotenv()


# ---------------------- IO helpers ----------------------
def encode_image_b64(p: Path) -> str:
    with p.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_test_list(path: Path) -> List[str]:
    out = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(line.split()[0])  # first token = rel path (may include subfolder like "test/28.jpg")
    return out

def load_topk_json_exact(topk_json_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Input shape:
      { "0": {"image_path": "/abs/.../test/28.jpg", "topk_cls": [...], "topk_probs": [...]}, ... }
    Returns map: image_path (as-given) -> {"topk_cls": [...], "topk_probs": [...]}
    """
    with topk_json_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("TOPK_JSON must be a dict keyed by string indices.")
    out = {}
    for _, rec in data.items():
        if not isinstance(rec, dict):
            continue
        img = rec.get("image_path")
        cls = rec.get("topk_cls")
        prb = rec.get("topk_probs")
        if isinstance(img, str) and isinstance(cls, list):
            out[img] = {
                "topk_cls": cls,
                "topk_probs": prb if isinstance(prb, list) else None
            }
    if not out:
        raise ValueError("No usable records found in TOPK_JSON.")
    return out


# ---------------------- Taxonomy & formatting ----------------------
def taxonomy_maps(taxonomy: Dict[str, Any]):
    """
    Build id -> common name and id -> scientific name maps.
    Prefers:
      common: most_common_name | common_name
      scientific: scientific_name | name
    """
    id2common, id2sci = {}, {}
    for k, v in taxonomy.items():
        try:
            cid = int(k)
        except Exception:
            try:
                cid = int(v.get("class_id", k))
            except Exception:
                continue
        common = v.get("most_common_name") or v.get("common_name")
        sci = v.get("scientific_name") or v.get("name")
        id2common[cid] = (str(common) if common else (str(sci) if sci else f"class_{cid}"))
        id2sci[cid] = (str(sci) if sci else id2common[cid])
    return id2common, id2sci

def format_species(name: Optional[str], sci: Optional[str]) -> str:
    name = (name or "").strip()
    sci = (sci or "").strip()
    if not name: return sci or "Unknown"
    if not sci or name == sci: return name
    return f"{name} ({sci})"

def get_confidence(row: pd.Series, k: int) -> Optional[float]:
    col = f"conf{k}"
    if col not in row:
        return None
    try:
        val = float(row[col])
    except Exception:
        return None
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))

def conf_str(p: Optional[float]) -> str:
    return f" [p={p:.2f}]" if p is not None else ""


# ---------------------- Prompt (EXACT blueprint) ----------------------
def build_explanation_prompt(row: pd.Series) -> str:
    """
    Minimal-diff prompt that:
    - Lists 5 candidates with common + scientific names and [p=..] from conf1..conf5
    - Treats ONLY top-1 confidence as a certainty signal (no numeric thresholds);
      Qwen decides if p1 feels strong/weak; visual evidence remains primary.
    """
    lines = []
    for k in range(1, 6):
        species = format_species(row.get(f"pred{k}_name"), row.get(f"pred{k}_sci"))
        #conf = conf_str(get_confidence(row, k))
        lines.append(f"{k}. {species}")

    return (
        "You are a helpful assistant that identifies bird species from images.\n\n"
        "Step 1: Carefully examine the bird in the image and note its distinguishing features "
        "(such as color, shape, size, beak, wings, or habitat).\n\n"
        "Step 2: Compare these features to the following five species:\n\n"
        + "\n".join(lines) + "\n\n"
        "Step 3: Choose the most likely species and explain why it is a better match than the other four.\n\n"
        "Respond in the following format:\n\n"
        "Most Likely: [1, 2, 3, 4, or 5]\n"
        "Explanation: [your explanation here]"
    )


# ---------------------- Backends ----------------------
def call_nebius(api_base: str, api_key: str, model: str, prompt: str, b64_img: str) -> str:
    if OpenAIClient is None:
        raise RuntimeError("openai client library not available")
    client = OpenAIClient(base_url=api_base, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
            ],
        }],
        temperature=0.4,
        max_tokens=900,
    )
    return (resp.choices[0].message.content or "").strip()

def call_ollama(ollama_host: str, model: str, prompt: str, b64_img: str) -> str:
    url = f"{ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [b64_img],
        }],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    msg = body.get("message", {})
    return (msg.get("content") or body.get("response") or "").strip()


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Qwen Top-5 Explanations (TOPK_JSON) — Blueprint schema.")
    # Flags expected from run_qwen.sh
    ap.add_argument("--backend", required=True, choices=["nebius", "ollama"])
    ap.add_argument("--api-model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--api-base", default="https://api.studio.nebius.com/v1/")
    ap.add_argument("--ollama-model", default="")
    ap.add_argument("--ollama-host", default="")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--image-paths", required=True)   # test.txt
    ap.add_argument("--topk-json", required=True)
    ap.add_argument("--taxonomy-json", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--error-file", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print_first_prompt", action="store_true")
    ap.add_argument("--throttle-sec", type=float, default=0.5)
    ap.add_argument("--timeout-sec", type=float, default=60.0)
    args = ap.parse_args()

    # IO paths
    image_dir = Path(args.image_dir)
    test_list = Path(args.image_paths)
    topk_json = Path(args.topk_json)
    taxonomy_json = Path(args.taxonomy_json)
    out_csv = Path(args.output_csv)
    err_file = Path(args.error_file)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    err_file.parent.mkdir(parents=True, exist_ok=True)

    # Backend setup
    backend = args.backend
    if backend == "nebius":
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            print("ERROR: NEBIUS_API_KEY is not set/exported or missing from .env", file=sys.stderr)
            sys.exit(2)
        model_name = args.api_model
    else:  # ollama
        if not args.ollama_host or not args.ollama_model:
            print("ERROR: --ollama-host and --ollama-model are required for backend=ollama", file=sys.stderr)
            sys.exit(2)
        model_name = args.ollama_model

    # Taxonomy maps
    with taxonomy_json.open("r") as f:
        taxonomy = json.load(f)
    id2common, id2sci = taxonomy_maps(taxonomy)

    # Load predictions + test list
    topk_map = load_topk_json_exact(topk_json)
    test_paths = read_test_list(test_list)

    # Build a fast basename index for TOPK entries
    by_base = {Path(p).name: rec for p, rec in topk_map.items()}

    # Materialize rows with pred1..5, names, scinames, conf1..5
    records = []
    for rel in test_paths:
        base = Path(rel).name
        rec = by_base.get(base)
        if rec is None:
            continue

        cls = rec.get("topk_cls") or []
        probs = rec.get("topk_probs") or []

        row = {"image_path": rel}
        for k in range(1, 6):
            cid = cls[k - 1] if k - 1 < len(cls) else None
            conf = probs[k - 1] if k - 1 < len(probs) else None
            cid_int = int(cid) if cid is not None else None
            row[f"pred{k}"] = cid_int
            row[f"pred{k}_name"] = id2common.get(cid_int) if cid_int is not None else None
            row[f"pred{k}_sci"] = id2sci.get(cid_int) if cid_int is not None else None
            row[f"conf{k}"] = float(conf) if conf is not None else None

        records.append(row)

    if not records:
        raise SystemExit("No intersecting samples between test.txt and TOPK_JSON (basename match).")

    df = pd.DataFrame.from_records(records)

    # Always show the first prompt (match blueprint behavior); exit only if flag set
    first_prompt = build_explanation_prompt(df.iloc[0])
    print("=== First Sample Prompt ===\n")
    print(first_prompt)
    print("\n===========================\n")
    if args.print_first_prompt:
        return

    # Resume support
    done_paths = set()
    if out_csv.exists():
        try:
            prev = pd.read_csv(out_csv)
            if "image_path" in prev.columns:
                done_paths = set(prev["image_path"].astype(str).tolist())
        except Exception:
            pass

    # Inference loop
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        rel_path = str(row["image_path"])
        if rel_path in done_paths:
            continue

        # Preserve subfolders from test.txt when locating the image
        rel_norm = rel_path.lstrip("./")
        img_path = Path(rel_norm) if Path(rel_norm).is_absolute() else (image_dir / rel_norm)

        if not img_path.exists():
            print(f"[{idx}] Missing image: {img_path}")
            with err_file.open("a") as logf:
                logf.write(f"{rel_path}: missing image\n")
            continue

        # Validate image
        try:
            Image.open(img_path).verify()
        except Exception:
            print(f"[{idx}] Corrupted image file: {rel_path}")
            with err_file.open("a") as logf:
                logf.write(f"{rel_path}: corrupted image\n")
            continue

        # Build exact blueprint prompt (common + scientific + [p=..])
        prompt = build_explanation_prompt(row)

        # Dry run
        if args.dry_run:
            out_row = {
                "image_path": rel_path,
                "MODEL_NAME": model_name,
                "prompt": prompt,
                "response": "(dry-run)",
            }
            pd.DataFrame([out_row]).to_csv(
                out_csv, mode="a", header=not out_csv.exists(), index=False
            )
            time.sleep(args.throttle_sec)
            continue

        # Call backend
        b64 = encode_image_b64(img_path)
        try:
            if backend == "nebius":
                answer = call_nebius(args.api_base, os.environ["NEBIUS_API_KEY"], model_name, prompt, b64)
            else:
                answer = call_ollama(args.ollama_host, model_name, prompt, b64)
        except Exception as e:
            print(f"[{idx}] Error: {rel_path}: {e}")
            with err_file.open("a") as logf:
                logf.write(f"{rel_path}: {str(e)}\n")
            continue

        # Blueprint output schema (ONLY these 4 columns)
        out_row = {
            "image_path": rel_path,
            "MODEL_NAME": model_name,
            "prompt": prompt,
            "response": answer,
        }
        pd.DataFrame([out_row]).to_csv(
            out_csv, mode="a", header=not out_csv.exists(), index=False
        )

        time.sleep(args.throttle_sec)


if __name__ == "__main__":
    main()
