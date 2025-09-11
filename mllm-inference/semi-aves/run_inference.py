#!/usr/bin/env python3
"""
Unified Qwen VL Inference Script with External Prompt Templates

Supports both single-image (text + 1 image) and multi-image 
(e.g., text + query_image + N reference_images) prompting strategies.
"""

import os
import sys
import json
import time
import base64
import argparse
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None

import urllib.request
import urllib.error

load_dotenv()

# ==============================================================================
# === I/O & Path Helpers =======================================================
# ==============================================================================

def encode_image_b64(p: Path) -> str:
    """Encodes an image file to a base64 string."""
    with p.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_test_list(path: Path) -> List[str]:
    """Reads a list of relative image paths from a text file."""
    out = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line.split()[0])
    return out

def load_topk_json_exact(topk_json_path: Path) -> Dict[str, Dict[str, Any]]:
    """Loads top-k predictions and maps them by their absolute image path."""
    with topk_json_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("TOPK_JSON must be a dict keyed by string indices.")
    out = {}
    for _, rec in data.items():
        if isinstance(rec, dict) and isinstance(rec.get("image_path"), str):
            out[rec["image_path"]] = {"topk_cls": rec.get("topk_cls"), "topk_probs": rec.get("topk_probs")}
    if not out:
        raise ValueError("No usable records found in TOPK_JSON.")
    return out

def normalize_path_for_matching(p: str, image_dir: Path) -> str:
    """Normalizes a path to a consistent relative format for matching."""
    p_str = str(p).strip()
    for split in ["test/", "train/", "val/", "validation/"]:
        if split in p_str:
            return p_str[p_str.find(split):]
    try:
        if Path(p_str).is_absolute() and p_str.startswith(str(image_dir.resolve())):
            return str(Path(p_str).relative_to(image_dir.resolve()))
    except ValueError:
        pass
    return Path(p_str).name

# ==============================================================================
# === Taxonomy Helpers =========================================================
# ==============================================================================

def _safe(s: Any) -> str:
    """Safely converts a value to a stripped string."""
    return (str(s) if s is not None else "").strip()

def build_taxonomy_maps(taxonomy_json_path: Path) -> Dict[str, Any]:
    """Builds comprehensive, fast-lookup taxonomy mappings from the taxonomy JSON file."""
    with taxonomy_json_path.open("r") as f:
        taxonomy = json.load(f)

    maps = {'id2meta': {}, 'id2common': {}, 'id2sci': {}}
    species_items_for_prompt = []

    for k, v in taxonomy.items():
        try:
            cid = int(v.get("class_id", k))
        except (ValueError, TypeError):
            continue

        sci = _safe(v.get("scientific_name")) or _safe(v.get("name"))
        common = _safe(v.get("most_common_name")) or _safe(v.get("common_name"))
        genus = _safe(v.get("genus"))
        family = _safe(v.get("family"))
        final_sci = sci or common or f"class_{cid}"
        final_common = common or sci or f"class_{cid}"

        maps['id2meta'][cid] = {"sci": final_sci, "common": final_common, "genus": genus, "family": family}
        maps['id2common'][cid] = final_common
        maps['id2sci'][cid] = final_sci
        species_items_for_prompt.append((cid, final_common, final_sci))

    species_items_for_prompt.sort(key=lambda x: x[0])
    formatted_list = [f"{cid}. {common} ({sci})" for cid, common, sci in species_items_for_prompt]
    maps['all_species_list'] = ", ".join(formatted_list)
    
    maps['name_to_classid'] = {
        (v.get("most_common_name") or "").lower().strip(): int(k)
        for k, v in taxonomy.items() if (v.get("most_common_name") or "").strip()
    }
    return maps

# ==============================================================================
# === Prompt Generation ========================================================
# ==============================================================================

def load_prompt_templates(prompt_dir: Path) -> Dict[str, str]:
    templates = {}
    if not prompt_dir.exists(): raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")
    for filepath in prompt_dir.glob("*.txt"):
        try: templates[filepath.stem] = filepath.read_text()
        except Exception as e: raise IOError(f"Error reading prompt template {filepath}: {e}")
    if not templates: raise FileNotFoundError(f"No prompt templates (.txt files) found in: {prompt_dir}")
    print(f">> Loaded {len(templates)} prompt templates from {prompt_dir}")
    return templates

def format_multimodal_prompt_for_display(prompt_list: list) -> str:
    output_lines = ["--- [Multimodal Prompt Structure] ---"]
    image_counter = 0
    for item in prompt_list:
        if item.get('type') == 'text': output_lines.append(f"\n[TEXT]:\n{item['text']}")
        elif item.get('type') == 'image_url':
            if image_counter == 0: output_lines.append("  [IMAGE: Query Image Embedded Here]")
            else: output_lines.append(f"  [IMAGE: Reference Candidate {image_counter} Embedded Here]")
            image_counter += 1
    output_lines.append("--- [End of Prompt] ---")
    return "\n".join(output_lines)

def format_prompt_for_csv(prompt_content: Union[str, list]) -> str:
    if not isinstance(prompt_content, list): return prompt_content
    parts_for_csv = []
    image_counter = 0
    for item in prompt_content:
        if item.get('type') == 'text': parts_for_csv.append(item['text'])
        elif item.get('type') == 'image_url':
            if image_counter == 0: parts_for_csv.append("\n[--- IMAGE: Query Image ---]\n")
            else: parts_for_csv.append(f"\n[--- IMAGE: Reference Candidate {image_counter} ---]\n")
            image_counter += 1
    return "\n".join(parts_for_csv)

def _format_species_flat(meta: Dict[str, Any]) -> str:
    sci, common, genus, family = meta.get("sci"), meta.get("common"), meta.get("genus"), meta.get("family")
    if not any([sci, common, genus, family]): return "Unknown"
    parts = []
    if sci: parts.append(sci)
    if common and common != sci: parts.append(f"also known as {common}")
    lineage = []
    if genus: lineage.append(f"genus {genus}")
    if family: lineage.append(f"family {family}")
    if lineage: parts.append("belongs to the " + ", ".join(lineage))
    return ", ".join(parts)

def _format_species_sci(meta: Dict[str, Any]) -> str:
    common, sci = meta.get("common"), meta.get("sci")
    if not common: return sci or "Unknown"
    if not sci or common == sci: return common
    return f"{common} ({sci})"

def _conf_str(p: Optional[float]) -> str:
    if p is None: return ""
    try: return f" [p={max(0.0, min(1.0, float(p))):.2f}]"
    except (ValueError, TypeError): return ""

def build_prompt_top5(row: pd.Series, tax_maps: Dict, templates: Dict[str, str], args: argparse.Namespace) -> str:
    base_template = templates.get("top5_base")
    if not base_template: raise KeyError("Could not find required 'top5_base.txt'.")
    use_confidence = "with-confidence" in args.prompt_template
    use_flat_format = "flat" in args.prompt_template
    use_sci_format = "sci" in args.prompt_template
    entry_lines = []
    for k in range(1, 6):
        cid = row.get(f"pred{k}")
        meta = tax_maps['id2meta'].get(int(cid), {}) if pd.notna(cid) else {}
        if use_flat_format: name = _format_species_flat(meta)
        elif use_sci_format: name = _format_species_sci(meta)
        else: name = meta.get("common", "Unknown")
        conf = _conf_str(row.get(f"conf{k}")) if use_confidence else ""
        entry_lines.append(f"{k}. {name}{conf}")
    species_list_content = "\n".join(entry_lines)
    confidence_note_content = ""
    if use_confidence:
        confidence_note_content = ("Note on confidence: The confidence shown for the highest-ranked candidate (p1) reflects how certain the underlying model was. "
                                   "Use it only as a signal of certainty. If p1 appears strong and aligns with visible evidence, you may lean toward that candidate. "
                                   "If p1 appears weak or the image contradicts it, give more weight to visual evidence and consider other candidates.\n\n")
    return base_template.format(species_list=species_list_content, confidence_note=confidence_note_content)

def build_prompt_zeroshot(tax_maps: Dict, templates: Dict[str, str], args: argparse.Namespace, **kwargs) -> str:
    ask_for_explanation = "explanation" in args.prompt_template
    use_all200_list = "all200" in args.prompt_template
    if use_all200_list:
        base_template = templates.get("zeroshot_all200")
        if not base_template: raise KeyError("Could not find 'zeroshot_all200.txt'.")
        response_format = "Most Likely: [...]\nExplanation: [...]" if ask_for_explanation else "Most Likely: [...]"
        return base_template.format(species_block=tax_maps['all_species_list'], response_format=response_format)
    else:
        base_template = templates.get("zeroshot_identify")
        if not base_template: raise KeyError("Could not find 'zeroshot_identify.txt'.")
        response_format = "Common Name: [...]\nScientific Name: [...]\nExplanation: [...]" if ask_for_explanation else "Common Name: [...]\nScientific Name: [...]"
        return base_template.format(response_format=response_format)

def build_prompt_multimodal_16shot(row: pd.Series, tax_maps: Dict, templates: Dict[str, str], args: argparse.Namespace) -> List[Dict[str, Any]]:
    base_template = templates.get("top5_multimodal_16shot")
    if not base_template: raise KeyError("Could not find 'top5_multimodal_16shot.txt'.")
    use_confidence = "with-confidence" in args.prompt_template
    query_image_path = Path(args.image_dir) / row["image_path"]
    if not query_image_path.exists(): raise FileNotFoundError(f"Query image not found: {query_image_path}")
    query_img_b64 = encode_image_b64(query_image_path)
    confidence_note_content = ""
    if use_confidence:
        confidence_note_content = ("Note on confidence: The confidence shown for the highest-ranked candidate (p1) reflects how certain the underlying model was. "
                                   "Use it only as a signal of certainty. If p1 appears strong and aligns with visible evidence, you may lean toward that candidate. "
                                   "If p1 appears weak or the image contradicts it, give more weight to visual evidence and consider other candidates.\n\n")
    final_template_text = base_template.format(confidence_note=confidence_note_content)
    messages = [{"type": "text", "text": final_template_text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_img_b64}"}}]
    name_to_classid = tax_maps['name_to_classid']
    for i in range(1, 6):
        sci_name_key = f"pred{i}_sci"
        if sci_name_key not in row or pd.isna(row[sci_name_key]): raise KeyError(f"Input data missing: {sci_name_key}")
        conf_str_val = _conf_str(row.get(f"conf{i}")) if use_confidence else ""
        candidate_text = f"Candidate {i}: {row[sci_name_key]}{conf_str_val}"
        name = str(row[sci_name_key]).lower().strip()
        class_id = name_to_classid.get(name)
        if class_id is None: raise KeyError(f"Species name '{name}' not in taxonomy map.")
        ref_image_path = Path(args.ref_image_dir) / f"{class_id}.jpg"
        if not ref_image_path.exists(): raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
        ref_img_b64 = encode_image_b64(ref_image_path)
        messages.append({"type": "text", "text": candidate_text})
        messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_img_b64}"}})
    return messages

# --- This map is the "switch-case" for all prompt templates ---
PROMPT_BUILDER_MAP = {
    'top5-simple': build_prompt_top5,
    'top5-simple-with-confidence': build_prompt_top5,
    'top5-sci': build_prompt_top5,
    'top5-sci-with-confidence': build_prompt_top5,
    'top5-flat': build_prompt_top5,
    'top5-flat-with-confidence': build_prompt_top5,
    'zeroshot': build_prompt_zeroshot,
    'zeroshot-explanation': build_prompt_zeroshot,
    'zeroshot-all200': build_prompt_zeroshot,
    'zeroshot-all200-explanation': build_prompt_zeroshot,
    'top5-multimodal-16shot': build_prompt_multimodal_16shot,
    'top5-multimodal-16shot-with-confidence': build_prompt_multimodal_16shot,
}

def build_prompt(args: argparse.Namespace, tax_maps: Dict, templates: Dict[str, str], row: Optional[pd.Series] = None) -> Union[str, list]:
    """Looks up the correct builder function from the map and calls it."""
    builder_func = PROMPT_BUILDER_MAP.get(args.prompt_template)
    if not builder_func:
        raise ValueError(f"Unknown prompt template name: '{args.prompt_template}'")
    
    # All builders need tax_maps, templates, and args
    # Only some need the row
    if builder_func in [build_prompt_top5, build_prompt_multimodal_16shot]:
        if row is None:
            raise ValueError(f"A data row is required for the '{args.prompt_template}' template.")
        return builder_func(row, tax_maps, templates, args)
    else: # zeroshot builders
        return builder_func(tax_maps, templates, args)


# ==============================================================================
# === Backend API Callers ======================================================
# ==============================================================================
def _prepare_content_payload(prompt: Union[str, list], b64_img: str) -> list:
    """Helper to create the correct 'content' payload list for an API call."""
    if isinstance(prompt, list): return prompt
    return [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]

def call_nebius(api_base: str, api_key: str, model: str, prompt: Union[str, list], b64_img: str) -> str:
    if OpenAIClient is None: raise RuntimeError("The 'openai' library is required.")
    client = OpenAIClient(base_url=api_base, api_key=api_key)
    content_payload = _prepare_content_payload(prompt, b64_img)
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content_payload}], temperature=0.4, max_tokens=900)
    return (resp.choices[0].message.content or "").strip()

def call_hyperbolic(api_base: str, api_key: str, model: str, prompt: Union[str, list], b64_img: str) -> str:
    api_url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    content_payload = _prepare_content_payload(prompt, b64_img)
    payload = {"model": model, "messages": [{"role": "user", "content": content_payload}], "max_tokens": 900, "temperature": 0.4}
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return (data['choices'][0]['message']['content'] or "").strip()

def call_ollama(ollama_host: str, model: str, prompt: Union[str, list], b64_img: str) -> str:
    url = f"{ollama_host.rstrip('/')}/api/chat"
    if isinstance(prompt, list):
        main_content = next((item['text'] for item in prompt if item['type'] == 'text'), '')
        images_b64 = [item['image_url']['url'].split(',')[-1] for item in prompt if item['type'] == 'image_url']
        messages = [{"role": "user", "content": main_content, "images": images_b64}]
    else:
        messages = [{"role": "user", "content": prompt, "images": [b64_img]}]
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.4, "num_predict": 900}}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp: body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e: raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")
    except Exception as e: raise RuntimeError(f"Ollama request failed: {e}")
    return (body.get("message", {}).get("content") or "").strip()

# ==============================================================================
# === Main Execution Logic =====================================================
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="Unified Qwen VL Inference Script with External Prompt Templates.")
    # --- This now automatically uses the keys from our map for validation ---
    ap.add_argument("--prompt-template", required=True, choices=PROMPT_BUILDER_MAP.keys())
    ap.add_argument("--prompt-dir", default="./prompt_templates")
    ap.add_argument("--backend", required=True, choices=["nebius", "hyperbolic", "ollama"])
    ap.add_argument("--api-model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--api-base", default="https://api.studio.nebius.com/v1/")
    ap.add_argument("--ollama-model", default="qwen:latest")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--image-paths", help="Path to a text file with a list of relative image paths to process.")
    ap.add_argument("--taxonomy-json", required=True)
    ap.add_argument("--topk-json", help="Path to top-k predictions JSON (REQUIRED for all 'top5' templates).")
    ap.add_argument("--ref-image-dir", help="Path to pre-generated reference images (for multimodal prompts).")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--error-file", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-first-prompt", action="store_true")
    ap.add_argument("--throttle-sec", type=float, default=0.5)
    args = ap.parse_args()

    # --- Validate arguments ---
    is_top5 = args.prompt_template.startswith("top5")
    is_multimodal = 'multimodal' in args.prompt_template
    if not args.image_paths: sys.exit("ERROR: --image-paths is required to specify which images to process.")
    if is_top5 and not args.topk_json: sys.exit(f"ERROR: --topk-json is required for the '{args.prompt_template}' template.")
    if is_multimodal and not args.ref_image_dir: sys.exit(f"ERROR: --ref-image-dir is required for the '{args.prompt_template}' template.")

    # --- Setup paths and backend ---
    image_dir = Path(args.image_dir); out_csv = Path(args.output_csv); err_file = Path(args.error_file)
    out_csv.parent.mkdir(parents=True, exist_ok=True); err_file.parent.mkdir(parents=True, exist_ok=True)
    model_name, api_key = "", None
    if args.backend in ["nebius", "hyperbolic"]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: sys.exit(f"ERROR: OPENAI_API_KEY must be set for the '{args.backend}' backend.")
        model_name = args.api_model
    elif args.backend == "ollama":
        if not args.ollama_host or not args.ollama_model: sys.exit("ERROR: --ollama-host and --ollama-model are required.")
        model_name = args.ollama_model

    # --- Unified Data Loading Pipeline ---
    templates = load_prompt_templates(Path(args.prompt_dir))
    tax_maps = build_taxonomy_maps(Path(args.taxonomy_json))
    print(f">> Loading image list from {args.image_paths}")
    test_paths = read_test_list(Path(args.image_paths))
    df = pd.DataFrame({"image_path": test_paths})

    if is_top5:
        print(f">> Loading top-5 predictions from {args.topk_json}")
        topk_map_abs = load_topk_json_exact(Path(args.topk_json))
        topk_map_rel = {normalize_path_for_matching(p, image_dir): rec for p, rec in topk_map_abs.items()}
        records = []
        for path in test_paths:
            norm_path = normalize_path_for_matching(path, image_dir)
            rec = topk_map_rel.get(norm_path)
            if rec:
                cls, probs = rec.get("topk_cls") or [], rec.get("topk_probs") or []
                new_row = {"image_path": path}
                for k in range(1, 6):
                    cid = cls[k-1] if k-1 < len(cls) else None
                    conf = probs[k-1] if k-1 < len(probs) else None
                    cid_int = int(cid) if cid is not None else None
                    new_row[f"pred{k}"] = cid_int
                    new_row[f"conf{k}"] = float(conf) if conf is not None else None
                    new_row[f"pred{k}_sci"] = tax_maps['id2common'].get(cid_int) if cid_int is not None else None
                records.append(new_row)
        df = pd.DataFrame.from_records(records)
    if df.empty: raise SystemExit("No data to process. The intersection of image_paths and topk_json might be empty.")

    # --- First Prompt Preview ---
    if not df.empty:
        first_row = df.iloc[0]
        # Pass the whole args object to the builder
        first_prompt = build_prompt(args, tax_maps, templates, first_row)
        print("=== First Sample Prompt ===")
        if isinstance(first_prompt, list):
            print(format_multimodal_prompt_for_display(first_prompt))
        else:
            print(first_prompt)
        print("\n===========================\n")
        if args.print_first_prompt: return

    # --- Resume Support ---
    done_paths = set()
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            prev_df = pd.read_csv(out_csv)
            if "image_path" in prev_df.columns:
                done_paths = set(prev_df["image_path"].astype(str).tolist())
                print(f">> Found {len(done_paths)} completed images. Resuming...")
        except Exception as e:
            print(f"Warning: Could not read existing output CSV for resuming: {e}", file=sys.stderr)

    # --- Main Inference Loop ---
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
        rel_path = str(row["image_path"])
        if rel_path in done_paths: continue
        try:
            prompt_content = build_prompt(args, tax_maps, templates, row)
            prompt_for_csv = format_prompt_for_csv(prompt_content)
            b64_img = ""
            if not is_multimodal:
                img_path = image_dir / rel_path
                if not img_path.exists(): raise FileNotFoundError(f"Image not found: {img_path}")
                with Image.open(img_path) as im: im.verify()
                b64_img = encode_image_b64(img_path)
            if args.dry_run:
                if len(done_paths) < 5: print(f"Dry run, would process: {rel_path}")
                time.sleep(0.01)
                continue
            answer = ""
            if args.backend == "nebius": answer = call_nebius(args.api_base, api_key, model_name, prompt_content, b64_img)
            elif args.backend == "hyperbolic": answer = call_hyperbolic(args.api_base, api_key, model_name, prompt_content, b64_img)
            else: answer = call_ollama(args.ollama_host, model_name, prompt_content, b64_img)
            out_row = {"image_path": rel_path, "MODEL_NAME": model_name, "prompt": prompt_for_csv, "response": answer}
            header = not out_csv.exists() or out_csv.stat().st_size == 0
            pd.DataFrame([out_row]).to_csv(out_csv, mode="a", header=header, index=False)
            done_paths.add(rel_path)
            time.sleep(args.throttle_sec)
        except Exception as e:
            print(f"\nERROR on {rel_path}: {e}", file=sys.stderr)
            with err_file.open("a") as logf: logf.write(f"{rel_path}: {str(e)}\n")
            continue
    print("\nInference complete.")

if __name__ == "__main__":
    main()