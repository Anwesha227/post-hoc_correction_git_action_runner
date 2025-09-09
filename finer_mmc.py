#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm  # progress bars

# ---- deps: pip install open_clip_torch torchvision tqdm ----
import open_clip

def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def random_augmentation(n_px: int):
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        lambda image: image.convert("RGB"),
        transforms.RandomChoice([
            transforms.CenterCrop(n_px),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomResizedCrop(224, (0.5, 1.0)),
        ]),
        transforms.RandomChoice([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
            transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
        ]),
        transforms.RandomChoice([
            transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.6),
            transforms.RandomApply([transforms.RandomRotation(30)], p=0.6),
            transforms.RandomApply([transforms.RandomPerspective()], p=0.6),
        ]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# Compact CLIP template set
CLIP_TEMPLATES = [
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
    "a photo of the {}.",
    "a good photo of a {}.",
    "a wildlife photo of a {}.",
    "a photo of a {} bird.",
    "a photo of one {}.",
    "a bright photo of a {}.",
    "a photo of a small {}.",
]

def read_listfile(list_path: Path) -> List[Tuple[str, int]]:
    """
    Each line: <relpath> <class_id> <extra>
    We only care about relpath and class_id.
    """
    items: List[Tuple[str, int]] = []
    with list_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Expected at least 2 tokens '<relpath> <class_id>', got: {line}")
            rel = parts[0]      # path
            y = int(parts[1])   # class id
            items.append((rel, y))
    return items

class ListFileImageDataset(Dataset):
    def __init__(self, root: Path, list_path: Path, transform):
        self.root = Path(root)
        self.items = read_listfile(Path(list_path))
        self.transform = transform
        self.num_classes = 1 + max(y for _, y in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel, y = self.items[idx]
        p = self.root / rel
        if not p.exists():
            raise FileNotFoundError(f"Could not resolve image path: {p}")
        img = Image.open(p).convert("RGB")
        return self.transform(img), y

def build_class_prompt_texts(
    metrics_json: Path,
    name_key: str = "most_common_name",
    fallback_keys: List[str] = ("scientific_name", "most_common_name_alt"),
    lowercase: bool = False
) -> Dict[int, List[str]]:
    with open(metrics_json, "r") as f:
        meta = json.load(f)
    ids = sorted(int(k) for k in meta.keys())
    out: Dict[int, List[str]] = {}
    for cid in ids:
        m = meta[str(cid)]
        name = m.get(name_key)
        if not name:
            for k in fallback_keys:
                if m.get(k):
                    name = m[k]; break
        if not name:
            raise ValueError(f"No usable name for class {cid} in {metrics_json}")
        name = name.strip()
        if lowercase:
            name = name.lower()
        out[cid] = [name]
    return out

def expand_with_templates(class_names: Dict[int, List[str]]) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for cid, names in class_names.items():
        texts = []
        for n in names:
            texts.extend([t.format(n) for t in CLIP_TEMPLATES])
        out[cid] = texts
    return out

@torch.no_grad()
def compute_text_weights_from_names(model, tokenizer, device, cid_to_texts: Dict[int, List[str]], normalize: bool = True) -> torch.Tensor:
    model.eval()
    C = 1 + max(cid_to_texts.keys())
    rows = []
    for cid in tqdm(range(C), desc="Encoding text prompts"):
        texts = cid_to_texts[cid]
        toks = tokenizer(texts).to(device)
        txt = model.encode_text(toks).float()
        if normalize: txt = l2norm(txt)
        w = txt.mean(dim=0)
        if normalize: w = l2norm(w)
        rows.append(w)
    W = torch.stack(rows, dim=0)
    if normalize: W = l2norm(W)
    return W  # (C, D)

@torch.no_grad()
def build_vision_prototypes(model, loader, device, aug_k: int = 0) -> torch.Tensor:
    model.eval()
    C = loader.dataset.num_classes
    buckets = [[] for _ in range(C)]
    for images, labels in tqdm(loader, desc="Building vision prototypes"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats = l2norm(model.encode_image(images).float())
        for f, y in zip(feats, labels):
            buckets[int(y.item())].append(f)

        for _ in range(max(0, aug_k)):
            feats = l2norm(model.encode_image(images).float())
            for f, y in zip(feats, labels):
                buckets[int(y.item())].append(f)

    rows = []
    for cls_list in buckets:
        if not cls_list:
            raise RuntimeError("Empty class bucket when building prototypes.")
        m = torch.stack(cls_list, dim=0).mean(dim=0)
        rows.append(l2norm(m))
    return torch.stack(rows, dim=0)  # (C, D)

def fuse_text_vision(W_txt: torch.Tensor, W_img: torch.Tensor, alpha: float) -> torch.Tensor:
    W_txt = l2norm(W_txt); W_img = l2norm(W_img)
    return l2norm(alpha * W_txt + (1.0 - alpha) * W_img)

@torch.no_grad()
def eval_top1(model, loader, W: torch.Tensor, device, logit_scale: float = 1.0) -> float:
    model.eval()
    correct = 0; total = 0
    WT = W.t().contiguous()
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feats = l2norm(model.encode_image(images).float())
        logits = feats @ WT
        if logit_scale != 1.0:
            logits = logits * logit_scale
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)

def main():
    ap = argparse.ArgumentParser("MMC eval (zero-shot + vision prototypes + fusion)")
    # Model
    ap.add_argument("--model_cfg", default="ViT-B-32", type=str)
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k", type=str)
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    # Data (LOCAL roots)
    ap.add_argument("--dataset_root_train", required=True, type=str,
                    help="e.g., datasets/semi-aves/trainval_images")
    ap.add_argument("--dataset_root_test",  required=True, type=str,
                    help="e.g., datasets/semi-aves/test")
    ap.add_argument("--train_list", required=True, type=str,
                    help="e.g., data/semi-aves/fewshot16_seed1.txt")
    ap.add_argument("--test_list",  required=True, type=str,
                    help="e.g., data/semi-aves/test.txt")
    # Names / prompts
    ap.add_argument("--metrics_json", required=True, type=str,
                    help="e.g., data/semi-aves/semi-aves_metrics-LAION400M.json")
    ap.add_argument("--name_key", default="most_common_name", type=str)
    ap.add_argument("--lowercase_names", action="store_true")
    # MMC & loader
    ap.add_argument("--use_random_aug", action="store_true",
                    help="Use paper's random augmentation for prototype building.")
    ap.add_argument("--aug_k", default=0, type=int,
                    help="Extra stochastic passes per batch when building prototypes.")
    ap.add_argument("--alpha", default=0.7, type=float,
                    help="Fusion weight: w_mm = alpha*w_text + (1-alpha)*w_img.")
    ap.add_argument("--batch_size", default=256, type=int)
    ap.add_argument("--workers", default=8, type=int)
    ap.add_argument("--logit_scale", default=1.0, type=float)

    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Model + preprocess
    model, _, base_preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_cfg,
        pretrained=args.pretrained,
        device=device
    )
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model_cfg)

    # Train transform
    if args.use_random_aug:
        print("[MMC] Using paper's random augmentation for prototypes.")
        train_transform = random_augmentation(224)
    else:
        train_transform = base_preprocess

    # Datasets / loaders
    train_ds = ListFileImageDataset(args.dataset_root_train, args.train_list, train_transform)
    test_ds  = ListFileImageDataset(args.dataset_root_test,  args.test_list,  base_preprocess)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # Recompute TEXT weights
    cid_to_names = build_class_prompt_texts(Path(args.metrics_json), name_key=args.name_key,
                                            lowercase=args.lowercase_names)
    cid_to_texts = expand_with_templates(cid_to_names)

    # Remap dataset labels to match JSON index order
    json_ids = sorted(cid_to_names.keys())
    id_to_row = {cid: i for i, cid in enumerate(json_ids)}
    train_ds.items = [(rel, id_to_row[y]) for rel, y in train_ds.items]
    test_ds.items  = [(rel, id_to_row[y]) for rel, y in test_ds.items]

    print("[DEBUG] First 5 remapped test entries:", test_ds.items[:5])

    print("[MMC] Encoding text prompts (most_common_name + CLIP templates) ...")
    W_txt = compute_text_weights_from_names(model, tokenizer, device, cid_to_texts, normalize=True)
    print(f"[MMC] W_txt shape: {tuple(W_txt.shape)}")

    # 1) Zero-shot (TEXT)
    print("[MMC] Evaluating TEXT head (zero-shot) ...")
    acc_txt = eval_top1(model, test_loader, W_txt, device, logit_scale=args.logit_scale)
    print(f"[MMC] Text-only Top-1: {acc_txt:.4f}")

    # 2) Vision prototypes
    print(f"[MMC] Building vision prototypes (aug_k={args.aug_k}) ...")
    W_img = build_vision_prototypes(model, train_loader, device, aug_k=args.aug_k)
    print(f"[MMC] W_img shape: {tuple(W_img.shape)}")

    print("[MMC] Evaluating VISION head ...")
    acc_img = eval_top1(model, test_loader, W_img, device, logit_scale=args.logit_scale)
    print(f"[MMC] Vision-only Top-1: {acc_img:.4f}")

    # 3) Fusion
    print(f"[MMC] Evaluating FUSED head (alpha={args.alpha}) ...")
    W_mm = fuse_text_vision(W_txt, W_img, alpha=args.alpha)
    acc_mm = eval_top1(model, test_loader, W_mm, device, logit_scale=args.logit_scale)
    print(f"[MMC] Fused Top-1: {acc_mm:.4f}")

    # Summary JSON
    print(json.dumps({
        "acc_text": round(float(acc_txt), 6),
        "acc_vision": round(float(acc_img), 6),
        "acc_fused": round(float(acc_mm), 6),
        "alpha": args.alpha,
        "aug_k": args.aug_k,
        "templates": len(CLIP_TEMPLATES),
        "classes_train": train_ds.num_classes,
        "classes_test": test_ds.num_classes,
        "model_cfg": args.model_cfg,
        "pretrained": args.pretrained
    }, indent=2))

if __name__ == "__main__":
    main()
