"""
Fine-grained evaluation with CLIP-style models:
- Text-only (zero-shot) head from prompt templates
- Vision prototypes from K random augmentations per training image
- Fused head: alpha * text + (1 - alpha) * vision

Changes over finer_mmc.py:
- Train/prototype dataset returns PIL images (no transform applied in Dataset)
- Explicit K (aug_repeats) random augmentations sampled INSIDE prototype builder
- Memory-safe batching for large K via --encode_chunk_size
- torchvision version compatibility for AdjustSharpness / RandomChoice

Requires:
  pip install open_clip_torch torchvision pillow tqdm
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# torchvision only used for augmentations
import torchvision.transforms as T

# OpenCLIP
import open_clip


# ---------------------------
# Helpers
# ---------------------------
def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def read_listfile(list_path: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with list_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 2:
                raise ValueError(f"List line must have at least 2 tokens: {line}")
            rel = toks[0]
            try:
                y = int(toks[1])
            except Exception:
                raise ValueError(f"Second token must be an int class id: {line}")
            items.append((rel, y))
    if not items:
        raise ValueError(f"No items found in list file: {list_path}")
    return items


# ---------------------------
# Dataset
# ---------------------------
class ListFileImageDataset(Dataset):
    """
    If return_pil=True: returns (PIL.Image, y, relpath) so caller can apply K random augs.
    If return_pil=False: returns (tensor, y) using provided transform (for test/eval).
    """
    def __init__(
        self,
        root: str,
        list_path: str,
        transform: Optional[T.Compose] = None,
        return_pil: bool = False,
    ):
        self.root = Path(root)
        self.items = read_listfile(Path(list_path))
        self.transform = transform
        self.return_pil = return_pil
        self.num_classes = 1 + max(y for _, y in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        rel, y = self.items[idx]
        p = self.root / rel
        if not p.exists():
            raise FileNotFoundError(f"Missing image: {p}")
        img = Image.open(p).convert("RGB")

        if self.return_pil:
            return img, y, rel

        if self.transform is None:
            raise ValueError("transform must be provided when return_pil=False")
        return self.transform(img), y


# ---------------------------
# Augmentation pipeline (version compatible)
# ---------------------------
def random_augmentation(n_px: int = 224) -> T.Compose:
    """
    Strong yet stable pipeline for prototypes.
    Compatible with older torchvision where RandomChoice has no scalar p,
    and AdjustSharpness may be missing.
    """
    # AdjustSharpness / RandomAdjustSharpness compatibility
    if hasattr(T, "AdjustSharpness"):
        sharpness_tf = T.AdjustSharpness(sharpness_factor=2.0)
    else:
        # older API: RandomAdjustSharpness takes a p itself
        sharpness_tf = T.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0)

    return T.Compose([
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC, antialias=True),

        # Choose one crop uniformly
        T.RandomChoice([
            T.CenterCrop(n_px),
            T.RandomCrop(n_px, padding=16, padding_mode="reflect"),
            T.RandomResizedCrop(
                n_px, scale=(0.5, 1.0),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True
            ),
        ]),

        # Optional color/sharpness
        T.RandomApply([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        ], p=0.5),
        T.RandomApply([sharpness_tf], p=0.5),

        # Apply one geometry transform 60% of the time (older torchvision-friendly)
        T.RandomApply([
            T.RandomChoice([
                T.RandomHorizontalFlip(p=1.0),
                T.RandomRotation(degrees=30, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomPerspective(distortion_scale=0.3, p=1.0,
                                    interpolation=T.InterpolationMode.BILINEAR),
            ])
        ], p=0.6),

        T.ToTensor(),
        # CLIP-style normalization
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])


# ---------------------------
# Class names & prompts
# ---------------------------
# CLIP_TEMPLATES = [
#     "a photo of a {}.",
#     "a close-up photo of a {}.",
#     "a cropped photo of a {}.",
#     "a photo of the {}.",
#     "a good photo of a {}.",
#     "a wildlife photo of a {}.",
#     "a photo of a {} bird.",
#     "a photo of one {}.",
#     "a bright photo of a {}.",
#     "a photo of a small {}.",
# ]

CLIP_TEMPLATES =[
    'a photo of a {}, a type of bird.',
]


def build_class_prompt_texts(
    metrics_json: str,
    name_key: str = "most_common_name",
    fallback_keys: Tuple[str, ...] = ("scientific_name", "most_common_name_alt"),
    lowercase: bool = False,
) -> Dict[int, List[str]]:
    with open(metrics_json, "r") as f:
        meta = json.load(f)
    ids = sorted(int(k) for k in meta.keys())
    out: Dict[int, List[str]] = {}
    for cid in ids:
        m = meta[str(cid)]
        name = m.get(name_key, None)
        if not name:
            for fk in fallback_keys:
                if m.get(fk, None):
                    name = m[fk]
                    break
        if not name:
            raise ValueError(f"No name found for class id {cid} in {metrics_json}")
        name = name.strip()
        if lowercase:
            name = name.lower()
        out[cid] = [name]
    return out


def expand_with_templates(cid_to_names: Dict[int, List[str]]) -> Dict[int, List[str]]:
    cid_to_texts: Dict[int, List[str]] = {}
    for cid, names in cid_to_names.items():
        prompts: List[str] = []
        for nm in names:
            prompts.extend([tmpl.format(nm) for tmpl in CLIP_TEMPLATES])
        cid_to_texts[cid] = prompts
    return cid_to_texts


# ---------------------------
# Text weights (zero-shot)
# ---------------------------
@torch.no_grad()
def compute_text_weights_from_names(
    model,
    tokenizer,
    device: torch.device,
    cid_to_texts: Dict[int, List[str]],
    normalize: bool = True,
) -> torch.Tensor:
    C = 1 + max(cid_to_texts.keys())
    rows: List[torch.Tensor] = []
    for cid in tqdm(range(C), desc="Encoding text prompts"):
        texts = cid_to_texts[cid]
        toks = tokenizer(texts).to(device)
        txt = model.encode_text(toks).float()  # (T, D)
        if normalize:
            txt = l2norm(txt)
        w = txt.mean(dim=0)                    # (D,)
        if normalize:
            w = l2norm(w)
        rows.append(w)
    W = torch.stack(rows, dim=0)               # (C, D)
    if normalize:
        W = l2norm(W)
    return W


# ---------------------------
# Vision prototypes with K augs
# ---------------------------
@torch.no_grad()
def build_vision_prototypes(
    model,
    loader: DataLoader,
    device: torch.device,
    transform: T.Compose,
    aug_repeats: int = 10,
    encode_chunk_size: int = 512,
) -> torch.Tensor:
    """
    Build class prototypes by sampling `aug_repeats` random augmented views
    per image using the given `transform`, and averaging features across all
    augmented views per CLASS.

    The loader must yield (PIL.Image, label, relpath).
    """
    assert transform is not None, "An augmentation transform must be provided"
    C = loader.dataset.num_classes
    buckets: List[List[torch.Tensor]] = [[] for _ in range(C)]

    for batch in tqdm(loader, desc="Building vision prototypes (K augs/img)"):
        # custom collate returns: images_pil (list[PIL]), labels (LongTensor), rels (list[str])
        images_pil, labels, *_ = batch

        if isinstance(labels, torch.Tensor):
            labels_list = labels.tolist()
        else:
            labels_list = list(labels)

        # Prepare augmented tensors and their target classes
        aug_tensors: List[torch.Tensor] = []
        aug_targets: List[int] = []
        for pil_img, y in zip(images_pil, labels_list):
            for _ in range(aug_repeats):
                x = transform(pil_img)  # (3, H, W)
                aug_tensors.append(x)
                aug_targets.append(int(y))

        if not aug_tensors:
            continue

        # Encode in chunks to avoid OOM
        feats_list: List[torch.Tensor] = []
        big_batch = torch.stack(aug_tensors, dim=0).to(device)  # (B*K, 3, H, W)

        for chunk in torch.split(big_batch, encode_chunk_size, dim=0):
            f = model.encode_image(chunk).float()
            f = l2norm(f)  # (chunk_size, D)
            feats_list.append(f)
        feats = torch.cat(feats_list, dim=0)  # (B*K, D)

        # Accumulate features into class buckets
        for f, y in zip(feats, aug_targets):
            buckets[y].append(f)

    # Average per class → L2 normalize
    rows: List[torch.Tensor] = []
    for cid, feat_list in enumerate(buckets):
        if not feat_list:
            raise RuntimeError(
                f"Empty bucket for class {cid}. "
                "Check your train_list coverage and label remapping."
            )
        m = torch.stack(feat_list, dim=0).mean(dim=0)
        rows.append(l2norm(m))
    return torch.stack(rows, dim=0)  # (C, D)


# ---------------------------
# Fusion & Eval
# ---------------------------
def fuse_text_vision(W_txt: torch.Tensor, W_img: torch.Tensor, alpha: float) -> torch.Tensor:
    W_txt = l2norm(W_txt)
    W_img = l2norm(W_img)
    W = alpha * W_txt + (1.0 - alpha) * W_img
    return l2norm(W)


@torch.no_grad()
def eval_top1(
    model,
    loader: DataLoader,
    W: torch.Tensor,
    device: torch.device,
    logit_scale: float = 1.0,
) -> float:
    WT = W.t().contiguous()  # (D, C)
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(device)
        feats = model.encode_image(images).float()
        feats = l2norm(feats)                   # (B, D)
        logits = feats @ WT                     # (B, C)
        if logit_scale != 1.0:
            logits = logits * logit_scale
        pred = logits.argmax(dim=-1)            # (B,)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(1, total)


# ---------------------------
# Collate (keeps PILs as a list)
# ---------------------------
def collate_pil(batch):
    """
    batch: List[(PIL.Image, int, str)]
    returns: (List[PIL.Image], torch.LongTensor, List[str])
    """
    imgs, ys, rels = zip(*batch)  # tuples
    return list(imgs), torch.tensor(ys, dtype=torch.long), list(rels)


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("finer_mmc_aug.py — CLIP text/vision/fused heads with K augs per image")
    # Model
    p.add_argument("--model_cfg", type=str, default="ViT-B-32")
    p.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--device", type=str, default="cuda")
    # Data
    p.add_argument("--dataset_root_train", type=str, required=True)
    p.add_argument("--dataset_root_test", type=str, required=True)
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--test_list", type=str, required=True)
    # Names/prompts
    p.add_argument("--metrics_json", type=str, required=True)
    p.add_argument("--name_key", type=str, default="most_common_name")
    p.add_argument("--lowercase_names", action="store_true")
    # Aug/prototypes
    p.add_argument("--use_random_aug", action="store_true",
                   help="Use the custom random augmentation pipeline (recommended for prototypes).")
    p.add_argument("--aug_repeats", type=int, default=10,
                   help="Number of random augmentations per image for prototype building.")
    p.add_argument("--encode_chunk_size", type=int, default=512,
                   help="Chunk size for image encoding during prototype building.")
    # Eval / fusion
    p.add_argument("--alpha", type=float, default=0.7, help="Fusion weight for text vs vision.")
    p.add_argument("--logit_scale", type=float, default=1.0)
    # Loader
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Model + preprocess (base_preprocess is CLIP eval transform; we'll use it for test set)
    model, _, base_preprocess = open_clip.create_model_and_transforms(
        model_name=args.model_cfg,
        pretrained=args.pretrained,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_cfg)

    # Choose augmentation pipeline for prototypes
    if args.use_random_aug:
        proto_transform = random_augmentation(224)
    else:
        # You can use base_preprocess for deterministic crops,
        # but for true "K augmentations", random_augmentation is recommended.
        proto_transform = base_preprocess

    # Datasets
    train_ds = ListFileImageDataset(
        root=args.dataset_root_train,
        list_path=args.train_list,
        transform=None,          # IMPORTANT: no transform here
        return_pil=True          # return PIL so we can sample K random augs later
    )
    test_ds = ListFileImageDataset(
        root=args.dataset_root_test,
        list_path=args.test_list,
        transform=base_preprocess,  # standard CLIP eval transform
        return_pil=False
    )

    # Load class names and expand with templates
    cid_to_names = build_class_prompt_texts(
        metrics_json=args.metrics_json,
        name_key=args.name_key,
        lowercase=args.lowercase_names
    )
    cid_to_texts = expand_with_templates(cid_to_names)

    # Remap dataset labels to [0..C-1] consistent with JSON keys order
    json_ids = sorted(cid_to_names.keys())
    id_to_row = {cid: i for i, cid in enumerate(json_ids)}

    train_ds.items = [(rel, id_to_row[y]) for (rel, y) in train_ds.items]
    test_ds.items  = [(rel, id_to_row[y]) for (rel, y) in test_ds.items]
    # Recompute num_classes after remap
    train_ds.num_classes = 1 + max(y for _, y in train_ds.items)
    test_ds.num_classes  = 1 + max(y for _, y in test_ds.items)

    # DataLoaders
    # NOTE: custom collate keeps images as list[PIL.Image] for the train/prototype loader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,      # CPU/MPS safe
        collate_fn=collate_pil,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,      # CPU/MPS safe
    )

    # Compute text weights (zero-shot)
    C = train_ds.num_classes
    contiguous_cid_to_texts = {i: cid_to_texts[json_ids[i]] for i in range(C)}
    W_txt = compute_text_weights_from_names(
        model, tokenizer, device, contiguous_cid_to_texts, normalize=True
    )
    print(f"[INFO] W_txt: {tuple(W_txt.shape)}")

    # Build vision prototypes with explicit K augmentations per image
    W_img = build_vision_prototypes(
        model=model,
        loader=train_loader,
        device=device,
        transform=proto_transform,
        aug_repeats=args.aug_repeats,
        encode_chunk_size=args.encode_chunk_size,
    )
    print(f"[INFO] W_img: {tuple(W_img.shape)}")

    # Evaluate
    acc_txt = eval_top1(model, test_loader, W_txt, device, logit_scale=args.logit_scale)
    acc_img = eval_top1(model, test_loader, W_img, device, logit_scale=args.logit_scale)

    # Fuse + eval
    W_fused = fuse_text_vision(W_txt, W_img, alpha=args.alpha)
    acc_fused = eval_top1(model, test_loader, W_fused, device, logit_scale=args.logit_scale)

    # Summary
    summary = {
        "acc_text": acc_txt,
        "acc_vision": acc_img,
        "acc_fused": acc_fused,
        "alpha": args.alpha,
        "aug_repeats": args.aug_repeats,
        "templates": len(CLIP_TEMPLATES),
        "classes_train": train_ds.num_classes,
        "classes_test": test_ds.num_classes,
        "model_cfg": args.model_cfg,
        "pretrained": args.pretrained,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
