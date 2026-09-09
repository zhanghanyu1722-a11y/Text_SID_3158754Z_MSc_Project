#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP ViT-B/32 cover-image feature extraction (dual-tower image tower)
======================================================================

Encodes every MicroLens-50k cover image with the frozen OpenAI CLIP image
encoder and stores one L2-normalised vector per item, aligned to the dense
contiguous item ids used by GRID.

Output tensor shape: ``[num_items, 512]`` (ViT-B/32), row ``i`` corresponds to
the item whose dense id is ``i``.  Rows whose cover could not be read are left
at their zero initialisation (a robustness safeguard; in MicroLens-50k every
item has a cover, so this path should never trigger).

Usage example (called by the pipeline scripts from the GRID root):

    python scripts/embedding_extraction/extract_clip.py \
        --covers-dir data/MicroLens-50k_covers \
        --titles-csv data/microLen/MicroLens-50k_titles.csv \
        --output data/microLen/clip_image_embeds.pt
"""
import argparse
import os
from pathlib import Path

import clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CLIP ViT-B/32 image features for MicroLens covers."
    )
    parser.add_argument(
        "--covers-dir",
        default="data/MicroLens-50k_covers",
        help="Directory of cover images named '<original_item_id>.jpg'.",
    )
    parser.add_argument(
        "--titles-csv",
        default="data/microLen/MicroLens-50k_titles.csv",
        help="Item CSV (columns 'item,title') used to align dense ids.",
    )
    parser.add_argument(
        "--output",
        default="data/microLen/clip_image_embeds.pt",
        help="Output .pt tensor, [num_items x 512].",
    )
    parser.add_argument("--clip-model", default="ViT-B/32", help="CLIP model name.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: cuda if available).",
    )
    return parser.parse_args()


def load_images_from_folder(folder_path: str) -> dict:
    """Return {original_item_id: image_path} for every image in the folder."""
    image_dict = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # File name is '<original_item_id>.jpg'.
            item_id = os.path.splitext(filename)[0]
            image_dict[item_id] = os.path.join(folder_path, filename)
    return image_dict


def extract_clip_features(image_paths: dict, model, preprocess, device: str,
                          batch_size: int = 64) -> dict:
    """Encode images in batches; returns {item_id: l2-normalised vector}."""
    model.eval()
    item_ids = list(image_paths.keys())
    embeddings = {}

    for i in tqdm(range(0, len(item_ids), batch_size), desc="CLIP features"):
        batch_ids = item_ids[i : i + batch_size]
        batch_images = []
        for item_id in batch_ids:
            try:
                img = Image.open(image_paths[item_id]).convert("RGB")
                batch_images.append(preprocess(img).unsqueeze(0))
            except Exception as e:
                print(f"[warn] failed to read image {item_id}: {e}")
                continue
        if not batch_images:
            continue

        batch_tensor = torch.cat(batch_images, dim=0).to(device)
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            # L2-normalise immediately after encoding.
            features = features / features.norm(dim=-1, keepdim=True)

        for item_id, feat in zip(batch_ids, features.cpu().numpy()):
            embeddings[item_id] = feat
    return embeddings


def main() -> None:
    args = parse_args()
    print(f"Device: {args.device}")

    print(f"Loading CLIP model: {args.clip_model}")
    model, preprocess = clip.load(args.clip_model, device=args.device)

    image_paths = load_images_from_folder(args.covers_dir)
    print(f"Found {len(image_paths)} cover images")

    embeddings = extract_clip_features(
        image_paths, model, preprocess, args.device, args.batch_size
    )
    print(f"Extracted {len(embeddings)} feature vectors")

    # Align to the dense contiguous ids (same ordering as data_process.py).
    print("Aligning to dense item ids ...")
    df_titles = pd.read_csv(args.titles_csv).sort_values("item").reset_index(drop=True)
    item_old_to_new = {
        str(old_id): new_id for new_id, old_id in enumerate(df_titles["item"])
    }

    embed_dim = 512  # ViT-B/32 output dimensionality
    num_items = len(item_old_to_new)
    final_tensor = torch.zeros((num_items, embed_dim), dtype=torch.float32)

    missing = 0
    for old_id, feat in embeddings.items():
        if old_id in item_old_to_new:
            final_tensor[item_old_to_new[old_id]] = torch.tensor(feat)
        else:
            missing += 1
    if missing:
        print(f"[warn] {missing} images have no matching id in the title CSV")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(final_tensor, args.output)
    print(f"Saved aligned CLIP features to: {args.output}")
    print(f"Tensor shape: {tuple(final_tensor.shape)}")


if __name__ == "__main__":
    main()
