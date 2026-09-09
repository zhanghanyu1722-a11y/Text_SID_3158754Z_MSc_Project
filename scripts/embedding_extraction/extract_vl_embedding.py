#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-Embedding-2B multimodal feature extraction (unified family)
=====================================================================

Encodes each MicroLens item by presenting its cover image together with the
title formatted as ``Represent this video titled: <title>`` and maps it into
the joint Qwen3-VL-Embedding space (one 2048-d vector per item).  This is the
"Multimodal B / unified" configuration of the dissertation.

Output tensor shape: ``[num_items, 2048]``, row ``i`` = dense item id ``i``.

Robustness / engineering details
--------------------------------
* Runs through ``sentence_transformers`` with ``trust_remote_code=True`` and
  output L2 normalisation.
* Missing/unreadable covers are replaced by a solid-black placeholder image at
  the modal cover resolution (640x360); the encoder always receives a valid
  image.
* Checkpoint-resume: the accumulated tensor and the index of the last saved
  item are persisted every ``--save-interval`` items, so an interrupted run
  can continue without re-encoding the whole corpus.
"""
import argparse
import json
import os

import pandas as pd
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Qwen3-VL-Embedding-2B features for title+cover items."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="HuggingFace model name.",
    )
    parser.add_argument(
        "--titles-csv",
        default="data/microLen/MicroLens-50k_titles.csv",
        help="Item CSV (columns 'item,title').",
    )
    parser.add_argument(
        "--covers-dir",
        default="data/MicroLens-50k_covers",
        help="Directory of cover images named '<original_item_id>.jpg'.",
    )
    parser.add_argument(
        "--output",
        default="data/microLen/mmB_qwen3vl_2B_embeds.pt",
        help="Output .pt tensor, [num_items x 2048].",
    )
    parser.add_argument(
        "--progress-file",
        default="data/microLen/mmB_qwen3vl_2B_extract_progress.json",
        help="Checkpoint-resume state file.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument(
        "--save-interval", type=int, default=500,
        help="Persist state every N items.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (default: cuda if available).",
    )
    return parser.parse_args()


def load_progress(progress_file: str, output: str):
    """Return (last_processed_index, accumulated tensor) if a checkpoint exists."""
    if os.path.exists(progress_file) and os.path.exists(output):
        with open(progress_file, "r") as f:
            data = json.load(f)
            return data.get("last_processed_idx", -1), torch.load(output)
    return -1, None


def save_progress(progress_file: str, output: str, current_idx: int, tensor) -> None:
    torch.save(tensor, output)
    with open(progress_file, "w") as f:
        json.dump({"last_processed_idx": current_idx}, f)


def main() -> None:
    args = parse_args()
    print(f"Device: {args.device}")

    # Ensure the output and progress-file directories exist before saving.
    for p in (args.output, args.progress_file):
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # 1. Build the dense-id mapping (same ordering as data_process.py).
    print("Building the dense item-id mapping ...")
    df_titles = pd.read_csv(args.titles_csv).sort_values("item").reset_index(drop=True)
    items = [
        {"new_id": new_id, "old_id": str(row["item"]), "title": str(row["title"])}
        for new_id, row in df_titles.iterrows()
    ]
    num_items = len(items)
    print(f"Total items: {num_items}")

    # 2. Restore resume state if present.
    last_idx, final_tensor = load_progress(args.progress_file, args.output)
    start_idx = last_idx + 1
    if start_idx > 0:
        print(f"Resuming from item index {start_idx} ...")
    else:
        print("Fresh extraction run ...")

    if start_idx >= num_items:
        print("Extraction already complete.")
        return

    # 3. Load the model.
    print(f"Loading model {args.model} ...")
    model = SentenceTransformer(args.model, trust_remote_code=True, device=args.device)

    missing_images = 0
    fallback_size = (640, 360)  # modal cover resolution used for the black placeholder

    # 4. Batch inference.
    for i in tqdm(range(start_idx, num_items, args.batch_size),
                  desc="Extraction", initial=start_idx, total=num_items):
        batch_items = items[i : i + args.batch_size]
        batch_inputs, batch_new_ids = [], []

        for item in batch_items:
            img_path = os.path.join(args.covers_dir, f"{item['old_id']}.jpg")
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", fallback_size, color="black")
                missing_images += 1

            batch_inputs.append({
                "text": f"Represent this video titled: {item['title']}",
                "image": img,
            })
            batch_new_ids.append(item["new_id"])

        # Encode; sentence-transformers fuses the modalities and L2-normalises.
        embeddings = model.encode(
            batch_inputs, convert_to_tensor=True, normalize_embeddings=True
        )

        if final_tensor is None:
            embed_dim = embeddings.shape[-1]
            final_tensor = torch.zeros(
                (num_items, embed_dim), dtype=torch.float32
            )
            print(f"Model output dim: {embed_dim}")

        for new_id, feat in zip(batch_new_ids, embeddings.cpu()):
            final_tensor[new_id] = feat

        # Periodic persistence for checkpoint-resume.
        current_max_idx = i + len(batch_items) - 1
        if (i + args.batch_size) % args.save_interval == 0 or current_max_idx == num_items - 1:
            save_progress(args.progress_file, args.output, current_max_idx, final_tensor)

    if missing_images:
        print(f"[warn] {missing_images} missing covers replaced by black placeholders")
    print(f"Done. Features saved to: {args.output}")


if __name__ == "__main__":
    main()
