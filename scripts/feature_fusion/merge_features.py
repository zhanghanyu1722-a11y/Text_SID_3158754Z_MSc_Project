#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Double-L2-normalised concatenation of text and image features (dual-tower)
==========================================================================

Implements the fusion used by the "Multimodal A" configuration: the Qwen3 text
vector and the CLIP image vector come from different encoders (different
dimensionalities, training objectives and output scales), so each block is
L2-normalised *before* concatenation, and the concatenated vector is
L2-normalised *again* afterwards.  This removes cross-encoder scale mismatch so
that neither modality dominates the fused representation because of raw vector
magnitude.  (It cannot align their semantics -- that is precisely the finding
of the dissertation: unaligned concatenation does not help.)

Usage example (called by the pipeline scripts from the GRID root):

    python scripts/feature_fusion/merge_features.py \
        --text-features  data/microLen/qwen3_text_0.6B_embeds.pt \
        --image-features data/microLen/clip_image_embeds.pt \
        --output         data/microLen/mmA_text_clip_concat_embeds.pt

Expected output shape: ``[num_items, 1024 + 512] = [num_items, 1536]``.
"""
import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate L2-normalised text and image features."
    )
    parser.add_argument(
        "--text-features",
        default="data/microLen/qwen3_text_0.6B_embeds.pt",
        help="Text embedding tensor, [num_items x D_text].",
    )
    parser.add_argument(
        "--image-features",
        default="data/microLen/clip_image_embeds.pt",
        help="Image embedding tensor, [num_items x D_image].",
    )
    parser.add_argument(
        "--output",
        default="data/microLen/mmA_text_clip_concat_embeds.pt",
        help="Output fused tensor, [num_items x (D_text + D_image)].",
    )
    return parser.parse_args()


def l2_normalise(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def main() -> None:
    args = parse_args()

    print("Loading feature tensors ...")
    text_features = torch.load(args.text_features)
    image_features = torch.load(args.image_features)

    print(f"Text features:  {tuple(text_features.shape)}")
    print(f"Image features: {tuple(image_features.shape)}")

    assert text_features.shape[0] == image_features.shape[0], (
        "Row counts of text and image features differ"
    )

    # Normalise each modality independently before concatenation.
    text_features = l2_normalise(text_features)
    image_features = l2_normalise(image_features)

    print("Concatenating along the feature dimension ...")
    fused = torch.cat((text_features, image_features), dim=1)

    # Normalise the concatenated vector as a whole (removes output-scale
    # differences between the two encoders).
    fused = l2_normalise(fused)

    torch.save(fused, args.output)
    print(f"Fused features saved to: {args.output}")
    print(f"Final tensor shape: {tuple(fused.shape)}")


if __name__ == "__main__":
    main()
