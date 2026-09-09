# Intermediate artifacts — frozen embeddings & Semantic-IDs

This folder intentionally ships **only this README + `MANIFEST.csv`** in the
repository and in the Moodle submission archive.

The actual intermediate tensors of the reported runs — the frozen item
embeddings and the Semantic-ID pickles that were fed to TIGER — total
**≈ 980 MB** (individual files up to ~600 MB).  They exceed:

* the **Moodle hard upload limit of 230 MB** for the whole submission, and
* **GitHub's 100 MB per-file limit** for normal git pushes,

so they are distributed as a **GitHub Release archive** instead of being
committed to this repository or bundled into the Moodle zip.

## Download the full set

**Release:** https://github.com/zhanghanyu1722-a11y/Text_SID_3158754Z_MSc_Project/releases/latest

Asset: `MSc_project_3158754Z_full_artifacts.zip`

After extracting you get:

```
Text_SID_3158754Z_MSc_Project/
├── artifacts/
│   ├── MANIFEST.csv                       sha256 + size of every tensor
│   ├── text_qwen3_0.6B/embedding/qwen3_text_0.6B_embeds.pt
│   ├── text_qwen3_0.6B/semantic_id/semantic_ids.pt
│   ├── text_qwen3_8B/embedding/qwen3_text_8B_embeds.pt
│   ├── text_qwen3_8B/semantic_id/semantic_ids.pt
│   ├── mmA_clip_text_concat/embedding/{clip_image_embeds,mmA_text_clip_concat_embeds}.pt
│   ├── mmA_clip_text_concat/semantic_id/semantic_ids.pt
│   ├── mmB_qwen3vl_2B/embedding/mmB_qwen3vl_2B_embeds.pt
│   └── mmB_qwen3vl_2B/semantic_id/semantic_ids.pt
└── (code / configs / results / docs …)
```

`MANIFEST.csv` (included here) lists the SHA-256 and byte size of every
tensor, so you can verify a downloaded archive:

```bash
# compare the shipped manifest with an extracted tree
python - <<'EOF'
import csv, hashlib, os
base = "Text_SID_3158754Z_MSc_Project/artifacts"
ok, bad = 0, []
with open(os.path.join(base, "MANIFEST.csv")) as f:
    for row in csv.DictReader(f):
        p = os.path.join(base, row["relative_path"])
        h = hashlib.sha256(open(p, "rb").read()).hexdigest().lower()
        (ok := ok + 1) if h == row["sha256"].lower() else bad.append(row["relative_path"])
print("verified:", ok, "mismatched:", bad)
EOF
```

## What the tensors are / how to use them

* One directory per configuration (`text_qwen3_0.6B`, `text_qwen3_8B`,
  `mmA_clip_text_concat`, `mmB_qwen3vl_2B`), each with `embedding/` and
  `semantic_id/` files, matching the canonical cache names used by the
  pipeline.  See [`../docs/artifacts_reference.md`](../docs/artifacts_reference.md)
  for shape/dtype/provenance details.
* **Skip embedding extraction:** copy each `embedding/*.pt` to the same
  canonical name under `<GRID_ROOT>/data/microLen/`; the pipeline detects the
  cache and jumps straight to RQ-VAE / TIGER (see the main README, "Usage").
* **Skip RQ-VAE as well:** the `semantic_id/semantic_ids.pt` pickles are the
  exact `semantic_id_path` inputs of the reported TIGER runs.

If you do not need to inspect or reuse the exact tensors, simply run the
pipeline from scratch — it regenerates every file (see the main README).
