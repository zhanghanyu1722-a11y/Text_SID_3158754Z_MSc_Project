# Acknowledgments, data and code provenance

## Author of this package
The code in this package was written by **Hanyu Zhang** for the MSc project
*"Text-Image Semantic IDs for Generative Recommendation"* (supervisor:
Prof. Joemon M. Jose, co-supervisor: Junchen Fu, School of Computing Science,
University of Glasgow).  It packages, cleans and documents the scripts used to
produce the experiments reported in the accompanying dissertation.

## What this package contains vs. what it references

This package ships **only the student-authored pieces** plus the exact
configuration and log files of the reported runs:

| Included here | Origin |
|---|---|
| `configs/*.yaml` | hydra experiment configs used for the reported runs (three are pinned copies of the upstream GRID configs the runs used; `rqvae_inference_flat.yaml` was added by the author because upstream GRID does not ship an RQ-VAE inference config) |
| `scripts/*.py` | preprocessing / embedding-extraction / fusion scripts written by the author (cleaned and parameterised for this package) |
| `pipeline/*.sh` | end-to-end reproduction scripts written for this package |
| `results/` | byte-exact copies of the logged CSVs that produced the dissertation tables |
| `artifacts/` | **guide** to the intermediate tensors of the reported runs: this archive/repository ships `README.md` + `MANIFEST.csv`; the full ~980 MB `.pt` set (the author's own computed representations; no raw dataset files) is distributed as a GitHub Release archive (see `artifacts/README.md`) |
| `docs/`, `README.md` | documentation |

Everything else (the GRID training/inference codebase, the datasets, the
pretrained models, the Amazon P5 data) is **referenced, not redistributed**;
see the README for exact acquisition steps.

## Third-party code, data and models (please cite)

- **TIGER** — Rajput, Shashank, et al. *"Recommender systems with generative
  retrieval."* Advances in Neural Information Processing Systems 36 (2023).
  https://arxiv.org/abs/2305.05065
- **GRID codebase** — *"Generative Recommendation with Semantic IDs: A
  Practitioner's Handbook"* (CIKM 2025), Snap Research.
  Code: https://github.com/snap-research/GRID (arXiv: https://arxiv.org/abs/2507.22224).
  The experiments were run inside a clone of this repository (the dissertation
  refers to it as "the reference implementation / GRID").  Respect its license
  when redistributing derived work.
- **MicroLens dataset** — Ni, Yongxin, et al. *"A Content-Driven Micro-Video
  Recommendation Dataset at Scale."* arXiv:2309.15379.
  Code/data portal: https://github.com/westlake-repl/MicroLens and
  https://recsys.westlake.edu.cn/.
- **Amazon Beauty (validation)** — P5-preprocessed data whose download link is
  given in the GRID README (original reviews: He & McAuley, WWW 2016).
- **Qwen3-Embedding** — Zhang, et al. *"Qwen3 Embedding: Advancing Text
  Embedding and Reranking through Foundation Models."* arXiv:2506.05176.
  Models `Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Embedding-8B`
  (https://huggingface.co/Qwen).
- **Qwen3-VL-Embedding** — Li, et al. *"Qwen3-VL-Embedding and
  Qwen3-VL-Reranker."* arXiv:2601.04720. Model `Qwen/Qwen3-VL-Embedding-2B`
  (https://huggingface.co/Qwen).
- **CLIP** — Radford, et al. *"Learning transferable visual models from
  natural language supervision."* ICML 2021 (https://arxiv.org/abs/2103.00020);
  OpenAI implementation: https://github.com/openai/CLIP.

## License & redistribution notes

- **MicroLens.** The dataset authors explicitly prohibit privately modified
  secondary redistribution of the dataset.  This package therefore does **not**
  ship any raw MicroLens files (interactions, titles, covers); reproduce them
  by downloading the official files (see README, "Data acquisition").  The
  derived intermediate tensors under `artifacts/` (frozen embeddings and
  Semantic-ID pickles computed by the author from the downloaded data for the
  reported runs) are distributed as a GitHub Release archive rather than being
  bundled here, because of the Moodle 230 MB / GitHub 100 MB-per-file limits;
  treat them with the same care as the rest of the dissertation material.
- **Models and the GRID codebase** are subject to their respective licenses;
  model checkpoints and pretrained weights keep their original licenses when
  downloaded.
- The author's own scripts in this package may be reused freely for academic
  purposes with attribution.
