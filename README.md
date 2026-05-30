# Speculative Encoding for Efficient Gigapixel Whole Slide Image Analysis

Official repository of the paper *'Speculative Encoding for Efficient Gigapixel Whole Slide Image Analysis'*

by Hoigi Seo<sup>\*</sup>, [Hyewon Bae](https://www.linkedin.com/in/hyewon-bae-05865a260)<sup>\*</sup>, Byung Hyun Lee<sup>\*</sup>, Jaehyun Cho, Joohoon Lee, Yonguk Kim, [Suh Yoon Jeon](https://www.linkedin.com/in/suhyoonjeon), Ji Ha Jang, Hayeon Kim, and [Se Young Chun](https://icl.snu.ac.kr/pi)<sup>†</sup>.

<sup>\*</sup> Equal contribution. &nbsp; <sup>†</sup> Corresponding author.

Link: arXiv (coming soon)

Reference implementation reproducing every cell of the paper's main table
(Tab. 1) on **CAMELYON16 (CM16)**, **CAMELYON17 (CM17)** and **TCGA-NSCLC**,
across **9 MIL aggregators** and **3 slide-level foundation models** (TITAN,
PRISM, Prov-GigaPath, plus a Prov-GigaPath full-finetune row).

```
Speculative Encoding = (cheap distilled patch encoder) → patch sampling
                       → only re-encode the sampled patches with the heavy
                          encoder → downstream MIL / slide encoder.
```

---

## Installation

One conda env covers everything — training, RAPIDS-accelerated sampler, eval.
Tested with Python 3.10, PyTorch 2.3+, CUDA 12.1 on H100 GPUs.

```bash
conda create -n speculative python=3.10 -y
conda activate speculative
pip install -r requirements.txt

# Prov-GigaPath slide encoder (only needed for the gigapath rows of Tab. 1):
pip install git+https://github.com/prov-gigapath/prov-gigapath.git
```

`requirements.txt` already pulls `cuml-cu12` + `cupy-cuda12x` from
`https://pypi.nvidia.com`. If you are on CUDA 11, swap them to `cuml-cu11` /
`cupy-cuda11x` before installing.

### Slide-encoder runtime gotchas

The HuggingFace slide encoders pull a few extra deps via `trust_remote_code`:

| Encoder | Extra dep (already in `requirements.txt`) | Notes |
| --- | --- | --- |
| TITAN | `einops-exts` | Used inside `MahmoodLab/TITAN`'s modeling file. |
| PRISM | `environs`, `protobuf`, `sentencepiece`, `sacremoses` | BioGPT text-decoder loader needs these even though we throw the decoder away. |
| Prov-GigaPath | `fairscale`, `flash-attn` | LongNet uses dilated attention (compiled CUDA). The `flash-attn` source build is slow — install a pre-built wheel matching your `(python, torch, cuda)` from <https://github.com/Dao-AILab/flash-attention/releases> if you can. |

The Prov-GigaPath python package itself ships a bundled `torchscale` —
do **not** also `pip install torchscale`, the system package shadows the
bundled one and breaks the LongNet import.

If the Prov-GigaPath rows complain about `force_download=True`, the gigapath
package re-downloads `slide_encoder.pth` even when present in HF cache. Pre-
download once and point the loader at it:

```bash
export GIGAPATH_SLIDE_CKPT=/path/to/slide_encoder.pth
```

### TITAN performance patch (recommended)

`MahmoodLab/TITAN`'s shipped `vision_transformer.py` builds its ALiBi
position bias with NumPy on the CPU per slide, which dominates runtime at
small patch budgets. For the paper's TITAN latency numbers, rewrite the body
of `get_alibi(...)` to compute the bias on-GPU (`torch.meshgrid` +
`torch.cdist`) — a behaviour-preserving change. The file lives in the
HuggingFace cache (`snapshot_download('MahmoodLab/TITAN')`), so re-apply it
after clearing the cache or when the upstream snapshot hash changes.

### External dependencies

The 9 MIL aggregators are vendored under `evaluator/mil/` (mostly from
[PathGen-1.6M / WSI_classification](https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology),
with `dftd.py` / `rrt.py` / `wikg.py` from [`mahmoodlab/MIL-Lab`](https://github.com/mahmoodlab/MIL-Lab)).
**No external clone is needed for Tab. 1.** Pre-trained slide encoders are
pulled at runtime:

| Source | Notes |
| --- | --- |
| `MahmoodLab/TITAN` (HuggingFace) | Auto-downloaded by `model/titan.py` at first use. *Recommended one-line patch — see "TITAN performance patch" below.* |
| `paige-ai/Prism`  (HuggingFace) | Auto-downloaded by `model/prism.py`. Gated → set `HF_TOKEN`. |
| `paige-ai/Virchow` (HuggingFace) | Patch-level tile encoder feeding PRISM. Gated → set `HF_TOKEN`. |
| [`prov-gigapath/prov-gigapath`](https://github.com/prov-gigapath/prov-gigapath) | Slide encoder code. `pip install git+https://...` or clone and point `$GIGAPATH_REPO` at it. |

---

## Configuration

`configs/paths.json` is the single source of truth for every path / interpreter:

```json
{
  "PYTHON":          "/path/to/conda/envs/speculative/bin/python",

  "WSI_ROOT":               "/data/raw_wsi",

  "FEATURE_ROOT":           "/data/patch_features/ps512/conch_v1_5",
  "COORD_ROOT":             "/data/patch_coords/ps512",
  "GIGAPATH_FEATURE_ROOT":  "/data/patch_features/ps256/provgigapath",
  "COORD_DIR_PS256":        "/data/patch_coords/ps256",
  "PRISM_FEATURE_ROOT":     "/data/patch_features/ps224/virchow",
  "COORD_DIR_PS224":        "/data/patch_coords/ps224",
  "DISTILLED_FEATURE_ROOT": "/data/distilled/patch_features/ps512/distilled_cls",

  "CM16_RAW_ROOT":   "/data/cm16_raw",     // contains lesion_annotations.zip
  "CM17_LABEL_CSV":  "/data/cm17/stages.csv",

  "CHECKPOINT_DIR":  "/runs/mil_checkpoints/cm16/checkpoints",

  "HF_HOME":         "/cache/hf",
  "HF_HUB_CACHE":    "/cache/hf/hub"
}
```

`eval.py` and every shell script auto-load this file (you can override with
`PATHS_JSON=/path/to/other.json`). All values listed above can also be passed
explicitly on the CLI via the matching `--feature-root`, `--coord-root`, etc.

### Datasets

| Dataset | Notes |
| --- | --- |
| **CAMELYON16** | Binary tumor / normal slide classification. Test labels read from the official `lesion_annotations.zip` under `$CM16_RAW_ROOT`. |
| **CAMELYON17** | Patient-level pN-staging. Requires `stages.csv` from the official challenge at `$CM17_LABEL_CSV`. |
| **TCGA-NSCLC** | LUAD vs LUSC. Patient-level stratified split. |
| **WSI-Bench** (Tab. 3, MLLM) | Pre-extracted patch features for the WSI-LLaVA report-generation benchmark. |

The codebase consumes **pre-extracted** patch features (`{slide}.pt`,
shape `[N_patches, D]`) plus per-slide `[N_patches, 2]` coord `.npy` files
in the CLAM-0402 layout. Pre-trained patch encoders are pulled from
HuggingFace at runtime. Splits at `splits/` were drawn with `seed=42`.

### Expected on-disk layout

The runners infer everything from the four root paths in `configs/paths.json`.
The directory structure under each root must match the CLAM-0402 layout:

```
$FEATURE_ROOT/                     # CONCH v1.5 patch features (768-d, ps=512)
├── cm16/
│   ├── train/  normal_001.pt  normal_002.pt  ...  tumor_001.pt  tumor_002.pt  ...
│   └── test/   test_001.pt    test_002.pt    ...
├── cm17/CAMELYON17/
│       patient_000_node_0.pt  patient_000_node_1.pt  ...  patient_099_node_4.pt
└── NSCLC/
    ├── LUAD/   TCGA-XX-XXXX-...-DX1.<UUID>.pt  ...
    └── LUSC/   TCGA-XX-XXXX-...-DX1.<UUID>.pt  ...

$COORD_ROOT/                       # patch coords for $FEATURE_ROOT (ps=512)
└── (same tree as $FEATURE_ROOT but each .pt → .npy with structured-array
     fields {x, y, tile_size_lv0})

$GIGAPATH_FEATURE_ROOT/            # Prov-GigaPath features (1536-d, ps=256)
└── (same tree as $FEATURE_ROOT but features come from prov-gigapath/prov-gigapath)

$COORD_DIR_PS256/                  # patch coords for the 256-px features
└── (same tree, .npy)

$PRISM_FEATURE_ROOT/                # Virchow tile features (2560-d, ps=224)
└── (same tree, .pt)

$DISTILLED_FEATURE_ROOT/            # output of the speculative-encoding student
└── (same tree as $FEATURE_ROOT, written by `scripts/distill.sh`)

$CM16_RAW_ROOT/                     # CAMELYON16 official annotations
├── test/  lesion_annotations_test.zip
└── train/ lesion_annotations_train.zip

$CM17_LABEL_CSV                     # CAMELYON17 stages.csv (single file)
                                    # columns: patient,stage,center
                                    # rows whose `patient` contains 'node' map to
                                    # the .pt slides above.
```

Each `.pt` file is a `torch.Tensor` of shape `[N_patches, D]`. Each `.npy`
coord file is a numpy structured array with fields `x`, `y`, `tile_size_lv0`
(N entries; coords are top-left pixel positions in level-0 space).

Slide naming: filename stems must match across feature root and coord root,
so the runner can pair them by slide id.

### Producing the feature roots

The repo includes `distill/extract_features.py` (wrapped by
`scripts/extract_features.sh`), which forwards either the **distilled student**
or the **original teacher** through an existing coord root and writes per-slide
``.pt`` files in the right layout. It only needs ``openslide-python`` to read
the raw WSI files; everything else is already in `requirements.txt`.

```bash
# (a) Original CONCH v1.5 features → $FEATURE_ROOT
bash scripts/extract_features.sh \
    --teacher_model conchv15 \
    --wsi_root   /data/raw_wsi/cm16 \
    --coord_root $COORD_ROOT/cm16 \
    --output_root $FEATURE_ROOT/cm16

# (b) Distilled student features → $DISTILLED_FEATURE_ROOT (sampler input)
bash scripts/extract_features.sh \
    --checkpoint outputs/distilled_models/<run>/checkpoint_10000.pt \
    --wsi_root   /data/raw_wsi/cm16 \
    --coord_root $COORD_ROOT/cm16 \
    --output_root $DISTILLED_FEATURE_ROOT/cm16

# Multi-GPU (8 ranks):
NPROC=8 bash scripts/extract_features.sh ...
```

Repeat (a) per encoder (`conchv15` for `$FEATURE_ROOT`, `virchow` for
`$PRISM_FEATURE_ROOT`, `provgigapath` for `$GIGAPATH_FEATURE_ROOT`) and (b)
once for the distilled student.

> **The repo does not extract patch coords from raw WSI** — that step (tissue
> mask + tiling + filtering) is delegated to standard tools such as
> [`mahmoodlab/CLAM`](https://github.com/mahmoodlab/CLAM)
> (`create_patches_fp.py`) or [`mahmoodlab/TRIDENT`](https://github.com/mahmoodlab/TRIDENT).
> Their ``.npy`` coord output drops directly into ``$COORD_ROOT``.

### From-scratch reproduction pipeline

Reviewers wanting to start from raw WSI files run the steps below once per
dataset. Steps **0** and **1** use external tools; **2–5** use the scripts in
this repo. The numbers align with the section anchors used inside
`scripts/`. Three coord roots are needed because the three slide encoders
expect different patch sizes (CONCH v1.5 → 512 px, Prov-GigaPath → 256 px,
Virchow/PRISM → 224 px); each is reused across all the slides in the
corresponding dataset.

#### Step 0. Acquire the raw WSI files

| Dataset | Source | Files needed |
| --- | --- | --- |
| CAMELYON16 | <https://camelyon17.grand-challenge.org/Data/> | All `train/` (normal_*.tif, tumor_*.tif) and `test/` (test_*.tif) slides + `test/lesion_annotations_test.zip` (test labels) |
| CAMELYON17 | <https://camelyon17.grand-challenge.org/Data/> | All `images/` (`patient_XXX_node_Y.tif`) + `stages.csv` |
| TCGA-NSCLC | [GDC Data Portal](https://portal.gdc.cancer.gov/) (TCGA-LUAD, TCGA-LUSC) | `*-DX1.<UUID>.svs` slides only (FFPE diagnostic) |

Place them in any directory; we'll reference them as `$WSI_ROOT/<dataset>/`.
Set `$CM16_RAW_ROOT` to the CM16 directory containing
`{train,test}/lesion_annotations_*.zip`, and `$CM17_LABEL_CSV` to the CM17
`stages.csv`.

#### Step 1. Extract patch coords (3 patch sizes per dataset)

[CLAM](https://github.com/mahmoodlab/CLAM)'s `create_patches_fp.py` does
everything — tissue segmentation + tiling — in one tool. We run it **three
times per dataset**, once per patch size. Tissue segmentation is identical
across the three runs, so we compute the masks once with `--seg --patch` and
then reuse them for the other two sizes via `--seg_dir`. The exact
invocations we used:

```bash
# (a) Compute tissue masks + 512-px coords (CONCH v1.5)
python create_patches_fp.py \
    --source $WSI_ROOT/cm16 \
    --save_dir $COORD_ROOT/cm16 \
    --patch_size 512 --step_size 512 --patch_level 0 \
    --preset tcga.csv  --seg --patch --no_auto_skip

# (b) Reuse the masks for 256-px coords (Prov-GigaPath)
python create_patches_fp.py \
    --source $WSI_ROOT/cm16 \
    --save_dir $COORD_DIR_PS256/cm16 \
    --patch_size 256 --step_size 256 --patch_level 0 \
    --preset tcga.csv  --patch --no_auto_skip \
    --seg_dir $COORD_ROOT/cm16/masks

# (c) Reuse the masks for 224-px coords (Virchow / PRISM)
python create_patches_fp.py \
    --source $WSI_ROOT/cm16 \
    --save_dir $COORD_DIR_PS224/cm16 \
    --patch_size 224 --step_size 224 --patch_level 0 \
    --preset tcga.csv  --patch --no_auto_skip \
    --seg_dir $COORD_ROOT/cm16/masks
```

Repeat for `cm17` (`--source $WSI_ROOT/cm17/CAMELYON17`) and `nsclc`
(`--source $WSI_ROOT/nsclc`).

CLAM writes `<save_dir>/patches/<slide_id>.h5` with the patch coords.
Convert each to the `.npy` structured array expected by this repo:

```bash
python -c "
import h5py, numpy as np
from pathlib import Path
for h5 in Path('$COORD_ROOT').rglob('*.h5'):
    with h5py.File(h5) as f:
        c = f['coords'][:]
        ts = f['coords'].attrs.get('patch_size', 512)
    arr = np.array([(int(x), int(y), int(ts)) for x,y in c],
                   dtype=[('x','i8'),('y','i8'),('tile_size_lv0','i8')])
    np.save(h5.with_suffix('.npy'), arr)"
```

#### Step 2. Extract teacher features (per encoder, per dataset)

```bash
# CONCH v1.5  →  $FEATURE_ROOT
NPROC=8 bash scripts/extract_features.sh \
    --teacher_model conchv15 \
    --wsi_root   $WSI_ROOT/cm16 \
    --coord_root $COORD_ROOT/cm16 \
    --output_root $FEATURE_ROOT/cm16

# Prov-GigaPath tile encoder  →  $GIGAPATH_FEATURE_ROOT
NPROC=8 bash scripts/extract_features.sh \
    --teacher_model provgigapath \
    --wsi_root   $WSI_ROOT/cm16 \
    --coord_root $COORD_DIR_PS256/cm16 \
    --output_root $GIGAPATH_FEATURE_ROOT/cm16

# Virchow tile encoder (PRISM input)  →  $PRISM_FEATURE_ROOT
NPROC=8 bash scripts/extract_features.sh \
    --teacher_model virchow \
    --wsi_root   $WSI_ROOT/cm16 \
    --coord_root $COORD_DIR_PS224/cm16 \
    --output_root $PRISM_FEATURE_ROOT/cm16
```

Repeat for `cm17` and `nsclc`. Set `HF_TOKEN` once for the gated
HuggingFace repos (`paige-ai/Virchow`, `prov-gigapath/prov-gigapath`).

#### Step 3. Distill the student patch encoder (Sec. 3.2)

Distillation reuses the CONCH v1.5 patch images (or any unlabeled patch image
directory). The training script crawls `$DISTILL_DATA_DIR` for
`*.{jpg,jpeg,png}` files, so dump CLAM's patch PNGs (or run any other tiling
that produces patch images) under `$DISTILL_DATA_DIR/`:

```bash
# Convert CLAM h5 patches to PNGs (one-time):
python -c "
import h5py, openslide, os
from pathlib import Path
from PIL import Image
for h5 in Path('$COORD_ROOT').rglob('*.h5'):
    slide_id = h5.stem
    wsi = next(Path('$WSI_ROOT').rglob(f'{slide_id}.*'))
    osl = openslide.OpenSlide(str(wsi))
    out = Path('$DISTILL_DATA_DIR') / slide_id; out.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5) as f:
        for i, (x, y) in enumerate(f['coords'][:]):
            tile = osl.read_region((int(x), int(y)), 0, (512, 512)).convert('RGB')
            tile.save(out / f'{i:06d}.png')"
```

Then launch distillation:

```bash
bash scripts/distill.sh configs/distill/conchv15.json
# → outputs/distilled_models/<run_tag>/checkpoint_<step>.pt
```

The exact distillation hyperparameters used in the paper are pinned in
`configs/distill/conchv15.json` (8 H100 GPUs, 10k steps, batch 1024,
bf16/fp16 mixed precision, c25A loss recipe).

#### Step 4. Extract distilled-student features

This produces the sampler input.

```bash
NPROC=8 bash scripts/extract_features.sh \
    --checkpoint outputs/distilled_models/<run_tag>/checkpoint_10000.pt \
    --wsi_root   $WSI_ROOT/cm16 \
    --coord_root $COORD_ROOT/cm16 \
    --output_root $DISTILLED_FEATURE_ROOT/cm16
```

Repeat for `cm17` and `nsclc`.

#### Step 5. Train the 9 MIL aggregators (one-off per dataset)

```bash
bash scripts/train_mil_checkpoints.sh --dataset cm16
bash scripts/train_mil_checkpoints.sh --dataset cm17
bash scripts/train_mil_checkpoints.sh --dataset nsclc
# → $CHECKPOINT_DIR/<arch>_fold<N>_best.pt
```

Hyperparameters: `train_epoch=30  lr=1e-4  wd=1e-5  eval_interval=5`
(paper defaults — override with `TRAIN_EPOCH=` etc. if you want).

#### Step 6. Run Tab. 1 — at this point everything is local

```bash
python eval.py --config configs/experiments/main_table/cm17_titan_ours_b25.json
python eval.py --config configs/experiments/main_table/cm17_mil_all_ours_b25.json \
               --checkpoint-dir $CHECKPOINT_DIR
# … one config per row …
```

The 24 ready-made configs at `configs/experiments/main_table/` cover every
cell in Tab. 1.

---

## TL;DR — running one experiment

Once `configs/paths.json` is filled in:

```bash
# Reproduce CM16 + TITAN, +Ours @ 25% budget (Tab. 1, TITAN row, CM16 column)
python eval.py --config configs/experiments/main_table/cm16_titan_ours_b25.json

# Same row but override the output dir on the fly
python eval.py --config configs/experiments/main_table/cm16_titan_ours_b25.json \
               --output-dir outputs/my_run

# All 12 models on CM16 (9 MIL + TITAN + PRISM + Prov-GigaPath)
python eval.py --dataset cm16 --model all --budget 0.25 \
               --sampling-config configs/sampling/main_table/cm16_default.json \
               --checkpoint-dir outputs/mil_checkpoints/cm16/checkpoints

# Random-baseline cell from Tab. 2 (motivation)
python eval.py --dataset cm16 --model abmil --budget 0.25 \
               --sampling-mode random \
               --checkpoint-dir outputs/mil_checkpoints/cm16/checkpoints
```

Every flag has an env-var equivalent loaded from `configs/paths.json`, so
once your dataset paths are filled in, you only need `--dataset`, `--model`,
`--budget` (and a sampler config).

---

## Repository layout

```
speculative_encoding/
├── README.md
├── requirements.txt              # pip deps for the eval / MIL env (see "Installation")
├── eval.py                       # ★ single CLI for every paper experiment
│
├── configs/
│   ├── paths.json                  # one place for every dataset / cache path
│   ├── distill/                    # student-encoder distillation hyperparams
│   ├── sampling/                   # sampler hyperparams
│   │   ├── canonical_25pct.json      # c25A (Sec. 3.2 leader recipe @ 25%)
│   │   ├── canonical_10pct.json      # @ 10%
│   │   ├── random_baseline.json    grid_baseline.json    kmeans_baseline.json
│   │   └── main_table/             # one file per Tab. 1 cell whose sampler
│   │                                 differs from the canonical recipe.
│   ├── ablation/                   # A2..A10 (Tab. 4)
│   ├── hp_ablation/                # κ, τ_b, K_n, λ sweeps (App. F Tab. 6)
│   └── experiments/main_table/     # ★ ready-to-run JSONs for every Tab. 1 row,
│                                    pass to `eval.py --config <path>`.
│
├── scripts/
│   ├── load_paths.sh               # parses configs/paths.json into env vars (sourced by every other script)
│   ├── distill.sh                  # config-driven patch-encoder distillation
│   ├── train_mil_checkpoints.sh    # ★ train 9 MIL aggregators on a dataset (one-off)
│   ├── sample.sh                   # config-driven sampler (alternative entry; eval.py calls `python -m sampling` directly)
│   ├── project_features.sh         # MLP-projector forward over distilled features
│   └── train_mlp_projector.sh      # train the MLP projector (ablation A6 + feature fill)
│
├── distill/                       # patch-encoder distillation (Sec. 3.2)
├── sampling/                      # inference-time patch sampler (Sec. 3.3)
├── model/                         # self-contained slide encoders
│   ├── titan.py                     # ★ TITAN wrapper (HF MahmoodLab/TITAN — fully inlined)
│   ├── prism.py                     # ★ PRISM wrapper (HF paige-ai/Prism)
│   └── gigapath.py                  # Prov-GigaPath wrapper (loads `gigapath` python package)
│
├── evaluator/
│   ├── metrics.py                   # acc / precision / recall / macro_f1 / auroc
│   ├── mil/                         # 9 vendored MIL aggregators (builder + arch files)
│   └── runners/
│       ├── mil_subsample.py           # 9-MIL eval at any patch budget (loads checkpoints)
│       ├── mil_comparison.py          # MIL training entry (saves per-fold checkpoints)
│       ├── titan_subsample.py         # TITAN linear-probe at any patch budget
│       ├── prism_subsample.py         # PRISM linear-probe at any patch budget
│       ├── gigapath_subsample.py      # Prov-GigaPath linear-probe + full FT
│       ├── feasibility_subsample.py   # shared helpers: subsample_indices, titan/gigapath_extract_embeddings
│       └── custom_index_utils.py      # load pre-computed sampler indices
│
└── splits/                        # pre-computed 5-fold splits used everywhere
    └── {cm16,cm17,nsclc}_*_seed42_n5_test20{.json,/}
```

---

## Pretrained models

We release the **distilled Prov-GigaPath student patch encoder** used to
produce the sampler input for the Prov-GigaPath rows of Tab. 1:

- **Google Drive:** [`distill_step_10000.pt`](https://drive.google.com/file/d/1Z_tgfIVzH8zRWX5r0CSbVyX3AOqe_Aov/view?usp=sharing) (≈495 MB)

Download the `.pt` file and pass it via `--checkpoint` when running
`scripts/extract_features.sh` (Step 4).

> The distilled CONCH v1.5 and Virchow students are **not** redistributed,
> due to the upstream teacher-model licenses. They can be reproduced with the
> from-scratch pipeline above (Step 3) given access to the gated teachers.

## Citations

This code is heavily based on
- [`mahmoodlab/CLAM`](https://github.com/mahmoodlab/CLAM) and [`mahmoodlab/TRIDENT`](https://github.com/mahmoodlab/TRIDENT) — patch-coordinate / tissue extraction and MIL baselines.
- [`mahmoodlab/MIL-Lab`](https://github.com/mahmoodlab/MIL-Lab) — MIL aggregators (DFTD, RRT, WiKG, and others).
- [`mahmoodlab/CONCH`](https://github.com/mahmoodlab/CONCH) — the `open_clip_custom` CONCH implementation.

We also build on and evaluate the following pretrained encoders and datasets;
please cite their original papers and respect their licenses: CONCH / UNI,
TITAN ([`MahmoodLab/TITAN`](https://huggingface.co/MahmoodLab/TITAN)),
PRISM ([`paige-ai/Prism`](https://huggingface.co/paige-ai/Prism)) with
Virchow ([`paige-ai/Virchow`](https://huggingface.co/paige-ai/Virchow)),
Prov-GigaPath ([`prov-gigapath/prov-gigapath`](https://github.com/prov-gigapath/prov-gigapath)),
CAMELYON16 / CAMELYON17 (<https://camelyon17.grand-challenge.org/>), and
TCGA-LUAD / TCGA-LUSC (<https://portal.gdc.cancer.gov/>).

## bibTeX

If our code is helpful for your research, please consider citing
```
@article{seo2026speculative,
  title   = {Speculative Encoding for Efficient Gigapixel Whole Slide Image Analysis},
  author  = {Seo, Hoigi and Bae, Hyewon and Lee, Byung Hyun and Cho, Jaehyun and Lee, Joohoon and Kim, Yonguk and Jeon, Suh Yoon and Jang, Ji Ha and Kim, Hayeon and Chun, Se Young},
  journal = {arXiv preprint (coming soon)},
  year    = {2026},
}
```

## License

No license is currently specified for this repository. The code is provided
for academic and research use; for any other use — and for the pretrained
encoders and datasets it depends on — refer to each upstream model / dataset
license (CONCH, UNI, PRISM, Prov-GigaPath, TITAN, CAMELYON16/17, TCGA).

Identifying paths, internal hostnames, and credentials have been removed —
every absolute path is resolved from `configs/paths.json` at runtime. The
pretrained teacher encoders are downloaded from their public (some gated)
HuggingFace repositories and, except for the distilled Prov-GigaPath student
above, are not redistributed here.
