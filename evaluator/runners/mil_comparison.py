"""
Evaluation-dataset MIL Comparison — CONCHv1.5 features
=======================================================

BRCA / Camelyon16 / Camelyon17에 대해 여러 MIL 방법 비교:
    ABMIL, CLAM, DFTD, DSMIL, ILRA, RRT, Transformer, TransMIL, WiKG

평가: k-fold stratified CV, mean ± std 비교 테이블

Data:
    ${FEATURE_ROOT}/
    각 .pt 파일: Tensor [N_patches, 768]

Usage:
    PYTHON=python

    # Camelyon16 (normal vs tumor, 5-fold CV on train split)
    $PYTHON -m eval.mil_comparison --dataset cm16

    # Camelyon17 (patient-level pN-staging, 5-fold CV, requires --label_csv)
    $PYTHON -m eval.mil_comparison --dataset cm17 --label_csv /path/to/cm17_labels.csv
"""

import argparse
import json
import os
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import csv

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── Repo path setup ──────────────────────────────────────────────────────────
# REPO_ROOT is the speculative_encoding/ folder (two levels above this file).
# load_paths.sh prepends it to PYTHONPATH already; we re-add it defensively.
SPEC_ENC_PATH = str(Path(__file__).resolve().parent.parent.parent)
if SPEC_ENC_PATH not in sys.path:
    sys.path.insert(0, SPEC_ENC_PATH)

# Default data roots come from env. Pass explicit --feature_root / --cm16_raw_root
# / --label_csv on the command line to override per call.
FEATURE_ROOT     = os.environ.get("FEATURE_ROOT", "")
FEATURE_DIM      = 768
PANDA_LABEL_CSV  = os.path.join(os.environ.get("PANDA_RAW_ROOT", ""), "train.csv") \
                   if os.environ.get("PANDA_RAW_ROOT") else ""
CM16_RAW_ROOT    = os.environ.get("CM16_RAW_ROOT", "")
SHARED_SPLIT_ROOT = Path(SPEC_ENC_PATH) / "splits"

ALL_MIL_ARCHS = [
    'abmil', 'clam_sb', 'dftd', 'dsmil', 'ilra', 'rrt', 'mha', 'transmil', 'wikg',
]

# Human-readable names for table display
ARCH_DISPLAY = {
    'abmil':    'ABMIL',
    'clam_sb':  'CLAM',
    'dftd':     'DFTD',
    'dsmil':    'DSMIL',
    'ilra':     'ILRA',
    'rrt':      'RRT',
    'mha':      'Transformer',
    'transmil': 'TransMIL',
    'wikg':     'WiKG',
}

from evaluator.metrics import get_eval_metrics_from_probs
from evaluator.mil import build_net, make_conf, set_seed, train_one_epoch, evaluate as mil_evaluate


# ── Data loading ──────────────────────────────────────────────────────────────

def load_slide(path: Path) -> torch.Tensor:
    """Load one .pt file → [N_patches, D]."""
    x = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, dict):
        key = 'features' if 'features' in x else list(x.keys())[0]
        return x[key].float()
    raise TypeError(f"Unexpected .pt content: {type(x)}")


# ── Dataset collectors ────────────────────────────────────────────────────────

def collect_brca(feature_root: str) -> Tuple[List[Path], List[int], List[str], List[str]]:
    """BRCA: IDC (label=0) vs ILC (label=1), from folder names."""
    brca_dir = Path(feature_root) / 'BRCA'
    class_names = ['IDC', 'ILC']
    paths, labels, names = [], [], []
    for label_idx, cls_name in enumerate(class_names):
        cls_dir = brca_dir / cls_name
        for pt_path in sorted(cls_dir.glob('*.pt')):
            paths.append(pt_path)
            labels.append(label_idx)
            names.append(pt_path.stem)
    print(f"\n[BRCA] {len(paths)} slides — IDC: {labels.count(0)}, ILC: {labels.count(1)}")
    return paths, labels, names, class_names


def _cm16_tumor_ids_from_zip(zip_path: Path) -> set[str]:
    if not zip_path.exists():
        raise FileNotFoundError(f'CM16 annotation zip not found: {zip_path}')
    tumor_ids = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith('.xml'):
                continue
            tumor_ids.add(Path(name).stem)
    return tumor_ids


def collect_cm16(
    feature_root: str,
    split: str = 'train',
    raw_root: Optional[str] = None,
) -> Tuple[List[Path], List[int], List[str], List[str]]:
    """Camelyon16 split loader supporting train-prefix labels and official test labels via annotation zips."""
    if split == 'all':
        tr_paths, tr_labels, tr_names, class_names = collect_cm16(
            feature_root,
            split='train',
            raw_root=raw_root,
        )
        te_paths, te_labels, te_names, _ = collect_cm16(
            feature_root,
            split='test',
            raw_root=raw_root,
        )
        paths = tr_paths + te_paths
        labels = tr_labels + te_labels
        names = tr_names + te_names
        print(f"\n[CM16:all] {len(paths)} slides — normal: {labels.count(0)}, tumor: {labels.count(1)}")
        return paths, labels, names, class_names

    cm16_dir = Path(feature_root) / 'cm16' / split
    class_names = ['normal', 'tumor']
    paths, labels, names = [], [], []
    if split == 'test':
        raw_root = raw_root or CM16_RAW_ROOT
        tumor_ids = _cm16_tumor_ids_from_zip(Path(raw_root) / 'test' / 'lesion_annotations_test.zip')
    else:
        tumor_ids = set()

    for pt_path in sorted(cm16_dir.glob('*.pt')):
        fname = pt_path.stem
        if split == 'train':
            if fname.startswith('normal'):
                labels.append(0)
            elif fname.startswith('tumor'):
                labels.append(1)
            else:
                print(f"  [WARN] Unknown prefix, skipping: {fname}")
                continue
        elif split == 'test':
            labels.append(1 if fname in tumor_ids else 0)
        else:
            raise ValueError(f'Unsupported cm16 split: {split}')
        paths.append(pt_path)
        names.append(fname)
    print(f"\n[CM16:{split}] {len(paths)} slides — normal: {labels.count(0)}, tumor: {labels.count(1)}")
    return paths, labels, names, class_names


def collect_cm17(
    feature_root: str,
    label_csv: Optional[str] = None,
) -> Tuple[List[Path], List[int], List[str], List[str]]:
    """
    Camelyon17: Slide-level (node-level) classification.
    
    Reads CSV with columns: patient, stage, center
    Maps node files (e.g., patient_000_node_0.tif) to their respective stage.
    """
    # CM17 feature layout has historically been either
    #   <feature_root>/cm17/CAMELYON17/images/*.pt   (older)
    # or
    #   <feature_root>/cm17/CAMELYON17/*.pt          (current)
    # Pick whichever exists.
    cm17_base = Path(feature_root) / 'cm17' / 'CAMELYON17'
    images_subdir = cm17_base / 'images'
    cm17_dir = images_subdir if images_subdir.is_dir() else cm17_base

    if label_csv is None:
        raise ValueError("Camelyon17 requires --label_csv")

    label_map = {}
    string_labels = set()

    # 1. CSV 파싱 및 Node-level 데이터만 필터링
    with open(label_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row['patient'].strip()
            stage_str = row['stage'].strip()

            # patient_XXX.zip 같은 환자 단위 행은 건너뛰고 node만 처리
            if 'node' not in file_name:
                continue

            # CSV의 '.tif'를 제거하여 pt_path.stem(예: patient_000_node_0)과 키를 맞춤
            key = file_name.replace('.tif', '')
            label_map[key] = stage_str
            string_labels.add(stage_str)

    if not label_map:
        raise ValueError("No node-level labels found in the CSV. Please check the CSV format.")

    # 💡 디버깅 코드 추가: CSV에는 있는데 실제 .pt 파일이 없는 녀석 찾기
    csv_keys = set(label_map.keys())
    pt_keys = set(p.stem for p in cm17_dir.glob('*.pt'))
    missing_files = csv_keys - pt_keys
    
    if missing_files:
        print(f"\n[경고] CSV에는 있지만 실제 .pt 파일이 누락된 슬라이드 ({len(missing_files)}개):")
        for missing in missing_files:
            print(f"  - {missing}.pt (Label: {label_map[missing]})")

    # 2. 문자열 라벨(negative, macro 등)을 정수형(0, 1, 2...)으로 매핑
    sorted_classes = sorted(list(string_labels))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}
    class_names = sorted_classes

    paths, labels, names = [], [], []

    # 3. 디렉토리 내의 .pt 피처 파일과 CSV 라벨 매핑
    for pt_path in sorted(cm17_dir.glob('*.pt')):
        fname = pt_path.stem  # 예: patient_000_node_0
        
        if fname not in label_map:
            continue
            
        stage_str = label_map[fname]
        label_idx = class_to_idx[stage_str]

        paths.append(pt_path)
        labels.append(label_idx)
        names.append(fname)

    # 4. 결과 출력
    n_class = len(class_names)
    print(f"\n[CM17] {len(paths)} slides (slide-level, {n_class}-class)")
    for i, c_name in enumerate(class_names):
        print(f"  [{i}] {c_name}: {labels.count(i)} slides")

    return paths, labels, names, class_names


def collect_panda(
    feature_root: str,
    label_csv: Optional[str] = None,
) -> Tuple[List[Path], List[int], List[str], List[str]]:
    """PANDA: ISUP grade classification from train.csv + train_images patch features."""
    panda_dir = (
        Path(feature_root)
        / 'prostate-cancer-grade-assessment'
        / 'train_images'
    )
    csv_path = Path(label_csv) if label_csv is not None else Path(PANDA_LABEL_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(f'PANDA label CSV not found: {csv_path}')

    paths, labels, names = [], [], []
    missing = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row['image_id'].strip()
            pt_path = panda_dir / f'{slide_id}.pt'
            if not pt_path.exists():
                missing.append(slide_id)
                continue
            paths.append(pt_path)
            labels.append(int(row['isup_grade']))
            names.append(slide_id)

    class_ids = sorted(set(labels))
    class_names = [str(class_id) for class_id in class_ids]

    print(f"\n[PANDA] {len(paths)} slides ({len(class_names)}-class ISUP)")
    for class_id in class_ids:
        print(f"  [{class_id}] ISUP {class_id}: {labels.count(class_id)} slides")
    if missing:
        print(f"  [WARN] Missing feature files for {len(missing)} slides (skipped)")

    return paths, labels, names, class_names


def get_tcga_patient_id(slide_name: str) -> str:
    """Extract TCGA patient ID from a slide filename/stem."""
    base = Path(slide_name).stem
    parts = base.split('-')
    if len(parts) < 3:
        raise ValueError(f'Cannot parse TCGA patient id from: {slide_name}')
    return '-'.join(parts[:3])


def collect_nsclc(feature_root: str) -> Tuple[List[Path], List[int], List[str], List[str]]:
    """NSCLC: LUAD (0) vs LUSC (1), labels from folder names."""
    nsclc_dir = Path(feature_root) / 'NSCLC'
    class_names = ['LUAD', 'LUSC']
    paths, labels, names = [], [], []
    for label_idx, cls_name in enumerate(class_names):
        cls_dir = nsclc_dir / cls_name
        for pt_path in sorted(cls_dir.glob('*.pt')):
            paths.append(pt_path)
            labels.append(label_idx)
            names.append(pt_path.stem)

    patient_to_labels: Dict[str, set] = {}
    for name, label in zip(names, labels):
        patient_id = get_tcga_patient_id(name)
        patient_to_labels.setdefault(patient_id, set()).add(label)
    mixed = [patient_id for patient_id, label_set in patient_to_labels.items() if len(label_set) > 1]
    if mixed:
        raise RuntimeError(f'Found patients with mixed LUAD/LUSC labels: {mixed[:10]}')

    print(f"\n[NSCLC] {len(paths)} slides — LUAD: {labels.count(0)}, LUSC: {labels.count(1)}")
    print(
        f"        {len(patient_to_labels)} patients — "
        f"LUAD: {sum(next(iter(v)) == 0 for v in patient_to_labels.values())}, "
        f"LUSC: {sum(next(iter(v)) == 1 for v in patient_to_labels.values())}"
    )
    return paths, labels, names, class_names


# ── Slide-level dataset (for MIL engine) ─────────────────────────────────────

class SlideDataset(Dataset):
    """Returns {'input': [1, N, D], 'label': [1]} per slide."""

    def __init__(self, paths, labels):
        self.paths  = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        # p can be a single Path or a list of Paths (for cm17 patient-level)
        if isinstance(p, list):
            feats = [load_slide(pp) for pp in p]
            feat = torch.cat(feats, dim=0)
        else:
            feat = load_slide(p)

        label = torch.tensor([self.labels[idx]], dtype=torch.long)
        return {
            'input': feat.unsqueeze(0),
            'label': label,
            'coords': torch.zeros(1, feat.shape[0], 2),
        }


def _collate_one(batch):
    assert len(batch) == 1
    return batch[0]


def make_loader(paths, labels, shuffle=False):
    ds = SlideDataset(paths, labels)
    return DataLoader(ds, batch_size=1, shuffle=shuffle,
                      num_workers=0, collate_fn=_collate_one)


# ── K-fold split ──────────────────────────────────────────────────────────────

def get_kfold_splits(labels, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(labels)), labels))


def get_repeated_stratified_splits(labels, n_splits=5, test_size=0.2, seed=42):
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=seed,
    )
    return list(splitter.split(np.zeros(len(labels)), labels))


def get_grouped_stratified_kfold_splits(
    sample_names,
    labels,
    n_folds=5,
    seed=42,
):
    patient_to_indices: Dict[str, List[int]] = {}
    patient_to_label: Dict[str, int] = {}
    for idx, (name, label) in enumerate(zip(sample_names, labels)):
        patient_id = get_tcga_patient_id(name)
        patient_to_indices.setdefault(patient_id, []).append(idx)
        if patient_id in patient_to_label and patient_to_label[patient_id] != label:
            raise RuntimeError(f'Patient {patient_id} has mixed labels')
        patient_to_label[patient_id] = label

    patient_ids = sorted(patient_to_indices)
    patient_labels = [patient_to_label[patient_id] for patient_id in patient_ids]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = []
    for train_pat_idx, val_pat_idx in skf.split(np.zeros(len(patient_ids)), patient_labels):
        train_ids = {patient_ids[i] for i in train_pat_idx}
        val_ids = {patient_ids[i] for i in val_pat_idx}

        train_indices = [idx for patient_id in train_ids for idx in patient_to_indices[patient_id]]
        val_indices = [idx for patient_id in val_ids for idx in patient_to_indices[patient_id]]
        splits.append((np.array(sorted(train_indices)), np.array(sorted(val_indices))))

    return splits


def get_grouped_repeated_stratified_splits(
    sample_names,
    labels,
    n_splits=5,
    test_size=0.2,
    seed=42,
):
    patient_to_indices: Dict[str, List[int]] = {}
    patient_to_label: Dict[str, int] = {}
    for idx, (name, label) in enumerate(zip(sample_names, labels)):
        patient_id = get_tcga_patient_id(name)
        patient_to_indices.setdefault(patient_id, []).append(idx)
        if patient_id in patient_to_label and patient_to_label[patient_id] != label:
            raise RuntimeError(f'Patient {patient_id} has mixed labels')
        patient_to_label[patient_id] = label

    patient_ids = sorted(patient_to_indices)
    patient_labels = [patient_to_label[patient_id] for patient_id in patient_ids]
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=seed,
    )

    splits = []
    for train_pat_idx, val_pat_idx in splitter.split(np.zeros(len(patient_ids)), patient_labels):
        train_ids = {patient_ids[i] for i in train_pat_idx}
        val_ids = {patient_ids[i] for i in val_pat_idx}
        train_indices = [idx for patient_id in train_ids for idx in patient_to_indices[patient_id]]
        val_indices = [idx for patient_id in val_ids for idx in patient_to_indices[patient_id]]
        splits.append((np.array(sorted(train_indices)), np.array(sorted(val_indices))))

    return splits


def _shared_split_path(
    dataset: str,
    mode: str,
    seed: int,
    n_splits: int,
    test_size: float,
    split_root: Optional[str] = None,
) -> Path:
    """Build the shared-split filename.

    Three filename schemas coexist in the public release:
      - ``{dataset}_{mode}_n{n_splits}.json`` — no seed / no test-size
        (currently only ``center_holdout``).
      - ``{dataset}_{mode}_seed{seed}_n{n_splits}.json`` — seeded but no
        test-size (k-fold modes: ``official_train_stratified_kfold``,
        ``all_stratified_kfold``).
      - ``{dataset}_{mode}_seed{seed}_n{n_splits}_test{tt}.json`` — full
        signature (the seeded stratified-shuffle protocols).
    """
    root = Path(split_root) if split_root is not None else SHARED_SPLIT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    if mode == "center_holdout":
        return root / f"{dataset}_{mode}_n{n_splits}.json"
    if mode in ("official_train_stratified_kfold", "all_stratified_kfold"):
        return root / f"{dataset}_{mode}_seed{seed}_n{n_splits}.json"
    test_tag = f"{int(round(test_size * 100)):02d}"
    return root / f"{dataset}_{mode}_seed{seed}_n{n_splits}_test{test_tag}.json"


def _shared_split_dir(split_path: Path) -> Path:
    return split_path.with_suffix('')


def _serialize_named_splits(
    sample_names,
    labels,
    splits,
) -> List[Dict[str, Any]]:
    serialized = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        serialized.append(
            {
                'fold': fold_idx,
                'train_names': [sample_names[i] for i in train_idx],
                'val_names': [sample_names[i] for i in val_idx],
                'train_labels': [int(labels[i]) for i in train_idx],
                'val_labels': [int(labels[i]) for i in val_idx],
            }
        )
    return serialized


def _load_named_splits(
    sample_names,
    split_records,
):
    name_to_idx = {name: idx for idx, name in enumerate(sample_names)}
    splits = []
    for record in split_records:
        train_idx = np.array([name_to_idx[name] for name in record['train_names']], dtype=int)
        val_idx = np.array([name_to_idx[name] for name in record['val_names']], dtype=int)
        splits.append((train_idx, val_idx))
    return splits


def _infer_split_groups(dataset: str, sample_names: List[str]) -> Tuple[Optional[str], Optional[List[str]]]:
    if dataset == 'nsclc':
        return 'patient_id', [get_tcga_patient_id(name) for name in sample_names]
    if dataset == 'cm17':
        groups = []
        for name in sample_names:
            tokens = Path(name).stem.split('_')
            if len(tokens) >= 2 and tokens[0] == 'patient':
                groups.append('_'.join(tokens[:2]))
            else:
                groups.append(Path(name).stem)
        return 'patient_id', groups
    return None, None


def _write_split_csv(
    csv_path: Path,
    indices,
    sample_names,
    labels,
    sample_paths: Optional[List[Path]] = None,
    class_names: Optional[List[str]] = None,
    group_field: Optional[str] = None,
    group_values: Optional[List[str]] = None,
) -> None:
    if csv_path.exists():
        return

    fieldnames = ['index', 'name', 'label']
    if class_names is not None:
        fieldnames.append('label_name')
    if group_field is not None:
        fieldnames.append(group_field)
    if sample_paths is not None:
        fieldnames.append('feature_path')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in indices:
            row = {
                'index': int(idx),
                'name': sample_names[idx],
                'label': int(labels[idx]),
            }
            if class_names is not None:
                row['label_name'] = class_names[int(labels[idx])]
            if group_field is not None and group_values is not None:
                row[group_field] = group_values[idx]
            if sample_paths is not None:
                row['feature_path'] = str(sample_paths[idx])
            writer.writerow(row)


def _materialize_shared_split_bundle(
    dataset: str,
    split_path: Path,
    mode: str,
    split_desc: str,
    sample_names,
    labels,
    splits,
    seed: int,
    n_splits: int,
    test_size: float,
    sample_paths: Optional[List[Path]] = None,
    class_names: Optional[List[str]] = None,
) -> Path:
    split_dir = _shared_split_dir(split_path)
    split_dir.mkdir(parents=True, exist_ok=True)

    group_field, group_values = _infer_split_groups(dataset, sample_names)
    metadata = {
        'dataset': dataset,
        'mode': mode,
        'seed': seed,
        'n_splits': n_splits,
        'test_size': test_size,
        'split_desc': split_desc,
        'split_json': str(split_path),
        'class_names': class_names,
        'group_field': group_field,
        'folds': [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_csv = split_dir / f'fold{fold_idx}_train.csv'
        test_csv = split_dir / f'fold{fold_idx}_test.csv'
        _write_split_csv(
            train_csv,
            train_idx,
            sample_names=sample_names,
            labels=labels,
            sample_paths=sample_paths,
            class_names=class_names,
            group_field=group_field,
            group_values=group_values,
        )
        _write_split_csv(
            test_csv,
            val_idx,
            sample_names=sample_names,
            labels=labels,
            sample_paths=sample_paths,
            class_names=class_names,
            group_field=group_field,
            group_values=group_values,
        )
        metadata['folds'].append(
            {
                'fold': fold_idx,
                'train_csv': str(train_csv),
                'test_csv': str(test_csv),
                'n_train': int(len(train_idx)),
                'n_test': int(len(val_idx)),
            }
        )

    metadata_path = split_dir / 'metadata.json'
    if not metadata_path.exists():
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    return split_dir


def get_or_create_shared_splits(
    dataset: str,
    sample_names,
    labels,
    n_splits: int = 5,
    test_size: float = 0.2,
    seed: int = 42,
    split_root: Optional[str] = None,
    sample_paths: Optional[List[Path]] = None,
    class_names: Optional[List[str]] = None,
):
    train_pct = int(round((1.0 - test_size) * 100))
    test_pct = int(round(test_size * 100))
    # Per-dataset default = the protocol used to produce Tab. 1:
    #   CM17  → leave-one-center-out 5-fold (`center_holdout`)
    #   CM16  → 5-fold stratified k-fold over the official train set
    #   NSCLC → patient-stratified 5-fold shuffle
    if dataset == 'nsclc':
        mode = 'patient_stratified_shuffle'
        split_desc = f"{n_splits}x patient-level stratified {train_pct}:{test_pct}"
    elif dataset == 'cm16':
        mode = 'official_train_stratified_kfold'
        split_desc = f"{n_splits}x stratified k-fold over CM16 official train"
    elif dataset == 'cm17':
        mode = 'center_holdout'
        split_desc = f"{n_splits}x center-holdout CV (test center=0..{n_splits - 1})"
    else:
        mode = 'stratified_shuffle'
        split_desc = f"{n_splits}x stratified {train_pct}:{test_pct}"

    split_path = _shared_split_path(
        dataset=dataset,
        mode=mode,
        seed=seed,
        n_splits=n_splits,
        test_size=test_size,
        split_root=split_root,
    )

    if split_path.exists():
        with open(split_path) as f:
            saved = json.load(f)
        splits = _load_named_splits(sample_names, saved['splits'])
        _materialize_shared_split_bundle(
            dataset=dataset,
            split_path=split_path,
            mode=saved.get('mode', mode),
            split_desc=saved.get('split_desc', split_desc),
            sample_names=sample_names,
            labels=labels,
            splits=splits,
            seed=saved.get('seed', seed),
            n_splits=saved.get('n_splits', n_splits),
            test_size=saved.get('test_size', test_size),
            sample_paths=sample_paths,
            class_names=class_names,
        )
        return splits, saved.get('split_desc', split_desc), split_path

    if dataset == 'nsclc':
        splits = get_grouped_repeated_stratified_splits(
            sample_names,
            labels,
            n_splits=n_splits,
            test_size=test_size,
            seed=seed,
        )
    else:
        splits = get_repeated_stratified_splits(
            labels,
            n_splits=n_splits,
            test_size=test_size,
            seed=seed,
        )

    payload = {
        'dataset': dataset,
        'mode': mode,
        'seed': seed,
        'n_splits': n_splits,
        'test_size': test_size,
        'split_desc': split_desc,
        'splits': _serialize_named_splits(sample_names, labels, splits),
    }
    with open(split_path, 'w') as f:
        json.dump(payload, f, indent=2)
    _materialize_shared_split_bundle(
        dataset=dataset,
        split_path=split_path,
        mode=mode,
        split_desc=split_desc,
        sample_names=sample_names,
        labels=labels,
        splits=splits,
        seed=seed,
        n_splits=n_splits,
        test_size=test_size,
        sample_paths=sample_paths,
        class_names=class_names,
    )
    return splits, split_desc, split_path


# ── MIL from scratch (one fold) ─────────────────────────────────────────────

def run_mil_one_fold(
    arch:         str,
    train_paths:  list, train_labels:  list,
    val_paths:    list, val_labels:    list,
    n_class:      int,
    feature_dim:  int,
    device:       str,
    train_epoch:  int   = 30,
    lr:           float = 1e-4,
    wd:           float = 1e-5,
    seed:         int   = 42,
    eval_interval: int  = 5,
    checkpoint_dir: Optional[str] = None,
    fold_tag:     str   = '',
) -> Dict[str, float]:
    """Train one MIL architecture from scratch, return best val metrics."""
    set_seed(seed)
    conf = make_conf(arch=arch, feature_dim=feature_dim, n_class=n_class,
                     train_epoch=train_epoch, lr=lr, wd=wd,
                     eval_interval=eval_interval)
    net = build_net(conf, torch.device(device))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, weight_decay=wd,
    )

    train_loader = make_loader(train_paths, train_labels, shuffle=True)
    val_loader   = make_loader(val_paths,   val_labels,   shuffle=False)

    best_score, best_metrics = -1.0, {}
    best_state = None

    for epoch in range(train_epoch):
        train_one_epoch(net, criterion, train_loader, optimizer,
                        torch.device(device), epoch, conf)
        if (epoch % eval_interval == 0) or (epoch == train_epoch - 1):
            t0 = time.time()
            (
                auroc,
                acc,
                f1,
                loss,
                precision,
                recall,
                detail_metrics,
                _,
                _,
            ) = mil_evaluate(
                net, criterion, val_loader, torch.device(device), conf,
                header=f'{arch} val', return_details=True)
            latency = time.time() - t0
            score = auroc + f1
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                best_metrics = {
                    'acc':       float(acc / 100.0 if acc > 1.0 else acc),
                    'precision': float(precision),
                    'recall':    float(recall),
                    'f1':        float(f1),
                    'auroc':     float(auroc),
                    'latency':   float(latency),
                }

    # ── save checkpoints ────────────────────────────────────────────────────
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tag = f'_{fold_tag}' if fold_tag else ''
        # best model
        if best_state is not None:
            best_path = ckpt_dir / f'{arch}{tag}_best.pt'
            torch.save({
                'arch': arch,
                'state_dict': best_state,
                'metrics': best_metrics,
                'n_class': n_class,
                'feature_dim': feature_dim,
                'seed': seed,
            }, best_path)
            print(f"    checkpoint saved → {best_path}")
        # last model
        last_path = ckpt_dir / f'{arch}{tag}_last.pt'
        torch.save({
            'arch': arch,
            'state_dict': {k: v.cpu() for k, v in net.state_dict().items()},
            'n_class': n_class,
            'feature_dim': feature_dim,
            'seed': seed,
        }, last_path)

    return best_metrics


# ── Aggregate folds ─────────────────────────────────────────────────────────

def _agg(fold_list: List[Dict[str, float]]) -> Dict[str, str]:
    """mean ± std over folds for each metric."""
    if not fold_list:
        return {}
    keys = [k for k in fold_list[0] if isinstance(fold_list[0][k], float)]
    out = {}
    for k in keys:
        vals = [f[k] for f in fold_list if k in f and not np.isnan(f[k])]
        if vals:
            out[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
    return out


# ── Comparison table ─────────────────────────────────────────────────────────

_TABLE_KEYS = ['acc', 'precision', 'recall', 'f1', 'auroc', 'latency']


def print_comparison(results: Dict[str, Dict[str, str]], dataset_name: str) -> None:
    methods = list(results.keys())
    col_w = max(22, max(len(m) for m in methods) + 2)
    bar = '=' * (18 + col_w * len(methods))

    print(f"\n{bar}")
    print(f"  {dataset_name} — {len(methods)} MIL methods comparison (CONCHv1.5)")
    print(f"{bar}")
    print(f"  {'Metric':<18}" + "".join(f"{m:>{col_w}}" for m in methods))
    print(f"  {'-' * (16 + col_w * len(methods))}")
    for key in _TABLE_KEYS:
        row = f"  {key:<18}"
        for m in methods:
            v = results[m].get(key, 'n/a')
            row += f"{str(v):>{col_w}}"
        print(row)
    print(f"{bar}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Eval-dataset MIL comparison (CONCHv1.5)')
    p.add_argument('--dataset', required=True, choices=['brca', 'cm16', 'cm17', 'panda', 'nsclc'],
                   help='Dataset to evaluate')
    p.add_argument('--feature_root', default=FEATURE_ROOT)
    p.add_argument('--feature_dim', type=int, default=FEATURE_DIM)
    p.add_argument('--cm16_split_mode', default='cv', choices=['cv', 'official'],
                   help='For cm16: cv=shared 5x8:2 on all slides, official=official train/test split')
    p.add_argument('--cm16_raw_root', default=CM16_RAW_ROOT,
                   help='Raw cm16 root containing lesion_annotations_{train,test}.zip')
    p.add_argument('--label_csv', default=None,
                   help='Label CSV for cm17 (columns: patient,label)')
    p.add_argument('--train_label_csv', default=None,
                   help='Train label CSV for fixed split (columns: patient,label)')
    p.add_argument('--test_label_csv', default=None,
                   help='Test label CSV for fixed split (columns: patient,label)')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--test_size', type=float, default=0.2,
                   help='Test ratio for saved repeated stratified splits (default: 0.2)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split_root', default=None,
                   help='Root directory for shared/materialized split files')

    # MIL training
    p.add_argument('--mil_archs', nargs='+', default=ALL_MIL_ARCHS,
                   choices=ALL_MIL_ARCHS)
    p.add_argument('--train_epoch', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--eval_interval', type=int, default=5)

    # misc
    p.add_argument('--device', default=None)
    p.add_argument('--output_dir', default=None,
                   help='Output directory (default: ./results/{dataset}_mil_comparison)')
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.output_dir is None:
        args.output_dir = f'./results/{args.dataset}_mil_comparison'
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── collect data ─────────────────────────────────────────────────────────
    fixed_split = (args.train_label_csv is not None and args.test_label_csv is not None)
    cm16_official_split = args.dataset == 'cm16' and args.cm16_split_mode == 'official'

    if cm16_official_split:
        tr_paths, tr_labels, tr_names, class_names = collect_cm16(
            args.feature_root, split='train', raw_root=args.cm16_raw_root)
        te_paths, te_labels, te_names, _ = collect_cm16(
            args.feature_root, split='test', raw_root=args.cm16_raw_root)
        n_class = len(set(tr_labels + te_labels))
        n_train, n_test = len(tr_paths), len(te_paths)

        print(f"\n{'='*60}")
        print(f"  Eval MIL Comparison — CONCHv1.5")
        print(f"  Dataset : {args.dataset.upper()} ({n_class}-class)")
        print(f"  Split   : official train/test (train={n_train}, test={n_test})")
        print(f"  Classes : {class_names}")
        print(f"  Methods : {[ARCH_DISPLAY[a] for a in args.mil_archs]}")
        print(f"  Device  : {device}")
        print(f"{'='*60}")

        ckpt_dir = str(out / 'checkpoints')
        arch_results: Dict[str, Dict[str, float]] = {}
        for arch in args.mil_archs:
            print(f"\n  [{ARCH_DISPLAY[arch]}]")
            m = run_mil_one_fold(
                arch=arch,
                train_paths=tr_paths, train_labels=tr_labels,
                val_paths=te_paths,   val_labels=te_labels,
                n_class=n_class, feature_dim=args.feature_dim, device=device,
                train_epoch=args.train_epoch, lr=args.lr, wd=args.wd,
                seed=args.seed,
                eval_interval=args.eval_interval,
                checkpoint_dir=ckpt_dir,
            )
            arch_results[arch] = m
            print(f"    acc={m.get('acc', 0):.4f}  prec={m.get('precision', 0):.4f}  "
                  f"rec={m.get('recall', 0):.4f}  f1={m.get('f1', 0):.4f}  "
                  f"auroc={m.get('auroc', float('nan')):.4f}  "
                  f"latency={m.get('latency', 0):.2f}s")

        agg = {}
        for arch in args.mil_archs:
            agg[ARCH_DISPLAY[arch]] = {
                k: f"{v:.4f}" for k, v in arch_results[arch].items()
                if isinstance(v, float)
            }
        print_comparison(agg, args.dataset.upper())

        summary = {
            'dataset':      args.dataset,
            'feature':      Path(args.feature_root).name,
            'feature_root': args.feature_root,
            'feature_dim':  args.feature_dim,
            'cm16_split_mode': 'official',
            'n_class':      n_class,
            'class_names':  class_names,
            'split':        'official_train_test',
            'n_train':      n_train,
            'n_test':       n_test,
            'mil_archs':    args.mil_archs,
            'train_epoch':  args.train_epoch,
            'lr':           args.lr,
            'results':      {arch: arch_results[arch] for arch in args.mil_archs},
        }
        summary_path = out / 'comparison_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"Results saved → {summary_path}")

    elif fixed_split:
        # Fixed train/test split mode
        assert args.dataset == 'cm17', "Fixed split only supported for cm17"
        tr_paths, tr_labels, tr_names, class_names = collect_cm17(
            args.feature_root, args.train_label_csv)
        te_paths, te_labels, te_names, _ = collect_cm17(
            args.feature_root, args.test_label_csv)
        n_class = len(set(tr_labels + te_labels))
        n_train, n_test = len(tr_paths), len(te_paths)

        print(f"\n{'='*60}")
        print(f"  Eval MIL Comparison — CONCHv1.5")
        print(f"  Dataset : {args.dataset.upper()} ({n_class}-class)")
        print(f"  Split   : fixed (train={n_train}, test={n_test})")
        print(f"  Classes : {class_names}")
        print(f"  Methods : {[ARCH_DISPLAY[a] for a in args.mil_archs]}")
        print(f"  Device  : {device}")
        print(f"{'='*60}")

        # ── single run per arch ─────────────────────────────────────────────
        ckpt_dir = str(out / 'checkpoints')
        arch_results: Dict[str, Dict[str, float]] = {}
        for arch in args.mil_archs:
            print(f"\n  [{ARCH_DISPLAY[arch]}]")
            m = run_mil_one_fold(
                arch=arch,
                train_paths=tr_paths, train_labels=tr_labels,
                val_paths=te_paths,   val_labels=te_labels,
                n_class=n_class, feature_dim=args.feature_dim, device=device,
                train_epoch=args.train_epoch, lr=args.lr, wd=args.wd,
                seed=args.seed,
                eval_interval=args.eval_interval,
                checkpoint_dir=ckpt_dir,
            )
            arch_results[arch] = m
            print(f"    acc={m.get('acc', 0):.4f}  prec={m.get('precision', 0):.4f}  "
                  f"rec={m.get('recall', 0):.4f}  f1={m.get('f1', 0):.4f}  "
                  f"auroc={m.get('auroc', float('nan')):.4f}  "
                  f"latency={m.get('latency', 0):.2f}s")

        # ── comparison table (no ± since single run) ────────────────────────
        agg = {}
        for arch in args.mil_archs:
            agg[ARCH_DISPLAY[arch]] = {
                k: f"{v:.4f}" for k, v in arch_results[arch].items()
                if isinstance(v, float)
            }
        print_comparison(agg, args.dataset.upper())

        # ── save ────────────────────────────────────────────────────────────
        summary = {
            'dataset':      args.dataset,
            'feature':      Path(args.feature_root).name,
            'feature_root': args.feature_root,
            'feature_dim':  args.feature_dim,
            'n_class':      n_class,
            'class_names':  class_names,
            'split':        'fixed',
            'n_train':      n_train,
            'n_test':       n_test,
            'mil_archs':    args.mil_archs,
            'train_epoch':  args.train_epoch,
            'lr':           args.lr,
            'results':      {arch: arch_results[arch] for arch in args.mil_archs},
        }
        summary_path = out / 'comparison_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"Results saved → {summary_path}")

    else:
        # ── k-fold CV mode ──────────────────────────────────────────────────
        if args.dataset == 'brca':
            paths, labels, names, class_names = collect_brca(args.feature_root)
        elif args.dataset == 'cm16':
            paths, labels, names, class_names = collect_cm16(args.feature_root, split='all', raw_root=args.cm16_raw_root)
        elif args.dataset == 'cm17':
            paths, labels, names, class_names = collect_cm17(args.feature_root, args.label_csv)
        elif args.dataset == 'panda':
            paths, labels, names, class_names = collect_panda(args.feature_root, args.label_csv)
        elif args.dataset == 'nsclc':
            paths, labels, names, class_names = collect_nsclc(args.feature_root)

        n_class = len(set(labels))
        n_total = len(paths)

        if args.dataset in {'brca', 'cm16', 'cm17', 'panda', 'nsclc'}:
            folds, split_desc, split_path = get_or_create_shared_splits(
                dataset=args.dataset,
                sample_names=names,
                labels=labels,
                n_splits=args.n_folds,
                test_size=args.test_size,
                seed=args.seed,
                split_root=args.split_root,
                sample_paths=paths,
                class_names=class_names,
            )
        else:
            split_desc = f"{args.n_folds}-fold CV"
            folds = get_kfold_splits(labels, n_folds=args.n_folds, seed=args.seed)
            split_path = None

        print(f"\n{'='*60}")
        print(f"  Eval MIL Comparison — CONCHv1.5")
        print(f"  Dataset : {args.dataset.upper()} ({n_class}-class, {n_total} samples)")
        print(f"  Classes : {class_names}")
        print(f"  Methods : {[ARCH_DISPLAY[a] for a in args.mil_archs]}")
        print(f"  Split   : {split_desc}  |  Device: {device}")
        print(f"{'='*60}")

        fold_results: Dict[str, List[Dict[str, float]]] = {
            arch: [] for arch in args.mil_archs
        }

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            if args.dataset == 'nsclc':
                tr_patients = len({get_tcga_patient_id(names[i]) for i in train_idx})
                va_patients = len({get_tcga_patient_id(names[i]) for i in val_idx})
                fold_header = (
                    f"  Fold {fold_idx + 1}/{args.n_folds}  "
                    f"(train={len(train_idx)} slides / {tr_patients} patients, "
                    f"val={len(val_idx)} slides / {va_patients} patients)"
                )
            else:
                fold_header = (
                    f"  Fold {fold_idx + 1}/{args.n_folds}  "
                    f"(train={len(train_idx)}, val={len(val_idx)})"
                )
            print(f"\n{'─'*60}")
            print(fold_header)
            print(f"{'─'*60}")

            tr_paths = [paths[i]  for i in train_idx]
            tr_lbl   = [labels[i] for i in train_idx]
            va_paths = [paths[i]  for i in val_idx]
            va_lbl   = [labels[i] for i in val_idx]

            for arch in args.mil_archs:
                print(f"\n  [{ARCH_DISPLAY[arch]}]")
                m = run_mil_one_fold(
                    arch=arch,
                    train_paths=tr_paths, train_labels=tr_lbl,
                    val_paths=va_paths,   val_labels=va_lbl,
                    n_class=n_class, feature_dim=args.feature_dim, device=device,
                    train_epoch=args.train_epoch, lr=args.lr, wd=args.wd,
                    seed=args.seed + fold_idx,
                    eval_interval=args.eval_interval,
                    checkpoint_dir=str(out / 'checkpoints'),
                    fold_tag=f'fold{fold_idx}',
                )
                fold_results[arch].append(m)
                print(f"    acc={m.get('acc', 0):.4f}  prec={m.get('precision', 0):.4f}  "
                      f"rec={m.get('recall', 0):.4f}  f1={m.get('f1', 0):.4f}  "
                      f"auroc={m.get('auroc', float('nan')):.4f}  "
                      f"latency={m.get('latency', 0):.2f}s")

        agg = {ARCH_DISPLAY[arch]: _agg(fold_results[arch]) for arch in args.mil_archs}
        print_comparison(agg, args.dataset.upper())

        summary = {
            'dataset':      args.dataset,
            'feature':      Path(args.feature_root).name,
            'feature_root': args.feature_root,
            'feature_dim':  args.feature_dim,
            'cm16_split_mode': args.cm16_split_mode if args.dataset == 'cm16' else None,
            'n_class':      n_class,
            'class_names':  class_names,
            'n_samples':    n_total,
            'n_folds':      args.n_folds,
            'test_size':    args.test_size if split_path is not None else None,
            'split':        split_desc,
            'split_file':   str(split_path) if split_path is not None else None,
            'split_dir':    str(_shared_split_dir(split_path)) if split_path is not None else None,
            'n_patients':   len({get_tcga_patient_id(name) for name in names}) if args.dataset == 'nsclc' else None,
            'mil_archs':    args.mil_archs,
            'train_epoch':  args.train_epoch,
            'lr':           args.lr,
            'aggregated':   agg,
            'fold_results': {arch: fold_results[arch] for arch in args.mil_archs},
        }
        summary_path = out / 'comparison_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"Results saved → {summary_path}")


if __name__ == '__main__':
    main()
