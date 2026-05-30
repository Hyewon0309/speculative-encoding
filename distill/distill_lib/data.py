"""Data pipelines for distillation training and CRC-100K evaluation."""

import concurrent.futures
import hashlib
import os
import queue
import random
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert any PIL image to RGB, correctly handling palette+transparency."""
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background.convert("RGB")
    return image.convert("RGB")


def _read_image_rgb(path: str | Path) -> torch.Tensor:
    """Decode an image with OpenCV and return a uint8 RGB tensor ``(3, H, W)``.

    Keeping the tensor in uint8 lets the downstream transform (v2) perform the
    expensive RandomResizedCrop interpolation on 1-byte data before upcasting
    to float and normalizing. This is materially faster than upcasting at
    decode time.
    """
    image = cv2.imread(os.fspath(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif channels == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif channels == 4:
            if image.dtype != np.uint8:
                if np.issubdtype(image.dtype, np.integer):
                    src_max = float(np.iinfo(image.dtype).max)
                else:
                    src_max = max(float(image.max()), 1.0)
                image = np.clip(
                    image.astype(np.float32) * (255.0 / src_max), 0.0, 255.0
                ).astype(np.uint8)
            color = image[:, :, :3].astype(np.float32)
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            bgr = color * alpha + 255.0 * (1.0 - alpha)
            bgr = np.clip(bgr, 0.0, 255.0).astype(np.uint8)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported channel count {channels} for image: {path}")
    else:
        raise ValueError(f"Unsupported image shape {image.shape} for image: {path}")

    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.integer):
            src_max = float(np.iinfo(rgb.dtype).max)
        else:
            src_max = max(float(rgb.max()), 1.0)
        rgb = np.clip(
            rgb.astype(np.float32) * (255.0 / src_max), 0.0, 255.0
        ).astype(np.uint8)

    rgb = np.ascontiguousarray(rgb)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


class ImageFolderDataset(Dataset):
    """Unlabeled image dataset from a directory of JPG/PNG files."""

    def __init__(self, data_dir: str, preprocess, max_samples: int | None = None,
                 seed: int = 42, paths: list[str] | None = None):
        self.preprocess = preprocess
        if paths is None:
            paths = scan_image_paths(data_dir)
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise FileNotFoundError(
                f"No image files found in {data_dir}. "
                f"Supported extensions: {_IMAGE_EXTENSIONS}"
            )
        if max_samples and max_samples < len(self.paths):
            rng = random.Random(seed)
            self.paths = rng.sample(self.paths, max_samples)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = _read_image_rgb(path)
        pixel_values = self._apply_preprocess(image)
        return {"pixel_values": pixel_values}

    def _apply_preprocess(self, image):
        try:
            processed = self.preprocess(image, return_tensors="pt")
        except TypeError:
            processed = self.preprocess(image)
        if isinstance(processed, dict):
            processed = processed.get("pixel_values", processed)
        if hasattr(processed, "pixel_values"):
            processed = processed.pixel_values
        if isinstance(processed, torch.Tensor) and processed.dim() == 4:
            processed = processed[0]
        return processed


def resolve_image_cache_path(data_dir: str, cache_dir: str) -> str:
    abs_data_dir = os.path.abspath(data_dir)
    digest = hashlib.sha1(abs_data_dir.encode("utf-8")).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"image_paths_{digest}.txt")


def _scan_image_paths_single_thread(data_dir: str) -> list[str]:
    paths = []
    dir_queue = [os.path.abspath(data_dir)]

    while dir_queue:
        current_dir = dir_queue.pop()
        try:
            with os.scandir(current_dir) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dir_queue.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            _, ext = os.path.splitext(entry.name)
                            if ext.lower() in _IMAGE_EXTENSIONS:
                                paths.append(os.path.abspath(entry.path))
                    except OSError:
                        continue
        except OSError:
            continue

    return sorted(paths)


def scan_image_paths(data_dir: str, num_workers: int = 1) -> list[str]:
    num_workers = max(int(num_workers or 1), 1)
    if num_workers == 1:
        return _scan_image_paths_single_thread(data_dir)

    root_dir = os.path.abspath(data_dir)
    paths = []
    paths_lock = threading.Lock()
    dir_queue = queue.Queue()
    dir_queue.put(root_dir)

    def worker():
        local_paths = []
        while True:
            current_dir = dir_queue.get()
            if current_dir is None:
                dir_queue.task_done()
                break
            try:
                with os.scandir(current_dir) as it:
                    for entry in it:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                dir_queue.put(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                _, ext = os.path.splitext(entry.name)
                                if ext.lower() in _IMAGE_EXTENSIONS:
                                    local_paths.append(os.path.abspath(entry.path))
                        except OSError:
                            continue
            except OSError:
                pass
            finally:
                dir_queue.task_done()

        if local_paths:
            with paths_lock:
                paths.extend(local_paths)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker) for _ in range(num_workers)]
        dir_queue.join()
        for _ in range(num_workers):
            dir_queue.put(None)
        dir_queue.join()
        for future in futures:
            future.result()

    return sorted(paths)


def save_image_paths_cache(paths: list[str], cache_path: str):
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for path in paths:
            f.write(path)
            f.write("\n")
    os.replace(tmp_path, cache_path)


def load_image_paths_cache(cache_path: str) -> list[str]:
    with open(cache_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _distill_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {"pixel_values": pixel_values}


class LabelEncoder:
    def __init__(self, mapping: dict | None = None):
        self.label_to_id = mapping or {}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    @property
    def fitted(self) -> bool:
        return len(self.label_to_id) > 0

    def fit(self, labels):
        unique = sorted(set(labels))
        self.label_to_id = {label: idx for idx, label in enumerate(unique)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

    def encode(self, label: str) -> int:
        return self.label_to_id[label]


def load_nctcrche100k(cache_dir, dataset_name="DykeF/NCTCRCHE100K",
                       max_train_samples=None, seed=42):
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    encoder = LabelEncoder()
    train_labels = list(train_ds["label"])
    val_labels = list(val_ds["label"])
    encoder.fit(train_labels + val_labels)

    if max_train_samples:
        train_ds = train_ds.shuffle(seed=seed).select(range(max_train_samples))

    return train_ds, val_ds, encoder


def build_eval_transform(preprocess, label_encoder):
    """Return a transform function for CRC-100K (labeled) datasets."""

    def _extract_pixel_values(processed):
        if processed is None:
            return processed
        if isinstance(processed, torch.Tensor):
            return processed
        if isinstance(processed, dict):
            return processed.get("pixel_values", processed)
        try:
            if "pixel_values" in processed:
                return processed["pixel_values"]
        except TypeError:
            pass
        if hasattr(processed, "pixel_values"):
            return processed.pixel_values
        return processed

    def _apply_single(image):
        try:
            processed = preprocess(image, return_tensors="pt")
        except TypeError:
            processed = preprocess(image)
        pixel_values = _extract_pixel_values(processed)
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 4:
            pixel_values = pixel_values[0]
        return pixel_values

    def transform(example):
        image = example["image"]
        if isinstance(image, list):
            images = [_to_rgb(img) if hasattr(img, "convert") else img for img in image]
            pixel_values = [_apply_single(img) for img in images]
            labels = [label_encoder.encode(label) for label in example["label"]]
            files = example.get("file", [""] * len(images))
            if not isinstance(files, list):
                files = [files] * len(images)
            return {"pixel_values": pixel_values, "label": labels, "file": files}

        image = _to_rgb(image)
        pixel_values = _apply_single(image)
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 4:
            pixel_values = pixel_values[0]
        return {
            "pixel_values": pixel_values,
            "label": label_encoder.encode(example["label"]),
            "file": example.get("file", ""),
        }

    return transform


def _eval_collate_fn(batch):
    """Collate for labeled evaluation batches (CRC-100K)."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    files = [item.get("file", "") for item in batch]
    return {"pixel_values": pixel_values, "labels": labels, "files": files}


def create_dataloader(dataset, batch_size, shuffle, sampler, num_workers,
                       pin_memory, persistent_workers, prefetch_factor, drop_last,
                       collate_fn=None):
    effective_shuffle = shuffle if sampler is None else False
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": effective_shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "drop_last": drop_last,
        "collate_fn": collate_fn or _distill_collate_fn,
    }
    if num_workers > 0 and prefetch_factor:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)
